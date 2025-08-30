import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import chamfer_distance

from pn_kit import farthest_point_sample_batch, index_points

# -------------------------
# Utility conv blocks
# -------------------------
def conv2d_bn_relu_v1(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

def conv1d_bn_relu_v1(in_c, out_c):
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=1, bias=False),
        nn.BatchNorm1d(out_c),
        nn.ReLU(inplace=True)
    )

# -------------------------
# PointNet++ SA blocks
# -------------------------
class PointNetSetAbstraction_v1(nn.Module):
    def __init__(self, npoint, K, in_channel, mlp, bn=True):
        super().__init__()
        self.npoint = npoint
        self.K = K
        self.in_channel = in_channel
        self.mlp = mlp
        self.bn = bn

        last = in_channel + 3
        layers = []
        for out in mlp:
            layers.append(conv2d_bn_relu(last, out) if bn else nn.Sequential(nn.Conv2d(last, out, 1), nn.ReLU()))
            last = out
        self.mlp_stack = nn.ModuleList(layers)

    def forward(self, xyz, points=None):
        B, N, _ = xyz.shape
        S = self.npoint

        if S == N:
            new_xyz = xyz
        else:
            idx = farthest_point_sample_batch(xyz, S)
            new_xyz = index_points(xyz, idx)

        dists, idx_knn, grouped_xyz = knn_points(new_xyz.float(), xyz.float(), K=self.K, return_nn=True)
        grouped_xyz = grouped_xyz - new_xyz.view(B, S, 1, 3)

        if points is not None:
            points_trans = points.permute(0, 2, 1)
            neigh_features = index_points(points_trans, idx_knn)
            grouped = torch.cat([grouped_xyz, neigh_features], dim=-1)
        else:
            grouped = grouped_xyz

        grouped = grouped.permute(0, 3, 2, 1).contiguous()

        for layer in self.mlp_stack:
            grouped = layer(grouped)
        new_points = torch.max(grouped, dim=2)[0]
        return new_xyz, new_points


class PointNetSetAbstractionMSG_v1(nn.Module):
    def __init__(self, npoint, scales, in_channel, bn=True):
        super().__init__()
        self.npoint = npoint
        self.branches = nn.ModuleList()
        for scale in scales:
            K = scale['K']
            mlp = scale['mlp']
            branch = PointNetSetAbstraction(npoint=npoint, K=K, in_channel=in_channel, mlp=mlp, bn=bn)
            self.branches.append(branch)

    def forward(self, xyz, points=None):
        outputs = []
        new_xyz = None
        for branch in self.branches:
            nxyz, npoints = branch(xyz, points)
            new_xyz = nxyz
            outputs.append(npoints)
        new_points = torch.cat(outputs, dim=1)
        return new_xyz, new_points


class PointNet2EncoderFull_v1(nn.Module):
    def __init__(self, sa_blocks=None, latent_dim=256, bn=True):
        super().__init__()
        if sa_blocks is None:
            sa_blocks = [
                {'type': 'MSG', 'npoint': 512, 'scales':[{'K':16,'mlp':[32,32,64]}, {'K':32,'mlp':[64,64,128]}], 'in_channel':0},
                {'type': 'SS',  'npoint': 128, 'K':32, 'mlp':[128,128,256], 'in_channel':64+128},
                {'type': 'SS',  'npoint': 32,  'K':32, 'mlp':[256,256,512], 'in_channel':256}
            ]
        modules = []
        for block in sa_blocks:
            if block['type'] == 'MSG':
                modules.append(PointNetSetAbstractionMSG(block['npoint'], block['scales'], block.get('in_channel', 0), bn=bn))
            else:
                modules.append(PointNetSetAbstraction(block['npoint'], block['K'], block.get('in_channel', 0), block['mlp'], bn=bn))
        self.sa_modules = nn.ModuleList(modules)

        last_block = sa_blocks[-1]
        if last_block['type'] == 'MSG':
            out_c = sum([s['mlp'][-1] for s in last_block['scales']])
        else:
            out_c = last_block['mlp'][-1]
        self.global_conv = nn.Sequential(
            nn.Conv1d(out_c, out_c, 1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, latent_dim, 1)
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        B, N, _ = x.shape
        xyz = x
        points = None

        for sa in self.sa_modules:
            xyz, points = sa(xyz, points)

        global_feat = torch.max(points, dim=2)[0]
        global_feat = global_feat.unsqueeze(-1)
        latent = self.global_conv(global_feat).squeeze(-1)
        return latent


# -------------------------
# Simpler PointNet2 encoder (per-point features + global)
# -------------------------
class PointNet2Encoder(nn.Module):
    def __init__(self, latent_dim=256, npoints=8192):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x):  # x: (B, N, 3)
        # per-point MLPs - keep per-point features and also a global pooled feature
        feat = self.mlp1(x)           # (B, N, 64)
        feat = self.mlp2(feat)        # (B, N, 128)
        feat_global = torch.max(feat, dim=1)[0]  # (B, 128)
        latent = self.fc(feat_global)  # (B, latent_dim)
        return latent, feat_global


# -------------------------
# Decoder (PCN-like)
# -------------------------
class PCNDecoder(nn.Module):
    def __init__(self, latent_dim=256, coarse_points=1024, final_npoints=8192):
        super().__init__()
        self.fc_coarse = nn.Sequential(
            nn.Linear(latent_dim, coarse_points), nn.ReLU(),
            nn.Linear(coarse_points, coarse_points * 3)
        )
        self.expansion = nn.Sequential(
            nn.Linear(coarse_points*3, final_npoints*3)
        )
        self.coarse_points = coarse_points
        self.final_npoints = final_npoints

    def forward(self, latent):
        B = latent.size(0)
        coarse = self.fc_coarse(latent).view(B, self.coarse_points, 3)
        fine = self.expansion(coarse.view(B, -1)).view(B, self.final_npoints, 3)
        return coarse, fine


# -------------------------
# ConditionalProbabilityModel (reworked)
# -------------------------
class ConditionalProbabilityModel_v1(nn.Module):
    """
    Condition per-latent distributions on a global PointNet2 feature.
    pmf is over L discrete bins per latent dimension.
    """
    def __init__(self, latent_dim=256, L=7, hidden=256):
        super().__init__()
        # encoder used to extract global conditioning features from input point cloud
        self.encoder = PointNet2Encoder(latent_dim=latent_dim)
        self.L = L
        self.latent_dim = latent_dim

        # fuse pre-quant latent (y_cont) with global feature
        self.fuse = nn.Sequential(
            nn.Linear(latent_dim + 128, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        self.mean_head  = nn.Linear(hidden, latent_dim)
        self.scale_head = nn.Linear(hidden, latent_dim)
        self.pmf_head   = nn.Linear(hidden, latent_dim * L)

    def forward(self, x_points, y_prequant):
        """
        x_points   : (B, N, 3) input point cloud used for conditioning
        y_prequant : (B, D) continuous latent (before quantization)
        Returns:
            mean:  (B, D)
            scale: (B, D) > 0
            pmf:   (B, D, L)
        """
        # get global conditioning feature from input point cloud
        _, feat_global = self.encoder(x_points)   # (B, 128)
        fused = torch.cat([y_prequant, feat_global], dim=-1)  # (B, D+128)
        h = self.fuse(fused)  # (B, hidden)

        mean  = self.mean_head(h)                 # (B, D)
        scale = F.softplus(self.scale_head(h)) + 1e-6

        pmf_logits = self.pmf_head(h).view(-1, self.latent_dim, self.L)  # (B, D, L)
        # numeric safety: add small eps before softmax, then normalize
        pmf = F.softmax(pmf_logits, dim=-1)
        pmf = pmf.clamp(min=1e-12)
        pmf = pmf / pmf.sum(dim=-1, keepdim=True)

        return mean, scale, pmf


# -------------------------
# RateDistortionLoss (kept hybrid)
# -------------------------
class RateDistortionLoss_v1(nn.Module):
    def __init__(self, loss_type="hybrid", alpha=0.7, eps=1e-9, max_rate=100.0):
        super(RateDistortionLoss_v1, self).__init__()
        self.loss_type = loss_type.lower()
        self.alpha = alpha
        self.eps = eps
        self.max_rate = max_rate

def forward(self, pc_pred, pc_target, fbpp, λ_dist=1000.0, λ_rate=1.0):
    if self.loss_type == "hybrid":
        chamfer, _ = chamfer_distance(pc_pred, pc_target, batch_reduction="mean")
        chamfer = chamfer / pc_pred.shape[1]
        l2 = torch.mean((pc_pred - pc_target) ** 2)
        dist = self.alpha * chamfer + (1 - self.alpha) * l2
    elif self.loss_type == "chamfer":
        dist, _ = chamfer_distance(pc_pred, pc_target, batch_reduction="mean")
        dist = dist / pc_pred.shape[1]
    else:
        dist = torch.mean((pc_pred - pc_target) ** 2)

    rate = torch.clamp(fbpp, min=0.0).mean()

    loss = λ_dist * dist + λ_rate * rate
    return loss, dist.detach(), rate.detach()


class get_loss(nn.Module):
    def __init__(self, loss_type="chamfer"):
        super(get_loss, self).__init__()
        self.criterion = RateDistortionLoss(loss_type=loss_type)

    def forward(self, pc_pred, pc_target, fbpp, λ=1.0):
        return self.criterion(pc_pred, pc_target, fbpp, λ=λ)


# -------------------------
# PointCloudAE with STE quantizer
# -------------------------
class PointCloudAE_v1(nn.Module):
    def __init__(self, latent_dim=256, L=7, npoints=8192):
        super().__init__()
        self.encoder = PointNet2Encoder(latent_dim=latent_dim, npoints=npoints)
        self.decoder = PCNDecoder(latent_dim=latent_dim, final_npoints=npoints)
        self.L = L
        self.latent_dim = latent_dim

    def _quantize_symbols(self, y_cont):
        """
        Map continuous y_cont (B, D) -> discrete symbols sym (B, D) in [0, L-1].
        Use a straight-through estimator so gradients flow to y_cont.
        Returns:
            sym: LongTensor (B, D)
            y_hat01: float proxy (B, D) in [0,1] used for decoder input (still differentiable)
        """
        # squash continuous latent to [0,1]
        y01 = torch.sigmoid(y_cont)  # (B, D)
        # scale to [0, L-1]
        y_scaled = y01 * (self.L - 1)
        # rounding -> integer symbols
        sym = torch.round(y_scaled).long()
        sym = sym.clamp(0, self.L - 1)

        # Straight-through: create a differentiable proxy y_hat that equals sym in forward
        # but uses y_scaled gradients in backward.
        y_hat_scaled = y_scaled + (sym.float() - y_scaled).detach()  # (B, D)
        y_hat01 = y_hat_scaled / (self.L - 1)  # in [0,1], differentiable w.r.t. y_cont
        return sym, y_hat01

    def forward(self, x):
        """
        x: (B, N, 3)
        Returns:
            coarse (B, Cc, 3), fine (B, N, 3), feats (B, 128), sym (B, D), y_cont (B, D)
        """
        y_cont, feats = self.encoder(x)          # y_cont: (B, D)
        sym, y_hat01 = self._quantize_symbols(y_cont)  # sym: (B,D) long, y_hat01: (B,D) float
        coarse, fine = self.decoder(y_hat01)  # decoder input is float proxy
        return coarse, fine, feats, sym, y_cont


# -------------------------
# Bits estimation helper (matches pmf output)
# -------------------------
def estimate_bits_per_point_conditional_v1(pmf, sym):
    """
    Compute average bits from pmf and integer symbols.
    Args:
        pmf : Tensor (B, D, L) probabilities over L bins for each latent channel
        sym : LongTensor (B, D) integer symbols in [0, L-1]
    Returns:
        fbpp: scalar tensor (average bits per latent-channel across batch)
    """
    # validate shapes
    assert pmf.dim() == 3 and sym.dim() == 2, "pmf must be (B,D,L), sym (B,D)"
    B, D, L = pmf.shape

    # clamp small probabilities
    pmf = pmf.clamp(min=1e-12)

    # gather probabilities for actual symbols
    idx = sym.unsqueeze(-1)  # (B, D, 1)
    p_sym = torch.gather(pmf, dim=2, index=idx).squeeze(-1)  # (B, D)

    # avoid log(0)
    p_sym = p_sym.clamp(min=1e-12)

    # bits per latent channel
    bits = -torch.log2(p_sym)  # (B, D)

    # average across batch and channels -> scalar fbpp
    fbpp = bits.mean()

    return fbpp

# Enhanced encoder / decoder for better 3D pattern learning
# ----- small building blocks -----
class SEBlock(nn.Module):
    """Squeeze-Excite for channel gating"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):  # x: (B, C)
        s = x.mean(dim=1) if x.dim() == 3 else x  # adapt if pass global features (B,C) or (B,C,1)
        a = F.relu(self.fc1(s))
        a = torch.sigmoid(self.fc2(a))
        return x * a.unsqueeze(-1) if x.dim() == 3 else x * a

class ResidualConv1D(nn.Module):
    """1D residual conv block applied to (B, C, N) features"""
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, 1, bias=False),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, 1, bias=False),
            nn.BatchNorm1d(ch_out),
        )
        if ch_in != ch_out:
            self.skip = nn.Sequential(
                nn.Conv1d(ch_in, ch_out, 1, bias=False),
                nn.BatchNorm1d(ch_out)
            )
        else:
            self.skip = None
    def forward(self, x):
        out = self.net(x)
        skip = self.skip(x) if self.skip is not None else x
        return F.relu(out + skip)

# ----- Single-scale SA block (optimized for memory) -----
class SA_Single(nn.Module):
    """
    Single-scale set abstraction using knn_points grouping + shared MLPs.
    Input xyz: (B,N,3), points (B, C, N) or None
    Returns new_xyz (B,S,3), new_points (B, C_out, S)
    """
    def __init__(self, npoint, K, in_channel, mlp_out_channels):
        super().__init__()
        self.npoint = npoint
        self.K = K
        self.mlp = nn.ModuleList()
        last = in_channel + 3
        for out_ch in mlp_out_channels:
            self.mlp.append(nn.Sequential(
                nn.Conv2d(last, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
            last = out_ch
    def forward(self, xyz, points=None):
        B, N, _ = xyz.shape
        S = self.npoint
        if S == N:
            new_xyz = xyz
            idx = None
        else:
            idx = farthest_point_sample_batch(xyz, S)  # [B, S]
            new_xyz = index_points(xyz, idx)           # [B, S, 3]
        # KNN grouping
        dists, idx_knn, grouped = knn_points(new_xyz.float(), xyz.float(), K=self.K, return_nn=True)
        # grouped: (B, S, K, 3)
        grouped = grouped - new_xyz.view(B, S, 1, 3)
        if points is not None:
            # points: (B, C, N) -> (B, N, C)
            neigh_feats = index_points(points.permute(0,2,1), idx_knn)  # (B,S,K,C)
            grouped = torch.cat([grouped, neigh_feats], dim=-1)  # (B,S,K,3+C)
        # to (B, C_in, K, S)
        grouped = grouped.permute(0, 3, 2, 1).contiguous()
        for m in self.mlp:
            grouped = m(grouped)
        # pool over K -> (B, C_out, S)
        new_points = torch.max(grouped, dim=2)[0]
        return new_xyz, new_points

# ----- Enhanced PointNet2 encoder (multi-scale) -----
class EnhancedPointNet2Encoder(nn.Module):
    def __init__(self, latent_dim=256, npoints=8192):
        super().__init__()
        self.npoints = npoints
        # Multi-scale first layer (MSG-like)
        self.sa1 = SA_Single(npoint=npoints//8, K=16, in_channel=0, mlp_out_channels=[32, 32, 64])
        # second SA (single-scale)
        self.sa2 = SA_Single(npoint=npoints//32, K=32, in_channel=64, mlp_out_channels=[128,128,256])
        # small residual convs on per-point features from early stage to feed decoder skips
        self.res1 = ResidualConv1D(64, 64)
        self.res2 = ResidualConv1D(256, 256)
        # global conv to latent
        self.global_conv = nn.Sequential(
            nn.Conv1d(256, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, latent_dim, 1)
        )
        self.se = SEBlock(latent_dim, reduction=8)

    def forward(self, x):
        """
        x: (B, N, 3)
        returns:
            y_cont: (B, latent_dim)  -- continuous latent before quant
            skip_feats: dict of per-scale features for decoder { 's1': (B,64,S1), 's2': (B,256,S2) }
            per_point_global: (B, 128) small global per-point descriptor (useful for prob)
        """
        B, N, _ = x.shape
        # SA1
        xyz1, p1 = self.sa1(x, points=None)   # p1: (B,64,S1)
        # keep per-point style features for skip
        s1 = p1  # (B, 64, S1)
        # SA2 (use s1 as points -> need to map to shape points expected)
        # Convert s1 to points format for SA2: SA expects points param (B, C, N) relative to xyz positions -> here s1 aligns with xyz1
        xyz2, p2 = self.sa2(xyz1, points=s1)  # p2: (B,256,S2)
        s2 = p2

        # global pooling from p2
        g = torch.max(p2, dim=2)[0]  # (B,256)
        # conv to latent
        g_in = g.unsqueeze(-1)  # (B,256,1)
        y_cont = self.global_conv(g_in).squeeze(-1)  # (B, latent_dim)
        y_cont = self.se(y_cont)  # gated latent

        # small per-point global descriptor (pool earlier per-point MLP)
        # produce a 128-d per-sample feature for use in ConditionalProbabilityModel
        per_point_global = F.adaptive_max_pool1d(p2, 1).squeeze(-1)  # (B,256) -> reduce to 128 via linear if desired
        per_point_global_small = per_point_global[:, :128] if per_point_global.shape[1] >= 128 else per_point_global

        skips = {'s1': s1, 's2': s2}
        return y_cont, skips, per_point_global_small

# ----- Enhanced PCN Decoder (coarse->fine with skips) -----
class EnhancedPCNDecoder(nn.Module):
    def __init__(self, latent_dim=256, coarse_points=1024, final_npoints=8192):
        super().__init__()
        self.coarse_points = coarse_points
        self.final_npoints = final_npoints
        self.fc_coarse = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(),
            nn.Linear(1024, coarse_points * 3)
        )
        # refine network that consumes coarse + upsampled skip features
        self.refine = nn.Sequential(
            nn.Linear(coarse_points * 3 + latent_dim + 64 + 256, 2048),
            nn.ReLU(),
            nn.Linear(2048, final_npoints * 3)
        )

    def forward(self, y_hat01, skips):
        """
        y_hat01: (B, D) or (B, latent_dim) float proxy
        skips: dict with 's1' and 's2' features (B, C, S)
        returns: coarse (B, Cc,3), fine (B, N, 3)
        """
        B = y_hat01.size(0)
        coarse = self.fc_coarse(y_hat01).view(B, self.coarse_points, 3)
        # flatten coarse + latent + simple pooled skips to feed refine
        s1 = skips['s1']  # (B,64,S1)
        s2 = skips['s2']  # (B,256,S2)
        s1_pool = torch.max(s1, dim=2)[0]  # (B,64)
        s2_pool = torch.max(s2, dim=2)[0]  # (B,256)
        x = torch.cat([coarse.view(B, -1), y_hat01, s1_pool, s2_pool], dim=1)
        fine = self.refine(x).view(B, self.final_npoints, 3)
        return coarse, fine

# ----- Example Combined AE -----
class EnhancedPointCloudAE(nn.Module):
    def __init__(self, latent_dim=256, L=7, npoints=8192):
        super().__init__()
        self.encoder = EnhancedPointNet2Encoder(latent_dim=latent_dim, npoints=npoints)
        self.decoder = EnhancedPCNDecoder(latent_dim=latent_dim, coarse_points=1024, final_npoints=npoints)
        self.L = L
        self.latent_dim = latent_dim

    def _quantize_symbols(self, y_cont):
        # same STE quantizer as earlier: sigmoid -> scale -> round -> clamp
        y01 = torch.sigmoid(y_cont)
        y_scaled = y01 * (self.L - 1)
        sym = torch.round(y_scaled).long().clamp(0, self.L - 1)
        y_hat_scaled = y_scaled + (sym.float() - y_scaled).detach()
        y_hat01 = y_hat_scaled / (self.L - 1)
        return sym, y_hat01

    def forward(self, x):
        # x: (B, N, 3)
        y_cont, skips, per_point_global = self.encoder(x)  # (B,D), skips dict
        sym, y_hat01 = self._quantize_symbols(y_cont)
        coarse, fine = self.decoder(y_hat01, skips)
        return coarse, fine, per_point_global, sym, y_cont

# file: pppe_pcd_ae_improved.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import chamfer_distance

from pn_kit import farthest_point_sample_batch, index_points, pmf_to_cdf, estimate_bits_from_pmf

# -------------------------
# Small conv helper blocks
# -------------------------
def conv2d_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

def conv1d_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=1, bias=False),
        nn.BatchNorm1d(out_c),
        nn.ReLU(inplace=True)
    )

# -------------------------
# PointNet++ SA blocks (single-scale and MSG)
# -------------------------
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, K, in_channel, mlp, bn=True):
        super().__init__()
        self.npoint = npoint
        self.K = K
        last = in_channel + 3
        layers = []
        for out in mlp:
            layers.append(conv2d_bn_relu(last, out) if bn else nn.Sequential(nn.Conv2d(last, out, 1), nn.ReLU()))
            last = out
        self.mlp_stack = nn.ModuleList(layers)

    def forward(self, xyz, points=None):
        """
        xyz: (B, N, 3)
        points: (B, C, N) or None
        returns: new_xyz (B, S, 3), new_points (B, C_out, S)
        """
        B, N, _ = xyz.shape
        S = self.npoint
        if S == N:
            new_xyz = xyz
        else:
            idx = farthest_point_sample_batch(xyz, S)   # expects (B,N,3) -> indices (B,S)
            new_xyz = index_points(xyz, idx)            # (B,S,3)

        dists, idx_knn, grouped_xyz = knn_points(new_xyz.float(), xyz.float(), K=self.K, return_nn=True)
        grouped_xyz = grouped_xyz - new_xyz.view(B, S, 1, 3)

        if points is not None:
            # points: (B, C, N) -> -> (B, N, C) for indexing
            pts_t = points.permute(0, 2, 1)
            neigh = index_points(pts_t, idx_knn)  # (B,S,K,C)
            grouped = torch.cat([grouped_xyz, neigh], dim=-1)  # (B,S,K,3+C)
        else:
            grouped = grouped_xyz  # (B,S,K,3)

        grouped = grouped.permute(0, 3, 2, 1).contiguous()  # -> (B, C_in, K, S)
        for layer in self.mlp_stack:
            grouped = layer(grouped)
        new_points = torch.max(grouped, dim=2)[0]  # (B, C_out, S)
        return new_xyz, new_points


class PointNetSetAbstractionMSG(nn.Module):
    def __init__(self, npoint, scales, in_channel, bn=True):
        super().__init__()
        self.branches = nn.ModuleList()
        for sc in scales:
            self.branches.append(PointNetSetAbstraction(npoint, sc['K'], in_channel, sc['mlp'], bn=bn))

    def forward(self, xyz, points=None):
        outs = []
        new_xyz = None
        for b in self.branches:
            nxyz, npts = b(xyz, points)
            new_xyz = nxyz
            outs.append(npts)
        new_points = torch.cat(outs, dim=1)  # concat channel dim
        return new_xyz, new_points

# -------------------------
# PointNet++ encoder (stackable)
# -------------------------
class PointNet2EncoderFull(nn.Module):
    def __init__(self, sa_blocks=None, latent_dim=256, bn=True):
        super().__init__()
        if sa_blocks is None:
            sa_blocks = [
                {'type': 'MSG', 'npoint': 512, 'scales':[{'K':16,'mlp':[32,32,64]}, {'K':32,'mlp':[64,64,128]}], 'in_channel':0},
                {'type': 'SS',  'npoint': 128, 'K':32, 'mlp':[128,128,256], 'in_channel':64+128},
                {'type': 'SS',  'npoint': 32,  'K':32, 'mlp':[256,256,512], 'in_channel':256}
            ]
        modules = []
        for block in sa_blocks:
            if block['type'] == 'MSG':
                modules.append(PointNetSetAbstractionMSG(block['npoint'], block['scales'], block.get('in_channel', 0), bn=bn))
            else:
                modules.append(PointNetSetAbstraction(block['npoint'], block['K'], block.get('in_channel', 0), block['mlp'], bn=bn))
        self.sa_modules = nn.ModuleList(modules)

        # get output channel of last SA
        last_block = sa_blocks[-1]
        if last_block['type'] == 'MSG':
            out_c = sum([s['mlp'][-1] for s in last_block['scales']])
        else:
            out_c = last_block['mlp'][-1]

        self.global_conv = nn.Sequential(
            nn.Conv1d(out_c, out_c, 1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, latent_dim, 1)
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        """
        x: (B, N, 3)
        Returns:
            latent (B, latent_dim)
            per_point_feature (B, F) -- global pooled features for conditioning
        """
        xyz = x
        points = None
        for sa in self.sa_modules:
            xyz, points = sa(xyz, points)  # new xyz (B,S,3), points (B,C,S)

        # points -> global pooling
        global_feat = torch.max(points, dim=2)[0]  # (B, C_out)
        gf = global_feat.unsqueeze(-1)
        latent = self.global_conv(gf).squeeze(-1)  # (B, latent_dim)
        # we'll also return pooled features to be used by conditional model (useful)
        return latent, global_feat  # global_feat (B, C_out)

# -------------------------
# PCN progressive decoder (lighter)
# -------------------------
class PCNDecoderSmall(nn.Module):
    def __init__(self, latent_dim=256, coarse_points=512, final_points=8192):
        super().__init__()
        # Map latent to coarse point cloud
        self.fc_coarse = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, coarse_points * 3)
        )
        # Expand coarse -> fine via small MLP and folding-like expansion
        self.expansion_mlp = nn.Sequential(
            nn.Linear(coarse_points * 3 + latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, final_points * 3)
        )
        self.coarse_points = coarse_points
        self.final_points = final_points

    def forward(self, latent):
        B = latent.size(0)
        coarse = self.fc_coarse(latent).view(B, self.coarse_points, 3)
        expand_input = torch.cat([coarse.view(B, -1), latent], dim=1)
        fine = self.expansion_mlp(expand_input).view(B, self.final_points, 3)
        return coarse, fine

# -------------------------
# Straight-through quantizer (keeps gradients)
# -------------------------
def quantize_st(x, min_val, max_val, levels):
    """
    Straight-through quantization to integer bins in [0, levels-1].
    x: real-valued (B, C) or (B, C, N) before quant mapping (expected scaled to some range)
    We map to integer bin indices in [0, levels-1] with STE.
    Returns integer-like tensor with gradient via STE (float dtype).
    """
    # scale input into [0, levels-1] range (caller must ensure x in appropriate range)
    x_clamped = torch.clamp(x, min_val, max_val)
    # map to bin centers scale
    scaled = (x_clamped - min_val) / (max_val - min_val + 1e-9) * (levels - 1)
    rounded = torch.round(scaled)
    # straight-through: use rounded in forward but preserve gradient of scaled
    y = rounded.detach() + (scaled - scaled.detach())
    # clipped to valid integer range
    y = torch.clamp(y, 0, levels - 1)
    return y  # still float but integer-valued in forward

# -------------------------
# Conditional Probability Model
# -------------------------
class ConditionalProbabilityModel(nn.Module):
    """
    Produces mean, scale and PMF logits for latent quantized bins.
    Input:
        latent_features: (B, d, N) or (B, d) depending on your latents
        cond_feats: per-point/global features from encoder (B, F) or (B, F, N)
    Output:
        mean (B, d, N) -- predicted mean for likelihood (optional use)
        scale (B, d, N) -- predicted scale (positive)
        pmf_logits (B, K, N) -- logits across K discrete bins per point (or per-feature)
    """
    def __init__(self, feature_dim=512, hidden_channels=128, latent_bins=16, latent_channels=3):
        super().__init__()
        self.latent_bins = latent_bins
        self.latent_channels = latent_channels

        # cond project (global features -> per-point if needed)
        self.cond_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # combine latent + cond (operates per point)
        self.combine = nn.Sequential(
            nn.Conv1d(latent_channels + hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 1)
        )

        self.mean_head = nn.Conv1d(hidden_channels, latent_channels, 1)
        self.scale_head = nn.Conv1d(hidden_channels, latent_channels, 1)
        self.pmf_head = nn.Conv1d(hidden_channels, latent_bins, 1)

    def forward(self, y, cond_feats):
        """
        y: (B, d, N)  -- latent codes per point (or treat d as channels)
        cond_feats: (B, F) or (B, F, N)
        Returns:
            mean (B, d, N), scale (B, d, N), pmf (B, K, N)
        """
        B, d, N = y.shape

        # Cond_feats can be global (B, F) or per-point (B, F, N)
        if cond_feats.ndim == 2:
            cond = self.cond_proj(cond_feats)        # (B, H)
            cond = cond.unsqueeze(-1).repeat(1, 1, N)  # (B, H, N)
        elif cond_feats.ndim == 3:
            # assume (B, F, N)
            cond = cond_feats
        else:
            raise ValueError("cond_feats must be (B,F) or (B,F,N)")

        # concat along channel axis
        x = torch.cat([y, cond], dim=1)  # (B, d+H, N)
        h = self.combine(x)              # (B, H, N)

        mean = self.mean_head(h)         # (B, d, N)
        scale = F.softplus(self.scale_head(h)) + 1e-6
        pmf_logits = self.pmf_head(h)    # (B, K, N)
        pmf = F.softmax(pmf_logits, dim=1).clamp(min=1e-9)  # (B, K, N) stable
        return mean, scale, pmf

# -------------------------
# Rate-distortion loss (hybrid + scaling)
# -------------------------
class RateDistortionLoss(nn.Module):
    def __init__(self, loss_type="hybrid", alpha=0.7, max_rate=100.0):
        """
        dist_scale: multiply distortion term by this to balance scale vs rate.
        Use dist_scale >> 1 when chamfer is very small compared to rate.
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.alpha = alpha
        self.max_rate = max_rate

    def forward(self, pc_recon, pc_target, fbpp, λ=1.0):
        # Distortion
        if self.loss_type == "chamfer":
            dist, _ = chamfer_distance(pc_recon, pc_target, batch_reduction="mean")
            dist = dist
        elif self.loss_type == "l1":
            dist = F.smooth_l1_loss(pc_recon, pc_target, reduction="mean")
        else:
            chamfer, _ = chamfer_distance(pc_recon, pc_target, batch_reduction="mean")
            l1 = F.smooth_l1_loss(pc_recon, pc_target, reduction="mean")
            dist = self.alpha * chamfer + (1 - self.alpha) * l1

        # Rate
        if isinstance(fbpp, torch.Tensor):
            rate = torch.clamp(fbpp, min=0.0, max=self.max_rate)
        else:
            rate = torch.tensor(fbpp, dtype=pc_recon.dtype, device=pc_recon.device)

        loss = dist + λ * rate
        
        # return loss and detached scalars for logging
        return loss, dist.detach(), rate.detach()

# -------------------------
# High level AE wrapper
# -------------------------
class PointCloudAE(nn.Module):
    def __init__(self, latent_dim=64, latent_bins=16, npoints=8192):
        super().__init__()
        # Encoder: stronger PointNet++ (returns latent and pooled features)
        self.encoder = PointNet2EncoderFull(latent_dim=latent_dim)
        # Decoder: progressive, much smaller than before
        self.decoder = PCNDecoderSmall(latent_dim=latent_dim, coarse_points=512, final_points=npoints)
        # Conditional prob model
        self.prob = ConditionalProbabilityModel(feature_dim=512, hidden_channels=128, latent_bins=latent_bins, latent_channels=latent_dim)
        self.latent_bins = latent_bins
        self.latent_dim = latent_dim
        # quantization range (assumes latents roughly in [0,1] range)
        self.q_min = 0.0
        self.q_max = latent_bins - 1.0

    def forward(self, x):
        """
        x: (B,N,3)
        returns: coarse (B, Cc, 3), fine (B, N, 3), cond_feats (B, F)
                 and latent (B, latent_dim) and quantized latent per-point style (B, latent_dim, 1) if needed
        """
        B, N, _ = x.shape
        latent, cond_feats = self.encoder(x)   # latent: (B, latent_dim), cond_feats: (B, C_out)
        # For per-point latents we need a per-point representation: we optionally tile latent to points
        # simplest: expand latent to (B, latent_dim, 1) and repeat to N -> (B, latent_dim, N)
        y = latent.unsqueeze(-1).repeat(1, 1, N)  # (B, d, N)
        # quantize (STE) into bins
        y_q = quantize_st(y, self.q_min, self.q_max, self.latent_bins)  # float int-like
        # decoder uses continuous latent (we can use latent directly or dequantized -> map back)
        # dequantize y_q into same scale used by decoder: map bins back to [0,1] approximate
        y_dequant = (y_q / (self.latent_bins - 1)) * (self.q_max - self.q_min) + self.q_min
        # collapse per-point latents back to global (take mean)
        y_global = y_dequant.mean(dim=2)  # (B, d)
        coarse, fine = self.decoder(y_global)
        return coarse, fine, cond_feats, y_q  # y_q used by prob model to compute rates

# -------------------------
# Helper: estimate bits-per-point using conditional model
# -------------------------
def estimate_bits_per_point_conditional(y_q, cond_feats, prob_model):
    """
    y_q: quantized latent (B, d, N) float (integer-valued forward)
    cond_feats: encoder features (B, F) or (B, F, N)
    prob_model: instance of ConditionalProbabilityModel (returns pmf)
    Returns scalar fbpp (detached) = bits per point averaged over B & N
    """
    with torch.no_grad():
        # pmf shape (B, K, N)
        mean, scale, pmf = prob_model(y_q, cond_feats)  # note: model expects (B,d,N) and cond
        # pmf: (B, K, N)
        B, K, N = pmf.shape

        # create integer indices in [0, K-1]
        # y_q may be float but integer-valued forward; make sure shape (B, d, N)
        # Condense channel dim (d) into index per-feature if pmf is per-feature channel.
        # For now we assume each channel corresponds to one distribution across K (or you adapt pmf head).
        # We'll collapse over latent channels by averaging bits over channels.
        # Convert y_q to indices with safe clamping
        idx = torch.clamp(y_q.long(), 0, K - 1)  # (B, d, N)

        # If pmf is per-feature-channel grouped differently, you need to align dims.
        # Here pmf is (B, K, N) and we will compute probability for the *first* channel index.
        # A more advanced design: pmf per (channel), with shape (B, d*K, N). Adjust accordingly.
        # We'll compute per-point mean probability across latent channels by converting probabilities
        # to log-likelihoods per channel using the same pmf (approx).
        # Gather probabilities for idx (take first channel if multiple)
        # For now take channel 0 index as representative:
        idx0 = idx[:, 0, :].unsqueeze(1)  # (B,1,N)
        # gather gives (B,1,N)
        probs = torch.gather(pmf, dim=1, index=idx0)  # safe because idx clipped
        # Bits = -log2(prob)
        bits = -torch.log2(probs.clamp(min=1e-9))
        # Average over B,N
        fbpp = bits.mean()
        return fbpp.detach()

# End of file
