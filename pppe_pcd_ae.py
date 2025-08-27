import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import chamfer_distance

from pn_kit import farthest_point_sample_batch, index_points

# --- PointNet++ expanded implementation ---

# -------------------------
# Utility conv blocks
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
# Single-scale Set Abstraction (SA) block
# -------------------------
class PointNetSetAbstraction(nn.Module):
    """
    Single-scale Set Abstraction block (PointNet++). Uses K-NN grouping via pytorch3d.knn_points.
    Input:
        xyz: (B, N, 3)
        points: optional features (B, D, N) or None
    Output:
        new_xyz: (B, S, 3)
        new_points: (B, D_out, S)
    Params:
        npoint: number of centroids S
        K: neighbors per centroid
        in_channel: input feature channels (points) (excludes xyz)
        mlp: list of integers for per-neighbor MLP output sizes (e.g. [64,64,128])
        bn: use batchnorm
    """
    def __init__(self, npoint, K, in_channel, mlp, bn=True):
        super().__init__()
        self.npoint = npoint
        self.K = K
        self.in_channel = in_channel
        self.mlp = mlp
        self.bn = bn

        # first conv expects (in_channel + 3) because we concat local coords
        last = in_channel + 3
        layers = []
        for out in mlp:
            layers.append(conv2d_bn_relu(last, out) if bn else nn.Sequential(nn.Conv2d(last, out, 1), nn.ReLU()))
            last = out
        self.mlp_stack = nn.ModuleList(layers)  # applied to grouped points

    def forward(self, xyz, points=None):
        """
        xyz: (B, N, 3)
        points: (B, D, N) or None
        returns:
            new_xyz: (B, S, 3)
            new_points: (B, D_out, S)
        """
        B, N, _ = xyz.shape
        S = self.npoint
        device = xyz.device

        # 1) sample centroids with FPS (if S==N pass through)
        if S == N:
            new_xyz = xyz  # (B, N, 3)
        else:
            # use farthest_point_sample_batch (assumes present in pn_kit namespace)
            # returns indices [B, S]
            idx = farthest_point_sample_batch(xyz, S)  # uses pn_kit's implementation
            new_xyz = index_points(xyz, idx)  # (B, S, 3)

        # 2) group: for each new_xyz centroid, find K nearest neighbors in xyz
        # knn_points expects input shapes (B, P, D) and (B, N, D)
        dists, idx_knn, grouped_xyz = knn_points(new_xyz.float(), xyz.float(), K=self.K, return_nn=True)
        # grouped_xyz: (B, S, K, 3); subtract centroid to get local coords
        grouped_xyz = grouped_xyz - new_xyz.view(B, S, 1, 3)

        # 3) optionally concat features if points is provided
        if points is not None:
            # points: (B, D, N) -> convert to (B, N, D)
            points_trans = points.permute(0, 2, 1)
            # index neighbor features [B, S, K, D]
            neigh_features = index_points(points_trans, idx_knn)  # uses pn_kit.index_points
            # concat local coords and neighbor features -> [B, S, K, 3 + D]
            grouped = torch.cat([grouped_xyz, neigh_features], dim=-1)
        else:
            grouped = grouped_xyz  # [B, S, K, 3]

        # transform to conv2d input shape: [B, C_in, K, S]
        grouped = grouped.permute(0, 3, 2, 1).contiguous()

        # apply shared MLPs and max-pool over K
        for layer in self.mlp_stack:
            grouped = layer(grouped)  # [B, C, K, S] -> possibly new C
        new_points = torch.max(grouped, dim=2)[0]  # [B, C_out, S]
        # return new_xyz as (B, S, 3) and new_points as (B, C_out, S)
        return new_xyz, new_points

# -------------------------
# Multi-scale grouping wrapper (MSG)
# -------------------------
class PointNetSetAbstractionMSG(nn.Module):
    """
    Multi-scale Set Abstraction: run several single-scale branches and concat outputs.
    scales: list of dicts each with keys {'K', 'mlp'} or call PointNetSetAbstraction directly with different K/mlp.
    Example:
        msg = PointNetSetAbstractionMSG(npoint=512, scales=[{'K':16,'mlp':[32,32,64]}, {'K':32,'mlp':[64,64,128]}], in_channel=0)
    """
    def __init__(self, npoint, scales, in_channel, bn=True):
        super().__init__()
        self.npoint = npoint
        self.branches = nn.ModuleList()
        for scale in scales:
            K = scale['K']
            mlp = scale['mlp']
            # For each branch, in_channel is the same
            branch = PointNetSetAbstraction(npoint=npoint, K=K, in_channel=in_channel, mlp=mlp, bn=bn)
            self.branches.append(branch)

    def forward(self, xyz, points=None):
        outputs = []
        new_xyz = None
        for branch in self.branches:
            nxyz, npoints = branch(xyz, points)
            new_xyz = nxyz  # identical across branches
            outputs.append(npoints)  # [B, Ci, S]
        # concat along channel dim
        new_points = torch.cat(outputs, dim=1)  # [B, sum(Ci), S]
        return new_xyz, new_points

# -------------------------
# Full PointNet++ Encoder (stacking SA blocks)
# -------------------------
class PointNet2EncoderFull(nn.Module):
    """
    Configurable PointNet++ encoder composed of SA (single or MSG) blocks followed by global pooling and FC layers.
    Example config (good balance):
        sa_blocks = [
            {'type':'MSG', 'npoint':1024, 'scales':[{'K':16,'mlp':[32,32,64]}, {'K':32,'mlp':[64,64,128]}], 'in_channel':0},
            {'type':'SS', 'npoint':256, 'K':32, 'mlp':[128,128,256], 'in_channel':256},
            {'type':'SS', 'npoint':64, 'K':32, 'mlp':[256,256,512], 'in_channel':512}
        ]
        encoder = PointNet2EncoderFull(sa_blocks, latent_dim=256)
    Inputs:
        x: (B, N, 3)
    Output:
        latent: (B, latent_dim)
    """
    def __init__(self, sa_blocks=None, latent_dim=256, bn=True):
        super().__init__()
        # default architecture if not provided (balanced)
        if sa_blocks is None:
            sa_blocks = [
                {'type': 'MSG', 'npoint': 512, 'scales':[{'K':16,'mlp':[32,32,64]}, {'K':32,'mlp':[64,64,128]}], 'in_channel':0},
                {'type': 'SS',  'npoint': 128, 'K':32, 'mlp':[128,128,256], 'in_channel':64+128},  # in_channel = concat of MSG outputs
                {'type': 'SS',  'npoint': 32,  'K':32, 'mlp':[256,256,512], 'in_channel':256}
            ]
        modules = []
        for block in sa_blocks:
            if block['type'] == 'MSG':
                scales = block['scales']
                modules.append(PointNetSetAbstractionMSG(block['npoint'], scales, block.get('in_channel', 0), bn=bn))
            else:  # SS single-scale
                modules.append(PointNetSetAbstraction(block['npoint'], block['K'], block.get('in_channel', 0), block['mlp'], bn=bn))
        self.sa_modules = nn.ModuleList(modules)

        # compute final feature dimension after last SA
        # We'll infer it by summing last mlp outputs; to be safe let user choose latent_dim and we add a conv.
        # Build a small conv stack: in_channels -> latent_dim
        # To get in_channels we examine last module's output channels (sum for MSG, last mlp for SS)
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
        returns latent: (B, latent_dim)
        """
        B, N, _ = x.shape
        xyz = x  # (B, N, 3)
        points = None  # start with no extra point features

        for sa in self.sa_modules:
            xyz, points = sa(xyz, points)  # xyz -> (B, S, 3), points -> (B, C, S)
            # set up next input: next SA expects points in (B, C, S)
            # note: our SA returns points already in (B, C, S)
            # next iteration uses these as 'points' param

        # points now is (B, C_out, S_final)
        # global pooling
        global_feat = torch.max(points, dim=2)[0]  # (B, C_out)
        global_feat = global_feat.unsqueeze(-1)  # (B, C_out, 1) for conv1d
        latent = self.global_conv(global_feat).squeeze(-1)  # (B, latent_dim)
        return latent

# -------------------------
# Utility conv blocks
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
# Single-scale Set Abstraction (SA) block
# -------------------------
class PointNetSetAbstraction(nn.Module):
    """
    Single-scale Set Abstraction block (PointNet++). Uses K-NN grouping via pytorch3d.knn_points.
    Input:
        xyz: (B, N, 3)
        points: optional features (B, D, N) or None
    Output:
        new_xyz: (B, S, 3)
        new_points: (B, D_out, S)
    Params:
        npoint: number of centroids S
        K: neighbors per centroid
        in_channel: input feature channels (points) (excludes xyz)
        mlp: list of integers for per-neighbor MLP output sizes (e.g. [64,64,128])
        bn: use batchnorm
    """
    def __init__(self, npoint, K, in_channel, mlp, bn=True):
        super().__init__()
        self.npoint = npoint
        self.K = K
        self.in_channel = in_channel
        self.mlp = mlp
        self.bn = bn

        # first conv expects (in_channel + 3) because we concat local coords
        last = in_channel + 3
        layers = []
        for out in mlp:
            layers.append(conv2d_bn_relu(last, out) if bn else nn.Sequential(nn.Conv2d(last, out, 1), nn.ReLU()))
            last = out
        self.mlp_stack = nn.ModuleList(layers)  # applied to grouped points

    def forward(self, xyz, points=None):
        """
        xyz: (B, N, 3)
        points: (B, D, N) or None
        returns:
            new_xyz: (B, S, 3)
            new_points: (B, D_out, S)
        """
        B, N, _ = xyz.shape
        S = self.npoint
        device = xyz.device

        # 1) sample centroids with FPS (if S==N pass through)
        if S == N:
            new_xyz = xyz  # (B, N, 3)
        else:
            # use farthest_point_sample_batch (assumes present in pn_kit namespace)
            # returns indices [B, S]
            idx = farthest_point_sample_batch(xyz, S)  # uses pn_kit's implementation
            new_xyz = index_points(xyz, idx)  # (B, S, 3)

        # 2) group: for each new_xyz centroid, find K nearest neighbors in xyz
        # knn_points expects input shapes (B, P, D) and (B, N, D)
        dists, idx_knn, grouped_xyz = knn_points(new_xyz, xyz, K=self.K, return_nn=True)
        # grouped_xyz: (B, S, K, 3); subtract centroid to get local coords
        grouped_xyz = grouped_xyz - new_xyz.view(B, S, 1, 3)

        # 3) optionally concat features if points is provided
        if points is not None:
            # points: (B, D, N) -> convert to (B, N, D)
            points_trans = points.permute(0, 2, 1)
            # index neighbor features [B, S, K, D]
            neigh_features = index_points(points_trans, idx_knn)  # uses pn_kit.index_points
            # concat local coords and neighbor features -> [B, S, K, 3 + D]
            grouped = torch.cat([grouped_xyz, neigh_features], dim=-1)
        else:
            grouped = grouped_xyz  # [B, S, K, 3]

        # transform to conv2d input shape: [B, C_in, K, S]
        grouped = grouped.permute(0, 3, 2, 1).contiguous()

        # apply shared MLPs and max-pool over K
        for layer in self.mlp_stack:
            grouped = layer(grouped)  # [B, C, K, S] -> possibly new C
        new_points = torch.max(grouped, dim=2)[0]  # [B, C_out, S]
        # return new_xyz as (B, S, 3) and new_points as (B, C_out, S)
        return new_xyz, new_points

# -------------------------
# Multi-scale grouping wrapper (MSG)
# -------------------------
class PointNetSetAbstractionMSG(nn.Module):
    """
    Multi-scale Set Abstraction: run several single-scale branches and concat outputs.
    scales: list of dicts each with keys {'K', 'mlp'} or call PointNetSetAbstraction directly with different K/mlp.
    Example:
        msg = PointNetSetAbstractionMSG(npoint=512, scales=[{'K':16,'mlp':[32,32,64]}, {'K':32,'mlp':[64,64,128]}], in_channel=0)
    """
    def __init__(self, npoint, scales, in_channel, bn=True):
        super().__init__()
        self.npoint = npoint
        self.branches = nn.ModuleList()
        for scale in scales:
            K = scale['K']
            mlp = scale['mlp']
            # For each branch, in_channel is the same
            branch = PointNetSetAbstraction(npoint=npoint, K=K, in_channel=in_channel, mlp=mlp, bn=bn)
            self.branches.append(branch)

    def forward(self, xyz, points=None):
        outputs = []
        new_xyz = None
        for branch in self.branches:
            nxyz, npoints = branch(xyz, points)
            new_xyz = nxyz  # identical across branches
            outputs.append(npoints)  # [B, Ci, S]
        # concat along channel dim
        new_points = torch.cat(outputs, dim=1)  # [B, sum(Ci), S]
        return new_xyz, new_points

# -------------------------
# Full PointNet++ Encoder (stacking SA blocks)
# -------------------------
class PointNet2EncoderFull(nn.Module):
    """
    Configurable PointNet++ encoder composed of SA (single or MSG) blocks followed by global pooling and FC layers.
    Example config (good balance):
        sa_blocks = [
            {'type':'MSG', 'npoint':1024, 'scales':[{'K':16,'mlp':[32,32,64]}, {'K':32,'mlp':[64,64,128]}], 'in_channel':0},
            {'type':'SS', 'npoint':256, 'K':32, 'mlp':[128,128,256], 'in_channel':256},
            {'type':'SS', 'npoint':64, 'K':32, 'mlp':[256,256,512], 'in_channel':512}
        ]
        encoder = PointNet2EncoderFull(sa_blocks, latent_dim=256)
    Inputs:
        x: (B, N, 3)
    Output:
        latent: (B, latent_dim)
    """
    def __init__(self, sa_blocks=None, latent_dim=256, bn=True):
        super().__init__()
        # default architecture if not provided (balanced)
        if sa_blocks is None:
            sa_blocks = [
                {'type': 'MSG', 'npoint': 512, 'scales':[{'K':16,'mlp':[32,32,64]}, {'K':32,'mlp':[64,64,128]}], 'in_channel':0},
                {'type': 'SS',  'npoint': 128, 'K':32, 'mlp':[128,128,256], 'in_channel':64+128},  # in_channel = concat of MSG outputs
                {'type': 'SS',  'npoint': 32,  'K':32, 'mlp':[256,256,512], 'in_channel':256}
            ]
        modules = []
        for block in sa_blocks:
            if block['type'] == 'MSG':
                scales = block['scales']
                modules.append(PointNetSetAbstractionMSG(block['npoint'], scales, block.get('in_channel', 0), bn=bn))
            else:  # SS single-scale
                modules.append(PointNetSetAbstraction(block['npoint'], block['K'], block.get('in_channel', 0), block['mlp'], bn=bn))
        self.sa_modules = nn.ModuleList(modules)

        # compute final feature dimension after last SA
        # We'll infer it by summing last mlp outputs; to be safe let user choose latent_dim and we add a conv.
        # Build a small conv stack: in_channels -> latent_dim
        # To get in_channels we examine last module's output channels (sum for MSG, last mlp for SS)
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
        returns latent: (B, latent_dim)
        """
        B, N, _ = x.shape
        xyz = x  # (B, N, 3)
        points = None  # start with no extra point features

        for sa in self.sa_modules:
            xyz, points = sa(xyz, points)  # xyz -> (B, S, 3), points -> (B, C, S)
            # set up next input: next SA expects points in (B, C, S)
            # note: our SA returns points already in (B, C, S)
            # next iteration uses these as 'points' param

        # points now is (B, C_out, S_final)
        # global pooling
        global_feat = torch.max(points, dim=2)[0]  # (B, C_out)
        global_feat = global_feat.unsqueeze(-1)  # (B, C_out, 1) for conv1d
        latent = self.global_conv(global_feat).squeeze(-1)  # (B, latent_dim)
        return latent

# -------------------------
# End of expanded PointNet++ encoder
# -------------------------

class PointNet2Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # Replace with PointNet++ Set Abstraction layers
        self.mlp1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x):  # x: (B, N, 3)
        feat = self.mlp1(x)
        feat = self.mlp2(feat)
        feat = torch.max(feat, dim=1)[0]  # global pooling
        return self.fc(feat)  # (B, latent_dim)

class PCNDecoder(nn.Module):
    def __init__(self, latent_dim=256, coarse_points=1024, final_points=2048):
        super().__init__()
        self.fc_coarse = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(),
            nn.Linear(1024, coarse_points * 3)
        )
        self.expansion = nn.Sequential(
            nn.Linear(coarse_points*3, final_points*3)
        )
        self.coarse_points = coarse_points
        self.final_points = final_points

    def forward(self, latent):
        B = latent.size(0)
        coarse = self.fc_coarse(latent).view(B, self.coarse_points, 3)
        fine = self.expansion(coarse.view(B, -1)).view(B, self.final_points, 3)
        return coarse, fine

# Loss calculation class
class RateDistortionLoss(nn.Module):
    """
    Rate–Distortion loss for point cloud compression:
        L = Distortion + λ * Rate
    where Distortion = Chamfer Distance (or alternative),
    and Rate = bits per point (fbpp).
    """
    def __init__(self, loss_type="chamfer"):
        super(RateDistortionLoss, self).__init__()
        self.loss_type = loss_type.lower()

    def forward(self, pc_pred, pc_target, fbpp, λ=1.0):
        """
        Args:
            pc_pred   : (B, N, 3) reconstructed point cloud
            pc_target : (B, N, 3) ground-truth point cloud
            fbpp      : (float or tensor) bits-per-point (rate term)
            λ         : (float) Lagrangian multiplier
        Returns:
            loss : scalar (total rate–distortion loss)
            dist : scalar (distortion only)
            rate : scalar (rate only)
        """
        # ----- Distortion -----
        if self.loss_type == "chamfer":
            dist, _ = chamfer_distance(pc_pred, pc_target, batch_reduction="mean")
        elif self.loss_type == "l2":
            dist = torch.mean((pc_pred - pc_target) ** 2)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # ----- Rate -----
        if isinstance(fbpp, torch.Tensor):
            rate = fbpp.mean()
        else:
            rate = torch.tensor(fbpp, dtype=pc_pred.dtype, device=pc_pred.device)

        # ----- Total Loss -----
        loss = dist + λ * rate
        return loss, dist, rate

# wrapper class for loss (RateDistortionLoss)
class get_loss(nn.Module):
    def __init__(self, loss_type="chamfer"):
        super(get_loss, self).__init__()
        self.criterion = RateDistortionLoss(loss_type=loss_type)

    def forward(self, pc_pred, pc_target, fbpp, λ=1.0):
        return self.criterion(pc_pred, pc_target, fbpp, λ=λ)

class PointCloudAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = PointNet2Encoder(latent_dim=latent_dim)
        self.decoder = PCNDecoder(latent_dim=latent_dim)

    def forward(self, x):
        latent = self.encoder(x)          # (B, latent_dim)
        coarse, fine = self.decoder(latent)
        return coarse, fine
