import torch
import torch.nn as nn
import torch.nn.functional as F
from AE import STEQuantize       # Quantizer of latent space
from pointnet_sa_module import PointnetSAModule # PointNet++ Set Abstraction
from pytorch3d.loss import chamfer_distance 

# POINTNET++ ENCODER uses set abstraction with 3D features
class PointNetPP(nn.Module):
    def __init__(
        self,
        points=512,
        sa1_mlp=[64, 64, 128],
        sa2_mlp=[128, 128, 128, 256],
        sa3_mlp=[256, 256, 512],
        feature_dim=1024,
        bn=False
    ):
        """
        Args:
            sa1_mlp (list[int]): MLP sizes for first Set Abstraction layer (default [64, 64, 128])
            sa2_mlp (list[int]): MLP sizes for second Set Abstraction layer (default [128, 128, 128, 256])
            sa3_mlp (list[int): MLP sizes for third Set Abstraction layer (default [256, 256, 512])
            feature_dim (int): output feature dimension (default 1024)
            bn (bool): whether to use batch norm (default False, matching snippet)
        """
        super(PointNetPP, self).__init__()

        self.sa1 = PointnetSAModule(npoint=points, radius=0.2, nsample=32,
            mlp=[3] + sa1_mlp, use_xyz=True, in_channels=0
        )
        self.sa2 = PointnetSAModule(npoint=128, radius=0.4, nsample=64,
            mlp=sa2_mlp, use_xyz=True, in_channels=128
        )
        self.sa3 = PointnetSAModule(npoint=32, radius=0.8, nsample=128,
            mlp=sa3_mlp + [feature_dim], use_xyz=True, in_channels=256
        )

    def forward(self, xyz, features=None):
        # xyz: (B, N, 3), features: (B, C, N) or None
        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        xyz, features = self.sa3(xyz, features) # (B, feature_dim, npoint=32)
        global_features = torch.max(features, dim=2)[0]  # (B, feature_dim)

        return xyz, global_features


# FOLDINGNET DECODER uses a 2D grid with learnable folding
class FoldingNet(nn.Module):
    """
    FoldingNet-style decoder: 
    - Generates a 2D grid
    - Concatenates with latent vector (quantized)
    - Applies folding MLPs to reconstruct 3D point cloud
    """
    def __init__(self, points=512, grid_size=45, feature_dim=1024):
        super(FoldingNet, self).__init__()
        self.grid_size = grid_size
        self.num_points = grid_size * grid_size
        self.feature_dim = feature_dim
        self.size = points

        # First folding stage
        self.mlp1 = nn.Sequential(
            nn.Conv1d(feature_dim + 2, self.size, 1),
            nn.ReLU(),
            nn.Conv1d(self.size, self.size, 1),
            nn.ReLU(),
            nn.Conv1d(self.size, 3, 1)
        )

        # Second folding stage
        self.mlp2 = nn.Sequential(
            nn.Conv1d(feature_dim + 3, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1)
        )

    def build_grid(self, batch_points, device):
        """Builds a fixed 2D grid in [-1,1]x[-1,1]."""
        x = torch.linspace(-1, 1, self.grid_size)
        y = torch.linspace(-1, 1, self.grid_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # [N,2]
        grid = grid.unsqueeze(0).repeat(batch_points, 1, 1).to(device)  # [B,N,2]
        return grid

    def forward(self, latent_quantized):
        """
        latent_quantized: [B, F]
        returns: [B, N, 3]
        """
        B = latent_quantized.size(0)
        device = latent_quantized.device

        grid = self.build_grid(B, device)         # [B, N, 2]
        latent_expanded = latent_quantized.unsqueeze(1).repeat(1, self.num_points, 1)  # [B, N, F]
        feat_grid = torch.cat([grid, latent_expanded], dim=-1)  # [B, N, F+2]

        x = feat_grid.transpose(2, 1)             # [B, F+2, N]
        coarse = self.mlp1(x)                     # [B, 3, N]

        feat_folding = torch.cat([coarse, latent_expanded.transpose(2, 1)], dim=1)  # [B, F+3, N]
        fine = self.mlp2(feat_folding)            # [B, 3, N]

        return fine.transpose(2, 1)               # [B, N, 3]
    
### PointNet++ FoldingNet Autoencoder (PointNetPP_FoldingNet_AE) consists of:
# * PointNet++ Encoder
# * FoldingNet Decoder Reconstruct 3D features from latent quantized representation
class PPPF_AE(nn.Module):
    def __init__(self, K=512, k=0, d=16, L=7, dim=1024):
        super(PPPF_AE, self).__init__()
        self.L = L
        self.encoder = PointNetPP(points=K, feature_dim=dim)
        self.decoder = FoldingNet(points=K, grid_size=d)

        # Projection layers
        self.enc_proj = nn.Linear(dim, d)
        self.dec_proj = nn.Linear(d, dim)

        # Quantizer
        self.quantize = STEQuantize.apply

    def forward(self, xyz):
        """
        xyz: [B, N, 3]
        """
        # Encode
        _, latent = self.encoder(xyz)  # <-- unpack (xyz_down, global_features=[B, feature_dim])

        # Quantization (elementwise over 1024-D latent)
        spread = self.L - 0.2
        latent = torch.sigmoid(latent) * spread - spread / 2
        # Down-project to small bottleneck
        z_bn = self.enc_proj(latent)            # [B, 16]

        # Quantize
        latent_quantized = self.quantize(z_bn)             # [B, 16]

        # Up-project back to 1024-D
        latent_decode = self.dec_proj(latent_quantized)           # [B, 1024]

        # Decode
        recon = self.decoder(latent_decode)            # [B, N, 3]

        return recon, latent, latent_quantized

# Loss of PointNet++ FoldingNet AE
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pc_pred, pc_target, fbpp, λ):
        """
        Args:
            pc_pred   : (B, N, 3) reconstructed point cloud
            pc_target : (B, N, 3) ground-truth point cloud
            fbpp      : (float or tensor) bits-per-point (rate term)
            λ         : (float) Lagrangian multiplier for rate-distortion trade-off
        Returns:
            loss : scalar loss = ChamferDist + λ * rate
        """
        # Chamfer distance
        d, _ = chamfer_distance(pc_pred, pc_target)

        # Rate term (ensure it's a scalar)
        if isinstance(fbpp, torch.Tensor) and fbpp.ndim > 0:
            r = fbpp.mean()
        else:
            r = fbpp

        # Rate–distortion loss
        loss = d + λ * r
        return loss

# Conditional probability model based on PointNet++ Encoder
class ConditionalProbabilityModel(nn.Module):
    def __init__(self, L, d):
        super(ConditionalProbabilityModel, self).__init__()
        self.L = L
        self.d = d
        # PointNet++ encoder backbone
        self.model_pnpp = PointNetPP(
            sa1_mlp=[64, 64, 128],
            sa2_mlp=[128, 128, 256],
            sa3_mlp=[256, 512, 1024],
            bn=False
        )
        # Adjusted output channels since PN++ gives 1024-d global feature
        feature_dim = 1024
        self.model_mlp = nn.Sequential(
            nn.Conv2d(3 + feature_dim, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, d * self.L, 1),
        )

    def forward(self, sampled_xyz):
        """
        sampled_xyz: (B, S, 3)
        returns pmf: (B, S, d, L)
        """
        B, S, _ = sampled_xyz.shape

        # PointNet++ encoder -> global feature (B, 1024)
        _, feature = self.model_pnpp(sampled_xyz)                 # <-- no transpose; unpack tuple

        # Expand global feature to all points
        feature_expand = feature.unsqueeze(1).repeat(1, S, 1)     # (B, S, 1024)

        # Concatenate raw XYZ + PN++ feature
        mlp_input = torch.cat((sampled_xyz, feature_expand), dim=2)  # (B, S, 3+1024)

        # Reshape for Conv2d
        mlp_input = mlp_input.unsqueeze(-1).transpose(1, 2)       # (B, 3+1024, S, 1)

        # MLP -> logits
        output = self.model_mlp(mlp_input)                        # (B, d*L, S, 1)

        # Reshape -> pmf
        output = output.transpose(1, 2).view(B, S, self.d, self.L)
        pmf = F.softmax(output, dim=3)
        return pmf

class AE(PPPF_AE):
    """Wrapper for generic AE usage."""
    pass
