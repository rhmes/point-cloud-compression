import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points, knn_points, knn_gather, ball_query


# Wrapper for pointnet++_ops using PyTorch3D
class PointnetPPOps:
    @staticmethod
    def furthest_point_sample(xyz, npoint):
        # xyz: (B, N, 3)
        _, idx = sample_farthest_points(xyz, K=npoint)
        return idx  # (B, npoint)

    @staticmethod
    def ball_query(radius, nsample, xyz, new_xyz):
        # xyz: (B, N, 3), new_xyz: (B, M, 3)
        idx = ball_query(new_xyz, xyz, K=nsample, radius=radius)  # (B, M, nsample)
        return idx

    @staticmethod
    def group_points(features, idx):
        # If user passed the full KNN object instead of just idx
        if hasattr(idx, "idx"):  
            idx = idx.idx  # extract indices tensor
        # features: (B, N, C), idx: (B, M, nsample)
        idx = idx.clamp(min=0)
        grouped = knn_gather(features, idx)  # (B, M, nsample, C)
        return grouped

    @staticmethod
    def knn_point(k, xyz, new_xyz):
        dists, idx, _ = knn_points(new_xyz, xyz, K=k, return_nn=True)
        return dists, idx


# PointNet++ Set Abstraction module
class PointnetSAModule(nn.Module):
    def __init__(self, npoint, radius, nsample, mlp, use_xyz=True, in_channels=0):
        super(PointnetSAModule, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz

        # input = in_channels + (3 if use_xyz else 0)
        last_channel = in_channels + (3 if use_xyz else 0)

        layers = []
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel

        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features=None):
        """
        xyz: (B, N, 3)
        features: (B, C, N)
        """
        B, N, _ = xyz.shape

        # ---- 1. FPS ----
        fps_idx = PointnetPPOps.furthest_point_sample(xyz, self.npoint)  # (B, npoint)
        fps_idx = fps_idx.clamp(min=0)  # avoid -1
        new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, npoint, 3)

        # ---- 2. Ball Query ----
        idx = PointnetPPOps.ball_query(self.radius, self.nsample, xyz, new_xyz)  # (B, npoint, nsample)

        # ---- 3. Grouping ----
        if features is not None:
            features = features.permute(0, 2, 1)  # (B, N, C)
            grouped_features = PointnetPPOps.group_points(features, idx)  # (B, npoint, nsample, C)
        else:
            grouped_features = None

        if self.use_xyz:
            grouped_xyz = PointnetPPOps.group_points(xyz, idx)  # (B, npoint, nsample, 3)
            if grouped_features is not None:
                grouped_features = torch.cat([grouped_features, grouped_xyz], dim=-1)
            else:
                grouped_features = grouped_xyz

        # ---- 4. Apply MLP ----
        # input to Conv2d should be (B, C, npoint, nsample)
        grouped_features = grouped_features.permute(0, 3, 1, 2)  # (B, C, npoint, nsample)
        new_features = self.mlp(grouped_features)  # (B, C_out, npoint, nsample)
        new_features = torch.max(new_features, 3)[0]  # (B, C_out, npoint)

        return new_xyz, new_features
