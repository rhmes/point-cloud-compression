import os
import glob
import torch
import numpy as np
import argparse
import struct
from plyfile import PlyData, PlyElement
from tqdm import tqdm

import pn_kit
from pppe_pcd_ae import PointCloudAE as PointNet2AE
from pppe_pcd_ae import ConditionalProbabilityModel
from train_pppe_pcd_ae import load_checkpoints

# ----------------------
# Utility Functions
# ----------------------

def load_point_cloud(ply_path, device):
    plydata = PlyData.read(ply_path)
    vertices = np.vstack([plydata['vertex']['x'],
                          plydata['vertex']['y'],
                          plydata['vertex']['z']]).T
    points = torch.tensor(vertices, dtype=torch.float32, device=device)
    return points


def save_point_cloud(points, out_path):
    points = points.cpu().numpy()
    vertex = np.array([(p[0], p[1], p[2]) for p in points],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(out_path)


def save_binary(latent, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = latent.detach().cpu().numpy().astype(np.float32)
    with open(out_path, "wb") as f:
        f.write(struct.pack("I", arr.shape[0]))
        arr.tofile(f)


def load_binary(in_path, device):
    with open(in_path, "rb") as f:
        n = struct.unpack("I", f.read(4))[0]
        arr = np.fromfile(f, dtype=np.float32).reshape(n, -1)
    return torch.tensor(arr, dtype=torch.float32, device=device)


# ----------------------
# Compression Pipeline
# ----------------------

def compress(ae, prob, pc, out_bin):
    ae.eval()
    prob.eval()

    # Ensure pc has batch dimension [1, N, 3]
    if pc.ndim == 2:
        pc = pc.unsqueeze(0)
    pc, center, longest = pn_kit.normalize(pc, margin=0.01)

    with torch.no_grad():
        latent, conv = ae.encoder(pc)  # [1, N, 3] -> latent
    save_binary(latent.squeeze(0), out_bin)


# ----------------------
# Main CLI
# ----------------------

def main(args):
    device = torch.device(args.device)

    # Load models
    ae = PointNet2AE(latent_dim=args.K, L=args.L, npoints=args.N).to(device)
    prob = ConditionalProbabilityModel(latent_dim=args.K).to(device)
    op = torch.optim.Adam(list(ae.parameters()) + list(prob.parameters()), lr=1e-4)  
    _ = load_checkpoints(ae, prob, op, args.model_load_folder, best=args.best)

    # Find all point clouds
    ply_files = glob.glob(args.input_glob, recursive=True)
    print(f"Found {len(ply_files)} point clouds.")

    # Compress each point cloud
    for ply_path in tqdm(ply_files, desc="Compressing", unit="file"):
        pc = load_point_cloud(ply_path, device)

        rel_path = os.path.relpath(ply_path, start=os.path.commonpath([ply_path, args.input_glob.split("**")[0]]))
        bin_path = os.path.join(args.compressed_path, rel_path).replace(".ply", ".bin")

        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        compress(ae, prob, pc, bin_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Point Cloud Compression",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_glob', help='Point clouds glob pattern for compression.',
                        default='/mnt/hdd/datasets_yk/ModelNet40_pc_01_8192p/**/test/*.ply')
    parser.add_argument('compressed_path', help='Compressed .bin files folder.',
                        default='./data/ModelNet40_K256_compressed/')
    parser.add_argument('model_load_folder', help='Directory where to load trained models.',
                        default='./model/K256/')
    parser.add_argument('--N', help='Number of points for the model.',
                        default=8192)
    parser.add_argument('--K', help='Latent space dimension.',
                        default=256)
    parser.add_argument('--L', type=int, help='Quantization level.', 
                        default=7)
    parser.add_argument('--best', action='store_true')

    # Device option
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', help='AE Model Device (cpu or cuda)', default=device)

    args = parser.parse_args()
    main(args)
