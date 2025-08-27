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

def load_binary(in_path, device):
    with open(in_path, "rb") as f:
        n = struct.unpack("I", f.read(4))[0]
        arr = np.fromfile(f, dtype=np.float32)
        arr = arr.reshape(1, n)  # [1, n] instead of [n, 1]
    return torch.tensor(arr, dtype=torch.float32, device=device)


def save_point_cloud(points, out_path):
    points = points.cpu().numpy()
    vertex = np.array([(p[0], p[1], p[2]) for p in points],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(out_path)

# ----------------------
# Decompression Pipeline
# ----------------------

def decompress(ae, latent, out_ply):
    ae.eval()
    with torch.no_grad():
        # latent: [N, d] or [1, N, d]        
        coarse, fine = ae.decoder(latent)  # [1, N, 3]
        points = fine.squeeze(0)
    save_point_cloud(points, out_ply)

# ----------------------
# Main CLI
# ----------------------

def main(args):
    device = torch.device(args.device)
    # Load model
    ae = PointNet2AE().to(device)
    prob = ConditionalProbabilityModel().to(device)
    op = torch.optim.Adam(list(ae.parameters()) + list(prob.parameters()), lr=1e-4)  
    _ = load_checkpoints(ae, prob, op, args.model_load_folder)
    # Find all binary files
    bin_files = glob.glob(args.input_glob, recursive=True)
    print(f"Found {len(bin_files)} compressed files.")
    # Decompress each file
    for bin_path in tqdm(bin_files, desc="Decompressing", unit="file"):
        latent = load_binary(bin_path, device)
        rel_path = os.path.relpath(bin_path, start=os.path.commonpath([bin_path, args.input_glob.split("**")[0]]))
        ply_path = os.path.join(args.decompressed_path, rel_path).replace(".bin", ".bin.ply")
        os.makedirs(os.path.dirname(ply_path), exist_ok=True)
        decompress(ae, latent, ply_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Point Cloud Decompression")
    parser.add_argument('input_glob', help='Compressed .bin files glob pattern.',
                        default='./data/ModelNet40_K256_compressed/**/*.bin')
    parser.add_argument('decompressed_path', help='Output folder for decompressed .ply files.',
                        default='./data/ModelNet40_K256_decompressed_ply/')
    parser.add_argument('model_load_folder', help='Directory where to load trained models.',
                        default='./model/K256/')
    parser.add_argument('N', help='Number of points for the model.',
                        default=8192)
    parser.add_argument('K', help='Latent space dimension.',
                        default=256)

    # Device option
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', help='AE Model Device (cpu or cuda)', default=device)
    args = parser.parse_args()
    main(args)
