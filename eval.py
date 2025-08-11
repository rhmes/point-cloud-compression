import os
# import subprocess
import argparse

import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from pyntcloud import PyntCloud
from plyfile import PlyData

import torch
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points
from pytorch3d.loss import chamfer_distance

import pn_kit
import open3d as o3d

parser = argparse.ArgumentParser(
    prog='eval.py',
    description='Evaluate point cloud patches',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', default='./data/ModelNet40_pc_01_8192p/**/test/*.ply', help='Point clouds glob pattern for compression.')
parser.add_argument('--compressed_path', default='./data/ModelNet40_K256_compressed/', help='Compressed .bin files folder.')
parser.add_argument('--decompressed_path', default='./data/ModelNet40_K256_decompressed/', help='Decompressed .ply files folder.')
parser.add_argument('--output_file', default='./eval/ModelNet40_K256.csv', help='Evaluation Detail saved as csv.')

# Set device for processing (cuda/cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--device', help='AE Model Device (cpu or cuda)', default=device)

args = parser.parse_args()
print(f"Processing on device (gpu/cpu): {args.device}")

# CALC PSNR BETWEEN FILE AND DECOMPRESSED FILE
import numpy as np
import open3d as o3d


def compute_p2point_p2plane_psnr(original_pc, reconstructed_pc):
    """
    Computes both p2point and p2plane PSNR metrics between two point clouds.
    
    Args:
        original_pc (str): Path to original point cloud file (.ply, .pcd, etc.)
        reconstructed_pc (str): Path to reconstructed/compressed point cloud file.

    Returns:
        dict: {"p2point_psnr": float, "p2plane_psnr": float}
    """
    # Load point clouds
    pc_orig = o3d.io.read_point_cloud(original_pc)
    pc_recon = o3d.io.read_point_cloud(reconstructed_pc)

    # Estimate normals if not present
    if not pc_orig.has_normals():
        pc_orig.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    # Convert to numpy arrays
    orig_points = np.asarray(pc_orig.points)
    orig_normals = np.asarray(pc_orig.normals)
    recon_points = np.asarray(pc_recon.points)

    # Build KDTree for nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pc_orig)

    p2point_errors = []
    p2plane_errors = []

    for p in recon_points:
        # Find nearest neighbor in original point cloud
        _, idx, _ = kdtree.search_knn_vector_3d(p, 1)
        nearest_pt = orig_points[idx[0]]
        normal = orig_normals[idx[0]]

        diff_vec = p - nearest_pt
        p2point_errors.append(np.sum(diff_vec ** 2))  # squared Euclidean
        p2plane_errors.append((np.dot(diff_vec, normal)) ** 2)  # squared projection

    # Compute MSEs
    p2point_mse = np.mean(p2point_errors)
    p2plane_mse = np.mean(p2plane_errors)

    # Compute bounding box diagonal
    max_range = np.max(orig_points, axis=0) - np.min(orig_points, axis=0)
    diag = np.linalg.norm(max_range)

    # Compute PSNRs
    p2point_psnr = 10 * np.log10((diag ** 2) / p2point_mse) if p2point_mse > 0 else float('inf')
    p2plane_psnr = 10 * np.log10((diag ** 2) / p2plane_mse) if p2plane_mse > 0 else float('inf')

    return {
        "p2point_psnr": p2point_psnr,
        "p2plane_psnr": p2plane_psnr
    }

def compute_psnr(original_pcd_path, reconstructed_pcd_path):
    pcd1 = o3d.io.read_point_cloud(original_pcd_path)
    pcd2 = o3d.io.read_point_cloud(reconstructed_pcd_path)

    # Ensure same number of points
    if len(pcd1.points) != len(pcd2.points):
        raise ValueError("Point clouds must have the same number of points")

    # Convert to numpy arrays
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)

    # Compute MSE
    mse = np.mean(np.sum((pts1 - pts2) ** 2, axis=1))

    # Compute MAX based on bounding box diagonal
    bbox = np.vstack((pts1, pts2))
    max_val = np.linalg.norm(np.max(bbox, axis=0) - np.min(bbox, axis=0))

    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr

def compute_bitrate(compressed_file_path, num_points):
    file_size_bytes = os.path.getsize(compressed_file_path)
    bitrate = (8 * file_size_bytes) / num_points
    return bitrate

def calc_uc(input_pc, decomp_pc):

    def KNN_Region(pc, point, K):
        pc = torch.Tensor(pc)
        point = torch.Tensor(point)
        dist, group_idx, grouped_xyz = knn_points(point.view(1, 1, 3), pc.unsqueeze(0), K=K, return_nn=True)
        grouped_xyz -= point.view(1, 1, 1, 3)
        x_patches = grouped_xyz.view(K, 3)
        x_patches = x_patches.numpy()
        return x_patches

    def calc_self_neighboor_dist(pc):
        pc = torch.Tensor(pc)
        dist = torch.cdist(pc, pc, p=2)
        values, indices = torch.topk(dist, k=2, largest=False)
        neighboor_dist = values[:, 1]
        neighboor_dist = neighboor_dist.numpy()
        return neighboor_dist

    input_region = KNN_Region(input_pc, input_pc[0], 1024)
    decomp_region = KNN_Region(decomp_pc, decomp_pc[0], 1024)
    input_region_dist = calc_self_neighboor_dist(input_region)
    decomp_region_dist = calc_self_neighboor_dist(decomp_region)
    uc = np.var(decomp_region_dist) / np.var(input_region_dist)
    return uc

def get_n_points(f):
    return len(PyntCloud.from_file(f).points)

def get_file_size_in_bits(f):
    return os.stat(f).st_size * 8

# GET FILE NAME FROM DECOMPRESSED PATH
files = np.array(glob(args.input_glob, recursive=True))
filenames = np.array([os.path.split(x)[1] for x in files])

# .csv COLUMNS: [filename, p2pointPSNR, p2planePSNR, n_points_input, n_points_output, bpp]
ipt_files, p2pointPSNRs, p2planePSNRs, chamfer_ds, n_points_inputs, n_points_outputs, bpps, ucs = [], [], [], [], [], [], [], []

print('Evaluating...')
for i in tqdm(range(len(filenames))):
    input_f = files[i]
    comp_s_f = os.path.join(args.compressed_path, filenames[i] + '.s.bin')
    comp_p_f = os.path.join(args.compressed_path, filenames[i] + '.p.bin')
    comp_c_f = os.path.join(args.compressed_path, filenames[i] + '.c.bin')
    decomp_f = os.path.join(args.decompressed_path, filenames[i] + '.bin.ply')

    if not os.path.exists(decomp_f):
        continue

    ipt_files.append(filenames[i])
    # GET PSNR
    # data = pc_error(input_f, decomp_f)
    data = compute_p2point_p2plane_psnr(input_f, decomp_f)
    p2pointPSNRs.append(round(data["p2point_psnr"], 3))
    p2planePSNRs.append(round(data["p2plane_psnr"], 3))
    # GET NUMBER OF POINTS
    n_points_input = get_n_points(input_f)
    n_points_output = get_n_points(decomp_f)
    n_points_inputs.append(n_points_input)
    n_points_outputs.append(n_points_output)
    # GET BPP
    bpp = (get_file_size_in_bits(comp_s_f) + get_file_size_in_bits(comp_p_f) + get_file_size_in_bits(comp_c_f)) / n_points_input
    bpps.append(bpp)

    # CALC THE UNIFORMITY COEFFICIENT
    input_pc = pn_kit.read_point_cloud(input_f)
    decomp_pc = pn_kit.read_point_cloud(decomp_f)
    uc = calc_uc(input_pc, decomp_pc)
    ucs.append(np.round(uc, 3))

    # normed chamfer distance
    input_pc_max = input_pc.max()
    input_pc_min = input_pc.min()
    input_pc = (input_pc - input_pc_min) / (input_pc_max - input_pc_min)
    decomp_pc = (decomp_pc - input_pc_min) / (input_pc_max - input_pc_min)

    chamfer_d, loss_normals = chamfer_distance(torch.Tensor(decomp_pc).unsqueeze(0).to(args.device), torch.Tensor(input_pc).unsqueeze(0).to(args.device))
    chamfer_ds.append(chamfer_d.item())

    
print(f'Done! The average p2pointPSNR: {round(np.array(p2pointPSNRs).mean(), 3)} | p2plane PSNR: {round(np.array(p2planePSNRs).mean(), 3)} | chamfer distance: {round(np.array(chamfer_ds).mean(), 8)} | bpp: {round(np.array(bpps).mean(), 3)} | uc: {round(np.array(ucs).mean(), 3)}')


# SAVE AS AN EXCEL .csv
df = pd.DataFrame()
df['filename'] = ipt_files
df['p2pointPSNR'] = p2pointPSNRs
df['p2planePSNR'] = p2planePSNRs
df['chamfer_distance'] = chamfer_ds
df['n_points_input'] = n_points_inputs
df['n_points_output'] = n_points_outputs
df['bpp'] = bpps
df['uniformity coefficient'] = ucs
df.to_csv(args.output_file)

print(f"Evaluation results saved to {args.output_file}")