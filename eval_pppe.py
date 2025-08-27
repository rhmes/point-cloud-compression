import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from pyntcloud import PyntCloud
from plyfile import PlyData
import torch
from pytorch3d.loss import chamfer_distance
import pn_kit
import open3d as o3d

def compute_p2point_p2plane_psnr(original_pc, reconstructed_pc):
    pc_orig = o3d.io.read_point_cloud(original_pc)
    pc_recon = o3d.io.read_point_cloud(reconstructed_pc)
    if not pc_orig.has_normals():
        pc_orig.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    orig_points = np.asarray(pc_orig.points)
    orig_normals = np.asarray(pc_orig.normals)
    recon_points = np.asarray(pc_recon.points)
    kdtree = o3d.geometry.KDTreeFlann(pc_orig)
    p2point_errors = []
    p2plane_errors = []
    for p in recon_points:
        _, idx, _ = kdtree.search_knn_vector_3d(p, 1)
        nearest_pt = orig_points[idx[0]]
        normal = orig_normals[idx[0]]
        diff_vec = p - nearest_pt
        p2point_errors.append(np.sum(diff_vec ** 2))
        p2plane_errors.append((np.dot(diff_vec, normal)) ** 2)
    p2point_mse = np.mean(p2point_errors)
    p2plane_mse = np.mean(p2plane_errors)
    max_range = np.max(orig_points, axis=0) - np.min(orig_points, axis=0)
    diag = np.linalg.norm(max_range)
    p2point_psnr = 10 * np.log10((diag ** 2) / p2point_mse) if p2point_mse > 0 else float('inf')
    p2plane_psnr = 10 * np.log10((diag ** 2) / p2plane_mse) if p2plane_mse > 0 else float('inf')
    return {"p2point_psnr": p2point_psnr, "p2plane_psnr": p2plane_psnr}

def get_n_points(f):
    return len(PyntCloud.from_file(f).points)

def get_file_size_in_bits(f):
    return os.stat(f).st_size * 8

def main():
    parser = argparse.ArgumentParser(description='Evaluate new compressed/decompressed point cloud data')
    parser.add_argument('--input_glob', default='./data/ModelNet40_pc_01_8192p/**/test/*.ply', help='Original point clouds glob pattern.')
    parser.add_argument('--compressed_path', default='./data/ModelNet40_K256_compressed_p1/', help='Compressed .bin files folder.')
    parser.add_argument('--decompressed_path', default='./data/ModelNet40_K256_decompressed_p1/', help='Decompressed .ply files folder.')
    parser.add_argument('--output_file', default='./eval/ModelNet40_pppe.csv', help='Evaluation Detail saved as csv.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', help='AE Model Device (cpu or cuda)', default=device)
    args = parser.parse_args()
    print(f"Processing on device (gpu/cpu): {args.device}")
    files = np.array(glob(args.input_glob, recursive=True))
    filenames = np.array([os.path.split(x)[1] for x in files])
    ipt_files, p2pointPSNRs, p2planePSNRs, chamfer_ds, n_points_inputs, n_points_outputs, bpps = [], [], [], [], [], [], []
    print('Evaluating...')
    for i in tqdm(range(len(filenames))):
        input_f = files[i]
        # Find compressed file recursively by filename
        comp_bin_candidates = glob(os.path.join(args.compressed_path, '**', filenames[i].replace('.ply', '.bin')), recursive=True)
        comp_bin_f = comp_bin_candidates[0] if comp_bin_candidates else None
        # Find decompressed file recursively by filename
        decomp_candidates = glob(os.path.join(args.decompressed_path, '**', filenames[i].replace('.ply', '.ply')), recursive=True)
        decomp_f = decomp_candidates[0] if decomp_candidates else None

        if not comp_bin_f or not decomp_f or not os.path.exists(decomp_f):
            continue
        ipt_files.append(filenames[i])
        data = compute_p2point_p2plane_psnr(input_f, decomp_f)
        p2pointPSNRs.append(round(data["p2point_psnr"], 3))
        p2planePSNRs.append(round(data["p2plane_psnr"], 3))
        n_points_input = get_n_points(input_f)
        n_points_output = get_n_points(decomp_f)
        n_points_inputs.append(n_points_input)
        n_points_outputs.append(n_points_output)
        bpp = get_file_size_in_bits(comp_bin_f) / n_points_input
        bpps.append(bpp)
        # normed chamfer distance
        input_pc = pn_kit.read_point_cloud(input_f)
        decomp_pc = pn_kit.read_point_cloud(decomp_f)
        input_pc_max = input_pc.max()
        input_pc_min = input_pc.min()
        input_pc = (input_pc - input_pc_min) / (input_pc_max - input_pc_min)
        decomp_pc = (decomp_pc - input_pc_min) / (input_pc_max - input_pc_min)
        chamfer_d, _ = chamfer_distance(torch.Tensor(decomp_pc).unsqueeze(0).to(args.device), torch.Tensor(input_pc).unsqueeze(0).to(args.device))
        chamfer_ds.append(chamfer_d.item())
    print(f'Done! The average p2pointPSNR: {round(np.array(p2pointPSNRs).mean(), 3)} | p2plane PSNR: {round(np.array(p2planePSNRs).mean(), 3)} | chamfer distance: {round(np.array(chamfer_ds).mean(), 8)} | bpp: {round(np.array(bpps).mean(), 3)}')
    df = pd.DataFrame()
    df['filename'] = ipt_files
    df['p2pointPSNR'] = p2pointPSNRs
    df['p2planePSNR'] = p2planePSNRs
    df['chamfer_distance'] = chamfer_ds
    df['n_points_input'] = n_points_inputs
    df['n_points_output'] = n_points_outputs
    df['bpp'] = bpps
    df.to_csv(args.output_file)
    print(f"Evaluation results saved to {args.output_file}")

if __name__ == "__main__":
    main()
