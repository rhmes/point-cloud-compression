import os
import numpy as np
from glob import glob
import argparse
import pn_kit

parser = argparse.ArgumentParser(description='Preload point cloud data and save as .npy for fast access')
parser.add_argument('--train_glob', help='Glob pattern for point cloud files', 
                    default='./data/ModelNet40_pc_01_8192p/**/train/*.ply')
parser.add_argument('--output_npy', help='Path to save the .npy file',
                    default='./data/ModelNet40_pc_01_8192p/train.npy')

if __name__ == '__main__':
    args = parser.parse_args()
    files = np.array(glob(args.train_glob, recursive=True))
    print(f'Found {len(files)} files')
    points = pn_kit.read_point_clouds(files)
    print(f'Loaded points shape: {points.shape}')
    np.save(args.output_npy, points)
    print(f'Saved to {args.output_npy}')
