# PointCloudCompression (based on IPDAE)

This project is based on the [IPDAE repository](https://github.com/I2-Multimedia-Lab/IPDAE) and implements an improved patch-based deep autoencoder for 3D point cloud geometry compression.

## Key Features

- **Autoencoder-based model** for point cloud compression.
- Supports **ModelNet**, **ShapeNet**, and **Stanford3D** datasets.
- Compresses and decompresses **3D points (XYZ)** only.  
  *TODO: Fuse RGB with XYZ compression in future updates.*
- Implements evaluation metrics: **PSNR**, **Chamfer Distance**, and **bitrate-per-pointcloud (bpp)**.
- Installation scripts for both **CPU** and **GPU** machines.
- Uses the produced `venv_{mode}` (e.g., `venv_cpu`, `venv_gpu`) to run all scripts.

## Usage


The project uses the **same inputs and commands** as the original IPDAE repo. See the original [IPDAE documentation](https://github.com/I2-Multimedia-Lab/IPDAE) for dataset preparation and usage instructions.

After running the appropriate setup script (`venv_cpu_setup.sh` or `venv_gpu_setup.sh`), activate the environment and use the following example commands:

### Training

Train the autoencoder model on the ModelNet40 training set:
```
python ./train.py './data/ModelNet40_pc_01_8192p/**/train/*.ply' './model/K256' --K 256
```

### Compression

Compress point cloud test files using the trained model:
```
python ./compress.py  './data/ModelNet40_pc_01_8192p/**/test/*.ply' './data/ModelNet40_K256_compressed' './model/K256' --K 256
```

### Decompression

Decompress the compressed point cloud files:
```
python ./decompress.py  './data/ModelNet40_K256_compressed' './data/ModelNet40_K256_decompressed' './model/K256' --K 256
```

### Evaluation

Evaluate the compression results using PSNR, Chamfer distance, and bpp metrics:
```
python ./eval.py './data/ModelNet40_pc_01_8192p/**/test/*.ply' './data/ModelNet40_K256_compressed' './data/ModelNet40_K256_decompressed' './eval/ModelNet40_K256.csv'  '../geo_dist/build/pc_error'
```

### Visualization

Visualize evaluation metrics and save plots:
```
python ./visualize.py --csv <path_to_eval_csv> --outdir <output_directory>
```

### Comparison

Compare original and reconstructed point clouds and display evaluation metrics:
```
python ./compare.py --input_dir <compressed_ply_dir> --recon_dir <decompressed_ply_dir> --csv_path <metrics_csv>
```

## Installation

1. For CPU:
   ```bash
   bash venv_cpu_setup.sh
   source venv_cpu/bin/activate
   ```
2. For GPU:
   ```bash
   bash venv_gpu_setup.sh
   source venv_gpu/bin/activate
   ```

## Notes

- All scripts should be run from within the activated virtual environment (`venv_cpu` or `venv_gpu`).
- For evaluation, additional metrics (PSNR, Chamfer distance, bpp) are included.
- The project currently focuses on geometry (XYZ) compression; RGB attribute compression is a planned feature.
