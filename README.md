# Point Cloud Compression (IPDAE-based)

An improved patch-based deep autoencoder for 3D point cloud geometry compression, based on the [IPDAE repository](https://github.com/I2-Multimedia-Lab/IPDAE). This project focuses on efficient geometry (XYZ) compression for large-scale 3D datasets, with support for ModelNet, ShapeNet, and Stanford3D.

## Key Features

- **Autoencoder-based model** for point cloud compression.
- Supports **ModelNet**, **ShapeNet**, and **Stanford3D** datasets.
- Compresses and decompresses **3D points (XYZ)** only.  
- *RGB attribute compression is planned for future updates.*
- Implements evaluation metrics: **PSNR**, **Chamfer Distance**, and **bitrate-per-pointcloud (bpp)**.
- Installation scripts for both **CPU** and **GPU** machines.
- Uses the produced `venv_{mode}` (e.g., `venv_gpu`, `venv_cpu`) to run all scripts.


## Requirements

- Python 3.8+
- See `requirements_cpu.txt` or `requirements_gpu.txt` for dependencies
- Datasets: ModelNet, ShapeNet, or Stanford3D (see original [IPDAE documentation](https://github.com/I2-Multimedia-Lab/IPDAE) for dataset preparation)


## Installation


Clone the repository and set up the environment:

**For GPU (default):**
```bash
bash venv_gpu_setup.sh
source venv_gpu/bin/activate
```
**For CPU:**
```bash
bash venv_cpu_setup.sh
source venv_cpu/bin/activate
```

## Data Preparation

### Option 1: Download Pre-Converted .ply Files (Recommended)

Pre-converted .ply files for ModelNet40, ShapeNet, and S3DIS are available for direct download:

- [ModelNet40 (8192 points, train/test)](https://drive.google.com/file/d/1Isa8seckZ9oNzstlE7VZcd6wVVx8LdMF/view?usp=sharing)
- [ShapeNet (2048 points, test set)](https://drive.google.com/file/d/1OzaU01kolBpfRRD0zKESYh67Hh2s2dbD/view?usp=sharing)
- [S3DIS Area 1 point clouds](https://drive.google.com/file/d/1etg29uMdV932CYmWijDD7OOupjXRKZJM/view?usp=sharing)

### Option 2: Convert Raw Datasets Manually

To prepare your own point clouds, follow the steps below for each dataset:

**ModelNet40**
1. Download ModelNet40 from [http://modelnet.cs.princeton.edu](http://modelnet.cs.princeton.edu)
2. Convert .off CAD models to .ply point clouds:
	```bash
	python sample_modelnet.py ./data/ModelNet40 ./data/ModelNet40_pc_8192 --n_point 8192
	```

**ShapeNet**
1. Download ShapeNet from [this link](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)
2. Sample point clouds:
	```bash
	python sample_shapenet.py ./data/shapenetcore_partanno_segmentation_benchmark_v0_normal ./data/ShapeNet_pc_2048 --n_point 2048
	```

**S3DIS**
1. Download S3DIS from [http://buildingparser.stanford.edu/dataset.html](http://buildingparser.stanford.edu/dataset.html)
2. Sample point clouds:
	```bash
	python sample_stanford3d.py ./data/Stanford3dDataset_v1.2_Aligned_Version/Area_1/*/*.txt ./data/Stanford3d_pc/Area_1
	```


## Usage

> **Note:** Before running any scripts, make sure your virtual environment is activated. If not already activated, use:
> ```bash
source {venv_dir}/bin/activate  # replace {venv_dir} with venv_gpu or venv_cpu
> ```

All scripts should be run from within the activated virtual environment (`venv_cpu` or `venv_gpu`).

The project uses the same input structure and commands as the original IPDAE repo. See the [IPDAE documentation](https://github.com/I2-Multimedia-Lab/IPDAE) for dataset preparation.

### 1. Training

Train the autoencoder model on the ModelNet40 training set:
```bash
python train.py './data/ModelNet40_pc_01_8192p/**/train/*.ply' './model/K256' --K 256
```

### 2. Compression

Compress point cloud test files using the trained model:
```bash
python compress.py './data/ModelNet40_pc_01_8192p/**/test/*.ply' './data/ModelNet40_K256_compressed' './model/K256' --K 256
```

### 3. Decompression

Decompress the compressed point cloud files:
```bash
python decompress.py './data/ModelNet40_K256_compressed' './data/ModelNet40_K256_decompressed' './model/K256' --K 256
```

### 4. Evaluation

Evaluate the compression results using PSNR, Chamfer distance, and bpp metrics:
```bash
python eval.py './data/ModelNet40_pc_01_8192p/**/test/*.ply' './data/ModelNet40_K256_compressed' './data/ModelNet40_K256_decompressed' './eval/ModelNet40_K256.csv' '../geo_dist/build/pc_error'
```

### 5. Visualization

Visualize evaluation metrics and save plots:
```bash
python visualize.py --csv <path_to_eval_csv> --outdir <output_directory>
```

### 6. Comparison

Compare original and reconstructed point clouds and display evaluation metrics:
```bash
python compare.py --input_dir <compressed_ply_dir> --recon_dir <decompressed_ply_dir> --csv_path <metrics_csv>
```


## (Optional) Jupyter Notebook Usage

You can also use the provided Jupyter notebook for an interactive workflow:

```bash
jupyter notebook point_cloud_compression_demo.ipynb
```

The notebook demonstrates environment setup, dataset checks, training, compression, decompression, evaluation, and visualization in a step-by-step manner.


## Notes

- For evaluation, additional metrics (PSNR, Chamfer distance, bpp) are included.
- The project currently focuses on geometry (XYZ) compression; RGB attribute compression is a planned feature.

## License

This project is for academic and research use. See the original [IPDAE license](https://github.com/I2-Multimedia-Lab/IPDAE) for upstream code.

## Citation

If you use this code, please cite the original IPDAE paper and repository.
