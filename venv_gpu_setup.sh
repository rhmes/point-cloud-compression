#!/bin/bash

# Exit if any command fails
set -e

# Step 1: Create and activate virtual environment
python3 -m venv venv_gpu
source venv_gpu/bin/activate

# Step 2: Upgrade pip and core tools
pip install --upgrade pip setuptools wheel

# Step 3: Install base project requirements
pip install -r requirements_gpu.txt

# Step 4: Check if torch with CUDA is installed correctly
python -c "import torch; print(f'Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Step 5: Install PyTorch3D from source (must come after torch)
sleep 3
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Step 6: Install Open3D (use pip for Linux/Windows GPU setups)
pip install open3d

echo "✅ Environment setup complete for GPU (CUDA)."
