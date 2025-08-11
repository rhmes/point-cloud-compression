#!/bin/bash

# Exit if any command fails
set -e

# Step 1: Create and activate virtual environment
python3 -m venv venv_mac
source venv_mac/bin/activate

# Step 2: Upgrade pip and core tools
pip install --upgrade pip setuptools wheel

# Step 3: Install additional project requirements
pip install -r requirements_mac.txt

# Step 4: Ensure torch is available before installing PyTorch3D
python -c "import torch; print(f'Torch version: {torch.__version__}')"

# Step 5: Install PyTorch3D from source (must be last)
# Delay helps pip recognize torch in the environment
sleep 3
pip install open3d

pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Step 6: Install Open3D
# Note: Open3D is installed via pip for macOS compatibility
echo "✅ Environment setup complete for macOS (CPU)."
