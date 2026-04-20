#!/bin/bash
set -e

ENV_DIR="./.venv"
PYTHON_VERSION="3.10"

echo "==========================================================="
echo " Setting up local Conda environment in $ENV_DIR"
echo "==========================================================="

if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in the PATH."
    exit 1
fi

echo "Creating conda environment with Python $PYTHON_VERSION..."
conda create --prefix "$ENV_DIR" python="$PYTHON_VERSION" pip -y

echo "Environment created. Installing dependencies using pip inside conda..."
PIP_EXEC="$ENV_DIR/bin/pip"

# Install PyTorch with specific CUDA index as defined in pyproject.toml
echo "Installing PyTorch ecosystem..."
$PIP_EXEC install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install core and emulator/server dependencies
echo "Installing project dependencies..."
$PIP_EXEC install datasets torchao transformers bitsandbytes \
    scipy pandas numpy fastapi uvicorn requests kubernetes

echo "==========================================================="
echo " Environment setup complete! "
echo " To activate this environment, run: "
echo " conda activate $ENV_DIR "
echo "==========================================================="
