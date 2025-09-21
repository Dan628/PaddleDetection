#!/bin/bash

# PaddleDetection Training Environment Setup Script
# Sets up a Python virtual environment with all dependencies for NDT defect detection training

set -e  # Exit on error

echo "=========================================="
echo "PaddleDetection Training Environment Setup"
echo "=========================================="

# Check if running from PaddleDetection root
if [ ! -f "setup.py" ] || [ ! -d "ppdet" ]; then
    echo "Error: This script must be run from the PaddleDetection root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Set environment name
ENV_NAME="paddle_ndt_env"
PYTHON_VERSION="3.8"

echo ""
echo "Step 1: Checking Python version..."
if command -v python3.8 &> /dev/null; then
    echo "✓ Python 3.8 found"
    PYTHON_CMD="python3.8"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -V | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [ "$PYTHON_VERSION" = "3.8" ] || [ "$PYTHON_VERSION" = "3.9" ] || [ "$PYTHON_VERSION" = "3.10" ]; then
        echo "✓ Python $PYTHON_VERSION found"
        PYTHON_CMD="python3"
    else
        echo "Warning: Python version $PYTHON_VERSION may not be fully compatible"
        echo "Recommended: Python 3.8, 3.9, or 3.10"
        PYTHON_CMD="python3"
    fi
else
    echo "Error: Python 3 not found. Please install Python 3.8-3.10"
    exit 1
fi

echo ""
echo "Step 2: Creating virtual environment..."
if [ -d "$ENV_NAME" ]; then
    echo "Virtual environment '$ENV_NAME' already exists."
    read -p "Do you want to delete and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$ENV_NAME"
        $PYTHON_CMD -m venv "$ENV_NAME"
        echo "✓ Virtual environment recreated"
    else
        echo "Using existing environment"
    fi
else
    $PYTHON_CMD -m venv "$ENV_NAME"
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Step 3: Activating virtual environment..."
source "$ENV_NAME/bin/activate"
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Step 4: Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"

# Check for GPU/CUDA
echo ""
echo "Step 5: Checking for GPU/CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    CUDA_VERSION=$(nvidia-smi | grep -Po 'CUDA Version: \K[0-9.]+')
    echo "✓ CUDA $CUDA_VERSION detected"

    # Install PaddlePaddle GPU version based on CUDA
    echo ""
    echo "Step 6: Installing PaddlePaddle GPU version..."
    if [[ "$CUDA_VERSION" == "11."* ]]; then
        pip install paddlepaddle-gpu==2.5.2 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
    elif [[ "$CUDA_VERSION" == "12."* ]]; then
        pip install paddlepaddle-gpu==2.5.2.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
    else
        echo "Warning: CUDA version $CUDA_VERSION may require manual PaddlePaddle installation"
        pip install paddlepaddle-gpu==2.5.2 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
    fi
else
    echo "No GPU detected, installing CPU version..."
    echo ""
    echo "Step 6: Installing PaddlePaddle CPU version..."
    pip install paddlepaddle==2.5.2 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
fi
echo "✓ PaddlePaddle installed"

# Install PaddleDetection dependencies
echo ""
echo "Step 7: Installing PaddleDetection requirements..."
pip install -r requirements.txt
echo "✓ Requirements installed"

# Install PaddleDetection in development mode
echo ""
echo "Step 8: Installing PaddleDetection in development mode..."
pip install -e .
echo "✓ PaddleDetection installed"

# Install additional useful packages
echo ""
echo "Step 9: Installing additional packages..."
pip install wandb matplotlib pycocotools lap motmetrics sklearn
pip install sahi  # For small object slicing if needed
echo "✓ Additional packages installed"

# Verify installation
echo ""
echo "Step 10: Verifying installation..."
python -c "import paddle; print(f'PaddlePaddle version: {paddle.__version__}')"
python -c "import ppdet; print('PaddleDetection import: ✓')"
python -c "import wandb; print(f'WandB version: {wandb.__version__}')"

# Check if GPU is properly detected by PaddlePaddle
python -c "
import paddle
if paddle.is_compiled_with_cuda():
    print(f'GPU available: ✓')
    print(f'GPU count: {paddle.device.cuda.device_count()}')
    paddle.device.set_device('gpu:0')
    print('GPU test: ✓')
else:
    print('Running in CPU mode')
"

# Create activation script
echo ""
echo "Step 11: Creating activation helper script..."
cat > activate_paddle_env.sh << 'EOF'
#!/bin/bash
# Quick activation script for PaddleDetection environment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/paddle_ndt_env/bin/activate"
echo "✓ PaddleDetection environment activated"
echo ""
echo "Ready to train! Example commands:"
echo "  - PP-YOLOE-L:      python tools/train.py -c configs/ppyoloe/ppyoloe_plus_l_ndt_custom.yml"
echo "  - PP-YOLOE-M:      python tools/train.py -c configs/ppyoloe/ppyoloe_plus_m_ndt_custom.yml"
echo "  - PP-YOLOE-SOD:    python tools/train.py -c configs/ppyoloe/ppyoloe_plus_sod_l_ndt_custom.yml"
echo "  - PP-YOLOE-P2:     python tools/train.py -c configs/ppyoloe/ppyoloe_p2_l_ndt_custom.yml"
echo "  - Cascade R-CNN:   python tools/train.py -c configs/cascade_rcnn/cascade_rcnn_r50_vd_fpn_ssld_2x_ndt.yml"
echo ""
echo "To evaluate: python tools/eval.py -c [config] -o weights=[checkpoint]"
echo "To export:   python tools/export_model.py -c [config] -o weights=[checkpoint]"
EOF
chmod +x activate_paddle_env.sh

echo "✓ Activation helper created"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To activate this environment in the future, use one of:"
echo "  1. source paddle_ndt_env/bin/activate"
echo "  2. ./activate_paddle_env.sh"
echo ""
echo "Current environment is already activated and ready to use!"
echo ""
echo "Quick test commands:"
echo "  - Check dataset: python tools/box_distribution.py --json_path /home/ubuntu/weld-training-fs/coco_labels_multiscale_final/instances_train.json --out_img ndt_distribution.jpg"
echo "  - Start training: python tools/train.py -c configs/ppyoloe/ppyoloe_plus_l_ndt_custom.yml"
echo ""
echo "Note: Remember to login to WandB first: wandb login"
echo ""