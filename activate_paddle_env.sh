#!/bin/bash
# Quick activation script for PaddleDetection environment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/paddle_ndt_env/bin/activate"

# Configure CUDNN library path
if [ -f "/usr/lib/python3/dist-packages/torch/lib/libcudnn.so" ]; then
    export LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/torch/lib:$LD_LIBRARY_PATH
elif [ -f "/usr/local/cuda/lib64/libcudnn.so" ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

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
