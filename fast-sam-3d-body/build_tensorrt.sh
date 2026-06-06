#!/bin/bash
# Convert all models to TensorRT format.
#
# Usage:
#   bash build_tensorrt.sh          # FP16 (default)
#   bash build_tensorrt.sh --int8   # INT8+FP16

set -e
cd "$(dirname "$0")"

INT8_FLAG=""
if [[ "$1" == "--int8" ]]; then
    INT8_FLAG="--int8"
fi

echo "============================================================"
echo " SAM 3D Body - TensorRT Engine Build${INT8_FLAG:+ (INT8+FP16)}"
echo "============================================================"
echo "Working directory: $(pwd)"
echo ""

# ============================================================
# 1. YOLO-Pose TensorRT
# ============================================================
echo "[Step 1] Converting YOLO-Pose to TensorRT..."
python convert_yolo_pose_trt.py --model ./checkpoints/yolo/yolo11m-pose.pt --imgsz 640 --half $INT8_FLAG
echo ""

# ============================================================
# 2. MoGe2 FOV Estimator TensorRT
# ============================================================
echo "[Step 2] Converting MoGe2 DINOv2 Encoder to TensorRT..."
python convert_moge_encoder_trt.py --convert_trt $INT8_FLAG
echo ""

# ============================================================
# 3. DINOv3 Backbone TensorRT
# ============================================================
echo "[Step 3] Converting DINOv3 Backbone to TensorRT..."
python convert_backbone_tensorrt.py --convert_trt $INT8_FLAG
echo ""
