# NDT Defect Detection Training Quick Start Guide
## Lambda Labs A100 (40GB VRAM) Training Commands

## 🚀 Initial Setup

### 1. Start tmux session (for persistent training)
```bash
# Create new tmux session
tmux new -s ndt_training

# Or attach to existing session
tmux attach -t ndt_training

# Tmux shortcuts:
# Ctrl+B, D = Detach (leave running)
# Ctrl+B, C = Create new window
# Ctrl+B, N = Next window
# Ctrl+B, P = Previous window
```

### 2. Setup Environment (first time only)
```bash
cd /home/ubuntu/PaddleDetection
chmod +x setup_training_env.sh
./setup_training_env.sh

# Login to WandB (first time only)
wandb login
```

### 3. Activate Environment (every time)
```bash
source paddle_ndt_env/bin/activate
# or
./activate_paddle_env.sh
```

### 4. Verify Dataset
```bash
# Check box distribution (recommended before training)
python tools/box_distribution.py \
  --json_path /home/ubuntu/weld-training-fs/coco_labels_multiscale_final/instances_train.json \
  --out_img ndt_distribution.jpg \
  --eval_size 640 \
  --small_stride 8
```

---

## 🎯 Training Commands by Model

### 1. **PP-YOLOE+-L** (Best Overall - START HERE)
**Memory:** ~28GB | **Batch Size:** 8-12 | **Speed:** Fast | **mAP:** High

```bash
# Standard batch size (8)
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_l_ndt_custom.yml

# Optimized for A100 40GB (batch size 12)
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_l_ndt_custom.yml \
  -o TrainReader.batch_size=12

# Multi-GPU (if using 2x A100)
python -m paddle.distributed.launch --gpus 0,1 tools/train.py \
  -c configs/ppyoloe/ppyoloe_plus_l_ndt_custom.yml \
  -o TrainReader.batch_size=8
```

---

### 2. **PP-YOLOE+-M** (Faster Training)
**Memory:** ~18GB | **Batch Size:** 12-16 | **Speed:** Faster | **mAP:** Good

```bash
# Standard batch size (8)
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_m_ndt_custom.yml

# Optimized for A100 40GB (batch size 16)
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_m_ndt_custom.yml \
  -o TrainReader.batch_size=16
```

---

### 3. **PP-YOLOE-SOD** (Small Object Optimized)
**Memory:** ~30GB | **Batch Size:** 8-10 | **Speed:** Fast | **mAP:** High for small objects

```bash
# Standard batch size (8)
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_sod_l_ndt_custom.yml

# Optimized for A100 40GB (batch size 10)
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_sod_l_ndt_custom.yml \
  -o TrainReader.batch_size=10
```

---

### 4. **PP-YOLOE-P2** (Ultra-Small Objects - High Memory!)
**Memory:** ~38GB | **Batch Size:** 4-6 | **Speed:** Slow | **mAP:** Best for tiny defects

```bash
# Standard batch size (4) - CAUTION: High memory usage
python tools/train.py -c configs/ppyoloe/ppyoloe_p2_l_ndt_custom.yml

# Maximum for A100 40GB (batch size 6)
python tools/train.py -c configs/ppyoloe/ppyoloe_p2_l_ndt_custom.yml \
  -o TrainReader.batch_size=6

# ⚠️ If OOM, reduce to batch size 4 or use gradient accumulation
```

---

### 5. **Cascade R-CNN** (Highest Precision)
**Memory:** ~35GB | **Batch Size:** 2-4 | **Speed:** Very Slow | **mAP:** Highest

```bash
# Standard batch size (2)
python tools/train.py -c configs/cascade_rcnn/cascade_rcnn_r50_vd_fpn_ssld_2x_ndt.yml

# Optimized for A100 40GB (batch size 4)
python tools/train.py -c configs/cascade_rcnn/cascade_rcnn_r50_vd_fpn_ssld_2x_ndt.yml \
  -o TrainReader.batch_size=4
```

---

## 📊 Recommended Training Order

### Phase 1: Baseline (Start Here)
```bash
# Train PP-YOLOE+-L first as baseline
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_l_ndt_custom.yml \
  -o TrainReader.batch_size=12
```

### Phase 2: Based on Results
- **If good results** → Try PP-YOLOE+-M for faster inference
- **If missing small defects** → Try PP-YOLOE-SOD
- **If still missing tiny defects** → Try PP-YOLOE-P2
- **If need maximum precision** → Try Cascade R-CNN

---

## 🔧 Advanced Options

### Enable AdamW Optimizer (Better Fine-tuning)
```bash
# Edit config to uncomment AdamW section, then:
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_l_ndt_custom.yml \
  -o OptimizerBuilder.optimizer.type=AdamW \
  -o LearningRate.base_lr=0.0001
```

### Resume Training from Checkpoint
```bash
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_l_ndt_custom.yml \
  -r output/ppyoloe_plus_l_ndt/epoch_20
```

### Mixed Precision Training (Save Memory)
```bash
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_l_ndt_custom.yml \
  --amp \
  -o TrainReader.batch_size=16  # Can use larger batch with AMP
```

### Gradient Accumulation (Simulate Larger Batch)
```bash
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_l_ndt_custom.yml \
  -o gradient_accumulation_steps=2 \
  -o TrainReader.batch_size=6  # Effective batch = 6*2 = 12
```

---

## 📈 Memory Usage Guide

| Model | Base BS | A100 40GB Optimal | 2x A100 80GB | Notes |
|-------|---------|-------------------|--------------|-------|
| PP-YOLOE+-L | 8 | 12 | 24 | Best overall |
| PP-YOLOE+-M | 8 | 16 | 32 | Fastest |
| PP-YOLOE-SOD | 8 | 10 | 20 | Small objects |
| PP-YOLOE-P2 | 4 | 6 | 12 | ⚠️ High memory |
| Cascade R-CNN | 2 | 4 | 8 | Slowest, most accurate |

---

## 🎓 Training Tips

1. **Start with PP-YOLOE+-L** at batch size 12
2. **Monitor WandB** at https://wandb.ai for live metrics
3. **Best model** saved automatically as `best_model.pdparams`
4. **Expect convergence** around epoch 40-50 with pretrained weights
5. **Use tmux** to keep training running after disconnect
6. **Check GPU usage**: `nvidia-smi -l 1` in another terminal

---

## 🔍 Evaluation Commands

```bash
# Evaluate best model
python tools/eval.py \
  -c configs/ppyoloe/ppyoloe_plus_l_ndt_custom.yml \
  -o weights=output/ppyoloe_plus_l_ndt/best_model.pdparams

# Evaluate with TTA (Test Time Augmentation)
python tools/eval.py \
  -c configs/ppyoloe/ppyoloe_plus_l_ndt_custom.yml \
  -o weights=output/ppyoloe_plus_l_ndt/best_model.pdparams \
  --classwise  # Show per-class AP
```

---

## 🚨 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
-o TrainReader.batch_size=4

# Enable mixed precision
--amp

# Use gradient accumulation
-o gradient_accumulation_steps=2
```

### Training Too Slow
```bash
# Reduce eval frequency
-o eval_interval=10

# Use smaller model (M instead of L)
# Reduce image size
-o TrainReader.sample_transforms.Resize.target_size=[512,512]
```

### Poor Convergence
```bash
# Try AdamW optimizer
# Reduce learning rate
-o LearningRate.base_lr=0.0001

# Increase warmup
-o LearningRate.schedulers[1].epochs=10
```

---

## 📞 Quick Commands Reference

```bash
# Tmux
tmux new -s ndt_training        # New session
tmux attach -t ndt_training     # Attach
Ctrl+B, D                       # Detach

# Training
python tools/train.py -c [config] -o TrainReader.batch_size=[N]

# Monitoring
nvidia-smi -l 1                 # GPU usage
tail -f output/*/train.log      # Training log
wandb login                      # Setup WandB

# Evaluation
python tools/eval.py -c [config] -o weights=[path]
```