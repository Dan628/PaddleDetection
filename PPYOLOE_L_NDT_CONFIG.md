# PP-YOLOE+ L Configuration for NDT Defect Detection

## Configuration File
`configs/ppyoloe/ppyoloe_plus_crn_l_ndt.yml`

## Conservative Modifications from Base Config

### 1. Optimizer Change: SGD/Momentum → AdamW
**Reference:** `ppdet/optimizer/adamw.py`

```yaml
OptimizerBuilder:
  optimizer:
    !AdamW
    weight_decay: 0.05  # Built-in weight decay
    beta1: 0.9
    beta2: 0.999
```

**Rationale:**
- AdamW is natively supported in PaddleDetection (see `ppdet/optimizer/adamw.py`)
- Weight decay of 0.05 is standard for vision transformers and modern architectures
- No separate L2 regularizer needed as AdamW has built-in weight decay

### 2. Learning Rate Adjustment
**Reference:** `configs/ppyoloe/_base_/optimizer_80e.yml`

```yaml
LearningRate:
  base_lr: 0.001  # 10x smaller than SGD default
```

**Rationale:**
- AdamW typically requires 10x smaller LR than SGD/Momentum
- Original uses 0.01 for SGD, we use 0.001 for AdamW
- Maintains CosineDecay schedule with 5 epoch warmup

### 3. WandB Integration
**Reference:** `examples/logging.txt`

```yaml
use_wandb: true
wandb:
  project: NDT-Defect-Detection
  name: ppyoloe_plus_l_ndt
  save_dir: ./wandb_logs
```

**Verified in existing configs:**
- Same syntax used in `configs/ppyoloe/ppyoloe_plus_l_ndt_final.yml`
- Follows official documentation pattern

### 4. Full Augmentation Strategy (Dataset is now uniform 640x640)
**Reference:** `configs/ppyoloe/_base_/ppyoloe_plus_reader.yml`

Since dataset is uniform size, using full augmentation pipeline:
- **Mosaic:** 100% probability with 640x640 input, includes MixUp
- **AugmentHSV:** Color space augmentations
- **RandomDistort, RandomExpand, RandomCrop, RandomFlip:** Standard augmentations
- **BatchRandomResize:** Multi-scale training from 320 to 640
- **worker_num: 2:** Critical field for proper data loading

### 5. Dataset Configuration
**Reference:** `configs/datasets/ndt_detection.yml`

Using simplified dataset syntax without YAML tags:
```yaml
TrainDataset:
  name: COCODataSet
  dataset_dir: /path/to/dataset
  image_dir: images/train
  anno_path: instances_train.json
```

## Training Command

```bash
# Activate environment
source activate_paddle_env.sh

# Run training
python tools/train.py \
    -c configs/ppyoloe/ppyoloe_plus_crn_l_ndt.yml \
    --eval \
    --amp
```

## Key Design Decisions

1. **Conservative approach:** Minimal changes from working base config
2. **No config inheritance issues:** Clear override hierarchy
3. **References to source:** Every change backed by library code or docs
4. **WandB ready:** Proper integration following examples
5. **Flexible augmentation:** Can easily add more once dataset issues resolved

## Files Referenced

- Base config: `configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml`
- Optimizer code: `ppdet/optimizer/adamw.py`, `ppdet/optimizer/optimizer.py`
- Dataset config: `configs/datasets/ndt_detection.yml`
- WandB docs: `examples/logging.txt`
- Reader base: `configs/ppyoloe/_base_/ppyoloe_plus_reader.yml`

## Validation Checklist

- [x] AdamW optimizer properly configured
- [x] Learning rate scaled appropriately
- [x] WandB integration added
- [x] Conservative augmentations
- [x] All changes documented with references
- [x] Backup of original config created