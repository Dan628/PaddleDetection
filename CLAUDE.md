# PaddleDetection Training Guide for Claude

## 🎯 DESIRED TRAINING CONFIGURATION

### Dataset Requirements
- **Dataset**: NDT Defect Detection (9 classes)
- **Images**: Mixed sizes (640x640 and 1280x1280) - needs resizing
- **Location**: `/home/ubuntu/weld-training-fs/`
- **Format**: COCO JSON annotations

### Model Configurations Desired

#### 1. PP-YOLOE+-L (Primary Model)
**Optimizer:**
- Type: AdamW
- Learning rate: 0.0001
- Weight decay: 0.05
- Beta1: 0.9
- Beta2: 0.999

**Augmentations:**
- Mosaic (prob: 1.0, with MixUp enabled at 0.15 prob)
- AugmentHSV (hgain: 0.015, sgain: 0.35, vgain: 0.30)
- RandomDistort (prob: 0.2)
- RandomGaussianBlur (prob: 0.25, sigma: [0.3, 1.0])
- RandomErasing (prob: 0.15)
- RandomExpand
- RandomCrop
- RandomFlip (prob: 0.5)
- BatchRandomResize (target_size: [512, 544, 576, 608, 640, 672, 704])

**Training Settings:**
- Epochs: 80
- Batch size: 12-16 (A100 40GB optimal)
- Mixed precision: Yes (AMP)
- Snapshot interval: 5 epochs
- Eval interval: 5 epochs

#### 2. PP-YOLOE+-M
- Same optimizer settings as L
- Batch size: 16-20
- Same augmentations

#### 3. PP-YOLOE-SOD (Small Object Detection)
- Same settings, optimized for small objects
- Batch size: 8-10

#### 4. PP-YOLOE-P2 (Ultra-small objects)
- Same settings
- Batch size: 4-6 (memory intensive)

#### 5. Cascade R-CNN
- Same optimizer settings
- Batch size: 2-4
- May need different augmentations

### Monitoring
**WandB Integration:**
```yaml
use_wandb: true
wandb:
  project: NDT-Defect-Detection
  name: model_name_run
```

## 🚫 LESSONS LEARNED - WHAT NOT TO DO

### 1. Config Inheritance Issues
**Problem:** Creating custom configs with `_BASE_` inheritance caused conflicts
- Overriding sections like `TrainReader` completely replaces them, doesn't merge
- Missing required fields (like `worker_num`) causes cryptic `KeyError: 'name'` errors
- Base optimizer configs conflict with custom optimizer settings (momentum vs AdamW)

**Solution:** Edit original configs directly instead of complex inheritance

### 2. Dataset Configuration Pitfalls
**Problem:** Dataset binding breaks easily
- `TrainDataset` must have exact structure
- Don't use `!COCODataSet` YAML tags
- Dataset paths: `dataset_dir` is parent, `image_dir` is subdirectory

### 3. Mixed Image Sizes
**Problem:** 640x640 and 1280x1280 images break Mosaic augmentation
- Causes "all input arrays must have the same shape" error
- Adding Resize in wrong place doesn't fix it

**Solution:** Either:
- Pre-resize dataset to uniform size
- Remove Mosaic augmentation
- Use only batch resize operations

### 4. WandB Integration
**Problem:** WandB doesn't initialize even with config
- Needs both `use_wandb: true` AND `wandb:` section
- Must be at root level of config
- Augmentation errors can prevent WandB from initializing

### 5. Required Fields for TrainReader
When overriding TrainReader, MUST include ALL:
- `sample_transforms`
- `batch_transforms`
- `batch_size`
- `worker_num` (CRITICAL - often forgotten)
- `shuffle`
- `drop_last`
- `use_shared_memory`
- `collate_batch`

## 📝 CORRECT APPROACH

### Step 1: Copy Original Config
```bash
cp configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml configs/ppyoloe/ppyoloe_plus_l_ndt.yml
```

### Step 2: Edit Directly
1. Change dataset paths
2. Set `num_classes: 9`
3. Replace optimizer section
4. Add wandb section
5. Modify augmentations if needed

### Step 3: No Complex Inheritance
- Don't use multiple `_BASE_` configs
- Don't partially override complex sections
- Keep it simple and direct

## 🔧 Config Sections to Edit

### 1. Dataset Section
```yaml
_BASE_: ['../datasets/ndt_detection.yml', ...]  # Change first base
# OR directly include:
TrainDataset:
  name: COCODataSet
  dataset_dir: /home/ubuntu/weld-training-fs/yolo_dataset_640_fixed
  image_dir: images/train
  anno_path: /path/to/instances_train.json
```

### 2. Optimizer Section
```yaml
OptimizerBuilder:
  optimizer:
    !AdamW  # Use YAML tag for optimizer type
    weight_decay: 0.05
    beta1: 0.9
    beta2: 0.999
  regularizer: null

LearningRate:
  base_lr: 0.0001
  schedulers:
    - !CosineDecay
      max_epochs: 96
    - !LinearWarmup
      start_factor: 0.
      epochs: 5
```

### 3. Model Settings
```yaml
architecture: YOLOv3
num_classes: 9
pretrain_weights: https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_l_obj365_pretrained.pdparams
depth_mult: 1.0
width_mult: 1.0
epoch: 80
```

### 4. WandB Section
```yaml
use_wandb: true
wandb:
  project: NDT-Defect-Detection
  name: ppyoloe_plus_l_run
```

## ⚠️ CRITICAL RULES TO PREVENT BREAKING CONFIGS

### 1. Dataset Configuration Rules
- **NEVER** use `!COCODataSet` YAML tag syntax - use `name: COCODataSet` instead
- **Dataset paths structure**:
  ```yaml
  TrainDataset:
    name: COCODataSet
    dataset_dir: /base/path/to/data       # Parent directory
    image_dir: images/train                # Subdirectory for images
    anno_path: annotations.json            # Path to COCO annotations
  ```

### 2. TrainReader Override Rules
When overriding `TrainReader`, you **MUST** include ALL these fields:
```yaml
TrainReader:
  sample_transforms: [...]    # Your augmentations
  batch_transforms: [...]      # Your batch transforms
  batch_size: 8
  worker_num: 4               # CRITICAL - NEVER FORGET THIS!
  shuffle: true
  drop_last: true
  use_shared_memory: true
  collate_batch: true
```

**Missing `worker_num` will cause `KeyError: 'name'` error!**

### 3. Required Top-Level Fields
Every config must have:
- `architecture: YOLOv3` (or appropriate architecture)
- `num_classes: 9` (or your number of classes)

### 4. Config Inheritance Order Matters
Base configs load in order - later ones override earlier ones:
```yaml
_BASE_: [
  '../datasets/ndt_detection.yml',    # Dataset config first
  '../runtime.yml',
  '_base_/ppyoloe_plus_crn.yml',
  '_base_/optimizer_80e.yml',
  '_base_/ppyoloe_plus_reader.yml',   # Reader config last
]
```

## 🚀 QUICK START COMMANDS

### Environment Setup (First Time Only)
```bash
cd /home/ubuntu/PaddleDetection
./setup_training_env.sh
```

### Activate Environment (Every Session)
```bash
./activate_paddle_env.sh
# or
source paddle_ndt_env/bin/activate
export LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/torch/lib:$LD_LIBRARY_PATH
```

### Training Commands

#### Basic Training
```bash
# PP-YOLOE+-L (Best overall)
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_l_ndt_copy.yml --amp

# With batch size override
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_l_ndt_copy.yml --amp \
  -o TrainReader.batch_size=16
```

#### With AdamW Optimizer
```bash
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_l_ndt_copy.yml \
  --amp \
  -o OptimizerBuilder.optimizer.type=AdamW \
  -o OptimizerBuilder.optimizer.weight_decay=0.05 \
  -o LearningRate.base_lr=0.0001 \
  -o TrainReader.batch_size=16
```

#### With WandB Logging
```bash
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_l_ndt_copy.yml \
  --amp \
  -o use_wandb=true \
  -o wandb.project=NDT-Defect-Detection \
  -o wandb.name=run_name \
  -o TrainReader.batch_size=16
```

### Evaluation
```bash
python tools/eval.py \
  -c configs/ppyoloe/ppyoloe_plus_l_ndt_copy.yml \
  -o weights=output/ppyoloe_plus_l_ndt/best_model.pdparams
```

### Export Model
```bash
python tools/export_model.py \
  -c configs/ppyoloe/ppyoloe_plus_l_ndt_copy.yml \
  -o weights=output/ppyoloe_plus_l_ndt/best_model.pdparams \
  --output_dir=output_inference
```

## 📁 FILE LOCATIONS

### Dataset Paths (Lambda Labs)
- Images: `/home/ubuntu/weld-training-fs/yolo_dataset_multiscale_final/images/`
- Train annotations: `/home/ubuntu/weld-training-fs/coco_labels_multiscale_final/instances_train.json`
- Val annotations: `/home/ubuntu/weld-training-fs/coco_labels_multiscale_final/instances_val.json`

### Config Files
- Working base: `configs/ppyoloe/ppyoloe_plus_l_ndt_copy.yml`
- Custom with augmentations: `configs/ppyoloe/ppyoloe_plus_l_ndt_final.yml`
- Dataset config: `configs/datasets/ndt_detection.yml`

### Output
- Checkpoints: `output/ppyoloe_plus_l_ndt/`
- Best model: `output/ppyoloe_plus_l_ndt/best_model.pdparams`
- Logs: `output/ppyoloe_plus_l_ndt/train.log`

## 🐛 TROUBLESHOOTING

### KeyError: 'name'
**Cause**: Missing `worker_num` in TrainReader or wrong dataset config format
**Fix**: Ensure all 6 required TrainReader fields are present

### CUDNN Library Error
**Fix**:
```bash
export LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/torch/lib:$LD_LIBRARY_PATH
```

### Dataset Not Loading
**Check**:
1. Dataset paths are correct
2. Using `name: COCODataSet` not `!COCODataSet`
3. `dataset_dir` is parent, `image_dir` is subdirectory

### Out of Memory
**Fix**: Reduce batch size
```bash
-o TrainReader.batch_size=4
```

## 🎯 BEST PRACTICES

1. **Start with working config**: Use `ppyoloe_plus_l_ndt_copy.yml` as base
2. **Test incrementally**: Add one feature at a time
3. **Use command-line overrides**: Safer than editing configs
4. **Keep backups**: `cp config.yml config.backup.yml`
5. **Monitor training**: Use tmux/screen for persistent sessions

## 📊 MEMORY USAGE GUIDE (A100 40GB)

| Model | Batch Size | Memory Usage |
|-------|------------|--------------|
| PP-YOLOE+-L | 12 | ~28GB |
| PP-YOLOE+-M | 16 | ~18GB |
| PP-YOLOE-SOD | 10 | ~30GB |
| PP-YOLOE-P2 | 6 | ~38GB |
| Cascade R-CNN | 4 | ~35GB |

## 🔄 RESUMING TRAINING

```bash
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_l_ndt_copy.yml \
  -r output/ppyoloe_plus_l_ndt/epoch_20 \
  --amp
```

## 📝 NOTES FOR CLAUDE

When helping with this codebase:
1. Always check if `worker_num` is present when user reports training errors
2. Never suggest using `!COCODataSet` syntax - it doesn't work here
3. Remind to set CUDNN path if on Lambda Labs
4. Working config is `ppyoloe_plus_l_ndt_copy.yml` - use as reference
5. Dataset config issues are usually path structure problems