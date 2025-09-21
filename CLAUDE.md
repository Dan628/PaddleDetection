# PaddleDetection Training Guide for Claude

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