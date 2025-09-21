#!/usr/bin/env python
import yaml
from ppdet.core.workspace import load_config
import pprint

# Load the config
cfg = load_config('configs/ppyoloe/ppyoloe_plus_l_ndt_custom.yml')

# Check what's in the config
print("=" * 50)
print("CHECKING CONFIG STRUCTURE")
print("=" * 50)

# Check if TrainDataset exists
if 'TrainDataset' in cfg:
    print("\n✓ TrainDataset found in config:")
    pprint.pprint(dict(cfg['TrainDataset']))
else:
    print("\n✗ TrainDataset NOT FOUND in config!")

# Check if EvalDataset exists
if 'EvalDataset' in cfg:
    print("\n✓ EvalDataset found in config:")
    pprint.pprint(dict(cfg['EvalDataset']))
else:
    print("\n✗ EvalDataset NOT FOUND in config!")

# Check other critical fields
print("\n" + "=" * 50)
print("OTHER KEY FIELDS:")
print("=" * 50)
print(f"architecture: {cfg.get('architecture', 'NOT FOUND')}")
print(f"num_classes: {cfg.get('num_classes', 'NOT FOUND')}")

# Check _base_ inheritance
print("\n" + "=" * 50)
print("BASE CONFIGS:")
print("=" * 50)
if '_base_' in cfg:
    for base in cfg['_base_']:
        print(f"  - {base}")

# Check TrainReader
if 'TrainReader' in cfg:
    print("\n✓ TrainReader found")
    print(f"  batch_size: {cfg['TrainReader'].get('batch_size', 'NOT FOUND')}")
    print(f"  worker_num: {cfg['TrainReader'].get('worker_num', 'NOT FOUND')}")