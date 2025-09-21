#!/usr/bin/env python
"""Debug script to check dataset configuration loading"""

import sys
import yaml
import pprint

# First, let's manually load the config files and see what we get
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

print("=" * 60)
print("MANUAL YAML LOADING TEST")
print("=" * 60)

# Load the ndt_detection.yml directly
ndt_config = load_yaml('configs/datasets/ndt_detection.yml')
print("\n1. ndt_detection.yml content:")
print("-" * 40)
pprint.pprint(ndt_config)

print("\n2. Checking TrainDataset in ndt_detection.yml:")
print("-" * 40)
if 'TrainDataset' in ndt_config:
    print("✓ TrainDataset found!")
    print("  Keys:", list(ndt_config['TrainDataset'].keys()))
    if 'name' in ndt_config['TrainDataset']:
        print("  ✓ 'name' field exists:", ndt_config['TrainDataset']['name'])
    else:
        print("  ✗ 'name' field MISSING!")
else:
    print("✗ TrainDataset NOT FOUND!")

print("\n" + "=" * 60)
print("PADDLEDET CONFIG LOADING TEST")
print("=" * 60)

try:
    # Now try PaddleDetection's config loading
    from ppdet.core.workspace import load_config, global_config

    cfg = load_config('configs/ppyoloe/ppyoloe_plus_l_ndt_minimal.yml')

    print("\n3. After PaddleDetection load_config:")
    print("-" * 40)

    # Check if TrainDataset is in the config
    if 'TrainDataset' in cfg:
        print("✓ TrainDataset found in loaded config!")
        print("  Type:", type(cfg['TrainDataset']))
        print("  Content:", dict(cfg['TrainDataset']) if hasattr(cfg['TrainDataset'], '__dict__') else cfg['TrainDataset'])
    else:
        print("✗ TrainDataset NOT in loaded config!")

    # Check global_config too
    print("\n4. Checking global_config:")
    print("-" * 40)
    if 'TrainDataset' in global_config:
        print("✓ TrainDataset found in global_config!")
        print("  Type:", type(global_config['TrainDataset']))
        if hasattr(global_config['TrainDataset'], 'keys'):
            print("  Keys:", list(global_config['TrainDataset'].keys()) if callable(global_config['TrainDataset'].keys) else 'N/A')
    else:
        print("✗ TrainDataset NOT in global_config!")

    # Try to see what create would receive
    print("\n5. Attempting to inspect what create() would receive:")
    print("-" * 40)
    from ppdet.core.workspace import create

    # Let's see what's registered
    from ppdet.core.workspace import serializable
    if hasattr(serializable, '__dict__'):
        registered = [k for k in serializable.__dict__.keys() if 'Dataset' in k]
        print("Registered dataset classes:", registered)

except Exception as e:
    print(f"Error during PaddleDetection loading: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("CHECKING FILE PATHS")
print("=" * 60)

import os
paths_to_check = [
    'configs/datasets/ndt_detection.yml',
    'configs/ppyoloe/ppyoloe_plus_l_ndt_minimal.yml',
    'configs/ppyoloe/_base_/ppyoloe_plus_crn.yml',
]

for path in paths_to_check:
    exists = os.path.exists(path)
    print(f"  {'✓' if exists else '✗'} {path}")