#!/usr/bin/env python
"""Clean debug script to check dataset configuration"""

print("=" * 60)
print("DEBUGGING DATASET CONFIGURATION ISSUE")
print("=" * 60)

# Step 1: Import dataset classes
print("\n1. Importing dataset classes...")
try:
    from ppdet.data.source import *
    from ppdet.data.source.dataset import TrainDataset, EvalDataset, TestDataset
    print("   ✓ Import successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")

# Step 2: Check registered modules
print("\n2. Checking registered modules...")
try:
    from ppdet.core.workspace import get_registered_modules
    registered = get_registered_modules()
    dataset_classes = [name for name in registered.keys() if 'Dataset' in name]
    print(f"   Found {len(dataset_classes)} dataset classes:")
    for cls in sorted(dataset_classes):
        print(f"     - {cls}")
except Exception as e:
    print(f"   ✗ Failed to get registered modules: {e}")

# Step 3: Try manual dataset creation
print("\n3. Testing manual dataset creation...")
try:
    test_config = {
        'name': 'COCODataSet',
        'dataset_dir': '/tmp',
        'image_dir': 'images',
        'anno_path': 'test.json'
    }
    dataset = TrainDataset(**test_config)
    print("   ✓ Manual TrainDataset creation successful!")
except Exception as e:
    print(f"   ✗ Manual creation failed: {e}")

# Step 4: Try using create function
print("\n4. Testing create() function...")
try:
    from ppdet.core.workspace import create
    # Try to create TrainDataset using create
    result = create('TrainDataset')
    print(f"   ✓ create('TrainDataset') returned: {type(result)}")
except Exception as e:
    print(f"   ✗ create('TrainDataset') failed: {e}")

# Step 5: Load and inspect config
print("\n5. Loading config file...")
try:
    from ppdet.core.workspace import load_config
    cfg = load_config('configs/ppyoloe/ppyoloe_plus_l_ndt_minimal.yml')

    # Check if TrainDataset is in config
    if 'TrainDataset' in cfg:
        print("   ✓ TrainDataset found in config")
        td = cfg['TrainDataset']
        print(f"     Type: {type(td)}")
        print(f"     Is dict: {isinstance(td, dict)}")
        print(f"     Has keys method: {hasattr(td, 'keys')}")
        if hasattr(td, 'keys'):
            print(f"     Keys: {list(td.keys()) if callable(td.keys) else 'not callable'}")
        print(f"     String repr: {str(td)[:100]}...")
    else:
        print("   ✗ TrainDataset NOT in config!")

except Exception as e:
    print(f"   ✗ Config loading failed: {e}")

print("\n" + "=" * 60)