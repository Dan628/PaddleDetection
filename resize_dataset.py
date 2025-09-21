#!/usr/bin/env python
"""
Resize images and adjust COCO annotations to uniform size
Handles mixed 640x640 and 1280x1280 images
"""

import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path

def resize_and_adjust_dataset(
    input_image_dir,
    input_json_path,
    output_image_dir,
    output_json_path,
    target_size=640,
    copy_if_same_size=True
):
    """
    Resize images and adjust COCO annotations

    Args:
        input_image_dir: Directory with original images
        input_json_path: Path to original COCO JSON
        output_image_dir: Directory for resized images
        output_json_path: Path for adjusted COCO JSON
        target_size: Target size (square images)
        copy_if_same_size: Copy instead of resize if already target size
    """

    # Create output directory
    os.makedirs(output_image_dir, exist_ok=True)

    # Load COCO annotations
    print(f"Loading annotations from {input_json_path}")
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    # Track resize stats
    stats = {'resized': 0, 'copied': 0, 'skipped': 0}

    # Process images and update image info
    print(f"\nProcessing {len(coco_data['images'])} images...")
    for img_info in tqdm(coco_data['images']):
        img_id = img_info['id']
        img_filename = img_info['file_name']

        # Handle potential subdirectories in file_name
        input_path = os.path.join(input_image_dir, img_filename)
        output_path = os.path.join(output_image_dir, os.path.basename(img_filename))

        if not os.path.exists(input_path):
            print(f"Warning: Image not found: {input_path}")
            stats['skipped'] += 1
            continue

        # Open image
        img = Image.open(input_path)
        orig_width, orig_height = img.size

        # Calculate scale factors
        scale_x = target_size / orig_width
        scale_y = target_size / orig_height

        # Check if resize needed
        if orig_width == target_size and orig_height == target_size:
            if copy_if_same_size:
                shutil.copy2(input_path, output_path)
                stats['copied'] += 1
        else:
            # Resize image
            img_resized = img.resize((target_size, target_size), Image.BILINEAR)
            img_resized.save(output_path)
            stats['resized'] += 1

        # Update image info in COCO data
        img_info['width'] = target_size
        img_info['height'] = target_size
        img_info['file_name'] = os.path.basename(img_filename)

        # Find and adjust all annotations for this image
        for ann in coco_data['annotations']:
            if ann['image_id'] == img_id:
                # Adjust bbox: [x, y, width, height]
                x, y, w, h = ann['bbox']
                ann['bbox'] = [
                    x * scale_x,
                    y * scale_y,
                    w * scale_x,
                    h * scale_y
                ]

                # Update area
                ann['area'] = ann['bbox'][2] * ann['bbox'][3]

                # Adjust segmentation if exists
                if 'segmentation' in ann and ann['segmentation']:
                    for seg_idx, seg in enumerate(ann['segmentation']):
                        if isinstance(seg, list):
                            # Polygon format: [x1, y1, x2, y2, ...]
                            adjusted_seg = []
                            for i in range(0, len(seg), 2):
                                adjusted_seg.append(seg[i] * scale_x)  # x coord
                                adjusted_seg.append(seg[i+1] * scale_y)  # y coord
                            ann['segmentation'][seg_idx] = adjusted_seg

    # Save adjusted COCO annotations
    print(f"\nSaving adjusted annotations to {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    # Print statistics
    print("\n" + "="*50)
    print("Dataset Resize Complete!")
    print("="*50)
    print(f"Images resized: {stats['resized']}")
    print(f"Images copied (already correct size): {stats['copied']}")
    print(f"Images skipped (not found): {stats['skipped']}")
    print(f"Output directory: {output_image_dir}")
    print(f"Output annotations: {output_json_path}")

    return stats

def main():
    """Process both train and val datasets"""

    base_dir = "/home/ubuntu/weld-training-fs"

    # Process training set
    print("\n" + "="*60)
    print("PROCESSING TRAINING SET")
    print("="*60)

    train_stats = resize_and_adjust_dataset(
        input_image_dir=f"{base_dir}/yolo_dataset_multiscale_final/images/train",
        input_json_path=f"{base_dir}/coco_labels_multiscale_final/instances_train.json",
        output_image_dir=f"{base_dir}/yolo_dataset_640_fixed/images/train",
        output_json_path=f"{base_dir}/coco_labels_640_fixed/instances_train.json",
        target_size=640
    )

    # Process validation set
    print("\n" + "="*60)
    print("PROCESSING VALIDATION SET")
    print("="*60)

    val_stats = resize_and_adjust_dataset(
        input_image_dir=f"{base_dir}/yolo_dataset_multiscale_final/images/val",
        input_json_path=f"{base_dir}/coco_labels_multiscale_final/instances_val.json",
        output_image_dir=f"{base_dir}/yolo_dataset_640_fixed/images/val",
        output_json_path=f"{base_dir}/coco_labels_640_fixed/instances_val.json",
        target_size=640
    )

    print("\n" + "="*60)
    print("ALL PROCESSING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Update your ndt_detection.yml to use the new paths:")
    print("   dataset_dir: /home/ubuntu/weld-training-fs/yolo_dataset_640_fixed")
    print("   anno_path for train: /home/ubuntu/weld-training-fs/coco_labels_640_fixed/instances_train.json")
    print("   anno_path for val: /home/ubuntu/weld-training-fs/coco_labels_640_fixed/instances_val.json")
    print("\n2. Run training with your config!")

if __name__ == "__main__":
    main()