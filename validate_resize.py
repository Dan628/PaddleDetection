#!/usr/bin/env python
"""
Validate that resizing preserves bounding box integrity
"""

import json
import os
from PIL import Image, ImageDraw
import numpy as np
import random

def validate_annotations(original_json, resized_json, original_img_dir, resized_img_dir, num_samples=5):
    """
    Validate that bounding boxes are correctly adjusted after resize
    """

    print("Loading annotations...")
    with open(original_json, 'r') as f:
        orig_data = json.load(f)
    with open(resized_json, 'r') as f:
        resized_data = json.load(f)

    # Create image ID to annotations mapping
    orig_anns = {}
    resized_anns = {}

    for ann in orig_data['annotations']:
        img_id = ann['image_id']
        if img_id not in orig_anns:
            orig_anns[img_id] = []
        orig_anns[img_id].append(ann)

    for ann in resized_data['annotations']:
        img_id = ann['image_id']
        if img_id not in resized_anns:
            resized_anns[img_id] = []
        resized_anns[img_id].append(ann)

    # Sample random images
    sample_img_ids = random.sample(list(orig_anns.keys()), min(num_samples, len(orig_anns)))

    print(f"\nValidating {num_samples} sample images...")

    for img_id in sample_img_ids:
        # Get image info
        orig_img_info = next(img for img in orig_data['images'] if img['id'] == img_id)
        resized_img_info = next(img for img in resized_data['images'] if img['id'] == img_id)

        print(f"\nImage ID: {img_id}")
        print(f"  Original: {orig_img_info['file_name']} ({orig_img_info['width']}x{orig_img_info['height']})")
        print(f"  Resized: {resized_img_info['file_name']} ({resized_img_info['width']}x{resized_img_info['height']})")

        # Calculate expected scale
        scale_x = resized_img_info['width'] / orig_img_info['width']
        scale_y = resized_img_info['height'] / orig_img_info['height']

        # Check each annotation
        orig_boxes = orig_anns.get(img_id, [])
        resized_boxes = resized_anns.get(img_id, [])

        print(f"  Annotations: {len(orig_boxes)} original, {len(resized_boxes)} resized")

        if len(orig_boxes) != len(resized_boxes):
            print("  ⚠️ WARNING: Different number of annotations!")
            continue

        # Sort by area to match annotations
        orig_boxes.sort(key=lambda x: x['area'])
        resized_boxes.sort(key=lambda x: x['area'])

        for i, (orig, resized) in enumerate(zip(orig_boxes, resized_boxes)):
            # Check bbox scaling
            ox, oy, ow, oh = orig['bbox']
            rx, ry, rw, rh = resized['bbox']

            expected_x = ox * scale_x
            expected_y = oy * scale_y
            expected_w = ow * scale_x
            expected_h = oh * scale_y

            # Check if within tolerance (0.1 pixel)
            tolerance = 0.1
            if (abs(rx - expected_x) > tolerance or
                abs(ry - expected_y) > tolerance or
                abs(rw - expected_w) > tolerance or
                abs(rh - expected_h) > tolerance):

                print(f"  ⚠️ Box {i} mismatch!")
                print(f"    Original: [{ox:.1f}, {oy:.1f}, {ow:.1f}, {oh:.1f}]")
                print(f"    Expected: [{expected_x:.1f}, {expected_y:.1f}, {expected_w:.1f}, {expected_h:.1f}]")
                print(f"    Got:      [{rx:.1f}, {ry:.1f}, {rw:.1f}, {rh:.1f}]")
            else:
                print(f"  ✓ Box {i} correctly scaled")

    return True

def visualize_comparison(original_json, resized_json, original_img_dir, resized_img_dir, output_dir, num_samples=3):
    """
    Create side-by-side visualizations of original vs resized with bboxes
    """

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCreating visualizations in {output_dir}")

    with open(original_json, 'r') as f:
        orig_data = json.load(f)
    with open(resized_json, 'r') as f:
        resized_data = json.load(f)

    # Get category names
    categories = {cat['id']: cat['name'] for cat in orig_data['categories']}

    # Sample images
    sample_imgs = random.sample(orig_data['images'], min(num_samples, len(orig_data['images'])))

    for img_info in sample_imgs:
        img_id = img_info['id']

        # Load original image
        orig_path = os.path.join(original_img_dir, img_info['file_name'])
        if not os.path.exists(orig_path):
            print(f"Skipping {orig_path} - not found")
            continue

        orig_img = Image.open(orig_path)

        # Load resized image
        resized_img_info = next(img for img in resized_data['images'] if img['id'] == img_id)
        resized_path = os.path.join(resized_img_dir, resized_img_info['file_name'])
        resized_img = Image.open(resized_path)

        # Draw bboxes on original
        orig_draw = ImageDraw.Draw(orig_img)
        for ann in orig_data['annotations']:
            if ann['image_id'] == img_id:
                x, y, w, h = ann['bbox']
                orig_draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
                cat_name = categories.get(ann['category_id'], 'unknown')
                orig_draw.text((x, y-10), cat_name, fill='red')

        # Draw bboxes on resized
        resized_draw = ImageDraw.Draw(resized_img)
        for ann in resized_data['annotations']:
            if ann['image_id'] == img_id:
                x, y, w, h = ann['bbox']
                resized_draw.rectangle([x, y, x+w, y+h], outline='green', width=2)
                cat_name = categories.get(ann['category_id'], 'unknown')
                resized_draw.text((x, y-10), cat_name, fill='green')

        # Create side-by-side comparison
        comparison = Image.new('RGB', (orig_img.width + resized_img.width, max(orig_img.height, resized_img.height)))
        comparison.paste(orig_img, (0, 0))
        comparison.paste(resized_img, (orig_img.width, 0))

        # Save
        output_path = os.path.join(output_dir, f"comparison_{img_id}.jpg")
        comparison.save(output_path)
        print(f"  Saved: {output_path}")

    print(f"\nVisualizations saved to {output_dir}")
    print("Check these images to verify annotations are correct!")

if __name__ == "__main__":
    base_dir = "/home/ubuntu/weld-training-fs"

    # Validate training set
    print("="*60)
    print("VALIDATING TRAINING SET")
    print("="*60)

    validate_annotations(
        original_json=f"{base_dir}/coco_labels_multiscale_final/instances_train.json",
        resized_json=f"{base_dir}/coco_labels_640_fixed/instances_train.json",
        original_img_dir=f"{base_dir}/yolo_dataset_multiscale_final/images/train",
        resized_img_dir=f"{base_dir}/yolo_dataset_640_fixed/images/train",
        num_samples=10
    )

    # Create visual comparisons
    visualize_comparison(
        original_json=f"{base_dir}/coco_labels_multiscale_final/instances_train.json",
        resized_json=f"{base_dir}/coco_labels_640_fixed/instances_train.json",
        original_img_dir=f"{base_dir}/yolo_dataset_multiscale_final/images/train",
        resized_img_dir=f"{base_dir}/yolo_dataset_640_fixed/images/train",
        output_dir=f"{base_dir}/resize_validation",
        num_samples=5
    )

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("\nCheck the comparison images in:")
    print(f"{base_dir}/resize_validation/")
    print("\nIf boxes look aligned, you're good to train!")