#!/usr/bin/env python3
"""
Fixed evaluation script for Lambda Labs
Handles the weights path issue properly
"""

import os
import sys
import json
import subprocess
from datetime import datetime
import shutil
import yaml

# CORRECT Configuration for your Lambda Labs setup
CONFIG = "configs/ppyoloe/ppyoloe_plus_crn_l_ndt.yml"
WEIGHTS = "output/best_model/model"  # Your actual weights location
DATASET_DIR = "/home/ubuntu/weld-training-fs/multiscale-final-coco-640s"
VAL_IMAGES = "images_s/val"
ANNOTATIONS = "instances_val.json"

# Class names in order
CLASS_NAMES = [
    'porosity', 'slag inclusion', 'crack',
    'lack of penetration', 'lack of fusion', 'undercut (at edge)',
    'tungsten inclusion', 'weld spatter', 'undercut (internal)'
]

def create_eval_config():
    """Create a temporary config file with correct weights path"""
    print("Creating temporary config with correct weights path...")

    # Read original config
    with open(CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    # Remove or update weights field if it exists
    if 'weights' in config:
        print(f"  Removing existing weights field: {config['weights']}")
        del config['weights']

    # Create temporary config
    temp_config = f"configs/ppyoloe/temp_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    print(f"  Created temporary config: {temp_config}")
    return temp_config

def run_evaluation_direct():
    """Run evaluation using direct command with all overrides"""
    print("=" * 60)
    print("PP-YOLOE+ Evaluation on NDT Dataset")
    print("=" * 60)
    print(f"Working Dir: {os.getcwd()}")
    print(f"Weights: {WEIGHTS}.pdparams")
    print(f"Dataset: {DATASET_DIR}")
    print("=" * 60)

    # Check if weights exist
    if not os.path.exists(f"{WEIGHTS}.pdparams"):
        print(f"ERROR: Weights not found at {WEIGHTS}.pdparams")
        print("\nChecking output directory structure:")
        os.system("ls -la output/")
        if os.path.exists("output/best_model"):
            print("\nFiles in output/best_model:")
            os.system("ls -la output/best_model/")
        return None

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"eval_results_{timestamp}"

    # Create temp config without weights field
    temp_config = create_eval_config()

    try:
        # Build evaluation command with explicit overrides
        cmd = [
            "python", "tools/eval.py",
            "-c", temp_config,  # Use temp config
            "-o", f"weights={WEIGHTS}",  # Override weights
            "-o", f"EvalDataset.dataset_dir={DATASET_DIR}",
            "-o", f"EvalDataset.image_dir={VAL_IMAGES}",
            "-o", f"EvalDataset.anno_path={ANNOTATIONS}",
            "-o", "use_gpu=true",
            "--output_eval", output_dir
        ]

        print("\nRunning evaluation...")
        print("Command: " + " ".join(cmd))
        print("")

        # Run evaluation
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Save output
        log_file = f"eval_log_{timestamp}.txt"
        with open(log_file, "w") as f:
            f.write("=== Command ===\n")
            f.write(" ".join(cmd) + "\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr)

        # Print output
        print(result.stdout)
        if result.stderr:
            print("\nStderr output:")
            print(result.stderr)

        # Parse results
        metrics = parse_results(result.stdout)

        print(f"\nLog saved to: {log_file}")
        print(f"Results directory: {output_dir}/")

        return metrics

    finally:
        # Clean up temp config
        if os.path.exists(temp_config):
            os.remove(temp_config)
            print(f"Cleaned up temp config: {temp_config}")

def parse_results(output):
    """Parse evaluation results from output"""
    lines = output.split('\n')

    results = {}
    metrics_found = False

    for line in lines:
        # Look for COCO evaluation results
        if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' in line:
            try:
                results['mAP'] = float(line.split('=')[-1].strip())
                metrics_found = True
            except:
                pass
        elif 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]' in line:
            try:
                results['mAP_50'] = float(line.split('=')[-1].strip())
            except:
                pass
        elif 'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]' in line:
            try:
                results['mAP_75'] = float(line.split('=')[-1].strip())
            except:
                pass
        elif 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]' in line:
            try:
                results['mAP_small'] = float(line.split('=')[-1].strip())
            except:
                pass
        elif 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]' in line:
            try:
                results['mAP_medium'] = float(line.split('=')[-1].strip())
            except:
                pass
        elif 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]' in line:
            try:
                results['mAP_large'] = float(line.split('=')[-1].strip())
            except:
                pass
        elif 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' in line:
            try:
                results['AR_100'] = float(line.split('=')[-1].strip())
            except:
                pass

    if metrics_found:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print(f"\n📊 Overall Performance:")
        print(f"  mAP@[0.5:0.95]: {results.get('mAP', 'N/A'):.3f}" if 'mAP' in results else "  mAP: Not found")
        print(f"  mAP@0.50:       {results.get('mAP_50', 'N/A'):.3f}" if 'mAP_50' in results else "  mAP@0.50: Not found")
        print(f"  mAP@0.75:       {results.get('mAP_75', 'N/A'):.3f}" if 'mAP_75' in results else "  mAP@0.75: Not found")

        if 'mAP_small' in results:
            print(f"\n📏 By Object Size:")
            print(f"  Small:  {results['mAP_small']:.3f}")
            print(f"  Medium: {results['mAP_medium']:.3f}")
            print(f"  Large:  {results['mAP_large']:.3f}")

        if 'AR_100' in results:
            print(f"\n🎯 Recall:")
            print(f"  AR@100: {results['AR_100']:.3f}")

        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'weights': f"{WEIGHTS}.pdparams",
            'dataset': DATASET_DIR,
            'results': results,
            'class_names': CLASS_NAMES
        }

        summary_file = f"eval_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📄 Summary saved to: {summary_file}")
        print("=" * 60)
    else:
        print("\n⚠️  No metrics found in output. Check the log file for errors.")

    return results

def alternative_direct_run():
    """Alternative: Run eval.py directly without subprocess"""
    print("\n🔧 Alternative: Direct Python import method")
    print("=" * 60)

    # Change to tools directory and import
    original_dir = os.getcwd()

    try:
        # Set arguments
        sys.argv = [
            'eval.py',
            '-c', CONFIG,
            '-o', f'weights={WEIGHTS}',
            '-o', f'EvalDataset.dataset_dir={DATASET_DIR}',
            '-o', f'EvalDataset.image_dir={VAL_IMAGES}',
            '-o', f'EvalDataset.anno_path={ANNOTATIONS}'
        ]

        # Import and run
        os.chdir('tools')
        import eval
        eval.main()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    print("PaddleDetection Evaluation Script (Fixed)")
    print(f"Running from: {os.getcwd()}")

    # Check environment
    if not os.path.exists("tools/eval.py"):
        print("ERROR: Not in PaddleDetection directory!")
        print("Please run from: /lambda/nfs/weld-training-fs/paddle/yoloe/PaddleDetection/")
        sys.exit(1)

    # Try main evaluation method
    metrics = run_evaluation_direct()

    if not metrics:
        print("\n❌ Evaluation failed. Checking for common issues...")
        print("\n1. Checking if weights file exists:")
        os.system(f"ls -la {WEIGHTS}*")

        print("\n2. Checking config file for weights field:")
        os.system(f"grep -n 'weights:' {CONFIG}")

        print("\n3. Try running directly with:")
        print(f"   python tools/eval.py -c {CONFIG} -o weights={WEIGHTS}")