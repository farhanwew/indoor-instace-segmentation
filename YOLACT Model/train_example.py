#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete YOLACT Training Example
Indoor Segmentation Dataset - 50 Epochs
"""

import os
import sys
import torch
import subprocess

def check_environment():
    """Check if training environment is ready"""
    print("🔍 CHECKING TRAINING ENVIRONMENT")
    print("=" * 50)

    # Check CUDA
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU Device: {torch.cuda.get_device_name()}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Check dataset
    dataset_path = "./panoptic-indoor-segmentation-5"
    if os.path.exists(dataset_path):
        print(f"✅ Dataset found: {dataset_path}")

        # Count training images
        train_dir = os.path.join(dataset_path, "train")
        if os.path.exists(train_dir):
            import glob
            image_files = glob.glob(os.path.join(train_dir, "*.jpg")) + glob.glob(os.path.join(train_dir, "*.png"))
            print(f"✅ Training images: {len(image_files)}")

        # Check annotations
        ann_file = os.path.join(train_dir, "_annotations.coco.json")
        if os.path.exists(ann_file):
            print(f"✅ Annotations found: {ann_file}")
        else:
            print(f"❌ Annotations NOT found: {ann_file}")
    else:
        print(f"❌ Dataset NOT found: {dataset_path}")

    # Check weights directory
    weights_dir = "./yolact/weights"
    if os.path.exists(weights_dir):
        print(f"✅ Weights directory: {weights_dir}")
    else:
        print(f"❌ Weights directory NOT found")

    print("=" * 50)
    return True

def show_training_config():
    """Display training configuration"""
    print("🎯 TRAINING CONFIGURATION")
    print("=" * 50)
    print("📊 Configuration: yolact_custom_50_epochs")
    print("🎯 Target Epochs: 50")
    print("📦 Batch Size: 4")
    print("📈 Learning Rate: 1e-4")
    print("🎭 Mask Training: ENABLED")
    print("🖼️ Image Size: 550x550")
    print("🔍 Anchor Scales: [24, 48, 96, 192, 384]")
    print("⚡ GPU Acceleration: ENABLED")
    print("=" * 50)

def run_training():
    """Execute the training process"""
    print("🚀 STARTING TRAINING")
    print("=" * 50)

    # Change to yolact directory
    os.chdir("./yolact")

    # Training command
    cmd = [
        "python", "train.py",
        "--config=yolact_custom_50_epochs",
        "--batch_size=4",
        "--num_workers=0",
        "--save_interval=1000"
    ]

    print("🔧 Command:", " ".join(cmd))
    print("=" * 50)

    try:
        # Start training
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 universal_newlines=True, bufsize=1)

        # Monitor training output
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())

                # Highlight important milestones
                if "Begin training!" in line:
                    print("🎉 TRAINING STARTED!")
                elif "Iteration" in line and "Loss" in line:
                    print("📊", line.rstrip())
                elif "Saving state" in line:
                    print("💾", line.rstrip())
                elif "Validation" in line:
                    print("🔍", line.rstrip())

        process.wait()

        if process.returncode == 0:
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        else:
            print(f"❌ Training failed with code: {process.returncode}")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        process.terminate()
    except Exception as e:
        print(f"❌ Training error: {e}")

def show_expected_results():
    """Show what to expect from training"""
    print("📈 EXPECTED TRAINING RESULTS")
    print("=" * 50)
    print("📊 Total Iterations: 12,500")
    print("🎯 Estimated Time: 2-4 hours (depending on GPU)")
    print("💾 Checkpoints saved every 1000 iterations")
    print("📈 LR reductions at: 8,750 | 10,625 | 11,875 iterations")
    print("🔍 Validation every 2 epochs")
    print("=" * 50)

    print("📁 OUTPUT FILES:")
    print("  🗂️  yolact/weights/yolact_custom_50_epochs_*.pth")
    print("  📊 Training logs in console")
    print("  🔍 Validation results during training")
    print("=" * 50)

def main():
    """Main training example function"""
    print("🎯 YOLACT INDOOR SEGMENTATION TRAINING EXAMPLE")
    print("=" * 60)
    print("Training for 50 epochs on custom indoor dataset")
    print("=" * 60)
    print()

    # Step 1: Check environment
    check_environment()
    print()

    # Step 2: Show configuration
    show_training_config()
    print()

    # Step 3: Show expected results
    show_expected_results()
    print()

    # Step 4: Ask user to proceed
    user_input = input("🚀 Start training? (y/n): ").lower().strip()

    if user_input == 'y' or user_input == 'yes':
        run_training()
    else:
        print("❌ Training cancelled")
        return

    print()
    print("🎉 EXAMPLE COMPLETED!")
    print("📊 Check yolact/weights/ for trained models")
    print("🔍 Use eval.py to test your trained model")

if __name__ == "__main__":
    main()