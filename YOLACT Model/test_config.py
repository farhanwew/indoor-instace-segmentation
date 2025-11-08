#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify YOLACT configurations work properly
"""

import sys
import os

# Add yolact to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolact'))

def test_configs():
    """Test all epoch-based configurations"""
    print("🔍 TESTING YOLACT CONFIGURATIONS")
    print("=" * 50)

    try:
        # Import the config
        from data.config import *
        print("✅ Config imported successfully")

        # Test 50 epochs config
        print("\n📊 Testing 50 epochs config...")
        print(f"   Name: {yolact_custom_50_epochs.name}")
        print(f"   Max Iter: {yolact_custom_50_epochs.max_iter}")
        print(f"   Dataset: {yolact_custom_50_epochs.dataset.name}")
        print(f"   Classes: {yolact_custom_50_epochs.num_classes}")
        print(f"   LR Steps: {yolact_custom_50_epochs.lr_steps}")
        print(f"   Train Masks: {yolact_custom_50_epochs.train_masks}")

        # Test 100 epochs config
        print("\n📊 Testing 100 epochs config...")
        print(f"   Name: {yolact_custom_100_epochs.name}")
        print(f"   Max Iter: {yolact_custom_100_epochs.max_iter}")
        print(f"   Dataset: {yolact_custom_100_epochs.dataset.name}")

        # Test 200 epochs config
        print("\n📊 Testing 200 epochs config...")
        print(f"   Name: {yolact_custom_200_epochs.name}")
        print(f"   Max Iter: {yolact_custom_200_epochs.max_iter}")
        print(f"   Dataset: {yolact_custom_200_epochs.dataset.name}")

        print("\n✅ All configurations work correctly!")
        return True

    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_training_commands():
    """Show available training commands"""
    print("\n🚀 AVAILABLE TRAINING COMMANDS")
    print("=" * 50)

    print("\n📊 50 Epochs (Quick Training - ~3 hours):")
    print("   cd yolact")
    print("   python train.py --config=yolact_custom_50_epochs --batch_size=4 --num_workers=0")

    print("\n📊 100 Epochs (Standard Training - ~6 hours):")
    print("   cd yolact")
    print("   python train.py --config=yolact_custom_100_epochs --batch_size=4 --num_workers=0")

    print("\n📊 200 Epochs (Full Training - ~12 hours):")
    print("   cd yolact")
    print("   python train.py --config=yolact_custom_200_epochs --batch_size=4 --num_workers=0")

    print("\n💡 TIPS:")
    print("   - Start with 50 epochs to test everything works")
    print("   - Use --num_workers=0 to avoid multiprocessing issues")
    print("   - Monitor training loss - it should decrease steadily")
    print("   - Check mask mAP after training (should be > 0)")

if __name__ == "__main__":
    success = test_configs()

    if success:
        show_training_commands()
        print("\n🎉 READY TO START TRAINING!")
    else:
        print("\n❌ Fix configuration errors before training")