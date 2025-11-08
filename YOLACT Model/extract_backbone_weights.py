#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract backbone weights from COCO-trained YOLACT models for transfer learning
This avoids the class count mismatch issue by loading only backbone weights
"""

import torch
import os
import sys

def extract_backbone_weights(coco_model_path, output_path):
    """
    Extract only backbone weights from a COCO-trained model
    """
    print(f"🔍 Loading COCO model: {coco_model_path}")

    try:
        # Load the COCO model
        coco_state_dict = torch.load(coco_model_path, map_location='cpu')

        # Extract only backbone weights (keys that don't contain prediction layers)
        backbone_weights = {}

        for key, value in coco_state_dict.items():
            # Keep only backbone-related weights
            if any(backbone_key in key for backbone_key in [
                'backbone',
                'fpn',
                'proto_net',  # Prototype network for masks
                # Exclude prediction layers, semantic segmentation, etc.
            ]) and not any(exclude_key in key for exclude_key in [
                'prediction_layers',
                'semantic_seg_conv',
                'conf_layer',  # Confidence layers with class-specific weights
                'bbox_layer',  # Bbox layers with class-specific weights
                'mask_layer',  # Mask layers with class-specific weights
            ]):
                backbone_weights[key] = value
                print(f"✅ Keeping: {key} -> {value.shape}")

        print(f"\n📊 Extracted {len(backbone_weights)} backbone layers")
        print(f"📁 Saving to: {output_path}")

        # Save the backbone weights
        torch.save(backbone_weights, output_path)
        print(f"✅ Successfully saved backbone weights!")

        return True

    except Exception as e:
        print(f"❌ Error extracting weights: {e}")
        return False

def main():
    """Main script to extract backbone weights"""
    print("🎯 YOLACT Backbone Weight Extractor")
    print("=" * 60)
    print("This script extracts only backbone weights from COCO models")
    print("to avoid class count mismatch issues during transfer learning.")
    print()

    # Check available COCO models
    weights_dir = os.path.join(os.getcwd(), 'yolact', 'weights')
    coco_models = [
        ('yolact_base_54_800000.pth', 'YOLACT Base'),
        ('yolact_im700_54_800000.pth', 'YOLACT IM700'),
        ('yolact_resnet50_54_800000.pth', 'YOLACT ResNet50'),
    ]

    available_models = []
    for filename, name in coco_models:
        filepath = os.path.join(weights_dir, filename)
        if os.path.exists(filepath):
            available_models.append((filepath, filename, name))
            print(f"✅ Found: {name} ({filename})")
        else:
            print(f"❌ Missing: {name} ({filename})")

    if not available_models:
        print("\n❌ No COCO models found!")
        print("Please download COCO weights first:")
        print("python download_coco_weights.py")
        return

    print(f"\n📦 Found {len(available_models)} COCO model(s)")

    # Extract backbone weights for each available model
    success_count = 0
    for filepath, filename, name in available_models:
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"Input: {filename}")

        # Create output filename
        backbone_filename = filename.replace('.pth', '_backbone_only.pth')
        output_path = os.path.join(weights_dir, backbone_filename)

        print(f"Output: {backbone_filename}")

        # Check if already exists
        if os.path.exists(output_path):
            print(f"⚠️  {backbone_filename} already exists, skipping...")
            success_count += 1
            continue

        # Extract backbone weights
        if extract_backbone_weights(filepath, output_path):
            success_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Extraction Summary: {success_count}/{len(available_models)} models processed")

    if success_count > 0:
        print("\n🎉 Ready for transfer learning with backbone weights!")
        print("\n🚀 Next steps:")
        print("1. Use the transfer learning configurations")
        print("2. Start training with backbone-only weights")
        print("\nExample commands:")
        print("cd yolact")
        print("python train.py --config=yolact_custom_from_base --resume=weights/yolact_base_54_800000_backbone_only.pth")

        print("\n📋 Available backbone weight files:")
        for filepath, filename, name in available_models:
            backbone_filename = filename.replace('.pth', '_backbone_only.pth')
            backbone_path = os.path.join(weights_dir, backbone_filename)
            if os.path.exists(backbone_path):
                print(f"  ✅ {backbone_filename}")
    else:
        print("\n❌ No backbone weights extracted.")

if __name__ == "__main__":
    main()