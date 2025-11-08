#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download COCO Pre-trained YOLACT Models for Transfer Learning
"""

import os
import urllib.request
import sys

def download_file(url, filename, description):
    """Download a file with progress bar"""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded * 100) // total_size)
        bar_length = 50
        filled_length = (percent * bar_length) // 100
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f'\r{description}: |{bar}| {percent}% ({downloaded}/{total_size} bytes)', end='')

    try:
        print(f'\n📥 Downloading {description}...')
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f'\n✅ Successfully downloaded {filename}')
        return True
    except Exception as e:
        print(f'\n❌ Failed to download {filename}: {e}')
        return False

def main():
    """Main download script"""
    print("🎯 YOLACT COCO Pre-trained Models Downloader")
    print("=" * 60)
    print("Choose which model(s) to download for transfer learning:")
    print()

    # Available models
    models = {
        '1': {
            'name': 'YOLACT Base (Recommended)',
            'filename': 'yolact_base_54_800000.pth',
            'url': 'https://huggingface.co/dbolya/yolact-base/resolve/main/yolact_base_54_800000.pth',
            'description': 'ResNet101-FPN, 550px, 29.8 mAP COCO'
        },
        '2': {
            'name': 'YOLACT IM700 (Higher Accuracy)',
            'filename': 'yolact_im700_54_800000.pth',
            'url': 'https://huggingface.co/dbolya/yolact-im700/resolve/main/yolact_im700_54_800000.pth',
            'description': 'ResNet101-FPN, 700px, 31.2 mAP COCO'
        },
        '3': {
            'name': 'YOLACT ResNet50 (Faster)',
            'filename': 'yolact_resnet50_54_800000.pth',
            'url': 'https://huggingface.co/dbolya/yolact-resnet50/resolve/main/yolact_resnet50_54_800000.pth',
            'description': 'ResNet50-FPN, 550px, 28.2 mAP COCO'
        }
    }

    # Display options
    for key, model in models.items():
        print(f"{key}. {model['name']}")
        print(f"   📊 {model['description']}")
        print(f"   💾 {model['filename']}")
        print()

    print("4. All models")
    print("0. Exit")
    print()

    # Get user choice
    choice = input("Enter your choice (1-4): ").strip()

    if choice == '0':
        print("❌ Download cancelled")
        return
    elif choice not in ['1', '2', '3', '4']:
        print("❌ Invalid choice")
        return

    # Create weights directory
    weights_dir = os.path.join(os.getcwd(), 'yolact', 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    print(f"📁 Weights directory: {weights_dir}")

    # Download selected models
    if choice == '4':
        # Download all models
        selected = list(models.values())
        print("📦 Downloading all models...")
    else:
        # Download selected model
        selected = [models[choice]]
        print(f"📦 Downloading: {selected[0]['name']}")

    success_count = 0
    for model in selected:
        filepath = os.path.join(weights_dir, model['filename'])
        if os.path.exists(filepath):
            print(f"⚠️  {model['filename']} already exists, skipping...")
            success_count += 1
            continue

        if download_file(model['url'], filepath, model['name']):
            success_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Download Summary: {success_count}/{len(selected)} models downloaded")

    if success_count > 0:
        print("\n🎉 Ready for transfer learning!")
        print("\n🚀 Next steps:")
        print("1. Choose your transfer config:")
        print("   - yolact_custom_from_base (for yolact_base_54_800000.pth)")
        print("   - yolact_custom_from_im700 (for yolact_im700_54_800000.pth)")
        print("   - yolact_custom_from_resnet50 (for yolact_resnet50_54_800000.pth)")
        print("\n2. Start training:")
        print("   cd yolact")
        print("   python train.py --config=yolact_custom_from_base --resume=weights/yolact_base_54_800000.pth")
    else:
        print("\n❌ No models downloaded. Please check your internet connection.")

if __name__ == "__main__":
    main()