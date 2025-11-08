#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Epoch Calculator for YOLACT Training
Helps convert epochs to iterations for precise training control
"""

def calculate_training_params(num_epochs, dataset_size, batch_size=4):
    """
    Calculate training parameters from epochs

    Args:
        num_epochs: Number of epochs you want to train
        dataset_size: Number of training images in your dataset
        batch_size: Batch size for training

    Returns:
        dict: Training parameters
    """
    iterations_per_epoch = dataset_size // batch_size
    total_iterations = num_epochs * iterations_per_epoch

    # Learning rate schedule (70%, 85%, 95% of training)
    lr_step1 = int(total_iterations * 0.7)
    lr_step2 = int(total_iterations * 0.85)
    lr_step3 = int(total_iterations * 0.95)

    return {
        'num_epochs': num_epochs,
        'dataset_size': dataset_size,
        'batch_size': batch_size,
        'iterations_per_epoch': iterations_per_epoch,
        'total_iterations': total_iterations,
        'lr_steps': (lr_step1, lr_step2, lr_step3),
        'estimated_hours': total_iterations / (dataset_size / batch_size * 3600)  # Rough estimate
    }

def print_training_config(params):
    """Print training configuration in a readable format"""
    print("=" * 60)
    print(f"🎯 TRAINING FOR {params['num_epochs']} EPOCHS")
    print("=" * 60)
    print(f"📊 Dataset Size: {params['dataset_size']} images")
    print(f"📦 Batch Size: {params['batch_size']}")
    print(f"🔄 Iterations per Epoch: {params['iterations_per_epoch']}")
    print(f"🚀 Total Iterations: {params['total_iterations']:,}")
    print(f"📈 Learning Rate Steps: {params['lr_steps']}")
    print(f"⏱️  Estimated Time: {params['estimated_hours']:.1f} hours")
    print("=" * 60)

    # Training command template
    print("🔧 TRAINING COMMAND:")
    print(f"python train.py --config=yolact_custom_{params['num_epochs']}_epochs --batch_size={params['batch_size']}")
    print("=" * 60)

if __name__ == "__main__":
    # Common epoch configurations
    configs = [
        (25, 1000, 4),   # Quick test
        (50, 1000, 4),   # Short training
        (100, 1000, 4),  # Standard training
        (200, 1000, 4),  # Full training
    ]

    print("🎯 YOLACT EPOCH CALCULATOR")
    print("Adjust dataset_size to match your actual dataset size!")
    print()

    for epochs, dataset_size, batch_size in configs:
        params = calculate_training_params(epochs, dataset_size, batch_size)
        print_training_config(params)
        print()