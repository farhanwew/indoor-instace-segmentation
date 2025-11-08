#!/bin/bash
# YOLACT Quick Training Commands

echo "🎯 YOLACT INDOOR SEGMENTATION - QUICK TRAINING COMMANDS"
echo "====================================================="

# Navigate to the correct directory
cd yolact

echo "📊 OPTION 1: Quick Training (50 epochs)"
echo "Time: ~2-4 hours"
echo "Command: python train.py --config=yolact_custom_50_epochs --batch_size=4 --num_workers=0"
echo ""

echo "📊 OPTION 2: Standard Training (100 epochs)"
echo "Time: ~4-8 hours"
echo "Command: python train.py --config=yolact_custom_100_epochs --batch_size=4 --num_workers=0"
echo ""

echo "📊 OPTION 3: Full Training (200 epochs)"
echo "Time: ~8-16 hours"
echo "Command: python train.py --config=yolact_custom_200_epochs --batch_size=4 --num_workers=0"
echo ""

echo "🔧 TRAINING OPTIONS:"
echo "--save_interval=2000    # Save every 2000 iterations"
echo "--resume=weights/file.pth  # Resume from checkpoint"
echo "--num_workers=0         # Use 0 to avoid multiprocessing issues"
echo ""

echo "📋 EXAMPLE START TRAINING:"
echo "python train.py --config=yolact_custom_50_epochs --batch_size=4 --num_workers=0 --save_interval=1000"
echo ""

echo "🔍 VALIDATION COMMAND:"
echo "python eval.py --trained_model=weights/yolact_custom_50_epochs_50_12500.pth --config=yolact_custom_50_epochs --score_threshold=0.15 --top_k=15"
echo ""

echo "💡 TIPS:"
echo "1. Start with 50 epochs to test setup"
echo "2. Monitor loss values during training"
echo "3. Check mask mAP after training"
echo "4. Use --resume to continue training if needed"