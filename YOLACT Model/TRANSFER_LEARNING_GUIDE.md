# YOLACT Transfer Learning Guide
## COCO Pre-trained Models → Custom Indoor Segmentation

This guide shows how to resume training from official COCO-pretrained YOLACT models for your custom indoor dataset.

## 🎯 Why Transfer Learning?

✅ **Better Initial Weights**: Start from models trained on 118K COCO images
✅ **Faster Convergence**: Fewer iterations needed to reach good performance
✅ **Better Masks**: Pre-trained mask heads understand object segmentation
✅ **Proven Architecture**: Uses official configs from YOLACT paper

## 📦 Available Pre-trained COCO Models

| Model | Backbone | Size | FPS | mAP | Download Link |
|-------|----------|------|-----|-----|---------------|
| `yolact_base_54_800000.pth` | ResNet101-FPN | 550px | 33.5 | 29.8 | [Download](https://huggingface.co/dbolya/yolact-base/resolve/main/yolact_base_54_800000.pth) |
| `yolact_im700_54_800000.pth` | ResNet101-FPN | 700px | 23.6 | 31.2 | [Download](https://huggingface.co/dbolya/yolact-im700/resolve/main/yolact_im700_54_800000.pth) |
| `yolact_resnet50_54_800000.pth` | ResNet50-FPN | 550px | 42.5 | 28.2 | [Download](https://huggingface.co/dbolya/yolact-resnet50/resolve/main/yolact_resnet50_54_800000.pth) |

## 🚀 Step-by-Step Transfer Learning

### Step 1: Download COCO Pre-trained Weights

```bash
cd yolact/weights

# Download YOLACT Base (Recommended starting point)
wget https://huggingface.co/dbolya/yolact-base/resolve/main/yolact_base_54_800000.pth

# OR Download YOLACT IM700 (Higher accuracy, slower)
wget https://huggingface.co/dbolya/yolact-im700/resolve/main/yolact_im700_54_800000.pth

# OR Download YOLACT ResNet50 (Faster, good baseline)
wget https://huggingface.co/dbolya/yolact-resnet50/resolve/main/yolact_resnet50_54_800000.pth
```

### Step 2: Choose Transfer Learning Configuration

#### **Option 1: Transfer from YOLACT Base (Recommended)**
- **Config**: `yolact_custom_from_base`
- **Best for**: General purpose, balanced speed/accuracy
- **Expected mAP**: 15-25 on custom dataset

```bash
cd yolact
python train.py --config=yolact_custom_from_base --resume=weights/yolact_base_54_800000.pth --batch_size=4
```

#### **Option 2: Transfer from YOLACT IM700**
- **Config**: `yolact_custom_from_im700`
- **Best for**: Higher accuracy, detailed segmentation
- **Expected mAP**: 18-28 on custom dataset

```bash
cd yolact
python train.py --config=yolact_custom_from_im700 --resume=weights/yolact_im700_54_800000.pth --batch_size=4
```

#### **Option 3: Transfer from YOLACT ResNet50**
- **Config**: `yolact_custom_from_resnet50`
- **Best for**: Faster training, good baseline
- **Expected mAP**: 12-22 on custom dataset

```bash
cd yolact
python train.py --config=yolact_custom_from_resnet50 --resume=weights/yolact_resnet50_54_800000.pth --batch_size=4
```

### Step 3: Training Parameters

All transfer configs use these optimized parameters:

```python
'lr': 1e-4                    # Standard learning rate for fine-tuning
'max_iter': 40000-50000       # Fewer iterations (starting from good weights)
'lr_steps': (25000, 32000, 36000)  # Learning rate reductions
'train_masks': True           # Enable mask training
'mask_alpha': 6.125          # Mask loss weight
'eval_mask_branch': True     # Enable mask evaluation
```

## 📊 Expected Training Progress

### **Phase 1: Adaptation (Iterations 0-10,000)**
- **Loss**: Rapid decrease as model adapts to new classes
- **Mask mAP**: From 0.00 → 5-10
- **Box mAP**: From 20-25 → 15-20 (temporary dip as adapting)

### **Phase 2: Learning (Iterations 10,000-30,000)**
- **Loss**: Steady decrease
- **Mask mAP**: 10-15 → 15-20
- **Box mAP**: Recovery to 20-25

### **Phase 3: Fine-tuning (Iterations 30,000-40,000)**
- **Loss**: Slower decrease, convergence
- **Mask mAP**: 15-20 → 20-25
- **Box mAP**: 25-30

## 🔧 Training Commands

### **Basic Training**
```bash
python train.py --config=yolact_custom_from_base --resume=weights/yolact_base_54_800000.pth
```

### **With Custom Settings**
```bash
python train.py --config=yolact_custom_from_base \
                --resume=weights/yolact_base_54_800000.pth \
                --batch_size=4 \
                --num_workers=0 \
                --save_interval=2000
```

### **Resume from Interrupted Training**
```bash
python train.py --config=yolact_custom_from_base \
                --resume=weights/yolact_custom_from_base_25_12500.pth \
                --start_iter=-1
```

### **Resume with Latest Weights**
```bash
python train.py --config=yolact_custom_from_base \
                --resume=latest
```

## 📈 Monitoring Training

### **Key Metrics to Watch:**
1. **Total Loss**: Should decrease steadily
2. **Mask Loss**: Important for fixing 0.00 mask mAP
3. **Box Loss**: Should also decrease
4. **Confidence Loss**: Classification performance

### **Good Training Indicators:**
- ✅ Loss decreases in first 1000 iterations
- ✅ No CUDA or data loading errors
- ✅ Models save successfully
- ✅ Validation shows improving mAP

### **Problem Indicators:**
- ❌ Loss stays high or increases
- ❌ Training crashes with CUDA errors
- ❌ Mask loss stays high (mask mAP will remain 0.00)
- ❌ No models saved

## 🎯 Expected Results

After 40,000-50,000 iterations of transfer learning:

```
| Model | Expected Box mAP | Expected Mask mAP | Training Time |
|-------|------------------|-------------------|---------------|
| From Base | 25-35 | 20-25 | 3-5 hours |
| From IM700 | 28-38 | 22-28 | 4-6 hours |
| From ResNet50 | 22-32 | 18-23 | 2-4 hours |
```

**vs Training from Scratch:**
- Box mAP: 15-25 (lower)
- Mask mAP: 10-18 (lower)
- Training Time: 8-12 hours (longer)

## 🔍 Validation

After training, validate your model:

```bash
python eval.py --trained_model=weights/yolact_custom_from_base_54_40000.pth \
               --config=yolact_custom_from_base \
               --score_threshold=0.15 \
               --top_k=15
```

## 💡 Pro Tips

1. **Start with YOLACT Base** - Best balance of speed and accuracy
2. **Use smaller batch size** if you get CUDA memory errors
3. **Monitor mask loss** - this is key to fixing 0.00 mask mAP
4. **Save checkpoints** every 2000 iterations for safety
5. **Validate every 10 epochs** to track progress
6. **Use --num_workers=0** to avoid multiprocessing issues

## 🚨 Common Issues & Solutions

### **Issue: CUDA Out of Memory**
```bash
# Solution: Reduce batch size
python train.py --config=yolact_custom_from_base --batch_size=2
```

### **Issue: Data Loading Errors**
```bash
# Solution: Disable multiprocessing
python train.py --config=yolact_custom_from_base --num_workers=0
```

### **Issue: Mask mAP Still 0.00**
- Check that your COCO annotations have `segmentation` data
- Verify `train_masks=True` in config
- Monitor mask loss during training

## 🎉 Success Criteria

Your transfer learning is successful when:
- ✅ Training completes without errors
- ✅ Final mask mAP > 15 (significant improvement from 0.00)
- ✅ Final box mAP > 25
- ✅ Model can segment your indoor objects correctly

Happy training! 🚀