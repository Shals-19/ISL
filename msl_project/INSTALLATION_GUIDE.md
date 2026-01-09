# Installation & Setup Guide for Dense Vision Transformer

## Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU training)
- At least 40GB GPU VRAM (A100 recommended, RTX 6000 minimum)

## Step 1: Clone/Navigate to Project
```bash
cd c:\Users\Admin\OneDrive\Documents\GitHub\msl_project
```

## Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- torch>=2.0.0
- torchvision>=0.15.0
- timm>=0.9.0 (Vision Transformer support)
- transformers>=4.30.0
- And other required packages

## Step 4: Verify Installation
```bash
python -c "import torch; import timm; print('Installation successful!')"
```

## Step 5: Configure Data Paths

Edit `configs/config.yaml`:
```yaml
data:
  video_dir: "YOUR_VIDEO_DIRECTORY"      # Path to video files
  csv_path: "YOUR_CSV_PATH"              # Path to CSV with annotations
```

## Step 6: Start Training

### Option A: Local Training (Single GPU)
```bash
python scripts/train.py --config configs/config.yaml
```

### Option B: A100 Server Training
```bash
python scripts/train.py --config configs/config_server.yaml
```

## Configuration Notes

### For Dense ViT Training:
- **Batch size**: 16 (smaller than MobileNetV3)
- **Learning rate**: 5e-5 (encoder), 1.5e-4 (decoder)
- **Epochs**: 200 (allows overfitting on 6000 videos)
- **Dropout**: 0.2 (slightly higher regularization)

### If You Get Out of Memory (OOM):
```yaml
# In config.yaml, reduce batch size:
training:
  batch_size: 8  # from 16
```

## Model Sizes
- **PyTorch state_dict**: ~380 MB
- **INT8 quantized**: ~95 MB
- **Parameters**: ~100M trainable

## Expected Training Time
- **With A100**: ~2-4 days for 200 epochs (6000 videos)
- **With V100**: ~7-10 days
- **With RTX 3090**: ~5-7 days

## GPU Memory Usage
- **Batch size 16**: ~35-38 GB (A100)
- **Batch size 32**: Would require ~70+ GB
- **Batch size 8**: ~18-20 GB (RTX 3090 compatible)

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size to 8
torch.cuda.empty_cache()
```

### Poor Convergence
- Lower learning rate: `encoder_lr: 2.5e-5`
- Check data normalization
- Verify video quality/preprocessing

### NaN Loss
- Reduce `max_grad_norm` from 0.5 to 0.3
- Lower learning rate
- Check for data issues (corrupted videos)

## Quick Model Info
```bash
python -c "
from src.models.translator import ISLTranslator
from configs.config import load_config

config = load_config('configs/config.yaml')
model = ISLTranslator(config)
print(f'Total params: {model.count_trainable_params() / 1e6:.1f}M')
print(model.get_param_count())
"
```

## Next Steps
1. Verify installation with sample video
2. Run quick_test.py to validate pipeline
3. Start training with smaller dataset first
4. Monitor training with tensorboard

## Support
For issues, check:
- ARCHITECTURE_UPDATE_SUMMARY.md (detailed architecture docs)
- README.md (general information)
- scripts/train.py (training implementation)
