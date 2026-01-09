# ISL Real-Time Translation

**Real-time Indian Sign Language to English text translation model optimized for mobile deployment.**

![Architecture](docs/architecture.png)

## 📋 Overview

This project implements a complete pipeline for translating Indian Sign Language (ISL) videos to English text. The model is specifically designed for:
- **Real-time inference** on mobile devices (Samsung Galaxy Tab S9)
- **Dense Vision Transformer** architecture for overfitting on sign language datasets
- **6000+ video training** support with capacity for fine-grained sign recognition

## 🏗️ Architecture

```
Video Frames (224×224×3, 16 frames)
        ↓
┌─────────────────────────────────────┐
│   Dense Vision Transformer (ViT)    │ ← Rich spatial feature extraction
│   24 layers, 1024 hidden dim        │ ← ~86M parameters
│   16 attention heads                │ ← Captures fine-grained details
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│   Temporal Convolutions (DENSE)     │ ← Motion modeling (4 blocks)
│   (depthwise separable + dense FFN) │ ← Captures temporal dynamics
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│   Dense Attention Pooling           │ ← Compress to 32 tokens
│   (learnable queries + refinement)  │ ← Rich token representation
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│   Transformer Decoder               │ ← Text generation
│   (4 layers, Pre-LN)                │ ← 8M parameters
└─────────────────────────────────────┘
        ↓
      Text Output
```

### Architecture Specifications
- **Backbone**: Dense Vision Transformer (ViT) - Not MobileNet!
  - 24 transformer blocks (vs 12 in standard ViT)
  - 1024 hidden dimension (vs 768 in ViT-Base)
  - 16 attention heads with 4096 FFN dimension
  - Designed for overfitting on 6000-video datasets
- **Temporal Modeling**: 4 dense temporal convolution blocks
- **Attention Pooling**: Dense with refinement layers
- **Total Parameters**: ~100M (enables fine-grained sign language modeling)
- **Optimization**: Designed for overfitting on smaller sign language datasets

## 🚀 Quick Start

### 1. Installation

```powershell
# Navigate to project directory
cd isl_realtime

# Create virtual environment (optional)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Paths

Edit `configs/config.yaml`:

```yaml
data:
  video_dir: "D:/datasets/iSign-videos"     # Your video directory
  csv_path: "D:/datasets/iSign_v1.1.csv"    # Your CSV path
```

### 3. Train

```powershell
# Train from scratch
python scripts/train.py --config configs/config.yaml

# Resume training
python scripts/train.py --config configs/config.yaml --resume checkpoints/best_model.pt
```

### 4. Evaluate

```powershell
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### 5. Demo

```powershell
# Interactive mode
python scripts/demo.py --checkpoint checkpoints/best_model.pt --mode interactive

# Live camera
python scripts/demo.py --checkpoint checkpoints/best_model.pt --mode live

# Single video
python scripts/demo.py --checkpoint checkpoints/best_model.pt --mode video --input path/to/video.mp4
```

### 6. Export for Mobile

```powershell
python scripts/export.py --checkpoint checkpoints/best_model.pt --format onnx --quantize
```

## 📁 Project Structure

```
isl_realtime/
├── configs/
│   └── config.yaml           # All hyperparameters
├── src/
│   ├── models/
│   │   ├── encoder.py        # Dense Vision Transformer + Temporal Attention
│   │   ├── decoder.py        # Transformer Decoder
│   │   └── translator.py     # Full model
│   ├── data/
│   │   ├── dataset.py        # Video dataset
│   │   └── augmentations.py  # Video augmentations
│   ├── training/
│   │   ├── trainer.py        # Training loop
│   │   └── losses.py         # CTC + CE loss
│   └── inference/
│       ├── live.py           # Real-time inference
│       └── export.py         # ONNX/TFLite export
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── demo.py               # Demo script
│   ├── export.py             # Export script
│   └── tune.py               # Hyperparameter tuning
├── requirements.txt
└── README.md
```

## 📊 Expected Results

| Epoch | Val Loss | Expected BLEU |
|-------|----------|---------------|
| 10    | ~3.0     | 5-10          |
| 30    | ~2.0     | 15-25         |
| 50    | ~1.5     | 25-35         |

**Note:** Sign Language Translation is challenging. BLEU scores of 25-35 are considered good for this task.

## 🔧 Key Features

### Training
- ✅ Mixed precision training (AMP)
- ✅ Gradient accumulation
- ✅ Backbone freezing/unfreezing
- ✅ Early stopping
- ✅ Automatic checkpointing
- ✅ Separate learning rates for encoder/decoder

### Model
- ✅ **Dense Vision Transformer** backbone (1024-dim, 24 layers, 16 heads)
- ✅ Optimized for overfitting on 6000-video datasets
- ✅ 4 deep temporal convolution blocks (motion modeling)
- ✅ Dense attention pooling with refinement layers
- ✅ Pre-LayerNorm Transformer decoder (stable training)
- ✅ CTC + CE hybrid loss

### Inference
- ✅ Greedy decoding
- ✅ Beam search
- ✅ Temperature sampling
- ✅ CTC streaming mode

### Export
- ✅ ONNX export
- ✅ INT8 quantization
- ✅ TorchScript
- 🔄 TFLite (via ONNX)

## 💡 Key Improvements from Previous Attempts

| Issue | Previous | Fixed |
|-------|----------|-------|
| Encoder output length | 1568 tokens | 32 tokens (attention pooling) |
| BOS/EOS mismatch | Inconsistent IDs | Always use 101/102 (BERT) |
| Metrics | Teacher-forcing BLEU | Autoregressive BLEU |
| Model size | ~110M params | ~16M params |
| Temporal modeling | None | Temporal convolutions |
| Training stability | Unstable | Pre-LN Transformer |

## 📱 Mobile Deployment

After training, export the model:

```powershell
# Export with INT8 quantization
python scripts/export.py --checkpoint checkpoints/best_model.pt --format onnx --quantize
```

The exported model can be used with:
- **Android**: ONNX Runtime for Android
- **iOS**: Core ML (convert from ONNX)

Expected mobile performance:
- **Model size**: ~20MB (INT8)
- **Inference time**: ~100-200ms per video segment

## 🔬 Hyperparameter Tuning

Run random search to find optimal hyperparameters:

```powershell
python scripts/tune.py --config configs/config.yaml --trials 20 --epochs 5
```

## ⚠️ Common Issues

### CUDA Out of Memory
Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Slow Training
- Enable mixed precision: `use_amp: true`
- Reduce `num_workers` if I/O bound
- Use SSD for video storage

### Poor Results
1. Check data paths in config
2. Ensure videos have consistent quality
3. Increase training epochs
4. Try hyperparameter tuning

## 📄 License

This project is for educational purposes (5th semester project).

## 🙏 Acknowledgments

- iSign dataset from HuggingFace
- Vision Transformer from timm (PyTorch Image Models)
- BERT tokenizer from HuggingFace Transformers
- Architecture optimized for sign language recognition
