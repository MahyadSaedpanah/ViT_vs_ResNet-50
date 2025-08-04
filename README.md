# ViT vs. ResNet for CIFAR-100 Classification

A modular PyTorch project comparing the performance of **Vision Transformer (ViT-Base-16)** and **ResNet-50** on the CIFAR-100 dataset.

---

## 📦 Structure
```
vision_project/
├── data/                # Data loading and augmentation
├── models/              # ViT and ResNet definitions
├── train/               # Training scripts
├── evaluate/            # Accuracy, speed, FLOPs
├── visualize/           # Attention & Grad-CAM
├── utils/               # Helpers (seed, save/load, etc.)
├── configs/             # config.yaml with hyperparameters
├── main.py              # Main pipeline script
└── README.md            # You are here
```

---

## 📊 Features
- ✅ Fine-tuning pretrained **ViT-Base-16** from `timm`
- ✅ Fine-tuning **ResNet-50** from `torchvision`
- ✅ Configurable training via YAML
- ✅ Modular & readable codebase
- ✅ Evaluation: accuracy, speed, model size, FLOPs
- ✅ Visualization: attention maps (ViT), Grad-CAM (ResNet)

---

## 🧪 Quick Start
```bash
# Clone and install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

---

## ⚙️ Configuration
Edit `configs/config.yaml` to change:
- learning rates
- optimizers
- batch size / epochs
- freezing ViT layers

---

## 📁 Dataset
Using CIFAR-100 (downloaded automatically).

---

## 📈 Evaluation Metrics
- Top-1 Accuracy (train / val / test)
- Training time per epoch
- Inference time per sample
- Number of parameters
- FLOPs (via `ptflops`)

---

## 📌 Visualization
- `vit_attention.py`: attention map overlay for ViT
- `resnet_gradcam.py`: Grad-CAM for CNN explanation

---
