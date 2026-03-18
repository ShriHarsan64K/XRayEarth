<div align="center">

# 🌍 XRayEarth
### Seeing through disaster with satellite vision

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green)
![License](https://img.shields.io/badge/License-MIT-purple)

*Post-disaster building damage assessment using Siamese Mask R-CNN + Focal Loss on the xBD dataset*

</div>

---

## 📌 Overview

XRayEarth addresses **extreme class imbalance** in satellite-based disaster damage assessment.  
It combines multi-temporal pre/post imagery via a **Siamese Mask R-CNN** architecture,  
optimized with **Focal Loss** to detect rare but critical instances of destroyed buildings.

| Component | Choice |
|---|---|
| Architecture | Siamese Mask R-CNN (ResNet50-FPN) |
| Fusion | Concat + Difference at FPN level |
| Loss | Focal Loss (γ=2.0, α=0.25) |
| Primary Metric | Macro F1-score |
| Optimization | AMP (FP16) + TensorRT |

---

## 🗂️ Dataset

**xBD Dataset** — [Kaggle Link](https://www.kaggle.com/datasets/qianlanzz/xbd-dataset)

Place dataset files as:
```
data/
├── pre/       ← pre-disaster images (.png)
├── post/      ← post-disaster images (.png)
└── labels/    ← JSON annotation files
```

---

## ⚙️ Setup

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/xrayearth.git
cd xrayearth
```

### 2. Environment
```bash
conda env create -f environment.yml
conda activate xrayearth
```

### 3. Configure paths
```bash
cp .env.example .env
# Edit .env with your local paths
```

### 4. Login to WandB
```bash
wandb login
```

---

## 🚀 Training

### Smoke test (Machine A — quick check)
```bash
bash scripts/smoke_test.sh
```

### Full training (Machine B)
```bash
bash scripts/train.sh v1    # Baseline
bash scripts/train.sh v10   # Full system
```

### Run all ablation variants
```bash
for v in v1 v2 v3 v4 v5 v6 v7 v8 v9 v10; do
    bash scripts/train.sh $v
done
```

---

## 📊 Ablation Study

| Version | Description |
|---|---|
| V1 | Baseline — single image, CrossEntropy |
| V2 | + Pretrained ResNet50 |
| V3 | + Deeper classifier head |
| V4 | + Data augmentation |
| V5 | + Freeze backbone |
| V6 | + Full fine-tuning |
| V7 | + Dropout |
| V8 | + GroupNorm |
| V9 | + Lighter backbone (ResNet34) |
| V10 | **Full system** — Siamese + Focal Loss + 512×512 |

---

## 🏗️ Project Structure

```
xrayearth/
├── src/           ← All Python source code
├── configs/       ← YAML configs (base + v1–v10)
├── scripts/       ← Training + export scripts
├── outputs/       ← Checkpoints, logs, predictions
├── data/          ← Dataset (local only, gitignored)
└── notebooks/     ← EDA and visualization
```

---

## 🖥️ Hardware

| Machine | GPU | Role |
|---|---|---|
| Machine A | RTX 3050 | Development + debugging |
| Machine B | RTX 5060 (8GB) | Full training + TensorRT |

---

## 📈 Tracking

All experiments logged to **WandB** under project `xrayearth`.  
Compare ablation runs via WandB parallel coordinates on `val/macro_f1`.

---

<div align="center">
Built with ❤️ for disaster response AI
</div>
