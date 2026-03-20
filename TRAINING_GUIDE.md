# 🖥️ XRayEarth — Machine B Training Guide
## RTX 5060 (8GB) — Full Training Workflow

---

## 📋 Prerequisites

- Windows 11 or Ubuntu 22.04
- NVIDIA RTX 5060 (8GB VRAM)
- CUDA 11.8 installed
- Git installed
- Python 3.10+ installed
- xBD dataset downloaded from Kaggle

---

## 🚀 Step 1 — Clone Repository

```powershell
git clone https://github.com/ShriHarsan64K/XRayEarth.git
cd XRayEarth
```

---

## 🐍 Step 2 — Setup Environment

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate    # Linux

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies
pip install -r requirements.txt

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

---

## 📁 Step 3 — Setup Dataset

Download xBD dataset from:
```
https://www.kaggle.com/datasets/qianlanzz/xbd-dataset
```

Place files as:
```
XRayEarth/
└── data/
    ├── pre/
    │   ├── hurricane-matthew_00000001_pre_disaster.png
    │   ├── hurricane-matthew_00000002_pre_disaster.png
    │   └── ...
    ├── post/
    │   ├── hurricane-matthew_00000001_post_disaster.png
    │   └── ...
    └── labels/
        ├── hurricane-matthew_00000001_post_disaster.json
        └── ...
```

---

## ⚙️ Step 4 — Configure Environment

```powershell
# Copy env template
cp .env.example .env
```

Edit `.env` with your Machine B paths:
```bash
DATA_DIR=./data
PRE_DIR=./data/pre
POST_DIR=./data/post
LABELS_DIR=./data/labels
CACHE_DIR=./data/tile_cache
OUTPUT_DIR=./outputs
CHECKPOINT_DIR=./outputs/checkpoints
WANDB_PROJECT=xrayearth
WANDB_ENTITY=your_wandb_username
SEED=42
```

---

## 📊 Step 5 — Setup WandB

```powershell
wandb login
# Enter your API key from https://wandb.ai/authorize
```

---

## 🔬 Step 6 — Smoke Test (Verify Everything Works)

```powershell
python src/train.py --config configs/v1.yaml --smoke-test
```

Expected output:
```
🌍 XRayEarth
✓ Config loaded: v1.yaml
✓ GPU detected: NVIDIA GeForce RTX 5060 (8.0 GB)
✓ Seed set to 42
✓ Model built: v1
Epoch 000 [train]: loss=X.XXXX ...
✅ Smoke test passed!
```

---

## 🏋️ Step 7 — Full Ablation Training

### Option A — Train one version at a time
```powershell
.\venv\Scripts\Activate.ps1
python src/train.py --config configs/v1.yaml
python src/train.py --config configs/v2.yaml
# ... and so on
```

### Option B — Train all versions sequentially (recommended)
```powershell
.\venv\Scripts\Activate.ps1
foreach ($v in @("v1","v2","v3","v4","v5","v6","v7","v8","v9","v10")) {
    Write-Host "Training $v..."
    python src/train.py --config configs/$v.yaml
}
```

### Option C — Use train.sh (if WSL/Git Bash available)
```bash
for v in v1 v2 v3 v4 v5 v6 v7 v8 v9 v10; do
    bash scripts/train.sh $v
done
```

---

## ⏱️ Expected Training Times (RTX 5060 8GB)

| Version | Epochs | Est. Time |
|---------|--------|-----------|
| V1  | 30 | ~2-3 hrs  |
| V2  | 30 | ~2-3 hrs  |
| V3  | 30 | ~2-3 hrs  |
| V4  | 30 | ~3-4 hrs  |
| V5  | 30 | ~1-2 hrs  |
| V6  | 30 | ~3-4 hrs  |
| V7  | 30 | ~3-4 hrs  |
| V8  | 30 | ~3-4 hrs  |
| V9  | 30 | ~2-3 hrs  |
| V10 | 30 | ~4-5 hrs  |
| **Total** | 300 | **~25-35 hrs** |

> 💡 Tip: Run overnight or use `nohup` on Linux

---

## 💾 Step 8 — Monitor Training

All runs logged to WandB automatically.

View at: `https://wandb.ai/your_username/xrayearth`

Key metrics to watch:
- `val/macro_f1` ← primary metric
- `train/loss_total`
- `train/loss_classifier`
- `system/gpu_memory_mb`

---

## 🔁 Step 9 — Resume from Checkpoint

If training is interrupted:

```powershell
python src/train.py \
    --config configs/v5.yaml \
    --resume outputs/checkpoints/v5_latest.pth
```

---

## 🚀 Step 10 — TensorRT Export (After Training)

```powershell
python scripts/export_trt.py \
    --checkpoint outputs/checkpoints/v10_best.pth \
    --config     configs/v10.yaml \
    --output-dir outputs/trt \
    --benchmark
```

Expected speedup: **3-5× faster** inference vs PyTorch

---

## 📈 Step 11 — Final Evaluation

```powershell
python src/eval.py \
    --checkpoint outputs/checkpoints/v10_best.pth \
    --config     configs/v10.yaml \
    --split      test \
    --report
```

---

## 🔧 Troubleshooting

### Out of memory (OOM)
```yaml
# In configs/base.yaml, reduce:
training:
  batch_size: 1      # from 2
dataset:
  tile_size: 384     # keep at 384, don't increase
```

### CUDA not available
```powershell
# Verify CUDA installation
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

### Slow data loading
```yaml
# In configs/base.yaml, adjust:
training:
  num_workers: 2    # reduce if RAM limited
```

### WandB offline mode
```powershell
$env:WANDB_MODE = "offline"
python src/train.py --config configs/v1.yaml
```

---

## 📊 Expected Results (Ablation Study)

| Version | Key Change | Expected macro_F1 |
|---------|-----------|------------------|
| V1  | Baseline | ~0.35-0.45 |
| V2  | +Pretrained | ~0.45-0.55 |
| V3  | +Deep head | ~0.48-0.58 |
| V4  | +Augmentation | ~0.50-0.60 |
| V5  | +Freeze backbone | ~0.45-0.55 |
| V6  | +Full fine-tune | ~0.52-0.62 |
| V7  | +Dropout | ~0.53-0.63 |
| V8  | +GroupNorm | ~0.54-0.64 |
| V9  | +ResNet34 | ~0.50-0.60 |
| V10 | **Full system** | **~0.60-0.72** |

> Note: Actual results depend on dataset split and training stability.

---

## 🌍 After Training — Push Results to GitHub

```powershell
# Save metrics summary
git add outputs/logs/
git add outputs/predictions/

git commit -m "📊 Training results — ablation v1-v10 complete"
git push origin dev
```

---

*XRayEarth — Seeing through disaster with satellite vision* 🌍
