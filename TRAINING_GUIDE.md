# 🖥️ XRayEarth — Machine B Training Guide
## RTX 5060 (8GB) — Full Training Workflow

---

## 📁 Real xBD Dataset Structure

Your Kaggle dataset should look like this:

```
xbd/
├── hold/
│   ├── images/    ← pre + post images together
│   ├── labels/    ← JSON annotations
│   └── masks/
├── test/
│   ├── images/
│   ├── labels/
│   └── masks/
├── tier1/
│   ├── images/
│   ├── labels/
│   └── masks/
├── tier3/
│   ├── images/
│   ├── labels/
│   └── masks/
└── train/         ← only 10 images, NOT used
    └── images/
```

**Split mapping used by XRayEarth:**

| Split | Folders Used | Purpose |
|-------|-------------|---------|
| train | tier1/ + tier3/ | Main training data |
| val   | hold/ | Validation (held-out) |
| test  | test/ | Final benchmark |

**Image naming convention:**
```
{disaster}_{id}_pre_disaster.png   ← pre image
{disaster}_{id}_post_disaster.png  ← post image
{disaster}_{id}_post_disaster.json ← annotations
```

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
.\venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate         # Linux

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

## ⚙️ Step 3 — Configure .env

```powershell
cp .env.example .env
```

Edit `.env` with your Machine B paths:

```bash
# Root xBD directory (contains tier1/, tier3/, hold/, test/)
DATA_DIR=D:\datasets\xbd

# Cache and output
CACHE_DIR=D:\datasets\xbd\tile_cache
OUTPUT_DIR=E:\XRayEarth\outputs
CHECKPOINT_DIR=E:\XRayEarth\outputs\checkpoints

# WandB
WANDB_PROJECT=xrayearth
WANDB_ENTITY=your_wandb_username

# Seed
SEED=42
```

---

## 📊 Step 4 — Setup WandB

```powershell
wandb login
# Enter API key from: https://wandb.ai/authorize
```

---

## 🔬 Step 5 — Run Self-Tests

```powershell
# Test all modules (no dataset needed)
python src/utils.py
python src/tiling.py
python src/loss.py
python src/eval.py
python src/model.py
```

All should print `✅ self-test passed!`

---

## 🧪 Step 6 — Smoke Test (2 batches only)

```powershell
python src/train.py --config configs/v1.yaml --smoke-test
```

Expected output:
```
🌍 XRayEarth
✓ GPU detected: NVIDIA GeForce RTX 5060 (8.0 GB)
✓ Config loaded: v1.yaml
✓ Model built: v1
Epoch 000 [train]: ...
✅ Smoke test passed!
```

---

## 🏋️ Step 7 — Full Ablation Training

### 10-Variant Sequential Study

| Version | Key Change | Notes |
|---------|-----------|-------|
| V1 | Baseline — no pretrain, CrossEntropy | Baseline from scratch |
| V2 | + Pretrained ResNet50 | Pretrained model |
| V3 | + Deeper classifier head | Check effect of classifier |
| V4 | + Data augmentation | Test augmentation |
| V5 | + Freeze backbone | Partial fine-tuning |
| V6 | + Full fine-tuning + optimized HP | Full model optimized |
| V7 | + Dropout | Regularization effect |
| V8 | + GroupNorm (remove BatchNorm) | Normalization effect |
| V9 | + ResNet34 (reduce depth) | Model complexity effect |
| V10 | + 512×512 tiles + Siamese + Focal Loss | Resolution + full system |

### Train one version:
```powershell
.\venv\Scripts\Activate.ps1
python src/train.py --config configs/v1.yaml
```

### Train all versions sequentially (recommended):
```powershell
.\venv\Scripts\Activate.ps1
foreach ($v in @("v1","v2","v3","v4","v5","v6","v7","v8","v9","v10")) {
    Write-Host "═══════════════════════════════"
    Write-Host "  Training $v..."
    Write-Host "═══════════════════════════════"
    python src/train.py --config configs/$v.yaml
}
```

### Resume interrupted training:
```powershell
python src/train.py --config configs/v5.yaml --resume outputs/checkpoints/v5_latest.pth
```

---

## ⏱️ Expected Training Times (RTX 5060 8GB)

| Version | Batch | Tile | Est. Time |
|---------|-------|------|-----------|
| V1  | 2 | 384 | ~2-3 hrs |
| V2  | 2 | 384 | ~2-3 hrs |
| V3  | 2 | 384 | ~2-3 hrs |
| V4  | 2 | 384 | ~3-4 hrs |
| V5  | 2 | 384 | ~1-2 hrs |
| V6  | 2 | 384 | ~3-4 hrs |
| V7  | 2 | 384 | ~3-4 hrs |
| V8  | 2 | 384 | ~3-4 hrs |
| V9  | 2 | 384 | ~2-3 hrs |
| V10 | 2 | 512 | ~4-5 hrs |
| **Total** | | | **~25-35 hrs** |

> 💡 Tip: Run overnight. Use `nohup` on Linux.

---

## 💾 Step 8 — Monitor on WandB

All runs auto-logged to WandB.

View at: `https://wandb.ai/your_username/xrayearth`

**Key metrics:**

| Metric | Description |
|--------|-------------|
| `val/macro_f1` | ⭐ PRIMARY — watch this |
| `val/m1_macro_f1` | Mode 1: per-building classification |
| `val/m2_binary_f1` | Mode 2: change detection |
| `val/mAP` | Mean average precision |
| `train/loss_total` | Total training loss |
| `train/loss_classifier` | Classification loss (Focal or CE) |
| `system/gpu_memory_mb` | GPU memory usage |

**Compare ablation runs:**
- WandB → Project `xrayearth`
- Click "Charts" → group by `config.project.version`
- Parallel coordinates plot on `val/macro_f1`

---

## 🚀 Step 9 — TensorRT Export

After training V10 (best model):

```powershell
python scripts/export_trt.py \
    --checkpoint outputs/checkpoints/v10_best.pth \
    --config     configs/v10.yaml \
    --output-dir outputs/trt \
    --benchmark
```

Expected output:
```
✓ ONNX exported: outputs/trt/v10.onnx
✓ TensorRT engine: outputs/trt/v10.engine
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric          ┃ PyTorch  ┃ TensorRT  ┃ Speedup ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
│ Mean latency ms │ 45.2     │ 12.8      │ 3.5×    │
│ Throughput FPS  │ 22.1     │ 78.1      │ 3.5×    │
└─────────────────┴──────────┴───────────┴─────────┘
```

---

## 🔧 Troubleshooting

### Out of GPU memory
```yaml
# Edit configs/base.yaml:
training:
  batch_size: 1       # reduce from 2
dataset:
  tile_size: 384      # keep at 384
  num_workers: 2      # reduce workers
```

### CUDA not found
```powershell
nvidia-smi                                    # check GPU
python -c "import torch; print(torch.version.cuda)"
```

### WandB offline mode
```powershell
$env:WANDB_MODE = "offline"
python src/train.py --config configs/v1.yaml
# Sync later: wandb sync outputs/wandb/
```

### Dataset not found error
```
RuntimeError: No images found for split 'train'
```
Check your `.env`:
```bash
# DATA_DIR must point to folder containing tier1/, tier3/, hold/, test/
DATA_DIR=D:\datasets\xbd    # ✅ correct
DATA_DIR=D:\datasets\xbd\tier1  # ❌ wrong — too deep
```

### Slow tile generation (first epoch)
Normal! Tiles are generated and cached on epoch 1.
From epoch 2 onwards, loading is fast from cache.
Cache location: `$CACHE_DIR/train/` and `$CACHE_DIR/val/`

---

## 📈 Expected Results

| Version | Key Change | Expected macro_F1 |
|---------|-----------|------------------|
| V1 | Baseline | ~0.35–0.45 |
| V2 | +Pretrained | ~0.45–0.55 |
| V3 | +Deep head | ~0.48–0.58 |
| V4 | +Augmentation | ~0.50–0.60 |
| V5 | +Freeze | ~0.45–0.55 |
| V6 | +Full fine-tune | ~0.52–0.62 |
| V7 | +Dropout | ~0.53–0.63 |
| V8 | +GroupNorm | ~0.54–0.64 |
| V9 | +ResNet34 | ~0.50–0.60 |
| **V10** | **Full system** | **~0.60–0.72** |

> Focal Loss in V10 should show significant improvement on
> rare classes (major-damage, destroyed) vs V1 CrossEntropy.

---

## 📤 Step 10 — Push Results to GitHub

After training completes:

```powershell
# Save metrics
git add outputs/logs/
git commit -m "📊 Ablation results — v1 through v10 complete"
git push origin dev

# Merge to main when all done
git checkout main
git merge dev --no-ff -m "🏆 Training complete — ablation study done"
git push origin main
```

---

*🌍 XRayEarth — Seeing through disaster with satellite vision*
