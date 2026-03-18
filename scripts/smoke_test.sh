#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  XRayEarth — smoke_test.sh
#  Quick 2-batch test on Machine A (no full training)
#  Usage: bash scripts/smoke_test.sh
# ═══════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════"
echo "  🔬 XRayEarth Smoke Test"
echo "═══════════════════════════════════════"

python src/train.py \
    --config configs/v1.yaml \
    --smoke-test \
    --max-batches 2 \
    --no-wandb

echo "✅ Smoke test passed!"
