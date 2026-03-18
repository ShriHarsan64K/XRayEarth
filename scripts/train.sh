#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  XRayEarth — train.sh
#  Usage: bash scripts/train.sh v1
#         bash scripts/train.sh v10
# ═══════════════════════════════════════════════════════════

set -e  # exit on error

VERSION=${1:-v1}
CONFIG="configs/${VERSION}.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "❌ Config not found: $CONFIG"
    exit 1
fi

echo "═══════════════════════════════════════"
echo "  🌍 XRayEarth Training — ${VERSION}"
echo "  Config: ${CONFIG}"
echo "═══════════════════════════════════════"

python src/train.py --config "$CONFIG"
