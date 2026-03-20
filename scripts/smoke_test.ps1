# ═══════════════════════════════════════════════════════════
#  XRayEarth — smoke_test.ps1
#  Windows PowerShell smoke test
#  Usage: .\scripts\smoke_test.ps1
# ═══════════════════════════════════════════════════════════

Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  🔬 XRayEarth Smoke Test"              -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan

python src/train.py `
    --config configs/v1.yaml `
    --smoke-test `
    --max-batches 2 `
    --no-wandb

Write-Host "✅ Smoke test passed!" -ForegroundColor Green
