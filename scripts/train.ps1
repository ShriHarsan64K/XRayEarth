# ═══════════════════════════════════════════════════════════
#  XRayEarth — train.ps1
#  Windows PowerShell training launcher
#  Usage: .\scripts\train.ps1 v1
#         .\scripts\train.ps1 v10
# ═══════════════════════════════════════════════════════════

param(
    [string]$Version = "v1"
)

$Config = "configs\$Version.yaml"

if (-Not (Test-Path $Config)) {
    Write-Host "❌ Config not found: $Config" -ForegroundColor Red
    exit 1
}

Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  🌍 XRayEarth Training — $Version"    -ForegroundColor Cyan
Write-Host "  Config: $Config"                      -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan

python src/train.py --config $Config
