# Phone clips (canonical into data\phone, models\phone)
# .\.venv\Scripts\snapshot.ps1 -Sounds "sounds"

# AIR IRs (canonical into data\rir_air, models\rir_air)
#.\.venv\Scripts\snapshot.ps1 -Sounds "sounds\rir_air"

# NEW
# .\scripts\snapshot.ps1 -Sounds "sounds\phone\phone_mic"

param(
  [string]$Sounds = "sounds",   # e.g., "sounds\rir_air" or just "sounds"
  [string]$Tag = ""             # optional override for folder tag
)

# --- Resolve project root (works even if you run from .venv\Scripts) ---
if ($env:VIRTUAL_ENV) { $root = Split-Path -Path $env:VIRTUAL_ENV -Parent }
else { $root = $PSScriptRoot }

# Python: prefer the local venv
$py = Join-Path $root ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }

# Tag: use the last part of the Sounds path if not supplied
if (-not $Tag -or $Tag -eq "") { $Tag = Split-Path $Sounds -Leaf }

# Canonical output locations
$canonicalData   = Join-Path $root ("data\"   + $Tag)
$canonicalModels = Join-Path $root ("models\" + $Tag)

# Timestamped backups
$stamp       = Get-Date -Format "yyyyMMdd-HHmm"
$backupData  = Join-Path $root ("backups\" + $Tag + "\data_"   + $stamp)
$backupModel = Join-Path $root ("backups\" + $Tag + "\models_" + $stamp)

# Paths to scripts and sounds (absolute)
$build = Join-Path $root "build_dataset.py"
$train = Join-Path $root "train_baseline.py"
$soundsAbs = (Resolve-Path (Join-Path $root $Sounds)).Path

# --- Build + Train (canonical) ---
& $py $build --sounds $soundsAbs --out $canonicalData
& $py $train --data   $canonicalData --out $canonicalModels

# --- Backups ---
New-Item -ItemType Directory -Force -Path $backupData   | Out-Null
New-Item -ItemType Directory -Force -Path $backupModel  | Out-Null
Copy-Item "$canonicalData\*"   $backupData  -Recurse
Copy-Item "$canonicalModels\*" $backupModel -Recurse

# --- Metrics history (with tag) ---
$mh = Join-Path $root "metrics_history.csv"
$header = 'stamp,tag,n_rows_full,n_rows_train,mae_rt60_s,f1_macro_quality,features_used'
if (!(Test-Path $mh)) { $header | Out-File $mh -Encoding utf8 }
else {
  $first = (Get-Content $mh -TotalCount 1)
  if ($first -ne $header) {
    $rest  = Get-Content $mh | Select-Object -Skip 1
    $header | Out-File $mh -Encoding utf8
    $rest   | Add-Content $mh
  }
}
$cur  = Import-Csv (Join-Path $canonicalModels "metrics.csv")
$line = "$stamp,$Tag,$($cur.n_rows_full),$($cur.n_rows_train),$($cur.mae_rt60_s),$($cur.f1_macro_quality),$($cur.features_used)"
Add-Content -Path $mh -Value $line

Write-Host "Snapshot '$stamp' complete:"
Write-Host "  from sounds -> $soundsAbs (tag: $Tag)"
Write-Host "  data        -> $canonicalData   (backup: $backupData)"
Write-Host "  models      -> $canonicalModels (backup: $backupModel)"
Write-Host "  history     -> $mh"
