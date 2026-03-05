# ======================================================================
# Flimmer Setup Script (Windows)
# ======================================================================
#
# One-command environment setup for Flimmer video LoRA training.
#
# What it does:
#   1. Creates a Python venv and installs all dependencies
#   2. Verifies GPU availability (nvidia-smi + PyTorch CUDA)
#   3. Downloads model weights (VAE, T5, and variant-specific DiT)
#
# Usage:
#   .\scripts\setup.ps1 -Variant 2.2_i2v
#   .\scripts\setup.ps1 -Variant 2.1_i2v_480p
#   .\scripts\setup.ps1 -SkipDownloads
#   .\scripts\setup.ps1 -Variant 2.2_t2v -VenvPath C:\path\to\venv
#
# Variants:
#   2.2_t2v       Wan 2.2 Text-to-Video (MoE, 2 expert DiTs)    ~69 GB total
#   2.2_i2v       Wan 2.2 Image-to-Video (MoE, 2 expert DiTs)   ~69 GB total
#   2.1_i2v_480p  Wan 2.1 Image-to-Video 480p (non-MoE, 1 DiT)  ~44 GB total
#   2.1_i2v_720p  Wan 2.1 Image-to-Video 720p (non-MoE, 1 DiT)  ~44 GB total
#
# ======================================================================

param(
    [string]$Variant,
    [switch]$SkipDownloads,
    [string]$VenvPath,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $ScriptDir

if (-not $VenvPath) {
    $VenvPath = Join-Path $RepoDir ".venv"
}

if ($Help) {
    Write-Host @"
Usage: .\scripts\setup.ps1 [OPTIONS]

Options:
  -Variant VARIANT     Which DiT model weights to download (required unless -SkipDownloads)
  -SkipDownloads       Skip weight downloads (venv + deps + GPU check only)
  -VenvPath PATH       Override venv path (default: .venv in repo root)
  -Help                Show this help message

Variants:
  2.2_t2v        Wan 2.2 Text-to-Video (MoE)    Shared: ~11.7 GB + DiT: ~57.2 GB = ~69 GB
  2.2_i2v        Wan 2.2 Image-to-Video (MoE)   Shared: ~11.7 GB + DiT: ~57.2 GB = ~69 GB
  2.1_i2v_480p   Wan 2.1 I2V 480p (non-MoE)     Shared: ~11.7 GB + DiT: ~32.8 GB = ~44 GB
  2.1_i2v_720p   Wan 2.1 I2V 720p (non-MoE)     Shared: ~11.7 GB + DiT: ~32.8 GB = ~44 GB
"@
    exit 0
}

$ValidVariants = @("2.2_t2v", "2.2_i2v", "2.1_i2v_480p", "2.1_i2v_720p")

if (-not $SkipDownloads -and -not $Variant) {
    Write-Host "ERROR: -Variant is required unless -SkipDownloads is set."
    Write-Host ""
    Write-Host "Available variants and estimated download sizes:"
    Write-Host ""
    Write-Host "  -Variant 2.2_t2v        Wan 2.2 T2V (MoE)     ~69 GB"
    Write-Host "  -Variant 2.2_i2v        Wan 2.2 I2V (MoE)     ~69 GB"
    Write-Host "  -Variant 2.1_i2v_480p   Wan 2.1 I2V 480p      ~44 GB"
    Write-Host "  -Variant 2.1_i2v_720p   Wan 2.1 I2V 720p      ~44 GB"
    Write-Host ""
    Write-Host "Or use -SkipDownloads to skip weight downloads entirely."
    exit 1
}

if ($Variant -and $Variant -notin $ValidVariants) {
    Write-Host "ERROR: Unknown variant: $Variant"
    Write-Host "Valid variants: $($ValidVariants -join ', ')"
    exit 1
}

# ── System Dependencies ─────────────────────────────────────────────

Write-Host ""
Write-Host "=== Flimmer Setup ==="
Write-Host ""

# Check ffmpeg
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($ffmpeg) {
    Write-Host "OK: ffmpeg found: $($ffmpeg.Source)"
} else {
    Write-Host "WARNING: ffmpeg not found."
    Write-Host "  Install it: winget install ffmpeg"
    Write-Host "  Or download from: https://ffmpeg.org/download.html"
    Write-Host "  Video operations will fail without ffmpeg."
}
Write-Host ""

# ── Venv Creation ───────────────────────────────────────────────────

if ($env:VIRTUAL_ENV) {
    Write-Host "WARNING: Already inside a virtual environment ($env:VIRTUAL_ENV)."
    Write-Host "  Skipping venv creation. Dependencies will be installed into the active venv."
    Write-Host ""
} else {
    Write-Host "Creating virtual environment at: $VenvPath"
    if (Test-Path $VenvPath) {
        Write-Host "  Venv already exists, reusing it."
    } else {
        python -m venv $VenvPath
        Write-Host "  Created."
    }

    $ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
    if (-not (Test-Path $ActivateScript)) {
        Write-Host "ERROR: Cannot find venv activation script at: $ActivateScript"
        exit 1
    }
    & $ActivateScript
    Write-Host "  Activated."
    Write-Host ""
}

Write-Host "Upgrading pip..."
pip install --upgrade pip
Write-Host "  Done."
Write-Host ""

# ── Dependency Installation ─────────────────────────────────────────

Write-Host "Installing flimmer with all extras..."
Write-Host "  (This includes torch, diffusers, transformers, peft, etc.)"
pip install -e "$RepoDir[all]"
Write-Host "  Done."
Write-Host ""

# ── GPU Verification ────────────────────────────────────────────────

Write-Host "=== GPU Verification ==="
Write-Host ""

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    Write-Host "FATAL: nvidia-smi not found."
    Write-Host "  Install NVIDIA drivers: https://developer.nvidia.com/cuda-downloads"
    exit 1
}

Write-Host "NVIDIA Driver and GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | ForEach-Object {
    Write-Host "  $_"
}
Write-Host ""

python -c @"
import sys, torch
if not torch.cuda.is_available():
    print('FATAL: PyTorch cannot access CUDA.')
    print(f'  PyTorch version: {torch.__version__}')
    print('  Ensure PyTorch is installed with CUDA support:')
    print('  pip install torch --index-url https://download.pytorch.org/whl/cu121')
    sys.exit(1)
print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    vram_gb = props.total_memory / (1024**3)
    print(f'  GPU {i}: {props.name} ({vram_gb:.1f} GB VRAM)')
    if vram_gb < 20:
        print(f'    WARNING: {vram_gb:.1f} GB may be insufficient for 14B models.')
        print(f'    Recommended: 24 GB+ (RTX 4090, A5000, A6000, etc.)')
"@
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "=== GPU OK ==="
Write-Host ""

# ── Weight Downloads ────────────────────────────────────────────────

function Download-Weight {
    param(
        [string]$Repo,
        [string]$RepoPath,
        [string]$DestName
    )

    $Dest = Join-Path $ModelsDir $DestName

    if (Test-Path $Dest) {
        Write-Host "  [skip] $DestName (already exists)"
        return
    }

    Write-Host "  [download] $DestName"
    $TmpCache = Join-Path $ModelsDir ".hf_tmp"

    python -c @"
from huggingface_hub import hf_hub_download
import shutil, os
tmp_cache = r'$TmpCache'
os.makedirs(tmp_cache, exist_ok=True)
cached = hf_hub_download('$Repo', '$RepoPath', cache_dir=tmp_cache)
shutil.move(cached, r'$Dest')
print(f'    Saved: $DestName')
shutil.rmtree(tmp_cache, ignore_errors=True)
"@
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path $TmpCache) { Remove-Item -Recurse -Force $TmpCache }
        Write-Host "  ERROR: Failed to download $DestName from $Repo"
        Write-Host "  Check your internet connection and HuggingFace access."
        Write-Host "  For gated models, run: huggingface-cli login"
        exit 1
    }
}

if ($SkipDownloads) {
    Write-Host "Skipping weight downloads (-SkipDownloads)."
    Write-Host ""
} else {
    $ModelsDir = Join-Path $RepoDir "models"
    if (-not (Test-Path $ModelsDir)) {
        New-Item -ItemType Directory -Path $ModelsDir | Out-Null
    }

    Write-Host "=== Weight Downloads ==="
    Write-Host ""
    Write-Host "Destination: $ModelsDir\"
    Write-Host ""
    Write-Host "Shared components (always downloaded):"
    Write-Host "  VAE:  wan_2.1_vae.safetensors          254 MB"
    Write-Host "  T5:   umt5_xxl_fp16.safetensors       11.4 GB"
    Write-Host "  Subtotal:                              ~11.7 GB"
    Write-Host ""

    switch ($Variant) {
        "2.2_t2v" {
            Write-Host "Variant: Wan 2.2 T2V (MoE - 2 expert DiTs):"
            Write-Host "  DiT High: wan2.2_t2v_high_noise_14B_fp16.safetensors  28.6 GB"
            Write-Host "  DiT Low:  wan2.2_t2v_low_noise_14B_fp16.safetensors   28.6 GB"
            Write-Host "  Total estimated download: ~69 GB"
        }
        "2.2_i2v" {
            Write-Host "Variant: Wan 2.2 I2V (MoE - 2 expert DiTs):"
            Write-Host "  DiT High: wan2.2_i2v_high_noise_14B_fp16.safetensors  28.6 GB"
            Write-Host "  DiT Low:  wan2.2_i2v_low_noise_14B_fp16.safetensors   28.6 GB"
            Write-Host "  Total estimated download: ~69 GB"
        }
        "2.1_i2v_480p" {
            Write-Host "Variant: Wan 2.1 I2V 480p (non-MoE - 1 DiT):"
            Write-Host "  DiT: wan2.1_i2v_480p_14B_fp16.safetensors             32.8 GB"
            Write-Host "  Total estimated download: ~44 GB"
        }
        "2.1_i2v_720p" {
            Write-Host "Variant: Wan 2.1 I2V 720p (non-MoE - 1 DiT):"
            Write-Host "  DiT: wan2.1_i2v_720p_14B_fp16.safetensors             32.8 GB"
            Write-Host "  Total estimated download: ~44 GB"
        }
    }
    Write-Host ""
    Write-Host "Files that already exist will be skipped."
    Write-Host ""

    # Shared components
    Write-Host "Downloading shared components..."

    Download-Weight `
        -Repo "Comfy-Org/Wan_2.1_ComfyUI_repackaged" `
        -RepoPath "split_files/vae/wan_2.1_vae.safetensors" `
        -DestName "wan_2.1_vae.safetensors"

    Download-Weight `
        -Repo "Comfy-Org/Wan_2.1_ComfyUI_repackaged" `
        -RepoPath "split_files/text_encoders/umt5_xxl_fp16.safetensors" `
        -DestName "umt5_xxl_fp16.safetensors"

    Write-Host ""

    # Variant-specific DiT weights
    Write-Host "Downloading DiT weights for variant: $Variant ..."

    switch ($Variant) {
        "2.2_t2v" {
            Download-Weight `
                -Repo "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" `
                -RepoPath "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors" `
                -DestName "wan2.2_t2v_high_noise_14B_fp16.safetensors"
            Download-Weight `
                -Repo "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" `
                -RepoPath "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors" `
                -DestName "wan2.2_t2v_low_noise_14B_fp16.safetensors"
        }
        "2.2_i2v" {
            Download-Weight `
                -Repo "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" `
                -RepoPath "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" `
                -DestName "wan2.2_i2v_high_noise_14B_fp16.safetensors"
            Download-Weight `
                -Repo "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" `
                -RepoPath "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" `
                -DestName "wan2.2_i2v_low_noise_14B_fp16.safetensors"
        }
        "2.1_i2v_480p" {
            Download-Weight `
                -Repo "Comfy-Org/Wan_2.1_ComfyUI_repackaged" `
                -RepoPath "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors" `
                -DestName "wan2.1_i2v_480p_14B_fp16.safetensors"
        }
        "2.1_i2v_720p" {
            Download-Weight `
                -Repo "Comfy-Org/Wan_2.1_ComfyUI_repackaged" `
                -RepoPath "split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors" `
                -DestName "wan2.1_i2v_720p_14B_fp16.safetensors"
        }
    }

    Write-Host ""
    Write-Host "=== Downloads Complete ==="
    Write-Host ""
}

# ── Summary ─────────────────────────────────────────────────────────

Write-Host "=== Setup Complete ==="
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Prepare your dataset (see config_templates\ for data config templates)"
Write-Host "  2. Run .\scripts\prepare.ps1 to encode latents"
Write-Host "  3. Run .\scripts\train.ps1 to start training"
Write-Host ""
Write-Host "Example configs (see config_templates\README.md for full guide):"
Write-Host "  config_templates\training\t2v_wan22.yaml        T2V training (Wan 2.2)"
Write-Host "  config_templates\training\i2v_wan21.yaml        I2V training (Wan 2.1)"
Write-Host "  config_templates\training\i2v_wan22.yaml        I2V training (Wan 2.2 MoE)"
Write-Host "  config_templates\projects\i2v_moe_phases.yaml   Multi-phase MoE workflow"
Write-Host "  config_templates\projects\t2v_phases.yaml       Multi-phase dataset workflow"
Write-Host ""
