#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# Flimmer Setup Script
# ══════════════════════════════════════════════════════════════════════
#
# One-command environment setup for Flimmer video LoRA training.
#
# What it does:
#   1. Creates a Python venv and installs all dependencies
#   2. Verifies GPU availability (nvidia-smi + PyTorch CUDA)
#   3. Downloads model weights (VAE, T5, and variant-specific DiT)
#
# Usage:
#   bash scripts/setup.sh --variant 2.2_i2v       # Full setup with Wan 2.2 I2V weights
#   bash scripts/setup.sh --variant 2.1_i2v_480p   # Full setup with Wan 2.1 I2V 480p weights
#   bash scripts/setup.sh --skip-downloads          # Setup venv + deps only, no weights
#   bash scripts/setup.sh --variant 2.2_t2v --venv /path/to/venv
#
# Variants:
#   2.2_t2v       Wan 2.2 Text-to-Video (MoE, 2 expert DiTs)    ~69 GB total
#   2.2_i2v       Wan 2.2 Image-to-Video (MoE, 2 expert DiTs)   ~69 GB total
#   2.1_i2v_480p  Wan 2.1 Image-to-Video 480p (non-MoE, 1 DiT)  ~44 GB total
#   2.1_i2v_720p  Wan 2.1 Image-to-Video 720p (non-MoE, 1 DiT)  ~44 GB total
#
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Path Resolution ─────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ── Argument Parsing ────────────────────────────────────────────────

SKIP_DOWNLOADS=false
VARIANT=""
VENV_DIR="$REPO_DIR/.venv"

usage() {
    echo "Usage: bash scripts/setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --variant VARIANT     Which DiT model weights to download (required unless --skip-downloads)"
    echo "  --skip-downloads      Skip weight downloads (venv + deps + GPU check only)"
    echo "  --venv PATH           Override venv path (default: .venv in repo root)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Variants:"
    echo "  2.2_t2v        Wan 2.2 Text-to-Video (MoE)    Shared: ~11.7 GB + DiT: ~57.2 GB = ~69 GB"
    echo "  2.2_i2v        Wan 2.2 Image-to-Video (MoE)   Shared: ~11.7 GB + DiT: ~57.2 GB = ~69 GB"
    echo "  2.1_i2v_480p   Wan 2.1 I2V 480p (non-MoE)     Shared: ~11.7 GB + DiT: ~32.8 GB = ~44 GB"
    echo "  2.1_i2v_720p   Wan 2.1 I2V 720p (non-MoE)     Shared: ~11.7 GB + DiT: ~32.8 GB = ~44 GB"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-downloads)
            SKIP_DOWNLOADS=true
            shift
            ;;
        --variant)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --variant requires a value"
                usage
                exit 1
            fi
            VARIANT="$2"
            shift 2
            ;;
        --venv)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --venv requires a path"
                usage
                exit 1
            fi
            VENV_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate variant if downloads are not skipped
if [[ "$SKIP_DOWNLOADS" = false && -z "$VARIANT" ]]; then
    echo "ERROR: --variant is required unless --skip-downloads is set."
    echo ""
    echo "Available variants and estimated download sizes:"
    echo ""
    echo "  --variant 2.2_t2v        Wan 2.2 T2V (MoE)     ~69 GB  (VAE 254 MB + T5 11.4 GB + 2x DiT 28.6 GB)"
    echo "  --variant 2.2_i2v        Wan 2.2 I2V (MoE)     ~69 GB  (VAE 254 MB + T5 11.4 GB + 2x DiT 28.6 GB)"
    echo "  --variant 2.1_i2v_480p   Wan 2.1 I2V 480p      ~44 GB  (VAE 254 MB + T5 11.4 GB + 1x DiT 32.8 GB)"
    echo "  --variant 2.1_i2v_720p   Wan 2.1 I2V 720p      ~44 GB  (VAE 254 MB + T5 11.4 GB + 1x DiT 32.8 GB)"
    echo ""
    echo "Or use --skip-downloads to skip weight downloads entirely."
    exit 1
fi

# Validate variant value
if [[ -n "$VARIANT" ]]; then
    case "$VARIANT" in
        2.2_t2v|2.2_i2v|2.1_i2v_480p|2.1_i2v_720p) ;;
        *)
            echo "ERROR: Unknown variant: $VARIANT"
            echo "Valid variants: 2.2_t2v, 2.2_i2v, 2.1_i2v_480p, 2.1_i2v_720p"
            exit 1
            ;;
    esac
fi


# ── Venv Creation ───────────────────────────────────────────────────

echo ""
echo "=== Flimmer Setup ==="
echo ""

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "WARNING: Already inside a virtual environment ($VIRTUAL_ENV)."
    echo "  Skipping venv creation. Dependencies will be installed into the active venv."
    echo ""
else
    echo "Creating virtual environment at: $VENV_DIR"
    if [[ -d "$VENV_DIR" ]]; then
        echo "  Venv already exists, reusing it."
    else
        python3 -m venv "$VENV_DIR"
        echo "  Created."
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    echo "  Activated."
    echo ""
fi

echo "Upgrading pip..."
pip install --upgrade pip
echo "  Done."
echo ""


# ── Dependency Installation ─────────────────────────────────────────

echo "Installing flimmer with all extras..."
echo "  (This includes torch, diffusers, transformers, peft, etc.)"
pip install -e "$REPO_DIR[all]"
echo "  Done."
echo ""


# ── GPU Verification ────────────────────────────────────────────────

verify_gpu() {
    echo "=== GPU Verification ==="
    echo ""

    # 1. Check nvidia-smi exists
    if ! command -v nvidia-smi &>/dev/null; then
        echo "FATAL: nvidia-smi not found."
        echo "  Install NVIDIA drivers: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi

    # 2. Driver and GPU info
    echo "NVIDIA Driver and GPU info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while IFS= read -r line; do
        echo "  $line"
    done
    echo ""

    # 3. PyTorch CUDA check
    python3 -c "
import sys
import torch

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
        print(f'    Recommended: 24 GB+ (A5000, A6000, RTX 4090, etc.)')
" || exit 1

    echo ""
    echo "=== GPU OK ==="
    echo ""
}

verify_gpu


# ── Weight Downloads ────────────────────────────────────────────────

# Download a single weight file to the flat ./models/ directory.
# Uses Python hf_hub_download() for clean single-file downloads.
# Skips files that already exist.
download_weight() {
    local repo="$1"
    local repo_path="$2"
    local dest_name="$3"
    local dest="$MODELS_DIR/$dest_name"

    if [[ -f "$dest" ]]; then
        echo "  [skip] $dest_name (already exists)"
        return 0
    fi

    echo "  [download] $dest_name"
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil
cached = hf_hub_download('$repo', '$repo_path')
shutil.copy2(cached, '$dest')
print(f'    Saved: $dest_name')
" || {
        echo "  ERROR: Failed to download $dest_name from $repo"
        echo "  Check your internet connection and HuggingFace access."
        echo "  For gated models, run: huggingface-cli login"
        return 1
    }
}

if [[ "$SKIP_DOWNLOADS" = true ]]; then
    echo "Skipping weight downloads (--skip-downloads)."
    echo ""
else
    MODELS_DIR="$REPO_DIR/models"
    mkdir -p "$MODELS_DIR"

    # Print download size estimates
    echo "=== Weight Downloads ==="
    echo ""
    echo "Destination: $MODELS_DIR/"
    echo ""
    echo "Shared components (always downloaded):"
    echo "  VAE:  wan_2.1_vae.safetensors          254 MB"
    echo "  T5:   umt5_xxl_fp16.safetensors       11.4 GB"
    echo "  Subtotal:                              ~11.7 GB"
    echo ""

    case "$VARIANT" in
        2.2_t2v)
            echo "Variant: Wan 2.2 T2V (MoE - 2 expert DiTs):"
            echo "  DiT High: wan2.2_t2v_high_noise_14B_fp16.safetensors  28.6 GB"
            echo "  DiT Low:  wan2.2_t2v_low_noise_14B_fp16.safetensors   28.6 GB"
            echo "  Subtotal:                                             ~57.2 GB"
            echo ""
            echo "Total estimated download: ~69 GB"
            ;;
        2.2_i2v)
            echo "Variant: Wan 2.2 I2V (MoE - 2 expert DiTs):"
            echo "  DiT High: wan2.2_i2v_high_noise_14B_fp16.safetensors  28.6 GB"
            echo "  DiT Low:  wan2.2_i2v_low_noise_14B_fp16.safetensors   28.6 GB"
            echo "  Subtotal:                                             ~57.2 GB"
            echo ""
            echo "Total estimated download: ~69 GB"
            ;;
        2.1_i2v_480p)
            echo "Variant: Wan 2.1 I2V 480p (non-MoE - 1 DiT):"
            echo "  DiT: wan2.1_i2v_480p_14B_fp16.safetensors             32.8 GB"
            echo ""
            echo "Total estimated download: ~44 GB"
            ;;
        2.1_i2v_720p)
            echo "Variant: Wan 2.1 I2V 720p (non-MoE - 1 DiT):"
            echo "  DiT: wan2.1_i2v_720p_14B_fp16.safetensors             32.8 GB"
            echo ""
            echo "Total estimated download: ~44 GB"
            ;;
    esac
    echo ""
    echo "Files that already exist will be skipped."
    echo ""

    # ── Shared components (all variants) ────────────────────────────

    echo "Downloading shared components..."

    download_weight \
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
        "split_files/vae/wan_2.1_vae.safetensors" \
        "wan_2.1_vae.safetensors"

    download_weight \
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
        "split_files/text_encoders/umt5_xxl_fp16.safetensors" \
        "umt5_xxl_fp16.safetensors"

    echo ""

    # ── Variant-specific DiT weights ────────────────────────────────

    echo "Downloading DiT weights for variant: $VARIANT ..."

    case "$VARIANT" in
        2.2_t2v)
            download_weight \
                "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
                "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors" \
                "wan2.2_t2v_high_noise_14B_fp16.safetensors"

            download_weight \
                "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
                "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors" \
                "wan2.2_t2v_low_noise_14B_fp16.safetensors"
            ;;

        2.2_i2v)
            download_weight \
                "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
                "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
                "wan2.2_i2v_high_noise_14B_fp16.safetensors"

            download_weight \
                "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
                "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
                "wan2.2_i2v_low_noise_14B_fp16.safetensors"
            ;;

        2.1_i2v_480p)
            download_weight \
                "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
                "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors" \
                "wan2.1_i2v_480p_14B_fp16.safetensors"
            ;;

        2.1_i2v_720p)
            download_weight \
                "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
                "split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors" \
                "wan2.1_i2v_720p_14B_fp16.safetensors"
            ;;
    esac

    echo ""
    echo "=== Downloads Complete ==="
    echo ""
fi


# ── Summary ─────────────────────────────────────────────────────────

echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Prepare your dataset (see examples/ for data config templates)"
echo "  2. Run scripts/prepare.sh to encode latents"
echo "  3. Run scripts/train.sh to start training"
echo ""
echo "Example configs (see examples/README.md for full guide):"
echo "  examples/training/t2v_wan22.yaml        T2V training (Wan 2.2)"
echo "  examples/training/i2v_wan21.yaml        I2V training (Wan 2.1)"
echo "  examples/training/i2v_wan22.yaml        I2V training (Wan 2.2 MoE)"
echo "  examples/projects/i2v_moe_phases.yaml   Multi-phase MoE workflow"
echo "  examples/projects/t2v_phases.yaml       Multi-phase dataset workflow"
echo ""
