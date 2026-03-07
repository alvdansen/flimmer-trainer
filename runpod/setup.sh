#!/bin/bash
# =============================================================================
# RunPod Setup Script for Flimmer Training
# =============================================================================
# Run this ONCE when you first create your pod, AND again after every restart
# (pod restarts wipe the container disk where Python packages live).
#
# Everything important lives on /workspace which persists across restarts.
# Python packages need reinstalling after restart, but models don't.
#
# IMPORTANT: Set Container Disk to 50 GB when creating the pod.
# The default 20 GB will run out during model downloads.
#
# Usage:
#   bash /workspace/flimmer-trainer/runpod/setup.sh --variant 2.2_i2v
#   bash /workspace/flimmer-trainer/runpod/setup.sh --variant 2.2_t2v
#   bash /workspace/flimmer-trainer/runpod/setup.sh --variant 2.1_i2v_480p
#   bash /workspace/flimmer-trainer/runpod/setup.sh --variant 2.1_i2v_720p
# =============================================================================

set -e  # Stop on any error

# --- Environment variables for training ---
# Unbuffered Python output so logs flush in real time
export PYTHONUNBUFFERED=1
# Reduce CUDA memory fragmentation for large models (14B+ params)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Argument parsing ---
VARIANT=""

usage() {
    echo "Usage: bash runpod/setup.sh --variant VARIANT"
    echo ""
    echo "Variants:"
    echo "  2.2_t2v        Wan 2.2 Text-to-Video (MoE)     ~69 GB"
    echo "  2.2_i2v        Wan 2.2 Image-to-Video (MoE)    ~69 GB"
    echo "  2.1_i2v_480p   Wan 2.1 I2V 480p (non-MoE)      ~44 GB"
    echo "  2.1_i2v_720p   Wan 2.1 I2V 720p (non-MoE)      ~44 GB"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --variant requires a value"
                usage
                exit 1
            fi
            VARIANT="$2"
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

if [[ -z "$VARIANT" ]]; then
    echo "ERROR: --variant is required."
    echo ""
    usage
    exit 1
fi

case "$VARIANT" in
    2.2_t2v|2.2_i2v|2.1_i2v_480p|2.1_i2v_720p) ;;
    *)
        echo "ERROR: Unknown variant: $VARIANT"
        echo "Valid variants: 2.2_t2v, 2.2_i2v, 2.1_i2v_480p, 2.1_i2v_720p"
        exit 1
        ;;
esac

echo "=============================================="
echo "  Flimmer — RunPod Setup"
echo "=============================================="

# --- 1. System packages ---
echo ""
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq ffmpeg tmux > /dev/null 2>&1
echo "  Done: ffmpeg, tmux"

# --- 2. Clone and install flimmer ---
echo ""
echo "[2/6] Setting up flimmer..."
if [ -d "/workspace/flimmer-trainer" ]; then
    echo "  flimmer-trainer already exists — pulling latest..."
    cd /workspace/flimmer-trainer
    git pull
else
    echo "  Cloning flimmer-trainer..."
    cd /workspace
    git clone https://github.com/alvdansen/flimmer-trainer.git
    cd /workspace/flimmer-trainer
fi

# Install flimmer with all dependencies.
# NOTE: These live on the container disk, so they need reinstalling after restart.
pip install -q -e ".[all]" 2>/dev/null || pip install -q -e ".[training]" 2>/dev/null || pip install -q -e . 2>/dev/null
pip install -q huggingface_hub hf_transfer
# bitsandbytes is required for adamw8bit optimizer
pip install -q bitsandbytes 2>/dev/null || true
echo "  Done: flimmer installed"

# --- 3. Set up HuggingFace ---
echo ""
echo "[3/6] Configuring HuggingFace..."
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ -z "$HF_TOKEN" ]; then
    echo "  WARNING: HF_TOKEN environment variable not set!"
    echo "  Model downloads may fail for gated repos."
    echo "  Fix: Add HF_TOKEN in RunPod pod settings > Environment Variables"
    echo "  Or run: huggingface-cli login"
else
    echo "  HF_TOKEN found"
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
fi

# --- 4. Download models ---
echo ""
echo "[4/6] Downloading models to /workspace/models/..."
echo "  This takes 10-20 minutes on first run."
echo "  Subsequent runs skip already-downloaded files."

mkdir -p /workspace/models

# Helper function: download if not already present.
# Downloads to /workspace (volume disk) not /tmp (container disk) to avoid
# "no space left on device" errors. Clears HF cache after each download.
download_model() {
    local repo=$1
    local filename=$2
    local dest=$3

    if [ -f "$dest" ]; then
        echo "  [CACHED] $(basename $dest)"
        return
    fi

    echo "  [DOWNLOADING] $repo / $(basename $filename) ..."
    huggingface-cli download "$repo" "$filename" \
        --local-dir /workspace/models/hf_tmp \
        --quiet 2>/dev/null
    mv "/workspace/models/hf_tmp/$filename" "$dest"
    rm -rf /workspace/models/hf_tmp
    # Clear HF cache on container disk to prevent filling it up
    rm -rf /root/.cache/huggingface/hub/* /tmp/hf_* 2>/dev/null || true
    echo "  Done: $(basename $dest)"
}

# Shared components (all variants)
echo ""
echo "  Downloading shared components..."

download_model \
    "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
    "split_files/vae/wan_2.1_vae.safetensors" \
    "/workspace/models/wan_2.1_vae.safetensors"

download_model \
    "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
    "split_files/text_encoders/umt5_xxl_fp16.safetensors" \
    "/workspace/models/umt5_xxl_fp16.safetensors"

# Variant-specific DiT weights
echo ""
echo "  Downloading DiT weights for variant: $VARIANT ..."

case "$VARIANT" in
    2.2_t2v)
        download_model \
            "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
            "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors" \
            "/workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors"

        download_model \
            "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
            "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors" \
            "/workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
        ;;

    2.2_i2v)
        download_model \
            "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
            "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
            "/workspace/models/wan2.2_i2v_high_noise_14B_fp16.safetensors"

        download_model \
            "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
            "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
            "/workspace/models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
        ;;

    2.1_i2v_480p)
        download_model \
            "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
            "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors" \
            "/workspace/models/wan2.1_i2v_480p_14B_fp16.safetensors"
        ;;

    2.1_i2v_720p)
        download_model \
            "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
            "split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors" \
            "/workspace/models/wan2.1_i2v_720p_14B_fp16.safetensors"
        ;;
esac

# --- 5. Persist environment variables ---
echo ""
echo "[5/6] Setting environment variables..."
grep -q PYTHONUNBUFFERED /root/.bashrc 2>/dev/null || {
    echo 'export PYTHONUNBUFFERED=1' >> /root/.bashrc
    echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> /root/.bashrc
}
echo "  Done: PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF"

# --- 6. Create directory structure ---
echo ""
echo "[6/6] Creating directory structure..."
mkdir -p /workspace/datasets
mkdir -p /workspace/outputs
echo "  Done"

# --- Done ---
echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "  Models downloaded:"
ls -lh /workspace/models/*.safetensors 2>/dev/null | awk '{print "    " $5 "  " $9}'
echo ""
echo "  Next steps:"
echo "  1. Upload your dataset via Jupyter Lab"
echo "     Drag your video clips + .txt captions into /workspace/datasets/my_dataset/"
echo ""
echo "  2. Copy and edit a training config:"
echo "     cp /workspace/flimmer-trainer/runpod/test-train.yaml /workspace/my_train.yaml"
echo "     Edit data_config to point to your dataset directory"
echo ""
echo "  3. Start training (in tmux so it survives disconnects):"
echo "     tmux new -s train"
echo "     python /workspace/flimmer-trainer/runpod/train.py --config /workspace/my_train.yaml"
echo ""
echo "  4. Download results from /workspace/outputs/ via Jupyter Lab"
echo ""
echo "  AFTER POD RESTART: Run this script again to reinstall Python packages."
echo "  Models and dataset won't re-download (they're on /workspace)."
echo ""
