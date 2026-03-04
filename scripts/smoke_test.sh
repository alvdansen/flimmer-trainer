#!/usr/bin/env bash
# ======================================================================
# Flimmer Smoke Test
# ======================================================================
#
# End-to-end pipeline verification: synthetic dataset → validate →
# pre-encode → train (1 epoch). Confirms the full pipeline works
# before committing to a real training run.
#
# Prerequisites:
#   - bash scripts/setup.sh --variant <variant>  (venv + deps + weights)
#   - ffmpeg installed (for synthetic clip generation)
#
# Usage:
#   bash scripts/smoke_test.sh                           # auto-detect variant from ./models/
#   bash scripts/smoke_test.sh --variant 2.2_i2v         # explicit variant
#   bash scripts/smoke_test.sh --dataset ./my_clips      # use real clips instead of synthetic
#   bash scripts/smoke_test.sh --keep                    # don't clean up test artifacts
#   bash scripts/smoke_test.sh --skip-training           # validate + encode only (no GPU training)
#
# What it does:
#   1. Creates a tiny synthetic dataset (3 clips, ~1s each) with captions
#   2. Generates matching data + training YAML configs
#   3. Runs dataset validation
#   4. Runs latent pre-encoding (VAE)
#   5. Runs text pre-encoding (T5)
#   6. Runs training for 1 epoch
#   7. Checks that a checkpoint was produced
#   8. Cleans up (unless --keep)
#
# Exit codes:
#   0  All steps passed
#   1  A pipeline step failed (see output for which one)
#
# ======================================================================
set -euo pipefail

# ── Path Resolution ───────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${FLIMMER_VENV:-$REPO_DIR/.venv}"
TEST_DIR="$REPO_DIR/.smoke_test"
MODELS_DIR="$REPO_DIR/models"

# ── Argument Parsing ──────────────────────────────────────────────────

VARIANT=""
USER_DATASET=""
KEEP=false
SKIP_TRAINING=false

usage() {
    echo "Usage: bash scripts/smoke_test.sh [OPTIONS]"
    echo ""
    echo "End-to-end pipeline smoke test."
    echo ""
    echo "Options:"
    echo "  --variant VARIANT   Model variant (2.2_t2v, 2.2_i2v, 2.1_i2v_480p, 2.1_i2v_720p)"
    echo "                      Auto-detected from ./models/ if omitted"
    echo "  --dataset PATH      Use real clips instead of synthetic test data"
    echo "  --keep              Don't clean up test artifacts after completion"
    echo "  --skip-training     Run validate + encode only (no GPU training step)"
    echo "  -h, --help          Show this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"; shift 2 ;;
        --dataset)
            USER_DATASET="$2"; shift 2 ;;
        --keep)
            KEEP=true; shift ;;
        --skip-training)
            SKIP_TRAINING=true; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────

step_num=0

step() {
    step_num=$((step_num + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Step $step_num: $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

pass() {
    echo "  ✓ $1"
}

fail() {
    echo "  ✗ $1"
    echo ""
    echo "SMOKE TEST FAILED at step $step_num."
    exit 1
}

cleanup() {
    if [[ "$KEEP" = false && -d "$TEST_DIR" ]]; then
        echo ""
        echo "Cleaning up $TEST_DIR ..."
        rm -rf "$TEST_DIR"
        echo "  Done."
    fi
}

# Clean up on exit unless --keep
if [[ "$KEEP" = false ]]; then
    trap cleanup EXIT
fi

# ── Activate Venv ─────────────────────────────────────────────────────

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    else
        echo "WARNING: No venv found at $VENV_DIR. Using system Python."
        echo "  Run 'bash scripts/setup.sh' first if you haven't."
    fi
fi

# ── Auto-Detect Variant ──────────────────────────────────────────────

if [[ -z "$VARIANT" ]]; then
    step "Auto-detecting model variant from $MODELS_DIR/"

    if [[ -f "$MODELS_DIR/wan2.2_i2v_high_noise_14B_fp16.safetensors" ]]; then
        VARIANT="2.2_i2v"
    elif [[ -f "$MODELS_DIR/wan2.2_t2v_high_noise_14B_fp16.safetensors" ]]; then
        VARIANT="2.2_t2v"
    elif [[ -f "$MODELS_DIR/wan2.1_i2v_480p_14B_fp16.safetensors" ]]; then
        VARIANT="2.1_i2v_480p"
    elif [[ -f "$MODELS_DIR/wan2.1_i2v_720p_14B_fp16.safetensors" ]]; then
        VARIANT="2.1_i2v_720p"
    else
        fail "No model weights found in $MODELS_DIR/. Run setup.sh first."
    fi

    pass "Detected variant: $VARIANT"
fi

# Derive variant properties
IS_I2V=false
IS_MOE=false
RESOLUTION=480

case "$VARIANT" in
    2.2_t2v)
        IS_MOE=true
        DIT_HIGH="$MODELS_DIR/wan2.2_t2v_high_noise_14B_fp16.safetensors"
        DIT_LOW="$MODELS_DIR/wan2.2_t2v_low_noise_14B_fp16.safetensors"
        ;;
    2.2_i2v)
        IS_I2V=true
        IS_MOE=true
        DIT_HIGH="$MODELS_DIR/wan2.2_i2v_high_noise_14B_fp16.safetensors"
        DIT_LOW="$MODELS_DIR/wan2.2_i2v_low_noise_14B_fp16.safetensors"
        ;;
    2.1_i2v_480p)
        IS_I2V=true
        DIT="$MODELS_DIR/wan2.1_i2v_480p_14B_fp16.safetensors"
        ;;
    2.1_i2v_720p)
        IS_I2V=true
        RESOLUTION=720
        DIT="$MODELS_DIR/wan2.1_i2v_720p_14B_fp16.safetensors"
        ;;
    *)
        fail "Unknown variant: $VARIANT"
        ;;
esac

VAE="$MODELS_DIR/wan_2.1_vae.safetensors"
T5="$MODELS_DIR/umt5_xxl_fp16.safetensors"

echo ""
echo "  Variant:    $VARIANT"
echo "  I2V:        $IS_I2V"
echo "  MoE:        $IS_MOE"
echo "  Resolution: ${RESOLUTION}p"
echo ""

# ── Verify Weight Files Exist ─────────────────────────────────────────

step "Verifying model weights"

[[ -f "$VAE" ]] && pass "VAE: $(basename "$VAE")" || fail "VAE not found: $VAE"
[[ -f "$T5" ]]  && pass "T5:  $(basename "$T5")"  || fail "T5 not found: $T5"

if [[ "$IS_MOE" = true ]]; then
    [[ -f "$DIT_HIGH" ]] && pass "DiT High: $(basename "$DIT_HIGH")" || fail "DiT High not found: $DIT_HIGH"
    [[ -f "$DIT_LOW" ]]  && pass "DiT Low:  $(basename "$DIT_LOW")"  || fail "DiT Low not found: $DIT_LOW"
else
    [[ -f "$DIT" ]] && pass "DiT: $(basename "$DIT")" || fail "DiT not found: $DIT"
fi

# ── Create Test Directory ─────────────────────────────────────────────

mkdir -p "$TEST_DIR/clips"
mkdir -p "$TEST_DIR/output"

# ── Generate Synthetic Dataset OR Use Real Clips ──────────────────────

CLIPS_DIR="$TEST_DIR/clips"

if [[ -n "$USER_DATASET" ]]; then
    step "Using provided dataset: $USER_DATASET"

    if [[ ! -d "$USER_DATASET" ]]; then
        fail "Dataset directory not found: $USER_DATASET"
    fi

    CLIPS_DIR="$USER_DATASET"
    clip_count=$(find "$CLIPS_DIR" -maxdepth 1 -name "*.mp4" | wc -l)
    pass "Found $clip_count clip(s)"
else
    step "Generating synthetic test clips"

    if ! command -v ffmpeg &>/dev/null; then
        fail "ffmpeg not found. Install it: apt install ffmpeg (Linux) or winget install ffmpeg (Windows)"
    fi

    # Wan expects 4n+1 frames. At 16fps, 17 frames = 1.0625s (smallest valid clip).
    # Resolution: 480p → 848x480 (16:9). 720p → 1280x720.
    if [[ "$RESOLUTION" = 720 ]]; then
        W=1280; H=720
    else
        W=848; H=480
    fi

    CAPTIONS=(
        "A person walks through a sunlit garden, flowers blooming around them"
        "Close-up of hands arranging flowers in a glass vase on a wooden table"
        "A woman stands by a window, warm light casting soft shadows on her face"
    )

    # Generate 3 short clips with different solid colors (fast, no complex encoding)
    COLORS=("0x3B5998" "0x8B4513" "0x2E8B57")
    for i in 0 1 2; do
        clip_name="test_clip_$(printf '%02d' $i).mp4"
        caption_file="test_clip_$(printf '%02d' $i).txt"

        # Create a 17-frame clip (4n+1) at 16fps with a solid color + slight motion
        ffmpeg -y -f lavfi \
            -i "color=c=${COLORS[$i]}:size=${W}x${H}:rate=16:d=1.0625,drawtext=text='clip $i frame %{n}':fontsize=36:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2" \
            -frames:v 17 \
            -c:v libx264 -pix_fmt yuv420p \
            "$CLIPS_DIR/$clip_name" \
            2>/dev/null

        echo "${CAPTIONS[$i]}" > "$CLIPS_DIR/$caption_file"
        pass "$clip_name (17 frames, ${W}x${H})"
    done

    # For I2V: extract first frames as reference images
    if [[ "$IS_I2V" = true ]]; then
        echo ""
        echo "  Extracting reference frames for I2V..."
        mkdir -p "$CLIPS_DIR/references"
        for clip in "$CLIPS_DIR"/*.mp4; do
            base=$(basename "$clip" .mp4)
            ffmpeg -y -i "$clip" -frames:v 1 "$CLIPS_DIR/references/${base}.png" 2>/dev/null
            pass "  ${base}.png"
        done
    fi
fi

# ── Generate Data Config ──────────────────────────────────────────────

step "Writing test configs"

REF_SOURCE="none"
if [[ "$IS_I2V" = true ]]; then
    REF_SOURCE="first_frame"
fi

cat > "$TEST_DIR/data.yaml" << EOF
dataset:
  name: smoke_test
  use_case: character

datasets:
  - path: $CLIPS_DIR
    repeats: 1

video:
  fps: 16
  resolution: $RESOLUTION

controls:
  text:
    anchor_word: smoketest
  images:
    reference:
      source: $REF_SOURCE
EOF

pass "data.yaml"

# ── Generate Training Config ─────────────────────────────────────────

# Build model section based on variant
if [[ "$IS_MOE" = true ]]; then
    MODEL_SECTION="model:
  variant: $VARIANT
  dit_high: $DIT_HIGH
  dit_low: $DIT_LOW
  vae: $VAE
  t5: $T5"
else
    MODEL_SECTION="model:
  variant: $VARIANT
  dit: $DIT
  vae: $VAE
  t5: $T5"
fi

# I2V uses higher caption dropout
if [[ "$IS_I2V" = true ]]; then
    CAPTION_DROPOUT="0.15"
else
    CAPTION_DROPOUT="0.10"
fi

# MoE section (only for 2.2 variants)
if [[ "$IS_MOE" = true ]]; then
    MOE_SECTION="
moe:
  enabled: true
  fork_enabled: false
  preload_experts: false"
else
    MOE_SECTION="
moe:
  enabled: false"
fi

cat > "$TEST_DIR/train.yaml" << EOF
$MODEL_SECTION

data_config: $TEST_DIR/data.yaml

lora:
  rank: 4
  alpha: 4
  loraplus_lr_ratio: 1.0
  dropout: 0.0
  target_modules: null

optimizer:
  type: adamw8bit
  learning_rate: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8
  max_grad_norm: 1.0

scheduler:
  type: constant
  warmup_steps: 0

training:
  mixed_precision: bf16
  base_model_precision: bf16
  gradient_checkpointing: true
  seed: 42
  unified_epochs: 1
  batch_size: 1
  gradient_accumulation_steps: 1
  caption_dropout_rate: $CAPTION_DROPOUT
  timestep_sampling: shift
$MOE_SECTION

save:
  output_dir: $TEST_DIR/output
  name: smoke_test_lora
  save_every_n_epochs: 1
  save_last: true
  format: safetensors

logging:
  backends: [console]
  log_every_n_steps: 1

sampling:
  enabled: false
EOF

pass "train.yaml"

# ── Validate Dataset ──────────────────────────────────────────────────

step "Validating dataset"

if python -m flimmer.dataset validate "$CLIPS_DIR"; then
    pass "Dataset validation passed"
else
    fail "Dataset validation failed"
fi

# ── Pre-Encode Latents (VAE) ─────────────────────────────────────────

step "Pre-encoding latents (VAE)"

if python -m flimmer.encoding cache-latents -c "$TEST_DIR/train.yaml"; then
    pass "Latent encoding complete"
else
    fail "Latent encoding failed"
fi

# ── Pre-Encode Text (T5) ─────────────────────────────────────────────

step "Pre-encoding text (T5)"

if python -m flimmer.encoding cache-text -c "$TEST_DIR/train.yaml"; then
    pass "Text encoding complete"
else
    fail "Text encoding failed"
fi

# ── Verify Cached Files ──────────────────────────────────────────────

step "Checking cached files"

latent_count=$(find "$CLIPS_DIR" -name "*.latent.pt" 2>/dev/null | wc -l)
text_count=$(find "$CLIPS_DIR" -name "*.text.pt" 2>/dev/null | wc -l)

if [[ "$latent_count" -gt 0 ]]; then
    pass "Found $latent_count cached latent file(s)"
else
    fail "No cached latent files found (expected *.latent.pt alongside clips)"
fi

if [[ "$text_count" -gt 0 ]]; then
    pass "Found $text_count cached text embedding file(s)"
else
    fail "No cached text embedding files found (expected *.text.pt alongside clips)"
fi

# ── Train (1 epoch) ──────────────────────────────────────────────────

if [[ "$SKIP_TRAINING" = true ]]; then
    echo ""
    echo "  Skipping training (--skip-training)."
    echo ""
else
    step "Training (1 epoch)"

    if python -m flimmer.training train -c "$TEST_DIR/train.yaml"; then
        pass "Training complete"
    else
        fail "Training failed"
    fi

    # ── Verify Checkpoint ─────────────────────────────────────────────

    step "Checking training output"

    checkpoint_count=$(find "$TEST_DIR/output" -name "*.safetensors" 2>/dev/null | wc -l)

    if [[ "$checkpoint_count" -gt 0 ]]; then
        pass "Found $checkpoint_count checkpoint file(s):"
        find "$TEST_DIR/output" -name "*.safetensors" | while read -r f; do
            size=$(du -h "$f" | cut -f1)
            echo "     $(basename "$f") ($size)"
        done
    else
        fail "No checkpoint files found in $TEST_DIR/output/"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SMOKE TEST PASSED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Variant:     $VARIANT"
echo "  Clips:       $(find "$CLIPS_DIR" -maxdepth 1 -name "*.mp4" | wc -l)"
echo "  Latents:     $latent_count cached"
echo "  Text:        $text_count cached"
if [[ "$SKIP_TRAINING" = false ]]; then
echo "  Checkpoints: $checkpoint_count"
fi
echo ""
if [[ "$KEEP" = true ]]; then
echo "  Test artifacts kept at: $TEST_DIR"
else
echo "  Cleaning up test artifacts..."
fi
echo ""
echo "  The pipeline works. You're ready for a real training run."
echo ""
