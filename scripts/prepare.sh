#!/usr/bin/env bash
# ======================================================================
# Flimmer Prepare Script
# ======================================================================
#
# Runs latent and text pre-encoding for a training config or project.
# This is step 2 of the three-script workflow: setup.sh -> prepare.sh -> train.sh
#
# Usage:
#   bash scripts/prepare.sh --config path/to/train.yaml
#   bash scripts/prepare.sh --project path/to/project.yaml
#   bash scripts/prepare.sh --config path/to/train.yaml --dry-run
#   bash scripts/prepare.sh --config path/to/train.yaml --force
#
# Options:
#   --config PATH    Direct path to a flimmer_train.yaml
#   --project PATH   Path to a project YAML (extracts base_config from it)
#   --dry-run        Preview encoding without running it
#   --force          Re-encode latents even if cached
#   -h, --help       Show this help message
#
# Exactly one of --config or --project is required.
#
# ======================================================================
set -euo pipefail

# -- Path Resolution ---------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${FLIMMER_VENV:-$REPO_DIR/.venv}"

# -- Argument Parsing --------------------------------------------------

CONFIG_PATH=""
PROJECT_PATH=""
DRY_RUN=false
FORCE=false

usage() {
    echo "Usage: bash scripts/prepare.sh [OPTIONS]"
    echo ""
    echo "Run latent and text pre-encoding for a training config or project."
    echo ""
    echo "Options:"
    echo "  --config PATH    Direct path to a flimmer_train.yaml"
    echo "  --project PATH   Path to a project YAML (extracts base_config)"
    echo "  --dry-run        Preview encoding without running it"
    echo "  --force          Re-encode latents even if cached"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Exactly one of --config or --project is required."
    echo ""
    echo "Examples:"
    echo "  bash scripts/prepare.sh --config examples/full_train.yaml"
    echo "  bash scripts/prepare.sh --project examples/project_moe.yaml"
    echo "  bash scripts/prepare.sh --config train.yaml --dry-run"
    echo "  bash scripts/prepare.sh --config train.yaml --force"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --config requires a path"
                usage
                exit 1
            fi
            CONFIG_PATH="$2"
            shift 2
            ;;
        --project)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --project requires a path"
                usage
                exit 1
            fi
            PROJECT_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
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

# Validate: exactly one of --config or --project
if [[ -n "$CONFIG_PATH" && -n "$PROJECT_PATH" ]]; then
    echo "ERROR: Cannot specify both --config and --project."
    echo ""
    usage
    exit 1
fi

if [[ -z "$CONFIG_PATH" && -z "$PROJECT_PATH" ]]; then
    echo "ERROR: One of --config or --project is required."
    echo ""
    usage
    exit 1
fi

# -- Activate Venv -----------------------------------------------------

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: Virtual environment not found at: $VENV_DIR"
    echo ""
    echo "Run setup.sh first to create the environment:"
    echo "  bash scripts/setup.sh --variant <variant>"
    echo ""
    echo "Or set FLIMMER_VENV to point to an existing venv."
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# -- Resolve Config Path -----------------------------------------------

if [[ -n "$PROJECT_PATH" ]]; then
    # Extract base_config from the project YAML
    if [[ ! -f "$PROJECT_PATH" ]]; then
        echo "ERROR: Project file not found: $PROJECT_PATH"
        exit 1
    fi

    CONFIG_PATH=$(python3 -c "
import yaml
from pathlib import Path
project_path = Path('$PROJECT_PATH')
data = yaml.safe_load(open(project_path))
base = data.get('base_config', '')
if not base:
    import sys
    print('ERROR: project YAML has no base_config field', file=sys.stderr)
    sys.exit(1)
# Resolve relative to project YAML directory
resolved = (project_path.parent / base).resolve()
print(resolved)
")

    echo "Resolved config from project: $CONFIG_PATH"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

# -- Build Flags -------------------------------------------------------

DRY_RUN_FLAG=""
FORCE_FLAG=""

if [[ "$DRY_RUN" = true ]]; then
    DRY_RUN_FLAG="--dry-run"
fi

if [[ "$FORCE" = true ]]; then
    FORCE_FLAG="--force"
fi

# -- Run Encoding ------------------------------------------------------

echo ""
echo "=== Flimmer: Latent Pre-Encoding ==="
echo ""
echo "Config: $CONFIG_PATH"
echo ""

# shellcheck disable=SC2086
python -m flimmer.encoding cache-latents --config "$CONFIG_PATH" $DRY_RUN_FLAG $FORCE_FLAG

echo ""
echo "=== Flimmer: Text Pre-Encoding ==="
echo ""

# shellcheck disable=SC2086
python -m flimmer.encoding cache-text --config "$CONFIG_PATH" $DRY_RUN_FLAG

echo ""
echo "=== Encoding Complete ==="
echo ""
echo "Next step: Run scripts/train.sh to start training"
echo ""
