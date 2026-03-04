#!/usr/bin/env bash
# ======================================================================
# Flimmer Training Script
# ======================================================================
#
# Launches training via project-based phase management or single config.
# This is step 3 of the three-script workflow: setup.sh -> prepare.sh -> train.sh
#
# Usage:
#   bash scripts/train.sh --project path/to/project.yaml
#   bash scripts/train.sh --project path/to/project.yaml --all
#   bash scripts/train.sh --project path/to/project.yaml --tmux
#   bash scripts/train.sh --project path/to/project.yaml --status
#   bash scripts/train.sh --config path/to/train.yaml
#   bash scripts/train.sh --config path/to/train.yaml --dry-run
#
# Options:
#   --project PATH   Path to a project YAML (multi-phase training)
#   --config PATH    Direct path to a flimmer_train.yaml (single run)
#   --tmux           Start training in a named tmux session
#   --all            Run all pending phases (only with --project)
#   --dry-run        Show what would happen without training
#   --status         Show project phase statuses (only with --project)
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
USE_TMUX=false
RUN_ALL=false
DRY_RUN=false
SHOW_STATUS=false

usage() {
    echo "Usage: bash scripts/train.sh [OPTIONS]"
    echo ""
    echo "Launch Flimmer training with project phase management or single config."
    echo ""
    echo "Options:"
    echo "  --project PATH   Path to a project YAML (multi-phase training)"
    echo "  --config PATH    Direct path to a flimmer_train.yaml (single run)"
    echo "  --tmux           Start training in a named tmux session"
    echo "  --all            Run all pending phases (only with --project)"
    echo "  --dry-run        Show what would happen without training"
    echo "  --status         Show project phase statuses (only with --project)"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Exactly one of --config or --project is required."
    echo ""
    echo "Examples:"
    echo "  bash scripts/train.sh --project examples/project_moe.yaml"
    echo "  bash scripts/train.sh --project examples/project_moe.yaml --all --tmux"
    echo "  bash scripts/train.sh --config examples/full_train.yaml"
    echo "  bash scripts/train.sh --project examples/project_moe.yaml --status"
    echo "  bash scripts/train.sh --project examples/project_moe.yaml --dry-run"
}

# Collect all original args for tmux re-invocation
ORIGINAL_ARGS=("$@")

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
        --tmux)
            USE_TMUX=true
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --status)
            SHOW_STATUS=true
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

# Validate: --all and --status only with --project
if [[ "$RUN_ALL" = true && -z "$PROJECT_PATH" ]]; then
    echo "ERROR: --all can only be used with --project."
    exit 1
fi

if [[ "$SHOW_STATUS" = true && -z "$PROJECT_PATH" ]]; then
    echo "ERROR: --status can only be used with --project."
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

# -- Handle --tmux -----------------------------------------------------

if [[ "$USE_TMUX" = true ]]; then
    if ! command -v tmux &>/dev/null; then
        echo "WARNING: --tmux requested but tmux is not installed. Running in foreground."
    else
        # Derive session name from project or config filename
        if [[ -n "$PROJECT_PATH" ]]; then
            SESSION="flimmer-$(basename "$PROJECT_PATH" .yaml)"
        else
            SESSION="flimmer-$(basename "$CONFIG_PATH" .yaml)"
        fi

        # Rebuild args without --tmux to avoid infinite recursion
        ARGS_WITHOUT_TMUX=()
        for arg in "${ORIGINAL_ARGS[@]}"; do
            if [[ "$arg" != "--tmux" ]]; then
                ARGS_WITHOUT_TMUX+=("$arg")
            fi
        done

        tmux new-session -d -s "$SESSION" "$0 ${ARGS_WITHOUT_TMUX[*]}"

        echo "Training started in tmux session: $SESSION"
        echo "  Attach: tmux attach -t $SESSION"
        exit 0
    fi
fi

# -- Handle --status (project mode only) --------------------------------

if [[ "$SHOW_STATUS" = true ]]; then
    python -m flimmer.project status --project "$PROJECT_PATH"
    exit 0
fi

# -- Handle --dry-run ---------------------------------------------------

if [[ "$DRY_RUN" = true ]]; then
    if [[ -n "$PROJECT_PATH" ]]; then
        python -m flimmer.project plan --project "$PROJECT_PATH"
    else
        python -m flimmer.training plan --config "$CONFIG_PATH"
    fi
    exit 0
fi

# -- Run Training -------------------------------------------------------

EXIT_CODE=0

if [[ -n "$PROJECT_PATH" ]]; then
    echo ""
    echo "=== Flimmer: Project Training ==="
    echo ""
    echo "Project: $PROJECT_PATH"
    echo ""

    CMD=(python -m flimmer.project run --project "$PROJECT_PATH")
    if [[ "$RUN_ALL" = true ]]; then
        CMD+=(--all)
    fi

    "${CMD[@]}" || EXIT_CODE=$?

    if [[ "$EXIT_CODE" -eq 0 ]]; then
        echo ""
        echo "=== Training Complete ==="
        echo ""
    else
        echo ""
        echo "=== Training Failed (exit code $EXIT_CODE) ==="
        echo ""
    fi
else
    echo ""
    echo "=== Flimmer: Single Config Training ==="
    echo ""
    echo "Config: $CONFIG_PATH"
    echo ""

    python -m flimmer.training train --config "$CONFIG_PATH" || EXIT_CODE=$?

    if [[ "$EXIT_CODE" -eq 0 ]]; then
        echo ""
        echo "=== Training Complete ==="
        echo ""
    else
        echo ""
        echo "=== Training Failed (exit code $EXIT_CODE) ==="
        echo ""
    fi
fi

exit "$EXIT_CODE"
