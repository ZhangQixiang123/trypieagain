#!/usr/bin/env bash
# cluster-setup.sh — Bootstrap Miniconda + pie-train env on a Linux GPU cluster
set -euo pipefail

CONDA_DIR="$HOME/miniconda3"
ENV_NAME="pie-train"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

err() { echo "ERROR: $*" >&2; exit 1; }

# ── 0. Preflight checks ─────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Linux" ]]; then
    err "This script requires Linux. Detected OS: $(uname -s)"
fi

if ! command -v wget &>/dev/null && ! command -v curl &>/dev/null; then
    err "Neither wget nor curl found. Install one of them first."
fi

if [ ! -f "$SCRIPT_DIR/environment.yml" ]; then
    err "environment.yml not found in $SCRIPT_DIR. Are you running from the repo root?"
fi

# ── 1. Install Miniconda if not present ──────────────────────────────────────
if ! command -v conda &>/dev/null && [ ! -f "$CONDA_DIR/bin/conda" ]; then
    echo "==> Installing Miniconda to $CONDA_DIR ..."
    INSTALLER="/tmp/miniconda.sh"
    if command -v wget &>/dev/null; then
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$INSTALLER"
    else
        curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$INSTALLER"
    fi
    bash "$INSTALLER" -b -p "$CONDA_DIR" || err "Miniconda installation failed."
    rm -f "$INSTALLER"
    echo "==> Miniconda installed."
else
    echo "==> Conda already available."
fi

# Ensure conda is on PATH for this script
if [ -f "$CONDA_DIR/bin/conda" ]; then
    eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
else
    err "Conda not found after installation. Check $CONDA_DIR/bin/conda"
fi

# ── 2. Create/update the conda environment ──────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "==> Updating existing '$ENV_NAME' environment ..."
    if ! conda env update -f "$SCRIPT_DIR/environment.yml" --prune; then
        err "Failed to update conda environment. Check the output above for dependency conflicts."
    fi
else
    echo "==> Creating '$ENV_NAME' environment (this may take 10-15 minutes) ..."
    if ! conda env create -f "$SCRIPT_DIR/environment.yml"; then
        echo ""
        err "Failed to create conda environment. Common fixes:
  - Install libmamba solver: conda install -n base conda-libmamba-solver && conda config --set solver libmamba
  - Check network connectivity (model downloads require internet)
  - Check disk space: df -h $(dirname $CONDA_DIR)"
    fi
fi

# ── 3. Training data ────────────────────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/training-data-lora-single.jsonl" ]; then
    echo ""
    echo "WARNING: training-data-lora-single.jsonl not found in $SCRIPT_DIR"
    echo "Copy your training data file here before running training:"
    echo "  cp /path/to/training-data-lora-single.jsonl $SCRIPT_DIR/"
fi

# ── 4. Done ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "  Setup complete!"
echo "========================================="
echo ""
echo "Activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Run training:"
echo "  bash run-training.sh              # single GPU"
echo "  bash run-training.sh --multi-gpu  # all GPUs"
echo ""
