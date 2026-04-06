#!/usr/bin/env bash
# cluster-setup.sh — Bootstrap Miniconda + pie-train env on a Linux GPU cluster
set -euo pipefail

CONDA_DIR="$HOME/miniconda3"
ENV_NAME="pie-train"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. Install Miniconda if not present ──────────────────────────────────────
if ! command -v conda &>/dev/null && [ ! -f "$CONDA_DIR/bin/conda" ]; then
    echo "==> Installing Miniconda to $CONDA_DIR ..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    echo "==> Miniconda installed."
else
    echo "==> Conda already available."
fi

# Ensure conda is on PATH for this script
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"

# ── 2. Create/update the conda environment ──────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "==> Updating existing '$ENV_NAME' environment ..."
    conda env update -f "$SCRIPT_DIR/environment.yml" --prune
else
    echo "==> Creating '$ENV_NAME' environment ..."
    conda env create -f "$SCRIPT_DIR/environment.yml"
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
