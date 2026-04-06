#!/usr/bin/env bash
# run-training.sh — Launch LoRA fine-tuning (single or multi-GPU)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTI_GPU=false

err() { echo "ERROR: $*" >&2; exit 1; }

# Parse flags — everything before "--" is for this script, everything after is
# forwarded to train.py
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --multi-gpu)
            MULTI_GPU=true
            shift
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Preflight checks ────────────────────────────────────────────────────────

# Check train.py exists
if [ ! -f "$SCRIPT_DIR/train.py" ]; then
    err "train.py not found in $SCRIPT_DIR. Are you running from the repo root?"
fi

# Check conda is available
if ! command -v conda &>/dev/null; then
    err "conda not found. Run 'bash cluster-setup.sh' first."
fi

# Activate conda env if needed
if [[ "${CONDA_DEFAULT_ENV:-}" != "pie-train" ]]; then
    echo "Activating pie-train environment ..."
    eval "$(conda shell.bash hook)" 2>/dev/null || err "Failed to initialize conda. Is Miniconda installed?"
    conda activate pie-train 2>/dev/null || err "Failed to activate 'pie-train' environment. Run 'bash cluster-setup.sh' first."
fi

# Check Python is available
if ! command -v python &>/dev/null; then
    err "python not found in the pie-train environment."
fi

# Check for NVIDIA GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null) || err "Failed to detect GPUs. Ensure PyTorch is installed with CUDA support.
  Try: python -c 'import torch; print(torch.cuda.is_available())'"

if [[ "$NUM_GPUS" -eq 0 ]]; then
    err "No GPUs detected. Check that:
  - NVIDIA drivers are installed: nvidia-smi
  - CUDA is available: python -c 'import torch; print(torch.cuda.is_available())'
  - GPUs are visible: echo \$CUDA_VISIBLE_DEVICES"
fi

echo "Visible GPUs: $NUM_GPUS"
python -c "import torch; [print(f'  [{i}] {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

# ── Launch training ──────────────────────────────────────────────────────────

if $MULTI_GPU; then
    if ! command -v accelerate &>/dev/null; then
        err "'accelerate' command not found. Install it: pip install accelerate"
    fi
    echo "==> Launching multi-GPU training on $NUM_GPUS GPUs ..."
    echo "NOTE: Unsloth gradient checkpointing may not work with DDP/FSDP."
    echo "      If you hit errors, use single-GPU mode or set --grad-accum higher."
    echo ""
    accelerate launch \
        --num_processes "$NUM_GPUS" \
        --mixed_precision bf16 \
        "$SCRIPT_DIR/train.py" "${TRAIN_ARGS[@]}"
else
    echo "==> Launching single-GPU training (GPU 0) ..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/train.py" "${TRAIN_ARGS[@]}"
fi
