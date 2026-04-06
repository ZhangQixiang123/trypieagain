#!/usr/bin/env bash
# run-training.sh — Launch LoRA fine-tuning (single or multi-GPU)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTI_GPU=false

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

# Verify conda env is active
if [[ "${CONDA_DEFAULT_ENV:-}" != "pie-train" ]]; then
    echo "Activating pie-train environment ..."
    eval "$(conda shell.bash hook)"
    conda activate pie-train
fi

# Verify GPU visibility
echo "Visible GPUs:"
python -c "import torch; n=torch.cuda.device_count(); print(f'  {n} GPU(s)'); [print(f'  [{i}] {torch.cuda.get_device_name(i)}') for i in range(n)]"
echo ""

if $MULTI_GPU; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
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
