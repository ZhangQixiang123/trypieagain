# Cluster Training Setup

Guide for running LoRA fine-tuning on a Linux GPU cluster (tested on 4x H100 80GB).

## Prerequisites

- Linux (x86_64)
- NVIDIA drivers installed (550+ recommended)
- CUDA 12.x (12.4–12.9 all work — PyTorch CUDA 12.4 is backward-compatible)
- Network access to download Miniconda and model weights

## Quick Start

```bash
# 1. Setup environment (installs Miniconda + conda env)
bash cluster-setup.sh

# 2. Activate
conda activate pie-train

# 3. Train (single GPU)
bash run-training.sh

# 4. Train (all GPUs)
bash run-training.sh --multi-gpu
```

## Training Data

Place `training-data-lora-single.jsonl` in the repo root. The setup script will warn you if it's missing.

To use a different data file:
```bash
bash run-training.sh --data /path/to/your/data.jsonl
```

## Customizing Training

All `train.py` arguments pass through `run-training.sh`:

```bash
# More epochs, lower learning rate, bigger batch
bash run-training.sh --epochs 10 --lr 1e-4 --batch 8

# Different base model
bash run-training.sh --model Qwen/Qwen2.5-Coder-14B-Instruct
```

Key defaults: 5 epochs, lr=2e-4, batch=4, grad_accum=4 (effective batch 16), LoRA r=32.

## Multi-GPU Notes

**Recommended approach**: Run independent single-GPU experiments on different data splits or hyperparameters. Unsloth's `use_gradient_checkpointing="unsloth"` gives 60% VRAM savings but is designed for single-GPU use.

With 4x H100 80GB, you can run 4 parallel experiments:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --output output-run1 --lr 2e-4 &
CUDA_VISIBLE_DEVICES=1 python train.py --output output-run2 --lr 1e-4 &
CUDA_VISIBLE_DEVICES=2 python train.py --output output-run3 --lr 5e-5 &
CUDA_VISIBLE_DEVICES=3 python train.py --output output-run4 --epochs 10 &
wait
```

The `--multi-gpu` flag uses `accelerate launch` for distributed training, but note that Unsloth's gradient checkpointing may conflict with DDP/FSDP. If you encounter issues, fall back to single-GPU mode.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `triton` import error | Ensure you're on Linux; `triton` doesn't support Windows |
| OOM on smaller GPUs | Reduce `--batch` to 1–2 or increase `--grad-accum` |
| Slow download of model weights | Set `HF_HOME` to a shared filesystem path |
| conda solve is slow | Run `conda install -n base conda-libmamba-solver` then `conda config --set solver libmamba` |
