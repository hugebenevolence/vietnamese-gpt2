#!/bin/bash
# ==============================================
# Vietnamese GPT-2 SFT Poem Training Script
# ==============================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export WANDB_PROJECT="vietnamese-gpt2"
export WANDB_MODE="online"

uv run python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

mkdir -p artifacts/logs

echo "=========================================="
echo "Vietnamese GPT-2 SFT Poem"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

uv run python src/train_sft.py 2>&1 | tee artifacts/logs/sft_poem_log.txt

echo ""
echo "Training completed!"
