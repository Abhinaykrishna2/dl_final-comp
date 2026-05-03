#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the active_matter dataset root}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-artifacts/vjepa_96}"
WANDB_MODE="${WANDB_MODE:-disabled}"

"$PYTHON_BIN" -m active_matter_ssl.train_vjepa \
    --data-root "$DATA_ROOT" \
    --out-dir "$RUN_ROOT" \
    --epochs "${EPOCHS:-40}" \
    --batch-size "${BATCH_SIZE:-16}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --prefetch-factor "${PREFETCH_FACTOR:-4}" \
    --resolution 96 \
    --context-frames 16 \
    --target-frames 16 \
    --target-mode future \
    --mask-ratio "${MASK_RATIO:-0.55}" \
    --ema-momentum "${EMA_MOMENTUM:-0.996}" \
    --ema-final-momentum "${EMA_FINAL_MOMENTUM:-1.0}" \
    --lr "${LR:-3e-4}" \
    --weight-decay "${WEIGHT_DECAY:-5e-4}" \
    --amp \
    --amp-dtype bfloat16 \
    --seed "${SEED:-42}" \
    --wandb-mode "$WANDB_MODE"
