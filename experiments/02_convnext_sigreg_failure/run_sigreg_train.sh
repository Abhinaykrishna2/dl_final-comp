#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the active_matter dataset root}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-artifacts/convnext_sigreg96_stopgrad}"
WANDB_MODE="${WANDB_MODE:-disabled}"

"$PYTHON_BIN" -m active_matter_ssl.train_jepa \
    --data-root "$DATA_ROOT" \
    --out-dir "$RUN_ROOT" \
    --epochs "${EPOCHS:-25}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --prefetch-factor "${PREFETCH_FACTOR:-4}" \
    --resolution 96 \
    --context-frames 16 \
    --target-frames 16 \
    --loss-type sigreg \
    --target-stop-grad \
    --lejepa-lambda 0.05 \
    --sigreg-on projection \
    --sigreg-slices 1024 \
    --sigreg-points 17 \
    --sigreg-t-max 3.0 \
    --lr "${LR:-5e-4}" \
    --weight-decay "${WEIGHT_DECAY:-5e-4}" \
    --amp \
    --amp-dtype bfloat16 \
    --seed "${SEED:-42}" \
    --wandb-mode "$WANDB_MODE"
