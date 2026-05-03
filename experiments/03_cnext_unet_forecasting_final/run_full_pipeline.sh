#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the active_matter dataset root}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-artifacts/cnext_unet96}"
EMB_ROOT="${EMB_ROOT:-artifacts/emb_cnext_unet96_avgmax}"
LP_ROOT="${LP_ROOT:-artifacts/lp_cnext_unet96_avgmax}"
KNN_ROOT="${KNN_ROOT:-artifacts/knn_cnext_unet96_avgmax}"
WANDB_MODE="${WANDB_MODE:-disabled}"

"$PYTHON_BIN" -m active_matter_ssl.train_cnext_forecaster \
    --data-root "$DATA_ROOT" \
    --out-dir "$RUN_ROOT" \
    --epochs "${EPOCHS:-50}" \
    --batch-size "${BATCH_SIZE:-16}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --prefetch-factor "${PREFETCH_FACTOR:-4}" \
    --resolution 96 \
    --context-frames 4 \
    --target-frames 1 \
    --lr "${LR:-3e-4}" \
    --weight-decay "${WEIGHT_DECAY:-5e-4}" \
    --amp \
    --amp-dtype bfloat16 \
    --save-every 2 \
    --seed "${SEED:-42}" \
    --wandb-mode "$WANDB_MODE"

"$PYTHON_BIN" -m active_matter_ssl.export_cnext_embeddings \
    --data-root "$DATA_ROOT" \
    --checkpoint "$RUN_ROOT/encoder_best.pt" \
    --out-dir "$EMB_ROOT" \
    --batch-size "${EXPORT_BATCH_SIZE:-64}" \
    --pool avgmax \
    --clip-frames 16 \
    --window-stride 1 \
    --amp

"$PYTHON_BIN" -m active_matter_ssl.sweep_linear_probe \
    --train-file "$EMB_ROOT/train.npz" \
    --valid-file "$EMB_ROOT/valid.npz" \
    --test-file "$EMB_ROOT/test.npz" \
    --out-dir "$LP_ROOT" \
    --epochs 200 \
    --patience 25 \
    --wandb-mode "$WANDB_MODE"

"$PYTHON_BIN" -m active_matter_ssl.eval_knn \
    --train-file "$EMB_ROOT/train.npz" \
    --valid-file "$EMB_ROOT/valid.npz" \
    --test-file "$EMB_ROOT/test.npz" \
    --out-dir "$KNN_ROOT" \
    --backend torch \
    --wandb-mode "$WANDB_MODE"
