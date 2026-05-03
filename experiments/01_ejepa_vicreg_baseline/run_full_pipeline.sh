#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the active_matter dataset root}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-artifacts/baseline_jepa96}"
EMB_ROOT="${EMB_ROOT:-artifacts/emb_baseline_jepa96_avg}"
LP_ROOT="${LP_ROOT:-artifacts/lp_baseline_jepa96}"
KNN_ROOT="${KNN_ROOT:-artifacts/knn_baseline_jepa96}"
WANDB_MODE="${WANDB_MODE:-disabled}"

"$PYTHON_BIN" -m baseline_jepa.train_jepa \
    --data-root "$DATA_ROOT" \
    --out-dir "$RUN_ROOT" \
    --epochs "${EPOCHS:-30}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --prefetch-factor "${PREFETCH_FACTOR:-4}" \
    --resolution 96 \
    --context-frames 16 \
    --target-frames 16 \
    --loss-type vicreg \
    --lr "${LR:-5e-4}" \
    --weight-decay "${WEIGHT_DECAY:-5e-4}" \
    --amp \
    --amp-dtype bfloat16 \
    --seed "${SEED:-42}" \
    --wandb-mode "$WANDB_MODE"

"$PYTHON_BIN" -m baseline_jepa.export_embeddings \
    --data-root "$DATA_ROOT" \
    --checkpoint "$RUN_ROOT/encoder_best.pt" \
    --out-dir "$EMB_ROOT" \
    --batch-size "${EXPORT_BATCH_SIZE:-32}" \
    --pool avg \
    --index-mode single_clip \
    --clip-selection center \
    --amp

"$PYTHON_BIN" -m baseline_jepa.sweep_linear_probe \
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
    --backend auto \
    --wandb-mode "$WANDB_MODE"
