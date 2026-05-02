#!/usr/bin/env bash
# Detached mask=0.90 ablation on GPU 0 (warm-started from videomae_main).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/envs/physrl/bin/activate
mkdir -p videomae/artifacts/logs

INIT_CKPT="${INIT_CKPT:-videomae/artifacts/videomae_main/last.pt}"
RESUME_FLAG=""
INIT_FLAG="--init-checkpoint ${INIT_CKPT}"
if [[ -f videomae/artifacts/videomae_mask90/last.pt ]]; then
    RESUME_FLAG="--resume auto"
    INIT_FLAG=""
fi

setsid nohup env CUDA_VISIBLE_DEVICES=0 python -u -m videomae.train_videomae \
    --data-root /root/data --out-dir videomae/artifacts/videomae_mask90 \
    --epochs 12 --batch-size 64 --num-workers 6 --prefetch-factor 2 \
    --train-index-mode sliding_window --train-stride 2 \
    --valid-index-mode single_clip \
    --mask-ratio 0.90 --tube-size 16,32,32 \
    --lr 7.5e-5 --min-lr 1.5e-6 --warmup-epochs 1 \
    --weight-decay 0.05 --grad-clip 1.0 \
    --amp --amp-dtype bfloat16 --save-every 1 --seed 44 \
    --wandb-mode offline --wandb-project dl-final-videomae \
    --wandb-run-name videomae_ablation_mask90 \
    $INIT_FLAG $RESUME_FLAG \
    > videomae/artifacts/logs/videomae_mask90.out 2>&1 < /dev/null &
echo $! > videomae/artifacts/logs/videomae_mask90.pid
disown -a 2>/dev/null || true
echo "videomae_mask90 PID: $(cat videomae/artifacts/logs/videomae_mask90.pid)"
