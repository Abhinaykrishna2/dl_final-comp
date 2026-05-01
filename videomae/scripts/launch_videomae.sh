#!/usr/bin/env bash
# Detached launch of the VideoMAE main run on GPU 0.
# Uses setsid + nohup so the process survives SSH drops, terminal closes,
# and agent restarts. Writes a clean console log to artifacts/logs/<run>.out.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/envs/physrl/bin/activate
mkdir -p videomae/artifacts/logs

# Resume if a checkpoint exists, otherwise start fresh.
RESUME_FLAG=""
if [[ -f videomae/artifacts/videomae_main/last.pt ]]; then
    RESUME_FLAG="--resume auto"
fi

setsid nohup env CUDA_VISIBLE_DEVICES=0 python -u -m videomae.train_videomae \
    --data-root /root/data --out-dir videomae/artifacts/videomae_main \
    --epochs 25 --batch-size 64 --num-workers 6 --prefetch-factor 2 \
    --train-index-mode sliding_window --train-stride 2 \
    --valid-index-mode single_clip \
    --mask-ratio 0.60 --tube-size 16,32,32 \
    --lr 1.5e-4 --min-lr 1.5e-6 --warmup-epochs 2 \
    --weight-decay 0.05 --grad-clip 1.0 \
    --amp --amp-dtype bfloat16 --save-every 1 --seed 42 \
    --wandb-mode offline --wandb-project dl-final-videomae \
    --wandb-run-name videomae_main_mask60 \
    $RESUME_FLAG \
    > videomae/artifacts/logs/videomae_main.out 2>&1 < /dev/null &
echo $! > videomae/artifacts/logs/videomae_main.pid
disown -a 2>/dev/null || true
echo "VideoMAE main PID: $(cat videomae/artifacts/logs/videomae_main.pid)"
