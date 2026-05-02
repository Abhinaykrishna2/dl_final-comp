#!/usr/bin/env bash
# Detached supervised retrain at seed=43.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/envs/physrl/bin/activate
mkdir -p videomae/artifacts/logs

RESUME_FLAG=""
if [[ -f videomae/artifacts/supervised_s43/last.pt ]]; then
    RESUME_FLAG="--resume auto"
fi

setsid nohup env CUDA_VISIBLE_DEVICES=1 python -u -m videomae.train_supervised \
    --data-root /root/data --out-dir videomae/artifacts/supervised_s43 \
    --epochs 30 --batch-size 64 --num-workers 4 --prefetch-factor 2 \
    --train-index-mode sliding_window --train-stride 4 \
    --valid-index-mode single_clip \
    --lr 3e-4 --min-lr 3e-6 --warmup-epochs 1 \
    --weight-decay 0.05 --grad-clip 1.0 \
    --amp --amp-dtype bfloat16 --save-every 1 --seed 43 \
    --wandb-mode offline --wandb-project dl-final-videomae \
    --wandb-run-name supervised_s43 \
    $RESUME_FLAG \
    > videomae/artifacts/logs/supervised_s43.out 2>&1 < /dev/null &
echo $! > videomae/artifacts/logs/supervised_s43.pid
disown -a 2>/dev/null || true
echo "supervised_s43 PID: $(cat videomae/artifacts/logs/supervised_s43.pid)"
