#!/usr/bin/env bash
# Detached launch of the VideoMAE mask=0.75 ablation on GPU 0.
# Warm-starts from the main VideoMAE encoder (videomae_main/last.pt) so it
# only needs ~12 epochs to converge. Tests SimMIM-0.60 vs VideoMAE-0.75
# question on physical fields.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/envs/physrl/bin/activate
mkdir -p videomae/artifacts/logs

INIT_CKPT="${INIT_CKPT:-videomae/artifacts/videomae_main/last.pt}"
if [[ ! -f "${INIT_CKPT}" ]]; then
    echo "[ablation] init checkpoint missing: ${INIT_CKPT}"
    exit 1
fi

RESUME_FLAG=""
if [[ -f videomae/artifacts/videomae_mask75/last.pt ]]; then
    RESUME_FLAG="--resume auto"
    INIT_FLAG=""
else
    INIT_FLAG="--init-checkpoint ${INIT_CKPT}"
fi

setsid nohup env CUDA_VISIBLE_DEVICES=0 python -u -m videomae.train_videomae \
    --data-root /root/data --out-dir videomae/artifacts/videomae_mask75 \
    --epochs 12 --batch-size 64 --num-workers 6 --prefetch-factor 2 \
    --train-index-mode sliding_window --train-stride 2 \
    --valid-index-mode single_clip \
    --mask-ratio 0.75 --tube-size 16,32,32 \
    --lr 7.5e-5 --min-lr 1.5e-6 --warmup-epochs 1 \
    --weight-decay 0.05 --grad-clip 1.0 \
    --amp --amp-dtype bfloat16 --save-every 1 --seed 43 \
    --wandb-mode offline --wandb-project dl-final-videomae \
    --wandb-run-name videomae_ablation_mask75 \
    $INIT_FLAG $RESUME_FLAG \
    > videomae/artifacts/logs/videomae_mask75.out 2>&1 < /dev/null &
echo $! > videomae/artifacts/logs/videomae_mask75.pid
disown -a 2>/dev/null || true
echo "Ablation mask75 PID: $(cat videomae/artifacts/logs/videomae_mask75.pid)"
