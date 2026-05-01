#!/usr/bin/env bash
# Run the full evaluation pipeline on every encoder we have.
#
# For each encoder in artifacts/<run>/:
#   1. Export embeddings (train+valid+test) to artifacts/<run>/embeddings/
#   2. Linear probe sweep -> artifacts/<run>/linear_probe/
#   3. kNN sweep -> artifacts/<run>/knn/
#   4. Analysis suite -> artifacts/<run>/analysis/
#
# Idempotent: each step skips if its output exists.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/envs/physrl/bin/activate

DATA="${DATA_ROOT:-/root/data}"
ART="${ART_ROOT:-videomae/artifacts}"
RUNS=("${@:-videomae_main supervised}")
if [[ $# -eq 0 ]]; then
    # Auto-detect runs that have an encoder_best.pt
    RUNS=()
    for d in "${ART}"/*/; do
        run=$(basename "$d")
        if [[ -f "$d/encoder_best.pt" ]]; then
            RUNS+=("$run")
        fi
    done
fi

GPU="${CUDA_VISIBLE_DEVICES:-0}"

for run in "${RUNS[@]}"; do
    if [[ ! -f "${ART}/${run}/encoder_best.pt" ]]; then
        echo "[eval] skipping ${run}: no encoder_best.pt"
        continue
    fi
    echo "[eval] === ${run} on GPU ${GPU} ==="
    if [[ ! -f "${ART}/${run}/embeddings/test.npz" ]]; then
        CUDA_VISIBLE_DEVICES=${GPU} python -m videomae.export_embeddings \
            --data-root "${DATA}" \
            --checkpoint "${ART}/${run}/encoder_best.pt" \
            --out-dir "${ART}/${run}/embeddings" \
            --split train valid test --batch-size 64 --num-workers 8 \
            --amp --amp-dtype bfloat16
    else
        echo "[eval] ${run} embeddings already exist"
    fi
    if [[ ! -f "${ART}/${run}/linear_probe/metrics.json" ]]; then
        CUDA_VISIBLE_DEVICES=${GPU} python -m videomae.sweep_linear_probe \
            --train-file "${ART}/${run}/embeddings/train.npz" \
            --valid-file "${ART}/${run}/embeddings/valid.npz" \
            --test-file  "${ART}/${run}/embeddings/test.npz" \
            --out-dir "${ART}/${run}/linear_probe" \
            --epochs 200 --patience 25 \
            --feature-norms zscore zscore_l2 l2 \
            --lrs 3e-4 1e-3 3e-3 \
            --weight-decays 0 1e-5 1e-4 1e-3 \
            --batch-sizes 0 1024 4096 \
            --wandb-mode offline --wandb-run-name "linprobe-${run}"
    else
        echo "[eval] ${run} linear probe already done"
    fi
    if [[ ! -f "${ART}/${run}/knn/metrics.json" ]]; then
        CUDA_VISIBLE_DEVICES=${GPU} python -m videomae.eval_knn \
            --train-file "${ART}/${run}/embeddings/train.npz" \
            --valid-file "${ART}/${run}/embeddings/valid.npz" \
            --test-file  "${ART}/${run}/embeddings/test.npz" \
            --out-dir "${ART}/${run}/knn" \
            --neighbors 1 3 5 10 20 50 \
            --weights uniform distance \
            --metric cosine euclidean \
            --feature-norm zscore zscore_l2 l2 \
            --wandb-mode offline --wandb-run-name "knn-${run}"
    else
        echo "[eval] ${run} kNN already done"
    fi
    if [[ ! -f "${ART}/${run}/analysis/analysis.json" ]]; then
        CUDA_VISIBLE_DEVICES=${GPU} python -m videomae.analyze_representations \
            --embeddings-dir "${ART}/${run}/embeddings" \
            --out-dir "${ART}/${run}/analysis" \
            --num-slices 2048
    else
        echo "[eval] ${run} analysis already done"
    fi
done
echo "[eval] all done."
