#!/usr/bin/env bash
# Reproduce the videomae stream's main result end-to-end.
#
# Usage:
#   bash videomae/scripts/reproduce_main_result.sh
#
# Assumes:
#   * /root/envs/physrl venv exists with PyTorch 2.7+ cu128 + the deps in
#     videomae/ENV.md installed.
#   * /root/data contains the active_matter splits (run data_loader.py first).
#
# Idempotent: each step skips if its output already exists. Re-run after a
# failure and it picks up where it left off.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
source /root/envs/physrl/bin/activate

DATA="${DATA_ROOT:-/root/data}"
ART="${ART_ROOT:-videomae/artifacts}"

mkdir -p "${ART}"

if [[ ! -d "${DATA}/train" ]]; then
    echo "[repro] downloading dataset to ${DATA}"
    python data_loader.py --root "${DATA}" --workers 16
fi

python count_dataset_files.py "${DATA}"

# 1. Train VideoMAE main run on GPU 0
if [[ ! -f "${ART}/videomae_main/encoder_best.pt" ]]; then
    echo "[repro] training VideoMAE main run on GPU 0"
    CUDA_VISIBLE_DEVICES=0 python -m videomae.train_videomae \
        --data-root "${DATA}" --out-dir "${ART}/videomae_main" \
        --epochs 25 --batch-size 64 --num-workers 12 --prefetch-factor 4 \
        --train-index-mode sliding_window --train-stride 2 \
        --valid-index-mode single_clip \
        --mask-ratio 0.60 --tube-size 16,32,32 \
        --lr 1.5e-4 --min-lr 1.5e-6 --warmup-epochs 2 \
        --weight-decay 0.05 --grad-clip 1.0 \
        --amp --amp-dtype bfloat16 --save-every 2 --seed 42 \
        --wandb-mode offline --wandb-run-name videomae_main_mask60
fi

# 2. Train supervised baseline on GPU 1 (or GPU 0 if only one available)
if [[ ! -f "${ART}/supervised/encoder_best.pt" ]]; then
    echo "[repro] training supervised baseline"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} python -m videomae.train_supervised \
        --data-root "${DATA}" --out-dir "${ART}/supervised" \
        --epochs 30 --batch-size 64 --num-workers 8 --prefetch-factor 4 \
        --train-index-mode sliding_window --train-stride 4 \
        --valid-index-mode single_clip \
        --lr 3e-4 --min-lr 3e-6 --warmup-epochs 1 \
        --weight-decay 0.05 --grad-clip 1.0 \
        --amp --amp-dtype bfloat16 --save-every 2 --seed 42 \
        --wandb-mode offline --wandb-run-name supervised_main
fi

# 3. Mask-ratio ablation: warm-start from main, train at 0.75 for 12 more epochs
if [[ ! -f "${ART}/videomae_mask75/encoder_best.pt" ]]; then
    echo "[repro] training VideoMAE mask=0.75 ablation (warm-start)"
    CUDA_VISIBLE_DEVICES=0 python -m videomae.train_videomae \
        --data-root "${DATA}" --out-dir "${ART}/videomae_mask75" \
        --init-checkpoint "${ART}/videomae_main/last.pt" \
        --epochs 12 --batch-size 64 --num-workers 12 --prefetch-factor 4 \
        --train-index-mode sliding_window --train-stride 2 \
        --valid-index-mode single_clip \
        --mask-ratio 0.75 --tube-size 16,32,32 \
        --lr 7.5e-5 --min-lr 1.5e-6 --warmup-epochs 1 \
        --weight-decay 0.05 --grad-clip 1.0 \
        --amp --amp-dtype bfloat16 --save-every 2 --seed 43 \
        --wandb-mode offline --wandb-run-name videomae_ablation_mask75
fi

# 4. Export embeddings + run probe + kNN + analysis on each encoder
for run in videomae_main videomae_mask75 supervised; do
    if [[ ! -f "${ART}/${run}/encoder_best.pt" ]]; then
        echo "[repro] skipping eval for ${run}: encoder_best.pt not found"
        continue
    fi
    if [[ ! -f "${ART}/${run}/embeddings/test.npz" ]]; then
        echo "[repro] exporting embeddings for ${run}"
        python -m videomae.export_embeddings \
            --data-root "${DATA}" \
            --checkpoint "${ART}/${run}/encoder_best.pt" \
            --out-dir "${ART}/${run}/embeddings" \
            --split train valid test --batch-size 64 --num-workers 8 \
            --amp --amp-dtype bfloat16
    fi
    if [[ ! -f "${ART}/${run}/linear_probe/metrics.json" ]]; then
        echo "[repro] linear probe sweep for ${run}"
        python -m videomae.sweep_linear_probe \
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
    fi
    if [[ ! -f "${ART}/${run}/knn/metrics.json" ]]; then
        echo "[repro] kNN sweep for ${run}"
        python -m videomae.eval_knn \
            --train-file "${ART}/${run}/embeddings/train.npz" \
            --valid-file "${ART}/${run}/embeddings/valid.npz" \
            --test-file  "${ART}/${run}/embeddings/test.npz" \
            --out-dir "${ART}/${run}/knn" \
            --neighbors 1 3 5 10 20 50 \
            --weights uniform distance \
            --metric cosine euclidean \
            --feature-norm zscore zscore_l2 l2 \
            --wandb-mode offline --wandb-run-name "knn-${run}"
    fi
    if [[ ! -f "${ART}/${run}/analysis/analysis.json" ]]; then
        echo "[repro] analysis suite for ${run}"
        python -m videomae.analyze_representations \
            --embeddings-dir "${ART}/${run}/embeddings" \
            --out-dir "${ART}/${run}/analysis" \
            --num-slices 2048
    fi
done

# 5. Generate the joint figures pdf
python -m videomae.plot_from_json \
    --runs "${ART}/videomae_main:VideoMAE-mask60" \
           "${ART}/videomae_mask75:VideoMAE-mask75" \
           "${ART}/supervised:Supervised" \
    --out-dir "${ART}/figures"

echo "[repro] done."
