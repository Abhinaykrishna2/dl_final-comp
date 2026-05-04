#!/usr/bin/env bash
# End-to-end reproduction of training, eval, aggregates, and figures for this stream.
#
# Usage:
#   bash videomae/scripts/reproduce_main_result.sh
#
# Optional environment overrides:
#   DATA_ROOT - dataset root (default /root/data)
#   ART_ROOT  - output artifacts dir (default videomae/artifacts)
#   GPU0      - first GPU id (default 0)
#   GPU1      - second GPU id (default 1; falls back to 0 if only one GPU)
#
# Assumptions:
#   * /root/envs/physrl venv exists with PyTorch 2.7+ cu128 + the deps in
#     videomae/ENV.md installed.
#
# The script is idempotent: every step skips if its output already exists.
# Re-run after a failure and it picks up where it left off.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
source /root/envs/physrl/bin/activate

DATA="${DATA_ROOT:-/root/data}"
ART="${ART_ROOT:-videomae/artifacts}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
mkdir -p "${ART}"

echo "[repro] DATA=${DATA} ART=${ART} GPU0=${GPU0} GPU1=${GPU1}"

# 0. Dataset
if [[ ! -d "${DATA}/train" ]]; then
    echo "[repro] downloading dataset to ${DATA}"
    python data_loader.py --root "${DATA}" --workers 16
fi
python count_dataset_files.py "${DATA}"

# 1. Headline trainings
# 1a. VideoMAE main run on GPU 0 (25 epochs, mask=0.60)
if [[ ! -f "${ART}/videomae_main/encoder_best.pt" ]]; then
    echo "[repro] training VideoMAE main run on GPU ${GPU0}"
    CUDA_VISIBLE_DEVICES=${GPU0} python -m videomae.train_videomae \
        --data-root "${DATA}" --out-dir "${ART}/videomae_main" \
        --epochs 25 --batch-size 64 --num-workers 6 --prefetch-factor 2 \
        --train-index-mode sliding_window --train-stride 2 \
        --valid-index-mode single_clip \
        --mask-ratio 0.60 --tube-size 16,32,32 \
        --lr 1.5e-4 --min-lr 1.5e-6 --warmup-epochs 2 \
        --weight-decay 0.05 --grad-clip 1.0 \
        --amp --amp-dtype bfloat16 --save-every 2 --seed 42 \
        --wandb-mode offline --wandb-run-name videomae_main_mask60
fi
# 1b. Supervised baseline on GPU 1 (30 epochs)
if [[ ! -f "${ART}/supervised/encoder_best.pt" ]]; then
    echo "[repro] training supervised baseline on GPU ${GPU1}"
    CUDA_VISIBLE_DEVICES=${GPU1} python -m videomae.train_supervised \
        --data-root "${DATA}" --out-dir "${ART}/supervised" \
        --epochs 30 --batch-size 64 --num-workers 4 --prefetch-factor 2 \
        --train-index-mode sliding_window --train-stride 4 \
        --valid-index-mode single_clip \
        --lr 3e-4 --min-lr 3e-6 --warmup-epochs 1 \
        --weight-decay 0.05 --grad-clip 1.0 \
        --amp --amp-dtype bfloat16 --save-every 2 --seed 42 \
        --wandb-mode offline --wandb-run-name supervised_main
fi

# 2. Mask-ratio ablations (warm-started, 12 epochs each)
for spec in mask75:0.75:43 mask90:0.90:44; do
    name="$(echo "$spec" | cut -d: -f1)"
    ratio="$(echo "$spec" | cut -d: -f2)"
    seed="$(echo "$spec" | cut -d: -f3)"
    if [[ ! -f "${ART}/videomae_${name}/encoder_best.pt" ]]; then
        echo "[repro] training VideoMAE ${name} (warm-start) seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU0} python -m videomae.train_videomae \
            --data-root "${DATA}" --out-dir "${ART}/videomae_${name}" \
            --init-checkpoint "${ART}/videomae_main/last.pt" \
            --epochs 12 --batch-size 64 --num-workers 6 --prefetch-factor 2 \
            --train-index-mode sliding_window --train-stride 2 \
            --valid-index-mode single_clip \
            --mask-ratio "${ratio}" --tube-size 16,32,32 \
            --lr 7.5e-5 --min-lr 1.5e-6 --warmup-epochs 1 \
            --weight-decay 0.05 --grad-clip 1.0 \
            --amp --amp-dtype bfloat16 --save-every 2 --seed "${seed}" \
            --wandb-mode offline --wandb-run-name "videomae_${name}"
    fi
done

# 3. Multi-seed stability runs
for seed in 43 44; do
    if [[ ! -f "${ART}/supervised_s${seed}/encoder_best.pt" ]]; then
        echo "[repro] training supervised seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU1} python -m videomae.train_supervised \
            --data-root "${DATA}" --out-dir "${ART}/supervised_s${seed}" \
            --epochs 30 --batch-size 64 --num-workers 4 --prefetch-factor 2 \
            --train-index-mode sliding_window --train-stride 4 \
            --valid-index-mode single_clip \
            --lr 3e-4 --min-lr 3e-6 --warmup-epochs 1 \
            --weight-decay 0.05 --grad-clip 1.0 \
            --amp --amp-dtype bfloat16 --save-every 2 --seed "${seed}" \
            --wandb-mode offline --wandb-run-name "supervised_s${seed}"
    fi
    if [[ ! -f "${ART}/videomae_main_s${seed}/encoder_best.pt" ]]; then
        echo "[repro] training videomae_main seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU0} python -m videomae.train_videomae \
            --data-root "${DATA}" --out-dir "${ART}/videomae_main_s${seed}" \
            --epochs 25 --batch-size 64 --num-workers 6 --prefetch-factor 2 \
            --train-index-mode sliding_window --train-stride 2 \
            --valid-index-mode single_clip \
            --mask-ratio 0.60 --tube-size 16,32,32 \
            --lr 1.5e-4 --min-lr 1.5e-6 --warmup-epochs 2 \
            --weight-decay 0.05 --grad-clip 1.0 \
            --amp --amp-dtype bfloat16 --save-every 2 --seed "${seed}" \
            --wandb-mode offline --wandb-run-name "videomae_main_s${seed}"
    fi
done

# 3b. Second seed for the best VideoMAE config (mask=0.90)
if [[ ! -f "${ART}/videomae_mask90_s45/encoder_best.pt" ]]; then
    echo "[repro] training videomae mask=0.90 seed=45"
    CUDA_VISIBLE_DEVICES=${GPU0} python -m videomae.train_videomae \
        --data-root "${DATA}" --out-dir "${ART}/videomae_mask90_s45" \
        --init-checkpoint "${ART}/videomae_main/last.pt" \
        --epochs 12 --batch-size 64 --num-workers 6 --prefetch-factor 2 \
        --train-index-mode sliding_window --train-stride 2 \
        --valid-index-mode single_clip \
        --mask-ratio 0.90 --tube-size 16,32,32 \
        --lr 7.5e-5 --min-lr 1.5e-6 --warmup-epochs 1 \
        --weight-decay 0.05 --grad-clip 1.0 \
        --amp --amp-dtype bfloat16 --save-every 2 --seed 45 \
        --wandb-mode offline --wandb-run-name "videomae_mask90_s45"
fi

# 3c. Longer 50-epoch VideoMAE run (decoupling test)
if [[ ! -f "${ART}/videomae_main_50ep_s46/encoder_best.pt" ]]; then
    echo "[repro] training videomae 50 epochs seed=46"
    CUDA_VISIBLE_DEVICES=${GPU1} python -m videomae.train_videomae \
        --data-root "${DATA}" --out-dir "${ART}/videomae_main_50ep_s46" \
        --epochs 50 --batch-size 64 --num-workers 6 --prefetch-factor 2 \
        --train-index-mode sliding_window --train-stride 2 \
        --valid-index-mode single_clip \
        --mask-ratio 0.60 --tube-size 16,32,32 \
        --lr 1.5e-4 --min-lr 1.5e-6 --warmup-epochs 2 \
        --weight-decay 0.05 --grad-clip 1.0 \
        --amp --amp-dtype bfloat16 --save-every 2 --seed 46 \
        --wandb-mode offline --wandb-run-name videomae_main_50ep_s46
fi

# 4. Standard eval (export embeddings + linear probe sweep + kNN sweep + analysis) for every encoder
ALL_RUNS=(supervised supervised_s43 supervised_s44 \
          videomae_main videomae_main_s43 videomae_main_s44 videomae_main_50ep_s46 \
          videomae_mask75 videomae_mask90 videomae_mask90_s45)
for run in "${ALL_RUNS[@]}"; do
    [[ -f "${ART}/${run}/encoder_best.pt" ]] || { echo "[repro] skipping ${run} (no encoder_best.pt)"; continue; }
    if [[ ! -f "${ART}/${run}/embeddings/test.npz" ]]; then
        CUDA_VISIBLE_DEVICES=${GPU0} python -m videomae.export_embeddings \
            --data-root "${DATA}" --checkpoint "${ART}/${run}/encoder_best.pt" \
            --out-dir "${ART}/${run}/embeddings" \
            --split train valid test --batch-size 64 --num-workers 6 \
            --amp --amp-dtype bfloat16
    fi
    if [[ ! -f "${ART}/${run}/linear_probe/metrics.json" ]]; then
        CUDA_VISIBLE_DEVICES=${GPU0} python -m videomae.sweep_linear_probe \
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
        CUDA_VISIBLE_DEVICES=${GPU0} python -m videomae.eval_knn \
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
        CUDA_VISIBLE_DEVICES=${GPU0} python -m videomae.analyze_representations \
            --embeddings-dir "${ART}/${run}/embeddings" \
            --out-dir "${ART}/${run}/analysis" \
            --num-slices 2048
    fi
done

# 5. Channel + frame ablations on every encoder
for run in "${ALL_RUNS[@]}"; do
    [[ -f "${ART}/${run}/encoder_best.pt" ]] || continue
    if [[ ! -f "${ART}/ablations/${run}/channel_ablation.json" ]]; then
        CUDA_VISIBLE_DEVICES=${GPU0} python -m videomae.run_ablations \
            --data-root "${DATA}" \
            --encoder-checkpoints "${ART}/${run}/encoder_best.pt" \
            --encoder-labels "${run}" \
            --out-dir "${ART}/ablations" \
            --batch-size 64 --num-workers 4
    fi
done

# 6. Cross-encoder analyses (no new training)
PC_ARGS=(); CKA_ARGS=(); CSEP_ARGS=()
for run in "${ALL_RUNS[@]}"; do
    [[ -f "${ART}/${run}/embeddings/test.npz" ]] || continue
    PC_ARGS+=("${ART}/${run}:${run}")
    CKA_ARGS+=("${ART}/${run}:${run}")
    CSEP_ARGS+=("${ART}/${run}:${run}")
done
[[ ! -f "${ART}/pc_alignment/pc_alignment.json" ]] && \
    CUDA_VISIBLE_DEVICES=${GPU0} python -m videomae.pc_alignment \
        --runs "${PC_ARGS[@]}" --out-dir "${ART}/pc_alignment"
[[ ! -f "${ART}/cka/cka_valid.json" ]] && \
    python -m videomae.cka_compare \
        --runs "${CKA_ARGS[@]}" --out-dir "${ART}/cka" --split valid
[[ ! -f "${ART}/class_separability/class_separability.json" ]] && \
    python -m videomae.class_separability \
        --runs "${CSEP_ARGS[@]}" --out-dir "${ART}/class_separability"

# 7. VideoMAE-specific analyses (need decoder + mask_token, so use last.pt not encoder_best.pt)
RECON_LAST=()
RECON_LABELS=()
for run in videomae_main videomae_main_s43 videomae_mask75 videomae_mask90 videomae_mask90_s45 videomae_main_50ep_s46; do
    [[ -f "${ART}/${run}/last.pt" ]] || continue
    RECON_LAST+=("${ART}/${run}/last.pt")
    RECON_LABELS+=("${run}")
done
if [[ ${#RECON_LAST[@]} -gt 0 ]]; then
    [[ ! -f "${ART}/per_channel_recon_mse/per_channel_recon_mse.json" ]] && \
        CUDA_VISIBLE_DEVICES=${GPU0} python -m videomae.per_channel_recon_mse \
            --data-root "${DATA}" \
            --checkpoints "${RECON_LAST[@]}" --labels "${RECON_LABELS[@]}" \
            --out-dir "${ART}/per_channel_recon_mse" \
            --split valid --batch-size 32 --num-workers 4 --n-mask-realizations 3
    [[ ! -d "${ART}/recon_viz" || ! -f "${ART}/recon_viz/recon_videomae_main_sample0_t8.pdf" ]] && \
        CUDA_VISIBLE_DEVICES=${GPU0} python -m videomae.visualize_reconstructions \
            --data-root "${DATA}" \
            --checkpoints "${RECON_LAST[@]}" --labels "${RECON_LABELS[@]}" \
            --out-dir "${ART}/recon_viz" --n-samples 3 --time-step 8
fi

# 8. Aggregate + figures
python -m videomae.aggregate_results --runs "${ALL_RUNS[@]}" --out-dir "${ART}/aggregated"
python -m videomae.plot_ablations --ablations-dir "${ART}/ablations"
python -m videomae.plot_from_json \
    --runs $(for r in "${ALL_RUNS[@]}"; do [[ -f "${ART}/${r}/encoder_best.pt" ]] && echo "${ART}/${r}:${r}"; done) \
    --out-dir "${ART}/figures"

echo "[repro] done. Figures: videomae/artifacts/figures/figures.pdf"
