#!/bin/bash
#SBATCH --job-name=lit_refs_gpu
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/lit_refs_gpu_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/lit_refs_gpu_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"
source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Literature Reference Trees — GPU Evaluation (PHYLA + ESM2)"
echo "Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Date: $(date)"
echo "============================================================"

# Run PHYLA evaluation
echo ""
echo "--- PHYLA Evaluation ---"
python evaluate_literature_refs_new.py \
    --dataset eval_preds/literature_refs_dataset.pickle \
    --output eval_preds/literature_refs_phyla_results.csv \
    --baselines phyla \
    --device cuda:0

echo ""
echo "--- ESM2 Evaluation ---"
python evaluate_literature_refs_new.py \
    --dataset eval_preds/literature_refs_dataset.pickle \
    --output eval_preds/literature_refs_esm2_results.csv \
    --baselines esm2 \
    --device cuda:0

echo ""
echo "Done: $(date)"
