#!/bin/bash
#SBATCH --job-name=lit_mafft_cpu
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/lit_mafft_cpu_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/lit_mafft_cpu_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"
source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Literature MAFFT Evaluation — CPU Baselines"
echo "Job: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "============================================================"

python evaluate_literature_mafft.py \
    --output eval_preds/literature_mafft_results.csv

echo ""
echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
