#!/bin/bash
#SBATCH --job-name=lit_refs_hamming
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/lit_refs_hamming_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/lit_refs_hamming_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"
source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Literature Reference Trees — CPU Baselines (Hamming/SID/Random)"
echo "Job: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "============================================================"

python evaluate_literature_refs_new.py \
    --dataset eval_preds/literature_refs_dataset.pickle \
    --output eval_preds/literature_refs_cpu_results.csv \
    --baselines hamming seqidentity random

echo "Done: $(date)"
