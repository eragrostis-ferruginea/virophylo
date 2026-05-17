#!/bin/bash
#SBATCH --job-name=full_lit_mafft
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/full_lit_mafft_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/full_lit_mafft_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"
source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Full Literature Evaluation — 318 MAFFT-aligned trees"
echo "Job: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "============================================================"

python evaluate_full_literature.py \
    --data-dir /home/jianpinhe3/virophylo/virus_data/literature_refs \
    --output eval_preds/literature_full_results_mafft.csv

echo ""
echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
