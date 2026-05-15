#!/bin/bash
#SBATCH --job-name=prep_lit_refs
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/prep_lit_refs_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/prep_lit_refs_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"
source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Prepare Literature Reference Dataset"
echo "Job: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "============================================================"

python prepare_literature_dataset.py \
    --output eval_preds/literature_refs_dataset.pickle

echo ""
echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
