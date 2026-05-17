#!/bin/bash
#SBATCH --job-name=prep_lit_mafft
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/prep_lit_mafft_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/prep_lit_mafft_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"
source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Prepare Literature Reference Dataset (MAFFT-aligned)"
echo "Job: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "============================================================"

python prepare_literature_dataset_mafft.py \
    --output eval_preds/literature_refs_dataset.pickle

echo ""
echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
