#!/bin/bash
#SBATCH --job-name=parse_lit
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/parse_lit_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/parse_lit_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

echo "Parsing literature reference data..."
echo "Date: $(date)"

python parse_literature_refs.py --output eval_preds/literature_eval_dataset.pickle

echo "Date: $(date)"
echo "Done: $?"
