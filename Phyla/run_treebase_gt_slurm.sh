#!/bin/bash
#SBATCH --job-name=treebase_gt
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/treebase_gt_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/treebase_gt_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

echo "============================================"
echo "TreeBase Expert Tree Ground-Truth Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

which mafft && echo "  MAFFT: OK" || echo "  MAFFT: MISSING"

python evaluate_treebase_gt.py --output-dir eval_preds

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "Date: $(date)"
echo "============================================"
