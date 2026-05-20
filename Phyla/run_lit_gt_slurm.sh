#!/bin/bash
#SBATCH --job-name=lit_gt_eval
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/lit_gt_eval_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/lit_gt_eval_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Literature Reference Tree Ground-Truth Eval"
echo "  Brown & Firth 2025 RdRp — 303 OTU trees"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

which mafft && echo "  MAFFT: OK" || echo "  MAFFT: MISSING"

python evaluate_literature_gt.py \
    --dataset /home/jianpinhe3/virophylo/virus_data/literature_refs/literature_dataset.pickle \
    --output-dir /home/jianpinhe3/virophylo/Phyla/eval_preds/literature \
    --min-seqs 4

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "Date: $(date)"
echo "============================================"