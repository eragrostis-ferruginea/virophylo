#!/bin/bash
#SBATCH --job-name=lit_baselines
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/lit_baselines_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/lit_baselines_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

echo "============================================"
echo " Literature Baseline Evaluation (CPU)"
echo "  Hamming+MSA+NJ, SeqID+MSA+NJ, Random"
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
    --min-seqs 4 \
    --workers 16

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "Date: $(date)"
echo "============================================"