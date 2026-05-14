#!/bin/bash
#SBATCH --job-name=phyla_esm2_eval
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/esm2_eval_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/esm2_eval_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

echo "============================================"
echo "ESM2-650M + NJ on VOGDB Virus"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Date: $(date)"
echo "============================================"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo

python run_esm2_eval.py \
    --ref-pickle virus_data/vogdb_treefam_v2.pickle \
    --faa-dir virus_data/faa \
    --output-csv eval_preds/virus_esm2_vs_fasttree.csv \
    --checkpoint eval_preds/virus_esm2_checkpoint.pickle \
    --batch-size 8

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "Date: $(date)"
echo "============================================"