#!/bin/bash
#SBATCH --job-name=phyla_groundtruth
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/groundtruth_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/groundtruth_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Ground-Truth Species Clustering Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "============================================"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

python evaluate_groundtruth.py \
    --ref-pickle virus_data/vogdb_treefam_v2.pickle \
    --pred-pickle virus_data/phyla_predictions.pickle \
    --faa-dir virus_data/faa \
    --msa-dir virus_data/msa \
    --output-dir eval_preds

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "Date: $(date)"
echo "============================================"
