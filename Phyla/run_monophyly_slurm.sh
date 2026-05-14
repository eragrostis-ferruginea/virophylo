#!/bin/bash
#SBATCH --job-name=monophyly_eval
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/monophyly_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/monophyly_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

echo "============================================"
echo "ICTV Monophyly Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

echo ""
echo "--- Genus-level monophyly ---"
python evaluate_monophyly.py \
    --ref-pickle virus_data/vogdb_treefam_v2.pickle \
    --pred-pickle virus_data/phyla_predictions.pickle \
    --ictv-map virus_data/vogdb_ictv_map.pickle \
    --faa-dir virus_data/faa \
    --msa-dir virus_data/msa \
    --output-dir eval_preds \
    --tax-rank genus

echo ""
echo "--- Family-level monophyly ---"
python evaluate_monophyly.py \
    --ref-pickle virus_data/vogdb_treefam_v2.pickle \
    --pred-pickle virus_data/phyla_predictions.pickle \
    --ictv-map virus_data/vogdb_ictv_map.pickle \
    --faa-dir virus_data/faa \
    --msa-dir virus_data/msa \
    --output-dir eval_preds \
    --tax-rank family

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "Date: $(date)"
echo "============================================"
