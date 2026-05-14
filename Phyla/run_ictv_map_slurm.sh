#!/bin/bash
#SBATCH --job-name=build_ictv_map
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/ictv_map_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/ictv_map_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Build VOGDB ↔ ICTV Taxonomy Map"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

python build_ictv_map.py \
    --msl-path virus_data/ictv_msl.xlsx \
    --faa-dir virus_data/faa \
    --family-list virus_data/vogdb_treefam_v2.pickle \
    --output-map virus_data/vogdb_ictv_map.pickle \
    --output-stats eval_preds/vogdb_ictv_stats.txt

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "Date: $(date)"
echo "============================================"
