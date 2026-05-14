#!/bin/bash
#SBATCH --job-name=ictv_map
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/ictv_map_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/ictv_map_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

echo "============================================"
echo "ICTV Taxonomy Mapping"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "============================================"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

python build_ictv_mapping.py

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "Date: $(date)"
echo "============================================"
