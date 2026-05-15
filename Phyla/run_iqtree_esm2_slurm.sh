#!/bin/bash
#SBATCH --job-name=esm2_iqtree
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/esm2_iqtree_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/esm2_iqtree_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"
source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

echo "ESM2-650M vs IQ-TREE reference"
echo "Job: $SLURM_JOB_ID  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Date: $(date)"

python run_iqtree_esm2.py --device cuda:0

echo "Done: $?  Date: $(date)"
