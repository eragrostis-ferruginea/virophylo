#!/bin/bash
#SBATCH --job-name=parse_lit2
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/parse_lit2_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/parse_lit2_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"
source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

python parse_literature_refs_v2.py
