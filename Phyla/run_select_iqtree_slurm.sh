#!/bin/bash
#SBATCH --job-name=select_iq
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/select_iq_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/select_iq_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpu1

cd /home/jianpinhe3/virophylo/Phyla
source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo

python select_iqtree_families.py
