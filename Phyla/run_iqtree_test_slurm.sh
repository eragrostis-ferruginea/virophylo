#!/bin/bash
#SBATCH --job-name=iqtree_test
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/iqtree_test_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/iqtree_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu1

cd /home/jianpinhe3/virophylo/Phyla
source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo

echo "Testing IQ-TREE on VFAM00037"
python run_iqtree_single.py --fam VFAM00037 --msa-dir virus_data/msa --output-dir virus_data/iqtree_trees
