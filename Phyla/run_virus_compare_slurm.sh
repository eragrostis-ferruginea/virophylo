#!/bin/bash
#SBATCH --job-name=phyla_virus_compare
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/virus_compare_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/virus_compare_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

echo "============================================"
echo "PHYLA vs FastTree Virus Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "============================================"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo

python compare_virus_trees.py \
    --ref-pickle virus_data/vogdb_treefam_v2.pickle \
    --pred-pickle virus_data/phyla_predictions.pickle \
    --output-csv eval_preds/virus_phyla_vs_fasttree.csv \
    --output-pickle virus_data/virus_eval_full.pickle

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "============================================"