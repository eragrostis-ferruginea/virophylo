#!/bin/bash
#SBATCH --job-name=virus_ref_trees
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/virus_data/slurm_ref_trees_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/virus_data/slurm_ref_trees_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Virus Reference Tree Building (FastTree)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "============================================"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo

echo "Python: $(which python)"

python build_virus_eval_dataset.py \
    --workers 16 \
    --min-seqs 4 \
    --faa-dir virus_data/faa \
    --msa-dir virus_data/msa \
    --ref-tree-dir virus_data/ref_trees \
    --output virus_data/vogdb_treefam_v2.pickle \
    --fasttree /home/jianpinhe3/virophylo/Phyla/fasttree

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "============================================"
