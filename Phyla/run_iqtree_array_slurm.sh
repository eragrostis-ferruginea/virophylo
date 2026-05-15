#!/bin/bash
#SBATCH --job-name=iqtree_ref
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/iqtree_ref_%A_%a.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/iqtree_ref_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu1
#SBATCH --array=1-1013%40

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

# Get the family ID for this array task
FAM=$(sed -n "${SLURM_ARRAY_TASK_ID}p" virus_data/iqtree_family_list.txt)
echo "Task: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "Family: $FAM"
echo "Node: $(hostname)"
echo "Date: $(date)"

python run_iqtree_single.py \
    --fam "$FAM" \
    --msa-dir virus_data/msa \
    --output-dir virus_data/iqtree_trees \
    --threads 4

RC=$?
echo "Exit code: $RC"
echo "Date: $(date)"
