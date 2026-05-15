#!/bin/bash
#SBATCH --job-name=iqtree_eval
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/iqtree_eval_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/iqtree_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

echo "IQ-TREE Reference Evaluation"
echo "Date: $(date)"

python evaluate_iqtree_ref.py \
    --ref-pickle virus_data/vogdb_treefam_v2.pickle \
    --pred-pickle virus_data/phyla_predictions.pickle \
    --iqtree-dir virus_data/iqtree_trees \
    --family-list virus_data/iqtree_family_list.txt \
    --msa-dir virus_data/msa \
    --output-dir eval_preds

echo "Date: $(date)"
echo "Done: $?"
