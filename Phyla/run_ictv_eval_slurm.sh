#!/bin/bash
#SBATCH --job-name=ictv_eval
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/ictv_eval_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/ictv_eval_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

echo "============================================"
echo "ICTV 4-Tier Virus Phylogeny Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

# Check prerequisites
for f in virus_data/vogdb_ictv_map.pickle virus_data/vogdb_treefam_v2.pickle virus_data/phyla_predictions.pickle; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found!"
        exit 1
    fi
    echo "  Found: $f ($(du -h "$f" | cut -f1))"
done

echo ""
echo "Running 3-Tier evaluation: monophyly + triplet + arbitration..."
python evaluate_ictv.py \
    --ref-pickle virus_data/vogdb_treefam_v2.pickle \
    --pred-pickle virus_data/phyla_predictions.pickle \
    --ictv-map virus_data/vogdb_ictv_map.pickle \
    --faa-dir virus_data/faa \
    --msa-dir virus_data/msa \
    --output-dir eval_preds \
    --tiers 1,2,3

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "Date: $(date)"
echo "============================================"
