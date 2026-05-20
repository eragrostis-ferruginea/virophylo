#!/bin/bash
#SBATCH --job-name=lit_phyla
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/lit_phyla_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/lit_phyla_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

echo "============================================"
echo " PHYLA-beta on Literature Reference Trees"
echo "  Brown & Firth 2025 RdRp — 303 OTU trees"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

python evaluate_literature_phyla.py \
    --dataset /home/jianpinhe3/virophylo/virus_data/literature_refs/literature_dataset.pickle \
    --output-dir /home/jianpinhe3/virophylo/Phyla/eval_preds/literature \
    --checkpoint weights/11564369 \
    --device cuda:0 \
    --min-seqs 4

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "Date: $(date)"
echo "============================================"