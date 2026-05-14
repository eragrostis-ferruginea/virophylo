#!/bin/bash
#SBATCH --job-name=phyla_treebase
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/phyla_treebase_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/phyla_treebase_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

echo "============================================"
echo "PHYLA on TreeBase Ground-Truth Virus Families"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Date: $(date)"
echo "============================================"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

python run_treebase_phyla.py \
    --checkpoint weights/11564369 \
    --device cuda:0

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "Date: $(date)"
echo "============================================"
