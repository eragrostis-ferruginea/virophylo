#!/bin/bash
#SBATCH --job-name=phyla_eval
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/phyla_eval_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/phyla_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu2

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

DATASET="${1:-treefam}"
DEVICE="cuda:0"

echo "============================================"
echo "PHYLA Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo "Dataset: $DATASET"
echo "Device: $DEVICE"
echo "============================================"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print('N/A')
")"

python run_eval.py --dataset "$DATASET" --device "$DEVICE"

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "============================================"
