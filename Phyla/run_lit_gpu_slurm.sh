#!/bin/bash
#SBATCH --job-name=lit_gpu
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/lit_gpu_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/lit_gpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1

cd /home/jianpinhe3/virophylo/Phyla
source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

echo "PHYLA + ESM2 on Literature Reference Trees"
echo "Job: $SLURM_JOB_ID  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Date: $(date)"

python evaluate_literature_gpu.py \
    --dataset eval_preds/literature_eval_dataset.pickle \
    --output eval_preds/literature_gpu_results.csv \
    --model both \
    --device cuda:0

echo "Date: $(date)"
echo "Done: $?"
