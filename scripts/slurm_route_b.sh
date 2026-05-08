#!/bin/bash
#SBATCH --job-name=virophylo_route_b
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/route_b_%j.log
#SBATCH --error=logs/route_b_%j.err

echo "=== ViroPhylo Route B: PHYLA + Phylogenetic Likelihood Dual Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

module load cuda/12.1.0
source ~/.bashrc
conda activate virophylo

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

DATA_DIR="${PROJECT_DIR}/data/processed"
OUTPUT_DIR="${PROJECT_DIR}/outputs/route_b"
mkdir -p "$OUTPUT_DIR" logs

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "--- Training PHYLAViral with Dual Loss (4x A100) ---"
torchrun --nproc_per_node=4 \
    src/training/route_b_train.py \
    --config configs/train/route_b_dual.yaml \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --bf16 \
    --seed 42

echo "Route B training complete: $(date)"
