#!/bin/bash
#SBATCH --job-name=virophylo_route_a
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/route_a_%j.log
#SBATCH --error=logs/route_a_%j.err

echo "=== ViroPhylo Route A: PhyloGPN Paradigm Transfer to Viruses ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

module load cuda/12.1.0
source ~/.bashrc
conda activate virophylo

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

DATA_DIR="${PROJECT_DIR}/data/processed"
OUTPUT_DIR="${PROJECT_DIR}/outputs/route_a"
mkdir -p "$OUTPUT_DIR" logs

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

echo "--- Training ViralPhyloGPN (4x RTX4080) ---"
python src/training/route_a_train.py configs/train/route_a_pretrain.yaml

echo "Route A training complete: $(date)"
