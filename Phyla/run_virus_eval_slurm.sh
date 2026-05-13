#!/bin/bash
#SBATCH --job-name=phyla_virus_eval
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/virus_eval_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/virus_eval_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu2

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

DEVICE="cuda:0"
PICKLE="${1:-virus_data/vogdb_treefam_v2.pickle}"
OUTPUT="${2:-eval_preds/virus_results.csv}"

echo "============================================"
echo "PHYLA Virus Phylogeny Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Pickle: $PICKLE"
echo "Output: $OUTPUT"
echo "============================================"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo

echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print('N/A')
")"

python run_virus_eval.py --pickle "$PICKLE" --output-csv "$OUTPUT" --device "$DEVICE"

RC=$?
echo "============================================"
echo "Done with exit code: $RC"
echo "============================================"
