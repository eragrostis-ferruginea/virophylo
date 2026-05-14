#!/bin/bash
#SBATCH --job-name=gen_report
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/report_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/report_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"
source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
python generate_report.py --csv-dir eval_preds --output-md eval_preds/evaluation_report.md
echo "Done"
