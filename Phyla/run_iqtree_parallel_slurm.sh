#!/bin/bash
#SBATCH --job-name=iqtree_ref
#SBATCH --output=/home/jianpinhe3/virophylo/Phyla/slurm_logs/iqtree_parallel_%j.out
#SBATCH --error=/home/jianpinhe3/virophylo/Phyla/slurm_logs/iqtree_parallel_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --partition=cpu1
#SBATCH --cpus-per-task=30

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
export PYTHONUNBUFFERED=1

echo "IQ-TREE parallel batch"
echo "Host: $(hostname)  CPUs: $(nproc)  Date: $(date)"

# Use xargs -P: 30 concurrent IQ-TREE processes (1 thread each, ~200MB)
cat virus_data/iqtree_family_list.txt | \
  xargs -I{} -P 30 \
    python run_iqtree_single.py \
      --fam {} \
      --msa-dir virus_data/msa \
      --output-dir virus_data/iqtree_trees

RC=$?
echo "Exit code: $RC"
COUNT=$(ls -1 virus_data/iqtree_trees/*.iqtree.nwk 2>/dev/null | wc -l)
echo "Trees: $COUNT / $(wc -l < virus_data/iqtree_family_list.txt)"
echo "Date: $(date)"
