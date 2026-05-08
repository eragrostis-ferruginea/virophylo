#!/bin/bash
#SBATCH --job-name=virophylo_baseline
#SBATCH --partition=cpu1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/baseline_%j.log
#SBATCH --error=logs/baseline_%j.err

echo "=== ViroPhylo: Traditional Baseline Evaluation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source ~/.bashrc
conda activate virophylo

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EVAL_DIR="${PROJECT_DIR}/data/eval"
OUTPUT_DIR="${PROJECT_DIR}/outputs/baselines"
mkdir -p "$OUTPUT_DIR" logs

echo "--- Running IQ-TREE 2 on evaluation datasets ---"
for aln in "$EVAL_DIR"/*.fasta; do
    [ -f "$aln" ] || continue
    base=$(basename "$aln" .fasta)
    echo "  IQ-TREE: $base"
    iqtree2 -s "$aln" -m GTR+G -T AUTO -pre "${OUTPUT_DIR}/${base}_iqtree" -redo 2>/dev/null || \
        echo "  IQ-TREE not available for $base"
done

echo "--- Running FastTree ---"
for aln in "$EVAL_DIR"/*.fasta; do
    [ -f "$aln" ] || continue
    base=$(basename "$aln" .fasta)
    echo "  FastTree: $base"
    FastTree -gtr -nt "$aln" > "${OUTPUT_DIR}/${base}_fasttree.nwk" 2>/dev/null || \
        fasttree -gtr -nt "$aln" > "${OUTPUT_DIR}/${base}_fasttree.nwk" 2>/dev/null || \
        echo "  FastTree not available for $base"
done

echo "--- Running K2P + NJ baseline (Python) ---"
python3 -c "
import os, sys
sys.path.insert(0, '${PROJECT_DIR}')
from src.models.distance.k2p_baseline import compute_k2p_matrix
from src.models.tree.nj_builder import nj_from_distance_matrix
from Bio import SeqIO

eval_dir = '${EVAL_DIR}'
out_dir = '${OUTPUT_DIR}'

for f in sorted(os.listdir(eval_dir)):
    if not f.endswith('.fasta'):
        continue
    base = f.rsplit('.', 1)[0]
    seqs, names = [], []
    for rec in SeqIO.parse(os.path.join(eval_dir, f), 'fasta'):
        seqs.append(str(rec.seq).upper())
        names.append(rec.id)
    if len(seqs) < 4:
        continue
    dist = compute_k2p_matrix(seqs)
    newick = nj_from_distance_matrix(dist, names)
    with open(os.path.join(out_dir, f'{base}_k2p_nj.nwk'), 'w') as out:
        out.write(newick)
    print(f'  K2P+NJ: {base}')
"

echo "Baseline evaluation complete: $(date)"
