#!/bin/bash
#SBATCH --job-name=virophylo_trees
#SBATCH --partition=cpu1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/trees_%j.log
#SBATCH --error=logs/trees_%j.err

echo "=== ViroPhylo: Building Reference Trees ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

module load cuda/12.1.0
source ~/.bashrc
conda activate virophylo

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

PROC_DIR="${PROJECT_DIR}/data/processed"

echo ""
echo "Building reference trees with IQ-TREE..."
for split in train val test; do
    aln_dir="${PROC_DIR}/alignments/${split}"
    tree_dir="${PROC_DIR}/trees/${split}"
    
    mkdir -p "$tree_dir"
    
    if [ ! -d "$aln_dir" ]; then
        echo "  SKIP: $aln_dir not found"
        continue
    fi
    
    n_trees=0
    for aln in "${aln_dir}"/*.fasta; do
        [ -f "$aln" ] || continue
        base=$(basename "$aln" .fasta)
        tree_out="${tree_dir}/${base}.nwk"
        
        if [ -f "$tree_out" ]; then
            echo "  EXISTS: ${split}/${base}"
            continue
        fi
        
        echo "  IQ-TREE: ${split}/${base}"
        iqtree -s "$aln" -m GTR+G -T 2 -pre "${tree_dir}/${base}" -quiet -redo 2>/dev/null
        
        if [ -f "${tree_dir}/${base}.treefile" ]; then
            mv "${tree_dir}/${base}.treefile" "$tree_out"
            n_trees=$((n_trees + 1))
        else
            echo "  WARNING: Tree not generated for ${base}"
        fi
    done
    echo "  Generated $n_trees trees for $split"
done

echo ""
echo "=== Computing distance matrices ==="
python3 << 'PYEOF'
import os
import numpy as np

PROJECT_DIR = os.environ.get("PROJECT_DIR", os.getcwd())
PROC_DIR = os.path.join(PROJECT_DIR, "data", "processed")

try:
    import dendropy
    HAS_DENDROPY = True
except ImportError:
    HAS_DENDROPY = False

if not HAS_DENDROPY:
    print("  dendropy not available, skipping distance matrix computation")
else:
    for split in ["train", "val", "test"]:
        tree_dir = os.path.join(PROC_DIR, "trees", split)
        dist_dir = os.path.join(PROC_DIR, "distances", split)
        os.makedirs(dist_dir, exist_ok=True)
        
        if not os.path.exists(tree_dir):
            continue
        
        n_matrices = 0
        for f in sorted(os.listdir(tree_dir)):
            if not f.endswith('.nwk'):
                continue
            base = f.rsplit('.', 1)[0]
            dist_path = os.path.join(dist_dir, base + '.npy')
            
            if os.path.exists(dist_path):
                continue
            
            tree_path = os.path.join(tree_dir, f)
            try:
                tree = dendropy.Tree.get(path=tree_path, schema="newick")
                pdm = tree.phylogenetic_distance_matrix()
                taxa = list(tree.taxon_namespace)
                n = len(taxa)
                dist_matrix = np.zeros((n, n))
                
                for i, t1 in enumerate(taxa):
                    for j, t2 in enumerate(taxa):
                        if i < j:
                            try:
                                d = pdm(t1, t2)
                                dist_matrix[i, j] = d
                                dist_matrix[j, i] = d
                            except:
                                pass
                
                np.save(dist_path, dist_matrix)
                n_matrices += 1
            except Exception as e:
                print(f"  ERROR {split}/{base}: {e}")
        
        print(f"  Generated {n_matrices} distance matrices for {split}")

PYEOF

echo ""
echo "=== Summary ==="
for split in train val test; do
    n_trees=$(ls "${PROC_DIR}/trees/${split}"/*.nwk 2>/dev/null | wc -l)
    n_dist=$(ls "${PROC_DIR}/distances/${split}"/*.npy 2>/dev/null | wc -l)
    echo "  $split: $n_trees trees, $n_dist distance matrices"
done

echo ""
echo "Complete: $(date)"