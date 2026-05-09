#!/bin/bash
#SBATCH --job-name=virophylo_process
#SBATCH --partition=cpu1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/process_%j.log
#SBATCH --error=logs/process_%j.err

echo "=== ViroPhylo Data Processing v2 (Fix: dedup, diversity, full-gene) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

module load cuda/12.1.0
source ~/.bashrc
conda activate virophylo

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

DATA_DIR="${PROJECT_DIR}/data"
RAW_DIR="${DATA_DIR}/raw"
PROC_DIR="${DATA_DIR}/processed"
EVAL_DIR="${DATA_DIR}/eval"

rm -rf "$PROC_DIR" "$EVAL_DIR"
mkdir -p "$PROC_DIR"/{alignments/{train,val,test},trees/{train,val,test},distances/{train,val,test}} "$EVAL_DIR" logs

echo ""
echo "=== Processing: deduplicate, diversity-filter, full-gene alignments ==="

python3 << 'PYEOF'
import os
import sys
import random
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

PROJECT_DIR = os.environ.get("PROJECT_DIR", os.getcwd())
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
PROC_DIR = os.path.join(PROJECT_DIR, "data", "processed")
EVAL_DIR = os.path.join(PROJECT_DIR, "data", "eval")

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(PROC_DIR, "alignments", split), exist_ok=True)
    os.makedirs(os.path.join(PROC_DIR, "trees", split), exist_ok=True)
    os.makedirs(os.path.join(PROC_DIR, "distances", split), exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

VIRAL_DATASETS = {
    "hiv1_pol": {
        "path": os.path.join(RAW_DIR, "hiv1_pol.fasta"),
        "family": "Retroviridae",
        "max_seqs": 200,
        "gene": "pol",
        "min_var_sites_frac": 0.15,
    },
    "sars2_spike": {
        "path": os.path.join(RAW_DIR, "sars2_nextstrain.fasta"),
        "family": "Coronaviridae",
        "max_seqs": 200,
        "gene": "spike",
        "min_var_sites_frac": 0.05,
    },
    "influenza_ha": {
        "path": os.path.join(RAW_DIR, "influenza_ha.fasta"),
        "family": "Orthomyxoviridae",
        "max_seqs": 200,
        "gene": "HA",
        "min_var_sites_frac": 0.15,
    },
    "dengue": {
        "path": os.path.join(RAW_DIR, "dengue.fasta"),
        "family": "Flaviviridae",
        "max_seqs": 200,
        "gene": "complete",
        "min_var_sites_frac": 0.15,
    },
    "hcv": {
        "path": os.path.join(RAW_DIR, "hcv.fasta"),
        "family": "Flaviviridae",
        "max_seqs": 150,
        "gene": "E1E2",
        "min_var_sites_frac": 0.15,
    },
    "rsv": {
        "path": os.path.join(RAW_DIR, "rsv.fasta"),
        "family": "Pneumoviridae",
        "max_seqs": 150,
        "gene": "complete",
        "min_var_sites_frac": 0.10,
    },
}

def count_variable_sites(seqs):
    if len(seqs) < 2:
        return 0
    aln_len = len(seqs[0])
    var_sites = 0
    for i in range(aln_len):
        col = set(s[i] for s in seqs if s[i] in 'ATCG')
        if len(col) > 1:
            var_sites += 1
    return var_sites

def deduplicate_sequences(records):
    seen_seqs = {}
    unique_records = []
    for r in records:
        seq_str = str(r.seq).upper().replace('-', '').replace('N', '')
        if seq_str not in seen_seqs:
            seen_seqs[seq_str] = r
            unique_records.append(r)
    return unique_records

def diversity_subsample(records, target_n, min_pairwise_dist=0.01):
    if len(records) <= target_n:
        return records
    
    seqs = [str(r.seq).upper() for r in records]
    n = len(seqs)
    
    # Compute pairwise distances (fraction of different sites, ignoring gaps/Ns)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            valid = 0
            diff = 0
            for a, b in zip(seqs[i], seqs[j]):
                if a in 'ATCG' and b in 'ATCG':
                    valid += 1
                    if a != b:
                        diff += 1
            d = diff / valid if valid > 0 else 0
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    
    # Greedy farthest-point sampling
    selected = [0]
    for _ in range(target_n - 1):
        min_dists = []
        for i in range(n):
            if i in selected:
                min_dists.append(-1)
            else:
                min_dists.append(min(dist_matrix[i][s] for s in selected))
        best = max(range(n), key=lambda i: min_dists[i])
        selected.append(best)
    
    return [records[i] for i in selected]

def load_and_subsample(fasta_path, max_seqs, min_len=100, min_var_sites_frac=0.10):
    if not os.path.exists(fasta_path) or os.path.getsize(fasta_path) == 0:
        print(f"  SKIP: {fasta_path} not found or empty")
        return None

    with open(fasta_path, 'r') as f:
        first_line = f.readline()
    
    if not first_line.startswith('>'):
        print(f"  SKIP: {fasta_path} is not a valid FASTA file")
        return None

    try:
        records = list(SeqIO.parse(fasta_path, "fasta-blast"))
    except Exception as e:
        print(f"  SKIP: {fasta_path} parse error: {str(e)[:50]}")
        return None

    valid = [r for r in records if len(str(r.seq).replace('-', '').replace('N', '')) >= min_len]
    print(f"  Loaded {len(valid)} valid sequences from {os.path.basename(fasta_path)}")

    # Step 1: Deduplicate
    before = len(valid)
    valid = deduplicate_sequences(valid)
    print(f"  After dedup: {len(valid)} (removed {before - len(valid)} identical)")

    if len(valid) < 4:
        print(f"  SKIP: only {len(valid)} unique sequences (< 4)")
        return None

    # Step 2: Diversity subsample
    if len(valid) > max_seqs:
        valid = diversity_subsample(valid, max_seqs)
        print(f"  After diversity subsample: {len(valid)}")

    # Step 3: Check variable sites
    seqs = [str(r.seq).upper() for r in valid]
    var_sites = count_variable_sites(seqs)
    aln_len = len(seqs[0]) if seqs else 1
    var_frac = var_sites / aln_len
    print(f"  Variable sites: {var_sites}/{aln_len} ({var_frac*100:.1f}%)")
    
    if var_frac < min_var_sites_frac:
        print(f"  WARNING: Low diversity ({var_frac*100:.1f}% < {min_var_sites_frac*100:.1f}%)")

    return valid

def split_dataset(records, train_ratio=0.7, val_ratio=0.15):
    n = len(records)
    indices = list(range(n))
    random.shuffle(indices)

    n_train = max(int(n * train_ratio), 1)
    n_val = max(int(n * val_ratio), 1)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": [records[i] for i in train_idx],
        "val": [records[i] for i in val_idx],
        "test": [records[i] for i in test_idx],
    }

total_train = 0
total_val = 0
total_test = 0

for name, config in VIRAL_DATASETS.items():
    print(f"\nProcessing {name} ({config['family']}, {config['gene']})...")
    records = load_and_subsample(
        config["path"], config["max_seqs"],
        min_var_sites_frac=config.get("min_var_sites_frac", 0.10)
    )
    if records is None:
        continue

    splits = split_dataset(records)

    for split_name, split_records in splits.items():
        if not split_records:
            continue

        # Use full gene alignment instead of sliding windows
        # This preserves phylogenetic signal
        out_path = os.path.join(
            PROC_DIR, "alignments", split_name,
            f"{name}.fasta"
        )
        SeqIO.write(split_records, out_path, "fasta")
        
        seqs = [str(r.seq).upper() for r in split_records]
        var_sites = count_variable_sites(seqs)
        aln_len = len(seqs[0]) if seqs else 1
        print(f"  {split_name}: {len(split_records)} seqs, {var_sites} var sites ({var_sites/aln_len*100:.1f}%)")

    for split_name in ["train", "val", "test"]:
        count = len(splits.get(split_name, []))
        if split_name == "train":
            total_train += count
        elif split_name == "val":
            total_val += count
        else:
            total_test += count

    test_records = splits.get("test", [])
    if test_records:
        eval_path = os.path.join(EVAL_DIR, f"{name}.fasta")
        SeqIO.write(test_records, eval_path, "fasta")
        print(f"  Evaluation set: {len(test_records)} seqs -> {eval_path}")

print(f"\n=== Data splitting complete ===")
print(f"Train sequences: {total_train}")
print(f"Val sequences:   {total_val}")
print(f"Test sequences:  {total_test}")
PYEOF

echo ""
echo "Building reference trees with IQ-TREE..."
for split in train val test; do
    for aln in "${PROC_DIR}/alignments/${split}"/*.fasta; do
        [ -f "$aln" ] || continue
        base=$(basename "$aln" .fasta)
        tree_out="${PROC_DIR}/trees/${split}/${base}.nwk"
        if [ ! -f "$tree_out" ]; then
            echo "  IQ-TREE: ${split}/${base}"
            iqtree -s "$aln" -m GTR+G -T 2 -pre "${PROC_DIR}/trees/${split}/${base}" -quiet -redo
            if [ -f "${PROC_DIR}/trees/${split}/${base}.treefile" ]; then
                mv "${PROC_DIR}/trees/${split}/${base}.treefile" "$tree_out"
            fi
        fi
    done
done

echo ""
echo "Computing patristic distance matrices..."
python3 << 'PYEOF'
import os, sys
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
    sys.exit(0)

for split in ["train", "val", "test"]:
    tree_dir = os.path.join(PROC_DIR, "trees", split)
    dist_dir = os.path.join(PROC_DIR, "distances", split)
    os.makedirs(dist_dir, exist_ok=True)

    if not os.path.exists(tree_dir):
        continue

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
            print(f"  {split}/{base}: {n}x{n} distance matrix, mean={dist_matrix[dist_matrix>0].mean():.6f}")
        except Exception as e:
            print(f"  ERROR {split}/{base}: {e}")

PYEOF

echo ""
echo "=== Data processing complete ==="
echo "Processed data: ${PROC_DIR}/"
echo "Evaluation data: ${EVAL_DIR}/"
echo ""
echo "Train alignments: $(ls "${PROC_DIR}/alignments/train/"*.fasta 2>/dev/null | wc -l)"
echo "Val alignments:   $(ls "${PROC_DIR}/alignments/val/"*.fasta 2>/dev/null | wc -l)"
echo "Test alignments:  $(ls "${PROC_DIR}/alignments/test/"*.fasta 2>/dev/null | wc -l)"

echo ""
echo "Processing complete: $(date)"