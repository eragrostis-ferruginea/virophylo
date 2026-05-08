#!/bin/bash
set -euo pipefail

echo "=== ViroPhylo Real Data Preparation (No Simulated Data) ==="

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
RAW_DIR="${DATA_DIR}/raw"
PROC_DIR="${DATA_DIR}/processed"
EVAL_DIR="${DATA_DIR}/eval"

mkdir -p "$RAW_DIR" "$PROC_DIR"/{alignments/{train,val,test},trees/{train,val,test},distances/{train,val,test}} "$EVAL_DIR"

echo ""
echo "=== Data Sources ==="
echo "All data must be REAL viral genome sequences from public databases."
echo "No simulated/fake data will be used."
echo ""

###############################################################################
# 1. HIV-1 pol from LANL HIV Database
###############################################################################
echo "--- [1/8] HIV-1 pol (LANL) ---"
if [ ! -f "${RAW_DIR}/hiv1_pol.fasta" ]; then
    echo "  Downloading HIV-1 pol reference alignment from LANL..."
    curl -L "https://www.hiv.lanl.gov/content/sequence/HIV/MAP/landmark.fasta" \
        -o "${RAW_DIR}/hiv1_landmark.fasta" 2>/dev/null || true

    curl -L "https://hivdb.stanford.edu/pages/data/geno/rx-all.fasta" \
        -o "${RAW_DIR}/hiv1_pol.fasta" 2>/dev/null || true

    if [ ! -s "${RAW_DIR}/hiv1_pol.fasta" ]; then
        echo "  WARNING: HIV-1 LANL requires web interface download."
        echo "  Go to https://www.hiv.lanl.gov/content/sequence/HIV/mainpage.html"
        echo "  Download pol region alignment (FASTA) and place as:"
        echo "    ${RAW_DIR}/hiv1_pol.fasta"
    else
        echo "  HIV-1 pol downloaded: $(grep -c '^>' ${RAW_DIR}/hiv1_pol.fasta 2>/dev/null || echo '?') sequences"
    fi
else
    echo "  HIV-1 pol already exists: $(grep -c '^>' ${RAW_DIR}/hiv1_pol.fasta 2>/dev/null || echo '?') sequences"
fi

###############################################################################
# 2. SARS-CoV-2 from Nextstrain (open data, no GISAID needed)
###############################################################################
echo "--- [2/8] SARS-CoV-2 (Nextstrain open data) ---"
if [ ! -f "${RAW_DIR}/sars2_nextstrain.fasta" ]; then
    echo "  Downloading SARS-CoV-2 from Nextstrain open data..."
    curl -L "https://data.nextstrain.org/files/ncov/open/global/sequences.fasta.xz" \
        -o "${RAW_DIR}/sars2_nextstrain.fasta.xz" 2>/dev/null && \
    xz -d "${RAW_DIR}/sars2_nextstrain.fasta.xz" 2>/dev/null && \
    mv "${RAW_DIR}/sars2_nextstrain.fasta" "${RAW_DIR}/sars2_nextstrain.fasta" 2>/dev/null || \
    echo "  Trying alternative: Nextstrain metadata + sequences..."

    if [ ! -s "${RAW_DIR}/sars2_nextstrain.fasta" ]; then
        curl -L "https://data.nextstrain.org/files/ncov/open/global/dna_sequences.fasta.xz" \
            -o "${RAW_DIR}/sars2_nextstrain.fasta.xz" 2>/dev/null && \
        xz -d "${RAW_DIR}/sars2_nextstrain.fasta.xz" 2>/dev/null || true
    fi

    if [ ! -s "${RAW_DIR}/sars2_nextstrain.fasta" ]; then
        echo "  WARNING: SARS-CoV-2 download failed."
        echo "  Download manually from https://nextstrain.org/ncov/open/global"
        echo "  or from https://github.com/nextstrain/ncov"
        echo "  Place as: ${RAW_DIR}/sars2_nextstrain.fasta"
    else
        echo "  SARS-CoV-2 downloaded: $(grep -c '^>' ${RAW_DIR}/sars2_nextstrain.fasta 2>/dev/null || echo '?') sequences"
    fi
else
    echo "  SARS-CoV-2 already exists: $(grep -c '^>' ${RAW_DIR}/sars2_nextstrain.fasta 2>/dev/null || echo '?') sequences"
fi

###############################################################################
# 3. Influenza A HA from NCBI/IVR (public, no GISAID needed)
###############################################################################
echo "--- [3/8] Influenza A HA (NCBI IVR) ---"
if [ ! -f "${RAW_DIR}/influenza_ha.fasta" ]; then
    echo "  Downloading Influenza A H3N2 HA from NCBI IVR..."
    curl -L "https://ftp.ncbi.nih.gov/genomes/INFLUENZA/influenza.fna" \
        -o "${RAW_DIR}/influenza_all.fna" 2>/dev/null || true

    if [ -s "${RAW_DIR}/influenza_all.fna" ]; then
        echo "  Extracting HA sequences..."
        python3 -c "
from Bio import SeqIO
import sys
ha_seqs = []
for rec in SeqIO.parse('${RAW_DIR}/influenza_all.fna', 'fasta'):
    desc = rec.description.upper()
    if 'HEMAGGLUTININ' in desc or 'HAEMAGGLUTININ' in desc or ' HA ' in desc:
        ha_seqs.append(rec)
SeqIO.write(ha_seqs, '${RAW_DIR}/influenza_ha.fasta', 'fasta')
print(f'  Extracted {len(ha_seqs)} HA sequences')
" 2>/dev/null || echo "  WARNING: HA extraction failed"
    fi

    if [ ! -s "${RAW_DIR}/influenza_ha.fasta" ]; then
        echo "  WARNING: Influenza data download failed."
        echo "  Download manually from https://www.ncbi.nlm.nih.gov/genomes/FLU/FLU.html"
        echo "  or GISAID EpiFlu (requires registration)."
        echo "  Place as: ${RAW_DIR}/influenza_ha.fasta"
    else
        echo "  Influenza HA: $(grep -c '^>' ${RAW_DIR}/influenza_ha.fasta 2>/dev/null || echo '?') sequences"
    fi
else
    echo "  Influenza HA already exists: $(grep -c '^>' ${RAW_DIR}/influenza_ha.fasta 2>/dev/null || echo '?') sequences"
fi

###############################################################################
# 4. Dengue Virus from ViPR (public)
###############################################################################
echo "--- [4/8] Dengue Virus (ViPR) ---"
if [ ! -f "${RAW_DIR}/dengue.fasta" ]; then
    echo "  Downloading Dengue virus complete genomes from ViPR..."
    curl -L "https://www.viprbrc.org/brc/downloadSequence.spg?datatype=genome&family=dengue&subfamily=&species=all&country=all&segment=all&host=all&collectionDate=all&genotype=all&sequenceLength=complete" \
        -o "${RAW_DIR}/dengue.fasta" 2>/dev/null || true

    if [ ! -s "${RAW_DIR}/dengue.fasta" ]; then
        echo "  WARNING: Dengue download failed."
        echo "  Download manually from https://www.viprbrc.org/brc/home.spg?decorator=flavi_dengue"
        echo "  Place as: ${RAW_DIR}/dengue.fasta"
    else
        echo "  Dengue: $(grep -c '^>' ${RAW_DIR}/dengue.fasta 2>/dev/null || echo '?') sequences"
    fi
else
    echo "  Dengue already exists: $(grep -c '^>' ${RAW_DIR}/dengue.fasta 2>/dev/null || echo '?') sequences"
fi

###############################################################################
# 5. HCV from LANL/EuHCVdb
###############################################################################
echo "--- [5/8] HCV (LANL/EuHCVdb) ---"
if [ ! -f "${RAW_DIR}/hcv.fasta" ]; then
    echo "  Downloading HCV from LANL..."
    curl -L "https://hcv.lanl.gov/content/sequence/HCV/alignments/genotype/consensus/E1E2.fasta" \
        -o "${RAW_DIR}/hcv_e1e2.fasta" 2>/dev/null || true

    if [ ! -s "${RAW_DIR}/hcv_e1e2.fasta" ]; then
        echo "  WARNING: HCV download failed."
        echo "  Download manually from https://hcv.lanl.gov/content/sequence/HCV/mainpage.html"
        echo "  or from https://euhcvdb.ibcp.fr/"
        echo "  Place as: ${RAW_DIR}/hcv.fasta"
    else
        cp "${RAW_DIR}/hcv_e1e2.fasta" "${RAW_DIR}/hcv.fasta"
        echo "  HCV: $(grep -c '^>' ${RAW_DIR}/hcv.fasta 2>/dev/null || echo '?') sequences"
    fi
else
    echo "  HCV already exists: $(grep -c '^>' ${RAW_DIR}/hcv.fasta 2>/dev/null || echo '?') sequences"
fi

###############################################################################
# 6. RSV from NCBI GenBank
###############################################################################
echo "--- [6/8] RSV (NCBI GenBank) ---"
if [ ! -f "${RAW_DIR}/rsv.fasta" ]; then
    echo "  Downloading RSV complete genomes from NCBI..."
    esearch -db nucleotide -query "Human respiratory syncytial virus[Organism] AND complete genome[Title]" 2>/dev/null | \
        efetch -format fasta > "${RAW_DIR}/rsv.fasta" 2>/dev/null || true

    if [ ! -s "${RAW_DIR}/rsv.fasta" ]; then
        echo "  WARNING: RSV download failed (requires NCBI E-utilities)."
        echo "  Download manually from NCBI GenBank."
        echo "  Place as: ${RAW_DIR}/rsv.fasta"
    else
        echo "  RSV: $(grep -c '^>' ${RAW_DIR}/rsv.fasta 2>/dev/null || echo '?') sequences"
    fi
else
    echo "  RSV already exists: $(grep -c '^>' ${RAW_DIR}/rsv.fasta 2>/dev/null || echo '?') sequences"
fi

###############################################################################
# 7. TreeBASE reference trees (real phylogenies from published studies)
###############################################################################
echo "--- [7/8] TreeBASE reference alignments ---"
if [ ! -f "${RAW_DIR}/treebase_manifest.txt" ]; then
    echo "  Downloading TreeBASE study list..."
    curl -L "https://treebase.org/treebase-web/download/studyList.csv" \
        -o "${RAW_DIR}/treebase_studies.csv" 2>/dev/null || true

    echo "  TreeBASE data requires manual download of individual studies."
    echo "  Recommended studies with viral phylogenies:"
    echo "    - S2197 (HIV-1): https://treebase.org/treebase-web/search/study/summary.html?id=2197"
    echo "    - S1164 (Flavivirus): https://treebase.org/treebase-web/search/study/summary.html?id=1164"
    echo "    - S1577 (Influenza): https://treebase.org/treebase-web/search/study/summary.html?id=1577"
    echo "  Download alignments + trees and place in ${RAW_DIR}/treebase/"
    mkdir -p "${RAW_DIR}/treebase"
fi

###############################################################################
# 8. Process all raw data: subsample, align, build reference trees, split
###############################################################################
echo "--- [8/8] Processing real data: subsample, align, build reference trees ---"

python3 << 'PYEOF'
import os
import sys
import random
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

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
        "max_seqs": 500,
        "gene": "pol",
    },
    "sars2_spike": {
        "path": os.path.join(RAW_DIR, "sars2_nextstrain.fasta"),
        "family": "Coronaviridae",
        "max_seqs": 500,
        "gene": "spike",
    },
    "influenza_ha": {
        "path": os.path.join(RAW_DIR, "influenza_ha.fasta"),
        "family": "Orthomyxoviridae",
        "max_seqs": 500,
        "gene": "HA",
    },
    "dengue": {
        "path": os.path.join(RAW_DIR, "dengue.fasta"),
        "family": "Flaviviridae",
        "max_seqs": 300,
        "gene": "complete",
    },
    "hcv": {
        "path": os.path.join(RAW_DIR, "hcv.fasta"),
        "family": "Flaviviridae",
        "max_seqs": 200,
        "gene": "E1E2",
    },
    "rsv": {
        "path": os.path.join(RAW_DIR, "rsv.fasta"),
        "family": "Pneumoviridae",
        "max_seqs": 200,
        "gene": "complete",
    },
}

def load_and_subsample(fasta_path, max_seqs, min_len=100):
    if not os.path.exists(fasta_path) or os.path.getsize(fasta_path) == 0:
        print(f"  SKIP: {fasta_path} not found or empty")
        return None, None

    records = list(SeqIO.parse(fasta_path, "fasta"))
    valid = [r for r in records if len(str(r.seq).replace('-', '').replace('N', '')) >= min_len]
    print(f"  Loaded {len(valid)} valid sequences from {os.path.basename(fasta_path)}")

    if len(valid) > max_seqs:
        valid = random.sample(valid, max_seqs)
        print(f"  Subsampled to {max_seqs}")

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

def create_sliding_windows(records, window_size=500, stride=250, min_seqs_per_window=4):
    all_windows = []
    seqs = [str(r.seq).upper() for r in records]
    names = [r.id for r in records]
    aln_len = len(seqs[0]) if seqs else 0

    for start in range(0, aln_len - window_size + 1, stride):
        end = start + window_size
        window_seqs = [s[start:end] for s in seqs]
        gap_frac = [s.count('-') / len(s) for s in window_seqs]
        valid = [(n, s) for n, s, gf in zip(names, window_seqs, gap_frac) if gf < 0.5]
        if len(valid) >= min_seqs_per_window:
            window_records = [SeqRecord(Seq(s), id=n, description="") for n, s in valid]
            all_windows.append(window_records)

    return all_windows

total_train = 0
total_val = 0
total_test = 0

for name, config in VIRAL_DATASETS.items():
    print(f"\nProcessing {name} ({config['family']}, {config['gene']})...")
    records = load_and_subsample(config["path"], config["max_seqs"])
    if records is None:
        continue

    splits = split_dataset(records)

    for split_name, split_records in splits.items():
        if not split_records:
            continue

        if len(split_records) >= 10:
            windows = create_sliding_windows(split_records, window_size=500, stride=250)
            print(f"  {split_name}: {len(split_records)} seqs -> {len(windows)} windows")

            for w_idx, window_records in enumerate(windows):
                out_path = os.path.join(
                    PROC_DIR, "alignments", split_name,
                    f"{name}_win{w_idx:04d}.fasta"
                )
                SeqIO.write(window_records, out_path, "fasta")
        else:
            out_path = os.path.join(
                PROC_DIR, "alignments", split_name,
                f"{name}.fasta"
            )
            SeqIO.write(split_records, out_path, "fasta")
            windows = []

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

treebase_dir = os.path.join(RAW_DIR, "treebase")
if os.path.exists(treebase_dir):
    print(f"\nProcessing TreeBASE data from {treebase_dir}...")
    for f in sorted(os.listdir(treebase_dir)):
        if f.endswith('.fasta') or f.endswith('.fa'):
            aln_path = os.path.join(treebase_dir, f)
            base = f.rsplit('.', 1)[0]
            records = list(SeqIO.parse(aln_path, "fasta"))
            if len(records) >= 4:
                out_path = os.path.join(PROC_DIR, "alignments", "train", f"treebase_{base}.fasta")
                SeqIO.write(records, out_path, "fasta")
                print(f"  TreeBASE {base}: {len(records)} seqs")

                tree_file = base + '.nwk'
                tree_path = os.path.join(treebase_dir, tree_file)
                if os.path.exists(tree_path):
                    import shutil
                    shutil.copy(tree_path, os.path.join(PROC_DIR, "trees", "train", f"treebase_{base}.nwk"))

print(f"\n=== Data splitting complete ===")
print(f"Train sequences: {total_train}")
print(f"Val sequences:   {total_val}")
print(f"Test sequences:  {total_test}")
PYEOF

echo ""
echo "Building reference trees with IQ-TREE 2..."
for split in train val test; do
    for aln in "${PROC_DIR}/alignments/${split}"/*.fasta; do
        [ -f "$aln" ] || continue
        base=$(basename "$aln" .fasta)
        tree_out="${PROC_DIR}/trees/${split}/${base}.nwk"
        if [ ! -f "$tree_out" ]; then
            echo "  IQ-TREE: ${split}/${base}"
            iqtree2 -s "$aln" -m GTR+G+ASC -T 2 -pre "${PROC_DIR}/trees/${split}/${base}" -quiet -redo 2>/dev/null || \
                echo "  WARNING: IQ-TREE failed for ${base}"
            if [ -f "${PROC_DIR}/trees/${split}/${base}.treefile" ]; then
                mv "${PROC_DIR}/trees/${split}/${base}.treefile" "$tree_out"
            fi
        fi
    done
done

echo ""
echo "Computing patristic distance matrices from reference trees..."
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
    print("  Distance matrices will be computed on-the-fly during training")
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
            print(f"  {split}/{base}: {n}x{n} distance matrix")
        except Exception as e:
            print(f"  ERROR {split}/{base}: {e}")

PYEOF

echo ""
echo "=== Real data preparation complete ==="
echo "Processed data: ${PROC_DIR}/"
echo "Evaluation data: ${EVAL_DIR}/"
echo ""
echo "Train alignments: $(ls "${PROC_DIR}/alignments/train/"*.fasta 2>/dev/null | wc -l)"
echo "Val alignments:   $(ls "${PROC_DIR}/alignments/val/"*.fasta 2>/dev/null | wc -l)"
echo "Test alignments:  $(ls "${PROC_DIR}/alignments/test/"*.fasta 2>/dev/null | wc -l)"
echo ""
echo "IMPORTANT: If any datasets show 0 alignments, place raw FASTA files"
echo "in ${RAW_DIR}/ and re-run this script."
