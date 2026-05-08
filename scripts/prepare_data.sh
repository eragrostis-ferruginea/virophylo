#!/bin/bash
set -euo pipefail

echo "=== ViroPhylo Data Preparation ==="

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
RAW_DIR="${DATA_DIR}/raw"
PROC_DIR="${DATA_DIR}/processed"
EVAL_DIR="${DATA_DIR}/eval"
SIM_DIR="${DATA_DIR}/simulated"

mkdir -p "$RAW_DIR" "$PROC_DIR"/{alignments/{train,val,test},trees/{train,val,test},distances/{train,val,test}} "$EVAL_DIR" "$SIM_DIR"

echo "--- [1/5] Downloading HIV-1 LANL data ---"
if [ ! -f "${RAW_DIR}/hiv1_pol.fasta" ]; then
    echo "  Downloading HIV-1 pol sequences from LANL..."
    curl -L "https://www.hiv.lanl.gov/content/sequence/HIV/MAP/landmark.fasta" \
        -o "${RAW_DIR}/hiv1_landmark.fasta" 2>/dev/null || \
    echo "  WARNING: LANL requires authentication. Place hiv1_pol.fasta in ${RAW_DIR}/ manually."
else
    echo "  HIV-1 data already exists, skipping."
fi

echo "--- [2/5] Downloading SARS-CoV-2 Nextstrain data ---"
if [ ! -f "${RAW_DIR}/sars2_nextstrain.fasta" ]; then
    echo "  Downloading SARS-CoV-2 from Nextstrain..."
    curl -L "https://data.nextstrain.org/files/ncov/open/global/sequences.fasta.xz" \
        -o "${RAW_DIR}/sars2_global.fasta.xz" 2>/dev/null && \
    xz -d "${RAW_DIR}/sars2_global.fasta.xz" || \
    echo "  WARNING: SARS-CoV-2 download may require GISAID access. Place data manually."
else
    echo "  SARS-CoV-2 data already exists, skipping."
fi

echo "--- [3/5] Downloading Influenza data ---"
if [ ! -f "${RAW_DIR}/influenza_ha.fasta" ]; then
    echo "  WARNING: GISAID EpiFlu data requires authentication."
    echo "  Place influenza_ha.fasta in ${RAW_DIR}/ manually."
else
    echo "  Influenza data already exists, skipping."
fi

echo "--- [4/5] Generating simulated training data ---"
echo "  Generating 5000 simulated alignments..."
python3 -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')
from src.data.simulation import SimulationDataGenerator

gen = SimulationDataGenerator('${SIM_DIR}', seed=42)
gen.generate_dataset(n_trees=5000, n_taxa_range=(10, 50), seq_length=1000)
"
echo "  Simulated data generated in ${SIM_DIR}/"

echo "--- [5/5] Processing and aligning sequences ---"
echo "  Running MAFFT alignment on raw sequences..."
for fasta in "$RAW_DIR"/*.fasta; do
    [ -f "$fasta" ] || continue
    base=$(basename "$fasta" .fasta)
    out_aln="${PROC_DIR}/alignments/train/${base}.fasta"
    if [ ! -f "$out_aln" ]; then
        echo "  Aligning $base..."
        mafft --auto --thread 4 "$fasta" > "$out_aln" 2>/dev/null || \
            cp "$fasta" "$out_aln"
    fi
done

echo "  Copying simulated data to processed directory..."
if [ -d "${SIM_DIR}/alignments/train" ]; then
    cp -n "${SIM_DIR}/alignments/train/"*.fasta "${PROC_DIR}/alignments/train/" 2>/dev/null || true
    cp -n "${SIM_DIR}/trees/train/"*.nwk "${PROC_DIR}/trees/train/" 2>/dev/null || true
    cp -n "${SIM_DIR}/distances/train/"*.npy "${PROC_DIR}/distances/train/" 2>/dev/null || true
fi
if [ -d "${SIM_DIR}/alignments/val" ]; then
    cp -n "${SIM_DIR}/alignments/val/"*.fasta "${PROC_DIR}/alignments/val/" 2>/dev/null || true
    cp -n "${SIM_DIR}/trees/val/"*.nwk "${PROC_DIR}/trees/val/" 2>/dev/null || true
    cp -n "${SIM_DIR}/distances/val/"*.npy "${PROC_DIR}/distances/val/" 2>/dev/null || true
fi
if [ -d "${SIM_DIR}/alignments/test" ]; then
    cp -n "${SIM_DIR}/alignments/test/"*.fasta "${PROC_DIR}/alignments/test/" 2>/dev/null || true
    cp -n "${SIM_DIR}/trees/test/"*.nwk "${PROC_DIR}/trees/test/" 2>/dev/null || true
    cp -n "${SIM_DIR}/distances/test/"*.npy "${PROC_DIR}/distances/test/" 2>/dev/null || true
fi

echo "  Building reference trees with IQ-TREE 2..."
for aln in "${PROC_DIR}/alignments/train/"*.fasta; do
    [ -f "$aln" ] || continue
    base=$(basename "$aln" .fasta)
    tree_out="${PROC_DIR}/trees/train/${base}.nwk"
    if [ ! -f "$tree_out" ]; then
        echo "  IQ-TREE: $base"
        iqtree2 -s "$aln" -m GTR+G -T 2 -pre "${PROC_DIR}/trees/train/${base}" -quiet -redo 2>/dev/null || \
            echo "  WARNING: IQ-TREE failed for $base"
        if [ -f "${PROC_DIR}/trees/train/${base}.treefile" ]; then
            mv "${PROC_DIR}/trees/train/${base}.treefile" "$tree_out"
        fi
    fi
done

echo "  Preparing evaluation datasets..."
for aln in "${PROC_DIR}/alignments/test/"*.fasta; do
    [ -f "$aln" ] || continue
    base=$(basename "$aln" .fasta)
    cp "$aln" "${EVAL_DIR}/${base}.fasta"
    if [ -f "${PROC_DIR}/trees/test/${base}.nwk" ]; then
        cp "${PROC_DIR}/trees/test/${base}.nwk" "${EVAL_DIR}/${base}.nwk"
    fi
done

echo ""
echo "=== Data preparation complete ==="
echo "Processed data: ${PROC_DIR}/"
echo "Evaluation data: ${EVAL_DIR}/"
echo "Simulated data: ${SIM_DIR}/"
echo ""
echo "Train alignments: $(ls "${PROC_DIR}/alignments/train/"*.fasta 2>/dev/null | wc -l)"
echo "Val alignments:   $(ls "${PROC_DIR}/alignments/val/"*.fasta 2>/dev/null | wc -l)"
echo "Test alignments:  $(ls "${PROC_DIR}/alignments/test/"*.fasta 2>/dev/null | wc -l)"
