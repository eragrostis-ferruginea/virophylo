#!/bin/bash
set -euo pipefail

echo "=== ViroPhylo Data Download (Login Node Only) ==="
echo "Run this on login node where internet is available"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
RAW_DIR="${DATA_DIR}/raw"

mkdir -p "$RAW_DIR"

echo "=== Data Sources ==="
echo "All data must be REAL viral genome sequences from public databases."
echo "No simulated/fake data will be used."
echo ""

echo "--- [1/6] HIV-1 pol (LANL) ---"
if [ ! -f "${RAW_DIR}/hiv1_pol.fasta" ]; then
    echo "  Downloading HIV-1 pol from Stanford HIVDB..."
    curl -L "https://hivdb.stanford.edu/pages/data/geno/rx-all.fasta" \
        -o "${RAW_DIR}/hiv1_pol.fasta"

    if [ ! -s "${RAW_DIR}/hiv1_pol.fasta" ]; then
        echo "  WARNING: HIV-1 download failed."
        echo "  Download manually from https://www.hiv.lanl.gov/content/sequence/HIV/mainpage.html"
        echo "  Place as: ${RAW_DIR}/hiv1_pol.fasta"
    else
        echo "  HIV-1 pol downloaded: $(grep -c '^>' ${RAW_DIR}/hiv1_pol.fasta) sequences"
    fi
else
    echo "  HIV-1 pol already exists: $(grep -c '^>' ${RAW_DIR}/hiv1_pol.fasta) sequences"
fi

echo ""
echo "--- [2/6] SARS-CoV-2 (Nextstrain open data) ---"
if [ ! -f "${RAW_DIR}/sars2_nextstrain.fasta" ]; then
    echo "  Downloading SARS-CoV-2 from Nextstrain..."
    curl -L "https://data.nextstrain.org/files/ncov/open/global/dna_sequences.fasta.xz" \
        -o "${RAW_DIR}/sars2_nextstrain.fasta.xz" && \
    xz -d "${RAW_DIR}/sars2_nextstrain.fasta.xz"

    if [ ! -s "${RAW_DIR}/sars2_nextstrain.fasta" ]; then
        echo "  WARNING: SARS-CoV-2 download failed."
        echo "  Download manually from https://nextstrain.org/ncov/open/global"
        echo "  Place as: ${RAW_DIR}/sars2_nextstrain.fasta"
    else
        echo "  SARS-CoV-2 downloaded: $(grep -c '^>' ${RAW_DIR}/sars2_nextstrain.fasta) sequences"
    fi
else
    echo "  SARS-CoV-2 already exists: $(grep -c '^>' ${RAW_DIR}/sars2_nextstrain.fasta) sequences"
fi

echo ""
echo "--- [3/6] Influenza A HA (NCBI IVR) ---"
if [ ! -f "${RAW_DIR}/influenza_ha.fasta" ]; then
    echo "  Downloading Influenza from NCBI..."
    curl -L "https://ftp.ncbi.nih.gov/genomes/INFLUENZA/influenza.fna" \
        -o "${RAW_DIR}/influenza_all.fna"

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
"
    fi

    if [ ! -s "${RAW_DIR}/influenza_ha.fasta" ]; then
        echo "  WARNING: Influenza data download failed."
        echo "  Download manually from https://www.ncbi.nlm.nih.gov/genomes/FLU/FLU.html"
        echo "  Place as: ${RAW_DIR}/influenza_ha.fasta"
    else
        echo "  Influenza HA: $(grep -c '^>' ${RAW_DIR}/influenza_ha.fasta) sequences"
    fi
else
    echo "  Influenza HA already exists: $(grep -c '^>' ${RAW_DIR}/influenza_ha.fasta) sequences"
fi

echo ""
echo "--- [4/6] Dengue Virus (ViPR) ---"
if [ ! -f "${RAW_DIR}/dengue.fasta" ]; then
    echo "  Downloading Dengue virus..."
    curl -L "https://www.viprbrc.org/brc/downloadSequence.spg?datatype=genome&family=dengue&subfamily=&species=all&country=all&segment=all&host=all&collectionDate=all&genotype=all&sequenceLength=complete" \
        -o "${RAW_DIR}/dengue.fasta"

    if [ ! -s "${RAW_DIR}/dengue.fasta" ]; then
        echo "  WARNING: Dengue download failed."
        echo "  Download manually from https://www.viprbrc.org/brc/home.spg?decorator=flavi_dengue"
        echo "  Place as: ${RAW_DIR}/dengue.fasta"
    else
        echo "  Dengue: $(grep -c '^>' ${RAW_DIR}/dengue.fasta) sequences"
    fi
else
    echo "  Dengue already exists: $(grep -c '^>' ${RAW_DIR}/dengue.fasta) sequences"
fi

echo ""
echo "--- [5/6] HCV (LANL/EuHCVdb) ---"
if [ ! -f "${RAW_DIR}/hcv.fasta" ]; then
    echo "  Downloading HCV..."
    curl -L "https://hcv.lanl.gov/content/sequence/HCV/alignments/genotype/consensus/E1E2.fasta" \
        -o "${RAW_DIR}/hcv_e1e2.fasta"

    if [ -s "${RAW_DIR}/hcv_e1e2.fasta" ]; then
        cp "${RAW_DIR}/hcv_e1e2.fasta" "${RAW_DIR}/hcv.fasta"
        echo "  HCV: $(grep -c '^>' ${RAW_DIR}/hcv.fasta) sequences"
    else
        echo "  WARNING: HCV download failed."
        echo "  Download manually from https://hcv.lanl.gov/content/sequence/HCV/mainpage.html"
        echo "  Place as: ${RAW_DIR}/hcv.fasta"
    fi
else
    echo "  HCV already exists: $(grep -c '^>' ${RAW_DIR}/hcv.fasta) sequences"
fi

echo ""
echo "--- [6/6] RSV (NCBI GenBank) ---"
if [ ! -f "${RAW_DIR}/rsv.fasta" ]; then
    echo "  Downloading RSV..."
    if command -v esearch &> /dev/null; then
        esearch -db nucleotide -query "Human respiratory syncytial virus[Organism] AND complete genome[Title]" | \
            efetch -format fasta > "${RAW_DIR}/rsv.fasta"
    fi

    if [ ! -s "${RAW_DIR}/rsv.fasta" ]; then
        echo "  WARNING: RSV download failed."
        echo "  Download manually from NCBI GenBank"
        echo "  Place as: ${RAW_DIR}/rsv.fasta"
    else
        echo "  RSV: $(grep -c '^>' ${RAW_DIR}/rsv.fasta) sequences"
    fi
else
    echo "  RSV already exists: $(grep -c '^>' ${RAW_DIR}/rsv.fasta) sequences"
fi

echo ""
echo "=== Download complete ==="
echo "Raw data location: ${RAW_DIR}/"
echo ""
echo "Next step: Submit processing job to compute node:"
echo "  bash scripts/submit_process.sh"