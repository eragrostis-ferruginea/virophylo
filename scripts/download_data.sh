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

ncbi_esearch_efetch() {
    local db="$1"
    local term="$2"
    local output="$3"
    local retmax="${4:-500}"

    local search_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    local fetch_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    local search_result
    search_result=$(curl -s "${search_url}?db=${db}&term=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${term}'))")&retmax=${retmax}&usehistory=y" || true)

    if [ -z "$search_result" ]; then
        echo "  ERROR: NCBI esearch returned empty result"
        return 1
    fi

    local query_key web_env count
    query_key=$(echo "$search_result" | python3 -c "import sys, xml.etree.ElementTree as ET; print(ET.fromstring(sys.stdin.read()).findtext('QueryKey',''))" 2>/dev/null || echo "")
    web_env=$(echo "$search_result" | python3 -c "import sys, xml.etree.ElementTree as ET; print(ET.fromstring(sys.stdin.read()).findtext('WebEnv',''))" 2>/dev/null || echo "")
    count=$(echo "$search_result" | python3 -c "import sys, xml.etree.ElementTree as ET; print(ET.fromstring(sys.stdin.read()).findtext('Count','0'))" 2>/dev/null || echo "0")

    if [ -z "$query_key" ] || [ -z "$web_env" ]; then
        echo "  ERROR: Could not parse NCBI search results"
        return 1
    fi

    echo "  Found ${count} sequences, downloading up to ${retmax}..."

    local batch_size=200
    local tmp_files=()
    for start in $(seq 0 $batch_size $((retmax - 1))); do
        local tmp_file="${output}.part.${start}"
        curl -s "${fetch_url}?db=${db}&query_key=${query_key}&WebEnv=${web_env}&rettype=fasta&retmode=text&retmax=${batch_size}&retstart=${start}" \
            -o "$tmp_file" || true
        if [ -s "$tmp_file" ]; then
            tmp_files+=("$tmp_file")
        fi
        sleep 0.5
    done

    cat "${tmp_files[@]}" > "$output" 2>/dev/null || true
    rm -f "${tmp_files[@]}" 2>/dev/null || true

    if [ -s "$output" ] && head -1 "$output" | grep -q '^>'; then
        local seq_count
        seq_count=$(grep -c '^>' "$output" 2>/dev/null || echo 0)
        echo "  Downloaded ${seq_count} sequences"
        return 0
    else
        echo "  ERROR: Downloaded file is not valid FASTA"
        rm -f "$output"
        return 1
    fi
}

echo "--- [1/6] HIV-1 pol (NCBI) ---"
if [ ! -f "${RAW_DIR}/hiv1_pol.fasta" ] || [ "$(grep -c '^>' "${RAW_DIR}/hiv1_pol.fasta" 2>/dev/null || echo 0)" -eq 0 ]; then
    echo "  Downloading HIV-1 pol from NCBI..."
    ncbi_esearch_efetch "nuccore" \
        '"HIV-1"[Organism] AND pol[Gene] AND 1000:6000[SLEN]' \
        "${RAW_DIR}/hiv1_pol.fasta" 500 || true

    if [ ! -s "${RAW_DIR}/hiv1_pol.fasta" ]; then
        echo "  WARNING: HIV-1 download failed."
        echo "  Download manually from https://www.hiv.lanl.gov/content/sequence/HIV/mainpage.html"
        echo "  Place as: ${RAW_DIR}/hiv1_pol.fasta"
    else
        echo "  HIV-1 pol downloaded: $(grep -c '^>' ${RAW_DIR}/hiv1_pol.fasta 2>/dev/null || echo 0) sequences"
    fi
else
    echo "  HIV-1 pol already exists: $(grep -c '^>' ${RAW_DIR}/hiv1_pol.fasta 2>/dev/null || echo 0) sequences"
fi

echo ""
echo "--- [2/6] SARS-CoV-2 (Nextstrain open data) ---"
if [ ! -f "${RAW_DIR}/sars2_nextstrain.fasta" ]; then
    echo "  Downloading SARS-CoV-2 from Nextstrain..."
    curl -L "https://data.nextstrain.org/files/ncov/open/global/dna_sequences.fasta.xz" \
        -o "${RAW_DIR}/sars2_nextstrain.fasta.xz" && \
    xz -d "${RAW_DIR}/sars2_nextstrain.fasta.xz" || true

    if [ ! -s "${RAW_DIR}/sars2_nextstrain.fasta" ]; then
        echo "  WARNING: SARS-CoV-2 download failed."
        echo "  Download manually from https://nextstrain.org/ncov/open/global"
        echo "  Place as: ${RAW_DIR}/sars2_nextstrain.fasta"
    else
        echo "  SARS-CoV-2 downloaded: $(grep -c '^>' ${RAW_DIR}/sars2_nextstrain.fasta 2>/dev/null || echo 0) sequences"
    fi
else
    echo "  SARS-CoV-2 already exists: $(grep -c '^>' ${RAW_DIR}/sars2_nextstrain.fasta 2>/dev/null || echo 0) sequences"
fi

echo ""
echo "--- [3/6] Influenza A HA (NCBI IVR) ---"
if [ ! -f "${RAW_DIR}/influenza_ha.fasta" ]; then
    echo "  Downloading Influenza from NCBI..."
    curl -L "https://ftp.ncbi.nih.gov/genomes/INFLUENZA/influenza.fna" \
        -o "${RAW_DIR}/influenza_all.fna" || true

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
" || true
    fi

    if [ ! -s "${RAW_DIR}/influenza_ha.fasta" ]; then
        echo "  WARNING: Influenza data download failed."
        echo "  Download manually from https://www.ncbi.nlm.nih.gov/genomes/FLU/FLU.html"
        echo "  Place as: ${RAW_DIR}/influenza_ha.fasta"
    else
        echo "  Influenza HA: $(grep -c '^>' ${RAW_DIR}/influenza_ha.fasta 2>/dev/null || echo 0) sequences"
    fi
else
    echo "  Influenza HA already exists: $(grep -c '^>' ${RAW_DIR}/influenza_ha.fasta 2>/dev/null || echo 0) sequences"
fi

echo ""
echo "--- [4/6] Dengue Virus (NCBI) ---"
if [ ! -f "${RAW_DIR}/dengue.fasta" ] || [ "$(grep -c '^>' "${RAW_DIR}/dengue.fasta" 2>/dev/null || echo 0)" -eq 0 ]; then
    echo "  Downloading Dengue virus from NCBI..."
    ncbi_esearch_efetch "nuccore" \
        '"Dengue virus"[Organism] AND "complete genome"[Title] AND 9000:12000[SLEN]' \
        "${RAW_DIR}/dengue.fasta" 500 || true

    if [ ! -s "${RAW_DIR}/dengue.fasta" ]; then
        echo "  WARNING: Dengue download failed."
        echo "  Download manually from https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/virus?VirusLineage_ss=Dengue%20virus"
        echo "  Place as: ${RAW_DIR}/dengue.fasta"
    else
        echo "  Dengue: $(grep -c '^>' ${RAW_DIR}/dengue.fasta 2>/dev/null || echo 0) sequences"
    fi
else
    echo "  Dengue already exists: $(grep -c '^>' ${RAW_DIR}/dengue.fasta 2>/dev/null || echo 0) sequences"
fi

echo ""
echo "--- [5/6] HCV (NCBI) ---"
if [ ! -f "${RAW_DIR}/hcv.fasta" ] || [ "$(grep -c '^>' "${RAW_DIR}/hcv.fasta" 2>/dev/null || echo 0)" -eq 0 ]; then
    echo "  Downloading HCV from NCBI..."
    ncbi_esearch_efetch "nuccore" \
        '"Hepatitis C virus"[Organism] AND "complete genome"[Title] AND 9000:12000[SLEN]' \
        "${RAW_DIR}/hcv.fasta" 500 || true

    if [ ! -s "${RAW_DIR}/hcv.fasta" ]; then
        echo "  WARNING: HCV download failed."
        echo "  Download manually from https://hcv.lanl.gov/content/sequence/HCV/mainpage.html"
        echo "  Place as: ${RAW_DIR}/hcv.fasta"
    else
        echo "  HCV: $(grep -c '^>' ${RAW_DIR}/hcv.fasta 2>/dev/null || echo 0) sequences"
    fi
else
    echo "  HCV already exists: $(grep -c '^>' ${RAW_DIR}/hcv.fasta 2>/dev/null || echo 0) sequences"
fi

echo ""
echo "--- [6/6] RSV (NCBI) ---"
if [ ! -f "${RAW_DIR}/rsv.fasta" ] || [ "$(grep -c '^>' "${RAW_DIR}/rsv.fasta" 2>/dev/null || echo 0)" -eq 0 ]; then
    echo "  Downloading RSV from NCBI..."
    ncbi_esearch_efetch "nuccore" \
        '"Human respiratory syncytial virus"[Organism] AND "complete genome"[Title]' \
        "${RAW_DIR}/rsv.fasta" 500 || true

    if [ ! -s "${RAW_DIR}/rsv.fasta" ]; then
        echo "  WARNING: RSV download failed."
        echo "  Download manually from NCBI GenBank"
        echo "  Place as: ${RAW_DIR}/rsv.fasta"
    else
        echo "  RSV: $(grep -c '^>' ${RAW_DIR}/rsv.fasta 2>/dev/null || echo 0) sequences"
    fi
else
    echo "  RSV already exists: $(grep -c '^>' ${RAW_DIR}/rsv.fasta 2>/dev/null || echo 0) sequences"
fi

echo ""
echo "=== Download complete ==="
echo "Raw data location: ${RAW_DIR}/"
echo ""
echo "Next step: Submit processing job to compute node:"
echo "  bash scripts/submit_process.sh"
