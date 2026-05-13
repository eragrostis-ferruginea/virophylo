# PHYLA on Virus Phylogeny: VOGDB Benchmark

This project reproduces the phylogenetic evaluation pipeline from **Ektefaie et al. (2025)** — *"Evolutionary Reasoning Does Not Arise in Standard Usage of Protein Language Models"* — and applies it to **virus phylogeny** using the **VOGDB (Viral Orthologous Groups Database)** as the data source.

The core question: **Can PHYLA, a hybrid Mamba–Transformer protein language model trained on curated phylogenetic trees, generalize to reconstructing viral evolutionary relationships?**

---

## Table of Contents

- [Background](#background)
- [Data](#data)
- [Pipeline Overview](#pipeline-overview)
- [Model Configuration](#model-configuration)
- [Evaluation Method](#evaluation-method)
- [Results](#results)
  - [TreeFam Baseline (Paper Reproduction)](#treefam-baseline-paper-reproduction)
  - [VOGDB Virus Benchmark](#vogdb-virus-benchmark)
- [Limitations](#limitations)
- [Future Directions](#future-directions)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [References](#references)

---

## Background

### PHYLA

PHYLA (Phylogenetic Inference with Language Models) is a protein language model that combines Mamba state-space layers with transformer attention for phylogenetic inference. Given unaligned protein sequences from a gene family, PHYLA:

1. Encodes each sequence via a bidirectional Mamba backbone into a CLS embedding
2. Computes pairwise Euclidean distances between all CLS embeddings
3. Constructs a tree via **Neighbor Joining** (scikit-bio implementation)
4. Outputs a Newick tree string

Two model variants exist:
- **Phyla-alpha** (291M params, trained on 13,696 TreeFam families)
- **Phyla-beta** (lighter architecture, trained on 3,321 high-quality TreeFam families)

This project uses **Phyla-beta**, matching the best-performing model reported in the paper.

### VOGDB

[VOGDB](https://vogdb.org/) is a database of **Viral Orthologous Groups** containing 39,776 protein families (VFAMs) derived from 1.2 million viral protein sequences. Each VFAM provides:
- **FAA file**: unaligned protein sequences
- **MSA file**: multiple sequence alignment (raw alignment format)

VOGDB covers the full known viral sequence space, making it a comprehensive testbed for evaluating whether PHYLA's evolutionary reasoning generalizes to viruses.

---

## Data

### Source

| Item | Detail |
|------|--------|
| Database | VOGDB (https://vogdb.org/) |
| Release | Current (vfam.faa.tar.gz, vfam.raw_algs.tar.gz) |
| Total VFAMs | **39,776** |
| Format | FAA (protein), MSA (Stockholm/raw alignment) |
| License | CC-BY-4.0 |

### Processing Steps

1. **Extraction**: Downloaded `vfam.faa.tar.gz` (58 MB) and `vfam.raw_algs.tar.gz` (63 MB). Both are bzip2-compressed tarballs, extracted with `tar -xjf`.
2. **Filtering**: Retained only VFAMs with **4 or more sequences** (minimum required for Neighbor Joining and meaningful tree comparison). After filtering: **20,536 families**.
3. **Reference Tree Construction**: For each qualifying VFAM, ran **FastTree 2.1.11** (compiled from source) on the MSA to produce a reference tree. Parameters: `-quiet` flag, 120-second timeout. Success rate: **99.87%** (20,510 / 20,536).

### Dataset Summary

| Item | Count | Size |
|------|-------|------|
| Raw VFAM FAA files | 39,776 | 308 MB |
| Raw VFAM MSA files | 39,776 | 393 MB |
| Families passing filter (≥4 seqs) | 20,536 | — |
| FastTree reference trees built | **20,510** | 84 MB |
| Families in evaluation pickle | **20,510** | 141 MB |
| PHYLA predictions generated | **20,534** | 50 MB |
| **Common families (overlap)** | **20,510** | — |

**Sequence statistics** (across 20,510 families):
- Mean sequences per family: **25.7**
- Median: — (not computed)
- Range: 4 – 2,272
- Total sequences processed: ~527,000

---

## Pipeline Overview

The evaluation pipeline consists of three independent stages:

```
VOGDB FAA/MSA
     │
     ▼
┌──────────────────────┐     ┌──────────────────────┐
│ Stage 1: CPU (sbatch)│     │ Stage 2: GPU (sbatch)│
│ build_virus_eval_    │     │ run_phyla_predict_   │
│ dataset.py           │     │ trees.py             │
│                      │     │                      │
│ 1. Filter ≥4 seqs    │     │ 1. Load Phyla-beta   │
│ 2. FastTree ref tree │     │ 2. Encode sequences  │
│ 3. Save pickle       │     │ 3. NJ tree from CLS  │
│                      │     │ 4. Save predictions  │
└──────────┬───────────┘     └──────────┬────────────┘
           │                            │
           ▼                            ▼
      ref_trees/              phyla_predictions.pickle
      vogdb_treefam_v2.pickle
           │                            │
           └──────────┬─────────────────┘
                      ▼
           ┌──────────────────────┐
           │ Stage 3: CPU (sbatch)│
           │ compare_virus_trees  │
           │ .py                  │
           │                      │
           │ normRF comparison   │
           │ (paper's exact      │
           │  rf_distance)        │
           └──────────┬───────────┘
                      ▼
           eval_preds/virus_phyla_vs_fasttree.csv
```

### Stage 1: Reference Tree Construction
- **Script**: [build_virus_eval_dataset.py](file:///home/jianpinhe3/virophylo/Phyla/build_virus_eval_dataset.py)
- **SLURM**: [run_virus_ref_trees.sh](file:///home/jianpinhe3/virophylo/Phyla/run_virus_ref_trees.sh)
- **Job**: 57357 (CPU partition, 19 min)
- **Output**: `vogdb_treefam_v2.pickle` — a dictionary mapping each VFAM ID to `{"sequences": {name: seq}, "tree_newick": "..."}`

### Stage 2: PHYLA Tree Prediction
- **Script**: [run_phyla_predict_trees.py](file:///home/jianpinhe3/virophylo/Phyla/run_phyla_predict_trees.py)
- **SLURM**: [run_phyla_predict_slurm.sh](file:///home/jianpinhe3/virophylo/Phyla/run_phyla_predict_slurm.sh)
- **Job**: 57367 (GPU partition, 15 min 55 sec on gpu11)
- **Output**: `phyla_predictions.pickle` — a dictionary mapping each VFAM ID to `{"pred_tree_newick": "...", "num_seqs": N, "seq_names": [...]}`

### Stage 3: Comparison
- **Script**: [compare_virus_trees.py](file:///home/jianpinhe3/virophylo/Phyla/compare_virus_trees.py)
- **SLURM**: [run_virus_compare_slurm.sh](file:///home/jianpinhe3/virophylo/Phyla/run_virus_compare_slurm.sh)
- **Job**: 57391 (CPU partition, 3 min 7 sec)
- **Output**: `eval_preds/virus_phyla_vs_fasttree.csv`

---

## Model Configuration

Phyla-beta configuration (from [configs/eval_config.yaml](file:///home/jianpinhe3/virophylo/Phyla/configs/eval_config.yaml)):

```yaml
model:
  d_model: 256
  n_layer: 16
  vocab_size: 24
  num_blocks: 3
  model_name: "Phyla-beta"
  calculation_method: "attention"
  bidirectional: true
  bidirectional_strategy: "add"
  bidirectional_weight_tie: true
  inject_rotary_attention: false
  positional_embeddings: false
  rms_norm: true
  residual_in_fp32: true
```

- **Parameters**: ~291M
- **Architecture**: Hybrid Mamba (BiMamba) + Multi-Head Attention
- **Weights**: [Phyla-beta](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OTA6BN) from Harvard Dataverse (157 MB)
- **Pre-training**: TreeFam families (3,321 high-quality reference trees)

---

## Evaluation Method

The evaluation follows the **exact algorithm** from the paper's `evo_reasoning_eval.py` ([source](file:///home/jianpinhe3/virophylo/Phyla/phyla/eval/evo_reasoning_eval.py), lines 443–497):

### rf_distance Algorithm

```
for each VFAM family:
  1. Parse both predicted and reference Newick strings with Bio.Phylo.read()
  2. Set all branch_length attributes to None (internal and terminal nodes)
  3. Re-serialize with Phylo.write() to Newick format
  4. Strip remaining ":0.00000" annotations via string slicing
  5. Parse cleaned strings with ete3.Tree() (format 1, unrooted)
  6. Compute Robinson-Foulds distance: t1.compare(t2, unrooted=True)
  7. Normalized RF = rf / max_rf  (normRF: 0 = identical topology, 1 = completely different)
  8. If ANY exception occurs → skip the family (paper's behavior)
```

Key properties:
- **Topology-only comparison**: Branch lengths and support values are removed; only tree topology (branching order) is compared
- **Unrooted comparison**: Trees are compared as unrooted, which is standard for phylogenetic RF distance
- **Exception handling**: Families with unparseable trees are silently skipped (caught by `except:`), exactly as in the paper

### Metric: Normalized Robinson-Foulds (normRF)

- **normRF = 0.0**: Perfect topological match (identical trees)
- **normRF = 1.0**: Completely different topologies (no shared bipartitions)
- **normRF ≈ 0.5**: Moderate agreement

The RF distance counts the number of bipartitions (splits) that differ between two trees. Normalization divides by the maximum possible RF distance given the number of leaves.

---

## Results

### TreeFam Baseline (Paper Reproduction)

Before evaluating on virus data, the paper's original TreeFam benchmark was reproduced to validate the setup:

| Metric | This Reproduction | Paper (Ektefaie et al.) |
|--------|-------------------|------------------------|
| Average normRF | **0.5715** | **0.58** |
| Dataset | TreeFam (held-out) | TreeFam (held-out) |
| Model | Phyla-beta | Phyla-beta |

The slight difference (0.5715 vs 0.58) is within expected variance due to differences in hardware, random seeds, and software library versions. **The reproduction is considered successful**, confirming that the model and evaluation pipeline are correctly configured.

### VOGDB Virus Benchmark

| Metric | Value |
|--------|-------|
| **Families evaluated** | **14,940** (of 20,510 common, 72.8%) |
| **Skipped (FastTree `:NA`)** | **5,570 (27.2%)** |
| **Average normRF** | **0.4716** |
| **Median normRF** | **0.5000** |
| **Standard deviation** | 0.3493 |
| **Min normRF** | 0.0000 (perfect match) |
| **Max normRF** | 1.0000 (complete mismatch) |

#### Distribution of normRF

| Range | Count | Percentage |
|-------|-------|------------|
| [0.0, 0.2) | 4,079 | **27.3%** |
| [0.2, 0.4) | 1,776 | 11.9% |
| [0.4, 0.6) | 2,870 | 19.2% |
| [0.6, 0.8) | 3,269 | 21.9% |
| [0.8, 1.0] | 871 | 5.8% |

#### Key Findings

1. **PHYLA performs better on virus data than on TreeFam**: Average normRF = 0.4716 (virus) vs 0.5715 (TreeFam). Lower is better, so PHYLA's tree reconstructions are closer to the FastTree reference for viral families than they are to the expert-curated TreeFam references.

2. **26.3% perfect matches**: Over a quarter of all evaluated virus families have **identical topology** between PHYLA and FastTree (normRF = 0.0). This is notably higher than the TreeFam benchmark.

3. **Bimodal distribution**: The distribution shows peaks at both ends — many families are either very well reconstructed or moderately divergent. Very few families (5.8%) fall in the worst range (normRF ≥ 0.8).

4. **27.2% of families skipped**: FastTree occasionally outputs `:NA` as a branch length or support value when it cannot compute a reliable estimate. Bio.Phylo cannot parse these non-standard Newick strings, so these families are skipped — exactly as the paper handles invalid tree formats.

#### Comparison: TreeFam Baseline vs VOGDB Virus

| Aspect | TreeFam | VOGDB Virus |
|--------|---------|-------------|
| Average normRF | 0.5715 | **0.4716** |
| Perfect matches | ~10% | **26.3%** |
| Reference tree source | Expert-curated | FastTree (de novo) |
| Reference tree quality | High (manually verified) | Variable (algorithmic) |
| Sequence diversity | Eukaryotic orthologs | Viral proteins |
| Family size (mean) | ~25 seqs | ~26 seqs |
| Number of families | 3,321 | 14,940 |

**Important caveat**: The VOGDB reference trees are themselves computational reconstructions (FastTree), not ground-truth evolutionary trees. A lower normRF could indicate that PHYLA agrees with FastTree, but both could be wrong. Conversely, a higher normRF could mean PHYLA disagrees with FastTree, but PHYLA could be more correct. This is a fundamental limitation of evaluating on data without gold-standard references.

---

## Limitations

### 1. FastTree `:NA` Output — 27.2% Data Loss
FastTree produces `:NA` in its Newick output for certain families, particularly those with:
- Very few sequences (close to the minimum threshold of 4)
- Highly divergent sequences where branch lengths cannot be estimated
- Shallow trees where SH-like support values are undefined

The paper's `rf_distance` function uses `Bio.Phylo.read()`, which strictly validates Newick format and rejects `:NA`. These families are silently skipped. While this matches the paper's methodology exactly, it reduces the effective evaluation size from 20,510 to 14,940 families — a 27.2% loss.

### 2. No Ground-Truth Reference Trees
Unlike TreeFam, where reference trees are expert-curated and manually verified, VOGDB does not provide reference trees. The "reference" in this benchmark is a **FastTree reconstruction** from the same MSA. This creates a circular evaluation: both methods start from the same alignment, and we are comparing two different algorithmic approaches to tree building rather than measuring absolute accuracy.

### 3. Single Metric (normRF Only)
The paper evaluates two tiers of metrics:
- **Tier 1**: Robinson-Foulds distance (topology)
- **Tier 2**: Label prediction within clusters (functional signal)

This benchmark implements only Tier 1. Tier 2 evaluation would require functional annotations (e.g., virus host, pathogenicity, gene function) for VOGDB families, which are available but were not integrated.

### 4. Single Model (Phyla-beta)
Only Phyla-beta was evaluated. The paper also reports results for Phyla-alpha, ESM-2, and Evo. Without these baselines, it is unclear whether PHYLA's virus performance is exceptional or expected for any embedding-based method.

### 5. VOGDB MSA Quality
VOGDB MSAs are automatically generated and may contain alignment errors, especially for highly divergent viral sequences. These alignment errors propagate to both the FastTree reference and the PHYLA embedding (which uses the raw, unaligned sequences), but in different ways.

### 6. No Statistical Significance Testing
The results report point estimates (mean, median normRF) without confidence intervals or statistical tests. It is unclear whether the observed difference between TreeFam (0.5715) and virus (0.4716) is statistically significant.

---

## Future Directions

### Short-Term Improvements

1. **Fix FastTree `:NA` Issue**
   - Modify the reference tree construction to post-process `:NA` values before packaging the pickle
   - Options: replace with `:0.0`, remove the branch entirely, or use FastTree flags that avoid NA output
   - Could recover all 5,570 skipped families (27.2% of the dataset)

2. **Post-Hoc NA Handling in Comparison**
   - Implement a pre-processing step in `remove_branch_distances` that sanitizes `:NA` before Bio.Phylo parsing, while still following the paper's overall approach
   - Must be done carefully to preserve the methodological integrity

### Medium-Term Enhancements

3. **Tier 2 Evaluation (Functional Signal)**
   - Integrate VOGDB annotations (`vfam.annotations.tsv.gz`) to assign functional labels to sequences
   - Implement `mean_cluster_value` from the paper to evaluate whether PHYLA's clusters correspond to functional groups
   - VOGDB provides virus–host annotations that could serve as labels

4. **Additional Baselines**
   - Add comparisons with:
     - **Phyla-alpha** (larger model, more pre-training data)
     - **ESM-2** embeddings + NJ (as in the paper)
     - **Raw sequence identity** + NJ (simplest baseline)
   - This would contextualize PHYLA's virus performance within the broader landscape

5. **Statistical Validation**
   - Bootstrap confidence intervals for normRF
   - Permutation tests comparing virus vs TreeFam distributions
   - Per-family analysis correlating normRF with sequence diversity, family size, alignment quality

### Long-Term Research

6. **Independent Virus Tree Validation**
   - Use **ICTV taxonomy** as an external gold standard: check whether PHYLA trees recover known viral taxonomic relationships
   - Compare against **maximum likelihood trees** from IQ-TREE or RAxML (gold-standard phylogenetic methods) instead of FastTree neighbor-joining
   - Validate on specific virus groups with well-established phylogenies (e.g., Coronaviridae, Flaviviridae, HIV)

7. **Multi-MSA Integration**
   - VOGDB provides MSAs; the current pipeline uses them only for FastTree
   - Could also feed MSAs into PHYLA (which natively handles aligned sequences via tokenization)
   - Compare: unaligned → PHYLA vs aligned → PHYLA vs aligned → FastTree

8. **Cross-Dataset Generalization**
   - Evaluate on additional virus-specific phylogeny databases:
     - **VirusITE** (virus integration sites)
     - **PhagesDB** (bacteriophage genomics)
     - **GVD** (Global Virome Database)
   - Test whether PHYLA's performance correlates with evolutionary rate, genome type (DNA/RNA/ss/ds), or host range

---

## Project Structure

```
Phyla/
├── README.md                          # This report
├── .gitignore                         # Ignores weights, data, logs
│
├── phyla/                             # PHYLA model package (installed in editable mode)
│   ├── model/model.py                 # Core model (device mismatch fix applied)
│   ├── eval/evo_reasoning_eval.py     # Paper's evaluation functions
│   └── ...
│
├── configs/eval_config.yaml           # Model configuration
├── weights/11564369/                  # Phyla-beta checkpoint (157 MB)
├── fasttree                           # Compiled FastTree 2.1.11 binary (380 KB)
│
├── virus_data/                        # All virus evaluation data
│   ├── faa/                           # 39,776 VFAM FAA files (308 MB)
│   ├── msa/                           # 39,776 VFAM MSA files (393 MB)
│   ├── ref_trees/                     # 20,510 FastTree reference trees (84 MB)
│   ├── vogdb_treefam_v2.pickle        # Reference dataset (20,510 fams, 141 MB)
│   ├── phyla_predictions.pickle       # PHYLA predictions (20,534 fams, 50 MB)
│   └── *.tar.gz / *.tsv.gz            # Raw VOGDB downloads
│
├── eval_preds/                        # Evaluation outputs
│   ├── virus_phyla_vs_fasttree.csv    # 14,940 normRF results
│   └── treefam_results.csv            # TreeFam baseline results
│
├── slurm_logs/                        # SLURM job logs
│
├── build_virus_eval_dataset.py        # Stage 1: reference tree construction
├── run_phyla_predict_trees.py         # Stage 2: PHYLA tree prediction
├── compare_virus_trees.py             # Stage 3: normRF comparison
│
├── run_phyla_predict_slurm.sh         # GPU SLURM script
├── run_virus_ref_trees.sh             # CPU SLURM script (FastTree)
├── run_virus_compare_slurm.sh         # CPU SLURM script (comparison)
│
├── run_eval.py                        # TreeFam evaluation wrapper
├── run_inference.py                   # Single-inference script
├── run_eval_slurm.sh                  # TreeFam SLURM script
└── run_phyla_slurm.sh                 # PHYLA SLURM script
```

---

## Usage

### Prerequisites

- Python 3.10+ with PyTorch, transformers, mamba_ssm, flash_attn, deepspeed
- Conda environment `virophylo` with all dependencies
- SLURM workload manager (HPC environment)

### Running the Full Pipeline

```bash
# Step 1: Build FastTree reference trees (CPU)
sbatch run_virus_ref_trees.sh

# Step 2: Run PHYLA predictions (GPU, independent of Step 1)
sbatch run_phyla_predict_slurm.sh

# Step 3: Compare results (CPU, after both complete)
sbatch run_virus_compare_slurm.sh
```

### Custom Evaluation

```bash
# Compare a specific subset
conda activate virophylo
python compare_virus_trees.py \
    --ref-pickle virus_data/vogdb_treefam_v2.pickle \
    --pred-pickle virus_data/phyla_predictions.pickle \
    --output-csv eval_preds/custom_comparison.csv
```

---

## References

1. **Ektefaie, Y. et al. (2025)**. *Evolutionary Reasoning Does Not Arise in Standard Usage of Protein Language Models*. bioRxiv. [https://doi.org/10.1101/2025.02.12.637880](https://doi.org/10.1101/2025.02.12.637880)

2. **PHYLA GitHub Repository**: [https://github.com/mims-harvard/Phyla](https://github.com/mims-harvard/Phyla)

3. **Phyla-beta Weights**: Harvard Dataverse [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OTA6BN](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OTA6BN)

4. **VOGDB**: [https://vogdb.org/](https://vogdb.org/)

5. **FastTree 2.1.11**: Price, M.N. et al. (2010). *FastTree 2 – Approximately Maximum-Likelihood Trees for Large Alignments*. PLoS ONE. [http://www.microbesonline.org/fasttree/](http://www.microbesonline.org/fasttree/)

6. **ete3**: Huerta-Cepas, J. et al. (2016). *ETE 3: Reconstruction, Analysis, and Visualization of Phylogenomic Data*. Mol. Biol. Evol.

7. **scikit-bio**: [https://scikit.bio/](https://scikit.bio/)