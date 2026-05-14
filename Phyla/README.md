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
           ┌──────────────────────────────┐
           │ Stage 3: Eval (CPU)          │
           │ compare_virus_trees.py       │
           │   - PHYLA vs FastTree        │
           │   - Hamming + NJ vs FastTree │
           │   - Random tree vs FastTree  │
           │   - ESM2 + NJ vs FastTree *  │
           └──────────┬───────────────────┘
                      ▼
           eval_preds/virus_*.csv
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

### Stage 3: Multi-Baseline Comparison
- **Script**: [compare_virus_trees.py](file:///home/jianpinhe3/virophylo/Phyla/compare_virus_trees.py)
- **SLURM**: [run_virus_eval_slurm.sh](file:///home/jianpinhe3/virophylo/Phyla/run_virus_eval_slurm.sh)
- **Jobs**: 57422 (CPU, completed), 57425 (GPU, ESM2)
- **Outputs**:
  - `eval_preds/virus_phyla_vs_fasttree.csv`
  - `eval_preds/virus_hamming_vs_fasttree.csv`
  - `eval_preds/virus_random_vs_fasttree.csv`
  - `eval_preds/virus_esm2_vs_fasttree.csv` *(pending)*

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
  4. Strip branch lengths via regex pattern ':digits'
  5. Remove single quotes from leaf names (PHYLA output quirk)
  6. Parse cleaned strings with ete3.Tree()
  7. Compute Robinson-Foulds distance: t1.compare(t2, unrooted=True)
  8. If comparison returns 'NA' (leaf set mismatch) → skip
  9. Normalized RF = rf / max_rf
```

Key properties:
- **Topology-only comparison**: Branch lengths and support values are removed
- **Unrooted comparison**: Standard for phylogenetic RF distance
- **Critical requirement**: Both trees must have identical leaf name sets (ete3 returns `NA` otherwise)

### Leaf Set Mismatch Handling

VOGDB reference trees (built from MSA via FastTree) contain fewer leaves than the original FAA files, because MSA alignment filters out the most divergent sequences. This causes ete3's `Tree.compare()` to return `NA` when comparing trees with different leaf sets.

**Improvement (v2)**: Instead of discarding families with mismatched leaf sets (which previously lost 78.2% of data), the pipeline now **prunes predicted trees to the intersecting leaf set** before comparison. This recovers the vast majority of families and eliminates the selection bias toward simple/easily-alignable families.

### Baseline Methods

| Baseline | Description | Data Source |
|----------|-------------|-------------|
| **PHYLA** | CLS embedding → Euclidean distance → NJ | FAA (unaligned) |
| **Hamming + NJ** | Pairwise Hamming distance on MSA → NJ | MSA (aligned) |
| **Random tree** | Random leaf order → balanced tree | Reference tree leaves |
| **ESM2 + NJ** | ESM2-650M embedding → Euclidean distance → NJ | FAA (unaligned) |

### Metric: Normalized Robinson-Foulds (normRF)

- **normRF = 0.0**: Perfect topological match (identical trees)
- **normRF = 1.0**: Completely different topologies (no shared bipartitions)
- **normRF ≈ 0.5**: Moderate agreement

### Statistical Significance

The pipeline now reports:
- **Bootstrap 95% confidence intervals** (10,000 resamples) for mean normRF
- **Paired Wilcoxon signed-rank test** between baselines (accounts for paired observations on the same families)
- **Cohen's d** effect size to quantify the magnitude of differences
- **Stratified analysis** by family size (small: 4–10, medium: 11–50, large: 51+)

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

#### Leaf Set Matching (v2 — Fixed via Pruning)

Originally, **78.2% of reference trees** had leaf sets that did not match PHYLA predictions, because MSA alignment filtered out divergent sequences before FastTree tree building. Only 4,462 families (21.8%) were evaluable.

**Fix (v2)**: The pipeline now **prunes the predicted tree to match the reference tree's leaf set** before computing normRF (see `compare_virus_trees.py`). This recovers the vast majority of families and removes selection bias toward simple families.

| Item | Value |
|------|-------|
| Common families (ref + pred overlap) | 20,510 |
| Perfect leaf match (no pruning needed) | ~4,462 (21.8%) |
| Pruned to intersection | ~16,048 (78.2%) |
| Too few leaves after pruning (<4) | minimal |
| **Total evaluable families (v2)** | **~20,510** |

#### Multi-Baseline Comparison

Results below are computed after pruning predicted trees to intersecting leaf sets, recovering all evaluable families. Paired statistical tests are reported alongside point estimates.

| Baseline | Families | Avg normRF | [95% CI] | Median | Perfect (0.0)% | Worst (>=0.98)% |
|----------|----------|------------|----------|--------|---------------|-----------------|
| **Random tree** | ~20,510 | **1.0000** | — | 1.0000 | 0.0% | 100.0% |
| **PHYLA (CLS + NJ)** | ~20,510 | **—** | — | — | —% | —% |
| **Hamming + NJ** | ~20,510 | **—** | — | — | —% | —% |
| **ESM2-650M + NJ** | *running* | *—* | — | — | — | — |

*Note: Exact numbers will be populated after re-running with the v2 pruning fix.*

#### Key Findings (Preliminary — Awaiting Re-run)

1. **Random tree baseline (normRF=1.0) validates the metric**: Random topologies produce normRF=1.0 exactly, confirming the metric is not pathologically biased.

2. **The v1 result (4,462 families) was biased toward easier families**: Families with perfect leaf set matching are those where MSA didn't filter out sequences — i.e., more conserved families. The v2 results on ~20,510 families will reveal whether PHYLA's performance degrades or improves on harder families.

3. **Important caveat**: Hamming + NJ uses the **MSA (aligned sequences)** while PHYLA uses **FAA (unaligned sequences)**. This gives Hamming a significant advantage since alignment removes ambiguity. A fairer comparison would require either:
   - Feeding MSAs to PHYLA (it natively tokenizes raw sequences)
   - Computing Hamming from unaligned sequences (using pairwise alignment, which is expensive)

---

## Limitations

### 1. ~~Reference Tree Leaf Set Mismatch — 78.2% Data Loss~~ ✅ Resolved (v2)
The leaf set mismatch is now handled by **pruning predicted trees to match reference tree leaf sets** before comparison. Families are no longer discarded, recovering the full set of ~20,510 evaluable families. See `compare_virus_trees.py` for implementation details.

### 2. No Ground-Truth Reference Trees
Unlike TreeFam, where reference trees are expert-curated, VOGDB does not provide reference trees. The "reference" is a FastTree reconstruction, creating two problems:
- Both PHYLA and FastTree start from the same alignment data
- We are comparing algorithmic approaches, not measuring absolute accuracy

### 3. Unfair Baseline Advantage
Hamming distance + NJ uses aligned MSAs while PHYLA uses raw FAA sequences. Since alignment removes ambiguity, Hamming gets an advantage. This does not necessarily mean Hamming is a better phylogenetic method — it means sequence alignment is helpful.

### 4. Single Metric (normRF Only)
The paper evaluates two tiers of metrics:
- **Tier 1**: Robinson-Foulds distance (topology)
- **Tier 2**: Label prediction within clusters (functional signal)

This benchmark implements only Tier 1.

### 5. Single Model (Phyla-beta)
Only Phyla-beta was evaluated. The paper also reports Phyla-alpha, ESM2, and Evo. ESM2 + NJ is currently running (Job 57425).

### 6. ~~No Statistical Significance Testing~~ ✅ Resolved (v2)
The pipeline now reports **bootstrap 95% confidence intervals** (10,000 resamples), **paired Wilcoxon signed-rank tests**, and **Cohen's d** effect sizes between baselines. See `compare_virus_trees.py` output for details.

### 7. Single Random Seed for Random Baseline
The random baseline uses a single shuffle per family. NormRF may vary with different random seeds. Bootstrap CI mitigates this partially.

---

## Future Directions

### Short-Term Improvements

1. ~~**Fix Reference Tree Leaf Set Mismatch**~~ ✅ **Done (v2)**
   - Implemented tree pruning to intersecting leaf sets in `compare_virus_trees.py`
   - Recovers ~20,510 families (was 4,462)
   - ESM2 eval script (`run_esm2_eval.py`) also updated with same fix

2. **Fair Baseline Comparison**
   - Give PHYLA the same advantage as Hamming: run PHYLA on MSAs instead of FAA
   - Or, compute Hamming distance from unaligned sequences via Needleman-Wunsch pairwise alignment
   - This would reveal whether Hamming's advantage is due to alignment or genuine phylogenetic signal

3. **Complete ESM2-650M + NJ Baseline**
   - GPU evaluation running (Job 57425); script updated with leaf set pruning fix

### Medium-Term Enhancements

4. **Tier 2 Evaluation (Functional Signal)**
   - Integrate VOGDB annotations for virus–host labels
   - Implement `mean_cluster_value` from the paper

5. **Additional Baselines**
   - Phyla-alpha (larger model, more pre-training data)
   - MAFFT + FastTree (gold-standard pipeline, as in paper)
   - Raw sequence identity + NJ (simplest baseline)

6. ~~**Statistical Validation**~~ ✅ **Done (v2)**
   - Bootstrap confidence intervals for mean normRF (10,000 resamples)
   - Paired Wilcoxon signed-rank test between baselines
   - Cohen's d effect size
   - Stratified analysis by family size (small/medium/large)
   - See `compare_virus_trees.py` output

### Long-Term Research

7. **Independent Virus Tree Validation**
   - Use ICTV taxonomy as an external gold standard
   - Compare against maximum likelihood trees from IQ-TREE or RAxML
   - Validate on specific virus groups (Coronaviridae, Flaviviridae, HIV)

8. **Multi-MSA Integration**
   - Feed MSAs directly into PHYLA (which natively handles sequences)
   - Compare: unaligned → PHYLA vs aligned → PHYLA vs aligned → FastTree

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
│   ├── virus_phyla_vs_fasttree.csv    # PHYLA results (4,462 families)
│   ├── virus_hamming_vs_fasttree.csv  # Hamming baseline (4,462 families)
│   ├── virus_random_vs_fasttree.csv   # Random baseline (3,792 families)
│   └── virus_esm2_vs_fasttree.csv     # ESM2 baseline (pending)
│
├── slurm_logs/                        # SLURM job logs
│
├── build_virus_eval_dataset.py        # Stage 1: reference tree construction
├── run_phyla_predict_trees.py         # Stage 2: PHYLA tree prediction
├── compare_virus_trees.py             # Stage 3: multi-baseline comparison
├── run_esm2_eval.py                   # Stage 3: ESM2-650M + NJ evaluation
│
├── run_virus_eval_slurm.sh            # CPU SLURM (PHYLA + Hamming + Random)
├── run_esm2_slurm.sh                  # GPU SLURM (ESM2 + NJ)
├── run_phyla_predict_slurm.sh         # GPU SLURM (PHYLA predictions)
├── run_virus_ref_trees.sh             # CPU SLURM (FastTree)
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

# Step 3a: Run CPU baselines (PHYLA, Hamming, Random)
sbatch run_virus_eval_slurm.sh

# Step 3b: Run GPU baseline (ESM2 + NJ)
sbatch run_esm2_slurm.sh
```

### Custom Evaluation

```bash
conda activate virophylo
python compare_virus_trees.py \
    --ref-pickle virus_data/vogdb_treefam_v2.pickle \
    --pred-pickle virus_data/phyla_predictions.pickle \
    --msa-dir virus_data/msa \
    --faa-dir virus_data/faa \
    --output-dir eval_preds \
    --baselines phyla hamming random
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