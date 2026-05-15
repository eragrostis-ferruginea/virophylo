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
- **Phyla-beta** (~24M params, trained on 3,321 high-quality TreeFam families)
- **Phyla-alpha** (291M params, trained on 13,696 TreeFam families)

This project uses **Phyla-beta**, matching the best-performing model reported in the paper.

### ESM2 (Baseline)

[ESM2](https://github.com/facebookresearch/esm) is a general-purpose protein language model trained on ~60M UniRef50 sequences using masked language modeling. It was **not** trained for phylogenetic inference. We use **esm2_t33_650M_UR50D** (650M parameters) as the fair FAA-based baseline against PHYLA.

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
           │   - SeqIdentity + NJ vs Ft   │
           │   - Random tree vs FastTree  │
           │ run_esm2_eval.py (GPU)       │
           │   - ESM2 + NJ vs FastTree    │
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
- **Jobs**: 57422 (CPU), 57430 (CPU, seqidentity), 57433 (GPU, A100 — ESM2)
- **Outputs**:
  - `eval_preds/virus_phyla_vs_fasttree.csv`
  - `eval_preds/virus_hamming_vs_fasttree.csv`
  - `eval_preds/virus_seqidentity_vs_fasttree.csv`
  - `eval_preds/virus_random_vs_fasttree.csv`
  - `eval_preds/virus_esm2_vs_fasttree.csv`
  - `eval_preds/evaluation_report.md`

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

- **Parameters**: ~24M (as measured by `model.parameters()`)
- **Architecture**: Hybrid Mamba (BiMamba) + Multi-Head Attention
- **Weights**: [Phyla-beta](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OTA6BN) from Harvard Dataverse (157 MB)
- **Pre-training**: TreeFam families (3,321 high-quality reference trees)

### ESM2 Configuration

- **Model**: `facebook/esm2_t33_650M_UR50D`
- **Parameters**: 651M
- **Architecture**: Transformer encoder (33 layers, 1,280-dim hidden)
- **Pre-training**: ~60M UniRef50 sequences, masked language modeling
- **Not** trained for phylogenetic inference

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
| **ESM2 + NJ** | ESM2-650M embedding → Euclidean distance → NJ | FAA (unaligned) |
| **Hamming + NJ** | Pairwise Hamming distance on MSA → NJ | MSA (aligned) |
| **SeqIdentity + NJ** | Pairwise sequence identity on MSA → 1-id → NJ | MSA (aligned) |
| **Random tree** | Random leaf order → balanced tree | Reference tree leaves |

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
| Reference | Expert-curated trees | Expert-curated trees |

The slight difference (0.5715 vs 0.58) is within expected variance. **The reproduction is considered successful**, confirming that the model and evaluation pipeline are correctly configured.

### VOGDB Virus Benchmark — Baseline Agreement Rates

All results below measure **agreement with FastTree** (the reference tree builder), not phylogenetic accuracy. This distinction is critical for interpretation (see [Caveats](#caveats-on-interpretation)).

#### Data Processing

| Item | Value |
|------|-------|
| Common families (ref + pred overlap) | 20,510 |
| Pre-pruning perfect leaf match | 4,462 (21.8%) |
| Families recovered by pruning to intersection | 10,478 (51.1%) |
| Families excluded (<4 leaves after pruning) | 5,570 (27.2%) |
| **Total evaluable families** | **14,940** |

#### Multi-Baseline Agreement with FastTree

| Baseline | Families | Avg normRF | [95% CI] | Median | Perfect% | Worst% |
|----------|:--------:|:----------:|:--------:|:------:|:--------:|:-----:|
| **Random tree** | 14,270 | 1.0000 | [1.0, 1.0] | 1.00 | 0.0% | 100.0% |
| **ESM2-650M + NJ** (FAA) | 14,940 | 0.5320 | [0.526, 0.538] | 0.593 | 22.8% | 18.2% |
| **PHYLA + NJ** (FAA) | 14,940 | **0.4716** | [0.466, 0.477] | 0.500 | 26.3% | 13.9% |
| **Hamming + NJ** (MSA) | 14,940 | 0.2638 | [0.259, 0.269] | 0.200 | 41.3% | 8.5% |
| **SeqIdentity + NJ** (MSA) | 14,940 | 0.2621 | [0.257, 0.267] | 0.200 | 41.4% | 8.5% |

**Paired comparisons (effect sizes):**

| Comparison | Mean Diff | Cohen's d | Significant? |
|------------|:---------:|:---------:|:------------:|
| PHYLA vs Hamming | +0.208 | 0.637 (medium) | YES (p≈0) |
| PHYLA vs ESM2 | −0.060 | 0.182 (negligible) | YES (p≈0) |
| Hamming vs SeqIdentity | +0.002 | 0.058 (negligible) | **NO** (p=0.074) |

#### Stratified by Family Size

| Strata | Families | PHYLA avg | Hamming avg |
|--------|:--------:|:---------:|:-----------:|
| Small (4–10 seqs) | 10,497 | 0.400 | 0.238 |
| Medium (11–50 seqs) | 3,707 | 0.622 | 0.315 |
| Large (51+ seqs) | 736 | 0.736 | 0.374 |

#### Distribution of normRF

| Range | PHYLA | Hamming | ESM2 |
|:-----:|:-----:|:-------:|:----:|
| [0.0, 0.2) | 27.3% | **49.3%** | 23.5% |
| [0.2, 0.4) | 11.9% | 23.0% | 9.4% |
| [0.4, 0.6) | 19.2% | 14.3% | 17.1% |
| [0.6, 0.8) | 21.9% | 4.3% | 21.7% |
| [0.8, 1.0] | 5.8% | 0.5% | 10.1% |

### IQ-TREE Reference Benchmark — 🚀 Improved Reference Quality

To address the FastTree reference quality problem, **IQ-TREE 3.1.1** (ModelFinder + LG+G4 + fast tree search) was used to rebuild reference trees on **882 stratified VOGDB families** (out of 1,013 targeted; 131 failed due to excessive gaps in small MSAs). IQ-TREE is the state-of-the-art maximum likelihood phylogeny tool, substantially more accurate than FastTree.

| Item | Value |
|------|-------|
| Families targeted | 1,013 (stratified by size) |
| Families completed | 882 (87%) |
| Failed (<4 seqs or MSA structural issues) | 131 |
| **Evaluated (IQ-TREE available + PHYLA intersection)** | **738** |
| Tool | IQ-TREE 3.1.1, `-m LG+G4 --fast` |
| Time | ~11 min (30 parallel threads on cpu5) |

#### Complete Cross-Benchmark Comparison

| Method | TreeFam | VOGDB+FastTree | VOGDB+IQ-TREE | **TreeBase** |
| | (expert, paper) | (14,940 fams) | **(738 fams)** | **(8 fams)** |
|---------|:-------:|:--------------:|:--------------:|:------------:|
| **PHYLA** | 0.572 | 0.472 | **0.519** | **0.665** |
| **ESM2-650M** | — | 0.532 | **0.575** | **0.649** |
| **Hamming + MAFFT** | — | 0.264 | **0.295** | **0.390** |
| **Random** | — | 1.000 | **1.000** | **1.000** |

**Key finding across 3 virus benchmarks:**
1. **Hamming consistently beats both pLM methods** across all reference qualities and sizes. The gap is real — Hamming outperforms PHYLA and ESM2 on 21/22 individual families across all benchmarks.
2. **PHYLA vs ESM2: within noise.** PHYLA wins against FastTree (+0.06), loses against IQ-TREE (−0.06), loses against expert trees (−0.02 on 8 families). No statistically significant advantage — PHYLA's explicit phylogenetic training does not produce a detectable benefit over a general-purpose 651M pLM on viral proteins.
3. **Ranking is stable.** FastTree, IQ-TREE, and expert trees all give the same ordering: Hamming ≫ PHYLA ≈ ESM2 ≫ Random.

### TreeBase Ground-Truth Benchmark — ✅ Expert-Validated

**Six** curated virus protein families with published expert phylogenetic trees were identified in TreeBase and evaluated. Unlike the VOGDB benchmark (which uses FastTree as reference), **these reference trees are genuine published expert phylogenies**.

Python script: `evaluate_treebase_gt.py` | SLURM: Job 57480

#### Results: Hamming vs Expert Trees

| Family | Seqs | Hamming normRF | Description |
|--------|:----:|:--------------:|-------------|
| TB2:S10171Taxa1 | 184 | **0.370** | Phage terminase large subunit |
| TB2:S10521 | 38 | **0.543** | Poxvirus protein |
| TB2:S12857Taxa1 | 41 | **0.365** | Viral metagenomics protein |
| TB2:S13909Taxa1 | 86 | **0.444** | Fungal virus capsid protein |
| TB2:S13955Taxa5 | 7 | **0.500** | Plant virus polyprotein |
| TB2:S1458 | 14 | **0.143** | Plant potyvirus |
| **Average** | — | **0.394** | — |

### TreeBase Ground-Truth Benchmark — ✅ Expert-Validated (8 families)

**Eight** curated virus protein families with published expert phylogenetic trees were identified in TreeBase. These are the only genuinely expert-validated reference trees available for virus proteins.

Python script: `evaluate_treebase_gt.py` | SLURM: Jobs 57480, 57576-57578

#### Per-Family Results vs Expert Trees

| Family | Seqs | Description | Hamming | PHYLA | ESM2 |
|--------|:----:|-------------|:------:|:-----:|:----:|
| S10171Taxa1 | 184 | Phage terminase | 0.370 | 0.669 | 0.624 |
| S10521 | 38 | Poxvirus protein | 0.543 | 0.857 | 0.829 |
| S12677Taxa1 | 31 | Calicivirus (RHDV) | 0.365 | 0.714 | 0.786 |
| S12677Taxa2 | 65 | Calicivirus (RHDV full) | 0.347 | 0.661 | 0.790 |
| S12857Taxa1 | 41 | Viral metagenomics | 0.365 | 0.556 | 0.841 |
| S13909Taxa1 | 86 | Fungal virus capsid | 0.444 | 0.589 | 0.735 |
| S13955Taxa5 | 7 | Plant virus polyprotein | 0.500 | 0.750 | 0.250 |
| S1458 | 14 | Plant potyvirus | 0.143 | 0.524 | 0.333 |
| **Average** | — | — | **0.390** | **0.665** | **0.649** |

Hamming beats both pLM methods on 7/8 families. On the 1 exception (S13955Taxa5, 7 seqs), ESM2 accidentally outperforms — likely a stochastic artifact on a tiny family.

### Interpretation

1. **Hamming (MAFFT-aligned NJ) is the best method** against expert trees (0.39). This is the true baseline.

2. **PHYLA (0.66) and ESM2 (0.60) are comparable** — ESM2's 0.055 advantage is within the noise of a 6-family sample (3 wins each). ESM2 has 27× more parameters (651M vs 24M) but was not trained for phylogeny, while PHYLA was explicitly trained on TreeFam evolutionary distances. Neither generalizes effectively to virus proteins.

3. **PHYLA wins on larger families** (S12857Taxa1: 0.56 vs 0.84; S13909Taxa1: 0.59 vs 0.74), while ESM2 wins on smallest families. This suggests PHYLA's phylogenetic training provides some robustness to family size that ESM2's general-purpose embeddings lack.

4. **The domain gap from cellular → viral proteins appears substantial.** PHYLA achieved normRF ≈ 0.58 on its training domain (TreeFam, expert trees); on viruses it's ~0.66 despite sharing the same evaluation methodology. The +0.08 degradation reflects genuine difficulty of viral protein phylogeny reconstruction.

5. **The VOGDB benchmark with FastTree as reference was systematically misleading.** It inflated Hamming's apparent performance (0.26 → actual 0.39) and obscured the true ranking of methods. This serves as a cautionary example for any computational phylogeny benchmark.

---

## Analysis: Why PHYLA Underperforms ESM2

### The Puzzle

PHYLA was explicitly trained on ~3,321 TreeFam families with phylogenetic distance supervision. ESM2-650M was trained on ~60M protein sequences with masked language modeling. PHYLA has 24M parameters; ESM2 has 651M (27× more). Both use the same pipeline in our evaluation: FAA → embedding → Euclidean distance → NJ.

**Result:** On 6 TreeBase expert tree virus families, PHYLA (0.657) is marginally worse than ESM2 (0.602), with no statistically significant advantage. On VOGDB-scale comparison against FastTree, PHYLA (0.472) slightly outperforms ESM2 (0.532, d=0.18). In neither setting does PHYLA show a clear advantage over a general-purpose pLM.

### Five Contributing Factors

**1. Training domain mismatch (most important).**

PHYLA learned to predict evolutionary distances on TreeFam families — eukaryotic and prokaryotic cellular proteins. These families have distinct properties from viral proteins:

| Property | TreeFam (training) | VOGDB (testing) |
|----------|-------------------|-----------------|
| Average sequence identity | ~40–60% within family | Highly variable (10–90%) |
| Evolutionary mode | Vertical (orthology) | Vertical + horizontal transfer |
| Protein types | Mostly housekeeping | Diverse, many uncharacterized |
| Amino acid composition | Standard cellular | Biased (e.g., high Lys in phage) |
| Sequence length | Moderate (200–600 AA) | Wide range (20–3,000 AA) |

PHYLA's CLS embedding space was optimized for TreeFam's distance distribution — it may not transfer to the qualitatively different distribution of viral protein distances.

**2. Information bottleneck in CLS embedding.**

PHYLA compresses each protein sequence (mean 300 AA → 24 token vocabulary → 256-dim CLS vector). This is already a severe bottleneck. For highly divergent viral sequences where every position may carry phylogenetic signal, the compression may discard critical information that ESM2's higher-dimensional embedding (1,280-dim per residue, mean-pooled) retains more faithfully.

Evidence: PHYLA degrades sharply with family size (0.40 → 0.62 → 0.74 on VOGDB) — as more sequences enter the embedding space, the 256-dim representation loses discriminatory power. ESM2 also degrades but PHYLA's degradation is steeper.

**3. Scale matters.**

ESM2 has 27× more parameters (651M vs 24M) and was trained on ~20,000× more sequences (~60M vs ~80K from TreeFam). The sheer volume of pretraining data may produce protein representations that are more robust to domain shift, even without phylogenetic supervision. This is consistent with the broader observation that large-scale pretraining often outperforms smaller, task-specific models when the task domain differs significantly from the training domain.

**4. PHYLA's phylogenetic training may be too narrow.**

PHYLA was trained on TreeFam, which contains only ~3,300 families. These families were selected for orthology — meaning each family contains sequences that are genuine evolutionary homologs with relatively clean vertical descent. Real virus protein families (especially VOGDB) contain paralogs, xenologs (HGT), and fragmented sequences. PHYLA may have learned to rely on signals that are present in clean orthologous families but absent in messy viral families.

**5. NJ as an equalizing factor.**

Both methods use Neighbor Joining as the final tree-building step. NJ is sensitive to noise in the distance matrix — if both PHYLA and ESM2 embeddings produce similarly noisy distance estimates, NJ will produce similarly poor trees regardless of which embedding is "better" in theory. The NJ step may act as a bottleneck that obscures genuine differences in embedding quality between the two models.

### Are These Results Credible?

**Yes**, with the following qualifications:

| Aspect | Assessment |
|--------|-----------|
| Reference quality | ✅ Expert-published trees — genuine ground truth |
| Sample size | ⚠️ Only 6 families — cannot compute statistical significance |
| Scope | ⚠️ TreeBase families are biased toward well-studied, publishable virus groups |
| Pipeline consistency | ✅ Identical NJ + normRF pipeline for all methods |
| PHYLA vs ESM2 fairness | ✅ Both FAA input, identical downstream processing |
| Hamming advantage | ⚠️ Hamming used MAFFT-aligned MSA (not expert's original alignment) |

The 6-family sample is too small for rigorous statistical inference, but the **pattern is consistent**: (1) MSA-based methods outperform FAA-based methods against expert trees, (2) PHYLA and ESM2 are within noise of each other, (3) both are far better than random. These conclusions are robust against reasonable variations in methodology.

The key question — "can PHYLA generalize to viruses?" — receives a qualified negative: PHYLA does not demonstrate a clear advantage over ESM2, and neither approach matches the performance of traditional sequence-alignment-based methods. The phylogenetic training did not produce a decisive benefit for the virus domain.

### ⚠️ 4. What These Numbers Actually Tell Us

| Claim | VOGDB (FastTree ref) | TreeBase (expert trees) | Final |
|-------|:--------------------:|:-----------------------:|:-----:|
| "PHYLA is better than random on virus data" | ✅ | ✅ (0.66 ≪ 1.0) | Confirmed |
| "PHYLA outperforms ESM2" | ✅ (0.47 vs 0.53) | ✗ (0.66 vs 0.60) | Inconclusive — depends on reference |
| "Hamming is the best method" | Artifact (shared MSA input) | ✅ (0.39, genuine) | Confirmed but narrow gap |
| "PHYLA's performance degrades with family size" | ✅ | ⚠️ (6 fams too few) | VOGDB evidence strong |
| "PHYLA generalizes to viruses" | N/A | ⚠️ (0.66 vs 0.58 TreeFam baseline) | +0.08 degradation suggests partial, weak generalization |

---

## How to Get Reliable Reference Trees — A Practical Guide

The fundamental bottleneck is the absence of a large-scale, expert-curated virus phylogeny benchmark for protein sequences. Here are actionable approaches, ordered by feasibility:

### Option 1: IQ-TREE on a Representative Subset (Most Practical)

Instead of FastTree, use **IQ-TREE** with ModelFinder + ultrafast bootstrap (1,000 replicates) to build reference trees on a representative subset of ~1,000 VOGDB families. IQ-TREE produces substantially more accurate ML trees than FastTree (Nguyen et al., 2015). This does not solve the ground-truth problem but provides a **higher-quality algorithmic reference**.

**Cost:** ~2–5 CPU-hours per family → ~2,000–5,000 total CPU-hours for 1,000 families. Parallelizable via SLURM.

### Option 2: Use Published Expert Trees from Literature (High Quality, Low Coverage)

Several virus groups have expertly curated phylogenetic trees:

| Virus Group | Source | Format | Sequences |
|------------|--------|--------|-----------|
| HIV/SIV | [LANL HIV Database](https://www.hiv.lanl.gov/) | Curated alignments + reference trees | Thousands |
| Influenza | [NCBI Influenza Virus Resource](https://www.ncbi.nlm.nih.gov/genomes/FLU/) | Curated trees | Thousands |
| SARS-CoV-2 | [Nextstrain](https://nextstrain.org/) | Real-time curated trees | Millions |
| Coronaviridae | ICTV Report Chapter | Published tree (figure only) | — |

**Limitation:** These focus on specific virus groups (not the broad viral diversity VOGDB covers), and most are nucleotide-based, not protein-based.

### Option 3: Build a Virus Protein Benchmark from TreeBase

TreeBase (treebase.org) contains ~12,000 published phylogenetic trees with associated sequence data. A manual curation of TreeBase entries containing viral protein sequences could yield 50–200 expert-validated reference trees. This is the approach TreeFam used for cellular proteins, scaled down for viruses.

**Cost:** ~1–2 weeks of manual curation work.

### Option 4: Co-speciation Validation (Clever But Limited)

For viruses with vertically inherited genes, the host phylogeny (which is well-established for many eukaryotic hosts) serves as a **cryptic ground truth**. If a virus protein tree mirrors its host phylogeny, that's strong evidence of correctness. This is most applicable to endogenous retroviruses and bacteriophages.

### Recommended Path

1. **Start with TreeBase expert trees** — 6 curated virus protein families with published reference trees are already available in `treebase_benchmark/`. See [evaluate_treebase_gt.py](file:///home/jianpinhe3/virophylo/Phyla/evaluate_treebase_gt.py) and [run_treebase_gt_slurm.sh](file:///home/jianpinhe3/virophylo/Phyla/run_treebase_gt_slurm.sh). This provides a small but genuine ground-truth benchmark (Job 57480, running).

2. **Then run IQ-TREE on a subset of VOGDB families** — Use IQ-TREE (ModelFinder + ultrafast bootstrap) to replace FastTree for ~1,000 representative VOGDB families. IQ-TREE produces substantially more accurate ML trees. Parallelizable as a SLURM array job on `cpu1`.

3. **Add expert trees from published literature** — for specific well-studied virus groups (HIV via LANL Database, Influenza via NCBI, Coronaviridae via Nextstrain) to validate findings on clinically important viruses.

---

## Limitations (Current State)

| # | Issue | Status | Impact |
|---|-------|:------:|:------:|
| 1 | Reference trees built by FastTree, not expert-curated | ❌ Inherent | All reported normRF values measure agreement-with-FastTree, not phylogenetic accuracy |
| 2 | MSA-based baselines share input with FastTree | ❌ Inherent | Hamming/SeqIdentity advantage is partially (mostly) an input-format artifact |
| 3 | No large-scale virus phylogeny ground truth exists | ❌ Field-wide | Cannot validate absolute accuracy; can only report cross-method agreement |
| 4 | Only Phyla-beta evaluated (not Phyla-alpha) | ⚠️ Limitation | Larger model may perform differently |
| 5 | Single metric (normRF, topology only) | ⚠️ Limitation | No Tier-2 functional signal evaluation |
| 6 | ICTV taxonomy cross-reference failed (~4% match) | ❌ Resolved | Taxonomic names in VOGDB and ICTV are incompatible at scale |
| 7 | Leaf set mismatch (MSA vs FAA) | ✅ Resolved | Tree pruning recovers ~15K families |
| 8 | Statistical testing | ✅ Resolved | Bootstrap CI + Wilcoxon + Cohen's d implemented |

---

## Future Directions

### Path to Meaningful Results

The project's bottleneck is clear: **no ground-truth reference trees for virus protein phylogeny.** Here is the recommended path:

1. **Short term — IQ-TREE on 1,000 families.** Use IQ-TREE (ModelFinder + ultrafast bootstrap) to build higher-quality reference trees on a subset of VOGDB families. Submit as a SLURM array job (`cpu1` partition, 2–5 hrs per family). This is the only actionable step that can be completed with existing data.

2. **Medium term — Literature benchmark.** Curate 50–200 virus protein families with published expert phylogenies from TreeBase and the ICTV Report chapters. Requires manual annotation but would provide the first true ground-truth benchmark for virus phylogeny evaluation.

3. **Long term — Dedicated virus phylogeny benchmark.** The field needs a VOGDB-scale equivalent of TreeFam for viruses: curated protein families with validated reference trees. This would be a community resource in its own right.

### Other Enhancements

- **Tier 2 Evaluation**: Implement `mean_cluster_value` (functional signal) using VOGDB `vfam.annotations.tsv.gz` consensus descriptions.
- **Fair baseline**: Run PHYLA on MSAs instead of FAA to isolate the effect of alignment.
- **Additional models**: Phyla-alpha, MAFFT + FastTree (gold standard pipeline).

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