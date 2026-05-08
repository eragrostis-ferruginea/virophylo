# ViroPhylo v2: 基于 LLM 的病毒系统发育分析系统

> 基于 Stage 1-4 实验教训的全面修订版。核心变化：从"从头训练"转向"预训练模型 + 系统发育微调"，从"纯 LLM"转向"LLM + 传统方法混合"，从"macOS 受限"转向"HPC 全力运行"。

---

## 1. 前期实验教训总结

### 1.1 已验证的失败模式

| 失败模式 | 实验阶段 | 根因 | 后果 |
|----------|----------|------|------|
| 从头训练小模型 | Stage 4 Exp 1-2 | 5.78M 参数 + 无预训练 + 仅 quartet loss | QA ≈ 0.28-0.45（随机水平），训练不收敛 |
| PHYLA 在 macOS 上复现失败 | Stage 4 Exp 3 | RMSNorm→LayerNorm 权重失配 + CPU 推理 + mamba-ssm 后端差异 | normRF ≈ 0.54（随机水平），论文 ~0.13 无法复现 |
| 原始 LLM 嵌入做系统发育 | Stage 1-3 | 嵌入各向异性（cosine ∈ [0.85, 0.99]）；信号源为组成偏好（GC corr=0.949） | 无法追踪进化距离，无法克服 LBA |
| 跨数据集泛化 | Stage 3 V11 | Cnidaria 上调参的方法在 Microsporidia 上失败（1/4） | 方法不可迁移 |

### 1.2 已验证的成功要素

| 成功要素 | 实验阶段 | 效果 | 关键数据 |
|----------|----------|------|----------|
| 特征融合（LLM + k-mer/组成特征） | Stage 1 | Pearson r 从 -0.63 提升至 -0.93 | 接近 NW 的 -0.994 |
| ZCA 白化 + Rank 变换 | Stage 3 V9 | 首次无先验知识恢复 Cnidaria 单系（3/4） | 白化后与组成偏好统计独立（shuffled corr = -0.004） |
| NT-500M 与 DNABERT-2 互补 | Stage 3 V7 | NT 抗 LBA + DB 恢复单系性 | Procrustes sim ≈ 0（正交信号） |
| 位点保守性加权 | Stage 3 V11 | 独立缓解 LBA（1/4→2/4） | 验证 PhyloGPN 核心假设 |
| LLM + K2P 混合 | Stage 3 V10 | 达到金标准 4/4 | LLM 贡献 3/4，K2P 贡献 1/4 |
| DNA 级输入优于蛋白级 | Stage 1 | DNABERT-2 raw r=-0.63 vs ESM-2 raw r=-0.11 | DNA 模型保留更多进化信号 |

### 1.3 核心认知升级

**旧认知**（v1 框架）：LLM 可以从头学习进化推理，只需更大的模型和更多的数据。

**新认知**（v2 框架）：
1. **进化推理不会从标准序列建模中涌现**（Ektefaie et al., NeurIPS 2025 铁证）
2. **必须显式注入系统发育监督信号**（quartet loss + phylogenetic likelihood loss）
3. **预训练 DNA LLM 是必要起点**，而非从零训练
4. **嵌入各向异性是根本瓶颈**，必须内置白化/校准
5. **LLM + 传统方法混合 > 纯 LLM**（互补性原则）
6. **PHYLA 本身尚未超越传统方法**，需要更优的训练目标和架构

---

## 2. 修订后的实现路线

### 2.1 三路线策略（基于前期实验的优先级排序）

```
Route C (优先)          Route A (并行)          Route B (长期)
预训练 DNA LLM          PhyloGPN 范式           PHYLA + 似然双训练
+ 系统发育 LoRA         转移到病毒              + 病毒特异性
    │                       │                       │
    ▼                       ▼                       ▼
 快速验证可行性         核心创新突破            终极性能目标
 (2-3 周)              (4-6 周)               (6-8 周)
```

### 2.2 HPC 环境带来的关键改变

| 之前（macOS） | 现在（HPC） | 影响 |
|---------------|-------------|------|
| CPU 推理，无 CUDA | 多 GPU CUDA 训练 | 可正确运行 Mamba SSM、FlashAttention |
| RMSNorm 替换为 LayerNorm | 原生 Triton + RMSNorm | PHYLA 权重可正确加载 |
| 5.78M 参数小模型 | 可训练 100M-500M 模型 | 架构不再受限 |
| 440 条病毒序列训练 | 可处理百万级序列 | 数据不再受限 |
| BF16 不支持 | BF16 混合精度训练 | 训练稳定性和速度大幅提升 |
| 单卡推理 | DeepSpeed ZeRO-3 多卡训练 | 分布式训练可行 |

---

## 3. Route C: 预训练 DNA LLM + 系统发育微调（优先执行）

### 3.1 核心思路

利用已有预训练 DNA 语言模型（DNABERT-2 / Nucleotide Transformer）的序列表征能力，通过 LoRA 高效微调 + 系统发育监督信号，重塑嵌入空间使其编码进化关系。

**为什么 Route C 优先**：
- Stage 1 已证明 DNABERT-2 raw embedding 有进化信号（r=-0.63），只需校准
- Stage 3 V9 已证明 ZCA 白化可提取系统发育信号（3/4 无先验）
- LoRA 微调计算量小，可在 HPC 上快速迭代
- PhyloTune (Nat. Comms. 2025) 已证明 DNABERT attention 与替换率相关（r=0.82）

### 3.2 架构设计

```
┌──────────────────────────────────────────────────────────────────┐
│                    ViroPhylo Route C Pipeline                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input: N virus genome sequences {s_1, ..., s_N}                 │
│    │                                                              │
│    ▼                                                              │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  Frozen Pretrained DNA LLM Backbone                      │     │
│  │  (DNABERT-2-117M or NT-500M-multi-species)              │     │
│  │    │                                                     │     │
│  │    ├─▶ LoRA Adapters (r=16, α=32)  ← trainable          │     │
│  │    ├─▶ [CLS] token extraction                            │     │
│  │    └─▶ Per-position embeddings                           │     │
│  └─────────────────────────────────────────────────────────┘     │
│    │                                                              │
│    ▼                                                              │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  Embedding Calibration Module (关键！)                    │     │
│  │    ├─▶ ZCA Whitening (learnable whitening matrix)        │     │
│  │    ├─▶ Composition Debiasing (regress out GC/k-mer)      │     │
│  │    └─▶ Rank Transform (eliminate anisotropy)             │     │
│  └─────────────────────────────────────────────────────────┘     │
│    │                                                              │
│    ▼                                                              │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  Phylogenetic Distance Head                               │     │
│  │    ├─▶ Pairwise: d_ij = MLP([e_i; e_j; e_i⊙e_j; |e_i-e_j|])│
│  │    ├─▶ Site-Weighted: w[pos] = f(entropy, gap_frac)      │     │
│  │    └─▶ Hybrid: d_final = α·d_LLM + (1-α)·d_K2P          │     │
│  └─────────────────────────────────────────────────────────┘     │
│    │                                                              │
│    ▼                                                              │
│  NJ / FastME → Phylogenetic Tree                                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 训练目标（三重损失）

```python
L_total = α · L_quartet + β · L_phylo_likelihood + γ · L_distance
```

**L_quartet（四重奏拓扑损失，参考 PHYLA）**：
- 从 n 条序列中采样四重奏 {a, b, c, d}
- 真实拓扑由参考树决定（三种拓扑之一）
- 预测：基于预测距离矩阵的四重奏拓扑概率
- 损失：交叉熵

**L_phylo_likelihood（系统发育似然损失，参考 PhyloGPN）**：
- 使用 GTR+Γ 替换模型（比 F81 更适合病毒的高突变率）
- 对每个位点，计算在参考树下的 Felsenstein 似然
- 模型预测 GTR 模型参数（6 个替换率 + 4 个稳态频率 + Γ 形状参数）
- 损失：负对数似然

**L_distance（距离回归损失）**：
- 预测成对进化距离 vs 参考树的 patristic distance
- 损失：Huber Loss（对异常值鲁棒）

**权重调度**：
- 前期：α=1.0, β=0.0, γ=0.5（先学拓扑）
- 中期：α=0.5, β=0.5, γ=0.5（引入似然约束）
- 后期：α=0.3, β=0.7, γ=0.3（强化进化模型拟合）

### 3.4 嵌入校准模块（核心创新，基于 Stage 1-3 教训）

**问题**：Stage 1-3 反复证明原始 LLM 嵌入存在严重各向异性（cosine ∈ [0.85, 0.99]），且信号源为组成偏好（GC content），而非位置同源。

**解决方案**：内置可学习的嵌入校准模块

```python
class EmbeddingCalibration(nn.Module):
    def __init__(self, embed_dim, n_composition_features=20):
        super().__init__()
        # 可学习白化矩阵（初始化为 ZCA 白化矩阵）
        self.whitening_matrix = nn.Parameter(torch.eye(embed_dim))
        self.whitening_bias = nn.Parameter(torch.zeros(embed_dim))

        # 组成偏好去除（回归去除 GC/k-mer 信号）
        self.composition_regressor = nn.Linear(n_composition_features, embed_dim)

        # 位点保守性加权（PhyloGPN 启发）
        self.site_weight_net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings, composition_features=None):
        # Step 1: ZCA 白化
        e_whitened = embeddings @ self.whitening_matrix + self.whitening_bias

        # Step 2: 组成偏好去除
        if composition_features is not None:
            composition_signal = self.composition_regressor(composition_features)
            e_debiased = e_whitened - composition_signal
        else:
            e_debiased = e_whitened

        # Step 3: 位点保守性加权
        site_weights = self.site_weight_net(embeddings)
        e_weighted = e_debiased * site_weights

        return e_weighted
```

### 3.5 混合距离策略（基于 Stage 3 V10-V11 教训）

Stage 3 V10 证明 LLM+K2P 混合达到金标准，V11 证明位点加权 K2P 独立缓解 LBA。

```python
class HybridDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.8))  # LLM 权重（可学习）
        self.cross_branch_weight = nn.Parameter(torch.tensor(0.7))

    def forward(self, d_llm, d_k2p, is_cross_branch_mask):
        # 基础混合
        d_hybrid = self.alpha * d_llm + (1 - self.alpha) * d_k2p

        # 跨分支校正（Stage 3 V10 策略）
        correction = torch.relu(d_k2p - d_llm) * self.cross_branch_weight
        d_corrected = d_hybrid + correction * is_cross_branch_mask

        return d_corrected
```

### 3.6 训练数据

| 数据集 | 来源 | 规模 | 用途 |
|--------|------|------|------|
| HIV-1 LANL | LANL Database | ~10,000 序列 | 微调 + 评估 |
| SARS-CoV-2 Nextstrain | GISAID/Nextstrain | ~5,000 采样 | 微调 + 评估 |
| Influenza GISAID | GISAID EpiFlu | ~5,000 采样 | 微调 + 评估 |
| Dengue Virus | ViPR | ~3,000 序列 | 评估 |
| HCV | LANL/ViPR | ~2,000 序列 | 评估 |
| TreeBASE | treebase.org | ~1,500 比对+树 | 通用评估 |
| 模拟数据 | DendroPy/INDELible | 10,000 棵树 | 训练+评估 |

**关键改进**：之前仅 440 条序列，现在至少 25,000 条，且每个病毒家族都有参考树。

### 3.7 LoRA 微调配置

```yaml
# configs/train/route_c_lora.yaml
backbone: "InstaDeepAI/nucleotide-transformer-500m-multi-species"
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.1
lora_target_modules:
  - "query"
  - "value"
  - "dense"
trainable_params: "~5M (LoRA) + ~2M (heads) = ~7M total"
learning_rate: 1e-4
batch_size: 32
gradient_accumulation: 4
epochs: 20
scheduler: "cosine_with_warmup"
warmup_ratio: 0.1
precision: "bf16"
```

---

## 4. Route A: PhyloGPN 范式转移到病毒系统发育（并行启动）

### 4.1 核心思路

将 PhyloGPN 的"系统发育似然训练 + 单序列推理"范式从哺乳动物基因组转移到病毒基因组。关键区别：病毒进化更快、基因组更短但数量更大、重组频繁。

### 4.2 架构设计

```
Input: 481bp virus genome window x(i)
  │
  ▼
One-Hot Encoder (A/C/G/T/N + pad)
  │
  ▼
RCE 1D Conv (kernel=9)  ← Reverse-Complement Equivariant
  │
  ▼
┌──────────────────────────────────────┐
│  ByteNet-style CNN (adapted from     │
│  PhyloGPN/CARP)                      │
│  40 ByteNet Blocks with RCE conv     │
│  Dilation: [1,5] alternating         │
│  Hidden dim: 960                     │
└──────────────────────────────────────┘
  │
  ▼
RCE 1D Conv → GeLU → LayerNorm
  │
  ▼
GTR Parameter Head:
  ├─ 6 substitution rates (A↔C, A↔G, A↔T, C↔G, C↔T, G↔T)
  ├─ 4 stationary frequencies (π_A, π_C, π_G, π_T)
  └─ 1 Γ shape parameter (alpha)
  │
  ▼
Felsenstein Pruning → L_phylo_likelihood
```

### 4.3 病毒特异性适配

| PhyloGPN (哺乳动物) | ViroPhylo (病毒) | 原因 |
|---------------------|-------------------|------|
| F81 替换模型 | GTR+Γ 替换模型 | 病毒替换率不对称性更强 |
| 447 哺乳动物全基因组比对 | 病毒 MSA（HIV/SARS/Flu） | 病毒比对更短但更密集 |
| 481bp 窗口 | 241bp 窗口（RNA 病毒基因） | 病毒基因更短 |
| 单参考基因组 | 多参考毒株 | 病毒无单一"参考基因组" |
| 无重组处理 | 重组感知训练 | 病毒重组频繁 |

### 4.4 训练数据构建

```
For each virus family:
  1. 收集 MSA (MAFFT --auto)
  2. 用 IQ-TREE 2 建参考树 (GTR+Γ+ASC)
  3. 对每个位点 i:
     - x(i) = 241bp 窗口（中心为参考序列第 i 位）
     - y(i) = 比对列中其他物种的核苷酸
     - T(i) = 参考树的最小生成子树
  4. 过滤：去除 gap 比例 >50% 的位点
```

### 4.5 损失函数

```python
# PhyloGPN 式损失 + 病毒适配
L_phylo = -1/n * sum_i log P_GTR(y_i | theta_i, T_i)  # GTR 似然
L_cond  = +1/n * sum_i log pi_i(theta)                  # 条件化（避免过拟合中心核苷酸）
L_stable = upper_bound(L_phylo + L_cond)                 # 数值稳定上界（sigmoid 近似）

# 额外：位点保守性加权
w_i = clip(1 - H_i/2.0, 0.05, 1.0) * (1 - f_gap_i)     # Stage 3 V11 验证有效
L_total = 1/n * sum_i w_i * L_stable_i
```

---

## 5. Route B: PHYLA 架构 + 系统发育似然双训练（长期目标）

### 5.1 核心思路

保留 PHYLA 的 BiMamba + Tree Head 多序列联合处理架构，但增加系统发育似然损失，实现双目标联合优化。

### 5.2 关键改进（基于 Stage 4 失败教训）

1. **在 HPC 上正确运行 PHYLA**：CUDA + Triton + 原生 RMSNorm，不再需要 macOS 替换
2. **增加系统发育似然损失**：PHYLA 仅用 quartet loss，我们增加 GTR 似然约束
3. **核苷酸级别适配**：PHYLA 处理蛋白质，我们适配为核苷酸输入
4. **更大训练数据**：3,000 蛋白质家族 → 10,000+ 病毒比对

### 5.3 架构

```
Input: N virus sequences {[CLS] tok(s_1) ∥ [CLS] tok(s_2) ∥ ... ∥ [CLS] tok(s_N)}
  │
  ▼
┌──────────────────────────────────────────────┐
│  BiMamba (Bidirectional Mamba SSM)           │
│  - 跨序列扫描：每个位点扫描 N 条序列          │
│  - Weight tying: forward/backward 共享权重    │
│  - CUDA parallel scan (HPC 优势)             │
└──────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────┐
│  Tree Head (MultiheadAttention)              │
│  - Query: [CLS] tokens                       │
│  - Key/Value: sequence tokens                │
│  - 输出: N 个序列级嵌入 e_1, ..., e_N        │
└──────────────────────────────────────────────┘
  │
  ├─▶ Distance Head: d_ij = ||e_i - e_j||_2
  │     └─▶ NJ → Tree → L_quartet
  │
  └─▶ GTR Parameter Head: theta_i = MLP(e_i)
        └─▶ Felsenstein Pruning → L_phylo_likelihood
```

### 5.4 双损失训练

```python
L_total = α · L_quartet + β · L_phylo_likelihood

# 动态权重调整
α = max(0.3, 1.0 - epoch / total_epochs)    # quartet 权重递减
β = min(0.7, epoch / total_epochs)           # 似然权重递增
```

**为什么双损失有效**：
- Quartet loss 确保拓扑正确性（全局结构）
- Phylogenetic likelihood loss 确保进化模型拟合（局部细节）
- 两者互补：quartet 关注"谁和谁最近"，likelihood 关注"进化了多少"

---

## 6. 统一评估框架

### 6.1 评估数据集（必须包含跨数据集验证！）

| 数据集 | 病毒 | 序列数 | 参考树来源 | 用途 |
|--------|------|--------|-----------|------|
| HIV-1 pol | Retroviridae | 500 | LANL | 同域评估 |
| SARS-CoV-2 spike | Coronaviridae | 500 | Nextstrain | 同域评估 |
| Influenza A HA | Orthomyxoviridae | 500 | GISAID | 同域评估 |
| Dengue E | Flaviviridae | 200 | ViPR | 同域评估 |
| HCV E2 | Flaviviridae | 200 | LANL | 同域评估 |
| RSV | Pneumoviridae | 200 | GenBank | **跨域评估** |
| Rabies lyssavirus | Rhabdoviridae | 150 | GenBank | **跨域评估** |
| Microsporidia 18S | Eukaryote | 28 | NCBI | **LBA 评估**（Stage 3 基准） |
| TreeBASE | Mixed | 1,533 | treebase.org | 通用评估 |
| 模拟数据 | Simulated | 10,000 trees | DendroPy | 受控评估 |

### 6.2 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| normalized RF (nRF) | 拓扑距离 | < 0.15（PHYLA 论文 ~0.13） |
| Quartet Accuracy (QA) | 四重奏恢复率 | > 0.85 |
| Kuhner-Felsenstein (KF) | 拓扑+分支长度 | 越低越好 |
| Branch Length Pearson r | 分支长度准确性 | > 0.90 |
| LBA Score | Microsporidia 18S 4/4 标准 | ≥ 3/4（无先验） |
| Cross-dataset nRF | 跨病毒家族泛化 | < 0.30 |

### 6.3 对比基线

| 方法 | 类型 | 说明 |
|------|------|------|
| NJ + K2P | 传统距离法 | Stage 3 基线（QA=1.0 on HIV/SARS） |
| IQ-TREE 2 (GTR+Γ) | 传统 ML | 当前主流 |
| FastTree 2 | 传统近似 ML | 快速基线 |
| FastME | 传统距离法 | Phyloformer 对比基线 |
| PHYLA (24M, CUDA) | DL 方法 | 在 HPC 上正确复现 |
| Phyloformer | DL 方法 | Attention 距离估计 |
| Hamming + NJ | 最简单基线 | PHYLA 论文中的下限 |

### 6.4 消融实验（必须做！）

| 消融项 | 验证什么 | 预期影响 |
|--------|----------|----------|
| 无 ZCA 白化 | 嵌入校准的必要性 | QA 大幅下降（Stage 1-3 教训） |
| 无组成偏好去除 | debiasing 的必要性 | 信号退化为 GC content |
| 无位点加权 | site weighting 的必要性 | LBA 加剧（V11 教训） |
| 无 K2P 混合 | 混合距离的必要性 | 无法达到金标准（V10 教训） |
| 仅 quartet loss | 似然损失的必要性 | 训练不收敛（Stage 4 教训） |
| 仅 phylo likelihood | quartet loss 的必要性 | 拓扑精度下降 |
| 无预训练（从零训练） | 预训练的必要性 | 随机水平（Stage 4 Exp 1-2 教训） |

---

## 7. HPC 部署与运行

### 7.1 环境配置

```bash
# HPC conda environment
conda create -n virophylo python=3.11 -y
conda activate virophylo

# PyTorch with CUDA
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Mamba SSM (需要 CUDA)
pip install mamba-ssm causal-conv1d

# Flash Attention
pip install flash-attn --no-build-isolation

# 其他依赖
pip install transformers>=4.40.0 datasets accelerate deepspeed
pip install biopython ete3 dendropy treeswift
pip install scikit-learn scipy numpy pandas
pip install mafft  # 或系统级安装

# IQ-TREE 2 (系统级)
conda install -c bioconda iqtree2
```

### 7.2 典型 SLURM 任务脚本

```bash
#!/bin/bash
#SBATCH --job-name=virophylo_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.log

module load cuda/12.1
conda activate virophylo

# Route C: LoRA fine-tuning
torchrun --nproc_per_node=4 \
    src/training/route_c_train.py \
    --config configs/train/route_c_lora.yaml \
    --output_dir outputs/route_c_run1 \
    --bf16 \
    --deepspeed configs/deepspeed/zeiro3.json
```

### 7.3 项目结构

```
virophylo/
├── src/
│   ├── models/
│   │   ├── backbone/
│   │   │   ├── dnabert2_wrapper.py     # DNABERT-2 + LoRA
│   │   │   ├── nt_wrapper.py           # Nucleotide Transformer + LoRA
│   │   │   └── phyla_model.py          # PHYLA BiMamba + Tree Head (CUDA)
│   │   ├── route_a/
│   │   │   ├── viral_phylogpn.py       # PhyloGPN 病毒适配版
│   │   │   ├── gtr_module.py           # GTR+Γ 替换模型
│   │   │   └── felsenstein.py          # Felsenstein pruning 向量化
│   │   ├── route_b/
│   │   │   ├── phyla_viral.py          # PHYLA + GTR 双损失
│   │   │   └── bimamba_block.py        # BiMamba SSM block
│   │   ├── calibration/
│   │   │   ├── zca_whitening.py        # 可学习 ZCA 白化
│   │   │   ├── composition_debias.py   # 组成偏好去除
│   │   │   └── site_weighting.py       # 位点保守性加权
│   │   ├── distance/
│   │   │   ├── distance_head.py        # 进化距离预测头
│   │   │   ├── hybrid_distance.py      # LLM+K2P 混合距离
│   │   │   └── k2p_baseline.py         # K2P 距离计算
│   │   └── tree/
│   │       ├── nj_builder.py           # Neighbor-Joining
│   │       └── tree_metrics.py         # RF, KF, QA 计算
│   ├── data/
│   │   ├── viral_dataset.py            # 病毒基因组数据集
│   │   ├── phylo_dataset.py            # 系统发育训练数据
│   │   ├── treebase_loader.py          # TreeBASE 数据加载
│   │   ├── nextstrain_loader.py        # Nextstrain 数据加载
│   │   └── simulation.py              # 模拟数据生成
│   ├── training/
│   │   ├── route_c_train.py            # Route C 训练入口
│   │   ├── route_a_train.py            # Route A 训练入口
│   │   ├── route_b_train.py            # Route B 训练入口
│   │   └── losses.py                   # 所有损失函数
│   ├── evaluation/
│   │   ├── benchmark.py                # 统一评估框架
│   │   ├── lba_eval.py                 # LBA 评估（Stage 3 标准）
│   │   ├── cross_dataset_eval.py       # 跨数据集评估
│   │   └── ablation.py                 # 消融实验
│   └── inference/
│       └── pipeline.py                 # 完整推理管线
├── configs/
│   ├── model/
│   │   ├── route_c_nt500m.yaml
│   │   ├── route_c_dnabert2.yaml
│   │   ├── route_a_phylogpn.yaml
│   │   └── route_b_phyla_viral.yaml
│   ├── train/
│   │   ├── route_c_lora.yaml
│   │   ├── route_a_pretrain.yaml
│   │   └── route_b_dual.yaml
│   ├── deepspeed/
│   │   └── zeiro3.json
│   └── eval/
│       └── benchmark.yaml
├── scripts/
│   ├── hpc_submit.sh                   # SLURM 提交脚本
│   ├── prepare_data.sh                 # 数据下载和预处理
│   └── run_all_evals.sh               # 运行所有评估
├── Articles/                           # 参考文献PDF
├── IMPLEMENTATION.md                   # 本文件
└── README.md
```

---

## 8. 开发时间线

### Phase 0: 环境搭建与基线复现（1 周）

- [ ] HPC 环境配置（CUDA, mamba-ssm, flash-attn, deepspeed）
- [ ] 在 HPC 上正确运行 PHYLA（验证 CUDA 消除 macOS 问题）
- [ ] 运行传统基线（NJ+K2P, IQ-TREE2）在所有评估数据集上
- [ ] 数据下载和预处理（HIV LANL, SARS-CoV-2, Influenza, TreeBASE）

**Phase 0 交付物**：
- PHYLA 在 CUDA 上的 normRF 结果（与论文 ~0.13 对比）
- 所有传统方法的基线数字
- 完整的评估数据集

### Phase 1: Route C 实现（2-3 周）

- [ ] 实现 NT-500M / DNABERT-2 + LoRA 包装器
- [ ] 实现嵌入校准模块（ZCA 白化 + 组成偏好去除 + 位点加权）
- [ ] 实现三重损失函数（quartet + phylo_likelihood + distance）
- [ ] 实现混合距离模块（LLM + K2P）
- [ ] 在 HIV-1 + SARS-CoV-2 上训练和评估
- [ ] 消融实验

**Phase 1 交付物**：
- Route C 在 HIV-1/SARS-CoV-2 上的 nRF/QA 数字
- 消融实验结果（验证每个组件的贡献）
- 与传统方法的对比表

### Phase 2: Route A 实现（3-4 周，可与 Phase 1 并行准备）

- [ ] 实现 GTR+Γ 模块和向量化 Felsenstein pruning
- [ ] 实现 ViralPhyloGPN 架构（ByteNet CNN + GTR head）
- [ ] 构建病毒 MSA + 参考树训练数据
- [ ] 训练和评估
- [ ] 与 Route C 对比

**Phase 2 交付物**：
- Route A 的 nRF/QA 数字
- Route A vs Route C 对比
- GTR vs F81 替换模型对比

### Phase 3: Route B 实现（3-4 周）

- [ ] PHYLA 核苷酸适配（氨基酸→核苷酸 tokenizer）
- [ ] 实现双损失训练（quartet + GTR likelihood）
- [ ] 在病毒数据上训练
- [ ] 与 Route A/C 对比

**Phase 3 交付物**：
- Route B 的 nRF/QA 数字
- 三路线对比总结

### Phase 4: 跨域评估与 LBA 测试（1-2 周）

- [ ] 在 RSV、Rabies 等跨域数据上评估
- [ ] Microsporidia 18S LBA 测试（Stage 3 标准）
- [ ] 确定最优路线和配置
- [ ] 撰写最终报告

---

## 9. 风险与缓解措施（基于前期教训更新）

| 风险 | 前期教训 | 缓解措施 |
|------|----------|----------|
| LoRA 微调不足以重塑嵌入空间 | Stage 4 证明标准嵌入+距离计算失败 | 嵌入校准模块（ZCA+debiasing）作为安全网；如果 LoRA 失败，升级为全参数微调 |
| PHYLA 在 HPC 上仍无法复现 | macOS 上 normRF=0.54 vs 论文 0.13 | Phase 0 优先验证；如仍失败，直接用 PHYLA 代码库而非自行实现 |
| 训练不收敛 | Stage 4 Exp 1-2 QA 震荡 | 三重损失 + 权重调度 + warmup；监控每个损失分量 |
| 跨数据集泛化差 | Microsporidia 仅 1/4 | 必须在多个病毒家族上训练；留出法评估 |
| 组成偏好无法完全去除 | DNABERT-2 LOO-CV R²=1.0 | ZCA 白化 + 显式回归去除 + rank 变换三重保险 |
| GTR 似然计算数值不稳定 | PhyloGPN 论文提到 double exponential | 使用 sigmoid 近似上界（PhyloGPN Eq.5） |
| 病毒 MSA 质量差 | 病毒序列高度分歧 | 使用 MAFFT --auto + gap 过滤；对齐后人工检查 |

---

## 10. 成功标准

### 最低标准（必须达到）

1. **Route C 在 HIV-1/SARS-CoV-2 上 nRF < 0.30**（优于随机基线 0.50）
2. **PHYLA 在 HPC 上成功复现**（normRF < 0.20）
3. **消融实验验证每个组件的贡献**

### 目标标准（期望达到）

1. **至少一条路线在 HIV-1 上 nRF < 0.15**（接近 PHYLA 论文水平）
2. **LBA 测试 ≥ 3/4（无先验）**（超越 Stage 3 V9 的最佳零样本结果）
3. **跨域评估 nRF < 0.25**

### 理想标准（超越传统方法）

1. **在位点间依赖模型下超越 IQ-TREE**（Phyloformer 已证明可行）
2. **推理速度 > 10× IQ-TREE**
3. **LBA 测试 4/4（无先验）**（当前无任何方法达到）

---

## 11. 参考文献

1. Albors C, Canal Li J, Benegas G, Ye C, Song YS. A Phylogenetic Approach to Genomic Language Modeling. arXiv:2503.03773v2, 2026.
2. Ektefaie Y, Shen A, Jain L, Farhat M, Zitnik M. Evolutionary Reasoning Does Not Arise in Standard Usage of Protein Language Models. NeurIPS 2025.
3. Zvyagin M, et al. GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics. IJHPCA, 2023.
4. Zhang X, Ding S, et al. NeuralNJ: Accurate and Efficient Phylogenetic Inference through End-To-End Deep Learning. MBE, 2025.
5. Nesterenko L, Blassel L, et al. Phyloformer: Fast, Accurate, and Versatile Phylogenetic Reconstruction with Deep Neural Networks. MBE, 2025.
6. Benegas G, et al. GPN-MSA: A DNA language model based on multispecies alignment. Nature Biotechnology, 43, 1960-1965, 2025.
7. PhyloTune: Pretrained DNA language model for efficient phylogenetic tree updating. Nature Communications, 2025.
8. Braichenko S, Borges R, Kosiol C. Phylogenetic Methods Meet Deep Learning. GBE, 2025.
9. Zhou Z, et al. DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome. ICLR, 2024.
10. Dalla Torre N, et al. The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics. Nature Methods, 2024.
