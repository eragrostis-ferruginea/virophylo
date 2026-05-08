#!/bin/bash
set -euo pipefail

echo "=== ViroPhylo HPC Environment Setup ==="
echo "EE HPC / OpenHPC 3.1 / Rocky Linux 9.4 / SLURM 23.11.6"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "--- [1/6] Loading system modules ---"
module load cuda/12.1.0
module avail cuda

echo "--- [2/6] Creating conda environment ---"
if ! conda env list | grep -q "virophylo"; then
    conda create -n virophylo python=3.11 -y
fi
eval "$(conda shell.bash hook)"
conda activate virophylo

echo "--- [3/6] Installing PyTorch with CUDA 12.1 ---"
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "--- [4/6] Installing Mamba SSM and Flash Attention ---"
pip install mamba-ssm causal-conv1d 2>/dev/null || echo "WARNING: mamba-ssm install failed; Route B will use fallback MLP"
pip install flash-attn --no-build-isolation 2>/dev/null || echo "WARNING: flash-attn install failed; using standard attention"

echo "--- [5/6] Installing project dependencies ---"
pip install -r requirements.txt
pip install -e .

echo "--- [6/6] Installing bioinformatics tools ---"
conda install -c bioconda iqtree2 mafft fasttree -y 2>/dev/null || \
    conda install -c conda-forge iqtree2 mafft fasttree -y 2>/dev/null || \
    echo "WARNING: Some bio tools not installed. Install manually: conda install -c bioconda iqtree2 mafft fasttree"

echo ""
echo "=== Verifying installation ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

try:
    import mamba_ssm
    print('Mamba SSM: available')
except ImportError:
    print('Mamba SSM: NOT available (Route B will use fallback)')

try:
    import flash_attn
    print('Flash Attention: available')
except ImportError:
    print('Flash Attention: NOT available')

import transformers
print(f'Transformers: {transformers.__version__}')

import peft
print(f'PEFT: {peft.__version__}')

from src.models.calibration.zca_whitening import EmbeddingCalibration
from src.models.distance.hybrid_distance import HybridDistance
from src.models.tree.nj_builder import nj_from_distance_matrix
print('Core modules: OK')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "EE HPC Cluster Info:"
echo "  Login:     ssh account@hpc1.ee.cityu.edu.hk"
echo "  Partitions: gpu1 (RTX 4080), gpu2 (A100), cpu1"
echo "  Max GPUs:  4 per user (gpu1), 4 per user (gpu2)"
echo "  Max time:  7 days"
echo ""
echo "Next steps:"
echo "  1. Transfer project:  scp -r virophylo/ account@hpc1.ee.cityu.edu.hk:~/"
echo "  2. SSH to HPC:        ssh account@hpc1.ee.cityu.edu.hk"
echo "  3. Setup environment:  bash scripts/setup_hpc.sh"
echo "  4. Prepare data:       bash scripts/prepare_data.sh"
echo "  5. Submit all routes:  bash scripts/run_all_routes.sh"
