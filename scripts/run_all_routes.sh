#!/bin/bash
set -euo pipefail

echo "=== ViroPhylo: Submit All Routes in Parallel ==="
echo "EE HPC Cluster - SLURM 23.11.6"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs outputs

echo "Submitting Baseline Evaluation (cpu1 partition)..."
JOB_BASE=$(sbatch --parsable scripts/slurm_baseline.sh)
echo "  Baseline Job ID: $JOB_BASE"

echo "Submitting Route C - DNABERT-2/NT-500M + LoRA (gpu2 partition, 4x A100)..."
JOB_C=$(sbatch --parsable scripts/slurm_route_c.sh)
echo "  Route C Job ID: $JOB_C"

echo "Submitting Route A - ViralPhyloGPN (gpu1 partition, 4x RTX4080)..."
JOB_A=$(sbatch --parsable scripts/slurm_route_a.sh)
echo "  Route A Job ID: $JOB_A"

echo "Submitting Route B - PHYLAViral Dual Loss (gpu2 partition, 4x A100)..."
JOB_B=$(sbatch --parsable scripts/slurm_route_b.sh)
echo "  Route B Job ID: $JOB_B"

echo ""
echo "=== All 4 jobs submitted in parallel ==="
echo ""
echo "  Baseline:  $JOB_BASE  (cpu1, ~24h)"
echo "  Route C:   $JOB_C     (gpu2/A100, ~7d)"
echo "  Route A:   $JOB_A     (gpu1/RTX4080, ~7d)"
echo "  Route B:   $JOB_B     (gpu2/A100, ~7d)"
echo ""
echo "Note: Route C and Route B both use gpu2 (A100)."
echo "  SLURM will queue them sequentially if only 1 gpu2 node is free."
echo "  If both gpu2 nodes (gpu11, gpu12) are available, they run in parallel."
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Cancel:   scancel $JOB_BASE $JOB_C $JOB_A $JOB_B"
echo "Logs:     ${PROJECT_DIR}/logs/"
echo ""
echo "Check specific job:"
echo "  scontrol show job $JOB_C"
