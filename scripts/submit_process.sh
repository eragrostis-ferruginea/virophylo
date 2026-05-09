#!/bin/bash
set -euo pipefail

echo "=== ViroPhylo: Submit Data Processing Job ==="
echo "EE HPC Cluster - SLURM"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

echo "Submitting data processing job (cpu1 partition)..."
JOB_ID=$(sbatch --parsable scripts/slurm_process_data.sh)
echo ""
echo "=== Job Submitted ==="
echo "  Job ID:    $JOB_ID"
echo "  Partition: cpu1"
echo "  CPUs:      8"
echo "  Memory:    32G"
echo "  Time:      24h"
echo ""
echo "Monitor:    squeue -u \$USER"
echo "Cancel:     scancel $JOB_ID"
echo "Logs:       ${PROJECT_DIR}/logs/process_${JOB_ID}.log"