---
description: "Use when submitting SLURM jobs, writing batch scripts, managing conda environments, or running computation on the HPC cluster. Covers partition selection, GPU requests, conda setup, and login node restrictions."
applyTo: ["**/*.sh", "**/*.py", "**/*.yaml", "**/*.yml"]
---

# HPC & SLURM Instructions (virophylo)

## Login Node Rules

- **Never run compute tasks directly on login nodes.** All computation must be submitted via `sbatch` or `srun`.
- Use `sbatch <script.sh>` to submit batch jobs.
- For quick interactive testing, use `srun --partition=cpu1 --time=00:30:00 --cpus-per-task=2 --pty bash` to get an interactive session on a compute node.
- Monitor jobs with `squeue -u $USER` or `sacct -j <job_id>`.

## SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=<short_name>
#SBATCH --output=<path_to_logs>/<name>_%j.out
#SBATCH --error=<path_to_logs>/<name>_%j.err
#SBATCH --time=HH:MM:SS          # Format: DD-HH:MM:SS for >24h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<N>
#SBATCH --mem=<size>G            # Optional: request specific memory
#SBATCH --partition=<partition>  # gpu2 for A100, gpu1 for RTX4080/4090, cpu1 for CPU

# GPU jobs: add --gres (gpu2: 4×A100, gpu1: 8×RTX)
#SBATCH --gres=gpu:<N>           # Number of GPUs (max 4 for gpu2, 8 for gpu1)

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"

# Activate conda environment (explicit path — do NOT use conda init)
source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo

# Run your command
python <script.py> <args>
```

## Conda Environment

- **Environment name**: `virophylo`
- **Activation method**: Always use `source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh` before `conda activate virophylo` in SBATCH scripts.
- **Installing packages**: Use `conda install -n virophylo <package>` or `pip install <package>` (after activating the environment).
- **Do NOT** use `conda init` or modify `~/.bashrc` for conda on the cluster — use the explicit `source` path instead.

## Partitions

| Partition | GPU | Nodes | GPU Details | Use Case |
|-----------|-----|-------|-------------|----------|
| `gpu1` | Yes | `gpu[1-2]` | **RTX 4080** × 8 each | 默认分区；通用 GPU 任务（轻量推理、调试） |
| `gpu1` | Yes | `gpu3` | **RTX 4090** × 8 | 默认分区；单卡显存更大 |
| `gpu2` | Yes | `gpu[11-12]` | **NVIDIA A100** × 4 each | **首选分区**；大模型推理（PHYLA、ESM-2）、训练任务 |
| `cpu1` | No | `cpu[1,3-7]` | — | CPU 任务（FastTree、数据处理、树比较） |

## Best Practices

1. **Always check GPU availability**: Include `python -c "import torch; print(torch.cuda.is_available())"` in GPU job scripts for debugging.
2. **Log everything**: Redirect stdout/stderr to `slurm_logs/` directory. Include timestamps and job metadata in output.
3. **Set realistic time limits**: Check SLURM logs to estimate runtime and adjust `--time` accordingly to avoid premature termination or excessive queue wait.
4. **Test with small jobs first**: Use `--time=00:30:00` and a small subset of data before scaling to full datasets.
5. **Check exit codes**: Always check `$?` after critical commands and handle failures gracefully. Use `set -e` at the top of scripts to stop on first error when appropriate.

## Python Script Patterns

- Use `argparse` for CLI arguments so scripts are SLURM-friendly.
- When writing a Python script that will be submitted via SLURM, use:
  ```python
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  ```
- Save intermediate results (checkpoints) periodically for long-running jobs.
- Use `multiprocessing` or `torch.utils.data.DataLoader` with `num_workers` set to match `--cpus-per-task`.
