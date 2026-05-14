---
description: "Use when submitting SLURM jobs, writing batch scripts, managing conda environments, or running computation on the HPC cluster. Covers partition selection, GPU requests, conda setup, and login node restrictions."
applyTo: "**/*.{sh,py,yaml,yml}"
---

# HPC & SLURM Instructions (virophylo)

## Login Node Rules

- **Never run compute tasks directly on login nodes.** All computation must be submitted via `sbatch` or `srun`.
- **Only login nodes have network access.** Compute nodes cannot access the internet. All data downloads and Python package installations (`pip install`, `conda install`) must be performed on the login node.
- Use `sbatch <script.sh>` to submit batch jobs.
- For quick interactive testing, use `srun --partition=cpu1 --time=00:30:00 --cpus-per-task=2 --pty bash` to get an interactive session on a compute node.
- Monitor jobs with `squeue -u $USER` or `sacct -j <job_id>`.

## SLURM Script Template

### Basic Template (CPU)

```bash
#!/bin/bash
#SBATCH --job-name=<short_name>
#SBATCH --output=../slurm_logs/<name>_%j.out
#SBATCH --error=../slurm_logs/<name>_%j.err
#SBATCH --time=HH:MM:SS
#SBATCH --cpus-per-task=<N>
#SBATCH --partition=cpu1

SCRIPT_DIR="/home/jianpinhe3/virophylo/Phyla"
cd "$SCRIPT_DIR"
source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh
conda activate virophylo
python <script.py> <args>
```

### GPU Template (add these lines)

```bash
#SBATCH --partition=gpu1|gpu2
#SBATCH --gres=gpu:<N>           # gpu1: max 8, gpu2: max 4
```

### Common Options

- `--mem=<size>G` — request specific memory (optional)
- `--time=DD-HH:MM:SS` — use this format for jobs >24h
- `--nodes=1 --ntasks=1` — implied, rarely need to change

## Conda Environment

- **Environment name**: `virophylo`
- **Activation method**: Always use `source /home/jianpinhe3/miniforge3/etc/profile.d/conda.sh` before `conda activate virophylo` in SBATCH scripts.
- **Installing packages**: Use `conda install -n virophylo <package>` or `pip install <package>` (after activating the environment).
- **Do NOT** use `conda init` or modify `~/.bashrc` for conda on the cluster — use the explicit `source` path instead.

## Partitions

### Decision Tree

```
Need GPU?
├─ Yes → How many GPUs?
│        ├─ 1–4 → Are you running PHYLA/ESM-2 or training?
│        │        ├─ Yes → gpu2 (A100, preferred)
│        │        └─ No  → gpu1 (RTX 4080/4090)
│        └─ 5–8 → gpu1 (RTX, up to 8 GPUs)
└─ No  → cpu1
```

### Partition Details

| Partition | GPU Model | GPUs/Node | Best For |
|-----------|-----------|-----------|----------|
| `cpu1` | — | — | FastTree, data processing, tree comparison |
| `gpu1` | RTX 4080/4090 | 8 | Light inference, debugging, general GPU |
| `gpu2` | A100 (80GB) | 4 | PHYLA, ESM-2, large model training/inference |

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
