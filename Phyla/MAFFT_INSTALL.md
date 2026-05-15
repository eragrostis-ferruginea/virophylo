# MAFFT Installation Guide

## Option 1: Conda (Recommended)

```bash
conda activate virophylo
conda install -c bioconda mafft -y
```

## Option 2: System Package Manager

### Ubuntu/Debian
```bash
sudo apt-get install mafft
```

### CentOS/RHEL
```bash
sudo yum install mafft
```

### macOS (Homebrew)
```bash
brew install mafft
```

## Option 3: Manual Installation

1. Download from: https://mafft.cbrc.jp/alignment/software/
2. Extract and add to PATH

## Verify Installation

```bash
which mafft
mafft --version
```
