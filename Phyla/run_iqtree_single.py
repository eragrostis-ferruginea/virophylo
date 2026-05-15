#!/usr/bin/env python3
"""
Run IQ-TREE on a single VOGDB family MSA and output the ML tree.
Called by SLURM array job. Usage:
  python run_iqtree_single.py --fam VFAM00001 --msa-dir virus_data/msa \
      --output-dir virus_data/iqtree_trees --threads 4

Output: virus_data/iqtree_trees/VFAM00001.iqtree.nwk (Newick tree)
"""
import os, sys, subprocess, argparse, time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fam', required=True, help='VFAM ID')
    parser.add_argument('--msa-dir', default='virus_data/msa')
    parser.add_argument('--output-dir', default='virus_data/iqtree_trees')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--redo', action='store_true', help='Force re-run if output exists')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    msa_path = os.path.join(script_dir, args.msa_dir, f'{args.fam}.msa')
    out_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Check MSA exists
    if not os.path.exists(msa_path):
        print(f'{args.fam}: MSA not found at {msa_path}')
        return 1

    # Output tree path
    nwk_path = os.path.join(out_dir, f'{args.fam}.iqtree.nwk')

    # Check if already done
    if os.path.exists(nwk_path) and os.path.getsize(nwk_path) > 10 and not args.redo:
        print(f'{args.fam}: already exists, skipping')
        return 0

    # Count sequences for logging
    with open(msa_path) as f:
        n_seqs = sum(1 for line in f if line.startswith('>'))

    # Run IQ-TREE
    # Use ModelFinder + fast tree search (-fast) for speed
    # -m MFP: ModelFinder Plus with tree search
    # -bb 1000: ultrafast bootstrap 1000
    # -nt: threads
    # -pre: prefix for all output files
    # --quiet: minimal stdout output
    prefix = os.path.join(out_dir, args.fam)
    # `--fast` conflicts with `-bb`, so skip bootstrap since we only need ML tree
    cmd = [
        'iqtree', '-s', msa_path,
        '-m', 'MFP',
        '--fast',
        '-nt', str(args.threads),
        '-pre', prefix,
        '--quiet'
    ]

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f'{args.fam}: IQ-TREE failed (rc={result.returncode}, {elapsed:.0f}s)')
            # Check stderr for details
            stderr_lines = result.stderr.strip().split('\n')[-5:]
            for l in stderr_lines:
                if l.strip():
                    print(f'  STDERR: {l.strip()}')
            return 1

        # IQ-TREE writes the tree to .contree (bootstrap consensus tree)
        # or .treefile (best ML tree). We want the best ML tree.
        treefile = f'{prefix}.treefile'
        if os.path.exists(treefile):
            import shutil
            shutil.copy(treefile, nwk_path)
            print(f'{args.fam}: OK ({n_seqs} seqs, {elapsed:.0f}s)')
            return 0
        else:
            print(f'{args.fam}: .treefile not found after run ({elapsed:.0f}s)')
            return 1

    except subprocess.TimeoutExpired:
        print(f'{args.fam}: TIMEOUT (7200s), {n_seqs} seqs')
        return 1
    except Exception as e:
        print(f'{args.fam}: ERROR: {e}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
