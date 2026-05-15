#!/usr/bin/env python3
"""
Download and convert Nextstrain virus reference trees to Newick + FAA format.
Parses auspice JSON (v2 format) to extract the ML tree topology and
aligned protein sequences.

Sources: LANL-HIV-DB (HIV), WHO-euro-flu (Influenza), ViennaRNA (CHIKV/TBEV)

Usage (on login node — this is a download, not compute):
  python download_nextstrain_refs.py --output-dir virus_data/nextstrain_refs
"""
import os, sys, json, re, argparse
from collections import defaultdict

def download_json(url, timeout=60):
    """Download Nextstrain JSON dataset."""
    import urllib.request
    print(f'  Downloading {url}...')
    req = urllib.request.Request(url, headers={'User-Agent': 'phyla-eval/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f'  FAILED: {e}')
        return None

def auspice_tree_to_newick(node):
    """Convert auspice JSON tree node to Newick string."""
    children = node.get('children', [])
    name = node.get('name', '')
    branch_length = node.get('branch_attrs', {}).get('length', 0)
    node_str = str(name) if name and not children else ''

    if not children:
        return f'{node_str}:{branch_length}' if branch_length else node_str

    child_newicks = []
    for child in children:
        child_newicks.append(auspice_tree_to_newick(child))

    return f'({",".join(child_newicks)}){node_str}:{branch_length}'

def extract_protein_sequences(data, gene):
    """Extract protein translations from Nextstrain metadata.
    The JSON has 'annotations' or 'translations' keys mapping sequence names to AA."""
    meta = data.get('meta', {})
    annotations = meta.get('annotations', {})
    translations = meta.get('translations', {})

    seqs = {}
    # Try multiple sources for protein sequences
    for source in [annotations, translations, meta.get('genome_annotations', {})]:
        for key, value in source.items():
            if isinstance(value, dict):
                # Could be structured annotations
                for strain, seq in value.items():
                    if isinstance(seq, str) and len(seq) > 10:
                        seqs[strain] = seq

    # Also check root node for AA data
    root = data.get('tree', {})
    for child in iter_nodes(root):
        if 'aa' in child or 'translations' in child:
            aa = child.get('aa') or child.get('translations', {})
            if isinstance(aa, dict):
                for g, s in aa.items():
                    if g == gene or g in ['HA', 'NA', 'env', 'gag', 'pol']:
                        seqs[child.get('name', '')] = s

    return seqs

def iter_nodes(node):
    yield node
    for child in node.get('children', []):
        yield from iter_nodes(child)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='virus_data/nextstrain_refs')
    ap.add_argument('--datasets', default='', help='comma-separated dataset names or leave empty for all')
    args = ap.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Dataset definitions: (name, URL, gene_name, description)
    DATASETS = [
        ('hiv_env',   'https://nextstrain.org/charon/getDataset?prefix=/groups/LANL-HIV-DB/HIV/env',   'env',  'HIV-1 Env protein'),
        ('hiv_gag',   'https://nextstrain.org/charon/getDataset?prefix=/groups/LANL-HIV-DB/HIV/gag',   'gag',  'HIV-1 Gag protein'),
        ('hiv_pol',   'https://nextstrain.org/charon/getDataset?prefix=/groups/LANL-HIV-DB/HIV/pol',   'pol',  'HIV-1 Pol protein'),
        ('flu_h1n1_ha', 'https://nextstrain.org/charon/getDataset?prefix=/groups/WHO-euro-flu/h1n1pdm/WIC/6y/ha', 'HA', 'Influenza H1N1 HA'),
        ('flu_h3n2_ha', 'https://nextstrain.org/charon/getDataset?prefix=/groups/WHO-euro-flu/h3n2/WIC/6y/ha', 'HA', 'Influenza H3N2 HA'),
        ('flu_h1n1_na', 'https://nextstrain.org/charon/getDataset?prefix=/groups/WHO-euro-flu/h1n1pdm/WIC/6y/na', 'NA', 'Influenza H1N1 NA'),
        ('flu_h3n2_na', 'https://nextstrain.org/charon/getDataset?prefix=/groups/WHO-euro-flu/h3n2/WIC/6y/na', 'NA', 'Influenza H3N2 NA'),
        ('chikv',     'https://nextstrain.org/charon/getDataset?prefix=/groups/ViennaRNA/CHIKVnext/v5.1', 'E1', 'Chikungunya virus E1'),
        ('tbev',      'https://nextstrain.org/charon/getDataset?prefix=/groups/ViennaRNA/TBEVnext/v2.0',  'E',  'TBEV Envelope protein'),
    ]

    if args.datasets:
        names = set(args.datasets.split(','))
        DATASETS = [d for d in DATASETS if d[0] in names]

    print(f'Downloading {len(DATASETS)} Nextstrain reference datasets...')
    print(f'Output: {out_dir}')

    for ds_name, url, gene, desc in DATASETS:
        print(f'\n[{ds_name}] {desc}')
        data = download_json(url)

        if not data:
            # Save URL for manual download
            with open(os.path.join(out_dir, f'{ds_name}_MANUAL_URL.txt'), 'w') as f:
                f.write(f'Manual download URL:\n{url}\n')
            continue

        # Extract tree as Newick
        root = data.get('tree', {})
        try:
            nwk = auspice_tree_to_newick(root) + ';'
            nwk_path = os.path.join(out_dir, f'{ds_name}.nwk')
            with open(nwk_path, 'w') as f:
                f.write(nwk)
            # Count tips
            n_tips = sum(1 for _ in iter_nodes(root) if not _.get('children'))
            print(f'  Tree: {n_tips} tips, saved to {ds_name}.nwk')
        except Exception as e:
            print(f'  Tree extraction FAILED: {e}')
            continue

        # Extract protein sequences if available
        seqs = extract_protein_sequences(data, gene)
        if seqs:
            faa_path = os.path.join(out_dir, f'{ds_name}.faa')
            with open(faa_path, 'w') as f:
                for strain, seq in seqs.items():
                    f.write(f'>{strain}\n{seq}\n')
            print(f'  Sequences: {len(seqs)} protein sequences saved to {ds_name}.faa')
        else:
            print(f'  WARNING: No protein sequences found in JSON')

        # Save full JSON for reference
        json_path = os.path.join(out_dir, f'{ds_name}.json')
        with open(json_path, 'w') as f:
            json.dump(data, f)
        print(f'  Full JSON saved to {ds_name}.json')

    # Summary
    nwk_files = [f for f in os.listdir(out_dir) if f.endswith('.nwk')]
    faa_files = [f for f in os.listdir(out_dir) if f.endswith('.faa')]
    print(f'\n{"="*60}')
    print(f'Download complete:')
    print(f'  Tree files:  {len(nwk_files)}')
    print(f'  FAA files:   {len(faa_files)}')
    faa_basenames = set(f.replace('.faa', '') for f in faa_files)
    both = sum(1 for n in nwk_files if n.replace('.nwk', '') in faa_basenames)
    print(f'  With both:   {both}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
