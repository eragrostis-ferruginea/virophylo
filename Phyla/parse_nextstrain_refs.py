#!/usr/bin/env python3
"""
Parse Nextstrain v2 JSON trees: extract protein sequences + Newick topology.
For each node, reconstruct AA sequence from root_sequence + accumulated mutations.
"""
import os, sys, json, re, argparse
from Bio.Seq import Seq
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def translate_frames(nuc_seq, frames):
    """Translate CDS frames from genome annotations."""
    translations = {}
    for gene_name, frame_info in frames.items():
        if gene_name == 'nuc': continue
        if isinstance(frame_info, dict):
            start = frame_info.get('start', 0)
            end = frame_info.get('end', len(nuc_seq))
            strand = frame_info.get('strand', 1)
            cds = nuc_seq[start:end]
            if strand == -1:
                cds = str(Seq(cds).reverse_complement())
            try:
                translations[gene_name] = str(Seq(cds).translate())
            except:
                translations[gene_name] = ''
    return translations

def reconstruct_protein(tree_data, gene, max_tips=None):
    """Walk tree, reconstruct AA sequence per node from root + mutations."""
    root_node = tree_data['tree']
    root_nuc = tree_data.get('root_sequence', {}).get('nuc', '')
    frames = tree_data.get('meta', {}).get('genome_annotations', {})

    # Get root translation
    root_trans = translate_frames(root_nuc, frames)
    if gene not in root_trans:
        # Try to find matching gene
        for g in root_trans:
            if g.upper() == gene.upper() or gene.upper() in g.upper():
                gene = g
                break
    root_aa = root_trans.get(gene, '')

    sequences = {}
    # Stack: (node, parent_aa_seq)
    def walk(node, parent_aa):
        name = node.get('name', '')
        muts = node.get('branch_attrs', {}).get('mutations', {})

        # Get AA mutations for this gene
        aa_muts = muts.get(gene, [])
        current_aa = list(parent_aa)

        for mut in aa_muts:
            # Format: "D222G" → position 221 (0-indexed), from D to G
            match = re.match(r'^([A-Z*])(\d+)([A-Z*])$', mut)
            if match:
                ref, pos, alt = match.groups()
                idx = int(pos) - 1
                if 0 <= idx < len(current_aa):
                    current_aa[idx] = alt

        cur_aa_str = ''.join(current_aa)

        if name and not node.get('children'):
            sequences[name] = cur_aa_str

        if max_tips and len(sequences) >= max_tips:
            return

        for child in node.get('children', []):
            walk(child, cur_aa_str)

    walk(root_node, root_aa)
    return sequences

def auspice_to_newick(node):
    children = node.get('children', [])
    name = node.get('name', '')
    bl = node.get('branch_attrs', {}).get('length', 0)
    if not children:
        return f'{name}:{bl}' if name else f':{bl}'
    child_strs = [auspice_to_newick(c) for c in children]
    return f'({",".join(child_strs)}){name}:{bl}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-dir', default='virus_data/nextstrain_refs')
    ap.add_argument('--output-dir', default='virus_data/nextstrain_refs')
    ap.add_argument('--max-tips', type=int, default=200)
    args = ap.parse_args()

    jdir = os.path.join(SCRIPT_DIR, args.json_dir)
    odir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(odir, exist_ok=True)

    # Gene mapping
    GENE_MAP = {k.split('.')[0]: None for k in os.listdir(jdir) if k.endswith('.json')}
    # Manually specify genes
    GENE_OVERRIDE = {
        'hiv_env': 'Env', 'hiv_gag': 'Gag', 'hiv_pol': 'Pol',
        'flu_h1n1_ha': 'HA1', 'flu_h3n2_ha': 'HA1',
        'flu_h1n1_na': 'NA', 'flu_h3n2_na': 'NA',
        'chikv': 'E1', 'tbev': 'E',
    }

    for fname in sorted(os.listdir(jdir)):
        if not fname.endswith('.json'): continue
        ds = fname.replace('.json', '')

        print(f'\n[{ds}]')
        data = json.load(open(os.path.join(jdir, fname)))
        gene = GENE_OVERRIDE.get(ds, 'HA1')

        # Extract tree as Newick
        nwk = auspice_to_newick(data['tree'])
        # Prune to max_tips
        all_tips = []
        def collect_tips(node):
            if not node.get('children'):
                all_tips.append(node)
            for c in node.get('children', []):
                collect_tips(c)
        collect_tips(data['tree'])
        if len(all_tips) > args.max_tips:
            # Keep every Nth tip
            step = max(1, len(all_tips) // args.max_tips)
            keep = set(n.get('name') for i, n in enumerate(all_tips) if i % step == 0)
            # Prune tree to keep only these tips
            def prune(node):
                if not node.get('children'):
                    return node if node.get('name') in keep else None
                children = [prune(c) for c in node['children']]
                children = [c for c in children if c is not None]
                if children:
                    node['children'] = children
                    return node
                return node if node.get('name') in keep else None
            data['tree'] = prune(data['tree']) or data['tree']

        # Get Newick
        nwk_str = auspice_to_newick(data['tree']) + ';'
        nwk_path = os.path.join(odir, f'{ds}.nwk')
        with open(nwk_path, 'w') as f:
            f.write(nwk_str)
        n_tips = sum(1 for _ in open(nwk_path) if ';' not in _)
        print(f'  Newick: ~{n_tips} tips (max {args.max_tips})')

        # Reconstruct protein sequences
        seqs = reconstruct_protein(data, gene, args.max_tips)
        if seqs:
            faa_path = os.path.join(odir, f'{ds}.faa')
            with open(faa_path, 'w') as f:
                for name, seq in seqs.items():
                    f.write(f'>{name}\n{seq}\n')
            print(f'  FAA: {len(seqs)} sequences, length={len(list(seqs.values())[0]) if seqs else 0}')
        else:
            print(f'  FAA: No sequences extracted')

    print('\nDone!')


if __name__ == '__main__':
    main()
