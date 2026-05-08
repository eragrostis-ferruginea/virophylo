import torch
import numpy as np
import os
import json
import yaml
from tqdm import tqdm
from Bio import SeqIO
from src.models.tree.tree_metrics import TreeMetrics, compute_rf_distance, compute_quartet_accuracy, compute_branch_length_correlation
from src.models.tree.nj_builder import nj_from_distance_matrix
from src.models.distance.k2p_baseline import K2PDistance, compute_k2p_matrix
from src.data.viral_dataset import CompositionFeatureExtractor


class Benchmark:
    def __init__(self, config_path=None, output_dir="outputs/benchmark"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = TreeMetrics()
        self.k2p = K2PDistance()
        self.comp_extractor = CompositionFeatureExtractor(k=4)

        self.datasets = {}
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f)
            self.datasets = config.get("datasets", {})

    def load_dataset(self, name, aln_path, tree_path=None):
        sequences, names = [], []
        for record in SeqIO.parse(aln_path, "fasta"):
            sequences.append(str(record.seq).upper())
            names.append(record.id)

        ref_tree = None
        if tree_path and os.path.exists(tree_path):
            with open(tree_path) as f:
                ref_tree = f.read().strip()

        self.datasets[name] = {
            "sequences": sequences,
            "names": names,
            "ref_tree": ref_tree,
            "aln_path": aln_path,
        }
        return sequences, names, ref_tree

    def evaluate_method(self, method_name, predict_fn, dataset_name=None):
        results = {}
        datasets_to_eval = {dataset_name: self.datasets[dataset_name]} if dataset_name else self.datasets

        for name, data in tqdm(datasets_to_eval.items(), desc=f"Evaluating {method_name}"):
            if data["ref_tree"] is None:
                print(f"Skipping {name}: no reference tree")
                continue

            try:
                pred_tree = predict_fn(data["sequences"], data["names"])
                metrics = self.metrics.evaluate(pred_tree, data["ref_tree"], dataset_name=f"{method_name}/{name}")
                results[name] = metrics
            except Exception as e:
                print(f"Error evaluating {method_name} on {name}: {e}")
                results[name] = {"error": str(e)}

        return results

    def evaluate_k2p_baseline(self, dataset_name=None):
        def predict_fn(sequences, names):
            dist = compute_k2p_matrix(sequences)
            return nj_from_distance_matrix(dist, names)

        return self.evaluate_method("K2P+NJ", predict_fn, dataset_name)

    def evaluate_model(self, model, dataset_name=None, device="cuda"):
        def predict_fn(sequences, names):
            model.eval()
            with torch.no_grad():
                comp_features = self.comp_extractor.extract_batch(sequences)
                comp_tensor = torch.from_numpy(comp_features).to(device)
                newick, _ = model.predict_tree(sequences, names, composition_features=comp_tensor)
            return newick

        return self.evaluate_method(type(model).__name__, predict_fn, dataset_name)

    def save_results(self, results, filename="benchmark_results.json"):
        path = os.path.join(self.output_dir, filename)
        serializable = {}
        for method, datasets in results.items():
            serializable[method] = {}
            for ds, metrics in datasets.items():
                if isinstance(metrics, dict):
                    serializable[method][ds] = {
                        k: float(v) if v is not None else None
                        for k, v in metrics.items()
                    }
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
        return path

    def print_comparison_table(self, all_results):
        header = f"{'Method':<25} {'Dataset':<20} {'nRF':>8} {'QA':>8} {'BL_r':>8}"
        print(header)
        print("=" * len(header))
        for method, datasets in all_results.items():
            for ds, metrics in datasets.items():
                if isinstance(metrics, dict) and "error" not in metrics:
                    nrf = metrics.get("nrf", float('nan'))
                    qa = metrics.get("qa", float('nan'))
                    blr = metrics.get("branch_length_pearson_r", float('nan'))
                    print(f"{method:<25} {ds:<20} {nrf:>8.4f} {qa:>8.4f} {blr:>8.4f}")
