import torch
import numpy as np
import os
import json
import copy
from itertools import product
from src.models.tree.tree_metrics import TreeMetrics
from src.data.viral_dataset import CompositionFeatureExtractor


class AblationStudy:
    def __init__(self, output_dir="outputs/ablation"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = TreeMetrics()
        self.comp_extractor = CompositionFeatureExtractor(k=4)

    def run_route_c_ablation(self, model_class, backbone, eval_data, device="cuda"):
        ablation_configs = [
            {"name": "full_model", "use_calibration": True, "use_hybrid_distance": True},
            {"name": "no_zca", "use_calibration": False, "use_hybrid_distance": True},
            {"name": "no_hybrid", "use_calibration": True, "use_hybrid_distance": False},
            {"name": "no_calibration_no_hybrid", "use_calibration": False, "use_hybrid_distance": False},
        ]

        results = {}
        for config in ablation_configs:
            model = model_class(
                backbone=backbone,
                embed_dim=backbone.embed_dim,
                use_calibration=config["use_calibration"],
                use_hybrid_distance=config["use_hybrid_distance"],
            ).to(device)

            metrics = self._evaluate_model(model, eval_data, device)
            results[config["name"]] = metrics
            print(f"Ablation {config['name']}: nRF={metrics.get('nrf', 'N/A'):.4f}, QA={metrics.get('qa', 'N/A'):.4f}")

        self._save_results(results, "route_c_ablation.json")
        return results

    def run_loss_ablation(self, model, eval_data, loss_scheduler, device="cuda"):
        loss_configs = [
            {"name": "triple_loss", "alpha": 1.0, "beta": 0.5, "gamma": 0.5},
            {"name": "quartet_only", "alpha": 1.0, "beta": 0.0, "gamma": 0.0},
            {"name": "likelihood_only", "alpha": 0.0, "beta": 1.0, "gamma": 0.0},
            {"name": "distance_only", "alpha": 0.0, "beta": 0.0, "gamma": 1.0},
            {"name": "quartet_plus_likelihood", "alpha": 1.0, "beta": 1.0, "gamma": 0.0},
        ]

        results = {}
        for config in loss_configs:
            loss_scheduler.set_weights(config["alpha"], config["beta"], config["gamma"])
            metrics = self._evaluate_model(model, eval_data, device)
            results[config["name"]] = metrics

        self._save_results(results, "loss_ablation.json")
        return results

    def _evaluate_model(self, model, eval_data, device):
        model.eval()
        all_metrics = []

        for data in eval_data:
            try:
                sequences = data["sequences"]
                names = data["names"]
                ref_tree = data["ref_tree"]

                if ref_tree is None:
                    continue

                comp_features = self.comp_extractor.extract_batch(sequences)
                comp_tensor = torch.from_numpy(comp_features).to(device)

                with torch.no_grad():
                    pred_tree, _ = model.predict_tree(sequences, names, composition_features=comp_tensor)

                metrics = self.metrics.evaluate(pred_tree, ref_tree)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error during ablation evaluation: {e}")
                continue

        if not all_metrics:
            return {"nrf": None, "qa": None, "branch_length_pearson_r": None}

        avg_metrics = {}
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics if m[key] is not None]
            avg_metrics[key] = np.mean(vals) if vals else None

        return avg_metrics

    def _save_results(self, results, filename):
        path = os.path.join(self.output_dir, filename)
        serializable = {}
        for name, metrics in results.items():
            serializable[name] = {
                k: float(v) if v is not None else None
                for k, v in metrics.items()
            }
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
