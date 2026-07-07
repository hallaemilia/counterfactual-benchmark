"""CLI for CDM, UC and CG metric computation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

from .cdm import compute_cdm
from .cg import compute_cg
from .factor_mapping import discover_latent_factor_mapping
from .uc import compute_uc
from .utils import build_dataset, load_predictors, load_scm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute causal disentanglement metrics.")
    parser.add_argument("--config", "-c", required=True, help="Benchmark Deep-SCM JSON config.")
    parser.add_argument("--classifier-config", "-clf", required=True, help="Benchmark classifier JSON config.")
    parser.add_argument("--model-name", required=True, choices=["vae", "hvae", "gan"], help="Name recorded in output rows.")
    parser.add_argument("--dataset-name", required=True, help="Name recorded in output rows, e.g. morphomnist.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--metrics", nargs="+", default=["cdm", "uc", "cg"], choices=["cdm", "uc", "cg"])
    parser.add_argument("--cdm-options", nargs="+", default=["top", "layerwise", "full"], choices=["top", "layerwise", "full"])
    parser.add_argument("--cdm-effects", nargs="+", default=["direct", "total"], choices=["direct", "total"])
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--cg-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-dims-per-layer", type=int, default=None, help="Debug/smoke-test cap.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/raw")
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


def _metric_rows(model_name: str, dataset_name: str, result: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if result["metric"] == "cdm":
        prefix = f"cdm_{result['option']}_{result['effect']}"
        for key in ["interventional_robustness", "counterfactual_generativeness"]:
            yield {
                "model": model_name,
                "dataset": dataset_name,
                "metric_name": f"{prefix}_{key}",
                "metric_value": result[key]["mean"],
                "standard_error": result[key]["standard_error"],
            }
        for layer, layer_result in result["per_layer"].items():
            for key in ["interventional_robustness", "counterfactual_generativeness"]:
                yield {
                    "model": model_name,
                    "dataset": dataset_name,
                    "metric_name": f"{prefix}_layer_{layer}_{key}",
                    "metric_value": layer_result[key]["mean"],
                    "standard_error": layer_result[key]["standard_error"],
                }
    elif result["metric"] == "uc":
        yield {
            "model": model_name,
            "dataset": dataset_name,
            "metric_name": "uc_full",
            "metric_value": result["uc"]["uc"],
            "standard_error": 0.0,
        }
    elif result["metric"] == "cg":
        yield {
            "model": model_name,
            "dataset": dataset_name,
            "metric_name": "cg",
            "metric_value": result["score"],
            "standard_error": result["standard_error"],
        }
        for factor, values in result["per_factor"].items():
            yield {
                "model": model_name,
                "dataset": dataset_name,
                "metric_name": f"cg_{factor}",
                "metric_value": values["mean"],
                "standard_error": values["standard_error"],
            }


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    scm, config = load_scm(args.config, device=args.device, temperature=args.temperature)
    dataset = build_dataset(config, split=args.split)
    predictors = load_predictors(config, args.classifier_config, device=args.device)
    factors = list(config["attribute_size"].keys())

    results: list[Dict[str, Any]] = []
    factor_mapping = None

    if "cdm" in args.metrics:
        for option in args.cdm_options:
            for effect in args.cdm_effects:
                results.append(
                    compute_cdm(
                        model=scm,
                        dataset=dataset,
                        causal_graph=config["causal_graph"],
                        predictors=predictors,
                        dataset_name=config["dataset"],
                        option=option,
                        effect=effect,
                        num_samples=args.num_samples,
                        batch_size=args.batch_size,
                        max_dims_per_layer=args.max_dims_per_layer,
                        device=args.device,
                    )
                )

    if "uc" in args.metrics or "cg" in args.metrics:
        factor_mapping = discover_latent_factor_mapping(
            model=scm,
            dataset=dataset,
            factors=factors,
            causal_graph=config["causal_graph"],
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=args.device,
        )

    if "uc" in args.metrics:
        results.append({"metric": "uc", "mapping": {k: sorted(v) for k, v in factor_mapping.items()}, "uc": compute_uc(factor_mapping, factors)})

    if "cg" in args.metrics:
        results.append(
            compute_cg(
                model=scm,
                dataset=dataset,
                factors=factors,
                factor_latent_mapping=factor_mapping,
                predictors=predictors,
                dataset_name=config["dataset"],
                causal_graph=config["causal_graph"],
                num_samples=args.cg_samples,
                batch_size=args.batch_size,
                device=args.device,
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.dataset_name}_{args.model_name}_{stamp}"
    raw_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    with open(raw_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": args.model_name,
                "dataset": args.dataset_name,
                "config": args.config,
                "classifier_config": args.classifier_config,
                "results": results,
            },
            handle,
            indent=2,
        )

    rows = [row for result in results for row in _metric_rows(args.model_name, args.dataset_name, result)]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "dataset", "metric_name", "metric_value", "standard_error"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {raw_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
