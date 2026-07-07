"""Visualization utilities for causal disentanglement metric outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-json", nargs="+", required=True, help="Raw JSON files from run_metrics.py.")
    parser.add_argument("--metrics-csv", nargs="+", required=True, help="Tidy metrics CSV files.")
    parser.add_argument("--output-dir", default="results/figures")
    return parser.parse_args()


def plot_cdm_heatmaps(raw_paths: list[str], output_dir: Path) -> None:
    for path in raw_paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for result in payload["results"]:
            if result.get("metric") != "cdm":
                continue
            ace = pd.DataFrame(result["ace"]).T
            if ace.empty:
                continue
            fig, ax = plt.subplots(figsize=(max(6, ace.shape[1] * 1.2), max(4, ace.shape[0] * 0.15)))
            im = ax.imshow(ace.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
            ax.set_yticks(range(len(ace.index)))
            ax.set_yticklabels(ace.index, fontsize=6)
            ax.set_xticks(range(len(ace.columns)))
            ax.set_xticklabels(ace.columns, rotation=45, ha="right")
            ax.set_title(f"{payload['dataset']} {payload['model']} CDM {result['option']} {result['effect']}")
            fig.colorbar(im, ax=ax, label="ACE")
            fig.tight_layout()
            fig.savefig(output_dir / f"cdm_heatmap_{payload['dataset']}_{payload['model']}_{result['option']}_{result['effect']}.png", dpi=200)
            plt.close(fig)


def plot_summary(metrics_paths: list[str], output_dir: Path) -> None:
    df = pd.concat([pd.read_csv(path) for path in metrics_paths], ignore_index=True)
    uc = df[df["metric_name"].str.startswith("uc")]
    if not uc.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = uc["model"] + "\n" + uc["dataset"]
        ax.bar(labels, uc["metric_value"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("UC")
        ax.set_title("UC across models and datasets")
        fig.tight_layout()
        fig.savefig(output_dir / "uc_bar_chart.png", dpi=200)
        plt.close(fig)

    cdm = df[df["metric_name"].str.contains("counterfactual_generativeness", regex=False)]
    eff = df[df["metric_name"].str.startswith("effectiveness")]
    if not cdm.empty and not eff.empty:
        merged = cdm.merge(eff, on=["model", "dataset"], suffixes=("_cdm", "_effectiveness"))
        if not merged.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(merged["metric_value_cdm"], merged["metric_value_effectiveness"])
            ax.set_xlabel("CDM counterfactual generativeness")
            ax.set_ylabel("Effectiveness")
            ax.set_title("CDM vs effectiveness")
            fig.tight_layout()
            fig.savefig(output_dir / "cdm_vs_effectiveness.png", dpi=200)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_cdm_heatmaps(args.raw_json, output_dir)
    plot_summary(args.metrics_csv, output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
