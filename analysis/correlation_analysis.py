"""Correlation and linear-model analysis for benchmark and disentanglement metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


DISENTANGLEMENT_PREFIXES = ("cdm", "uc", "cg")
BENCHMARK_PREFIXES = ("composition", "effectiveness", "fid", "minimality", "cld", "l1", "lpips", "mae", "f1", "accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", nargs="+", required=True, help="Tidy CSV files from benchmark and CDM runs.")
    parser.add_argument("--output-dir", default="results/tables")
    return parser.parse_args()


def load_metrics(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def wide_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index=["model", "dataset"], columns="metric_name", values="metric_value", aggfunc="mean").reset_index()


def is_disentanglement(metric: str) -> bool:
    return metric.startswith(DISENTANGLEMENT_PREFIXES)


def is_benchmark(metric: str) -> bool:
    return metric.startswith(BENCHMARK_PREFIXES) and not is_disentanglement(metric)


def correlations(wide: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in wide.columns if col not in {"model", "dataset"}]
    disent_cols = [col for col in metric_cols if is_disentanglement(col)]
    bench_cols = [col for col in metric_cols if is_benchmark(col)]
    rows = []
    for disent in disent_cols:
        for bench in bench_cols:
            pair = wide[[disent, bench]].dropna()
            if len(pair) < 3:
                continue
            pearson = pearsonr(pair[disent], pair[bench])
            spearman = spearmanr(pair[disent], pair[bench])
            rows.append(
                {
                    "disentanglement_metric": disent,
                    "benchmark_metric": bench,
                    "n": len(pair),
                    "pearson_r": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                    "spearman_r": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                }
            )
    return pd.DataFrame(rows)


def regression_added_r2(wide: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in wide.columns if col not in {"model", "dataset"}]
    disent_cols = [col for col in metric_cols if is_disentanglement(col)]
    bench_cols = [col for col in metric_cols if is_benchmark(col)]
    rows = []
    if not disent_cols:
        return pd.DataFrame(rows)
    base = pd.get_dummies(wide[["model", "dataset"]], drop_first=True)
    for target in bench_cols:
        data = pd.concat([wide[[target] + disent_cols], base], axis=1).dropna()
        if len(data) < 4:
            continue
        y = data[target].to_numpy()
        x_base = data[base.columns].to_numpy() if len(base.columns) else np.ones((len(data), 1))
        x_full = data[list(base.columns) + disent_cols].to_numpy()
        base_model = LinearRegression().fit(x_base, y)
        full_model = LinearRegression().fit(x_full, y)
        r2_base = r2_score(y, base_model.predict(x_base))
        r2_full = r2_score(y, full_model.predict(x_full))
        rows.append(
            {
                "benchmark_metric": target,
                "n": len(data),
                "r2_without_disentanglement": r2_base,
                "r2_with_disentanglement": r2_full,
                "delta_r2": r2_full - r2_base,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(args.metrics_csv)
    wide = wide_table(df)
    wide.to_csv(output_dir / "summary_wide_metrics.csv", index=False)
    correlations(wide).to_csv(output_dir / "correlations.csv", index=False)
    regression_added_r2(wide).to_csv(output_dir / "regression_added_r2.csv", index=False)
    print(f"Wrote analysis tables to {output_dir}")


if __name__ == "__main__":
    main()
