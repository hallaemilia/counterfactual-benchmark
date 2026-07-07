"""Prepare HVAE counterfactual fine-tuning ablation configs."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, help="Base HVAE JSON config.")
    parser.add_argument("--output-dir", default="configs/causal_disentanglement/ablation")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--checkpoint-root", default="${EDDIE_SCRATCH_ROOT}/checkpoints/ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.base_config, "r", encoding="utf-8") as handle:
        base = json.load(handle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for value in args.lambdas:
        cfg = deepcopy(base)
        image_params = cfg["mechanism_models"]["image"]["params"]
        image_params["cf_fine_tune"] = "True"
        image_params["evaluate_cf_model"] = "False"
        image_params["fixed_lagrange_lambda"] = value
        cfg["checkpoint_dir"] = f"{args.checkpoint_root}/lambda_{value:g}"
        out = output_dir / f"hvae_lambda_{value:g}.json"
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(cfg, handle, indent=2)
        print(out)


if __name__ == "__main__":
    main()
