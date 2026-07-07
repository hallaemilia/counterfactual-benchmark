"""Causally Disentangled Generation Metric (CDM).

The implementation adapts An et al. (2023) to the benchmark's image mechanisms.
For HVAE models, latent coordinates can be swept at the top layer, per layer, or
across the full hierarchy. Counterfactual generativeness assumes faithful
anti-causal predictors, as in the original CDM setup.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .latent_intervention import (
    LatentIndex,
    encode_image,
    flatten_latents,
    iter_layer_indices,
    latent_intervention,
    latent_layers,
    layer_numel,
)
from .utils import descendants, predictor_outputs, scalar_effect, standard_error


def _attribute_batch(batch: Mapping[str, torch.Tensor], attributes: Iterable[str], device: str) -> Dict[str, torch.Tensor]:
    return {attr: batch[attr].to(device) for attr in attributes}


@torch.no_grad()
def collect_latent_ranges(
    model: Any,
    dataset,
    causal_graph: Optional[Mapping[str, Sequence[str]]] = None,
    num_samples: int = 1000,
    batch_size: int = 128,
    device: str = "cuda",
) -> list[dict[str, torch.Tensor]]:
    """Collect dataset-wide min/max for each flattened latent layer."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    mins: list[torch.Tensor] = []
    maxs: list[torch.Tensor] = []
    seen = 0
    for batch in loader:
        encoded, _ = encode_image(model, batch, causal_graph=causal_graph, device=device)
        for layer_idx, layer in enumerate(latent_layers(encoded)):
            flat = layer.flatten(start_dim=1)
            while len(mins) <= layer_idx:
                mins.append(torch.full((flat.shape[1],), float("inf"), device=device))
                maxs.append(torch.full((flat.shape[1],), float("-inf"), device=device))
            mins[layer_idx] = torch.minimum(mins[layer_idx], flat.min(dim=0).values)
            maxs[layer_idx] = torch.maximum(maxs[layer_idx], flat.max(dim=0).values)
        seen += batch["image"].shape[0]
        if seen >= num_samples:
            break
    return [{"min": mn.detach(), "max": mx.detach()} for mn, mx in zip(mins, maxs)]


@torch.no_grad()
def compute_ace(
    model: Any,
    latent_dim_index: int,
    layer: int,
    dataset,
    predictors: Mapping[str, torch.nn.Module],
    causal_graph: Optional[Mapping[str, Sequence[str]]] = None,
    dataset_name: str = "morphomnist",
    num_samples: int = 1000,
    batch_size: int = 128,
    effect: str = "direct",
    latent_ranges: Optional[list[dict[str, torch.Tensor]]] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """Compute ACE for one latent coordinate against every predicted attribute.

    ACE(c, i) = E[u_c | do(z_i=max)] - E[u_c | do(z_i=min)].
    Vector-valued outputs, such as digit probabilities, are reduced with mean
    absolute change over the output vector.
    """

    attributes = list(predictors.keys())
    if latent_ranges is None:
        latent_ranges = collect_latent_ranges(
            model,
            dataset,
            causal_graph=causal_graph,
            num_samples=num_samples,
            batch_size=batch_size,
            device=device,
        )

    z_min = latent_ranges[layer]["min"][latent_dim_index]
    z_max = latent_ranges[layer]["max"][latent_dim_index]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    diffs: dict[str, list[torch.Tensor]] = {attr: [] for attr in attributes}
    seen = 0
    for batch in tqdm(loader, desc=f"ACE layer={layer} dim={latent_dim_index}", leave=False):
        encoded, parents = encode_image(model, batch, causal_graph=causal_graph, device=device)
        attrs = _attribute_batch(batch, attributes, device)

        low_state = latent_intervention(encoded, [LatentIndex(layer, latent_dim_index)], z_min, effect=effect)
        high_state = latent_intervention(encoded, [LatentIndex(layer, latent_dim_index)], z_max, effect=effect)

        img_model = model.models["image"] if hasattr(model, "models") and "image" in model.models else model
        low_images = img_model.decode(low_state, parents)
        high_images = img_model.decode(high_state, parents)

        low_scores = predictor_outputs(low_images, attrs, predictors, dataset_name)
        high_scores = predictor_outputs(high_images, attrs, predictors, dataset_name)
        for attr in attributes:
            diffs[attr].append(scalar_effect(high_scores[attr] - low_scores[attr]).detach().cpu())

        seen += batch["image"].shape[0]
        if seen >= num_samples:
            break

    return {attr: float(torch.cat(values).mean().item()) for attr, values in diffs.items()}


def _selected_layers(first_encoded: Any, option: str) -> list[int]:
    layers = latent_layers(first_encoded)
    if option == "top":
        return [0]
    if option == "layerwise" or option == "full":
        return list(range(len(layers)))
    raise ValueError("option must be one of: top, layerwise, full")


def _assign_latents_to_factors(ace_table: Mapping[tuple[int, int], Mapping[str, float]]) -> Dict[tuple[int, int], str]:
    assignments = {}
    for coord, effects in ace_table.items():
        assignments[coord] = max(effects.items(), key=lambda item: abs(item[1]))[0]
    return assignments


@torch.no_grad()
def compute_cdm(
    model: Any,
    dataset,
    causal_graph: Mapping[str, Sequence[str]],
    predictors: Mapping[str, torch.nn.Module],
    dataset_name: str,
    option: str = "top",
    effect: str = "direct",
    num_samples: int = 1000,
    batch_size: int = 128,
    max_dims_per_layer: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Compute CDM summaries and the latent-attribute ACE heatmap.

    ``option='top'`` sweeps only the first/top stochastic layer. ``layerwise``
    returns per-layer summaries. ``full`` flattens all layers into one summary.
    """

    first_batch = next(iter(DataLoader(dataset, batch_size=min(batch_size, 8), shuffle=False)))
    first_encoded, _ = encode_image(model, first_batch, causal_graph=causal_graph, device=device)
    selected_layers = _selected_layers(first_encoded, option)
    latent_ranges = collect_latent_ranges(model, dataset, causal_graph, num_samples, batch_size, device)

    ace_table: dict[tuple[int, int], dict[str, float]] = {}
    for layer_idx in selected_layers:
        n_dims = layer_numel(latent_layers(first_encoded)[layer_idx])
        if max_dims_per_layer is not None:
            n_dims = min(n_dims, max_dims_per_layer)
        for dim_idx in range(n_dims):
            ace_table[(layer_idx, dim_idx)] = compute_ace(
                model=model,
                latent_dim_index=dim_idx,
                layer=layer_idx,
                dataset=dataset,
                predictors=predictors,
                causal_graph=causal_graph,
                dataset_name=dataset_name,
                num_samples=num_samples,
                batch_size=batch_size,
                effect=effect,
                latent_ranges=latent_ranges,
                device=device,
            )

    assignments = _assign_latents_to_factors(ace_table)
    desc = {factor: descendants(causal_graph, factor) | {factor} for factor in causal_graph if factor != "image"}

    buckets: dict[str, list[float]] = {"interventional_robustness": [], "counterfactual_generativeness": []}
    layer_buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"interventional_robustness": [], "counterfactual_generativeness": []})

    for coord, attr_effects in ace_table.items():
        source_factor = assignments[coord]
        for attr, value in attr_effects.items():
            if attr in desc.get(source_factor, set()):
                buckets["counterfactual_generativeness"].append(value)
                layer_buckets[coord[0]]["counterfactual_generativeness"].append(value)
            else:
                buckets["interventional_robustness"].append(value)
                layer_buckets[coord[0]]["interventional_robustness"].append(value)

    def summarize(values: Sequence[float], invert: bool = False) -> dict[str, float]:
        if not values:
            return {"mean": float("nan"), "standard_error": float("nan")}
        arr = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        if invert:
            arr = 1.0 - arr
        return {"mean": float(arr.mean()), "standard_error": standard_error(arr)}

    result = {
        "metric": "cdm",
        "option": option,
        "effect": effect,
        "faithfulness_assumption": (
            "Counterfactual generativeness is interpreted under the assumption "
            "that anti-causal predictors faithfully estimate the annotated factors."
        ),
        "interventional_robustness": summarize(buckets["interventional_robustness"], invert=True),
        "counterfactual_generativeness": summarize(buckets["counterfactual_generativeness"], invert=False),
        "latent_factor_assignment": {f"{layer}:{dim}": factor for (layer, dim), factor in assignments.items()},
        "ace": {f"{layer}:{dim}": effects for (layer, dim), effects in ace_table.items()},
        "per_layer": {},
    }
    for layer_idx, values in layer_buckets.items():
        result["per_layer"][str(layer_idx)] = {
            "interventional_robustness": summarize(values["interventional_robustness"], invert=True),
            "counterfactual_generativeness": summarize(values["counterfactual_generativeness"], invert=False),
        }
    return result
