"""Counterfactual Generativeness (CG) metric from Reddy et al. (2022)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Set

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .latent_intervention import LatentIndex, encode_image, flatten_latents, latent_intervention, latent_layers, layer_numel
from .utils import predictor_outputs, scalar_effect, standard_error


def _flat_to_layer_indices(encoded_state: Any, flat_indices: Set[int]) -> list[LatentIndex]:
    result: list[LatentIndex] = []
    offset = 0
    for layer_idx, layer in enumerate(latent_layers(encoded_state)):
        width = layer_numel(layer)
        for flat_idx in sorted(flat_indices):
            if offset <= flat_idx < offset + width:
                result.append(LatentIndex(layer_idx, flat_idx - offset))
        offset += width
    return result


def _mapping_indices(mapping: Mapping[str, set[str]], factor: str, total_dim: int) -> tuple[set[int], set[int]]:
    own = {int(item.split(":", 1)[1]) for item in mapping.get(factor, set()) if item.startswith("flat:")}
    own = {idx for idx in own if 0 <= idx < total_dim}
    return own, set(range(total_dim)) - own


@torch.no_grad()
def compute_latent_baseline(
    model: Any,
    dataset,
    causal_graph: Optional[Mapping[str, Sequence[str]]] = None,
    num_samples: int = 1000,
    batch_size: int = 128,
    device: str = "cuda",
) -> torch.Tensor:
    values = []
    seen = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2):
        encoded, _ = encode_image(model, batch, causal_graph=causal_graph, device=device)
        values.append(flatten_latents(encoded).detach().cpu())
        seen += batch["image"].shape[0]
        if seen >= num_samples:
            break
    return torch.cat(values).mean(dim=0).to(device)


@torch.no_grad()
def compute_cg(
    model: Any,
    dataset,
    factors: Sequence[str],
    factor_latent_mapping: Mapping[str, set[str]],
    predictors: Mapping[str, torch.nn.Module],
    dataset_name: str,
    causal_graph: Optional[Mapping[str, Sequence[str]]] = None,
    num_samples: int = 500,
    batch_size: int = 64,
    baseline: str = "mean",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Compute CG from direct latent interventions.

    For each factor, the score compares changes from intervening on Z_I against
    changes from intervening on Z_not_I. Larger values indicate stronger
    factor-specific counterfactual generativeness.
    """

    if baseline != "mean":
        raise NotImplementedError("Only baseline='mean' is implemented.")
    latent_mean = compute_latent_baseline(model, dataset, causal_graph, num_samples, batch_size, device)

    factor_scores: dict[str, list[float]] = {factor: [] for factor in factors}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    seen = 0
    for batch in tqdm(loader, desc="CG"):
        encoded, parents = encode_image(model, batch, causal_graph=causal_graph, device=device)
        flat = flatten_latents(encoded)
        total_dim = flat.shape[1]
        img_model = model.models["image"] if hasattr(model, "models") and "image" in model.models else model
        attrs = {factor: batch[factor].to(device) for factor in predictors.keys()}
        factual_images = batch["image"].to(device)
        factual_scores = predictor_outputs(factual_images, attrs, predictors, dataset_name)

        for factor in factors:
            own, not_own = _mapping_indices(factor_latent_mapping, factor, total_dim)
            if not own:
                factor_scores[factor].append(0.0)
                continue

            own_indices = _flat_to_layer_indices(encoded, own)
            not_indices = _flat_to_layer_indices(encoded, not_own)
            own_values = latent_mean[sorted(own)] if own else torch.tensor([], device=device)
            not_values = latent_mean[sorted(not_own)] if not_own else torch.tensor([], device=device)

            own_state = latent_intervention(encoded, own_indices, own_values, effect="direct")
            own_images = img_model.decode(own_state, parents)
            own_scores = predictor_outputs(own_images, attrs, predictors, dataset_name)
            ice_own = scalar_effect(own_scores[factor] - factual_scores[factor])

            if not_indices:
                not_state = latent_intervention(encoded, not_indices, not_values, effect="direct")
                not_images = img_model.decode(not_state, parents)
                not_scores = predictor_outputs(not_images, attrs, predictors, dataset_name)
                ice_not = scalar_effect(not_scores[factor] - factual_scores[factor])
            else:
                ice_not = torch.zeros_like(ice_own)

            factor_scores[factor].extend(torch.clamp((ice_own - ice_not).abs(), 0.0, 1.0).detach().cpu().tolist())

        seen += batch["image"].shape[0]
        if seen >= num_samples:
            break

    per_factor = {
        factor: {
            "mean": float(np.mean(values)) if values else float("nan"),
            "standard_error": standard_error(values),
        }
        for factor, values in factor_scores.items()
    }
    all_scores = [score for values in factor_scores.values() for score in values]
    return {
        "metric": "cg",
        "score": float(np.mean(all_scores)) if all_scores else float("nan"),
        "standard_error": standard_error(all_scores),
        "per_factor": per_factor,
    }
