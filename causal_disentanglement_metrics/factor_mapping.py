"""Discover latent-factor mappings for UC and CG."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence, Set

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .latent_intervention import encode_image, flatten_latents


def _choose_intervention_values(dataset, batch, factor: str, device: str) -> torch.Tensor:
    if hasattr(dataset, "possible_values") and factor in dataset.possible_values:
        possible = torch.as_tensor(dataset.possible_values[factor], device=device, dtype=batch[factor].dtype)
        current = batch[factor].to(device)
        if possible.ndim == 1:
            values = possible[torch.randint(0, possible.shape[0], (current.shape[0],), device=device)].view_as(current)
            same = values == current
            if same.any() and possible.shape[0] > 1:
                values[same] = possible[(torch.searchsorted(possible, current[same].flatten()) + 1) % possible.shape[0]].view_as(values[same])
            return values
        indices = torch.randint(0, possible.shape[0], (current.shape[0],), device=device)
        return possible[indices].to(dtype=current.dtype)
    perm = torch.randperm(batch[factor].shape[0], device=device)
    return batch[factor].to(device)[perm]


@torch.no_grad()
def discover_latent_factor_mapping(
    model: Any,
    dataset,
    factors: Sequence[str],
    causal_graph: Optional[Mapping[str, Sequence[str]]] = None,
    method: str = "irs",
    threshold: str = "percentile",
    top_percent: float = 10.0,
    fixed_threshold: Optional[float] = None,
    num_samples: int = 1000,
    batch_size: int = 128,
    device: str = "cuda",
) -> Dict[str, Set[str]]:
    """Discover factor-specific latent coordinates.

    The default IRS-style method estimates per-factor latent change under
    interventions and scores a coordinate by high change for the target factor and
    low average change under other-factor interventions.
    """

    if method not in {"irs", "gradient"}:
        raise ValueError("method must be 'irs' or 'gradient'.")
    if method == "gradient":
        raise NotImplementedError("gradient mapping requires differentiable factor inputs; use method='irs'.")

    per_factor_changes: dict[str, list[torch.Tensor]] = {factor: [] for factor in factors}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    seen = 0

    for batch in tqdm(loader, desc="discover latent-factor mapping"):
        factual_encoded, _ = encode_image(model, batch, causal_graph=causal_graph, device=device)
        factual_flat = flatten_latents(factual_encoded)
        device_batch = {key: value.to(device) for key, value in batch.items()}
        abducted = model.encode(**device_batch) if hasattr(model, "models") else None
        for factor in factors:
            cf_value = _choose_intervention_values(dataset, batch, factor, device)
            if abducted is not None:
                cf_batch = model.decode({factor: cf_value}, **abducted)
            else:
                cf_batch = {key: value.clone() for key, value in batch.items()}
                cf_batch[factor] = cf_value.detach().cpu()
            cf_encoded, _ = encode_image(model, cf_batch, causal_graph=causal_graph, device=device)
            cf_flat = flatten_latents(cf_encoded)
            per_factor_changes[factor].append((cf_flat - factual_flat).abs().detach().cpu())
        seen += batch["image"].shape[0]
        if seen >= num_samples:
            break

    mean_changes = {factor: torch.cat(changes).mean(dim=0) for factor, changes in per_factor_changes.items()}
    mapping: Dict[str, Set[str]] = {}
    all_factors = list(factors)
    for factor in factors:
        own = mean_changes[factor]
        other = torch.stack([mean_changes[item] for item in all_factors if item != factor]).mean(dim=0) if len(factors) > 1 else torch.zeros_like(own)
        score = own - other
        if threshold == "percentile":
            cutoff = torch.quantile(score, 1.0 - top_percent / 100.0)
        elif threshold == "fixed":
            cutoff = torch.tensor(0.0 if fixed_threshold is None else fixed_threshold)
        else:
            raise ValueError("threshold must be 'percentile' or 'fixed'.")
        mapping[factor] = {f"flat:{idx}" for idx in torch.where(score >= cutoff)[0].tolist()}
    return mapping
