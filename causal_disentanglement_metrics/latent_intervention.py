"""Latent-space access and interventions for VAE, HVAE, GAN and SCM wrappers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch


Batch = Mapping[str, torch.Tensor]
Latents = Any


@dataclass(frozen=True)
class LatentIndex:
    """A flattened latent coordinate inside one stochastic layer."""

    layer: int
    index: int


def image_model(model: Any) -> Any:
    """Return the image mechanism whether ``model`` is an SCM or the mechanism itself."""

    if hasattr(model, "models") and "image" in model.models:
        return model.models["image"]
    return model


def parent_names(model: Any, causal_graph: Optional[Mapping[str, Sequence[str]]] = None) -> List[str]:
    """Infer image parent order from the SCM graph or from an explicit graph."""

    graph = causal_graph or getattr(model, "graph_structure", None)
    if graph is not None and "image" in graph:
        return list(graph["image"])
    raise ValueError("causal_graph with an 'image' entry is required for direct image-model calls.")


def parents_from_batch(
    batch: Batch,
    model: Any,
    causal_graph: Optional[Mapping[str, Sequence[str]]] = None,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Concatenate image parent attributes in benchmark order."""

    names = parent_names(model, causal_graph)
    tensors = [batch[name] for name in names]
    if device is not None:
        tensors = [tensor.to(device) for tensor in tensors]
    return torch.cat(tensors, dim=1)


def _clone_latents(latents: Latents) -> Latents:
    if torch.is_tensor(latents):
        return latents.clone()
    if isinstance(latents, tuple):
        return tuple(_clone_latents(item) for item in latents)
    if isinstance(latents, list):
        return [_clone_latents(item) for item in latents]
    return deepcopy(latents)


def _image_latents(encoded: Any) -> Any:
    """Extract the latent container from model.encode output."""

    if isinstance(encoded, tuple):
        return encoded[0]
    return encoded


def encode_image(
    model: Any,
    batch: Batch,
    causal_graph: Optional[Mapping[str, Sequence[str]]] = None,
    device: torch.device | str = "cuda",
) -> Tuple[Any, torch.Tensor]:
    """Encode image latents and return ``(encoded_state, parents)``."""

    img_model = image_model(model)
    image = batch["image"].to(device)
    parents = parents_from_batch(batch, model, causal_graph, device=device)
    encoded = img_model.encode(image, parents)
    return encoded, parents


def decode_image(
    model: Any,
    encoded_state: Any,
    parents: torch.Tensor,
    causal_graph: Optional[Mapping[str, Sequence[str]]] = None,
    return_loc: bool = True,
) -> torch.Tensor:
    """Decode an encoded image state with possibly modified latent values."""

    img_model = image_model(model)

    if hasattr(img_model, "decode"):
        decoded = img_model.decode(encoded_state, parents)
    elif hasattr(img_model, "forward_latents"):
        latents = _image_latents(encoded_state)
        decoded = img_model.forward_latents(latents, parents, return_loc=return_loc)[0]
    else:
        raise TypeError(f"Unsupported model type for latent decoding: {type(img_model)!r}")

    return decoded


def latent_layers(encoded_state: Any) -> List[torch.Tensor]:
    """Return a list of latent tensors for flat, VAE, GAN, or HVAE encoded states."""

    latents = _image_latents(encoded_state)
    if torch.is_tensor(latents):
        return [latents]
    if isinstance(latents, tuple):
        return [item for item in latents if torch.is_tensor(item)]
    if isinstance(latents, list):
        return [item for item in latents if torch.is_tensor(item)]
    raise TypeError(f"Unsupported latent container: {type(latents)!r}")


def layer_numel(layer: torch.Tensor) -> int:
    """Number of flattened per-example coordinates in a latent layer."""

    return int(torch.tensor(layer.shape[1:]).prod().item())


def latent_shapes(encoded_state: Any) -> List[Tuple[int, ...]]:
    return [tuple(layer.shape[1:]) for layer in latent_layers(encoded_state)]


def flatten_latents(encoded_state: Any) -> torch.Tensor:
    """Flatten all latent layers to ``[batch, total_dim]``."""

    return torch.cat([layer.flatten(start_dim=1) for layer in latent_layers(encoded_state)], dim=1)


def _replace_latents(encoded_state: Any, new_latents: List[Optional[torch.Tensor]]) -> Any:
    if torch.is_tensor(_image_latents(encoded_state)):
        assert len(new_latents) == 1
        return new_latents[0]

    if isinstance(encoded_state, tuple):
        head = encoded_state[0]
        if isinstance(head, list):
            return (new_latents, *encoded_state[1:])
        if isinstance(head, tuple):
            return (tuple(new_latents), *encoded_state[1:])
    if isinstance(encoded_state, list):
        return new_latents
    raise TypeError(f"Unsupported encoded state for replacement: {type(encoded_state)!r}")


def latent_intervention(
    encoded_state: Any,
    latent_indices: Iterable[LatentIndex | Tuple[int, int]],
    values: torch.Tensor | float,
    effect: str = "direct",
) -> Any:
    """Set selected flattened latent coordinates.

    ``effect='direct'`` keeps all other HVAE layers fixed. ``effect='total'`` sets
    lower layers after the highest intervened layer to ``None`` so the HVAE decoder
    samples them from the conditional prior. This follows the project convention:
    direct effect tests only the edited coordinate, while total effect samples a
    new lower-level world consistent with the intervention.
    """

    if effect not in {"direct", "total"}:
        raise ValueError("effect must be 'direct' or 'total'.")

    cloned = _clone_latents(encoded_state)
    layers = latent_layers(cloned)
    index_objs = [
        item if isinstance(item, LatentIndex) else LatentIndex(layer=item[0], index=item[1])
        for item in latent_indices
    ]
    if not index_objs:
        return cloned

    value_tensor = values
    per_coordinate_values = torch.is_tensor(value_tensor) and value_tensor.numel() == len(index_objs)
    for value_pos, item in enumerate(index_objs):
        layer = layers[item.layer]
        flat = layer.flatten(start_dim=1)
        if per_coordinate_values:
            assigned = value_tensor[value_pos].to(flat.device, dtype=flat.dtype).repeat(flat.shape[0])
        elif not torch.is_tensor(value_tensor):
            assigned = torch.full((flat.shape[0],), float(value_tensor), device=flat.device, dtype=flat.dtype)
        else:
            assigned = value_tensor.to(flat.device, dtype=flat.dtype).reshape(-1)
            if assigned.numel() == 1:
                assigned = assigned.repeat(flat.shape[0])
        flat[:, item.index] = assigned
        layers[item.layer] = flat.view_as(layer)

    if effect == "total":
        max_layer = max(item.layer for item in index_objs)
        for layer_idx in range(max_layer + 1, len(layers)):
            layers[layer_idx] = None

    return _replace_latents(cloned, layers)


def iter_layer_indices(encoded_state: Any, layers: Optional[Sequence[int]] = None) -> Iterable[LatentIndex]:
    """Yield flattened coordinates for selected layers."""

    layer_tensors = latent_layers(encoded_state)
    selected = list(range(len(layer_tensors))) if layers is None else list(layers)
    for layer_idx in selected:
        for flat_idx in range(layer_numel(layer_tensors[layer_idx])):
            yield LatentIndex(layer=layer_idx, index=flat_idx)
