"""Shared utilities for causal disentanglement metric scripts."""

from __future__ import annotations

import json
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "counterfactual_benchmark"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(PACKAGE_ROOT / "methods" / "deepscm") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "methods" / "deepscm"))

from datasets.adni.dataset import ADNI  # noqa: E402
from datasets.celeba.dataset import Celeba  # noqa: E402
from datasets.morphomnist.dataset import MorphoMNISTLike  # noqa: E402
from datasets.transforms import ReturnDictTransform, get_attribute_ids  # noqa: E402
from model import SCM  # noqa: E402
from models.classifiers.adni_classifier import ADNIClassifier  # noqa: E402
from models.classifiers.celeba_classifier import CelebaClassifier  # noqa: E402
from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier  # noqa: E402
from models.classifiers.classifier import Classifier  # noqa: E402


DATASETS = {
    "morphomnist": MorphoMNISTLike,
    "celeba": Celeba,
    "adni": ADNI,
}


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_scm(config_path: str | os.PathLike[str], device: str = "cuda", temperature: float = 0.1) -> tuple[SCM, Dict[str, Any]]:
    """Load the benchmark SCM exactly as ``methods/deepscm/evaluate.py`` does."""

    config = load_json(config_path)
    models = {}
    for variable in config["causal_graph"].keys():
        if variable not in config["mechanism_models"]:
            continue
        model_config = config["mechanism_models"][variable]
        module = import_module(model_config["module"])
        model_class = getattr(module, model_config["model_class"])
        model = model_class(params=model_config["params"], attr_size=config["attribute_size"])
        if "finetune" in model_config["params"] and model_config["params"]["finetune"] == 1:
            model.name += "_finetuned"
        models[variable] = model

    if device != "cuda" and not torch.cuda.is_available():
        # The original SCM loader hard-codes cuda. Loading on CPU is useful for
        # smoke tests, so we temporarily monkeypatch CUDA availability through map_location.
        pass

    scm = SCM(
        checkpoint_dir=config["checkpoint_dir"],
        graph_structure=config["causal_graph"],
        temperature=temperature,
        **models,
    )
    scm.to(device)
    scm.eval()
    return scm, config


def build_dataset(config: Mapping[str, Any], split: str):
    dataset_name = config["dataset"]
    data_class = DATASETS[dataset_name]
    return data_class(config["attribute_size"], split=split, transform=ReturnDictTransform(config["attribute_size"]))


def _classifier_for(dataset_name: str, attr: str, config: Mapping[str, Any], classifier_config: Mapping[str, Any]):
    attribute_size = config["attribute_size"]
    if dataset_name == "morphomnist":
        return Classifier(
            attr=attr,
            num_outputs=attribute_size[attr],
            context_dim=len(list(classifier_config["anticausal_graph"][attr])),
        )
    if dataset_name == "celeba":
        if sum(attribute_size.values()) == 4:
            return CelebaComplexClassifier(
                attr=attr,
                context_dim=len(list(classifier_config["anticausal_graph"][attr])),
                num_outputs=classifier_config[attr + "_num_out"],
                lr=classifier_config["lr"],
                version=classifier_config["version"],
            )
        return CelebaClassifier(attr=attr, num_outputs=classifier_config[attr + "_num_out"], lr=classifier_config["lr"])
    attribute_ids = get_attribute_ids(attribute_size)
    return ADNIClassifier(
        attr=attr,
        num_outputs=classifier_config["attribute_size"][attr],
        children=classifier_config["anticausal_graph"][attr],
        num_slices=classifier_config["attribute_size"]["slice"],
        attribute_ids=attribute_ids,
        arch=classifier_config["arch"],
    )


def load_predictors(
    config: Mapping[str, Any],
    classifier_config_path: str | os.PathLike[str],
    device: str = "cuda",
) -> Dict[str, nn.Module]:
    """Load anti-causal predictors used by the benchmark effectiveness metric."""

    classifier_config = load_json(classifier_config_path)
    dataset_name = config["dataset"]
    predictors = {
        attr: _classifier_for(dataset_name, attr, config, classifier_config)
        for attr in config["attribute_size"].keys()
    }
    ckpt_path = classifier_config["ckpt_path"]
    for key, predictor in predictors.items():
        file_name = next((file for file in os.listdir(ckpt_path) if file.startswith(key)), None)
        if file_name is None:
            raise FileNotFoundError(f"No classifier checkpoint starting with {key!r} in {ckpt_path}")
        state = torch.load(os.path.join(ckpt_path, file_name), map_location=torch.device(device))
        predictor.load_state_dict(state["state_dict"])
        predictor.to(device)
        predictor.eval()
    return predictors


@torch.no_grad()
def predictor_outputs(
    images: torch.Tensor,
    attributes: Mapping[str, torch.Tensor],
    predictors: Mapping[str, nn.Module],
    dataset_name: str,
) -> Dict[str, torch.Tensor]:
    """Return normalized predictor outputs for ACE/CG computations."""

    outputs: Dict[str, torch.Tensor] = {}
    if dataset_name == "celeba":
        batch = {"image": images, **attributes}
        for key, predictor in predictors.items():
            if getattr(predictor, "conditions", None) is not None and predictor.conditions[key] is not None:
                cond = torch.cat([batch[att] for att in predictor.conditions[key]], dim=1)
                logits = predictor(images, cond)
            else:
                logits = predictor(images)
            outputs[key] = torch.sigmoid(logits).detach()
        return outputs

    if dataset_name == "morphomnist":
        for key, predictor in predictors.items():
            if key == "thickness":
                pred = predictor(images, attributes["intensity"])
            else:
                pred = predictor(images)
            if pred.shape[-1] > 1:
                outputs[key] = torch.softmax(pred, dim=-1).detach()
            else:
                outputs[key] = torch.sigmoid(pred).detach()
        return outputs

    batch = {"image": images, **attributes}
    for key, predictor in predictors.items():
        if getattr(predictor, "cond_atts", None) is not None:
            cond = torch.cat([batch[att] for att in predictor.cond_atts], dim=1)
            logits = predictor(images, cond) if predictor.image_as_input else predictor(cond)
        else:
            logits = predictor(images)
        outputs[key] = torch.sigmoid(logits).detach() if logits.shape[-1] == 1 else torch.softmax(logits, dim=-1).detach()
    return outputs


def descendants(causal_graph: Mapping[str, Sequence[str]], node: str) -> set[str]:
    children = {name: [] for name in causal_graph}
    for child, parents in causal_graph.items():
        for parent in parents:
            children.setdefault(parent, []).append(child)
    seen: set[str] = set()
    stack = list(children.get(node, []))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(children.get(current, []))
    return seen


def scalar_effect(delta: torch.Tensor) -> torch.Tensor:
    """Convert vector-valued attribute changes to one scalar per example."""

    if delta.ndim == 1 or delta.shape[-1] == 1:
        return delta.reshape(delta.shape[0], -1).abs().mean(dim=1)
    return delta.abs().mean(dim=1)


def standard_error(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))
