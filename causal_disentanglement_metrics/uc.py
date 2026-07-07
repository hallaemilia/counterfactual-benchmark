"""Unconfoundedness (UC) metric from Reddy et al. (2022)."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, Mapping, Sequence, Set

import numpy as np


def compute_uc(factor_latent_mapping: Mapping[str, Iterable[str]], factors: Sequence[str]) -> Dict[str, float]:
    """Return UC in [0, 1], where 1 means no factor-pair overlap."""

    if len(factors) < 2:
        return {"uc": 1.0, "mean_jaccard": 0.0}

    jaccards = []
    for first, second in combinations(factors, 2):
        a: Set[str] = set(factor_latent_mapping.get(first, set()))
        b: Set[str] = set(factor_latent_mapping.get(second, set()))
        union = a | b
        jaccard = 0.0 if not union else len(a & b) / len(union)
        jaccards.append(jaccard)

    mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0
    return {
        "uc": float(1.0 - mean_jaccard),
        "mean_jaccard": mean_jaccard,
        "num_factor_pairs": len(jaccards),
    }
