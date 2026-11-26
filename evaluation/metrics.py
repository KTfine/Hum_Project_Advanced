from __future__ import annotations

from collections.abc import Callable

import numpy as np


def paired_bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    scorer: Callable[[np.ndarray], float] = np.mean,
    iterations: int = 5000,
    seed: int = 3800,
) -> dict:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or not len(first):
        raise ValueError("paired arrays must have the same non-empty 1-D shape")
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        indices = rng.integers(0, len(first), size=len(first))
        differences[iteration] = float(scorer(first[indices]) - scorer(second[indices]))
    point = float(scorer(first) - scorer(second))
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "difference": point,
        "ci95": [float(low), float(high)],
        "iterations": int(iterations),
        "seed": int(seed),
    }
