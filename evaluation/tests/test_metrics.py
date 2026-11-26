import numpy as np

from evaluation.metrics import paired_bootstrap_difference


def test_paired_bootstrap_is_reproducible():
    left = np.array([1.0, 2.0, 3.0, 4.0])
    right = np.array([0.0, 1.0, 2.0, 3.0])
    first = paired_bootstrap_difference(left, right, iterations=200, seed=7)
    second = paired_bootstrap_difference(left, right, iterations=200, seed=7)
    assert first == second
    assert first["difference"] == 1.0
    assert first["ci95"] == [1.0, 1.0]
