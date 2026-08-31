"""Synthetic XOR-blobs dataset.

Four Gaussian blobs at the corners of a square, labeled XOR-style, so the
classes are not linearly separable: a linear model tops out near chance
on the diagonal pairs while a small non-linear model separates them.
Deterministic given the seed.
"""

from __future__ import annotations

import numpy as np

CENTERS = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
LABELS = np.array([0, 0, 1, 1])
NOISE = 0.45


def make_dataset(
    n_per_blob: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for center, label in zip(CENTERS, LABELS):
        xs.append(center + rng.normal(scale=NOISE, size=(n_per_blob, 2)))
        ys.append(np.full(n_per_blob, label))
    x = np.concatenate(xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    split = int(0.8 * len(x))
    return x[:split], y[:split], x[split:], y[split:]
