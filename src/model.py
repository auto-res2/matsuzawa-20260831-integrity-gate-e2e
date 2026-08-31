"""Two tiny classifiers with a shared train/predict interface.

Plain NumPy full-batch gradient descent — the experiment is a pipeline
integration check, so the models are deliberately minimal and CPU-cheap
while still producing a real accuracy difference on the XOR data.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _bce(p: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


class LogisticRegression:
    """The linear baseline: cannot separate XOR blobs."""

    def __init__(self, seed: int, lr: float = 0.5):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(scale=0.1, size=2)
        self.b = 0.0
        self.lr = lr

    def step(self, x: np.ndarray, y: np.ndarray) -> float:
        p = _sigmoid(x @ self.w + self.b)
        grad = p - y
        self.w -= self.lr * (x.T @ grad) / len(x)
        self.b -= self.lr * float(np.mean(grad))
        return _bce(p, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (_sigmoid(x @ self.w + self.b) >= 0.5).astype(np.float64)


class TinyMLP:
    """The proposed model: one tanh hidden layer, enough for XOR."""

    def __init__(self, seed: int, hidden: int = 16, lr: float = 0.5):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(scale=0.5, size=(2, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(scale=0.5, size=hidden)
        self.b2 = 0.0
        self.lr = lr

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(x @ self.w1 + self.b1)
        return h, _sigmoid(h @ self.w2 + self.b2)

    def step(self, x: np.ndarray, y: np.ndarray) -> float:
        h, p = self._forward(x)
        n = len(x)
        d_out = (p - y) / n
        self.w2 -= self.lr * (h.T @ d_out)
        self.b2 -= self.lr * float(np.sum(d_out))
        d_h = np.outer(d_out, self.w2) * (1 - h**2)
        self.w1 -= self.lr * (x.T @ d_h)
        self.b1 -= self.lr * d_h.sum(axis=0)
        return _bce(p, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self._forward(x)[1] >= 0.5).astype(np.float64)


def build(model_name: str, seed: int):
    if model_name == "mlp":
        return TinyMLP(seed)
    if model_name == "logreg":
        return LogisticRegression(seed)
    raise ValueError(f"unknown model: {model_name}")
