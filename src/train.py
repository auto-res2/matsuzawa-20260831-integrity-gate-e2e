"""Single-run training executor: trains one run_id and writes its metrics."""

from __future__ import annotations

import json
from pathlib import Path

from src.model import build
from src.preprocess import make_dataset

SEED = 42

RUN_CONFIGS = {
    "proposed": {"model": "mlp"},
    "comparative-1": {"model": "logreg"},
}

MODE_SCALE = {
    "sanity": {"epochs": 8, "n_per_blob": 20},
    "pilot": {"epochs": 80, "n_per_blob": 100},
    "full": {"epochs": 400, "n_per_blob": 250},
}


def train_run(run_id: str, mode: str, results_dir: str) -> dict:
    if run_id not in RUN_CONFIGS:
        raise ValueError(f"unknown run_id: {run_id} (known: {sorted(RUN_CONFIGS)})")
    scale = MODE_SCALE[mode]
    x_train, y_train, x_test, y_test = make_dataset(scale["n_per_blob"], SEED)
    model = build(RUN_CONFIGS[run_id]["model"], SEED)

    losses = [model.step(x_train, y_train) for _ in range(scale["epochs"])]
    accuracy = float((model.predict(x_test) == y_test).mean())

    metrics = {
        "accuracy": accuracy,
        "loss": {"initial": losses[0], "final": losses[-1]},
        "epochs": scale["epochs"],
        "n_train": len(x_train),
        "n_test": len(x_test),
    }
    out_dir = Path(results_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[{run_id}] mode={mode} accuracy={accuracy:.3f} "
          f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    return metrics
