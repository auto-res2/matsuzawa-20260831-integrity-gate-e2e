"""Single-run training executor: trains one run_id and writes its metrics.

In full mode each method is trained over several seeds and the reported
accuracy is the mean, with the per-seed values and their standard
deviation recorded alongside — the honest way to report a stochastic
result, and it keeps every number traceable to the seeds that produced
it.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from src.model import build
from src.preprocess import make_dataset

BASE_SEED = 42

RUN_CONFIGS = {
    "proposed": {"model": "mlp"},
    "comparative-1": {"model": "logreg"},
}

MODE_SCALE = {
    "sanity": {"epochs": 8, "n_per_blob": 20, "seeds": 1},
    "pilot": {"epochs": 80, "n_per_blob": 100, "seeds": 2},
    "full": {"epochs": 400, "n_per_blob": 250, "seeds": 5},
}


def _train_once(run_id: str, scale: dict, seed: int) -> tuple[float, list[float]]:
    x_train, y_train, x_test, y_test = make_dataset(scale["n_per_blob"], seed)
    model = build(RUN_CONFIGS[run_id]["model"], seed)
    losses = [model.step(x_train, y_train) for _ in range(scale["epochs"])]
    accuracy = float((model.predict(x_test) == y_test).mean())
    return accuracy, losses


def train_run(run_id: str, mode: str, results_dir: str) -> dict:
    if run_id not in RUN_CONFIGS:
        raise ValueError(f"unknown run_id: {run_id} (known: {sorted(RUN_CONFIGS)})")
    scale = MODE_SCALE[mode]
    seeds = [BASE_SEED + i for i in range(scale["seeds"])]

    accuracies: list[float] = []
    first_losses: list[float] = []
    for seed in seeds:
        accuracy, losses = _train_once(run_id, scale, seed)
        accuracies.append(accuracy)
        if seed == seeds[0]:
            first_losses = losses

    metrics = {
        "accuracy": statistics.fmean(accuracies),
        "accuracy_std": statistics.stdev(accuracies) if len(seeds) > 1 else 0.0,
        "accuracy_per_seed": accuracies,
        "seeds": seeds,
        "loss": {"initial": first_losses[0], "final": first_losses[-1]},
        "epochs": scale["epochs"],
        "n_train": int(4 * scale["n_per_blob"] * 0.8),
        "n_test": int(4 * scale["n_per_blob"] * 0.2),
    }
    out_dir = Path(results_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"[{run_id}] mode={mode} accuracy={metrics['accuracy']:.3f}"
        f"±{metrics['accuracy_std']:.3f} over {len(seeds)} seed(s), "
        f"loss {first_losses[0]:.3f} -> {first_losses[-1]:.3f}"
    )
    return metrics
