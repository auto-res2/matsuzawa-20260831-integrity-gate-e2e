"""Aggregate finished runs into comparison/aggregated_metrics.json.

Reads the per-run metrics.json files this repository's runs wrote (no
external services) and emits the comparison record the AIRAS paper
tooling addresses as 'comparison.…'.

CLI shape (per AGENTS.md):
    uv run python -u -m src.evaluate results_dir=... run_ids='["a", "b"]'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def aggregate(results_dir: str, run_ids: list[str]) -> dict:
    metrics = {}
    for run_id in run_ids:
        path = Path(results_dir) / run_id / "metrics.json"
        metrics[run_id] = json.loads(path.read_text())

    proposed = [r for r in run_ids if r.startswith("proposed")]
    baselines = [r for r in run_ids if r.startswith("comparative")]
    best_proposed = max(metrics[r]["accuracy"] for r in proposed)
    best_baseline = max(metrics[r]["accuracy"] for r in baselines)

    aggregated = {
        "primary_metric": "accuracy",
        "metrics": metrics,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": best_proposed - best_baseline,
    }
    out_dir = Path(results_dir) / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregated_metrics.json").write_text(
        json.dumps(aggregated, indent=2) + "\n"
    )
    print(
        f"[comparison] best_proposed={best_proposed:.3f} "
        f"best_baseline={best_baseline:.3f} gap={aggregated['gap']:.3f}"
    )
    return aggregated


def main() -> None:
    args = dict(arg.split("=", 1) for arg in sys.argv[1:])
    results_dir = args.get("results_dir", ".research/results")
    run_ids = json.loads(args.get("run_ids", '["proposed", "comparative-1"]'))
    aggregate(results_dir, run_ids)


if __name__ == "__main__":
    main()
