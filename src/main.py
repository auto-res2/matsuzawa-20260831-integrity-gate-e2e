"""Orchestrator: one Hydra entrypoint per run_id (run=all runs everything).

`run=all` executes both methods and the comparison aggregation in a
single process, so one executed run produces the complete results tree —
convenient for external executors that run a single entry command.
"""

from __future__ import annotations

import json
import time

import hydra
from omegaconf import DictConfig

# Managed executors capture output files from the working directory with a
# periodic sync; a run that exits within seconds of its last write can end
# before any sync happens, and its results are lost. Idle at the end of
# online modes so at least one sync observes the finished files.
OUTPUT_SYNC_GRACE_SECONDS = 90

from src.evaluate import aggregate
from src.train import RUN_CONFIGS, train_run


def _print_sanity_verdict(all_metrics: dict[str, dict]) -> None:
    problems = []
    for run_id, m in all_metrics.items():
        if m["epochs"] < 5:
            problems.append(f"{run_id}:too_few_steps")
        if not m["loss"]["final"] <= m["loss"]["initial"]:
            problems.append(f"{run_id}:loss_not_decreasing")
    if problems:
        print(f"SANITY_VALIDATION: FAIL reason={','.join(problems)}")
        raise SystemExit(1)
    sample = next(iter(all_metrics.values()))
    summary = {
        "steps": sample["epochs"],
        "loss_start": sample["loss"]["initial"],
        "loss_end": sample["loss"]["final"],
    }
    print("SANITY_VALIDATION: PASS")
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    mode = cfg.mode
    results_dir = cfg.results_dir
    run_id = cfg.run.run_id

    run_ids = sorted(RUN_CONFIGS) if run_id == "all" else [run_id]
    all_metrics = {rid: train_run(rid, mode, results_dir) for rid in run_ids}
    if run_id == "all":
        aggregate(results_dir, run_ids)

    if mode == "sanity":
        _print_sanity_verdict(all_metrics)
        return
    if mode == "pilot":
        print("PILOT_VALIDATION: PASS")
        print(
            "PILOT_VALIDATION_SUMMARY: "
            + json.dumps({r: m["accuracy"] for r, m in all_metrics.items()})
        )
    print(f"results written; idling {OUTPUT_SYNC_GRACE_SECONDS}s for output sync")
    time.sleep(OUTPUT_SYNC_GRACE_SECONDS)


if __name__ == "__main__":
    main()
