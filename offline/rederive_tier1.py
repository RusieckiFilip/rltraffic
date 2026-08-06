"""Re-issue P2.5's Tier 1 fixed-time numbers under prereg A1 (``att_horizon``).

P2.5 reported the k=3 / k=4 / shipped-plan figures and the grid4x4 inversion in *running mean*.
This script re-measures them as ``att_horizon`` (A1 primary) alongside ``att_running_mean``, with
the prereg §3.1 co-reported counts: ``completed`` (vehicles that finished) and ``entered``
(``= completed + still-in-network at the horizon``). It is measurement-only; it collects no corpus.

The k=4 ruling is **not** reopened -- it rests on the structural clearance-overhead criterion
(12.50% vs shipped 14.29% vs k=3 16.67%), which is aggregation-independent.

Settings match P2.5: cf_hz1x1 / cf_grid4x4, CityFlow, acyclic, delta_time=10, max_steps=360,
nominal demand (draw 0). Fixed-time cells use the equal-split cycle (hz1x1 greens are ascending, so
this equals the shipped green order -- verified: plan=None and plan=shipped give identical numbers);
the shipped-plan cell is the 1 s ``replay_plan``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
HZ1X1_CFG = REPO_ROOT / "configs" / "sim" / "cityflow1x1.json"
GRID4X4_CFG = REPO_ROOT / "configs" / "sim" / "cityflow_grid4x4.json"

__all__ = ["main"]

_METRICS = ["average_travel_time", "count_of_vehicles_completing_journey"]


def _fixedtime_row(
    cfg_path: Path, env_id: str, k: int, *, seed: int = 101,
    max_steps: int = 360, delta_time: int = 10,
) -> dict[str, Any]:
    """FixedTimeController (equal split) held k decision steps, measured under both aggregations."""
    from experiments.config import EnvSpec, SETTING_DEFAULTS
    from experiments.envs import make_env

    from offline.horizon_metric import horizon_rollout
    from offline.policies.fixed_time import FixedTimeController

    settings = dict(SETTING_DEFAULTS)
    settings.update(
        max_steps=max_steps, delta_time=delta_time, control_mode="acyclic", metrics=_METRICS
    )
    spec = EnvSpec(
        id=env_id, backend="cityflow",
        paths={"config": str(cfg_path.resolve())}, settings=settings,
    )
    env = make_env(spec)
    try:
        ctrl = FixedTimeController(env, k=k, plan=None)
        roll = horizon_rollout(env, lambda _e, info: ctrl.act(info), episodes=1, seed=seed)
    finally:
        env.close()
    completed = int(round(roll.final_completed))
    vehicle_count = int(round(roll.final_vehicle_count))
    return {
        "env_id": env_id,
        "label": f"k={k}",
        "att_horizon": roll.att_horizon,
        "att_running_mean": roll.att_running_mean,
        "entered": completed + vehicle_count,
        "completed": completed,
        "vehicle_count": vehicle_count,
    }


def _shipped_row(
    cfg_path: Path, env_id: str, *, seed: int = 0, max_steps: int = 360, delta_time: int = 10,
) -> dict[str, Any]:
    """The actual shipped signal plan, measured by the 1 s ``replay_plan`` under both aggregations."""
    from offline.policies.fixed_time import _scenario_dir_from_config  # our own package (not frozen)
    from offline.policies.plan_replay import read_plan_phases, replay_plan

    plan_path = _scenario_dir_from_config(cfg_path) / "signal_plan_template.txt"
    _header, phases = read_plan_phases(plan_path)
    res = replay_plan(
        str(cfg_path), phases, delta_time=delta_time, max_steps=max_steps,
        metric_names=["average_travel_time"], seed=seed,
    )
    return {
        "env_id": env_id,
        "label": "shipped",
        "att_horizon": res.att_horizon,
        "att_running_mean": res.att_running_mean,
        "entered": res.entered,
        "completed": res.completed,
        "vehicle_count": res.vehicle_count,
    }


def _format_table(rows: list[dict[str, Any]]) -> str:
    head = (
        f"{'env':11s} {'variant':9s} {'att_horizon':>12s} {'att_running_mean':>17s} "
        f"{'entered':>8s} {'completed':>10s}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['env_id']:11s} {r['label']:9s} {r['att_horizon']:12.2f} "
            f"{r['att_running_mean']:17.2f} {r['entered']:8d} {r['completed']:10d}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the Tier 1 table under both aggregations. Returns a process exit code."""
    argparse.ArgumentParser(description="Re-issue P2.5 Tier 1 numbers under att_horizon.").parse_args(
        argv
    )
    rows = [
        _fixedtime_row(HZ1X1_CFG, "cf_hz1x1", 3),
        _shipped_row(HZ1X1_CFG, "cf_hz1x1"),
        _fixedtime_row(HZ1X1_CFG, "cf_hz1x1", 4),
        _fixedtime_row(GRID4X4_CFG, "cf_grid4x4", 4),
    ]
    print(_format_table(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
