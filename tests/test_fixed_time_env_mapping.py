"""Live-CityFlow tests for the fixed-time controller (P2.5, ladder Tier 1).

Two things can only be proven against a real engine, and both are load-bearing:

1. **The action -> file-phase mapping.** The controller emits *green action
   indices*; a silent off-by-one would mis-label every Tier 1 trajectory and the
   corpus would look valid. ``test_action_maps_to_file_phase_live`` steps a real
   env and reads back ``current_phase``; ``test_off_by_one_mapping_is_caught_live``
   proves the same check fails against a mapping mutated by one.
2. **The replay/env measurement-pipeline agreement** (brief §9, Ruling 1). The
   shipped-plan ground truth is measured by a raw-engine replay whose
   ``average_travel_time`` comes from a *different code path* than the env's. Before
   that number may be compared to the k=3 / k=4 numbers, the two paths must agree on
   a degenerate plan both can express (hold green 1 all episode).

These self-skip when CityFlow is unavailable (a reasoned skip: backend tests are
opt-in in this repo).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from offline.policies.fixed_time import FixedTimeController, green_action_phases
from offline.policies.plan_replay import replay_plan

ROOT = Path(__file__).resolve().parents[1]
CFG = str(ROOT / "configs" / "sim" / "cityflow1x1.json")
IX = "intersection_1_1"
SEED = 101


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _cityflow_available(), reason="cityflow not installed")


def _make_env(max_steps: int = 64, delta_time: int = 10) -> Any:
    from experiments.config import EnvSpec, SETTING_DEFAULTS
    from experiments.envs import make_env

    settings = dict(SETTING_DEFAULTS)
    settings.update(max_steps=max_steps, delta_time=delta_time, control_mode="acyclic")
    spec = EnvSpec(
        id="cf_hz1x1",
        backend="cityflow",
        paths={"config": str(Path(CFG).resolve())},
        settings=settings,
    )
    return make_env(spec)


def _roll_and_check(env: Any, actions: list[int], expected_phase_fn) -> None:
    """Step *actions* through *env*, asserting current_phase == expected_phase_fn(a)."""
    env.reset(seed=SEED)
    for t, a in enumerate(actions):
        _, _, _, info = env.step(np.array([a], dtype=np.int64))
        observed = int(info["intersections"][IX]["current_phase"])
        expected = int(expected_phase_fn(a))
        assert observed == expected, (
            f"mapping mismatch at step {t}: action {a} produced file phase "
            f"{observed}, expected {expected}"
        )


def test_action_maps_to_file_phase_live() -> None:
    """The true mapping (action a -> file phase a+1) holds in a real engine."""
    env = _make_env()
    try:
        greens = green_action_phases(env.intersections[0])
        assert greens == [1, 2, 3, 4, 5, 6, 7, 8]
        actions = [a for a in range(len(greens)) for _ in range(2)]
        _roll_and_check(env, actions, lambda a: greens[a])  # a -> a+1, passes
    finally:
        env.close()


def test_off_by_one_mapping_is_caught_live() -> None:
    """The same live check rejects a mapping mutated by one -- proof it can fail."""
    env = _make_env()
    try:
        greens = green_action_phases(env.intersections[0])
        actions = list(range(len(greens)))
        with pytest.raises(AssertionError, match="mapping mismatch"):
            _roll_and_check(env, actions, lambda a: greens[a] + 1)
    finally:
        env.close()


@pytest.mark.parametrize("k", [3, 4])
def test_controller_rollout_phases_match_its_schedule(k: int) -> None:
    """Every phase the full controller drives the env into is the one it intended."""
    env = _make_env(max_steps=8 * k + 4)
    try:
        greens = green_action_phases(env.intersections[0])
        ctrl = FixedTimeController(env, k=k, plan=None)
        info = env.reset(seed=SEED)
        for _ in range(env.max_steps):
            action = ctrl.act(info)
            intended_phase = greens[int(action[0])]
            _, _, _, info = env.step(action)
            assert int(info["intersections"][IX]["current_phase"]) == intended_phase
    finally:
        env.close()


def _env_hold_green1_att_both(max_steps: int, delta_time: int) -> tuple[float, float]:
    """``(att_horizon, att_running_mean)`` for 'hold green 1' the whole episode.

    ``att_horizon`` is the per-step ``average_travel_time`` at the horizon (prereg A1's primary
    metric); ``att_running_mean`` is the mean of the per-step samples (the legacy runner.py
    quantity). Both come from the same per-step samples the replay harness produces, so the two
    measurement pipelines are compared under both aggregations."""
    env = _make_env(max_steps=max_steps, delta_time=delta_time)
    try:
        env.reset(seed=SEED)
        zero = np.array([0], dtype=np.int64)
        samples: list[float] = []
        for _ in range(env.max_steps):
            _, _, _, info = env.step(zero)
            samples.append(float(info["average_travel_time"]))
        return samples[-1], sum(samples) / len(samples)
    finally:
        env.close()


def test_replay_pipeline_matches_env_on_degenerate_hold() -> None:
    """Brief §9 gate, flipped to ``att_horizon`` (P8.0): env (acyclic, action 0) and the replay
    harness must agree on BOTH ``att_horizon`` and ``att_running_mean`` for 'hold green 1 the whole
    episode'. If they diverge the replay ground truth is not comparable to k=3 / k=4 and must not be
    used to choose k."""
    max_steps, delta_time = 360, 10
    env_horizon, env_running_mean = _env_hold_green1_att_both(max_steps, delta_time)
    result = replay_plan(
        CFG,
        [1] * (max_steps * delta_time),
        delta_time=delta_time,
        max_steps=max_steps,
        metric_names=["average_travel_time"],
        seed=0,
    )
    # Exact equality, not approx: both sides are the same metric class over engines in identical
    # states, so a difference is a real signal, never float noise (same reasoning as the P0.6
    # anchor test).
    assert result.att_horizon == env_horizon, (
        f"att_horizon pipelines disagree: replay={result.att_horizon}, env={env_horizon}; "
        "the replay ground truth is not comparable to the k numbers"
    )
    assert result.att_running_mean == env_running_mean, (
        f"att_running_mean pipelines disagree: replay={result.att_running_mean}, "
        f"env={env_running_mean}"
    )


def test_collect_records_fixed_time_provenance(tmp_path: Path) -> None:
    """--policy fixedtime runs end to end and the manifest records k, source, sha256."""
    from offline import collect

    out = tmp_path / "ds"
    rc = collect.main(
        [
            "--backend", "cityflow",
            "--env-config", CFG,
            "--policy", "fixedtime",
            "--fixed-time-k", "4",
            "--episodes", "1",
            "--max-steps", "5",
            "--delta-time", "10",
            "--base-seed", "0",
            "--out-dir", str(out),
        ]
    )
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rm = manifest["run_metadata"]
    assert rm["behavior_policy"] == "fixedtime"
    assert rm["fixed_time_k"] == 4
    assert rm["fixed_time_schedule_source"] == "shipped_plan"
    assert isinstance(rm["fixed_time_plan_sha256"], str) and len(rm["fixed_time_plan_sha256"]) == 64
