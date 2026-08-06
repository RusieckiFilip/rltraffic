"""The horizon reader is exactly two aggregations of the env's own per-step metric (prereg A1).

Two load-bearing facts, both against a live CityFlow engine (self-skip when it is absent -- backend
tests are opt-in here):

1. ``att_horizon`` is the last per-step ``info["average_travel_time"]`` sample and
   ``att_running_mean`` is the mean of those samples -- the §3.1 double-computation rule: the reader
   is checked by recomputing both from the raw samples through an independent rollout.
2. ``att_running_mean`` is **byte-identical** to the frozen ``experiments.runner.evaluate_policy``'s
   ``average_travel_time`` for the same env/seed/policy. This is what proves the reader reproduces
   the anchor pipeline (same quantity, one aggregation apart), so the re-derived anchors are
   trustworthy by construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from offline.horizon_metric import horizon_rollout

ROOT = Path(__file__).resolve().parents[1]
CFG = str(ROOT / "configs" / "sim" / "cityflow1x1.json")
SEED = 101


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _cityflow_available(), reason="cityflow not installed")


def _make_env(max_steps: int = 40, delta_time: int = 10) -> Any:
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


def _max_pressure_chooser(env: Any):
    from experiments.runner import _baseline_chooser

    return _baseline_chooser("max_pressure", env, 0)


def _manual_samples(env: Any, chooser, seed: int) -> list[float]:
    """One episode; collect the per-step ``info["average_travel_time"]`` samples directly."""
    info = env.reset(seed=seed)
    samples: list[float] = []
    for _ in range(int(env.max_steps)):
        action = chooser(env, info)
        _, terminated, truncated, info = env.step(action)
        samples.append(float(info["average_travel_time"]))
        if terminated or truncated:
            break
    return samples


def test_horizon_and_running_mean_are_two_aggregations_of_the_same_samples() -> None:
    """Independent-route double-compute: horizon == last sample, running mean == mean(samples)."""
    env_a = _make_env()
    try:
        rollout = horizon_rollout(env_a, _max_pressure_chooser(env_a), episodes=1, seed=SEED)
    finally:
        env_a.close()

    env_b = _make_env()
    try:
        samples = _manual_samples(env_b, _max_pressure_chooser(env_b), SEED)
    finally:
        env_b.close()

    assert samples, "no samples collected -- rollout did not step"
    # Exact equality, not approx: MaxPressure is deterministic and CityFlow demand is fixed, so the
    # two rollouts are the identical state sequence; any difference is a real signal, not noise.
    assert rollout.per_episode_horizon[0] == samples[-1]
    assert rollout.per_episode_running_mean[0] == float(np.mean(samples))
    # Single episode: the over-episode aggregate equals the one episode's value.
    assert rollout.att_horizon == rollout.per_episode_horizon[0]
    assert rollout.att_running_mean == rollout.per_episode_running_mean[0]
    # A1's whole point: the two aggregations differ on a real policy.
    assert rollout.att_horizon != rollout.att_running_mean


def test_running_mean_is_byte_identical_to_frozen_evaluate_policy() -> None:
    """The reader reproduces the anchor pipeline's ``average_travel_time`` exactly."""
    from experiments.runner import evaluate_policy

    env_a = _make_env()
    try:
        rollout = horizon_rollout(env_a, _max_pressure_chooser(env_a), episodes=2, seed=SEED)
    finally:
        env_a.close()

    env_b = _make_env()
    try:
        legacy = evaluate_policy(env_b, _max_pressure_chooser(env_b), episodes=2, seed=SEED)
    finally:
        env_b.close()

    assert rollout.att_running_mean == legacy["average_travel_time"]
    assert rollout.episode_reward == legacy["episode_reward"]
    assert rollout.final_vehicle_count == legacy["final_vehicle_count"]
