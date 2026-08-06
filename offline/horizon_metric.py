"""Read the registered primary metric at the episode horizon (prereg A1).

A1 (`v0.2-prereg-a1`) pins the paper's primary metric to the value of the registered,
survivorship-free metric ``average_travel_time`` (``metrics/cityflow.py``) **at the episode
horizon** -- the mean over all vehicles that entered the network during the episode -- and
explicitly *not* the mean of the per-step samples that ``experiments/runner.py`` reports.

``experiments/runner.py`` is frozen and may keep computing the running mean. This module is the
horizon reader we use for everything the paper reports. It reads the metric through the *same*
path the env uses -- every ``info`` dict already carries ``info["average_travel_time"] =
metrics.get("average_travel_time")`` (``envs/cityflow_env.py``) -- so the two quantities below are
the identical metric, one aggregation apart:

* ``att_horizon``       -- the metric's value at the final decision step (A1's primary metric).
* ``att_running_mean``  -- the mean of the per-step samples (the legacy ``runner.py`` quantity).

**Naming ruling (brief P8.0 / A1):** no episode-level field, variable or dict key here is named
bare ``average_travel_time``. That bare name appears only as the registry-metric string when the
env's ``info`` is *read* (``info["average_travel_time"]``).

Format / aggregation convention
-------------------------------
* Version: horizon-reader v1.0.
* Per episode: ``samples`` are the per-decision-step ``info["average_travel_time"]`` values, one
  per env ``step`` (``T`` of them for a ``T``-step episode); ``horizon = samples[-1]`` (the value
  at the horizon), ``running_mean = mean(samples)``.
* Over episodes: both are averaged over the ``episodes`` rollouts, mirroring
  ``experiments.runner.evaluate_policy`` exactly so ``att_running_mean`` is byte-identical to the
  frozen runner's ``average_travel_time`` for the same env/seed/policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

__all__ = ["HorizonRollout", "horizon_rollout"]

ChooseAction = Callable[[Any, dict[str, Any]], np.ndarray]


@dataclass(frozen=True)
class HorizonRollout:
    """Both aggregations of the primary metric, from one rollout of ``episodes`` episodes.

    ``att_horizon`` is the A1 primary metric (mean over episodes of the per-episode last sample);
    ``att_running_mean`` is the legacy ``runner.py`` quantity (mean over episodes of the per-episode
    sample mean). ``final_completed`` is ``count_of_vehicles_completing_journey`` at the horizon of
    the last episode when that metric is in the env's requested set, else ``nan`` -- Tier 1 derives
    ``entered = final_completed + final_vehicle_count`` from it.
    """

    att_horizon: float
    att_running_mean: float
    episode_reward: float
    final_vehicle_count: float
    final_completed: float
    per_episode_horizon: tuple[float, ...]
    per_episode_running_mean: tuple[float, ...]
    episodes: int
    seed: int


def horizon_rollout(
    env: Any,
    choose_action: ChooseAction,
    episodes: int,
    seed: int,
) -> HorizonRollout:
    """Roll out *choose_action* for *episodes* and report both metric aggregations.

    Mirrors ``experiments.runner.evaluate_policy``'s loop line-for-line (same reset seeding
    ``seed + ep``, same ``max_steps`` bound, same break-on-terminate/truncate, same ``.get(..., 0.0)``
    defaults and same ``float(np.mean(...))`` reductions), additionally recording the per-step
    ``info["average_travel_time"]`` samples so both ``att_horizon`` and ``att_running_mean`` come
    from a single rollout. This byte-identity with the frozen runner is what makes ``att_running_mean``
    reproduce the committed anchors. The caller owns *env* and closes it.
    """
    max_steps = int(env.max_steps)
    rewards: list[float] = []
    per_episode_horizon: list[float] = []
    per_episode_running_mean: list[float] = []
    vehicles: list[float] = []
    final_completed = float("nan")

    for ep in range(int(episodes)):
        info = env.reset(seed=seed + ep)
        reward_sum = 0.0
        samples: list[float] = []
        last_vehicle_count = 0.0
        last_info: dict[str, Any] = info

        for _ in range(max_steps):
            action = choose_action(env, info)
            reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            samples.append(float(info.get("average_travel_time", 0.0)))
            last_vehicle_count = float(info.get("vehicle_count", 0.0))
            last_info = info
            if terminated or truncated:
                break

        rewards.append(reward_sum)
        # A1: the horizon value is the metric at the final sampled step; the running mean is the
        # legacy per-step average. Both are two reductions of the SAME `samples` list.
        per_episode_horizon.append(samples[-1] if samples else 0.0)
        per_episode_running_mean.append(float(np.mean(samples)) if samples else 0.0)
        vehicles.append(last_vehicle_count)
        # Co-reported throughput at the horizon, when the env's requested metric set carries it
        # (Tier 1 enables it; the anchor set does not, in which case this stays NaN and is unused).
        metrics = last_info.get("metrics", {}) or {}
        if "count_of_vehicles_completing_journey" in metrics:
            final_completed = float(metrics["count_of_vehicles_completing_journey"])
        else:
            final_completed = float("nan")

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return HorizonRollout(
        att_horizon=_mean(per_episode_horizon),
        att_running_mean=_mean(per_episode_running_mean),
        episode_reward=_mean(rewards),
        final_vehicle_count=_mean(vehicles),
        final_completed=final_completed,
        per_episode_horizon=tuple(per_episode_horizon),
        per_episode_running_mean=tuple(per_episode_running_mean),
        episodes=int(episodes),
        seed=int(seed),
    )
