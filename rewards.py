from __future__ import annotations

from collections.abc import Callable
from typing import Any

RewardFn = Callable[[dict[str, Any]], float]

METRIC_HALTING = "number_of_all_halting_vehicles_for_the_last_time_step_in_simulation"
METRIC_AVG_TRAVEL = "average_travel_time"
METRIC_THROUGHPUT_DELTA = "throughput_delta"
METRIC_AVG_PRESSURE = "average_intersection_pressure"
METRIC_WAIT_ALL = "waiting_time_all_vehicles_for_the_last_time_step_in_simulation"
METRIC_STICKY_WAIT = "resco_sticky_waiting_time_on_the_incoming_lanes"

# RESCo normalises per-junction waiting time by 224 and clips to [-4, 4].
# Summed across 3 BB5B junctions that is a [-12, 12] range.
# We use the same divisor so reward magnitudes are directly comparable.
_WAIT_NORM_DIVISOR = 224.0
_WAIT_NORM_CLIP_LOCAL = 4.0
_WAIT_NORM_CLIP_GLOBAL = 12.0


_REWARD_REQUIRED_METRICS: dict[str, set[str]] = {
    "queue_length": {METRIC_HALTING},
    "average_travel_time": {METRIC_AVG_TRAVEL},
    "pressure": {METRIC_AVG_PRESSURE},
    "presslight": {METRIC_AVG_PRESSURE},
    "throughput": {METRIC_THROUGHPUT_DELTA},
    "combined": {METRIC_HALTING, METRIC_THROUGHPUT_DELTA, METRIC_AVG_TRAVEL},
    "wait_norm": {METRIC_WAIT_ALL},
    "sticky_wait_norm": {METRIC_STICKY_WAIT},
}


def required_metrics_for_reward(reward_fn: str | RewardFn) -> set[str]:
    if callable(reward_fn):
        return set()
    return set(_REWARD_REQUIRED_METRICS.get(reward_fn, set()))


def _metric_float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def reward_queue_length(metrics: dict[str, Any]) -> float:
    """Negative halting-vehicle count for the current step."""
    return -_metric_float(metrics, METRIC_HALTING)


def reward_average_travel_time(metrics: dict[str, Any]) -> float:
    """Negative average travel time (smaller is better)."""
    return -_metric_float(metrics, METRIC_AVG_TRAVEL)


def reward_pressure(metrics: dict[str, Any]) -> float:
    """Negative absolute average intersection pressure.

    Same objective as :func:`reward_presslight` (max-pressure theory
    minimises ``|incoming - outgoing|``); kept as a separate name for
    config compatibility.  Note it previously returned the raw signed
    pressure, which *rewarded* accumulating incoming queues.
    """
    return -abs(_metric_float(metrics, METRIC_AVG_PRESSURE))


def reward_presslight(metrics: dict[str, Any]) -> float:
    """PressLight reward: negative absolute intersection pressure.

    From Wei et al., *PressLight: Learning Max Pressure Control to Coordinate
    Traffic Signals in Arterial Network* (KDD 2019). Pressure is
    ``incoming - outgoing`` vehicle count; minimising ``|pressure|`` is proven
    (max-pressure theory) to maximise network throughput / minimise overall
    travel time. Designed as a *per-intersection* reward, so each agent gets
    local credit — the standard, well-validated choice for multi-intersection
    RL signal control. Fed the per-intersection metrics dict it scores that
    junction; fed the global dict it scores the network-average pressure.
    """
    return -abs(_metric_float(metrics, METRIC_AVG_PRESSURE))


def reward_throughput(metrics: dict[str, Any]) -> float:
    """Positive reward for vehicles leaving since last step."""
    return _metric_float(metrics, METRIC_THROUGHPUT_DELTA)


def reward_combined(metrics: dict[str, Any]) -> float:
    """Balanced objective over halting vehicles, throughput and travel time."""
    return (
        reward_queue_length(metrics)
        + 0.1 * reward_throughput(metrics)
        + 0.01 * reward_average_travel_time(metrics)
    )


def _wait_norm(metrics: dict[str, Any], clip: float) -> float:
    total_wait = _metric_float(metrics, METRIC_WAIT_ALL)
    raw = -total_wait / _WAIT_NORM_DIVISOR
    return float(max(-clip, min(clip, raw)))


def reward_wait_norm(metrics: dict[str, Any]) -> float:
    """Approximates RESCo's wait_norm globally:
    clip(-total_wait / 224, -12, 12).

    Uses the simulation-wide accumulated waiting time.  ReSCo clips per
    junction at [-4, 4] with divisor 224; summing across BB5B's 3 junctions
    gives the [-12, 12] range used here.  Note this is only an
    approximation: clipping the sum differs from summing per-junction
    clips whenever junctions are imbalanced — use the local variant for
    faithful RESCo semantics.
    """
    return _wait_norm(metrics, _WAIT_NORM_CLIP_GLOBAL)


def reward_wait_norm_local(metrics: dict[str, Any]) -> float:
    """RESCo's true per-junction wait_norm: clip(-wait / 224, -4, 4).

    Expects the *per-intersection* metrics dict, where
    ``METRIC_WAIT_ALL`` holds the accumulated waiting time of vehicles
    on that intersection's incoming lanes.
    """
    return _wait_norm(metrics, _WAIT_NORM_CLIP_LOCAL)


def reward_sticky_wait_norm(metrics: dict[str, Any]) -> float:
    """RESCo's exact per-junction wait_norm: clip(-sticky_wait / 224, -4, 4).

    ``wait_norm`` approximates the waiting time with SUMO's accumulated
    waiting; this variant instead uses ``METRIC_STICKY_WAIT``, which
    reproduces RESCo's ``Signal.observe`` bookkeeping (a vehicle counts
    from its first stop near the junction, keeps accruing while it is
    detectable — even when crawling — and is forgotten when it leaves).
    """
    total_wait = _metric_float(metrics, METRIC_STICKY_WAIT)
    raw = -total_wait / _WAIT_NORM_DIVISOR
    return float(max(-_WAIT_NORM_CLIP_LOCAL, min(_WAIT_NORM_CLIP_LOCAL, raw)))


_REWARD_REGISTRY: dict[str, RewardFn] = {
    "queue_length": reward_queue_length,
    "average_travel_time": reward_average_travel_time,
    "pressure": reward_pressure,
    "presslight": reward_presslight,
    "throughput": reward_throughput,
    "combined": reward_combined,
    "wait_norm": reward_wait_norm,
}

# Reward functions that can be applied *per intersection*: they receive
# that intersection's local metrics dict (see
# ``BaseMetrics.compute_all_local``) instead of the global one.  Only
# rewards whose required metrics have per-intersection implementations
# are listed; "wait_norm" swaps in RESCo's per-junction clipping.
_LOCAL_REWARD_REGISTRY: dict[str, RewardFn] = {
    "queue_length": reward_queue_length,
    "pressure": reward_pressure,
    "presslight": reward_presslight,
    "wait_norm": reward_wait_norm_local,
    "sticky_wait_norm": reward_sticky_wait_norm,
}

LOCAL_REWARD_NAMES = frozenset(_LOCAL_REWARD_REGISTRY.keys())


def resolve_reward_fn(reward_fn: str | RewardFn) -> RewardFn:
    if callable(reward_fn):
        return reward_fn

    try:
        base_fn = _REWARD_REGISTRY[reward_fn]
    except KeyError as exc:
        known = ", ".join(sorted(_REWARD_REGISTRY.keys()))
        raise ValueError(
            f"Unknown reward_fn '{reward_fn}'. Expected one of: {known}."
        ) from exc

    return base_fn


def resolve_local_reward_fn(reward_fn: str | RewardFn) -> RewardFn:
    """Resolve a per-intersection reward function.

    Custom callables receive one intersection's local metrics dict and
    must return a float.
    """
    if callable(reward_fn):
        return reward_fn

    try:
        return _LOCAL_REWARD_REGISTRY[reward_fn]
    except KeyError as exc:
        known = ", ".join(sorted(_LOCAL_REWARD_REGISTRY.keys()))
        raise ValueError(
            f"Unknown local_reward_fn '{reward_fn}'. "
            f"Expected one of: {known}."
        ) from exc
