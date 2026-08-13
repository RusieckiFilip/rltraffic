"""P4.3 -- probe-calibrated return prompting, ablated in-domain.

Artifact format version: ``p4.3-rtg/1.0``.  Nothing here trains anything: the return-to-go target
is an **inference-time** parameter, and P4's five committed checkpoints are used exactly as they
were reported.

WHAT IS DECLARED HERE, AND WHY THE DECLARATION IS THE POINT
-----------------------------------------------------------
``PREREGISTRATION.md`` **A8(a)** requires the calibration rule to be a declared function of the
in-domain probe's return distribution, written down **before any target-domain evaluation number
exists**, and it makes the sweep *the ablation around that rule* rather than the mechanism that
selects it.  ``docs/plans/p4.3.md`` is that declaration; this module is its executable form:

* :data:`DECLARED_GRID` -- the nine targets, frozen.  :func:`grid_targets` refuses any point that
  is not one of them, so a grid derived from results cannot be reported.
* :func:`rule_b_target` -- **the mechanism**: ``R_best_source x (S(R_target_probe) /
  S(R_source_probe))``.  In-domain the two probe statistics coincide, the ratio is exactly 1 and
  the rule returns the naive target identically, so **this task cannot validate it** -- it can
  assert the identity, measure the source-domain half of the ratio that P7 consumes, and say so.
* :func:`rule_a_target` -- a quantile of the probe's own returns, evaluated as a **declared
  alternative whose failure is predicted in advance**: it targets *probe-achievable* rather than
  *best-achievable* return, and MaxPressure is not the behaviour policy.

**Neither criterion in this module selects anything.**  ATT scores the points and never chooses
one (``PREREGISTRATION.md`` section 6.1, D5).  The in-support fraction was withdrawn as a selector
on 2026-08-13 (``BRIEF_15`` section 12.1) and is a **reliability diagnostic**: ``target = 0`` sits
outside the training support for an entire episode while being the best point measured at n = 1, so
a criterion preferring in-support targets would point the wrong way.  Conditioning a
return-conditioned model on an optimistic -- therefore extreme -- return is what the mechanism *is*.

THE TWO QUANTITIES, AND THE ALIGNMENT THEY REST ON
---------------------------------------------------
*Probe return* ``R_i`` is the **per-intersection episode return under the collection reward**::

    R_i = sum over t of info_t["intersections"][ix]["reward"]        t = 0 .. T-1, T = max_steps

which is the stream C6 stores as ``local_reward`` and the stream ``DTAgent.act`` advances the RTG
on.  C6's alignment convention governs it: the reward and the ``info`` returned by step ``t`` both
describe the state **after** step ``t``, so the reward carried by an ``info`` and the lane counts
carried by that same ``info`` are the same instant.  That makes the second, independent route exact
rather than approximate::

    info_t["intersections"][ix]["reward"] == -sum(info_t["lane_waiting_vehicle_count"][lid]
                                                  for lid in ix.incoming_lanes)

verified in source: ``metrics/base.py:329-345`` sums halting vehicles over the intersection's
incoming lanes, ``rewards.py:48`` negates it, and ``envs/base_traffic_env.py:204-219`` composes
``global_reward_weight * global + local_fn`` -- with the collection's registered weight of 0.0 the
global part vanishes (``PREREGISTRATION.md`` section 3.3, threat 5).  Both routes are integral, so
:func:`episode_return_two_routes` compares them with ``==`` and the campaign refuses to report a
probe whose two routes disagree.

*In-support fraction* is measured against the **training** RTG range read from each checkpoint's own
frozen statistics (``[-9991.0, -6.0]`` over 72,000 rows for P4's checkpoints)::

    rtg_t = target_rtg - sum(r_s for s < t)          (unscaled)
    in support  <=>  rtg_min <= rtg_t <= rtg_max     (closed at both ends)
    below       <=>  rtg_t < rtg_min                 (strict)
    above       <=>  rtg_t > rtg_max                 (strict)

so the three counts partition the decisions exactly.  The scaled input ``rtg_t / rtg_scale`` gives
the identical verdict because the divisor is positive; the range is stated unscaled because that is
the form the training statistics are recorded in.

WHY THE TARGET IS APPLIED AFTER ``load()``
-------------------------------------------
``DTAgent.load`` overwrites ``_target_rtg`` and ``_rtg_scale`` from the checkpoint payload
(``agent/DTAgent.py:803-804``) and ``from_checkpoint`` takes no override, so a target applied before
loading is silently discarded and every grid point would be the same point wearing different labels.
:func:`agent_with_target` is the single place that applies it, and
``tests/test_rtg_calibration.py`` verifies it **by effect** -- the value reaching the model, spied at
``forward`` -- because reading the attribute back would pass under exactly that defect.

NAMING (contract C9)
--------------------
``dt`` denotes the offline multi-agent Decision Transformer of P4 -- the same model P4's committed
``docs/data/p4_gate.json`` keys as ``madt``.  The alias is documented here because this module
introduces the neutral key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from offline.dt_gate import (
    HELD_OUT_DRAWS,
    TRAINING_SEEDS,
    EpisodeResult,
    _cell,
    env_settings_from_manifest,
    evaluate_arm,
    load_gate_checkpoint,
    mean_ci95,
    runtime_provenance,
    write_json_atomic,
)

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "DECLARED_GRID",
    "GateAResult",
    "GRID_POINT_KEYS",
    "LeafDiff",
    "NAIVE_TARGET",
    "PROBE_ARM",
    "PROBE_DRAW_START",
    "PROBE_K_VALUES",
    "PROBE_STATISTICS",
    "PointRun",
    "ProbeEpisode",
    "RULE_A_K",
    "RULE_A_POINT_KEY",
    "RULE_A_QUANTILE",
    "SECONDARY_QUANTILES",
    "SupportCounts",
    "agent_with_target",
    "assert_probe_draws_disjoint",
    "build_parser",
    "effect_size_sidecar",
    "episode_return_two_routes",
    "evaluate_point",
    "gate_a_result",
    "grid_targets",
    "in_support_counts",
    "leaf_diff",
    "main",
    "point_artifact",
    "probe_artifact",
    "probe_draw_ids",
    "probe_statistic",
    "report_artifact",
    "requested_runs",
    "rule_a_target",
    "rule_b_target",
    "run_probe",
    "source_domain_ratio",
    "training_rtg_range",
]

ARTIFACT_FORMAT_VERSION = "p4.3-rtg/1.0"

#: The declared grid, frozen in ``docs/plans/p4.3.md`` before any evaluation number existed.
#: Anchored on quantities that predate this task: the training RTG support ``[-9991, -6]``, the
#: naive target (P4's ``target_rtg``), and the two points P4's independent review published at
#: n = 1.  Dense near 0 because that is where the only prior evidence pointed, and extended past
#: the support because a target derived from a weaker probe policy can land there.
DECLARED_GRID: tuple[float, ...] = (
    0.0,
    -1000.0,
    -2000.0,
    -3000.0,
    -4000.0,
    -5762.0,
    -7500.0,
    -9991.0,
    -13000.0,
)

#: Stable keys for the grid points, in declaration order.  Positional by design: a key names a
#: DECLARED point, so a renumbering is visible in every artifact rather than silent.
GRID_POINT_KEYS: tuple[str, ...] = tuple(f"dt_g{i}" for i in range(len(DECLARED_GRID)))

#: The naive rule's target: the maximum episode return of the training split, which is what P4
#: reported.  Asserted equal to the checkpoints' own ``target_rtg`` before any point is evaluated.
NAIVE_TARGET = -5762.0

#: Rule A, the declared alternative (``BRIEF_15`` section 12.2 item 4): one evaluated point.
RULE_A_POINT_KEY = "dt_rule_a_k100"
RULE_A_QUANTILE = 1.0
RULE_A_K = 100

#: Reported as target values and located on the grid, never rolled out (``BRIEF_15`` section 12.6).
SECONDARY_QUANTILES: tuple[float, ...] = (0.5, 0.75, 0.9)

#: The probe: MaxPressure, on the unused band above the corpus's 1-200 and below the held-out
#: 1000-1099.  Nested, so ``k = 5``'s draws are a prefix of ``k = 20``'s.
PROBE_ARM = "probe_maxpressure"
PROBE_DRAW_START = 201
PROBE_K_VALUES: tuple[int, ...] = (5, 20, 100)

#: Both probe statistics are reported; P7 declares which one it uses before it runs.
PROBE_STATISTICS: tuple[str, ...] = ("max", "mean")


@dataclass(frozen=True)
class ProbeEpisode:
    """One MaxPressure probe episode: the rule's input, measured by two independent routes."""

    draw_id: int
    local_return: float
    local_return_from_lanes: float
    att_horizon: float
    horizon_vehicle_count: float
    decisions: int


@dataclass(frozen=True)
class SupportCounts:
    """How one episode's conditioning trajectory sat against the training RTG range.

    ``in_support + below + above == n`` exactly: the three counts partition the decisions.
    """

    n: int
    in_support: int
    below: int
    above: int
    rtg_first: float
    rtg_last: float
    rtg_min: float
    rtg_max: float

    @property
    def fraction(self) -> float:
        """The in-support fraction -- a reliability diagnostic, never a selection criterion."""
        return self.in_support / self.n


@dataclass(frozen=True)
class PointRun:
    """One evaluated target: its episodes, its diagnostics and the constants it ran under."""

    point_key: str
    target_rtg: float
    rtg_scale: float
    episodes: tuple[EpisodeResult, ...]
    support: tuple[SupportCounts, ...]
    support_index: tuple[tuple[int, int], ...]
    canary_rtg_series: tuple[float, ...]
    canary_cell: tuple[int, int]


@dataclass(frozen=True)
class GateAResult:
    """Whether the naive point reproduced P4's committed episode records exactly."""

    status: str
    n_compared: int
    n_mismatched: int
    mismatches: tuple[str, ...]
    reference_path: str
    reference_sha256: str
    reference_mean: float
    reproduced_mean: float


@dataclass(frozen=True)
class LeafDiff:
    """A recursive leaf comparison of two JSON artifacts."""

    removed: tuple[str, ...]
    added: tuple[str, ...]
    changed: tuple[str, ...]


# ----------------------------------------------------------------------
# The probe
# ----------------------------------------------------------------------


def probe_draw_ids(k: int) -> tuple[int, ...]:
    """The ``k`` probe draws, ``PROBE_DRAW_START .. PROBE_DRAW_START + k - 1``.

    Nested by construction, so the ``k`` ablation is a prefix ablation and the three budgets
    share their episodes instead of sampling three unrelated sets.
    """
    count = int(k)
    if count < 1:
        raise ValueError(f"a probe needs at least one episode, got k={k!r}")
    return tuple(range(PROBE_DRAW_START, PROBE_DRAW_START + count))


def assert_probe_draws_disjoint(
    draw_ids: Sequence[int],
    *,
    training_draw_ids: Sequence[int],
    held_out_draws: Sequence[int],
) -> None:
    """Refuse a probe that touches the corpus's training draws or the held-out pool.

    *training_draw_ids* is the set the corpus **actually used**, read from the training artifact,
    not the registered 1-999 pool: the pool would accept draw 200, which the corpus trained on.
    """
    requested = [int(d) for d in draw_ids]
    if not requested:
        raise ValueError("the probe draw set is empty; a calibration rule needs episodes")
    training_overlap = sorted(set(requested) & {int(d) for d in training_draw_ids})
    held_out_overlap = sorted(set(requested) & {int(d) for d in held_out_draws})
    if training_overlap or held_out_overlap:
        raise ValueError(
            f"the probe draw set is not disjoint: {len(training_overlap)} draw(s) are training "
            f"draws (first {training_overlap[:5]}) and {len(held_out_overlap)} are held-out "
            f"draws (first {held_out_overlap[:5]}). PREREGISTRATION.md section 5 keeps the three "
            "bands apart, and a probe drawn from either of them calibrates on data the model "
            "trained on or on the pool its result is measured over"
        )


def episode_return_two_routes(
    infos: Sequence[Mapping[str, Any]],
    *,
    ix_id: str,
    incoming_lanes: Sequence[str],
) -> tuple[float, float]:
    """The episode's per-intersection return by both routes: ``local_reward`` and lane waiting.

    *infos* are the **post-step** infos, one per decision (C6: outcomes are ``T`` rows, and the
    ``info`` returned by step ``t`` describes the state after step ``t``).  The reset info is not
    among them: no reward precedes the first action.

    Route 1 reads ``info["intersections"][ix]["reward"]`` -- the stream the corpus stores as
    ``local_reward`` and the stream the RTG advances on.  Route 2 recomputes the same quantity
    from ``info["lane_waiting_vehicle_count"]`` over the intersection's incoming lanes, which is
    what ``metrics/base.py:329-345`` sums and ``rewards.py:48`` negates.  They agree exactly on a
    real rollout; both are returned so the caller can *check* rather than assume.
    """
    if not infos:
        raise ValueError("an episode with no post-step infos has no return to compute")
    lanes = [str(lane) for lane in incoming_lanes]
    if not lanes:
        raise ValueError(
            f"intersection {ix_id!r} lists no incoming lanes, so the lane route cannot be "
            "computed and the two routes would agree vacuously"
        )
    rewards: list[float] = []
    waiting: list[float] = []
    for step, info in enumerate(infos):
        payload = info["intersections"][ix_id]
        if "reward" not in payload:
            raise KeyError(
                f"intersection {ix_id!r} carries no 'reward' at post-step info {step}; the env "
                "must run with a local_reward_fn (contract C2) or the probe measures nothing"
            )
        rewards.append(float(payload["reward"]))
        counts = info["lane_waiting_vehicle_count"]
        missing = [lane for lane in lanes if lane not in counts]
        if missing:
            raise KeyError(
                f"post-step info {step} carries no waiting count for lanes {missing[:3]}; the "
                "second return route cannot be computed from it"
            )
        waiting.append(math.fsum(float(counts[lane]) for lane in lanes))
    return math.fsum(rewards), -math.fsum(waiting)


def run_probe(
    *,
    draw_ids: Sequence[int],
    config_for_draw: Callable[[int], Path],
    env_settings: Mapping[str, Any],
    scenario_id: str,
    engine_seed: int,
) -> list[ProbeEpisode]:
    """Roll MaxPressure over *draw_ids*, one episode per draw, recording both return routes.

    The loop mirrors ``offline.horizon_metric.horizon_rollout`` line for line -- same reset
    seeding, same ``max_steps`` bound, same break on terminate/truncate, same ``samples[-1]``
    horizon reading -- because the probe's ``att_horizon`` is cross-checked against
    ``dt_gate.evaluate_arm`` on the same draws and the two must agree exactly.  It is a separate
    loop only because ``horizon_rollout`` does not expose per-intersection rewards, which is the
    quantity the rule is a function of.
    """
    from algorithms.max_pressure import MaxPressureAgent
    from experiments.config import EnvSpec
    from experiments.envs import make_env

    episodes: list[ProbeEpisode] = []
    for draw_id in draw_ids:
        config_path = Path(config_for_draw(int(draw_id)))
        if not config_path.is_file():
            raise FileNotFoundError(
                f"probe draw {draw_id} has no materialised sim config at {config_path}; run "
                "offline.materialise_draws for the probe band first"
            )
        env = make_env(
            EnvSpec(
                id=scenario_id,
                backend="cityflow",
                paths={"config": str(config_path)},
                settings=dict(env_settings),
            )
        )
        try:
            intersections = list(env.intersections)
            if len(intersections) != 1:
                raise ValueError(
                    f"the probe records one intersection's return and this scenario has "
                    f"{len(intersections)}; a multi-intersection probe needs a per-intersection "
                    "rule, which this task does not build"
                )
            ix_id = str(intersections[0].id)
            lanes = list(intersections[0].incoming_lanes)
            agent = MaxPressureAgent(env)

            info = env.reset(seed=int(engine_seed))
            post_step: list[Mapping[str, Any]] = []
            samples: list[float] = []
            last_vehicle_count = 0.0
            for _ in range(int(env.max_steps)):
                action = agent.act(info)
                _reward, terminated, truncated, info = env.step(action)
                post_step.append(info)
                samples.append(float(info.get("average_travel_time", 0.0)))
                last_vehicle_count = float(info.get("vehicle_count", 0.0))
                if terminated or truncated:
                    break
        finally:
            env.close()

        from_rewards, from_lanes = episode_return_two_routes(
            post_step, ix_id=ix_id, incoming_lanes=lanes
        )
        episodes.append(
            ProbeEpisode(
                draw_id=int(draw_id),
                local_return=from_rewards,
                local_return_from_lanes=from_lanes,
                att_horizon=samples[-1] if samples else 0.0,
                horizon_vehicle_count=last_vehicle_count,
                decisions=len(post_step),
            )
        )
    return episodes


# ----------------------------------------------------------------------
# The rules
# ----------------------------------------------------------------------


def probe_statistic(returns: Sequence[float], statistic: str) -> float:
    """``max`` or ``mean`` of the probe returns; an unknown statistic raises.

    Both are reported everywhere they appear: ``max`` is the functional the naive rule uses and
    is sample-size sensitive by construction, ``mean`` is stable in ``k`` and is what a small
    probe budget can estimate.  Neither is selected here -- P7 declares which it uses.
    """
    values = [float(v) for v in returns]
    if not values:
        raise ValueError("no probe episodes: a probe statistic needs at least one return")
    if statistic == "max":
        return max(values)
    if statistic == "mean":
        return math.fsum(values) / len(values)
    raise ValueError(
        f"unknown probe statistic {statistic!r}; the declared ones are {list(PROBE_STATISTICS)}"
    )


def rule_a_target(returns: Sequence[float], quantile: float) -> float:
    """RULE A: a quantile of the probe's own returns (numpy ``method="linear"``).

    **The declared alternative, predicted in advance to be poor** (``docs/plans/p4.3.md`` section
    2.4).  It applies the naive rule's functional to the wrong distribution: the naive target is
    the maximum of the *behaviour policy's* returns, and MaxPressure is not the behaviour policy,
    so this asks the model to achieve probe-level return.  Kept, evaluated and reported because a
    registered prediction that a plausible-looking rule fails is the evidence that the mechanism
    must be relative.
    """
    values = np.asarray([float(v) for v in returns], dtype=np.float64)
    if values.size == 0:
        raise ValueError("no probe episodes: rule A is a quantile of the probe's returns")
    q = float(quantile)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"a quantile must lie in [0, 1], got {quantile!r}")
    return float(np.quantile(values, q, method="linear"))


def rule_b_target(
    *,
    best_source_return: float,
    probe_source_stat: float,
    probe_target_stat: float,
) -> float:
    """RULE B, the mechanism: ``best_source x (probe_target / probe_source)``.

    **The association is load-bearing and is pinned by a test.**  Forming the ratio first makes
    the in-domain case an exact identity: when the two probe statistics are the same float the
    ratio is exactly ``1.0`` and the product returns ``best_source`` bit-for-bit.  The other
    association, ``probe_target x (best_source / probe_source)``, is algebraically equal and
    numerically is not -- it fails for **7.06 %** of random real-valued statistics (measured
    2026-08-13 over 200,000 draws), which would turn a structural identity into an
    approximately-true claim exactly where P7 needs it to be exact.
    """
    source = float(probe_source_stat)
    if source == 0.0:
        raise ValueError(
            "the source probe statistic is zero, so Rule B's ratio is undefined; a probe that "
            "scored a zero return has not measured the return scale of its domain"
        )
    ratio = float(probe_target_stat) / source
    return float(best_source_return) * ratio


def source_domain_ratio(
    returns: Sequence[float], *, best_source_return: float, statistic: str
) -> float:
    """``R_best_source / S(R_probe_source)`` -- the half of Rule B's ratio measurable in domain.

    The deliverable P7 consumes (``BRIEF_15`` section 12.2 item 2): how much better the best
    achievable return is than the probe's own, in the domain where both are known.  P7 multiplies
    it by the probe statistic it measures in the target domain.
    """
    denominator = probe_statistic(returns, statistic)
    if denominator == 0.0:
        raise ValueError(
            "the source probe statistic is zero, so the source-domain ratio is undefined"
        )
    return float(best_source_return) / denominator


# ----------------------------------------------------------------------
# Conditioning: the target, the support, the agent
# ----------------------------------------------------------------------


def training_rtg_range(checkpoint_path: str | Path) -> tuple[float, float]:
    """``(min, max)`` of the RTG values the checkpoint's frozen training statistics recorded.

    Read from the checkpoint rather than restated as a constant here: the support a point is
    measured against must be the support the *reported weights* were trained on, and a checkpoint
    from another tier or another scenario carries a different one.
    """
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    stats = payload.get("stats")
    if not stats or "rtg" not in stats:
        raise ValueError(
            f"{checkpoint_path}: the checkpoint carries no frozen training statistics, so the "
            "training RTG range it was conditioned within cannot be recovered; the in-support "
            "diagnostic would have to invent a range"
        )
    per_scenario = stats["rtg"]
    pairs = [
        (scenario, ix_id, summary)
        for scenario, per_ix in per_scenario.items()
        for ix_id, summary in per_ix.items()
    ]
    if len(pairs) != 1:
        raise ValueError(
            f"{checkpoint_path}: the statistics cover {len(pairs)} (scenario, intersection) "
            f"pairs {[(s, i) for s, i, _ in pairs]}; this task measures one support range and "
            "will not silently take the first"
        )
    summary = pairs[0][2]
    return float(summary["min"]), float(summary["max"])


def in_support_counts(
    rtg_values: Sequence[float], *, rtg_min: float, rtg_max: float
) -> SupportCounts:
    """Partition a conditioning trajectory into in-support / below / above.

    The interval is **closed at both ends** and ``below``/``above`` are strict, so the three
    counts partition the decisions exactly -- asserted here rather than assumed, because the
    training minimum is itself a declared grid target and an episode really does start on it.

    This is a **reliability diagnostic**.  It was withdrawn as a selection criterion on
    2026-08-13 (``BRIEF_15`` section 12.1): ``target = 0`` scores 0.000 here and is the best point
    measured at n = 1, so preferring in-support targets would point the wrong way.
    """
    values = [float(v) for v in rtg_values]
    if not values:
        raise ValueError("an empty conditioning trajectory has no in-support fraction")
    low = float(rtg_min)
    high = float(rtg_max)
    if low > high:
        raise ValueError(f"the training RTG range is inverted: min {low} > max {high}")
    below = sum(1 for v in values if v < low)
    above = sum(1 for v in values if v > high)
    inside = len(values) - below - above
    counts = SupportCounts(
        n=len(values),
        in_support=inside,
        below=below,
        above=above,
        rtg_first=values[0],
        rtg_last=values[-1],
        rtg_min=min(values),
        rtg_max=max(values),
    )
    if counts.in_support + counts.below + counts.above != counts.n:
        raise AssertionError("the in-support counts do not partition the decisions")
    return counts


def agent_with_target(
    gym_env: Any,
    checkpoint_path: str | Path,
    *,
    declared_gradient_steps: int,
    target_rtg: float,
    device: str | None = None,
) -> Any:
    """Load a checkpoint, then apply *target_rtg* -- in that order, which is the whole point.

    ``DTAgent.load`` overwrites ``_target_rtg`` and ``_rtg_scale`` from the payload
    (``agent/DTAgent.py:803-804``) and ``from_checkpoint`` accepts no override, so a target
    applied to the constructor is discarded and **every grid point would silently run P4's
    original target**.  The whole campaign would then be one point wearing nine labels, with
    entirely plausible numbers.

    ``load_gate_checkpoint`` is reused rather than bypassed, so the no-online-selection guard
    (the recorded step count must be the declared one) still applies to every point.
    ``rtg_scale`` is asserted unchanged against the payload: only the target varies in this task.
    """
    path = Path(checkpoint_path)
    agent = load_gate_checkpoint(gym_env, path, int(declared_gradient_steps), device=device)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_scale = float(payload["rtg_scale"])

    agent._target_rtg = float(target_rtg)
    agent.reset_context()

    if float(agent._rtg_scale) != expected_scale:
        raise ValueError(
            f"{path}: rtg_scale is {agent._rtg_scale} but the checkpoint records "
            f"{expected_scale}; this task varies the target only and the normalisation divisor "
            "must come from the trained model"
        )
    resting = set(agent.current_rtg().values())
    if resting != {float(target_rtg)}:
        raise ValueError(
            f"{path}: after applying the target the agent conditions on {sorted(resting)} rather "
            f"than {float(target_rtg)}; the override did not take effect"
        )
    return agent


# ----------------------------------------------------------------------
# The campaign
# ----------------------------------------------------------------------


def grid_targets() -> dict[str, float]:
    """The declared grid as ``{point_key: target}``, in declaration order."""
    return dict(zip(GRID_POINT_KEYS, DECLARED_GRID))


def requested_runs(
    point_key: str, seeds: Sequence[int], draw_ids: Sequence[int]
) -> list[tuple[str, int, int]]:
    """The run set this point OWES, enumerated from the declaration and never from the data.

    ``PROJECT_PLAN`` section 7: a completeness check whose expectation is derived from the
    episodes it checks is a tautology.  The inputs here are the declared seeds and the registered
    held-out pool, both of which exist before a single rollout.
    """
    return [
        (str(point_key), int(seed), int(draw))
        for seed in seeds
        for draw in draw_ids
    ]


def evaluate_point(
    *,
    point_key: str,
    target_rtg: float,
    checkpoints: Mapping[int, str],
    draw_ids: Sequence[int],
    config_for_draw: Callable[[int], Path],
    env_settings: Mapping[str, Any],
    scenario_id: str,
    engine_seed: int,
    declared_gradient_steps: int,
    device: str | None = None,
) -> PointRun:
    """Evaluate one target over every seed and draw, recording ATT and the RTG diagnostics.

    The rollouts go through ``dt_gate.evaluate_arm``, unchanged, so this point is measured by the
    same instrument that produced P4's gate and P4.4's and P4.5's baselines.  The RTG trajectory
    is captured from the agent itself: after ``act(info_t)`` the agent's ``current_rtg()`` is
    exactly ``target - sum(r_s for s <= t)``, which is the value that decision was conditioned on.

    ``evaluate_arm`` calls the action factory once per draw, in draw order, so one recorder per
    call maps to one draw -- and the mapping is asserted, not assumed: the recorder count must
    equal the draw count and every episode must carry a full complement of decisions.
    """
    expected_decisions = int(env_settings["max_steps"])
    episodes: list[EpisodeResult] = []
    support: list[SupportCounts] = []
    index: list[tuple[int, int]] = []
    canary_series: tuple[float, ...] = ()
    canary_cell = (int(sorted(checkpoints)[0]), int(draw_ids[0]))

    rtg_min, rtg_max = training_rtg_range(next(iter(sorted(checkpoints.items())))[1])

    for seed in sorted(checkpoints):
        path = checkpoints[seed]
        seed_range = training_rtg_range(path)
        if seed_range != (rtg_min, rtg_max):
            raise ValueError(
                f"checkpoint {path} was trained within RTG range {seed_range} while seed "
                f"{sorted(checkpoints)[0]}'s is {(rtg_min, rtg_max)}; one support range cannot "
                "describe both and the diagnostic would be measured against the wrong one"
            )
        recorders: list[list[float]] = []

        def factory(env: Any, _path: str = path, _rec: list[list[float]] = recorders) -> Any:
            agent = agent_with_target(
                env,
                _path,
                declared_gradient_steps=int(declared_gradient_steps),
                target_rtg=float(target_rtg),
                device=device,
            )
            ids = list(agent.intersection_ids)
            if len(ids) != 1:
                raise ValueError(
                    f"this task records one intersection's conditioning trajectory and the env "
                    f"has {len(ids)}; a multi-intersection diagnostic is P5's"
                )
            series: list[float] = []
            _rec.append(series)

            def choose(_env: Any, info: dict[str, Any]) -> np.ndarray:
                action = agent.act(info, explore=False, update_memory=True)
                series.append(float(agent.current_rtg()[ids[0]]))
                return action

            return choose

        results = evaluate_arm(
            arm=str(point_key),
            seed=int(seed),
            draw_ids=[int(d) for d in draw_ids],
            config_for_draw=config_for_draw,
            env_settings=dict(env_settings),
            scenario_id=str(scenario_id),
            choose_action_factory=factory,
            engine_seed=int(engine_seed),
        )
        if len(recorders) != len(draw_ids):
            raise ValueError(
                f"seed {seed}: {len(recorders)} conditioning trajectories were recorded for "
                f"{len(draw_ids)} draws; the recorder-to-draw mapping is not one to one and the "
                "diagnostic cannot be attributed"
            )
        for result, series in zip(results, recorders):
            if len(series) != expected_decisions:
                raise ValueError(
                    f"seed {seed} draw {result.draw_id}: {len(series)} decisions recorded but "
                    f"the episode horizon is {expected_decisions}; a short episode would make "
                    "the in-support fraction a fraction of the wrong denominator"
                )
            support.append(in_support_counts(series, rtg_min=rtg_min, rtg_max=rtg_max))
            index.append((int(seed), int(result.draw_id)))
            if (int(seed), int(result.draw_id)) == canary_cell:
                canary_series = tuple(series)
        episodes.extend(results)

    if not canary_series:
        raise ValueError(
            f"the canary cell {canary_cell} was not evaluated, so this point carries no "
            "recomputable conditioning trajectory for a later reader"
        )
    return PointRun(
        point_key=str(point_key),
        target_rtg=float(target_rtg),
        rtg_scale=float(
            torch.load(
                Path(next(iter(sorted(checkpoints.items())))[1]),
                map_location="cpu",
                weights_only=False,
            )["rtg_scale"]
        ),
        episodes=tuple(episodes),
        support=tuple(support),
        support_index=tuple(index),
        canary_rtg_series=canary_series,
        canary_cell=canary_cell,
    )


def gate_a_result(
    produced: Sequence[EpisodeResult], *, reference_path: str | Path
) -> GateAResult:
    """Compare the naive point against P4's committed records, per episode, with ``==``.

    Per **episode**, never by cell mean: a run in which two draws exchanged their values has the
    same mean and is not a reproduction.  All three recorded quantities are compared, because a
    matching ATT with a different vehicle count is two different episodes agreeing by accident.
    """
    path = Path(reference_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    reference = {
        (int(e["seed"]), int(e["draw_id"])): e
        for e in payload["episodes"]
    }
    if not reference:
        raise ValueError(f"{path}: carries no episode records to reproduce")
    got = {(int(e.seed), int(e.draw_id)): e for e in produced}

    uncovered = sorted(set(reference) - set(got))
    if uncovered:
        raise ValueError(
            f"this run does not cover {len(uncovered)} of the reference's "
            f"{len(reference)} cells (first {uncovered[:5]}); a reproduction check over a subset "
            "is not a reproduction"
        )

    mismatches: list[str] = []
    mismatched_cells: set[tuple[int, int]] = set()
    for cell in sorted(reference):
        want = reference[cell]
        have = got[cell]
        for field, mine in (
            ("att_horizon", have.att_horizon),
            ("horizon_vehicle_count", have.horizon_vehicle_count),
            ("episode_reward", have.episode_reward),
        ):
            theirs = float(want[field])
            if float(mine) != theirs:
                mismatches.append(
                    f"seed {cell[0]} draw {cell[1]} {field}: {mine!r} != {theirs!r}"
                )
                mismatched_cells.add(cell)
    return GateAResult(
        status="PASS" if not mismatches else "FAIL",
        n_compared=len(reference),
        n_mismatched=len(mismatched_cells),
        mismatches=tuple(mismatches[:20]),
        reference_path=str(path),
        reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        reference_mean=float(np.mean([float(e["att_horizon"]) for e in reference.values()])),
        reproduced_mean=float(np.mean([float(got[c].att_horizon) for c in reference])),
    )


# ----------------------------------------------------------------------
# Artifacts
# ----------------------------------------------------------------------


def locate_on_grid(target: float) -> dict[str, Any]:
    """Where *target* falls on the declared grid: the bracketing points, or which end it is past.

    ``BRIEF_15`` section 12.6: the secondary quantiles are **a labelling of the target space the
    grid already spans**, so each one is reported by saying which declared points it sits between
    rather than by being rolled out.
    """
    value = float(target)
    ordered = sorted(zip(DECLARED_GRID, GRID_POINT_KEYS))
    below = [(t, k) for t, k in ordered if t <= value]
    above = [(t, k) for t, k in ordered if t >= value]
    return {
        "target_rtg": value,
        "below_key": below[-1][1] if below else None,
        "below_target": below[-1][0] if below else None,
        "above_key": above[0][1] if above else None,
        "above_target": above[0][0] if above else None,
        "outside_grid": not below or not above,
    }


def probe_artifact(
    *,
    episodes: Sequence[ProbeEpisode],
    training_draw_ids: Sequence[int],
    best_source_return: float,
    rtg_range: tuple[float, float],
    rtg_scale: float,
    env_settings: Mapping[str, Any],
    engine_seed: int,
    scenario_id: str,
) -> dict[str, Any]:
    """``docs/data/p4_3_probe.json``: the probe, both rules' inputs and the source-domain ratio.

    Everything a later reader needs to recompute both rules without re-running anything: the
    per-episode returns by both routes, the distribution, every ``(q, k)`` Rule A target with its
    position on the declared grid, ``ratio_S(k)`` for both statistics, and Rule B's in-domain
    identity stated as the numbers that produce it.
    """
    if not episodes:
        raise ValueError("the probe artifact needs at least one episode")
    disagreeing = [
        e for e in episodes if e.local_return != e.local_return_from_lanes
    ]
    if disagreeing:
        raise ValueError(
            f"{len(disagreeing)} probe episode(s) disagree between their two routes "
            f"(first: draw {disagreeing[0].draw_id}, {disagreeing[0].local_return} from rewards "
            f"against {disagreeing[0].local_return_from_lanes} from lane waiting counts). The "
            "reward the return-to-go advances on is then not the queue it is claimed to be, and "
            "no target derived from it may be reported"
        )
    draws = [e.draw_id for e in episodes]
    assert_probe_draws_disjoint(
        draws, training_draw_ids=training_draw_ids, held_out_draws=HELD_OUT_DRAWS
    )
    returns = [e.local_return for e in episodes]
    by_draw = {e.draw_id: e.local_return for e in episodes}

    budgets: dict[str, Any] = {}
    for k in PROBE_K_VALUES:
        wanted = probe_draw_ids(k)
        if not set(wanted) <= set(by_draw):
            continue
        subset = [by_draw[d] for d in wanted]
        rule_a = {
            f"q{q:g}": {
                "quantile": float(q),
                "target_rtg": rule_a_target(subset, q),
                "grid_position": locate_on_grid(rule_a_target(subset, q)),
                "evaluated": bool(q == RULE_A_QUANTILE and k == RULE_A_K),
            }
            for q in (*SECONDARY_QUANTILES, RULE_A_QUANTILE)
        }
        budgets[f"k{k}"] = {
            "k": int(k),
            "draw_ids": list(wanted),
            "statistics": {
                statistic: probe_statistic(subset, statistic) for statistic in PROBE_STATISTICS
            },
            "source_domain_ratio": {
                statistic: source_domain_ratio(
                    subset, best_source_return=best_source_return, statistic=statistic
                )
                for statistic in PROBE_STATISTICS
            },
            "rule_b_in_domain_target": {
                statistic: rule_b_target(
                    best_source_return=best_source_return,
                    probe_source_stat=probe_statistic(subset, statistic),
                    probe_target_stat=probe_statistic(subset, statistic),
                )
                for statistic in PROBE_STATISTICS
            },
            "rule_a_targets": rule_a,
        }

    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "the in-domain MaxPressure probe: the input to both calibration rules, and the "
            "source-domain half of Rule B's ratio that P7 consumes"
        ),
        "probe_arm": PROBE_ARM,
        "probe_policy": "maxpressure",
        "scenario_id": str(scenario_id),
        "engine_seed": int(engine_seed),
        "env_settings": {k: v for k, v in dict(env_settings).items() if k != "compare_with"},
        "n_episodes": len(episodes),
        "draw_ids": sorted(draws),
        "disjointness": {
            "probe_draw_start": PROBE_DRAW_START,
            "training_draw_ids_source": "docs/data/p4_training.json",
            "n_training_draws": len(list(training_draw_ids)),
            "overlap_with_training": [],
            "overlap_with_held_out": [],
            "held_out_range": [int(HELD_OUT_DRAWS[0]), int(HELD_OUT_DRAWS[-1])],
        },
        "return_definition": (
            "per-intersection episode return under the collection reward, summed over the "
            "post-step infos of one episode; verified against -sum(lane_waiting over the "
            "intersection's incoming lanes) on every episode"
        ),
        "returns": {
            "min": min(returns),
            "max": max(returns),
            "mean": math.fsum(returns) / len(returns),
            "quantiles": {
                f"q{q:g}": float(np.quantile(np.asarray(returns, dtype=np.float64), q, method="linear"))
                for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            },
        },
        "best_source_return": float(best_source_return),
        "best_source_return_meaning": (
            "the maximum episode return of the training split -- the naive rule's target, and "
            "R_best_source in Rule B"
        ),
        "training_rtg_range": [float(rtg_range[0]), float(rtg_range[1])],
        "rtg_scale": float(rtg_scale),
        "budgets": budgets,
        "rules": {
            "rule_a": (
                "target = quantile(probe returns); the DECLARED ALTERNATIVE, predicted before "
                "the run to be poor because it targets probe-achievable rather than "
                "best-achievable return"
            ),
            "rule_b": (
                "target = best_source x (probe_target_stat / probe_source_stat); THE MECHANISM. "
                "In domain the two statistics coincide, the ratio is exactly 1 and the rule "
                "returns the naive target, so this task asserts the identity and measures the "
                "source-domain half of the ratio rather than validating the rule"
            ),
        },
        "episodes": [
            {
                "draw_id": e.draw_id,
                "local_return": e.local_return,
                "local_return_from_lanes": e.local_return_from_lanes,
                "att_horizon": e.att_horizon,
                "horizon_vehicle_count": e.horizon_vehicle_count,
                "decisions": e.decisions,
            }
            for e in sorted(episodes, key=lambda e: e.draw_id)
        ],
        "runtime": runtime_provenance(),
    }


def point_artifact(
    run: PointRun,
    *,
    rtg_range: tuple[float, float],
    engine_seed: int,
    env_settings: Mapping[str, Any],
    gate_a: GateAResult | None = None,
) -> dict[str, Any]:
    """One point's own artifact, written as its job finishes (``BRIEF_15`` section 12.5).

    Per-point rather than accumulated, so a crash or a ``DEFERRED`` 41 deadlock costs one point
    and not the campaign.  Both criteria travel together and the in-support block says in the
    artifact itself that it selects nothing.
    """
    fractions = [c.fraction for c in run.support]
    payload: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "point_key": run.point_key,
        "target_rtg": run.target_rtg,
        "rtg_scale": run.rtg_scale,
        "training_rtg_range": [float(rtg_range[0]), float(rtg_range[1])],
        "engine_seed": int(engine_seed),
        "env_settings": {k: v for k, v in dict(env_settings).items() if k != "compare_with"},
        "cell": _cell(list(run.episodes)),
        "in_support": {
            "selects": False,
            "role": (
                "reliability diagnostic and caveat generator; withdrawn as a selection criterion "
                "on 2026-08-13 (BRIEF_15 section 12.1)"
            ),
            "interval": "closed at both ends; below and above are strict",
            "mean_fraction": math.fsum(fractions) / len(fractions),
            "mean_in_support": math.fsum(c.in_support for c in run.support) / len(run.support),
            "mean_below": math.fsum(c.below for c in run.support) / len(run.support),
            "mean_above": math.fsum(c.above for c in run.support) / len(run.support),
            "n_episodes": len(run.support),
            "episodes": [
                {
                    "seed": seed,
                    "draw_id": draw,
                    "n": c.n,
                    "in_support": c.in_support,
                    "below": c.below,
                    "above": c.above,
                    "rtg_first": c.rtg_first,
                    "rtg_last": c.rtg_last,
                    "rtg_min": c.rtg_min,
                    "rtg_max": c.rtg_max,
                }
                for (seed, draw), c in zip(run.support_index, run.support)
            ],
        },
        "canary": {
            "role": (
                "one episode's full conditioning trajectory, so a later reader can recompute the "
                "in-support fraction without re-running the rollout"
            ),
            "cell": [int(run.canary_cell[0]), int(run.canary_cell[1])],
            "rtg_series": list(run.canary_rtg_series),
        },
        "episodes": [
            {
                "arm": e.arm,
                "seed": e.seed,
                "draw_id": e.draw_id,
                "att_horizon": e.att_horizon,
                "horizon_vehicle_count": e.horizon_vehicle_count,
                "episode_reward": e.episode_reward,
            }
            for e in run.episodes
        ],
        "runtime": runtime_provenance(),
    }
    if gate_a is not None:
        payload["gate_a"] = {
            "status": gate_a.status,
            "n_compared": gate_a.n_compared,
            "n_mismatched": gate_a.n_mismatched,
            "mismatches": list(gate_a.mismatches),
            "reference_path": gate_a.reference_path,
            "reference_sha256": gate_a.reference_sha256,
            "reference_mean": gate_a.reference_mean,
            "reproduced_mean": gate_a.reproduced_mean,
        }
    return payload


def report_artifact(
    *,
    points: Sequence[Mapping[str, Any]],
    probe: Mapping[str, Any],
    rtg_range: tuple[float, float],
    checkpoints: Mapping[int, str],
) -> dict[str, Any]:
    """``docs/data/p4_3_rtg.json``: the landscape, both criteria, and the paired statistics.

    Every point is checked against the **declaration** before it is reported: an undeclared key
    and a declared key carrying a moved target are both refused, so a grid read off the results
    cannot be published.  All statistics are labelled exploratory (``PREREGISTRATION.md`` section
    2 files calibrated-versus-naive RTG prompting there), and the selection block records in the
    artifact that neither criterion chose anything.
    """
    from offline.offline_baselines import paired_comparison

    declared = grid_targets()
    scales = {float(p["rtg_scale"]) for p in points}
    if len(scales) > 1:
        raise ValueError(
            f"the points disagree about rtg_scale ({sorted(scales)}); it is the normalisation "
            "divisor, not a prompt, and points measured under two divisors are not comparable"
        )
    for point in points:
        key = str(point["point_key"])
        if key in declared:
            if float(point["target_rtg"]) != declared[key]:
                raise ValueError(
                    f"point {key} carries target {point['target_rtg']} but its declared target "
                    f"is {declared[key]}; the grid is declared in docs/plans/p4.3.md and may not "
                    "be re-derived from a run"
                )
        elif key != RULE_A_POINT_KEY:
            raise ValueError(
                f"{key} is not a declared point: the declared grid is {sorted(declared)} plus "
                f"{RULE_A_POINT_KEY!r}, and an undeclared point may not enter the report"
            )

    episodes_by_key = {
        str(p["point_key"]): [EpisodeResult(**e) for e in p.get("episodes", [])] for p in points
    }
    naive_key = next(
        (k for k, target in declared.items() if target == NAIVE_TARGET), None
    )
    comparisons: dict[str, Any] = {}
    if naive_key in episodes_by_key and episodes_by_key[naive_key]:
        for key, results in episodes_by_key.items():
            if key == naive_key or not results:
                continue
            comparison = paired_comparison(results, episodes_by_key[naive_key])
            comparisons[f"{key}_vs_{naive_key}"] = {
                "left_arm": comparison.left_arm,
                "right_arm": comparison.right_arm,
                "n_shared_draws": comparison.n_shared_draws,
                "mean_left": comparison.mean_left,
                "mean_right": comparison.mean_right,
                "mean_difference": comparison.mean_difference,
                "ci95_low": comparison.ci95_low,
                "ci95_high": comparison.ci95_high,
                "median_difference": comparison.median_difference,
                "wins": comparison.wins,
                "losses": comparison.losses,
                "ties": comparison.ties,
                "rank_biserial": comparison.rank_biserial,
                "p_value": comparison.wilcoxon.p_value,
                "z": comparison.wilcoxon.z,
            }

    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "P4.3's RTG landscape on the registered held-out pool: the declared grid, the naive "
            "rule's position on it, Rule A as the declared alternative, and the in-support "
            "reliability diagnostic. EXPLORATORY (PREREGISTRATION.md section 2): effect sizes and "
            "CIs, no inferential claim, no multiplicity correction"
        ),
        "analysis_status": "exploratory",
        "declared_grid": list(DECLARED_GRID),
        "declared_grid_keys": list(GRID_POINT_KEYS),
        "naive_target": NAIVE_TARGET,
        "naive_point_key": naive_key,
        "rule_a_point_key": RULE_A_POINT_KEY,
        "training_rtg_range": [float(rtg_range[0]), float(rtg_range[1])],
        "checkpoints": {str(seed): str(path) for seed, path in sorted(checkpoints.items())},
        "selection": {
            "att_may_select": False,
            "in_support_may_select": False,
            "note": (
                "PREREGISTRATION.md A8(a) and section 6.1: no target reported here was chosen "
                "because it scored well. The in-support fraction was withdrawn as a selector on "
                "2026-08-13 and is a reliability diagnostic. The mechanism (Rule B) is justified "
                "by its form and is an identity in domain"
            ),
        },
        "points": list(points),
        "comparisons_vs_naive": comparisons,
        "probe": probe,
        "runtime": runtime_provenance(
            measurement_git_commits=[
                str(p["runtime"]["git_commit"])
                for p in points
                if isinstance(p.get("runtime"), Mapping) and p["runtime"].get("git_commit")
            ]
        ),
    }


def effect_size_sidecar(
    gate_path: str | Path, thresholds_path: str | Path | None = None
) -> dict[str, Any]:
    """``DEFERRED`` 31: P4's missing effect sizes, recomputed from its own committed records.

    A **sidecar**, ruled 2026-08-13 (``BRIEF_15`` section 12.3): the committed artifact is read
    and its sha256 recorded, never edited.  ``PREREGISTRATION.md`` section 8 makes effect sizes
    mandatory beside every p-value and P4 reported three p-values with none; this is arithmetic
    over committed data, so it costs no re-run and moves no reported number.
    """
    gate_file = Path(gate_path)
    gate = json.loads(gate_file.read_text(encoding="utf-8"))
    thresholds_file = (
        Path(thresholds_path)
        if thresholds_path is not None
        else gate_file.parent / "p4_heldout_thresholds.json"
    )
    thresholds = json.loads(thresholds_file.read_text(encoding="utf-8"))

    from offline.offline_baselines import paired_comparison

    episodes = [EpisodeResult(**e) for e in gate["episodes"]]
    episodes.extend(EpisodeResult(**e) for e in thresholds["episodes"])
    by_arm: dict[str, list[EpisodeResult]] = {}
    for episode in episodes:
        by_arm.setdefault(episode.arm, []).append(episode)

    comparisons: dict[str, Any] = {}
    for name, recorded in gate["wilcoxon"].items():
        left_arm, _, right_arm = name.partition("_vs_")
        if left_arm not in by_arm or right_arm not in by_arm:
            raise ValueError(
                f"{gate_file}: comparison {name!r} names arms that carry no episode records "
                f"({sorted(by_arm)}); its effect size cannot be recomputed from committed data"
            )
        comparison = paired_comparison(by_arm[left_arm], by_arm[right_arm])
        if comparison.n_shared_draws != int(recorded["n_shared_draws"]):
            raise ValueError(
                f"{name}: recomputed over {comparison.n_shared_draws} shared draws but the "
                f"artifact recorded {recorded['n_shared_draws']}; the two are not the same test"
            )
        if comparison.wilcoxon.p_value != float(recorded["p_value"]):
            raise ValueError(
                f"{name}: recomputed p-value {comparison.wilcoxon.p_value} does not equal the "
                f"committed {recorded['p_value']}; this sidecar may only annotate P4's numbers, "
                "never restate them differently"
            )
        comparisons[name] = {
            "rank_biserial": comparison.rank_biserial,
            "mean_difference": comparison.mean_difference,
            "ci95_low": comparison.ci95_low,
            "ci95_high": comparison.ci95_high,
            "median_difference": comparison.median_difference,
            "wins": comparison.wins,
            "losses": comparison.losses,
            "ties": comparison.ties,
            "n_shared_draws": comparison.n_shared_draws,
            "p_value": comparison.wilcoxon.p_value,
            "z": comparison.wilcoxon.z,
        }

    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "DEFERRED 31: the effect sizes PREREGISTRATION.md section 8 requires beside every "
            "p-value, for P4's three paired tests. A SIDECAR: p4_gate.json is read and hashed, "
            "never edited (ruled 2026-08-13, BRIEF_15 section 12.3)"
        ),
        "source": {
            "path": str(gate_file),
            "sha256": hashlib.sha256(gate_file.read_bytes()).hexdigest(),
            "thresholds_path": str(thresholds_file),
            "thresholds_sha256": hashlib.sha256(thresholds_file.read_bytes()).hexdigest(),
        },
        "effect_size_definition": (
            "rank-biserial correlation (w_plus - w_minus) / (w_plus + w_minus), with the sign "
            "convention of offline.offline_baselines.rank_biserial: negative means the LEFT arm "
            "(the DT) had the lower ATT more often and by larger ranks"
        ),
        "comparisons": comparisons,
        "runtime": runtime_provenance(
            measurement_git_commits=[
                c
                for c in [
                    str(gate.get("runtime", {}).get("git_commit", "")),
                    str(thresholds.get("runtime", {}).get("git_commit", "")),
                ]
                if c
            ]
        ),
    }


def leaf_diff(left: Any, right: Any) -> LeafDiff:
    """Recursive leaf comparison of two JSON structures, for ``DEFERRED`` 39's proof.

    Paths are dotted with bracketed list indices (``b.c[2].d``).  The instrument exists because
    an artifact that embeds a description of its own creation cannot be compared for byte
    identity across creations: the differences have to be **named in advance** and checked
    exhaustively, which a whole-file hash cannot do (Decisions Log, 2026-08-12).
    """
    removed: list[str] = []
    added: list[str] = []
    changed: list[str] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in a:
                child = f"{path}.{key}" if path else str(key)
                if key not in b:
                    removed.append(child)
                else:
                    walk(a[key], b[key], child)
            for key in b:
                if key not in a:
                    added.append(f"{path}.{key}" if path else str(key))
            return
        if isinstance(a, list) and isinstance(b, list):
            for index in range(min(len(a), len(b))):
                walk(a[index], b[index], f"{path}[{index}]")
            for index in range(len(b), len(a)):
                removed.append(f"{path}[{index}]")
            for index in range(len(a), len(b)):
                added.append(f"{path}[{index}]")
            return
        if type(a) is not type(b) or a != b:
            changed.append(path)

    walk(left, right, "")
    return LeafDiff(
        removed=tuple(sorted(removed)),
        added=tuple(sorted(added)),
        changed=tuple(sorted(changed)),
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def checkpoint_map(specs: Sequence[str]) -> dict[int, str]:
    """``SEED=PATH`` specifications into ``{seed: path}``, refusing a set that is not the declared one."""
    out: dict[int, str] = {}
    for spec in specs:
        seed, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"--checkpoint expects SEED=PATH, got {spec!r}")
        out[int(seed)] = path
    if tuple(sorted(out)) != tuple(sorted(TRAINING_SEEDS)):
        raise ValueError(
            f"the declared training seeds are {list(TRAINING_SEEDS)} and this run was given "
            f"{sorted(out)}; a point measured over a different seed set is not comparable with "
            "the others"
        )
    return out


def assert_checkpoints_declare_the_naive_target(checkpoints: Mapping[int, str]) -> None:
    """Every checkpoint's own ``target_rtg`` must equal the grid's naive point.

    The declared grid is anchored on P4's configuration.  If a checkpoint disagrees, the naive
    grid point is no longer P4's target, Gate A compares two different configurations, and the
    landscape's most important label is wrong.
    """
    for seed, path in sorted(checkpoints.items()):
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        recorded = float(payload["target_rtg"])
        if recorded != NAIVE_TARGET:
            raise ValueError(
                f"checkpoint for seed {seed} declares target_rtg {recorded} but the declared "
                f"naive grid point is {NAIVE_TARGET}; the grid is anchored on the configuration "
                "P4 reported and these must not diverge"
            )


def build_parser() -> argparse.ArgumentParser:
    """CLI: ``probe``, ``evaluate``, ``report``, ``effect-sizes``, ``provenance-proof``."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.rtg_calibration",
        description=(
            "P4.3: the in-domain RTG landscape, the MaxPressure probe and the two calibration "
            "rules. Nothing here trains anything -- the target is an inference-time parameter."
        ),
    )
    parser.add_argument("--manifest", required=True, help="a corpus manifest; env settings come from it")
    parser.add_argument("--draws-root", default="scenarios/draws")
    parser.add_argument("--scenario-key", default="cityflow1x1")
    parser.add_argument("--scenario-id", default="cityflow1x1")
    parser.add_argument("--engine-seed", type=int, default=1000)
    parser.add_argument("--out-dir", default="docs/data")
    parser.add_argument("--work-dir", default="output/p4_3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=1, help="torch threads; 1 is the pin")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="run the MaxPressure probe and write its artifact")
    probe.add_argument("--k", type=int, default=max(PROBE_K_VALUES))
    probe.add_argument("--checkpoint", action="append", default=[], metavar="SEED=PATH")

    evaluate = sub.add_parser("evaluate", help="evaluate ONE declared point over the held-out pool")
    evaluate.add_argument("--point", required=True, help=f"one of {list(GRID_POINT_KEYS)} or {RULE_A_POINT_KEY}")
    evaluate.add_argument("--checkpoint", action="append", default=[], metavar="SEED=PATH")
    evaluate.add_argument("--steps", type=int, default=40_000)
    evaluate.add_argument("--reference", default="docs/data/p4_gate.json")

    report = sub.add_parser("report", help="merge the per-point artifacts into the committed one")
    report.add_argument("--checkpoint", action="append", default=[], metavar="SEED=PATH")

    sub.add_parser("effect-sizes", help="DEFERRED 31: the sidecar for P4's paired tests")

    proof = sub.add_parser(
        "provenance-proof",
        help="DEFERRED 39: compare two artifacts against a DECLARED expected-difference set",
    )
    proof.add_argument("--left", required=True, help="the committed artifact")
    proof.add_argument("--right", required=True, help="the regenerated artifact")
    proof.add_argument("--expect-changed", action="append", default=[])
    proof.add_argument("--expect-added", action="append", default=[])
    proof.add_argument("--expect-removed", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    from offline.materialise_draws import draw_config_path
    from offline.offline_baselines import pin_torch_threads

    args = build_parser().parse_args(argv)
    pin_torch_threads(int(args.threads))
    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"--out-dir does not exist: {out_dir}")

    if args.command == "provenance-proof":
        return _run_provenance_proof(args)
    if args.command == "effect-sizes":
        write_json_atomic(
            effect_size_sidecar(out_dir / "p4_gate.json"),
            out_dir / "p4_3_p4_gate_effect_sizes.json",
        )
        print(f"wrote {out_dir / 'p4_3_p4_gate_effect_sizes.json'}", flush=True)
        return 0

    settings = env_settings_from_manifest(args.manifest)

    def config_for_draw(draw_id: int) -> Path:
        return draw_config_path(args.scenario_key, draw_id, out_root=args.draws_root)

    if args.command == "probe":
        return _run_probe(args, settings, config_for_draw, out_dir)
    if args.command == "evaluate":
        return _run_evaluate(args, settings, config_for_draw, out_dir)
    return _run_report(args, out_dir)


def _run_probe(
    args: argparse.Namespace,
    settings: dict[str, Any],
    config_for_draw: Callable[[int], Path],
    out_dir: Path,
) -> int:
    """The probe: the rule's input, measured before any point is evaluated."""
    checkpoints = checkpoint_map(args.checkpoint)
    assert_checkpoints_declare_the_naive_target(checkpoints)
    reference = sorted(checkpoints.items())[0][1]
    rtg_range = training_rtg_range(reference)
    payload = torch.load(Path(reference), map_location="cpu", weights_only=False)

    training = json.loads((out_dir / "p4_training.json").read_text(encoding="utf-8"))
    draws = probe_draw_ids(int(args.k))
    assert_probe_draws_disjoint(
        draws,
        training_draw_ids=training["training_draw_ids"],
        held_out_draws=HELD_OUT_DRAWS,
    )
    print(f"probe: MaxPressure over draws {draws[0]}-{draws[-1]} ({len(draws)})", flush=True)
    episodes = run_probe(
        draw_ids=draws,
        config_for_draw=config_for_draw,
        env_settings=settings,
        scenario_id=args.scenario_id,
        engine_seed=int(args.engine_seed),
    )
    artifact = probe_artifact(
        episodes=episodes,
        training_draw_ids=training["training_draw_ids"],
        best_source_return=float(payload["target_rtg"]),
        rtg_range=rtg_range,
        rtg_scale=float(payload["rtg_scale"]),
        env_settings=settings,
        engine_seed=int(args.engine_seed),
        scenario_id=args.scenario_id,
    )
    write_json_atomic(artifact, out_dir / "p4_3_probe.json")
    returns = artifact["returns"]
    print(
        f"  probe returns: max {returns['max']:.1f}  mean {returns['mean']:.1f}  "
        f"min {returns['min']:.1f}",
        flush=True,
    )
    for name, block in sorted(artifact["budgets"].items()):
        rule_a = block["rule_a_targets"][f"q{RULE_A_QUANTILE:g}"]["target_rtg"]
        print(
            f"  {name}: rule A (q=1) -> {rule_a:.1f}   ratio_max "
            f"{block['source_domain_ratio']['max']:.4f}   ratio_mean "
            f"{block['source_domain_ratio']['mean']:.4f}",
            flush=True,
        )
    return 0


def _target_for_point(point_key: str, out_dir: Path) -> float:
    """The target a point key denotes: a declared grid value, or Rule A's probe-derived one."""
    declared = grid_targets()
    if point_key in declared:
        return declared[point_key]
    if point_key != RULE_A_POINT_KEY:
        raise ValueError(
            f"{point_key!r} is not a declared point: the grid is {sorted(declared)} plus "
            f"{RULE_A_POINT_KEY!r}"
        )
    probe = json.loads((out_dir / "p4_3_probe.json").read_text(encoding="utf-8"))
    budget = probe["budgets"][f"k{RULE_A_K}"]
    return float(budget["rule_a_targets"][f"q{RULE_A_QUANTILE:g}"]["target_rtg"])


def _run_evaluate(
    args: argparse.Namespace,
    settings: dict[str, Any],
    config_for_draw: Callable[[int], Path],
    out_dir: Path,
) -> int:
    """Evaluate ONE point, then write ITS OWN artifact so a crash costs one point."""
    checkpoints = checkpoint_map(args.checkpoint)
    assert_checkpoints_declare_the_naive_target(checkpoints)
    target = _target_for_point(args.point, out_dir)
    draws = list(HELD_OUT_DRAWS)
    requested = requested_runs(args.point, TRAINING_SEEDS, draws)

    print(
        f"{args.point}: target_rtg {target} over {len(TRAINING_SEEDS)} seeds x {len(draws)} draws",
        flush=True,
    )
    started = time.time()
    run = evaluate_point(
        point_key=args.point,
        target_rtg=target,
        checkpoints=checkpoints,
        draw_ids=draws,
        config_for_draw=config_for_draw,
        env_settings=settings,
        scenario_id=args.scenario_id,
        engine_seed=int(args.engine_seed),
        declared_gradient_steps=int(args.steps),
        device=args.device,
    )
    from offline.offline_baselines import assert_campaign_complete

    assert_campaign_complete(requested, list(run.episodes))

    gate = None
    if target == NAIVE_TARGET:
        gate = gate_a_result(list(run.episodes), reference_path=args.reference)

    rtg_range = training_rtg_range(sorted(checkpoints.items())[0][1])
    artifact = point_artifact(
        run,
        rtg_range=rtg_range,
        engine_seed=int(args.engine_seed),
        env_settings=settings,
        gate_a=gate,
    )
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(artifact, work_dir / f"eval_{args.point}.json")

    cell = artifact["cell"]
    print(
        f"  att_horizon {cell['att_horizon_mean']:.4f} +/- {cell['att_horizon_ci95']:.4f}  "
        f"vehicle_count {cell['horizon_vehicle_count_mean']:.2f}  "
        f"in-support {artifact['in_support']['mean_fraction']:.4f} "
        f"(below {artifact['in_support']['mean_below']:.1f}, "
        f"above {artifact['in_support']['mean_above']:.1f})  "
        f"n={cell['n_episodes']}  {time.time() - started:.0f}s",
        flush=True,
    )
    if gate is not None:
        print(
            f"  GATE A {gate.status}: {gate.n_compared} cells compared, "
            f"{gate.n_mismatched} mismatched, reference mean {gate.reference_mean:.10f} "
            f"against {gate.reproduced_mean:.10f}",
            flush=True,
        )
        for line in gate.mismatches[:5]:
            print(f"    {line}", flush=True)
        if gate.status != "PASS":
            return 1
    return 0


def _run_report(args: argparse.Namespace, out_dir: Path) -> int:
    """Merge the per-point artifacts, refusing to report anything if Gate A did not pass."""
    work_dir = Path(args.work_dir)
    checkpoints = checkpoint_map(args.checkpoint)
    points = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(work_dir.glob("eval_*.json"))
    ]
    if not points:
        raise FileNotFoundError(f"no per-point artifacts under {work_dir}")

    gates = [p["gate_a"] for p in points if "gate_a" in p]
    if len(gates) != 1:
        raise ValueError(
            f"exactly one point carries Gate A (the naive target) and {len(gates)} do; without "
            "it the instrument has not been shown to reproduce P4 and no number may be reported"
        )
    if gates[0]["status"] != "PASS":
        raise ValueError(
            f"Gate A did not pass ({gates[0]['status']}, {gates[0]['n_mismatched']} mismatched "
            "cells); no number from this task may be reported"
        )

    probe = json.loads((out_dir / "p4_3_probe.json").read_text(encoding="utf-8"))
    artifact = report_artifact(
        points=points,
        probe=probe,
        rtg_range=tuple(points[0]["training_rtg_range"]),
        checkpoints=checkpoints,
    )
    write_json_atomic(artifact, out_dir / "p4_3_rtg.json")
    print(f"wrote {out_dir / 'p4_3_rtg.json'} with {len(points)} points", flush=True)
    for point in sorted(points, key=lambda p: -float(p["target_rtg"])):
        print(
            f"  {point['point_key']:>16s}  target {point['target_rtg']:>9.1f}  "
            f"att {point['cell']['att_horizon_mean']:8.4f} "
            f"+/- {point['cell']['att_horizon_ci95']:.4f}  "
            f"in-support {point['in_support']['mean_fraction']:.4f}",
            flush=True,
        )
    return 0


def _run_provenance_proof(args: argparse.Namespace) -> int:
    """``DEFERRED`` 39: the observed leaf differences must equal the DECLARED ones, exactly.

    Declared on the command line and therefore visible in the log: an expected difference named
    in advance is a specification, and the same difference explained afterwards is an excuse
    (Decisions Log, 2026-08-12).
    """
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    diff = leaf_diff(left, right)
    expected = {
        "changed": tuple(sorted(args.expect_changed)),
        "added": tuple(sorted(args.expect_added)),
        "removed": tuple(sorted(args.expect_removed)),
    }
    observed = {"changed": diff.changed, "added": diff.added, "removed": diff.removed}
    print(f"left  {args.left}\nright {args.right}", flush=True)
    for kind in ("changed", "added", "removed"):
        print(f"  {kind:>7s}: declared {list(expected[kind])} observed {list(observed[kind])}", flush=True)
    if observed != expected:
        print("PROOF FAILED: the observed difference set is not the declared one", flush=True)
        return 1
    print("PROOF PASSED: every leaf outside the declared set is identical", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
