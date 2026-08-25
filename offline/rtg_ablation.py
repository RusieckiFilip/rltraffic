"""P5.3a: does the return prompt do anything at all?  A teacher-forced RTG sensitivity probe.

``PREREGISTRATION`` A9 names **probe-calibrated return prompting** as this paper's one contribution.
P4.3 swept the return target across a 13,000-unit grid and moved held-out ATT by **0.9026**
(``docs/data/p4_3_rtg.json``), which is consistent with two entirely different worlds: a model that
is *robust* to the target, and a model that *ignores* the RTG token because the corpus lacks the
return spread that would make the conditioning identifiable.  An inference-time ATT sweep cannot
tell them apart.  This module can, because it measures the **actions** rather than their outcome.

Artifact format version: ``p5.3a-rtg-probe/1.0``.

How the probe works, and why teacher forcing is the whole design
----------------------------------------------------------------
For one committed ``dt`` checkpoint and a set of logged episodes from the corpus it trained on, the
episode's **logged states and logged actions** are replayed through the model.  The state sequence is
therefore **identical across interventions by construction**, so any difference in the output is
attributable to the return-to-go alone.  A live rollout could not make that claim: a changed action
changes the next state, and the two effects would be inseparable.

Windows come from ``DTAgent._window`` and normalisation from ``DTAgent._normalise_state`` -- the
production inference path, not a re-implementation of it.  In particular the current step's action
slot carries ``PAD_ACTION``, exactly as it does when the agent drives, rather than the loader's
training-time layout.

Two conventions worth stating, because getting either wrong yields a plausible number
--------------------------------------------------------------------------------------
* :func:`conditioning_series` returns the **unscaled** return-to-go, as ``DTAgent`` carries it in
  ``_Context.rtg``.  The division by ``rtg_scale`` happens once, inside ``DTAgent._window``
  (``agent/DTAgent.py:596``).  A probe that scaled twice would look entirely reasonable.
* :func:`teacher_forced_logits` returns the model's **raw** head output.  The availability mask is
  applied once, inside :func:`compare_logits`, through the frozen
  ``agent.DTAgent.masked_action_logits``.

⚠️ **``zero`` and ``grid_g0`` are different interventions.**  ``grid_g0`` sets the *target* to 0 and
keeps decrementing, giving ``-cumsum(r)``; ``zero`` feeds a constant 0, which is what a
``rtg_mode="zero"`` model manufactures for itself.

What this module reports, and what it refuses to
-------------------------------------------------
``flip_rate`` is the headline (registered in ``docs/plans/p5.3a.md``, chosen because it is the direct
analogue of P5.1's *"48.83 % of actions flip without the graph"*).  ``tvd`` is **required beside it
and is not decoration**: a flip rate of 0 with a large TVD means the logits moved a great deal and
never crossed a decision boundary, which is a different finding from an inert token and the paper
must be able to tell them apart.

The artifact emits **no verdict** (``assert_no_verdicts``).  It emits measured quantities; the
reading belongs to the coordinator.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from agent.DTAgent import DTAgent, masked_action_logits
from offline.dataset import DRAW_SPLITS, MANIFEST_NAME, RTG_QUANTILES, RtgSummary
from offline.dt_gate import (
    HELD_OUT_DRAWS,
    TRAINING_SEEDS,
    build_training_dataset,
    runtime_provenance,
    write_json_atomic,
)
from offline.method_tier_grid import (
    CONTEXT_LENGTH,
    DECLARED_GRADIENT_STEPS,
    TIERS,
    arm_key,
    assert_no_verdicts,
    canonical_digest_of,
    env_settings_for_tiers,
    measurement_commits,
    tier_dirs,
    tier_spec,
    training_streams,
)
from offline.offline_baselines import StreamReturn
from offline.rtg_calibration import DECLARED_GRID, agent_with_target
from offline.trajectory_logger import load_episode

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "GRID_INTERVENTION_KEYS",
    "INTERVENTION_KEYS",
    "PROBE_SEEDS",
    "PROBE_STREAM_COUNT",
    "PROBE_STREAM_STRIDE",
    "PROBE_TIERS",
    "RTG_SPREAD_TIMESTEPS",
    "Intervention",
    "InterventionComparison",
    "ProbeCell",
    "assert_declared_interventions",
    "behaviour_margin_degenerate",
    "between_episode_rtg_spread",
    "checkpoint_path_for",
    "committed_rtg_summary",
    "compare_logits",
    "conditioning_series",
    "crosscheck_targets",
    "declared_interventions",
    "delta_table",
    "episode_return_stats",
    "flip_rate",
    "main",
    "probe_cell",
    "ramp_prediction",
    "recomputed_rtg_summary",
    "report_artifact",
    "selected_stream_indices",
    "spread_table",
    "teacher_forced_logits",
    "total_variation_distance",
]

ARTIFACT_FORMAT_VERSION = "p5.3a-rtg-probe/1.0"

#: The eight hz1x1 tiers of P4.6 and P4.7.  The probe and the tables share one list (AMENDMENT B1):
#: ``flip_rate`` is only interpretable against the between-episode spread, and a correlation over
#: five points where the mixtures are the only structurally different case is not a measurement.
PROBE_TIERS: tuple[str, ...] = (
    "fixedtime",
    "mappo1000",
    "mappo500",
    "maxpressure",
    "mix33",
    "mix50",
    "mix67",
    "random",
)

PROBE_SEEDS: tuple[int, ...] = TRAINING_SEEDS

#: 20 streams at stride 10 out of each tier's declared 200-stream training set.  The stride is what
#: makes the selection stratified: the ``mappo`` and mixture tiers draw from five behaviour-seed
#: directories of 40 episodes each, so the first 20 in key order would all come from one of them.
PROBE_STREAM_COUNT = 20
PROBE_STREAM_STRIDE = 10

#: Row B's fixed timesteps.  The model sees an RTG at EVERY step, so identifiability is about how
#: much that value varies across episodes at a given point, not about the episode-return spread.
RTG_SPREAD_TIMESTEPS: tuple[int, ...] = (0, 90, 180, 270)

#: Positional keys for P4.3's nine declared targets, in declaration order.
GRID_INTERVENTION_KEYS: tuple[str, ...] = tuple(f"grid_g{i}" for i in range(len(DECLARED_GRID)))

#: The twelve declared interventions.  Registered before any number existed; the grid may not grow.
INTERVENTION_KEYS: tuple[str, ...] = ("baseline", *GRID_INTERVENTION_KEYS, "zero", "frozen")

#: Which tiers' checkpoints live where.  ``mappo1000``'s ``dt`` column is P4's, re-used by P4.6.
_CHECKPOINT_LAYOUT: Mapping[str, str] = {
    "mappo1000": "p4_dt/dt_seed{seed}.pt",
    "fixedtime": "p4_6/checkpoints/fixedtime_dt_seed{seed}.pt",
    "mappo500": "p4_6/checkpoints/mappo500_dt_seed{seed}.pt",
    "maxpressure": "p4_6/checkpoints/maxpressure_dt_seed{seed}.pt",
    "random": "p4_6/checkpoints/random_dt_seed{seed}.pt",
    "mix33": "p4_7/checkpoints/mix33_dt_seed{seed}.pt",
    "mix50": "p4_7/checkpoints/mix50_dt_seed{seed}.pt",
    "mix67": "p4_7/checkpoints/mix67_dt_seed{seed}.pt",
}

#: The crosscheck's two targets: ``DECLARED_GRID``'s endpoints.
_CROSSCHECK_TIER = "mappo1000"
_CROSSCHECK_SEED = 101
_CROSSCHECK_DRAW = 1000

_LIMITATION = (
    "The probe measures sensitivity on BEHAVIOUR-POLICY states -- the distribution the "
    "conditioning was learned on, and not the distribution the DT visits when it drives. The "
    "crosscheck bounds that gap for exactly one cell by rolling the model live under the grid's "
    "two endpoints; it does not close it for the other 39."
)


@dataclass(frozen=True)
class Intervention:
    """One RTG intervention: which series to feed the model, and under which name."""

    key: str
    kind: str
    target_rtg: float


@dataclass(frozen=True)
class InterventionComparison:
    """One intervention measured against the baseline on the same states."""

    key: str
    flip_rate: float
    tvd: float
    mean_abs_logit_delta: float
    n_steps_compared: int

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "flip_rate": float(self.flip_rate),
            "tvd": float(self.tvd),
            "mean_abs_logit_delta": float(self.mean_abs_logit_delta),
            "n_steps_compared": int(self.n_steps_compared),
        }


@dataclass(frozen=True)
class ProbeCell:
    """One (tier, seed) probe cell and every intervention measured on it."""

    tier: str
    seed: int
    checkpoint: str
    target_rtg: float
    rtg_scale: float
    n_streams: int
    n_steps: int
    comparisons: tuple[InterventionComparison, ...]

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "seed": int(self.seed),
            "checkpoint": self.checkpoint,
            "target_rtg": float(self.target_rtg),
            "rtg_scale": float(self.rtg_scale),
            "n_streams": int(self.n_streams),
            "n_steps": int(self.n_steps),
            "interventions": {c.key: c.to_json_obj() for c in self.comparisons},
        }


# ----------------------------------------------------------------------
# The declared interventions and the series each one feeds
# ----------------------------------------------------------------------


def declared_interventions(target_rtg: float) -> tuple[Intervention, ...]:
    """The twelve declared interventions for a checkpoint whose own target is *target_rtg*."""
    target = float(target_rtg)
    out = [Intervention(key="baseline", kind="decrement", target_rtg=target)]
    out.extend(
        Intervention(key=key, kind="decrement", target_rtg=float(value))
        for key, value in zip(GRID_INTERVENTION_KEYS, DECLARED_GRID)
    )
    out.append(Intervention(key="zero", kind="zero", target_rtg=0.0))
    out.append(Intervention(key="frozen", kind="frozen", target_rtg=target))
    return tuple(out)


def conditioning_series(
    target_rtg: float, rewards: np.ndarray, *, kind: str = "decrement"
) -> np.ndarray:
    """The UNSCALED return-to-go the model conditions on at each decision step.

    ``decrement`` is the inference-time construction ``rtg_t = target - sum(r_k for k < t)``
    (``agent/DTAgent.py:662``); ``frozen`` holds the target and never decrements, which quantifies
    the hazard ``agent/DTAgent.py:560,628`` documents in prose; ``zero`` is a constant 0, which is
    what a ``rtg_mode="zero"`` model manufactures for itself.

    Accumulated with an explicit forward loop rather than ``np.cumsum`` for the reason
    ``offline/dataset._returns_to_go`` gives for the same choice: the tests recompute this **with**
    ``np.cumsum``, and a check that shares its implementation with the thing it checks is not a
    check.  The accumulation order is identical, so the two agree bit for bit.
    """
    values = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("an episode with no rewards has no conditioning series")
    if kind == "zero":
        return np.zeros(values.shape, dtype=np.float64)
    if kind == "frozen":
        return np.full(values.shape, float(target_rtg), dtype=np.float64)
    if kind != "decrement":
        raise ValueError(
            f"unknown conditioning kind {kind!r}; the declared kinds are 'decrement', 'frozen' "
            "and 'zero'"
        )
    out = np.empty(values.shape, dtype=np.float64)
    consumed = 0.0
    for t in range(values.size):
        out[t] = float(target_rtg) - consumed
        consumed += float(values[t])
    return out


def selected_stream_indices(
    total: int, *, count: int = PROBE_STREAM_COUNT, stride: int = PROBE_STREAM_STRIDE
) -> tuple[int, ...]:
    """Indices ``0, stride, 2*stride, ...`` -- the registered stratified selection rule (R4)."""
    size = int(total)
    if size < 1:
        raise ValueError(f"a tier needs at least one stream, got {total!r}")
    if int(count) < 1 or int(stride) < 1:
        raise ValueError(f"count and stride must both be >= 1, got {count!r} and {stride!r}")
    span = (int(count) - 1) * int(stride) + 1
    if size < span:
        raise ValueError(
            f"cannot select {int(count)} streams at stride {int(stride)} from {size}: the rule "
            f"needs at least {span} streams, and silently taking fewer would make the probe's "
            "sample size depend on the tier"
        )
    return tuple(range(0, int(count) * int(stride), int(stride)))


# ----------------------------------------------------------------------
# The instrument
# ----------------------------------------------------------------------


def teacher_forced_logits(
    agent: Any,
    ix_id: str,
    *,
    state: np.ndarray,
    avail_mask: np.ndarray,
    action: np.ndarray,
    rtg_unscaled: np.ndarray,
) -> np.ndarray:
    """Raw ``(T, n_actions)`` action logits, one row per decision, under a given RTG series.

    The context is advanced with the **logged** action at every step, which is what makes this
    teacher forcing: the state and action history are the corpus's, so two calls differing only in
    *rtg_unscaled* differ only in the return prompt.

    The availability mask is **not** applied here -- it is applied once in :func:`compare_logits`.
    """
    states = np.asarray(state, dtype=np.float32)
    masks = np.asarray(avail_mask, dtype=np.bool_)
    actions = np.asarray(action, dtype=np.int64).reshape(-1)
    series = np.asarray(rtg_unscaled, dtype=np.float64).reshape(-1)

    steps = int(actions.size)
    if series.size != steps:
        raise ValueError(
            f"the conditioning series has {series.size} entries and the episode has {steps} "
            "decisions; they index the same steps and must match"
        )
    if states.shape[0] < steps or masks.shape[0] < steps:
        raise ValueError(
            f"observations carry T+1 rows and decisions T (contract C6): got {states.shape[0]} "
            f"state rows and {masks.shape[0]} mask rows for {steps} decisions"
        )

    model = agent._ensure_model(int(states.shape[1]))
    config = agent.config
    if steps > config.max_ep_len:
        raise ValueError(
            f"episode has {steps} decisions but the timestep embedding holds "
            f"{config.max_ep_len}"
        )

    agent.reset_context()
    context = agent._contexts[ix_id]
    span = config.context_length
    windows: list[dict[str, np.ndarray]] = []

    for t in range(steps):
        row = agent._normalise_state(ix_id, states[t])
        mask = masks[t]
        windows.append(agent._window(context, float(series[t]), row, mask, t, config))
        # Advance with the LOGGED action, never a predicted one.
        context.rtg.append(float(series[t]))
        context.state.append(row)
        context.action.append(int(actions[t]))
        context.timestep.append(t)
        context.avail.append(mask)
        context.next_step = t + 1
        for buffer in (
            context.rtg,
            context.state,
            context.action,
            context.timestep,
            context.avail,
        ):
            del buffer[: max(0, len(buffer) - span)]

    def _stack(key: str) -> torch.Tensor:
        return torch.from_numpy(np.stack([window[key] for window in windows])).to(agent.device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(
                _stack("rtg"),
                _stack("state"),
                _stack("action"),
                _stack("timestep"),
                _stack("attention_mask"),
                None,
            )[:, -1]
    finally:
        model.train(was_training)
    return logits.detach().cpu().numpy().astype(np.float32)


def flip_rate(baseline_actions: np.ndarray, other_actions: np.ndarray) -> float:
    """Fraction of decision steps whose greedy action differs from the baseline's."""
    left = np.asarray(baseline_actions).reshape(-1)
    right = np.asarray(other_actions).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(
            f"two action sequences compared on the same states must have the same number of "
            f"decision steps, got {left.shape[0]} and {right.shape[0]}"
        )
    if left.size == 0:
        raise ValueError("an empty action sequence has no flip rate")
    return float(np.count_nonzero(left != right)) / float(left.size)


def total_variation_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Row-wise ``0.5 * sum |p - q|``: 0 for identical rows, 1 for disjoint point masses."""
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"distribution shapes differ: {p.shape} against {q.shape}")
    return 0.5 * np.abs(p - q).sum(axis=-1)


def compare_logits(
    key: str, baseline: np.ndarray, other: np.ndarray, avail_mask: np.ndarray
) -> InterventionComparison:
    """Measure one intervention against the baseline on the same states.

    The mask is applied here and only here, through the frozen ``masked_action_logits``, so the
    greedy action this reports is the one ``DTAgent.act(explore=False)`` would take.
    ``mean_abs_logit_delta`` is computed over **legal actions only**: an illegal column is
    ``-inf`` in both arms and subtracting them would give NaN, not zero.
    """
    base = np.asarray(baseline, dtype=np.float32)
    alt = np.asarray(other, dtype=np.float32)
    mask = np.asarray(avail_mask, dtype=np.bool_)
    if base.shape != alt.shape:
        raise ValueError(f"logit shapes differ: {base.shape} against {alt.shape}")
    if mask.shape != base.shape:
        raise ValueError(
            f"avail_mask shape {mask.shape} does not match the logits {base.shape}"
        )

    masked_base = masked_action_logits(torch.from_numpy(base), torch.from_numpy(mask))
    masked_alt = masked_action_logits(torch.from_numpy(alt), torch.from_numpy(mask))
    rate = flip_rate(
        masked_base.argmax(dim=-1).numpy(), masked_alt.argmax(dim=-1).numpy()
    )
    tvd = float(
        np.mean(
            total_variation_distance(
                torch.softmax(masked_base, dim=-1).numpy(),
                torch.softmax(masked_alt, dim=-1).numpy(),
            )
        )
    )
    # Widened to float64 BEFORE the reduction, and this is not a formality: numpy reduces a
    # float32 array with a float32 accumulator, and this mean runs over ~57,600 entries per cell
    # (20 streams x 360 steps x 8 actions).  Caught by
    # test_compare_logits_reports_the_step_count_and_a_finite_logit_delta, which asked for 1/6 at
    # rel=1e-12 and got the float32 answer 0.1666666716337204.
    legal = np.abs(base.astype(np.float64) - alt.astype(np.float64))[mask]
    delta = float(legal.mean()) if legal.size else 0.0
    return InterventionComparison(
        key=str(key),
        flip_rate=rate,
        tvd=tvd,
        mean_abs_logit_delta=delta,
        n_steps_compared=int(base.shape[0]),
    )


# ----------------------------------------------------------------------
# Corpus access
# ----------------------------------------------------------------------


def checkpoint_path_for(tier: str, seed: int, *, output_root: str | Path) -> Path:
    """Where a (tier, seed) ``dt`` checkpoint lives.  ``mappo1000``'s is P4's, re-used by P4.6."""
    if tier not in _CHECKPOINT_LAYOUT:
        raise ValueError(f"no checkpoint layout declared for tier {tier!r}; {sorted(_CHECKPOINT_LAYOUT)}")
    return Path(output_root) / _CHECKPOINT_LAYOUT[tier].format(seed=int(seed))


def _split_episode_paths(dataset_dirs: Sequence[str | Path], split: str) -> list[Path]:
    """Episode files of *split*, in the order ``TrajectoryWindowDataset`` would load them.

    Directory order, then manifest order within each directory -- **not** ``glob`` order.  The order
    matters: ``_fit_stats`` concatenates per-stream arrays in it, and float64 reductions over a
    different order give a different last bit.
    """
    if split not in DRAW_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(DRAW_SPLITS)}")
    low, high = DRAW_SPLITS[split]
    paths: list[Path] = []
    for directory in dataset_dirs:
        folder = Path(directory)
        manifest = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
        for entry in manifest.get("episodes", []):
            if low <= int(entry["flow_draw"]) <= high:
                paths.append(folder / str(entry["filename"]))
    if not paths:
        raise ValueError(
            f"no episodes in the {split!r} split across {[str(d) for d in dataset_dirs]}"
        )
    return paths


def _suffix_sums(rewards: np.ndarray) -> np.ndarray:
    """``sum(r_k for k >= t)`` as float32, by the reversed-cumsum route.

    ``offline/dataset._returns_to_go`` computes the same quantity with an explicit backward loop and
    its docstring records that the two agree bit for bit; using the other route here keeps this an
    independent recomputation rather than a second call to the same code.
    """
    values = np.asarray(rewards, dtype=np.float64)
    return np.cumsum(values[::-1])[::-1].astype(np.float32)


def committed_rtg_summary(checkpoint_path: str | Path) -> RtgSummary:
    """The ``RtgSummary`` frozen inside a checkpoint's normalisation statistics.

    **This payload is the only committed copy** -- there is no ``stats.json`` anywhere in the repo
    (verified 2026-08-25), which is why the artifact records where it came from.
    """
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    stats = payload.get("stats")
    if not stats or "rtg" not in stats:
        raise ValueError(
            f"{checkpoint_path}: the checkpoint carries no frozen training statistics, so the "
            "committed RTG summary cannot be read from it"
        )
    pairs = [
        (scenario, ix_id, summary)
        for scenario, per_ix in stats["rtg"].items()
        for ix_id, summary in per_ix.items()
    ]
    if len(pairs) != 1:
        raise ValueError(
            f"{checkpoint_path}: the statistics cover {len(pairs)} (scenario, intersection) pairs "
            f"{[(s, i) for s, i, _ in pairs]}; this task summarises one and will not take the first"
        )
    entry = pairs[0][2]
    return RtgSummary(
        count=int(entry["count"]),
        min=float(entry["min"]),
        max=float(entry["max"]),
        mean=float(entry["mean"]),
        std=float(entry["std"]),
        quantiles=tuple((float(q), float(v)) for q, v in entry["quantiles"]),
    )


def recomputed_rtg_summary(
    dataset_dirs: Sequence[str | Path], *, split: str = "train"
) -> RtgSummary:
    """The same summary, recomputed from the raw ``.npz`` by an independent route.

    **Same population and same estimator as ``_fit_stats``** (``offline/dataset.py:702``), which is
    AMENDMENT A6's correction: the committed summary is over the concatenated per-step RTG of every
    stream in the tier's WHOLE training split at ``ddof=0``, not over per-episode returns and not
    over the tier's 200-stream declared subsample.  Comparing it against either of those would
    condemn a correct implementation.
    """
    chunks: list[np.ndarray] = []
    seen: set[tuple[str, str]] = set()
    for path in _split_episode_paths(dataset_dirs, split):
        episode = load_episode(path)
        for ix_id in episode.ix_ids:
            arrays = episode.intersections[ix_id]
            rewards = np.asarray(arrays.local_reward, dtype=np.float32)
            if bool(np.isnan(rewards).any()):
                raise ValueError(f"{path}: intersection {ix_id!r} has NaN local_reward")
            chunks.append(_suffix_sums(rewards))
            seen.add((str(episode.ix_ids[0]), str(ix_id)))
    if len({ix for _, ix in seen}) != 1:
        raise ValueError(
            f"these directories span {len({ix for _, ix in seen})} intersections; this summary is "
            "per (scenario, intersection) and will not pool them"
        )
    values = np.concatenate(chunks).astype(np.float64)
    return RtgSummary(
        count=int(values.size),
        min=float(values.min()),
        max=float(values.max()),
        mean=float(values.mean()),
        std=float(values.std()),
        quantiles=tuple(
            (float(q), float(np.quantile(values, q, method="linear"))) for q in RTG_QUANTILES
        ),
    )


# ----------------------------------------------------------------------
# Table 1: rows A, B and the ramp decomposition
# ----------------------------------------------------------------------


def episode_return_stats(returns: Sequence[float]) -> dict[str, float]:
    """Row A: ``mean, sd, IQR, min, max`` of the per-episode return.  ``ddof=0``, like everything else."""
    values = np.asarray(list(returns), dtype=np.float64)
    if values.size == 0:
        raise ValueError("no returns: the per-episode statistics are undefined")
    low = np.quantile(values, 0.25, method="linear")
    high = np.quantile(values, 0.75, method="linear")
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "iqr": float(high - low),
        "q25": float(low),
        "q75": float(high),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def between_episode_rtg_spread(
    rtg_rows: np.ndarray, *, timesteps: Sequence[int] = RTG_SPREAD_TIMESTEPS
) -> dict[str, Any]:
    """⭐ Row B: the sd of the RTG **across episodes** at a fixed timestep, and pooled.

    This is the quantity C4's identifiability hypothesis is actually about.  The model sees an RTG
    at every step, so what matters is how much that value varies **between** episodes at a given
    point -- not how much it varies **within** one episode, which is the deterministic ramp from the
    target down to zero and is the same shape in every episode of every tier.
    """
    rows = np.asarray(rtg_rows, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"expected a (n_episodes, T) matrix, got shape {rows.shape}")
    episodes, length = rows.shape
    if episodes < 2:
        raise ValueError(
            f"a between-episode spread needs at least two episodes, got {episodes}"
        )
    wanted = [int(t) for t in timesteps]
    outside = [t for t in wanted if not 0 <= t < length]
    if outside:
        raise ValueError(
            f"timestep {outside[0]} is outside the episodes, which run 0..{length - 1}; a spread "
            "at a step the corpus never reached would be an invented number"
        )
    per = {str(t): float(rows[:, t].std()) for t in wanted}
    return {
        "n_episodes": int(episodes),
        "episode_length": int(length),
        "per_timestep": per,
        "pooled": float(np.mean([per[str(t)] for t in wanted])),
    }


def ramp_prediction(target_rtg: float, rtg_scale: float) -> float:
    """Scaled sd of a PERFECTLY deterministic RTG decaying from *target_rtg* to 0.

    A linear decay over the episode is uniform on ``[0, |target|]``, whose standard deviation is
    ``|target| / (2 sqrt 3)``.  Dividing by ``rtg_scale`` puts it in the units the network receives
    (``agent/DTAgent.py:596``).  Reported beside the measured marginal spread so a reader can see how
    much of that spread is ramp and how much is information: on the five single-policy tiers the
    ratio is 1.08-1.54, on the three mixtures it is 7.60-7.65.
    """
    scale = float(rtg_scale)
    if scale == 0.0:
        raise ValueError("rtg_scale must be non-zero; it divides the RTG input")
    return abs(float(target_rtg)) / (2.0 * math.sqrt(3.0)) / scale


# ----------------------------------------------------------------------
# Table 2: delta, MEASURED and never a rule (AMENDMENT A7)
# ----------------------------------------------------------------------


def behaviour_margin_degenerate(entry: Mapping[str, Any]) -> bool:
    """Does the paired CI contain zero?  Closed interval, because an endpoint at 0 is degenerate too."""
    return float(entry["ci95_low"]) <= 0.0 <= float(entry["ci95_high"])


def delta_table(grids: Sequence[Mapping[str, Any]], tiers: Sequence[str]) -> dict[str, Any]:
    """Per tier: the DT's paired margin over its behaviour reference, read from committed grids.

    ⛔ **This is a measured table and NOT a decision rule.**  AMENDMENT A7 withdrew the rule after
    measuring that delta spans eleven orders of magnitude across these eight tiers, is a *deficit*
    on ``maxpressure`` (the DT loses to its own behaviour policy) and is ``-5.68e-16`` on
    ``fixedtime``.  A margin that cannot return an answer on part of its domain is not a margin.
    ``behaviour_margin_degenerate`` marks the two tiers where the CI contains zero.
    """
    found: dict[str, Mapping[str, Any]] = {}
    for grid in grids:
        for comparison in grid.get("behaviour_comparisons", []):
            arm = str(comparison.get("left_arm", ""))
            if not arm.startswith("dt@"):
                continue
            found.setdefault(arm[len("dt@") :], comparison)

    missing = [tier for tier in tiers if tier not in found]
    if missing:
        raise ValueError(
            f"no dt behaviour comparison for tier(s) {missing} in the committed grids; a tier "
            "absent from both p4_6_grid.json and p4_7_grid.json has no measured margin and must "
            "not appear as an empty row"
        )

    table: dict[str, Any] = {}
    for tier in tiers:
        entry = found[tier]
        table[tier] = {
            "left_arm": str(entry["left_arm"]),
            "right_arm": str(entry["right_arm"]),
            "mean_difference": float(entry["mean_difference"]),
            "median_difference": float(entry["median_difference"]),
            "ci95_low": float(entry["ci95_low"]),
            "ci95_high": float(entry["ci95_high"]),
            "ci95_half_width": float(entry["ci95_half_width"]),
            "ci95_width": float(entry["ci95_width"]),
            "rank_biserial": float(entry["rank_biserial"]),
            "wins": int(entry["wins"]),
            "losses": int(entry["losses"]),
            "ties": int(entry["ties"]),
            "n_shared_draws": int(entry["n_shared_draws"]),
            "wilcoxon": dict(entry["wilcoxon"]),
            "behaviour_margin_degenerate": behaviour_margin_degenerate(entry),
        }
    return table


# ----------------------------------------------------------------------
# The campaign
# ----------------------------------------------------------------------


def _tier_streams(tier: str, corpus_root: str | Path) -> tuple[StreamReturn, ...]:
    """A tier's declared 200-stream training set, through the public campaign helper."""
    spec = tier_spec(tier)
    dataset = build_training_dataset(tier_dirs(spec, corpus_root), CONTEXT_LENGTH)
    if spec.subsample != "mixture":
        return training_streams(spec, dataset)
    components = {
        name: training_streams(
            tier_spec(name),
            build_training_dataset(tier_dirs(tier_spec(name), corpus_root), CONTEXT_LENGTH),
        )
        for name in spec.components
    }
    return training_streams(spec, dataset, component_streams=components)


def probe_cell(
    tier: str,
    seed: int,
    *,
    checkpoint_path: str | Path,
    corpus_root: str | Path,
    device: str | None = None,
    streams: Sequence[StreamReturn] | None = None,
) -> ProbeCell:
    """Measure every declared intervention on one (tier, seed) cell.

    The agent is built through ``agent_with_target`` at the checkpoint's **own** recorded target, so
    the baseline arm is exactly the configuration P4.6 evaluated; the interventions then vary only
    the series handed to :func:`teacher_forced_logits`, never the weights and never the states.
    """
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    target = float(payload["target_rtg"])
    scale = float(payload["rtg_scale"])

    spec = tier_spec(tier)
    chosen = list(streams) if streams is not None else list(_tier_streams(tier, corpus_root))
    picked = [chosen[i] for i in selected_stream_indices(len(chosen))]

    settings = env_settings_for_tiers([spec], corpus_root)
    from experiments.config import EnvSpec
    from experiments.envs import make_env
    from offline.materialise_draws import draw_config_path

    env = make_env(
        EnvSpec(
            id="cityflow1x1",
            backend="cityflow",
            paths={"config": str(draw_config_path("cityflow1x1", int(HELD_OUT_DRAWS[0])))},
            settings=settings,
        )
    )
    try:
        agent = agent_with_target(
            env,
            path,
            declared_gradient_steps=DECLARED_GRADIENT_STEPS,
            target_rtg=target,
            device=device,
        )
        interventions = declared_interventions(target)
        totals: dict[str, list[np.ndarray]] = {i.key: [] for i in interventions}
        masks: list[np.ndarray] = []

        for stream in picked:
            episode = load_episode(Path(stream.dataset_dir) / stream.episode_file)
            arrays = episode.intersections[stream.ix_id]
            rewards = np.asarray(arrays.local_reward, dtype=np.float32)
            state = np.asarray(arrays.state, dtype=np.float32)
            avail = np.asarray(arrays.avail_mask, dtype=np.bool_)
            action = np.asarray(arrays.action, dtype=np.int64)
            masks.append(avail[: action.size])
            for intervention in interventions:
                series = conditioning_series(
                    intervention.target_rtg, rewards, kind=intervention.kind
                )
                totals[intervention.key].append(
                    teacher_forced_logits(
                        agent,
                        stream.ix_id,
                        state=state,
                        avail_mask=avail,
                        action=action,
                        rtg_unscaled=series,
                    )
                )
    finally:
        env.close()

    mask = np.concatenate(masks)
    baseline = np.concatenate(totals["baseline"])
    comparisons = tuple(
        compare_logits(i.key, baseline, np.concatenate(totals[i.key]), mask)
        for i in interventions
    )
    return ProbeCell(
        tier=tier,
        seed=int(seed),
        checkpoint=str(path),
        target_rtg=target,
        rtg_scale=scale,
        n_streams=len(picked),
        n_steps=int(baseline.shape[0]),
        comparisons=comparisons,
    )


def spread_table(
    corpus_root: str | Path,
    *,
    output_root: str | Path,
    tiers: Sequence[str] = PROBE_TIERS,
    seed: int = 101,
) -> dict[str, Any]:
    """Table 1 for every tier: row A, row B and row C, in raw and scaled units."""
    table: dict[str, Any] = {}
    for tier in tiers:
        spec = tier_spec(tier)
        checkpoint = checkpoint_path_for(tier, seed, output_root=output_root)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        scale = float(payload["rtg_scale"])
        target = float(payload["target_rtg"])

        picked = _tier_streams(tier, corpus_root)
        rows: list[np.ndarray] = []
        returns: list[float] = []
        for stream in picked:
            episode = load_episode(Path(stream.dataset_dir) / stream.episode_file)
            rewards = np.asarray(
                episode.intersections[stream.ix_id].local_reward, dtype=np.float32
            )
            suffix = _suffix_sums(rewards)
            rows.append(suffix.astype(np.float64))
            returns.append(float(suffix[0]))
        matrix = np.stack(rows)

        raw_b = between_episode_rtg_spread(matrix)
        scaled_b = between_episode_rtg_spread(matrix / scale)
        committed = committed_rtg_summary(checkpoint)
        recomputed = recomputed_rtg_summary([str(d) for d in tier_dirs(spec, corpus_root)])
        if recomputed != committed:
            raise ValueError(
                f"{tier}: the recomputed RtgSummary does not match the committed one; the two "
                "routes disagree and no spread row may be reported until they do"
            )

        ramp = ramp_prediction(target, scale)
        table[tier] = {
            "target_rtg": target,
            "rtg_scale": scale,
            "training_streams": len(picked),
            "episode_return_raw": episode_return_stats(returns),
            "episode_return_scaled": episode_return_stats([r / scale for r in returns]),
            "between_episode_rtg_raw": raw_b,
            "between_episode_rtg_scaled": scaled_b,
            "ramp_prediction_scaled": ramp,
            "pooled_scaled_over_ramp": scaled_b["pooled"] / ramp,
            "marginal_std_scaled": committed.std / scale,
            "marginal_std_over_ramp": (committed.std / scale) / ramp,
            "rtg_summary_committed": _summary_json(committed),
            "rtg_summary_recomputed": _summary_json(recomputed),
            "rtg_summary_routes_agree": True,
            "rtg_summary_population": (
                "concatenated per-step RTG of every stream in the tier's whole training split "
                f"over {[Path(d).name for d in tier_dirs(spec, corpus_root)]}, ddof=0 -- NOT the "
                "declared 200-stream training set, and for the mixture tiers not the mixture"
            ),
            "rtg_summary_source": (
                "the NormalizationStats payload inside the checkpoint; there is no other "
                "committed copy in the repository"
            ),
        }
    return table


def _summary_json(summary: RtgSummary) -> dict[str, Any]:
    return {
        "count": int(summary.count),
        "min": float(summary.min),
        "max": float(summary.max),
        "mean": float(summary.mean),
        "std": float(summary.std),
        "quantiles": [[float(q), float(v)] for q, v in summary.quantiles],
    }


def crosscheck_targets() -> tuple[float, float]:
    """``DECLARED_GRID``'s endpoints -- the widest pair P4.3 registered."""
    return float(DECLARED_GRID[0]), float(DECLARED_GRID[-1])


def crosscheck(
    corpus_root: str | Path, *, output_root: str | Path, device: str | None = None
) -> dict[str, Any]:
    """The second route: two LIVE rollouts of one cell under the grid's two endpoints.

    The probe measures behaviour-policy states.  This converts that into "and also on the model's
    own states, for one cell" -- CLAUDE.md section 2's *critical quantities get computed twice*
    applied to the one number this task exists to produce.
    """
    from experiments.config import EnvSpec
    from experiments.envs import make_env
    from offline.horizon_metric import horizon_rollout
    from offline.materialise_draws import draw_config_path

    spec = tier_spec(_CROSSCHECK_TIER)
    settings = env_settings_for_tiers([spec], corpus_root)
    checkpoint = checkpoint_path_for(_CROSSCHECK_TIER, _CROSSCHECK_SEED, output_root=output_root)
    low, high = crosscheck_targets()

    runs: dict[str, Any] = {}
    sequences: dict[str, np.ndarray] = {}
    for label, target in (("target_first", low), ("target_last", high)):
        env = make_env(
            EnvSpec(
                id="cityflow1x1",
                backend="cityflow",
                paths={"config": str(draw_config_path("cityflow1x1", _CROSSCHECK_DRAW))},
                settings=settings,
            )
        )
        try:
            agent = agent_with_target(
                env,
                checkpoint,
                declared_gradient_steps=DECLARED_GRADIENT_STEPS,
                target_rtg=float(target),
                device=device,
            )
            taken: list[int] = []

            def choose(_env: Any, info: dict[str, Any], _agent: Any = agent, _taken: list[int] = taken) -> np.ndarray:
                action = _agent.act(info, explore=False, update_memory=True)
                _taken.append(int(action[0]))
                return action

            rollout = horizon_rollout(env, choose, episodes=1, seed=int(HELD_OUT_DRAWS[0]))
        finally:
            env.close()
        sequences[label] = np.asarray(taken, dtype=np.int64)
        runs[label] = {
            "target_rtg": float(target),
            "att_horizon": float(rollout.per_episode_horizon[0]),
            "decisions": int(sequences[label].size),
        }

    first, last = sequences["target_first"], sequences["target_last"]
    compared = int(min(first.size, last.size))
    return {
        "tier": _CROSSCHECK_TIER,
        "seed": _CROSSCHECK_SEED,
        "draw_id": _CROSSCHECK_DRAW,
        "checkpoint": str(checkpoint),
        "runs": runs,
        "n_decisions_compared": compared,
        "action_flip_rate": flip_rate(first[:compared], last[:compared]),
        "att_difference": runs["target_last"]["att_horizon"] - runs["target_first"]["att_horizon"],
        "route": (
            "live rollout on the model's OWN states, against the teacher-forced probe's "
            "behaviour-policy states"
        ),
    }


def assert_declared_interventions(payload: Any) -> None:
    """Refuse a result set whose intervention keys are not exactly the twelve declared ones."""
    keys = set(payload)
    undeclared = sorted(keys - set(INTERVENTION_KEYS))
    if undeclared:
        raise ValueError(
            f"undeclared intervention key(s) {undeclared}; the twelve were registered in "
            "docs/plans/p5.3a.md before any number existed and the grid may not grow after the fact"
        )
    missing = sorted(set(INTERVENTION_KEYS) - keys)
    if missing:
        raise ValueError(
            f"missing declared intervention(s) {missing}; a partial result set would report a "
            "narrower sweep than the one registered"
        )


def report_artifact(
    *,
    cells: Sequence[ProbeCell],
    tables: Mapping[str, Any],
    crosscheck: Mapping[str, Any],
    timings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the one committed artifact.  It carries measured quantities and NO verdict."""
    for cell in cells:
        assert_declared_interventions({c.key: c for c in cell.comparisons})

    by_tier: dict[str, Any] = {}
    for cell in cells:
        by_tier.setdefault(cell.tier, {})[str(cell.seed)] = cell.to_json_obj()

    payload: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "P5.3a: teacher-forced sensitivity of the trained DT's ACTIONS to the return prompt, "
            "plus the two Gate-0 tables P5.3b's registration needs. Measured quantities only; the "
            "reading belongs to the coordinator and lives in docs/returns/P5.3a.md"
        ),
        "declared": {
            "interventions": list(INTERVENTION_KEYS),
            "grid": [float(v) for v in DECLARED_GRID],
            "tiers": list(PROBE_TIERS),
            "seeds": [int(s) for s in PROBE_SEEDS],
            "streams_per_cell": PROBE_STREAM_COUNT,
            "stream_stride": PROBE_STREAM_STRIDE,
            "headline_statistic": "flip_rate",
            "rtg_spread_timesteps": list(RTG_SPREAD_TIMESTEPS),
        },
        "probe": {
            "cells": by_tier,
            "limitation": _LIMITATION,
            "n_cells": len(cells),
        },
        "tables": dict(tables),
        "crosscheck": dict(crosscheck),
        "timings": dict(timings or {}),
        "runtime": runtime_provenance(measurement_commits([])),
    }
    assert_no_verdicts(payload)
    return payload


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """CLI: ``probe``, ``tables``, ``crosscheck``, ``report``."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.rtg_ablation",
        description="P5.3a: teacher-forced RTG sensitivity of the committed dt checkpoints.",
    )
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--work-dir", default="output/p5_3a")
    parser.add_argument("--out-dir", default="docs/data")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="measure one tier's five seeds")
    probe.add_argument("--tier", required=True, choices=list(PROBE_TIERS))

    sub.add_parser("tables", help="the two Gate-0 tables")
    sub.add_parser("crosscheck", help="two live rollouts under the grid's endpoints")
    sub.add_parser("report", help="assemble the artifact from the work directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    from offline.offline_baselines import pin_torch_threads

    args = build_parser().parse_args(argv)
    pin_torch_threads(args.torch_threads)
    work = Path(args.work_dir)
    out_dir = Path(args.out_dir)

    if args.command == "probe":
        streams = _tier_streams(args.tier, args.corpus_root)
        cells = []
        timings = {}
        for seed in PROBE_SEEDS:
            started = time.monotonic()
            cell = probe_cell(
                args.tier,
                int(seed),
                checkpoint_path=checkpoint_path_for(
                    args.tier, int(seed), output_root=args.output_root
                ),
                corpus_root=args.corpus_root,
                device=args.device,
                streams=streams,
            )
            elapsed = time.monotonic() - started
            timings[f"{args.tier}@{seed}"] = elapsed
            cells.append(cell.to_json_obj())
            print(
                f"{arm_key('dt', args.tier)} seed {seed}: "
                f"flip_rate zero={cell.comparisons[-2].flip_rate:.6f} "
                f"frozen={cell.comparisons[-1].flip_rate:.6f} "
                f"n={cell.n_steps} in {elapsed:.1f}s",
                flush=True,
            )
        work.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            {
                "format_version": ARTIFACT_FORMAT_VERSION,
                "tier": args.tier,
                "cells": cells,
                "timings_seconds": timings,
                "runtime": runtime_provenance(),
            },
            work / f"probe_{args.tier}.json",
        )
        return 0

    if args.command == "tables":
        payload = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "spread": spread_table(args.corpus_root, output_root=args.output_root),
            "delta": delta_table(
                [
                    json.loads((out_dir / name).read_text(encoding="utf-8"))
                    for name in ("p4_6_grid.json", "p4_7_grid.json")
                ],
                PROBE_TIERS,
            ),
            "runtime": runtime_provenance(),
        }
        work.mkdir(parents=True, exist_ok=True)
        write_json_atomic(payload, work / "tables.json")
        return 0

    if args.command == "crosscheck":
        payload = crosscheck(args.corpus_root, output_root=args.output_root, device=args.device)
        work.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            {"format_version": ARTIFACT_FORMAT_VERSION, **payload, "runtime": runtime_provenance()},
            work / "crosscheck.json",
        )
        return 0

    cells: list[ProbeCell] = []
    timings: dict[str, Any] = {}
    for tier in PROBE_TIERS:
        path = work / f"probe_{tier}.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path}: run `probe --tier {tier}` first")
        chunk = json.loads(path.read_text(encoding="utf-8"))
        timings.update(chunk.get("timings_seconds", {}))
        for entry in chunk["cells"]:
            cells.append(
                ProbeCell(
                    tier=entry["tier"],
                    seed=int(entry["seed"]),
                    checkpoint=entry["checkpoint"],
                    target_rtg=float(entry["target_rtg"]),
                    rtg_scale=float(entry["rtg_scale"]),
                    n_streams=int(entry["n_streams"]),
                    n_steps=int(entry["n_steps"]),
                    comparisons=tuple(
                        InterventionComparison(
                            key=key,
                            flip_rate=float(value["flip_rate"]),
                            tvd=float(value["tvd"]),
                            mean_abs_logit_delta=float(value["mean_abs_logit_delta"]),
                            n_steps_compared=int(value["n_steps_compared"]),
                        )
                        for key, value in entry["interventions"].items()
                    ),
                )
            )
    tables = json.loads((work / "tables.json").read_text(encoding="utf-8"))
    checked = json.loads((work / "crosscheck.json").read_text(encoding="utf-8"))
    payload = report_artifact(
        cells=cells,
        tables={"spread": tables["spread"], "delta": tables["delta"]},
        crosscheck=checked,
        timings={"probe_seconds": timings},
    )
    write_json_atomic(payload, out_dir / "p5_3a_rtg_probe.json")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
