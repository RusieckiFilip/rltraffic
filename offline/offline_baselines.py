"""P4.4: train BC, %BC and IQL on the P4 corpus and compare them against the DT, paired.

Artifact format version: ``p4.4-baselines/1.0``.

WHAT THIS TASK DECIDES
----------------------
P4's Decision Transformer beat the policy whose data it trained on by **0.6263 ATT**.  Until a
behaviour-cloning model is measured on the same corpus, the same draws and the same protocol,
that margin cannot be attributed to anything.  ``PROJECT_PLAN.md`` section 1 has said since
2026-07-10: *"if BC matches MADT, sequence modeling adds nothing -- must be tested."*

THE PROTOCOL IS P4'S, REUSED RATHER THAN RESTATED
-------------------------------------------------
Every rollout goes through ``offline.dt_gate.evaluate_arm`` with env settings read from the
collection manifest by ``env_settings_from_manifest``; the cells are built by ``dt_gate._cell``,
the pairing by ``dt_gate._paired``, the descriptives by ``mean_ci95`` and the paired test by
``wilcoxon_signed_rank``.  Importing them -- including the two private helpers -- is deliberate:
a second implementation of the same protocol is exactly how two arms stop being comparable, and
these are the functions that produced the numbers this task compares against.

**Gate A** re-rolls all 1100 cited P4 episodes (MADT, MAPPO@1000, MaxPressure) on this
instrument and refuses to continue unless every one reproduces exactly.  A difference would mean
P4.4's numbers were measured on a different instrument than P4's, and no comparison between them
would mean anything.

THE DECISION RULE (``PREREGISTRATION.md`` A6, declared before any baseline existed)
-----------------------------------------------------------------------------------
``delta = 0.6263 ATT`` -- the DT's own paired margin over MAPPO@1000 on the 100 held-out draws.
With ``d = ATT_DT - ATT_baseline`` per shared draw and lower ATT better:

* CI(d) entirely within ``[-delta, +delta]``          -> the baseline **matches** the DT;
* CI(d) entirely below ``-delta``                     -> the DT is **genuinely better**;
* CI(d) entirely above ``+delta``                     -> the **baseline** is genuinely better --
  arithmetically possible, **not named by A6**, and reported separately rather than folded into
  "inconclusive", because it is a decisive result in the other direction;
* anything else                                       -> **inconclusive at this power**, with
  the CI width, because "contains 0" is a failure to reject and not a demonstration of
  equivalence.

**The recovered fraction ``(MAPPO@1000 - arm) / (MAPPO@1000 - DT)`` is reported unconditionally
beside every verdict, in every branch** (A6's clarification of 2026-08-11): delta equals the
effect under study, so a baseline landing 0.5 ATT worse than the DT sits inside the margin and
returns *matches* while having recovered only 20.2 % of the DT's margin over its own behaviour
policy.

WHAT A DIFFERENCE AGAINST THESE BASELINES DOES NOT SAY
-------------------------------------------------------
DT minus BC is a **combined** difference -- attention/context **plus** return-to-go conditioning
**plus** the timestep embedding.  It is never reported as "sequence modelling adds X"; P4.3 (RTG)
and P5.3 (no-RTG, context length) own the decomposition.  And IQL here is **untuned**, running
published D4RL-locomotion values transplanted onto a discrete 8-phase action space: a losing
untuned IQL cannot support any claim of MADT superiority over IQL, which is stated in the same
sentence as its number rather than in a caveats section.

TRUNCATION, AND WHY THERE IS NO ``done`` IN THIS FILE
------------------------------------------------------
``terminated`` is hardcoded ``False`` on this platform (``envs/base_traffic_env.py``) and every
episode ends by time-limit truncation, so the value learner **bootstraps through the boundary**:
the last transition's target is ``r + gamma * V(s_T)``, never ``r``.  Treating a timeout as
terminal causes systematic value underestimation near episode end and would hand the DT an
unearned win over its own baselines (``PREREGISTRATION.md`` section 7; Decisions Log 2026-07-26).

That is also why this module reads episodes through
``offline.trajectory_logger.load_episode``: ``TrajectoryWindowDataset`` structurally never yields
observation row ``T`` -- "observation row T is never an input" -- and row ``T`` is precisely the
bootstrap target of the final transition.  ``load_episode`` is the format's **single sanctioned
reader**, the one ``offline/dataset.py`` itself calls, so this is not a second reader of the
on-disk format; episode selection, the draw-split leakage guard and the normalisation statistics
all stay in ``TrajectoryWindowDataset``, and a test cross-checks every ``(s_t, a_t)`` against it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from agent.OfflineBaselines import (
    BCAgent,
    IQLAgent,
    MLPTrunk,
    TrunkConfig,
    action_loss,
    canonical_state_dict_digest,
    masked_action_logits,
)
from offline.dataset import NormalizationStats, TrajectoryWindowDataset
from offline.dt_gate import (
    GRAD_CLIP,
    HELD_OUT_DRAWS,
    LEARNING_RATE,
    PLATEAU_TOLERANCE,
    PLATEAU_WINDOW,
    TRAINING_SEEDS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    EpisodeResult,
    WilcoxonResult,
    _cell,
    _paired,
    _sha256_file,
    build_training_dataset,
    env_settings_from_manifest,
    evaluate_arm,
    mean_ci95,
    runtime_provenance,
    stack_dataset,
    wilcoxon_signed_rank,
    window_means,
    plateau_reached,
    write_json_atomic,
)
from offline.trajectory_logger import load_episode

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "DECLARED_GRADIENT_STEPS",
    "DELTA_ATT",
    "DELTA_ATT_DERIVATION",
    "IQL_BATCH_TRANSITIONS",
    "METHODS",
    "PairedComparison",
    "StreamReturn",
    "TrainRecord",
    "TransitionTable",
    "assert_campaign_complete",
    "baselines_artifact",
    "build_transitions",
    "equivalence_verdict",
    "filter_stacked_to_streams",
    "iql_reward_scale",
    "iql_targets",
    "main",
    "paired_comparison",
    "rank_biserial",
    "recovered_fraction",
    "stream_returns",
    "top_return_streams",
    "train_bc",
    "train_iql",
]

ARTIFACT_FORMAT_VERSION = "p4.4-baselines/1.0"

#: Declared in ``docs/plans/p4.4.md`` section 3.1 before the first gradient step: the DT's own
#: REPORTED budget, which is what section 6.3's "same tuning budget" matches.  The DT has spent
#: its one pre-declared raise, so no raise is available to this task.
DECLARED_GRADIENT_STEPS = 40_000

#: ``PREREGISTRATION.md`` amendment A6's declared literal, and the full-precision quantity it was
#: rounded from (105.58203462874322 - 104.95575898180847, from the committed ``p4_gate.json``).
#: Both are recorded because the multiplier of 1.0 on the DT's margin is a CHOICE, as A6's own
#: clarification says, and a verdict must never turn on the rounding: the artifact recomputes
#: every verdict under both values and refuses to report if they disagree.
DELTA_ATT = 0.6263
DELTA_ATT_DERIVATION = 0.6262756469347437

BC_BATCH_WINDOWS = 64
IQL_BATCH_TRANSITIONS = 1_280
TOP_RETURN_FRACTION = 0.10

#: Published IQL, D4RL locomotion.  Unswept -- see ``docs/plans/p4.4.md`` section 3.5 for the
#: decision and its reason (the authorised selection criterion cannot rank either parameter).
IQL_TAU = 0.7
IQL_BETA = 3.0
IQL_GAMMA = 0.99
IQL_POLYAK = 0.005
IQL_WEIGHT_CLIP = 100.0

METHODS: tuple[str, ...] = ("bc", "bc_top10", "iql")

#: The arms cited from P4 and re-rolled by Gate A before anything else is measured.
CITED_ARMS: tuple[str, ...] = ("madt", "mappo1000", "maxpressure")

VERDICT_MATCHES = "matches"
VERDICT_DT_BETTER = "dt_genuinely_better"
VERDICT_BASELINE_BETTER = "baseline_genuinely_better"
VERDICT_INCONCLUSIVE = "inconclusive_at_this_power"

#: How many rows the post-training diagnostics pass reads.  Deterministic and capped, so the
#: diagnostic costs the same on the fixture and on the 72,000-window tier.
DIAGNOSTIC_ROWS = 20_000


@dataclass(frozen=True)
class StreamReturn:
    """One (episode, intersection) stream and its undiscounted return.

    ``total_return`` is the stream's own ``rtg[t=0]``, i.e. ``sum(local_reward)`` -- the same
    quantity the ladder calls an episode return, per intersection.
    """

    dataset_dir: str
    episode_file: str
    ix_id: str
    ix_index: int
    episode_index: int
    flow_draw: int
    group: tuple[int, int]
    total_return: float

    @property
    def key(self) -> tuple[str, str, str]:
        """Identity of the stream: directory, episode file, intersection id."""
        return (self.dataset_dir, self.episode_file, self.ix_id)


@dataclass(frozen=True)
class TransitionTable:
    """Flat ``(s, a, r, s')`` transitions.

    **There is no ``done`` field, and that is the point.**  Every episode on this platform ends
    by time-limit truncation, so every transition bootstraps -- including the last one of each
    stream, whose ``next_state`` is observation row ``T``.

    ``state`` and ``next_state`` are normalised with the frozen training-split statistics, the
    same ones the loader applied to its windows.
    """

    state: torch.Tensor
    next_state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    stream_index: torch.Tensor
    t: torch.Tensor
    reward_scale: float

    def __len__(self) -> int:
        return int(self.state.shape[0])

    def select(self, index: torch.Tensor) -> TransitionTable:
        """A sub-table of the given rows, so a batch is still a table.

        The training loop takes its targets through :func:`iql_targets` on one of these rather
        than inlining ``r + gamma * V(s')``, so the test that pins the bootstrap guards the path
        training actually takes.
        """
        return TransitionTable(
            state=self.state[index],
            next_state=self.next_state[index],
            action=self.action[index],
            reward=self.reward[index],
            stream_index=self.stream_index[index],
            t=self.t[index],
            reward_scale=self.reward_scale,
        )

    def to(self, device: torch.device) -> TransitionTable:
        """The same table on *device*."""
        return TransitionTable(
            state=self.state.to(device),
            next_state=self.next_state.to(device),
            action=self.action.to(device),
            reward=self.reward.to(device),
            stream_index=self.stream_index.to(device),
            t=self.t.to(device),
            reward_scale=self.reward_scale,
        )


@dataclass(frozen=True)
class TrainRecord:
    """One trained (method, seed): the reported checkpoint and its provenance.

    ``canonical_digest`` is the identity of the weights; ``file_sha256`` is the identity of the
    file.  They answer different questions and both are kept (``DEFERRED`` 29).
    """

    method: str
    seed: int
    gradient_steps: int
    declared_gradient_steps: int
    losses: tuple[float, ...]
    window_means: tuple[float, ...]
    plateaued: bool
    checkpoint_path: str
    canonical_digest: str
    file_sha256: str
    seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairedComparison:
    """A paired per-draw comparison with the effect sizes section 8 makes mandatory.

    ``mean_difference`` is ``mean(left - right)`` over the shared draws.  ``wins`` counts draws
    where the **left** arm had the lower (better) ATT.
    """

    left_arm: str
    right_arm: str
    n_shared_draws: int
    draw_ids: tuple[int, ...]
    mean_left: float
    mean_right: float
    mean_difference: float
    ci95_half_width: float
    ci95_width: float
    ci95_low: float
    ci95_high: float
    median_difference: float
    wins: int
    losses: int
    ties: int
    rank_biserial: float
    wilcoxon: WilcoxonResult

    def to_json_obj(self) -> dict[str, Any]:
        """JSON-ready mapping, including the Wilcoxon result as a nested object."""
        return {
            "left_arm": self.left_arm,
            "right_arm": self.right_arm,
            "n_shared_draws": self.n_shared_draws,
            "draw_ids": list(self.draw_ids),
            "mean_left": self.mean_left,
            "mean_right": self.mean_right,
            "mean_difference": self.mean_difference,
            "ci95_half_width": self.ci95_half_width,
            "ci95_width": self.ci95_width,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "median_difference": self.median_difference,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "rank_biserial": self.rank_biserial,
            "wilcoxon": {
                "w_plus": self.wilcoxon.w_plus,
                "w_minus": self.wilcoxon.w_minus,
                "statistic": self.wilcoxon.statistic,
                "n_used": self.wilcoxon.n_used,
                "n_zero": self.wilcoxon.n_zero,
                "z": self.wilcoxon.z,
                "p_value": self.wilcoxon.p_value,
            },
        }


# ----------------------------------------------------------------------
# The data path: stream returns, the %BC filter, and IQL's transitions
# ----------------------------------------------------------------------


def stream_returns(dataset: TrajectoryWindowDataset) -> tuple[StreamReturn, ...]:
    """Undiscounted return of every (episode, intersection) stream, from the loader alone.

    Read as the item at ``t = 0``, whose returns-to-go entry is ``sum(r_k for k >= 0)``.  The
    loader is the single definition of that quantity; a test recomputes it from the raw reward
    arrays by a second route.
    """
    out: list[StreamReturn] = []
    for index in range(len(dataset)):
        meta = dataset.item_meta(index)
        if meta.t != 0:
            continue
        item = dataset[index]
        out.append(
            StreamReturn(
                dataset_dir=meta.dataset_dir,
                episode_file=meta.episode_file,
                ix_id=meta.ix_id,
                ix_index=meta.ix_index,
                episode_index=meta.episode_index,
                flow_draw=meta.flow_draw,
                group=(int(item["state"].shape[-1]), int(item["avail_mask"].shape[-1])),
                total_return=float(item["rtg"][-1, 0]),
            )
        )
    return tuple(out)


def top_return_streams(
    dataset: TrajectoryWindowDataset, fraction: float = TOP_RETURN_FRACTION
) -> tuple[StreamReturn, ...]:
    """The top *fraction* of streams by return -- %BC's filter, declared and deterministic.

    ``ceil(fraction * n)`` streams, never fewer than one, ordered by descending return with ties
    broken by ``(dataset_dir, episode_file, ix_id)`` so the selection does not depend on load
    order.  On the P4 tier the cut is 20 of 200 streams at -6419.0, with the next stream at
    -6449.0: measured, and not a tie.
    """
    value = float(fraction)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction!r}")
    streams = stream_returns(dataset)
    if not streams:
        raise ValueError("this dataset has no streams to filter")
    keep = max(1, math.ceil(value * len(streams)))
    ordered = sorted(
        streams, key=lambda s: (-s.total_return, s.dataset_dir, s.episode_file, s.ix_id)
    )
    return tuple(ordered[:keep])


def filter_stacked_to_streams(
    dataset: TrajectoryWindowDataset,
    stacked: dict[str, torch.Tensor],
    streams: Sequence[StreamReturn],
) -> dict[str, torch.Tensor]:
    """Restrict stacked windows to those belonging to *streams*, keeping ``item_index``.

    Provenance survives the filter: row ``r`` of the result is still
    ``dataset[item_index[r]]``, so a filtered row can be traced back to its episode file.
    """
    if "item_index" not in stacked:
        raise ValueError(
            "the stacked mapping carries no item_index, so its rows cannot be traced back to "
            "streams; build it with offline.dt_gate.stack_dataset"
        )
    wanted = {stream.key for stream in streams}
    if not wanted:
        raise ValueError("no streams were given: a filter that keeps nothing is not a filter")
    keep = torch.tensor(
        [
            (
                dataset.item_meta(int(index)).dataset_dir,
                dataset.item_meta(int(index)).episode_file,
                dataset.item_meta(int(index)).ix_id,
            )
            in wanted
            for index in stacked["item_index"]
        ],
        dtype=torch.bool,
    )
    if not bool(keep.any()):
        raise ValueError(
            "the filter kept no windows: the requested streams are not present in this stack, "
            "which usually means the streams and the stack belong to different groups"
        )
    return {name: value[keep] for name, value in stacked.items()}


def iql_reward_scale(returns: Sequence[float]) -> float:
    """IQL's published locomotion normalisation: ``1000 / (max_return - min_return)``.

    Computed on the **training split only**, like every other statistic here.  It is the one
    data-dependent quantity in the IQL configuration; the rest are published constants applied
    unchanged.
    """
    values = np.asarray(list(returns), dtype=np.float64)
    if values.size == 0:
        raise ValueError("iql_reward_scale received no returns")
    span = float(values.max() - values.min())
    if span <= 0.0:
        raise ValueError(
            "the returns are constant, so IQL's reward normalisation is undefined "
            f"(min == max == {float(values.min())})"
        )
    return 1000.0 / span


def _resolve_group(
    dataset: TrajectoryWindowDataset, group: tuple[int, int] | None
) -> tuple[int, int]:
    groups = dataset.groups
    if group is None:
        if len(groups) != 1:
            raise ValueError(
                f"this dataset has {len(groups)} (state_dim, n_actions) groups "
                f"{sorted(groups)}; pass group= to say which one to use. C6 forbids padding "
                "across intersections, so each group needs its own tensors"
            )
        return next(iter(groups))
    key = (int(group[0]), int(group[1]))
    if key not in groups:
        raise ValueError(f"group {key} is not present; this dataset has {sorted(groups)}")
    return key


def build_transitions(
    dataset: TrajectoryWindowDataset,
    *,
    group: tuple[int, int] | None = None,
    reward_scale: float = 1.0,
) -> TransitionTable:
    """Flat transitions for one group, including the final one of every stream.

    The final transition's ``next_state`` is **observation row ``T``** -- the post-step state
    after the last decision, which the window loader never yields and which is exactly the
    bootstrap target the registered fairness constraint is about.  Episodes are read through
    ``offline.trajectory_logger.load_episode``, the format's single sanctioned reader, and are
    selected from ``dataset.episode_records`` so the draw-split leakage guard stays in the
    loader.

    States are normalised with the dataset's frozen statistics; rewards are the stream's own
    ``local_reward``, multiplied by *reward_scale*.
    """
    key = _resolve_group(dataset, group)
    stats = dataset.stats
    scale = float(reward_scale)

    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    stream_ids: list[np.ndarray] = []
    steps: list[np.ndarray] = []

    stream_index = 0
    for record in dataset.episode_records:
        episode = load_episode(Path(record.dataset_dir) / record.episode_file)
        length = int(record.episode_length)
        for ix_id in record.ix_ids:
            arrays = episode.intersections[ix_id]
            observation = np.asarray(arrays.state, dtype=np.float32)
            mask = np.asarray(arrays.avail_mask, dtype=np.bool_)
            if (int(observation.shape[1]), int(mask.shape[1])) != key:
                continue
            if observation.shape[0] != length + 1:
                raise ValueError(
                    f"{record.episode_file}: intersection {ix_id!r} has "
                    f"{observation.shape[0]} observation rows for an episode of length "
                    f"{length} (expected {length + 1}); the file violates the C6 alignment "
                    "convention and the final bootstrap target cannot be read from it"
                )
            rows = stats.normalize_state(record.scenario_id, ix_id, observation)
            states.append(rows[:length])
            next_states.append(rows[1 : length + 1])
            actions.append(np.asarray(arrays.action, dtype=np.int64))
            rewards.append(np.asarray(arrays.local_reward, dtype=np.float32) * np.float32(scale))
            stream_ids.append(np.full(length, stream_index, dtype=np.int64))
            steps.append(np.arange(length, dtype=np.int64))
            stream_index += 1

    if not states:
        raise ValueError(
            f"no stream of group {key} in this dataset; it has {sorted(dataset.groups)}"
        )
    return TransitionTable(
        state=torch.from_numpy(np.concatenate(states)),
        next_state=torch.from_numpy(np.concatenate(next_states)),
        action=torch.from_numpy(np.concatenate(actions)),
        reward=torch.from_numpy(np.concatenate(rewards)),
        stream_index=torch.from_numpy(np.concatenate(stream_ids)),
        t=torch.from_numpy(np.concatenate(steps)),
        reward_scale=scale,
    )


def iql_targets(
    table: TransitionTable, next_values: torch.Tensor, gamma: float = IQL_GAMMA
) -> torch.Tensor:
    """``r + gamma * V(s')`` for every transition -- **no ``done`` term anywhere**.

    This is the registered fairness constraint in one line: ``terminated`` is hardcoded ``False``
    and every episode ends by time-limit truncation, so the horizon is not absorbing and the last
    transition of every stream bootstraps like any other.  The training loop takes its targets
    through this function, so the test that pins the behaviour guards the path training takes.
    """
    values = next_values.reshape(-1)
    if values.shape != table.reward.shape:
        raise ValueError(
            f"next_values has {tuple(values.shape)} entries against {tuple(table.reward.shape)} "
            "transitions"
        )
    return table.reward + float(gamma) * values


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def _schedule(optimiser: torch.optim.Optimizer, total: int) -> torch.optim.lr_scheduler.LambdaLR:
    """The DT's linear warmup, shortened only when the run is shorter than the warmup."""
    warmup = min(WARMUP_STEPS, max(1, total // 2))
    return torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: min(1.0, (step + 1) / warmup)
    )


def _curve(losses: Sequence[float], total: int) -> tuple[tuple[float, ...], bool]:
    """Window means of the training loss and the plateau diagnostic, as P4 computes them."""
    window = min(PLATEAU_WINDOW, total)
    while total % window:
        window -= 1
    means = window_means(losses, window)
    return means, (plateau_reached(means) if len(means) >= 3 else False)


def _prepare_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Filesystem-mutation barrier: refuse before training, and create no directory."""
    destination = Path(checkpoint_path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {destination.parent}; nothing is created here"
        )
    return destination


def _diagnostic_rows(count: int, device: torch.device) -> torch.Tensor:
    """A deterministic, capped row sample for the post-training diagnostics."""
    if count <= DIAGNOSTIC_ROWS:
        return torch.arange(count, dtype=torch.int64, device=device)
    stride = count // DIAGNOSTIC_ROWS
    return torch.arange(0, stride * DIAGNOSTIC_ROWS, stride, dtype=torch.int64, device=device)


def train_bc(
    stacked: dict[str, torch.Tensor],
    *,
    state_dim: int,
    n_actions: int,
    seed: int,
    method: str,
    declared_gradient_steps: int,
    batch_size: int,
    device: torch.device,
    checkpoint_path: str | Path,
    stats: NormalizationStats,
    scenario_id: str,
    provenance: dict[str, Any],
    log_every: int = 0,
) -> TrainRecord:
    """Train one BC (or %BC) seed for exactly *declared_gradient_steps* steps.

    The batches, the loss and the optimiser are the DT's: uniform sampling with replacement from
    a ``numpy`` generator seeded by *seed*, ``action_loss`` with ``ignore_index = -1`` over every
    valid window position, masked logits, AdamW with the DT's learning rate, weight decay,
    warmup and gradient clip.  **The only difference from the DT is the architecture** -- a
    per-position MLP that sees one state and nothing else.
    """
    from agent.utils.utils import Utils

    total = int(declared_gradient_steps)
    if total < 1:
        raise ValueError(f"declared_gradient_steps must be >= 1, got {declared_gradient_steps}")
    count = int(stacked["state"].shape[0])
    if count < 1:
        raise ValueError("the stacked dataset is empty")
    destination = _prepare_checkpoint_path(checkpoint_path)

    Utils.seed_everything(int(seed), seed_python_random=False)
    config = TrunkConfig(state_dim=int(state_dim), n_actions=int(n_actions))
    model = MLPTrunk(config, int(n_actions)).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    schedule = _schedule(optimiser, total)

    tensors = {key: value.to(device) for key, value in stacked.items()}
    generator = np.random.default_rng(int(seed))
    losses: list[float] = []
    model.train()
    started = time.time()
    for step in range(total):
        index = torch.from_numpy(
            generator.integers(0, count, size=int(batch_size)).astype(np.int64)
        ).to(device)
        action = tensors["action"][index]
        logits = masked_action_logits(
            model(tensors["state"][index]), tensors["avail_mask"][index]
        )
        loss = action_loss(logits, action)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimiser.step()
        schedule.step()
        losses.append(float(loss.detach()))
        if log_every and (step + 1) % log_every == 0:
            print(
                f"  {method} seed {seed} step {step + 1}/{total} "
                f"loss {np.mean(losses[-log_every:]):.5f}",
                flush=True,
            )
    seconds = time.time() - started

    means, plateaued = _curve(losses, total)
    diagnostics = _bc_diagnostics(model, tensors, device)
    diagnostics["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
    diagnostics["method"] = method
    diagnostics["training_rows"] = count

    merged = {f"policy.{key}": value.detach().cpu() for key, value in model.state_dict().items()}
    digest = canonical_state_dict_digest(merged)
    torch.save(
        {
            "format_version": "bc-checkpoint/1.0",
            "config": config.to_json_obj(),
            "model": merged,
            "canonical_digest": digest,
            "normalise": True,
            "scenario_id": str(scenario_id),
            "stats": stats.to_json_obj(),
            "intersection_ids": [],
            "provenance": {
                **dict(provenance),
                "method": method,
                "seed": int(seed),
                "gradient_steps": int(total),
                "declared_gradient_steps": int(declared_gradient_steps),
                "batch_size": int(batch_size),
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "device": str(device),
                "window_means": list(means),
                "plateaued": bool(plateaued),
                "diagnostics": diagnostics,
                "runtime": runtime_provenance(),
            },
        },
        destination,
    )
    return TrainRecord(
        method=method,
        seed=int(seed),
        gradient_steps=total,
        declared_gradient_steps=int(declared_gradient_steps),
        losses=tuple(losses),
        window_means=means,
        plateaued=bool(plateaued),
        checkpoint_path=str(destination),
        canonical_digest=digest,
        file_sha256=_sha256_file(destination),
        seconds=float(seconds),
        diagnostics=diagnostics,
    )


def _bc_diagnostics(
    model: MLPTrunk, tensors: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, Any]:
    """How often the trained policy reproduces the logged action, on TRAINING data only.

    A mechanism number, not a selection criterion: nothing in this task chooses a model, a
    hyperparameter or a checkpoint by it.
    """
    rows = _diagnostic_rows(int(tensors["state"].shape[0]), device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = masked_action_logits(
                model(tensors["state"][rows]), tensors["avail_mask"][rows]
            )
            action = tensors["action"][rows]
            valid = action >= 0
            agreement = (logits.argmax(dim=-1) == action) & valid
            total = int(valid.sum())
    finally:
        model.train(was_training)
    return {
        "behaviour_agreement": float(int(agreement.sum()) / total) if total else float("nan"),
        "diagnostic_positions": total,
    }


def train_iql(
    table: TransitionTable,
    *,
    state_dim: int,
    n_actions: int,
    seed: int,
    declared_gradient_steps: int,
    batch_size: int,
    device: torch.device,
    checkpoint_path: str | Path,
    stats: NormalizationStats,
    scenario_id: str,
    provenance: dict[str, Any],
    log_every: int = 0,
) -> TrainRecord:
    """Train one IQL seed for exactly *declared_gradient_steps* steps.

    Published D4RL-locomotion hyperparameters, unswept: expectile ``tau = 0.7``, AWR temperature
    ``beta = 3.0``, ``gamma = 0.99``, Polyak ``0.005``, weight clip ``100``.  One step updates
    ``V``, then ``Q``, then the policy, then the target network -- the published order.

    The Q target comes from :func:`iql_targets`, which carries no ``done`` term, so the final
    transition of every episode bootstraps through the truncation boundary.

    The policy loss is computed on **unmasked** logits: the transition table carries no
    availability mask, the logged action is legal by construction, and every mask in this corpus
    is all-``True`` (P3 review), so masking here would be a no-op with a second convention to
    keep consistent.  ``act()`` masks at evaluation, where it can bind.
    """
    from agent.utils.utils import Utils

    total = int(declared_gradient_steps)
    if total < 1:
        raise ValueError(f"declared_gradient_steps must be >= 1, got {declared_gradient_steps}")
    count = len(table)
    if count < 1:
        raise ValueError("the transition table is empty")
    destination = _prepare_checkpoint_path(checkpoint_path)

    Utils.seed_everything(int(seed), seed_python_random=False)
    config = TrunkConfig(state_dim=int(state_dim), n_actions=int(n_actions))
    policy = MLPTrunk(config, int(n_actions)).to(device)
    q = MLPTrunk(config, int(n_actions)).to(device)
    value = MLPTrunk(config, 1).to(device)
    q_target = MLPTrunk(config, int(n_actions)).to(device)
    q_target.load_state_dict(q.state_dict())
    for parameter in q_target.parameters():
        parameter.requires_grad_(False)

    optimisers = {
        "policy": torch.optim.AdamW(
            policy.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        ),
        "q": torch.optim.AdamW(q.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        "v": torch.optim.AdamW(value.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    }
    schedules = {name: _schedule(opt, total) for name, opt in optimisers.items()}

    device_table = table.to(device)
    generator = np.random.default_rng(int(seed))
    losses: list[float] = []
    component: dict[str, list[float]] = {"v": [], "q": [], "policy": []}
    for module in (policy, q, value):
        module.train()
    started = time.time()
    for step in range(total):
        index = torch.from_numpy(
            generator.integers(0, count, size=int(batch_size)).astype(np.int64)
        ).to(device)
        batch = device_table.select(index)
        action = batch.action.reshape(-1, 1)

        with torch.no_grad():
            q_taken = q_target(batch.state).gather(1, action).reshape(-1)

        # 1. V: expectile regression of the target Q onto V.
        v_pred = value(batch.state).reshape(-1)
        residual = q_taken - v_pred
        weight = torch.where(
            residual < 0, torch.full_like(residual, 1.0 - IQL_TAU), torch.full_like(residual, IQL_TAU)
        )
        v_loss = (weight * residual.pow(2)).mean()
        optimisers["v"].zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), GRAD_CLIP)
        optimisers["v"].step()
        schedules["v"].step()

        # 2. Q: TD regression onto r + gamma * V(s'), bootstrapping through every boundary.
        with torch.no_grad():
            next_value = value(batch.next_state).reshape(-1)
            target = iql_targets(batch, next_value, gamma=IQL_GAMMA)
        q_pred = q(batch.state).gather(1, action).reshape(-1)
        q_loss = F.mse_loss(q_pred, target)
        optimisers["q"].zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), GRAD_CLIP)
        optimisers["q"].step()
        schedules["q"].step()

        # 3. Policy: advantage-weighted regression onto the logged action.
        with torch.no_grad():
            advantage = q_taken - v_pred.detach()
            awr_weight = torch.clamp(torch.exp(IQL_BETA * advantage), max=IQL_WEIGHT_CLIP)
        log_probs = torch.log_softmax(policy(batch.state), dim=-1).gather(1, action).reshape(-1)
        policy_loss = -(awr_weight * log_probs).mean()
        optimisers["policy"].zero_grad(set_to_none=True)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        optimisers["policy"].step()
        schedules["policy"].step()

        # 4. Polyak update of the target network.
        with torch.no_grad():
            for online, target_parameter in zip(q.parameters(), q_target.parameters()):
                target_parameter.mul_(1.0 - IQL_POLYAK).add_(online, alpha=IQL_POLYAK)

        losses.append(float(policy_loss.detach()))
        component["v"].append(float(v_loss.detach()))
        component["q"].append(float(q_loss.detach()))
        component["policy"].append(float(policy_loss.detach()))
        if log_every and (step + 1) % log_every == 0:
            print(
                f"  iql seed {seed} step {step + 1}/{total} "
                f"policy {np.mean(component['policy'][-log_every:]):.5f} "
                f"q {np.mean(component['q'][-log_every:]):.5f} "
                f"v {np.mean(component['v'][-log_every:]):.5f}",
                flush=True,
            )
    seconds = time.time() - started

    means, plateaued = _curve(losses, total)
    diagnostics = _iql_diagnostics(policy, q_target, value, device_table, device)
    diagnostics.update(
        {
            "tau": IQL_TAU,
            "beta": IQL_BETA,
            "gamma": IQL_GAMMA,
            "polyak": IQL_POLYAK,
            "weight_clip": IQL_WEIGHT_CLIP,
            "reward_scale": float(table.reward_scale),
            "hyperparameter_provenance": (
                "published IQL D4RL-locomotion values, unswept, transplanted onto a discrete "
                "8-phase action space; see docs/plans/p4.4.md section 3.5 for why no sweep ships"
            ),
            "final_v_loss": component["v"][-1],
            "final_q_loss": component["q"][-1],
            "final_policy_loss": component["policy"][-1],
            "parameter_count": int(
                sum(p.numel() for module in (policy, q, value) for p in module.parameters())
            ),
            "training_rows": count,
        }
    )

    merged: dict[str, torch.Tensor] = {}
    for prefix, module in (("policy", policy), ("q", q), ("v", value), ("q_target", q_target)):
        for key, tensor in module.state_dict().items():
            merged[f"{prefix}.{key}"] = tensor.detach().cpu()
    digest = canonical_state_dict_digest(merged)
    torch.save(
        {
            "format_version": "iql-checkpoint/1.0",
            "config": config.to_json_obj(),
            "model": merged,
            "canonical_digest": digest,
            "normalise": True,
            "scenario_id": str(scenario_id),
            "stats": stats.to_json_obj(),
            "intersection_ids": [],
            "provenance": {
                **dict(provenance),
                "method": "iql",
                "seed": int(seed),
                "gradient_steps": int(total),
                "declared_gradient_steps": int(declared_gradient_steps),
                "batch_size": int(batch_size),
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "device": str(device),
                "window_means": list(means),
                "plateaued": bool(plateaued),
                "diagnostics": diagnostics,
                "runtime": runtime_provenance(),
            },
        },
        destination,
    )
    return TrainRecord(
        method="iql",
        seed=int(seed),
        gradient_steps=total,
        declared_gradient_steps=int(declared_gradient_steps),
        losses=tuple(losses),
        window_means=means,
        plateaued=bool(plateaued),
        checkpoint_path=str(destination),
        canonical_digest=digest,
        file_sha256=_sha256_file(destination),
        seconds=float(seconds),
        diagnostics=diagnostics,
    )


def _iql_diagnostics(
    policy: MLPTrunk,
    q_target: MLPTrunk,
    value: MLPTrunk,
    table: TransitionTable,
    device: torch.device,
) -> dict[str, Any]:
    """Whether the transplanted configuration degenerated -- computed on TRAINING data only.

    ``docs/plans/p4.4.md`` section 3.5 declines the authorised tau/beta sweep because its
    selection criterion cannot rank either parameter (beta enters no value loss at all, and tau
    defines the expectile loss).  What ships instead is this: if every AWR weight is clipped, or
    every weight is ~1, the extraction has collapsed toward uniform or toward plain BC, and the
    number says so mechanistically.  **Nothing here selects anything.**
    """
    rows = _diagnostic_rows(len(table), device)
    sample = table.select(rows)
    modes = [module.training for module in (policy, q_target, value)]
    for module in (policy, q_target, value):
        module.eval()
    try:
        with torch.no_grad():
            action = sample.action.reshape(-1, 1)
            q_taken = q_target(sample.state).gather(1, action).reshape(-1)
            v_pred = value(sample.state).reshape(-1)
            advantage = q_taken - v_pred
            raw = torch.exp(IQL_BETA * advantage)
            weights = torch.clamp(raw, max=IQL_WEIGHT_CLIP)
            agreement = (policy(sample.state).argmax(dim=-1) == sample.action).float().mean()
    finally:
        for module, mode in zip((policy, q_target, value), modes):
            module.train(mode)

    weights64 = weights.double()
    total = float(weights64.sum())
    quantiles = torch.quantile(
        weights64, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, device=weights64.device)
    )
    return {
        "awr_weight_mean": float(weights64.mean()),
        "awr_weight_q10": float(quantiles[0]),
        "awr_weight_median": float(quantiles[1]),
        "awr_weight_q90": float(quantiles[2]),
        "awr_weight_clipped_fraction": float((raw >= IQL_WEIGHT_CLIP).double().mean()),
        "awr_weight_near_zero_fraction": float((weights64 < 0.01).double().mean()),
        "awr_weight_effective_sample_size": (
            float(total * total / float((weights64 * weights64).sum()) / weights64.numel())
            if total > 0
            else 0.0
        ),
        "advantage_mean": float(advantage.double().mean()),
        "advantage_std": float(advantage.double().std(unbiased=True)),
        "behaviour_agreement": float(agreement),
        "diagnostic_positions": int(weights64.numel()),
    }


# ----------------------------------------------------------------------
# Effect sizes, the A6 verdict and the recovered fraction
# ----------------------------------------------------------------------


def rank_biserial(result: WilcoxonResult) -> float:
    """Matched-pairs rank-biserial correlation ``(W+ - W-) / (W+ + W-)``.

    The effect size that belongs beside a signed-rank p-value: the test is scale-free, so a
    tiny but perfectly consistent shift is highly significant, and the p-value alone cannot say
    whether the difference matters.  Positive means the FIRST argument of the test was larger
    more often and by larger ranks.
    """
    total = float(result.w_plus) + float(result.w_minus)
    if total <= 0.0:
        return 0.0
    return (float(result.w_plus) - float(result.w_minus)) / total


def paired_comparison(
    left: Sequence[EpisodeResult], right: Sequence[EpisodeResult]
) -> PairedComparison:
    """Paired per-draw comparison of two arms over their shared draws.

    The per-draw unit is the mean over training seeds, as in P4, so seed and draw stay crossed.
    Amendment A5 point 3 makes a comparison without shared draws **void**, which
    ``dt_gate._paired`` enforces by raising.
    """
    left_values, right_values, shared = _paired(left, right)
    arms = (sorted({r.arm for r in left}), sorted({r.arm for r in right}))
    if len(arms[0]) != 1 or len(arms[1]) != 1:
        raise ValueError(f"a paired comparison needs one arm on each side, got {arms}")

    differences = [a - b for a, b in zip(left_values, right_values)]
    cell = mean_ci95(differences)
    result = wilcoxon_signed_rank(left_values, right_values)
    return PairedComparison(
        left_arm=arms[0][0],
        right_arm=arms[1][0],
        n_shared_draws=len(shared),
        draw_ids=tuple(shared),
        mean_left=float(np.mean(left_values)),
        mean_right=float(np.mean(right_values)),
        mean_difference=cell.mean,
        ci95_half_width=cell.ci95,
        ci95_width=2.0 * cell.ci95,
        ci95_low=cell.mean - cell.ci95,
        ci95_high=cell.mean + cell.ci95,
        median_difference=float(np.median(differences)),
        wins=int(sum(1 for d in differences if d < 0)),
        losses=int(sum(1 for d in differences if d > 0)),
        ties=int(sum(1 for d in differences if d == 0)),
        rank_biserial=rank_biserial(result),
        wilcoxon=result,
    )


def equivalence_verdict(
    mean_difference: float, ci95_half_width: float, delta: float = DELTA_ATT
) -> str:
    """A6's verdict for a paired difference ``DT - baseline`` (lower ATT is better).

    ``[-delta, +delta]`` is **closed**: a CI landing exactly on the margin lies within it.
    "Entirely below ``-delta``" is **strict**, so a CI whose upper end touches ``-delta`` is
    inconclusive rather than decisive -- the conservative reading in both directions.

    A6 names three branches.  The fourth, a CI entirely above ``+delta``, is returned as
    :data:`VERDICT_BASELINE_BETTER` instead of being folded into "inconclusive": it is a
    decisive result in the other direction, and reporting it as an absence of one would be
    wrong.  The asymmetry is raised with the coordinator rather than resolved here.
    """
    half = float(ci95_half_width)
    margin = float(delta)
    if half < 0.0:
        raise ValueError(f"ci95_half_width must be >= 0, got {ci95_half_width!r}")
    if margin <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta!r}")
    low = float(mean_difference) - half
    high = float(mean_difference) + half
    if low >= -margin and high <= margin:
        return VERDICT_MATCHES
    if high < -margin:
        return VERDICT_DT_BETTER
    if low > margin:
        return VERDICT_BASELINE_BETTER
    return VERDICT_INCONCLUSIVE


def recovered_fraction(att_reference: float, att_arm: float, att_dt: float) -> float:
    """``(reference - arm) / (reference - dt)`` -- reported unconditionally (A6 clarified).

    1.0 means the arm recovered the DT's entire margin over the behaviour policy, 0.0 means it
    recovered none of it, and a negative value means it lands below the behaviour policy.  It
    carries what the binary verdict compresses away: a baseline inside the equivalence margin
    can still have recovered only a fifth of the effect under study.
    """
    denominator = float(att_reference) - float(att_dt)
    if denominator == 0.0:
        raise ValueError(
            "the reference and the DT have the same ATT, so the recovered fraction divides by "
            "zero; there is no margin to recover"
        )
    return (float(att_reference) - float(att_arm)) / denominator


# ----------------------------------------------------------------------
# Campaign integrity and the artifact
# ----------------------------------------------------------------------


def assert_campaign_complete(
    requested: Sequence[tuple[str, int | None, int]],
    produced: Sequence[EpisodeResult],
) -> None:
    """Refuse to report a partial campaign: completed runs must equal runs requested.

    The condition attached to running the evaluation in-session.  A campaign that aborted
    halfway would otherwise produce a smaller, quieter, perfectly plausible cell.
    """
    from collections import Counter

    want = Counter(tuple(item) for item in requested)
    got = Counter((r.arm, r.seed, r.draw_id) for r in produced)
    missing = sorted(str(k) for k in (want - got))
    extra = sorted(str(k) for k in (got - want))
    if missing:
        raise ValueError(
            f"incomplete campaign: {len(missing)} of {sum(want.values())} requested runs are "
            f"missing, first {missing[:5]}"
        )
    if extra:
        raise ValueError(
            f"{len(extra)} produced run(s) were not requested, first {extra[:5]}; a cell must "
            "describe exactly the runs the campaign asked for"
        )


def _grouped(episodes: Sequence[EpisodeResult]) -> dict[str, list[EpisodeResult]]:
    out: dict[str, list[EpisodeResult]] = {}
    for episode in episodes:
        out.setdefault(episode.arm, []).append(episode)
    return out


def baselines_artifact(
    *,
    episodes: Sequence[EpisodeResult],
    training: dict[str, Any],
    gate_a: dict[str, Any],
    env_settings: dict[str, Any],
    engine_seed: int,
    delta: float = DELTA_ATT,
) -> dict[str, Any]:
    """The reported artifact: cells, paired comparisons, verdicts and recovered fractions.

    Every quantity A5 and A6 make unconditional is present for every arm: ``att_horizon``,
    ``vehicle_count`` at the horizon, the draw ids, the CI width, the effect size, the verdict
    and the recovered fraction.  The verdict is computed under **both** the declared ``delta``
    and its full-precision derivation, and the artifact refuses to be built if the two disagree
    -- A6's multiplier is a choice, and a verdict must not turn on a rounding.
    """
    by_arm = _grouped(episodes)
    for required in ("madt", "mappo1000"):
        if required not in by_arm:
            raise ValueError(
                f"the artifact needs the {required!r} arm: without it there is no comparison "
                f"and no recovered fraction to report; present arms are {sorted(by_arm)}"
            )

    cells = {arm: _cell(results) for arm, results in sorted(by_arm.items())}
    att = {arm: cell["att_horizon_mean"] for arm, cell in cells.items()}
    reference, dt_mean = att["mappo1000"], att["madt"]

    comparisons: dict[str, Any] = {}
    for method in METHODS:
        if method not in by_arm:
            continue
        comparison = paired_comparison(by_arm["madt"], by_arm[method])
        verdict = equivalence_verdict(
            comparison.mean_difference, comparison.ci95_half_width, delta
        )
        alternative = equivalence_verdict(
            comparison.mean_difference, comparison.ci95_half_width, DELTA_ATT_DERIVATION
        )
        if verdict != alternative:
            raise ValueError(
                f"the verdict for {method!r} depends on delta's rounding: {verdict!r} at "
                f"delta={delta} and {alternative!r} at delta={DELTA_ATT_DERIVATION}. A6's "
                "multiplier is a choice and a verdict must not turn on it; report this instead "
                "of picking one"
            )
        direct = recovered_fraction(reference, att[method], dt_mean)
        # Second route to the same number: 1 - mean(arm - DT) / (reference - DT).  The two agree
        # exactly only for a balanced design, which is what "seed crossed with draw" gives.
        paired_route = 1.0 - (-comparison.mean_difference) / (reference - dt_mean)
        if abs(direct - paired_route) > 1e-9:
            raise ValueError(
                f"the recovered fraction for {method!r} disagrees between its two routes: "
                f"{direct} against {paired_route}; that means the seed-by-draw design is not "
                "balanced and the cell mean is not the mean of the per-draw means"
            )
        comparisons[f"madt_vs_{method}"] = {
            **comparison.to_json_obj(),
            "verdict": verdict,
            "verdict_at_full_precision_delta": alternative,
            "delta": float(delta),
            "recovered_fraction": direct,
            "recovered_fraction_paired_route": paired_route,
            "distance_from_ci_endpoints_to_delta": min(
                abs(comparison.ci95_low + delta),
                abs(comparison.ci95_low - delta),
                abs(comparison.ci95_high + delta),
                abs(comparison.ci95_high - delta),
            ),
        }

    behaviour: dict[str, Any] = {}
    for arm in ("madt", *METHODS):
        if arm in by_arm and arm != "mappo1000":
            behaviour[f"mappo1000_vs_{arm}"] = paired_comparison(
                by_arm["mappo1000"], by_arm[arm]
            ).to_json_obj()

    draw_ids = sorted({episode.draw_id for episode in episodes})
    forecasts = _forecast_outcomes(comparisons, by_arm, delta)
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "P4.4: BC, %BC and IQL against the P4 Decision Transformer on the registered "
            "held-out pool, paired by draw"
        ),
        "evaluation_pool": "registered held-out draws 1000-1099 (PREREGISTRATION.md D4)",
        "draw_ids": draw_ids,
        "engine_seed": int(engine_seed),
        "env_settings": {k: v for k, v in env_settings.items() if k != "compare_with"},
        "equivalence_margin_delta": float(delta),
        "equivalence_margin_delta_derivation": DELTA_ATT_DERIVATION,
        "decision_rule": (
            "PREREGISTRATION.md A6: matches iff the 95% CI of the paired per-draw difference "
            "(DT - baseline) lies entirely within [-delta, +delta]; the DT is genuinely better "
            "iff it lies entirely below -delta; a CI entirely above +delta is the baseline "
            "genuinely better, a branch A6 does not name; anything else is inconclusive at this "
            "power, reported with the CI width"
        ),
        "attribution_caveat": (
            "DT minus baseline is a COMBINED difference: attention/context plus return-to-go "
            "conditioning plus the timestep embedding. It is not a sequence-modelling effect; "
            "P4.3 (RTG) and P5.3 (no-RTG, context length K) own the decomposition"
        ),
        "iql_caveat": (
            "IQL here is untuned, running published D4RL-locomotion values transplanted onto a "
            "discrete 8-phase action space: a losing untuned IQL is not evidence that MADT "
            "outperforms IQL"
        ),
        "cells": cells,
        "comparisons": comparisons,
        "behaviour_policy_comparisons": behaviour,
        "registered_forecasts": forecasts,
        "gate_a": dict(gate_a),
        "training": dict(training),
        "episodes": [
            {
                "arm": e.arm,
                "seed": e.seed,
                "draw_id": e.draw_id,
                "att_horizon": e.att_horizon,
                "horizon_vehicle_count": e.horizon_vehicle_count,
                "episode_reward": e.episode_reward,
            }
            for e in episodes
        ],
        "runtime": runtime_provenance(),
    }


def _forecast_outcomes(
    comparisons: Mapping[str, Any],
    by_arm: Mapping[str, list[EpisodeResult]],
    delta: float,
) -> dict[str, Any]:
    """Score the forecasts registered in ``docs/plans/p4.4.md`` section 4 before training."""
    out: dict[str, Any] = {
        "registered_in": "docs/plans/p4.4.md section 4, before the first gradient step"
    }
    if "madt_vs_bc" in comparisons:
        difference = float(comparisons["madt_vs_bc"]["mean_difference"])
        out["primary_bc_within_delta_of_the_dt"] = {
            "statement": "BC lands within delta of the DT on the paired held-out mean",
            "paired_mean_difference_dt_minus_bc": difference,
            "delta": float(delta),
            "held": bool(abs(difference) <= float(delta)),
        }
    if "bc" in by_arm and "bc_top10" in by_arm:
        secondary = paired_comparison(by_arm["bc"], by_arm["bc_top10"])
        out["secondary_bc_top10_within_delta_of_bc"] = {
            "statement": "%BC lands within delta of BC on the paired held-out mean",
            "paired_mean_difference_bc_minus_bc_top10": secondary.mean_difference,
            "delta": float(delta),
            "held": bool(abs(secondary.mean_difference) <= float(delta)),
        }
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """CLI: ``gate`` (reproduce P4), ``train``, ``evaluate`` (one method), ``report``."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.offline_baselines",
        description="Train and evaluate P4.4's offline baselines: BC, %BC and IQL.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="a collection manifest of the training scenario; the evaluation env settings are "
        "read from it rather than restated",
    )
    parser.add_argument("--draws-root", default="scenarios/draws")
    parser.add_argument("--scenario-key", default="cityflow1x1")
    parser.add_argument("--scenario-id", default="cityflow1x1")
    parser.add_argument("--engine-seed", type=int, default=1000)
    parser.add_argument("--out-dir", default="docs/data")
    parser.add_argument("--work-dir", default="output/p4_4")
    parser.add_argument("--checkpoint-dir", default="output/p4_4/checkpoints")
    parser.add_argument("--device", default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    gate = sub.add_parser("gate", help="re-roll P4's cited arms and refuse on any difference")
    gate.add_argument("--dt-checkpoint", action="append", default=[], metavar="SEED=PATH")
    gate.add_argument("--mappo-checkpoint", action="append", default=[], metavar="SEED=PATH")
    gate.add_argument("--dt-steps", type=int, default=40_000)
    gate.add_argument("--reference", default="docs/data/p4_gate.json")
    gate.add_argument("--thresholds", default="docs/data/p4_heldout_thresholds.json")

    train = sub.add_parser("train", help="train every method and seed to the declared budget")
    train.add_argument("--dataset-dir", action="append", required=True)
    train.add_argument("--steps", type=int, default=DECLARED_GRADIENT_STEPS)
    train.add_argument("--methods", default=",".join(METHODS))
    train.add_argument("--log-every", type=int, default=2000)

    evaluate = sub.add_parser("evaluate", help="evaluate ONE method over the held-out pool")
    evaluate.add_argument("--method", required=True, choices=list(METHODS))
    evaluate.add_argument("--steps", type=int, default=DECLARED_GRADIENT_STEPS)

    report = sub.add_parser("report", help="merge the per-method runs into the artifact")
    report.add_argument("--methods", default=",".join(METHODS))
    return parser


def _checkpoint_map(specs: Iterable[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for spec in specs:
        seed, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"expected SEED=PATH, got {spec!r}")
        out[int(seed)] = path
    return out


def load_baseline_checkpoint(
    gym_env: Any, path: str | Path, declared_gradient_steps: int
) -> dict[str, Any]:
    """Read a baseline checkpoint, refusing one whose step count is not the declared one.

    The mechanical form of "no online model selection" (``PREREGISTRATION.md`` section 6.1),
    mirroring ``dt_gate.load_gate_checkpoint``: a checkpoint saved at a different step -- an
    earlier one that scored better, say -- cannot be evaluated by this path at all.  The check
    runs before any agent is constructed, so a refusal builds nothing.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    recorded = payload.get("provenance", {}).get("gradient_steps")
    if recorded is None:
        raise ValueError(
            f"{path}: the checkpoint records no gradient step count, so it cannot be shown to "
            "be the pre-declared one; refusing rather than reporting an unidentifiable model"
        )
    if int(recorded) != int(declared_gradient_steps):
        raise ValueError(
            f"{path}: checkpoint was saved at {int(recorded)} gradient steps but the declared "
            f"count is {int(declared_gradient_steps)}. PREREGISTRATION.md section 6 forbids "
            "reporting a checkpoint chosen by anything other than the declared budget"
        )
    return payload


def _baseline_factory(method: str, path: str, declared: int, device: str | None) -> Callable[[Any], Any]:
    def factory(env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
        load_baseline_checkpoint(env, path, declared)
        agent_class = IQLAgent if method == "iql" else BCAgent
        agent = agent_class.from_checkpoint(env, path, device=device)
        return lambda _env, info: agent.act(info, explore=False, update_memory=False)

    return factory


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    from offline.materialise_draws import draw_config_path

    args = build_parser().parse_args(argv)
    settings = env_settings_from_manifest(args.manifest)
    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"--out-dir does not exist: {out_dir}")

    def config_for_draw(draw_id: int) -> Path:
        return draw_config_path(args.scenario_key, draw_id, out_root=args.draws_root)

    if args.command == "gate":
        return _run_gate(args, settings, config_for_draw, work_dir)
    if args.command == "train":
        return _run_train(args, out_dir)
    if args.command == "evaluate":
        return _run_evaluate(args, settings, config_for_draw, out_dir, work_dir)
    return _run_report(args, settings, out_dir, work_dir)


def _episode_key(episode: EpisodeResult) -> tuple[str, int | None, int]:
    return (episode.arm, episode.seed, episode.draw_id)


def _run_gate(
    args: argparse.Namespace,
    settings: dict[str, Any],
    config_for_draw: Callable[[int], Path],
    work_dir: Path,
) -> int:
    """Gate A: re-roll P4's cited arms here and refuse to continue on any difference."""
    from offline.dt_gate import _maxpressure_factory, _mappo_factory, load_gate_checkpoint

    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    committed = {
        _episode_key(EpisodeResult(**e)): EpisodeResult(**e)
        for e in [*reference["episodes"], *thresholds["episodes"]]
        if e["arm"] in CITED_ARMS
    }
    if not committed:
        raise ValueError("no cited episodes found in the reference artifacts")

    draws = list(HELD_OUT_DRAWS)
    dt_paths = _checkpoint_map(args.dt_checkpoint)
    mappo_paths = _checkpoint_map(args.mappo_checkpoint)
    work_dir.mkdir(parents=True, exist_ok=True)

    produced: list[EpisodeResult] = []
    mismatches: list[dict[str, Any]] = []

    def compare(results: Sequence[EpisodeResult]) -> None:
        for result in results:
            expected = committed.get(_episode_key(result))
            if expected is None:
                raise ValueError(f"{_episode_key(result)} is not in the committed reference")
            if (
                result.att_horizon != expected.att_horizon
                or result.horizon_vehicle_count != expected.horizon_vehicle_count
                or result.episode_reward != expected.episode_reward
            ):
                mismatches.append(
                    {
                        "key": list(_episode_key(result)),
                        "measured": [result.att_horizon, result.horizon_vehicle_count],
                        "committed": [expected.att_horizon, expected.horizon_vehicle_count],
                    }
                )
                raise ValueError(
                    f"GATE A FAILED at {_episode_key(result)}: measured "
                    f"{result.att_horizon!r} against the committed {expected.att_horizon!r}. "
                    "P4.4 would be measuring on a different instrument than P4; stopping"
                )
        produced.extend(results)

    print(f"Gate A: MaxPressure over {len(draws)} held-out draws", flush=True)
    compare(
        evaluate_arm(
            arm="maxpressure",
            seed=None,
            draw_ids=draws,
            config_for_draw=config_for_draw,
            env_settings=settings,
            scenario_id=args.scenario_id,
            choose_action_factory=_maxpressure_factory,
            engine_seed=args.engine_seed,
        )
    )
    for seed, path in sorted(mappo_paths.items()):
        print(f"Gate A: MAPPO@1000 seed {seed}", flush=True)
        compare(
            evaluate_arm(
                arm="mappo1000",
                seed=seed,
                draw_ids=draws,
                config_for_draw=config_for_draw,
                env_settings=settings,
                scenario_id=args.scenario_id,
                choose_action_factory=_mappo_factory(path, args.device),
                engine_seed=args.engine_seed,
            )
        )
    for seed, path in sorted(dt_paths.items()):
        print(f"Gate A: MADT seed {seed}", flush=True)

        def dt_factory(env: Any, path: str = path) -> Callable[[Any, dict[str, Any]], np.ndarray]:
            agent = load_gate_checkpoint(env, path, int(args.dt_steps), device=args.device)
            return lambda _env, info: agent.act(info, explore=False, update_memory=True)

        compare(
            evaluate_arm(
                arm="madt",
                seed=seed,
                draw_ids=draws,
                config_for_draw=config_for_draw,
                env_settings=settings,
                scenario_id=args.scenario_id,
                choose_action_factory=dt_factory,
                engine_seed=args.engine_seed,
            )
        )

    assert_campaign_complete(list(committed), produced)
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "Gate A: P4's cited arms re-rolled on this instrument before any baseline number "
            "was measured; exact equality per (arm, seed, draw) or the task stops"
        ),
        "status": "PASS",
        "compared": len(produced),
        "mismatches": len(mismatches),
        "arms": sorted({e.arm for e in produced}),
        "engine_seed": int(args.engine_seed),
        "env_settings": {k: v for k, v in settings.items() if k != "compare_with"},
        "episodes": [
            {
                "arm": e.arm,
                "seed": e.seed,
                "draw_id": e.draw_id,
                "att_horizon": e.att_horizon,
                "horizon_vehicle_count": e.horizon_vehicle_count,
                "episode_reward": e.episode_reward,
            }
            for e in produced
        ],
        "runtime": runtime_provenance(),
    }
    write_json_atomic(payload, work_dir / "gate_a.json")
    print(
        f"\nGATE A PASS: {len(produced)} episodes reproduced exactly, 0 mismatches", flush=True
    )
    return 0


def _run_train(args: argparse.Namespace, out_dir: Path) -> int:
    """Train every requested method for every seed, at the declared budget."""
    from agent.utils.utils import Utils
    from offline.dt_gate import CONTEXT_LENGTH

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown method(s) {unknown}; known methods are {list(METHODS)}")

    dataset = build_training_dataset(args.dataset_dir, CONTEXT_LENGTH)
    stacked = stack_dataset(dataset)                        # one group, or it raises
    (state_dim, n_actions) = next(iter(dataset.groups))
    scenario_id = dataset.episode_records[0].scenario_id
    streams = stream_returns(dataset)
    kept = top_return_streams(dataset, TOP_RETURN_FRACTION)
    scale = iql_reward_scale([s.total_return for s in streams])

    device = torch.device(args.device) if args.device else Utils.resolve_device(None)
    checkpoints = Path(args.checkpoint_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    provenance = {
        "tier": "mappo1000",
        "dataset_dirs": [str(d) for d in args.dataset_dir],
        "training_draw_ids": list(dataset.stats.draw_ids),
        "scenario_id": scenario_id,
    }

    print(
        f"training windows {len(dataset)}  state_dim {state_dim}  n_actions {n_actions}\n"
        f"streams {len(streams)}  top-10% kept {len(kept)} (cut "
        f"{min(k.total_return for k in kept)}, next "
        f"{max((s.total_return for s in streams if s not in kept), default=float('nan'))})\n"
        f"iql reward scale {scale}  device {device}",
        flush=True,
    )

    records: list[TrainRecord] = []
    table: TransitionTable | None = None
    for method in methods:
        for seed in TRAINING_SEEDS:
            path = checkpoints / f"{method}_seed{seed}.pt"
            if method == "iql":
                if table is None:
                    table = build_transitions(dataset, reward_scale=scale)
                    print(f"transitions {len(table)}", flush=True)
                records.append(
                    train_iql(
                        table,
                        state_dim=state_dim,
                        n_actions=n_actions,
                        seed=seed,
                        declared_gradient_steps=int(args.steps),
                        batch_size=IQL_BATCH_TRANSITIONS,
                        device=device,
                        checkpoint_path=path,
                        stats=dataset.stats,
                        scenario_id=scenario_id,
                        provenance=provenance,
                        log_every=args.log_every,
                    )
                )
            else:
                batch = stacked
                if method == "bc_top10":
                    batch = filter_stacked_to_streams(dataset, stacked, kept)
                records.append(
                    train_bc(
                        batch,
                        state_dim=state_dim,
                        n_actions=n_actions,
                        seed=seed,
                        method=method,
                        declared_gradient_steps=int(args.steps),
                        batch_size=BC_BATCH_WINDOWS,
                        device=device,
                        checkpoint_path=path,
                        stats=dataset.stats,
                        scenario_id=scenario_id,
                        provenance=provenance,
                        log_every=args.log_every,
                    )
                )
            last = records[-1]
            print(
                f"  {last.method} seed {last.seed}: {last.seconds:.1f}s  "
                f"final loss {last.losses[-1]:.5f}  plateaued {last.plateaued}  "
                f"digest {last.canonical_digest[:12]}",
                flush=True,
            )

    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "training record for P4.4's offline baselines; the reported checkpoint is the one "
            "at the declared step count and no raise is available to this task"
        ),
        "declared_gradient_steps": int(args.steps),
        "declared_in": "docs/plans/p4.4.md section 3.1, before the first gradient step",
        "raise_available": False,
        "seeds": list(TRAINING_SEEDS),
        "training_draw_ids": list(dataset.stats.draw_ids),
        "scenario_id": scenario_id,
        "group": [int(state_dim), int(n_actions)],
        "window_count": len(dataset),
        "batch_sizes": {"bc": BC_BATCH_WINDOWS, "bc_top10": BC_BATCH_WINDOWS, "iql": IQL_BATCH_TRANSITIONS},
        "top_return_filter": {
            "fraction": TOP_RETURN_FRACTION,
            "streams_total": len(streams),
            "streams_kept": len(kept),
            "cut_return": min(k.total_return for k in kept),
            "next_return_below_the_cut": max(
                (s.total_return for s in streams if s.key not in {k.key for k in kept}),
                default=float("nan"),
            ),
            "kept_streams": [
                {"episode_file": k.episode_file, "flow_draw": k.flow_draw, "return": k.total_return}
                for k in kept
            ],
        },
        "iql": {
            "tau": IQL_TAU,
            "beta": IQL_BETA,
            "gamma": IQL_GAMMA,
            "polyak": IQL_POLYAK,
            "weight_clip": IQL_WEIGHT_CLIP,
            "reward_scale": scale,
            "sweep": "none",
            "sweep_decision": (
                "declared in docs/plans/p4.4.md section 3.5 before the first gradient step: the "
                "authorised criterion (training-split expectile/TD loss) cannot rank either "
                "parameter it is authorised to rank -- beta appears in no value loss, and "
                "selecting it on the policy loss provably drives beta to 0, at which point IQL's "
                "policy extraction IS plain BC; tau defines the expectile loss, so two tau values "
                "are values of two different objectives"
            ),
            "reported_as": "untuned",
        },
        "runs": [
            {
                "method": r.method,
                "seed": r.seed,
                "gradient_steps": r.gradient_steps,
                "plateaued": r.plateaued,
                "window_means": list(r.window_means),
                "final_loss": r.losses[-1],
                "seconds": r.seconds,
                "checkpoint": r.checkpoint_path,
                "canonical_digest": r.canonical_digest,
                "file_sha256": r.file_sha256,
                "diagnostics": r.diagnostics,
            }
            for r in records
        ],
        "runtime": runtime_provenance(),
    }
    write_json_atomic(payload, out_dir / "p4_4_training.json")
    return 0


def _run_evaluate(
    args: argparse.Namespace,
    settings: dict[str, Any],
    config_for_draw: Callable[[int], Path],
    out_dir: Path,
    work_dir: Path,
) -> int:
    """Evaluate ONE method over the held-out pool, so no single job runs long."""
    training = json.loads((out_dir / "p4_4_training.json").read_text(encoding="utf-8"))
    if int(training["declared_gradient_steps"]) != int(args.steps):
        raise ValueError(
            f"the training artifact was produced under a declared budget of "
            f"{training['declared_gradient_steps']} steps but this evaluation was asked for "
            f"{int(args.steps)}; PREREGISTRATION.md section 6 fixes the budget before training"
        )
    leaked = sorted(set(HELD_OUT_DRAWS) & set(training["training_draw_ids"]))
    if leaked:
        raise ValueError(
            f"evaluation draws {leaked[:5]} are also training draws; the held-out pool is not "
            "held out and no number from this run may be reported"
        )

    runs = [r for r in training["runs"] if r["method"] == args.method]
    if not runs:
        raise ValueError(f"the training artifact records no runs for method {args.method!r}")

    draws = list(HELD_OUT_DRAWS)
    requested = [(args.method, int(r["seed"]), d) for r in runs for d in draws]
    produced: list[EpisodeResult] = []
    for run in runs:
        print(f"{args.method} seed {run['seed']} over {len(draws)} draws", flush=True)
        produced.extend(
            evaluate_arm(
                arm=args.method,
                seed=int(run["seed"]),
                draw_ids=draws,
                config_for_draw=config_for_draw,
                env_settings=settings,
                scenario_id=args.scenario_id,
                choose_action_factory=_baseline_factory(
                    args.method, run["checkpoint"], int(args.steps), args.device
                ),
                engine_seed=args.engine_seed,
            )
        )
    assert_campaign_complete(requested, produced)

    work_dir.mkdir(parents=True, exist_ok=True)
    cell = _cell(produced)
    write_json_atomic(
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "method": args.method,
            "declared_gradient_steps": int(args.steps),
            "engine_seed": int(args.engine_seed),
            "cell": cell,
            "episodes": [
                {
                    "arm": e.arm,
                    "seed": e.seed,
                    "draw_id": e.draw_id,
                    "att_horizon": e.att_horizon,
                    "horizon_vehicle_count": e.horizon_vehicle_count,
                    "episode_reward": e.episode_reward,
                }
                for e in produced
            ],
            "runtime": runtime_provenance(),
        },
        work_dir / f"eval_{args.method}.json",
    )
    print(
        f"  {args.method}: att_horizon {cell['att_horizon_mean']:.4f} "
        f"+/- {cell['att_horizon_ci95']:.4f}  vehicle_count "
        f"{cell['horizon_vehicle_count_mean']:.2f}  n={cell['n_episodes']}",
        flush=True,
    )
    return 0


def _run_report(
    args: argparse.Namespace, settings: dict[str, Any], out_dir: Path, work_dir: Path
) -> int:
    """Merge Gate A and the per-method runs into the one committed artifact."""
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    gate = json.loads((work_dir / "gate_a.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise ValueError(
            f"Gate A did not pass ({gate.get('status')!r}); no baseline number may be reported "
            "against P4's arms until the instrument reproduces them"
        )
    training = json.loads((out_dir / "p4_4_training.json").read_text(encoding="utf-8"))

    episodes = [EpisodeResult(**e) for e in gate["episodes"]]
    for method in methods:
        payload = json.loads((work_dir / f"eval_{method}.json").read_text(encoding="utf-8"))
        episodes.extend(EpisodeResult(**e) for e in payload["episodes"])

    # The expected run set is derived from the DECLARED design -- the cited arms, the training
    # artifact's seeds and the registered held-out pool -- and never from the episodes being
    # checked.  Deriving it from them would make the completeness check a tautology, which is
    # what it was in the first draft of this function.
    requested: list[tuple[str, int | None, int]] = [
        ("maxpressure", None, draw) for draw in HELD_OUT_DRAWS
    ]
    for arm in ("madt", "mappo1000"):
        requested += [(arm, seed, draw) for seed in TRAINING_SEEDS for draw in HELD_OUT_DRAWS]
    for method in methods:
        method_seeds = [int(r["seed"]) for r in training["runs"] if r["method"] == method]
        requested += [(method, seed, draw) for seed in method_seeds for draw in HELD_OUT_DRAWS]
    assert_campaign_complete(requested, episodes)
    artifact = baselines_artifact(
        episodes=episodes,
        training={
            key: training[key]
            for key in (
                "declared_gradient_steps",
                "declared_in",
                "raise_available",
                "seeds",
                "training_draw_ids",
                "top_return_filter",
                "iql",
                "runs",
            )
            if key in training
        },
        gate_a={key: gate[key] for key in ("status", "compared", "mismatches", "arms")},
        env_settings=settings,
        engine_seed=int(args.engine_seed),
    )
    write_json_atomic(artifact, out_dir / "p4_4_baselines.json")

    print("\narm            att_horizon   +/- CI    veh     n", flush=True)
    for arm, cell in sorted(artifact["cells"].items()):
        print(
            f"  {arm:12s} {cell['att_horizon_mean']:10.4f} {cell['att_horizon_ci95']:8.4f} "
            f"{cell['horizon_vehicle_count_mean']:7.2f} {cell['n_episodes']:5d}",
            flush=True,
        )
    print("", flush=True)
    for name, entry in sorted(artifact["comparisons"].items()):
        print(
            f"  {name:20s} diff {entry['mean_difference']:+.4f} "
            f"CI [{entry['ci95_low']:+.4f}, {entry['ci95_high']:+.4f}] "
            f"width {entry['ci95_width']:.4f}  p {entry['wilcoxon']['p_value']:.3e}  "
            f"r {entry['rank_biserial']:+.3f}  recovered {entry['recovered_fraction']:+.4f}  "
            f"-> {entry['verdict']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
