"""Train the P4 Decision Transformer and evaluate the pre-registered P4.2 gate.

Artifact format version: ``p4-gate/1.0``.

WHAT THE GATE IS, AND WHERE IT IS MEASURED
------------------------------------------
``PREREGISTRATION.md`` section 9, verbatim and not renegotiable::

    ATT_MADT <= ATT_MaxPressure   AND   ATT_MADT <= 1.05 * ATT_best_online

Evaluated on the **registered held-out pool, all 100 draws 1000-1099** (BRIEF_10 section 8),
with the DT, MaxPressure and MAPPO@1000 measured live and **paired by draw id**.  Section 8 of
``PREREGISTRATION.md`` registered the held-out flow draw as the unit of replication on
2026-08-03, which is why the ladder's ``110.73`` -- measured on the *training* draws 1-200 --
is reported as a labelled secondary reading and is never the verdict.

Amendment A5 governs the reporting: ``vehicle_count`` at the horizon accompanies every ATT cell
**unconditionally**, every cell carries its **draw ids**, and a comparison without shared draws
is void.  Here every arm runs on the identical draw set by construction, so no cell needs
recomputing over an intersection.

THE LEAKAGE RULES THIS MODULE MECHANISES (``PREREGISTRATION.md`` section 6)
---------------------------------------------------------------------------
* **No online model selection.** The reported checkpoint is the one at the declared step count.
  :func:`load_gate_checkpoint` refuses a checkpoint whose recorded step count differs from the
  declared one, so "the best checkpoint" cannot be reported by accident.
* **The single budget raise reads the training curve only.** :func:`plateau_reached` consumes
  training-loss window means and nothing else; no evaluation number can reach it.  The raise is
  all-or-nothing across seeds (amendment A3's strict aggregation), which is why the decision is
  taken by the CLI over every seed's result rather than inside :func:`train_dt`.
* **Statistics are fitted on the training split only** -- delegated to
  ``offline.dataset.TrajectoryWindowDataset``, which fits them only for ``split="train"``, and
  carried into the checkpoint so evaluation reuses the frozen numbers.
* **Held-out purity** is asserted from the artifact: the training and evaluation draw id sets
  must not intersect.

WHY THE EVALUATION ENV IS DERIVED FROM THE CORPUS MANIFEST
-----------------------------------------------------------
:func:`env_settings_from_manifest` reads ``max_steps``, ``delta_time``, ``control_mode``,
``state_features``, the reward functions and the global reward weight out of the collection
manifest instead of restating them.  A DT evaluated in an env whose state block differs from the
collection env reads a different feature space and produces a plausible, wrong number; deriving
the settings from the manifest makes that drift impossible to introduce silently, and
``tests/test_dt_gate.py`` additionally replays a stored episode to prove the reconstruction is
bit-exact.

WILCOXON
--------
scipy is not installed and no repo file imports it, so the paired signed-rank test required by
BRIEF_10 section 8 is implemented here in numpy/stdlib rather than added as a dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from agent.DTAgent import DTAgent, action_loss
from offline.dataset import DRAW_SPLITS, NormalizationStats, TrajectoryWindowDataset

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "CellStats",
    "EpisodeResult",
    "GATE_RATIO",
    "GateVerdict",
    "HELD_OUT_DRAWS",
    "TrainResult",
    "WilcoxonResult",
    "build_training_dataset",
    "config_artifact",
    "env_settings_from_manifest",
    "expected_reported_steps",
    "evaluate_arm",
    "gate_verdict",
    "load_gate_checkpoint",
    "main",
    "mean_ci95",
    "plateau_reached",
    "policy_source_for",
    "runtime_provenance",
    "stack_dataset",
    "train_dt",
    "wilcoxon_signed_rank",
    "window_means",
    "write_json_atomic",
]

ARTIFACT_FORMAT_VERSION = "p4-gate/1.0"

#: The registered held-out evaluation pool, used whole so no slice can be selected.
HELD_OUT_DRAWS: tuple[int, ...] = tuple(
    range(DRAW_SPLITS["heldout"][0], DRAW_SPLITS["heldout"][1] + 1)
)

#: The registered gate ratio against the best online policy.
GATE_RATIO = 1.05

#: Frozen training configuration, declared in ``docs/plans/p4.md`` before any training existed.
DECLARED_GRADIENT_STEPS = 20_000
DECLARED_RAISE_TO = 40_000
PLATEAU_WINDOW = 2_000
PLATEAU_TOLERANCE = 0.05
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 1_000
GRAD_CLIP = 0.25
CONTEXT_LENGTH = 20
TRAINING_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)


@dataclass(frozen=True)
class EpisodeResult:
    """One rollout of one arm on one draw: the primary metric and A5's companion."""

    arm: str
    seed: int | None
    draw_id: int
    att_horizon: float
    horizon_vehicle_count: float
    episode_reward: float


@dataclass(frozen=True)
class CellStats:
    """Mean with a 95% CI, matching ``offline/att_ladder.py``'s convention (``ddof=1``)."""

    n: int
    mean: float
    std: float
    ci95: float


@dataclass(frozen=True)
class WilcoxonResult:
    """Paired signed-rank test over shared draws; ties get average ranks, zeros are dropped."""

    w_plus: float
    w_minus: float
    statistic: float
    n_used: int
    n_zero: int
    z: float
    p_value: float


@dataclass(frozen=True)
class GateVerdict:
    """The two registered inequalities, evaluated explicitly.  ``<=``: equality passes."""

    att_madt: float
    att_maxpressure: float
    att_best_online: float
    ratio: float
    threshold_online: float
    gate_a: bool
    gate_b: bool
    passed: bool


@dataclass(frozen=True)
class TrainResult:
    """One trained seed: the reported checkpoint and the curve that chose its length."""

    seed: int
    gradient_steps: int
    declared_gradient_steps: int
    losses: tuple[float, ...]
    window_means: tuple[float, ...]
    plateaued: bool
    checkpoint_path: str
    seconds: float


# ----------------------------------------------------------------------
# Descriptives and the registered paired test
# ----------------------------------------------------------------------


def mean_ci95(values: Sequence[float]) -> CellStats:
    """Mean, sample standard deviation (``ddof=1``) and a 95% CI half-width."""
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        raise ValueError("mean_ci95 received no values")
    count = int(data.size)
    if count == 1:
        return CellStats(n=1, mean=float(data[0]), std=0.0, ci95=0.0)
    std = float(data.std(ddof=1))
    return CellStats(
        n=count,
        mean=float(data.mean()),
        std=std,
        ci95=1.96 * std / math.sqrt(count),
    )


def _average_ranks(values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Ranks of *values* with ties sharing their average rank; also the tie-group sizes."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ties: list[int] = []
    position = 0
    while position < order.size:
        end = position
        while end + 1 < order.size and values[order[end + 1]] == values[order[position]]:
            end += 1
        group = order[position : end + 1]
        ranks[group] = (position + end) / 2.0 + 1.0
        ties.append(int(group.size))
        position = end + 1
    return ranks, ties


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via ``erfc``; stdlib only, and exact enough far into the tail."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> WilcoxonResult:
    """Two-sided paired Wilcoxon signed-rank test of ``x - y``.

    Zero differences are dropped (Wilcoxon's own convention), ties share an average rank, and
    the p-value uses the normal approximation with the tie correction and a continuity
    correction.  The approximation is what the registered analysis plan needs at n >= 20; it is
    reported with ``n_used`` so a caller can see how many pairs it actually rested on.

    The test is **scale-free**: it reads the signs and the rank order of the differences, not
    their size.  A tiny but perfectly consistent shift is therefore highly significant, which is
    a property of the test and not a defect in it.
    """
    left = np.asarray(list(x), dtype=np.float64)
    right = np.asarray(list(y), dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(
            f"wilcoxon_signed_rank needs paired samples of equal length, got "
            f"{left.shape[0]} and {right.shape[0]}"
        )
    if left.size == 0:
        raise ValueError("wilcoxon_signed_rank received no pairs")

    difference = left - right
    n_zero = int(np.count_nonzero(difference == 0.0))
    nonzero = difference[difference != 0.0]
    n_used = int(nonzero.size)
    if n_used == 0:
        return WilcoxonResult(
            w_plus=0.0, w_minus=0.0, statistic=0.0, n_used=0, n_zero=n_zero, z=0.0, p_value=1.0
        )

    ranks, ties = _average_ranks(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    statistic = min(w_plus, w_minus)

    expected = n_used * (n_used + 1) / 4.0
    variance = n_used * (n_used + 1) * (2 * n_used + 1) / 24.0
    variance -= sum(size**3 - size for size in ties) / 48.0
    if variance <= 0.0:
        return WilcoxonResult(
            w_plus=w_plus,
            w_minus=w_minus,
            statistic=statistic,
            n_used=n_used,
            n_zero=n_zero,
            z=0.0,
            p_value=1.0,
        )

    z = (statistic - expected + 0.5) / math.sqrt(variance)
    p_value = min(1.0, 2.0 * _normal_cdf(z))
    return WilcoxonResult(
        w_plus=w_plus,
        w_minus=w_minus,
        statistic=statistic,
        n_used=n_used,
        n_zero=n_zero,
        z=z,
        p_value=p_value,
    )


def gate_verdict(
    att_madt: float,
    att_maxpressure: float,
    att_best_online: float,
    ratio: float = GATE_RATIO,
) -> GateVerdict:
    """Evaluate both registered inequalities.  ``<=`` exactly as registered: equality passes."""
    threshold = float(att_best_online) * float(ratio)
    gate_a = float(att_madt) <= float(att_maxpressure)
    gate_b = float(att_madt) <= threshold
    return GateVerdict(
        att_madt=float(att_madt),
        att_maxpressure=float(att_maxpressure),
        att_best_online=float(att_best_online),
        ratio=float(ratio),
        threshold_online=threshold,
        gate_a=bool(gate_a),
        gate_b=bool(gate_b),
        passed=bool(gate_a and gate_b),
    )


# ----------------------------------------------------------------------
# The declared budget and its single pre-declared raise
# ----------------------------------------------------------------------


def window_means(losses: Sequence[float], window: int) -> tuple[float, ...]:
    """Mean training loss per consecutive window of *window* gradient steps."""
    values = np.asarray(list(losses), dtype=np.float64)
    size = int(window)
    if size < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if values.size == 0 or values.size % size:
        raise ValueError(
            f"a curve of {values.size} steps does not divide into windows of {size}; the "
            "plateau criterion is defined on whole windows"
        )
    return tuple(float(chunk.mean()) for chunk in values.reshape(-1, size))


def plateau_reached(means: Sequence[float], tolerance: float = PLATEAU_TOLERANCE) -> bool:
    """Whether the last two consecutive relative changes are below *tolerance*.

    Mirrors amendment A3's criterion, applied to the training loss instead of the training
    return, and reads the training curve only -- no evaluation number can enter this decision.
    """
    values = [float(v) for v in means]
    if len(values) < 3:
        raise ValueError(
            f"the plateau criterion needs at least three windows, got {len(values)}"
        )
    for current, previous in ((values[-1], values[-2]), (values[-2], values[-3])):
        if previous == 0.0:
            return False
        if abs(current - previous) / abs(previous) >= tolerance:
            return False
    return True


# ----------------------------------------------------------------------
# The corpus side
# ----------------------------------------------------------------------


def env_settings_from_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Env settings for evaluation, read out of the collection manifest rather than restated."""
    from experiments.config import SETTING_DEFAULTS

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    metadata = payload.get("run_metadata", {})
    required = (
        "max_steps",
        "delta_time",
        "control_mode",
        "state_features",
        "global_reward_fn",
        "local_reward_fn",
        "global_reward_weight",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(
            f"{manifest_path}: run_metadata is missing {missing}; the evaluation env cannot be "
            "reconstructed from it, and restating the settings here is exactly the drift this "
            "function exists to prevent"
        )
    if metadata["local_reward_fn"] is None:
        raise ValueError(
            f"{manifest_path}: the collection ran with local_reward_fn=None, so its info dicts "
            "carry no per-intersection reward and a Decision Transformer's return-to-go cannot "
            "be advanced during evaluation"
        )

    settings = dict(SETTING_DEFAULTS)
    settings.update(
        {
            "max_steps": int(metadata["max_steps"]),
            "delta_time": int(metadata["delta_time"]),
            "control_mode": str(metadata["control_mode"]),
            "state_features": list(metadata["state_features"]),
            "global_reward_fn": metadata["global_reward_fn"],
            "local_reward_fn": metadata["local_reward_fn"],
            "global_reward_weight": float(metadata["global_reward_weight"]),
            "metrics": metadata.get("metrics"),
            "thread_num": 1,
            "gui": False,
        }
    )
    return settings


def build_training_dataset(
    dataset_dirs: Sequence[str | Path], context_length: int
) -> TrajectoryWindowDataset:
    """The training-split window dataset over *dataset_dirs*, with statistics fitted there.

    ``split="train"`` is not a default being accepted: it is the mechanism that makes a
    held-out draw raise instead of quietly entering the corpus.
    """
    return TrajectoryWindowDataset(
        [Path(d) for d in dataset_dirs],
        context_length=int(context_length),
        split="train",
        normalize=True,
    )


def stack_dataset(
    dataset: TrajectoryWindowDataset, group: tuple[int, int] | None = None
) -> dict[str, torch.Tensor]:
    """Materialise one group's windows into contiguous tensors, in dataset order.

    Built by iterating the dataset's own ``__getitem__``, so the loader stays the single
    definition of what a window is; the tests re-check a random sample of rows against
    ``dataset[i]``.

    **One group only, and the group is explicit.**  C6 forbids padding across intersections, so
    windows of different ``(state_dim, n_actions)`` cannot share these tensors.  ``group=None``
    is accepted only when the dataset has exactly one group -- which is the P4 scenario -- and
    raises otherwise rather than silently taking the first.  ``cf_cologne3`` has three groups,
    so a caller there must say which one it means.

    The returned mapping carries an extra ``item_index`` tensor: row ``r`` of every stacked
    tensor is ``dataset[item_index[r]]``.  Without it a caller could not map a stacked row back
    to its provenance once the rows are a subset of the dataset.
    """
    groups = dataset.groups
    if group is None:
        if len(groups) != 1:
            raise ValueError(
                f"this dataset has {len(groups)} (state_dim, n_actions) groups "
                f"{sorted(groups)}; pass group= to say which one to stack. C6 forbids padding "
                "across intersections, so each group needs its own tensors"
            )
        group = next(iter(groups))
    key = (int(group[0]), int(group[1]))
    if key not in groups:
        raise ValueError(f"group {key} is not present; this dataset has {sorted(groups)}")

    indices = list(groups[key])
    items = [dataset[i] for i in indices]
    stacked = {name: torch.stack([item[name] for item in items]) for name in items[0]}
    stacked["item_index"] = torch.tensor(indices, dtype=torch.int64)
    return stacked


def runtime_provenance(
    measurement_git_commits: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Device, library and repo state -- the state that determines float reduction order.

    P8.0 finding N8: the provenance file omitted exactly this, in the document whose MAPPO rows
    were attributed to reduction order.

    **Measurement provenance versus written-at provenance** (``DEFERRED`` 39, extended additively
    on 2026-08-13 under the authorisation in ``BRIEF_15`` section 8 and ``PROJECT_PLAN`` section 6).
    ``git_commit`` records ``git rev-parse HEAD`` **at write time** and keeps that meaning exactly,
    so nothing that reads it changes behaviour.  Two facts make it insufficient on its own:

    * an artifact can never be committed at the commit it records -- committing moves ``HEAD``,
      the same fixed point as a document that hashes itself;
    * it is already wrong for a chunked campaign.  Measured 2026-08-12: ``output/p4_4/gate_a.json``
      carries ``738884b`` while the three ``eval_*.json`` carry ``c13aaa9``, so **no single commit
      produced all the measurements** and the report's single value means "when the report was
      assembled".

    So three keys are ADDED and none is changed: ``written_at_git_commit`` names the write-time
    commit unambiguously (``None`` when git is unavailable, which the legacy ``git_commit`` reports
    as ``""`` and continues to), ``measurement_git_commits`` carries the sorted, de-duplicated
    commits **of the inputs** -- empty when the caller supplies none, which is every pre-existing
    call site -- and ``unreachable_measurement_commits`` carries the ones that failed the check
    below.

    **REACHABILITY IS CHECKED, and that check is the point** (P4.3 review, finding F6).  The first
    artifact written under this schema recorded ``8b647a40...``, a commit that had been amended away
    and resolved only because git had not yet garbage-collected it: **a provenance field naming an
    object a future reader cannot check out defeats the whole deliverable.**  Every supplied commit
    is therefore tested with ``git merge-base --is-ancestor <commit> HEAD`` at write time.  Ones
    that fail are **moved, not dropped** -- silently discarding them would hide exactly the
    situation this field exists to expose -- and the artifact records both lists.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:  # pragma: no cover - git is present in this repo
        commit = ""
    reachable, unreachable = _partition_reachable_commits(measurement_git_commits or ())
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "torch_num_threads": int(torch.get_num_threads()),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "git_commit": commit,
        "written_at_git_commit": commit or None,
        "measurement_git_commits": reachable,
        "unreachable_measurement_commits": unreachable,
    }


def _partition_reachable_commits(commits: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split *commits* into those reachable from ``HEAD`` and those that are not.

    ``git merge-base --is-ancestor A B`` exits 0 when ``A`` is an ancestor of ``B``, 1 when it is
    not, and 128 when the object does not exist at all -- an amended-away or garbage-collected
    commit hits one of the latter two, and both belong in the unreachable list.
    """
    reachable: list[str] = []
    unreachable: list[str] = []
    for candidate in sorted({str(c) for c in commits if str(c)}):
        try:
            result = subprocess.run(
                ["git", "merge-base", "--is-ancestor", candidate, "HEAD"],
                capture_output=True,
                check=False,
            ).returncode
        except OSError:  # pragma: no cover - git is present in this repo
            result = 1
        (reachable if result == 0 else unreachable).append(candidate)
    return reachable, unreachable


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def train_dt(
    stacked: dict[str, torch.Tensor],
    *,
    state_dim: int,
    n_actions: int,
    seed: int,
    declared_gradient_steps: int,
    raise_to: int | None,
    context_length: int,
    batch_size: int,
    device: torch.device,
    checkpoint_path: str | Path,
    stats: NormalizationStats,
    scenario_id: str,
    target_rtg: float,
    rtg_scale: float,
    provenance: dict[str, Any],
    log_every: int = 0,
) -> TrainResult:
    """Train one seed for exactly *declared_gradient_steps* steps and save the checkpoint.

    ``raise_to`` is recorded, never acted on here: the registered criterion is all-or-nothing
    across seeds, so the raise is orchestrated by the caller once every seed's curve exists.
    """
    from agent.DTAgent import DecisionTransformer, DTConfig
    from agent.utils.utils import Utils

    total = int(declared_gradient_steps)
    if total < 1:
        raise ValueError(f"declared_gradient_steps must be >= 1, got {declared_gradient_steps}")
    count = int(stacked["state"].shape[0])
    if count < 1:
        raise ValueError("the stacked dataset is empty")

    Utils.seed_everything(int(seed), seed_python_random=False)
    config = DTConfig(
        state_dim=int(state_dim),
        n_actions=int(n_actions),
        context_length=int(context_length),
        max_ep_len=int(stacked["timestep"].max()) + 1,
    )
    model = DecisionTransformer(config).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    warmup = min(WARMUP_STEPS, max(1, total // 2))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: min(1.0, (step + 1) / warmup)
    )

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
        logits = model(
            tensors["rtg"][index] / float(rtg_scale),
            tensors["state"][index],
            action,
            tensors["timestep"][index],
            tensors["attention_mask"][index],
            tensors["avail_mask"][index],
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
                f"  seed {seed} step {step + 1}/{total} loss {np.mean(losses[-log_every:]):.5f}",
                flush=True,
            )
    seconds = time.time() - started

    window = min(PLATEAU_WINDOW, total)
    while total % window:
        window -= 1
    means = window_means(losses, window)
    plateaued = plateau_reached(means) if len(means) >= 3 else False

    destination = Path(checkpoint_path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {destination.parent}; nothing is created here"
        )
    torch.save(
        {
            "format_version": "dt-checkpoint/1.0",
            "config": config.to_json_obj(),
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "target_rtg": float(target_rtg),
            "rtg_scale": float(rtg_scale),
            "normalise": True,
            "scenario_id": str(scenario_id),
            "stats": stats.to_json_obj(),
            "intersection_ids": [],
            "provenance": {
                **dict(provenance),
                "seed": int(seed),
                "gradient_steps": int(total),
                "declared_gradient_steps": int(declared_gradient_steps),
                "raise_to": None if raise_to is None else int(raise_to),
                "batch_size": int(batch_size),
                "context_length": int(context_length),
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_steps": int(warmup),
                "grad_clip": GRAD_CLIP,
                "device": str(device),
                "window_means": list(means),
                "plateaued": bool(plateaued),
                "runtime": runtime_provenance(),
            },
        },
        destination,
    )
    return TrainResult(
        seed=int(seed),
        gradient_steps=total,
        declared_gradient_steps=int(declared_gradient_steps),
        losses=tuple(losses),
        window_means=means,
        plateaued=bool(plateaued),
        checkpoint_path=str(destination),
        seconds=float(seconds),
    )


def expected_reported_steps(training: dict[str, Any], *, declared: int) -> int:
    """The only step count this evaluation may report, derived from the DECLARATION.

    Previously the evaluator read ``reported_gradient_steps`` straight out of the training
    artifact and handed it to :func:`load_gate_checkpoint`, which then compared a number to
    itself: the guard could catch a stale checkpoint file but never "you reported a budget other
    than the declared one", because both sides came from the same run.  Here the declaration is
    the input and the artifact is the thing being checked:

    * the artifact's own ``declared_gradient_steps`` must equal *declared*;
    * the reported count must be ``declared``, or ``raise_to`` and only if the artifact records
      that the single pre-declared raise was actually taken.

    Anything else raises, so a run that quietly trained longer cannot be evaluated at all.
    """
    recorded_declared = int(training["declared_gradient_steps"])
    if recorded_declared != int(declared):
        raise ValueError(
            f"the training artifact was produced under a declared budget of "
            f"{recorded_declared} steps but this evaluation was asked for {int(declared)}; "
            "PREREGISTRATION.md section 6 fixes the budget before training, so these cannot differ"
        )
    reported = int(training["reported_gradient_steps"])
    if reported == recorded_declared:
        return reported
    raise_to = training.get("raise_to")
    if bool(training.get("raise_taken")) and raise_to is not None and reported == int(raise_to):
        return reported
    raise ValueError(
        f"the training artifact reports {reported} gradient steps, which is neither the declared "
        f"{recorded_declared} nor the single pre-declared raise to {raise_to} "
        f"(raise_taken={training.get('raise_taken')}); no checkpoint from it may be reported"
    )


def load_gate_checkpoint(
    gym_env: Any, path: str | Path, declared_gradient_steps: int, device: str | None = None
) -> DTAgent:
    """Load a DT checkpoint, refusing one whose recorded step count is not the declared one.

    This is the mechanical form of "no online model selection": a checkpoint saved at a
    different step -- an earlier one that scored better, say -- cannot be evaluated by this
    path at all.  The check runs before the agent is constructed, so a refusal builds nothing.
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
    return DTAgent.from_checkpoint(gym_env, str(path), device=device)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------


def evaluate_arm(
    *,
    arm: str,
    seed: int | None,
    draw_ids: Sequence[int],
    config_for_draw: Callable[[int], Path],
    env_settings: dict[str, Any],
    scenario_id: str,
    choose_action_factory: Callable[[Any], Callable[[Any, dict[str, Any]], np.ndarray]],
    engine_seed: int,
) -> list[EpisodeResult]:
    """Roll one arm over *draw_ids*, one episode per draw, through ``horizon_rollout``.

    ``episodes=1`` per draw is deliberate: at that setting the reader's ``final_vehicle_count``
    is that single episode's value, so P8.0 finding B2 -- a last-episode ``final_completed``
    mixed with a mean ``final_vehicle_count`` -- cannot arise.  ``final_completed`` is not read.
    """
    from experiments.config import EnvSpec
    from experiments.envs import make_env
    from offline.horizon_metric import horizon_rollout

    results: list[EpisodeResult] = []
    for draw_id in draw_ids:
        config_path = Path(config_for_draw(int(draw_id)))
        if not config_path.is_file():
            raise FileNotFoundError(
                f"draw {draw_id} has no materialised sim config at {config_path}; run "
                "offline.materialise_draws for the held-out pool first"
            )
        env = make_env(
            EnvSpec(
                id=scenario_id,
                backend="cityflow",
                paths={"config": str(config_path)},
                settings=env_settings,
            )
        )
        try:
            rollout = horizon_rollout(
                env, choose_action_factory(env), episodes=1, seed=int(engine_seed)
            )
        finally:
            env.close()
        results.append(
            EpisodeResult(
                arm=arm,
                seed=seed,
                draw_id=int(draw_id),
                att_horizon=float(rollout.per_episode_horizon[0]),
                horizon_vehicle_count=float(rollout.final_vehicle_count),
                episode_reward=float(rollout.episode_reward),
            )
        )
    return results


# ----------------------------------------------------------------------
# Artifacts
# ----------------------------------------------------------------------


def config_artifact(
    checkpoint_path: str | Path, training: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The frozen configuration, READ BACK OUT OF A TRAINED CHECKPOINT.

    ``PREREGISTRATION.md`` section 6.2 requires the configuration tuned on hangzhou_1x1 to be
    recorded "as a committed artifact, not as prose", because it binds every later scenario,
    tier, perturbation and backend.  Deriving it from a checkpoint rather than from this
    module's constants is deliberate: the constants describe what the code would do next time,
    the checkpoint describes what actually produced the reported number, and only the second
    one is evidence.

    **The budget block comes from the TRAINING artifact when one is supplied, and it must be.**
    A checkpoint's own ``declared_gradient_steps`` records the budget of the call that wrote it,
    so after the single pre-declared raise every checkpoint says ``40000`` and the fact that
    ``20000`` was the declaration survives only in ``p4_training.json``.  Reading the budget out
    of a checkpoint alone would therefore state, in the artifact that fixes the project's
    configuration, that we declared what we actually raised to.
    """
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    provenance = payload["provenance"]
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "the frozen P4 configuration (PREREGISTRATION.md section 6.2); tuned on "
            "hangzhou_1x1 #1 only and applied unchanged to every later scenario, tier, "
            "perturbation and backend"
        ),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": _sha256_file(checkpoint_path),
        "architecture": payload["config"],
        "optimisation": {
            key: provenance[key]
            for key in (
                "batch_size",
                "learning_rate",
                "weight_decay",
                "warmup_steps",
                "grad_clip",
                "context_length",
            )
        },
        "budget": {
            "declared_gradient_steps": (
                int(training["declared_gradient_steps"])
                if training is not None
                else provenance["declared_gradient_steps"]
            ),
            "raise_to": (
                training.get("raise_to") if training is not None else provenance["raise_to"]
            ),
            "raise_taken": (
                bool(training["raise_taken"]) if training is not None else None
            ),
            "reported_gradient_steps": provenance["gradient_steps"],
            "budget_source": "training artifact" if training is not None else "checkpoint only",
            "plateau_window": PLATEAU_WINDOW,
            "plateau_tolerance": PLATEAU_TOLERANCE,
        },
        "conditioning": {
            "target_rtg": payload["target_rtg"],
            "rtg_scale": payload["rtg_scale"],
            "rule": "target_rtg = max episode return in the training split; rtg_scale = max|rtg|",
            "action_selection_at_evaluation": "greedy argmax over masked logits",
        },
        "seeds": list(TRAINING_SEEDS),
        "normalisation": {
            "fitted_on_split": payload["stats"]["split"],
            "fitted_on_draw_ids": payload["stats"]["draw_ids"],
        },
        "tier": provenance.get("tier"),
        "runtime": provenance.get("runtime", {}),
    }


def _sha256_file(path: str | Path) -> str:
    """sha256 of a file, so an artifact can name the exact weights behind it."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(payload: dict[str, Any], path: str | Path) -> None:
    """Write *payload* as JSON atomically, after validation, into an existing directory.

    Filesystem-mutation barrier: a missing parent directory raises **before** anything is
    created, so a refused write leaves any previous file untouched and creates no directories.
    """
    destination = Path(path)
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(
            f"destination directory does not exist: {parent}; this function never creates one"
        )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    handle, tmp_name = tempfile.mkstemp(dir=str(parent), prefix=".p4-", suffix=".json.tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(tmp_name, destination)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


#: How each arm's actions were produced.  Queue item 0b requires this to be machine-readable and
#: TRUE; a heuristic has no checkpoint, so calling it one is a provenance error, not a label.
POLICY_SOURCES: dict[str, str] = {
    "maxpressure": "deterministic_heuristic",
    "fixedtime": "deterministic_heuristic",
    "random": "stochastic_heuristic",
}


def policy_source_for(arm: str) -> str:
    """``"checkpoint"`` unless the arm is a heuristic with no weights to load."""
    return POLICY_SOURCES.get(arm, "checkpoint")


def _cell(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    """One reported cell: the primary metric, A5's unconditional companion and the draw ids.

    ``policy_source`` is derived from the arm name here rather than patched by the caller: it
    was previously hardcoded to ``"checkpoint"`` and corrected only in the baselines path, so
    the gate artifact -- the one later tasks reuse -- claimed MaxPressure ran from a checkpoint.
    """
    att = mean_ci95([r.att_horizon for r in results])
    vehicles = mean_ci95([r.horizon_vehicle_count for r in results])
    arms = sorted({r.arm for r in results})
    if len(arms) != 1:
        raise ValueError(f"a cell must describe one arm, got {arms}")
    return {
        "n_episodes": len(results),
        "att_horizon_mean": att.mean,
        "att_horizon_std": att.std,
        "att_horizon_ci95": att.ci95,
        "horizon_vehicle_count_mean": vehicles.mean,
        "horizon_vehicle_count_std": vehicles.std,
        "draw_ids": sorted({r.draw_id for r in results}),
        "seeds": sorted({r.seed for r in results if r.seed is not None}),
        "policy_source": policy_source_for(arms[0]),
    }


def _per_draw_means(results: Sequence[EpisodeResult]) -> dict[int, float]:
    """Mean ``att_horizon`` per draw, averaging over seeds -- the paired unit for Wilcoxon."""
    buckets: dict[int, list[float]] = {}
    for result in results:
        buckets.setdefault(result.draw_id, []).append(result.att_horizon)
    return {draw: float(np.mean(values)) for draw, values in buckets.items()}


def _paired(
    left: Sequence[EpisodeResult], right: Sequence[EpisodeResult]
) -> tuple[list[float], list[float], list[int]]:
    """Shared draws only (A5 point 3): a comparison without them is void, not approximate."""
    a = _per_draw_means(left)
    b = _per_draw_means(right)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError(
            "no shared draws between the two arms; amendment A5 point 3 makes this comparison "
            "void, and it must not be reported"
        )
    return [a[d] for d in shared], [b[d] for d in shared], shared


def build_parser() -> argparse.ArgumentParser:
    """CLI: ``baselines`` (pre-training thresholds), ``train``, ``evaluate``."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.dt_gate",
        description="Train the P4 Decision Transformer and evaluate the P4.2 gate.",
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
    parser.add_argument("--checkpoint-dir", default="output/p4_dt")
    parser.add_argument("--device", default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    baselines = sub.add_parser("baselines", help="measure the thresholds before any training")
    baselines.add_argument("--maxpressure", action="store_true", default=True)
    baselines.add_argument(
        "--mappo-checkpoint",
        action="append",
        default=[],
        metavar="TIER=PATH",
        help="repeatable, e.g. mappo1000=/path/cf_hz1x1__mappo__seed101.pt",
    )

    train = sub.add_parser("train", help="train every seed to the declared step count")
    train.add_argument("--dataset-dir", action="append", required=True)
    train.add_argument("--steps", type=int, default=DECLARED_GRADIENT_STEPS)
    train.add_argument("--raise-to", type=int, default=DECLARED_RAISE_TO)
    train.add_argument("--log-every", type=int, default=2000)

    evaluate = sub.add_parser("evaluate", help="evaluate the saved checkpoints and emit the gate")
    evaluate.add_argument("--steps", type=int, default=DECLARED_GRADIENT_STEPS)
    evaluate.add_argument("--thresholds", required=True, help="the baselines artifact")
    evaluate.add_argument(
        "--draws",
        default="heldout",
        choices=["heldout", "training"],
        help="'heldout' is the gate; 'training' is the labelled draws-1-200 secondary reading",
    )
    evaluate.add_argument("--training-draws", type=int, nargs=2, default=[1, 200])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    from offline.materialise_draws import draw_config_path

    args = build_parser().parse_args(argv)
    settings = env_settings_from_manifest(args.manifest)
    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"--out-dir does not exist: {out_dir}")

    def config_for_draw(draw_id: int) -> Path:
        return draw_config_path(args.scenario_key, draw_id, out_root=args.draws_root)

    if args.command == "baselines":
        return _run_baselines(args, settings, config_for_draw, out_dir)
    if args.command == "train":
        return _run_train(args, out_dir)
    return _run_evaluate(args, settings, config_for_draw, out_dir)


def _maxpressure_factory(env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
    from algorithms.max_pressure import MaxPressureAgent

    agent = MaxPressureAgent(env)
    return lambda _env, info: agent.act(info)


def _mappo_factory(path: str, device: str | None) -> Callable[[Any], Any]:
    def factory(env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
        from agent.MAPPOAgent import MAPPOAgent

        agent = MAPPOAgent(env, device=device, seed=0)
        agent.load(path)
        return lambda _env, info: agent.act(info, explore=False, update_memory=False)

    return factory


def _run_baselines(
    args: argparse.Namespace,
    settings: dict[str, Any],
    config_for_draw: Callable[[int], Path],
    out_dir: Path,
) -> int:
    """Measure every baseline arm on the held-out pool.  No DT exists when this runs."""
    draws = list(HELD_OUT_DRAWS)
    arms: dict[str, list[EpisodeResult]] = {}

    print(f"MaxPressure over {len(draws)} held-out draws", flush=True)
    arms["maxpressure"] = evaluate_arm(
        arm="maxpressure",
        seed=None,
        draw_ids=draws,
        config_for_draw=config_for_draw,
        env_settings=settings,
        scenario_id=args.scenario_id,
        choose_action_factory=_maxpressure_factory,
        engine_seed=args.engine_seed,
    )
    for spec in args.mappo_checkpoint:
        tier, _, path = spec.partition("=")
        seed = int(Path(path).stem.rsplit("seed", 1)[-1])
        print(f"{tier} seed {seed} over {len(draws)} held-out draws", flush=True)
        arms.setdefault(tier, []).extend(
            evaluate_arm(
                arm=tier,
                seed=seed,
                draw_ids=draws,
                config_for_draw=config_for_draw,
                env_settings=settings,
                scenario_id=args.scenario_id,
                choose_action_factory=_mappo_factory(path, args.device),
                engine_seed=args.engine_seed,
            )
        )

    cells = {name: _cell(results) for name, results in arms.items()}
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": "thresholds measured BEFORE the first gradient step",
        "evaluation_pool": "registered held-out draws 1000-1099 (PREREGISTRATION.md D4)",
        "engine_seed": int(args.engine_seed),
        "env_settings": {k: v for k, v in settings.items() if k != "compare_with"},
        "cells": cells,
        "episodes": [asdict(r) for results in arms.values() for r in results],
        "runtime": runtime_provenance(),
    }
    write_json_atomic(payload, out_dir / "p4_heldout_thresholds.json")
    for name, cell in sorted(cells.items()):
        print(
            f"  {name:14s} att_horizon {cell['att_horizon_mean']:8.3f} "
            f"+/- {cell['att_horizon_ci95']:5.3f}  "
            f"vehicle_count {cell['horizon_vehicle_count_mean']:7.2f}  "
            f"n={cell['n_episodes']}",
            flush=True,
        )
    return 0


def _run_train(args: argparse.Namespace, out_dir: Path) -> int:
    """Train every seed, then apply the all-seeds plateau rule once, exactly once."""
    from agent.utils.utils import Utils

    dataset = build_training_dataset(args.dataset_dir, CONTEXT_LENGTH)
    stacked = stack_dataset(dataset)                     # one group, or it raises
    (state_dim, n_actions) = next(iter(dataset.groups))
    scenario_id = dataset.episode_records[0].scenario_id
    ix_id = dataset.episode_records[0].ix_ids[0]
    summary = dataset.stats.rtg[scenario_id][ix_id]
    rtg_scale = max(abs(summary.min), abs(summary.max))
    target_rtg = max(
        float(stacked["rtg"][index][-1, 0])
        for index in range(len(dataset))
        if dataset.item_meta(index).t == 0
    )

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
        f"target_rtg {target_rtg}  rtg_scale {rtg_scale}  device {device}",
        flush=True,
    )

    def run(steps: int) -> list[TrainResult]:
        return [
            train_dt(
                stacked,
                state_dim=state_dim,
                n_actions=n_actions,
                seed=seed,
                declared_gradient_steps=steps,
                raise_to=args.raise_to,
                context_length=CONTEXT_LENGTH,
                batch_size=BATCH_SIZE,
                device=device,
                checkpoint_path=checkpoints / f"dt_seed{seed}.pt",
                stats=dataset.stats,
                scenario_id=scenario_id,
                target_rtg=target_rtg,
                rtg_scale=rtg_scale,
                provenance=provenance,
                log_every=args.log_every,
            )
            for seed in TRAINING_SEEDS
        ]

    results = run(int(args.steps))
    raised = False
    if not all(result.plateaued for result in results) and args.raise_to:
        # A3's strict aggregation: all seeds or none, decided on the training curve alone.
        print(
            f"plateau NOT reached by all seeds at {args.steps} steps "
            f"({sum(r.plateaued for r in results)}/{len(results)}); taking the single "
            f"pre-declared raise to {args.raise_to}",
            flush=True,
        )
        results = run(int(args.raise_to))
        raised = True

    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": "training record; the reported checkpoint is the declared step count",
        "declared_gradient_steps": int(args.steps),
        "raise_to": int(args.raise_to) if args.raise_to else None,
        "raise_taken": raised,
        "reported_gradient_steps": int(results[0].gradient_steps),
        "target_rtg": float(target_rtg),
        "rtg_scale": float(rtg_scale),
        "training_draw_ids": list(dataset.stats.draw_ids),
        "seeds": [
            {
                "seed": r.seed,
                "gradient_steps": r.gradient_steps,
                "plateaued": r.plateaued,
                "window_means": list(r.window_means),
                "final_loss": r.losses[-1],
                "seconds": r.seconds,
                "checkpoint": r.checkpoint_path,
            }
            for r in results
        ],
        "runtime": runtime_provenance(),
    }
    write_json_atomic(payload, out_dir / "p4_training.json")
    write_json_atomic(
        config_artifact(results[0].checkpoint_path, payload), out_dir / "p4_dt_config.json"
    )
    return 0


def _run_evaluate(
    args: argparse.Namespace,
    settings: dict[str, Any],
    config_for_draw: Callable[[int], Path],
    out_dir: Path,
) -> int:
    """Evaluate the declared checkpoints and emit the gate verdict."""
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    training = json.loads((out_dir / "p4_training.json").read_text(encoding="utf-8"))
    steps = expected_reported_steps(training, declared=int(args.steps))
    if args.draws == "heldout":
        draws = list(HELD_OUT_DRAWS)
    else:
        low, high = args.training_draws
        draws = list(range(int(low), int(high) + 1))

    leaked = sorted(set(draws) & set(training["training_draw_ids"]))
    if args.draws == "heldout" and leaked:
        raise ValueError(
            f"evaluation draws {leaked[:5]} are also training draws; the held-out pool is not "
            "held out and no number from this run may be reported"
        )

    def dt_factory(path: str) -> Callable[[Any], Any]:
        def factory(env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
            agent = load_gate_checkpoint(env, path, steps, device=args.device)
            return lambda _env, info: agent.act(info, explore=False, update_memory=True)

        return factory

    madt: list[EpisodeResult] = []
    for entry in training["seeds"]:
        print(f"MADT seed {entry['seed']} over {len(draws)} draws", flush=True)
        madt.extend(
            evaluate_arm(
                arm="madt",
                seed=int(entry["seed"]),
                draw_ids=draws,
                config_for_draw=config_for_draw,
                env_settings=settings,
                scenario_id=args.scenario_id,
                choose_action_factory=dt_factory(entry["checkpoint"]),
                engine_seed=args.engine_seed,
            )
        )

    cells: dict[str, Any] = {"madt": _cell(madt)}
    payload: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "evaluation_pool": args.draws,
        "draw_ids": sorted(draws),
        "reported_gradient_steps": steps,
        "declared_gradient_steps": int(args.steps),
        "engine_seed": int(args.engine_seed),
        "env_settings": {k: v for k, v in settings.items() if k != "compare_with"},
        "checkpoints": {
            str(entry["seed"]): {
                "path": entry["checkpoint"],
                "sha256": _sha256_file(entry["checkpoint"]),
            }
            for entry in training["seeds"]
        },
        "cells": cells,
        "episodes": [asdict(r) for r in madt],
        "runtime": runtime_provenance(),
    }

    if args.draws == "heldout":
        baseline_episodes = [EpisodeResult(**e) for e in thresholds["episodes"]]
        by_arm: dict[str, list[EpisodeResult]] = {}
        for episode in baseline_episodes:
            by_arm.setdefault(episode.arm, []).append(episode)
        for name, results in by_arm.items():
            cells[name] = _cell(results)

        verdict = gate_verdict(
            att_madt=cells["madt"]["att_horizon_mean"],
            att_maxpressure=cells["maxpressure"]["att_horizon_mean"],
            att_best_online=cells["mappo1000"]["att_horizon_mean"],
        )
        payload["gate"] = asdict(verdict)
        payload["gate_verdict"] = "PASS" if verdict.passed else "FAIL"
        payload["wilcoxon"] = {}
        for name in sorted(by_arm):
            left, right, shared = _paired(madt, by_arm[name])
            payload["wilcoxon"][f"madt_vs_{name}"] = {
                **asdict(wilcoxon_signed_rank(left, right)),
                "n_shared_draws": len(shared),
            }
        print(
            f"\nGATE {'PASS' if verdict.passed else 'FAIL'}\n"
            f"  A: ATT_MADT {verdict.att_madt:.3f} <= MaxPressure "
            f"{verdict.att_maxpressure:.3f}  -> {verdict.gate_a}\n"
            f"  B: ATT_MADT {verdict.att_madt:.3f} <= 1.05 x {verdict.att_best_online:.3f} "
            f"= {verdict.threshold_online:.3f}  -> {verdict.gate_b}",
            flush=True,
        )
        write_json_atomic(payload, out_dir / "p4_gate.json")
    else:
        write_json_atomic(payload, out_dir / "p4_secondary_training_draws.json")
        print(
            f"\nSECONDARY (training draws, in-sample demand): att_horizon "
            f"{cells['madt']['att_horizon_mean']:.3f} "
            f"+/- {cells['madt']['att_horizon_ci95']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
