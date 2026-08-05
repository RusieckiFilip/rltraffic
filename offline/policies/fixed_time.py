"""Fixed-time signal controller -- Tier 1 of the C1 dataset ladder (P2.5).

A deterministic, stateless controller that cycles each intersection's green phases
on a fixed schedule, approximating the plan a real city ships
(``signal_plan_template.txt``).  Registered into :data:`offline.collect.POLICIES`
as ``"fixedtime"``.

Alignment convention (verified against a live CityFlow engine, not assumed)
--------------------------------------------------------------------------
* Under ``acyclic`` control, an action index ``a`` selects the ``a``-th *green*
  file phase, greens taken in ascending file-phase order -- identical to
  ``envs.phase_control.AcyclicPhases.target_phase``.  For the hangzhou 1x1
  scenario this is ``file_phase = a + 1`` (phase 0 is the 5 s clearance;
  phases 1..8 are the 30 s greens).  The live proof is
  ``tests/test_fixed_time_env_mapping.py``.
* The schedule is a **pure function of the decision-step index** ``info["step"]``:
  ``action(t) = cycle[(t // k) % len(cycle)]``.  There is no internal step
  counter, so a caller that skips, retries or evaluates out of order stays in
  sync with simulation time (brief §4.1).  The controller uses no RNG and is
  byte-reproducible.
* Offset: cycle index 0 at step 0.  The env resets to the first green, so step 0
  emits action 0 (same green) and no spurious clearance is played before the
  schedule starts -- matching the shipped plan, which begins on its first green.

On-disk format read here: ``signal_plan_template.txt`` v1 (header line = intersection
id; one integer file-phase index per simulated second; greens and a repeating
clearance phase run-length-encode the cycle).

Control-mode scope: ``acyclic`` only.  The other modes raise -- see
:class:`FixedTimeController`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from envs.phase_control import TRANSITION_PHASE_MAX_DURATION

__all__ = [
    "PlanSchedule",
    "PlanResolution",
    "parse_signal_plan",
    "green_action_phases",
    "equal_split_cycle",
    "FixedTimeController",
    "resolve_plan",
    "fixed_time_provenance",
    "make_fixedtime",
]

PolicyFn = Callable[[dict[str, Any]], np.ndarray]


# ----------------------------------------------------------------------
# Shipped-plan parsing
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PlanSchedule:
    """A parsed ``signal_plan_template.txt`` (format v1).

    ``green_order`` is the sequence of distinct green *file phases* in the order
    the plan cycles them; ``green_durations`` maps each green file phase to its
    configured green seconds; ``clearance_phase`` / ``clearance_duration`` describe
    the repeating clearance between greens.  ``cycle_seconds`` is the wall-clock
    cycle length (sum of green + one clearance per green).
    """

    header: str
    green_order: tuple[int, ...]
    green_durations: Mapping[int, int]
    clearance_phase: int
    clearance_duration: int
    cycle_seconds: int


def parse_signal_plan(text: str) -> PlanSchedule:
    """Parse a ``signal_plan_template.txt`` body (format v1).

    Line 0 is the intersection id; every remaining non-blank line is one integer
    file-phase index per simulated second.  The clearance phase is identified as
    the phase whose longest run is shortest (the 5 s all-red between 30 s greens);
    greens are the remaining phases in first-appearance order.

    Raises ``ValueError`` (message contains ``"signal plan"``) on any malformed
    input -- missing header, no phase rows, a non-integer row, or a body with no
    green phases.  Callers that want a fallback catch this in :func:`resolve_plan`.
    """
    raw = text.splitlines()
    if not raw or raw[0].strip() == "":
        raise ValueError("signal plan is missing its intersection header line")
    header = raw[0].strip()

    phases: list[int] = []
    for line in raw[1:]:
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            phases.append(int(stripped))
        except ValueError:
            raise ValueError(
                f"signal plan row is not an integer file-phase index: {stripped!r}"
            ) from None
    if not phases:
        raise ValueError("signal plan has no phase rows after the header")

    # Run-length encode the per-second sequence.
    segments: list[list[int]] = []
    for phase in phases:
        if segments and segments[-1][0] == phase:
            segments[-1][1] += 1
        else:
            segments.append([phase, 1])

    max_run: dict[int, int] = {}
    for phase, run in segments:
        max_run[phase] = max(max_run.get(phase, 0), run)
    if len(max_run) < 2:
        raise ValueError(
            "signal plan has fewer than two distinct phases; a fixed-time cycle "
            "needs greens and a clearance"
        )
    # Clearance = the phase with the shortest maximum run (5 s vs 30 s greens);
    # ties broken by lowest index so the choice is deterministic.
    clearance_phase = min(max_run, key=lambda p: (max_run[p], p))

    green_order: list[int] = []
    green_durations: dict[int, int] = {}
    clearance_duration = 0
    for phase, run in segments:
        if phase == clearance_phase:
            if clearance_duration == 0:
                clearance_duration = run
            continue
        if phase not in green_durations:
            green_order.append(phase)
            green_durations[phase] = run
    if not green_order:
        raise ValueError("signal plan has no green phases distinct from the clearance")

    cycle_seconds = sum(green_durations.values()) + len(green_order) * clearance_duration
    return PlanSchedule(
        header=header,
        green_order=tuple(green_order),
        green_durations=green_durations,
        clearance_phase=int(clearance_phase),
        clearance_duration=int(clearance_duration),
        cycle_seconds=int(cycle_seconds),
    )


# ----------------------------------------------------------------------
# Green derivation and cycle construction
# ----------------------------------------------------------------------


def green_action_phases(ix: Any) -> list[int]:
    """Green *file phases* for one intersection, ascending == action-index order.

    Mirrors ``AcyclicPhases``: greens are phases whose configured file duration
    exceeds ``TRANSITION_PHASE_MAX_DURATION``.  Falls back to phases with active
    road links when duration metadata is unavailable (lightweight stub envs);
    real CityFlow / SUMO intersections always carry durations.
    """
    num_phases = int(getattr(ix, "num_phases", 0) or 0)
    durations = list(getattr(ix, "phase_durations", None) or [])
    if num_phases > 0 and len(durations) >= num_phases:
        greens = [
            p for p in range(num_phases)
            if float(durations[p]) > TRANSITION_PHASE_MAX_DURATION
        ]
        if greens:
            return greens
    mapping = list(getattr(ix, "phase_roadlink_mapping", None) or [])
    greens = [p for p, links in enumerate(mapping) if links]
    return greens or list(range(num_phases))


def equal_split_cycle(n_greens: int) -> tuple[int, ...]:
    """Equal-split fallback cycle: the greens in ascending action-index order."""
    if n_greens < 1:
        raise ValueError("an intersection needs at least one green phase")
    return tuple(range(n_greens))


# ----------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------


class FixedTimeController:
    """Cycle each intersection's greens on a fixed ``k``-decision-step schedule.

    Parameters
    ----------
    env
        Traffic env; ``env.control_mode`` must be ``"acyclic"`` and
        ``env.intersections`` a list of ``IntersectionInfo``.
    k
        Decision steps to hold each green before advancing to the next.
    plan
        Parsed shipped plan whose green *order* is followed; ``None`` uses the
        equal-split fallback (ascending green actions).  A plan whose green set
        does not match an intersection's greens is rejected (anti-mislabel).
    rng
        Accepted for registry-signature parity and ignored: the controller is
        deterministic by construction.
    """

    def __init__(
        self,
        env: Any,
        *,
        k: int,
        plan: PlanSchedule | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if int(k) < 1:
            raise ValueError(f"fixed-time k must be >= 1, got {k}")
        mode = str(getattr(env, "control_mode", "acyclic"))
        if mode != "acyclic":
            raise ValueError(
                f"FixedTimeController supports control_mode 'acyclic' only, got "
                f"{mode!r}. The C1 dataset ladder is collected under one control mode; "
                "a tier collected under a different action vocabulary is not comparable "
                "to the others (ladder comparability). Do not add a silent fallback "
                "for the other modes -- raise, so this decision stays visible."
            )
        self._k = int(k)
        self._ix_ids: list[str] = [str(ix.id) for ix in env.intersections]
        self._greens: dict[str, list[int]] = {}
        self._cycle: dict[str, tuple[int, ...]] = {}
        for ix in env.intersections:
            ix_id = str(ix.id)
            greens = green_action_phases(ix)
            self._greens[ix_id] = greens
            self._cycle[ix_id] = self._build_cycle(ix_id, greens, plan)
        _ = rng  # deterministic; no RNG is drawn

    def _build_cycle(
        self, ix_id: str, greens: Sequence[int], plan: PlanSchedule | None
    ) -> tuple[int, ...]:
        if plan is None:
            return equal_split_cycle(len(greens))
        if set(plan.green_order) != set(greens):
            raise ValueError(
                f"shipped plan green set {sorted(plan.green_order)} does not match "
                f"intersection {ix_id}'s green phases {sorted(greens)}; refusing to "
                "build a schedule that would mis-label the corpus"
            )
        index = {phase: action for action, phase in enumerate(greens)}
        return tuple(index[phase] for phase in plan.green_order)

    @property
    def k(self) -> int:
        return self._k

    def cycle_for(self, ix_id: str) -> tuple[int, ...]:
        """The action-index cycle followed at intersection ``ix_id``."""
        return self._cycle[str(ix_id)]

    def act(self, info: dict[str, Any]) -> np.ndarray:
        step = int(info["step"])
        per_ix = info.get("intersections", {}) or {}
        actions: list[int] = []
        for ix_id in self._ix_ids:
            cycle = self._cycle[ix_id]
            scheduled = int(cycle[(step // self._k) % len(cycle)])
            payload = per_ix.get(ix_id, {}) or {}
            avail = [int(a) for a in (payload.get("avail_actions") or [])]
            actions.append(self._legalize(ix_id, scheduled, avail, payload))
        return np.asarray(actions, dtype=np.int64)

    def _legalize(
        self, ix_id: str, scheduled: int, avail: list[int], payload: dict[str, Any]
    ) -> int:
        """Return the scheduled action, or a documented fallback if it is masked.

        Under ``acyclic`` every green is always available, so the fallback is
        unreachable there; it is defence in depth. When the scheduled action is
        illegal we hold the current green's action if that is legal (the
        fixed-time-faithful choice: keep, do not jump to an off-schedule green),
        else the lowest available action. If no mask is supplied (stub envs) the
        scheduled action is trusted.
        """
        if not avail:
            return scheduled
        if scheduled in avail:
            return scheduled
        greens = self._greens[ix_id]
        current_phase = int(payload.get("current_phase", -1))
        if current_phase in greens:
            current_action = greens.index(current_phase)
            if current_action in avail:
                return current_action
        return min(avail)


# ----------------------------------------------------------------------
# Plan resolution + provenance (for offline.collect)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PlanResolution:
    """Where a run's schedule came from, for the manifest and the factory."""

    source: str  # "shipped_plan" | "equal_split"
    plan: PlanSchedule | None
    path: str | None
    sha256: str | None


def _scenario_dir_from_config(config_path: str | Path) -> Path:
    """Absolute scenario directory a CityFlow sim config points at.

    Mirrors ``offline.collect._cityflow_flow_source``: a relative ``dir`` resolves
    against the current working directory.
    """
    cfg = json.loads(Path(config_path).read_bytes())
    cfg_dir = cfg.get("dir", "")
    if not os.path.isabs(cfg_dir):
        cfg_dir = str(Path.cwd() / cfg_dir)
    return Path(os.path.normpath(cfg_dir))


def resolve_plan(args: Any) -> PlanResolution:
    """Locate and parse the scenario's ``signal_plan_template.txt``.

    Returns a ``shipped_plan`` resolution when the file exists and parses, else an
    ``equal_split`` fallback -- absent file, non-CityFlow config, or a malformed
    plan all fall back (brief §6). The fallback is recorded in the manifest, so it
    is never silent.
    """
    config = getattr(args, "env_config", None)
    if not config:
        return PlanResolution("equal_split", None, None, None)
    try:
        plan_path = _scenario_dir_from_config(config) / "signal_plan_template.txt"
        if not plan_path.is_file():
            return PlanResolution("equal_split", None, None, None)
        raw = plan_path.read_bytes()
        plan = parse_signal_plan(raw.decode("utf-8"))
    except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError):
        return PlanResolution("equal_split", None, None, None)
    return PlanResolution("shipped_plan", plan, str(plan_path), hashlib.sha256(raw).hexdigest())


def fixed_time_provenance(args: Any) -> dict[str, Any]:
    """Manifest fields recording the fixed-time schedule (``None`` for other policies).

    Follows the ``epsilon`` precedent (``offline.collect``): a policy-specific set of
    keys, populated only for ``--policy fixedtime``. Without these, two Tier 1 runs
    with different ``k`` are indistinguishable from the data.
    """
    if getattr(args, "policy", None) != "fixedtime":
        return {
            "fixed_time_k": None,
            "fixed_time_schedule_source": None,
            "fixed_time_plan_sha256": None,
        }
    resolution = resolve_plan(args)
    return {
        "fixed_time_k": int(args.fixed_time_k),
        "fixed_time_schedule_source": resolution.source,
        "fixed_time_plan_sha256": resolution.sha256,
    }


def make_fixedtime(env: Any, args: Any, rng: np.random.Generator) -> PolicyFn:
    """Factory for :data:`offline.collect.POLICIES`; returns ``act(info)``."""
    resolution = resolve_plan(args)
    controller = FixedTimeController(
        env, k=int(args.fixed_time_k), plan=resolution.plan, rng=rng
    )
    return controller.act
