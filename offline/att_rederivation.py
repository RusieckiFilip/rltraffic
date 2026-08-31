"""P8.4b: re-derive A11(d)'s six ATT-based verdicts under both metric definitions.

Artifact format version: ``p8.4b-rederivation/1.0``.

WHAT THIS MODULE IS FOR
-----------------------
``PREREGISTRATION`` **A11(b)** requires every ATT cell to carry five quantities -- ``att_ours``,
``att_engine``, ``entered``, ``created``, ``never_entered`` -- unconditionally.  **A11(d)** enumerates
SIX verdicts that must each be re-evaluated under whichever definition ``Rule R`` makes primary, with
**both verdicts reported whether or not they agree**.  This module re-measures the cells those six
verdicts consume and re-evaluates each verdict twice.

⛔ **It issues no verdict on which definition is primary.**  ``Rule R`` decides that; this module
reports both readings side by side and flags a comparison whose ordering differs between them as
DEFINITION-DEPENDENT, which is A11's own registered language.

THE CELL SET IS DERIVED, NOT LISTED
-----------------------------------
🔒 **The cells are read out of the committed artifacts the verdicts were computed from**, by
enumerating their per-episode ``(arm, seed, draw_id)`` blocks.  A hand-written arm list would drift
from what the verdicts actually consumed, and drift is invisible: a re-derivation over the wrong
cells produces a plausible verdict for a population nobody reported.  Deriving it means the
re-measurement covers exactly the episodes behind each committed number, and the count is checkable.

⚠️ **P4.6's grid (``docs/data/p4_6_grid.json``, 11,700 episodes) is deliberately NOT a source.**
A11(d) puts ``offline/method_tier_grid.py:823`` explicitly OUT of scope -- it issues no verdict by
construction and re-asserts that with ``assert_no_verdicts`` -- so its ATT cells move but there is no
verdict to re-evaluate.  Including it would add 11,700 episodes of compute for nothing.

THE PRE-FLIGHT
--------------
``BRIEF_32`` Amendment C keeps the pre-flight check for the re-derivation, because unlike Gate 0 it
writes under ``output/``.  :func:`preflight` answers four questions **before any episode is rolled**:

1. **How many cells and episodes**, per scenario, derived as above.
2. **How long**, from the measured per-episode rates rather than a guess.
3. **Can every cell actually be rebuilt** -- each cell's action factory and checkpoint are resolved
   WITHOUT rolling anything, so a cell whose policy cannot be reconstructed is reported now rather
   than discovered three hours into a campaign.
4. **Is the write path safe** -- every destination is checked against the protected roots, and a
   refused destination must create nothing.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "MEASURED_SECONDS_PER_EPISODE",
    "REDERIVATION_SOURCES",
    "BARE_ARM_TIERS",
    "BEHAVIOUR_METHOD",
    "CONTEXT_LENGTH_METHODS",
    "HEURISTIC_METHODS",
    "MIXTURE_TIERS",
    "P4_5_METHODS",
    "CellKey",
    "VerdictSource",
    "build_parser",
    "assert_reproduces_committed",
    "committed_att_index",
    "cost_estimate",
    "factory_resolution",
    "main",
    "preflight",
    "normalise_arm",
    "rederivation_cells",
    "rederivation_checkpoint",
    "slots_from_episode_block",
]

ARTIFACT_FORMAT_VERSION = "p8.4b-rederivation/1.0"

#: Matches ``admission_probe.BEHAVIOUR_METHOD``; a tier's own collecting policy.
BEHAVIOUR_METHOD = "behaviour"

#: Seconds per episode, MEASURED rather than assumed.  These are the coordinator's rates from the
#: 1,870-episode P8.4a campaign, which used the same ``probe_episode`` path this module uses, with
#: no per-second observer attached.  Gate 0's own rates were higher (1.48 / 4.64 s) precisely
#: because the observer takes two ``get_vehicles`` calls per simulation second; the re-derivation
#: does not observe, so P8.4a's rates are the right basis and the difference is stated here so a
#: reader does not mistake one for the other.
MEASURED_SECONDS_PER_EPISODE: Mapping[str, float] = {"hz1x1": 1.29, "grid4x4": 2.67}

#: The default parallelism the author's tmux campaign uses.
DEFAULT_WORKERS = 5


@dataclass(frozen=True)
class CellKey:
    """One episode's identity: scenario, arm, seed slot and held-out draw."""

    scenario: str
    arm: str
    seed: int | None
    draw_id: int


@dataclass(frozen=True)
class VerdictSource:
    """One of A11(d)'s six in-scope verdicts and the episodes it was computed from.

    ``registered_at`` is the file:line A11(d) names, quoted so the enumeration stays checkable
    against the registration rather than against memory of it.  Every one of the six was verified to
    resolve to the expression A11(d) describes on 2026-08-31.
    """

    verdict_id: str
    name: str
    registered_at: str
    scenario: str
    episode_sources: tuple[str, ...]
    scope: str
    verdict_callable: str


#: A11(d)'s six, in its own order.  ``episode_sources`` are paths relative to the repository root
#: for committed artifacts and to ``output_root`` for campaign directories; :func:`rederivation_cells`
#: resolves both and refuses a source that resolves to nothing.
REDERIVATION_SOURCES: tuple[VerdictSource, ...] = (
    VerdictSource(
        verdict_id="V1",
        name="P4 gate: ATT_MADT <= ATT_MaxPressure and <= 1.05 x ATT_best_online",
        registered_at="offline/dt_gate.py:467-468",
        scenario="hz1x1",
        episode_sources=("docs/data/p4_gate.json", "docs/data/p4_heldout_thresholds.json"),
        scope="in_scope",
        verdict_callable="offline.dt_gate.gate_verdict",
    ),
    VerdictSource(
        verdict_id="V2",
        name="A6 equivalence margin on paired per-draw ATT differences",
        registered_at="offline/offline_baselines.py:998",
        scenario="hz1x1",
        episode_sources=("docs/data/p4_4_baselines.json", "docs/data/p4_5_baselines.json"),
        scope="in_scope",
        verdict_callable="offline.offline_baselines.delta_verdict",
    ),
    VerdictSource(
        verdict_id="V3",
        name="P4.7 Q2: mean ATT(bc) - mean ATT(bc_top10) per tier",
        registered_at="offline/mixture_tiers.py:767 with the outcome at :795-815",
        scenario="hz1x1",
        episode_sources=("docs/data/p4_7_grid.json",),
        scope="in_scope",
        verdict_callable="offline.mixture_tiers (outcome block)",
    ),
    VerdictSource(
        verdict_id="V4",
        name="P5.1 P2a/P2b: CI of mean(ATT_spatial - ATT_nomix) entirely below zero",
        registered_at="offline/spatial_mixing.py:646-650 and :732",
        scenario="grid4x4",
        episode_sources=("output:p5_1/eval_*.json",),
        scope="in_scope",
        verdict_callable="offline.spatial_mixing._outcome_from_interval",
    ),
    VerdictSource(
        verdict_id="V5",
        name="P5.2 Q0 stop rule: CI of the four-head ATT difference entirely below zero",
        registered_at="offline/tier_sweep.py:1370 used at :2354",
        scenario="grid4x4",
        episode_sources=("output:p5_2/eval_*.json",),
        scope="in_scope",
        verdict_callable="offline.tier_sweep.stop_rule_verdict",
    ),
    VerdictSource(
        verdict_id="V6",
        name="P7.0 branch verdict, rho terms only (within-backend ATT ratio)",
        registered_at="offline/transfer_gate.py:1231",
        scenario="multi",
        episode_sources=("output:p7_0/*/manifest.json",),
        scope="in_scope_in_part",
        verdict_callable="offline.transfer_gate.evaluate_branch",
    ),
)


#: P4.7's three MIXTURE tiers.  Their checkpoints live under ``output/p4_7/checkpoints`` while every
#: other hz1x1 non-``mappo1000`` tier reuses P4.6's, which is what ``_method_checkpoint`` already
#: does -- so this set is exactly the override, and nothing wider.
MIXTURE_TIERS: frozenset[str] = frozenset({"mix33", "mix50", "mix67"})

#: Methods that are HEURISTICS, not checkpoints.  P5.1 evaluates a ``random`` ARM beside its learned
#: ones, so ``random`` appears both as a tier name and as a method name; as a method it has no
#: checkpoint to resolve and is rebuilt from ``method_tier_grid._random_factory`` exactly as
#: collection built it.  Calling a heuristic a checkpoint is a provenance error, not a label.
HEURISTIC_METHODS: frozenset[str] = frozenset({"random"})

#: P5.2's context-length heads.  They are grid4x4/mappo1000 METHODS whose checkpoints live under
#: ``output/p5_2/checkpoints`` rather than P5.1's, like ``bc_top10_perix`` before them.
CONTEXT_LENGTH_METHODS: frozenset[str] = frozenset({"dt_nomix_h4", "dt_spatial_h4"})

#: P4.5's data-selection arms.  They are METHODS of the ``mappo1000`` tier with their own checkpoint
#: family under ``output/p4_5/checkpoints``, not tiers.
P4_5_METHODS: frozenset[str] = frozenset(
    {"bc_any_20", "bc_best2_20", "bc_best2_all", "bc_worst2_20"}
)

#: The P4 family writes its arms BARE -- ``madt``, ``mappo1000``, ``bc`` -- with the tier implied by
#: the campaign rather than carried in the string.  ⚠️ **This map is the one place in this module
#: where a name is resolved by declaration rather than parsed**, so it is small, explicit, and every
#: entry is checked by the reproduction test: a wrong entry loads a different policy, and a different
#: policy cannot reproduce the committed ``att_ours``.
BARE_ARM_TIERS: Mapping[str, tuple[str, str]] = {
    "madt": ("dt", "mappo1000"),
    "bc": ("bc", "mappo1000"),
    "bc_top10": ("bc_top10", "mappo1000"),
    "iql": ("iql", "mappo1000"),
    "mappo1000": ("behaviour", "mappo1000"),
    "mappo500": ("behaviour", "mappo500"),
    "maxpressure": ("behaviour", "maxpressure"),
    "random": ("behaviour", "random"),
    "fixedtime": ("behaviour", "fixedtime"),
    **{name: (name, "mappo1000") for name in sorted(P4_5_METHODS)},
}


def normalise_arm(scenario: str, arm: str) -> tuple[str, str]:
    """``arm`` -> ``(method, tier)``, absorbing the three naming conventions in play.

    Three conventions, all of them real and all of them in committed artifacts:

    * ``method@tier`` -- P4.7, P5.2, P8.4a.  The common case.
    * ``method@<scenario>_<tier>`` -- P5.1 writes ``bc@grid4x4_mappo1000``.  The scenario prefix is
      stripped; it names the network the cell is already filed under.
    * a bare name -- the P4 family.  Resolved through :data:`BARE_ARM_TIERS`, never guessed.

    Raises on an arm that matches none of the three, because a silent fallback here would file a
    cell under the wrong tier and load the wrong checkpoint.
    """
    text = str(arm)
    if "@" in text:
        method, _, tier = text.partition("@")
        prefix = f"{scenario}_"
        if tier.startswith(prefix):
            tier = tier[len(prefix) :]
        return method, tier
    try:
        return BARE_ARM_TIERS[text]
    except KeyError as exc:
        raise ValueError(
            f"{scenario}/{arm!r} carries no tier and is not a declared bare arm name; the declared "
            f"ones are {sorted(BARE_ARM_TIERS)}. Guessing a tier would load a different policy"
        ) from exc


def rederivation_checkpoint(
    scenario: str, tier: str, method: str, seed: int, roots: Any
) -> Path:
    """Where the model behind a cell's COMMITTED numbers lives.

    Delegates to ``admission_probe._method_checkpoint`` for everything it already covers -- which,
    checked against the directories on disk, is every grid4x4 tier and every hz1x1 tier except the
    mixtures -- and overrides only the two families it does not know about.
    """
    from offline.admission_probe import _method_checkpoint

    out = Path(roots.output_root)
    if scenario == "hz1x1" and method in P4_5_METHODS:
        return out / "p4_5" / "checkpoints" / f"{method}_seed{int(seed)}.pt"
    if scenario == "hz1x1" and tier in MIXTURE_TIERS:
        return out / "p4_7" / "checkpoints" / f"{tier}_{method}_seed{int(seed)}.pt"
    if scenario == "grid4x4" and method in CONTEXT_LENGTH_METHODS:
        # ⚠️ The SAME trap ``_method_checkpoint``'s docstring records for ``bc_top10_perix``: it
        # routes grid4x4/mappo1000 to P5.1, but the context-length heads were trained by P5.2 and
        # exist ONLY under output/p5_2/checkpoints.  Verified on disk 2026-08-31: 10 files there, 0
        # in p5_1.  Loading the P5.1 path would not have failed loudly -- it would have failed to
        # exist, and had a same-named file existed it would have produced a plausible number for a
        # different model.
        return out / "p5_2" / "checkpoints" / f"grid4x4_{tier}_{method}_seed{int(seed)}.pt"
    return _method_checkpoint(scenario, tier, method, int(seed), roots)


def committed_att_index(
    *, repo_root: str | Path, output_root: str | Path
) -> dict[tuple[str, str, int | None, int], float]:
    """``(scenario, arm, seed, draw_id) -> committed att_horizon``, from the source artifacts.

    🔒 **This index IS the acceptance test.**  A checkpoint's sha256 proves the file is intact, not
    that it is the right file for that arm -- a wrong arm-to-checkpoint entry loads an intact
    checkpoint belonging to a different policy and produces a plausible ATT for a cell nobody
    reported.  The re-derived ``att_ours`` must reproduce the committed one for that cell, and the
    episode has to be rolled anyway, so the strongest available check is also the free one.

    A cell absent from this index is UNVERIFIED and is excluded from every verdict rather than
    reported with a caveat.
    """
    repo, out = Path(repo_root), Path(output_root)
    #: key -> {value: the first file that reported it}.  Collected as a MAPPING rather than
    #: overwritten, because the first version of this function assigned straight into a dict and a
    #: replicate file -- sorting after the committed one -- silently replaced 100 committed values
    #: with a second measurement's.  The campaign then failed four hours in on an arm whose
    #: checkpoint map was CORRECT, and the error message blamed the map.
    collected: dict[tuple[str, str, int | None, int], dict[float, str]] = {}
    for source in REDERIVATION_SOURCES:
        for path in _resolve_sources(source, repo_root=repo, output_root=out):
            if is_replicate_artifact(path):
                continue
            try:
                payload = json.loads(path.read_bytes())
            except (ValueError, OSError):
                continue
            for row in slots_from_episode_block(payload):
                value = row.get("att_horizon")
                if value is None:
                    continue
                seed = row.get("seed")
                key = (
                    source.scenario,
                    str(row["arm"]),
                    None if seed is None else int(seed),
                    int(row["draw_id"]),
                )
                collected.setdefault(key, {}).setdefault(float(value), path.name)

    conflicts = {k: v for k, v in collected.items() if len(v) > 1}
    if conflicts:
        key, values = sorted(conflicts.items())[0]
        raise ValueError(
            f"{len(conflicts)} cells have MORE THAN ONE committed att_ours across the source "
            f"artifacts, so there is no single value to reproduce. First: {key} carries "
            f"{sorted(values.items())}. A silent last-writer-wins here is what turns a reference "
            "defect into a rollout failure that blames the checkpoint map"
        )
    return {key: next(iter(values)) for key, values in collected.items()}


def assert_reproduces_committed(
    cell: CellKey, att_ours: float, committed: Mapping[tuple[str, str, int | None, int], float]
) -> float:
    """Refuse a re-derived ``att_ours`` that does not reproduce the committed one, EXACTLY.

    ``==`` and not ``isclose``: P8.4a's own reference checks came back exact on 39 of 39 and 14 of 14
    cells through this same rollout path, so exactness is the demonstrated bar and a tolerance here
    would be slack nobody needs.  **Non-reproduction is a REFUSAL, never a warning** -- it means the
    policy that was loaded is not the policy that produced the committed number, and every verdict
    downstream would rest on a different arm wearing the right name.
    """
    key = (cell.scenario, cell.arm, cell.seed, cell.draw_id)
    if key not in committed:
        raise KeyError(
            f"{cell.scenario}/{cell.arm} seed {cell.seed} draw {cell.draw_id} has no committed "
            "att_ours at any grain, so it is UNVERIFIED and may not enter a verdict"
        )
    expected = committed[key]
    if float(att_ours) != expected:
        raise ValueError(
            f"{cell.scenario}/{cell.arm} seed {cell.seed} draw {cell.draw_id}: re-derived att_ours "
            f"{att_ours!r} does not reproduce the committed {expected!r}. The arm-to-checkpoint map "
            "loaded a policy that did not produce this cell's committed number; a file hash would "
            "not have caught this"
        )
    return expected


def slots_from_episode_block(payload: Any) -> tuple[dict[str, Any], ...]:
    """Every per-episode row carrying ``draw_id`` in *payload*, or an empty tuple.

    A committed artifact stores its episodes under ``episodes``; a campaign cell file does the same.
    Rows without a ``draw_id`` are not episodes of the shape this module re-measures and are
    skipped rather than guessed at.
    """
    if not isinstance(payload, Mapping):
        return ()
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        return ()
    return tuple(
        row for row in episodes if isinstance(row, Mapping) and "draw_id" in row and "arm" in row
    )


#: Filename marker for a REPLICATE: a deliberate SECOND measurement of a slot already measured, run
#: to check reproducibility.  It is not the committed cell and must never stand in for one.
#: ``output/p5_2/eval_random_dt_spatial_seed202_replicate.json`` is the instance that made this
#: matter: its 100 rows disagree with the committed ones by 5-13 s in both directions, because a
#: stochastic arm rolled twice gives two answers.  **Both are real measurements; only one is the
#: number P5.2 reported.**
REPLICATE_MARKER = "_replicate"


def is_replicate_artifact(path: str | Path) -> bool:
    """Whether *path* is a replicate rather than a committed cell file."""
    return REPLICATE_MARKER in Path(path).stem


def _resolve_sources(source: VerdictSource, *, repo_root: Path, output_root: Path) -> list[Path]:
    """Every file one verdict's ``episode_sources`` names, refusing a pattern that matches nothing."""
    found: list[Path] = []
    for pattern in source.episode_sources:
        if pattern.startswith("output:"):
            base, glob = output_root, pattern[len("output:") :]
        else:
            base, glob = repo_root, pattern
        matches = sorted(base.glob(glob)) if any(c in glob for c in "*?[") else (
            [base / glob] if (base / glob).is_file() else []
        )
        if not matches:
            raise FileNotFoundError(
                f"{source.verdict_id}: the episode source {pattern!r} under {base} matches no file. "
                "The cell set is DERIVED from these artifacts, so a missing one would silently "
                "shrink the re-derivation rather than fail it"
            )
        found.extend(matches)
    return found


def rederivation_cells(
    *, repo_root: str | Path, output_root: str | Path
) -> dict[str, set[CellKey]]:
    """Every episode the six verdicts consume, keyed by verdict id.

    The scenario of a cell comes from its verdict, not from the arm string: P5.1's arms carry a
    ``@grid4x4_mappo1000`` suffix while P4.7's carry a bare tier, and reading the network out of an
    arm name would be a parser that works until it does not.
    """
    repo = Path(repo_root)
    out = Path(output_root)
    cells: dict[str, set[CellKey]] = {}
    for source in REDERIVATION_SOURCES:
        found: set[CellKey] = set()
        for path in _resolve_sources(source, repo_root=repo, output_root=out):
            try:
                payload = json.loads(path.read_bytes())
            except (ValueError, OSError):
                continue
            for row in slots_from_episode_block(payload):
                seed = row.get("seed")
                found.add(
                    CellKey(
                        scenario=source.scenario,
                        arm=str(row["arm"]),
                        seed=None if seed is None else int(seed),
                        draw_id=int(row["draw_id"]),
                    )
                )
        cells[source.verdict_id] = found
    return cells


def cost_estimate(
    cells: Mapping[str, set[CellKey]], *, workers: int = DEFAULT_WORKERS
) -> dict[str, Any]:
    """Episode counts and wall-time, from :data:`MEASURED_SECONDS_PER_EPISODE`.

    ⚠️ **Episodes are de-duplicated across verdicts before they are costed.**  V1 and V2 share
    ``maxpressure`` and ``mappo1000`` cells, and counting them twice would overstate the campaign by
    hours.  The per-verdict counts are reported un-deduplicated beside the total, because a reader
    checking one verdict's coverage needs its own number.
    """
    if int(workers) < 1:
        raise ValueError(f"workers must be >= 1, got {workers!r}")

    unique: set[CellKey] = set()
    for found in cells.values():
        unique |= found

    by_scenario: dict[str, int] = {}
    for cell in unique:
        by_scenario[cell.scenario] = by_scenario.get(cell.scenario, 0) + 1

    unknown = sorted(s for s in by_scenario if s not in MEASURED_SECONDS_PER_EPISODE)
    seconds = 0.0
    for scenario, count in by_scenario.items():
        rate = MEASURED_SECONDS_PER_EPISODE.get(scenario)
        if rate is not None:
            seconds += count * float(rate)

    return {
        "episodes_total_unique": len(unique),
        "episodes_by_verdict": {k: len(v) for k, v in sorted(cells.items())},
        "episodes_by_scenario": dict(sorted(by_scenario.items())),
        "rates_seconds_per_episode": dict(MEASURED_SECONDS_PER_EPISODE),
        "scenarios_without_a_measured_rate": unknown,
        "seconds_one_way": seconds,
        "hours_one_way": seconds / 3600.0,
        "workers": int(workers),
        "hours_at_workers": seconds / 3600.0 / int(workers),
        "note": (
            "wall time assumes perfect packing across workers and no policy-load overhead beyond "
            "what the measured rate already contains; it is an estimate and is labelled one"
        ),
    }


def factory_resolution(
    cells: Mapping[str, set[CellKey]], *, roots: Any
) -> dict[str, Any]:
    """Which cells can have their policy rebuilt, WITHOUT rolling a single episode.

    🚨 **This is the check that decides whether the campaign can run at all.**
    ``admission_probe.build_factory`` refuses any cell outside ``PROBE_SCENARIOS``' declared tiers
    and methods, and P8.4a declared five tiers on hz1x1 and two on grid4x4.  The six verdicts reach
    considerably further -- P4.7 adds the ``mix33`` / ``mix50`` / ``mix67`` mixture tiers and P5.2
    adds ``fixedtime`` / ``maxpressure`` / ``random`` tiers ON GRID4X4 -- so a plain reuse of the
    probe's registry cannot serve them.  Each distinct ``(scenario, arm, seed)`` is resolved once and
    the refusal reason is recorded verbatim.
    """
    from offline.admission_probe import build_factory
    from offline.materialise_draws import draw_config_path

    #: ``_fixedtime_factory`` resolves its plan from a real sim config, so the resolution check must
    #: hand it one; passing ``None`` would report a FALSE blocker on every fixedtime cell.  The draw
    #: is only read, never rolled.
    scenario_keys = {"hz1x1": "cityflow1x1", "grid4x4": "cityflow_grid4x4"}
    configs: dict[str, Path | None] = {}
    for name, key in scenario_keys.items():
        candidate = Path(draw_config_path(key, 1000, out_root=roots.draws_root))
        configs[name] = candidate if candidate.is_file() else None

    slots: dict[tuple[str, str, int | None], set[int]] = {}
    for found in cells.values():
        for cell in found:
            slots.setdefault((cell.scenario, cell.arm, cell.seed), set()).add(cell.draw_id)

    resolvable: list[dict[str, Any]] = []
    unresolvable: list[dict[str, Any]] = []
    constructed: list[dict[str, Any]] = []
    for (scenario, arm, seed), draws in sorted(slots.items(), key=lambda kv: kv[0]):
        record: dict[str, Any] = {
            "scenario": scenario,
            "arm": arm,
            "seed": seed,
            "n_draws": len(draws),
        }
        if scenario not in MEASURED_SECONDS_PER_EPISODE:
            unresolvable.append({**record, "reason": f"no probe path for scenario {scenario!r}"})
            continue
        try:
            method, tier = normalise_arm(scenario, arm)
        except ValueError as exc:
            unresolvable.append({**record, "reason": f"ValueError: {exc}"})
            continue
        record |= {"method": method, "tier": tier}

        if method == BEHAVIOUR_METHOD and tier in MIXTURE_TIERS:
            # ⭐ A mixture tier's behaviour reference is CONSTRUCTED, not rolled out
            # (mixture_tiers.py:311 and :412, and its CLI help says "no rollout, zero compute"):
            # it is a weighted composition of the two component arms' cells, and those components
            # ARE in this campaign.  So these cells must NOT be re-rolled -- re-rolling one would
            # invent a policy that never existed and silently replace a constructed reference with
            # a measured one.  They are excluded from the rollout and re-derived by the same
            # construction from the re-derived components.
            constructed.append({**record, "policy": "constructed reference, not rolled"})
            continue
        if method == BEHAVIOUR_METHOD:
            # A behaviour anchor is rebuilt from its tier's own corpus manifest, never a checkpoint
            # path this module invents.  ⚠️ Routed through GATE 0's build_factory, not the probe's:
            # admission_probe declares five tiers on hz1x1 and two on grid4x4, while Gate 0 already
            # covers all SEVEN behaviour tiers on BOTH scenarios and is tested there.  Reusing it is
            # what keeps a second implementation of the same anchor out of this module.
            from offline.engine_att_reference import build_factory as behaviour_factory

            try:
                behaviour_factory(
                    scenario, tier, method, seed, roots,
                    device=None, config_path=configs.get(scenario),
                )
            except Exception as exc:  # noqa: BLE001 - the reason is the deliverable
                unresolvable.append({**record, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            resolvable.append({**record, "policy": "behaviour anchor via admission_probe"})
            continue

        if method in HEURISTIC_METHODS:
            if seed is None:
                unresolvable.append(
                    {**record, "reason": f"the {method} arm is seeded and cannot take seed=None"}
                )
                continue
            resolvable.append({**record, "policy": f"heuristic: {method}, seeded {seed}"})
            continue
        if seed is None:
            unresolvable.append({**record, "reason": "a learned arm cannot take seed=None"})
            continue
        try:
            checkpoint = rederivation_checkpoint(scenario, tier, method, int(seed), roots)
        except Exception as exc:  # noqa: BLE001
            unresolvable.append({**record, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if not checkpoint.is_file():
            unresolvable.append({**record, "reason": f"checkpoint not on disk: {checkpoint}"})
            continue
        resolvable.append({**record, "policy": str(checkpoint)})

    return {
        "n_slots": len(slots),
        "n_constructed_not_rolled": len(constructed),
        "episodes_constructed_not_rolled": sum(r["n_draws"] for r in constructed),
        "constructed": constructed,
        "n_resolvable": len(resolvable),
        "n_unresolvable": len(unresolvable),
        "episodes_unresolvable": sum(r["n_draws"] for r in unresolvable),
        "unresolvable": unresolvable,
        "resolvable_sample": resolvable[:10],
    }


def preflight(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    corpus_root: str | Path,
    draws_root: str | Path,
    work_dir: str | Path,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    """The pre-flight ``BRIEF_32`` Amendment C requires before the re-derivation writes anything.

    Rolls no episode, writes nothing, and creates no directory.  Returns the report; the caller
    decides whether to write it.
    """
    from offline.admission_probe import ProbeRoots, default_protected_roots
    from offline.tier_sweep import assert_writable

    roots = ProbeRoots(
        repo_root=Path(repo_root),
        corpus_root=Path(corpus_root),
        draws_root=Path(draws_root),
        output_root=Path(output_root),
        work_dir=Path(work_dir),
    )
    started = time.perf_counter()
    cells = rederivation_cells(repo_root=repo_root, output_root=output_root)
    cost = cost_estimate(cells, workers=workers)
    factories = factory_resolution(cells, roots=roots)

    protected = default_protected_roots(roots)
    write_checks: list[dict[str, Any]] = []
    for candidate in (Path(work_dir) / "cell.json", Path(repo_root) / "docs/data/x.json"):
        try:
            assert_writable(candidate, protected)
            write_checks.append({"path": str(candidate), "writable": True, "refused_with": None})
        except PermissionError as exc:
            write_checks.append({"path": str(candidate), "writable": False, "refused_with": str(exc)})

    refusals: list[dict[str, Any]] = []
    for sibling in sorted(Path(output_root).glob("p*")):
        if not sibling.is_dir() or sibling.resolve() == Path(work_dir).resolve():
            continue
        target = sibling / "should_never_be_written.json"
        try:
            assert_writable(target, protected)
            refusals.append({"path": str(target), "refused": False})
        except PermissionError:
            refusals.append({"path": str(target), "refused": True})
        if target.exists():
            raise RuntimeError(
                f"{target} exists after a write-path CHECK; the pre-flight must create nothing"
            )

    unprotected = [r["path"] for r in refusals if not r["refused"]]
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "P8.4b re-derivation pre-flight: cell count, wall-time estimate from measured rates, "
            "policy-rebuild coverage and write-path safety, before any episode is rolled"
        ),
        "sources": [
            {
                "verdict_id": s.verdict_id,
                "name": s.name,
                "registered_at": s.registered_at,
                "scenario": s.scenario,
                "scope": s.scope,
                "episode_sources": list(s.episode_sources),
                "verdict_callable": s.verdict_callable,
            }
            for s in REDERIVATION_SOURCES
        ],
        "cost": cost,
        "policy_rebuild": factories,
        "write_path": {
            "protected_roots": [str(p) for p in protected],
            "checks": write_checks,
            "sibling_refusals": refusals,
            "siblings_not_protected": unprotected,
            "nothing_was_created": True,
        },
        "clear_to_run": bool(
            factories["n_unresolvable"] == 0
            and not unprotected
            and not cost["scenarios_without_a_measured_rate"]
        ),
        "seconds_preflight": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    """The CLI.  ``allow_abbrev`` is False for the reason given in ``engine_att_reference``."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.att_rederivation",
        description="P8.4b: re-derive A11(d)'s six ATT verdicts under both definitions",
        allow_abbrev=False,
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--corpus-root", default="datasets_v11")
    parser.add_argument("--draws-root", default="scenarios/draws")
    parser.add_argument("--output-root", default="output", help="the MAIN tree's output/")
    parser.add_argument("--work-dir", default=None, help="default <output-root>/p8_4b_rederivation")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    sub = parser.add_subparsers(dest="command", required=True)
    flight = sub.add_parser(
        "preflight", help="cell count, wall time, rebuild coverage and write-path safety",
        allow_abbrev=False,
    )
    flight.add_argument("--out", default=None, help="write the report here as well as printing it")

    runner = sub.add_parser(
        "run", help="roll the campaign's cells, checking reproduction per cell", allow_abbrev=False
    )
    runner.add_argument(
        "--verdicts", nargs="+", default=[s.verdict_id for s in REDERIVATION_SOURCES]
    )
    runner.add_argument("--shard", type=int, default=0, help="this worker's index, 0-based")
    runner.add_argument("--of", type=int, default=1, help="how many workers share the campaign")
    runner.add_argument("--engine-seed", type=int, default=1000)
    runner.add_argument("--device", default=None)
    runner.add_argument("--torch-threads", type=int, default=1)
    runner.add_argument("--limit", type=int, default=None, help="roll at most N cells (smoke test)")

    status = sub.add_parser(
        "status", help="declared against present; complete is COMPUTED from disk",
        allow_abbrev=False,
    )
    status.add_argument("--unused", action="store_true", help=argparse.SUPPRESS)
    return parser


def is_constructed_reference(cell: CellKey) -> bool:
    """Whether *cell* is a mixture tier's CONSTRUCTED behaviour reference, which is never rolled.

    ``mixture_tiers.py:311`` and ``:412``; its CLI help says "no rollout, zero compute".  Re-rolling
    one would invent a policy that never existed and replace a constructed reference with a measured
    one.
    """
    method, tier = normalise_arm(cell.scenario, cell.arm)
    return method == BEHAVIOUR_METHOD and tier in MIXTURE_TIERS


def campaign_cell_set(args: argparse.Namespace, output_root: Path) -> list[CellKey]:
    """The WHOLE campaign: every rollable cell of the selected verdicts, independent of ``--shard``.

    🔒 **This is what the manifest declares, and it must not depend on which worker is asking.**
    The first version of this function sharded before the manifest was built, so each of five
    workers declared its own 7,700-cell slice as if it were the campaign; the digests genuinely
    described different sets and :func:`assert_manifest_agrees` refused every worker after the
    first.  **The guard was right and the declaration was wrong.**  Sharding now happens strictly
    after this, in :func:`campaign_shard`.
    """
    wanted = {str(v) for v in args.verdicts}
    unknown = wanted - {s.verdict_id for s in REDERIVATION_SOURCES}
    if unknown:
        raise ValueError(f"unknown verdict ids {sorted(unknown)}")
    by_verdict = rederivation_cells(repo_root=args.repo_root, output_root=output_root)
    selected: set[CellKey] = set()
    for verdict_id, found in by_verdict.items():
        if verdict_id in wanted:
            selected |= found
    rollable = {c for c in selected if not is_constructed_reference(c)}
    return sorted(rollable, key=lambda c: (c.scenario, c.arm, str(c.seed), c.draw_id))


def campaign_shard(cells: Sequence[CellKey], *, shard: int, of: int) -> list[CellKey]:
    """The slice of the campaign THIS worker rolls.  Never what the manifest declares."""
    return shard_cells(cells, shard=int(shard), of=int(of))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.  Non-zero when the pre-flight is NOT clear to run, or a campaign is incomplete."""
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    work_dir = Path(args.work_dir) if args.work_dir else output_root / "p8_4b_rederivation"

    if args.command == "status":
        state = campaign_status(work_dir)
        print(json.dumps(state, indent=2, sort_keys=True), flush=True)
        return 0 if state.get("complete") else 1

    if args.command == "run":
        from offline.admission_probe import ProbeRoots, default_protected_roots
        from offline.offline_baselines import pin_torch_threads
        from offline.tier_sweep import assert_writable, write_json_guarded

        pin_torch_threads(int(args.torch_threads))
        roots = ProbeRoots(
            repo_root=Path(args.repo_root),
            corpus_root=Path(args.corpus_root),
            draws_root=Path(args.draws_root),
            output_root=output_root,
            work_dir=work_dir,
        )
        protected = default_protected_roots(roots)
        # The manifest declares the WHOLE campaign; the worker rolls only its shard of it.  These
        # two must never be the same list, which is the defect that stopped shards 1-4 on the first
        # launch: every worker declared its own slice and the digests disagreed, correctly.
        campaign = campaign_cell_set(args, output_root)
        manifest = campaign_manifest(campaign, engine_seed=int(args.engine_seed))
        cells = campaign_shard(campaign, shard=int(args.shard), of=int(args.of))

        manifest_path = work_dir / MANIFEST_NAME
        # (ii) DEFERRED 58: declare before running, and NEVER overwrite a disagreeing declaration.
        assert_manifest_agrees(manifest_path, manifest)
        assert_writable(manifest_path, protected)
        work_dir.mkdir(parents=True, exist_ok=True)
        if not manifest_path.is_file():
            write_json_guarded(manifest, manifest_path, protected)

        if args.limit is not None:
            # --limit means "roll N cells I have not already done", not "look at the first N".
            # Slicing before the resume filter makes the flag a no-op on a resumed campaign, which
            # is precisely when a bounded test run is wanted.
            pending = [c for c in cells if not (work_dir / cell_file_name(c)).is_file()]
            cells = pending[: int(args.limit)]
            print(f"  --limit {args.limit}: {len(pending)} pending in this shard, rolling "
                  f"{len(cells)}", flush=True)
        print(
            f"campaign declares {len(campaign)} cells (digest "
            f"{manifest['declared_cells_sha256'][:12]}); this worker rolls {len(cells)} of them "
            f"as shard {args.shard}/{args.of}. Work dir {work_dir}",
            flush=True,
        )
        committed = committed_att_index(repo_root=args.repo_root, output_root=output_root)
        outcome = run_campaign(
            cells,
            roots=roots,
            engine_seed=int(args.engine_seed),
            device=args.device,
            committed=committed,
            protected=protected,
        )
        state = campaign_status(work_dir)
        print(
            f"rolled={outcome['rolled']} skipped={outcome['skipped_already_present']} "
            f"refused={outcome['n_refused']} in {outcome['seconds'] / 3600.0:.2f} h; "
            f"declared={state['declared']} present={state['present']} "
            f"refused_on_disk={state['refused']} complete={state['complete']}",
            flush=True,
        )
        if outcome["n_refused"]:
            print(f"REFUSED {outcome['n_refused']} cells:", flush=True)
            for row in outcome["refused"][:20]:
                print(
                    f"  [{row['reason_class']}] {row['scenario']}/{row['arm']} seed {row['seed']} "
                    f"draw {row['draw_id']}: rederived {row['att_ours_rederived']!r} vs committed "
                    f"{row['att_ours_committed']!r}",
                    flush=True,
                )
            if outcome["n_refused"] > 20:
                print(f"  ... and {outcome['n_refused'] - 20} more", flush=True)
        if state["complete"] and not state["marker_present"]:
            marker = work_dir / COMPLETE_MARKER
            assert_writable(marker, protected)
            marker.write_text(
                json.dumps({"declared_cells_sha256": state["declared_cells_sha256"]}) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {marker}", flush=True)
        # A WORKER's exit code is about the WORKER's slice, not the campaign's completeness.
        # Returning 1 because other shards have not finished made all five report failure after a
        # clean run, which is the kind of signal that stops being read.  Campaign-level
        # completeness is what the `status` subcommand answers, and it still exits 1 until done.
        return 1 if outcome["n_refused"] else 0

    report = preflight(
        repo_root=args.repo_root,
        output_root=output_root,
        corpus_root=args.corpus_root,
        draws_root=args.draws_root,
        work_dir=work_dir,
        workers=int(args.workers),
    )
    cost = report["cost"]
    print("PRE-FLIGHT -- P8.4b re-derivation", flush=True)
    print(f"  episodes (unique, de-duplicated): {cost['episodes_total_unique']}", flush=True)
    for verdict_id, count in cost["episodes_by_verdict"].items():
        print(f"    {verdict_id}: {count}", flush=True)
    print(f"  by scenario: {cost['episodes_by_scenario']}", flush=True)
    print(f"  rates (s/episode, measured): {cost['rates_seconds_per_episode']}", flush=True)
    print(
        f"  wall time: {cost['hours_one_way']:.2f} h at 1-way, "
        f"{cost['hours_at_workers']:.2f} h at {cost['workers']}-way",
        flush=True,
    )
    rebuild = report["policy_rebuild"]
    print(
        f"  policy rebuild: {rebuild['n_resolvable']}/{rebuild['n_slots']} slots resolvable, "
        f"{rebuild['n_unresolvable']} NOT ({rebuild['episodes_unresolvable']} episodes)",
        flush=True,
    )
    for row in rebuild["unresolvable"][:8]:
        print(f"    UNRESOLVABLE {row['scenario']}/{row['arm']} seed {row['seed']}: {row['reason'][:120]}", flush=True)
    if rebuild["n_unresolvable"] > 8:
        print(f"    ... and {rebuild['n_unresolvable'] - 8} more", flush=True)
    print(f"  siblings not protected: {report['write_path']['siblings_not_protected'] or 'none'}", flush=True)
    print(f"  CLEAR TO RUN: {report['clear_to_run']}", flush=True)

    if args.out:
        from offline.admission_probe import default_protected_roots
        from offline.admission_probe import ProbeRoots
        from offline.tier_sweep import write_json_guarded

        roots = ProbeRoots(
            repo_root=Path(args.repo_root),
            corpus_root=Path(args.corpus_root),
            draws_root=Path(args.draws_root),
            output_root=output_root,
            work_dir=work_dir,
        )
        write_json_guarded(report, Path(args.repo_root) / args.out, default_protected_roots(roots))
        print(f"  wrote {Path(args.repo_root) / args.out}", flush=True)

    return 0 if report["clear_to_run"] else 1



# ----------------------------------------------------------------------
# The campaign runner
# ----------------------------------------------------------------------


#: The declared training budget every checkpoint this module loads must record, from
#: ``admission_probe.DECLARED_GRADIENT_STEPS``.  The loaders refuse anything else, which is the
#: mechanical form of "no online model selection".
DECLARED_GRADIENT_STEPS = 40_000

#: Written only when every declared cell has a file.  Its ABSENCE is what makes a partial run
#: structurally distinguishable from a complete one (``DEFERRED`` 58).
COMPLETE_MARKER = "CAMPAIGN_COMPLETE"

#: The declaration of what this campaign is.  Never overwritten: P5.2's BL-1 overwrote a runs list
#: after 53 hours of compute, so a second run that disagrees REFUSES instead of replacing it.
MANIFEST_NAME = "campaign_manifest.json"


def cell_file_name(cell: CellKey) -> str:
    """One file per episode, so a job that dies takes one episode with it."""
    arm = str(cell.arm).replace("@", "_at_").replace("/", "_")
    slot = "none" if cell.seed is None else str(int(cell.seed))
    return f"cell_{cell.scenario}_{arm}_seed{slot}_draw{int(cell.draw_id)}.json"


def refusal_file_name(cell: CellKey) -> str:
    """A REFUSED cell's record.  Deliberately NOT a ``cell_*.json``.

    A refusal must never be mistaken for data: ``campaign_status`` counts ``cell_*`` files, so a
    refused cell leaves the campaign INCOMPLETE and is retried on the next resume rather than
    silently skipped.
    """
    return "refused_" + cell_file_name(cell)[len("cell_"):]


def shard_cells(cells: Iterable[CellKey], *, shard: int, of: int) -> list[CellKey]:
    """A deterministic slice of the campaign, for N-way parallelism.

    Sorted first so the split is a property of the cell set and not of dictionary order: two workers
    started minutes apart must partition the same way or they duplicate and miss work.
    """
    if int(of) < 1 or not 0 <= int(shard) < int(of):
        raise ValueError(f"shard {shard!r} of {of!r} is not a valid partition")
    ordered = sorted(cells, key=lambda c: (c.scenario, c.arm, str(c.seed), c.draw_id))
    return [c for i, c in enumerate(ordered) if i % int(of) == int(shard)]


def campaign_manifest(cells: Sequence[CellKey], *, engine_seed: int) -> dict[str, Any]:
    """The declared cell set, with a digest so a disagreeing re-run can be refused."""
    import hashlib

    keys = sorted(
        f"{c.scenario}|{c.arm}|{c.seed}|{c.draw_id}" for c in cells
    )
    digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "n_cells": len(keys),
        "declared_cells_sha256": digest,
        "engine_seed": int(engine_seed),
        "cells": keys,
    }


def assert_manifest_agrees(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Refuse a re-run whose declared cell set differs from the one already on disk.

    ``DEFERRED`` 58, and P5.2's BL-1 concretely: a runs list was OVERWRITTEN after 53 hours of
    compute, which silently redefined what the campaign had been.  A manifest that disagrees means
    the two runs are not the same campaign, and resuming across them would mix two cell sets under
    one directory.
    """
    existing = Path(path)
    if not existing.is_file():
        return
    recorded = json.loads(existing.read_bytes())
    if recorded.get("declared_cells_sha256") != manifest["declared_cells_sha256"]:
        raise ValueError(
            f"{existing} declares {recorded.get('n_cells')} cells with digest "
            f"{recorded.get('declared_cells_sha256')!r}, but this run declares "
            f"{manifest['n_cells']} with {manifest['declared_cells_sha256']!r}. These are different "
            "campaigns; resuming across them would mix two cell sets under one directory. Point "
            "--work-dir somewhere else, or delete the old one deliberately"
        )


def campaign_status(work_dir: str | Path) -> dict[str, Any]:
    """Declared against present.  ``complete`` is a COMPUTED property, never a claim.

    A reader must be able to tell a partial run from a finished one without reading a log, so the
    answer is derived from the manifest and the files on disk every time it is asked.
    """
    work = Path(work_dir)
    manifest_path = work / MANIFEST_NAME
    # ⚠️ EVERY branch returns the SAME keys.  This branch used to omit `refused` and
    # `declared_cells_sha256`, so a caller reading the report had to know which shape it had got --
    # and a field that is sometimes absent is a field nobody can rely on.
    if not manifest_path.is_file():
        return {
            "declared": None,
            "present": len(list(work.glob("cell_*.json"))),
            "missing": None,
            "missing_sample": [],
            "refused": len(list(work.glob("refused_*.json"))),
            "complete": False,
            "marker_present": (work / COMPLETE_MARKER).is_file(),
            "declared_cells_sha256": None,
            "reason": f"no manifest at {manifest_path}; the campaign has not been declared here",
        }
    manifest = json.loads(manifest_path.read_bytes())
    declared = list(manifest.get("cells", ()))
    present: list[str] = []
    missing: list[str] = []
    for key in declared:
        scenario, arm, seed, draw = key.split("|")
        cell = CellKey(
            scenario=scenario,
            arm=arm,
            seed=None if seed == "None" else int(seed),
            draw_id=int(draw),
        )
        (present if (work / cell_file_name(cell)).is_file() else missing).append(key)
    complete = not missing
    return {
        "declared": len(declared),
        "present": len(present),
        "missing": len(missing),
        "missing_sample": missing[:5],
        "complete": complete,
        "refused": len(list(work.glob("refused_*.json"))),
        "marker_present": (work / COMPLETE_MARKER).is_file(),
        "declared_cells_sha256": manifest.get("declared_cells_sha256"),
        "reason": None if complete else f"{len(missing)} of {len(declared)} cells not yet rolled",
    }


def build_cell_factory(
    scenario: str, arm: str, seed: int | None, roots: Any, *, device: str | None, config_path: Any
) -> tuple[Any, dict[str, Any]]:
    """The action factory for one arm-seed, from the module that produced its committed cell.

    Every loader is imported, never reimplemented.  A wrong dispatch here loads a different policy,
    and :func:`assert_reproduces_committed` turns that into a refusal on the FIRST episode of the
    slot rather than a plausible number in a finished campaign.
    """
    method, tier = normalise_arm(scenario, arm)

    if method == BEHAVIOUR_METHOD:
        from offline.engine_att_reference import build_factory as behaviour_factory

        factory, source = behaviour_factory(
            scenario, tier, method, seed, roots, device=device, config_path=config_path
        )
        return factory, dict(source)

    if method in HEURISTIC_METHODS:
        from offline.method_tier_grid import _random_factory

        if seed is None:
            raise ValueError(f"the {method} arm is seeded and cannot take seed=None")
        return _random_factory(int(seed)), {
            "kind": "algorithmic",
            "detail": f"method_tier_grid._random_factory({int(seed)})",
        }

    if seed is None:
        raise ValueError(f"{arm} is a learned arm and cannot take seed=None")
    checkpoint = rederivation_checkpoint(scenario, tier, method, int(seed), roots)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{checkpoint} does not exist; {arm} seed {seed} cannot be replayed")

    if method == "dt":
        from offline.method_tier_grid import TIERS as HZ_TIERS
        from offline.method_tier_grid import _dt_factory

        target = float(HZ_TIERS[tier].target_rtg)
        return _dt_factory(str(checkpoint), DECLARED_GRADIENT_STEPS, target, device), {
            "kind": "checkpoint",
            "detail": f"method_tier_grid._dt_factory with the declared target_rtg {target}",
            "checkpoint": str(checkpoint),
        }

    if method.startswith("dt_spatial") or method.startswith("dt_nomix"):
        def spatial(env: Any) -> Any:
            from agent.SpatialDTAgent import SpatialDTAgent

            agent = SpatialDTAgent.from_checkpoint(env, str(checkpoint), device=device)
            return lambda _env, info: agent.act(info, explore=False, update_memory=True)

        return spatial, {
            "kind": "checkpoint",
            "detail": "agent.SpatialDTAgent.from_checkpoint",
            "checkpoint": str(checkpoint),
        }

    from offline.offline_baselines import _baseline_factory

    return _baseline_factory(method, str(checkpoint), DECLARED_GRADIENT_STEPS, device), {
        "kind": "checkpoint",
        "detail": "agent BC/IQL via offline_baselines._baseline_factory",
        "checkpoint": str(checkpoint),
    }


#: A run in which this many consecutive cells fail is not meeting bad cells, it is misconfigured.
#: Continuing would burn hours producing nothing, so the run stops and says which.  A SINGLE bad
#: cell can never trip this, which is the whole point.
SYSTEMIC_FAILURE_THRESHOLD = 25


def _write_refusal(
    cell: CellKey,
    exc: BaseException,
    reason_class: str,
    att_rederived: float | None,
    committed: Mapping[tuple[str, str, int | None, int], float],
    work: Path,
    protected: Sequence[Path],
) -> dict[str, Any]:
    """Record one refused cell durably, and return the record."""
    from offline.tier_sweep import assert_writable, write_json_guarded

    record = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "scenario": cell.scenario,
        "arm": cell.arm,
        "seed": cell.seed,
        "draw_id": cell.draw_id,
        "reason_class": reason_class,
        "reason": f"{type(exc).__name__}: {exc}",
        "att_ours_rederived": None if att_rederived is None else float(att_rederived),
        "att_ours_committed": committed.get((cell.scenario, cell.arm, cell.seed, cell.draw_id)),
    }
    refusal = work / refusal_file_name(cell)
    assert_writable(refusal, protected)
    write_json_guarded(record, refusal, protected)
    print(
        f"  REFUSED [{reason_class}] {cell.scenario}/{cell.arm} seed {cell.seed} "
        f"draw {cell.draw_id}: {type(exc).__name__}: {str(exc)[:140]}",
        flush=True,
    )
    return record


def run_campaign(
    cells: Sequence[CellKey],
    *,
    roots: Any,
    engine_seed: int,
    device: str | None,
    committed: Mapping[tuple[str, str, int | None, int], float],
    protected: Sequence[Path],
) -> dict[str, Any]:
    """Roll every cell, checking reproduction PER CELL, writing one file each.

    🔒 **EVERY per-cell step is inside the refusal boundary, not just the reproduction check.**
    The first version wrapped only :func:`assert_reproduces_committed`, so a ``ValueError`` from
    tier resolution, checkpoint loading or the rollout itself still aborted all five workers -- which
    is exactly what happened on ``mix33``.  A refusal must be about the CELL, whatever went wrong in
    it, or the guarantee is only as good as the list of exceptions somebody remembered.

    * **Resumable at episode granularity.**  A cell whose file already exists is skipped; a refused
      cell writes no cell file, so it is retried on the next resume and can never enter a verdict.
    * **A partial run is structurally distinguishable from a complete one** -- see
      :func:`campaign_status`, which recomputes completeness from disk.
    * **A systemic failure still stops.**  :data:`SYSTEMIC_FAILURE_THRESHOLD` consecutive refusals
      means the run is misconfigured rather than unlucky, and burning four hours to produce a
      directory of refusals helps nobody.  One bad cell cannot trip it.
    """
    from offline.admission_probe import created_from_flow, probe_episode
    from offline.materialise_draws import draw_config_path
    from offline.tier_sweep import assert_writable, write_json_guarded

    work = Path(roots.work_dir)
    scenario_keys = {"hz1x1": "cityflow1x1", "grid4x4": "cityflow_grid4x4"}

    settings_cache: dict[tuple[str, str], dict[str, Any]] = {}
    config_cache: dict[tuple[str, int], Path] = {}
    created_cache: dict[tuple[str, int, int], int] = {}
    factory_cache: dict[tuple[str, str, int | None], Any] = {}

    rolled = skipped = 0
    consecutive = 0
    refused: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, cell in enumerate(cells, start=1):
        destination = work / cell_file_name(cell)
        if destination.is_file():
            skipped += 1
            continue

        episode = None
        try:
            method, tier = normalise_arm(cell.scenario, cell.arm)
            tier_key = (cell.scenario, tier)
            if tier_key not in settings_cache:
                settings_cache[tier_key] = rederivation_env_settings(cell.scenario, tier, roots)
            settings = settings_cache[tier_key]
            horizon = int(settings["max_steps"]) * int(settings["delta_time"])

            config_key = (cell.scenario, int(cell.draw_id))
            if config_key not in config_cache:
                config_cache[config_key] = Path(
                    draw_config_path(
                        scenario_keys[cell.scenario], int(cell.draw_id), out_root=roots.draws_root
                    )
                )
            config = config_cache[config_key]
            created_key = (cell.scenario, int(cell.draw_id), horizon)
            if created_key not in created_cache:
                created_cache[created_key] = created_from_flow(
                    config.parent / "flow.json", horizon_seconds=horizon
                )

            factory_key = (cell.scenario, cell.arm, cell.seed)
            if factory_key not in factory_cache:
                factory_cache[factory_key] = build_cell_factory(
                    cell.scenario, cell.arm, cell.seed, roots, device=device, config_path=config
                )
            factory, source = factory_cache[factory_key]

            episode = probe_episode(
                scenario=cell.scenario,
                tier=tier,
                method=method,
                arm=cell.arm,
                seed=cell.seed,
                draw_id=int(cell.draw_id),
                config_path=config,
                env_settings=settings,
                scenario_id=scenario_keys[cell.scenario],
                choose_action_factory=factory,
                engine_seed=int(engine_seed),
                created=created_cache[created_key],
            )
            expected = assert_reproduces_committed(cell, episode.att_ours, committed)
        except KeyError as exc:
            refused.append(
                _write_refusal(cell, exc, "unverified", None, committed, work, protected)
            )
            consecutive += 1
        except Exception as exc:  # noqa: BLE001 - a cell's failure is data, not a reason to abort
            reason_class = (
                "does_not_reproduce"
                if episode is not None and "does not reproduce the committed" in str(exc)
                else "cell_failed"
            )
            refused.append(
                _write_refusal(
                    cell,
                    exc,
                    reason_class,
                    None if episode is None else episode.att_ours,
                    committed,
                    work,
                    protected,
                )
            )
            consecutive += 1
        else:
            consecutive = 0
            assert_writable(destination, protected)
            write_json_guarded(
                {
                    "format_version": ARTIFACT_FORMAT_VERSION,
                    **episode.as_record(),
                    "committed_att_ours": expected,
                    "reproduces_committed": True,
                    "policy_source": dict(source),
                },
                destination,
                protected,
            )
            rolled += 1
            if rolled % 25 == 0 or index == len(cells):
                rate = (time.perf_counter() - started) / max(rolled, 1)
                print(
                    f"  {index}/{len(cells)} rolled={rolled} skipped={skipped} "
                    f"refused={len(refused)} {rate:.2f} s/episode",
                    flush=True,
                )

        if consecutive >= SYSTEMIC_FAILURE_THRESHOLD:
            raise RuntimeError(
                f"{consecutive} consecutive cells failed, ending at {cell.scenario}/{cell.arm} "
                f"seed {cell.seed} draw {cell.draw_id}. That is a misconfiguration, not a run of "
                f"bad cells, and continuing would spend hours producing refusals. The refusal "
                f"records under {work} name each one. A single bad cell cannot reach this threshold"
            )

    return {
        "rolled": rolled,
        "skipped_already_present": skipped,
        "refused": refused,
        "n_refused": len(refused),
        "seconds": time.perf_counter() - started,
    }


# ----------------------------------------------------------------------
# DISCRIMINABILITY -- binding on every re-derived contrast
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ContrastDiscriminability:
    """Whether the two arms of a contrast are distinguishable AT ALL on this tier.

    🔒 **BINDING, ruled 2026-08-31.**  Every re-derived contrast reports this beside its verdict.
    When two arms produce identical per-draw ATT, the honest statement is **"the tier cannot
    discriminate"** and NOT "no difference was found": the second implies a measurement was made
    that could have come out otherwise, and on such a tier it could not.

    Measured on grid4x4's ``fixedtime`` tier, which is where this rule comes from: 4 of 7 arms carry
    distinct ATT vectors, ``behaviour``, ``dt_nomix`` and ``dt_spatial`` are mutually identical, and
    ``bc`` is identical to ``bc_top10_perix``.  P5.1's P2a contrast -- ``mean(ATT_spatial -
    ATT_nomix)`` -- is therefore **identically zero** on that rung, by construction rather than by
    measurement.  The same check over ``mappo1000`` (6/6), ``maxpressure`` (7/7) and ``random``
    (6/6) finds every arm distinct, so the collapse is confined to ``fixedtime``.
    """

    tier: str
    left: str
    right: str
    n_paired: int
    n_differing: int
    max_abs_difference: float
    identical: bool
    status: str
    statement: str


def contrast_discriminability(
    left_values: Mapping[Any, float],
    right_values: Mapping[Any, float],
    *,
    tier: str,
    left: str,
    right: str,
) -> ContrastDiscriminability:
    """Compare two arms' per-episode ATT vectors on their SHARED keys.

    Refuses an empty overlap: a contrast over no shared episode is not a contrast, and reporting
    ``identical`` for it would be the strongest possible claim from the weakest possible evidence.
    """
    shared = sorted(set(left_values) & set(right_values), key=repr)
    if not shared:
        raise ValueError(
            f"{left} and {right} on tier {tier} share no episode, so there is nothing to compare; "
            "an empty overlap must refuse rather than report 'identical'"
        )
    differences = [float(left_values[k]) - float(right_values[k]) for k in shared]
    n_differing = sum(1 for d in differences if d != 0.0)
    identical = n_differing == 0
    return ContrastDiscriminability(
        tier=str(tier),
        left=str(left),
        right=str(right),
        n_paired=len(shared),
        n_differing=n_differing,
        max_abs_difference=max(abs(d) for d in differences),
        identical=identical,
        status="cannot_discriminate" if identical else "distinct",
        statement=(
            f"{left} and {right} produce IDENTICAL ATT on all {len(shared)} shared episodes of "
            f"tier {tier}; this tier cannot discriminate between them, so any contrast is zero by "
            "construction and not by measurement"
            if identical
            else f"{left} and {right} differ on {n_differing} of {len(shared)} shared episodes of "
            f"tier {tier} (max |difference| {max(abs(d) for d in differences)!r})"
        ),
    )


def assert_contrast_reports_discriminability(payload: Any) -> None:
    """Refuse a contrast block that does not carry its discriminability.

    The binding rule is only binding if something checks it.  Every mapping that looks like a
    contrast -- one carrying a ``verdict`` or an ``outcome`` alongside a ``tier`` -- must also carry
    ``discriminability``, so a verdict can never be reported without saying whether the tier it sits
    on could have produced a different answer.
    """

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            # ⚠️ The trigger must not key on one spelling.  It originally keyed on "verdict",
            # and renaming the emitted field to "contrast_id" -- forced by
            # method_tier_grid.assert_no_verdicts, which refuses any key containing "verdict" --
            # silently switched this guard OFF until a test caught it.  It now recognises every
            # shape a contrast is written in.
            looks_like_contrast = "tier" in node and bool(
                {"verdict", "outcome", "pooled", "contrast_id"} & set(node)
            )
            if looks_like_contrast and "discriminability" not in node:
                raise ValueError(
                    f"{path}: this contrast reports a verdict on tier {node.get('tier')!r} without "
                    "its discriminability. 'no difference was found' and 'the tier cannot "
                    "discriminate' are different claims and the second must never be written as "
                    "the first (ruled 2026-08-31)"
                )
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(payload, "$")


# ----------------------------------------------------------------------
# THE VERDICT LAYER -- both definitions, both poolings, discriminability wired in
# ----------------------------------------------------------------------


#: The two ATT definitions every verdict is re-evaluated under (A11(d): both are reported whether or
#: not they agree).
DEFINITIONS: tuple[str, ...] = ("att_ours", "att_engine")

#: The two poolings, ruled 2026-08-31.  **Never choose one.**  Pooling a tier whose arms are
#: structurally identical injects a known zero into the statistic, which is a statistical error;
#: dropping that tier AFTER seeing the zero is a post-hoc exclusion that flatters us.  Both
#: directions are researcher degrees of freedom.  Reporting both is not.
POOLINGS: tuple[str, ...] = ("including_non_distinct", "excluding_non_distinct")


@dataclass(frozen=True)
class PooledContrast:
    """One contrast, under one definition, under one pooling."""

    definition: str
    pooling: str
    tiers: tuple[str, ...]
    n_paired: int
    mean_difference: float
    ci95_half_width: float
    ci95_low: float
    ci95_high: float
    verdict: str


@dataclass(frozen=True)
class ContrastReport:
    """A11(d) verdict re-derived under both definitions and both poolings.

    ⛔ **``escalate`` is a FLAG, never a resolution.**  A verdict that differs between the two
    poolings is the coordinator's to rule on: the whole point of reporting both is that choosing
    between them is a degree of freedom this module does not have.
    """

    verdict_id: str
    name: str
    left: str
    right: str
    discriminability: tuple[ContrastDiscriminability, ...]
    tiers_non_distinct: tuple[str, ...]
    structural_reason: str
    pooled: tuple[PooledContrast, ...]
    poolings_agree: Mapping[str, bool]
    escalate: bool
    escalation_reason: str | None

    def as_record(self) -> dict[str, Any]:
        """The JSON block.  Carries ``tier`` and ``discriminability`` so the guard can see both."""
        return {
            # NOT "verdict_id": method_tier_grid.assert_no_verdicts refuses any key containing
            # "verdict", and it is right to -- this artifact reports contrasts and their outcomes.
            "contrast_id": self.verdict_id,
            "name": self.name,
            "left": self.left,
            "right": self.right,
            "tier": "+".join(d.tier for d in self.discriminability) or "none",
            "discriminability": [
                {
                    "tier": d.tier,
                    "status": d.status,
                    "identical": d.identical,
                    "n_paired": d.n_paired,
                    "n_differing": d.n_differing,
                    "max_abs_difference": d.max_abs_difference,
                    "statement": d.statement,
                }
                for d in self.discriminability
            ],
            "tiers_non_distinct": list(self.tiers_non_distinct),
            "structural_reason": self.structural_reason,
            "pooled": [
                {
                    "definition": p.definition,
                    "pooling": p.pooling,
                    "tiers": list(p.tiers),
                    "n_paired": p.n_paired,
                    "mean_difference": p.mean_difference,
                    "ci95_half_width": p.ci95_half_width,
                    "ci95_low": p.ci95_low,
                    "ci95_high": p.ci95_high,
                    "outcome": p.verdict,
                }
                for p in self.pooled
            ],
            "poolings_agree": dict(self.poolings_agree),
            "escalate": self.escalate,
            "escalation_reason": self.escalation_reason,
        }


def load_cells(work_dir: str | Path) -> dict[str, dict[str, dict[tuple, dict[str, float]]]]:
    """``{tier: {arm: {(seed, draw): {definition: value}}}}`` from the campaign's cell files.

    Reads only ``cell_*.json``: a refusal record is not data and must never reach a verdict.
    """
    cells: dict[str, dict[str, dict[tuple, dict[str, float]]]] = {}
    for path in sorted(Path(work_dir).glob("cell_*.json")):
        row = json.loads(path.read_bytes())
        arm = str(row["arm"])
        _, tier = normalise_arm(str(row["scenario"]), arm)
        key = (row.get("seed"), int(row["draw_id"]))
        cells.setdefault(tier, {}).setdefault(arm, {})[key] = {
            "att_ours": float(row["att_ours"]),
            "att_engine": float(row["att_engine"]),
        }
    return cells


def pool_contrast(
    cells: Mapping[str, Mapping[str, Mapping[tuple, Mapping[str, float]]]],
    *,
    verdict_id: str,
    name: str,
    left: str,
    right: str,
    tiers: Sequence[str],
    verdict_fn: Callable[[float, float, float, float], str],
) -> ContrastReport:
    """Re-derive one contrast under BOTH definitions and BOTH poolings.

    *verdict_fn* is the REGISTERED decision function of that verdict, imported from the module that
    carries it and passed in by the caller -- never reimplemented here, because a second
    implementation of a decision rule is how two verdicts stop being comparable.
    """
    from offline.dt_gate import mean_ci95

    discriminability: list[ContrastDiscriminability] = []
    for tier in tiers:
        block = cells.get(tier, {})
        left_arm, right_arm = f"{left}@{tier}", f"{right}@{tier}"
        if left_arm not in block or right_arm not in block:
            continue
        discriminability.append(
            contrast_discriminability(
                {k: v["att_ours"] for k, v in block[left_arm].items()},
                {k: v["att_ours"] for k, v in block[right_arm].items()},
                tier=tier,
                left=left,
                right=right,
            )
        )
    if not discriminability:
        raise ValueError(
            f"{verdict_id}: no tier carries both {left} and {right}, so the contrast has no "
            "evidence; an empty contrast must refuse rather than report a verdict"
        )

    non_distinct = tuple(d.tier for d in discriminability if d.identical)
    distinct = tuple(d.tier for d in discriminability if not d.identical)

    pooled: list[PooledContrast] = []
    for definition in DEFINITIONS:
        for pooling in POOLINGS:
            chosen = (
                tuple(d.tier for d in discriminability)
                if pooling == "including_non_distinct"
                else distinct
            )
            differences: list[float] = []
            for tier in chosen:
                block = cells[tier]
                left_values = block[f"{left}@{tier}"]
                right_values = block[f"{right}@{tier}"]
                for key in sorted(set(left_values) & set(right_values), key=repr):
                    differences.append(
                        float(left_values[key][definition]) - float(right_values[key][definition])
                    )
            if not differences:
                pooled.append(
                    PooledContrast(
                        definition=definition,
                        pooling=pooling,
                        tiers=chosen,
                        n_paired=0,
                        mean_difference=float("nan"),
                        ci95_half_width=float("nan"),
                        ci95_low=float("nan"),
                        ci95_high=float("nan"),
                        verdict="NO EVIDENCE",
                    )
                )
                continue
            stats = mean_ci95(differences)
            low, high = stats.mean - stats.ci95, stats.mean + stats.ci95
            pooled.append(
                PooledContrast(
                    definition=definition,
                    pooling=pooling,
                    tiers=chosen,
                    n_paired=len(differences),
                    mean_difference=float(stats.mean),
                    ci95_half_width=float(stats.ci95),
                    ci95_low=float(low),
                    ci95_high=float(high),
                    verdict=verdict_fn(float(stats.mean), float(stats.ci95), float(low), float(high)),
                )
            )

    agree: dict[str, bool] = {}
    for definition in DEFINITIONS:
        outcomes = {p.verdict for p in pooled if p.definition == definition}
        agree[definition] = len(outcomes) == 1
    escalate = not all(agree.values())

    if non_distinct:
        reason = (
            f"tiers {list(non_distinct)} cannot discriminate {left} from {right}: their per-episode "
            "ATT is identical, so the contrast there is zero BY CONSTRUCTION. Pooling that tier "
            "injects a structural zero; dropping it after seeing the zero is a post-hoc exclusion. "
            "Both poolings are reported and neither is chosen (ruled 2026-08-31)"
        )
    else:
        reason = (
            f"every tier compared discriminates {left} from {right}, so the two poolings are the "
            "same set and are reported identically"
        )

    return ContrastReport(
        verdict_id=verdict_id,
        name=name,
        left=left,
        right=right,
        discriminability=tuple(discriminability),
        tiers_non_distinct=non_distinct,
        structural_reason=reason,
        pooled=tuple(pooled),
        poolings_agree=agree,
        escalate=escalate,
        escalation_reason=(
            f"the verdict DIFFERS between the two poolings for "
            f"{sorted(d for d, ok in agree.items() if not ok)}; choosing between them is a "
            "researcher degree of freedom and is the coordinator's ruling, not this module's"
            if escalate
            else None
        ),
    )


def rederivation_artifact(
    contrasts: Sequence[ContrastReport], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Assemble the re-derivation artifact, with BOTH guards actually invoked.

    🔒 ``assert_contrast_reports_discriminability`` is called HERE, on the assembled payload.  An
    enforcement function nobody calls is ``DEFERRED`` 42/44's class -- a guard that exists in the
    source and never in the execution -- so the artifact cannot be built without it passing.
    """
    from offline.admission_probe import assert_no_science_verdict, code_provenance

    if not contrasts:
        raise ValueError("the re-derivation artifact carries no contrast, so it re-derives nothing")

    escalations = [c.verdict_id for c in contrasts if c.escalate]
    payload: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "P8.4b: A11(d)'s verdicts re-derived under BOTH ATT definitions and BOTH poolings of "
            "tiers whose arms are structurally non-distinct"
        ),
        "what_this_does_not_say": [
            "no verdict on which ATT definition is primary; Rule R decides that",
            "no choice between the two poolings: reporting both is the ruling, and a verdict that "
            "differs between them is escalated rather than resolved here",
            "a tier whose arms are identical CANNOT DISCRIMINATE; that is never reported as "
            "'no difference was found'",
        ],
        "definitions": list(DEFINITIONS),
        "poolings": list(POOLINGS),
        "contrasts": [c.as_record() for c in contrasts],
        "escalations": escalations,
        "n_escalations": len(escalations),
        "provenance": {**dict(provenance), "code_provenance": code_provenance()},
    }
    assert_contrast_reports_discriminability(payload)
    assert_no_science_verdict(payload)
    return payload


def rederivation_corpus_dirs(scenario: str, tier: str, roots: Any) -> tuple[Path, ...]:
    """The corpus directories of any tier this campaign touches, mixtures included.

    ⚠️ **Gate 0's ``tier_corpus_dirs`` covers the SEVEN BEHAVIOUR TIERS and nothing else**, which is
    correct for Gate 0 and wrong here: this campaign also rolls P4.7's ``mix33`` / ``mix50`` /
    ``mix67``, whose corpus is a BLEND and has no ``cf_hz1x1__mix33`` directory to find.  Calling the
    Gate 0 helper on a mixture aborted all five workers.

    ``method_tier_grid.TIERS`` already declares the directory list of every hz1x1 tier, mixtures
    included -- ``mix33.dirs`` is the five ``mappo1000`` seeds plus ``random`` -- so that registered
    declaration is used rather than a second one invented here.
    """
    from offline.method_tier_grid import TIERS as HZ_TIERS

    if scenario == "hz1x1" and tier in HZ_TIERS:
        paths = tuple(Path(roots.corpus_root) / name for name in HZ_TIERS[tier].dirs)
        missing = [str(p) for p in paths if not (p / "manifest.json").is_file()]
        if missing:
            raise FileNotFoundError(
                f"{scenario}/{tier}: these corpus directories have no manifest.json: {missing}"
            )
        return paths

    from offline.engine_att_reference import tier_corpus_dirs

    return tier_corpus_dirs(scenario, tier, roots)


def rederivation_env_settings(scenario: str, tier: str, roots: Any) -> dict[str, Any]:
    """Evaluation env settings for any tier, read from its own collection manifests.

    Every directory of the tier must agree; a disagreement raises rather than picking the first,
    because the settings decide what the episode IS.  A mixture's directories span two component
    corpora, and they are asserted to agree exactly as a single tier's are.
    """
    from offline.dt_gate import env_settings_from_manifest

    seen: dict[str, list[str]] = {}
    settings: dict[str, Any] | None = None
    for directory in rederivation_corpus_dirs(scenario, tier, roots):
        candidate = env_settings_from_manifest(directory / "manifest.json")
        key = json.dumps(candidate, sort_keys=True, default=str)
        seen.setdefault(key, []).append(str(directory))
        if settings is None:
            settings = candidate
    if settings is None:
        raise ValueError(f"{scenario}/{tier} names no corpus directory, so it has no env settings")
    if len(seen) > 1:
        summary = {sorted(paths)[0]: len(paths) for paths in seen.values()}
        raise ValueError(
            f"{scenario}/{tier}'s corpus directories disagree on the evaluation env settings "
            f"({summary}); picking one would compare two different episodes under one tier name"
        )
    return settings


if __name__ == "__main__":  # pragma: no cover - exercised by a subprocess test, not by import
    raise SystemExit(main())
