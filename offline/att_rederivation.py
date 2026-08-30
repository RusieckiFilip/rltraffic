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
from typing import Any, Iterable, Mapping, Sequence

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
    index: dict[tuple[str, str, int | None, int], float] = {}
    for source in REDERIVATION_SOURCES:
        for path in _resolve_sources(source, repo_root=repo, output_root=out):
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
                index[key] = float(value)
    return index


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.  Non-zero when the pre-flight is NOT clear to run."""
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    work_dir = Path(args.work_dir) if args.work_dir else output_root / "p8_4b_rederivation"

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


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
