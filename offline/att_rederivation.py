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
    "CellKey",
    "VerdictSource",
    "build_parser",
    "cost_estimate",
    "factory_resolution",
    "main",
    "preflight",
    "rederivation_cells",
    "slots_from_episode_block",
]

ARTIFACT_FORMAT_VERSION = "p8.4b-rederivation/1.0"

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
    for (scenario, arm, seed), draws in sorted(slots.items(), key=lambda kv: kv[0]):
        method, at, tier = arm.partition("@")
        record = {
            "scenario": scenario,
            "arm": arm,
            "method": method,
            "tier": tier,
            "seed": seed,
            "n_draws": len(draws),
        }
        if scenario not in MEASURED_SECONDS_PER_EPISODE:
            unresolvable.append({**record, "reason": f"no probe path for scenario {scenario!r}"})
            continue
        if not at:
            # P4.4/P4.5 name their arms bare -- 'madt', 'bc', 'mappo1000', 'maxpressure' -- with the
            # tier implied by the artifact rather than carried in the string.  That is a naming
            # convention to resolve, not a missing policy, and it is reported as its own class so it
            # is not mistaken for one.
            unresolvable.append(
                {**record, "reason": "bare arm name carries no tier; the tier is implied by the "
                                     "source artifact and must be resolved from it"}
            )
            continue
        try:
            build_factory(
                scenario,
                tier,
                method,
                seed,
                roots,
                device=None,
                config_path=configs.get(scenario),
            )
        except Exception as exc:  # noqa: BLE001 - the reason is the deliverable
            unresolvable.append({**record, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        resolvable.append(record)

    return {
        "n_slots": len(slots),
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
