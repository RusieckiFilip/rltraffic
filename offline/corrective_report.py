"""P9.1's report (``BRIEF_46`` 3.4 ``rescore`` / ``report``, 3.5): the five A27 artifacts, pure functions of the run
directories and the committed artifacts.

Format ``p9.1-report/1.0``.  Every per-episode record carries ``arm``, ``tier``, ``seed``, ``draw_id``, ``att_ours``,
``att_engine``, ``entered``, ``created``, ``never_entered``, ``entered_fraction``, ``episode_reward``,
``horizon_vehicle_count`` and ``source``.  **Alignment convention:** one record per held-out ``(seed, draw)`` of a cell,
in ``(seed, draw)`` order; ``att_ours``, ``horizon_vehicle_count`` and ``episode_reward`` are route (i)'s (the registered
evaluation: ``method_tier_grid`` / ``tier_sweep`` ``evaluate``) and ``att_engine`` with the companions are route (ii)'s
(P8.4b's per-second observer, ``att_rederivation.run_campaign``), joined on ``(seed, draw)`` and refused unless route
(ii)'s ``att_ours`` equals route (i)'s under ``==``.  An OLD hz1x1 cell's ``att_ours`` is its committed artifact's,
joined to its P8.4b re-roll the same way (A27(a)).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import offline.corrective_round as cr

REPORT_FORMAT_VERSION = "p9.1-report/1.0"

#: The fields of every per-episode record (``BRIEF_46`` 3.5), ``entered_fraction`` added beside the companions.
EPISODE_FIELDS: tuple[str, ...] = (
    "arm", "tier", "seed", "draw_id", "att_ours", "att_engine", "entered", "created", "never_entered",
    "entered_fraction", "episode_reward", "horizon_vehicle_count", "source",
)


def _relative(roots: cr.Roots, path: str | Path) -> str:
    """A path under the main tree's ``output/``, written as the repository names it."""
    return f"output/{Path(path).resolve().relative_to(Path(roots.output_root).resolve()).as_posix()}"


def _seed_of(value: Any) -> int | None:
    return None if value is None else int(value)


def _joined_row(arm: str, tier: str, first: Mapping[str, Any], rerolled: Mapping[str, Any], *, att_ours: float,
                source: str, refusal: str) -> dict[str, Any]:
    """One record: *att_ours* and *first*'s two route-(i) companions, *rerolled*'s ``att_engine`` and A11(b)'s
    companions -- refused unless *rerolled* is the same episode and its ``att_ours`` (and the value its own run was
    checked against) equal *att_ours* under ``==``."""
    seed, draw = _seed_of(first.get("seed")), int(first["draw_id"])
    if (str(rerolled["arm"]), _seed_of(rerolled.get("seed")), int(rerolled["draw_id"])) != (arm, seed, draw):
        raise ValueError(f"{arm} seed {seed} draw {draw}: the P8.4b file holds {rerolled['arm']} seed "
                         f"{rerolled.get('seed')} draw {rerolled['draw_id']}")
    if not (float(rerolled["att_ours"]) == att_ours and float(rerolled["committed_att_ours"]) == att_ours
            and rerolled["reproduces_committed"] is True):
        raise ValueError(f"{arm} seed {seed} draw {draw}: {refusal} (att_ours {att_ours!r} against "
                         f"{rerolled['att_ours']!r}, checked against {rerolled['committed_att_ours']!r})")
    return {
        "arm": arm, "tier": tier, "seed": seed, "draw_id": draw, "att_ours": att_ours,
        "att_engine": float(rerolled["att_engine"]), "entered": int(rerolled["entered"]),
        "created": int(rerolled["created"]), "never_entered": int(rerolled["never_entered"]),
        "entered_fraction": float(rerolled["entered_fraction"]), "episode_reward": float(first["episode_reward"]),
        "horizon_vehicle_count": float(first["horizon_vehicle_count"]), "source": source,
    }


def _rerolled(work: Path, scenario: str, arm: str, seed: int | None, draw: int, missing: str) -> dict[str, Any]:
    from offline import att_rederivation as ar

    path = work / ar.cell_file_name(ar.CellKey(scenario, arm, seed, draw))
    if not path.is_file():
        raise ValueError(f"{scenario}:{arm} seed {seed} draw {draw}: {missing} ({path})")
    return json.loads(path.read_text(encoding="utf-8"))


def new_cell_rows(roots: cr.Roots, cell: cr.Cell) -> tuple[dict[str, Any], ...]:
    """A NEW cell's records: route (i)'s complete cell file joined to route (ii)'s per-episode files."""
    path = cr.route_i_output(roots, cell)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cr._assert_route_i_complete(payload, cell)
    work = Path(roots.runs_root) / "eval_ii" / cell.scenario
    source = f"new: route (i) {_relative(roots, path)}; route (ii) {_relative(roots, work)}, att_ours equal under =="
    rows = []
    for episode in sorted(payload["episodes"], key=lambda e: (int(e["seed"]), int(e["draw_id"]))):
        seed, draw = int(episode["seed"]), int(episode["draw_id"])
        rerolled = _rerolled(work, cell.scenario, cell.key, seed, draw, "route (ii) has no episode")
        rows.append(_joined_row(cell.key, cell.tier, episode, rerolled, att_ours=float(episode["att_horizon"]),
                                source=source, refusal="the two routes disagree"))
    return tuple(rows)


def old_cell_tier(arm: str) -> str:
    """An old hz1x1 arm's tier, by P8.4b's own name resolution (P4.4's bare names are ``mappo1000``'s)."""
    from offline import att_rederivation as ar

    return ar.normalise_arm("hz1x1", arm)[1]


def old_hz1x1_rows(roots: cr.Roots, artifact: str, arm: str) -> tuple[dict[str, Any], ...]:
    """An OLD hz1x1 cell's records: its committed artifact's episodes joined to their P8.4b re-rolls
    (``output/p9_1_runs/old_cells/hz1x1``), each re-roll refused unless it reproduces the committed ``att_ours``."""
    payload = json.loads((Path(roots.repo_root) / artifact).read_text(encoding="utf-8"))
    episodes = sorted((e for e in payload["episodes"] if str(e["arm"]) == arm),
                      key=lambda e: (-1 if e.get("seed") is None else int(e["seed"]), int(e["draw_id"])))
    if not episodes:
        raise ValueError(f"{artifact} carries no episodes of {arm!r}")
    work = Path(roots.runs_root) / "old_cells" / "hz1x1"
    tier = old_cell_tier(arm)
    source = f"{artifact}; att_engine and companions re-rolled, reproduces committed att_ours"
    rows = []
    for episode in episodes:
        rerolled = _rerolled(work, "hz1x1", arm, _seed_of(episode.get("seed")), int(episode["draw_id"]),
                             "not re-rolled")
        rows.append(_joined_row(arm, tier, episode, rerolled, att_ours=float(episode["att_horizon"]), source=source,
                                refusal="the re-roll does not reproduce the committed att_ours"))
    keys = [(row["seed"], row["draw_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{artifact}: an episode of {arm} appears twice")
    return tuple(rows)


#: The keys of P5.2c's cell rows (``tier_statements.Cell``), in which the scorers read every cell.
STATEMENT_ROW_KEYS: tuple[str, ...] = (
    "arm", "seed", "draw_id", "att_ours", "att_engine", "entered", "created", "never_entered", "entered_fraction",
    "horizon_vehicle_count", "episode_reward",
)


def as_statement_cell(cell_id: str, method: str, tier: str, arm: str, source: str, eval_file: str,
                      rows: Sequence[Mapping[str, Any]]) -> Any:
    """*rows* as P5.2c's ``tier_statements.Cell`` -- the type every registered scorer reads -- with P5.2c's row keys."""
    import offline.tier_statements as tst

    spec = tst.CellSpec(cell_id=cell_id, method=method, tier=tier, source=source, arm=arm, eval_file=eval_file,
                        manifest="none: a P9.1 run directory, digested in the artifact's inputs")
    return tst.Cell(spec=spec, rows=tuple({key: row[key] for key in STATEMENT_ROW_KEYS}
                                          for row in sorted(rows, key=lambda r: (r["seed"], r["draw_id"]))))


def new_cell_id(cell: cr.Cell) -> str:
    """``p9_1:<scenario>:<arm>@<tier>`` -- the scenario is part of the id, because ``<arm>@<tier>`` alone names two
    cells (``iql_t@mappo1000`` and ``behaviour@random_fresh`` exist on both scenarios)."""
    return f"p9_1:{cell.scenario}:{cell.key}"


def statement_cell(roots: cr.Roots, cell: cr.Cell) -> Any:
    """A NEW cell as a ``tier_statements.Cell``: id :func:`new_cell_id`, method the A27 arm."""
    return as_statement_cell(new_cell_id(cell), cell.arm, cell.tier, cell.key, "p9_1",
                             _relative(roots, cr.route_i_output(roots, cell)), new_cell_rows(roots, cell))


def new_training_runs(roots: cr.Roots, cell: cr.Cell, *, seeds: Any = None) -> list[dict[str, Any]]:
    """Every run of a new cell: ``seconds`` and ``gradient_steps`` from its write-once record, the uncontrolled-clock
    label (A27(a)) refused unless the record carries it."""
    from offline.dt_gate import TRAINING_SEEDS

    runs = []
    for seed in (TRAINING_SEEDS if seeds is None else seeds):
        path = cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, int(seed))
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("clock") != cr.CLOCK_LABEL or int(record["seed"]) != int(seed):
            raise ValueError(f"{path} is not a p9.1 run record of seed {seed} with A27(a)'s clock label")
        runs.append({"scenario": cell.scenario, "tier": cell.tier, "arm": cell.arm, "seed": int(seed),
                     "seconds": float(record["seconds"]), "gradient_steps": int(record["gradient_steps"]),
                     "clock": cr.CLOCK_LABEL, "source": _relative(roots, path)})
    return runs


# ----------------------------------------------------------------------
# The artifacts (BRIEF_46 3.5): pure functions of a Sources
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Sources:
    """Everything the five artifacts are built from, loaded once: the cells (``tier_statements.Cell``) by id -- new
    (``p9_1:<scenario>:<arm>@<tier>``), hz1x1 old (``p4_4:<arm>``, ``p4_6:<arm>``, ``p4_7:<arm>``), grid old (P5.2c's
    ids) -- with their per-episode records and training runs, the declarations, P5.2c's books, P3a's rows, the tiers'
    training schedules, the D4RL re-run, registered constants, the inputs by digest and the git block."""

    cells: Mapping[str, Any]
    records: Mapping[str, Sequence[Mapping[str, Any]]]
    runs: Mapping[str, Sequence[Mapping[str, Any]]]
    declarations: Mapping[str, Any]
    grid_books: Mapping[str, Mapping[tuple[str, str], Any]]
    p3a: Mapping[str, Sequence[Mapping[str, Any]]]
    schedules: Mapping[str, Mapping[str, Any]]
    d4rl: Mapping[str, Any] | None
    registered: Mapping[str, Any]
    inputs: Sequence[Mapping[str, Any]]
    git: Mapping[str, Any]


REGISTRATION = "PREREGISTRATION.md A27 (tag v2.8-prereg-a27)"


def _referenced(node: Any) -> list[str]:
    """Every cell id an artifact block names (``left`` / ``right`` / ``inputs`` / ``original_inputs`` / ``cell``)."""
    found: list[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key in ("left", "right", "cell") and isinstance(value, str):
                found.append(value)
            elif key in ("inputs", "original_inputs") and isinstance(value, list):
                found.extend(v for v in value if isinstance(v, str))
            else:
                found.extend(_referenced(value))
    elif isinstance(node, (list, tuple)):
        for item in node:
            found.extend(_referenced(item))
    return found


def _finish(point: str, src: Sources, body: Mapping[str, Any], *, role: str,
            what_this_is_not: Sequence[str]) -> dict[str, Any]:
    """The common block, then the per-episode records and the training runs of EVERY cell the body names (A27(a): the
    records of every new cell and every old cell it is compared with)."""
    ids = sorted({cid for cid in _referenced(body) if cid in src.cells})
    missing = [cid for cid in ids if cid not in src.records]
    if missing:
        raise ValueError(f"{point}: no per-episode records for {missing[:3]}; every compared cell carries them")
    return {
        "format_version": REPORT_FORMAT_VERSION, "label": cr.LABELS[point], "registered_in": f"{point}, {REGISTRATION}",
        "role": role, "git": dict(src.git), "inputs": [dict(entry) for entry in src.inputs], **body,
        "records": {cid: [{key: row[key] for key in EPISODE_FIELDS} for row in src.records[cid]] for cid in ids},
        "training_runs": {cid: [dict(run) for run in src.runs[cid]] for cid in ids if cid in src.runs},
        "clock": cr.CLOCK_LABEL, "what_this_is_not": list(what_this_is_not),
    }


def _levels(cells: Mapping[str, Any]) -> dict[str, Any]:
    """Each cell's level under both definitions: the mean over its episodes by ``dt_gate.mean_ci95`` (P5.2c's level)."""
    from offline.dt_gate import mean_ci95
    import offline.tier_statements as tst

    return {name: {"cell": cell.spec.cell_id, **{definition: float(mean_ci95(
        [e.att_horizon for e in tst.episodes_of(cell, definition)]).mean) for definition in ("att_engine", "att_ours")}}
        for name, cell in cells.items()}


MIXTURE_TIER = "mix_m1000_m500"
#: The parents' registered cells (A27(e) quotes their order: P4.4's mappo1000 cells, which P4.6 re-used, and P4.6's
#: mappo500 cells).
MIXTURE_PARENTS: Mapping[str, Mapping[str, str]] = {
    "mappo1000": {"dt": "p4_4:madt", "bc": "p4_4:bc", "bc_top10": "p4_4:bc_top10", "iql": "p4_4:iql"},
    "mappo500": {"dt": "p4_6:dt@mappo500", "bc": "p4_6:bc@mappo500", "bc_top10": "p4_6:bc_top10@mappo500",
                 "iql": "p4_6:iql@mappo500"},
}


def build_mixture(src: Sources) -> dict[str, Any]:
    """``p9_1_mixture.json`` (A27(e)): the declaration, the four contrasts, the 5-s window, ``iql_t`` beside, the
    parents' levels beside."""
    import offline.corrective_verdicts as cv

    cells = {arm: src.cells[f"p9_1:hz1x1:{arm}@{MIXTURE_TIER}"] for arm in ("dt", "bc", "bc_top10", "iql", "iql_t")}
    body = {
        "declaration": src.declarations[MIXTURE_TIER],
        "predictions": {"orderings": cv.mixture_orderings(cells, tier=MIXTURE_TIER),
                        "within_5_s": cv.within_seconds(cells, tier=MIXTURE_TIER)},
        "beside": {"iql_t_within_5_s": cv.within_seconds(cells, tier=MIXTURE_TIER, arms=("iql_t",))},
        "levels": _levels(cells),
        "parents": {tier: _levels({arm: src.cells[cid] for arm, cid in arms.items()})
                    for tier, arms in MIXTURE_PARENTS.items()},
    }
    return _finish("A27(e)", src, body, role="A27(e): hz1x1's overlap mixture mappo1000 + mappo500 at 0.50, five arms",
                   what_this_is_not=(
                       "not a test of the filter on the original mixtures: their random halves replay two schedules "
                       "(A27(d)); mix_m1000_m500 replaces them for the selection question",
                       "not a new confirmatory family: the four contrasts and the window are tests registered after "
                       "results (A27(e)), judged by P5.2c's rule",
                       "iql_t is reported beside, not predicted"))


#: A27(g)'s registered 40,000-step cells per tier (P4.4's mappo1000 cells; P4.7's mix50 cells).
BUDGET_40K: Mapping[str, Mapping[str, str]] = {
    "mappo1000": MIXTURE_PARENTS["mappo1000"],
    "mix50": {"dt": "p4_7:dt@mix50", "bc": "p4_7:bc@mix50", "bc_top10": "p4_7:bc_top10@mix50", "iql": "p4_7:iql@mix50"},
}
BUDGET_SCOPE: Mapping[str, str] = {
    "mappo1000": "A8(b)'s secondary (the registered ceiling of 100,000 steps)",
    "mix50": "an exploratory extension beyond A8(b)",
}


def build_budget(src: Sources) -> dict[str, Any]:
    """``p9_1_budget.json`` (A27(g)): per tier the paired 100k - 40k differences and the concordance; the plateau
    diagnostic at 100,000; the counts the paper quotes."""
    import offline.corrective_verdicts as cv

    tiers = {}
    for tier, old in BUDGET_40K.items():
        at_40k = {arm: src.cells[cid] for arm, cid in old.items()}
        at_100k = {arm: src.cells[f"p9_1:hz1x1:{arm}_100k@{tier}"] for arm in cv.BUDGET_ARMS}
        tiers[tier] = {**cv.budget_block(at_40k, at_100k, tier=tier), "scope": BUDGET_SCOPE[tier]}
    plateau = [{"cell": cid, "seed": int(run["seed"]), "plateaued": run["plateaued"],
                "window_means": list(run["window_means"])}
               for cid in sorted(c for c in src.cells if c.startswith("p9_1:hz1x1:") and "_100k@" in c)
               for run in src.runs[cid]]
    body = {"tiers": tiers, "plateau_at_100k": plateau,
            "paper_counts_at_40k": dict(src.registered["plateau_counts"]) if "plateau_counts" in src.registered else None}
    return _finish("A27(g)", src, body, role="A27(g): A8(b)'s budget-sensitivity secondary at 100,000 steps",
                   what_this_is_not=(
                       "not a new primary: the 40,000-step cells stay the registered primary, the 100,000-step cells "
                       "are A8(b)'s secondary beside them",
                       "mix50 goes beyond A8(b) and is an exploratory extension",
                       "iql_t is not run at 100,000 steps (a fifth arm would be a new design, not A8(b)'s)"))


def plateau_counts(repo_root: str | Path) -> dict[str, int]:
    """A27(g)'s counts over the committed training records, by the author's command of 2026-10-15: runs; not
    plateaued; of those, still falling over the last three windows, rose in the last window, fluctuated."""
    runs = [run for name, key in (("p4_training", "seeds"), ("p4_4_training", "runs"), ("p4_6_training", "runs"),
                                  ("p4_7_training", "runs"))
            for run in json.loads((Path(repo_root) / "docs" / "data" / f"{name}.json").read_text(encoding="utf-8"))[key]]
    curves = [run["window_means"] for run in runs if not run["plateaued"]]
    falling = sum(all(a > b for a, b in zip(w[-3:], w[-2:])) for w in curves)
    rose = sum((not all(a > b for a, b in zip(w[-3:], w[-2:]))) and w[-1] > w[-2] for w in curves)
    return {"runs": len(runs), "not_plateaued": len(curves), "still_falling": falling,
            "rose_in_the_last_window": rose, "fluctuated": len(curves) - falling - rose}


def write_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Sorted keys, no rounding, no timestamp, a trailing newline; refused from a dirty tree and on a non-finite
    number -- both before the file is opened."""
    if payload.get("git", {}).get("dirty") is not False:
        raise ValueError("the artifact's git block is not clean (dirty must be false): a report is built from a clean "
                         "committed tree")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target = Path(path)
    target.write_text(text, encoding="utf-8")
    return target


def _exact(node: Any) -> Any:
    """A ``Fraction`` as ``{"exact": "n/d", "value": float}``; everything else unchanged (JSON carries no fraction)."""
    from fractions import Fraction

    if isinstance(node, Fraction):
        return {"exact": f"{node.numerator}/{node.denominator}", "value": float(node)}
    if isinstance(node, Mapping):
        return {str(key): _exact(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_exact(item) for item in node]
    return node


GRID_TIERS: tuple[str, ...] = ("mappo1000", "maxpressure", "fixedtime", "random")
#: A27(b)'s matched arm for each registered grid baseline.
A27B_MATCHED: Mapping[str, str] = {"bc": "bc_m", "bc_top10": "bc_top10_m", "bc_top10_perix": "bc_top10_perix_m",
                                   "iql": "iql_m"}
#: A27(b)'s list of P5.2 statements that contrast a DT with a baseline ("Q1's cells, Q2a/Q2b/Q2b-hard, Q3a/Q3b/Q3c,
#: Q5a/Q5b/Q5c"), with their per-tier and aggregate entries; the leaders are re-read beside (descriptive).
A27B_STATEMENT_IDS: tuple[str, ...] = (
    "Q1", "Q1b", "Q1b.both", "Q2a", "Q2b", "Q2b.aggregate", "Q2b_hard", "Q3a", "Q3b", "Q3b.aggregate", "Q3c", "Q5a",
    "Q5a.aggregate", "Q5b", "Q5b.aggregate", "Q5c", "Q5c.aggregate",
)
MOVED_RULE = ("A27(b): a verdict MOVES iff the registered rule's decision changes -- each statement is re-scored by P5.2c's "
              "own scorer (tier_statements.all_statements) on the registered book (P5.2b's random-tier IQL cell) with "
              "the new cell in place of its original, under both definitions; the original decision is beside it")


def _rescore_block(books: Mapping[str, Any], substitution: Mapping[tuple[str, str], Any], *,
                   counted: Sequence[str] | None) -> dict[str, Any]:
    """P5.2c's statements re-scored with *substitution*; refused if a statement that reads no replaced cell moves (the
    substitution may only reach the statements that read it)."""
    import offline.corrective_verdicts as cv
    import offline.tier_statements as tst

    moved = cv.moved_block(tst.all_statements(books), cv.rescore_p5_2(books, substitution))
    replaced = {books["corrected"][key].spec.cell_id for key in substitution}
    stray = [e for e in moved if not set(e["original_inputs"]) & replaced and (e["moved"] or e["reading_changed"])]
    if stray:
        raise ValueError(f"{stray[0]['id']} at {stray[0]['tier']} moved without reading a replaced cell")
    pool = [e for e in moved if counted is None or e["id"] in counted]
    verdicts = [e for e in pool if e["kind"] == "verdict"]
    return {"statements": pool, "rule": MOVED_RULE,
            "substitution": {f"{m}@{t}": {"original": books["corrected"][(m, t)].spec.cell_id,
                                          "replacement": cell.spec.cell_id} for (m, t), cell in substitution.items()},
            "n_statements": {d: sum(e["definition"] == d for e in verdicts) for d in ("att_engine", "att_ours")},
            "n_moved": {d: sum(e["moved"] for e in verdicts if e["definition"] == d) for d in ("att_engine", "att_ours")}}


def _leaders(books: Mapping[str, Any], substitution: Mapping[tuple[str, str], Any]) -> list[dict[str, Any]]:
    """P5.2c's descriptive leaders per tier and definition, on the registered book and on the substituted one."""
    import offline.tier_statements as tst

    book = {**books["corrected"], **substitution}
    before = {(b["tier"], b["definition"]): b for b in tst.leaders_block(books) if b["version"] in ("both", "corrected")}
    after = {(b["tier"], b["definition"]): b for b in tst.leaders_block({"corrected": book, "as_committed": book})}
    return [{"tier": tier, "definition": definition, "registered": False,
             "original_reading": before[(tier, definition)]["reading"], "reading": after[(tier, definition)]["reading"],
             "original_leaders": before[(tier, definition)]["leaders"], "leaders": after[(tier, definition)]["leaders"],
             "reading_changed": before[(tier, definition)]["reading"] != after[(tier, definition)]["reading"],
             "original_inputs": before[(tier, definition)]["inputs"], "inputs": after[(tier, definition)]["inputs"]}
            for tier, definition in sorted(before)]


def build_exposure_parity(src: Sources) -> dict[str, Any]:
    """``p9_1_exposure_parity.json`` (A27(b)): the exposure table from the constants, every A27(b) statement re-scored
    with the matched arms, the count that moves, the leaders beside, the sentence's two contrasts."""
    import offline.a27_arms as a27_arms
    import offline.corrective_verdicts as cv

    books = src.grid_books
    substitution = {(m, t): src.cells[f"p9_1:grid4x4:{A27B_MATCHED[m]}@{t}"] for m, t in cv.A27B_KEYS}
    dt = books["corrected"][("dt_nomix", "mappo1000")]
    body = {
        "exposure_table": _exact(a27_arms.exposure_table()),
        "rescore": _rescore_block(books, substitution, counted=A27B_STATEMENT_IDS),
        "counted_statement_ids": list(A27B_STATEMENT_IDS),
        "leaders": _leaders(books, substitution),
        "sentence_contrasts": {
            "matched": [cv.contrast_entry(dt, substitution[(m, "mappo1000")], tier="mappo1000") for m in ("bc", "iql")],
            "original": [cv.contrast_entry(dt, books["corrected"][(m, "mappo1000")], tier="mappo1000")
                         for m in ("bc", "iql")]},
        "confound": ("per-intersection return scaling, which parity does not remove: the DT's return prompt and scale are "
                     "set per intersection (sixteen per-node target_rtg / rtg_scale pairs, p5_2_declaration_*.json "
                     "prompts), IQL's reward scale is one number over all intersections; the spatial DT's joint context "
                     "is a second difference with no baseline counterpart (A27(b))"),
    }
    return _finish("A27(b)", src, body, role="A27(b): grid4x4 exposure parity, the matched baselines beside the originals",
                   what_this_is_not=(
                       "not a re-training of the DTs: they keep their registered checkpoints",
                       "not exposure parity on hz1x1: its only asymmetry is the top-decile arms', which runs the other "
                       "way and is not re-run here",
                       "not a removal of the return-scaling confound in any DT-versus-IQL contrast"))


#: The registered hz1x1 IQL cell of each tier (P4.4's mappo1000 cell, P4.6's four, P4.7's three mixtures).
HZ1X1_IQL_OLD: Mapping[str, str] = {
    "mappo1000": "p4_4:iql", "maxpressure": "p4_6:iql@maxpressure", "fixedtime": "p4_6:iql@fixedtime",
    "random": "p4_6:iql@random", "mappo500": "p4_6:iql@mappo500", "mix33": "p4_7:iql@mix33", "mix50": "p4_7:iql@mix50",
    "mix67": "p4_7:iql@mix67",
}

#: A27(c)'s eight hz1x1 tiers, in the registration's order.
HZ1X1_IQL_TIERS: tuple[str, ...] = ("random", "fixedtime", "maxpressure", "mappo500", "mappo1000", "mix33", "mix50",
                                    "mix67")


def build_iql_targets(src: Sources) -> dict[str, Any]:
    """``p9_1_iql_targets.json`` (A27(c)): ``iql_t - iql`` on twelve tiers, ``iql_m - iql_t`` on the grid (the two
    effects separable), P4.4's A6 statement, the IQL lead on hz1x1's random tier, P5.2's IQL statements, D4RL."""
    import offline.corrective_verdicts as cv

    books = src.grid_books
    hz = cv.paired_differences({t: (src.cells[f"p9_1:hz1x1:iql_t@{t}"], src.cells[HZ1X1_IQL_OLD[t]])
                                for t in HZ1X1_IQL_TIERS}, left="iql_t", right="iql")
    grid = cv.paired_differences({t: (src.cells[f"p9_1:grid4x4:iql_t@{t}"], books["corrected"][("iql", t)])
                                  for t in GRID_TIERS}, left="iql_t", right="iql")
    twelve = {}
    for definition in ("att_engine", "att_ours"):
        entries = [e["by_definition"][definition] for e in hz["tiers"] + grid["tiers"]]
        points = [e["mean_difference"] for e in entries]
        twelve[definition] = {"n_tiers": len(entries), "n_excluding_zero": sum(e["excludes_zero"] for e in entries),
                              "min": min(points), "max": max(points)}
    substitution = {("iql", t): src.cells[f"p9_1:grid4x4:iql_t@{t}"] for t in GRID_TIERS}
    rescore = _rescore_block(books, substitution, counted=None)
    new_ids = {cell.spec.cell_id for cell in substitution.values()}
    rescore["statements"] = [e for e in rescore["statements"] if set(e["inputs"]) & new_ids]
    rescore["n_statements"] = {d: sum(e["kind"] == "verdict" and e["definition"] == d for e in rescore["statements"])
                               for d in ("att_engine", "att_ours")}
    rescore["n_moved"] = {d: sum(e["moved"] for e in rescore["statements"] if e["definition"] == d)
                          for d in ("att_engine", "att_ours")}
    random_dt = src.cells["p4_6:dt@random"]
    body = {
        "hz1x1": hz, "grid4x4": grid, "twelve_tiers": twelve,
        "grid4x4_original_random_beside": cv.contrast_entry(src.cells["p9_1:grid4x4:iql_t@random"],
                                                            books["as_committed"][("iql", "random")], tier="random"),
        "grid4x4_exposure_against_dropout": cv.paired_differences(
            {t: (src.cells[f"p9_1:grid4x4:iql_m@{t}"], src.cells[f"p9_1:grid4x4:iql_t@{t}"]) for t in GRID_TIERS},
            left="iql_m", right="iql_t"),
        "p4_4_a6": cv.a6_rescore(src.cells["p4_4:madt"], src.cells["p4_4:iql"], src.cells["p9_1:hz1x1:iql_t@mappo1000"],
                                 delta=float(src.registered["p4_4_delta"])),
        "hz1x1_iql_lead_on_random": [cv.contrast_entry(src.cells["p4_6:iql@random"], random_dt, tier="random"),
                                     cv.contrast_entry(src.cells["p9_1:hz1x1:iql_t@random"], random_dt, tier="random")],
        "p5_2_rescore": rescore,
        "d4rl": src.d4rl,
    }
    return _finish("A27(c)", src, body, role="A27(c): IQL's three target forwards in eval(), beside the registered IQL",
                   what_this_is_not=(
                       "not a change to IQL's hyper-parameters: tau, beta, gamma, Polyak, clip, batch, steps, warm-up "
                       "and seeds are the registered ones; only the three target forwards' module mode differs",
                       "not the published IQL update order: the advantage still uses V from before its update",
                       "P4.4's A6 statement is the only registered P4.4 statement in which IQL appears (madt_vs_iql); "
                       "the registration's wording 'BC-vs-IQL contrast' names no P4.4 contrast"))


HZ1X1_FRESH_ARMS: tuple[str, ...] = ("dt", "bc", "bc_top10", "iql", "iql_t")
GRID_FRESH_ARMS: tuple[str, ...] = ("dt_nomix", "bc_m", "bc_top10_m", "bc_top10_perix_m", "iql_m")


def build_random_fresh(src: Sources) -> dict[str, Any]:
    """``p9_1_random_fresh.json`` (A27(d)): the declarations, every collapse check against both random references on
    both scenarios (the new tier and the original one), Q6 against both anchors, P3a with its beside-table, P3b."""
    import offline.corrective_verdicts as cv
    import offline.tier_statements as tst
    import offline.tier_sweep as ts

    books = src.grid_books
    hz_refs = {"registered_five_schedule": src.cells["p4_6:behaviour@random"],
               "random_fresh": src.cells["p9_1:hz1x1:behaviour@random_fresh"]}
    grid_refs = {"registered_five_schedule": books["corrected"][tst.ANCHOR],
                 "random_fresh": src.cells["p9_1:grid4x4:behaviour@random_fresh"]}
    collapse = {
        "hz1x1": {
            "random_fresh_tier": cv.collapse_rows({a: src.cells[f"p9_1:hz1x1:{a}@random_fresh"] for a in HZ1X1_FRESH_ARMS},
                                                  hz_refs, tier="random_fresh"),
            "original_random_tier": cv.collapse_rows({a: src.cells[f"p4_6:{a}@random"] for a in ("dt", "bc", "bc_top10",
                                                                                                "iql")},
                                                     hz_refs, tier="random")},
        "grid4x4": {
            "random_fresh_tier": cv.collapse_rows({a: src.cells[f"p9_1:grid4x4:{a}@random_fresh"]
                                                   for a in GRID_FRESH_ARMS}, grid_refs, tier="random_fresh"),
            "original_random_tier": cv.collapse_rows({m: books["corrected"][(m, "random")] for m in ts.METHODS},
                                                     grid_refs, tier="random")},
    }
    fresh_book = {**books["corrected"], tst.ANCHOR: grid_refs["random_fresh"]}
    q6 = {"registered_five_schedule": [e for d in ("att_engine", "att_ours")
                                       for e in tst.score_q6(books["corrected"], definition=d)],
          "random_fresh": [e for d in ("att_engine", "att_ours") for e in tst.score_q6(fresh_book, definition=d)]}
    hz_dt, grid_dt = "p9_1:hz1x1:dt@random_fresh", "p9_1:grid4x4:dt_nomix@random_fresh"
    p3a = {"hz1x1": {**cv.p3a(src.p3a[hz_dt], per_intersection=False), "cell": hz_dt},
           "grid4x4": {**cv.p3a(src.p3a[grid_dt], per_intersection=True), "cell": grid_dt}}
    beside = {
        "hz1x1": cv.p3a_beside(src.p3a[hz_dt], training=src.schedules["hz1x1"], references={
            "registered_five_schedule": src.p3a["p4_6:behaviour@random"],
            "random_fresh": src.p3a["p9_1:hz1x1:behaviour@random_fresh"]}),
        "grid4x4": cv.p3a_beside(src.p3a[grid_dt], training=src.schedules["grid4x4"], per_intersection=True, references={
            "registered_five_schedule": src.p3a[tst.ANCHOR_CELL],
            "random_fresh": src.p3a["p9_1:grid4x4:behaviour@random_fresh"]}),
    }
    p3b = [
        {**cv.contrast_entry(src.cells["p9_1:hz1x1:iql@random_fresh"], src.cells[hz_dt], tier="random_fresh"),
         "scenario": "hz1x1"},
        {**cv.contrast_entry(src.cells["p9_1:hz1x1:iql_t@random_fresh"], src.cells[hz_dt], tier="random_fresh"),
         "scenario": "hz1x1"},
        {**cv.contrast_entry(src.cells["p4_6:iql@random"], src.cells["p4_6:dt@random"], tier="random"),
         "scenario": "hz1x1"},
        {**cv.contrast_entry(src.cells["p9_1:grid4x4:iql_m@random_fresh"], src.cells[grid_dt], tier="random_fresh"),
         "scenario": "grid4x4"},
        {**cv.contrast_entry(books["corrected"][("iql", "random")], books["corrected"][("dt_nomix", "random")],
                             tier="random"), "scenario": "grid4x4"},
    ]
    body = {
        "declarations": {"hz1x1": src.declarations["random_fresh_hz1x1"],
                         "grid4x4": src.declarations["random_fresh_grid4x4"]},
        "collapse": collapse, "q6_both_anchors": q6, "p3a": p3a, "p3a_beside": beside,
        "p3b": {"contrasts": p3b, "prediction": None,
                "note": "A27(d)'s P3b: no directional expectation for the IQL lead on random; re-measured beside the "
                        "original tier's (on the grid IQL also has matched exposure and dropout-free targets)"},
        "normalisation_basis": "the new tiers fit the state normalisation, the return target and scale and IQL's "
                               "reward scale on their own 200 training episodes; the original random tier fitted them "
                               "on its 400-episode split, so the before/after comparison changes the generator AND the "
                               "normalisation basis (A27(d))",
    }
    return _finish("A27(d)", src, body, role="A27(d): a true random tier (200 distinct schedules) on both scenarios",
                   what_this_is_not=(
                       "not a replacement of the original random tiers: they stay the record, their defect named",
                       "not a test of the mixtures' selection question (A27(e) replaces them for it)",
                       "the beside-table carries no prediction; P3a alone is predicted"))
