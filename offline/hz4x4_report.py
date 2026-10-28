"""P9.3 Phase B's report (A27 point 8): hz4x4's ladder, beside the grid's -- ``old-cells`` and ``report``.

Format versions: ``p9.3-old-cells/1.0`` (``docs/data/p9_3_old_cells.json``) and ``p9.3-hz4x4/1.0``
(``docs/data/p9_3_hz4x4.json``).  Alignment convention: none of its own -- every value is read from a record another
module wrote (``offline.hz4x4_phase_b``'s evaluation and training records; ``offline.tier_statements.load_inputs`` for the
grid's committed cells), and every row carries ``att_ours`` and ``att_engine`` of ONE held-out episode (seed, draw).

WHAT THIS IS -- AND WHAT IT IS NOT
----------------------------------
``PREREGISTRATION`` A27 (i), label 8: a conditional generalizability extension registered after the grid results.  hz4x4
has NO registered statement (A27 (i) registers none) and NO primary definition (A11's gate failed there by construction,
criterion 4).  Its leaders are the grid's DESCRIPTIVE rule (``BRIEF_48`` Amendment G, G.2, A-B2), under both definitions,
``registered: false``, ``definition_dependent`` flagged; the references' levels stand beside the leaders and never compete
for the lead.  The grid's corresponding cells come by reference and per-episode copy from their committed sources, with
path and digest, and nothing is re-rolled on the grid (Amendment A, A.5 Q4).

WHY A NEW MODULE (``BRIEF_48`` Amendment H, H.2-4)
--------------------------------------------------
``offline/hz4x4_phase_b.py`` and ``offline/hz4x4_ladder.py`` are frozen at the campaign's commit, so that the code which
wrote the records is provably the code the branch carries; the report reads their records and imports their constants,
and changes neither.

WHY A MIRROR OF THE LEADERS RULE
--------------------------------
``tier_statements.leaders_block`` loops over the grid's tiers, methods, versions and primary definition -- module
constants.  :func:`hz4x4_leaders_block` is its body with those four made parameters; every function it calls
(``level_groups``, ``_level``, ``contrast``, ``_seed_levels``) is ``tier_statements``' own, unchanged.  A test holds it to
the original on the grid's committed books and on a synthetic grid book.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import offline.hz4x4_ladder as hz
import offline.hz4x4_phase_b as pb
import offline.tier_statements as st

ARTIFACT_FORMAT_REPORT = "p9.3-hz4x4/1.0"
ARTIFACT_FORMAT_OLD_CELLS = "p9.3-old-cells/1.0"

#: The source tag of hz4x4's cells (``CellSpec.source``; cell ids are ``p9_3:<method>@<tier>``).
SOURCE = "p9_3"

GATE_ARTIFACT = Path("docs/data/p9_3_engine_att_gate.json")
DECLARATION_ARTIFACT = Path("docs/data/p9_3_declaration.json")

#: The five references (``hz4x4_phase_b.REFERENCE_CELLS``) as arm labels.
REFERENCE_ARMS: tuple[str, ...] = tuple(f"{method}@{tier}" for method, tier in pb.REFERENCE_CELLS)

#: hz4x4 cell -> the grid's committed counterpart (Q4).  ``random@mappo1000`` is P5.2c's shared anchor
#: (``tier_statements.ANCHOR_CELL``, which its books file under ``tier_statements.ANCHOR``).
GRID_COUNTERPARTS: Mapping[tuple[str, str], str] = {
    ("dt_nomix", "mappo1000"): "p5_1:dt_nomix@grid4x4_mappo1000",
    ("dt_nomix", "maxpressure"): "p5_2:dt_nomix@maxpressure",
    ("dt_nomix", "fixedtime"): "p5_2:dt_nomix@fixedtime",
    ("behaviour", "mappo1000"): "p5_1:behaviour@grid4x4_mappo1000",
    ("behaviour", "maxpressure"): "p5_2:behaviour@maxpressure",
    ("behaviour", "fixedtime"): "p5_2:behaviour@fixedtime",
    ("random", "mappo1000"): st.ANCHOR_CELL,
}

#: Every hz4x4 cell whose grid counterpart is P9.1's (A27 (b), (d)) -- the matched arms on every tier, ``dt_nomix`` and
#: ``behaviour`` on ``random_fresh`` -- and so is not committed until P9.1's campaign is.
P9_1_COUNTERPARTS: tuple[tuple[str, str], ...] = tuple(
    (cell.method, cell.tier) for cell in pb.evaluation_cells() if (cell.method, cell.tier) not in GRID_COUNTERPARTS
)

#: What the artifact is not, in the order the paper's limitations section reads them.
WHAT_THIS_IS_NOT: tuple[str, ...] = (
    "not a new topology: the grid's 4x4 lattice with real demand and real road geometry (A27 (i))",
    "not a new claim: a replication of the grid's ladder, reported beside it (A27 (i))",
    "not a C3 target: hz4x4 is audited and is not one (A15)",
    "not registered: A27 (i) registers no hz4x4 statement, so its leaders are descriptive (BRIEF_48 Amendment G, G.2)",
    "not single-definition: A11's gate failed on hz4x4 (criterion 4), so no claim rests on att_ours or att_engine alone",
    "not a timing claim: every training second is an uncontrolled clock (A27 (a))",
)


def load_cells(
    output_root: str | Path, *, seeds: Sequence[int] = hz.SEEDS, draws: Sequence[int] | None = None
) -> dict[str, Any]:
    """Every evaluation cell as a ``tier_statements.Cell``, its rows read from its records in (seed, draw) order --
    refused unless every planned record exists and says it reproduced, and no refusal is on disk.

    A row is P5.2c's: ``arm``, ``seed``, ``draw_id``, ``att_ours`` and ``att_engine`` with A11 (b)'s companions from
    route (ii), and ``horizon_vehicle_count`` / ``episode_reward`` from route (i) (the evaluation file's, as on the
    grid).  Route (ii)'s ``att_ours`` is held to route (i)'s ``att_horizon`` under ``==`` once more on reading.
    """
    import json

    from offline.dt_gate import HELD_OUT_DRAWS

    pool = sorted(int(d) for d in (HELD_OUT_DRAWS if draws is None else draws))
    refused_dir = Path(output_root) / pb.PHASE_B_DIR / "eval_refused"
    refusals = sorted(refused_dir.glob("*.json")) if refused_dir.is_dir() else []
    if refusals:
        raise ValueError(f"{len(refusals)} refused evaluation record(s) under {refused_dir}, first {refusals[0].name}: "
                         "the evaluation stopped there, and a refused episode never enters a report")
    cells: dict[str, Any] = {}
    for cell in pb.evaluation_cells():
        rows = []
        for seed in sorted(int(s) for s in seeds):
            for draw in pool:
                path = pb.eval_record_path(output_root, cell, seed, draw)
                if not path.is_file():
                    raise FileNotFoundError(f"{path} is missing: {cell.arm} is incomplete, and the report reads complete "
                                            "cells only")
                record = json.loads(path.read_bytes())
                found = (record.get("arm"), record.get("method"), record.get("tier"), record.get("seed"),
                         record.get("draw_id"))
                if found != (cell.arm, cell.method, cell.tier, seed, draw):
                    raise ValueError(f"{path} is filed under {cell.arm} seed {seed} draw {draw} but holds {found}")
                if record.get("reproduces") is not True or float(record["att_ours"]) != float(
                        record["route_i"]["att_horizon"]):
                    raise ValueError(f"{path}: route (ii)'s att_ours {record['att_ours']!r} is not route (i)'s "
                                     f"att_horizon {record['route_i']['att_horizon']!r} under =="
                                     " -- the record may not enter a report")
                rows.append({
                    "arm": cell.arm, "seed": seed, "draw_id": draw, "att_ours": float(record["att_ours"]),
                    "att_engine": float(record["att_engine"]), "created": int(record["created"]),
                    "entered": int(record["entered"]), "never_entered": int(record["never_entered"]),
                    "entered_fraction": float(record["entered_fraction"]),
                    "horizon_vehicle_count": float(record["route_i"]["horizon_vehicle_count"]),
                    "episode_reward": float(record["route_i"]["episode_reward"]),
                })
        spec = st.CellSpec(cell_id=f"{SOURCE}:{cell.arm}", method=cell.method, tier=cell.tier, source=SOURCE,
                           arm=cell.arm, eval_file=f"{pb.PHASE_B_DIR}/eval/{cell.arm}__seed*__draw*.json",
                           manifest="none: each record is written once (hz4x4_phase_b.write_eval_record)")
        cells[spec.cell_id] = st.Cell(spec=spec, rows=tuple(rows))
    return cells


def book_of(cells: Mapping[str, Any]) -> dict[tuple[str, str], Any]:
    """``(method, tier)`` -> cell, for the leaders rule."""
    return {(cell.spec.method, cell.spec.tier): cell for cell in cells.values()}


def hz4x4_leaders_block(
    books: Mapping[str, Mapping[tuple[str, str], Any]],
    *,
    tiers: Sequence[str],
    methods: Sequence[str],
    primary_definition: str | None,
    versions_of: Callable[[str], Sequence[str]],
) -> list[dict[str, Any]]:
    """``tier_statements.leaders_block``'s body with its four grid constants made parameters, and nothing else.

    The original loops over the grid's ``TIERS``, ``ts.METHODS`` and ``_versions_of`` and marks ``PRIMARY_DEFINITION``.
    Here those are *tiers*, *methods*, *versions_of* and *primary_definition* (``None`` on hz4x4: no definition is
    primary there).  Every function it calls -- ``_level``, ``level_groups``, ``contrast``, ``_seed_levels`` -- and
    ``DEFINITIONS`` are ``tier_statements``' own, reached at call time.  Descriptive, not registered: the blocks say so.
    A test holds it to the original, under ``==`` on the serialised blocks, on the grid's committed books and on a
    synthetic grid book.
    """
    blocks = []
    for tier in tiers:
        for definition in st.DEFINITIONS:
            for version in versions_of(tier):
                book = books["corrected" if version == "both" else version]
                levels = {method: st._level(book[(method, tier)], definition) for method in methods}
                groups = st.level_groups(levels)
                leaders, runners = groups[0], (groups[1] if len(groups) > 1 else [])
                contrasts = []
                if len(leaders) == 1:
                    contrasts = [st.contrast(book[(leaders[0], tier)], book[(runner, tier)], definition=definition,
                                          tier=tier) for runner in runners]
                    resolved = [block["ci95_high"] < 0.0 for block in contrasts]
                    lead = ("the lead resolves (the CI lies entirely below zero)" if all(resolved) else
                            "the lead does not resolve (the CI includes zero)" if not any(resolved) else
                            "the lead resolves over some of the tied runners-up only")
                    reading = f"{leaders[0]} leads {' = '.join(runners)}; {lead}"
                else:
                    reading = f"{' = '.join(leaders)} tie exactly for the lead"
                seed_levels = st._seed_levels(book, tier, definition, methods)
                firsts = {seed: st.level_groups(values)[0] for seed, values in seed_levels.items()}
                blocks.append({
                    "tier": tier, "definition": definition, "primary": definition == primary_definition,
                    "version": version, "registered": False, "order": groups, "levels": levels, "leaders": leaders,
                    "runners_up": runners, "leader_minus_runner_up": contrasts, "reading": reading,
                    "per_seed_first_places": firsts, "per_seed_levels": seed_levels,
                    "pooled_leaders_lose_first_place_on_seeds": [seed for seed, first in firsts.items()
                                                                 if first != leaders],
                    "inputs": [book[(method, tier)].spec.cell_id for method in methods],
                })
    for block in blocks:
        other = next(b for b in blocks if b["tier"] == block["tier"] and b["version"] == block["version"]
                     and b["definition"] != block["definition"])
        block["definition_dependent"] = block["reading"] != other["reading"]
        block["definition_dependent_on"] = "reading"
        block["per_seed_first_places_definition_dependent"] = block["per_seed_first_places"] != other[
            "per_seed_first_places"]
    return blocks


def leaders(cells: Mapping[str, Any]) -> list[dict[str, Any]]:
    """hz4x4's leaders: the mirror over the five arms on each tier, no primary, the references' levels beside.

    One book and one version (``both``): hz4x4 has no corrected cell.  ``primary`` is ``False`` under both definitions
    -- A11's gate failed on hz4x4 -- and each block carries the five references' levels (each cell's level by
    ``tier_statements._level``), which are never in its order, leaders or runners-up.
    """
    book = book_of(cells)
    blocks = hz4x4_leaders_block({"corrected": book}, tiers=hz.TIER_ORDER, methods=pb.ARMS, primary_definition=None,
                                 versions_of=lambda tier: ("both",))
    for block in blocks:
        block["references"] = {arm: st._level(book[tuple(arm.split("@"))], block["definition"]) for arm in REFERENCE_ARMS}
        block["references_rule"] = (
            "the five references' levels -- dt_gate.mean_ci95 over each cell's episodes -- beside the leaders, never "
            "competing for the lead (BRIEF_48 Amendment G, G.2, A-B2)")
        block["primary_note"] = ("no definition is primary on hz4x4: A11's gate failed there by construction (criterion "
                                 "4; docs/data/p9_3_engine_att_gate.json)")
    return blocks


def cell_summaries(cells: Mapping[str, Any]) -> dict[str, Any]:
    """``tier_statements.cells_block``'s per-cell computation, with hz4x4's sources.

    n; under each definition ``iql_correction.cell_summary`` (mean, sd, CI, per-seed means) with the CI's width, the
    median and the IQR by ``numpy.percentile(method='linear')``; A11 (b)'s companions by
    ``iql_correction.admission_summary``; A5's vehicle count -- every quantity the grid's block carries, by the same
    calls.  A test holds every field but ``sources`` to ``tier_statements.cells_block`` on the same cells.
    """
    import numpy as np

    from offline import iql_correction as ic
    from offline.dt_gate import mean_ci95

    out: dict[str, Any] = {}
    for cell_id, cell in cells.items():
        spec = cell.spec
        entry: dict[str, Any] = {"method": spec.method, "tier": spec.tier, "source": spec.source, "arm": spec.arm,
                                 "n": len(cell.rows)}
        for definition in st.DEFINITIONS:
            episodes = st.episodes_of(cell, definition)
            summary = ic.cell_summary(episodes)
            values = np.asarray([episode.att_horizon for episode in episodes], dtype=np.float64)
            q25, q50, q75 = (float(x) for x in np.percentile(values, [25, 50, 75], method="linear"))
            entry[definition] = {**summary, "ci95_width": float(summary["ci95_high"] - summary["ci95_low"]),
                                 "median": q50, "iqr": [q25, q75], "iqr_width": q75 - q25}
        entry["companions"] = ic.admission_summary(cell.rows)
        entry["horizon_vehicle_count_mean"] = float(mean_ci95([float(r["horizon_vehicle_count"])
                                                               for r in cell.rows]).mean)
        files = f"output/{pb.PHASE_B_DIR}/eval/{spec.arm}__seed*__draw*.json"
        entry["sources"] = {"att_ours": files, "horizon_vehicle_count": files, "att_engine": files, "companions": files}
        entry["quantiles"] = "numpy.percentile(method='linear') over the cell's episodes"
        out[cell_id] = entry
    return out


def training_block(output_root: str | Path) -> list[dict[str, Any]]:
    """The 100 training records, each checkpoint's sha256 verified against its record.

    Refused unless every cell of ``hz4x4_phase_b.training_cells`` has its record and its checkpoint, the record is filed
    under its cell, the checkpoint's sha256 is the record's, every cell trained the registered 40,000 steps, and its
    seconds carry the uncontrolled-clock label -- so a smoke's or a timing's two-step cells can never enter a report.
    """
    import json

    import offline.tier_sweep as ts

    rows: list[dict[str, Any]] = []
    for tier, arm, seed in pb.training_cells():
        record_path = pb.train_record_path(output_root, tier, arm, seed)
        checkpoint = pb.checkpoint_path(output_root, tier, arm, seed)
        if not record_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"{tier}/{arm} seed {seed}: the training record {record_path} or its checkpoint "
                                    f"{checkpoint} is missing; the report reads the complete campaign")
        record = json.loads(record_path.read_bytes())
        if (record.get("tier"), record.get("arm"), record.get("seed")) != (tier, arm, seed):
            raise ValueError(f"{record_path} is filed under {tier}/{arm} seed {seed} but holds "
                             f"{record.get('tier')}/{record.get('arm')} seed {record.get('seed')}")
        if hz._sha256_file(checkpoint) != record.get("file_sha256"):
            raise ValueError(f"{checkpoint} is not the checkpoint its record {record_path} names")
        steps = int(ts.DECLARED_GRADIENT_STEPS)
        if int(record["gradient_steps"]) != steps or int(record["declared_gradient_steps"]) != steps:
            raise ValueError(f"{record_path}: {record['gradient_steps']} of {record['declared_gradient_steps']} "
                             f"gradient steps; every campaign cell trains the registered {steps}")
        if record.get("clock") != pb.CLOCK_LABEL:
            raise ValueError(f"{record_path} does not carry the uncontrolled-clock label (A27 (a))")
        rows.append(record)
    return rows


def old_cells(grid_cells: Mapping[str, Any], *, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The grid's corresponding cells, copied per episode with source path and digest; P9.1's named as pending.

    *grid_cells* and *provenance* are ``tier_statements.load_inputs``' -- P5.2c's reader, which reads every evaluation
    file at its manifest's digest and every ``att_engine`` row from P8.4b's campaign at its anchor.  Each copied cell
    carries its evaluation file, that file's digest and anchor as the reader recorded them, and its 500 rows; the
    ``att_engine`` rows' anchor is carried once.  Nothing is re-rolled (``BRIEF_48`` Amendment A, A.5 Q4).
    """
    digests = {str(entry["file"]): entry for entry in provenance["inputs"]}
    cells: dict[str, Any] = {}
    for (method, tier), cell_id in GRID_COUNTERPARTS.items():
        cell = grid_cells[cell_id]
        file = f"output/{cell.spec.eval_file}"
        if file not in digests:
            raise ValueError(f"{cell_id}: no digest for {file} in the reader's provenance; a copied cell names its source "
                             "and the digest it was read at")
        cells[f"{method}@{tier}"] = {
            "counterpart": cell_id, "source": cell.spec.source, "grid_arm": cell.spec.arm, "eval_file": file,
            "eval_file_sha256": digests[file]["sha256"], "eval_file_anchor": digests[file]["anchor"],
            "n": len(cell.rows), "rows": [dict(row) for row in cell.rows],
        }
    return {
        "format_version": ARTIFACT_FORMAT_OLD_CELLS,
        "registered_in": "PREREGISTRATION A27 (a) (the old cells a new cell is compared with); BRIEF_48 Amendment A, A.5 Q4",
        "rule": ("the grid's corresponding cells, by reference and per-episode copy from their committed sources with "
                 "path and digest; nothing re-rolled on the grid"),
        "reader": ("offline.tier_statements.load_inputs (P5.2c): every evaluation file at its manifest digest, every "
                   "att_engine row from P8.4b's campaign at its anchor"),
        "att_engine_rows": {"p8_4b_campaign": provenance["p8_4b_campaign"], "p8_4b_rows": provenance["p8_4b_rows"]},
        "cells": cells,
        "pending": {f"{method}@{tier}": ("P9.1's committed artifact (BRIEF_46; A27 (b), (d)), not committed when this "
                                         "was built")
                    for method, tier in P9_1_COUNTERPARTS},
    }


def report_artifact(
    *,
    cells: Mapping[str, Any],
    training: Sequence[Mapping[str, Any]],
    old: Mapping[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """``docs/data/p9_3_hz4x4.json``'s payload: A27's label and sentence, what it is not, the gate and the declaration
    by digest, the exposure table, the leaders, every cell's summary and per-episode rows, every training row under the
    clock label, and the old cells by digest."""
    import hashlib
    import json

    root = Path(repo_root) if repo_root is not None else hz._repo_root()
    gate_bytes = (root / GATE_ARTIFACT).read_bytes()
    declaration_bytes = (root / DECLARATION_ARTIFACT).read_bytes()
    declared = json.loads(declaration_bytes)
    return {
        "format_version": ARTIFACT_FORMAT_REPORT,
        "registered_in": "PREREGISTRATION A27 (i), with (a)-(d); BRIEF_48 section 4, Amendments G.2 and H",
        "label": hz.A27_LABEL,
        "paper_sentence": hz.PAPER_SENTENCE,
        "what_this_is_not": list(WHAT_THIS_IS_NOT),
        "primary_definition": None,
        "primary_note": ("A11's gate failed on hz4x4 (criterion 4: four behaviour tiers against the registered seven), so "
                         "every ordering is reported under both definitions and as DEFINITION-DEPENDENT where they differ"),
        "gate": {"path": str(GATE_ARTIFACT), "sha256": hashlib.sha256(gate_bytes).hexdigest(),
                 "outcome": json.loads(gate_bytes)["outcome"]},
        "declaration": {"path": str(DECLARATION_ARTIFACT), "sha256": hashlib.sha256(declaration_bytes).hexdigest()},
        "exposure": {tier: declared["tiers"][tier]["exposure"] for tier in hz.TIER_ORDER},
        "leaders": leaders(cells),
        "cells": cell_summaries(cells),
        "episodes": {cell_id: [dict(row) for row in cell.rows] for cell_id, cell in cells.items()},
        "training": {"clock": pb.CLOCK_LABEL, "rows": list(training)},
        "old_cells": {"format_version": old["format_version"],
                      "sha256": hashlib.sha256(json.dumps(old, sort_keys=True).encode("utf-8")).hexdigest(),
                      "counterparts": {arm: entry["counterpart"] for arm, entry in old["cells"].items()},
                      "pending": list(old["pending"])},
    }


def build_parser() -> argparse.ArgumentParser:
    """``old-cells`` and ``report``.  Run from the main tree with this branch's code on ``PYTHONPATH``."""
    parser = argparse.ArgumentParser(prog="python -P -m offline.hz4x4_report",
                                     description="P9.3 Phase B's report: hz4x4's ladder beside the grid's.")
    sub = parser.add_subparsers(dest="command", required=True)
    old = sub.add_parser("old-cells", help="the grid's corresponding cells, copied with path and digest (Q4)")
    old.add_argument("--out", required=True, help="docs/data/p9_3_old_cells.json")
    old.add_argument("--output-root", default=str(hz.MAIN_OUTPUT_ROOT))
    report = sub.add_parser("report", help="docs/data/p9_3_hz4x4.json from the campaign's records")
    report.add_argument("--out", required=True, help="docs/data/p9_3_hz4x4.json")
    report.add_argument("--old-cells", required=True, help="the committed old-cells artifact")
    report.add_argument("--output-root", default=str(hz.MAIN_OUTPUT_ROOT))
    return parser


def _run_old_cells(args: argparse.Namespace) -> int:
    out = hz._refuse_existing(args.out)
    inputs = st.load_inputs(st.Roots(repo_root=hz._repo_root(), output_root=Path(args.output_root)))
    payload = {**old_cells(inputs.cells, provenance=inputs.provenance), "git": hz._git()}
    hz._write_artifact(payload, out)
    print(f"wrote {out}: {len(payload['cells'])} copied cell(s), {len(payload['pending'])} pending", flush=True)
    return 0


def _run_report(args: argparse.Namespace) -> int:
    import json

    out = hz._refuse_existing(args.out)
    old = json.loads(Path(args.old_cells).read_bytes())
    if old.get("format_version") != ARTIFACT_FORMAT_OLD_CELLS:
        raise ValueError(f"{args.old_cells} is not an old-cells artifact ({old.get('format_version')!r})")
    output_root = hz.assert_main_tree_path(args.output_root, what="the output root")
    cells = load_cells(output_root)
    training = training_block(output_root)
    commits = sorted({(row["git"]["commit"], bool(row["git"]["dirty"])) for row in training})
    payload = {**report_artifact(cells=cells, training=training, old=old),
               "campaign_code": {"training_records_git": [{"commit": c, "dirty": d} for c, d in commits],
                                 "note": "the code that trained every cell, as each training record states it"},
               "git": hz._git()}
    hz._write_artifact(payload, out)
    print(f"wrote {out}: {len(cells)} cells, {len(training)} trainings, {len(payload['leaders'])} leaders blocks",
          flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(None if argv is None else list(argv))
    if args.command == "old-cells":
        return _run_old_cells(args)
    if args.command == "report":
        return _run_report(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover - the CLI is exercised through main() by the tests
    raise SystemExit(main())
