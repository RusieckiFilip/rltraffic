"""Tests for P9.3 Phase B's report (``BRIEF_48`` E.3.6, Amendments G.2 and H.2-4; ``docs/plans/p9.3.md`` section 10.3)
-- written before the code they test.

* the leaders rule is ``tier_statements.leaders_block``'s body with its grid constants made parameters: held to the
  original on a synthetic grid book that exercises a resolving lead, an exact tie and a definition-dependent reading, and
  on the grid's committed books where the main tree's ``output/`` exists;
* hz4x4's leaders: the five arms on each tier under both definitions, no primary, ``registered: false``, the references'
  levels beside the leaders and never among them, ``definition_dependent`` flagged (A-B2, ruled in G.2);
* the cells are read from their records in (seed, draw) order, refused unless complete, reproduced and free of refusals;
* a cell's summary is ``tier_statements.cells_block``'s per-cell computation, field for field, with hz4x4's sources;
* the training block carries every one of the 100 records under the uncontrolled-clock label, each checkpoint verified;
* the old cells are the grid's counterparts copied per episode with path and digest; P9.1's are named as pending (Q4);
* the report cites the gate and the declaration by digest, carries the exposure table and says what it is not.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import offline.hz4x4_ladder as hz
import offline.hz4x4_phase_b as pb
import offline.hz4x4_report as rp
import offline.tier_statements as st
import offline.tier_sweep as ts
from offline.dt_gate import mean_ci95

REPO = Path(__file__).resolve().parents[1]
SEEDS = (101, 202, 303, 404, 505)
DRAWS = (1000, 1001, 1002)

needs_grid_output = pytest.mark.skipif(
    not (hz.MAIN_OUTPUT_ROOT / "SHA256SUMS_p5_2.txt").is_file(),
    reason="the grid's committed evaluation output (the main tree's output/SHA256SUMS_p5_* and their files) is not here",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jiggle(level: float, seed: int, draw: int) -> float:
    """A deterministic per-episode variation, so every CI is non-degenerate and identical cells stay identical; the
    small seed and draw terms make every value of a cell distinct, so a quantile's interpolation is never hidden by a
    tie (mutant S2: ``method="lower"`` for ``"linear"`` survived the tied values)."""
    return float(level + (((seed // 101) * 7 + (draw - 1000) * 3) % 5 - 2) / 2.0 + seed * 1e-3 + (draw - 1000) * 1e-5)


def _row(arm: str, seed: int, draw: int, ours: float, engine: float, *, never: int = 0) -> dict[str, Any]:
    created = 2700
    return {"arm": arm, "seed": seed, "draw_id": draw, "att_ours": float(ours), "att_engine": float(engine),
            "created": created, "entered": created - never, "never_entered": never,
            "entered_fraction": (created - never) / created, "horizon_vehicle_count": float(10 + seed % 7),
            "episode_reward": float(-ours)}


def _cell(cell_id: str, method: str, tier: str, source: str, arm: str, rows: list[dict[str, Any]],
          eval_file: str | None = None) -> Any:
    spec = st.CellSpec(cell_id=cell_id, method=method, tier=tier, source=source, arm=arm,
                       eval_file=eval_file or f"{source}/eval_{tier}_{method}.json", manifest="SHA256SUMS_test.txt")
    return st.Cell(spec=spec, rows=tuple(sorted(rows, key=lambda r: (r["seed"], r["draw_id"]))))


def _levelled(cell_id: str, method: str, tier: str, source: str, arm: str, ours: float, engine: float,
              eval_file: str | None = None) -> Any:
    rows = [_row(arm, s, d, _jiggle(ours, s, d), _jiggle(engine, s, d), never=(s + d) % 3) for s in SEEDS for d in DRAWS]
    return _cell(cell_id, method, tier, source, arm, rows, eval_file)


# ======================================================================
# The leaders rule: a mirror of tier_statements.leaders_block
# ======================================================================


def _grid_books() -> dict[str, dict[tuple[str, str], Any]]:
    """A synthetic grid book: a resolving unique lead on most tiers, an exact tie for the lead at fixedtime (identical
    cells), a lead that changes with the definition at maxpressure, and the random tier's two IQL versions."""
    book: dict[tuple[str, str], Any] = {}
    for t_index, tier in enumerate(st.TIERS):
        for m_index, method in enumerate(ts.METHODS):
            ours = 200.0 + 10.0 * m_index + t_index
            engine = ours - 3.0 if tier != "maxpressure" else 200.0 + 10.0 * (len(ts.METHODS) - 1 - m_index)
            book[(method, tier)] = _levelled(f"syn:{method}@{tier}", method, tier, "syn", f"{method}@{tier}", ours,
                                             engine)
    leader = book[("dt_spatial", "mappo1000")]
    offsets = {1000: -1.0, 1001: 0.8, 1002: -0.1}  # a mean lead of 0.1 whose per-draw differences change sign
    close = [dict(r, arm="dt_nomix@mappo1000", att_ours=r["att_ours"] + offsets[r["draw_id"]],
                  att_engine=r["att_engine"] + offsets[r["draw_id"]]) for r in leader.rows]
    book[("dt_nomix", "mappo1000")] = _cell("syn:dt_nomix@mappo1000", "dt_nomix", "mappo1000", "syn",
                                            "dt_nomix@mappo1000", close)
    tied = book[("dt_spatial", "fixedtime")]
    book[("dt_nomix", "fixedtime")] = _cell("syn:dt_nomix@fixedtime", "dt_nomix", "fixedtime", "syn",
                                            "dt_nomix@fixedtime", [dict(r, arm="dt_nomix@fixedtime") for r in tied.rows])
    original = dict(book)
    original[("iql", "random")] = _levelled("syn:iql@random#original", "iql", "random", "syn", "iql@random", 190.0,
                                            185.0)
    return {"corrected": book, "as_committed": original}


def _grid_parameters() -> dict[str, Any]:
    return dict(tiers=st.TIERS, methods=ts.METHODS, primary_definition=st.PRIMARY_DEFINITION,
                versions_of=st._versions_of)


def test_the_leaders_mirror_is_tier_statements_leaders_block_on_a_synthetic_grid_book() -> None:
    books = _grid_books()
    theirs = st.leaders_block(books)
    mine = rp.hz4x4_leaders_block(books, **_grid_parameters())
    assert json.dumps(mine, sort_keys=True) == json.dumps(theirs, sort_keys=True)
    assert len(theirs) == 10
    readings = [block["reading"] for block in theirs]
    assert any("tie exactly for the lead" in reading for reading in readings)
    assert any("the lead resolves" in reading for reading in readings)
    # mutant L4 (the CI's low end read in place of its high end) survived a book whose every lead resolved
    assert any("the lead does not resolve" in reading for reading in readings)
    assert any(block["definition_dependent"] for block in theirs)
    assert not all(block["definition_dependent"] for block in theirs)


@needs_grid_output
def test_the_leaders_mirror_is_tier_statements_leaders_block_on_the_grids_committed_books() -> None:
    inputs = st.load_inputs(st.Roots(repo_root=REPO, output_root=hz.MAIN_OUTPUT_ROOT))
    books = st.make_books(inputs.cells)
    assert json.dumps(rp.hz4x4_leaders_block(books, **_grid_parameters()), sort_keys=True) == json.dumps(
        st.leaders_block(books), sort_keys=True)


# ======================================================================
# hz4x4's leaders
# ======================================================================

REFERENCE_LEVELS: dict[str, tuple[float, float]] = {
    "behaviour@maxpressure": (370.0, 365.0), "behaviour@fixedtime": (534.0, 555.0), "behaviour@mappo1000": (330.0, 325.0),
    "behaviour@random_fresh": (615.0, 670.0), "random@mappo1000": (608.0, 653.0)}


def _hz4x4_cells(*, dependent_tier: str = "fixedtime") -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for t_index, tier in enumerate(hz.TIER_ORDER):
        for m_index, method in enumerate(pb.ARMS):
            arm = f"{method}@{tier}"
            ours = 300.0 + 5.0 * m_index + t_index
            engine = ours - 4.0 if tier != dependent_tier else 300.0 + 5.0 * (len(pb.ARMS) - 1 - m_index)
            cells[f"p9_3:{arm}"] = _levelled(f"p9_3:{arm}", method, tier, "p9_3", arm, ours, engine)
    for method, tier in pb.REFERENCE_CELLS:
        arm = f"{method}@{tier}"
        ours, engine = REFERENCE_LEVELS[arm]
        cells[f"p9_3:{arm}"] = _levelled(f"p9_3:{arm}", method, tier, "p9_3", arm, ours, engine)
    return cells


def test_hz4x4s_leaders_have_no_primary_keep_the_references_beside_and_flag_definition_dependence() -> None:
    cells = _hz4x4_cells()
    blocks = rp.leaders(cells)
    assert [(b["tier"], b["definition"]) for b in blocks] == [(t, d) for t in hz.TIER_ORDER for d in st.DEFINITIONS]
    assert all((b["primary"], b["registered"], b["version"]) == (False, False, "both") for b in blocks)
    for block in blocks:
        assert list(block["levels"]) == list(pb.ARMS)
        assert list(block["references"]) == list(rp.REFERENCE_ARMS)
        assert not set(block["references"]) & set(block["leaders"] + block["runners_up"])
    by = {(b["tier"], b["definition"]): b for b in blocks}
    assert (by[("fixedtime", "att_ours")]["leaders"], by[("fixedtime", "att_engine")]["leaders"]) == (["dt_nomix"],
                                                                                                    ["iql_m"])
    assert by[("fixedtime", "att_ours")]["definition_dependent"] is True
    assert by[("maxpressure", "att_ours")]["definition_dependent"] is False
    rows = cells["p9_3:behaviour@maxpressure"].rows
    assert by[("maxpressure", "att_engine")]["references"]["behaviour@maxpressure"] == float(
        mean_ci95([r["att_engine"] for r in sorted(rows, key=lambda r: (r["seed"], r["draw_id"]))]).mean)
    mirrored = rp.hz4x4_leaders_block({"corrected": rp.book_of(cells)}, tiers=hz.TIER_ORDER, methods=pb.ARMS,
                                      primary_definition=None, versions_of=lambda tier: ("both",))
    assert [{key: value for key, value in block.items() if key in mirrored[0]} for block in blocks] == mirrored


# ======================================================================
# The cells, read from the records
# ======================================================================


def _eval_record(cell: Any, seed: int, draw: int) -> dict[str, Any]:
    ours = 300.0 + seed / 1000.0 + draw / 10000.0
    return {"format_version": pb.ARTIFACT_FORMAT_EVAL, "scenario": "hz4x4", "arm": cell.arm, "method": cell.method,
            "tier": cell.tier, "seed": seed, "draw_id": draw, "engine_seed": 1000, "att_ours": ours,
            "att_engine": ours + 1.5, "created": 2700, "entered": 2690, "never_entered": 10,
            "entered_fraction": 2690 / 2700,
            "route_i": {"att_horizon": ours, "horizon_vehicle_count": 12.0 + seed % 5, "episode_reward": -ours},
            "route_ii": {"att_ours": ours}, "seconds_route_i": 1.0, "reproduces": True, "policy_source": {"policy": "x"}}


def _write_records(out: Path, seeds: tuple[int, ...], draws: tuple[int, ...]) -> None:
    for cell, seed, draw in pb.planned_episodes(pb.evaluation_cells(), seeds=seeds, draws=draws):
        pb.write_eval_record(pb.eval_record_path(out, cell, seed, draw), _eval_record(cell, seed, draw))


def test_the_cells_are_read_from_their_records_in_seed_and_draw_order(tmp_path: Path) -> None:
    _write_records(tmp_path, (202, 101), (1001, 1000))
    cells = rp.load_cells(tmp_path, seeds=(202, 101), draws=(1001, 1000))
    assert list(cells) == [f"p9_3:{c.arm}" for c in pb.evaluation_cells()]
    cell = cells["p9_3:iql_m@random_fresh"]
    assert (cell.spec.method, cell.spec.tier, cell.spec.source, cell.spec.arm) == ("iql_m", "random_fresh", "p9_3",
                                                                                   "iql_m@random_fresh")
    assert [(r["seed"], r["draw_id"]) for r in cell.rows] == [(101, 1000), (101, 1001), (202, 1000), (202, 1001)]
    record = _eval_record(pb.EvalCell("iql_m", "random_fresh"), 202, 1001)
    assert cell.rows[3] == {"arm": "iql_m@random_fresh", "seed": 202, "draw_id": 1001, "att_ours": record["att_ours"],
                            "att_engine": record["att_engine"], "created": 2700, "entered": 2690, "never_entered": 10,
                            "entered_fraction": 2690 / 2700, "horizon_vehicle_count": 14.0,
                            "episode_reward": -record["att_ours"]}


def test_the_cells_are_refused_unless_complete_reproduced_and_free_of_refusals(tmp_path: Path) -> None:
    _write_records(tmp_path, (101,), (1000, 1001))
    cell = pb.EvalCell("dt_nomix", "fixedtime")
    path = pb.eval_record_path(tmp_path, cell, 101, 1001)
    saved = path.read_bytes()
    path.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        rp.load_cells(tmp_path, seeds=(101,), draws=(1000, 1001))
    record = json.loads(saved)
    record["att_ours"] = record["att_ours"] + 1.0
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="route \\(i\\)"):
        rp.load_cells(tmp_path, seeds=(101,), draws=(1000, 1001))
    path.write_bytes(saved)
    assert len(rp.load_cells(tmp_path, seeds=(101,), draws=(1000, 1001))) == 25
    pb.write_eval_record(pb.refusal_record_path(tmp_path, cell, 101, 1001), {"refusal": "does not reproduce"})
    with pytest.raises(ValueError, match="refused"):
        rp.load_cells(tmp_path, seeds=(101,), draws=(1000, 1001))


# ======================================================================
# Cell summaries, training, old cells
# ======================================================================


def test_a_cells_summary_is_the_grids_per_cell_computation_with_hz4x4s_sources() -> None:
    cells = _hz4x4_cells()
    mine, theirs = rp.cell_summaries(cells), st.cells_block(cells)
    assert list(mine) == list(cells)
    for cell_id, cell in cells.items():
        assert {k: v for k, v in mine[cell_id].items() if k != "sources"} == {
            k: v for k, v in theirs[cell_id].items() if k != "sources"}
        files = f"output/p9_3/eval/{cell.spec.arm}__seed*__draw*.json"
        assert mine[cell_id]["sources"] == {"att_ours": files, "att_engine": files, "companions": files,
                                            "horizon_vehicle_count": files}


def _training_tree(out: Path) -> None:
    for tier, arm, seed in pb.training_cells():
        payload = f"{tier}/{arm}/{seed}".encode()
        path = pb.checkpoint_path(out, tier, arm, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        record = pb.train_record_path(out, tier, arm, seed)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({
            "format_version": pb.ARTIFACT_FORMAT_TRAIN, "tier": tier, "arm": arm, "seed": seed,
            "seconds": 100.0 + seed / 1000.0, "gradient_steps": 40_000, "declared_gradient_steps": 40_000,
            "clock": pb.CLOCK_LABEL, "file_sha256": _sha(payload), "checkpoint_path": str(path),
            "load_average_1min": {"start": 1.0, "end": 1.5}, "git": {"commit": "c" * 40, "dirty": False}}),
            encoding="utf-8")


def test_the_training_block_carries_every_cell_under_the_clock_label_and_verifies_each_checkpoint(tmp_path: Path) -> None:
    _training_tree(tmp_path)
    rows = rp.training_block(tmp_path)
    assert [(r["tier"], r["arm"], r["seed"]) for r in rows] == pb.training_cells()
    assert all(r["clock"] == pb.CLOCK_LABEL and r["gradient_steps"] == 40_000 for r in rows)
    assert rows[7]["seconds"] == 100.0 + rows[7]["seed"] / 1000.0
    tier, arm, seed = pb.training_cells()[42]
    pb.checkpoint_path(tmp_path, tier, arm, seed).write_bytes(b"other weights")
    with pytest.raises(ValueError, match="is not the checkpoint"):
        rp.training_block(tmp_path)
    pb.checkpoint_path(tmp_path, tier, arm, seed).write_bytes(f"{tier}/{arm}/{seed}".encode())
    record_path = pb.train_record_path(tmp_path, tier, arm, seed)
    record = json.loads(record_path.read_bytes())
    record_path.write_text(json.dumps(dict(record, gradient_steps=2)), encoding="utf-8")
    with pytest.raises(ValueError, match="40000"):
        rp.training_block(tmp_path)
    record_path.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        rp.training_block(tmp_path)


def _grid_counterparts() -> tuple[dict[str, Any], dict[str, Any]]:
    grid: dict[str, Any] = {}
    files = []
    for index, ((method, tier), cell_id) in enumerate(rp.GRID_COUNTERPARTS.items()):
        source, arm = cell_id.split(":")
        eval_file = f"{source}/eval_{index}.json"
        grid[cell_id] = _levelled(cell_id, method, tier, source, arm, 160.0 + index, 155.0 + index, eval_file)
        files.append({"file": f"output/{eval_file}", "sha256": f"{index:064x}", "anchor": "its line in a manifest"})
    grid["p5_2:iql@random"] = _levelled("p5_2:iql@random", "iql", "random", "p5_2", "iql@random", 1.0, 1.0)
    provenance = {"inputs": files, "p8_4b_campaign": {"declared_cells_sha256": "a" * 64},
                  "p8_4b_rows": {"sha256": "b" * 64}}
    return grid, provenance


def test_old_cells_copy_each_counterpart_with_its_source_and_digest_and_name_p9_1s_as_pending() -> None:
    grid, provenance = _grid_counterparts()
    old = rp.old_cells(grid, provenance=provenance)
    assert old["format_version"] == rp.ARTIFACT_FORMAT_OLD_CELLS
    assert list(old["cells"]) == [f"{m}@{t}" for m, t in rp.GRID_COUNTERPARTS]
    for index, ((method, tier), cell_id) in enumerate(rp.GRID_COUNTERPARTS.items()):
        entry = old["cells"][f"{method}@{tier}"]
        assert entry["counterpart"] == cell_id
        assert entry["rows"] == [dict(r) for r in grid[cell_id].rows] and len(entry["rows"]) == 15
        assert (entry["eval_file"], entry["eval_file_sha256"]) == (f"output/{grid[cell_id].spec.eval_file}",
                                                                   f"{index:064x}")
    assert len(old["pending"]) == 18
    assert list(old["pending"]) == [f"{m}@{t}" for m, t in rp.P9_1_COUNTERPARTS]
    assert old["att_engine_rows"] == {"p8_4b_campaign": provenance["p8_4b_campaign"],
                                      "p8_4b_rows": provenance["p8_4b_rows"]}
    provenance["inputs"] = provenance["inputs"][1:]
    with pytest.raises(ValueError, match="no digest"):
        rp.old_cells(grid, provenance=provenance)


@needs_grid_output
def test_old_cells_on_the_grids_committed_inputs_carry_the_files_own_digest() -> None:
    inputs = st.load_inputs(st.Roots(repo_root=REPO, output_root=hz.MAIN_OUTPUT_ROOT))
    old = rp.old_cells(inputs.cells, provenance=inputs.provenance)
    assert len(old["cells"]) == 7 and {len(entry["rows"]) for entry in old["cells"].values()} == {500}
    entry = old["cells"]["dt_nomix@maxpressure"]
    assert entry["eval_file"] == "output/p5_2/eval_maxpressure_dt_nomix.json"
    assert entry["eval_file_sha256"] == _sha((hz.MAIN_OUTPUT_ROOT / "p5_2" / "eval_maxpressure_dt_nomix.json").read_bytes())
    assert old["cells"]["random@mappo1000"]["counterpart"] == st.ANCHOR_CELL


# ======================================================================
# The report
# ======================================================================


def test_the_report_cites_the_gate_and_the_declaration_by_digest_and_says_what_it_is_not() -> None:
    cells = _hz4x4_cells()
    training = [{"tier": t, "arm": a, "seed": s, "seconds": 1.0, "gradient_steps": 40_000, "clock": pb.CLOCK_LABEL}
                for t, a, s in pb.training_cells()]
    grid, provenance = _grid_counterparts()
    old = rp.old_cells(grid, provenance=provenance)
    artifact = rp.report_artifact(cells=cells, training=training, old=old, repo_root=REPO)
    gate_bytes = (REPO / rp.GATE_ARTIFACT).read_bytes()
    declaration_bytes = (REPO / rp.DECLARATION_ARTIFACT).read_bytes()
    declared = json.loads(declaration_bytes)
    assert artifact["format_version"] == rp.ARTIFACT_FORMAT_REPORT
    assert (artifact["label"], artifact["paper_sentence"]) == (hz.A27_LABEL, hz.PAPER_SENTENCE)
    assert artifact["primary_definition"] is None
    assert artifact["gate"] == {"path": str(rp.GATE_ARTIFACT), "sha256": _sha(gate_bytes),
                                "outcome": json.loads(gate_bytes)["outcome"]}
    assert artifact["declaration"] == {"path": str(rp.DECLARATION_ARTIFACT), "sha256": _sha(declaration_bytes)}
    assert artifact["exposure"] == {tier: declared["tiers"][tier]["exposure"] for tier in hz.TIER_ORDER}
    assert artifact["what_this_is_not"] == list(rp.WHAT_THIS_IS_NOT)
    assert artifact["leaders"] == rp.leaders(cells)
    assert artifact["cells"] == rp.cell_summaries(cells)
    assert artifact["episodes"] == {cell_id: [dict(r) for r in cell.rows] for cell_id, cell in cells.items()}
    assert artifact["training"] == {"clock": pb.CLOCK_LABEL, "rows": training}
    assert artifact["old_cells"] == {"format_version": old["format_version"],
                                     "sha256": _sha(json.dumps(old, sort_keys=True).encode("utf-8")),
                                     "counterparts": {arm: entry["counterpart"] for arm, entry in old["cells"].items()},
                                     "pending": list(old["pending"])}


def test_the_cli_names_its_two_commands() -> None:
    parse = rp.build_parser().parse_args
    old = parse(["old-cells", "--out", "o.json"])
    assert (old.command, old.out, old.output_root) == ("old-cells", "o.json", str(hz.MAIN_OUTPUT_ROOT))
    report = parse(["report", "--out", "r.json", "--old-cells", "o.json"])
    assert (report.command, report.out, report.old_cells, report.output_root) == ("report", "r.json", "o.json",
                                                                                  str(hz.MAIN_OUTPUT_ROOT))


def test_the_exposure_is_each_tiers_own(tmp_path: Path) -> None:
    # Mutant R2 (every tier given maxpressure's exposure) cannot be told apart on the committed declaration, whose four
    # tiers hold the same 1,152,000 decisions; a declaration whose tiers differ tells it apart.
    data = tmp_path / "docs" / "data"
    data.mkdir(parents=True)
    (data / rp.GATE_ARTIFACT.name).write_bytes((REPO / rp.GATE_ARTIFACT).read_bytes())
    declared = json.loads((REPO / rp.DECLARATION_ARTIFACT).read_bytes())
    for index, tier in enumerate(hz.TIER_ORDER):
        declared["tiers"][tier]["exposure"] = {"steps": 40_000 + index, "marker": tier}
    (data / rp.DECLARATION_ARTIFACT.name).write_text(json.dumps(declared), encoding="utf-8")
    grid, provenance = _grid_counterparts()
    artifact = rp.report_artifact(cells=_hz4x4_cells(), training=[], old=rp.old_cells(grid, provenance=provenance),
                                  repo_root=tmp_path)
    assert artifact["exposure"] == {tier: {"steps": 40_000 + index, "marker": tier}
                                    for index, tier in enumerate(hz.TIER_ORDER)}
