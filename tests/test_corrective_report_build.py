"""P9.1's report, part 2: the artifacts (``offline/corrective_report.py``; ``BRIEF_46`` 3.5; A27(e), (g) here).

Format ``p9.1-report/1.0``.  Each artifact is a pure function of a :class:`offline.corrective_report.Sources` -- the
cells (new and old), their per-episode records and training runs, the declarations -- and carries A27's label verbatim,
``registered_in``, the inputs by digest, the git block, the per-episode records of EVERY cell any of its blocks reads,
every compared run's ``seconds`` and ``gradient_steps`` with A27(a)'s clock label, the verdicts, and
``what_this_is_not``.  Written with sorted keys, no rounding, no timestamp; refused unless the tree is clean.

The synthetic sources below are P5.2c's fixture grid (3 seeds x 4 draws, dyadic values); every expected count is
recomputed here by its own route.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import offline.corrective_report as rep
import offline.corrective_round as cr
import offline.corrective_verdicts as cv
import offline.tier_statements as tst
from tests import p5_2c_fixtures as fx

REPO = Path(__file__).resolve().parents[1]


def _cell(cell_id: str, method: str, tier: str, level: float, arm: str | None = None) -> tst.Cell:
    ours = {(s, d): level + fx._term(cell_id, s, d) for s in fx.SEEDS for d in fx.DRAWS}
    built = fx.simple_cell(method, tier, ours, source=cell_id.split(":")[0], arm=arm or f"{method}@{tier}")
    return tst.Cell(spec=tst.CellSpec(cell_id=cell_id, method=method, tier=tier, source=built.spec.source,
                                      arm=built.spec.arm, eval_file=built.spec.eval_file, manifest="test"),
                    rows=built.rows)


def _records(cell: tst.Cell) -> tuple[dict[str, Any], ...]:
    return tuple({**row, "tier": cell.spec.tier, "source": f"test:{cell.spec.cell_id}"} for row in cell.rows)


def _run(cell_id: str, seed: int, seconds: float) -> dict[str, Any]:
    return {"cell": cell_id, "seed": seed, "seconds": seconds, "gradient_steps": 40000, "clock": cr.CLOCK_LABEL,
            "source": f"test:{cell_id}", "window_means": [3.0, 2.0, 1.0], "plateaued": False}


def _sources(cells: Mapping[str, tst.Cell], **extra: Any) -> rep.Sources:
    return rep.Sources(
        cells=dict(cells), records={cid: _records(cell) for cid, cell in cells.items()},
        runs={cid: tuple(_run(cid, s, 100.0 + s) for s in fx.SEEDS) for cid in cells
              if not cells[cid].spec.method.startswith(("behaviour", "random"))},
        declarations=extra.get("declarations", {"mix_m1000_m500": {"tier": "mix_m1000_m500"}}),
        grid_books=extra.get("grid_books", {}), p3a=extra.get("p3a", {}), schedules=extra.get("schedules", {}),
        d4rl=extra.get("d4rl"), registered=extra.get("registered", {"p4_4_delta": 0.6263}),
        inputs=({"file": "test", "sha256": "0" * 64},), git={"commit": "f" * 40, "dirty": False, "code_root": "/x"})


def _mixture_cells() -> dict[str, tst.Cell]:
    levels = {"bc_top10": 103.0, "iql": 104.5, "dt": 106.0, "bc": 108.0, "iql_t": 104.0}
    cells = {f"p9_1:hz1x1:{arm}@mix_m1000_m500": _cell(f"p9_1:hz1x1:{arm}@mix_m1000_m500", arm, "mix_m1000_m500", lv)
             for arm, lv in levels.items()}
    for cid, method, tier, level in (("p4_4:madt", "dt", "mappo1000", 104.96), ("p4_4:bc", "bc", "mappo1000", 105.16),
                                     ("p4_4:bc_top10", "bc_top10", "mappo1000", 103.16),
                                     ("p4_4:iql", "iql", "mappo1000", 103.48),
                                     ("p4_6:dt@mappo500", "dt", "mappo500", 106.66),
                                     ("p4_6:bc@mappo500", "bc", "mappo500", 107.28),
                                     ("p4_6:bc_top10@mappo500", "bc_top10", "mappo500", 104.75),
                                     ("p4_6:iql@mappo500", "iql", "mappo500", 104.82)):
        cells[cid] = _cell(cid, method, tier, level, arm=cid.split(":")[1])
    return cells


def _referenced(node: Any) -> set[str]:
    """Every cell id the artifact's blocks name (``left`` / ``right`` / ``inputs``), by its own walk."""
    found: set[str] = set()
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key in ("left", "right") and isinstance(value, str):
                found.add(value)
            elif key in ("inputs", "original_inputs") and isinstance(value, list):
                found.update(v for v in value if isinstance(v, str))
            else:
                found |= _referenced(value)
    elif isinstance(node, list):
        for item in node:
            found |= _referenced(item)
    return found


def _assert_common(payload: Mapping[str, Any], point: str, src: rep.Sources) -> None:
    assert payload["format_version"] == "p9.1-report/1.0"
    assert payload["label"] == cr.LABELS[point] and payload["registered_in"].startswith(point)
    assert payload["git"]["dirty"] is False and payload["inputs"]
    assert payload["what_this_is_not"] and all(isinstance(s, str) and s for s in payload["what_this_is_not"])
    blocks = {k: v for k, v in payload.items() if k not in ("records", "training_runs")}
    referenced = {cid for cid in _referenced(blocks) if cid in src.cells}
    assert referenced and referenced <= set(payload["records"]), sorted(referenced - set(payload["records"]))[:3]
    for cid, rows in payload["records"].items():
        assert [tuple(sorted(r)) for r in rows] == [tuple(sorted(rep.EPISODE_FIELDS))] * len(rows), cid
    for cid, runs in payload["training_runs"].items():
        assert all(run["clock"] == cr.CLOCK_LABEL for run in runs), cid


# ----------------------------------------------------------------------
# A27(e)
# ----------------------------------------------------------------------


def test_the_mixture_artifact_scores_the_four_contrasts_and_the_window_on_the_mixtures_own_cells() -> None:
    src = _sources(_mixture_cells())
    payload = rep.build_mixture(src)
    _assert_common(payload, "A27(e)", src)
    orderings = payload["predictions"]["orderings"]["contrasts"]
    assert [(c["left"], c["right"]) for c in orderings] == [
        ("p9_1:hz1x1:bc_top10@mix_m1000_m500", "p9_1:hz1x1:bc@mix_m1000_m500"),
        ("p9_1:hz1x1:bc_top10@mix_m1000_m500", "p9_1:hz1x1:iql@mix_m1000_m500"),
        ("p9_1:hz1x1:iql@mix_m1000_m500", "p9_1:hz1x1:dt@mix_m1000_m500"),
        ("p9_1:hz1x1:dt@mix_m1000_m500", "p9_1:hz1x1:bc@mix_m1000_m500")]
    assert [c["verdict"] for c in orderings] == ["HELD"] * 4
    assert [a["arm"] for a in payload["predictions"]["within_5_s"]["arms"]] == ["dt", "bc", "iql"]
    assert [a["arm"] for a in payload["beside"]["iql_t_within_5_s"]["arms"]] == ["iql_t"]
    assert payload["declaration"] == {"tier": "mix_m1000_m500"}
    parents = payload["parents"]
    assert set(parents) == {"mappo1000", "mappo500"} and set(parents["mappo1000"]) == {"dt", "bc", "bc_top10", "iql"}
    assert parents["mappo1000"]["dt"]["cell"] == "p4_4:madt"


def test_an_artifact_is_written_deterministically_and_refused_from_a_dirty_tree_or_with_a_nan(tmp_path: Path) -> None:
    src = _sources(_mixture_cells())
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    rep.write_artifact(first, rep.build_mixture(src))
    rep.write_artifact(second, rep.build_mixture(src))
    assert first.read_bytes() == second.read_bytes()
    text = first.read_text()
    assert text.endswith("\n") and json.loads(text) == json.loads(json.dumps(json.loads(text), sort_keys=True))
    with pytest.raises(ValueError, match="dirty"):
        rep.write_artifact(tmp_path / "c.json", {**json.loads(text), "git": {"commit": "f" * 40, "dirty": True}})
    with pytest.raises(ValueError, match="not JSON compliant"):
        rep.write_artifact(tmp_path / "d.json", {**json.loads(text), "x": float("nan")})
    assert not (tmp_path / "c.json").exists() and not (tmp_path / "d.json").exists()


# ----------------------------------------------------------------------
# A27(g)
# ----------------------------------------------------------------------


def _budget_cells() -> dict[str, tst.Cell]:
    cells = {}
    for cid, method, tier, level in (("p4_4:madt", "dt", "mappo1000", 104.96), ("p4_4:bc", "bc", "mappo1000", 105.16),
                                     ("p4_4:bc_top10", "bc_top10", "mappo1000", 103.16),
                                     ("p4_4:iql", "iql", "mappo1000", 103.48),
                                     ("p4_7:dt@mix50", "dt", "mix50", 110.0), ("p4_7:bc@mix50", "bc", "mix50", 233.7),
                                     ("p4_7:bc_top10@mix50", "bc_top10", "mix50", 104.2),
                                     ("p4_7:iql@mix50", "iql", "mix50", 130.7)):
        cells[cid] = _cell(cid, method, tier, level, arm=cid.split(":")[1])
    for tier, levels in (("mappo1000", {"dt": 104.0, "bc": 105.0, "bc_top10": 103.0, "iql": 103.5}),
                         ("mix50", {"dt": 108.0, "bc": 200.0, "bc_top10": 104.0, "iql": 120.0})):
        for arm, level in levels.items():
            cid = f"p9_1:hz1x1:{arm}_100k@{tier}"
            cells[cid] = _cell(cid, f"{arm}_100k", tier, level)
    return cells


def test_the_budget_artifact_pairs_each_100k_cell_with_its_registered_40k_cell_and_labels_mix50_beyond_a8b() -> None:
    src = _sources(_budget_cells())
    payload = rep.build_budget(src)
    _assert_common(payload, "A27(g)", src)
    assert payload["tiers"]["mappo1000"]["scope"] == "A8(b)'s secondary (the registered ceiling of 100,000 steps)"
    assert payload["tiers"]["mix50"]["scope"] == "an exploratory extension beyond A8(b)"
    pairs = {(t, d["arm"]): (d["left"], d["right"]) for t, b in payload["tiers"].items() for d in b["differences"]}
    assert pairs[("mappo1000", "dt")] == ("p9_1:hz1x1:dt_100k@mappo1000", "p4_4:madt")
    assert pairs[("mix50", "iql")] == ("p9_1:hz1x1:iql_100k@mix50", "p4_7:iql@mix50")
    assert payload["tiers"]["mappo1000"]["concordance"]["att_engine"]["verdict"] == "HELD"
    plateau = payload["plateau_at_100k"]
    assert {entry["cell"] for entry in plateau} == {cid for cid in src.cells if "_100k@" in cid}
    assert all(set(entry) >= {"seed", "plateaued", "window_means"} for entry in plateau)


def test_the_plateau_counts_the_paper_quotes_reproduce_from_the_committed_training_records() -> None:
    """A27(g): the author's command over the committed records prints 160 127 111 11 5; recomputed here by that
    command's own logic, and by the module."""
    runs = [r for name, key in (("p4_training", "seeds"), ("p4_4_training", "runs"), ("p4_6_training", "runs"),
                                ("p4_7_training", "runs"))
            for r in json.loads((REPO / f"docs/data/{name}.json").read_text())[key]]
    n = [r["window_means"] for r in runs if not r["plateaued"]]
    fall = sum(all(a > b for a, b in zip(w[-3:], w[-2:])) for w in n)
    rose = sum((not all(a > b for a, b in zip(w[-3:], w[-2:]))) and w[-1] > w[-2] for w in n)
    assert (len(runs), len(n), fall, rose, len(n) - fall - rose) == (160, 127, 111, 11, 5)
    assert rep.plateau_counts(REPO) == {"runs": 160, "not_plateaued": 127, "still_falling": 111,
                                        "rose_in_the_last_window": 11, "fluctuated": 5}


# ----------------------------------------------------------------------
# A27(b): exposure parity on the grid
# ----------------------------------------------------------------------

MATCHED = {"bc": "bc_m", "bc_top10": "bc_top10_m", "bc_top10_perix": "bc_top10_perix_m", "iql": "iql_m"}
GRID_TIERS = ("mappo1000", "maxpressure", "fixedtime", "random")


def _copy(cell: tst.Cell, cell_id: str, method: str, *, shift: float = 0.0) -> tst.Cell:
    rows = tuple({**row, "arm": f"{method}@{cell.spec.tier}", "att_ours": row["att_ours"] + shift,
                  "att_engine": row["att_engine"] + shift} for row in cell.rows)
    spec = tst.CellSpec(cell_id=cell_id, method=method, tier=cell.spec.tier, source="p9_1",
                        arm=f"{method}@{cell.spec.tier}", eval_file="x", manifest="test")
    return tst.Cell(spec=spec, rows=rows)


def _grid_sources(*, shift: Mapping[tuple[str, str], float] | None = None, **extra: Any) -> rep.Sources:
    synthetic = fx.synthetic_cells()
    books = tst.make_books(synthetic)
    cells: dict[str, tst.Cell] = dict(synthetic)
    for method, tier in cv.A27B_KEYS:
        cid = f"p9_1:grid4x4:{MATCHED[method]}@{tier}"
        cells[cid] = _copy(books["corrected"][(method, tier)], cid, MATCHED[method],
                           shift=(shift or {}).get((method, tier), 0.0))
    return _sources(cells, grid_books=books, **extra)


def test_exposure_parity_with_matched_cells_equal_to_their_originals_moves_nothing() -> None:
    src = _grid_sources()
    payload = rep.build_exposure_parity(src)
    _assert_common(payload, "A27(b)", src)
    rescore = payload["rescore"]
    assert rescore["n_statements"]["att_engine"] > 0
    assert rescore["n_moved"] == {"att_engine": 0, "att_ours": 0}
    assert payload["exposure_table"]


def test_exposure_parity_counts_the_registered_statements_that_move_by_its_own_route() -> None:
    src = _grid_sources(shift={("bc_top10", "random"): -200.0})
    rescore = rep.build_exposure_parity(src)["rescore"]
    counted = [e for e in rescore["statements"] if e["id"] in rep.A27B_STATEMENT_IDS and e["kind"] == "verdict"]
    for definition in ("att_engine", "att_ours"):
        mine = [e for e in counted if e["definition"] == definition]
        assert rescore["n_statements"][definition] == len(mine)
        assert rescore["n_moved"][definition] == sum(e["moved"] for e in mine) > 0
    untouched = [e for e in rescore["statements"]
                 if not set(e["original_inputs"]) & {f"p5_2:bc_top10@random"}]
    assert untouched and not any(e["moved"] or e["reading_changed"] for e in untouched)


def test_the_exposure_sentence_contrasts_are_dt_nomix_against_the_matched_bc_and_iql_on_mappo1000() -> None:
    sentence = rep.build_exposure_parity(_grid_sources())["sentence_contrasts"]
    assert [(e["left"], e["right"]) for e in sentence["matched"]] == [
        ("p5_1:dt_nomix@grid4x4_mappo1000", "p9_1:grid4x4:bc_m@mappo1000"),
        ("p5_1:dt_nomix@grid4x4_mappo1000", "p9_1:grid4x4:iql_m@mappo1000")]
    assert [(e["left"], e["right"]) for e in sentence["original"]] == [
        ("p5_1:dt_nomix@grid4x4_mappo1000", "p5_1:bc@grid4x4_mappo1000"),
        ("p5_1:dt_nomix@grid4x4_mappo1000", "p5_1:iql@grid4x4_mappo1000")]


# ----------------------------------------------------------------------
# A27(c): IQL targets without dropout
# ----------------------------------------------------------------------

HZ_TIERS = ("random", "fixedtime", "maxpressure", "mappo500", "mappo1000", "mix33", "mix50", "mix67")


def _iql_sources() -> rep.Sources:
    synthetic = fx.synthetic_cells()
    books = tst.make_books(synthetic)
    cells: dict[str, tst.Cell] = dict(synthetic)
    for i, tier in enumerate(HZ_TIERS):
        old = rep.HZ1X1_IQL_OLD[tier]
        cells[old] = _cell(old, "iql", tier, 104.0 + i, arm=old.split(":")[1])
        new = f"p9_1:hz1x1:iql_t@{tier}"
        cells[new] = _cell(new, "iql_t", tier, 104.0 + i + (0.5 if i % 2 else -3.0))
    cells["p4_4:madt"] = _cell("p4_4:madt", "dt", "mappo1000", 105.0, arm="madt")
    cells["p4_6:dt@random"] = _cell("p4_6:dt@random", "dt", "random", 112.0, arm="dt@random")
    for tier in GRID_TIERS:
        iql = books["corrected"][("iql", tier)]
        cells[f"p9_1:grid4x4:iql_t@{tier}"] = _copy(iql, f"p9_1:grid4x4:iql_t@{tier}", "iql_t", shift=-2.0)
        cells[f"p9_1:grid4x4:iql_m@{tier}"] = _copy(iql, f"p9_1:grid4x4:iql_m@{tier}", "iql_m", shift=-3.0)
    d4rl = {"arm": "iql_t", "contrasts": {"bc - iql_t": {"mean_difference": -1.0}}}
    # Q8 (BRIEF_46 Amendment B, B.2): P1's other three methods on every hz1x1 tier, the behaviour cells of the
    # comparisons with IQL (P4.6's maxpressure and fixedtime seedless, as committed), the constructed mixture
    # references, and the committed statements the originals must reproduce -- computed here on the artifacts' wiring
    # as this test spells it, so a report wired otherwise is refused.
    cells["p4_4:bc"] = _cell("p4_4:bc", "bc", "mappo1000", 105.5, arm="bc")
    cells["p4_4:bc_top10"] = _cell("p4_4:bc_top10", "bc_top10", "mappo1000", 103.5, arm="bc_top10")
    cells["p4_4:mappo1000"] = _cell("p4_4:mappo1000", "behaviour", "mappo1000", 101.0, arm="mappo1000")
    for i, tier in enumerate(("mappo500", "maxpressure", "fixedtime", "random", *MIXTURES)):
        for j, method in enumerate(("dt", "bc", "bc_top10")):
            cid = f"{'p4_7' if tier in MIXTURES else 'p4_6'}:{method}@{tier}"
            cells.setdefault(cid, _cell(cid, method, tier, 103.0 + i + 1.5 * j))
    for tier in P1_TIERS_P4_6:
        cid = f"p4_6:behaviour@{tier}"
        cells[cid] = _seedless(cid, tier, 99.0) if tier in ("maxpressure", "fixedtime") else _cell(cid, "behaviour",
                                                                                                    tier, 99.5)
    for tier in MIXTURES:
        cells[f"constructed:behaviour@{tier}"] = _cell(f"constructed:behaviour@{tier}", "behaviour", tier, 100.0)
    p1 = {name: cv.p1_scores({t: {m: cells[_p1_slot(m, t)] for m in ("dt", "bc", "bc_top10", "iql")} for t in tiers})
          ["att_ours"] for name, tiers in (("p4_6", P1_TIERS_P4_6), ("p4_7", P1_TIERS_P4_6 + MIXTURES))}
    comparisons = {"p4_4:mappo1000_vs_iql": cv.paired_both(cells["p4_4:mappo1000"], cells["p4_4:iql"],
                                                           serializer="offline_baselines")["att_ours"]}
    for task, tier, behaviour in ([("p4_6", t, f"p4_6:behaviour@{t}") for t in P1_TIERS_P4_6]
                                  + [("p4_7", t, f"constructed:behaviour@{t}") for t in MIXTURES]):
        comparisons[f"{task}:iql@{tier} against behaviour@{tier}"] = cv.paired_both(
            cells[_p1_slot("iql", tier)], cells[behaviour], serializer="method_tier_grid",
            labels=(f"iql@{tier}", f"behaviour@{tier}"))["att_ours"]
    registered = {"p4_4_delta": 0.6263, "p1_committed": p1, "iql_behaviour_committed": comparisons}
    return _sources(cells, grid_books=books, d4rl=d4rl, registered=registered)


P1_TIERS_P4_6 = ("mappo1000", "mappo500", "maxpressure", "fixedtime", "random")
MIXTURES = ("mix33", "mix50", "mix67")


def _p1_slot(method: str, tier: str) -> str:
    """The cell P1 reads for *method* on *tier*: P4.4's on mappo1000, P4.7's on a mixture, P4.6's elsewhere."""
    if tier == "mappo1000":
        return {"dt": "p4_4:madt", "bc": "p4_4:bc", "bc_top10": "p4_4:bc_top10", "iql": "p4_4:iql"}[method]
    return f"{'p4_7' if tier in MIXTURES else 'p4_6'}:{method}@{tier}"


def _seedless(cell_id: str, tier: str, level: float) -> tst.Cell:
    """A deterministic behaviour policy's cell as the report loads it: one episode per draw, ``seed`` None."""
    rows = tuple({"arm": f"behaviour@{tier}", "seed": None, "draw_id": d, "att_ours": level + fx._term(cell_id, 0, d),
                  "att_engine": level - 7.0 + fx._term(cell_id, 0, d), "created": 1300, "entered": 1300,
                  "never_entered": 0, "entered_fraction": 1.0, "horizon_vehicle_count": 10.0, "episode_reward": -1.0}
                 for d in fx.DRAWS)
    return tst.Cell(spec=tst.CellSpec(cell_id=cell_id, method="behaviour", tier=tier, source=cell_id.split(":")[0],
                                      arm=f"behaviour@{tier}", eval_file="x", manifest="test"), rows=rows)


def test_iql_targets_pairs_every_tier_and_summarises_the_twelve() -> None:
    src = _iql_sources()
    payload = rep.build_iql_targets(src)
    _assert_common(payload, "A27(c)", src)
    hz, grid = payload["hz1x1"]["tiers"], payload["grid4x4"]["tiers"]
    assert [e["tier"] for e in hz] == list(HZ_TIERS) and [e["tier"] for e in grid] == list(GRID_TIERS)
    assert {e["right"] for e in grid if e["tier"] == "random"} == {tst.CORRECTED_IQL}
    for definition in ("att_engine", "att_ours"):
        points = [e["by_definition"][definition]["mean_difference"] for e in hz + grid]
        excluding = sum(e["by_definition"][definition]["excludes_zero"] for e in hz + grid)
        assert payload["twelve_tiers"][definition] == {"n_tiers": 12, "n_excluding_zero": excluding,
                                                       "min": min(points), "max": max(points)}
    assert [e["tier"] for e in payload["grid4x4_exposure_against_dropout"]["tiers"]] == list(GRID_TIERS)
    assert payload["p4_4_a6"]["original"]["right"] == "p4_4:iql"
    assert payload["p4_4_a6"]["rescored"]["right"] == "p9_1:hz1x1:iql_t@mappo1000"
    assert payload["d4rl"] == src.d4rl


def test_the_iql_rescore_keeps_only_p5_2_statements_that_read_an_iql_cell() -> None:
    payload = rep.build_iql_targets(_iql_sources())
    iql_ids = {f"p9_1:grid4x4:iql_t@{t}" for t in GRID_TIERS}
    entries = payload["p5_2_rescore"]["statements"]
    assert entries and all(set(e["inputs"]) & iql_ids for e in entries)
    assert payload["p5_2_rescore"]["n_moved"] == {d: sum(e["moved"] for e in entries if e["definition"] == d)
                                                  for d in ("att_engine", "att_ours")}


def test_q8_rescores_p1_of_p4_6_and_p4_7_with_iql_t_in_iqls_slot_beside_the_original() -> None:
    """Q8: P1 on P4.6's five tiers and P4.7's eight, each original reproducing its committed P1 under att_ours, the
    re-score reading iql_t where IQL stood and nothing else changed.  On maxpressure BC is second of four among the
    originals (dt 104, bc 105.5, iql 106, bc_top10 107) and third with iql_t at 103 -- ranks read off the levels,
    which no per-episode term (|term| <= 14/64) can reorder."""
    src = _iql_sources()
    payload = rep.build_iql_targets(src)
    _assert_common(payload, "A27(c)", src)
    blocks = payload["p1_rescore"]
    assert sorted(blocks) == ["p4_6", "p4_7"]
    for name, tiers in (("p4_6", P1_TIERS_P4_6), ("p4_7", P1_TIERS_P4_6 + MIXTURES)):
        block = blocks[name]
        original = {_p1_slot(m, t) for t in tiers for m in ("dt", "bc", "bc_top10", "iql")}
        assert sorted(block["original_inputs"]) == sorted(original)
        assert sorted(block["inputs"]) == sorted({_p1_slot(m, t) for t in tiers for m in ("dt", "bc", "bc_top10")}
                                                 | {f"p9_1:hz1x1:iql_t@{t}" for t in tiers})
        assert block["substituted"] == {t: {"original": _p1_slot("iql", t), "replacement": f"p9_1:hz1x1:iql_t@{t}"}
                                        for t in tiers}
        assert block["original"]["att_ours"] == src.registered["p1_committed"][name]
        for definition in ("att_engine", "att_ours"):
            assert block["original"][definition]["bc_rank_by_tier"]["maxpressure"] == 2
            assert block["rescored"][definition]["bc_rank_by_tier"]["maxpressure"] == 3
            assert block["moved"][definition] == (block["original"][definition]["outcome"]
                                                  != block["rescored"][definition]["outcome"])
    assert {"p4_6:bc@maxpressure", "p4_7:dt@mix33", "p4_4:bc_top10"} <= set(payload["records"])


def test_q8_rescores_every_behaviour_comparison_with_iql_in_its_artifacts_order() -> None:
    """P4.4's ``mappo1000_vs_iql`` (behaviour left, P4.4's serialisation), P4.6's five ``iql@T`` against
    ``behaviour@T`` (IQL left; on mappo1000 P4.4's IQL cell under P4.6's name) and P4.7's three mixtures against the
    constructed references; P4.7's phase-1 five are P4.6's and listed once.  The re-score reads iql_t in IQL's place."""
    src = _iql_sources()
    payload = rep.build_iql_targets(src)
    entries = payload["iql_behaviour_comparisons"]["comparisons"]
    expected = [("p4_4", "mappo1000", ["p4_4:mappo1000", "p4_4:iql"], ["p4_4:mappo1000", "p9_1:hz1x1:iql_t@mappo1000"],
                 ("mappo1000", "iql"), ("mappo1000", "iql_t@mappo1000"))]
    expected += [("p4_6", t, [_p1_slot("iql", t), f"p4_6:behaviour@{t}"], [f"p9_1:hz1x1:iql_t@{t}", f"p4_6:behaviour@{t}"],
                  (f"iql@{t}", f"behaviour@{t}"), (f"iql_t@{t}", f"behaviour@{t}")) for t in P1_TIERS_P4_6]
    expected += [("p4_7", t, [f"p4_7:iql@{t}", f"constructed:behaviour@{t}"],
                  [f"p9_1:hz1x1:iql_t@{t}", f"constructed:behaviour@{t}"],
                  (f"iql@{t}", f"behaviour@{t}"), (f"iql_t@{t}", f"behaviour@{t}")) for t in MIXTURES]
    got = [(e["artifact"], e["tier"], e["original_inputs"], e["inputs"],
            (e["original"]["att_ours"]["left_arm"], e["original"]["att_ours"]["right_arm"]),
            (e["rescored"]["att_engine"]["left_arm"], e["rescored"]["att_engine"]["right_arm"])) for e in entries]
    assert got == expected
    for e in entries:
        for definition in ("att_engine", "att_ours"):
            for version in ("original", "rescored"):
                low, high = e[version][definition]["ci95_low"], e[version][definition]["ci95_high"]
                assert e["side_of_zero"][version][definition] == (
                    "below zero" if high < 0.0 else "above zero" if low > 0.0 else "includes zero")
            assert e["side_changed"][definition] == (e["side_of_zero"]["original"][definition]
                                                     != e["side_of_zero"]["rescored"][definition])
    seedless = next(e for e in entries if (e["artifact"], e["tier"]) == ("p4_6", "maxpressure"))
    assert seedless["original"]["att_engine"]["n_shared_draws"] == len(fx.DRAWS)
    assert {"p4_6:behaviour@fixedtime", "constructed:behaviour@mix67", "p4_4:mappo1000"} <= set(payload["records"])


def test_an_original_that_does_not_reproduce_its_committed_statement_is_refused() -> None:
    """The originals are the registered cells joined to their re-rolls: under att_ours each must BE its committed
    statement, so a report reading a different cell or order -- or a committed value that differs in its last bit --
    is refused rather than reported."""
    src = _iql_sources()
    p1 = dict(src.registered["p1_committed"])
    p1["p4_7"] = {**p1["p4_7"], "bc_rank_by_tier": {**p1["p4_7"]["bc_rank_by_tier"], "mix50": 0}}
    with pytest.raises(ValueError, match="committed P1"):
        rep.build_iql_targets(rep.Sources(**{**src.__dict__, "registered": {**src.registered, "p1_committed": p1}}))
    key = "p4_6:iql@maxpressure against behaviour@maxpressure"
    comparisons = dict(src.registered["iql_behaviour_committed"])
    value = comparisons[key]["mean_difference"]
    comparisons[key] = {**comparisons[key], "mean_difference": value + abs(value) * 2.0 ** -52}
    with pytest.raises(ValueError, match="committed comparison"):
        rep.build_iql_targets(rep.Sources(**{**src.__dict__, "registered": {**src.registered,
                                                                             "iql_behaviour_committed": comparisons}}))


# ----------------------------------------------------------------------
# A27(d): the true random tier
# ----------------------------------------------------------------------


def _p3a_rows(prefix: str, *, repeat_on: int | None = None, columns: int = 1) -> list[dict[str, Any]]:
    rows = []
    for s in fx.SEEDS:
        for d in fx.DRAWS:
            tag = f"{prefix}{s}-{1000 if (repeat_on == s and d == 1001) else d}"
            rows.append({"seed": s, "draw_id": d, "att_ours": 1.0, "n_decisions": 360, "action_sequence_sha256": tag,
                         "per_intersection_sha256": [f"{tag}/{i}" for i in range(columns)]})
    return rows


def _fresh_sources() -> rep.Sources:
    synthetic = fx.synthetic_cells()
    books = tst.make_books(synthetic)
    cells: dict[str, tst.Cell] = dict(synthetic)
    for arm, level in (("dt", 250.0), ("bc", 262.0), ("bc_top10", 300.0), ("iql", 190.0), ("iql_t", 188.0),
                       ("behaviour", 255.0)):
        cid = f"p9_1:hz1x1:{arm}@random_fresh"
        cells[cid] = _cell(cid, arm, "random_fresh", level)
    for arm, level in (("dt", 240.0), ("bc", 260.0), ("bc_top10", 330.0), ("iql", 191.0), ("behaviour", 256.0)):
        cid = f"p4_6:{arm}@random"
        cells[cid] = _cell(cid, arm, "random", level, arm=f"{arm}@random")
    for arm, level in (("dt_nomix", 250.0), ("bc_m", 262.0), ("bc_top10_m", 300.0), ("bc_top10_perix_m", 270.0),
                       ("iql_m", 190.0), ("behaviour", 249.0)):
        cid = f"p9_1:grid4x4:{arm}@random_fresh"
        cells[cid] = _cell(cid, arm, "random_fresh", level)
    p3a = {"p9_1:hz1x1:dt@random_fresh": _p3a_rows("hdt"), "p9_1:hz1x1:behaviour@random_fresh": _p3a_rows("hfr"),
           "p4_6:behaviour@random": _p3a_rows("hfive"),
           "p9_1:grid4x4:dt_nomix@random_fresh": _p3a_rows("gdt", repeat_on=202, columns=16),
           "p9_1:grid4x4:behaviour@random_fresh": _p3a_rows("gfr", columns=16),
           tst.ANCHOR_CELL: _p3a_rows("gfive", columns=16)}
    schedules = {"hz1x1": {"joint": {"hdt101-1000"}, "per_intersection": [{"hdt101-1000/0"}]},
                 "grid4x4": {"joint": set(), "per_intersection": [set() for _ in range(16)]}}
    declarations = {"random_fresh_hz1x1": {"scenario": "hz1x1"}, "random_fresh_grid4x4": {"scenario": "grid4x4"}}
    return _sources(cells, grid_books=books, p3a=p3a, schedules=schedules, declarations=declarations)


def test_random_fresh_scores_p3a_per_scenario_with_the_beside_table_and_both_references() -> None:
    src = _fresh_sources()
    payload = rep.build_random_fresh(src)
    _assert_common(payload, "A27(d)", src)
    assert payload["p3a"]["hz1x1"]["verdict"] == "HELD" and payload["p3a"]["hz1x1"]["repeated_episodes"] == 0
    assert payload["p3a"]["grid4x4"]["verdict"] == "FAILED"
    assert payload["p3a"]["grid4x4"]["repeated"] == [[202, 1000], [202, 1001]]
    beside = payload["p3a_beside"]["hz1x1"]["per_seed"]["101"]
    assert beside["replays_a_training_schedule"] == 1
    assert set(beside["identical_to_reference"]) == {"registered_five_schedule", "random_fresh"}
    assert payload["declarations"] == {"hz1x1": {"scenario": "hz1x1"}, "grid4x4": {"scenario": "grid4x4"}}


def test_every_collapse_check_runs_against_both_references_on_both_scenarios() -> None:
    payload = rep.build_random_fresh(_fresh_sources())
    for scenario, arms in (("hz1x1", ("dt", "bc", "bc_top10", "iql", "iql_t")),
                           ("grid4x4", ("dt_nomix", "bc_m", "bc_top10_m", "bc_top10_perix_m", "iql_m"))):
        block = payload["collapse"][scenario]["random_fresh_tier"]
        assert {(r["arm"], r["anchor"]) for r in block["rows"]} == {
            (a, ref) for a in arms for ref in ("registered_five_schedule", "random_fresh")}
        for definition, counts in block["collapsed_count"].items():
            for ref, n in counts.items():
                assert n == sum(r["by_definition"][definition]["verdict"] == "COLLAPSED" for r in block["rows"]
                                if r["anchor"] == ref)
    old = payload["collapse"]["hz1x1"]["original_random_tier"]
    assert {r["anchor"] for r in old["rows"]} == {"registered_five_schedule", "random_fresh"}
    q6 = payload["q6_both_anchors"]
    assert set(q6) == {"registered_five_schedule", "random_fresh"}
    assert all(e["id"] == "Q6" for entries in q6.values() for e in entries)


def test_p3b_reports_the_iql_lead_on_both_tiers_of_both_scenarios() -> None:
    p3b = rep.build_random_fresh(_fresh_sources())["p3b"]
    pairs = {(e["scenario"], e["tier"], e["left"], e["right"]) for e in p3b["contrasts"]}
    assert ("hz1x1", "random_fresh", "p9_1:hz1x1:iql@random_fresh", "p9_1:hz1x1:dt@random_fresh") in pairs
    assert ("hz1x1", "random", "p4_6:iql@random", "p4_6:dt@random") in pairs
    assert ("grid4x4", "random_fresh", "p9_1:grid4x4:iql_m@random_fresh", "p9_1:grid4x4:dt_nomix@random_fresh") in pairs
    assert ("grid4x4", "random", tst.CORRECTED_IQL, "p5_2:dt_nomix@random") in pairs
    assert p3b["prediction"] is None


# ----------------------------------------------------------------------
# Added after the C6c mutation table (B5, B9, B16, B17 survived)
# ----------------------------------------------------------------------


def test_every_100k_cell_is_paired_with_the_registered_40k_cell_of_its_own_tier_and_arm() -> None:
    """B5: the eight pairings, written here from the registration's sources (P4.4's mappo1000 cells, P4.7's mix50)."""
    payload = rep.build_budget(_sources(_budget_cells()))
    pairs = {(t, d["arm"]): d["right"] for t, b in payload["tiers"].items() for d in b["differences"]}
    assert pairs == {("mappo1000", "dt"): "p4_4:madt", ("mappo1000", "bc"): "p4_4:bc",
                     ("mappo1000", "bc_top10"): "p4_4:bc_top10", ("mappo1000", "iql"): "p4_4:iql",
                     ("mix50", "dt"): "p4_7:dt@mix50", ("mix50", "bc"): "p4_7:bc@mix50",
                     ("mix50", "bc_top10"): "p4_7:bc_top10@mix50", ("mix50", "iql"): "p4_7:iql@mix50"}


def test_a_statement_that_moves_without_reading_a_replaced_cell_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """B9: the substitution may only reach the statements that read it."""
    real = cv.moved_block

    def tampered(original: Any, rescored: Any) -> list[dict[str, Any]]:
        block = real(original, rescored)
        victim = next(e for e in block if e["id"] == "Q0a")
        victim["moved"] = True
        return block

    monkeypatch.setattr(cv, "moved_block", tampered)
    with pytest.raises(ValueError, match="moved without reading a replaced cell"):
        rep.build_exposure_parity(_grid_sources())


def test_on_the_grid_p3a_counts_a_repeated_intersection_even_when_the_joint_sequence_is_new() -> None:
    """B16: per intersection, not the joint array -- one column equal on two draws of a seed, joints all distinct."""
    src = _fresh_sources()
    rows = [dict(r, per_intersection_sha256=list(r["per_intersection_sha256"]))
            for r in src.p3a["p9_1:grid4x4:dt_nomix@random_fresh"]]
    rows = [dict(r, action_sequence_sha256=f"joint-{r['seed']}-{r['draw_id']}") for r in rows]
    for r in rows:
        r["per_intersection_sha256"] = [f"col{i}-{r['seed']}-{r['draw_id']}" for i in range(16)]
        if r["seed"] == 303 and r["draw_id"] in (1002, 1003):
            r["per_intersection_sha256"][7] = "shared"
    p3a = dict(src.p3a)
    p3a["p9_1:grid4x4:dt_nomix@random_fresh"] = rows
    payload = rep.build_random_fresh(rep.Sources(**{**src.__dict__, "p3a": p3a}))
    assert payload["p3a"]["grid4x4"]["repeated"] == [[303, 1002], [303, 1003]]
    assert payload["p3a"]["grid4x4"]["verdict"] == "FAILED"


def test_q6_against_the_fresh_reference_reads_the_fresh_reference_and_the_registered_one_reads_p5_1s_anchor() -> None:
    """B17: the anchor is substituted, not merely relabelled."""
    q6 = rep.build_random_fresh(_fresh_sources())["q6_both_anchors"]
    assert all(tst.ANCHOR_CELL in e["inputs"] for e in q6["registered_five_schedule"])
    assert all("p9_1:grid4x4:behaviour@random_fresh" in e["inputs"] and tst.ANCHOR_CELL not in e["inputs"]
               for e in q6["random_fresh"])


def test_the_random_fresh_artifact_carries_p5_3bs_old_tier_numbers_beside_p3a() -> None:
    src = _fresh_sources()
    old = {"source": "docs/data/p5_3b_nortg.json mechanism.action_identity", "sha256": "b" * 64,
           "n_distinct_action_sequences": {"dt@random seed 101": 1}, "attractor": {"mappo1000": {}}}
    payload = rep.build_random_fresh(rep.Sources(**{**src.__dict__, "registered": {**src.registered,
                                                                                    "p5_3b_old_tier": old}}))
    assert payload["p3a_beside"]["hz1x1_original_tier_p5_3b"] == old
