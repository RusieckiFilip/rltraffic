"""P9.1's stage runners and CLI (``offline/corrective_round.py``; ``BRIEF_46`` 3.4): the evaluation cell list, the
per-cell evaluation (route (i) -> route (ii) -> P3a), the status computed from disk, the pre-token check, and the
command line the driver calls.

Formats: route (i)'s cell files (``p4.6-grid/1.0`` / ``p5.2-tier-sweep/1.0``), P8.4b's campaign manifest
(``p8.4b-rederivation/1.0``), ``p9.1-actions/1.0``.  The routes are stubbed here -- their own behaviour is pinned in
``tests/test_corrective_round_eval.py`` -- so these tests see only the ORDER and the refusals of the orchestration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import offline.corrective_round as cr


def _roots(tmp_path: Path) -> cr.Roots:
    output = tmp_path / "output"
    output.mkdir()
    return cr.Roots(repo_root=Path(__file__).resolve().parents[1], output_root=output, corpus_root=tmp_path / "corpus",
                    draws_root=tmp_path / "draws")


def test_the_evaluation_cells_are_the_trained_cells_and_the_two_fresh_references() -> None:
    cells = cr.evaluation_cells()
    assert set(cells) == set(cr.training_cells()) | set(cr.reference_cells())
    assert len(cells) == 53


def test_the_p3a_cells_are_a27ds_dts_and_both_references_on_both_scenarios() -> None:
    assert {(c.scenario, c.key) for c in cr.p3a_cells()} == {
        ("hz1x1", "dt@random_fresh"), ("hz1x1", "behaviour@random_fresh"), ("hz1x1", "behaviour@random"),
        ("grid4x4", "dt_nomix@random_fresh"), ("grid4x4", "behaviour@random_fresh"), ("grid4x4", "random@mappo1000"),
    }


def test_shards_partition_the_evaluation_cells_deterministically() -> None:
    shards = [cr.shard_of(cr.evaluation_cells(), shard=k, of=6) for k in range(6)]
    flat = [cell for shard in shards for cell in shard]
    assert sorted(flat, key=cr.cell_sort_key) == sorted(cr.evaluation_cells(), key=cr.cell_sort_key)
    assert len(flat) == len(set(flat))
    with pytest.raises(ValueError, match="partition"):
        cr.shard_of(cr.evaluation_cells(), shard=6, of=6)


def test_a_cell_is_evaluated_only_once_its_runs_are_done(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    cell = next(c for c in cr.training_cells() if c.arm == "iql_t" and c.scenario == "hz1x1")
    with pytest.raises(ValueError, match="not trained"):
        cr.evaluate_cell(roots, cell)


def test_evaluate_cell_runs_route_i_then_route_ii_with_route_is_values(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _roots(tmp_path)
    cell = next(c for c in cr.reference_cells() if c.scenario == "grid4x4")
    order: list[str] = []
    payload = {"episodes": [{"arm": cell.key, "seed": 101, "draw_id": 1000, "att_horizon": 250.0}]}
    monkeypatch.setattr(cr, "route_i", lambda r, c: order.append("i") or payload)
    monkeypatch.setattr(cr, "run_route_ii",
                        lambda r, scenario, keys, committed: order.append("ii") or {"rolled": len(keys), "keys": keys,
                                                                                     "committed": committed})
    monkeypatch.setattr(cr, "record_p3a", lambda r, c, values: order.append("p3a") or [])
    outcome = cr.evaluate_cell(roots, cell)
    assert order == ["i", "ii", "p3a"]
    assert outcome["route_ii"]["committed"] == {("grid4x4", cell.key, 101, 1000): 250.0}


def test_status_is_computed_from_disk_and_reports_every_stage(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    status = cr.status(roots)
    assert status["training"]["runs_done"] == 0 and status["training"]["runs_declared"] == 255
    assert status["evaluation"]["route_i_cells_done"] == 0 and status["evaluation"]["cells_declared"] == 53
    assert status["complete"] is False


def test_check_writes_nothing_and_refuses_a_missing_corpus(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    before = sorted(p.name for p in tmp_path.rglob("*"))
    with pytest.raises(FileNotFoundError, match="corpus"):
        cr.check(roots, stage="campaign")
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("argv", [
    ["status"], ["check", "--stage", "collect"], ["check", "--stage", "campaign"],
    ["collect-random-fresh", "--scenario", "hz1x1"], ["declare", "--what", "mixture"],
    ["declare", "--what", "random_fresh", "--scenario", "grid4x4"], ["train", "--scenario", "grid4x4", "--tier", "random"],
    ["evaluate", "--shard", "2", "--of", "6"], ["old-cells"],
])
def test_the_cli_parses_every_stage_the_driver_calls(argv: list[str]) -> None:
    args = cr.build_parser().parse_args(["--output-root", "/o", "--corpus-root", "/c", "--draws-root", "/d", *argv])
    assert args.command == argv[0]


def test_the_cli_refuses_and_exits_2_naming_the_reason(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cr.main(["--output-root", str(tmp_path / "missing_output"), "--corpus-root", str(tmp_path),
                    "--draws-root", str(tmp_path), "check", "--stage", "campaign"])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err


# ----------------------------------------------------------------------
# Added after the C5 mutation table (K18, K19 survived) and for the serial evaluation declaration
# ----------------------------------------------------------------------


def test_a_cells_shard_does_not_depend_on_the_order_the_cells_are_listed_in() -> None:
    """K19: a shard is a function of the cell SET -- the driver's six processes each rebuild the list, and two
    processes that listed it differently must still never evaluate the same cell or skip one."""
    cells = list(cr.evaluation_cells())
    by_arm = sorted(cells, key=lambda c: (c.arm, c.tier, c.scenario))
    for k in range(6):
        assert cr.shard_of(cells, shard=k, of=6) == cr.shard_of(cells[::-1], shard=k, of=6)
        assert cr.shard_of(cells, shard=k, of=6) == cr.shard_of(by_arm, shard=k, of=6)


def _campaign_inputs_present(roots: cr.Roots) -> None:
    """Everything ``check --stage campaign`` reads before the prompt: both fresh manifests (A27(d)'s collection) and
    the three declarations; the held-out draws are NOT materialised."""
    Path(roots.corpus_root).mkdir()
    Path(roots.draws_root).mkdir()
    meta = {**cr.A27D_COLLECTION, "random_rng_scheme": "per-episode", "episodes": 1}
    for _, fresh in cr.RANDOM_FRESH_DIRS.values():
        directory = Path(roots.corpus_root) / fresh
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({"run_metadata": meta, "episodes": [{}] * 200}))
    Path(roots.run_root).mkdir(parents=True)
    for name in ("declaration_mix_m1000_m500.json", "declaration_random_fresh_hz1x1.json",
                 "declaration_random_fresh_grid4x4.json"):
        (Path(roots.run_root) / name).write_text("{}")


def test_check_refuses_the_campaign_while_hz1x1_random_freshs_prompt_is_untyped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K18 (plan Q2): with every other input present, an untyped prompt is the refusal; typed, the check moves on to
    the next input (the held-out draws, absent here)."""
    import dataclasses

    import offline.method_tier_grid as grid

    roots = _roots(tmp_path)
    _campaign_inputs_present(roots)
    spec = grid.A27_TIERS["random_fresh"]
    monkeypatch.setitem(grid.A27_TIERS, "random_fresh",
                        dataclasses.replace(spec, target_rtg=float("nan"), rtg_scale=float("nan")))
    with pytest.raises(ValueError, match="not typed"):
        cr.check(roots, stage="campaign")
    monkeypatch.setitem(grid.A27_TIERS, "random_fresh", dataclasses.replace(spec, target_rtg=-100.0, rtg_scale=50.0))
    with pytest.raises(FileNotFoundError, match="not materialised"):
        cr.check(roots, stage="campaign")


def _declared(path: Path) -> list[str]:
    return list(json.loads(path.read_text(encoding="utf-8"))["cells"])


def test_declare_evaluation_writes_every_manifest_once_from_the_cell_lists(tmp_path: Path) -> None:
    """The three campaign manifests (route (ii) per scenario, the old cells) are declared ONCE, serially, before any
    shard starts; recomputed here from the registered draws and seeds and from the committed artifacts' raw episodes."""
    from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS

    roots = _roots(tmp_path)
    cr.declare_evaluation(roots)
    runs = Path(roots.runs_root)
    for scenario in ("hz1x1", "grid4x4"):
        cells = [c for c in cr.evaluation_cells() if c.scenario == scenario]
        expected = sorted(f"{scenario}|{c.key}|{s}|{d}" for c in cells for s in TRAINING_SEEDS for d in HELD_OUT_DRAWS)
        assert _declared(runs / "eval_ii" / scenario / "campaign_manifest.json") == expected
    repo = Path(__file__).resolve().parents[1]
    expected_old = []
    for artifact, arm in cr.HZ1X1_OLD_CELLS:
        for episode in json.loads((repo / artifact).read_text(encoding="utf-8"))["episodes"]:
            if episode["arm"] == arm:
                expected_old.append(f"hz1x1|{arm}|{episode.get('seed')}|{episode['draw_id']}")
    assert _declared(runs / "old_cells" / "hz1x1" / "campaign_manifest.json") == sorted(expected_old)
    before = {p: p.read_bytes() for p in runs.rglob("campaign_manifest.json")}
    cr.declare_evaluation(roots)
    assert {p: p.read_bytes() for p in runs.rglob("campaign_manifest.json")} == before


def test_declare_evaluation_refuses_a_manifest_that_declares_another_cell_set(tmp_path: Path) -> None:
    """Two ways a manifest on disk is not this campaign's: another campaign's (count and digest differ -- P8.4b's
    agreement check), and a ``cells`` list edited under the original count and digest (the field
    ``campaign_status`` reads; P8.4b's check compares only the count and the digest)."""
    from offline import att_rederivation as ar

    roots = _roots(tmp_path)
    cr.declare_evaluation(roots)
    path = Path(roots.runs_root) / "eval_ii" / "grid4x4" / "campaign_manifest.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    other = [ar.CellKey(*key.split("|")[:2], int(key.split("|")[2]), int(key.split("|")[3]))
             for key in original["cells"][1:]]
    path.write_text(json.dumps(ar.campaign_manifest(other, engine_seed=1000)), encoding="utf-8")
    with pytest.raises(ValueError, match="but this run declares"):
        cr.declare_evaluation(roots)
    path.write_text(json.dumps({**original, "cells": original["cells"][1:] + ["grid4x4|x@y|101|1000"]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="is not this campaign's manifest"):
        cr.declare_evaluation(roots)
    with pytest.raises(ValueError, match="is not this campaign's manifest"):
        cr.assert_evaluation_declared(roots, "eval_ii/grid4x4")


def test_evaluate_refuses_before_the_evaluation_is_declared_and_writes_nothing(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    with pytest.raises(FileNotFoundError, match="declare-evaluation"):
        cr.evaluate(roots, shard=0, of=6)
    assert list(Path(roots.output_root).rglob("*")) == []


def test_old_cell_shards_partition_the_registered_cells() -> None:
    shards = [cr.old_cells_shard(k, 6) for k in range(6)]
    flat = [cell for shard in shards for cell in shard]
    assert sorted(flat) == sorted(cr.HZ1X1_OLD_CELLS) and len(flat) == len(set(flat))
    with pytest.raises(ValueError, match="partition"):
        cr.old_cells_shard(0, 0)


def test_the_old_cells_cli_refuses_without_its_declared_manifest_and_rolls_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    roots = _roots(tmp_path)
    calls: list[Any] = []
    monkeypatch.setattr(cr, "roll_old_cells", lambda r, cells=None: calls.append(cells) or {"n_refused": 0})
    code = cr.main(["--output-root", str(roots.output_root), "--corpus-root", str(tmp_path), "--draws-root",
                    str(tmp_path), "old-cells", "--shard", "1", "--of", "6"])
    assert code == 2 and "declare-evaluation" in capsys.readouterr().err
    assert calls == []
    cr.declare_evaluation(roots)
    assert cr.main(["--output-root", str(roots.output_root), "--corpus-root", str(tmp_path), "--draws-root",
                    str(tmp_path), "old-cells", "--shard", "1", "--of", "6"]) == 0
    assert calls == [cr.old_cells_shard(1, 6)]


def test_status_reports_the_declared_old_cells_and_route_ii_sets(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    assert cr.status(roots)["old_cells_hz1x1"]["declared"] is None
    cr.declare_evaluation(roots)
    report = cr.status(roots)
    assert report["old_cells_hz1x1"]["declared"] == len(
        _declared(Path(roots.runs_root) / "old_cells" / "hz1x1" / "campaign_manifest.json"))
    assert report["evaluation"]["route_ii"]["grid4x4"]["declared"] == 500 * len(
        [c for c in cr.evaluation_cells() if c.scenario == "grid4x4"])
    assert report["complete"] is False


def test_status_is_complete_only_when_every_stage_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``complete`` is the conjunction of training, route (i), route (ii) per scenario, P3a and the old cells; each
    component is stubbed complete and then the old cells alone are left incomplete."""
    from offline import att_rederivation as ar

    roots = _roots(tmp_path)
    done = tmp_path / "done.json"
    done.write_text("{}")
    monkeypatch.setattr(cr, "training_status", lambda r: {"complete": True})
    monkeypatch.setattr(cr, "route_i_output", lambda r, cell: done)
    monkeypatch.setattr(cr, "_assert_route_i_complete", lambda payload, cell: None)
    monkeypatch.setattr(cr, "_p3a_path", lambda r, cell: done)
    old_complete = {"value": False}
    monkeypatch.setattr(ar, "campaign_status", lambda work: {
        "complete": old_complete["value"] if Path(work).parent.name == "old_cells" else True})
    assert cr.status(roots)["complete"] is False
    old_complete["value"] = True
    assert cr.status(roots)["complete"] is True
