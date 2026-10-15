"""P9.1's report, part 1: the per-episode records every artifact carries (``offline/corrective_report.py``; ``BRIEF_46``
3.5, A27(a)).

Format ``p9.1-report/1.0``.  Each record: ``arm``, ``tier``, ``seed``, ``draw_id``, ``att_ours``, ``att_engine``,
``entered``, ``created``, ``never_entered``, ``entered_fraction``, ``episode_reward``, ``horizon_vehicle_count``,
``source``.  Alignment: one record per held-out ``(seed, draw)`` of the cell, in ``(seed, draw)`` order; ``att_ours``,
``horizon_vehicle_count`` and ``episode_reward`` are route (i)'s (the registered evaluation's) and ``att_engine`` with the
companions route (ii)'s (P8.4b's per-second observer), joined on ``(seed, draw)`` and refused unless route (ii)'s
``att_ours`` equals route (i)'s under ``==``.  An OLD hz1x1 cell's ``att_ours`` is its committed artifact's, joined to
its re-roll the same way.  Every expected value below is written by the test itself, never read back through the
function under test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import offline.corrective_report as rep
import offline.corrective_round as cr
from offline import att_rederivation as ar
from offline import dt_gate

SEEDS = (101, 202)
DRAWS = (1000, 1001, 1002)


@pytest.fixture
def small_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dt_gate, "TRAINING_SEEDS", SEEDS)
    monkeypatch.setattr(dt_gate, "HELD_OUT_DRAWS", DRAWS)


def _roots(tmp_path: Path) -> cr.Roots:
    output = tmp_path / "output"
    output.mkdir()
    return cr.Roots(repo_root=tmp_path / "repo", output_root=output, corpus_root=tmp_path / "corpus",
                    draws_root=tmp_path / "draws")


def _att(seed: int, draw: int) -> float:
    return 100.0 + seed / 1000.0 + draw / 7.0


def _route_ii_row(scenario: str, arm: str, seed: int | None, draw: int, att_ours: float) -> dict[str, Any]:
    method, tier = ar.normalise_arm(scenario, arm)
    return {"format_version": "p8.4b-rederivation/1.0", "scenario": scenario, "tier": tier, "method": method,
            "arm": arm, "seed": seed, "draw_id": draw, "created": 900 + draw % 3, "entered": 880 + draw % 3,
            "never_entered": 20, "entered_fraction": (880 + draw % 3) / (900 + draw % 3), "completed_at_horizon": 700,
            "running_at_horizon": 150, "waiting_at_horizon": 30, "att_ours": att_ours,
            "att_engine": att_ours + 0.5, "horizon_vehicle_count": 999.0, "episode_reward": -999.0, "seconds": 1.0,
            "seconds_rollout": 0.9, "committed_att_ours": att_ours, "reproduces_committed": True,
            "policy_source": {"kind": "test"}}


def _write_new_cell(roots: cr.Roots, cell: cr.Cell, *, shift: dict[tuple[int, int], float] | None = None,
                    skip_ii: tuple[int, int] | None = None) -> None:
    episodes = [{"arm": cell.key, "seed": s, "draw_id": d, "att_horizon": _att(s, d),
                 "horizon_vehicle_count": 10.0 * s + d, "episode_reward": -float(s + d)}
                for s in SEEDS for d in DRAWS]
    path = cr.route_i_output(roots, cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"arm": cell.key, "episodes": episodes}))
    work = Path(roots.runs_root) / "eval_ii" / cell.scenario
    work.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        for d in DRAWS:
            if (s, d) == skip_ii:
                continue
            value = _att(s, d) + (shift or {}).get((s, d), 0.0)
            name = ar.cell_file_name(ar.CellKey(cell.scenario, cell.key, s, d))
            (work / name).write_text(json.dumps(_route_ii_row(cell.scenario, cell.key, s, d, value)))


def _cell(scenario: str, tier: str, arm: str) -> cr.Cell:
    return next(c for c in cr.evaluation_cells() if (c.scenario, c.tier, c.arm) == (scenario, tier, arm))


def test_a_new_cells_records_join_route_i_and_route_ii_on_seed_and_draw(tmp_path: Path, small_grid: None) -> None:
    roots = _roots(tmp_path)
    cell = _cell("grid4x4", "mappo1000", "bc_m")
    _write_new_cell(roots, cell)
    rows = rep.new_cell_rows(roots, cell)
    assert [tuple(sorted(row)) for row in rows] == [tuple(sorted(rep.EPISODE_FIELDS))] * len(SEEDS) * len(DRAWS)
    assert [(row["seed"], row["draw_id"]) for row in rows] == [(s, d) for s in SEEDS for d in DRAWS]
    for row in rows:
        s, d = row["seed"], row["draw_id"]
        assert row["arm"] == "bc_m@mappo1000" and row["tier"] == "mappo1000"
        assert row["att_ours"] == _att(s, d) and row["att_engine"] == _att(s, d) + 0.5
        assert row["horizon_vehicle_count"] == 10.0 * s + d and row["episode_reward"] == -float(s + d)
        assert (row["created"], row["entered"], row["never_entered"]) == (900 + d % 3, 880 + d % 3, 20)
        assert row["source"] == ("new: route (i) output/p9_1_runs/eval_i/grid4x4/eval_mappo1000_bc_m.json; route (ii) "
                                 "output/p9_1_runs/eval_ii/grid4x4, att_ours equal under ==")


def test_a_route_ii_value_that_is_not_route_is_value_refuses_the_cell(tmp_path: Path, small_grid: None) -> None:
    roots = _roots(tmp_path)
    cell = _cell("grid4x4", "mappo1000", "bc_m")
    _write_new_cell(roots, cell, shift={(202, 1001): 1e-12})
    with pytest.raises(ValueError, match="routes disagree"):
        rep.new_cell_rows(roots, cell)


def test_a_missing_route_ii_episode_refuses_the_cell(tmp_path: Path, small_grid: None) -> None:
    roots = _roots(tmp_path)
    cell = _cell("hz1x1", "maxpressure", "iql_t")
    _write_new_cell(roots, cell, skip_ii=(101, 1002))
    with pytest.raises(ValueError, match="route \\(ii\\) has no episode"):
        rep.new_cell_rows(roots, cell)


def test_an_incomplete_route_i_cell_is_refused(tmp_path: Path, small_grid: None) -> None:
    roots = _roots(tmp_path)
    cell = _cell("hz1x1", "maxpressure", "iql_t")
    _write_new_cell(roots, cell)
    path = cr.route_i_output(roots, cell)
    payload = json.loads(path.read_text())
    payload["episodes"] = payload["episodes"][1:]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not the complete"):
        rep.new_cell_rows(roots, cell)


def _write_old_artifact(repo: Path, artifact: str, arm: str, *, seeds: tuple[Any, ...] = SEEDS) -> None:
    path = repo / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = [{"arm": arm, "seed": s, "draw_id": d, "att_horizon": _att(s or 0, d),
                 "horizon_vehicle_count": 5.0 + d, "episode_reward": -1.0 * d} for s in seeds for d in DRAWS]
    episodes.append({"arm": "someone_else", "seed": 101, "draw_id": 1000, "att_horizon": 1.0,
                     "horizon_vehicle_count": 1.0, "episode_reward": 1.0})
    path.write_text(json.dumps({"episodes": episodes}))


def _write_old_rerolls(roots: cr.Roots, arm: str, *, seeds: tuple[Any, ...] = SEEDS,
                       shift: dict[tuple[Any, int], float] | None = None) -> None:
    work = Path(roots.runs_root) / "old_cells" / "hz1x1"
    work.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        for d in DRAWS:
            value = _att(s or 0, d) + (shift or {}).get((s, d), 0.0)
            name = ar.cell_file_name(ar.CellKey("hz1x1", arm, s, d))
            (work / name).write_text(json.dumps(_route_ii_row("hz1x1", arm, s, d, value)))


def test_an_old_hz1x1_cells_records_carry_the_committed_att_ours_and_the_reroll_beside(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_old_artifact(Path(roots.repo_root), "docs/data/p4_6_grid.json", "iql@maxpressure")
    _write_old_rerolls(roots, "iql@maxpressure")
    rows = rep.old_hz1x1_rows(roots, "docs/data/p4_6_grid.json", "iql@maxpressure")
    assert [(row["seed"], row["draw_id"]) for row in rows] == [(s, d) for s in SEEDS for d in DRAWS]
    for row in rows:
        s, d = row["seed"], row["draw_id"]
        assert row["tier"] == "maxpressure" and row["arm"] == "iql@maxpressure"
        assert row["att_ours"] == _att(s, d) and row["att_engine"] == _att(s, d) + 0.5
        assert row["horizon_vehicle_count"] == 5.0 + d and row["episode_reward"] == -1.0 * d
        assert row["source"] == ("docs/data/p4_6_grid.json; att_engine and companions re-rolled, reproduces "
                                 "committed att_ours")


def test_an_old_cell_whose_reroll_does_not_reproduce_or_is_missing_is_refused(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_old_artifact(Path(roots.repo_root), "docs/data/p4_4_baselines.json", "iql")
    _write_old_rerolls(roots, "iql", shift={(202, 1000): 0.25})
    with pytest.raises(ValueError, match="does not reproduce"):
        rep.old_hz1x1_rows(roots, "docs/data/p4_4_baselines.json", "iql")
    (Path(roots.runs_root) / "old_cells" / "hz1x1" / ar.cell_file_name(ar.CellKey("hz1x1", "iql", 202, 1000))).unlink()
    with pytest.raises(ValueError, match="not re-rolled"):
        rep.old_hz1x1_rows(roots, "docs/data/p4_4_baselines.json", "iql")


def test_p4_4s_arms_take_p4_4s_tier() -> None:
    assert rep.old_cell_tier("madt") == "mappo1000" and rep.old_cell_tier("iql") == "mappo1000"
    assert rep.old_cell_tier("bc_top10@mix50") == "mix50"


def test_a_statement_cell_is_p5_2cs_cell_type_with_the_rows_in_seed_draw_order(tmp_path: Path, small_grid: None) -> None:
    import offline.tier_statements as tst

    roots = _roots(tmp_path)
    cell = _cell("grid4x4", "mappo1000", "bc_m")
    _write_new_cell(roots, cell)
    built = rep.statement_cell(roots, cell)
    assert isinstance(built, tst.Cell)
    assert (built.spec.cell_id, built.spec.method, built.spec.tier, built.spec.arm, built.spec.source) == (
        "p9_1:bc_m@mappo1000", "bc_m", "mappo1000", "bc_m@mappo1000", "p9_1")
    engine = tst.episodes_of(built, "att_engine")
    assert [(e.seed, e.draw_id, e.att_horizon) for e in engine] == [(s, d, _att(s, d) + 0.5) for s in SEEDS for d in DRAWS]


def test_new_training_runs_carry_seconds_and_steps_with_the_uncontrolled_clock_label(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    cell = next(c for c in cr.training_cells() if (c.scenario, c.tier, c.arm) == ("grid4x4", "random", "iql_m"))
    for seed, seconds in ((101, 1500.25), (202, 1499.5)):
        path = cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"seed": seed, "seconds": seconds, "gradient_steps": 40000,
                                    "declared_steps": 40000, "clock": cr.CLOCK_LABEL}))
    runs = rep.new_training_runs(roots, cell, seeds=(101, 202))
    assert runs == [
        {"scenario": "grid4x4", "tier": "random", "arm": "iql_m", "seed": 101, "seconds": 1500.25,
         "gradient_steps": 40000, "clock": "uncontrolled; no Best Performance regime, no canary; not a benchmark",
         "source": "output/p9_1_runs/training/grid4x4/random__iql_m__seed101.json"},
        {"scenario": "grid4x4", "tier": "random", "arm": "iql_m", "seed": 202, "seconds": 1499.5,
         "gradient_steps": 40000, "clock": "uncontrolled; no Best Performance regime, no canary; not a benchmark",
         "source": "output/p9_1_runs/training/grid4x4/random__iql_m__seed202.json"},
    ]
