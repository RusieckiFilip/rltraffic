"""P9.1's sessions (``offline/corrective_round.py``; ``BRIEF_46`` Amendment E, E.4; ``PROJECT_PLAN`` section 7 rule (5)).

No format of its own: runs write the campaign's ``p9.1-run/1.0`` records, route (ii) and the old cells P8.4b's
per-episode files.  The rules under test:
* a training session plans the next missing runs in ``training_runs()`` order whose PLANNED seconds (planning
  estimates, the pre-flight's per-arm figures) fit its hours -- the first always included, at most ``max_runs`` -- prints
  the plan, trains exactly those runs, and closes with a summary;
* it refuses to start while ``output/p9_1_runs/STOP`` exists, checks the file before each run and then ends after the
  run in flight, and never deletes it;
* an evaluation session ends between units of work -- a cell for route (i) and P3a, an episode for route (ii) and the
  old cells -- at its deadline or at the stop file, and the next resumes from disk;
* the evaluate stage refuses while any training run is missing.

The training runs below use the REAL ``train_run`` and the real trainers at two gradient steps on
``tests/test_corrective_round_train.py``'s fixture tier; where a run must see the stop file appear, only the innermost
trainer is wrapped, to create it mid-run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

import offline.corrective_round as cr
import offline.method_tier_grid as grid
from tests.test_corrective_round_train import _cell, _hz1x1_fixture, _roots, _two_steps

#: The pre-flight's 40.5-hour table, per run (docs/returns/P9.1-preflight.md; P9.1_evidence/g2_cost_bc.py.txt).
FIGURES = {
    ("hz1x1", "dt"): 204.0, ("hz1x1", "bc"): 110.0, ("hz1x1", "bc_top10"): 120.0, ("hz1x1", "iql"): 361.0,
    ("hz1x1", "iql_t"): 361.0 * 1.15, ("hz1x1", "dt_100k"): 2.5 * 204.0, ("hz1x1", "bc_100k"): 2.5 * 110.0,
    ("hz1x1", "bc_top10_100k"): 2.5 * 120.0, ("hz1x1", "iql_100k"): 2.5 * 361.0,
    ("grid4x4", "bc_m"): 543.0, ("grid4x4", "iql_m"): 2187.3, ("grid4x4", "bc_top10_m"): 97.0,
    ("grid4x4", "bc_top10_perix_m"): 97.0, ("grid4x4", "iql_t"): 298.0 * 1.15, ("grid4x4", "dt_nomix"): 3487.0,
}
RUNS = cr.training_runs()


def _label(index: int) -> dict[str, Any]:
    cell, seed = RUNS[index - 1]
    return {"run": index, "cell": f"{cell.scenario}:{cell.key}", "seed": int(seed)}


def _printed(text: str, marker: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.startswith(marker + " ")]
    assert len(lines) == 1, (marker, lines)
    return json.loads(lines[0][len(marker) + 1:])


def _done_through(monkeypatch: pytest.MonkeyPatch, last: int) -> None:
    """Runs 1..*last* of training_runs() read as done, every other as missing (no file needed)."""
    done = set(RUNS[:last])
    monkeypatch.setattr(cr, "run_status", lambda roots, cell, seed: "done" if (cell, int(seed)) in done else "missing")


# ----------------------------------------------------------------------
# The plan
# ----------------------------------------------------------------------


def test_the_planned_seconds_are_the_pre_flights_figures_labelled_planning_estimates(tmp_path: Path) -> None:
    assert dict(cr.PLANNED_RUN_SECONDS) == FIGURES
    assert {(cell.scenario, cell.arm) for cell in cr.training_cells()} == set(FIGURES)
    assert round(sum(cr.planned_seconds(cell) for cell, _ in RUNS) / 3600.0, 1) == 40.5
    assert "planning estimates" in cr.PLANNING_NOTE and "not measurements" in cr.PLANNING_NOTE


def test_a_plan_is_the_next_missing_runs_that_fit_its_hours_the_first_always_included(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With runs 1-3 done (E.2) and 6 hours, the plan is BRIEF_46 E.5's first training session: runs 4 to 65."""
    roots = _roots(tmp_path)
    _done_through(monkeypatch, 3)
    plan = cr.session_plan(roots, hours=6.0)
    assert (plan.as_json()["first"], plan.as_json()["last"], plan.as_json()["count"]) == (_label(4), _label(65), 62)
    expected = 0.0
    for cell, _ in RUNS[3:65]:
        expected += FIGURES[(cell.scenario, cell.arm)]
    assert plan.planned_seconds == expected <= 6 * 3600 < expected + FIGURES[(RUNS[65][0].scenario, RUNS[65][0].arm)]
    assert (plan.as_json()["missing"], plan.as_json()["done"]) == (252, 3)
    assert cr.session_plan(roots, hours=0.01).as_json()["runs"] == [_label(4)]
    assert [r["run"] for r in cr.session_plan(roots, hours=6.0, max_runs=3).as_json()["runs"]] == [4, 5, 6]
    assert cr.session_plan(roots).as_json()["count"] == 252
    for bad in ({"hours": 0.0}, {"hours": -1.0}, {"max_runs": 0}):
        with pytest.raises(ValueError, match="a session"):
            cr.session_plan(roots, **bad)


# ----------------------------------------------------------------------
# A training session: the real train_run, the real trainers at two steps
# ----------------------------------------------------------------------


def _session_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[cr.Roots, cr.Cell]:
    roots = _hz1x1_fixture(tmp_path, monkeypatch)
    _two_steps(monkeypatch, "hz1x1", "iql_t")
    return roots, _cell("hz1x1", "maxpressure", "iql_t")


def test_a_session_trains_exactly_its_plan_says_so_and_the_next_resumes_from_disk(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Planned 415.15 s per run: 0.25 hours holds two of the three runs."""
    roots, cell = _session_fixture(tmp_path, monkeypatch)
    seeds = (101, 202, 303)
    assert cr.train(roots, cells=[cell], seeds=seeds, device="cpu", hours=0.25) == {"trained": 2, "skipped_done": 0}
    assert [cr.run_status(roots, cell, s) for s in seeds] == ["done", "done", "missing"]
    out = capsys.readouterr().out
    plan, summary = _printed(out, "SESSION PLAN"), _printed(out, "SESSION SUMMARY")
    assert [r["seed"] for r in plan["runs"]] == [101, 202] and plan["note"] == cr.PLANNING_NOTE
    recorded = [json.loads(cr.run_record_path(roots, "hz1x1", "maxpressure", "iql_t", s).read_text())["seconds"]
                for s in (101, 202)]
    assert [(r["seed"], r["seconds"]) for r in summary["trained"]] == list(zip((101, 202), recorded))
    assert summary["ended_by"] == "the plan" and summary["next_missing"]["seed"] == 303
    assert summary["planned_hours_left"] == FIGURES[("hz1x1", "iql_t")] / 3600.0
    assert cr.train(roots, cells=[cell], seeds=seeds, device="cpu", hours=0.25) == {"trained": 1, "skipped_done": 2}
    assert _printed(capsys.readouterr().out, "SESSION SUMMARY")["next_missing"] is None


def test_max_runs_caps_a_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots, cell = _session_fixture(tmp_path, monkeypatch)
    assert cr.train(roots, cells=[cell], seeds=(101, 202, 303), device="cpu", max_runs=1) == {"trained": 1,
                                                                                            "skipped_done": 0}


def test_no_session_starts_while_the_stop_file_exists_and_the_code_never_deletes_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    roots, cell = _session_fixture(tmp_path, monkeypatch)
    stop = Path(roots.runs_root) / "STOP"
    assert cr.stop_file(roots) == stop
    stop.parent.mkdir(parents=True)
    stop.touch()
    with pytest.raises(ValueError, match="no session starts while it exists"):
        cr.train(roots, cells=[cell], seeds=(101,), device="cpu", hours=1.0)
    assert stop.is_file() and cr.run_status(roots, cell, 101) == "missing"
    code = cr.main(["--output-root", str(roots.output_root), "--corpus-root", str(roots.corpus_root), "--draws-root",
                    str(roots.draws_root), "train", "--hours", "1"])
    assert code == 2 and "no session starts while it exists" in capsys.readouterr().err and stop.is_file()


def test_the_stop_file_ends_the_session_after_the_run_in_flight(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The file appears DURING the second run (the innermost trainer creates it): that run completes and is recorded,
    the third never starts, the summary says why, and the file is still there."""
    roots, cell = _session_fixture(tmp_path, monkeypatch)
    real, seeds_trained = grid.train_iql, []

    def trainer(*args: Any, **kwargs: Any) -> Any:
        seeds_trained.append(int(kwargs["seed"]))
        if len(seeds_trained) == 2:
            cr.stop_file(roots).touch()
        return real(*args, **kwargs)

    monkeypatch.setattr(grid, "train_iql", trainer)
    result = cr.train(roots, cells=[cell], seeds=(101, 202, 303), device="cpu", hours=10.0)
    assert result == {"trained": 2, "skipped_done": 0} and seeds_trained == [101, 202]
    assert [cr.run_status(roots, cell, s) for s in (101, 202, 303)] == ["done", "done", "missing"]
    summary = _printed(capsys.readouterr().out, "SESSION SUMMARY")
    assert summary["ended_by"] == "the stop file" and summary["next_missing"]["seed"] == 303
    assert cr.stop_file(roots).is_file()


def test_the_train_cli_hands_its_hours_and_max_runs_and_plan_only_trains_nothing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    roots = _roots(tmp_path)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cr, "train", lambda roots_, **kw: calls.append(kw) or {"trained": 0, "skipped_done": 0})
    base = ["--output-root", str(roots.output_root), "--corpus-root", str(roots.corpus_root), "--draws-root",
            str(roots.draws_root)]
    assert cr.main([*base, "train", "--hours", "1.5", "--max-runs", "2"]) == 0
    assert (calls[0]["hours"], calls[0]["max_runs"]) == (1.5, 2)
    _done_through(monkeypatch, 3)
    assert cr.main([*base, "train", "--hours", "6", "--plan-only"]) == 0 and len(calls) == 1
    assert _printed(capsys.readouterr().out, "SESSION PLAN")["last"] == _label(65)


# ----------------------------------------------------------------------
# The evaluate stage and an evaluation session
# ----------------------------------------------------------------------


def test_the_evaluate_stage_refuses_while_any_training_run_is_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="255 of 255 training runs are missing"):
        cr.check(_roots(tmp_path), stage="evaluate")


@pytest.mark.parametrize("stage", ["train", "evaluate"])
def test_no_train_or_evaluate_session_passes_its_check_while_the_stop_file_exists(stage: str, tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    cr.stop_file(roots).parent.mkdir(parents=True)
    cr.stop_file(roots).touch()
    with pytest.raises(ValueError, match="no session starts while it exists"):
        cr.check(roots, stage=stage)
    assert cr.stop_file(roots).is_file()


def _evaluation_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[cr.Roots, list[str]]:
    roots = _roots(tmp_path)
    Path(roots.runs_root).mkdir(parents=True)
    monkeypatch.setattr(cr, "assert_evaluation_declared", lambda roots_, *names: None)
    monkeypatch.setattr(cr, "run_status", lambda roots_, cell, seed: "done")
    seen: list[str] = []
    return roots, seen


def test_an_evaluation_session_ends_between_cells_at_the_stop_file_or_its_deadline(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots, seen = _evaluation_fixture(tmp_path, monkeypatch)
    keys = [c.key for c in cr.shard_of([c for c in cr.evaluation_cells() if c.scenario == "hz1x1"], shard=0, of=1)]

    planted: list[bool] = []

    def cell_unit(roots_: Any, cell: cr.Cell, *, deadline_unix: float | None = None) -> dict[str, Any]:
        seen.append(cell.key)
        if len(seen) == 2 and not planted:  # the author creates the file once, during the second cell
            planted.append(True)
            cr.stop_file(roots).touch()
        return {"cell": cell.key, "stopped_by": None}

    monkeypatch.setattr(cr, "evaluate_cell", cell_unit)
    result = cr.evaluate(roots, scenario="hz1x1")
    assert seen == keys[:2] and result["stopped_by"] == "the stop file"
    assert result["cells_left"] == [f"hz1x1:{k}" for k in keys[2:]] and cr.stop_file(roots).is_file()
    cr.stop_file(roots).unlink()  # the author removes it
    seen.clear()
    result = cr.evaluate(roots, scenario="hz1x1", deadline_unix=time.time() - 1.0)
    assert seen == [] and result["stopped_by"] == "the hours bound" and len(result["cells_left"]) == len(keys)
    result = cr.evaluate(roots, scenario="hz1x1", deadline_unix=time.time() + 3600.0)
    assert seen == keys and result["stopped_by"] is None and result["cells_left"] == []


def test_a_cell_stopped_inside_route_ii_records_no_p3a_and_says_why(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """Route (ii) is handed the session's stop question and ends between episodes; P3a, a whole-cell unit, waits."""
    roots, _ = _evaluation_fixture(tmp_path, monkeypatch)
    cell = next(c for c in cr.p3a_cells() if (c.scenario, c.key) == ("hz1x1", "dt@random_fresh"))
    monkeypatch.setattr(cr, "route_i", lambda roots_, cell_: {"episodes": []})
    monkeypatch.setattr(cr, "committed_from_route_i", lambda payload, s, k: {("hz1x1", k, 101, 1000): 1.0})
    asked: list[Any] = []

    def route_ii(roots_: Any, scenario: str, keys: Any, committed: Any, *, deadline_unix: Any = None) -> dict[str, Any]:
        asked.append(deadline_unix)
        cr.stop_file(roots).touch()  # the author's file appears while route (ii) rolls
        return {"rolled": 0, "skipped_already_present": 0, "refused": [], "n_refused": 0, "seconds": 0.0,
                "stopped": True, "not_reached": 1}

    p3a: list[Any] = []
    monkeypatch.setattr(cr, "run_route_ii", route_ii)
    monkeypatch.setattr(cr, "record_p3a", lambda *a, **k: p3a.append(a) or [])
    deadline = time.time() + 3600.0
    outcome = cr.evaluate_cell(roots, cell, deadline_unix=deadline)
    assert outcome["stopped_by"] == "the stop file" and p3a == [] and asked == [deadline]


def test_route_ii_asks_the_session_before_every_episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real ``run_route_ii`` hands P8.4b's runner the session's question: the stop file always, the deadline when
    one is given."""
    roots, _ = _evaluation_fixture(tmp_path, monkeypatch)
    import offline.att_rederivation as ar

    handed: list[Any] = []
    monkeypatch.setattr(ar, "run_campaign", lambda keys, **kw: handed.append(kw["should_stop"]) or {"n_refused": 0})
    cr.run_route_ii(roots, "hz1x1", [], {})
    cr.run_route_ii(roots, "hz1x1", [], {}, deadline_unix=time.time() - 1.0)
    assert (handed[0](), handed[1]()) == (False, True)
    cr.stop_file(roots).touch()
    assert handed[0]() is True


def test_the_old_cells_and_p3a_references_end_between_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots, _ = _evaluation_fixture(tmp_path, monkeypatch)
    import offline.att_rederivation as ar

    handed: list[Any] = []

    def campaign(keys: Any, **kw: Any) -> dict[str, Any]:
        handed.append(kw.get("should_stop"))
        return {"rolled": 0, "skipped_already_present": 0, "refused": [], "n_refused": 0, "seconds": 0.0,
                "stopped": True, "not_reached": len(keys)}

    monkeypatch.setattr(ar, "run_campaign", campaign)
    monkeypatch.setattr(cr, "old_cell_committed", lambda repo, artifact, arm: {("hz1x1", arm, 101, 1000): 1.0})
    past = time.time() - 1.0
    outcome = cr.roll_old_cells(roots, [("docs/data/p4_6_grid.json", "iql@maxpressure")], deadline_unix=past)
    assert outcome["stopped"] is True and handed[0] is not None and handed[0]() is True
    recorded: list[Any] = []
    monkeypatch.setattr(cr, "record_p3a", lambda *a, **k: recorded.append(a) or [])
    result = cr.record_registered_references(roots, deadline_unix=past)
    assert recorded == [] and result["stopped_by"] == "the hours bound"


def test_the_rolling_clis_hand_their_deadline_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _roots(tmp_path)
    got: dict[str, Any] = {}
    monkeypatch.setattr(cr, "evaluate", lambda roots_, **kw: got.update(evaluate=kw) or {})
    monkeypatch.setattr(cr, "assert_evaluation_declared", lambda roots_, *names: None)
    monkeypatch.setattr(cr, "roll_old_cells", lambda roots_, cells=None, **kw: got.update(old=kw) or {})
    monkeypatch.setattr(cr, "record_registered_references", lambda roots_, **kw: got.update(p3a=kw) or {})
    base = ["--output-root", str(roots.output_root), "--corpus-root", str(roots.corpus_root), "--draws-root",
            str(roots.draws_root)]
    for command in (["evaluate", "--shard", "1", "--of", "6"], ["old-cells", "--shard", "1", "--of", "6"],
                    ["p3a-references"]):
        assert cr.main([*base, *command, "--deadline-unix", "123.5"]) == 0
    assert got["evaluate"]["deadline_unix"] == 123.5 and got["old"] == {"deadline_unix": 123.5}
    assert got["p3a"] == {"deadline_unix": 123.5}


def test_a_stop_file_that_appears_while_the_session_plans_starts_no_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The file appears AFTER the start's check and BEFORE the first run (while the plan reads every run's status from
    disk): the session starts no run, says so, and leaves the file."""
    roots, cell = _session_fixture(tmp_path, monkeypatch)
    real = cr.run_status

    def status(roots_: Any, cell_: cr.Cell, seed: int) -> str:
        if int(seed) == 303:
            Path(roots.runs_root).mkdir(parents=True, exist_ok=True)
            cr.stop_file(roots).touch()
        return real(roots_, cell_, seed)

    monkeypatch.setattr(cr, "run_status", status)
    assert cr.train(roots, cells=[cell], seeds=(101, 202, 303), device="cpu", hours=10.0) == {"trained": 0,
                                                                                            "skipped_done": 0}
    monkeypatch.setattr(cr, "run_status", real)
    assert [cr.run_status(roots, cell, s) for s in (101, 202, 303)] == ["missing"] * 3
    summary = _printed(capsys.readouterr().out, "SESSION SUMMARY")
    assert summary["ended_by"] == "the stop file" and summary["trained"] == [] and cr.stop_file(roots).is_file()
