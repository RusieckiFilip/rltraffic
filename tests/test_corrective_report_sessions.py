"""P9.1's report names each run's code commit by its session (``BRIEF_46`` Amendment G, G.2).

A checkpoint's ``runtime.git_commit`` is the main tree's HEAD when it was written (the driver runs from the main tree),
so a run's code commit comes from its session: the session whose window -- from its token line's stamp to the next
session's, of ANY stage -- holds the run's ``started_unix``.  That token line (``offline/campaigns/p9_1_corrective.sh``:
``token consumed; stage <stage> <stamp> from <run tree> at <commit>; logs <dir>``) names the commit and the run tree the
driver checked are clean at it.  No format of its own: the captures are written here as the driver's ``tee`` writes
them.  Expected values are written by hand; every Unix time is computed here by ``datetime`` in UTC, another route than
the report's.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import offline.corrective_report as rep
import offline.corrective_round as cr


def _roots(tmp_path: Path) -> cr.Roots:
    output = tmp_path / "output"
    (output / cr.RUNS_DIR).mkdir(parents=True)
    return cr.Roots(repo_root=Path(__file__).resolve().parents[1], output_root=output, corpus_root=tmp_path / "corpus",
                    draws_root=tmp_path / "draws")


def _unix(stamp: str) -> float:
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()


def _token(stage: str, stamp: str, tree: str, commit: str) -> str:
    return f"token consumed; stage {stage} {stamp} from {tree} at {commit}; logs /trees/output/p9_1_runs/logs/{stamp}"


#: The real campaign's sessions in shape: collect, the campaign stage (runs 1-3), d4rl, two training sessions, and an
#: evaluation, each capture holding the driver's other lines around its token lines.
CAPTURES: dict[str, list[str]] = {
    "collect_capture.txt": [_token("collect", "20261016T140119Z", "/trees/run", "140149e"), "collect_hz1x1: ok",
                            "stage collect 20261016T140119Z: COMPLETE -- next: plan Q2's typed commit"],
    "campaign_capture.txt": [_token("campaign", "20261028T151149Z", "/trees/run-c", "cb61ee7"),
                             "TRAIN hz1x1:iql_t@random seed 101 -> /trees/output/p9_1_runs/checkpoints/x.pt"],
    "d4rl_capture.txt": [_token("d4rl", "20261028T192029Z", "/trees/run-c", "cb61ee7"), "d4rl: ok",
                         "stage d4rl 20261028T192029Z: COMPLETE (outputs /trees/d4rl)"],
    "train_capture.txt": [_token("train", "20261029T154931Z", "/trees/run-d", "c58afb8"),
                          "SESSION: runs #4 (hz1x1:iql_t@random seed 404) to #65 (...): 62 runs, 5.95 planned hours",
                          "stage train 20261029T154931Z: SESSION COMPLETE -- the next session resumes from disk",
                          _token("train", "20261029T220237Z", "/trees/run-d", "c58afb8"),
                          "stage train 20261029T220237Z: SESSION COMPLETE -- the next session resumes from disk"],
    "evaluate_capture.txt": [_token("evaluate", "20261105T101034Z", "/trees/run-d", "c58afb8"),
                             "declare_evaluation: ok"],
}


def _write_captures(roots: cr.Roots, captures: dict[str, list[str]]) -> None:
    for name, lines in captures.items():
        (Path(roots.runs_root) / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _session(stage: str, stamp: str, tree: str, commit: str, capture: str, line: int) -> dict[str, Any]:
    return {"stage": stage, "stamp": stamp, "start_unix": _unix(stamp), "run_tree": tree, "code_commit": commit,
            "capture": f"output/p9_1_runs/{capture}", "line": line}


def test_every_token_line_of_every_capture_is_one_session_in_start_order(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_captures(roots, CAPTURES)
    assert _unix("20261028T151149Z") == 1_793_200_309.0  # the helper's own anchor: run 1 started 6 s after it
    assert rep.driver_sessions(roots) == [
        _session("collect", "20261016T140119Z", "/trees/run", "140149e", "collect_capture.txt", 1),
        _session("campaign", "20261028T151149Z", "/trees/run-c", "cb61ee7", "campaign_capture.txt", 1),
        _session("d4rl", "20261028T192029Z", "/trees/run-c", "cb61ee7", "d4rl_capture.txt", 1),
        _session("train", "20261029T154931Z", "/trees/run-d", "c58afb8", "train_capture.txt", 1),
        _session("train", "20261029T220237Z", "/trees/run-d", "c58afb8", "train_capture.txt", 4),
        _session("evaluate", "20261105T101034Z", "/trees/run-d", "c58afb8", "evaluate_capture.txt", 1),
    ]


def _attribution(commit: str, tree: str, session: str, capture: str, line: int) -> dict[str, str]:
    return {"code_commit": commit, "run_tree": tree, "session": session,
            "code_commit_source": f"output/p9_1_runs/{capture} line {line}: the session's token line (BRIEF_46 G.2)"}


@pytest.mark.parametrize(("started", "seconds", "expected"), [
    # run 1 of the campaign stage, 6 s after its token
    (_unix("20261028T151149Z") + 6.153, 343.9, _attribution("cb61ee7", "/trees/run-c", "campaign 20261028T151149Z",
                                                            "campaign_capture.txt", 1)),
    # T1's first run, and a run that starts in the very second of T2's stamp
    (_unix("20261029T154931Z") + 8.0, 330.0, _attribution("c58afb8", "/trees/run-d", "train 20261029T154931Z",
                                                          "train_capture.txt", 1)),
    (_unix("20261029T220237Z"), 330.0, _attribution("c58afb8", "/trees/run-d", "train 20261029T220237Z",
                                                     "train_capture.txt", 4)),
    # the last second of T2's window: it ends where the evaluation starts
    (_unix("20261105T101034Z") - 400.0, 399.0, _attribution("c58afb8", "/trees/run-d", "train 20261029T220237Z",
                                                            "train_capture.txt", 4)),
])
def test_a_run_takes_the_commit_and_tree_of_the_session_whose_window_holds_its_start(
        started: float, seconds: float, expected: dict[str, str], tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_captures(roots, CAPTURES)
    run = {"cell": "p9_1:hz1x1:iql_t@random", "seed": 101, "started_unix": started, "finished_unix": started + seconds}
    assert rep.run_session(run, rep.driver_sessions(roots)) == expected


@pytest.mark.parametrize(("started", "message"), [
    (_unix("20261016T140119Z") - 1.0, "started before every session"),
    (_unix("20261016T140119Z") + 60.0, "inside the collect session 20261016T140119Z, which trains nothing"),
    (_unix("20261028T192029Z") + 60.0, "inside the d4rl session 20261028T192029Z, which trains nothing"),
    (_unix("20261105T101034Z") + 60.0, "inside the evaluate session 20261105T101034Z, which trains nothing"),
])
def test_a_run_that_started_outside_every_training_session_is_refused(started: float, message: str,
                                                                      tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_captures(roots, CAPTURES)
    run = {"cell": "p9_1:hz1x1:iql_t@random", "seed": 101, "started_unix": started, "finished_unix": started + 30.0}
    with pytest.raises(ValueError, match=message):
        rep.run_session(run, rep.driver_sessions(roots))


def test_a_run_that_outlives_its_session_into_the_next_is_refused(tmp_path: Path) -> None:
    """Two P9.1 stages never run at once (the driver's pgrep refusal), so a run that ends after the next session began
    is not a run of the session it started in."""
    roots = _roots(tmp_path)
    _write_captures(roots, CAPTURES)
    started = _unix("20261028T192029Z") - 100.0
    run = {"cell": "p9_1:hz1x1:iql_t@random", "seed": 404, "started_unix": started, "finished_unix": started + 101.0}
    with pytest.raises(ValueError, match="finished after the next session, d4rl 20261028T192029Z, started"):
        rep.run_session(run, rep.driver_sessions(roots))


@pytest.mark.parametrize("line", [
    "token consumed; stage train 20261029T154931Z from /trees/run-d at HEAD; logs /trees/logs",
    "token consumed; stage train 20261029T154931Z from /trees/run-d at c58afb8",
    "token consumed; stage train 2026-10-29T15:49:31Z from /trees/run-d at c58afb8; logs /trees/logs",
])
def test_a_token_line_the_report_cannot_read_is_refused(line: str, tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_captures(roots, {**CAPTURES, "train_capture.txt": ["SESSION: earlier output", line]})
    with pytest.raises(ValueError, match="train_capture.txt line 2 is a token line the report cannot read"):
        rep.driver_sessions(roots)


def test_two_token_lines_with_one_stamp_are_refused(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    twice = CAPTURES["train_capture.txt"] + [_token("train", "20261029T154931Z", "/trees/run-d", "c58afb8")]
    _write_captures(roots, {**CAPTURES, "train_capture.txt": twice})
    with pytest.raises(ValueError, match="two token lines carry the stamp 20261029T154931Z"):
        rep.driver_sessions(roots)


# ----------------------------------------------------------------------
# Q20 (BRIEF_46 Amendment S, S.2): the evaluation's code commit, and the D4RL session, by the same token lines
# ----------------------------------------------------------------------

#: The real evaluate capture's shape: three sessions, their token lines at lines 1, 7 and 14.
EVALUATE = [_token("evaluate", "20261105T101034Z", "/trees/run-d", "c58afb8"), *["E1 output"] * 5,
            _token("evaluate", "20261105T200318Z", "/trees/run-d", "c58afb8"), *["E2 output"] * 6,
            _token("evaluate", "20261106T152511Z", "/trees/run-d", "c58afb8"), "E3 output"]


def test_the_evaluation_names_the_one_commit_and_run_tree_of_every_evaluate_session(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_captures(roots, {**CAPTURES, "evaluate_capture.txt": EVALUATE})
    assert rep.stage_code_commit(rep.driver_sessions(roots), "evaluate") == {
        "stage": "evaluate", "code_commit": "c58afb8", "run_tree": "/trees/run-d",
        "sessions": ["evaluate 20261105T101034Z", "evaluate 20261105T200318Z", "evaluate 20261106T152511Z"],
        "token_lines": ["output/p9_1_runs/evaluate_capture.txt line 1", "output/p9_1_runs/evaluate_capture.txt line 7",
                        "output/p9_1_runs/evaluate_capture.txt line 14"]}


def test_the_d4rl_session_is_named_by_its_token_line(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_captures(roots, CAPTURES)
    assert rep.stage_code_commit(rep.driver_sessions(roots), "d4rl") == {
        "stage": "d4rl", "code_commit": "cb61ee7", "run_tree": "/trees/run-c", "sessions": ["d4rl 20261028T192029Z"],
        "token_lines": ["output/p9_1_runs/d4rl_capture.txt line 1"]}


@pytest.mark.parametrize(("commit", "tree"), [("0a1b2c3", "/trees/run-d"), ("c58afb8", "/trees/run-e")])
def test_evaluate_sessions_that_name_two_commits_or_two_run_trees_are_refused(commit: str, tree: str,
                                                                              tmp_path: Path) -> None:
    """A per-output attribution would then be needed, which the report does not attempt (S.2)."""
    roots = _roots(tmp_path)
    _write_captures(roots, {**CAPTURES, "evaluate_capture.txt": EVALUATE[:-2] + [
        _token("evaluate", "20261106T152511Z", tree, commit), "E3 output"]})
    with pytest.raises(ValueError, match="the evaluate sessions name the commits .* from the run trees .*: one commit "
                                         "and one run tree"):
        rep.stage_code_commit(rep.driver_sessions(roots), "evaluate")


def test_a_stage_with_no_session_in_the_captures_is_refused(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_captures(roots, {name: lines for name, lines in CAPTURES.items() if name != "evaluate_capture.txt"})
    with pytest.raises(ValueError, match="no evaluate session in the captures"):
        rep.stage_code_commit(rep.driver_sessions(roots), "evaluate")
