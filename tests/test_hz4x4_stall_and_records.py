"""Tests for ``BRIEF_48`` Amendment B: B.5.1's committed ``collect verify`` record and B.3's stall remedy.

Written before the code they test (red first).

* **B.5.1** -- ``collect verify --out`` writes one record of every run it verified, so the verification is a committed
  artifact rather than a terminal's scrollback.
* **B.3** -- every process that rolls CityFlow episodes for the gate (G3) and for Phase B's evaluation (G3b) is cut by
  ``faulthandler.dump_traceback_later(timeout, exit=True)`` re-armed per episode, and its driver restarts it from the
  records on disk, at most 20 times, each restart logged; any other non-zero exit stops.  The four registered cases:
  a blocked fake episode is cut and the run completes on restart; a raising one is not restarted; a healthy run longer
  than the timeout in total is never cut; the cap stops a run that hangs every time.

The fake episodes run in real child processes (``sys.executable``), with a timeout of one or two seconds in place of
the registered 120 (E.3.2), so the faulthandler cut is the real one.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import offline.hz4x4_ladder as hz

REPO = Path(__file__).resolve().parents[1]


# ======================================================================
# B.5.1: the collect verify record
# ======================================================================


def _verified(name: str, episodes: int, git_hash: str = "5918dadc", checkpoint: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"name": name, "episodes": episodes, "flow_digests_checked": episodes, "git_hash": git_hash}
    if checkpoint is not None:
        record["checkpoint_sha256"] = checkpoint
    return record


def test_the_verify_record_carries_every_run_and_the_totals() -> None:
    records = [_verified("cf_hz4x4__maxpressure", 200)] + [
        _verified(f"cf_hz4x4__mappo1000__seed{s}", 40, checkpoint=f"{s:064d}") for s in hz.SEEDS
    ]
    artifact = hz.corpus_verify_artifact(records, tiers=["maxpressure", "mappo1000"])
    assert artifact["format_version"] == hz.ARTIFACT_FORMAT_CORPUS_VERIFY
    assert artifact["runs"] == records
    assert (artifact["n_runs"], artifact["n_episodes"], artifact["n_flow_digests_checked"]) == (6, 400, 400)
    assert artifact["tiers"] == ["maxpressure", "mappo1000"]
    assert artifact["collected_by_git_hashes"] == ["5918dadc"]
    assert artifact["all_verified"] is True


def test_the_verify_record_refuses_nothing_verified_and_names_every_code_tree() -> None:
    with pytest.raises(ValueError, match="no run"):
        hz.corpus_verify_artifact([], tiers=["maxpressure"])
    mixed = [_verified("cf_hz4x4__maxpressure", 200), _verified("cf_hz4x4__mappo1000__seed101", 40, git_hash="0abc")]
    assert hz.corpus_verify_artifact(mixed, tiers=["maxpressure", "mappo1000"])["collected_by_git_hashes"] == [
        "0abc", "5918dadc"]


def test_collect_verify_takes_an_out_path() -> None:
    namespace = hz.build_parser().parse_args(["collect", "verify", "--sweep", "s.json", "--out", "v.json"])
    assert namespace.out == "v.json"
    assert hz.build_parser().parse_args(["collect", "plan", "--sweep", "s.json"]).out is None


# ======================================================================
# B.3: the stall remedy -- a per-episode watchdog and a restarting driver
# ======================================================================

_CHILD = textwrap.dedent(
    """
    import json, sys, time
    from pathlib import Path
    import offline.hz4x4_ladder as hz

    work, mode = Path(sys.argv[1]), sys.argv[2]
    timeout, episodes, seconds = float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
    for index in range(episodes):
        record = work / f"episode_{index}.json"
        if record.exists():
            continue
        with hz.episode_watchdog(timeout):
            if mode == "hang_once" and index == 1 and not (work / "hung").exists():
                (work / "hung").write_text("1")
                time.sleep(120)
            if mode == "hang_always" and index == 0:
                time.sleep(120)
            if mode == "raise" and index == 1:
                raise RuntimeError("an ordinary failure, not a stall")
            time.sleep(seconds)
        record.write_text(json.dumps({"index": index}))
    """
)


def _child(tmp_path: Path, mode: str, *, timeout: float, episodes: int, seconds: float) -> tuple[list[str], Path, dict]:
    script = tmp_path / "fake_episodes.py"
    script.write_text(_CHILD, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS="1")
    command = [sys.executable, str(script), str(work), mode, str(timeout), str(episodes), str(seconds)]
    return command, work, env


def test_the_registered_watchdog_is_120_seconds_and_60_restarts() -> None:
    # BRIEF_48 Amendment E, E.3.2 corrected B.3's values (600 s and 20 were set without computing the expected hangs).
    assert hz.WATCHDOG_SECONDS == 120
    assert hz.WATCHDOG_MAX_RESTARTS == 60


def test_watchdog_fired_reads_faulthandlers_timeout_header() -> None:
    assert hz.watchdog_fired("Timeout (0:10:00)!\nThread 0x00007f (most recent call first):\n") is True
    assert hz.watchdog_fired("loading\nTimeout (0:00:01.500000)!\n") is True
    assert hz.watchdog_fired("Traceback (most recent call last):\nRuntimeError: boom\n") is False
    assert hz.watchdog_fired("a line saying Timeout (0:10:00)! in passing\n") is False


def test_a_blocked_episode_is_cut_and_the_run_completes_on_restart(tmp_path: Path) -> None:
    command, work, env = _child(tmp_path, "hang_once", timeout=1.0, episodes=3, seconds=0.05)
    result = hz.run_with_restarts(command, log_dir=tmp_path / "logs", label="fake", max_restarts=3, env=env)
    assert result["status"] == "completed"
    assert result["restarts"] == 1
    assert [a["returncode"] for a in result["attempts"]] == [1, 0]
    assert [a["watchdog_cut"] for a in result["attempts"]] == [True, False]
    assert "Timeout (0:00:01)!" in Path(result["attempts"][0]["log"]).read_text(encoding="utf-8")
    assert sorted(p.name for p in work.glob("episode_*.json")) == ["episode_0.json", "episode_1.json", "episode_2.json"]
    restarts = Path(result["restart_log"]).read_text(encoding="utf-8").splitlines()
    assert len(restarts) == 1 and "restart 1 of at most 3" in restarts[0]


def test_a_raising_episode_is_not_restarted(tmp_path: Path) -> None:
    command, work, env = _child(tmp_path, "raise", timeout=5.0, episodes=3, seconds=0.05)
    result = hz.run_with_restarts(command, log_dir=tmp_path / "logs", label="fake", max_restarts=3, env=env)
    assert result["status"] == "failed"
    assert result["restarts"] == 0
    assert [(a["returncode"], a["watchdog_cut"]) for a in result["attempts"]] == [(1, False)]
    assert "RuntimeError: an ordinary failure" in Path(result["attempts"][0]["log"]).read_text(encoding="utf-8")
    assert sorted(p.name for p in work.glob("episode_*.json")) == ["episode_0.json"]


def test_a_healthy_run_longer_than_the_timeout_in_total_is_never_cut(tmp_path: Path) -> None:
    command, work, env = _child(tmp_path, "healthy", timeout=2.0, episodes=5, seconds=0.6)
    result = hz.run_with_restarts(command, log_dir=tmp_path / "logs", label="fake", max_restarts=3, env=env)
    assert result["status"] == "completed"
    assert result["restarts"] == 0
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["seconds"] > 2.0
    assert len(list(work.glob("episode_*.json"))) == 5


def test_the_cap_stops_a_run_that_hangs_every_time(tmp_path: Path) -> None:
    command, work, env = _child(tmp_path, "hang_always", timeout=1.0, episodes=2, seconds=0.05)
    result = hz.run_with_restarts(command, log_dir=tmp_path / "logs", label="fake", max_restarts=2, env=env)
    assert result["status"] == "cap_reached"
    assert result["restarts"] == 2
    assert [a["watchdog_cut"] for a in result["attempts"]] == [True, True, True]
    assert list(work.glob("episode_*.json")) == []
    assert len(Path(result["restart_log"]).read_text(encoding="utf-8").splitlines()) == 3


# ======================================================================
# B.3 wired into the gate: its records on disk are what a restart resumes from
# ======================================================================


def _episode(cell: Any, **overrides: Any) -> Any:
    from offline.engine_att_reference import GateEpisode

    fields = dict(
        scenario=cell.scenario, tier=cell.tier, method=cell.method, arm=cell.arm, seed=cell.seed,
        draw_id=cell.draw_id, role=cell.role, engine_seed=1000, att_reference_engine_population=300.0,
        att_reference_entered_running=299.0, att_reference_entered_population=300.0,
        att_reference_metric_cadence=250.0, att_engine_call=300.0, att_ours=250.0, n_reference_ids=2700,
        n_entered_ids=2700, created_from_flow=2700, entered=2700, never_entered=0, admission_latency_mean=1.0,
        admission_latency_max=5.0, n_admission_delayed=10, interval=1.0, n_observations=3600, seconds=1.0,
        seconds_rollout=1.0,
    )
    fields.update(overrides)
    return GateEpisode(**fields)


def test_gate_records_on_disk_are_what_a_restart_resumes_from(tmp_path: Path) -> None:
    cells = hz.gate_cells_for(["maxpressure", "fixedtime"])
    for cell in cells[:2]:
        hz.write_gate_record(tmp_path, _episode(cell))
    assert hz.pending_gate_cells(tmp_path, cells) == list(cells[2:])
    with pytest.raises(ValueError, match="missing"):
        hz.load_gate_episodes(tmp_path, cells)
    for cell in cells[2:]:
        hz.write_gate_record(tmp_path, _episode(cell))
    assert hz.pending_gate_cells(tmp_path, cells) == []
    loaded = hz.load_gate_episodes(tmp_path, cells)
    assert [(e.tier, e.seed, e.draw_id) for e in loaded] == [(c.tier, c.seed, c.draw_id) for c in cells]
    with pytest.raises(FileExistsError, match="written once"):
        hz.write_gate_record(tmp_path, _episode(cells[0]))


def test_a_gate_record_filed_under_another_cell_is_refused(tmp_path: Path) -> None:
    cells = hz.gate_cells_for(["maxpressure"])
    path = hz.write_gate_record(tmp_path, _episode(cells[0]))
    path.rename(hz.gate_record_path(tmp_path, cells[1]))
    with pytest.raises(ValueError, match="filed"):
        hz.pending_gate_cells(tmp_path, cells)


def test_the_gate_worker_is_a_subcommand() -> None:
    namespace = hz.build_parser().parse_args(["gate-worker", "--work-dir", "w", "--tiers", "maxpressure"])
    assert (namespace.command, namespace.work_dir, namespace.tiers) == ("gate-worker", "w", ["maxpressure"])
