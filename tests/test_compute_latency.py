"""P8.2 (``BRIEF_43`` §6 and Amendment A3): the latency harness ``offline/compute_latency.py``.

* **T-timer (load-bearing):** a fake env whose ``step`` sleeps 50 ms and a fake decision of 2 ms -> about 2 ms per
  decision, never about 52; the warm-up excluded by count; the CUDA path synchronises before BOTH clock readings
  (a spy). *Mutations:* the timer around ``step`` -> dies; the warm-up not excluded -> dies; one sync missing -> dies.
* **T-stats:** the median and the nearest-rank p95 against a sorted-index computation written HERE, under ``==``.
* **T-no-outcome:** a latency record and a canary record carry no key naming an episode quantity; the detector is
  shown to find a planted one, so an empty answer is not a vacuous one.
* **T-regime:** the CPU timing refuses unless one torch thread, ``OMP/MKL = 1`` and ``CUBLAS_WORKSPACE_CONFIG`` unset.
* **Amendment A3.1:** the record is written BEFORE any episode env is closed; ``write_once`` is exclusive and leaves
  nothing behind on failure; the SUPERVISOR, through real subprocesses: a process that hangs AFTER writing is killed,
  its record kept and not re-run; one that hangs BEFORE writing is killed and re-run, at most three attempts, then
  the run FAILS naming the row; one that exits without a record fails at once.
* **The registry** equals the approved plan's Appendix A (36 checkpoint rows, each at its digest) plus six heuristics.

GATES: the tests that read checkpoints, the corpus or the draws skip naming ``RLTRAFFIC_OUTPUT_ROOT``,
``RLTRAFFIC_CORPUS_V11`` or ``RLTRAFFIC_DRAWS`` (each gitignored, in the main tree). Nothing here reads git history.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import offline.compute_latency as cl

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = REPO_ROOT / "docs" / "plans" / "p8.2.md"


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class FakeEnv:
    """The smallest env the timing loop drives: ``reset`` / ``step`` / ``close``, with an optional slow step."""

    def __init__(self, max_steps: int, step_sleep_s: float = 0.0, events: list[str] | None = None,
                 name: str = "env") -> None:
        self.max_steps = int(max_steps)
        self.step_sleep_s = float(step_sleep_s)
        self.events = events if events is not None else []
        self.name = name
        self.reset_seeds: list[int | None] = []
        self._t = 0

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        self.reset_seeds.append(seed)
        self.events.append("reset")
        self._t = 0
        return {"step": 0}

    def step(self, action: Any) -> tuple[float, bool, bool, dict[str, Any]]:
        if self.step_sleep_s:
            time.sleep(self.step_sleep_s)
        self.events.append("step")
        self._t += 1
        return 0.0, False, self._t >= self.max_steps, {"step": self._t}

    def close(self) -> None:
        self.events.append(f"close:{self.name}")


def _sleeping_choose(decision_s: float):
    def choose(_env: Any, _info: dict[str, Any]) -> np.ndarray:
        time.sleep(decision_s)
        return np.zeros(1, dtype=np.int64)

    return choose


def _row(**overrides: Any) -> cl.LatencyRow:
    fields: dict[str, Any] = {
        "row_id": "hz1x1.maxpressure",
        "scenario": "hz1x1",
        "kind": "maxpressure",
        "checkpoint": None,
        "sha256": None,
        "sha256_source": None,
        "method": None,
        "declared_gradient_steps": None,
        "target_rtg": None,
        "target_source": None,
        "fine_tune_k": None,
        "settings_corpus_dir": "cf_hz1x1__maxpressure",
        "devices": ("cpu",),
    }
    fields.update(overrides)
    return cl.LatencyRow(**fields)


def _record_kwargs() -> dict[str, Any]:
    return {
        "draws": (1000, 1001, 1002),
        "prompt": None,
        "factory": "offline.dt_gate._maxpressure_factory",
        "regime": {"torch_num_threads": 1, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "CUBLAS_WORKSPACE_CONFIG": None},
        "machine": {"cpu_model": "test cpu", "logical_cpus": 16},
        "load_before": [0.01, 0.05, 0.08],
        "load_after": [0.02, 0.05, 0.08],
        "git": {"commit": "0" * 40, "dirty": False},
    }


def _three_episodes(value: int = 1_500_000) -> list[list[int]]:
    return [[value] * cl.DECISIONS_PER_EPISODE for _ in cl.TIMING_DRAWS]


# ----------------------------------------------------------------------
# T-timer (load-bearing)
# ----------------------------------------------------------------------


def test_t_timer_the_timer_wraps_the_decision_alone_never_the_step() -> None:
    env = FakeEnv(max_steps=30, step_sleep_s=0.050)
    ns = cl.time_episode(env, _sleeping_choose(0.002), engine_seed=1000)
    assert len(ns) == 30
    stats = cl.latency_stats([ns], warmup=5)
    assert 1.0 <= stats["median_ms"] <= 3.0, stats
    assert max(ns[5:]) < 40_000_000, "a timed decision took about as long as a decision PLUS a step"
    assert env.reset_seeds == [1000]


def test_t_timer_the_warmup_decisions_are_excluded_by_count() -> None:
    w, n = cl.WARMUP, cl.DECISIONS_PER_EPISODE
    episodes = [[10**9] * w + [1_000 + i for i in range(n - w)] for _ in range(3)]
    stats = cl.latency_stats(episodes, warmup=w)
    assert stats["n_timed"] == 3 * (n - w) == 1020
    # 60 of 1,080 values (5.6 %) are warm-up: had they entered, the nearest-rank p95 would be one of them.
    assert stats["p95_ns"] < 10**9
    assert stats["median_ns"] < 10**9


def test_t_timer_the_cuda_path_synchronizes_before_both_clock_readings() -> None:
    events: list[str] = []
    env = FakeEnv(max_steps=4, events=events)
    ticks = iter(range(10, 10_000, 10))

    def clock() -> int:
        events.append("clock")
        return next(ticks)

    def sync() -> None:
        events.append("sync")

    def choose(_env: Any, _info: dict[str, Any]) -> np.ndarray:
        events.append("decide")
        return np.zeros(1, dtype=np.int64)

    ns = cl.time_episode(env, choose, engine_seed=1000, sync=sync, clock=clock)
    assert len(ns) == 4 and all(v == 10 for v in ns)
    assert events[0] == "reset"
    assert events[1:] == ["sync", "clock", "decide", "sync", "clock", "step"] * 4


def test_t_timer_the_cpu_path_reads_the_clock_around_the_decision_only() -> None:
    events: list[str] = []
    env = FakeEnv(max_steps=3, events=events)
    ticks = iter(range(7, 7_000, 7))

    def clock() -> int:
        events.append("clock")
        return next(ticks)

    def choose(_env: Any, _info: dict[str, Any]) -> np.ndarray:
        events.append("decide")
        return np.zeros(1, dtype=np.int64)

    cl.time_episode(env, choose, engine_seed=1000, sync=None, clock=clock)
    assert events[1:] == ["clock", "decide", "clock", "step"] * 3


# ----------------------------------------------------------------------
# T-stats
# ----------------------------------------------------------------------


def test_t_stats_median_and_p95_equal_a_sorted_index_route_on_a_fixed_sample() -> None:
    rng = np.random.default_rng(20261003)
    episodes = [rng.integers(500_000, 5_000_000, size=cl.DECISIONS_PER_EPISODE).tolist() for _ in range(3)]
    stats = cl.latency_stats(episodes, warmup=cl.WARMUP)

    timed = sorted(int(v) for episode in episodes for v in episode[cl.WARMUP:])
    n = len(timed)
    assert n == 1020
    rank = (95 * n + 99) // 100  # ceil(0.95 n) in integers: the nearest-rank position, 1-based
    expected_p95 = timed[rank - 1]
    expected_median = (timed[n // 2 - 1] + timed[n // 2]) / 2  # n is even
    assert stats["n_timed"] == n
    assert stats["p95_ns"] == expected_p95
    assert stats["median_ns"] == expected_median
    assert stats["p95_ms"] == expected_p95 / 1e6
    assert stats["median_ms"] == expected_median / 1e6


def test_t_stats_an_odd_count_takes_the_middle_value() -> None:
    episodes = [[5, 1, 9, 3, 7]]
    stats = cl.latency_stats(episodes, warmup=0)
    assert stats["median_ns"] == 5
    assert stats["p95_ns"] == 9  # ceil(0.95 x 5) = 5th of 5


# ----------------------------------------------------------------------
# T-no-outcome
# ----------------------------------------------------------------------


def test_t_no_outcome_the_detector_finds_planted_metric_keys_and_spares_innocent_ones() -> None:
    planted = {
        "row": "x",
        "attempts": [{"status": "ok"}],
        "episodes": [{"draw": 1000, "decision_ns": [1, 2], "episode_reward": -3.0}],
        "machine": {"vehicle_count": 4},
        "att_horizon": 1.0,
    }
    found = cl.find_outcome_keys(planted)
    assert sorted(found) == ["att_horizon", "episodes[0].episode_reward", "machine.vehicle_count"]
    assert cl.find_outcome_keys({"attempts": 1, "attention": 2, "pressure_free": None}) == ["pressure_free"]


def test_t_no_outcome_a_latency_record_names_no_episode_quantity() -> None:
    record = cl.build_record(_row(), "cpu", _three_episodes(), **_record_kwargs())
    assert cl.find_outcome_keys(record) == []
    assert record["format_version"] == cl.FORMAT_VERSION
    assert record["n_timed"] == 1020
    assert [episode["draw"] for episode in record["episodes"]] == [1000, 1001, 1002]


def test_t_no_outcome_build_record_refuses_a_block_that_names_a_metric() -> None:
    kwargs = _record_kwargs()
    kwargs["machine"] = {"cpu_model": "x", "vehicle_count": 3}
    with pytest.raises(ValueError, match="vehicle_count"):
        cl.build_record(_row(), "cpu", _three_episodes(), **kwargs)


def test_t_no_outcome_the_canary_record_holds_no_value_of_the_canary_episode() -> None:
    record = cl.build_canary_record("open", 0.83, reproduced=True, git={"commit": "0" * 40, "dirty": False})
    assert cl.find_outcome_keys(record) == []
    assert record["format_version"] == cl.CANARY_FORMAT_VERSION
    assert record["seconds"] == 0.83 and record["threshold_seconds"] == 2.0
    assert record["verdict"] == "at speed" and record["reproduced"] is True
    slow = cl.build_canary_record("close", 2.4, reproduced=True, git={})
    assert slow["verdict"] == "throttled"


# ----------------------------------------------------------------------
# The record's own refusals
# ----------------------------------------------------------------------


def test_build_record_refuses_an_episode_with_another_decision_count() -> None:
    episodes = _three_episodes()
    episodes[1] = episodes[1][:-1]
    with pytest.raises(ValueError, match="359"):
        cl.build_record(_row(), "cpu", episodes, **_record_kwargs())


def test_build_record_refuses_fewer_than_one_thousand_timed_decisions() -> None:
    kwargs = _record_kwargs()
    kwargs["draws"] = (1000, 1001)
    with pytest.raises(ValueError, match="1000"):
        cl.build_record(_row(), "cpu", _three_episodes()[:2], **kwargs)


def test_build_record_states_the_per_intersection_figure_as_derived() -> None:
    grid = _row(row_id="grid4x4.maxpressure", scenario="grid4x4", settings_corpus_dir="cf_grid4x4__maxpressure")
    record = cl.build_record(grid, "cpu", _three_episodes(1_600_000), **_record_kwargs())
    assert record["n_intersections"] == 16
    assert record["per_intersection"]["median_ms"] == record["median_ms"] / 16
    assert record["per_intersection"]["p95_ms"] == record["p95_ms"] / 16
    assert "derived" in record["per_intersection"]["note"]


# ----------------------------------------------------------------------
# T-regime
# ----------------------------------------------------------------------


@pytest.fixture
def torch_threads_restored():
    import torch

    before = torch.get_num_threads()
    yield torch
    torch.set_num_threads(before)


PINNED = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def test_t_regime_refuses_more_than_one_torch_thread(torch_threads_restored: Any) -> None:
    torch_threads_restored.set_num_threads(2)
    with pytest.raises(ValueError, match="one torch thread"):
        cl.assert_cpu_regime(dict(PINNED))


def test_t_regime_refuses_omp_or_mkl_unpinned(torch_threads_restored: Any) -> None:
    torch_threads_restored.set_num_threads(1)
    with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
        cl.assert_cpu_regime({"MKL_NUM_THREADS": "1"})
    with pytest.raises(ValueError, match="MKL_NUM_THREADS"):
        cl.assert_cpu_regime({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "4"})


def test_t_regime_refuses_a_cublas_workspace_config(torch_threads_restored: Any) -> None:
    torch_threads_restored.set_num_threads(1)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        cl.assert_cpu_regime({**PINNED, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})


def test_t_regime_accepts_the_registered_regime_and_reports_it(torch_threads_restored: Any) -> None:
    torch_threads_restored.set_num_threads(1)
    block = cl.assert_cpu_regime(dict(PINNED))
    assert block["torch_num_threads"] == 1
    assert block["OMP_NUM_THREADS"] == "1" and block["MKL_NUM_THREADS"] == "1"
    assert block["CUBLAS_WORKSPACE_CONFIG"] is None


# ----------------------------------------------------------------------
# write_once and the order of record and teardown (Amendment A3.1)
# ----------------------------------------------------------------------


def test_write_once_refuses_an_existing_record_and_leaves_it_untouched(tmp_path: Path) -> None:
    path = tmp_path / "row_cpu.json"
    cl.write_once(path, {"format_version": cl.FORMAT_VERSION, "a": 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="row_cpu.json"):
        cl.write_once(path, {"format_version": cl.FORMAT_VERSION, "a": 2})
    assert path.read_bytes() == before
    assert json.loads(before)["a"] == 1


def test_write_once_leaves_no_record_when_the_payload_cannot_be_written(tmp_path: Path) -> None:
    path = tmp_path / "row_cpu.json"
    with pytest.raises(TypeError, match="not JSON serializable"):
        cl.write_once(path, {"format_version": cl.FORMAT_VERSION, "a": object()})
    assert not path.exists()


def test_the_record_is_written_before_any_episode_env_is_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torch_threads_restored: Any
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    events: list[str] = []

    def env_builder(draw: int) -> FakeEnv:
        events.append(f"build:{draw}")
        return FakeEnv(max_steps=cl.DECISIONS_PER_EPISODE, events=events, name=str(draw))

    def factory_builder():
        return lambda _env: (lambda _e, _info: np.zeros(1, dtype=np.int64))

    def writer(path: Path, payload: dict[str, Any]) -> None:
        events.append("write")
        cl.write_once(path, payload)

    out = cl.run_row(
        _row(),
        "cpu",
        out_dir=tmp_path,
        output_root=tmp_path / "output",
        corpus_root=tmp_path / "corpus",
        draws_root=tmp_path / "draws",
        data_dir=REPO_ROOT / "docs" / "data",
        env_builder=env_builder,
        factory_builder=factory_builder,
        writer=writer,
    )
    assert out == tmp_path / "hz1x1.maxpressure_cpu.json"
    written_at = events.index("write")
    closes = [i for i, event in enumerate(events) if event.startswith("close:")]
    assert sorted(event for event in events if event.startswith("close:")) == ["close:1000", "close:1001", "close:1002"]
    assert all(i > written_at for i in closes), events[:5] + ["..."] + events[-6:]
    record = json.loads(out.read_text())
    assert record["n_timed"] == 1020 and cl.find_outcome_keys(record) == []


# ----------------------------------------------------------------------
# The supervisor, through real subprocesses (Amendment A3.1)
# ----------------------------------------------------------------------


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _writer_snippet(record: Path) -> str:
    return textwrap.dedent(
        f"""
        import json, os, pathlib
        record = pathlib.Path({str(record)!r})
        tmp = record.parent / (record.name + ".tmp")
        tmp.write_text(json.dumps({{"format_version": {cl.FORMAT_VERSION!r}, "row": "fake"}}))
        os.link(tmp, record)
        tmp.unlink()
        """
    )


def test_a3_a_process_that_hangs_after_writing_is_killed_its_record_kept_and_not_rerun(tmp_path: Path) -> None:
    record = tmp_path / "fake.row_cpu.json"
    pids = tmp_path / "pids.txt"
    script = (
        f"import os, pathlib, time\n"
        f"with open({str(pids)!r}, 'a') as handle: handle.write(str(os.getpid()) + chr(10))\n"
        + _writer_snippet(record)
        + "time.sleep(600)\n"
    )
    outcome = cl.supervise(
        "fake.row_cpu", [sys.executable, "-c", script], record_path=record, timeout_s=3.0, kill_grace_s=1.0
    )
    assert outcome["status"] == "kept", outcome
    assert [a["status"] for a in outcome["attempts"]] == ["hung"]
    assert outcome["attempts"][0]["record_written"] is True
    assert json.loads(record.read_text()) == {"format_version": cl.FORMAT_VERSION, "row": "fake"}
    launched = [int(line) for line in pids.read_text().split()]
    assert len(launched) == 1, "the process was re-run although its record existed"
    assert not _alive(launched[0])


def test_a3_a_process_that_hangs_before_writing_is_rerun_at_most_three_times_then_fails_naming_the_row(
    tmp_path: Path,
) -> None:
    record = tmp_path / "fake.row_cpu.json"
    pids = tmp_path / "pids.txt"
    script = (
        f"import os, time\n"
        f"with open({str(pids)!r}, 'a') as handle: handle.write(str(os.getpid()) + chr(10))\n"
        "time.sleep(600)\n"
    )
    outcome = cl.supervise(
        "fake.row_cpu", [sys.executable, "-c", script], record_path=record, timeout_s=2.0, kill_grace_s=1.0
    )
    assert outcome["status"] == "failed"
    assert [a["status"] for a in outcome["attempts"]] == ["hung", "hung", "hung"]
    assert "fake.row_cpu" in outcome["reason"]
    assert not record.exists()
    launched = [int(line) for line in pids.read_text().split()]
    assert len(launched) == 3 and not any(_alive(pid) for pid in launched)


def test_a3_a_process_that_hangs_once_before_writing_is_rerun_and_its_second_attempt_stands(tmp_path: Path) -> None:
    record = tmp_path / "fake.row_cpu.json"
    counter = tmp_path / "attempt.txt"
    script = (
        "import pathlib, time\n"
        f"counter = pathlib.Path({str(counter)!r})\n"
        "n = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(n))\n"
        "if n == 1:\n"
        "    time.sleep(600)\n"
        + _writer_snippet(record)
    )
    outcome = cl.supervise(
        "fake.row_cpu", [sys.executable, "-c", script], record_path=record, timeout_s=3.0, kill_grace_s=1.0
    )
    assert outcome["status"] == "ok", outcome
    assert [a["status"] for a in outcome["attempts"]] == ["hung", "ok"]
    assert counter.read_text() == "2"


def test_a3_a_process_that_exits_without_a_record_fails_at_once_without_a_rerun(tmp_path: Path) -> None:
    record = tmp_path / "fake.row_cpu.json"
    outcome = cl.supervise(
        "fake.row_cpu", [sys.executable, "-c", "raise SystemExit(3)"], record_path=record, timeout_s=10.0
    )
    assert outcome["status"] == "failed"
    assert [a["status"] for a in outcome["attempts"]] == ["failed"]
    assert outcome["attempts"][0]["returncode"] == 3
    assert "fake.row_cpu" in outcome["reason"] and "3" in outcome["reason"]


def test_a3_the_run_fails_at_the_first_row_that_hangs_three_times_and_launches_nothing_after_it(
    tmp_path: Path,
) -> None:
    hang = [sys.executable, "-c", "import time; time.sleep(600)"]
    later = tmp_path / "b_cpu.json"
    jobs = [
        ("a_cpu", hang, tmp_path / "a_cpu.json", 1.5),
        ("b_cpu", [sys.executable, "-c", _writer_snippet(later)], later, 10.0),
    ]
    result = cl.run_jobs(jobs, kill_grace_s=1.0)
    assert result["status"] == "failed"
    assert result["failed"] == "a_cpu"
    assert "a_cpu" in result["reason"]
    assert [o["label"] for o in result["outcomes"]] == ["a_cpu"]
    assert len(result["outcomes"][0]["attempts"]) == 3
    assert not later.exists()


# ----------------------------------------------------------------------
# The run's orchestration, with fake processes
# ----------------------------------------------------------------------


def _fake_command(script: str) -> list[str]:
    return [sys.executable, "-c", script]


def _fake_record_script(path: Path, payload: dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        import json, os, pathlib
        path = pathlib.Path({str(path)!r})
        tmp = path.parent / (path.name + ".tmp")
        tmp.write_text(json.dumps({payload!r}))
        os.link(tmp, path)
        tmp.unlink()
        """
    )


def _run_all_fakes(tmp_path: Path, *, close_seconds: float = 0.8, reproduced: bool = True):
    rows = [_row(), _row(row_id="grid4x4.maxpressure", scenario="grid4x4", settings_corpus_dir="cf_grid4x4__maxpressure")]

    def command_for(row: cl.LatencyRow, device: str, out_dir: Path) -> list[str]:
        payload = cl.build_record(row, device, _three_episodes(), **_record_kwargs())
        return _fake_command(_fake_record_script(out_dir / f"{row.row_id}_{device}.json", payload))

    def canary_command_for(phase: str, out_dir: Path) -> list[str]:
        seconds = 0.8 if phase == "open" else close_seconds
        payload = cl.build_canary_record(phase, seconds, reproduced=reproduced, git={})
        return _fake_command(_fake_record_script(out_dir / f"canary_{phase}.json", payload))

    return rows, command_for, canary_command_for


def test_run_all_writes_complete_the_run_record_and_the_manifest(tmp_path: Path) -> None:
    rows, command_for, canary_command_for = _run_all_fakes(tmp_path)
    latency_root = tmp_path / "output" / "p8_2" / "latency"
    manifest = tmp_path / "output" / "SHA256SUMS_p8_2.txt"
    code = cl.run_all(
        stamp="20261003T120000Z", rows=rows, latency_root=latency_root, manifest_path=manifest,
        command_for=command_for, canary_command_for=canary_command_for,
        timeouts={"hz1x1": 30.0, "grid4x4": 30.0}, canary_timeout_s=30.0, git={"commit": "0" * 40, "dirty": False},
    )
    run_dir = latency_root / "20261003T120000Z"
    assert code == 0
    assert (run_dir / "COMPLETE").is_file() and not (run_dir / "FAILED").exists()
    run = json.loads((run_dir / "run.json").read_text())
    assert run["status"] == "COMPLETE"
    assert [o["label"] for o in run["outcomes"]] == ["canary_open", "hz1x1.maxpressure_cpu", "grid4x4.maxpressure_cpu",
                                                     "canary_close"]
    lines = manifest.read_text().splitlines()
    listed = {line.split("  ", 1)[1] for line in lines}
    assert "20261003T120000Z/hz1x1.maxpressure_cpu.json" in listed and "20261003T120000Z/run.json" in listed
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256((latency_root / rel).read_bytes()).hexdigest() == digest


def test_run_all_fails_on_a_throttled_closing_canary_and_writes_no_manifest(tmp_path: Path) -> None:
    rows, command_for, canary_command_for = _run_all_fakes(tmp_path, close_seconds=2.4)
    latency_root = tmp_path / "output" / "p8_2" / "latency"
    manifest = tmp_path / "output" / "SHA256SUMS_p8_2.txt"
    code = cl.run_all(
        stamp="S", rows=rows, latency_root=latency_root, manifest_path=manifest, command_for=command_for,
        canary_command_for=canary_command_for, timeouts={"hz1x1": 30.0, "grid4x4": 30.0}, canary_timeout_s=30.0,
        git={},
    )
    assert code == 1
    assert (latency_root / "S" / "FAILED").is_file() and not (latency_root / "S" / "COMPLETE").exists()
    assert "throttled" in json.loads((latency_root / "S" / "run.json").read_text())["reason"]
    assert not manifest.exists()


def test_run_all_fails_before_any_row_when_the_engine_does_not_reproduce_the_canary(tmp_path: Path) -> None:
    rows, command_for, canary_command_for = _run_all_fakes(tmp_path, reproduced=False)
    latency_root = tmp_path / "output" / "p8_2" / "latency"
    code = cl.run_all(
        stamp="S", rows=rows, latency_root=latency_root, manifest_path=tmp_path / "m.txt", command_for=command_for,
        canary_command_for=canary_command_for, timeouts={"hz1x1": 30.0, "grid4x4": 30.0}, canary_timeout_s=30.0,
        git={},
    )
    assert code == 1
    assert not list((latency_root / "S").glob("*maxpressure*.json"))


def test_run_all_refuses_an_existing_manifest_before_creating_anything(tmp_path: Path) -> None:
    rows, command_for, canary_command_for = _run_all_fakes(tmp_path)
    manifest = tmp_path / "SHA256SUMS_p8_2.txt"
    manifest.write_text("x\n")
    latency_root = tmp_path / "output" / "p8_2" / "latency"
    with pytest.raises(FileExistsError, match="SHA256SUMS_p8_2.txt"):
        cl.run_all(
            stamp="S", rows=rows, latency_root=latency_root, manifest_path=manifest, command_for=command_for,
            canary_command_for=canary_command_for, timeouts={"hz1x1": 30.0, "grid4x4": 30.0}, canary_timeout_s=30.0,
            git={},
        )
    assert not latency_root.exists()


# ----------------------------------------------------------------------
# The G1 pre-flight's arithmetic
# ----------------------------------------------------------------------


def test_derive_timeouts_takes_three_times_the_slowest_process_of_each_scenario_with_a_floor() -> None:
    measured = {
        ("hz1x1.dt_k20", "cpu"): 20.2,
        ("hz1x1.dt_k20", "cuda"): 25.1,
        ("grid4x4.dt_nomix_h4", "cpu"): 100.4,
        ("grid4x4.dt_nomix_h4", "cuda"): 30.0,
    }
    assert cl.derive_timeouts(measured) == {"hz1x1": 120.0, "grid4x4": 302.0}


def test_expected_duration_costs_every_process_at_its_scenarios_measured_process_on_that_device() -> None:
    measured = {
        ("hz1x1.dt_k20", "cpu"): 20.0,
        ("hz1x1.dt_k20", "cuda"): 25.0,
        ("grid4x4.dt_nomix_h4", "cpu"): 100.0,
        ("grid4x4.dt_nomix_h4", "cuda"): 30.0,
    }
    rows = [
        _row(row_id="hz1x1.bc", kind="baseline", devices=("cpu", "cuda")),
        _row(row_id="hz1x1.maxpressure"),
        _row(row_id="grid4x4.bc", scenario="grid4x4", kind="baseline", devices=("cpu", "cuda")),
    ]
    estimate = cl.expected_duration(measured, 0.8, rows=rows)
    assert estimate["seconds"] == 20.0 + 25.0 + 20.0 + 100.0 + 30.0 + 2 * 0.8
    assert estimate["processes"] == 5
    assert "upper bound" in estimate["basis"]


# ----------------------------------------------------------------------
# The registry against the approved plan
# ----------------------------------------------------------------------


def _appendix_a() -> dict[str, tuple[str, str]]:
    text = PLAN.read_text(encoding="utf-8")
    appendix = text.split("## Appendix A", 1)[1]
    rows: dict[str, tuple[str, str]] = {}
    for line in appendix.splitlines():
        match = re.match(r"^\| `([^`]+)` \| `([^`]+)` \| `([0-9a-f]{64})` \|", line)
        if match:
            rows[match.group(1)] = (match.group(2), match.group(3))
    return rows


def test_the_registry_holds_every_appendix_a_row_at_its_path_and_digest() -> None:
    appendix = _appendix_a()
    assert len(appendix) == 36
    registry = {row.row_id: row for row in cl.ROWS if row.checkpoint is not None}
    assert sorted(registry) == sorted(appendix)
    for row_id, (path, digest) in appendix.items():
        assert registry[row_id].checkpoint == path, row_id
        assert registry[row_id].sha256 == digest, row_id


def test_the_registry_adds_exactly_six_heuristic_rows_timed_on_cpu_only() -> None:
    heuristics = sorted(row.row_id for row in cl.ROWS if row.checkpoint is None)
    assert heuristics == sorted(f"{s}.{h}" for s in ("hz1x1", "grid4x4") for h in ("maxpressure", "fixedtime", "random"))
    for row in cl.ROWS:
        expected = ("cpu",) if row.checkpoint is None else ("cpu", "cuda")
        assert row.devices == expected, row.row_id
    assert len({row.row_id for row in cl.ROWS}) == len(cl.ROWS) == 42


def test_every_registry_row_names_a_scenario_its_settings_and_a_factory_kind() -> None:
    kinds = {"dt", "spatial_dt", "spatial_dt_targets", "baseline", "mappo", "maxpressure", "fixedtime", "random"}
    assert len(cl.ROWS) == 42
    for row in cl.ROWS:
        assert row.scenario in cl.SCENARIOS, row.row_id
        assert row.row_id.startswith(row.scenario + "."), row.row_id
        assert row.kind in kinds, row.row_id
        assert row.settings_corpus_dir.startswith(f"cf_{row.scenario}__"), row.row_id
        assert (row.sha256_source is None) == (row.checkpoint is None), row.row_id
    assert cl.SCENARIOS["hz1x1"].n_intersections == 1 and cl.SCENARIOS["grid4x4"].n_intersections == 16
    with pytest.raises(KeyError, match="no.such.row"):
        cl.row_by_id("no.such.row")


# ----------------------------------------------------------------------
# Gated: the registry against the files and the records that name them
# ----------------------------------------------------------------------


def _root(variable: str, default: Path, marker: str) -> Path:
    value = os.environ.get(variable)
    candidate = Path(value) if value else default
    if not (candidate / marker).exists():
        pytest.skip(f"{candidate / marker} not found: set {variable} to the main tree's copy (it is gitignored)")
    return candidate


def _output_root() -> Path:
    return _root("RLTRAFFIC_OUTPUT_ROOT", REPO_ROOT / "output", "p4_dt/dt_seed101.pt")


def _corpus_root() -> Path:
    return _root("RLTRAFFIC_CORPUS_V11", REPO_ROOT / "datasets_v11", "cf_hz1x1__mappo1000__seed101/manifest.json")


def _draws_root() -> Path:
    return _root("RLTRAFFIC_DRAWS", REPO_ROOT / "scenarios" / "draws", "cityflow1x1/draw_1000/cityflow.json")


def _resolve(document: Any, path: str) -> Any:
    """This test's own JSON-path reader (``$.a['b'].c[3]``), independent of the module's."""
    assert path.startswith("$")
    for token in re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)|\['([^']*)'\]|\[(\d+)\]", path[1:]):
        name, quoted, index = token
        if index:
            document = document[int(index)]
        else:
            document = document[name or quoted]
    return document


def test_every_representative_checkpoint_hashes_to_the_digest_its_named_source_holds() -> None:
    output_root = _output_root()
    corpus_root = _corpus_root()
    main_root = output_root.parent
    checked = 0
    for row in cl.ROWS:
        if row.checkpoint is None:
            continue
        checked += 1
        actual = hashlib.sha256((main_root / row.checkpoint).read_bytes()).hexdigest()
        assert actual == row.sha256, row.row_id
        kind, file, locator = row.sha256_source
        if kind == "json":
            named = _resolve(json.loads((REPO_ROOT / file).read_text()), locator)
        elif kind == "sums":
            table = dict(
                reversed(line.split("  ", 1)) for line in (output_root / file).read_text().splitlines()
            )
            named = table[locator]
        else:
            assert kind == "corpus_manifest", row.row_id
            named = _resolve(json.loads((corpus_root / file).read_text()), locator)
        assert named == row.sha256, row.row_id
    assert checked == 36, f"{checked} checkpoint rows checked: an empty registry must not pass this test"


def test_a_real_row_process_writes_one_record_of_1020_timed_decisions_and_no_outcome(tmp_path: Path) -> None:
    output_root = _output_root()
    corpus_root = _corpus_root()
    draws_root = _draws_root()
    env = {key: value for key, value in os.environ.items() if key != "CUBLAS_WORKSPACE_CONFIG"}
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": str(REPO_ROOT)})
    command = [
        sys.executable, "-P", "-m", "offline.compute_latency", "run-row", "--row", "hz1x1.bc", "--device", "cpu",
        "--out-dir", str(tmp_path), "--output-root", str(output_root), "--corpus-root", str(corpus_root),
        "--draws-root", str(draws_root), "--data-dir", str(REPO_ROOT / "docs" / "data"),
    ]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    record = json.loads((tmp_path / "hz1x1.bc_cpu.json").read_text())
    assert record["format_version"] == cl.FORMAT_VERSION
    assert record["row"] == "hz1x1.bc" and record["device"] == "cpu"
    assert record["checkpoint"]["sha256"] == cl.row_by_id("hz1x1.bc").sha256
    assert record["n_timed"] == 1020
    assert [len(episode["decision_ns"]) for episode in record["episodes"]] == [360, 360, 360]
    assert record["regime"]["torch_num_threads"] == 1
    assert cl.find_outcome_keys(record) == []
    for forbidden in ("att", "reward", "travel"):
        assert not re.search(rf"\b{forbidden}", completed.stdout.lower()), completed.stdout
