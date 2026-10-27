"""Tests for C9 (``BRIEF_47`` Amendment B, B.3; rebuilt under Amendment C): the per-cell watchdog of
``offline.demand_shift.evaluate_cells`` and the restart loop of ``offline/campaigns/p9_2_demand_shift.sh``'s shard mode.

CityFlow's ``Engine::~Engine()`` can wait forever at a two-party barrier (``docs/notes/P5.3c_HANG_2026-09-30.md``;
``DEFERRED`` 104).  B.3: ``evaluate_cells`` arms ``faulthandler.dump_traceback_later(600, exit=True)`` before each cell and
cancels it after, the shard's log as the dump's file; the driver restarts a shard that exits with the watchdog's dump in its
log -- resuming from the records on disk -- at most 20 times, each restart logged with the time and the dump's top frame; any
other non-zero exit stops the shard as before.

**How the hang is faked:** a libc ``sleep`` called through ``ctypes.PyDLL`` inside a fake cell -- a C-level wait that HOLDS the
GIL, as the engine's destructor does.  The watchdog is a C thread that needs no GIL, which is why it can cut it.

**The driver-level tests run the REAL driver** in shard mode (``--run-shard``) with ``P92_PYTHON`` pointing at a stand-in that
execs the real interpreter on a generated entry: the entry replaces ``roll_cell``, ``declared_cells`` and the guard check of
``offline.demand_shift`` with fakes and calls the real ``main`` -- so the CLI options, the watchdog and the restart loop are the
shipped ones.  The watchdog is scaled down (``P92_WATCHDOG_SECONDS``) and so is the cap (``P92_MAX_RESTARTS``) where a test
needs it; every child runs in its own session and is killed with its group if it outlives the test's timeout.
"""

from __future__ import annotations

import inspect
import os
import re
import signal
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import offline.demand_shift as ds
from tests.test_demand_shift_evaluate import _fake_roll

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "offline" / "campaigns" / "p9_2_demand_shift.sh"
TIMEOUT_BANNER = re.compile(r"(?m)^Timeout \([0-9]+:[0-9]{2}:[0-9]{2}(\.[0-9]+)?\)!$")

#: The four fake cells every driver-level test rolls; the third is the one that hangs or raises.
FAKE_CELLS = [
    cell for cell in ds.declared_cells(stage=1)
    if cell["file"] == "kn-hz" and cell["arm"] == "random_fresh" and cell["seed"] == 101
][:4]

FAKE_ENTRY = r'''
"""A stand-in for ``python -m offline.demand_shift``: the REAL ``main`` over four fake cells."""
import ctypes
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import offline.demand_shift as ds

MODE = os.environ["FAKE_MODE"]  # healthy | hang_once | hang_always | raise
MARK = Path(os.environ["FAKE_MARK"])
CELLS = [c for c in ds.declared_cells(stage=1)
         if c["file"] == "kn-hz" and c["arm"] == "random_fresh" and c["seed"] == 101][:4]
TROUBLE = CELLS[2]


def hang():
    # A C-level wait that holds the GIL, as CityFlow's Engine::~Engine() does at its barrier.
    ctypes.PyDLL(None).sleep(int(os.environ.get("FAKE_HANG_SECONDS", "30")))


def roll(cell, **_kwargs):
    if cell == TROUBLE:
        if MODE == "raise":
            raise ValueError("a fake cell that raises")
        if MODE == "hang_always" or (MODE == "hang_once" and not MARK.exists()):
            MARK.write_text("hung\n", encoding="utf-8")
            hang()
    p4_6 = SimpleNamespace(att_horizon=100.0, horizon_vehicle_count=10.0, episode_reward=-50.0,
                           action_sequence_sha256="d" * 64)
    p8_4b = SimpleNamespace(att_ours=100.0, att_engine=99.0, created=100, entered=99, never_entered=1,
                            entered_fraction=0.99, horizon_vehicle_count=10.0, episode_reward=-50.0, seconds=0.1,
                            seconds_rollout=0.1, action_sequence_sha256="d" * 64)
    record = ds.cell_record(cell, p4_6, p8_4b, source={"kind": "stand-in"})
    record.update(git_commit="c" * 40, git_dirty=False)
    return record


ds.declared_cells = lambda stage: list(CELLS)
ds.roll_cell = roll
ds._guard_refusal = lambda path: None
raise SystemExit(ds.main(sys.argv[1:]))
'''

STUB_PYTHON = r'''#!/usr/bin/env bash
# A stand-in interpreter: it records its arguments and runs `-m offline.demand_shift ...` as the generated entry.
printf '%s\n' "$*" >> "$STUB_LOG"
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "offline.demand_shift" ]; then
  shift 2
  exec "$REAL_PYTHON" "$FAKE_ENTRY" "$@"
fi
echo "the stand-in interpreter was asked for something other than offline.demand_shift: $*" >&2
exit 97
'''


def _run(command: list[str], *, env: dict[str, str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Run *command* in its own session; past *timeout* its whole process group is killed and the test fails."""
    process = subprocess.Popen(command, env=env, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail(f"{command} outlived {timeout} s and was killed with its group\n{stdout}\n{stderr}")
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


class _Shard:
    """One run of the driver's shard mode over the four fake cells, and what it left behind."""

    def __init__(self, tmp_path: Path, mode: str, *, watchdog: str | None = "1.5", max_restarts: int | None = None,
                 timeout: float = 240.0) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        entry = tmp_path / "fake_entry.py"
        entry.write_text(FAKE_ENTRY, encoding="utf-8")
        stub = bin_dir / "python"
        stub.write_text(STUB_PYTHON, encoding="utf-8")
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.calls_log = tmp_path / "python_calls.log"
        self.log_path = tmp_path / "runs" / "logs" / "shift_stage1_shard0.log"
        self.work = tmp_path / "output" / "p9_2" / "demand_shift"
        env = {key: value for key, value in os.environ.items()
               if key not in ("STAGE", "SHARDS", "IQL_T_PINS", "P92_MAX_RESTARTS", "P92_WATCHDOG_SECONDS")}
        env.update(
            P92_PYTHON=str(stub), P92_OUTPUT_ROOT=str(tmp_path / "output"), P92_RUNS=str(tmp_path / "runs"),
            SHARDS="1", STAGE="1", PYTHONPATH=str(REPO_ROOT), REAL_PYTHON=sys.executable, FAKE_ENTRY=str(entry),
            FAKE_MODE=mode, FAKE_MARK=str(tmp_path / "hung.mark"), FAKE_HANG_SECONDS="30", STUB_LOG=str(self.calls_log),
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        )
        if watchdog is not None:
            env["P92_WATCHDOG_SECONDS"] = watchdog
        if max_restarts is not None:
            env["P92_MAX_RESTARTS"] = str(max_restarts)
        self.result = _run(["bash", str(DRIVER), "--run-shard", "0"], env=env, timeout=timeout)
        self.log = self.log_path.read_text(encoding="utf-8") if self.log_path.exists() else ""

    def evaluate_calls(self) -> list[str]:
        calls = self.calls_log.read_text(encoding="utf-8").splitlines() if self.calls_log.exists() else []
        return [call for call in calls if " evaluate " in f" {call} "]

    def restarts(self) -> list[tuple[str, str, str, str]]:
        return re.findall(r"(?m)^SHARD 0 RESTART (\d+)/(\d+) at (\S+): .* top frame: (.*)$", self.log)

    def records(self) -> list[str]:
        return sorted(path.name for path in self.work.glob("cell_*.json"))

    def explain(self) -> str:
        return f"exit {self.result.returncode}\n--- log\n{self.log}\n--- stderr\n{self.result.stderr}"


# ----------------------------------------------------------------------
# B.3 (a): a cell blocked in a C-level wait is cut, and the shard completes on restart
# ----------------------------------------------------------------------


def test_a_cell_blocked_in_a_c_level_wait_is_cut_by_the_watchdog_and_the_shard_completes_on_restart(
    tmp_path: Path,
) -> None:
    shard = _Shard(tmp_path, "hang_once")
    assert shard.result.returncode == 0, shard.explain()
    assert len(TIMEOUT_BANNER.findall(shard.log)) == 1, shard.explain()
    restarts = shard.restarts()
    assert [(number, cap) for number, cap, _time, _frame in restarts] == [("1", "20")], shard.explain()
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", restarts[0][2])
    assert "fake_entry.py" in restarts[0][3] and restarts[0][3].endswith("in hang"), restarts[0][3]
    assert len(shard.evaluate_calls()) == 2
    # the restart resumed from the records on disk: the two cells written before the hang were skipped, not rolled again
    assert "stage 1 shard 0/1: rolled 2, skipped 2 of 4" in shard.log, shard.explain()
    assert shard.records() == sorted(ds.cell_name(cell) for cell in FAKE_CELLS)
    assert re.search(r"(?m)^SHARD 0 EXIT: 0$", shard.log), shard.explain()


# ----------------------------------------------------------------------
# B.3 (b): a cell that raises is not restarted
# ----------------------------------------------------------------------


def test_a_cell_that_raises_is_not_restarted(tmp_path: Path) -> None:
    """An ordinary exception also exits 1 -- but with no watchdog dump in the log, so the shard stops as before."""
    shard = _Shard(tmp_path, "raise")
    assert shard.result.returncode == 1, shard.explain()
    assert "ValueError: a fake cell that raises" in shard.log, shard.explain()
    assert not TIMEOUT_BANNER.search(shard.log), shard.explain()
    assert shard.restarts() == []
    assert len(shard.evaluate_calls()) == 1
    assert re.search(r"(?m)^SHARD 0 STOPPED: exit 1 without the watchdog's dump; not restarted$", shard.log), shard.explain()
    assert re.search(r"(?m)^SHARD 0 EXIT: 1$", shard.log), shard.explain()


# ----------------------------------------------------------------------
# B.3 (c): the watchdog is cancelled between cells
# ----------------------------------------------------------------------


def test_the_watchdog_is_armed_before_each_rolled_cell_and_cancelled_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spy on ``faulthandler``: arm (600 s, exit, the given file) -> the cell -> cancel, for every cell ROLLED, in order;
    a cell already on disk arms nothing; a cell that raises is cancelled too."""
    events: list[tuple[Any, ...]] = []

    def arm(timeout: float, repeat: bool = False, file: Any = None, exit: bool = False) -> None:
        events.append(("arm", timeout, repeat, exit, file))

    monkeypatch.setattr(ds.faulthandler, "dump_traceback_later", arm)
    monkeypatch.setattr(ds.faulthandler, "cancel_dump_traceback_later", lambda: events.append(("cancel",)))
    cells = FAKE_CELLS[:3]
    work = tmp_path / "w"
    ds.evaluate_cells(cells[:1], work_dir=work, shard=0, of=1, roll_fn=_fake_roll)
    events.clear()

    def rolling(cell: dict[str, Any]) -> dict[str, Any]:
        events.append(("roll", ds.cell_name(cell)))
        return _fake_roll(cell)

    with (tmp_path / "dump.log").open("a", encoding="utf-8") as dump:
        result = ds.evaluate_cells(cells, work_dir=work, shard=0, of=1, roll_fn=rolling, watchdog_file=dump)
        expected: list[tuple[Any, ...]] = []
        for cell in sorted(cells[1:], key=ds.cell_name):
            expected += [("arm", 600.0, False, True, dump), ("roll", ds.cell_name(cell)), ("cancel",)]
        assert events == expected
        assert (result["rolled"], result["skipped"]) == (2, 1)

        events.clear()

        def raising(cell: dict[str, Any]) -> dict[str, Any]:
            events.append(("roll", ds.cell_name(cell)))
            raise ValueError("a fake cell that raises")

        with pytest.raises(ValueError, match="a fake cell that raises"):
            ds.evaluate_cells(FAKE_CELLS[3:], work_dir=work, shard=0, of=1, roll_fn=raising, watchdog_file=dump)
        assert events == [("arm", 600.0, False, True, dump), ("roll", ds.cell_name(FAKE_CELLS[3])), ("cancel",)]


HEALTHY_SHARD = r'''
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import offline.demand_shift as ds
from tests.test_demand_shift_evaluate import _fake_roll

cells = [c for c in ds.declared_cells(stage=1)
         if c["file"] == "kn-hz" and c["arm"] == "random_fresh" and c["seed"] == 101][:5]


def roll(cell):
    time.sleep(0.4)  # each cell well inside the watchdog ...
    return _fake_roll(cell)


with open(sys.argv[3], "a", encoding="utf-8") as dump:
    result = ds.evaluate_cells(cells, work_dir=Path(sys.argv[2]), shard=0, of=1, roll_fn=roll, watchdog_seconds=1.0,
                               watchdog_file=dump)
    print("rolled", result["rolled"], flush=True)
    time.sleep(1.5)  # ... and a pause after the last cell, outside every cell, longer than the watchdog
print("alive", flush=True)
'''


def test_a_healthy_shard_longer_than_the_watchdog_is_never_killed(tmp_path: Path) -> None:
    """Scaled down: a 1-s watchdog, five cells of 0.4 s (2 s in all), then 1.5 s after the last cell -- a watchdog armed
    once per shard, or never cancelled, would cut it.  (On the code before C9 this passes trivially, there being no watchdog
    to cut anything; its red state is the cancel-removed mutant.)"""
    script = tmp_path / "healthy_shard.py"
    script.write_text(HEALTHY_SHARD, encoding="utf-8")
    dump = tmp_path / "dump.log"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = _run([sys.executable, str(script), str(REPO_ROOT), str(tmp_path / "w"), str(dump)], env=env, timeout=120)
    text = dump.read_text(encoding="utf-8") if dump.exists() else ""
    assert result.returncode == 0, f"{result.returncode}\n{result.stdout}\n{result.stderr}\n{text}"
    assert result.stdout.split() == ["rolled", "5", "alive"]
    assert not TIMEOUT_BANNER.search(text)


# ----------------------------------------------------------------------
# B.3 (d): the restart cap
# ----------------------------------------------------------------------


def test_the_restart_cap_stops_a_shard_that_hangs_on_every_attempt(tmp_path: Path) -> None:
    """The cap scaled down to 2: three attempts, each cut by the watchdog, two restarts logged, then the shard stops."""
    shard = _Shard(tmp_path, "hang_always", max_restarts=2)
    assert shard.result.returncode == 1, shard.explain()
    assert len(TIMEOUT_BANNER.findall(shard.log)) == 3, shard.explain()
    assert [(number, cap) for number, cap, _time, _frame in shard.restarts()] == [("1", "2"), ("2", "2")]
    assert len(shard.evaluate_calls()) == 3
    assert re.search(
        r"(?m)^SHARD 0 STOPPED: the watchdog cut it on 3 attempts; the cap of 2 restarts is reached$", shard.log
    ), shard.explain()
    assert re.search(r"(?m)^SHARD 0 EXIT: 1$", shard.log), shard.explain()


# ----------------------------------------------------------------------
# The defaults: 600 s, the shard's own log as the dump's file, 20 restarts
# ----------------------------------------------------------------------


def test_by_default_the_shard_runs_a_600_s_watchdog_writing_to_its_own_log(tmp_path: Path) -> None:
    shard = _Shard(tmp_path, "healthy", watchdog=None)
    assert shard.result.returncode == 0, shard.explain()
    (call,) = shard.evaluate_calls()
    assert " --watchdog-seconds 600 " in f" {call} "
    assert f" --watchdog-file {shard.log_path} " in f" {call} "
    assert ds.WATCHDOG_SECONDS == 600.0
    assert inspect.signature(ds.evaluate_cells).parameters["watchdog_seconds"].default == 600.0
    assert ds.build_parser().parse_args(["evaluate"]).watchdog_seconds == 600.0
    text = DRIVER.read_text(encoding="utf-8")
    assert "MAX_RESTARTS=${P92_MAX_RESTARTS:-20}" in text and "WATCHDOG_SECONDS=${P92_WATCHDOG_SECONDS:-600}" in text
