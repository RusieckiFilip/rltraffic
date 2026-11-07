"""P9.1's no-progress watchdog (``offline/stall_guard.py``; ``BRIEF_46`` Amendment B, B.3; ``DEFERRED`` 104).

No format.  The protocol under test: a worker armed with :class:`offline.stall_guard.StallGuard` that makes no progress
for its window writes faulthandler's dump -- first characters :data:`offline.stall_guard.BANNER` -- and exits 1, even
when its main thread is blocked inside C holding the GIL (as ``Engine::~Engine()`` is at its barrier); progress
re-arms it, at most once per re-arm interval; ``CityFlowEnv.reset`` and ``step`` are progress; the three rolling
subcommands arm it from ``--stall-dump``.  The restart loop that reads the dump is the driver's
(``tests/test_corrective_round_driver.py``).

Every guard that is really armed runs in a SUBPROCESS: an armed watchdog left in the pytest process would end it.
In-process tests replace faulthandler's two calls with recorders.
"""

from __future__ import annotations

import faulthandler
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import offline.corrective_round as cr
import offline.stall_guard as sg

REPO = Path(__file__).resolve().parents[1]


def _python(code: str, tmp_path: Path, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent(code), encoding="utf-8")
    return subprocess.run([sys.executable, "-P", str(script), *args], capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, "PYTHONPATH": str(REPO)})


class _Recorder:
    def __init__(self) -> None:
        self.armed: list[tuple[float, bool, Any]] = []
        self.cancelled = 0

    def arm(self, timeout: float, repeat: bool = False, file: Any = None, exit: bool = False) -> None:
        self.armed.append((float(timeout), bool(exit), file))

    def cancel(self) -> None:
        self.cancelled += 1


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(faulthandler, "dump_traceback_later", rec.arm)
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", rec.cancel)
    return rec


def test_the_constants_are_b3s_with_amendment_b2s_window() -> None:
    """B.3's guard, its window corrected from 600 s to 120 s by ``BRIEF_46`` Amendment B.2: re-armed at most once per
    10 s, a stall is cut after 110-120 s without a reset or a step."""
    assert (sg.STALL_SECONDS, sg.REARM_SECONDS, sg.BANNER, sg.PROGRESS_METHODS) == (120.0, 10.0, "Timeout (",
                                                                                     ("reset", "step"))


def test_a_worker_blocked_in_c_holding_the_gil_dumps_its_stack_with_the_banner_and_exits_1(tmp_path: Path) -> None:
    """``ctypes.PyDLL`` calls libc's ``sleep`` WITHOUT releasing the GIL -- the condition of the destructor's hang."""
    dump = tmp_path / "stall.txt"
    result = _python("""
        import ctypes, sys
        from offline.stall_guard import StallGuard
        guard = StallGuard(sys.argv[1], seconds=1.0, rearm_seconds=0.0)
        ctypes.PyDLL(None).sleep(60)
        print("unreachable")
    """, tmp_path, str(dump))
    assert result.returncode == 1 and "unreachable" not in result.stdout
    text = dump.read_text(encoding="utf-8")
    assert text.startswith(sg.BANNER) and "worker.py" in text


def test_progress_keeps_a_healthy_worker_alive_past_its_window(tmp_path: Path) -> None:
    """Three seconds of work in a one-second window, progress every quarter second: never ended, dump empty."""
    dump = tmp_path / "stall.txt"
    result = _python("""
        import sys, time
        from offline.stall_guard import StallGuard
        guard = StallGuard(sys.argv[1], seconds=1.0, rearm_seconds=0.0)
        for _ in range(12):
            time.sleep(0.25)
            guard.progress()
        guard.cancel()
        print("finished")
    """, tmp_path, str(dump))
    assert result.returncode == 0 and result.stdout.strip() == "finished", result.stderr
    assert dump.read_text(encoding="utf-8") == ""


def test_the_guard_arms_on_construction_and_re_arms_at_most_once_per_interval(tmp_path: Path,
                                                                             recorder: _Recorder) -> None:
    now = [100.0]
    guard = sg.StallGuard(tmp_path / "d.txt", seconds=600.0, rearm_seconds=10.0, clock=lambda: now[0])
    assert [(t, e) for t, e, _ in recorder.armed] == [(600.0, True)]
    for t, expected in ((105.0, 1), (109.999, 1), (110.0, 2), (119.0, 2), (125.0, 3)):
        now[0] = t
        guard.progress()
        assert len(recorder.armed) == expected, t
    assert all(f is recorder.armed[0][2] and not f.closed for _, _, f in recorder.armed)
    guard.arm()
    assert len(recorder.armed) == 4
    guard.cancel()
    assert recorder.cancelled == 1 and recorder.armed[0][2].closed


class _Base:
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        return {"reset": seed}

    def step(self, action: Any) -> tuple[float, bool, bool, dict[str, Any]]:
        return 0.0, False, False, {"base": action}


class _Env(_Base):
    def step(self, action: Any) -> tuple[float, bool, bool, dict[str, Any]]:
        return -1.0, False, True, {"env": action}


def test_wrapped_methods_report_progress_then_run_unchanged_and_are_restored_exactly(tmp_path: Path,
                                                                                    recorder: _Recorder) -> None:
    """``_Env`` inherits ``reset`` and defines ``step``: while wrapped both report progress and return what the
    originals return; restored, ``reset`` is inherited again and ``step`` is the same function object."""
    calls: list[str] = []
    guard = sg.StallGuard(tmp_path / "d.txt", rearm_seconds=0.0)
    guard.progress = lambda: calls.append("progress")  # type: ignore[method-assign]
    own_step = _Env.__dict__["step"]
    restore = sg.wrap_progress(guard, [_Env])
    env = _Env()
    assert env.reset(seed=42) == {"reset": 42} and calls == ["progress"]
    assert env.step([3]) == (-1.0, False, True, {"env": [3]}) and calls == ["progress", "progress"]
    assert _Base().reset(seed=1) == {"reset": 1} and calls == ["progress", "progress"]
    restore()
    assert "reset" not in _Env.__dict__ and _Env.__dict__["step"] is own_step
    env.reset(seed=0)
    env.step([0])
    assert calls == ["progress", "progress"]
    guard.cancel()


def test_install_wraps_the_real_cityflow_env_for_the_process(tmp_path: Path) -> None:
    pytest.importorskip("cityflow")
    dump = tmp_path / "stall.txt"
    result = _python("""
        import sys
        from envs.cityflow_env import CityFlowEnv
        before = {name: getattr(CityFlowEnv, name) for name in ("reset", "step")}
        from offline.stall_guard import install
        guard = install(sys.argv[1], seconds=30.0)
        print(all(getattr(CityFlowEnv, name) is not before[name] for name in before))
        guard.cancel()
    """, tmp_path, str(dump))
    assert result.returncode == 0 and result.stdout.strip() == "True", result.stderr


@pytest.mark.parametrize("command", [["evaluate", "--shard", "2", "--of", "6"], ["old-cells", "--shard", "1", "--of", "6"],
                                     ["p3a-references"]])
def test_the_three_rolling_subcommands_arm_the_guard_before_they_roll(command: list[str], tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[Any] = []
    monkeypatch.setattr(sg, "install", lambda path, **kw: events.append(("install", Path(path))))
    monkeypatch.setattr(cr, "evaluate", lambda roots, **kw: events.append("evaluate") or {})
    monkeypatch.setattr(cr, "assert_evaluation_declared", lambda roots, *names: None)
    monkeypatch.setattr(cr, "roll_old_cells", lambda roots, cells=None: events.append("old-cells") or {})
    monkeypatch.setattr(cr, "record_registered_references", lambda roots: events.append("p3a-references") or {})
    dump = tmp_path / "stall.txt"
    argv = ["--output-root", str(tmp_path), "--corpus-root", str(tmp_path), "--draws-root", str(tmp_path), *command,
            "--stall-dump", str(dump)]
    assert cr.main(argv) == 0
    assert events == [("install", dump), command[0]]


@pytest.mark.parametrize(("extra", "expected"), [([], 120.0), (["--stall-seconds", "45"], 45.0)])
def test_the_rolling_subcommands_hand_their_window_to_the_guard(extra: list[str], expected: float, tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """The driver passes ``--stall-seconds`` (its ``P91_STALL_SECONDS``, default 120) to every worker; without it the
    worker arms the module's default."""
    armed: list[tuple[Path, float]] = []
    monkeypatch.setattr(sg, "install", lambda path, *, seconds: armed.append((Path(path), float(seconds))))
    monkeypatch.setattr(cr, "assert_evaluation_declared", lambda roots, *names: None)
    monkeypatch.setattr(cr, "roll_old_cells", lambda roots, cells=None: {})
    dump = tmp_path / "stall.txt"
    argv = ["--output-root", str(tmp_path), "--corpus-root", str(tmp_path), "--draws-root", str(tmp_path), "old-cells",
            "--stall-dump", str(dump), *extra]
    assert cr.main(argv) == 0
    assert armed == [(dump, expected)]


def test_no_other_subcommand_takes_a_stall_dump(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``collect`` runs without the remedy (B.3) and training and the declarations roll no CityFlow episode."""
    for command in (["train"], ["collect-random-fresh", "--scenario", "hz1x1"], ["declare-evaluation"], ["status"]):
        with pytest.raises(SystemExit, match="^2$"):
            cr.build_parser().parse_args(["--output-root", "o", "--corpus-root", "c", "--draws-root", "d", *command,
                                          "--stall-dump", str(tmp_path / "x")])
        assert "unrecognized arguments: --stall-dump" in capsys.readouterr().err, command
