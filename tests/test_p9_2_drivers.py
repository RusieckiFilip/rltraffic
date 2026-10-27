"""The two P9.2 campaign drivers (``offline/campaigns/p9_2_sumo_anchors.sh``, ``p9_2_demand_shift.sh``): they parse, they
fail closed, and a refused start creates nothing.

A driver refuses BEFORE its token is consumed -- on a dirty run tree (every cell records ``git_dirty`` and would be
refused) or on a missing token -- and exits 2 having written no log, no work directory and no draw.  Which refusal fires
first depends on the tree the test runs in, so the test asserts the exit code, the ``REFUSING TO START`` line and that
the runs root it was pointed at is still empty; it never needs, and never writes, a real token.

The token's consumption (``BRIEF_47`` Amendment A.2.3) is tested by running each driver for real in a CLEAN scratch git tree
that holds a copy of it, with a stand-in interpreter (``P92_PYTHON``) that answers each subcommand with a configured exit
code and a stand-in ``tmux`` that records the windows it would open -- so the driver's own control flow is what is tested:
the token is consumed, and shard windows opened, only past a gate (guard) record that ``check-gate`` (``check-guard``)
says authorises.  Whether a record authorises is the module's rule, tested in ``tests/test_sumo_anchor_rescore.py`` and
``tests/test_demand_shift_evaluate.py``.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVERS = ("p9_2_sumo_anchors.sh", "p9_2_demand_shift.sh")


@pytest.mark.parametrize("name", DRIVERS)
def test_the_driver_parses(name: str) -> None:
    result = subprocess.run(["bash", "-n", str(REPO_ROOT / "offline" / "campaigns" / name)], capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", DRIVERS)
def test_a_start_without_a_token_refuses_and_creates_nothing(tmp_path: Path, name: str) -> None:
    runs = tmp_path / "runs"
    output = tmp_path / "output"
    env = {**os.environ, "P92_RUNS": str(runs), "P92_OUTPUT_ROOT": str(output), "P92_SESSION": "p92-test-never"}
    result = subprocess.run(["bash", str(REPO_ROOT / "offline" / "campaigns" / name)], capture_output=True, text=True,
                            env=env, timeout=60)
    assert result.returncode == 2
    assert "REFUSING TO START" in result.stderr
    assert not runs.exists() and not output.exists()


@pytest.mark.parametrize("name", DRIVERS)
def test_the_driver_fails_closed_and_pins_one_thread(name: str) -> None:
    text = (REPO_ROOT / "offline" / "campaigns" / name).read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1" in text
    # the token is consumed only after every refusable step
    assert text.index('rm -f "$TOKEN"') > text.index("canary")
    assert text.index('rm -f "$TOKEN"') > text.index("gate" if "sumo" in name else "guard")


# ----------------------------------------------------------------------
# The token is consumed only past a record that authorises (BRIEF_47 Amendment A.2.3)
# ----------------------------------------------------------------------

SPEC = {
    "p9_2_sumo_anchors.sh": {"token": "TOKEN_p92_sumo", "record": ("p9_2", "anchors", "replica_gate.json"),
                             "check": "check-gate", "run": "gate"},
    "p9_2_demand_shift.sh": {"token": "TOKEN_p92_shift", "record": ("p9_2", "demand_shift", "guard.json"),
                             "check": "check-guard", "run": "guard"},
}

STUB_PYTHON = """#!/usr/bin/env bash
# A stand-in interpreter: it records its arguments -- and, beside them, the directory it was called from (BRIEF_47
# Amendment E) -- and answers by subcommand (its last argument).
printf '%s\\n' "$*" >> "$STUB_LOG"
printf '%s\\t%s\\n' "$(pwd -P)" "$*" >> "$STUB_WHERE_LOG"
last="${!#}"
case "$last" in
  canary) echo "canary 0.80 s (stand-in)"; exit 0 ;;
  check-gate|check-guard) exit "${STUB_CHECK_EXIT:-0}" ;;
  gate|guard) exit "${STUB_RUN_EXIT:-0}" ;;
  *) exit 0 ;;
esac
"""

STUB_TMUX = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$STUB_TMUX_LOG"
"""


def _executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class _Harness:
    """One driver in a clean scratch git tree, its runs and output roots, a scratch MAIN tree (``P92_MAIN_TREE``: the shift
    driver materialises from its main tree, which on a host without ``/home/filip/rltraffic`` must be given -- ``BRIEF_47``
    Amendment E), and the two stand-ins."""

    def __init__(self, tmp_path: Path, name: str) -> None:
        self.name = name
        self.tree = tmp_path / "tree"
        (self.tree / "offline" / "campaigns").mkdir(parents=True)
        shutil.copyfile(REPO_ROOT / "offline" / "campaigns" / name, self.tree / "offline" / "campaigns" / name)
        (self.tree / "docs" / "data").mkdir(parents=True)
        shutil.copyfile(REPO_ROOT / "docs" / "data" / "p9_2_demand_shift_audit.json",
                        self.tree / "docs" / "data" / "p9_2_demand_shift_audit.json")
        identity = ["-c", "user.name=driver-test", "-c", "user.email=driver-test@example.invalid"]
        for command in (["init", "-q"], [*identity, "add", "-A"], [*identity, "commit", "-q", "-m", "scratch run tree"]):
            subprocess.run(["git", "-C", str(self.tree), *command], check=True, capture_output=True)
        assert subprocess.run(["git", "-C", str(self.tree), "status", "--porcelain"], capture_output=True,
                              text=True, check=True).stdout == ""
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        _executable(self.bin / "python", STUB_PYTHON)
        _executable(self.bin / "tmux", STUB_TMUX)
        self.runs = tmp_path / "runs"
        self.output = tmp_path / "output"
        self.runs.mkdir()
        self.token = self.runs / SPEC[name]["token"]
        self.token.write_text("driver test token\n", encoding="utf-8")
        self.python_log = tmp_path / "python.log"
        self.where_log = tmp_path / "python_where.log"
        self.tmux_log = tmp_path / "tmux.log"
        self.main_tree = tmp_path / "main_tree"
        self.main_tree.mkdir()

    def write_record(self) -> None:
        record = self.output.joinpath(*SPEC[self.name]["record"])
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text('{"passed": false}\n', encoding="utf-8")

    def __call__(self, *, run_exit: int = 0, check_exit: int = 0) -> subprocess.CompletedProcess[str]:
        env = {key: value for key, value in os.environ.items() if key not in ("STAGE", "IQL_T_PINS", "SHARDS")}
        env.update(PATH=f"{self.bin}:{os.environ['PATH']}", P92_PYTHON=str(self.bin / "python"), P92_RUNS=str(self.runs),
                   P92_OUTPUT_ROOT=str(self.output), P92_SESSION="p92-test-never", STUB_LOG=str(self.python_log),
                   STUB_TMUX_LOG=str(self.tmux_log), STUB_RUN_EXIT=str(run_exit), STUB_CHECK_EXIT=str(check_exit),
                   P92_MAIN_TREE=str(self.main_tree), STUB_WHERE_LOG=str(self.where_log))
        return subprocess.run(["bash", str(self.tree / "offline" / "campaigns" / self.name)], capture_output=True,
                              text=True, env=env, timeout=60)

    def python_calls(self) -> list[str]:
        return self.python_log.read_text(encoding="utf-8").splitlines() if self.python_log.exists() else []

    def tmux_calls(self) -> list[str]:
        return self.tmux_log.read_text(encoding="utf-8").splitlines() if self.tmux_log.exists() else []

    def calls_with_cwd(self) -> list[tuple[str, str]]:
        """``(the stand-in's working directory, its arguments)`` per call."""
        lines = self.where_log.read_text(encoding="utf-8").splitlines() if self.where_log.exists() else []
        return [tuple(line.split("\t", 1)) for line in lines]  # type: ignore[misc]


@pytest.mark.parametrize("name", DRIVERS)
@pytest.mark.parametrize(
    ("record_on_disk", "run_exit", "check_exit", "expected_exit", "token_kept"),
    [
        pytest.param(True, 0, 2, 2, True, id="failed-record-on-disk-the-reviewers-case"),
        pytest.param(True, 0, 0, 0, False, id="authorising-record-on-disk"),
        pytest.param(False, 2, 0, 2, True, id="fresh-run-fails"),
        pytest.param(False, 0, 2, 2, True, id="fresh-run-exits-0-but-its-record-does-not-authorise"),
        pytest.param(False, 0, 0, 0, False, id="fresh-run-passes"),
    ],
)
def test_the_token_is_consumed_only_past_a_record_that_authorises(
    tmp_path: Path, name: str, record_on_disk: bool, run_exit: int, check_exit: int, expected_exit: int,
    token_kept: bool,
) -> None:
    harness = _Harness(tmp_path, name)
    if record_on_disk:
        harness.write_record()
    result = harness(run_exit=run_exit, check_exit=check_exit)
    assert result.returncode == expected_exit, result.stdout + result.stderr
    assert harness.token.is_file() is token_kept
    assert len(harness.tmux_calls()) == (0 if token_kept else 3)
    calls = harness.python_calls()
    ran_gate = any(call.endswith(f" {SPEC[name]['run']}") for call in calls)
    asked = any(call.endswith(f" {SPEC[name]['check']}") for call in calls)
    assert ran_gate is not record_on_disk, calls
    assert asked is (run_exit == 0), calls
    if token_kept and run_exit == 0:
        assert "the token is kept" in result.stderr


# ----------------------------------------------------------------------
# The main tree is the driver's to be given (BRIEF_47 Amendment E, C10)
# ----------------------------------------------------------------------


def test_the_shift_driver_materialises_from_the_main_tree_it_is_given(tmp_path: Path) -> None:
    """The driver ``cd``s into its main tree to materialise the draws (DEFERRED 55).  Given ``P92_MAIN_TREE`` -- as it is
    given ``P92_PYTHON`` -- it must materialise from THAT tree: here a scratch one, so the test never enters the real main
    tree, and runs on a host (CI) where ``/home/filip/rltraffic`` does not exist."""
    harness = _Harness(tmp_path, "p9_2_demand_shift.sh")
    result = harness(run_exit=0, check_exit=0)
    assert result.returncode == 0, result.stdout + result.stderr
    materialise = [(cwd, call) for cwd, call in harness.calls_with_cwd() if call.endswith(" materialise")]
    assert len(materialise) == 1, harness.calls_with_cwd()
    assert Path(materialise[0][0]) == harness.main_tree.resolve(), materialise


def test_unset_p92_main_tree_leaves_the_campaigns_main_tree_unchanged(tmp_path: Path) -> None:
    """The pin (Amendment E, E.2.3): the driver's ONE assignment of ``MAIN_TREE`` is the override with
    ``/home/filip/rltraffic`` as its default -- evaluated by bash itself, unset and set, so no such directory is needed."""
    text = (REPO_ROOT / "offline" / "campaigns" / "p9_2_demand_shift.sh").read_text(encoding="utf-8")
    assignments = [line for line in text.splitlines() if line.startswith("MAIN_TREE=")]
    assert assignments == ["MAIN_TREE=${P92_MAIN_TREE:-/home/filip/rltraffic}"]
    env = {key: value for key, value in os.environ.items() if key != "P92_MAIN_TREE"}
    probe = [assignments[0], 'printf "%s" "$MAIN_TREE"']
    unset = subprocess.run(["bash", "-c", "; ".join(probe)], env=env, capture_output=True, text=True, check=True)
    assert unset.stdout == "/home/filip/rltraffic"
    given = subprocess.run(["bash", "-c", "; ".join(probe)], env={**env, "P92_MAIN_TREE": str(tmp_path)},
                           capture_output=True, text=True, check=True)
    assert given.stdout == str(tmp_path)
