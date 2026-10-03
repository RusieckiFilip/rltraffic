"""P8.2 C3 (``BRIEF_43`` Amendment A, Q10 and A3.1): the timing run's driver ``offline/campaigns/p8_2_latency.sh`` and
the G1 pre-flight it runs.

* **The pre-flight's record**, through ``run_preflight`` with fake processes: the measured process wall times, the
  per-scenario timeouts and the canary timeout derived from them, the run's expected duration; a throttled canary
  makes it FAILED, and ``timeouts_from_preflight`` refuses anything but a COMPLETE record.
* **The driver's text, comments stripped:** strict mode; ``-P`` on every interpreter call; the regime (``OMP`` /
  ``MKL`` = 1, ``CUBLAS_WORKSPACE_CONFIG`` unset) exported before any interpreter starts; the pre-flight branch needs
  no token and leaves before the run's checks; in the run, every refusal precedes the token's consumption and
  ``run-all`` follows it; the timeouts come from a pre-flight record pinned by path and sha256, never typed; nothing
  printed names an episode quantity; the header carries the foreground start, the token, the power rule and the hang
  rule.
* **Executed on a snapshot clone** (no data is needed): the run refuses while no pre-flight is pinned and creates
  nothing; a tree at another commit is refused; the pre-flight refuses a tree with uncommitted changes.

GATES: the executed tests need the main tree's interpreter (``/home/filip/rltraffic/.venv``) and skip naming it.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import offline.compute_latency as cl
from tests.test_p7_3d_campaign_path import MAIN_INTERPRETER, _git, _snapshot_clone, _substitute

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "offline" / "campaigns" / "p8_2_latency.sh"


# ----------------------------------------------------------------------
# The pre-flight's record, with fake processes
# ----------------------------------------------------------------------


def _record_script(path: Path, payload: dict[str, Any], *, sleep_s: float = 0.0) -> list[str]:
    script = textwrap.dedent(
        f"""
        import json, os, pathlib, time
        time.sleep({sleep_s!r})
        path = pathlib.Path({str(path)!r})
        tmp = path.parent / (path.name + ".tmp")
        tmp.write_text(json.dumps({payload!r}))
        os.link(tmp, path)
        tmp.unlink()
        """
    )
    return [sys.executable, "-c", script]


def _episodes() -> list[list[int]]:
    return [[1_000_000] * cl.DECISIONS_PER_EPISODE for _ in cl.TIMING_DRAWS]


_KWARGS = {
    "draws": cl.TIMING_DRAWS, "prompt": None, "factory": "fake",
    "regime": {"torch_num_threads": 1, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": None},
    "machine": {"cpu_model": "fake"}, "load_before": [0.0, 0.0, 0.0], "load_after": [0.0, 0.0, 0.0], "git": {},
}


def _preflight(tmp_path: Path, *, open_seconds: float = 0.8) -> tuple[int, Path]:
    rows = [cl.row_by_id("hz1x1.dt_k20"), cl.row_by_id("grid4x4.dt_nomix_h4")]

    def command_for(row: cl.LatencyRow, device: str, out_dir: Path) -> list[str]:
        payload = cl.build_record(row, device, _episodes(), **_KWARGS)
        return _record_script(out_dir / f"{row.row_id}_{device}.json", payload, sleep_s=0.3)

    def canary_command_for(phase: str, out_dir: Path) -> list[str]:
        seconds = open_seconds if phase == "open" else 0.8
        payload = cl.build_canary_record(phase, seconds, reproduced=True, git={})
        return _record_script(out_dir / f"canary_{phase}.json", payload, sleep_s=0.1)

    code = cl.run_preflight(stamp="20261003T000000Z", rows=rows, runs_root=tmp_path, command_for=command_for,
                            canary_command_for=canary_command_for, provisional_timeout_s=60.0, git={})
    return code, tmp_path / "preflight_20261003T000000Z" / "preflight.json"


def test_the_preflight_derives_the_timeouts_and_the_expected_duration_from_its_measured_processes(tmp_path: Path) -> None:
    code, path = _preflight(tmp_path)
    assert code == 0
    record = json.loads(path.read_text())
    assert record["format_version"] == cl.PREFLIGHT_FORMAT_VERSION and record["status"] == "COMPLETE"
    measured = {(item["row"], item["device"]): item["seconds"] for item in record["measured_process_seconds"]}
    assert set(measured) == {("hz1x1.dt_k20", "cpu"), ("hz1x1.dt_k20", "cuda"),
                             ("grid4x4.dt_nomix_h4", "cpu"), ("grid4x4.dt_nomix_h4", "cuda")}
    assert all(seconds > 0.3 for seconds in measured.values())
    assert record["timeouts_s"] == cl.derive_timeouts(measured)
    canary_walls = [attempt["seconds"] for outcome in record["outcomes"] if outcome["label"].startswith("canary_")
                    for attempt in outcome["attempts"] if attempt["record_written"]]
    assert len(canary_walls) == 2
    assert record["canary_timeout_s"] == float(max(120, math.ceil(3 * max(canary_walls))))
    assert record["expected_duration"] == cl.expected_duration(measured, max(canary_walls))
    assert cl.timeouts_from_preflight(path) == {
        "hz1x1": record["timeouts_s"]["hz1x1"], "grid4x4": record["timeouts_s"]["grid4x4"],
        "canary": record["canary_timeout_s"],
    }


def test_a_throttled_canary_fails_the_preflight_and_its_record_yields_no_timeouts(tmp_path: Path) -> None:
    code, path = _preflight(tmp_path, open_seconds=2.6)
    assert code == 1
    record = json.loads(path.read_text())
    assert record["status"] == "FAILED" and "throttled" in record["reason"]
    assert "timeouts_s" not in record and record["measured_process_seconds"] == []
    with pytest.raises(ValueError, match="COMPLETE"):
        cl.timeouts_from_preflight(path)


# ----------------------------------------------------------------------
# The driver's text
# ----------------------------------------------------------------------


def _text() -> str:
    return DRIVER.read_text(encoding="utf-8")


def _code() -> list[str]:
    return [line for line in _text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _first(lines: list[str], needle: str) -> int:
    hits = [i for i, line in enumerate(lines) if needle in line]
    assert hits, f"the driver has no line with {needle!r}"
    return hits[0]


def test_the_driver_runs_in_strict_mode() -> None:
    assert "set -euo pipefail" in _code()


def test_every_interpreter_call_carries_minus_p() -> None:
    calls = [line for line in _code() if '"$PY"' in line and "-x" not in line]
    assert len(calls) >= 4
    for line in calls:
        assert '"$PY" -P' in line, line


def test_the_regime_is_exported_before_any_interpreter_starts() -> None:
    lines = _code()
    first_call = _first(lines, '"$PY" -P')
    for needle in ("export OMP_NUM_THREADS=1", "export MKL_NUM_THREADS=1", "unset CUBLAS_WORKSPACE_CONFIG"):
        assert _first(lines, needle) < first_call, needle


def test_the_preflight_branch_needs_no_token_and_leaves_before_the_runs_checks() -> None:
    lines = _code()
    start = _first(lines, 'if [ "$MODE" = preflight ]')
    end = start + next(i for i, line in enumerate(lines[start:]) if line.strip() == "fi")
    block = lines[start:end + 1]
    assert any("preflight --stamp" in line for line in block)
    assert any(line.strip().startswith("exit") for line in block)
    assert not any("TOKEN" in line for line in block)
    assert end < _first(lines, 'rm -- "$TOKEN"')


def test_every_refusal_of_the_run_precedes_the_token_and_run_all_follows_it() -> None:
    lines = _code()
    consumed = _first(lines, 'rm -- "$TOKEN"')
    refusals = [i for i, line in enumerate(lines) if "refuse " in line and "refuse()" not in line]
    assert refusals and max(refusals) < consumed
    assert _first(lines, "run-all") > consumed
    assert _first(lines, '[ -f "$TOKEN" ]') < consumed
    assert _first(lines, '[ ! -e "$MANIFEST" ]') < consumed


def test_the_run_reads_its_timeouts_from_a_preflight_record_pinned_by_path_and_digest() -> None:
    text = _text()
    record = re.search(r"^PREFLIGHT_RECORD=(\S+)$", text, flags=re.M)
    digest = re.search(r"^PREFLIGHT_SHA256=(\S+)$", text, flags=re.M)
    assert record and digest
    assert record.group(1) == "UNSET" or re.fullmatch(r"p8_2_runs/preflight_\d{8}T\d{6}Z/preflight\.json",
                                                      record.group(1))
    assert digest.group(1) == "UNSET" or re.fullmatch(r"[0-9a-f]{64}", digest.group(1))
    assert (record.group(1) == "UNSET") == (digest.group(1) == "UNSET")
    lines = _code()
    assert _first(lines, '[ "$PREFLIGHT_SHA256" != UNSET ]') < _first(lines, "sha256sum")
    assert _first(lines, "sha256sum") < _first(lines, "timeouts --preflight-record")
    assert not re.search(r"--timeout-(hz1x1|grid4x4) [0-9]", text), "a timeout typed into the driver"


def test_the_driver_prints_no_outcome() -> None:
    printed = [line for line in _code() if re.match(r"\s*(echo|printf)\b", line)]
    for line in printed:
        assert not re.search(r"\b(att|reward|return|travel|queue|vehicle)", line.lower()), line


def test_the_header_documents_the_foreground_start_the_token_the_power_rule_and_the_hang_rule() -> None:
    header = "\n".join(line for line in _text().splitlines() if line.startswith("#"))
    for needle in ("tee -i -a", "${PIPESTATUS[0]}", "TOKEN_latency", "mains power", "--preflight", "DEFERRED 104",
                   "worktree add --detach"):
        assert needle in header, needle


# ----------------------------------------------------------------------
# Executed on a snapshot clone
# ----------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[Path, Path]:
    if not MAIN_INTERPRETER.is_file():
        pytest.skip(f"needs the main tree's interpreter at {MAIN_INTERPRETER}")
    clone = _snapshot_clone(tmp_path)
    main = tmp_path / "main"
    main.mkdir()
    driver = clone / "offline" / "campaigns" / "p8_2_latency.sh"
    text = driver.read_text(encoding="utf-8")
    text = _substitute(text, "MAIN=/home/filip/rltraffic\n", f"MAIN={main}\n", 1)
    text = _substitute(text, "PY=$MAIN/.venv/bin/python\n", f"PY={MAIN_INTERPRETER}\n", 1)
    driver.write_text(text, encoding="utf-8")
    _git("-c", "user.email=t@t", "-c", "user.name=t", "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "-am",
         "the sandbox's roots", cwd=clone)
    return clone, main


def _run(clone: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(clone / "offline" / "campaigns" / "p8_2_latency.sh"), *args],
                          capture_output=True, text=True, timeout=300, check=False)


def _pin(clone: Path, record: str, digest: str) -> None:
    """Rewrite the sandbox driver's two pin lines, whatever they hold, and commit (the run tree must be clean)."""
    driver = clone / "offline" / "campaigns" / "p8_2_latency.sh"
    text = re.sub(r"^PREFLIGHT_RECORD=\S+$", f"PREFLIGHT_RECORD={record}", driver.read_text(), count=1, flags=re.M)
    text = re.sub(r"^PREFLIGHT_SHA256=\S+$", f"PREFLIGHT_SHA256={digest}", text, count=1, flags=re.M)
    driver.write_text(text)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "-c", "core.hooksPath=/dev/null", "commit", "--quiet",
         "--allow-empty", "-am", "the sandbox's pin", cwd=clone)


def test_the_run_refuses_while_no_preflight_is_pinned_and_creates_nothing(sandbox: tuple[Path, Path]) -> None:
    clone, main = sandbox
    _pin(clone, "UNSET", "UNSET")
    completed = _run(clone, _git("rev-parse", "HEAD", cwd=clone).strip())
    assert completed.returncode == 2, completed.stderr
    assert "PREFLIGHT" in completed.stderr
    assert list(main.iterdir()) == []


def test_the_run_refuses_a_pinned_preflight_record_that_is_absent_or_at_another_digest(
    sandbox: tuple[Path, Path]
) -> None:
    clone, main = sandbox
    _pin(clone, "p8_2_runs/preflight_20261003T000000Z/preflight.json", "0" * 64)
    completed = _run(clone, _git("rev-parse", "HEAD", cwd=clone).strip())
    assert completed.returncode == 2, completed.stderr
    assert "pinned digest" in completed.stderr
    assert list(main.iterdir()) == []


def test_the_run_refuses_a_tree_that_is_not_at_the_named_commit(sandbox: tuple[Path, Path]) -> None:
    clone, main = sandbox
    other = _git("rev-parse", "HEAD~1", cwd=clone).strip()
    completed = _run(clone, other)
    assert completed.returncode == 2, completed.stderr
    assert "is not at" in completed.stderr
    assert list(main.iterdir()) == []


def test_the_preflight_refuses_a_tree_with_uncommitted_changes(sandbox: tuple[Path, Path]) -> None:
    clone, main = sandbox
    target = clone / "offline" / "compute_latency.py"
    target.write_text(target.read_text() + "\n# an uncommitted change\n")
    completed = _run(clone, "--preflight")
    assert completed.returncode == 2, completed.stderr
    assert "uncommitted" in completed.stderr
    assert list(main.iterdir()) == []
