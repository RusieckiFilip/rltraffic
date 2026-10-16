"""P9.1's driver, ``offline/campaigns/p9_1_corrective.sh`` (``BRIEF_46`` 3.4: the campaign refuses to start without the
author's token exactly as ``offline/iql_correction.py``'s campaign does -- P5.2b's ``offline/campaigns/
p5_2b_correction.sh``).

No format is written.  The checks are the mechanism's: the token is a FILE the author creates, checked after every
refusal and consumed by ``rm`` before the first write; nothing whose failure must stop the run is piped
(``PROJECT_PLAN`` section 7, 2026-09-19); the stage and the worker count are validated before anything runs; the
evaluation's manifests are declared by ONE process before the shards start (a shard that declared its own could race
another on the same file); no second driver runs beside a first.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[1] / "offline" / "campaigns" / "p9_1_corrective.sh"


def _code() -> list[str]:
    return [line for line in DRIVER.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")]


def _branch(stage: str) -> str:
    """The body of one ``case`` branch of the stage dispatch."""
    code = "\n".join(_code())
    match = re.search(rf"^  {stage}\)\n(.*?)^    ;;", code, flags=re.MULTILINE | re.DOTALL)
    assert match, f"no {stage} branch"
    return match.group(1)


def test_the_driver_parses_and_runs_under_set_euo_pipefail() -> None:
    assert subprocess.run(["bash", "-n", str(DRIVER)], capture_output=True).returncode == 0
    assert "set -euo pipefail" in _code()


def test_every_refusal_precedes_the_token_and_the_token_is_consumed_before_any_stage() -> None:
    code = "\n".join(_code())
    token_check = code.index('[ -f "$TOKEN" ] || refuse')
    consumed = code.index('rm -- "$TOKEN"')
    assert code.index("\nstage_check\n") < token_check < consumed, "the CALL, not the function's definition"
    for refusal in ("uncommitted changes", "is not at $COMMIT", "DETACHED worktree", "another P9.1 stage",
                    "the module loads from"):
        assert code.index(refusal) < token_check, refusal
    first_stage = min(code.index(word) for word in ("run_logged collect_hz1x1", "run_logged train", "run_logged d4rl"))
    assert consumed < first_stage


def test_the_token_is_per_stage_and_lives_in_the_runs_directory() -> None:
    code = "\n".join(_code())
    assert 'TOKEN=$RUNS/TOKEN_$STAGE' in code and "RUNS=$OUTPUT/p9_1_runs" in code


def test_no_second_p9_1_stage_runs_beside_a_first() -> None:
    """Every stage command the driver launches is in the pgrep refusal, so a second driver is refused before its
    token -- a second evaluation would race the first on the same cell files."""
    code = "\n".join(_code())
    pattern = re.search(r"pgrep -f '([^']+)'", code)
    assert pattern, "no pgrep refusal"
    for command in ("train", "evaluate", "old-cells", "p3a-references", "collect-random-fresh", "declare",
                    "declare-evaluation", "d4rl_run"):
        line = f"/x/.venv/bin/python -P -m offline.corrective_round --repo-root /r --output-root /o {command} --y 1"
        if command == "d4rl_run":
            line = "/x/d4rl-calib-venv/bin/python -P -m calibration.d4rl_run --threads 1 train --arms iql"
        assert re.search(pattern.group(1), line), command
    assert not re.search(pattern.group(1), "/x/.venv/bin/python -P -m offline.corrective_round --output-root /o status")


def test_no_command_whose_failure_must_stop_the_run_is_piped() -> None:
    for line in _code():
        if re.search(r"\bcr (train|evaluate|collect-random-fresh|declare|declare-evaluation|old-cells|p3a-references)\b",
                     line) or "calibration.d4rl_run" in line:
            assert "|" not in line.replace("||", ""), line


def test_the_campaign_declares_the_evaluation_once_before_any_shard_and_shards_the_old_cells() -> None:
    body = _branch("campaign")
    assert body.index("run_logged train") < body.index("run_logged declare_evaluation cr declare-evaluation")
    assert body.index("cr declare-evaluation") < body.index("shards evaluate") < body.index("shards old-cells")
    assert body.index("shards old-cells") < body.index("run_logged p3a_references")
    code = "\n".join(_code())
    assert 'cr "$what" --shard "$k" --of "$WORKERS"' in code


def test_the_d4rl_stage_reruns_the_iql_arm_only_with_its_targets_off_beside_the_registered_artifact() -> None:
    body = _branch("d4rl")
    code = "\n".join(_code())
    assert "D4RL_PY=/home/filip/d4rl-calib-venv/bin/python" in code
    assert "D4RL_DATA=/home/filip/d4rl-calib-data/halfcheetah_medium_expert-v2.hdf5" in code
    assert "D4RL_OUT=/home/filip/d4rl-calib-out/p9_1" in code
    for piece in ("-m calibration.d4rl_run --threads 1 train", '--dataset "$D4RL_DATA"', '--out "$D4RL_OUT"',
                  "--arms iql", "--iql-target-dropout off",
                  '--compare-with "$WORK_TREE/docs/data/p8_3_d4rl_calibration.json"'):
        assert piece in body, piece
    assert "--seeds" not in body, "the registered five seeds, by default"


@pytest.mark.parametrize(("argv", "env", "message"), [
    (["fly", "abc"], {}, "unknown stage fly"),
    (["evaluate", "abc"], {}, "unknown stage evaluate"),
    (["campaign", "abc"], {"WORKERS": "7"}, "WORKERS must be 1-6"),
    (["campaign", "abc"], {"WORKERS": "0"}, "WORKERS must be 1-6"),
])
def test_a_bad_stage_or_worker_count_is_refused_before_anything_runs(argv: list[str], env: dict[str, str],
                                                                       message: str, tmp_path: Path) -> None:
    result = subprocess.run(["bash", str(DRIVER), *argv], capture_output=True, text=True, cwd=tmp_path,
                            env={**os.environ, **env})
    assert result.returncode == 2
    assert message in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_the_regime_is_p5_2s_default_and_no_canary_runs() -> None:
    code = "\n".join(_code())
    assert "unset CUBLAS_WORKSPACE_CONFIG" in code and "export OMP_NUM_THREADS=1" in code
    assert "canary" not in code and "power-check" not in code


def test_the_campaign_says_complete_only_when_the_status_computed_from_disk_says_so() -> None:
    """Every step exiting 0 is not the claim; the status's computed ``complete`` is (a cell left waiting, a missing
    P3a record or an old cell refused into a retry would otherwise read as done)."""
    body = _branch("campaign")
    status = body.index('cr status >"$LOGS/status.json" 2>"$LOGS/status.err" || status_code=$?')
    check = body.index('["complete"] is True else 1)')
    assert body.index("run_logged p3a_references") < status < check < body.index("COMPLETE")
    assert 'fail "the campaign is not complete on disk' in body


def test_the_campaign_keeps_a_unix_time_gpu_trace_from_before_training_to_the_end() -> None:
    """A suspension of the machine shows as a gap between two samples; the report flags the runs it interrupted."""
    body = _branch("campaign")
    code = "\n".join(_code())
    assert body.index("start_trace") < body.index("run_logged train")
    assert 'date -u +%s.%N' in code and 'nvidia-smi --query-gpu=memory.used,utilization.gpu' in code
    assert '"$LOGS/gpu_trace.csv"' in code and "trap stop_trace EXIT" in code


# ----------------------------------------------------------------------
# B.3 (BRIEF_46 Amendment B; DEFERRED 104): a worker without progress is restarted, nothing else is
# ----------------------------------------------------------------------

FAKE_WORKER = '''
import argparse, ctypes, sys, time
from pathlib import Path
from offline.stall_guard import StallGuard

parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("state")
parser.add_argument("--stall-dump", required=True)
args = parser.parse_args()
state = Path(args.state)
runs = int(state.read_text()) + 1 if state.exists() else 1
state.write_text(str(runs))
guard = StallGuard(args.stall_dump, seconds=1.0, rearm_seconds=0.0)
if args.mode == "stall-always" or (args.mode == "stall-once" and runs == 1):
    ctypes.PyDLL(None).sleep(60)  # a C-level wait holding the GIL, as Engine::~Engine() at its barrier
if args.mode == "fail":
    sys.exit(3)
if args.mode == "raise":
    raise RuntimeError("an ordinary failure: exit 1 and no banner")
if args.mode == "healthy":
    for _ in range(12):
        time.sleep(0.25)
        guard.progress()
guard.cancel()
print("done", runs)
'''


def _driver_function(name: str) -> str:
    match = re.search(rf"^{name}\(\) {{\n.*?^}}\n", DRIVER.read_text(encoding="utf-8"), flags=re.MULTILINE | re.DOTALL)
    assert match, f"no {name} function"
    return match.group(0)


def _driver_constant(name: str) -> str:
    match = re.search(rf"^{name}=.*$", DRIVER.read_text(encoding="utf-8"), flags=re.MULTILINE)
    assert match, f"no {name}"
    return match.group(0)


def _restarting(tmp_path: Path, mode: str, *, max_restarts: str | None = None) -> tuple[int, Path]:
    """The driver's own ``restarting`` function, cut out of the driver and run around the fake worker."""
    import sys

    logs = tmp_path / "logs"
    logs.mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER, encoding="utf-8")
    script = "\n".join([
        "set -euo pipefail", f"LOGS={logs}", _driver_constant("STALL_BANNER"), _driver_constant("MAX_RESTARTS"),
        *([f"MAX_RESTARTS={max_restarts}"] if max_restarts else []), _driver_function("restarting"),
        "code=0",
        f'restarting w {sys.executable} -P {worker} {mode} {tmp_path / "state"} >"$LOGS/w.log" 2>&1 || code=$?',
        'echo "exit=$code"'])
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    assert result.stdout.startswith("exit="), result.stderr
    return int(result.stdout.strip().split("=")[1]), logs


def test_the_cap_is_twenty_restarts_per_worker_and_the_banner_is_the_guards() -> None:
    from offline.stall_guard import BANNER

    assert _driver_constant("MAX_RESTARTS") == "MAX_RESTARTS=20"
    assert _driver_constant("STALL_BANNER") == f"STALL_BANNER='{BANNER}'"


def test_a_worker_blocked_in_a_c_level_wait_is_restarted_and_completes(tmp_path: Path) -> None:
    code, logs = _restarting(tmp_path, "stall-once")
    assert code == 0 and (tmp_path / "state").read_text() == "2"
    assert (logs / "w.stall0.txt").read_text().startswith("Timeout (")
    assert (logs / "w.stall1.txt").read_text() == ""
    log = (logs / "restarts.log").read_text()
    assert re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ w: no progress .* restart 1 of at most 20", log)
    assert "    Timeout (0:00:01)!" in log and "worker.py" in log and log.count(" restart ") == 1
    assert "done 2" in (logs / "w.log").read_text()


@pytest.mark.parametrize(("mode", "expected"), [("fail", 3), ("raise", 1)])
def test_a_failing_worker_is_not_restarted(mode: str, expected: int, tmp_path: Path) -> None:
    """Exit 3, and exit 1 from an exception -- faulthandler's own code, but no banner: one run, the code passed up."""
    code, logs = _restarting(tmp_path, mode)
    assert code == expected and (tmp_path / "state").read_text() == "1"
    assert not (logs / "restarts.log").exists() and (logs / "w.stall0.txt").read_text() == ""


def test_a_healthy_worker_is_never_killed(tmp_path: Path) -> None:
    code, logs = _restarting(tmp_path, "healthy")
    assert code == 0 and (tmp_path / "state").read_text() == "1"
    assert not (logs / "restarts.log").exists() and (logs / "w.stall0.txt").read_text() == ""


def test_a_worker_that_stalls_past_the_cap_stops_the_stage(tmp_path: Path) -> None:
    code, logs = _restarting(tmp_path, "stall-always", max_restarts="2")
    assert code != 0 and (tmp_path / "state").read_text() == "3"
    log = (logs / "restarts.log").read_text()
    assert log.count(" restart ") == 2 and "after 2 restarts; stopping" in log


def test_every_worker_that_rolls_cityflow_episodes_runs_under_the_restart_loop_and_nothing_else_does() -> None:
    """Route (i) and (ii) (the evaluate shards), the old cells, and P3a's registered references; not the collection
    (B.3: it runs without the remedy), the training, the declarations or the D4RL re-run."""
    code = "\n".join(_code())
    assert 'restarting "${what}_shard$k" cr "$what" --shard "$k" --of "$WORKERS"' in code
    body = _branch("campaign")
    assert "run_logged p3a_references restarting p3a_references cr p3a-references" in body
    assert body.count("restarting") == 1
    for stage in ("collect", "d4rl"):
        assert "restarting" not in _branch(stage)
    for line in _code():
        if re.search(r"\bcr (train|declare|declare-evaluation|collect-random-fresh)\b", line):
            assert "restarting" not in line, line
