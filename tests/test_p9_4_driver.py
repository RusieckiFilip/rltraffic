"""P9.4's driver, ``offline/campaigns/p9_4_mix50_fresh.sh`` (plan section 7): three stages run by the author in tmux,
each session under its own token, every refusal before the token, OMP/MKL at one thread with CUBLAS_WORKSPACE_CONFIG
unset, at most six workers, every CityFlow worker under the stall restart loop, nothing whose failure must stop the run
piped, and the stop file never created or removed by the driver.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[1] / "offline" / "campaigns" / "p9_4_mix50_fresh.sh"


def _code() -> list[str]:
    return [line for line in DRIVER.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")]


def _text() -> str:
    return "\n".join(_code())


def _branch(stage: str) -> str:
    match = re.search(rf"^  {stage}\)\n(.*?)^    ;;", _text(), flags=re.MULTILINE | re.DOTALL)
    assert match, f"no {stage} branch"
    return match.group(1)


def test_the_driver_parses_and_runs_under_set_euo_pipefail() -> None:
    assert subprocess.run(["bash", "-n", str(DRIVER)], capture_output=True).returncode == 0
    assert "set -euo pipefail" in _code()


def test_every_refusal_precedes_the_token_and_the_token_is_consumed_before_any_stage() -> None:
    code = _text()
    token_check = code.index('[ -f "$TOKEN" ] || refuse')
    consumed = code.index('rm -- "$TOKEN"')
    assert code.index("\nstage_check\n") < token_check < consumed
    for refusal in ("uncommitted changes", "is not at $COMMIT", "DETACHED worktree", "another P9.4 stage",
                    "the module loads from", "no session starts while it exists", "P94_SESSION_HOURS must be",
                    "WORKERS must be 1-6"):
        assert code.index(refusal) < token_check, refusal
    first_stage = min(code.index(word) for word in ("run_logged train", "restarting route_i", "run_logged sumo"))
    assert consumed < first_stage


def test_the_token_is_per_stage_and_lives_in_the_runs_directory() -> None:
    code = _text()
    assert "TOKEN=$RUNS/TOKEN_$STAGE" in code and "RUNS=$OUTPUT/p9_4_runs" in code


def test_the_token_line_is_the_one_the_report_reads_each_sessions_code_commit_from() -> None:
    from offline.corrective_report import TOKEN_LINE

    line = 'echo "token consumed; stage $STAGE $STAMP from $WORK_TREE at $COMMIT; logs $LOGS"'
    assert line in _text()
    assert TOKEN_LINE.fullmatch("token consumed; stage sumo 20261120T101010Z from /home/filip/rltraffic-p94-run at "
                                "0123456789abcdef0123456789abcdef01234567; logs /x/logs/20261120T101010Z")


def test_no_second_p9_4_stage_runs_beside_a_first() -> None:
    code = _text()
    pattern = code[code.index("if pgrep -f '"):].split("'")[1]
    for command in ("train", "route-i", "declare-route-ii", "route-ii", "sumo"):
        assert re.search(pattern, f"/home/filip/rltraffic/.venv/bin/python -P -m offline.p9_4_mix50_fresh --repo-root "
                                  f"x {command} --x 1"), command


def test_no_command_whose_failure_must_stop_the_run_is_piped() -> None:
    for line in _code():
        stripped = line.strip()
        if re.match(r"^(run_logged|restarting|cr|shards)\b", stripped):
            assert "|" not in stripped.replace("||", ""), stripped


def test_the_regime_is_one_thread_and_cublas_unset_and_workers_at_most_six() -> None:
    code = _text()
    assert "export OMP_NUM_THREADS=1" in code and "export MKL_NUM_THREADS=1" in code
    assert "unset CUBLAS_WORKSPACE_CONFIG" in code
    assert '[[ "$WORKERS" =~ ^[1-6]$ ]] || refuse' in code


def test_the_stages_run_their_module_commands_in_order() -> None:
    train = _branch("train")
    assert train.index("cr train --hours") < train.index("run_logged train cr train --hours")
    cityflow = _branch("cityflow")
    assert cityflow.index("restarting route_i") < cityflow.index("run_logged declare_route_ii") < cityflow.index(
        "shards route-ii")
    sumo = _branch("sumo")
    assert "run_logged sumo cr sumo --workers \"$WORKERS\" --deadline-unix \"$DEADLINE\"" in sumo


def test_every_cityflow_worker_runs_under_the_stall_restart_loop_and_nothing_else_does() -> None:
    code = _text()
    assert "restarting route_i cr route-i" in code
    assert 'restarting "${what}_shard$k" cr "$what" --shard "$k" --of "$WORKERS"' in code
    for stage in ("train", "sumo"):
        assert "restarting" not in _branch(stage)


def test_the_driver_never_creates_or_removes_the_stop_file_or_anything_under_output() -> None:
    code = _text()
    assert not re.search(r"(rm|touch|mv|cp)\b[^\n]*STOP", code)
    assert [line for line in _code() if re.search(r"\brm\b", line)] == ['rm -- "$TOKEN"']
    assert not re.search(r"\bmv\b", code)


@pytest.mark.parametrize(("argv", "env", "message"), [
    (["evaluate", "0" * 40], {}, "unknown stage"),
    (["sumo", "0" * 40], {"WORKERS": "7"}, "WORKERS must be 1-6"),
    (["sumo", "0" * 40], {"P94_SESSION_HOURS": "0"}, "P94_SESSION_HOURS must be"),
    (["sumo", "0" * 40], {"P94_SESSION_HOURS": "abc"}, "P94_SESSION_HOURS must be"),
])
def test_a_bad_stage_worker_count_or_session_hours_is_refused_before_anything_runs(
    argv: list[str], env: dict[str, str], message: str, tmp_path: Path
) -> None:
    import os

    result = subprocess.run(["bash", str(DRIVER), *argv], capture_output=True, text=True,
                            env={**os.environ, **env}, cwd=tmp_path)
    assert result.returncode == 2 and message in result.stderr


def test_the_usage_says_first_how_many_hours_a_session_runs_and_where_the_capture_goes() -> None:
    header = DRIVER.read_text(encoding="utf-8").split("set -euo pipefail")[0]
    assert "P94_SESSION_HOURS" in header and "hours without a break" in header
    assert "/home/filip/rltraffic/output/p9_4_runs/" in header and 'echo "DRIVER EXIT $?"' in header
