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
