"""P5.2b (``BRIEF_44`` §4.3, plan §8): the correction run's driver ``offline/campaigns/p5_2b_correction.sh``.

* **The driver's text, comments stripped:** strict mode; ``-P`` on every interpreter call; the regime (``OMP`` / ``MKL``
  = 1, ``CUBLAS_WORKSPACE_CONFIG`` unset) and the tracer refusal before any interpreter; the pre-flight branch needs no
  token and leaves before the run's checks; in the run, every refusal precedes the token's removal -- the pinned
  pre-flight, the manifest's absence, P8.2's ``power-check``, the module's ``check`` -- and the stages follow it in the
  plan's order (canary open, train, canary close, (i), (ii), manifest, report); the evaluation stages run under their
  timeouts with three attempts; the timeouts come from a pinned record, never typed; nothing printed names an episode
  quantity; the header documents the foreground start, the token, the power rule, the hang rule and the restart rule.
* **Executed on a snapshot clone** (no data is needed): no pinned pre-flight → refused, nothing created; a record at
  another digest → refused; a tree at another commit → refused; a tree with uncommitted changes → the pre-flight
  refuses; a tracer variable → refused before any interpreter; the power check failing → refused before the token, the
  token left in place.

GATES: the executed tests need the main tree's interpreter (``/home/filip/rltraffic/.venv``) and skip naming it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.test_p7_3d_campaign_path import MAIN_INTERPRETER, _git, _snapshot_clone, _substitute

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "offline" / "campaigns" / "p5_2b_correction.sh"
_TRACER_VARIABLES = ("COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG", "PYTHONTRACEMALLOC", "PYTHONDEVMODE",
                     "PYTHONMALLOC", "PYTHONPROFILEIMPORTTIME")


def _text() -> str:
    return DRIVER.read_text(encoding="utf-8")


def _code() -> list[str]:
    return [line for line in _text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _first(lines: list[str], needle: str) -> int:
    hits = [i for i, line in enumerate(lines) if needle in line]
    assert hits, f"the driver has no line with {needle!r}"
    return hits[0]


def _preflight_branch(lines: list[str]) -> tuple[int, int]:
    start = _first(lines, 'if [ "$MODE" = preflight ]')
    return start, start + next(i for i, line in enumerate(lines[start:]) if line.strip() == "fi")


def test_the_driver_runs_in_strict_mode() -> None:
    assert "set -euo pipefail" in _code()


def test_every_interpreter_call_carries_minus_p() -> None:
    calls = [line for line in _code() if '"$PY"' in line and "-x" not in line]
    assert len(calls) >= 6, calls
    for line in calls:
        assert '"$PY" -P' in line, line


def test_the_regime_and_the_tracer_refusal_come_before_any_interpreter_starts() -> None:
    lines = _code()
    first_call = _first(lines, '"$PY" -P')
    for needle in ("export OMP_NUM_THREADS=1", "export MKL_NUM_THREADS=1", "unset CUBLAS_WORKSPACE_CONFIG",
                   *_TRACER_VARIABLES):
        assert _first(lines, needle) < first_call, needle


def test_the_preflight_branch_needs_no_token_and_leaves_before_the_runs_checks() -> None:
    lines = _code()
    start, end = _preflight_branch(lines)
    block = lines[start:end + 1]
    assert any("preflight" in line and "--stamp" in line for line in block)
    assert any("canary --phase open" in line for line in block) and any("canary --phase close" in line
                                                                        for line in block)
    assert any(line.strip().startswith("exit") for line in block)
    assert not any("TOKEN" in line for line in block)
    assert end < _first(lines, 'rm -- "$TOKEN"')


def test_every_refusal_of_the_run_precedes_the_token_and_the_stages_follow_it_in_the_plans_order() -> None:
    lines = _code()
    consumed = _first(lines, 'rm -- "$TOKEN"')
    refusals = [i for i, line in enumerate(lines) if "refuse " in line and "refuse()" not in line]
    assert refusals and max(refusals) < consumed
    removals = [i for i, line in enumerate(lines) if re.search(r"\brm\b", line) and "TOKEN" in line]
    assert removals == [consumed], [lines[i] for i in removals]
    for needle in ('[ "$PREFLIGHT_SHA256" != UNSET ]', '[ ! -e "$MANIFEST" ]', "power-check", "check \"${ROOTS[@]}\"",
                   '[ -f "$TOKEN" ]'):
        assert _first(lines, needle) < consumed, needle
    _, end = _preflight_branch(lines)
    stages = [i for i in (_first(lines[end:], "canary --phase open") + end, _first(lines[end:], " train ") + end,
                          _first(lines[end:], "canary --phase close") + end, _first(lines, "evaluate-p5-2"),
                          _first(lines, "evaluate-p8-4b"), _first(lines, " manifest "), _first(lines, " report "))]
    assert stages == sorted(stages) and stages[0] > consumed, [lines[i] for i in stages]


def test_the_power_check_precedes_the_modules_check_and_both_precede_the_token() -> None:
    lines = _code()
    _, end = _preflight_branch(lines)
    calls = [i for i, line in enumerate(lines) if "power-check" in line]
    assert len(calls) == 1 and '"$PY" -P -m offline.compute_latency power-check || refuse ' in lines[calls[0]]
    assert end < calls[0] < _first(lines, 'check "${ROOTS[@]}"') < _first(lines, 'rm -- "$TOKEN"')


def test_the_evaluation_stages_run_under_their_timeouts_with_three_attempts() -> None:
    lines = _code()
    assert "MAX_ATTEMPTS=3" in lines
    for stage, variable in (("evaluate-p5-2", "$T_EVAL_I"), ("evaluate-p8-4b", "$T_EVAL_II")):
        line = lines[_first(lines, stage)]
        assert line.lstrip().startswith("attempt ") and f'"{variable}" "$MAX_ATTEMPTS"' in line, line
    assert any("timeout -k " in line for line in lines[_first(lines, "attempt() {"):])


def test_the_run_reads_its_timeouts_from_a_preflight_record_pinned_by_path_and_digest() -> None:
    text = _text()
    record = re.search(r"^PREFLIGHT_RECORD=(\S+)$", text, flags=re.M)
    digest = re.search(r"^PREFLIGHT_SHA256=(\S+)$", text, flags=re.M)
    assert record and digest
    assert record.group(1) == "UNSET" or re.fullmatch(r"p5_2b_runs/preflight_\d{8}T\d{6}Z/preflight\.json",
                                                      record.group(1))
    assert digest.group(1) == "UNSET" or re.fullmatch(r"[0-9a-f]{64}", digest.group(1))
    assert (record.group(1) == "UNSET") == (digest.group(1) == "UNSET")
    lines = _code()
    assert _first(lines, '[ "$PREFLIGHT_SHA256" != UNSET ]') < _first(lines, "sha256sum")
    assert _first(lines, "sha256sum") < _first(lines, "timeouts --preflight-record")
    assert not re.search(r"\$T_(TRAIN|EVAL_I|EVAL_II)=[0-9]", text) and not re.search(r"T_(TRAIN|EVAL_I+)=[0-9]", text)


def test_the_liveness_check_matches_an_interpreter_running_the_module_not_any_mention_of_it() -> None:
    lines = [line for line in _code() if "pgrep" in line]
    assert len(lines) == 1
    assert "pgrep -f '^[^ ]*python[^ ]* -P -m offline[.]iql_correction'" in lines[0]


def test_the_driver_prints_no_outcome() -> None:
    printed = [line for line in _code() if re.match(r"\s*(echo|printf)\b", line)]
    for line in printed:
        assert not re.search(r"\b(att|reward|return|travel|queue|vehicle)", line.lower()), line


def test_the_header_documents_the_start_the_token_the_power_the_hang_and_the_restart_rules() -> None:
    header = "\n".join(line for line in _text().splitlines() if line.startswith("#"))
    for needle in ("tee -i -a", "${PIPESTATUS[0]}", "TOKEN_correction", "mains power", "Best Performance", "--preflight",
                   "DEFERRED 104", "worktree add --detach", "move output/p5_2b aside", "Amendment A"):
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
    driver = clone / "offline" / "campaigns" / "p5_2b_correction.sh"
    text = driver.read_text(encoding="utf-8")
    text = _substitute(text, "MAIN=/home/filip/rltraffic\n", f"MAIN={main}\n", 1)
    text = _substitute(text, "PY=$MAIN/.venv/bin/python\n", f"PY={MAIN_INTERPRETER}\n", 1)
    driver.write_text(text, encoding="utf-8")
    _git("-c", "user.email=t@t", "-c", "user.name=t", "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "-am",
         "the sandbox's roots", cwd=clone)
    return clone, main


def _run(clone: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(clone / "offline" / "campaigns" / "p5_2b_correction.sh"), *args],
                          capture_output=True, text=True, timeout=300, check=False, env=env)


def _pin(clone: Path, record: str, digest: str) -> None:
    """Rewrite the sandbox driver's two pin lines, whatever they hold, and commit (the run tree must be clean)."""
    driver = clone / "offline" / "campaigns" / "p5_2b_correction.sh"
    text = re.sub(r"^PREFLIGHT_RECORD=\S+$", f"PREFLIGHT_RECORD={record}", driver.read_text(), count=1, flags=re.M)
    text = re.sub(r"^PREFLIGHT_SHA256=\S+$", f"PREFLIGHT_SHA256={digest}", text, count=1, flags=re.M)
    driver.write_text(text)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "-c", "core.hooksPath=/dev/null", "commit", "--quiet",
         "--allow-empty", "-am", "the sandbox's pin", cwd=clone)


def _listing(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


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
    _pin(clone, "p5_2b_runs/preflight_20261006T000000Z/preflight.json", "0" * 64)
    completed = _run(clone, _git("rev-parse", "HEAD", cwd=clone).strip())
    assert completed.returncode == 2, completed.stderr
    assert "pinned digest" in completed.stderr
    assert list(main.iterdir()) == []


def test_the_run_refuses_a_tree_that_is_not_at_the_named_commit(sandbox: tuple[Path, Path]) -> None:
    clone, main = sandbox
    completed = _run(clone, _git("rev-parse", "HEAD~1", cwd=clone).strip())
    assert completed.returncode == 2, completed.stderr
    assert "is not at" in completed.stderr
    assert list(main.iterdir()) == []


def test_the_preflight_refuses_a_tree_with_uncommitted_changes(sandbox: tuple[Path, Path]) -> None:
    clone, main = sandbox
    target = clone / "offline" / "iql_correction.py"
    target.write_text(target.read_text() + "\n# an uncommitted change\n")
    completed = _run(clone, "--preflight")
    assert completed.returncode == 2, completed.stderr
    assert "uncommitted" in completed.stderr
    assert list(main.iterdir()) == []


@pytest.mark.parametrize("variable", ["COVERAGE_PROCESS_START", "PYTHONDEVMODE"])
def test_a_set_tracer_variable_is_refused_and_nothing_is_created(sandbox: tuple[Path, Path], variable: str) -> None:
    clone, main = sandbox
    completed = _run(clone, "--preflight", env={**os.environ, variable: "1"})
    assert completed.returncode == 2, completed.stderr[-1500:]
    assert variable in completed.stderr
    assert list(main.iterdir()) == []


def test_the_run_refuses_before_the_token_when_the_power_check_fails(sandbox: tuple[Path, Path]) -> None:
    clone, main = sandbox
    runs = main / "output" / "p5_2b_runs"
    record = runs / "preflight_20261006T000000Z" / "preflight.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"format_version": "p5.2b-preflight/1.0", "status": "COMPLETE",
                                  "reproduce": {"p5_2_path": {"reproduced": True}, "p8_4b_path": {"reproduced": True}},
                                  "estimate": {"timeouts": {"train": 9000.0, "evaluate_p5_2": 4000.0,
                                                            "evaluate_p8_4b": 4500.0}}}))
    token = runs / "TOKEN_correction"
    token.write_text("")
    before = _listing(main)
    driver = clone / "offline" / "campaigns" / "p5_2b_correction.sh"
    driver.write_text(_substitute(driver.read_text(encoding="utf-8"), "-m offline.compute_latency power-check",
                                  "-c 'import sys; sys.exit(\"power regime: the sandbox refuses (simulated)\")'", 1),
                      encoding="utf-8")
    _pin(clone, "p5_2b_runs/preflight_20261006T000000Z/preflight.json", hashlib.sha256(record.read_bytes()).hexdigest())
    _git("checkout", "--quiet", "--detach", cwd=clone)
    completed = _run(clone, _git("rev-parse", "HEAD", cwd=clone).strip())
    assert completed.returncode == 2, completed.stderr[-1500:]
    assert "power regime" in completed.stderr
    assert token.is_file(), "the token was consumed although the power check refused"
    assert _listing(main) == before


# ----------------------------------------------------------------------
# BRIEF_44 Amendment B: B1.3 (no second realisation) and B2 (the pre-flight G1 accepted, pinned)
# ----------------------------------------------------------------------

#: The pre-flight gate G1 read from disk and kept pinned (Amendment B, B0 and B2).
G1_PREFLIGHT_RECORD = "p5_2b_runs/preflight_20261006T195405Z/preflight.json"
G1_PREFLIGHT_SHA256 = "a280452734494479f6d2941b09ff12c16825155080404ee7ce84b95752c4bb4b"

#: A stand-in for the interpreter: it logs every call and answers the three queries the driver reads from stdout.
_STAND_IN = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_LOG"
case " $* " in
  *" -c "*) printf '%s\\n' "$PYTHONPATH/offline/iql_correction.py" ;;
  *" offline.iql_correction timeouts "*) echo "5000 3000 3000" ;;
  *" --stage training "*) echo "$FAKE_TRAINING" ;;
  *" --stage canaries "*) echo "$FAKE_CANARIES" ;;
esac
exit 0
"""


def _stages(log: Path) -> list[str]:
    """The stages the driver started, in order, read from the stand-in's log."""
    out = []
    for line in log.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if "-m" not in tokens:
            continue
        module, command = tokens[tokens.index("-m") + 1], tokens[tokens.index("-m") + 2]
        if module == "offline.compute_latency" and command == "canary":
            out.append(f"canary {tokens[tokens.index('--phase') + 1]}")
        elif module == "offline.iql_correction" and command == "status":
            out.append(f"status {tokens[tokens.index('--stage') + 1]}")
        else:
            out.append(command)
    return out


@pytest.mark.parametrize("training, canaries", [("complete", "closing_pending"), ("absent", "absent"),
                                                ("complete", "complete")])
def test_b1_3_the_restart_after_a_missing_closing_canary_takes_it_alone_and_trains_nothing(
    sandbox: tuple[Path, Path], tmp_path: Path, training: str, canaries: str
) -> None:
    """Executed, with every interpreter call answered by a stand-in that logs it: a training complete on disk whose
    closing canary is missing gets ``close-late`` and then the closing canary alone -- no opening canary, no ``train``
    -- before (i), (ii), the manifest and the report; a fresh run still takes canary, train, canary; a run whose
    training and both canaries are complete takes none of them."""
    clone, main = sandbox
    stand_in = tmp_path / "bin" / "stand_in"
    stand_in.parent.mkdir()
    stand_in.write_text(_STAND_IN, encoding="utf-8")
    stand_in.chmod(0o755)
    driver = clone / "offline" / "campaigns" / "p5_2b_correction.sh"
    driver.write_text(_substitute(driver.read_text(encoding="utf-8"), f"PY={MAIN_INTERPRETER}\n", f"PY={stand_in}\n", 1),
                      encoding="utf-8")
    runs = main / "output" / "p5_2b_runs"
    record = runs / "preflight_20261006T000000Z" / "preflight.json"
    record.parent.mkdir(parents=True)
    record.write_text("{}\n", encoding="utf-8")
    (runs / "TOKEN_correction").write_text("", encoding="utf-8")
    _pin(clone, "p5_2b_runs/preflight_20261006T000000Z/preflight.json", hashlib.sha256(record.read_bytes()).hexdigest())
    _git("checkout", "--quiet", "--detach", cwd=clone)
    log = tmp_path / "calls.log"
    completed = _run(clone, _git("rev-parse", "HEAD", cwd=clone).strip(),
                     env={**os.environ, "FAKE_LOG": str(log), "FAKE_TRAINING": training, "FAKE_CANARIES": canaries})
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "COMPLETE" in completed.stdout
    stages = _stages(log)
    tail = ["evaluate-p5-2", "evaluate-p8-4b", "manifest", "report"]
    assert [stage for stage in stages if stage in tail] == tail, stages
    if (training, canaries) == ("complete", "closing_pending"):
        assert "train" not in stages and "canary open" not in stages, stages
        assert stages.count("close-late") == 1 and stages.count("canary close") == 1, stages
        assert stages.index("close-late") < stages.index("canary close") < stages.index("evaluate-p5-2"), stages
        assert (main / "output" / "p5_2b" / "logs" / "canary_close_late.attempt1.log").is_file()
    elif training == "absent":
        assert stages.index("canary open") < stages.index("train") < stages.index("canary close") < stages.index(
            "evaluate-p5-2"), stages
        assert "close-late" not in stages, stages
    else:
        assert not {"canary open", "train", "close-late", "canary close"} & set(stages), stages


def test_b1_3_the_late_branch_reads_the_canaries_state_and_marks_before_it_canaries() -> None:
    lines = _code()
    late = _first(lines, "closing_pending")
    assert "--stage canaries" in lines[late] and lines[late].lstrip().startswith("elif"), lines[late]
    mark = _first(lines, "close-late")
    assert late < mark < _first(lines[mark:], "canary --phase close") + mark < _first(lines, "evaluate-p5-2")
    assert "|| fail" in lines[mark]


def test_b1_3_the_header_documents_that_a_complete_training_is_never_trained_again() -> None:
    header = "\n".join(line for line in _text().splitlines() if line.startswith("#"))
    for needle in ("close-late", "closing canary", "late", "never trained again", "Amendment B"):
        assert needle in header, needle


def test_b2_the_driver_pins_the_preflight_g1_accepted() -> None:
    text = _text()
    assert re.search(rf"^PREFLIGHT_RECORD={re.escape(G1_PREFLIGHT_RECORD)}$", text, flags=re.M)
    assert re.search(rf"^PREFLIGHT_SHA256={G1_PREFLIGHT_SHA256}$", text, flags=re.M)
