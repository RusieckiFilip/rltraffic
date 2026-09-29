"""P5.3c C2 (``BRIEF_42`` §4 T-driver): the training driver ``offline/campaigns/p5_3c_train.sh`` -- its text, and its
refusals EXECUTED.

* **Comment-free text assertions, per mode.**  ``train``: check-inputs (with the timing record) and the pre-token resume
  scan, then the canary, the traps and the token; after it the start directory, ``record-canary``, per run the resume
  decision, the attempt marker and the training; then the K = 20 measurement, the manifest (re-verified) and the record,
  LAST.  ``timing``: no token at all, every write under ``fenced_timing/<stamp>/``.  Both: ``-P`` on every interpreter
  call, ``set -euo pipefail``, nothing deleted but the token, the run worktree AND its commit enforced, the working
  directory the run tree (plan Q14), the regime (OMP/MKL one thread, ``CUBLAS_WORKSPACE_CONFIG`` unset), ``FAILED`` on
  every path after the token through an ``EXIT`` trap keyed on a success flag, written with ``printf`` before any echo,
  the liveness guard naming this module, and no outcome printed.  The canary's TIMING half (``CANARY_MAX_SECONDS`` =
  2.0 s) as text AND executed on a stub interpreter (Amendment B, B3.2(b)).
* **The header's documented lines** -- the two-step foreground start with ``tee -i -a`` and ``${PIPESTATUS[0]}``.
* **The driver EXECUTED** on a sandbox (``offline/campaigns/p7_3c_finetune.sh``'s pattern, ``tests/
  test_p7_3c_finetune_driver.py``): a committed snapshot clone registered as the run tree; every WRITABLE root in
  ``tmp_path``; the published K = 20 checkpoints and P4.7's manifest LINKED read-only from the main tree, because
  ``check-inputs`` verifies them by digest and no synthetic stand-in can carry those digests; the corpus left at P4's
  absolute root, which ``check-inputs`` only reads (its statistics record that root); the training and the
  timing slot replaced by stubs that train nothing, so no test can train whatever breaks.  **The canary is real in ONE
  test** (no token) and stubbed with a valid line elsewhere -- ``BRIEF_42`` §4: at most four real CityFlow episodes in the
  whole suite.

GATES -- each ``skip`` names what it consumes: the main tree's interpreter, ``setsid``, CUDA (``check-inputs`` refuses
without it), the corpus under the main tree's ``datasets_v11``, the published checkpoints under its ``output/``, and
``nvidia-smi`` for the timing mode.  The executed tests start the driver, whose liveness guard refuses while another run
is live; they are run only when none is, and pytest is invoked from a command whose text does not carry that pattern.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from tests.test_p7_3d_campaign_path import _commit_clone, _git, _snapshot_clone, _substitute

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "offline" / "campaigns" / "p5_3c_train.sh"
TIMING_STAMP = "20260928T230000Z"

TRAIN_CALL = (
    'PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep train --run "$1" --device cuda --output-root "$OUTPUT" '
    '--corpus-root "$CORPUS"'
)
TRAIN_STUB = "sh -c 'echo \"STUB train (tests/test_p5_3c_train_driver.py): nothing trained\"; exit 97'"
SLOT_CALL = (
    'PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep timing run --output-root "$OUTPUT" --corpus-root '
    '"$CORPUS" --stamp "$STAMP" --slot "$1"'
)
SLOT_STUB = "sh -c 'echo \"STUB timing run (tests/test_p5_3c_train_driver.py): nothing trained\"; exit 97'"
CANARY_CALL = (
    'PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_calibration --draws-root "$DRAWS" --output-root "$OUTPUT" '
    '--work-dir "$TRAINING" canary'
)
#: A canary line in the grammar ``record-canary`` parses, carrying the reference facts (the stub is not a measurement).
CANARY_STUB = (
    "echo 'canary 0.75 s {\"att_horizon\": 247.75089149261333, \"decisions\": 360, \"local_return\": -32648.0, "
    "\"two_routes_agree\": true}'"
)


def _text() -> str:
    return DRIVER.read_text(encoding="utf-8")


def _code(text: str) -> str:
    """The driver without its comment lines -- what the shell executes, and nothing a comment could satisfy."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _function(code: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}$", code, flags=re.MULTILINE | re.DOTALL)
    assert match, f"the driver defines no function {name}()"
    return match.group(1)


def _in_order(body: str, order: list[tuple[str, str]]) -> None:
    positions = []
    for label, needle in order:
        assert body.count(needle) == 1, f"{label}: expected ONE {needle!r}, found {body.count(needle)}"
        positions.append(body.index(needle))
    assert positions == sorted(positions), [label for label, _needle in order]


def _main_tree(text: str) -> Path:
    match = re.search(r"^MAIN=(\S+)$", text, flags=re.MULTILINE)
    assert match, "the driver declares no MAIN="
    return Path(match.group(1))


# ----------------------------------------------------------------------------------------------------------------------
# The text, comments stripped
# ----------------------------------------------------------------------------------------------------------------------


def test_train_mode_checks_everything_before_the_token_and_writes_the_record_last() -> None:
    """*Mutation:* the resume scan moved after the token (or the record before the manifest) -> this dies."""
    body = _function(_code(_text()), "run_train")
    _in_order(
        body,
        [
            ("check-inputs, with the timing record", '--data-dir "$DATA" --timing "$TIMING"'),
            ("the pre-token resume scan", "resume-decision --all"),
            ("the canary", "canary_both_halves"),
            ("the traps", "install_traps"),
            ("the token check", 'if [ ! -f "$TOKEN" ]; then'),
            ("the token consumed", 'rm -f "$TOKEN"'),
            ("the start directory", 'mkdir -p "$START_DIR"'),
            ("the canary recorded", 'record-canary --line "$CANARY_LINE"'),
            ("the per-run decision", 'resume-decision --run "$RUN"'),
            ("the attempt marker", 'attempt --run "$RUN"'),
            ("the training", 'train_one "$RUN"'),
            ("the K = 20 measurement", "compare-k20 --output-root"),
            ("the manifest", "context_sweep manifest --output-root"),
            ("the manifest verified", "sha256sum -c --quiet SHA256SUMS_p5_3c_train.txt"),
            ("the record, last", "context_sweep record --output-root"),
            ("success", "SUCCESS=1"),
        ],
    )
    assert '[ "$N_LINES" -ne "$EXPECTED_MANIFEST_LINES" ]' in body
    assert re.search(r"^EXPECTED_MANIFEST_LINES=60$", _code(_text()), flags=re.MULTILINE)


def test_timing_mode_takes_no_token_and_writes_only_under_its_fenced_stamp() -> None:
    body = _function(_code(_text()), "run_timing")
    assert "TOKEN" not in body
    assert "rm " not in body
    _in_order(
        body,
        [
            ("check-inputs, no timing record", 'check-inputs --output-root "$OUTPUT" --corpus-root "$CORPUS" --data-dir "$DATA")'),
            ("a stamp is never reused", 'if [ -e "$STAMP_DIR" ]; then'),
            ("the canary", "canary_both_halves"),
            ("the traps", "install_traps"),
            ("the stamp directory", 'mkdir -p "$STAMP_DIR"'),
            ("the first slot", "sample_slot alone"),
            ("the same-seed repeat", "sample_slot repeat"),
            ("the summary", "timing summarize --output-root"),
            ("success", "SUCCESS=1"),
        ],
    )
    assert re.search(r'^\s*STAMP_DIR=\$FENCED/\$STAMP$', body, flags=re.MULTILINE)
    assert re.search(r"^FENCED=\$TRAINING/fenced_timing$", _code(_text()), flags=re.MULTILINE)


def test_every_interpreter_call_carries_minus_P() -> None:
    """Every token after ``"$PY"`` is ``-P`` -- except the ``];`` that closes the interpreter's existence check."""
    code = _code(_text())
    following = re.findall(r'"\$PY"\s+(\S+)', code)
    calls = [token for token in following if not token.startswith("]")]
    assert len(calls) >= 12, following
    assert set(calls) == {"-P"}, following
    assert len(following) - len(calls) == 1 and 'if [ ! -x "$PY" ]; then' in code


def test_the_regime_is_one_thread_and_cublas_workspace_config_unset() -> None:
    code = _code(_text())
    assert re.search(r"^export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1$", code, flags=re.MULTILINE)
    assert re.search(r"^unset CUBLAS_WORKSPACE_CONFIG$", code, flags=re.MULTILINE)


def test_strict_mode_and_nothing_deleted_but_the_token() -> None:
    code = _code(_text())
    assert re.search(r"^set -euo pipefail$", code, flags=re.MULTILINE)
    removals = re.findall(r"\brm\b[^\n]*", code)
    assert removals == ['rm -f "$TOKEN"'], removals
    assert not re.search(r"\bmv\b", code)


def test_the_run_worktree_its_commit_and_the_working_directory_are_enforced() -> None:
    code = _code(_text())
    assert re.search(r"^RUN_TREE=/home/filip/rltraffic-p53c-run$", code, flags=re.MULTILINE)
    assert re.search(r"^IMPLEMENTER_TREE=/home/filip/rltraffic-p53c$", code, flags=re.MULTILINE)
    assert 'if [ "$WORK_TREE" = "$IMPLEMENTER_TREE" ]; then' in code
    assert 'if [ "$WORK_TREE" != "$RUN_TREE" ]; then' in code
    assert 'if [ "$HEAD_COMMIT" != "$EXPECTED_COMMIT" ]; then' in code
    assert '[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]' in code
    # Q14: the working directory is the run tree, set before the first interpreter call.
    assert code.index('cd "$WORK_TREE"') < code.index('"$PY" -P')
    assert 'cd "$MAIN"' not in code


def test_failed_is_written_on_every_path_after_the_token() -> None:
    code = _code(_text())
    on_exit = _function(code, "on_exit")
    assert 'if [ "$SUCCESS" -ne 1 ]' in on_exit and "printf" in on_exit and "FAILED" in on_exit
    fail = _function(code, "fail")
    assert fail.index("printf") < fail.index("echo")
    traps = _function(code, "install_traps")
    assert "trap on_exit EXIT" in traps and "trap on_signal INT TERM HUP" in traps
    signal = _function(code, "on_signal")
    assert "kill -- -$$" in signal and "printf" in signal


def test_the_liveness_guard_names_this_module_and_the_group_leader_and_sigign_are_checked() -> None:
    code = _code(_text())
    assert re.search(r"pgrep -f 'python\.\*offline\\\.\([^)]*\bcontext_sweep\b[^)]*\)'", code), "liveness pattern"
    assert '"$(ps -o pgid= -p $$ | tr -d \' \')" != "$$"' in code
    assert "SigIgn" in code


def test_the_device_sampler_starts_before_its_slot_and_can_never_outlive_it() -> None:
    """The stubbed-slot hang of 2026-09-29: a SIGTERM that reaches the child before its exec is lost.

    *Mutations:* the start's wait for ``nvidia-smi`` removed, or the stop's SIGKILL escalation removed -> this dies (and
    the executed timing test, whose slot ends at once, can hang into its timeout).
    """
    code = _code(_text())
    start = _function(code, "start_sampler")
    assert 'until [ "$(cat "/proc/$SAMPLER/comm" 2>/dev/null)" = "nvidia-smi" ]; do' in start
    stop = _function(code, "stop_sampler")
    assert 'kill -9 "$pid"' in stop and '"$tries" -lt 40' in stop and "sleep 0.25" in stop
    assert stop.index('kill "$pid"') < stop.index('kill -9 "$pid"') < stop.index('wait "$pid"')
    slot = _function(code, "sample_slot")
    assert slot.index("start_sampler") < slot.index("timing run") < slot.index("stop_sampler")
    assert 'kill -9 "$SAMPLER"' in _function(code, "on_exit")


def test_the_canarys_timing_half_refuses_a_canary_slower_than_its_ceiling(tmp_path: Path) -> None:
    """The canary's TIMING half (``BRIEF_42`` Amendment B, B3.2(b)): after the correctness half, a canary slower than
    ``CANARY_MAX_SECONDS`` = 2.0 s refuses the start.  Pinned as text in ``canary_both_halves``, then EXECUTED: the
    driver's own ``refuse`` and ``canary_both_halves``, cut from its text, run by bash against a stub interpreter that
    prints a canary line at a chosen number of seconds -- no python, no CityFlow.  10.5 s must refuse too, which a
    string comparison ("10.5" < "2.0") would admit.

    *Mutation:* the timing-half refusal removed (reviewer B's M5 at G1, four lines) -> this dies.
    """
    code = _code(_text())
    ceiling = re.search(r"^CANARY_MAX_SECONDS=(\S+)$", code, flags=re.MULTILINE)
    assert ceiling, "the driver declares no CANARY_MAX_SECONDS="
    assert ceiling.group(1) == "2.0"
    body = _function(code, "canary_both_halves")
    _in_order(
        body,
        [
            ("the correctness half", "-m offline.transfer_calibration"),
            ("the seconds read", "CANARY=$(echo \"$CANARY_LINE\" | awk '{print $2}')"),
            ("the ceiling compared", "if awk -v c=\"$CANARY\" -v m=\"$CANARY_MAX_SECONDS\" 'BEGIN { exit !(c > m) }'; then"),
            ("the refusal", 'refuse "canary $CANARY s exceeds $CANARY_MAX_SECONDS s'),
        ],
    )

    stub = tmp_path / "python"
    stub.write_text('#!/bin/sh\necho "canary $STUB_SECONDS s {\\"two_routes_agree\\": true}"\n', encoding="utf-8")
    stub.chmod(0o755)
    harness = "\n".join(
        [
            "set -euo pipefail",
            f"PY={stub}",
            "WORK_TREE=/nonexistent DRAWS=/nonexistent OUTPUT=/nonexistent TRAINING=/nonexistent",
            f"CANARY_MAX_SECONDS={ceiling.group(1)}",
            "refuse() {",
            _function(code, "refuse"),
            "}",
            "canary_both_halves() {",
            body,
            "}",
            "canary_both_halves",
            'echo "ADMITTED $CANARY"',
        ]
    )
    verdicts = {}
    for seconds in ("0.75", "1.99", "2.0", "2.01", "10.5"):
        result = subprocess.run(
            ["bash", "-c", harness], env={**os.environ, "STUB_SECONDS": seconds}, capture_output=True, text=True,
            timeout=60, check=False,
        )
        verdicts[seconds] = (result.returncode, result.stdout, result.stderr)
    for seconds in ("0.75", "1.99", "2.0"):
        status, stdout, _stderr = verdicts[seconds]
        assert status == 0 and f"ADMITTED {seconds}" in stdout, (seconds, verdicts[seconds])
    for seconds in ("2.01", "10.5"):
        status, stdout, stderr = verdicts[seconds]
        assert status == 2, (seconds, verdicts[seconds])
        assert f"REFUSING TO START: canary {seconds} s exceeds 2.0 s" in stderr, (seconds, verdicts[seconds])
        assert "ADMITTED" not in stdout, (seconds, verdicts[seconds])


def test_the_driver_prints_no_outcome() -> None:
    code = _code(_text())
    for word in ("final_loss", "att_engine", "att_ours", "window_means", "losses"):
        assert word not in code, word
    assert not re.search(r"\bcat\b[^\n]*\.json", code)


def _header_lines(text: str) -> list[str]:
    driver = "bash /home/filip/rltraffic-p53c-run/offline/campaigns/p5_3c_train.sh"
    return [line for line in text.splitlines() if line.startswith("#") and "tee -i -a" in line and driver in line]


def test_the_header_documents_both_foreground_forms_from_the_run_tree_with_tee_i_a() -> None:
    text = _text()
    lines = _header_lines(text)
    assert len(lines) == 2, lines
    for mode, capture in (("timing", "train_timing_capture.txt"), ("train", "train_capture.txt")):
        line = next(item for item in lines if f"p5_3c_train.sh {mode} " in item)
        path = f"/home/filip/rltraffic/output/p5_3c_runs/{capture}"
        assert "bash /home/filip/rltraffic-p53c-run/offline/campaigns/p5_3c_train.sh" in line
        assert f"2>&1 | tee -i -a {path}; echo \"DRIVER EXIT: ${{PIPESTATUS[0]}}\" | tee -i -a {path}" in line
    assert "mkdir -p /home/filip/rltraffic/output/p5_3c_runs && tmux new -s p53c_train" in text
    assert "NOT `tmux new -s NAME '<cmd>'`" in text


# ----------------------------------------------------------------------------------------------------------------------
# The driver EXECUTED on a sandbox
# ----------------------------------------------------------------------------------------------------------------------

SEEDS = (101, 202, 303, 404, 505)
CORPUS_DIRS = tuple(f"cf_hz1x1__mappo1000__seed{seed}" for seed in SEEDS) + ("cf_hz1x1__random",)


def _needs_the_driver_environment(*, nvidia: bool = False) -> Path:
    main = _main_tree(_text())
    interpreter = main / ".venv" / "bin" / "python"
    if not interpreter.is_file():
        pytest.skip(f"needs the main tree's interpreter at {interpreter}")
    if shutil.which("setsid") is None:
        pytest.skip("setsid is not installed: the driver must lead its own process group")
    if nvidia and shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi is not installed: the timing mode samples the device with it")
    for name in CORPUS_DIRS:
        if not (main / "datasets_v11" / name / "manifest.json").is_file():
            pytest.skip(f"{main / 'datasets_v11' / name} is absent: check-inputs builds both subjects from the corpus")
    for seed in SEEDS:
        for relative in (f"p4_dt/dt_seed{seed}.pt", f"p4_7/checkpoints/mix50_dt_seed{seed}.pt"):
            if not (main / "output" / relative).is_file():
                pytest.skip(f"{main / 'output' / relative} is absent: check-inputs verifies it by digest")
    if not (main / "output" / "SHA256SUMS_p4_7.txt").is_file():
        pytest.skip(f"{main / 'output' / 'SHA256SUMS_p4_7.txt'} is absent: check-inputs reads P4.7's digests from it")
    probe = subprocess.run(
        [str(interpreter), "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip("CUDA is not available to the main tree's interpreter: check-inputs refuses without it")
    return main


class TrainSandbox:
    """One executed driver: a clean snapshot clone as the run tree, every writable root in ``tmp_path``."""

    def __init__(self, *, base: Path, clone: Path, driver: Path, sandbox: Path) -> None:
        self.base = base
        self.clone = clone
        self.driver = driver
        self.sandbox = sandbox
        self.output = sandbox / "output"
        self.training = self.output / "p5_3c_training"
        self.token = sandbox / "TOKEN_train"

    @property
    def head(self) -> str:
        return _git("rev-parse", "HEAD", cwd=self.clone).strip()

    def write_token(self) -> None:
        self.token.write_text("authorised by tests/test_p5_3c_train_driver.py\n", encoding="utf-8")

    def run(self, *args: str, timeout: float = 900.0) -> subprocess.CompletedProcess[str]:
        """``setsid --wait`` makes the driver a process-group leader, which it requires."""
        result = subprocess.run(
            ["setsid", "--wait", "bash", str(self.driver), *args],
            capture_output=True, text=True, cwd=str(self.clone), timeout=timeout,
        )
        (self.base / "driver_capture.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        return result

    def contents(self) -> dict[str, bytes | None]:
        """Every entry under the sandbox, symlinks NOT followed (the linked real inputs are listed, never read)."""
        found: dict[str, bytes | None] = {}
        for directory, subdirectories, files in os.walk(self.sandbox, followlinks=False):
            for name in subdirectories + files:
                path = Path(directory) / name
                relative = str(path.relative_to(self.sandbox))
                found[relative] = None if path.is_dir() or path.is_symlink() else path.read_bytes()
        return dict(sorted(found.items()))


@pytest.fixture
def train_sandbox(tmp_path: Path) -> Iterator[Callable[..., TrainSandbox]]:
    """The ONE place an executed training driver is built: roots redirected, training and timing slots stubbed."""
    main = _needs_the_driver_environment()

    def build(
        *,
        register_run_tree: bool = True,
        real_canary: bool = False,
        populate: Callable[[TrainSandbox], None] | None = None,
    ) -> TrainSandbox:
        sandbox = tmp_path / "sandbox"
        output = sandbox / "output"
        (output / "p4_7" / "checkpoints").mkdir(parents=True)
        (output / "p4_dt").symlink_to(main / "output" / "p4_dt", target_is_directory=True)
        for seed in SEEDS:
            name = f"mix50_dt_seed{seed}.pt"
            (output / "p4_7" / "checkpoints" / name).symlink_to(main / "output" / "p4_7" / "checkpoints" / name)
        (output / "SHA256SUMS_p4_7.txt").symlink_to(main / "output" / "SHA256SUMS_p4_7.txt")
        stamp = output / "p5_3c_training" / "fenced_timing" / TIMING_STAMP
        stamp.mkdir(parents=True)
        (stamp / "timing.json").write_text(
            json.dumps(
                {
                    "format_version": "p5.3c-train-timing/1.0", "concurrency": 1,
                    "slots": {"alone": {"device_peak_mib": 100.0}, "repeat": {"device_peak_mib": 100.0}},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        clone = _snapshot_clone(tmp_path)
        path = clone / "offline" / "campaigns" / "p5_3c_train.sh"
        text = path.read_text(encoding="utf-8")
        if register_run_tree:
            text = _substitute(text, "RUN_TREE=/home/filip/rltraffic-p53c-run\n", f"RUN_TREE={clone}\n", 1)
        text = _substitute(text, "OUTPUT=$MAIN/output\n", f"OUTPUT={output}\n", 1)
        # CORPUS is NOT redirected: it stays P4's absolute root, because check-inputs compares each subject's statistics,
        # which record their directories, with P4's and P4.7's checkpoints; every command the driver runs on it reads.
        assert text.count("CORPUS=$MAIN/datasets_v11\n") == 1
        text = _substitute(text, "TOKEN=$OUTPUT/p5_3c_runs/TOKEN_train\n", f"TOKEN={sandbox / 'TOKEN_train'}\n", 1)
        text = _substitute(text, TRAIN_CALL, TRAIN_STUB, 1)
        text = _substitute(text, SLOT_CALL, SLOT_STUB, 1)
        if not real_canary:
            text = _substitute(text, CANARY_CALL, CANARY_STUB, 1)
        path.write_text(text, encoding="utf-8")
        _commit_clone(clone, "the training driver's roots redirected to the sandbox, its training stubbed")
        built = TrainSandbox(base=tmp_path, clone=clone, driver=path, sandbox=sandbox)
        if populate is not None:
            populate(built)
        return built

    yield build


def test_the_fixture_redirects_every_writable_root_and_stubs_the_training(train_sandbox: Any) -> None:
    sb = train_sandbox()
    code = _code(sb.driver.read_text(encoding="utf-8"))
    assert f"OUTPUT={sb.output}\n" in code and f"TOKEN={sb.token}\n" in code and f"RUN_TREE={sb.clone}\n" in code
    assert "$MAIN/output" not in code and "p5_3c_runs/TOKEN" not in code
    assert TRAIN_STUB in code and SLOT_STUB in code and CANARY_STUB in code
    assert "offline.context_sweep train --run" not in code and "offline.context_sweep timing run" not in code
    assert _git("status", "--porcelain", cwd=sb.clone) == ""


def test_train_without_a_token_runs_every_check_and_the_real_canary_and_creates_nothing(train_sandbox: Any) -> None:
    sb = train_sandbox(real_canary=True)
    before = sb.contents()
    result = sb.run("train", sb.head, TIMING_STAMP)
    output = result.stdout + result.stderr
    assert result.returncode == 2, output[-3000:]
    assert "check_inputs PASSED" in result.stdout
    assert "resume_decision: 60 to train, 0 to skip" in result.stdout
    assert re.search(r"^canary \d+\.\d+ s ", result.stdout, flags=re.MULTILINE)
    assert f"REFUSING TO START: no run token at {sb.token}" in output
    assert sb.contents() == before, "a refused start created or changed something"


def _garbage_checkpoint(sb: TrainSandbox) -> None:
    target = sb.training / "checkpoints" / "mappo1000_k5_b64_seed101.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not a checkpoint")
    sb.write_token()


def test_an_invalid_existing_checkpoint_is_refused_before_the_canary_and_the_token(train_sandbox: Any) -> None:
    sb = train_sandbox(populate=_garbage_checkpoint)
    before = sb.contents()
    result = sb.run("train", sb.head, TIMING_STAMP)
    output = result.stdout + result.stderr
    assert result.returncode == 2, output[-3000:]
    assert "mappo1000_k5_b64_seed101.pt exists but cannot be read" in output
    assert "canary " not in result.stdout
    assert sb.token.is_file(), "the token must not be consumed by a refused start"
    assert sb.contents() == before


def test_with_a_token_it_is_consumed_the_attempt_written_and_the_stubbed_training_fails_the_run(
    train_sandbox: Any,
) -> None:
    sb = train_sandbox(populate=TrainSandbox.write_token)
    result = sb.run("train", sb.head, TIMING_STAMP)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output[-3000:]
    assert not sb.token.exists(), "the token is consumed on start"
    starts = sorted((sb.training / "starts").iterdir())
    assert len(starts) == 1
    assert (starts[0] / "canary.json").is_file()
    assert (starts[0] / "FAILED").read_text(encoding="utf-8").startswith("TRAIN train FAILED at train mappo1000_k20_b64_seed101")
    assert sorted(path.name for path in (sb.training / "attempts").iterdir()) == ["mappo1000_k20_b64_seed101.1"]
    assert sorted((sb.training / "checkpoints").iterdir()) == []
    assert sorted((sb.training / "staging").iterdir()) == []
    assert "STUB train" in result.stdout


def test_a_copy_outside_the_run_worktree_is_refused(train_sandbox: Any) -> None:
    sb = train_sandbox(register_run_tree=False, populate=TrainSandbox.write_token)
    before = sb.contents()
    result = sb.run("train", sb.head, TIMING_STAMP)
    assert result.returncode == 2
    assert "not the run worktree /home/filip/rltraffic-p53c-run" in result.stderr
    assert sb.contents() == before


def test_a_run_tree_at_another_commit_is_refused(train_sandbox: Any) -> None:
    sb = train_sandbox(populate=TrainSandbox.write_token)
    before = sb.contents()
    other = "0" * 40
    result = sb.run("train", other, TIMING_STAMP)
    assert result.returncode == 2
    assert f"the run worktree is at {sb.head}, not {other}" in result.stderr
    assert sb.contents() == before


def _file_where_a_directory_belongs(sb: TrainSandbox) -> None:
    (sb.training / "staging").write_bytes(b"a file where the staging directory belongs")
    sb.write_token()


def test_a_file_where_a_training_directory_belongs_is_refused_before_the_canary_and_the_token(
    train_sandbox: Any,
) -> None:
    sb = train_sandbox(populate=_file_where_a_directory_belongs)
    before = sb.contents()
    result = sb.run("train", sb.head, TIMING_STAMP)
    assert result.returncode == 2
    assert f"{sb.training / 'staging'} exists and is not a directory" in result.stderr
    assert "canary " not in result.stdout
    assert sb.contents() == before


def test_timing_writes_only_under_a_new_fenced_stamp_and_fails_at_the_stubbed_slot(train_sandbox: Any) -> None:
    _needs_the_driver_environment(nvidia=True)
    sb = train_sandbox(populate=TrainSandbox.write_token)
    before = sb.contents()
    result = sb.run("timing", sb.head, timeout=300.0)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output[-3000:]
    assert sb.token.is_file(), "the timing mode takes no token"
    after = sb.contents()
    created = sorted(set(after) - set(before))
    fenced = [entry for entry in created if not entry.startswith("output/p5_3c_training/fenced_timing/")]
    assert fenced == [], fenced
    stamps = [entry for entry in created if entry.count("/") == 3]
    assert len(stamps) == 1 and re.fullmatch(r"output/p5_3c_training/fenced_timing/\d{8}T\d{6}Z", stamps[0])
    failed = after[f"{stamps[0]}/FAILED"]
    assert failed is not None and failed.startswith(b"TRAIN timing FAILED")
    assert f"{stamps[0]}/nvidia_smi_alone.csv" in after
    assert all(after[key] == value for key, value in before.items())
