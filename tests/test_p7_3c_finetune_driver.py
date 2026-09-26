"""P7.3c C4 (``BRIEF_41``): the training driver ``offline/campaigns/p7_3c_finetune.sh`` -- its text, and its refusals EXECUTED.

T-driver for C4 (``BRIEF_41`` §4, plan section 8, Amendment B's B3 fix list applied from the start):

* **Comment-free text assertions, per mode.**  ``train``: check-inputs, the concurrency from G5's record and the
  pre-token resume scan, then the canary, the traps and the token; after it the start directory, ``record-canary``,
  per run the resume decision, the attempt marker and the training; then the manifest (re-verified) and the record,
  LAST.  ``timing``: no token at all, and every write under ``fenced_timing/<stamp>/``.  Both: ``-P`` on every
  interpreter call, ``set -euo pipefail``, nothing deleted but the token, the run worktree ENFORCED (B3.1: the tree
  and its HEAD), ``FAILED`` on every path after the token through an ``EXIT`` trap keyed on a success flag, written
  with ``printf`` before any echo (B3.4).
* **The header's documented lines** -- the foreground form with ``tee -i -a`` (B3.4) and ``${PIPESTATUS[0]}``.
* **The driver EXECUTED** on a sandbox copy: a committed snapshot clone registered as the run tree, every writable
  root in ``tmp_path``, A20(a)'s five checkpoints LINKED read-only into the sandbox's output tree, a synthetic
  corpus with its gate record, and the training and timing calls replaced by stubs that train nothing -- so no test
  can train, whatever breaks.

⚠️ ``BRIEF_41`` Amendment B, B5: the EXECUTED tests start the driver, whose liveness guard refuses while another run is
live, and whose own children match that guard's pattern.  They are run only when no corpus or campaign run is live.
The two harness traps of §4 hold: a mutant is COMMITTED (the driver refuses a dirty tree), and pytest is invoked from
a command whose text does not carry the liveness pattern.

GATES -- each ``skip`` names what it consumes: the main tree's interpreter, ``setsid``, A20(a)'s five checkpoints
under the main tree's ``output/``, CUDA (``check-inputs`` refuses without it), ``nvidia-smi`` for the timing mode, and
``tmux`` for the header test.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import pytest

import offline.transfer_calibration as tc
from tests.p7_3c_fewshot_fixtures import write_training_corpus
from tests.test_p7_3d_campaign_path import _commit_clone, _git, _snapshot_clone, _substitute

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "offline" / "campaigns" / "p7_3c_finetune.sh"
TIMING_STAMP = "20260925T230000Z"

#: The two calls that would train, replaced in every executed test: the suite never trains through the driver.
TRAIN_CALL = (
    'PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot train --run "$1" --device cuda --output-root "$OUTPUT" '
    '--corpus-dir "$CORPUS" --data-dir "$DATA"'
)
TRAIN_STUB = "sh -c 'echo \"STUB train (tests/test_p7_3c_finetune_driver.py): nothing trained\"; exit 97'"
SLOT_CALL = (
    'PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot timing run --output-root "$OUTPUT" --corpus-dir "$CORPUS" '
    '--data-dir "$DATA" --stamp "$STAMP" --slot "$1"'
)
SLOT_STUB = "sh -c 'echo \"STUB timing run (tests/test_p7_3c_finetune_driver.py): nothing trained\"; exit 97'"


def _text() -> str:
    return DRIVER.read_text(encoding="utf-8")


def _code(text: str) -> str:
    """The driver without its comment lines -- what the shell executes, and nothing a comment could satisfy."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _function(code: str, name: str) -> str:
    """The body of the shell function *name* (from its ``name() {`` line to the first ``}`` at column 0)."""
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}$", code, flags=re.MULTILINE | re.DOTALL)
    assert match, f"the driver defines no function {name}()"
    return match.group(1)


def _main_tree(text: str) -> Path:
    match = re.search(r"^MAIN=(\S+)$", text, flags=re.MULTILINE)
    assert match, "the driver declares no MAIN="
    return Path(match.group(1))


def _in_order(body: str, order: list[tuple[str, str]]) -> None:
    positions = []
    for label, needle in order:
        assert body.count(needle) == 1, f"{label}: expected ONE {needle!r}, found {body.count(needle)}"
        positions.append(body.index(needle))
    assert positions == sorted(positions), [label for label, _needle in order]


# ----------------------------------------------------------------------------------------------
# The text, comments stripped
# ----------------------------------------------------------------------------------------------


def test_train_mode_checks_everything_before_the_token_and_writes_the_record_last() -> None:
    """*Mutation:* the resume scan moved after the token -> this dies."""
    body = _function(_code(_text()), "run_train")
    _in_order(
        body,
        [
            ("check-inputs, with G5's record", '--gate-record "$GATE_RECORD" --data-dir "$DATA" --timing "$TIMING"'),
            ("the concurrency, from G5's record", 'timing concurrency --output-root "$OUTPUT" --timing "$TIMING"'),
            ("the pre-token resume scan", "resume-decision --all"),
            ("the canary", "canary_both_halves"),
            ("the traps", "install_traps"),
            ("the token check", 'if [ ! -f "$TOKEN" ]; then'),
            ("the token consumed", 'rm -f "$TOKEN"'),
            ("the start directory", 'mkdir -p "$START_DIR"'),
            ("record-canary", "record-canary --line"),
            ("the resume decision per run", 'resume-decision --run "$RUN"'),
            ("the attempt marker", 'attempt --run "$RUN"'),
            ("the training", 'train_one "$RUN" &'),
            ("the manifest", "manifest --output-root"),
            ("the manifest re-verified", "sha256sum -c --quiet SHA256SUMS_p7_3c_finetune.txt"),
            ("the record", "record --output-root"),
            ("the success flag", "SUCCESS=1"),
            ("COMPLETE", 'tee "$START_DIR/COMPLETE"'),
        ],
    )
    assert '"$PY"' not in body[body.index("record --output-root") + 1 :], "the record is the LAST module call"


def test_timing_mode_takes_no_token_and_writes_only_under_its_fenced_stamp() -> None:
    body = _function(_code(_text()), "run_timing")
    assert "TOKEN" not in body, "G5's timing is fenced and unregistered: it consumes no token"
    _in_order(
        body,
        [
            ("check-inputs", '--gate-record "$GATE_RECORD" --data-dir "$DATA")'),
            ("a stamp never reused", 'if [ -e "$STAMP_DIR" ]; then'),
            ("the canary", "canary_both_halves"),
            ("the traps", "install_traps"),
            ("the stamp directory", 'mkdir -p "$STAMP_DIR"'),
            ("alone", "run_slots alone"),
            ("two at once", "run_slots pair1 pair2"),
            ("three at once", "run_slots triple1 triple2 triple3"),
            ("the same-seed repeat, alone", "run_slots repeat"),
            ("the k = 100 build, no step", 'timing build --output-root "$OUTPUT" --corpus-dir "$CORPUS"'),
            ("the summary", 'timing summarize --output-root "$OUTPUT" --stamp "$STAMP"'),
            ("the success flag", "SUCCESS=1"),
            ("COMPLETE", 'tee "$STAMP_DIR/COMPLETE"'),
        ],
    )
    assert re.search(r'^\s*STAMP_DIR=\$FENCED/\$STAMP$', body, flags=re.MULTILINE)
    assert re.search(r"^FENCED=\$TRAINING/fenced_timing$", _code(_text()), flags=re.MULTILINE)


def test_every_interpreter_call_carries_minus_P() -> None:
    code = _code(_text())
    calls = [line for line in code.splitlines() if '"$PY"' in line and "-x" not in line]
    assert len(calls) >= 14
    assert [line for line in calls if re.search(r'"\$PY"(?! -P)', line)] == []


def test_the_numerical_regime_is_p5_2s_one_thread_and_no_cublas_workspace_config() -> None:
    """A24(b) trains "in the subject's regime ... as P5.2's": ``offline/campaigns/p5_2.sh`` exports OMP and MKL at one
    thread and UNSETS ``CUBLAS_WORKSPACE_CONFIG`` outside its deterministic regime (its lines 98-117: a launch shell
    carrying it would change cuBLAS's GEMM selection, silently).  Both before the first interpreter call."""
    code = _code(_text())
    first_call = code.index('"$PY" -P')
    for line in ("export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1", "unset CUBLAS_WORKSPACE_CONFIG"):
        match = re.search(rf"^{re.escape(line)}$", code, flags=re.MULTILINE)
        assert match, f"the driver lacks the top-level line {line!r}"
        assert match.start() < first_call, f"{line!r} comes after the first interpreter call"


def test_strict_mode_nothing_deleted_but_the_token_and_no_overwrite() -> None:
    code = _code(_text())
    assert re.search(r"^set -euo pipefail$", code, flags=re.MULTILINE)
    assert re.findall(r"\brm\b[^\n]*", code) == ['rm -f "$TOKEN"']
    assert re.findall(r"\bmv\b[^\n]*", code) == []
    assert "--overwrite" not in code and "--force" not in code


def test_the_run_worktree_and_its_commit_are_enforced_b3_1() -> None:
    code = _code(_text())
    assert 'WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)' in code
    assert 'if [ "$WORK_TREE" = "$IMPLEMENTER_TREE" ]; then' in code
    assert 'if [ "$WORK_TREE" != "$RUN_TREE" ]; then' in code
    assert 'HEAD_COMMIT=$(git -C "$WORK_TREE" rev-parse HEAD)' in code
    assert 'if [ "$HEAD_COMMIT" != "$EXPECTED_COMMIT" ]; then' in code
    assert "[[ \"$EXPECTED_COMMIT\" =~ ^[0-9a-f]{40}$ ]]" in code
    liveness = re.search(r"pgrep -f '([^']+)'", code)
    assert liveness and "offline\\.(collect|transfer_calibration|transfer_curve|few_shot|g2_measure)" in liveness.group(1)


def test_failed_is_written_on_every_path_after_the_token_b3_4() -> None:
    """An ``EXIT`` trap keyed on a success flag, and ``FAILED`` written with ``printf ... >`` before any echo."""
    code = _code(_text())
    on_exit = _function(code, "on_exit")
    assert 'if [ "$SUCCESS" -ne 1 ]' in on_exit and 'printf' in on_exit and '> "$MARKER_DIR/FAILED"' in on_exit
    fail = _function(code, "fail")
    assert fail.index("printf") < fail.index("echo"), "FAILED is on disk before anything reaches a pipe"
    on_signal = _function(code, "on_signal")
    assert on_signal.index("printf") < on_signal.index("echo")
    install = _function(code, "install_traps")
    assert "trap on_exit EXIT" in install and "trap on_signal INT TERM HUP" in install


def _header_lines(text: str) -> dict[str, tuple[str, str, str]]:
    """The two Step-2 lines by mode, with the driver path and capture path each names -- read, never restated."""
    found: dict[str, tuple[str, str, str]] = {}
    for match in re.finditer(r"^#\s+Step 2 \((\w+)\), at ITS PROMPT:\s+(bash .+)$", text, flags=re.MULTILINE):
        mode, line = match.group(1), match.group(2)
        parts = re.match(rf"bash (\S+p7_3c_finetune\.sh) {mode} \S+(?: \S+)? 2>&1 \| tee -i -a (\S+); ", line)
        assert parts, f"the {mode} Step-2 line is not the foreground form: {line}"
        found[mode] = (line, parts.group(1), parts.group(2))
    return found


def test_the_header_documents_both_foreground_forms_from_the_run_tree() -> None:
    lines = _header_lines(_text())
    assert sorted(lines) == ["timing", "train"]
    for mode, (line, driver_path, capture_path) in lines.items():
        assert "${PIPESTATUS[0]}" in line, f"{mode}: B.5-3, the pane carries the DRIVER's exit status"
        assert line.count(f"tee -i -a {capture_path}") == 2, f"{mode}: B3.4, tee ignores the interrupt"
        assert "/rltraffic-p73c-run/" in driver_path, "J1(e): the documented tree is the RUN worktree"


# ----------------------------------------------------------------------------------------------
# The driver EXECUTED on a sandbox copy
# ----------------------------------------------------------------------------------------------


def _needs_the_driver_environment() -> Path:
    main = _main_tree(_text())
    interpreter = main / ".venv" / "bin" / "python"
    if not interpreter.is_file():
        pytest.skip(f"needs the main tree's interpreter at {interpreter}")
    if shutil.which("setsid") is None:
        pytest.skip("setsid is not installed")
    for seed in tc.TRAINING_SEEDS:
        path = main / "output" / tc.GRID4X4_CHECKPOINT_SUBDIR / f"{tc.GRID4X4_CHECKPOINT_STEM}{seed}.pt"
        if not path.is_file():
            pytest.skip(f"{path} is absent: check-inputs reads A20(a)'s five checkpoints")
    probe = subprocess.run(
        [str(interpreter), "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip("CUDA is not available to the main tree's interpreter: check-inputs refuses without it")
    return main


class FinetuneSandbox:
    """One executed driver: a clean snapshot clone as the run tree, every writable root in ``tmp_path``."""

    def __init__(self, *, base: Path, clone: Path, driver: Path, sandbox: Path) -> None:
        self.base = base
        self.clone = clone
        self.driver = driver
        self.sandbox = sandbox
        self.output = sandbox / "output"
        self.training = self.output / "p7_3c_training"
        self.token = sandbox / "TOKEN_finetune"

    @property
    def head(self) -> str:
        return _git("rev-parse", "HEAD", cwd=self.clone).strip()

    def write_token(self) -> None:
        self.token.write_text("authorised by tests/test_p7_3c_finetune_driver.py\n", encoding="utf-8")

    def run(self, *args: str, timeout: float = 900.0) -> subprocess.CompletedProcess[str]:
        """``setsid --wait`` makes the driver a process-group leader, which it requires."""
        result = subprocess.run(
            ["setsid", "--wait", "bash", str(self.driver), *args],
            capture_output=True, text=True, cwd=str(self.clone), timeout=timeout,
        )
        (self.base / "driver_capture.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        return result

    def contents(self) -> dict[str, bytes | None]:
        return {
            str(path.relative_to(self.sandbox)): (None if path.is_dir() or path.is_symlink() else path.read_bytes())
            for path in sorted(self.sandbox.rglob("*"))
        }


def _timing_record() -> dict[str, Any]:
    return {
        "format_version": "p7.3c-finetune-timing/1.0",
        "concurrency": 1,
        "phases": {
            "alone": {"ms_per_step": 122.0, "device_peak_mib": 3000.0},
            "pair": {"ms_per_step": 250.0, "device_peak_mib": 5000.0},
            "triple": {"ms_per_step": 380.0, "device_peak_mib": 7000.0},
        },
    }


@pytest.fixture
def finetune_sandbox(tmp_path: Path) -> Callable[..., FinetuneSandbox]:
    """B.7-3's rule: the ONE place an executed training driver is built -- roots redirected, training stubbed."""
    main = _needs_the_driver_environment()

    def build(*, register_run_tree: bool = True, populate: Callable[[FinetuneSandbox], None] | None = None) -> FinetuneSandbox:
        sandbox = tmp_path / "sandbox"
        sources = sandbox / "output" / tc.GRID4X4_CHECKPOINT_SUBDIR
        sources.mkdir(parents=True)
        for seed in tc.TRAINING_SEEDS:
            name = f"{tc.GRID4X4_CHECKPOINT_STEM}{seed}.pt"
            (sources / name).symlink_to(main / "output" / tc.GRID4X4_CHECKPOINT_SUBDIR / name)
        corpus = write_training_corpus(sandbox / "corpus", decisions=3)
        gate = sandbox / "output" / "p7_3c_corpus" / "a17f_gate.json"
        gate.parent.mkdir(parents=True)
        gate.write_text(
            json.dumps(
                {
                    "format_version": "p7.3c-corpus-gate/1.0", "corpus_dir": str(corpus), "all_match": True,
                    "n_draws": 100, "n_intersections": 16, "n_checked": 1600, "n_matching": 1600,
                    "engine_events": {"n_teleports": 0, "n_collisions": 0},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        stamp = sandbox / "output" / "p7_3c_training" / "fenced_timing" / TIMING_STAMP
        stamp.mkdir(parents=True)
        (stamp / "timing.json").write_text(json.dumps(_timing_record()) + "\n", encoding="utf-8")

        clone = _snapshot_clone(tmp_path)
        path = clone / "offline" / "campaigns" / "p7_3c_finetune.sh"
        text = path.read_text(encoding="utf-8")
        if register_run_tree:
            text = _substitute(text, "RUN_TREE=/home/filip/rltraffic-p73c-run\n", f"RUN_TREE={clone}\n", 1)
        text = _substitute(text, "OUTPUT=$MAIN/output\n", f"OUTPUT={sandbox / 'output'}\n", 1)
        text = _substitute(
            text, "CORPUS=$MAIN/datasets_sumo_v11/grid4x4_sumo_maxpressure\n", f"CORPUS={corpus}\n", 1
        )
        text = _substitute(text, "TOKEN=$OUTPUT/p7_3c_runs/TOKEN_finetune\n", f"TOKEN={sandbox / 'TOKEN_finetune'}\n", 1)
        text = _substitute(text, TRAIN_CALL, TRAIN_STUB, 1)
        text = _substitute(text, SLOT_CALL, SLOT_STUB, 1)
        path.write_text(text, encoding="utf-8")
        _commit_clone(clone, "the training driver's roots redirected to the sandbox, its training stubbed")
        built = FinetuneSandbox(base=tmp_path, clone=clone, driver=path, sandbox=sandbox)
        if populate is not None:
            populate(built)
        return built

    return build


def test_the_fixture_redirects_every_writable_root_and_stubs_both_training_calls(finetune_sandbox: Any) -> None:
    sb = finetune_sandbox()
    code = _code(sb.driver.read_text(encoding="utf-8"))
    assert f"OUTPUT={sb.output}\n" in code and f"TOKEN={sb.token}\n" in code and f"RUN_TREE={sb.clone}\n" in code
    assert "$MAIN/output" not in code and "datasets_sumo_v11" not in code and "p7_3c_runs/TOKEN" not in code
    assert TRAIN_STUB in code and SLOT_STUB in code
    assert "offline.few_shot train --run" not in code and "offline.few_shot timing run" not in code
    assert _git("status", "--porcelain", cwd=sb.clone) == ""


def test_train_without_a_token_runs_every_check_and_creates_nothing(finetune_sandbox: Any) -> None:
    sb = finetune_sandbox()
    before = sb.contents()
    result = sb.run("train", sb.head, TIMING_STAMP)
    output = result.stdout + result.stderr
    assert result.returncode == 2, output[-3000:]
    assert "check_inputs PASSED" in result.stdout
    assert "resume_decision: 30 to train, 0 to skip" in result.stdout
    assert re.search(r"^canary \d+\.\d+ s ", result.stdout, flags=re.MULTILINE)
    assert f"REFUSING TO START: no run token at {sb.token}" in output
    assert sb.contents() == before, "a refused start created or changed something"


def _garbage_checkpoint(sb: FinetuneSandbox) -> None:
    target = sb.training / "checkpoints" / "ft_k5_seed101.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not a checkpoint")
    sb.write_token()


def test_an_invalid_existing_checkpoint_is_refused_before_the_canary_and_the_token(finetune_sandbox: Any) -> None:
    sb = finetune_sandbox(populate=_garbage_checkpoint)
    before = sb.contents()
    result = sb.run("train", sb.head, TIMING_STAMP)
    output = result.stdout + result.stderr
    assert result.returncode == 2, output[-3000:]
    assert "ft_k5_seed101.pt exists but cannot be read" in output
    assert not re.search(r"^canary ", result.stdout, flags=re.MULTILINE), "the scan precedes the canary"
    assert sb.token.is_file()
    assert sb.contents() == before


def test_with_a_token_it_is_consumed_the_attempt_counted_and_the_stubbed_training_fails_the_run(
    finetune_sandbox: Any,
) -> None:
    sb = finetune_sandbox()
    sb.write_token()
    result = sb.run("train", sb.head, TIMING_STAMP)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output[-3000:]
    assert not sb.token.exists() and output.count("token consumed and deleted") == 1
    (start,) = list((sb.training / "starts").iterdir())
    assert (start / "canary.json").is_file()
    assert (start / "FAILED").read_text(encoding="utf-8").strip() == "FINETUNE train FAILED at train ft_k5_seed101"
    assert sorted(path.name for path in (sb.training / "attempts").iterdir()) == ["ft_k5_seed101.1"]
    assert list((sb.training / "checkpoints").iterdir()) == []
    assert not (sb.output / "SHA256SUMS_p7_3c_finetune.txt").exists()
    assert not (sb.training / "p7_3c_finetune.json").exists()
    assert not (start / "COMPLETE").exists()


def _file_where_a_directory_belongs(sb: FinetuneSandbox) -> None:
    target = sb.training / "checkpoints"
    target.write_text("a file where the checkpoints directory belongs\n", encoding="utf-8")
    sb.write_token()


def test_a_file_where_a_training_directory_belongs_is_refused_before_the_canary_and_the_token(
    finetune_sandbox: Any,
) -> None:
    """The barrier covers the layout too: a non-directory at ``checkpoints/`` is refused before anything is consumed.
    Without this refusal the start consumes the token and dies at its first ``mkdir``."""
    sb = finetune_sandbox(populate=_file_where_a_directory_belongs)
    before = sb.contents()
    result = sb.run("train", sb.head, TIMING_STAMP)
    output = result.stdout + result.stderr
    assert result.returncode == 2, output[-3000:]
    assert f"REFUSING TO START: {sb.training / 'checkpoints'} exists and is not a directory" in output
    assert not re.search(r"^canary ", result.stdout, flags=re.MULTILINE)
    assert sb.token.is_file()
    assert sb.contents() == before


#: A step after the token and after the start directory, turned into a failure no ``|| fail`` guards.
INJECTED_AFTER_TOKEN = 'echo "  start dir    $START_DIR"'


def test_an_unguarded_failure_after_the_token_still_leaves_failed_through_the_exit_trap(
    finetune_sandbox: Any,
) -> None:
    """B3.4, EXECUTED: ``set -e`` aborts on a failure no ``fail`` call names, and the ``EXIT`` trap still writes
    ``FAILED``.  *Mutation:* ``trap on_exit EXIT`` removed -> no ``FAILED`` -> this dies (the text test alone saw it
    before this test existed)."""
    sb = finetune_sandbox()
    text = _substitute(sb.driver.read_text(encoding="utf-8"), INJECTED_AFTER_TOKEN, "false", 1)
    sb.driver.write_text(text, encoding="utf-8")
    _commit_clone(sb.clone, "an unguarded failure injected after the token")
    sb.write_token()
    result = sb.run("train", sb.head, TIMING_STAMP)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output[-3000:]
    assert not sb.token.exists()
    (start,) = list((sb.training / "starts").iterdir())
    assert (start / "FAILED").read_text(encoding="utf-8").strip() == "FINETUNE train FAILED (exit 1)"
    assert not (start / "COMPLETE").exists()


def test_a_copy_outside_the_run_worktree_is_refused(finetune_sandbox: Any) -> None:
    sb = finetune_sandbox(register_run_tree=False)
    sb.write_token()
    before = sb.contents()
    result = sb.run("train", sb.head, TIMING_STAMP)
    assert result.returncode == 2
    assert f"this copy is in {sb.clone}, not the run worktree /home/filip/rltraffic-p73c-run" in result.stdout + result.stderr
    assert sb.contents() == before


def test_a_run_tree_at_another_commit_is_refused(finetune_sandbox: Any) -> None:
    sb = finetune_sandbox()
    sb.write_token()
    before = sb.contents()
    result = sb.run("train", "0" * 40, TIMING_STAMP)
    assert result.returncode == 2
    assert f"the run worktree is at {sb.head}, not {'0' * 40}" in result.stdout + result.stderr
    assert sb.contents() == before


def test_timing_writes_only_under_a_new_fenced_stamp_and_fails_at_the_stubbed_run(finetune_sandbox: Any) -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi is not installed: the timing mode samples the device with it")
    sb = finetune_sandbox()
    before = sb.contents()
    result = sb.run("timing", sb.head)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output[-3000:]
    fenced = sb.training / "fenced_timing"
    stamps = sorted(path.name for path in fenced.iterdir() if path.name != TIMING_STAMP)
    assert len(stamps) == 1 and re.fullmatch(r"\d{8}T\d{6}Z", stamps[0])
    stamp_dir = fenced / stamps[0]
    assert (stamp_dir / "FAILED").read_text(encoding="utf-8").strip() == "FINETUNE timing FAILED at timing alone"
    created = sorted(set(sb.contents()) - set(before))
    assert created and all(name.startswith(f"output/p7_3c_training/fenced_timing/{stamps[0]}") for name in created)
    assert sb.token.exists() is False, "no token was written, and none is needed"


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_the_train_headers_own_line_passes_the_guard_and_the_pane_carries_the_drivers_exit_code(
    finetune_sandbox: Any,
) -> None:
    sb = finetune_sandbox()
    line, driver_path, capture_path = _header_lines(sb.driver.read_text(encoding="utf-8"))["train"]
    capture = sb.base / "pane_capture.txt"
    line = _substitute(line, driver_path, str(sb.driver), 1)
    line = _substitute(line, capture_path, str(capture), 2)
    line = re.sub(r"p7_3c_finetune\.sh train \S+ \S+", f"p7_3c_finetune.sh train {sb.head} {TIMING_STAMP}", line)

    session = f"p73c_ft_{uuid.uuid4().hex[:8]}"
    subprocess.run(["tmux", "new-session", "-d", "-s", session], check=True)
    try:
        subprocess.run(["tmux", "send-keys", "-t", session, line, "Enter"], check=True)
        for _ in range(720):
            if "DRIVER EXIT:" in (capture.read_text(encoding="utf-8") if capture.exists() else ""):
                break
            time.sleep(0.5)
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session], check=False)

    text = capture.read_text(encoding="utf-8")
    assert "not a process-group leader" not in text
    assert "REFUSING TO START: no run token" in text
    assert text.rstrip().splitlines()[-1] == "DRIVER EXIT: 2"
