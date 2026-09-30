"""P5.3c C3 (``BRIEF_42`` §4 T-driver, the campaign): ``offline/campaigns/p5_3c_eval.sh`` -- its text, and its refusals and
its gate's stop EXECUTED.

* **Comment-free text assertions.**  Every check that can refuse precedes the token (the run worktree AND its commit,
  the modules from it, no live runner, the group leader, SigIgn, the committed inputs, the dirty tree, the RSS budget,
  the layout, ``campaign-inputs``, ``resume-check``, the canary's two halves, the fifteen re-rolled reference cells
  counted); the traps precede the token; after it: ``record-canary``, the device sampler, the REFERENCE stage, the GATE,
  then the SWEEP, the report and the manifest, ``COMPLETE`` last.  ``-P`` on every interpreter call, ``set -euo
  pipefail``, nothing deleted but the token and the two terminal markers, ``FAILED`` on every path after the token, the
  sampler never outliving the driver, no outcome printed.
* **The header** -- the two-step foreground start with ``tee -i -a`` and ``${PIPESTATUS[0]}``; B5's canary-hang rule in
  section 0 (Amendment B.1, B.1.3(3)).
* **Executed on a sandbox** (the training driver's pattern): a committed snapshot clone as the run tree, every writable
  root in ``tmp_path``, the inputs check, the fifteen re-rolls and the stages STUBBED (they need the pinned training
  record, CUDA and 7,000 episodes); the canary real in ONE test.  No token -> nothing created; the gate's stop -> FAILED
  written, the token consumed, NO sweep cell rolled, no report.  *Mutation:* the gate's ``|| fail`` removed -> dies.

GATES: each ``skip`` names what it consumes -- the main tree's interpreter, ``setsid``, CUDA (the stubbed checks
aside, the real canary is a CityFlow episode the driver runs), ``nvidia-smi`` for the device sampler.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from tests.test_p7_3d_campaign_path import _commit_clone, _git, _snapshot_clone, _substitute

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "offline" / "campaigns" / "p5_3c_eval.sh"

INPUTS_CALL = 'PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep campaign-inputs "${COMMON[@]}"'
INPUTS_STUB = "echo 'campaign_inputs PASSED (STUB, tests/test_p5_3c_eval_driver.py)'"
REROLL_CALL = (
    'PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep reference-reroll-check "${COMMON[@]}" '
    '--canary-seconds "$CANARY" --workers "$WORKERS"'
)
REROLL_STUB = (
    "sh -c 'for n in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do "
    "echo \"reference_reroll_check MATCH cell_stub_$n.json\"; done'"
)
REFERENCE_CALL = (
    'PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep cells "${COMMON[@]}" --stage reference '
    '--canary-seconds "$CANARY" --workers "$WORKERS"'
)
REFERENCE_STUB = "echo 'cells reference: STUB, nothing rolled'"
GATE_CALL = 'PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep reference-gate --output-root "$OUTPUT" --data-dir "$DATA"'
GATE_STUB = "sh -c 'echo \"reference_gate FAILED: 1 of 500 ref_mappo1000_k20 chunks differ (STUB)\"; exit 1'"
SWEEP_CALL = (
    'PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep cells "${COMMON[@]}" --stage sweep '
    '--canary-seconds "$CANARY" --workers "$WORKERS"'
)
CANARY_CALL = (
    'PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_calibration --draws-root "$DRAWS" --output-root "$OUTPUT" '
    '--work-dir "$WORK" canary'
)
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


def test_every_refusal_precedes_the_token_and_the_gate_stands_between_the_reference_stage_and_the_sweep() -> None:
    """*Mutation:* the sweep moved before the gate, or the gate's ``|| fail`` removed -> this dies."""
    code = _code(_text())
    _in_order(
        code,
        [
            ("the commit argument", '[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]'),
            ("the interpreter", 'if [ ! -x "$PY" ]; then'),
            ("not the implementer's tree", 'if [ "$WORK_TREE" = "$IMPLEMENTER_TREE" ]; then'),
            ("the run tree", 'if [ "$WORK_TREE" != "$RUN_TREE" ]; then'),
            ("its commit", 'if [ "$HEAD_COMMIT" != "$EXPECTED_COMMIT" ]; then'),
            ("the cwd", 'cd "$MAIN"'),
            ("the liveness guard", "ALIVE=$(pgrep -f"),
            ("the group leader", '"$(ps -o pgid= -p $$ | tr -d \' \')" != "$$"'),
            ("SigIgn", "SIGIGN_MASK=$(awk"),
            ("the committed inputs", 'for INPUT in "$DATA/p4_k20_att_engine_rows.json"'),
            ("the dirty tree", 'WORK_TREE_DIRTY=$(git -C "$WORK_TREE" status --porcelain)'),
            ("the RSS budget", 'if [ -n "$AVAILABLE_MIB" ] && [ "$AVAILABLE_MIB" -lt "$RSS_BUDGET_MIB" ]; then'),
            ("the layout", 'refuse_non_directories "$CAMPAIGN_DIR" "$WORK" "$ARTIFACTS" "$G2_DIR"'),
            ("the inputs by digest", INPUTS_CALL),
            ("resume-check", "-m offline.context_sweep resume-check"),
            ("the canary", CANARY_CALL),
            ("the canary's timing half", "if awk -v c=\"$CANARY\" -v m=\"$CANARY_MAX_SECONDS\" 'BEGIN { exit !(c > m) }'; then"),
            ("the fifteen re-rolls", REROLL_CALL),
            ("their count", 'if [ "$N_REFERENCE_RESULTS" -ne "$REROLL_CELLS" ] || [ "$N_REFERENCE_MATCH" -ne "$REROLL_CELLS" ]; then'),
            ("the EXIT trap", "trap on_exit EXIT"),
            ("the signal trap", "trap on_signal INT TERM HUP"),
            ("the token check", 'if [ ! -f "$TOKEN" ]; then'),
            ("the token consumed", 'rm -f "$TOKEN"'),
            ("the work directory", 'mkdir -p "$WORK" "$ARTIFACTS"'),
            ("the canary recorded", 'record-canary --line "$CANARY_LINE" || fail "record-canary"'),
            ("the device sampler", 'start_sampler "$STARTED_UTC"'),
            ("the reference stage", REFERENCE_CALL + ' || fail "cells reference"'),
            ("the GATE", GATE_CALL + ' || fail "reference-gate"'),
            ("the sweep", SWEEP_CALL + ' || fail "cells sweep"'),
            ("the sampler stopped, its peak read", 'DEVICE_PEAK=$(sort -n "$SAMPLES" | tail -1)'),
            ("the report", '-m offline.context_sweep report "${COMMON[@]}" || fail "report"'),
            ("the manifest", '-m offline.context_sweep campaign-manifest --output-root "$OUTPUT" || fail "campaign-manifest"'),
            ("the manifest verified", "sha256sum -c --quiet SHA256SUMS_p5_3c.txt"),
            ("success", "SUCCESS=1"),
        ],
    )
    assert re.search(r"^REROLL_CELLS=15$", code, flags=re.MULTILINE)
    assert re.search(r"^CANARY_MAX_SECONDS=2\.0$", code, flags=re.MULTILINE)
    assert re.search(r"^WORKERS=12$", code, flags=re.MULTILINE)
    assert re.search(r"^RSS_BUDGET_MIB=24216$", code, flags=re.MULTILINE)
    assert re.search(r"^TOKEN=\$OUTPUT/p5_3c_runs/TOKEN_campaign$", code, flags=re.MULTILINE)


def test_every_interpreter_call_carries_minus_P() -> None:
    """Every token after ``"$PY"`` is ``-P`` -- except the ``]`` that closes the interpreter's existence check."""
    code = _code(_text())
    following = re.findall(r'"\$PY"\s+(\S+)', code)
    calls = [token for token in following if not token.startswith("]")]
    assert len(calls) >= 11, following
    assert set(calls) == {"-P"}, following
    assert len(following) - len(calls) == 1 and 'if [ ! -x "$PY" ]; then' in code
    assert all(line.count("PYTHONPATH=$WORK_TREE") == line.count('"$PY" -P') for line in code.splitlines())


def test_strict_mode_the_regime_and_nothing_deleted_but_the_token_and_the_two_markers() -> None:
    code = _code(_text())
    assert re.search(r"^set -euo pipefail$", code, flags=re.MULTILINE)
    assert re.search(r"^export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1$", code, flags=re.MULTILINE)
    assert re.search(r"^unset CUBLAS_WORKSPACE_CONFIG$", code, flags=re.MULTILINE)
    removals = re.findall(r"\brm\b[^\n]*", code)
    assert removals == ['rm -f "$TOKEN"', 'rm -f "$WORK/FAILED" "$WORK/COMPLETE"'], removals
    assert not re.search(r"\bmv\b", code)


def test_the_run_worktree_its_commit_and_the_modules_from_it_are_enforced() -> None:
    code = _code(_text())
    assert re.search(r"^RUN_TREE=/home/filip/rltraffic-p53c-run$", code, flags=re.MULTILINE)
    assert re.search(r"^IMPLEMENTER_TREE=/home/filip/rltraffic-p53c$", code, flags=re.MULTILINE)
    assert "import offline.context_sweep as c, offline.transfer_calibration as t" in code
    assert code.index('cd "$MAIN"') < code.index('"$PY" -P')
    assert 'cd "$WORK_TREE"' not in code


def test_failed_is_written_on_every_path_after_the_token_and_the_sampler_never_outlives_the_driver() -> None:
    code = _code(_text())
    on_exit = _function(code, "on_exit")
    assert 'kill -9 "$SAMPLER"' in on_exit
    assert 'if [ "$SUCCESS" -ne 1 ]' in on_exit and "printf" in on_exit and "FAILED" in on_exit
    fail = _function(code, "fail")
    assert fail.index("printf") < fail.index("echo")
    signal = _function(code, "on_signal")
    assert "kill -- -$$" in signal and "printf" in signal
    marker = _function(code, "failed_marker")
    assert 'echo "$FAILED_FALLBACK"' in marker and '"$TOKEN_CONSUMED" -eq 1' in marker
    start = _function(code, "start_sampler")
    assert 'until [ "$(cat "/proc/$SAMPLER/comm" 2>/dev/null)" = "nvidia-smi" ]; do' in start
    stop = _function(code, "stop_sampler")
    assert stop.index('kill "$pid"') < stop.index('kill -9 "$pid"') < stop.index('wait "$pid"')


def test_the_liveness_guard_names_this_module_and_the_group_leader_and_sigign_are_checked() -> None:
    code = _code(_text())
    assert re.search(r"pgrep -f 'python\.\*offline\\\.\([^)]*\bcontext_sweep\b[^)]*\)'", code), "liveness pattern"
    assert re.search(r"pgrep -f 'python\.\*offline\\\.\([^)]*\btransfer_calibration\b[^)]*\)'", code)
    assert "SigIgn" in code


def test_the_driver_prints_no_outcome() -> None:
    # The committed rows file's NAME carries "att_engine"; it is an input the driver checks exists, never a value.
    code = _code(_text()).replace("p4_k20_att_engine_rows.json", "")
    for word in ("att_engine", "att_ours", "outcome", "p_value", "sentence", "reject"):
        assert word not in code, word
    assert not re.search(r"\bcat\b[^\n]*\.json", code)
    assert not re.search(r"\bjq\b", code)


def _header_lines(text: str) -> list[str]:
    driver = "bash /home/filip/rltraffic-p53c-run/offline/campaigns/p5_3c_eval.sh"
    return [line for line in text.splitlines() if line.startswith("#") and "tee -i -a" in line and driver in line]


def test_the_header_documents_the_foreground_start_from_the_run_tree_with_tee_i_a() -> None:
    text = _text()
    [line] = _header_lines(text)
    path = "/home/filip/rltraffic/output/p5_3c_runs/campaign_capture.txt"
    assert "bash /home/filip/rltraffic-p53c-run/offline/campaigns/p5_3c_eval.sh <commit>" in line
    assert f"2>&1 | tee -i -a {path}; echo \"DRIVER EXIT: ${{PIPESTATUS[0]}}\" | tee -i -a {path}" in line
    assert "mkdir -p /home/filip/rltraffic/output/p5_3c_runs && tmux new -s p53c_campaign" in text
    assert "NOT `tmux new -s NAME '<cmd>'`" in text


def test_the_header_carries_b5s_canary_hang_rule_in_section_0() -> None:
    """Amendment B.1, B.1.3(3).  *Mutation:* the rule's paragraph deleted -> this dies."""
    text = _text()
    section_0 = text[text.index("# 0. USAGE"):text.index("# 1. WHAT IT PRODUCES")]
    prose = " ".join(line.lstrip("#").strip() for line in section_0.splitlines())
    for phrase in (
        "DEFERRED 101",
        "if the driver prints nothing for a minute after `resume_check PASSED`",
        "the canary is hung: Ctrl-C -- the canary precedes the token, so NOTHING is consumed -- and start again",
        "A second occurrence is a finding for the plan, not a rate question",
    ):
        assert phrase in prose, phrase


def test_the_driver_reads_the_pinned_record_from_the_run_tree_and_never_names_the_training_tree() -> None:
    """Amendment C, C4.1 and C4.3: the committed inputs -- the pinned training record among them -- are read from the
    RUN tree's ``docs/data``; nothing under ``output/p5_3c_training/`` is written to again, and the driver's code never
    names it (the sixty are read through the pinned record by ``campaign-inputs`` and the stages)."""
    code = _code(_text())
    assert "\nDATA=$WORK_TREE/docs/data\n" in code
    assert '"$DATA/p5_3c_train.json"' in code
    assert "p5_3c_training" not in code


def test_complete_precedes_success_and_the_header_carries_the_two_restart_remedies() -> None:
    """``BRIEF_42`` Amendment D, D4.1: ``COMPLETE`` is written BEFORE ``SUCCESS=1``, so a failed final write still reaches
    ``on_exit``'s FAILED marker; header section 2 carries the two restart remedies (a stale gate record, a ``.tmp``
    after Ctrl-C); the comment on the pre-token writes says what section 1 says (the fenced re-roll, nothing else).

    *Mutation:* the two lines swapped back -> this dies.
    """
    text = _text()
    tail = _code(text)[_code(text).index("ELAPSED=$(( $(date +%s) - START ))"):]
    _in_order(tail, [("COMPLETE written", '> "$WORK/COMPLETE"'), ("then SUCCESS", "SUCCESS=1")])
    section_2 = text[text.index("# 2. THE STAGES AND THE GATE"):text.index("# 3. ORDERING")]
    prose = " ".join(line.lstrip("#").strip() for line in section_2.splitlines())
    for phrase in (
        "RESUME -- two leftovers of a stopped run need a hand",
        "a restart that re-rolled a reference chunk finds the gate record stale -- `reference-gate` refuses "
        '"already exists and differs" after the token; move output/p5_3c/reference_gate.json aside by hand and start '
        "again",
        "after a Ctrl-C a worker's .cell_<...>.json.<pid>.tmp may remain in cells/ -- `resume-check` names it before "
        "the token; move it aside by hand",
    ):
        assert phrase in prose, phrase
    comments = " ".join(line.lstrip("#").strip() for line in text.splitlines() if line.lstrip().startswith("#"))
    assert "Before the token nothing is written, on any path" not in comments
    assert ("Before the token nothing is written but the fenced re-roll's records under output/p5_3c/g2/ (section 1)"
            in comments)


# ----------------------------------------------------------------------------------------------------------------------
# The driver EXECUTED on a sandbox
# ----------------------------------------------------------------------------------------------------------------------


def _needs_the_driver_environment() -> Path:
    main = _main_tree(_text())
    interpreter = main / ".venv" / "bin" / "python"
    if not interpreter.is_file():
        pytest.skip(f"needs the main tree's interpreter at {interpreter}")
    if shutil.which("setsid") is None:
        pytest.skip("setsid is not installed: the driver must lead its own process group")
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi is not installed: the driver samples the device during the stages")
    if not (main / "scenarios" / "draws" / "cityflow1x1" / "draw_0000").is_dir():
        pytest.skip(f"{main / 'scenarios' / 'draws' / 'cityflow1x1' / 'draw_0000'} is absent: the canary rolls it")
    probe = subprocess.run(
        [str(interpreter), "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip("CUDA is not available to the main tree's interpreter: the campaign evaluates on it (A26.1(b))")
    return main


class EvalSandbox:
    """One executed campaign driver: a clean snapshot clone as the run tree, every writable root in ``tmp_path``."""

    def __init__(self, *, base: Path, clone: Path, driver: Path, sandbox: Path) -> None:
        self.base = base
        self.clone = clone
        self.driver = driver
        self.sandbox = sandbox
        self.output = sandbox / "output"
        self.campaign = self.output / "p5_3c"
        self.token = sandbox / "TOKEN_campaign"
        self.sweep_marker = sandbox / "SWEEP_RAN"

    @property
    def head(self) -> str:
        return _git("rev-parse", "HEAD", cwd=self.clone).strip()

    def write_token(self) -> None:
        self.token.write_text("authorised by tests/test_p5_3c_eval_driver.py\n", encoding="utf-8")

    def run(self, *args: str, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["setsid", "--wait", "bash", str(self.driver), *args],
            capture_output=True, text=True, cwd=str(self.clone), timeout=timeout,
        )
        (self.base / "driver_capture.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        return result

    def contents(self) -> dict[str, bytes | None]:
        found: dict[str, bytes | None] = {}
        for directory, subdirectories, files in os.walk(self.sandbox, followlinks=False):
            for name in subdirectories + files:
                path = Path(directory) / name
                relative = str(path.relative_to(self.sandbox))
                found[relative] = None if path.is_dir() or path.is_symlink() else path.read_bytes()
        return dict(sorted(found.items()))


@pytest.fixture
def eval_sandbox(tmp_path: Path) -> Iterator[Callable[..., EvalSandbox]]:
    """The ONE place an executed campaign driver is built: roots redirected; the inputs check, the re-rolls and the
    stages stubbed; the RSS budget set to zero (``DEFERRED`` 102: a whole-suite run's tmpfs scratch must not decide
    this test); the sweep stub leaves a marker if it is ever reached."""
    main = _needs_the_driver_environment()

    def build(*, register_run_tree: bool = True, real_canary: bool = False) -> EvalSandbox:
        sandbox = tmp_path / "sandbox"
        output = sandbox / "output"
        output.mkdir(parents=True)
        clone = _snapshot_clone(tmp_path)
        path = clone / "offline" / "campaigns" / "p5_3c_eval.sh"
        text = path.read_text(encoding="utf-8")
        if register_run_tree:
            text = _substitute(text, "RUN_TREE=/home/filip/rltraffic-p53c-run\n", f"RUN_TREE={clone}\n", 1)
        text = _substitute(text, "OUTPUT=$MAIN/output\n", f"OUTPUT={output}\n", 1)
        text = _substitute(text, "TOKEN=$OUTPUT/p5_3c_runs/TOKEN_campaign\n", f"TOKEN={sandbox / 'TOKEN_campaign'}\n", 1)
        text = _substitute(text, "RSS_BUDGET_MIB=24216\n", "RSS_BUDGET_MIB=0\n", 1)
        text = _substitute(text, INPUTS_CALL, INPUTS_STUB, 1)
        text = _substitute(text, REROLL_CALL, REROLL_STUB, 1)
        text = _substitute(text, REFERENCE_CALL, REFERENCE_STUB, 1)
        text = _substitute(text, GATE_CALL, GATE_STUB, 1)
        text = _substitute(text, SWEEP_CALL, f"touch {sandbox / 'SWEEP_RAN'}", 1)
        if not real_canary:
            text = _substitute(text, CANARY_CALL, CANARY_STUB, 1)
        path.write_text(text, encoding="utf-8")
        record = clone / "docs" / "data" / "p5_3c_train.json"
        if not record.exists():
            # The clone carries the record gate G3 committed (c55693a, merged from main); a clone without it gets this
            # placeholder, since the driver only tests that it EXISTS (campaign-inputs, which reads it at its pin, is
            # stubbed here).
            record.write_text('{"placeholder": "tests/test_p5_3c_eval_driver.py"}\n', encoding="utf-8")
        _commit_clone(clone, "the campaign driver's roots redirected to the sandbox, its checks and stages stubbed")
        return EvalSandbox(base=tmp_path, clone=clone, driver=path, sandbox=sandbox)

    assert main.is_dir()
    yield build


def test_the_fixture_redirects_every_writable_root_and_stubs_the_stages(eval_sandbox: Any) -> None:
    sb = eval_sandbox()
    code = _code(sb.driver.read_text(encoding="utf-8"))
    assert f"OUTPUT={sb.output}\n" in code and f"TOKEN={sb.token}\n" in code and f"RUN_TREE={sb.clone}\n" in code
    assert "$MAIN/output" not in code and "p5_3c_runs/TOKEN" not in code
    for stub in (INPUTS_STUB, REROLL_STUB, REFERENCE_STUB, GATE_STUB, CANARY_STUB):
        assert stub in code
    # The stubbed MODULE calls are gone; the words survive in `fail "reference-gate"` and the header's echo, by design.
    for call in ("context_sweep campaign-inputs", "context_sweep reference-reroll-check", "context_sweep cells",
                 "context_sweep reference-gate", "transfer_calibration --draws-root \"$DRAWS\" --output-root "
                 "\"$OUTPUT\" --work-dir \"$WORK\" canary"):
        assert call not in code, call
    assert _git("status", "--porcelain", cwd=sb.clone) == ""


def test_without_a_token_every_check_and_the_real_canary_run_and_nothing_is_created(eval_sandbox: Any) -> None:
    sb = eval_sandbox(real_canary=True)
    before = sb.contents()
    result = sb.run(sb.head)
    output = result.stdout + result.stderr
    assert result.returncode == 2, output[-3000:]
    assert "campaign_inputs PASSED" in result.stdout and "resume_check PASSED" in result.stdout
    assert re.search(r"^canary \d+\.\d+ s ", result.stdout, flags=re.MULTILINE)
    assert result.stdout.count("reference_reroll_check MATCH ") == 15
    assert f"REFUSING TO START: no run token at {sb.token}" in output
    assert sb.contents() == before, "a refused start created or changed something"


def test_the_gates_stop_writes_failed_consumes_the_token_and_rolls_no_sweep_cell(eval_sandbox: Any) -> None:
    """*Mutation:* the gate's ``|| fail`` removed -> the sweep stub runs and this dies."""
    sb = eval_sandbox()
    sb.write_token()
    result = sb.run(sb.head)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output[-3000:]
    assert not sb.token.exists(), "the token is consumed on start"
    assert "reference_gate FAILED" in result.stdout
    assert (sb.campaign / "cells" / "FAILED").read_text(encoding="utf-8") == "CAMPAIGN FAILED at reference-gate\n"
    assert (sb.campaign / "cells" / "canary.json").is_file()
    assert not sb.sweep_marker.exists(), "no sweep cell may be rolled after a failed gate"
    assert not (sb.campaign / "artifacts" / "p5_3c_context_sweep.json").exists()
    assert not (sb.output / "SHA256SUMS_p5_3c.txt").exists()
    assert not (sb.campaign / "cells" / "COMPLETE").exists()


def test_a_copy_outside_the_run_worktree_is_refused(eval_sandbox: Any) -> None:
    sb = eval_sandbox(register_run_tree=False)
    sb.write_token()
    before = sb.contents()
    result = sb.run(sb.head)
    assert result.returncode == 2
    assert "not the run worktree /home/filip/rltraffic-p53c-run" in result.stderr
    assert sb.contents() == before


def test_a_run_tree_at_another_commit_is_refused(eval_sandbox: Any) -> None:
    sb = eval_sandbox()
    sb.write_token()
    before = sb.contents()
    other = "0" * 40
    result = sb.run(other)
    assert result.returncode == 2
    assert f"the run worktree is at {sb.head}, not {other}" in result.stderr
    assert sb.contents() == before
