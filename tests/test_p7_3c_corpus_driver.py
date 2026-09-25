"""P7.3c C2 (``BRIEF_41``): the corpus driver ``offline/campaigns/p7_3c_corpus.sh`` -- its text, and its refusals EXECUTED.

T-driver for C2 (``BRIEF_41`` §4):

* **Comment-free text assertions** -- the barrier, the pre-token record and the canary before the trap and the token;
  ``record-canary`` after the token; the corpus, its ``SHA256SUMS`` and then A17(f)'s gate LAST; ``-P`` on every
  interpreter call; ``set -euo pipefail``; one worker; ``--overwrite`` never passed; nothing deleted but the token.
* **The header's documented line** -- the foreground form with ``tee -a`` and ``${PIPESTATUS[0]}`` -- read from the
  header and typed at a tmux pane's prompt.
* **The driver EXECUTED** on a sandbox copy: a committed snapshot clone of this working copy, the corpus, the run
  directory and the token redirected into ``tmp_path``, and the collection call replaced by a stub that collects
  nothing -- so no test can collect an episode, whatever breaks (``BRIEF_39`` B.7-3's rule, by construction).

The main tree's paths are read from the driver's own ``MAIN=`` line rather than restated here.  ⚠️ ``BRIEF_41`` §4's
two harness traps: a mutant must be COMMITTED (the driver refuses a dirty tree), and pytest must be invoked from a
command whose text does not carry the driver's liveness pattern.

GATES -- each ``skip`` names what it consumes: the main tree's interpreter, ``setsid``, A20(a)'s five checkpoints
under the main tree's ``output/``, the 100 parity configs of the probe band, RESCO's grid4x4 network, and ``tmux``
for the header test.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import pytest

import offline.transfer_calibration as tc
from tests.test_p7_3d_campaign_path import _commit_clone, _git, _snapshot_clone, _substitute

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "offline" / "campaigns" / "p7_3c_corpus.sh"
GRID = "cityflow_grid4x4"

#: The collection call, replaced in every executed test: the suite never collects a corpus episode.
COLLECT_CALL = (
    'PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_calibration "${COMMON[@]}" collect-corpus '
    '--corpus-dir "$CORPUS" --draws-range "${DRAWS_RANGE[@]}"'
)
COLLECT_STUB = "sh -c 'echo \"STUB collect-corpus (tests/test_p7_3c_corpus_driver.py): nothing collected\"; exit 97'"


def _text() -> str:
    return DRIVER.read_text(encoding="utf-8")


def _code(text: str) -> str:
    """The driver without its comment lines -- what the shell executes, and nothing a comment could satisfy."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _main_tree(text: str) -> Path:
    """The main tree, as the driver itself declares it (``MAIN=...``)."""
    match = re.search(r"^MAIN=(\S+)$", text, flags=re.MULTILINE)
    assert match, "the driver declares no MAIN="
    return Path(match.group(1))


# ----------------------------------------------------------------------------------------------
# The text, comments stripped
# ----------------------------------------------------------------------------------------------


def test_every_refusal_precedes_the_token_and_the_gate_runs_last() -> None:
    """The barrier, the pre-token record and the canary, then the trap, then the token; afterwards the record, the
    corpus, its sums and the gate, in that order, and no interpreter call after the gate.

    *Mutation:* the gate moved before ``SHA256SUMS`` -> this dies.
    """
    code = _code(_text())
    order = [
        ("the barrier", 'echo "REFUSING TO START: $TARGET is not empty"'),
        ("corpus-preflight", "corpus-preflight --draws-range"),
        ("the canary", '"${COMMON[@]}" canary)'),
        ("the trap", "trap on_signal INT TERM HUP"),
        ("the token check", 'if [ ! -f "$TOKEN" ]; then'),
        ("the token consumed", 'rm -f "$TOKEN"'),
        ("record-canary", "record-canary --line"),
        ("collect-corpus", "collect-corpus --corpus-dir"),
        ("SHA256SUMS written", "mv SHA256SUMS.tmp SHA256SUMS"),
        ("SHA256SUMS verified", "sha256sum -c --quiet SHA256SUMS"),
        ("corpus-gate", "corpus-gate --corpus-dir"),
        ("COMPLETE", 'tee "$RUN_DIR/COMPLETE"'),
    ]
    positions = []
    for label, needle in order:
        assert code.count(needle) == 1, f"{label}: expected ONE {needle!r} in the code, found {code.count(needle)}"
        positions.append(code.index(needle))
    assert positions == sorted(positions), [label for label, _needle in order]
    after_gate = code[code.index("\n", code.index("corpus-gate --corpus-dir")):]
    assert '"$PY"' not in after_gate, "the gate is the LAST module call"


def test_every_interpreter_call_carries_minus_P() -> None:
    """``BRIEF_39`` B.3-2: without ``-P`` the MAIN tree's cwd shadows the worktree's ``offline``."""
    code = _code(_text())
    # Every line that CALLS the interpreter -- the one `[ ! -x "$PY" ]` existence test is not a call (the
    # exclusion tests/test_g2_measure.py makes for P7.3d's drivers).
    calls = [line for line in code.splitlines() if '"$PY"' in line and "-x" not in line]
    assert len(calls) >= 6
    assert [line for line in calls if re.search(r'"\$PY"(?! -P)', line)] == []


def test_strict_mode_one_worker_and_no_overwrite() -> None:
    code = _code(_text())
    assert re.search(r"^set -euo pipefail$", code, flags=re.MULTILINE)
    assert re.search(r"^WORKERS=1$", code, flags=re.MULTILINE), "Amendment A, Q11: one process"
    assert "--overwrite" not in code


def test_nothing_is_deleted_or_moved_but_the_token_and_the_sums_file() -> None:
    """The filesystem-mutation barrier, by text: the ONE deletion is the consumed token, the ONE move the atomic
    rename of the sums file.  *Mutation:* an ``rm -rf "$CORPUS"`` before the collection -> this dies."""
    code = _code(_text())
    assert re.findall(r"\brm\b[^\n]*", code) == ['rm -f "$TOKEN"']
    assert re.findall(r"\bmv [^\s)]+ [^\s)]+", code) == ["mv SHA256SUMS.tmp SHA256SUMS"]


def test_the_driver_derives_its_tree_refuses_the_implementers_and_guards_liveness() -> None:
    code = _code(_text())
    assert 'WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)' in code
    assert 'if [ "$WORK_TREE" = "$IMPLEMENTER_TREE" ]; then' in code
    liveness = re.search(r"pgrep -f '([^']+)'", code)
    assert liveness, "the driver has no liveness guard"
    assert "offline\\.(collect|transfer_calibration|" in liveness.group(1)


def _header_line(text: str) -> tuple[str, str, str]:
    """The header's Step-2 line, and the driver path and capture path it names -- read, never restated."""
    match = re.search(r"^#\s+Step 2, at ITS PROMPT:\s+(bash .+)$", text, flags=re.MULTILINE)
    assert match, "the header has no Step-2 line"
    line = match.group(1)
    parts = re.match(r"bash (\S+p7_3c_corpus\.sh) 2>&1 \| tee -a (\S+); ", line)
    assert parts, f"the Step-2 line is not the foreground form: {line}"
    return line, parts.group(1), parts.group(2)


def test_the_header_documents_the_foreground_form_from_the_run_tree() -> None:
    line, driver_path, capture_path = _header_line(_text())
    assert "${PIPESTATUS[0]}" in line, "B.5-3: the pane carries the DRIVER's exit status"
    assert line.count(f"tee -a {capture_path}") == 2
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
            pytest.skip(f"{path} is absent: corpus-preflight reads A20(a)'s five checkpoints")
    for draw in tc.GRID4X4_CORPUS_DRAWS:
        config = main / "scenarios" / "draws" / GRID / f"draw_{draw:04d}" / "parity" / "noteleport.sumocfg"
        if not config.is_file():
            pytest.skip(f"{config} is absent: the driver refuses without the rendered probe band")
    resco = Path(os.environ.get("RLTRAFFIC_GRID4X4_RESCO") or main / "scenarios" / "grid4x4_candidates")
    if not (resco / "resco" / "resco_benchmark" / "environments" / "grid4x4" / "grid4x4.net.xml").is_file():
        pytest.skip(f"RESCO's grid4x4 network is not under {resco}")
    return main


class CorpusSandbox:
    """One executed driver: a clean snapshot clone, every writable root in ``tmp_path``, the collection stubbed."""

    def __init__(self, *, base: Path, clone: Path, driver: Path, sandbox: Path) -> None:
        self.base = base
        self.clone = clone
        self.driver = driver
        self.sandbox = sandbox
        self.corpus = sandbox / "corpus"
        self.run_dir = sandbox / "run"
        self.token = sandbox / "TOKEN_corpus"

    def write_token(self) -> None:
        self.token.write_text("authorised by tests/test_p7_3c_corpus_driver.py\n", encoding="utf-8")

    def run(self, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
        """``setsid --wait`` makes the driver a process-group leader, which it requires."""
        result = subprocess.run(
            ["setsid", "--wait", "bash", str(self.driver)],
            capture_output=True, text=True, cwd=str(self.clone), timeout=timeout,
        )
        (self.base / "driver_capture.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        return result

    def contents(self) -> set[str]:
        return {str(path.relative_to(self.sandbox)) for path in self.sandbox.rglob("*")}


@pytest.fixture
def corpus_sandbox(tmp_path: Path) -> Callable[..., CorpusSandbox]:
    """B.7-3's rule: the ONE place an executed corpus driver is built -- roots redirected, collection stubbed."""
    _needs_the_driver_environment()

    def build(*, populate: Callable[[CorpusSandbox], None] | None = None) -> CorpusSandbox:
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        clone = _snapshot_clone(tmp_path)
        path = clone / "offline" / "campaigns" / "p7_3c_corpus.sh"
        text = path.read_text(encoding="utf-8")
        text = _substitute(
            text, "CORPUS=$MAIN/datasets_sumo_v11/grid4x4_sumo_maxpressure\n", f"CORPUS={sandbox / 'corpus'}\n", 1
        )
        text = _substitute(text, "RUN_DIR=$MAIN/output/p7_3c_corpus\n", f"RUN_DIR={sandbox / 'run'}\n", 1)
        text = _substitute(
            text, "TOKEN=$MAIN/output/p7_3c_runs/TOKEN_corpus\n", f"TOKEN={sandbox / 'TOKEN_corpus'}\n", 1
        )
        text = _substitute(text, COLLECT_CALL, COLLECT_STUB, 1)
        path.write_text(text, encoding="utf-8")
        _commit_clone(clone, "the corpus driver's roots redirected to the sandbox, its collection stubbed")
        built = CorpusSandbox(base=tmp_path, clone=clone, driver=path, sandbox=sandbox)
        if populate is not None:
            populate(built)
        return built

    return build


def test_the_fixture_redirects_every_writable_root_and_stubs_the_collection(corpus_sandbox: Any) -> None:
    """The rule by construction: the sandbox driver cannot write the real corpus, the real run record or consume the
    real token, and cannot collect an episode.  Its evidence is the mutation that removes the stub."""
    sb = corpus_sandbox()
    code = _code(sb.driver.read_text(encoding="utf-8"))
    assert f"CORPUS={sb.corpus}\n" in code and f"RUN_DIR={sb.run_dir}\n" in code and f"TOKEN={sb.token}\n" in code
    # The EXECUTABLE lines only: the header's comments name the real token path for the author, and a comment runs
    # nothing.
    assert "datasets_sumo_v11" not in code and "p7_3c_runs/TOKEN" not in code
    assert "$MAIN/output/p7_3c_corpus" not in code
    assert COLLECT_STUB in code and "collect-corpus --corpus-dir" not in code
    assert _git("status", "--porcelain", cwd=sb.clone) == ""


def _with_file(relative: str) -> Callable[[CorpusSandbox], None]:
    def populate(sb: CorpusSandbox) -> None:
        target = sb.sandbox / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("left by an earlier run\n", encoding="utf-8")
        sb.write_token()

    return populate


@pytest.mark.parametrize("leftover", ["corpus/ep000000_seed1000_draw201.npz", "run/canary.json"])
def test_a_non_empty_corpus_or_run_directory_is_refused_before_anything_is_consumed(
    corpus_sandbox: Any, leftover: str
) -> None:
    """The barrier, EXECUTED, with a token on disk: exit 2, the directory named, the canary never run, the token
    untouched, nothing written or removed.  *Mutation:* the barrier's refusal removed -> this dies."""
    sb = corpus_sandbox(populate=_with_file(leftover))
    before = sb.contents()
    result = sb.run()
    output = result.stdout + result.stderr
    assert result.returncode == 2, output[-2000:]
    directory = sb.sandbox / Path(leftover).parts[0]
    assert f"REFUSING TO START: {directory} is not empty" in output
    assert not re.search(r"^canary ", result.stdout, flags=re.MULTILINE), "the barrier precedes the canary"
    assert sb.token.is_file(), "a refused start consumed the token"
    assert sb.contents() == before, "a refused start wrote or removed something"


def test_without_a_token_every_check_runs_and_nothing_is_created(corpus_sandbox: Any) -> None:
    """No token: corpus-preflight and the canary run for real, then the refusal -- and still nothing exists."""
    sb = corpus_sandbox()
    result = sb.run()
    output = result.stdout + result.stderr
    assert result.returncode == 2, output[-2000:]
    assert "not a process-group leader" not in output
    assert "corpus_preflight PASSED: draws 201-300 (100) disjoint" in result.stdout
    assert re.search(r"^canary \d+\.\d+ s ", result.stdout, flags=re.MULTILINE)
    assert f"REFUSING TO START: no run token at {sb.token}" in output
    assert sb.contents() == set(), "a refused start created something"


def test_with_a_token_it_is_consumed_the_canary_recorded_and_the_stubbed_collection_fails_the_run(
    corpus_sandbox: Any,
) -> None:
    """Past the token, on the stub: the token consumed ONCE, ``canary.json`` written, ``FAILED`` at collect-corpus,
    no ``COMPLETE``, no gate record, no corpus."""
    sb = corpus_sandbox()
    sb.write_token()
    result = sb.run()
    output = result.stdout + result.stderr
    assert result.returncode == 1, output[-2000:]
    assert not sb.token.exists()
    assert output.count("token consumed and deleted") == 1
    assert (sb.run_dir / "canary.json").is_file()
    assert (sb.run_dir / "FAILED").read_text(encoding="utf-8").strip() == "CORPUS RUN FAILED at collect-corpus"
    assert not (sb.run_dir / "COMPLETE").exists() and not (sb.run_dir / "a17f_gate.json").exists()
    assert not sb.corpus.exists()


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_the_headers_own_line_passes_the_guard_and_the_pane_carries_the_drivers_exit_code(
    corpus_sandbox: Any,
) -> None:
    """B.3-1 + B.5-3, on the redirected COPY, NO token: the header's Step-2 line typed at a pane's prompt passes the
    group-leader check, refuses at the token, and the capture's last line is ``DRIVER EXIT: 2`` -- the same code the
    driver returns under ``setsid``.  *Mutation:* ``${PIPESTATUS[0]}`` -> ``$?`` in the header -> this dies."""
    sb = corpus_sandbox()
    line, driver_path, capture_path = _header_line(sb.driver.read_text(encoding="utf-8"))
    capture = sb.base / "pane_capture.txt"
    line = _substitute(line, driver_path, str(sb.driver), 1)
    line = _substitute(line, capture_path, str(capture), 2)

    session = f"p73c_hdr_{uuid.uuid4().hex[:8]}"
    subprocess.run(["tmux", "new-session", "-d", "-s", session], check=True)
    try:
        subprocess.run(["tmux", "send-keys", "-t", session, line, "Enter"], check=True)
        for _ in range(360):
            if "DRIVER EXIT:" in (capture.read_text(encoding="utf-8") if capture.exists() else ""):
                break
            time.sleep(0.5)
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session], check=False)

    text = capture.read_text(encoding="utf-8")
    assert "not a process-group leader" not in text
    assert "REFUSING TO START: no run token" in text
    assert text.rstrip().splitlines()[-1] == "DRIVER EXIT: 2"
    assert sb.run().returncode == 2
