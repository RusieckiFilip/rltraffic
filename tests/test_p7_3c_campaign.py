"""P7.3c C7 (``BRIEF_41`` C7; Amendment F, F3.2): the campaign driver ``offline/campaigns/p7_3c_grid4x4.sh``, P7.3c's
``check-inputs`` and its manifest.

* **check-inputs, P7.3c's.**  P7.3d's inputs, then the zero-shot artifact and the training record at their pins, the
  training manifest against the record, and the thirty -- each resolved through the pinned record (the fence, the
  digests, F4's guard) AND validated against its source (the frozen parts: ``few_shot.validate_checkpoint``).
* **The manifest**: ``output/SHA256SUMS_p7_3c.txt`` over ``output/p7_3c/`` only, as P7.3d's is over its own.
* **T-driver (campaign)**, comment-free text: every refusal before ONE token; after it, the three stages in order with
  ``stage1-check`` between stage 1 and stage 2, its exit enforced; ``report`` and the manifest last; ``-P`` on every
  interpreter call; the run tree AND its commit enforced; FAILED on every path after the token.
* **The driver EXECUTED** on a sandbox copy -- P7.3d's pattern (``tests/test_p7_3d_campaign_path.py``): a committed
  snapshot clone registered as the run tree, the campaign directory and the token in ``tmp_path``, both re-roll checks
  stubbed, and ``--limit 0`` on all three ``cells`` lines, so no test can roll a cell whatever breaks.  One mismatching
  stage-1 chunk among 700 -> FAILED at the gate and no stage-2 chunk; a complete 4,700-chunk set -> COMPLETE, the
  artifact and the manifest.

GATES, each named by its ``skip``: the main tree's interpreter, ``setsid``, the rendered parity band, A20(a)'s five and
the thirty trained checkpoints with their manifest under the main tree's ``output/``, and ``tmux`` for the header test.
Nothing here runs while a campaign or a training is live (``BRIEF_41`` Amendment B, B5).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import pytest

import offline.few_shot as few_shot
import offline.transfer_calibration as tc
import offline.transfer_curve as tcv
from tests.test_p7_3c_report import _committed, _stage1_chunk, build_report_set
from tests.test_p7_3d_campaign_path import (
    _IDENTICAL_STUB,
    _REFERENCE_CALL,
    _REROLL_CALL,
    _commit_clone,
    _git,
    _six_match_stub,
    _snapshot_clone,
    _substitute,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"
DRIVER = REPO_ROOT / "offline" / "campaigns" / "p7_3c_grid4x4.sh"
MAIN = Path("/home/filip/rltraffic")
OUTPUT_ROOT = MAIN / "output"
DRAWS_ROOT = MAIN / "scenarios" / "draws"
INTERPRETER = MAIN / ".venv" / "bin" / "python"
RUN_TREE = "/home/filip/rltraffic-p73c-run"
HEADER_CAPTURE = "/home/filip/rltraffic/output/p7_3c_runs/campaign_capture.txt"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _needs_the_real_inputs() -> None:
    for seed in tc.TRAINING_SEEDS:
        path = OUTPUT_ROOT / tc.GRID4X4_CHECKPOINT_SUBDIR / f"{tc.GRID4X4_CHECKPOINT_STEM}{seed}.pt"
        if not path.is_file():
            pytest.skip(f"{path} is absent: check-inputs reads A20(a)'s five checkpoints")
    checkpoints = OUTPUT_ROOT / "p7_3c_training" / "checkpoints"
    if len(list(checkpoints.glob("*.pt"))) != 30 or not (OUTPUT_ROOT / few_shot.MANIFEST_FILENAME).is_file():
        pytest.skip(f"{checkpoints} and its manifest are not the thirty trained checkpoints (gitignored, main tree only)")
    for draw in tcv.HELD_OUT_DRAWS:
        config = DRAWS_ROOT / "cityflow_grid4x4" / f"draw_{draw:04d}" / "parity" / "noteleport.sumocfg"
        if not config.is_file():
            pytest.skip(f"{config} is absent: the rendered parity band lives in the main tree only")


def _inputs(output_root: Path = OUTPUT_ROOT) -> list[str]:
    return tcv.check_p7_3c_campaign_inputs(data_dir=DATA, output_root=output_root, draws_root=DRAWS_ROOT)


# ==================================================================================================================
# check-inputs, P7.3c's
# ==================================================================================================================


def test_the_p7_3c_inputs_pass_on_the_real_tree_and_name_each_digest() -> None:
    _needs_the_real_inputs()
    lines = _inputs()
    p7_3d = tcv.check_campaign_inputs(data_dir=DATA, output_root=OUTPUT_ROOT, draws_root=DRAWS_ROOT)
    assert lines[: len(p7_3d)] == p7_3d
    assert any(tcv.P7_3D_GRID4X4_SHA256 in line for line in lines)
    assert any(str(tcv.P7_3C_FINETUNE_SHA256) in line for line in lines)
    assert any(_sha(OUTPUT_ROOT / few_shot.MANIFEST_FILENAME) in line for line in lines)
    assert any("30/30" in line for line in lines)


def test_each_of_the_thirty_is_resolved_through_the_record_and_validated_against_its_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _needs_the_real_inputs()
    resolved: list[str] = []
    validated: list[str] = []
    identity, validate = tcv.p7_3c_trained_checkpoint_identity, few_shot.validate_checkpoint

    def resolving(subject: str, seed: int, **kwargs: Any) -> Any:
        resolved.append(f"{subject}_seed{seed}")
        return identity(subject, seed, **kwargs)

    def validating(spec: Any, **kwargs: Any) -> Any:
        validated.append(spec.name)
        return validate(spec, **kwargs)

    monkeypatch.setattr(tcv, "p7_3c_trained_checkpoint_identity", resolving)
    monkeypatch.setattr(few_shot, "validate_checkpoint", validating)
    _inputs()
    names = sorted(spec.name for spec in few_shot.registered_runs())
    assert sorted(resolved) == names and sorted(validated) == names


def test_a_frozen_part_that_differs_from_its_source_is_refused_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _needs_the_real_inputs()
    validate = few_shot.validate_checkpoint

    def tampered(spec: Any, **kwargs: Any) -> dict[str, bool]:
        checks = validate(spec, **kwargs)
        return {**checks, "stats": False} if spec.name == "ft_k20_seed404" else checks

    monkeypatch.setattr(few_shot, "validate_checkpoint", tampered)
    with pytest.raises(ValueError, match=r"ft_k20_seed404 does not validate against its source \(failed: \['stats'\]\)"):
        _inputs()


def _mirrored_output(tmp_path: Path) -> Path:
    """An output root mirroring the real one by symlink -- A20(a)'s five, the thirty -- with the manifest COPIED."""
    root = tmp_path / "output"
    root.mkdir(parents=True)
    (root / "p5_2").symlink_to(OUTPUT_ROOT / "p5_2")
    (root / "p7_3c_training").symlink_to(OUTPUT_ROOT / "p7_3c_training")
    shutil.copyfile(OUTPUT_ROOT / few_shot.MANIFEST_FILENAME, root / few_shot.MANIFEST_FILENAME)
    return root


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("absent", r"SHA256SUMS_p7_3c_finetune\.txt is absent"),
        ("edited", r"SHA256SUMS_p7_3c_finetune\.txt has sha256 [0-9a-f]{64}, not the [0-9a-f]{64} the pinned record names"),
    ],
)
def test_a_training_manifest_that_is_absent_or_moved_is_refused(tmp_path: Path, case: str, message: str) -> None:
    _needs_the_real_inputs()
    root = _mirrored_output(tmp_path)
    manifest = root / few_shot.MANIFEST_FILENAME
    if case == "absent":
        manifest.unlink()
    else:
        text = manifest.read_text(encoding="utf-8")
        manifest.write_text("0" * 64 + text[64:], encoding="utf-8")
    assert len(_inputs(_mirrored_output(tmp_path / "control"))) > 0  # the same mirror, untouched, passes
    with pytest.raises(ValueError, match=message):
        _inputs(root)


def test_the_check_inputs_command_takes_the_campaign(capsys: pytest.CaptureFixture[str]) -> None:
    _needs_the_real_inputs()
    common = ["--data-dir", str(DATA), "--output-root", str(OUTPUT_ROOT), "--draws-root", str(DRAWS_ROOT)]
    assert tcv.main([*common, "check-inputs", "--campaign", "p7_3c"]) == 0
    assert "30/30" in capsys.readouterr().out
    assert tcv.main([*common, "check-inputs"]) == 0  # P7.3d's, by default, exactly as its driver calls it
    assert "30/30" not in capsys.readouterr().out


# ==================================================================================================================
# The manifest: output/SHA256SUMS_p7_3c.txt over output/p7_3c/ ONLY
# ==================================================================================================================


def test_the_p7_3c_manifest_covers_its_own_directory_only_and_verifies_with_sha256sum(tmp_path: Path) -> None:
    output = tmp_path / "output"
    for relative in ("p7_3c/cells/cell_b.json", "p7_3c/cells/cell_a.json", "p7_3c/artifacts/p7_3c_grid4x4.json",
                     "p7_3c/g2/verdict.json", "p7_3d/cells/cell_x.json"):
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
        (output / relative).write_text(relative + "\n", encoding="utf-8")
    record = tcv.write_manifest(campaign_dir=output / "p7_3c")
    manifest = output / "SHA256SUMS_p7_3c.txt"
    assert record["path"] == str(manifest) and record["n_files"] == 4
    lines = manifest.read_text(encoding="utf-8").splitlines()
    names = [line.split("  ", 1)[1] for line in lines]
    assert names == ["p7_3c/artifacts/p7_3c_grid4x4.json", "p7_3c/cells/cell_a.json", "p7_3c/cells/cell_b.json",
                     "p7_3c/g2/verdict.json"]
    for line in lines:
        digest, name = line.split("  ", 1)
        assert digest == _sha(output / name)
    assert subprocess.run(["sha256sum", "-c", "--quiet", manifest.name], cwd=output).returncode == 0
    assert not (output / "SHA256SUMS_p7_3d.txt").exists()
    with pytest.raises(ValueError, match="p7_3c"):
        tcv.write_manifest(campaign_dir=output / "p7_3b_anchor")


# ==================================================================================================================
# T-driver (campaign): the text
# ==================================================================================================================


def _code(text: str) -> str:
    """The driver without its comments: text assertions are about what RUNS."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _in_order(code: str, order: list[tuple[str, str]]) -> None:
    position = -1
    for label, needle in order:
        found = code.find(needle, position + 1)
        assert found > position, f"{label} ({needle!r}) is missing or out of order"
        position = found


def test_every_refusal_precedes_the_one_token_and_the_three_stages_follow_it_in_order() -> None:
    code = _code(DRIVER.read_text(encoding="utf-8"))
    _in_order(
        code,
        [
            ("the commit argument", "EXPECTED_COMMIT=${1:-}"),
            ("the run tree", 'if [ "$WORK_TREE" != "$RUN_TREE" ]'),
            ("its commit", 'if [ "$HEAD_COMMIT" != "$EXPECTED_COMMIT" ]'),
            ("liveness", "pgrep -f 'python.*offline\\.(collect|transfer_calibration|transfer_curve|few_shot|g2_measure)'"),
            ("the group leader", "ps -o pgid= -p $$"),
            ("the inputs by digest", "check-inputs --campaign p7_3c"),
            ("the dirty tree", 'git -C "$WORK_TREE" status --porcelain'),
            ("the RSS budget", "MemAvailable"),
            ("the layout", 'refuse_non_directories "$CAMPAIGN_DIR"'),
            ("resume-check", "resume-check --stage p7_3c"),
            ("the canary", '"${COMMON[@]}" canary)'),
            ("the DT re-roll check", 'dt-reroll-check --g2-dir "$G2_DIR"'),
            ("the reference re-roll check", 'reference-reroll-check --g2-dir "$G2_DIR"'),
            ("the EXIT trap", "trap on_exit EXIT"),
            ("the token", 'rm -f "$TOKEN"'),
            ("record-canary", 'record-canary --line "$CANARY_LINE"'),
            ("stage 1", "cells --stage p7_3c_reproduce"),
            ("the gate", '"${COMMON[@]}" stage1-check)'),
            ("the gate's refusal", 'fail "stage1-check"'),
            ("stage 2", "cells --stage p7_3c_primary"),
            ("stage 3", "cells --stage p7_3c_controls"),
            ("report", "report --stage p7_3c"),
            ("the manifest", 'manifest --campaign-dir "$CAMPAIGN_DIR"'),
            ("success", "SUCCESS=1"),
            ("COMPLETE", '> "$WORK/COMPLETE"'),
        ],
    )
    assert code.count('rm -f "$TOKEN"') == 1 and code.count("TOKEN=") == 1
    assert "TOKEN=$MAIN/output/p7_3c_runs/TOKEN_campaign\n" in code
    assert "CAMPAIGN_DIR=$MAIN/output/p7_3c\n" in code and "WORKERS=12\n" in code and "RSS_BUDGET_MIB=24216\n" in code
    assert "set -euo pipefail" in code


def test_every_interpreter_call_carries_minus_p_and_no_outcome_is_printed() -> None:
    code = _code(DRIVER.read_text(encoding="utf-8"))
    # the finetune driver's selector: `[ ! -x "$PY" ]` is an existence check, not a call
    calls = [line for line in code.splitlines() if '"$PY"' in line and "-x" not in line]
    assert len(calls) >= 12
    assert [line for line in calls if re.search(r'"\$PY"(?! -P)', line)] == []
    # No outcome of any cell: the driver never reads a chunk or the artifact back, and the gate prints one line of
    # names. (`report` prints the artifact's path and its cell count; `cells` prints names and wall clocks.)
    for forbidden in ("cat ", "jq ", "grep -o", "rho", "e_sumo", "att_env"):
        assert forbidden not in code, forbidden


def test_failed_is_written_on_every_path_after_the_token() -> None:
    code = _code(DRIVER.read_text(encoding="utf-8"))
    assert "SUCCESS=0" in code and 'trap on_signal INT TERM HUP' in code
    assert "printf 'CAMPAIGN FAILED (exit %s)\\n'" in code
    assert "printf 'CAMPAIGN FAILED at %s\\n'" in code and "printf 'CAMPAIGN INTERRUPTED by a signal\\n'" in code
    token = code.index('rm -f "$TOKEN"')
    assert code.index("trap on_exit EXIT") < token and code.index("trap on_signal INT TERM HUP") < token


def _header_line(text: str) -> str:
    (line,) = [line.lstrip("# ").split("ITS PROMPT:", 1)[1].strip() for line in text.splitlines()
               if "ITS PROMPT:" in line]
    return line


def test_the_header_documents_the_foreground_form_from_the_run_tree() -> None:
    line = _header_line(DRIVER.read_text(encoding="utf-8"))
    assert line == (
        f"bash {RUN_TREE}/offline/campaigns/p7_3c_grid4x4.sh <commit> 2>&1 | tee -i -a {HEADER_CAPTURE}; "
        f'echo "DRIVER EXIT: ${{PIPESTATUS[0]}}" | tee -i -a {HEADER_CAPTURE}'
    )


# ==================================================================================================================
# T-driver (campaign): EXECUTED on a sandbox copy
# ==================================================================================================================


class CampaignSandbox:
    """One executed driver: a clean snapshot clone as the run tree, the campaign and the token in ``tmp_path``."""

    def __init__(self, *, base: Path, clone: Path, driver: Path, sandbox: Path) -> None:
        self.base = base
        self.clone = clone
        self.driver = driver
        self.sandbox = sandbox
        self.token = sandbox / "TOKEN_campaign"
        self.campaign = sandbox / "p7_3c"
        self.work = self.campaign / "cells"
        self.artifacts = self.campaign / "artifacts"

    @property
    def head(self) -> str:
        return _git("rev-parse", "HEAD", cwd=self.clone).strip()

    def write_token(self) -> None:
        self.token.write_text("authorised by tests/test_p7_3c_campaign.py\n", encoding="utf-8")

    def run(self, commit: str | None = None, timeout: float = 1800.0) -> subprocess.CompletedProcess[str]:
        """``setsid --wait`` makes the driver a process-group leader, which it requires."""
        result = subprocess.run(
            ["setsid", "--wait", "bash", str(self.driver), commit or self.head],
            capture_output=True, text=True, cwd=str(self.clone), timeout=timeout,
        )
        (self.base / "driver_capture.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        return result

    def chunks(self) -> set[str]:
        return {path.name for path in self.sandbox.rglob("cell_*.json")}

    def contents(self) -> dict[str, bytes | None]:
        return {
            str(path.relative_to(self.sandbox)): (None if path.is_dir() else path.read_bytes())
            for path in sorted(self.sandbox.rglob("*"))
        }


def _needs_the_driver_environment() -> None:
    if not INTERPRETER.is_file():
        pytest.skip(f"needs the main tree's interpreter at {INTERPRETER}")
    if shutil.which("setsid") is None:
        pytest.skip("setsid is not installed")
    _needs_the_real_inputs()


#: The three ``cells`` lines, each given ``--limit 0`` in every executed test: no test can roll a cell.
CELLS_LINES = tuple(
    f'cells --stage {stage} --workers "$WORKERS" || fail "cells {stage}"'
    for stage in ("p7_3c_reproduce", "p7_3c_primary", "p7_3c_controls")
)


@pytest.fixture
def campaign_sandbox(tmp_path: Path) -> Callable[..., CampaignSandbox]:
    """B.7-3's rule: the ONE place an executed campaign driver is built -- roots redirected, re-rolls stubbed,
    ``--limit 0`` on every ``cells`` line."""
    _needs_the_driver_environment()

    def build(
        *, register_run_tree: bool = True, populate: Callable[[CampaignSandbox], None] | None = None,
        edit: Callable[[str], str] | None = None,
    ) -> CampaignSandbox:
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        clone = _snapshot_clone(tmp_path)
        path = clone / "offline" / "campaigns" / "p7_3c_grid4x4.sh"
        text = path.read_text(encoding="utf-8")
        if register_run_tree:
            text = _substitute(text, f"RUN_TREE={RUN_TREE}\n", f"RUN_TREE={clone}\n", 1)
        text = _substitute(text, "CAMPAIGN_DIR=$MAIN/output/p7_3c\n", f"CAMPAIGN_DIR={sandbox / 'p7_3c'}\n", 1)
        text = _substitute(
            text, "TOKEN=$MAIN/output/p7_3c_runs/TOKEN_campaign\n", f"TOKEN={sandbox / 'TOKEN_campaign'}\n", 1
        )
        text = _substitute(text, _REROLL_CALL, _IDENTICAL_STUB, 1)
        text = _substitute(text, _REFERENCE_CALL, _six_match_stub(), 1)
        for line in CELLS_LINES:
            text = _substitute(text, line, line.replace('--workers "$WORKERS"', '--workers "$WORKERS" --limit 0'), 1)
        if edit is not None:
            text = edit(text)
        path.write_text(text, encoding="utf-8")
        _commit_clone(clone, "the campaign driver's roots redirected to the sandbox, its re-rolls stubbed, limit 0")
        built = CampaignSandbox(base=tmp_path, clone=clone, driver=path, sandbox=sandbox)
        if populate is not None:
            populate(built)
        return built

    return build


def test_the_fixture_redirects_every_writable_root_and_limits_every_cells_line(campaign_sandbox: Any) -> None:
    sb = campaign_sandbox()
    code = _code(sb.driver.read_text(encoding="utf-8"))
    assert f"CAMPAIGN_DIR={sb.campaign}\n" in code and f"TOKEN={sb.token}\n" in code and f"RUN_TREE={sb.clone}\n" in code
    assert "$MAIN/output/p7_3c\n" not in code and "p7_3c_runs/TOKEN" not in code
    assert code.count("--limit 0") == 3 and "dt-reroll-check" not in code and "reference-reroll-check" not in code
    assert _git("status", "--porcelain", cwd=sb.clone) == ""


def test_without_a_token_every_check_runs_and_nothing_is_created(campaign_sandbox: Any) -> None:
    sb = campaign_sandbox()
    before = sb.contents()
    result = sb.run()
    output = result.stdout + result.stderr
    assert result.returncode == 2, output[-3000:]
    assert "30/30" in result.stdout and re.search(r"^canary \d+\.\d+ s ", result.stdout, flags=re.MULTILINE)
    assert "dt_reroll_check IDENTICAL" in result.stdout
    assert len(re.findall(r"^reference_reroll_check MATCH ", result.stdout, flags=re.MULTILINE)) == 6
    assert f"REFUSING TO START: no run token at {sb.token}" in output
    assert sb.contents() == before, "a refused start created or changed something"


def test_a_copy_outside_the_run_tree_is_refused_before_anything_is_consumed(campaign_sandbox: Any) -> None:
    outside = campaign_sandbox(register_run_tree=False)
    outside.write_token()
    before = outside.contents()
    result = outside.run()
    assert result.returncode == 2
    assert f"this copy is in {outside.clone}, not the run worktree {RUN_TREE}" in result.stdout + result.stderr
    assert outside.contents() == before


def test_a_run_tree_at_another_commit_is_refused_before_anything_is_consumed(campaign_sandbox: Any) -> None:
    sb = campaign_sandbox()
    sb.write_token()
    before = sb.contents()
    result = sb.run("0" * 40)
    assert result.returncode == 2
    assert f"the run worktree is at {sb.head}, not {'0' * 40}" in result.stdout + result.stderr
    assert sb.contents() == before


def test_a_work_root_that_cannot_be_made_after_the_token_still_leaves_failed_beside_the_token(
    campaign_sandbox: Any,
) -> None:
    """Amendment H, H3.2 (the G8 reviewer's R5-m3): the token is consumed BEFORE the work directory exists, so a failed
    ``mkdir`` would leave no FAILED marker.  The EXIT trap falls back to ``FAILED_campaign`` beside the token.
    *Mutant:* the fallback removed -> this dies."""

    def unwritable(sb: CampaignSandbox) -> None:
        sb.campaign.mkdir(parents=True)
        sb.campaign.chmod(0o555)
        sb.write_token()

    sb = campaign_sandbox(populate=unwritable)
    try:
        result = sb.run()
    finally:
        sb.campaign.chmod(0o755)
    assert result.returncode == 1, (result.stdout + result.stderr)[-3000:]
    assert not sb.token.exists() and not sb.work.exists()
    assert (sb.sandbox / "FAILED_campaign").read_text(encoding="utf-8").strip() == "CAMPAIGN FAILED (exit 1)"


def _stray_chunk(work: Path) -> Path:
    """A chunk file no cell of P7.3c's declaration names (k = 50 is not registered)."""
    stray = work / "cell_cityflow_grid4x4_ft_k50_b_mean_k100_seed101_draw1000.json"
    work.mkdir(parents=True, exist_ok=True)
    stray.write_text("{}\n", encoding="utf-8")
    return stray


def test_resume_check_refuses_an_undeclared_chunk_name_and_leaves_p7_3ds_stage_as_it_was(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Amendment H, H3.3 (the G8 reviewer's R5-m2): a chunk file no declared cell names is refused BEFORE the token, not by
    ``report`` after the campaign.  *Mutant:* the refusal removed -> this dies."""
    work = tmp_path / "cells"
    stray = _stray_chunk(work)
    declared = work / tcv.cell_chunk_name(tcv.declared_cells(tcv.STAGE_P7_3C_PRIMARY)[0])
    declared.write_text("{}\n", encoding="utf-8")  # a declared NAME is not a stray, whatever its content
    for stage in (tcv.STAGE_P7_3C, tcv.STAGE_P7_3C_REPRODUCE):
        assert tcv.main(["--work-dir", str(work), "resume-check", "--stage", stage]) == 2
        err = capsys.readouterr().err
        assert f"resume-check: {stray.name} is not a declared cell of P7.3c's declaration" in err
        assert declared.name not in err
    stray.unlink()
    assert tcv.main(["--work-dir", str(work), "resume-check", "--stage", tcv.STAGE_P7_3C]) == 0
    capsys.readouterr()
    # P7.3d's stage is untouched by the new refusal
    _stray_chunk(work)
    assert tcv.main(["--work-dir", str(work), "resume-check", "--stage", tcv.STAGE_GRID4X4]) == 0


def test_an_undeclared_chunk_name_refuses_the_driver_before_the_token(campaign_sandbox: Any) -> None:
    def populate(sb: CampaignSandbox) -> None:
        _stray_chunk(sb.work)
        sb.write_token()

    sb = campaign_sandbox(populate=populate)
    before = sb.contents()
    result = sb.run()
    output = result.stdout + result.stderr
    assert result.returncode == 2, output[-3000:]
    assert "is not a declared cell of P7.3c's declaration" in output
    assert sb.token.is_file() and sb.contents() == before


#: A step after the token and after the work directory exists, turned into a failure no ``fail`` guards.
INJECTED_AFTER_TOKEN = 'echo "  started      $(date -Is)"'


def test_an_unguarded_failure_after_the_token_still_leaves_failed_through_the_exit_trap(campaign_sandbox: Any) -> None:
    sb = campaign_sandbox(edit=lambda text: _substitute(text, INJECTED_AFTER_TOKEN, "false", 1))
    sb.write_token()
    result = sb.run()
    assert result.returncode == 1, (result.stdout + result.stderr)[-3000:]
    assert not sb.token.exists()
    assert (sb.work / "FAILED").read_text(encoding="utf-8").strip() == "CAMPAIGN FAILED (exit 1)"
    assert not (sb.work / "COMPLETE").exists()


def _stage1_only(mismatch: bool) -> Callable[[CampaignSandbox], None]:
    """Stage 1's 700 chunks from the committed records at the clone's commit -- one of them changed if asked."""

    def populate(sb: CampaignSandbox) -> None:
        committed = _committed()
        events: dict[str, list[dict[str, Any]]] = {}
        for block in committed["collisions"]["per_arm"].values():
            for event in block["events"]:
                events.setdefault(str(event["cell"]), []).append(
                    {key: value for key, value in event.items() if key not in ("cell", "arm", "seed", "draw_id")}
                )
        sb.work.mkdir(parents=True)
        for record in committed["cells"]:
            name = tcv.cell_chunk_name(record)
            chunk = _stage1_chunk(record, events.get(name, []), git_commit=sb.head)
            if mismatch and (record["arm"], record["draw_id"]) == ("maxpressure", 1044):
                chunk["n_created"] = int(chunk["n_created"]) + 1
            (sb.work / name).write_text(json.dumps(chunk), encoding="utf-8")
        sb.write_token()

    return populate


def test_one_mismatching_stage1_chunk_among_700_stops_the_campaign_before_stage_2(campaign_sandbox: Any) -> None:
    """A24(c): ANY mismatch stops the campaign before stage 2's first cell -- the driver's gate, with no human step.
    *Mutation:* the gate's exit ignored -> stage 2 starts -> this dies."""
    sb = campaign_sandbox(populate=_stage1_only(mismatch=True))
    stage1 = sb.chunks()
    assert len(stage1) == 700
    result = sb.run()
    output = result.stdout + result.stderr
    assert result.returncode == 1, output[-3000:]
    assert not sb.token.exists()
    assert "stage1_check NOT REPRODUCED 1/700" in result.stdout
    assert (sb.work / "FAILED").read_text(encoding="utf-8").strip() == "CAMPAIGN FAILED at stage1-check"
    assert sb.chunks() == stage1, "a chunk beyond stage 1's was written"
    assert '"stage": "p7_3c_primary"' not in result.stdout, "stage 2's cells ran"
    assert not (sb.work / "COMPLETE").exists() and not sb.artifacts.joinpath("p7_3c_grid4x4.json").exists()


def test_a_complete_set_runs_through_the_gate_the_report_and_the_manifest_to_complete(campaign_sandbox: Any) -> None:
    def populate(sb: CampaignSandbox) -> None:
        build_report_set(sb.work, git_commit=sb.head)
        sb.write_token()

    sb = campaign_sandbox(populate=populate)
    result = sb.run()
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    assert "stage1_check REPRODUCED 700/700" in result.stdout
    assert (sb.work / "COMPLETE").is_file() and not (sb.work / "FAILED").exists()
    artifact = json.loads((sb.artifacts / "p7_3c_grid4x4.json").read_bytes())
    assert artifact["format_version"] == "p7.3c-grid4x4/1.0" and len(artifact["cells"]) == 4700
    manifest = sb.sandbox / "SHA256SUMS_p7_3c.txt"
    listed = [line.split("  ", 1)[1] for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert "p7_3c/artifacts/p7_3c_grid4x4.json" in listed and all(name.startswith("p7_3c/") for name in listed)
    assert len([name for name in listed if "/cell_" in name]) == 4700


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_the_headers_own_line_passes_the_guard_and_the_pane_carries_the_drivers_exit_code(
    campaign_sandbox: Any,
) -> None:
    sb = campaign_sandbox()
    capture = sb.base / "pane_capture.txt"
    line = _header_line(sb.driver.read_text(encoding="utf-8"))
    line = _substitute(line, f"{RUN_TREE}/offline/campaigns/p7_3c_grid4x4.sh", str(sb.driver), 1)
    line = _substitute(line, HEADER_CAPTURE, str(capture), 2)
    line = _substitute(line, "<commit>", sb.head, 1)
    session = f"p73c_campaign_{uuid.uuid4().hex[:8]}"
    subprocess.run(["tmux", "new-session", "-d", "-s", session], check=True)
    try:
        subprocess.run(["tmux", "send-keys", "-t", session, line, "Enter"], check=True)
        for _ in range(1200):
            if "DRIVER EXIT:" in (capture.read_text(encoding="utf-8") if capture.exists() else ""):
                break
            time.sleep(0.5)
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session], check=False)
    text = capture.read_text(encoding="utf-8")
    assert "not a process-group leader" not in text
    assert "REFUSING TO START: no run token" in text
    assert text.rstrip().splitlines()[-1] == "DRIVER EXIT: 2"
