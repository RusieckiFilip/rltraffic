"""End-to-end tests for the frozen-file guard ``scripts/claude_guard.sh``.

The guard is the mechanism that protects the entire frozen set, and until now it had
**zero** tests -- which is why defect G1 survived several close readings of its regexes
(``docs/notes/D3_falsification.md`` section 4). This file exercises the *real* script,
not a reimplementation of its logic: it copies ``scripts/claude_guard.sh`` byte-for-byte
into a scratch git repository (sha256-verified, so the test cannot drift onto a stale
copy) and asserts the guard's ``--frozen-only`` verdict on a truth table of dirty paths.

Format / behaviour notes the truth table depends on:

* The guard derives everything from ``git status --porcelain``. git **collapses a
  wholly-untracked directory into a single entry**, so a new file deep inside a new
  directory arrives as ``experiments/newpkg/`` rather than ``experiments/newpkg/foo.py``.
  The baseline below therefore tracks at least one file in every directory whose collapse
  depth matters, so the scratch repo collapses at exactly the depths the real repo does.
* Exit ``2`` == BLOCKED (a frozen file was touched), exit ``0`` == PERMITTED.
* Row ``new_pkg_py_G1`` is the G1 regression: a new ``.py`` file inside a new
  ``experiments/`` subdirectory. Against the pre-patch guard it is PERMITTED (the defect);
  the guard must BLOCK it. It is the falsification row this file exists for.
* The ``experiments/configs/`` carve-out is deliberately narrow: ``experiments/configs/[^/]*\.json$``,
  NOT a bare prefix. So ``config_py_not_exempt`` (a ``.py`` heredoc-written into ``configs/``) stays
  BLOCKED -- a bare prefix would have reopened G1's exact hole one directory over -- and
  ``config_subdir_fail_closed`` (a config in a NEW subdirectory, collapsed by git to
  ``experiments/configs/sub/``) fails closed. Nested config trees are not a thing today; if they are
  ever wanted, that should cost a deliberate patch, not leak through the carve-out.

A fresh scratch repo is built per parametrized case, so one case's leftovers can never
contaminate the next (the ``git checkout -- .`` trap recorded in the D3 note is avoided by
construction rather than by cleanup discipline).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_GUARD = REPO_ROOT / "scripts" / "claude_guard.sh"

# Baseline tracked layout. Every directory whose git-status collapse depth matters carries
# at least one tracked file, so untracked children collapse at the real repo's depths.
_BASELINE_TRACKED: dict[str, str | None] = {
    "scripts/claude_guard.sh": None,  # filled with the real guard's exact bytes
    "experiments/runner.py": "# placeholder tracked module\n",
    "experiments/configs/existing.json": "{}\n",
    "envs/pkg.py": "# placeholder tracked module\n",
    "docs/keep.md": "# placeholder\n",
}

# (row id, repo-relative path, action, expected exit code).
#   action "create" makes a new file; "modify" appends to an already-tracked file.
#   exit 2 == BLOCKED, exit 0 == PERMITTED.
_TRUTH_TABLE: list[tuple[str, str, str, int]] = [
    ("modify_tracked_runner",   "experiments/runner.py",         "modify", 2),
    ("new_top_level_py",        "experiments/newfile.py",        "create", 2),
    ("new_pkg_py_G1",           "experiments/newpkg/foo.py",     "create", 2),
    ("new_pkg_sub_py_G1",       "experiments/newpkg/sub/f.py",   "create", 2),
    ("new_pkg_under_envs",      "envs/newpkg/foo.py",            "create", 2),
    ("new_script",              "scripts/brand_new.sh",          "create", 2),
    ("claude_settings",         ".claude/settings.json",         "create", 2),
    ("except_check_english",    "scripts/check_english.sh",      "create", 0),
    ("except_check_hygiene",    "scripts/check_test_hygiene.sh", "create", 0),
    ("config_json_writable",    "experiments/configs/new.json",  "create", 0),
    ("config_py_not_exempt",    "experiments/configs/evil.py",   "create", 2),  # heredoc .py stays BLOCKED
    ("config_subdir_fail_closed", "experiments/configs/sub/new.json", "create", 2),  # nested -> fail closed
    ("docs_control",            "docs/anything.md",              "create", 0),  # PERMITTED control
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )


def _run_guard(repo: Path) -> int:
    """Run the scratch copy of the guard in ``--frozen-only`` mode; return its exit code."""
    proc = subprocess.run(
        ["bash", str(repo / "scripts" / "claude_guard.sh"), "--frozen-only"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return proc.returncode


@pytest.fixture()
def scratch_repo(tmp_path: Path) -> Path:
    """A hermetic git repo with a byte-identical copy of the real guard, committed clean."""
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("git and bash are required to exercise the real guard end-to-end")

    repo = tmp_path / "scratch"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "guard-test@example.com")
    _git(repo, "config", "user.name", "guard test")

    for rel, content in _BASELINE_TRACKED.items():
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            shutil.copyfile(REAL_GUARD, dst)
        else:
            dst.write_text(content, encoding="utf-8")

    # The copied guard must be byte-identical to the repo's, or the test silently drifts
    # onto a stale copy and proves nothing about the guard actually in force.
    assert _sha256(repo / "scripts" / "claude_guard.sh") == _sha256(REAL_GUARD), (
        "copied guard is not byte-identical to scripts/claude_guard.sh"
    )

    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "baseline")

    # Clean tree before any case: the guard must PERMIT it (exit 0). Had the copied guard
    # been left untracked it would dirty git status and BLOCK every case -- the harness trap
    # that once read as a triumphant all-BLOCKED confirmation (D3 note, section 3).
    assert _git(repo, "status", "--porcelain").stdout == "", "baseline tree is not clean"
    assert _run_guard(repo) == 0, "guard blocked a clean tree; the harness is broken"
    return repo


@pytest.mark.parametrize(
    "row_id, rel, action, expected",
    _TRUTH_TABLE,
    ids=[row[0] for row in _TRUTH_TABLE],
)
def test_frozen_guard_truth_table(
    scratch_repo: Path, row_id: str, rel: str, action: str, expected: int
) -> None:
    """The real guard's ``--frozen-only`` verdict on exactly one dirty path."""
    target = scratch_repo / rel
    if action == "modify":
        assert target.exists(), f"{rel} must be tracked in the baseline for a modify case"
        with target.open("a", encoding="utf-8") as handle:
            handle.write("# dirtied by the test\n")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("dirty\n", encoding="utf-8")

    verdict = "BLOCKED" if expected == 2 else "PERMITTED"
    actual = _run_guard(scratch_repo)
    assert actual == expected, (
        f"[{row_id}] expected {verdict} (exit {expected}) for {rel!r}, got exit {actual}"
    )
