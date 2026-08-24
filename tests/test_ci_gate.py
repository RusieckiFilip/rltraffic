"""Tests for the CI gates in ``.github/ci/ci_gate.py`` (gate format version 1).

Every test here invokes the **real** script as a subprocess, exactly as
``.github/workflows/ci.yml`` does -- the style of ``tests/test_claude_guard.py``, which
exercises the real guard rather than a reimplementation of its logic.  A gate tested
through a reimplementation proves the reimplementation.

**What these tests are for.**  ``BRIEF_23`` section 3 makes the point this file exists to
answer: a CI job that cannot go red is decoration.  So the majority of the cases below are
*red* cases -- a disagreement between the two parsers, a skip above the ceiling, a guard
regression, a guard that produced no output at all -- and each asserts the **reason**
printed on stdout, not merely a non-zero exit code.  Asserting the code alone would pass
against a stub that raises ``NotImplementedError``, since an uncaught exception also exits
1; that is precisely the false green this file must not have.

Conventions the assertions depend on:

* exit ``0`` = gate passed, ``1`` = gate failed, ``2`` = usage error;
* every verdict line starts with ``OK``, ``FAIL`` or ``NOTE`` on **stdout**;
* a guard finding is keyed by ``(file, rule)`` and counted -- never by line number, which
  moves under any edit, and never by a bare total, which lets a fix in one file pay for a
  new violation in another.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE = REPO_ROOT / ".github" / "ci" / "ci_gate.py"
BASELINE_FILE = REPO_ROOT / ".github" / "ci" / "ci_baseline.json"
WORKFLOW_FILE = REPO_ROOT / ".github" / "workflows" / "ci.yml"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def run_gate(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the real gate script the way the workflow does."""
    return subprocess.run(
        [sys.executable, str(GATE), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def junit_xml(tests: int, skipped: int, failures: int = 0, errors: int = 0) -> str:
    """A minimal JUnit report in the shape ``pytest --junit-xml`` writes."""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<testsuites>"
        f'<testsuite name="pytest" errors="{errors}" failures="{failures}" '
        f'skipped="{skipped}" tests="{tests}" time="89.98" timestamp="2026-08-17T00:00:00">'
        "</testsuite>"
        "</testsuites>\n"
    )


def write_run(
    tmp_path: Path,
    tail: str,
    *,
    tests: int,
    skipped: int,
    failures: int = 0,
    errors: int = 0,
) -> tuple[Path, Path]:
    """Write a terminal-tail file and a JUnit file; return both paths."""
    terminal = tmp_path / "pytest.txt"
    terminal.write_text(
        "tests/test_x.py ....\n\n" + tail + "\n",
        encoding="utf-8",
    )
    junit = tmp_path / "junit.xml"
    junit.write_text(junit_xml(tests, skipped, failures, errors), encoding="utf-8")
    return terminal, junit


def write_baseline(tmp_path: Path, *, ceiling: int, guards: dict | None = None) -> Path:
    """Write a baseline file with the schema the gate reads."""
    payload = {
        "schema_version": 1,
        "pytest": {
            "skip_ceiling": ceiling,
            "provisional": True,
            "re_measure_required_at": {
                "event": "merge of P5.1",
                "reason": "P5.1 adds corpus-gated tests",
                "mandated_by": "coordinator ruling 2026-08-17",
            },
        },
        "guards": guards
        or {
            "hygiene": {
                "script": "scripts/check_test_hygiene.sh",
                "total": 3,
                "by_file": {
                    "tests/test_phase_control.py": {"TH006": 2},
                    "tests/test_base_traffic_env.py": {"TH006": 1},
                },
            },
            "english": {
                "script": "scripts/check_english.sh",
                "total": 1,
                "by_file": {"README.md": {"PL": 1}},
            },
        },
    }
    path = tmp_path / "ci_baseline.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# The hygiene guard is line-oriented: it scans this file's *text*, not its AST.  A fixture
# quoting the guard's own output verbatim is therefore counted as fresh TH006 findings in
# this file -- a real regression against the very baseline this file records, and it fired
# on the first draft.  The waiver comment the guard supports cannot help, because a comment
# inside a triple-quoted string is data rather than a comment.  So the quoted source lines
# are assembled from fragments: at run time the fixture is byte-identical to what
# ``scripts/check_test_hygiene.sh`` prints, and no line of this file matches its pattern.
_RAISES = "pytest." + "raises"
_TH006 = f"[TH006] {_RAISES} without match=: any error of that class satisfies it"

HYGIENE_OUTPUT = (
    f"tests/test_phase_control.py:41: {_TH006}\n"
    f"    with {_RAISES}(ValueError):\n"
    f"tests/test_phase_control.py:58: {_TH006}\n"
    f"    with {_RAISES}(ValueError):\n"
    f"tests/test_base_traffic_env.py:77: {_TH006}\n"
    f"    with {_RAISES}(KeyError):\n"
    "\n"
    "Test-hygiene check failed (scripts/check_test_hygiene.sh).\n"
)

ENGLISH_OUTPUT = """BLOCKED: non-English text found. CLAUDE.md section 3 requires every on-disk artifact in English:
README.md:30

Translate the prose. If a hit is a proper noun, add it to ALLOWED_NAMES in
scripts/check_english.sh instead of removing the diacritics from someone's name.
"""


def write_guard_output(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / f"{name}.out"
    path.write_text(text, encoding="utf-8")
    return path


def make_wheel(path: Path, members: list[str]) -> Path:
    """Build a wheel-shaped zip containing exactly ``members``."""
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, "# placeholder\n")
        archive.writestr("zpp_traffic_control-0.1.0.dist-info/METADATA", "Name: x\n")
    return path


def make_source_tree(root: Path, packages: list[str]) -> Path:
    """Build a source tree where each entry in ``packages`` is a package directory."""
    for package in packages:
        directory = root / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    return root


# --------------------------------------------------------------------------------------
# pytest-gate
# --------------------------------------------------------------------------------------
def test_pytest_gate_passes_when_both_parsers_agree_and_skips_are_under_the_ceiling(
    tmp_path: Path,
) -> None:
    """The happy path, on the exact tail this repo's pinned suite produced (M6)."""
    terminal, junit = write_run(
        tmp_path,
        "923 passed, 3 skipped, 16 warnings in 89.98s (0:01:29)",
        tests=926,
        skipped=3,
    )
    baseline = write_baseline(tmp_path, ceiling=62)

    result = run_gate(
        "pytest-gate",
        "--terminal", str(terminal),
        "--junit", str(junit),
        "--baseline", str(baseline),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
    assert "passed=923" in result.stdout
    assert "skipped=3" in result.stdout


def test_pytest_gate_fails_when_the_two_independent_parsers_disagree(tmp_path: Path) -> None:
    """The double-compute is the gate's own load-bearing check, so it must be able to fire.

    A single parser that silently mis-reads is exactly how a skip count stops being part
    of the result while the badge stays green.
    """
    terminal, junit = write_run(
        tmp_path,
        "923 passed, 3 skipped, 16 warnings in 89.98s (0:01:29)",
        tests=926,
        skipped=30,  # the JUnit report disagrees with the tail
    )
    baseline = write_baseline(tmp_path, ceiling=62)

    result = run_gate(
        "pytest-gate",
        "--terminal", str(terminal),
        "--junit", str(junit),
        "--baseline", str(baseline),
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL" in result.stdout
    assert "disagree" in result.stdout.lower()


def test_pytest_gate_fails_when_skips_exceed_the_declared_ceiling(tmp_path: Path) -> None:
    """Requirement 3 of BRIEF_23 section 3.1: exceeding the ceiling FAILS the job."""
    terminal, junit = write_run(
        tmp_path,
        "863 passed, 63 skipped, 15 warnings in 75.87s",
        tests=926,
        skipped=63,
    )
    baseline = write_baseline(tmp_path, ceiling=62)

    result = run_gate(
        "pytest-gate",
        "--terminal", str(terminal),
        "--junit", str(junit),
        "--baseline", str(baseline),
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL" in result.stdout
    assert "63" in result.stdout and "62" in result.stdout


def test_pytest_gate_accepts_skips_exactly_at_the_ceiling(tmp_path: Path) -> None:
    """The ceiling is inclusive; the boundary is stated in the file's docstring."""
    terminal, junit = write_run(
        tmp_path,
        "864 passed, 62 skipped, 15 warnings in 75.87s",
        tests=926,
        skipped=62,
    )
    baseline = write_baseline(tmp_path, ceiling=62)

    result = run_gate(
        "pytest-gate",
        "--terminal", str(terminal),
        "--junit", str(junit),
        "--baseline", str(baseline),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_pytest_gate_fails_on_a_failing_suite_even_below_the_ceiling(tmp_path: Path) -> None:
    """A gate that certifies a red suite because the skips looked fine is worse than none."""
    terminal, junit = write_run(
        tmp_path,
        "1 failed, 853 passed, 72 skipped, 3 warnings in 44.68s",
        tests=926,
        skipped=72,
        failures=1,
    )
    baseline = write_baseline(tmp_path, ceiling=100)

    result = run_gate(
        "pytest-gate",
        "--terminal", str(terminal),
        "--junit", str(junit),
        "--baseline", str(baseline),
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL" in result.stdout
    assert "failed=1" in result.stdout


def test_pytest_gate_reports_every_violation_and_not_only_the_first(tmp_path: Path) -> None:
    """A red suite must not hide that the run also breached its skip ceiling.

    Found on the gate's own first run against a real GitHub runner (32021100181): the suite
    was red *and* 10 skips over the declared ceiling, and the original ``elif`` chain
    printed only the failure -- which would have left the ceiling check looking untested on
    a runner while it had in fact fired.
    """
    terminal, junit = write_run(
        tmp_path,
        "1 failed, 891 passed, 72 skipped, 1 warning in 68.52s (0:01:08)",
        tests=964,
        skipped=72,
        failures=1,
    )
    baseline = write_baseline(tmp_path, ceiling=62)

    result = run_gate(
        "pytest-gate",
        "--terminal", str(terminal),
        "--junit", str(junit),
        "--baseline", str(baseline),
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "the suite is red" in result.stdout
    assert "72 tests skipped against a declared ceiling of 62" in result.stdout


def test_pytest_gate_refuses_an_unparseable_tail(tmp_path: Path) -> None:
    """A parser that returns zeros on garbage is the badge that cannot go red."""
    terminal = tmp_path / "pytest.txt"
    terminal.write_text("the runner died before pytest printed anything\n", encoding="utf-8")
    junit = tmp_path / "junit.xml"
    junit.write_text(junit_xml(926, 3), encoding="utf-8")
    baseline = write_baseline(tmp_path, ceiling=62)

    result = run_gate(
        "pytest-gate",
        "--terminal", str(terminal),
        "--junit", str(junit),
        "--baseline", str(baseline),
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL" in result.stdout
    assert "parse" in result.stdout.lower()


def test_pytest_gate_refuses_a_category_it_does_not_model(tmp_path: Path) -> None:
    """``xfailed`` counts as skipped in JUnit but not in the tail, so the gate refuses it.

    Refusing to certify what it cannot count keeps the double-compute honest instead of
    quietly mapping a category onto the wrong bucket.
    """
    terminal, junit = write_run(
        tmp_path,
        "900 passed, 3 skipped, 2 xfailed in 90.00s",
        tests=926,
        skipped=5,
    )
    baseline = write_baseline(tmp_path, ceiling=62)

    result = run_gate(
        "pytest-gate",
        "--terminal", str(terminal),
        "--junit", str(junit),
        "--baseline", str(baseline),
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL" in result.stdout
    assert "xfailed" in result.stdout


def test_pytest_gate_writes_counts_and_the_ceiling_expiry_into_the_job_summary(
    tmp_path: Path,
) -> None:
    """Requirement 1 of section 3.1, plus the ruling that the expiry must not live in a chat.

    The summary file is ``$GITHUB_STEP_SUMMARY`` on a runner, so this is the text a human
    reads without opening the log.
    """
    terminal, junit = write_run(
        tmp_path,
        "864 passed, 62 skipped, 15 warnings in 75.87s",
        tests=926,
        skipped=62,
    )
    baseline = write_baseline(tmp_path, ceiling=62)
    summary = tmp_path / "summary.md"

    result = run_gate(
        "pytest-gate",
        "--terminal", str(terminal),
        "--junit", str(junit),
        "--baseline", str(baseline),
        "--summary-file", str(summary),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    written = summary.read_text(encoding="utf-8")
    assert "864" in written and "62" in written
    assert "provisional" in written.lower()
    assert "merge of P5.1" in written


# --------------------------------------------------------------------------------------
# guard-gate
# --------------------------------------------------------------------------------------
def test_guard_gate_passes_when_the_findings_equal_the_recorded_baseline(tmp_path: Path) -> None:
    """DEFERRED 45's findings are known and may not be "fixed" to make CI green."""
    output = write_guard_output(tmp_path, "hygiene", HYGIENE_OUTPUT)
    baseline = write_baseline(tmp_path, ceiling=62)

    result = run_gate(
        "guard-gate",
        "--name", "hygiene",
        "--output", str(output),
        "--guard-exit", "1",
        "--baseline", str(baseline),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
    assert "3" in result.stdout


def test_guard_gate_fails_on_one_extra_finding_in_a_file_already_in_the_baseline(
    tmp_path: Path,
) -> None:
    """The regression case the whole baseline exists for."""
    extra = HYGIENE_OUTPUT + (
        f"tests/test_phase_control.py:91: {_TH006}\n" f"    with {_RAISES}(TypeError):\n"
    )
    output = write_guard_output(tmp_path, "hygiene", extra)
    baseline = write_baseline(tmp_path, ceiling=62)

    result = run_gate(
        "guard-gate",
        "--name", "hygiene",
        "--output", str(output),
        "--guard-exit", "1",
        "--baseline", str(baseline),
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL" in result.stdout
    assert "tests/test_phase_control.py" in result.stdout
    assert "TH006" in result.stdout


def test_guard_gate_fails_on_a_finding_in_a_file_absent_from_the_baseline(
    tmp_path: Path,
) -> None:
    """A new file with a violation must not pass because the totals happen to match."""
    swapped = HYGIENE_OUTPUT.replace(
        "tests/test_base_traffic_env.py:77", "tests/test_brand_new_file.py:12"
    )
    output = write_guard_output(tmp_path, "hygiene", swapped)
    baseline = write_baseline(tmp_path, ceiling=62)

    result = run_gate(
        "guard-gate",
        "--name", "hygiene",
        "--output", str(output),
        "--guard-exit", "1",
        "--baseline", str(baseline),
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "tests/test_brand_new_file.py" in result.stdout


def test_guard_gate_passes_with_a_note_when_a_finding_was_genuinely_fixed(
    tmp_path: Path,
) -> None:
    """Only a regression fails; an improvement passes and says the baseline can tighten."""
    fewer = "\n".join(HYGIENE_OUTPUT.splitlines()[2:])
    output = write_guard_output(tmp_path, "hygiene", fewer)
    baseline = write_baseline(tmp_path, ceiling=62)

    result = run_gate(
        "guard-gate",
        "--name", "hygiene",
        "--output", str(output),
        "--guard-exit", "1",
        "--baseline", str(baseline),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "NOTE" in result.stdout


def test_guard_gate_fails_when_the_guard_reported_nothing_at_all(tmp_path: Path) -> None:
    """Empty output plus exit 0 is indistinguishable from a broken invocation, so it fails.

    Without this, pointing the workflow at a mistyped path would turn every guard green
    and nothing would say so -- the decoration failure in its purest form.
    """
    output = write_guard_output(tmp_path, "hygiene", "")
    baseline = write_baseline(tmp_path, ceiling=62)

    result = run_gate(
        "guard-gate",
        "--name", "hygiene",
        "--output", str(output),
        "--guard-exit", "0",
        "--baseline", str(baseline),
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL" in result.stdout


def test_guard_gate_fails_when_the_guard_itself_crashed(tmp_path: Path) -> None:
    """Exit 127 is "script not found"; only 0 and 1 are verdicts these guards produce."""
    output = write_guard_output(tmp_path, "hygiene", "")
    baseline = write_baseline(tmp_path, ceiling=62)

    result = run_gate(
        "guard-gate",
        "--name", "hygiene",
        "--output", str(output),
        "--guard-exit", "127",
        "--baseline", str(baseline),
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "127" in result.stdout


def test_guard_gate_parses_the_english_guard_output_shape(tmp_path: Path) -> None:
    """``check_english.sh`` prints bare ``file:line`` pairs, not ``file:line: [RULE]``."""
    output = write_guard_output(tmp_path, "english", ENGLISH_OUTPUT)
    baseline = write_baseline(tmp_path, ceiling=62)

    result = run_gate(
        "guard-gate",
        "--name", "english",
        "--output", str(output),
        "--guard-exit", "1",
        "--baseline", str(baseline),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_guard_gate_ignores_the_guards_prose_and_counts_only_findings(tmp_path: Path) -> None:
    """The advice footer contains paths; counting them would inflate every run."""
    output = write_guard_output(tmp_path, "english", ENGLISH_OUTPUT)
    baseline = write_baseline(
        tmp_path,
        ceiling=62,
        guards={
            "english": {
                "script": "scripts/check_english.sh",
                "total": 1,
                "by_file": {"README.md": {"PL": 1}},
            }
        },
    )

    result = run_gate(
        "guard-gate",
        "--name", "english",
        "--output", str(output),
        "--guard-exit", "1",
        "--baseline", str(baseline),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "findings=1" in result.stdout


# --------------------------------------------------------------------------------------
# wheel-gate
# --------------------------------------------------------------------------------------
def test_wheel_gate_fails_on_a_wheel_that_ships_the_platform_but_not_the_research_code(
    tmp_path: Path,
) -> None:
    """This is P0.10's defect, reproduced synthetically: no ``offline`` in the wheel."""
    source = make_source_tree(tmp_path / "src", ["agent", "envs", "offline", "offline/policies"])
    wheel = make_wheel(
        tmp_path / "no_offline.whl",
        ["agent/__init__.py", "envs/__init__.py"],
    )

    result = run_gate("wheel-gate", "--wheel", str(wheel), "--repo-root", str(source))

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL" in result.stdout
    assert "offline" in result.stdout


def test_wheel_gate_fails_when_only_a_subpackage_is_missing(tmp_path: Path) -> None:
    """``offline`` present but ``offline.policies`` absent still breaks every collector."""
    source = make_source_tree(tmp_path / "src", ["agent", "offline", "offline/policies"])
    wheel = make_wheel(
        tmp_path / "partial.whl",
        ["agent/__init__.py", "offline/__init__.py"],
    )

    result = run_gate("wheel-gate", "--wheel", str(wheel), "--repo-root", str(source))

    assert result.returncode == 1, result.stdout + result.stderr
    assert "offline/policies" in result.stdout


def test_wheel_gate_passes_when_every_declared_package_is_in_the_wheel(tmp_path: Path) -> None:
    source = make_source_tree(tmp_path / "src", ["agent", "offline", "offline/policies"])
    wheel = make_wheel(
        tmp_path / "complete.whl",
        ["agent/__init__.py", "offline/__init__.py", "offline/policies/__init__.py"],
    )

    result = run_gate("wheel-gate", "--wheel", str(wheel), "--repo-root", str(source))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_wheel_gate_does_not_require_the_test_package(tmp_path: Path) -> None:
    """``tests`` carries an ``__init__.py`` but is deliberately not distributed."""
    source = make_source_tree(tmp_path / "src", ["agent", "tests"])
    wheel = make_wheel(tmp_path / "no_tests.whl", ["agent/__init__.py"])

    result = run_gate("wheel-gate", "--wheel", str(wheel), "--repo-root", str(source))

    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------------------
# the committed baseline itself
# --------------------------------------------------------------------------------------
def test_committed_baseline_records_the_guard_findings_measured_on_this_tree() -> None:
    """The 16 and the 4, recorded independently of the JSON so a silent widening fails.

    The per-file split is ``DEFERRED`` 45's own enumeration (8 + 5 + 2 + 1 = 16), counted
    four independent times by the coordinator, and re-measured for P0.10 from the full
    output of a no-argument run.

    ⚠️ **The English total moved 5 -> 4 when ``Mikolaj`` joined ``ALLOWED_NAMES``** (the patch
    applied on ``main``; the baseline records 4 and the tree measures 4).  It is pinned here, in a
    second file, so that widening it is a visible edit in a diff rather than a quiet number change
    -- which is exactly why this literal must be moved deliberately whenever the guard's allow-list
    changes.  ``DEFERRED`` 54.
    """
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    hygiene = baseline["guards"]["hygiene"]
    english = baseline["guards"]["english"]

    assert hygiene["by_file"] == {
        "tests/test_cityflow_env_phase_constraints.py": {"TH006": 8},
        "tests/test_sumo_env_phase_constraints.py": {"TH006": 5},
        "tests/test_phase_control.py": {"TH006": 2},
        "tests/test_base_traffic_env.py": {"TH006": 1},
    }
    assert hygiene["total"] == 16
    assert english["total"] == 4

    # Recomputed from the map rather than trusted: the totals must be the sum of the parts.
    for guard in (hygiene, english):
        recomputed = sum(
            count for rules in guard["by_file"].values() for count in rules.values()
        )
        assert recomputed == guard["total"]


def test_committed_ceiling_declares_that_it_is_provisional_and_names_its_expiry() -> None:
    """A ceiling whose expiry lives only in a chat is a ceiling that silently breaches.

    The baseline must carry the re-measure mandate in the file a runner actually reads, so every
    field of it is asserted non-empty below.

    ⚠️ **The event NAME is deliberately NOT pinned, and the removal was ruled rather than drifted
    into** (``DEFERRED`` 54).  This file's purpose is that widening a LOOSENABLE quantity is a
    visible edit in a diff; the two numbers -- the English total and the skip ceiling -- are those
    quantities, and they already force an edit here at every expiry move, so the speed bump
    survives.  **The event is a POINTER and cannot be widened.**  Pinning it fires at every
    CORRECT retarget -- when P5.1's merge landed and the mandate properly moved to P5.2, and again
    when P5.2's does -- which is the class this repo refuses: a check that condemns correct
    artifacts teaches the reader to ignore it.  The completeness loop above is the load-bearing
    half and it stays.
    """
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    block = baseline["pytest"]

    assert block["provisional"] is True
    assert isinstance(block["skip_ceiling"], int)
    expiry = block["re_measure_required_at"]
    # The full contracted key set, not a subset: ``reason`` was dropped from the baseline in
    # 69680fa and only this loop noticed, because ci_gate.py:308 reads it through a
    # ``.get(..., "")`` default and prints an empty clause rather than failing.  ``predicted_delta``
    # and ``what_to_do`` are pinned for the same reason -- they carry the protocol that replaced a
    # prediction, and a silent deletion would leave the next person without it.
    for key in ("event", "reason", "predicted_delta", "what_to_do", "mandated_by"):
        assert key in expiry, (
            f"{key} is missing: ci_gate.py:39 declares it part of this file's format and "
            f"ci_gate.py:308 reads it behind a .get default, so the gate cannot notice its loss"
        )
        assert expiry[key].strip(), f"{key} must say something a reader can act on"


def test_committed_baseline_records_why_the_ceiling_has_the_value_it_has() -> None:
    """Ruled 2026-08-17: the ceiling is what it is BECAUSE CityFlow is not built in CI.

    A future reader must be able to see that the number is a consequence of a choice, and
    what the number would be under the other choice.

    The chain is **104 -> 98 -> 72 -> 62**, newest first.  Only the root was ever WRONG: 62 came
    from a simulated runner (see section "harness errors" of ``docs/returns/P0.10.md``).  72 and 98
    EXPIRED EXACTLY AS THEIR OWN ENTRIES SAID THEY WOULD, on the merges of P5.1 and P5.2.  The chain
    is pinned here, in a second file, precisely so that widening it is a visible edit in a diff
    rather than a quiet number change.

    ⚠️ **Moving the ceiling is now ONE literal in this file: ``CEILING_CHAIN``.**  It used to be
    four, one per nesting depth, under two different key names, and that shape guaranteed its own
    recurrence -- every move added a
    level, so it needed a NEW assertion each time, and **at the 98 -> 104 move the root 62 fell out
    of the pinned set entirely**, which is the one thing the previous docstring said must never
    happen.  ``DEFERRED`` 54, third instance.  The walk below reaches the root at whatever depth it
    lands.  *(If a skip CATEGORY moves rather than the total, ``skip_breakdown`` in the baseline
    moves too and the arithmetic below fails until it does -- that is intended.)*
    """
    #: The declared ceiling and every value it supersedes, newest first.  ONE literal, walked
    #: rather than addressed by depth -- see the docstring for why the depth-addressed form let
    #: the root drop out of the record.
    CEILING_CHAIN = (104, 98, 72, 62)

    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    block = baseline["pytest"]

    assert block["measured"]["measured_on_a_github_runner"] is True
    # The headline and its own evidence are two fields that have always been edited by hand and
    # independently; nothing asserted they agree until now.
    assert block["skip_ceiling"] == block["measured"]["value"], (
        "skip_ceiling and measured.value are the same number recorded twice and they disagree"
    )

    chain = [int(block["skip_ceiling"])]
    why: list[str] = []
    link = block["measured"]
    while "superseded" in link:
        link = link["superseded"]
        chain.append(int(link["value"]))
        why.append(str(link["why_it_was_wrong"]))

    assert tuple(chain) == CEILING_CHAIN, (
        f"the ceiling chain on disk is {tuple(chain)} against the pinned {CEILING_CHAIN}"
    )
    assert chain[-1] == 62, (
        "the ONLY link that was ever actually wrong -- the simulated 62 -- must stay in the "
        "record at whatever depth the next move pushes it to"
    )
    assert len(why) == len(CEILING_CHAIN) - 1
    assert all(text.strip() for text in why), "the reason a number moved is part of the number"
    # One relation, one key name, at every depth.  The baseline used to say "superseded" at the
    # first level and "its_own_superseded" below it, and a generic walk stops dead at the second
    # name -- which is why the chain could previously only be pinned one depth at a time, and
    # which truncated the first draft of this very walk to (104, 98).
    #
    # Asserted over KEYS, recursively, and not over ``json.dumps(block)``: a substring test over
    # the dump cannot tell a key from PROSE ABOUT a key, and the first draft of this assertion was
    # tripped by the baseline note that documents the rename.  That is the same weak shape as the
    # ``"40" in provenance`` test removed below, written one assertion after removing it.
    def every_key(node: object) -> Iterator[str]:
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from every_key(value)
        elif isinstance(node, list):
            for item in node:
                yield from every_key(item)

    assert "its_own_superseded" not in set(every_key(block)), (
        "a second name for the superseded relation is back as a KEY; the walk above will "
        "silently truncate at it and the root will drop out of the record again"
    )

    # The alternative -- what the ceiling would be if CityFlow were built -- is checked as
    # ARITHMETIC over a declared breakdown.  What stood here was ``assert "40" in json.dumps(block)``,
    # and that assertion EXPIRED rather than having always been empty.  Measured 2026-08-24 on both
    # eras: with the alternative deleted the expression is False on the 98-era baseline (4866d52)
    # and True on this one, because the 104 measurement itself introduced the string "1240 passed"
    # into ``measured.result``.  A substring assertion's discriminating power is a function of data
    # it does not name, so it can stop discriminating with nobody editing it -- and that is how the
    # sentence it was guarding stayed at the 72-era numbers through two ceiling moves.
    breakdown = block["skip_breakdown"]
    assert sum(breakdown.values()) == block["skip_ceiling"], (
        f"the skip breakdown sums to {sum(breakdown.values())} against a ceiling of "
        f"{block['skip_ceiling']}; a total and an enumeration that disagree is DEFERRED 53's class"
    )
    assert (
        block["ceiling_if_cityflow_were_built"]
        == block["skip_ceiling"] - breakdown["cityflow"]
    ), "the recorded alternative does not equal ceiling minus the CityFlow-gated skips"
    # Every category the ceiling is MADE of must carry a recorded reason.  This ties the prose to
    # the arithmetic: an unexplained category, or a silently emptied explanation list, now fails.
    # It is also what would have caught the campaign-output category, which was counted in
    # ``measured.breakdown`` from the 104 move and never added to this list until 2026-08-24.
    # ``assert "CityFlow" in json.dumps(block)`` stood here and SURVIVED deleting the entire
    # CityFlow entry, because "CityFlow not built" also appears in ``measured.condition`` -- the
    # third dump-grep in this one test to be satisfied by a string it was not asking about.
    consequences = block["ceiling_is_a_consequence_of"]
    assert len(consequences) == len(breakdown), (
        f"{len(breakdown)} skip categories against {len(consequences)} recorded reasons; "
        "every category the ceiling is made of must say why it is there"
    )
    assert any("CityFlow" in entry for entry in consequences), (
        "the CityFlow choice is the ceiling's largest single cause and must stay stated in prose"
    )


# --------------------------------------------------------------------------------------
# the workflow file
# --------------------------------------------------------------------------------------
def test_workflow_pins_threads_and_invokes_every_gate_and_guard() -> None:
    """Textual, and weak on purpose: it catches deletion, never semantics.

    There is no YAML parser in this project's dependency set, so this asserts on the file's
    text.  It cannot tell whether the workflow *works* -- only a runner can, and the run is
    reported in ``docs/returns/P0.10.md``.
    """
    text = WORKFLOW_FILE.read_text(encoding="utf-8")

    # DEFERRED 41: the suite can deadlock unpinned.
    assert "OMP_NUM_THREADS" in text
    assert "MKL_NUM_THREADS" in text

    # The skip reasons are part of the result, so -rs must be there, and so must the
    # machine-readable report the gate cross-checks the tail against.
    assert "-rs" in text
    assert "--junit-xml" in text

    for guard in (
        "scripts/claude_guard.sh",
        "scripts/check_english.sh",
        "scripts/check_test_hygiene.sh",
    ):
        assert guard in text, f"{guard} is not invoked by the workflow"

    for subcommand in ("pytest-gate", "guard-gate", "wheel-gate"):
        assert subcommand in text, f"the workflow never calls {subcommand}"

    assert "ci_baseline.json" in text


def test_workflow_does_not_silently_swallow_the_suite_result() -> None:
    """``continue-on-error`` on the suite step would make every later gate advisory."""
    text = WORKFLOW_FILE.read_text(encoding="utf-8")
    assert "continue-on-error" not in text


@pytest.mark.parametrize("path", [GATE, BASELINE_FILE, WORKFLOW_FILE])
def test_ci_files_exist(path: Path) -> None:
    """A missing file must fail as itself, not as a confusing parse error elsewhere."""
    assert path.is_file(), f"{path} is missing"


# --------------------------------------------------------------------------------------
# libc-matrix -- review item R2: the second C library must be declared, not incidental
# --------------------------------------------------------------------------------------
def write_leg(directory: Path, image: str, libc: tuple[str, str]) -> Path:
    """Write one matrix leg's recorded C library, in the shape the workflow emits."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"libc-{image}.json"
    path.write_text(json.dumps({"image": image, "libc": list(libc)}), encoding="utf-8")
    return path


def test_libc_matrix_passes_when_the_legs_really_differ(tmp_path: Path) -> None:
    """The point of the matrix: two legs, two different C libraries."""
    legs = tmp_path / "legs"
    write_leg(legs, "ubuntu-22.04", ("glibc", "2.35"))
    write_leg(legs, "ubuntu-24.04", ("glibc", "2.39"))

    result = run_gate("libc-matrix", "--inputs", str(legs))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
    assert "2.35" in result.stdout and "2.39" in result.stdout


def test_libc_matrix_fails_when_both_legs_have_the_same_c_library(tmp_path: Path) -> None:
    """THE POINT OF R2. The day GitHub bumps both images to one glibc, this must go red.

    Before this gate existed, the dev-2.43-versus-runner-2.39 coverage was an accident of
    which image `ubuntu-latest` happened to resolve to, and its loss would have been silent.
    """
    legs = tmp_path / "legs"
    write_leg(legs, "ubuntu-24.04", ("glibc", "2.39"))
    write_leg(legs, "ubuntu-24.04-arm", ("glibc", "2.39"))

    result = run_gate("libc-matrix", "--inputs", str(legs))

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL" in result.stdout
    assert "2.39" in result.stdout


def test_libc_matrix_fails_when_a_leg_is_missing(tmp_path: Path) -> None:
    """One leg cannot demonstrate cross-library agreement, and a silent pass would claim it."""
    legs = tmp_path / "legs"
    write_leg(legs, "ubuntu-24.04", ("glibc", "2.39"))

    result = run_gate("libc-matrix", "--inputs", str(legs))

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL" in result.stdout


def test_libc_matrix_fails_when_a_leg_reported_no_version(tmp_path: Path) -> None:
    """``platform.libc_ver()`` returns ``('', '')`` on a musl or non-glibc image."""
    legs = tmp_path / "legs"
    write_leg(legs, "ubuntu-24.04", ("glibc", "2.39"))
    write_leg(legs, "alpine", ("", ""))

    result = run_gate("libc-matrix", "--inputs", str(legs))

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL" in result.stdout


def workflow_configuration() -> str:
    """The workflow with comments stripped -- the configuration, not the prose about it.

    Same distinction the ``dt_gate`` libm guard draws with an AST: the file *documents* why
    ``ubuntu-latest`` was removed, and a naive substring scan would flag that sentence. A guard
    with false positives is a guard that gets deleted.
    """
    lines = []
    for line in WORKFLOW_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.split(" #", 1)[0] if " #" in line else line
        if stripped.lstrip().startswith("#"):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def test_the_workflow_pins_its_images_and_never_uses_latest() -> None:
    """R2: `ubuntu-latest` makes the second C library an accident that can vanish silently."""
    text = workflow_configuration()

    assert "ubuntu-latest" not in text, (
        "a floating image makes the two-libc coverage incidental: the day GitHub bumps it, "
        "the matrix legs can collapse onto one C library and nothing fails"
    )
    assert "ubuntu-22.04" in text and "ubuntu-24.04" in text
    assert "libc-matrix" in text, "the workflow never checks that the legs actually differ"
    assert "libc_ver" in text, "no leg records its C library"
