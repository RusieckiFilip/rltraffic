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
    """The 16 and the 5, recorded independently of the JSON so a silent widening fails.

    The per-file split is ``DEFERRED`` 45's own enumeration (8 + 5 + 2 + 1 = 16), counted
    four independent times by the coordinator, and re-measured for P0.10 from the full
    output of a no-argument run.
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
    assert english["total"] == 5

    # Recomputed from the map rather than trusted: the totals must be the sum of the parts.
    for guard in (hygiene, english):
        recomputed = sum(
            count for rules in guard["by_file"].values() for count in rules.values()
        )
        assert recomputed == guard["total"]


def test_committed_ceiling_declares_that_it_is_provisional_and_names_its_expiry() -> None:
    """A ceiling whose expiry lives only in a chat is a ceiling that silently breaches.

    P5.1 is in flight and its merge adds roughly 24-25 corpus-gated skips, so the baseline
    must carry the re-measure mandate in the file a runner actually reads.
    """
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    block = baseline["pytest"]

    assert block["provisional"] is True
    assert isinstance(block["skip_ceiling"], int)
    expiry = block["re_measure_required_at"]
    for key in ("event", "reason", "mandated_by"):
        assert expiry[key].strip(), f"{key} must say something a reader can act on"
    assert "P5.1" in expiry["event"]


def test_committed_baseline_records_why_the_ceiling_has_the_value_it_has() -> None:
    """Ruled 2026-08-17: the ceiling is 62 BECAUSE CityFlow is not built in CI.

    A future reader must be able to see that the number is a consequence of a choice, and
    what the number would be under the other choice.
    """
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    block = baseline["pytest"]

    assert block["skip_ceiling"] == 62
    provenance = json.dumps(block)
    assert "CityFlow" in provenance
    assert "31" in provenance, "the alternative (CityFlow built) must be recorded too"


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
