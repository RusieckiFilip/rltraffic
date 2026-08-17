#!/usr/bin/env python3
"""CI gates for the rltraffic repository -- gate format version ``1``.

Why this is a Python file and not shell inside the workflow YAML: a gate that has never
been observed failing is decoration (``BRIEF_23`` section 3).  Logic in a file can be run
locally, fed synthetic input and **shown going red**; logic in YAML can only be shown
going green on a runner.  Every command in ``.github/workflows/ci.yml`` is one call into
this module, and ``tests/test_ci_gate.py`` drives this same file as a subprocess.

Three subcommands, each reading artifacts produced by the step before it:

``pytest-gate``
    Reads the suite's terminal tail **and** its JUnit XML, parses the counts out of both
    **independently**, and refuses to certify the run if the two disagree.  Then: no
    failures, no errors, and ``skipped <= skip_ceiling``.  The counts are printed and
    written to the job summary, because on a runner with no corpus and no simulator a
    green tick otherwise means *"what could run, ran"* rather than *"the suite passed"*.

``guard-gate``
    Runs nothing itself; it compares a guard's captured output against the **known
    baseline** in ``ci_baseline.json``.  ``scripts/check_test_hygiene.sh`` and
    ``scripts/check_english.sh`` both exit 1 on pre-existing findings (``DEFERRED`` 45 and
    P0.9), and neither may be "fixed" to make CI green.  Only a **regression against the
    baseline** fails the job.

``wheel-gate``
    Asserts a built wheel contains every package the repository declares.  This is the
    regression cover for the defect P0.10 exists to fix: ``offline*`` was absent from
    ``[tool.setuptools.packages.find]``, so ``pip install .`` shipped the platform and not
    the research code.

Baseline file format (``.github/ci/ci_baseline.json``), schema version ``1``::

    {
      "schema_version": 1,
      "pytest": {
        "skip_ceiling": int,          # inclusive: skipped == ceiling passes
        "provisional": bool,
        "re_measure_required_at": {"event": str, "reason": str, "mandated_by": str},
        ...
      },
      "guards": {
        "<name>": {
          "script": str,
          "total": int,
          "by_file": {"<path>": {"<rule>": int}}   # NOT file:line -- line numbers move
        }
      }
    }

**Baseline convention.** A finding is identified by ``(file, rule)`` and counted.  Line
numbers are deliberately not part of the key: any edit above a finding would move it and
turn an unrelated commit red.  A bare total is deliberately not the key either: it would
let a fix in one file pay for a new violation in another.

**Exit codes.** ``0`` the gate passed · ``1`` the gate failed · ``2`` usage error.
Every verdict is printed on stdout with a leading ``OK`` / ``FAIL`` / ``NOTE`` token so a
caller can assert on the reason rather than only on the code -- an uncaught exception also
exits 1, so a test that checked the code alone would pass against a stub.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ElementTree
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# ``tests`` carries an ``__init__.py`` -- that is how pytest ends up prepending the repo
# root to ``sys.path`` -- but is deliberately not distributed.  Kept in step with
# ``tests/test_packaging.py::NOT_DISTRIBUTED``.
NOT_DISTRIBUTED = frozenset({"tests"})

# Categories this gate does not model.  JUnit folds ``xfailed`` into ``skipped`` while the
# terminal tail reports it separately, so accepting one would silently break the
# cross-check that is the whole point of reading both.
UNMODELLED_CATEGORIES = frozenset({"xfailed", "xpassed", "deselected", "rerun", "reruns"})

_COUNT_TOKEN = re.compile(
    r"(\d+)\s+(passed|failed|skipped|error|errors|xfailed|xpassed|deselected|rerun|reruns"
    r"|warning|warnings)\b"
)
_ELAPSED = re.compile(r"\bin \d+(\.\d+)?s")

# ``scripts/check_test_hygiene.sh`` prints ``file:line: [RULE] message`` and then the
# offending source line indented by four spaces; the indent is what keeps the continuation
# out of this pattern.
_HYGIENE_FINDING = re.compile(r"^(?P<file>[^\s:]+):(?P<line>\d+):\s+\[(?P<rule>TH\d+)\]")
# ``scripts/check_english.sh`` prints bare ``file:line`` pairs between a header and an
# advice footer; the anchors are what keep the prose out.
_ENGLISH_FINDING = re.compile(r"^(?P<file>[^\s:]+):(?P<line>\d+)\s*$")

_GUARD_PATTERNS = {"hygiene": _HYGIENE_FINDING, "english": _ENGLISH_FINDING}
# check_english.sh reports a location, not a rule id; ``PL`` names what it detects.
_GUARD_DEFAULT_RULE = {"hygiene": None, "english": "PL"}


class GateError(Exception):
    """A gate could not reach a verdict (unparseable input, missing artifact)."""


@dataclass(frozen=True)
class Counts:
    """Outcome counts of one pytest run, as reported by one parser."""

    passed: int
    skipped: int
    failed: int
    errors: int

    def describe(self) -> str:
        return (
            f"passed={self.passed} skipped={self.skipped} "
            f"failed={self.failed} errors={self.errors}"
        )


def parse_terminal_tail(text: str) -> Counts:
    """Parse pytest's terminal summary line (``-q`` form or decorated form).

    Raises ``GateError`` if no summary line is present, or if the run reports a category
    this gate does not model -- refusing to certify what it cannot count is the point.
    """
    for raw in reversed(text.splitlines()):
        line = raw.strip().strip("=").strip()
        if not _ELAPSED.search(line):
            continue
        tokens = _COUNT_TOKEN.findall(line)
        if not tokens:
            continue

        counts = {"passed": 0, "skipped": 0, "failed": 0, "errors": 0}
        for number, category in tokens:
            if category in UNMODELLED_CATEGORIES:
                raise GateError(
                    f"the terminal summary reports {category!r}, a category this gate does "
                    "not model; extend ci_gate.py rather than letting it be miscounted"
                )
            if category in {"warning", "warnings"}:
                continue
            key = "errors" if category.startswith("error") else category
            counts[key] += int(number)
        return Counts(**counts)

    raise GateError(
        "could not parse a pytest summary line out of the terminal output; the run may "
        "have died before pytest reported anything"
    )


def parse_junit(xml_text: str) -> Counts:
    """Parse the same counts out of a JUnit XML report, by a route sharing no code."""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:  # pragma: no cover - exercised via GateError path
        raise GateError(f"could not parse the JUnit report: {exc}") from exc

    suites = list(root.iter("testsuite"))
    if not suites:
        raise GateError("the JUnit report contains no <testsuite> element")

    total = skipped = failures = errors = 0
    for suite in suites:
        total += int(suite.get("tests", 0))
        skipped += int(suite.get("skipped", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))

    passed = total - skipped - failures - errors
    if passed < 0:
        raise GateError(f"the JUnit report is inconsistent: {total} tests but {passed} passed")
    return Counts(passed=passed, skipped=skipped, failed=failures, errors=errors)


def parse_guard_findings(name: str, text: str) -> dict[str, dict[str, int]]:
    """Parse a guard's captured output into ``{file: {rule: count}}``."""
    pattern = _GUARD_PATTERNS.get(name)
    if pattern is None:
        raise GateError(f"unknown guard {name!r}; known guards: {sorted(_GUARD_PATTERNS)}")

    findings: dict[str, dict[str, int]] = {}
    for line in text.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        rule = match.groupdict().get("rule") or _GUARD_DEFAULT_RULE[name]
        assert rule is not None  # every pattern yields a rule or has a default
        findings.setdefault(match.group("file"), {}).setdefault(rule, 0)
        findings[match.group("file")][rule] += 1
    return findings


def compare_to_baseline(
    observed: dict[str, dict[str, int]],
    baseline: dict[str, dict[str, int]],
) -> list[str]:
    """Return one human-readable line per regression; empty means no regression."""
    regressions: list[str] = []
    for path in sorted(observed):
        for rule in sorted(observed[path]):
            seen = observed[path][rule]
            known = baseline.get(path, {}).get(rule, 0)
            if seen > known:
                regressions.append(f"{path} [{rule}]: baseline {known}, now {seen}")
    return regressions


def _total(findings: dict[str, dict[str, int]]) -> int:
    return sum(count for rules in findings.values() for count in rules.values())


def wheel_package_files(wheel_path: Path) -> set[str]:
    """Return the ``__init__.py``-bearing package paths a wheel actually ships."""
    with zipfile.ZipFile(wheel_path) as archive:
        return {
            name
            for name in archive.namelist()
            if name.endswith("__init__.py") and ".dist-info/" not in name
        }


def declared_package_files(repo_root: Path) -> set[str]:
    """Return the package paths the repository declares, from the filesystem.

    Derived from the tree rather than from the include list on purpose: the include list
    is the thing under test, so trusting it would make the gate agree with the defect.
    """
    declared: set[str] = set()
    for entry in sorted(repo_root.iterdir()):
        if not entry.is_dir() or entry.name in NOT_DISTRIBUTED:
            continue
        if not (entry / "__init__.py").is_file():
            continue
        for init in sorted(entry.rglob("__init__.py")):
            if "__pycache__" in init.parts:
                continue
            declared.add(init.relative_to(repo_root).as_posix())
    return declared


def _emit(lines: list[str], summary_file: str | None) -> None:
    """Print every line, and append it to the job summary when one was requested."""
    for line in lines:
        print(line)
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")


def _load_baseline(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pytest_gate(args: argparse.Namespace) -> int:
    baseline = _load_baseline(args.baseline)["pytest"]
    ceiling = int(baseline["skip_ceiling"])

    try:
        terminal = parse_terminal_tail(Path(args.terminal).read_text(encoding="utf-8"))
        junit = parse_junit(Path(args.junit).read_text(encoding="utf-8"))
    except GateError as exc:
        _emit([f"FAIL pytest-gate: {exc}"], args.summary_file)
        return 1

    lines = ["## pytest gate", f"counts: {terminal.describe()}"]
    verdict = 0

    if terminal != junit:
        # A disagreement short-circuits: if the two counts differ, every check below is
        # being applied to a number that may not be the run's.
        lines.append(
            "FAIL pytest-gate: the two independent parsers disagree -- "
            f"terminal says [{terminal.describe()}], JUnit says [{junit.describe()}]"
        )
        verdict = 1
    else:
        # Every violation is reported, not just the first. Found on this gate's own first
        # run against a real runner (32021100181): the suite was red AND over its ceiling,
        # and an elif chain showed only the failure, so the ceiling check looked untested
        # when it had in fact fired.
        problems: list[str] = []
        if terminal.failed or terminal.errors:
            problems.append(f"FAIL pytest-gate: the suite is red ({terminal.describe()})")
        if terminal.skipped > ceiling:
            problems.append(
                f"FAIL pytest-gate: {terminal.skipped} tests skipped against a declared "
                f"ceiling of {ceiling}. The skip count is part of the result: either the "
                "environment lost something it used to have (corpus paths, an engine), or "
                "the ceiling is stale and must be re-measured and committed with the reason."
            )
        if problems:
            lines.extend(problems)
            verdict = 1
        else:
            lines.append(
                f"OK pytest-gate: {terminal.describe()}, skip ceiling {ceiling} "
                f"({ceiling - terminal.skipped} to spare)"
            )

    if baseline.get("provisional"):
        expiry = baseline.get("re_measure_required_at", {})
        lines.append(
            f"NOTE the skip ceiling is provisional: re-measure at {expiry.get('event', '?')} "
            f"-- {expiry.get('reason', '')} ({expiry.get('mandated_by', '')})"
        )
    for reason in baseline.get("ceiling_is_a_consequence_of", []):
        lines.append(f"NOTE {reason}")

    skipped_block = [
        line
        for line in Path(args.terminal).read_text(encoding="utf-8").splitlines()
        if line.startswith("SKIPPED")
    ]
    if skipped_block:
        lines.append("")
        lines.append("<details><summary>skip reasons (pytest -rs)</summary>")
        lines.append("")
        lines.extend(f"    {line}" for line in skipped_block)
        lines.append("")
        lines.append("</details>")

    _emit(lines, args.summary_file)
    return verdict


def _guard_gate(args: argparse.Namespace) -> int:
    guards = _load_baseline(args.baseline)["guards"]
    if args.name not in guards:
        _emit([f"FAIL guard-gate: no baseline recorded for guard {args.name!r}"], args.summary_file)
        return 1
    baseline = guards[args.name]

    if args.guard_exit not in (0, 1):
        _emit(
            [
                f"FAIL guard-gate[{args.name}]: the guard exited {args.guard_exit}; these "
                "guards return 0 (clean) or 1 (findings), so anything else means the "
                "script crashed or was not found"
            ],
            args.summary_file,
        )
        return 1

    try:
        observed = parse_guard_findings(
            args.name, Path(args.output).read_text(encoding="utf-8")
        )
    except GateError as exc:
        _emit([f"FAIL guard-gate[{args.name}]: {exc}"], args.summary_file)
        return 1

    seen = _total(observed)
    known = int(baseline["total"])
    lines = [f"## guard gate: {args.name}", f"findings={seen} baseline={known}"]

    if seen == 0 and known > 0:
        lines.append(
            f"FAIL guard-gate[{args.name}]: the guard reported no findings at all while "
            f"{known} are on record. Either they were genuinely fixed -- then tighten the "
            "baseline in this same commit -- or the invocation is broken and the gate was "
            "about to certify nothing."
        )
        _emit(lines, args.summary_file)
        return 1

    regressions = compare_to_baseline(observed, baseline["by_file"])
    if regressions:
        lines.append(f"FAIL guard-gate[{args.name}]: {len(regressions)} regression(s):")
        lines.extend(f"  - {entry}" for entry in regressions)
        lines.append(
            "The known findings are recorded, not tolerated silently; a NEW one must be "
            "fixed in the change that introduced it. Do not edit the guard."
        )
        _emit(lines, args.summary_file)
        return 1

    lines.append(f"OK guard-gate[{args.name}]: findings={seen}, no regression against baseline")
    if seen < known:
        lines.append(
            f"NOTE {known - seen} recorded finding(s) are gone; the baseline can be "
            "tightened to lock the improvement in."
        )
    _emit(lines, args.summary_file)
    return 0


def _wheel_gate(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    declared = declared_package_files(repo_root)
    shipped = wheel_package_files(Path(args.wheel))

    missing = sorted(declared - shipped)
    lines = [
        "## wheel gate",
        f"packages declared={len(declared)} shipped={len(shipped)}",
    ]
    if missing:
        lines.append(
            f"FAIL wheel-gate: {len(missing)} package(s) exist in the tree but not in the "
            "wheel, so a clean install would ImportError on them:"
        )
        lines.extend(f"  - {entry}" for entry in missing)
        _emit(lines, args.summary_file)
        return 1

    lines.append(f"OK wheel-gate: every declared package is in {Path(args.wheel).name}")
    _emit(lines, args.summary_file)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary-file", default=None, help="append the verdict here too")
    subparsers = parser.add_subparsers(dest="command", required=True)

    suite = subparsers.add_parser("pytest-gate", help="counts, cross-check and skip ceiling")
    suite.add_argument("--terminal", required=True)
    suite.add_argument("--junit", required=True)
    suite.add_argument("--baseline", required=True)
    suite.add_argument("--summary-file", default=None)
    suite.set_defaults(handler=_pytest_gate)

    guard = subparsers.add_parser("guard-gate", help="guard findings against the baseline")
    guard.add_argument("--name", required=True, choices=sorted(_GUARD_PATTERNS))
    guard.add_argument("--output", required=True)
    guard.add_argument("--guard-exit", required=True, type=int)
    guard.add_argument("--baseline", required=True)
    guard.add_argument("--summary-file", default=None)
    guard.set_defaults(handler=_guard_gate)

    wheel = subparsers.add_parser("wheel-gate", help="every declared package is in the wheel")
    wheel.add_argument("--wheel", required=True)
    wheel.add_argument("--repo-root", required=True)
    wheel.add_argument("--summary-file", default=None)
    wheel.set_defaults(handler=_wheel_gate)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess in tests
    raise SystemExit(main())
