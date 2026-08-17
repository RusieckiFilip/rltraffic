"""Tests for the distribution metadata: what ``pip install .`` actually ships, and pins.

**The defect these tests exist for.**  Until P0.10, ``[tool.setuptools.packages.find].include``
listed ``agent*, algorithms*, envs*, experiments*, metrics*, states*, utils*`` and **not**
``offline*``.  A clean install therefore shipped the platform and not the research code:
``offline/dataset.py``, ``offline/dt_gate.py``, ``offline/offline_baselines.py``,
``offline/method_tier_grid.py``, ``offline/mixture_tiers.py`` and ``offline/transfer_gate.py``
-- every number in the paper comes from those -- were absent from the wheel.  Editable mode
hid it: the editable finder's ``MAPPING`` has no ``offline`` key either, and the tests only
resolve ``offline`` because ``tests/`` carries an ``__init__.py``, which makes pytest prepend
the repository root to ``sys.path``.  Nothing that runs from inside the repo can see the bug.

So the load-bearing test here is the one that inspects a **built wheel**, not the one that
reads the config: a config test proves the string is present, an artifact test proves the
file is in the distribution.

**Pins.**  ``requirements-frozen.txt`` records the environment that produced the merged
numbers; it is a record, not a supported matrix and not an installer input.  The pins in
``pyproject.toml`` must agree with it, which is what stops the two drifting into a state
where the paper's environment cannot be reconstructed from either.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest
from setuptools import find_packages

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
FROZEN = REPO_ROOT / "requirements-frozen.txt"

# ``tests`` carries an ``__init__.py`` (that is how pytest ends up prepending the repo root
# to sys.path) but is deliberately not distributed: a test package in a wheel would ship
# fixtures and scratch-repo helpers to every consumer.  Every other top-level package is
# expected in the distribution, and that expectation is the general form of the P0.10 fix.
NOT_DISTRIBUTED = {"tests"}


def load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def configured_include() -> list[str]:
    return load_pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]


def discovered_packages() -> set[str]:
    """Run setuptools' own finder with the repository's own configuration."""
    return set(find_packages(where=str(REPO_ROOT), include=configured_include()))


def top_level_package_dirs() -> set[str]:
    """Directories at the repository root that Python would import as packages."""
    return {
        entry.name
        for entry in REPO_ROOT.iterdir()
        if entry.is_dir() and (entry / "__init__.py").is_file()
    }


def normalise(name: str) -> str:
    """PEP 503 normalisation, so ``Foo_Bar`` and ``foo-bar`` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_dependencies() -> dict[str, str]:
    """``{normalised name: pinned version}`` from ``[project]`` and its optional extras."""
    data = load_pyproject()["project"]
    requirements = list(data.get("dependencies", []))
    for extra in data.get("optional-dependencies", {}).values():
        requirements.extend(extra)

    pins: dict[str, str] = {}
    for requirement in requirements:
        match = re.fullmatch(r"\s*([A-Za-z0-9._-]+)\s*==\s*([^\s;]+)\s*", requirement)
        assert match is not None, (
            f"{requirement!r} is not an exact pin; P0.10 requires == for every runtime "
            "dependency so the environment can be reconstructed"
        )
        pins[normalise(match.group(1))] = match.group(2)
    return pins


def frozen_versions() -> dict[str, str]:
    """``{normalised name: version}`` from the frozen environment record."""
    versions: dict[str, str] = {}
    for line in FROZEN.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-e "):
            continue
        if "==" not in stripped:
            # ``name @ file:///...`` entries are records of a local install, not pins.
            continue
        name, version = stripped.split("==", 1)
        versions[normalise(name)] = version.strip()
    return versions


# --------------------------------------------------------------------------------------
# what gets packaged
# --------------------------------------------------------------------------------------
def test_offline_is_discovered_by_the_configured_package_finder() -> None:
    """The direct form of the fix: setuptools must see ``offline`` and its subpackage."""
    discovered = discovered_packages()

    assert "offline" in discovered
    assert "offline.policies" in discovered


def test_every_top_level_package_is_packaged_or_explicitly_excluded() -> None:
    """The general form: this is the test that would have caught the original defect.

    It is written against the filesystem rather than against a hard-coded list, so the
    *next* package added to the repository is covered without anyone remembering to.
    """
    discovered = discovered_packages()
    expected = top_level_package_dirs() - NOT_DISTRIBUTED

    missing = sorted(name for name in expected if name not in discovered)
    assert missing == [], (
        f"{missing} exist as importable packages but are not in the distribution; "
        "add them to [tool.setuptools.packages.find].include or to NOT_DISTRIBUTED "
        "with a reason"
    )


def test_the_excluded_packages_are_the_ones_we_meant_to_exclude() -> None:
    """Pins the allowlist itself, so widening it is a visible edit rather than a default."""
    assert NOT_DISTRIBUTED == {"tests"}
    for name in NOT_DISTRIBUTED:
        assert (REPO_ROOT / name / "__init__.py").is_file(), (
            f"{name} is allowlisted as a package that is not distributed, but it is not "
            "a package at all -- the allowlist has gone stale"
        )


def build_wheel(destination: Path) -> Path:
    """Build a real wheel from the tracked working tree and return its path.

    The sources are copied out of the worktree first, so the build cannot leave ``build/``
    or ``*.egg-info`` behind in the repository, and only *tracked* files are copied, so a
    stray local file cannot make the wheel look better than a fresh clone would.
    """
    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [Path(name) for name in listing.stdout.split("\0") if name]

    roots = top_level_package_dirs() - NOT_DISTRIBUTED
    wanted_files = {"pyproject.toml", "README.md", "LICENSE", "rewards.py"}
    source = destination / "src"
    for relative in tracked:
        if relative.as_posix() in wanted_files or relative.parts[0] in roots:
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / relative, target)

    output = destination / "wheel"
    subprocess.run(
        [
            sys.executable, "-m", "pip", "wheel", str(source),
            "--no-deps", "--no-build-isolation", "--quiet",
            "--wheel-dir", str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    wheels = sorted(output.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required to list tracked files")
def test_a_real_wheel_ships_the_research_code(tmp_path: Path) -> None:
    """The artifact test.  Before P0.10 this wheel contained no ``offline`` entry at all."""
    wheel = build_wheel(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    for member in (
        "offline/__init__.py",
        "offline/dataset.py",
        "offline/dt_gate.py",
        "offline/method_tier_grid.py",
        "offline/mixture_tiers.py",
        "offline/offline_baselines.py",
        "offline/transfer_gate.py",
        "offline/policies/__init__.py",
    ):
        assert member in names, f"{member} is missing from the built wheel"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required to list tracked files")
def test_the_wheel_carries_every_package_directory_the_repository_declares(
    tmp_path: Path,
) -> None:
    """Recomputed from the filesystem, not from the include list -- an independent route.

    ``test_every_top_level_package_is_packaged_or_explicitly_excluded`` asks setuptools
    what it *would* collect; this asks the zip what it *did*.
    """
    wheel = build_wheel(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        shipped = {name for name in archive.namelist() if name.endswith("__init__.py")}

    for root in sorted(top_level_package_dirs() - NOT_DISTRIBUTED):
        for init in sorted((REPO_ROOT / root).rglob("__init__.py")):
            if "__pycache__" in init.parts:
                continue
            relative = init.relative_to(REPO_ROOT).as_posix()
            assert relative in shipped, f"{relative} exists in the tree but not in the wheel"


# --------------------------------------------------------------------------------------
# pins
# --------------------------------------------------------------------------------------
def test_every_runtime_dependency_is_an_exact_pin() -> None:
    """Lower bounds cannot reconstruct an environment; ``declared_dependencies`` enforces it."""
    pins = declared_dependencies()
    assert {"gymnasium", "numpy", "torch", "traci"} <= set(pins)


def test_declared_pins_match_the_frozen_environment() -> None:
    """The two files must describe the same environment or neither can be trusted."""
    frozen = frozen_versions()

    for name, pinned in declared_dependencies().items():
        assert name in frozen, f"{name} is pinned in pyproject.toml but absent from {FROZEN.name}"
        installed = frozen[name]
        base = installed.split("+", 1)[0]
        assert base == pinned, (
            f"{name}: pyproject pins {pinned}, the frozen environment records {installed}"
        )


def test_the_torch_local_version_is_recorded_rather_than_pinned() -> None:
    """A ``+cu128`` pin would make the project uninstallable from PyPI, so it lives here.

    PEP 440 says a specifier with no local segment is satisfied by a version that has one,
    so ``torch==2.11.0`` is true of this environment.  The exact build is what the frozen
    file is for -- and this test checks the local segment explicitly, so the ``split("+")``
    in the test above cannot silently hide a change of build.
    """
    assert frozen_versions()["torch"] == "2.11.0+cu128"


def test_requires_python_matches_the_interpreter_that_produced_the_numbers() -> None:
    """Measured 3.12.13 against a declared ``>=3.9``; C5 and CLAUDE.md both say >=3.12."""
    requires = load_pyproject()["project"]["requires-python"]

    assert requires == ">=3.12", requires
    assert sys.version_info >= (3, 12)


def test_the_frozen_file_records_the_interpreter_and_says_what_it_is() -> None:
    """A bare ``pip freeze`` dump would not say which interpreter or tree produced it."""
    text = FROZEN.read_text(encoding="utf-8")
    header = "\n".join(line for line in text.splitlines() if line.startswith("#"))

    assert "3.12.13" in header
    assert "record" in header.lower()
    for package in ("numpy", "torch", "gymnasium", "traci"):
        assert re.search(rf"^{package}==", text, re.MULTILINE | re.IGNORECASE), package
