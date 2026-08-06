"""Mechanical guard for the P8.0 naming ruling (a grep is stronger than a convention).

The horizon-metric defect was born from a name collision: an *episode-level* number was carried
under the registry name of the *per-step* metric, ``average_travel_time``. The ruling
(``docs/briefs/BRIEF_06_p8.0_horizon_metric.md`` §2) forbids any episode-level field, column,
variable or dict key in ``offline/`` from being named bare ``average_travel_time``. The bare name
is legitimate *only* as the registry-metric string when the metric is read
(``get("average_travel_time")``, ``info["average_travel_time"]``, a ``metric_names`` list).

This test enforces that with an AST scan and a positive control proving the scanner can fail.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_DIR = REPO_ROOT / "offline"
FORBIDDEN = "average_travel_time"


def _violations(source: str, where: str) -> list[str]:
    """Return human-readable violations of the naming ruling in *source*.

    Flags ``average_travel_time`` used as: an attribute (``x.average_travel_time``), an
    assignment/annotation/loop target or function argument (a *name* we bind), a keyword-argument
    name, or a **constructed** dict-literal key (``{"average_travel_time": ...}``). It does NOT
    flag a bare string constant used to *read* the registry metric -- ``info["average_travel_time"]``
    (a Subscript) or ``get("average_travel_time")`` (a call argument) -- which are the only allowed
    appearances.
    """
    tree = ast.parse(source, filename=where)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == FORBIDDEN:
            out.append(f"{where}:{node.lineno}: attribute .{FORBIDDEN}")
        elif isinstance(node, ast.Name) and node.id == FORBIDDEN and isinstance(
            node.ctx, ast.Store
        ):
            out.append(f"{where}:{node.lineno}: bound name {FORBIDDEN}")
        elif isinstance(node, ast.arg) and node.arg == FORBIDDEN:
            out.append(f"{where}:{node.lineno}: function arg {FORBIDDEN}")
        elif isinstance(node, ast.keyword) and node.arg == FORBIDDEN:
            out.append(f"{where}:{node.lineno}: keyword arg {FORBIDDEN}")
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value == FORBIDDEN:
                    out.append(f"{where}:{node.lineno}: dict-literal key '{FORBIDDEN}'")
    return out


def test_scanner_flags_the_forbidden_shapes() -> None:
    """Positive control: every forbidden shape is caught, and no allowed shape is."""
    bad = (
        "x = obj.average_travel_time\n"
        "d = {'average_travel_time': 1.0}\n"
        "average_travel_time = 2.0\n"
        "def f(average_travel_time):\n    return average_travel_time\n"
        "g(average_travel_time=3.0)\n"
    )
    flagged = _violations(bad, "<synthetic-bad>")
    assert len(flagged) == 5, f"expected 5 violations, got {flagged}"

    good = (
        "v = info['average_travel_time']\n"
        "w = metrics.get('average_travel_time')\n"
        "names = ['average_travel_time']\n"
        "'''average_travel_time in a docstring is fine'''\n"
    )
    assert _violations(good, "<synthetic-good>") == []


def test_offline_has_no_episode_level_average_travel_time() -> None:
    """No file under ``offline/`` names an episode-level thing bare ``average_travel_time``."""
    py_files = sorted(p for p in OFFLINE_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    assert py_files, "no python files found under offline/ -- wrong path?"
    all_violations: list[str] = []
    for path in py_files:
        rel = str(path.relative_to(REPO_ROOT))
        all_violations.extend(_violations(path.read_text(encoding="utf-8"), rel))
    assert all_violations == [], (
        "episode-level use of the bare per-step metric name (P8.0 naming ruling):\n"
        + "\n".join(all_violations)
    )
