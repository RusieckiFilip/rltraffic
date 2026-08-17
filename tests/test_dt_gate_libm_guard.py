"""Structural guard: no platform libm may reach the p-value path (P0.10 review, R1).

WHY A STRUCTURAL GUARD AND NOT A RUNTIME ONE
--------------------------------------------
The first attempt at this property was a runtime landmine:
``monkeypatch.setattr(math, "erfc", raises)``.  The independent reviewer walked straight past
it with the **whole suite green** (``997 passed, 3 skipped``) by reverting the path with an
*early binding* -- exactly what ``from math import erfc`` produces::

    _ERFC = math.erfc
    G._normal_cdf = lambda z: 0.5 * _ERFC(-z / math.sqrt(2.0))
    G._erfc_deterministic = _ERFC

``setattr(math, "erfc", ...)`` rebinds a **module attribute** and cannot see a reference bound
earlier, so the landmine was pinned to one *spelling* of the regression rather than to the
regression.  **And no test of the value can help**: on the machine that runs this suite, libm
reproduces all 322 committed p-values exactly, so the value is blind to the revert *by
construction*.

**The general form, which is the transferable part:** when a property is about **how a value
was produced** rather than **what it is**, no test of the value can guard it.  The guard has
to be structural.  This file is that guard, in the shape of ``tests/test_offline_naming_guard.py``
-- an AST scan with a positive control proving the scanner can fail -- plus runtime *identity*
checks that catch a rebinding the source scan cannot see.

WHAT IS ALLOWED, AND WHY EACH ONE
---------------------------------
* ``math.sqrt`` -- IEEE 754 **mandates** it be correctly rounded, so it is portable in the way
  ``erfc`` is not.
* ``math.isnan``, ``math.factorial``, ``math.comb`` -- exact predicates and integer arithmetic.
* ``math.fsum`` -- exactly rounded by construction, and order-independent.
* ``math.log`` -- **the one accepted exception**, review item **R4**: it sizes the working
  precision and never touches the value (a 1-ulp difference shifts the digit count by at most
  one out of 45 guard digits).  ``BRIEF_25`` §4 rules it recorded rather than fixed;
  ``DEFERRED`` row 49 carries the libm-free replacement, already verified never to lower the
  precision.  **It is allow-listed by name, so substituting any other libm call there fails.**
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from offline import dt_gate

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARDED_FILE = REPO_ROOT / "offline" / "dt_gate.py"

# Platform libm functions that are NOT correctly rounded and therefore differ between C
# libraries.  This is the set whose appearance in the value path caused the defect.
NON_PORTABLE = frozenset(
    {
        "erf", "erfc", "exp", "exp2", "expm1", "log", "log1p", "log2", "log10", "pow",
        "gamma", "lgamma", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "hypot", "cbrt",
    }
)

# name -> the reason it is tolerated.  A reason is mandatory: an allowance nobody can justify
# in a clause is an allowance that should not be taken.
ALLOWED: dict[str, str] = {
    "log": (
        "R4: sizes the working precision only, never the value; 45 guard digits absorb a "
        "1-ulp difference. DEFERRED 49 carries the libm-free replacement."
    ),
}


def _libm_references(source: str, where: str) -> list[str]:
    """Every reference to a non-portable libm function in *source*, by AST.

    Catches the four shapes a revert can take: ``math.erfc(...)``, ``from math import erfc``,
    ``import math as m`` + ``m.erfc(...)``, and an alias binding ``_ERFC = math.erfc`` (which
    is an ``Attribute`` load and therefore caught by the first rule).

    Docstrings and comments are invisible to the AST, which is the point -- ``dt_gate.py``
    discusses ``math.erfc`` in four docstrings, and a grep-based guard would either flag those
    or be weakened until it stopped flagging anything.
    """
    tree = ast.parse(source, filename=where)

    module_aliases = {"math"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "math" and alias.asname:
                    module_aliases.add(alias.asname)

    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in module_aliases and node.attr in NON_PORTABLE:
                if node.attr not in ALLOWED:
                    found.append(f"{where}:{node.lineno}: {node.value.id}.{node.attr}")
        elif isinstance(node, ast.ImportFrom) and node.module == "math":
            for alias in node.names:
                if alias.name in NON_PORTABLE and alias.name not in ALLOWED:
                    found.append(
                        f"{where}:{node.lineno}: from math import {alias.name}"
                        " (an early binding a runtime patch cannot see)"
                    )
    return sorted(found)


# --------------------------------------------------------------------------------------
# positive control -- the scanner must be able to fail, in every shape
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("source", "expected_fragment"),
    [
        ("import math\ndef f(x):\n    return math.erfc(x)\n", "math.erfc"),
        ("from math import erfc\ndef f(x):\n    return erfc(x)\n", "from math import erfc"),
        ("import math as m\ndef f(x):\n    return m.erfc(x)\n", "m.erfc"),
        ("import math\n_ERFC = math.erfc\ndef f(x):\n    return _ERFC(x)\n", "math.erfc"),
        ("import math\ndef f(x):\n    return math.exp(-x * x)\n", "math.exp"),
    ],
)
def test_the_scanner_catches_every_shape_a_revert_could_take(
    source: str, expected_fragment: str
) -> None:
    """The reviewer's revert is the fourth row; ``from math import`` is the third."""
    hits = _libm_references(source, "<control>")
    assert hits, f"the scanner missed {expected_fragment!r}"
    assert any(expected_fragment in hit for hit in hits), hits


@pytest.mark.parametrize(
    "source",
    [
        '"""A docstring that mentions math.erfc and from math import erfc."""\nimport math\n',
        "import math\ndef f(x):\n    return math.sqrt(x)\n",           # IEEE-mandated
        "import math\ndef f(x):\n    return math.isnan(x)\n",          # exact predicate
        "import math\ndef f(x):\n    return math.fsum([x, x])\n",      # exactly rounded
        "import math\ndef f(x):\n    return math.log(x)\n",            # allow-listed, R4
    ],
)
def test_the_scanner_does_not_flag_what_is_portable_or_allowed(source: str) -> None:
    """A guard with false positives is a guard that gets deleted."""
    assert _libm_references(source, "<control>") == []


# --------------------------------------------------------------------------------------
# the real scan
# --------------------------------------------------------------------------------------
def test_dt_gate_has_no_platform_libm_in_the_value_path() -> None:
    """THE GUARD. A source-level revert of the P0.10 fix fails here."""
    hits = _libm_references(GUARDED_FILE.read_text(encoding="utf-8"), "offline/dt_gate.py")
    assert hits == [], (
        "a non-portable libm call reached offline/dt_gate.py: "
        + "; ".join(hits)
        + ". This is the defect P0.10 removed -- the value it produces will differ between C "
        "libraries. Use the decimal path, or add an entry to ALLOWED with a written reason."
    )


def test_the_allowance_list_is_justified_and_minimal() -> None:
    """An allowlist that grows silently is how a guard stops guarding."""
    assert set(ALLOWED) == {"log"}
    for name, reason in ALLOWED.items():
        assert len(reason) >= 40, f"{name} is allowed without a usable reason"


# --------------------------------------------------------------------------------------
# runtime identity -- catches a REBINDING, which the source scan cannot see
# --------------------------------------------------------------------------------------
def test_erfc_deterministic_is_not_the_platform_function() -> None:
    """The reviewer's exact revert sets this attribute to ``math.erfc``; this catches it.

    ``math.erfc`` is a builtin and carries no ``__code__``, so the identity check and the
    provenance check are two independent ways to notice the same substitution.
    """
    assert dt_gate._erfc_deterministic is not math.erfc
    assert hasattr(dt_gate._erfc_deterministic, "__code__"), (
        "_erfc_deterministic is not a Python function any more -- it has been replaced by a "
        "builtin, which is what a revert to platform libm looks like at runtime"
    )
    assert dt_gate._erfc_deterministic.__code__.co_filename == str(GUARDED_FILE)


def test_normal_cdf_is_still_the_function_defined_in_dt_gate() -> None:
    """The revert also replaces ``_normal_cdf`` with a lambda closing over the libm call."""
    function = dt_gate._normal_cdf

    assert function.__module__ == "offline.dt_gate"
    assert function.__name__ == "_normal_cdf", f"replaced by {function.__name__!r}"
    assert function.__code__.co_filename == str(GUARDED_FILE)
    assert "_erfc_deterministic" in function.__code__.co_names, (
        "_normal_cdf no longer calls _erfc_deterministic; its globals are "
        f"{function.__code__.co_names}"
    )
