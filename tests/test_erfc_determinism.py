"""The p-value path must be identical on every machine (P0.10, 2026-08-17).

WHY THIS FILE EXISTS
--------------------
The project's first cross-machine CI run failed on a 1-ulp difference.  Every intermediate
of ``offline.dt_gate.wilcoxon_signed_rank`` was bit-identical on the dev machine and on a
GitHub runner -- ``w_plus``, ``variance``, ``z``, and the erfc argument itself, to the last
hex digit -- and only ``math.erfc`` differed (glibc 2.43 against 2.39).  ``math.erfc`` is the
platform's libm and is not correctly rounded, so it is not a portable function.

``offline.dt_gate._erfc_deterministic`` replaces it with a :mod:`decimal` computation whose
arithmetic is exactly specified, and therefore identical everywhere.  These tests pin the
three properties that makes the replacement safe:

1. **it is correct** -- it agrees with libm wherever libm is right, and it is right where
   they disagree (checked against an independently computed high-precision value);
2. **no committed number moves** -- every ``(z, p_value)`` pair in ``docs/data/*.json``
   still reproduces exactly;
3. **it does not depend on ambient state** -- the caller's :mod:`decimal` context cannot
   change the answer.

⚠️ Property 2 is the load-bearing one.  If it ever fails, the change is not a bug fix, it is
a silent revision of published numbers.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, getcontext, localcontext
from pathlib import Path

import pytest

from offline.dt_gate import _erfc_deterministic, _normal_cdf, _pi_at

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "docs" / "data"

# The exact argument at which CI failed, and the exact value the two libms disagreed on.
FAILING_ARGUMENT = float.fromhex("0x1.c2c68c1f67a84p+1")
COMMITTED_P_VALUE = float.fromhex("0x1.54a3046cb4c1dp-21")  # 6.344854575852746e-07
RUNNERS_LIBM_VALUE = float.fromhex("0x1.54a3046cb4c1cp-21")  # 6.344854575852745e-07


def test_pi_is_computed_correctly() -> None:
    """The constant is computed, not transcribed, so it must be checked against a known one."""
    assert float(_pi_at(60)) == math.pi
    assert str(_pi_at(60))[:17] == "3.141592653589793"


@pytest.mark.parametrize(
    "x",
    [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, -0.5, -1.0, -2.0, -3.5, -6.0],
)
def test_agrees_with_the_platform_libm_where_the_platform_is_right(x: float) -> None:
    """A replacement that disagreed with libm everywhere would be a new bug, not a fix.

    These are points where this machine's glibc happens to be correctly rounded, which is
    the overwhelmingly common case -- the whole issue is the rare last-ulp disagreement.
    """
    assert _erfc_deterministic(x) == math.erfc(x)


def test_it_returns_the_correctly_rounded_value_where_two_libms_disagree() -> None:
    """The point CI failed on, decided by an independent high-precision computation.

    The exact value is 6.34485457585274551150639655222548177...e-07, so the correctly
    rounded double is the committed one and the runner's libm was the inaccurate side.
    That is recomputed here from scratch rather than asserted from the docstring.
    """
    with localcontext() as context:
        context.prec = 60
        argument = Decimal(FAILING_ARGUMENT)
        total, n = Decimal(0), 0
        while n < 200:
            term = (-1) ** n * argument ** (2 * n + 1) / Decimal(math.factorial(n) * (2 * n + 1))
            total += term
            n += 1
        exact = 1 - 2 / _pi_at(60).sqrt() * total

    assert float(exact) == COMMITTED_P_VALUE
    assert float(exact) != RUNNERS_LIBM_VALUE
    assert _erfc_deterministic(FAILING_ARGUMENT) == COMMITTED_P_VALUE


def test_the_committed_p_values_all_still_reproduce_exactly() -> None:
    """THE LOAD-BEARING TEST: 322 published p-values, none of which may move.

    Recomputed from each artifact's own recorded ``z`` through the replacement routine and
    compared with ``==``.  A tolerance here would defeat the purpose: the question is
    precisely whether any published number changes.
    """
    pairs: list[tuple[str, float, float]] = []

    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            if "z" in node and "p_value" in node and isinstance(node.get("z"), (int, float)):
                pairs.append((path, float(node["z"]), float(node["p_value"])))
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    for artifact in sorted(DATA_DIR.glob("*.json")):
        walk(json.loads(artifact.read_text(encoding="utf-8")), artifact.name)

    assert len(pairs) == 322, (
        f"expected the 322 committed (z, p_value) pairs measured on 2026-08-17, found "
        f"{len(pairs)}; if artifacts were added, re-measure and update this count deliberately"
    )

    moved = [
        (path, z, committed, min(1.0, 2.0 * _normal_cdf(z)))
        for path, z, committed in pairs
        if min(1.0, 2.0 * _normal_cdf(z)) != committed
    ]
    assert moved == [], f"{len(moved)} published p-value(s) would change: {moved[:5]}"


def test_the_p_value_path_does_not_call_the_platform_libm_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property is *cross-machine* determinism, which one machine cannot falsify.

    On this machine glibc happens to be correctly rounded at every point tested, so
    reverting to ``math.erfc`` passes every other test in this file -- measured, not
    assumed (P0.10 mutation K survived until this test existed).  What *is* checkable
    locally is the mechanism: no platform ``erfc`` may appear in the path at all.  So it is
    replaced with a landmine.

    ``math.sqrt`` is deliberately left alone: IEEE 754 requires it to be correctly rounded,
    so it is portable in a way ``erfc`` is not.
    """

    def landmine(_: float) -> float:
        raise AssertionError("math.erfc reached: the p-value path is platform-dependent again")

    monkeypatch.setattr(math, "erfc", landmine)

    assert _normal_cdf(0.0) == 0.5
    assert _erfc_deterministic(FAILING_ARGUMENT) == COMMITTED_P_VALUE
    assert min(1.0, 2.0 * _normal_cdf(-4.98046875)) > 0.0


def test_the_callers_decimal_context_cannot_change_the_answer() -> None:
    """Ambient global state must not reach the result, or "deterministic" is not true.

    :mod:`decimal` contexts are per-thread and mutable; a routine that read the caller's
    precision would be reproducible only by accident.
    """
    baseline = _erfc_deterministic(FAILING_ARGUMENT)
    original = getcontext().prec
    try:
        for precision in (1, 7, 28, 200):
            getcontext().prec = precision
            assert _erfc_deterministic(FAILING_ARGUMENT) == baseline
    finally:
        getcontext().prec = original


def test_the_extremes_are_handled_without_a_series() -> None:
    """Beyond |x| = 30 the double result is saturated, and the series would be ruinous."""
    assert _erfc_deterministic(30.0) == 0.0
    assert _erfc_deterministic(1e300) == 0.0
    assert _erfc_deterministic(-30.0) == 2.0
    assert _erfc_deterministic(-1e300) == 2.0
    assert math.isnan(_erfc_deterministic(float("nan")))


def test_it_is_monotone_decreasing_across_the_working_range() -> None:
    """A series that lost convergence in the tail would most likely break monotonicity."""
    grid = [i / 4.0 for i in range(-40, 60)]
    values = [_erfc_deterministic(x) for x in grid]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[0] == pytest.approx(2.0)
    assert values[-1] >= 0.0


def test_the_normal_cdf_still_means_what_its_name_says() -> None:
    """Anchors the wiring: a wrong sign or a missing halving would sail past the tests above."""
    assert _normal_cdf(0.0) == 0.5
    assert _normal_cdf(-1.959963984540054) == pytest.approx(0.025, abs=1e-9)
    assert _normal_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-9)
