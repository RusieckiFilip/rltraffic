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

1. **it is correct** -- checked against an *independent* algorithm (Laplace's continued
   fraction, computed in :mod:`decimal`), never against the platform libm, which is the
   thing under suspicion.  Agreement with libm is asserted only as a **1-ulp bound**;
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
import struct
from decimal import (
    Decimal,
    FloatOperation,
    Inexact,
    ROUND_05UP,
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
    ROUND_HALF_DOWN,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    ROUND_UP,
    Rounded,
    Subnormal,
    Underflow,
    localcontext,
)
from pathlib import Path

import pytest

from offline.dt_gate import _erfc_deterministic, _normal_cdf, _pi_at

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "docs" / "data"

# The exact argument at which CI failed, and the exact value the two libms disagreed on.
FAILING_ARGUMENT = float.fromhex("0x1.c2c68c1f67a84p+1")
COMMITTED_P_VALUE = float.fromhex("0x1.54a3046cb4c1dp-21")  # 6.344854575852746e-07
RUNNERS_LIBM_VALUE = float.fromhex("0x1.54a3046cb4c1cp-21")  # 6.344854575852745e-07

# Captured once, under the DEFAULT context, so the parametrised context tests below compare
# against a fixed value rather than against another call made under the same odd context.
EXPECTED_AT_FAILING_ARGUMENT = COMMITTED_P_VALUE
# ⚠️ These context tests MUST call _pi_at at a precision nothing else has used. An earlier
# draft precomputed three constants at module level; every case then hit the memo, the series
# never ran under the odd context, and mutation S (removing _pi_at's trap clearing) SURVIVED.
# The precision offsets below keep every case on a cold path.
ROUNDING_MODES = [
    ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_DOWN,
    ROUND_HALF_UP, ROUND_UP, ROUND_05UP, ROUND_HALF_EVEN,
]
TRAP_SIGNALS = [Inexact, Rounded, FloatOperation, Underflow, Subnormal]

#: Derived from the oracle's Euler formula, defined below, so the context tests compare
#: Machin against an independent constant rather than against themselves.
PI_REFERENCE_PREFIX = ""  # filled at import time, after _pi_euler is defined


@pytest.mark.parametrize("precision", [28, 45, 60, 120, 300])
def test_pi_agrees_with_an_independently_derived_pi(precision: int) -> None:
    """R7: 16 of 65+ digits was not a check, and the old check shared the code's own formula.

    Machin (in the code) against Euler (here), compared over every digit the code reports.
    """
    # NO unary + on either side: it would re-round to the AMBIENT context (28 digits by
    # default) and silently truncate the comparison to a prefix both values share. That is
    # exactly what the first draft of this test did, and it "failed" for that reason.
    from_code = str(_pi_at(precision))
    from_euler = str(_pi_euler(precision))
    assert len(from_code.replace(".", "")) >= precision
    assert len(from_euler.replace(".", "")) >= precision

    compared = precision  # digits, excluding the leading "3."
    assert from_code[2 : 2 + compared] == from_euler[2 : 2 + compared], (
        f"Machin and Euler disagree within the reported {precision} digits:\n"
        f"  machin {from_code}\n  euler  {from_euler}"
    )
    assert float(_pi_at(precision)) == math.pi
    assert from_code.startswith("3.1415926535897932384626433832795028841971693993751"[: 2 + precision])


def test_pi_is_bit_identical_to_what_the_previous_exit_condition_produced() -> None:
    """R3's fix must be INERT, and this is what makes that mechanical rather than my word.

    The exit test changed from "wait for the addend to underflow to zero" (925,608 loop
    iterations, and non-terminating under ``ROUND_UP``) to a magnitude threshold below the
    guard region (~40 iterations). These strings are what the OLD implementation returned,
    captured from it before the change and verified identical at six precisions -- so a
    future edit to the threshold that moves a digit fails here instead of quietly changing
    every p-value's last place.
    """
    assert str(_pi_at(45)) == "3.141592653589793238462643383279502884197169399375105826"
    assert str(_pi_at(60)) == (
        "3.141592653589793238462643383279502884197169399375105820974944592307812"
    )


def _pi_reference_prefix() -> str:
    """A pi prefix from the oracle's OWN formula, so the context tests are not self-referential."""
    return str(_pi_euler(80))[:60]


def _ulp_distance(a: float, b: float) -> int:
    """Number of representable doubles between *a* and *b* (same sign, finite)."""
    return abs(struct.unpack("<q", struct.pack("<d", a))[0] - struct.unpack("<q", struct.pack("<d", b))[0])


def _pi_euler(precision: int) -> Decimal:
    """An INDEPENDENT pi for the oracle: Euler's ``4*(atan(1/2) + atan(1/3))``.

    Review item **R7**: the oracle previously called the code's own ``_pi_at``, so the two
    shared a premise and their agreement measured that premise as much as the result.  This
    is a different formula (Euler's, not Machin's) reaching the same constant.
    """
    with localcontext() as context:
        context.prec = precision + 15

        def arctan_inverse(n: int) -> Decimal:
            reciprocal = Decimal(1) / Decimal(n)
            total = term = reciprocal
            k = 0
            while True:
                k += 1
                term = -term / (Decimal(n) ** 2)
                addend = term / (2 * k + 1)
                if abs(addend) < Decimal(10) ** -(precision + 14):
                    return total
                total += addend

        return +(4 * (arctan_inverse(2) + arctan_inverse(3)))


def _erfc_continued_fraction(x: float, precision: int = 80) -> float:
    """An INDEPENDENT high-precision erfc: Laplace's continued fraction, for ``x > 0``.

    Deliberately a different algorithm from the implementation under test -- a different
    series, and :meth:`Decimal.exp` instead of a Taylor sum -- so agreement is evidence
    rather than a restatement.  This is the "compute it twice by different routes" rule
    applied to a special function.

    ⚠️ **DO NOT "improve" this by passing an exact real instead of the float.**  The oracle
    takes the identical ``float64`` the implementation takes, and that shared premise is the
    point: the question under test is *"what is erfc of this double?"*, not *"what is erfc of
    the real number the double approximates?"*.  The coordinator's independent check of this
    module first appeared to contradict it at the 13th digit for exactly that reason -- both
    of their methods were fed the exact ``|z|/sqrt(2)`` while the code passes the rounded
    double, and one ulp in the argument times ``d/dx erfc = -4.6e-6`` is ~2e-21 out.  **Two
    algorithms that share a premise measure only that premise** (project rule, 0d21a19).
    """
    with localcontext() as context:
        context.prec = precision
        value = Decimal(x)
        fraction = Decimal(0)
        for k in range(400, 0, -1):
            fraction = Decimal(k) / 2 / (value + fraction)
        fraction = 1 / (value + fraction)
        return float((-value * value).exp() / _pi_euler(precision).sqrt() * fraction)


#: Both helpers exist by now, so the independent reference can be materialised.
PI_REFERENCE_PREFIX = _pi_reference_prefix()


@pytest.mark.parametrize("x", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5216841843933384, 4.0, 5.0, 6.0, 8.0])
def test_is_correctly_rounded_against_an_independent_continued_fraction(x: float) -> None:
    """The real correctness oracle, and it depends on no platform library.

    ⚠️ This replaced an earlier test of mine that asserted equality with ``math.erfc``.
    That test passed on the dev machine and **failed at 6 of 17 points on the CI runner**,
    because glibc 2.39 is 1 ulp off there while 2.43 is not -- it asserted the portability
    of the very thing this module exists to stop depending on.  See docs/returns/P0.10.md.
    """
    assert _erfc_deterministic(x) == _erfc_continued_fraction(x)


@pytest.mark.parametrize(
    "x",
    [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, -0.5, -1.0, -2.0, -3.5, -6.0],
)
def test_stays_within_one_ulp_of_the_platform_libm(x: float) -> None:
    """A replacement that disagreed with libm *wildly* would be a new bug, not a fix.

    The bound is 1 ulp rather than equality on purpose: libm is not correctly rounded and
    differs between C libraries, so equality with it is not a portable property. Measured
    2026-08-17 -- glibc 2.43 agrees exactly at all 17 of these points, glibc 2.39 at 11.
    """
    assert _ulp_distance(_erfc_deterministic(x), math.erfc(x)) <= 1


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
    assert _erfc_deterministic(FAILING_ARGUMENT) == COMMITTED_P_VALUE

    # The two candidates are ADJACENT doubles, which is a fact about the disagreement rather
    # than a restatement of the line above. (The previous `!= RUNNERS_LIBM_VALUE` here was
    # implied by the equality and could not fail independently -- review item, theatre.)
    assert _ulp_distance(COMMITTED_P_VALUE, RUNNERS_LIBM_VALUE) == 1
    assert RUNNERS_LIBM_VALUE < COMMITTED_P_VALUE


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

    # R8: rglob, not glob -- docs/data/ has subdirectories, and a future artifact placed in
    # one would have escaped this sweep WITHOUT tripping the == 322 count guard.
    for artifact in sorted(DATA_DIR.rglob("*.json")):
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


@pytest.mark.parametrize("precision", [1, 7, 28, 200])
def test_the_callers_precision_cannot_change_the_answer(precision: int) -> None:
    """Ambient global state must not reach the result, or "deterministic" is not true.

    :mod:`decimal` contexts are per-thread and mutable; a routine that read the caller's
    precision would be reproducible only by accident.
    """
    with localcontext() as context:
        context.prec = precision
        assert _erfc_deterministic(FAILING_ARGUMENT) == EXPECTED_AT_FAILING_ARGUMENT
        assert str(_pi_at(400 + precision)).startswith(PI_REFERENCE_PREFIX)


@pytest.mark.parametrize("rounding", ROUNDING_MODES)
def test_the_callers_rounding_mode_cannot_change_the_answer(rounding: str) -> None:
    """Review item R3, and the two modes that used to HANG are in this list.

    ``localcontext()`` copies the caller's whole context, and ``_pi_at``'s exit test was
    ``addend == 0`` -- it waited for underflow. Under ``ROUND_UP`` and ``ROUND_05UP`` an
    addend never reaches zero, so the loop never terminated (the reviewer measured both past
    8 s against a 0.31 s baseline). The exit is now a magnitude threshold and the rounding
    mode is pinned inside the function, so all eight modes agree **and finish**.
    """
    with localcontext() as context:
        context.rounding = rounding
        assert _erfc_deterministic(FAILING_ARGUMENT) == EXPECTED_AT_FAILING_ARGUMENT
        assert str(_pi_at(500 + ROUNDING_MODES.index(rounding))).startswith(PI_REFERENCE_PREFIX)


@pytest.mark.parametrize("signal", TRAP_SIGNALS)
def test_a_trap_the_caller_enabled_cannot_break_the_computation(signal: type) -> None:
    """R3's third axis: five ``traps`` settings used to raise instead of computing.

    These signals describe ordinary events in an inexact series -- they are informational,
    not errors -- so the function clears exactly those and leaves genuine error traps
    (``InvalidOperation``, ``DivisionByZero``) alone.
    """
    with localcontext() as context:
        context.traps[signal] = True
        assert _erfc_deterministic(FAILING_ARGUMENT) == EXPECTED_AT_FAILING_ARGUMENT
        assert str(_pi_at(600 + TRAP_SIGNALS.index(signal))).startswith(PI_REFERENCE_PREFIX)


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

    # Both endpoints were theatre: `approx(2.0)` at rel 1e-6 admits a catastrophically wrong
    # series, and `>= 0.0` is satisfied by returning zero. Pinned exactly instead -- erfc(-10)
    # is 2 to the last bit, and the tail value is checked against the independent oracle.
    assert values[0] == 2.0
    assert values[-1] == _erfc_continued_fraction(grid[-1])
    assert 0.0 < values[-1] < 1e-90


def test_the_normal_cdf_still_means_what_its_name_says() -> None:
    """Anchors the wiring: a wrong sign or a missing halving would sail past the tests above."""
    assert _normal_cdf(0.0) == 0.5
    assert _normal_cdf(-1.959963984540054) == pytest.approx(0.025, abs=1e-9)
    assert _normal_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-9)
