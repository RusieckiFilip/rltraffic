"""P7.3a T6-T8: the declared arms, rho's arithmetic, the two stages and ``report``.

Written against ``BRIEF_37`` §3.5 / §4 + Amendments A (A1, A2), B (B1, B2) and C (C2, C6), and
``docs/plans/p7.3a.md``.

**Nothing scientific is chosen in this file or in the module it tests.** H3, rho, the arms, the
seeds, the pool and the stages are registered; these tests assert that what runs is what was
registered, and that a number cannot be produced for a cell nobody declared.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from offline import transfer_curve as tcv

REPO_DATA = Path(__file__).resolve().parents[1] / "docs" / "data"
P7_2B = REPO_DATA / "p7_2b_calibration.json"


# ----------------------------------------------------------------------------------
# T6 -- the arm set and the targets, READ from the sha-pinned artifact
# ----------------------------------------------------------------------------------
def test_t6_exactly_four_declared_arms_per_subject_with_the_registered_one_marked() -> None:
    """T6. The four arms ``BRIEF_37`` §2 closed, and the registered prompt is the artifact's.

    The lookup is by ``(rule, statistic, k)`` and the ``role`` is then cross-checked against the
    artifact's own: a table that disagreed would silently re-register the prompt, which is the one
    thing A17 does not permit this task to do.
    """
    artifact = json.loads(P7_2B.read_bytes())
    assert [spec.name for spec in tcv.DECLARED_ARMS] == [
        "b_mean_k100", "b_max_k100", "a_q1.0", "naive",
    ]

    for subject, expected_registered in (
        ("mappo1000", -7185.354721543778),
        ("mix50", -7431.0185327454665),
    ):
        targets = tcv.targets_for_subject(subject, artifact)
        assert sorted(targets) == ["a_q1.0", "b_max_k100", "b_mean_k100", "naive"]
        assert targets["b_mean_k100"]["role"] == "registered_prompt"
        assert targets["b_mean_k100"]["target_rtg"] == expected_registered
        assert [t["role"] for name, t in sorted(targets.items()) if name != "b_mean_k100"] == [
            "ablation", "ablation", "ablation",
        ]

    # k = 5 and k = 20 are EXCLUDED BY DECLARATION, before any number existed (A17(d)).
    assert all(spec.k in (100, None) for spec in tcv.DECLARED_ARMS)


def test_t6_a_role_disagreement_with_the_artifact_is_refused() -> None:
    """Reading ``b_max`` where ``b_mean`` is declared must not quietly produce a target.

    The mutation this is built against swaps the statistic in the lookup.  Without the role
    cross-check that swap returns a perfectly well-formed number for an arm labelled
    ``registered_prompt`` -- the H3 arm conditioned on the wrong prompt, with nothing to say so.
    """
    artifact = json.loads(P7_2B.read_bytes())
    tampered = json.loads(json.dumps(artifact))
    for row in tampered["targets"]["mappo1000"]:
        if (row["rule"], row["statistic"], row["k"]) == ("rule_b", "mean", 100):
            row["role"] = "ablation"

    with pytest.raises(ValueError, match="registered_prompt|role"):
        tcv.targets_for_subject("mappo1000", tampered)


# ----------------------------------------------------------------------------------
# T7 -- rho's arithmetic: the anchors are 0 and 1 BY CONSTRUCTION
# ----------------------------------------------------------------------------------
def test_t7_the_anchors_are_zero_and_one_exactly() -> None:
    """T7. §3.4's formula, with each anchor substituted for the arm: exactly 0.0 and exactly 1.0.

    Under ``==``, not ``approx``: the identity is algebraic, and a rho that is 0.9999999 for
    fixed-time is a defect in the pairing rather than a rounding question.
    """
    for fixedtime, maxpressure in ((366.07, 364.15), (1000.0, 1.0), (2.5, -7.5)):
        assert tcv.rho(fixedtime, fixedtime, maxpressure) == 0.0
        assert tcv.rho(maxpressure, fixedtime, maxpressure) == 1.0


def test_t7_values_outside_the_unit_interval_are_not_clipped() -> None:
    """§3.4: *"Values may exceed 100 or fall below 0; that is expected and is not clipped."*

    A DT worse than fixed-time gives a negative rho and a DT better than MaxPressure gives one
    above 1.  Both are results.  Clipping would turn a reportable outcome into a bound, and H3's
    first clause is precisely a statement about the sign.
    """
    assert tcv.rho(500.0, 400.0, 300.0) == -1.0      # worse than fixed-time
    assert tcv.rho(200.0, 400.0, 300.0) == 2.0       # better than MaxPressure


def test_t7_a_zero_denominator_is_refused_rather_than_reported() -> None:
    """Equal anchors make the normalisation undefined; any finite answer would be an invention."""
    with pytest.raises(ValueError, match="denominator is zero"):
        tcv.rho(350.0, 400.0, 400.0)


# ----------------------------------------------------------------------------------
# B1 -- the two declared stages
# ----------------------------------------------------------------------------------
def test_the_declared_cell_set_is_exactly_what_amendment_b1_declares() -> None:
    """B1's arithmetic, asserted as counts so a silent change to the design is visible.

    4 arms x 2 subjects x 5 seeds x 100 draws = 4,000 DT cells; ``fixedtime`` and ``maxpressure``
    100 each; ``random`` 5 policy seeds x 100 = 500. Total 4,700 -- the number the brief costs.
    """
    cells = tcv.declared_cells()
    assert len(cells) == 4700
    assert sum(1 for c in cells if c["kind"] == "dt") == 4000
    assert sum(1 for c in cells if c["arm"] == "random") == 500
    assert sum(1 for c in cells if c["arm"] == "fixedtime") == 100
    assert sum(1 for c in cells if c["arm"] == "maxpressure") == 100

    stage1 = tcv.declared_cells(tcv.STAGE_CONFIRMATORY)
    assert len(stage1) == 1200
    assert {c["arm"] for c in stage1} == {"b_mean_k100", "fixedtime", "maxpressure"}
    assert sum(1 for c in stage1 if c["kind"] == "dt") == 1000

    rest = tcv.declared_cells("rest")
    assert len(rest) == 3500
    assert {c["arm"] for c in rest} == {"b_max_k100", "a_q1.0", "naive", "random"}
    # The two stages PARTITION the campaign: nothing is dropped and nothing is run twice.
    assert len(stage1) + len(rest) == len(cells)
    names = [tcv.cell_chunk_name(c) for c in cells]
    assert len(set(names)) == len(names), "two cells share a chunk name and would overwrite"


def test_every_declared_draw_is_in_the_held_out_pool_and_none_is_a_training_draw() -> None:
    """A18(c) and PREREGISTRATION §5: the evaluation pool is 1000-1099 and nothing else."""
    from offline.materialise_draws import classify_draw_pool

    draws = {c["draw_id"] for c in tcv.declared_cells()}
    assert draws == set(range(1000, 1100))
    assert {classify_draw_pool(d) for d in draws} == {"held_out"}


def test_the_halting_cross_check_subset_is_declared_and_small() -> None:
    """Amendment C2: ON for every cell on draw 1000, OFF elsewhere -- 47 cells of 4,700.

    Value-neutral (A9b: identical ``att_env``, ``e_sumo`` and counts with it on and off) and 3.5x
    the cost, so it verifies the recorder on a declared subset rather than on every cell or none.
    """
    cells = tcv.declared_cells()
    checked = [c for c in cells if c["draw_id"] == tcv.HALTING_CHECK_DRAW]
    assert len(checked) == 47, "2 subjects x 4 arms x 5 seeds + fixedtime + maxpressure + 5 random"
    assert sum(1 for c in checked if c["kind"] == "dt") == 40
