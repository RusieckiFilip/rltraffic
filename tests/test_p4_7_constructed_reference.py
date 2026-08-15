"""The mixture tiers' CONSTRUCTED behaviour reference (``BRIEF_19`` section 3).

**Why a mixture's reference is constructed rather than rolled out.**  C1's per-tier question is *"did
the method beat the policy that produced its data?"*, and on a mixture there is no such policy: a
33 %-expert corpus is a **composition**, not something anyone ran.  There is nothing to roll out.
Substituting a training-draw number instead is **void under ``PREREGISTRATION`` A5** (a comparison
must be over shared draw ids), and P4.6 shipped exactly that bug into a log.

**So the reference is built the way the corpus is built.**  ``mixture_training_streams`` composes a
corpus of exactly ``round(200 * f)`` expert streams and the rest random -- a **fixed composition, not
a coin flip** -- so the matching held-out reference assigns exactly ``round(100 * f)`` of each seed's
100 held-out draws to the ``mappo1000`` cell and the remainder to the ``random`` cell, by a declared
RNG, reading the **stored per-draw values** of the two committed cells.  Zero compute; paired by draw
and A5-compliant by construction.

⚠️ **A REALISATION, NOT AN EXPECTATION, and the tests enforce the difference.**  Taking
``f * ATT_expert(d) + (1 - f) * ATT_random(d)`` per draw would give the same *mean* and an
**understated variance**, because it removes the composition's own randomness -- and a paired CI
computed against an expectation would overstate precision.  ``test_every_constructed_value_is_one_of
_the_two_components`` is what makes the average impossible to ship by accident.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from offline.method_tier_grid import MIXTURE_EXPERT_FRACTION, MIXTURE_TIER_ORDER
from offline.mixture_tiers import (
    CONSTRUCTED_EPISODE_FIELDS,
    CONSTRUCTED_REFERENCE_FORMAT_VERSION,
    CONSTRUCTED_REFERENCE_RNG_SEED,
    component_episodes,
    constructed_reference_artifact,
    constructed_reference_assignment,
)

SEEDS = (101, 202, 303, 404, 505)
DRAWS = tuple(range(1000, 1100))


def synthetic_component(base: float, seeds=SEEDS, draws=DRAWS) -> dict[tuple[int, int], dict[str, float]]:
    """A component cell whose every value is unique, so its provenance is traceable.

    ``base`` separates the two components by two orders of magnitude, exactly as the real ones are
    separated (``mappo1000`` 105.58 against ``random`` 428.88), and the per-record offsets make every
    single value distinguishable -- so a test can tell not just *which component* a number came from
    but *which record*.
    """
    return {
        (int(seed), int(draw)): {
            "att_horizon": base + 0.001 * (draw - 1000) + 0.1 * seeds.index(seed),
            "horizon_vehicle_count": base * 10 + draw - 1000,
            "episode_reward": -base * 100 - (draw - 1000),
        }
        for seed in seeds
        for draw in draws
    }


@pytest.fixture()
def components() -> tuple[dict[tuple[int, int], dict[str, float]], dict[tuple[int, int], dict[str, float]]]:
    return synthetic_component(100.0), synthetic_component(400.0)


# ----------------------------------------------------------------------
# The assignment
# ----------------------------------------------------------------------


def test_each_seed_gets_exactly_the_declared_number_of_expert_draws() -> None:
    """``round(100 * f)`` = 33 / 50 / 67, per seed, without replacement.

    Per seed rather than pooled: a pooled draw could give one seed 10 expert draws and another 56
    and still average to the fraction, which would make the reference's per-seed composition differ
    from the corpus's while its mean looked right.
    """
    for tier, fraction in MIXTURE_EXPERT_FRACTION.items():
        expected = int(round(100 * fraction))
        assignment = constructed_reference_assignment(tier, SEEDS, DRAWS)
        assert sorted(assignment) == sorted(SEEDS), tier
        for seed, expert_draws in assignment.items():
            assert len(expert_draws) == expected, (tier, seed)
            assert len(set(expert_draws)) == expected, (tier, seed)
            assert set(expert_draws) <= set(DRAWS), (tier, seed)
            assert expert_draws == sorted(expert_draws), (tier, seed)


def test_the_expected_counts_are_the_ones_the_plan_declares() -> None:
    """33 / 50 / 67, derived from the fractions rather than trusted as literals."""
    assert [int(round(100 * MIXTURE_EXPERT_FRACTION[t])) for t in MIXTURE_TIER_ORDER] == [33, 50, 67]


def test_the_assignment_is_reproducible_and_independent_across_seeds_and_tiers() -> None:
    """One declared RNG stream per (tier, seed).

    Two seeds sharing an assignment would make the reference's five seeds redundant, and two tiers
    sharing one would tie ``mix33``'s reference to ``mix67``'s by construction rather than by chance.
    """
    first = {tier: constructed_reference_assignment(tier, SEEDS, DRAWS) for tier in MIXTURE_TIER_ORDER}
    again = {tier: constructed_reference_assignment(tier, SEEDS, DRAWS) for tier in MIXTURE_TIER_ORDER}
    assert first == again
    for tier, assignment in first.items():
        distinct = {tuple(draws) for draws in assignment.values()}
        assert len(distinct) == len(SEEDS), tier
    assert first["mix33"][101] != first["mix50"][101][: len(first["mix33"][101])]


def test_an_unknown_tier_is_refused() -> None:
    """A non-mixture tier has a rolled-out reference and must not get a constructed one."""
    with pytest.raises(ValueError, match="mix33"):
        constructed_reference_assignment("mappo1000", SEEDS, DRAWS)


# ----------------------------------------------------------------------
# The realisation
# ----------------------------------------------------------------------


def test_every_constructed_value_is_one_of_the_two_components(components) -> None:
    """⚠️ The test that makes the per-draw AVERAGE impossible to ship by accident.

    An expectation ``f * A + (1 - f) * B`` would reproduce the mean and understate the variance, so
    a paired CI against it would overstate precision.  Every constructed record must be one of the
    two measured records for that exact (seed, draw), and all three of its fields must come from the
    SAME one -- a record mixing ATT from one component with the vehicle count from the other would
    describe an episode that no rollout produced.
    """
    expert, random_pool = components
    for tier in MIXTURE_TIER_ORDER:
        payload = constructed_reference_artifact(
            tier, expert, random_pool, seeds=SEEDS, draws=DRAWS
        )
        for record in payload["episodes"]:
            key = (int(record["seed"]), int(record["draw_id"]))
            assert record[CONSTRUCTED_EPISODE_FIELDS[0]] in (
                expert[key]["att_horizon"],
                random_pool[key]["att_horizon"],
            ), (tier, key)
            source = expert[key] if record["att_horizon"] == expert[key]["att_horizon"] else random_pool[key]
            for field in CONSTRUCTED_EPISODE_FIELDS:
                assert record[field] == source[field], (tier, key, field)


def test_the_realised_composition_matches_the_recorded_assignment(components) -> None:
    """The label is not decoration: the values follow the assignment the artifact records."""
    expert, random_pool = components
    for tier in MIXTURE_TIER_ORDER:
        payload = constructed_reference_artifact(
            tier, expert, random_pool, seeds=SEEDS, draws=DRAWS
        )
        assignment = {int(seed): set(draws) for seed, draws in payload["assignment"].items()}
        for record in payload["episodes"]:
            key = (int(record["seed"]), int(record["draw_id"]))
            expected = expert[key] if key[1] in assignment[key[0]] else random_pool[key]
            assert record["att_horizon"] == expected["att_horizon"], (tier, key)


def test_the_cell_is_the_same_grid_as_every_method_cell(components) -> None:
    """500 records on exactly the 5 seeds x 100 draws every other arm is measured on (A5)."""
    expert, random_pool = components
    payload = constructed_reference_artifact("mix50", expert, random_pool, seeds=SEEDS, draws=DRAWS)
    grid = {(int(r["seed"]), int(r["draw_id"])) for r in payload["episodes"]}
    assert grid == {(s, d) for s in SEEDS for d in DRAWS}
    assert len(payload["episodes"]) == 500
    assert payload["cell"]["n_episodes"] == 500
    assert payload["cell"]["draw_ids"] == list(DRAWS)
    assert payload["cell"]["seeds"] == list(SEEDS)


def test_the_mean_lies_strictly_inside_the_bracket_and_moves_with_the_fraction(components) -> None:
    """An interpolation between two measured endpoints, ordered by the expert fraction.

    Strictly inside, not merely between: a reference equal to either endpoint would mean the
    composition never took effect.
    """
    expert, random_pool = components
    low = sum(v["att_horizon"] for v in expert.values()) / len(expert)
    high = sum(v["att_horizon"] for v in random_pool.values()) / len(random_pool)
    means = {}
    for tier in MIXTURE_TIER_ORDER:
        payload = constructed_reference_artifact(
            tier, expert, random_pool, seeds=SEEDS, draws=DRAWS
        )
        mean = payload["cell"]["att_horizon_mean"]
        assert low < mean < high, tier
        means[tier] = mean
    assert means["mix67"] < means["mix50"] < means["mix33"]


def test_the_mean_is_the_fsum_of_the_records_it_reports(components) -> None:
    """The reported cell mean, recomputed by an independent route over the emitted records."""
    expert, random_pool = components
    payload = constructed_reference_artifact("mix33", expert, random_pool, seeds=SEEDS, draws=DRAWS)
    recomputed = math.fsum(float(r["att_horizon"]) for r in payload["episodes"]) / len(
        payload["episodes"]
    )
    assert payload["cell"]["att_horizon_mean"] == pytest.approx(recomputed, abs=1e-12)


# ----------------------------------------------------------------------
# The label, the provenance and the refusals
# ----------------------------------------------------------------------


def test_the_artifact_labels_itself_constructed_and_records_how(components) -> None:
    """A reader must be able to tell it from a rolled-out cell without reading code."""
    expert, random_pool = components
    payload = constructed_reference_artifact("mix33", expert, random_pool, seeds=SEEDS, draws=DRAWS)
    assert payload["format_version"] == CONSTRUCTED_REFERENCE_FORMAT_VERSION
    assert payload["arm"] == "behaviour@mix33"
    assert payload["method"] == "behaviour"
    assert payload["tier"] == "mix33"
    reference = payload["behaviour_reference"]
    assert reference["source"] == "constructed"
    assert reference["source"] != "measured"
    assert reference["rng_seed"] == CONSTRUCTED_REFERENCE_RNG_SEED
    assert reference["expert_fraction"] == MIXTURE_EXPERT_FRACTION["mix33"]
    assert reference["expert_draws_per_seed"] == 33
    assert reference["components"] == ["behaviour@mappo1000", "behaviour@random"]
    # The realised assignment is recorded in full, not as a digest: a reader must be able to rebuild
    # the cell, and a digest only lets them check one they already have.
    assert sorted(int(s) for s in payload["assignment"]) == sorted(SEEDS)
    assert all(len(draws) == 33 for draws in payload["assignment"].values())
    assert "realisation" in reference["role"].lower()


def test_the_bracket_is_reported_beside_the_reference(components) -> None:
    """``BRIEF_19`` section 3 constraint 3: legible as an interpolation between two measurements."""
    expert, random_pool = components
    payload = constructed_reference_artifact("mix50", expert, random_pool, seeds=SEEDS, draws=DRAWS)
    bracket = payload["bracket"]
    assert set(bracket) == {"behaviour@mappo1000", "behaviour@random"}
    assert bracket["behaviour@mappo1000"]["att_horizon_mean"] < payload["cell"]["att_horizon_mean"]
    assert bracket["behaviour@random"]["att_horizon_mean"] > payload["cell"]["att_horizon_mean"]
    assert bracket["behaviour@mappo1000"]["n_episodes"] == 500


def test_a_component_missing_a_draw_is_refused(components) -> None:
    """A hole in a component is a refusal, never a shorter cell.

    A 499-record reference would still produce a mean, a CI and a paired comparison -- against a
    different draw set from every other arm, which A5 makes void.
    """
    expert, random_pool = components
    del expert[(303, 1042)]
    with pytest.raises(ValueError, match="1042"):
        constructed_reference_artifact("mix33", expert, random_pool, seeds=SEEDS, draws=DRAWS)


def test_a_component_missing_a_field_is_refused(components) -> None:
    """Every carried field must exist in the source record."""
    expert, random_pool = components
    del random_pool[(101, 1000)]["horizon_vehicle_count"]
    with pytest.raises(ValueError, match="horizon_vehicle_count"):
        constructed_reference_artifact("mix67", expert, random_pool, seeds=SEEDS, draws=DRAWS)


def test_a_component_must_match_its_primary_artifact_exactly() -> None:
    """The grid carries a COPY of each behaviour cell; the copy is checked against its source.

    Reading only the copy would be unverified, and the check is not free of consequence: it is also
    what proves the secured raw evidence in ``output/p4_6/`` is faithful to what the merged artifact
    reports.  A single differing value is refused and the refusal names the (seed, draw).
    """
    from offline.mixture_tiers import component_from_sources

    grid = {
        "episodes": [
            {"arm": "behaviour@random", "seed": 101, "draw_id": 1000, "att_horizon": 428.5,
             "horizon_vehicle_count": 12.0, "episode_reward": -3.0},
            {"arm": "behaviour@random", "seed": 101, "draw_id": 1001, "att_horizon": 429.5,
             "horizon_vehicle_count": 13.0, "episode_reward": -4.0},
        ]
    }
    primary = {
        "episodes": [
            {"arm": "behaviour@random", "seed": 101, "draw_id": 1000, "att_horizon": 428.5,
             "horizon_vehicle_count": 12.0, "episode_reward": -3.0},
            {"arm": "behaviour@random", "seed": 101, "draw_id": 1001, "att_horizon": 429.5,
             "horizon_vehicle_count": 13.0, "episode_reward": -4.0},
        ]
    }
    got = component_from_sources(grid, "behaviour@random", primary, "behaviour@random")
    assert sorted(got) == [(101, 1000), (101, 1001)]

    primary["episodes"][1]["att_horizon"] = 429.5000000001
    with pytest.raises(ValueError, match="draw 1001"):
        component_from_sources(grid, "behaviour@random", primary, "behaviour@random")

    primary["episodes"][1]["att_horizon"] = 429.5
    primary["episodes"].append(
        {"arm": "behaviour@random", "seed": 101, "draw_id": 1002, "att_horizon": 430.0,
         "horizon_vehicle_count": 14.0, "episode_reward": -5.0}
    )
    with pytest.raises(ValueError, match="different .seed, draw. sets"):
        component_from_sources(grid, "behaviour@random", primary, "behaviour@random")


def test_component_episodes_reads_one_arm_of_a_grid_artifact(tmp_path: Path) -> None:
    """The committed grid artifact is the source, and duplicates are refused rather than merged."""
    payload: dict[str, Any] = {
        "episodes": [
            {"arm": "behaviour@mappo1000", "seed": 101, "draw_id": 1000, "att_horizon": 1.0,
             "horizon_vehicle_count": 2.0, "episode_reward": -3.0},
            {"arm": "behaviour@random", "seed": 101, "draw_id": 1000, "att_horizon": 9.0,
             "horizon_vehicle_count": 8.0, "episode_reward": -7.0},
        ]
    }
    got = component_episodes(payload, "behaviour@mappo1000")
    assert got == {(101, 1000): {"att_horizon": 1.0, "horizon_vehicle_count": 2.0,
                                 "episode_reward": -3.0}}
    payload["episodes"].append(dict(payload["episodes"][0]))
    with pytest.raises(ValueError, match="twice"):
        component_episodes(payload, "behaviour@mappo1000")
    with pytest.raises(ValueError, match="artifact carries no episodes"):
        component_episodes(payload, "behaviour@nothing")
