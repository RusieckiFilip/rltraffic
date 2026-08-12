"""Tests for P4.5's stream selectors, arm declaration and selection artifact.

P4.4 measured that %BC's top-10 % return filter drew 19 of its 20 streams from MAPPO seeds 101
and 202 -- the two best behaviour checkpoints -- and DEDUCED that the filter performs checkpoint
selection.  This task measures it with matched-size arms, so the tests here exist to protect one
thing above all: **the arms must differ in WHICH checkpoints produced their data and in nothing
else.**  ``test_the_design_check_refuses_unequal_matched_arm_sizes`` is the load-bearing one --
every other test can pass while two arms differ in size, and then the decisive comparison measures
data quantity while claiming to measure seed identity.

Three layers, and the Return Packet says which of them ran:

* the selectors, over the synthetic corpus of ``tests/test_offline_dataset.py`` -- always run;
* the design invariants, over hand-built artifact payloads -- always run, and they are the same
  function the artifact writer calls before it writes a byte;
* the committed-artifact checks, which live in ``test_offline_baselines.py`` beside P4.4's and are
  added once the campaign has produced the artifact.

Alignment convention and format versions are stated in the module under test; nothing here
restates them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import torch

import offline.dt_gate as dt_gate_module
import offline.offline_baselines as baselines_module
from offline.dataset import TrajectoryWindowDataset
from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS, EpisodeResult
from offline.offline_baselines import (
    DELTA_ATT,
    MATCHED_SUBSET_COUNT,
    SELECTION_ARMS,
    VERDICT_INCONCLUSIVE,
    VERDICT_MATCHES,
    VERDICT_BASELINE_BETTER,
    VERDICT_DT_BETTER,
    ArmSpec,
    arm_spec_for_flags,
    assert_reused_arm_reproduces,
    assert_selection_design,
    delta_verdict,
    filter_stacked_to_streams,
    random_stream_subset,
    select_arm_streams,
    selection_artifact,
    stream_returns,
    streams_from_datasets,
    thread_regime,
)
from offline.dt_gate import stack_dataset

import tests.test_offline_dataset as dataset_fixtures
from tests.test_offline_dataset import write_dataset_dir

REPO_ROOT = Path(__file__).resolve().parents[1]

CONTEXT = 4
FIXTURE_T = 8                        # decision rows per stream in the shared fixture
ALPHA_GROUP = (4, 3)                 # ix_alpha: state_dim 4, n_actions 3

#: The neutral verdict names, mapped to the DT-oriented ones the registered rule returns.  The
#: map is the specification of the double-compute in
#: ``test_the_neutral_delta_verdict_agrees_with_the_registered_one``.
VERDICT_NAME_MAP = {
    VERDICT_MATCHES: "within_delta",
    VERDICT_DT_BETTER: "left_genuinely_better",
    VERDICT_BASELINE_BETTER: "right_genuinely_better",
    VERDICT_INCONCLUSIVE: "inconclusive_at_this_power",
}


@pytest.fixture()
def three_block_dataset(tmp_path: Path) -> TrajectoryWindowDataset:
    """Three dataset directories, one per behaviour seed, over disjoint training draws.

    Twelve streams: 3 directories x 2 episodes x 2 intersections.  Disjoint draws keep every
    episode's arrays distinct, so no two streams are the same data under two labels.
    """
    first = write_dataset_dir(tmp_path, "fixture__policy__seed101", draws=(1, 2))
    second = write_dataset_dir(tmp_path, "fixture__policy__seed202", draws=(3, 4))
    third = write_dataset_dir(tmp_path, "fixture__policy__seed303", draws=(5, 6))
    return TrajectoryWindowDataset(
        [first, second, third], context_length=CONTEXT, split="train"
    )


def _canonical_key(stream: Any) -> tuple[str, str, str]:
    return (stream.dataset_dir, stream.episode_file, stream.ix_id)


# ----------------------------------------------------------------------
# The two selectors (BRIEF_13 section 4)
# ----------------------------------------------------------------------


def test_streams_from_datasets_returns_exactly_the_named_directories_streams(
    three_block_dataset: TrajectoryWindowDataset,
) -> None:
    """T1.  The membership is recomputed from the streams' own directories, not from the call.

    Killed by: returning every stream regardless of the requested directories.
    """
    everything = stream_returns(three_block_dataset)
    directories = sorted({s.dataset_dir for s in everything})
    assert len(directories) == 3, "the fixture must carry three directories to discriminate"

    wanted = [d for d in directories if d.endswith("seed101") or d.endswith("seed202")]
    selected = streams_from_datasets(three_block_dataset, wanted)

    expected = {s.key for s in everything if s.dataset_dir in set(wanted)}
    assert {s.key for s in selected} == expected
    assert len(selected) == 8                       # 2 directories x 2 episodes x 2 intersections
    assert len(selected) < len(everything), "a filter that keeps everything cannot fail this test"
    assert all(not s.dataset_dir.endswith("seed303") for s in selected)
    assert list(selected) == sorted(selected, key=_canonical_key)


def test_streams_from_datasets_names_both_sides_when_a_directory_yields_nothing(
    three_block_dataset: TrajectoryWindowDataset,
) -> None:
    """T1b.  A silent empty side would make an arm's pool smaller than it was declared to be."""
    with pytest.raises(ValueError, match="seed999.*no streams|no streams.*seed999"):
        streams_from_datasets(three_block_dataset, ["/nowhere/fixture__policy__seed999"])


def test_the_random_subset_is_deterministic_for_a_seed_and_differs_for_another(
    three_block_dataset: TrajectoryWindowDataset,
) -> None:
    """T2.  Killed by: seeding the generator from the clock instead of taking the caller's."""
    pool = stream_returns(three_block_dataset)
    first = random_stream_subset(pool, 4, np.random.default_rng(101))
    again = random_stream_subset(pool, 4, np.random.default_rng(101))
    other = random_stream_subset(pool, 4, np.random.default_rng(202))

    assert [s.key for s in first] == [s.key for s in again]
    assert [s.key for s in first] != [s.key for s in other], (
        "two different generator seeds drew the same subset: this fixture cannot detect a "
        "constant selector"
    )


def test_the_random_subset_matches_an_independently_recomputed_draw(
    three_block_dataset: TrajectoryWindowDataset,
) -> None:
    """T2b.  The declared contract, recomputed by the reader's own route.

    ``random_stream_subset`` documents its draw as ``rng.choice(n, size=count, replace=False)``
    over the CANONICALLY SORTED pool.  That contract is what makes a recorded rng seed enough to
    regenerate a subset years later, so it is pinned here by recomputing the draw rather than by
    calling the function twice.
    """
    pool = stream_returns(three_block_dataset)
    canonical = sorted(pool, key=_canonical_key)
    positions = np.random.default_rng(303).choice(len(canonical), size=5, replace=False)
    expected = sorted((canonical[int(p)] for p in positions), key=_canonical_key)

    subset = random_stream_subset(pool, 5, np.random.default_rng(303))
    assert [s.key for s in subset] == [s.key for s in expected]


def test_the_random_subset_is_canonically_ordered_and_independent_of_input_order(
    three_block_dataset: TrajectoryWindowDataset,
) -> None:
    """T3.  Killed by: returning the streams in draw order.

    Two properties, and both are needed.  Input-order invariance alone would survive a selector
    that canonicalises first and then returns the draw unsorted; canonical output alone would
    survive one whose SELECTION depends on the caller's ordering.  The vacuity guard recomputes
    the documented draw and asserts it is not already sorted -- without it, "return in draw
    order" and "return in canonical order" would be the same tuple and the test would pass on a
    mutant.
    """
    pool = list(stream_returns(three_block_dataset))
    canonical = sorted(pool, key=_canonical_key)
    positions = list(np.random.default_rng(404).choice(len(canonical), size=6, replace=False))
    assert positions != sorted(positions), (
        "the documented draw came out already sorted, so draw order and canonical order "
        "coincide and this test could not kill the mutant; change the rng seed or the count"
    )

    shuffled = list(pool)
    np.random.default_rng(7).shuffle(shuffled)
    assert [s.key for s in shuffled] != [s.key for s in pool], "the shuffle must reorder the input"

    from_canonical = random_stream_subset(pool, 6, np.random.default_rng(404))
    from_shuffled = random_stream_subset(shuffled, 6, np.random.default_rng(404))

    assert [s.key for s in from_canonical] == [s.key for s in from_shuffled]
    assert list(from_canonical) == sorted(from_canonical, key=_canonical_key)


def test_sampling_more_streams_than_exist_raises_naming_both_numbers(
    three_block_dataset: TrajectoryWindowDataset,
) -> None:
    """T5.  Killed by: clamping the count to the pool size, which silently shrinks an arm."""
    pool = stream_returns(three_block_dataset)
    assert len(pool) == 12
    with pytest.raises(ValueError, match="13.*12|12.*13"):
        random_stream_subset(pool, 13, np.random.default_rng(0))
    with pytest.raises(ValueError, match="a subset needs at least one stream"):
        random_stream_subset(pool, 0, np.random.default_rng(0))


def test_filtering_to_a_random_subset_keeps_only_that_subsets_rows(
    three_block_dataset: TrajectoryWindowDataset,
) -> None:
    """T4, the load-bearing row test.  Killed by: an off-by-one in the row-index map.

    Every kept row is traced back through ``item_index`` to its own stream and must belong to the
    subset; and the row count must be exactly ``T x count``, which is what makes two matched-size
    arms comparable at all.  On this fixture ``T = 8``; the corpus-gated test in
    ``test_offline_baselines.py`` asserts the real ``360 x 20``.
    """
    alpha = [s for s in stream_returns(three_block_dataset) if s.ix_id == "ix_alpha"]
    assert len(alpha) == 6
    subset = random_stream_subset(alpha, 3, np.random.default_rng(505))

    stacked = stack_dataset(three_block_dataset, group=ALPHA_GROUP)
    assert int(stacked["state"].shape[0]) == FIXTURE_T * len(alpha), "rows must exist to drop"

    filtered = filter_stacked_to_streams(three_block_dataset, stacked, subset)
    assert int(filtered["state"].shape[0]) == FIXTURE_T * len(subset)

    kept_keys = {s.key for s in subset}
    for row in range(int(filtered["item_index"].shape[0])):
        meta = three_block_dataset.item_meta(int(filtered["item_index"][row]))
        assert (meta.dataset_dir, meta.episode_file, meta.ix_id) in kept_keys


# ----------------------------------------------------------------------
# The arm declaration (plan section 4.1 item 3; BRIEF_13 section 10.6)
# ----------------------------------------------------------------------


def test_the_declared_arms_are_the_two_best_and_two_worst_by_held_out_att() -> None:
    """T13.  The declaration is checked against the DATA it claims to describe.

    BRIEF_13 section 10.7 names 505 and 303 as the two worst checkpoints and warns that it is
    NOT 303+404 -- a mistake that is invisible in a shell command and fatal to the ordering
    prediction.  The per-seed held-out ATT is recomputed here from the committed P4.4 episode
    records, so the declaration cannot drift from the measurement.

    Killed by: declaring 303+404 as the worst two.
    """
    assert set(SELECTION_ARMS) == {
        "bc_best2_20",
        "bc_any_20",
        "bc_worst2_20",
        "bc_best2_all",
    }

    episodes = json.loads(
        (REPO_ROOT / "docs/data/p4_4_baselines.json").read_text(encoding="utf-8")
    )["episodes"]
    by_seed: dict[int, list[float]] = {}
    for episode in episodes:
        if episode["arm"] == "mappo1000" and episode["seed"] is not None:
            by_seed.setdefault(int(episode["seed"]), []).append(float(episode["att_horizon"]))
    ranked = sorted(by_seed, key=lambda seed: float(np.mean(by_seed[seed])))
    assert len(ranked) == 5

    assert SELECTION_ARMS["bc_best2_20"].behaviour_seeds == tuple(sorted(ranked[:2]))
    assert SELECTION_ARMS["bc_worst2_20"].behaviour_seeds == tuple(sorted(ranked[-2:]))
    assert SELECTION_ARMS["bc_best2_all"].behaviour_seeds == tuple(sorted(ranked[:2]))
    assert SELECTION_ARMS["bc_any_20"].behaviour_seeds == ()

    matched = [name for name, spec in SELECTION_ARMS.items() if spec.count is not None]
    assert sorted(matched) == ["bc_any_20", "bc_best2_20", "bc_worst2_20"]
    assert {SELECTION_ARMS[name].count for name in matched} == {MATCHED_SUBSET_COUNT}
    assert SELECTION_ARMS["bc_best2_all"].count is None


def test_the_declared_behaviour_att_equals_the_committed_measurement() -> None:
    """The behaviour axis is COPIED from the merged artifacts, never re-measured here.

    Two committed sources, two exact comparisons.  The per-seed values must equal
    ``p4_4_training.json``'s composition block field for field; and the mixture derived from
    them must equal ``p4_4_baselines.json``'s 500-episode ``mappo1000`` cell mean under ``==``,
    which is the second route: five per-seed means against one 500-episode mean, equal only
    because the design is balanced.

    Killed by: retyping any of the five constants.
    """
    training = json.loads(
        (REPO_ROOT / "docs/data/p4_4_training.json").read_text(encoding="utf-8")
    )
    committed = training["top_return_filter"]["composition"]["per_seed_heldout_att"]
    assert {int(k): v for k, v in committed.items()} == dict(
        baselines_module.BEHAVIOUR_HELDOUT_ATT
    )

    cell = json.loads(
        (REPO_ROOT / "docs/data/p4_4_baselines.json").read_text(encoding="utf-8")
    )["cells"]["mappo1000"]["att_horizon_mean"]
    assert baselines_module._BEHAVIOUR_MIXTURE_ATT == cell

    assert baselines_module.BEHAVIOUR_GAP_TO_BEST_TWO == cell - (
        committed["101"] + committed["202"]
    ) / 2.0
    assert baselines_module.BEHAVIOUR_GAP_TO_BEST_SINGLE == cell - min(committed.values())


def test_cli_flags_that_disagree_with_the_declaration_are_refused() -> None:
    """T9.  A shell typo must not be able to redefine an arm.

    Killed by: accepting the flags and ignoring the declaration.
    """
    spec = SELECTION_ARMS["bc_best2_20"]
    accepted = arm_spec_for_flags(
        "bc_best2_20",
        selector=spec.selector,
        behaviour_seeds=spec.behaviour_seeds,
        count=spec.count,
    )
    assert accepted == spec

    with pytest.raises(ValueError, match="the declared behaviour seeds are"):
        arm_spec_for_flags(
            "bc_best2_20", selector=spec.selector, behaviour_seeds=(303, 505), count=spec.count
        )
    with pytest.raises(ValueError, match="a matched-size design is the whole experiment"):
        arm_spec_for_flags(
            "bc_best2_20", selector=spec.selector, behaviour_seeds=spec.behaviour_seeds, count=19
        )
    with pytest.raises(ValueError, match="the declared selector is"):
        arm_spec_for_flags(
            "bc_best2_20",
            selector="top_return",
            behaviour_seeds=spec.behaviour_seeds,
            count=spec.count,
        )
    with pytest.raises(ValueError, match="unknown arm"):
        arm_spec_for_flags("bc_middle2_20", selector="random_subset", behaviour_seeds=(), count=20)


def test_selecting_an_arms_streams_uses_only_its_declared_seeds(
    three_block_dataset: TrajectoryWindowDataset,
) -> None:
    """An arm restricted to two behaviour seeds must draw from those two and no other."""
    directories = sorted({s.dataset_dir for s in stream_returns(three_block_dataset)})
    spec = ArmSpec(
        arm="fixture_best2_4",
        selector="random_subset",
        behaviour_seeds=(101, 202),
        count=4,
        role="a fixture arm, declared here rather than in the module",
    )
    selected = select_arm_streams(
        three_block_dataset, spec, dataset_dirs=directories, rng=np.random.default_rng(11)
    )
    assert len(selected) == 4
    assert {baselines_module._behaviour_seed(s) for s in selected} <= {101, 202}

    whole_pool = ArmSpec(
        arm="fixture_best2_all",
        selector="datasets",
        behaviour_seeds=(101, 202),
        count=None,
        role="every stream of the two directories",
    )
    everything = select_arm_streams(
        three_block_dataset, whole_pool, dataset_dirs=directories, rng=np.random.default_rng(11)
    )
    assert len(everything) == 8
    assert {baselines_module._behaviour_seed(s) for s in everything} == {101, 202}


# ----------------------------------------------------------------------
# The design invariants -- the same function the artifact writer calls
# ----------------------------------------------------------------------


def _arm_block(arm: str, *, digest: str = "stats-digest-shared") -> dict[str, Any]:
    """One arm's block of a selection artifact, built to match the REAL declaration.

    Rewritten 2026-08-12 (review finding N1).  The first version used three-stream arms, which
    only worked because the validator took the size from the block it was checking; now that the
    size comes from :data:`SELECTION_ARMS`, a fixture has to be the shape the declaration says --
    20 streams and ``360 x 20`` rows -- which is a better fixture in any case.
    """
    spec = SELECTION_ARMS[arm]
    seeds = list(spec.behaviour_seeds) or [101, 202, 303, 404, 505]
    count = spec.count if spec.count is not None else 40 * len(seeds)
    per_seed: dict[str, Any] = {}
    for training_seed in TRAINING_SEEDS:
        streams = [
            {
                "dataset_dir": f"/corpus/cf_hz1x1__mappo1000__seed{seeds[index % len(seeds)]}",
                "episode_file": f"ep{index:06d}_seed1000_draw{1 + index}.npz",
                "ix_id": "ix0",
                "flow_draw": 1 + index,
                "total_return": -6000.0 - index,
                "behaviour_seed": seeds[index % len(seeds)],
            }
            for index in range(count)
        ]
        composition: dict[str, int] = {}
        for stream in streams:
            key = str(stream["behaviour_seed"])
            composition[key] = composition.get(key, 0) + 1
        per_seed[str(training_seed)] = {
            "rng_seed": training_seed,
            "streams": streams,
            "per_behaviour_seed_composition": composition,
            "training_rows": 360 * count,
        }
    return {
        "selector": spec.selector,
        "selector_parameters": {"behaviour_seeds": seeds, "count": spec.count},
        "declared_count": count,
        "normalisation_digest": digest,
        "per_training_seed": per_seed,
    }


def _design_payload() -> dict[str, Any]:
    """A selection artifact that satisfies every invariant, shaped like the real one."""
    return {
        "held_out_draws": list(HELD_OUT_DRAWS),
        "arms": {arm: _arm_block(arm) for arm in ("bc_best2_20", "bc_any_20", "bc_worst2_20")},
        "matched_arms": ["bc_any_20", "bc_best2_20", "bc_worst2_20"],
    }


def test_the_design_check_accepts_a_sound_payload() -> None:
    """The invariants must be satisfiable, or the refusal tests prove nothing."""
    assert_selection_design(_design_payload())


# -- N1: every reference comes from the DECLARATION, none from the payload ---------------


def test_the_leakage_check_uses_the_registered_pool_and_not_the_payloads_field() -> None:
    """N1, the reviewer's exact probe: empty the pool field, plant a real leak.

    ``flow_draw = 1042`` is inside the registered held-out pool.  The first version of this
    validator read the pool from ``payload["held_out_draws"]``, so emptying that field made the
    leak invisible and the suite stayed green.

    The second half is the one that proves the reference moved: with the field **absent
    entirely** the leak must still be caught, because the pool now comes from ``HELD_OUT_DRAWS``.

    Killed by: reading the pool from the payload.
    """
    payload = _design_payload()
    payload["held_out_draws"] = []
    payload["arms"]["bc_any_20"]["per_training_seed"]["101"]["streams"][0]["flow_draw"] = 1042
    with pytest.raises(ValueError, match="not the registered pool"):
        assert_selection_design(payload)

    absent = _design_payload()
    del absent["held_out_draws"]
    absent["arms"]["bc_any_20"]["per_training_seed"]["101"]["streams"][0]["flow_draw"] = 1042
    with pytest.raises(ValueError, match="held-out draws"):
        assert_selection_design(absent)


def test_the_size_check_uses_the_declared_count_and_not_the_payloads() -> None:
    """N1 second instance: every arm declaring 19 was accepted, because 19 was its own reference.

    Killed by: reading ``declared_count`` from the block being validated.
    """
    payload = _design_payload()
    for arm, block in payload["arms"].items():
        block["declared_count"] = 19
        for entry in block["per_training_seed"].values():
            entry["streams"] = entry["streams"][:19]
            composition: dict[str, int] = {}
            for stream in entry["streams"]:
                key = str(stream["behaviour_seed"])
                composition[key] = composition.get(key, 0) + 1
            entry["per_behaviour_seed_composition"] = composition
            entry["training_rows"] = 360 * 19
    with pytest.raises(ValueError, match="the arm table declares 20"):
        assert_selection_design(payload)


def test_the_seed_check_uses_the_declared_training_seeds_and_not_the_first_arms() -> None:
    """N1 third instance: a UNIFORMLY wrong seed set passed, because arm one defined the truth.

    Killed by: taking the reference seed set from the first arm's own record.
    """
    payload = _design_payload()
    for block in payload["arms"].values():
        block["per_training_seed"] = {"999": block["per_training_seed"]["101"]}
    with pytest.raises(ValueError, match="the declared seeds are"):
        assert_selection_design(payload)


def test_the_row_count_must_equal_the_stream_count_times_the_episode_length() -> None:
    """N1's missing invariant: nothing tied ``training_rows`` to the streams it claims to cover.

    Killed by: dropping the ``rows == 360 x len(streams)`` check.
    """
    payload = _design_payload()
    payload["arms"]["bc_best2_20"]["per_training_seed"]["101"]["training_rows"] = 7199
    with pytest.raises(ValueError, match="360 x 20"):
        assert_selection_design(payload)


def test_an_arm_the_declaration_does_not_know_is_refused() -> None:
    """An undeclared arm has no reference to be checked against, so it cannot be validated."""
    payload = _design_payload()
    payload["arms"]["bc_middle2_20"] = payload["arms"]["bc_best2_20"]
    with pytest.raises(ValueError, match="not declared in the arm table"):
        assert_selection_design(payload)


def test_a_stream_from_an_undeclared_checkpoint_is_refused() -> None:
    """An arm labelled "the two best" whose data came from elsewhere would be a false label."""
    payload = _design_payload()
    entry = payload["arms"]["bc_best2_20"]["per_training_seed"]["101"]
    entry["streams"][0]["behaviour_seed"] = 303
    entry["per_behaviour_seed_composition"] = {"101": 9, "202": 10, "303": 1}
    with pytest.raises(ValueError, match="the arm table does not declare"):
        assert_selection_design(payload)


def test_the_recorded_composition_must_recompute_from_its_own_stream_list() -> None:
    """A composition that does not match its streams is a record of a different draw."""
    payload = _design_payload()
    entry = payload["arms"]["bc_best2_20"]["per_training_seed"]["202"]
    entry["per_behaviour_seed_composition"] = {"101": 20}
    with pytest.raises(ValueError, match="not the one its own stream list implies"):
        assert_selection_design(payload)


def test_the_payloads_matched_arm_set_is_cross_checked_against_the_declaration() -> None:
    """Which arms the size invariant protects is a declaration, not the payload's to choose."""
    payload = _design_payload()
    payload["matched_arms"] = ["bc_best2_20"]
    with pytest.raises(ValueError, match="not the payload's to choose"):
        assert_selection_design(payload)


def test_the_design_check_refuses_a_leaked_selection() -> None:
    """T6.  A training stream drawn from the held-out pool voids every number in the task.

    Killed by: dropping the leakage check.
    """
    payload = _design_payload()
    leaked = payload["arms"]["bc_any_20"]["per_training_seed"]["101"]["streams"][0]
    leaked["flow_draw"] = 1000
    with pytest.raises(ValueError, match="training streams drawn from held-out draws"):
        assert_selection_design(payload)


def test_the_design_check_refuses_unequal_matched_arm_sizes() -> None:
    """T7.  THE ONE THAT PROTECTS THE CONCLUSION.

    Every other test can pass while two matched arms differ in size, and then the decisive
    comparison measures data quantity while claiming to measure seed identity.

    ⚠️ **Restructured 2026-08-12, and the reason matters.**  Under P4.5's own declaration this
    invariant is UNREACHABLE: all three matched arms declare 20, the size check pins each arm's
    stream count to its declared 20, and the row check pins rows to ``360 x 20`` -- so equal rows
    follow, and the mutation that disables this check SURVIVED against the old fixture.  It is
    reachable exactly when a declaration carries matched arms of DIFFERENT sizes, which is the
    shape P4.6 will have, and that is what this test now builds.

    Killed by: dropping the equal-rows check across matched arms.
    """
    declaration = {
        "arm_twenty": ArmSpec("arm_twenty", "random_subset", (101, 202), 20, "matched, 20"),
        "arm_nineteen": ArmSpec("arm_nineteen", "random_subset", (101, 202), 19, "matched, 19"),
    }
    payload = {
        "held_out_draws": list(HELD_OUT_DRAWS),
        "arms": {
            "arm_twenty": _arm_block("bc_best2_20"),
            "arm_nineteen": _arm_block("bc_best2_20"),
        },
        "matched_arms": ["arm_nineteen", "arm_twenty"],
    }
    # Make the second arm internally consistent at 19 streams: 19 rows-worth, 19 in composition.
    for entry in payload["arms"]["arm_nineteen"]["per_training_seed"].values():
        entry["streams"] = entry["streams"][:19]
        composition: dict[str, int] = {}
        for stream in entry["streams"]:
            key = str(stream["behaviour_seed"])
            composition[key] = composition.get(key, 0) + 1
        entry["per_behaviour_seed_composition"] = composition
        entry["training_rows"] = 360 * 19
    payload["arms"]["arm_nineteen"]["declared_count"] = 19

    with pytest.raises(ValueError, match="do not have equal training rows"):
        assert_selection_design(payload, declaration=declaration)


def test_an_arm_with_fewer_streams_than_declared_is_refused() -> None:
    """The size check, on the path P4.5's own declaration does reach."""
    payload = _design_payload()
    payload["arms"]["bc_worst2_20"]["per_training_seed"]["101"]["streams"].pop()
    with pytest.raises(ValueError, match="against a declared count of 20"):
        assert_selection_design(payload)


def test_the_design_check_refuses_a_missing_per_seed_subset_record() -> None:
    """T8.  The subset draw must be auditable rather than asserted.

    Killed by: dropping the per-training-seed record.
    """
    payload = _design_payload()
    del payload["arms"]["bc_best2_20"]["per_training_seed"]["202"]
    with pytest.raises(ValueError, match="per training seed|training seeds"):
        assert_selection_design(payload)

    other = _design_payload()
    del other["arms"]["bc_best2_20"]["per_training_seed"]["101"][
        "per_behaviour_seed_composition"
    ]
    with pytest.raises(ValueError, match="no per-behaviour-seed composition"):
        assert_selection_design(other)


def test_the_design_check_refuses_arms_with_different_normalisation_statistics() -> None:
    """T11.  Refitting the statistics per arm would make the arms incomparable silently.

    Killed by: dropping the shared-statistics check.
    """
    payload = _design_payload()
    payload["arms"]["bc_worst2_20"]["normalisation_digest"] = "stats-digest-refitted"
    with pytest.raises(ValueError, match="different normalisation digests"):
        assert_selection_design(payload)


# ----------------------------------------------------------------------
# The thread regime (BRIEF_13 section 11.1)
# ----------------------------------------------------------------------


def test_the_thread_regime_is_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """T12.  Killed by: reading the environment once at import and caching it.

    ``p4_4_training.json`` records ``torch_num_threads = 1`` beside 15 timings and neither env
    var, and those are a different knob -- the one that fixed this task's suite hang.  A cached
    block would record the regime of the interpreter's startup rather than of the run.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("MKL_NUM_THREADS", "5")
    first = thread_regime()
    assert first["OMP_NUM_THREADS"] == "3"
    assert first["MKL_NUM_THREADS"] == "5"
    assert first["torch_get_num_threads"] == int(torch.get_num_threads())

    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    second = thread_regime()
    assert second["OMP_NUM_THREADS"] == "7"
    assert second["MKL_NUM_THREADS"] is None


# ----------------------------------------------------------------------
# The neutral verdict and the artifact
# ----------------------------------------------------------------------


def test_the_neutral_delta_verdict_agrees_with_the_registered_one() -> None:
    """T10.  The double-compute: same arithmetic, neutral names, checked over a grid.

    Writing ``dt_genuinely_better`` into an artifact describing a BC-versus-BC pair would be a
    false label on disk -- the defect class review finding F1 was raised for.  The names differ;
    the decision must not.

    Killed by: flipping one branch of the neutral verdict.
    """
    checked = 0
    for mean in (-2.0, -0.7, -0.6263, -0.3, 0.0, 0.3, 0.6263, 0.7, 2.0):
        for half in (0.0, 0.05, 0.3, 0.9):
            expected = VERDICT_NAME_MAP[
                baselines_module.equivalence_verdict(mean, half, DELTA_ATT)
            ]
            assert delta_verdict(mean, half, DELTA_ATT) == expected, (mean, half)
            checked += 1
    assert checked == 36
    assert len(set(VERDICT_NAME_MAP.values())) == 4, "the map must not collapse two branches"


def _episodes(arm: str, means: dict[int, float], draws: Sequence[int]) -> list[EpisodeResult]:
    """One arm's episodes: a per-seed constant offset so the paired maths is hand-checkable."""
    out: list[EpisodeResult] = []
    for seed, offset in means.items():
        for draw in draws:
            out.append(
                EpisodeResult(
                    arm=arm,
                    seed=seed,
                    draw_id=draw,
                    att_horizon=offset + 0.1 * (draw - draws[0]),
                    horizon_vehicle_count=40.0,
                    episode_reward=-100.0,
                )
            )
    return out


def _artifact_inputs() -> dict[str, Any]:
    draws = (1000, 1001, 1002, 1003)
    episodes = [
        *_episodes("bc_top10", {101: 103.0, 202: 103.4}, draws),
        *_episodes("bc_best2_20", {101: 103.1, 202: 103.5}, draws),
        *_episodes("bc_any_20", {101: 104.1, 202: 104.5}, draws),
        *_episodes("bc_worst2_20", {101: 105.1, 202: 105.5}, draws),
        *_episodes("bc_best2_all", {101: 103.0, 202: 103.6}, draws),
    ]
    return {
        "episodes": episodes,
        "selection": _design_payload(),
        "gate_b": {"status": "PASS", "compared": 25, "mismatches": []},
        "env_settings": {"max_steps": 360},
        "engine_seed": 1000,
    }


def test_the_artifact_reports_every_pair_and_scores_the_three_predictions() -> None:
    """T14.  Ten unordered pairs, reported unconditionally, and the registered predictions scored.

    The predictions are scored by the rules fixed in ``docs/plans/p4.5.md`` section 2.1 BEFORE
    the run: the forecast on the point estimate, the equivalence claim on the whole CI, the
    ordering on the cell means.  This fixture is built so the ordering holds and the primary is
    inside delta, which is what lets the test assert the scoring rather than the data.

    Killed by: scoring the ordering on a SINGLE adjacent contrast instead of the three cell means.
    ⚠️ Narrowed 2026-08-12 (review finding N6): this line claimed the test killed "anything but
    the three cell means", and a scoring that reads both adjacent contrasts survives on this
    fixture, where the ordering and the contrasts agree. The claim now names what was executed.
    """
    artifact = selection_artifact(**_artifact_inputs())

    assert len(artifact["comparisons"]) == 10
    assert "bc_top10_vs_bc_best2_20" in artifact["comparisons"]
    for name, entry in artifact["comparisons"].items():
        for field in (
            "mean_difference",
            "ci95_low",
            "ci95_high",
            "ci95_width",
            "rank_biserial",
            "delta_verdict",
        ):
            assert field in entry, f"{name} is missing {field}, which section 3 makes unconditional"

    predictions = artifact["registered_predictions"]
    assert predictions["registered_in"].startswith("docs/plans/p4.5.md")

    primary = predictions["primary_bc_best2_20_within_delta_of_bc_top10"]
    assert primary["scored_by"] == "abs(mean_paired_difference) <= delta"
    assert primary["held"] is True
    assert "equivalence_demonstrated_at_this_power" in primary

    ordering = predictions["ordering_best2_then_any_then_worst2"]
    assert ordering["held"] is True
    assert ordering["cell_means"] == [
        artifact["cells"]["bc_best2_20"]["att_horizon_mean"],
        artifact["cells"]["bc_any_20"]["att_horizon_mean"],
        artifact["cells"]["bc_worst2_20"]["att_horizon_mean"],
    ]

    secondary = predictions["secondary_bc_any_20_worse_than_bc_best2_20"]
    assert secondary["scored_by"].startswith("directional")
    assert secondary["mean_difference_any_minus_best2"] > 0.0


def test_the_null_bound_is_the_largest_effect_the_data_leave_standing() -> None:
    """T14b.  Section 2.2's X, recomputed by a second route.

    A null on the decisive contrast may only be written as "no effect larger than +/-X found,
    against a 3.8190 ATT spread in the behaviour policies themselves".  X is
    ``max(|ci_low|, |ci_high|)`` of that contrast -- the largest effect the data leave standing,
    not the half-width around a convenient centre.

    Killed by: reporting the CI half-width as X.
    """
    artifact = selection_artifact(**_artifact_inputs())
    bound = artifact["null_bound"]
    contrast = artifact["comparisons"]["bc_best2_20_vs_bc_any_20"]

    assert bound["contrast"] == "bc_best2_20_vs_bc_any_20"
    assert bound["x"] == max(abs(contrast["ci95_low"]), abs(contrast["ci95_high"]))
    assert bound["x"] != contrast["ci95_half_width"], (
        "this fixture must separate the two definitions, or the mutation survives"
    )
    assert bound["behaviour_spread_att"] == pytest.approx(3.819007598325129, abs=1e-12)
    assert bound["power_contrast"] == "bc_best2_20_vs_bc_worst2_20"
    assert bound["power_contrast_ci95_width"] == artifact["comparisons"][
        "bc_best2_20_vs_bc_worst2_20"
    ]["ci95_width"]


def test_the_artifact_states_the_reuse_and_carries_no_dt_comparison() -> None:
    """bc_top10's episodes are re-used rather than re-rolled, and that must be legible on disk.

    No DT-versus-baseline sentence may enter this task at all (docs/reviews/P4.4.md section 8.6),
    so the artifact must carry neither a ``madt`` cell nor a comparison against one.

    ⚠️ Strengthened 2026-08-12 (review finding N7): the first version asserted "no madt cell" on a
    fixture that contained only BC arms, so **that half could not fail**.  A DT arm is now
    REFUSED by the artifact builder, and the second half of this test feeds it one.

    Killed by: dropping the refusal and letting a foreign arm through silently.
    """
    artifact = selection_artifact(**_artifact_inputs())
    assert "madt" not in artifact["cells"]
    assert not [name for name in artifact["comparisons"] if "madt" in name]
    assert artifact["reused_arm"]["arm"] == "bc_top10"
    assert "re-used" in artifact["reused_arm"]["statement"]
    assert artifact["gate_b"]["status"] == "PASS"

    with_dt = _artifact_inputs()
    with_dt["episodes"] = [
        *with_dt["episodes"],
        *_episodes("madt", {101: 104.9}, (1000, 1001, 1002, 1003)),
    ]
    with pytest.raises(ValueError, match="no DT arm"):
        selection_artifact(**with_dt)


# ----------------------------------------------------------------------
# Gate B -- re-use is only sound if this session's instrument is P4.4's
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# The committed artifacts.  Written AFTER the campaign, which is why the
# invariant tests above are the red-first ones: these assert facts about data
# that did not exist when the invariants were written.
# ----------------------------------------------------------------------


def _committed(name: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / "docs/data" / name).read_text(encoding="utf-8"))


def test_the_committed_selection_satisfies_every_design_invariant() -> None:
    """T6, T7, T8 and T11 against the artifact that was actually reported.

    The invariants are exercised on fixtures above, where each one is killed by its own
    mutation; this runs the same function over the real thing, so a hand-edited artifact cannot
    pass review.
    """
    selection = _committed("p4_5_selection.json")
    assert_selection_design(selection)
    assert sorted(selection["arms"]) == sorted(SELECTION_ARMS)
    assert len(selection["runs"]) == 20


def test_the_committed_matched_arms_trained_on_exactly_the_same_number_of_rows() -> None:
    """T7 on the reported artifact: the decisive comparison is size-matched or it is nothing.

    360 rows per stream x 20 streams = 7,200, for all three matched arms and all five training
    seeds -- recomputed here from the stream lists rather than read from ``training_rows``.
    """
    selection = _committed("p4_5_selection.json")
    matched = sorted(selection["matched_arms"])
    assert matched == ["bc_any_20", "bc_best2_20", "bc_worst2_20"]

    rows = set()
    for arm in matched:
        block = selection["arms"][arm]
        assert block["declared_count"] == MATCHED_SUBSET_COUNT
        for seed, entry in sorted(block["per_training_seed"].items()):
            assert len(entry["streams"]) == MATCHED_SUBSET_COUNT, (arm, seed)
            assert entry["training_rows"] == 360 * MATCHED_SUBSET_COUNT, (arm, seed)
            rows.add(entry["training_rows"])
    assert rows == {7200}

    unmatched = selection["arms"]["bc_best2_all"]
    assert unmatched["declared_count"] == 80
    assert {e["training_rows"] for e in unmatched["per_training_seed"].values()} == {28800}


def test_the_committed_subsets_came_only_from_their_declared_checkpoints() -> None:
    """T8 on the reported artifact: composition is recorded per training seed and is honest.

    Each stream's ``behaviour_seed`` is checked against the arm's declaration, so an arm whose
    label says "the two best" and whose data came from elsewhere cannot pass.
    """
    selection = _committed("p4_5_selection.json")
    for arm, spec in sorted(SELECTION_ARMS.items()):
        block = selection["arms"][arm]
        assert sorted(block["per_training_seed"]) == sorted(str(s) for s in (101, 202, 303, 404, 505))
        for seed, entry in sorted(block["per_training_seed"].items()):
            seen = {int(s["behaviour_seed"]) for s in entry["streams"]}
            if spec.behaviour_seeds:
                assert seen <= set(spec.behaviour_seeds), (arm, seed, seen)
            composition = {int(k): int(v) for k, v in entry["per_behaviour_seed_composition"].items()}
            assert sum(composition.values()) == block["declared_count"]
            recomputed: dict[int, int] = {}
            for stream in entry["streams"]:
                key = int(stream["behaviour_seed"])
                recomputed[key] = recomputed.get(key, 0) + 1
            assert recomputed == composition, (arm, seed)


def test_the_committed_result_reuses_bc_top10_unchanged() -> None:
    """The re-used records must be the committed ones, field for field, not merely in mean."""
    p44 = _committed("p4_4_baselines.json")["episodes"]
    p45 = _committed("p4_5_baselines.json")["episodes"]
    key = lambda e: (e["seed"], e["draw_id"])
    source = sorted([e for e in p44 if e["arm"] == "bc_top10"], key=key)
    reused = sorted([e for e in p45 if e["arm"] == "bc_top10"], key=key)
    assert len(source) == 500
    assert reused == source

    artifact = _committed("p4_5_baselines.json")
    assert len(artifact["episodes"]) == 2500
    assert artifact["cells"]["bc_top10"]["att_horizon_mean"] == _committed(
        "p4_4_baselines.json"
    )["cells"]["bc_top10"]["att_horizon_mean"]
    assert artifact["gate_b"]["status"] == "PASS"
    assert artifact["gate_b"]["compared"] == 25


def test_the_committed_result_carries_every_timing_with_its_thread_regime() -> None:
    """BRIEF_13 section 11.1: a timing without its thread regime is not reproducible."""
    selection = _committed("p4_5_selection.json")
    for run in selection["runs"]:
        assert "seconds" in run
        regime = run["thread_regime"]
        assert regime["OMP_NUM_THREADS"] == "1"
        assert regime["MKL_NUM_THREADS"] == "1"
        assert regime["torch_get_num_threads"] == 1


def test_a_reused_arm_must_reproduce_exactly_or_the_gate_refuses() -> None:
    """T15.  Exact equality, never a tolerance: a 1e-12 drift means a different instrument.

    Killed by: comparing with a tolerance, or comparing only the mean.
    """
    committed = _episodes("bc_top10", {101: 103.0}, (1000, 1025))
    record = assert_reused_arm_reproduces(committed, list(committed))
    assert record["compared"] == 2
    assert record["mismatches"] == []
    assert record["status"] == "PASS"

    drifted = [
        EpisodeResult(
            arm=e.arm,
            seed=e.seed,
            draw_id=e.draw_id,
            att_horizon=e.att_horizon + (1e-12 if e.draw_id == 1025 else 0.0),
            horizon_vehicle_count=e.horizon_vehicle_count,
            episode_reward=e.episode_reward,
        )
        for e in committed
    ]
    with pytest.raises(ValueError, match="does not reproduce|1025"):
        assert_reused_arm_reproduces(committed, drifted)

    with pytest.raises(ValueError, match="were not re-rolled"):
        assert_reused_arm_reproduces(committed, committed[:1])


# ----------------------------------------------------------------------
# The CLI runners (review finding N3): they had zero coverage, and the
# property that makes the reported CIs mean what the packet says -- one
# subset per TRAINING SEED -- was unprotected.
# ----------------------------------------------------------------------


def _stub_train_bc(monkeypatch: pytest.MonkeyPatch, seen: list[dict[str, Any]]) -> None:
    """Replace training with a record of what it was handed.

    The runner under test selects streams, filters rows and writes the artifact; the 40,000
    gradient steps are not what N3 is about and would make this test a campaign.
    """

    def fake(stacked: dict[str, torch.Tensor], **kwargs: Any) -> Any:
        seen.append({"rows": int(stacked["state"].shape[0]), "seed": kwargs["seed"]})
        path = Path(kwargs["checkpoint_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")
        return baselines_module.TrainRecord(
            method=kwargs["method"],
            seed=int(kwargs["seed"]),
            gradient_steps=int(kwargs["declared_gradient_steps"]),
            declared_gradient_steps=int(kwargs["declared_gradient_steps"]),
            losses=(1.0, 0.5),
            window_means=(1.0, 0.5),
            plateaued=False,
            checkpoint_path=str(path),
            canonical_digest="0" * 64,
            file_sha256="1" * 64,
            seconds=0.1,
            diagnostics={},
        )

    monkeypatch.setattr(baselines_module, "train_bc", fake)


def test_the_training_runner_draws_a_subset_from_every_training_seeds_own_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N3: the property that makes the reported CIs cover SUBSET variance as well as training.

    The reviewer mutated ``np.random.default_rng(int(seed))`` to ``default_rng(101)`` -- one
    subset shared by all five training seeds -- and the whole suite stayed green.  The packet
    claims the five-seed spread averages over five subsets; without this test that claim rests on
    reading the source.

    The expected subsets are recomputed here through the documented rng contract, so the check is
    on the actual draws and not merely on "they differ".

    Killed by: drawing every training seed's subset from one generator.
    """
    # A SINGLE-INTERSECTION fixture corpus, which is the shape production always has: the runner
    # calls stack_dataset(dataset) with no group, and that refuses a multi-group dataset outright.
    # Building it this way removes the seam a two-group fixture would need -- and the first draft,
    # which used one, was caught by this task's own new rows == T x streams invariant, because a
    # subset could contain a stream absent from the single stacked group.
    monkeypatch.setattr(dataset_fixtures, "IX_SPECS", (("ix_alpha", 4, 3),))
    monkeypatch.setattr(dataset_fixtures, "IX_IDS", ("ix_alpha",))
    first = write_dataset_dir(tmp_path, "fixture__policy__seed101", draws=(1, 2))
    second = write_dataset_dir(tmp_path, "fixture__policy__seed202", draws=(3, 4))
    third = write_dataset_dir(tmp_path, "fixture__policy__seed303", draws=(5, 6))
    dirs = [str(first), str(second), str(third)]

    spec = ArmSpec(
        arm="fixture_arm",
        selector="random_subset",
        behaviour_seeds=(101, 202),
        count=2,
        role="a fixture arm sized to the fixture corpus",
    )
    monkeypatch.setattr(baselines_module, "SELECTION_ARMS", {"fixture_arm": spec})
    monkeypatch.setattr(baselines_module, "DECISION_ROWS_PER_STREAM", FIXTURE_T)
    monkeypatch.setattr(baselines_module, "STREAMS_PER_BEHAVIOUR_SEED", 2)
    monkeypatch.setattr(dt_gate_module, "CONTEXT_LENGTH", CONTEXT)
    seen: list[dict[str, Any]] = []
    _stub_train_bc(monkeypatch, seen)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    args = argparse.Namespace(
        selection_arm="fixture_arm",
        stream_selector="random_subset",
        selector_seed=[101, 202],
        subset_count=2,
        methods=",".join(baselines_module.METHODS),
        dataset_dir=dirs,
        steps=2,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_every=0,
    )
    assert baselines_module._run_train_selection(args, out_dir) == 0

    payload = json.loads((out_dir / "p4_5_selection.json").read_text(encoding="utf-8"))
    block = payload["arms"]["fixture_arm"]
    assert len(seen) == len(TRAINING_SEEDS)
    assert {row["rows"] for row in seen} == {FIXTURE_T * 2}

    dataset = TrajectoryWindowDataset(
        [Path(d) for d in dirs], context_length=CONTEXT, split="train"
    )
    distinct: set[tuple[str, ...]] = set()
    for training_seed in TRAINING_SEEDS:
        expected = select_arm_streams(
            dataset, spec, dataset_dirs=dirs, rng=np.random.default_rng(int(training_seed))
        )
        recorded = block["per_training_seed"][str(training_seed)]
        assert [s["episode_file"] for s in recorded["streams"]] == [
            s.episode_file for s in expected
        ], training_seed
        assert recorded["rng_seed"] == int(training_seed)
        distinct.add(tuple(s["episode_file"] for s in recorded["streams"]))
    assert len(distinct) > 1, (
        "every training seed drew the same subset, so this fixture cannot detect a shared "
        "generator; change the pool size or the count"
    )


def test_the_gate_refuses_weights_that_are_not_the_recorded_ones_before_any_rollout(
    tmp_path: Path,
) -> None:
    """N3: Gate B's weight identity must refuse, and must refuse BEFORE it rolls anything.

    A rollout on the wrong checkpoint would be a measurement of the wrong model; the digest check
    exists to make that unreachable.  ``config_for_draw`` raises if it is ever called, so the
    test also proves the ordering.

    Killed by: comparing the digest after the rollouts, or not at all.
    """
    checkpoint = tmp_path / "bc_top10_seed101.pt"
    torch.save({"model": {"policy.w": torch.zeros(2)}, "provenance": {"gradient_steps": 2}}, checkpoint)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "p4_4_training.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "method": "bc_top10",
                        "seed": 101,
                        "checkpoint": str(checkpoint),
                        "canonical_digest": "deadbeef" * 8,
                        "file_sha256": "0" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "p4_4_baselines.json").write_text(
        json.dumps({"episodes": []}), encoding="utf-8"
    )

    def never(draw_id: int) -> Path:
        raise AssertionError(f"a rollout was started for draw {draw_id} despite a wrong digest")

    args = argparse.Namespace(
        reused_arm="bc_top10",
        training=None,
        baselines=None,
        draw=[1000],
        steps=2,
        scenario_id="fixture",
        device="cpu",
        engine_seed=1000,
    )
    with pytest.raises(ValueError, match="canonical digest"):
        baselines_module._run_gate_selection(args, {}, never, out_dir, tmp_path / "work")
    assert not (tmp_path / "work").exists(), "a refused gate must create nothing"


def test_the_selection_report_refuses_a_gate_that_did_not_pass(tmp_path: Path) -> None:
    """N3: without Gate B, re-using bc_top10's episodes is a comparison across two instruments.

    Killed by: dropping the status check in ``_run_report_selection``.
    """
    work_dir = tmp_path / "work"
    out_dir = tmp_path / "out"
    work_dir.mkdir()
    out_dir.mkdir()
    (work_dir / "gate_b.json").write_text(
        json.dumps({"status": "FAIL", "compared": 25, "mismatches": [{"draw_id": 1000}]}),
        encoding="utf-8",
    )
    args = argparse.Namespace(baselines=None, engine_seed=1000)
    with pytest.raises(ValueError, match="Gate B did not pass"):
        baselines_module._run_report_selection(args, {}, out_dir, work_dir)
    assert not (out_dir / "p4_5_baselines.json").exists()


def test_the_selection_report_refuses_a_partial_arm_set(tmp_path: Path) -> None:
    """N3: three arms answer a different question than four, and the design is four.

    Killed by: reporting whichever arms happen to be present.
    """
    work_dir = tmp_path / "work"
    out_dir = tmp_path / "out"
    work_dir.mkdir()
    out_dir.mkdir()
    (work_dir / "gate_b.json").write_text(
        json.dumps({"status": "PASS", "compared": 25, "mismatches": []}), encoding="utf-8"
    )
    (out_dir / "p4_5_selection.json").write_text(
        json.dumps({"arms": {"bc_best2_20": {}, "bc_any_20": {}}}), encoding="utf-8"
    )
    args = argparse.Namespace(baselines=None, engine_seed=1000)
    with pytest.raises(ValueError, match="missing declared arm"):
        baselines_module._run_report_selection(args, {}, out_dir, work_dir)
    assert not (out_dir / "p4_5_baselines.json").exists()


def test_the_selection_artifact_refuses_a_verdict_decided_by_deltas_rounding() -> None:
    """N4: the packet's section 0.1 read as though this guard ran, and it did not exist.

    A guard that can only ever REFUSE to emit a verdict is safe to add after a result, because it
    cannot manufacture one.  The committed comparisons sit 0.01296 or further from the margin --
    13x the tolerance -- so it never fired on the reported artifact, which a second test asserts
    from the committed file.

    Killed by: removing the proximity check from ``selection_artifact``.
    """
    inputs = _artifact_inputs()
    # Every paired difference exactly -delta, so both CI endpoints land on the margin.
    draws = (1000, 1001, 1002, 1003)
    inputs["episodes"] = [
        *_episodes("bc_top10", {101: 103.0}, draws),
        *_episodes("bc_best2_20", {101: 103.0 + DELTA_ATT}, draws),
        *_episodes("bc_any_20", {101: 106.0}, draws),
        *_episodes("bc_worst2_20", {101: 108.0}, draws),
        *_episodes("bc_best2_all", {101: 110.0}, draws),
    ]
    with pytest.raises(ValueError, match="the P4.5 pair"):
        selection_artifact(**inputs)


def test_the_committed_comparisons_are_far_from_the_margin_so_the_guard_never_fired() -> None:
    """The other half of N4: the guard must not be silently load-bearing on a reported verdict."""
    artifact = _committed("p4_5_baselines.json")
    distances = []
    for name, entry in artifact["comparisons"].items():
        distance = min(
            abs(entry["ci95_low"] + DELTA_ATT),
            abs(entry["ci95_low"] - DELTA_ATT),
            abs(entry["ci95_high"] + DELTA_ATT),
            abs(entry["ci95_high"] - DELTA_ATT),
        )
        assert distance > baselines_module.DELTA_PROXIMITY_TOLERANCE, f"{name}: {distance}"
        assert entry["delta_verdict"] == entry["delta_verdict_at_full_precision_delta"], name
        assert entry["delta_verdict_turns_on_the_rounding"] is False, name
        distances.append(distance)
    assert min(distances) == pytest.approx(0.012963537457243324, abs=1e-12)
