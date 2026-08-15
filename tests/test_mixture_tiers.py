"""P4.7 -- the mixture tiers, verified on the REAL corpus before anything trains.

**Why this file exists at all.**  ``offline/method_tier_grid.py`` has declared ``mix33``, ``mix50``
and ``mix67`` since P4.6, and P4.6's independent reviewer listed that code under *what I could not
verify*: **declared and unit-tested, never executed** (``docs/reviews/P4.6.md`` section 8).  Its only
test was a twenty-stream synthetic fixture at 50 %.  ``BRIEF_19`` section 2 therefore requires the
mixture path to be exercised **end to end on the real corpus, with the realised composition asserted
from the BUILT STREAM LIST rather than from the spec**, before a single gradient step.

**The rule this file enforces on itself: nothing is asserted from the declaration.**  Every count,
every provenance claim and every return is read back off the objects the production code actually
produced, and the prompt target is recomputed from the raw ``.npz`` episodes by a route that never
touches :func:`offline.offline_baselines.stream_returns` -- ``load_episode`` plus ``math.fsum`` over
``local_reward``.  A convention error shared between the code and its test would otherwise be
invisible, and the prompt is the one quantity that has no other check behind it.

**Alignment convention** is contract C6's, unchanged: a stream's return is ``sum(local_reward)`` over
its ``T`` outcome rows, per (episode, intersection).  On ``cf_hz1x1`` each episode carries exactly one
intersection, so one episode is one stream.

Real-corpus tests skip with a reason naming ``RLTRAFFIC_CORPUS_V11``; the arithmetic ones always run.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import pytest

from offline.method_tier_grid import (
    MIXTURE_EXPERT_FRACTION,
    MIXTURE_RNG_BASE,
    MIXTURE_TIER_ORDER,
    TRAINING_STREAM_COUNT,
    assert_declaration_matches_corpus,
    mixture_training_streams,
    recomputed_target_and_scale,
    tier_dataset,
    tier_dirs,
    tier_spec,
    training_streams,
)
from offline.offline_baselines import StreamReturn

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The five expert directories and the one random directory a mixture may draw from.  Written out
#: rather than derived from the spec: this file's job is to check the spec, not to echo it.
EXPERT_DIR_NAMES = frozenset(
    f"cf_hz1x1__mappo1000__seed{seed}" for seed in (101, 202, 303, 404, 505)
)
RANDOM_DIR_NAME = "cf_hz1x1__random"

#: ``round(200 * f)`` for the three declared fractions, written as literals for the same reason.
EXPECTED_EXPERT_COUNT = {"mix33": 66, "mix50": 100, "mix67": 134}


@pytest.fixture(scope="module")
def corpus_v11_root() -> Path:
    """``RLTRAFFIC_CORPUS_V11``, else ``<repo>/datasets_v11``; skip if neither exists."""
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO_ROOT / "datasets_v11"
    if not candidate.is_dir():
        pytest.skip(
            f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to a "
            "collected datasets_v11/ directory to run the corpus-backed P4.7 tests"
        )
    return candidate


@pytest.fixture(scope="module")
def component_sets(corpus_v11_root: Path) -> dict[str, tuple[StreamReturn, ...]]:
    """The two components' own declared training sets, through the PRODUCTION helper.

    ``method_tier_grid._component_streams`` is what ``_run_train`` calls: the ``mappo1000`` tier's
    200 streams and the ``random`` tier's size-matched one-per-draw 200, each built by that tier's
    own declared rule.
    """
    from offline.dt_gate import CONTEXT_LENGTH
    from offline.method_tier_grid import _component_streams

    return _component_streams(
        tier_spec("mix50"), corpus_v11_root, context_length=CONTEXT_LENGTH
    )


@pytest.fixture(scope="module")
def mixtures(
    corpus_v11_root: Path, component_sets: dict[str, tuple[StreamReturn, ...]]
) -> dict[str, tuple[StreamReturn, ...]]:
    """Each mixture's realised training set, built the way ``_run_train`` builds it.

    ⚠️ **Through :func:`training_streams`, never through :func:`mixture_training_streams`
    directly.**  An earlier draft of this file called the selector straight, and a mutation that
    **swapped the two component arguments inside `training_streams`' mixture branch** -- turning
    ``mix33`` into ``mix67``'s composition under ``mix33``'s name -- **survived all fourteen tests**,
    because that dispatch was the one line the fixture bypassed.  The production entry point is the
    only one that exercises "which component is the expert".
    """
    return {
        tier: training_streams(
            tier_spec(tier),
            tier_dataset(tier_spec(tier), corpus_v11_root),
            component_streams=component_sets,
        )
        for tier in MIXTURE_TIER_ORDER
    }


@pytest.fixture(scope="module")
def raw_returns(
    component_sets: dict[str, tuple[StreamReturn, ...]],
) -> dict[tuple[str, str, str], float]:
    """Every component stream's return, re-derived from the raw ``.npz`` by an INDEPENDENT route.

    ``math.fsum`` over the stored ``local_reward`` rows, straight off disk.  Nothing here calls
    ``stream_returns``, so a shared convention error between the loader and the selector cannot hide.
    """
    from offline.trajectory_logger import load_episode

    out: dict[tuple[str, str, str], float] = {}
    for streams in component_sets.values():
        for entry in streams:
            if entry.key in out:
                continue
            episode = load_episode(Path(entry.dataset_dir) / entry.episode_file)
            rewards = np.asarray(
                episode.intersections[entry.ix_id].local_reward, dtype=np.float32
            )
            out[entry.key] = math.fsum(float(r) for r in rewards)
    return out


# ----------------------------------------------------------------------
# The realised composition, asserted from the built stream list
# ----------------------------------------------------------------------


def test_the_declared_fractions_round_to_the_counts_this_file_asserts() -> None:
    """66 / 100 / 134, derived here from the declared fractions rather than trusted as literals."""
    for tier, fraction in MIXTURE_EXPERT_FRACTION.items():
        assert int(round(TRAINING_STREAM_COUNT * fraction)) == EXPECTED_EXPERT_COUNT[tier], tier
    assert sorted(MIXTURE_EXPERT_FRACTION) == sorted(EXPECTED_EXPERT_COUNT)


def test_each_mixture_holds_exactly_two_hundred_distinct_streams(
    mixtures: dict[str, tuple[StreamReturn, ...]],
) -> None:
    """Size matching is the whole design (``BRIEF_19`` section 5.1); a repeat would fake it."""
    for tier, streams in mixtures.items():
        assert len(streams) == TRAINING_STREAM_COUNT, tier
        assert len({s.key for s in streams}) == TRAINING_STREAM_COUNT, tier


def test_the_expert_count_is_the_declared_rounded_fraction(
    mixtures: dict[str, tuple[StreamReturn, ...]],
    component_sets: dict[str, tuple[StreamReturn, ...]],
) -> None:
    """Counted from each stream's OWN directory, never from the spec that asked for it.

    The second leg ties the count to ``spec.components[0]`` **by name**: the expert fraction is the
    fraction of the FIRST declared component, so swapping the two at the call site turns ``mix33``
    into ``mix67`` under ``mix33``'s name -- silently, since both compositions are legal.
    """
    for tier, streams in mixtures.items():
        expert = sum(1 for s in streams if Path(s.dataset_dir).name in EXPERT_DIR_NAMES)
        randoms = sum(1 for s in streams if Path(s.dataset_dir).name == RANDOM_DIR_NAME)
        assert expert == EXPECTED_EXPERT_COUNT[tier], tier
        assert randoms == TRAINING_STREAM_COUNT - EXPECTED_EXPERT_COUNT[tier], tier

        first_component = tier_spec(tier).components[0]
        assert first_component == "mappo1000", tier
        from_first = {s.key for s in component_sets[first_component]}
        assert sum(1 for s in streams if s.key in from_first) == EXPECTED_EXPERT_COUNT[tier], tier


def test_every_stream_comes_from_a_declared_component_directory(
    mixtures: dict[str, tuple[StreamReturn, ...]],
) -> None:
    """No third source: a mixture is two components and the artifact must be able to say so."""
    for tier, streams in mixtures.items():
        names = {Path(s.dataset_dir).name for s in streams}
        assert names <= (EXPERT_DIR_NAMES | {RANDOM_DIR_NAME}), (tier, sorted(names))
        assert names & EXPERT_DIR_NAMES, tier
        assert RANDOM_DIR_NAME in names, tier


def test_every_selected_stream_belongs_to_its_components_training_set(
    mixtures: dict[str, tuple[StreamReturn, ...]],
    component_sets: dict[str, tuple[StreamReturn, ...]],
) -> None:
    """A mixture may not reach data its components do not themselves train on.

    The expert half must be a subset of ``mappo1000``'s declared 200 and the random half a subset of
    ``random``'s size-matched 200 -- not of that tier's 400-episode split.  ``BRIEF_17`` section 11
    finding A1's one-episode-per-draw rule reaches the mixtures only through this containment.
    """
    expert_keys = {s.key for s in component_sets["mappo1000"]}
    random_keys = {s.key for s in component_sets["random"]}
    assert not (expert_keys & random_keys)
    for tier, streams in mixtures.items():
        for entry in streams:
            pool = expert_keys if Path(entry.dataset_dir).name in EXPERT_DIR_NAMES else random_keys
            assert entry.key in pool, (tier, entry.key)


def test_the_selection_is_reproducible_and_decided_by_the_sort_key_not_by_load_order(
    component_sets: dict[str, tuple[StreamReturn, ...]],
) -> None:
    """Two builds agree, and a SHUFFLED component gives the same answer.

    ``mixture_training_streams`` sorts each component by ``(dataset_dir, episode_file, ix_id)``
    before drawing, so the selection depends on the declared RNG and on nothing else.  Feeding it a
    shuffled pool is the only way to prove that the sort is what fixes the outcome: with the pools in
    their natural order, dropping the sort would be invisible.
    """
    expert = list(component_sets["mappo1000"])
    randoms = list(component_sets["random"])
    shuffler = random.Random(20260815)
    shuffled_expert = expert[:]
    shuffled_random = randoms[:]
    shuffler.shuffle(shuffled_expert)
    shuffler.shuffle(shuffled_random)
    assert [s.key for s in shuffled_expert] != [s.key for s in expert]

    for tier in MIXTURE_TIER_ORDER:
        spec = tier_spec(tier)
        first = mixture_training_streams(spec, expert, randoms)
        again = mixture_training_streams(spec, expert, randoms)
        shuffled = mixture_training_streams(spec, shuffled_expert, shuffled_random)
        assert [s.key for s in first] == [s.key for s in again], tier
        assert [s.key for s in first] == [s.key for s in shuffled], tier


def test_the_three_mixtures_are_drawn_by_the_declared_rng_and_differ_from_each_other(
    mixtures: dict[str, tuple[StreamReturn, ...]],
) -> None:
    """One declared stream per fraction: ``MIXTURE_RNG_BASE + round(100 * f)``.

    The three seeds are 20260846 / 20260863 / 20260880, so the three selections are independent
    draws and not one selection re-cut at three sizes.  If they were nested, ``mix33``'s expert half
    would be a subset of ``mix67``'s and the three tiers would share their most-expert streams by
    construction rather than by chance.
    """
    assert {MIXTURE_RNG_BASE + int(round(100 * f)) for f in MIXTURE_EXPERT_FRACTION.values()} == {
        20260846,
        20260863,
        20260880,
    }
    experts = {
        tier: {s.key for s in streams if Path(s.dataset_dir).name in EXPERT_DIR_NAMES}
        for tier, streams in mixtures.items()
    }
    assert not experts["mix33"] <= experts["mix67"]


def test_the_mixture_dataset_resolves_to_exactly_one_state_action_group(
    corpus_v11_root: Path,
) -> None:
    """``_run_train`` takes ``next(iter(dataset.groups))``: a second group would be dropped silently.

    Every ``cf_hz1x1`` directory carries one intersection of the same width, so the union of the six
    must be a single ``(state_dim, n_actions)`` group.  This is cheap to assert and expensive to
    discover after 40,000 gradient steps on half a corpus.
    """
    for tier in MIXTURE_TIER_ORDER:
        dataset = tier_dataset(tier_spec(tier), corpus_v11_root)
        assert len(dataset.groups) == 1, (tier, sorted(dataset.groups))


# ----------------------------------------------------------------------
# The prompt: declared against the COMPOSED training set, by an independent route
# ----------------------------------------------------------------------


def test_the_returns_the_selector_carries_are_the_returns_on_disk(
    mixtures: dict[str, tuple[StreamReturn, ...]],
    raw_returns: dict[tuple[str, str, str], float],
) -> None:
    """Every selected stream's ``total_return`` equals ``fsum(local_reward)`` read off the ``.npz``.

    Exact equality, not a tolerance: both routes sum the same stored ``float32`` rows, and the
    prompt is derived from a maximum over exactly these numbers.
    """
    for tier, streams in mixtures.items():
        for entry in streams:
            assert float(entry.total_return) == raw_returns[entry.key], (tier, entry.key)


def test_the_declared_target_and_scale_are_the_composed_training_sets_own_maxima(
    mixtures: dict[str, tuple[StreamReturn, ...]],
    raw_returns: dict[tuple[str, str, str], float],
) -> None:
    """``BRIEF_17`` section 11 finding A4, and ``BRIEF_19`` section 5.5, applied to a mixture.

    The rule is the module's own: ``target_rtg = max(return)`` and ``rtg_scale = max|return|`` over
    the episodes the model actually trains on -- for a mixture, **after composition**.  A target
    derived from the six directories' union instead would ask the model for a return that may sit
    outside its own training set, which is the failure ``assert_declaration_matches_corpus`` exists
    to refuse.

    Both legs are checked: the declared spec against an independently recomputed pair, and the
    production guard against the same selection.
    """
    for tier, streams in mixtures.items():
        spec = tier_spec(tier)
        expected_target = max(raw_returns[s.key] for s in streams)
        expected_scale = max(abs(raw_returns[s.key]) for s in streams)
        assert float(spec.target_rtg) == expected_target, tier
        assert float(spec.rtg_scale) == expected_scale, tier

        target, scale = recomputed_target_and_scale(streams)
        assert target == expected_target, tier
        assert scale == expected_scale, tier
        declared = assert_declaration_matches_corpus(spec, streams)
        assert declared["target_rtg"] == expected_target, tier
        assert declared["training_streams"] == TRAINING_STREAM_COUNT, tier


def test_a_mixtures_target_is_not_assumed_to_be_the_experts_own_maximum(
    mixtures: dict[str, tuple[StreamReturn, ...]],
    component_sets: dict[str, tuple[StreamReturn, ...]],
) -> None:
    """The composed maximum is a fact about the draw, not a property of the expert component.

    ``docs/plans/p4.6.md`` section 8 declared all three mixtures at the union's maximum, which is
    ``mappo1000``'s best stream -- true only if that one stream is drawn, which happens with
    probability equal to the expert fraction.  This test records which mixtures actually contain it,
    so the packet reports the realised fact rather than the assumed one.
    """
    best_expert = max(component_sets["mappo1000"], key=lambda s: s.total_return)
    for tier, streams in mixtures.items():
        keys = {s.key for s in streams}
        spec = tier_spec(tier)
        if best_expert.key in keys:
            assert float(spec.target_rtg) == float(best_expert.total_return), tier
        else:
            assert float(spec.target_rtg) < float(best_expert.total_return), tier


# ----------------------------------------------------------------------
# ``--artifact-prefix`` (RULING 3): additive, and the default must preserve behaviour
# ----------------------------------------------------------------------


def test_the_default_artifact_prefix_reproduces_every_pre_p4_7_filename() -> None:
    """The four names P4.6 wrote, reconstructed from the helper with the default prefix.

    Written as literals on the right-hand side deliberately: this is the regression gate on RULING
    3's "additive, default preserves current behaviour", and a gate that derives its expectation
    from the same constant it is checking would pass under any rename.
    """
    from offline.method_tier_grid import DEFAULT_ARTIFACT_PREFIX, artifact_path

    assert DEFAULT_ARTIFACT_PREFIX == "p4_6"
    for name, expected in (
        ("declaration", "p4_6_declaration.json"),
        ("selection_diagnostics", "p4_6_selection_diagnostics.json"),
        ("training", "p4_6_training.json"),
        ("grid", "p4_6_grid.json"),
    ):
        assert artifact_path("docs/data", name) == Path("docs/data") / expected
        assert artifact_path("docs/data", name, "p4_7") == Path("docs/data") / expected.replace(
            "p4_6", "p4_7"
        )
    with pytest.raises(ValueError, match="non-empty"):
        artifact_path("docs/data", "grid", "  ")


def test_a_hand_built_namespace_still_gets_the_pre_p4_7_filenames() -> None:
    """The fallback that let RULING 3 land without editing one pre-existing test.

    P4.6's ``test_the_training_cli_path_runs_end_to_end_on_a_fixture`` calls ``_run_train`` with an
    ``argparse.Namespace`` it builds itself, so a newly required attribute would break it at setup
    rather than at an assertion.  **The production side absorbs that instead of the test being
    edited**, and the fallback is the documented default -- so a hand-built namespace writes exactly
    the filenames it wrote before the argument existed.
    """
    import argparse

    from offline.method_tier_grid import _artifact_prefix

    assert _artifact_prefix(argparse.Namespace()) == "p4_6"
    assert _artifact_prefix(argparse.Namespace(artifact_prefix="p4_7")) == "p4_7"


def test_the_cli_default_prefix_is_p4_6_and_the_flag_only_moves_the_name(
    corpus_v11_root: Path, tmp_path: Path
) -> None:
    """End to end through ``main``: same payload, different filename.

    Run twice on one real tier, once with the default and once with ``p4_7``.  Everything but
    ``runtime`` -- which records wall-clock and git state -- must be identical, so the flag is proved
    to move the name and nothing else.  A prefix that also perturbed a declared value would show up
    here as a payload difference rather than as a silently different artifact.
    """
    import json

    from offline.method_tier_grid import build_parser, main

    assert build_parser().parse_args(
        ["--corpus-root", "x", "declare"]
    ).artifact_prefix == "p4_6"

    argv = ["--corpus-root", str(corpus_v11_root), "--out-dir", str(tmp_path), "declare",
            "--tiers", "mappo1000"]
    assert main(argv) == 0
    assert main([*argv[:4], "--artifact-prefix", "p4_7", *argv[4:]]) == 0

    default = json.loads((tmp_path / "p4_6_declaration.json").read_text(encoding="utf-8"))
    prefixed = json.loads((tmp_path / "p4_7_declaration.json").read_text(encoding="utf-8"))
    assert sorted(p.name for p in tmp_path.glob("*.json")) == [
        "p4_6_declaration.json",
        "p4_7_declaration.json",
    ]
    default.pop("runtime")
    prefixed.pop("runtime")
    assert default == prefixed


def test_the_mixture_training_set_reports_both_composition_axes(
    mixtures: dict[str, tuple[StreamReturn, ...]],
) -> None:
    """``BRIEF_17`` section 11 finding A5: expert-versus-random AND the behaviour-seed histogram.

    ``DEFERRED`` 28's confound is inherited here and must be legible rather than fixed: the five
    ``mappo1000`` seeds occupy disjoint contiguous 40-draw blocks, so a mixture's expert component
    carries a seed-by-demand confound it did not create.  The histogram is what lets the packet
    report which seeds and which draw blocks each mixture drew from.
    """
    from offline.method_tier_grid import kept_composition

    for tier, streams in mixtures.items():
        composition = kept_composition(streams)
        assert composition["total"] == TRAINING_STREAM_COUNT, tier
        assert sum(composition["by_dataset_dir"].values()) == TRAINING_STREAM_COUNT, tier
        assert composition["by_dataset_dir"].get(RANDOM_DIR_NAME) == (
            TRAINING_STREAM_COUNT - EXPECTED_EXPERT_COUNT[tier]
        ), tier
        assert sum(composition["by_behaviour_seed"].values()) == EXPECTED_EXPERT_COUNT[tier], tier
        assert composition["without_a_behaviour_seed"] == (
            TRAINING_STREAM_COUNT - EXPECTED_EXPERT_COUNT[tier]
        ), tier
