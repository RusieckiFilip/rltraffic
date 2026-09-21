"""P5.4: the trained-model footprint of the spatial layer -- what F's support must be, and what r is.

The load-bearing pair is :func:`test_the_support_is_the_masks_reach_on_the_three_node_stub` and
:func:`test_a_two_layer_path_graph_separates_the_two_readings_of_the_support_check`.  A22(b) first required exact zero
at **every** non-neighbour pair; that is an identity for one spatial layer, and the registered checkpoints have three,
so a correct model is influenced within three hops.  A22.1 corrected the clause, and its (d) requires a test that can
tell the two readings apart -- which the three-node stub cannot, because its non-neighbour is isolated and no two-hop
path exists there.  That is T1b, and it is why the path graph is here.

Every mutation named in ``docs/plans/p5.4.md`` section 4 is run against these tests and pasted in the Return Packet.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest
import torch

from offline import spatial_footprint as sf
from tests.test_spatial_dt_agent import IDENTITY_MASK, NEIGHBOUR_MASK, _batch, _model

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 0 -- 1 -- 2 -- 3, with the self-loops ``AdjacencySpec.attention_mask`` opens in both modes.
PATH_MASK = np.array(
    [
        [True, True, False, False],
        [True, True, True, False],
        [False, True, True, True],
        [False, False, True, True],
    ],
    dtype=np.bool_,
)


# ----------------------------------------------------------------------
# Fixtures that stay small on purpose: these tests check wiring, not capacity.
# ----------------------------------------------------------------------


def _stub_inputs(batch: int = 6, seed: int = 0) -> dict[str, Any]:
    """``tests/test_spatial_dt_agent._batch`` plus the availability key the footprint reads."""
    inputs: dict[str, Any] = dict(_batch(batch=batch, seed=seed))
    inputs["avail_mask"] = None
    return inputs


def _path_inputs(batch: int = 6, seed: int = 3) -> dict[str, Any]:
    """The same shape as ``_batch`` for a four-node graph."""
    generator = torch.Generator().manual_seed(seed)
    nodes, context, state_dim, n_actions = 4, 4, 4, 3
    return {
        "rtg": torch.randn(batch, nodes, context, 1, generator=generator),
        "state": torch.randn(batch, nodes, context, state_dim, generator=generator),
        "action": torch.randint(0, n_actions, (batch, nodes, context), generator=generator),
        "timestep": torch.arange(context)
        .view(1, 1, context)
        .expand(batch, nodes, context)
        .contiguous(),
        "attention_mask": torch.ones(batch, nodes, context, dtype=torch.bool),
        "avail_mask": None,
    }


def _donor(count: int, seed: int = 11) -> np.ndarray:
    return sf.cyclic_derangement(count, np.random.default_rng(seed))


class _MaskIgnoringModel(torch.nn.Module):
    """A model that discards the mask it is handed -- exactly the defect the support check exists to catch."""

    def __init__(self, inner: Any) -> None:
        super().__init__()
        self.inner = inner

    def forward(  # noqa: D102 - mirrors SpatialDecisionTransformer.forward
        self,
        rtg: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        timestep: torch.Tensor,
        spatial_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        avail_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.inner(
            rtg, state, action, timestep, torch.ones_like(spatial_mask), attention_mask, avail_mask
        )


def _synthetic_footprint(
    arm: str, seed: int, r: float, *, violations: Sequence[tuple[str, str]] = ()
) -> sf.CheckpointFootprint:
    """A footprint whose numbers are made up but whose pin, per-hop block and shape are real."""
    pin = sf.checkpoint_pin(arm, seed)
    size = NEIGHBOUR_MASK.shape[0]
    matrix = np.zeros((size, size), dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    matrix[0, 1] = matrix[1, 0] = float(r)
    return sf.CheckpointFootprint(
        pin=pin,
        sha256=pin.sha256,
        n_layer=1,
        spatial_mixing=pin.spatial_mixing,
        F=matrix,
        r=float(r),
        att=sf.A22_ATT[(arm, seed)],
        violations=tuple(violations),
        per_hop=sf.per_hop_report(matrix, sf.hop_distances(NEIGHBOUR_MASK), 1),
        legal_min=3,
        legal_max=3,
    )


def _synthetic_sample() -> sf.WindowSample:
    return sf.WindowSample(
        node_ids=("n0", "n1", "n2"),
        joint_rows=np.array([5, 9], dtype=np.int64),
        episode_index=np.array([0, 1], dtype=np.int64),
        t=np.array([19, 20], dtype=np.int64),
        donor=np.array([1, 0], dtype=np.int64),
        refs=(
            sf.WindowRef("cf_grid4x4__mappo1000__seed101", "episode_0001.npz", 1, 19),
            sf.WindowRef("cf_grid4x4__mappo1000__seed101", "episode_0002.npz", 2, 20),
        ),
        tensors={},
    )


def _synthetic_context() -> sf.RunContext:
    return sf.RunContext(
        node_ids=("n0", "n1", "n2"),
        corpus_dirs=("cf_grid4x4__mappo1000__seed101",),
        manifest_sha256={"cf_grid4x4__mappo1000__seed101": "0" * 64},
        stats_match_corpus=True,
        roadnet_sha256="1" * 64,
        degree_histogram={2: 4, 3: 8, 4: 4},
        undirected_edges=24,
        neighbours=NEIGHBOUR_MASK & ~np.eye(3, dtype=np.bool_),
        hop_distances=sf.hop_distances(NEIGHBOUR_MASK),
        context_length=20,
        joint_windows_total=72_000,
        eligible_windows=68_200,
        torch_threads=sf.TORCH_THREADS,
        eval_cells={
            arm: {"path": f"{sf.CANONICAL_EVAL_DIR}/{pin['filename']}", "sha256": pin["sha256"]}
            for arm, pin in sf.EVAL_CELLS.items()
        },
    )


def _artifact_for(median_target: float, **overrides: Any) -> dict[str, Any]:
    """Ten synthetic footprints whose spatial median is *median_target*."""
    spatial = [median_target] * 5
    footprints = [
        _synthetic_footprint("dt_spatial_h4", seed, value, **overrides)
        for seed, value in zip((101, 202, 303, 404, 505), spatial)
    ] + [
        _synthetic_footprint("dt_nomix_h4", seed, 0.0)
        for seed in (101, 202, 303, 404, 505)
    ]
    return sf.build_artifact(footprints, _synthetic_sample(), _synthetic_context())


# ----------------------------------------------------------------------
# T1 / T1b -- the support of F, and the two readings of A22(b)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_layer", [1, 2])
def test_the_support_is_the_masks_reach_on_the_three_node_stub(n_layer: int) -> None:
    """T1. Node 2 is isolated, so ``reach^L == mask`` at any depth and the support must equal the mask exactly."""
    model = _model(NEIGHBOUR_MASK, n_layer=n_layer)
    inputs = _stub_inputs()
    result = sf.influence(model, inputs, NEIGHBOUR_MASK, _donor(6))

    assert np.array_equal(result.F != 0.0, NEIGHBOUR_MASK), (
        f"the support {(result.F != 0.0).astype(int).tolist()} is not the mask"
    )
    assert np.array_equal(sf.reachability(NEIGHBOUR_MASK, n_layer), NEIGHBOUR_MASK)
    assert (
        sf.support_violations(
            result, sf.must_be_zero_pairs(NEIGHBOUR_MASK, n_layer), ("A", "B", "C")
        )
        == ()
    )
    # Discriminating power: a layer that did nothing would pass the line above with an all-zero F.
    assert float(result.F[0, 1]) > 0.0 and float(result.F[1, 0]) > 0.0


def test_with_mixing_off_no_other_node_influences_any_prediction() -> None:
    """T1, the control half: the identity mask leaves the off-diagonal exactly zero."""
    model = _model(IDENTITY_MASK, spatial_mixing=False, n_layer=2)
    result = sf.influence(model, _stub_inputs(), IDENTITY_MASK, _donor(6))

    assert np.array_equal(result.F != 0.0, np.eye(3, dtype=np.bool_))
    off_diagonal = result.F[~np.eye(3, dtype=np.bool_)]
    assert np.array_equal(off_diagonal, np.zeros_like(off_diagonal))
    assert (
        sf.support_violations(result, sf.must_be_zero_pairs(IDENTITY_MASK, 2), ("A", "B", "C"))
        == ()
    )


def test_a_two_layer_path_graph_separates_the_two_readings_of_the_support_check() -> None:
    """T1b (A22.1(d)). Two hops is reachable at two layers; three hops is not, and must be exactly zero.

    The three-node stub cannot make this distinction -- its non-neighbour is isolated -- which is why A22(b)'s
    one-hop reading survived until P5.4's plan gate.
    """
    model = _model(PATH_MASK, n_nodes=4, n_layer=2)
    result = sf.influence(model, _path_inputs(), PATH_MASK, _donor(6))

    # The registered claim, as two outcomes on one fixture.
    assert float(result.F[0, 2]) > 0.0, "node 2 is two hops from node 0 and must reach it at two layers"
    assert bool(result.unchanged[0, 3]), "node 3 is three hops from node 0 and must not reach it"
    assert float(result.F[0, 3]) == 0.0

    assert np.array_equal(result.F != 0.0, sf.reachability(PATH_MASK, 2))
    assert set(
        map(tuple, np.argwhere(sf.must_be_zero_pairs(PATH_MASK, 2)).tolist())
    ) == {(0, 3), (3, 0)}
    assert sf.support_violations(result, sf.must_be_zero_pairs(PATH_MASK, 2), "ABCD") == ()


def test_hop_distances_match_a_hand_written_breadth_first_search() -> None:
    """The distances the per-hop block is keyed by, recomputed by a route that shares no arithmetic."""
    size = PATH_MASK.shape[0]
    adjacency = PATH_MASK & ~np.eye(size, dtype=np.bool_)
    expected = np.full((size, size), -1, dtype=np.int64)
    for source in range(size):
        expected[source, source] = 0
        frontier, depth = [source], 0
        while frontier:
            depth += 1
            nxt: list[int] = []
            for node in frontier:
                for other in np.flatnonzero(adjacency[node]):
                    if expected[source, other] < 0:
                        expected[source, other] = depth
                        nxt.append(int(other))
            frontier = nxt
    assert np.array_equal(sf.hop_distances(PATH_MASK), expected)


# ----------------------------------------------------------------------
# T2 / T3 -- the statistic and the threshold
# ----------------------------------------------------------------------


def _cycle_fixture() -> tuple[np.ndarray, np.ndarray]:
    """A four-cycle whose r is exact in binary floating point, with non-zero entries off the mask."""
    F = np.zeros((4, 4), dtype=np.float64)
    np.fill_diagonal(F, [1.0, 2.0, 3.0, 2.0])          # mean 2.0
    F[0, 1], F[0, 3] = 0.5, 0.25                        # neighbour mean 0.375
    F[1, 0], F[1, 2] = 0.125, 0.375                     # 0.25
    F[2, 1], F[2, 3] = 0.75, 0.25                       # 0.5
    F[3, 0], F[3, 2] = 0.5, 0.5                         # 0.5
    F[0, 2], F[2, 0] = 8.0, 4.0                         # NOT neighbours: the mutations must see these
    F[1, 3], F[3, 1] = 16.0, 2.0
    neighbours = np.array(
        [
            [False, True, False, True],
            [True, False, True, False],
            [False, True, False, True],
            [True, False, True, False],
        ],
        dtype=np.bool_,
    )
    return F, neighbours


def test_the_ratio_equals_its_hand_computation_exactly() -> None:
    """T2. ``(0.375 + 0.25 + 0.5 + 0.5) / 4 = 0.40625``; ``(1 + 2 + 3 + 2) / 4 = 2.0``; r = 0.203125.

    Every value is a dyadic rational, so the arithmetic is exact and the comparison is ``==`` rather than a tolerance.
    """
    F, neighbours = _cycle_fixture()
    assert sf.footprint_ratio(F, neighbours) == 0.203125


def test_the_ratio_refuses_a_neighbour_set_that_carries_the_diagonal() -> None:
    """Counting F[i, i] in the numerator would inflate r by its own denominator."""
    F, neighbours = _cycle_fixture()
    with_diagonal = neighbours | np.eye(4, dtype=np.bool_)
    with pytest.raises(ValueError, match="diagonal"):
        sf.footprint_ratio(F, with_diagonal)


def test_the_threshold_is_strict_and_the_boundary_is_non_trivial() -> None:
    """T3. TRIVIAL iff the median is BELOW 0.10 (A22(c)); 0.10 itself is NON-TRIVIAL."""
    assert sf.verdict(0.09) == sf.VERDICT_TRIVIAL
    assert sf.verdict(0.10) == sf.VERDICT_NON_TRIVIAL
    assert sf.verdict(0.11) == sf.VERDICT_NON_TRIVIAL


def test_the_verdict_is_taken_on_the_median_of_the_five_seeds() -> None:
    """T3, second half: the median of five values, and the boundary reached through it."""
    assert sf.median_verdict([0.5, 0.01, 0.2, 0.05, 0.09]) == (0.09, sf.VERDICT_TRIVIAL)
    assert sf.median_verdict([0.5, 0.01, 0.2, 0.05, 0.10]) == (0.10, sf.VERDICT_NON_TRIVIAL)
    assert sf.median_verdict([0.5, 0.01, 0.2, 0.05, 0.11]) == (0.11, sf.VERDICT_NON_TRIVIAL)


# ----------------------------------------------------------------------
# T7 -- the pairing
# ----------------------------------------------------------------------


def test_the_pairing_is_a_derangement_and_is_fixed_by_the_seed() -> None:
    """T7. Every window donates once, receives once, and is never its own donor."""
    first = sf.cyclic_derangement(200, np.random.default_rng(sf.WINDOW_SEED))
    again = sf.cyclic_derangement(200, np.random.default_rng(sf.WINDOW_SEED))

    assert np.array_equal(first, again)
    assert sorted(first.tolist()) == list(range(200))
    assert not np.any(first == np.arange(200)), "a window would be its own donor"
    assert not np.array_equal(
        first, sf.cyclic_derangement(200, np.random.default_rng(sf.WINDOW_SEED + 1))
    )
    with pytest.raises(ValueError, match="at least two"):
        sf.cyclic_derangement(1, np.random.default_rng(0))


def test_only_full_windows_are_eligible_to_be_drawn() -> None:
    """T9. A22.1(e) fixes full windows only, and nothing else in this file pinned the rule.

    Added after a mutation survived: relaxing ``t >= context_length - 1`` to ``t >= 0`` left all nineteen tests
    green, because the gated draw happened not to pick a padded instant.  A padded donor block would put
    normalised zeros -- the per-feature mean, not an observation -- into a real position of the recipient.
    """
    steps = np.arange(6, dtype=np.int64)
    assert sf.eligible_instants(steps, 3).tolist() == [2, 3, 4, 5]
    assert sf.eligible_instants(steps, 1).tolist() == [0, 1, 2, 3, 4, 5]
    assert sf.eligible_instants(steps, 6).tolist() == [5]
    assert sf.eligible_instants(steps, 7).tolist() == []
    # The corpus's own shape: 341 of each episode's 360 instants carry twenty real steps.
    assert sf.eligible_instants(np.arange(360, dtype=np.int64), 20).size == 341


# ----------------------------------------------------------------------
# T5 -- refusals.  Every one precedes a filesystem call.
# ----------------------------------------------------------------------


def test_a_checkpoint_digest_outside_a22_is_refused(tmp_path: Path) -> None:
    """T5(a). The ten files are registered by digest; anything else is a different model."""
    impostor = tmp_path / sf.checkpoint_pin("dt_spatial_h4", 101).filename
    impostor.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="A22"):
        sf.verify_checkpoint(impostor, "dt_spatial_h4", 101)
    with pytest.raises(ValueError, match="A22"):
        sf.checkpoint_pin("dt_spatial_h4", 999)


def test_a_tier_other_than_mappo1000_is_refused_before_any_read(tmp_path: Path) -> None:
    """T5(b). A22(a) registers one tier; another corpus would give a plausible F these models never saw."""
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="mappo1000"):
        sf.tier_directories(missing, tier="maxpressure")
    assert not missing.exists(), "the refusal touched the filesystem"


def test_the_writer_refuses_a_support_violation_and_writes_nothing(tmp_path: Path) -> None:
    """T5(c). The violation is produced by a model that ignores its mask, not by an edited field."""
    inner = _model(NEIGHBOUR_MASK, n_layer=1)
    leaky = _MaskIgnoringModel(inner).eval()
    result = sf.influence(leaky, _stub_inputs(), NEIGHBOUR_MASK, _donor(6))
    violations = sf.support_violations(
        result, sf.must_be_zero_pairs(NEIGHBOUR_MASK, 1), ("A", "B", "C")
    )
    assert violations, "the leaky model did not leak, so this test proves nothing"

    artifact = _artifact_for(0.5, violations=violations)
    out_dir = tmp_path / "data"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="support"):
        sf.write_artifact(artifact, out_dir / "p5_2_spatial_footprint.json")
    assert list(out_dir.iterdir()) == [], "a refused write left a file behind"

    absent = tmp_path / "absent"
    with pytest.raises((ValueError, FileNotFoundError), match="support|directory"):
        sf.write_artifact(artifact, absent / "p5_2_spatial_footprint.json")
    assert not absent.exists(), "a refused write created a directory"


def test_the_writer_refuses_a_verdict_that_disagrees_with_its_own_median(tmp_path: Path) -> None:
    """The gate recomputes the median from the recorded per-seed r rather than trusting the summary."""
    artifact = _artifact_for(0.5)
    artifact["summary"]["verdict"] = sf.VERDICT_TRIVIAL
    with pytest.raises(ValueError, match="verdict"):
        sf.write_artifact(artifact, tmp_path / "p5_2_spatial_footprint.json")
    assert list(tmp_path.iterdir()) == []


def test_the_writer_accepts_a_clean_artifact_and_writes_it_atomically(tmp_path: Path) -> None:
    """Discriminating power for the refusals above: the same writer does write a clean artifact."""
    destination = tmp_path / "p5_2_spatial_footprint.json"
    sf.write_artifact(_artifact_for(0.5), destination)
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written["format_version"] == sf.FORMAT_VERSION
    assert written["summary"]["verdict"] == sf.VERDICT_NON_TRIVIAL


# ----------------------------------------------------------------------
# T6 / T8 -- what the paper writes, and what the registration pinned
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "median_target,expected",
    [(0.05, sf.VERDICT_TRIVIAL), (0.5, sf.VERDICT_NON_TRIVIAL)],
)
def test_both_registered_sentences_are_carried_verbatim_whichever_verdict_lands(
    median_target: float, expected: str
) -> None:
    """T6. The road not taken stays visible, and the clauses are compared against the registration itself."""
    registration = (REPO_ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    artifact = _artifact_for(median_target)

    assert artifact["summary"]["verdict"] == expected
    carried = artifact["paper_sentences"]
    assert set(carried) == {sf.VERDICT_TRIVIAL, sf.VERDICT_NON_TRIVIAL}
    for name, block in carried.items():
        assert block["a22_d_clause"] in registration, f"{name} is not A22(d)'s clause"
        assert block["sentence"] in block["a22_d_clause"]
        assert block["chosen"] is (name == expected)
    assert sum(block["chosen"] for block in carried.values()) == 1


def test_the_pinned_digests_are_the_ones_a22_registered() -> None:
    """T8. A22(a) registers eight hex characters per file; the module pins the full digest under each."""
    registration = (REPO_ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    assert len(sf.A22_CHECKPOINTS) == 10

    for pin in sf.A22_CHECKPOINTS:
        assert pin.sha256.startswith(pin.a22_prefix)
        assert len(pin.sha256) == 64
        assert f"`{pin.a22_prefix}…`" in registration

    for arm in ("dt_spatial_h4", "dt_nomix_h4"):
        prefixes = [pin.a22_prefix for pin in sf.A22_CHECKPOINTS if pin.arm == arm]
        assert len(prefixes) == 5
        quoted = ", ".join(f"`{prefix}…`" for prefix in prefixes)
        assert quoted in registration, f"{arm}'s digests are not in A22(a)'s order"


# ----------------------------------------------------------------------
# T4 -- one real checkpoint, gated on the tree carrying P5.2's outputs and the corpus
# ----------------------------------------------------------------------

_OUTPUT_ROOT = Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT", str(REPO_ROOT / "output")))
_CORPUS_ROOT = Path(os.environ.get("RLTRAFFIC_CORPUS_V11", str(REPO_ROOT / "datasets_v11")))
_REAL_CHECKPOINT = (
    _OUTPUT_ROOT / "p5_2" / "checkpoints" / "grid4x4_mappo1000_dt_spatial_h4_seed101.pt"
)
_REAL_CORPUS = _CORPUS_ROOT / "cf_grid4x4__mappo1000__seed101"


@pytest.mark.skipif(
    not _REAL_CHECKPOINT.is_file() or not (_REAL_CORPUS / "manifest.json").is_file(),
    reason=(
        f"P5.2's checkpoint {_REAL_CHECKPOINT} or the corpus {_REAL_CORPUS} is not in this tree; "
        "set RLTRAFFIC_OUTPUT_ROOT and RLTRAFFIC_CORPUS_V11 to run the trained-model footprint test"
    ),
)
def test_a_real_spatial_checkpoint_has_an_exact_support_and_a_finite_footprint() -> None:
    """T4. The mechanism on a trained model: exact zero beyond reach, F finite, r defined and non-negative.

    One draw (one episode, eight windows) keeps this a wiring test.  The measurement itself is the CLI's job.
    """
    from offline.tier_sweep import adjacency_for_tier, node_ids_from_corpus, tier_spec

    spec = tier_spec(sf.TIER)
    node_ids = node_ids_from_corpus(spec, _CORPUS_ROOT)
    adjacency = adjacency_for_tier(spec, _CORPUS_ROOT, node_ids)
    subject = sf.load_subject(_REAL_CHECKPOINT, node_ids, "dt_spatial_h4", 101)
    context_length = int(subject.config.context_length)

    dataset = sf.footprint_dataset(_CORPUS_ROOT, context_length=context_length, draw_ids=[1])
    sample = sf.draw_windows(dataset, node_ids, context_length=context_length, n_windows=8)
    inputs = sf.model_inputs(sample, subject)
    result = sf.influence(subject.agent.model, inputs, subject.mask, sample.donor)

    n_layer = int(subject.config.n_layer)
    violations = sf.support_violations(
        result, sf.must_be_zero_pairs(subject.mask, n_layer), node_ids
    )
    assert violations == (), f"the recorded mask does not hold on a trained model: {violations[:8]}"
    assert np.isfinite(result.F).all()
    # Non-vacuity: a model that ignored every state would satisfy the support check with an all-zero F.
    assert (np.diagonal(result.F) > 0.0).all()

    ratio = sf.footprint_ratio(result.F, adjacency.undirected)
    assert ratio >= 0.0

    report = sf.per_hop_report(result.F, sf.hop_distances(adjacency.undirected | np.eye(16, dtype=np.bool_)), n_layer)
    assert report["max_abs_beyond_reach"] == 0.0


@pytest.mark.skipif(
    not _REAL_CHECKPOINT.is_file() or not (_REAL_CORPUS / "manifest.json").is_file(),
    reason=(
        f"P5.2's checkpoint {_REAL_CHECKPOINT} or the corpus {_REAL_CORPUS} is not in this tree; "
        "set RLTRAFFIC_OUTPUT_ROOT and RLTRAFFIC_CORPUS_V11 to run the trained-model footprint test"
    ),
)
def test_each_nodes_return_to_go_is_divided_by_its_own_recorded_scale() -> None:
    """T10. The trainer divided each node's RTG by ITS OWN ``rtg_scale``; feeding one node's scale to all sixteen
    is the multi-intersection mutant this project has already been bitten by (``DEFERRED`` 37).
    """
    from offline.tier_sweep import node_ids_from_corpus, tier_spec

    node_ids = node_ids_from_corpus(tier_spec(sf.TIER), _CORPUS_ROOT)
    subject = sf.load_subject(_REAL_CHECKPOINT, node_ids, "dt_spatial_h4", 101)
    span = int(subject.config.context_length)
    dataset = sf.footprint_dataset(_CORPUS_ROOT, context_length=span, draw_ids=[1])
    sample = sf.draw_windows(dataset, node_ids, context_length=span, n_windows=4)
    inputs = sf.model_inputs(sample, subject)

    scales = [float(subject.rtg_scale[ix_id]) for ix_id in node_ids]
    assert len(set(scales)) > 1, "every node shares a scale here, so this test could not see the mutant"
    for position, scale in enumerate(scales):
        expected = sample.tensors["rtg"][:, position] / scale
        assert torch.equal(inputs["rtg"][:, position], expected), (
            f"node {node_ids[position]} was not divided by its own rtg_scale {scale}"
        )


def test_the_att_column_is_the_cells_own_per_seed_mean_and_is_refused_when_it_drifts() -> None:
    """T11. A22(c) quotes these five numbers per arm; the artifact may not carry a number the cell no longer holds."""
    from offline.dt_gate import HELD_OUT_DRAWS

    def payload(value: float, *, method: str = "dt_spatial_h4", tier: str = sf.TIER) -> dict[str, Any]:
        """A cell carrying TWO seeds at very different levels.

        The second seed is what makes this test able to see an implementation that averages the whole cell instead
        of the seed it was asked for -- with one seed present the two are the same number.
        """
        return {
            "method": method,
            "tier": tier,
            "episodes": [
                {"seed": seed, "draw_id": draw, "att_horizon": level}
                for seed, level in ((101, value), (202, value + 100.0))
                for draw in HELD_OUT_DRAWS
            ],
        }

    registered = sf.A22_ATT[("dt_spatial_h4", 101)]
    assert sf.per_seed_att(payload(registered), "dt_spatial_h4", 101) == pytest.approx(registered)
    with pytest.raises(ValueError, match="does not round to"):
        sf.per_seed_att(payload(registered + 1.0), "dt_spatial_h4", 101)
    with pytest.raises(ValueError, match="dt_spatial_h4"):
        sf.per_seed_att(payload(registered, method="dt_nomix_h4"), "dt_spatial_h4", 101)
    with pytest.raises(ValueError, match="draws"):
        short = payload(registered)
        short["episodes"] = short["episodes"][:10]
        sf.per_seed_att(short, "dt_spatial_h4", 101)
