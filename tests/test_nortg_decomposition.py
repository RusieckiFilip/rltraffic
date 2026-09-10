"""``offline.nortg_decomposition`` -- A13(b)'s decomposition, without a simulator or a GPU.

What this file defends, in order of what it would cost to get wrong
-------------------------------------------------------------------
1. **The 3,000-episode reproduction check.**  The decomposition is only the required quantity if it
   was measured on the SAME trajectory the campaign reported.  ``report`` must refuse -- and write
   **nothing** -- on a single episode that does not reproduce the committed cell under either
   definition.  A ``1e-9`` perturbation is enough; this is ``==``, never ``allclose``.
2. **A chunk may not vouch for itself.**  ``docs/reviews/P5.3b.md`` and P8.4b before it both found
   ``reproduces_committed`` being read as evidence.  Here the committed pair is re-read from
   ``docs/data/p5_3b_nortg.json`` and from P8.4b's re-derived cells, and the chunk's own idea of
   what is committed is checked against them.
3. **The identity is checked, not assumed.**  The three terms are stored, so they can be wrong;
   they are verified both against their own total and against an independent recomputation from the
   constructor fields.
4. **The fence is default-deny and knows the ``p5_3b`` / ``p5_3b_decomp`` trap**, which a
   string-prefix implementation cannot tell apart.
5. **A guard nobody calls is not a guard** (``DEFERRED`` 63) -- so the call site is pinned too.

⚠️ These tests need no corpus, no checkpoint and no simulator: every fixture is synthetic and every
number in them is exactly representable in binary, so ``==`` is the right assertion throughout.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pytest

from offline import nortg_campaign, nortg_decomposition
from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS
from offline.nortg_decomposition import (
    CONTRAST_TOLERANCE,
    DECOMP_METHODS,
    ActionRecorder,
    action_sequence_sha256,
    assert_chunks_complete,
    assert_contrast_consistent,
    assert_decomposition_writable,
    assert_identity_is_exact,
    assert_rows_reproduce_committed,
    chunk_is_reusable,
    chunk_name,
    reusable_chunk_at,
    decomposition_artifact,
    recording_factory,
)

TIERS = nortg_campaign.NORTG_TIERS

#: The synthetic episode, per method.  Chosen so that every value and every term is exactly
#: representable in binary and the identity closes at 0.0 with no rounding whatsoever:
#:
#:     term_population   = P - E     term_clock_origin = W - P     term_cadence = C - W
#:     total             = C - E     residual          = 0.0       deviation_c1 = deviation_c3c = 0.0
#:
#: ``dt_nortg`` is given a large NEGATIVE clock-origin term because that is the shape the real
#: collapsed arm is expected to have -- a policy holding vehicles at the boundary -- so the fixture
#: exercises the sign the orientation note warns about rather than a tidy positive one.
_SHAPE: Mapping[str, Mapping[str, float]] = {
    #        E        P       W        C
    "dt": {"E": 100.0, "P": 99.0, "W": 98.5, "C": 106.5},
    "dt_nortg": {"E": 500.0, "P": 496.0, "W": 400.0, "C": 402.0},
}

#: Hand-computed from ``_SHAPE``, written as literals so the expectation is an independent route
#: rather than the code under test run twice.  ``mean_difference`` is ``mean(ATT_dt - ATT_dt_nortg)``
#: (``paired_stats``' registered sign convention), and every cell shares one shape, so the mean over
#: draws is that shape's own difference.
_DECLARED_ENGINE = 100.0 - 500.0  # -400.0
_DECLARED_OURS = 106.5 - 402.0  # -295.5
#: delta_ours - delta_engine = 104.5, and the three term contrasts are 3.0 + 95.5 + 6.0 = 104.5.
_EXPECTED_IDENTITY_GAP = 104.5


def _episode(method: str, tier: str, seed: int, draw: int, **over: Any) -> dict[str, Any]:
    """One synthetic episode row, self-consistent unless ``over`` breaks it on purpose."""
    shape = _SHAPE[method]
    # a bump that is identical for both arms, so the per-draw DIFFERENCE stays exactly the shape's
    # own and _DECLARED_* remain hand-computable literals
    bump = float(((draw - 1000) % 4) * 0.25 + (TRAINING_SEEDS.index(seed) * 0.5))
    engine = shape["E"] + bump
    pool = shape["P"] + bump
    running = shape["W"] + bump
    ours = shape["C"] + bump
    row = {
        "scenario": "hz1x1",
        "tier": tier,
        "method": method,
        "arm": f"{method}@{tier}",
        "seed": int(seed),
        "draw_id": int(draw),
        "role": "tier",
        "engine_seed": 1000,
        "att_reference_engine_population": engine,
        "att_reference_entered_running": running,
        "att_reference_entered_population": pool,
        "att_reference_metric_cadence": ours,
        "att_engine_call": engine,
        "att_ours": ours,
        "n_reference_ids": 1813,
        "n_entered_ids": 1813,
        "created_from_flow": 1813,
        "entered": 1813,
        "never_entered": 0,
        "admission_latency_mean": 0.5,
        "admission_latency_max": 4.0,
        "n_admission_delayed": 465,
        "interval": 1.0,
        "n_observations": 3600,
        "seconds": 3.3,
        "seconds_rollout": 2.9,
        "deviation_c1": 0.0,
        "deviation_c3c": 0.0,
        "difference_c3a_running": abs(engine - running),
        "difference_c3a_population": abs(engine - pool),
        "term_population": pool - engine,
        "term_clock_origin": running - pool,
        "term_cadence": ours - running,
        "decomposition_residual": 0.0,
        "difference_ours_minus_engine": ours - engine,
        "committed_att_engine": engine,
        "committed_att_ours": ours,
        "reproduces_committed": True,
        "n_decisions": 360,
        "action_counts": [[180, 180]],
        "action_sequence_sha256": "0" * 64,
        "checkpoint": f"output/x/{tier}_{method}_seed{seed}.pt",
        "checkpoint_sha256": "a" * 64,
    }
    row.update(over)
    return row


def _chunk(method: str, tier: str, seed: int, *, draws: Sequence[int] | None = None,
           **over: Any) -> dict[str, Any]:
    ids = list(draws if draws is not None else HELD_OUT_DRAWS)
    episodes = [_episode(method, tier, seed, draw) for draw in ids]
    chunk = {
        "format_version": nortg_decomposition.ARTIFACT_FORMAT_VERSION,
        "method": method,
        "tier": tier,
        "seed": int(seed),
        "arm": f"{method}@{tier}",
        "draws": ids,
        "is_complete": len(ids) == len(HELD_OUT_DRAWS),
        "n_mismatches": 0,
        "episodes": episodes,
        "seconds": 330.0,
        "seconds_per_episode": 3.3,
    }
    chunk.update(over)
    return chunk


def _chunks() -> list[dict[str, Any]]:
    """All thirty cells: two methods x three tiers x five seeds."""
    return [
        _chunk(method, tier, seed)
        for method in DECOMP_METHODS
        for tier in TIERS
        for seed in TRAINING_SEEDS
    ]


def _committed(chunks: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int, int], dict[str, float]]:
    """The committed source rows the chunks must agree with."""
    return {
        (row["method"], row["tier"], int(row["seed"]), int(row["draw_id"])): {
            "att_engine": _SHAPE[row["method"]]["E"]
            + float(((int(row["draw_id"]) - 1000) % 4) * 0.25
                    + TRAINING_SEEDS.index(int(row["seed"])) * 0.5),
            "att_ours": _SHAPE[row["method"]]["C"]
            + float(((int(row["draw_id"]) - 1000) % 4) * 0.25
                    + TRAINING_SEEDS.index(int(row["seed"])) * 0.5),
        }
        for chunk in chunks
        for row in chunk["episodes"]
    }


def _declared() -> dict[str, dict[str, float]]:
    return {tier: {"att_engine": _DECLARED_ENGINE, "att_ours": _DECLARED_OURS} for tier in TIERS}


def _rows(chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [row for chunk in chunks for row in chunk["episodes"]]


# ----------------------------------------------------------------------
# 1. 🔒 The reproduction check, and the filesystem-mutation barrier behind it
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["att_engine_call", "att_ours", "deviation_c1", "deviation_c3c", "decomposition_residual"],
)
def test_one_episode_that_does_not_reproduce_the_committed_cell_refuses_the_whole_artifact(
    field: str,
) -> None:
    """🔒 A ``1e-9`` perturbation on ONE of 3,000 episodes is enough to refuse.

    Gate 0 established ``c1 = 0.0`` and ``c3c = 0.0`` on 46 episodes of OTHER arms.  Whether the
    observed DT trajectory equals the campaign's is a new fact, and this is the check that
    establishes it -- a decomposition of a different trajectory is not the required quantity.
    ``==``, never ``allclose`` (``CLAUDE.md`` section 2).
    """
    chunks = _chunks()
    row = chunks[7]["episodes"][42]
    row[field] = float(row[field]) + 1e-9
    with pytest.raises(ValueError, match="reproduc|deviation|residual|identity"):
        decomposition_artifact(
            chunks=chunks, committed=_committed(_chunks()), declared=_declared()
        )


def test_a_refused_report_writes_nothing_and_creates_no_directory(tmp_path: Path) -> None:
    """🔒 The barrier itself, through the real CLI: validation precedes every byte.

    ``CLAUDE.md`` section 5's filesystem-mutation barrier has shipped broken twice in this project.
    A refusal must leave the destination absent, not truncated and not partially written.
    """
    work = tmp_path / "output" / "p5_3b_decomp"
    data = tmp_path / "docs" / "data"
    work.mkdir(parents=True)
    data.mkdir(parents=True)
    chunks = _chunks()
    chunks[0]["episodes"][0]["att_ours"] += 1e-9
    for chunk in chunks:
        (work / chunk_name(chunk["method"], chunk["tier"], chunk["seed"])).write_text(
            json.dumps(chunk), encoding="utf-8"
        )
    _write_committed_sources(tmp_path, chunks)

    destination = data / "p5_3b_decomposition.json"
    # ``match=`` matters here beyond hygiene: without it a ValueError raised by a typo in this
    # test's own sandbox would masquerade as the refusal under test, and the assertion below
    # ("nothing was written") would pass for the wrong reason.
    # ⚠️ ``--allow-dirty`` is not laziness: without it this test's outcome depends on whether the
    # REPO's working tree happens to be clean when pytest runs, because `report` refuses a dirty
    # tree before it reaches the check under test. A test whose result depends on uncommitted files
    # elsewhere is not deterministic, and determinism is a feature here (CLAUDE.md section 2). The
    # tree guard has its own test below.
    with pytest.raises(ValueError, match="committed|reproduc"):
        nortg_decomposition.main(
            [
                "--output-root", str(tmp_path / "output"),
                "--work-dir", str(work),
                "--out-dir", str(data),
                "--allow-dirty",
                "report",
            ]
        )
    assert not destination.exists(), "a refused report must leave no artifact behind"


def test_the_report_refuses_a_tree_it_cannot_vouch_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``report`` writes a COMMITTED artifact carrying ``runtime.git_commit``.

    ``nortg_campaign._run_report`` is exempt from this check and that exemption is review finding
    MJ-7; ``BRIEF_33`` section 3.5 says there is no reason to repeat it. A commit that did not
    produce the bytes is a provenance claim that is simply false.
    """
    monkeypatch.setattr(
        nortg_campaign,
        "runtime_provenance",
        lambda *a, **k: {"git_commit": "deadbeef" * 5, "git_dirty": True},
    )
    with pytest.raises(ValueError, match="modified|undetermined"):
        nortg_decomposition.main(
            [
                "--output-root", str(tmp_path / "output"),
                "--work-dir", str(tmp_path / "output" / "p5_3b_decomp"),
                "--out-dir", str(tmp_path),
                "report",
            ]
        )


def _write_committed_sources(root: Path, chunks: Sequence[Mapping[str, Any]]) -> None:
    """A minimal but REAL set of committed sources: the campaign artifact and P8.4b's cells.

    Written rather than monkeypatched, so the end-to-end test exercises the same readers production
    uses -- including ``rederived_dt_episodes``' own check against the committed grids.
    """
    data = root / "docs" / "data"
    data.mkdir(parents=True, exist_ok=True)
    committed = _committed(chunks)
    nortg_rows = [
        {
            "arm": f"dt_nortg@{tier}", "tier": tier, "method": "dt_nortg",
            "seed": seed, "draw_id": draw,
            "att_engine": committed[("dt_nortg", tier, seed, draw)]["att_engine"],
            "att_ours": committed[("dt_nortg", tier, seed, draw)]["att_ours"],
        }
        for tier in TIERS for seed in TRAINING_SEEDS for draw in HELD_OUT_DRAWS
    ]
    (data / "p5_3b_nortg.json").write_text(
        json.dumps(
            {
                "format_version": nortg_campaign.ARTIFACT_FORMAT_VERSION,
                "episodes": nortg_rows,
                "comparisons": {
                    tier: {
                        "by_definition": {
                            "att_engine": {"paired": {"mean_difference": _DECLARED_ENGINE}},
                            "att_ours": {"paired": {"mean_difference": _DECLARED_OURS}},
                        }
                    }
                    for tier in TIERS
                },
            }
        ),
        encoding="utf-8",
    )
    for artifact in ("p4_6_grid.json", "p4_7_grid.json"):
        (data / artifact).write_text(
            json.dumps(
                {
                    "episodes": [
                        {
                            "arm": f"dt@{tier}", "seed": seed, "draw_id": draw,
                            "att_horizon": committed[("dt", tier, seed, draw)]["att_ours"],
                        }
                        for tier in TIERS for seed in TRAINING_SEEDS for draw in HELD_OUT_DRAWS
                    ]
                }
            ),
            encoding="utf-8",
        )
    rederived = root / "output" / nortg_campaign.REDERIVATION_DIRNAME
    rederived.mkdir(parents=True, exist_ok=True)
    for tier in TIERS:
        for seed in TRAINING_SEEDS:
            for draw in HELD_OUT_DRAWS:
                pair = committed[("dt", tier, seed, draw)]
                (rederived / nortg_campaign.REDERIVED_CELL_TEMPLATE.format(
                    scenario="hz1x1", method="dt", tier=tier, seed=seed, draw=draw
                )).write_text(
                    json.dumps(
                        {
                            "arm": f"dt@{tier}", "tier": tier, "method": "dt",
                            "seed": seed, "draw_id": draw,
                            "att_engine": pair["att_engine"], "att_ours": pair["att_ours"],
                            "reproduces_committed": True,
                        }
                    ),
                    encoding="utf-8",
                )


def test_a_valid_corpus_of_thirty_cells_is_accepted() -> None:
    """The positive control.  A refusal test proves nothing if nothing is ever accepted."""
    payload = decomposition_artifact(
        chunks=_chunks(), committed=_committed(_chunks()), declared=_declared()
    )
    assert payload["format_version"] == nortg_decomposition.ARTIFACT_FORMAT_VERSION
    assert len(payload["episodes"]) == 3000
    assert payload["orientation"] == "att_ours - att_engine"
    assert "negation" in payload["orientation_note"].lower()


# ----------------------------------------------------------------------
# 2. 🔒 The identity is checked, not assumed
# ----------------------------------------------------------------------


def test_stored_terms_that_do_not_sum_to_the_total_are_refused() -> None:
    """🔒 The three terms are STORED, so they can be wrong; the sum is verified against the total."""
    rows = _rows(_chunks())
    rows[11]["term_clock_origin"] = float(rows[11]["term_clock_origin"]) + 0.5
    with pytest.raises(ValueError, match="identity|term"):
        assert_identity_is_exact(rows)


def test_a_tampered_reconstruction_is_caught_by_the_second_route() -> None:
    """🔒 Route two: the derived keys are recomputed from the constructor fields.

    Moving ``att_reference_entered_population`` keeps the STORED terms summing to the STORED total,
    so route one still passes -- and the recomputation from the reconstructions does not.
    """
    rows = _rows(_chunks())
    rows[3]["att_reference_entered_population"] = float(
        rows[3]["att_reference_entered_population"]
    ) + 0.25
    with pytest.raises(ValueError, match="recomput|term|reconstruction"):
        assert_identity_is_exact(rows)


def test_a_self_consistent_corpus_passes_the_identity_check() -> None:
    record = assert_identity_is_exact(_rows(_chunks()))
    assert record["n_episodes"] == 3000
    assert record["residual_max"] == 0.0


# ----------------------------------------------------------------------
# 3. Action recording is inert
# ----------------------------------------------------------------------


def test_the_action_recorder_returns_the_callables_own_object_and_records_a_copy() -> None:
    """Inert by construction: the wrapper returns the inner object itself, and stores a copy.

    The digest is recomputed here by a second route -- ``hashlib`` over
    ``np.asarray(seq, dtype=np.int64).tobytes()`` -- rather than by calling the function under test.
    """
    sequence = [np.array([0, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64),
                np.array([0, 0], dtype=np.int64)]
    returned: list[np.ndarray] = []

    def inner(_env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
        step = iter(sequence)

        def choose(_e: Any, _info: dict[str, Any]) -> np.ndarray:
            value = next(step)
            returned.append(value)
            return value

        return choose

    factory, recorder = recording_factory(inner)
    env = _StubEnv(n_actions=[2, 2])
    choose = factory(env)
    outputs = [choose(env, {}) for _ in sequence]

    for produced, expected in zip(outputs, sequence):
        assert produced is expected, "the wrapper must return the inner callable's own object"
        assert np.array_equal(produced, expected)
        assert produced.dtype == expected.dtype

    summary = recorder.summary()
    assert summary["n_decisions"] == 3
    expected_digest = hashlib.sha256(
        np.asarray(sequence, dtype=np.int64).tobytes()
    ).hexdigest()
    assert summary["action_sequence_sha256"] == expected_digest
    assert summary["action_sequence_sha256"] == action_sequence_sha256(sequence)
    # intersection 0 played action 0 twice and action 1 once; intersection 1 the reverse
    assert summary["action_counts"] == [[2, 1], [1, 2]]


def test_the_action_histogram_is_as_wide_as_the_env_declares_not_as_wide_as_the_data() -> None:
    """A collapsed policy must produce a full-width histogram of zeros, not a narrower one.

    ⚠️ ``max(observed) + 1`` would make a degenerate arm look like a different environment, and this
    module exists partly to describe a degenerate arm.
    """
    factory, recorder = recording_factory(
        lambda _env: (lambda _e, _i: np.array([0], dtype=np.int64))
    )
    env = _StubEnv(n_actions=[4])
    choose = factory(env)
    for _ in range(5):
        choose(env, {})
    assert recorder.summary()["action_counts"] == [[5, 0, 0, 0]]


class _StubEnv:
    """Just enough env for ``Utils.infer_action_counts`` to answer."""

    def __init__(self, n_actions: Sequence[int]) -> None:
        self.action_space = None
        self.intersections = [_StubIntersection(n) for n in n_actions]


class _StubIntersection:
    def __init__(self, num_phases: int) -> None:
        self.num_phases = num_phases
        self.id = f"ix{num_phases}"


# ----------------------------------------------------------------------
# 4. The fence
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "p5_3b/x.json",
        "p5_3b/checkpoints/mix50_dt_nortg_seed101.pt",
        "p8_4b_rederivation/cell_x.json",
        "p4_dt/dt_seed101.pt",
        "p4_6/checkpoints/x.pt",
        "never_seen/x.json",
        "SHA256SUMS_p5_3b.txt",
    ],
)
def test_the_fence_refuses_everything_under_output_that_is_not_this_task(
    tmp_path: Path, relative: str
) -> None:
    """Default-deny, including this campaign's own PREDECESSOR.

    ``output/p5_3b`` is the tree this task READS -- the ablated checkpoints and P5.3b's chunks -- and
    it must never be written.  A string-prefix implementation cannot tell ``p5_3b`` from
    ``p5_3b_decomp``, which is precisely the trap here.
    """
    with pytest.raises(ValueError, match="another campaign|read-only|belongs"):
        assert_decomposition_writable(tmp_path / "output" / relative)


@pytest.mark.parametrize(
    "relative", ["p5_3b_decomp/decomp_dt_mix50_seed101.json", "SHA256SUMS_p5_3b_decomp.txt",
                 "p5_3b_decomp/smoke/decomp_dt_mix50_seed101.json"]
)
def test_the_fence_allows_exactly_this_tasks_own_outputs(tmp_path: Path, relative: str) -> None:
    """Positive control: a fence that refuses everything is not a fence, it is a wall."""
    target = tmp_path / "output" / relative
    assert assert_decomposition_writable(target) == target


# ----------------------------------------------------------------------
# 5. 🔒 The contrast explains the difference the campaign already reported
# ----------------------------------------------------------------------


def test_the_measured_terms_must_explain_the_committed_definition_difference() -> None:
    """🔒 Ties BL-2(b)'s measurement to the headline it decomposes.

    Without this the module could emit three internally consistent terms of some OTHER quantity and
    nothing would notice.
    """
    payload = decomposition_artifact(
        chunks=_chunks(), committed=_committed(_chunks()), declared=_declared()
    )
    for tier in TIERS:
        contrast = payload["summary"]["contrast"][tier]
        assert contrast["delta_ours_minus_delta_engine"] == _EXPECTED_IDENTITY_GAP
        recomputed = sum(contrast[key] for key in nortg_decomposition.TERM_KEYS)
        assert abs(recomputed - _EXPECTED_IDENTITY_GAP) < CONTRAST_TOLERANCE
        assert abs(contrast["identity_gap"]) < CONTRAST_TOLERANCE


def test_a_contrast_that_does_not_explain_the_declared_difference_is_refused() -> None:
    declared = _declared()
    declared["mix50"] = {"att_engine": _DECLARED_ENGINE, "att_ours": _DECLARED_OURS + 1e-3}
    with pytest.raises(ValueError, match="identity|decomposition|explain"):
        decomposition_artifact(
            chunks=_chunks(), committed=_committed(_chunks()), declared=declared
        )


def test_a_discrepancy_inside_the_tolerance_is_accepted_because_summation_order_differs() -> None:
    """⚠️ ``==`` is the wrong test at tier level and the docstring says why.

    The two sides reduce the same 3,000 floats in different orders.  A tolerance here is not a
    loosening; the PER-EPISODE identity is still checked exactly, and that is the strong claim.
    """
    declared = _declared()
    declared["mix50"] = {"att_engine": _DECLARED_ENGINE, "att_ours": _DECLARED_OURS + 1e-12}
    payload = decomposition_artifact(
        chunks=_chunks(), committed=_committed(_chunks()), declared=declared
    )
    assert payload["summary"]["contrast"]["mix50"]["identity_gap"] != 0.0


# ----------------------------------------------------------------------
# 6. Completeness -- "compared nothing" must never read as "found no differences"
# ----------------------------------------------------------------------


def test_twenty_nine_cells_are_refused() -> None:
    chunks = _chunks()[:-1]
    with pytest.raises(ValueError, match="29|thirty|30|incomplete"):
        assert_chunks_complete(chunks)


def test_a_cell_missing_one_draw_is_refused() -> None:
    chunks = _chunks()
    chunks[4]["episodes"] = chunks[4]["episodes"][:-1]
    chunks[4]["draws"] = chunks[4]["draws"][:-1]
    with pytest.raises(ValueError, match="99|draw|incomplete"):
        assert_chunks_complete(chunks)


def test_a_duplicated_cell_is_refused() -> None:
    chunks = _chunks()
    chunks[-1] = _chunk(chunks[0]["method"], chunks[0]["tier"], chunks[0]["seed"])
    with pytest.raises(ValueError, match="duplicate|29|missing"):
        assert_chunks_complete(chunks)


def test_thirty_complete_cells_are_accepted() -> None:
    record = assert_chunks_complete(_chunks())
    assert record["n_cells"] == 30
    assert record["n_episodes"] == 3000


# ----------------------------------------------------------------------
# 6b. 🔒 A chunk may not vouch for itself
# ----------------------------------------------------------------------


def test_a_chunk_that_agrees_with_itself_but_not_with_the_committed_column_is_refused() -> None:
    """🔒 The load-bearing design decision (``BRIEF_33`` AMENDMENT A4).

    Here ``att_engine_call`` and ``committed_att_engine`` are moved TOGETHER, so the chunk is
    perfectly self-consistent and its own ``reproduces_committed`` is ``true``.  It still has to
    agree with ``docs/data/p5_3b_nortg.json`` and with P8.4b's re-derived cells, and it does not.
    P8.4b's ``reproduces_committed`` was read as evidence once already.
    """
    chunks = _chunks()
    row = chunks[2]["episodes"][17]
    row["att_engine_call"] = float(row["att_engine_call"]) + 1e-9
    row["committed_att_engine"] = float(row["committed_att_engine"]) + 1e-9
    assert row["reproduces_committed"] is True
    with pytest.raises(ValueError, match="committed"):
        assert_rows_reproduce_committed(_rows(chunks), committed=_committed(_chunks()))


@pytest.mark.parametrize("field", ["att_engine_call", "att_ours"])
def test_a_chunk_inconsistent_with_its_own_committed_pair_is_refused_by_the_reproduction_guard(
    field: str,
) -> None:
    """🔒 The reproduction guard, exercised DIRECTLY rather than through the pipeline.

    ⚠️ Written after a mutation survived: deleting the ``att_engine_call ==
    committed_att_engine`` equality left the end-to-end test green, because
    :func:`assert_identity_is_exact` reached the same tampered row first and refused for its own
    reason.  A test that passes because a *different* guard fired does not pin the guard it names,
    and the only way to tell the two apart is to call the guard on its own.
    """
    chunks = _chunks()
    row = chunks[7]["episodes"][42]
    row[field] = float(row[field]) + 1e-9
    with pytest.raises(ValueError, match="reproduce the committed cell"):
        assert_rows_reproduce_committed(_rows(chunks), committed=_committed(_chunks()))


def test_the_reproduction_check_does_not_read_the_chunks_own_flag() -> None:
    """A row flagged ``reproduces_committed: false`` that DOES reproduce is accepted on the numbers.

    The flag is a convenience for the driver, never evidence; this pins that it carries no weight.
    """
    chunks = _chunks()
    chunks[0]["episodes"][0]["reproduces_committed"] = False
    record = assert_rows_reproduce_committed(_rows(chunks), committed=_committed(_chunks()))
    assert record["n_episodes"] == 3000
    assert record["n_reproducing"] == 3000


# ----------------------------------------------------------------------
# 6c. The skip predicate -- a bad chunk must not survive a restart
# ----------------------------------------------------------------------


def test_a_complete_and_clean_chunk_may_be_skipped() -> None:
    assert chunk_is_reusable(
        _chunk("dt", "mix50", 101), method="dt", tier="mix50", seed=101
    ) is True


@pytest.mark.parametrize(
    ("why", "over"),
    [
        ("a partial chunk", {"draws": list(HELD_OUT_DRAWS)[:50], "is_complete": False}),
        ("a chunk with a mismatch", {"n_mismatches": 1}),
        ("a chunk from another format version", {"format_version": "p5.3b-decomposition/0.9"}),
        ("a chunk with a non-numeric mismatch count", {"n_mismatches": "none"}),
    ],
)
def test_a_partial_or_mismatching_chunk_is_re_run_not_skipped(why: str, over: dict[str, Any]) -> None:
    """``p5_3b.sh`` skipped on ``[ -f ]`` alone and ``assert_probe_cell_is_ablated``'s docstring
    records the cost: *"a bad chunk survived every restart."*"""
    draws = over.pop("draws", None)
    chunk = _chunk("dt", "mix50", 101, draws=draws, **over)
    assert chunk_is_reusable(chunk, method="dt", tier="mix50", seed=101) is False, why


@pytest.mark.parametrize(
    ("why", "chunk_cell"),
    [
        ("another arm entirely", ("dt_nortg", "random", 505)),
        ("the right arm, the wrong tier", ("dt", "random", 101)),
        ("the right cell but the wrong seed", ("dt", "mix50", 202)),
        ("the right tier and seed, the wrong method", ("dt_nortg", "mix50", 101)),
    ],
)
def test_a_chunk_for_another_cell_under_the_right_filename_is_re_run(
    why: str, chunk_cell: tuple[str, str, int]
) -> None:
    """🔒 The pre-flight's **M1**, which was a LIVELOCK rather than a corruption.

    A chunk describing ``dt_nortg@random`` seed 505 saved as ``decomp_dt_mix50_seed101.json`` was
    skipped as *"complete and clean"*.  ``report`` still refused the campaign, so it could never
    reach the artifact -- and every restart skipped it again, so the run could never finish either.
    **A predicate that reads only the filename trusts the filename.**
    """
    chunk = _chunk(*chunk_cell)
    assert chunk_is_reusable(chunk, method="dt", tier="mix50", seed=101) is False, why


def test_a_chunk_whose_header_is_right_but_whose_rows_are_another_cells_is_re_run() -> None:
    """The header alone can be edited; the rows are what the artifact would actually carry."""
    chunk = _chunk("dt", "mix50", 101)
    chunk["episodes"][17] = _episode("dt", "random", 101, 1017)
    assert chunk_is_reusable(chunk, method="dt", tier="mix50", seed=101) is False


@pytest.mark.parametrize(
    ("why", "contents"),
    [
        ("a truncated chunk", '{"format_version": "p5.3b-decomposition/1.0", "episo'),
        ("an empty file", ""),
        ("whitespace only", "   \n"),
        ("valid JSON that is not an object", "[1, 2, 3]"),
    ],
)
def test_a_chunk_that_does_not_parse_is_re_run_and_never_crashes(
    tmp_path: Path, why: str, contents: str
) -> None:
    """🔒 The pre-flight's **M5**.  ``json.loads`` ran BEFORE the predicate, so a truncated chunk
    raised ``JSONDecodeError``, the driver wrote ``FAILED``, and every restart hit the same crash --
    while the docstring promised a partial chunk would be re-run.  A file we cannot read is a file we
    have no evidence about, and the only safe reading of no evidence is *roll it again*."""
    path = tmp_path / chunk_name("dt", "mix50", 101)
    path.write_text(contents, encoding="utf-8")
    assert reusable_chunk_at(path, method="dt", tier="mix50", seed=101) is False, why


def test_a_chunk_that_is_not_there_at_all_is_re_run(tmp_path: Path) -> None:
    assert reusable_chunk_at(
        tmp_path / chunk_name("dt", "mix50", 101), method="dt", tier="mix50", seed=101
    ) is False


def test_a_good_chunk_on_disk_is_still_skippable(tmp_path: Path) -> None:
    """Positive control: the M5 hardening must not make everything unskippable."""
    path = tmp_path / chunk_name("dt", "mix50", 101)
    path.write_text(json.dumps(_chunk("dt", "mix50", 101)), encoding="utf-8")
    assert reusable_chunk_at(path, method="dt", tier="mix50", seed=101) is True


@pytest.mark.parametrize(
    "relative", ["p5_3b", "p8_4b_rederivation", "p4_dt", "never_seen"]
)
def test_the_report_refuses_to_write_the_artifact_into_another_campaigns_tree(
    tmp_path: Path, relative: str
) -> None:
    """🔒 The pre-flight's **M4**: the fence was applied to ``--work-dir`` only.

    ``report --out-dir <root>/output/p5_3b`` wrote ``p5_3b_decomposition.json`` straight into the
    predecessor's tree -- the one the module docstring says it must never write.  The driver
    hardcodes a safe ``--out-dir``, so no live path reached it; but *"every other component under
    output/ is refused"* was not true of this path, and a fence with an undocumented exception is
    not a fence.
    """
    with pytest.raises(ValueError, match="another campaign|read-only|belongs"):
        nortg_decomposition.main(
            [
                "--output-root", str(tmp_path / "output"),
                "--work-dir", str(tmp_path / "output" / "p5_3b_decomp"),
                "--out-dir", str(tmp_path / "output" / relative),
                "--allow-dirty",
                "report",
            ]
        )


def test_the_report_accepts_the_docs_data_destination_it_actually_uses(tmp_path: Path) -> None:
    """Positive control for M4: the real destination must still be allowed.

    It gets past the fence and fails later, on the absent chunks -- which is the proof that the
    fence let it through rather than that nothing was checked.
    """
    with pytest.raises(FileNotFoundError, match="needs every cell chunk"):
        nortg_decomposition.main(
            [
                "--output-root", str(tmp_path / "output"),
                "--work-dir", str(tmp_path / "output" / "p5_3b_decomp"),
                "--out-dir", str(tmp_path / "docs" / "data"),
                "--allow-dirty",
                "report",
            ]
        )


# ----------------------------------------------------------------------
# 6d. 🔒 A guard nobody calls is not a guard (DEFERRED 63)
# ----------------------------------------------------------------------


def test_the_report_actually_calls_the_decomposition_guard() -> None:
    """🔒 ``BRIEF_33`` AMENDMENT A2: pin the CALL SITE, not only the guard.

    ``DEFERRED`` 63 (P8.4b MJ-1/MN-2) is a guard that is defined, tested in isolation and never
    invoked -- every one of its own tests passes while it protects nothing.  Deleting the call from
    ``_run_report`` must fail a test, and this is that test.
    """
    source = inspect.getsource(nortg_campaign._run_report)
    assert "assert_decomposition_embedded" in source
    assert callable(nortg_campaign.assert_decomposition_embedded)


def _embedded_block(**over: Any) -> dict[str, Any]:
    """One well-formed ``definition_difference_decomposition`` block."""
    arm = {"population": 3.0, "clock_origin": 95.5, "cadence": 6.0, "total": 104.5}
    block = {
        "source": {"path": "docs/data/p5_3b_decomposition.json", "sha256": "b" * 64},
        "orientation": "att_ours - att_engine",
        "orientation_note": "the negation of A13(b)'s writing",
        "identity": "att_ours - att_engine = population + clock_origin + cadence",
        "n_episodes_per_arm": 500,
        "per_arm": {"dt": dict(arm), "dt_nortg": dict(arm)},
        "contrast": dict(arm),
        "delta_ours_minus_delta_engine": 104.5,
        "identity_gap": 0.0,
        "residual_max": 0.0,
        "residual_is_not_independent": "corroborates criterion 1 and nothing further",
    }
    block.update(over)
    return block


def _payload_with_blocks(**per_tier: Any) -> dict[str, Any]:
    return {"comparisons": {tier: {"definition_difference_decomposition": block}
                            for tier, block in per_tier.items()}}


def test_a_report_missing_the_decomposition_on_any_tier_is_refused() -> None:
    """🔒 A13(b) makes it REQUIRED, so a tier without it is a refusal and not a smaller artifact."""
    payload = _payload_with_blocks(**{tier: _embedded_block() for tier in TIERS})
    del payload["comparisons"]["mix50"]["definition_difference_decomposition"]
    with pytest.raises(ValueError, match="REQUIRED reported quantity"):
        nortg_campaign.assert_decomposition_embedded(payload)


def test_an_embedded_block_whose_terms_do_not_sum_to_its_total_is_refused() -> None:
    """A block that does not close its own identity decomposes nothing."""
    blocks = {tier: _embedded_block() for tier in TIERS}
    blocks["random"]["contrast"]["cadence"] = 6.5
    with pytest.raises(ValueError, match="identity"):
        nortg_campaign.assert_decomposition_embedded(_payload_with_blocks(**blocks))


def test_a_well_formed_embedded_decomposition_is_accepted() -> None:
    record = nortg_campaign.assert_decomposition_embedded(
        _payload_with_blocks(**{tier: _embedded_block() for tier in TIERS})
    )
    assert record["n_tiers"] == len(TIERS)
