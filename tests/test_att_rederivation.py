"""Tests for ``offline/att_rederivation.py`` -- P8.4b's re-derivation campaign.

⭐ **The multi-worker tests here exist because a one-worker smoke test cannot exercise a multi-worker
guard.** The first launch of this campaign stopped four of five workers: each declared its own shard
as the campaign manifest, the digests genuinely disagreed, and ``assert_manifest_agrees`` refused --
correctly. Six cells rolled by a single worker never disagree with themselves, so nothing before the
real launch could have found it. Every test below that says ``multi_worker`` is that lesson.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from offline.att_rederivation import (
    ARTIFACT_FORMAT_VERSION,
    BEHAVIOUR_METHOD,
    MIXTURE_TIERS,
    CellKey,
    assert_manifest_agrees,
    assert_reproduces_committed,
    campaign_manifest,
    campaign_shard,
    campaign_status,
    cell_file_name,
    is_constructed_reference,
    normalise_arm,
    shard_cells,
)


def cells(n: int = 20, scenario: str = "hz1x1") -> list[CellKey]:
    """A synthetic campaign of *n* cells across two arms."""
    return [
        CellKey(scenario=scenario, arm=f"bc@tier{i % 2}", seed=101, draw_id=1000 + i)
        for i in range(n)
    ]


# ----------------------------------------------------------------------
# Sharding
# ----------------------------------------------------------------------


def test_the_shards_partition_the_campaign_exactly_once() -> None:
    """Every cell is rolled by exactly one worker: no duplicates, no gaps."""
    campaign = cells(37)
    shards = [campaign_shard(campaign, shard=i, of=5) for i in range(5)]
    seen: list[CellKey] = [c for s in shards for c in s]
    assert sorted(seen, key=repr) == sorted(campaign, key=repr)
    assert len(seen) == len(set(seen)) == 37
    # No worker is starved and none carries more than one extra.
    sizes = sorted(len(s) for s in shards)
    assert sizes[-1] - sizes[0] <= 1


def test_the_partition_does_not_depend_on_input_order() -> None:
    """Two workers started minutes apart must partition identically.

    ``shard_cells`` sorts before slicing, so the split is a property of the SET. Handing it a
    reversed list -- which is what a differently-ordered ``set`` iteration looks like -- must not
    move a single cell between workers, or two workers would duplicate and miss work.
    """
    campaign = cells(23)
    for index in range(3):
        forward = campaign_shard(campaign, shard=index, of=3)
        backward = campaign_shard(list(reversed(campaign)), shard=index, of=3)
        assert forward == backward


def test_shard_cells_refuses_an_invalid_partition() -> None:
    with pytest.raises(ValueError, match="is not a valid partition"):
        shard_cells(cells(4), shard=5, of=5)
    with pytest.raises(ValueError, match="is not a valid partition"):
        shard_cells(cells(4), shard=0, of=0)


# ----------------------------------------------------------------------
# THE MULTI-WORKER MANIFEST -- the defect that stopped the first launch
# ----------------------------------------------------------------------


def test_multi_worker_manifests_agree_because_the_manifest_is_campaign_scoped() -> None:
    """🔒 THE REGRESSION TEST FOR THE FIRST LAUNCH FAILURE.

    Five workers each build a manifest from the SAME campaign cell set and then take their own
    shard. Every manifest must be byte-identical, because the manifest declares the campaign and not
    the worker. Under the original code each worker built its manifest from its own shard, so five
    workers produced five different digests and four of them were refused.

    The test asserts the shards are genuinely DIFFERENT at the same time, so it cannot pass by the
    degenerate route of every worker rolling the same thing.
    """
    campaign = cells(37)
    manifests = [campaign_manifest(campaign, engine_seed=1000) for _ in range(5)]
    shards = [campaign_shard(campaign, shard=i, of=5) for i in range(5)]

    digests = {m["declared_cells_sha256"] for m in manifests}
    assert len(digests) == 1, "five workers produced more than one campaign digest"
    assert all(m["n_cells"] == 37 for m in manifests), [m["n_cells"] for m in manifests]

    # ...and the workers really are doing different work.
    assert len({tuple(repr(c) for c in s) for s in shards}) == 5
    assert all(len(s) < 37 for s in shards)


def test_multi_worker_launch_does_not_refuse_the_second_worker(tmp_path: Path) -> None:
    """Worker 0 writes the manifest; workers 1-4 must be accepted by it.

    This is the failure the author hit, reproduced at the level of the two functions that produced
    it: a manifest written from the campaign is accepted by every other worker, and one written from
    a shard is not.
    """
    campaign = cells(37)
    path = tmp_path / "campaign_manifest.json"

    # Worker 0 declares the campaign.
    first = campaign_manifest(campaign, engine_seed=1000)
    path.write_text(json.dumps(first), encoding="utf-8")

    # Workers 1-4 each recompute it and must be accepted.
    for index in range(1, 5):
        _ = campaign_shard(campaign, shard=index, of=5)
        assert_manifest_agrees(path, campaign_manifest(campaign, engine_seed=1000))

    # The original defect, pinned so it cannot come back: a SHARD-scoped manifest is refused.
    shard_scoped = campaign_manifest(campaign_shard(campaign, shard=1, of=5), engine_seed=1000)
    with pytest.raises(ValueError, match="different campaigns"):
        assert_manifest_agrees(path, shard_scoped)


def test_a_manifest_from_a_different_verdict_selection_is_refused(tmp_path: Path) -> None:
    """DEFERRED 58: two campaigns must never share a work directory."""
    path = tmp_path / "campaign_manifest.json"
    path.write_text(json.dumps(campaign_manifest(cells(20), engine_seed=1000)), encoding="utf-8")
    with pytest.raises(ValueError, match="different campaigns"):
        assert_manifest_agrees(path, campaign_manifest(cells(30), engine_seed=1000))


def test_assert_manifest_agrees_accepts_a_fresh_work_directory(tmp_path: Path) -> None:
    """No manifest yet is the first worker's normal case and must not raise."""
    assert_manifest_agrees(tmp_path / "absent.json", campaign_manifest(cells(5), engine_seed=1000))


# ----------------------------------------------------------------------
# Completeness: a partial run must not read as a finished one
# ----------------------------------------------------------------------


def test_status_is_computed_from_disk_and_a_partial_run_is_not_complete(tmp_path: Path) -> None:
    """``complete`` is recomputed every time it is asked, never a claim a run makes about itself."""
    campaign = cells(10)
    manifest = campaign_manifest(campaign, engine_seed=1000)
    (tmp_path / "campaign_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert campaign_status(tmp_path)["complete"] is False
    for cell in campaign[:4]:
        (tmp_path / cell_file_name(cell)).write_text("{}", encoding="utf-8")
    partial = campaign_status(tmp_path)
    assert partial["declared"] == 10
    assert partial["present"] == 4
    assert partial["missing"] == 6
    assert partial["complete"] is False
    assert partial["marker_present"] is False

    for cell in campaign[4:]:
        (tmp_path / cell_file_name(cell)).write_text("{}", encoding="utf-8")
    done = campaign_status(tmp_path)
    assert done["present"] == 10 and done["missing"] == 0 and done["complete"] is True


def test_status_without_a_manifest_is_not_complete(tmp_path: Path) -> None:
    """An empty directory must never read as a finished campaign."""
    state = campaign_status(tmp_path)
    assert state["complete"] is False
    assert state["declared"] is None


def test_a_stray_marker_does_not_make_an_incomplete_run_complete(tmp_path: Path) -> None:
    """The marker is reported, never trusted: completeness comes from the declared set."""
    campaign = cells(6)
    (tmp_path / "campaign_manifest.json").write_text(
        json.dumps(campaign_manifest(campaign, engine_seed=1000)), encoding="utf-8"
    )
    (tmp_path / "CAMPAIGN_COMPLETE").write_text("{}", encoding="utf-8")
    state = campaign_status(tmp_path)
    assert state["marker_present"] is True
    assert state["complete"] is False, "a marker must not override the files on disk"


# ----------------------------------------------------------------------
# The acceptance test
# ----------------------------------------------------------------------


def test_reproduction_refuses_a_cell_that_does_not_match_the_committed_value() -> None:
    """A wrong arm-to-checkpoint entry loads an INTACT file for a different policy."""
    cell = CellKey(scenario="hz1x1", arm="madt", seed=101, draw_id=1000)
    committed = {("hz1x1", "madt", 101, 1000): 106.4561500275786}
    assert assert_reproduces_committed(cell, 106.4561500275786, committed) == 106.4561500275786
    with pytest.raises(ValueError, match="does not reproduce the committed"):
        assert_reproduces_committed(cell, 104.80143408714838, committed)


def test_a_cell_with_no_committed_value_is_unverified_and_excluded() -> None:
    """No committed ``att_ours`` at any grain means the cell may not enter a verdict."""
    cell = CellKey(scenario="hz1x1", arm="madt", seed=101, draw_id=9999)
    with pytest.raises(KeyError, match="UNVERIFIED"):
        assert_reproduces_committed(cell, 100.0, {})


def test_reproduction_is_exact_not_approximate() -> None:
    """P8.4a reproduced 39/39 and 14/14 cells exactly through this path; a tolerance is slack."""
    cell = CellKey(scenario="hz1x1", arm="madt", seed=101, draw_id=1000)
    committed = {("hz1x1", "madt", 101, 1000): 100.0}
    with pytest.raises(ValueError, match="does not reproduce the committed"):
        assert_reproduces_committed(cell, 100.0 + 1e-12, committed)


# ----------------------------------------------------------------------
# Naming and the constructed references
# ----------------------------------------------------------------------


def test_normalise_arm_absorbs_the_three_conventions_and_refuses_a_fourth() -> None:
    assert normalise_arm("hz1x1", "bc@mappo1000") == ("bc", "mappo1000")
    assert normalise_arm("grid4x4", "bc@grid4x4_mappo1000") == ("bc", "mappo1000")
    assert normalise_arm("hz1x1", "madt") == ("dt", "mappo1000")
    assert normalise_arm("hz1x1", "mappo1000") == (BEHAVIOUR_METHOD, "mappo1000")
    with pytest.raises(ValueError, match="carries no tier"):
        normalise_arm("hz1x1", "not_a_declared_bare_name")


def test_a_mixture_tiers_behaviour_reference_is_never_rolled() -> None:
    """Re-rolling one would replace a CONSTRUCTED reference with a measured number."""
    for tier in sorted(MIXTURE_TIERS):
        assert is_constructed_reference(
            CellKey(scenario="hz1x1", arm=f"behaviour@{tier}", seed=101, draw_id=1000)
        )
    # Only the BEHAVIOUR arm of a mixture tier is constructed; its learned arms are rolled.
    assert not is_constructed_reference(
        CellKey(scenario="hz1x1", arm="bc@mix33", seed=101, draw_id=1000)
    )
    assert not is_constructed_reference(
        CellKey(scenario="hz1x1", arm="behaviour@mappo1000", seed=101, draw_id=1000)
    )


def test_cell_file_names_are_unique_per_episode() -> None:
    """One file per episode is what makes the campaign resumable at episode granularity."""
    names = {
        cell_file_name(CellKey(scenario="hz1x1", arm="bc@mix33", seed=s, draw_id=d))
        for s in (101, 202)
        for d in (1000, 1001)
    }
    assert len(names) == 4
    assert all(n.startswith("cell_") and n.endswith(".json") for n in names)
    # '@' must not survive into a filename.
    assert all("@" not in n for n in names)


def test_the_manifest_records_the_format_version_and_the_engine_seed() -> None:
    manifest = campaign_manifest(cells(3), engine_seed=4242)
    assert manifest["format_version"] == ARTIFACT_FORMAT_VERSION
    assert manifest["engine_seed"] == 4242
    assert manifest["n_cells"] == 3
    assert len(manifest["cells"]) == 3


# ----------------------------------------------------------------------
# END TO END, THROUGH main() -- because the defect lived in the WIRING
# ----------------------------------------------------------------------


def _campaign_inputs_present() -> bool:
    """Whether the artifacts every verdict source names are on disk."""
    repo = Path(__file__).resolve().parents[1]
    output = Path("/home/filip/rltraffic/output")
    return (repo / "docs/data/p4_gate.json").is_file() and (output / "p5_1").is_dir()


def test_multi_worker_launch_through_main_agrees_on_the_manifest(tmp_path: Path) -> None:
    """🔒 THE TEST THAT WOULD ACTUALLY HAVE CAUGHT THE FIRST LAUNCH FAILURE.

    ⚠️ **The seventeen unit tests above do NOT catch it, and that was verified by re-introducing the
    defect: they all passed.** The functions were each correct in isolation; the bug was in how
    ``main`` COMPOSED them -- it sharded before building the manifest. A test over the pieces cannot
    see a wiring defect, so this one drives ``main`` itself, twice, as two different workers.

    ``--limit 0`` means no episode is rolled, so the test is fast and still exercises the exact path
    that failed: build the cell set, declare the manifest, check it against what is already there.
    """
    from offline.att_rederivation import main

    if not _campaign_inputs_present():
        pytest.skip(
            "the campaign's source artifacts are not present (docs/data/p4_gate.json and "
            "output/p5_1); this test resolves every verdict source"
        )

    work = tmp_path / "work"
    common = [
        "--repo-root", str(Path(__file__).resolve().parents[1]),
        "--corpus-root", "/home/filip/rltraffic/datasets_v11",
        "--draws-root", "/home/filip/rltraffic/scenarios/draws",
        "--output-root", "/home/filip/rltraffic/output",
        "--work-dir", str(work),
    ]
    for shard in range(3):
        # Each worker must get past assert_manifest_agrees.  Under the original wiring the second
        # one raised ValueError('...different campaigns...') and the campaign could never start.
        code = main([*common, "run", "--verdicts", "V1", "--shard", str(shard), "--of", "3",
                     "--limit", "0"])
        # A WORKER's exit code is about its own slice, not the campaign's completeness: a worker
        # that rolled its cells with no refusals succeeded even though other shards have not run.
        assert code == 0, "a worker with no refusals must report success"

    manifest = json.loads((work / "campaign_manifest.json").read_bytes())
    status = campaign_status(work)
    assert manifest["n_cells"] == status["declared"]
    assert manifest["n_cells"] > 3, "the manifest must declare the campaign, not a 3-cell shard"
    assert status["complete"] is False


# ----------------------------------------------------------------------
# The committed reference: one value per cell, or refuse
# ----------------------------------------------------------------------


def test_a_replicate_artifact_is_recognised_and_never_the_committed_cell() -> None:
    """A replicate is a SECOND measurement of a slot, not the number that was reported."""
    from offline.att_rederivation import is_replicate_artifact

    assert is_replicate_artifact("eval_random_dt_spatial_seed202_replicate.json")
    assert is_replicate_artifact(Path("/x/y/eval_mappo1000_bc_seed303_replicate.json"))
    assert not is_replicate_artifact("eval_random_dt_spatial.json")
    assert not is_replicate_artifact("p4_gate.json")


def test_the_committed_index_refuses_a_cell_with_two_different_values(tmp_path: Path) -> None:
    """🔒 REGRESSION: a silent last-writer-wins cost a four-hour campaign.

    Two artifacts reported different ``att_horizon`` for the same ``(arm, seed, draw)``. The index
    assigned straight into a dict, so the file that sorted later overwrote the committed value with
    a replicate's -- and the campaign then died on an arm whose checkpoint map was CORRECT, with an
    error message blaming the map. Detecting the collision is what turns that into an immediate,
    accurate refusal.
    """
    from offline.att_rederivation import REDERIVATION_SOURCES, committed_att_index

    docs = tmp_path / "docs" / "data"
    docs.mkdir(parents=True)

    def write(name: str, value: float) -> None:
        (docs / name).write_text(
            json.dumps(
                {"episodes": [{"arm": "bc@x", "seed": 101, "draw_id": 1000, "att_horizon": value}]}
            ),
            encoding="utf-8",
        )

    # Both files are named by V1's source list, so both are read for the same scenario.
    write("p4_gate.json", 100.0)
    write("p4_heldout_thresholds.json", 100.0)
    # Agreement is fine and must NOT raise: 1,800 real keys are reported by two files with the
    # same value, and refusing those would refuse the whole campaign.
    only_v1 = tuple(s for s in REDERIVATION_SOURCES if s.verdict_id == "V1")
    import offline.att_rederivation as module

    original = module.REDERIVATION_SOURCES
    module.REDERIVATION_SOURCES = only_v1
    try:
        index = committed_att_index(repo_root=tmp_path, output_root=tmp_path)
        assert index[("hz1x1", "bc@x", 101, 1000)] == 100.0

        # Disagreement must raise, naming the cell and both values.
        write("p4_heldout_thresholds.json", 263.5563114134543)
        with pytest.raises(ValueError, match="MORE THAN ONE committed att_ours"):
            committed_att_index(repo_root=tmp_path, output_root=tmp_path)
    finally:
        module.REDERIVATION_SOURCES = original


def test_a_refusal_record_is_not_a_cell_file_and_so_is_never_skipped() -> None:
    """A REFUSED cell must never be mistaken for data or for completed work.

    ``campaign_status`` counts ``cell_*.json``; a refusal writes ``refused_*.json``, so the campaign
    stays incomplete, the cell is retried on the next resume, and it can never enter a verdict.
    """
    from offline.att_rederivation import refusal_file_name

    cell = CellKey(scenario="grid4x4", arm="bc@fixedtime", seed=202, draw_id=1000)
    assert refusal_file_name(cell).startswith("refused_")
    assert not refusal_file_name(cell).startswith("cell_")
    assert refusal_file_name(cell) != cell_file_name(cell)


def test_status_counts_refusals_separately_from_present_cells(tmp_path: Path) -> None:
    """A refused cell leaves the campaign incomplete and is reported as its own number."""
    from offline.att_rederivation import refusal_file_name

    campaign = cells(4)
    (tmp_path / "campaign_manifest.json").write_text(
        json.dumps(campaign_manifest(campaign, engine_seed=1000)), encoding="utf-8"
    )
    (tmp_path / cell_file_name(campaign[0])).write_text("{}", encoding="utf-8")
    (tmp_path / refusal_file_name(campaign[1])).write_text("{}", encoding="utf-8")
    state = campaign_status(tmp_path)
    assert state["present"] == 1
    assert state["refused"] == 1
    assert state["missing"] == 3, "a refused cell is MISSING, not present"
    assert state["complete"] is False


# ----------------------------------------------------------------------
# DISCRIMINABILITY -- binding, ruled 2026-08-31
# ----------------------------------------------------------------------


def test_identical_arms_report_cannot_discriminate_not_no_difference() -> None:
    """🔒 The binding rule: those are different claims and only one of them is honest here."""
    from offline.att_rederivation import contrast_discriminability

    same = {(202, d): 100.0 + d for d in range(5)}
    got = contrast_discriminability(
        same, dict(same), tier="fixedtime", left="dt_spatial", right="dt_nomix"
    )
    assert got.identical is True
    assert got.status == "cannot_discriminate"
    assert got.n_differing == 0
    assert got.max_abs_difference == 0.0
    assert "cannot discriminate" in got.statement
    assert "no difference was found" not in got.statement


def test_distinct_arms_report_distinct_with_their_margin() -> None:
    from offline.att_rederivation import contrast_discriminability

    left = {(202, d): 100.0 + d for d in range(5)}
    right = dict(left)
    right[(202, 3)] = 107.5
    got = contrast_discriminability(left, right, tier="random", left="a", right="b")
    assert got.identical is False
    assert got.status == "distinct"
    assert got.n_differing == 1
    assert got.max_abs_difference == pytest.approx(4.5)


def test_a_contrast_over_no_shared_episode_refuses() -> None:
    """Reporting 'identical' from an empty overlap is the strongest claim from no evidence."""
    from offline.att_rederivation import contrast_discriminability

    with pytest.raises(ValueError, match="share no episode"):
        contrast_discriminability({(1, 1): 1.0}, {(2, 2): 1.0}, tier="t", left="a", right="b")


def test_a_verdict_without_its_discriminability_is_refused() -> None:
    """The binding rule is only binding if something checks it."""
    from offline.att_rederivation import assert_contrast_reports_discriminability

    ok = {
        "contrasts": [
            {"tier": "fixedtime", "outcome": "NOT RESOLVED", "discriminability": {"status": "x"}}
        ]
    }
    assert_contrast_reports_discriminability(ok)

    with pytest.raises(ValueError, match="without its discriminability"):
        assert_contrast_reports_discriminability(
            {"contrasts": [{"tier": "fixedtime", "outcome": "NOT RESOLVED"}]}
        )
    # Nested, because a shallow check would miss exactly the one a reader would.
    with pytest.raises(ValueError, match="without its discriminability"):
        assert_contrast_reports_discriminability(
            {"v4": {"blocks": {"p2a": {"tier": "random", "verdict": "HELD"}}}}
        )
    # Every spelling a contrast is written in must trip it, not just "verdict": keying on one
    # spelling is how this guard was silently switched off by a field rename.
    for shape in ({"tier": "t", "outcome": "X"}, {"tier": "t", "pooled": []},
                  {"tier": "t", "contrast_id": "V4"}):
        with pytest.raises(ValueError, match="without its discriminability"):
            assert_contrast_reports_discriminability({"c": [shape]})


def test_the_fixedtime_collapse_is_reproduced_from_the_committed_artifacts() -> None:
    """The finding itself, re-measured from P5.2's own cells rather than quoted.

    Confined to ``fixedtime``: ``mappo1000``, ``maxpressure`` and ``random`` have every arm distinct.
    On ``fixedtime``, ``dt_spatial`` and ``dt_nomix`` are IDENTICAL, which is P5.1's P2a contrast and
    makes it zero by construction on that rung.
    """
    import glob
    from collections import defaultdict

    from offline.att_rederivation import contrast_discriminability

    root = "/home/filip/rltraffic/output/p5_2"
    files = [f for f in sorted(glob.glob(f"{root}/eval_*.json")) if "_replicate" not in f]
    if not files:
        pytest.skip(f"P5.2's campaign cells are not present under {root}")

    by_tier: dict[str, dict[str, dict[tuple, float]]] = defaultdict(dict)
    for path in files:
        payload = json.loads(Path(path).read_bytes())
        arm = payload.get("arm")
        if not arm or "@" not in arm:
            continue
        tier = arm.split("@")[1]
        for episode in payload.get("episodes", []):
            by_tier[tier].setdefault(arm, {})[
                (episode["seed"], episode["draw_id"])
            ] = episode["att_horizon"]

    assert by_tier, "no tiers were read -- the check would pass vacuously"
    distinct_counts = {}
    for tier, arms in by_tier.items():
        signatures = {arm: tuple(sorted(v.items())) for arm, v in arms.items()}
        distinct_counts[tier] = (len(set(signatures.values())), len(arms))

    assert distinct_counts["mappo1000"][0] == distinct_counts["mappo1000"][1]
    assert distinct_counts["maxpressure"][0] == distinct_counts["maxpressure"][1]
    assert distinct_counts["random"][0] == distinct_counts["random"][1]
    assert distinct_counts["fixedtime"][0] < distinct_counts["fixedtime"][1], (
        "the fixedtime collapse is the finding; if every arm is distinct it has gone away and the "
        "result note must be corrected"
    )

    # P5.1's P2a contrast, on the rung where it cannot discriminate.
    got = contrast_discriminability(
        by_tier["fixedtime"]["dt_spatial@fixedtime"],
        by_tier["fixedtime"]["dt_nomix@fixedtime"],
        tier="fixedtime",
        left="dt_spatial",
        right="dt_nomix",
    )
    assert got.status == "cannot_discriminate"
    assert got.max_abs_difference == 0.0


# ----------------------------------------------------------------------
# THE VERDICT LAYER: both definitions, both poolings, guard actually wired
# ----------------------------------------------------------------------


def _outcome(mean: float, half: float, low: float, high: float) -> str:
    """P5.1's registered rule, restated here only as a TEST double."""
    if high < 0.0:
        return "HELD"
    if low > 0.0:
        return "FAILED"
    return "NOT RESOLVED"


def _cells_two_tiers(*, collapse: bool, effect: float) -> dict:
    """Two tiers: ``good`` always discriminates; ``flat`` collapses when *collapse* is set."""
    cells: dict = {}
    for tier in ("good", "flat"):
        cells[tier] = {}
        for arm, shift in (("l@" + tier, 0.0), ("r@" + tier, None)):
            cells[tier][arm] = {}
        for i in range(40):
            key = (101, 1000 + i)
            base = 100.0 + (i % 7)
            cells[tier]["l@" + tier][key] = {"att_ours": base, "att_engine": base + 2.0}
            if tier == "flat" and collapse:
                right = base                      # identical -> cannot discriminate
            else:
                right = base - effect
            cells[tier]["r@" + tier][key] = {"att_ours": right, "att_engine": right + 2.0}
    return cells


def test_a_collapsed_tier_is_pooled_BOTH_ways_and_neither_is_chosen() -> None:
    """🔒 THE RULING: report including and excluding, with the structural reason. Never choose."""
    from offline.att_rederivation import POOLINGS, pool_contrast

    report = pool_contrast(
        _cells_two_tiers(collapse=True, effect=6.0),
        verdict_id="V4",
        name="test contrast",
        left="l",
        right="r",
        tiers=("good", "flat"),
        verdict_fn=_outcome,
    )
    assert report.tiers_non_distinct == ("flat",)
    assert "zero BY CONSTRUCTION" in report.structural_reason
    assert "post-hoc exclusion" in report.structural_reason

    # Four pooled rows: 2 definitions x 2 poolings, and BOTH poolings are present for each.
    assert len(report.pooled) == 4
    for definition in ("att_ours", "att_engine"):
        got = {p.pooling for p in report.pooled if p.definition == definition}
        assert got == set(POOLINGS)

    including = next(p for p in report.pooled if p.definition == "att_ours"
                     and p.pooling == "including_non_distinct")
    excluding = next(p for p in report.pooled if p.definition == "att_ours"
                     and p.pooling == "excluding_non_distinct")
    # The structural zero halves the pooled effect, which is exactly why it must not be hidden.
    assert including.n_paired == 80
    assert excluding.n_paired == 40
    assert including.tiers == ("good", "flat")
    assert excluding.tiers == ("good",)
    assert abs(including.mean_difference) < abs(excluding.mean_difference)


def test_a_verdict_that_differs_between_poolings_is_ESCALATED_not_resolved() -> None:
    """Choosing between the poolings is a degree of freedom this module does not have."""
    from offline.att_rederivation import pool_contrast

    # An explicit threshold on the pooled mean, so the disagreement is a property of the POOLING
    # and not of CI arithmetic: the structural zeros halve the mean and carry it across the bar.
    def threshold_rule(mean: float, half: float, low: float, high: float) -> str:
        return "RESOLVES" if abs(mean) >= 1.0 else "DOES NOT RESOLVE"

    report = pool_contrast(
        _cells_two_tiers(collapse=True, effect=1.4),
        verdict_id="V4",
        name="test contrast",
        left="l",
        right="r",
        tiers=("good", "flat"),
        verdict_fn=threshold_rule,
    )
    outcomes = {
        (p.definition, p.pooling): p.verdict for p in report.pooled
    }
    assert outcomes[("att_ours", "including_non_distinct")] != outcomes[
        ("att_ours", "excluding_non_distinct")
    ], "this fixture is meant to make the two poolings disagree"
    assert report.escalate is True
    assert report.poolings_agree["att_ours"] is False
    assert "coordinator" in (report.escalation_reason or "")


def test_when_every_tier_discriminates_the_two_poolings_coincide() -> None:
    from offline.att_rederivation import pool_contrast

    report = pool_contrast(
        _cells_two_tiers(collapse=False, effect=6.0),
        verdict_id="V4", name="t", left="l", right="r",
        tiers=("good", "flat"), verdict_fn=_outcome,
    )
    assert report.tiers_non_distinct == ()
    assert report.escalate is False
    including = next(p for p in report.pooled if p.pooling == "including_non_distinct"
                     and p.definition == "att_ours")
    excluding = next(p for p in report.pooled if p.pooling == "excluding_non_distinct"
                     and p.definition == "att_ours")
    assert including.n_paired == excluding.n_paired == 80
    assert including.verdict == excluding.verdict


def test_a_contrast_with_no_tier_carrying_both_arms_refuses() -> None:
    from offline.att_rederivation import pool_contrast

    with pytest.raises(ValueError, match="no tier carries both"):
        pool_contrast({}, verdict_id="V4", name="t", left="l", right="r",
                      tiers=("good",), verdict_fn=_outcome)


def test_the_artifact_cannot_be_built_without_the_discriminability_guard_passing() -> None:
    """🔒 DEFERRED 42/44: an enforcement function nobody calls is not an enforcement function.

    The guard is invoked by ``rederivation_artifact`` on the assembled payload, so a contrast that
    somehow lost its discriminability cannot be written. Proved by stripping it and re-checking.
    """
    from offline.att_rederivation import (
        assert_contrast_reports_discriminability,
        pool_contrast,
        rederivation_artifact,
    )

    report = pool_contrast(
        _cells_two_tiers(collapse=True, effect=6.0),
        verdict_id="V4", name="t", left="l", right="r",
        tiers=("good", "flat"), verdict_fn=_outcome,
    )
    payload = rederivation_artifact([report], provenance={"runtime": {}})
    assert payload["contrasts"][0]["discriminability"]
    assert payload["n_escalations"] == 0

    stripped = json.loads(json.dumps(payload))
    del stripped["contrasts"][0]["discriminability"]
    with pytest.raises(ValueError, match="without its discriminability"):
        assert_contrast_reports_discriminability(stripped)


def test_the_artifact_refuses_an_empty_contrast_list() -> None:
    from offline.att_rederivation import rederivation_artifact

    with pytest.raises(ValueError, match="re-derives nothing"):
        rederivation_artifact([], provenance={})


# ----------------------------------------------------------------------
# Mixture tiers, and the refusal boundary around the WHOLE per-cell body
# ----------------------------------------------------------------------


def test_every_tier_the_campaign_rolls_can_resolve_its_env_settings() -> None:
    """🔒 REGRESSION: mix33 killed all five workers through Gate 0's seven-tier helper.

    Gate 0's ``tier_corpus_dirs`` covers the seven BEHAVIOUR tiers, which is right for Gate 0 and
    wrong for this campaign: P4.7's mixtures have no ``cf_hz1x1__mix33`` directory because their
    corpus is a blend. Every tier the campaign actually rolls must resolve.
    """
    from offline.admission_probe import ProbeRoots
    from offline.att_rederivation import rederivation_corpus_dirs, rederivation_env_settings

    corpus = Path("/home/filip/rltraffic/datasets_v11")
    if not corpus.is_dir():
        pytest.skip(f"the corpus is not present at {corpus}")
    roots = ProbeRoots(
        repo_root=Path("."), corpus_root=corpus,
        draws_root=Path("/home/filip/rltraffic/scenarios/draws"),
        output_root=Path("/home/filip/rltraffic/output"), work_dir=Path("/tmp/unused"),
    )
    expected = {
        "hz1x1": ("mappo1000", "mappo500", "maxpressure", "fixedtime", "random",
                  "mix33", "mix50", "mix67"),
        "grid4x4": ("mappo1000", "maxpressure", "fixedtime", "random"),
    }
    for scenario, tiers in expected.items():
        for tier in tiers:
            dirs = rederivation_corpus_dirs(scenario, tier, roots)
            assert dirs, f"{scenario}/{tier} resolved no corpus directory"
            settings = rederivation_env_settings(scenario, tier, roots)
            assert int(settings["max_steps"]) > 0 and int(settings["delta_time"]) > 0
    # The mixtures blend two component corpora, so they name MORE directories than a plain tier.
    assert len(rederivation_corpus_dirs("hz1x1", "mix33", roots)) > len(
        rederivation_corpus_dirs("hz1x1", "random", roots)
    )


def test_any_per_cell_failure_is_a_refusal_not_an_abort(monkeypatch, tmp_path: Path) -> None:
    """🔒 REGRESSION: the refusal path wrapped ONLY the reproduction check.

    A ``ValueError`` from tier resolution therefore still aborted all five workers, which is exactly
    what mix33 did. The boundary now covers the whole per-cell body, so a failure ANYWHERE in a cell
    is recorded and skipped. Three cells, the middle one broken; the run must finish with 2 rolled
    and 1 refused.
    """
    import offline.att_rederivation as module

    calls: list[str] = []

    def fake_settings(scenario: str, tier: str, roots):
        calls.append(tier)
        if tier == "boom":
            raise ValueError("'boom' is not a gated tier")
        return {"max_steps": 1, "delta_time": 1}

    def fake_probe(**kwargs):
        class Episode:
            att_ours = 42.0

            def as_record(self):
                return {"arm": kwargs["arm"], "draw_id": kwargs["draw_id"], "att_ours": 42.0}

        return Episode()

    monkeypatch.setattr(module, "rederivation_env_settings", fake_settings)
    monkeypatch.setattr(module, "build_cell_factory", lambda *a, **k: (None, {}))
    monkeypatch.setattr(module, "normalise_arm", lambda s, arm: ("bc", arm.split("@")[1]))
    monkeypatch.setattr("offline.admission_probe.probe_episode", fake_probe)
    monkeypatch.setattr("offline.admission_probe.created_from_flow", lambda *a, **k: 1)
    monkeypatch.setattr("offline.materialise_draws.draw_config_path", lambda *a, **k: tmp_path / "c.json")

    cells = [
        CellKey("hz1x1", "bc@good", 101, 1000),
        CellKey("hz1x1", "bc@boom", 101, 1000),
        CellKey("hz1x1", "bc@good", 101, 1001),
    ]
    committed = {("hz1x1", "bc@good", 101, 1000): 42.0, ("hz1x1", "bc@good", 101, 1001): 42.0}
    roots = type("R", (), {"work_dir": tmp_path, "draws_root": tmp_path,
                           "corpus_root": tmp_path, "output_root": tmp_path,
                           "repo_root": tmp_path})()
    out = module.run_campaign(
        cells, roots=roots, engine_seed=1,
        device=None, committed=committed, protected=(),
    )
    assert out["rolled"] == 2, "the run must continue past the broken cell"
    assert out["n_refused"] == 1
    assert out["refused"][0]["reason_class"] == "cell_failed"
    assert "boom" in out["refused"][0]["reason"]
    assert len(list(tmp_path.glob("cell_*.json"))) == 2
    assert len(list(tmp_path.glob("refused_*.json"))) == 1


def test_a_systemic_failure_still_stops_but_one_bad_cell_never_can(monkeypatch, tmp_path: Path) -> None:
    """Burning four hours to produce a directory of refusals helps nobody."""
    import offline.att_rederivation as module

    monkeypatch.setattr(
        module, "rederivation_env_settings",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("everything is broken")),
    )
    monkeypatch.setattr(module, "normalise_arm", lambda s, arm: ("bc", "t"))
    many = [CellKey("hz1x1", "bc@t", 101, 1000 + i) for i in range(60)]
    with pytest.raises(RuntimeError, match="consecutive cells failed"):
        module.run_campaign(
            many, roots=type("R", (), {"work_dir": tmp_path, "draws_root": tmp_path,
                                       "corpus_root": tmp_path, "output_root": tmp_path,
                                       "repo_root": tmp_path})(), engine_seed=1,
            device=None, committed={}, protected=(),
        )
    assert module.SYSTEMIC_FAILURE_THRESHOLD > 1, "one bad cell must never trip the threshold"
    assert len(list(tmp_path.glob("refused_*.json"))) == module.SYSTEMIC_FAILURE_THRESHOLD


def test_two_workers_actually_rolling_real_cells_end_to_end(tmp_path: Path) -> None:
    """🔒 THE TEST THREE LAUNCH FAILURES SAY IS MISSING. Two workers, real cells, really rolled.

    Every orchestration defect so far survived the unit tests and died on a real launch:

    1. the manifest was shard-scoped, so four of five workers refused;
    2. a replicate artifact silently overwrote 100 committed values, and the campaign died on an arm
       whose checkpoint map was correct;
    3. ``mix33`` reached Gate 0's seven-tier helper and a ``ValueError`` aborted all five shards,
       because the refusal boundary wrapped only the reproduction check.

    The end-to-end test written after (1) used ``--limit 0``, so it exercised the manifest path and
    **never rolled a cell** -- which is precisely why it could not catch (2) or (3). This one rolls.
    It drives ``main`` twice as two different workers against one work directory and checks the
    composition: agreed manifest, disjoint work, real episodes, real reproduction, no refusals.

    It is deliberately small -- three cells per worker -- because what is under test is the
    composition, not the throughput.
    """
    from offline.att_rederivation import main, rederivation_env_settings

    corpus = Path("/home/filip/rltraffic/datasets_v11")
    draws = Path("/home/filip/rltraffic/scenarios/draws")
    output = Path("/home/filip/rltraffic/output")
    if not (corpus.is_dir() and draws.is_dir() and (output / "p5_1").is_dir()):
        pytest.skip("the corpus, draws and campaign artifacts are not all present")
    try:
        import cityflow  # noqa: F401
    except ImportError:
        pytest.skip("the CityFlow engine is not installed, so no cell can be rolled")

    work = tmp_path / "work"
    repo = Path(__file__).resolve().parents[1]
    common = [
        "--repo-root", str(repo),
        "--corpus-root", str(corpus),
        "--draws-root", str(draws),
        "--output-root", str(output),
        "--work-dir", str(work),
    ]

    codes = [
        main([*common, "run", "--verdicts", "V1", "--shard", str(shard), "--of", "2",
              "--limit", "3", "--torch-threads", "1"])
        for shard in (0, 1)
    ]
    assert codes == [0, 0], f"a worker that rolled its cells with no refusals must exit 0: {codes}"

    # Both workers accepted one campaign-scoped manifest.
    manifest = json.loads((work / "campaign_manifest.json").read_bytes())
    status = campaign_status(work)
    assert manifest["n_cells"] == status["declared"] > 6, (
        "the manifest must declare the CAMPAIGN, not either worker's shard"
    )

    # Real work happened, it is disjoint, and nothing was refused.
    rolled = sorted(work.glob("cell_*.json"))
    assert len(rolled) == 6, f"two workers x 3 cells must produce 6 cell files, got {len(rolled)}"
    assert not list(work.glob("refused_*.json")), "no cell should have been refused"
    assert status["present"] == 6 and status["refused"] == 0
    assert status["complete"] is False

    # Every rolled cell is a real episode carrying all five A11(b) quantities, and every one
    # reproduced its committed att_ours exactly -- which is what "rolled" is supposed to mean.
    keys = set()
    for path in rolled:
        row = json.loads(path.read_bytes())
        key = (row["scenario"], row["arm"], row["seed"], row["draw_id"])
        assert key not in keys, f"{key} was rolled by BOTH workers; the shards are not disjoint"
        keys.add(key)
        for field in ("att_ours", "att_engine", "entered", "created", "never_entered"):
            assert field in row, f"{path.name} is missing A11(b)'s {field}"
        assert row["reproduces_committed"] is True
        assert row["att_ours"] == row["committed_att_ours"]
        assert int(row["created"]) == int(row["entered"]) + int(row["never_entered"])

    # The composition-level form of the mix33 defect: EVERY tier the declared campaign contains must
    # resolve its env settings.  Rolling all 38,500 cells to learn this would take hours; resolving
    # every declared tier takes a second and fails on exactly the same bug.
    from offline.admission_probe import ProbeRoots
    from offline.att_rederivation import normalise_arm

    roots = ProbeRoots(repo_root=repo, corpus_root=corpus, draws_root=draws,
                       output_root=output, work_dir=work)
    # ⚠️ Over the FULL campaign, not this run's --verdicts V1 slice: V1 touches three hz1x1 tiers
    # and no mixture at all, so checking only the declared slice would have missed mix33 entirely.
    import argparse

    from offline.att_rederivation import REDERIVATION_SOURCES, campaign_cell_set

    every_cell = campaign_cell_set(
        argparse.Namespace(
            repo_root=str(repo), verdicts=[v.verdict_id for v in REDERIVATION_SOURCES]
        ),
        output,
    )
    declared_tiers = {
        (c.scenario, normalise_arm(c.scenario, c.arm)[1]) for c in every_cell
    }
    assert declared_tiers, "no tier was declared -- the check would pass vacuously"
    assert any(tier in MIXTURE_TIERS for _, tier in declared_tiers), (
        "the full campaign must contain P4.7's mixture tiers; if it does not, the check that "
        "would have caught mix33 is not actually covering them"
    )
    for scenario, tier in sorted(declared_tiers):
        settings = rederivation_env_settings(scenario, tier, roots)
        assert int(settings["max_steps"]) > 0, f"{scenario}/{tier} resolved no usable settings"

    # The composition-level form of the REPLICATE defect, by an INDEPENDENT route.  The affected
    # slot was grid4x4/dt_spatial@random seed 202, which this run does not roll -- and a reference
    # defect anywhere poisons the acceptance test everywhere, so the whole index is checked.
    #
    # The index is rebuilt here from the COMMITTED cell files only, applying the replicate rule
    # directly rather than through the production helper, and the production helper is then required
    # to agree with it on every key.  Re-scanning with the same helper would prove nothing; this way
    # a change that readmits a replicate makes the two disagree.
    from collections import defaultdict

    from offline.att_rederivation import (
        _resolve_sources,
        committed_att_index,
        is_replicate_artifact,
        slots_from_episode_block,
    )

    independent: dict[tuple, set[float]] = defaultdict(set)
    for source in REDERIVATION_SOURCES:
        for path in _resolve_sources(source, repo_root=repo, output_root=output):
            if is_replicate_artifact(path):
                continue
            try:
                payload = json.loads(path.read_bytes())
            except (ValueError, OSError):
                continue
            for episode_row in slots_from_episode_block(payload):
                value = episode_row.get("att_horizon")
                if value is None:
                    continue
                independent[
                    (source.scenario, str(episode_row["arm"]),
                     episode_row.get("seed"), int(episode_row["draw_id"]))
                ].add(float(value))
    assert independent, "no committed value was read -- the check would pass vacuously"

    ambiguous = {k: v for k, v in independent.items() if len(v) > 1}
    assert not ambiguous, (
        f"{len(ambiguous)} cells carry more than one committed value among the COMMITTED artifacts "
        f"alone. First: {sorted(ambiguous)[0]} -> {sorted(ambiguous[sorted(ambiguous)[0]])}"
    )

    production = committed_att_index(repo_root=repo, output_root=output)
    assert len(production) == len(independent)
    disagreeing = [
        (k, production[k], next(iter(v)))
        for k, v in independent.items()
        if k in production and production[k] != next(iter(v))
    ]
    assert not disagreeing, (
        f"{len(disagreeing)} cells where the committed index disagrees with an independent scan of "
        f"the committed artifacts -- a replicate or a second measurement has been admitted as the "
        f"reference. First: {disagreeing[0]}"
    )


def test_the_module_actually_runs_as_a_command_line_program(tmp_path: Path) -> None:
    """🔒 REGRESSION: the ``__main__`` block was deleted and NOTHING caught it.

    A wholesale rewrite of ``run_campaign`` sliced out
    ``if __name__ == "__main__": raise SystemExit(main())``. From that commit until this test,
    ``python -m offline.att_rederivation <anything>`` imported the module, defined every function and
    **exited 0 having done nothing** -- so ``status`` printed nothing and succeeded, and a worker
    would have "finished" its shard instantly with an empty log.

    Every other test in this file calls ``main()`` as a Python function, which is precisely why none
    of them could see it. This one uses a SUBPROCESS, which is how the campaign is actually launched.
    Silence and success must never be indistinguishable.
    """
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    work = tmp_path / "empty_work"
    work.mkdir()

    done = subprocess.run(
        [sys.executable, "-m", "offline.att_rederivation",
         "--output-root", str(tmp_path), "--work-dir", str(work), "status"],
        cwd=str(repo), capture_output=True, text=True, timeout=180,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(repo), "OMP_NUM_THREADS": "1"},
    )
    assert done.stdout.strip(), (
        f"`status` produced NO OUTPUT. stderr: {done.stderr[-400:]}. A command that prints nothing "
        "and exits is indistinguishable from one that never ran"
    )
    state = json.loads(done.stdout)
    for field in ("declared", "present", "refused", "complete"):
        assert field in state, f"`status` must always report {field}"
    assert done.returncode != 0, "`status` must exit non-zero while the campaign is incomplete"

    # `--help` is the cheapest possible proof that the entry point is wired at all.
    helped = subprocess.run(
        [sys.executable, "-m", "offline.att_rederivation", "--help"],
        cwd=str(repo), capture_output=True, text=True, timeout=180,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(repo), "OMP_NUM_THREADS": "1"},
    )
    assert helped.returncode == 0 and "preflight" in helped.stdout and "run" in helped.stdout


def test_a_run_with_no_work_still_says_what_it_decided(tmp_path: Path) -> None:
    """A worker that finds nothing to do must say so, not exit silently.

    Same defect class as the worker exit codes: an empty log and a successful run are the same
    observation to whoever is watching five tmux windows.
    """
    from offline.att_rederivation import main

    corpus = Path("/home/filip/rltraffic/datasets_v11")
    output = Path("/home/filip/rltraffic/output")
    if not (corpus.is_dir() and (output / "p5_1").is_dir()):
        pytest.skip("the campaign's source artifacts are not present")

    repo = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main([
            "--repo-root", str(repo), "--corpus-root", str(corpus),
            "--draws-root", "/home/filip/rltraffic/scenarios/draws",
            "--output-root", str(output), "--work-dir", str(work),
            "run", "--verdicts", "V1", "--limit", "0",
        ])
    printed = buffer.getvalue()
    assert code == 0
    assert "campaign declares" in printed, "a run must state the campaign it accepted"
    assert "rolls 0 of them" in printed, "a run with no work must SAY it has no work"
    assert "rolled=0" in printed, "a run must state its outcome even when it did nothing"


def test_every_name_a_module_exports_actually_exists() -> None:
    """🔒 THE THIRD INSTANCE of one defect class: a DECLARATION that no longer matches the artifact.

    ``engine_att_reference.__all__`` listed ``PerSecondEngineObserver`` after that class was made
    lazily-constructed and stopped being a module attribute, so ``from ... import *`` would have
    raised. Nothing caught it because nothing checked ``__all__``.

    The same shape as the two edits before it: the entry point deleted by a slice rewrite, and a
    string replace whose match was never asserted. In every case the declaration and the artifact
    disagreed and only a reader could tell.
    """
    import importlib

    for name in ("offline.engine_att_reference", "offline.att_rederivation"):
        module = importlib.import_module(name)
        exported = list(getattr(module, "__all__", []))
        assert exported, f"{name} exports nothing -- the check would pass vacuously"
        missing = [entry for entry in exported if not hasattr(module, entry)]
        assert missing == [], (
            f"{name}.__all__ names {missing}, which do not exist on the module. "
            "`from module import *` would raise, and a declaration that does not match the "
            "artifact is exactly the class of defect this test exists for"
        )


def test_every_offline_module_with_a_main_can_actually_be_run() -> None:
    """A ``main()`` with no ``__main__`` block is a CLI that silently does nothing.

    Generalised from my own defect: I deleted ``if __name__ == "__main__"`` from
    ``att_rederivation`` and ``python -m offline.att_rederivation`` then exited 0 having done
    nothing, for every subcommand, for two commits. This scans the whole package so the next
    instance is caught wherever it happens.
    """
    repo = Path(__file__).resolve().parents[1]
    offline = repo / "offline"
    assert offline.is_dir(), f"no offline package at {offline}"

    with_main = []
    missing_entry = []
    for path in sorted(offline.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "\ndef main(" not in source:
            continue
        with_main.append(path.name)
        if '__name__ == "__main__"' not in source:
            missing_entry.append(path.name)

    assert with_main, "no module under offline/ defines main() -- the check would pass vacuously"
    assert missing_entry == [], (
        f"these modules define main() but have no __main__ block, so `python -m` on them imports "
        f"and exits 0 having done nothing: {missing_entry}"
    )
