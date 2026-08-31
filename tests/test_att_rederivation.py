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
