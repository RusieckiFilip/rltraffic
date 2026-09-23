"""P7.3d's B.6 fix round: the campaign PATH, from the driver's argument to a written grid4x4 chunk.

``BRIEF_39`` Amendment B.6 (gate G4 NOT CLEAR, ``docs/reviews/P7.3d-preflight.md``), with B.5-1 and
B.5-3.  B.6-1 names the class this file exists for: *every C3b/C4/C6 piece was tested where it was
built and none through the path the campaign runs*.  T-16 drove the spatial loader and the 16-id
refusal directly; T-driver asserted only the absence of the group-leader text; the one execution of
the header's line refused at the dirty-tree check, before the canary.  The delivered driver could
not pass its own canary, and ``run_cell`` had no grid4x4 branch -- and the suite was green.

So the load-bearing test here is ONE end-to-end execution of the DELIVERED driver on draw 5
(:func:`test_the_campaign_path_from_the_drivers_argument_to_a_written_grid4x4_chunk`).  Everything
else pins one piece that execution forces, so that a mutant of that piece dies somewhere cheaper.

HOW THE DRIVER IS EXECUTED WITHOUT TOUCHING ANY REAL PATH
--------------------------------------------------------
* **A clean snapshot of THIS working copy.**  The driver refuses a dirty tree and ``run_cell``
  refuses a ``git_dirty`` chunk, so :func:`_snapshot_clone` builds what
  ``tests/test_transfer_curve.py::test_k_finding3`` built -- ``git clone --shared`` + checkout --
  then copies in every path the working copy changed (modified, added, deleted, untracked) and
  commits it IN THE CLONE.  The code under test is the code on disk, committed or not.
* **The redirection**, every substitution asserted to have happened exactly as often as stated:
  ``WORK_TREE`` -> the clone; ``CAMPAIGN_DIR`` and ``TOKEN`` -> the sandbox; the ``dt-reroll-check``
  invocation -> a stub line (the real check is thirteen SUMO episodes, run once by hand and never by
  the suite).  Nothing under ``/home/filip/rltraffic/output`` is written; the interpreter, the draws
  and the checkpoints are READ from the main tree, as the driver reads them.
* **Draw 5, through the real module.**  The end-to-end test's ``cells`` and ``report`` lines run
  :data:`_SHIM`, four lines that import the REAL ``offline.transfer_curve``, replace ONE function --
  the grid4x4 declaration, with draw 1000's seven cells moved to draw 5 and the (dt seed 101,
  fixedtime) pair first -- and call the REAL ``main()``.  ``--limit 2`` therefore rolls exactly the
  fenced pair on the smoke draw (outside both pools, Amendment B.2-2), and the declaration still
  counts 700, so ``report`` must refuse with **698** missing.  (``sitecustomize`` cannot do this:
  ``-m`` runs the module as ``__main__``, a different object from the one a startup hook patches.)

B.7 AND B.7.1 (the second round)
--------------------------------
* **One fixture builds every executed driver** (:func:`driver_sandbox`, B.7-3): the roots, the
  token and ``--limit 0`` by construction, and every refusal test asserts that no chunk was written.
  The rule it enforces was learned the hard way: in B.6's first mutation run a refusal test with no
  limit let a mutant roll 70 held-out cells into a sandbox.
* **The driver derives its tree from its own location** and refuses the implementer's (B.7.1-1).
* **``report``'s grid4x4 body** (B.7-2) is pinned by the complete-set test: 700 stubs through the
  REAL driver to a written ``p7_3d_grid4x4.json``, asserted field by field against expectations the
  test derives from its own stubs, and one swapped reference-cell value refused.
* **Per-intersection returns on every grid4x4 chunk** (B.7.1-2), by the probe's two routes.

GATES, each naming what it consumes
-----------------------------------
The main tree's interpreter, draws tree and ``output/`` (the driver hardcodes ``MAIN``, so these do
too, as ``tests/test_transfer_curve.py`` does); ``RLTRAFFIC_GRID4X4_RESCO`` for anything that runs
grid4x4 SUMO; ``sumo`` and ``traci``; ``setsid`` and ``tmux`` for the executed forms.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import pytest

import offline.transfer_calibration as tc
import offline.transfer_curve as tcv

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"
DRIVER = REPO_ROOT / "offline" / "campaigns" / "p7_3d_grid4x4.sh"

#: The MAIN tree, as the driver names it (``MAIN=/home/filip/rltraffic``).  ``DEFERRED`` 82: the
#: gated tests name the artifact they consume by the same path the code under test reads.
MAIN_TREE = Path("/home/filip/rltraffic")
MAIN_INTERPRETER = MAIN_TREE / ".venv" / "bin" / "python"
DRAWS_ROOT = MAIN_TREE / "scenarios" / "draws"
OUTPUT_ROOT = MAIN_TREE / "output"

GRID = "cityflow_grid4x4"
FENCED_DRAW = 5
N_IDS = 16
N_ACTIONS = 8
DECISIONS = 360

#: The header's documented paths, which the executed-form test substitutes.  The driver path is the
#: RUN worktree's (B.7.1-1(ii)): the campaign never runs from the implementer's tree.
HEADER_DRIVER_PATH = "/home/filip/rltraffic-p73d-run/offline/campaigns/p7_3d_grid4x4.sh"
HEADER_CAPTURE_PATH = "/home/filip/rltraffic/output/p7_3d_runs/campaign_capture.txt"
#: The implementer's worktree, which the driver refuses as a home (B.7.1-1(iv)).
IMPLEMENTER_TREE = "/home/filip/rltraffic-p73d"

#: The digests B.6-2(5) names, as prefixes; the tests also recompute them from the files.
GRID4X4_CFG_PREFIX = "c27d31e8"
HZ1X1_CFG_PREFIX = "c177e962"


def _head_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


HEAD_SHA = _head_sha()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _calibration() -> dict[str, Any]:
    return json.loads((DATA / "p7_3d_calibration.json").read_text(encoding="utf-8"))


#: The 16 ids, targets and support ranges, read by THIS file's own route (``json``), never by the
#: module under test.
IDS: list[str] = [str(ix) for ix in _calibration()["intersection_ids"]]
TARGETS: dict[str, float] = {
    ix: float(_calibration()["per_intersection"][ix]["budgets"]["k100"]["target"]) for ix in IDS
}
SUPPORT: dict[str, list[float]] = {
    ix: [float(v) for v in _calibration()["per_intersection"][ix]["support_range"]] for ix in IDS
}


def _parity(scenario: str, draw: int) -> Path:
    return DRAWS_ROOT / scenario / f"draw_{draw:04d}" / "parity"


def _grid_cell(kind: str, arm: str, draw: int, *, seed: int | None = None) -> dict[str, Any]:
    """One grid4x4 cell in exactly the shape :func:`offline.transfer_curve.grid4x4_cells` emits."""
    return {
        "kind": kind,
        "subject": tcv.GRID4X4_SUBJECT if kind == "dt" else None,
        "arm": arm,
        "seed": seed,
        "draw_id": int(draw),
        "scenario": GRID,
        "stage": tcv.STAGE_GRID4X4,
    }


def _grid_payload(
    cell: Mapping[str, Any], *, config_sha: str, routes_sha: str, **overrides: Any
) -> dict[str, Any]:
    """A complete, valid grid4x4 chunk for *cell*, from which a test perturbs exactly one thing.

    The shape ``run_cell`` writes for a grid4x4 cell (format ``p7.3d-grid4x4/1.0``): every
    per-intersection quantity is a mapping keyed by id, ``actions`` is the decisions x 16 matrix in
    ``intersection_ids`` order, and the RTG obeys D1's shift-by-one (a constant series under zero
    rewards, which is D1-consistent).  ``local_return`` / ``local_return_from_lanes`` are the 16
    per-intersection episode returns by the probe's two routes (B.7.1-2), on BOTH kinds of cell.
    """
    draw = int(cell["draw_id"])
    checked = draw == tcv.HALTING_CHECK_DRAW
    returns = {ix: -(200.0 + 8.0 * j) for j, ix in enumerate(IDS)}
    payload: dict[str, Any] = {
        "format_version": tcv.GRID4X4_ARTIFACT_FORMAT_VERSION,
        **dict(cell),
        "policy_seed": None,
        "engine_seed_requested": tcv.ENGINE_SEED,
        "engine_seed_drawn": 424242,
        "decisions": DECISIONS,
        "intersection_ids": list(IDS),
        "actions": [[(t + j) % N_ACTIONS for j in range(N_IDS)] for t in range(DECISIONS)],
        "actions_in_range": True,
        "action_space_n": N_ACTIONS,
        "episode_reward": -4242.0,
        "att_horizon": 78.9,
        "att_env": 78.9,
        "att_running_mean": 77.9,
        "horizon_vehicle_count": 11.0,
        "e_sumo": 123.456,
        "p_sumo": 110.0,
        "w_sumo": 100.0,
        "mean_depart_delay": 3.0,
        "n_created": 1335,
        "n_entered": 1330,
        "n_never_entered": 5,
        "n_pending_at_horizon": 2,
        "n_teleports": 0,
        "n_vanished_without_arrival": 0,
        "n_arrived_never_observed_at_a_boundary": 0,
        "max_abs_depart_clock_deviation": 0.0,
        "n_observations": 3600,
        "vehicle_types_seen": ["cf_parity"],
        "time_to_teleport_option": "-1",
        "halting_checked": checked,
        "halting_max_abs_difference": 0 if checked else None,
        "halting_n_lane_seconds": 691200 if checked else None,
        "halting_n_disagreeing_lane_seconds": 0 if checked else None,
        "config_sha256": config_sha,
        "routes_sha256": routes_sha,
        "calibration_sha256": tcv.P7_3D_CALIBRATION_SHA256,
        "anchor_training_sha256": None,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "sha256_checked_against": None,
        "target_rtg": None,
        "rtg_first": None,
        "rtg_last": None,
        "rtg_series": None,
        "reward_series": None,
        "rtg_advanced_every_decision": None,
        "n_decisions_in_support": None,
        "support_range": None,
        "in_support_counts": None,
        "local_return": dict(returns),
        "local_return_from_lanes": dict(returns),
        "canary_seconds": 0.8,
        "seconds": 31.0,
        "git_commit": HEAD_SHA,
        "git_dirty": False,
    }
    if cell["kind"] == "dt":
        seed = int(cell["seed"])
        series = {ix: [TARGETS[ix]] * DECISIONS for ix in IDS}
        rewards = {ix: [-0.0] + [0.0] * (DECISIONS - 1) for ix in IDS}
        payload.update(
            {
                "checkpoint": str(
                    OUTPUT_ROOT / tc.GRID4X4_CHECKPOINT_SUBDIR / f"{tc.GRID4X4_CHECKPOINT_STEM}{seed}.pt"
                ),
                "checkpoint_sha256": tc.GRID4X4_CHECKPOINT_SHA256[seed],
                "sha256_checked_against": ["A20(a)"],
                "target_rtg": dict(TARGETS),
                "rtg_first": {ix: series[ix][0] for ix in IDS},
                "rtg_last": {ix: series[ix][-1] for ix in IDS},
                "rtg_series": series,
                "reward_series": rewards,
                "rtg_advanced_every_decision": {ix: True for ix in IDS},
                "n_decisions_in_support": {ix: DECISIONS for ix in IDS},
                "support_range": {ix: list(SUPPORT[ix]) for ix in IDS},
                "in_support_counts": {
                    ix: {"in_support": DECISIONS, "below": 0, "above": 0} for ix in IDS
                },
            }
        )
    payload.update(overrides)
    return payload


def _grid_digests(draw: int) -> tuple[str, str]:
    parity = _parity(GRID, draw)
    return _sha256_path(parity / "noteleport.sumocfg"), _sha256_path(parity / "routes.rou.xml")


def _hz1x1_digests(draw: int) -> tuple[str, str]:
    parity = _parity("cityflow1x1", draw)
    return _sha256_path(parity / "noteleport.sumocfg"), _sha256_path(parity / "routes.rou.xml")


def _needs_parity(*pairs: tuple[str, int]) -> None:
    for scenario, draw in pairs:
        path = _parity(scenario, draw) / "noteleport.sumocfg"
        if not path.is_file():
            pytest.skip(f"{path} is absent: the gitignored draws tree lives in the main tree only")


def _needs_checkpoints() -> None:
    for seed in tc.TRAINING_SEEDS:
        path = OUTPUT_ROOT / tc.GRID4X4_CHECKPOINT_SUBDIR / f"{tc.GRID4X4_CHECKPOINT_STEM}{seed}.pt"
        if not path.is_file():
            pytest.skip(f"{path} is absent: A20(a)'s checkpoints are gitignored, main tree only")


def _canary_record(work: Path) -> None:
    """The run's canary record, as the driver parks it right after the token (E1.4)."""
    work.mkdir(parents=True, exist_ok=True)
    (work / tc.CANARY_RECORD_NAME).write_text(
        json.dumps(
            {
                "seconds": 0.8,
                "facts": {
                    "decisions": tc.CANARY_REFERENCE_DECISIONS,
                    "local_return": tc.CANARY_REFERENCE_LOCAL_RETURN,
                    "att_horizon": tc.CANARY_REFERENCE_ATT_HORIZON,
                    "two_routes_agree": True,
                },
                "git_commit": HEAD_SHA,
                "git_dirty": False,
            }
        ),
        encoding="utf-8",
    )


# ==================================================================================
# B.6-2(2) -- ONE scenario-aware call per identity, at all three sites
# ==================================================================================
def test_the_scenario_key_is_strict_and_hz1x1_is_the_absent_default() -> None:
    """A.1-2's lesson: no branch may be keyed on a coincidence, and an unregistered key must not
    fall through to hz1x1's pins."""
    assert tcv.scenario_of({"draw_id": 1000}) == "cityflow1x1"
    assert tcv.scenario_of({"scenario": None}) == "cityflow1x1"
    assert tcv.scenario_of({"scenario": "cityflow1x1"}) == "cityflow1x1"
    assert tcv.scenario_of({"scenario": GRID}) == GRID
    with pytest.raises(ValueError, match="cityflow_grid9x9"):
        tcv.scenario_of({"scenario": "cityflow_grid9x9"})


def test_the_demand_identity_follows_the_cells_scenario() -> None:
    """The pre-flight's probe (iii), as a test: hz1x1's files for an hz1x1 cell, grid4x4's for a
    grid4x4 cell -- digests recomputed here from the files, not read from the module."""
    _needs_parity((GRID, 1000), ("cityflow1x1", 1000))
    grid = tcv.demand_identity_for(_grid_cell("anchor", "fixedtime", 1000), out_root=DRAWS_ROOT)
    hz = tcv.demand_identity_for(
        {"kind": "anchor", "subject": None, "arm": "fixedtime", "seed": None, "draw_id": 1000},
        out_root=DRAWS_ROOT,
    )
    assert Path(grid["config_path"]).parent == _parity(GRID, 1000)
    assert (grid["config_sha256"], grid["routes_sha256"]) == _grid_digests(1000)
    assert grid["config_sha256"].startswith(GRID4X4_CFG_PREFIX)
    assert (hz["config_sha256"], hz["routes_sha256"]) == _hz1x1_digests(1000)
    assert hz["config_sha256"].startswith(HZ1X1_CFG_PREFIX)
    assert grid["config_sha256"] != hz["config_sha256"]


def test_a_grid4x4_chunk_carrying_hz1x1s_demand_is_refused_not_reused() -> None:
    """B.6-2(2): the reviewer's sandbox found this chunk REUSED at ``8c79778``.

    ``run_cell``, ``chunk_is_reusable`` and ``report`` all re-derived the demand WITHOUT the
    scenario, so a grid4x4 anchor chunk recording hangzhou's digests looked consistent to every
    check.  The control -- the same chunk with grid4x4's digests -- must be reusable, or the
    refusal would be refusing everything.
    """
    _needs_parity((GRID, 1000), ("cityflow1x1", 1000))
    cell = _grid_cell("anchor", "fixedtime", 1000)
    keys = {"cell": cell, "out_root": DRAWS_ROOT, "output_root": OUTPUT_ROOT, "data_dir": DATA}

    right = _grid_payload(cell, config_sha=_grid_digests(1000)[0], routes_sha=_grid_digests(1000)[1])
    wrong = _grid_payload(cell, config_sha=_hz1x1_digests(1000)[0], routes_sha=_hz1x1_digests(1000)[1])

    assert tcv.chunk_is_reusable(right, **keys) is True
    assert tcv.chunk_is_reusable(wrong, **keys) is False


def test_report_refuses_a_grid4x4_chunk_carrying_hz1x1s_demand_at_the_digest_step(
    tmp_path: Path,
) -> None:
    """The same chunk, offered to ``report``: refused at step 6, naming the digest.  The control
    passes step 6 and is stopped at step 7 (one anchor has no pair), which proves step 6 accepted it
    -- without writing any artifact."""
    _needs_parity((GRID, 1000), ("cityflow1x1", 1000))
    cell = _grid_cell("anchor", "fixedtime", 1000)
    work = tmp_path / "work"
    _canary_record(work)

    tcv.write_chunk(
        _grid_payload(cell, config_sha=_hz1x1_digests(1000)[0], routes_sha=_hz1x1_digests(1000)[1]),
        work_dir=work,
    )
    kwargs = {
        "work_dir": work, "out_path": tmp_path / "out" / "artifact.json", "output_root": OUTPUT_ROOT,
        "out_root": DRAWS_ROOT, "data_dir": DATA, "cells": [cell],
    }
    with pytest.raises(ValueError, match=r"config_sha256 .* is not draw 1000's"):
        tcv.report(**kwargs)

    tcv.write_chunk(
        _grid_payload(cell, config_sha=_grid_digests(1000)[0], routes_sha=_grid_digests(1000)[1]),
        work_dir=work,
    )
    with pytest.raises(ValueError, match=r"carries anchors \['fixedtime'\]"):
        tcv.report(**kwargs)
    assert not (tmp_path / "out").exists(), "a refusal writes nothing"


def test_the_grid4x4_checkpoint_identity_is_a20a_s_pin_and_hz1x1_s_tables_still_refuse_it() -> None:
    """B.6-2(2): the checkpoint identity comes from the five A20(a) digests pinned in ``fcf22fc``,
    NOT from ``checkpoint_identity``'s hz1x1 tables -- which keep refusing this subject."""
    _needs_checkpoints()
    for seed in tc.TRAINING_SEEDS:
        path = OUTPUT_ROOT / tc.GRID4X4_CHECKPOINT_SUBDIR / f"{tc.GRID4X4_CHECKPOINT_STEM}{seed}.pt"
        identity = tcv.grid4x4_checkpoint_identity(seed, output_root=OUTPUT_ROOT)
        assert identity["file_sha256"] == _sha256_path(path) == tc.GRID4X4_CHECKPOINT_SHA256[seed]
        assert identity["path"] == str(path)
        assert identity["subject"] == tcv.GRID4X4_SUBJECT and identity["seed"] == seed
        assert "SHA256SUMS_p5_2.txt" in identity["sha256_checked_against"]
        via_cell = tcv.checkpoint_identity_for(
            _grid_cell("dt", tcv.GRID4X4_ARM, 1000, seed=seed), output_root=OUTPUT_ROOT, data_dir=DATA
        )
        assert via_cell == identity
    with pytest.raises(ValueError, match="unknown subject"):
        tcv.checkpoint_identity(tcv.GRID4X4_SUBJECT, 101, output_root=OUTPUT_ROOT, data_dir=DATA)


def test_a_grid4x4_checkpoint_at_another_digest_is_refused(tmp_path: Path) -> None:
    path = tmp_path / tc.GRID4X4_CHECKPOINT_SUBDIR / f"{tc.GRID4X4_CHECKPOINT_STEM}101.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"other weights under the registered name")
    with pytest.raises(ValueError, match="A20"):
        tcv.grid4x4_checkpoint_identity(101, output_root=tmp_path)


def test_the_support_ranges_come_from_the_pinned_calibration_artifact() -> None:
    ranges = tcv.grid4x4_support_ranges(data_dir=DATA)
    assert list(ranges) == IDS
    assert {ix: list(value) for ix, value in ranges.items()} == SUPPORT


# ==================================================================================
# validate_cell_payload on a grid4x4 chunk -- the 16-id refusal has a call site now
# ==================================================================================
def _valid_dt() -> dict[str, Any]:
    cell = _grid_cell("dt", tcv.GRID4X4_ARM, 1001, seed=101)
    return _grid_payload(cell, config_sha="c" * 64, routes_sha="r" * 64)


def test_a_valid_grid4x4_chunk_validates_against_its_cell() -> None:
    payload = _valid_dt()
    tcv.validate_cell_payload(payload, cell=_grid_cell("dt", tcv.GRID4X4_ARM, 1001, seed=101))
    anchor = _grid_payload(_grid_cell("anchor", "maxpressure", 1000), config_sha="c" * 64, routes_sha="r" * 64)
    tcv.validate_cell_payload(anchor, cell=_grid_cell("anchor", "maxpressure", 1000))


def test_one_intersection_that_missed_its_target_is_refused_by_name() -> None:
    """B.6-2(2) / M3: ``assert_rtg_first_matches_targets`` CALLED on the finished chunk.

    ``rtg_first`` is left equal to the targets so the dict comparison cannot fire; only the per-id
    refusal over the SERIES can.  *Mutation:* the call removed from ``validate_cell_payload`` ->
    this survives no longer.
    """
    payload = _valid_dt()
    payload["rtg_series"] = dict(payload["rtg_series"])
    payload["rtg_series"]["C1"] = [TARGETS["C1"] - 1.0] + payload["rtg_series"]["C1"][1:]
    with pytest.raises(ValueError, match="C1"):
        tcv.validate_cell_payload(payload)


def test_a_grid4x4_chunk_is_refused_under_each_hz1x1_pin() -> None:
    """The calibration pin, the format version, the arm set and the scenario identity are per
    scenario: a grid4x4 chunk carrying any hz1x1 value is a different computation."""
    cell = _grid_cell("dt", tcv.GRID4X4_ARM, 1001, seed=101)
    with pytest.raises(ValueError, match="calibration"):
        tcv.validate_cell_payload({**_valid_dt(), "calibration_sha256": tcv.P7_2B_CALIBRATION_SHA256})
    with pytest.raises(ValueError, match="format_version"):
        tcv.validate_cell_payload({**_valid_dt(), "format_version": tcv.ARTIFACT_FORMAT_VERSION})
    with pytest.raises(ValueError, match="naive"):
        tcv.validate_cell_payload({**_valid_dt(), "arm": "naive"})
    # ... and an ANCHOR under `random`, which only the arm-set check can refuse: on a DT chunk the
    # subject/arm check refuses `naive` as well, so the line above alone let the arm-set check be
    # deleted with every test green (mutation M18, SURVIVED on the first run; this closes it).
    random_anchor = _grid_payload(
        {**_grid_cell("anchor", "random", 1001), "seed": 1000}, config_sha="c" * 64, routes_sha="r" * 64
    )
    with pytest.raises(ValueError, match="random"):
        tcv.validate_cell_payload(random_anchor)
    hz_cell = {k: v for k, v in cell.items() if k != "scenario"}
    with pytest.raises(ValueError, match="scenario"):
        tcv.validate_cell_payload(_valid_dt(), cell=hz_cell)
    short = _valid_dt()
    short["reward_series"] = {**short["reward_series"], "D3": short["reward_series"]["D3"][:-1]}
    with pytest.raises(ValueError, match="D3"):
        tcv.validate_cell_payload(short)


def test_resume_refuses_a_grid4x4_dt_chunk_whose_targets_are_not_the_pinned_ones() -> None:
    """A self-consistent chunk (series == its own targets) whose targets are NOT the artifact's:
    ``validate`` cannot see it, so ``chunk_is_reusable`` re-derives the 16 from the pin."""
    _needs_checkpoints()
    _needs_parity((GRID, 1001),)
    cell = _grid_cell("dt", tcv.GRID4X4_ARM, 1001, seed=101)
    keys = {"cell": cell, "out_root": DRAWS_ROOT, "output_root": OUTPUT_ROOT, "data_dir": DATA}
    config, routes = _grid_digests(1001)
    right = _grid_payload(cell, config_sha=config, routes_sha=routes)
    assert tcv.chunk_is_reusable(right, **keys) is True

    shifted = {ix: value + 1.0 for ix, value in TARGETS.items()}
    wrong = _grid_payload(
        cell, config_sha=config, routes_sha=routes,
        target_rtg=shifted, rtg_first=dict(shifted), rtg_last=dict(shifted),
        rtg_series={ix: [shifted[ix]] * DECISIONS for ix in IDS},
    )
    tcv.validate_cell_payload(wrong, cell=cell)
    assert tcv.chunk_is_reusable(wrong, **keys) is False
    assert tcv.chunk_is_reusable({**right, "checkpoint_sha256": "0" * 64}, **keys) is False


def test_report_refuses_a_grid4x4_dt_chunk_at_another_checkpoint_or_under_other_targets(
    tmp_path: Path,
) -> None:
    """``BRIEF_39`` §4 T-report, for the grid4x4 subject: ``report`` re-derives the checkpoint
    against A20(a)'s pin and the 16 prompts against the pinned artifact, at step 6.  The control
    passes step 6 and stops at step 7 (no anchors on its draw), without writing anything."""
    _needs_checkpoints()
    _needs_parity((GRID, 1001),)
    cell = _grid_cell("dt", tcv.GRID4X4_ARM, 1001, seed=101)
    config, routes = _grid_digests(1001)
    work = tmp_path / "work"
    _canary_record(work)
    kwargs = {
        "work_dir": work, "out_path": tmp_path / "out" / "artifact.json", "output_root": OUTPUT_ROOT,
        "out_root": DRAWS_ROOT, "data_dir": DATA, "cells": [cell],
    }

    tcv.write_chunk(
        _grid_payload(cell, config_sha=config, routes_sha=routes, checkpoint_sha256="9" * 64),
        work_dir=work,
    )
    with pytest.raises(ValueError, match="not the registered ones"):
        tcv.report(**kwargs)

    shifted = {ix: value + 1.0 for ix, value in TARGETS.items()}
    tcv.write_chunk(
        _grid_payload(
            cell, config_sha=config, routes_sha=routes, target_rtg=shifted,
            rtg_first=dict(shifted), rtg_last=dict(shifted),
            rtg_series={ix: [shifted[ix]] * DECISIONS for ix in IDS},
        ),
        work_dir=work,
    )
    with pytest.raises(ValueError, match="16 registered prompts"):
        tcv.report(**kwargs)

    tcv.write_chunk(_grid_payload(cell, config_sha=config, routes_sha=routes), work_dir=work)
    with pytest.raises(ValueError, match=r"carries anchors \[\]"):
        tcv.report(**kwargs)
    assert not (tmp_path / "out").exists(), "a refusal writes nothing"


def test_report_refuses_a_grid4x4_dt_chunk_counted_against_another_support_range(
    tmp_path: Path,
) -> None:
    """B.7-2(i): the in-support block sums the chunks' per-id counts against the PINNED ranges, so a
    chunk whose recorded range differs was counted against something else and is refused at step 6.
    Written after the implementation; its evidence is the mutation that removes the refusal."""
    _needs_checkpoints()
    _needs_parity((GRID, 1001),)
    cell = _grid_cell("dt", tcv.GRID4X4_ARM, 1001, seed=101)
    config, routes = _grid_digests(1001)
    work = tmp_path / "work"
    _canary_record(work)
    moved = {ix: list(SUPPORT[ix]) for ix in IDS}
    moved["D1"] = [moved["D1"][0] - 1.0, moved["D1"][1]]
    tcv.write_chunk(
        _grid_payload(cell, config_sha=config, routes_sha=routes, support_range=moved), work_dir=work
    )
    with pytest.raises(ValueError, match="support_range"):
        tcv.report(
            work_dir=work, out_path=tmp_path / "out" / "artifact.json", output_root=OUTPUT_ROOT,
            out_root=DRAWS_ROOT, data_dir=DATA, cells=[cell],
        )
    assert not (tmp_path / "out").exists()


# ==================================================================================
# B.6-2(3) -- the declaration, the artifact's name, and the manifest
# ==================================================================================
def test_the_grid4x4_stage_is_its_own_whole_declaration() -> None:
    """The pre-flight's probe (iv): ``declarations_for('grid4x4_confirmatory', None)`` returned
    hz1x1's 4,700 as "every declared cell", so a COMPLETE campaign would have been refused as 700
    undeclared chunks.  *Mutation:* ``whole = None`` for this stage -> dies here."""
    whole, sliced = tcv.declarations_for(tcv.STAGE_GRID4X4, None)
    assert len(whole) == 700 and len(sliced) == 700
    assert whole == sliced == tcv.grid4x4_cells()
    names = {tcv.cell_chunk_name(c) for c in whole}
    assert all(name.startswith(f"cell_{GRID}_") for name in names)
    hz = {tcv.cell_chunk_name(c) for c in tcv.declared_cells(None)}
    assert names.isdisjoint(hz), "the two declarations are disjoint by the scenario prefix"
    # hz1x1's own pairs are untouched
    assert [len(x) for x in tcv.declarations_for(None, None)] == [4700, 4700]
    assert [len(x) for x in tcv.declarations_for(tcv.STAGE_CONFIRMATORY, None)] == [4700, 1200]


def test_the_artifact_is_named_by_stage_and_main_uses_that_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B.6-2(3): the grid4x4 stage writes ``p7_3d_grid4x4.json``; P7.3a's and P7.3b's names are
    unchanged.  ``main`` is driven with ``report`` substituted, so the NAME is what is pinned."""
    expected = {
        tcv.STAGE_GRID4X4: "p7_3d_grid4x4.json",
        None: "p7_3a_zero_shot.json",
        tcv.STAGE_CONFIRMATORY: "p7_3a_zero_shot_stage1.json",
        tcv.STAGE_ANCHOR: "p7_3b_anchor.json",
    }
    for stage, name in expected.items():
        assert tcv.artifact_name_for_stage(stage) == name

    seen: list[Path] = []

    def recording_report(**kwargs: Any) -> dict[str, Any]:
        seen.append(Path(kwargs["out_path"]))
        return {"cells": []}

    monkeypatch.setattr(tcv, "report", recording_report, raising=True)
    for stage, name in expected.items():
        argv = ["--out-dir", str(tmp_path), "--work-dir", str(tmp_path / "work"), "report"]
        if stage is not None:
            argv += ["--stage", stage]
        assert tcv.main(argv) == 0
        assert seen[-1] == tmp_path / name


def _campaign_dir(tmp_path: Path) -> Path:
    root = tmp_path / "output" / "p7_3d"
    for relative, text in (
        ("cells/cell_b.json", "b"),
        ("cells/cell_a.json", "a"),
        ("cells/failed/cell_a.json", "moved aside"),
        ("g2/g2_measurement.json", "g2"),
        ("artifacts/p7_3d_grid4x4.json", "artifact"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    sibling = tmp_path / "output" / "p7_3b_anchor" / "cell_x.json"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("another campaign", encoding="utf-8")
    return root


def test_the_manifest_covers_p7_3d_only_is_atomic_and_verifies_with_sha256sum(tmp_path: Path) -> None:
    """B.6-2(3): ``output/SHA256SUMS_p7_3d.txt`` over ``output/p7_3d/`` ONLY, tmp -> mv, re-verified.
    The format is checked by the real ``sha256sum -c`` -- a second route to every digest."""
    root = _campaign_dir(tmp_path)
    record = tcv.write_manifest(campaign_dir=root)

    manifest = tmp_path / "output" / "SHA256SUMS_p7_3d.txt"
    assert record["path"] == str(manifest)
    lines = manifest.read_text(encoding="utf-8").splitlines()
    names = [line.split("  ", 1)[1] for line in lines]
    assert names == sorted(names)
    assert names == [
        "p7_3d/artifacts/p7_3d_grid4x4.json",
        "p7_3d/cells/cell_a.json",
        "p7_3d/cells/cell_b.json",
        "p7_3d/cells/failed/cell_a.json",
        "p7_3d/g2/g2_measurement.json",
    ]
    for line in lines:
        digest, name = line.split("  ", 1)
        assert digest == _sha256_path(tmp_path / "output" / name)
    assert not any("p7_3b_anchor" in name for name in names)
    assert list((tmp_path / "output").glob("*.tmp")) == []
    if shutil.which("sha256sum"):
        check = subprocess.run(
            ["sha256sum", "-c", "--quiet", manifest.name],
            cwd=str(manifest.parent), capture_output=True, text=True,
        )
        assert check.returncode == 0, check.stdout + check.stderr


def test_the_manifest_refuses_before_it_writes_and_its_re_verification_bites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _campaign_dir(tmp_path)
    wrong_name = tmp_path / "output" / "p7_3b_anchor"
    with pytest.raises(ValueError, match="p7_3d"):
        tcv.write_manifest(campaign_dir=wrong_name)
    empty = tmp_path / "empty" / "p7_3d"
    empty.mkdir(parents=True)
    with pytest.raises(ValueError, match="no file"):
        tcv.write_manifest(campaign_dir=empty)
    assert not (tmp_path / "output" / "SHA256SUMS_p7_3d.txt").exists()

    # A file that changes between the write and the re-verification: the second hash of one file
    # disagrees.  *Mutation:* the re-verification removed -> this survives no longer.
    real = tcv._sha256_file
    calls: dict[str, int] = {}

    def drifting(path: str | Path) -> str:
        key = str(path)
        calls[key] = calls.get(key, 0) + 1
        digest = real(path)
        if key.endswith("cell_b.json") and calls[key] == 2:
            return "0" * 64
        return digest

    monkeypatch.setattr(tcv, "_sha256_file", drifting, raising=True)
    with pytest.raises(ValueError, match="cell_b.json"):
        tcv.write_manifest(campaign_dir=root)


def test_the_manifest_is_a_subcommand(tmp_path: Path) -> None:
    root = _campaign_dir(tmp_path)
    assert tcv.main(["manifest", "--campaign-dir", str(root)]) == 0
    assert (tmp_path / "output" / "SHA256SUMS_p7_3d.txt").is_file()


# ==================================================================================
# B.6-2(4) m1 / m2 -- the inputs by DIGEST, and git resolvability BEFORE the token
# ==================================================================================
def test_the_input_pins_are_the_committed_files_digests() -> None:
    """The pin is a DECLARATION -- *these values came from THAT file* -- and moves only with it."""
    assert tcv.P7_3D_CALIBRATION_SHA256 == _sha256_path(DATA / "p7_3d_calibration.json")
    assert tcv.P7_3D_REFERENCE_CELLS_SHA256 == _sha256_path(DATA / "p7_3d_reference_cells.json")
    assert tcv.P7_3D_CAP_E_SHA256 == _sha256_path(DATA / "p7_3d_cap_e.json")


def test_check_inputs_passes_today_and_refuses_a_one_byte_edit_by_name(tmp_path: Path) -> None:
    """m1: the driver checked the three committed inputs by EXISTENCE only.  *Mutation:* the
    comparison made existence-only again -> the edited copy passes -> this dies."""
    _needs_checkpoints()
    _needs_parity((GRID, 1000),)
    lines = tcv.check_campaign_inputs(data_dir=DATA, output_root=OUTPUT_ROOT, draws_root=DRAWS_ROOT)
    joined = "\n".join(lines)
    for needle in ("p7_3d_calibration.json", "p7_3d_reference_cells.json", "p7_3d_cap_e.json",
                   "grid4x4.net.xml", "seed505"):
        assert needle in joined, needle

    copy = tmp_path / "data"
    copy.mkdir()
    for name in ("p7_3d_calibration.json", "p7_3d_reference_cells.json", "p7_3d_cap_e.json"):
        shutil.copyfile(DATA / name, copy / name)
    with (copy / "p7_3d_cap_e.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="p7_3d_cap_e.json"):
        tcv.check_campaign_inputs(data_dir=copy, output_root=OUTPUT_ROOT, draws_root=DRAWS_ROOT)


def test_resume_check_finds_exactly_the_chunk_whose_commit_git_cannot_resolve(
    tmp_path: Path,
) -> None:
    """m2: a 40-hex commit unknown to git makes ``chunk_is_reusable`` RAISE (deliberately, J1(c)),
    which inside ``run_stage`` happened after ``rm -f "$TOKEN"``.  The pre-token check reaches the
    same call on exactly the chunks that would reach it, and refuses there instead."""
    work = tmp_path / "work"
    cells = tcv.grid4x4_cells()
    unknown = "f" * 40
    bad_cell = next(c for c in cells if c["kind"] == "anchor" and c["draw_id"] == 1001)
    ok_cell = next(c for c in cells if c["kind"] == "anchor" and c["draw_id"] == 1002)
    dirty_cell = next(c for c in cells if c["kind"] == "anchor" and c["draw_id"] == 1003)

    tcv.write_chunk(_grid_payload(bad_cell, config_sha="c" * 64, routes_sha="r" * 64,
                                  git_commit=unknown), work_dir=work)
    tcv.write_chunk(_grid_payload(ok_cell, config_sha="c" * 64, routes_sha="r" * 64), work_dir=work)
    # a chunk that fails validation is MOVED ASIDE by run_stage and never reaches git: not reported
    tcv.write_chunk(_grid_payload(dirty_cell, config_sha="c" * 64, routes_sha="r" * 64,
                                  git_commit=unknown, git_dirty=True), work_dir=work)
    (work / tcv.cell_chunk_name(cells[0])).write_text("{ truncated", encoding="utf-8")

    problems = tcv.unresolvable_chunk_commits(work_dir=work, stage=tcv.STAGE_GRID4X4)
    assert [p["chunk"] for p in problems] == [tcv.cell_chunk_name(bad_cell)]
    assert problems[0]["git_commit"] == unknown

    # ... and it IS the raise run_stage would have hit, after the token:
    with pytest.raises(RuntimeError, match="git"):
        tcv.chunk_is_reusable(
            json.loads((work / tcv.cell_chunk_name(bad_cell)).read_bytes()),
            cell=bad_cell, out_root=DRAWS_ROOT, output_root=OUTPUT_ROOT, data_dir=DATA,
        )
    assert tcv.main(["--work-dir", str(work), "resume-check", "--stage", tcv.STAGE_GRID4X4]) == 2
    (work / tcv.cell_chunk_name(bad_cell)).unlink()
    assert tcv.main(["--work-dir", str(work), "resume-check", "--stage", tcv.STAGE_GRID4X4]) == 0


# ==================================================================================
# B.5-1 -- the DT re-roll check, on stubs (the real one is thirteen SUMO episodes, run by hand)
# ==================================================================================
def _stub_reroll_worker(task: tuple[dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
    """A stand-in for ``run_cell`` in a spawned worker.  The MODE travels in ``out_root`` because
    every task carries identical keyword arguments, exactly as the real runner's do."""
    cell, kwargs, role = task
    mode = str(kwargs["out_root"])
    payload = _grid_payload(cell, config_sha="c" * 64, routes_sha="r" * 64, seconds=float(len(role)))
    if mode == "stub:one_action" and role == "pool_01":
        payload["actions"] = [list(row) for row in payload["actions"]]
        payload["actions"][0][0] = (payload["actions"][0][0] + 1) % N_ACTIONS
    if mode == "stub:no_actions":
        payload.pop("actions")
    if mode == "stub:fails" and role == "w1":
        return {"role": role, "ok": False, "payload": None, "error": "ValueError: e_sumo 123.456"}
    return {"role": role, "ok": True, "payload": payload, "error": None}


def test_the_reroll_cell_is_the_fenced_smoke_cell_under_the_registered_prompt() -> None:
    cell = tcv.dt_reroll_check_cell()
    assert cell == {
        "kind": "dt", "subject": tcv.GRID4X4_SUBJECT, "arm": tcv.GRID4X4_ARM, "seed": 101,
        "draw_id": FENCED_DRAW, "scenario": GRID, "stage": tcv.PILOT_STAGE,
    }
    assert tcv.PILOT_STAGE not in tcv.STAGES and tcv.PILOT_STAGE != tcv.STAGE_GRID4X4
    assert FENCED_DRAW not in tcv.HELD_OUT_DRAWS
    declared = {tcv.cell_chunk_name(c) for c in tcv.grid4x4_cells()}
    assert tcv.cell_chunk_name(cell) not in declared, "a stray re-roll chunk is no declared cell"


def test_the_comparison_excludes_the_clock_by_name_and_nothing_else() -> None:
    base = _valid_dt()
    same_but_slower = {**base, "seconds": base["seconds"] + 9.0}
    result = tcv.compare_reroll_payloads([base, same_but_slower])
    assert result["verdict"] == "IDENTICAL" and result["differing_fields"] == []
    assert len(set(result["sha256_minus_clocks"])) == 1
    assert tuple(tcv.REROLL_CLOCK_FIELDS) == ("seconds",)

    one_action = json.loads(json.dumps(base))
    one_action["actions"][359][15] = (one_action["actions"][359][15] + 1) % N_ACTIONS
    result = tcv.compare_reroll_payloads([base, same_but_slower, one_action])
    assert result["verdict"] == "NOT IDENTICAL" and result["differing_fields"] == ["actions"]
    assert len(set(result["sha256_minus_clocks"])) == 2

    for field in ("rtg_series", "reward_series", "e_sumo", "att_env", "episode_reward", "n_teleports"):
        assert field in tcv.REROLL_COMPARED_FIELDS
        with pytest.raises(ValueError, match=field):
            tcv.compare_reroll_payloads([base, {k: v for k, v in base.items() if k != field}])


def test_a_one_action_difference_in_one_pooled_roll_is_not_identical_and_prints_no_value(
    tmp_path: Path,
) -> None:
    """B.5-1's named test: the check on a stub, a one-action difference -> NOT IDENTICAL.

    Through the REAL runner: a spawn pool of one worker, then a spawn pool of three (the campaign's
    twelve, reduced so a unit test stays cheap), every chunk under ``fenced_do_not_report``.
    *Mutation:* the comparison ignoring ``actions`` -> IDENTICAL -> this dies.
    """
    g2 = tmp_path / "g2"
    result = tcv.run_dt_reroll_check(
        g2_dir=g2, out_root="stub:one_action", output_root=tmp_path, data_dir=DATA,
        canary_seconds=0.8, workers=3, worker=_stub_reroll_worker,
    )
    assert result["verdict"] == "NOT IDENTICAL"
    assert result["differing_fields"] == ["actions"]
    line = result["line"]
    assert line.startswith("dt_reroll_check NOT IDENTICAL")
    assert "actions" in line
    for value in ("123.456", "78.9", "4242", "-124.9"):
        assert value not in line, f"an outcome value reached the result line: {value}"
    run_dir = Path(result["run_dir"])
    assert run_dir.parent == g2
    written = sorted(path.name for path in run_dir.iterdir())
    assert written == ["pool_00.json", "pool_01.json", "pool_02.json", "verdict.json", "w1.json"]
    for name in ("w1.json", "pool_00.json", "pool_01.json", "pool_02.json"):
        record = json.loads((run_dir / name).read_text(encoding="utf-8"))
        assert set(record) >= {"format_version", tcv.FENCED_KEY}
        assert "e_sumo" not in record, "the outcome sits under the fence key only"

    identical = tcv.run_dt_reroll_check(
        g2_dir=tmp_path / "g2b", out_root="stub:same", output_root=tmp_path, data_dir=DATA,
        canary_seconds=0.8, workers=3, worker=_stub_reroll_worker,
    )
    assert identical["verdict"] == "IDENTICAL"
    assert identical["line"].startswith("dt_reroll_check IDENTICAL")


def test_a_failed_roll_is_not_a_verdict_and_a_missing_field_is_not_a_comparison(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="ValueError") as raised:
        tcv.run_dt_reroll_check(
            g2_dir=tmp_path / "g2", out_root="stub:fails", output_root=tmp_path, data_dir=DATA,
            canary_seconds=0.8, workers=2, worker=_stub_reroll_worker,
        )
    assert "123.456" not in str(raised.value), "a failure message may carry an outcome; it stays fenced"
    with pytest.raises(ValueError, match="actions"):
        tcv.run_dt_reroll_check(
            g2_dir=tmp_path / "g2c", out_root="stub:no_actions", output_root=tmp_path,
            data_dir=DATA, canary_seconds=0.8, workers=2, worker=_stub_reroll_worker,
        )


def test_the_reroll_subcommand_exits_two_on_not_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for verdict, code in (("NOT IDENTICAL", 2), ("IDENTICAL", 0)):
        line = f"dt_reroll_check {verdict}: stub"
        monkeypatch.setattr(
            tcv, "run_dt_reroll_check",
            lambda **_k: {"verdict": verdict, "line": line},  # noqa: B023 - consumed in-loop
            raising=True,
        )
        assert tcv.main(["dt-reroll-check", "--g2-dir", str(tmp_path)]) == code
        assert capsys.readouterr().out.strip().splitlines()[-1] == line


# ==================================================================================
# B.7.1-2 -- the per-intersection return on EVERY grid4x4 chunk, by the probe's two routes
# ==================================================================================
class _Ix:
    def __init__(self, ix_id: str, lanes: list[str]) -> None:
        self.id = ix_id
        self.incoming_lanes = list(lanes)


def _post_step(
    rewards: dict[str, float], waiting: dict[str, float], translation: dict[str, str] | None = None
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "intersections": {ix: {"reward": value} for ix, value in rewards.items()},
        "lane_waiting_vehicle_count": dict(waiting),
    }
    if translation is not None:
        info["lane_id_translation"] = dict(translation)
    return info


def test_the_return_lanes_follow_the_frame_the_info_is_keyed_in() -> None:
    """The anchor's info is keyed by SUMO lane ids -- the SUMO probe's own call.  The DT's ALIGNED
    info re-keys lane counts to canonical ids and carries ``lane_id_translation`` (canonical ->
    SUMO), so the same SUMO incoming lanes are read through its inverse.  A SUMO lane with no
    canonical partner refuses rather than being dropped from the second route."""
    intersections = [_Ix("A", ["a1", "a2"]), _Ix("B", ["b1"])]
    raw = _post_step({"A": -3.0, "B": -1.0}, {"a1": 2, "a2": 1, "b1": 1})
    assert tcv.grid4x4_return_lanes(intersections, raw) == {"A": ["a1", "a2"], "B": ["b1"]}
    aligned = _post_step(
        {"A": -3.0, "B": -1.0}, {"ca1": 2, "ca2": 1, "cb1": 1},
        translation={"ca1": "a1", "ca2": "a2", "cb1": "b1"},
    )
    assert tcv.grid4x4_return_lanes(intersections, aligned) == {"A": ["ca1", "ca2"], "B": ["cb1"]}
    broken = _post_step({"A": -3.0, "B": -1.0}, {"ca1": 2, "cb1": 1}, translation={"ca1": "a1", "cb1": "b1"})
    with pytest.raises(ValueError, match="a2"):
        tcv.grid4x4_return_lanes(intersections, broken)


def test_the_per_intersection_return_is_the_post_step_sum_by_two_routes() -> None:
    """B.7.1-2's quantity: per intersection, the sum over the POST-STEP infos of its reward, equal
    under ``==`` to minus its incoming lanes' waiting counts; a disagreement refuses by name."""
    lanes = {"A": ["a1", "a2"], "B": ["b1"]}
    infos = [
        _post_step({"A": -3.0, "B": -1.0}, {"a1": 2, "a2": 1, "b1": 1}),
        _post_step({"A": -1.0, "B": 0.0}, {"a1": 1, "a2": 0, "b1": 0}),
    ]
    returns, from_lanes = tcv.per_intersection_local_returns(infos, lanes)
    assert returns == {"A": -4.0, "B": -1.0} == from_lanes
    infos[1]["intersections"]["B"]["reward"] = -5.0
    with pytest.raises(ValueError, match="'B'"):
        tcv.per_intersection_local_returns(infos, lanes)


def test_a_grid4x4_chunk_whose_return_routes_disagree_or_are_absent_is_refused() -> None:
    """``validate_cell_payload`` re-checks the two routes at CONSUMPTION, on both kinds of cell."""
    payload = _valid_dt()
    tcv.validate_cell_payload(payload)
    moved = {**payload["local_return"], "C3": payload["local_return"]["C3"] - 1.0}
    with pytest.raises(ValueError, match="C3"):
        tcv.validate_cell_payload({**payload, "local_return_from_lanes": moved})
    for key in ("local_return", "local_return_from_lanes"):
        with pytest.raises(ValueError, match=key):
            tcv.validate_cell_payload({k: v for k, v in payload.items() if k != key})
    anchor = _grid_payload(_grid_cell("anchor", "fixedtime", 1001), config_sha="c" * 64, routes_sha="r" * 64)
    tcv.validate_cell_payload(anchor)
    with pytest.raises(ValueError, match="local_return"):
        tcv.validate_cell_payload({**anchor, "local_return": {"A0": -1.0}, "local_return_from_lanes": {"A0": -1.0}})


# ==================================================================================
# B.7-2(iii) and (iv) -- the reference-cell comparison, and A21's sentences
# ==================================================================================
def test_the_a21_scope_items_are_carried_verbatim() -> None:
    """A21(b)'s two items, character for character, as ``PREREGISTRATION.md`` registers them."""
    assert list(tcv.A21_SCOPE_SENTENCES) == _a21_items()
    assert len(_a21_items()) == 2


def test_the_reference_cells_are_read_only_at_their_pinned_digest(tmp_path: Path) -> None:
    loaded = tcv.load_reference_cells(data_dir=DATA)
    assert len(loaded["cells"]) == 6
    (tmp_path / "p7_3d_reference_cells.json").write_bytes(
        (DATA / "p7_3d_reference_cells.json").read_bytes() + b" "
    )
    with pytest.raises(ValueError, match="sha256"):
        tcv.load_reference_cells(data_dir=tmp_path)


def _perturbed(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, list):
        return [*value, "DEFAULT_VEHTYPE"]
    raise TypeError(type(value))


def test_the_reference_cell_comparison_covers_every_shared_field_and_the_halting_rule() -> None:
    """B.7-2(iii): bit-for-bit on every field BOTH records (the frozen rows' keys, read from disk),
    through the documented aliases; the halting counts only where the frozen run CHECKED (it records 0
    where it did not, the chunk ``None`` -- measured); ``episode_reward`` is not in the frozen record,
    so it cannot be compared (the amendment names it; the repository wins).
    *Mutations:* one compared field dropped, the halting rule inverted -> this dies."""
    frozen = _frozen_rows()
    assert [tuple(pair) for pair in tcv.REFERENCE_CELL_COMPARED_FIELDS] == list(FROZEN_TO_CHUNK.items())
    assert tuple(tcv.REFERENCE_CELL_HALTING_COUNTS) == HALTING_COUNTS
    assert all("episode_reward" not in row for row in frozen.values())
    assert {row["halting_checked"] for row in frozen.values()} == {True, False}
    for (arm, draw), row in frozen.items():
        chunk = _grid_payload(_grid_cell("anchor", arm, draw), config_sha="c" * 64, routes_sha="r" * 64)
        chunk.update({chunk_key: row[key] for key, chunk_key in FROZEN_TO_CHUNK.items()})
        if row["halting_checked"]:
            chunk.update({key: row[key] for key in HALTING_COUNTS})
        else:
            assert chunk["halting_n_lane_seconds"] is None and row["halting_n_lane_seconds"] == 0
        assert tcv.reference_cell_differences(chunk, row) == [], (arm, draw)
        assert tcv.reference_cell_differences({**chunk, "episode_reward": -1.0}, row) == []
        for chunk_key in FROZEN_TO_CHUNK.values():
            moved = {**chunk, chunk_key: _perturbed(chunk[chunk_key])}
            assert tcv.reference_cell_differences(moved, row) == [chunk_key], (arm, draw, chunk_key)
        for key in HALTING_COUNTS:
            moved = {**chunk, key: 7}
            assert tcv.reference_cell_differences(moved, row) == ([key] if row["halting_checked"] else [])


# ==================================================================================
# The driver's TEXT -- what the executed tests below then prove behaviourally
# ==================================================================================
def _driver_code() -> str:
    return "\n".join(
        line for line in DRIVER.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def test_every_module_call_puts_common_before_the_subcommand() -> None:
    """B1: ``build_parser`` defines the roots on the PARENT parser, so ``"${COMMON[@]}"`` after the
    subcommand is ``error: unrecognized arguments`` and the driver refused at its own canary on
    every machine (``p7_3b_anchor.sh:230`` is the shape)."""
    code = _driver_code()
    calls = code.count("-m offline.transfer_curve")
    assert calls >= 8, "check-inputs, resume-check, canary, dt-reroll-check, record-canary, cells, report, manifest"
    assert code.count('-m offline.transfer_curve "${COMMON[@]}"') == calls
    for subcommand in ("check-inputs", "resume-check", "canary", "dt-reroll-check",
                       "record-canary", "cells", "report", "manifest"):
        assert re.search(rf'"\$\{{COMMON\[@\]\}}"[^\n]*(\\\n[^\n]*)?\b{re.escape(subcommand)}\b', code), subcommand


def test_cells_is_given_the_canary_seconds_and_every_post_token_step_writes_failed() -> None:
    """F-B6-1: without ``--canary-seconds`` every chunk records ``canary_seconds: None`` and
    ``report`` raises on the first one -- after all 700 cells.  M1: FAILED / COMPLETE files on every
    terminal path, ``p7_3b_anchor.sh:251, 265, 360``'s shape."""
    code = _driver_code()
    cells_call = code[code.index('-m offline.transfer_curve "${COMMON[@]}" \\\n  --canary-seconds "$CANARY" cells'):]
    assert cells_call.index('--canary-seconds "$CANARY"') < cells_call.index(" cells ")
    for step in ("record-canary", "cells", "report", "manifest"):
        assert f'|| fail "{step}"' in code, step
    fail_body = code[code.index("fail() {"):code.index("on_signal() {")]
    assert 'tee "$WORK/FAILED"' in fail_body
    signal_body = code[code.index("on_signal() {"):code.index("trap on_signal INT TERM HUP")]
    assert 'tee "$WORK/FAILED"' in signal_body
    assert 'tee "$WORK/COMPLETE"' in code
    token = code.index('rm -f "$TOKEN"')
    assert code.index('rm -f "$WORK/FAILED" "$WORK/COMPLETE"') > token
    assert code.index('record-canary --line "$CANARY_LINE"') > token
    assert code.index('record-canary --line "$CANARY_LINE"') < code.index(" cells --stage")


def test_the_driver_derives_its_tree_from_its_own_location_and_refuses_the_implementers() -> None:
    """B.7.1-1 (i), (ii) and (iv) by text; the executed tests below prove them behaviourally.  J1(e):
    the campaign runs from a DETACHED run worktree no session edits -- any commit or merge reaching
    the running tree would make every chunk already rolled "rolled by different code"."""
    code = _driver_code()
    assert 'WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)' in code
    assert "WORK_TREE=/home/filip/rltraffic-p73d" not in code, "hardcoded again"
    assert f"IMPLEMENTER_TREE={IMPLEMENTER_TREE}\n" in code
    refusal = code.index('if [ "$WORK_TREE" = "$IMPLEMENTER_TREE" ]; then')
    assert refusal < code.index("REFUSING TO START: offline.transfer_curve loaded from")
    assert refusal < code.index('rm -f "$TOKEN"')
    assert '"$WORK_TREE"/*) ;;' in code, "the import check asserts the module loaded from under it"
    header = DRIVER.read_text(encoding="utf-8")
    assert f"bash {HEADER_DRIVER_PATH} confirmatory" in header
    assert "worktree add --detach /home/filip/rltraffic-p73d-run" in header


# ==================================================================================
# The EXECUTED driver -- ONE fixture builds every executed test's driver (B.7-3)
# ==================================================================================
def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    ).stdout


def _snapshot_clone(base: Path) -> Path:
    """A CLEAN clone whose committed tree is this working copy, byte for byte.

    ``git clone --shared`` at HEAD, then every path the working copy changed -- tracked
    modifications and deletions, and untracked files git does not ignore -- is copied in and
    committed IN THE CLONE (hooks off, a throwaway identity).  Nothing is written to this
    repository: ``--shared`` only reads its objects.
    """
    clone = base / "tree"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO_ROOT), str(clone)],
        capture_output=True, check=True,
    )
    _git("checkout", "--quiet", HEAD_SHA, cwd=clone)
    changed = [p for p in _git("diff", "--name-only", "-z", "HEAD", cwd=REPO_ROOT).split("\0") if p]
    untracked = [
        p for p in _git("ls-files", "--others", "--exclude-standard", "-z", cwd=REPO_ROOT).split("\0") if p
    ]
    for relative in changed + untracked:
        source, target = REPO_ROOT / relative, clone / relative
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        elif target.exists():
            target.unlink()
    _commit_clone(clone, "snapshot of the working copy under test")
    return clone


def _commit_clone(clone: Path, message: str) -> None:
    _git("add", "-A", cwd=clone)
    if _git("status", "--porcelain", cwd=clone):
        _git("-c", "user.email=t@t", "-c", "user.name=t", "-c", "core.hooksPath=/dev/null",
             "commit", "--quiet", "-m", message, cwd=clone)
    assert _git("status", "--porcelain", cwd=clone) == "", "the clone must be clean"


#: The ``dt-reroll-check`` invocation, replaced by a stub in every executed test.
_REROLL_CALL = (
    'PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" --canary-seconds '
    '"$CANARY" dt-reroll-check --g2-dir "$G2_DIR" --workers "$WORKERS"'
)
_IDENTICAL_STUB = "echo 'dt_reroll_check IDENTICAL (stubbed by tests/test_p7_3d_campaign_path.py)'"
_NOT_IDENTICAL_STUB = (
    "sh -c 'echo \"dt_reroll_check NOT IDENTICAL: differing fields [actions] (stub)\"; exit 2'"
)

#: The draw-5 shim: the REAL module with ONE function replaced -- the grid4x4 declaration, draw
#: 1000's seven cells moved to draw 5 and put FIRST (the dt-seed-101 and fixedtime pair leading), so
#: a limit of at most seven can only ever roll the fenced smoke draw -- and the REAL ``main``.
_SHIM = '''"""tests/test_p7_3d_campaign_path.py: the REAL module with draw 1000's seven cells on draw 5."""
import offline.transfer_curve as real

_declared = real.grid4x4_cells


def fenced_declaration():
    cells = [dict(cell) for cell in _declared()]
    for cell in cells:
        if cell["draw_id"] == 1000:
            cell["draw_id"] = 5
    first = [c for c in cells if c["draw_id"] == 5 and c["kind"] == "dt" and c["seed"] == 101]
    first += [c for c in cells if c["draw_id"] == 5 and c["arm"] == "fixedtime"]
    first += [c for c in cells if c["draw_id"] == 5 and c not in first]
    return first + [c for c in cells if c not in first]


real.grid4x4_cells = fenced_declaration

if __name__ == "__main__":
    raise SystemExit(real.main())
'''

#: The only way to a non-zero ``--limit``: the fenced declaration's leading draw-5 cells.
MAX_FENCED_CELLS = 7


def _substitute(text: str, old: str, new: str, count: int) -> str:
    found = text.count(old)
    assert found == count, f"expected {count} occurrence(s) of {old!r} in the driver, found {found}"
    return text.replace(old, new)


def _needs_the_driver_environment() -> None:
    if not MAIN_INTERPRETER.is_file():
        pytest.skip(f"needs the main tree's interpreter at {MAIN_INTERPRETER}")
    if shutil.which("setsid") is None:
        pytest.skip("setsid is not installed")
    for seed in tc.TRAINING_SEEDS:
        path = OUTPUT_ROOT / tc.GRID4X4_CHECKPOINT_SUBDIR / f"{tc.GRID4X4_CHECKPOINT_STEM}{seed}.pt"
        if not path.is_file():
            pytest.skip(f"{path} is absent: the driver refuses without A20(a)'s checkpoints")
    for draw in (FENCED_DRAW, *tcv.HELD_OUT_DRAWS):
        config = _parity(GRID, draw) / "noteleport.sumocfg"
        if not config.is_file():
            pytest.skip(f"{config} is absent: the driver refuses without the rendered parity band")


class DriverSandbox:
    """One executed test's driver: a clean snapshot clone, every root in the sandbox, ``--limit 0``.

    Built ONLY by the :func:`driver_sandbox` fixture (B.7-3's rule: *a test of a refusal must not be
    able to run the experiment when the refusal breaks*).  ``CAMPAIGN_DIR`` and ``TOKEN`` point
    into the sandbox and ``--limit`` is 0 by construction; a test that must roll cells asks for
    ``fenced_cells`` and gets the draw-5 declaration with it, never the real one.
    """

    def __init__(self, *, base: Path, clone: Path, driver: Path, sandbox: Path) -> None:
        self.base = base
        self.clone = clone
        self.driver = driver
        self.sandbox = sandbox
        self.token = sandbox / "TOKEN_confirmatory"
        self.campaign = sandbox / "p7_3d"
        self.work = self.campaign / "cells"
        self.artifacts = self.campaign / "artifacts"
        self.head = _git("rev-parse", "HEAD", cwd=clone).strip()

    def write_token(self) -> None:
        self.token.write_text("authorised by tests/test_p7_3d_campaign_path.py\n", encoding="utf-8")

    def run(self, timeout: float = 900.0) -> subprocess.CompletedProcess[str]:
        """``setsid --wait`` makes the driver a process-group leader, which it requires."""
        result = subprocess.run(
            ["setsid", "--wait", "bash", str(self.driver), "confirmatory"],
            capture_output=True, text=True, cwd=str(self.clone), timeout=timeout,
        )
        # Kept with pytest's retained tmp_path: the capture a reviewer reads when a test fails.
        (self.base / "driver_capture.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        return result

    def chunks(self) -> set[str]:
        return {str(path.relative_to(self.sandbox)) for path in self.sandbox.rglob("cell_*.json")}

    def assert_no_experiment_started(self, *, already_there: set[str] | None = None) -> None:
        """The refusal's other half: no chunk was written -- nothing was rolled."""
        assert self.chunks() == (already_there or set()), (
            "a refused start wrote a chunk: the experiment began"
        )


@pytest.fixture
def driver_sandbox(tmp_path: Path) -> Any:
    """B.7-3: the ONE place an executed driver is built.  Returns a builder.

    ``fenced_cells`` (0 unless asked, at most :data:`MAX_FENCED_CELLS`) is the ONLY route to a
    non-zero ``--limit``, and it installs :data:`_SHIM` on the ``cells`` and ``report`` lines at the
    same time, so a non-zero limit can only select draw-5 cells.  ``implementer_tree_here`` points
    the driver's ``IMPLEMENTER_TREE`` at this clone, to watch that refusal fire.
    """
    _needs_the_driver_environment()

    def build(
        *,
        reroll: str = _IDENTICAL_STUB,
        fenced_cells: int = 0,
        implementer_tree_here: bool = False,
        populate: Any = None,
    ) -> DriverSandbox:
        if not 0 <= int(fenced_cells) <= MAX_FENCED_CELLS:
            raise ValueError(f"fenced_cells must be 0..{MAX_FENCED_CELLS}, got {fenced_cells}")
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        clone = _snapshot_clone(tmp_path)
        path = clone / "offline" / "campaigns" / "p7_3d_grid4x4.sh"
        text = path.read_text(encoding="utf-8")
        text = _substitute(text, "CAMPAIGN_DIR=$MAIN/output/p7_3d\n", f"CAMPAIGN_DIR={sandbox / 'p7_3d'}\n", 1)
        text = _substitute(
            text, "TOKEN=$MAIN/output/p7_3d_runs/TOKEN_confirmatory\n",
            f"TOKEN={sandbox / 'TOKEN_confirmatory'}\n", 1,
        )
        text = _substitute(text, _REROLL_CALL, reroll, 1)
        text = _substitute(
            text, 'cells --stage "$STAGE_ARG" --workers "$WORKERS" || fail "cells"',
            f'cells --stage "$STAGE_ARG" --workers "$WORKERS" --limit {int(fenced_cells)} || fail "cells"', 1,
        )
        if fenced_cells:
            shim = tmp_path / "shim"
            shim.mkdir()
            (shim / "p73d_fenced_declaration.py").write_text(_SHIM, encoding="utf-8")
            for tail in ('\\\n  --canary-seconds "$CANARY" cells', '\\\n  report --stage'):
                text = _substitute(
                    text,
                    f'PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${{COMMON[@]}}" {tail}',
                    f'PYTHONPATH={shim}:$WORK_TREE "$PY" -P -m p73d_fenced_declaration "${{COMMON[@]}}" {tail}',
                    1,
                )
        if implementer_tree_here:
            text = _substitute(
                text, f"IMPLEMENTER_TREE={IMPLEMENTER_TREE}\n", f"IMPLEMENTER_TREE={clone}\n", 1
            )
        path.write_text(text, encoding="utf-8")
        _commit_clone(clone, "roots redirected to the sandbox")
        built = DriverSandbox(base=tmp_path, clone=clone, driver=path, sandbox=sandbox)
        if populate is not None:
            populate(built)
        return built

    return build


def test_the_fixture_builds_every_driver_with_sandbox_roots_and_limit_zero(driver_sandbox: Any) -> None:
    """B.7-3's rule, by construction: the redirected driver's roots are the sandbox's and its
    ``cells`` line carries ``--limit 0`` unless ``fenced_cells`` asked for more; the fenced route is
    capped at the leading draw-5 cells.  Written after the fixture; its evidence is the mutation that
    removes the fixture's limit."""
    sb = driver_sandbox()
    text = sb.driver.read_text(encoding="utf-8")
    assert f"CAMPAIGN_DIR={sb.sandbox / 'p7_3d'}\n" in text
    assert f"TOKEN={sb.sandbox / 'TOKEN_confirmatory'}\n" in text
    assert 'cells --stage "$STAGE_ARG" --workers "$WORKERS" --limit 0 || fail "cells"' in text
    assert "$MAIN/output/p7_3d" not in text and "p7_3d_runs/TOKEN" not in text
    with pytest.raises(ValueError, match="fenced_cells"):
        driver_sandbox(fenced_cells=MAX_FENCED_CELLS + 1)


def test_the_campaign_path_from_the_drivers_argument_to_a_written_grid4x4_chunk(
    driver_sandbox: Any,
) -> None:
    """B.6-2(5): the end-to-end test.  The delivered driver, redirected, with a token, on draw 5 with
    ``--limit 2`` (one DT cell seed 101, one fixedtime) through the real module; then its own
    ``report`` on the two chunks.  B.7.1-2: both chunks carry the 16 per-intersection returns."""
    try:
        import traci  # noqa: F401
    except ImportError:
        pytest.skip("traci is not importable")
    if shutil.which("sumo") is None:
        pytest.skip("the sumo binary is not on PATH")
    if not os.environ.get("RLTRAFFIC_GRID4X4_RESCO"):
        pytest.skip("RLTRAFFIC_GRID4X4_RESCO is unset: this test runs two grid4x4 SUMO episodes")

    sb = driver_sandbox(fenced_cells=2)
    sb.write_token()
    result = sb.run()
    shutil.rmtree(sb.clone, ignore_errors=True)
    output = result.stdout + result.stderr

    # ---- the driver: past the canary, the token consumed ONCE, report's refusal through `fail`
    assert "not a process-group leader" not in output
    assert "REFUSING TO START" not in output, output[-3000:]
    assert result.returncode == 1, f"exit {result.returncode}:\n{output[-3000:]}"
    assert not sb.token.exists(), "the token must be consumed"
    assert output.count("token consumed and deleted") == 1
    assert (sb.work / "FAILED").read_text(encoding="utf-8").strip() == "CAMPAIGN FAILED at report"
    assert not (sb.work / "COMPLETE").exists()
    assert not (sb.artifacts / "p7_3d_grid4x4.json").exists()

    # ---- report refused ONLY on completeness, and named no undeclared chunk
    assert "698 declared cell(s) have no chunk" in output
    assert "not declared cells of ANY stage" not in output

    # ---- the two chunks, under the grid4x4 names
    dt_cell = _grid_cell("dt", tcv.GRID4X4_ARM, FENCED_DRAW, seed=101)
    ft_cell = _grid_cell("anchor", "fixedtime", FENCED_DRAW)
    chunks = sorted(path.name for path in sb.work.glob("cell_*.json"))
    assert chunks == sorted([tcv.cell_chunk_name(dt_cell), tcv.cell_chunk_name(ft_cell)])
    dt = json.loads((sb.work / tcv.cell_chunk_name(dt_cell)).read_bytes())
    ft = json.loads((sb.work / tcv.cell_chunk_name(ft_cell)).read_bytes())

    grid_cfg, grid_routes = _grid_digests(FENCED_DRAW)
    hz_cfg, _hz_routes = _hz1x1_digests(FENCED_DRAW)
    canary_line = next(line for line in output.splitlines() if line.startswith("canary "))
    canary_seconds = float(canary_line.split()[1])
    for chunk, cell in ((dt, dt_cell), (ft, ft_cell)):
        assert chunk["config_sha256"] == grid_cfg and grid_cfg.startswith(GRID4X4_CFG_PREFIX)
        assert chunk["config_sha256"] != hz_cfg and hz_cfg.startswith(HZ1X1_CFG_PREFIX)
        assert chunk["routes_sha256"] == grid_routes
        assert chunk["scenario"] == GRID and chunk["stage"] == tcv.STAGE_GRID4X4
        assert {k: chunk[k] for k in cell} == cell
        assert chunk["format_version"] == tcv.GRID4X4_ARTIFACT_FORMAT_VERSION
        assert chunk["calibration_sha256"] == _sha256_path(DATA / "p7_3d_calibration.json")
        assert chunk["canary_seconds"] == canary_seconds, "F-B6-1: the cells line carries the canary"
        assert chunk["git_commit"] == sb.head and chunk["git_dirty"] is False
        assert chunk["decisions"] == DECISIONS and chunk["n_teleports"] == 0
        assert chunk["intersection_ids"] == IDS
        assert len(chunk["actions"]) == DECISIONS
        assert all(len(row) == N_IDS for row in chunk["actions"])
        assert all(0 <= a < N_ACTIONS for row in chunk["actions"] for a in row)
        assert chunk["vehicle_types_seen"] == ["cf_parity"] and chunk["time_to_teleport_option"] == "-1"
        assert chunk["halting_checked"] is False
        # B.7.1-2: sixteen per-intersection returns over the 360 POST-STEP infos, two routes equal,
        # on the anchor as on the DT cell -- the quantity per-intersection rho is defined on.
        assert sorted(chunk["local_return"]) == sorted(IDS)
        assert chunk["local_return"] == chunk["local_return_from_lanes"]
        assert all(value <= 0.0 and value == int(value) for value in chunk["local_return"].values())
    # ... and it IS the 360-decision sum, not the reward series': the series is read before each act,
    # so it lacks the LAST decision's reward. That difference is the last post-step reward -- integral,
    # never positive -- and on a real horizon with queued vehicles it is non-zero somewhere.
    last = {ix: dt["local_return"][ix] - sum(dt["reward_series"][ix][1:]) for ix in IDS}
    assert all(value <= 0.0 and value == int(value) for value in last.values()), last
    assert any(value != 0.0 for value in last.values()), (
        "local_return equals the reward series' sum on every intersection: the last decision's "
        "reward is missing from it (B.7.1-2)"
    )

    # ---- the DT chunk: the 16 registered prompts reached the model, per intersection
    assert dt["rtg_first"] == TARGETS
    assert dt["target_rtg"] == TARGETS
    for ix in IDS:
        assert dt["rtg_series"][ix][0] == TARGETS[ix], ix
        assert len(dt["rtg_series"][ix]) == len(dt["reward_series"][ix]) == DECISIONS
        # the series is read BEFORE each act, so it carries the reset's reward and not the last
        # decision's: the per-intersection return is NOT its sum (B.7.1-2's premise, measured here)
        assert dt["reward_series"][ix][0] in (0.0, -0.0)
    assert dt["checkpoint_sha256"] == tc.GRID4X4_CHECKPOINT_SHA256[101]
    assert ft["target_rtg"] is None and ft["checkpoint_sha256"] is None


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_the_headers_own_line_passes_the_guard_and_the_pane_carries_the_drivers_exit_code(
    driver_sandbox: Any,
) -> None:
    """B.3-1(2) + B.5-3 + B.6-2(1) + B.7.1-1(iii), on a redirected COPY of the driver, NO token.

    The header's Step-2 line -- which names the RUN worktree (B.7.1-1(ii)) -- is read from the
    header, its two paths are substituted, and it is typed at a tmux pane's prompt.  The copy lives
    in a sandbox tree and DERIVES its ``WORK_TREE`` from its own location: the capture must name
    that tree, never the source.  The driver must pass the leader check, run the canary, refuse at
    the token -- and the capture's last line must be ``DRIVER EXIT: <n>``, ``n`` equal to the same
    driver's exit code under ``setsid``.  *Mutations:* ``${PIPESTATUS[0]}`` -> ``$?``; ``WORK_TREE``
    hardcoded again -> this dies.
    """
    sb = driver_sandbox()
    header = sb.driver.read_text(encoding="utf-8")
    match = re.search(r"^#\s+Step 2, at ITS PROMPT:\s+(bash .+)$", header, flags=re.MULTILINE)
    assert match, "the header has no Step-2 line"
    line = match.group(1)
    assert "${PIPESTATUS[0]}" in line, "B.5-3: the pane carries the DRIVER's exit status"
    capture = sb.sandbox / "capture.txt"
    line = _substitute(line, HEADER_DRIVER_PATH, str(sb.driver), 1)
    line = _substitute(line, HEADER_CAPTURE_PATH, str(capture), 2)

    session = f"p73d_hdr_{uuid.uuid4().hex[:8]}"
    subprocess.run(["tmux", "new-session", "-d", "-s", session], check=True)
    try:
        subprocess.run(["tmux", "send-keys", "-t", session, line, "Enter"], check=True)
        for _ in range(240):
            if "DRIVER EXIT:" in (capture.read_text(encoding="utf-8") if capture.exists() else ""):
                break
            time.sleep(0.5)
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session], check=False)

    text = capture.read_text(encoding="utf-8") if capture.exists() else ""
    assert "DRIVER EXIT:" in text, f"the driver never finished under the header's own form: {text[-800:]}"
    assert f"WORK_TREE {sb.clone}" in text, "B.7.1-1(iii): the copy derives its tree from its location"
    assert f"WORK_TREE {IMPLEMENTER_TREE}\n" not in text
    assert "not a process-group leader" not in text
    assert "REFUSING TO START: no run token" in text
    lines = text.splitlines()
    canary_at = next(i for i, l in enumerate(lines) if l.startswith("canary "))
    token_at = next(i for i, l in enumerate(lines) if "no run token" in l)
    assert canary_at < token_at, "the refusal must come AFTER the canary (B.6-2(1))"

    direct = sb.run(timeout=300.0)
    shutil.rmtree(sb.clone, ignore_errors=True)
    assert direct.returncode == 2, direct.stdout + direct.stderr
    assert text.rstrip("\n").splitlines()[-1] == f"DRIVER EXIT: {direct.returncode}"
    assert not sb.campaign.exists(), "a refused start creates nothing"
    sb.assert_no_experiment_started()


def test_a_not_identical_reroll_refuses_the_token(driver_sandbox: Any) -> None:
    """B.5-1: NOT IDENTICAL -> the driver refuses to consume the token, and nothing is rolled.
    *Mutation:* the refusal turned into a warning -> the token is consumed -> this dies."""
    sb = driver_sandbox(reroll=_NOT_IDENTICAL_STUB)
    sb.write_token()
    result = sb.run(timeout=300.0)
    shutil.rmtree(sb.clone, ignore_errors=True)
    assert result.returncode == 2, (result.stdout + result.stderr)[-2000:]
    assert "dt_reroll_check NOT IDENTICAL" in result.stdout, "the result line reaches the capture"
    assert "REFUSING TO START: dt_reroll_check" in result.stderr
    assert sb.token.is_file(), "NOT IDENTICAL must not consume the token"
    assert not sb.campaign.exists()
    sb.assert_no_experiment_started()


def test_an_unresolvable_chunk_commit_refuses_before_the_token(driver_sandbox: Any) -> None:
    """m2, executed: the chunk that made ``run_stage`` raise AFTER ``rm -f "$TOKEN"`` refuses the
    start, the token intact, the chunk where it was, and nothing else written.  *Mutation:* the check
    moved after the token -> the token is gone -> this dies."""
    cell = next(c for c in tcv.grid4x4_cells() if c["kind"] == "anchor" and c["draw_id"] == 1001)

    def populate(sb: DriverSandbox) -> None:
        tcv.write_chunk(
            _grid_payload(cell, config_sha="c" * 64, routes_sha="r" * 64, git_commit="f" * 40),
            work_dir=sb.work,
        )

    sb = driver_sandbox(populate=populate)
    before = sorted(str(p.relative_to(sb.sandbox)) for p in sb.sandbox.rglob("*"))
    existing = sb.chunks()
    sb.write_token()
    result = sb.run(timeout=300.0)
    shutil.rmtree(sb.clone, ignore_errors=True)
    assert result.returncode == 2, (result.stdout + result.stderr)[-2000:]
    assert "REFUSING TO START: a chunk on disk records a commit" in result.stderr
    assert tcv.cell_chunk_name(cell) in result.stderr
    assert sb.token.is_file()
    after = sorted(str(p.relative_to(sb.sandbox)) for p in sb.sandbox.rglob("*") if p != sb.token)
    assert after == before, "nothing moved aside, nothing written"
    sb.assert_no_experiment_started(already_there=existing)


def test_a_driver_copy_in_the_implementers_tree_refuses_to_start(driver_sandbox: Any) -> None:
    """B.7.1-1(iv): J1(e) -- the campaign runs from the detached RUN worktree, and a copy of the
    driver whose derived ``WORK_TREE`` is the implementer's refuses before anything else, the token
    intact.  Exercised by pointing ``IMPLEMENTER_TREE`` at the sandbox clone, so the REAL implementer
    tree is never the one executed.  *Mutation:* the refusal removed -> the start proceeds -> dies."""
    sb = driver_sandbox(implementer_tree_here=True)
    sb.write_token()
    result = sb.run(timeout=300.0)
    shutil.rmtree(sb.clone, ignore_errors=True)
    assert result.returncode == 2, (result.stdout + result.stderr)[-2000:]
    assert "REFUSING TO START: this copy of the driver is in the implementer's worktree" in result.stderr
    assert "/home/filip/rltraffic-p73d-run" in result.stderr, "the refusal names the legal home"
    assert "canary " not in result.stdout, "it refuses before the canary runs"
    assert sb.token.is_file()
    assert not sb.campaign.exists()
    sb.assert_no_experiment_started()


# ==================================================================================
# B.7-2's PIN: a COMPLETE synthetic set through the DRIVER -> report -> manifest -> COMPLETE
# ==================================================================================
REFERENCE_DRAWS = (1000, 1001, 1002)
SEED_DELTA = {101: -0.25, 202: -0.125, 303: 0.0, 404: 0.125, 505: 0.25}
#: The draw on which the co-reported env-ATT anchors tie EXACTLY (the 2026-09-17 exclusion rule).
ENV_ZERO_DRAW = 1050
#: The (intersection, draw) pairs on which the per-intersection anchors tie exactly (B.7.1-2).
PER_IX_ZERO_ID = "B2"


def _per_ix_zero(ix: str, draw: int) -> bool:
    return ix == PER_IX_ZERO_ID and (draw - 1000) % 10 == 7


#: The test's OWN statement of the reference-cell mapping (frozen row key -> chunk key).
FROZEN_TO_CHUNK = {
    "e_sumo": "e_sumo", "att_env": "att_env", "n_teleports": "n_teleports",
    "decisions": "decisions", "n_observations": "n_observations",
    "engine_seed_requested": "engine_seed_requested", "engine_seed_drawn": "engine_seed_drawn",
    "e_sumo_n_ids": "n_created", "n_departed": "n_entered", "n_never_inserted": "n_never_entered",
    "n_pending_at_horizon": "n_pending_at_horizon",
    "n_vanished_without_arrival": "n_vanished_without_arrival",
    "vehicle_types_seen": "vehicle_types_seen", "time_to_teleport_option": "time_to_teleport_option",
    "config_sha256": "config_sha256", "routes_sha256": "routes_sha256",
    "halting_checked": "halting_checked",
}
HALTING_COUNTS = ("halting_n_lane_seconds", "halting_n_disagreeing_lane_seconds")


def _frozen_rows() -> dict[tuple[str, int], dict[str, Any]]:
    payload = json.loads((DATA / "p7_3d_reference_cells.json").read_text(encoding="utf-8"))
    return {(str(row["arm"]), int(row["draw_id"])): row for row in payload["cells"]}


def _a21_items() -> list[str]:
    text = (REPO_ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| 2026-09-19 | **A21"))
    return [body for _tag, body in re.findall(r"\*\((i|ii)\) (.+?)\*", row)]


def _complete_set(*, git_commit: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The 700 declared cells as stubs, and the test's OWN expectations, from ONE definition.

    Synthetic values are dyadic (every sum exact, so no summation order can move a bit); on the
    three reference draws the anchors carry the FROZEN values on every compared field and the arm
    is set equal to one anchor, so rho is exactly 0 or 1 there.
    """
    frozen = _frozen_rows()
    payloads: list[dict[str, Any]] = []
    rho: dict[str, dict[int, dict[int, float | None]]] = {"e_sumo": {}, "att_env": {}}
    anchors: dict[str, dict[int, tuple[float, float]]] = {"e_sumo": {}, "att_env": {}}
    arm_values: dict[str, dict[int, dict[int, float]]] = {"e_sumo": {}, "att_env": {}}
    per_ix_rho: dict[str, dict[int, dict[int, float]]] = {ix: {} for ix in IDS}
    for draw in tcv.HELD_OUT_DRAWS:
        k = draw - 1000
        config, routes = _grid_digests(draw)
        if draw in REFERENCE_DRAWS:
            f = {key: float(frozen[("fixedtime", draw)][key]) for key in ("e_sumo", "att_env")}
            m = {key: float(frozen[("maxpressure", draw)][key]) for key in ("e_sumo", "att_env")}
        else:
            f = {"e_sumo": 400.0 + k, "att_env": 300.0 + k}
            m = {"e_sumo": f["e_sumo"] - 256.0,
                 "att_env": f["att_env"] - (0.0 if draw == ENV_ZERO_DRAW else 128.0)}
        r_ft = {ix: -(200.0 + 8.0 * j + float(k % 5)) for j, ix in enumerate(IDS)}
        r_mp = {ix: r_ft[ix] if _per_ix_zero(ix, draw) else r_ft[ix] + 64.0 for ix in IDS}
        for key in ("e_sumo", "att_env"):
            anchors[key][draw] = (f[key], m[key])
        for arm, values, returns in (("fixedtime", f, r_ft), ("maxpressure", m, r_mp)):
            cell = _grid_cell("anchor", arm, draw)
            overrides: dict[str, Any] = {
                "e_sumo": values["e_sumo"], "att_env": values["att_env"],
                "att_horizon": values["att_env"],
                "local_return": dict(returns), "local_return_from_lanes": dict(returns),
            }
            if draw in REFERENCE_DRAWS:
                row = frozen[(arm, draw)]
                overrides.update({chunk: row[key] for key, chunk in FROZEN_TO_CHUNK.items()})
                if row["halting_checked"]:
                    overrides.update({key: row[key] for key in HALTING_COUNTS})
            payloads.append(
                _grid_payload(cell, config_sha=config, routes_sha=routes, git_commit=git_commit, **overrides)
            )
        for seed in tc.TRAINING_SEEDS:
            cell = _grid_cell("dt", tcv.GRID4X4_ARM, draw, seed=seed)
            d: dict[str, float] = {}
            for key, base, scale in (("e_sumo", 0.5 + (k % 4) * 0.0625, 256.0),
                                     ("att_env", 0.25 + (k % 3) * 0.125, 128.0)):
                if draw in REFERENCE_DRAWS:
                    d[key] = m[key] if draw == 1001 else f[key]
                    value: float | None = 1.0 if draw == 1001 else 0.0
                else:
                    target = base + SEED_DELTA[seed]
                    d[key] = f[key] - scale * target
                    value = target
                if f[key] - m[key] == 0.0:
                    value = None
                rho[key].setdefault(draw, {})[seed] = value
                arm_values[key].setdefault(draw, {})[seed] = d[key]
            r_dt: dict[str, float] = {}
            for j, ix in enumerate(IDS):
                target_i = 0.25 + (j % 4) * 0.125 + (k % 2) * 0.0625 + SEED_DELTA[seed] / 2.0
                r_dt[ix] = r_ft[ix] + 64.0 * target_i
                if not _per_ix_zero(ix, draw):
                    per_ix_rho[ix].setdefault(draw, {})[seed] = target_i
            in_support = {ix: DECISIONS - (j % 3) for j, ix in enumerate(IDS)}
            payloads.append(
                _grid_payload(
                    cell, config_sha=config, routes_sha=routes, git_commit=git_commit,
                    e_sumo=d["e_sumo"], att_env=d["att_env"], att_horizon=d["att_env"],
                    local_return=dict(r_dt), local_return_from_lanes=dict(r_dt),
                    n_decisions_in_support=dict(in_support),
                    in_support_counts={
                        ix: {"in_support": in_support[ix], "below": DECISIONS - in_support[ix], "above": 0}
                        for ix in IDS
                    },
                )
            )
    expected = {"rho": rho, "anchors": anchors, "arm_values": arm_values, "per_ix_rho": per_ix_rho}
    return payloads, expected


def _write_stubs(payloads: list[dict[str, Any]], work: Path) -> None:
    """Compact JSON -- a fixture, not a chunk the module writes; ``report`` reads either."""
    work.mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        (work / tcv.cell_chunk_name(payload)).write_text(json.dumps(payload), encoding="utf-8")


def _stats(values: list[float]) -> tuple[float, float, float]:
    """mean, ci95_low, ci95_high by the primitives ``dt_gate.mean_ci95`` uses, over OUR values."""
    import math

    import numpy as np

    data = np.asarray(values, dtype=np.float64)
    mean = float(data.mean())
    ci95 = 1.96 * float(data.std(ddof=1)) / math.sqrt(int(data.size))
    return mean, mean - ci95, mean + ci95


def test_report_writes_the_grid4x4_artifact_through_the_driver_from_a_complete_synthetic_set(
    driver_sandbox: Any, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """B.7-2's pin: the e2e path continued through ``report``, ``manifest`` and ``COMPLETE``.

    The fixture's driver (``--limit 0``: nothing is rolled) over a work dir pre-populated with the
    700 declared cells as stubs carrying the clone's commit, the real demand digests, A20(a)'s
    checkpoints and the 16 real targets; the six reference anchors carry the frozen values.  The
    artifact is then asserted field by field against expectations derived from the stubs' own
    definition -- (i) in-support, (ii) scenario / digest / subject / arms, (iii) the reference-cell
    record, (iv) H3 per A20(e), A21's items verbatim, paired ATT, the denominator diagnostic, per-seed,
    per-draw and per-intersection rho.  Then, in-process, ONE reference value swapped -> refused.
    """
    import numpy as np

    payloads, expected = _complete_set(git_commit="0" * 40)

    def populate(sb: DriverSandbox) -> None:
        _write_stubs([{**p, "git_commit": sb.head} for p in payloads], sb.work)

    sb = driver_sandbox(populate=populate)
    sb.write_token()
    result = sb.run(timeout=1200.0)
    shutil.rmtree(sb.clone, ignore_errors=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-3000:]
    assert (sb.work / "COMPLETE").read_text(encoding="utf-8").startswith("CAMPAIGN COMPLETE (confirmatory)")
    assert not (sb.work / "FAILED").exists()
    manifest = sb.sandbox / "SHA256SUMS_p7_3d.txt"
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 700 + 2  # + canary.json, artifact
    if shutil.which("sha256sum"):
        check = subprocess.run(["sha256sum", "-c", "--quiet", manifest.name], cwd=str(sb.sandbox),
                               capture_output=True, text=True)
        assert check.returncode == 0, check.stdout + check.stderr
    artifact = json.loads((sb.artifacts / "p7_3d_grid4x4.json").read_text(encoding="utf-8"))

    # ---- (ii) the scenario, the digest, the subject, A21's arm set
    assert artifact["format_version"] == "p7.3d-grid4x4/1.0"
    assert artifact["scenario_key"] == GRID and artifact["subject"] == tcv.GRID4X4_SUBJECT
    assert artifact["inputs"]["calibration_sha256"] == _sha256_path(DATA / "p7_3d_calibration.json")
    assert artifact["inputs"]["calibration_sha256"] != tcv.P7_2B_CALIBRATION_SHA256
    assert artifact["arms"]["evaluated"] == {"registered": ["b_mean_k100"], "anchors": ["fixedtime", "maxpressure"]}
    assert set(artifact["arms"]["not_evaluated"]) == {"naive", "random", "b_max_k100", "a_q1.0"}
    assert artifact["n_cells_declared"] == 700 and len(artifact["cells"]) == 700
    assert {row["arm"] for row in artifact["cells"]} == {"b_mean_k100", "fixedtime", "maxpressure"}
    assert {row["subject"] for row in artifact["cells"] if row["kind"] == "dt"} == {tcv.GRID4X4_SUBJECT}
    for row in artifact["cells"]:
        assert not {"rtg_series", "reward_series", "actions"} & set(row), "the series stay in the chunks"
        assert sorted(row["local_return"]) == sorted(IDS)

    # ---- (iii) the six reference cells, bit-for-bit, recorded
    record = artifact["reference_cells"]
    assert record["all_reproduce"] is True and record["n_checked"] == 6
    assert record["artifact_sha256"] == _sha256_path(DATA / "p7_3d_reference_cells.json")
    assert [pair[1] for pair in record["compared_fields"]] == list(FROZEN_TO_CHUNK.values())

    # ---- (iv) per-draw rho, both definitions, under ==
    by_draw = artifact["rho"]["by_draw"]
    per_draw: dict[str, list[float]] = {"e_sumo": [], "att_env": []}
    for draw in tcv.HELD_OUT_DRAWS:
        for key in ("e_sumo", "att_env"):
            seeds = [expected["rho"][key][draw][s] for s in tc.TRAINING_SEEDS]
            want = None if seeds[0] is None else sum(seeds) / len(seeds)
            assert by_draw[str(draw)][key] == want, (draw, key)
            if want is not None:
                per_draw[key].append(want)
    assert by_draw[str(ENV_ZERO_DRAW)]["att_env"] is None
    assert len(per_draw["e_sumo"]) == 100

    # ---- (iv) per-seed rho
    for entry in artifact["rho"]["by_seed"]:
        for key in ("e_sumo", "att_env"):
            values = [expected["rho"][key][d][entry["seed"]] for d in tcv.HELD_OUT_DRAWS]
            usable = [v for v in values if v is not None]
            assert entry[f"mean_rho_{key}"] == sum(usable) / len(usable), (entry["seed"], key)
    assert sorted(entry["seed"] for entry in artifact["rho"]["by_seed"]) == list(tc.TRAINING_SEEDS)

    # ---- (iv) H3 per A20(e): clause 1 confirmatory and scored as arithmetic, clause 2 NOT scored,
    # clause 3 void
    h3 = artifact["h3"]
    for clause in h3["clauses"]:
        mean, low, high = _stats(per_draw[clause["definition"]])
        assert clause["mean_rho"] == mean and clause["ci95_low"] == low and clause["ci95_high"] == high
        if clause["clause"] == 1:
            assert clause["inequality"] == "rho_sumo(b_mean_k100) > 0"
            assert "CONFIRMATORY" in clause["status"]
            assert clause["point_estimate_satisfies"] == (mean > 0.0)
            assert clause["ci95_entirely_satisfies"] == (low > 0.0)
        else:
            assert clause["clause"] == 2 and clause["inequality"] == "rho_sumo(b_mean_k100) < 1"
            assert "not scored" in clause["status"]
            assert "point_estimate_satisfies" not in clause and "ci95_entirely_satisfies" not in clause
    assert sorted((c["clause"], c["definition"]) for c in h3["clauses"]) == [
        (1, "att_env"), (1, "e_sumo"), (2, "att_env"), (2, "e_sumo")
    ]
    assert "void" in h3["clause_3"] and "A19" in h3["clause_3"]

    # ---- (iv) A21(b)'s two items, VERBATIM, first under what_this_does_not_say
    assert artifact["what_this_does_not_say"][:2] == _a21_items()

    # ---- (iv) the paired ATT against both anchors, from OUR per-draw values
    for key in ("e_sumo", "att_env"):
        left = [float(np.mean([expected["arm_values"][key][d][s] for s in tc.TRAINING_SEEDS]))
                for d in tcv.HELD_OUT_DRAWS]
        for index, anchor in enumerate(("fixedtime", "maxpressure")):
            right = [expected["anchors"][key][d][index] for d in tcv.HELD_OUT_DRAWS]
            block = artifact["rho"]["registered_arm"]["paired_att"][key][anchor]
            assert block["n_shared_draws"] == 100
            assert block["left_arm"] == f"{tcv.GRID4X4_SUBJECT}:b_mean_k100" and block["right_arm"] == anchor
            assert block["mean_left"] == float(np.mean(left)) and block["mean_right"] == float(np.mean(right))
            assert block["mean_difference"] == _stats([a - b for a, b in zip(left, right)])[0]

    # ---- (iv) the denominator diagnostic
    for key in ("e_sumo", "att_env"):
        denominators = [expected["anchors"][key][d][0] - expected["anchors"][key][d][1]
                        for d in tcv.HELD_OUT_DRAWS]
        diagnostic = artifact["rho"]["definitions"][key]["denominator_diagnostic"]
        assert diagnostic["n_draws"] == 100
        assert diagnostic["min"] == min(denominators) and diagnostic["max"] == max(denominators)
        assert diagnostic["n_non_positive"] == sum(1 for v in denominators if v <= 0.0)
    tied = [d for d in tcv.HELD_OUT_DRAWS
            if expected["anchors"]["att_env"][d][0] - expected["anchors"]["att_env"][d][1] == 0.0]
    assert ENV_ZERO_DRAW in tied
    assert [row["draw_id"] for row in artifact["rho"]["definitions"]["att_env"]["excluded_draws"]] == tied
    assert len(per_draw["att_env"]) == 100 - len(tied)

    # ---- (iv) per-intersection rho: descriptive, seeds within a draw, zero denominators excluded
    block = artifact["per_intersection_rho"]
    assert "no CI" in block["status"] and "exploratory" in block["status"]
    for ix in IDS:
        per_draw_i = [sum(s.values()) / len(s) for _d, s in sorted(expected["per_ix_rho"][ix].items())]
        entry = block["per_intersection"][ix]
        assert entry["mean_rho"] == sum(per_draw_i) / len(per_draw_i), ix
        assert entry["n_draws_used"] == len(per_draw_i)
        excluded = [d for d in tcv.HELD_OUT_DRAWS if _per_ix_zero(ix, d)]
        assert entry["draw_ids_excluded"] == excluded and entry["n_draws_excluded"] == len(excluded)
        assert not {"ci95", "ci95_low", "ci95_high"} & set(entry)
    assert block["per_intersection"][PER_IX_ZERO_ID]["n_draws_excluded"] == 10

    # ---- (i) the in-support block, from the pinned ranges and the chunks' per-id counts
    support = artifact["in_support"]["per_intersection"]
    calibration = _calibration()
    for j, ix in enumerate(IDS):
        assert support[ix]["support_range"] == SUPPORT[ix]
        assert support[ix]["target_position"] == calibration["per_intersection"][ix]["budgets"]["k100"]["in_support"]["position"]
        assert support[ix]["n_decisions"] == 500 * DECISIONS
        assert support[ix]["decisions_in_support"] == 500 * (DECISIONS - (j % 3))
        assert support[ix]["decisions_below"] == 500 * (j % 3) and support[ix]["decisions_above"] == 0

    # ---- ONE reference value swapped -> report REFUSES, naming the cell, writing nothing (in-process,
    # so the stubs carry THIS repository's HEAD, which J1(c) resolves)
    work = tmp_path_factory.mktemp("swapped") / "work"
    swapped = [dict(p, git_commit=HEAD_SHA) for p in payloads]
    ft = next(p for p in swapped if p["kind"] == "anchor" and p["arm"] == "fixedtime" and p["draw_id"] == 1001)
    mp = next(p for p in swapped if p["kind"] == "anchor" and p["arm"] == "maxpressure" and p["draw_id"] == 1001)
    ft["e_sumo"], mp["e_sumo"] = mp["e_sumo"], ft["e_sumo"]
    _write_stubs(swapped, work)
    _canary_record(work)
    out = work.parent / "out" / "p7_3d_grid4x4.json"
    with pytest.raises(ValueError, match=r"reference cell.*1001.*e_sumo"):
        tcv.report(work_dir=work, out_path=out, output_root=OUTPUT_ROOT, out_root=DRAWS_ROOT,
                   data_dir=DATA, stage=tcv.STAGE_GRID4X4)
    assert not out.parent.exists(), "every refusal precedes every write"
