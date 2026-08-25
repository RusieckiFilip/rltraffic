"""⭐ THE LOAD-BEARING TEST: with ``rtg_mode`` present and defaulted, the default path did not move.

``BRIEF_28`` section 6.1.  This is the one test that protects five merged tasks (P4.2, P4.3, P4.6,
P4.7 and P5.2's reused column) from the only edit this project has ever made to ``agent/DTAgent.py``.
It re-rolls two committed ``dt`` cells end to end and asserts **bit-exact** per-episode ATT against
the value in ``docs/data/p4_6_grid.json`` -- ``==``, never ``np.allclose``, because a tolerance here
would accept precisely the drift the test exists to detect.

Two cells, and the second one is the point (``BRIEF_28`` A4)
------------------------------------------------------------
* **Cell 1** ``mappo1000`` seed 101, the headline tier: A6's delta and P4.3's whole 13,000-unit sweep
  were measured on it.
* **Cell 2** ``random`` seed 101, the tier with the largest committed ``RtgSummary.std`` raw AND
  scaled among those with checkpoints under ``output/p4_6/checkpoints/``.  **The identity test is
  strongest where the RTG carries the most variance, because that is where forcing
  ``rtg_mode="zero"`` does the most damage** -- and mutation 1 of section 6.1 has to actually fail.
  Had this landed on ``fixedtime`` instead (delta ``-5.68e-16``; P5.2's reviewer measured the grid4x4
  DT reproducing the fixed-time controller with 0 of 5760 actions differing) the mutation could have
  survived and the test would have certified nothing.

⚠️ **What this test cannot see** is written out in ``docs/plans/p5.3a.md`` section 3, seven items.
The two that matter most: it exercises only the legacy-payload branch of ``rtg_mode``, and it is
blind by construction to any logit change that never crosses a decision boundary.

Identity of the weights is checked FIRST, by two different routes, because they protect different
things (``BRIEF_28`` B3):

* cell 1 through ``assert_reused_checkpoint_identity`` -- a **file sha256** against
  ``p4_gate.json``.  ``p4_training.json`` never carried a canonical digest for these five
  (``offline/method_tier_grid.py:1233-1235``).
* cell 2 through ``canonical_digest_of`` against ``p4_6_training.json``'s committed digest **and**
  through ``output/SHA256SUMS_p4_6.txt`` at consumption.

🚨 ``output/p4_dt/`` appears in **no** integrity manifest (``DEFERRED`` 56), so cell 1 rests on a
filename-dependent file hash with nothing behind it.  The two cells are **not** equally protected and
the Return Packet says so.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pytest

from offline.dt_gate import HELD_OUT_DRAWS, EpisodeResult, evaluate_arm
from offline.materialise_draws import (
    CITYFLOW_CONFIG_FILENAME,
    FLOW_FILENAME,
    PROVENANCE_FILENAME,
    draw_config_path,
    draw_dir,
)
from offline.method_tier_grid import (
    DECLARED_GRADIENT_STEPS,
    TIERS,
    arm_key,
    assert_reused_checkpoint_identity,
    canonical_digest_of,
    env_settings_for_tiers,
    file_sha256,
    tier_spec,
)
from offline.rtg_calibration import agent_with_target

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs" / "data"
OUTPUT = REPO / "output"
DRAWS_ROOT = REPO / "scenarios" / "draws"
SCENARIO_KEY = "cityflow1x1"
SCENARIO_ID = "cityflow1x1"
ENGINE_SEED = 1000

#: The five draws that survived the retirement of the p46/p47 worktrees; the generator's output for
#: these is what licenses trusting it for the 95 that had to be rebuilt.
SURVIVING_DRAWS = (1000, 1001, 1002, 1003, 1004)

CELL_ONE = ("mappo1000", 101, "p4_dt/dt_seed101.pt")
CELL_TWO = ("random", 101, "p4_6/checkpoints/random_dt_seed101.pt")


# ----------------------------------------------------------------------
# Gating: these run on real artifacts or they skip with a reason naming them
# ----------------------------------------------------------------------


def _corpus_root() -> Path:
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO / "datasets_v11"
    if not candidate.is_dir():
        pytest.skip(
            f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to a "
            "collected corpus to run the committed-cell re-roll"
        )
    return candidate


def _checkpoint(relative: str) -> Path:
    path = OUTPUT / relative
    if not path.is_file():
        pytest.skip(f"checkpoint not present in this tree: {path}")
    return path


def _require_every_held_out_draw() -> None:
    missing = [
        draw
        for draw in HELD_OUT_DRAWS
        if not draw_config_path(SCENARIO_KEY, int(draw), out_root=DRAWS_ROOT).is_file()
    ]
    if missing:
        pytest.skip(
            f"{len(missing)} of {len(HELD_OUT_DRAWS)} held-out draws are not materialised "
            f"(first {missing[:3]}); run `python -m offline.materialise_draws` for the "
            "held-out pool before the committed column can be re-rolled"
        )


def _committed_episodes(arm: str, seed: int) -> list[dict[str, Any]]:
    grid = json.loads((DATA / "p4_6_grid.json").read_text(encoding="utf-8"))
    records = [
        entry
        for entry in grid["episodes"]
        if entry["arm"] == arm and int(entry["seed"]) == seed
    ]
    assert len(records) == len(HELD_OUT_DRAWS), (
        f"{arm} seed {seed}: the committed grid holds {len(records)} episodes, not "
        f"{len(HELD_OUT_DRAWS)}"
    )
    return records


def _dt_choose_action_factory(
    checkpoint: Path, target_rtg: float
) -> Callable[[Any], Callable[[Any, dict[str, Any]], np.ndarray]]:
    """P4.6's own evaluation path: load, THEN apply the target, then act greedily.

    ``agent_with_target`` is reused rather than hand-rolled precisely because ``DTAgent.load``
    overwrites ``_target_rtg`` from the payload, so a target passed to the constructor is silently
    discarded (``offline/rtg_calibration.py:605-616``).
    """

    def factory(env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
        agent = agent_with_target(
            env,
            checkpoint,
            declared_gradient_steps=DECLARED_GRADIENT_STEPS,
            target_rtg=float(target_rtg),
            device=None,
        )
        return lambda _env, info: agent.act(info, explore=False, update_memory=True)

    return factory


def _reroll(tier: str, seed: int, checkpoint: Path) -> list[EpisodeResult]:
    spec = tier_spec(tier)
    root = _corpus_root()
    return evaluate_arm(
        arm=arm_key("dt", tier),
        seed=int(seed),
        draw_ids=list(HELD_OUT_DRAWS),
        config_for_draw=lambda draw: draw_config_path(
            SCENARIO_KEY, int(draw), out_root=DRAWS_ROOT
        ),
        env_settings=env_settings_for_tiers([spec], root),
        scenario_id=SCENARIO_ID,
        choose_action_factory=_dt_choose_action_factory(checkpoint, spec.target_rtg),
        engine_seed=ENGINE_SEED,
    )


def _assert_bit_exact(produced: Sequence[EpisodeResult], committed: Sequence[dict[str, Any]]) -> int:
    """Exact float equality on every episode.  A tolerance would accept the drift we are hunting."""
    by_draw = {int(result.draw_id): result for result in produced}
    assert len(by_draw) == len(committed)
    compared = 0
    for record in committed:
        draw = int(record["draw_id"])
        result = by_draw[draw]
        assert float(result.att_horizon) == float(record["att_horizon"]), (
            f"draw {draw}: att_horizon {result.att_horizon!r} against committed "
            f"{record['att_horizon']!r}"
        )
        assert float(result.horizon_vehicle_count) == float(record["horizon_vehicle_count"]), (
            f"draw {draw}: horizon_vehicle_count {result.horizon_vehicle_count!r} against "
            f"committed {record['horizon_vehicle_count']!r}"
        )
        compared += 1
    return compared


# ----------------------------------------------------------------------
# Gate -1: the regenerated draws are the draws P4.6 evaluated on
# ----------------------------------------------------------------------


def test_the_surviving_draws_regenerate_byte_identically(tmp_path: Path) -> None:
    """If the generator reproduces the five survivors, the 95 it rebuilt are the originals too.

    ⚠️ ``cityflow.json`` carries ``flowFile`` as a path **relative to the draw directory's root**,
    so it is root-dependent by design and is compared field by field with that one key excluded.
    ``flow.json`` -- the demand itself, and the only thing the simulation reads -- is compared as
    bytes, and its sha256 is compared through the provenance record as a second route.
    """
    from offline.materialise_draws import materialise

    for draw in SURVIVING_DRAWS:
        if not draw_config_path(SCENARIO_KEY, draw, out_root=DRAWS_ROOT).is_file():
            pytest.skip(f"surviving draw {draw} is absent from {DRAWS_ROOT}")

    scratch = tmp_path / "draws_control"
    materialise(
        source_config=str(REPO / "configs" / "sim" / "cityflow1x1.json"),
        draw_ids=list(SURVIVING_DRAWS),
        out_root=scratch,
    )

    for draw in SURVIVING_DRAWS:
        original = draw_dir(SCENARIO_KEY, draw, out_root=DRAWS_ROOT)
        rebuilt = draw_dir(SCENARIO_KEY, draw, out_root=scratch)

        assert (rebuilt / FLOW_FILENAME).read_bytes() == (original / FLOW_FILENAME).read_bytes(), (
            f"draw {draw}: the regenerated demand differs from the one P4.6 evaluated on"
        )

        left = json.loads((original / CITYFLOW_CONFIG_FILENAME).read_text(encoding="utf-8"))
        right = json.loads((rebuilt / CITYFLOW_CONFIG_FILENAME).read_text(encoding="utf-8"))
        assert left.pop("flowFile") != right.pop("flowFile"), "the roots must actually differ"
        assert left == right

        left_prov = json.loads((original / PROVENANCE_FILENAME).read_text(encoding="utf-8"))
        right_prov = json.loads((rebuilt / PROVENANCE_FILENAME).read_text(encoding="utf-8"))
        assert right_prov["files"][FLOW_FILENAME] == left_prov["files"][FLOW_FILENAME]
        assert right_prov["draw"] == left_prov["draw"]
        assert right_prov["randomizer"] == left_prov["randomizer"]
        assert right_prov["pool"] == left_prov["pool"] == "held_out"


# ----------------------------------------------------------------------
# Gate 1: identity of the weights, before either re-roll
# ----------------------------------------------------------------------


def test_cell_one_checkpoint_is_the_one_the_p4_gate_records() -> None:
    """Route (a): file sha256 through the P4 gate artifact.  There is no canonical digest to check."""
    _checkpoint(CELL_ONE[2])
    for method in ("bc", "bc_top10", "iql"):
        if not (OUTPUT / "p4_4" / "checkpoints" / f"{method}_seed101.pt").is_file():
            pytest.skip("P4.4 baseline checkpoints are not present in this tree")

    training = json.loads((DATA / "p4_4_training.json").read_text(encoding="utf-8"))
    gate = json.loads((DATA / "p4_gate.json").read_text(encoding="utf-8"))
    record = assert_reused_checkpoint_identity(
        training,
        gate,
        baselines_root=OUTPUT / "p4_4" / "checkpoints",
        dt_root=OUTPUT / "p4_dt",
    )
    assert record["verified"] == 20
    assert len(record["dt"]) == 5
    seed_101 = [entry for entry in record["dt"] if entry["seed"] == 101]
    assert len(seed_101) == 1
    assert seed_101[0]["file_sha256"] == gate["checkpoints"]["101"]["sha256"]


def test_cell_two_checkpoint_matches_both_its_committed_digest_and_the_manifest() -> None:
    """Route (b): canonical digest AND the integrity manifest, re-checked AT CONSUMPTION."""
    path = _checkpoint(CELL_TWO[2])
    training = json.loads((DATA / "p4_6_training.json").read_text(encoding="utf-8"))
    runs = [
        run
        for run in training["runs"]
        if run["tier"] == CELL_TWO[0] and run["method"] == "dt" and int(run["seed"]) == CELL_TWO[1]
    ]
    assert len(runs) == 1
    assert canonical_digest_of(path) == runs[0]["canonical_digest"]

    manifest = OUTPUT / "SHA256SUMS_p4_6.txt"
    assert manifest.is_file(), f"integrity manifest missing: {manifest}"
    entries = dict(
        (parts[1], parts[0])
        for parts in (line.split() for line in manifest.read_text(encoding="utf-8").splitlines())
        if len(parts) == 2
    )
    relative = CELL_TWO[2]
    assert relative in entries, f"{relative} is not covered by {manifest.name}"
    assert file_sha256(path) == entries[relative]


def test_the_p4_dt_checkpoints_are_covered_by_no_integrity_manifest() -> None:
    """``DEFERRED`` 56, asserted so the gap cannot close silently and go unnoticed.

    This is a **statement of a known deficiency**, not a guard.  If a future task writes a manifest
    covering ``output/p4_dt/``, this test fails and should be deleted along with the note in the
    packet -- that is the intended way for it to end.
    """
    manifests = sorted(OUTPUT.glob("SHA256SUMS_*.txt"))
    if not manifests:
        pytest.skip("no integrity manifests in this tree")
    covered = [
        line
        for manifest in manifests
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if "p4_dt/" in line
    ]
    assert covered == [], (
        "output/p4_dt/ is now covered by an integrity manifest; DEFERRED 56 is closed and this "
        f"test should be deleted (found {len(covered)} entries)"
    )


# ----------------------------------------------------------------------
# Section 6.1 proper
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("tier", "seed", "relative"), [CELL_ONE, CELL_TWO])
def test_a_committed_dt_cell_reproduces_bit_exactly_through_the_changed_code(
    tier: str, seed: int, relative: str
) -> None:
    """Every per-episode ATT equals the committed value exactly, on all 100 held-out draws."""
    checkpoint = _checkpoint(relative)
    _corpus_root()
    _require_every_held_out_draw()

    committed = _committed_episodes(arm_key("dt", tier), seed)
    produced = _reroll(tier, seed, checkpoint)

    assert len(produced) == len(HELD_OUT_DRAWS)
    compared = _assert_bit_exact(produced, committed)
    assert compared == len(HELD_OUT_DRAWS) == 100


def test_the_two_identity_cells_are_the_registered_ones() -> None:
    """R8, restated where a test can see it: a quiet change of cell would change what is protected."""
    assert CELL_ONE[0] == "mappo1000" and CELL_ONE[1] == 101
    assert CELL_TWO[0] == "random" and CELL_TWO[1] == 101
    assert TIERS[CELL_ONE[0]].target_rtg == -5762.0
    assert TIERS[CELL_TWO[0]].target_rtg == -38369.0
    assert DECLARED_GRADIENT_STEPS == 40000
    assert len(HELD_OUT_DRAWS) == 100
