"""Tests for the three D3 tools: campaign derivation, the bit-identity gate, ATT emission.

All three read corpora, so every fixture here builds a **synthetic** one in ``tmp_path``
rather than touching ``datasets/``. No simulator is required.

The load-bearing tests are the ones that prove a gate can FAIL: a gate that only ever
passes certifies nothing, and these three exist precisely to stop a drifted corpus and an
uncitable number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from offline import att_ladder, compare_corpora, recollect_v11
from offline.recollect_v11 import CampaignError, build_campaign

T = 4
LANES = ("lane_a", "lane_b")


def _episode_arrays(*, seed: int, draw: int, att: float, vehicles: int) -> dict:
    return {
        "format_version": np.asarray("1.1"),
        "ix_ids": np.asarray(["ix0"], dtype=np.str_),
        "lane_ids": np.asarray(list(LANES), dtype=np.str_),
        "metric_keys": np.asarray(["only_metric"], dtype=np.str_),
        "vehicle_count": np.full(T + 1, vehicles, dtype=np.int64),
        "sim_time": np.arange(T + 1, dtype=np.float32),
        "step": np.arange(T + 1, dtype=np.int64),
        "att_per_step": np.linspace(0.0, att, T + 1).astype(np.float32),
        "metrics": np.zeros((T + 1, 1), dtype=np.float32),
        "lane_vehicle_count": np.ones((T + 1, 2), dtype=np.int32),
        "lane_waiting_vehicle_count": np.zeros((T + 1, 2), dtype=np.int32),
        "global_reward": np.full(T, -1.0, dtype=np.float32),
        "episode_length": np.asarray(T, dtype=np.int64),
        "terminated": np.asarray(False, dtype=np.bool_),
        "truncated": np.asarray(True, dtype=np.bool_),
        "engine_seed": np.asarray(seed, dtype=np.int64),
        "flow_draw": np.asarray(draw, dtype=np.int64),
        "ix0_state": np.zeros((T + 1, 3), dtype=np.float32),
        "ix0_avail_mask": np.ones((T + 1, 2), dtype=np.bool_),
        "ix0_current_phase": np.zeros(T + 1, dtype=np.int64),
        "ix0_time_in_phase": np.zeros(T + 1, dtype=np.float32),
        "ix0_action": np.zeros(T, dtype=np.int64),
        "ix0_local_reward": np.full(T, -1.0, dtype=np.float32),
    }


def _write_run(
    root: Path, name: str, *, n: int = 2, att: float = 100.0, vehicles: int = 50,
    version: str = "1.1", policy: str | None = None, drop_att: bool = False,
    draws: range | None = None, meta_overrides: dict | None = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    draw_ids = list(draws) if draws is not None else list(range(1, n + 1))
    episodes = []
    for i, draw in enumerate(draw_ids):
        arrays = _episode_arrays(seed=1000 + i, draw=draw, att=att, vehicles=vehicles)
        arrays["format_version"] = np.asarray(version)
        if drop_att:
            del arrays["att_per_step"]
        filename = f"ep{i:06d}_seed{1000 + i}_draw{draw}.npz"
        with open(run_dir / filename, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        episodes.append({
            "filename": filename, "episode_length": T, "total_global_reward": -float(T),
            "engine_seed": 1000 + i, "flow_draw": draw,
            "episode_sha256": f"hash-{name}-{draw}",
        })
    meta = {
        "backend": "cityflow", "base_seed": 1000, "delta_time": 10, "max_steps": 360,
        "control_mode": "acyclic", "global_reward_fn": "queue_length",
        "local_reward_fn": "queue_length", "global_reward_weight": 0.0,
        "state_features": ["lane_vehicle_count"], "metrics": None,
        "behavior_policy": policy or name.split("__")[1], "checkpoint": None,
        "episodes": 1, "flow_draw_ids": draw_ids,
        "env_paths": {"config": "/tmp/fake.json"},
    }
    meta.update(meta_overrides or {})
    (run_dir / "manifest.json").write_text(
        json.dumps({"format_version": version, "run_metadata": meta,
                    "episodes": episodes}, indent=2),
        encoding="utf-8",
    )
    return run_dir


# ----------------------------------------------------------------------
# 1. Campaign derivation
# ----------------------------------------------------------------------


def test_campaign_derives_commands_and_applies_the_tuned_k(tmp_path: Path) -> None:
    source = tmp_path / "datasets"
    _write_run(source, "cf_hz1x1__fixedtime")
    _write_run(source, "cf_grid4x4__maxpressure")

    specs = {s.name: s for s in build_campaign(source, tmp_path / "datasets_v11")}
    ft = specs["cf_hz1x1__fixedtime"]
    assert ft.fixed_time_k == recollect_v11.TUNED_FIXED_TIME_K["cf_hz1x1"] == 6
    assert "--fixed-time-k" in ft.argv and "6" in ft.argv
    # Only fixedtime carries k; nothing else is perturbed.
    assert specs["cf_grid4x4__maxpressure"].fixed_time_k is None
    assert "--fixed-time-k" not in specs["cf_grid4x4__maxpressure"].argv
    assert str(tmp_path / "datasets_v11" / "cf_hz1x1__fixedtime") in ft.argv


def test_campaign_rejects_a_held_out_draw_id(tmp_path: Path) -> None:
    """D4: draws 1000-1099 must never enter a training corpus."""
    source = tmp_path / "datasets"
    _write_run(source, "cf_hz1x1__random", n=2, draws=range(1000, 1002))

    with pytest.raises(CampaignError, match="held-out evaluation pool"):
        build_campaign(source, tmp_path / "out")


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("global_reward_weight", 1.0),
        ("local_reward_fn", None),
        ("base_seed", 7),
        ("delta_time", 5),
        ("max_steps", 100),
    ],
)
def test_campaign_rejects_each_recorded_invariant(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    """Bounds "inherits anything wrong" to "wrong in a way we never recorded"."""
    source = tmp_path / "datasets"
    _write_run(source, "cf_hz1x1__random", meta_overrides={field: bad_value})

    with pytest.raises(CampaignError, match=field):
        build_campaign(source, tmp_path / "out")


def test_campaign_rejects_a_wrong_metric_width_read_from_the_npz(tmp_path: Path) -> None:
    """The manifest records metrics=null, so this can only be checked in the episodes."""
    source = tmp_path / "datasets"
    run = _write_run(source, "cf_hz1x1__random")
    path = sorted(run.glob("ep*.npz"))[0]
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["metric_keys"] = np.asarray(["a", "b"], dtype=np.str_)
    arrays["metrics"] = np.zeros((T + 1, 2), dtype=np.float32)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)

    with pytest.raises(CampaignError, match="metric keys, expected exactly 1"):
        build_campaign(source, tmp_path / "out")


def test_campaign_refuses_a_populated_target_before_running_anything(
    tmp_path: Path,
) -> None:
    source, target = tmp_path / "datasets", tmp_path / "datasets_v11"
    _write_run(source, "cf_hz1x1__random")
    (target / "cf_hz1x1__random").mkdir(parents=True)
    (target / "cf_hz1x1__random" / "stray.npz").write_bytes(b"x")

    code = recollect_v11.main(
        ["--source-root", str(source), "--target-root", str(target), "--run"]
    )
    assert code == 1


# ----------------------------------------------------------------------
# 2. The bit-identity gate
# ----------------------------------------------------------------------


def test_gate_passes_when_only_att_per_step_was_added(tmp_path: Path) -> None:
    old, new = tmp_path / "v10", tmp_path / "v11"
    _write_run(old, "cf_hz1x1__maxpressure", version="1.0", drop_att=True)
    _write_run(new, "cf_hz1x1__maxpressure")

    assert compare_corpora.main(["--old-root", str(old), "--new-root", str(new)]) == 0


def test_gate_fails_on_a_changed_trajectory_array(tmp_path: Path) -> None:
    """The whole point: a drifted unchanged tier must stop the pipeline."""
    old, new = tmp_path / "v10", tmp_path / "v11"
    _write_run(old, "cf_hz1x1__maxpressure", version="1.0", drop_att=True)
    run = _write_run(new, "cf_hz1x1__maxpressure")
    path = sorted(run.glob("ep*.npz"))[0]
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["ix0_action"] = arrays["ix0_action"] + 1  # one different action
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)

    results = compare_corpora.compare_corpora(old, new)
    assert results[0].differing_arrays == ("ix0_action",)
    assert compare_corpora.main(["--old-root", str(old), "--new-root", str(new)]) == 1


def test_gate_exempts_fixedtime_but_requires_it_to_actually_differ(
    tmp_path: Path,
) -> None:
    """A fixedtime tier identical to the untuned corpus means k never took effect."""
    old, new = tmp_path / "v10", tmp_path / "v11"
    _write_run(old, "cf_hz1x1__fixedtime", version="1.0", drop_att=True)
    _write_run(new, "cf_hz1x1__fixedtime")  # identical -> tuning did not happen

    assert compare_corpora.main(["--old-root", str(old), "--new-root", str(new)]) == 1

    # With a genuine difference, the exemption applies and the gate passes.
    run = sorted((new / "cf_hz1x1__fixedtime").glob("ep*.npz"))[0]
    with np.load(run) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["ix0_action"] = arrays["ix0_action"] + 1
    with open(run, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    assert compare_corpora.main(["--old-root", str(old), "--new-root", str(new)]) == 0


def test_gate_rejects_an_unexpected_extra_array(tmp_path: Path) -> None:
    """"v1.1 is v1.0 plus one field" is checked, not promised."""
    old, new = tmp_path / "v10", tmp_path / "v11"
    _write_run(old, "cf_hz1x1__maxpressure", version="1.0", drop_att=True)
    run = _write_run(new, "cf_hz1x1__maxpressure")
    path = sorted(run.glob("ep*.npz"))[0]
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["surprise"] = np.zeros(3, dtype=np.float32)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)

    result = compare_corpora.compare_run(old / "cf_hz1x1__maxpressure",
                                         new / "cf_hz1x1__maxpressure")
    assert result.error is not None and "key sets differ" in result.error


# ----------------------------------------------------------------------
# 3. The ATT emitter
# ----------------------------------------------------------------------


def test_att_horizon_is_the_last_row_not_the_mean(tmp_path: Path) -> None:
    """A1: the reported quantity is the horizon value; the mean is att_running_mean."""
    root = tmp_path / "v11"
    _write_run(root, "cf_hz1x1__maxpressure", n=3, att=200.0)

    cell = att_ladder.tier_cells(root)[0]
    assert cell.att_horizon_mean == pytest.approx(200.0)
    # The running mean of linspace(0, 200, 5) is 100.0 -- half the horizon value. If the
    # emitter ever reports that, this assertion is what catches it.
    assert cell.att_horizon_mean != pytest.approx(100.0)


def test_every_cell_carries_the_co_report_unconditionally(tmp_path: Path) -> None:
    """A5 point 1: no threshold at which horizon vehicle_count is omitted."""
    root = tmp_path / "v11"
    _write_run(root, "cf_hz1x1__maxpressure", vehicles=77)
    cell = att_ladder.tier_cells(root)[0]
    assert cell.vehicle_count_mean == pytest.approx(77.0)


def test_no_validity_threshold_survives_anywhere(tmp_path: Path) -> None:
    """A5 withdrew A4's >5% condition; nothing may re-introduce it by accident.

    Asserted against the module's own surface rather than its docstring: a leftover
    threshold constant or verdict property is what a partial revert would leave behind.
    """
    assert not hasattr(att_ladder, "VALIDITY_THRESHOLD")
    assert not hasattr(att_ladder, "screen_comparisons")
    assert not hasattr(att_ladder, "Comparison")
    source = Path(att_ladder.__file__).read_text(encoding="utf-8")
    assert "0.05" not in source


def test_seed_split_runs_of_one_tier_are_pooled(tmp_path: Path) -> None:
    root = tmp_path / "v11"
    _write_run(root, "cf_hz1x1__mappo1000__seed101", n=2, draws=range(1, 3))
    _write_run(root, "cf_hz1x1__mappo1000__seed202", n=2, draws=range(3, 5))

    cells = {c.tier: c for c in att_ladder.tier_cells(root)}
    assert cells["mappo1000"].n_episodes == 4


def test_a_huge_vehicle_count_gap_is_reported_not_invalidated(tmp_path: Path) -> None:
    """The A5 correction, as a test.

    Under A4 this pair was INVALID; A5 withdrew that because horizon vehicle_count is a
    control outcome, so a large gap *is* the result rather than a defect in the
    comparison. cf_grid4x4 MaxPressure vs MAPPO differs by 98.1% on real anchors.
    """
    root = tmp_path / "v11"
    _write_run(root, "cf_hz1x1__maxpressure", att=100.0, vehicles=10)
    _write_run(root, "cf_hz1x1__random", att=300.0, vehicles=1000)  # 100x apart

    cells = {c.tier: c for c in att_ladder.tier_cells(root)}
    assert cells["maxpressure"].vehicle_count_mean == pytest.approx(10.0)
    assert cells["random"].vehicle_count_mean == pytest.approx(1000.0)
    # Same draws, so the pair is comparable however large the outcome gap is.
    overlap = att_ladder.draw_overlaps(list(cells.values()))[0]
    assert overlap.void is False
    assert overlap.identical is True


def test_every_cell_reports_its_draw_ids(tmp_path: Path) -> None:
    """A5 point 3: the draw ids must be reported alongside the cell."""
    root = tmp_path / "v11"
    _write_run(root, "cf_hz1x1__maxpressure", n=3, draws=range(5, 8))

    cell = att_ladder.tier_cells(root)[0]
    assert cell.draw_ids == (5, 6, 7)
    assert "draws=3 [5-7]" in cell.line()


def test_pooled_seed_splits_report_the_union_of_their_draws(tmp_path: Path) -> None:
    root = tmp_path / "v11"
    _write_run(root, "cf_hz1x1__mappo1000__seed101", n=2, draws=range(1, 3))
    _write_run(root, "cf_hz1x1__mappo1000__seed202", n=2, draws=range(3, 5))

    cell = {c.tier: c for c in att_ladder.tier_cells(root)}["mappo1000"]
    assert cell.draw_ids == (1, 2, 3, 4)


def test_tiers_with_no_shared_draws_are_void(tmp_path: Path) -> None:
    """A5's replacement voiding rule: binary and checkable, not thresholded."""
    root = tmp_path / "v11"
    _write_run(root, "cf_hz1x1__maxpressure", n=2, draws=range(1, 3))
    _write_run(root, "cf_hz1x1__random", n=2, draws=range(90, 92))

    overlap = att_ladder.draw_overlaps(att_ladder.tier_cells(root))[0]
    assert overlap.void is True
    assert overlap.n_shared == 0


def test_partial_draw_overlap_is_surfaced_not_pooled(tmp_path: Path) -> None:
    """A partial overlap must be visible; the tool must not average over it silently."""
    root = tmp_path / "v11"
    _write_run(root, "cf_hz1x1__maxpressure", n=3, draws=range(1, 4))
    _write_run(root, "cf_hz1x1__random", n=3, draws=range(3, 6))

    overlap = att_ladder.draw_overlaps(att_ladder.tier_cells(root))[0]
    assert overlap.void is False
    assert overlap.identical is False
    assert overlap.n_shared == 1


def test_emitter_refuses_a_v10_corpus(tmp_path: Path) -> None:
    """att_horizon is not recoverable from v1.0; refusing beats inventing."""
    root = tmp_path / "v10"
    _write_run(root, "cf_hz1x1__maxpressure", version="1.0", drop_att=True)

    with pytest.raises(ValueError, match="has no att_per_step"):
        att_ladder.tier_cells(root)


def test_report_header_states_its_scope(tmp_path: Path, capsys) -> None:
    """A table lifted out of a terminal must carry its own scope caveat."""
    root = tmp_path / "v11"
    _write_run(root, "cf_hz1x1__maxpressure")
    assert att_ladder.main(["--root", str(root)]) == 0

    out = capsys.readouterr().out
    assert "RANDOMISED DRAWS 1-200 ONLY" in out
    assert "does NOT settle the nominal draw-0 comparison" in out
    assert "att_running_mean" in out
    # And the reminder repeats at the end, where a reader stops.
    assert out.rstrip().endswith("remain open.")
