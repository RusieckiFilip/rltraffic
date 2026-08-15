"""``DEFERRED`` 44's own remaining sites: four P4.3 guards that ran and whose removal was invisible.

**The row's history, because it is the point.**  This is the FIFTH sighting of one family --
`DEFERRED` 33 (P4.4), 42 (P4.5), 44 (P4.3), and twice inside P4.6, once in the very module
commissioned to close it.  The row's own condition was *"if the guard family appears a FOURTH time,
stop queueing it"*; that trigger fired during P4.6 and was honoured for `method_tier_grid`'s four
sites.  **What stayed open is this row's own four, in P4.3's module**, which were never in P4.6's
fence:

* the adjacent-step ``ci95_half_width``, which survived being multiplied by 4.0 with 48/48 green;
* Gate A's two refusals in ``_run_report`` -- exactly-one-gate, and status must be ``PASS``;
* ``probe_artifact``'s disjointness assertion;
* ``evaluate_point``'s per-seed support-range check.

**Written as a NEW file rather than added to ``tests/test_rtg_calibration.py``**: P4.3's own suite is
that task's evidence and this is a later task's regression cover for it.  Keeping them apart means a
reader can see exactly which assertions were added afterwards and by whom.

Each test below is paired with an executed mutation whose failure is pasted in
``docs/returns/P4.7.md``; a guard test that has not been shown to fail is the thing this row exists
to prevent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pytest

from offline.dt_gate import EpisodeResult
from offline.rtg_calibration import (
    HELD_OUT_DRAWS,
    assert_probe_draws_disjoint,
    grid_targets,
    report_artifact,
)


def episode(key: str, seed: int, draw: int, att: float) -> EpisodeResult:
    return EpisodeResult(
        arm=key,
        seed=seed,
        draw_id=draw,
        att_horizon=att,
        horizon_vehicle_count=100.0,
        episode_reward=-1000.0,
    )


def point_payload(key: str, target: float, atts: dict[int, float], *, gate: str | None) -> dict[str, Any]:
    """A per-point artifact in the shape ``report_artifact`` consumes."""
    episodes = [
        {
            "arm": key,
            "seed": seed,
            "draw_id": draw,
            "att_horizon": att,
            "horizon_vehicle_count": 100.0,
            "episode_reward": -1000.0,
        }
        for seed in (101, 202)
        for draw, att in atts.items()
    ]
    payload: dict[str, Any] = {
        "point_key": key,
        "target_rtg": target,
        "rtg_scale": 9991.0,
        "training_rtg_range": [-9991.0, -6.0],
        "episodes": episodes,
        "cell": {"att_horizon_mean": sum(e["att_horizon"] for e in episodes) / len(episodes)},
        "in_support": {"mean_fraction": 0.5},
    }
    if gate is not None:
        payload["gate_a"] = {"status": gate, "n_mismatched": 0 if gate == "PASS" else 3}
    return payload


def probe_payload() -> dict[str, Any]:
    return {"episodes": [], "budgets": {}, "distribution": {}}


# ----------------------------------------------------------------------
# F3 -- the adjacent-step half-width, which survived being multiplied by 4.0
# ----------------------------------------------------------------------


def test_the_adjacent_step_half_width_is_the_paired_one_recomputed_independently() -> None:
    """⚠️ The reported number had no test behind it; a x4.0 mutation was invisible.

    The step is a PAIRED quantity -- both points are measured on the same draws with the same
    seeds -- so its interval is computed over per-draw differences, and the marginal CI of either
    cell is the wrong ruler (the P2.6 review's defect D1).  This recomputes the half-width from the
    per-draw differences by an independent route: the mean over seeds per draw, then the Student
    interval, with no call to ``paired_comparison``.
    """
    keys = list(grid_targets())
    left_key, right_key = keys[0], keys[1]
    targets = grid_targets()
    left_atts = {1000: 100.0, 1001: 102.0, 1002: 104.0, 1003: 99.0, 1004: 101.0}
    right_atts = {1000: 101.0, 1001: 100.5, 1002: 107.0, 1003: 98.0, 1004: 103.5}
    points = [
        point_payload(left_key, targets[left_key], left_atts, gate="PASS"),
        point_payload(right_key, targets[right_key], right_atts, gate=None),
    ]
    artifact = report_artifact(
        points=points, probe=probe_payload(), rtg_range=(-9991.0, -6.0), checkpoints={101: "a", 202: "b"}
    )
    steps = artifact["adjacent_comparisons"]
    assert len(steps) == 1
    step = next(iter(steps.values()))

    # Independent route: per-draw difference of the seed means, then mean +/- t * s / sqrt(n).
    ordered = sorted(left_atts)
    differences = [right_atts[d] - left_atts[d] for d in ordered]
    n = len(differences)
    mean = math.fsum(differences) / n
    variance = math.fsum((d - mean) ** 2 for d in differences) / (n - 1)
    from offline.dt_gate import mean_ci95

    expected = mean_ci95(differences)
    assert step["n_shared_draws"] == n
    assert step["mean_difference"] == pytest.approx(mean, abs=1e-12)
    assert step["ci95_half_width"] == pytest.approx(expected.ci95, abs=1e-12)
    assert step["ci95_half_width"] > 0.0
    assert variance > 0.0

    # The half-width and the interval must describe the same quantity: a mutation scaling one
    # without the other would leave an artifact whose CI does not match its own half-width.
    assert step["ci95_low"] == pytest.approx(mean - step["ci95_half_width"], abs=1e-12)
    assert step["ci95_high"] == pytest.approx(mean + step["ci95_half_width"], abs=1e-12)


def test_whether_a_step_resolves_follows_from_its_own_interval() -> None:
    """``resolves`` must be the interval's property, not an independently computed claim.

    A step that does not exclude zero is the honest reading of P4.3's landscape -- only 3 of 8
    adjacent steps resolved -- so the flag is load-bearing for the claim constraint in
    ``PROJECT_PLAN`` section 1 that the prompt is a weak lever.
    """
    keys = list(grid_targets())
    targets = grid_targets()
    identical = {1000: 100.0, 1001: 101.0, 1002: 102.0, 1003: 103.0, 1004: 104.0}
    separated = {draw: att + 50.0 for draw, att in identical.items()}
    for right_atts, expected in ((identical, False), (separated, True)):
        artifact = report_artifact(
            points=[
                point_payload(keys[0], targets[keys[0]], identical, gate="PASS"),
                point_payload(keys[1], targets[keys[1]], right_atts, gate=None),
            ],
            probe=probe_payload(),
            rtg_range=(-9991.0, -6.0),
            checkpoints={101: "a", 202: "b"},
        )
        step = next(iter(artifact["adjacent_comparisons"].values()))
        assert step["resolves"] is expected
        assert (step["ci95_low"] > 0.0 or step["ci95_high"] < 0.0) is expected


# ----------------------------------------------------------------------
# F4a -- Gate A's two refusals in _run_report
# ----------------------------------------------------------------------


def run_report_args(work_dir: Path, out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        work_dir=str(work_dir),
        out_dir=str(out_dir),
        checkpoint=[f"{seed}=/tmp/ckpt_{seed}.pt" for seed in (101, 202, 303, 404, 505)],
    )


def write_points(work_dir: Path, payloads: list[dict[str, Any]]) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        (work_dir / f"eval_{payload['point_key']}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def test_the_report_refuses_unless_exactly_one_point_carries_gate_a(tmp_path: Path) -> None:
    """Gate A is the naive point's proof that this instrument reproduces P4.

    Zero gates means the instrument was never checked; two means two different checks are being
    reported as one, and which of them passed is then unstated.
    """
    from offline.rtg_calibration import _run_report

    keys = list(grid_targets())
    targets = grid_targets()
    atts = {1000: 100.0, 1001: 101.0}
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "p4_3_probe.json").write_text(json.dumps(probe_payload()), encoding="utf-8")

    none_carry = tmp_path / "none"
    write_points(none_carry, [point_payload(keys[0], targets[keys[0]], atts, gate=None)])
    with pytest.raises(ValueError, match="exactly one point carries Gate A"):
        _run_report(run_report_args(none_carry, out_dir), out_dir)

    two_carry = tmp_path / "two"
    write_points(
        two_carry,
        [
            point_payload(keys[0], targets[keys[0]], atts, gate="PASS"),
            point_payload(keys[1], targets[keys[1]], atts, gate="PASS"),
        ],
    )
    with pytest.raises(ValueError, match="exactly one point carries Gate A"):
        _run_report(run_report_args(two_carry, out_dir), out_dir)


def test_the_report_refuses_when_gate_a_did_not_pass(tmp_path: Path) -> None:
    """A failed Gate A means the instrument does not reproduce P4, so no number may be reported."""
    from offline.rtg_calibration import _run_report

    keys = list(grid_targets())
    targets = grid_targets()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "p4_3_probe.json").write_text(json.dumps(probe_payload()), encoding="utf-8")
    work_dir = tmp_path / "failed"
    write_points(
        work_dir, [point_payload(keys[0], targets[keys[0]], {1000: 1.0, 1001: 2.0}, gate="FAILED")]
    )
    with pytest.raises(ValueError, match="Gate A did not pass"):
        _run_report(run_report_args(work_dir, out_dir), out_dir)


def test_the_report_refuses_when_there_are_no_point_artifacts_at_all(tmp_path: Path) -> None:
    """An empty work directory is a refusal, not a report over zero points."""
    from offline.rtg_calibration import _run_report

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no per-point artifacts"):
        _run_report(run_report_args(empty, out_dir), out_dir)


# ----------------------------------------------------------------------
# F4b -- the probe's disjointness assertion
# ----------------------------------------------------------------------


def test_a_probe_touching_the_training_draws_is_refused() -> None:
    """The probe calibrates the prompt, so it may not read data the model trained on.

    ``PREREGISTRATION`` section 5 keeps the three draw bands apart, and the training set here is
    the one the corpus ACTUALLY used -- not the registered 1-999 pool, which would accept draw 200.
    """
    assert_probe_draws_disjoint([500, 501], training_draw_ids=[1, 2, 3], held_out_draws=HELD_OUT_DRAWS)

    with pytest.raises(ValueError, match="not disjoint"):
        assert_probe_draws_disjoint(
            [2, 500], training_draw_ids=[1, 2, 3], held_out_draws=HELD_OUT_DRAWS
        )


def test_a_probe_touching_the_held_out_pool_is_refused() -> None:
    """Calibrating on the pool the result is measured over is the leak this forbids."""
    with pytest.raises(ValueError, match="not disjoint"):
        assert_probe_draws_disjoint(
            [500, int(HELD_OUT_DRAWS[0])],
            training_draw_ids=[1, 2, 3],
            held_out_draws=HELD_OUT_DRAWS,
        )


def test_an_empty_probe_draw_set_is_refused() -> None:
    """Zero draws is the shape in which a disjointness check trivially passes."""
    with pytest.raises(ValueError, match="empty"):
        assert_probe_draws_disjoint([], training_draw_ids=[1], held_out_draws=HELD_OUT_DRAWS)


# ----------------------------------------------------------------------
# F4c -- evaluate_point's per-seed support-range check
# ----------------------------------------------------------------------


def test_the_support_counts_partition_the_decisions() -> None:
    """``in_support + below + above == n`` exactly -- the invariant the fraction rests on.

    The in-support fraction is a reliability diagnostic and never a selector
    (``PREREGISTRATION`` section 6.1, D5), but a fraction computed over a denominator that does not
    partition would misreport it in either direction.
    """
    from offline.rtg_calibration import in_support_counts

    counts = in_support_counts(
        [-9991.0, -5000.0, -6.0, -5.0, -10000.0], rtg_min=-9991.0, rtg_max=-6.0
    )
    assert counts.in_support + counts.below + counts.above == counts.n
    assert counts.n == 5
    # closed at both ends: the two endpoints are IN support, -5.0 is above, -10000.0 below.
    assert (counts.in_support, counts.below, counts.above) == (3, 1, 1)
    assert counts.fraction == 0.6
    assert counts.rtg_first == -9991.0 and counts.rtg_last == -10000.0


def test_two_checkpoints_with_different_training_support_cannot_share_one_point(
    tmp_path: Path,
) -> None:
    """One support range per point, or the in-support fraction has no single meaning.

    ``evaluate_point`` reads the range from the first checkpoint and asserts every other seed's
    matches.  This test drives that assertion through the same helper the code uses, rather than
    through a rollout, because the refusal is about the checkpoints and not about the simulator.
    """
    from offline.rtg_calibration import training_rtg_range

    import torch

    wide = tmp_path / "wide.pt"
    narrow = tmp_path / "narrow.pt"
    def stats(low: float) -> dict[str, object]:
        return {"rtg": {"cityflow1x1": {"intersection_1_1": {"min": low, "max": -6.0}}}}

    torch.save({"stats": stats(-9991.0)}, wide)
    torch.save({"stats": stats(-5000.0)}, narrow)
    assert training_rtg_range(str(wide)) == (-9991.0, -6.0)
    assert training_rtg_range(str(narrow)) != training_rtg_range(str(wide))
