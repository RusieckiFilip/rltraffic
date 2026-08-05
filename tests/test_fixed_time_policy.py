"""Engine-free tests for the fixed-time controller (P2.5, ladder Tier 1).

These exercise the schedule algebra, the plan parser, the acyclic-only guard and
the availability-mask fallback without a simulator, so they run everywhere. The
live-env mapping proof and the replay/env validation gate live in
``test_fixed_time_env_mapping.py`` (CityFlow-gated).

Format / convention under test
------------------------------
* action index ``a`` selects the ``a``-th *green* file phase, greens sorted
  ascending -- this mirrors ``envs.phase_control.AcyclicPhases.target_phase`` and
  is verified against a live engine in the companion file.
* the schedule is a pure function of ``info["step"]``:
  ``action(t) = cycle[(t // k) % len(cycle)]``.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pytest

from utils.common_utils import IntersectionInfo

from offline.policies.fixed_time import (
    FixedTimeController,
    PlanSchedule,
    equal_split_cycle,
    green_action_phases,
    parse_signal_plan,
)

IX_ID = "intersection_1_1"


# ----------------------------------------------------------------------
# Stubs: a hangzhou-shaped intersection (9 file phases, 0 = clearance)
# ----------------------------------------------------------------------


def _hz_like_ix(ix_id: str = IX_ID) -> IntersectionInfo:
    """9 phases: phase 0 is a 5 s clearance (no roadlinks); 1..8 are 30 s greens."""
    return IntersectionInfo(
        id=ix_id,
        incoming_lanes=[],
        outgoing_lanes=[],
        num_phases=9,
        phase_roadlink_mapping=[[]] + [[2 * p, 2 * p + 1] for p in range(8)],
        phase_durations=[5.0] + [30.0] * 8,
        phase_states=[],
        roadlink_lanes=[],
    )


class _StubEnv:
    def __init__(self, intersections: list[IntersectionInfo], control_mode: str = "acyclic") -> None:
        self.intersections = intersections
        self.control_mode = control_mode


def _info(step: int, avail: Iterable[int], current_phase: int, ix_id: str = IX_ID) -> dict[str, Any]:
    return {
        "step": int(step),
        "intersections": {
            ix_id: {"avail_actions": [int(a) for a in avail], "current_phase": int(current_phase)}
        },
    }


def _synthetic_plan(order: list[int], green_s: int, clear_s: int, cycles: int = 3) -> str:
    """Build a signal_plan_template.txt body: header then one phase index per second."""
    rows = [IX_ID]
    for _ in range(cycles):
        for phase in order:
            rows += [str(phase)] * green_s
            rows += ["0"] * clear_s
    return "\n".join(rows) + "\n"


# ----------------------------------------------------------------------
# Green derivation + a+1 mapping (algebra; live proof is in the companion file)
# ----------------------------------------------------------------------


def test_green_action_phases_are_the_non_clearance_phases_ascending() -> None:
    greens = green_action_phases(_hz_like_ix())
    assert greens == [1, 2, 3, 4, 5, 6, 7, 8]
    # action a -> file phase a+1 for this scenario
    assert [greens[a] for a in range(len(greens))] == [a + 1 for a in range(8)]


def test_green_derivation_falls_back_to_roadlinks_without_durations() -> None:
    ix = _hz_like_ix()
    ix.phase_durations = []  # stub env without duration metadata
    assert green_action_phases(ix) == [1, 2, 3, 4, 5, 6, 7, 8]


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------


def test_parse_signal_plan_extracts_order_durations_and_clearance() -> None:
    plan = parse_signal_plan(_synthetic_plan([1, 2, 3], green_s=20, clear_s=5))
    assert plan.header == IX_ID
    assert plan.green_order == (1, 2, 3)
    assert dict(plan.green_durations) == {1: 20, 2: 20, 3: 20}
    assert plan.clearance_phase == 0
    assert plan.clearance_duration == 5


def test_parse_signal_plan_matches_the_real_shipped_structure() -> None:
    plan = parse_signal_plan(_synthetic_plan([1, 2, 3, 4, 5, 6, 7, 8], green_s=30, clear_s=5))
    assert plan.green_order == (1, 2, 3, 4, 5, 6, 7, 8)
    assert set(plan.green_durations.values()) == {30}
    assert plan.clearance_duration == 5
    assert plan.cycle_seconds == 8 * (30 + 5)  # 280 s


@pytest.mark.parametrize(
    "text",
    ["", IX_ID + "\n", IX_ID + "\nnot_an_int\n1\n", "\n\n"],
    ids=["empty", "header_only", "non_integer_row", "blank_lines"],
)
def test_parse_signal_plan_rejects_malformed_input(text: str) -> None:
    with pytest.raises(ValueError, match="signal plan"):
        parse_signal_plan(text)


# ----------------------------------------------------------------------
# Schedule construction: shipped-plan order vs equal-split fallback
# ----------------------------------------------------------------------


def test_equal_split_cycle_is_ascending_green_actions() -> None:
    assert equal_split_cycle(8) == (0, 1, 2, 3, 4, 5, 6, 7)


def test_controller_without_plan_uses_equal_split_and_never_selects_clearance() -> None:
    env = _StubEnv([_hz_like_ix()])
    ctrl = FixedTimeController(env, k=4, plan=None)
    assert ctrl.cycle_for(IX_ID) == (0, 1, 2, 3, 4, 5, 6, 7)
    # every emitted action is a green action index (0..7), never the clearance phase
    seen = {int(ctrl.act(_info(t, range(8), current_phase=1))[0]) for t in range(8 * 4)}
    assert seen == set(range(8))


def test_controller_with_plan_follows_the_plan_order() -> None:
    env = _StubEnv([_hz_like_ix()])
    plan = parse_signal_plan(_synthetic_plan([1, 2, 3, 4, 5, 6, 7, 8], green_s=30, clear_s=5))
    ctrl = FixedTimeController(env, k=3, plan=plan)
    # ascending shipped order maps to ascending action indices
    assert ctrl.cycle_for(IX_ID) == (0, 1, 2, 3, 4, 5, 6, 7)


def test_controller_rejects_a_plan_whose_greens_do_not_match_the_intersection() -> None:
    env = _StubEnv([_hz_like_ix()])
    # plan references file phase 9, which is not a green of this intersection
    bad = PlanSchedule(
        header=IX_ID,
        green_order=(1, 2, 9),
        green_durations={1: 30, 2: 30, 9: 30},
        clearance_phase=0,
        clearance_duration=5,
        cycle_seconds=3 * 35,
    )
    with pytest.raises(ValueError, match="does not match"):
        FixedTimeController(env, k=3, plan=bad)


# ----------------------------------------------------------------------
# Cycle structure: period, per-phase hold, coverage
# ----------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 4])
def test_schedule_period_hold_and_coverage(k: int) -> None:
    env = _StubEnv([_hz_like_ix()])
    ctrl = FixedTimeController(env, k=k, plan=None)
    seq = [int(ctrl.act(_info(t, range(8), current_phase=1))[0]) for t in range(8 * k * 2)]
    # pure function of step: action(t) == (t // k) % 8
    assert seq == [(t // k) % 8 for t in range(8 * k * 2)]
    # every green appears
    assert set(seq) == set(range(8))
    # each green held exactly k consecutive steps within a cycle
    for block in range(8):
        chunk = seq[block * k : block * k + k]
        assert chunk == [block] * k
    # period is 8*k
    assert seq[: 8 * k] == seq[8 * k : 16 * k]


def test_a_controller_stuck_on_one_phase_would_fail_coverage() -> None:
    # Guards the coverage assertion above: a degenerate all-zero schedule must
    # not pass the "every green appears" check.
    seq = [0 for _ in range(8 * 4)]
    assert set(seq) != set(range(8))


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


def test_two_controllers_same_inputs_are_byte_identical() -> None:
    env = _StubEnv([_hz_like_ix()])
    rng_a = np.random.default_rng(1234)
    rng_b = np.random.default_rng(1234)
    ctrl_a = FixedTimeController(env, k=3, plan=None, rng=rng_a)
    ctrl_b = FixedTimeController(env, k=3, plan=None, rng=rng_b)
    seq_a = np.stack([ctrl_a.act(_info(t, range(8), 1)) for t in range(50)])
    seq_b = np.stack([ctrl_b.act(_info(t, range(8), 1)) for t in range(50)])
    assert np.array_equal(seq_a, seq_b)
    assert seq_a.dtype == np.int64


def test_schedule_is_a_pure_function_of_step_not_an_internal_counter() -> None:
    """Anti-drift (brief §4.1): the action must depend solely on info["step"], so a
    caller that skips, retries or evaluates out of order stays in sync with sim time.
    An implementation using ``self._t += 1`` fails this: controller B, seeing only
    step 9, would return its *first* action instead of step 9's."""
    env = _StubEnv([_hz_like_ix()])
    k = 3

    ctrl_a = FixedTimeController(env, k=k, plan=None)
    a_last = 0
    for t in range(10):
        a_last = int(ctrl_a.act(_info(t, range(8), 1))[0])  # 0,1,..,9 in order

    ctrl_b = FixedTimeController(env, k=k, plan=None)
    b_only = int(ctrl_b.act(_info(9, range(8), 1))[0])  # step 9 alone (caller skipped ahead)

    assert a_last == b_only == (9 // k) % 8

    # Out-of-order replay: each action is determined only by its step value.
    ctrl_c = FixedTimeController(env, k=k, plan=None)
    for t in [0, 1, 2, 7, 3, 9, 4]:
        assert int(ctrl_c.act(_info(t, range(8), 1))[0]) == (t // k) % 8


def test_action_is_int64_one_per_intersection_in_env_order() -> None:
    env = _StubEnv([_hz_like_ix("A"), _hz_like_ix("B")])
    ctrl = FixedTimeController(env, k=2, plan=None)
    info = {
        "step": 5,
        "intersections": {
            "A": {"avail_actions": list(range(8)), "current_phase": 1},
            "B": {"avail_actions": list(range(8)), "current_phase": 1},
        },
    }
    act = ctrl.act(info)
    assert act.shape == (2,)
    assert act.dtype == np.int64
    assert int(act[0]) == (5 // 2) % 8  # both share the schedule algebra
    assert int(act[1]) == (5 // 2) % 8


# ----------------------------------------------------------------------
# Acyclic-only guard (must explain *why*: ladder comparability)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["acyclic_bounded", "cyclic", "resco_cyclic"])
def test_non_acyclic_control_mode_is_rejected_with_ladder_reason(mode: str) -> None:
    env = _StubEnv([_hz_like_ix()], control_mode=mode)
    with pytest.raises(ValueError, match="ladder"):
        FixedTimeController(env, k=3, plan=None)


# ----------------------------------------------------------------------
# Availability-mask fallback (unreachable under acyclic; still defended)
# ----------------------------------------------------------------------


def test_mask_fallback_holds_current_green_when_scheduled_is_unavailable() -> None:
    env = _StubEnv([_hz_like_ix()])
    ctrl = FixedTimeController(env, k=3, plan=None)
    # step 3 schedules action 1; make it unavailable, current phase is 1 (action 0)
    info = _info(step=3, avail=[0, 2, 3, 4, 5, 6, 7], current_phase=1)
    assert int(ctrl.act(info)[0]) == 0  # holds current green's action, which is legal


def test_mask_fallback_uses_lowest_available_when_current_also_unavailable() -> None:
    env = _StubEnv([_hz_like_ix()])
    ctrl = FixedTimeController(env, k=3, plan=None)
    # step 3 schedules action 1; neither action 1 nor current (action 0) is available
    info = _info(step=3, avail=[2, 5, 7], current_phase=1)
    assert int(ctrl.act(info)[0]) == 2
