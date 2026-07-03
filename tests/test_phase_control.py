from __future__ import annotations

import numpy as np
import pytest

from envs.phase_control import (
    AcyclicBoundedPhases,
    AcyclicPhases,
    CyclicPhases,
    RescoCyclicPhases,
)


def test_cyclic_available_actions_from_duration_constraints() -> None:
    control = CyclicPhases()
    control.initialize(4)
    bounds = np.zeros((4, 2), dtype=np.int64)
    bounds[:, 0] = 2
    bounds[:, 1] = 3
    control.set_phase_bounds(bounds)

    assert control.available_actions() == [0]
    control.tick(2)
    assert control.available_actions() == [0, 1]
    control.tick(1)
    assert control.available_actions() == [1]
    assert control.time_in_phase() == 3


def test_cyclic_target_phase_mapping() -> None:
    control = CyclicPhases()
    control.initialize(4)
    control.apply_phase(2)

    assert control.target_phase(action=0) == 2
    assert control.target_phase(action=1) == 3


@pytest.mark.parametrize("action", [-1, 2, 3])
def test_cyclic_rejects_unknown_actions(action: int) -> None:
    control = CyclicPhases()
    control.initialize(4)
    with pytest.raises(ValueError):
        control.target_phase(action=action)


def test_acyclic_bounded_available_actions_from_duration_constraints() -> None:
    control = AcyclicBoundedPhases()
    control.initialize(4, phase_durations=[30, 30, 30, 30], delta_time=10)
    control.apply_phase(1)
    bounds = np.zeros((4, 2), dtype=np.int64)
    bounds[:, 0] = 2
    bounds[:, 1] = 3
    control.set_phase_bounds(bounds)

    assert control.available_actions() == [1]
    control.tick(2)
    assert control.available_actions() == [0, 1, 2, 3]
    control.tick(1)
    assert control.available_actions() == [0, 2, 3]


def test_acyclic_ignores_duration_bounds() -> None:
    control = AcyclicPhases()
    control.initialize(4, phase_durations=[30, 30, 30, 30], delta_time=10)
    bounds = np.zeros((4, 2), dtype=np.int64)
    bounds[:, 0] = 2
    bounds[:, 1] = 3
    control.set_phase_bounds(bounds)

    # The forced clearance already enforces safe timing, so every green is
    # selectable at every tick regardless of bounds.
    assert control.available_actions() == [0, 1, 2, 3]
    control.tick(5)
    assert control.available_actions() == [0, 1, 2, 3]


def test_acyclic_target_phase_mapping() -> None:
    control = AcyclicPhases()
    control.initialize(4, phase_durations=[30, 30, 30, 30], delta_time=10)
    control.apply_phase(2)

    assert control.target_phase(action=0) == 0
    assert control.target_phase(action=3) == 3


@pytest.mark.parametrize("action", [-1, 4, 9])
def test_acyclic_rejects_out_of_range_actions(action: int) -> None:
    control = AcyclicPhases()
    control.initialize(4, phase_durations=[30, 30, 30, 30], delta_time=10)
    with pytest.raises(ValueError):
        control.target_phase(action=action)


def test_resco_cyclic_switch_plan_after_guard() -> None:
    control = RescoCyclicPhases()
    control.initialize(
        6,
        phase_roadlink_mapping=[[0], [0], [], [1], [1], []],
        phase_durations=[30, 3, 2, 30, 3, 2],
        phase_states=["GG", "yg", "rr", "GG", "yg", "rr"],
        delta_time=10,
    )
    bounds = np.zeros((6, 2), dtype=np.int64)
    bounds[:, 0] = 5
    bounds[:, 1] = 90
    control.set_phase_bounds(bounds)
    control.tick(10)

    plan = control.phase_plan(action=1, delta_seconds=10)

    assert [(s.phase, s.duration) for s in plan.segments] == [
        (1, 3),
        (2, 2),
        (3, 5),
    ]
    assert plan.action_applied


def test_resco_cyclic_early_switch_is_keep() -> None:
    control = RescoCyclicPhases()
    control.initialize(
        6,
        phase_roadlink_mapping=[[0], [], [], [1], [], []],
        phase_durations=[30, 3, 2, 30, 3, 2],
        delta_time=10,
    )
    control.tick(9)

    plan = control.phase_plan(action=1, delta_seconds=10)

    assert [(s.phase, s.duration) for s in plan.segments] == [(0, 10)]
    assert not plan.action_applied
    assert control.available_actions() == [0, 1]
    control.apply_plan(plan)
    assert control.time_in_phase() == 19


def test_resco_cyclic_ignores_short_permissive_clearance_as_green() -> None:
    control = RescoCyclicPhases()
    control.initialize(
        6,
        phase_roadlink_mapping=[[0], [0], [0], [1], [1], [1]],
        phase_durations=[20, 3, 2, 20, 3, 2],
        phase_states=["GG", "gg", "gr", "GG", "gg", "gr"],
        delta_time=10,
    )
    control.tick(10)

    plan = control.phase_plan(action=1, delta_seconds=10)

    assert [(s.phase, s.duration) for s in plan.segments] == [
        (1, 3),
        (2, 2),
        (3, 5),
    ]


def test_resco_cyclic_rejects_missing_transition_phases() -> None:
    control = RescoCyclicPhases()

    with pytest.raises(ValueError, match="transition phases"):
        control.initialize(
            4,
            phase_roadlink_mapping=[[0], [1], [2], [3]],
            phase_durations=[30, 30, 30, 30],
            delta_time=10,
        )


def test_resco_cyclic_rejects_transition_that_fills_delta_time() -> None:
    control = RescoCyclicPhases()

    with pytest.raises(ValueError, match="shorter than delta_time"):
        control.initialize(
            6,
            phase_roadlink_mapping=[[0], [], [], [1], [], []],
            phase_durations=[30, 3, 2, 30, 3, 2],
            delta_time=5,
        )
