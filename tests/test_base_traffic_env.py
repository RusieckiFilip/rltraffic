"""Backend-agnostic tests for BaseTrafficEnv contracts.

Uses a small fake intersection + simulator so the tests don't depend on
cityflow/SUMO being installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from envs.base_traffic_env import BaseTrafficEnv
from envs.phase_control import PhaseSegment, RescoCyclicPhases


@dataclass
class _IntersectionStub:
    id: str
    num_phases: int = 4
    incoming_lanes: list[str] = field(default_factory=lambda: ["l_in_0"])
    outgoing_lanes: list[str] = field(default_factory=lambda: ["l_out_0"])
    # Defaults sized for the default num_phases=4; the acyclic controls
    # require duration metadata (all greens: duration > 5 s).
    phase_roadlink_mapping: list[list[int]] = field(
        default_factory=lambda: [[0], [1], [0], [1]]
    )
    phase_durations: list[float] = field(
        default_factory=lambda: [30.0, 30.0, 30.0, 30.0]
    )
    roadlink_lanes: list[tuple[list[str], list[str]]] = field(default_factory=list)


class _StubEnv(BaseTrafficEnv):
    """Minimal env used to exercise BaseTrafficEnv plumbing."""

    def __init__(self, **kwargs: Any) -> None:
        intersections = kwargs.pop("intersections", None)
        self._stub_intersections = list(
            intersections
            if intersections is not None
            else [_IntersectionStub(id="ix_0", num_phases=4)]
        )
        super().__init__(**kwargs)
        self._intersections = list(self._stub_intersections)
        self._num_intersections = len(self._intersections)
        self.applied_phases: list[tuple[int, int]] = []
        self.applied_phase_durations: list[tuple[int, int, int | None]] = []
        self.applied_transitions: list[tuple[int, str, int]] = []
        self.simulated_steps: list[int] = []
        self.received_actions: list[np.ndarray] = []
        self._setup_spaces()

    def _init_simulator(self) -> None:
        self._intersections = list(self._stub_intersections)
        self._num_intersections = len(self._intersections)
        self.applied_phases = []
        self.applied_phase_durations = []
        self.simulated_steps = []

    def _simulate(self, num_steps: int) -> None:
        self.simulated_steps.append(int(num_steps))

    def _set_phase(
        self,
        ix_idx: int,
        phase: int,
        duration: int | None = None,
    ) -> None:
        self.applied_phases.append((ix_idx, phase))
        self.applied_phase_durations.append((ix_idx, phase, duration))

    def _apply_segment(self, ix_idx: int, segment: PhaseSegment) -> None:
        # This fake backend has no yellow/all-red rendering; record the
        # synthetic transition segments instead of raising so the acyclic
        # controls can be exercised.
        if segment.kind == "phase":
            self._set_phase(ix_idx, segment.phase, segment.duration)
        else:
            self.applied_transitions.append(
                (ix_idx, segment.kind, int(segment.duration))
            )

    def _get_info(self) -> dict[str, Any]:
        lane_vehicle = {lid: 0 for ix in self._intersections for lid in ix.incoming_lanes}
        lane_waiting = dict(lane_vehicle)
        return {
            "lane_vehicle_count": lane_vehicle,
            "lane_waiting_vehicle_count": lane_waiting,
            "intersections": self._build_intersections_info(lane_vehicle, lane_waiting),
        }


def test_step_coerces_action_to_flat_int64() -> None:
    env = _StubEnv()
    env.reset()
    env.step(np.array([[2]], dtype=np.float64))
    assert env.applied_phases == [(0, 2)]


def test_step_accepts_python_int_action() -> None:
    env = _StubEnv()
    env.reset()
    env.step(3)
    assert env.applied_phases == [(0, 3)]


def test_set_phase_durations_rejects_undersized_second_dim() -> None:
    env = _StubEnv()
    env.reset()
    too_small = np.zeros((1, 2, 2), dtype=np.int64)  # only 2 phases, env has 4
    with pytest.raises(ValueError, match="second dimension"):
        env.set_phase_durations(too_small)


def test_set_phase_durations_survives_reset_without_initial_bounds() -> None:
    env = _StubEnv()
    env.reset()

    bounds = np.zeros((1, 4, 2), dtype=np.int64)
    bounds[..., 0] = 1
    bounds[..., 1] = 5
    env.set_phase_durations(bounds)

    env.reset()
    assert env._phase_bounds is not None
    np.testing.assert_array_equal(env._phase_bounds, bounds)


def test_initial_bounds_do_not_overwrite_post_construction_bounds_on_reset() -> None:
    initial = np.zeros((1, 4, 2), dtype=np.int64)
    initial[..., 1] = 10
    env = _StubEnv(phase_bounds=initial)
    env.reset()

    custom = np.zeros((1, 4, 2), dtype=np.int64)
    custom[..., 0] = 2
    custom[..., 1] = 7
    env.set_phase_durations(custom)

    env.reset()

    assert env._phase_bounds is not None
    np.testing.assert_array_equal(env._phase_bounds, custom)


def test_info_contains_global_lane_count_keys() -> None:
    env = _StubEnv()
    info = env.reset()
    assert "lane_vehicle_count" in info
    assert "lane_waiting_vehicle_count" in info
    assert isinstance(info["lane_vehicle_count"], dict)
    assert isinstance(info["lane_waiting_vehicle_count"], dict)


def test_base_env_rejects_unknown_kwargs() -> None:
    with pytest.raises(TypeError):
        _StubEnv(typo_argument=42)


def test_resco_cyclic_base_env_executes_switch_segments() -> None:
    ix = _IntersectionStub(
        id="ix_0",
        num_phases=6,
        phase_roadlink_mapping=[[0], [], [], [1], [], []],
        phase_durations=[30, 3, 2, 30, 3, 2],
    )
    env = _StubEnv(
        intersections=[ix],
        phase_control_cls=RescoCyclicPhases,
        delta_time=10,
    )
    env.reset()
    env._phase_controls[0].tick(10)

    _, _, _, info = env.step(np.array([1], dtype=np.int64))

    assert env.applied_phase_durations == [
        (0, 1, 3),
        (0, 2, 2),
        (0, 3, 5),
    ]
    assert env.simulated_steps == [3, 2, 5]
    assert info["intersections"]["ix_0"]["current_phase"] == 3
    assert info["intersections"]["ix_0"]["time_in_phase"] == 5
    assert info["intersections"]["ix_0"]["action_applied"]


def test_resco_cyclic_base_env_keeps_when_switch_is_too_early() -> None:
    ix = _IntersectionStub(
        id="ix_0",
        num_phases=6,
        phase_roadlink_mapping=[[0], [], [], [1], [], []],
        phase_durations=[30, 3, 2, 30, 3, 2],
    )
    env = _StubEnv(
        intersections=[ix],
        phase_control_cls=RescoCyclicPhases,
        delta_time=10,
    )
    env.reset()

    _, _, _, info = env.step(np.array([1], dtype=np.int64))

    assert env.applied_phase_durations == [(0, 0, 10)]
    assert env.simulated_steps == [10]
    assert info["intersections"]["ix_0"]["current_phase"] == 0
    assert info["intersections"]["ix_0"]["time_in_phase"] == 10
    assert not info["intersections"]["ix_0"]["action_applied"]


def test_resco_cyclic_base_env_synchronizes_multiple_intersection_segments() -> None:
    ix0 = _IntersectionStub(
        id="ix_0",
        num_phases=6,
        phase_roadlink_mapping=[[0], [], [], [1], [], []],
        phase_durations=[30, 3, 2, 30, 3, 2],
    )
    ix1 = _IntersectionStub(
        id="ix_1",
        num_phases=6,
        phase_roadlink_mapping=[[0], [], [], [1], [], []],
        phase_durations=[30, 4, 1, 30, 4, 1],
    )
    env = _StubEnv(
        intersections=[ix0, ix1],
        phase_control_cls=RescoCyclicPhases,
        delta_time=10,
    )
    env.reset()
    env._phase_controls[0].tick(10)
    env._phase_controls[1].tick(10)

    _, _, _, info = env.step(np.array([1, 1], dtype=np.int64))

    assert env.applied_phase_durations == [
        (0, 1, 3),
        (1, 1, 4),
        (0, 2, 2),
        (1, 2, 1),
        (0, 3, 5),
        (1, 3, 5),
    ]
    assert env.simulated_steps == [3, 1, 1, 5]
    assert info["intersections"]["ix_0"]["current_phase"] == 3
    assert info["intersections"]["ix_0"]["time_in_phase"] == 5
    assert info["intersections"]["ix_1"]["current_phase"] == 3
    assert info["intersections"]["ix_1"]["time_in_phase"] == 5
    assert info["intersections"]["ix_0"]["action_applied"]
    assert info["intersections"]["ix_1"]["action_applied"]
