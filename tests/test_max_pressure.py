"""End-to-end coverage for MaxPressureAgent.

Acyclic mode: agent must return green-phase *action* indices (indices
into the control's green-phase list, restricted to ``avail_actions``).
Cyclic mode: agent must return binary keep/switch actions, never raw
phase ids that the env's Discrete(2) action space would reject.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from algorithms.max_pressure import MaxPressureAgent


@dataclass
class _IxStub:
    id: str
    num_phases: int
    incoming_lanes: list[str] = field(default_factory=list)
    outgoing_lanes: list[str] = field(default_factory=list)
    phase_roadlink_mapping: list[list[int]] = field(default_factory=list)
    roadlink_lanes: list[tuple[list[str], list[str]]] = field(default_factory=list)


class _EnvStub:
    def __init__(self, control_mode: str) -> None:
        ix = _IxStub(
            id="ix0",
            num_phases=4,
            incoming_lanes=["L_in_a", "L_in_b", "L_in_c", "L_in_d"],
            outgoing_lanes=["L_out_a", "L_out_b", "L_out_c", "L_out_d"],
            phase_roadlink_mapping=[[0], [1], [2], [3]],
            roadlink_lanes=[
                (["L_in_a"], ["L_out_a"]),
                (["L_in_b"], ["L_out_b"]),
                (["L_in_c"], ["L_out_c"]),
                (["L_in_d"], ["L_out_d"]),
            ],
        )
        self.intersections = [ix]
        self.control_mode = control_mode


def _info(
    curr_phase: int,
    lane_pressures: dict[str, int],
    avail: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "lane_vehicle_count": lane_pressures,
        "intersections": {
            "ix0": {
                "current_phase": curr_phase,
                "avail_actions": [0, 1] if avail is None else avail,
                "state": [0.0],
            }
        },
    }


def test_max_pressure_acyclic_returns_green_action_index() -> None:
    agent = MaxPressureAgent(_EnvStub(control_mode="acyclic"))
    lane_counts = {
        "L_in_a": 0, "L_out_a": 0,
        "L_in_b": 0, "L_out_b": 0,
        "L_in_c": 10, "L_out_c": 0,  # best pressure on phase 2
        "L_in_d": 0, "L_out_d": 0,
    }
    action = agent.act(
        _info(curr_phase=0, lane_pressures=lane_counts, avail=[0, 1, 2, 3])
    )
    assert action.tolist() == [2]


def test_max_pressure_acyclic_honours_available_actions() -> None:
    agent = MaxPressureAgent(_EnvStub(control_mode="acyclic"))
    lane_counts = {
        "L_in_a": 0, "L_out_a": 0,
        "L_in_b": 5, "L_out_b": 0,   # best among the available actions
        "L_in_c": 10, "L_out_c": 0,  # globally best, but unavailable
        "L_in_d": 0, "L_out_d": 0,
    }
    action = agent.act(
        _info(curr_phase=0, lane_pressures=lane_counts, avail=[0, 1])
    )
    assert action.tolist() == [1]


def test_max_pressure_cyclic_returns_switch_when_desired_phase_differs() -> None:
    agent = MaxPressureAgent(_EnvStub(control_mode="cyclic"))
    lane_counts = {
        "L_in_a": 0, "L_out_a": 0,
        "L_in_b": 0, "L_out_b": 0,
        "L_in_c": 10, "L_out_c": 0,
        "L_in_d": 0, "L_out_d": 0,
    }
    action = agent.act(_info(curr_phase=0, lane_pressures=lane_counts))
    assert action.tolist() == [1]


def test_max_pressure_cyclic_returns_keep_when_already_on_best_phase() -> None:
    agent = MaxPressureAgent(_EnvStub(control_mode="cyclic"))
    lane_counts = {
        "L_in_a": 0, "L_out_a": 0,
        "L_in_b": 0, "L_out_b": 0,
        "L_in_c": 10, "L_out_c": 0,
        "L_in_d": 0, "L_out_d": 0,
    }
    action = agent.act(_info(curr_phase=2, lane_pressures=lane_counts))
    assert action.tolist() == [0]


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _cityflow_available(), reason="cityflow not installed")
def test_max_pressure_drives_cityflow_cyclic_episode_without_crash() -> None:
    from pathlib import Path

    from envs.cityflow_env import CityFlowEnv
    from envs.phase_control import CyclicPhases

    cfg = Path(__file__).resolve().parent.parent / "configs" / "sim" / "config_1x1.json"
    env = CityFlowEnv(
        cityflow_config_path=str(cfg),
        max_steps=10,
        delta_time=1,
        phase_control_cls=CyclicPhases,
    )
    try:
        agent = MaxPressureAgent(env)
        info = env.reset()
        for _ in range(5):
            action = agent.act(info)
            assert set(action.tolist()) <= {0, 1}
            _, _, truncated, info = env.step(action)
            if truncated:
                break
    finally:
        env.close()


@pytest.mark.skipif(not _cityflow_available(), reason="cityflow not installed")
def test_max_pressure_drives_cityflow_acyclic_episode_without_crash() -> None:
    from pathlib import Path

    from envs.cityflow_env import CityFlowEnv

    cfg = Path(__file__).resolve().parent.parent / "configs" / "sim" / "config_1x1.json"
    env = CityFlowEnv(
        cityflow_config_path=str(cfg),
        max_steps=10,
        delta_time=10,
    )
    try:
        agent = MaxPressureAgent(env)
        info = env.reset()
        for _ in range(5):
            action = agent.act(info)
            assert 0 <= int(action[0]) < env.intersections[0].num_phases
            _, _, truncated, info = env.step(action)
            if truncated:
                break
    finally:
        env.close()
