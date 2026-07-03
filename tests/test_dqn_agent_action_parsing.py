"""Action parsing for DQN/IPPO/MAPPO agents.

The agents delegate avail-action parsing to ``Utils.extract_valid_actions``.
These tests pin down the parser semantics (list of indices, not a mask) and
the agent's behaviour against a small stub env.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from agent.DQNAgent import IDQNAgent
from agent.utils.utils import Utils


@dataclass
class _IntersectionStub:
    id: str
    num_phases: int


class _EnvStub:
    def __init__(self, intersections: list[_IntersectionStub], control_mode: str) -> None:
        self.intersections = intersections
        self.control_mode = control_mode
        if control_mode == "cyclic":
            self.action_space = spaces.Discrete(2)
        elif control_mode == "acyclic":
            self.action_space = spaces.Discrete(intersections[0].num_phases)
        else:
            raise ValueError("Unsupported control mode in test stub.")
        self.max_steps = 10


def test_extract_valid_actions_returns_indices_not_mask() -> None:
    payload = {"state": [0.0, 1.0], "avail_actions": [0, 1]}
    parsed = Utils.extract_valid_actions(payload, n_actions=2)
    assert parsed == [0, 1]


def test_extract_valid_actions_filters_out_of_range_values() -> None:
    payload = {"state": [0.0], "avail_actions": [0, 5, "bad", 2, 2]}
    parsed = Utils.extract_valid_actions(payload, n_actions=3)
    assert parsed == [0, 2]


def test_extract_valid_actions_defaults_when_missing() -> None:
    payload = {"state": [0.0]}
    parsed = Utils.extract_valid_actions(payload, n_actions=3)
    assert parsed == [0, 1, 2]


def test_extract_valid_actions_falls_back_when_empty() -> None:
    payload = {"state": [0.0], "avail_actions": [99, -1]}
    parsed = Utils.extract_valid_actions(payload, n_actions=2)
    assert parsed == [0, 1]


def test_extract_valid_actions_accepts_numpy_array() -> None:
    payload = {"state": [0.0], "avail_actions": np.array([1, 0])}
    parsed = Utils.extract_valid_actions(payload, n_actions=3)
    assert parsed == [0, 1]


def test_dqn_agent_respects_avail_actions_in_cyclic_mode() -> None:
    env = _EnvStub([_IntersectionStub("ix0", 4)], control_mode="cyclic")
    agent = IDQNAgent(env, epsilon_start=0.0, epsilon_end=0.0)

    info = {
        "intersections": {
            "ix0": {"state": [0.0, 1.0, 2.0], "avail_actions": [1]},
        }
    }
    action = agent.act(info, explore=False)
    assert action.tolist() == [1]
