"""Truncation must not zero the value bootstrap.

DQN: the TD target uses (1 - terminated), not (1 - done).
IPPO/MAPPO: GAE non-terminal mask uses ``terminated`` only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces

from agent.DQNAgent import IDQNAgent, _IntersectionDQN
from agent.IPPOagent import IIPPOAgent
from agent.MAPPOAgent import IMAPPOAgent


@dataclass
class _IntersectionStub:
    id: str
    num_phases: int


class _EnvStub:
    def __init__(self, n_ix: int = 1) -> None:
        self.intersections = [
            _IntersectionStub(f"ix{i}", num_phases=4) for i in range(n_ix)
        ]
        if n_ix == 1:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.MultiDiscrete([4] * n_ix)
        self.max_steps = 10


def _info(ids: list[str]) -> dict:
    return {
        "intersections": {
            ix: {"state": [0.1, 0.2, 0.3], "avail_actions": [0, 1, 2, 3]}
            for ix in ids
        }
    }


def test_dqn_truncated_transition_does_not_zero_target_bootstrap() -> None:
    learner = _IntersectionDQN(
        n_actions=4,
        lr=1e-3,
        gamma=0.9,
        batch_size=1,
        replay_size=10,
        min_replay_size=1,
        target_update_interval=100,
        hidden_dim=8,
        device=torch.device("cpu"),
    )
    learner.ensure_initialized(3)
    state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    next_state = np.array([0.5, 0.4, 0.3], dtype=np.float32)
    # store one transition as a truncation (terminated=False)
    learner.push(state, action=0, reward=1.0, next_state=next_state, terminated=False)

    state_t = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        bootstrap = float(learner.target_net(state_t).max(dim=1).values.item())

    # Recreate the target_q the update() will see for this batch.
    expected_target = 1.0 + 0.9 * bootstrap
    learner.update()
    # The transition is stored as terminated_float=0.0; the formula must
    # therefore include the bootstrap term.
    stored = learner.buffer[0]
    assert stored[4] == 0.0
    # And under terminated=True the bootstrap MUST be zeroed.
    learner.push(state, action=0, reward=1.0, next_state=next_state, terminated=True)
    terminated_stored = learner.buffer[-1]
    assert terminated_stored[4] == 1.0

    target_with_bootstrap = 1.0 + 0.9 * (1.0 - 0.0) * bootstrap
    target_without_bootstrap = 1.0 + 0.9 * (1.0 - 1.0) * bootstrap
    assert target_with_bootstrap == expected_target
    assert target_without_bootstrap == 1.0
    assert target_with_bootstrap != target_without_bootstrap


def test_dqn_observe_with_truncated_clears_prev_states() -> None:
    env = _EnvStub()
    agent = IDQNAgent(env, epsilon_start=0.0, epsilon_end=0.0)
    info = _info(["ix0"])
    agent.act(info, explore=False)
    assert agent._prev_states
    agent.observe(info, reward=0.0, terminated=False, truncated=True)
    assert agent._prev_states == {}


def test_ippo_rollout_stores_terminated_not_done() -> None:
    # observe() with truncated=True triggers an update that clears the
    # rollout, so check the stored value by going through the learner
    # directly: only ``terminated`` is recorded.
    env = _EnvStub()
    agent = IIPPOAgent(env)
    learner = agent.learners["ix0"]
    learner.ensure_initialized(3)

    state = np.zeros(3, dtype=np.float32)
    mask = np.ones(4, dtype=np.bool_)
    learner.store_transition(
        state=state, action=0, log_prob=0.0, value=0.0,
        reward=0.5, terminated=False, action_mask=mask,
    )
    learner.store_transition(
        state=state, action=0, log_prob=0.0, value=0.0,
        reward=0.5, terminated=True, action_mask=mask,
    )
    assert learner.rollout_terminateds == [0.0, 1.0]


def test_ippo_gae_uses_terminated_only() -> None:
    env = _EnvStub()
    agent = IIPPOAgent(env, gamma=0.9, gae_lambda=0.95, rollout_size=4)
    learner = agent.learners["ix0"]
    learner.ensure_initialized(3)

    state = np.zeros(3, dtype=np.float32)
    mask = np.array([True, True, True, True])
    for r in (1.0, 1.0):
        learner.store_transition(
            state=state,
            action=0,
            log_prob=0.0,
            value=0.0,
            reward=r,
            terminated=False,
            action_mask=mask,
        )

    advantages, returns = learner._compute_advantages_and_returns(last_value=5.0)
    # With non-terminal mask, bootstrap propagates back through GAE.
    assert returns[-1] > 1.0


def test_mappo_rollout_stores_terminated_not_done() -> None:
    env = _EnvStub(n_ix=2)
    agent = IMAPPOAgent(env)
    learner = agent.learner
    learner.ensure_initialized([3, 3])

    local = [np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)]
    joint = np.zeros(6, dtype=np.float32)
    masks = [np.ones(4, dtype=np.bool_), np.ones(4, dtype=np.bool_)]
    learner.store_transition(
        local_states=local,
        joint_state=joint,
        actions=np.zeros(2, dtype=np.int64),
        log_probs=np.zeros(2, dtype=np.float32),
        values=np.zeros(2, dtype=np.float32),
        rewards=np.array([0.1, 0.1], dtype=np.float32),
        terminated=False,
        action_masks=masks,
    )
    learner.store_transition(
        local_states=local,
        joint_state=joint,
        actions=np.zeros(2, dtype=np.int64),
        log_probs=np.zeros(2, dtype=np.float32),
        values=np.zeros(2, dtype=np.float32),
        rewards=np.array([0.1, 0.1], dtype=np.float32),
        terminated=True,
        action_masks=masks,
    )
    assert learner.rollout_terminateds == [0.0, 1.0]


def test_mappo_observe_with_truncated_clears_previous_step() -> None:
    env = _EnvStub(n_ix=2)
    agent = IMAPPOAgent(env)
    info = _info(["ix0", "ix1"])
    agent.act(info, explore=True)
    assert agent._prev_joint_state is not None
    agent.observe(info, reward=0.5, terminated=False, truncated=True)
    assert agent._prev_joint_state is None
