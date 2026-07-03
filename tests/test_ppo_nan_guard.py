"""PPO and MAPPO must skip optimiser steps when the loss is non-finite."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces

from agent.IPPOagent import IIPPOAgent
from agent.MAPPOAgent import IMAPPOAgent


@dataclass
class _Ix:
    id: str
    num_phases: int


class _EnvStub:
    def __init__(self, n: int = 1) -> None:
        self.intersections = [_Ix(f"ix{i}", 4) for i in range(n)]
        if n == 1:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.MultiDiscrete([4] * n)
        self.max_steps = 10


def _record_weight(model) -> torch.Tensor:
    p = next(model.parameters())
    return p.detach().clone()


def test_ippo_update_with_nan_reward_does_not_corrupt_weights() -> None:
    env = _EnvStub()
    agent = IIPPOAgent(env, minibatch_size=4)
    learner = agent.learners["ix0"]
    learner.ensure_initialized(3)

    state = np.zeros(3, dtype=np.float32)
    mask = np.ones(4, dtype=np.bool_)
    for _ in range(8):
        learner.store_transition(
            state=state, action=0, log_prob=0.0, value=0.0,
            reward=float("nan"), terminated=False, action_mask=mask,
        )

    before = _record_weight(learner.model)
    learner.update(last_value=0.0)
    after = _record_weight(learner.model)
    assert torch.equal(before, after)


def test_mappo_update_with_nan_reward_does_not_corrupt_weights() -> None:
    env = _EnvStub(n=2)
    agent = IMAPPOAgent(env, minibatch_size=4)
    learner = agent.learner
    learner.ensure_initialized([3, 3])

    local = [np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)]
    joint = np.zeros(6, dtype=np.float32)
    masks = [np.ones(4, dtype=np.bool_), np.ones(4, dtype=np.bool_)]
    nan_rewards = np.array([float("nan"), float("nan")], dtype=np.float32)
    for _ in range(8):
        learner.store_transition(
            local_states=local,
            joint_state=joint,
            actions=np.zeros(2, dtype=np.int64),
            log_probs=np.zeros(2, dtype=np.float32),
            values=np.zeros(2, dtype=np.float32),
            rewards=nan_rewards,
            terminated=False,
            action_masks=masks,
        )

    before = _record_weight(learner.critic)
    learner.update(np.zeros(2, dtype=np.float32))
    after = _record_weight(learner.critic)
    assert torch.equal(before, after)
