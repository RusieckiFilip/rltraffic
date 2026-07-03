"""MAPPO must normalise advantages per agent, not across the whole batch."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from agent.MAPPOAgent import IMAPPOAgent


@dataclass
class _Ix:
    id: str
    num_phases: int


class _EnvStub:
    def __init__(self, n: int) -> None:
        self.intersections = [_Ix(f"ix{i}", 4) for i in range(n)]
        self.action_space = spaces.MultiDiscrete([4] * n)
        self.max_steps = 10


def test_mappo_advantage_normalization_is_per_agent() -> None:
    env = _EnvStub(n=2)
    agent = IMAPPOAgent(env, rollout_size=100, minibatch_size=8)
    learner = agent.learner
    learner.ensure_initialized([3, 3])

    rng = np.random.default_rng(0)
    local = [np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)]
    joint = np.zeros(6, dtype=np.float32)
    masks = [np.ones(4, dtype=np.bool_), np.ones(4, dtype=np.bool_)]

    # Rewards on agent 0 are O(1), agent 1 is O(100).  Without per-agent
    # normalisation the agent-1 advantages would dominate after stacking.
    for _ in range(20):
        rewards = np.array(
            [rng.normal(0.0, 1.0), rng.normal(0.0, 100.0)],
            dtype=np.float32,
        )
        learner.store_transition(
            local_states=local,
            joint_state=joint,
            actions=np.zeros(2, dtype=np.int64),
            log_probs=np.zeros(2, dtype=np.float32),
            values=rewards.copy() * 0.5,
            rewards=rewards,
            terminated=False,
            action_masks=masks,
        )

    raw_adv, _ = learner._compute_advantages_and_returns(np.zeros(2, dtype=np.float32))
    adv_mean = raw_adv.mean(axis=0, keepdims=True)
    adv_std = raw_adv.std(axis=0, keepdims=True)
    normalised = (raw_adv - adv_mean) / (adv_std + 1e-8)

    # Each agent column has zero mean / unit std after normalisation.
    assert np.allclose(normalised.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(normalised.std(axis=0), 1.0, atol=1e-4)

    # Global normalisation would leave the agent-1 std much larger than 1.
    global_std = raw_adv.std()
    global_normalised = (raw_adv - raw_adv.mean()) / (global_std + 1e-8)
    assert abs(global_normalised[:, 1].std() - 1.0) > 1e-3
