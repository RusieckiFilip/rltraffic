"""Action masks block disallowed actions in IPPO and MAPPO."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces

from agent.IPPOagent import IIPPOAgent, _IntersectionIPPO
from agent.MAPPOAgent import IMAPPOAgent, _CentralizedMAPPO


@dataclass
class _Ix:
    id: str
    num_phases: int


class _EnvStub:
    def __init__(self, n: int) -> None:
        self.intersections = [_Ix(f"ix{i}", 4) for i in range(n)]
        if n == 1:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.MultiDiscrete([4] * n)
        self.max_steps = 10


def test_ippo_action_mask_zeroes_disallowed_logits() -> None:
    learner = _IntersectionIPPO(
        n_actions=4,
        lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.0,
        value_coef=0.5,
        update_epochs=1,
        minibatch_size=1,
        rollout_size=1,
        hidden_dim=8,
        max_grad_norm=0.5,
        device=torch.device("cpu"),
    )
    learner.ensure_initialized(3)

    logits = torch.zeros((1, 4), dtype=torch.float32)
    logits[0, 0] = 1.0
    mask = torch.tensor([[False, True, True, True]])
    masked = learner._apply_action_mask(logits, mask)
    assert masked[0, 0].item() == torch.finfo(masked.dtype).min
    assert masked[0, 1].item() == 0.0


def test_ippo_select_action_respects_valid_actions() -> None:
    env = _EnvStub(n=1)
    agent = IIPPOAgent(env)
    learner = agent.learners["ix0"]
    learner.ensure_initialized(3)

    state = np.zeros(3, dtype=np.float32)
    # Only action 2 is allowed; deterministic selection must pick it.
    action, _log_prob, _value, mask = learner.select_action(
        state, valid_actions=[2], deterministic=True
    )
    assert action == 2
    assert mask.tolist() == [False, False, True, False]


def test_ippo_action_mask_falls_back_when_all_masked() -> None:
    # _apply_action_mask must leave logits untouched on rows with no
    # valid actions so torch.distributions.Categorical can still sample.
    learner = _IntersectionIPPO(
        n_actions=4,
        lr=1e-3, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
        entropy_coef=0.0, value_coef=0.5, update_epochs=1,
        minibatch_size=1, rollout_size=1, hidden_dim=4,
        max_grad_norm=0.5, device=torch.device("cpu"),
    )
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    mask = torch.tensor([[False, False, False, False]])
    masked = learner._apply_action_mask(logits, mask)
    assert torch.allclose(masked, logits)


def test_mappo_action_mask_zeroes_disallowed_logits() -> None:
    learner = _CentralizedMAPPO(
        action_counts=[4, 4],
        lr=1e-3, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
        entropy_coef=0.0, value_coef=0.5, update_epochs=1,
        minibatch_size=1, rollout_size=1, hidden_dim=8,
        max_grad_norm=0.5, device=torch.device("cpu"),
    )
    logits = torch.zeros((1, 4), dtype=torch.float32)
    logits[0, 1] = 5.0
    mask = torch.tensor([[True, False, True, True]])
    masked = learner._apply_action_mask(logits, mask)
    assert masked[0, 1].item() == torch.finfo(masked.dtype).min


def test_mappo_select_actions_respects_per_agent_valid_actions() -> None:
    env = _EnvStub(n=2)
    agent = IMAPPOAgent(env)
    local = [np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)]
    actions, _log_probs, _values, masks, _joint = agent.learner.select_actions(
        local,
        valid_actions=[[0], [3]],
        deterministic=True,
    )
    assert actions.tolist() == [0, 3]
    assert masks[0].tolist() == [True, False, False, False]
    assert masks[1].tolist() == [False, False, False, True]
