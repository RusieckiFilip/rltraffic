from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import BaseAgent
from .utils.utils import Utils


class _ActorCritic(nn.Module):
    """Small actor-critic network for one intersection."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value


class _IntersectionIPPO:
    """Independent PPO learner for one intersection."""

    def __init__(
        self,
        n_actions: int,
        lr: float,
        gamma: float,
        gae_lambda: float,
        clip_ratio: float,
        entropy_coef: float,
        value_coef: float,
        update_epochs: int,
        minibatch_size: int,
        rollout_size: int,
        hidden_dim: int,
        max_grad_norm: float,
        device: torch.device,
    ) -> None:
        self.n_actions = int(n_actions)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)
        self.rollout_size = int(rollout_size)
        self.hidden_dim = int(hidden_dim)
        self.max_grad_norm = float(max_grad_norm)
        self.device = device

        self.state_dim: int | None = None
        self.model: _ActorCritic | None = None
        self.optimizer: optim.Optimizer | None = None

        self.rollout_states: list[np.ndarray] = []
        self.rollout_actions: list[int] = []
        self.rollout_log_probs: list[float] = []
        self.rollout_values: list[float] = []
        self.rollout_rewards: list[float] = []
        # Stores only true terminations (gym ``terminated``) for GAE masking;
        # truncations bootstrap the value normally.
        self.rollout_terminateds: list[float] = []
        self.rollout_action_masks: list[np.ndarray] = []

    def ensure_initialized(self, state_dim: int) -> None:
        state_dim = int(state_dim)
        if self.state_dim is None:
            self.state_dim = state_dim
            self.model = _ActorCritic(state_dim, self.n_actions, self.hidden_dim).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            return
        if self.state_dim != state_dim:
            raise ValueError(
                f"State size changed for local IPPO learner: expected {self.state_dim}, got {state_dim}"
            )

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: list[int],
        deterministic: bool,
    ) -> tuple[int, float, float, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Call ensure_initialized before select_action.")

        action_mask = self._build_action_mask(valid_actions)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.model(state_t)

            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            masked_logits = self._apply_action_mask(logits, mask_t)
            dist = torch.distributions.Categorical(logits=masked_logits)

            if deterministic:
                action_t = torch.argmax(masked_logits, dim=1)
            else:
                action_t = dist.sample()

            log_prob_t = dist.log_prob(action_t)

        action = int(action_t.item())
        log_prob = float(log_prob_t.item())
        value_est = float(value.squeeze(-1).item())
        return action, log_prob, value_est, action_mask

    def estimate_value(self, state: np.ndarray) -> float:
        if self.model is None:
            raise RuntimeError("Call ensure_initialized before estimate_value.")
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, value = self.model(state_t)
        return float(value.squeeze(-1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        terminated: bool,
        action_mask: np.ndarray,
    ) -> None:
        self.rollout_states.append(state.copy())
        self.rollout_actions.append(int(action))
        self.rollout_log_probs.append(float(log_prob))
        self.rollout_values.append(float(value))
        self.rollout_rewards.append(float(reward))
        self.rollout_terminateds.append(float(terminated))
        self.rollout_action_masks.append(action_mask.astype(np.bool_).copy())

    def ready_to_update(self, episode_boundary: bool) -> bool:
        if not self.rollout_states:
            return False
        if len(self.rollout_states) >= self.rollout_size:
            return True
        return bool(episode_boundary)

    def update(self, last_value: float) -> dict[str, float] | None:
        if self.model is None or self.optimizer is None:
            return None
        if not self.rollout_states:
            return None

        advantages, returns = self._compute_advantages_and_returns(float(last_value))

        states = np.stack(self.rollout_states).astype(np.float32)
        actions = np.asarray(self.rollout_actions, dtype=np.int64)
        old_log_probs = np.asarray(self.rollout_log_probs, dtype=np.float32)
        action_masks = np.stack(self.rollout_action_masks).astype(np.bool_)

        # A single sample would normalize to zero and erase the policy
        # gradient; raw GAE advantages are still valid unnormalized.
        if advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)

        batch_size = states_t.shape[0]
        effective_minibatch = batch_size if self.minibatch_size <= 0 else min(self.minibatch_size, batch_size)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        total_updates = 0

        for _ in range(self.update_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, effective_minibatch):
                idx = permutation[start : start + effective_minibatch]

                mb_states = states_t[idx]
                mb_actions = actions_t[idx]
                mb_old_log_probs = old_log_probs_t[idx]
                mb_returns = returns_t[idx]
                mb_advantages = advantages_t[idx]
                mb_masks = masks_t[idx]

                logits, values = self.model(mb_states)
                masked_logits = self._apply_action_mask(logits, mb_masks)
                dist = torch.distributions.Categorical(logits=masked_logits)

                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surrogate_1 = ratio * mb_advantages
                surrogate_2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio,
                ) * mb_advantages

                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), mb_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                if not torch.isfinite(loss):
                    # Skip this minibatch to avoid corrupting weights.
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_loss += float(loss.item())
                total_updates += 1

        self._clear_rollout()

        if total_updates == 0:
            return None

        inv = 1.0 / float(total_updates)
        return {
            "policy_loss": total_policy_loss * inv,
            "value_loss": total_value_loss * inv,
            "entropy": total_entropy * inv,
            "total_loss": total_loss * inv,
        }

    def export_state(self) -> dict[str, Any] | None:
        if self.model is None or self.optimizer is None or self.state_dim is None:
            return None
        return {
            "state_dim": self.state_dim,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state(self, payload: dict[str, Any]) -> None:
        self.ensure_initialized(int(payload["state_dim"]))
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Learner must be initialized before loading state.")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])

    def _compute_advantages_and_returns(self, last_value: float) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(self.rollout_rewards, dtype=np.float32)
        values = np.asarray(self.rollout_values, dtype=np.float32)
        terminateds = np.asarray(self.rollout_terminateds, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = float(last_value)

        for t in range(len(rewards) - 1, -1, -1):
            non_terminal = 1.0 - terminateds[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    def _build_action_mask(self, valid_actions: list[int]) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=np.bool_)
        for action in valid_actions:
            if 0 <= action < self.n_actions:
                mask[action] = True
        if not mask.any():
            mask[:] = True
        return mask

    def _apply_action_mask(self, logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("Expected logits to have shape (batch, actions).")
        if action_mask.shape != logits.shape:
            raise ValueError("Action mask shape must match logits shape.")

        floor = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~action_mask, floor)

        empty_rows = (~action_mask).all(dim=1)
        if torch.any(empty_rows):
            masked_logits[empty_rows] = logits[empty_rows]

        return masked_logits

    def _clear_rollout(self) -> None:
        self.rollout_states.clear()
        self.rollout_actions.clear()
        self.rollout_log_probs.clear()
        self.rollout_values.clear()
        self.rollout_rewards.clear()
        self.rollout_terminateds.clear()
        self.rollout_action_masks.clear()


class IIPPOAgent(BaseAgent):
    """
    Independent PPO agent for traffic signal control.

    One local PPO learner is created per intersection.
    Every learner has its own policy/value network and rollout buffer.
    """

    def __init__(
        self,
        gym_env: Any,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 128,
        rollout_size: int = 1024,
        hidden_dim: int = 128,
        max_grad_norm: float = 0.5,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(gym_env)
        self.env = gym_env

        self.intersections = list(self.env.intersections)
        if not self.intersections:
            raise ValueError("No controllable intersections found in environment.")
        self.intersection_ids = [ix.id for ix in self.intersections]
        action_counts = Utils.infer_action_counts(self.env.action_space, self.intersections)

        self.device = Utils.resolve_device(device)

        Utils.seed_everything(seed, seed_python_random=False)

        self.learners: dict[str, _IntersectionIPPO] = {}
        for i, ix in enumerate(self.intersections):
            self.learners[ix.id] = _IntersectionIPPO(
                n_actions=action_counts[i],
                lr=lr,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_ratio=clip_ratio,
                entropy_coef=entropy_coef,
                value_coef=value_coef,
                update_epochs=update_epochs,
                minibatch_size=minibatch_size,
                rollout_size=rollout_size,
                hidden_dim=hidden_dim,
                max_grad_norm=max_grad_norm,
                device=self.device,
            )

        self.steps_done = 0
        self._last_info: dict[str, Any] = {}
        self._last_reward: float = 0.0

        self._prev_states: dict[str, np.ndarray] = {}
        self._prev_actions: dict[str, int] = {}
        self._prev_log_probs: dict[str, float] = {}
        self._prev_values: dict[str, float] = {}
        self._prev_action_masks: dict[str, np.ndarray] = {}

    def get_info(self) -> dict[str, Any]:
        return self._last_info

    def get_reward(self) -> float:
        return self._last_reward

    def get_avail_action(self, info: dict[str, Any] | None = None) -> dict[str, list[int]]:  
        if info is None:
            return {
                ix.id: list(range(self.learners[ix.id].n_actions))
                for ix in self.intersections
            }

        per_ix = Utils.extract_per_intersection_info(info, self.intersection_ids)
        out: dict[str, list[int]] = {}
        for ix in self.intersections:
            out[ix.id] = Utils.extract_valid_actions(per_ix[ix.id], self.learners[ix.id].n_actions)
        return out

    def next_action(self, ob: dict[str, Any]) -> np.ndarray:
        return self.act(ob, explore=True)

    def act(
        self,
        info: dict[str, Any],
        explore: bool = True,
        update_memory: bool = True,
    ) -> np.ndarray:
        # Evaluation calls interleaved with training must pass
        # update_memory=False, or the next observe() would store an
        # off-policy log-prob/value pair for the eval state.
        per_ix = Utils.extract_per_intersection_info(info, self.intersection_ids)

        actions: list[int] = []
        states: dict[str, np.ndarray] = {}
        log_probs: dict[str, float] = {}
        values: dict[str, float] = {}
        action_masks: dict[str, np.ndarray] = {}

        deterministic = not explore

        for ix in self.intersections:
            ix_id = ix.id
            ix_payload = per_ix[ix_id]
            state = Utils.state_from_info(ix_payload)

            learner = self.learners[ix_id]
            learner.ensure_initialized(state.shape[0])

            valid_actions = Utils.extract_valid_actions(ix_payload, learner.n_actions)
            action, log_prob, value, action_mask = learner.select_action(
                state,
                valid_actions,
                deterministic=deterministic,
            )

            states[ix_id] = state
            log_probs[ix_id] = log_prob
            values[ix_id] = value
            action_masks[ix_id] = action_mask
            actions.append(action)

        self._last_info = info
        if update_memory:
            self._prev_states = states
            self._prev_actions = dict(zip(self.intersection_ids, actions))
            self._prev_log_probs = log_probs
            self._prev_values = values
            self._prev_action_masks = action_masks

        return np.asarray(actions, dtype=np.int64)

    def observe(
        self,
        next_info: dict[str, Any],
        reward: Any,
        terminated: bool,
        truncated: bool = False,
    ) -> dict[str, dict[str, float]]:
        self._last_info = next_info
        self._last_reward = Utils.scalar_reward(reward)

        if not self._prev_states:
            return {}

        episode_boundary = bool(terminated) or bool(truncated)
        per_ix = Utils.extract_per_intersection_info(next_info, self.intersection_ids)
        metrics_per_intersection: dict[str, dict[str, float]] = {}

        for ix in self.intersections:
            ix_id = ix.id
            learner = self.learners[ix_id]
            ix_payload = per_ix[ix_id]
            next_state = Utils.state_from_info(ix_payload)

            local_reward = Utils.reward_for_intersection(
                ix_id,
                reward,
                ix_payload,
            )

            learner.store_transition(
                state=self._prev_states[ix_id],
                action=self._prev_actions[ix_id],
                log_prob=self._prev_log_probs[ix_id],
                value=self._prev_values[ix_id],
                reward=local_reward,
                terminated=bool(terminated),
                action_mask=self._prev_action_masks[ix_id],
            )

            if learner.ready_to_update(episode_boundary):
                # True termination zeroes the bootstrap; truncation keeps it.
                bootstrap_value = (
                    0.0 if terminated else learner.estimate_value(next_state)
                )
                metrics = learner.update(bootstrap_value)
                if metrics is not None:
                    metrics_per_intersection[ix_id] = metrics

        self.steps_done += 1

        if episode_boundary:
            self._prev_states = {}
            self._prev_actions = {}
            self._prev_log_probs = {}
            self._prev_values = {}
            self._prev_action_masks = {}

        return metrics_per_intersection

    def save(self, path: str) -> None:
        payload: dict[str, Any] = {
            "steps_done": self.steps_done,
            "learners": {},
        }

        for ix_id, learner in self.learners.items():
            state = learner.export_state()
            if state is not None:
                payload["learners"][ix_id] = state

        torch.save(payload, path)

    def load(self, path: str) -> None:  
        payload = torch.load(path, map_location=self.device)
        self.steps_done = int(payload.get("steps_done", 0))

        for ix_id, learner_state in payload.get("learners", {}).items():
            if ix_id not in self.learners:
                continue
            self.learners[ix_id].load_state(learner_state)


class IPPOAgent(IIPPOAgent):
    """Alias for Independent PPO agent."""

    pass
