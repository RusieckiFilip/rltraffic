from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import BaseAgent
from .utils.utils import Utils


class _Actor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CentralCritic(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _RunningNorm:
    """Running mean/std normalizer (Welford) for critic inputs and value targets.

    Stats only advance through :meth:`update`, which the learner calls during
    training; evaluation paths read frozen stats. Batches smaller than two
    samples are ignored so a single-sample flush cannot collapse the variance.
    """

    def __init__(self, dim: int, device: torch.device, epsilon: float = 1e-4) -> None:
        self.dim = int(dim)
        self.device = device
        self.mean = torch.zeros(self.dim, device=device)
        self.var = torch.ones(self.dim, device=device)
        self.count = float(epsilon)

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, self.dim).to(self.device)
        batch_count = x.shape[0]
        if batch_count < 2:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / total)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * (self.count * batch_count / total)
        self.var = m2 / total
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(self.var + 1e-8) + self.mean

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.mean = payload["mean"].to(self.device)
        self.var = payload["var"].to(self.device)
        self.count = float(payload["count"])


class _CentralizedMAPPO:
    def __init__(
        self,
        action_counts: list[int],
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
        self.action_counts = [int(v) for v in action_counts]
        if not self.action_counts:
            raise ValueError("MAPPO requires at least one controlled intersection.")
        self.num_agents = len(self.action_counts)
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

        self.local_state_dims: list[int] | None = None
        self.global_feature_dim: int | None = None
        self.global_state_dim: int | None = None
        self.actors: nn.ModuleList | None = None
        self.critic: _CentralCritic | None = None
        self.optimizer: optim.Optimizer | None = None
        # Running normalizers: critic input (concat locals + global block) and
        # per-agent value targets. Created in ensure_initialized().
        self.input_norm: _RunningNorm | None = None
        self.value_norm: _RunningNorm | None = None

        self.rollout_local_states: list[list[np.ndarray]] = []
        self.rollout_joint_states: list[np.ndarray] = []
        self.rollout_actions: list[np.ndarray] = []
        self.rollout_log_probs: list[np.ndarray] = []
        self.rollout_values: list[np.ndarray] = []
        self.rollout_rewards: list[np.ndarray] = []
        # Stores only true terminations (gym ``terminated``) for GAE masking;
        # truncations bootstrap the value normally.
        self.rollout_terminateds: list[float] = []
        self.rollout_action_masks: list[list[np.ndarray]] = []

    def ensure_initialized(
        self,
        local_state_dims: list[int],
        global_feature_dim: int = 0,
    ) -> None:
        local_state_dims = [int(v) for v in local_state_dims]
        global_feature_dim = int(global_feature_dim)
        if len(local_state_dims) != self.num_agents:
            raise ValueError(
                "State sizes count does not match MAPPO controlled intersections."
            )
        # Centralized critic sees every local state plus a shared global block.
        global_state_dim = int(sum(local_state_dims)) + global_feature_dim

        if self.local_state_dims is None:
            self.local_state_dims = local_state_dims
            self.global_feature_dim = global_feature_dim
            self.global_state_dim = global_state_dim
            self.actors = nn.ModuleList(
                [
                    _Actor(state_dim, n_actions, self.hidden_dim)
                    for state_dim, n_actions in zip(local_state_dims, self.action_counts)
                ]
            ).to(self.device)
            self.critic = _CentralCritic(
                global_state_dim,
                self.num_agents,
                self.hidden_dim,
            ).to(self.device)
            self.input_norm = _RunningNorm(global_state_dim, self.device)
            self.value_norm = _RunningNorm(self.num_agents, self.device)
            params = list(self.actors.parameters()) + list(self.critic.parameters())
            self.optimizer = optim.Adam(params, lr=self.lr)
            return

        if self.local_state_dims != local_state_dims:
            raise ValueError(
                "State sizes changed for MAPPO: "
                f"expected {self.local_state_dims}, got {local_state_dims}"
            )
        if self.global_feature_dim != global_feature_dim:
            raise ValueError(
                "Global feature size changed for MAPPO: "
                f"expected {self.global_feature_dim}, got {global_feature_dim}"
            )

    def select_actions(
        self,
        local_states: list[np.ndarray],
        valid_actions: list[list[int]],
        global_features: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
        if global_features is None:
            global_features = np.zeros(0, dtype=np.float32)
        global_features = np.asarray(global_features, dtype=np.float32).reshape(-1)
        self.ensure_initialized(
            [state.shape[0] for state in local_states],
            global_features.shape[0],
        )
        if self.actors is None:
            raise RuntimeError("MAPPO actors were not initialized.")

        joint_state = self.build_joint_state(local_states, global_features)
        if len(valid_actions) != self.num_agents:
            raise ValueError(
                "Valid actions count does not match MAPPO controlled intersections."
            )
        action_masks = [
            self._build_action_mask(actions, n_actions)
            for actions, n_actions in zip(valid_actions, self.action_counts)
        ]

        actions: list[int] = []
        log_probs: list[float] = []

        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                state_t = torch.as_tensor(
                    local_states[i],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                logits = actor(state_t)
                mask_t = torch.as_tensor(
                    action_masks[i],
                    dtype=torch.bool,
                    device=self.device,
                ).unsqueeze(0)
                masked_logits = self._apply_action_mask(logits, mask_t)
                dist = torch.distributions.Categorical(logits=masked_logits)
                action_t = (
                    torch.argmax(masked_logits, dim=1)
                    if deterministic
                    else dist.sample()
                )
                actions.append(int(action_t.item()))
                log_probs.append(float(dist.log_prob(action_t).item()))

            values = self.estimate_values(joint_state)

        return (
            np.asarray(actions, dtype=np.int64),
            np.asarray(log_probs, dtype=np.float32),
            values,
            action_masks,
            joint_state,
        )

    @staticmethod
    def build_joint_state(
        local_states: list[np.ndarray],
        global_features: np.ndarray,
    ) -> np.ndarray:
        """Concatenate every local state with the shared global feature block."""
        parts = [np.asarray(s, dtype=np.float32).reshape(-1) for s in local_states]
        parts.append(np.asarray(global_features, dtype=np.float32).reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    def estimate_values(self, joint_state: np.ndarray) -> np.ndarray:
        if self.critic is None or self.input_norm is None or self.value_norm is None:
            raise RuntimeError("MAPPO critic was not initialized.")
        with torch.no_grad():
            state_t = torch.as_tensor(
                joint_state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            norm_values = self.critic(self.input_norm.normalize(state_t)).squeeze(0)
            # Critic predicts in normalized space; GAE needs raw-scale values.
            values = self.value_norm.denormalize(norm_values)
        return values.detach().cpu().numpy().astype(np.float32)

    def store_transition(
        self,
        local_states: list[np.ndarray],
        joint_state: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        terminated: bool,
        action_masks: list[np.ndarray],
    ) -> None:
        self.rollout_local_states.append([state.copy() for state in local_states])
        self.rollout_joint_states.append(joint_state.copy())
        self.rollout_actions.append(actions.astype(np.int64).copy())
        self.rollout_log_probs.append(log_probs.astype(np.float32).copy())
        self.rollout_values.append(values.astype(np.float32).copy())
        self.rollout_rewards.append(rewards.astype(np.float32).copy())
        self.rollout_terminateds.append(float(terminated))
        self.rollout_action_masks.append(
            [mask.astype(np.bool_).copy() for mask in action_masks]
        )

    def ready_to_update(self, episode_boundary: bool) -> bool:
        if not self.rollout_joint_states:
            return False
        return (
            len(self.rollout_joint_states) >= self.rollout_size
            or bool(episode_boundary)
        )

    def update(self, last_values: np.ndarray) -> dict[str, float] | None:
        if self.actors is None or self.critic is None or self.optimizer is None:
            return None
        if not self.rollout_joint_states:
            return None

        advantages, returns = self._compute_advantages_and_returns(last_values)
        # Per-agent normalisation: each column (agent) has its own scale.
        # A single sample would normalize to zero and erase the policy
        # gradient; raw GAE advantages are still valid unnormalized.
        if advantages.shape[0] > 1:
            adv_mean = advantages.mean(axis=0, keepdims=True)
            adv_std = advantages.std(axis=0, keepdims=True)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        joint_states_t = torch.as_tensor(
            np.stack(self.rollout_joint_states),
            dtype=torch.float32,
            device=self.device,
        )
        actions_t = torch.as_tensor(
            np.stack(self.rollout_actions),
            dtype=torch.int64,
            device=self.device,
        )
        old_log_probs_t = torch.as_tensor(
            np.stack(self.rollout_log_probs),
            dtype=torch.float32,
            device=self.device,
        )
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(
            advantages,
            dtype=torch.float32,
            device=self.device,
        )

        # Refresh running stats on this rollout, then train the critic in
        # normalized space: inputs via input_norm, targets via value_norm.
        if self.input_norm is None or self.value_norm is None:
            raise RuntimeError("MAPPO normalizers were not initialized.")
        self.input_norm.update(joint_states_t)
        self.value_norm.update(returns_t)
        norm_returns_t = self.value_norm.normalize(returns_t)

        local_states_t = [
            torch.as_tensor(
                np.stack([step_states[i] for step_states in self.rollout_local_states]),
                dtype=torch.float32,
                device=self.device,
            )
            for i in range(self.num_agents)
        ]
        masks_t = [
            torch.as_tensor(
                np.stack([step_masks[i] for step_masks in self.rollout_action_masks]),
                dtype=torch.bool,
                device=self.device,
            )
            for i in range(self.num_agents)
        ]

        batch_size = joint_states_t.shape[0]
        effective_minibatch = (
            batch_size
            if self.minibatch_size <= 0
            else min(self.minibatch_size, batch_size)
        )

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
        }
        updates = 0

        for _ in range(self.update_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, effective_minibatch):
                idx = permutation[start : start + effective_minibatch]

                policy_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                entropy = torch.zeros((), dtype=torch.float32, device=self.device)

                for i, actor in enumerate(self.actors):
                    logits = actor(local_states_t[i][idx])
                    masked_logits = self._apply_action_mask(logits, masks_t[i][idx])
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    new_log_probs = dist.log_prob(actions_t[idx, i])
                    ratio = torch.exp(new_log_probs - old_log_probs_t[idx, i])

                    surrogate_1 = ratio * advantages_t[idx, i]
                    surrogate_2 = torch.clamp(
                        ratio,
                        1.0 - self.clip_ratio,
                        1.0 + self.clip_ratio,
                    ) * advantages_t[idx, i]
                    policy_loss = policy_loss - torch.min(surrogate_1, surrogate_2).mean()
                    entropy = entropy + dist.entropy().mean()

                policy_loss = policy_loss / self.num_agents
                entropy = entropy / self.num_agents
                values = self.critic(self.input_norm.normalize(joint_states_t[idx]))
                value_loss = F.mse_loss(values, norm_returns_t[idx])
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                if not torch.isfinite(loss):
                    # Skip this minibatch to avoid corrupting weights.
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actors.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                totals["total_loss"] += float(loss.item())
                updates += 1

        self._clear_rollout()
        if updates == 0:
            return None
        return {key: value / updates for key, value in totals.items()}

    def export_state(self) -> dict[str, Any] | None:
        if (
            self.actors is None
            or self.critic is None
            or self.optimizer is None
            or self.local_state_dims is None
            or self.input_norm is None
            or self.value_norm is None
        ):
            return None
        return {
            "local_state_dims": self.local_state_dims,
            "global_feature_dim": self.global_feature_dim,
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "input_norm": self.input_norm.state_dict(),
            "value_norm": self.value_norm.state_dict(),
        }

    def load_state(self, payload: dict[str, Any]) -> None:
        self.ensure_initialized(
            [int(v) for v in payload["local_state_dims"]],
            int(payload.get("global_feature_dim", 0)),
        )
        if (
            self.actors is None
            or self.critic is None
            or self.optimizer is None
            or self.input_norm is None
            or self.value_norm is None
        ):
            raise RuntimeError("MAPPO learner must be initialized before loading state.")
        self.actors.load_state_dict(payload["actors"])
        self.critic.load_state_dict(payload["critic"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if "input_norm" in payload:
            self.input_norm.load_state_dict(payload["input_norm"])
        if "value_norm" in payload:
            self.value_norm.load_state_dict(payload["value_norm"])

    def _compute_advantages_and_returns(
        self,
        last_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.stack(self.rollout_rewards).astype(np.float32)
        values = np.stack(self.rollout_values).astype(np.float32)
        terminateds = np.asarray(self.rollout_terminateds, dtype=np.float32)
        last_values = np.asarray(last_values, dtype=np.float32).reshape(self.num_agents)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(self.num_agents, dtype=np.float32)
        next_values = last_values

        for t in range(len(rewards) - 1, -1, -1):
            non_terminal = 1.0 - terminateds[t]
            delta = rewards[t] + self.gamma * next_values * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_values = values[t]

        return advantages, advantages + values

    def _build_action_mask(self, valid_actions: list[int], n_actions: int) -> np.ndarray:
        mask = np.zeros(n_actions, dtype=np.bool_)
        for action in valid_actions:
            if 0 <= action < n_actions:
                mask[action] = True
        if not mask.any():
            mask[:] = True
        return mask

    def _apply_action_mask(
        self,
        logits: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
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
        self.rollout_local_states.clear()
        self.rollout_joint_states.clear()
        self.rollout_actions.clear()
        self.rollout_log_probs.clear()
        self.rollout_values.clear()
        self.rollout_rewards.clear()
        self.rollout_terminateds.clear()
        self.rollout_action_masks.clear()


# ----------------------------------------------------------------------
# Contract C8 -- a MAPPO checkpoint freezes the env's global metric set.
# Added under AUTHORISATION C (2026-08-06); see docs/patches/README.md.
# ----------------------------------------------------------------------


def env_global_metric_keys(env: Any) -> list[str] | None:
    """The metric keys *env* will expose in ``info["metrics"]``, or ``None``.

    THE single derivation of that set.  ``offline/migrate_mappo_checkpoints.py``
    imports this function instead of reimplementing it: two copies of a key-set
    derivation are exactly where a guard and the data it guards drift apart with
    nothing failing.

    Mirrors ``metrics/base.py::compute_all``, which builds
    ``{n: self.get(n) for n in self._requested if self.REGISTERED[n].compute is not
    None}``.  The filter is copied rather than approximated, so the result equals
    ``sorted(info["metrics"])`` **by construction** rather than by luck.

    Pure -- reads only ``requested`` and ``REGISTERED``, makes no engine call -- so
    it is safe on an env that has been constructed but never reset, which is the
    state ``offline/collect.py`` loads a checkpoint in.  ``None`` means "this env
    runs no metrics pipeline, so the set cannot be derived"; it is not the same as
    an empty set, and the caller must treat it as "cannot check", never as "match".
    """
    metrics = getattr(env, "metrics", None)
    if metrics is None:
        return None
    requested = getattr(metrics, "requested", None)
    registered = getattr(metrics, "REGISTERED", None)
    if requested is None or registered is None:
        return None
    return sorted(
        str(name) for name in requested if registered[name].compute is not None
    )


def _assert_checkpoint_metric_keys(
    payload: dict[str, Any], env: Any, path: str
) -> None:
    """Enforce contract C8 when a checkpoint is loaded.  Exactly five cases.

    ================================  ==========================================
    checkpoint / env                  behaviour
    ================================  ==========================================
    key absent                        loud warning (pre-migration checkpoint)
    key present, ``None``             loud warning (saved before features built)
    key present, env has no metrics   loud warning (cannot check)
    key present, sets equal           silent
    key present, sets differ          ``ValueError`` naming the difference
    ================================  ==========================================

    Presence is tested with ``in``, never ``payload.get(...) is not None``: those
    two collapse the first two rows together, and only the first means "this file
    predates the migration".

    Why a *set* comparison and not the width.  ``_build_global_features`` orders the
    global feature block by this key set, so swapping metric A for metric B leaves
    ``global_feature_dim`` unchanged while the critic reads different semantics under
    the same indices.  ``ensure_initialized``'s existing width check cannot see that;
    this is the check that can.
    """
    if "global_metric_keys" not in payload:
        warnings.warn(
            f"{path}: pre-C8 MAPPO checkpoint -- it does not record the global "
            "metric key set it was trained with, so this env's set cannot be "
            "checked and a same-width metric swap would be silent (contract C8). "
            "Migrate it: python -m offline.migrate_mappo_checkpoints --apply",
            RuntimeWarning,
            stacklevel=3,
        )
        return

    checkpoint_keys = payload["global_metric_keys"]
    if checkpoint_keys is None:
        warnings.warn(
            f"{path}: this MAPPO checkpoint records global_metric_keys=None, i.e. "
            "it was saved before the agent ever built a global feature vector. "
            "There is nothing to check against this env's metric set (contract C8).",
            RuntimeWarning,
            stacklevel=3,
        )
        return

    env_keys = env_global_metric_keys(env)
    if env_keys is None:
        warnings.warn(
            f"{path}: this env exposes no metrics pipeline, so the checkpoint's "
            f"global metric set {sorted(checkpoint_keys)} cannot be verified "
            "against it (contract C8).",
            RuntimeWarning,
            stacklevel=3,
        )
        return

    checkpoint_set = {str(k) for k in checkpoint_keys}
    env_set = set(env_keys)
    if checkpoint_set == env_set:
        return

    raise ValueError(
        f"{path}: MAPPO checkpoint global metric set does not match this env "
        f"(contract C8).\n"
        f"  checkpoint          : {sorted(checkpoint_set)}\n"
        f"  env                 : {sorted(env_set)}\n"
        f"  symmetric difference: {sorted(checkpoint_set ^ env_set)}\n"
        "The centralised critic indexes its global feature block by this ordered "
        "key set, so a same-width swap reads different semantics under the same "
        "indices and the global_feature_dim check cannot detect it. Restore the "
        "env's metric set to the one above, or retrain."
    )


class IMAPPOAgent(BaseAgent):
    """
    Multi-agent PPO for traffic signal control.

    Actors use local intersection state. The centralized critic uses the
    concatenated state of all controlled intersections *plus* a shared global
    feature block (episode progress, network-wide metrics) drawn from ``info``,
    and is trained with running input/value normalization for stability.
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
        action_counts = Utils.infer_action_counts(
            self.env.action_space,
            self.intersections,
        )

        self.device = Utils.resolve_device(device)
        Utils.seed_everything(seed, seed_python_random=False)

        self.learner = _CentralizedMAPPO(
            action_counts=action_counts,
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
        # Ordered global-metric keys, frozen on first use so the global feature
        # block keeps a stable width across the run.
        self._global_metric_keys: list[str] | None = None

        self._prev_local_states: list[np.ndarray] = []
        self._prev_joint_state: np.ndarray | None = None
        self._prev_actions: np.ndarray | None = None
        self._prev_log_probs: np.ndarray | None = None
        self._prev_values: np.ndarray | None = None
        self._prev_action_masks: list[np.ndarray] = []

    def get_info(self) -> dict[str, Any]:
        return self._last_info

    def _build_global_features(self, info: dict[str, Any]) -> np.ndarray:
        """Shared global state for the centralized critic.

        Combines episode progress and network-wide signals that the
        per-intersection actor states cannot easily reconstruct: the fraction
        of the episode elapsed, the live vehicle count, and every global metric
        the env exposes in ``info["metrics"]``. The metric key set is frozen on
        first call so the feature width stays constant.
        """
        metrics = info.get("metrics") or {}
        if self._global_metric_keys is None:
            self._global_metric_keys = sorted(metrics.keys())

        max_steps = max(1, int(getattr(self.env, "max_steps", 1)))
        feats = [
            float(info.get("step", 0)) / max_steps,
            float(info.get("vehicle_count", 0.0)),
        ]
        feats.extend(float(metrics.get(key, 0.0)) for key in self._global_metric_keys)
        return np.asarray(feats, dtype=np.float32)

    def get_reward(self) -> float:
        return self._last_reward

    def get_avail_action(self, info: dict[str, Any] | None = None) -> dict[str, list[int]]:
        if info is None:
            return {
                ix_id: list(range(n_actions))
                for ix_id, n_actions in zip(
                    self.intersection_ids,
                    self.learner.action_counts,
                )
            }

        per_ix = Utils.extract_per_intersection_info(info, self.intersection_ids)
        return {
            ix_id: Utils.extract_valid_actions(per_ix[ix_id], n_actions)
            for ix_id, n_actions in zip(
                self.intersection_ids,
                self.learner.action_counts,
            )
        }

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
        local_states = [
            Utils.state_from_info(per_ix[ix_id]) for ix_id in self.intersection_ids
        ]
        valid_actions = [
            Utils.extract_valid_actions(per_ix[ix_id], n_actions)
            for ix_id, n_actions in zip(self.intersection_ids, self.learner.action_counts)
        ]
        global_features = self._build_global_features(info)

        (
            actions,
            log_probs,
            values,
            action_masks,
            joint_state,
        ) = self.learner.select_actions(
            local_states,
            valid_actions,
            global_features,
            deterministic=not explore,
        )

        self._last_info = info
        if update_memory:
            self._prev_local_states = local_states
            self._prev_joint_state = joint_state
            self._prev_actions = actions
            self._prev_log_probs = log_probs
            self._prev_values = values
            self._prev_action_masks = action_masks
        return actions

    def observe(
        self,
        next_info: dict[str, Any],
        reward: Any,
        terminated: bool,
        truncated: bool = False,
    ) -> dict[str, float]:
        self._last_info = next_info
        self._last_reward = Utils.scalar_reward(reward)

        if self._prev_joint_state is None:
            return {}

        episode_boundary = bool(terminated) or bool(truncated)
        per_ix = Utils.extract_per_intersection_info(next_info, self.intersection_ids)
        next_local_states = [
            Utils.state_from_info(per_ix[ix_id]) for ix_id in self.intersection_ids
        ]
        next_joint_state = self.learner.build_joint_state(
            next_local_states,
            self._build_global_features(next_info),
        )
        local_rewards = np.asarray(
            [
                Utils.reward_for_intersection(ix_id, reward, per_ix[ix_id])
                for ix_id in self.intersection_ids
            ],
            dtype=np.float32,
        )

        if (
            self._prev_actions is None
            or self._prev_log_probs is None
            or self._prev_values is None
        ):
            raise RuntimeError("MAPPO previous action state is incomplete.")

        self.learner.store_transition(
            local_states=self._prev_local_states,
            joint_state=self._prev_joint_state,
            actions=self._prev_actions,
            log_probs=self._prev_log_probs,
            values=self._prev_values,
            rewards=local_rewards,
            terminated=bool(terminated),
            action_masks=self._prev_action_masks,
        )

        metrics: dict[str, float] = {}
        if self.learner.ready_to_update(episode_boundary):
            # True termination zeroes the bootstrap; truncation keeps it.
            last_values = (
                np.zeros(len(self.intersection_ids), dtype=np.float32)
                if terminated
                else self.learner.estimate_values(next_joint_state)
            )
            metrics = self.learner.update(last_values) or {}

        self.steps_done += 1
        if episode_boundary:
            self._clear_previous_step()

        return metrics

    def save(self, path: str) -> None:
        torch.save(
            {
                "steps_done": self.steps_done,
                "learner": self.learner.export_state(),
                # Contract C8: the critic's global feature block is ordered by this
                # key set, so the checkpoint is only interpretable alongside it.
                # None means the agent never built a global feature vector.
                "global_metric_keys": (
                    None
                    if self._global_metric_keys is None
                    else list(self._global_metric_keys)
                ),
            },
            path,
        )

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        # Checked BEFORE any state is adopted, so a rejected checkpoint leaves this
        # agent exactly as it was -- the filesystem-mutation barrier, applied to
        # in-memory state.
        _assert_checkpoint_metric_keys(payload, self.env, path)
        # Deliberately NOT self._global_metric_keys = payload["global_metric_keys"].
        # _build_global_features reads metrics.get(key, 0.0), so keys adopted from a
        # checkpoint against a differing env would substitute a silent 0.0 at an
        # unchanged width -- invisible to the width guard AND to the check above,
        # which has already returned by then. Leaving the field None makes the agent
        # freeze from the env's own info, which is where the check can still fire.
        self.steps_done = int(payload.get("steps_done", 0))
        learner_state = payload.get("learner")
        if learner_state is not None:
            self.learner.load_state(learner_state)

    def _clear_previous_step(self) -> None:
        self._prev_local_states = []
        self._prev_joint_state = None
        self._prev_actions = None
        self._prev_log_probs = None
        self._prev_values = None
        self._prev_action_masks = []


class MAPPOAgent(IMAPPOAgent):
    """Alias for centralized multi-agent PPO."""

    pass
