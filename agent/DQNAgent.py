from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import BaseAgent
from .utils.utils import Utils

'''
assumes:
info = {
    "intersections": {
        "ix_0": {
            "state": [ ... numeric vector ... ],
            "reward": -1.2,              # optional
            "avail_actions": [0, 2],     # optional
        },
        "ix_1": {
            "state": [ ... ],
        },
    }
}
'''


'''usage:
agent = IDQNAgent(env)

info = env.reset()
for _ in range(10000):
    action = agent.next_action(info)
    reward, terminated, truncated, next_info = env.step(action)
    agent.observe(next_info, reward, terminated, truncated)
    info = next_info
    if terminated or truncated:
        info = env.reset()
'''

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _IntersectionDQN:
    def __init__(
        self,
        n_actions: int,
        lr: float,
        gamma: float,
        batch_size: int,
        replay_size: int,
        min_replay_size: int,
        target_update_interval: int,
        hidden_dim: int,
        device: torch.device,
    ) -> None:
        self.n_actions = int(n_actions)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.min_replay_size = int(min_replay_size)
        self.target_update_interval = int(target_update_interval)
        self.hidden_dim = int(hidden_dim)
        self.device = device

        self.state_dim: int | None = None
        self.q_net: QNetwork | None = None
        self.target_net: QNetwork | None = None
        self.optimizer: optim.Optimizer | None = None

        # Ring buffer of (state, action, reward, next_state, terminated_float,
        # next_action_mask); random.sample on a list is O(batch) vs O(n) on a
        # deque. Only true terminations zero the bootstrap; truncations keep
        # it. next_action_mask marks the actions valid in next_state so the
        # bootstrap max never picks an illegal action.
        self.buffer: list[tuple[np.ndarray, int, float, np.ndarray, float, np.ndarray]] = []
        self._buffer_capacity = max(1, int(replay_size))
        self._next_idx = 0
        self.train_updates = 0

    def ensure_initialized(self, state_dim: int) -> None:
        state_dim = int(state_dim)
        if self.state_dim is None:
            self.state_dim = state_dim
            self.q_net = QNetwork(state_dim, self.n_actions, self.hidden_dim).to(self.device)
            self.target_net = QNetwork(state_dim, self.n_actions, self.hidden_dim).to(self.device)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
            return
        if state_dim != self.state_dim:
            raise ValueError(
                f"State size changed for local DQN: expected {self.state_dim}, got {state_dim}"
            )

    def select_action(self, state: np.ndarray, epsilon: float, valid_actions: list[int]) -> int:
        if self.q_net is None:
            raise RuntimeError("Call ensure_initialized before select_action.")
        if not valid_actions:
            valid_actions = list(range(self.n_actions))

        if random.random() < epsilon:
            return int(random.choice(valid_actions))

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t).squeeze(0).detach().cpu().numpy()

        best_action = max(valid_actions, key=lambda a: float(q_values[a]))
        return int(best_action)

    def q_values_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Q-values for *state* left on-device (no host sync).

        The agent enqueues this for every greedy intersection and reads them
        all back with a single ``.cpu()`` per step.
        """
        if self.q_net is None:
            raise RuntimeError("Call ensure_initialized before q_values_tensor.")
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_net(state_t).squeeze(0)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        next_valid_actions: list[int] | None = None,
    ) -> None:
        next_mask = self._build_action_mask(next_valid_actions)
        item = (state.copy(), int(action), float(reward), next_state.copy(), float(terminated), next_mask)
        if len(self.buffer) < self._buffer_capacity:
            self.buffer.append(item)
        else:
            self.buffer[self._next_idx] = item
        self._next_idx = (self._next_idx + 1) % self._buffer_capacity

    def _build_action_mask(self, valid_actions: list[int] | None) -> np.ndarray:
        if valid_actions is None:
            return np.ones(self.n_actions, dtype=np.bool_)
        mask = np.zeros(self.n_actions, dtype=np.bool_)
        for action in valid_actions:
            if 0 <= action < self.n_actions:
                mask[action] = True
        if not mask.any():
            mask[:] = True
        return mask

    def _update_grad_step(self) -> torch.Tensor | None:
        """Run one gradient step; return the detached loss tensor (no host sync).

        Keeping the loss on-device lets the agent read every intersection's
        loss back with a single ``.cpu()`` per step instead of one ``.item()``
        per learner (which serialises the GPU each call).
        """
        if self.q_net is None or self.target_net is None or self.optimizer is None:
            return None
        if len(self.buffer) < max(self.batch_size, self.min_replay_size):
            return None

        batch = random.sample(self.buffer, self.batch_size)
        states = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([b[1] for b in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        terminateds = torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(np.stack([b[5] for b in batch]), dtype=torch.bool, device=self.device)

        current_q = self.q_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_all = self.target_net(next_states)
            # The mask builder guarantees at least one valid action per row,
            # so the max is always a real Q-value; the finite floor (vs -inf)
            # only guards against 0 * inf = NaN if that invariant ever broke.
            floor = torch.finfo(next_q_all.dtype).min
            next_q = next_q_all.masked_fill(~next_masks, floor).max(dim=1).values
            target_q = rewards + (1.0 - terminateds) * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_updates += 1
        if self.train_updates % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.detach()

    def update(self) -> float | None:
        """Run one gradient step and return the loss as a Python float.

        Thin synchronising wrapper around :meth:`_update_grad_step` (it calls
        ``.item()``). The training hot path uses ``_update_grad_step`` directly
        and batches the host readback; this stays for tests / external callers.
        """
        loss = self._update_grad_step()
        return None if loss is None else float(loss.item())


class IDQNAgent(BaseAgent):
    """
    Independent DQN agent for traffic signal control.

    One local DQN learner is created per intersection.
    State comes from the info dict, not from observation.
    """

    def __init__(
        self,
        gym_env: Any,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        replay_size: int = 100_000,
        min_replay_size: int = 1_000,
        target_update_interval: int = 200,
        hidden_dim: int = 128,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
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

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.steps_done = 0

        Utils.seed_everything(seed)

        self.learners: dict[str, _IntersectionDQN] = {}
        for i, ix in enumerate(self.intersections):
            n_actions = action_counts[i]
            self.learners[ix.id] = _IntersectionDQN(
                n_actions=n_actions,
                lr=lr,
                gamma=gamma,
                batch_size=batch_size,
                replay_size=replay_size,
                min_replay_size=min_replay_size,
                target_update_interval=target_update_interval,
                hidden_dim=hidden_dim,
                device=self.device,
            )

        self._last_info: dict[str, Any] = {}
        self._last_reward: float = 0.0
        self._prev_states: dict[str, np.ndarray] = {}
        self._prev_actions: dict[str, int] = {}

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
            out[ix.id] = Utils.extract_valid_actions(
                per_ix[ix.id], self.learners[ix.id].n_actions
            )
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
        # update_memory=False, or the next observe() would pair the eval
        # state/action with a training reward.
        per_ix = Utils.extract_per_intersection_info(info, self.intersection_ids)
        epsilon = self._epsilon() if explore else 0.0

        actions: list[int] = [0] * len(self.intersections)
        states: dict[str, np.ndarray] = {}
        # Greedy intersections: enqueue the Q-forward on-device now and read
        # them all back with a single host sync after the loop (one .cpu() per
        # step instead of one per intersection). Per-intersection .cpu() calls
        # serialise the GPU and let a co-resident simulator's async work leak
        # into agent time. RNG order is unchanged: random.random()/choice() are
        # still called per intersection in the same order as before.
        pending: list[tuple[int, list[int], torch.Tensor]] = []

        for idx, ix in enumerate(self.intersections):
            ix_id = ix.id
            ix_payload = per_ix[ix_id]
            state = Utils.state_from_info(ix_payload)
            states[ix_id] = state

            learner = self.learners[ix_id]
            learner.ensure_initialized(state.shape[0])

            valid_actions = Utils.extract_valid_actions(ix_payload, learner.n_actions)
            if not valid_actions:
                valid_actions = list(range(learner.n_actions))

            if random.random() < epsilon:
                actions[idx] = int(random.choice(valid_actions))
            else:
                pending.append((idx, valid_actions, learner.q_values_tensor(state)))

        if pending:
            flat = torch.cat([q for _, _, q in pending]).detach().cpu().numpy()
            offset = 0
            for idx, valid_actions, q in pending:
                n = q.shape[0]
                q_values = flat[offset:offset + n]
                offset += n
                actions[idx] = int(
                    max(valid_actions, key=lambda a, qv=q_values: float(qv[a]))
                )

        self._last_info = info
        if update_memory:
            self._prev_states = states
            self._prev_actions = dict(zip(self.intersection_ids, actions))
        return np.asarray(actions, dtype=np.int64)

    def observe(
        self,
        next_info: dict[str, Any],
        reward: Any,
        terminated: bool,
        truncated: bool = False,
    ) -> dict[str, float]:
        self._last_info = next_info
        self._last_reward = Utils.scalar_reward(reward)

        if not self._prev_states:
            return {}

        per_ix = Utils.extract_per_intersection_info(next_info, self.intersection_ids)
        # Push every transition and run each learner's gradient step, but keep
        # the losses on-device; read them all back with a single .cpu() after
        # the loop (one sync/step instead of one .item() per intersection).
        pending_losses: list[tuple[str, torch.Tensor]] = []

        for ix in self.intersections:
            ix_id = ix.id
            ix_payload = per_ix[ix_id]
            next_state = Utils.state_from_info(ix_payload)
            local_reward = Utils.reward_for_intersection(ix_id, reward, ix_payload)

            learner = self.learners[ix_id]
            next_valid_actions = Utils.extract_valid_actions(ix_payload, learner.n_actions)
            learner.push(
                self._prev_states[ix_id],
                self._prev_actions[ix_id],
                local_reward,
                next_state,
                terminated,
                next_valid_actions,
            )
            loss_t = learner._update_grad_step()
            if loss_t is not None:
                pending_losses.append((ix_id, loss_t))

        losses: dict[str, float] = {}
        if pending_losses:
            vals = torch.stack([t for _, t in pending_losses]).cpu().tolist()
            losses = {
                ix_id: float(v)
                for (ix_id, _), v in zip(pending_losses, vals)
            }

        self.steps_done += 1
        if terminated or truncated:
            self._prev_states = {}
            self._prev_actions = {}

        return losses

    def save(self, path: str) -> None:
        payload: dict[str, Any] = {
            "steps_done": self.steps_done,
            "learners": {},
        }
        for ix_id, learner in self.learners.items():
            if learner.q_net is None or learner.target_net is None or learner.optimizer is None:
                continue
            payload["learners"][ix_id] = {
                "state_dim": learner.state_dim,
                "q_net": learner.q_net.state_dict(),
                "target_net": learner.target_net.state_dict(),
                "optimizer": learner.optimizer.state_dict(),
            }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.steps_done = int(payload.get("steps_done", 0))

        for ix_id, data in payload.get("learners", {}).items():
            if ix_id not in self.learners:
                continue
            learner = self.learners[ix_id]
            learner.ensure_initialized(int(data["state_dim"]))
            assert learner.q_net is not None and learner.target_net is not None and learner.optimizer is not None
            learner.q_net.load_state_dict(data["q_net"])
            learner.target_net.load_state_dict(data["target_net"])
            learner.optimizer.load_state_dict(data["optimizer"])

    def _epsilon(self) -> float:
        if self.epsilon_decay_steps <= 0:
            return self.epsilon_end
        progress = min(1.0, self.steps_done / float(self.epsilon_decay_steps))
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)


class DQNAgent(IDQNAgent):
    pass
