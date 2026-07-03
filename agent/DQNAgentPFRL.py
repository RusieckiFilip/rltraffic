"""RESCO-parity independent DQN built on the pfrl library.

This is the same learner RESCO (resco-for-malaysia) uses for its IDQN:
one ``pfrl.agents.DQN`` per intersection with RESCO's exact
hyperparameters and network (``Conv2d(1, 64, 2x2)`` over the
``drq_norm`` observation, two 64-unit fully-connected layers, and a
``DiscreteActionValueHead``).

Requires the ``pfrl`` package (``pip install pfrl``) and an env
configured with ``state_features=["drq_norm"]`` (available on the SUMO
and MOSS backends).

Differences from :class:`agent.DQNAgent.IDQNAgent`:

- pfrl's DQN has no action masking, so this agent only supports phase
  controls whose actions are always all available (``RescoCyclicPhases``).
- Exploration/learning are coupled like in pfrl: ``act(explore=True)``
  must be followed by ``observe()`` (training mode), while
  ``act(explore=False)`` is greedy and stores nothing (eval mode).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .base import BaseAgent
from .utils.utils import Utils

try:
    from pfrl import explorers, replay_buffers
    from pfrl.agents import DQN
    from pfrl.q_functions import DiscreteActionValueHead

    _PFRL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised without pfrl
    _PFRL_IMPORT_ERROR = exc


def _conv2d_size_out(size: int, kernel_size: int = 2, stride: int = 1) -> int:
    """Output size of a conv dimension (RESCO's calculate_output_size)."""
    return (size - (kernel_size - 1) - 1) // stride + 1


class RescoDQNNetwork(nn.Module):
    """1:1 port of RESCO's ``DQNDefaultModel`` (ReLU activation)."""

    def __init__(self, obs_shape: tuple[int, ...], n_actions: int) -> None:
        super().__init__()
        if len(obs_shape) != 3:
            raise ValueError(
                "RescoDQNNetwork expects a (channels, n_lanes, n_features) "
                f"observation, got shape {obs_shape}. Use the env's "
                "state_features=['drq_norm']."
            )
        channels, h, w = (int(d) for d in obs_shape)
        conv_h = _conv2d_size_out(h)
        conv_w = _conv2d_size_out(w)
        if conv_h <= 0 or conv_w <= 0:
            raise ValueError(
                f"Observation {obs_shape} is too small for the 2x2 conv "
                "(needs at least 2 lanes and 2 features)."
            )
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_h * conv_w * 64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)
        self.discrete_action_value_head = DiscreteActionValueHead()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Any:
        x = self.activation(self.conv1(x))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return self.discrete_action_value_head(x)


class RescoIDQNAgent(BaseAgent):
    """Independent DQN agent mirroring RESCO's pfrl-based IDQN.

    Defaults are RESCO's IDQN config: ``ReplayBuffer(10000)``,
    ``minibatch_size=32``, ``replay_start_size=32`` (= batch size),
    ``gamma=0.99``, ``target_update_interval=500``,
    ``LinearDecayEpsilonGreedy(1.0 -> 0.0)`` and Adam with pfrl/torch's
    default ``lr=1e-3``, updating every step.
    """

    def __init__(
        self,
        gym_env: Any,
        *,
        batch_size: int = 32,
        gamma: float = 0.99,
        replay_size: int = 10_000,
        replay_start_size: int | None = None,
        target_update_interval: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.0,
        epsilon_decay_steps: int = 50_000,
        lr: float = 1e-3,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        if _PFRL_IMPORT_ERROR is not None:
            raise ImportError(
                "RescoIDQNAgent requires pfrl. Install it with "
                "`pip install pfrl`."
            ) from _PFRL_IMPORT_ERROR

        super().__init__(gym_env)
        self.env = gym_env

        self.intersections = list(self.env.intersections)
        if not self.intersections:
            raise ValueError(
                "No controllable intersections found in environment."
            )
        self.intersection_ids = [ix.id for ix in self.intersections]
        action_counts = Utils.infer_action_counts(
            self.env.action_space, self.intersections
        )
        self._n_actions: dict[str, int] = {
            ix.id: int(n)
            for ix, n in zip(self.intersections, action_counts)
        }

        self.device = Utils.resolve_device(device)
        self._gpu = (
            self.device.index if self.device.type == "cuda" else None
        )

        self._batch_size = int(batch_size)
        self._gamma = float(gamma)
        self._replay_size = int(replay_size)
        self._replay_start_size = int(
            replay_start_size if replay_start_size is not None else batch_size
        )
        self._target_update_interval = int(target_update_interval)
        self._epsilon_start = float(epsilon_start)
        self._epsilon_end = float(epsilon_end)
        self._epsilon_decay_steps = int(epsilon_decay_steps)
        self._lr = float(lr)

        Utils.seed_everything(seed)

        # Lazily built on first act() (obs shapes come from the info dict).
        self._agents: dict[str, Any] = {}
        self._models: dict[str, RescoDQNNetwork] = {}
        self._optimizers: dict[str, torch.optim.Optimizer] = {}
        self._obs_shapes: dict[str, tuple[int, ...]] = {}

        self._last_info: dict[str, Any] = {}
        self._last_reward: float = 0.0

    # ------------------------------------------------------------------
    # BaseAgent API
    # ------------------------------------------------------------------

    def get_info(self) -> dict[str, Any]:
        return self._last_info

    def get_reward(self) -> float:
        return self._last_reward

    def get_avail_action(
        self, info: dict[str, Any] | None = None
    ) -> dict[str, list[int]]:
        if info is None:
            return {
                ix_id: list(range(self._n_actions[ix_id]))
                for ix_id in self.intersection_ids
            }
        per_ix = Utils.extract_per_intersection_info(
            info, self.intersection_ids
        )
        return {
            ix_id: Utils.extract_valid_actions(
                per_ix[ix_id], self._n_actions[ix_id]
            )
            for ix_id in self.intersection_ids
        }

    def next_action(self, ob: dict[str, Any]) -> np.ndarray:
        return self.act(ob, explore=True)

    def act(
        self,
        info: dict[str, Any],
        explore: bool = True,
        update_memory: bool | None = None,
    ) -> np.ndarray:
        """Choose one action per intersection.

        ``explore=True`` runs the pfrl agents in training mode
        (epsilon-greedy, transitions recorded on the next
        :meth:`observe`); ``explore=False`` runs them in eval mode
        (greedy, nothing recorded). pfrl couples exploration with
        memory, so ``update_memory`` must match ``explore`` when given.
        """
        if update_memory is not None and bool(update_memory) != bool(explore):
            raise ValueError(
                "pfrl couples exploration with replay memory: use "
                "explore=True with update_memory=True (training) or "
                "explore=False with update_memory=False (evaluation)."
            )

        per_ix = Utils.extract_per_intersection_info(
            info, self.intersection_ids
        )
        actions: list[int] = []
        for ix_id in self.intersection_ids:
            payload = per_ix[ix_id]
            self._check_mask(ix_id, payload)
            state = self._state_from_payload(ix_id, payload)
            agent = self._ensure_agent(ix_id, state.shape)
            agent.training = bool(explore)
            actions.append(int(agent.act(state)))

        self._last_info = info
        return np.asarray(actions, dtype=np.int64)

    def observe(
        self,
        next_info: dict[str, Any],
        reward: Any,
        terminated: bool,
        truncated: bool = False,
    ) -> dict[str, float]:
        """Feed the post-step state/reward into every pfrl learner.

        Maps Gymnasium's ``(terminated, truncated)`` onto pfrl's
        ``(done, reset)``: only true terminations zero the bootstrap,
        truncations end the episode while keeping it.
        """
        self._last_info = next_info
        self._last_reward = Utils.scalar_reward(reward)

        per_ix = Utils.extract_per_intersection_info(
            next_info, self.intersection_ids
        )
        losses: dict[str, float] = {}
        for ix_id in self.intersection_ids:
            agent = self._agents.get(ix_id)
            if agent is None or not agent.training:
                continue
            payload = per_ix[ix_id]
            next_state = self._state_from_payload(ix_id, payload)
            local_reward = Utils.reward_for_intersection(
                ix_id, reward, payload
            )
            agent.observe(
                next_state,
                float(local_reward),
                bool(terminated),
                bool(truncated),
            )
            stats = dict(agent.get_statistics())
            loss = stats.get("average_loss")
            if loss is not None:
                losses[ix_id] = float(loss)
        return losses

    def save(self, path: str) -> None:
        payload: dict[str, Any] = {"learners": {}}
        for ix_id, agent in self._agents.items():
            payload["learners"][ix_id] = {
                "obs_shape": self._obs_shapes[ix_id],
                "model": self._models[ix_id].state_dict(),
                "optimizer": self._optimizers[ix_id].state_dict(),
                "t": int(agent.t),
            }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        for ix_id, data in payload.get("learners", {}).items():
            if ix_id not in self._n_actions:
                continue
            agent = self._ensure_agent(ix_id, tuple(data["obs_shape"]))
            self._models[ix_id].load_state_dict(data["model"])
            self._optimizers[ix_id].load_state_dict(data["optimizer"])
            agent.t = int(data.get("t", 0))
            # pfrl syncs the target net on its own schedule; force one
            # so a freshly-loaded agent doesn't bootstrap from random
            # target weights.
            agent.sync_target_network()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_mask(self, ix_id: str, payload: dict[str, Any]) -> None:
        n_actions = self._n_actions[ix_id]
        valid = Utils.extract_valid_actions(payload, n_actions)
        if len(valid) != n_actions:
            raise ValueError(
                f"Intersection '{ix_id}' restricts actions to {valid} "
                f"(of {n_actions}), but pfrl's DQN does not support "
                "action masks. Use a phase control whose actions are "
                "always available (e.g. RescoCyclicPhases) or the "
                "custom IDQNAgent."
            )

    def _state_from_payload(
        self, ix_id: str, payload: dict[str, Any]
    ) -> np.ndarray:
        state = np.asarray(payload["state"], dtype=np.float32)
        if state.ndim != 3:
            raise ValueError(
                f"Intersection '{ix_id}' state has shape {state.shape}; "
                "RescoIDQNAgent expects the (1, n_lanes, n_features) "
                "drq_norm observation. Construct the env with "
                "state_features=['drq_norm']."
            )
        return state

    def _ensure_agent(self, ix_id: str, obs_shape: tuple[int, ...]) -> Any:
        agent = self._agents.get(ix_id)
        if agent is not None:
            if tuple(obs_shape) != self._obs_shapes[ix_id]:
                raise ValueError(
                    f"Observation shape changed for '{ix_id}': expected "
                    f"{self._obs_shapes[ix_id]}, got {tuple(obs_shape)}."
                )
            return agent

        model = RescoDQNNetwork(tuple(obs_shape), self._n_actions[ix_id])
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        n_actions = self._n_actions[ix_id]
        explorer = explorers.LinearDecayEpsilonGreedy(
            self._epsilon_start,
            self._epsilon_end,
            max(1, self._epsilon_decay_steps),
            lambda n=n_actions: np.random.randint(n),
        )
        agent = DQN(
            model,
            optimizer,
            replay_buffers.ReplayBuffer(self._replay_size),
            self._gamma,
            explorer,
            gpu=self._gpu,
            minibatch_size=self._batch_size,
            replay_start_size=self._replay_start_size,
            phi=lambda x: np.asarray(x, dtype=np.float32),
            target_update_interval=self._target_update_interval,
        )
        self._agents[ix_id] = agent
        self._models[ix_id] = model
        self._optimizers[ix_id] = optimizer
        self._obs_shapes[ix_id] = tuple(obs_shape)
        return agent
