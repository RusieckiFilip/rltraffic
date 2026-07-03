from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np
import torch
from gymnasium import spaces as gym_spaces


class Utils:
    @staticmethod
    def infer_action_counts(action_space: Any, intersections: list[Any]) -> list[int]:
        if isinstance(action_space, gym_spaces.Discrete):
            if len(intersections) != 1:
                raise ValueError(
                    "Discrete action space requires exactly one controlled intersection."
                )
            return [int(action_space.n)]

        if isinstance(action_space, gym_spaces.MultiDiscrete):
            nvec = np.asarray(action_space.nvec, dtype=np.int64).reshape(-1)
            if nvec.size != len(intersections):
                raise ValueError(
                    "MultiDiscrete action space size does not match intersections count."
                )
            return [int(v) for v in nvec.tolist()]

        return [int(ix.num_phases) for ix in intersections]

    @staticmethod
    def resolve_device(device: str | torch.device | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def seed_everything(seed: int | None, *, seed_python_random: bool = True) -> None:
        if seed is None:
            return
        if seed_python_random:
            random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def scalar_reward(reward: Any) -> float:
        if isinstance(reward, dict):
            if not reward:
                return 0.0
            values = [float(v) for v in reward.values()]
            return float(np.mean(values))

        if isinstance(reward, (bool, int, float, np.integer, np.floating, np.bool_)):
            return float(reward)

        arr = np.asarray(reward, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return 0.0
        return float(arr.mean())

    @staticmethod
    def reward_for_intersection(
        ix_id: str,
        reward: Any,
        ix_payload: dict[str, Any],
        scalar_reward_fn: Callable[[Any], float] | None = None,
    ) -> float:
        if isinstance(reward, dict) and ix_id in reward:
            return float(reward[ix_id])
        if "reward" in ix_payload and isinstance(ix_payload["reward"], (int, float, bool, np.integer, np.floating, np.bool_)):
            return float(ix_payload["reward"])
        if scalar_reward_fn is not None:
            return scalar_reward_fn(reward)
        return Utils.scalar_reward(reward)

    @staticmethod
    def extract_valid_actions(ix_payload: dict[str, Any], n_actions: int) -> list[int]:
        if "avail_actions" in ix_payload:
            raw = ix_payload["avail_actions"]
        elif "available_actions" in ix_payload:
            raw = ix_payload["available_actions"]
        else:
            raw = None

        if raw is None:
            return list(range(n_actions))

        if isinstance(raw, np.ndarray):
            raw = raw.tolist()

        if isinstance(raw, (list, tuple)):
            ids: list[int] = []
            for v in raw:
                try:
                    a = int(v)
                except (TypeError, ValueError):
                    continue
                if 0 <= a < n_actions:
                    ids.append(a)
            ids = sorted(set(ids))
            return ids if ids else list(range(n_actions))

        return list(range(n_actions))

    @staticmethod
    def extract_per_intersection_info(
        info: dict[str, Any],
        intersection_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        source: dict[str, Any] | None = None

        if isinstance(info.get("intersections"), dict):
            source = info["intersections"]
        elif all(isinstance(info.get(ix_id), dict) for ix_id in intersection_ids):
            source = info

        if source is None:
            raise KeyError(
                "Expected per-intersection info in either info['intersections'][ix_id] "
                "or info[ix_id]."
            )

        per_ix: dict[str, dict[str, Any]] = {}
        for ix_id in intersection_ids:
            if ix_id not in source:
                raise KeyError(f"Missing intersection '{ix_id}' in info payload.")
            item = source[ix_id]
            if not isinstance(item, dict):
                raise TypeError(f"info for intersection '{ix_id}' must be a dictionary.")
            per_ix[ix_id] = item
        return per_ix

    @staticmethod
    def state_from_info(ix_payload: dict[str, Any]) -> np.ndarray:
        state = np.asarray(ix_payload["state"], dtype=np.float32).reshape(-1)
        if state.size == 0:
            raise ValueError("Found empty 'state' vector in intersection info.")
        return state
