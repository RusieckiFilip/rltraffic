"""Offline baselines for P4.4: behaviour cloning and independent per-intersection IQL.

SKELETON -- signatures only.  Every callable raises ``NotImplementedError`` so the red-first
run reaches the real API surface instead of one shared ``ModuleNotFoundError``.

Checkpoint format versions: ``bc-checkpoint/1.0`` and ``iql-checkpoint/1.0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from offline.dataset import NormalizationStats

from .base import BaseAgent
from .DTAgent import action_loss, masked_action_logits

__all__ = [
    "action_loss",
    "masked_action_logits",
    "BC_CHECKPOINT_FORMAT_VERSION",
    "IQL_CHECKPOINT_FORMAT_VERSION",
    "BCAgent",
    "IQLAgent",
    "MLPTrunk",
    "TrunkConfig",
    "canonical_state_dict_digest",
]

BC_CHECKPOINT_FORMAT_VERSION = "bc-checkpoint/1.0"
IQL_CHECKPOINT_FORMAT_VERSION = "iql-checkpoint/1.0"


@dataclass(frozen=True)
class TrunkConfig:
    """Architecture of one state-conditioned network."""

    state_dim: int
    n_actions: int
    d_model: int = 128
    n_layer: int = 3
    dropout: float = 0.1

    def to_json_obj(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_json_obj(cls, payload: dict[str, Any]) -> TrunkConfig:
        raise NotImplementedError


def canonical_state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    """sha256 over the tensor bytes in sorted key order (``DEFERRED`` 29)."""
    raise NotImplementedError


class MLPTrunk(nn.Module):
    """The DT's per-position feed-forward stack with attention removed."""

    def __init__(self, config: TrunkConfig, out_dim: int) -> None:
        raise NotImplementedError

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BCAgent(BaseAgent):
    """Behaviour cloning over the current state only.  Serves BC and %BC alike."""

    def __init__(
        self,
        gym_env: Any,
        d_model: int = 128,
        n_layer: int = 3,
        dropout: float = 0.1,
        stats: NormalizationStats | None = None,
        scenario_id: str | None = None,
        method: str = "bc",
        device: str | None = None,
        seed: int | None = None,
        state_dim: int | None = None,
    ) -> None:
        raise NotImplementedError

    @property
    def config(self) -> TrunkConfig:
        raise NotImplementedError

    def policy_logits(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def act(
        self, info: dict[str, Any], explore: bool = True, update_memory: bool = True
    ) -> np.ndarray:
        raise NotImplementedError

    def observe(
        self,
        next_info: dict[str, Any],
        reward: Any,
        terminated: bool,
        truncated: bool = False,
    ) -> dict[str, float]:
        raise NotImplementedError

    def canonical_digest(self) -> str:
        raise NotImplementedError

    def save(self, path: str, provenance: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def from_checkpoint(
        cls, gym_env: Any, path: str, device: str | None = None
    ) -> BCAgent:
        raise NotImplementedError


class IQLAgent(BaseAgent):
    """Independent per-intersection IQL: expectile V, Q, and an AWR-extracted policy."""

    def __init__(
        self,
        gym_env: Any,
        d_model: int = 128,
        n_layer: int = 3,
        dropout: float = 0.1,
        stats: NormalizationStats | None = None,
        scenario_id: str | None = None,
        device: str | None = None,
        seed: int | None = None,
        state_dim: int | None = None,
    ) -> None:
        raise NotImplementedError

    @property
    def config(self) -> TrunkConfig:
        raise NotImplementedError

    def policy_logits(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def q_values(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def state_values(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def act(
        self, info: dict[str, Any], explore: bool = True, update_memory: bool = True
    ) -> np.ndarray:
        raise NotImplementedError

    def observe(
        self,
        next_info: dict[str, Any],
        reward: Any,
        terminated: bool,
        truncated: bool = False,
    ) -> dict[str, float]:
        raise NotImplementedError

    def canonical_digest(self) -> str:
        raise NotImplementedError

    def save(self, path: str, provenance: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def from_checkpoint(
        cls, gym_env: Any, path: str, device: str | None = None
    ) -> IQLAgent:
        raise NotImplementedError
