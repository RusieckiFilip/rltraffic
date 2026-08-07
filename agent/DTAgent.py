"""Causal Decision Transformer over ``(RTG, state, action)`` tokens, one model per intersection.

SKELETON -- signatures only.  Every body raises ``NotImplementedError`` so the red tests reach
the real API surface instead of one shared import error.

Checkpoint format version: ``dt-checkpoint/1.0``.

Alignment convention (inherited from contract C6 and ``offline/dataset.py``; not re-derived)
--------------------------------------------------------------------------------------------
One decision step contributes three tokens, in this order::

    position 3t     RTG_t     the return-to-go BEFORE action t, i.e. sum(r_k for k >= t)
    position 3t+1   s_t       the state the agent saw BEFORE decision t
    position 3t+2   a_t       the action taken at step t

``a_t`` is predicted from the **state** token at position ``3t+1``.  Under causal masking that
position attends to ``RTG_t``, ``s_t`` and everything before them, and **never to ``a_t``**,
which sits one position later.  So the action slot of the current step is a placeholder at
inference time and cannot leak into its own prediction.

Padding is on the LEFT, matching the loader.  Two distinct roles for the same ``-1``:

* the loss **target** keeps ``PAD_ACTION = -1``, so ``ignore_index=-1`` is the explicit spelling
  and a forgotten mask crashes instead of training on fabricated action-0 targets;
* the embedding **input** cannot take ``-1`` (``nn.Embedding`` rejects it), so padded action
  inputs are remapped to a dedicated pad index ``n_actions``, and the embedding table has
  ``n_actions + 1`` rows.

Two NaN traps, both real and both guarded here
----------------------------------------------
1. A fully padded query position has no attendable key under causal + key-padding masking, and
   a softmax over an all-``-inf`` row is NaN.  NaN survives ``ignore_index`` into the backward
   pass as ``0 * NaN``.  **The diagonal is therefore always attendable.**
2. ``avail_mask`` is all-``False`` on padded rows, so masking logits there would build the same
   all-``-inf`` row.  **The availability mask is applied only to rows with at least one legal
   action.**

Action masking is kept although it never binds on the current corpus -- the P3 review scanned
32000/32000 streams and found every mask all-True (``docs/reviews/P3.md``) -- because the env
raises on an illegal action and the mask does bind under cyclic control modes and P6
perturbations.  No capability claim is made from it here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from offline.dataset import NormalizationStats, PAD_ACTION

from .base import BaseAgent
from .utils.utils import Utils

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "DTAgent",
    "DTConfig",
    "DecisionTransformer",
    "action_loss",
    "masked_action_logits",
]

CHECKPOINT_FORMAT_VERSION = "dt-checkpoint/1.0"


@dataclass(frozen=True)
class DTConfig:
    """Architecture of one Decision Transformer.  Frozen at construction and checkpointed."""

    state_dim: int
    n_actions: int
    context_length: int = 20
    n_layer: int = 3
    n_head: int = 1
    d_model: int = 128
    dropout: float = 0.1
    max_ep_len: int = 360

    def to_json_obj(self) -> dict[str, Any]:
        """JSON-ready mapping; every field round-trips exactly."""
        raise NotImplementedError

    @classmethod
    def from_json_obj(cls, payload: dict[str, Any]) -> DTConfig:
        """Rebuild a config written by :meth:`to_json_obj`."""
        raise NotImplementedError


def masked_action_logits(
    logits: torch.Tensor, avail_mask: torch.Tensor | None
) -> torch.Tensor:
    """Set logits at illegal actions to ``-inf``, leaving rows with no legal action untouched.

    A row of ``avail_mask`` that is entirely ``False`` is a *padded* row, not an intersection
    with no legal action: masking it would produce an all-``-inf`` row whose softmax is NaN,
    and that NaN reaches the backward pass even though ``ignore_index`` drops the position from
    the loss.  Such rows pass through unchanged and are excluded from the loss by their ``-1``
    target instead.
    """
    raise NotImplementedError


def action_loss(logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over action logits with ``ignore_index=PAD_ACTION``.

    ``PAD_ACTION`` is ``-1`` and marks exactly the padded positions (the loader guarantees
    ``action == -1`` iff ``attention_mask`` is ``False``).  Dropping ``ignore_index`` here does
    not silently train on fabricated targets -- it raises, which is the whole point of the
    loader choosing ``-1`` over ``0``.
    """
    raise NotImplementedError


class DecisionTransformer(nn.Module):
    """Causal transformer over interleaved ``(RTG, state, action)`` tokens."""

    def __init__(self, config: DTConfig) -> None:
        raise NotImplementedError

    def forward(
        self,
        rtg: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        avail_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return action logits ``(B, K, n_actions)``, one prediction per decision step.

        ``rtg`` is ``(B, K, 1)`` and already scaled by the caller; ``state`` is ``(B, K, D)``
        and already normalised; ``action`` is ``(B, K)`` int64 and may carry ``PAD_ACTION``;
        ``timestep`` is ``(B, K)`` int64; ``attention_mask`` is ``(B, K)`` bool marking the real
        steps; ``avail_mask`` is ``(B, K, n_actions)`` bool and, when given, is applied to the
        returned logits through :func:`masked_action_logits`.
        """
        raise NotImplementedError


class DTAgent(BaseAgent):
    """``BaseAgent``-compatible Decision Transformer controller.

    Constructor shape mirrors ``agent/MAPPOAgent.py`` (``gym_env`` first, ``device`` and
    ``seed`` last).  One model serves every controlled intersection; each intersection keeps its
    **own** context buffer, keyed by intersection id and never by position.  A heterogeneous
    intersection set is refused, naming both shapes -- C6 forbids padding across intersections.
    The intersections do not see each other: this is the independent per-intersection DT, not a
    spatial mixing layer (that is P5.1).

    Return-to-go is advanced in exactly one place, :meth:`act`, from
    ``info["intersections"][ix]["reward"]`` -- the same stream the corpus stores as
    ``local_reward``.  :meth:`observe` records the last info and reward for the ``BaseAgent``
    contract and deliberately does **not** touch the RTG; two routes would double-decrement it.
    """

    def __init__(
        self,
        gym_env: Any,
        context_length: int = 20,
        n_layer: int = 3,
        n_head: int = 1,
        d_model: int = 128,
        dropout: float = 0.1,
        max_ep_len: int = 360,
        target_rtg: float | None = None,
        rtg_scale: float | None = None,
        stats: NormalizationStats | None = None,
        scenario_id: str | None = None,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        raise NotImplementedError

    #: The underlying network.  Exposed because a checkpoint test has to be able to prove that
    #: a *refused* load changed no parameter, which is unobservable from the agent's outputs.
    model: DecisionTransformer

    @property
    def config(self) -> DTConfig:
        """The frozen architecture this agent was built with."""
        raise NotImplementedError

    def current_rtg(self) -> dict[str, float]:
        """The return-to-go each intersection is currently conditioning on, keyed by id."""
        raise NotImplementedError

    def reset_context(self) -> None:
        """Drop every intersection's context and restart the return-to-go at the target."""
        raise NotImplementedError

    def get_info(self) -> dict[str, Any]:
        """The last ``info`` this agent saw (``BaseAgent`` contract)."""
        raise NotImplementedError

    def get_reward(self) -> float:
        """The last scalar reward this agent was handed (``BaseAgent`` contract)."""
        raise NotImplementedError

    def get_avail_action(self, info: dict[str, Any] | None = None) -> dict[str, list[int]]:
        """Legal actions per intersection, read from *info* when given."""
        raise NotImplementedError

    def next_action(self, ob: dict[str, Any]) -> np.ndarray:
        """``BaseAgent`` alias for :meth:`act`."""
        raise NotImplementedError

    def act(
        self,
        info: dict[str, Any],
        explore: bool = True,
        update_memory: bool = True,
    ) -> np.ndarray:
        """One action per intersection, ordered by ``[ix.id for ix in env.intersections]``.

        A new episode is detected by ``info["step"] == 0`` and resets every context.  At
        ``step > 0`` the reward of the previous step is read from
        ``info["intersections"][ix]["reward"]`` and subtracted from the return-to-go; if that
        key is absent the call **raises** rather than freezing the RTG at its target, because a
        frozen RTG still produces plausible actions and a plausible number.

        ``explore=False`` takes the argmax over masked logits (the declared evaluation path);
        ``explore=True`` samples from the masked softmax.  ``update_memory=False`` leaves every
        buffer untouched, so an interleaved evaluation call cannot advance the context.
        """
        raise NotImplementedError

    def observe(
        self,
        next_info: dict[str, Any],
        reward: Any,
        terminated: bool,
        truncated: bool = False,
    ) -> dict[str, float]:
        """Record the last info and reward.  Does NOT advance the return-to-go -- see the class docstring."""
        raise NotImplementedError

    def save(self, path: str, provenance: dict[str, Any] | None = None) -> None:
        """Write weights, config, normalisation statistics, RTG constants and provenance."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Adopt a checkpoint, after checking it against this env.

        The shape check happens **before** any state is adopted, so a rejected checkpoint
        leaves this agent exactly as it was -- the in-memory form of the filesystem-mutation
        barrier, mirroring ``MAPPOAgent.load``'s C8 check.
        """
        raise NotImplementedError

    @classmethod
    def from_checkpoint(
        cls, gym_env: Any, path: str, device: str | None = None
    ) -> DTAgent:
        """Build an agent whose architecture comes from the checkpoint, then load it."""
        raise NotImplementedError
