"""Causal Decision Transformer over ``(RTG, state, action)`` tokens, one model per intersection.

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
inference time and cannot leak into its own prediction -- asserted by
``tests/test_dt_agent.py::test_the_last_action_input_cannot_change_the_last_prediction``.

Padding is on the LEFT, matching the loader.  Two distinct roles for the same ``-1``:

* the loss **target** keeps ``PAD_ACTION = -1``, so ``ignore_index=-1`` is the explicit spelling
  and a forgotten mask crashes instead of training on fabricated action-0 targets;
* the embedding **input** cannot take ``-1`` (``nn.Embedding`` rejects it), so padded action
  inputs are remapped to a dedicated pad index ``n_actions``, and the embedding table has
  ``n_actions + 1`` rows.

Two NaN traps -- ONE of them real here, and the difference was measured, not assumed
------------------------------------------------------------------------------------
1. **Fully padded query rows: NOT a live trap on this implementation.** A fully padded query
   position has no attendable key under causal + key-padding masking, and a *manual* softmax
   over an all-``-inf`` row is NaN (measured: ``torch.softmax`` of such a row gives
   ``[nan, nan, nan]``), which would survive ``ignore_index`` into the backward pass as
   ``0 * NaN``.  **``F.scaled_dot_product_attention`` does not take that route**: measured on
   torch 2.11.0+cu128, a fully masked query row returns **zeros, with no NaN and no NaN
   gradient**, on CPU and CUDA and on every backend available here (MATH on both, FLASH on
   CPU, EFFICIENT on CUDA).  The always-attendable diagonal in :func:`_attention_bias` is
   therefore **defence in depth, not the thing standing between this model and a NaN** --
   deleting it leaves the whole suite green, which is stated here because a guard whose
   removal changes nothing must not be described as load-bearing.  It is kept so that "no
   query row is ever fully masked" holds by construction rather than by the current backend's
   behaviour, and so a later rewrite to manual attention cannot silently reintroduce the trap.
2. **Availability masking: a real trap, and the guard is load-bearing.** ``avail_mask`` is
   all-``False`` on padded rows, and masking those would build an all-``-inf`` logit row whose
   softmax *is* NaN, because that softmax is ours and not SDPA's.  **The availability mask is
   applied only to rows with at least one legal action** (:func:`masked_action_logits`).
   Removing that guard fails two tests, so this one is verified by mutation rather than by
   argument.

Action masking is kept although it never binds on the current corpus -- the P3 review scanned
32000/32000 streams and found every mask all-True (``docs/reviews/P3.md``) -- because the env
raises on an illegal action and the mask does bind under cyclic control modes and P6
perturbations.  No capability claim is made from it here.

Return-to-go at inference
-------------------------
``RTG_0`` is the declared target and ``RTG_{t+1} = RTG_t - r_t``, with ``r_t`` read from
``info["intersections"][ix]["reward"]`` of the info the env returned from step ``t`` -- the same
stream the corpus stores as ``local_reward`` (contract C6).  :meth:`DTAgent.act` is the **only**
place that advances it; :meth:`DTAgent.observe` deliberately does not, because two routes would
double-decrement it and the resulting number would still look plausible.

Lazy model construction mirrors ``agent/MAPPOAgent.py``: the state width is not knowable from an
env, only from an ``info``, so the network is built on first use from the observed width and
frozen against later changes.  The seed is re-applied at build time, so the weights depend on
``(seed, shapes)`` and not on when the build happened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

#: Tokens per decision step: (RTG, state, action).
TOKENS_PER_STEP = 3


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

    def __post_init__(self) -> None:
        if self.state_dim < 1 or self.n_actions < 1:
            raise ValueError(
                f"state_dim and n_actions must be >= 1, got {self.state_dim} and {self.n_actions}"
            )
        if self.context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {self.context_length}")
        if self.d_model % self.n_head:
            raise ValueError(
                f"d_model {self.d_model} is not divisible by n_head {self.n_head}"
            )
        if self.max_ep_len < 1:
            raise ValueError(f"max_ep_len must be >= 1, got {self.max_ep_len}")

    def to_json_obj(self) -> dict[str, Any]:
        """JSON-ready mapping; every field round-trips exactly."""
        return {
            "state_dim": int(self.state_dim),
            "n_actions": int(self.n_actions),
            "context_length": int(self.context_length),
            "n_layer": int(self.n_layer),
            "n_head": int(self.n_head),
            "d_model": int(self.d_model),
            "dropout": float(self.dropout),
            "max_ep_len": int(self.max_ep_len),
        }

    @classmethod
    def from_json_obj(cls, payload: dict[str, Any]) -> DTConfig:
        """Rebuild a config written by :meth:`to_json_obj`."""
        return cls(
            state_dim=int(payload["state_dim"]),
            n_actions=int(payload["n_actions"]),
            context_length=int(payload["context_length"]),
            n_layer=int(payload["n_layer"]),
            n_head=int(payload["n_head"]),
            d_model=int(payload["d_model"]),
            dropout=float(payload["dropout"]),
            max_ep_len=int(payload["max_ep_len"]),
        )


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
    if avail_mask is None:
        return logits
    mask = avail_mask.to(torch.bool)
    if mask.shape != logits.shape:
        raise ValueError(
            f"avail_mask shape {tuple(mask.shape)} does not match logits "
            f"{tuple(logits.shape)}"
        )
    any_legal = mask.any(dim=-1, keepdim=True)
    blocked = (~mask) & any_legal
    return logits.masked_fill(blocked, float("-inf"))


def action_loss(logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over action logits with ``ignore_index=PAD_ACTION``.

    ``PAD_ACTION`` is ``-1`` and marks exactly the padded positions (the loader guarantees
    ``action == -1`` iff ``attention_mask`` is ``False``).  Dropping ``ignore_index`` here does
    not silently train on fabricated action-0 targets -- it raises, which is the whole point of
    the loader choosing ``-1`` over ``0``.
    """
    n_actions = int(logits.shape[-1])
    return F.cross_entropy(
        logits.reshape(-1, n_actions),
        action.reshape(-1),
        ignore_index=PAD_ACTION,
    )


def _attention_bias(
    attention_mask: torch.Tensor | None,
    length: int,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive ``(B, 1, L, L)`` mask: causal, key-padded, and always self-attendable.

    The diagonal is forced open so that no query row is ever fully masked.  **Measured, not
    assumed: this changes nothing on torch 2.11's ``scaled_dot_product_attention``**, which
    returns zeros (not NaN) for a fully masked row on CPU and CUDA alike, so removing this line
    leaves the suite green.  It is kept as defence in depth for a manual-softmax rewrite, where
    the NaN is real -- see the module docstring, which states the measurement rather than
    claiming a guard that does not currently guard anything.
    """
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    allowed = causal.unsqueeze(0).expand(batch, length, length)
    if attention_mask is not None:
        # One step contributes TOKENS_PER_STEP consecutive tokens, so a padded step pads all
        # three of them.
        key_ok = attention_mask.to(torch.bool).repeat_interleave(TOKENS_PER_STEP, dim=1)
        allowed = allowed & key_ok.unsqueeze(1)
    allowed = allowed | torch.eye(length, dtype=torch.bool, device=device).unsqueeze(0)
    bias = torch.zeros(batch, 1, length, length, device=device, dtype=dtype)
    return bias.masked_fill(~allowed.unsqueeze(1), float("-inf"))


class _SelfAttention(nn.Module):
    """Multi-head self-attention with an explicit additive mask."""

    def __init__(self, d_model: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.n_head = int(n_head)
        self.d_head = int(d_model) // int(n_head)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        batch, length, d_model = x.shape
        query, key, value = self.qkv(x).split(d_model, dim=2)
        shape = (batch, length, self.n_head, self.d_head)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        merged = attended.transpose(1, 2).reshape(batch, length, d_model)
        return self.resid_drop(self.proj(merged))


class _Block(nn.Module):
    """Pre-LayerNorm transformer block."""

    def __init__(self, d_model: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = _SelfAttention(d_model, n_head, dropout)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_attn(x), bias)
        return x + self.mlp(self.ln_mlp(x))


class DecisionTransformer(nn.Module):
    """Causal transformer over interleaved ``(RTG, state, action)`` tokens."""

    def __init__(self, config: DTConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_rtg = nn.Linear(1, config.d_model)
        self.embed_state = nn.Linear(config.state_dim, config.d_model)
        # n_actions + 1 rows: the extra one is the pad index for PAD_ACTION inputs.
        self.embed_action = nn.Embedding(config.n_actions + 1, config.d_model)
        self.embed_timestep = nn.Embedding(config.max_ep_len, config.d_model)
        self.embed_ln = nn.LayerNorm(config.d_model)
        self.blocks = nn.ModuleList(
            [
                _Block(config.d_model, config.n_head, config.dropout)
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.n_actions)

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
        config = self.config
        batch, steps, state_dim = state.shape
        if state_dim != config.state_dim:
            raise ValueError(
                f"state width {state_dim} does not match the model's {config.state_dim}"
            )
        if tuple(rtg.shape) != (batch, steps, 1):
            raise ValueError(f"rtg must be (B, K, 1), got {tuple(rtg.shape)}")
        if tuple(action.shape) != (batch, steps) or tuple(timestep.shape) != (batch, steps):
            raise ValueError(
                f"action {tuple(action.shape)} and timestep {tuple(timestep.shape)} must both "
                f"be (B, K) = {(batch, steps)}"
            )
        if int(timestep.max()) >= config.max_ep_len:
            raise ValueError(
                f"timestep {int(timestep.max())} is outside the embedding's range "
                f"(max_ep_len {config.max_ep_len})"
            )
        if int(action.min()) < PAD_ACTION or int(action.max()) >= config.n_actions:
            raise ValueError(
                f"action values must lie in [{PAD_ACTION}, {config.n_actions - 1}], got "
                f"[{int(action.min())}, {int(action.max())}]"
            )

        time_embedding = self.embed_timestep(timestep)
        # PAD_ACTION is a loss TARGET, never an embedding input: nn.Embedding rejects -1, so
        # padded inputs take the dedicated pad row n_actions.
        action_input = torch.where(
            action >= 0, action, torch.full_like(action, config.n_actions)
        )
        tokens = torch.stack(
            [
                self.embed_rtg(rtg) + time_embedding,
                self.embed_state(state) + time_embedding,
                self.embed_action(action_input) + time_embedding,
            ],
            dim=2,
        ).reshape(batch, TOKENS_PER_STEP * steps, config.d_model)

        hidden = self.embed_ln(tokens)
        bias = _attention_bias(
            attention_mask,
            TOKENS_PER_STEP * steps,
            batch=batch,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        for block in self.blocks:
            hidden = block(hidden, bias)
        hidden = self.ln_f(hidden)

        # Token 1 of each step is the state token, and it is the one that predicts that step's
        # action; the action token sits one position later and is therefore causally invisible.
        state_tokens = hidden.reshape(batch, steps, TOKENS_PER_STEP, config.d_model)[:, :, 1]
        return masked_action_logits(self.head(state_tokens), avail_mask)


@dataclass
class _Context:
    """One intersection's rolling window plus the state its return-to-go is in."""

    rtg: list[float]
    state: list[np.ndarray]
    action: list[int]
    timestep: list[int]
    avail: list[np.ndarray]
    reward_sum: float
    next_step: int


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
        state_dim: int | None = None,
    ) -> None:
        super().__init__(gym_env)
        self.env = gym_env

        self.intersections = list(self.env.intersections)
        if not self.intersections:
            raise ValueError("No controllable intersections found in environment.")
        self.intersection_ids = [str(ix.id) for ix in self.intersections]

        action_counts = Utils.infer_action_counts(
            getattr(self.env, "action_space", None), self.intersections
        )
        distinct = sorted({int(count) for count in action_counts})
        if len(distinct) > 1:
            raise ValueError(
                "DTAgent needs one shape for every controlled intersection, but n_actions "
                f"differs across them: {dict(zip(self.intersection_ids, action_counts))}. "
                "C6 forbids padding across intersections, so a heterogeneous set needs one "
                "model per shape, which this task does not build."
            )
        self._n_actions = int(distinct[0])

        self.device = Utils.resolve_device(device)
        self._seed = None if seed is None else int(seed)
        Utils.seed_everything(seed, seed_python_random=False)

        if stats is not None and scenario_id is None:
            raise ValueError(
                "normalisation statistics were supplied without a scenario_id; the statistics "
                "are keyed [scenario_id][ix_id] and cannot be applied without one"
            )
        self._stats = stats
        self._scenario_id = scenario_id
        self._normalise = stats is not None

        self._target_rtg = 0.0 if target_rtg is None else float(target_rtg)
        self._rtg_scale = 1.0 if rtg_scale is None else float(rtg_scale)
        if self._rtg_scale == 0.0:
            raise ValueError("rtg_scale must be non-zero; it divides the RTG input")

        self._config_template = DTConfig(
            state_dim=1 if state_dim is None else int(state_dim),
            n_actions=self._n_actions,
            context_length=int(context_length),
            n_layer=int(n_layer),
            n_head=int(n_head),
            d_model=int(d_model),
            dropout=float(dropout),
            max_ep_len=int(max_ep_len),
        )
        self._config: DTConfig | None = None
        self.model: DecisionTransformer | None = None  # type: ignore[assignment]
        if state_dim is not None:
            self._build_model(int(state_dim))

        self._contexts: dict[str, _Context] = {}
        self.reset_context()
        self._last_info: dict[str, Any] = {}
        self._last_reward: float = 0.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_model(self, state_dim: int) -> None:
        """Build the network for an observed state width, re-applying the seed first.

        Mirrors ``MAPPOAgent``'s lazy build: an env does not expose its state width, only an
        ``info`` does.  Re-seeding here makes the weights a function of ``(seed, shapes)``
        rather than of when the first ``act()`` happened.
        """
        Utils.seed_everything(self._seed, seed_python_random=False)
        self._config = replace(self._config_template, state_dim=int(state_dim))
        self.model = DecisionTransformer(self._config).to(self.device)

    def _ensure_model(self, state_dim: int) -> DecisionTransformer:
        if self.model is None:
            self._build_model(state_dim)
        assert self._config is not None and self.model is not None
        if int(state_dim) != self._config.state_dim:
            raise ValueError(
                f"state width changed for DTAgent: expected {self._config.state_dim}, "
                f"got {state_dim}"
            )
        return self.model

    @property
    def config(self) -> DTConfig:
        """The frozen architecture this agent was built with."""
        if self._config is None:
            raise ValueError(
                "the model has not been built yet: its state width is only known once an "
                "info has been seen, or once a checkpoint has been loaded"
            )
        return self._config

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def reset_context(self) -> None:
        """Drop every intersection's context and restart the return-to-go at the target."""
        self._contexts = {
            ix_id: _Context(
                rtg=[], state=[], action=[], timestep=[], avail=[], reward_sum=0.0, next_step=0
            )
            for ix_id in self.intersection_ids
        }

    def current_rtg(self) -> dict[str, float]:
        """The return-to-go each intersection is currently conditioning on, keyed by id."""
        return {
            ix_id: self._target_rtg - context.reward_sum
            for ix_id, context in self._contexts.items()
        }

    # ------------------------------------------------------------------
    # BaseAgent surface
    # ------------------------------------------------------------------

    def get_info(self) -> dict[str, Any]:
        """The last ``info`` this agent saw (``BaseAgent`` contract)."""
        return self._last_info

    def get_reward(self) -> float:
        """The last scalar reward this agent was handed (``BaseAgent`` contract)."""
        return self._last_reward

    def get_avail_action(self, info: dict[str, Any] | None = None) -> dict[str, list[int]]:
        """Legal actions per intersection, read from *info* when given."""
        if info is None:
            return {ix_id: list(range(self._n_actions)) for ix_id in self.intersection_ids}
        per_ix = Utils.extract_per_intersection_info(info, self.intersection_ids)
        return {
            ix_id: Utils.extract_valid_actions(per_ix[ix_id], self._n_actions)
            for ix_id in self.intersection_ids
        }

    def next_action(self, ob: dict[str, Any]) -> np.ndarray:
        """``BaseAgent`` alias for :meth:`act`."""
        return self.act(ob, explore=True)

    def _reward_for(self, ix_id: str, payload: dict[str, Any]) -> float:
        if "reward" not in payload:
            raise KeyError(
                f"intersection {ix_id!r} carries no 'reward' in info, so the return-to-go "
                "cannot be advanced. The env must be built with a local_reward_fn (contract "
                "C2); freezing the RTG at its target instead would still produce plausible "
                "actions and a plausible number"
            )
        return float(payload["reward"])

    def _normalise_state(self, ix_id: str, state: np.ndarray) -> np.ndarray:
        if not self._normalise:
            return np.asarray(state, dtype=np.float32)
        assert self._stats is not None and self._scenario_id is not None
        rows = np.asarray(state, dtype=np.float32).reshape(1, -1)
        return self._stats.normalize_state(self._scenario_id, ix_id, rows).reshape(-1)

    def _window(self, context: _Context, rtg: float, state: np.ndarray, avail: np.ndarray,
                timestep: int, config: DTConfig) -> dict[str, np.ndarray]:
        """The K-step window ending at the decision about to be made, left-padded like the loader."""
        span = config.context_length
        history = span - 1
        past_rtg = context.rtg[-history:] if history else []
        past_state = context.state[-history:] if history else []
        past_action = context.action[-history:] if history else []
        past_timestep = context.timestep[-history:] if history else []
        past_avail = context.avail[-history:] if history else []

        filled = len(past_rtg) + 1
        start = span - filled

        rtg_out = np.zeros((span, 1), dtype=np.float32)
        state_out = np.zeros((span, config.state_dim), dtype=np.float32)
        # The current step's action is unknown, so it takes PAD_ACTION -- causally invisible to
        # its own prediction, and remapped to the pad embedding row on the way in.
        action_out = np.full(span, PAD_ACTION, dtype=np.int64)
        timestep_out = np.zeros(span, dtype=np.int64)
        avail_out = np.zeros((span, config.n_actions), dtype=np.bool_)
        attention_out = np.zeros(span, dtype=np.bool_)

        for offset, value in enumerate(list(past_rtg) + [rtg]):
            rtg_out[start + offset, 0] = np.float32(value / self._rtg_scale)
        for offset, row in enumerate(list(past_state) + [state]):
            state_out[start + offset] = row
        for offset, value in enumerate(past_action):
            action_out[start + offset] = int(value)
        for offset, value in enumerate(list(past_timestep) + [timestep]):
            timestep_out[start + offset] = int(value)
        for offset, row in enumerate(list(past_avail) + [avail]):
            avail_out[start + offset] = row
        attention_out[start:] = True

        return {
            "rtg": rtg_out,
            "state": state_out,
            "action": action_out,
            "timestep": timestep_out,
            "avail_mask": avail_out,
            "attention_mask": attention_out,
        }

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
        per_ix = Utils.extract_per_intersection_info(info, self.intersection_ids)
        step = int(info.get("step", 0))
        if step == 0:
            self.reset_context()

        states = [Utils.state_from_info(per_ix[ix_id]) for ix_id in self.intersection_ids]
        model = self._ensure_model(int(states[0].shape[0]))
        config = self.config
        if step >= config.max_ep_len:
            raise ValueError(
                f"decision step {step} is outside the timestep embedding "
                f"(max_ep_len {config.max_ep_len})"
            )

        windows: list[dict[str, np.ndarray]] = []
        reward_sums: list[float] = []
        normalised: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for ix_id, raw_state in zip(self.intersection_ids, states):
            context = self._contexts[ix_id]
            if step != context.next_step:
                raise RuntimeError(
                    f"intersection {ix_id!r}: expected decision {context.next_step}, got an "
                    f"info at step {step}. The context and the return-to-go would both be "
                    "wrong for this decision; reset the episode instead"
                )
            reward = 0.0 if step == 0 else self._reward_for(ix_id, per_ix[ix_id])
            reward_sum = context.reward_sum + reward
            rtg = self._target_rtg - reward_sum

            row = self._normalise_state(ix_id, raw_state)
            legal = Utils.extract_valid_actions(per_ix[ix_id], self._n_actions)
            mask = np.zeros(self._n_actions, dtype=np.bool_)
            mask[np.asarray(legal, dtype=np.int64)] = True

            windows.append(self._window(context, rtg, row, mask, step, config))
            reward_sums.append(reward_sum)
            normalised.append(row)
            masks.append(mask)

        def _stack(key: str) -> torch.Tensor:
            return torch.from_numpy(np.stack([window[key] for window in windows])).to(self.device)

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits = model(
                    _stack("rtg"),
                    _stack("state"),
                    _stack("action"),
                    _stack("timestep"),
                    _stack("attention_mask"),
                    _stack("avail_mask"),
                )[:, -1]
        finally:
            model.train(was_training)

        if explore:
            chosen = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).reshape(-1)
        else:
            chosen = torch.argmax(logits, dim=-1)
        actions = chosen.detach().cpu().numpy().astype(np.int64)

        self._last_info = info
        if update_memory:
            span = config.context_length
            for index, ix_id in enumerate(self.intersection_ids):
                context = self._contexts[ix_id]
                context.reward_sum = reward_sums[index]
                context.rtg.append(self._target_rtg - reward_sums[index])
                context.state.append(normalised[index])
                context.action.append(int(actions[index]))
                context.timestep.append(step)
                context.avail.append(masks[index])
                context.next_step = step + 1
                for buffer in (
                    context.rtg,
                    context.state,
                    context.action,
                    context.timestep,
                    context.avail,
                ):
                    del buffer[: max(0, len(buffer) - span)]
        return actions

    def observe(
        self,
        next_info: dict[str, Any],
        reward: Any,
        terminated: bool,
        truncated: bool = False,
    ) -> dict[str, float]:
        """Record the last info and reward.  Does NOT advance the return-to-go.

        ``act()`` is the single route that advances it, from the reward carried in ``info``.
        Advancing it here as well would subtract every reward twice, and the resulting
        trajectory would still look like a plausible DT rollout.
        """
        self._last_info = next_info
        self._last_reward = Utils.scalar_reward(reward)
        return {}

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save(self, path: str, provenance: dict[str, Any] | None = None) -> None:
        """Write weights, config, normalisation statistics, RTG constants and provenance."""
        if self.model is None:
            raise ValueError(
                "nothing to save: the model has not been built yet (its state width is only "
                "known once an info has been seen)"
            )
        torch.save(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "config": self.config.to_json_obj(),
                "model": {
                    key: value.detach().cpu()
                    for key, value in self.model.state_dict().items()
                },
                "target_rtg": float(self._target_rtg),
                "rtg_scale": float(self._rtg_scale),
                "normalise": bool(self._normalise),
                "scenario_id": self._scenario_id,
                "stats": None if self._stats is None else self._stats.to_json_obj(),
                "intersection_ids": list(self.intersection_ids),
                "provenance": dict(provenance or {}),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Adopt a checkpoint, after checking it against this env.

        The shape check happens **before** any state is adopted, so a rejected checkpoint
        leaves this agent exactly as it was -- the in-memory form of the filesystem-mutation
        barrier, mirroring ``MAPPOAgent.load``'s C8 check.
        """
        payload = torch.load(path, map_location=self.device, weights_only=False)
        version = str(payload.get("format_version", ""))
        if version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"checkpoint format {version!r} is not readable by this build "
                f"(expected {CHECKPOINT_FORMAT_VERSION!r})"
            )
        config = DTConfig.from_json_obj(payload["config"])
        if config.n_actions != self._n_actions:
            raise ValueError(
                f"checkpoint n_actions {config.n_actions} does not match this env's "
                f"{self._n_actions}; it was trained on a different action space"
            )
        if self._config is not None and config.state_dim != self._config.state_dim:
            raise ValueError(
                f"checkpoint state_dim {config.state_dim} does not match this agent's "
                f"{self._config.state_dim}"
            )
        if bool(payload.get("normalise", False)) and payload.get("stats") is None:
            raise ValueError(
                "checkpoint says it was trained on normalised states but carries no "
                "statistics; evaluating it on raw states would silently feed the model a "
                "different input distribution"
            )

        # Everything above is a check.  Only now is any state adopted.
        self._config = config
        self.model = DecisionTransformer(config).to(self.device)
        self.model.load_state_dict(payload["model"])
        self._target_rtg = float(payload["target_rtg"])
        self._rtg_scale = float(payload["rtg_scale"])
        self._normalise = bool(payload.get("normalise", False))
        self._scenario_id = payload.get("scenario_id")
        stats_payload = payload.get("stats")
        self._stats = (
            None if stats_payload is None else NormalizationStats.from_json_obj(stats_payload)
        )
        self.reset_context()

    @classmethod
    def from_checkpoint(
        cls, gym_env: Any, path: str, device: str | None = None
    ) -> DTAgent:
        """Build an agent whose architecture comes from the checkpoint, then load it."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = DTConfig.from_json_obj(payload["config"])
        agent = cls(
            gym_env,
            context_length=config.context_length,
            n_layer=config.n_layer,
            n_head=config.n_head,
            d_model=config.d_model,
            dropout=config.dropout,
            max_ep_len=config.max_ep_len,
            device=device,
            state_dim=config.state_dim,
        )
        agent.load(path)
        return agent
