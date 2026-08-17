"""Decision Transformer with a spatial mixing layer interleaved with temporal causal attention.

Checkpoint format version: ``spatial-dt-checkpoint/1.0``.

WHAT THIS IS FOR
----------------
Every offline arm this project has measured is **independent per intersection**: BC, %BC and IQL by
construction, and ``agent/DTAgent.py`` by design.  Across eight tiers the Decision Transformer leads
zero (``PROJECT_PLAN`` section 1b, R7).  This model is the first that can use information the
baselines structurally cannot -- what a neighbouring intersection is doing right now -- and P5.1
exists to find out whether that changes anything.

ARCHITECTURE: INTERLEAVED, NOT BOLTED ON
-----------------------------------------
One block, repeated ``n_layer`` times::

    h = h + temporal_attention(ln(h), causal_bias)     # (B*N, 3K, d)  within one intersection
    h = h + spatial_attention (ln(h), adjacency_bias)  # (B*3K, N, d)  across intersections
    h = h + mlp(ln(h))

Temporal attention is ``agent/DTAgent.py``'s, imported rather than re-implemented, and so is its
causal + key-padding bias.  Spatial attention runs **at every token position independently**: node
``i``'s token at position ``p`` attends to node ``j``'s token at position ``p``, never at another.

THE TOKEN LAYOUT AND THE ALIGNMENT CONVENTION ARE INHERITED, NOT RE-DERIVED
---------------------------------------------------------------------------
Per node, per decision step, in this order::

    position 3t     RTG_t     the return-to-go BEFORE action t
    position 3t+1   s_t       the state seen BEFORE decision t   <- predicts a_t
    position 3t+2   a_t       the action taken at step t

Left padding, ``PAD_ACTION = -1`` as the loss target with a dedicated embedding row for the input,
and "the state token predicts that step's action" all come from ``agent/DTAgent.py`` and
``offline/dataset.py`` unchanged.

⚠️ WHY THE CONCURRENT ACTION CANNOT LEAK, AND WHY THE ORDER OF THE TWO SUBLAYERS IS LOAD-BEARING
--------------------------------------------------------------------------------------------------
All N intersections act **simultaneously**, so "node ``i`` may see node ``j``'s state at step ``t``"
is legitimate and "node ``i`` may see node ``j``'s **action** at step ``t``" is leakage -- it is
precisely the quantity ``i`` is being asked to predict for itself, from a policy that in deployment
has not chosen it yet.

The composition is safe, and the argument is structural rather than empirical:

1. **Temporal runs before spatial in every block.**  After the temporal sublayer, node ``j``'s token
   at position ``3t+1`` has absorbed ``j``'s history **up to and including ``3t+1``**, which excludes
   ``a_t^j`` at ``3t+2``.
2. **Spatial mixes position-wise.**  Node ``i`` at ``3t+1`` reads node ``j`` at ``3t+1`` only.
3. **Temporal attention is causal, so nothing flows backwards.**  Node ``i``'s token at ``3t+2`` --
   which *has* absorbed ``j``'s action token through the spatial sublayer -- can never reach position
   ``3t+1`` in a later block.

Swapping the two sublayers would break step 1, and the model would still train.
``tests/test_spatial_dt_agent.py::test_no_other_nodes_action_at_step_t_can_change_the_prediction_at_step_t``
is the mechanical form of this argument and is the single most load-bearing test in the file.

THE NO-MIXING CONTROL
---------------------
``spatial_mixing=False`` replaces the adjacency mask with the **identity**.  Parameter count and
FLOP count are unchanged (852,872 parameters in both, measured on the grid4x4 shape), so the two
arms differ in the **information** available to them and never in capacity -- which is what makes a
measured difference attributable, and what stops a deficit being read as "the control was smaller".
Neither arm carries a node-identity embedding, so a difference cannot be "it learned which
intersection it is" either.

RETURN-TO-GO IS PER INTERSECTION
---------------------------------
``target_rtg`` and ``rtg_scale`` are mappings keyed by intersection id, not scalars.  P4.6's rule --
*target = max episode return in the training set, scale = max|return|* -- is applied **within each
intersection's own stream set**, which reduces exactly to P4.6 on a one-intersection scenario.  On
``cf_grid4x4__mappo1000`` a single global scalar would condition **15 of 16 nodes outside their own
training support** (measured over the tier's 200 episodes), which is the infeasible-RTG failure mode
``PROJECT_PLAN`` section 9 rates as a live risk.  See ``docs/plans/p5.1.md`` decision D3.

``RTG_0`` is the node's declared target and ``RTG_{t+1} = RTG_t - r_t``, with ``r_t`` read from
``info["intersections"][ix]["reward"]`` -- contract C6's stream, the same one the corpus stores as
``local_reward``.  :meth:`SpatialDTAgent.act` is the only place that advances it.

⚠️ NORMALISATION IS PER INTERSECTION AND THAT IS LOAD-BEARING HERE (``DEFERRED`` 37)
-------------------------------------------------------------------------------------
On ``cf_hz1x1`` every episode carries one intersection, so feeding ``act()`` intersection 0's
statistics for every node is an **equivalent mutant** -- it survived 58 of 58 tests.  On grid4x4 all
**120** intersection pairs have different normalisation means (measured; ``A0`` against ``C1`` differs
by up to 1.2161), so the mutation is live and produces a plausible, wrong grid.  It is killed by
``tests/test_multi_intersection_normalisation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from offline.dataset import NormalizationStats, PAD_ACTION

from .base import BaseAgent
from .DTAgent import TOKENS_PER_STEP, _attention_bias, _SelfAttention, masked_action_logits
from .utils.utils import Utils

__all__ = [
    "SPATIAL_CHECKPOINT_FORMAT_VERSION",
    "SpatialDTAgent",
    "SpatialDTConfig",
    "SpatialDecisionTransformer",
    "spatial_attention_bias",
]

SPATIAL_CHECKPOINT_FORMAT_VERSION = "spatial-dt-checkpoint/1.0"


@dataclass(frozen=True)
class SpatialDTConfig:
    """Architecture of one spatial Decision Transformer.  Frozen at construction, checkpointed.

    ``spatial_mixing`` is part of the architecture record precisely so a checkpoint cannot be
    mistaken for the other arm: the two are weight-compatible by design.
    """

    state_dim: int
    n_actions: int
    n_nodes: int
    context_length: int = 20
    n_layer: int = 3
    n_head: int = 1
    d_model: int = 128
    dropout: float = 0.1
    max_ep_len: int = 360
    spatial_mixing: bool = True

    def __post_init__(self) -> None:
        if self.state_dim < 1 or self.n_actions < 1:
            raise ValueError(
                f"state_dim and n_actions must be >= 1, got {self.state_dim} and {self.n_actions}"
            )
        if self.n_nodes < 1:
            raise ValueError(f"n_nodes must be >= 1, got {self.n_nodes}")
        if self.context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {self.context_length}")
        if self.d_model % self.n_head:
            raise ValueError(
                f"d_model {self.d_model} is not divisible by n_head {self.n_head}"
            )

    def to_json_obj(self) -> dict[str, Any]:
        """JSON-ready architecture record."""
        return {
            "state_dim": int(self.state_dim),
            "n_actions": int(self.n_actions),
            "n_nodes": int(self.n_nodes),
            "context_length": int(self.context_length),
            "n_layer": int(self.n_layer),
            "n_head": int(self.n_head),
            "d_model": int(self.d_model),
            "dropout": float(self.dropout),
            "max_ep_len": int(self.max_ep_len),
            "spatial_mixing": bool(self.spatial_mixing),
        }

    @classmethod
    def from_json_obj(cls, payload: dict[str, Any]) -> SpatialDTConfig:
        """Rebuild a config from its record, refusing an unknown shape."""
        if "spatial_mixing" not in payload:
            raise ValueError(
                "this architecture record carries no 'spatial_mixing' flag; the mixing arm and "
                "its control are weight-compatible by design, so a checkpoint without the flag "
                "cannot be identified as either and is refused"
            )
        return cls(
            state_dim=int(payload["state_dim"]),
            n_actions=int(payload["n_actions"]),
            n_nodes=int(payload["n_nodes"]),
            context_length=int(payload["context_length"]),
            n_layer=int(payload["n_layer"]),
            n_head=int(payload["n_head"]),
            d_model=int(payload["d_model"]),
            dropout=float(payload["dropout"]),
            max_ep_len=int(payload["max_ep_len"]),
            spatial_mixing=bool(payload["spatial_mixing"]),
        )


def spatial_attention_bias(
    mask: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive ``(1, 1, N, N)`` bias from a boolean adjacency mask.

    ``True`` means attendable.  A fully masked query row is refused rather than produced: the
    diagonal is the caller's responsibility (``AdjacencySpec.attention_mask`` opens it in both
    modes), and a row with no attendable key would make this softmax -- which is ours, not
    ``scaled_dot_product_attention``'s -- return NaN.
    """
    if isinstance(mask, torch.Tensor):
        allowed = mask.to(device=device, dtype=torch.bool)
    else:
        allowed = torch.from_numpy(np.asarray(mask, dtype=np.bool_)).to(device)
    if allowed.ndim != 2 or allowed.shape[0] != allowed.shape[1]:
        raise ValueError(
            f"the spatial mask must be a square (N, N) boolean matrix, got "
            f"{tuple(allowed.shape)}"
        )
    empty = (~allowed.any(dim=1)).nonzero().reshape(-1).tolist()
    if empty:
        raise ValueError(
            f"rows {empty[:8]} of the spatial mask have no attendable node. This softmax is ours, "
            "not scaled_dot_product_attention's, so an all-minus-infinity row is NaN and reaches "
            "the backward pass; open the diagonal instead"
        )
    bias = torch.zeros(allowed.shape, device=device, dtype=dtype)
    return bias.masked_fill(~allowed, float("-inf")).unsqueeze(0).unsqueeze(0)


class _SpatialBlock(nn.Module):
    """Pre-LayerNorm block: temporal attention, then spatial attention, then MLP."""

    def __init__(self, d_model: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.ln_temporal = nn.LayerNorm(d_model)
        self.temporal = _SelfAttention(d_model, n_head, dropout)
        self.ln_spatial = nn.LayerNorm(d_model)
        self.spatial = _SelfAttention(d_model, n_head, dropout)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, temporal_bias: torch.Tensor, spatial_bias: torch.Tensor
    ) -> torch.Tensor:
        """``x`` is ``(B, N, L, d)``; both biases are additive masks.

        ⚠️ **The order of the first two sublayers is load-bearing, not stylistic.**  Temporal must
        run first, so that when the spatial sublayer mixes position ``3t+1`` across nodes, no
        node's token at that position has yet absorbed its own action at ``3t+2``.  Swapping them
        leaks every peer's concurrent action into every prediction, and the model still trains.
        """
        batch, nodes, length, width = x.shape

        # Temporal: each intersection attends along its own token sequence, causally.
        hidden = x.reshape(batch * nodes, length, width)
        hidden = hidden + self.temporal(self.ln_temporal(hidden), temporal_bias)

        # Spatial: at each token position independently, intersections attend to their neighbours.
        hidden = hidden.reshape(batch, nodes, length, width)
        hidden = hidden.permute(0, 2, 1, 3).reshape(batch * length, nodes, width)
        hidden = hidden + self.spatial(self.ln_spatial(hidden), spatial_bias)
        hidden = hidden.reshape(batch, length, nodes, width).permute(0, 2, 1, 3)

        return hidden + self.mlp(self.ln_mlp(hidden))


class SpatialDecisionTransformer(nn.Module):
    """Causal transformer over ``(RTG, state, action)`` tokens, mixing across intersections."""

    def __init__(self, config: SpatialDTConfig) -> None:
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
                _SpatialBlock(config.d_model, config.n_head, config.dropout)
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
        spatial_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        avail_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return action logits ``(B, N, K, n_actions)``, one prediction per node per step.

        ``rtg`` is ``(B, N, K, 1)`` already scaled by the caller; ``state`` is ``(B, N, K, D)``
        already normalised **with each node's own statistics**; ``action`` is ``(B, N, K)`` int64
        and may carry ``PAD_ACTION``; ``timestep`` is ``(B, N, K)`` int64; ``spatial_mask`` is
        ``(N, N)`` bool with ``True`` where a node may attend; ``attention_mask`` is ``(B, N, K)``
        bool marking real steps; ``avail_mask`` is ``(B, N, K, n_actions)`` bool.
        """
        config = self.config
        if state.ndim != 4:
            raise ValueError(f"state must be (B, N, K, D), got {tuple(state.shape)}")
        batch, nodes, steps, state_dim = state.shape
        if state_dim != config.state_dim:
            raise ValueError(
                f"state width {state_dim} does not match the model's {config.state_dim}"
            )
        if nodes != config.n_nodes:
            raise ValueError(
                f"node count {nodes} does not match the model's {config.n_nodes}; the adjacency "
                "and the batch must describe the same intersection set"
            )
        if tuple(rtg.shape) != (batch, nodes, steps, 1):
            raise ValueError(f"rtg must be (B, N, K, 1), got {tuple(rtg.shape)}")
        if tuple(action.shape) != (batch, nodes, steps) or tuple(timestep.shape) != (
            batch,
            nodes,
            steps,
        ):
            raise ValueError(
                f"action {tuple(action.shape)} and timestep {tuple(timestep.shape)} must both be "
                f"(B, N, K) = {(batch, nodes, steps)}"
            )
        if tuple(spatial_mask.shape) != (nodes, nodes):
            raise ValueError(
                f"the spatial mask must be (N, N) = {(nodes, nodes)}, got "
                f"{tuple(spatial_mask.shape)}"
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
        if attention_mask is not None:
            if tuple(attention_mask.shape) != (batch, nodes, steps):
                raise ValueError(
                    f"attention_mask must be (B, N, K), got {tuple(attention_mask.shape)}"
                )
            # Every intersection of one decision instant shares its timestep, so the padding is
            # the same for all of them.  Ragged padding would let a real node attend spatially to
            # a padded node's zeros, which no downstream number would reveal.
            reference = attention_mask[:, :1]
            if not torch.equal(attention_mask, reference.expand_as(attention_mask)):
                raise ValueError(
                    "the padding differs across intersections within a window; every node of one "
                    "decision instant shares its timestep, so this window was not built by "
                    "offline.joint_windows and a real node would mix with a padded one"
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
            dim=3,
        ).reshape(batch, nodes, TOKENS_PER_STEP * steps, config.d_model)

        hidden = self.embed_ln(tokens)
        length = TOKENS_PER_STEP * steps
        temporal_bias = _attention_bias(
            None if attention_mask is None else attention_mask.reshape(batch * nodes, steps),
            length,
            batch=batch * nodes,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        spatial_bias = spatial_attention_bias(
            spatial_mask, device=hidden.device, dtype=hidden.dtype
        )
        for block in self.blocks:
            hidden = block(hidden, temporal_bias, spatial_bias)
        hidden = self.ln_f(hidden)

        # Token 1 of each step is the state token, and it is the one that predicts that step's
        # action; the action token sits one position later and is therefore causally invisible.
        state_tokens = hidden.reshape(
            batch, nodes, steps, TOKENS_PER_STEP, config.d_model
        )[:, :, :, 1]
        return masked_action_logits(self.head(state_tokens), avail_mask)


@dataclass
class _NodeContext:
    """One intersection's rolling window plus the state its return-to-go is in."""

    rtg: list[float]
    state: list[np.ndarray]
    action: list[int]
    timestep: list[int]
    avail: list[np.ndarray]
    reward_sum: float
    next_step: int


class SpatialDTAgent(BaseAgent):
    """``BaseAgent``-compatible spatial Decision Transformer over a fixed intersection graph.

    Constructor shape mirrors ``agent/DTAgent.py``.  Unlike that agent, one :meth:`act` performs a
    **single joint forward** over every controlled intersection, and the adjacency mask says which
    of them may see which.  The node order is ``[ix.id for ix in env.intersections]`` and the
    adjacency must be indexed by exactly that order -- checked at construction, never assumed.
    """

    def __init__(
        self,
        gym_env: Any,
        adjacency: Any = None,
        context_length: int = 20,
        n_layer: int = 3,
        n_head: int = 1,
        d_model: int = 128,
        dropout: float = 0.1,
        max_ep_len: int = 360,
        spatial_mixing: bool = True,
        target_rtg: Mapping[str, float] | None = None,
        rtg_scale: Mapping[str, float] | None = None,
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
                "SpatialDTAgent needs one shape for every controlled intersection, but n_actions "
                f"differs across them: {dict(zip(self.intersection_ids, action_counts))}. "
                "C6 forbids padding across intersections, so a heterogeneous set needs one model "
                "per shape, which this task does not build."
            )
        self._n_actions = int(distinct[0])
        self._mask = self._resolve_adjacency(adjacency, bool(spatial_mixing))

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

        self._target_rtg = self._resolve_per_node(target_rtg, "target_rtg", default=0.0)
        self._rtg_scale = self._resolve_per_node(rtg_scale, "rtg_scale", default=1.0)
        zero = sorted(ix for ix, value in self._rtg_scale.items() if value == 0.0)
        if zero:
            raise ValueError(
                f"rtg_scale must be non-zero; it divides the RTG input, and it is 0.0 for {zero}"
            )

        self._config_template = SpatialDTConfig(
            state_dim=1 if state_dim is None else int(state_dim),
            n_actions=self._n_actions,
            n_nodes=len(self.intersection_ids),
            context_length=int(context_length),
            n_layer=int(n_layer),
            n_head=int(n_head),
            d_model=int(d_model),
            dropout=float(dropout),
            max_ep_len=int(max_ep_len),
            spatial_mixing=bool(spatial_mixing),
        )
        self._config: SpatialDTConfig | None = None
        self.model: SpatialDecisionTransformer | None = None  # type: ignore[assignment]
        if state_dim is not None:
            self._build_model(int(state_dim))

        self._contexts: dict[str, _NodeContext] = {}
        self.reset_context()
        self._last_info: dict[str, Any] = {}
        self._last_reward: float = 0.0

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _resolve_adjacency(self, adjacency: Any, spatial_mixing: bool) -> np.ndarray:
        """The ``(N, N)`` mask in env order, refusing any graph that is not indexed by it.

        Accepts an ``AdjacencySpec`` (whose ``node_ids`` must equal the env order -- this is the
        pairing-key proof, not a convenience), a boolean array, or ``None`` for the identity.
        """
        size = len(self.intersection_ids)
        if adjacency is None:
            return np.eye(size, dtype=np.bool_)

        node_ids = getattr(adjacency, "node_ids", None)
        if node_ids is not None:
            if list(node_ids) != self.intersection_ids:
                raise ValueError(
                    f"the adjacency node order {list(node_ids)[:6]} is not this env's "
                    f"{self.intersection_ids[:6]}. Every array here is indexed by env order, so a "
                    "graph in a different order masks the wrong intersections and still trains"
                )
            return np.asarray(
                adjacency.attention_mask(spatial_mixing=spatial_mixing), dtype=np.bool_
            )

        mask = np.asarray(adjacency, dtype=np.bool_)
        if mask.shape != (size, size):
            raise ValueError(
                f"the adjacency must be ({size}, {size}) for this env's {size} intersections, "
                f"got {mask.shape}"
            )
        if not spatial_mixing:
            return np.eye(size, dtype=np.bool_)
        return mask | np.eye(size, dtype=np.bool_)

    def _resolve_per_node(
        self, values: Mapping[str, float] | None, name: str, *, default: float
    ) -> dict[str, float]:
        """Per-intersection RTG constants, refusing a partial mapping."""
        if values is None:
            return {ix_id: float(default) for ix_id in self.intersection_ids}
        missing = [ix_id for ix_id in self.intersection_ids if ix_id not in values]
        if missing:
            raise ValueError(
                f"{name} is missing {len(missing)} intersection(s): {missing[:8]}. P4.6's rule is "
                "applied within each intersection's own stream set (docs/plans/p5.1.md D3), so "
                "every controlled intersection needs its own value and none is defaulted"
            )
        return {ix_id: float(values[ix_id]) for ix_id in self.intersection_ids}

    def _build_model(self, state_dim: int) -> None:
        """Build the network for an observed state width, re-applying the seed first."""
        Utils.seed_everything(self._seed, seed_python_random=False)
        self._config = replace(self._config_template, state_dim=int(state_dim))
        self.model = SpatialDecisionTransformer(self._config).to(self.device)

    def _ensure_model(self, state_dim: int) -> SpatialDecisionTransformer:
        if self.model is None:
            self._build_model(state_dim)
        assert self._config is not None and self.model is not None
        if int(state_dim) != self._config.state_dim:
            raise ValueError(
                f"state width changed for SpatialDTAgent: expected {self._config.state_dim}, "
                f"got {state_dim}"
            )
        return self.model

    @property
    def config(self) -> SpatialDTConfig:
        """The frozen architecture this agent was built with."""
        if self._config is None:
            raise ValueError(
                "the model has not been built yet: its state width is only known once an info "
                "has been seen, or once a checkpoint has been loaded"
            )
        return self._config

    @property
    def spatial_mask(self) -> np.ndarray:
        """The ``(N, N)`` boolean mask actually used, in ``intersection_ids`` order."""
        return self._mask.copy()

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def reset_context(self) -> None:
        """Drop every node's context and restart each return-to-go at its own target."""
        self._contexts = {
            ix_id: _NodeContext(
                rtg=[], state=[], action=[], timestep=[], avail=[], reward_sum=0.0, next_step=0
            )
            for ix_id in self.intersection_ids
        }

    def current_rtg(self) -> dict[str, float]:
        """The return-to-go each intersection is conditioning on, keyed by id."""
        return {
            ix_id: self._target_rtg[ix_id] - context.reward_sum
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
                f"intersection {ix_id!r} carries no 'reward' in info, so the return-to-go cannot "
                "be advanced. The env must be built with a local_reward_fn (contract C2); "
                "freezing the RTG at its target instead would still produce plausible actions "
                "and a plausible number"
            )
        return float(payload["reward"])

    def _normalise_state(self, ix_id: str, state: np.ndarray) -> np.ndarray:
        """Normalise with **this intersection's own** statistics (``DEFERRED`` 37)."""
        if not self._normalise:
            return np.asarray(state, dtype=np.float32)
        assert self._stats is not None and self._scenario_id is not None
        rows = np.asarray(state, dtype=np.float32).reshape(1, -1)
        return self._stats.normalize_state(self._scenario_id, ix_id, rows).reshape(-1)

    def _window(
        self,
        context: _NodeContext,
        ix_id: str,
        rtg: float,
        state: np.ndarray,
        avail: np.ndarray,
        timestep: int,
        config: SpatialDTConfig,
    ) -> dict[str, np.ndarray]:
        """One node's K-step window ending at the decision about to be made, left-padded."""
        span = config.context_length
        history = span - 1
        past_rtg = context.rtg[-history:] if history else []
        past_state = context.state[-history:] if history else []
        past_action = context.action[-history:] if history else []
        past_timestep = context.timestep[-history:] if history else []
        past_avail = context.avail[-history:] if history else []

        filled = len(past_rtg) + 1
        start = span - filled
        scale = self._rtg_scale[ix_id]

        rtg_out = np.zeros((span, 1), dtype=np.float32)
        state_out = np.zeros((span, config.state_dim), dtype=np.float32)
        action_out = np.full(span, PAD_ACTION, dtype=np.int64)
        timestep_out = np.zeros(span, dtype=np.int64)
        avail_out = np.zeros((span, config.n_actions), dtype=np.bool_)
        attention_out = np.zeros(span, dtype=np.bool_)

        for offset, value in enumerate(list(past_rtg) + [rtg]):
            rtg_out[start + offset, 0] = np.float32(value / scale)
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

        A new episode is detected by ``info["step"] == 0``.  Unlike ``DTAgent``, this performs a
        **single joint forward** over every intersection, so the mixing layer can see them all.
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
                    f"intersection {ix_id!r}: expected decision {context.next_step}, got an info "
                    f"at step {step}. The context and the return-to-go would both be wrong for "
                    "this decision; reset the episode instead"
                )
            reward = 0.0 if step == 0 else self._reward_for(ix_id, per_ix[ix_id])
            reward_sum = context.reward_sum + reward
            rtg = self._target_rtg[ix_id] - reward_sum

            row = self._normalise_state(ix_id, raw_state)
            legal = Utils.extract_valid_actions(per_ix[ix_id], self._n_actions)
            mask = np.zeros(self._n_actions, dtype=np.bool_)
            mask[np.asarray(legal, dtype=np.int64)] = True

            windows.append(self._window(context, ix_id, rtg, row, mask, step, config))
            reward_sums.append(reward_sum)
            normalised.append(row)
            masks.append(mask)

        def _stack(key: str) -> torch.Tensor:
            # (1, N, K, ...) -- one joint decision instant, every intersection.
            stacked = np.stack([window[key] for window in windows])
            return torch.from_numpy(stacked).unsqueeze(0).to(self.device)

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits = model(
                    _stack("rtg"),
                    _stack("state"),
                    _stack("action"),
                    _stack("timestep"),
                    torch.from_numpy(self._mask).to(self.device),
                    _stack("attention_mask"),
                    _stack("avail_mask"),
                )[0, :, -1]
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
                context.rtg.append(self._target_rtg[ix_id] - reward_sums[index])
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

        Two routes advancing it would double-decrement, and the resulting number would still look
        plausible.  :meth:`act` is the only place it moves.
        """
        self._last_info = next_info
        self._last_reward = float(Utils.scalar_reward(reward))
        return {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, provenance: dict[str, Any] | None = None) -> None:
        """Write weights, config, statistics, the per-node RTG constants and the graph."""
        if self.model is None:
            raise ValueError(
                "nothing to save: the model has not been built yet (its state width is only "
                "known once an info has been seen)"
            )
        torch.save(
            {
                "format_version": SPATIAL_CHECKPOINT_FORMAT_VERSION,
                "config": self.config.to_json_obj(),
                "model": {
                    key: value.detach().cpu()
                    for key, value in self.model.state_dict().items()
                },
                "target_rtg": dict(self._target_rtg),
                "rtg_scale": dict(self._rtg_scale),
                "normalise": bool(self._normalise),
                "scenario_id": self._scenario_id,
                "stats": None if self._stats is None else self._stats.to_json_obj(),
                "intersection_ids": list(self.intersection_ids),
                "spatial_mask": self._mask.tolist(),
                "provenance": dict(provenance or {}),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Adopt a checkpoint after checking it against this env; a refusal changes nothing."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        version = str(payload.get("format_version", ""))
        if version != SPATIAL_CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"checkpoint format {version!r} is not readable by this build "
                f"(expected {SPATIAL_CHECKPOINT_FORMAT_VERSION!r})"
            )
        config = SpatialDTConfig.from_json_obj(payload["config"])
        if config.n_actions != self._n_actions:
            raise ValueError(
                f"checkpoint n_actions {config.n_actions} does not match this env's "
                f"{self._n_actions}; it was trained on a different action space"
            )
        if config.n_nodes != len(self.intersection_ids):
            raise ValueError(
                f"checkpoint n_nodes {config.n_nodes} does not match this env's "
                f"{len(self.intersection_ids)}"
            )
        recorded_ids = [str(ix) for ix in payload.get("intersection_ids", [])]
        if recorded_ids and recorded_ids != self.intersection_ids:
            raise ValueError(
                f"checkpoint intersection order {recorded_ids[:6]} does not match this env's "
                f"{self.intersection_ids[:6]}; the adjacency and the per-node constants are both "
                "indexed by that order"
            )
        if self._config is not None and config.state_dim != self._config.state_dim:
            raise ValueError(
                f"checkpoint state_dim {config.state_dim} does not match this agent's "
                f"{self._config.state_dim}"
            )
        if bool(payload.get("normalise", False)) and payload.get("stats") is None:
            raise ValueError(
                "checkpoint says it was trained on normalised states but carries no statistics; "
                "evaluating it on raw states would silently feed the model a different input "
                "distribution"
            )
        recorded_mask = payload.get("spatial_mask")
        if recorded_mask is None:
            raise ValueError(
                "checkpoint carries no spatial_mask; the mixing arm and its control are "
                "weight-compatible, so the graph a model was trained under cannot be inferred "
                "from its weights and is not defaulted"
            )

        # Everything above is a check.  Only now is any state adopted.
        self._config = config
        self._mask = np.asarray(recorded_mask, dtype=np.bool_)
        self.model = SpatialDecisionTransformer(config).to(self.device)
        self.model.load_state_dict(payload["model"])
        self._target_rtg = {str(k): float(v) for k, v in payload["target_rtg"].items()}
        self._rtg_scale = {str(k): float(v) for k, v in payload["rtg_scale"].items()}
        self._normalise = bool(payload.get("normalise", False))
        self._scenario_id = payload.get("scenario_id")
        stats_payload = payload.get("stats")
        self._stats = (
            None if stats_payload is None else NormalizationStats.from_json_obj(stats_payload)
        )
        self.reset_context()

    @classmethod
    def from_checkpoint(
        cls, gym_env: Any, path: str, device: str | None = None, adjacency: Any = None
    ) -> SpatialDTAgent:
        """Build an agent whose architecture and graph come from the checkpoint, then load it.

        ⚠️ *adjacency* is accepted only so this signature matches the constructor's; whatever it
        says is **overwritten** by the checkpoint's own recorded mask in :meth:`load`.  A model
        evaluated under a graph other than the one it trained under is a different model.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = SpatialDTConfig.from_json_obj(payload["config"])
        agent = cls(
            gym_env,
            adjacency=adjacency,
            context_length=config.context_length,
            n_layer=config.n_layer,
            n_head=config.n_head,
            d_model=config.d_model,
            dropout=config.dropout,
            max_ep_len=config.max_ep_len,
            spatial_mixing=config.spatial_mixing,
            target_rtg={str(k): float(v) for k, v in payload["target_rtg"].items()},
            rtg_scale={str(k): float(v) for k, v in payload["rtg_scale"].items()},
            device=device,
            state_dim=config.state_dim,
        )
        agent.load(path)
        return agent
