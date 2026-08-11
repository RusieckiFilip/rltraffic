"""Offline baselines for P4.4: behaviour cloning and independent per-intersection IQL.

Checkpoint format versions: ``bc-checkpoint/1.0`` and ``iql-checkpoint/1.0``.

WHAT THESE ARE FOR, AND WHAT A DIFFERENCE AGAINST THEM MEANS
------------------------------------------------------------
The P4 Decision Transformer beat the policy whose data it trained on by 0.6263 ATT.  These
baselines exist to say whether that margin needed a sequence model.  **Both agents here see one
state and nothing else** -- no context, no return-to-go, no timestep -- so a difference between
the DT and BC is a **combined** difference (attention/context **plus** return conditioning
**plus** the timestep embedding) and must never be reported as "sequence modelling adds X".
Decomposing it is P4.3's (RTG) and P5.3's (no-RTG, context length K) registered work.

Architecture (declared in ``docs/plans/p4.4.md`` section 3.2, before any training)
---------------------------------------------------------------------------------
:class:`MLPTrunk` is the DT's per-position feed-forward stack **with attention removed**: an
input projection, ``n_layer`` pre-LayerNorm residual blocks ``Linear(d, 4d) -> GELU ->
Linear(4d, d)``, a final LayerNorm and a linear head.  Choosing the DT's own block rather than a
plainer MLP is deliberate: a different architecture family would confound "no sequence" with
"different network", and the parameter difference is then exactly the attention weights.

Alignment convention (inherited from contract C6 and ``offline/dataset.py``; not re-derived)
--------------------------------------------------------------------------------------------
A window's row ``k`` is the state seen **before** decision ``k``, and the loss target at that
row is the action taken there.  BC trains on **every valid row of every window the DT saw**,
through the DT's own :func:`~agent.DTAgent.action_loss` with ``ignore_index = PAD_ACTION = -1``,
so a forgotten mask crashes instead of training on fabricated action-0 targets.  At inference the
agent sees exactly one row -- the current state -- which is the row a training window ends on;
``tests/test_offline_baseline_agents.py`` asserts that equality exactly, for every step of a real
episode, because P4's review found the DT's equivalent path unprotected and three surviving
mutations moved the reported number.

Masking is applied through :func:`~agent.DTAgent.masked_action_logits`, which leaves a row with
no legal action untouched -- masking such a row would build an all-``-inf`` logit row whose
softmax is NaN.  On this corpus every mask is all-``True`` (P3 review, 32000/32000 streams), so
no capability claim is made from masking here; it is kept because the env raises on an illegal
action and masks do bind under cyclic control and P6 perturbations.

Normalisation
-------------
Both agents apply the **frozen training-split statistics** to every state they see, through
``NormalizationStats.normalize_state`` -- the same function and the same numbers the loader used
during training.  Refitting at evaluation time is the leakage the loader's docstring exists to
prevent, and feeding raw states to a model trained on normalised ones was worth **+32.5 ATT** on
the DT (P4 review, MAJOR-1).

The canonical checkpoint digest (``DEFERRED`` 29)
-------------------------------------------------
``torch.save`` names the zip root after the output file, so two checkpoints written under
different names can never be byte-identical however deterministic the training was, and a
checkpoint's file hash also moves with its provenance block.  **No claim of the form "the model
reproduces byte-identically" is testable at file level.**  :func:`canonical_state_dict_digest`
hashes the ``state_dict`` tensor bytes in **sorted key order**, with each tensor's key, shape and
dtype folded in, so it is independent of both the filename and the provenance.  Every determinism
claim in P4.4 is made with it; the file sha256 is kept beside it for what it does prove --
transport integrity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from offline.dataset import NormalizationStats

from .base import BaseAgent
from .DTAgent import action_loss, masked_action_logits
from .utils.utils import Utils

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
    """Architecture of one state-conditioned network.  Frozen at construction, checkpointed."""

    state_dim: int
    n_actions: int
    d_model: int = 128
    n_layer: int = 3
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.state_dim < 1 or self.n_actions < 1:
            raise ValueError(
                f"state_dim and n_actions must be >= 1, got {self.state_dim} and {self.n_actions}"
            )
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {self.n_layer}")

    def to_json_obj(self) -> dict[str, Any]:
        """JSON-ready mapping; every field round-trips exactly."""
        return {
            "state_dim": int(self.state_dim),
            "n_actions": int(self.n_actions),
            "d_model": int(self.d_model),
            "n_layer": int(self.n_layer),
            "dropout": float(self.dropout),
        }

    @classmethod
    def from_json_obj(cls, payload: dict[str, Any]) -> TrunkConfig:
        """Rebuild a config written by :meth:`to_json_obj`."""
        return cls(
            state_dim=int(payload["state_dim"]),
            n_actions=int(payload["n_actions"]),
            d_model=int(payload["d_model"]),
            n_layer=int(payload["n_layer"]),
            dropout=float(payload["dropout"]),
        )


def canonical_state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    """sha256 over the tensor bytes in sorted key order (``DEFERRED`` 29).

    Filename- and provenance-independent by construction: only the parameters enter it.  Each
    entry contributes ``key|shape|dtype|`` followed by the tensor's bytes in **little-endian**
    order, so the digest does not depend on the host's byte order either, and two tensors with
    the same bytes under different keys or shapes do not collide.

    This is the quantity every determinism claim in P4.4 is made with.  A file sha256 answers a
    different question -- whether a file arrived intact -- and both are recorded.
    """
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().to("cpu").contiguous()
        array = tensor.numpy()
        digest.update(f"{key}|{tuple(tensor.shape)}|{tensor.dtype}|".encode("utf-8"))
        digest.update(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes())
    return digest.hexdigest()


class _FeedForwardBlock(nn.Module):
    """The DT's ``_Block`` with its attention half removed: pre-LayerNorm MLP plus residual."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.ln(x))


class MLPTrunk(nn.Module):
    """State -> ``out_dim`` values, with no access to any other position.

    Accepts any leading batch shape: ``(B, D)`` at inference and ``(B, K, D)`` at training, so
    BC's training input is literally the DT's batch with the sequence axis left alone.
    """

    def __init__(self, config: TrunkConfig, out_dim: int) -> None:
        super().__init__()
        if int(out_dim) < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        self.config = config
        self.out_dim = int(out_dim)
        self.embed = nn.Linear(config.state_dim, config.d_model)
        self.blocks = nn.ModuleList(
            [_FeedForwardBlock(config.d_model, config.dropout) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, self.out_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """``(..., state_dim) -> (..., out_dim)``."""
        if int(state.shape[-1]) != int(self.config.state_dim):
            raise ValueError(
                f"state width {int(state.shape[-1])} does not match the model's "
                f"{self.config.state_dim}"
            )
        hidden = self.embed(state)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.ln_f(hidden))


class _StateConditionedAgent(BaseAgent):
    """Shared ``BaseAgent`` face for the offline baselines: one state in, one action out.

    Constructor shape mirrors ``agent/MAPPOAgent.py`` and ``agent/DTAgent.py`` (``gym_env``
    first, ``device`` and ``seed`` last).  One network set serves every controlled intersection;
    a heterogeneous ``n_actions`` set is refused naming both shapes, because C6 forbids padding
    across intersections.  The intersections do not see each other.

    Networks are built lazily from the first observed state width, mirroring
    ``MAPPOAgent.ensure_initialized``: an env does not expose that width, only an ``info`` does.
    The seed is re-applied at build time, so the weights are a function of ``(seed, shapes)``
    rather than of when the first ``act()`` happened.
    """

    checkpoint_format_version = ""

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
                f"{type(self).__name__} needs one shape for every controlled intersection, but "
                f"n_actions differs across them: "
                f"{dict(zip(self.intersection_ids, action_counts))}. C6 forbids padding across "
                "intersections, so a heterogeneous set needs one model per shape, which this "
                "task does not build."
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

        self._config_template = TrunkConfig(
            state_dim=1 if state_dim is None else int(state_dim),
            n_actions=self._n_actions,
            d_model=int(d_model),
            n_layer=int(n_layer),
            dropout=float(dropout),
        )
        self._config: TrunkConfig | None = None
        self._last_info: dict[str, Any] = {}
        self._last_reward: float = 0.0
        if state_dim is not None:
            self._build_networks(int(state_dim))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_networks(self, state_dim: int) -> None:
        """Re-apply the seed, then build every network for the observed state width."""
        Utils.seed_everything(self._seed, seed_python_random=False)
        self._config = replace(self._config_template, state_dim=int(state_dim))
        self._make_networks(self._config)

    def _make_networks(self, config: TrunkConfig) -> None:
        raise NotImplementedError

    def _network_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        """Prefix -> ``state_dict``, in a fixed order.  The digest sorts, so order is cosmetic."""
        raise NotImplementedError

    def _load_network_state_dicts(self, merged: Mapping[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def policy_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Action logits for a batch of states.  The single route ``act()`` takes."""
        raise NotImplementedError

    def _ensure_networks(self, state_dim: int) -> None:
        if self._config is None:
            self._build_networks(state_dim)
        assert self._config is not None
        if int(state_dim) != self._config.state_dim:
            raise ValueError(
                f"state width changed for {type(self).__name__}: expected "
                f"{self._config.state_dim}, got {state_dim}"
            )

    @property
    def config(self) -> TrunkConfig:
        """The frozen architecture this agent was built with."""
        if self._config is None:
            raise ValueError(
                "the model has not been built yet: its state width is only known once an info "
                "has been seen, or once a checkpoint has been loaded"
            )
        return self._config

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

    def _normalise_state(self, ix_id: str, state: np.ndarray) -> np.ndarray:
        """Apply the FROZEN training statistics, or pass the raw state through unchanged."""
        if not self._normalise:
            return np.asarray(state, dtype=np.float32)
        assert self._stats is not None and self._scenario_id is not None
        rows = np.asarray(state, dtype=np.float32).reshape(1, -1)
        return self._stats.normalize_state(self._scenario_id, ix_id, rows).reshape(-1)

    def _modules(self) -> list[nn.Module]:
        raise NotImplementedError

    def act(
        self, info: dict[str, Any], explore: bool = True, update_memory: bool = True
    ) -> np.ndarray:
        """One action per intersection, ordered by ``[ix.id for ix in env.intersections]``.

        ``explore=False`` takes the argmax over masked logits -- the declared evaluation path,
        matching the DT's.  ``explore=True`` samples from the masked softmax.  These policies
        are Markovian, so ``update_memory`` has nothing to update and is accepted only for
        ``BaseAgent`` compatibility; it is recorded in the docstring rather than silently
        ignored.
        """
        per_ix = Utils.extract_per_intersection_info(info, self.intersection_ids)
        states = [Utils.state_from_info(per_ix[ix_id]) for ix_id in self.intersection_ids]
        self._ensure_networks(int(states[0].shape[0]))

        rows: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for ix_id, raw_state in zip(self.intersection_ids, states):
            rows.append(self._normalise_state(ix_id, raw_state))
            legal = Utils.extract_valid_actions(per_ix[ix_id], self._n_actions)
            mask = np.zeros(self._n_actions, dtype=np.bool_)
            mask[np.asarray(legal, dtype=np.int64)] = True
            masks.append(mask)

        state = torch.from_numpy(np.stack(rows)).to(self.device)
        avail = torch.from_numpy(np.stack(masks)).to(self.device)

        modules = self._modules()
        was_training = [module.training for module in modules]
        for module in modules:
            module.eval()
        try:
            with torch.no_grad():
                logits = masked_action_logits(self.policy_logits(state), avail)
        finally:
            for module, mode in zip(modules, was_training):
                module.train(mode)

        if explore:
            chosen = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).reshape(-1)
        else:
            chosen = torch.argmax(logits, dim=-1)

        self._last_info = info
        return chosen.detach().cpu().numpy().astype(np.int64)

    def observe(
        self,
        next_info: dict[str, Any],
        reward: Any,
        terminated: bool,
        truncated: bool = False,
    ) -> dict[str, float]:
        """Record the last info and reward.  These policies carry no other state."""
        self._last_info = next_info
        self._last_reward = Utils.scalar_reward(reward)
        return {}

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def merged_state_dict(self) -> dict[str, torch.Tensor]:
        """Every network's parameters in one mapping, prefixed by network name, on the CPU."""
        if self._config is None:
            raise ValueError(
                "nothing to read: the model has not been built yet (its state width is only "
                "known once an info has been seen)"
            )
        merged: dict[str, torch.Tensor] = {}
        for prefix, state in self._network_state_dicts().items():
            for key, value in state.items():
                merged[f"{prefix}.{key}"] = value.detach().to("cpu")
        return merged

    def canonical_digest(self) -> str:
        """The filename- and provenance-independent identity of these weights."""
        return canonical_state_dict_digest(self.merged_state_dict())

    def save(self, path: str, provenance: dict[str, Any] | None = None) -> None:
        """Write weights, config, normalisation statistics, the digest and provenance."""
        if self._config is None:
            raise ValueError(
                "nothing to save: the model has not been built yet (its state width is only "
                "known once an info has been seen)"
            )
        merged = self.merged_state_dict()
        torch.save(
            {
                "format_version": self.checkpoint_format_version,
                "config": self.config.to_json_obj(),
                "model": merged,
                "canonical_digest": canonical_state_dict_digest(merged),
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

        Every check runs **before** any state is adopted, so a rejected checkpoint leaves this
        agent exactly as it was -- the in-memory form of the filesystem-mutation barrier,
        mirroring ``MAPPOAgent.load``'s C8 check and ``DTAgent.load``.
        """
        payload = torch.load(path, map_location=self.device, weights_only=False)
        version = str(payload.get("format_version", ""))
        if version != self.checkpoint_format_version:
            raise ValueError(
                f"checkpoint format {version!r} is not readable by this build "
                f"(expected {self.checkpoint_format_version!r})"
            )
        config = TrunkConfig.from_json_obj(payload["config"])
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
        self._make_networks(config)
        self._load_network_state_dicts(payload["model"])
        self._normalise = bool(payload.get("normalise", False))
        self._scenario_id = payload.get("scenario_id")
        stats_payload = payload.get("stats")
        self._stats = (
            None if stats_payload is None else NormalizationStats.from_json_obj(stats_payload)
        )

    @classmethod
    def from_checkpoint(
        cls, gym_env: Any, path: str, device: str | None = None
    ) -> Any:
        """Build an agent whose architecture comes from the checkpoint, then load it."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = TrunkConfig.from_json_obj(payload["config"])
        agent = cls(
            gym_env,
            d_model=config.d_model,
            n_layer=config.n_layer,
            dropout=config.dropout,
            device=device,
            state_dim=config.state_dim,
        )
        agent.load(path)
        return agent


class BCAgent(_StateConditionedAgent):
    """Behaviour cloning over the current state only.  Serves BC and %BC alike.

    The two differ solely in which windows the optimiser saw -- %BC keeps the top-10 % of
    trajectories by return -- so they share one class and the checkpoint records ``method``.
    """

    checkpoint_format_version = BC_CHECKPOINT_FORMAT_VERSION

    def __init__(self, *args: Any, method: str = "bc", **kwargs: Any) -> None:
        self.method = str(method)
        self.model: MLPTrunk | None = None
        super().__init__(*args, **kwargs)

    def _make_networks(self, config: TrunkConfig) -> None:
        self.model = MLPTrunk(config, config.n_actions).to(self.device)

    def _modules(self) -> list[nn.Module]:
        assert self.model is not None
        return [self.model]

    def _network_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        assert self.model is not None
        return {"policy": self.model.state_dict()}

    def _load_network_state_dicts(self, merged: Mapping[str, torch.Tensor]) -> None:
        assert self.model is not None
        self.model.load_state_dict(_strip_prefix(merged, "policy"))

    def policy_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Action logits ``(..., n_actions)`` from the state alone."""
        assert self.model is not None
        return self.model(state)

    def save(self, path: str, provenance: dict[str, Any] | None = None) -> None:
        """As the base class, plus the ``method`` label that separates BC from %BC."""
        super().save(path, {**dict(provenance or {}), "method": self.method})


class IQLAgent(_StateConditionedAgent):
    """Independent per-intersection IQL: expectile ``V``, ``Q``, and an AWR-extracted policy.

    ``act()`` reads the **policy** network.  Taking ``argmax Q`` instead would still produce
    plausible actions and a plausible number, which is why a test perturbs ``Q`` and asserts the
    decision does not move.

    ⚠️ The value learner **bootstraps through the episode boundary**: ``terminated`` is hardcoded
    ``False`` on this platform and every episode ends by time-limit truncation, so there is no
    ``done`` term anywhere in this file or in ``offline/offline_baselines.py``.  Treating a
    timeout as terminal causes systematic value underestimation near episode end and would hand
    the DT an unearned win over its own baselines (``PREREGISTRATION.md`` section 7, Decisions
    Log 2026-07-26).
    """

    checkpoint_format_version = IQL_CHECKPOINT_FORMAT_VERSION

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.policy: MLPTrunk | None = None
        self.q: MLPTrunk | None = None
        self.v: MLPTrunk | None = None
        self.q_target: MLPTrunk | None = None
        super().__init__(*args, **kwargs)

    def _make_networks(self, config: TrunkConfig) -> None:
        self.policy = MLPTrunk(config, config.n_actions).to(self.device)
        self.q = MLPTrunk(config, config.n_actions).to(self.device)
        self.v = MLPTrunk(config, 1).to(self.device)
        self.q_target = MLPTrunk(config, config.n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

    def _modules(self) -> list[nn.Module]:
        assert self.policy is not None and self.q is not None and self.v is not None
        assert self.q_target is not None
        return [self.policy, self.q, self.v, self.q_target]

    def _network_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        assert self.policy is not None and self.q is not None and self.v is not None
        assert self.q_target is not None
        return {
            "policy": self.policy.state_dict(),
            "q": self.q.state_dict(),
            "v": self.v.state_dict(),
            "q_target": self.q_target.state_dict(),
        }

    def _load_network_state_dicts(self, merged: Mapping[str, torch.Tensor]) -> None:
        assert self.policy is not None and self.q is not None and self.v is not None
        assert self.q_target is not None
        self.policy.load_state_dict(_strip_prefix(merged, "policy"))
        self.q.load_state_dict(_strip_prefix(merged, "q"))
        self.v.load_state_dict(_strip_prefix(merged, "v"))
        self.q_target.load_state_dict(_strip_prefix(merged, "q_target"))

    def policy_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Action logits from the AWR-extracted policy -- never from ``Q``."""
        assert self.policy is not None
        return self.policy(state)

    def q_values(self, state: torch.Tensor) -> torch.Tensor:
        """``Q(s, .)`` for every action."""
        assert self.q is not None
        return self.q(state)

    def state_values(self, state: torch.Tensor) -> torch.Tensor:
        """``V(s)``, with the trailing singleton dimension removed."""
        assert self.v is not None
        return self.v(state).squeeze(-1)


def _strip_prefix(merged: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """The entries of *merged* under ``prefix.``, with the prefix removed.

    Raises when the prefix is absent rather than loading an empty state dict: ``load_state_dict``
    with ``strict=True`` would catch it, but the message would name every missing parameter
    instead of the network.
    """
    head = f"{prefix}."
    out = {key[len(head) :]: value for key, value in merged.items() if key.startswith(head)}
    if not out:
        raise ValueError(
            f"checkpoint carries no parameters for the {prefix!r} network; it has "
            f"{sorted({key.split('.', 1)[0] for key in merged})}"
        )
    return out
