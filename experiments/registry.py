"""Agent and baseline registry for the experiment framework.

Maps a short agent ``type`` (``"dqn"``, ``"ippo"``, ``"mappo"``) to its
implementation class plus the hyperparameters it accepts.  The agent
classes (which pull in torch) are imported lazily in :func:`build_agent`,
so importing this module — and therefore loading/validating a config or
running ``--dry-run`` — stays cheap.

Baselines are non-trained reference policies (``"random"``,
``"max_pressure"``) evaluated alongside the trained agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentType:
    """Static description of a registered trainable agent."""

    type: str
    class_path: str
    label: str
    # Default hyperparameters mirror the agent's ``__init__`` signature.
    # Only these keys may be overridden from a config; ``device`` and
    # ``seed`` are supplied by the runner and are intentionally absent.
    default_params: Mapping[str, Any]

    def load_class(self) -> type[Any]:
        module_name, class_name = self.class_path.rsplit(".", 1)
        return getattr(import_module(module_name), class_name)


# ``epsilon_decay_steps`` is the only sentinel: ``None`` means "decay over
# the whole training budget" and is resolved to ``train_episodes * max_steps``
# in :func:`build_agent`.  Every other default matches the class default.
_PPO_DEFAULTS: dict[str, Any] = {
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "update_epochs": 4,
    "minibatch_size": 128,
    "rollout_size": 1024,
    "hidden_dim": 128,
    "max_grad_norm": 0.5,
}

AGENT_TYPES: dict[str, AgentType] = {
    "dqn": AgentType(
        type="dqn",
        class_path="agent.DQNAgent.DQNAgent",
        label="DQN",
        default_params={
            "lr": 1e-3,
            "gamma": 0.99,
            "batch_size": 64,
            "replay_size": 100_000,
            "min_replay_size": 1_000,
            "target_update_interval": 200,
            "hidden_dim": 128,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": None,
        },
    ),
    "ippo": AgentType(
        type="ippo",
        class_path="agent.IPPOagent.IPPOAgent",
        label="IPPO",
        default_params=dict(_PPO_DEFAULTS),
    ),
    "mappo": AgentType(
        type="mappo",
        class_path="agent.MAPPOAgent.MAPPOAgent",
        label="MAPPO",
        default_params=dict(_PPO_DEFAULTS),
    ),
}

# Non-trained reference policies; label is what shows up in the report.
BASELINE_LABELS: dict[str, str] = {
    "random": "Random",
    "max_pressure": "MaxPressure",
}
BASELINE_NAMES: tuple[str, ...] = tuple(BASELINE_LABELS)
RESERVED_LABELS = frozenset(BASELINE_LABELS.values())


def get_agent_type(type_name: str) -> AgentType:
    key = str(type_name).lower()
    if key not in AGENT_TYPES:
        supported = ", ".join(sorted(AGENT_TYPES))
        raise ValueError(
            f"unsupported agent type '{type_name}'. Supported types: {supported}"
        )
    return AGENT_TYPES[key]


def validate_agent_params(
    type_name: str,
    params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge *params* onto the type defaults, rejecting unknown keys.

    Returns the full resolved parameter dict for the agent.
    """
    spec = get_agent_type(type_name)
    merged = dict(spec.default_params)
    for key in (params or {}):
        if key not in merged:
            allowed = ", ".join(sorted(merged))
            raise ValueError(
                f"unsupported parameter '{key}' for agent '{spec.type}'. "
                f"Allowed parameters: {allowed}"
            )
    merged.update(params or {})
    return merged


def build_agent(
    type_name: str,
    env: Any,
    params: Mapping[str, Any],
    *,
    device: str,
    seed: int,
    train_episodes: int,
    max_steps: int,
) -> Any:
    """Instantiate a trained-agent class against *env*.

    Resolves the ``epsilon_decay_steps=None`` sentinel to the full training
    budget so short experiments still anneal exploration to the end.
    """
    spec = get_agent_type(type_name)
    cls = spec.load_class()
    kwargs = dict(params)
    if "epsilon_decay_steps" in kwargs and kwargs["epsilon_decay_steps"] is None:
        kwargs["epsilon_decay_steps"] = max(1, int(train_episodes) * int(max_steps))
    return cls(env, device=device, seed=int(seed), **kwargs)
