"""Load and validate experiment configs.

A config is a single JSON object describing a comparison matrix of
``environments x agents x seeds``.  Validation is strict and happens up
front: an unknown key, an unsupported backend/reward/agent, or a bad
hyperparameter raises a clear ``ValueError`` before any simulation starts.

Heavy backends (cityflow/torch) are never imported here, so loading and
validating a config — including ``--dry-run`` — is cheap.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.registry import (
    BASELINE_NAMES,
    RESERVED_LABELS,
    validate_agent_params,
)

# Phase-control modes accepted in ``control_mode`` (resolved to classes in
# experiments/envs.py).  Kept as a plain set so config validation needs no
# import of envs.phase_control.
CONTROL_MODES = frozenset(
    {"acyclic", "acyclic_bounded", "cyclic", "resco_cyclic"}
)
BACKENDS = frozenset({"cityflow", "sumo", "moss"})

# Backend -> required path keys in an environment entry.
_BACKEND_PATH_KEYS: dict[str, tuple[str, ...]] = {
    "cityflow": ("config",),
    "sumo": ("config",),
    "moss": ("map_file", "person_file"),
}
_ALL_PATH_KEYS = frozenset({"config", "map_file", "person_file"})

# Cell settings that may appear in top-level ``defaults`` or per-env
# ``overrides``.  Value = default applied when absent everywhere.
SETTING_DEFAULTS: dict[str, Any] = {
    "train_episodes": 10,
    "eval_episodes": 2,
    "max_steps": 360,
    "delta_time": 10,
    "control_mode": "acyclic",
    "global_reward_fn": "queue_length",
    "local_reward_fn": None,
    "global_reward_weight": 1.0,
    "state_features": ["lane_vehicle_count", "lane_waiting", "phase_onehot"],
    "device": "cpu",
    "compare_with": list(BASELINE_NAMES),
    # backend-specific knobs (silently ignored by backends that don't use them)
    "thread_num": 1,  # cityflow
    "gui": False,  # sumo
    "libsumo": False,  # sumo
    "metrics": None,  # explicit metric list (usually derived from reward/state)
    "obs_norm": None,
}

_TOP_LEVEL_KEYS = frozenset(
    {"name", "seeds", "defaults", "environments", "agents", "output_dir", "checkpoint_dir"}
)
_REQUIRED_TOP_LEVEL = frozenset({"name", "seeds", "environments", "agents"})
_ENV_ENTRY_KEYS = frozenset(
    {"id", "backend", "config", "map_file", "person_file", "overrides"}
)
_AGENT_ENTRY_KEYS = frozenset({"id", "type", "params"})


@dataclass(frozen=True)
class EnvSpec:
    id: str
    backend: str
    paths: dict[str, str]
    settings: dict[str, Any]  # fully resolved + validated cell settings


@dataclass(frozen=True)
class AgentSpec:
    id: str
    type: str
    params: dict[str, Any]  # full resolved hyperparameters


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seeds: tuple[int, ...]
    environments: tuple[EnvSpec, ...]
    agents: tuple[AgentSpec, ...]
    defaults: dict[str, Any]
    output_dir: Path
    checkpoint_dir: Path | None
    source_path: Path

    def baselines_for(self, env: EnvSpec) -> tuple[str, ...]:
        return tuple(env.settings["compare_with"])


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_unknown_keys(obj: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise ValueError(
            f"unknown key(s) in {where}: {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_settings(settings: Mapping[str, Any], where: str) -> None:
    """Validate a fully merged settings dict for one environment."""
    for key in ("train_episodes", "eval_episodes", "max_steps", "delta_time", "thread_num"):
        value = settings[key]
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 1,
            f"{where}: '{key}' must be an integer >= 1, got {value!r}",
        )
    for key in ("gui", "libsumo"):
        _require(isinstance(settings[key], bool), f"{where}: '{key}' must be a boolean")
    _require(
        not (settings["gui"] and settings["libsumo"]),
        f"{where}: 'gui' and 'libsumo' cannot both be true (SUMO GUI needs TraCI)",
    )
    _require(
        _is_number(settings["global_reward_weight"]),
        f"{where}: 'global_reward_weight' must be a number",
    )
    _require(isinstance(settings["device"], str), f"{where}: 'device' must be a string")

    _require(
        settings["control_mode"] in CONTROL_MODES,
        f"{where}: unsupported control_mode '{settings['control_mode']}'. "
        f"Supported: {', '.join(sorted(CONTROL_MODES))}",
    )
    # Acyclic controls inject a 5s all-red clearance that must be strictly
    # shorter than delta_time (see envs/phase_control.py); catch the bad combo
    # here with a clear message instead of crashing at env construction.
    if settings["control_mode"] in {"acyclic", "acyclic_bounded"} and settings["delta_time"] <= 5:
        raise ValueError(
            f"{where}: control_mode '{settings['control_mode']}' needs delta_time > 5 "
            f"(it plays a 5s all-red transition), got {settings['delta_time']}"
        )

    # Reward names are validated by reusing rewards.py's own resolvers, so the
    # accepted set never drifts from the implementation.
    from rewards import resolve_local_reward_fn, resolve_reward_fn

    global_reward = settings["global_reward_fn"]
    _require(isinstance(global_reward, str), f"{where}: 'global_reward_fn' must be a string")
    try:
        resolve_reward_fn(global_reward)
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc

    local_reward = settings["local_reward_fn"]
    if local_reward is not None:
        _require(isinstance(local_reward, str), f"{where}: 'local_reward_fn' must be a string or null")
        try:
            resolve_local_reward_fn(local_reward)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc

    state_features = settings["state_features"]
    _require(
        isinstance(state_features, list) and all(isinstance(f, str) for f in state_features),
        f"{where}: 'state_features' must be a list of strings",
    )

    compare_with = settings["compare_with"]
    _require(
        isinstance(compare_with, list) and all(isinstance(b, str) for b in compare_with),
        f"{where}: 'compare_with' must be a list of baseline names",
    )
    unknown_baselines = sorted(set(compare_with) - set(BASELINE_NAMES))
    _require(
        not unknown_baselines,
        f"{where}: unsupported baseline(s) {unknown_baselines}. "
        f"Supported: {', '.join(BASELINE_NAMES)}",
    )

    metrics = settings["metrics"]
    if metrics is not None:
        _require(
            isinstance(metrics, list) and all(isinstance(m, str) for m in metrics),
            f"{where}: 'metrics' must be a list of strings or null",
        )
    obs_norm = settings["obs_norm"]
    if obs_norm is not None:
        _require(isinstance(obs_norm, dict), f"{where}: 'obs_norm' must be an object or null")


def _resolve_settings(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    # deepcopy so list/dict-valued defaults (state_features, compare_with) are
    # never shared between environments or with the module-level defaults.
    merged = copy.deepcopy(SETTING_DEFAULTS)
    merged.update(defaults)
    merged.update(overrides)
    return merged


def _resolve_path(raw: Any, base_dir: Path, where: str) -> str:
    _require(isinstance(raw, str) and raw, f"{where} must be a non-empty path string")
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------


def _parse_env_entry(
    raw: Any,
    index: int,
    defaults: Mapping[str, Any],
    base_dir: Path,
    seen_ids: set[str],
) -> EnvSpec:
    where = f"environments[{index}]"
    _require(isinstance(raw, dict), f"{where} must be an object")
    _check_unknown_keys(raw, _ENV_ENTRY_KEYS, where)

    env_id = raw.get("id")
    _require(isinstance(env_id, str) and env_id, f"{where}: 'id' is required and must be a non-empty string")
    _require(env_id not in seen_ids, f"{where}: duplicate environment id '{env_id}'")
    seen_ids.add(env_id)
    where = f"environments['{env_id}']"

    backend = raw.get("backend")
    _require(
        isinstance(backend, str) and backend in BACKENDS,
        f"{where}: 'backend' must be one of {', '.join(sorted(BACKENDS))}",
    )

    paths: dict[str, str] = {}
    for key in _BACKEND_PATH_KEYS[backend]:
        _require(key in raw, f"{where}: backend '{backend}' requires path key '{key}'")
        paths[key] = _resolve_path(raw[key], base_dir, f"{where}.{key}")

    stray_paths = sorted((set(raw) & _ALL_PATH_KEYS) - set(_BACKEND_PATH_KEYS[backend]))
    _require(
        not stray_paths,
        f"{where}: path key(s) {stray_paths} are not used by backend '{backend}'",
    )

    overrides = raw.get("overrides", {}) or {}
    _require(isinstance(overrides, dict), f"{where}: 'overrides' must be an object")
    _check_unknown_keys(overrides, frozenset(SETTING_DEFAULTS), f"{where}.overrides")

    settings = _resolve_settings(defaults, overrides)
    _validate_settings(settings, where)
    return EnvSpec(id=env_id, backend=backend, paths=paths, settings=settings)


def _parse_agent_entry(raw: Any, index: int, seen_ids: set[str]) -> AgentSpec:
    where = f"agents[{index}]"
    _require(isinstance(raw, dict), f"{where} must be an object")
    _check_unknown_keys(raw, _AGENT_ENTRY_KEYS, where)

    agent_id = raw.get("id")
    _require(isinstance(agent_id, str) and agent_id, f"{where}: 'id' is required and must be a non-empty string")
    _require(agent_id not in seen_ids, f"{where}: duplicate agent id '{agent_id}'")
    _require(
        agent_id not in RESERVED_LABELS,
        f"{where}: agent id '{agent_id}' is reserved for a baseline",
    )
    seen_ids.add(agent_id)
    where = f"agents['{agent_id}']"

    agent_type = raw.get("type")
    _require(isinstance(agent_type, str) and agent_type, f"{where}: 'type' is required")

    params = raw.get("params", {}) or {}
    _require(isinstance(params, dict), f"{where}: 'params' must be an object")
    resolved = validate_agent_params(agent_type, params)
    return AgentSpec(id=agent_id, type=agent_type.lower(), params=resolved)


def load_config(path: str | Path) -> ExperimentConfig:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"config not found: {resolved}")

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in config {resolved}: {exc}") from exc
    _require(isinstance(payload, dict), "config JSON must be an object")

    _check_unknown_keys(payload, _TOP_LEVEL_KEYS, "config")
    missing = sorted(_REQUIRED_TOP_LEVEL - set(payload))
    _require(not missing, f"missing required key(s) in config: {', '.join(missing)}")

    name = payload["name"]
    _require(isinstance(name, str) and name, "config: 'name' must be a non-empty string")
    # name is interpolated into the output path; keep it a single path segment.
    _require(
        "/" not in name and "\\" not in name and name not in {".", ".."},
        "config: 'name' must not contain path separators",
    )

    raw_seeds = payload["seeds"]
    _require(
        isinstance(raw_seeds, list) and raw_seeds,
        "config: 'seeds' must be a non-empty list of integers",
    )
    _require(
        all(isinstance(s, int) and not isinstance(s, bool) for s in raw_seeds),
        "config: 'seeds' must contain only integers",
    )
    seeds = tuple(int(s) for s in raw_seeds)
    _require(len(set(seeds)) == len(seeds), "config: 'seeds' must not contain duplicates")

    defaults = payload.get("defaults", {}) or {}
    _require(isinstance(defaults, dict), "config: 'defaults' must be an object")
    _check_unknown_keys(defaults, frozenset(SETTING_DEFAULTS), "config.defaults")

    base_dir = resolved.parent

    raw_envs = payload["environments"]
    _require(isinstance(raw_envs, list) and raw_envs, "config: 'environments' must be a non-empty list")
    seen_env_ids: set[str] = set()
    environments = tuple(
        _parse_env_entry(raw, i, defaults, base_dir, seen_env_ids)
        for i, raw in enumerate(raw_envs)
    )

    raw_agents = payload["agents"]
    _require(isinstance(raw_agents, list) and raw_agents, "config: 'agents' must be a non-empty list")
    seen_agent_ids: set[str] = set()
    agents = tuple(_parse_agent_entry(raw, i, seen_agent_ids) for i, raw in enumerate(raw_agents))

    raw_out = payload.get("output_dir") or "output/experiments"
    _require(isinstance(raw_out, str) and raw_out, "config: 'output_dir' must be a non-empty string")
    output_dir = (Path(raw_out) / name).resolve()

    raw_checkpoint_dir = payload.get("checkpoint_dir")
    checkpoint_dir = None
    if raw_checkpoint_dir is not None:
        _require(
            isinstance(raw_checkpoint_dir, str) and raw_checkpoint_dir,
            "config: 'checkpoint_dir' must be a non-empty string when provided",
        )
        checkpoint_dir = (Path(raw_checkpoint_dir) / name).resolve()

    return ExperimentConfig(
        name=name,
        seeds=seeds,
        environments=environments,
        agents=agents,
        defaults=dict(defaults),
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        source_path=resolved,
    )
