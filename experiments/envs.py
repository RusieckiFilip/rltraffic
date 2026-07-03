"""Build simulator environments from an :class:`EnvSpec`.

This is the only place that imports the heavy backend engines, and it does
so lazily (inside the functions) so config loading and ``--dry-run`` stay
import-light.  Every backend is constructed through the shared
:class:`envs.base_traffic_env.BaseTrafficEnv` contract, so the runner never
needs to branch on the backend.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from experiments.config import EnvSpec


def phase_control_cls(control_mode: str) -> type[Any]:
    """Map a ``control_mode`` string to its phase-control class."""
    from envs.phase_control import (
        AcyclicBoundedPhases,
        AcyclicPhases,
        CyclicPhases,
        RescoCyclicPhases,
    )

    mapping = {
        "acyclic": AcyclicPhases,
        "acyclic_bounded": AcyclicBoundedPhases,
        "cyclic": CyclicPhases,
        "resco_cyclic": RescoCyclicPhases,
    }
    try:
        return mapping[control_mode]
    except KeyError as exc:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unsupported control_mode: {control_mode}") from exc


def backend_ready(
    backend: str,
    paths: dict[str, str],
    libsumo: bool = False,
) -> tuple[bool, str]:
    """Return ``(ready, reason)`` for a backend without constructing the env.

    Checks that the scenario files exist and the backend engine can be
    imported, so an unavailable backend yields a clean *skip* instead of a
    crash. ``libsumo`` selects which SUMO engine to probe so the readiness
    check matches what :func:`make_env` will actually use.
    """
    for key, value in paths.items():
        if not Path(value).exists():
            return False, f"missing {key}: {value}"

    if backend == "cityflow":
        try:
            import cityflow  # noqa: F401
        except Exception as exc:  # ImportError or a broken native build
            return False, f"cityflow import failed: {exc}"
        return True, ""

    if backend == "sumo":
        # libsumo is in-process and needs no `sumo` binary on PATH.
        if not libsumo and shutil.which("sumo") is None:
            return False, "sumo binary not found in PATH"
        engine = "libsumo" if libsumo else "traci"
        try:
            __import__(engine)
        except Exception as exc:
            return False, f"{engine} import failed: {exc}"
        return True, ""

    if backend == "moss":
        try:
            import moss  # noqa: F401
        except Exception as exc:
            return False, f"moss import failed: {exc}"
        return True, ""

    return False, f"unknown backend: {backend}"


def make_env(env_spec: EnvSpec) -> Any:
    """Construct and return a fresh env instance for *env_spec*.

    Callers own the instance and must call ``env.close()`` when done.
    """
    settings = env_spec.settings
    common: dict[str, Any] = {
        "max_steps": settings["max_steps"],
        "delta_time": settings["delta_time"],
        "global_reward_fn": settings["global_reward_fn"],
        "local_reward_fn": settings["local_reward_fn"],
        "global_reward_weight": settings["global_reward_weight"],
        "phase_control_cls": phase_control_cls(settings["control_mode"]),
        "state_features": settings["state_features"],
    }
    if settings["metrics"] is not None:
        common["metrics"] = settings["metrics"]
    if settings["obs_norm"] is not None:
        common["obs_norm"] = settings["obs_norm"]

    backend = env_spec.backend
    if backend == "cityflow":
        from envs.cityflow_env import CityFlowEnv

        return CityFlowEnv(
            cityflow_config_path=env_spec.paths["config"],
            thread_num=settings["thread_num"],
            **common,
        )

    if backend == "sumo":
        from envs.sumo_env import SumoEnv

        return SumoEnv(
            sumocfg_path=env_spec.paths["config"],
            gui=settings["gui"],
            libsumo=settings["libsumo"],
            **common,
        )

    if backend == "moss":
        from envs.moss_env import MossEnv

        # MossEnv takes the map / person protobufs positionally.
        return MossEnv(
            env_spec.paths["map_file"],
            env_spec.paths["person_file"],
            **common,
        )

    raise ValueError(f"unsupported backend: {backend}")  # pragma: no cover
