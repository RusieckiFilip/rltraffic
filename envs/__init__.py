from envs.base_traffic_env import BaseTrafficEnv

try:
    from envs.cityflow_env import CityFlowEnv
except ModuleNotFoundError as exc:
    if exc.name != "cityflow":
        raise
    CityFlowEnv = None  # type: ignore[assignment]

# ``moss_env`` and ``sumo_env`` only import their third-party engines
# lazily (inside ``_init_simulator``), so it's safe to import the env
# classes here even when MOSS / SUMO aren't installed.
from envs.moss_env import MossEnv
from envs.sumo_env import SumoEnv


def register_envs() -> None:
    """Deprecated compatibility shim; envs are constructed directly now."""

__all__ = ["BaseTrafficEnv", "CityFlowEnv", "MossEnv", "SumoEnv", "register_envs"]
