"""Callable reward_fn must specify metrics explicitly."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
CITYFLOW_CONFIG = ROOT_DIR / "configs" / "sim" / "config_1x1.json"


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _cityflow_available(), reason="cityflow not installed")
def test_callable_reward_without_metrics_raises() -> None:
    from envs.cityflow_env import CityFlowEnv

    def my_reward(_metrics: dict[str, Any]) -> float:
        return 0.0

    with pytest.raises(ValueError, match="metrics="):
        CityFlowEnv(
            cityflow_config_path=str(CITYFLOW_CONFIG),
            global_reward_fn=my_reward,
            max_steps=2,
            delta_time=10,
        )


@pytest.mark.skipif(not _cityflow_available(), reason="cityflow not installed")
def test_callable_reward_with_metrics_is_accepted() -> None:
    from envs.cityflow_env import CityFlowEnv

    seen: dict[str, Any] = {}

    def my_reward(metrics: dict[str, Any]) -> float:
        seen.update(metrics)
        return -float(
            metrics.get(
                "number_of_all_halting_vehicles_for_the_last_time_step_in_simulation",
                0.0,
            )
        )

    env = CityFlowEnv(
        cityflow_config_path=str(CITYFLOW_CONFIG),
        global_reward_fn=my_reward,
        metrics=[
            "number_of_all_halting_vehicles_for_the_last_time_step_in_simulation",
        ],
        max_steps=2,
        delta_time=10,
    )
    env.reset()
    reward, _, _, _ = env.step(env.action_space.sample())
    assert "number_of_all_halting_vehicles_for_the_last_time_step_in_simulation" in seen
    assert isinstance(reward, float)
    env.close()
