from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from agent.DQNAgent import IDQNAgent
from envs.phase_control import CyclicPhases

ROOT_DIR = Path(__file__).resolve().parent.parent
CITYFLOW_CONFIG = ROOT_DIR / "configs" / "sim" / "config_1x1.json"
SUMO_CONFIG = ROOT_DIR / "scenarios" / "cologne1" / "cologne1.sumocfg"


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
        return True
    except ImportError:
        return False


def _sumo_available() -> bool:
    try:
        import traci  # noqa: F401
    except ImportError:
        return False
    return shutil.which("sumo") is not None


@pytest.mark.skipif(not _cityflow_available(), reason="cityflow not installed")
def test_dqn_cityflow_uses_queue_length_reward() -> None:
    from envs.cityflow_env import CityFlowEnv

    env = CityFlowEnv(
        cityflow_config_path=str(CITYFLOW_CONFIG),
        global_reward_fn="queue_length",
        max_steps=20,
        delta_time=1,
        phase_control_cls=CyclicPhases,
    )
    agent = IDQNAgent(env)

    info = env.reset()
    action = agent.next_action(info)
    reward, terminated, truncated, next_info = env.step(action)
    agent.observe(next_info, reward, bool(terminated), bool(truncated))

    assert isinstance(reward, float)
    assert reward <= 0.0
    assert np.isfinite(reward)

    env.close()


@pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not available")
def test_dqn_sumo_uses_queue_length_reward() -> None:
    from envs.sumo_env import SumoEnv

    env = SumoEnv(
        sumocfg_path=str(SUMO_CONFIG),
        global_reward_fn="queue_length",
        max_steps=20,
        delta_time=1,
        gui=False,
        phase_control_cls=CyclicPhases,
    )
    agent = IDQNAgent(env)

    info = env.reset()
    action = agent.next_action(info)
    reward, terminated, truncated, next_info = env.step(action)
    agent.observe(next_info, reward, bool(terminated), bool(truncated))

    assert isinstance(reward, float)
    assert reward <= 0.0
    assert np.isfinite(reward)

    env.close()
