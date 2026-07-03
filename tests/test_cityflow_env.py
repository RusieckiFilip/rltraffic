"""Tests for the CityFlow gymnasium environment."""

from pathlib import Path
import json

import numpy as np
import pytest

from utils.cityflow_utils import parse_roadnet

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "sim" / "config_1x1.json"
ROADNET_PATH = ROOT_DIR / "scenarios" / "aigen_1x1" / "roadnet_1x1.json"


# ── Roadnet parsing tests ────────────────────────────────────────────


class TestParseRoadnet:
    def test_parses_intersections(self):
        info = parse_roadnet(ROADNET_PATH)
        # Only the central intersection is non-virtual
        assert len(info.intersections) == 1
        assert info.intersections[0].id == "intersection_0_0"

    def test_phase_count(self):
        info = parse_roadnet(ROADNET_PATH)
        ix = info.intersections[0]
        assert ix.num_phases == 4

    def test_incoming_lanes(self):
        info = parse_roadnet(ROADNET_PATH)
        ix = info.intersections[0]
        # 4 roads × 2 lanes each → 8 incoming lanes
        assert len(ix.incoming_lanes) == 8

    def test_lane_ids(self):
        info = parse_roadnet(ROADNET_PATH)
        # 8 roads × 2 lanes = 16 total lanes
        assert len(info.lane_ids) == 16

    def test_skips_gt_virtual_intersections(self, tmp_path):
        roadnet = {
            "roads": [
                {
                    "id": "r_in",
                    "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
                    "lanes": [{"maxSpeed": 10}],
                },
                {
                    "id": "r_out",
                    "points": [{"x": 1, "y": 0}, {"x": 2, "y": 0}],
                    "lanes": [{"maxSpeed": 10}],
                },
            ],
            "intersections": [
                {
                    "id": "controlled",
                    "virtual": False,
                    "gt_virtual": False,
                    "roadLinks": [
                        {
                            "startRoad": "r_in",
                            "endRoad": "r_out",
                            "laneLinks": [
                                {"startLaneIndex": 0, "endLaneIndex": 0}
                            ],
                        }
                    ],
                    "trafficLight": {
                        "lightphases": [
                            {"time": 10, "availableRoadLinks": [0]}
                        ]
                    },
                },
                {
                    "id": "boundary",
                    "virtual": False,
                    "gt_virtual": True,
                    "roadLinks": [],
                    "trafficLight": {
                        "lightphases": [
                            {"time": 30, "availableRoadLinks": []}
                        ]
                    },
                },
            ],
        }
        path = tmp_path / "roadnet.json"
        path.write_text(json.dumps(roadnet), encoding="utf-8")

        info = parse_roadnet(path)

        assert [ix.id for ix in info.intersections] == ["controlled"]


# ── Environment tests (require cityflow installed) ───────────────────


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _cityflow_available(), reason="cityflow not installed")
class TestCityFlowEnv:
    def _make_env(self, **kwargs):
        from envs.cityflow_env import CityFlowEnv

        return CityFlowEnv(
            cityflow_config_path=str(CONFIG_PATH), **kwargs
        )

    def test_step_returns_correct_tuple(self):
        env = self._make_env()
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 4
        reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_action_space_is_discrete_for_single_intersection(self):
        from gymnasium.spaces import Discrete
        env = self._make_env()
        env.reset()
        assert isinstance(env.action_space, Discrete)
        env.close()

    def test_episode_truncation(self):
        env = self._make_env(max_steps=5, delta_time=10)
        env.reset()
        truncated = False
        for _ in range(10):
            _, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            if terminated or truncated:
                break
        assert truncated
        env.close()

    def test_different_reward_functions(self):
        for name in ["queue_length", "average_travel_time", "pressure", "throughput", "combined"]:
            env = self._make_env(global_reward_fn=name)
            env.reset()
            reward, _, _, _ = env.step(env.action_space.sample())
            assert isinstance(reward, float), f"global_reward_fn={name} failed"
            env.close()

    def test_info_contains_intersections_state_schema(self):
        env = self._make_env()
        info = env.reset()

        assert "intersections" in info
        assert isinstance(info["intersections"], dict)

        ix = env.intersections[0]
        ix_payload = info["intersections"][ix.id]
        assert "state" in ix_payload
        assert "avail_actions" in ix_payload

        env.close()
