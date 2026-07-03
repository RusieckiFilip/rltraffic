"""Phase duration and cyclic-action tests for CityFlowEnv."""

from pathlib import Path

import numpy as np
import pytest

from envs.phase_control import AcyclicBoundedPhases, AcyclicPhases, CyclicPhases

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "sim" / "config_1x1.json"


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _cityflow_available(), reason="cityflow not installed")
class TestCityFlowPhaseConstraints:
    def _make_env(self, **kwargs):
        from envs.cityflow_env import CityFlowEnv

        params = {
            "cityflow_config_path": str(CONFIG_PATH),
            "delta_time": 1,
        }
        params.update(kwargs)

        return CityFlowEnv(
            **params,
        )

    def _phase_from_info(self, info, ix_id: str) -> int:
        return int(info["intersections"][ix_id]["current_phase"])

    def _time_from_info(self, info, ix_id: str) -> int:
        return int(info["intersections"][ix_id]["time_in_phase"])

    def test_bounds_validation(self):
        env = self._make_env(delta_time=10)
        env.reset()

        with pytest.raises(ValueError):
            env.set_phase_durations(np.zeros((1, 4), dtype=np.int64))

        with pytest.raises(ValueError):
            env.set_phase_durations(np.zeros((2, 4, 2), dtype=np.int64))

        with pytest.raises(ValueError):
            bad = np.zeros((1, 4, 2), dtype=np.int64)
            bad[..., 0] = 3
            bad[..., 1] = 2
            env.set_phase_durations(bad)

        env.close()

    def test_non_cyclic_min_duration_blocks_and_then_allows(self):
        env = self._make_env(
            phase_control_cls=AcyclicBoundedPhases, delta_time=10
        )
        env.reset()

        ix = env.intersections[0]
        bounds = np.zeros((1, ix.num_phases, 2), dtype=np.int64)
        bounds[..., 0] = 15
        bounds[..., 1] = 100
        env.set_phase_durations(bounds)
        info = env._get_info()

        start_phase = self._phase_from_info(info, ix.id)
        desired = (start_phase + 1) % ix.num_phases

        assert info["intersections"][ix.id]["avail_actions"] == [start_phase]

        _, _, _, info = env.step(np.array([start_phase], dtype=np.int64))
        assert info["intersections"][ix.id]["avail_actions"] == [start_phase]

        _, _, _, info = env.step(np.array([start_phase], dtype=np.int64))
        assert desired in info["intersections"][ix.id]["avail_actions"]

        env.step(np.array([desired], dtype=np.int64))
        post_info = env._get_info()
        assert self._phase_from_info(post_info, ix.id) == desired

        env.close()

    def test_non_cyclic_invalid_action_raises_when_unavailable(self):
        env = self._make_env(
            phase_control_cls=AcyclicBoundedPhases, delta_time=10
        )
        env.reset()

        ix = env.intersections[0]
        bounds = np.zeros((1, ix.num_phases, 2), dtype=np.int64)
        bounds[..., 0] = 30
        bounds[..., 1] = 100
        env.set_phase_durations(bounds)

        info = env._get_info()
        start_phase = self._phase_from_info(info, ix.id)
        desired = (start_phase + 1) % ix.num_phases

        with pytest.raises(ValueError):
            env.step(np.array([desired], dtype=np.int64))

        env.close()

    def test_non_cyclic_max_duration_disallows_stay_without_auto_switch(self):
        env = self._make_env(
            phase_control_cls=AcyclicBoundedPhases, delta_time=10
        )
        info = env.reset()

        ix = env.intersections[0]
        bounds = np.zeros((1, ix.num_phases, 2), dtype=np.int64)
        bounds[..., 0] = 0
        bounds[..., 1] = 20
        env.set_phase_durations(bounds)

        start_phase = self._phase_from_info(info, ix.id)

        assert start_phase in info["intersections"][ix.id]["avail_actions"]

        _, _, _, info = env.step(np.array([start_phase], dtype=np.int64))
        assert self._phase_from_info(info, ix.id) == start_phase
        assert start_phase in info["intersections"][ix.id]["avail_actions"]

        _, _, _, info = env.step(np.array([start_phase], dtype=np.int64))
        assert self._phase_from_info(info, ix.id) == start_phase
        assert start_phase not in info["intersections"][ix.id]["avail_actions"]

        with pytest.raises(ValueError):
            env.step(np.array([start_phase], dtype=np.int64))

        expected = (start_phase + 1) % ix.num_phases
        env.step(np.array([expected], dtype=np.int64))
        info = env._get_info()
        assert self._phase_from_info(info, ix.id) == expected

        env.close()

    def test_cyclic_action_space_and_semantics(self):
        from gymnasium.spaces import Discrete

        env = self._make_env(phase_control_cls=CyclicPhases)
        env.reset()

        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 2

        ix = env.intersections[0]
        bounds = np.zeros((1, ix.num_phases, 2), dtype=np.int64)
        bounds[..., 0] = 2
        bounds[..., 1] = 100
        env.set_phase_durations(bounds)
        info = env._get_info()

        start_phase = self._phase_from_info(info, ix.id)
        assert info["intersections"][ix.id]["avail_actions"] == [0]

        with pytest.raises(ValueError):
            env.step(np.array([1], dtype=np.int64))

        _, _, _, info = env.step(np.array([0], dtype=np.int64))
        assert info["intersections"][ix.id]["avail_actions"] == [0]

        _, _, _, info = env.step(np.array([0], dtype=np.int64))
        assert info["intersections"][ix.id]["avail_actions"] == [0, 1]

        env.step(np.array([1], dtype=np.int64))
        expected = (start_phase + 1) % ix.num_phases
        info = env._get_info()
        assert self._phase_from_info(info, ix.id) == expected

        env.close()

    def test_cyclic_info_action_applied_reports_success(self):
        env = self._make_env(phase_control_cls=CyclicPhases)
        env.reset()

        ix = env.intersections[0]
        bounds = np.zeros((1, ix.num_phases, 2), dtype=np.int64)
        bounds[..., 0] = 2
        bounds[..., 1] = 100
        env.set_phase_durations(bounds)

        env.step(np.array([0], dtype=np.int64))
        env.step(np.array([0], dtype=np.int64))
        _, _, _, info = env.step(np.array([1], dtype=np.int64))
        assert info["intersections"][ix.id]["action_applied"]

        env.close()

    def test_non_cyclic_info_action_applied_reports_success(self):
        env = self._make_env(
            phase_control_cls=AcyclicBoundedPhases, delta_time=10
        )
        env.reset()

        ix = env.intersections[0]
        bounds = np.zeros((1, ix.num_phases, 2), dtype=np.int64)
        bounds[..., 0] = 15
        bounds[..., 1] = 100
        env.set_phase_durations(bounds)

        info = env._get_info()
        start_phase = self._phase_from_info(info, ix.id)
        desired = (start_phase + 1) % ix.num_phases

        env.step(np.array([start_phase], dtype=np.int64))
        env.step(np.array([start_phase], dtype=np.int64))
        _, _, _, info = env.step(np.array([desired], dtype=np.int64))
        assert info["intersections"][ix.id]["action_applied"]

        env.close()

    def test_cyclic_max_duration_forces_available_action_one_only(self):
        env = self._make_env(phase_control_cls=CyclicPhases, delta_time=1)
        env.reset()

        ix = env.intersections[0]
        bounds = np.zeros((1, ix.num_phases, 2), dtype=np.int64)
        bounds[..., 0] = 0
        bounds[..., 1] = 1
        env.set_phase_durations(bounds)
        info = env._get_info()

        assert info["intersections"][ix.id]["avail_actions"] == [0, 1]

        _, _, _, info = env.step(np.array([0], dtype=np.int64))
        assert info["intersections"][ix.id]["avail_actions"] == [1]

        with pytest.raises(ValueError):
            env.step(np.array([0], dtype=np.int64))

        env.step(np.array([1], dtype=np.int64))
        info = env._get_info()
        assert self._phase_from_info(info, ix.id) == 1

        env.close()

    def test_equal_min_max_allows_only_switch_in_cyclic(self):
        env = self._make_env(phase_control_cls=CyclicPhases, delta_time=1)
        env.reset()

        ix = env.intersections[0]
        bounds = np.zeros((1, ix.num_phases, 2), dtype=np.int64)
        # Exact-duration phases: both constraints hit at the same second.
        bounds[..., 0] = 1
        bounds[..., 1] = 1
        env.set_phase_durations(bounds)
        info = env._get_info()

        assert info["intersections"][ix.id]["avail_actions"] == [0]
        env.step(np.array([0], dtype=np.int64))
        _, _, _, info = env.step(np.array([1], dtype=np.int64))
        assert info["intersections"][ix.id]["avail_actions"] == [1]

        env.close()

    def test_reset_clears_phase_timers(self):
        env = self._make_env(phase_control_cls=AcyclicPhases, delta_time=10)
        env.reset()

        ix = env.intersections[0]
        bounds = np.zeros((1, ix.num_phases, 2), dtype=np.int64)
        bounds[..., 0] = 10
        bounds[..., 1] = 100
        env.set_phase_durations(bounds)

        info = env._get_info()
        env.step(
            np.array([self._phase_from_info(info, ix.id)], dtype=np.int64)
        )
        info = env._get_info()
        assert self._time_from_info(info, ix.id) > 0

        info = env.reset()
        assert self._time_from_info(info, ix.id) == 0
        assert self._phase_from_info(info, ix.id) == 0

        env.close()

    def test_phase_specific_min_durations_are_used(self):
        env = self._make_env(
            phase_control_cls=AcyclicBoundedPhases, delta_time=10
        )
        env.reset()

        ix = env.intersections[0]
        bounds = np.zeros((1, ix.num_phases, 2), dtype=np.int64)
        bounds[..., 0] = 0
        bounds[..., 1] = 100

        # Phase 0 needs 30s (three decision steps), phase 1 needs 0s.
        bounds[0, 0, 0] = 30
        bounds[0, 1, 0] = 0
        env.set_phase_durations(bounds)
        info = env._get_info()

        assert info["intersections"][ix.id]["avail_actions"] == [0]

        # Try switching out of phase 0 too early (must raise).
        with pytest.raises(ValueError):
            env.step(np.array([1], dtype=np.int64))

        env.step(np.array([0], dtype=np.int64))
        env.step(np.array([0], dtype=np.int64))
        _, _, _, info = env.step(np.array([0], dtype=np.int64))
        assert info["intersections"][ix.id]["avail_actions"] == [0, 1, 2, 3]

        env.step(np.array([1], dtype=np.int64))
        info = env._get_info()
        assert self._phase_from_info(info, ix.id) == 1

        # Switching out of phase 1 should be immediate (min=0).
        env.step(np.array([2], dtype=np.int64))
        info = env._get_info()
        assert self._phase_from_info(info, ix.id) == 2

        env.close()

    def test_info_contains_per_intersection_state(self):
        env = self._make_env(phase_control_cls=AcyclicPhases, delta_time=10)
        info = env.reset()

        ix = env.intersections[0]
        ix_info = info["intersections"][ix.id]
        assert "state" in ix_info
        assert "avail_actions" in ix_info

        n_in = len(ix.incoming_lanes)
        expected_dim = 2 * n_in + ix.num_phases
        assert len(ix_info["state"]) == expected_dim

        env.close()
