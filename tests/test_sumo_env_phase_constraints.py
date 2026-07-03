"""Phase duration and available-action tests for SumoEnv."""

from pathlib import Path
import shutil

import numpy as np
import pytest

from envs.phase_control import (
    TRANSITION_PHASE_MAX_DURATION,
    AcyclicBoundedPhases,
    AcyclicPhases,
    CyclicPhases,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SUMOCFG_PATH = ROOT_DIR / "scenarios" / "cologne1" / "cologne1.sumocfg"


def _sumo_available() -> bool:
    try:
        import traci  # noqa: F401
    except ImportError:
        return False
    return shutil.which("sumo") is not None


@pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not available")
class TestSumoPhaseConstraints:
    def _make_env(self, **kwargs):
        from envs.sumo_env import SumoEnv

        params = {
            "sumocfg_path": str(SUMOCFG_PATH),
            "delta_time": 1,
            "max_steps": 20,
            "gui": False,
        }
        params.update(kwargs)
        return SumoEnv(**params)

    def _bounds_for_env(self, env, min_v: int, max_v: int) -> np.ndarray:
        max_phases = max(ix.num_phases for ix in env.intersections)
        bounds = np.zeros((len(env.intersections), max_phases, 2), dtype=np.int64)
        bounds[..., 0] = min_v
        bounds[..., 1] = max_v
        return bounds

    def _phase_from_info(self, info, ix_id: str) -> int:
        return int(info["intersections"][ix_id]["current_phase"])

    @staticmethod
    def _n_green_phases(ix) -> int:
        """Greens by the acyclic controls' rule: file duration > 5 s."""
        return sum(
            1
            for duration in ix.phase_durations
            if float(duration) > TRANSITION_PHASE_MAX_DURATION
        )

    def test_bounds_validation(self):
        env = self._make_env(delta_time=10)
        env.reset()

        with pytest.raises(ValueError):
            env.set_phase_durations(np.zeros((1, 4), dtype=np.int64))

        with pytest.raises(ValueError):
            env.set_phase_durations(np.zeros((len(env.intersections) + 1, 4, 2), dtype=np.int64))

        with pytest.raises(ValueError):
            bad = self._bounds_for_env(env, 3, 2)
            env.set_phase_durations(bad)

        env.close()

    def test_cyclic_action_space_and_mask_semantics(self):
        env = self._make_env(phase_control_cls=CyclicPhases)
        env.reset()

        if len(env.intersections) == 1:
            from gymnasium.spaces import Discrete

            assert isinstance(env.action_space, Discrete)
            assert env.action_space.n == 2
        else:
            from gymnasium.spaces import MultiDiscrete

            assert isinstance(env.action_space, MultiDiscrete)
            assert np.all(env.action_space.nvec == 2)

        env.set_phase_durations(self._bounds_for_env(env, 2, 100))
        info = env._get_info()
        ix0 = env.intersections[0]
        ix0_info = info["intersections"][ix0.id]
        assert ix0_info["avail_actions"] == [0]

        env.close()

    def test_cyclic_keep_prevents_sumo_auto_advance(self):
        env = self._make_env(phase_control_cls=CyclicPhases, delta_time=10)
        env.reset()

        ix0 = env.intersections[0]
        hold_all = np.zeros(len(env.intersections), dtype=np.int64)

        try:
            for _ in range(3):
                env.step(hold_all)
                assert env._phase_controls[0].current_phase() == 0
                assert env._sumo.trafficlight.getPhase(ix0.id) == 0
        finally:
            env.close()

    def test_non_cyclic_action_space_uses_green_phase_count(self):
        env = self._make_env(phase_control_cls=AcyclicPhases, delta_time=10)
        env.reset()

        if len(env.intersections) == 1:
            from gymnasium.spaces import Discrete

            assert isinstance(env.action_space, Discrete)
            assert env.action_space.n == self._n_green_phases(
                env.intersections[0]
            )
        else:
            from gymnasium.spaces import MultiDiscrete

            assert isinstance(env.action_space, MultiDiscrete)
            expected = np.array(
                [self._n_green_phases(ix) for ix in env.intersections],
                dtype=np.int64,
            )
            assert np.array_equal(env.action_space.nvec, expected)

        env.close()

    def test_invalid_action_raises_when_unavailable(self):
        env = self._make_env(
            phase_control_cls=AcyclicBoundedPhases, delta_time=10
        )
        env.reset()

        bounds = self._bounds_for_env(env, 30, 100)
        env.set_phase_durations(bounds)

        # Below the min green time only the current action is available.
        info = env._get_info()
        actions = np.array(
            [
                info["intersections"][ix.id]["avail_actions"][0]
                for ix in env.intersections
            ],
            dtype=np.int64,
        )
        ix0 = env.intersections[0]
        avail0 = info["intersections"][ix0.id]["avail_actions"]
        n_actions0 = env._phase_controls[0].action_count(ix0.num_phases)
        actions[0] = next(
            a for a in range(n_actions0) if a not in avail0
        )

        with pytest.raises(ValueError):
            env.step(actions)

        env.close()

    def test_cyclic_max_duration_forces_action_one_only(self):
        env = self._make_env(phase_control_cls=CyclicPhases)
        env.reset()

        env.set_phase_durations(self._bounds_for_env(env, 0, 1))
        hold_all = np.zeros(len(env.intersections), dtype=np.int64)

        _, _, _, info = env.step(hold_all)
        ix0 = env.intersections[0]
        ix0_info = info["intersections"][ix0.id]
        assert ix0_info["avail_actions"] == [1]

        with pytest.raises(ValueError):
            env.step(hold_all)

        env.close()

    def test_info_contains_intersections_state_schema(self):
        env = self._make_env(phase_control_cls=AcyclicPhases, delta_time=10)
        info = env.reset()

        ix0 = env.intersections[0]
        ix0_info = info["intersections"][ix0.id]
        assert "state" in ix0_info
        assert "avail_actions" in ix0_info

        n_in = len(ix0.incoming_lanes)
        expected_dim = 2 * n_in + ix0.num_phases
        assert len(ix0_info["state"]) == expected_dim

        env.close()

    def test_reset_supports_multiple_episodes(self):
        env = self._make_env(
            phase_control_cls=AcyclicPhases, max_steps=5, delta_time=10
        )
        try:
            for _ in range(2):
                info = env.reset()
                hold_all = np.zeros(len(env.intersections), dtype=np.int64)
                for _ in range(3):
                    _, _, _, info = env.step(hold_all)
                assert "intersections" in info
        finally:
            env.close()
