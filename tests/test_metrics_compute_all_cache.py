"""compute_all() must memoise within a step and reset after update()."""
from __future__ import annotations

from metrics.base import BaseMetrics, register


class _CountingMetrics(BaseMetrics):
    """Backend-free metrics impl that counts compute() invocations."""

    def __init__(self) -> None:
        self.compute_calls = 0
        super().__init__(
            engine=None,
            intersections=[],
            metric_names=["q"],
            delta_time=1.0,
        )

    @register("q")
    def _q(self) -> float:
        self.compute_calls += 1
        return 1.23

    def lane_waiting_vehicle_count(self) -> dict[str, int]:
        return {}

    def lane_vehicle_count(self) -> dict[str, int]:
        return {}

    def _init_episode_state(self) -> None:
        return None

    def _run_step_hooks(self) -> None:
        return None

    def _snapshot_pre_action(self) -> None:
        return None


def test_compute_all_caches_within_a_step() -> None:
    m = _CountingMetrics()
    first = m.compute_all()
    second = m.compute_all()
    assert first == {"q": 1.23}
    assert first == second
    assert m.compute_calls == 1


def test_update_resets_compute_all_cache() -> None:
    m = _CountingMetrics()
    m.compute_all()
    m.update()
    m.compute_all()
    assert m.compute_calls == 2


def test_compute_all_returns_independent_dict() -> None:
    m = _CountingMetrics()
    first = m.compute_all()
    first["q"] = 999.0
    second = m.compute_all()
    assert second == {"q": 1.23}
