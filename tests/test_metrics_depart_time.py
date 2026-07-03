"""Newly-departed vehicles use the window midpoint as depart_time.

t - delta_time was the old behaviour: it assumed every fresh vehicle
appeared at the very start of the (t - delta, t] window. Using the
midpoint halves the bias on average.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # noqa: F401  – keeps pyflakes happy in editor envs

from metrics.cityflow import CityFlowMetrics
from metrics.sumo import SumoMetrics


@dataclass
class _Ix:
    id: str = "ix"
    incoming_lanes: list[str] = None
    outgoing_lanes: list[str] = None

    def __post_init__(self) -> None:
        if self.incoming_lanes is None:
            self.incoming_lanes = []
        if self.outgoing_lanes is None:
            self.outgoing_lanes = []


class _FakeCityFlowEngine:
    def __init__(self) -> None:
        self.time = 0.0

    def get_vehicles(self, include_waiting: bool = False) -> list[str]:
        return ["v0"] if self.time > 0.0 else []

    def get_vehicle_speed(self) -> dict[str, float]:
        return {"v0": 5.0} if self.time > 0.0 else {}

    def get_vehicle_info(self, vid: str) -> dict[str, str]:
        return {"route": "", "road": ""}

    def get_current_time(self) -> float:
        return self.time

    def get_lane_vehicle_count(self) -> dict[str, int]:
        return {}

    def get_lane_waiting_vehicle_count(self) -> dict[str, int]:
        return {}

    def get_lane_vehicles(self) -> dict[str, list[str]]:
        return {}

    def get_average_travel_time(self) -> float:
        return 0.0


class _FakeRoadnet:
    road_lengths: dict[str, float] = {}
    road_max_speeds: dict[str, float] = {}


def test_cityflow_metrics_uses_midpoint_depart_time() -> None:
    engine = _FakeCityFlowEngine()
    m = CityFlowMetrics(
        engine=engine,
        intersections=[_Ix()],
        metric_names=[],
        delta_time=10.0,
        roadnet=_FakeRoadnet(),
    )
    engine.time = 10.0
    m.update()
    assert m._episode["depart_time"]["v0"] == 10.0 - 10.0 / 2.0


class _FakeSumoVehicle:
    def getIDList(self) -> list[str]:
        return ["v0"]

    def getSpeed(self, _vid: str) -> float:
        return 5.0

    def getAccumulatedWaitingTime(self, _vid: str) -> float:
        return 0.0

    def getDistance(self, _vid: str) -> float:
        return 1.0

    def getAllowedSpeed(self, _vid: str) -> float:
        return 13.9

    def getRoute(self, _vid: str) -> list[str]:
        return []


class _FakeSumoSimulation:
    def __init__(self) -> None:
        self.time = 0.0

    def getTime(self) -> float:
        return self.time

    def getArrivedIDList(self) -> list[str]:
        return []


class _FakeSumoLane:
    def getLastStepHaltingNumber(self, _lid: str) -> int:
        return 0

    def getLastStepVehicleNumber(self, _lid: str) -> int:
        return 0


class _FakeSumoEngine:
    def __init__(self) -> None:
        self.vehicle = _FakeSumoVehicle()
        self.simulation = _FakeSumoSimulation()
        self.lane = _FakeSumoLane()


def test_sumo_metrics_uses_midpoint_depart_time() -> None:
    engine = _FakeSumoEngine()
    m = SumoMetrics(
        engine=engine,
        intersections=[_Ix()],
        metric_names=["average_travel_time"],
        delta_time=10.0,
        roadnet=_FakeRoadnet(),
    )
    engine.simulation.time = 10.0
    m.update()
    assert m._episode["depart_time"]["v0"] == 10.0 - 10.0 / 2.0


def test_sumo_average_travel_time_uses_completed_trips() -> None:
    engine = _FakeSumoEngine()
    m = SumoMetrics(
        engine=engine,
        intersections=[_Ix()],
        metric_names=[],
        delta_time=10.0,
        roadnet=_FakeRoadnet(),
    )
    m._episode["completed"] = [
        {"real_tt": 12.0},
        {"real_tt": 18.0},
    ]

    assert m.get("average_travel_time") == 15.0
