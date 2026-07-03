from __future__ import annotations

from dataclasses import dataclass

from metrics.sumo import SumoMetrics


HALTING_METRIC = "number_of_all_halting_vehicles_for_the_last_time_step_in_simulation"


@dataclass
class _Ix:
    id: str = "ix"
    incoming_lanes: list[str] = None
    outgoing_lanes: list[str] = None

    def __post_init__(self) -> None:
        if self.incoming_lanes is None:
            self.incoming_lanes = ["in"]
        if self.outgoing_lanes is None:
            self.outgoing_lanes = ["out"]


class _CountingVehicle:
    def __init__(self) -> None:
        self.id_list_calls = 0
        self.speed_calls = 0
        self.distance_calls = 0
        self.waiting_calls = 0

    def getIDList(self) -> list[str]:
        self.id_list_calls += 1
        return ["v0"]

    def getSpeed(self, _vid: str) -> float:
        self.speed_calls += 1
        return 0.0

    def getDistance(self, _vid: str) -> float:
        self.distance_calls += 1
        return 1.0

    def getAccumulatedWaitingTime(self, _vid: str) -> float:
        self.waiting_calls += 1
        return 0.0


class _Simulation:
    def getTime(self) -> float:
        return 10.0

    def getArrivedIDList(self) -> list[str]:
        return []


class _Lane:
    def getIDList(self) -> list[str]:
        return ["in", "out", ":internal"]

    def getLastStepHaltingNumber(self, lid: str) -> int:
        return {"in": 2, "out": 1}.get(lid, 0)

    def getLastStepVehicleNumber(self, lid: str) -> int:
        return {"in": 3, "out": 1}.get(lid, 0)


class _Engine:
    def __init__(self) -> None:
        self.vehicle = _CountingVehicle()
        self.simulation = _Simulation()
        self.lane = _Lane()


@dataclass
class _Roadnet:
    road_lengths: dict[str, float] = None
    road_max_speeds: dict[str, float] = None

    def __post_init__(self) -> None:
        if self.road_lengths is None:
            self.road_lengths = {}
        if self.road_max_speeds is None:
            self.road_max_speeds = {}


def test_sumo_queue_metric_avoids_per_vehicle_calls() -> None:
    engine = _Engine()
    metrics = SumoMetrics(
        engine=engine,
        intersections=[_Ix()],
        metric_names=[HALTING_METRIC],
        delta_time=1.0,
        roadnet=_Roadnet(),
    )

    metrics.update()
    assert metrics.compute_all() == {HALTING_METRIC: 3.0}
    assert engine.vehicle.id_list_calls == 0
    assert engine.vehicle.speed_calls == 0
    assert engine.vehicle.distance_calls == 0
    assert engine.vehicle.waiting_calls == 0
