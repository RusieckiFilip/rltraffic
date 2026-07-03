"""Unit tests for SUMO .sumocfg input resolution."""

from pathlib import Path

import pytest

from utils.sumo_utils import (
    extract_sumo_intersections,
    extract_sumo_roadnet,
    iter_sumo_input_paths,
    validate_sumo_inputs_exist,
)


class _FakePhase:
    def __init__(self, duration: float, state: str) -> None:
        self.duration = duration
        self.state = state


class _FakeLogic:
    def __init__(self, phases: list[_FakePhase]) -> None:
        self._phases = phases

    def getPhases(self) -> list[_FakePhase]:
        return self._phases


class _FakeTrafficLight:
    def __init__(self) -> None:
        self._ids = ["b", "empty", "a"]
        self._logics = {
            "a": [_FakeLogic([_FakePhase(30, "Grsy"), _FakePhase(5, "rrrr")])],
            "b": [_FakeLogic([_FakePhase(20, "gG"), _FakePhase(4, "yy")])],
            "empty": [],
        }
        self._links = {
            "a": [
                [("a_in_0", "a_out_0", ":a_0")],
                [("a_in_1", "a_out_1", ":a_1")],
                [("a_in_2", "a_out_2", ":a_2")],
                [("a_in_3", "a_out_3", ":a_3")],
            ],
            "b": [
                [("b_in_0", "b_out_0", ":b_0")],
                [("b_in_1", "b_out_1", ":b_1")],
            ],
            "empty": [],
        }

    def getIDList(self) -> list[str]:
        return self._ids

    def getCompleteRedYellowGreenDefinition(self, tl_id: str) -> list[_FakeLogic]:
        return self._logics[tl_id]

    def getControlledLinks(self, tl_id: str) -> list[list[tuple[str, str, str]]]:
        return self._links[tl_id]


class _FakeLane:
    def __init__(self) -> None:
        self._ids = ["edge_a_0", "edge_a_1", "edge_b_0"]
        self._lengths = {
            "edge_a_0": 100.0,
            "edge_a_1": 110.0,
            "edge_b_0": 80.0,
        }
        self._speeds = {
            "edge_a_0": 10.0,
            "edge_a_1": 12.0,
            "edge_b_0": 8.0,
        }

    def getIDList(self) -> list[str]:
        return self._ids

    def getLength(self, lane_id: str) -> float:
        return self._lengths[lane_id]

    def getMaxSpeed(self, lane_id: str) -> float:
        return self._speeds[lane_id]


class _FakeEdge:
    def getIDList(self) -> list[str]:
        return ["edge_a", "edge_b"]

    def getLaneNumber(self, edge_id: str) -> int:
        return {"edge_a": 2, "edge_b": 1}[edge_id]


class _FakeSumo:
    def __init__(self) -> None:
        self.trafficlight = _FakeTrafficLight()
        self.lane = _FakeLane()
        self.edge = _FakeEdge()


def test_parse_sumocfg_collects_input_paths(tmp_path: Path) -> None:
    cfg = tmp_path / "scenario.sumocfg"
    cfg.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file value="net.net.xml"/>
    <route-files value="a.rou.xml;b.rou.xml"/>
    <additional-files value="add.xml, extra.xml"/>
  </input>
</configuration>
""",
        encoding="utf-8",
    )

    assert iter_sumo_input_paths(cfg) == [
        (tmp_path / "net.net.xml").resolve(),
        (tmp_path / "a.rou.xml").resolve(),
        (tmp_path / "b.rou.xml").resolve(),
        (tmp_path / "add.xml").resolve(),
        (tmp_path / "extra.xml").resolve(),
    ]


def test_validate_raises_when_net_missing(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.sumocfg"
    cfg.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file value="missing.net.xml"/>
  </input>
</configuration>
""",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match=r"missing\.net\.xml"):
        validate_sumo_inputs_exist(cfg)


def test_validate_passes_when_touch_net_only(tmp_path: Path) -> None:
    net = tmp_path / "exists.net.xml"
    net.write_text("<net/>", encoding="utf-8")
    cfg = tmp_path / "ok.sumocfg"
    cfg.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file value="{net.name}"/>
  </input>
</configuration>
""",
        encoding="utf-8",
    )

    validate_sumo_inputs_exist(cfg)


def test_extract_sumo_intersections_builds_neutral_topology() -> None:
    intersections = extract_sumo_intersections(_FakeSumo())

    assert [ix.id for ix in intersections] == ["a", "b"]

    ix_a = intersections[0]
    assert ix_a.incoming_lanes == ["a_in_0", "a_in_1", "a_in_2", "a_in_3"]
    assert ix_a.outgoing_lanes == ["a_out_0", "a_out_1", "a_out_2", "a_out_3"]
    assert ix_a.num_phases == 2
    assert ix_a.phase_durations == [30.0, 5.0]
    assert ix_a.phase_roadlink_mapping == [[0], []]
    assert ix_a.roadlink_lanes[0] == (["a_in_0"], ["a_out_0"])

    ix_b = intersections[1]
    assert ix_b.phase_roadlink_mapping == [[0, 1], []]


def test_extract_sumo_roadnet_caches_static_edge_metadata() -> None:
    roadnet = extract_sumo_roadnet(_FakeSumo())

    assert [ix.id for ix in roadnet.intersections] == ["a", "b"]
    assert roadnet.lane_ids == ["edge_a_0", "edge_a_1", "edge_b_0"]
    assert roadnet.road_ids == ["edge_a", "edge_b"]
    assert roadnet.intersection_ids == ["a", "b"]
    assert roadnet.road_lengths == {"edge_a": 110.0, "edge_b": 80.0}
    assert roadnet.road_max_speeds == {"edge_a": 10.0, "edge_b": 8.0}
