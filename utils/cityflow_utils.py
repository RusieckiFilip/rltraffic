"""
Utilities for parsing CityFlow roadnet files and extracting topology
information needed to build observation / action spaces.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from utils.common_utils import IntersectionInfo, RoadnetInfo


def _polyline_length(points: list[dict[str, float]]) -> float:
    """Euclidean length of a CityFlow road polyline (``points`` array)."""
    total = 0.0
    for a, b in zip(points, points[1:]):
        dx = float(b.get("x", 0.0)) - float(a.get("x", 0.0))
        dy = float(b.get("y", 0.0)) - float(a.get("y", 0.0))
        total += math.hypot(dx, dy)
    return total


def _lane_id(road_id: str, lane_index: int) -> str:
    """Construct a CityFlow lane id: ``<road_id>_<lane_index>``."""
    return f"{road_id}_{lane_index}"


def parse_roadnet(roadnet_path: str | Path) -> RoadnetInfo:
    """Parse a CityFlow roadnet JSON file and return structured info.

    Parameters
    ----------
    roadnet_path:
        Path to the ``roadnet.json`` file.

    Returns
    -------
    RoadnetInfo
        Topology summary used to build gym spaces and compute features.
    """
    roadnet_path = Path(roadnet_path)
    with open(roadnet_path, "r") as f:
        data: dict[str, Any] = json.load(f)

    # ── Roads & lanes ────────────────────────────────────────────────
    road_ids: list[str] = []
    lane_ids: list[str] = []
    road_num_lanes: dict[str, int] = {}
    road_lengths: dict[str, float] = {}
    road_max_speeds: dict[str, float] = {}

    for road in data.get("roads", []):
        rid = road["id"]
        road_ids.append(rid)
        lanes = road.get("lanes", [])
        num_lanes = len(lanes)
        road_num_lanes[rid] = num_lanes
        for idx in range(num_lanes):
            lane_ids.append(_lane_id(rid, idx))

        road_lengths[rid] = _polyline_length(road.get("points", []))

        lane_speeds = [
            float(ln.get("maxSpeed", 0.0))
            for ln in lanes
            if ln.get("maxSpeed") is not None
        ]
        # Use the slowest lane to give a conservative ideal speed.
        # (CityFlow scenarios typically use identical maxSpeed per lane.)
        road_max_speeds[rid] = min(lane_speeds) if lane_speeds else 0.0

    # ── Intersections ────────────────────────────────────────────────
    intersections: list[IntersectionInfo] = []

    for ix in data.get("intersections", []):
        # Skip virtual/peripheral intersections – they are not agent-controlled.
        if ix.get("virtual", False) or ix.get("gt_virtual", False):
            continue

        ix_id: str = ix["id"]
        road_links: list[dict[str, Any]] = ix.get("roadLinks", [])

        # Determine incoming / outgoing lanes via roadLinks
        incoming_lanes: list[str] = []
        outgoing_lanes: list[str] = []
        seen_in: set[str] = set()
        seen_out: set[str] = set()

        # Per-roadlink lane mapping (needed by MaxPressure etc.)
        roadlink_lanes: list[tuple[list[str], list[str]]] = []

        for rl in road_links:
            start_road = rl["startRoad"]
            end_road = rl["endRoad"]
            rl_in: list[str] = []
            rl_out: list[str] = []
            rl_seen_in: set[str] = set()
            rl_seen_out: set[str] = set()
            for ll in rl.get("laneLinks", []):
                in_lane = _lane_id(start_road, ll["startLaneIndex"])
                out_lane = _lane_id(end_road, ll["endLaneIndex"])
                if in_lane not in seen_in:
                    incoming_lanes.append(in_lane)
                    seen_in.add(in_lane)
                if out_lane not in seen_out:
                    outgoing_lanes.append(out_lane)
                    seen_out.add(out_lane)
                if in_lane not in rl_seen_in:
                    rl_in.append(in_lane)
                    rl_seen_in.add(in_lane)
                if out_lane not in rl_seen_out:
                    rl_out.append(out_lane)
                    rl_seen_out.add(out_lane)
            roadlink_lanes.append((rl_in, rl_out))

        # Traffic-light phases
        tl = ix.get("trafficLight", {})
        phases = tl.get("lightphases", [])
        num_phases = len(phases)
        phase_roadlink_mapping = [
            p.get("availableRoadLinks", []) for p in phases
        ]
        phase_durations = [float(p.get("time", 0.0)) for p in phases]
        phase_states = ["" for _ in phases]

        intersections.append(
            IntersectionInfo(
                id=ix_id,
                incoming_lanes=incoming_lanes,
                outgoing_lanes=outgoing_lanes,
                num_phases=num_phases,
                phase_roadlink_mapping=phase_roadlink_mapping,
                phase_durations=phase_durations,
                phase_states=phase_states,
                roadlink_lanes=roadlink_lanes,
            )
        )

    return RoadnetInfo(
        intersections=intersections,
        lane_ids=lane_ids,
        road_ids=road_ids,
        intersection_ids=[ix.id for ix in intersections],
        road_lengths=road_lengths,
        road_max_speeds=road_max_speeds,
    )
