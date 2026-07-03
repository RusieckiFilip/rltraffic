"""SUMO topology extraction and input-file helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from utils.common_utils import IntersectionInfo, RoadnetInfo


def iter_sumo_input_paths(sumocfg_path: Path) -> list[Path]:
    resolved_cfg = Path(sumocfg_path).resolve()
    cfg_dir = resolved_cfg.parent
    root = ET.parse(resolved_cfg).getroot()
    out: list[Path] = []
    seen: set[Path] = set()

    for tag in ("net-file", "route-files", "additional-files"):
        for el in root.iter(tag):
            raw = el.attrib.get("value")
            if not raw:
                continue
            for fragment in raw.replace(";", ",").split(","):
                piece = fragment.strip()
                if not piece:
                    continue
                path = Path(piece)
                if not path.is_absolute():
                    path = cfg_dir / path
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                out.append(resolved)

    return out


def validate_sumo_inputs_exist(sumocfg_path: str | Path) -> None:
    cfg_path = Path(sumocfg_path).resolve()
    missing = [path for path in iter_sumo_input_paths(cfg_path) if not path.is_file()]
    if not missing:
        return

    detail = "\n".join(f"  - {path}" for path in missing)
    raise FileNotFoundError(
        f"SUMO inputs referenced by {cfg_path} are missing:\n{detail}\n"
        "(Until SUMO starts, TraCI prints 'Retrying in 1 seconds'.)"
    )


def _extract_sumo_intersections(sumo: Any) -> list[IntersectionInfo]:
    intersections: list[IntersectionInfo] = []
    for tl_id in sumo.trafficlight.getIDList():
        definitions = sumo.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)
        if not definitions:
            continue

        phases = definitions[0].getPhases()
        num_phases = len(phases)
        links = sumo.trafficlight.getControlledLinks(tl_id)

        incoming_lanes: set[str] = set()
        outgoing_lanes: set[str] = set()
        roadlink_lanes: dict[int, tuple[list[str], list[str]]] = {}

        # links[link_index][connection_index] = (incoming, outgoing, internal)
        for link_idx, link_group in enumerate(links):
            in_lanes_for_link: list[str] = []
            out_lanes_for_link: list[str] = []
            for connection in link_group:
                if len(connection) < 2:
                    continue
                incoming = connection[0]
                outgoing = connection[1]
                incoming_lanes.add(incoming)
                outgoing_lanes.add(outgoing)
                in_lanes_for_link.append(incoming)
                out_lanes_for_link.append(outgoing)
            roadlink_lanes[link_idx] = (in_lanes_for_link, out_lanes_for_link)

        if not incoming_lanes:
            continue

        phase_roadlink_mapping: list[list[int]] = []
        phase_durations: list[float] = []
        phase_states: list[str] = []
        for phase in phases:
            active_links = [
                link_idx
                for link_idx, signal_char in enumerate(phase.state)
                if signal_char in {"g", "G"}
            ]
            phase_roadlink_mapping.append(active_links)
            phase_durations.append(float(phase.duration))
            phase_states.append(str(phase.state))

        intersections.append(
            IntersectionInfo(
                id=tl_id,
                incoming_lanes=sorted(incoming_lanes),
                outgoing_lanes=sorted(outgoing_lanes),
                num_phases=num_phases,
                phase_roadlink_mapping=phase_roadlink_mapping,
                phase_durations=phase_durations,
                phase_states=phase_states,
                roadlink_lanes=[
                    roadlink_lanes[idx] for idx in sorted(roadlink_lanes)
                ],
            )
        )

    intersections.sort(key=lambda x: x.id)
    return intersections


def _sumo_lane_ids(sumo: Any) -> list[str]:
    try:
        return sorted(str(lid) for lid in sumo.lane.getIDList())
    except Exception:
        return []


def _sumo_road_ids(sumo: Any, lane_ids: list[str]) -> list[str]:
    try:
        return sorted(str(eid) for eid in sumo.edge.getIDList())
    except Exception:
        road_ids = {lid.rsplit("_", 1)[0] for lid in lane_ids if "_" in lid}
        return sorted(road_ids)


def _representative_lanes_by_road(
    sumo: Any,
    road_ids: list[str],
    lane_ids: list[str],
) -> dict[str, list[str]]:
    lane_set = set(lane_ids)
    by_road: dict[str, list[str]] = {}

    for rid in road_ids:
        lanes: list[str] = []
        try:
            lane_count = int(sumo.edge.getLaneNumber(rid))
        except Exception:
            lane_count = 0
        for idx in range(lane_count):
            lid = f"{rid}_{idx}"
            if not lane_set or lid in lane_set:
                lanes.append(lid)
        if lanes:
            by_road[rid] = lanes

    edge_api_roads = set(by_road)
    for lid in lane_ids:
        if "_" not in lid:
            continue
        rid = lid.rsplit("_", 1)[0]
        if rid in edge_api_roads:
            continue
        by_road.setdefault(rid, []).append(lid)
    return by_road


def extract_sumo_roadnet(sumo: Any) -> RoadnetInfo:
    """Extract static road network metadata from an active SUMO API."""
    intersections = _extract_sumo_intersections(sumo)
    lane_ids = _sumo_lane_ids(sumo)
    road_ids = _sumo_road_ids(sumo, lane_ids)
    lanes_by_road = _representative_lanes_by_road(sumo, road_ids, lane_ids)

    road_lengths: dict[str, float] = {}
    road_max_speeds: dict[str, float] = {}
    lane_api = sumo.lane
    for rid in road_ids:
        lengths: list[float] = []
        speeds: list[float] = []
        for lid in lanes_by_road.get(rid, []):
            try:
                lengths.append(float(lane_api.getLength(lid)))
                speeds.append(float(lane_api.getMaxSpeed(lid)))
            except Exception:
                continue
        road_lengths[rid] = max(lengths) if lengths else 0.0
        # Match CityFlow's conservative route-time semantics.
        road_max_speeds[rid] = min(speeds) if speeds else 0.0

    return RoadnetInfo(
        intersections=intersections,
        lane_ids=lane_ids,
        road_ids=road_ids,
        intersection_ids=[ix.id for ix in intersections],
        road_lengths=road_lengths,
        road_max_speeds=road_max_speeds,
    )


def extract_sumo_intersections(sumo: Any) -> list[IntersectionInfo]:
    """Extract controllable intersection topology from an active SUMO API."""
    return _extract_sumo_intersections(sumo)
