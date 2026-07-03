"""Backend-neutral traffic topology models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoadnetInfo:
    """Backend-agnostic road network metadata."""

    intersections: list["IntersectionInfo"] = field(default_factory=list)
    """Only controllable intersections."""

    lane_ids: list[str] = field(default_factory=list)
    """All lane IDs in the network."""

    road_ids: list[str] = field(default_factory=list)
    """All road/edge IDs in the network."""

    intersection_ids: list[str] = field(default_factory=list)
    """IDs of controllable intersections, same order as ``intersections``."""

    road_lengths: dict[str, float] = field(default_factory=dict)
    """Per-road/edge length in metres."""

    road_max_speeds: dict[str, float] = field(default_factory=dict)
    """Per-road/edge speed limit in m/s."""


@dataclass
class IntersectionInfo:
    """Backend-agnostic metadata for a single controllable intersection."""

    id: str
    """Intersection identifier as it appears in the simulator/network."""

    incoming_lanes: list[str] = field(default_factory=list)
    """Lane IDs whose traffic flows *into* this intersection."""

    outgoing_lanes: list[str] = field(default_factory=list)
    """Lane IDs whose traffic flows *out of* this intersection."""

    num_phases: int = 0
    """Number of traffic-light phases defined for this intersection."""

    phase_roadlink_mapping: list[list[int]] = field(default_factory=list)
    """For each phase index, active movement/link indices."""

    phase_durations: list[float] = field(default_factory=list)
    """Configured simulator duration in seconds for each phase index."""

    phase_states: list[str] = field(default_factory=list)
    """Raw simulator phase state strings when available (SUMO), else empty."""

    roadlink_lanes: list[tuple[list[str], list[str]]] = field(
        default_factory=list
    )
    """Per movement/link (incoming_lanes, outgoing_lanes) tuples.

    Index *i* corresponds to movement/link *i* as referenced in
    ``phase_roadlink_mapping``.
    """
