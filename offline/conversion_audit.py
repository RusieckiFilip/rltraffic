"""P7.1 half B: the Conversion Audit Protocol (CAP), criteria A-F, from files alone.

Artifact format version: ``p7.1-conversion-audit/1.0``.
Written against ``PREREGISTRATION`` A14 / ``PROJECT_PLAN`` §6 P11.5's CAP definition, and
``BRIEF_34`` Amendments A1, B and C.

WHY THIS EXISTS
---------------
``PROJECT_PLAN`` §1 claimed our scenarios were *"natively authored in both backends, so the converter
confound is absent by construction"*.  **That is false and was withdrawn on 2026-09-11**: every
hangzhou ``.net.xml`` was produced by ``netconvert 1.13.0`` from ``.nod/.edg/.con/.tll`` inputs under
a LibSignal working tree, i.e. from the CityFlow roadnet; cologne runs the other way; grid4x4's
CityFlow files are LibSignal's s2c output.  **No pair in this repo is natively authored in both
backends.**  What replaces the claim is CAP: *topology, routes and signal timing are shared BY
CONSTRUCTION through one documented conversion, vehicle dynamics are matched by the parity vType,
and the conversion's structural artefacts are AUDITED and enumerated.*

⛔ **This module audits; it does not admit.**  A14 makes CAP the admissibility rule and the
coordinator applies it.  Nothing here returns a verdict.

THE CRITERIA, AS P11.5 DEFINES THEM
-----------------------------------
* **(A)** a bijection of non-virtual intersections <-> TLS junctions and roads <-> edges, by a key
  proven from structure; lane counts per road equal.
* **(B)** junction coordinates (residual), per-lane length and speed limit, with max abs difference.
* **(C)** per intersection the connection set under the movement correspondence against CityFlow's
  ``laneLinks``; every EXTRA SUMO connection (u-turns ``t``, netconvert defaults) enumerated with
  counts; a MISSING one is a finding.
* **(D)** signals as ``SumoEnv`` consumes them: released-lane sets per action, phase counts, the
  ``phase_onehot`` map, and the fact that the env renders transitions ITSELF.
* **(E)** demand: vehicle count, per-vehicle departure time and route.
* **(F)** provenance: tool, version, date and the recorded inputs, from the ``.net.xml`` header.
* **(G)** is half A's output and is CROSS-REFERENCED, never recomputed here.

FIVE TRAPS, EACH MEASURED RATHER THAN ASSUMED
---------------------------------------------
Every one of these produces a plausible wrong number in a tool that does the obvious thing:

1. **``gt_virtual``.**  An s2c-converted roadnet marks perimeter intersections ``gt_virtual``, not
   ``virtual``: grid4x4 has 32 intersections, 0 of them ``virtual``, and 16 real ones.  cologne3 has
   29 and 3.  The frozen parser (``utils/cityflow_utils.py:80``) reads both keys, so **this module
   uses the frozen parser and never its own filter.**
2. **``availableRoadLinks`` is not sorted.**  56 of grid4x4's 256 phase lists and 16 of hz4x4's 144
   are stored in non-ascending order.  Compared as LISTS they look like differences; they are not.
   **Compared as sets.**
3. **The ``linkIndex`` -> roadLink permutation is not the identity.**  It is on grid4x4 and hz1x1;
   on hz4x4 it is ``[10, 11, 9, 6, 7, 8, 3, 4, 5, 2, 0, 1]``.  A tool that assumes identity
   mis-maps every hz4x4 phase, so the permutation is DERIVED per intersection.
4. **Demand order differs between the two files on hz4x4.**  Index-aligned it matches on 35 of 2,983
   vehicles; as a MULTISET of ``(depart, route)`` it matches on 2,983 of 2,983.  hz1x1 and grid4x4
   happen to be index-aligned.  **Compared as a multiset, with the index-aligned count reported
   beside it** so the difference is visible rather than hidden.
5. **The clearance character is scenario-specific**: ``r`` on hz1x1 (which has no right turns at
   all), ``s`` on hz4x4, ``y`` on grid4x4.  It is reported, not asserted.

⚠️ **The candidates tree is READ-ONLY and gitignored** (`BRIEF_34` Amendment C): RESCO's grid4x4 net
and the route file inside ``grid4x4.zip`` are read in place, the zip member with ``zipfile`` and
never extracted, and the files are CC BY-NC-SA 4.0 and are never copied into the repo.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "AUDIT_FORMAT_VERSION",
    "audit_artifact",
    "audit_pair",
    "cityflow_phase_links",
    "demand_from_cityflow_flow",
    "demand_from_route_file",
    "link_index_permutation",
    "sumo_connection_counts",
    "sumo_phase_links",
    "sumo_provenance",
]

AUDIT_FORMAT_VERSION = "p7.1-conversion-audit/1.0"

#: SUMO signal characters that let a movement proceed.  ``s`` is "green right-of-way yielding" and
#: IS a release: grid4x4's clearance phases keep right turns on ``s`` and CityFlow's converted
#: ``availableRoadLinks`` include them, which is why its clearance phases are not empty.
SUMO_RELEASING_STATES = frozenset({"G", "g", "s"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------
# (F) provenance
# ----------------------------------------------------------------------


def sumo_provenance(net_xml: str | Path) -> dict[str, Any]:
    """Tool, version, date and recorded inputs from the ``.net.xml``'s generator comment.

    ⚠️ A net can record NOTHING: RESCO's grid4x4 header carries a ``<configuration>`` with
    ``processing``/``junctions``/``pedestrian``/``report`` and **no ``<input>``**, so the file cannot
    be traced to its sources from itself.  That absence is a provenance FINDING and is reported as
    ``recorded_inputs: {}`` with ``records_its_inputs: false`` rather than left blank.
    """
    path = Path(net_xml)
    head = path.read_text(encoding="utf-8", errors="replace")[:4000]
    tool = version = generated = None
    for line in head.splitlines():
        if "generated on" in line:
            generated = line.split("generated on", 1)[1].strip().rstrip("-").strip()
            for marker in ("netconvert", "netedit", "duarouter"):
                if marker in line:
                    tool = marker
                    version = line.split("Version", 1)[1].strip() if "Version" in line else None
            break
    inputs: dict[str, str] = {}
    for tag in ("node-files", "edge-files", "connection-files", "tllogic-files", "sumo-net-file"):
        marker = f'<{tag} value="'
        if marker in head:
            inputs[tag] = head.split(marker, 1)[1].split('"', 1)[0]
    return {
        "file": str(path),
        "sha256": _sha256(path),
        "tool": tool,
        "version": version,
        "generated_on": None if generated is None else generated.split(" by ")[0].strip(),
        "records_its_inputs": bool(inputs),
        "recorded_inputs": inputs,
    }


# ----------------------------------------------------------------------
# (C) connections
# ----------------------------------------------------------------------


def sumo_connection_counts(net_xml: str | Path) -> dict[str, Any]:
    """Connection counts by ``dir``, split by whether the source edge is internal.

    ⚠️ Both numbers are reported because both are legitimate and they differ: counting every
    ``<connection>`` element gives grid4x4 1,344 with ``l = 576``, while counting only those whose
    ``from`` is a normal edge gives 576 with ``l = 192`` -- a left turn crosses an internal junction
    and is recorded as two segments.  A table that quotes one of them without saying which is
    unreadable.
    """
    root = ET.parse(Path(net_xml)).getroot()
    everything: Counter[str] = Counter()
    from_normal: Counter[str] = Counter()
    controlled = 0
    for connection in root.findall("connection"):
        direction = str(connection.get("dir"))
        everything[direction] += 1
        edge = str(connection.get("from") or "")
        if not edge.startswith(":"):
            from_normal[direction] += 1
            if connection.get("tl") is not None:
                controlled += 1
    return {
        "all_connections": dict(sorted(everything.items())),
        "from_non_internal_edges": dict(sorted(from_normal.items())),
        "n_signal_controlled": controlled,
        "extra_directions": {
            direction: count
            for direction, count in sorted(from_normal.items())
            if direction not in {"l", "s", "r"}
        },
    }


# ----------------------------------------------------------------------
# (D) signals
# ----------------------------------------------------------------------


def link_index_permutation(net_xml: str | Path, tls_id: str, roadlinks: Sequence[tuple[str, str]]) -> tuple[int, ...]:
    """SUMO ``linkIndex`` blocks -> CityFlow roadLink indices, DERIVED per intersection.

    A SUMO signal state string is indexed by ``linkIndex``; CityFlow's ``availableRoadLinks`` indexes
    its own ``roadLinks`` list.  The two orders coincide on grid4x4 and hz1x1 and **do not** on
    hz4x4, where it is ``[10, 11, 9, 6, 7, 8, 3, 4, 5, 2, 0, 1]``.  Each controlled link is grouped
    by its ``(from_road, to_road)`` pair and matched against *roadlinks*; a pair the CityFlow side
    does not declare is a finding and raises.
    """
    root = ET.parse(Path(net_xml)).getroot()
    pair_of_index: dict[int, tuple[str, str]] = {}
    for connection in root.findall("connection"):
        if connection.get("tl") != tls_id or connection.get("linkIndex") is None:
            continue
        pair_of_index[int(str(connection.get("linkIndex")))] = (
            str(connection.get("from")),
            str(connection.get("to")),
        )
    wanted = {pair: index for index, pair in enumerate(roadlinks)}
    permutation: list[int] = []
    seen: set[int] = set()
    for index in sorted(pair_of_index):
        pair = pair_of_index[index]
        if pair not in wanted:
            raise ValueError(
                f"{tls_id}: SUMO link {index} runs {pair[0]} -> {pair[1]}, which the CityFlow "
                "roadnet does not declare as a roadLink; the two files disagree on connectivity"
            )
        target = wanted[pair]
        if target not in seen:
            seen.add(target)
            permutation.append(target)
    return tuple(permutation)


def sumo_phase_links(
    net_xml: str | Path, tls_id: str, permutation: Sequence[int]
) -> list[dict[str, Any]]:
    """Per SUMO phase: its duration and the CityFlow roadLink indices it releases."""
    root = ET.parse(Path(net_xml)).getroot()
    logic = next(
        (entry for entry in root.findall("tlLogic") if entry.get("id") == tls_id), None
    )
    if logic is None:
        raise KeyError(f"{tls_id} has no <tlLogic> in {net_xml}")
    index_to_roadlink: dict[int, int] = {}
    per_link: dict[int, tuple[str, str]] = {}
    for connection in root.findall("connection"):
        if connection.get("tl") != tls_id or connection.get("linkIndex") is None:
            continue
        per_link[int(str(connection.get("linkIndex")))] = (
            str(connection.get("from")),
            str(connection.get("to")),
        )
    ordered_pairs: list[tuple[str, str]] = []
    for link_index in sorted(per_link):
        pair = per_link[link_index]
        if pair not in ordered_pairs:
            ordered_pairs.append(pair)
        index_to_roadlink[link_index] = permutation[ordered_pairs.index(pair)]
    phases = []
    for position, phase in enumerate(logic.findall("phase")):
        state = str(phase.get("state"))
        released = {
            index_to_roadlink[link_index]
            for link_index, character in enumerate(state)
            if link_index in index_to_roadlink and character in SUMO_RELEASING_STATES
        }
        phases.append(
            {
                "phase": position,
                "duration": float(str(phase.get("duration"))),
                "state": state,
                "released_roadlinks": sorted(released),
            }
        )
    return phases


def cityflow_phase_links(roadnet: str | Path, ix_id: str) -> list[dict[str, Any]]:
    """Per CityFlow light phase: its time and ``availableRoadLinks``, and whether it was sorted."""
    data = json.loads(Path(roadnet).read_text(encoding="utf-8"))
    entry = next(ix for ix in data["intersections"] if ix["id"] == ix_id)
    phases = []
    for position, phase in enumerate(entry.get("trafficLight", {}).get("lightphases", [])):
        links = list(phase.get("availableRoadLinks", []))
        phases.append(
            {
                "phase": position,
                "time": phase.get("time"),
                "released_roadlinks": sorted(links),
                "stored_sorted": links == sorted(links),
            }
        )
    return phases


# ----------------------------------------------------------------------
# (E) demand
# ----------------------------------------------------------------------


def demand_from_route_file(
    route_file: str | Path, *, zip_member: str | None = None
) -> list[tuple[float, tuple[str, ...]]]:
    """``(depart, route)`` per vehicle, in document order, from a ``.rou.xml``.

    *zip_member* reads the file from inside an archive **without extracting it** -- RESCO ships 1,400
    route files in one 19 MB zip and the tree is read-only.
    """
    path = Path(route_file)
    if zip_member is None:
        root = ET.parse(path).getroot()
    else:
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read(zip_member))
    demand: list[tuple[float, tuple[str, ...]]] = []
    for vehicle in root.iter("vehicle"):
        route = vehicle.find("route")
        if route is None:
            raise ValueError(f"vehicle {vehicle.get('id')!r} in {path} declares no route")
        demand.append(
            (float(str(vehicle.get("depart"))), tuple(str(route.get("edges")).split()))
        )
    return demand


def demand_from_cityflow_flow(flow_json: str | Path) -> list[tuple[float, tuple[str, ...]]]:
    """``(startTime, route)`` per flow entry, in document order."""
    data = json.loads(Path(flow_json).read_text(encoding="utf-8"))
    return [(float(entry["startTime"]), tuple(entry["route"])) for entry in data]


def _compare_demand(
    cityflow: Sequence[tuple[float, tuple[str, ...]]],
    sumo: Sequence[tuple[float, tuple[str, ...]]],
) -> dict[str, Any]:
    """As a MULTISET, with the index-aligned count beside it (trap 4)."""
    aligned = sum(1 for left, right in zip(cityflow, sumo) if left == right)
    left_counts, right_counts = Counter(cityflow), Counter(sumo)
    only_cityflow = left_counts - right_counts
    only_sumo = right_counts - left_counts
    return {
        "n_cityflow": len(cityflow),
        "n_sumo": len(sumo),
        "counts_equal": len(cityflow) == len(sumo),
        "multiset_equal": left_counts == right_counts,
        "n_index_aligned_equal": aligned,
        "order_matches": aligned == len(cityflow) == len(sumo),
        "n_only_in_cityflow": sum(only_cityflow.values()),
        "n_only_in_sumo": sum(only_sumo.values()),
        "depart_range_cityflow": [min(d for d, _ in cityflow), max(d for d, _ in cityflow)]
        if cityflow
        else None,
        "depart_range_sumo": [min(d for d, _ in sumo), max(d for d, _ in sumo)] if sumo else None,
    }


# ----------------------------------------------------------------------
# (A) and (B), and the assembly
# ----------------------------------------------------------------------


def _sumo_geometry(net_xml: str | Path) -> dict[str, Any]:
    """Junction coordinates, the net offset, and the per-lane length/speed of normal edges."""
    root = ET.parse(Path(net_xml)).getroot()
    location = root.find("location")
    offset = (0.0, 0.0)
    if location is not None and location.get("netOffset"):
        x, y = str(location.get("netOffset")).split(",")
        offset = (float(x), float(y))
    junctions = {
        str(j.get("id")): (float(str(j.get("x"))), float(str(j.get("y"))))
        for j in root.findall("junction")
        if j.get("type") != "internal"
    }
    types = Counter(str(j.get("type")) for j in root.findall("junction"))
    lanes: dict[str, dict[str, float]] = {}
    edges: dict[str, list[str]] = {}
    for edge in root.findall("edge"):
        if edge.get("function") == "internal" or str(edge.get("id", "")).startswith(":"):
            continue
        edge_lanes = []
        for lane in edge.findall("lane"):
            lane_id = str(lane.get("id"))
            edge_lanes.append(lane_id)
            lanes[lane_id] = {
                "length": float(str(lane.get("length"))),
                "speed": float(str(lane.get("speed"))),
            }
        edges[str(edge.get("id"))] = edge_lanes
    return {
        "net_offset": list(offset),
        "proj_parameter": None if location is None else location.get("projParameter"),
        "junctions": junctions,
        "junction_types": dict(sorted(types.items())),
        "edges": edges,
        "lanes": lanes,
    }


def audit_pair(
    name: str,
    *,
    cityflow_roadnet: str | Path,
    sumo_net: str | Path,
    cityflow_flow: str | Path | None = None,
    sumo_routes: str | Path | None = None,
    sumo_routes_zip_member: str | None = None,
    direction: str,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """CAP criteria A-F for one converted pair, from files alone.  No simulation, no verdict."""
    from utils.cityflow_utils import parse_roadnet

    roadnet_path, net_path = Path(cityflow_roadnet), Path(sumo_net)
    data = json.loads(roadnet_path.read_text(encoding="utf-8"))
    parsed = parse_roadnet(roadnet_path)
    geometry = _sumo_geometry(net_path)
    controlled = sorted(sumo_incoming := set(_tls_ids(net_path)))

    # --- (A) bijection -------------------------------------------------
    cityflow_ids = [ix.id for ix in parsed.intersections]
    cityflow_roads = {road["id"]: len(road["lanes"]) for road in data["roads"]}
    sumo_edges = {edge: len(lanes) for edge, lanes in geometry["edges"].items()}
    shared_roads = sorted(set(cityflow_roads) & set(sumo_edges))
    criterion_a = {
        "n_cityflow_intersections_non_virtual": len(cityflow_ids),
        "n_sumo_traffic_lights": len(controlled),
        "intersection_ids_equal": sorted(cityflow_ids) == controlled,
        "n_cityflow_roads": len(cityflow_roads),
        "n_sumo_normal_edges": len(sumo_edges),
        "road_ids_equal": sorted(cityflow_roads) == sorted(sumo_edges),
        "n_roads_with_equal_lane_counts": sum(
            1 for road in shared_roads if cityflow_roads[road] == sumo_edges[road]
        ),
        "n_shared_road_ids": len(shared_roads),
        "n_cityflow_lanes": sum(cityflow_roads.values()),
        "n_sumo_lanes": sum(sumo_edges.values()),
        "virtual_key_used": "gt_virtual" if any("gt_virtual" in ix for ix in data["intersections"]) else "virtual",
        "n_intersection_entries_in_file": len(data["intersections"]),
    }

    # --- (B) geometry --------------------------------------------------
    residuals = []
    for entry in data["intersections"]:
        sumo_point = geometry["junctions"].get(entry["id"])
        if sumo_point is None:
            continue
        point = entry.get("point", {})
        residuals.append(
            max(
                abs(float(point.get("x", 0.0)) + geometry["net_offset"][0] - sumo_point[0]),
                abs(float(point.get("y", 0.0)) + geometry["net_offset"][1] - sumo_point[1]),
            )
        )
    cityflow_speeds = Counter(
        float(lane["maxSpeed"]) for road in data["roads"] for lane in road["lanes"]
    )
    sumo_speeds = Counter(lane["speed"] for lane in geometry["lanes"].values())
    criterion_b = {
        "net_offset": geometry["net_offset"],
        "proj_parameter": geometry["proj_parameter"],
        "n_junctions_compared": len(residuals),
        "max_abs_coordinate_residual": max(residuals) if residuals else None,
        "cityflow_lane_speeds": {str(k): v for k, v in sorted(cityflow_speeds.items())},
        "sumo_lane_speeds": {str(k): v for k, v in sorted(sumo_speeds.items())},
        "lane_speeds_equal": set(cityflow_speeds) == set(sumo_speeds),
        "sumo_lane_length_min": min(l["length"] for l in geometry["lanes"].values()),
        "sumo_lane_length_max": max(l["length"] for l in geometry["lanes"].values()),
    }

    # --- (C) connections -----------------------------------------------
    criterion_c = sumo_connection_counts(net_path)

    # --- (D) signals ----------------------------------------------------
    per_intersection: dict[str, Any] = {}
    for entry_id in cityflow_ids:
        entry = next(ix for ix in data["intersections"] if ix["id"] == entry_id)
        roadlinks = [(link["startRoad"], link["endRoad"]) for link in entry.get("roadLinks", [])]
        try:
            permutation = link_index_permutation(net_path, entry_id, roadlinks)
            sumo_phases = sumo_phase_links(net_path, entry_id, permutation)
        except (KeyError, ValueError) as exc:
            per_intersection[entry_id] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        cityflow_phases = cityflow_phase_links(roadnet_path, entry_id)
        # ⭐ The negative control, carried in the artifact rather than asserted in prose: re-score
        # the same phases under the IDENTITY permutation. If (D) matched either way it would have
        # no discriminating power and the 100 % would mean nothing.
        identity_phases = sumo_phase_links(net_path, entry_id, tuple(range(len(permutation))))
        # The two programs need not have the same phase COUNT: hz1x1 and hz4x4 collapse SUMO's 8
        # identical clearance phases into CityFlow's single phase 0, grid4x4 keeps all 16.
        if len(sumo_phases) == len(cityflow_phases):
            pairs = list(zip(cityflow_phases, sumo_phases))
        else:
            pairs = [(cityflow_phases[0], sumo_phases[1])] + [
                (cityflow_phases[k], sumo_phases[2 * (k - 1)])
                for k in range(1, len(cityflow_phases))
            ]
        if len(identity_phases) == len(cityflow_phases):
            identity_pairs = list(zip(cityflow_phases, identity_phases))
        else:
            identity_pairs = [(cityflow_phases[0], identity_phases[1])] + [
                (cityflow_phases[k], identity_phases[2 * (k - 1)])
                for k in range(1, len(cityflow_phases))
            ]
        per_intersection[entry_id] = {
            "link_index_permutation": list(permutation),
            "n_phases_matching_under_the_identity_permutation": sum(
                1 for cf, su in identity_pairs if cf["released_roadlinks"] == su["released_roadlinks"]
            ),
            "permutation_is_identity": list(permutation) == sorted(permutation),
            "n_cityflow_phases": len(cityflow_phases),
            "n_sumo_phases": len(sumo_phases),
            "n_phases_matching_as_sets": sum(
                1 for cf, su in pairs if cf["released_roadlinks"] == su["released_roadlinks"]
            ),
            "n_phases_compared": len(pairs),
            "n_cityflow_phase_lists_stored_unsorted": sum(
                1 for phase in cityflow_phases if not phase["stored_sorted"]
            ),
            "clearance_characters": sorted(
                {
                    character
                    for phase in sumo_phases
                    if float(phase["duration"]) <= 5.0
                    for character in phase["state"]
                }
            ),
        }
    criterion_d = {
        "per_intersection": per_intersection,
        "n_intersections": len(per_intersection),
        "n_phases_matching_as_sets": sum(
            entry["n_phases_matching_as_sets"]
            for entry in per_intersection.values()
            if "error" not in entry
        ),
        "n_phases_compared": sum(
            entry["n_phases_compared"] for entry in per_intersection.values() if "error" not in entry
        ),
        "n_phases_matching_under_the_identity_permutation": sum(
            entry["n_phases_matching_under_the_identity_permutation"]
            for entry in per_intersection.values()
            if "error" not in entry
        ),
        "n_cityflow_phase_lists_stored_unsorted": sum(
            entry["n_cityflow_phase_lists_stored_unsorted"]
            for entry in per_intersection.values()
            if "error" not in entry
        ),
        "n_fully_matching": sum(
            1
            for entry in per_intersection.values()
            if "error" not in entry
            and entry["n_phases_matching_as_sets"] == entry["n_phases_compared"]
        ),
        "how_the_env_uses_them": (
            "SumoEnv renders transitions ITSELF via setRedYellowGreenState from phase_states[phase] "
            "(envs/sumo_env.py:252-278) with the recipe ('yellow', 3), ('all_red', 2), while "
            "CityFlow uses ('all_red', 5) (envs/phase_control.py:25). So the converter's clearance "
            "DURATIONS are inert -- the env never plays them -- and its clearance STATES are only "
            "the base string the env recolours. The phase COUNTS still matter, because phase_onehot "
            "is num_phases wide."
        ),
    }

    # --- (E) demand ------------------------------------------------------
    criterion_e: dict[str, Any] = {"compared": False}
    if cityflow_flow is not None and sumo_routes is not None:
        criterion_e = {
            "compared": True,
            "cityflow_flow": str(cityflow_flow),
            "sumo_routes": str(sumo_routes)
            + ("" if sumo_routes_zip_member is None else f"::{sumo_routes_zip_member}"),
            **_compare_demand(
                demand_from_cityflow_flow(cityflow_flow),
                demand_from_route_file(sumo_routes, zip_member=sumo_routes_zip_member),
            ),
        }

    return {
        "pair": name,
        "direction": direction,
        "cityflow_roadnet": str(roadnet_path),
        "sumo_net": str(net_path),
        "A_bijection": criterion_a,
        "B_geometry": criterion_b,
        "C_connections": criterion_c,
        "D_signals": criterion_d,
        "E_demand": criterion_e,
        "F_provenance": sumo_provenance(net_path),
        "notes": list(notes),
    }


def _tls_ids(net_xml: str | Path) -> list[str]:
    root = ET.parse(Path(net_xml)).getroot()
    return [str(logic.get("id")) for logic in root.findall("tlLogic")]


def audit_artifact(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``docs/data/p7_1_conversion_audit.json``: CAP's table.  It audits; it admits nothing."""
    return {
        "format_version": AUDIT_FORMAT_VERSION,
        "task": "P7.1",
        "registered_in": "PREREGISTRATION A14 / PROJECT_PLAN section 6 P11.5 (CAP, criteria A-H)",
        "role": (
            "the structural audit that replaces the withdrawn 'natively authored' claim: what each "
            "documented conversion preserved and what it changed, from files alone"
        ),
        "what_this_does_not_say": [
            "It issues NO admissibility verdict: A14 makes CAP the rule and the coordinator "
            "applies it.",
            "(G) -- the P7.0 protocol, rho, the OVL/KS table and half A's decomposition -- is "
            "CROSS-REFERENCED from docs/data/p7_0_gate.json and docs/data/p7_1_metric_freeze.json, "
            "never recomputed here.",
            "No pair in this repo is natively authored in both backends; every row names its "
            "conversion direction.",
            "Nothing here was simulated, and no file outside the repo was copied into it.",
        ],
        "criteria": {
            "A": "bijection of intersections and roads, lane counts",
            "B": "junction coordinates, per-lane length and speed",
            "C": "connection sets, extras enumerated",
            "D": "signals as SumoEnv consumes them",
            "E": "demand per vehicle",
            "F": "provenance from the network header",
            "G": "cross-referenced, never recomputed here",
        },
        "n_pairs": len(pairs),
        "pairs": list(pairs),
    }
