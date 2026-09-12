"""P7.1 half B: the movement-keyed adapter between a SUMO ``info`` and the CityFlow corpus's frame.

Artifact format version: ``p7.1-alignment/1.0``.
Written against ``BRIEF_34`` §3 + Amendments A–F, and ``docs/plans/p7.1.md``.

WHAT THIS MODULE IS FOR
-----------------------
A CityFlow-trained model reads a per-intersection ``state`` whose lane blocks are in the CityFlow
corpus's order and whose phase block is 9 wide.  A SUMO ``info`` carries the same three features in a
different lane order, under lane ids that **denote different physical lanes**, with a 16-wide phase
block.  :func:`align_info` translates the second into the first, or refuses.

🚨 **THE DEFECT THIS EXISTS TO PREVENT, AND IT IS NOT HYPOTHETICAL.**  P7.0 aligned by lane id, which
looks obviously right and is wrong here: CityFlow indexes a road's lanes left-to-right and SUMO
right-to-left, so ``road_0_1_0_0`` is the LEFT-TURN lane in one file and the THROUGH lane in the
other, on all four incoming roads of ``hangzhou_1x1_bc-tyc``.  The by-id reading compared a
left-turn queue against a through queue and returned a minimum per-feature overlap of ``0.0554``
where the movement-paired reading returns ``0.3407``.  **Sorting the lanes does not repair it**: it
fixes the ORDER and leaves the DENOTATION wrong.  ``PROJECT_PLAN`` §7's rule, added from that
finding: *any cross-system comparison must prove its pairing key from structure alone, before using
it.*  Here the proof is :func:`offline.transfer_gate.lane_semantic_correspondence`, which reads only
the ``type``/``dir`` attributes of the two network files and refuses anything but a unique match.
It is IMPORTED, never re-implemented.

THE CANONICAL FRAME
-------------------
**Canonical = the CityFlow corpus's per-intersection ``incoming_lanes`` order** -- discovery order
over ``roadLinks``/``laneLinks`` (``utils/cityflow_utils.py:95-110``), NOT sorted.  It is canonical
for one reason: the trained models read that order and the corpus cannot move.  SUMO's own order is
``sorted()`` (``utils/sumo_utils.py:102``), so the two differ even where the ids agree.

    CityFlow hz1x1: road_0_1_0_1, road_0_1_0_0, road_1_0_1_1, road_1_0_1_0,
                    road_2_1_2_1, road_2_1_2_0, road_1_2_3_0, road_1_2_3_1
    SUMO     hz1x1: road_0_1_0_0, road_0_1_0_1, road_1_0_1_0, road_1_0_1_1,
                    road_1_2_3_0, road_1_2_3_1, road_2_1_2_0, road_2_1_2_1

THE PHASE MAP
-------------
CityFlow's file carries 9 light phases (index 0 = the all-red clearance, 1-8 the greens); SUMO's
``tlLogic`` carries 16 (greens at 0, 2, ..., 14; every odd index all-red).  So::

    green action k  ->  CityFlow file phase k + 1  ->  SUMO file phase 2k
    SUMO phase 2k   ->  CityFlow phase k + 1;  SUMO odd phase -> CityFlow phase 0

**The clearance slot is dead in the data the models trained on**, measured on the population rather
than a sample: 577,600 ``current_phase`` rows across all 1,600 hz1x1 episodes of ``datasets_v11/``
carry phases 1-8 and none carries 0, because a transition always completes inside the 10 s step.
The map is therefore exact on everything the corpus contains, and the width-9 encoding keeps the
dead slot so the frozen feature width does not move.

WHAT ``align_info`` DOES AND REFUSES
------------------------------------
Per intersection: the ``lane_vehicle_count`` and ``lane_waiting`` blocks of ``state`` are permuted
into the canonical order **through the movement correspondence**; ``phase_onehot`` is re-encoded to
width 9 and ``current_phase`` mapped by the rule above; ``avail_actions`` passes through with its
width asserted equal to CityFlow's action count.  Globally the two lane dicts are re-keyed to
CityFlow lane ids -- **a key translation, and the SUMO id is kept in a sidecar rather than lost** --
and ``metrics`` is filtered to a declared tuple (C8: ``SumoEnv`` unions ``average_travel_time`` into
its metric set and CityFlow does not, so a checkpoint trained on one raises on the other).
Everything else is passed through untouched.

**It raises on any lane it cannot place. There is no positional fallback**, because a positional
fallback is exactly the defect above wearing a different hat.

IDEMPOTENCE, CHOSEN AND DOCUMENTED
----------------------------------
``align_info`` applied to an already-canonical ``info`` **RAISES**.  The alternative -- detect and
no-op -- hides a double application, and a double application means some caller's model of the
pipeline is wrong.  The precondition it fails on is real and cheap: a canonical ``info``'s phase
block is 9 wide and its lane ids are CityFlow's, neither of which a SUMO ``info`` has.

PURITY
------
The input dict is never mutated; a new dict is returned.  ``tests/test_backend_alignment.py``
deep-copies the input, calls the function and asserts the original compares equal afterwards.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "ALIGNMENT_FORMAT_VERSION",
    "CITYFLOW_CLEARANCE_PHASE",
    "DEFAULT_METRIC_KEYS",
    "IntersectionAlignment",
    "ScenarioAlignment",
    "align_info",
    "alignment_for_scenario",
    "canonical_order_artifact",
    "cityflow_phase_for_sumo_phase",
    "paired_scenario_table",
    "sumo_incoming_lanes_from_net",
    "sumo_phase_for_action",
    "sumo_phase_counts_from_net",
]

#: On-disk format of ``docs/data/p7_1_canonical_order.json`` and ``p7_1_paired_scenarios.json``.
ALIGNMENT_FORMAT_VERSION = "p7.1-alignment/1.0"

#: CityFlow's file phase 0: the all-red clearance, with an empty ``availableRoadLinks``.  Never
#: observed at a decision boundary in the corpus (577,600 rows), and kept only so the one-hot width
#: matches what the models were trained on.
CITYFLOW_CLEARANCE_PHASE = 0

#: The metric keys the aligned ``info`` carries.  Empty by default: the DT reads per-intersection
#: ``state`` only, and C8 makes any wider set a checkpoint-compatibility hazard rather than a
#: convenience.  A caller that wants metrics must name them.
DEFAULT_METRIC_KEYS: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntersectionAlignment:
    """How one intersection's SUMO observation is rewritten into the CityFlow corpus's frame."""

    intersection_id: str
    canonical_lanes: tuple[str, ...]
    sumo_lanes: tuple[str, ...]
    correspondence: Mapping[str, str]
    permutation: tuple[int, ...]
    cityflow_num_phases: int
    sumo_num_phases: int
    n_actions: int

    def canonical_state_width(self) -> int:
        """``2 * L + cityflow_num_phases`` -- what an aligned ``state`` must measure."""
        return 2 * len(self.canonical_lanes) + int(self.cityflow_num_phases)

    def sumo_state_width(self) -> int:
        """``2 * L + sumo_num_phases`` -- what the SUMO ``state`` handed in must measure."""
        return 2 * len(self.sumo_lanes) + int(self.sumo_num_phases)


@dataclass(frozen=True)
class ScenarioAlignment:
    """Every intersection of one paired scenario, plus the metric policy."""

    scenario: str
    intersections: Mapping[str, IntersectionAlignment]
    metric_keys: tuple[str, ...]
    #: Every lane id the CityFlow roadnet declares, incoming and outgoing.  ``align_info`` drops the
    #: outgoing ones (they are monitored by ``SumoMetrics`` but are not part of the canonical
    #: incoming frame) and REFUSES anything outside this set -- exact membership, never a prefix.
    known_lanes: frozenset[str] = frozenset()
    #: True where the pair is hangzhou-shaped (CityFlow 9, SUMO 16) and the ``2k -> k+1`` map
    #: applies; False where the two programs have the same width and the map is the identity.
    phase_map_is_hangzhou_shaped: bool = True


def sumo_phase_for_action(action: int) -> int:
    """Green action ``k`` -> SUMO file phase ``2k``."""
    index = int(action)
    if index < 0:
        raise ValueError(f"a green action index cannot be negative, got {action!r}")
    return 2 * index


def cityflow_phase_for_sumo_phase(phase: int, *, n_sumo_phases: int = 16) -> int:
    """SUMO file phase -> CityFlow file phase: ``2k -> k + 1``, odd -> the clearance slot.

    ⚠️ **This map is HANGZHOU-SHAPED and is only correct where CityFlow has 9 phases against SUMO's
    16** -- i.e. where the conversion collapsed SUMO's eight identical clearance phases into
    CityFlow's single phase 0.  On a same-width pair such as grid4x4 (16 against 16) the two
    programs are in 1:1 correspondence and the identity is the map; applying this one there sends
    SUMO phase 14 to CityFlow phase 8, which on that scenario means something else entirely.
    :func:`align_info` chooses by :attr:`ScenarioAlignment.phase_map_is_hangzhou_shaped`.

    The odd phases map onto CityFlow's phase 0 and that slot is DEAD in the corpus (577,600 rows,
    none of them 0), so that branch exists to be total rather than to be exercised.
    """
    index = int(phase)
    if not 0 <= index < int(n_sumo_phases):
        raise ValueError(
            f"a SUMO file phase must be in [0, {int(n_sumo_phases)}), got {phase!r}; the map is "
            "defined on the tlLogic's own phase indices and a phase outside them would silently "
            "become a green the program does not have"
        )
    if index % 2:
        return CITYFLOW_CLEARANCE_PHASE
    return index // 2 + 1


def sumo_incoming_lanes_from_net(net_xml: str | Path) -> dict[str, tuple[str, ...]]:
    """Each traffic light's controlled incoming lanes, SORTED, from the ``.net.xml`` alone.

    Mirrors what ``utils/sumo_utils.py:62-102`` builds from ``getControlledLinks`` at run time and
    then ``sorted()``s -- derived from the file so an alignment can be built with no simulator.
    ⚠️ It is DERIVED, so it is also CHECKED: a SUMO-gated test asserts this equals the live env's
    ``incoming_lanes`` on hz1x1, which is the only thing that makes the derivation trustworthy.
    """
    root = ET.parse(Path(net_xml)).getroot()
    by_tls: dict[str, set[str]] = {}
    for connection in root.findall("connection"):
        tls = connection.get("tl")
        edge = connection.get("from")
        if tls is None or edge is None or edge.startswith(":"):
            continue
        by_tls.setdefault(tls, set()).add(f"{edge}_{int(str(connection.get('fromLane')))}")
    return {tls: tuple(sorted(lanes)) for tls, lanes in by_tls.items()}


def sumo_phase_counts_from_net(net_xml: str | Path) -> dict[str, int]:
    """Each traffic light's phase count, from its ``<tlLogic>`` -- 16 on the hangzhou networks."""
    root = ET.parse(Path(net_xml)).getroot()
    return {
        str(logic.get("id")): len(logic.findall("phase"))
        for logic in root.findall("tlLogic")
    }


def alignment_for_scenario(
    scenario: str,
    *,
    cityflow_roadnet: str | Path,
    sumo_net: str | Path,
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
    intersections: Sequence[str] | None = None,
) -> ScenarioAlignment:
    """Build the alignment from the two network FILES, never from a running env.

    The CityFlow side comes from the FROZEN parser (``utils/cityflow_utils.parse_roadnet``), so the
    canonical order is by construction the one the corpus was collected in -- including its handling
    of virtual intersections, which the two conversion directions mark differently (``virtual`` on a
    CityFlow-native roadnet, ``gt_virtual`` on an s2c-converted one; cologne3 has 29 entries and 3
    real intersections).  The SUMO side comes from :func:`sumo_incoming_lanes_from_net`.

    The correspondence comes from :func:`transfer_gate.lane_semantic_correspondence`, which pairs
    each CityFlow lane with the SUMO lane on the same road serving the same movement set and refuses
    anything but a unique match.  Raises if an intersection is missing from either side, if the two
    lane counts differ, or if the derived permutation is not a permutation.
    """
    from utils.cityflow_utils import parse_roadnet
    from offline.transfer_gate import cityflow_lane_turns, lane_semantic_correspondence, sumo_lane_turns

    roadnet_path = Path(cityflow_roadnet)
    net_path = Path(sumo_net)
    parsed = parse_roadnet(roadnet_path)
    sumo_lanes_by_tls = sumo_incoming_lanes_from_net(net_path)
    sumo_phases = sumo_phase_counts_from_net(net_path)
    sumo_turns = sumo_lane_turns(net_path)

    wanted = list(intersections) if intersections is not None else [ix.id for ix in parsed.intersections]
    by_id = {ix.id: ix for ix in parsed.intersections}
    built: dict[str, IntersectionAlignment] = {}
    for ix_id in wanted:
        if ix_id not in by_id:
            raise KeyError(
                f"{ix_id!r} is not a non-virtual intersection of {roadnet_path}; the CityFlow side "
                f"has {sorted(by_id)}"
            )
        if ix_id not in sumo_lanes_by_tls:
            raise KeyError(
                f"{ix_id!r} controls no traffic light in {net_path}; the two files disagree on the "
                f"intersection key, which is a topology finding and not something to map by "
                f"position. SUMO has {sorted(sumo_lanes_by_tls)[:5]}"
            )
        entry = by_id[ix_id]
        canonical = tuple(str(lane) for lane in entry.incoming_lanes)
        sumo_lanes = sumo_lanes_by_tls[ix_id]
        if len(canonical) != len(sumo_lanes):
            raise ValueError(
                f"{ix_id!r}: CityFlow controls {len(canonical)} incoming lanes and SUMO "
                f"{len(sumo_lanes)}; a lane-count difference is a topology difference and cannot be "
                "aligned by a permutation"
            )
        correspondence = lane_semantic_correspondence(
            cityflow_lane_turns(roadnet_path, ix_id), sumo_turns, canonical
        )
        index_of = {lane: index for index, lane in enumerate(sumo_lanes)}
        permutation = tuple(index_of[correspondence[lane]] for lane in canonical)
        if sorted(permutation) != list(range(len(canonical))):
            raise ValueError(
                f"{ix_id!r}: the movement correspondence is not one-to-one onto SUMO's lanes "
                f"({permutation}); two canonical lanes claim the same SUMO lane"
            )
        built[ix_id] = IntersectionAlignment(
            intersection_id=ix_id,
            canonical_lanes=canonical,
            sumo_lanes=sumo_lanes,
            correspondence=dict(correspondence),
            permutation=permutation,
            cityflow_num_phases=int(entry.num_phases),
            sumo_num_phases=int(sumo_phases[ix_id]),
            n_actions=_env_green_action_count(entry),
        )
    shapes = {
        (ix.cityflow_num_phases, ix.sumo_num_phases) for ix in built.values()
    }
    if len(shapes) > 1:
        raise ValueError(
            f"{scenario}: the intersections do not share a phase shape ({sorted(shapes)}); one "
            "phase map cannot serve two of them and this must be handled per intersection or "
            "reported, never averaged"
        )
    hangzhou_shaped = shapes == {(9, 16)}
    if not hangzhou_shaped and shapes and any(cf != su for cf, su in shapes):
        raise ValueError(
            f"{scenario}: phase counts {sorted(shapes)} are neither hangzhou-shaped (9, 16) nor "
            "equal on the two sides, so neither the 2k -> k+1 map nor the identity is correct here. "
            "A third mapping is design work, not a default"
        )
    return ScenarioAlignment(
        scenario=str(scenario),
        intersections=built,
        metric_keys=tuple(metric_keys),
        known_lanes=frozenset(str(lane) for lane in parsed.lane_ids),
        phase_map_is_hangzhou_shaped=hangzhou_shaped,
    )


def _env_green_action_count(entry: Any) -> int:
    """How many GREEN actions the env exposes, by the env's own rule.

    ⚠️ **Not "phases with a non-empty roadlink list".**  ``envs/phase_control.py`` treats a phase as
    a transition when its duration is at most ``TRANSITION_PHASE_MAX_DURATION``; on grid4x4 every one
    of the 16 phases releases something (its clearance phases keep right turns on ``s``), so counting
    non-empty lists gives 16 where the env gives 8.  Amendment C(D) asked for the env's number and
    this is it.
    """
    from envs.phase_control import TRANSITION_PHASE_MAX_DURATION

    durations = list(getattr(entry, "phase_durations", None) or [])
    mapping = list(getattr(entry, "phase_roadlink_mapping", None) or [])
    if not durations:
        return sum(1 for links in mapping if links)
    return sum(
        1
        for index, duration in enumerate(durations)
        if float(duration) > float(TRANSITION_PHASE_MAX_DURATION)
        and (index >= len(mapping) or mapping[index])
    )


def align_info(
    info: Mapping[str, Any], alignment: ScenarioAlignment
) -> dict[str, Any]:
    """Rewrite a SUMO ``info`` (C2) into the CityFlow corpus's frame, or raise.

    Pure: *info* is not mutated.  See the module docstring for what moves, what is re-keyed, and why
    a second application raises instead of being a no-op.
    """
    payloads = info.get("intersections")
    if not isinstance(payloads, Mapping):
        raise ValueError("the info carries no 'intersections' mapping, so it is not a C2 info")
    unknown = sorted(set(payloads) - set(alignment.intersections))
    if unknown:
        raise KeyError(
            f"the info carries {len(unknown)} intersection(s) the alignment does not know "
            f"({unknown[:3]}); an intersection cannot be aligned by position"
        )
    missing = sorted(set(alignment.intersections) - set(payloads))
    if missing:
        raise KeyError(
            f"the info is missing {len(missing)} aligned intersection(s) ({missing[:3]}); a partial "
            "info would produce a state vector for a network that is not the one being observed"
        )

    lane_counts = info.get("lane_vehicle_count", {})
    lane_waiting = info.get("lane_waiting_vehicle_count", {})
    translated_counts: dict[str, Any] = {}
    translated_waiting: dict[str, Any] = {}
    translation: dict[str, str] = {}
    known_sumo_lanes: set[str] = set()

    aligned_payloads: dict[str, Any] = {}
    for ix_id, ix in alignment.intersections.items():
        payload = payloads[ix_id]
        state = list(payload["state"])
        if len(state) != ix.sumo_state_width():
            raise ValueError(
                f"{ix_id}: the state width is {len(state)} but this SUMO intersection produces "
                f"{ix.sumo_state_width()} (2 x {len(ix.sumo_lanes)} lanes + "
                f"{ix.sumo_num_phases} phases). An already-aligned info has width "
                f"{ix.canonical_state_width()} and is refused here rather than aligned twice"
            )
        n_lanes = len(ix.sumo_lanes)
        counts_block = state[:n_lanes]
        waiting_block = state[n_lanes : 2 * n_lanes]
        onehot = state[2 * n_lanes :]
        hot = [index for index, value in enumerate(onehot) if value]
        if len(hot) != 1:
            raise ValueError(
                f"{ix_id}: the phase one-hot has {len(hot)} hot entries, not 1; it cannot be "
                "re-encoded into CityFlow's width without inventing a phase"
            )
        cityflow_phase = (
            cityflow_phase_for_sumo_phase(hot[0], n_sumo_phases=ix.sumo_num_phases)
            if alignment.phase_map_is_hangzhou_shaped
            else hot[0]
        )
        if int(payload["current_phase"]) != hot[0]:
            raise ValueError(
                f"{ix_id}: current_phase is {payload['current_phase']!r} but the one-hot is hot at "
                f"{hot[0]}; the two disagree about which phase is running"
            )
        # ⚠️ C2: avail_actions are the legal actions RIGHT NOW, so a SHORT list is normal and must
        # pass. What cannot pass is an INDEX the canonical frame has no green phase for -- that
        # would hand the model an action it was never trained to take.
        avail = list(payload["avail_actions"])
        out_of_range = [a for a in avail if not 0 <= int(a) < ix.n_actions]
        if out_of_range:
            raise ValueError(
                f"{ix_id}: avail_actions carries {out_of_range} outside CityFlow's "
                f"{ix.n_actions} green actions; the canonical frame has no phase for those, so "
                "passing them through would offer the model an action it cannot mean"
            )

        aligned_counts = [counts_block[index] for index in ix.permutation]
        aligned_waiting = [waiting_block[index] for index in ix.permutation]
        aligned_onehot = [0.0] * int(ix.cityflow_num_phases)
        aligned_onehot[cityflow_phase] = 1.0

        aligned_payloads[ix_id] = {
            **{k: v for k, v in payload.items() if k not in {"state", "current_phase"}},
            "state": aligned_counts + aligned_waiting + aligned_onehot,
            "current_phase": cityflow_phase,
        }

        for canonical_lane in ix.canonical_lanes:
            partner = ix.correspondence[canonical_lane]
            known_sumo_lanes.add(partner)
            translation[canonical_lane] = partner
            for source, destination in (
                (lane_counts, translated_counts),
                (lane_waiting, translated_waiting),
            ):
                if partner not in source:
                    raise KeyError(
                        f"{partner!r} is in the correspondence for {canonical_lane!r} but absent "
                        "from the info's lane dicts; the observation does not cover the network "
                        "the alignment was built from"
                    )
                destination[canonical_lane] = source[partner]

    # ⚠️ Outgoing lanes are monitored by SumoMetrics and are NOT part of the canonical incoming
    # frame; they are dropped rather than carried under a SUMO id, and the count is reported so the
    # drop is visible. A lane that is neither incoming nor a known outgoing one is a refusal.
    for lane in lane_counts:
        if lane in known_sumo_lanes:
            continue
        if any(lane in ix.sumo_lanes for ix in alignment.intersections.values()):
            continue
        if _is_a_known_lane(lane, alignment):
            continue
        raise KeyError(
            f"{lane!r} appears in the info's lane dict but belongs to no intersection this "
            "alignment knows; a lane that cannot be placed is refused, never dropped silently"
        )

    metrics = info.get("metrics", {}) or {}
    absent = [key for key in alignment.metric_keys if key not in metrics]
    if absent:
        raise ValueError(
            f"the declared metric_keys {absent} are not in the info's metrics {sorted(metrics)}; "
            "C8 makes the metric SET part of a MAPPO checkpoint's MDP, so a silently smaller set is "
            "a different environment rather than a smaller table"
        )

    aligned = {
        key: value
        for key, value in info.items()
        if key
        not in {"intersections", "lane_vehicle_count", "lane_waiting_vehicle_count", "metrics"}
    }
    aligned["intersections"] = aligned_payloads
    aligned["lane_vehicle_count"] = translated_counts
    aligned["lane_waiting_vehicle_count"] = translated_waiting
    aligned["metrics"] = {key: metrics[key] for key in alignment.metric_keys}
    aligned["lane_id_translation"] = translation
    return aligned


def _is_a_known_lane(lane: str, alignment: ScenarioAlignment) -> bool:
    """Exact membership of the roadnet's incoming-or-outgoing lane set.

    🚨 **This was a name-PREFIX test and it silently dropped lanes.**  The half-B review probed
    ``road_1_1_99_0`` -- no such road, but a prefix of a known intersection's naming -- and the
    adapter ACCEPTED and dropped it, while ``road_9_9_9_0`` raised.  A prefix is not membership, and
    the module's own contract is that it *"raises on any lane it cannot place"*.  The set is now the
    roadnet's own, read once when the alignment is built.
    """
    return lane in alignment.known_lanes


def canonical_order_artifact(
    alignments: Sequence[ScenarioAlignment], *, roadnets: Mapping[str, str | Path]
) -> dict[str, Any]:
    """``docs/data/p7_1_canonical_order.json``: each canonical lane with its road, index and moves.

    The frozen half of the feature freeze: this file is what a later task compares against to show
    the order has not moved.  Every lane carries the movement set it serves on the CityFlow side,
    so a reader can see that position 0 is a through lane and position 1 a left turn without
    re-deriving it.
    """
    from offline.transfer_gate import cityflow_lane_turns

    scenarios: dict[str, Any] = {}
    for alignment in alignments:
        roadnet = roadnets[alignment.scenario]
        entries: dict[str, Any] = {}
        for ix_id, ix in alignment.intersections.items():
            turns = cityflow_lane_turns(roadnet, ix_id)
            entries[ix_id] = {
                "canonical_order": [
                    {
                        "position": position,
                        "lane": lane,
                        "road": lane.rsplit("_", 1)[0],
                        "lane_index": int(lane.rsplit("_", 1)[1]),
                        "movements": sorted(turns.get(lane, ())),
                        "sumo_lane": ix.correspondence[lane],
                        "sumo_position": ix.permutation[position],
                    }
                    for position, lane in enumerate(ix.canonical_lanes)
                ],
                "cityflow_num_phases": ix.cityflow_num_phases,
                "sumo_num_phases": ix.sumo_num_phases,
                "n_green_actions": ix.n_actions,
                "canonical_state_width": ix.canonical_state_width(),
                "sumo_state_width": ix.sumo_state_width(),
                "n_lanes_denoting_a_different_physical_lane": sum(
                    1 for lane, partner in ix.correspondence.items() if lane != partner
                ),
            }
        scenarios[alignment.scenario] = {
            "roadnet": str(roadnet),
            "intersections": entries,
        }
    return {
        "format_version": ALIGNMENT_FORMAT_VERSION,
        "task": "P7.1",
        "role": (
            "the frozen canonical feature order: the CityFlow corpus's own per-intersection "
            "incoming_lanes discovery order, with each lane's movement set and its SUMO partner"
        ),
        "what_this_does_not_say": [
            "It does not claim the two backends' lanes are interchangeable: the pairing is by "
            "MOVEMENT, and 8 of 8 lane ids on hz1x1 denote a different physical lane.",
            "It fixes an ORDER and a PHASE MAP, not a state normalisation: obs_norm is out of "
            "scope and unchanged.",
        ],
        "scenarios": scenarios,
    }


def paired_scenario_table(repo_root: str | Path) -> dict[str, Any]:
    """``docs/data/p7_1_paired_scenarios.json``: §3.1's inventory, built from files alone.

    One row per ``scenarios/**/*.sumocfg``: whether its inputs exist, which CityFlow config points at
    the same directory, lane-set and monitored-set equality, intersection-id equality, the
    correspondence outcome per intersection **with the raising message when it fails**, the vType
    binding, lane-speed extremes and phase counts.  grid4x4 is a ROW (inputs missing), not an
    omission; cologne3's failure is recorded with counts and is not fixed.

    ⚠️ Every field is derived from the two files.  Nothing here starts a simulator, so a scenario
    whose engine cannot run is still audited.
    """
    from utils.cityflow_utils import parse_roadnet
    from offline.transfer_gate import cityflow_lane_turns, lane_semantic_correspondence, sumo_lane_turns

    root = Path(repo_root).resolve()
    cityflow_configs = _cityflow_configs_by_directory(root)
    rows: list[dict[str, Any]] = []
    for sumocfg in sorted((root / "scenarios").rglob("*.sumocfg")):
        # ⚠️ The gitignored candidates clones are NOT our scenarios: they are third-party,
        # CC BY-NC-SA and present on one machine. Walking them made this table 44 rows on a tree
        # that had them and 20 on a clone, and read files the task is not licensed to ship.
        if any(part.endswith("_candidates") for part in sumocfg.parts):
            continue
        row: dict[str, Any] = {
            "sumocfg": str(sumocfg.relative_to(root)),
            "scenario_dir": str(sumocfg.parent.relative_to(root)),
        }
        inputs = _sumocfg_inputs(sumocfg, root)
        row["inputs"] = inputs
        row["inputs_exist"] = all(entry["exists"] for entry in inputs.values())
        config = cityflow_configs.get(sumocfg.parent.resolve())
        row["cityflow_config"] = None if config is None else str(Path(config["path"]).relative_to(root))
        row["cityflow_roadnet"] = None if config is None else config["roadnet"]
        # ⚠️ A .sumocfg may reference a network in ANOTHER scenario directory -- the parity files
        # do exactly that, deliberately, so two copies of one network cannot drift. Without this
        # field their rows read as unpaired, which they are not.
        net_declared = inputs.get("net-file", {}).get("resolved")
        if config is None and net_declared is not None:
            sibling = cityflow_configs.get((root / net_declared).resolve().parent)
            row["pairs_through"] = (
                None if sibling is None else str(Path(sibling["path"]).relative_to(root))
            )

        if not row["inputs_exist"]:
            row["status"] = "sumo inputs missing"
            rows.append(row)
            continue
        net = root / inputs["net-file"]["resolved"]
        row["sumo"] = _sumo_network_facts(net)
        row["vtype_binding"] = _route_vtype_facts(
            [root / inputs[key]["resolved"] for key in inputs if key.startswith("route")]
        )
        if config is None:
            row["status"] = (
                "pairs through the sibling directory its network lives in"
                if row.get("pairs_through")
                else "SUMO only: no CityFlow config names this directory"
            )
            rows.append(row)
            continue

        roadnet_path = root / config["roadnet"]
        parsed = parse_roadnet(roadnet_path)
        cityflow_lanes = set(parsed.lane_ids)
        sumo_lanes = set(row["sumo"]["lane_ids"])
        monitored = {
            lane
            for ix in parsed.intersections
            for lane in list(ix.incoming_lanes) + list(ix.outgoing_lanes)
        }
        row["cityflow"] = {
            "n_intersections": len(parsed.intersections),
            "intersection_ids": [ix.id for ix in parsed.intersections],
            "n_lanes": len(cityflow_lanes),
            "num_phases": sorted({ix.num_phases for ix in parsed.intersections}),
        }
        row["lane_sets_equal"] = cityflow_lanes == sumo_lanes
        row["n_lanes_only_in_cityflow"] = len(cityflow_lanes - sumo_lanes)
        row["n_lanes_only_in_sumo"] = len(sumo_lanes - cityflow_lanes)
        # What SumoMetrics actually logs: incoming | outgoing of the controlled lights
        # (metrics/sumo.py:62-66). A corpus collected on SUMO would carry THIS lane set.
        row["monitored_set"] = {
            "n_cityflow_monitored": len(monitored),
            "n_cityflow_all_lanes": len(cityflow_lanes),
            "monitored_equals_all": monitored == cityflow_lanes,
        }
        row["intersection_ids_equal"] = sorted(row["cityflow"]["intersection_ids"]) == sorted(
            row["sumo"]["traffic_light_ids"]
        )

        # Amendment H2 item 8: the direction counts belong in the row, not only in prose. cologne3's
        # `t` 39 / `L` 1 / `R` 1 are WHY its correspondence raises, and a reader of the table should
        # not have to open the network to see it.
        from offline.conversion_audit import sumo_connection_counts

        row["connection_directions"] = sumo_connection_counts(net)["from_non_internal_edges"]
        sumo_turns = sumo_lane_turns(net)
        outcomes: dict[str, Any] = {}
        for ix in parsed.intersections:
            try:
                turns = cityflow_lane_turns(roadnet_path, ix.id)
                mapping = lane_semantic_correspondence(
                    turns, sumo_turns, [str(lane) for lane in ix.incoming_lanes]
                )
            except (KeyError, ValueError) as exc:
                outcomes[ix.id] = {"resolved": False, "error": f"{type(exc).__name__}: {exc}"}
                continue
            outcomes[ix.id] = {
                "resolved": True,
                "n_lanes": len(mapping),
                "n_denoting_a_different_lane": sum(1 for k, v in mapping.items() if k != v),
            }
        row["correspondence"] = {
            "n_resolved": sum(1 for entry in outcomes.values() if entry["resolved"]),
            "n_intersections": len(outcomes),
            "per_intersection": outcomes,
        }
        row["status"] = (
            "alignable"
            if row["correspondence"]["n_resolved"] == row["correspondence"]["n_intersections"]
            and row["lane_sets_equal"]
            and row["intersection_ids_equal"]
            else "paired but NOT alignable as the files stand"
        )
        rows.append(row)

    unpaired = [
        {"config": str(Path(entry["path"]).relative_to(root)), "scenario_dir": str(directory.relative_to(root))}
        for directory, entry in sorted(cityflow_configs.items(), key=lambda kv: str(kv[0]))
        if not any((root / row["scenario_dir"]).resolve() == directory for row in rows)
    ]
    return {
        "format_version": ALIGNMENT_FORMAT_VERSION,
        "task": "P7.1",
        "role": (
            "which scenario pairs exist on disk and which of them the movement correspondence can "
            "align; it reports, it does not repair"
        ),
        "what_this_does_not_say": [
            "'alignable' means the lane sets, the intersection ids and the movement correspondence "
            "agree -- NOT that the two engines' dynamics agree, and not that a vType parity file "
            "exists.",
            "cologne3's failure is recorded with its message and is NOT fixed: DEFERRED 74 owns "
            "the topology-level design work.",
        ],
        "n_sumocfg": len(rows),
        "n_alignable": sum(1 for row in rows if row.get("status") == "alignable"),
        "rows": rows,
        "cityflow_configs_without_a_sumo_pair": unpaired,
    }


def _cityflow_configs_by_directory(root: Path) -> dict[Path, dict[str, str]]:
    """Every ``configs/sim/*.json`` keyed by the scenario directory it names."""
    found: dict[Path, dict[str, str]] = {}
    for config in sorted((root / "configs" / "sim").glob("*.json")):
        payload = json.loads(config.read_text(encoding="utf-8"))
        directory = payload.get("dir")
        roadnet = payload.get("roadnetFile")
        if not directory or not roadnet:
            continue
        resolved = (root / str(directory)).resolve()
        found.setdefault(
            resolved,
            {"path": str(config), "roadnet": str(Path(directory) / str(roadnet))},
        )
    return found


def _sumocfg_inputs(sumocfg: Path, repo_root: Path) -> dict[str, dict[str, Any]]:
    """The files a ``.sumocfg`` declares, resolved against its own directory, with existence.

    Paths are recorded RELATIVE to the repo root: an artifact that embeds one contributor's home
    directory is not the same artifact on a clone, and this one is meant to regenerate byte-identically.
    """
    root = ET.parse(sumocfg).getroot()
    inputs: dict[str, dict[str, Any]] = {}
    for element in root.iter():
        if element.tag not in {"net-file", "route-files", "route-file", "additional-files"}:
            continue
        value = str(element.get("value") or "")
        for index, name in enumerate(
            part for chunk in value.split(",") for part in chunk.split()
        ):
            resolved = (sumocfg.parent / name).resolve()
            key = element.tag if index == 0 else f"{element.tag}[{index}]"
            try:
                recorded = str(resolved.relative_to(repo_root))
            except ValueError:
                recorded = str(resolved)
            inputs[key] = {
                "declared": name,
                "resolved": recorded,
                "exists": resolved.is_file(),
            }
    return inputs


def _sumo_network_facts(net: Path) -> dict[str, Any]:
    """Lane ids, traffic-light ids and phase counts, and the lane-speed extremes, from a net file."""
    root = ET.parse(net).getroot()
    lane_ids: list[str] = []
    speeds: list[float] = []
    lengths: list[float] = []
    for edge in root.findall("edge"):
        if edge.get("function") == "internal" or str(edge.get("id", "")).startswith(":"):
            continue
        for lane in edge.findall("lane"):
            lane_ids.append(str(lane.get("id")))
            speeds.append(float(str(lane.get("speed"))))
            lengths.append(float(str(lane.get("length"))))
    logics = {str(logic.get("id")): len(logic.findall("phase")) for logic in root.findall("tlLogic")}
    location = root.find("location")
    return {
        "n_lanes": len(lane_ids),
        "lane_ids": sorted(lane_ids),
        "traffic_light_ids": sorted(logics),
        "num_phases": sorted(set(logics.values())),
        "lane_speed_min": min(speeds) if speeds else None,
        "lane_speed_max": max(speeds) if speeds else None,
        "lane_length_min": min(lengths) if lengths else None,
        "lane_length_max": max(lengths) if lengths else None,
        "net_offset": None if location is None else location.get("netOffset"),
        "proj_parameter": None if location is None else location.get("projParameter"),
    }


def _route_vtype_facts(route_files: Sequence[Path]) -> dict[str, Any]:
    """Whether a vType is DEFINED, whether it is BOUND, and whether it sets ``tau``.

    ⚠️ Defined is not bound: the shipped hangzhou route files declare a correct-looking ``pkw`` and
    bind it to 0 of 2021 vehicles, which is the defect ``offline/parity.py`` exists because of.
    """
    n_vehicles = 0
    n_typed = 0
    types: dict[str, dict[str, str]] = {}
    for route_file in route_files:
        if not route_file.is_file():
            continue
        root = ET.parse(route_file).getroot()
        for vtype in root.iter("vType"):
            types[str(vtype.get("id"))] = {k: str(v) for k, v in vtype.attrib.items()}
        for vehicle in root.iter("vehicle"):
            n_vehicles += 1
            n_typed += 1 if vehicle.get("type") else 0
    return {
        "n_vehicles": n_vehicles,
        "n_typed": n_typed,
        "fully_bound": bool(n_vehicles) and n_typed == n_vehicles,
        "vtypes": types,
        "tau_present": {name: "tau" in attrs for name, attrs in types.items()},
    }


# ----------------------------------------------------------------------
# The regenerating CLI -- Amendment H2 item 4
# ----------------------------------------------------------------------

#: The scenarios `alignment_for_scenario` is run over for the canonical-order artifact.  Declared
#: here rather than passed on a command line so a regeneration cannot quietly change its scope.
CANONICAL_SCENARIOS: tuple[tuple[str, str, str], ...] = (
    (
        "hangzhou_1x1_bc-tyc",
        "scenarios/hangzhou_1x1_bc-tyc_18041610_1h/roadnet.json",
        "scenarios/hangzhou_1x1_bc-tyc_18041610_1h/hangzhou_1x1_bc-tyc_18041610_1h.net.xml",
    ),
    (
        "hangzhou_4x4_gudang",
        "scenarios/hangzhou_4x4_gudang_18041610_1h/roadnet_4X4.json",
        "scenarios/hangzhou_4x4_gudang_18041610_1h/hangzhou_4x4_gudang_18041610_1h.net.xml",
    ),
)


def write_json_stable(payload: Mapping[str, Any], destination: str | Path) -> Path:
    """Write an artifact deterministically, so a regeneration is byte-comparable.

    ``indent=2`` and a trailing newline, exactly as the committed files carry, and no timestamp or
    absolute path anywhere in the payload -- the whole point of Amendment H2 item 4 is that a second
    run of the same command produces the same bytes.
    """
    target = Path(destination)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def build_parser() -> "Any":
    """``python -m offline.backend_alignment tables`` -- regenerate all three artifacts."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m offline.backend_alignment",
        description=(
            "P7.1 half B: regenerate the canonical-order, paired-scenario and conversion-audit "
            "artifacts from the files. Deterministic: the same tree gives the same bytes."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="docs/data")
    parser.add_argument(
        "--candidates-root",
        default=None,
        help=(
            "the read-only, gitignored third-party clones (RESCO / LibSignal). Paths from it are "
            "recorded RELATIVE to it, so the artifact is the same on a machine that has it "
            "elsewhere; without it the grid4x4 row is omitted and the artifact says so."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("tables", help="write all three artifacts", allow_abbrev=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate the three half-B artifacts; returns a process exit code."""
    from offline import conversion_audit

    args = build_parser().parse_args(argv)
    root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        print(f"REFUSED: {out_dir} is not a directory; this command never creates one", flush=True)
        return 2

    alignments = []
    roadnets = {}
    for scenario, roadnet, net in CANONICAL_SCENARIOS:
        alignments.append(
            alignment_for_scenario(scenario, cityflow_roadnet=root / roadnet, sumo_net=root / net)
        )
        roadnets[scenario] = roadnet
    written = [
        write_json_stable(
            canonical_order_artifact(alignments, roadnets=roadnets),
            out_dir / "p7_1_canonical_order.json",
        ),
        write_json_stable(
            paired_scenario_table(root), out_dir / "p7_1_paired_scenarios.json"
        ),
        write_json_stable(
            conversion_audit.audit_all(root, candidates_root=args.candidates_root),
            out_dir / "p7_1_conversion_audit.json",
        ),
    ]
    for path in written:
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
