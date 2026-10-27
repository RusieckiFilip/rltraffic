"""MaxPressure with SET semantics -- A27(f)'s anchor-alignment control ``maxpressure_1to1`` (P9.2, ``BRIEF_47`` §3.1).

Format version of the provenance this module writes into a cell: ``p9.2-maxpressure-1to1/1.0``
(:data:`PROVENANCE_FORMAT_VERSION`).  Alignment convention: none -- this is a policy and a structural record; it
records no trajectory.  The action an agent returns for ``info_t`` is the green-action index of the acyclic controls,
one ``int64`` per intersection in ``env.intersections`` order (contract C1), exactly as the frozen agent returns it.

WHAT THE REGISTERED ANCHOR COMPUTES, AND WHY A VARIANT EXISTS
-------------------------------------------------------------
``algorithms.max_pressure.MaxPressureAgent._phase_pressure`` (frozen) sums PER LINK::

    pressure(phase) = sum over the phase's links of [ sum(count of the link's incoming lanes)
                                                       - sum(count of the link's outgoing lanes) ]

so a lane released by two of the phase's links is counted twice.  Measured on 2026-10-15 (A27(f)) and again by
``tests/test_max_pressure_1to1.py``: on SUMO hz1x1 every selectable green releases 4 links over 2 distinct incoming
lanes (each twice) into 4 distinct outgoing lanes (each once) -- the upstream queue weighs 2:1 against the downstream;
on SUMO grid4x4 every selectable green visits each released incoming lane three times; on CityFlow grid4x4 incoming
lanes once but an outgoing lane is fed by two roadlinks on every selectable green; on CityFlow hz1x1 every lane once.
The registered anchor therefore means a different weighting on each backend and scenario.

THE VARIANT, A27(f) VERBATIM IN SUBSTANCE
-----------------------------------------
*pressure(phase) = sum over the DISTINCT incoming lanes the phase releases of their vehicle count - sum over the
DISTINCT outgoing lanes it feeds* (:func:`set_pressure`).  Where every lane appears once per phase the two formulas are
the same number; elsewhere they differ by exactly ``sum_in (visits - 1) * count - sum_out (visits - 1) * count``.

**What is inherited, not copied.**  :class:`MaxPressure1to1Agent` subclasses the frozen agent and overrides ONLY the
static ``_phase_pressure``.  ``act``, ``_acyclic_action`` (the candidate set from ``avail_actions``, the first-maximum
tie-breaking) and ``_green_phases`` (the acyclic controls' selectable greens, file duration above
``TRANSITION_PHASE_MAX_DURATION``) are the frozen class's own code, reached through ``self`` -- so the two agents differ in
the pressure formula and in nothing else, and on a one-to-one intersection they are the same policy.  ``algorithms/`` is
untouched.  Only the acyclic controls are admitted: the frozen cyclic path maps a phase choice onto keep/switch, which no
registered anchor uses.

PROVENANCE
----------
:func:`roadlink_structure_digest` (sha256 over a canonical JSON of every intersection's id, ``roadlink_lanes``,
``phase_roadlink_mapping`` and ``phase_durations``) and :func:`phase_multiplicities` (per intersection and file phase:
links, incoming and outgoing visits, distinct lanes, per-lane maximum and minimum, on two bases -- the selectable greens,
and every file phase that carries a link, which is the basis A27(f)'s "160" and "160 of 256" were counted on) are
written into every cell this variant produces.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from algorithms.max_pressure import MaxPressureAgent

__all__ = [
    "ARM",
    "PROVENANCE_FORMAT_VERSION",
    "MaxPressure1to1Agent",
    "make_maxpressure_1to1",
    "maxpressure_1to1_factory",
    "phase_lane_visits",
    "phase_multiplicities",
    "roadlink_structure_digest",
    "set_pressure",
]

#: The NEW reference-arm name A27(a) requires: a new cell never wears a registered arm's name.
ARM = "maxpressure_1to1"

PROVENANCE_FORMAT_VERSION = "p9.2-maxpressure-1to1/1.0"

PolicyFn = Callable[[dict[str, Any]], np.ndarray]


def phase_lane_visits(ix: Any, active_roadlinks: Sequence[int]) -> tuple[Counter[str], Counter[str]]:
    """How many of the phase's links release each incoming lane and feed each outgoing lane.

    Read from ``ix.roadlink_lanes`` -- the SAME source the frozen agent's pressure reads -- one ``(in_lanes, out_lanes)``
    pair per link, every lane of every pair counted once per link.
    """
    incoming: Counter[str] = Counter()
    outgoing: Counter[str] = Counter()
    for link in active_roadlinks:
        in_lanes, out_lanes = ix.roadlink_lanes[link]
        incoming.update(str(lane) for lane in in_lanes)
        outgoing.update(str(lane) for lane in out_lanes)
    return incoming, outgoing


def set_pressure(ix: Any, active_roadlinks: Sequence[int], lane_counts: Mapping[str, int]) -> float:
    """A27(f)'s pressure: the DISTINCT incoming lanes' counts minus the DISTINCT outgoing lanes' counts.

    ``lane_counts[lane]`` is subscripted, never defaulted, exactly as the frozen agent subscripts it: a lane missing from
    the info raises ``KeyError`` on both.  Returned as a ``float`` like the frozen value, so the two agents compare the
    same type; the sums are of integer counts and are exact.
    """
    incoming, outgoing = phase_lane_visits(ix, active_roadlinks)
    upstream = sum(int(lane_counts[lane]) for lane in incoming)
    downstream = sum(int(lane_counts[lane]) for lane in outgoing)
    return float(upstream - downstream)


class MaxPressure1to1Agent(MaxPressureAgent):
    """The frozen MaxPressure agent with :func:`set_pressure` as its pressure -- and nothing else changed.

    ``act(info)`` is inherited: it calls ``self._acyclic_action``, which calls ``self._green_phases`` and
    ``self._phase_pressure``; the override below is the only method this class defines besides the constructor.
    """

    def __init__(self, env: Any) -> None:
        mode = str(getattr(env, "control_mode", "acyclic"))
        if mode != "acyclic":
            raise ValueError(
                f"maxpressure_1to1 is defined for the acyclic control only, got control_mode {mode!r}: every registered "
                "anchor runs acyclic, and the frozen cyclic path maps a phase choice onto keep/switch, so a cyclic run "
                "would not be the control A27(f) registers"
            )
        super().__init__(env)

    @staticmethod
    def _phase_pressure(ix: Any, active_roadlinks: list[int], lane_counts: dict[str, int]) -> float:
        return set_pressure(ix, active_roadlinks, lane_counts)


def make_maxpressure_1to1(env: Any, args: Any, rng: np.random.Generator) -> PolicyFn:
    """The collection-registry signature (``offline.collect.POLICIES``); ``args`` and ``rng`` are unused, as for MaxPressure."""
    del args, rng
    return MaxPressure1to1Agent(env).act


def maxpressure_1to1_factory(env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
    """The evaluation-route signature (``dt_gate.evaluate_arm`` / ``admission_probe.probe_episode``), as
    ``dt_gate._maxpressure_factory`` is for the frozen agent: one agent per env, ``act(info)`` per decision."""
    agent = MaxPressure1to1Agent(env)
    return lambda _env, info: agent.act(info)


def roadlink_structure_digest(intersections: Sequence[Any]) -> str:
    """sha256 over the roadlink structure the pressure is computed from, in ``env.intersections`` order.

    Canonical JSON (``sort_keys``, compact separators) of a list with one object per intersection: ``id``,
    ``roadlink_lanes`` (each link's incoming and outgoing lane lists, in the parser's order), ``phase_roadlink_mapping`` and
    ``phase_durations`` (which decide the selectable greens).  Two envs with the same digest give both formulas the same
    inputs.
    """
    payload = [
        {
            "id": str(ix.id),
            "roadlink_lanes": [[[str(lane) for lane in lanes_in], [str(lane) for lane in lanes_out]]
                               for lanes_in, lanes_out in ix.roadlink_lanes],
            "phase_roadlink_mapping": [[int(link) for link in links] for links in ix.phase_roadlink_mapping],
            "phase_durations": [float(value) for value in (getattr(ix, "phase_durations", None) or [])],
        }
        for ix in intersections
    ]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _phase_record(ix: Any, phase: int, greens: Sequence[int]) -> dict[str, Any]:
    links = list(ix.phase_roadlink_mapping[phase])
    incoming, outgoing = phase_lane_visits(ix, links)
    return {
        "file_phase": int(phase),
        "action": greens.index(phase) if phase in greens else None,
        "selectable": phase in greens,
        "n_links": len(links),
        "incoming_visits": int(sum(incoming.values())),
        "incoming_distinct": len(incoming),
        "incoming_max": max(incoming.values()) if incoming else None,
        "incoming_min": min(incoming.values()) if incoming else None,
        "outgoing_visits": int(sum(outgoing.values())),
        "outgoing_distinct": len(outgoing),
        "outgoing_max": max(outgoing.values()) if outgoing else None,
        "outgoing_min": min(outgoing.values()) if outgoing else None,
    }


def _basis_summary(ix_phases: Mapping[str, Sequence[Mapping[str, Any]]], intersections: Sequence[Any],
                   *, selectable_only: bool) -> dict[str, Any]:
    chosen = [
        (ix, phase)
        for ix in intersections
        for phase in ix_phases[str(ix.id)]
        if (phase["selectable"] if selectable_only else phase["n_links"] > 0)
    ]
    incoming_multiplicities: set[int] = set()
    outgoing_multiplicities: set[int] = set()
    for ix, phase in chosen:
        incoming, outgoing = phase_lane_visits(ix, list(ix.phase_roadlink_mapping[phase["file_phase"]]))
        incoming_multiplicities.update(incoming.values())
        outgoing_multiplicities.update(outgoing.values())
    return {
        "n_phases": len(chosen),
        "n_incoming_repeated": sum(1 for _ix, phase in chosen if (phase["incoming_max"] or 0) > 1),
        "n_outgoing_repeated": sum(1 for _ix, phase in chosen if (phase["outgoing_max"] or 0) > 1),
        "incoming_multiplicities": sorted(int(value) for value in incoming_multiplicities),
        "outgoing_multiplicities": sorted(int(value) for value in outgoing_multiplicities),
    }


def phase_multiplicities(intersections: Sequence[Any]) -> dict[str, Any]:
    """Per intersection and file phase, how often each lane is visited -- the measurement A27(f) and ``BRIEF_47`` require.

    Two summaries, because two bases are in use: ``selectable_greens`` (the phases MaxPressure chooses among, through the
    frozen ``_green_phases``) and ``file_phases_with_links`` (every file phase carrying at least one link -- the basis
    A27(f) counted "160 green phases" and "160 of the 256" on; it adds the transition phases the acyclic control never
    selects).  ``*_multiplicities`` lists every per-lane visit count observed on that basis; ``n_*_repeated`` counts the
    phases on which some lane is visited more than once.
    """
    ix_phases: dict[str, list[dict[str, Any]]] = {}
    n_file_phases = 0
    for ix in intersections:
        greens = list(MaxPressureAgent._green_phases(ix))
        ix_phases[str(ix.id)] = [_phase_record(ix, phase, greens) for phase in range(len(ix.phase_roadlink_mapping))]
        n_file_phases += len(ix.phase_roadlink_mapping)
    file_basis = _basis_summary(ix_phases, intersections, selectable_only=False)
    file_basis["n_file_phases"] = n_file_phases
    return {
        "format_version": PROVENANCE_FORMAT_VERSION,
        "intersections": ix_phases,
        "summary": {
            "selectable_greens": _basis_summary(ix_phases, intersections, selectable_only=True),
            "file_phases_with_links": file_basis,
        },
    }
