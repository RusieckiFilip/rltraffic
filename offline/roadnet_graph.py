"""Derive intersection adjacency from a CityFlow road network, and prove it before using it.

Graph format version: ``roadnet-graph/1.0``.

WHY THIS MODULE EXISTS AND WHY IT IS NOT IN ``utils/``
------------------------------------------------------
``PROJECT_PLAN`` section 6 describes P5.1 as graph attention *"over road-network adjacency from
``RoadnetInfo``"*.  **That premise is false and was checked before this file was written:**
``utils/common_utils.py`` exposes ``intersections``, ``lane_ids``, ``road_ids``,
``intersection_ids``, ``road_lengths`` and ``road_max_speeds``, and per intersection
``incoming_lanes``, ``outgoing_lanes``, ``num_phases``, ``phase_roadlink_mapping``,
``phase_durations``, ``phase_states`` and ``roadlink_lanes`` -- **no adjacency, neighbour or graph
field of any kind.**  ``utils/`` is frozen, so the field cannot be added there.  Adjacency is
therefore DERIVED here, in ``offline/``, and never assumed.

THE RULE
--------
::

    intersection A feeds intersection B  iff  A.outgoing_lanes & B.incoming_lanes != {}

THE PAIRING KEY IS PROVED, NEVER ASSUMED
-----------------------------------------
``PROJECT_PLAN`` section 7, rule of 2026-08-16 (earned by P7.0, where an unproved pairing key voided
a registered criterion): *any cross-system comparison must PROVE its pairing key before using it,
and the proof must be derivable from structure alone.*  Here the two systems are **the corpus rows**
and **the graph nodes**.  :func:`derive_adjacency` therefore takes the node order from its caller --
who reads it from the data (``episode ix_ids`` / ``env.intersections``) -- and refuses unless that
set is exactly the roadnet's controllable set.  A graph whose rows are permuted relative to the
tensor it masks still trains and still produces plausible numbers; that is the failure this
refusal exists to prevent.

🚨 THE TRAP, MEASURED, NOT HYPOTHESISED
----------------------------------------
``scenarios/grid4x4/grid4x4_roadnet_red.json`` holds **32** intersections and **``virtual`` is
``False`` on all 32**; the discriminator is **``gt_virtual``**, true on 16.  Selecting nodes by
"not virtual" therefore takes all 32 and yields **80 directed / 40 undirected** edges with degree
histogram ``{1: 16, 4: 16}`` -- in which **every real intersection reads degree 4 and a corner is
indistinguishable from an interior node.**  The wrong graph is *uniform*, not obviously broken.
Restricted to the 16 controllable intersections the same file yields **48 directed / 24 undirected**
and ``{2: 4, 3: 8, 4: 4}``, which is the known 4x4 pattern.  Both figures were measured on
2026-08-17.  This module never selects nodes itself: the caller supplies them and the set is checked.

DIRECTED VERSUS UNDIRECTED
--------------------------
The derived relation is directed by construction (A feeds B).  :attr:`AdjacencySpec.undirected` is
its symmetrisation, and :meth:`AdjacencySpec.attention_mask` uses **the undirected relation plus
self-loops**, because coordination information flows both upstream and downstream.
⚠️ **On grid4x4 the choice is empirically moot and that is recorded rather than relied on:** the
directed relation there is already symmetric (measured -- every road has a counterpart in the
opposite direction), so undirected and directed give the same mask.  On a network with one-way
streets they would differ, and a later scenario must revisit this rather than inherit it.

WHAT IS DELIBERATELY NOT HERE
------------------------------
The second, independent derivation route -- edges read from the roadnet's ``roads`` array via
``startIntersection``/``endIntersection``, touching neither lanes nor ``roadLinks`` -- is
implemented **in the test**, following the precedent of ``offline/dataset.py::_returns_to_go``
(explicit loop in production, ``np.cumsum`` in the test): a check that shares its implementation
with the thing it checks is not a check.  :func:`assert_reproduces_from_roads` is a *runtime guard*
that runs the roads route before a campaign trains on the graph; the test's own third route is what
proves that guard is not vacuous.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from utils.common_utils import RoadnetInfo

__all__ = [
    "GRAPH_FORMAT_VERSION",
    "AdjacencySpec",
    "adjacency_from_roadnet_file",
    "adjacency_from_sim_config",
    "assert_reproduces_from_roads",
    "derive_adjacency",
    "roadnet_path_from_sim_config",
]

GRAPH_FORMAT_VERSION = "roadnet-graph/1.0"


@dataclass(frozen=True)
class AdjacencySpec:
    """A derived intersection graph, keyed by an explicit node order.

    ``node_ids`` is the order the caller supplied and the order every array here is indexed by.
    ``directed[i, j]`` is True iff node ``i`` feeds node ``j``; ``undirected`` is its
    symmetrisation.  Neither carries self-loops -- those are added by
    :meth:`attention_mask`, so that "is i adjacent to j" and "may i attend to j" stay distinct
    questions with distinct answers.
    """

    node_ids: tuple[str, ...]
    directed: np.ndarray
    undirected: np.ndarray
    roadnet_path: str
    roadnet_sha256: str

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self.node_ids)

    @property
    def is_symmetric(self) -> bool:
        """Whether the DIRECTED relation is already symmetric (true on grid4x4, measured)."""
        return bool(np.array_equal(self.directed, self.directed.T))

    def degrees(self) -> dict[str, int]:
        """Undirected degree per node id."""
        counts = self.undirected.sum(axis=1)
        return {node: int(counts[index]) for index, node in enumerate(self.node_ids)}

    def degree_histogram(self) -> dict[int, int]:
        """``degree -> how many nodes have it``.  grid4x4 must give ``{2: 4, 3: 8, 4: 4}``."""
        histogram: dict[int, int] = {}
        for degree in self.degrees().values():
            histogram[degree] = histogram.get(degree, 0) + 1
        return dict(sorted(histogram.items()))

    def directed_edges(self) -> frozenset[tuple[str, str]]:
        """``(feeder, fed)`` id pairs."""
        return frozenset(
            (self.node_ids[i], self.node_ids[j])
            for i, j in zip(*np.nonzero(self.directed))
        )

    def undirected_edges(self) -> frozenset[frozenset[str]]:
        """Unordered id pairs.  grid4x4 must give 24 of them."""
        return frozenset(
            frozenset((self.node_ids[i], self.node_ids[j]))
            for i, j in zip(*np.nonzero(self.undirected))
        )

    def attention_mask(self, *, spatial_mixing: bool = True) -> np.ndarray:
        """``(N, N)`` bool: True where a query node may attend to a key node.

        ``spatial_mixing=True`` gives the undirected relation with the diagonal forced open.
        ``spatial_mixing=False`` gives the **identity** -- the registered no-mixing control, which
        keeps parameter count and compute identical and removes only the information flow.

        The diagonal is open in **both** modes, so no query row is ever fully masked and the
        no-mixing arm is a well-posed attention rather than a degenerate one.
        """
        identity = np.eye(self.n_nodes, dtype=np.bool_)
        if not spatial_mixing:
            return identity
        return np.asarray(self.undirected, dtype=np.bool_) | identity

    def to_json_obj(self) -> dict[str, Any]:
        """JSON-ready record, including the derivation rule and the provenance of the network."""
        return {
            "format_version": GRAPH_FORMAT_VERSION,
            "rule": (
                "intersection A feeds intersection B iff "
                "A.outgoing_lanes & B.incoming_lanes is non-empty"
            ),
            "mask_rule": (
                "attention uses the UNDIRECTED relation plus self-loops; the no-mixing control "
                "replaces it with the identity, leaving parameters and compute unchanged"
            ),
            "n_nodes": self.n_nodes,
            "node_ids": list(self.node_ids),
            "directed_edges": len(self.directed_edges()),
            "undirected_edges": len(self.undirected_edges()),
            "degree_histogram": {str(k): v for k, v in self.degree_histogram().items()},
            "degrees": self.degrees(),
            "directed_relation_is_symmetric": self.is_symmetric,
            "roadnet_path": self.roadnet_path,
            "roadnet_sha256": self.roadnet_sha256,
        }


def derive_adjacency(
    roadnet: RoadnetInfo,
    node_ids: Sequence[str],
    *,
    roadnet_path: str | Path,
) -> AdjacencySpec:
    """Derive the graph over *node_ids*, refusing any set that is not the controllable set.

    *node_ids* comes from the data the model will be fed -- the corpus's ``ix_ids`` or
    ``[ix.id for ix in env.intersections]`` -- and defines the row order of every array returned.
    Every check runs before anything is built, so a refusal constructs nothing.
    """
    order = [str(node) for node in node_ids]
    if not order:
        raise ValueError(
            "node_ids is empty: the node order must come from the data the model is fed "
            "(the corpus's ix_ids, or [ix.id for ix in env.intersections]), never from this module"
        )
    duplicates = sorted({node for node in order if order.count(node) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate node ids in the requested order: {duplicates}; the order indexes every "
            "row of the adjacency matrix and a repeat would silently alias two rows"
        )

    controllable = {ix.id: ix for ix in roadnet.intersections}
    requested, available = set(order), set(controllable)
    if requested != available:
        missing = sorted(available - requested)
        unknown = sorted(requested - available)
        raise ValueError(
            f"the requested node set is not the roadnet's controllable set: "
            f"{len(unknown)} unknown id(s) {unknown[:8]} and {len(missing)} missing id(s) "
            f"{missing[:8]}. The roadnet exposes {len(available)} controllable intersections. "
            "⚠️ A roadnet may list non-controllable nodes with virtual=False -- grid4x4 lists 32 "
            "of which 16 are gt_virtual -- and deriving over those gives a uniform, wrong graph "
            "that still trains. The node order must come from the data, and it is checked here"
        )

    outgoing = {node: set(controllable[node].outgoing_lanes) for node in order}
    incoming = {node: set(controllable[node].incoming_lanes) for node in order}

    size = len(order)
    directed = np.zeros((size, size), dtype=np.bool_)
    for i, feeder in enumerate(order):
        for j, fed in enumerate(order):
            if i != j and outgoing[feeder] & incoming[fed]:
                directed[i, j] = True
    undirected = directed | directed.T

    path = Path(roadnet_path).resolve()
    return AdjacencySpec(
        node_ids=tuple(order),
        directed=directed,
        undirected=undirected,
        roadnet_path=str(path),
        roadnet_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def roadnet_path_from_sim_config(config_path: str | Path) -> Path:
    """Resolve ``dir`` + ``roadnetFile`` out of a CityFlow sim config into an absolute path.

    Mirrors ``envs/cityflow_env.py:82-88`` -- a relative ``dir`` resolves against the **current
    working directory** -- so the graph is derived from the same file the env loads.  Because that
    makes the result cwd-dependent, a path that does not exist falls back to the repository root
    and, if neither exists, **raises naming both candidates** rather than resolving to something
    plausible.  ``PROJECT_PLAN`` section 7's rule about relative paths applies here directly.
    """
    config = Path(config_path).resolve()
    payload = json.loads(config.read_text(encoding="utf-8"))
    directory = str(payload.get("dir", ""))
    roadnet_file = str(payload.get("roadnetFile", ""))
    if not roadnet_file:
        raise ValueError(f"{config}: the sim config carries no 'roadnetFile'")

    if Path(directory).is_absolute():
        candidate = Path(directory) / roadnet_file
        if not candidate.is_file():
            raise FileNotFoundError(f"{config}: roadnet not found at {candidate}")
        return candidate.resolve()

    from_cwd = (Path.cwd() / directory / roadnet_file).resolve()
    if from_cwd.is_file():
        return from_cwd
    from_repo = (Path(__file__).resolve().parents[1] / directory / roadnet_file).resolve()
    if from_repo.is_file():
        return from_repo
    raise FileNotFoundError(
        f"{config}: roadnet {roadnet_file!r} under relative dir {directory!r} was not found "
        f"from the working directory ({from_cwd}) or from the repository root ({from_repo})"
    )


def adjacency_from_roadnet_file(
    roadnet_path: str | Path, node_ids: Sequence[str]
) -> AdjacencySpec:
    """Parse a roadnet file through the frozen parser and derive the graph over *node_ids*."""
    from utils.cityflow_utils import parse_roadnet

    path = Path(roadnet_path).resolve()
    return derive_adjacency(parse_roadnet(path), node_ids, roadnet_path=path)


def adjacency_from_sim_config(
    config_path: str | Path, node_ids: Sequence[str]
) -> AdjacencySpec:
    """The graph of the network a sim config points at -- the corpus's own network."""
    return adjacency_from_roadnet_file(roadnet_path_from_sim_config(config_path), node_ids)


def assert_reproduces_from_roads(spec: AdjacencySpec) -> dict[str, Any]:
    """Refuse the graph unless a second, lane-free route reproduces its edge set exactly.

    The second route reads the roadnet's ``roads`` array and keeps the pairs whose
    ``startIntersection`` and ``endIntersection`` are both in ``spec.node_ids``.  It touches
    neither ``lanes`` nor ``roadLinks``, so it shares no arithmetic with :func:`derive_adjacency`.
    Compared with ``==`` on frozensets, never with a tolerance -- both sides are exact sets.
    """
    payload = json.loads(Path(spec.roadnet_path).read_text(encoding="utf-8"))
    keep = set(spec.node_ids)
    from_roads = frozenset(
        (str(road["startIntersection"]), str(road["endIntersection"]))
        for road in payload.get("roads", [])
        if str(road.get("startIntersection")) in keep
        and str(road.get("endIntersection")) in keep
        and road.get("startIntersection") != road.get("endIntersection")
    )
    from_lanes = spec.directed_edges()
    if from_roads != from_lanes:
        only_lanes = sorted(from_lanes - from_roads)
        only_roads = sorted(from_roads - from_lanes)
        raise ValueError(
            "the roads route disagrees with the lane route on this network: "
            f"{len(only_lanes)} edge(s) only in the lane derivation {only_lanes[:8]} and "
            f"{len(only_roads)} only in the roads derivation {only_roads[:8]}. "
            "The two routes share no arithmetic, so a disagreement means one of them is wrong "
            "and neither may be used until it is resolved"
        )
    return {
        "route": "roads.startIntersection/endIntersection",
        "directed_edges": len(from_roads),
        "undirected_edges": len({frozenset(edge) for edge in from_roads}),
        "agrees_with_lane_route": True,
        "roadnet_sha256": spec.roadnet_sha256,
    }
