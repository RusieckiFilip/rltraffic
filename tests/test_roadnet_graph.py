"""The derived intersection graph, and the controls that prove it is the right one.

⚠️ **A graph attention layer over a WRONG adjacency still trains and still produces plausible
numbers.**  That is the failure this file exists to prevent, and it is invisible without these
controls.  Four wrong derivations are computed here and asserted to disagree with the shipped one,
so each control stays in the suite permanently rather than being demonstrated once in a packet.

The double-compute (``test_the_edge_set_reproduces_from_the_roads_array``) recomputes the edge set
from the roadnet's ``roads`` array, using ``startIntersection``/``endIntersection`` and touching
neither ``lanes`` nor ``roadLinks``.  It is written **here** rather than imported from the module
under test, following ``offline/dataset.py::_returns_to_go``'s precedent, and it is compared with
``==`` on frozensets because both routes produce exact sets and nothing rounds.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from offline.roadnet_graph import (
    GRAPH_FORMAT_VERSION,
    AdjacencySpec,
    adjacency_from_roadnet_file,
    adjacency_from_sim_config,
    assert_reproduces_from_roads,
    derive_adjacency,
    roadnet_path_from_sim_config,
)
from utils.cityflow_utils import parse_roadnet

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID4X4_ROADNET = REPO_ROOT / "scenarios" / "grid4x4" / "grid4x4_roadnet_red.json"
GRID4X4_SIM_CONFIG = REPO_ROOT / "configs" / "sim" / "cityflow_grid4x4.json"

#: The known 4x4 pattern (BRIEF_22 section 2): 4 corners at 2, 8 edges at 3, 4 interior at 4.
EXPECTED_DEGREE_HISTOGRAM = {2: 4, 3: 8, 4: 4}
EXPECTED_UNDIRECTED_EDGES = 24
EXPECTED_DIRECTED_EDGES = 48
EXPECTED_NODES = 16


# ----------------------------------------------------------------------
# Fixtures: the node order comes from the data, never from the roadnet
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid_node_ids() -> tuple[str, ...]:
    """The controllable order, as the corpus and the env both report it."""
    return tuple(parse_roadnet(GRID4X4_ROADNET).intersection_ids)


@pytest.fixture(scope="module")
def grid_spec(grid_node_ids: tuple[str, ...]) -> AdjacencySpec:
    return adjacency_from_roadnet_file(GRID4X4_ROADNET, grid_node_ids)


def _roads_route(roadnet_path: Path, node_ids) -> frozenset[tuple[str, str]]:
    """INDEPENDENT ROUTE: edges from road endpoints; never reads lanes or roadLinks."""
    payload = json.loads(Path(roadnet_path).read_text(encoding="utf-8"))
    keep = set(node_ids)
    return frozenset(
        (road["startIntersection"], road["endIntersection"])
        for road in payload["roads"]
        if road["startIntersection"] in keep
        and road["endIntersection"] in keep
        and road["startIntersection"] != road["endIntersection"]
    )


# ----------------------------------------------------------------------
# The positive control: the known 4x4 pattern
# ----------------------------------------------------------------------


def test_grid4x4_exposes_exactly_sixteen_controllable_intersections(grid_node_ids):
    assert len(grid_node_ids) == EXPECTED_NODES
    assert len(set(grid_node_ids)) == EXPECTED_NODES
    assert grid_node_ids[0] == "A0"
    assert grid_node_ids[-1] == "D3"


def test_grid4x4_has_twenty_four_undirected_edges(grid_spec):
    assert len(grid_spec.undirected_edges()) == EXPECTED_UNDIRECTED_EDGES


def test_grid4x4_has_forty_eight_directed_edges(grid_spec):
    assert len(grid_spec.directed_edges()) == EXPECTED_DIRECTED_EDGES


def test_grid4x4_degree_histogram_is_the_known_two_three_four_pattern(grid_spec):
    assert grid_spec.degree_histogram() == EXPECTED_DEGREE_HISTOGRAM


def test_the_four_corners_are_the_four_degree_two_nodes(grid_spec):
    degrees = grid_spec.degrees()
    corners = sorted(node for node, degree in degrees.items() if degree == 2)
    assert corners == ["A0", "A3", "D0", "D3"]


def test_the_directed_relation_is_symmetric_on_grid4x4(grid_spec):
    # Recorded rather than relied on: it makes undirected-versus-directed moot HERE only.
    assert grid_spec.is_symmetric
    assert np.array_equal(grid_spec.directed, grid_spec.directed.T)


def test_no_node_feeds_itself(grid_spec):
    assert not grid_spec.directed.diagonal().any()
    assert not grid_spec.undirected.diagonal().any()


# ----------------------------------------------------------------------
# DOUBLE COMPUTE: a second route that shares no arithmetic with the first
# ----------------------------------------------------------------------


def test_the_edge_set_reproduces_from_the_roads_array(grid_spec, grid_node_ids):
    """Lane route == road route, exactly.  Not ``allclose``: both are sets."""
    assert grid_spec.directed_edges() == _roads_route(GRID4X4_ROADNET, grid_node_ids)


def test_the_roads_route_would_catch_a_single_missing_edge(grid_spec, grid_node_ids):
    """Discriminating power: the check must fail on a graph that is wrong by ONE edge."""
    corrupted = grid_spec.directed.copy()
    row, col = np.argwhere(corrupted)[0]
    corrupted[row, col] = False
    broken = AdjacencySpec(
        node_ids=grid_spec.node_ids,
        directed=corrupted,
        undirected=corrupted | corrupted.T,
        roadnet_path=grid_spec.roadnet_path,
        roadnet_sha256=grid_spec.roadnet_sha256,
    )
    assert broken.directed_edges() != _roads_route(GRID4X4_ROADNET, grid_node_ids)
    with pytest.raises(ValueError, match="roads route disagrees"):
        assert_reproduces_from_roads(broken)


def test_the_runtime_guard_accepts_the_shipped_graph(grid_spec):
    report = assert_reproduces_from_roads(grid_spec)
    assert report["directed_edges"] == EXPECTED_DIRECTED_EDGES
    assert report["route"] == "roads.startIntersection/endIntersection"


# ----------------------------------------------------------------------
# WRONG DERIVATION CONTROLS -- each computed here and asserted to disagree
# ----------------------------------------------------------------------


def test_control_one_incoming_intersect_incoming_finds_no_edges_at_all(grid_node_ids):
    """BRIEF_22 section 2's named wrong derivation.  It yields an EMPTY graph."""
    roadnet = parse_roadnet(GRID4X4_ROADNET)
    incoming = {ix.id: set(ix.incoming_lanes) for ix in roadnet.intersections}
    wrong = {
        (a, b)
        for a in grid_node_ids
        for b in grid_node_ids
        if a != b and incoming[a] & incoming[b]
    }
    assert len(wrong) == 0
    assert len(wrong) != EXPECTED_DIRECTED_EDGES


def test_control_two_deriving_over_all_thirty_two_non_virtual_nodes_is_the_uniform_wrong_graph():
    """🚨 The trap: ``virtual`` is False on ALL 32 nodes; ``gt_virtual`` is the discriminator.

    The wrong graph is not obviously broken -- it is UNIFORM, every real intersection reading
    degree 4, corners indistinguishable from interior nodes.
    """
    payload = json.loads(GRID4X4_ROADNET.read_text(encoding="utf-8"))
    raw = payload["intersections"]
    assert len(raw) == 32
    assert sum(1 for ix in raw if ix.get("virtual", False)) == 0, (
        "the trap depends on virtual being False everywhere; if this ever changes the control "
        "below stops measuring what it was written for"
    )
    assert sum(1 for ix in raw if ix.get("gt_virtual", False)) == 16

    trap_nodes = [ix["id"] for ix in raw if not ix.get("virtual", False)]
    assert len(trap_nodes) == 32

    trap_edges = _roads_route(GRID4X4_ROADNET, trap_nodes)
    trap_undirected = {frozenset(edge) for edge in trap_edges}
    degrees: Counter[str] = Counter()
    for edge in trap_undirected:
        for node in edge:
            degrees[node] += 1

    assert len(trap_edges) == 80
    assert len(trap_undirected) == 40
    assert dict(sorted(Counter(degrees.values()).items())) == {1: 16, 4: 16}
    # The point of the control: it disagrees with the shipped graph on every headline number.
    assert len(trap_undirected) != EXPECTED_UNDIRECTED_EDGES
    assert dict(Counter(degrees.values())) != EXPECTED_DEGREE_HISTOGRAM


def test_control_three_index_adjacency_over_the_id_list_is_not_the_road_network(grid_node_ids):
    """A plausible-looking grid built from list positions, which pairs A3 with B0."""
    side = 4
    wrong = set()
    for index, node in enumerate(grid_node_ids):
        for offset in (1, side):
            other = index + offset
            if other < len(grid_node_ids):
                wrong.add((node, grid_node_ids[other]))
                wrong.add((grid_node_ids[other], node))
    truth = _roads_route(GRID4X4_ROADNET, grid_node_ids)
    assert wrong != truth
    assert ("A3", "B0") in wrong
    assert ("A3", "B0") not in truth


def test_control_four_a_permuted_node_order_is_a_different_matrix(grid_node_ids):
    """The P7.0 pairing-key failure: same edges, wrong rows.

    ``PROJECT_PLAN`` section 7 (2026-08-16): a registered criterion can silently measure something
    other than what it was named for when the key pairing two systems is assumed.  A permuted
    adjacency matrix masks the wrong tensor rows and trains perfectly happily.

    ⚠️ **The permutation here swaps a degree-2 corner with a degree-4 interior node, and that
    choice is load-bearing.**  The obvious permutation -- reversing the id list -- is a graph
    AUTOMORPHISM of this grid and leaves the matrix bit-identical; see
    ``test_a_grid_automorphism_leaves_the_matrix_identical...`` below, which is why the node ids
    must travel with the matrix instead of being inferred from it.
    """
    straight = adjacency_from_roadnet_file(GRID4X4_ROADNET, grid_node_ids)
    assert straight.degrees()["A0"] == 2 and straight.degrees()["B1"] == 4

    swapped = list(grid_node_ids)
    left, right = swapped.index("A0"), swapped.index("B1")
    swapped[left], swapped[right] = swapped[right], swapped[left]
    permuted = adjacency_from_roadnet_file(GRID4X4_ROADNET, tuple(swapped))

    # Same graph as a SET of edges ...
    assert straight.directed_edges() == permuted.directed_edges()
    # ... and a different matrix, which is the thing the model actually consumes.
    assert not np.array_equal(straight.directed, permuted.directed)
    assert straight.node_ids != permuted.node_ids
    # Row 0 now carries B1's degree instead of A0's -- the corruption a silent permutation causes.
    assert int(straight.undirected[0].sum()) == 2
    assert int(permuted.undirected[0].sum()) == 4


def test_a_grid_automorphism_leaves_the_matrix_identical_so_the_order_must_be_carried(
    grid_node_ids,
):
    """🚨 Found while writing this file: the adjacency matrix CANNOT reveal every permutation.

    Reversing the controllable order maps ``A0 <-> D3``, ``A1 <-> D2``, ... -- a point reflection
    through the centre of the grid -- and the 4x4 grid is invariant under it.  The relabelled edge
    set is **equal** to the original and the matrix is **bit-identical**, so a reader handed only
    the matrix cannot tell the two orders apart.

    **This strengthens the pairing-key rule rather than weakening it:** ``node_ids`` must travel
    with the matrix and be checked against the data, because for this network there is provably no
    way to recover the order from the matrix alone.
    """
    straight = adjacency_from_roadnet_file(GRID4X4_ROADNET, grid_node_ids)
    reversed_ids = tuple(reversed(grid_node_ids))
    mapping = dict(zip(grid_node_ids, reversed_ids))

    image = frozenset(
        (mapping[a], mapping[b]) for a, b in straight.directed_edges()
    )
    assert image == straight.directed_edges(), "the reversal is not an automorphism after all"

    reflected = adjacency_from_roadnet_file(GRID4X4_ROADNET, reversed_ids)
    assert np.array_equal(straight.directed, reflected.directed)
    # The ONLY thing that distinguishes them is the carried order.
    assert straight.node_ids != reflected.node_ids


# ----------------------------------------------------------------------
# The refusals that make the pairing key a proof rather than a hope
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# A ONE-WAY network: on grid4x4 the directed relation is already symmetric, so
# "undirected = directed | directed.T" is an EQUIVALENT MUTANT there (measured -- it
# survives all 27 grid4x4 tests).  Same family as DEFERRED 37: equivalent on today's
# data, live on tomorrow's.  This fixture makes it live now.
# ----------------------------------------------------------------------


def _one_way_roadnet(path: Path) -> tuple[str, ...]:
    """Write a minimal roadnet with a single one-way road ``X -> Y``; return the node order.

    ``W`` and ``Z`` are boundary nodes carrying ``gt_virtual`` so the frozen parser skips them,
    exactly as grid4x4's 16 boundary nodes are skipped.  ``X`` feeds ``Y`` and nothing feeds ``X``
    from ``Y``, so the directed relation is asymmetric by construction.
    """

    def road(rid: str, start: str, end: str) -> dict:
        return {
            "id": rid,
            "startIntersection": start,
            "endIntersection": end,
            "points": [{"x": 0.0, "y": 0.0}, {"x": 100.0, "y": 0.0}],
            "lanes": [{"width": 4.0, "maxSpeed": 11.11}],
        }

    def controllable(iid: str, start_road: str, end_road: str) -> dict:
        return {
            "id": iid,
            "point": {"x": 0.0, "y": 0.0},
            "width": 10.0,
            "roads": [start_road, end_road],
            "roadLinks": [
                {
                    "startRoad": start_road,
                    "endRoad": end_road,
                    "laneLinks": [{"startLaneIndex": 0, "endLaneIndex": 0}],
                }
            ],
            "trafficLight": {"roadLinkIndices": [0], "lightphases": [{"time": 30.0,
                                                                     "availableRoadLinks": [0]}]},
            "virtual": False,
            "gt_virtual": False,
        }

    def boundary(iid: str) -> dict:
        return {"id": iid, "point": {"x": 0.0, "y": 0.0}, "width": 0.0, "roads": [],
                "roadLinks": [], "trafficLight": {"roadLinkIndices": [], "lightphases": []},
                "virtual": False, "gt_virtual": True}

    payload = {
        "intersections": [
            controllable("X", "WX", "XY"),
            controllable("Y", "XY", "YZ"),
            boundary("W"),
            boundary("Z"),
        ],
        "roads": [road("WX", "W", "X"), road("XY", "X", "Y"), road("YZ", "Y", "Z")],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return ("X", "Y")


def test_on_a_one_way_network_the_directed_relation_is_asymmetric(tmp_path):
    roadnet = tmp_path / "one_way_roadnet.json"
    nodes = _one_way_roadnet(roadnet)
    spec = adjacency_from_roadnet_file(roadnet, nodes)

    assert spec.directed_edges() == frozenset({("X", "Y")})
    assert not spec.is_symmetric
    assert not np.array_equal(spec.directed, spec.directed.T)


def test_the_undirected_relation_really_is_the_symmetrisation(tmp_path):
    """Kills the ``undirected = directed`` mutant, which grid4x4 alone cannot kill."""
    roadnet = tmp_path / "one_way_roadnet.json"
    nodes = _one_way_roadnet(roadnet)
    spec = adjacency_from_roadnet_file(roadnet, nodes)

    assert np.array_equal(spec.undirected, spec.directed | spec.directed.T)
    assert not np.array_equal(spec.undirected, spec.directed)
    assert spec.undirected_edges() == frozenset({frozenset({"X", "Y"})})
    # Both nodes see each other once the relation is symmetrised; only X sees Y before it.
    assert spec.degrees() == {"X": 1, "Y": 1}
    assert int(spec.directed[1].sum()) == 0


def test_the_mixing_mask_is_symmetric_even_on_a_one_way_network(tmp_path):
    """Coordination information flows both ways; the mask must not inherit the road direction."""
    roadnet = tmp_path / "one_way_roadnet.json"
    nodes = _one_way_roadnet(roadnet)
    mask = adjacency_from_roadnet_file(roadnet, nodes).attention_mask(spatial_mixing=True)

    assert np.array_equal(mask, mask.T)
    assert np.array_equal(mask, np.ones((2, 2), dtype=bool))


def test_the_roads_route_agrees_on_a_one_way_network_too(tmp_path):
    roadnet = tmp_path / "one_way_roadnet.json"
    nodes = _one_way_roadnet(roadnet)
    spec = adjacency_from_roadnet_file(roadnet, nodes)

    assert spec.directed_edges() == _roads_route(roadnet, nodes)
    assert assert_reproduces_from_roads(spec)["directed_edges"] == 1


def test_a_node_set_that_is_not_the_controllable_set_is_refused():
    roadnet = parse_roadnet(GRID4X4_ROADNET)
    with pytest.raises(ValueError, match="not the roadnet's controllable set"):
        derive_adjacency(
            roadnet, ["A0", "A1", "bottom0"], roadnet_path=GRID4X4_ROADNET
        )


def test_a_missing_node_is_refused_and_both_differences_are_named(grid_node_ids):
    roadnet = parse_roadnet(GRID4X4_ROADNET)
    with pytest.raises(ValueError, match="not the roadnet's controllable set") as excinfo:
        derive_adjacency(roadnet, grid_node_ids[:-1], roadnet_path=GRID4X4_ROADNET)
    message = str(excinfo.value)
    assert "D3" in message


def test_duplicate_node_ids_are_refused(grid_node_ids):
    roadnet = parse_roadnet(GRID4X4_ROADNET)
    with pytest.raises(ValueError, match="duplicate node ids"):
        derive_adjacency(
            roadnet, list(grid_node_ids) + ["A0"], roadnet_path=GRID4X4_ROADNET
        )


def test_an_empty_node_list_is_refused():
    roadnet = parse_roadnet(GRID4X4_ROADNET)
    with pytest.raises(ValueError, match="node_ids is empty"):
        derive_adjacency(roadnet, [], roadnet_path=GRID4X4_ROADNET)


# ----------------------------------------------------------------------
# The attention mask, and the no-mixing control
# ----------------------------------------------------------------------


def test_the_mixing_mask_is_the_undirected_relation_plus_self_loops(grid_spec):
    mask = grid_spec.attention_mask(spatial_mixing=True)
    assert mask.dtype == np.bool_
    assert mask.shape == (EXPECTED_NODES, EXPECTED_NODES)
    assert np.array_equal(mask, grid_spec.undirected | np.eye(EXPECTED_NODES, dtype=bool))
    # 24 undirected edges seen from both ends, plus 16 self-loops.
    assert int(mask.sum()) == 2 * EXPECTED_UNDIRECTED_EDGES + EXPECTED_NODES


def test_the_no_mixing_mask_is_exactly_the_identity(grid_spec):
    """The registered control: information removed, capacity untouched."""
    mask = grid_spec.attention_mask(spatial_mixing=False)
    assert np.array_equal(mask, np.eye(EXPECTED_NODES, dtype=bool))
    assert int(mask.sum()) == EXPECTED_NODES


def test_the_two_masks_differ_so_the_control_is_not_the_treatment(grid_spec):
    assert not np.array_equal(
        grid_spec.attention_mask(spatial_mixing=True),
        grid_spec.attention_mask(spatial_mixing=False),
    )


def test_no_query_row_is_ever_fully_masked_in_either_mode(grid_spec):
    for mixing in (True, False):
        mask = grid_spec.attention_mask(spatial_mixing=mixing)
        assert bool(mask.any(axis=1).all()), f"a fully masked query row at spatial_mixing={mixing}"


# ----------------------------------------------------------------------
# Provenance: the graph must come from the network the corpus was collected on
# ----------------------------------------------------------------------


def test_the_sim_config_resolves_to_the_grid4x4_roadnet():
    assert roadnet_path_from_sim_config(GRID4X4_SIM_CONFIG) == GRID4X4_ROADNET.resolve()


def test_adjacency_from_the_sim_config_matches_the_direct_route(grid_spec, grid_node_ids):
    from_config = adjacency_from_sim_config(GRID4X4_SIM_CONFIG, grid_node_ids)
    assert np.array_equal(from_config.directed, grid_spec.directed)
    assert from_config.node_ids == grid_spec.node_ids
    assert from_config.roadnet_sha256 == grid_spec.roadnet_sha256


def test_the_spec_records_the_rule_and_the_network_it_came_from(grid_spec):
    payload = grid_spec.to_json_obj()
    assert payload["format_version"] == GRAPH_FORMAT_VERSION
    assert payload["n_nodes"] == EXPECTED_NODES
    assert payload["undirected_edges"] == EXPECTED_UNDIRECTED_EDGES
    assert payload["degree_histogram"] == {str(k): v for k, v in EXPECTED_DEGREE_HISTOGRAM.items()}
    assert "outgoing_lanes" in payload["rule"] and "incoming_lanes" in payload["rule"]
    assert len(payload["roadnet_sha256"]) == 64
    assert payload["node_ids"] == list(grid_spec.node_ids)


def test_the_recorded_digest_is_the_file_on_disk(grid_spec):
    import hashlib

    expected = hashlib.sha256(GRID4X4_ROADNET.read_bytes()).hexdigest()
    assert grid_spec.roadnet_sha256 == expected
