"""Tests for ``offline/conversion_audit.py`` -- CAP, criteria A-F.

The tool exists because five obvious implementations give plausible wrong numbers.  Each of the five
traps below is tested WITH the control that shows the naive route fails, because a checker that
agrees with the naive route on every input is not a checker.

``grid4x4``'s SUMO side lives in the gitignored, read-only candidates tree, so those tests skip with
a reason naming ``RLTRAFFIC_GRID4X4_RESCO``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from offline import conversion_audit as ca

REPO = Path(__file__).resolve().parents[1]
HZ1X1 = REPO / "scenarios" / "hangzhou_1x1_bc-tyc_18041610_1h"
HZ4X4 = REPO / "scenarios" / "hangzhou_4x4_gudang_18041610_1h"


@pytest.fixture(scope="module")
def resco() -> Path:
    """RESCO's grid4x4 directory, reached from the CANDIDATES ROOT the variable actually names.

    ⚠️ Two corrections from the half-B review (MINOR 1). The variable is the candidates ROOT, as its
    skip text always said, not the grid4x4 directory — the two disagreed, so the same variable meant
    different things to the reader and to the code. And there is **no `/home/filip` default**: with
    the variable unset these tests SKIP, instead of passing on one machine and skipping everywhere
    else while the packet reports them as run.
    """
    env_value = os.environ.get("RLTRAFFIC_GRID4X4_RESCO")
    if not env_value:
        pytest.skip(
            "RLTRAFFIC_GRID4X4_RESCO is unset: point it at the read-only, gitignored candidates "
            "root (the directory holding resco/ and libsignal/) to run the grid4x4 half of the "
            "audit. It is CC BY-NC-SA and is never copied into the tree."
        )
    candidate = Path(env_value) / "resco/resco_benchmark/environments/grid4x4"
    if not (candidate / "grid4x4.net.xml").is_file():
        pytest.skip(f"RESCO's grid4x4 net is not at {candidate}")
    return candidate


def _roadlinks(roadnet: Path, ix_id: str) -> list[tuple[str, str]]:
    data = json.loads(roadnet.read_text(encoding="utf-8"))
    entry = next(ix for ix in data["intersections"] if ix["id"] == ix_id)
    return [(link["startRoad"], link["endRoad"]) for link in entry["roadLinks"]]


# ----------------------------------------------------------------------
# Trap 1 -- gt_virtual
# ----------------------------------------------------------------------


def test_the_audit_counts_intersections_with_the_frozen_parsers_filter(resco: Path) -> None:
    """🔒 Trap 1: an s2c roadnet marks perimeter intersections ``gt_virtual``, not ``virtual``.

    grid4x4 has 32 entries and 0 of them carry ``virtual: true``; the platform controls 16. A tool
    reading only ``virtual`` reports 32 and is wrong about the agent count.
    """
    from utils.cityflow_utils import parse_roadnet

    roadnet = REPO / "scenarios" / "grid4x4" / "grid4x4_roadnet_red.json"
    data = json.loads(roadnet.read_text(encoding="utf-8"))

    naive = [ix for ix in data["intersections"] if not ix.get("virtual", False)]
    frozen = parse_roadnet(roadnet).intersections

    assert len(data["intersections"]) == 32
    assert len(naive) == 32  # the trap
    assert len(frozen) == 16  # the truth
    audit = ca.audit_pair(
        "grid4x4",
        cityflow_roadnet=roadnet,
        sumo_net=resco / "grid4x4.net.xml",
        direction="SUMO -> CityFlow",
    )
    assert audit["A_bijection"]["n_cityflow_intersections_non_virtual"] == 16
    assert audit["A_bijection"]["virtual_key_used"] == "gt_virtual"
    assert audit["A_bijection"]["n_intersection_entries_in_file"] == 32


def test_the_hangzhou_roadnets_use_the_other_key() -> None:
    """The control for trap 1: a c2s roadnet uses ``virtual`` and the same tool must still be right."""
    audit = ca.audit_pair(
        "hangzhou_4x4_gudang",
        cityflow_roadnet=HZ4X4 / "roadnet_4X4.json",
        sumo_net=HZ4X4 / "hangzhou_4x4_gudang_18041610_1h.net.xml",
        direction="CityFlow -> SUMO",
    )

    assert audit["A_bijection"]["virtual_key_used"] == "virtual"
    assert audit["A_bijection"]["n_cityflow_intersections_non_virtual"] == 16
    assert audit["A_bijection"]["n_intersection_entries_in_file"] == 32


# ----------------------------------------------------------------------
# Traps 2 and 3 -- unsorted phase lists, and the non-identity permutation
# ----------------------------------------------------------------------


def test_phase_lists_are_compared_as_sets_because_the_files_store_them_unsorted() -> None:
    """🔒 Trap 2: 16 of hz4x4's 144 stored ``availableRoadLinks`` are not in ascending order."""
    audit = ca.audit_pair(
        "hangzhou_4x4_gudang",
        cityflow_roadnet=HZ4X4 / "roadnet_4X4.json",
        sumo_net=HZ4X4 / "hangzhou_4x4_gudang_18041610_1h.net.xml",
        direction="CityFlow -> SUMO",
    )
    signals = audit["D_signals"]

    assert signals["n_cityflow_phase_lists_stored_unsorted"] == 16
    assert signals["n_phases_matching_as_sets"] == signals["n_phases_compared"] == 144


def test_the_link_index_permutation_is_derived_and_is_not_the_identity() -> None:
    """🔒 Trap 3, on the scenario where it bites: hz4x4's mapping is not the identity."""
    permutation = ca.link_index_permutation(
        HZ4X4 / "hangzhou_4x4_gudang_18041610_1h.net.xml",
        "intersection_1_1",
        _roadlinks(HZ4X4 / "roadnet_4X4.json", "intersection_1_1"),
    )

    assert list(permutation) == [10, 11, 9, 6, 7, 8, 3, 4, 5, 2, 0, 1]
    assert sorted(permutation) == list(range(12))


@pytest.mark.parametrize(
    "scenario, roadnet, net, matched, under_identity",
    [
        ("hz1x1", HZ1X1 / "roadnet.json", HZ1X1 / "hangzhou_1x1_bc-tyc_18041610_1h.net.xml", 9, 1),
        (
            "hz4x4",
            HZ4X4 / "roadnet_4X4.json",
            HZ4X4 / "hangzhou_4x4_gudang_18041610_1h.net.xml",
            144,
            0,
        ),
    ],
)
def test_the_signal_check_has_discriminating_power(
    scenario: str, roadnet: Path, net: Path, matched: int, under_identity: int
) -> None:
    """⭐ Trap 3's control, and the reason (D)'s 100 % means anything.

    Every phase matches under the derived permutation; under the identity almost none does. If both
    matched, the criterion would be satisfied by any mapping and would be measuring nothing.
    """
    audit = ca.audit_pair(
        scenario, cityflow_roadnet=roadnet, sumo_net=net, direction="CityFlow -> SUMO"
    )
    signals = audit["D_signals"]

    assert signals["n_phases_matching_as_sets"] == signals["n_phases_compared"] == matched
    assert signals["n_phases_matching_under_the_identity_permutation"] == under_identity


def test_a_connection_the_cityflow_side_does_not_declare_is_a_finding() -> None:
    """The permutation refuses rather than dropping a link it cannot place."""
    with pytest.raises(ValueError, match="does not declare"):
        ca.link_index_permutation(
            HZ4X4 / "hangzhou_4x4_gudang_18041610_1h.net.xml",
            "intersection_1_1",
            [("nonexistent_road", "another_one")],
        )


# ----------------------------------------------------------------------
# Trap 4 -- demand order
# ----------------------------------------------------------------------


def test_demand_is_compared_as_a_multiset_because_hz4x4s_two_files_disagree_on_order() -> None:
    """🔒 Trap 4: index-aligned, hz4x4 matches on 1 of 2,983; as a multiset, on 2,983 of 2,983.

    ``flow.json`` is grouped by origin and the ``.rou.xml`` is sorted by departure. The demand is
    preserved exactly; only the record order differs, and an index-aligned comparison would report a
    2,982-vehicle difference that does not exist.
    """
    audit = ca.audit_pair(
        "hangzhou_4x4_gudang",
        cityflow_roadnet=HZ4X4 / "roadnet_4X4.json",
        sumo_net=HZ4X4 / "hangzhou_4x4_gudang_18041610_1h.net.xml",
        cityflow_flow=HZ4X4 / "hangzhou_4x4_gudang_18041610_1h.json",
        sumo_routes=HZ4X4 / "hangzhou_4x4_gudang_18041610_1h.rou.xml",
        direction="CityFlow -> SUMO",
    )
    demand = audit["E_demand"]

    assert demand["n_cityflow"] == demand["n_sumo"] == 2983
    assert demand["multiset_equal"] is True
    assert demand["order_matches"] is False
    assert demand["n_index_aligned_equal"] == 1
    assert demand["n_only_in_cityflow"] == demand["n_only_in_sumo"] == 0


def test_hz1x1s_demand_is_index_aligned_and_the_same_tool_says_so() -> None:
    """The control for trap 4: where the order DOES match, the tool must not cry wolf."""
    audit = ca.audit_pair(
        "hangzhou_1x1_bc-tyc",
        cityflow_roadnet=HZ1X1 / "roadnet.json",
        sumo_net=HZ1X1 / "hangzhou_1x1_bc-tyc_18041610_1h.net.xml",
        cityflow_flow=HZ1X1 / "flow.json",
        sumo_routes=HZ1X1 / "hangzhou_1x1_bc-tyc_18041610_1h.rou.xml",
        direction="CityFlow -> SUMO",
    )
    demand = audit["E_demand"]

    assert demand["n_cityflow"] == demand["n_sumo"] == 2021
    assert demand["multiset_equal"] is True
    assert demand["order_matches"] is True
    assert demand["n_index_aligned_equal"] == 2021


def test_the_grid4x4_demand_is_rescos_seed_1_read_from_inside_the_zip(resco: Path) -> None:
    """Amendment C's provenance claim, re-derived: 1,473/1,473 against the zip member.

    ⚠️ Read with ``zipfile`` in place. The candidates tree is read-only and CC BY-NC-SA 4.0, and
    nothing is extracted into the repo.
    """
    audit = ca.audit_pair(
        "grid4x4",
        cityflow_roadnet=REPO / "scenarios" / "grid4x4" / "grid4x4_roadnet_red.json",
        sumo_net=resco / "grid4x4.net.xml",
        cityflow_flow=REPO / "scenarios" / "grid4x4" / "grid4x4_flow.json",
        sumo_routes=resco / "grid4x4.zip",
        sumo_routes_zip_member="grid4x4_1.rou.xml",
        direction="SUMO -> CityFlow",
    )
    demand = audit["E_demand"]

    assert demand["n_cityflow"] == demand["n_sumo"] == 1473
    assert demand["multiset_equal"] is True
    assert demand["n_index_aligned_equal"] == 1473
    assert not (resco / "grid4x4_1.rou.xml").exists()  # nothing was extracted


# ----------------------------------------------------------------------
# CAP(B) and CAP(C) -- the two halves the half-B review found uncomputed and untested
# ----------------------------------------------------------------------


def _audit(name: str, roadnet: Path, net: Path) -> dict[str, Any]:
    return ca.audit_pair(name, cityflow_roadnet=roadnet, sumo_net=net, direction="CityFlow -> SUMO")


@pytest.mark.parametrize(
    "name, roadnet, net, length_max, speed_max, speeds_equal",
    [
        (
            "hz1x1",
            HZ1X1 / "roadnet.json",
            HZ1X1 / "hangzhou_1x1_bc-tyc_18041610_1h.net.xml",
            10.4,
            0.0,
            True,
        ),
        (
            "hz4x4",
            HZ4X4 / "roadnet_4X4.json",
            HZ4X4 / "hangzhou_4x4_gudang_18041610_1h.net.xml",
            27.2,
            0.001,
            False,
        ),
    ],
)
def test_cap_b_enumerates_length_and_speed_per_lane_with_the_max_abs_difference(
    name: str, roadnet: Path, net: Path, length_max: float, speed_max: float, speeds_equal: bool
) -> None:
    """🔒 H2.1 / B-1. CAP(B) asks for per-lane length AND speed with the maximum absolute difference.

    The audit previously reported a SUMO-only length range and a SET comparison of speeds, so the
    length half did not exist and the speed half could not see hz4x4's 11.111 against 11.11 (one
    distinct value on each side). Both numbers are converter residuals and the table labels them so.

    ⚠️ These kill the reviewer's **M12** (speed sets -> count equality): a count comparison returns
    equal on hz4x4, where the truth is a 0.001 m/s difference on all 240 lanes.
    """
    b = _audit(name, roadnet, net)["B_geometry"]

    assert b["max_abs_length_difference"] == pytest.approx(length_max, abs=5e-4)
    assert b["max_abs_speed_difference"] == pytest.approx(speed_max, abs=1e-9)
    assert b["lane_speeds_equal"] is speeds_equal
    per_lane = b["per_lane"]
    assert per_lane["n_lanes_compared"] == len(per_lane["length"]["per_lane"])
    assert per_lane["n_lanes_only_in_cityflow"] == per_lane["n_lanes_only_in_sumo"] == 0
    # Every row carries both sides and their difference, so the table is enumerable, not a summary.
    row = per_lane["length"]["per_lane"][0]
    assert set(row) == {"lane", "cityflow", "sumo", "abs_difference"}
    assert row["abs_difference"] == abs(row["cityflow"] - row["sumo"])


def test_cap_b_coordinate_residual_uses_the_net_offset() -> None:
    """🔒 H2.1. Kills the reviewer's **M9** (netOffset ignored): the residual is 0.0 only WITH it.

    Without the offset the same comparison gives 300.0 on hz1x1 and 800.0 on gudang, which is
    exactly the recorded `netOffset` — a tool that dropped it would report a 300 m disagreement
    between two files that agree perfectly.
    """
    hz1 = _audit("hz1x1", HZ1X1 / "roadnet.json", HZ1X1 / "hangzhou_1x1_bc-tyc_18041610_1h.net.xml")
    hz4 = _audit(
        "hz4x4", HZ4X4 / "roadnet_4X4.json", HZ4X4 / "hangzhou_4x4_gudang_18041610_1h.net.xml"
    )

    assert hz1["B_geometry"]["max_abs_coordinate_residual"] == 0.0
    assert hz4["B_geometry"]["max_abs_coordinate_residual"] == 0.0
    assert hz1["B_geometry"]["net_offset"] == [300.0, 300.0]
    assert hz4["B_geometry"]["net_offset"] == [800.0, 600.0]
    assert hz1["B_geometry"]["n_junctions_compared"] == 5
    assert hz4["B_geometry"]["n_junctions_compared"] == 32


@pytest.mark.parametrize(
    "name, roadnet, net, n_links",
    [
        ("hz1x1", HZ1X1 / "roadnet.json", HZ1X1 / "hangzhou_1x1_bc-tyc_18041610_1h.net.xml", 16),
        (
            "hz4x4",
            HZ4X4 / "roadnet_4X4.json",
            HZ4X4 / "hangzhou_4x4_gudang_18041610_1h.net.xml",
            576,
        ),
    ],
)
def test_cap_c_compares_the_connection_SETS_and_not_only_their_counts(
    name: str, roadnet: Path, net: Path, n_links: int
) -> None:
    """🔒 H2.1 / B-1. CAP(C): the (in-lane → out-lane) set must EQUAL CityFlow's `laneLinks`.

    The audit previously reported direction counts, which cannot see a connection that moved. The
    equality holds on both pairs with zero missing, and the reversal rule the out-lanes are
    translated through is itself checked against the movement correspondence.
    """
    c = _audit(name, roadnet, net)["C_connections"]
    equality = c["lane_link_equality"]

    assert equality["n_cityflow_lane_links"] == n_links
    assert equality["n_equal"] == n_links
    assert equality["sets_equal"] is True
    assert equality["n_missing_in_sumo"] == 0
    assert equality["n_extra_in_sumo"] == 0
    assert equality["n_untranslatable"] == 0
    agreement = c["reversal_agrees_with_the_movement_correspondence"]
    assert agreement["n_disagree"] == 0
    assert agreement["n_agree"] == (8 if name == "hz1x1" else 192)


@pytest.mark.parametrize(
    "label, mutate, n_missing, n_extra",
    [
        # Redirecting a roadLink to another road creates pairs SUMO does not have AND orphans the
        # ones it does: both directions fire.
        ("redirected-road", lambda e: e["roadLinks"][0].__setitem__("endRoad", "road_1_1_2"), 2, 2),
        # Flipping an out-lane index COLLIDES with the sibling laneLink, so the CityFlow set shrinks
        # to 15 and SUMO carries one CityFlow does not. Missing is 0 here, and that is correct --
        # a set comparison reports what actually differs, not what a symmetric story would predict.
        (
            "collided-out-lane",
            lambda e: e["roadLinks"][0]["laneLinks"][0].__setitem__("endLaneIndex", 1),
            0,
            1,
        ),
        ("deleted-lane-link", lambda e: e["roadLinks"][0]["laneLinks"].pop(0), 0, 1),
    ],
)
def test_cap_c_detects_a_connection_that_moved(
    label: str, mutate: Any, n_missing: int, n_extra: int
) -> None:
    """⭐ The control: without it, "sets_equal True" is satisfied by any implementation.

    Three different perturbations of a COPY of the roadnet, each with the counts it actually
    produces. ⚠️ The middle case is the one I got wrong first: flipping an out-lane index does not
    create a missing connection, it collides with the sibling laneLink and shrinks the set. The
    expectation is the measurement, not the symmetry one imagines.
    """
    import tempfile

    data = json.loads((HZ1X1 / "roadnet.json").read_text(encoding="utf-8"))
    entry = next(ix for ix in data["intersections"] if ix["id"] == "intersection_1_1")
    mutate(entry)
    moved = Path(tempfile.mkdtemp(prefix="cap_c_control_")) / "roadnet.json"
    moved.write_text(json.dumps(data), encoding="utf-8")

    equality = ca.connection_set_equality(
        moved,
        HZ1X1 / "hangzhou_1x1_bc-tyc_18041610_1h.net.xml",
        lanes_per_road={road["id"]: len(road["lanes"]) for road in data["roads"]},
        intersections=["intersection_1_1"],
    )

    assert equality["sets_equal"] is False
    assert equality["n_missing_in_sumo"] == n_missing
    assert equality["n_extra_in_sumo"] == n_extra


@pytest.mark.parametrize(
    "name, roadnet, net, extras",
    [
        (
            "hz1x1",
            HZ1X1 / "roadnet.json",
            HZ1X1 / "hangzhou_1x1_bc-tyc_18041610_1h.net.xml",
            {"t": 4},
        ),
        (
            "hz4x4",
            HZ4X4 / "roadnet_4X4.json",
            HZ4X4 / "hangzhou_4x4_gudang_18041610_1h.net.xml",
            {"t": 16},
        ),
    ],
)
def test_cap_c_enumerates_the_extra_sumo_connections(
    name: str, roadnet: Path, net: Path, extras: dict[str, int]
) -> None:
    """🔒 H2.1. Kills the reviewer's **M11** (extras never enumerated).

    CAP(C) requires every EXTRA SUMO connection to be enumerated with counts; the `t` turnarounds
    are netconvert's and are not signalled, which is why the set equality above still holds.
    """
    c = _audit(name, roadnet, net)["C_connections"]

    assert c["extra_directions"] == extras
    assert set(c["from_non_internal_edges"]) - {"l", "s", "r"} == set(extras)


def test_the_demand_comparison_detects_a_demand_that_actually_differs() -> None:
    """🔒 H2.5 / the reviewer's M8a: multiset equality with a NEGATIVE CONTROL.

    `multiset_equal := counts equal` passed every test, because every pair on disk agrees. One
    changed departure time and one changed route must each break it, or "2983/2983 as a multiset" is
    protected by nothing but the index-aligned count.
    """
    real = ca.demand_from_route_file(HZ1X1 / "hangzhou_1x1_bc-tyc_18041610_1h.rou.xml")
    reference = ca.demand_from_cityflow_flow(HZ1X1 / "flow.json")

    assert ca._compare_demand(reference, real)["multiset_equal"] is True

    moved_depart = list(real)
    moved_depart[7] = (moved_depart[7][0] + 1.0, moved_depart[7][1])
    shifted = ca._compare_demand(reference, moved_depart)
    assert shifted["multiset_equal"] is False
    assert shifted["n_only_in_cityflow"] == shifted["n_only_in_sumo"] == 1
    assert shifted["counts_equal"] is True  # same LENGTH, different demand

    moved_route = list(real)
    moved_route[3] = (moved_route[3][0], ("a_road_that_does_not_exist",))
    assert ca._compare_demand(reference, moved_route)["multiset_equal"] is False

    # And a pure REORDER must still be equal -- that is the whole point of the multiset.
    assert ca._compare_demand(reference, list(reversed(real)))["multiset_equal"] is True


def test_the_vtype_binding_is_reported_because_defined_is_not_bound() -> None:
    """CAP(E)'s parameter half: hz1x1 and hz4x4 define `pkw` and bind it to zero vehicles."""
    shipped = ca._route_vtype_facts(HZ1X1 / "hangzhou_1x1_bc-tyc_18041610_1h.rou.xml")
    parity_file = ca._route_vtype_facts(
        REPO / "scenarios/hangzhou_1x1_bc-tyc_18041610_1h_parity"
        / "hangzhou_1x1_bc-tyc_18041610_1h_parity.rou.xml"
    )

    assert shipped["n_vehicles"] == 2021 and shipped["n_typed"] == 0
    assert shipped["fully_bound"] is False
    assert list(shipped["vtypes"]) == ["pkw"] and shipped["tau_present"] == {"pkw": False}
    assert parity_file["n_typed"] == parity_file["n_vehicles"] == 2021
    assert parity_file["fully_bound"] is True
    assert parity_file["tau_present"] == {"cf_parity": True}


# ----------------------------------------------------------------------
# Trap 5, and (F)
# ----------------------------------------------------------------------


def test_the_clearance_character_is_reported_per_scenario_and_differs() -> None:
    """🔒 Trap 5: ``r`` on hz1x1, ``s`` on hz4x4 -- one hard-coded expectation would be wrong."""
    hz1 = ca.audit_pair(
        "hz1x1",
        cityflow_roadnet=HZ1X1 / "roadnet.json",
        sumo_net=HZ1X1 / "hangzhou_1x1_bc-tyc_18041610_1h.net.xml",
        direction="CityFlow -> SUMO",
    )["D_signals"]["per_intersection"]["intersection_1_1"]
    hz4 = ca.audit_pair(
        "hz4x4",
        cityflow_roadnet=HZ4X4 / "roadnet_4X4.json",
        sumo_net=HZ4X4 / "hangzhou_4x4_gudang_18041610_1h.net.xml",
        direction="CityFlow -> SUMO",
    )["D_signals"]["per_intersection"]["intersection_1_1"]

    assert hz1["clearance_characters"] == ["r"]
    assert hz4["clearance_characters"] == ["r", "s"]


def test_provenance_records_the_libsignal_lineage_for_hangzhou() -> None:
    """(F): the header names the tool, the version, the date and the inputs it was built from."""
    provenance = ca.sumo_provenance(HZ1X1 / "hangzhou_1x1_bc-tyc_18041610_1h.net.xml")

    assert provenance["tool"] == "netconvert"
    assert provenance["version"] == "1.13.0"
    assert provenance["generated_on"] == "2022-10-19 18:38:30"
    assert provenance["records_its_inputs"] is True
    assert set(provenance["recorded_inputs"]) == {
        "node-files",
        "edge-files",
        "connection-files",
        "tllogic-files",
    }
    assert all(
        "LibSignal" in path for path in provenance["recorded_inputs"].values()
    )


def test_a_net_that_records_no_inputs_is_reported_as_a_provenance_gap(resco: Path) -> None:
    """⚠️ RESCO's net has a ``<configuration>`` with NO ``<input>``: it cannot be traced from itself.

    An absence has to be reported as an absence, not as an empty field a reader skims past.
    """
    provenance = ca.sumo_provenance(resco / "grid4x4.net.xml")

    assert provenance["tool"] == "netedit"
    assert provenance["version"] == "1.9.0"
    assert provenance["records_its_inputs"] is False
    assert provenance["recorded_inputs"] == {}


def test_the_audit_artifact_issues_no_verdict() -> None:
    """⛔ CAP is the admissibility rule and A14 gives it to the coordinator, not to this module."""
    artifact = ca.audit_artifact([])
    blob = json.dumps(artifact).lower()

    assert artifact["n_pairs"] == 0
    assert "admissible" not in blob.replace("admissibility", "")
    assert any("no admissibility verdict" in note.lower() for note in artifact["what_this_does_not_say"])
