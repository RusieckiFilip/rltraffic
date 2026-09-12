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

import pytest

from offline import conversion_audit as ca

REPO = Path(__file__).resolve().parents[1]
HZ1X1 = REPO / "scenarios" / "hangzhou_1x1_bc-tyc_18041610_1h"
HZ4X4 = REPO / "scenarios" / "hangzhou_4x4_gudang_18041610_1h"


@pytest.fixture(scope="module")
def resco() -> Path:
    """RESCO's grid4x4 directory; skip unless the read-only candidates tree is present."""
    env_value = os.environ.get("RLTRAFFIC_GRID4X4_RESCO")
    candidate = (
        Path(env_value)
        if env_value
        else Path("/home/filip/rltraffic/scenarios/grid4x4_candidates/resco")
        / "resco_benchmark/environments/grid4x4"
    )
    if not (candidate / "grid4x4.net.xml").is_file():
        pytest.skip(
            f"RESCO's grid4x4 net is not at {candidate}: set RLTRAFFIC_GRID4X4_RESCO to the "
            "read-only candidates directory to run the grid4x4 half of the audit"
        )
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
