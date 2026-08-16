"""P7.0 commit 2 -- the vType binding, its positive control, and its provenance.

Everything here guards `BRIEF_21` section 2's mandated precondition and `BRIEF_04`
section 3's parity contract, which `BRIEF_21` declares still binding.

The load-bearing pair is :func:`test_shipped_file_is_not_bound_positive_control`
and :func:`test_parity_file_binds_every_vehicle`.  The first is the control the
brief requires: *"the check must be shown to FAIL on the unfixed file before it is
trusted"*.  Without it, a binding check that always returned True would pass the
second test and prove nothing.

Convention pinned here, because a later reader will otherwise assume the wrong one:
the parity type is declared under a **distinct id** (``cf_parity``), never as a
redefinition of the shipped ``pkw``.  The shipped ``pkw`` omits ``tau``, and a name
that means two different things in two files is the defect class `docs/CONTRACTS.md`
C8 note 2 records.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from offline import parity
from offline.policies.fixed_time import (
    equal_split_cycle,
    green_action_phases,
    parse_signal_plan,
)
from utils.cityflow_utils import parse_roadnet
from utils.sumo_utils import validate_sumo_inputs_exist

EXPECTED_VEHICLES = 2021


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _vehicle_tuples(rou_path: Path) -> list[tuple[str, str, str]]:
    """``(id, depart, route edges)`` in file order -- the demand's identity."""
    root = ET.parse(rou_path).getroot()
    out: list[tuple[str, str, str]] = []
    for veh in root.findall("vehicle"):
        route = veh.find("route")
        assert route is not None, f"vehicle {veh.get('id')} carries no <route>"
        out.append((str(veh.get("id")), str(veh.get("depart")), str(route.get("edges"))))
    return out


# ----------------------------------------------------------------------
# T1 / T2 -- the mechanical acceptance check and its positive control
# ----------------------------------------------------------------------


def test_shipped_file_is_not_bound_positive_control() -> None:
    """The unfixed file must FAIL the check (`BRIEF_21` section 2).

    This is the control, not a description of a defect: it proves the checker can
    return False, so :func:`test_parity_file_binds_every_vehicle` means something.
    """
    report = parity.vtype_binding_report(parity.DECLARED_SOURCE_ROU)

    assert report.vehicle_count == EXPECTED_VEHICLES
    assert report.vehicles_with_type == 0
    assert report.distinct_type_values == ()
    assert report.vtype_ids == ("pkw",)
    # The shipped type is precisely the one the parity contract cannot use.
    assert "tau" not in report.vtype_attributes
    assert report.vtype_attributes["maxSpeed"] == "11.111"

    assert parity.binding_is_complete(report) is False


def test_parity_file_binds_every_vehicle() -> None:
    report = parity.vtype_binding_report(parity.DECLARED_PARITY_ROU)

    assert report.vehicle_count == EXPECTED_VEHICLES
    assert report.vehicles_with_type == report.vehicle_count
    assert report.distinct_type_values == (parity.PARITY_VTYPE_ID,)
    assert report.vtype_ids == (parity.PARITY_VTYPE_ID,)

    assert parity.binding_is_complete(report) is True


def test_vehicle_count_double_computed_by_regex_and_by_parser() -> None:
    """Two independent counts of the same quantity, asserted exactly equal."""
    for path in (parity.DECLARED_SOURCE_ROU, parity.DECLARED_PARITY_ROU):
        text = Path(path).read_text(encoding="utf-8")
        by_regex = len(re.findall(r"<vehicle\b", text))
        by_parser = len(ET.parse(path).getroot().findall("vehicle"))
        assert by_regex == by_parser == EXPECTED_VEHICLES, path


def test_binding_is_complete_rejects_a_partial_binding(tmp_path: Path) -> None:
    """One unbound vehicle is enough to fail -- the check is not a majority vote."""
    text = Path(parity.DECLARED_PARITY_ROU).read_text(encoding="utf-8")
    broken = text.replace(f' type="{parity.PARITY_VTYPE_ID}">', ">", 1)
    assert broken != text, "the mutation did not apply; the fixture is vacuous"
    target = tmp_path / "broken.rou.xml"
    target.write_text(broken, encoding="utf-8")

    report = parity.vtype_binding_report(target)
    assert report.vehicles_with_type == EXPECTED_VEHICLES - 1
    assert parity.binding_is_complete(report) is False


def test_binding_is_complete_rejects_a_foreign_type_id(tmp_path: Path) -> None:
    text = Path(parity.DECLARED_PARITY_ROU).read_text(encoding="utf-8")
    broken = text.replace(f'type="{parity.PARITY_VTYPE_ID}">', 'type="pkw">', 1)
    assert broken != text, "the mutation did not apply; the fixture is vacuous"
    target = tmp_path / "foreign.rou.xml"
    target.write_text(broken, encoding="utf-8")

    report = parity.vtype_binding_report(target)
    assert report.vehicles_with_type == EXPECTED_VEHICLES
    assert set(report.distinct_type_values) == {parity.PARITY_VTYPE_ID, "pkw"}
    assert parity.binding_is_complete(report) is False


# ----------------------------------------------------------------------
# T3 / T4 -- the emitted type equals the declared table equals flow.json
# ----------------------------------------------------------------------


def test_emitted_vtype_equals_the_declared_table_exactly() -> None:
    report = parity.vtype_binding_report(parity.DECLARED_PARITY_ROU)
    assert report.vtype_attributes == parity.parity_vtype_attributes()


def test_declared_table_equals_flow_json_attribute_by_attribute() -> None:
    """`BRIEF_04` section 6: the committed table cannot silently drift from its source.

    Read independently here rather than through the module, and compared with ``==``
    on floats: both sides parse the same decimal literal, so exact equality holds and
    loosening it would hide precisely the drift this test exists to catch.
    """
    raw = json.loads(Path(parity.DECLARED_SOURCE_FLOW_JSON).read_bytes())
    blocks = {json.dumps(entry["vehicle"], sort_keys=True) for entry in raw}
    assert len(blocks) == 1, (
        "flow.json carries more than one distinct vehicle parameter block, so a single "
        f"parity type cannot represent it: {sorted(blocks)}"
    )
    block = json.loads(blocks.pop())

    sourced = [attr for attr in parity.PARITY_VTYPE if attr.source_key is not None]
    assert sourced, "no parity attribute claims a flow.json source"
    for attr in sourced:
        assert attr.source_key in block, attr.source_key
        assert float(attr.value) == float(block[attr.source_key]), (
            f"{attr.name}: declared {attr.value} against flow.json "
            f"{attr.source_key}={block[attr.source_key]}"
        )

    assert parity.flow_json_disagreements(parity.DECLARED_SOURCE_FLOW_JSON) == []


def test_declared_table_carries_the_binding_parity_contract() -> None:
    """`BRIEF_04` section 3: tau and speedFactor are what the shipped pkw omits."""
    attrs = parity.parity_vtype_attributes()
    assert attrs["id"] == parity.PARITY_VTYPE_ID
    assert attrs["tau"] == "2.0"
    assert attrs["speedFactor"] == "1.0"
    assert attrs["maxSpeed"] == "11.11"
    for name in ("accel", "decel", "length", "minGap", "width"):
        assert name in attrs, name


def test_unmatchable_parameters_are_declared_and_never_emitted() -> None:
    """`BRIEF_04` section 3: match parameters, never models."""
    names = {name for name, _reason in parity.UNMATCHABLE_PARAMETERS}
    assert {"sigma", "usualPosAcc", "usualNegAcc"} <= names
    emitted = parity.parity_vtype_attributes()
    for name, reason in parity.UNMATCHABLE_PARAMETERS:
        assert name not in emitted, f"{name} is declared unmatchable but emitted"
        assert len(reason) >= 10, f"{name} is declared unmatchable with no reason"


def test_flow_json_disagreements_reports_a_planted_drift(tmp_path: Path) -> None:
    """The drift detector must be able to fire, or its empty list means nothing."""
    raw = json.loads(Path(parity.DECLARED_SOURCE_FLOW_JSON).read_bytes())
    for entry in raw:
        entry["vehicle"]["headwayTime"] = 1.5
    planted = tmp_path / "flow.json"
    planted.write_text(json.dumps(raw), encoding="utf-8")

    disagreements = parity.flow_json_disagreements(planted)
    assert len(disagreements) == 1
    assert "tau" in disagreements[0]
    assert "1.5" in disagreements[0]


# ----------------------------------------------------------------------
# T5 / T6 -- regeneration, and the demand surviving the transformation
# ----------------------------------------------------------------------


def test_parity_rou_regenerates_byte_identically() -> None:
    source_text = Path(parity.DECLARED_SOURCE_ROU).read_text(encoding="utf-8")
    regenerated = parity.render_parity_rou_text(source_text)
    committed = Path(parity.DECLARED_PARITY_ROU).read_text(encoding="utf-8")
    assert regenerated == committed


def test_rendering_the_parity_file_leaves_the_shipped_file_untouched() -> None:
    """The training-domain scenario does not move (`BRIEF_04` section 5)."""
    before = _sha256(parity.DECLARED_SOURCE_ROU)
    parity.render_parity_rou_text(
        Path(parity.DECLARED_SOURCE_ROU).read_text(encoding="utf-8")
    )
    assert _sha256(parity.DECLARED_SOURCE_ROU) == before


def test_demand_is_preserved_as_an_ordered_list() -> None:
    """2021 vehicles, same ids, same departures, same routes, same order."""
    shipped = _vehicle_tuples(Path(parity.DECLARED_SOURCE_ROU))
    parity_side = _vehicle_tuples(Path(parity.DECLARED_PARITY_ROU))
    assert len(shipped) == EXPECTED_VEHICLES
    assert shipped == parity_side


def test_rendering_refuses_a_source_that_already_binds_a_type() -> None:
    text = Path(parity.DECLARED_PARITY_ROU).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="already carries a type= binding"):
        parity.render_parity_rou_text(text)


def test_rendering_refuses_a_source_with_more_than_one_vtype() -> None:
    text = Path(parity.DECLARED_SOURCE_ROU).read_text(encoding="utf-8")
    extra = text.replace(
        "<vehicle", '<vType id="second" length="4.0"/>\n\t\t<vehicle', 1
    )
    with pytest.raises(ValueError, match="exactly one <vType> declaration"):
        parity.render_parity_rou_text(extra)


# ----------------------------------------------------------------------
# The generated .sumocfg
# ----------------------------------------------------------------------


def test_parity_sumocfg_inputs_resolve() -> None:
    validate_sumo_inputs_exist(parity.DECLARED_PARITY_SUMOCFG)


def test_parity_sumocfg_points_at_the_shipped_net_and_the_parity_routes() -> None:
    root = ET.parse(parity.DECLARED_PARITY_SUMOCFG).getroot()
    net = root.find(".//net-file")
    routes = root.find(".//route-files")
    assert net is not None and routes is not None
    net_path = (Path(parity.DECLARED_PARITY_SUMOCFG).parent / str(net.get("value"))).resolve()
    routes_path = (
        Path(parity.DECLARED_PARITY_SUMOCFG).parent / str(routes.get("value"))
    ).resolve()
    assert net_path == Path(parity.DECLARED_SOURCE_NET).resolve(), (
        "the parity scenario must reference the SHIPPED network; a second copy is a "
        "drift source"
    )
    assert routes_path == Path(parity.DECLARED_PARITY_ROU).resolve()


def test_parity_sumocfg_end_exceeds_the_env_horizon() -> None:
    """A shorter <end> would truncate SUMO while CityFlow ran on -- a
    backend-asymmetric truncation inside the comparison the gate makes."""
    root = ET.parse(parity.DECLARED_PARITY_SUMOCFG).getroot()
    end = root.find(".//end")
    assert end is not None
    assert int(str(end.get("value"))) == parity.SUMO_END_SECONDS
    assert parity.SUMO_END_SECONDS > parity.ENV_HORIZON_SECONDS


# ----------------------------------------------------------------------
# The filesystem-mutation barrier
# ----------------------------------------------------------------------


def test_a_failed_construction_creates_no_directory(tmp_path: Path) -> None:
    """CLAUDE.md's barrier: validation completes before any write."""
    bad_source = tmp_path / "bad.rou.xml"
    bad_source.write_text("<routes></routes>", encoding="utf-8")
    dest = tmp_path / "never_created"

    with pytest.raises(ValueError, match="exactly one <vType> declaration"):
        parity.write_parity_scenario(
            source_rou=bad_source,
            source_net=parity.DECLARED_SOURCE_NET,
            source_flow_json=parity.DECLARED_SOURCE_FLOW_JSON,
            dest_dir=dest,
            dest_stem="never",
        )

    assert not dest.exists()


def test_a_flow_json_disagreement_blocks_the_write(tmp_path: Path) -> None:
    raw = json.loads(Path(parity.DECLARED_SOURCE_FLOW_JSON).read_bytes())
    for entry in raw:
        entry["vehicle"]["maxSpeed"] = 13.39
    planted = tmp_path / "flow.json"
    planted.write_text(json.dumps(raw), encoding="utf-8")
    dest = tmp_path / "never_created"

    with pytest.raises(ValueError, match="disagrees with its declared source"):
        parity.write_parity_scenario(
            source_rou=parity.DECLARED_SOURCE_ROU,
            source_net=parity.DECLARED_SOURCE_NET,
            source_flow_json=planted,
            dest_dir=dest,
            dest_stem="never",
        )

    assert not dest.exists()


def test_write_parity_scenario_round_trips_into_a_clean_directory(tmp_path: Path) -> None:
    dest = tmp_path / "generated"
    written = parity.write_parity_scenario(
        source_rou=parity.DECLARED_SOURCE_ROU,
        source_net=parity.DECLARED_SOURCE_NET,
        source_flow_json=parity.DECLARED_SOURCE_FLOW_JSON,
        dest_dir=dest,
        dest_stem=parity.DECLARED_PARITY_STEM,
    )
    assert set(written) == {"rou", "sumocfg"}
    assert Path(written["rou"]).read_text(encoding="utf-8") == Path(
        parity.DECLARED_PARITY_ROU
    ).read_text(encoding="utf-8")
    report = parity.vtype_binding_report(written["rou"])
    assert parity.binding_is_complete(report) is True


# ----------------------------------------------------------------------
# T7 -- the fixed-time anchor is the same controller in both backends
# ----------------------------------------------------------------------


def test_shipped_plan_cycle_equals_the_equal_split_cycle_on_this_scenario() -> None:
    """Why this is load-bearing rather than trivia.

    ``offline/policies/fixed_time.py`` resolves the shipped plan by ``json.loads``-ing
    the env config, so a ``.sumocfg`` raises and SUMO silently falls back to
    ``equal_split`` while CityFlow follows the plan.  On this scenario the two coincide
    because the plan's green order is ascending -- so rho's zero anchor is the same
    controller in both backends.  If that ever stops being true the anchor is
    confounded, and this test is what says so.
    """
    plan = parse_signal_plan(
        Path(parity.DECLARED_SCENARIO_DIR / "signal_plan_template.txt").read_text(
            encoding="utf-8"
        )
    )
    roadnet = parse_roadnet(parity.DECLARED_SCENARIO_DIR / "roadnet.json")
    assert len(roadnet.intersections) == 1
    greens = green_action_phases(roadnet.intersections[0])
    assert len(greens) == 8

    index = {phase: action for action, phase in enumerate(greens)}
    shipped_cycle = tuple(index[phase] for phase in plan.green_order)
    assert shipped_cycle == equal_split_cycle(len(greens))
    assert shipped_cycle == (0, 1, 2, 3, 4, 5, 6, 7)
