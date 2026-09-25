"""P7.3d C1: the parity contract PER SCENARIO, and gate G1's artifact (``BRIEF_39`` + Amendment A).

Written against ``docs/briefs/BRIEF_39_p7.3d_grid4x4_zero_shot.md`` §3 C1, §4 (T-regress (a),
T-parity-values, T-cap-e) and Amendment A (A5: the templated shape is
``materialised-draw-parity/1.1``; A8: the RESCO net is REFERENCED, never copied).

Every test materialises into ``tmp_path``.  None touches the real ``scenarios/draws/`` tree, the
RESCO candidates tree is only ever READ, and nothing is written under ``output/``.

GATES, each naming the artifact it consumes
-------------------------------------------
* ``RLTRAFFIC_GRID4X4_RESCO`` -- the read-only, gitignored candidates ROOT holding RESCO's grid4x4
  net and route archive (CC BY-NC-SA 4.0).  It has **no default**: unset means skip, naming it.
* ``RLTRAFFIC_DRAWS`` (default: this tree's ``scenarios/draws``) -- the materialised draw tree, for
  the one test that compares against a parity directory P7.2a wrote.

THE LOAD-BEARING CLAIM is T-regress (a): the hangzhou scenario is the default of every entry point
and regenerates **byte-identically** through the per-scenario code.  The committed digests come from
``docs/data/p7_3a_zero_shot.json`` -- read from the artifact, never retyped here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from offline import parity
from offline.materialise_draws import (
    CAP_E_FORMAT_VERSION,
    PARITY_DIRNAME,
    PARITY_FORMAT_VERSION,
    PARITY_FORMAT_VERSION_TEMPLATED,
    PARITY_ROUTES_FILENAME,
    PARITY_SUMOCFG_FILENAME,
    PROVENANCE_FILENAME,
    READABLE_PARITY_FORMAT_VERSIONS,
    assert_cap_e_exact,
    cap_e_report,
    draw_dir,
    load_parity_provenance,
    main,
    materialise,
    materialise_parity,
    parity_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HZ1X1_CONFIG = REPO_ROOT / "configs/sim/cityflow1x1.json"
GRID4X4_CONFIG = REPO_ROOT / "configs/sim/cityflow_grid4x4.json"
HZ1X1_KEY = "cityflow1x1"
GRID4X4_KEY = "cityflow_grid4x4"
P7_3A_ARTIFACT = REPO_ROOT / "docs/data/p7_3a_zero_shot.json"
P8_4A_ARTIFACT = REPO_ROOT / "docs/data/p8_4a_admission.json"

#: ``PREREGISTRATION`` A15(g), typed here ON PURPOSE and never imported: the test must not be able
#: to pass by comparing a constant with itself.  ``id`` and ``speedFactor`` are the parity ruling's.
A15G_GRID4X4_VTYPE = {
    "id": "cf_parity",
    "maxSpeed": "13.39",
    "tau": "1.5",
    "accel": "2.6",
    "decel": "4.5",
    "length": "5.0",
    "minGap": "2.5",
    "width": "1.8",
    "speedFactor": "1.0",
}

#: What P7.0 committed for hangzhou, typed here for the same reason.
P7_0_HZ1X1_VTYPE = {
    "id": "cf_parity",
    "maxSpeed": "11.11",
    "tau": "2.0",
    "accel": "2.0",
    "decel": "4.5",
    "length": "5.0",
    "minGap": "2.5",
    "width": "2.0",
    "speedFactor": "1.0",
}

RESCO_NET_SHA256 = "8d192de462497c192ed51f0a147b7522626e706b411ceba6f16b083b13183b38"
RESCO_MEMBER_SHA256 = "2350dce7e086b8dd27b94c772a4ff883c04c8232c1248c4b889d7c466f4127cc"

#: A drawn rendering as ``FlowRandomizer.render_sumo`` writes one from a template that declares no
#: ``<vType>`` and binds none: exactly what RESCO's ``grid4x4_1.rou.xml`` produces.
UNBOUND_NO_VTYPE = (
    "<?xml version='1.0' encoding='utf-8'?>\n"
    '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
    'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n'
    '    <vehicle id="0" depart="0.00">\n'
    '        <route edges="top3D3 D3D2 D2C2" />\n'
    "    </vehicle>\n"
    '    <vehicle id="1" depart="9.00">\n'
    '        <route edges="left0A0 A0A1" />\n'
    "    </vehicle>\n"
    "</routes>"
)


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tree_snapshot(root: str | Path) -> dict[str, str]:
    """Every file under *root* by relative path -> sha256; whole-tree, so a DELETED file shows."""
    base = Path(root)
    return {
        str(path.relative_to(base)): _sha256_file(path)
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def _vtype_attributes_in(rou_path: Path) -> dict[str, str]:
    """The single ``<vType>``'s attributes AS WRITTEN IN THE FILE."""
    vtypes = ET.parse(rou_path).getroot().findall("vType")
    assert len(vtypes) == 1, f"{rou_path} declares {len(vtypes)} vTypes"
    return {str(k): str(v) for k, v in vtypes[0].attrib.items()}


def _vehicle_types(rou_path: Path) -> Counter[str]:
    root = ET.parse(rou_path).getroot()
    return Counter(str(vehicle.get("type")) for vehicle in root.findall("vehicle"))


@pytest.fixture(scope="module")
def resco_root() -> Path:
    """The read-only candidates ROOT; skip -- naming the variable -- when it is unset."""
    value = os.environ.get("RLTRAFFIC_GRID4X4_RESCO")
    if not value:
        pytest.skip(
            "RLTRAFFIC_GRID4X4_RESCO is unset: point it at the read-only, gitignored candidates "
            "root (the directory holding resco/) to run the grid4x4 half of P7.3d C1. RESCO's "
            "grid4x4 files are CC BY-NC-SA 4.0 and are never copied into the tree"
        )
    net = Path(value) / "resco/resco_benchmark/environments/grid4x4/grid4x4.net.xml"
    if not net.is_file():
        pytest.skip(f"RLTRAFFIC_GRID4X4_RESCO is set but RESCO's grid4x4 net is not at {net}")
    return Path(value)


# ==================================================================================
# T-regress (a) and (c): hangzhou is the default, and it has not moved by one byte
# ==================================================================================
def test_every_hz1x1_default_resolves_to_todays_value() -> None:
    """T-regress (c), parity half.  *Mutation this is built against:* the default stem changed."""
    assert parity.HZ1X1_SCENARIO.key == "cityflow1x1"
    assert parity.HZ1X1_SCENARIO.stem == "hangzhou_1x1_bc-tyc_18041610_1h"
    assert parity.HZ1X1_SCENARIO.flow_json == parity.DECLARED_SOURCE_FLOW_JSON
    assert parity.HZ1X1_SCENARIO.external is None
    assert parity.HZ1X1_SCENARIO.template_declares_vtype is True
    assert parity.PARITY_CONTRACT_VERSION == "1.0"

    # The no-argument forms are what every call written before P7.3d uses.
    assert parity.parity_vtype_attributes() == P7_0_HZ1X1_VTYPE
    assert parity.flow_json_disagreements(parity.DECLARED_SOURCE_FLOW_JSON) == []
    assert parity.render_parity_vtype_element() == (
        '<vType accel="2.0" decel="4.5" id="cf_parity" length="5.0" maxSpeed="11.11" '
        'minGap="2.5" speedFactor="1.0" tau="2.0" width="2.0"/>'
    )


def test_hz1x1s_derived_vtype_is_byte_equal_to_the_table_p7_0_committed() -> None:
    """Deriving from the flow block must not move hangzhou: the STRINGS are compared, not floats.

    ``PARITY_CONTRACT_VERSION`` stays ``1.0`` if and only if this holds (``BRIEF_39`` §3 C1).
    """
    derived = parity.derived_vtype_attributes(parity.HZ1X1_SCENARIO)
    assert derived == P7_0_HZ1X1_VTYPE
    assert list(derived) == [attr.name for attr in parity.PARITY_VTYPE], "declaration order kept"


def test_hz1x1_draw_1000_regenerates_to_the_digests_p7_3a_committed(tmp_path: Path) -> None:
    """T-regress (a), load-bearing.  Draw 1000's parity files against a COMMITTED artifact.

    ``routes.rou.xml`` has no path in it, so the regenerated file's digest is compared directly.
    ``noteleport.sumocfg`` names the network by a path RELATIVE to the parity directory, which is
    ``../../../../hangzhou_...`` in the real tree and something longer under ``tmp_path``; the
    reference is therefore rewritten to the real tree's before hashing, and the test asserts the
    rewrite touched exactly one attribute, so it cannot hide a second difference.

    *Mutation this is built against:* the default stem (or the default table) changed -> the bound
    ``<vType>`` moves -> the routes digest moves -> this dies.
    """
    committed = json.loads(P7_3A_ARTIFACT.read_text(encoding="utf-8"))["inputs"]["demand_by_draw"]["1000"]

    materialise(HZ1X1_CONFIG, [1000], out_root=tmp_path)
    materialise_parity(HZ1X1_CONFIG, [1000], out_root=tmp_path)
    target = parity_dir(HZ1X1_KEY, 1000, out_root=tmp_path)

    assert _sha256_file(target / PARITY_ROUTES_FILENAME) == committed["routes_sha256"]

    record = load_parity_provenance(HZ1X1_KEY, 1000, out_root=tmp_path)
    assert record["format_version"] == PARITY_FORMAT_VERSION == "materialised-draw-parity/1.0"
    real_tree_parity = REPO_ROOT / "scenarios/draws" / HZ1X1_KEY / "draw_1000" / PARITY_DIRNAME
    real_reference = os.path.relpath(parity.DECLARED_SOURCE_NET, real_tree_parity)
    generated = (target / PARITY_SUMOCFG_FILENAME).read_text(encoding="utf-8")
    assert generated.count(record["net"]["reference"]) == 1
    rewritten = generated.replace(record["net"]["reference"], real_reference)
    assert _sha256_text(rewritten) == committed["config_sha256"]


def test_hz1x1_draw_5_regenerates_to_the_parity_directory_p7_2a_wrote(tmp_path: Path) -> None:
    """T-regress (a), second draw.  Draw 5 is in no committed artifact, so this one is gated on the
    gitignored draw tree and compares against the digests its own ``provenance.json`` records."""
    env_value = os.environ.get("RLTRAFFIC_DRAWS")
    draws = Path(env_value) if env_value else REPO_ROOT / "scenarios/draws"
    on_disk = draws / HZ1X1_KEY / "draw_0005" / PARITY_DIRNAME / PROVENANCE_FILENAME
    if not on_disk.is_file():
        pytest.skip(
            f"{on_disk} is absent: set RLTRAFFIC_DRAWS to the scenarios/draws tree P7.2a "
            "materialised (it is gitignored) to compare draw 5 against its parity directory"
        )
    recorded = json.loads(on_disk.read_text(encoding="utf-8"))

    materialise(HZ1X1_CONFIG, [5], out_root=tmp_path)
    materialise_parity(HZ1X1_CONFIG, [5], out_root=tmp_path)
    target = parity_dir(HZ1X1_KEY, 5, out_root=tmp_path)
    record = load_parity_provenance(HZ1X1_KEY, 5, out_root=tmp_path)

    assert _sha256_file(target / PARITY_ROUTES_FILENAME) == recorded["files"][PARITY_ROUTES_FILENAME]
    assert record["vtype_attributes"] == recorded["vtype_attributes"]
    assert record["demand_audit"] == recorded["demand_audit"]
    assert record["format_version"] == recorded["format_version"]
    assert set(record) == set(recorded), "the hangzhou record gained or lost no key"


# ==================================================================================
# T-parity-values: grid4x4's eight registered values come OUT OF THE DATA
# ==================================================================================
def test_grid4x4s_vtype_is_derived_from_its_flow_block_and_the_rendered_file_says_so(
    tmp_path: Path,
) -> None:
    """A15(g)'s eight values, read back from the RENDERED file and compared with a literal.

    *Mutation this is built against:* ``tau`` read from the wrong flow key -> the derivation
    disagrees with the registered table and refuses -> this dies.
    """
    derived = parity.derived_vtype_attributes(parity.GRID4X4_SCENARIO)
    rendered = parity.render_parity_rou_text(UNBOUND_NO_VTYPE, derived, insert_when_absent=True)

    out = tmp_path / "routes.rou.xml"
    out.write_text(rendered, encoding="utf-8")
    assert _vtype_attributes_in(out) == A15G_GRID4X4_VTYPE
    assert _vehicle_types(out) == Counter({"cf_parity": 2})

    report = parity.vtype_binding_report(out)
    assert parity.binding_is_complete(report, derived)
    assert not parity.binding_is_complete(report), "hangzhou's table must NOT accept grid4x4's file"


def test_a_flow_block_edited_by_one_field_is_refused_naming_the_field(tmp_path: Path) -> None:
    """A scenario whose flow block disagrees with the contract is a REFUSAL, not a warning."""
    entries = json.loads(parity.GRID4X4_SCENARIO.flow_json.read_text(encoding="utf-8"))
    for entry in entries:
        entry["vehicle"]["headwayTime"] = 1.6
    edited = tmp_path / "flow.json"
    edited.write_text(json.dumps(entries), encoding="utf-8")

    assert parity.flow_json_disagreements(edited, parity.GRID4X4_SCENARIO) == [
        "tau: declared 1.5 disagrees with its declared source headwayTime=1.6"
    ]
    with pytest.raises(ValueError, match=r"tau: declared 1\.5 disagrees"):
        parity.derived_vtype_attributes(parity.GRID4X4_SCENARIO, edited)
    # ... and the unedited block carries no disagreement under its OWN table, while hangzhou's
    # table rejects it: parity is per scenario.
    assert parity.flow_json_disagreements(parity.GRID4X4_SCENARIO.flow_json, parity.GRID4X4_SCENARIO) == []
    assert len(parity.flow_json_disagreements(parity.GRID4X4_SCENARIO.flow_json)) == 4


def test_the_inserted_vtype_leaves_every_vehicle_element_for_element() -> None:
    """The insertion edits two things -- one new ``<vType>``, one ``type=`` per vehicle -- and the
    text outside them is byte-identical, so the result diffs cleanly against its source."""
    derived = parity.derived_vtype_attributes(parity.GRID4X4_SCENARIO)
    rendered = parity.render_parity_rou_text(UNBOUND_NO_VTYPE, derived, insert_when_absent=True)

    element = parity.render_parity_vtype_element(derived)
    assert rendered.count(element) == 1
    stripped = rendered.replace("\n    " + element, "", 1).replace(' type="cf_parity"', "")
    assert stripped == UNBOUND_NO_VTYPE
    assert rendered.index(element) < rendered.index("<vehicle"), "the type precedes its first use"


def test_the_binding_mode_is_strict_in_both_directions() -> None:
    """Insertion where a ``<vType>`` exists, or replacement where none does, is a refusal: either
    means the source is not the file the scenario's audit described."""
    derived = parity.derived_vtype_attributes(parity.GRID4X4_SCENARIO)
    with_vtype = UNBOUND_NO_VTYPE.replace(
        '    <vehicle id="0"', '    <vType id="pkw" accel="2.6"/>\n    <vehicle id="0"', 1
    )
    with pytest.raises(ValueError, match="declares 1"):
        parity.render_parity_rou_text(with_vtype, derived, insert_when_absent=True)
    # The default mode is hangzhou's, and its message has not moved.
    with pytest.raises(ValueError, match="exactly one <vType> declaration to replace, found 0"):
        parity.render_parity_rou_text(UNBOUND_NO_VTYPE)


def test_an_unknown_scenario_key_is_refused_by_name() -> None:
    assert parity.scenario_for_key("cityflow1x1") is parity.HZ1X1_SCENARIO
    assert parity.scenario_for_key("cityflow_grid4x4") is parity.GRID4X4_SCENARIO
    with pytest.raises(ValueError, match="cityflow_cologne3"):
        parity.scenario_for_key("cityflow_cologne3")


# ==================================================================================
# The RESCO root: no default, read in place, pinned by digest
# ==================================================================================
def test_the_resco_root_has_no_default_and_an_unset_variable_is_refused_by_name(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="RLTRAFFIC_GRID4X4_RESCO is unset"):
        parity.resolve_external_source(parity.GRID4X4_SCENARIO, environ={})
    with pytest.raises(FileNotFoundError, match="RLTRAFFIC_GRID4X4_RESCO"):
        parity.resolve_external_source(
            parity.GRID4X4_SCENARIO, environ={"RLTRAFFIC_GRID4X4_RESCO": str(tmp_path)}
        )
    with pytest.raises(ValueError, match="cityflow1x1 has no external"):
        parity.resolve_external_source(parity.HZ1X1_SCENARIO, environ={})


def test_a_candidates_tree_holding_a_different_file_is_refused_with_both_digests(
    tmp_path: Path,
) -> None:
    """The digests are A15(g)'s.  A different net under the same name is a different scenario."""
    base = tmp_path / "resco/resco_benchmark/environments/grid4x4"
    base.mkdir(parents=True)
    (base / "grid4x4.net.xml").write_text("<net/>", encoding="utf-8")
    (base / "grid4x4.sumocfg").write_text("<configuration/>", encoding="utf-8")
    with zipfile.ZipFile(base / "grid4x4.zip", "w") as archive:
        archive.writestr("grid4x4_1.rou.xml", "<routes/>")

    with pytest.raises(ValueError, match=RESCO_NET_SHA256[:12]):
        parity.resolve_external_source(
            parity.GRID4X4_SCENARIO, environ={"RLTRAFFIC_GRID4X4_RESCO": str(tmp_path)}
        )


def test_the_real_resco_files_resolve_at_a15gs_digests(resco_root: Path) -> None:
    resolved = parity.resolve_external_source(
        parity.GRID4X4_SCENARIO, environ={"RLTRAFFIC_GRID4X4_RESCO": str(resco_root)}
    )
    assert resolved.net_sha256 == RESCO_NET_SHA256 == _sha256_file(resolved.net)
    assert resolved.routes_member_sha256 == RESCO_MEMBER_SHA256
    text = parity.read_route_template_text(resolved)
    assert _sha256_text(text) == RESCO_MEMBER_SHA256, "read in place, byte for byte"
    assert text.count("<vehicle ") == 1473 and text.count("<vType") == 0


def test_grid4x4_parity_is_refused_naming_the_variable_when_it_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset means unresolvable, and a refusal writes nothing (the filesystem-mutation barrier)."""
    monkeypatch.delenv("RLTRAFFIC_GRID4X4_RESCO", raising=False)
    materialise(GRID4X4_CONFIG, [1], out_root=tmp_path)
    before = _tree_snapshot(tmp_path)

    with pytest.raises(ValueError, match="RLTRAFFIC_GRID4X4_RESCO is unset"):
        materialise_parity(GRID4X4_CONFIG, [1], out_root=tmp_path)

    assert _tree_snapshot(tmp_path) == before
    assert [path.name for path in tmp_path.rglob(".staging*")] == []
    assert main(["--env-config", str(GRID4X4_CONFIG), "--draws", "1", "--parity",
                 "--out-root", str(tmp_path)]) == 1


# ==================================================================================
# The templated shape, end to end (RESCO-gated)
# ==================================================================================
def test_grid4x4_parity_end_to_end_binds_every_vehicle_and_touches_no_parent_byte(
    tmp_path: Path, resco_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLTRAFFIC_GRID4X4_RESCO", str(resco_root))
    materialise(GRID4X4_CONFIG, [1, 2], out_root=tmp_path)
    before = _tree_snapshot(tmp_path)

    results = materialise_parity(GRID4X4_CONFIG, [1, 2], out_root=tmp_path)

    after = _tree_snapshot(tmp_path)
    assert {name: digest for name, digest in after.items() if name in before} == before
    assert len(after) == len(before) + 6, "two parity dirs, three files each"
    assert [record.action for record in results] == ["written", "written"]

    for draw_id in (1, 2):
        parent = draw_dir(GRID4X4_KEY, draw_id, out_root=tmp_path)
        target = parity_dir(GRID4X4_KEY, draw_id, out_root=tmp_path)
        assert sorted(path.name for path in parent.iterdir()) == [
            "cityflow.json", "flow.json", "parity", "provenance.json",
        ], "the parent gained a directory and NO file: no routes.rou.xml is added beside it"
        n_flow = len(json.loads((parent / "flow.json").read_text(encoding="utf-8")))

        bound = target / PARITY_ROUTES_FILENAME
        assert _vtype_attributes_in(bound) == A15G_GRID4X4_VTYPE
        assert _vehicle_types(bound) == Counter({"cf_parity": n_flow})

        cfg = ET.parse(target / PARITY_SUMOCFG_FILENAME).getroot()
        assert cfg.find(".//time-to-teleport").get("value") == "-1"  # type: ignore[union-attr]
        assert int(cfg.find(".//end").get("value")) > 3600  # type: ignore[union-attr, arg-type]
        assert cfg.find(".//route-files").get("value") == PARITY_ROUTES_FILENAME  # type: ignore[union-attr]
        net_reference = str(cfg.find(".//net-file").get("value"))  # type: ignore[union-attr]
        assert not os.path.isabs(net_reference), "A8: a RELATIVE reference, never an absolute path"
        assert _sha256_file((target / net_reference).resolve()) == RESCO_NET_SHA256

    # Second run: everything kept, not one byte moved.
    snapshot = _tree_snapshot(tmp_path)
    again = materialise_parity(GRID4X4_CONFIG, [1, 2], out_root=tmp_path)
    assert [record.action for record in again] == ["kept", "kept"]
    assert _tree_snapshot(tmp_path) == snapshot


def test_the_grid4x4_config_says_whose_network_it_references_and_hz1x1s_header_is_unmoved(
    tmp_path: Path, resco_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generated header is hashed into every later cell's ``config_sha256``, so it must be TRUE
    before the first cell exists: grid4x4's network is RESCO's, outside the repository -- not "the
    SHIPPED one", which is what the hangzhou header says and must keep saying, byte for byte
    (T-regress (a) pins that digest).  An XML comment may not contain a double hyphen, so a note
    carrying one is refused rather than written into a file SUMO would reject."""
    default = parity.render_parity_sumocfg_text("x.net.xml", "routes.rou.xml", time_to_teleport=-1)
    assert "The network is the SHIPPED one, referenced rather than copied." in default
    assert default == parity.render_parity_sumocfg_text(
        "x.net.xml", "routes.rou.xml", time_to_teleport=-1, network_note=None
    )
    with pytest.raises(ValueError, match="double hyphen"):
        parity.render_parity_sumocfg_text("x.net.xml", "r.rou.xml", network_note="a -- b")
    with pytest.raises(ValueError, match="one line"):
        parity.render_parity_sumocfg_text("x.net.xml", "r.rou.xml", network_note="a\nb")

    monkeypatch.setenv("RLTRAFFIC_GRID4X4_RESCO", str(resco_root))
    materialise(GRID4X4_CONFIG, [1], out_root=tmp_path)
    materialise_parity(GRID4X4_CONFIG, [1], out_root=tmp_path)
    text = (parity_dir(GRID4X4_KEY, 1, out_root=tmp_path) / PARITY_SUMOCFG_FILENAME).read_text(
        encoding="utf-8"
    )
    assert "SHIPPED" not in text
    assert "RESCO" in text and "CC BY-NC-SA 4.0" in text and "OUTSIDE the repository" in text
    ET.fromstring(text)  # well-formed: the note did not break the comment


def test_the_templated_record_is_version_1_1_and_pins_both_resco_digests(
    tmp_path: Path, resco_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amendment A5, and the brief's *every provenance record carries the RESCO net's and route
    member's sha256*.  No RESCO path is recorded absolutely: the root is written ``<candidates>``."""
    monkeypatch.setenv("RLTRAFFIC_GRID4X4_RESCO", str(resco_root))
    materialise(GRID4X4_CONFIG, [1], out_root=tmp_path)
    materialise_parity(GRID4X4_CONFIG, [1], out_root=tmp_path)
    record = load_parity_provenance(GRID4X4_KEY, 1, out_root=tmp_path)
    parent = draw_dir(GRID4X4_KEY, 1, out_root=tmp_path)

    assert record["format_version"] == PARITY_FORMAT_VERSION_TEMPLATED == "materialised-draw-parity/1.1"
    assert record["parent"]["routes_sha256"] is None
    assert "grid4x4.rou.xml" in record["parent"]["routes_absent_reason"]
    assert record["parent"]["flow_sha256"] == _sha256_file(parent / "flow.json")
    assert record["parent"]["provenance_sha256"] == _sha256_file(parent / "provenance.json")
    assert record["vtype_attributes"] == A15G_GRID4X4_VTYPE
    assert record["net"]["sha256"] == RESCO_NET_SHA256
    assert record["net"]["resolved"].startswith("<candidates>/")
    template = record["route_template"]
    assert template["member"] == "grid4x4_1.rou.xml"
    assert template["member_sha256"] == RESCO_MEMBER_SHA256
    assert template["archive"].startswith("<candidates>/") and template["declares_vtype"] is False
    assert template["sumocfg_sha256"] == _sha256_file(
        resco_root / "resco/resco_benchmark/environments/grid4x4/grid4x4.sumocfg"
    )
    assert record["demand_audit"]["multiset_equal"] is True
    assert record["demand_audit"]["order_matches"] is True
    assert record["n_bound"] == record["n_vehicles"] == record["demand_audit"]["n_cityflow"]
    assert str(resco_root) not in json.dumps(
        {k: v for k, v in record.items() if k != "net"}
    ), "the candidates root must not be recorded outside net.reference, which is relative"


def test_a_differing_grid4x4_parity_dir_is_refused_and_nothing_is_overwritten(
    tmp_path: Path, resco_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLTRAFFIC_GRID4X4_RESCO", str(resco_root))
    materialise(GRID4X4_CONFIG, [1], out_root=tmp_path)
    materialise_parity(GRID4X4_CONFIG, [1], out_root=tmp_path)
    routes = parity_dir(GRID4X4_KEY, 1, out_root=tmp_path) / PARITY_ROUTES_FILENAME
    routes.write_text(routes.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered = _tree_snapshot(tmp_path)

    with pytest.raises(FileExistsError, match="differs"):
        materialise_parity(GRID4X4_CONFIG, [1], out_root=tmp_path)
    assert _tree_snapshot(tmp_path) == tampered


def test_an_unknown_parity_format_version_is_refused_by_the_reader(tmp_path: Path) -> None:
    assert READABLE_PARITY_FORMAT_VERSIONS == (
        "materialised-draw-parity/1.0",
        "materialised-draw-parity/1.1",
    )
    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path)
    path = parity_dir(HZ1X1_KEY, 1, out_root=tmp_path) / PROVENANCE_FILENAME
    record = json.loads(path.read_text(encoding="utf-8"))
    record["format_version"] = "materialised-draw-parity/9.9"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match=r"materialised-draw-parity/9\.9"):
        load_parity_provenance(HZ1X1_KEY, 1, out_root=tmp_path)


# ==================================================================================
# T-cap-e: gate G1's artifact (RESCO-gated)
# ==================================================================================
def _grid4x4_tree(tmp_path: Path, draws: list[int]) -> Path:
    materialise(GRID4X4_CONFIG, draws, out_root=tmp_path)
    materialise_parity(GRID4X4_CONFIG, draws, out_root=tmp_path)
    return tmp_path


def test_cap_e_is_exact_on_a_rendered_draw_and_the_artifact_says_what_it_read(
    tmp_path: Path, resco_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLTRAFFIC_GRID4X4_RESCO", str(resco_root))
    root = _grid4x4_tree(tmp_path, [1, 1000])

    report = cap_e_report(GRID4X4_CONFIG, [1, 1000], out_root=root, pedigree_artifact=P8_4A_ARTIFACT)

    assert report["format_version"] == CAP_E_FORMAT_VERSION
    assert report["scenario_key"] == GRID4X4_KEY
    assert report["n_draws"] == 2 and report["n_exact"] == 2 and report["condition_met"] is True
    assert report["vtype_attributes_registered"] == A15G_GRID4X4_VTYPE
    assert report["resco"] == {"net_sha256": RESCO_NET_SHA256, "route_member_sha256": RESCO_MEMBER_SHA256}
    for row in report["draws"]:
        parent_flow = draw_dir(GRID4X4_KEY, row["draw_id"], out_root=root) / "flow.json"
        n_flow = len(json.loads(parent_flow.read_text(encoding="utf-8")))
        assert row["exact"] is True and row["reasons"] == []
        assert row["e_demand"]["n_cityflow"] == row["e_demand"]["n_sumo"] == n_flow
        assert row["e_demand"]["multiset_equal"] is True and row["e_demand"]["order_matches"] is True
        assert row["binding"]["n_bound"] == row["binding"]["n_vehicles"] == n_flow
        assert row["binding"]["vtype_attributes_in_file"] == A15G_GRID4X4_VTYPE
        assert row["binding"]["distinct_type_values"] == ["cf_parity"]
    assert_cap_e_exact(report)

    # The pedigree block: draw 1000's regenerated flow.json is the file P8.4a committed a digest of.
    pedigree = report["pedigree"]
    committed = json.loads(P8_4A_ARTIFACT.read_text(encoding="utf-8"))
    expected = committed["draw_restoration"]["grid4x4"]["flow_sha256"]["1000"]
    assert pedigree["n_checked"] == 1 and pedigree["n_matching"] == 1
    assert pedigree["draws"] == {"1000": {"flow_sha256": expected, "matches": True}}
    assert pedigree["not_covered"] == [1], "a draw outside the held-out pool has no committed digest"
    assert str(resco_root) not in json.dumps(report), "no machine-local root reaches docs/data/"


def test_cap_e_refuses_one_shifted_depart_naming_the_draw(
    tmp_path: Path, resco_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One vehicle's depart moved by a second, with the recorded digest moved WITH it -- so the
    only check that can see the defect is the (depart, route) multiset.

    That is the failure CAP(E) exists for: a renderer that shifts a departure writes a file that is
    perfectly consistent with its own provenance.  *Mutation this is built against:* compare COUNTS
    only -> the shifted depart survives -> this dies.
    """
    monkeypatch.setenv("RLTRAFFIC_GRID4X4_RESCO", str(resco_root))
    root = _grid4x4_tree(tmp_path, [1, 2])
    target = parity_dir(GRID4X4_KEY, 2, out_root=root)
    routes = target / PARITY_ROUTES_FILENAME
    text = routes.read_text(encoding="utf-8")
    first = ET.parse(routes).getroot().findall("vehicle")[5]
    old = f'depart="{first.get("depart")}"'
    new = f'depart="{float(str(first.get("depart"))) + 1.0:.2f}"'
    assert text.count(f'id="5" {old}') == 1
    routes.write_text(text.replace(f'id="5" {old}', f'id="5" {new}', 1), encoding="utf-8")
    provenance = target / PROVENANCE_FILENAME
    record = json.loads(provenance.read_text(encoding="utf-8"))
    record["files"][PARITY_ROUTES_FILENAME] = _sha256_file(routes)
    provenance.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = cap_e_report(GRID4X4_CONFIG, [1, 2], out_root=root)

    by_draw = {row["draw_id"]: row for row in report["draws"]}
    assert by_draw[1]["exact"] is True
    assert by_draw[2]["exact"] is False
    assert by_draw[2]["e_demand"]["counts_equal"] is True, "the counts are blind to this defect"
    assert by_draw[2]["e_demand"]["multiset_equal"] is False
    assert report["n_exact"] == 1 and report["condition_met"] is False
    with pytest.raises(ValueError, match=r"draw 2\b.*multiset"):
        assert_cap_e_exact(report)


def test_cap_e_refuses_a_draw_with_one_unbound_vehicle(
    tmp_path: Path, resco_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The binding half of G1: bound == vehicles on EVERY draw."""
    monkeypatch.setenv("RLTRAFFIC_GRID4X4_RESCO", str(resco_root))
    root = _grid4x4_tree(tmp_path, [1])
    target = parity_dir(GRID4X4_KEY, 1, out_root=root)
    routes = target / PARITY_ROUTES_FILENAME
    text = routes.read_text(encoding="utf-8")
    assert text.count('<vehicle id="0" ') == 1
    head, tail = text.split('<vehicle id="0" ', 1)
    routes.write_text(head + '<vehicle id="0" ' + tail.replace(' type="cf_parity"', "", 1), encoding="utf-8")
    provenance = target / PROVENANCE_FILENAME
    record = json.loads(provenance.read_text(encoding="utf-8"))
    record["files"][PARITY_ROUTES_FILENAME] = _sha256_file(routes)
    provenance.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = cap_e_report(GRID4X4_CONFIG, [1], out_root=root)

    row = report["draws"][0]
    n_vehicles = row["binding"]["n_vehicles"]
    assert row["exact"] is False
    assert row["binding"]["n_bound"] == n_vehicles - 1
    # The REASON is pinned, not only the verdict: a second guard (the full-table check) also sees
    # this file as incomplete, and with the count check removed it would refuse under a reason
    # that says "every vehicle is bound" -- false, and written into a registered gate's artifact.
    assert row["reasons"] == [
        f"binding: {n_vehicles - 1} of {n_vehicles} vehicles are bound to a type"
    ]
    with pytest.raises(ValueError, match=rf"draw 1\b.*{n_vehicles - 1} of {n_vehicles} vehicles are bound"):
        assert_cap_e_exact(report)


def test_cap_e_refuses_a_fully_bound_draw_whose_vtype_carries_the_wrong_tau(
    tmp_path: Path, resco_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every vehicle bound -- to a type whose ``tau`` is SUMO's default-shaped 1.0, not 1.5.

    ``vehicles carrying type= == vehicles`` alone accepts this file, and ``tau`` is the parameter
    the whole parity precondition exists for (P7.0 measured its omission at +49 % travel time), so
    the attributes are read back from the rendered file and compared with the derived table.
    *Mutation this is built against:* the full-table check dropped -> this dies.
    """
    monkeypatch.setenv("RLTRAFFIC_GRID4X4_RESCO", str(resco_root))
    root = _grid4x4_tree(tmp_path, [1])
    target = parity_dir(GRID4X4_KEY, 1, out_root=root)
    routes = target / PARITY_ROUTES_FILENAME
    text = routes.read_text(encoding="utf-8")
    assert text.count('tau="1.5"') == 1
    routes.write_text(text.replace('tau="1.5"', 'tau="1.0"', 1), encoding="utf-8")
    provenance = target / PROVENANCE_FILENAME
    record = json.loads(provenance.read_text(encoding="utf-8"))
    record["files"][PARITY_ROUTES_FILENAME] = _sha256_file(routes)
    provenance.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = cap_e_report(GRID4X4_CONFIG, [1], out_root=root)

    row = report["draws"][0]
    assert row["binding"]["n_bound"] == row["binding"]["n_vehicles"], "fully bound, and still wrong"
    assert row["binding"]["vtype_attributes_in_file"]["tau"] == "1.0"
    assert row["e_demand"]["multiset_equal"] is True and row["e_demand"]["order_matches"] is True
    assert row["exact"] is False
    assert len(row["reasons"]) == 1 and "not to exactly the parity <vType>" in row["reasons"][0]
    with pytest.raises(ValueError, match=r"draw 1\b.*not to exactly the parity <vType>"):
        assert_cap_e_exact(report)


def test_cap_e_refuses_a_file_that_is_not_the_one_its_provenance_records(
    tmp_path: Path, resco_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLTRAFFIC_GRID4X4_RESCO", str(resco_root))
    root = _grid4x4_tree(tmp_path, [1])
    routes = parity_dir(GRID4X4_KEY, 1, out_root=root) / PARITY_ROUTES_FILENAME
    routes.write_text(routes.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = cap_e_report(GRID4X4_CONFIG, [1], out_root=root)

    assert report["draws"][0]["exact"] is False
    with pytest.raises(ValueError, match=r"draw 1\b.*digest"):
        assert_cap_e_exact(report)


def test_the_cap_e_cli_writes_the_artifact_outside_the_draw_tree_and_exits_by_the_verdict(
    tmp_path: Path, resco_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLTRAFFIC_GRID4X4_RESCO", str(resco_root))
    root = tmp_path / "draws"
    _grid4x4_tree(root, [1])
    out = tmp_path / "artifact" / "p7_3d_cap_e.json"
    base = ["--env-config", str(GRID4X4_CONFIG), "--draws", "1", "--out-root", str(root)]

    assert main(base + ["--report-cap-e", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["condition_met"] is True and written["n_exact"] == written["n_draws"] == 1

    inside = root / "p7_3d_cap_e.json"
    assert main(base + ["--report-cap-e", str(inside)]) == 1
    assert not inside.exists(), "this mode writes nothing under the draws tree"

    # A verdict that is NOT met is still an artifact -- G1 is "decided by the artifact" -- and the
    # exit code carries it.
    routes = parity_dir(GRID4X4_KEY, 1, out_root=root) / PARITY_ROUTES_FILENAME
    routes.write_text(routes.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    failed = tmp_path / "artifact" / "failed.json"
    assert main(base + ["--report-cap-e", str(failed)]) == 1
    assert json.loads(failed.read_text(encoding="utf-8"))["condition_met"] is False

    shutil.rmtree(parity_dir(GRID4X4_KEY, 1, out_root=root))
    missing = tmp_path / "artifact" / "missing.json"
    assert main(base + ["--report-cap-e", str(missing)]) == 1
    assert not missing.exists(), "a refusal precedes the write"
