"""Vehicle-parameter parity between CityFlow and SUMO, as a committed artifact.

Written for P7.0 (`docs/briefs/BRIEF_21_p7.0_transfer_gate.md` section 2 and
`docs/plans/p7.0.md` section 4).  It exists as a module rather than as a note because
P7.3 must consume the matched values without re-deriving them, and because a binding
performed by hand once is a binding nobody can re-check.

Parity contract version
-----------------------
**v1.0** -- `BRIEF_04` section 3's ruling of 2026-08-04, which `BRIEF_21` declares
still binding.  Matched, because both engines expose them: ``tau``/``headwayTime``,
``accel``, ``decel``, ``maxSpeed``, ``length``, ``minGap``.  Direction: **the SUMO
side is configured to match CityFlow**, not because CityFlow is authoritative but
because CityFlow is the training domain and the substrate of the whole C1/C2 corpus,
so it must not move.  ``speedFactor`` is pinned to 1.0.  ``sigma`` stays native and
is declared unmatchable -- it is a *model* parameter, and zeroing it would replace
SUMO's car-following model rather than align it.  **Match parameters, never models.**

On-disk formats written here
----------------------------
1. A SUMO route file (``routes_file.xsd``), produced from the shipped one by exactly
   two edits: the single ``<vType>`` is replaced by the parity type, and every
   ``<vehicle>`` open tag gains ``type="cf_parity"``.  **Ordering convention: vehicle
   order, ids, departure times and routes are preserved element for element**, and
   :func:`render_parity_rou_text` verifies that by parsing its own output before it is
   returned.  The transformation is textual so that everything it does not touch --
   indentation, line endings, attribute order on ``<vehicle>`` -- stays byte-identical
   and the result diffs cleanly against its source.
2. A SUMO configuration (``sumoConfiguration.xsd``) whose ``net-file`` points at the
   **shipped** network by relative path.  No copy of the network is made: two copies
   of one road network is a drift source, and the gate's whole claim is that the two
   backends see the same one.

Why the parity type has a NEW id
--------------------------------
`BRIEF_21` section 2 suggested reusing ``pkw``.  The shipped ``pkw`` does not set
``tau``, and `docs/notes/P7.0_vtype_investigation.md` measured that omission at +49 %
travel time -- so the parity type is not the shipped type, and giving the two the same
name would make one identifier mean two different things in two files.  That is the
failure mode `docs/CONTRACTS.md` C8 note 2 records ("a same-width swap is SILENT and
worse").  The id is therefore ``cf_parity``, and :func:`binding_is_complete` checks
the id and the attributes as well as the count, which is strictly stronger than the
brief's mechanical criterion.

⚠️ **The check must be shown to fail on the unfixed file before it is trusted.** That
control lives in ``tests/test_parity_vtype.py``; the shipped route file declares one
correct-looking ``<vType>`` and binds it to **0 of 2021** vehicles, which is exactly
the failure this module exists to make impossible to ship unnoticed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__all__ = [
    "PARITY_CONTRACT_VERSION",
    "REPO_ROOT",
    "DECLARED_SCENARIO_DIR",
    "DECLARED_SOURCE_ROU",
    "DECLARED_SOURCE_NET",
    "DECLARED_SOURCE_FLOW_JSON",
    "DECLARED_PARITY_DIR",
    "DECLARED_PARITY_STEM",
    "DECLARED_PARITY_ROU",
    "DECLARED_PARITY_SUMOCFG",
    "PARITY_VTYPE_ID",
    "PARITY_VTYPE",
    "UNMATCHABLE_PARAMETERS",
    "SUMO_END_SECONDS",
    "ENV_HORIZON_SECONDS",
    "ParityAttribute",
    "BindingReport",
    "parity_vtype_attributes",
    "read_cityflow_vehicle_block",
    "flow_json_disagreements",
    "render_parity_vtype_element",
    "render_parity_rou_text",
    "render_parity_sumocfg_text",
    "vtype_binding_report",
    "binding_is_complete",
    "write_parity_scenario",
    "main",
]

PARITY_CONTRACT_VERSION = "1.0"

PARITY_VTYPE_ID = "cf_parity"

#: SUMO ``<end>`` for the generated .sumocfg, and the env horizon it must exceed.
#:
#: The env, not the config, is the authority on episode length: ``max_steps`` (360) x
#: ``delta_time`` (10) = 3600 simulated seconds.  At the shipped ``end=3600`` SUMO
#: would terminate on exactly the boundary step, so SUMO would truncate while CityFlow
#: ran on -- a backend-asymmetric truncation inside the very comparison P7.0 makes.
#: The last departure in this scenario is at t = 3599, so the extension adds no demand.
SUMO_END_SECONDS = 4000
ENV_HORIZON_SECONDS = 3600

REPO_ROOT = Path(__file__).resolve().parent.parent

_DECLARED_STEM = "hangzhou_1x1_bc-tyc_18041610_1h"
DECLARED_SCENARIO_DIR = REPO_ROOT / "scenarios" / _DECLARED_STEM
DECLARED_SOURCE_ROU = DECLARED_SCENARIO_DIR / f"{_DECLARED_STEM}.rou.xml"
DECLARED_SOURCE_NET = DECLARED_SCENARIO_DIR / f"{_DECLARED_STEM}.net.xml"
DECLARED_SOURCE_FLOW_JSON = DECLARED_SCENARIO_DIR / "flow.json"

DECLARED_PARITY_STEM = f"{_DECLARED_STEM}_parity"
DECLARED_PARITY_DIR = REPO_ROOT / "scenarios" / DECLARED_PARITY_STEM
DECLARED_PARITY_ROU = DECLARED_PARITY_DIR / f"{DECLARED_PARITY_STEM}.rou.xml"
DECLARED_PARITY_SUMOCFG = DECLARED_PARITY_DIR / f"{DECLARED_PARITY_STEM}.sumocfg"


@dataclass(frozen=True)
class ParityAttribute:
    """One SUMO ``<vType>`` attribute, with the source of its value named.

    ``value`` is the string emitted into the XML, so the rendering is exact and does
    not depend on float formatting.  ``source_key`` is the CityFlow ``flow.json``
    ``vehicle`` key the value is read from, or ``None`` for an attribute fixed by the
    parity ruling rather than by the scenario.
    """

    name: str
    value: str
    source_key: str | None
    source: str


@dataclass(frozen=True)
class BindingReport:
    """What a route file actually declares and binds -- the mechanical check's input."""

    path: str
    vehicle_count: int
    vehicles_with_type: int
    distinct_type_values: tuple[str, ...]
    vtype_ids: tuple[str, ...]
    vtype_attributes: dict[str, str]


#: The parity type.  Every value with a ``source_key`` is verified against the
#: scenario's own ``flow.json`` by :func:`flow_json_disagreements`, so the committed
#: table cannot silently drift from the file it claims to come from.
PARITY_VTYPE: tuple[ParityAttribute, ...] = (
    ParityAttribute("id", PARITY_VTYPE_ID, None, "P7.0: a distinct id, never the shipped 'pkw'"),
    ParityAttribute("accel", "2.0", "maxPosAcc", "hangzhou flow.json vehicle.maxPosAcc"),
    ParityAttribute("decel", "4.5", "maxNegAcc", "hangzhou flow.json vehicle.maxNegAcc"),
    ParityAttribute("length", "5.0", "length", "hangzhou flow.json vehicle.length"),
    ParityAttribute("maxSpeed", "11.11", "maxSpeed", "hangzhou flow.json vehicle.maxSpeed"),
    ParityAttribute("minGap", "2.5", "minGap", "hangzhou flow.json vehicle.minGap"),
    ParityAttribute(
        "tau",
        "2.0",
        "headwayTime",
        "hangzhou flow.json vehicle.headwayTime -- the parameter the shipped 'pkw' "
        "omits, measured at +49 % travel time in docs/notes/P7.0_vtype_investigation.md",
    ),
    ParityAttribute("width", "2.0", "width", "hangzhou flow.json vehicle.width"),
    ParityAttribute(
        "speedFactor",
        "1.0",
        None,
        "BRIEF_04 section 3 ruling: a sampling distribution over an already-matched "
        "parameter, and CityFlow has exactly one vehicle parameter set",
    ),
)

#: Declared unmatchable in both directions, and therefore emitted nowhere.  Declaring
#: them is the point: a paper that says which parameters were matched and which have
#: no analogue does not get the gap found for it by a reviewer.
UNMATCHABLE_PARAMETERS: tuple[tuple[str, str], ...] = (
    (
        "sigma",
        "SUMO driver imperfection: a car-following MODEL parameter with no CityFlow "
        "counterpart. Zeroing it would replace SUMO's model rather than align it.",
    ),
    (
        "speedFactorDistribution",
        "SUMO samples speedFactor from a distribution; CityFlow cannot represent a "
        "spread, so the distribution is collapsed to the point value 1.0.",
    ),
    (
        "usualPosAcc",
        "CityFlow comfortable acceleration: no SUMO counterpart.",
    ),
    (
        "usualNegAcc",
        "CityFlow comfortable deceleration: no SUMO counterpart.",
    ),
)

_VTYPE_RE = re.compile(r"<vType\b[^>]*?/?>")
_VEHICLE_OPEN_RE = re.compile(r"<vehicle\b([^>]*?)(/?)>")
_TYPE_ATTR_RE = re.compile(r"\btype\s*=")


def parity_vtype_attributes() -> dict[str, str]:
    """The parity ``<vType>`` as an attribute mapping, ``id`` included."""
    return {attr.name: attr.value for attr in PARITY_VTYPE}


def read_cityflow_vehicle_block(flow_json_path: str | Path) -> dict[str, float]:
    """The single distinct ``vehicle`` parameter block of a CityFlow ``flow.json``.

    Raises when the file carries more than one, because a single parity ``<vType>``
    cannot then represent the demand and silently picking the first would be the
    quietest possible way to get the gate wrong.
    """
    raw = json.loads(Path(flow_json_path).read_bytes())
    blocks = {json.dumps(entry["vehicle"], sort_keys=True) for entry in raw}
    if len(blocks) != 1:
        raise ValueError(
            f"{flow_json_path} carries {len(blocks)} distinct vehicle parameter "
            "blocks; a single parity <vType> cannot represent them"
        )
    return {str(k): float(v) for k, v in json.loads(blocks.pop()).items()}


def flow_json_disagreements(flow_json_path: str | Path) -> list[str]:
    """Attributes where :data:`PARITY_VTYPE` disagrees with its named source.

    Compared with ``==`` on floats rather than a tolerance: both sides parse the same
    decimal literal to the same double, so exact equality holds, and loosening it
    would hide exactly the drift this function exists to catch.
    """
    block = read_cityflow_vehicle_block(flow_json_path)
    out: list[str] = []
    for attr in PARITY_VTYPE:
        if attr.source_key is None:
            continue
        if attr.source_key not in block:
            out.append(
                f"{attr.name}: declared source key {attr.source_key!r} is absent from "
                f"{flow_json_path}"
            )
            continue
        found = block[attr.source_key]
        if float(attr.value) != found:
            out.append(
                f"{attr.name}: declared {attr.value} disagrees with its declared "
                f"source {attr.source_key}={found}"
            )
    return out


def render_parity_vtype_element() -> str:
    """The parity ``<vType>`` as a self-closing XML element, attributes sorted."""
    attrs = parity_vtype_attributes()
    rendered = " ".join(f'{name}="{attrs[name]}"' for name in sorted(attrs))
    return f"<vType {rendered}/>"


def render_parity_rou_text(source_text: str) -> str:
    """Bind the parity type onto every vehicle of a SUMO route file.

    Textual, so everything not named here stays byte-identical; then parsed, so the
    text edit cannot quietly corrupt the demand.  Raises rather than returning a
    partially-bound file: a route file that binds *most* vehicles is the failure this
    task exists to prevent, wearing a passing check.
    """
    vtype_matches = _VTYPE_RE.findall(source_text)
    if len(vtype_matches) != 1:
        raise ValueError(
            "the parity renderer needs exactly one <vType> declaration to replace, "
            f"found {len(vtype_matches)}"
        )

    vehicle_matches = list(_VEHICLE_OPEN_RE.finditer(source_text))
    if not vehicle_matches:
        raise ValueError("the source route file declares no <vehicle> elements")
    already = [m for m in vehicle_matches if _TYPE_ATTR_RE.search(m.group(1))]
    if already:
        raise ValueError(
            f"{len(already)} of {len(vehicle_matches)} vehicles already carries a "
            "type= binding; refusing to rebind a file that is not the shipped source"
        )

    rendered = _VTYPE_RE.sub(lambda _m: render_parity_vtype_element(), source_text, count=1)
    rendered = _VEHICLE_OPEN_RE.sub(
        lambda m: f'<vehicle{m.group(1)} type="{PARITY_VTYPE_ID}"{m.group(2)}>',
        rendered,
    )

    _verify_rendered_rou(source_text, rendered, len(vehicle_matches))
    return rendered


def _vehicle_identity(text: str) -> list[tuple[str, str, str]]:
    root = ET.fromstring(text)
    out: list[tuple[str, str, str]] = []
    for veh in root.findall("vehicle"):
        route = veh.find("route")
        edges = "" if route is None else str(route.get("edges"))
        out.append((str(veh.get("id")), str(veh.get("depart")), edges))
    return out


def _verify_rendered_rou(source_text: str, rendered: str, expected_vehicles: int) -> None:
    """Parse the rendered text and prove it says what the transformation intended."""
    root = ET.fromstring(rendered)
    vehicles = root.findall("vehicle")
    if len(vehicles) != expected_vehicles:
        raise ValueError(
            f"the rendered route file holds {len(vehicles)} vehicles against the "
            f"source's {expected_vehicles}"
        )
    unbound = [v.get("id") for v in vehicles if v.get("type") != PARITY_VTYPE_ID]
    if unbound:
        raise ValueError(
            f"{len(unbound)} rendered vehicles are not bound to {PARITY_VTYPE_ID!r}; "
            f"first: {unbound[0]}"
        )
    vtypes = root.findall("vType")
    if len(vtypes) != 1 or dict(vtypes[0].attrib) != parity_vtype_attributes():
        raise ValueError(
            "the rendered route file does not declare exactly the parity <vType>"
        )
    if _vehicle_identity(source_text) != _vehicle_identity(rendered):
        raise ValueError(
            "the rendered route file changed a vehicle id, departure or route; the "
            "demand must survive the binding element for element"
        )


def render_parity_sumocfg_text(
    net_file_reference: str,
    route_file_reference: str,
    end_seconds: int = SUMO_END_SECONDS,
) -> str:
    """A SUMO configuration for the parity route file.

    ``net_file_reference`` is emitted verbatim, so the caller decides whether the
    network is referenced relatively (which is what keeps there being one copy of it).
    """
    if int(end_seconds) <= ENV_HORIZON_SECONDS:
        raise ValueError(
            f"end={end_seconds} does not exceed the env horizon of "
            f"{ENV_HORIZON_SECONDS} s, so SUMO would truncate the episode while "
            "CityFlow ran on -- a backend-asymmetric truncation inside the "
            "comparison P7.0 makes"
        )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<!-- Generated by offline/parity.py (parity contract v"
        f"{PARITY_CONTRACT_VERSION}). Do not edit by hand.\n"
        "     The network is the SHIPPED one, referenced rather than copied.\n"
        f"     end={end_seconds} exceeds the env horizon of {ENV_HORIZON_SECONDS} s so\n"
        "     the env, not this file, owns the episode length on both backends. -->\n"
        "<configuration>\n"
        "\t<input>\n"
        f'\t\t<net-file value="{net_file_reference}"/>\n'
        f'\t\t<route-files value="{route_file_reference}"/>\n'
        "\t</input>\n"
        "\t<time>\n"
        '\t\t<begin value="0"/>\n'
        f'\t\t<end value="{int(end_seconds)}"/>\n'
        "\t</time>\n"
        "</configuration>\n"
    )


def vtype_binding_report(rou_path: str | Path) -> BindingReport:
    """What ``rou_path`` declares and binds. Describes; never judges."""
    root = ET.parse(rou_path).getroot()
    vehicles = root.findall("vehicle")
    typed = [str(v.get("type")) for v in vehicles if v.get("type") is not None]
    vtypes = root.findall("vType")
    attributes: dict[str, str] = (
        {str(k): str(v) for k, v in vtypes[0].attrib.items()} if len(vtypes) == 1 else {}
    )
    return BindingReport(
        path=str(rou_path),
        vehicle_count=len(vehicles),
        vehicles_with_type=len(typed),
        distinct_type_values=tuple(sorted(set(typed))),
        vtype_ids=tuple(str(t.get("id")) for t in vtypes),
        vtype_attributes=attributes,
    )


def binding_is_complete(report: BindingReport) -> bool:
    """`BRIEF_21` section 2's acceptance, strengthened by the id and the attributes.

    The brief's criterion is ``vehicles carrying type= == vehicles``.  That alone
    would accept a file binding every vehicle to a type whose ``tau`` is wrong, which
    is the confound this whole precondition exists to remove -- so the parity id and
    the full attribute set are checked too.
    """
    return (
        report.vehicle_count > 0
        and report.vehicles_with_type == report.vehicle_count
        and report.distinct_type_values == (PARITY_VTYPE_ID,)
        and report.vtype_ids == (PARITY_VTYPE_ID,)
        and report.vtype_attributes == parity_vtype_attributes()
    )


def write_parity_scenario(
    source_rou: str | Path,
    source_net: str | Path,
    source_flow_json: str | Path,
    dest_dir: str | Path,
    dest_stem: str,
) -> dict[str, str]:
    """Write the bound route file and its configuration into ``dest_dir``.

    Filesystem-mutation barrier: every validation -- the flow.json agreement, the
    single ``<vType>``, the unbound source, the parsed re-verification of the rendered
    text -- completes **before** a directory is created or a byte is written.  A
    failed construction leaves no directory behind.
    """
    dest_dir = Path(dest_dir)
    disagreements = flow_json_disagreements(source_flow_json)
    if disagreements:
        raise ValueError(
            "the declared parity table disagrees with its declared source "
            f"{source_flow_json}: " + "; ".join(disagreements)
        )

    rou_text = render_parity_rou_text(Path(source_rou).read_text(encoding="utf-8"))

    net_reference = os.path.relpath(Path(source_net).resolve(), dest_dir.resolve())
    route_reference = f"{dest_stem}.rou.xml"
    cfg_text = render_parity_sumocfg_text(net_reference, route_reference)

    rou_target = dest_dir / route_reference
    cfg_target = dest_dir / f"{dest_stem}.sumocfg"

    # --- every write happens below this line, and only below it ---
    dest_dir.mkdir(parents=True, exist_ok=True)
    rou_target.write_text(rou_text, encoding="utf-8")
    cfg_target.write_text(cfg_text, encoding="utf-8")
    return {"rou": str(rou_target), "sumocfg": str(cfg_target)}


def main(argv: Sequence[str] | None = None) -> int:
    """``--check`` verifies the committed artifacts; ``--write`` regenerates them."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.parity",
        description=(
            "Bind the CityFlow-parity vehicle type onto the declared hangzhou "
            "scenario's SUMO route file (P7.0)."
        ),
    )
    parser.add_argument("--write", action="store_true", help="regenerate the artifacts")
    args = parser.parse_args(argv)

    disagreements = flow_json_disagreements(DECLARED_SOURCE_FLOW_JSON)
    print(f"parity contract v{PARITY_CONTRACT_VERSION}; flow.json disagreements: {len(disagreements)}")
    for line in disagreements:
        print(f"  {line}")
    if disagreements:
        return 1

    if args.write:
        written = write_parity_scenario(
            source_rou=DECLARED_SOURCE_ROU,
            source_net=DECLARED_SOURCE_NET,
            source_flow_json=DECLARED_SOURCE_FLOW_JSON,
            dest_dir=DECLARED_PARITY_DIR,
            dest_stem=DECLARED_PARITY_STEM,
        )
        for key, path in sorted(written.items()):
            print(f"wrote {key}: {path}")

    shipped = vtype_binding_report(DECLARED_SOURCE_ROU)
    print(
        f"shipped  {Path(shipped.path).name}: vehicles={shipped.vehicle_count} "
        f"with type={shipped.vehicles_with_type} vTypes={list(shipped.vtype_ids)} "
        f"bound={binding_is_complete(shipped)}"
    )
    if not DECLARED_PARITY_ROU.is_file():
        print("parity route file is absent; run with --write")
        return 1
    bound = vtype_binding_report(DECLARED_PARITY_ROU)
    print(
        f"parity   {Path(bound.path).name}: vehicles={bound.vehicle_count} "
        f"with type={bound.vehicles_with_type} vTypes={list(bound.vtype_ids)} "
        f"bound={binding_is_complete(bound)}"
    )
    return 0 if binding_is_complete(bound) and not binding_is_complete(shipped) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
