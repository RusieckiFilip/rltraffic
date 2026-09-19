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

Parity is PER SCENARIO (P7.3d, ``BRIEF_39`` C1)
-----------------------------------------------
The contract version is still **v1.0** and the two on-disk formats above are unchanged; what
P7.3d adds is a second registered scenario.  A :class:`ParityScenario` names the draws-tree key,
the CityFlow ``flow.json`` the values come from, and the REGISTERED table: hangzhou's
(:data:`PARITY_VTYPE`: ``maxSpeed 11.11, tau 2.0, accel 2.0, width 2.0``) and grid4x4's, which is
``PREREGISTRATION`` A15(g)'s (``maxSpeed 13.39, tau 1.5, accel 2.6, decel 4.5, length 5.0,
minGap 2.5, width 1.8, speedFactor 1.0``).  The emitted values are DERIVED from the flow block
(:func:`derived_vtype_attributes`) and refused when they disagree with the registration.
**The hangzhou scenario is the default of every function that existed before P7.3d, and its
output is byte-identical** -- which is the only reason the contract version did not move.

**Ordering convention, unchanged:** vehicle order, ids, departure times and routes survive the
binding element for element, in either binding mode.  Hangzhou's route template declares exactly
one ``<vType>``, which is REPLACED.  RESCO's grid4x4 template declares NONE, so the parity type is
INSERTED as the first child of ``<routes>`` (``insert_when_absent``); each mode refuses a source
that does not match it.  RESCO's files are CC BY-NC-SA 4.0: they are located through
``RLTRAFFIC_GRID4X4_RESCO`` (no default), read in place -- the route member with :mod:`zipfile` --
verified against the digests A15(g) recorded, and never copied into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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
    "NOTELEPORT_SUMOCFG",
    "PARITY_VTYPE_ID",
    "PARITY_VTYPE",
    "GRID4X4_RESCO_ENV",
    "GRID4X4_PARITY_VTYPE",
    "GRID4X4_NETWORK_NOTE",
    "HZ1X1_SCENARIO",
    "GRID4X4_SCENARIO",
    "PARITY_SCENARIOS",
    "ExternalSumoSource",
    "ResolvedExternalSource",
    "ParityScenario",
    "scenario_for_key",
    "derived_vtype_attributes",
    "resolve_external_source",
    "read_route_template_text",
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

#: The hand-written teleport-free configuration P7.1 committed (``18ed5cb``), and the
#: reference any generated teleport-free config is compared against.  A15(c) makes the
#: regime binding on every SUMO measurement recorded after its commit, so this path is
#: exported rather than re-derived at each call site.
NOTELEPORT_SUMOCFG = DECLARED_PARITY_DIR / f"{DECLARED_PARITY_STEM}_noteleport.sumocfg"


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
_ROUTES_OPEN_RE = re.compile(r"<routes\b[^>]*>")
_FIRST_VEHICLE_INDENT_RE = re.compile(r"\n([ \t]*)<vehicle\b")


# ----------------------------------------------------------------------
# Parity PER SCENARIO (P7.3d, ``BRIEF_39`` C1 + Amendment A)
# ----------------------------------------------------------------------

#: The variable naming the read-only candidates ROOT that holds RESCO's grid4x4 files.  It has
#: **no default, by design** (``docs/returns/P7.1.md`` Minors): unset means the grid4x4 scenario is
#: unresolvable -- tests skip naming it, entry points refuse naming it -- and never ``/home/...``.
GRID4X4_RESCO_ENV = "RLTRAFFIC_GRID4X4_RESCO"


@dataclass(frozen=True)
class ExternalSumoSource:
    """A scenario's SUMO side that lives OUTSIDE the repository, pinned by digest.

    RESCO's grid4x4 files are CC BY-NC-SA 4.0 and stay in the gitignored candidates tree
    (``PREREGISTRATION`` A15(g)); they are located through ``env_var`` and **read in place** -- the
    route file with :mod:`zipfile`, never extracted into any tree.  Both digests are the ones A15(g)'s
    audit recorded, so a different file under the same name is a refusal, not a different scenario.
    """

    env_var: str
    relative_dir: str
    net_name: str
    net_sha256: str
    sumocfg_name: str
    routes_archive: str
    routes_member: str
    routes_member_sha256: str
    licence: str


@dataclass(frozen=True)
class ResolvedExternalSource:
    """An :class:`ExternalSumoSource` located on this machine, with both digests VERIFIED."""

    root: Path
    net: Path
    sumocfg: Path
    routes_archive: Path
    routes_member: str
    net_sha256: str
    routes_member_sha256: str


@dataclass(frozen=True)
class ParityScenario:
    """One scenario the parity contract applies to.

    ``key`` is the sim-config stem, which is also the scenario directory of the draws tree
    (``offline/materialise_draws.py``: ``configs/sim/cityflow1x1.json`` -> ``cityflow1x1``), so no
    second registry exists.  ``declared_vtype`` is the REGISTERED table; the emitted values are
    DERIVED from the CityFlow flow block by :func:`derived_vtype_attributes` and any disagreement
    between the two is a refusal.  ``template_declares_vtype`` says what the scenario's SUMO route
    template looks like and therefore what the binding does: hangzhou's declares exactly one
    ``<vType>`` (REPLACED); RESCO's grid4x4 declares none (the parity type is INSERTED).
    """

    key: str
    stem: str
    flow_json: Path
    declared_vtype: tuple[ParityAttribute, ...]
    template_declares_vtype: bool
    external: ExternalSumoSource | None = None


#: A15(g)'s registered parity values for grid4x4, each with the flow-block key it is read from.
GRID4X4_PARITY_VTYPE: tuple[ParityAttribute, ...] = (
    ParityAttribute("id", PARITY_VTYPE_ID, None, "P7.0: a distinct id, never a shipped type"),
    ParityAttribute("accel", "2.6", "maxPosAcc", "A15(g): grid4x4 flow.json vehicle.maxPosAcc"),
    ParityAttribute("decel", "4.5", "maxNegAcc", "A15(g): grid4x4 flow.json vehicle.maxNegAcc"),
    ParityAttribute("length", "5.0", "length", "A15(g): grid4x4 flow.json vehicle.length"),
    ParityAttribute("maxSpeed", "13.39", "maxSpeed", "A15(g): grid4x4 flow.json vehicle.maxSpeed"),
    ParityAttribute("minGap", "2.5", "minGap", "A15(g): grid4x4 flow.json vehicle.minGap"),
    ParityAttribute("tau", "1.5", "headwayTime", "A15(g): grid4x4 flow.json vehicle.headwayTime"),
    ParityAttribute("width", "1.8", "width", "A15(g): grid4x4 flow.json vehicle.width"),
    ParityAttribute(
        "speedFactor",
        "1.0",
        None,
        "BRIEF_04 section 3 ruling, unchanged per scenario: CityFlow has one vehicle parameter set",
    ),
)

HZ1X1_SCENARIO = ParityScenario(
    key="cityflow1x1",
    stem=_DECLARED_STEM,
    flow_json=DECLARED_SOURCE_FLOW_JSON,
    declared_vtype=PARITY_VTYPE,
    template_declares_vtype=True,
)

GRID4X4_SCENARIO = ParityScenario(
    key="cityflow_grid4x4",
    stem="grid4x4",
    flow_json=REPO_ROOT / "scenarios" / "grid4x4" / "grid4x4_flow.json",
    declared_vtype=GRID4X4_PARITY_VTYPE,
    template_declares_vtype=False,
    external=ExternalSumoSource(
        env_var=GRID4X4_RESCO_ENV,
        relative_dir="resco/resco_benchmark/environments/grid4x4",
        net_name="grid4x4.net.xml",
        net_sha256="8d192de462497c192ed51f0a147b7522626e706b411ceba6f16b083b13183b38",
        sumocfg_name="grid4x4.sumocfg",
        routes_archive="grid4x4.zip",
        routes_member="grid4x4_1.rou.xml",
        routes_member_sha256="2350dce7e086b8dd27b94c772a4ff883c04c8232c1248c4b889d7c466f4127cc",
        licence="CC BY-NC-SA 4.0 (RESCO grid4x4 data; cited by URL and sha256, never redistributed)",
    ),
)

#: The header sentence of a generated ``.sumocfg`` for a scenario whose network is external.
GRID4X4_NETWORK_NOTE = (
    "The network is RESCO's grid4x4.net.xml (CC BY-NC-SA 4.0), OUTSIDE the repository: "
    "referenced by relative path, never copied; its sha256 is in provenance.json."
)

#: Every scenario the contract is registered for, by draws-tree key.  The hangzhou entry is the
#: default of every pre-existing entry point, so every call written before P7.3d is unchanged.
PARITY_SCENARIOS: dict[str, ParityScenario] = {
    HZ1X1_SCENARIO.key: HZ1X1_SCENARIO,
    GRID4X4_SCENARIO.key: GRID4X4_SCENARIO,
}


def scenario_for_key(key: str) -> ParityScenario:
    """The registered :class:`ParityScenario` for a draws-tree key, or a refusal naming the key."""
    if key not in PARITY_SCENARIOS:
        raise ValueError(
            f"no parity scenario is registered for {key!r}; the contract is registered for "
            f"{sorted(PARITY_SCENARIOS)}. Parity is per scenario (the values come from each "
            "scenario's own flow block), so an unregistered one has no table to be checked against"
        )
    return PARITY_SCENARIOS[key]


def derived_vtype_attributes(
    scenario: ParityScenario, flow_json: str | Path | None = None
) -> dict[str, str]:
    """The parity ``<vType>`` for *scenario*, DERIVED from a CityFlow flow block.

    Every attribute with a ``source_key`` is READ from the block (``flow_json``, or the scenario's
    own source flow when omitted) and emitted as ``repr(float(value))``; ``id`` and ``speedFactor``
    come from the parity ruling.  The registered table is the CONTRACT the derivation is checked
    against, under float ``==``: any disagreement raises, naming the attribute -- a scenario whose
    flow block disagrees with its registration is a refusal, not a warning.

    For hangzhou the derived strings are byte-equal to the table P7.0 committed (pinned by
    ``tests/test_parity_scenarios.py``), which is why ``PARITY_CONTRACT_VERSION`` is still 1.0.
    """
    path = scenario.flow_json if flow_json is None else Path(flow_json)
    disagreements = flow_json_disagreements(path, scenario)
    if disagreements:
        raise ValueError(
            f"{scenario.key}: the flow block of {path} disagrees with the registered parity "
            "table: " + "; ".join(disagreements)
        )
    block = read_cityflow_vehicle_block(path)
    return {
        attr.name: attr.value if attr.source_key is None else repr(float(block[attr.source_key]))
        for attr in scenario.declared_vtype
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_external_source(
    scenario: ParityScenario, *, environ: Mapping[str, str] | None = None
) -> ResolvedExternalSource:
    """Locate *scenario*'s external SUMO side and verify both pinned digests.

    Read-only.  ``environ`` defaults to ``os.environ``; the variable has **no default value**, so an
    unset one raises naming it -- a path guessed from one contributor's home directory would pass
    on one machine and mean nothing on any other.  The route member is hashed from inside the
    archive, in place.
    """
    external = scenario.external
    if external is None:
        raise ValueError(
            f"{scenario.key} has no external SUMO source: its network and route template are "
            "shipped in the repository and are resolved from the scenario's own .sumocfg"
        )
    env = os.environ if environ is None else environ
    raw = env.get(external.env_var)
    if not raw:
        raise ValueError(
            f"{external.env_var} is unset, so {scenario.key}'s SUMO side cannot be located. Point "
            "it at the read-only candidates ROOT (the directory holding resco/). It has no "
            f"default by design; the files are {external.licence} and stay outside the repository"
        )
    root = Path(raw).expanduser().resolve()
    base = root / external.relative_dir
    net = base / external.net_name
    sumocfg = base / external.sumocfg_name
    archive = base / external.routes_archive
    missing = [str(path) for path in (net, sumocfg, archive) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{external.env_var}={raw} does not hold {scenario.key}'s SUMO side: missing {missing}"
        )

    net_sha256 = _sha256_bytes(net.read_bytes())
    if net_sha256 != external.net_sha256:
        raise ValueError(
            f"{net} hashes to {net_sha256}, not the registered {external.net_sha256}; a different "
            "network under the same name is a different scenario, and A15(g)'s audit does not "
            "cover it"
        )
    with zipfile.ZipFile(archive) as handle:
        if external.routes_member not in handle.namelist():
            raise FileNotFoundError(
                f"{archive} has no member {external.routes_member!r} ({external.env_var}={raw})"
            )
        member_sha256 = _sha256_bytes(handle.read(external.routes_member))
    if member_sha256 != external.routes_member_sha256:
        raise ValueError(
            f"{archive}::{external.routes_member} hashes to {member_sha256}, not the registered "
            f"{external.routes_member_sha256}; the route template is not the file A15(g) audited"
        )
    return ResolvedExternalSource(
        root=root,
        net=net,
        sumocfg=sumocfg,
        routes_archive=archive,
        routes_member=external.routes_member,
        net_sha256=net_sha256,
        routes_member_sha256=member_sha256,
    )


def read_route_template_text(resolved: ResolvedExternalSource) -> str:
    """The route template's text, read IN PLACE from the archive (never extracted into a tree)."""
    with zipfile.ZipFile(resolved.routes_archive) as handle:
        data = handle.read(resolved.routes_member)
    if _sha256_bytes(data) != resolved.routes_member_sha256:
        raise ValueError(
            f"{resolved.routes_archive}::{resolved.routes_member} changed between resolution and "
            "reading; refusing to use a template whose digest is no longer the verified one"
        )
    return data.decode("utf-8")


def parity_vtype_attributes(scenario: ParityScenario | None = None) -> dict[str, str]:
    """The REGISTERED parity ``<vType>`` as an attribute mapping, ``id`` included.

    ``None`` is the hangzhou scenario, which is what this function returned before it took an
    argument.  This is the DECLARED table; what a rendering emits is
    :func:`derived_vtype_attributes`, and the two are refused if they disagree.
    """
    declared = PARITY_VTYPE if scenario is None else scenario.declared_vtype
    return {attr.name: attr.value for attr in declared}


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


def flow_json_disagreements(
    flow_json_path: str | Path, scenario: ParityScenario | None = None
) -> list[str]:
    """Attributes where a scenario's REGISTERED table disagrees with its named source.

    ``scenario=None`` is the hangzhou scenario (:data:`PARITY_VTYPE`), which is what this function
    checked before it took a second argument.

    Compared with ``==`` on floats rather than a tolerance: both sides parse the same
    decimal literal to the same double, so exact equality holds, and loosening it
    would hide exactly the drift this function exists to catch.
    """
    declared = PARITY_VTYPE if scenario is None else scenario.declared_vtype
    block = read_cityflow_vehicle_block(flow_json_path)
    out: list[str] = []
    for attr in declared:
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


def render_parity_vtype_element(vtype: Mapping[str, str] | None = None) -> str:
    """The parity ``<vType>`` as a self-closing XML element, attributes sorted.

    ``vtype=None`` renders the hangzhou table, byte-identical to what P7.0 committed.
    """
    attrs = parity_vtype_attributes() if vtype is None else dict(vtype)
    rendered = " ".join(f'{name}="{attrs[name]}"' for name in sorted(attrs))
    return f"<vType {rendered}/>"


def render_parity_rou_text(
    source_text: str,
    vtype: Mapping[str, str] | None = None,
    *,
    insert_when_absent: bool = False,
) -> str:
    """Bind the parity type onto every vehicle of a SUMO route file.

    Textual, so everything not named here stays byte-identical; then parsed, so the
    text edit cannot quietly corrupt the demand.  Raises rather than returning a
    partially-bound file: a route file that binds *most* vehicles is the failure this
    task exists to prevent, wearing a passing check.

    ``vtype=None`` binds the hangzhou table, exactly as before this function took the argument.

    Two binding modes, selected by the caller from the SCENARIO and never guessed from the text,
    each strict: by default the source must declare exactly ONE ``<vType>``, which is REPLACED
    (hangzhou's template declares ``pkw``); with ``insert_when_absent`` it must declare exactly
    NONE and the parity type is INSERTED as the first child of ``<routes>``, at the indentation of
    the first ``<vehicle>`` (RESCO's grid4x4 template declares no type at all).  A source that does
    not match its mode is not the file the scenario's audit described, and is refused.
    """
    attrs = parity_vtype_attributes() if vtype is None else dict(vtype)
    vtype_id = attrs["id"]
    element = render_parity_vtype_element(attrs)

    vtype_matches = _VTYPE_RE.findall(source_text)
    if insert_when_absent:
        if vtype_matches:
            raise ValueError(
                "the parity renderer was told this scenario's route template declares no <vType> "
                f"(the parity type is INSERTED), but the source declares {len(vtype_matches)}; "
                "refusing: the source is not the file this scenario's audit described"
            )
    elif len(vtype_matches) != 1:
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

    if insert_when_absent:
        opens = list(_ROUTES_OPEN_RE.finditer(source_text))
        if len(opens) != 1:
            raise ValueError(
                f"the source declares {len(opens)} <routes> open tags; the parity <vType> is "
                "inserted as the first child of exactly one"
            )
        indent_match = _FIRST_VEHICLE_INDENT_RE.search(source_text)
        indent = "" if indent_match is None else indent_match.group(1)
        cut = opens[0].end()
        rendered = source_text[:cut] + "\n" + indent + element + source_text[cut:]
    else:
        rendered = _VTYPE_RE.sub(lambda _m: element, source_text, count=1)
    rendered = _VEHICLE_OPEN_RE.sub(
        lambda m: f'<vehicle{m.group(1)} type="{vtype_id}"{m.group(2)}>',
        rendered,
    )

    _verify_rendered_rou(source_text, rendered, len(vehicle_matches), attrs)
    return rendered


def _vehicle_identity(text: str) -> list[tuple[str, str, str]]:
    root = ET.fromstring(text)
    out: list[tuple[str, str, str]] = []
    for veh in root.findall("vehicle"):
        route = veh.find("route")
        edges = "" if route is None else str(route.get("edges"))
        out.append((str(veh.get("id")), str(veh.get("depart")), edges))
    return out


def _verify_rendered_rou(
    source_text: str,
    rendered: str,
    expected_vehicles: int,
    vtype: Mapping[str, str] | None = None,
) -> None:
    """Parse the rendered text and prove it says what the transformation intended."""
    attrs = parity_vtype_attributes() if vtype is None else dict(vtype)
    vtype_id = attrs["id"]
    root = ET.fromstring(rendered)
    vehicles = root.findall("vehicle")
    if len(vehicles) != expected_vehicles:
        raise ValueError(
            f"the rendered route file holds {len(vehicles)} vehicles against the "
            f"source's {expected_vehicles}"
        )
    unbound = [v.get("id") for v in vehicles if v.get("type") != vtype_id]
    if unbound:
        raise ValueError(
            f"{len(unbound)} rendered vehicles are not bound to {vtype_id!r}; "
            f"first: {unbound[0]}"
        )
    vtypes = root.findall("vType")
    if len(vtypes) != 1 or dict(vtypes[0].attrib) != attrs:
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
    *,
    time_to_teleport: int | None = None,
    network_note: str | None = None,
) -> str:
    """A SUMO configuration for the parity route file.

    ``net_file_reference`` is emitted verbatim, so the caller decides whether the
    network is referenced relatively (which is what keeps there being one copy of it).

    ``time_to_teleport`` is ``None`` by default and the output is then **byte-identical
    to what P7.0 committed** -- the committed artifacts are the control, so the default
    may never move.  An integer adds SUMO's ``<processing>`` block and one sentence to
    the header naming A15(c).  A **positive** value is refused: SUMO documents
    non-positive values as disabling teleporting, so a positive one re-enables the
    mechanism A15(c) froze off, and this project does not ship an argument that silently
    undoes a registration.

    ``network_note`` replaces the header's one sentence about the network.  ``None`` keeps the
    hangzhou sentence -- *the network is the SHIPPED one* -- byte for byte; a scenario whose
    network is NOT shipped (grid4x4: RESCO's, outside the repository) must say so, because this
    header is hashed into every later cell's ``config_sha256`` and a false sentence there cannot be
    corrected afterwards without re-rolling the cells.  One line, and no double hyphen: XML forbids
    one inside a comment, and SUMO would reject the file.
    """
    if network_note is not None:
        if "\n" in network_note or "\r" in network_note:
            raise ValueError("network_note must be one line; it replaces one header sentence")
        if "--" in network_note:
            raise ValueError(
                "network_note contains a double hyphen, which XML forbids inside a comment; the "
                "generated configuration would not parse"
            )
    if int(end_seconds) <= ENV_HORIZON_SECONDS:
        raise ValueError(
            f"end={end_seconds} does not exceed the env horizon of "
            f"{ENV_HORIZON_SECONDS} s, so SUMO would truncate the episode while "
            "CityFlow ran on -- a backend-asymmetric truncation inside the "
            "comparison P7.0 makes"
        )
    if time_to_teleport is not None and int(time_to_teleport) > 0:
        raise ValueError(
            f"time-to-teleport={time_to_teleport} is positive, which RE-ENABLES the "
            "teleporting A15(c) froze off (SUMO documents non-positive values as "
            "disabling it, and its own default of 300 s teleported 13 vehicles per "
            "MaxPressure episode on this scenario at P7.1 G1). This function does not "
            "ship an argument that silently undoes a registration; pass -1, or 0, or None"
        )

    header = [
        "<!-- Generated by offline/parity.py (parity contract v"
        f"{PARITY_CONTRACT_VERSION}). Do not edit by hand.",
        (
            "     The network is the SHIPPED one, referenced rather than copied."
            if network_note is None
            else f"     {network_note}"
        ),
        f"     end={end_seconds} exceeds the env horizon of {ENV_HORIZON_SECONDS} s so",
        "     the env, not this file, owns the episode length on both backends.",
    ]
    processing = ""
    if time_to_teleport is not None:
        header.append(
            "     Teleporting is DISABLED under A15(c): every SUMO measurement recorded after"
        )
        header.append(
            "     that amendment runs a configuration carrying <time-to-teleport>, because"
        )
        header.append(
            "     SUMO's default of 300 s teleports and CityFlow never does."
        )
        processing = (
            "\t<processing>\n"
            f'\t\t<time-to-teleport value="{int(time_to_teleport)}"/>\n'
            "\t</processing>\n"
        )

    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + "\n".join(header)
        + " -->\n"
        "<configuration>\n"
        "\t<input>\n"
        f'\t\t<net-file value="{net_file_reference}"/>\n'
        f'\t\t<route-files value="{route_file_reference}"/>\n'
        "\t</input>\n"
        "\t<time>\n"
        '\t\t<begin value="0"/>\n'
        f'\t\t<end value="{int(end_seconds)}"/>\n'
        "\t</time>\n"
        f"{processing}"
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


def binding_is_complete(report: BindingReport, vtype: Mapping[str, str] | None = None) -> bool:
    """`BRIEF_21` section 2's acceptance, strengthened by the id and the attributes.

    The brief's criterion is ``vehicles carrying type= == vehicles``.  That alone
    would accept a file binding every vehicle to a type whose ``tau`` is wrong, which
    is the confound this whole precondition exists to remove -- so the parity id and
    the full attribute set are checked too.

    ``vtype=None`` checks against the hangzhou table, as this function always did.
    """
    attrs = parity_vtype_attributes() if vtype is None else dict(vtype)
    vtype_id = attrs["id"]
    return (
        report.vehicle_count > 0
        and report.vehicles_with_type == report.vehicle_count
        and report.distinct_type_values == (vtype_id,)
        and report.vtype_ids == (vtype_id,)
        and report.vtype_attributes == attrs
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
