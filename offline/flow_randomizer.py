"""Seeded demand randomisation for CityFlow scenarios, renderable to SUMO.

WHY THIS MODULE EXISTS
----------------------
Measured on real CityFlow (2026-07-27): engine seeds 1000 and 1001 produced
byte-identical trajectories, identical ``episode_sha256`` and identical return
(-502.000).  The cause is structural rather than a seeding bug.  ``Engine`` reads the
flow file exactly once, in its constructor (``CityFlow/src/engine/engine.cpp:65``,
``loadFlow(dir + flowFile)``), and ``Engine::reset()`` (``:744-760``) only calls
``flow.reset()`` on the already-parsed in-memory ``flows`` vector -- it never re-reads
the file.  With ``laneChange: false`` and fully specified routes, the engine RNG has
nothing observable left to influence.  ``reset(seed=X)`` therefore cannot change vehicle
arrivals, and varying the *vehicle list* is the only lever available.

**Consequence for callers: a new draw requires a fresh env object, not a ``reset()``.**

SOURCE FORMAT (CityFlow ``flow.json``) -- verified against the repo 2026-07-28
-----------------------------------------------------------------------------
A flat JSON list of individual vehicle insertions.  Every entry has exactly the keys
``vehicle``, ``route``, ``interval``, ``startTime``, ``endTime``; ``vehicle`` is a block
of nine physical parameters.  Across all four flow files in this repo (hangzhou_bc-tyc
2021, cologne1 2011, cologne3 2800, grid4x4 1473 entries) ``endTime == startTime`` for
**0 mismatches out of 8305 entries**, so each entry is a single insertion and
``interval`` is inert.  ``startTime`` is already sorted ascending in all four.

Numeric types are **not** uniform across scenarios: ``interval`` is ``int`` in hangzhou
and ``float`` elsewhere, while ``startTime`` / ``endTime`` are ``int`` everywhere.  The
renderer preserves each source entry's own types, because a corpus is traced back to the
demand that produced it by file hash.

BYTE-IDENTITY INVARIANT
-----------------------
``render_cityflow(draw(0))`` is **byte-identical** to the source file -- draw 0 is the
nominal-flow control condition for every experiment, so it must not perturb the demand
even cosmetically.  Achieving that needs per-file formatting: hangzhou is serialised with
``indent=2``, cologne1 / cologne3 / grid4x4 with ``indent=4``, and none of them ends in a
newline.  The indent width and trailing-newline flag are therefore sniffed from the raw
bytes at construction, and ``__init__`` *verifies* the sniff by re-serialising the parsed
source and comparing against the original bytes.  A ``FlowRandomizer`` that cannot
reproduce its own source raises rather than existing, which turns the guarantee into a
construction-time invariant instead of something only a test happens to check.

TRANSFORMS
----------
All three are driven by one ``np.random.default_rng(base_seed + draw_id)``, so the same
``draw_id`` always yields the same draw, applied in this order:

1. **Bernoulli thinning** -- keep entry *i* iff ``rng.random(N)[i] >= thin_p``.
2. **Volume scaling** -- ``target = round(len(kept) * volume_scale)``; below, drop
   uniformly without replacement; above, sample the extras uniformly *with* replacement.
   Relative order is preserved either way.
3. **Departure jitter** -- every surviving entry, duplicates included and each drawing its
   own sample, gets ``startTime += N(0, sigma)`` clipped at ``>= 0`` and coerced back to
   the source entry's numeric type; ``endTime`` follows ``startTime``.
4. A stable sort by ``startTime`` closes the pipeline.

Jitter runs last deliberately: it is the only transform whose output needs re-sorting, so
one sort at the end suffices, and duplicates created by step 2 pick up independent jitter
for free -- which is what makes ``volume_scale > 1`` produce distinct vehicles rather than
exact copies.  Jitter is **not** clipped at the top of the horizon: a vehicle pushed past
the episode end simply never appears, which is honest, whereas clipping would pile
vehicles up on the boundary.  Expected surviving count is
``N * (1 - thin_p) * volume_scale`` -- thinning and ``volume_scale < 1`` compose
multiplicatively.

``draw_id == 0`` is the identity: an explicit early return before any RNG is constructed.
Its :attr:`FlowDraw.params` records the parameters *actually* applied (all-identity), so a
manifest describing a draw-0 corpus is self-describing rather than misleadingly quoting
the constructor's settings.

The source file is never opened for writing by any method here.

SUMO RENDERING
--------------
The CityFlow -> SUMO edge-id mapping is the **identity**, verified against the repo:
``flow.json`` route ids equal the ``.rou.xml`` ``<route edges>`` ids and are a subset of
the ``.net.xml`` normal (non-internal) edges -- hangzhou 8/8, cologne1 10/10, cologne3
48/48.  Nothing is invented or guessed.

``render_sumo`` copies the scenario's own ``<vType>`` block **verbatim** from a template
``.rou.xml``.  None of the three transforms touches vehicle physics -- they change only
which vehicles depart and when -- so the scenario's calibrated vehicle model is exactly
the right one, and no CityFlow-to-SUMO physics mapping has to be invented.  (Synthesising
a vType from the ``vehicle`` block would silently drop ``usualPosAcc`` / ``usualNegAcc``,
which have no SUMO counterpart.)

The two formats do **not** share a time base: ``depart = startTime + <begin>`` where
``<begin>`` comes from the scenario's ``.sumocfg`` (0 for hangzhou and grid4x4, 25200 for
cologne1 and cologne3; the relation holds exactly at both ends of both ranges).  That
offset is always parsed via :func:`sumo_begin_from_sumocfg` and never defaulted --
``depart_offset`` is keyword-only with no default, and the parser raises on a missing or
unparseable ``<begin>``, because a hardcoded 25200 would silently break every future
scenario.

grid4x4 is CityFlow-only in practice: its ``.sumocfg`` references a ``grid4x4.rou.xml``
that does not exist, and it ships no ``.net.xml``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "DEFAULT_BASE_SEED",
    "DEFAULT_JITTER_SIGMA_S",
    "DEFAULT_THIN_P",
    "DEFAULT_VOLUME_SCALE",
    "FlowDraw",
    "FlowRandomizer",
    "sumo_begin_from_sumocfg",
]

# Chosen to produce a visible but not pathological change: on hangzhou's 2021 vehicles
# this removes a *different* ~202 vehicles per draw and reshuffles every departure by
# N(0, 30 s), i.e. a clearly different demand pattern at roughly -10 % volume.
DEFAULT_JITTER_SIGMA_S = 30.0
DEFAULT_THIN_P = 0.10
DEFAULT_VOLUME_SCALE = 1.0
DEFAULT_BASE_SEED = 1000

#: Entry keys every CityFlow flow record carries (verified on all four repo scenarios).
_ENTRY_KEYS = frozenset({"vehicle", "route", "interval", "startTime", "endTime"})


def _coerce_like(value: float, template: Any) -> int | float:
    """Return *value* as the same numeric type *template* has.

    Numeric types are not uniform across scenarios -- ``interval`` is ``int`` in hangzhou
    and ``float`` elsewhere, ``startTime`` is ``int`` everywhere -- and a corpus is traced
    back to its demand by file hash, so a jittered ``5`` must not be written back as
    ``5.0``.  ``bool`` is excluded explicitly because it is a subclass of ``int``.
    """
    if isinstance(template, bool) or not isinstance(template, int):
        return float(value)
    return int(round(value))


@dataclass(frozen=True)
class FlowDraw:
    """Provenance for one draw.

    ``source_sha256`` pins the exact demand file a corpus came from; ``params`` records
    the transform parameters *actually applied*, which for ``draw_id == 0`` is the
    identity rather than the constructor's settings.
    """

    draw_id: int
    seed: int
    n_vehicles: int
    source_sha256: str
    params: dict[str, float]


def sumo_begin_from_sumocfg(sumocfg_path: str | Path) -> float:
    """Return the ``<time><begin>`` value of a ``.sumocfg``, in seconds.

    Raises ``ValueError`` when the element is missing or unparseable rather than
    defaulting to 0.0: the CityFlow and SUMO time bases differ per scenario (0 for
    hangzhou, 25200 for cologne), so a silent default would render vehicles outside the
    simulated window and produce an empty SUMO run that still looks successful.
    """
    path = Path(sumocfg_path)
    root = ET.parse(path).getroot()
    node = root.find("./time/begin")
    if node is None:
        node = root.find(".//begin")
    raw = None if node is None else node.get("value")
    if raw is None:
        raise ValueError(
            f"{path} has no readable <time><begin value=...>; the SUMO departure time "
            "base cannot be derived and must not be defaulted (hangzhou begins at 0, "
            "cologne at 25200 -- guessing either would silently move every vehicle "
            "outside the simulated window)"
        )
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} has a <begin> value that is not a number ({raw!r}); refusing to "
            "guess the SUMO departure time base"
        ) from exc


class FlowRandomizer:
    """Seeded, engine-independent demand randomiser for one CityFlow flow file.

    Randomising at the vehicle-list level keeps every draw renderable to *both* backends,
    which is what keeps claim C3 (CityFlow -> SUMO transfer) reachable.
    """

    def __init__(
        self,
        source_flow_path: str | Path,
        *,
        base_seed: int = DEFAULT_BASE_SEED,
        jitter_sigma_s: float = DEFAULT_JITTER_SIGMA_S,
        thin_p: float = DEFAULT_THIN_P,
        volume_scale: float = DEFAULT_VOLUME_SCALE,
    ) -> None:
        if not 0.0 <= thin_p < 1.0:
            raise ValueError(f"thin_p must be in [0, 1), got {thin_p!r}")
        if jitter_sigma_s < 0.0:
            raise ValueError(f"jitter_sigma_s must be >= 0, got {jitter_sigma_s!r}")
        if volume_scale <= 0.0:
            raise ValueError(f"volume_scale must be > 0, got {volume_scale!r}")

        self._path = Path(source_flow_path)
        raw = self._path.read_bytes()
        self._source_sha256 = hashlib.sha256(raw).hexdigest()

        entries = json.loads(raw)
        if not isinstance(entries, list):
            raise ValueError(
                f"{self._path} is not a CityFlow flow file: expected a JSON list of "
                f"vehicle insertions, got {type(entries).__name__}"
            )
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or not _ENTRY_KEYS <= set(entry):
                raise ValueError(
                    f"{self._path} entry {i} is missing required keys; expected at "
                    f"least {sorted(_ENTRY_KEYS)}"
                )
        self._entries: list[dict[str, Any]] = entries

        self._base_seed = int(base_seed)
        self._jitter_sigma_s = float(jitter_sigma_s)
        self._thin_p = float(thin_p)
        self._volume_scale = float(volume_scale)

        self._indent, self._trailing_newline = self._sniff_formatting(raw)

    # -- formatting --------------------------------------------------------

    @staticmethod
    def _sniff_formatting(raw: bytes) -> tuple[int, bool]:
        """Recover the source's indent width and trailing-newline flag.

        Verified, not guessed: the sniffed settings are used to re-serialise the parsed
        source and the result is compared against *raw*.  A ``FlowRandomizer`` that
        cannot reproduce its own source refuses to construct, which makes
        ``render_cityflow(draw(0)) == source`` a construction-time invariant rather than
        something only a test happens to check.
        """
        text = raw.decode("utf-8")
        parsed = json.loads(raw)

        candidates: list[tuple[int, bool]] = []
        match = re.match(r"\[\s*?\n([ ]*)\S", text)
        if match is not None:
            candidates.append((len(match.group(1)), text.endswith("\n")))
        # The two indents actually used in this repo, then a short fallback sweep.
        for indent in (2, 4, 0, 1, 3, 6, 8):
            for trailing in (False, True):
                candidates.append((indent, trailing))

        for indent, trailing in candidates:
            rendered = json.dumps(parsed, indent=indent)
            if trailing:
                rendered += "\n"
            if rendered.encode("utf-8") == raw:
                return indent, trailing

        raise ValueError(
            "cannot reproduce the source flow file byte-for-byte with any supported "
            "JSON formatting (tried indents 0-8, with and without a trailing newline). "
            "draw(0) is the nominal control condition for every experiment and must "
            "render byte-identically, so this file is refused rather than silently "
            "reformatted. Re-save it with json.dumps(entries, indent=2 or 4)."
        )

    # -- introspection -----------------------------------------------------

    @property
    def source_sha256(self) -> str:
        """sha256 of the source flow file, as read at construction."""
        return self._source_sha256

    @property
    def n_source_vehicles(self) -> int:
        """Number of entries in the source flow file."""
        return len(self._entries)

    @property
    def params(self) -> dict[str, float]:
        """The transform parameters this instance was configured with."""
        return {
            "jitter_sigma_s": self._jitter_sigma_s,
            "thin_p": self._thin_p,
            "volume_scale": self._volume_scale,
        }

    # -- drawing -----------------------------------------------------------

    def draw(self, draw_id: int) -> tuple[list[dict[str, Any]], FlowDraw]:
        """Return ``(entries, provenance)`` for *draw_id*; 0 is the identity.

        The returned entries are always freshly built, so a caller mutating them cannot
        reach this object's parsed copy of the source.
        """
        draw_id = int(draw_id)
        if draw_id < 0:
            raise ValueError(f"draw_id must be >= 0, got {draw_id}")

        if draw_id == 0:
            entries = copy.deepcopy(self._entries)
            return entries, FlowDraw(
                draw_id=0,
                seed=self._base_seed,
                n_vehicles=len(entries),
                source_sha256=self._source_sha256,
                # The identity is what was applied; quoting the constructor's settings
                # here would misdescribe a nominal-flow corpus.
                params={"jitter_sigma_s": 0.0, "thin_p": 0.0, "volume_scale": 1.0},
            )

        rng = np.random.default_rng(self._base_seed + draw_id)
        n = len(self._entries)

        # 1. Bernoulli thinning.
        kept = np.flatnonzero(rng.random(n) >= self._thin_p)

        # 2. Volume scaling, relative order preserved.
        if len(kept) > 0:
            target = int(round(len(kept) * self._volume_scale))
            if target < len(kept):
                chosen = np.sort(rng.choice(len(kept), size=target, replace=False))
                kept = kept[chosen]
            elif target > len(kept):
                extra = rng.integers(0, len(kept), size=target - len(kept))
                kept = np.concatenate([kept, kept[extra]])

        # 3. Departure jitter -- one independent sample per surviving entry, so the
        #    duplicates created above separate instead of stacking.
        shifts = rng.normal(0.0, self._jitter_sigma_s, size=len(kept))

        entries = []
        for index, shift in zip(kept.tolist(), shifts.tolist()):
            entry = copy.deepcopy(self._entries[index])
            entry["startTime"] = _coerce_like(
                max(0.0, float(entry["startTime"]) + shift), entry["startTime"]
            )
            entry["endTime"] = entry["startTime"]
            entries.append(entry)

        # 4. Stable sort; jitter is the only step that can disorder the list.
        entries.sort(key=lambda e: e["startTime"])

        return entries, FlowDraw(
            draw_id=draw_id,
            seed=self._base_seed + draw_id,
            n_vehicles=len(entries),
            source_sha256=self._source_sha256,
            params=self.params,
        )

    # -- rendering ---------------------------------------------------------

    def render_cityflow(
        self, entries: Sequence[dict[str, Any]], out_path: str | Path
    ) -> Path:
        """Write *entries* as a CityFlow ``flow.json``, in the source's own formatting."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(list(entries), indent=self._indent)
        if self._trailing_newline:
            text += "\n"
        out.write_bytes(text.encode("utf-8"))
        return out

    def render_sumo(
        self,
        entries: Sequence[dict[str, Any]],
        out_path: str | Path,
        *,
        template_rou_path: str | Path,
        depart_offset: float,
    ) -> Path:
        """Write *entries* as a SUMO ``.rou.xml``, reusing the template's ``<vType>``.

        *depart_offset* is deliberately required: it comes from the scenario's
        ``.sumocfg`` via :func:`sumo_begin_from_sumocfg` and differs per scenario.

        The template's ``<vType>`` elements are copied verbatim and its ``<vehicle>``
        elements are inspected for a ``type`` attribute, whose presence is mirrored --
        cologne carries ``type="pkw"`` while hangzhou omits it and therefore uses SUMO's
        default vehicle type.  Mirroring reproduces each scenario's real behaviour rather
        than quietly "fixing" hangzhou.
        """
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        template_root = ET.parse(Path(template_rou_path)).getroot()
        vtypes = template_root.findall("vType")
        template_vehicles = template_root.findall("vehicle")
        type_attr = template_vehicles[0].get("type") if template_vehicles else None

        routes = ET.Element(
            "routes",
            {
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:noNamespaceSchemaLocation": (
                    "http://sumo.dlr.de/xsd/routes_file.xsd"
                ),
            },
        )
        for vtype in vtypes:
            ET.SubElement(routes, "vType", dict(vtype.attrib))

        # SUMO rejects a route file whose departures are out of order; draw() already
        # sorts, and sorting again here keeps render_sumo correct for any caller.
        ordered = sorted(entries, key=lambda e: float(e["startTime"]))
        for i, entry in enumerate(ordered):
            attrs = {
                "id": str(i),
                "depart": f"{float(entry['startTime']) + float(depart_offset):.2f}",
            }
            if type_attr is not None:
                attrs["type"] = type_attr
            vehicle = ET.SubElement(routes, "vehicle", attrs)
            ET.SubElement(vehicle, "route", {"edges": " ".join(entry["route"])})

        tree = ET.ElementTree(routes)
        ET.indent(tree, space="    ")
        tree.write(out, encoding="utf-8", xml_declaration=True)
        return out
