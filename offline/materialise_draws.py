"""Materialise flow draws into stable, runnable scenario directories.

WHY THIS MODULE EXISTS
----------------------
:mod:`offline.flow_randomizer` can *produce* a draw and render it; :mod:`offline.collect`
writes one **ephemerally**, under ``<out-dir>/flows/``, and deletes it on the next run
(``offline/collect.py:424-440``).  Nothing wrote a draw somewhere stable and made it
runnable.  Three things need exactly that: ``mappo_dr`` (RUNSPEC_01 §1), the reported
online-MAPPO baseline on the held-out draws 1000-1099 (D4), and the P2.2 collection
campaign.

FORMAT VERSION
--------------
``materialised-draw/1.0`` -- written into every ``provenance.json`` as
``format_version``.  Any change to the layout or to the meaning of a field bumps it.

LAYOUT (the draw id is in the path, deliberately)
-------------------------------------------------
::

    <out_root>/<scenario_key>/draw_<NNNN>/
        flow.json          CityFlow demand for this draw, in the source's own formatting
        cityflow.json      sim config: the source config with dir/flowFile repointed here
        routes.rou.xml     SUMO rendering; present only when the scenario is paired
        provenance.json    the record described below

``scenario_key`` is the **stem of the source sim config** (``configs/sim/cityflow1x1.json``
-> ``cityflow1x1``), which is the same key :mod:`offline.collect` already writes into
every manifest as ``scenario_id`` (``offline/collect.py:518-523``), so no second registry
exists that could drift away from it.  ``draw_<NNNN>`` is zero-padded to four digits so
lexical order equals numeric order across the whole registered range.  Putting the id in
the path is what makes a config impossible to confuse with another draw, and makes a pool
greppable (``grep -rl draw_10 <out_root>``).

Callers ask for "scenario S, draw D" through :func:`draw_config_path`, a **pure path
function**: no I/O and no directory scan, so the draw-cycling trainer can pre-build one
env per draw and rotate them without touching the filesystem per episode.

DRAW IDENTITY (the reason nothing here is a CLI flag)
-----------------------------------------------------
A draw is identified by ``(base_seed, draw_id)``, not by ``draw_id`` alone.  This module
therefore pins the randomiser to the **constants exported by**
:mod:`offline.flow_randomizer` -- ``DEFAULT_BASE_SEED``, ``DEFAULT_JITTER_SIGMA_S``,
``DEFAULT_THIN_P``, ``DEFAULT_VOLUME_SCALE`` -- imported rather than re-typed, so the
canonical definition cannot drift from the module that implements it, and records all
four in every provenance record.  Exposing them as flags would let two callers create two
different demands under the same id, which is precisely the defect this pinning closes.
(``offline.collect`` still derives its randomiser seed from ``--base-seed``, which is
overloaded three ways there; realigning it is a follow-up task, not this module's job.)

Pools, from ``PREREGISTRATION.md`` §5 and D4: **0** is the nominal control, reported
separately and never pooled; **1-999** is the training pool; **1000-1099** is the held-out
evaluation pool that must never enter any training corpus.  :func:`classify_draw_pool`
labels each written draw, and the label lands in ``provenance.json``.

IDEMPOTENCE AND WHAT "IDENTICAL" MEANS
--------------------------------------
Re-materialising an existing draw is a **no-op** (``action == "kept"``), so filling in the
rest of the 1000-1099 pool later never rebuilds what is already there.  Identity is judged
on the *artifacts and the parameters that determine them* -- the three rendered files
byte-for-byte, plus every provenance field except those in :data:`_NON_IDENTITY_FIELDS`:
``git_commit`` / ``git_dirty``, which describe **when** a draw was materialised, and
``source_config`` / ``source_flow`` / ``source_roadnet``, which describe **where the source
happened to live**.  Including the first pair would make every commit invalidate the whole
tree; including the second group made the same scenario reached from a second worktree look
like different demand (``BRIEF_31`` Amendments D2 and E3, ``DEFERRED`` 61).  **The
``*_sha256`` companion of each of those three paths remains an identity field, so a genuinely
different source still refuses.**  A draw that exists but differs is **refused**, never
silently rewritten, and only ``force=True`` replaces it.

WARNING: a path exemption does NOT make the tool working-directory independent.  The rendered
``cityflow.json`` embeds ``dir`` as an absolute path resolved against the process working
directory, and rendered files are compared *before* any provenance field -- so re-materialising
from a different working directory still refuses, with ``cityflow.json differs byte-for-byte``.
Measured on 2026-08-28 against the ten ``cityflow_grid4x4`` held-out draws: all ten rendered
configs differed across worktrees while all ten ``flow.json`` -- the drawn demand -- were
byte-identical.

FILESYSTEM-MUTATION BARRIER
---------------------------
Three ordered phases, and the order is the guarantee:

1. **Validate.** Sources exist and parse, ids are legal and unique, the scenario key is a
   single safe path component, every planned output path resolves **inside** ``out_root``,
   and every already-existing draw is classified as kept or conflicting.  No ``mkdir``, no
   write, no delete happens in this phase -- a refused run creates nothing at all.
2. **Build in memory.** Entries, rendered bytes for all three files, digests, provenance.
   The rendering helpers write files by design, so this phase renders into an OS temp
   directory (``tempfile``) and reads the bytes back: nothing inside ``out_root`` or the
   repo is touched while a run can still fail.
3. **Stage, then commit.** Files are written into ``<out_root>/.staging-*`` and moved into
   place with :func:`os.replace` (same filesystem, atomic).  Renames already made are
   rolled back if a later one fails, a replacement moves the old directory aside and
   unlinks it only after the new one is in place, and the staging root is removed in a
   ``finally`` -- so a failed run also leaves no new directories behind.

REUSED, NOT REIMPLEMENTED
-------------------------
``dir``/``flowFile`` resolution comes from :mod:`offline.collect`
(``_cityflow_flow_source``, ``_write_draw_config``), which is verified against a real
engine.  A second implementation of the same ``dir + flowFile`` trick is exactly how two
subtly different draw paths appear.  Two consequences, both accepted and visible:
importing this module pulls torch (``collect`` -> ``agent.utils.utils``), and a written
config carries an **absolute** ``dir``, because that is what the verified helper produces.
The tree is git-ignored and machine-local, so absoluteness costs nothing and removes the
cwd dependence ``configs/sim/*.json`` has.  Resolving the *source* config's relative
``dir`` still happens against the process cwd (inherited from the same helper), so this
module validates that the resolved flow file exists and fails loudly rather than
materialising something empty when run from the wrong directory.

SUMO
----
Rendered where -- and only where -- the scenario's own ``.sumocfg`` names a route-file
template that exists: hangzhou and cologne3 qualify, grid4x4 does not (its ``.sumocfg``
names ``grid4x4.rou.xml``, which the repo does not contain).  The pairing is read from
the ``.sumocfg`` rather than guessed from a directory listing, and an unusable pairing is
recorded as ``sumo_skipped_reason`` instead of being silently dropped.

A rendered file whose template leaves ``<vType>`` unbound (hangzhou) says so in its
provenance: SUMO would run those vehicles as ``DEFAULT_VEHTYPE``, so **the parent's
``routes.rou.xml`` must never feed a transfer measurement**.  It is kept as the input of the
derivation below, and nothing else.

THE PARITY SUBDIRECTORY (P7.2a, ``BRIEF_35``)
---------------------------------------------
``materialise_parity()`` adds one subdirectory per draw and **never writes to the parent**::

    <out_root>/<scenario_key>/draw_<NNNN>/parity/
        routes.rou.xml      the parent's rendering with the parity <vType> bound to EVERY vehicle
        noteleport.sumocfg  the SHIPPED network by relative path, <time-to-teleport value="-1"/>
        provenance.json     format ``materialised-draw-parity/1.0`` (the record below)

**Format version** ``materialised-draw-parity/1.0``, written into every parity
``provenance.json`` as ``format_version``, beside the parent's ``materialised-draw/1.0``.

**Alignment convention: unchanged, and that is the point.**  Vehicle ids stay ``0..n-1`` in
``startTime`` order and ``depart`` stays ``startTime + <begin>`` at two decimals -- the
convention :func:`offline.flow_randomizer.FlowRandomizer.render_sumo` writes.  The parity step
edits exactly two things, the single ``<vType>`` element and each ``<vehicle>``'s ``type``
attribute, and :func:`offline.parity._verify_rendered_rou` re-parses the result to prove ids,
departures and routes survived element for element.  Each draw's provenance then records CAP(E)
over that draw -- the CityFlow demand against the bound rendering, multiset and index-aligned --
so "the SUMO demand equals the CityFlow demand" is a measurement per draw, not an inference from
the two having been produced by the same randomiser.

The subdirectory is invisible to :func:`_existing_conflict`, which compares **files** only, so
the parity phase cannot make a later ``materialise()`` refuse; a file placed beside the parent's
four would.  ``<time-to-teleport value="-1"/>`` is A15(c), binding on every SUMO measurement
recorded after 2026-09-12 and on every ``.sumocfg`` generated for drawn demand.

THE TEMPLATED SHAPE (P7.3d, ``BRIEF_39`` C1 + Amendments A5 and A.1-3)
---------------------------------------------------------------------
**Format version** ``materialised-draw-parity/1.1``, written ONLY for a scenario whose SUMO side
lives outside the repository (grid4x4: RESCO's net and route archive, CC BY-NC-SA 4.0, located
through ``RLTRAFFIC_GRID4X4_RESCO``, read in place, pinned by sha256).  Its parent holds three
files and ``"sumo": null`` -- the repo ships no route template for it -- so there is no parent
rendering to bind and none may be added beside the parent's files.  The bound routes are
therefore rendered INSIDE ``parity/``: the parent's ``flow.json`` entries go through the same
:meth:`offline.flow_randomizer.FlowRandomizer.render_sumo`, with the external route file as the
template and its ``.sumocfg``'s ``<begin>`` as the depart offset, and the parity ``<vType>`` is
INSERTED (the template declares none) rather than replaced.  **The alignment convention is the
one above, unchanged:** ids ``0..n-1`` in ``startTime`` order, ``depart = startTime + <begin>`` at
two decimals, and CAP(E) per draw recorded in the provenance.

What differs in the 1.1 record, and why it is a version and not an added key:
``parent.routes_sha256`` is ``null`` -- a field a 1.0 reader expects to be a digest -- with
``parent.routes_absent_reason`` beside it; a ``route_template`` block records the archive member,
its sha256, the external ``.sumocfg``'s sha256 and the licence; ``net.resolved`` is written
relative to the candidates root (``<candidates>/...``), never absolutely, while ``net.reference``
stays a path relative to the parity directory, exactly as in 1.0.  **The hangzhou shape still
writes 1.0, byte for byte**, so every parity directory written before P7.3d classifies as
``kept``; :func:`load_parity_provenance` reads both and refuses anything else.

The parity ``<vType>`` is PER SCENARIO (:class:`offline.parity.ParityScenario`) and DERIVED from
the scenario's CityFlow flow block; a block that disagrees with the registered table is refused.

GATE G1's ARTIFACT (``--report-cap-e``)
---------------------------------------
**Format version** ``p7.3d-cap-e/1.0`` (:func:`cap_e_report`): A15(g)'s condition re-run on the
RENDERED parity files -- per draw, CAP(E) on the key ``(depart, route)``, unshifted, as a
multiset and index-aligned through :func:`offline.conversion_audit.audit_pair`, plus the vType
binding read back from the rendered file.  Read-only under ``--out-root``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from collections import Counter

from offline import parity
from offline.collect import _cityflow_flow_source, _write_draw_config
from offline.flow_randomizer import (
    DEFAULT_BASE_SEED,
    DEFAULT_JITTER_SIGMA_S,
    DEFAULT_THIN_P,
    DEFAULT_VOLUME_SCALE,
    FlowRandomizer,
    sumo_begin_from_sumocfg,
)
from offline.trajectory_logger import _repo_git_hash

__all__ = [
    "CITYFLOW_CONFIG_FILENAME",
    "DEFAULT_OUT_ROOT",
    "FLOW_FILENAME",
    "FORMAT_VERSION",
    "HELD_OUT_POOL",
    "MaterialisedDraw",
    "P4_3_PROBE_ARTIFACT",
    "CAP_E_FORMAT_VERSION",
    "PARITY_DIRNAME",
    "PARITY_FORMAT_VERSION",
    "PARITY_FORMAT_VERSION_TEMPLATED",
    "READABLE_PARITY_FORMAT_VERSIONS",
    "PARITY_ROUTES_FILENAME",
    "PARITY_SUMOCFG_FILENAME",
    "PROVENANCE_FILENAME",
    "ParityResult",
    "ProbeCheck",
    "SUMO_ROUTES_FILENAME",
    "TRAINING_POOL",
    "assert_cap_e_exact",
    "build_parser",
    "cap_e_report",
    "classify_draw_pool",
    "draw_config_path",
    "draw_dir",
    "load_parity_provenance",
    "load_provenance",
    "main",
    "materialise",
    "materialise_parity",
    "parity_dir",
    "parity_sumocfg_path",
    "scenario_key_for_config",
    "P4_HELDOUT_THRESHOLDS_ARTIFACT",
    "verify_against_artifact",
    "verify_p4_3_probe",
]

FORMAT_VERSION = "materialised-draw/1.0"

#: Format version of the additive parity subdirectory (P7.2a, ``BRIEF_35``).  Bumped by any
#: change to that layout or to the meaning of a field in its ``provenance.json``.
PARITY_FORMAT_VERSION = "materialised-draw-parity/1.0"

#: The TEMPLATED shape (P7.3d, ``BRIEF_39`` Amendment A5): a scenario whose parent draw carries NO
#: SUMO rendering, so the bound routes are rendered inside ``parity/`` from the parent's
#: ``flow.json`` and an external route template.  ``parent.routes_sha256`` is ``null`` there -- a
#: field a 1.0 reader expects to be a digest -- and THAT is what forces the bump; added keys alone
#: would not.  The hangzhou shape keeps writing 1.0 byte-for-byte, so every parity directory
#: written before P7.3d still classifies as ``kept``.
PARITY_FORMAT_VERSION_TEMPLATED = "materialised-draw-parity/1.1"

#: What :func:`load_parity_provenance` accepts.  An unknown version is refused, never guessed at.
READABLE_PARITY_FORMAT_VERSIONS: tuple[str, ...] = (
    PARITY_FORMAT_VERSION,
    PARITY_FORMAT_VERSION_TEMPLATED,
)

#: Format of ``docs/data/p7_3d_cap_e.json`` -- gate G1's artifact (A15(g)'s condition).
CAP_E_FORMAT_VERSION = "p7.3d-cap-e/1.0"

DEFAULT_OUT_ROOT = Path("scenarios/draws")

FLOW_FILENAME = "flow.json"
CITYFLOW_CONFIG_FILENAME = "cityflow.json"
SUMO_ROUTES_FILENAME = "routes.rou.xml"
PROVENANCE_FILENAME = "provenance.json"

#: The parity subdirectory and its three files.  ``PARITY_DIRNAME`` is also the suffix the
#: commit-time target guard requires, so no ``os.replace`` of this phase can land on a draw.
PARITY_DIRNAME = "parity"
PARITY_ROUTES_FILENAME = "routes.rou.xml"
PARITY_SUMOCFG_FILENAME = "noteleport.sumocfg"

#: DEFERRED 80's reference: the held-out pool's CityFlow parents (``--verify-heldout-thresholds``).
#: The SUMO held-out draws P7.3a evaluates on are rendered FROM these, so their pedigree is
#: proved by re-running them rather than inherited from P5.3a's five survivors (``BRIEF_37`` §3.0).
P4_HELDOUT_THRESHOLDS_ARTIFACT = (
    Path(__file__).resolve().parent.parent / "docs" / "data" / "p4_heldout_thresholds.json"
)

#: P4.3's committed in-domain probe -- the band gate's reference (``--verify-p4-3-probe``).
P4_3_PROBE_ARTIFACT = Path(__file__).resolve().parent.parent / "docs" / "data" / "p4_3_probe.json"

#: Registered draw pools (``PREREGISTRATION.md`` §5, D4).  Draw 0 is the nominal control
#: and belongs to neither: it is reported separately and never pooled.
TRAINING_POOL = range(1, 1000)
HELD_OUT_POOL = range(1000, 1100)

#: A scenario key must be one safe path component: no separator, no leading dot, so
#: neither ``..`` nor ``.`` nor ``a/b`` can ever reach a path join.
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Provenance fields that describe *when* and *where* a draw was materialised rather than *what*
#: it is, and are therefore excluded from the identity comparison (see the module docstring).
#:
#: ``source_config`` and ``source_roadnet`` joined on 2026-08-29 under ``BRIEF_31`` Amendment D2,
#: which closes ``DEFERRED`` 61.  **The change strictly NARROWS identity to content.**  Each of them
#: has a ``*_sha256`` companion that stays an identity field, and the digest is what says whether the
#: source is the same file; the bare string only says which directory the person who ran the tool was
#: standing in.  ``source_config`` is recorded verbatim from the caller's argument and
#: ``source_roadnet`` is resolved against the process working directory, so the SAME scenario reached
#: from a second worktree, or through an absolute rather than a relative path, produced a different
#: string and an identical draw was refused as though its demand had changed.
#:
#: ``source_flow`` joined on 2026-08-29 under Amendment E3, which closed the asymmetry D2 left.  It
#: is stored ABSOLUTE (``/.../scenarios/grid4x4/grid4x4_flow.json``), so it was not a latent case --
#: it was the next field to fire from any other tree, and only the first-mismatch return of
#: :func:`_existing_conflict` kept it hidden behind ``source_config``.
#:
#: **The invariant to preserve: every path field here has a ``*_sha256`` twin that is NOT here.**
#: Exempting a digest would delete the check rather than narrow it, and a test asserts both halves.
_NON_IDENTITY_FIELDS = frozenset(
    {"git_commit", "git_dirty", "source_config", "source_flow", "source_roadnet"}
)

_STAGING_PREFIX = ".staging-"

_SUMO_CAVEAT_UNBOUND = (
    "the template's <vehicle> elements carry no type attribute, so this rendering "
    "faithfully inherits the scenario's unbound <vType>: SUMO would run these vehicles "
    "as DEFAULT_VEHTYPE. Do not feed this file into a transfer measurement until P7.0's "
    "parity contract lands (docs/briefs/BRIEF_04_p7.0_transfer_gate.md section 3)."
)
_SUMO_CAVEAT_BOUND = (
    "the template binds its <vType>, but no .sumocfg is generated here (P7.3 owns it) "
    "and P7.0's parity contract (docs/briefs/BRIEF_04_p7.0_transfer_gate.md section 3) "
    "still governs any transfer measurement using this file."
)


@dataclass(frozen=True)
class MaterialisedDraw:
    """One draw's on-disk result.

    ``action`` is ``"written"`` (new), ``"kept"`` (already present and identical),
    ``"replaced"`` (existed, differed, ``force=True``) or ``"planned"`` (``dry_run``).
    """

    scenario_key: str
    draw_id: int
    pool: str
    directory: Path
    config_path: Path
    flow_path: Path
    sumo_path: Path | None
    n_vehicles: int
    flow_sha256: str
    action: str


@dataclass(frozen=True)
class ParityResult:
    """One draw's parity subdirectory.

    ``action`` is ``"written"``, ``"kept"``, ``"replaced"`` or ``"planned"`` (dry run), and
    ``parent_action`` is what :func:`materialise` reported for the parent of the same draw, so a
    caller can tell "the parent was created for me" from "the parent was already here".
    """

    scenario_key: str
    draw_id: int
    pool: str
    directory: Path
    routes_path: Path
    sumocfg_path: Path
    provenance_path: Path
    n_vehicles: int
    n_bound: int
    action: str
    parent_action: str
    #: What the phase DECIDED, whether or not it was carried out: ``"written"``, ``"kept"``,
    #: ``"replaced"`` or ``"differs"``.  Under ``dry_run`` ``action`` is ``"planned"`` for every
    #: draw -- the convention :class:`MaterialisedDraw` already uses -- so the decision would
    #: otherwise be unreportable, and the dry run's whole purpose is to report it.
    planned_action: str


@dataclass(frozen=True)
class ProbeCheck:
    """One draw's comparison against ``docs/data/p4_3_probe.json`` (the band gate)."""

    draw_id: int
    matches: bool
    observed: dict[str, float]
    expected: dict[str, float]
    differing: tuple[str, ...]


@dataclass(frozen=True)
class _BuiltDraw:
    """Everything one draw needs, held in memory before anything is written."""

    draw_id: int
    pool: str
    files: dict[str, bytes]
    n_vehicles: int
    flow_sha256: str
    has_sumo: bool


# -- small helpers ---------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> tuple[str, bool]:
    """Repo revision that produced a draw, plus whether the tree was dirty.

    The revision comes from :func:`offline.trajectory_logger._repo_git_hash` -- the same
    helper the corpus manifests use, so a corpus and the demand it came from quote the
    hash the same way.  ``git_dirty`` is recorded because a commit hash under-describes a
    modified tree.
    """
    commit = _repo_git_hash()
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return commit, False
    if result.returncode != 0:
        return commit, False
    return commit, bool(result.stdout.strip())


def _checked_scenario_key(scenario_key: str) -> str:
    """Return *scenario_key* if it is a single safe path component, else raise."""
    if not isinstance(scenario_key, str) or not _KEY_RE.match(scenario_key):
        raise ValueError(
            f"scenario key {scenario_key!r} is not a single safe path component; it "
            "must match [A-Za-z0-9][A-Za-z0-9._-]* so a key can never contain a "
            "separator or a leading dot and escape out_root"
        )
    return scenario_key


def _checked_draw_id(draw_id: int) -> int:
    value = int(draw_id)
    if value < 0:
        raise ValueError(f"draw ids must be >= 0, got {value}; draw 0 is the nominal flow")
    return value


def _checked_draw_ids(draw_ids: Sequence[int]) -> list[int]:
    """Validate and return the requested ids, in the order given.

    Duplicates are refused rather than deduplicated: a caller asking for the same draw
    twice has a bug in its id set, and silently collapsing it would hide that.
    """
    ids = [_checked_draw_id(draw_id) for draw_id in draw_ids]
    if not ids:
        raise ValueError("no draw ids requested; nothing to materialise")
    seen = {draw_id for draw_id in ids if ids.count(draw_id) > 1}
    if seen:
        raise ValueError(
            f"repeated draw id(s) {sorted(seen)}; each draw is materialised once and "
            "re-running is already a no-op, so a repeat is a bug in the caller's id set"
        )
    return ids


def _checked_output_path(out_root: str | Path, path: str | Path) -> Path:
    """Return *path* if it resolves inside *out_root*, else raise.

    Defence in depth behind :func:`_checked_scenario_key`: every path this module writes
    passes through here, so no combination of key, id and root can place a file outside
    the tree the caller named.
    """
    root = Path(out_root).resolve()
    resolved = Path(path).resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise ValueError(
            f"refusing to write {resolved} because it resolves outside out_root {root}"
        )
    return Path(path)


# -- public path helpers ---------------------------------------------------


def scenario_key_for_config(source_config: str | Path) -> str:
    """Return the directory key for a source sim config (its stem)."""
    return _checked_scenario_key(Path(source_config).stem)


def classify_draw_pool(draw_id: int) -> str:
    """Return the registered pool of *draw_id*.

    ``"nominal"`` (0), ``"training"`` (1-999), ``"held_out"`` (1000-1099) or
    ``"unregistered"`` -- the last is not an error here, but it is a label a reader can
    grep for before an unregistered draw reaches a reported number.
    """
    value = _checked_draw_id(draw_id)
    if value == 0:
        return "nominal"
    if value in TRAINING_POOL:
        return "training"
    if value in HELD_OUT_POOL:
        return "held_out"
    return "unregistered"


def draw_dir(
    scenario_key: str, draw_id: int, *, out_root: str | Path = DEFAULT_OUT_ROOT
) -> Path:
    """Return the directory holding one materialised draw. Pure path arithmetic."""
    key = _checked_scenario_key(scenario_key)
    value = _checked_draw_id(draw_id)
    root = Path(out_root)
    return _checked_output_path(root, root / key / f"draw_{value:04d}")


def draw_config_path(
    scenario_key: str, draw_id: int, *, out_root: str | Path = DEFAULT_OUT_ROOT
) -> Path:
    """Return the CityFlow sim config for one materialised draw. Pure path arithmetic.

    This is the "give me scenario S, draw D" lookup: no I/O, so a trainer can build the
    whole rotation up front.
    """
    return draw_dir(scenario_key, draw_id, out_root=out_root) / CITYFLOW_CONFIG_FILENAME


def parity_dir(
    scenario_key: str, draw_id: int, *, out_root: str | Path = DEFAULT_OUT_ROOT
) -> Path:
    """Return the parity subdirectory of one materialised draw. Pure path arithmetic."""
    return draw_dir(scenario_key, draw_id, out_root=out_root) / PARITY_DIRNAME


def parity_sumocfg_path(
    scenario_key: str, draw_id: int, *, out_root: str | Path = DEFAULT_OUT_ROOT
) -> Path:
    """Return the teleport-free SUMO config of one draw. Pure path arithmetic, no I/O.

    This is P7.2b's lookup: "give me scenario S, draw D, runnable on SUMO under the parity
    contract", resolvable without a directory scan so a probe can build its whole rotation up
    front.
    """
    return parity_dir(scenario_key, draw_id, out_root=out_root) / PARITY_SUMOCFG_FILENAME


def load_parity_provenance(
    scenario_key: str, draw_id: int, *, out_root: str | Path = DEFAULT_OUT_ROOT
) -> dict[str, Any]:
    """Return the provenance record of one draw's parity subdirectory.

    Both shapes are readable (:data:`READABLE_PARITY_FORMAT_VERSIONS`): ``1.0``, whose
    ``parent.routes_sha256`` is a digest, and the templated ``1.1``, where it is ``null`` and a
    ``route_template`` block says what the routes were rendered from.  Any other version is refused
    rather than read as if it were one of them.
    """
    path = parity_dir(scenario_key, draw_id, out_root=out_root) / PROVENANCE_FILENAME
    record = json.loads(path.read_bytes())
    version = record.get("format_version")
    if version not in READABLE_PARITY_FORMAT_VERSIONS:
        raise ValueError(
            f"{path}: parity provenance format {version!r} is not readable by this build "
            f"(readable: {list(READABLE_PARITY_FORMAT_VERSIONS)})"
        )
    return record


def load_provenance(
    scenario_key: str, draw_id: int, *, out_root: str | Path = DEFAULT_OUT_ROOT
) -> dict[str, Any]:
    """Return the provenance record of one materialised draw."""
    path = (
        draw_dir(scenario_key, draw_id, out_root=out_root) / PROVENANCE_FILENAME
    )
    return json.loads(path.read_bytes())


# -- SUMO pairing ----------------------------------------------------------


def _scenario_dir(source_config: str | Path) -> Path:
    """Absolute scenario directory a CityFlow sim config points at."""
    cfg = json.loads(Path(source_config).read_bytes())
    cfg_dir = cfg.get("dir", "")
    if not os.path.isabs(cfg_dir):
        cfg_dir = str(Path.cwd() / cfg_dir)
    return Path(os.path.normpath(cfg_dir))


def _vtype_is_bound(template_rou: Path) -> bool:
    """Whether the template's vehicles actually reference one of its ``<vType>`` ids."""
    root = ET.parse(template_rou).getroot()
    vtype_ids = {vtype.get("id") for vtype in root.findall("vType")}
    vehicles = root.findall("vehicle")
    type_attr = vehicles[0].get("type") if vehicles else None
    return type_attr is not None and type_attr in vtype_ids


def _sumo_pairing(source_config: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(pairing, skip_reason)`` for a scenario, exactly one of them non-None.

    The pairing is read from the scenario's own ``.sumocfg`` -- which names its route
    file -- rather than guessed from a directory listing: cologne3 ships two ``.rou.xml``
    files and only one of them is the paired template.  Every unusable case yields a
    reason string that is written into provenance, so a missing SUMO rendering is never
    silent.
    """
    scenario_dir = _scenario_dir(source_config)
    candidates = sorted(scenario_dir.glob("*.sumocfg"))
    if not candidates:
        return None, f"no .sumocfg in {scenario_dir}"
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        return None, (
            f"{len(candidates)} .sumocfg files in {scenario_dir} ({names}); refusing to "
            "guess which one pairs with this CityFlow config"
        )

    sumocfg = candidates[0]
    root = ET.parse(sumocfg).getroot()
    node = root.find("./input/route-files")
    if node is None:
        node = root.find(".//route-files")
    raw = None if node is None else node.get("value")
    if not raw:
        return None, f"{sumocfg.name} has no <route-files value=...>"

    names = [name for name in re.split(r"[,\s]+", raw.strip()) if name]
    if len(names) != 1:
        return None, (
            f"{sumocfg.name} names {len(names)} route files ({raw!r}); refusing to guess "
            "which one carries this scenario's demand"
        )

    template = scenario_dir / names[0]
    if not template.is_file():
        return None, (
            f"{sumocfg.name} names route file {names[0]!r}, which does not exist in "
            f"{scenario_dir}; this scenario is CityFlow-only in this repo"
        )

    try:
        depart_offset = sumo_begin_from_sumocfg(sumocfg)
    except (ValueError, ET.ParseError) as exc:
        return None, f"{sumocfg.name}: {exc}"

    bound = _vtype_is_bound(template)
    return (
        {
            "sumocfg": str(sumocfg),
            "template_rou": str(template),
            "depart_offset": depart_offset,
            "vtype_bound": bound,
            "caveat": _SUMO_CAVEAT_BOUND if bound else _SUMO_CAVEAT_UNBOUND,
        },
        None,
    )


# -- building (phase 2: in memory only) ------------------------------------


def _build_draw(
    randomizer: FlowRandomizer,
    *,
    source_config: Path,
    scenario_key: str,
    draw_id: int,
    target: Path,
    sources: dict[str, Any],
    sumo: dict[str, Any] | None,
    sumo_skipped_reason: str | None,
    scratch: Path,
) -> _BuiltDraw:
    """Render one draw entirely in memory.

    *scratch* is an OS temp directory: the renderers write files by design, so they
    render there and the bytes are read back.  Nothing under ``out_root`` is created
    while the run can still fail.  ``_write_draw_config`` is handed the **final** flow
    path, never the scratch one -- it stores ``flowFile`` as a path relative to the
    scenario directory, so a scratch path would survive into the committed config.
    """
    entries, provenance = randomizer.draw(draw_id)
    flow_bytes = randomizer.render_cityflow_bytes(entries)

    config_scratch = scratch / f"{scenario_key}_{draw_id}_{CITYFLOW_CONFIG_FILENAME}"
    _write_draw_config(source_config, target / FLOW_FILENAME, config_scratch)
    config_bytes = config_scratch.read_bytes()

    files: dict[str, bytes] = {
        FLOW_FILENAME: flow_bytes,
        CITYFLOW_CONFIG_FILENAME: config_bytes,
    }
    if sumo is not None:
        routes_scratch = scratch / f"{scenario_key}_{draw_id}_{SUMO_ROUTES_FILENAME}"
        randomizer.render_sumo(
            entries,
            routes_scratch,
            template_rou_path=sumo["template_rou"],
            depart_offset=sumo["depart_offset"],
        )
        files[SUMO_ROUTES_FILENAME] = routes_scratch.read_bytes()

    pool = classify_draw_pool(draw_id)
    record: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "scenario_key": scenario_key,
        "draw_id": draw_id,
        "pool": pool,
        # Draw 0 preserves source vehicle order while k > 0 sorts globally, so it is the
        # nominal control: reported separately, never pooled with the randomised draws.
        "is_nominal_control": draw_id == 0,
        "source_config": sources["config"],
        "source_config_sha256": sources["config_sha256"],
        "source_flow": sources["flow"],
        "source_flow_sha256": sources["flow_sha256"],
        "source_roadnet": sources["roadnet"],
        "source_roadnet_sha256": sources["roadnet_sha256"],
        "randomizer": {
            "base_seed": DEFAULT_BASE_SEED,
            "jitter_sigma_s": DEFAULT_JITTER_SIGMA_S,
            "thin_p": DEFAULT_THIN_P,
            "volume_scale": DEFAULT_VOLUME_SCALE,
        },
        "draw": {
            "seed": provenance.seed,
            "n_vehicles": provenance.n_vehicles,
            "source_sha256": provenance.source_sha256,
            # The parameters ACTUALLY applied: the identity for draw 0, which is not the
            # same thing as the configured parameters above.
            "params": dict(provenance.params),
        },
        "files": {name: _sha256(data) for name, data in sorted(files.items())},
        "sumo": sumo,
        "sumo_skipped_reason": sumo_skipped_reason,
        # NEP 19 gives np.random.Generator no cross-version stream guarantee, so the
        # numpy that drew this demand is part of its provenance.
        "numpy_version": np.__version__,
    }
    commit, dirty = _git_commit()
    record["git_commit"] = commit
    record["git_dirty"] = dirty

    files[PROVENANCE_FILENAME] = _provenance_bytes(record)
    return _BuiltDraw(
        draw_id=draw_id,
        pool=pool,
        files=files,
        n_vehicles=provenance.n_vehicles,
        flow_sha256=_sha256(flow_bytes),
        has_sumo=sumo is not None,
    )


def _provenance_bytes(record: dict[str, Any]) -> bytes:
    """Serialise a provenance record deterministically.

    Sorted keys and no wall-clock field: a materialised draw is a deterministic function
    of its inputs, so "this re-run was a no-op" stays checkable by byte equality.
    """
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _existing_conflict(target: Path, built: _BuiltDraw) -> str | None:
    """Return why *target* differs from *built*, or ``None`` when it is identical.

    Rendered files are compared byte-for-byte; the provenance record is then compared field by
    field with :data:`_NON_IDENTITY_FIELDS` excluded, so neither a later commit nor a different
    working directory makes an existing, correct draw look stale.

    WARNING: this returns on the FIRST mismatch, and the provenance loop walks
    ``sorted(set(on_disk) | set(fresh))``.  So a complaint about ``source_config`` means the fields
    after it alphabetically -- ``source_config_sha256`` among them -- were never reached, and the
    ABSENCE of a digest complaint is the loop stopping rather than the digests agreeing.  That is
    ``DEFERRED`` 54's class (assertions after the first failure never run) and it misled a reader on
    2026-08-28; it is recorded here so the next one is not misled the same way.
    """
    expected = set(built.files)
    present = {path.name for path in target.iterdir() if path.is_file()}
    extra = present - expected
    if extra:
        return f"it holds unexpected file(s) {sorted(extra)}"
    missing = expected - present
    if missing:
        return f"it is missing {sorted(missing)}"

    for name, data in sorted(built.files.items()):
        if name == PROVENANCE_FILENAME:
            continue
        if (target / name).read_bytes() != data:
            return f"{name} differs byte-for-byte"

    try:
        on_disk = json.loads((target / PROVENANCE_FILENAME).read_bytes())
    except json.JSONDecodeError as exc:
        return f"{PROVENANCE_FILENAME} is not readable JSON ({exc})"
    fresh = json.loads(built.files[PROVENANCE_FILENAME])
    for field in sorted(set(on_disk) | set(fresh)):
        if field in _NON_IDENTITY_FIELDS:
            continue
        if on_disk.get(field) != fresh.get(field):
            return f"{PROVENANCE_FILENAME} field {field!r} differs"
    return None


# -- committing (phase 3) --------------------------------------------------


def _stage_draw(staging_root: Path, scenario_key: str, built: _BuiltDraw) -> Path:
    staged = staging_root / f"{scenario_key}__draw_{built.draw_id:04d}"
    staged.mkdir(parents=True)
    for name, data in sorted(built.files.items()):
        (staged / name).write_bytes(data)
    return staged


def _commit(
    plans: list[tuple[str, _BuiltDraw, Path]],
    *,
    scenario_key: str,
    out_root: Path,
) -> None:
    """Stage every draw, then move them into place, rolling back on any failure."""
    out_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=out_root))
    done: list[tuple[Path, Path | None]] = []
    try:
        staged_by_id = {
            built.draw_id: _stage_draw(staging_root, scenario_key, built)
            for action, built, _target in plans
            if action != "kept"
        }
        for action, built, target in plans:
            if action == "kept":
                continue
            staged = staged_by_id[built.draw_id]
            target.parent.mkdir(parents=True, exist_ok=True)
            aside: Path | None = None
            if target.exists():
                # Move aside rather than delete: the old draw survives until the new one
                # is in place, so a failure here cannot destroy prior data.
                aside = staging_root / f"aside__draw_{built.draw_id:04d}"
                os.replace(target, aside)
            try:
                os.replace(staged, target)
            except BaseException:
                if aside is not None:
                    os.replace(aside, target)
                raise
            done.append((target, aside))
    except BaseException:
        for target, aside in reversed(done):
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            if aside is not None and aside.exists():
                os.replace(aside, target)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


# -- the entry point -------------------------------------------------------


def materialise(
    source_config: str | Path,
    draw_ids: Sequence[int],
    *,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    force: bool = False,
    dry_run: bool = False,
) -> list[MaterialisedDraw]:
    """Materialise *draw_ids* of one source scenario under *out_root*.

    Returns one :class:`MaterialisedDraw` per requested id, in the order requested.
    Validation, building and writing are strictly ordered (see the module docstring): a
    run that raises has written and deleted nothing.
    """
    # ---- phase 1: validation only, no filesystem mutation ----------------
    source = Path(source_config)
    if not source.is_file():
        raise FileNotFoundError(f"source sim config not found: {source}")
    ids = _checked_draw_ids(draw_ids)
    scenario_key = scenario_key_for_config(source)
    root = Path(out_root)

    cfg = json.loads(source.read_bytes())
    for key in ("dir", "roadnetFile", "flowFile"):
        if key not in cfg:
            raise ValueError(f"{source} is not a CityFlow sim config: no {key!r} key")

    flow_source = _cityflow_flow_source(source)
    if not flow_source.is_file():
        raise FileNotFoundError(
            f"the flow file {source} points at does not exist: {flow_source}. A relative "
            "'dir' is resolved against the process working directory, so run this from "
            "the repository root."
        )
    roadnet = _scenario_dir(source) / cfg["roadnetFile"]
    if not roadnet.is_file():
        raise FileNotFoundError(
            f"the roadnet {source} points at does not exist: {roadnet}"
        )

    targets = {draw_id: draw_dir(scenario_key, draw_id, out_root=root) for draw_id in ids}
    sumo, sumo_skipped_reason = _sumo_pairing(source)

    # ---- phase 2: build everything in memory -----------------------------
    randomizer = FlowRandomizer(
        flow_source,
        base_seed=DEFAULT_BASE_SEED,
        jitter_sigma_s=DEFAULT_JITTER_SIGMA_S,
        thin_p=DEFAULT_THIN_P,
        volume_scale=DEFAULT_VOLUME_SCALE,
    )
    sources = {
        "config": str(source),
        "config_sha256": _sha256_file(source),
        "flow": str(flow_source),
        "flow_sha256": randomizer.source_sha256,
        "roadnet": str(roadnet),
        "roadnet_sha256": _sha256_file(roadnet),
    }

    with tempfile.TemporaryDirectory(prefix="materialise-draws-") as scratch_name:
        scratch = Path(scratch_name)
        built_draws = [
            _build_draw(
                randomizer,
                source_config=source,
                scenario_key=scenario_key,
                draw_id=draw_id,
                target=targets[draw_id],
                sources=sources,
                sumo=sumo,
                sumo_skipped_reason=sumo_skipped_reason,
                scratch=scratch,
            )
            for draw_id in ids
        ]

    # ---- still phase 1 in spirit: classify, and refuse before writing ----
    plans: list[tuple[str, _BuiltDraw, Path]] = []
    for built in built_draws:
        target = targets[built.draw_id]
        if not target.exists():
            plans.append(("written", built, target))
            continue
        conflict = _existing_conflict(target, built)
        if conflict is None:
            plans.append(("kept", built, target))
        elif force:
            plans.append(("replaced", built, target))
        else:
            raise FileExistsError(
                f"{target} differs from the draw that would be materialised there: "
                f"{conflict}. Nothing has been written. Re-run with force=True / "
                "--force to replace it, or delete it deliberately."
            )

    if not dry_run:
        # ---- phase 3: stage, then commit ---------------------------------
        _commit(plans, scenario_key=scenario_key, out_root=root)

    records = []
    for action, built, target in plans:
        records.append(
            MaterialisedDraw(
                scenario_key=scenario_key,
                draw_id=built.draw_id,
                pool=built.pool,
                directory=target,
                config_path=target / CITYFLOW_CONFIG_FILENAME,
                flow_path=target / FLOW_FILENAME,
                sumo_path=(target / SUMO_ROUTES_FILENAME) if built.has_sumo else None,
                n_vehicles=built.n_vehicles,
                flow_sha256=built.flow_sha256,
                action="planned" if dry_run else action,
            )
        )
    return records


# -- the parity phase (P7.2a, BRIEF_35) ------------------------------------

#: The registered regime (A15(c)).  The generator accepts 0 as well, because SUMO documents
#: every non-positive value as disabling teleporting; **this phase writes -1 and nothing else**,
#: and the provenance echoes it so what ran is never inferred from a default.
PARITY_TIME_TO_TELEPORT = -1


@dataclass(frozen=True)
class _BuiltParity:
    """One draw's parity subdirectory, held in memory before anything is written."""

    draw_id: int
    pool: str
    files: dict[str, bytes]
    n_vehicles: int
    n_bound: int


def _linked_worktree_marker(path: str | Path) -> Path | None:
    """Return the ``.git`` FILE that puts *path* inside a linked worktree, else ``None``.

    ``DEFERRED`` 55: ``scenarios/draws/`` is gitignored and per-worktree, and retiring a
    worktree deleted held-out draws that merged numbers depend on.  The signal is the
    repository marker itself -- a **directory** in a main tree, a **file** in a linked
    worktree -- so this reads the filesystem rather than shelling out to git.
    """
    current = Path(path).resolve()
    for candidate in (current, *current.parents):
        marker = candidate / ".git"
        if marker.is_file():
            return marker
        if marker.is_dir():
            return None
    return None


def _checked_out_root_shape(out_root: str | Path) -> Path:
    """Return *out_root* if it is a draws ROOT, else raise. ``BRIEF_35`` Amendment B1.

    The pre-flight's item 9a: with ``out_root`` pointing at a draw directory this phase wrote a
    nested ``draw_NNNN/<scenario_key>/draw_NNNN/{four files, parity/}`` **inside a pre-existing
    draw**.  Not one byte was altered -- but the polluted draw then failed
    :func:`_validate_parent_for_parity` on every later legitimate run until someone deleted the
    nest by hand, so a mistyped ``--out-root`` naming ``draw_1000`` would have blocked a
    held-out draw.  The freedom is inherited from :func:`materialise`, where it was harmless
    because that function only ever created sibling directories; it is not harmless in a phase
    whose purpose is to write *inside* existing draws, so the guard sits here, in front of the
    only call site that does.

    Three refusals.  Two are on the path itself -- no component may be a draw directory or the
    parity directory.  The third is on **filesystem evidence**: a directory that directly holds
    ``draw_NNNN`` children *is* a scenario directory, and pointing at one would nest a second
    scenario level beside the real draws.  Evidence rather than the name, deliberately: a root
    that merely happens to be *called* ``cityflow1x1`` is legitimate and must keep working, and
    a name-only heuristic would refuse it.
    """
    resolved = Path(out_root).resolve()
    for component in resolved.parts:
        if re.fullmatch(r"draw_\d{4}", component):
            raise ValueError(
                f"refusing to use {resolved} as a draws root: its component {component!r} is a "
                "draw directory, so this phase would write a nested tree INSIDE an existing "
                "draw and every later run would then refuse that draw. The root is the "
                "directory that holds the scenario keys, e.g. scenarios/draws"
            )
        if component == PARITY_DIRNAME:
            raise ValueError(
                f"refusing to use {resolved} as a draws root: its component "
                f"{PARITY_DIRNAME!r} is a parity directory, which this phase writes, not one it "
                "writes into"
            )
    if resolved.is_dir():
        draws = sorted(
            path.name
            for path in resolved.iterdir()
            if path.is_dir() and re.fullmatch(r"draw_\d{4}", path.name)
        )
        if draws:
            raise ValueError(
                f"refusing to use {resolved} as a draws root: it holds draw directories "
                f"({draws[:3]}{' ...' if len(draws) > 3 else ''}) directly, so it is a scenario "
                "directory and the root is its parent. Pointing here would nest a second "
                "scenario level beside the real draws"
            )
    return Path(out_root)


def _checked_parity_target(target: str | Path, out_root: str | Path) -> Path:
    """Return *target* if it is a parity directory of a draw inside *out_root*, else raise.

    ``_commit``'s ``os.replace(staged, target)`` is one wrong *target* away from replacing a
    **draw** instead of a parity subdirectory, and draws 1000-1099 are what every merged
    held-out number since P4.6 resolves through.  Four properties, checked for every planned
    rename before the first one runs.

    The fourth is depth (Amendment B2, pre-flight R3): name, parent pattern and containment
    together still accept ``<out_root>/draw_0001/parity`` -- well-formed at the leaf and wrong
    in the middle, which is the same class B1 closes at the other end.  The target is exactly
    ``<out_root>/<scenario_key>/draw_NNNN/parity``: four levels, no more, no fewer.
    """
    path = Path(target)
    if path.name != PARITY_DIRNAME:
        raise ValueError(
            f"refusing to replace {path}: the target of this phase must be named "
            f"{PARITY_DIRNAME!r}, so a rename can never land on a draw directory"
        )
    if not re.fullmatch(r"draw_\d{4}", path.parent.name):
        raise ValueError(
            f"refusing to replace {path}: its parent {path.parent.name!r} is not a "
            "zero-padded draw directory"
        )
    checked = _checked_output_path(out_root, path)
    resolved = Path(path).resolve()
    if resolved.parent.parent.parent != Path(out_root).resolve():
        raise ValueError(
            f"refusing to replace {path}: it sits at the wrong depth under {out_root}. The "
            "target is exactly <out_root>/<scenario_key>/draw_NNNN/parity, and a path that is "
            "well formed at the leaf can still be wrong in the middle"
        )
    return checked


def _render_bound_routes(
    text: str,
    *,
    draw_id: int,
    vtype: Mapping[str, str] | None = None,
    insert_when_absent: bool = False,
) -> str:
    """Bind the parity ``<vType>`` onto a drawn rendering, naming the draw on failure.

    A thin seam over :func:`offline.parity.render_parity_rou_text` -- which refuses anything
    that does not match its binding mode and re-parses its own output -- so that a failure says
    *which* draw failed in a 206-draw run.  The defaults are hangzhou's: one ``<vType>`` replaced.
    """
    try:
        return parity.render_parity_rou_text(text, vtype, insert_when_absent=insert_when_absent)
    except ValueError as exc:
        raise ValueError(f"draw {draw_id}: {exc}") from exc


def _scenario_net_file(sumocfg: str | Path) -> Path:
    """The network a scenario's ``.sumocfg`` names, resolved as SUMO resolves it."""
    cfg = Path(sumocfg).resolve()
    root = ET.parse(cfg).getroot()
    node = root.find("./input/net-file")
    if node is None:
        node = root.find(".//net-file")
    raw = None if node is None else node.get("value")
    if not raw:
        raise ValueError(f"{cfg} has no <net-file value=...>, so no network can be referenced")
    net = Path(raw)
    if not net.is_absolute():
        net = cfg.parent / net
    return net.resolve()


def _parity_demand_audit(
    flow_path: str | Path, routes_path: str | Path, *, depart_offset: float
) -> dict[str, Any]:
    """CAP(E) on one draw: the CityFlow demand against the bound SUMO rendering.

    Both sides come from :mod:`offline.conversion_audit`'s extractors, which return document
    order and sort nothing -- so "the same multiset" and "the same order" stay two different
    questions.  ``depart_range_sumo`` is the range **as written in the file** (unshifted); the
    shift that makes the two comparable is recorded beside it as ``depart_offset``.
    """
    from offline.conversion_audit import demand_from_cityflow_flow, demand_from_route_file

    cityflow = demand_from_cityflow_flow(flow_path)
    raw = demand_from_route_file(routes_path)
    offset = float(depart_offset)
    sumo = [(depart - offset, route) for depart, route in raw]

    def _range(items: list[tuple[float, tuple[str, ...]]]) -> list[float]:
        return [min(t for t, _ in items), max(t for t, _ in items)] if items else []

    return {
        "n_cityflow": len(cityflow),
        "n_sumo": len(sumo),
        "multiset_equal": Counter(cityflow) == Counter(sumo),
        "n_index_aligned_equal": sum(1 for a, b in zip(cityflow, sumo) if a == b),
        "order_matches": cityflow == sumo,
        "depart_range_cityflow": _range(cityflow),
        "depart_range_sumo": _range(raw),
        "depart_offset": offset,
    }


def _validate_parent_for_parity(
    target: Path, draw_id: int, scenario: parity.ParityScenario | None = None
) -> dict[str, Any]:
    """Everything the parent must satisfy to be a legal input. Read-only; never repairs.

    Returns the parent's provenance record.  A parent that fails any check is refused, because
    the parity artifacts are a *derivation* of it: binding a rendering whose demand no longer
    matches its own record would produce a file that looks authoritative and is not.

    Two legal shapes, selected by the SCENARIO and never guessed from the directory.  The
    hangzhou shape (``scenario`` is ``None`` or has no external source) holds four files and a
    ``sumo`` record whose rendering is unbound.  The TEMPLATED shape (the scenario's SUMO side is
    external -- grid4x4) holds three files, ``"sumo": null`` and a ``sumo_skipped_reason``: there
    is no parent rendering to bind, and none may be added beside the parent's files.
    """
    templated = scenario is not None and scenario.external is not None
    if not target.is_dir():
        raise FileNotFoundError(f"draw {draw_id}: {target} does not exist")

    entries = sorted(target.iterdir())
    files = {path.name for path in entries if path.is_file()}
    expected = {FLOW_FILENAME, CITYFLOW_CONFIG_FILENAME, PROVENANCE_FILENAME}
    if not templated:
        expected.add(SUMO_ROUTES_FILENAME)
    if files != expected and templated:
        raise ValueError(
            f"draw {draw_id}: {target} holds {sorted(files)}, not the three files a materialised "
            f"draw of {scenario.key} holds ({sorted(expected)}): this scenario's parent carries "  # type: ignore[union-attr]
            "no SUMO rendering, and its bound routes are rendered inside parity/"
        )
    if files != expected:
        raise ValueError(
            f"draw {draw_id}: {target} holds {sorted(files)}, not the four files a "
            f"materialised draw holds ({sorted(expected)})"
        )
    foreign = {path.name for path in entries if path.is_dir()} - {PARITY_DIRNAME}
    if foreign:
        raise ValueError(
            f"draw {draw_id}: {target} holds unexpected subdirector{'y' if len(foreign) == 1 else 'ies'} "
            f"{sorted(foreign)}; refusing to derive parity artifacts from a draw directory "
            "whose contents this tool did not write"
        )

    record = json.loads((target / PROVENANCE_FILENAME).read_bytes())
    if record.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"draw {draw_id}: provenance format is {record.get('format_version')!r}, "
            f"not {FORMAT_VERSION!r}"
        )
    for name, digest in sorted(record.get("files", {}).items()):
        actual = _sha256_file(target / name)
        if actual != digest:
            raise ValueError(
                f"draw {draw_id}: {name} does not match the digest in its own provenance "
                f"({actual[:12]}... against {str(digest)[:12]}...); the parent is not the "
                "draw its record describes"
            )

    sumo = record.get("sumo")
    if templated:
        if sumo is not None or not record.get("sumo_skipped_reason"):
            raise ValueError(
                f"draw {draw_id}: the parent's provenance records sumo={sumo!r} and "
                f"sumo_skipped_reason={record.get('sumo_skipped_reason')!r}; a parent of "
                f"{scenario.key} records NO SUMO rendering and the reason it has none"  # type: ignore[union-attr]
            )
        disagreements = parity.flow_json_disagreements(target / FLOW_FILENAME, scenario)
        if disagreements:
            raise ValueError(
                f"draw {draw_id}: the registered parity table disagrees with the drawn "
                f"{FLOW_FILENAME}: " + "; ".join(disagreements)
            )
        return record
    if not sumo:
        raise ValueError(
            f"draw {draw_id}: the parent carries no SUMO rendering "
            f"({record.get('sumo_skipped_reason')}), so there is nothing to bind"
        )
    if sumo.get("vtype_bound") is not False:
        raise ValueError(
            f"draw {draw_id}: the parent's provenance says sumo.vtype_bound is "
            f"{sumo.get('vtype_bound')!r}; a rendering that is already bound is not this "
            "phase's input, and binding twice is what render_parity_rou_text refuses"
        )
    if _SUMO_CAVEAT_UNBOUND not in str(sumo.get("caveat", "")):
        raise ValueError(
            f"draw {draw_id}: the parent's SUMO caveat is not the unbound one, so its "
            "rendering is not the file this phase expects to bind"
        )

    disagreements = parity.flow_json_disagreements(target / FLOW_FILENAME)
    if disagreements:
        raise ValueError(
            f"draw {draw_id}: the declared parity table disagrees with the drawn "
            f"{FLOW_FILENAME}: " + "; ".join(disagreements)
        )
    return record


def _build_parity(
    *,
    draw_id: int,
    parent_dir: Path,
    parent_record: dict[str, Any],
    net_path: Path,
    net_resolved: str,
    target: Path,
    depart_offset: float,
    scratch: Path,
    vtype: Mapping[str, str] | None = None,
    template: Mapping[str, Any] | None = None,
    scenario: parity.ParityScenario | None = None,
) -> _BuiltParity:
    """Render one draw's parity artifacts entirely in memory.

    The rendering helpers are pure text functions, so the only reason a file is written here is
    that :func:`offline.parity.vtype_binding_report` and the demand extractors read paths.  That
    write goes to an OS temp directory: nothing under ``out_root`` is touched while the run can
    still fail.

    ``template`` selects the TEMPLATED shape (``materialised-draw-parity/1.1``).  The parent then
    carries no SUMO rendering, so the unbound rendering is produced HERE, in scratch, by the same
    :meth:`offline.flow_randomizer.FlowRandomizer.render_sumo` the hangzhou parents were rendered
    with -- from the parent's own ``flow.json`` entries, so vehicle ids stay ``0..n-1`` in
    ``startTime`` order and ``depart`` stays ``startTime + <begin>`` at two decimals, the alignment
    convention of the module docstring -- and the parity type is INSERTED rather than replaced.
    """
    templated = template is not None
    if template is not None:
        entries = json.loads((parent_dir / FLOW_FILENAME).read_bytes())
        unbound = scratch / f"draw_{draw_id:04d}_unbound_{SUMO_ROUTES_FILENAME}"
        template["randomizer"].render_sumo(
            entries,
            unbound,
            template_rou_path=template["path"],
            depart_offset=depart_offset,
        )
        source_text = unbound.read_text(encoding="utf-8")
    else:
        source_text = (parent_dir / SUMO_ROUTES_FILENAME).read_text(encoding="utf-8")
    # The branch is keyed on the SCENARIO, never on two tables happening to be equal (BRIEF_39
    # Amendment A.1-2): a future scenario whose derived table equalled hangzhou's must not
    # silently take hangzhou's path.
    hangzhou = scenario is None or scenario.key == parity.HZ1X1_SCENARIO.key
    if hangzhou:
        if templated:
            raise ValueError(
                f"draw {draw_id}: the hangzhou scenario binds its parent's own rendering; a route "
                "template was supplied for it, which is the templated shape of another scenario"
            )
        if vtype is not None and dict(vtype) != parity.parity_vtype_attributes():
            raise ValueError(
                f"draw {draw_id}: the table derived from the hangzhou flow block ({dict(vtype)}) "
                "is not byte-equal to the registered one; emitting it would move every hangzhou "
                "parity file, so PARITY_CONTRACT_VERSION would have to move with it"
            )
        # The hangzhou call, LITERALLY as it was before P7.3d: same seam, same two arguments, so
        # the per-scenario code changes nothing a caller, or a test substituting this seam
        # (tests/test_materialise_parity.py), could observe.
        bound_text = _render_bound_routes(source_text, draw_id=draw_id)
    else:
        bound_text = _render_bound_routes(
            source_text, draw_id=draw_id, vtype=vtype, insert_when_absent=templated
        )

    staged_routes = scratch / f"draw_{draw_id:04d}_{PARITY_ROUTES_FILENAME}"
    staged_routes.write_text(bound_text, encoding="utf-8")

    report = parity.vtype_binding_report(staged_routes)
    if not parity.binding_is_complete(report, vtype):
        raise ValueError(
            f"draw {draw_id}: the bound rendering does not satisfy the parity contract "
            f"({report.vehicles_with_type} of {report.vehicle_count} vehicles typed, "
            f"vTypes {list(report.vtype_ids)})"
        )

    audit = _parity_demand_audit(
        parent_dir / FLOW_FILENAME, staged_routes, depart_offset=depart_offset
    )
    if not audit["multiset_equal"]:
        raise ValueError(
            f"draw {draw_id}: the bound rendering's demand is not the parent's demand "
            f"({audit['n_sumo']} SUMO vehicles against {audit['n_cityflow']} CityFlow "
            "entries, multisets differ); A14(E) requires it exact per vehicle"
        )
    if report.vehicle_count != audit["n_cityflow"]:
        raise ValueError(
            f"draw {draw_id}: the bound rendering holds {report.vehicle_count} vehicles "
            f"against the parent's {audit['n_cityflow']} flow entries"
        )

    net_reference = os.path.relpath(net_path, target.resolve())
    cfg_text = parity.render_parity_sumocfg_text(
        net_reference,
        PARITY_ROUTES_FILENAME,
        time_to_teleport=PARITY_TIME_TO_TELEPORT,
    )

    files: dict[str, bytes] = {
        PARITY_ROUTES_FILENAME: bound_text.encode("utf-8"),
        PARITY_SUMOCFG_FILENAME: cfg_text.encode("utf-8"),
    }
    attributes = parity.parity_vtype_attributes() if vtype is None else dict(vtype)
    parent_block: dict[str, Any] = {
        "flow_sha256": _sha256_file(parent_dir / FLOW_FILENAME),
        "routes_sha256": (
            None if templated else _sha256_file(parent_dir / SUMO_ROUTES_FILENAME)
        ),
        "provenance_sha256": _sha256_file(parent_dir / PROVENANCE_FILENAME),
    }
    if templated:
        # 1.1: the field a 1.0 reader expects to be a digest is null, and the reason is beside it.
        parent_block["routes_absent_reason"] = str(parent_record["sumo_skipped_reason"])
    record: dict[str, Any] = {
        "format_version": PARITY_FORMAT_VERSION_TEMPLATED if templated else PARITY_FORMAT_VERSION,
        "scenario_key": parent_record["scenario_key"],
        "draw_id": parent_record["draw_id"],
        "pool": parent_record["pool"],
        "parent": parent_block,
        "parity_contract_version": parity.PARITY_CONTRACT_VERSION,
        "vtype_id": attributes["id"],
        "vtype_attributes": attributes,
        "files": {name: _sha256(data) for name, data in sorted(files.items())},
        "net": {
            "reference": net_reference,
            "resolved": net_resolved,
            "sha256": _sha256_file(net_path),
        },
        "sumocfg": {
            "begin": 0,
            "end": parity.SUMO_END_SECONDS,
            "time_to_teleport": PARITY_TIME_TO_TELEPORT,
        },
        "demand_audit": audit,
        "n_vehicles": report.vehicle_count,
        "n_bound": report.vehicles_with_type,
    }
    if template is not None:
        record["route_template"] = dict(template["record"])
    commit, dirty = _git_commit()
    record["git_commit"] = commit
    record["git_dirty"] = dirty

    files[PROVENANCE_FILENAME] = _provenance_bytes(record)
    return _BuiltParity(
        draw_id=draw_id,
        pool=str(parent_record["pool"]),
        files=files,
        n_vehicles=report.vehicle_count,
        n_bound=report.vehicles_with_type,
    )


def _existing_parity_conflict(target: Path, built: _BuiltParity) -> str | None:
    """Why *target* differs from *built*, or ``None`` when it is identical.

    The same shape as :func:`_existing_conflict` and for the same reason: rendered files
    byte-for-byte, then the record field by field with :data:`_NON_IDENTITY_FIELDS` excluded, so
    a later commit does not make a correct artifact look stale.  It returns on the FIRST
    mismatch, so the absence of a later complaint is the loop stopping, not agreement.
    """
    expected = set(built.files)
    present = {path.name for path in target.iterdir() if path.is_file()}
    extra = present - expected
    if extra:
        return f"it holds unexpected file(s) {sorted(extra)}"
    missing = expected - present
    if missing:
        return f"it is missing {sorted(missing)}"
    for name, data in sorted(built.files.items()):
        if name == PROVENANCE_FILENAME:
            continue
        if (target / name).read_bytes() != data:
            return f"{name} differs byte-for-byte"
    try:
        on_disk = json.loads((target / PROVENANCE_FILENAME).read_bytes())
    except json.JSONDecodeError as exc:
        return f"{PROVENANCE_FILENAME} is not readable JSON ({exc})"
    fresh = json.loads(built.files[PROVENANCE_FILENAME])
    for field in sorted(set(on_disk) | set(fresh)):
        if field in _NON_IDENTITY_FIELDS:
            continue
        if on_disk.get(field) != fresh.get(field):
            return f"{PROVENANCE_FILENAME} field {field!r} differs"
    return None


def _commit_parity(
    plans: list[tuple[str, _BuiltParity, Path]], *, out_root: Path
) -> None:
    """Stage every parity directory, then move them into place, rolling back on any failure.

    Mirrors :func:`_commit`, with one addition and one subtraction: every target passes
    :func:`_checked_parity_target` **before the first rename**, and no parent directory is ever
    created -- this phase writes inside draws that already exist and must never invent one.

    ⚠️ **Accepted residual, inherited verbatim from :func:`_commit`** (``DEFERRED`` 79, which
    names ``_commit`` as its owner; P7.2a's pre-flight R1/R2, found by reading and not
    reproducible in its sandbox): a **second** fault occurring *during* rollback -- a partial
    ``shutil.rmtree`` of the new target, or the ``aside -> target`` restore itself raising --
    can leave the moved-aside original under ``staging_root``, where the ``finally`` deletes it.
    One fault is safe, and both injected single faults were verified to restore the original
    sha256-identically; two are not, and fixing it means moving the aside directory out of
    ``staging_root``, which is a change to the shared pattern the 106 existing parents were
    written through and therefore not this task's to make.
    """
    for _action, _built, target in plans:
        _checked_parity_target(target, out_root)

    staging_root = Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=out_root))
    done: list[tuple[Path, Path | None]] = []
    try:
        staged_by_id = {}
        for _action, built, _target in plans:
            staged = staging_root / f"draw_{built.draw_id:04d}__{PARITY_DIRNAME}"
            staged.mkdir(parents=True)
            for name, data in sorted(built.files.items()):
                (staged / name).write_bytes(data)
            staged_by_id[built.draw_id] = staged

        for _action, built, target in plans:
            if not target.parent.is_dir():
                raise FileNotFoundError(
                    f"the draw directory {target.parent} is gone; this phase never creates one"
                )
            staged = staged_by_id[built.draw_id]
            aside: Path | None = None
            if target.exists():
                # Move aside rather than delete: the old directory survives until the new one
                # is in place, so a failure here cannot destroy prior data.
                aside = staging_root / f"aside__draw_{built.draw_id:04d}"
                os.replace(target, aside)
            try:
                os.replace(staged, target)
            except BaseException:
                if aside is not None:
                    os.replace(aside, target)
                raise
            done.append((target, aside))
    except BaseException:
        for target, aside in reversed(done):
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            if aside is not None and aside.exists():
                os.replace(aside, target)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def materialise_parity(
    source_config: str | Path,
    draw_ids: Sequence[int],
    *,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    force: bool = False,
    dry_run: bool = False,
    allow_worktree: bool = False,
) -> list[ParityResult]:
    """Add ``draw_NNNN/parity/`` to every requested draw of one scenario.

    Phase order (``BRIEF_35`` Amendment A6): scenario-level refusals, then **every parent that
    already exists is validated**, then the missing parents are materialised through
    :func:`materialise`, then the new parents are validated, then everything is built in memory,
    classified, staged and committed.  The consequence is the point of the order: a refusal
    caused by the tree, the scenario or an existing parent happens **before any write**.

    ``force`` replaces a differing ``parity/`` subdirectory only.  It is never forwarded to
    :func:`materialise`, so no parent can be replaced through this entry point.
    """
    # ---- phase 0: scenario-level refusals, no filesystem mutation --------
    source = Path(source_config)
    if not source.is_file():
        raise FileNotFoundError(f"source sim config not found: {source}")
    ids = _checked_draw_ids(draw_ids)
    scenario_key = scenario_key_for_config(source)
    root = _checked_out_root_shape(out_root)

    marker = _linked_worktree_marker(root)
    if marker is not None and not allow_worktree:
        raise ValueError(
            f"refusing to materialise into {Path(root).resolve()}: it is inside a LINKED "
            f"WORKTREE ({marker} is a file, not a directory). scenarios/draws/ is gitignored "
            "and per-worktree, and retiring a worktree has already deleted held-out draws that "
            "merged numbers depend on (DEFERRED 55), so draws are materialised in the MAIN "
            "tree. Pass allow_worktree=True / --allow-worktree if you mean it."
        )

    # The parity table is PER SCENARIO (P7.3d).  A key with no registered scenario falls back to
    # the hangzhou table, which is what every scenario was checked against before P7.3d -- and a
    # flow block that disagrees with it is refused below, exactly as before.
    scenario = parity.PARITY_SCENARIOS.get(scenario_key, parity.HZ1X1_SCENARIO)
    sumo, sumo_skipped_reason = _sumo_pairing(source)
    resolved_external: parity.ResolvedExternalSource | None = None
    if scenario.external is None:
        if sumo is None:
            raise ValueError(
                f"{source} has no usable SUMO pairing, so there is nothing to bind: "
                f"{sumo_skipped_reason}"
            )
    else:
        if sumo is not None:
            raise ValueError(
                f"{source} pairs with {sumo['template_rou']} inside the repository, but "
                f"{scenario.key}'s registered SUMO side is external ({scenario.external.env_var}); "
                "two SUMO sides for one scenario is an ambiguity this tool does not resolve"
            )
        try:
            resolved_external = parity.resolve_external_source(scenario)
        except (ValueError, FileNotFoundError) as exc:
            raise ValueError(
                f"{source} has no usable SUMO pairing inside the repository "
                f"({sumo_skipped_reason}), and its registered external SUMO side is "
                f"unavailable: {exc}"
            ) from exc

    source_flow = _cityflow_flow_source(source)
    disagreements = parity.flow_json_disagreements(source_flow, scenario)
    if disagreements:
        raise ValueError(
            f"the declared parity table disagrees with {source_flow}: "
            + "; ".join(disagreements)
        )
    # DERIVED from the flow block (BRIEF_39 C1); for hangzhou the strings are byte-equal to the
    # registered table, which is what keeps every pre-P7.3d parity directory `kept`.
    vtype = parity.derived_vtype_attributes(scenario, source_flow)

    if resolved_external is None:
        assert sumo is not None
        net_path = _scenario_net_file(sumo["sumocfg"])
        if not net_path.is_file():
            raise FileNotFoundError(
                f"{sumo['sumocfg']} names a network that does not exist: {net_path}"
            )
        net_resolved = os.path.relpath(net_path, _scenario_dir(source).parent.parent)
        depart_offset = float(sumo["depart_offset"])
        template_record: dict[str, Any] | None = None
    else:
        external = scenario.external
        assert external is not None
        net_path = resolved_external.net
        # Recorded relative to the candidates ROOT, never absolutely: the artifact must read the
        # same on a machine that keeps the clone elsewhere (the convention of
        # docs/data/p7_1_conversion_audit.json).
        net_resolved = f"<candidates>/{external.relative_dir}/{external.net_name}"
        depart_offset = float(sumo_begin_from_sumocfg(resolved_external.sumocfg))
        template_record = {
            "archive": f"<candidates>/{external.relative_dir}/{external.routes_archive}",
            "member": external.routes_member,
            "member_sha256": resolved_external.routes_member_sha256,
            "declares_vtype": scenario.template_declares_vtype,
            "sumocfg": f"<candidates>/{external.relative_dir}/{external.sumocfg_name}",
            "sumocfg_sha256": _sha256_file(resolved_external.sumocfg),
            "depart_offset": depart_offset,
            "licence": external.licence,
        }

    targets = {draw_id: parity_dir(scenario_key, draw_id, out_root=root) for draw_id in ids}
    for target in targets.values():
        _checked_parity_target(target, root)

    # ---- phase 2a: validate every parent that ALREADY exists (A6) -------
    existed = {
        draw_id
        for draw_id in ids
        if draw_dir(scenario_key, draw_id, out_root=root).is_dir()
    }
    parents: dict[int, dict[str, Any]] = {
        draw_id: _validate_parent_for_parity(
            draw_dir(scenario_key, draw_id, out_root=root), draw_id, scenario
        )
        for draw_id in ids
        if draw_id in existed
    }

    # ---- phase 1: the missing parents, through the existing entry point --
    records = materialise(source, ids, out_root=root, force=False, dry_run=dry_run)
    parent_actions = {record.draw_id: record.action for record in records}

    # ---- phase 2b: validate the parents phase 1 just created ------------
    for draw_id in ids:
        if draw_id in parents:
            continue
        target_parent = draw_dir(scenario_key, draw_id, out_root=root)
        if dry_run and not target_parent.is_dir():
            continue  # a dry run wrote nothing, so there is nothing to validate yet
        parents[draw_id] = _validate_parent_for_parity(target_parent, draw_id, scenario)

    # ---- phase 3: build every byte in memory ----------------------------
    built_by_id: dict[int, _BuiltParity] = {}
    with tempfile.TemporaryDirectory(prefix="materialise-parity-") as scratch_name:
        scratch = Path(scratch_name)
        template: dict[str, Any] | None = None
        if resolved_external is not None and template_record is not None:
            # The route template is read IN PLACE from the archive and handed to render_sumo --
            # which reads a path -- through this OS temp directory.  It is outside every tree and
            # is removed with the scratch directory; no RESCO file is copied into the repository
            # or the draws tree (BRIEF_39 section 2, Amendment A8).
            template_path = scratch / f"template_{resolved_external.routes_member}"
            template_path.write_text(
                parity.read_route_template_text(resolved_external), encoding="utf-8"
            )
            template = {
                "path": template_path,
                "randomizer": FlowRandomizer(source_flow),
                "record": template_record,
            }
        for draw_id in ids:
            if draw_id not in parents:
                continue
            built_by_id[draw_id] = _build_parity(
                draw_id=draw_id,
                parent_dir=draw_dir(scenario_key, draw_id, out_root=root),
                parent_record=parents[draw_id],
                net_path=net_path,
                net_resolved=net_resolved,
                target=targets[draw_id],
                depart_offset=depart_offset,
                scratch=scratch,
                vtype=vtype,
                template=template,
                scenario=scenario,
            )

    # ---- phase 4: classify, and refuse before writing -------------------
    plans: list[tuple[str, _BuiltParity | None, Path]] = []
    for draw_id in ids:
        target = targets[draw_id]
        built = built_by_id.get(draw_id)
        if built is None:  # dry run, parent not there yet
            plans.append(("written", None, target))
            continue
        if not target.exists():
            plans.append(("written", built, target))
            continue
        conflict = _existing_parity_conflict(target, built)
        if conflict is None:
            plans.append(("kept", built, target))
        elif force:
            plans.append(("replaced", built, target))
        elif dry_run:
            plans.append(("differs", built, target))
        else:
            raise FileExistsError(
                f"the parity directory {target} differs (refused): {conflict}. Nothing has "
                "been written. Re-run with force=True / --force to replace it -- which "
                "replaces the parity subdirectory only, never the draw."
            )

    # ---- phase 5: stage, then commit ------------------------------------
    if not dry_run:
        writable = [
            (action, built, target)
            for action, built, target in plans
            if action in {"written", "replaced"} and built is not None
        ]
        if writable:
            _commit_parity(writable, out_root=root)

    results = []
    for (action, built, target), draw_id in zip(plans, ids):
        results.append(
            ParityResult(
                scenario_key=scenario_key,
                draw_id=draw_id,
                pool=classify_draw_pool(draw_id),
                directory=target,
                routes_path=target / PARITY_ROUTES_FILENAME,
                sumocfg_path=target / PARITY_SUMOCFG_FILENAME,
                provenance_path=target / PROVENANCE_FILENAME,
                n_vehicles=0 if built is None else built.n_vehicles,
                n_bound=0 if built is None else built.n_bound,
                action="planned" if dry_run else action,
                parent_action=(
                    ("kept" if draw_id in existed else "planned")
                    if dry_run
                    else parent_actions[draw_id]
                ),
                planned_action=action,
            )
        )
    return results


#: Which field of a :class:`offline.rtg_calibration.ProbeEpisode` carries each field an artifact may
#: record.  Every entry is an identity of DEFINITION, not an observed agreement, except
#: ``episode_reward`` -- see :func:`_refuse_unjustified_mappings`.
_ARTIFACT_FIELD_SOURCES: dict[str, str] = {
    "att_horizon": "att_horizon",
    "horizon_vehicle_count": "horizon_vehicle_count",
    "decisions": "decisions",
    "local_return": "local_return",
    "local_return_from_lanes": "local_return_from_lanes",
    "episode_reward": "local_return",
}


def _refuse_unjustified_mappings(fields: Sequence[str], env_settings: Mapping[str, Any]) -> None:
    """Refuse any comparison whose field mapping is not an identity under these settings.

    ``episode_reward`` is the only conditional one.  ``horizon_rollout`` records the env's SCALAR
    reward while ``run_probe`` records the single intersection's LOCAL return, and the two are the
    same number only when ``global_reward_weight == 0.0`` -- then the scalar reward *is* the local
    reward -- and only on one intersection, which ``run_probe`` enforces by raising.  Measured on
    ``docs/data/p4_heldout_thresholds.json``: draws 1000 and 1001 reproduce ``-16428.0`` and
    ``-19021.0`` under ``==``.  Under a non-zero weight they are different quantities, and a
    comparison that happened to pass would be a coincidence waiting to break, so it is refused.
    """
    if "episode_reward" not in fields:
        return
    weight = env_settings.get("global_reward_weight")
    if weight != 0.0:
        raise ValueError(
            f"the artifact records episode_reward under global_reward_weight {weight!r}; the probe "
            "records the single intersection's local return, and the two are the same quantity "
            "only at global_reward_weight 0.0, where the scalar reward IS the local reward. "
            "Refusing to compare two different quantities and call the agreement a check"
        )


def verify_against_artifact(
    artifact_path: str | Path,
    *,
    arm: str | None = None,
    scenario_key: str,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    draw_ids: Sequence[int] | None = None,
    scenario_id: str | None = None,
) -> list[ProbeCheck]:
    """Re-run the probe on materialised draws and compare **every field the artifact records**.

    ``BRIEF_37`` §3.0, generalising :func:`verify_p4_3_probe` off P4.3's artifact so ``DEFERRED``
    80's gate can run against ``docs/data/p4_heldout_thresholds.json`` too: the held-out SUMO draws
    are derived from these CityFlow parents, and their pedigree is proved here rather than
    inherited.

    **Every recorded field is compared, not a chosen subset.**  The fields come from the artifact's
    own rows, so an artifact that records more is checked on more; a field the probe cannot produce
    is a refusal, never a silent skip -- an unchecked field in a gate is the gate's worst failure
    mode, because it reports 100/100 while comparing nothing that moved.

    Nothing is written.  The loop is :func:`offline.rtg_calibration.run_probe` -- P4.3's own -- and
    ``env_settings`` and ``engine_seed`` come from the artifact rather than being retyped.
    """
    from offline.rtg_calibration import run_probe

    path = Path(artifact_path)
    artifact = json.loads(path.read_bytes())
    rows = list(artifact["episodes"])
    if arm is not None:
        rows = [row for row in rows if str(row.get("arm")) == str(arm)]
        if not rows:
            raise ValueError(
                f"{path.name} records no episode with arm {arm!r}; the arms present are "
                f"{sorted({str(r.get('arm')) for r in artifact['episodes']})}"
            )
    expected_by_id = {int(row["draw_id"]): row for row in rows}
    if len(expected_by_id) != len(rows):
        raise ValueError(
            f"{path.name} records more than one episode per draw for arm {arm!r}; the comparison "
            "would silently use whichever came last"
        )

    ids = _checked_draw_ids(
        [int(d) for d in (sorted(expected_by_id) if draw_ids is None else draw_ids)]
    )
    unknown = [draw_id for draw_id in ids if draw_id not in expected_by_id]
    if unknown:
        raise ValueError(
            f"{path.name} records no episode for draw(s) {unknown}, so there is nothing to "
            "compare them against"
        )

    # The fields to compare are the artifact's, minus the bookkeeping columns that are not
    # measurements. A recorded field with no probe counterpart stops the gate.
    bookkeeping = {"draw_id", "arm", "seed"}
    fields = [key for key in sorted(expected_by_id[ids[0]]) if key not in bookkeeping]
    unsupported = [key for key in fields if key not in _ARTIFACT_FIELD_SOURCES]
    if unsupported:
        raise ValueError(
            f"{path.name} records {unsupported}, which the probe does not produce; refusing to "
            "report a comparison that silently skips a recorded field"
        )
    env_settings = artifact["env_settings"]
    _refuse_unjustified_mappings(fields, env_settings)

    episodes = run_probe(
        draw_ids=ids,
        config_for_draw=lambda draw_id: draw_config_path(
            scenario_key, draw_id, out_root=out_root
        ),
        env_settings=env_settings,
        scenario_id=str(scenario_id if scenario_id is not None else artifact["scenario_id"]),
        engine_seed=int(artifact["engine_seed"]),
    )

    checks: list[ProbeCheck] = []
    for episode in episodes:
        recorded = expected_by_id[episode.draw_id]
        observed = {
            key: getattr(episode, _ARTIFACT_FIELD_SOURCES[key]) for key in fields
        }
        expected = {key: recorded[key] for key in fields}
        differing = tuple(key for key in fields if observed[key] != expected[key])
        checks.append(
            ProbeCheck(
                draw_id=episode.draw_id,
                matches=not differing,
                observed=observed,
                expected=expected,
                differing=differing,
            )
        )
    return checks


def verify_p4_3_probe(
    source_config: str | Path,
    draw_ids: Sequence[int] | None = None,
    *,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    probe_artifact: str | Path = P4_3_PROBE_ARTIFACT,
) -> list[ProbeCheck]:
    """Re-run P4.3's in-domain probe on materialised draws and compare it to the artifact.

    ``DEFERRED`` 55: a ``draw_ids`` list inside a committed JSON reads as self-contained data
    and is in fact a **pointer into a gitignored directory**, so an id is not evidence that the
    demand behind it still exists.  This is what turns the id back into evidence -- the probe
    P4.3 ran, on the regenerated draws, compared under ``==``.

    Since ``BRIEF_37`` §3.0 this is a thin call to :func:`verify_against_artifact`, which does the
    same work for any artifact; the behaviour here is unchanged and a test asserts it field for
    field.  P4.3's artifact carries no ``arm`` column -- it is MaxPressure throughout -- and does
    carry ``draw_ids`` and ``scenario_id``, which this reads exactly as before.
    """
    artifact = json.loads(Path(probe_artifact).read_bytes())
    return verify_against_artifact(
        probe_artifact,
        arm=None,
        scenario_key=scenario_key_for_config(source_config),
        out_root=out_root,
        draw_ids=(artifact["draw_ids"] if draw_ids is None else draw_ids),
        scenario_id=str(artifact["scenario_id"]),
    )



# -- CLI -------------------------------------------------------------------


# -- gate G1: CAP(E) re-run on the rendered parity files (P7.3d, BRIEF_39 C1) ----


#: The E fields an artifact row carries.  ``audit_pair`` also returns the two file paths and the
#: route file's vType facts; the paths are machine-local and the binding is reported separately,
#: from :func:`offline.parity.vtype_binding_report`, so neither is copied.
_CAP_E_FIELDS: tuple[str, ...] = (
    "n_cityflow",
    "n_sumo",
    "counts_equal",
    "multiset_equal",
    "n_index_aligned_equal",
    "order_matches",
    "n_only_in_cityflow",
    "n_only_in_sumo",
    "depart_range_cityflow",
    "depart_range_sumo",
)


def _git_provenance_strict() -> tuple[str, bool]:
    """``(commit, dirty)`` measured from this module's tree; a failure of either RAISES.

    :func:`_git_commit` fails OPEN -- it reports a clean tree whenever ``git status`` cannot run
    (``BRIEF_37`` Amendment J1).  Every parity record keeps that helper so records written before
    P7.3d stay comparable, but gate G1's artifact is a verdict a registration rests on, and an
    unmeasured tree must not read as a clean one there.
    """
    cwd = str(Path(__file__).resolve().parent)
    outputs = []
    for arguments in (["rev-parse", "HEAD"], ["status", "--porcelain"]):
        result = subprocess.run(
            ["git", *arguments], cwd=cwd, capture_output=True, text=True, timeout=30, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(arguments)} failed in {cwd} ({result.stderr.strip()!r}); the "
                "provenance of the CAP(E) artifact cannot be measured, and it is not guessed"
            )
        outputs.append(result.stdout)
    return outputs[0].strip(), bool(outputs[1].strip())


def _pedigree_digests(
    pedigree_artifact: str | Path, scenario_key: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """The committed per-draw ``flow.json`` digests for *scenario_key*, and the artifact's identity.

    Read from ``draw_restoration`` of P8.4a's admission artifact, which recorded the sha256 of all
    100 held-out parents before any SUMO work existed -- so "the draws on disk are the draws every
    merged held-out number was computed on" is a digest comparison, not a re-collection.
    """
    path = Path(pedigree_artifact)
    data = json.loads(path.read_bytes())
    blocks = [
        block
        for block in data.get("draw_restoration", {}).values()
        if isinstance(block, dict) and block.get("scenario_key") == scenario_key
    ]
    if len(blocks) != 1:
        raise ValueError(
            f"{path} carries {len(blocks)} draw_restoration blocks for {scenario_key!r}; the "
            "pedigree check needs exactly one"
        )
    digests = {str(draw): str(digest) for draw, digest in blocks[0]["flow_sha256"].items()}
    identity = {"artifact": path.name, "artifact_sha256": _sha256_file(path)}
    return identity, digests


def cap_e_report(
    source_config: str | Path,
    draw_ids: Sequence[int],
    *,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    pedigree_artifact: str | Path | None = None,
) -> dict[str, Any]:
    """A15(g)'s condition as an artifact: CAP(E) and the vType binding on every requested draw.

    **Format version** ``p7.3d-cap-e/1.0``.  **Alignment convention:** E compares, per draw, the
    parent's CityFlow ``flow.json`` against the rendered ``parity/routes.rou.xml`` on the key
    ``(depart, route)`` -- CityFlow's ``startTime`` against SUMO's ``depart`` UNSHIFTED, which is
    why a scenario whose ``<begin>`` is not 0 is refused rather than mis-audited -- as a MULTISET
    and index-aligned, through :func:`offline.conversion_audit.audit_pair` (the function A15(g)'s
    verdict was produced with), never through this module's own extractor.

    Read-only: nothing under ``out_root`` is written.  A draw with no parity directory RAISES --
    there is nothing to audit, which is not the same as a draw that fails.  A draw that fails is a
    ROW with ``exact: false`` and its reasons, because G1 is decided by the artifact and a failed
    condition is a registered outcome (``PREREGISTRATION`` A15(g), A20(f)).

    Per draw, ``exact`` requires ALL of: both rendered files hash to what their provenance
    records, and the parent's ``flow.json`` to what the parity record says it was derived from;
    E exact in count, multiset and order; every vehicle bound to the parity type, whose attributes
    -- read from the rendered file -- equal the ones derived from the scenario's flow block; the
    teleport-free regime and an ``end`` above the env horizon; and, where a pedigree artifact
    covers the draw, the parent's digest equal to the committed one.
    """
    from offline.conversion_audit import audit_pair

    source = Path(source_config)
    if not source.is_file():
        raise FileNotFoundError(f"source sim config not found: {source}")
    ids = _checked_draw_ids(draw_ids)
    scenario_key = scenario_key_for_config(source)
    scenario = parity.scenario_for_key(scenario_key)
    root = Path(out_root)

    cfg = json.loads(source.read_bytes())
    roadnet = _scenario_dir(source) / cfg["roadnetFile"]
    if not roadnet.is_file():
        raise FileNotFoundError(f"the roadnet {source} points at does not exist: {roadnet}")
    registered = parity.parity_vtype_attributes(scenario)
    derived = parity.derived_vtype_attributes(scenario, _cityflow_flow_source(source))

    resco: dict[str, str] | None = None
    if scenario.external is not None:
        resolved = parity.resolve_external_source(scenario)
        net_path = resolved.net
        resco = {
            "net_sha256": resolved.net_sha256,
            "route_member_sha256": resolved.routes_member_sha256,
        }
    else:
        sumo, reason = _sumo_pairing(source)
        if sumo is None:
            raise ValueError(f"{source} has no usable SUMO pairing: {reason}")
        net_path = _scenario_net_file(sumo["sumocfg"])

    pedigree_identity: dict[str, Any] | None = None
    committed: dict[str, str] = {}
    if pedigree_artifact is not None:
        pedigree_identity, committed = _pedigree_digests(pedigree_artifact, scenario_key)

    # ---- every refusal that means "there is nothing to audit", before any row is built ----
    for draw_id in ids:
        parent = draw_dir(scenario_key, draw_id, out_root=root)
        target = parity_dir(scenario_key, draw_id, out_root=root)
        for path in (
            parent / FLOW_FILENAME,
            target / PARITY_ROUTES_FILENAME,
            target / PARITY_SUMOCFG_FILENAME,
            target / PROVENANCE_FILENAME,
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"draw {draw_id}: {path} is absent, so there is nothing to audit; render the "
                    "parity directory first (--parity)"
                )

    rows: list[dict[str, Any]] = []
    structure: dict[str, Any] | None = None
    pedigree_rows: dict[str, dict[str, Any]] = {}
    not_covered: list[int] = []
    for draw_id in ids:
        parent = draw_dir(scenario_key, draw_id, out_root=root)
        target = parity_dir(scenario_key, draw_id, out_root=root)
        record = load_parity_provenance(scenario_key, draw_id, out_root=root)
        reasons: list[str] = []

        offset = float(record["demand_audit"]["depart_offset"])
        if offset != 0.0:
            raise ValueError(
                f"draw {draw_id}: the rendering shifts departures by {offset} s; audit_pair "
                "compares UNSHIFTED (depart, route) keys, so this report would mis-audit it"
            )

        flow_sha256 = _sha256_file(parent / FLOW_FILENAME)
        files = {
            PARITY_ROUTES_FILENAME: _sha256_file(target / PARITY_ROUTES_FILENAME),
            PARITY_SUMOCFG_FILENAME: _sha256_file(target / PARITY_SUMOCFG_FILENAME),
        }
        for name, digest in sorted(files.items()):
            recorded = record.get("files", {}).get(name)
            if recorded != digest:
                reasons.append(
                    f"{name} digest {digest[:12]}... is not the {str(recorded)[:12]}... its "
                    "provenance records"
                )
        if record["parent"]["flow_sha256"] != flow_sha256:
            reasons.append(
                f"the parent flow.json digest {flow_sha256[:12]}... is not the one the parity "
                "record was derived from"
            )

        pair = audit_pair(
            f"{scenario_key}/draw_{draw_id:04d}",
            cityflow_roadnet=roadnet,
            sumo_net=net_path,
            cityflow_flow=parent / FLOW_FILENAME,
            sumo_routes=target / PARITY_ROUTES_FILENAME,
            direction="CAP(E) re-run on the rendered parity files (P7.3d, gate G1)",
        )
        demand = {field: pair["E_demand"][field] for field in _CAP_E_FIELDS}
        if not demand["counts_equal"]:
            reasons.append(
                f"E: {demand['n_cityflow']} CityFlow entries against {demand['n_sumo']} SUMO vehicles"
            )
        if not demand["multiset_equal"]:
            reasons.append(
                f"E: the (depart, route) multiset differs ({demand['n_only_in_cityflow']} only in "
                f"CityFlow, {demand['n_only_in_sumo']} only in SUMO)"
            )
        if not demand["order_matches"]:
            reasons.append(
                f"E: record order differs ({demand['n_index_aligned_equal']} of "
                f"{demand['n_cityflow']} index-aligned)"
            )
        if structure is None:
            bijection = pair["A_bijection"]
            structure = {
                "intersection_ids_equal": bijection["intersection_ids_equal"],
                "road_ids_equal": bijection["road_ids_equal"],
                "n_cityflow_lanes": bijection["n_cityflow_lanes"],
                "n_sumo_lanes": bijection["n_sumo_lanes"],
                "lane_links_equal": pair["C_connections"]["lane_link_equality"]["sets_equal"],
                "n_phases_compared": pair["D_signals"]["n_phases_compared"],
                "n_phases_matching_as_sets": pair["D_signals"]["n_phases_matching_as_sets"],
            }

        binding_report = parity.vtype_binding_report(target / PARITY_ROUTES_FILENAME)
        binding = {
            "n_vehicles": binding_report.vehicle_count,
            "n_bound": binding_report.vehicles_with_type,
            "distinct_type_values": list(binding_report.distinct_type_values),
            "vtype_ids": list(binding_report.vtype_ids),
            "vtype_attributes_in_file": dict(binding_report.vtype_attributes),
        }
        if binding_report.vehicles_with_type != binding_report.vehicle_count:
            reasons.append(
                f"binding: {binding_report.vehicles_with_type} of {binding_report.vehicle_count} "
                "vehicles are bound to a type"
            )
        elif not parity.binding_is_complete(binding_report, derived):
            reasons.append(
                "binding: every vehicle is bound, but not to exactly the parity <vType> derived "
                f"from the flow block (file declares {dict(binding_report.vtype_attributes)})"
            )

        sumocfg = ET.parse(target / PARITY_SUMOCFG_FILENAME).getroot()
        teleport = sumocfg.find(".//time-to-teleport")
        end = sumocfg.find(".//end")
        regime = {
            "time_to_teleport": None if teleport is None else teleport.get("value"),
            "end": None if end is None else end.get("value"),
        }
        if regime["time_to_teleport"] != str(PARITY_TIME_TO_TELEPORT):
            reasons.append(
                f"regime: time-to-teleport is {regime['time_to_teleport']!r}, not "
                f"{str(PARITY_TIME_TO_TELEPORT)!r} (A15(c))"
            )
        if regime["end"] is None or float(regime["end"]) <= parity.ENV_HORIZON_SECONDS:
            reasons.append(
                f"regime: end is {regime['end']!r}, not above the env horizon of "
                f"{parity.ENV_HORIZON_SECONDS} s"
            )

        if pedigree_identity is not None:
            if str(draw_id) in committed:
                matches = committed[str(draw_id)] == flow_sha256
                pedigree_rows[str(draw_id)] = {"flow_sha256": flow_sha256, "matches": matches}
                if not matches:
                    reasons.append(
                        f"pedigree: flow.json digest {flow_sha256[:12]}... is not the "
                        f"{committed[str(draw_id)][:12]}... {pedigree_identity['artifact']} committed"
                    )
            else:
                not_covered.append(draw_id)

        rows.append(
            {
                "draw_id": draw_id,
                "pool": classify_draw_pool(draw_id),
                "exact": not reasons,
                "reasons": reasons,
                "parity_format_version": record["format_version"],
                "flow_sha256": flow_sha256,
                "files": files,
                "e_demand": demand,
                "binding": binding,
                "regime": regime,
            }
        )

    n_exact = sum(1 for row in rows if row["exact"])
    commit, dirty = _git_provenance_strict()
    pedigree: dict[str, Any] | None = None
    if pedigree_identity is not None:
        pedigree = {
            **pedigree_identity,
            "what": "sha256 of each parent's flow.json against the digest committed by P8.4a",
            "n_checked": len(pedigree_rows),
            "n_matching": sum(1 for row in pedigree_rows.values() if row["matches"]),
            "draws": pedigree_rows,
            "not_covered": not_covered,
        }
    return {
        "format_version": CAP_E_FORMAT_VERSION,
        "registered_in": "PREREGISTRATION A14(b)(E), A15(g), A20(f); BRIEF_39 gate G1",
        "scenario_key": scenario_key,
        "n_draws": len(rows),
        "n_exact": n_exact,
        "condition_met": bool(rows) and n_exact == len(rows),
        "e_key": "(depart, route), unshifted; multiset AND index-aligned (offline.conversion_audit)",
        "parity_contract_version": parity.PARITY_CONTRACT_VERSION,
        "vtype_attributes_registered": registered,
        "vtype_attributes_derived_from_the_flow_block": derived,
        "resco": resco,
        "structure_reconfirmed": structure,
        "pedigree": pedigree,
        "draws": rows,
        "what_this_does_not_say": [
            "CAP is structure-only: no simulation ran to produce this file, and it says nothing "
            "about dynamics, travel times or any policy.",
            "structure_reconfirmed repeats A15(g)'s A, C and D counts from the same two network "
            "files as a consistency record; it is not part of the condition and gates nothing.",
            "An exact E says the SUMO demand equals the CityFlow demand per vehicle; it does not "
            "say the two engines insert or route those vehicles identically.",
        ],
        "git_commit": commit,
        "git_dirty": dirty,
    }


def assert_cap_e_exact(report: Mapping[str, Any]) -> None:
    """Raise, naming the first draw and the reason, unless every draw of *report* is exact."""
    failing = [row for row in report["draws"] if not row["exact"]]
    if not failing and report["condition_met"] and report["n_exact"] == report["n_draws"]:
        return
    if not failing:
        raise ValueError(
            f"the report claims {report['n_exact']} of {report['n_draws']} exact and "
            f"condition_met={report['condition_met']!r} while no row fails; it is inconsistent"
        )
    first = failing[0]
    raise ValueError(
        f"CAP(E) is not exact on {len(failing)} of {report['n_draws']} draws; first: draw "
        f"{first['draw_id']}: " + "; ".join(first["reasons"])
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.materialise_draws",
        description=(
            "Materialise flow draws into stable, runnable scenario directories. "
            "Re-running is a no-op for draws that already exist and match."
        ),
    )
    parser.add_argument(
        "--env-config",
        action="append",
        required=True,
        metavar="PATH",
        help="source CityFlow sim config; repeat for several scenarios",
    )
    parser.add_argument(
        "--draws",
        type=int,
        nargs="+",
        default=[],
        metavar="ID",
        help="explicit draw ids, e.g. --draws 0 1 2 3 4 5",
    )
    parser.add_argument(
        "--draws-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="draw ids over the half-open interval [START, END), matching "
        "range() and offline.collect's --flow-draws-range",
    )
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_OUT_ROOT),
        help=f"root of the materialised tree (default: {DEFAULT_OUT_ROOT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing draw that differs; without it, a difference is refused",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report what would be written, without writing anything",
    )
    parser.add_argument(
        "--parity",
        action="store_true",
        help="add draw_NNNN/parity/ (bound routes + teleport-free .sumocfg) to every "
        "requested draw, materialising missing parents first; the parent is never written to",
    )
    parser.add_argument(
        "--allow-worktree",
        action="store_true",
        help="permit an --out-root inside a LINKED WORKTREE. DEFERRED 55: scenarios/draws/ is "
        "gitignored and per-worktree, and retiring a worktree deleted held-out draws that merged "
        "numbers depend on, so materialising into one is refused unless this is passed",
    )
    parser.add_argument(
        "--verify-p4-3-probe",
        action="store_true",
        help="re-run P4.3's in-domain MaxPressure probe on the materialised draws and compare "
        "every recorded number against docs/data/p4_3_probe.json; writes nothing under --out-root",
    )
    parser.add_argument(
        "--verify-heldout-thresholds",
        action="store_true",
        help="re-run the held-out pool's CityFlow parents (draws 1000-1099, maxpressure) against "
        "docs/data/p4_heldout_thresholds.json and compare every recorded field; writes nothing",
    )
    parser.add_argument(
        "--report-cap-e",
        metavar="PATH",
        help="gate G1 (P7.3d): re-run CAP(E) and the vType binding report on the requested "
        "draws' parity directories and write the artifact here (must be outside --out-root); "
        "exit 0 only when every draw is exact; writes nothing under --out-root",
    )
    parser.add_argument(
        "--pedigree-artifact",
        metavar="PATH",
        help="with --report-cap-e: also compare each parent's flow.json digest against the "
        "per-draw digests this committed artifact records (docs/data/p8_4a_admission.json)",
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        help="with --verify-p4-3-probe, also write the comparison as JSON here (must be "
        "outside --out-root)",
    )
    return parser


def _resolve_cli_draw_ids(args: argparse.Namespace) -> list[int]:
    """Union of ``--draws`` and ``--draws-range``, in the order given."""
    ids = [int(draw_id) for draw_id in args.draws]
    if args.draws_range is not None:
        start, end = (int(value) for value in args.draws_range)
        if start >= end:
            raise ValueError(
                f"--draws-range is half-open [START, END), so START must be < END; got "
                f"[{start}, {end}), which selects no draws"
            )
        ids.extend(range(start, end))
    if not ids:
        raise ValueError("no draw ids requested; use --draws and/or --draws-range")
    return ids


def _parity_state_word(record: ParityResult, *, dry_run: bool) -> str:
    """The one phrase describing what happened, or would happen, to one draw."""
    if not dry_run:
        return f"parity {record.action}"
    if record.parent_action == "planned":
        return "parent missing (would materialise)"
    if record.planned_action == "kept":
        return "parity kept"
    if record.planned_action == "differs":
        return "parity differs (refused)"
    return "would add parity"


def _report_parity(args: argparse.Namespace, env_config: str, ids: Sequence[int]) -> int:
    """Run the parity phase for one scenario and print one line per draw."""
    records = materialise_parity(
        env_config,
        ids,
        out_root=args.out_root,
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        allow_worktree=bool(args.allow_worktree),
    )
    for record in records:
        print(
            f"{record.scenario_key} draw {record.draw_id:>4} [{record.pool}] "
            f"{_parity_state_word(record, dry_run=bool(args.dry_run))}: "
            f"{record.n_bound}/{record.n_vehicles} bound -> {record.directory}",
            flush=True,
        )
    differing = [record.draw_id for record in records if record.planned_action == "differs"]
    if differing:
        print(
            f"materialise_draws: {len(differing)} parity director(y/ies) differ and would be "
            f"refused: draws {differing[:5]}{' ...' if len(differing) > 5 else ''}",
            flush=True,
        )
        return 1
    return 0


def _report_cap_e(args: argparse.Namespace, ids: Sequence[int]) -> int:
    """Gate G1's artifact for one scenario; 0 only when every draw is exact.

    Every refusal precedes the write, and the write is atomic.  A verdict that is NOT met is still
    written -- the gate is decided by the artifact -- and the exit code carries it.
    """
    if len(args.env_config) != 1:
        raise ValueError("--report-cap-e audits one scenario; pass exactly one --env-config")
    out_path = Path(args.report_cap_e).resolve()
    root = Path(args.out_root).resolve()
    if out_path == root or out_path.is_relative_to(root):
        raise ValueError(
            f"--report-cap-e {out_path} is inside --out-root {root}; this mode writes nothing "
            "under the draws tree"
        )
    report = cap_e_report(
        args.env_config[0],
        ids,
        out_root=args.out_root,
        pedigree_artifact=args.pedigree_artifact,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    staged = out_path.with_name(out_path.name + ".tmp")
    staged.write_text(payload, encoding="utf-8")
    os.replace(staged, out_path)

    print(
        f"CAP(E) on the rendered parity files: {report['n_exact']}/{report['n_draws']} draws "
        f"exact -> condition_met={report['condition_met']}",
        flush=True,
    )
    pedigree = report["pedigree"]
    if pedigree is not None:
        print(
            f"  pedigree against {pedigree['artifact']}: {pedigree['n_matching']}/"
            f"{pedigree['n_checked']} parent digests match "
            f"({len(pedigree['not_covered'])} draws not covered by it)",
            flush=True,
        )
    for row in report["draws"]:
        if not row["exact"]:
            print(f"  FIRST FAILING draw {row['draw_id']}: " + "; ".join(row["reasons"]), flush=True)
            break
    print(f"wrote {out_path}", flush=True)
    return 0 if report["condition_met"] else 1


def _report_heldout_thresholds(args: argparse.Namespace, ids: Sequence[int]) -> int:
    """``DEFERRED`` 80's gate for the held-out band; 0 when every draw reproduces every field.

    ``BRIEF_37`` §3.0: **100/100 or the task stops.** The held-out SUMO draws P7.3a evaluates on
    are rendered from these CityFlow parents, so this is where their pedigree is established.
    """
    checks = verify_against_artifact(
        P4_HELDOUT_THRESHOLDS_ARTIFACT,
        arm="maxpressure",
        # The artifact is hangzhou's (docs/data/p4_heldout_thresholds.json), so the key is the
        # hangzhou scenario's -- read from the parity registry rather than retyped as a literal.
        scenario_key=parity.HZ1X1_SCENARIO.key,
        out_root=args.out_root,
        draw_ids=ids or None,
        scenario_id=parity.HZ1X1_SCENARIO.key,
    )
    reproduced = [check for check in checks if check.matches]
    fields = sorted(checks[0].observed) if checks else []
    print(
        f"held-out threshold reproduction: {len(reproduced)}/{len(checks)} draws reproduce "
        f"every recorded field {fields}",
        flush=True,
    )
    first_bad = next((check for check in checks if not check.matches), None)
    if first_bad is not None:
        print(
            f"  FIRST DIFFERING draw {first_bad.draw_id}: "
            + "; ".join(
                f"{key}: observed {first_bad.observed[key]!r} against recorded "
                f"{first_bad.expected[key]!r}"
                for key in first_bad.differing
            ),
            flush=True,
        )
        return 1
    return 0 if checks else 1


def _report_probe(args: argparse.Namespace, env_config: str, ids: Sequence[int]) -> int:
    """Run the P4.3 band gate for one scenario; return 0 when every draw reproduces."""
    checks = verify_p4_3_probe(env_config, ids, out_root=args.out_root)
    reproduced = [check for check in checks if check.matches]
    print(
        f"P4.3 probe reproduction: {len(reproduced)}/{len(checks)} draws reproduce every "
        "recorded number",
        flush=True,
    )
    if checks:
        example = checks[0]
        print(
            f"  example draw {example.draw_id}: "
            + ", ".join(f"{key}={value!r}" for key, value in sorted(example.observed.items())),
            flush=True,
        )
    first_bad = next((check for check in checks if not check.matches), None)
    if first_bad is not None:
        print(
            f"  FIRST DIFFERING draw {first_bad.draw_id}: "
            + "; ".join(
                f"{key}: observed {first_bad.observed[key]!r} against recorded "
                f"{first_bad.expected[key]!r}"
                for key in first_bad.differing
            ),
            flush=True,
        )

    if args.report:
        report_path = Path(args.report).resolve()
        root = Path(args.out_root).resolve()
        if report_path == root or report_path.is_relative_to(root):
            raise ValueError(
                f"--report {report_path} is inside --out-root {root}; this mode writes "
                "nothing under the draws tree"
            )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "n_draws": len(checks),
                    "n_reproduced": len(reproduced),
                    "draws": [
                        {
                            "draw_id": check.draw_id,
                            "matches": check.matches,
                            "observed": check.observed,
                            "expected": check.expected,
                            "differing": list(check.differing),
                        }
                        for check in checks
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {report_path}", flush=True)
    return 0 if first_bad is None else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run one materialisation; returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        ids = _resolve_cli_draw_ids(args)
        if args.verify_heldout_thresholds:
            return _report_heldout_thresholds(args, ids)
        if args.report_cap_e:
            return _report_cap_e(args, ids)
        for env_config in args.env_config:
            if args.verify_p4_3_probe:
                if _report_probe(args, env_config, ids) != 0:
                    return 1
                continue
            if args.parity:
                if _report_parity(args, env_config, ids) != 0:
                    return 1
                continue
            records = materialise(
                env_config,
                ids,
                out_root=args.out_root,
                force=bool(args.force),
                dry_run=bool(args.dry_run),
            )
            for record in records:
                sumo = "" if record.sumo_path is None else " +sumo"
                print(
                    f"{record.scenario_key} draw {record.draw_id:>4} "
                    f"[{record.pool}] {record.action}: {record.n_vehicles} vehicles"
                    f"{sumo} -> {record.directory}",
                    flush=True,
                )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"materialise_draws: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
