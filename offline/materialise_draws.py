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

**No ``.sumocfg`` is generated** -- that is P7.3's, along with the ``route-files``/``begin``
agreement it needs.  A rendered file whose template leaves ``<vType>`` unbound (hangzhou)
says so in its provenance: SUMO would run those vehicles as ``DEFAULT_VEHTYPE``, so the
file must not feed a transfer measurement until P7.0's parity contract lands
(``docs/briefs/BRIEF_04_p7.0_transfer_gate.md`` §3).
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
from typing import Any, Sequence

import numpy as np

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
    "PROVENANCE_FILENAME",
    "SUMO_ROUTES_FILENAME",
    "TRAINING_POOL",
    "build_parser",
    "classify_draw_pool",
    "draw_config_path",
    "draw_dir",
    "load_provenance",
    "main",
    "materialise",
    "scenario_key_for_config",
]

FORMAT_VERSION = "materialised-draw/1.0"
DEFAULT_OUT_ROOT = Path("scenarios/draws")

FLOW_FILENAME = "flow.json"
CITYFLOW_CONFIG_FILENAME = "cityflow.json"
SUMO_ROUTES_FILENAME = "routes.rou.xml"
PROVENANCE_FILENAME = "provenance.json"

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


# -- CLI -------------------------------------------------------------------


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run one materialisation; returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        ids = _resolve_cli_draw_ids(args)
        for env_config in args.env_config:
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
