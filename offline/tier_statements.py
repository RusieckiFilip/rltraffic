"""P5.2c (``BRIEF_45`` and its Amendment A): P5.2's registered statements, scored by committed code on every tier, under
both ATT definitions and both random-tier IQL cells.

FORMAT
------
The artifact is ``p5.2-statements/1.0`` (:data:`ARTIFACT_FORMAT_VERSION`), a JSON object written once by ``build`` at the
path its command line names (``docs/data/p5_2_statements.json``): ``json.dumps(..., indent=2, sort_keys=True,
allow_nan=False) + "\\n"``, no rounding, no timestamp, every input path relative to its root.  It is a pure function of
its inputs and of the git block it records, so ``tests/test_tier_statements.py``'s T-regress regenerates it byte for byte.

ALIGNMENT CONVENTION (the paired unit)
--------------------------------------
* A **cell** is one arm on one tier: 5 training seeds x 100 held-out draws, each ``(seed, draw)`` once.  Its **level** is
  ``dt_gate.mean_ci95`` over its 500 episodes in ``(seed, draw)`` order (``iql_correction.levels_of``).
* A **contrast** ``left - right`` is formed only between two complete cells covering the IDENTICAL ``(seed, draw)``
  grid; anything else is refused.  Seeds are pooled per draw FIRST (``dt_gate._per_draw_means``: ``numpy.mean`` of the
  five seeds), the differences are taken per draw in ascending draw order, and the interval is ``dt_gate.mean_ci95`` over
  the 100 differences (``ddof=1``, ``1.96 * sd / sqrt(n)``).  This is P5.2's registered route
  (``docs/plans/p5.2.md:407-412``), and the one arithmetic route of this module (Amendment A, A2 Q1).
* ``I = d4 - d1`` is a difference of two contrasts, formed per draw from the four cells' per-draw means.
* A definition (``att_ours`` = P5.2's ``att_horizon``, or ``att_engine``) is carried in ``EpisodeResult.att_horizon``'s
  slot by ``iql_correction.episodes_from_rows`` (P5.2b's convention), so every imported scorer reads it unchanged.

WHAT IS IMPORTED AND WHAT IS RESTATED (plan section 4, Amendment A)
-------------------------------------------------------------------
Every scorer and anchor is imported from the module that owns it, unchanged.  Three functions are RESTATED, each with a
test proving it equal to the original where the original applies: :func:`tier_composer` (``iql_correction.
random_tier_statements``, hard-wired to the random tier), :func:`read_listed_json` (``iql_correction.read_p5_2_json``,
hard-wired to P5.2's manifest name) and :func:`assert_note_means` (``iql_correction.assert_c1_note_means``, random-only).

No episode is rolled, no checkpoint is loaded, and nothing under ``output/`` is written, moved or deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import offline.att_rederivation as ar
import offline.compute_latency as cl
import offline.dt_gate as dt_gate
import offline.iql_correction as ic
import offline.spatial_mixing as sm
import offline.tier_sweep as ts
from offline.dt_gate import EpisodeResult, _per_draw_means, mean_ci95
from offline.offline_baselines import paired_comparison
from offline.spatial_mixing import collapse_criterion

ARTIFACT_FORMAT_VERSION = "p5.2-statements/1.0"

SCENARIO = "grid4x4"

#: The two ATT definitions (A11).  Under Rule R the second is the primary one on grid4x4.
DEFINITIONS: tuple[str, ...] = ("att_ours", "att_engine")
PRIMARY_DEFINITION = "att_engine"

#: ``corrected``: the random tier's IQL cell is P5.2b's (the paper's); ``as_committed``: P5.2's original cell.
VERSIONS: tuple[str, ...] = ("corrected", "as_committed")

TIERS: tuple[str, ...] = ts.TIER_ORDER
OUT_OF_SAMPLE_TIERS: tuple[str, ...] = ("maxpressure", "fixedtime", "random")
DT_ARMS: tuple[str, ...] = ("dt_spatial", "dt_nomix")
NON_DT_METHODS: tuple[str, ...] = ic.NON_DT_METHODS

#: A11(b)'s companions, read from P8.4b's rows (and P5.2b's for the corrected cell).
COMPANIONS: tuple[str, ...] = ("entered", "created", "never_entered", "entered_fraction")

#: The logical key of the shared random anchor (D12): P5.1's ``random`` cell serves every tier.
ANCHOR: tuple[str, str] = ("random", "mappo1000")

P5_1_SUMS_NAME = "SHA256SUMS_p5_1.txt"
P5_2_SUMS_NAME = ic.P5_2_SUMS_NAME
P5_2B_SUMS_NAME = ic.MANIFEST_NAME
P8_4B_DIR = ic.P8_4B_DIR
P5_2B_RUN_DIR = ic.RUN_DIR
P5_2B_REDERIVATION_DIR = f"{ic.RUN_DIR}/{ic.REDERIVATION_SUBDIR}"
P5_2B_TRAINING_RECORD = f"{ic.RUN_DIR}/{ic.TRAINING_RECORD_NAME}"
P5_2_STOP_RULE_RECORD = "p5_2/stop_rule_mappo1000.json"

P5_2B_ARTIFACT = "docs/data/p5_2b_iql_correction.json"
P8_4B_ARTIFACT = "docs/data/p8_4b_rederivation.json"
P5_1_GRID = "docs/data/p5_1_grid.json"
C1_NOTE = ic.C1_NOTE_RELPATH
DEFAULT_OUT = "docs/data/p5_2_statements.json"

#: The files a ``file:line`` citation may point into (Amendment A, A4.2): history, never a living document.
CITABLE_FILES: tuple[str, ...] = (
    "docs/plans/p5.2.md",
    "docs/plans/p5.1.md",
    "docs/briefs/BRIEF_27_p5.2_tier_sweep.md",
    "docs/briefs/BRIEF_44_p5.2b_iql_correction.md",
    "docs/returns/P5.2.md",
    "PREREGISTRATION.md",
)

#: Every ``offline`` module a fresh interpreter loads to import this one (Amendment A, A4.1; the list is asserted by a
#: test that imports the module in a subprocess).  ``build`` and ``check`` refuse unless each, and every other loaded
#: ``offline`` module, resolves under ``<--repo-root>/offline``; ``provenance.code`` records each with its sha256.
CODE_MODULES: tuple[str, ...] = (
    "offline",
    "offline.att_rederivation",
    "offline.compute_latency",
    "offline.dataset",
    "offline.dt_gate",
    "offline.iql_correction",
    "offline.joint_windows",
    "offline.offline_baselines",
    "offline.roadnet_graph",
    "offline.spatial_mixing",
    "offline.tier_statements",
    "offline.tier_sweep",
    "offline.trajectory_logger",
)


# ----------------------------------------------------------------------
# Pins and roots
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Pins:
    """The digests every input is read at.  Tests pass their own for synthetic trees; ``build`` and ``check`` use
    :data:`PINS`."""

    p5_1_sums_sha256: str
    p5_2_sums_sha256: str
    p5_2b_sums_sha256: str
    p8_4b_declared_cells_sha256: str
    p5_2b_rederivation_declared_cells_sha256: str
    p5_2b_artifact_sha256: str
    p8_4b_artifact_sha256: str
    p5_1_grid_sha256: str
    c1_note_sha256: str


#: The real pins, each recomputed with ``sha256sum`` in the plan session (plan V1-V4) and in the C1 session.
PINS = Pins(
    p5_1_sums_sha256="023607ffe65a881932f0069412f442c65b1097bb87fdd7852873a1211fe24c61",
    p5_2_sums_sha256=ic.PINS.p5_2_sums_sha256,
    p5_2b_sums_sha256="f3e1994c575f923f42f9a1bc7dae9fbaf25b418cff4c4ff8b1e208e4d41f10ed",
    p8_4b_declared_cells_sha256=ic.PINS.p8_4b_declared_cells_sha256,
    p5_2b_rederivation_declared_cells_sha256="09568e28296e7cf8fa292d0db32e3cf1532990ec66c3af4108cf420b3b778c9e",
    p5_2b_artifact_sha256="216b9f24243c317d22b13a3cdcd346471bf9cfe9832e5a5495b8158f04eea1f8",
    p8_4b_artifact_sha256="d92321d422acc06a35bde7a499cef41232846ced33ceea594c4ef199792c96d9",
    p5_1_grid_sha256="ccb9a315960224e597aeee6cf7910bff9cd1d1fd9dacfe98ef8185ead4e883b2",
    c1_note_sha256=ic.PINS.c1_note_sha256,
)


@dataclass(frozen=True)
class Roots:
    """``repo_root``: the checkout whose ``docs/`` is read and whose code runs (A4.1); ``output_root``: the main tree's
    ``output/``, read only."""

    repo_root: Path
    output_root: Path


# ----------------------------------------------------------------------
# The cells
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CellSpec:
    """One cell any statement reads.  ``cell_id`` is ``<source>:<arm>``; ``tier`` is the LOGICAL tier (P5.1's cells are
    ``mappo1000``); ``arm`` is the string the eval file and P8.4b's rows carry; ``eval_file`` is relative to the output
    root and is listed in ``manifest``."""

    cell_id: str
    method: str
    tier: str
    source: str
    arm: str
    eval_file: str
    manifest: str


def _p5_1(method: str) -> CellSpec:
    arm = f"{method}@grid4x4_mappo1000"
    return CellSpec(cell_id=f"p5_1:{arm}", method=method, tier="mappo1000", source="p5_1", arm=arm,
                    eval_file=f"p5_1/eval_{method}.json", manifest=P5_1_SUMS_NAME)


def _p5_2(method: str, tier: str) -> CellSpec:
    arm = f"{method}@{tier}"
    return CellSpec(cell_id=f"p5_2:{arm}", method=method, tier=tier, source="p5_2", arm=arm,
                    eval_file=f"p5_2/{ts.cell_filename(tier, method)}", manifest=P5_2_SUMS_NAME)


#: Every cell read: P5.1's seven reused ``mappo1000`` cells, P5.2's 24 (the six methods on three tiers, two behaviour
#: anchors, ``bc_top10_perix``, the two 4-head cells and the re-rolled random anchor at ``mappo1000``) and P5.2b's
#: corrected IQL cell.  32 cells; P8.4b's rows exist for the first 31.
CELL_SPECS: tuple[CellSpec, ...] = (
    *(_p5_1(method) for method in ("dt_spatial", "dt_nomix", "bc", "bc_top10", "iql", "behaviour", "random")),
    *(_p5_2(method, tier) for tier in OUT_OF_SAMPLE_TIERS for method in ts.METHODS),
    _p5_2("behaviour", "maxpressure"),
    _p5_2("behaviour", "fixedtime"),
    _p5_2("bc_top10_perix", "mappo1000"),
    _p5_2("dt_spatial_h4", "mappo1000"),
    _p5_2("dt_nomix_h4", "mappo1000"),
    _p5_2("random", "mappo1000"),
    CellSpec(cell_id="p5_2b:iql@random", method="iql", tier="random", source="p5_2b", arm="iql@random",
             eval_file=f"{ic.RUN_DIR}/{ic.EVAL_NAME}", manifest=P5_2B_SUMS_NAME),
)

CORRECTED_IQL = "p5_2b:iql@random"
ORIGINAL_IQL = "p5_2:iql@random"
ORIGINAL_IQL_EVAL = "p5_2/eval_random_iql.json"
ANCHOR_CELL = "p5_1:random@grid4x4_mappo1000"
REROLLED_ANCHOR_CELL = "p5_2:random@mappo1000"
H4_CELLS: tuple[str, str] = ("p5_2:dt_spatial_h4@mappo1000", "p5_2:dt_nomix_h4@mappo1000")

#: How A4.3's two digests are formed, stated in the artifact beside each.
ROWS_DIGEST_CONSTRUCTION = (
    "sha256 over the text '<file sha256>  <path relative to --output-root>\\n' for every file read, in path order -- "
    "the sha256sum line format, so `sha256sum <files> | LC_ALL=C sort -k2 | sha256sum`, run in --output-root over the "
    "same files, recomputes it")


@dataclass(frozen=True)
class Cell:
    """A loaded cell: its spec and its episodes in ``(seed, draw)`` order.  Each row carries ``arm``, ``seed``,
    ``draw_id``, ``att_ours`` (the eval file's ``att_horizon``), ``att_engine`` and A11(b)'s companions (P8.4b's row, or
    P5.2b's for the corrected cell), and ``horizon_vehicle_count`` and ``episode_reward`` (the eval file's)."""

    spec: CellSpec
    rows: tuple[Mapping[str, Any], ...]


def make_books(cells: Mapping[str, Cell]) -> dict[str, dict[tuple[str, str], Cell]]:
    """The two versions' books: logical ``(method, tier)`` -> cell.  Identical but at ``("iql", "random")``, which is
    :data:`CORRECTED_IQL` under ``corrected`` and :data:`ORIGINAL_IQL` under ``as_committed``.  ``("behaviour",
    "random")`` and :data:`ANCHOR` are both P5.1's random cell (D12)."""
    missing = sorted({spec.cell_id for spec in CELL_SPECS} - set(cells))
    if missing:
        raise ValueError(f"the cells {missing[:4]} are not loaded; every statement's inputs must exist")
    common: dict[tuple[str, str], Cell] = {}
    for spec in CELL_SPECS:
        if spec.cell_id in (CORRECTED_IQL, ORIGINAL_IQL, REROLLED_ANCHOR_CELL):
            continue
        common[(spec.method, spec.tier)] = cells[spec.cell_id]
    common[("behaviour", "random")] = cells[ANCHOR_CELL]
    return {"corrected": {**common, ("iql", "random"): cells[CORRECTED_IQL]},
            "as_committed": {**common, ("iql", "random"): cells[ORIGINAL_IQL]}}


def episodes_of(cell: Cell, definition: str) -> list[EpisodeResult]:
    """The cell's episodes under *definition*, by ``iql_correction.episodes_from_rows`` (refuses a duplicate key)."""
    if definition not in DEFINITIONS:
        raise ValueError(f"{definition!r} is not one of {DEFINITIONS}")
    return ic.episodes_from_rows(cell.rows, value_key=definition)


# ----------------------------------------------------------------------
# Inputs, each read at its anchor, all before anything is written
# ----------------------------------------------------------------------


def read_manifest(output_root: str | Path, name: str, pin: str) -> dict[str, str]:
    """``output/<name>`` parsed by ``tier_sweep.parse_sha256sums``, refused unless the file is at *pin*."""
    path = Path(output_root) / name
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != pin:
        raise ValueError(f"{path} is at sha256 {digest}, not the pinned {pin}: a manifest is the anchor of every file it "
                         "lists, so a changed manifest anchors nothing")
    return ts.parse_sha256sums(data.decode("utf-8"))


def read_listed_json(output_root: str | Path, relative: str, sums: Mapping[str, str], *, manifest_name: str
                     ) -> dict[str, Any]:
    """A JSON record (``relative`` to ``output/``), refused unless its sha256 is its line in *manifest_name*.
    ``iql_correction.read_p5_2_json`` restated for any manifest (Amendment A, A2 Q3)."""
    expected = sums.get(relative)
    if expected is None:
        raise ValueError(f"{relative} has no line in {manifest_name}; an unlisted file cannot be shown to be the record's")
    path = Path(output_root) / relative
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise ValueError(f"{path} is at sha256 {digest}, which does not match its line in {manifest_name} ({expected}); "
                         "its bytes changed after the record was written")
    return json.loads(data)


def read_pinned_json(repo_root: str | Path, relative: str, pin: str) -> dict[str, Any]:
    """A committed document under ``repo_root``, refused unless its blob sha256 is *pin*."""
    path = Path(repo_root) / relative
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != pin:
        raise ValueError(f"{path} is at sha256 {digest}, not the pinned {pin}")
    return json.loads(data)


def assert_note_means(output_root: str | Path, campaign_cells: Sequence[str], note: Mapping[str, Any], *, tier: str,
                      arms: Mapping[str, str]) -> dict[str, Any]:
    """``iql_correction.assert_c1_note_means`` restated for any tier (plan F13): the six means under both definitions,
    by the note's own route (``statistics.mean`` over each arm's P8.4b cell files listed in *campaign_cells*), refused
    unless each equals ``note["tiers"]["grid4x4/<tier>"]``'s.  *arms* maps each method to its P8.4b arm string."""
    work = Path(output_root) / P8_4B_DIR
    block = note["tiers"][f"{SCENARIO}/{tier}"]
    methods = {arm: method for method, arm in arms.items()}
    values: dict[str, dict[str, list[float]]] = {method: {name: [] for name in DEFINITIONS} for method in ts.METHODS}
    for key in campaign_cells:
        scenario, arm, seed, draw = str(key).split("|")
        if scenario != SCENARIO or arm not in methods:
            continue
        method = methods[arm]
        name = ar.cell_file_name(ar.CellKey(scenario=scenario, arm=arm, seed=int(seed), draw_id=int(draw)))
        row = json.loads((work / name).read_bytes())
        identity = (str(row["scenario"]), str(row["tier"]), str(row["method"]), str(row["arm"]), int(row["seed"]),
                    int(row["draw_id"]))
        if identity != (scenario, tier, method, arm, int(seed), int(draw)):
            raise ValueError(f"P8.4b's {name} is not the cell its name says ({key}): it holds {identity}")
        for definition in DEFINITIONS:
            values[method][definition].append(float(row[definition]))
    for method in ts.METHODS:
        for definition in DEFINITIONS:
            if not values[method][definition]:
                raise ValueError(f"P8.4b's campaign lists no cell of {arms[method]}")
            mine = statistics.mean(values[method][definition])
            theirs = block[definition]["means"][method]
            if mine != theirs:
                raise ValueError(
                    f"{arms[method]} under {definition}: the mean of P8.4b's {len(values[method][definition])} cells by "
                    f"the note's own route (statistics.mean) is {mine!r}, and {C1_NOTE} says {theirs!r}: the note's anchor "
                    f"does not hold on the {tier} tier, so no P8.4b value is read (plan F13)")
    return {"arms": sorted(ts.METHODS), "definitions": list(DEFINITIONS), "route": "statistics.mean",
            "cells_per_arm": {method: len(values[method][DEFINITIONS[0]]) for method in ts.METHODS}}


def assert_anchor_identity(p5_1: Mapping[str, Any], p5_2: Mapping[str, Any], p5_1_rows: Sequence[Mapping[str, Any]],
                           p5_2_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """D12's shared anchor: refused unless P5.2's re-rolled random cell is episode-identical to P5.1's
    (``tier_sweep.assert_cells_identical``, every episode of the protocol) and P8.4b's rows of the two arms agree on
    ``att_ours``, ``att_engine`` and every companion."""
    expected_n = len(tuple(dt_gate.TRAINING_SEEDS)) * len(tuple(dt_gate.HELD_OUT_DRAWS))
    identity = ts.assert_cells_identical(p5_1, p5_2, expected_n=expected_n)
    left = {(int(r["seed"]), int(r["draw_id"])): r for r in p5_1_rows}
    right = {(int(r["seed"]), int(r["draw_id"])): r for r in p5_2_rows}
    if set(left) != set(right) or len(left) != expected_n:
        raise ValueError(f"P8.4b's rows of the two anchor arms cover {len(left)} and {len(right)} episodes, not the same "
                         f"{expected_n}")
    fields = ("att_ours", "att_engine", *COMPANIONS)
    differences = [f"{key} {field}" for key in sorted(left) for field in fields if left[key][field] != right[key][field]]
    if differences:
        raise ValueError(f"P8.4b's rows of the two anchor arms differ on {len(differences)} value(s), the first "
                         f"{differences[:5]}: D12's one anchor is not one cell")
    return {**identity, "p8_4b_fields": list(fields), "p8_4b_rows_compared": expected_n}


def rows_digest(pairs: Iterable[tuple[str, str]]) -> str:
    """Amendment A, A4.3: one sha256 over sorted ``(path relative to --output-root, file sha256)`` pairs, taken over the
    text ``"<file sha256>  <path>\\n"`` per pair in path order -- the ``sha256sum`` line format."""
    text = "".join(f"{digest}  {path}\n" for path, digest in sorted(pairs))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_code_under_root(repo_root: str | Path) -> list[dict[str, Any]]:
    """Amendment A, A4.1: refuse unless every module of :data:`CODE_MODULES` and every loaded ``offline`` module
    resolves to ``<repo_root>/offline/...``; return each :data:`CODE_MODULES` entry's path relative to the root and its
    sha256."""
    import importlib

    root = Path(repo_root).resolve()
    package = root / "offline"

    def located(name: str) -> list[Path]:
        module = sys.modules.get(name) or importlib.import_module(name)
        file = getattr(module, "__file__", None)
        if file:
            return [Path(file).resolve()]
        paths = [Path(entry).resolve() for entry in getattr(module, "__path__", [])]
        if not paths:
            raise ValueError(f"{name} names no file, so it cannot be shown to come from {root}")
        return paths

    loaded = {name for name in list(sys.modules) if name == "offline" or name.startswith("offline.")}
    for name in sorted(loaded | set(CODE_MODULES)):
        for path in located(name):
            inside = path == package or package in path.parents
            if not inside:
                raise ValueError(
                    f"{name} is loaded from {path}, which does not resolve under {package}: the artifact would record "
                    f"{root}'s git commit for code that did not come from it (BRIEF_45 Amendment A, A4.1)")
    offline_package = sys.modules["offline"]
    if [Path(entry).resolve() for entry in getattr(offline_package, "__path__", [])] != [package]:
        raise ValueError(f"the offline package searches {list(getattr(offline_package, '__path__', []))}, not exactly "
                         f"{package}: a later import could come from another tree (BRIEF_45 Amendment A, A4.1)")
    entries = []
    for name in CODE_MODULES:
        path = located(name)[0]
        entries.append({"module": name, "path": path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return entries


def git_block() -> dict[str, Any]:
    """``compute_latency._git_provenance()``: the commit, the dirty flag (tracked files) and the code root."""
    return dict(cl._git_provenance())


@dataclass(frozen=True)
class Inputs:
    """Everything the artifact is built from, verified.  ``cells`` by id; ``provenance`` every file read with its anchor
    and A4.3's two digests; ``records`` the committed records the comparison reads."""

    cells: Mapping[str, Cell]
    provenance: Mapping[str, Any]
    records: Mapping[str, Any]


def _ic_pins(pins: Pins) -> ic.Pins:
    """The ``iql_correction.Pins`` its readers take.  ``declaration_sha256`` and ``corpus_manifest_sha256`` are read by
    no function this module calls (they anchor P5.2b's training inputs) and are copied from ``iql_correction.PINS``."""
    return ic.Pins(p5_2_sums_sha256=pins.p5_2_sums_sha256, declaration_sha256=ic.PINS.declaration_sha256,
                   c1_note_sha256=pins.c1_note_sha256, p8_4b_declared_cells_sha256=pins.p8_4b_declared_cells_sha256,
                   corpus_manifest_sha256=ic.PINS.corpus_manifest_sha256)


def _arm_tier(arm: str) -> str:
    """The tier label inside an arm string: ``grid4x4_mappo1000`` for P5.1's, the tier for P5.2's."""
    return str(arm).partition("@")[2]


def _assert_eval_payload(spec: CellSpec, payload: Mapping[str, Any], seeds: Sequence[int], draws: Sequence[int]
                         ) -> list[EpisodeResult]:
    """One evaluation record: the cell its spec says, complete (``tier_sweep.cell_is_complete``), its level and A5's
    vehicle count equal to its own ``cell`` block.  Returns its episodes in ``(seed, draw)`` order (``att_horizon``)."""
    where = f"output/{spec.eval_file}"
    if (payload.get("arm"), payload.get("method"), payload.get("tier")) != (spec.arm, spec.method, _arm_tier(spec.arm)):
        raise ValueError(f"{where} is the record of {payload.get('arm')!r} ({payload.get('method')!r} at "
                         f"{payload.get('tier')!r}), not of {spec.arm}")
    if any(str(episode.get("arm")) != spec.arm for episode in payload["episodes"]):
        raise ValueError(f"{where} carries episodes of an arm other than {spec.arm}")
    if not ts.cell_is_complete(payload, seeds=seeds, draws=draws, declared_steps=ts.DECLARED_GRADIENT_STEPS,
                               method=spec.method):
        raise ValueError(f"{where} is not a complete cell: {len(payload['episodes'])} episodes against the protocol's "
                         f"{len(seeds)} seeds x {len(draws)} draws, each once, at {ts.DECLARED_GRADIENT_STEPS} declared "
                         "steps")
    episodes = ic.episodes_from_rows(payload["episodes"], value_key="att_horizon")
    level = ic.levels_of({(spec.method, spec.tier): episodes})[(spec.method, spec.tier)]
    if level != float(payload["cell"]["att_horizon_mean"]):
        raise ValueError(f"{where}: the level {level!r} by mean_ci95 in (seed, draw) order is not the record's own cell "
                         f"mean {payload['cell']['att_horizon_mean']!r}")
    vehicles = float(mean_ci95([episode.horizon_vehicle_count for episode in episodes]).mean)
    if vehicles != float(payload["cell"]["horizon_vehicle_count_mean"]):
        raise ValueError(f"{where}: A5's horizon vehicle count {vehicles!r} by mean_ci95 is not the record's own "
                         f"{payload['cell']['horizon_vehicle_count_mean']!r}")
    return episodes


def _merged_rows(spec: CellSpec, episodes: Sequence[EpisodeResult], rows: Sequence[Mapping[str, Any]]
                 ) -> tuple[dict[str, Any], ...]:
    """The cell's rows: ``att_ours`` and A5's companions from the evaluation record, ``att_engine`` and A11(b)'s
    companions from the re-derivation row of the same ``(seed, draw)``."""
    out = []
    for episode, row in zip(episodes, rows, strict=True):
        if (int(row["seed"]), int(row["draw_id"])) != (episode.seed, episode.draw_id):
            raise ValueError(f"{spec.cell_id}: the evaluation and re-derivation rows are not aligned at "
                             f"{(episode.seed, episode.draw_id)}")
        out.append({"arm": spec.arm, "seed": int(episode.seed), "draw_id": int(episode.draw_id),
                    "att_ours": float(episode.att_horizon), "att_engine": float(row["att_engine"]),
                    "entered": int(row["entered"]), "created": int(row["created"]),
                    "never_entered": int(row["never_entered"]), "entered_fraction": float(row["entered_fraction"]),
                    "horizon_vehicle_count": float(episode.horizon_vehicle_count),
                    "episode_reward": float(episode.episode_reward)})
    return tuple(out)


def _read_rows_digested(output: Path, directory: str, spec: CellSpec, episodes: Sequence[EpisodeResult],
                        rows: Sequence[Mapping[str, Any]], pairs: list[tuple[str, str]],
                        listed: Mapping[str, str] | None = None) -> None:
    """A4.3: each re-derivation row's file read once more, hashed, and refused unless its bytes parse to the row just
    checked (and, given *listed*, unless the hash is its manifest line); ``(path, sha256)`` appended to *pairs*."""
    for episode, row in zip(episodes, rows, strict=True):
        name = ar.cell_file_name(ar.CellKey(scenario=SCENARIO, arm=spec.arm, seed=episode.seed,
                                            draw_id=episode.draw_id))
        relative = f"{directory}/{name}"
        data = (output / relative).read_bytes()
        if json.loads(data) != row:
            raise ValueError(f"output/{relative} changed while it was being read")
        digest = hashlib.sha256(data).hexdigest()
        if listed is not None and listed.get(relative) != digest:
            raise ValueError(f"output/{relative} is at sha256 {digest}, which does not match its line in "
                             f"{P5_2B_SUMS_NAME} ({listed.get(relative)})")
        pairs.append((relative, digest))


def _assert_d17(output: Path, sums: Mapping[str, str], payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """``BRIEF_45`` section 3.2, Q0: D17's deterministic 1-head pair was never trained, verified on disk -- P5.2's
    manifest lists no complete 1-head ``mappo1000`` DT cell, its 1-head ``mappo1000`` DT training records hold only
    default-regime runs, and every P5.2 evaluation read is default-regime -- so ``d1`` is P5.1's pair."""
    listed = [f"p5_2/{ts.cell_filename('mappo1000', method)}" for method in DT_ARMS]
    present = [name for name in listed if name in sums]
    if present:
        raise ValueError(f"P5.2's manifest lists {present}: a complete 1-head mappo1000 DT cell from P5.2 would make "
                         "d1 ambiguous between it and P5.1's pair (D17)")
    records = {}
    for method in DT_ARMS:
        relative = f"p5_2/{ts.training_record_name('mappo1000', method)}"
        record = ic.read_p5_2_json(output, relative, sums)
        runs = [{"seed": int(run["seed"]), "deterministic": run.get("deterministic")} for run in record.get("runs", [])]
        if record.get("deterministic") is not False or any(run["deterministic"] is not False for run in runs):
            raise ValueError(f"output/{relative} records a deterministic run ({runs}): D17's deterministic 1-head pair "
                             "would exist, and d1 would not be P5.1's default-regime pair")
        records[f"output/{relative}"] = runs
    regimes = sorted({str(payload.get("deterministic")) for cell_id, payload in payloads.items()
                      if cell_id.startswith("p5_2:")})
    if regimes != ["False"]:
        raise ValueError(f"P5.2's evaluation records state deterministic = {regimes}, not only False (D17)")
    return {"complete_1_head_mappo1000_dt_cells_in_p5_2": present, "training_records": records,
            "p5_2_evaluations_deterministic": False,
            "reading": "D17's deterministic 1-head pair was never trained; d1 is P5.1's default-regime pair"}


def _book_arm(method: str, tier: str) -> str:
    """The P8.4b arm string of a method's cell on a tier, as the books read it (P5.1's at ``mappo1000`` but for
    ``bc_top10_perix``, which only P5.2 ran there)."""
    if tier == "mappo1000" and method != "bc_top10_perix":
        return f"{method}@grid4x4_mappo1000"
    return f"{method}@{tier}"


def load_inputs(roots: Roots, *, pins: Pins = PINS) -> Inputs:
    """Every input at its anchor (plan section 6.2), refusing before anything is written.  The order is the order of
    trust: manifests, committed documents, evaluation records, P8.4b's campaign and every row read, the shared anchor,
    the note's means on every tier, P5.2b's corrected cell and its committed blocks."""
    output, repo = Path(roots.output_root), Path(roots.repo_root)
    seeds, draws = tuple(int(s) for s in dt_gate.TRAINING_SEEDS), tuple(int(d) for d in dt_gate.HELD_OUT_DRAWS)
    files: list[dict[str, Any]] = []
    sums = {P5_1_SUMS_NAME: read_manifest(output, P5_1_SUMS_NAME, pins.p5_1_sums_sha256),
            P5_2_SUMS_NAME: ic.p5_2_sums(output, pins=_ic_pins(pins)),
            P5_2B_SUMS_NAME: read_manifest(output, P5_2B_SUMS_NAME, pins.p5_2b_sums_sha256)}
    for name, pin in ((P5_1_SUMS_NAME, pins.p5_1_sums_sha256), (P5_2_SUMS_NAME, pins.p5_2_sums_sha256),
                      (P5_2B_SUMS_NAME, pins.p5_2b_sums_sha256)):
        files.append({"file": f"output/{name}", "sha256": pin, "anchor": "pinned", "lines": len(sums[name])})
    documents = {}
    for relative, pin in ((P5_2B_ARTIFACT, pins.p5_2b_artifact_sha256), (P8_4B_ARTIFACT, pins.p8_4b_artifact_sha256),
                          (P5_1_GRID, pins.p5_1_grid_sha256)):
        documents[relative] = read_pinned_json(repo, relative, pin)
        files.append({"file": relative, "sha256": pin, "anchor": "pinned"})
    note = ic._verified_note(repo, _ic_pins(pins))
    files.append({"file": note["file"], "sha256": note["sha256"], "anchor": "pinned"})

    payloads: dict[str, dict[str, Any]] = {}
    episodes: dict[str, list[EpisodeResult]] = {}
    for spec in CELL_SPECS:
        if spec.manifest == P5_2_SUMS_NAME:
            payload = ic.read_p5_2_json(output, spec.eval_file, sums[P5_2_SUMS_NAME])
        else:
            payload = read_listed_json(output, spec.eval_file, sums[spec.manifest], manifest_name=spec.manifest)
        episodes[spec.cell_id] = _assert_eval_payload(spec, payload, seeds, draws)
        payloads[spec.cell_id] = payload
        files.append({"file": f"output/{spec.eval_file}", "sha256": sums[spec.manifest][spec.eval_file],
                      "anchor": f"its line in output/{spec.manifest}"})
    stop_record = ic.read_p5_2_json(output, P5_2_STOP_RULE_RECORD, sums[P5_2_SUMS_NAME])
    files.append({"file": f"output/{P5_2_STOP_RULE_RECORD}", "sha256": sums[P5_2_SUMS_NAME][P5_2_STOP_RULE_RECORD],
                  "anchor": f"its line in output/{P5_2_SUMS_NAME}"})
    d17 = _assert_d17(output, sums[P5_2_SUMS_NAME], payloads)
    for relative in d17["training_records"]:
        files.append({"file": relative, "sha256": sums[P5_2_SUMS_NAME][relative.removeprefix("output/")],
                      "anchor": f"its line in output/{P5_2_SUMS_NAME}"})

    campaign = ic.assert_p8_4b_campaign(output, pins=_ic_pins(pins))
    declared: dict[str, set[tuple[int, int]]] = {}
    for key in campaign["cells"]:
        scenario, arm, seed, draw = str(key).split("|")
        if scenario == SCENARIO and seed != "None":
            declared.setdefault(arm, set()).add((int(seed), int(draw)))
    cells: dict[str, Cell] = {}
    p8_pairs: list[tuple[str, str]] = []
    for spec in CELL_SPECS:
        if spec.source == "p5_2b":
            continue
        reference = episodes[spec.cell_id]
        keys = {(episode.seed, episode.draw_id) for episode in reference}
        if declared.get(spec.arm, set()) != keys:
            raise ValueError(f"P8.4b's campaign declares {len(declared.get(spec.arm, set()))} cells of {spec.arm}, not "
                             f"the {len(keys)} episodes of output/{spec.eval_file}")
        method, tier = spec.arm.split("@")
        rows = ic._p8_4b_rows(output, method, tier, reference)
        _read_rows_digested(output, P8_4B_DIR, spec, reference, rows, p8_pairs)
        cells[spec.cell_id] = Cell(spec=spec, rows=_merged_rows(spec, reference, rows))

    anchor = assert_anchor_identity(payloads[ANCHOR_CELL], payloads[REROLLED_ANCHOR_CELL], cells[ANCHOR_CELL].rows,
                                    cells[REROLLED_ANCHOR_CELL].rows)
    note_checks = {"random": ic.assert_c1_note_means(output, campaign["cells"], note["payload"])}
    for tier in ("mappo1000", "maxpressure", "fixedtime"):
        arms = {method: _book_arm(method, tier) for method in ts.METHODS}
        note_checks[tier] = assert_note_means(output, campaign["cells"], note["payload"], tier=tier, arms=arms)

    protocol = ic.original_protocol(payloads[ORIGINAL_IQL])
    ic.assert_protocol(protocol)
    record = read_listed_json(output, P5_2B_TRAINING_RECORD, sums[P5_2B_SUMS_NAME], manifest_name=P5_2B_SUMS_NAME)
    files.append({"file": f"output/{P5_2B_TRAINING_RECORD}", "sha256": sums[P5_2B_SUMS_NAME][P5_2B_TRAINING_RECORD],
                  "anchor": f"its line in output/{P5_2B_SUMS_NAME}"})
    corrected_spec = next(spec for spec in CELL_SPECS if spec.cell_id == CORRECTED_IQL)
    ic.assert_evaluated_models(payloads[CORRECTED_IQL], record)
    corrected_ours = episodes[CORRECTED_IQL]
    corrected_rows = ic._corrected_rows(output / P5_2B_RUN_DIR, protocol, corrected_ours, record)
    corrected_pairs: list[tuple[str, str]] = []
    _read_rows_digested(output, P5_2B_REDERIVATION_DIR, corrected_spec, corrected_ours, corrected_rows,
                        corrected_pairs, listed=sums[P5_2B_SUMS_NAME])
    rederivation = ic._campaign_anchor(output / P5_2B_REDERIVATION_DIR)
    if rederivation["declared_cells_sha256"] != pins.p5_2b_rederivation_declared_cells_sha256:
        raise ValueError(f"P5.2b's re-derivation campaign declares {rederivation['declared_cells_sha256']}, not the "
                         f"pinned {pins.p5_2b_rederivation_declared_cells_sha256}")
    for name in (ar.MANIFEST_NAME, ar.COMPLETE_MARKER):
        relative = f"{P5_2B_REDERIVATION_DIR}/{name}"
        digest = hashlib.sha256((output / relative).read_bytes()).hexdigest()
        if sums[P5_2B_SUMS_NAME].get(relative) != digest:
            raise ValueError(f"output/{relative} is at sha256 {digest}, which does not match its line in "
                             f"{P5_2B_SUMS_NAME}")
        files.append({"file": f"output/{relative}", "sha256": digest, "anchor": f"its line in output/{P5_2B_SUMS_NAME}"})
    cells[CORRECTED_IQL] = Cell(spec=corrected_spec, rows=_merged_rows(corrected_spec, corrected_ours, corrected_rows))

    committed = documents[P5_2B_ARTIFACT]
    for which, cell_id in (("corrected", CORRECTED_IQL), ("original", ORIGINAL_IQL)):
        cell = cells[cell_id]
        mine = {"att_ours": ic.cell_summary(episodes_of(cell, "att_ours")),
                "att_engine": ic.cell_summary(episodes_of(cell, "att_engine")),
                "admission": ic.admission_summary(cell.rows)}
        for key, value in mine.items():
            if committed["cells"][which][key] != value:
                raise ValueError(f"{cell_id}'s {key} block is not P5.2b's committed cells.{which}.{key} "
                                 f"({P5_2B_ARTIFACT}): the cell read is not the cell P5.2b reported")
    if committed["inputs"]["run_manifest"]["sha256"] != pins.p5_2b_sums_sha256:
        raise ValueError(f"{P5_2B_ARTIFACT} names its run manifest at {committed['inputs']['run_manifest']['sha256']}, "
                         f"not the pinned {pins.p5_2b_sums_sha256}")

    provenance = {
        "inputs": sorted(files, key=lambda entry: entry["file"]),
        "protocol": {"seeds": list(seeds), "draws": {"first": draws[0], "last": draws[-1], "n": len(draws)},
                     "source": "dt_gate.TRAINING_SEEDS and dt_gate.HELD_OUT_DRAWS; every cell asserted complete on "
                               "this grid (tier_sweep.cell_is_complete)"},
        "p8_4b_campaign": {"declared_cells_sha256": campaign["declared_cells_sha256"], "n_cells": campaign["n_cells"],
                           "engine_seed": campaign["engine_seed"],
                           "anchor": "iql_correction.assert_p8_4b_campaign: the digest recomputed from the manifest's "
                                     "own cell list equals CAMPAIGN_COMPLETE's and the pin, and the campaign is "
                                     "complete; each arm's declared cells are its evaluation record's episodes"},
        "p8_4b_rows": {"n": len(p8_pairs), "sha256": rows_digest(p8_pairs),
                       "checks": "iql_correction._p8_4b_rows per row (reproduces_committed, identity, att_ours equal "
                                 "to the evaluation record's att_horizon); the README note's six means on every tier",
                       "construction": ROWS_DIGEST_CONSTRUCTION},
        "p5_2b_rederivation": {**rederivation, "anchor": "iql_correction._campaign_anchor, equal to the pin"},
        "p5_2b_corrected_rows": {"n": len(corrected_pairs), "sha256": rows_digest(corrected_pairs),
                                 "checks": "iql_correction._corrected_rows (identity, att_ours equal to "
                                           "output/p5_2b/eval_random_iql.json, the corrected checkpoint); each file at "
                                           "its line in output/SHA256SUMS_p5_2b.txt",
                                 "construction": ROWS_DIGEST_CONSTRUCTION},
        "p5_2b_run_manifest": {"file": f"output/{P5_2B_SUMS_NAME}", "sha256": pins.p5_2b_sums_sha256,
                               "lines": len(sums[P5_2B_SUMS_NAME])},
        "note_means": note_checks,
        "random_anchor_identity": {**anchor, "cells": [ANCHOR_CELL, REROLLED_ANCHOR_CELL]},
        "d17": d17,
        "protocol_of_the_corrected_cell": {"seeds": list(protocol.seeds), "n_draws": len(protocol.draw_ids),
                                           "engine_seed": protocol.engine_seed,
                                           "deterministic": protocol.deterministic,
                                           "declared_gradient_steps": protocol.declared_gradient_steps,
                                           "read_from": f"output/{ORIGINAL_IQL_EVAL}"},
    }
    records = {"p5_2b_artifact": committed, "p8_4b_artifact": documents[P8_4B_ARTIFACT],
               "p5_1_grid": documents[P5_1_GRID], "stop_rule_record": stop_record, "note": note["payload"],
               "h4_payloads": {cell_id: payloads[cell_id] for cell_id in H4_CELLS}}
    return Inputs(cells=cells, provenance=provenance, records=records)


# ----------------------------------------------------------------------
# Outcomes and contrasts
# ----------------------------------------------------------------------


def interval_outcome(low: float, high: float, *, predicted: str) -> str:
    """HELD / FAILED / NOT RESOLVED for a quantity predicted ``negative`` or ``positive``, by
    ``spatial_mixing._outcome_from_interval`` (on ``(-high, -low)`` for a positive one).  A CI touching zero, and an
    exact zero (CI [0, 0]), is NOT RESOLVED."""
    if predicted == "negative":
        return sm._outcome_from_interval(float(low), float(high))
    if predicted == "positive":
        return sm._outcome_from_interval(-float(high), -float(low))
    raise ValueError(f"predicted must be 'negative' or 'positive', got {predicted!r}")


def conjunction(outcomes: Sequence[str]) -> str:
    """``BRIEF_45`` section 3.3: HELD iff every tier is HELD, FAILED iff any tier is FAILED, NOT RESOLVED otherwise."""
    values = list(outcomes)
    if not values or any(value not in ("HELD", "FAILED", "NOT RESOLVED") for value in values):
        raise ValueError(f"a conjunction needs one or more HELD / FAILED / NOT RESOLVED outcomes, got {values}")
    if "FAILED" in values:
        return "FAILED"
    if all(value == "HELD" for value in values):
        return "HELD"
    return "NOT RESOLVED"


def sign_of(value: float) -> str:
    """``negative`` / ``zero`` / ``positive`` of a point estimate."""
    if value < 0.0:
        return "negative"
    if value > 0.0:
        return "positive"
    return "zero"


def _grid_of(cell: Cell, episodes: Sequence[EpisodeResult]) -> set[tuple[int, int]]:
    keys = {(int(episode.seed), int(episode.draw_id)) for episode in episodes}
    if not keys:
        raise ValueError(f"{cell.spec.cell_id} holds no episode")
    return keys


def _interval_reading(low: float, high: float, exactly_zero: bool, identical: bool) -> str:
    if exactly_zero:
        return ("exactly zero (identical episodes)" if identical
                else "exactly zero on every draw, from episodes that are not identical")
    if high < 0.0:
        return "resolves: the CI lies entirely below zero"
    if low > 0.0:
        return "resolves: the CI lies entirely above zero"
    return "does not resolve: the CI includes zero"


def contrast(left: Cell, right: Cell, *, definition: str, tier: str) -> dict[str, Any]:
    """``left - right`` on the paired unit (module docstring), refused unless the two cells cover the identical
    ``(seed, draw)`` grid.  Carries the CI and its width, ``exactly_zero``, the point estimate's sign,
    ``tier_sweep.per_seed_block`` (D9), ``att_rederivation.contrast_discriminability`` and D8's rank-biserial block from
    ``offline_baselines.paired_comparison`` -- refused unless that call's ``mean_difference``, ``ci95_low`` and
    ``ci95_high`` equal this block's under ``==`` (Amendment A, A2 Q8)."""
    a, b = episodes_of(left, definition), episodes_of(right, definition)
    if _grid_of(left, a) != _grid_of(right, b):
        raise ValueError(f"{left.spec.cell_id} and {right.spec.cell_id} do not cover the same (seed, draw) grid "
                         f"({len(a)} and {len(b)} episodes): a paired contrast is formed on identical grids only")
    left_means, right_means = _per_draw_means(a), _per_draw_means(b)
    draws = sorted(left_means)
    differences = [left_means[draw] - right_means[draw] for draw in draws]
    stats = mean_ci95(differences)
    low, high = stats.mean - stats.ci95, stats.mean + stats.ci95
    exactly_zero = all(value == 0.0 for value in differences)
    comparison = paired_comparison(a, b)
    if (comparison.mean_difference, comparison.ci95_low, comparison.ci95_high) != (stats.mean, low, high):
        raise ValueError(
            f"offline_baselines.paired_comparison gives {comparison.mean_difference!r} "
            f"[{comparison.ci95_low!r}, {comparison.ci95_high!r}] for {left.spec.cell_id} - {right.spec.cell_id} under "
            f"{definition}, and this module {stats.mean!r} [{low!r}, {high!r}]: the registered route is one route, so "
            "the two must be equal under == (BRIEF_45 Amendment A, A2 Q8)")
    per_seed = ts.per_seed_block(left.rows, right.rows, left_name=left.spec.arm, right_name=right.spec.arm,
                                 metric=definition)
    discriminability = ar.contrast_discriminability(
        {(episode.seed, episode.draw_id): episode.att_horizon for episode in a},
        {(episode.seed, episode.draw_id): episode.att_horizon for episode in b},
        tier=tier, left=left.spec.arm, right=right.spec.arm)
    wilcoxon = comparison.wilcoxon
    defined = int(wilcoxon.n_used) > 0
    return {
        "left": left.spec.cell_id, "right": right.spec.cell_id, "left_arm": left.spec.arm, "right_arm": right.spec.arm,
        "definition": definition, "tier": tier,
        "n_draws": len(draws), "n_seeds": len({episode.seed for episode in a}),
        "mean_difference": float(stats.mean), "std": float(stats.std), "ci95_half_width": float(stats.ci95),
        "ci95_low": float(low), "ci95_high": float(high), "ci95_width": float(high - low),
        "exactly_zero": exactly_zero, "sign": sign_of(stats.mean),
        "reading": _interval_reading(low, high, exactly_zero, discriminability.identical),
        "per_seed": per_seed["per_seed"], "seed_reversal_qualifier": per_seed["seed_reversal_qualifier"],
        "discriminability": {"status": discriminability.status, "identical": discriminability.identical,
                             "n_paired": discriminability.n_paired, "n_differing": discriminability.n_differing,
                             "max_abs_difference": discriminability.max_abs_difference,
                             "statement": discriminability.statement},
        "paired_comparison": {
            "rank_biserial": float(comparison.rank_biserial), "rank_biserial_defined": defined,
            "rank_biserial_note": (
                "(W+ - W-) / (W+ + W-) over the per-draw differences; positive means the left arm's ATT is higher "
                "(worse) more often and by larger ranks" if defined else
                "undefined: every per-draw difference is exactly zero (wilcoxon n_used = 0), and "
                "offline_baselines.rank_biserial returns 0.0 by convention; that 0.0 is not an effect size"),
            "median_difference": float(comparison.median_difference), "wins": int(comparison.wins),
            "losses": int(comparison.losses), "ties": int(comparison.ties),
            "wilcoxon": {"w_plus": float(wilcoxon.w_plus), "w_minus": float(wilcoxon.w_minus),
                         "statistic": float(wilcoxon.statistic), "n_used": int(wilcoxon.n_used),
                         "n_zero": int(wilcoxon.n_zero), "z": float(wilcoxon.z), "p_value": float(wilcoxon.p_value)},
            "source": ("offline_baselines.paired_comparison, unchanged (D8); its mean_difference, ci95_low and "
                       "ci95_high equal this block's under =="),
            "p_value_read_by_a_verdict": False,
        },
    }


def composite_contrast(left: tuple[Cell, Cell], right: tuple[Cell, Cell], *, definition: str, tier: str
                       ) -> dict[str, Any]:
    """``I = (left[0] - left[1]) - (right[0] - right[1])`` per draw from the four cells' per-draw means, its CI by
    ``mean_ci95``, the counts of draws where it is negative / zero / positive, and the per-seed values
    ``(mean_s(left[0]) - mean_s(left[1])) - (mean_s(right[0]) - mean_s(right[1]))`` (``BRIEF_45`` section 3.3).  A
    difference of two differences, so no rank-biserial, and the block says so."""
    cells = (*left, *right)
    episodes = [episodes_of(cell, definition) for cell in cells]
    grids = [_grid_of(cell, eps) for cell, eps in zip(cells, episodes)]
    if any(grid != grids[0] for grid in grids):
        raise ValueError(f"{[cell.spec.cell_id for cell in cells]} do not cover the same (seed, draw) grid: I is formed "
                         "on identical grids only")
    means = [_per_draw_means(eps) for eps in episodes]
    draws = sorted(means[0])
    values = [(means[0][d] - means[1][d]) - (means[2][d] - means[3][d]) for d in draws]
    stats = mean_ci95(values)
    low, high = stats.mean - stats.ci95, stats.mean + stats.ci95
    seeds = sorted({int(episode.seed) for episode in episodes[0]})

    def seed_mean(eps: Sequence[EpisodeResult], seed: int) -> float:
        return float(np.mean([episode.att_horizon for episode in eps if int(episode.seed) == seed]))

    by_seed = {str(seed): (seed_mean(episodes[0], seed) - seed_mean(episodes[1], seed))
               - (seed_mean(episodes[2], seed) - seed_mean(episodes[3], seed)) for seed in seeds}
    seed_values = np.asarray(list(by_seed.values()), dtype=np.float64)
    pooled = float(seed_values.mean())
    against = int(sum(1 for value in by_seed.values() if (value < 0) != (pooled < 0) and value != 0.0))
    per_seed = {"by_seed": by_seed, "n_seeds": len(seeds), "mean": pooled,
                "between_seed_sd": float(seed_values.std(ddof=1)) if len(seeds) > 1 else 0.0,
                "min": float(seed_values.min()), "max": float(seed_values.max()),
                "n_seeds_negative": int((seed_values < 0).sum()), "reverses_against_pooled_sign": against,
                "definition": ("per training seed, (mean over draws of left[0] - of left[1]) - (mean over draws of "
                               "right[0] - of right[1]); the pooled sign is the mean of the per-seed values, as "
                               "spatial_mixing.per_seed_advantages takes it")}
    exactly_zero = all(value == 0.0 for value in values)
    return {
        "inputs": [cell.spec.cell_id for cell in cells], "definition": definition, "tier": tier,
        "n_draws": len(draws), "n_seeds": len(seeds),
        "mean_difference": float(stats.mean), "std": float(stats.std), "ci95_half_width": float(stats.ci95),
        "ci95_low": float(low), "ci95_high": float(high), "ci95_width": float(high - low),
        "exactly_zero": exactly_zero, "sign": sign_of(stats.mean),
        "draws_negative": sum(1 for value in values if value < 0.0),
        "draws_zero": sum(1 for value in values if value == 0.0),
        "draws_positive": sum(1 for value in values if value > 0.0),
        "per_seed": per_seed,
        "seed_reversal_qualifier": sm.seed_reversal_qualifier(per_seed, "d4", "d1"),
        "rank_biserial": None,
        "rank_biserial_note": ("none: I is a difference of two differences, not a pair of arms, so D8's rank-biserial "
                               "does not apply (BRIEF_45 Amendment A, A2 Q8)"),
    }


# ----------------------------------------------------------------------
# Orderings and ties
# ----------------------------------------------------------------------


def tier_composer(cells: Mapping[tuple[str, str], Sequence[EpisodeResult]], *, tier: str, with_hard_subset: bool
                  ) -> dict[str, Any]:
    """``iql_correction.random_tier_statements`` restated for any out-of-sample tier (plan F2, Amendment A, A2 Q2): at
    ``tier="random"`` it returns every field that function returns, equal, and may add fields."""
    raise NotImplementedError("P5.2c C1 skeleton: tier_composer")


def level_groups(levels: Mapping[str, float]) -> list[list[str]]:
    """Arms grouped by EXACTLY equal level, groups in ascending level, arms within a group in the input's order."""
    raise NotImplementedError("P5.2c C1 skeleton: level_groups")


def tie_audit(levels: Mapping[str, float], predicted: Sequence[str], *, subject: str = "dt_nomix") -> dict[str, Any]:
    """Every tie-dependent integer of one ordering: Q2b's and Q2b-hard's concordance under the declared tie rule and its
    range over every order consistent with *levels*, the possible first places, and *subject*'s rank (1 + the arms
    strictly lower) with its range; with the number of orders enumerated."""
    raise NotImplementedError("P5.2c C1 skeleton: tie_audit")


# ----------------------------------------------------------------------
# The statements
# ----------------------------------------------------------------------


def score_q0(book: Mapping[tuple[str, str], Cell], *, definition: str) -> list[dict[str, Any]]:
    """Q0's ``d1`` (context), Q0a, Q0b and the stop rule at ``mappo1000``."""
    raise NotImplementedError("P5.2c C1 skeleton: score_q0")


def score_q1(book: Mapping[tuple[str, str], Cell], *, definition: str) -> list[dict[str, Any]]:
    """Q1 (19 cells, ``tier_sweep.score_level``), Q1b per tier and Q1b's two-cell aggregate."""
    raise NotImplementedError("P5.2c C1 skeleton: score_q1")


def score_q2_q3(book: Mapping[tuple[str, str], Cell], *, definition: str) -> list[dict[str, Any]]:
    """Per out-of-sample tier: the ranking, Q2a, Q2b, Q2b-hard, D9, Q3a and Q3c (from :func:`tier_composer`, with the tie
    audit added); Q2b's aggregate; and ``mappo1000``'s Q3a and Q3c as context."""
    raise NotImplementedError("P5.2c C1 skeleton: score_q2_q3")


def score_q3b(book: Mapping[tuple[str, str], Cell], *, definition: str) -> list[dict[str, Any]]:
    """Q3b per out-of-sample tier (interval rule, predicted negative; the sub-prediction matched or not), its aggregate,
    and ``mappo1000`` as context."""
    raise NotImplementedError("P5.2c C1 skeleton: score_q3b")


def score_q4(book: Mapping[tuple[str, str], Cell], *, definition: str) -> list[dict[str, Any]]:
    """``d(t)`` on four tiers, Q4a per out-of-sample tier and its aggregate, Q4b on the four-rung chain and the bullet's
    three-rung form."""
    raise NotImplementedError("P5.2c C1 skeleton: score_q4")


def score_q5(book: Mapping[tuple[str, str], Cell], *, definition: str) -> list[dict[str, Any]]:
    """Q5a (three tiers), Q5b and Q5c (four tiers), each per tier and aggregated by :func:`conjunction`."""
    raise NotImplementedError("P5.2c C1 skeleton: score_q5")


def score_q6(book: Mapping[tuple[str, str], Cell], *, definition: str) -> list[dict[str, Any]]:
    """Q6 for ``dt_spatial`` and ``dt_nomix`` on four tiers against the shared anchor; P5.1's three-valued outcome
    beside it from ``spatial_mixing.collapse_criterion``, refused unless that call's interval equals this one's."""
    raise NotImplementedError("P5.2c C1 skeleton: score_q6")


def all_statements(books: Mapping[str, Mapping[tuple[str, str], Cell]]) -> list[dict[str, Any]]:
    """Every statement under both definitions; version-dependent ones in both versions, every other one computed in
    both, refused unless identical, and stored once with ``version: "both"``; ``definition_dependent`` set."""
    raise NotImplementedError("P5.2c C1 skeleton: all_statements")


def leaders_block(books: Mapping[str, Mapping[tuple[str, str], Cell]]) -> list[dict[str, Any]]:
    """Descriptive, not registered: per tier, definition and (at ``random``) version, the level order with ties, the
    leader(s), the runner(s)-up, leader - runner-up as a contrast, and the per-seed first places."""
    raise NotImplementedError("P5.2c C1 skeleton: leaders_block")


def ties_block(books: Mapping[str, Mapping[tuple[str, str], Cell]]) -> list[dict[str, Any]]:
    """Per tier, definition and (at ``random``) version: exactly-equal levels, identical-episode pairs with the fields
    they share, and every tie-dependent integer with its value under the rule and its range, pooled and per seed."""
    raise NotImplementedError("P5.2c C1 skeleton: ties_block")


def collapse_audit(books: Mapping[str, Mapping[tuple[str, str], Cell]]) -> dict[str, Any]:
    """``descriptive_collapse_audit`` (Amendment A, A2 Q9), not registered: Q6's rule applied verbatim to the non-DT
    arms on four tiers, both definitions, both random-tier IQL cells; whether the README's sentence holds."""
    raise NotImplementedError("P5.2c C1 skeleton: collapse_audit")


def cells_block(cells: Mapping[str, Cell]) -> dict[str, Any]:
    """Per cell: n, level, CI, per-seed means, median and IQR under each definition; A11(b)'s companions; A5's vehicle
    count; its sources."""
    out: dict[str, Any] = {}
    for cell_id, cell in cells.items():
        spec = cell.spec
        entry: dict[str, Any] = {"method": spec.method, "tier": spec.tier, "source": spec.source, "arm": spec.arm,
                                 "n": len(cell.rows)}
        for definition in DEFINITIONS:
            episodes = episodes_of(cell, definition)
            summary = ic.cell_summary(episodes)
            values = np.asarray([episode.att_horizon for episode in episodes], dtype=np.float64)
            q25, q50, q75 = (float(x) for x in np.percentile(values, [25, 50, 75], method="linear"))
            entry[definition] = {**summary, "ci95_width": float(summary["ci95_high"] - summary["ci95_low"]),
                                 "median": q50, "iqr": [q25, q75], "iqr_width": q75 - q25}
        entry["companions"] = ic.admission_summary(cell.rows)
        entry["horizon_vehicle_count_mean"] = float(mean_ci95([float(r["horizon_vehicle_count"])
                                                               for r in cell.rows]).mean)
        rows_from = (f"output/{P5_2B_REDERIVATION_DIR}/cell_grid4x4_iql_at_random_*" if spec.source == "p5_2b" else
                     f"output/{P8_4B_DIR}/" + ar.cell_file_name(ar.CellKey(scenario=SCENARIO, arm=spec.arm, seed=0,
                                                                           draw_id=0)).replace("seed0_draw0", "*"))
        entry["sources"] = {"att_ours": f"output/{spec.eval_file}", "horizon_vehicle_count": f"output/{spec.eval_file}",
                            "att_engine": rows_from, "companions": rows_from}
        entry["quantiles"] = "numpy.percentile(method='linear') over the cell's episodes"
        out[cell_id] = entry
    return out


# ----------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Quote:
    """A registered sentence, verbatim but for ``**`` emphasis and line breaks, at ``file:lines``.  :func:`normalise`
    applied to the cited lines must contain :func:`normalise` of ``text`` (T-quotes)."""

    file: str
    lines: tuple[int, int]
    text: str


def normalise(text: str) -> str:
    """The comparison form of a quotation: blockquote markers and ``**`` removed, ``\\*`` unescaped, U+2212 read as
    ``-``, whitespace collapsed."""
    kept = []
    for line in str(text).split("\n"):
        stripped = line.strip()
        while stripped.startswith(">"):
            stripped = stripped[1:].strip()
        kept.append(stripped)
    joined = " ".join(kept).replace("**", "").replace("\\*", "*").replace("−", "-")
    return " ".join(joined.split())


#: The registered text each statement implements, quoted from the files (``BRIEF_45`` section 3.2; plan section 3).
RULES: Mapping[str, tuple[Quote, ...]] = {
    "common": (Quote("docs/plans/p5.2.md", (407, 412),
                     "Primary metric `att_horizon` (A1). Paired unit = per-draw mean over the 5 seeds across the 100 "
                     "shared held-out draws (`dt_gate._per_draw_means`). Interval = the 95 % CI of the mean paired "
                     "difference (`mean_ci95`, `ddof=1`, `1.96 σ/√n`). Lower ATT is better. Every outcome is one of "
                     "HELD / FAILED / NOT RESOLVED, and `NOT RESOLVED` is always reported with the CI width rather than "
                     "folded into either neighbour."),),
    "Q0": (Quote("docs/plans/p5.2.md", (513, 515),
                 "d1 = ATT(dt_spatial@1H) − ATT(dt_nomix@1H) measured: +39.5649, CI [+36.0510, +43.0787] "
                 "d4 = ATT(dt_spatial@4H) − ATT(dt_nomix@4H) to be measured I = d4 − d1 the reported quantity"),),
    "Q0a": (Quote("docs/plans/p5.2.md", (521, 522),
                  "Q0a — HELD iff the CI of `mean(d4)` lies entirely above 0 (mixing still hurts at 4 heads). FAILED "
                  "iff entirely below. NOT RESOLVED otherwise."),),
    "Q0b": (Quote("docs/plans/p5.2.md", (523, 524),
                  "Q0b — HELD iff the CI of `mean(I)` includes 0 (the head count does not change the contrast). FAILED "
                  "otherwise, with its sign reported."),),
    "Q0.stop_rule": (Quote("docs/plans/p5.2.md", (526, 528),
                           "THE STOP RULE, declared as §2 requires and enforced mechanically by the campaign script "
                           "rather than remembered: if the 95 % CI of `mean(d4)` lies ENTIRELY BELOW 0 — spatial mixing "
                           "helps at 4 heads — the ladder sweep is NOT RUN."),
                     Quote("docs/plans/p5.2.md", (529, 529),
                           "A CI that straddles zero is not a reversal and phase B proceeds")),
    "Q1": (Quote("docs/plans/p5.2.md", (560, 560),
                 "Per cell: HELD iff `|measured − predicted| / predicted ≤ 0.30`."),
           Quote("docs/plans/p5.2.md", (561, 563),
                 "Aggregate (the registered claim): HELD iff ≥ `k` of the `N` out-of-sample cells hold, where "
                 "`k = ceil(9/13 · N)` — the rule that carries the originally registered rate to any denominator (I3). "
                 "At today's `N = 19` that is `k = 14`.")),
    "Q1b": (Quote("docs/plans/p5.2.md", (576, 577),
                  "Q1b (instrument check, scored separately, not in the denominator): `behaviour@maxpressure` lands "
                  "within ±10 % of `167.4920`."),
            Quote("docs/plans/p5.2.md", (497, 497),
                  "| `behaviour` | 160.2780* | 167.4920 (Q1b) | 206.9318 (Q1b) |")),
    "Q2a": (Quote("docs/plans/p5.2.md", (604, 605),
                  "Q2a (first place): HELD on a tier iff the predicted 1st-place arm is the measured 1st-place arm. "
                  "Scored on the 2 out-of-sample tiers and reported per tier."),),
    "tie_rule": (Quote("docs/plans/p5.2.md", (606, 606),
                       "TIE-BREAK RULE, declared 2026-08-24 after `fixedtime` produced two EXACT ties"),
                 Quote("docs/plans/p5.2.md", (608, 609),
                       "A tie is scored as DISCORDANT for both orders: neither arm is placed ahead of the other, so a "
                       "predicted ordering between two tied arms cannot be credited."),
                 Quote("docs/plans/p5.2.md", (615, 616),
                       "Under the rule declared here the count is the lower end, 7 of 18, and the range must travel "
                       "with it.")),
    "Q2b": (Quote("docs/plans/p5.2.md", (618, 619),
                  "Q2b (concordance): of the 15 unordered arm pairs per tier, the count ordered as predicted. HELD iff "
                  "≥ 12 of 15 on each out-of-sample tier (≥ 24 of 30 overall)."),),
    "Q2b_hard": (Quote("docs/plans/p5.2.md", (621, 623),
                       "Q2b-hard (an ADDITIONAL report, registered here before any data exists — C4): the same "
                       "concordance restricted to the 6 pairs among `{dt_nomix, bc_top10_perix, bc, dt_spatial}`, "
                       "reported per tier beside the 15-pair count and never in place of it."),
                 Quote("docs/plans/p5.2.md", (630, 631),
                       "No threshold is registered on the hard subset — it is reported as a count out of 6 per tier")),
    "D9": (Quote("docs/plans/p5.2.md", (633, 634),
                 "Both are reported with the per-seed ordering beside the pooled one (D9): a first place that reverses "
                 "on a seed is reported as reversing."),
           Quote("docs/plans/p5.2.md", (316, 316),
                 "Every comparison carries its per-seed agreement count and its per-seed range, and a non-zero "
                 "`reverses_against_pooled_sign` emits the qualifier as a field")),
    "Q3": (Quote("docs/plans/p5.2.md", (645, 646),
                 "Out-of-sample tiers only (`maxpressure`, `fixedtime`, `random`); `mappo1000` is the seen cell and is "
                 "reported as context, never scored."),),
    "Q3a": (Quote("docs/plans/p5.2.md", (648, 648),
                  "Q3a (rank): `dt_nomix` has the lowest held-out mean of the 6 method arms."),),
    "Q3b": (Quote("docs/plans/p5.2.md", (651, 654),
                  "Q3b (against its own behaviour policy): `mean[ATT(dt_nomix) − ATT(behaviour)] < 0`. Rule says "
                  "−0.9655 at `maxpressure`, −1.9906 at `fixedtime` and −10.1702 at `random`. Registered: sign "
                  "negative on all three; the CI excludes 0 at `random` and is predicted NOT to resolve at "
                  "`maxpressure` or `fixedtime`, whose predicted margins are under 2 ATT."),
            Quote("docs/plans/p5.2.md", (319, 319),
                  "On the `random` tier the behaviour anchor and the collapse reference are consequently the same "
                  "cell, reported under both roles with the identity stated, not rolled twice")),
    "Q3c": (Quote("docs/plans/p5.2.md", (655, 659),
                  "Q3c (against the best non-DT arm): `mean[ATT(dt_nomix) − ATT(best non-DT arm)]`, where the best "
                  "non-DT arm is the minimum-mean arm of `{bc, bc_top10, bc_top10_perix, iql}` measured on that tier "
                  "(a mechanical definition, fixed here, not chosen after the fact). Rule says +0.4326 at "
                  "`maxpressure` (`bc_top10_perix` leads), −17.1639 at `fixedtime` and −21.9354 at `random` (`bc` is "
                  "the best non-DT arm at both)."),),
    "Q4": (Quote("docs/plans/p5.2.md", (667, 667),
                 "Per tier, `d(t) = mean[ATT(dt_spatial) − ATT(dt_nomix)]`."),),
    "Q4a": (Quote("docs/plans/p5.2.md", (669, 669),
                  "Q4a: `d(t) > 0` with the CI excluding 0, on all three out-of-sample tiers."),),
    "Q4b": (Quote("docs/plans/p5.2.md", (670, 672),
                  "`d` increases monotonically with worse data, i.e. `d(mappo1000) < d(maxpressure) < d(random)`. "
                  "Scored on the point estimates and reported with all three CIs and the per-seed ranges."),
            Quote("docs/plans/p5.2.md", (673, 673),
                  "Predicted, in measured-ATT tier order: +39.5649* → +41.7402 → +51.3689 → +62.7106."),
            Quote("docs/plans/p5.2.md", (674, 675),
                  "`fixedtime` is what makes Q4b a real monotonicity test rather than a two-point line: three rungs and "
                  "a gap became four graded rungs, and the new value falls between its neighbours.")),
    "Q5a": (Quote("docs/plans/p5.2.md", (683, 684),
                  "Q5a (the load sorter replicates): `mean[ATT(bc_top10) − ATT(bc)] > 0` with the CI excluding 0 on "
                  "all three out-of-sample tiers. Predicted +575.34, +820.65 and +970.57."),),
    "Q5b": (Quote("docs/plans/p5.2.md", (685, 686),
                  "Q5b (the per-intersection filter repairs it): `mean[ATT(bc_top10_perix) − ATT(bc_top10)] < 0` with "
                  "the CI excluding 0 on all four tiers. Predicted −583.81, −584.97, −812.15, −967.88."),),
    "Q5c": (Quote("docs/plans/p5.2.md", (687, 689),
                  "Q5c (sign pattern against plain BC): `bc_top10_perix` is better than `bc` at `mappo1000` and "
                  "`maxpressure` and worse at `fixedtime` and `random` — mirroring `cf_hz1x1`'s measured %BC sign "
                  "change (§1b R2), which also flipped at `fixedtime`. Predicted −3.21, −9.63, +8.49, +2.69."),
            Quote("docs/plans/p5.2.md", (691, 692),
                  "Q5c the least certain of the three — its `random` conjunct is +2.69 ATT and may not resolve.")),
    "Q6": (Quote("docs/plans/p5.2.md", (696, 697),
                 "Reference: the shared `random` anchor cell (D12), 260.3602, on the same 100 held-out draws — a "
                 "shared-draw comparison, not an A5-void cross-draw-set one."),
           Quote("docs/plans/p5.2.md", (699, 701),
                 "COLLAPSED (for a DT arm on a tier) iff its held-out mean `att_horizon` is worse than the `random` "
                 "anchor's with the paired CI excluding 0. Reported for `dt_spatial` and `dt_nomix` separately, per "
                 "tier."),
           Quote("docs/plans/p5.1.md", (212, 213),
                 "COLLAPSED (for either DT arm) iff its held-out mean `att_horizon` is worse than the `random` "
                 "behaviour arm's, with the paired CI excluding 0.")),
    "Q6.expectation": (Quote("docs/plans/p5.2.md", (705, 706),
                             "`dt_spatial` COLLAPSES on the `random` tier (predicted 312.90 against 260.36) and nowhere "
                             "else; `dt_nomix` collapses nowhere."),),
    "D8": (Quote("docs/plans/p5.2.md", (315, 315),
                 "Every pair reports a paired difference, a CI, the CI width and a rank-biserial; no verdict string."),),
    "D15": (Quote("docs/plans/p5.2.md", (324, 324),
                  "every arm reports median, IQR and `vehicle_count` at the horizon (A5, unconditional), and any arm "
                  "whose distribution is visibly not unimodal is reported with the distributional comparison beside "
                  "its mean"),),
    "not_registered": (Quote("docs/plans/p5.2.md", (839, 839),
                             "No level prediction for the two 4-head cells (§4.0.1)."),
                       Quote("docs/plans/p5.2.md", (843, 843), "No equivalence verdict anywhere (D8).")),
    "rule_r": (Quote("PREREGISTRATION.md", (397, 397),
                     "PASS on S: `att_engine` is the primary metric for every claim on S, and `att_ours` is reported "
                     "beside it in every table with the three counts."),
               Quote("PREREGISTRATION.md", (397, 397), "both verdicts are reported whether or not they agree."),
               Quote("PREREGISTRATION.md", (397, 397),
                     "Every reported ATT cell carries, unconditionally, all five of `att_ours`, `att_engine`, "
                     "`entered`, `created` and `never_entered`, re-derived by deterministic replay.")),
    "att_engine_label": (Quote("docs/briefs/BRIEF_44_p5.2b_iql_correction.md", (224, 225),
                               "the `q1` block under `att_engine` itself carries the flag that the predictions are the "
                               "registered `att_horizon`-era values"),),
}


@dataclass(frozen=True)
class RecordRow:
    """One value P5.2's record states, at ``file:lines``, and the key of this artifact's value at that precision
    (:func:`record_values`).  ``exception`` names a declared, verified reason the two differ."""

    file: str
    lines: tuple[int, int]
    recorded: str
    value: str
    exception: str | None = None


_PACKET = "docs/returns/P5.2.md"
_BRIEF_27 = "docs/briefs/BRIEF_27_p5.2_tier_sweep.md"
_PLAN = "docs/plans/p5.2.md"

#: Every value P5.2's packet (section 1, D-3 and the findings), ``BRIEF_27`` GATE 2 and Note N, and P5.2's plan (after
#: the data) record for the statements this artifact scores, under ``as_committed`` and ``att_ours`` -- the basis they
#: were scored on.  Q5's verdict is not a literal row: :data:`Q5_RECORDS` carries it (Amendment A, A3).
RECORD: tuple[RecordRow, ...] = (
    RecordRow(_PACKET, (24, 24), "did not fire", "q0.stop_rule"),
    RecordRow(_PACKET, (25, 25), "HELD, 15 of 19", "q1.outcome"),
    RecordRow(_PACKET, (25, 25), "0.04 % / 0.09 %", "q1b.relative_errors"),
    RecordRow(_PACKET, (26, 26), "FAILED 3/3", "q2a.failed"),
    RecordRow(_PACKET, (26, 26), "`bc_top10_perix`/`dt_nomix`/`dt_nomix`", "q2a.predicted"),
    RecordRow(_PACKET, (26, 26), "`bc`/`iql`/`iql`", "q2a.observed"),
    RecordRow(_PACKET, (27, 27), "FAILED 3/3", "q2b.failed"),
    RecordRow(_PACKET, (27, 27), "11 · 8 · 8", "q2b.counts", "deferred_108"),
    RecordRow(_PACKET, (28, 28), "FAILED 3/3", "q3a.failed"),
    RecordRow(_PACKET, (28, 28), "2/6 · 3/6 · 3/6", "q3a.ranks", "q3a_fixedtime"),
    RecordRow(_PACKET, (29, 29), "INVERTED", "q4b.reading"),
    RecordRow(_PACKET, (31, 31), "no DT arm collapsed on any tier", "q6.none"),
    RecordRow(_PACKET, (34, 34), "1042.75", "q1.predicted.bc_top10@fixedtime"),
    RecordRow(_PACKET, (34, 34), "207.47", "q1.measured.bc_top10@fixedtime"),
    RecordRow(_PACKET, (34, 34), "360.85", "q1.predicted.iql@fixedtime"),
    RecordRow(_PACKET, (34, 34), "203.95", "q1.measured.iql@fixedtime"),
    RecordRow(_PACKET, (35, 35), "1242.69", "q1.predicted.bc_top10@random"),
    RecordRow(_PACKET, (35, 35), "350.34", "q1.measured.bc_top10@random"),
    RecordRow(_PACKET, (35, 35), "414.99", "q1.predicted.iql@random"),
    RecordRow(_PACKET, (35, 35), "190.96", "q1.measured.iql@random"),
    RecordRow(_PACKET, (42, 42), "+39.5649", "q4.mappo1000.mean"),
    RecordRow(_PACKET, (42, 42), "[+36.0510, +43.0787]", "q4.mappo1000.ci"),
    RecordRow(_PACKET, (42, 42), "5/5", "q4.mappo1000.seeds"),
    RecordRow(_PACKET, (43, 43), "+0.7267", "q4.maxpressure.mean"),
    RecordRow(_PACKET, (43, 43), "[+0.6206, +0.8328]", "q4.maxpressure.ci"),
    RecordRow(_PACKET, (43, 43), "5/5", "q4.maxpressure.seeds"),
    RecordRow(_PACKET, (44, 44), "0.0000", "q4.fixedtime.mean"),
    RecordRow(_PACKET, (44, 44), "exact", "q4.fixedtime.ci"),
    RecordRow(_PACKET, (44, 44), "n/a", "q4.fixedtime.seeds"),
    RecordRow(_PACKET, (45, 45), "−0.7063", "q4.random.mean"),
    RecordRow(_PACKET, (45, 45), "[−1.3172, −0.0953]", "q4.random.ci"),
    RecordRow(_PACKET, (45, 45), "4/5", "q4.random.seeds"),
    RecordRow(_PACKET, (48, 49), "39.56 at the best tier against −0.71 at the worst", "q4.endpoints"),
    RecordRow(_PACKET, (58, 59), "0 of 500 episodes differ between all three", "fixedtime.three_identical"),
    RecordRow(_PACKET, (59, 59), "`bc_top10_perix` is likewise identical to `bc`", "fixedtime.perix_identical"),
    RecordRow(_PACKET, (59, 60), "`bc` differs from `behaviour` on 500 of 500", "fixedtime.bc_vs_behaviour"),
    RecordRow(_PACKET, (66, 67), "Three contrasts are exactly `0.0000` at that tier", "fixedtime.zero_contrasts"),
    RecordRow(_PACKET, (74, 74), "−0.7294", "q3b.maxpressure.mean"),
    RecordRow(_PACKET, (74, 74), "CI [−0.9139, −0.5450]", "q3b.maxpressure.ci_prefixed"),
    RecordRow(_PACKET, (74, 74), "−5.0360", "q3b.random.mean"),
    RecordRow(_PACKET, (74, 74), "CI [−5.5005, −4.5716]", "q3b.random.ci_prefixed"),
    RecordRow(_PACKET, (75, 75), "equals it exactly on `fixedtime`", "q3b.fixedtime.exact"),
    RecordRow(_PACKET, (79, 79), "the DT arms are the only arms that never collapse", "audit.only_dt"),
    RecordRow(_PACKET, (87, 87), "2/6 · 2/6 · 3/6 = 7 of 18", "q2b_hard.counts_total"),
    RecordRow(_PACKET, (92, 93), "`maxpressure` 2..2 · `fixedtime` 2..4 · `random` 3..3, i.e. 7..9 of 18",
              "q2b_hard.ranges_packet"),
    RecordRow(_PACKET, (95, 95), "7 of 18", "q2b_hard.total"),
    RecordRow(_PACKET, (96, 96), "(max 9 < 12)", "q2b.fixedtime_max"),
    RecordRow(_PACKET, (96, 96), "(rank ≥ 2 always)", "q3a.rank_floor"),
    RecordRow(_PACKET, (100, 100), "27/45", "q2b.overall", "deferred_108"),
    RecordRow(_PACKET, (265, 265), "679.96 on `maxpressure`", "level.bc_top10@maxpressure"),
    RecordRow(_PACKET, (266, 266), "210.27", "level.bc_top10_perix@maxpressure.bare"),
    RecordRow(_PACKET, (567, 567), "`+39.5649` at the best tier, `−0.7063` at the", "q4.endpoints_packet"),
    RecordRow(_PACKET, (570, 570), "2/6 · 3/6 · 3/6", "q3a.ranks", "q3a_fixedtime"),
    RecordRow(_BRIEF_27, (1216, 1216), "HELD, 15 of 19", "q1.outcome"),
    RecordRow(_BRIEF_27, (1216, 1216), "(threshold 14)", "q1.threshold"),
    RecordRow(_BRIEF_27, (1216, 1216), "0.04 % / 0.09 %", "q1b.relative_errors"),
    RecordRow(_BRIEF_27, (1219, 1219), "1042.75", "q1.predicted.bc_top10@fixedtime"),
    RecordRow(_BRIEF_27, (1219, 1219), "207.47", "q1.measured.bc_top10@fixedtime"),
    RecordRow(_BRIEF_27, (1219, 1219), "80.1 %", "q1.relative_error.bc_top10@fixedtime"),
    RecordRow(_BRIEF_27, (1220, 1220), "360.85", "q1.predicted.iql@fixedtime"),
    RecordRow(_BRIEF_27, (1220, 1220), "203.95", "q1.measured.iql@fixedtime"),
    RecordRow(_BRIEF_27, (1220, 1220), "(43.5 %)", "q1.relative_error.iql@fixedtime.parenthesised"),
    RecordRow(_BRIEF_27, (1220, 1220), "1242.69", "q1.predicted.bc_top10@random"),
    RecordRow(_BRIEF_27, (1220, 1220), "350.34", "q1.measured.bc_top10@random"),
    RecordRow(_BRIEF_27, (1221, 1221), "(71.8 %)", "q1.relative_error.bc_top10@random.parenthesised"),
    RecordRow(_BRIEF_27, (1221, 1221), "414.99", "q1.predicted.iql@random"),
    RecordRow(_BRIEF_27, (1221, 1221), "190.96", "q1.measured.iql@random"),
    RecordRow(_BRIEF_27, (1221, 1221), "(54.0 %)", "q1.relative_error.iql@random.parenthesised"),
    RecordRow(_BRIEF_27, (1226, 1226), "FAILED 3/3", "q2a.failed"),
    RecordRow(_BRIEF_27, (1226, 1226), "`bc_top10_perix`/`dt_nomix`/`dt_nomix`", "q2a.predicted"),
    RecordRow(_BRIEF_27, (1227, 1227), "`bc`/`iql`/`iql`", "q2a.observed"),
    RecordRow(_BRIEF_27, (1227, 1227), "11 · 8 · 8", "q2b.counts", "deferred_108"),
    RecordRow(_BRIEF_27, (1228, 1228), "FAILED 3/3", "q2b.failed"),
    RecordRow(_BRIEF_27, (1231, 1231), "27/45", "q2b.overall", "deferred_108"),
    RecordRow(_BRIEF_27, (1234, 1234), "2/6 · 2/6 · 3/6", "q2b_hard.counts"),
    RecordRow(_BRIEF_27, (1234, 1234), "7 of 18", "q2b_hard.total"),
    RecordRow(_BRIEF_27, (1237, 1238),
              "`maxpressure` 2..2 · `fixedtime` 2..4 · `random` 3..3, giving a total range of 7..9",
              "q2b_hard.ranges_gate2"),
    RecordRow(_BRIEF_27, (1240, 1240), "11 · 8 · 8", "q2b.counts", "deferred_108"),
    RecordRow(_BRIEF_27, (1242, 1242), "(max 9 < 12)", "q2b.fixedtime_max"),
    RecordRow(_BRIEF_27, (1242, 1242), "(rank ≥ 2 always)", "q3a.rank_floor"),
    RecordRow(_BRIEF_27, (1249, 1249), "2/6 · 3/6 · 3/6", "q3a.ranks", "q3a_fixedtime"),
    RecordRow(_BRIEF_27, (1249, 1250), "every paired difference against the best non-DT arm resolves against the DT",
              "q3c.all_against"),
    RecordRow(_BRIEF_27, (1252, 1252), "`−0.7294`, CI [−0.9139, −0.5450]", "q3b.maxpressure.gate2"),
    RecordRow(_BRIEF_27, (1252, 1252), "equals it EXACTLY on `fixedtime`", "q3b.fixedtime.exact_upper"),
    RecordRow(_BRIEF_27, (1254, 1254), "the DT arms are the ONLY arms that never collapse", "audit.only_dt_upper"),
    RecordRow(_BRIEF_27, (1260, 1260), "+39.5649", "q4.mappo1000.mean"),
    RecordRow(_BRIEF_27, (1260, 1260), "[+36.0510, +43.0787]", "q4.mappo1000.ci"),
    RecordRow(_BRIEF_27, (1260, 1260), "5/5", "q4.mappo1000.seeds"),
    RecordRow(_BRIEF_27, (1261, 1261), "+0.7267", "q4.maxpressure.mean"),
    RecordRow(_BRIEF_27, (1261, 1261), "[+0.6206, +0.8328]", "q4.maxpressure.ci"),
    RecordRow(_BRIEF_27, (1261, 1261), "5/5", "q4.maxpressure.seeds"),
    RecordRow(_BRIEF_27, (1262, 1262), "0.0000", "q4.fixedtime.mean"),
    RecordRow(_BRIEF_27, (1262, 1262), "exact", "q4.fixedtime.ci"),
    RecordRow(_BRIEF_27, (1262, 1262), "n/a", "q4.fixedtime.seeds"),
    RecordRow(_BRIEF_27, (1263, 1263), "−0.7063", "q4.random.mean"),
    RecordRow(_BRIEF_27, (1263, 1263), "[−1.3172, −0.0953]", "q4.random.ci"),
    RecordRow(_BRIEF_27, (1263, 1263), "4/5", "q4.random.seeds"),
    RecordRow(_BRIEF_27, (1265, 1266), "on `random` the sign REVERSES with the CI excluding zero", "q4.random_reverses"),
    RecordRow(_BRIEF_27, (1267, 1267), "39.56 at the best tier against −0.71 at the worst", "q4.endpoints"),
    RecordRow(_BRIEF_27, (1276, 1276), "−469.69", "q5b.maxpressure.mean"),
    RecordRow(_BRIEF_27, (1276, 1276), "[−477.81, −461.58]", "q5b.maxpressure.ci"),
    RecordRow(_BRIEF_27, (1276, 1276), "−49.16", "q5b.random.mean"),
    RecordRow(_BRIEF_27, (1277, 1277), "[−55.26, −43.05]", "q5b.random.ci"),
    RecordRow(_BRIEF_27, (1277, 1277), "−0.28", "q5b.fixedtime.mean"),
    RecordRow(_BRIEF_27, (1277, 1277), "[−0.67, +0.11] unresolved", "q5b.fixedtime.ci_unresolved"),
    RecordRow(_BRIEF_27, (1278, 1278), "679.96 on `maxpressure`", "level.bc_top10@maxpressure"),
    RecordRow(_BRIEF_27, (1279, 1279), "(210.27)", "level.bc_top10_perix@maxpressure"),
    RecordRow(_BRIEF_27, (1621, 1621), "+39.5649", "q0.d1.mean"),
    RecordRow(_BRIEF_27, (1621, 1621), "[+36.0510, +43.0787]", "q0.d1.ci"),
    RecordRow(_BRIEF_27, (1622, 1622), "+27.1004", "q0.d4.mean"),
    RecordRow(_BRIEF_27, (1622, 1622), "[+23.8150, +30.3858]", "q0.d4.ci"),
    RecordRow(_BRIEF_27, (1623, 1623), "−12.4645", "q0.i.mean"),
    RecordRow(_BRIEF_27, (1623, 1623), "[−16.3575, −8.5714]", "q0.i.ci"),
    RecordRow(_BRIEF_27, (1628, 1628), "`I < 0` on 70 of 100 draws", "q0.i.negative_draws"),
    RecordRow(_BRIEF_27, (1628, 1628), "The 31.5 % reduction", "q0.i.reduction"),
    RecordRow(_BRIEF_27, (1668, 1668), "31.5 % of it is attributable", "q0.i.attributable"),
    RecordRow(_PLAN, (513, 513), "measured: +39.5649, CI [+36.0510, +43.0787]", "q0.d1.plan"),
    RecordRow(_PLAN, (606, 607),
              "`dt_spatial` = `dt_nomix` = 207.11922169043655 and `bc` = `bc_top10_perix` = 207.1886372764493",
              "fixedtime.tied_levels"),
    RecordRow(_PLAN, (611, 612), "`maxpressure` 2..2 · `fixedtime` 2..4 · `random` 3..3", "q2b_hard.ranges"),
    RecordRow(_PLAN, (612, 612), "(max 9 < 12)", "q2b.fixedtime_max"),
    RecordRow(_PLAN, (613, 613), "(rank ≥ 2 always)", "q3a.rank_floor"),
    RecordRow(_PLAN, (614, 614), "the true range is 7..9", "q2b_hard.total_range"),
    RecordRow(_PLAN, (615, 615), "7 of 18", "q2b_hard.total"),
)

#: The two declared exceptions, each verified by the build (``BRIEF_45`` section 1.4): the recorded integer is reachable
#: by some tie-break order, and the declared rule gives another.
RECORD_EXCEPTIONS: Mapping[str, str] = {
    "deferred_108": (
        "DEFERRED 108: at fixedtime dt_spatial = dt_nomix and bc = bc_top10_perix exactly, and under the tie rule P5.2 "
        "declared on 2026-08-24 (a tie is discordant for both orders) Q2b's count there is 7, the lower end of its "
        "range 7..9; the record's 8 is one tie-break order's count, not the declared rule's"),
    "q3a_fixedtime": (
        "BRIEF_45 section 1.4(b): at fixedtime dt_nomix ties dt_spatial exactly, so its rank is 2 counting only the "
        "arms strictly lower (the declared rule, P5.2b's) and 3 if the tie is broken against it; the record's 3 is "
        "one tie-break order's rank, the range is 2..3"),
}

#: The three records of Q5's verdict (Amendment A, A3), each with the coordinator's statement, verbatim.
Q5_RECORDS: tuple[Quote, ...] = (
    Quote(_PACKET, (30, 30), "CONFIRMED twice, with a predicted null on `fixedtime`"),
    Quote(_PACKET, (572, 573),
          "The load sorter is confirmed across tiers, and `bc_top10_perix` is the repair — with a *predicted* null on "
          "`fixedtime` that the mechanism explains."),
    Quote(_BRIEF_27, (1274, 1274), "THE LOAD-SORTER MECHANISM IS CONFIRMED, decisively and twice"),
)
Q5_STATEMENT = (
    "the record scored Q5b on two of its four registered tiers, scored neither Q5a nor Q5c, and called predicted a null "
    "the registration did not predict.")

#: ``BRIEF_45`` section 1.3: the statements never scored in any committed record, under either definition.
NEVER_SCORED: tuple[tuple[str, str | None], ...] = (
    ("Q0a", "mappo1000"), ("Q0b", "mappo1000"), ("Q5a", None), ("Q5c", None), ("Q3c", "maxpressure"),
    ("Q3c", "fixedtime"), ("Q6", None),
)

#: The README sentence the collapse audit backs, quoted as text without a line number (Amendment A, A4.2).
README_COLLAPSE_SENTENCE = (
    "The DT arms are the only arms that never collapse below the random-policy anchor (P5.2).")

WHAT_THIS_DOES_NOT_SAY: tuple[str, ...] = (
    "No new prediction: Q1's, Q1b's and Q2's predictions exist on att_horizon (att_ours) only; under att_engine they are "
    "compared with att_engine measurements and labelled exactly so, and rule R' is not recomputed from att_engine "
    "inputs (BRIEF_45 section 2).",
    "No episode was rolled, no checkpoint loaded, no model trained or evaluated: every number is a recomputation of "
    "committed records read at their digests.",
    "P5.2's statements stand as written in its packet; this artifact scores them by the registered rules beside them and "
    "names every difference in comparison_with_the_record.",
    "Out of scope: P4.x, P5.1 and P5.3 statements (P8.4b's V1-V7 hold their registered verdicts), the E1 and I1/J1 "
    "envelopes (replicates carry no att_engine; att_rederivation.is_replicate_artifact), hz1x1 and cologne3.",
    "No level prediction exists for the two 4-head cells and no equivalence verdict is made anywhere (D8).",
    "D15's clause on arms whose distribution is visibly not unimodal is a visual judgement this artifact does not make: "
    "it reports each cell's median and IQR and no distributional comparison.",
    "The rank-biserial and the Wilcoxon p-value are reported beside every two-arm contrast (D8) and are never read by a "
    "verdict; I, a difference of two differences, carries none.",
    "Q3c registers a value per tier and no outcome rule: it is reported with its reading and no verdict.",
    "The leaders block and the descriptive collapse audit are not registered statements.",
    "P8.4b's att_engine values are pinned by its campaign's completeness, the per-row checks, the README note's means on "
    "every tier and provenance.p8_4b_rows.sha256 -- not by a manifest P8.4b wrote, which does not exist.",
    "Rule R makes att_engine the primary metric on grid4x4; the att_ours verdict is reported beside every one and "
    "definition_dependent marks where they differ.",
)


def record_values(statements: Sequence[Mapping[str, Any]], audit: Mapping[str, Any],
                  ties: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Every :data:`RECORD` value key, formatted at the record's precision from the ``as_committed`` ``att_ours``
    statements (the basis P5.2 scored on)."""
    raise NotImplementedError("P5.2c C1 skeleton: record_values")


def comparison_with_the_record(statements: Sequence[Mapping[str, Any]], audit: Mapping[str, Any],
                               ties: Sequence[Mapping[str, Any]], inputs: Inputs, *,
                               record: Sequence[RecordRow] = RECORD) -> dict[str, Any]:
    """Every recorded value beside this artifact's, equal or explained; refused if a row is neither, or if a declared
    exception does not verify.  Also: P5.2b's statements field by field, P8.4b's V4 and V5, P5.1's P1 and P3, P5.2's
    stop-rule record (each recomputed by its own route where the routes differ), Q5's three records (A3) and the
    never-scored list."""
    raise NotImplementedError("P5.2c C1 skeleton: comparison_with_the_record")


def changes_block(statements: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per statement and tier: as committed ``att_ours`` -> corrected ``att_ours`` -> corrected ``att_engine``."""
    raise NotImplementedError("P5.2c C1 skeleton: changes_block")


# ----------------------------------------------------------------------
# The artifact, the barrier and the command line
# ----------------------------------------------------------------------


def build_artifact(inputs: Inputs, *, git: Mapping[str, Any], code: Sequence[Mapping[str, Any]],
                   record: Sequence[RecordRow] = RECORD) -> dict[str, Any]:
    """The ``p5.2-statements/1.0`` artifact from verified inputs; pure."""
    raise NotImplementedError("P5.2c C1 skeleton: build_artifact")


def serialize(artifact: Mapping[str, Any]) -> bytes:
    """The canonical bytes: sorted keys, two-space indent, no NaN, one trailing newline."""
    raise NotImplementedError("P5.2c C1 skeleton: serialize")


def assert_out_path(path: str | Path, roots: Roots, inputs: Inputs) -> Path:
    """The barrier (``PROJECT_PLAN`` section 7): refuse an out path under ``--output-root``, under any tree holding
    ``SHA256SUMS_p5_2.txt``, equal to an input, not ``.json``, whose parent does not exist, or that already exists."""
    raise NotImplementedError("P5.2c C1 skeleton: assert_out_path")


def write_artifact(path: Path, data: bytes) -> Path:
    """Write *data* once: a dot-temporary in the same directory, then ``os.link`` onto *path* (refusing an existing
    file); the temporary is removed whatever happens."""
    raise NotImplementedError("P5.2c C1 skeleton: write_artifact")


def build_parser() -> argparse.ArgumentParser:
    """``build`` and ``check``, each with ``--repo-root`` and ``--output-root`` (required) and ``--out``."""
    raise NotImplementedError("P5.2c C1 skeleton: build_parser")


def main(argv: Sequence[str] | None = None) -> int:
    """Exit 0 accepted; 2 refused, with ``REFUSED: <type>: <reason>`` on stderr."""
    raise NotImplementedError("P5.2c C1 skeleton: main")


if __name__ == "__main__":
    sys.exit(main())
