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
    """Amendment A, A4.1: refuse unless the code that RUNS -- this module's own file and, when it runs as a script,
    ``__main__``'s -- and every module of :data:`CODE_MODULES` and every loaded ``offline`` module resolves to
    ``<repo_root>/offline/...`` (Amendment B, B2.4); return each :data:`CODE_MODULES` entry's path relative to the root
    and its sha256, the entry for ``offline.tier_statements`` being the file that runs."""
    import importlib

    root = Path(repo_root).resolve()
    package = root / "offline"
    running = {"this module's own file": Path(__file__).resolve()}
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if __name__ == "__main__" and main_file:
        running["__main__"] = Path(main_file).resolve()
    for what, path in running.items():
        if package not in path.parents:
            raise ValueError(
                f"the running file ({what}) is {path}, which does not resolve under {package}: the artifact would "
                f"record {root}'s git commit for code that did not come from it (BRIEF_45 Amendment B, B2.4; "
                "Amendment A, A4.1)")

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
        path = running["this module's own file"] if name == "offline.tier_statements" else located(name)[0]
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
    for name in (ar.MANIFEST_NAME, ar.COMPLETE_MARKER):  # BRIEF_45 Amendment B, B2.14: listed with their sha256
        relative = f"{P8_4B_DIR}/{name}"
        files.append({"file": f"output/{relative}", "sha256": hashlib.sha256((output / relative).read_bytes()).hexdigest(),
                      "anchor": ("iql_correction.assert_p8_4b_campaign: the declared-cells digest recomputed from the "
                                 "manifest's own cell list equals CAMPAIGN_COMPLETE's and the pin")})
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
    ``mean_ci95``, the counts of draws where it is negative / zero / positive, and the per-seed block by the imported
    worker (``tier_sweep.per_seed_block`` -> ``spatial_mixing.per_seed_advantages``) over composite rows: per
    ``(seed, draw)``, ``left[0] - left[1]`` against ``right[0] - right[1]`` (``BRIEF_45`` Amendment B, B2.2), so the
    block carries P5.1's full schema.  By linearity its values are section 3.3's ``(mean_s(left[0]) - mean_s(left[1]))
    - (mean_s(right[0]) - mean_s(right[1]))``.  A difference of two differences, so no rank-biserial, and it says so."""
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
    by_key = [{(int(episode.seed), int(episode.draw_id)): episode.att_horizon for episode in eps} for eps in episodes]
    keys = sorted(by_key[0])
    left_rows = [{"seed": seed, "draw_id": draw, definition: by_key[0][(seed, draw)] - by_key[1][(seed, draw)]}
                 for seed, draw in keys]
    right_rows = [{"seed": seed, "draw_id": draw, definition: by_key[2][(seed, draw)] - by_key[3][(seed, draw)]}
                  for seed, draw in keys]
    worker = ts.per_seed_block(left_rows, right_rows, left_name="d4", right_name="d1", metric=definition)
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
        "per_seed": worker["per_seed"],
        "seed_reversal_qualifier": worker["seed_reversal_qualifier"],
        "per_seed_route": ("tier_sweep.per_seed_block over composite rows: per (seed, draw), left[0] - left[1] (d4's "
                           "pair) on the left and right[0] - right[1] (d1's pair) on the right, so each seed's value "
                           "is the mean of the first minus the mean of the second (BRIEF_45 Amendment B, B2.2)"),
        "rank_biserial": None,
        "rank_biserial_note": ("none: I is a difference of two differences, not a pair of arms, so D8's rank-biserial "
                               "does not apply (BRIEF_45 Amendment A, A2 Q8)"),
    }


# ----------------------------------------------------------------------
# Orderings and ties
# ----------------------------------------------------------------------


def _tier_per_seed_orderings(cells: Mapping[tuple[str, str], Sequence[EpisodeResult]], predicted: Sequence[str],
                             tier: str) -> dict[str, Any]:
    """``iql_correction._per_seed_orderings`` restated for any tier: each training seed's levels, order, first place and
    concordance, refused unless the six arms cover the same seeds.  A seed's episodes are taken in the cell's own
    (seed, draw) order."""
    covered = {method: sorted({int(episode.seed) for episode in cells[(method, tier)]}) for method in ts.METHODS}
    seeds = covered["iql"]
    if any(value != seeds for value in covered.values()):
        raise ValueError(f"the six arms at {tier} do not cover the same seeds {covered}: a per-seed ordering compares "
                         "like with like")
    out: dict[str, Any] = {}
    for seed in seeds:
        levels = {method: float(mean_ci95([episode.att_horizon for episode in cells[(method, tier)]
                                           if int(episode.seed) == seed]).mean) for method in ts.METHODS}
        order = sorted(ts.METHODS, key=lambda method: levels[method])
        lowest = min(levels.values())
        firsts = [method for method in ts.METHODS if levels[method] == lowest]
        concordance = ts.concordance(list(predicted), order, measured_levels=levels)
        out[str(seed)] = {"levels": levels, "order": order, "first": order[0],
                          "tied_for_first": firsts if len(firsts) > 1 else [], "firsts": firsts,
                          "n_concordant": concordance["n_concordant"], "n_tied": concordance["n_tied"]}
    return out


def tier_composer(cells: Mapping[tuple[str, str], Sequence[EpisodeResult]], *, tier: str, with_hard_subset: bool
                  ) -> dict[str, Any]:
    """``iql_correction.random_tier_statements`` restated for any out-of-sample tier (plan F2, Amendment A, A2 Q2): at
    ``tier="random"`` it returns every field that function returns, equal, and may add fields.  The restatement is
    line for line, with the function's ``TIER`` and ``METHOD`` constants made the parameter *tier* and ``"iql"``; the
    one key that carries the tier's name is the q1 entry's, ``iql_<tier>`` (``iql_random`` at the random tier)."""
    if tier not in OUT_OF_SAMPLE_TIERS:
        raise ValueError(f"{tier!r} is not an out-of-sample tier {OUT_OF_SAMPLE_TIERS}: Q2 and Q3 are scored there only")
    method_name = "iql"
    levels = ic.levels_of(cells)
    q1 = ts.score_level(levels)
    entry = next(cell for cell in q1["cells"] if cell["cell"] == [method_name, tier])
    measured = {method: levels[(method, tier)] for method in ts.METHODS}
    order = sorted(ts.METHODS, key=lambda method: measured[method])
    predicted = list(ts.predicted_order(tier))
    lowest = min(measured.values())
    firsts = [method for method in ts.METHODS if measured[method] == lowest]
    statements: dict[str, Any] = {
        "q1": {f"{method_name}_{tier}": entry, "n_held": q1["n_held"], "n_cells": q1["n_cells"],
               "threshold": q1["threshold"], "band": q1["band"], "outcome": q1["outcome"],
               "rule": "tier_sweep.score_level: |measured - predicted| / predicted <= band per cell; k of N"},
        "ranking": {"order": order, "levels": measured,
                    "rule": "the six methods by level, lowest first (P5.2's measured ordering)"},
        "q2a": {"predicted_first": predicted[0], "measured_first": order[0],
                "tied_for_first": firsts if len(firsts) > 1 else [],
                "outcome": "HELD" if firsts == [predicted[0]] else "FAILED",
                "rule": "docs/plans/p5.2.md section 4 Q2a: HELD iff the predicted first-place arm is the measured "
                        "one; an exact tie for first is no unique first place"},
    }
    concordance = ts.concordance(predicted, order, measured_levels=measured)
    discordant = {tuple(pair) for pair in concordance["discordant"]}
    statements["q2b"] = {
        **concordance, "threshold": ts.CONCORDANCE_THRESHOLD,
        "outcome": "HELD" if concordance["n_concordant"] >= ts.CONCORDANCE_THRESHOLD else "FAILED",
        "iql_pairs": [{"pair": list(pair), "concordant": tuple(pair) not in discordant}
                      for pair in itertools.combinations(sorted(ts.METHODS), 2) if method_name in pair],
        "rule": "tier_sweep.concordance and predicted_order, P5.2's own, with its declared tie rule",
    }
    if with_hard_subset:
        statements["q2b_hard"] = {**ts.concordance(predicted, order, subset=ts.HARD_SUBSET, measured_levels=measured),
                                  "rule": "the same, on tier_sweep.HARD_SUBSET; no threshold is registered"}
    per_seed = _tier_per_seed_orderings(cells, predicted, tier)
    reversing = [int(seed) for seed, block in per_seed.items() if block["firsts"] != [order[0]]]
    statements["per_seed"] = {
        "seeds": {seed: {key: value for key, value in block.items() if key != "firsts"} for seed, block in per_seed.items()},
        "rule": ("docs/plans/p5.2.md section 4 Q2 (D9): each training seed's ordering beside the pooled one; a seed's "
                 "level is the mean of its episodes by dt_gate.mean_ci95 in draw order, its order and first place as "
                 "the pooled ones, its concordance by tier_sweep.concordance with the declared tie rule"),
    }
    statements["q2a"]["reverses_on_seeds"] = reversing
    statements["q2a"]["reversal"] = None if not reversing else (
        f"the pooled first place, {order[0]}, reverses on seed{'s' if len(reversing) > 1 else ''} "
        + ", ".join(str(seed) for seed in reversing) + ": "
        + "; ".join(f"{' and '.join(per_seed[str(seed)]['firsts'])} "
                    f"{'tie for' if len(per_seed[str(seed)]['firsts']) > 1 else 'is'} first on seed {seed}"
                    for seed in reversing)
        + " (docs/plans/p5.2.md section 4 Q2, D9: a first place that reverses on a seed is reported as reversing)")
    # BRIEF_45 Amendment B, B2.3: beside P5.2b's two fields above (kept, so the random tier stays equal to P5.2b's field
    # by field), the first place under the tie rule -- null when the pooled first place is itself tied -- and the
    # seeds whose SET of first places differs from the pooled set.  With a unique pooled first the list is
    # reverses_on_seeds.
    statements["q2a"]["measured_first_under_rule"] = order[0] if len(firsts) == 1 else None
    statements["q2a"]["reverses_on_seeds_against_pooled_firsts"] = [
        int(seed) for seed, block in per_seed.items() if set(block["firsts"]) != set(firsts)]
    statements["q2b"]["per_seed_n_concordant"] = {seed: block["n_concordant"] for seed, block in per_seed.items()}
    others = [method for method in ts.METHODS if method != "dt_nomix"]
    statements["q3a"] = {
        "rank": 1 + sum(measured[method] < measured["dt_nomix"] for method in others), "of": len(ts.METHODS),
        "outcome": "HELD" if all(measured[method] > measured["dt_nomix"] for method in others) else "FAILED",
        "rule": "docs/plans/p5.2.md section 4 Q3a: dt_nomix has the lowest level of the six; rank counts the arms "
                "strictly lower",
    }
    best = min(NON_DT_METHODS, key=lambda method: measured[method])
    left = _per_draw_means(list(cells[("dt_nomix", tier)]))
    right = _per_draw_means(list(cells[(best, tier)]))
    if set(left) != set(right):
        raise ValueError(f"dt_nomix and {best} at {tier} do not cover the same draws")
    shared = sorted(left)
    stats = mean_ci95([left[draw] - right[draw] for draw in shared])
    low, high = stats.mean - stats.ci95, stats.mean + stats.ci95
    statements["q3c"] = {
        "best_non_dt": best, "tied_best_non_dt": [m for m in NON_DT_METHODS if measured[m] == measured[best]][1:],
        "mean": float(stats.mean), "std": float(stats.std), "ci95_low": float(low), "ci95_high": float(high),
        "ci95_width": float(high - low), "n_draws": len(shared),
        "reading": "resolves against the DT" if low > 0 else ("resolves for the DT" if high < 0 else "NOT RESOLVED"),
        "rule": "docs/plans/p5.2.md section 4 Q3c and its common scoring: mean[ATT(dt_nomix) - ATT(best non-DT arm)] "
                "over per-draw means of the five seeds (dt_gate._per_draw_means), the 95 % CI by dt_gate.mean_ci95",
    }
    return statements


def level_groups(levels: Mapping[str, float]) -> list[list[str]]:
    """Arms grouped by EXACTLY equal level, groups in ascending level, arms within a group in the input's order."""
    return [[arm for arm in levels if levels[arm] == value] for value in sorted(set(levels.values()))]


def tie_audit(levels: Mapping[str, float], predicted: Sequence[str], *, subject: str = "dt_nomix") -> dict[str, Any]:
    """Every tie-dependent integer of one ordering: Q2b's and Q2b-hard's concordance under the declared tie rule and its
    range over every order consistent with *levels*, the possible first places, and *subject*'s rank (1 + the arms
    strictly lower) with its range; with the number of orders enumerated."""
    groups = level_groups(levels)
    orders = [[arm for group in arrangement for arm in group]
              for arrangement in itertools.product(*(itertools.permutations(group) for group in groups))]
    measured = [arm for group in groups for arm in group]
    counts = [ts.concordance(predicted, order)["n_concordant"] for order in orders]
    hard = [ts.concordance(predicted, order, subset=ts.HARD_SUBSET)["n_concordant"] for order in orders]
    ranks = [order.index(subject) + 1 for order in orders]
    return {
        "groups": groups, "tied_groups": [group for group in groups if len(group) > 1],
        "orders_enumerated": len(orders),
        "q2b": {"under_rule": ts.concordance(predicted, measured, measured_levels=levels)["n_concordant"],
                "range": [min(counts), max(counts)]},
        "q2b_hard": {"under_rule": ts.concordance(predicted, measured, subset=ts.HARD_SUBSET,
                                                  measured_levels=levels)["n_concordant"],
                     "range": [min(hard), max(hard)]},
        "first_place": {"under_rule": groups[0][0] if len(groups[0]) == 1 else None,
                        "possible": sorted({order[0] for order in orders})},
        "rank": {"under_rule": 1 + sum(1 for arm in levels if arm != subject and levels[arm] < levels[subject]),
                 "range": [min(ranks), max(ranks)],
                 "tied_with": [arm for arm in levels if arm != subject and levels[arm] == levels[subject]]},
        "rule": ("an order is consistent with the levels iff it never places a higher level ahead of a lower one; "
                 "every such order is enumerated (the product of the permutations within each group of exactly equal "
                 "levels), and each integer's range is its minimum and maximum over them; under_rule is the declared "
                 "rule's value (a tie is discordant for both orders; rank counts the arms strictly lower)"),
    }


# ----------------------------------------------------------------------
# The statements
# ----------------------------------------------------------------------


def _quotes(*keys: str) -> list[dict[str, Any]]:
    """The registered text a statement implements, as quoted in :data:`RULES`."""
    return [{"rule": key, "file": quote.file, "lines": list(quote.lines), "text": quote.text}
            for key in keys for quote in RULES[key]]


def _entry(statement_id: str, tier: str | None, *, definition: str, kind: str, numbers: Mapping[str, Any],
           inputs: Sequence[Cell], rules: Sequence[str], verdict: str | None = None, reading: str | None = None,
           arm: str | None = None) -> dict[str, Any]:
    """One statement entry.  ``version`` and ``definition_dependent`` are set by :func:`all_statements`."""
    ids: list[str] = []
    for cell in inputs:
        if cell.spec.cell_id not in ids:
            ids.append(cell.spec.cell_id)
    return {"id": statement_id, "tier": tier, "arm": arm, "definition": definition,
            "primary": definition == PRIMARY_DEFINITION, "kind": kind, "verdict": verdict, "reading": reading,
            "numbers": dict(numbers), "inputs": ids, "rule": _quotes(*rules)}


def _registered_on(definition: str) -> str:
    """``BRIEF_45`` section 2 and ``BRIEF_44`` Amendment B, B1.7(d): what the predictions were registered on."""
    if definition == "att_ours":
        return ("att_horizon (att_ours): P5.2 registered these predictions on the definition measured here "
                "(docs/plans/p5.2.md:407-412)")
    return ("att_horizon (att_ours), not att_engine: these are P5.2's registered att_horizon-era predictions, compared "
            "here with att_engine measurements; no prediction was registered under att_engine (BRIEF_45 section 2; "
            "BRIEF_44 Amendment B, B1.7(d))")


def _level(cell: Cell, definition: str) -> float:
    """A cell's level: ``mean_ci95`` over its episodes in ``(seed, draw)`` order (``iql_correction.levels_of``)."""
    key = (cell.spec.method, cell.spec.tier)
    return ic.levels_of({key: episodes_of(cell, definition)})[key]


def _seed_levels(book: Mapping[tuple[str, str], Cell], tier: str, definition: str, methods: Sequence[str]
                 ) -> dict[str, dict[str, float]]:
    """Per training seed, each method's level on *tier*: ``mean_ci95`` over the seed's episodes in draw order."""
    out: dict[str, dict[str, float]] = {}
    for method in methods:
        episodes = episodes_of(book[(method, tier)], definition)
        for seed in sorted({int(episode.seed) for episode in episodes}):
            out.setdefault(str(seed), {})[method] = float(
                mean_ci95([episode.att_horizon for episode in episodes if int(episode.seed) == seed]).mean)
    return out


def _q3c_reading(low: float, high: float) -> str:
    """P5.2b's reading of Q3c (``iql_correction.random_tier_statements``)."""
    return "resolves against the DT" if low > 0 else ("resolves for the DT" if high < 0 else "NOT RESOLVED")


#: The values the registration states per tier, quoted (``docs/plans/p5.2.md:651-689``).
REGISTERED_VALUES: Mapping[str, Mapping[str, float]] = {
    "Q3b": {"maxpressure": -0.9655, "fixedtime": -1.9906, "random": -10.1702},
    "Q3c": {"maxpressure": 0.4326, "fixedtime": -17.1639, "random": -21.9354},
    "Q4": {"mappo1000": 39.5649, "maxpressure": 41.7402, "fixedtime": 51.3689, "random": 62.7106},
    "Q5a": {"maxpressure": 575.34, "fixedtime": 820.65, "random": 970.57},
    "Q5b": {"mappo1000": -583.81, "maxpressure": -584.97, "fixedtime": -812.15, "random": -967.88},
    "Q5c": {"mappo1000": -3.21, "maxpressure": -9.63, "fixedtime": 8.49, "random": 2.69},
}


def score_q0(book: Mapping[tuple[str, str], Cell], *, definition: str) -> list[dict[str, Any]]:
    """Q0's ``d1`` (context), Q0a, Q0b and the stop rule at ``mappo1000``."""
    tier = "mappo1000"
    s1, n1 = book[("dt_spatial", tier)], book[("dt_nomix", tier)]
    s4, n4 = book[("dt_spatial_h4", tier)], book[("dt_nomix_h4", tier)]
    d1 = contrast(s1, n1, definition=definition, tier=tier)
    d4 = contrast(s4, n4, definition=definition, tier=tier)
    interaction = composite_contrast((s4, n4), (s1, n1), definition=definition, tier=tier)
    q0a = interval_outcome(d4["ci95_low"], d4["ci95_high"], predicted="positive")
    if interaction["exactly_zero"]:
        q0b, q0b_reading = "HELD", "exactly zero (identical episodes): the CI [0, 0] includes zero, so the registered " \
                                   "inclusion rule gives HELD (docs/plans/p5.2.md:523-524); the exact zero is labelled " \
                                   "and is evidence for no sign (BRIEF_45 Amendment B, B2.1)"
    elif interaction["ci95_low"] <= 0.0 <= interaction["ci95_high"]:
        q0b, q0b_reading = "HELD", "the CI of mean(I) includes zero"
    else:
        q0b, q0b_reading = "FAILED", f"the CI of mean(I) excludes zero; its sign is {interaction['sign']}"
    stop = ts.stop_rule_verdict(d4["ci95_low"], d4["ci95_high"])
    d1_numbers = {**d1, "role": "d1, the 1-head contrast: P5.1's pair, reused (D17 verified on disk)",
                  "registered_value": REGISTERED_VALUES["Q4"]["mappo1000"]}
    return [
        _entry("Q0.d1", tier, definition=definition, kind="quantity", numbers=d1_numbers, inputs=(s1, n1),
               rules=("Q0", "common"), reading=d1["reading"]),
        _entry("Q0a", tier, definition=definition, kind="verdict", verdict=q0a, reading=d4["reading"],
               numbers={**d4, "role": "d4, the 4-head contrast", "predicted_sign": "positive"}, inputs=(s4, n4),
               rules=("Q0", "Q0a", "common")),
        _entry("Q0b", tier, definition=definition, kind="verdict", verdict=q0b, reading=q0b_reading,
               numbers={**interaction, "role": "I = d4 - d1, per shared draw"}, inputs=(s4, n4, s1, n1),
               rules=("Q0", "Q0b", "common")),
        _entry("Q0.stop_rule", tier, definition=definition, kind="verdict", verdict=stop,
               reading=("STOP: the CI of mean(d4) lies entirely below zero" if stop == "STOP" else
                        "CONTINUE: the CI of mean(d4) does not lie entirely below zero"),
               numbers={"quantity": "d4", "mean_difference": d4["mean_difference"], "ci95_low": d4["ci95_low"],
                        "ci95_high": d4["ci95_high"], "ci95_width": d4["ci95_width"],
                        "verdict_function": "tier_sweep.stop_rule_verdict"},
               inputs=(s4, n4), rules=("Q0.stop_rule",)),
    ]


def score_q1(book: Mapping[tuple[str, str], Cell], *, definition: str) -> list[dict[str, Any]]:
    """Q1 (19 cells, ``tier_sweep.score_level``), Q1b per tier and Q1b's two-cell aggregate."""
    cells = [book[cell] for cell in ts.OUT_OF_SAMPLE_CELLS]
    levels = {cell: _level(book[cell], definition) for cell in ts.OUT_OF_SAMPLE_CELLS}
    q1 = {**ts.score_level(levels), "predictions_registered_on": _registered_on(definition)}
    rules_q1 = ("Q1",) if definition == "att_ours" else ("Q1", "att_engine_label")
    entries = [_entry("Q1", None, definition=definition, kind="verdict", verdict=q1["outcome"], numbers=q1,
                      inputs=cells, rules=rules_q1, reading=f"{q1['n_held']} of {q1['n_cells']} cells hold")]
    held = {}
    for tier in ("maxpressure", "fixedtime"):
        cell = ("behaviour", tier)
        scored = ts.score_level({cell: _level(book[cell], definition)}, cells=(cell,), band=ts.ANCHOR_BAND, threshold=1)
        per_cell = scored["cells"][0]
        held[tier] = "HELD" if per_cell["held"] else "FAILED"
        entries.append(_entry("Q1b", tier, definition=definition, kind="verdict", verdict=held[tier],
                              numbers={**per_cell, "band": scored["band"],
                                       "predictions_registered_on": _registered_on(definition)},
                              inputs=(book[cell],), rules=("Q1b",) if definition == "att_ours" else
                              ("Q1b", "att_engine_label")))
    both = "HELD" if all(value == "HELD" for value in held.values()) else "FAILED"
    entries.append(_entry(
        "Q1b.both", None, definition=definition, kind="verdict", verdict=both,
        reading="both cells hold" if both == "HELD" else "not both cells hold",
        numbers={"tiers": held, "label": "both cells hold (BRIEF_45 Amendment A, A2 Q7)",
                 "sources": {"maxpressure": "the registered bullet's tier (docs/plans/p5.2.md:576-578)",
                             "fixedtime": "the level table's second (Q1b) marking (docs/plans/p5.2.md:497)"},
                 "rule": "HELD iff both cells hold, each within tier_sweep.ANCHOR_BAND of its predicted level"},
        inputs=[book[("behaviour", tier)] for tier in held], rules=("Q1b",)))
    return entries


def score_q2_q3(book: Mapping[tuple[str, str], Cell], *, definition: str) -> list[dict[str, Any]]:
    """Per out-of-sample tier: the ranking, Q2a, Q2b, Q2b-hard, D9, Q3a and Q3c (from :func:`tier_composer`, with the tie
    audit added); Q2b's aggregate; and ``mappo1000``'s Q3a and Q3c as context."""
    oos = {cell: episodes_of(book[cell], definition) for cell in ts.OUT_OF_SAMPLE_CELLS}
    entries: list[dict[str, Any]] = []
    q2b_outcomes: dict[str, str] = {}
    q2b_counts: dict[str, int] = {}
    q2b_ranges: dict[str, list[int]] = {}
    aggregate_inputs: list[Cell] = []
    for tier in OUT_OF_SAMPLE_TIERS:
        composed = tier_composer(oos, tier=tier, with_hard_subset=True)
        levels = composed["ranking"]["levels"]
        predicted = list(ts.predicted_order(tier))
        audit = tie_audit(levels, predicted)
        six = [book[(method, tier)] for method in ts.METHODS]
        aggregate_inputs.extend(six)
        seed_audits = {seed: tie_audit(block["levels"], predicted) for seed, block in composed["per_seed"]["seeds"].items()}
        entries.append(_entry("ranking", tier, definition=definition, kind="context",
                              reading=" < ".join(" = ".join(group) for group in audit["groups"]),
                              numbers={**composed["ranking"], "groups": audit["groups"]}, inputs=six,
                              rules=("Q2a", "tie_rule")))
        entries.append(_entry("Q2a", tier, definition=definition, kind="verdict", verdict=composed["q2a"]["outcome"],
                              reading=(f"measured first {composed['q2a']['measured_first']}" if not
                                       composed["q2a"]["tied_for_first"] else
                                       "no unique first place: " + " = ".join(composed["q2a"]["tied_for_first"])),
                              numbers={**composed["q2a"], "first_place_under_rule": audit["first_place"]["under_rule"],
                                       "possible_firsts": audit["first_place"]["possible"],
                                       "orders_enumerated": audit["orders_enumerated"]},
                              inputs=six, rules=("Q2a", "tie_rule", "D9")))
        entries.append(_entry("Q2b", tier, definition=definition, kind="verdict", verdict=composed["q2b"]["outcome"],
                              reading=f"{composed['q2b']['n_concordant']} of 15 (range {audit['q2b']['range'][0]}.."
                                      f"{audit['q2b']['range'][1]})",
                              numbers={**composed["q2b"], "range": audit["q2b"]["range"],
                                       "orders_enumerated": audit["orders_enumerated"],
                                       "per_seed_range": {seed: a["q2b"]["range"] for seed, a in seed_audits.items()}},
                              inputs=six, rules=("Q2b", "tie_rule", "D9")))
        hard = composed["q2b_hard"]
        entries.append(_entry("Q2b_hard", tier, definition=definition, kind="count",
                              reading=f"{hard['n_concordant']} of 6 (range {audit['q2b_hard']['range'][0]}.."
                                      f"{audit['q2b_hard']['range'][1]}); no threshold is registered",
                              numbers={**hard, "range": audit["q2b_hard"]["range"],
                                       "orders_enumerated": audit["orders_enumerated"],
                                       "per_seed": {seed: a["q2b_hard"]["under_rule"] for seed, a in seed_audits.items()},
                                       "per_seed_range": {seed: a["q2b_hard"]["range"] for seed, a in
                                                          seed_audits.items()}},
                              inputs=[book[(method, tier)] for method in ts.HARD_SUBSET],
                              rules=("Q2b_hard", "tie_rule")))
        seeds_block = {}
        for seed, block in composed["per_seed"]["seeds"].items():
            a = seed_audits[seed]
            seeds_block[seed] = {**block, "n_concordant_range": a["q2b"]["range"],
                                 "hard_n_concordant": a["q2b_hard"]["under_rule"],
                                 "hard_n_concordant_range": a["q2b_hard"]["range"],
                                 "first_place_possible": a["first_place"]["possible"],
                                 "dt_nomix_rank": a["rank"]["under_rule"], "dt_nomix_rank_range": a["rank"]["range"],
                                 "orders_enumerated": a["orders_enumerated"]}
        entries.append(_entry("D9", tier, definition=definition, kind="context",
                              reading="; ".join(f"seed {seed}: " + (" = ".join(block["tied_for_first"])
                                                                    if block["tied_for_first"] else block["first"])
                                                + " first" for seed, block in seeds_block.items()),
                              numbers={**composed["per_seed"], "seeds": seeds_block}, inputs=six, rules=("D9",)))
        entries.append(_entry("Q3a", tier, definition=definition, kind="verdict", verdict=composed["q3a"]["outcome"],
                              reading=f"rank {composed['q3a']['rank']} of 6 (range {audit['rank']['range'][0]}.."
                                      f"{audit['rank']['range'][1]})",
                              numbers={**composed["q3a"], "tied_with": audit["rank"]["tied_with"],
                                       "rank_range": audit["rank"]["range"], "levels": levels,
                                       "per_seed": {seed: {"rank": a["rank"]["under_rule"], "range": a["rank"]["range"]}
                                                    for seed, a in seed_audits.items()}},
                              inputs=six, rules=("Q3", "Q3a", "tie_rule")))
        q3c = composed["q3c"]
        best = q3c["best_non_dt"]
        block = contrast(book[("dt_nomix", tier)], book[(best, tier)], definition=definition, tier=tier)
        if (block["mean_difference"], block["ci95_low"], block["ci95_high"]) != (q3c["mean"], q3c["ci95_low"],
                                                                                 q3c["ci95_high"]):
            raise ValueError(f"Q3c at {tier}: the composer's interval and the contrast's differ; one route only")
        entries.append(_entry("Q3c", tier, definition=definition, kind="reading", reading=q3c["reading"],
                              numbers={**q3c, "registered_value": REGISTERED_VALUES["Q3c"][tier], "contrast": block,
                                       "no_verdict": ("Q3c registers a value per tier and no outcome rule "
                                                      "(docs/plans/p5.2.md:655-659): it is reported with P5.2b's "
                                                      "reading and no verdict")},
                              inputs=[book[("dt_nomix", tier)], *(book[(m, tier)] for m in NON_DT_METHODS)],
                              rules=("Q3", "Q3c", "common")))
        q2b_outcomes[tier] = composed["q2b"]["outcome"]
        q2b_counts[tier] = composed["q2b"]["n_concordant"]
        q2b_ranges[tier] = audit["q2b"]["range"]
    entries.append(_entry(
        "Q2b.aggregate", None, definition=definition, kind="verdict", verdict=conjunction(list(q2b_outcomes.values())),
        numbers={"tiers": q2b_outcomes, "threshold": ts.CONCORDANCE_THRESHOLD,
                 "rule": "HELD iff >= 12 of 15 on each out-of-sample tier (BRIEF_45 section 3.3's conjunction)",
                 "overall": {"n_concordant": sum(q2b_counts.values()), "of": 15 * len(q2b_counts),
                             "range": [sum(r[0] for r in q2b_ranges.values()), sum(r[1] for r in q2b_ranges.values())],
                             "scored": False,
                             "why_not_scored": ("the registered parenthetical (>= 24 of 30 overall) was written for two "
                                                "out-of-sample tiers; against three it is stale and is not scored "
                                                "(BRIEF_27 GATE 2, G2)")}},
        inputs=aggregate_inputs, rules=("Q2b",)))
    tier = "mappo1000"
    six = [book[(method, tier)] for method in ts.METHODS]
    measured = {method: _level(book[(method, tier)], definition) for method in ts.METHODS}
    audit = tie_audit(measured, list(ts.predicted_order(tier)))
    rank = audit["rank"]["under_rule"]
    context = "mappo1000 is the seen cell: reported as context, never scored (docs/plans/p5.2.md:645-646)"
    entries.append(_entry("Q3a", tier, definition=definition, kind="context",
                          reading=f"rank {rank} of 6 (range {audit['rank']['range'][0]}..{audit['rank']['range'][1]}); "
                                  "context, not scored",
                          numbers={"rank": rank, "of": len(ts.METHODS), "tied_with": audit["rank"]["tied_with"],
                                   "rank_range": audit["rank"]["range"], "levels": measured, "groups": audit["groups"],
                                   "context": context},
                          inputs=six, rules=("Q3", "Q3a")))
    best = min(NON_DT_METHODS, key=lambda method: measured[method])
    block = contrast(book[("dt_nomix", tier)], book[(best, tier)], definition=definition, tier=tier)
    entries.append(_entry("Q3c", tier, definition=definition, kind="context",
                          reading=_q3c_reading(block["ci95_low"], block["ci95_high"]),
                          numbers={"best_non_dt": best,
                                   "tied_best_non_dt": [m for m in NON_DT_METHODS if measured[m] == measured[best]][1:],
                                   "contrast": block, "context": context},
                          inputs=[book[("dt_nomix", tier)], *(book[(m, tier)] for m in NON_DT_METHODS)],
                          rules=("Q3", "Q3c")))
    return entries


def score_q3b(book: Mapping[tuple[str, str], Cell], *, definition: str) -> list[dict[str, Any]]:
    """Q3b per out-of-sample tier (interval rule, predicted negative; the sub-prediction matched or not), its aggregate,
    and ``mappo1000`` as context."""
    predicted_to_resolve = {"maxpressure": False, "fixedtime": False, "random": True}
    entries = []
    outcomes = {}
    for tier in OUT_OF_SAMPLE_TIERS:
        left, right = book[("dt_nomix", tier)], book[("behaviour", tier)]
        block = contrast(left, right, definition=definition, tier=tier)
        outcome = interval_outcome(block["ci95_low"], block["ci95_high"], predicted="negative")
        resolved = block["ci95_low"] > 0.0 or block["ci95_high"] < 0.0
        matched = resolved == predicted_to_resolve[tier]
        sub_reading = (f"{'matched' if matched else 'not matched'}: predicted "
                       f"{'to resolve' if predicted_to_resolve[tier] else 'not to resolve'}, and the CI "
                       f"{'resolves' if resolved else 'does not resolve'}")
        if block["exactly_zero"]:
            source = ("identical episodes" if block["discriminability"]["identical"] else
                      "an exact zero from episodes that are not identical")
            sub_reading += (f"; the difference is {block['reading']}, so the {'match' if matched else 'miss'} comes "
                            f"from {source}, not from a measured difference that failed to resolve (BRIEF_45 "
                            "Amendment B, B2.1)")
        numbers = {**block, "predicted_sign": "negative", "registered_value": REGISTERED_VALUES["Q3b"][tier],
                   "sub_prediction": {"predicted_to_resolve": predicted_to_resolve[tier], "resolved": resolved,
                                      "matched": matched, "exactly_zero": block["exactly_zero"],
                                      "reading": sub_reading},
                   "sign_reading": f"the point estimate is {block['sign']}"}
        if tier == "random":
            numbers["behaviour_cell"] = "the shared random anchor, P5.1's cell, which is the random tier's behaviour " \
                                        "policy (D12)"
        entries.append(_entry("Q3b", tier, definition=definition, kind="verdict", verdict=outcome,
                              reading=block["reading"], numbers=numbers, inputs=(left, right),
                              rules=("Q3b", "common")))
        outcomes[tier] = outcome
    entries.append(_entry("Q3b.aggregate", None, definition=definition, kind="verdict",
                          verdict=conjunction(list(outcomes.values())),
                          numbers={"tiers": outcomes, "rule": "sign negative on all three, by the interval rule per "
                                                              "tier and BRIEF_45 section 3.3's conjunction"},
                          inputs=[c for t in OUT_OF_SAMPLE_TIERS for c in (book[("dt_nomix", t)],
                                                                          book[("behaviour", t)])],
                          rules=("Q3b",)))
    left, right = book[("dt_nomix", "mappo1000")], book[("behaviour", "mappo1000")]
    block = contrast(left, right, definition=definition, tier="mappo1000")
    would_be = interval_outcome(block["ci95_low"], block["ci95_high"], predicted="negative")
    entries.append(_entry("Q3b", "mappo1000", definition=definition, kind="context",
                          reading=f"{would_be} by the interval rule; context, not scored",
                          numbers={**block, "context": "mappo1000 is the seen cell: reported as context, never scored "
                                                       "(docs/plans/p5.2.md:645-646)"},
                          inputs=(left, right), rules=("Q3", "Q3b")))
    return entries


def _chain(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [row["mean_difference"] for row in rows]
    increasing = all(a < b for a, b in zip(values, values[1:]))
    decreasing = all(a > b for a, b in zip(values, values[1:]))
    return {"chain": list(rows), "strictly_increasing": increasing, "strictly_decreasing": decreasing,
            "verdict": "HELD" if increasing else "FAILED",
            "reading": ("d increases strictly along the chain" if increasing else
                        "inverted: d decreases strictly along the chain" if decreasing else
                        "d is not monotone along the chain")}


def score_q4(book: Mapping[tuple[str, str], Cell], *, definition: str) -> list[dict[str, Any]]:
    """``d(t)`` on four tiers, Q4a per out-of-sample tier and its aggregate, Q4b on the four-rung chain and the bullet's
    three-rung form."""
    entries = []
    blocks = {}
    for tier in TIERS:
        left, right = book[("dt_spatial", tier)], book[("dt_nomix", tier)]
        blocks[tier] = contrast(left, right, definition=definition, tier=tier)
        entries.append(_entry("Q4.d", tier, definition=definition, kind="quantity", reading=blocks[tier]["reading"],
                              numbers={**blocks[tier], "registered_value": REGISTERED_VALUES["Q4"][tier]},
                              inputs=(left, right), rules=("Q4", "common")))
    outcomes = {}
    for tier in OUT_OF_SAMPLE_TIERS:
        outcomes[tier] = interval_outcome(blocks[tier]["ci95_low"], blocks[tier]["ci95_high"], predicted="positive")
        entries.append(_entry("Q4a", tier, definition=definition, kind="verdict", verdict=outcomes[tier],
                              reading=blocks[tier]["reading"], numbers={**blocks[tier], "predicted_sign": "positive"},
                              inputs=(book[("dt_spatial", tier)], book[("dt_nomix", tier)]), rules=("Q4", "Q4a")))
    entries.append(_entry("Q4a.aggregate", None, definition=definition, kind="verdict",
                          verdict=conjunction(list(outcomes.values())),
                          numbers={"tiers": outcomes, "rule": "d(t) > 0 with the CI excluding 0 on all three "
                                                              "out-of-sample tiers (BRIEF_45 section 3.3's conjunction)"},
                          inputs=[c for t in OUT_OF_SAMPLE_TIERS for c in (book[("dt_spatial", t)],
                                                                          book[("dt_nomix", t)])],
                          rules=("Q4a",)))
    for statement_id, tiers in (("Q4b", TIERS), ("Q4b.three_rung", ("mappo1000", "maxpressure", "random"))):
        rows = [{"tier": tier, "mean_difference": blocks[tier]["mean_difference"],
                 "ci95_low": blocks[tier]["ci95_low"], "ci95_high": blocks[tier]["ci95_high"],
                 "exactly_zero": blocks[tier]["exactly_zero"],
                 "per_seed_min": blocks[tier]["per_seed"]["min"], "per_seed_max": blocks[tier]["per_seed"]["max"],
                 "registered_value": REGISTERED_VALUES["Q4"][tier]} for tier in tiers]
        chain = _chain(rows)
        form = ("the four-rung chain d(mappo1000) < d(maxpressure) < d(fixedtime) < d(random) (BRIEF_45 section 3.2; "
                "docs/plans/p5.2.md:674-675)" if statement_id == "Q4b" else
                "the registered bullet's three-rung form d(mappo1000) < d(maxpressure) < d(random) "
                "(docs/plans/p5.2.md:670-672), reported beside the four-rung one")
        entries.append(_entry(statement_id, None, definition=definition, kind="verdict", verdict=chain["verdict"],
                              reading=chain["reading"],
                              numbers={key: value for key, value in chain.items() if key != "verdict"} | {
                                  "form": form, "scored_on": "the point estimates, strictly"},
                              inputs=[c for t in tiers for c in (book[("dt_spatial", t)], book[("dt_nomix", t)])],
                              rules=("Q4b",)))
    return entries


def score_q5(book: Mapping[tuple[str, str], Cell], *, definition: str) -> list[dict[str, Any]]:
    """Q5a (three tiers), Q5b and Q5c (four tiers), each per tier and aggregated by :func:`conjunction`."""
    patterns = (
        ("Q5a", ("bc_top10", "bc"), {tier: "positive" for tier in OUT_OF_SAMPLE_TIERS}),
        ("Q5b", ("bc_top10_perix", "bc_top10"), {tier: "negative" for tier in TIERS}),
        ("Q5c", ("bc_top10_perix", "bc"), {"mappo1000": "negative", "maxpressure": "negative",
                                           "fixedtime": "positive", "random": "positive"}),
    )
    entries = []
    for statement_id, (left_method, right_method), signs in patterns:
        outcomes = {}
        inputs: list[Cell] = []
        for tier, sign in signs.items():
            left, right = book[(left_method, tier)], book[(right_method, tier)]
            block = contrast(left, right, definition=definition, tier=tier)
            outcomes[tier] = interval_outcome(block["ci95_low"], block["ci95_high"], predicted=sign)
            predicted_from_levels = (ts.PREDICTED_LEVELS[left_method][tier] - ts.PREDICTED_LEVELS[right_method][tier])
            entries.append(_entry(statement_id, tier, definition=definition, kind="verdict", verdict=outcomes[tier],
                                  reading=block["reading"],
                                  numbers={**block, "predicted_sign": sign,
                                           "registered_value": REGISTERED_VALUES[statement_id][tier],
                                           "registered_value_from_predicted_levels": predicted_from_levels},
                                  inputs=(left, right), rules=(statement_id, "common")))
            inputs.extend((left, right))
        entries.append(_entry(f"{statement_id}.aggregate", None, definition=definition, kind="verdict",
                              verdict=conjunction(list(outcomes.values())),
                              numbers={"tiers": outcomes, "rule": f"{statement_id} on all its registered tiers, by the "
                                                                  "interval rule per tier and BRIEF_45 section 3.3's "
                                                                  "conjunction"},
                              inputs=inputs, rules=(statement_id,)))
    return entries


def score_q6(book: Mapping[tuple[str, str], Cell], *, definition: str) -> list[dict[str, Any]]:
    """Q6 for ``dt_spatial`` and ``dt_nomix`` on four tiers against the shared anchor; P5.1's three-valued outcome
    beside it from ``spatial_mixing.collapse_criterion``, refused unless that call's interval equals this one's."""
    anchor = book[ANCHOR]
    entries = []
    for tier in TIERS:
        for arm in DT_ARMS:
            cell = book[(arm, tier)]
            block = contrast(cell, anchor, definition=definition, tier=tier)
            p5_1 = collapse_criterion(episodes_of(cell, definition), episodes_of(anchor, definition))
            if (p5_1["mean_difference"], p5_1["ci95_low"], p5_1["ci95_high"]) != (
                    block["mean_difference"], block["ci95_low"], block["ci95_high"]):
                raise ValueError(f"spatial_mixing.collapse_criterion gives {p5_1['mean_difference']!r} "
                                 f"[{p5_1['ci95_low']!r}, {p5_1['ci95_high']!r}] for {cell.spec.cell_id} - anchor and "
                                 f"this module {block['mean_difference']!r} [{block['ci95_low']!r}, "
                                 f"{block['ci95_high']!r}]: one route, so they must be equal under == (BRIEF_45 "
                                 "Amendment A, A2 Q7)")
            verdict = "COLLAPSED" if block["ci95_low"] > 0.0 else "NOT COLLAPSED"
            entries.append(_entry("Q6", tier, arm=arm, definition=definition, kind="verdict", verdict=verdict,
                                  reading=block["reading"],
                                  numbers={**block, "p5_1_outcome": p5_1["outcome"], "seen": tier == "mappo1000",
                                           "registered_expectation": ("COLLAPSED" if (arm, tier) == ("dt_spatial",
                                                                                                    "random")
                                                                      else "NOT COLLAPSED"),
                                           "rule_two_valued": ("COLLAPSED iff the arm is worse than the anchor with "
                                                               "the paired CI excluding 0, else NOT COLLAPSED (the "
                                                               "primary vocabulary, BRIEF_45 section 3.3)"),
                                           "rule_three_valued": p5_1["rule"]},
                                  inputs=(cell, anchor), rules=("Q6", "Q6.expectation")))
    return entries


#: The order statements are listed in (``BRIEF_45`` section 3.4: statement, then tier, definition and version).
STATEMENT_ORDER: tuple[str, ...] = (
    "Q0.d1", "Q0a", "Q0b", "Q0.stop_rule", "Q1", "Q1b", "Q1b.both", "ranking", "Q2a", "Q2b", "Q2b.aggregate",
    "Q2b_hard", "D9", "Q3a", "Q3b", "Q3b.aggregate", "Q3c", "Q4.d", "Q4a", "Q4a.aggregate", "Q4b", "Q4b.three_rung",
    "Q5a", "Q5a.aggregate", "Q5b", "Q5b.aggregate", "Q5c", "Q5c.aggregate", "Q6",
)


def all_statements(books: Mapping[str, Mapping[tuple[str, str], Cell]]) -> list[dict[str, Any]]:
    """Every statement under both definitions; version-dependent ones in both versions, every other one computed in
    both, refused unless identical, and stored once with ``version: "both"``; ``definition_dependent`` set."""
    scorers = (score_q0, score_q1, score_q2_q3, score_q3b, score_q4, score_q5, score_q6)
    computed: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for version in VERSIONS:
        entries: dict[tuple[Any, ...], dict[str, Any]] = {}
        for definition in DEFINITIONS:
            for scorer in scorers:
                for entry in scorer(books[version], definition=definition):
                    key = (entry["id"], entry["tier"], entry["arm"], entry["definition"])
                    if key in entries:
                        raise ValueError(f"two statements share the key {key}")
                    entries[key] = entry
        computed[version] = entries
    if set(computed["corrected"]) != set(computed["as_committed"]):
        raise ValueError("the two versions do not score the same statements")
    out: list[dict[str, Any]] = []
    for key, corrected in computed["corrected"].items():
        original = computed["as_committed"][key]
        if CORRECTED_IQL in corrected["inputs"] or ORIGINAL_IQL in original["inputs"]:
            out.append({**corrected, "version": "corrected"})
            out.append({**original, "version": "as_committed"})
            continue
        if json.dumps(corrected, sort_keys=True) != json.dumps(original, sort_keys=True):
            raise ValueError(f"{key} differs between versions although it reads no random-tier IQL cell: the two "
                             "versions may differ only through that cell (BRIEF_45 section 3.3)")
        out.append({**corrected, "version": "both"})
    index = {(e["id"], e["tier"], e["arm"], e["definition"], e["version"]): e for e in out}
    for entry in out:
        other_definition = "att_engine" if entry["definition"] == "att_ours" else "att_ours"
        other = index[(entry["id"], entry["tier"], entry["arm"], other_definition, entry["version"])]
        if entry["kind"] == "verdict":
            entry["definition_dependent"] = entry["verdict"] != other["verdict"]
            entry["definition_dependent_on"] = "verdict"
        elif entry["reading"] is not None:
            entry["definition_dependent"] = entry["reading"] != other["reading"]
            entry["definition_dependent_on"] = "reading"
        else:
            entry["definition_dependent"] = None
            entry["definition_dependent_on"] = None
    tier_rank = {tier: i for i, tier in enumerate(TIERS)}
    return sorted(out, key=lambda e: (STATEMENT_ORDER.index(e["id"]), tier_rank.get(e["tier"], len(TIERS)),
                                      DT_ARMS.index(e["arm"]) if e["arm"] in DT_ARMS else -1,
                                      DEFINITIONS.index(e["definition"]),
                                      ("both", *VERSIONS).index(e["version"])))


def _versions_of(tier: str) -> tuple[str, ...]:
    """The versions a tier-level block is computed in: the random tier holds the IQL cell, every other tier one."""
    return VERSIONS if tier == "random" else ("both",)


def leaders_block(books: Mapping[str, Mapping[tuple[str, str], Cell]]) -> list[dict[str, Any]]:
    """Descriptive, not registered: per tier, definition and (at ``random``) version, the level order with ties, the
    leader(s), the runner(s)-up, leader - runner-up as a contrast, and the per-seed first places."""
    blocks = []
    for tier in TIERS:
        for definition in DEFINITIONS:
            for version in _versions_of(tier):
                book = books["corrected" if version == "both" else version]
                levels = {method: _level(book[(method, tier)], definition) for method in ts.METHODS}
                groups = level_groups(levels)
                leaders, runners = groups[0], (groups[1] if len(groups) > 1 else [])
                contrasts = []
                if len(leaders) == 1:
                    contrasts = [contrast(book[(leaders[0], tier)], book[(runner, tier)], definition=definition,
                                          tier=tier) for runner in runners]
                    resolved = [block["ci95_high"] < 0.0 for block in contrasts]
                    lead = ("the lead resolves (the CI lies entirely below zero)" if all(resolved) else
                            "the lead does not resolve (the CI includes zero)" if not any(resolved) else
                            "the lead resolves over some of the tied runners-up only")
                    reading = f"{leaders[0]} leads {' = '.join(runners)}; {lead}"
                else:
                    reading = f"{' = '.join(leaders)} tie exactly for the lead"
                seed_levels = _seed_levels(book, tier, definition, ts.METHODS)
                firsts = {seed: level_groups(values)[0] for seed, values in seed_levels.items()}
                blocks.append({
                    "tier": tier, "definition": definition, "primary": definition == PRIMARY_DEFINITION,
                    "version": version, "registered": False, "order": groups, "levels": levels, "leaders": leaders,
                    "runners_up": runners, "leader_minus_runner_up": contrasts, "reading": reading,
                    "per_seed_first_places": firsts, "per_seed_levels": seed_levels,
                    "pooled_leaders_lose_first_place_on_seeds": [seed for seed, first in firsts.items()
                                                                 if first != leaders],
                    "inputs": [book[(method, tier)].spec.cell_id for method in ts.METHODS],
                })
    for block in blocks:
        other = next(b for b in blocks if b["tier"] == block["tier"] and b["version"] == block["version"]
                     and b["definition"] != block["definition"])
        block["definition_dependent"] = block["reading"] != other["reading"]
        block["definition_dependent_on"] = "reading"
        block["per_seed_first_places_definition_dependent"] = block["per_seed_first_places"] != other[
            "per_seed_first_places"]
    return blocks


#: The per-episode fields whose identity the ties block reports, pair by pair.
IDENTITY_FIELDS: tuple[str, ...] = ("att_ours", "att_engine", *COMPANIONS, "horizon_vehicle_count", "episode_reward")


def ties_block(books: Mapping[str, Mapping[tuple[str, str], Cell]]) -> list[dict[str, Any]]:
    """Per tier, definition and (at ``random``) version: exactly-equal levels, identical-episode pairs with the fields
    they share, and every tie-dependent integer with its value under the rule and its range, pooled and per seed."""
    blocks = []
    arms = (*ts.METHODS, "behaviour")
    for tier in TIERS:
        for definition in DEFINITIONS:
            for version in _versions_of(tier):
                book = books["corrected" if version == "both" else version]
                levels = {arm: _level(book[(arm, tier)], definition) for arm in arms}
                groups = [group for group in level_groups(levels) if len(group) > 1]
                identical, partial = [], []
                for a, b in itertools.combinations(arms, 2):
                    rows_a, rows_b = book[(a, tier)].rows, book[(b, tier)].rows
                    pairs = list(zip(rows_a, rows_b, strict=True))
                    if any((x["seed"], x["draw_id"]) != (y["seed"], y["draw_id"]) for x, y in pairs):
                        raise ValueError(f"{a} and {b} at {tier} are not aligned by (seed, draw)")
                    equal = sum(1 for x, y in pairs if x[definition] == y[definition])
                    if equal == len(pairs):
                        identical.append({"pair": [a, b], "n_episodes": len(pairs),
                                          "identical_fields": [f for f in IDENTITY_FIELDS
                                                               if all(x[f] == y[f] for x, y in pairs)]})
                    if equal:
                        partial.append({"pair": [a, b], "n_equal": equal, "n": len(pairs)})
                six = {method: levels[method] for method in ts.METHODS}
                predicted = list(ts.predicted_order(tier))
                integers = []
                scopes = [("pooled", six)] + [(f"seed {seed}", values) for seed, values in
                                               _seed_levels(book, tier, definition, ts.METHODS).items()]
                for scope, values in scopes:
                    audit = tie_audit(values, predicted)
                    n = audit["orders_enumerated"]
                    if tier != "mappo1000":
                        for name in ("q2b", "q2b_hard"):
                            integers.append({"name": "Q2b" if name == "q2b" else "Q2b_hard", "scope": scope,
                                             "under_rule": audit[name]["under_rule"], "range": audit[name]["range"],
                                             "orders_enumerated": n})
                    integers.append({"name": "Q3a", "scope": scope, "under_rule": audit["rank"]["under_rule"],
                                     "range": audit["rank"]["range"], "tied_with": audit["rank"]["tied_with"],
                                     "orders_enumerated": n})
                    integers.append({"name": "first_place", "scope": scope,
                                     "under_rule": audit["first_place"]["under_rule"],
                                     "possible": audit["first_place"]["possible"], "orders_enumerated": n})
                blocks.append({
                    "tier": tier, "definition": definition, "version": version,
                    "cells": {arm: book[(arm, tier)].spec.cell_id for arm in arms},
                    "equal_level_groups": groups,
                    "equal_level_pairs": [list(pair) for group in groups for pair in itertools.combinations(group, 2)],
                    "identical_episode_pairs": identical, "pairs_with_equal_episodes": partial,
                    "integers": integers,
                    "rule": ("exactly equal levels are not an ordering: a tie is discordant for both orders "
                             "(docs/plans/p5.2.md:606-609); Q3a's rank is 1 + the arms strictly lower; every integer "
                             "carries its range over all tie-break orders (tie_audit)"),
                    "note": None if tier != "mappo1000" else "Q2b and Q2b-hard are not registered at mappo1000",
                })
    return blocks


def collapse_audit(books: Mapping[str, Mapping[tuple[str, str], Cell]]) -> dict[str, Any]:
    """``descriptive_collapse_audit`` (Amendment A, A2 Q9), not registered: Q6's rule applied verbatim to the non-DT
    arms on four tiers, both definitions, both random-tier IQL cells; whether the README's sentence holds."""
    rows = []
    for definition in DEFINITIONS:
        for arm in (*DT_ARMS, *NON_DT_METHODS):
            for tier in TIERS:
                versions = VERSIONS if (arm, tier) == ("iql", "random") else ("both",)
                for version in versions:
                    book = books["corrected" if version == "both" else version]
                    block = contrast(book[(arm, tier)], book[ANCHOR], definition=definition, tier=tier)
                    rows.append({"arm": arm, "tier": tier, "definition": definition, "version": version,
                                 "dt_arm": arm in DT_ARMS,
                                 "verdict": "COLLAPSED" if block["ci95_low"] > 0.0 else "NOT COLLAPSED",
                                 "contrast": block})
    holds = []
    for definition in DEFINITIONS:
        for version in VERSIONS:
            mine = [r for r in rows if r["definition"] == definition and r["version"] in ("both", version)]
            dt_never = all(r["verdict"] == "NOT COLLAPSED" for r in mine if r["dt_arm"])
            collapsing = {arm: [r["tier"] for r in mine if r["arm"] == arm and r["verdict"] == "COLLAPSED"]
                          for arm in NON_DT_METHODS}
            holds.append({"definition": definition, "version": version, "primary": definition == PRIMARY_DEFINITION,
                          "dt_arms_never_collapse": dt_never, "non_dt_arms_collapse_on": collapsing,
                          "holds": dt_never and all(collapsing.values())})
    return {
        "registered": False,
        "label": "descriptive, not registered (BRIEF_45 Amendment A, A2 Q9)",
        "rule": ("Q6's rule verbatim (docs/plans/p5.2.md:699-701): COLLAPSED iff the arm's mean is worse than the shared "
                 "random anchor's with the paired CI excluding 0, else NOT COLLAPSED"),
        "readme_sentence": README_COLLAPSE_SENTENCE,
        "rows": [r for r in rows if not r["dt_arm"]],
        "dt_rows": [r for r in rows if r["dt_arm"]],
        "sentence_holds": holds,
    }


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

#: The three records of Q5's verdict (Amendment A, A3), each with the coordinator's statement, verbatim.  GATE 2's is
#: quoted from its heading's claim through the sentence that is the error, "the `fixedtime` null is *predicted* by the
#: mechanism" (Amendment B, B2.6).
Q5_RECORDS: tuple[Quote, ...] = (
    Quote(_PACKET, (30, 30), "CONFIRMED twice, with a predicted null on `fixedtime`"),
    Quote(_PACKET, (572, 573),
          "The load sorter is confirmed across tiers, and `bc_top10_perix` is the repair — with a *predicted* null on "
          "`fixedtime` that the mechanism explains."),
    Quote(_BRIEF_27, (1274, 1281),
          "THE LOAD-SORTER MECHANISM IS CONFIRMED, decisively and twice\n\n`bc_top10_perix − bc_top10`: −469.69 "
          "[−477.81, −461.58] on `maxpressure`, −49.16 [−55.26, −43.05] on `random`, −0.28 [−0.67, +0.11] unresolved "
          "on `fixedtime`. ⭐ The global return quantile collapses (679.96 on `maxpressure`) exactly where the "
          "per-intersection filter does not (210.27). §1b's R2/R6 scope condition is no longer a one-tier observation "
          "— and the `fixedtime` null is *predicted* by the mechanism, because a state-independent controller makes "
          "every stream equivalent so both filters select the same thing."),
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
    "definition_dependent compares verdicts (readings where an entry has no verdict): Q0b's sign and Q4b's chain shape "
    "are reported per definition beside the flag, and the flag does not compare them (BRIEF_45 Amendment B, B2.13).",
)


def _find(statements: Sequence[Mapping[str, Any]], statement_id: str, tier: str | None, definition: str,
          version: str, arm: str | None = None) -> Mapping[str, Any]:
    """The one statement entry of *statement_id* at *tier* under *definition* in *version* (or stored once, ``both``)."""
    found = [s for s in statements if s["id"] == statement_id and s["tier"] == tier and s["arm"] == arm
             and s["definition"] == definition and s["version"] in (version, "both")]
    if len(found) != 1:
        raise ValueError(f"{len(found)} statement entries match {statement_id} at {tier} ({arm}) under {definition} in "
                         f"{version}; exactly one must")
    return found[0]


def _f4(value: float) -> str:
    """Four decimals with a sign, and an exact zero as P5.2's record writes it."""
    return "0.0000" if value == 0 else f"{value:+.4f}"


def _ci4(low: float, high: float) -> str:
    return f"[{_f4(low)}, {_f4(high)}]"


_NUMBER_WORDS: tuple[str, ...] = ("No", "One", "Two", "Three", "Four", "Five", "Six")


def record_values(statements: Sequence[Mapping[str, Any]], audit: Mapping[str, Any],
                  ties: Sequence[Mapping[str, Any]], books: Mapping[str, Mapping[tuple[str, str], Cell]]
                  ) -> dict[str, str]:
    """Every :data:`RECORD` value key, formatted at the record's precision from the ``as_committed`` ``att_ours``
    statements (the basis P5.2 scored on)."""
    def st(statement_id: str, tier: str | None, arm: str | None = None) -> Mapping[str, Any]:
        return _find(statements, statement_id, tier, "att_ours", "as_committed", arm)

    tiers = OUT_OF_SAMPLE_TIERS
    v: dict[str, str] = {}
    v["q0.stop_rule"] = "did not fire" if st("Q0.stop_rule", "mappo1000")["verdict"] == "CONTINUE" else "fired"
    q1 = st("Q1", None)
    v["q1.outcome"] = f"{q1['verdict']}, {q1['numbers']['n_held']} of {q1['numbers']['n_cells']}"
    v["q1.threshold"] = f"(threshold {q1['numbers']['threshold']})"
    relative = {t: st("Q1b", t)["numbers"]["relative_error"] for t in ("maxpressure", "fixedtime")}
    v["q1b.relative_errors"] = f"{100 * relative['maxpressure']:.2f} % / {100 * relative['fixedtime']:.2f} %"
    for entry in q1["numbers"]["cells"]:
        name = f"{entry['cell'][0]}@{entry['cell'][1]}"
        v[f"q1.measured.{name}"] = f"{entry['measured']:.2f}"
        v[f"q1.predicted.{name}"] = f"{entry['predicted']:.2f}"
        v[f"q1.relative_error.{name}"] = f"{100 * entry['relative_error']:.1f} %"
        v[f"q1.relative_error.{name}.parenthesised"] = f"({100 * entry['relative_error']:.1f} %)"
    q2a = [st("Q2a", t) for t in tiers]
    v["q2a.failed"] = f"FAILED {sum(e['verdict'] == 'FAILED' for e in q2a)}/{len(tiers)}"
    v["q2a.predicted"] = "/".join(f"`{e['numbers']['predicted_first']}`" for e in q2a)
    v["q2a.observed"] = "/".join(f"`{e['numbers']['measured_first']}`" for e in q2a)
    q2b = [st("Q2b", t) for t in tiers]
    v["q2b.failed"] = f"FAILED {sum(e['verdict'] == 'FAILED' for e in q2b)}/{len(tiers)}"
    v["q2b.counts"] = " · ".join(str(e["numbers"]["n_concordant"]) for e in q2b)
    v["q2b.overall"] = f"{sum(e['numbers']['n_concordant'] for e in q2b)}/{15 * len(tiers)}"
    v["q2b.fixedtime_max"] = f"(max {st('Q2b', 'fixedtime')['numbers']['range'][1]} < {ts.CONCORDANCE_THRESHOLD})"
    hard = [st("Q2b_hard", t)["numbers"] for t in tiers]
    v["q2b_hard.counts"] = " · ".join(f"{h['n_concordant']}/6" for h in hard)
    v["q2b_hard.total"] = f"{sum(h['n_concordant'] for h in hard)} of 18"
    v["q2b_hard.counts_total"] = f"{v['q2b_hard.counts']} = {v['q2b_hard.total']}"
    v["q2b_hard.ranges"] = " · ".join(f"`{t}` {h['range'][0]}..{h['range'][1]}" for t, h in zip(tiers, hard))
    total = f"{sum(h['range'][0] for h in hard)}..{sum(h['range'][1] for h in hard)}"
    v["q2b_hard.ranges_packet"] = f"{v['q2b_hard.ranges']}, i.e. {total} of 18"
    v["q2b_hard.ranges_gate2"] = f"{v['q2b_hard.ranges']}, giving a total range of {total}"
    v["q2b_hard.total_range"] = f"the true range is {total}"
    q3a = [st("Q3a", t) for t in tiers]
    v["q3a.failed"] = f"FAILED {sum(e['verdict'] == 'FAILED' for e in q3a)}/{len(tiers)}"
    v["q3a.ranks"] = " · ".join(f"{e['numbers']['rank']}/{e['numbers']['of']}" for e in q3a)
    v["q3a.rank_floor"] = f"(rank ≥ {min(e['numbers']['rank_range'][0] for e in q3a)} always)"
    against = all(st("Q3c", t)["reading"] == "resolves against the DT" for t in tiers)
    v["q3c.all_against"] = ("every paired difference against the best non-DT arm resolves against the DT" if against
                            else "not every paired difference against the best non-DT arm resolves against the DT")
    for tier in tiers:
        n = st("Q3b", tier)["numbers"]
        v[f"q3b.{tier}.mean"] = _f4(n["mean_difference"])
        v[f"q3b.{tier}.ci_prefixed"] = f"CI {_ci4(n['ci95_low'], n['ci95_high'])}"
    v["q3b.maxpressure.gate2"] = f"`{v['q3b.maxpressure.mean']}`, {v['q3b.maxpressure.ci_prefixed']}"
    exact = st("Q3b", "fixedtime")["numbers"]["exactly_zero"]
    v["q3b.fixedtime.exact"] = f"{'equals' if exact else 'does not equal'} it exactly on `fixedtime`"
    v["q3b.fixedtime.exact_upper"] = f"{'equals' if exact else 'does not equal'} it EXACTLY on `fixedtime`"
    d = {t: st("Q4.d", t)["numbers"] for t in TIERS}
    for tier, n in d.items():
        v[f"q4.{tier}.mean"] = _f4(n["mean_difference"]) if not n["exactly_zero"] else "0.0000"
        v[f"q4.{tier}.ci"] = "exact" if n["exactly_zero"] else _ci4(n["ci95_low"], n["ci95_high"])
        seeds = n["per_seed"]["n_seeds"]
        v[f"q4.{tier}.seeds"] = ("n/a" if n["exactly_zero"] else
                                 f"{seeds - n['per_seed']['reverses_against_pooled_sign']}/{seeds}")
    best, worst = d["mappo1000"]["mean_difference"], d["random"]["mean_difference"]
    v["q4.endpoints"] = f"{best:.2f} at the best tier against {worst:.2f} at the worst"
    v["q4.endpoints_packet"] = f"`{_f4(best)}` at the best tier, `{_f4(worst)}` at the"
    reverses = d["random"]["ci95_high"] < 0.0 < d["mappo1000"]["ci95_low"]
    v["q4.random_reverses"] = ("on `random` the sign REVERSES with the CI excluding zero" if reverses else
                               "on `random` the sign does not reverse with the CI excluding zero")
    v["q4b.reading"] = "INVERTED" if st("Q4b", None)["numbers"]["strictly_decreasing"] else "NOT INVERTED"
    for tier in tiers:
        n = st("Q5b", tier)["numbers"]
        v[f"q5b.{tier}.mean"] = f"{n['mean_difference']:+.2f}"
        v[f"q5b.{tier}.ci"] = f"[{n['ci95_low']:+.2f}, {n['ci95_high']:+.2f}]"
    v["q5b.fixedtime.ci_unresolved"] = (f"{v['q5b.fixedtime.ci']} " +
                                        ("unresolved" if st("Q5b", "fixedtime")["verdict"] == "NOT RESOLVED"
                                         else "resolved"))
    levels = st("ranking", "maxpressure")["numbers"]["levels"]
    v["level.bc_top10@maxpressure"] = f"{levels['bc_top10']:.2f} on `maxpressure`"
    v["level.bc_top10_perix@maxpressure"] = f"({levels['bc_top10_perix']:.2f})"
    v["level.bc_top10_perix@maxpressure.bare"] = f"{levels['bc_top10_perix']:.2f}"
    collapsed = [s for s in statements if s["id"] == "Q6" and s["definition"] == "att_ours"
                 and s["verdict"] == "COLLAPSED"]
    v["q6.none"] = ("no DT arm collapsed on any tier" if not collapsed else
                    f"{len(collapsed)} DT arm-tier cells collapsed")
    holds = next(h for h in audit["sentence_holds"] if h["definition"] == "att_ours" and h["version"] == "as_committed")
    v["audit.only_dt"] = ("the DT arms are the only arms that never collapse" if holds["holds"] else
                          "the DT arms are not the only arms that never collapse")
    v["audit.only_dt_upper"] = ("the DT arms are the ONLY arms that never collapse" if holds["holds"] else
                                "the DT arms are NOT the ONLY arms that never collapse")
    book = books["as_committed"]

    def rows(method: str) -> tuple[Mapping[str, Any], ...]:
        return book[(method, "fixedtime")].rows

    triple = list(zip(rows("dt_spatial"), rows("dt_nomix"), rows("behaviour"), strict=True))
    differing = sum(1 for a, b, c in triple if not a["att_ours"] == b["att_ours"] == c["att_ours"])
    v["fixedtime.three_identical"] = f"{differing} of {len(triple)} episodes differ between all three"
    perix = all(a["att_ours"] == b["att_ours"] for a, b in zip(rows("bc_top10_perix"), rows("bc"), strict=True))
    v["fixedtime.perix_identical"] = ("`bc_top10_perix` is likewise identical to `bc`" if perix else
                                      "`bc_top10_perix` is not identical to `bc`")
    pairs = list(zip(rows("bc"), rows("behaviour"), strict=True))
    v["fixedtime.bc_vs_behaviour"] = (f"`bc` differs from `behaviour` on "
                                      f"{sum(1 for a, b in pairs if a['att_ours'] != b['att_ours'])} of {len(pairs)}")
    zeros = sum(1 for statement_id in ("Q3b", "Q4.d", "Q5a", "Q5b", "Q5c")
                if st(statement_id, "fixedtime")["numbers"]["exactly_zero"])
    v["fixedtime.zero_contrasts"] = f"{_NUMBER_WORDS[zeros]} contrasts are exactly `0.0000` at that tier"
    fixed = st("ranking", "fixedtime")["numbers"]["levels"]
    if fixed["dt_spatial"] == fixed["dt_nomix"] and fixed["bc"] == fixed["bc_top10_perix"]:
        v["fixedtime.tied_levels"] = (f"`dt_spatial` = `dt_nomix` = {fixed['dt_spatial']!r} and `bc` = "
                                      f"`bc_top10_perix` = {fixed['bc']!r}")
    else:
        v["fixedtime.tied_levels"] = "the two fixedtime ties do not hold"
    d1 = st("Q0.d1", "mappo1000")["numbers"]
    d4 = st("Q0a", "mappo1000")["numbers"]
    i = st("Q0b", "mappo1000")["numbers"]
    v["q0.d1.mean"], v["q0.d1.ci"] = _f4(d1["mean_difference"]), _ci4(d1["ci95_low"], d1["ci95_high"])
    v["q0.d1.plan"] = f"measured: {v['q0.d1.mean']}, CI {v['q0.d1.ci']}"
    v["q0.d4.mean"], v["q0.d4.ci"] = _f4(d4["mean_difference"]), _ci4(d4["ci95_low"], d4["ci95_high"])
    v["q0.i.mean"], v["q0.i.ci"] = _f4(i["mean_difference"]), _ci4(i["ci95_low"], i["ci95_high"])
    v["q0.i.negative_draws"] = f"`I < 0` on {i['draws_negative']} of {i['n_draws']} draws"
    share = 100 * abs(i["mean_difference"]) / d1["mean_difference"]
    v["q0.i.reduction"] = f"The {share:.1f} % {'reduction' if i['mean_difference'] < 0 else 'increase'}"
    v["q0.i.attributable"] = f"{share:.1f} % of it is attributable"
    return v


def _differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Every field of *expected* missing from *actual* or not equal to it (dicts may gain keys; lists must be equal);
    a bool is equal only to a bool."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected a dict"]
        out: list[str] = []
        for key, value in expected.items():
            out.extend([f"{path}.{key}: missing"] if key not in actual else _differences(value, actual[key],
                                                                                           f"{path}.{key}"))
        return out
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [f"{path}: {expected!r} != {actual!r}"]
        return [problem for i, (e, a) in enumerate(zip(expected, actual)) for problem in _differences(e, a, f"{path}[{i}]")]
    if isinstance(expected, bool) or isinstance(actual, bool):
        return [] if expected is actual else [f"{path}: {expected!r} != {actual!r}"]
    return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]


def _leaves(node: Any) -> int:
    if isinstance(node, dict):
        return sum(_leaves(value) for value in node.values())
    if isinstance(node, list):
        return sum(_leaves(value) for value in node) if node else 1
    return 1


def _source(file: str, lines: Sequence[int]) -> str:
    return f"{file}:{lines[0]}" + (f"-{lines[1]}" if lines[1] != lines[0] else "")


def _verify_exception(row: RecordRow, statements: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A declared exception holds only if the recorded integer is reachable by some tie-break order and the declared
    rule gives another; otherwise the build is refused (the explanation would be false)."""
    import re

    def st(statement_id: str, tier: str) -> Mapping[str, Any]:
        return _find(statements, statement_id, tier, "att_ours", "as_committed")

    if row.exception == "deferred_108":
        entries = [st("Q2b", t)["numbers"] for t in OUT_OF_SAMPLE_TIERS]
        rule = [e["n_concordant"] for e in entries]
        ranges = [e["range"] for e in entries]
        recorded = [int(x) for x in re.findall(r"\d+", row.recorded)]
        if row.value == "q2b.counts":
            ok = len(recorded) == 3 and all(lo <= r <= hi for r, (lo, hi) in zip(recorded, ranges)) and recorded != rule
        elif row.value == "q2b.overall":
            low, high = sum(r[0] for r in ranges), sum(r[1] for r in ranges)
            ok = low <= recorded[0] <= high and recorded[0] != sum(rule)
            rule, ranges = [sum(rule)], [[low, high]]
            recorded = recorded[:1]
        else:
            ok = False
    elif row.exception == "q3a_fixedtime":
        entries = [st("Q3a", t)["numbers"] for t in OUT_OF_SAMPLE_TIERS]
        rule = [e["rank"] for e in entries]
        ranges = [e["rank_range"] for e in entries]
        recorded = [int(x) for x in re.findall(r"(\d+)/6", row.recorded)]
        ok = len(recorded) == 3 and all(lo <= r <= hi for r, (lo, hi) in zip(recorded, ranges)) and recorded != rule
    else:
        raise ValueError(f"{row.exception!r} is not a declared exception")
    if not ok:
        raise ValueError(f"{_source(row.file, row.lines)}: the declared exception {row.exception} does not verify "
                         f"(recorded {row.recorded!r}, rule {rule}, ranges {ranges})")
    return {"recorded_integers": recorded, "under_the_rule": rule, "ranges": ranges,
            "check": "the recorded integers lie in their ranges over every tie-break order, and the rule's differ"}


#: The most a value of this artifact may differ from P8.4b's V4/V5 or P5.2's stop-rule record, in units in the last
#: place of the larger magnitude, before the build refuses (``BRIEF_45`` Amendment B, B2.5).
ULP_BOUND = 4


def assert_within_ulps(mine: float, recorded: float, *, what: str) -> None:
    """Refuse unless ``|mine - recorded| <= ULP_BOUND * math.ulp(max(|mine|, |recorded|))``: the route explanation
    below covers a few units in the last place and nothing more, so it is verified, not asserted (Amendment B, B2.5)."""
    if abs(mine - recorded) > ULP_BOUND * math.ulp(max(abs(mine), abs(recorded))):
        raise ValueError(
            f"{what}: this artifact's {mine!r} and the record's {recorded!r} differ by more than {ULP_BOUND} ulps "
            f"({mine - recorded!r}), so the arithmetic route cannot explain the difference (BRIEF_45 Amendment B, B2.5)")


#: Why P8.4b's V4 and V5 under att_ours sit a unit in the last place from this artifact's values (plan F1, V10-V11).
P8_4B_ROUTE_EXPLANATION = (
    "P8.4b averaged each draw's five seeds with Python's built-in sum (att_rederivation._per_draw_means), which is "
    "compensated on Python 3.12; this artifact uses the registered route, dt_gate._per_draw_means (numpy.mean, a plain "
    "left-to-right sum over five seeds; docs/plans/p5.2.md:407-412). The record's own route over the same rows "
    "reproduces the recorded values exactly (record_route_reproduces), so the inputs are V4's and V5's and the "
    "difference is the arithmetic route's alone (BRIEF_45 Amendment A, A2 Q1); the build refuses any difference "
    "larger than 4 ulps of the larger magnitude (Amendment B, B2.5)")


STOP_RULE_ROUTE_EXPLANATION = (
    "P5.2's stop-rule record was written by tier_sweep.score_stop_rule, which averages each draw's seeds with Python's "
    "built-in sum (compensated on Python 3.12); this artifact's d4 is the registered route's (dt_gate._per_draw_means). "
    "score_stop_rule over the same evaluation records reproduces the record exactly (record_route_reproduces); the "
    "verdict is the same, and the build refuses any difference larger than 4 ulps of the larger magnitude (BRIEF_45 "
    "Amendment B, B2.5)")


def comparison_with_the_record(statements: Sequence[Mapping[str, Any]], audit: Mapping[str, Any],
                               ties: Sequence[Mapping[str, Any]], inputs: Inputs, *,
                               record: Sequence[RecordRow] = RECORD) -> dict[str, Any]:
    """Every recorded value beside this artifact's, equal or explained; refused if a row is neither, or if a declared
    exception does not verify.  Also: P5.2b's statements field by field, P8.4b's V4 and V5, P5.1's P1 and P3, P5.2's
    stop-rule record (each recomputed by its own route where the routes differ), Q5's three records (A3) and the
    never-scored list."""
    books = make_books(inputs.cells)
    values = record_values(statements, audit, ties, books)
    literal = []
    for row in record:
        if row.value not in values:
            raise ValueError(f"{_source(row.file, row.lines)} names the value {row.value!r}, which this artifact does "
                             "not compute")
        mine = values[row.value]
        equal = normalise(mine) == normalise(row.recorded)
        where = _source(row.file, row.lines)
        if equal and row.exception is not None:
            raise ValueError(f"{where} is declared exception {row.exception}, but the record and this artifact agree "
                             f"({mine!r}): the declared explanation would be false")
        if not equal and row.exception is None:
            raise ValueError(f"{where} records {row.recorded!r} and this artifact gives {mine!r}: neither equal nor "
                             "explained")
        literal.append({"source": where, "recorded": row.recorded, "this_artifact": mine, "equal": equal,
                        "value": row.value, "exception": None if equal else row.exception,
                        "explanation": None if equal else RECORD_EXCEPTIONS[row.exception],
                        "verified": None if equal else _verify_exception(row, statements)})

    committed = inputs.records["p5_2b_artifact"]["statements"]
    compared = []
    for definition in DEFINITIONS:
        for when, version in (("before", "as_committed"), ("after", "corrected")):
            oos = {cell: episodes_of(books[version][cell], definition) for cell in ts.OUT_OF_SAMPLE_CELLS}
            mine = json.loads(json.dumps(tier_composer(oos, tier="random", with_hard_subset=True)))
            theirs = json.loads(json.dumps(committed[definition][when]))
            theirs["q1"].pop("predictions_registered_on", None)
            problems = _differences(theirs, mine)
            if problems:
                raise ValueError(f"{P5_2B_ARTIFACT}'s statements.{definition}.{when} are not reproduced field by field "
                                 f"at random: {problems[:4]}")
            compared.append({"statements": f"{definition}.{when}", "version": version,
                             "fields_compared": _leaves(theirs)})
    p5_2b = {"file": P5_2B_ARTIFACT, "equal": True, "compared": compared,
             "excluded": "q1.predictions_registered_on, the one field P5.2b's report adds after its composer",
             "route": ("tier_composer at random over this artifact's cells, every committed field present and equal "
                       "(added fields allowed; BRIEF_45 Amendment A, A2 Q2); cells.corrected and cells.original are "
                       "refused at load unless equal to this artifact's (load_inputs)")}

    grid = inputs.records["p5_1_grid"]["predictions"]
    d1 = _find(statements, "Q0.d1", "mappo1000", "att_ours", "as_committed")["numbers"]

    def pairs_of(mine: Mapping[str, Any], theirs: Mapping[str, Any], outcome: tuple[Any, Any] | None
                 ) -> dict[str, dict[str, Any]]:
        out = {key: {"recorded": theirs[key], "this_artifact": mine[key]}
               for key in ("mean_difference", "ci95_low", "ci95_high")}
        for key in ("rank_biserial", "median_difference", "wins", "losses", "ties"):
            out[key] = {"recorded": theirs[key], "this_artifact": mine["paired_comparison"][key]}
        out["p_value"] = {"recorded": theirs["p_value"], "this_artifact": mine["paired_comparison"]["wilcoxon"]["p_value"]}
        if outcome is not None:
            out["outcome"] = {"recorded": outcome[0], "this_artifact": outcome[1]}
        return out

    p1 = pairs_of(d1, grid["P1"], None)
    p3 = {}
    for arm in DT_ARMS:
        q6 = _find(statements, "Q6", "mappo1000", "att_ours", "as_committed", arm)["numbers"]
        p3[arm] = pairs_of(q6, grid["P3"][arm], (grid["P3"][arm]["outcome"], q6["p5_1_outcome"]))
    unequal = [f"P1.{k}" for k, pair in p1.items() if pair["recorded"] != pair["this_artifact"]] + [
        f"P3.{arm}.{k}" for arm, block in p3.items() for k, pair in block.items()
        if pair["recorded"] != pair["this_artifact"]]
    if unequal:
        raise ValueError(f"{P5_1_GRID}'s committed {unequal[:4]} are not reproduced under ==: P5.1's route is the "
                         "registered one, so they must be")
    p5_1 = {"file": P5_1_GRID, "equal": True, "P1": p1, "P3": p3,
            "route": "d1 (Q0) is P5.1's P1 and Q6 at mappo1000 is its P3, both on the registered route"}

    contrasts = {c["contrast_id"]: c for c in inputs.records["p8_4b_artifact"]["contrasts"]}
    p8 = []
    for contrast_id, (left_id, right_id), statement_id, verdict_of in (
            ("V4", ("p5_1:dt_spatial@grid4x4_mappo1000", "p5_1:dt_nomix@grid4x4_mappo1000"), "Q0.d1",
             sm._outcome_from_interval),
            ("V5", H4_CELLS, "Q0a", ts.stop_rule_verdict)):
        for definition in DEFINITIONS:
            recorded = contrasts[contrast_id]["by_definition"][definition]
            left, right = inputs.cells[left_id], inputs.cells[right_id]
            block = {c.spec.arm: {(r["seed"], r["draw_id"]): {definition: r[definition]} for r in c.rows}
                     for c in (left, right)}
            differences = ar._paired_differences(block, left.spec.arm, right.spec.arm, definition)
            stats = mean_ci95(differences)
            route = (stats.mean, stats.mean - stats.ci95, stats.mean + stats.ci95)
            keys = ("mean_difference", "ci95_low", "ci95_high")
            if route != tuple(recorded[k] for k in keys) or len(differences) != recorded["n_paired"]:
                raise ValueError(f"P8.4b's own route (att_rederivation._paired_differences + mean_ci95) over the rows "
                                 f"read does not reproduce {contrast_id} under {definition}: {route} against "
                                 f"{[recorded[k] for k in keys]}")
            mine = _find(statements, statement_id, "mappo1000", definition, "as_committed")["numbers"]
            outcome = verdict_of(mine["ci95_low"], mine["ci95_high"])
            if outcome != recorded["outcome"]:
                raise ValueError(f"{contrast_id} under {definition}: this artifact's interval gives {outcome}, P8.4b "
                                 f"recorded {recorded['outcome']}")
            for k in keys:
                assert_within_ulps(mine[k], recorded[k], what=f"P8.4b's {contrast_id} under {definition}, {k}")
            equal = all(mine[k] == recorded[k] for k in keys)
            p8.append({"contrast_id": contrast_id, "definition": definition, "statement": statement_id,
                       "recorded": {**{k: recorded[k] for k in keys}, "n_paired": recorded["n_paired"],
                                    "outcome": recorded["outcome"]},
                       "this_artifact": {**{k: mine[k] for k in keys}, "n_paired": mine["n_draws"], "outcome": outcome},
                       "equal": equal, "outcome_equal": True,
                       "difference": {k: mine[k] - recorded[k] for k in keys},
                       "difference_in_ulps_of_the_record": {k: (mine[k] - recorded[k]) / math.ulp(recorded[k])
                                                            for k in keys},
                       "within_ulp_bound": {"bound": f"{ULP_BOUND} * math.ulp(max(|a|, |b|)) per value", "holds": True},
                       "record_route_reproduces": True,
                       "explanation": None if equal else P8_4B_ROUTE_EXPLANATION})

    stop_record = inputs.records["stop_rule_record"]
    spatial, nomix = (inputs.records["h4_payloads"][cell_id] for cell_id in H4_CELLS)
    route = ts.score_stop_rule(spatial, nomix, list(dt_gate.TRAINING_SEEDS))
    keys = ("mean_difference", "ci95_low", "ci95_high", "verdict")
    if tuple(route[k] for k in keys) != tuple(stop_record[k] for k in keys):
        raise ValueError(f"tier_sweep.score_stop_rule over the two 4-head records does not reproduce "
                         f"output/{P5_2_STOP_RULE_RECORD}")
    d4 = _find(statements, "Q0a", "mappo1000", "att_ours", "as_committed")["numbers"]
    verdict = _find(statements, "Q0.stop_rule", "mappo1000", "att_ours", "as_committed")["verdict"]
    if verdict != stop_record["verdict"]:
        raise ValueError(f"the stop rule's verdict is {verdict} here and {stop_record['verdict']} in P5.2's record")
    for k in keys[:3]:
        assert_within_ulps(d4[k], stop_record[k], what=f"P5.2's stop-rule record, {k}")
    equal = all(d4[k] == stop_record[k] for k in keys[:3])
    stop = {"file": f"output/{P5_2_STOP_RULE_RECORD}",
            "recorded": {k: stop_record[k] for k in keys}, "this_artifact": {**{k: d4[k] for k in keys[:3]},
                                                                              "verdict": verdict},
            "equal": equal, "difference": {k: d4[k] - stop_record[k] for k in keys[:3]},
            "within_ulp_bound": {"bound": f"{ULP_BOUND} * math.ulp(max(|a|, |b|)) per value", "holds": True},
            "record_route_reproduces": True, "explanation": None if equal else STOP_RULE_ROUTE_EXPLANATION}

    pairs = {"Q5a": ("bc_top10", "bc"), "Q5b": ("bc_top10_perix", "bc_top10"), "Q5c": ("bc_top10_perix", "bc")}
    fixedtime = {}
    for statement_id, (left_method, right_method) in pairs.items():
        from_levels = ts.PREDICTED_LEVELS[left_method]["fixedtime"] - ts.PREDICTED_LEVELS[right_method]["fixedtime"]
        quoted = REGISTERED_VALUES[statement_id]["fixedtime"]
        if round(from_levels, 2) != quoted:
            raise ValueError(f"{statement_id}'s registered fixedtime value {quoted} is not the difference of "
                             f"tier_sweep.PREDICTED_LEVELS ({from_levels!r})")
        fixedtime[statement_id] = {
            "registered": quoted, "registered_from_predicted_levels": from_levels,
            "measured": {d: {**{k: _find(statements, statement_id, "fixedtime", d, "as_committed")["numbers"][k]
                                for k in ("mean_difference", "ci95_low", "ci95_high", "exactly_zero")},
                             "verdict": _find(statements, statement_id, "fixedtime", d, "as_committed")["verdict"]}
                         for d in DEFINITIONS}}
    verdicts = {d: {s: _find(statements, f"{s}.aggregate", None, d, "as_committed")["verdict"] for s in pairs}
                for d in DEFINITIONS}
    per_tier = {d: {s: _find(statements, f"{s}.aggregate", None, d, "as_committed")["numbers"]["tiers"] for s in pairs}
                for d in DEFINITIONS}
    descriptive = {d: {"reading": "the global return filter collapses where the per-intersection one does not",
                       "tiers_where_q5b_resolves": [t for t, verdict in per_tier[d]["Q5b"].items() if verdict == "HELD"],
                       "label": "descriptive, not a registered verdict (BRIEF_45 Amendment A, A3)"} for d in DEFINITIONS}
    q5 = [{"record": {"source": _source(quote.file, quote.lines), "text": quote.text},
           "registered_verdicts": verdicts, "primary": verdicts[PRIMARY_DEFINITION], "per_tier": per_tier,
           "fixedtime_registered_beside_measured": fixedtime, "statement": Q5_STATEMENT,
           "statement_detail": ("the record reported Q5b's values on three tiers (maxpressure, fixedtime, random), "
                                "counted the two that resolved as confirmations, gave no value at mappo1000, where Q5b "
                                "is also registered, and read fixedtime's unresolved interval as a predicted null; the "
                                "registration predicted +820.65 (Q5a), -812.15 (Q5b) and +8.49 (Q5c) there"),
           "descriptive_reading": descriptive,
           "whose_error": ("BRIEF_27's GATE 2 is a coordinator ruling, so the error is the coordinator's; it is logged "
                           "in PROJECT_PLAN section 8 (dated 2026-10-10)")} for quote in Q5_RECORDS]

    never = []
    for statement_id, tier in NEVER_SCORED:
        if statement_id in ("Q5a", "Q5c"):
            entries = [s for s in statements if s["id"] in (statement_id, f"{statement_id}.aggregate")]
        elif statement_id == "Q6":
            entries = [s for s in statements if s["id"] == "Q6"]
        else:
            entries = [s for s in statements if s["id"] == statement_id and s["tier"] == tier]
        never.append({"statement": statement_id, "tier": tier,
                      "recorded": ("Q6's numbers: never scored; the verdict was recorded (`docs/returns/P5.2.md:31`)"
                                   if statement_id == "Q6" else
                                   "never scored in any committed record, under either definition (BRIEF_45 section "
                                   "1.3)"),
                      "this_artifact": [{key: entry.get(key) for key in ("id", "tier", "arm", "definition", "version",
                                                                          "verdict", "reading")}
                                        | {key: entry["numbers"].get(key) for key in ("mean_difference", "ci95_low",
                                                                                     "ci95_high")}
                                        for entry in entries]})
    return {
        "basis": ("the record scored P5.2's original cells under att_ours (P5.2's att_horizon), so every literal row "
                  "compares this artifact's as_committed att_ours value at the record's precision"),
        "literal_records": literal, "exceptions": dict(RECORD_EXCEPTIONS), "q5": q5, "never_scored": never,
        "p5_2b_artifact": p5_2b, "p5_1_grid": p5_1, "p8_4b": p8, "p5_2_stop_rule_record": stop,
    }


def changes_block(statements: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per statement and tier: as committed ``att_ours`` -> corrected ``att_ours`` -> corrected ``att_engine``."""
    def summary(entry: Mapping[str, Any]) -> dict[str, Any]:
        numbers = entry["numbers"]
        out = {"verdict": entry["verdict"], "reading": entry["reading"]}
        for key in ("mean_difference", "ci95_low", "ci95_high", "n_concordant", "rank", "n_held"):
            if key in numbers:
                out[key] = numbers[key]
        return out

    keys: list[tuple[str, str | None, str | None]] = []
    for entry in statements:
        key = (entry["id"], entry["tier"], entry["arm"])
        if key not in keys:
            keys.append(key)
    out = []
    for statement_id, tier, arm in keys:
        steps = [summary(_find(statements, statement_id, tier, definition, version, arm))
                 for definition, version in (("att_ours", "as_committed"), ("att_ours", "corrected"),
                                             ("att_engine", "corrected"))]
        outcome = [(step["verdict"], step["reading"]) for step in steps]
        out.append({"id": statement_id, "tier": tier, "arm": arm, "as_committed_att_ours": steps[0],
                    "corrected_att_ours": steps[1], "corrected_att_engine": steps[2],
                    "changed_by_the_correction": outcome[0] != outcome[1], "changed_by_rule_r": outcome[1] != outcome[2]})
    return out


# ----------------------------------------------------------------------
# The artifact, the barrier and the command line
# ----------------------------------------------------------------------


def definitions_block() -> dict[str, Any]:
    """Rule R, the two definitions, the paired unit, the interval, the arithmetic route and the vocabularies."""
    return {
        "rule_r": {"quotes": _quotes("rule_r"), "primary": PRIMARY_DEFINITION, "beside": "att_ours",
                   "fired": ("on grid4x4, by P8.4b's gate (docs/data/p8_4b_g0_reference.json); recorded in PROJECT_PLAN "
                             "section 10 under its dated heading of 2026-08-31"),
                   "reporting": ("every statement under both definitions; primary marks att_engine's entry; "
                                 "definition_dependent marks where the two verdicts (or readings) differ")},
        "att_ours": ("P5.2's att_horizon: the platform's average travel time at the horizon, read from the evaluation "
                     "records (P5.1's, P5.2's and P5.2b's)"),
        "att_engine": ("the engine's average travel time of the same episodes, re-derived by deterministic replay "
                       "(P8.4b's campaign; P5.2b's re-derivation for the corrected cell)"),
        "common_scoring": _quotes("common"),
        "paired_unit": ("per draw, the mean over the five training seeds (dt_gate._per_draw_means), over the identical "
                        "(seed, draw) grid of two complete cells; differences per draw in ascending draw order"),
        "interval": ("dt_gate.mean_ci95 over the per-draw differences: ddof=1, half-width 1.96 * sd / sqrt(n); low = "
                     "mean - half-width, high = mean + half-width, width = high - low"),
        "arithmetic_route": ("one route everywhere, the registered one: numpy.mean per draw (a plain left-to-right sum "
                             "over five seeds) and mean_ci95 over the draws.  P8.4b's att_rederivation._per_draw_means "
                             "and tier_sweep.score_stop_rule use Python's built-in sum (compensated on Python 3.12), "
                             "which can differ in the last place; comparison_with_the_record names each such "
                             "difference, reproduces the record by its own route (BRIEF_45 Amendment A, A2 Q1) and "
                             "refuses a difference larger than 4 ulps of the larger magnitude (Amendment B, B2.5)"),
        "outcomes": {"interval": ("HELD / FAILED / NOT RESOLVED by the side of zero the CI lies on relative to the "
                                  "predicted sign (spatial_mixing._outcome_from_interval); a CI touching zero is NOT "
                                  "RESOLVED"),
                     "conjunction": ("a statement registered on all three or four tiers is HELD iff every tier is "
                                     "HELD, FAILED iff any tier is FAILED, NOT RESOLVED otherwise (BRIEF_45 section "
                                     "3.3)"),
                     "q6": ("COLLAPSED iff the arm is worse than the shared anchor with the paired CI excluding 0, "
                            "else NOT COLLAPSED; P5.1's three-valued outcome beside it (collapse_criterion)"),
                     "q0_stop_rule": "STOP iff the CI of mean(d4) lies entirely below zero (tier_sweep.stop_rule_verdict)",
                     "q3c": "no verdict: a value per tier and P5.2b's reading (for / against the DT, NOT RESOLVED)"},
        "exact_zero": ("a difference identically zero on every draw has CI [0, 0] and sign zero, and the registered rule "
                       "decides the verdict (BRIEF_45 Amendment B, B2.1): NOT RESOLVED for a predicted-sign interval "
                       "rule, where [0, 0] lies on neither side of zero; HELD for Q0b's inclusion rule, where [0, 0] "
                       "includes zero. An exact zero is always labelled exactly zero (identical episodes, or not "
                       "identical where the episodes differ) and is never offered as evidence for a predicted sign"),
        "ties": {"quotes": _quotes("tie_rule"),
                 "rule": ("exactly equal levels are not an ordering; Q3a's rank is 1 + the arms strictly lower; every "
                          "tie-dependent integer carries its range over all tie-break orders")},
        "versions": {"corrected": "the random tier's IQL cell is P5.2b's (the paper's)",
                     "as_committed": "the random tier's IQL cell is P5.2's original (DEFERRED 106)",
                     "both": ("the statement reads no random-tier IQL cell; computed in both versions, asserted "
                              "identical, stored once")},
        "per_seed": ("every two-arm contrast carries tier_sweep.per_seed_block (P5.1's per_seed_advantages and "
                     "seed_reversal_qualifier), and I carries the same worker over composite rows (BRIEF_45 "
                     "Amendment B, B2.2); for the behaviour and random-anchor cells a seed label is an "
                     "evaluation label, not a training seed"),
        "d8": {"quotes": _quotes("D8"),
               "carried": ("rank_biserial, median_difference, wins, losses, ties and the Wilcoxon block from "
                           "offline_baselines.paired_comparison, unchanged; the p-value is never read by a verdict")},
        "d15": {"quotes": _quotes("D15"),
                "carried": ("each cell's median and IQR per definition (numpy.percentile, method='linear') and A5's "
                            "horizon vehicle count, asserted equal to each evaluation record's own cell block")},
        "not_registered": _quotes("not_registered"),
    }


ROLE = (
    "P5.2c: P5.2's registered statements (docs/plans/p5.2.md section 4, Q0-Q6) scored by committed code on every tier "
    "as registered, under both ATT definitions (Rule R makes att_engine primary on grid4x4) and both random-tier IQL "
    "cells (P5.2b's corrected cell, the paper's, and P5.2's original); every tie-dependent integer with its range; every "
    "value P5.2's record states beside this artifact's, equal or explained. Computed from committed records read at "
    "their digests; no episode was rolled.")


REGISTERED_IN = (
    "docs/briefs/BRIEF_45_p5.2c_statements.md, its Amendment A (gate G0; the plan docs/plans/p5.2c.md at 1267715) "
    "and its Amendment B (gate G1); the author's ruling of 2026-10-08 (PROJECT_PLAN section 8)")


def build_artifact(inputs: Inputs, *, git: Mapping[str, Any], code: Sequence[Mapping[str, Any]],
                   record: Sequence[RecordRow] = RECORD) -> dict[str, Any]:
    """The ``p5.2-statements/1.0`` artifact from verified inputs; pure."""
    books = make_books(inputs.cells)
    statements = all_statements(books)
    ties = ties_block(books)
    audit = collapse_audit(books)
    comparison = comparison_with_the_record(statements, audit, ties, inputs, record=record)
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": ROLE,
        "registered_in": REGISTERED_IN,
        "git": dict(git),
        "provenance": {**inputs.provenance, "code": [dict(entry) for entry in code]},
        "definitions": definitions_block(),
        "cells": cells_block(inputs.cells),
        "statements": statements,
        "ties": ties,
        "leaders": leaders_block(books),
        "descriptive_collapse_audit": audit,
        "comparison_with_the_record": comparison,
        "changes": changes_block(statements),
        "what_this_does_not_say": list(WHAT_THIS_DOES_NOT_SAY),
    }


def serialize(artifact: Mapping[str, Any]) -> bytes:
    """The canonical bytes: sorted keys, two-space indent, no NaN, one trailing newline."""
    return (json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def assert_out_path(path: str | Path, roots: Roots, inputs: Inputs) -> Path:
    """The barrier (``PROJECT_PLAN`` section 7): refuse an out path under ``--output-root``, under any tree holding
    ``SHA256SUMS_p5_2.txt``, equal to an input, not ``.json``, whose parent does not exist, that already exists, or
    that is a symlink, dangling or not -- the UNRESOLVED path is examined, so the artifact lands at the path the command
    line names (``BRIEF_45`` Amendment B, B2.9)."""
    repo = Path(roots.repo_root).resolve()
    candidate = Path(path)
    unresolved = candidate if candidate.is_absolute() else repo / candidate
    resolved = unresolved.resolve()
    if resolved.suffix != ".json":
        raise ValueError(f"{resolved}: the artifact must be a .json file")
    ts.assert_writable(resolved, ts.protected_roots_from([roots.output_root]))
    for ancestor in resolved.parents:
        if (ancestor / P5_2_SUMS_NAME).is_file():
            raise PermissionError(f"{resolved} lies under {ancestor}, which holds {P5_2_SUMS_NAME}: a real output tree "
                                  "is read-only to this task whatever --output-root says")
    inputs_read = {(Path(roots.output_root) / entry["file"].removeprefix("output/")).resolve()
                   if entry["file"].startswith("output/") else (repo / entry["file"]).resolve()
                   for entry in inputs.provenance["inputs"]}
    if resolved in inputs_read:
        raise ValueError(f"{resolved} is an input of this artifact")
    if not resolved.parent.is_dir():
        raise FileNotFoundError(f"the out path's parent directory {resolved.parent} does not exist; nothing is created")
    if unresolved.is_symlink() or unresolved.exists():
        what = "is a symlink" if unresolved.is_symlink() else "already exists"
        raise FileExistsError(f"{unresolved} {what}: the artifact is written once, at the path named (remove it "
                              "deliberately to rebuild)")
    return resolved


def write_artifact(path: Path, data: bytes) -> Path:
    """Write *data* once: a dot-temporary in the same directory, then ``os.link`` onto *path* (refusing an existing
    file); the temporary is removed whatever happens."""
    target = Path(path)
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        with open(temporary, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def build_parser() -> argparse.ArgumentParser:
    """``build`` and ``check``, each with ``--repo-root`` and ``--output-root`` (required) and ``--out``."""
    parser = argparse.ArgumentParser(
        prog="python -P -m offline.tier_statements",
        description=("P5.2c: P5.2's registered statements scored under both ATT definitions. Run with PYTHONPATH set to "
                     "the tree whose code runs, and --repo-root naming the same tree (BRIEF_45 Amendment A, A1.1)."))
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("build", "verify every input, then write the artifact once at --out"),
                       ("check", "verify every input and print what build would read; write nothing")):
        command = commands.add_parser(name, help=text)
        command.add_argument("--repo-root", required=True, help="the checkout whose docs/ is read and whose code runs")
        command.add_argument("--output-root", required=True, help="the main tree's output/ (read only)")
        command.add_argument("--out", help="the artifact's path (build); with check, whether build would accept it")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Exit 0 accepted; 2 refused, with ``REFUSED: <type>: <reason>`` on stderr."""
    args = build_parser().parse_args(argv)
    try:
        repo_root = Path(args.repo_root).resolve()
        code = assert_code_under_root(repo_root)
        git = git_block()
        roots = Roots(repo_root=repo_root, output_root=Path(args.output_root).resolve())
        if args.command == "build":
            if not args.out:
                raise ValueError("build needs --out, the artifact's path (docs/data/p5_2_statements.json)")
            if git.get("dirty") is not False:
                raise ValueError(f"the tree is dirty or unreadable (git dirty = {git.get('dirty')!r}): the artifact "
                                 "records a commit, and a dirty tree's commit is not the code that ran")
        inputs = load_inputs(roots, pins=PINS)
        if args.command == "check":
            files = inputs.provenance["inputs"]
            print(f"check: every input verified at its anchor ({len(files)} files, and the rows below)")
            for entry in files:
                print(f"  {entry['sha256']}  {entry['file']}  [{entry['anchor']}]")
            campaign = inputs.provenance["p8_4b_campaign"]
            print(f"P8.4b campaign: declared_cells_sha256 {campaign['declared_cells_sha256']}, {campaign['n_cells']} "
                  f"cells, engine seed {campaign['engine_seed']}")
            for name in ("p8_4b_rows", "p5_2b_corrected_rows"):
                block = inputs.provenance[name]
                print(f"{name}: {block['n']} files read, digest {block['sha256']}")
            print(f"git: commit {git.get('commit')}, dirty {git.get('dirty')}"
                  + ("" if git.get("dirty") is False else "  (build would refuse a dirty tree)"))
            for entry in code:
                print(f"code: {entry['sha256']}  {entry['path']}")
            if args.out:
                print(f"build would write {assert_out_path(args.out, roots, inputs)}")
            return 0
        out = assert_out_path(args.out, roots, inputs)
        data = serialize(build_artifact(inputs, git=git, code=code, record=RECORD))
        if assert_code_under_root(repo_root) != code:
            raise ValueError("the code under --repo-root changed while the artifact was built")
        write_artifact(out, data)
        print(f"wrote {out}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}")
        return 0
    except (ValueError, KeyError, OSError) as error:
        print(f"REFUSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
