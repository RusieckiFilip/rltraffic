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

#: The modules this one imports by name (Amendment A, A4.1).  ``build`` and ``check`` refuse unless each, and every other
#: loaded ``offline`` module, resolves under ``--repo-root``; ``provenance.code`` records each with its sha256.
CODE_MODULES: tuple[str, ...] = (
    "offline",
    "offline.tier_statements",
    "offline.att_rederivation",
    "offline.compute_latency",
    "offline.dt_gate",
    "offline.iql_correction",
    "offline.offline_baselines",
    "offline.spatial_mixing",
    "offline.tier_sweep",
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
ANCHOR_CELL = "p5_1:random@grid4x4_mappo1000"
REROLLED_ANCHOR_CELL = "p5_2:random@mappo1000"


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
    raise NotImplementedError("P5.2c C1 skeleton: make_books")


def episodes_of(cell: Cell, definition: str) -> list[EpisodeResult]:
    """The cell's episodes under *definition*, by ``iql_correction.episodes_from_rows`` (refuses a duplicate key)."""
    raise NotImplementedError("P5.2c C1 skeleton: episodes_of")


# ----------------------------------------------------------------------
# Inputs, each read at its anchor, all before anything is written
# ----------------------------------------------------------------------


def read_manifest(output_root: str | Path, name: str, pin: str) -> dict[str, str]:
    """``output/<name>`` parsed by ``tier_sweep.parse_sha256sums``, refused unless the file is at *pin*."""
    raise NotImplementedError("P5.2c C1 skeleton: read_manifest")


def read_listed_json(output_root: str | Path, relative: str, sums: Mapping[str, str], *, manifest_name: str
                     ) -> dict[str, Any]:
    """A JSON record (``relative`` to ``output/``), refused unless its sha256 is its line in *manifest_name*.
    ``iql_correction.read_p5_2_json`` restated for any manifest (Amendment A, A2 Q3)."""
    raise NotImplementedError("P5.2c C1 skeleton: read_listed_json")


def read_pinned_json(repo_root: str | Path, relative: str, pin: str) -> dict[str, Any]:
    """A committed document under ``repo_root``, refused unless its blob sha256 is *pin*."""
    raise NotImplementedError("P5.2c C1 skeleton: read_pinned_json")


def assert_note_means(output_root: str | Path, campaign_cells: Sequence[str], note: Mapping[str, Any], *, tier: str,
                      arms: Mapping[str, str]) -> dict[str, Any]:
    """``iql_correction.assert_c1_note_means`` restated for any tier (plan F13): the six means under both definitions,
    by the note's own route (``statistics.mean`` over each arm's P8.4b cell files listed in *campaign_cells*), refused
    unless each equals ``note["tiers"]["grid4x4/<tier>"]``'s.  *arms* maps each method to its P8.4b arm string."""
    raise NotImplementedError("P5.2c C1 skeleton: assert_note_means")


def assert_anchor_identity(p5_1: Mapping[str, Any], p5_2: Mapping[str, Any], p5_1_rows: Sequence[Mapping[str, Any]],
                           p5_2_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """D12's shared anchor: refused unless P5.2's re-rolled random cell is episode-identical to P5.1's
    (``tier_sweep.assert_cells_identical``, 500 episodes) and P8.4b's rows of the two arms agree on ``att_ours``,
    ``att_engine`` and every companion."""
    raise NotImplementedError("P5.2c C1 skeleton: assert_anchor_identity")


def rows_digest(pairs: Iterable[tuple[str, str]]) -> str:
    """Amendment A, A4.3: one sha256 over sorted ``(path relative to --output-root, file sha256)`` pairs, taken over the
    text ``"<file sha256>  <path>\\n"`` per pair in path order -- the ``sha256sum`` line format."""
    raise NotImplementedError("P5.2c C1 skeleton: rows_digest")


def assert_code_under_root(repo_root: str | Path) -> list[dict[str, Any]]:
    """Amendment A, A4.1: refuse unless every module of :data:`CODE_MODULES` and every loaded ``offline`` module
    resolves to ``<repo_root>/offline/...``; return each :data:`CODE_MODULES` entry's path relative to the root and its
    sha256."""
    raise NotImplementedError("P5.2c C1 skeleton: assert_code_under_root")


def git_block() -> dict[str, Any]:
    """``compute_latency._git_provenance()``: the commit, the dirty flag (tracked files) and the code root."""
    raise NotImplementedError("P5.2c C1 skeleton: git_block")


@dataclass(frozen=True)
class Inputs:
    """Everything the artifact is built from, verified.  ``cells`` by id; ``provenance`` every file read with its anchor
    and A4.3's two digests; ``records`` the committed records the comparison reads."""

    cells: Mapping[str, Cell]
    provenance: Mapping[str, Any]
    records: Mapping[str, Any]


def load_inputs(roots: Roots, *, pins: Pins = PINS) -> Inputs:
    """Every input at its anchor (plan section 6.2), refusing before anything is written."""
    raise NotImplementedError("P5.2c C1 skeleton: load_inputs")


# ----------------------------------------------------------------------
# Outcomes and contrasts
# ----------------------------------------------------------------------


def interval_outcome(low: float, high: float, *, predicted: str) -> str:
    """HELD / FAILED / NOT RESOLVED for a quantity predicted ``negative`` or ``positive``, by
    ``spatial_mixing._outcome_from_interval`` (on ``(-high, -low)`` for a positive one).  A CI touching zero, and an
    exact zero (CI [0, 0]), is NOT RESOLVED."""
    raise NotImplementedError("P5.2c C1 skeleton: interval_outcome")


def conjunction(outcomes: Sequence[str]) -> str:
    """``BRIEF_45`` section 3.3: HELD iff every tier is HELD, FAILED iff any tier is FAILED, NOT RESOLVED otherwise."""
    raise NotImplementedError("P5.2c C1 skeleton: conjunction")


def sign_of(value: float) -> str:
    """``negative`` / ``zero`` / ``positive`` of a point estimate."""
    raise NotImplementedError("P5.2c C1 skeleton: sign_of")


def contrast(left: Cell, right: Cell, *, definition: str, tier: str) -> dict[str, Any]:
    """``left - right`` on the paired unit (module docstring), refused unless the two cells cover the identical
    ``(seed, draw)`` grid.  Carries the CI and its width, ``exactly_zero``, the point estimate's sign,
    ``tier_sweep.per_seed_block`` (D9), ``att_rederivation.contrast_discriminability`` and D8's rank-biserial block from
    ``offline_baselines.paired_comparison`` -- refused unless that call's ``mean_difference``, ``ci95_low`` and
    ``ci95_high`` equal this block's under ``==`` (Amendment A, A2 Q8)."""
    raise NotImplementedError("P5.2c C1 skeleton: contrast")


def composite_contrast(left: tuple[Cell, Cell], right: tuple[Cell, Cell], *, definition: str, tier: str
                       ) -> dict[str, Any]:
    """``I = (left[0] - left[1]) - (right[0] - right[1])`` per draw from the four cells' per-draw means, its CI by
    ``mean_ci95``, the counts of draws where it is negative / zero / positive, and the per-seed values
    ``(mean_s(left[0]) - mean_s(left[1])) - (mean_s(right[0]) - mean_s(right[1]))`` (``BRIEF_45`` section 3.3).  A
    difference of two differences, so no rank-biserial, and the block says so."""
    raise NotImplementedError("P5.2c C1 skeleton: composite_contrast")


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
    raise NotImplementedError("P5.2c C1 skeleton: cells_block")


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
    raise NotImplementedError("P5.2c C1 skeleton: normalise")


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
