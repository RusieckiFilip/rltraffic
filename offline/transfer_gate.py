"""P7.0 -- the CityFlow to SUMO dynamics-shift gate.

`docs/plans/p7.0.md` sections 5 and 6 are the registered specification; this module is
its implementation and adds no decision of its own.  **P7.0 reports; it does not rule**
(`BRIEF_04` section 4.4, not superseded by `BRIEF_21`).

Artifact format version
-----------------------
**v1.0** (:data:`GATE_FORMAT_VERSION`), written to ``docs/data/p7_0_gate.json``.

What is compared, and the alignment convention that makes it meaningful
-----------------------------------------------------------------------
⚠️ **Lane features are aligned by LANE ID, never by position, and this is not
defensive coding.**  Measured 2026-08-16: ``utils/cityflow_utils.py`` appends an
intersection's ``incoming_lanes`` in roadLinks/laneLinks discovery order, whose first
entry for hangzhou's ``intersection_1_1`` is ``road_0_1_0_1``; ``utils/sumo_utils.py``
stores ``sorted(incoming_lanes)``, whose first entry is ``road_0_1_0_0``.  **A
positional comparison would silently compare different lanes in the two backends** --
`DEFERRED` 23's failure class, one level below the intersection keying that row is
about.  The corpus's own lane arrays are keyed by a sorted ``lane_ids`` vector, so
reading them by id is both correct and cheap; the per-intersection ``state`` vector is
the artifact that carries the mismatch, and the gate reports that rather than using it.

Both statistics are exact rationals
-----------------------------------
The two compared arrays are integer counts, so the empirical pmf is exact and no bin
width has to be chosen -- one fewer researcher degree of freedom.  Both statistics
then reduce to an **integer numerator over n * m**::

    D   = max_v abs(cx_v * m - cy_v * n) / (n * m)
    OVL = sum_v min(cx_v * m, cy_v * n) / (n * m)

so the double-computation in ``tests/test_transfer_gate.py`` asserts ``==`` on Python
ints and never on floats.  A tolerance on a load-bearing quantity is a tolerance on
the answer.

No raw travel time crosses the backend boundary
-----------------------------------------------
`PREREGISTRATION` section 3.4 forbids it on the platform thesis's own evidence.  ATT is
recorded per (backend, policy) so the *within-backend* arithmetic is checkable, and
every cross-backend statement is about ``rho`` or about a distribution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "GATE_FORMAT_VERSION",
    "LANE_ARRAYS",
    "CRITERION_SCALES",
    "OVL_PATHOLOGICAL",
    "OVL_COMPARABLE",
    "KS_LARGE",
    "KS_MAX_COMPARABLE",
    "KS_LARGE_COUNT_PATHOLOGICAL",
    "KS_LARGE_COUNT_COMPARABLE",
    "RHO_COMPARABLE_FACTOR",
    "RHO_PATHOLOGICAL_FACTOR",
    "FeatureComparison",
    "CriterionRow",
    "BranchVerdict",
    "ks_statistic_exact",
    "ks_statistic",
    "overlap_coefficient_exact",
    "overlap_coefficient",
    "rho",
    "lane_feature_samples",
    "compare_lane_features",
    "evaluate_branch",
    "metric_set_independence",
    "intersection_enumeration",
    "main",
]

GATE_FORMAT_VERSION = "1.0"

#: The two backend-neutral lane-derived observation arrays the corpus stores.
LANE_ARRAYS: tuple[str, ...] = ("lane_vehicle_count", "lane_waiting_vehicle_count")

# --- docs/plans/p7.0.md section 6.2, declared before any number existed. ---
# These are conventional small/large reading points for the two statistics. They are
# CHOSEN, not derived, and they are written down in a committed plan precisely so the
# choice could not be made while looking at the table.
OVL_PATHOLOGICAL = 0.30
OVL_COMPARABLE = 0.50
KS_LARGE = 0.50
KS_MAX_COMPARABLE = 0.70
KS_LARGE_COUNT_PATHOLOGICAL = 9
KS_LARGE_COUNT_COMPARABLE = 4
RHO_COMPARABLE_FACTOR = 0.5
RHO_PATHOLOGICAL_FACTOR = 2.0

#: docs/plans/p7.0.md section 6.3 -- the per-criterion scale used to make margins
#: comparable across criteria that carry different units.  A READING AID ONLY: the
#: branch is decided by section 6.2's raw criteria and by nothing else.
CRITERION_SCALES: Mapping[str, str] = {
    "B1": "delta_cityflow (the training-domain anchor span)",
    "B2": "1.0 (the overlap coefficient already lies on [0, 1])",
    "B3": "the number of features",
    "B4": "M = max(1.0, abs(rho_cityflow_random))",
    "A1": "delta_cityflow (the training-domain anchor span)",
    "A2": "1.0 (the overlap coefficient already lies on [0, 1])",
    "A3max": "1.0 (the KS statistic already lies on [0, 1])",
    "A3count": "the number of features",
    "A4": "M = max(1.0, abs(rho_cityflow_random))",
}


@dataclass(frozen=True)
class FeatureComparison:
    """One row of the per-feature table. **Never pooled with another row.**

    Pooling is what would hide a single catastrophic feature inside an average, which
    is why ``BRIEF_21`` section 4 forbids it and why criterion B2 is a minimum.
    """

    feature: str
    array: str
    lane_id: str
    n_cityflow: int
    n_sumo: int
    ks_statistic: float
    overlap_coefficient: float
    mean_cityflow: float
    mean_sumo: float


@dataclass(frozen=True)
class CriterionRow:
    """One declared criterion, its margin, and whether it points away from transfer.

    ``fired`` means **the criterion points AWAY from comparability**: for a ``B`` row
    the pathology holds, for an ``A`` row the comparability requirement failed.
    ``comparison`` is the operator under which the criterion is SATISFIED, so ``fired``
    is recomputable from ``statistic``, ``threshold`` and ``comparison`` alone.

    ``signed_distance`` is oriented so that a negative value means ``fired``.  ⚠️ The
    two criteria whose declared inequality fires ON the boundary (``B1``: ``delta <=
    0``; ``B3``: ``count >= 9``) and ``A1``, which is ``B1``'s complement, are exact at
    a statistic equal to their threshold: there ``signed_distance`` is ``0.0`` while
    ``fired`` is ``True``.  ``comparison`` is the field that stays exact everywhere.
    """

    criterion: str
    branch: str
    statistic: float | None
    threshold: float | None
    comparison: str
    signed_distance: float | None
    fired: bool
    scale: float | None
    relative_distance: float | None
    detail: str


@dataclass(frozen=True)
class BranchVerdict:
    """The gate's reading. It names a branch; it does not act on one."""

    branch: str
    firing_criteria: tuple[str, ...]
    failed_a_criteria: tuple[str, ...]
    nearest_non_firing: str | None
    rows: tuple[CriterionRow, ...]


# ----------------------------------------------------------------------
# Distribution statistics -- exact rationals over integer counts
# ----------------------------------------------------------------------


def _as_integer_sample(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        raise ValueError(f"{name} is an empty sample; a distribution needs observations")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"{name} has dtype {array.dtype}; these statistics are defined on the "
            "exact integer counts the corpus stores, so no binning choice is needed"
        )
    return array.reshape(-1)


def ks_statistic_exact(x: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    """``(numerator, n * m)`` with ``D == numerator / (n * m)``, in exact integers."""
    xs = _as_integer_sample(x, "x")
    ys = _as_integer_sample(y, "y")
    n, m = int(xs.size), int(ys.size)
    support = np.unique(np.concatenate((xs, ys)))
    cx = np.searchsorted(np.sort(xs), support, side="right").astype(object)
    cy = np.searchsorted(np.sort(ys), support, side="right").astype(object)
    numerator = max(abs(int(a) * m - int(b) * n) for a, b in zip(cx, cy))
    return int(numerator), n * m


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic ``max_t abs(F_x(t) - F_y(t))``."""
    numerator, denominator = ks_statistic_exact(x, y)
    return numerator / denominator


def overlap_coefficient_exact(x: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    """``(numerator, n * m)`` with ``OVL == numerator / (n * m)``, in exact integers."""
    xs = _as_integer_sample(x, "x")
    ys = _as_integer_sample(y, "y")
    n, m = int(xs.size), int(ys.size)
    cx = Counter(int(v) for v in xs)
    cy = Counter(int(v) for v in ys)
    numerator = sum(min(cx[v] * m, cy[v] * n) for v in set(cx) | set(cy))
    return int(numerator), n * m


def overlap_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Histogram intersection on the exact integer support; equals ``1 - TV``."""
    numerator, denominator = overlap_coefficient_exact(x, y)
    return numerator / denominator


def rho(att_fixedtime: float, att_policy: float, att_maxpressure: float) -> float:
    """`PREREGISTRATION` section 3.4's within-backend normalised return.

    fixed-time = 0, MaxPressure = 1, **computed inside one backend**.  All three
    arguments must come from the same backend; this function cannot check that, and
    the caller that mixes them produces the one number section 3.4 forbids.
    """
    span = float(att_fixedtime) - float(att_maxpressure)
    if span == 0.0:
        raise ValueError(
            "the anchor span is zero (fixed-time equals MaxPressure), so the "
            "normalisation of PREREGISTRATION section 3.4 is undefined in this backend"
        )
    return (float(att_fixedtime) - float(att_policy)) / span


# ----------------------------------------------------------------------
# Per-feature comparison, aligned by lane id
# ----------------------------------------------------------------------


def lane_feature_samples(
    episodes: Sequence[Any],
    array_name: str,
    lane_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    """Pool each lane's column across ``episodes``, **selected by lane id**.

    Every row of every episode is included, row 0 (the reset state) as well: the
    marginal the gate compares is over the whole observation stream, and dropping the
    reset row would be a silent choice about which states count.
    """
    if array_name not in LANE_ARRAYS:
        raise ValueError(f"{array_name!r} is not one of {list(LANE_ARRAYS)}")
    if not episodes:
        raise ValueError("no episodes were supplied")

    collected: dict[str, list[np.ndarray]] = {str(lid): [] for lid in lane_ids}
    for episode in episodes:
        index = {str(lid): i for i, lid in enumerate(episode.lane_ids)}
        array = np.asarray(getattr(episode, array_name))
        for lane_id in collected:
            if lane_id not in index:
                raise KeyError(
                    f"lane {lane_id!r} is not present in this episode's lane_ids; "
                    "the gate aligns by lane id and refuses to fall back to a "
                    "positional read"
                )
            collected[lane_id].append(array[:, index[lane_id]])
    return {lane_id: np.concatenate(cols) for lane_id, cols in collected.items()}


def compare_lane_features(
    cityflow_episodes: Sequence[Any],
    sumo_episodes: Sequence[Any],
    lane_ids: Sequence[str],
) -> list[FeatureComparison]:
    """The per-feature table: one row per (lane array, lane id). Never pooled."""
    rows: list[FeatureComparison] = []
    for array_name in LANE_ARRAYS:
        left = lane_feature_samples(cityflow_episodes, array_name, lane_ids)
        right = lane_feature_samples(sumo_episodes, array_name, lane_ids)
        for lane_id in lane_ids:
            x, y = left[str(lane_id)], right[str(lane_id)]
            rows.append(
                FeatureComparison(
                    feature=f"{array_name}@{lane_id}",
                    array=array_name,
                    lane_id=str(lane_id),
                    n_cityflow=int(x.size),
                    n_sumo=int(y.size),
                    ks_statistic=ks_statistic(x, y),
                    overlap_coefficient=overlap_coefficient(x, y),
                    mean_cityflow=float(np.mean(x)),
                    mean_sumo=float(np.mean(y)),
                )
            )
    return rows


# ----------------------------------------------------------------------
# The registered branch rule
# ----------------------------------------------------------------------

_COMPARISONS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def _row(
    criterion: str,
    branch: str,
    statistic: float,
    threshold: float,
    comparison: str,
    scale: float | None,
    detail: str,
) -> CriterionRow:
    """Build one criterion row. ``comparison`` is the operator that SATISFIES it."""
    satisfied = _COMPARISONS[comparison](statistic, threshold)
    # Oriented so a negative distance means the criterion points away from transfer.
    if comparison in (">", ">="):
        signed = float(statistic) - float(threshold)
    else:
        signed = float(threshold) - float(statistic)
    relative = None if not scale else signed / float(scale)
    return CriterionRow(
        criterion=criterion,
        branch=branch,
        statistic=float(statistic),
        threshold=float(threshold),
        comparison=comparison,
        signed_distance=signed,
        fired=not satisfied,
        scale=None if scale is None else float(scale),
        relative_distance=relative,
        detail=detail,
    )


def evaluate_branch(
    features: Sequence[FeatureComparison],
    delta_cityflow: float,
    delta_sumo: float,
    rho_cityflow_random: float,
    rho_sumo_random: float,
) -> BranchVerdict:
    """Apply ``docs/plans/p7.0.md`` section 6.2's registered rule, and only that rule.

    Total and mutually exclusive by construction: ``A2``/``B2`` are disjoint,
    ``A3count``/``B3`` are disjoint, ``A1`` is the complement of ``B1``, and
    ``A4``/``B4`` are disjoint.  Branch B if any B criterion fires; branch A if none
    does and every A criterion holds; branch C otherwise.
    """
    if not features:
        raise ValueError("the branch rule needs at least one feature comparison row")

    n_features = len(features)
    min_ovl = min(row.overlap_coefficient for row in features)
    max_ks = max(row.ks_statistic for row in features)
    n_large_ks = sum(1 for row in features if row.ks_statistic > KS_LARGE)
    min_delta = min(float(delta_cityflow), float(delta_sumo))

    scale_m = max(1.0, abs(float(rho_cityflow_random)))
    rho_gap = abs(float(rho_sumo_random) - float(rho_cityflow_random))
    same_sign = (float(rho_sumo_random) >= 0.0) == (float(rho_cityflow_random) >= 0.0)
    delta_scale = delta_cityflow if delta_cityflow > 0.0 else None

    rows: list[CriterionRow] = [
        _row(
            "B1", "B", min_delta, 0.0, ">", delta_scale,
            f"anchor span: cityflow={delta_cityflow:.6f} sumo={delta_sumo:.6f}; "
            "a non-positive span makes PREREGISTRATION 3.4's normalisation ill-posed",
        ),
        _row(
            "B2", "B", min_ovl, OVL_PATHOLOGICAL, ">=", 1.0,
            f"minimum overlap coefficient over {n_features} features; a minimum and "
            "never a mean, so one catastrophic feature cannot hide in an average",
        ),
        _row(
            "B3", "B", float(n_large_ks), float(KS_LARGE_COUNT_PATHOLOGICAL), "<",
            float(n_features),
            f"{n_large_ks} of {n_features} features exceed a KS statistic of {KS_LARGE}",
        ),
    ]

    if not same_sign:
        rows.append(
            CriterionRow(
                criterion="B4", branch="B", statistic=None, threshold=None,
                comparison="sign", signed_distance=None, fired=True, scale=None,
                relative_distance=None,
                detail=(
                    "rho for random has opposite sign across the backends "
                    f"(cityflow={rho_cityflow_random:.6f}, sumo={rho_sumo_random:.6f}); "
                    "a sign flip has no continuous margin and none is invented"
                ),
            )
        )
    else:
        rows.append(
            _row(
                "B4", "B", rho_gap, RHO_PATHOLOGICAL_FACTOR * scale_m, "<=", scale_m,
                f"rho gap for random: cityflow={rho_cityflow_random:.6f} "
                f"sumo={rho_sumo_random:.6f}, same sign",
            )
        )

    rows.append(
        _row(
            "A1", "A", min_delta, 0.0, ">", delta_scale,
            "both anchor spans must be positive for rho to be well-posed",
        )
    )
    rows.append(
        _row(
            "A2", "A", min_ovl, OVL_COMPARABLE, ">=", 1.0,
            f"minimum overlap coefficient over {n_features} features",
        )
    )
    rows.append(
        _row(
            "A3max", "A", max_ks, KS_MAX_COMPARABLE, "<=", 1.0,
            f"largest KS statistic over {n_features} features",
        )
    )
    rows.append(
        _row(
            "A3count", "A", float(n_large_ks), float(KS_LARGE_COUNT_COMPARABLE), "<=",
            float(n_features),
            f"{n_large_ks} of {n_features} features exceed a KS statistic of {KS_LARGE}",
        )
    )
    if not same_sign:
        rows.append(
            CriterionRow(
                criterion="A4", branch="A", statistic=None, threshold=None,
                comparison="sign", signed_distance=None, fired=True, scale=None,
                relative_distance=None,
                detail=(
                    "rho for random has opposite sign across the backends, so the "
                    "same-sign half of A4 fails before its magnitude half is reached"
                ),
            )
        )
    else:
        rows.append(
            _row(
                "A4", "A", rho_gap, RHO_COMPARABLE_FACTOR * scale_m, "<=", scale_m,
                f"rho gap for random: cityflow={rho_cityflow_random:.6f} "
                f"sumo={rho_sumo_random:.6f}, same sign",
            )
        )

    firing = tuple(r.criterion for r in rows if r.branch == "B" and r.fired)
    failed_a = tuple(r.criterion for r in rows if r.branch == "A" and r.fired)
    if firing:
        branch = "B"
    elif not failed_a:
        branch = "A"
    else:
        branch = "C"

    candidates = [
        r for r in rows if not r.fired and r.relative_distance is not None
    ]
    nearest = (
        min(candidates, key=lambda r: abs(float(r.relative_distance))).criterion
        if candidates
        else None
    )
    return BranchVerdict(
        branch=branch,
        firing_criteria=firing,
        failed_a_criteria=failed_a,
        nearest_non_firing=nearest,
        rows=tuple(rows),
    )


# ----------------------------------------------------------------------
# Lane-convention diagnostic -- IS A LANE ID THE SAME PHYSICAL LANE IN BOTH
# BACKENDS?  Added 2026-08-16, after the registered run, and it selects nothing.
# ----------------------------------------------------------------------
#
# ⚠️ WHY THIS EXISTS, stated so it is never mistaken for part of the gate.
# ``docs/plans/p7.0.md`` section 5.3 registered a per-feature comparison "aligned by
# lane id".  That rests on an unstated premise -- that a lane id denotes the same
# PHYSICAL lane in both backends -- and the premise is false on this scenario.  Read
# from the two scenario files, which is a structural fact and not an outcome:
# CityFlow's ``roadnet.json`` gives ``road_X_0`` the ``turn_left`` roadLink and
# ``road_X_1`` the ``go_straight`` one, while SUMO's ``.net.xml`` gives ``road_X_0``
# ``dir="s"`` and ``road_X_1`` ``dir="l"``.  SUMO indexes lanes right-to-left, CityFlow
# left-to-right, so the two conventions are reversed and the registered table compared
# a left-turn lane against a through lane.
#
# **These functions produce a DIAGNOSTIC. They do not feed evaluate_branch, and no
# branch may be re-derived from them without the coordinator re-registering section
# 5.3.** Choosing a correspondence after seeing a verdict is the researcher degree of
# freedom the registration exists to remove; the correspondence below is justified by
# the ``type``/``dir`` attributes alone, which were read before any corrected number
# was computed.

_CITYFLOW_TO_SUMO_TURN = {
    "go_straight": "s",
    "turn_left": "l",
    "turn_right": "r",
}


def cityflow_lane_turns(roadnet_path: str | Path, ix_id: str) -> dict[str, frozenset[str]]:
    """Movements each incoming lane serves, from a CityFlow ``roadnet.json``."""
    data = json.loads(Path(roadnet_path).read_bytes())
    matches = [ix for ix in data.get("intersections", []) if ix.get("id") == ix_id]
    if len(matches) != 1:
        raise ValueError(f"{ix_id!r} matches {len(matches)} intersections in {roadnet_path}")
    turns: dict[str, set[str]] = {}
    for link in matches[0].get("roadLinks", []):
        kind = str(link.get("type"))
        if kind not in _CITYFLOW_TO_SUMO_TURN:
            raise ValueError(f"unknown CityFlow roadLink type {kind!r}")
        for lane_link in link.get("laneLinks", []):
            lane_id = f"{link['startRoad']}_{lane_link['startLaneIndex']}"
            turns.setdefault(lane_id, set()).add(_CITYFLOW_TO_SUMO_TURN[kind])
    return {lane_id: frozenset(kinds) for lane_id, kinds in turns.items()}


def sumo_lane_turns(net_xml_path: str | Path) -> dict[str, frozenset[str]]:
    """Movements each lane serves, from a SUMO ``.net.xml``'s ``<connection>`` set."""
    import xml.etree.ElementTree as ET

    root = ET.parse(net_xml_path).getroot()
    turns: dict[str, set[str]] = {}
    for conn in root.findall("connection"):
        edge = conn.get("from")
        if edge is None or edge.startswith(":"):
            continue
        lane_id = f"{edge}_{int(str(conn.get('fromLane')))}"
        turns.setdefault(lane_id, set()).add(str(conn.get("dir")))
    return {lane_id: frozenset(kinds) for lane_id, kinds in turns.items()}


def lane_semantic_correspondence(
    cityflow_turns: Mapping[str, frozenset[str]],
    sumo_turns: Mapping[str, frozenset[str]],
    lane_ids: Sequence[str],
) -> dict[str, str]:
    """Map each CityFlow lane to the SUMO lane on the same road serving the same turns.

    Refuses anything but a unique match: a road where the movement sets do not pair up
    one-to-one is a genuine topology difference between the two scenario files, not a
    numbering convention, and it must be reported rather than resolved by a heuristic.
    """
    by_road: dict[str, list[str]] = {}
    for lane_id in sumo_turns:
        by_road.setdefault(lane_id.rsplit("_", 1)[0], []).append(lane_id)

    mapping: dict[str, str] = {}
    for lane_id in lane_ids:
        road = str(lane_id).rsplit("_", 1)[0]
        if lane_id not in cityflow_turns:
            raise KeyError(f"{lane_id!r} serves no CityFlow roadLink")
        wanted = cityflow_turns[lane_id]
        candidates = [c for c in by_road.get(road, []) if sumo_turns[c] == wanted]
        if len(candidates) != 1:
            raise ValueError(
                f"{lane_id!r} serving {sorted(wanted)} matches {len(candidates)} SUMO "
                f"lanes on road {road!r}; the two scenario files disagree on topology "
                "rather than on lane numbering, which is a finding and not something "
                "to resolve with a heuristic"
            )
        mapping[str(lane_id)] = candidates[0]
    return mapping


def road_level_samples(
    episodes: Sequence[Any],
    array_name: str,
    lane_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    """Sum each road's lane columns. Invariant to the lane-numbering convention."""
    per_lane = lane_feature_samples(episodes, array_name, lane_ids)
    roads: dict[str, np.ndarray] = {}
    for lane_id, column in per_lane.items():
        road = lane_id.rsplit("_", 1)[0]
        roads[road] = column if road not in roads else roads[road] + column
    return roads


def diagnose_lane_convention(
    cityflow_episodes: Sequence[Any],
    sumo_episodes: Sequence[Any],
    lane_ids: Sequence[str],
    roadnet_path: str | Path,
    net_xml_path: str | Path,
    ix_id: str,
) -> dict[str, Any]:
    """Two convention-independent readings of the same episodes. Selects nothing.

    ``road_level`` aggregates each road's lanes, so it is invariant to the numbering
    convention outright.  ``semantic_per_feature`` re-pairs the lanes by the movements
    they serve.  Both are labelled diagnostics; the registered table stands unchanged.
    """
    cf_turns = cityflow_lane_turns(roadnet_path, ix_id)
    su_turns = sumo_lane_turns(net_xml_path)
    mapping = lane_semantic_correspondence(cf_turns, su_turns, lane_ids)
    reversed_ids = sorted(k for k, v in mapping.items() if k != v)

    road_rows: list[dict[str, Any]] = []
    for array_name in LANE_ARRAYS:
        left = road_level_samples(cityflow_episodes, array_name, lane_ids)
        right = road_level_samples(sumo_episodes, array_name, lane_ids)
        for road in sorted(left):
            x, y = left[road], right[road]
            road_rows.append(
                {
                    "feature": f"{array_name}@{road}",
                    "ks_statistic": ks_statistic(x, y),
                    "overlap_coefficient": overlap_coefficient(x, y),
                    "mean_cityflow": float(np.mean(x)),
                    "mean_sumo": float(np.mean(y)),
                }
            )

    semantic_rows: list[dict[str, Any]] = []
    for array_name in LANE_ARRAYS:
        left = lane_feature_samples(cityflow_episodes, array_name, lane_ids)
        right = lane_feature_samples(
            sumo_episodes, array_name, sorted(set(mapping.values()))
        )
        for lane_id in lane_ids:
            x, y = left[str(lane_id)], right[mapping[str(lane_id)]]
            semantic_rows.append(
                {
                    "feature": f"{array_name}@{lane_id}",
                    "cityflow_lane": str(lane_id),
                    "sumo_lane": mapping[str(lane_id)],
                    "turns": sorted(cf_turns[str(lane_id)]),
                    "ks_statistic": ks_statistic(x, y),
                    "overlap_coefficient": overlap_coefficient(x, y),
                    "mean_cityflow": float(np.mean(x)),
                    "mean_sumo": float(np.mean(y)),
                }
            )

    return {
        "is_a_diagnostic_not_a_gate_input": True,
        "cityflow_lane_turns": {k: sorted(v) for k, v in sorted(cf_turns.items())},
        "sumo_lane_turns": {
            k: sorted(v) for k, v in sorted(su_turns.items()) if k in set(mapping.values())
        },
        "semantic_correspondence": mapping,
        "identity_correspondence": all(k == v for k, v in mapping.items()),
        "lanes_whose_id_denotes_a_different_physical_lane": reversed_ids,
        "road_level": road_rows,
        "semantic_per_feature": semantic_rows,
    }


# ----------------------------------------------------------------------
# DEFERRED 18 -- is info["average_travel_time"] metric-set independent on SUMO?
# ----------------------------------------------------------------------

_DEFAULT_METRICS_A = ("average_travel_time",)
_DEFAULT_METRICS_B = (
    "average_travel_time",
    "count_of_vehicles_completing_journey",
    "waiting_time_all_vehicles_for_the_last_time_step_in_simulation",
)


def _att_sequence(
    sumocfg_path: str | Path,
    metrics: Sequence[str],
    steps: int,
    seed: int,
    policy: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Roll one SUMO episode and return its per-step ATT plus the live metric keys."""
    from algorithms.max_pressure import MaxPressureAgent
    from envs.sumo_env import SumoEnv

    env = SumoEnv(
        sumocfg_path=str(sumocfg_path),
        max_steps=int(steps),
        delta_time=10,
        global_reward_fn="average_travel_time",
        metrics=list(metrics),
    )
    try:
        info = env.reset(seed=int(seed))
        keys = tuple(sorted(info["metrics"].keys()))
        values = [float(info["average_travel_time"])]
        agent = MaxPressureAgent(env)
        n_actions = [int(len(ix.phase_durations)) for ix in env.intersections]
        rng = np.random.default_rng(int(seed))
        for _ in range(int(steps)):
            if policy == "maxpressure":
                action = agent.act(info)
            elif policy == "random":
                action = np.array(
                    [
                        int(
                            rng.choice(
                                info["intersections"][ix.id]["avail_actions"]
                            )
                        )
                        for ix in env.intersections
                    ],
                    dtype=np.int64,
                )
            else:  # pragma: no cover - guarded by the caller
                raise ValueError(f"unsupported control policy {policy!r}")
            _reward, _terminated, _truncated, info = env.step(action)
            values.append(float(info["average_travel_time"]))
        _ = n_actions
        return np.asarray(values, dtype=np.float64), keys
    finally:
        env.close()


def metric_set_independence(
    sumocfg_path: str | Path,
    steps: int,
    seed: int,
    metrics_a: Sequence[str] = _DEFAULT_METRICS_A,
    metrics_b: Sequence[str] = _DEFAULT_METRICS_B,
) -> dict[str, Any]:
    """`DEFERRED` 18 on SUMO, with both of its controls.

    Two halves, because the question has a structural half the empirical one cannot
    answer.  **Structural:** ``envs/sumo_env.py:119-125`` unions
    ``average_travel_time`` into ``_metric_names`` unconditionally, so through
    ``SumoEnv`` the metric is never absent and ``metrics/sumo.py:77``'s
    ``_track_completed`` gate is always open -- reported as ``att_always_requested``
    and as the honest scope limit of this check.  **Empirical:** the paired 1-metric
    versus 3-metric rollout, compared with ``np.array_equal``.

    Two controls, without which the verdict would be worthless:
    ``metrics are identical`` is refused outright (a check that compares an env with
    itself always passes), and a third rollout under a different policy must make
    ``np.array_equal`` return ``False``, proving the comparison can fail.
    """
    keys_a, keys_b = tuple(sorted(set(metrics_a))), tuple(sorted(set(metrics_b)))
    if keys_a == keys_b:
        raise ValueError(
            "the two requested metric sets are identical, so this check would compare "
            "an env with itself and pass regardless of the answer"
        )

    att_a, live_a = _att_sequence(sumocfg_path, metrics_a, steps, seed, "maxpressure")
    att_b, live_b = _att_sequence(sumocfg_path, metrics_b, steps, seed, "maxpressure")
    att_control, _ = _att_sequence(sumocfg_path, metrics_a, steps, seed, "random")

    identical = bool(np.array_equal(att_a, att_b))
    return {
        "metrics_a": list(keys_a),
        "metrics_b": list(keys_b),
        "live_metric_keys_a": list(live_a),
        "live_metric_keys_b": list(live_b),
        "att_always_requested": bool(
            "average_travel_time" in live_a and "average_travel_time" in live_b
        ),
        "n_rows": int(att_a.size),
        "identical": identical,
        "max_abs_difference": float(np.max(np.abs(att_a - att_b))),
        "att_horizon_a": float(att_a[-1]),
        "att_horizon_b": float(att_b[-1]),
        "control_policy": "random",
        "control_identical": bool(np.array_equal(att_a, att_control)),
        "control_att_horizon": float(att_control[-1]),
    }


# ----------------------------------------------------------------------
# DEFERRED 23 -- intersection (and lane) enumeration order per backend
# ----------------------------------------------------------------------


def intersection_enumeration(backend: str, config_path: str | Path) -> dict[str, Any]:
    """``[ix.id for ix in env.intersections]`` plus each one's incoming-lane order.

    The lane order is reported beside the intersection order because it is the same
    failure class one level down, and on hangzhou it is already known to differ.
    """
    from experiments.config import EnvSpec, SETTING_DEFAULTS
    from experiments.envs import make_env

    settings = dict(SETTING_DEFAULTS)
    settings["max_steps"] = 1
    spec = EnvSpec(
        id=Path(config_path).stem,
        backend=backend,
        paths={"config": str(Path(config_path).resolve())},
        settings=settings,
    )
    env = make_env(spec)
    try:
        return {
            "backend": backend,
            "config": str(config_path),
            "intersection_order": [str(ix.id) for ix in env.intersections],
            "incoming_lane_order": {
                str(ix.id): [str(lid) for lid in ix.incoming_lanes]
                for ix in env.intersections
            },
            "num_phases": {str(ix.id): int(ix.num_phases) for ix in env.intersections},
        }
    finally:
        env.close()


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip()
    except OSError:  # pragma: no cover - defensive
        return ""


def _serialise(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"cannot serialise {type(obj)!r}")


def write_artifact(payload: Mapping[str, Any], path: str | Path) -> str:
    """Write the gate artifact. Serialisation is validated before anything is created."""
    text = json.dumps(payload, indent=2, sort_keys=False, default=_serialise) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return str(target)


# ----------------------------------------------------------------------
# The campaign: six cells, then the report
# ----------------------------------------------------------------------

#: The declared cells of ``docs/plans/p7.0.md`` section 5.2, in report order.
CELLS: tuple[tuple[str, str], ...] = (
    ("cityflow", "maxpressure"),
    ("cityflow", "fixedtime"),
    ("cityflow", "random"),
    ("sumo", "maxpressure"),
    ("sumo", "fixedtime"),
    ("sumo", "random"),
)

#: Settings held identical across every cell; only backend and policy vary.
COLLECT_SETTINGS: tuple[str, ...] = (
    "--max-steps", "360",
    "--delta-time", "10",
    "--control-mode", "acyclic",
    "--global-reward-fn", "queue_length",
    "--local-reward-fn", "queue_length",
    "--global-reward-weight", "0.0",
    "--state-features", "lane_vehicle_count", "lane_waiting", "phase_onehot",
    "--fixed-time-k", "4",
    "--thread-num", "1",
)


def backend_config(backend: str) -> Path:
    """The declared env config for a backend (plan section 5.2)."""
    from offline import parity

    if backend == "cityflow":
        return parity.REPO_ROOT / "configs" / "sim" / "cityflow1x1.json"
    if backend == "sumo":
        return Path(parity.DECLARED_PARITY_SUMOCFG)
    raise ValueError(f"unsupported backend {backend!r}")


def collect_cell(
    backend: str, policy: str, out_dir: Path, episodes: int, base_seed: int
) -> list[str]:
    """Run one cell through ``offline.collect``. Returns the argv actually used."""
    from offline import collect

    argv = [
        "--backend", backend,
        "--env-config", str(backend_config(backend)),
        "--policy", policy,
        "--episodes", str(int(episodes)),
        "--base-seed", str(int(base_seed)),
        "--out-dir", str(out_dir),
        "--overwrite",
        *COLLECT_SETTINGS,
    ]
    code = collect.main(argv)
    if code != 0:
        raise RuntimeError(f"collection failed for {backend}/{policy} with exit {code}")
    return argv


def load_cell_episodes(out_dir: str | Path) -> list[Any]:
    """Load a cell's episodes in manifest order (never in filesystem order)."""
    from offline.trajectory_logger import MANIFEST_NAME, load_episode

    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    return [load_episode(out_dir / entry["filename"]) for entry in manifest["episodes"]]


def cell_summary(episodes: Sequence[Any]) -> dict[str, Any]:
    """ATT at the horizon with `PREREGISTRATION` A5's unconditional co-report."""
    if not episodes:
        raise ValueError("a cell with no episodes cannot be summarised")
    horizons: list[float] = []
    counts: list[int] = []
    finals: list[float] = []
    fingerprints: set[bytes] = set()
    for episode in episodes:
        if episode.att_per_step is None:
            raise ValueError(
                "this episode predates format v1.1 and carries no att_per_step; the "
                "registered primary metric (A1) cannot be read from it"
            )
        horizons.append(float(episode.att_per_step[-1]))
        counts.append(int(episode.vehicle_count[-1]))
        finals.append(float(episode.sim_time[-1]))
        fingerprints.add(np.asarray(episode.att_per_step).tobytes())
    return {
        "n_episodes": len(episodes),
        "n_distinct_episodes": len(fingerprints),
        "att_horizon_mean": float(np.mean(horizons)),
        "att_horizon_per_episode": horizons,
        "att_horizon_sd": float(np.std(horizons, ddof=1)) if len(horizons) > 1 else 0.0,
        "vehicle_count_horizon_mean": float(np.mean(counts)),
        "vehicle_count_horizon_per_episode": counts,
        "final_sim_time_per_episode": finals,
        "episode_length": [int(e.episode_length) for e in episodes],
        "engine_seeds": [int(e.engine_seed) for e in episodes],
        "flow_draws": [int(e.flow_draw) for e in episodes],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the six declared cells and write the gate artifact.

    ``--skip-collect`` reuses episodes already on disk, which is what makes the
    analysis re-runnable without re-simulating.
    """
    from offline import parity

    parser = argparse.ArgumentParser(
        prog="python -m offline.transfer_gate",
        description=(
            "P7.0: report the CityFlow to SUMO dynamics shift. Reports; does not rule."
        ),
    )
    parser.add_argument("--out-root", required=True, help="episode root (gitignored)")
    parser.add_argument("--artifact", required=True, help="gate JSON to write")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--deferred-18-steps", type=int, default=360)
    args = parser.parse_args(argv)

    out_root = Path(args.out_root)
    invocations: dict[str, list[str]] = {}
    cells: dict[str, dict[str, Any]] = {}
    episodes_by_cell: dict[str, list[Any]] = {}

    for backend, policy in CELLS:
        key = f"{backend}__{policy}"
        cell_dir = out_root / key
        if not args.skip_collect:
            invocations[key] = collect_cell(
                backend, policy, cell_dir, args.episodes, args.base_seed
            )
            print(f"collected {key} -> {cell_dir}", flush=True)
        loaded = load_cell_episodes(cell_dir)
        episodes_by_cell[key] = loaded
        cells[key] = cell_summary(loaded)

    enumeration = {
        "hangzhou_1x1_bc-tyc__cityflow": intersection_enumeration(
            "cityflow", backend_config("cityflow")
        ),
        "hangzhou_1x1_bc-tyc__sumo": intersection_enumeration(
            "sumo", backend_config("sumo")
        ),
        "cologne3__cityflow": intersection_enumeration(
            "cityflow", parity.REPO_ROOT / "configs" / "sim" / "cityflow_cologne3.json"
        ),
        "cologne3__sumo": intersection_enumeration(
            "sumo", parity.REPO_ROOT / "scenarios" / "cologne3" / "cologne3.sumocfg"
        ),
    }

    cf_enum = enumeration["hangzhou_1x1_bc-tyc__cityflow"]
    sumo_enum = enumeration["hangzhou_1x1_bc-tyc__sumo"]
    ix_id = cf_enum["intersection_order"][0]
    cf_lanes = list(cf_enum["incoming_lane_order"][ix_id])
    sumo_lanes = list(sumo_enum["incoming_lane_order"].get(ix_id, []))
    shared = [lid for lid in cf_lanes if lid in set(sumo_lanes)]
    lane_note = {
        "intersection": ix_id,
        "cityflow_incoming_lane_order": cf_lanes,
        "sumo_incoming_lane_order": sumo_lanes,
        "orders_agree": cf_lanes == sumo_lanes,
        "sets_agree": sorted(cf_lanes) == sorted(sumo_lanes),
        "only_in_cityflow": sorted(set(cf_lanes) - set(sumo_lanes)),
        "only_in_sumo": sorted(set(sumo_lanes) - set(cf_lanes)),
        "compared_lanes": shared,
    }

    features = compare_lane_features(
        episodes_by_cell["cityflow__maxpressure"],
        episodes_by_cell["sumo__maxpressure"],
        shared,
    )

    att = {key: cells[key]["att_horizon_mean"] for key in cells}
    delta_cf = att["cityflow__fixedtime"] - att["cityflow__maxpressure"]
    delta_sumo = att["sumo__fixedtime"] - att["sumo__maxpressure"]
    rho_table = {
        backend: {
            policy: rho(
                att[f"{backend}__fixedtime"],
                att[f"{backend}__{policy}"],
                att[f"{backend}__maxpressure"],
            )
            for policy in ("fixedtime", "maxpressure", "random")
        }
        for backend in ("cityflow", "sumo")
    }

    verdict = evaluate_branch(
        features=features,
        delta_cityflow=delta_cf,
        delta_sumo=delta_sumo,
        rho_cityflow_random=rho_table["cityflow"]["random"],
        rho_sumo_random=rho_table["sumo"]["random"],
    )

    deferred_18 = metric_set_independence(
        parity.DECLARED_PARITY_SUMOCFG,
        steps=int(args.deferred_18_steps),
        seed=int(args.base_seed),
    )

    # Additive, and it feeds nothing above: `evaluate_branch` has already run on the
    # registered inputs by this point and is not consulted again.
    diagnostic = diagnose_lane_convention(
        episodes_by_cell["cityflow__maxpressure"],
        episodes_by_cell["sumo__maxpressure"],
        shared,
        parity.DECLARED_SCENARIO_DIR / "roadnet.json",
        parity.DECLARED_SOURCE_NET,
        ix_id,
    )
    n_rows = features[0].n_cityflow if features else 0
    m_rows = features[0].n_sumo if features else 0

    payload: dict[str, Any] = {
        "format_version": GATE_FORMAT_VERSION,
        "task": "P7.0",
        "scenario": "hangzhou_1x1_bc-tyc_18041610_1h",
        "reports_but_does_not_rule": (
            "BRIEF_04 section 4.4: the go/no-go ruling belongs to the coordinator."
        ),
        "parity": {
            "contract_version": parity.PARITY_CONTRACT_VERSION,
            "vtype_id": parity.PARITY_VTYPE_ID,
            "vtype": parity.parity_vtype_attributes(),
            "unmatchable": [
                {"parameter": name, "reason": reason}
                for name, reason in parity.UNMATCHABLE_PARAMETERS
            ],
            "flow_json_disagreements": parity.flow_json_disagreements(
                parity.DECLARED_SOURCE_FLOW_JSON
            ),
            "shipped_binding": asdict(
                parity.vtype_binding_report(parity.DECLARED_SOURCE_ROU)
            ),
            "parity_binding": asdict(
                parity.vtype_binding_report(parity.DECLARED_PARITY_ROU)
            ),
            "shipped_binding_is_complete": parity.binding_is_complete(
                parity.vtype_binding_report(parity.DECLARED_SOURCE_ROU)
            ),
            "parity_binding_is_complete": parity.binding_is_complete(
                parity.vtype_binding_report(parity.DECLARED_PARITY_ROU)
            ),
        },
        "horizon_margin": {
            "sumo_end_seconds": parity.SUMO_END_SECONDS,
            "env_horizon_seconds": parity.ENV_HORIZON_SECONDS,
            "margin_seconds": parity.SUMO_END_SECONDS - parity.ENV_HORIZON_SECONDS,
            "measured_final_sim_time": {
                key: cells[key]["final_sim_time_per_episode"]
                for key in cells
                if key.startswith("sumo__")
            },
        },
        "cells": cells,
        "lane_alignment": lane_note,
        "per_feature": [asdict(row) for row in features],
        "ks_noise_floor": {
            "pooled_n_cityflow": n_rows,
            "pooled_n_sumo": m_rows,
            "critical_value_5pct_pooled": (
                1.358 * ((n_rows + m_rows) / (n_rows * m_rows)) ** 0.5
                if n_rows and m_rows
                else None
            ),
            "note": (
                "a reading aid, never a test: the rows are autocorrelated and a "
                "deterministic backend repeats its episodes, so the effective sample "
                "size is the distinct-episode count reported per cell"
            ),
        },
        "rho": {
            "formula": "(ATT_fixedtime - ATT_policy) / (ATT_fixedtime - ATT_maxpressure)",
            "computed_within_each_backend": True,
            "no_raw_cross_backend_att_comparison": True,
            "delta_cityflow": delta_cf,
            "delta_sumo": delta_sumo,
            "values": rho_table,
        },
        "branch": {
            "branch": verdict.branch,
            "firing_criteria": list(verdict.firing_criteria),
            "failed_a_criteria": list(verdict.failed_a_criteria),
            "nearest_non_firing": verdict.nearest_non_firing,
            "criterion_scales": dict(CRITERION_SCALES),
            "rows": [asdict(row) for row in verdict.rows],
        },
        "deferred_18": deferred_18,
        "deferred_23": enumeration,
        "lane_convention_diagnostic": diagnostic,
        "runtime": {
            "written_at_git_commit": _git_commit(),
            "collect_invocations": invocations,
            "episodes_per_cell": int(args.episodes),
            "base_seed": int(args.base_seed),
            "skip_collect": bool(args.skip_collect),
        },
    }

    written = write_artifact(payload, args.artifact)
    print(f"branch={verdict.branch} firing={list(verdict.firing_criteria)} "
          f"failed_a={list(verdict.failed_a_criteria)} "
          f"nearest_non_firing={verdict.nearest_non_firing}")
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
