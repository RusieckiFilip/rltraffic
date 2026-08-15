"""P4.7 -- the mixture tiers: phase 2 of the method x tier grid.

This module owns everything phase 2 needs that :mod:`offline.method_tier_grid` does not already
provide.  **Training and evaluation are that module's, unchanged** -- P4.7 invokes its ``train`` and
``evaluate`` subcommands with ``--artifact-prefix p4_7`` so the twelve new cells are measured by
exactly the instrument that measured the twenty re-used ones.

**Artifact formats written here** (every payload carries its version):

* ``p4.7-draw-identity/1.0`` -- Gate D: the regenerated held-out demand against the surviving
  directories, before anything is evaluated on it.

**What this module deliberately does NOT do.**  It does not re-declare a tier, a fraction, an RNG or
a behaviour reference that ``method_tier_grid`` already declares; it imports them.  Two sources of
truth for a declared quantity is the ``BEHAVIOUR_ATT`` / ``behaviour_cells`` hazard, and P4.7 is the
task whose whole subject is declarations.

*Alignment* is contract C6's throughout, unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

DRAW_IDENTITY_FORMAT_VERSION = "p4.7-draw-identity/1.0"

#: The files a materialised draw directory holds.  ``routes.rou.xml`` is SUMO's and is unused by
#: this task's CityFlow evaluation, but it is compared anyway: a recipe that reproduces two of three
#: outputs has not been shown to reproduce.
DRAW_FILES = ("flow.json", "cityflow.json", "routes.rou.xml")

#: The one key of a materialised ``cityflow.json`` that legitimately differs between two checkouts:
#: it is the absolute path of the scenario directory the config was rendered against.  Every other
#: key -- including ``flowFile``, which is stored RELATIVE to ``dir`` -- must agree exactly.
DRAW_CONFIG_PATH_KEY = "dir"


def file_sha256(path: str | Path) -> str:
    """sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_draw_directories(reference: str | Path, candidate: str | Path) -> dict[str, Any]:
    """Compare one regenerated draw against a surviving one, file by file.

    **Why this is not a plain byte comparison of the whole directory.**  A materialised
    ``cityflow.json`` stores ``dir`` as the **absolute** path of the source scenario directory
    (``offline/collect.py::_write_draw_config``), so the same recipe run from two checkouts of the
    same commit produces configs that differ in exactly that one string and in nothing else.
    ``flow.json`` -- the demand, which is the quantity that must not move -- carries no path and is
    required to be **byte-identical**.

    Returns a record of what matched; raises :class:`ValueError` naming the draw and the file when
    anything does not.  The record is what Gate D writes into its artifact, so a later reader sees
    which files were compared rather than a bare "PASS".
    """
    reference_dir = Path(reference)
    candidate_dir = Path(candidate)
    identical: list[str] = []
    for name in DRAW_FILES:
        left, right = reference_dir / name, candidate_dir / name
        if not left.is_file():
            raise ValueError(f"{reference_dir.name}: the surviving draw has no {name}")
        if not right.is_file():
            raise ValueError(f"{candidate_dir.name}: the regenerated draw has no {name}")
        left_digest, right_digest = file_sha256(left), file_sha256(right)
        if left_digest == right_digest:
            identical.append(name)
            continue
        if name != "cityflow.json":
            raise ValueError(
                f"{reference_dir.name}/{name}: the regenerated demand does not reproduce "
                f"({right_digest} against the surviving {left_digest}); every P4.6 number was "
                "measured on the surviving bytes and a mixture cell measured on these would not be "
                "comparable with them"
            )
        _assert_configs_differ_only_in_the_scenario_path(left, right)
    return {
        "draw_dir": candidate_dir.name,
        "byte_identical": identical,
        "path_normalised": [name for name in DRAW_FILES if name not in identical],
    }


def _assert_configs_differ_only_in_the_scenario_path(reference: Path, candidate: Path) -> None:
    """Two rendered sim configs may differ in ``dir`` and in nothing else."""
    left = json.loads(reference.read_text(encoding="utf-8"))
    right = json.loads(candidate.read_text(encoding="utf-8"))
    if set(left) != set(right):
        raise ValueError(
            f"{reference.parent.name}/cityflow.json: the regenerated config has different keys "
            f"({sorted(set(right) - set(left))} added, {sorted(set(left) - set(right))} missing)"
        )
    differing = sorted(key for key in left if left[key] != right[key])
    if differing != [DRAW_CONFIG_PATH_KEY]:
        raise ValueError(
            f"{reference.parent.name}/cityflow.json: the regenerated config differs on {differing}, "
            f"but only {DRAW_CONFIG_PATH_KEY!r} may differ between two checkouts; a difference "
            "anywhere else means the two configs describe different simulations"
        )
    for path in (left[DRAW_CONFIG_PATH_KEY], right[DRAW_CONFIG_PATH_KEY]):
        roadnet = Path(path) / str(left["roadnetFile"])
        if not roadnet.is_file():
            raise ValueError(
                f"{reference.parent.name}/cityflow.json: {roadnet} does not exist, so this config's "
                f"{DRAW_CONFIG_PATH_KEY!r} does not point at a usable scenario"
            )
    digests = {
        file_sha256(Path(cfg[DRAW_CONFIG_PATH_KEY]) / str(cfg["roadnetFile"]))
        for cfg in (left, right)
    }
    if len(digests) != 1:
        raise ValueError(
            f"{reference.parent.name}/cityflow.json: the two {DRAW_CONFIG_PATH_KEY!r} values point "
            f"at different roadnets ({sorted(digests)}); the configs would run different networks"
        )


def draw_identity_artifact(
    reference_root: str | Path,
    candidate_root: str | Path,
    draws: Sequence[int],
    *,
    scenario_key: str = "cityflow1x1",
) -> dict[str, Any]:
    """Gate D: every surviving held-out draw reproduces from the recipe.

    ⚠️ **The exposure this gate exists to close, stated in the artifact rather than only in a
    packet.**  Of the hundred held-out draws every merged P4.6 number was measured on, **five
    directories survive anywhere on this machine**; the other ninety-five exist only as
    :mod:`offline.materialise_draws` plus a seed.  Until this gate ran, *the recipe behind those
    numbers had never been checked against the bytes they were produced from*.

    Raises rather than reporting a failure code: a mismatch means the mixture cells would be
    measured on different demand from the cells they are compared with, and no P4.7 number may
    exist in that case.
    """
    reference = Path(reference_root)
    candidate = Path(candidate_root)
    requested = [int(draw) for draw in draws]
    if not requested:
        raise ValueError("Gate D needs at least one surviving draw to compare against")
    records = []
    for draw_id in requested:
        name = f"draw_{draw_id:04d}"
        left, right = reference / scenario_key / name, candidate / scenario_key / name
        if not left.is_dir():
            raise ValueError(f"no surviving draw at {left}; Gate D compares against the survivors")
        if not right.is_dir():
            raise ValueError(f"no regenerated draw at {right}")
        record = compare_draw_directories(left, right)
        record["draw_id"] = draw_id
        records.append(record)
    return {
        "format_version": DRAW_IDENTITY_FORMAT_VERSION,
        "role": (
            "Gate D: the held-out demand P4.6's merged cells were measured on, regenerated from "
            "the recipe and compared byte for byte against the surviving directories"
        ),
        "scenario_key": str(scenario_key),
        "reference_root": str(reference),
        "candidate_root": str(candidate),
        "draws_compared": requested,
        "survivors_available": sorted(
            int(path.name.removeprefix("draw_"))
            for path in (reference / scenario_key).glob("draw_1*")
            if path.is_dir()
        ),
        "files_compared": list(DRAW_FILES),
        "path_key_allowed_to_differ": DRAW_CONFIG_PATH_KEY,
        "draws": records,
        "status": "PASS",
    }


# ----------------------------------------------------------------------
# The CONSTRUCTED behaviour reference (BRIEF_19 section 3, docs/plans/p4.7.md section 5)
# ----------------------------------------------------------------------

CONSTRUCTED_REFERENCE_FORMAT_VERSION = "p4.7-constructed-reference/1.0"

#: Declared in ``docs/plans/p4.7.md`` section 5, before any P4.7 number existed.  The repo's date
#: convention, and deliberately NOT ``MIXTURE_RNG_BASE``: the composition draw and the reference
#: draw are independent objects and sharing a stream between them would tie two unrelated
#: realisations together for no reason.
CONSTRUCTED_REFERENCE_RNG_SEED = 20_260_814

#: The two arms a mixture's behaviour reference is composed from, in the order the tier's
#: ``components`` declares them.
CONSTRUCTED_REFERENCE_ARMS = ("behaviour@mappo1000", "behaviour@random")

#: The per-episode fields carried through from whichever component a draw is assigned to.  **All of
#: them come from the SAME source episode**: mixing fields across components would fabricate an
#: episode that no rollout produced.
CONSTRUCTED_EPISODE_FIELDS = ("att_horizon", "horizon_vehicle_count", "episode_reward")


def component_episodes(
    payload: Mapping[str, Any], arm: str
) -> dict[tuple[int, int], dict[str, float]]:
    """``(seed, draw) -> record`` for one arm of a committed grid artifact.

    A repeated ``(seed, draw)`` is refused rather than resolved: whichever record won would be an
    arbitrary choice between two measurements of the same cell, and the constructed reference would
    inherit it silently.
    """
    out: dict[tuple[int, int], dict[str, float]] = {}
    for record in payload.get("episodes", ()):
        if str(record.get("arm")) != str(arm):
            continue
        key = (int(record["seed"]), int(record["draw_id"]))
        if key in out:
            raise ValueError(
                f"{arm}: (seed {key[0]}, draw {key[1]}) appears twice in this artifact, so there is "
                "no single stored value for it"
            )
        out[key] = {
            field: float(record[field]) for field in CONSTRUCTED_EPISODE_FIELDS if field in record
        }
    if not out:
        raise ValueError(f"this artifact carries no episodes for {arm!r}")
    return out


def component_from_sources(
    grid: Mapping[str, Any], arm: str, primary: Mapping[str, Any], primary_arm: str
) -> dict[tuple[int, int], dict[str, float]]:
    """One component, read from the committed grid and proved equal to its PRIMARY artifact.

    The grid artifact carries a copy of each behaviour cell; the primary artifact is where that cell
    was written when it was measured (``docs/data/p4_heldout_thresholds.json`` for ``mappo1000``,
    ``output/p4_6/eval_random_behaviour.json`` for ``random``).  Reading the copy is convenient and
    reading only the copy is unverified, so both are read and every value must agree **exactly** --
    which also proves the secured raw evidence is faithful to what the merged artifact reports.
    """
    copy = component_episodes(grid, arm)
    source = component_episodes(primary, primary_arm)
    if set(copy) != set(source):
        only_copy = sorted(set(copy) - set(source))[:3]
        only_source = sorted(set(source) - set(copy))[:3]
        raise ValueError(
            f"{arm}: the committed grid and {primary_arm!r} describe different (seed, draw) sets "
            f"(grid only {only_copy}, source only {only_source})"
        )
    for key in sorted(copy):
        for field in CONSTRUCTED_EPISODE_FIELDS:
            if copy[key][field] != source[key][field]:
                raise ValueError(
                    f"{arm}: (seed {key[0]}, draw {key[1]}) has {field} {copy[key][field]!r} in the "
                    f"committed grid but {source[key][field]!r} in {primary_arm!r}; the constructed "
                    "reference may not be built from a copy that does not match its source"
                )
    return copy


def constructed_reference_assignment(
    tier: str, seeds: Sequence[int], draws: Sequence[int]
) -> dict[int, list[int]]:
    """Per seed, the draws assigned to the EXPERT component.

    Exactly ``round(100 * fraction)`` of each seed's held-out draws, drawn without replacement by
    ``default_rng([CONSTRUCTED_REFERENCE_RNG_SEED, round(100 * fraction), seed])`` -- one independent
    stream per ``(tier, seed)``, so no two seeds and no two tiers share a realisation.

    **Per seed rather than pooled**, because the corpus's composition is per corpus: a pooled draw
    could give one seed 10 expert draws and another 56, average to the fraction, and produce a
    reference whose per-seed composition differs from the corpus it stands for.
    """
    import numpy as np

    from offline.method_tier_grid import MIXTURE_EXPERT_FRACTION

    key = str(tier)
    if key not in MIXTURE_EXPERT_FRACTION:
        raise ValueError(
            f"{key!r} has no constructed reference; only the mixture tiers "
            f"{sorted(MIXTURE_EXPERT_FRACTION)} are compositions rather than policies, and every "
            "other tier's reference is rolled out or committed"
        )
    fraction = float(MIXTURE_EXPERT_FRACTION[key])
    pool = sorted(int(draw) for draw in draws)
    count = int(round(len(pool) * fraction))
    out: dict[int, list[int]] = {}
    for seed in seeds:
        rng = np.random.default_rng(
            [CONSTRUCTED_REFERENCE_RNG_SEED, int(round(100 * fraction)), int(seed)]
        )
        chosen = rng.choice(len(pool), size=count, replace=False)
        out[int(seed)] = sorted(pool[int(index)] for index in chosen)
    return out


def constructed_reference_artifact(
    tier: str,
    expert: Mapping[tuple[int, int], Mapping[str, float]],
    random_pool: Mapping[tuple[int, int], Mapping[str, float]],
    *,
    seeds: Sequence[int],
    draws: Sequence[int],
) -> dict[str, Any]:
    """A mixture's behaviour reference, CONSTRUCTED and labelled as such.

    ⚠️ **This is a REALISATION, not an expectation, and the distinction is the whole design.**  Each
    ``(seed, draw)`` takes the **stored record of one component**, chosen by the declared assignment;
    every carried field comes from that same record.  The alternative --
    ``f * ATT_expert(d) + (1 - f) * ATT_random(d)`` -- would reproduce the mean and **understate the
    variance**, because it removes the composition's own randomness, so a paired CI computed against
    it would overstate precision.

    Zero compute: both components are already measured on draws 1000-1099 at five seeds, and this
    only decides which of the two stands for each draw.
    """
    from offline.dt_gate import EpisodeResult
    from offline.method_tier_grid import MIXTURE_EXPERT_FRACTION, arm_key, cell_stats

    key = str(tier)
    assignment = constructed_reference_assignment(key, seeds, draws)
    fraction = float(MIXTURE_EXPERT_FRACTION[key])
    arm = arm_key("behaviour", key)
    ordered_seeds = [int(seed) for seed in seeds]
    ordered_draws = sorted(int(draw) for draw in draws)

    # ⚠️ BOTH components are validated over the WHOLE grid before anything is built, and not
    # lazily as each draw is consulted.  A lazy check passes or fails according to the random
    # assignment -- a hole in the expert cell at (303, 1042) is invisible whenever that draw
    # happens to fall to the random side -- so the refusal would be a coin flip.  It also protects
    # the bracket, which is computed over every record of each component and would otherwise
    # average 499 of them while reporting an endpoint.
    for source_name, source_map in zip(CONSTRUCTED_REFERENCE_ARMS, (expert, random_pool)):
        for seed in ordered_seeds:
            for draw in ordered_draws:
                if (seed, draw) not in source_map:
                    raise ValueError(
                        f"{arm}: {source_name} has no stored record for (seed {seed}, draw {draw}), "
                        "so this draw has no measured value to stand for it; a reference with a "
                        "hole would be compared against a different draw set from every other arm "
                        "(PREREGISTRATION A5)"
                    )
                missing = [
                    field
                    for field in CONSTRUCTED_EPISODE_FIELDS
                    if field not in source_map[(seed, draw)]
                ]
                if missing:
                    raise ValueError(
                        f"{arm}: {source_name}'s record for (seed {seed}, draw {draw}) is missing "
                        f"{missing}; every carried field comes from the same source episode and a "
                        "partially filled record would describe an episode no rollout produced"
                    )

    records: list[dict[str, Any]] = []
    for seed in ordered_seeds:
        expert_draws = set(assignment[seed])
        for draw in ordered_draws:
            source_name, source_map = (
                (CONSTRUCTED_REFERENCE_ARMS[0], expert)
                if draw in expert_draws
                else (CONSTRUCTED_REFERENCE_ARMS[1], random_pool)
            )
            source = source_map[(seed, draw)]
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "draw_id": draw,
                    "component": source_name,
                    **{field: float(source[field]) for field in CONSTRUCTED_EPISODE_FIELDS},
                }
            )

    def _cell(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
        return cell_stats(
            [
                EpisodeResult(
                    arm=label,
                    seed=int(row["seed"]),
                    draw_id=int(row["draw_id"]),
                    att_horizon=float(row["att_horizon"]),
                    horizon_vehicle_count=float(row["horizon_vehicle_count"]),
                    episode_reward=float(row["episode_reward"]),
                )
                for row in rows
            ]
        )

    bracket = {
        name: _cell(
            [
                {"seed": seed, "draw_id": draw, **values}
                for (seed, draw), values in sorted(source.items())
            ],
            name,
        )
        for name, source in zip(CONSTRUCTED_REFERENCE_ARMS, (expert, random_pool))
    }
    return {
        "format_version": CONSTRUCTED_REFERENCE_FORMAT_VERSION,
        "arm": arm,
        "tier": key,
        "method": "behaviour",
        "role": (
            "the behaviour reference of a mixture tier, CONSTRUCTED rather than rolled out: a "
            "33/50/67 %-expert corpus is a composition, not a policy anyone ran, so there is "
            "nothing to roll out and a training-draw substitute would be void under "
            "PREREGISTRATION A5"
        ),
        "behaviour_reference": {
            "source": "constructed",
            "role": (
                "a REALISATION of the composition, not an expectation: each draw takes one "
                "component's stored record, never a weighted average of the two, because an "
                "average would reproduce the mean and understate the variance"
            ),
            "rng_seed": CONSTRUCTED_REFERENCE_RNG_SEED,
            "rng": (
                "numpy.random.default_rng([rng_seed, round(100 * expert_fraction), seed]), one "
                "independent stream per (tier, seed)"
            ),
            "expert_fraction": fraction,
            "expert_draws_per_seed": int(round(len(ordered_draws) * fraction)),
            "components": list(CONSTRUCTED_REFERENCE_ARMS),
            "artifact": "docs/data/p4_6_grid.json, the committed held-out behaviour cells",
            "declared_in": "docs/plans/p4.7.md section 5, before any P4.7 number existed",
            "cost": "zero compute; both components were measured on draws 1000-1099 at 5 seeds",
        },
        "assignment": {str(seed): list(draws_) for seed, draws_ in assignment.items()},
        "bracket": bracket,
        "cell": _cell(records, arm),
        "episodes": records,
    }


# ----------------------------------------------------------------------
# Gate P1 and the registered predictions Q1-Q3 (docs/plans/p4.7.md sections 4 and 6.3)
# ----------------------------------------------------------------------

MIXTURE_GRID_FORMAT_VERSION = "p4.7-mixture-grid/1.0"

#: Imported once, at module scope, so every scorer iterates the tiers in the SAME declared order
#: and no function re-lists them.
from offline.method_tier_grid import MIXTURE_TIER_ORDER as MIXTURE_TIER_ORDER_LOCAL

#: Q1's threshold: at least this fraction of the kept top decile comes from the expert dirs.
Q1_EXPERT_FRACTION_FLOOR = 0.90

#: The registered predictions, in the order the plan states them.
MIXTURE_PREDICTION_ORDER = ("Q1", "Q2", "Q3")

#: Q1's and Q3's composition check share ``P3_ALPHA``; declared in the plan, inherited unchanged.


def expert_dir_names() -> frozenset[str]:
    """The directory names a mixture's EXPERT component may come from, from the declared spec.

    Derived from ``tier_spec(components[0]).dirs`` rather than written out again: a second copy of
    the expert tier's directory list is a second source of truth for a declared quantity.
    """
    from offline.method_tier_grid import tier_spec

    mixture = tier_spec(MIXTURE_TIER_ORDER_LOCAL[0])
    return frozenset(tier_spec(mixture.components[0]).dirs)


def kept_expert_counts(declaration: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Per mixture tier, how much of the kept top decile came from the expert component.

    Counted from the kept set's own ``by_dataset_dir`` composition -- what the filter actually
    selected -- and never from the fraction the tier was composed at.
    """
    experts = expert_dir_names()
    out: dict[str, dict[str, int]] = {}
    for tier in MIXTURE_TIER_ORDER_LOCAL:
        entry = declaration["tiers"][tier]
        by_dir = entry["top_decile_composition"]["by_dataset_dir"]
        kept = int(entry["top_decile_composition"]["total"])
        expert = sum(int(count) for name, count in by_dir.items() if name in experts)
        out[tier] = {
            "expert": expert,
            "random": kept - expert,
            "kept": kept,
            "training_expert": sum(
                int(count)
                for name, count in entry["training_composition"]["by_dataset_dir"].items()
                if name in experts
            ),
            "training_streams": int(entry["training_composition"]["total"]),
        }
    return out


def component_return_overlap(declaration: Mapping[str, Any], tier: str) -> dict[str, Any]:
    """How far the two components' return distributions overlap -- Q1's tautology disclosure.

    ⚠️ **Registered before the number was seen** (``docs/plans/p4.7.md`` section 4.1): if the two
    components' returns do not overlap, "the filter selects the expert fraction" is arithmetic rather
    than a finding, because the top decile by return *is* the expert fraction by construction.  This
    is the same move P4.6's review forced on ``maxpressure``'s circular check, made in advance
    instead of afterwards.
    """
    experts = expert_dir_names()
    streams = declaration["tiers"][str(tier)]["streams"]
    expert_returns = [
        float(s["total_return"]) for s in streams if Path(s["dataset_dir"]).name in experts
    ]
    random_returns = [
        float(s["total_return"]) for s in streams if Path(s["dataset_dir"]).name not in experts
    ]
    if not expert_returns or not random_returns:
        raise ValueError(f"{tier}: a mixture needs both components to compare their returns")
    pairs = len(expert_returns) * len(random_returns)
    wins = sum(1 for a in expert_returns for b in random_returns if a > b)
    ties = sum(1 for a in expert_returns for b in random_returns if a == b)
    effect = (wins + 0.5 * ties) / pairs
    above = sum(1 for b in random_returns if b > min(expert_returns))
    return {
        "expert_streams": len(expert_returns),
        "random_streams": len(random_returns),
        "expert_return_min": min(expert_returns),
        "expert_return_max": max(expert_returns),
        "random_return_min": min(random_returns),
        "random_return_max": max(random_returns),
        "common_language_effect_size": effect,
        "random_above_expert_minimum": above,
        "separates_completely": above == 0,
        "role": (
            "P(expert return > random return) over all cross-component pairs, ties at half.  1.0 "
            "means the components do not overlap at all, so Q1's kept-set composition is arithmetic "
            "rather than evidence about the filter; 0.5 means the return carries no component "
            "information at all"
        ),
    }


def score_q1(declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Q1: does the filter FIND the expert fraction?  Leakage-free, no training.

    **HELD** iff at least :data:`Q1_EXPERT_FRACTION_FLOOR` of every mixture's kept top decile comes
    from the expert directories.  The statistic is an integer against an integer threshold, so no
    tie is possible.
    """
    from offline.method_tier_grid import hypergeometric_upper_tail

    counts = kept_expert_counts(declaration)
    by_tier: dict[str, Any] = {}
    failing: list[str] = []
    for tier in MIXTURE_TIER_ORDER_LOCAL:
        entry = counts[tier]
        fraction = entry["expert"] / entry["kept"] if entry["kept"] else 0.0
        if fraction < Q1_EXPERT_FRACTION_FLOOR:
            failing.append(tier)
        by_tier[tier] = {
            **entry,
            "fraction": fraction,
            "floor": Q1_EXPERT_FRACTION_FLOOR,
            "minimum_kept_expert_to_hold": math.ceil(
                Q1_EXPERT_FRACTION_FLOOR * entry["kept"]
            ),
            "hypergeometric_p_value": hypergeometric_upper_tail(
                entry["training_streams"], entry["training_expert"], entry["kept"], entry["expert"]
            ),
            "expected_expert": entry["kept"] * entry["training_expert"] / entry["training_streams"],
            "by_behaviour_seed": dict(
                declaration["tiers"][tier]["top_decile_composition"]["by_behaviour_seed"]
            ),
            "companion_note": (
                "the exact hypergeometric tail and the seed histogram carry no threshold; the "
                "histogram is BRIEF_17 section 11 finding A5's second axis, without which "
                "'the filter selected the expert fraction' is confounded with the checkpoint "
                "selection P4.5 already established"
            ),
            "overlap_disclosure": (
                component_return_overlap(declaration, tier)
                if "streams" in declaration["tiers"][tier]
                else None
            ),
        }
    return {
        "prediction": (
            "Q1 -- the filter FINDS the expert fraction: on all three mixtures, at least 90 % of "
            "%BC's kept top-decile streams come from the mappo1000 dirs"
        ),
        "rule": (
            f"HELD iff kept_expert / kept >= {Q1_EXPERT_FRACTION_FLOOR} on every mixture tier; "
            "FAILED if any tier is below it; no tie is possible, the statistic is an integer"
        ),
        "outcome": "FAILED" if failing else "HELD",
        "failing_tiers": failing,
        "by_tier": by_tier,
        "leakage": "none: this reads the selection only and costs no training and no rollout",
    }


def score_q2(
    cells_by_tier: Mapping[str, Mapping[str, Any]], comparisons: Sequence[Any]
) -> dict[str, Any]:
    """Q2: is %BC's advantage over BC positive on all three, and decreasing in the fraction?

    ``advantage = mean ATT(bc) - mean ATT(bc_top10)``, so a positive value means %BC is better --
    lower ATT is better.  **HELD** iff all three are positive **and** strictly decreasing in the
    expert fraction; **NOT RESOLVED** on an exact tie, which makes "decreases" undefined.
    """
    advantage: dict[str, float] = {}
    for tier in MIXTURE_TIER_ORDER_LOCAL:
        methods = cells_by_tier.get(tier)
        if not methods or "bc" not in methods or "bc_top10" not in methods:
            continue
        advantage[tier] = float(methods["bc"]["att_horizon_mean"]) - float(
            methods["bc_top10"]["att_horizon_mean"]
        )

    intervals: dict[str, Any] = {}
    for comparison in comparisons:
        left, right = str(comparison.left_arm), str(comparison.right_arm)
        left_method, _, left_tier = left.partition("@")
        right_method, _, right_tier = right.partition("@")
        if left_tier != right_tier or left_tier not in MIXTURE_TIER_ORDER_LOCAL:
            continue
        if {left_method, right_method} != {"bc", "bc_top10"}:
            continue
        if left_method != "bc":
            raise ValueError(
                f"the bc/bc_top10 comparison for {left_tier} arrived as {left} vs {right}; Q2's "
                "advantage is defined as bc minus bc_top10 and the pair order fixes its sign"
            )
        intervals[left_tier] = {
            "mean_difference": comparison.mean_difference,
            "ci95_low": comparison.ci95_low,
            "ci95_high": comparison.ci95_high,
            "ci95_width": comparison.ci95_width,
            "rank_biserial": comparison.rank_biserial,
        }

    ordered = [advantage[t] for t in MIXTURE_TIER_ORDER_LOCAL if t in advantage]
    if len(advantage) != len(MIXTURE_TIER_ORDER_LOCAL):
        outcome = "NOT SCORABLE"
        all_positive = ordering = None
    else:
        all_positive = all(value > 0.0 for value in ordered)
        tied = any(a == b for a, b in zip(ordered, ordered[1:]))
        ordering = all(a > b for a, b in zip(ordered, ordered[1:]))
        outcome = (
            "NOT RESOLVED" if tied else ("HELD" if (all_positive and ordering) else "FAILED")
        )
    return {
        "prediction": (
            "Q2 -- %BC's advantage over BC is POSITIVE on all three mixtures, and DECREASES as the "
            "expert fraction RISES"
        ),
        "rule": (
            "advantage = mean ATT(bc) - mean ATT(bc_top10), positive meaning %BC is better.  HELD "
            "iff all three are positive AND advantage(mix33) > advantage(mix50) > advantage(mix67) "
            "strictly; NOT RESOLVED if any two are exactly equal; NOT SCORABLE without all three"
        ),
        "outcome": outcome,
        "all_positive": all_positive,
        "ordering_holds": ordering,
        "advantage_by_tier": advantage,
        "advantage_intervals": intervals,
        "companion_note": (
            "the paired difference, its 95 % CI, the CI WIDTH and the rank-biserial effect size are "
            "reported beside the rule and carry no threshold (BRIEF_17 section 4: no equivalence "
            "verdict anywhere)"
        ),
    }


def score_q3(declaration: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Q3: composition signature present, demand signature absent.

    The composition check is exact and shares P3's alpha: the hypergeometric tail of the kept set's
    expert count under "20 streams drawn uniformly from the 200".  The demand checks are
    ``BRIEF_17`` section 6's two, unchanged, scored by P4.6's own per-tier rule.
    """
    from offline.method_tier_grid import P3_ALPHA

    counts = kept_expert_counts(declaration)
    q1 = score_q1(declaration)
    by_tier: dict[str, Any] = {}
    for tier in MIXTURE_TIER_ORDER_LOCAL:
        entry = diagnostics["tiers"][tier]
        volume, difficulty = entry["volume"], entry["difficulty"]
        excludes_zero = bool(volume["excludes_zero"])
        withdrawn = bool(difficulty.get("withdrawn", False))
        significant = (not withdrawn) and float(difficulty["p_value"]) < P3_ALPHA
        composition_p = float(q1["by_tier"][tier]["hypergeometric_p_value"])
        by_tier[tier] = {
            "kept_expert": counts[tier]["expert"],
            "expected_expert": q1["by_tier"][tier]["expected_expert"],
            "composition_p_value": composition_p,
            "composition_signature": composition_p < P3_ALPHA,
            "volume_difference": float(volume["difference"]),
            "volume_ci95": [float(volume["ci95_low"]), float(volume["ci95_high"])],
            "volume_excludes_zero": excludes_zero,
            "difficulty_overlap": int(difficulty["overlap"]),
            "difficulty_p_value": float(difficulty["p_value"]),
            "difficulty_withdrawn": withdrawn,
            "return_versus_difficulty_rho": entry.get("return_versus_difficulty_rho"),
            "demand_signature": bool(significant or excludes_zero),
            "demand_signature_carried_by": (
                [n for n, on in (("volume", excludes_zero), ("difficulty", significant)) if on]
                or None
            ),
        }
    held = all(
        entry["composition_signature"] and not entry["demand_signature"]
        for entry in by_tier.values()
    )
    no_composition = [t for t, e in by_tier.items() if not e["composition_signature"]]
    return {
        "prediction": (
            "Q3 -- the kept set's COMPOSITION signature is strong where P4.6's DEMAND signature was "
            "absent: on a mixture the filter selects MODE, not DIFFICULTY"
        ),
        "rule": (
            f"a composition signature iff the exact hypergeometric tail is below {P3_ALPHA}; a "
            f"demand signature iff the difficulty p-value is below {P3_ALPHA} OR the volume 95 % "
            "interval excludes zero.  HELD iff every mixture carries the first and none carries "
            "the second"
        ),
        "outcome": "HELD" if held else "FAILED",
        "by_tier": by_tier,
        "falsifies_r2": bool(no_composition),
        "tiers_without_a_composition_signature": no_composition,
        "falsification_note": (
            "a null composition signature would falsify section 1b's R2 outright and is the most "
            "informative outcome available here; it is reported as such rather than explained away"
        ),
        "multiplicity": (
            "the volume check operates on SETS of draw ids, and on a mixture the same draw can "
            "enter through both components, so its kept and discarded sets lose that multiplicity; "
            "the per-tier count of draws present in both components is reported with the artifact"
        ),
    }


#: Compared leaf by leaf by Gate P1.  Listed rather than discovered so a field ADDED to a cell by a
#: later change cannot silently escape the gate.
PHASE1_CELL_FIELDS = (
    "att_horizon_mean",
    "att_horizon_std",
    "att_horizon_ci95",
    "horizon_vehicle_count_mean",
    "horizon_vehicle_count_std",
    "n_episodes",
    "draw_ids",
    "seeds",
)

PHASE1_COMPARISON_FIELDS = (
    "mean_difference",
    "ci95_low",
    "ci95_high",
    "ci95_width",
    "ci95_half_width",
    "rank_biserial",
    "n_shared_draws",
    "median_difference",
    "wins",
    "losses",
    "ties",
    "mean_left",
    "mean_right",
)


def _compare_leaves(
    candidate: Mapping[str, Any], committed: Mapping[str, Any], fields: Sequence[str], label: str
) -> int:
    """Every field present in the committed record must be EXACTLY equal in the candidate."""
    compared = 0
    for field in fields:
        if field not in committed:
            continue
        if field not in candidate:
            raise ValueError(f"{label}: the regenerated record is missing {field!r}")
        if candidate[field] != committed[field]:
            raise ValueError(
                f"{label}: {field} regenerated as {candidate[field]!r} against the committed "
                f"{committed[field]!r}; the re-used phase-1 column must regenerate bit-identically "
                "or this task is BLOCKED (docs/plans/p4.7.md section 6.3)"
            )
        compared += 1
    return compared


def assert_phase1_reproduces(
    candidate: Mapping[str, Any], committed: Mapping[str, Any]
) -> dict[str, Any]:
    """Gate P1: every re-used phase-1 quantity regenerates bit-identically.

    ⚠️ **Exact equality, never a tolerance.**  The twenty phase-1 method cells, the five phase-1
    behaviour cells and every phase-1 paired comparison are re-derived in this session from the raw
    per-cell JSONs and must equal the committed ``docs/data/p4_6_grid.json`` leaf for leaf.  A
    tolerance here would be a tolerance on *"is this the same measurement?"*, which has no
    scientifically meaningful width -- and P4.6's own review measured that a single swapped episode
    moves a cell mean by 0.01-0.5 ATT, so anything a tolerance would forgive is a real change.

    The mixture tiers are deliberately NOT compared: they do not exist in the committed artifact.
    """
    from offline.method_tier_grid import PHASE1_TIER_ORDER

    phase1 = set(PHASE1_TIER_ORDER)
    cells = behaviour_cells = comparisons = behaviour_comparisons = 0
    leaves = 0

    for tier in sorted(phase1 & set(committed.get("cells_by_tier", {}))):
        for method, cell in sorted(committed["cells_by_tier"][tier].items()):
            label = f"{method}@{tier}"
            got = candidate.get("cells_by_tier", {}).get(tier, {}).get(method)
            if got is None:
                raise ValueError(
                    f"{label}: the regenerated artifact has no such cell, but the committed one "
                    "does; a re-used column may not lose a cell"
                )
            leaves += _compare_leaves(got, cell, PHASE1_CELL_FIELDS, label)
            cells += 1

    for tier in sorted(phase1 & set(committed.get("behaviour_cells", {}))):
        label = f"behaviour@{tier}"
        got = candidate.get("behaviour_cells", {}).get(tier)
        if got is None:
            raise ValueError(f"{label}: the regenerated artifact has no behaviour cell for {tier}")
        leaves += _compare_leaves(got, committed["behaviour_cells"][tier], PHASE1_CELL_FIELDS, label)
        behaviour_cells += 1

    for key, count_name in (("comparisons", "comparisons"), ("behaviour_comparisons", "behaviour")):
        index = {
            (str(c["left_arm"]), str(c["right_arm"])): c for c in candidate.get(key, ())
        }
        for record in committed.get(key, ()):
            arms = (str(record["left_arm"]), str(record["right_arm"]))
            tiers = {arm.partition("@")[2] for arm in arms}
            if not tiers <= phase1:
                continue
            label = f"{arms[0]} vs {arms[1]}"
            if arms not in index:
                raise ValueError(f"{label}: the regenerated artifact is missing this comparison")
            leaves += _compare_leaves(index[arms], record, PHASE1_COMPARISON_FIELDS, label)
            if count_name == "comparisons":
                comparisons += 1
            else:
                behaviour_comparisons += 1

    return {
        "status": "PASS",
        "role": (
            "Gate P1: the twenty re-used phase-1 cells, the five phase-1 behaviour cells and every "
            "phase-1 paired comparison, regenerated in this session and compared to the committed "
            "docs/data/p4_6_grid.json with exact equality"
        ),
        "cells_compared": cells,
        "behaviour_cells_compared": behaviour_cells,
        "comparisons_compared": comparisons,
        "behaviour_comparisons_compared": behaviour_comparisons,
        "numeric_leaves_compared": leaves,
        "tolerance": "none: exact equality on every leaf",
    }


def label_constructed_references(
    payload: MutableMapping[str, Any], references: Mapping[str, Mapping[str, Any]]
) -> None:
    """Stamp every mixture behaviour cell with its ``constructed`` label, or refuse.

    ``method_tier_grid.grid_artifact`` labels a behaviour cell from
    ``BEHAVIOUR_REFERENCE_BY_TIER``, which deliberately declares nothing for a mixture -- a mixture
    has two behaviour policies and no single reference.  P4.7 supplies one of a kind that module
    does not know about, so the label is applied here **and its absence is an error**: an unlabelled
    constructed cell is indistinguishable from a rolled-out one, which is exactly what
    ``BRIEF_19`` section 3 constraint 1 forbids.
    """
    cells = payload.get("behaviour_cells", {})
    for tier in MIXTURE_TIER_ORDER_LOCAL:
        if tier not in cells:
            continue
        reference = references.get(tier)
        if reference is None:
            raise ValueError(
                f"behaviour@{tier} is in the artifact with no constructed-reference record; a "
                "mixture's reference is CONSTRUCTED and must say so, or a reader cannot tell it "
                "from a rolled-out cell (BRIEF_19 section 3, constraint 1)"
            )
        cells[tier]["reference"] = dict(reference)
    for tier, cell in cells.items():
        if cell.get("reference") is None:
            raise ValueError(
                f"behaviour@{tier} carries no reference record at all, so its provenance is unstated"
            )


def mixture_grid_artifact(
    declaration: Mapping[str, Any],
    training: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    gate: Mapping[str, Any],
    gate_d: Mapping[str, Any],
    episodes_by_arm: Mapping[str, Sequence[Any]],
    references: Mapping[str, Mapping[str, Any]],
    committed_grid: Mapping[str, Any],
    *,
    inputs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """P4.7's reported artifact: phase 1 re-used under Gate P1, phase 2 measured, Q1-Q3 scored."""
    from offline.method_tier_grid import assert_no_verdicts, grid_artifact

    payload = grid_artifact(declaration, training, diagnostics, gate, episodes_by_arm, inputs=inputs)
    label_constructed_references(payload, references)

    payload["inner_format_version"] = payload["format_version"]
    payload["format_version"] = MIXTURE_GRID_FORMAT_VERSION
    payload["role"] = (
        "P4.7: the three OffLight-matched mixture tiers (33/50/67 % expert) beside P4.6's five "
        "single-controller tiers, at equal training-set size, on the registered held-out pool.  "
        "The mixture axis varies COMPOSITION at roughly constant state coverage, which is the "
        "design PROJECT_PLAN section 1's constraint (a) says the single-controller ladder cannot "
        "provide"
    )
    payload["gate_d"] = dict(gate_d)
    payload["gate_p1"] = assert_phase1_reproduces(payload, committed_grid)
    payload["constructed_references"] = {
        tier: dict(reference) for tier, reference in references.items()
    }
    payload["mixture_predictions"] = {
        "Q1": score_q1(declaration),
        "Q2": score_q2(payload["cells_by_tier"], _comparison_objects(payload["comparisons"])),
        "Q3": score_q3(declaration, diagnostics),
    }
    payload["reused_declaration"] = {
        "artifact": "docs/data/p4_6_declaration.json",
        "tiers": ["mappo500", "maxpressure", "fixedtime"],
        "note": (
            "three phase-1 tiers are re-used whole and are declared in P4.6's artifact, not in "
            "P4.7's; this artifact re-declares only the tiers it composes from (mappo1000 and "
            "random) and the three mixtures"
        ),
    }
    payload["inherited_predictions"] = _p4_6_prediction_sidecar(payload, committed_grid)
    assert_no_verdicts(payload)
    return payload


class _Comparison:
    """A comparison read back from JSON, in the shape :func:`score_q2` reads."""

    def __init__(self, record: Mapping[str, Any]) -> None:
        self.left_arm = record["left_arm"]
        self.right_arm = record["right_arm"]
        self.mean_difference = record["mean_difference"]
        self.ci95_low = record["ci95_low"]
        self.ci95_high = record["ci95_high"]
        self.ci95_width = record["ci95_width"]
        self.rank_biserial = record["rank_biserial"]


def _comparison_objects(records: Sequence[Mapping[str, Any]]) -> list[_Comparison]:
    return [_Comparison(record) for record in records]


def _p4_6_prediction_sidecar(
    payload: Mapping[str, Any], committed: Mapping[str, Any]
) -> dict[str, Any]:
    """P4.6's P1 and P2 at their newly scorable state, BESIDE their original outcomes.

    Coordinator RULING 2 of 2026-08-15.  **Both are reported and neither replaces the other**, and
    the full-design outcome may not be used to rescue a prediction P4.6 recorded as failed.

    **Why this is not optional stopping**, stated because a referee will ask: the rule was fixed
    before P4.6 ran; the mixture tiers were always part of the registered design (``score_p2``'s own
    ``full_rule`` says *"scorable only when all three mixture tiers are present"*, written before
    phase 1 ran); and the phase split was declared in ``BRIEF_17`` section 3 before any number
    existed.  **A pre-declared design completing, not a run continued until it passed** -- the
    distinguishing feature being that the stopping point was written down first and no datum could
    have moved it.
    """
    return {
        "ruling": "coordinator RULING 2 of 2026-08-15; docs/data/p4_6_grid.json is not modified",
        "as_scored_by_p4_6": {
            name: {
                key: committed["predictions"][name].get(key)
                for key in ("outcome", "full_outcome", "partial_outcome")
                if key in committed["predictions"][name]
            }
            for name in ("P1", "P2", "P3")
        },
        "as_scored_with_the_full_design": {
            name: {
                key: payload["predictions"][name].get(key)
                for key in ("outcome", "full_outcome", "partial_outcome")
                if key in payload["predictions"][name]
            }
            for name in ("P1", "P2", "P3")
        },
        "tiers_available_to_p4_6": committed["tiers_present"],
        "tiers_available_now": payload["tiers_present"],
        "binding": (
            "both outcomes are reported and neither replaces the other; a prediction P4.6 recorded "
            "as FAILED may not be rescued by the fuller scoring, which is a second and "
            "better-powered reading of the SAME registered rule and is labelled as one"
        ),
        "not_optional_stopping": (
            "the rule was fixed before P4.6 ran, the mixture tiers were always part of the "
            "registered design (score_p2's full_rule was written before phase 1 ran) and the phase "
            "split was declared in BRIEF_17 section 3 before any number existed: a pre-declared "
            "design completing, not a run continued until it passed"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI: ``gate-d`` today; phase 2's reporting subcommands are added as they land."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.mixture_tiers",
        description="P4.7: the mixture tiers, phase 2 of the method x tier grid.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    gate_d = sub.add_parser(
        "gate-d", help="prove the regenerated held-out demand reproduces the surviving draws"
    )
    gate_d.add_argument("--reference-root", required=True)
    gate_d.add_argument("--candidate-root", required=True)
    gate_d.add_argument("--draws", type=int, nargs="+", required=True)
    gate_d.add_argument("--scenario-key", default="cityflow1x1")
    gate_d.add_argument("--out", required=True)

    reference = sub.add_parser(
        "construct-reference",
        help="build a mixture tier's CONSTRUCTED behaviour reference; no rollout, zero compute",
    )
    reference.add_argument("--tier", required=True)
    reference.add_argument(
        "--grid", default="docs/data/p4_6_grid.json", help="the committed phase-1 grid artifact"
    )
    reference.add_argument(
        "--expert-source",
        default="docs/data/p4_heldout_thresholds.json",
        help="the PRIMARY artifact of the mappo1000 held-out cell, cross-checked against the grid",
    )
    reference.add_argument("--expert-source-arm", default="mappo1000")
    reference.add_argument(
        "--random-source",
        default="/home/filip/rltraffic/output/p4_6/eval_random_behaviour.json",
        help="the PRIMARY artifact of the random held-out cell, cross-checked against the grid",
    )
    reference.add_argument("--random-source-arm", default="behaviour@random")
    reference.add_argument("--out", required=True)

    report = sub.add_parser(
        "report", help="assemble docs/data/p4_7_grid.json: phase 1 re-used, phase 2 measured"
    )
    report.add_argument("--work-dir", default="output/p4_7")
    report.add_argument("--out-dir", default="docs/data")
    report.add_argument("--artifact-prefix", default="p4_7")
    report.add_argument("--committed-grid", default="docs/data/p4_6_grid.json")
    report.add_argument("--baselines", default="docs/data/p4_4_baselines.json")
    report.add_argument("--thresholds", default="docs/data/p4_heldout_thresholds.json")
    return parser


#: Where each re-used phase-1 arm's episodes come from.  ``mappo1000``'s four method cells are
#: P4.4's merged artifact under contract C9's alias; the other four tiers' are the raw per-cell
#: JSONs P4.6 secured, copied here and re-verified against ``output/SHA256SUMS_p4_6.txt``.
PHASE1_BEHAVIOUR_SOURCES = {
    "mappo1000": ("thresholds", "mappo1000"),
    "mappo500": ("thresholds", "mappo500"),
    "maxpressure": ("thresholds", "maxpressure"),
    "fixedtime": ("copied", "behaviour@fixedtime"),
    "random": ("copied", "behaviour@random"),
}


def _load_report_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Every input the report reads, loaded before anything is computed or written."""
    from offline.method_tier_grid import (
        METHODS,
        MIXTURE_TIER_ORDER,
        PHASE1_TIER_ORDER,
        REUSED_ARM_KEYS,
        REUSED_TIER,
        arm_key,
    )

    work = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    prefix = str(args.artifact_prefix)

    def read(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist; the report reads it and cannot proceed")
        return json.loads(path.read_text(encoding="utf-8"))

    loaded = {
        "declaration": read(out_dir / f"{prefix}_declaration.json"),
        "diagnostics": read(out_dir / f"{prefix}_selection_diagnostics.json"),
        "training": read(out_dir / f"{prefix}_training.json"),
        "gate": read(work / "gate.json"),
        "gate_d": read(work / "gate_d.json"),
        "committed": read(Path(args.committed_grid)),
        "baselines": read(Path(args.baselines)),
        "thresholds": read(Path(args.thresholds)),
    }

    episodes: dict[str, list[dict[str, Any]]] = {}
    inputs: list[Mapping[str, Any]] = [
        loaded["gate"], loaded["gate_d"], loaded["declaration"], loaded["diagnostics"],
        loaded["training"], loaded["baselines"],
    ]
    for tier in PHASE1_TIER_ORDER:
        for method in METHODS:
            arm = arm_key(method, tier)
            if tier == REUSED_TIER:
                source = REUSED_ARM_KEYS[method]
                records = [r for r in loaded["baselines"]["episodes"] if r["arm"] == source]
                if not records:
                    raise ValueError(f"{args.baselines} carries no episodes for {source!r}")
            else:
                payload = read(work / f"phase1_eval_{tier}_{method}.json")
                inputs.append(payload)
                records = payload["episodes"]
            episodes[arm] = [{**r, "arm": arm} for r in records]
        kind, source_arm = PHASE1_BEHAVIOUR_SOURCES[tier]
        arm = arm_key("behaviour", tier)
        if kind == "thresholds":
            records = [r for r in loaded["thresholds"]["episodes"] if r["arm"] == source_arm]
            if not records:
                raise ValueError(f"{args.thresholds} carries no episodes for {source_arm!r}")
        else:
            payload = read(work / f"phase1_eval_{tier}_behaviour.json")
            inputs.append(payload)
            records = payload["episodes"]
        episodes[arm] = [{**r, "arm": arm} for r in records]

    references: dict[str, dict[str, Any]] = {}
    for tier in MIXTURE_TIER_ORDER:
        for method in METHODS:
            payload = read(work / f"eval_{tier}_{method}.json")
            inputs.append(payload)
            episodes[arm_key(method, tier)] = payload["episodes"]
        payload = read(work / f"eval_{tier}_behaviour.json")
        inputs.append(payload)
        episodes[arm_key("behaviour", tier)] = payload["episodes"]
        references[tier] = {
            **payload["behaviour_reference"],
            "assignment": payload["assignment"],
            "bracket": {
                name: cell["att_horizon_mean"] for name, cell in payload["bracket"].items()
            },
            "cross_checked_against": payload.get("cross_checked_against"),
        }
    loaded["episodes"] = episodes
    loaded["references"] = references
    loaded["inputs"] = inputs
    return loaded


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    from offline.dt_gate import write_json_atomic

    args = build_parser().parse_args(argv)
    if args.command == "gate-d":
        payload = draw_identity_artifact(
            args.reference_root,
            args.candidate_root,
            args.draws,
            scenario_key=args.scenario_key,
        )
        write_json_atomic(payload, Path(args.out))
        for record in payload["draws"]:
            print(
                f"draw {record['draw_id']}: byte-identical {record['byte_identical']}  "
                f"path-normalised {record['path_normalised']}",
                flush=True,
            )
        print(
            f"GATE D PASS: {len(payload['draws'])} of the "
            f"{len(payload['survivors_available'])} surviving held-out draws reproduce",
            flush=True,
        )
        return 0

    if args.command == "construct-reference":
        from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS

        grid = json.loads(Path(args.grid).read_text(encoding="utf-8"))
        expert = component_from_sources(
            grid,
            CONSTRUCTED_REFERENCE_ARMS[0],
            json.loads(Path(args.expert_source).read_text(encoding="utf-8")),
            args.expert_source_arm,
        )
        random_pool = component_from_sources(
            grid,
            CONSTRUCTED_REFERENCE_ARMS[1],
            json.loads(Path(args.random_source).read_text(encoding="utf-8")),
            args.random_source_arm,
        )
        payload = constructed_reference_artifact(
            args.tier,
            expert,
            random_pool,
            seeds=list(TRAINING_SEEDS),
            draws=list(HELD_OUT_DRAWS),
        )
        payload["cross_checked_against"] = {
            CONSTRUCTED_REFERENCE_ARMS[0]: f"{args.expert_source}::{args.expert_source_arm}",
            CONSTRUCTED_REFERENCE_ARMS[1]: f"{args.random_source}::{args.random_source_arm}",
        }
        write_json_atomic(payload, Path(args.out))
        cell = payload["cell"]
        low = payload["bracket"][CONSTRUCTED_REFERENCE_ARMS[0]]["att_horizon_mean"]
        high = payload["bracket"][CONSTRUCTED_REFERENCE_ARMS[1]]["att_horizon_mean"]
        print(
            f"{payload['arm']} CONSTRUCTED: att_horizon {cell['att_horizon_mean']:.4f} +/- "
            f"{cell['att_horizon_ci95']:.4f}  n={cell['n_episodes']}  "
            f"{payload['behaviour_reference']['expert_draws_per_seed']} expert draws per seed\n"
            f"  bracket: mappo1000 {low:.4f} < {cell['att_horizon_mean']:.4f} < random {high:.4f}",
            flush=True,
        )
        return 0

    if args.command == "report":
        from offline.method_tier_grid import METHODS, MIXTURE_TIER_ORDER
        from offline.method_tier_grid import _episodes_from_records

        loaded = _load_report_inputs(args)
        episodes_by_arm = {
            arm: _episodes_from_records(records, arm)
            for arm, records in loaded["episodes"].items()
        }
        payload = mixture_grid_artifact(
            loaded["declaration"],
            loaded["training"],
            loaded["diagnostics"],
            loaded["gate"],
            loaded["gate_d"],
            episodes_by_arm,
            loaded["references"],
            loaded["committed"],
            inputs=loaded["inputs"],
        )
        write_json_atomic(
            payload, Path(args.out_dir) / f"{args.artifact_prefix}_grid.json"
        )
        gate_p1 = payload["gate_p1"]
        print(
            f"GATE P1 {gate_p1['status']}: {gate_p1['cells_compared']} phase-1 cells, "
            f"{gate_p1['behaviour_cells_compared']} behaviour cells, "
            f"{gate_p1['comparisons_compared']} comparisons and "
            f"{gate_p1['behaviour_comparisons_compared']} behaviour comparisons regenerate "
            f"bit-identically ({gate_p1['numeric_leaves_compared']} leaves, no tolerance)",
            flush=True,
        )
        header = "tier          behaviour[1000-1099]  " + "  ".join(
            f"{method}[1000-1099]" for method in METHODS
        )
        print(header, flush=True)
        for tier in payload["tiers_present"]:
            row = payload["cells_by_tier"][tier]
            reference = payload["behaviour_cells"].get(tier)
            label = "" if reference is None else (
                " (constructed)" if reference["reference"]["source"] == "constructed" else ""
            )
            behaviour = float("nan") if reference is None else reference["att_horizon_mean"]
            print(
                f"{tier:12s} {behaviour:12.4f}{label:14s}  "
                + "  ".join(
                    f"{method} {row[method]['att_horizon_mean']:8.4f}" for method in METHODS
                ),
                flush=True,
            )
        for name in MIXTURE_PREDICTION_ORDER:
            prediction = payload["mixture_predictions"][name]
            print(f"{name}: {prediction['outcome']}", flush=True)
        for name in ("P1", "P2", "P3"):
            inherited = payload["inherited_predictions"]
            print(
                f"{name}: P4.6 {inherited['as_scored_by_p4_6'][name]} -> full design "
                f"{inherited['as_scored_with_the_full_design'][name]}",
                flush=True,
            )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
