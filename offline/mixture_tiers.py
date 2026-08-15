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
from collections.abc import Mapping, Sequence
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
    return parser


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


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
