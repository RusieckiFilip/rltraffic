"""Re-run the 2026-08-06 collection campaign at corpus format v1.1, into ``datasets_v11/``.

Every command is **derived from the existing ``datasets/*/manifest.json``**, never retyped.
Each manifest's ``run_metadata`` already records the policy, checkpoint, draw ids,
episodes-per-draw, base seed and every env setting that produced that tier, so the campaign
is reconstructed from the corpus it is reproducing.  Exactly two things change:

* ``--out-dir``      ``datasets/X``  ->  ``datasets_v11/X``
* ``--fixed-time-k`` 4 -> the tuned value from ``docs/data/fixed_time_sweep/README.md``

Retyping 69 commands is precisely the transcription error the C8 migration is written to
avoid one module over; there is no reason to accept it here.

    python -m offline.recollect_v11                 # dry run (default): prints, runs nothing
    python -m offline.recollect_v11 --run           # execute, sequentially, aborting on error

**This is hours of simulation.** Run it in tmux, not in a session.

DERIVED IS NOT THE SAME AS TRUSTED
----------------------------------
Deriving from the manifests inherits anything wrong in them.  That risk is bounded by
asserting, before any command is issued, every invariant this project has actually written
down (:data:`INVARIANTS`): ``global_reward_weight == 0.0``, ``local_reward_fn ==
"queue_length"``, ``base_seed == 1000``, ``delta_time == 10``, ``max_steps == 360``,
exactly one metric key, and every draw id inside the registered training pool 1-999.  What
remains is not "anything wrong" but "wrong in a way we never recorded", which is a much
smaller set.

The metric-key invariant is read from an **episode ``.npz``**, not from the manifest: the
manifest records the *requested* list, which is ``null`` on every one of these runs because
the set is derived by the env.  Checking the manifest would assert nothing at all.

**The draw-pool check is the one that cannot be relaxed.** D4 / ``PREREGISTRATION.md`` §5
reserves draws 1000-1099 as the held-out evaluation pool. Collecting them into a training
corpus is not recoverable by re-labelling -- it means the model was trained on its own test
set.

CAMPAIGN-SCRIPT RULES (PROJECT_PLAN §7, added 2026-08-06 after this campaign failed)
------------------------------------------------------------------------------------
The 2026-08-06 run had none of these: one tier was refused by the out-dir guard, fifteen-plus
raised ``ValueError`` from contract C8, and the loop ran on to a clean-looking end.  Overnight
it would have reported success with half the corpus missing.

1. **Abort on the first failure.**  A non-zero exit from any collection stops the campaign.
2. **Check every ``out_dir`` is absent or empty first**, for all 69 runs, before issuing any
   command -- a command that trips the populated-out-dir guard wastes its slot.
3. **Assert completions == requests at the end**, and additionally that each tier's episode
   count matches the manifest it was derived from.  A loop that ran is not a corpus that exists.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: Tuned cycle multiplier per scenario (docs/data/fixed_time_sweep/README.md, 2026-08-06).
#: The ONLY intended behavioural difference between datasets/ and datasets_v11/.
TUNED_FIXED_TIME_K: dict[str, int] = {
    "cf_hz1x1": 6,
    "cf_grid4x4": 1,
    "cf_cologne3": 3,
}

#: Registered training draw pool (PREREGISTRATION.md §5 / D4). 1000-1099 is held out.
TRAINING_DRAW_MIN = 1
TRAINING_DRAW_MAX = 999

#: run_metadata invariants asserted before any command is issued.
INVARIANTS: dict[str, Any] = {
    "global_reward_weight": 0.0,
    "local_reward_fn": "queue_length",
    "global_reward_fn": "queue_length",
    "base_seed": 1000,
    "delta_time": 10,
    "max_steps": 360,
    "control_mode": "acyclic",
    "backend": "cityflow",
}

SOURCE_ROOT = Path("datasets")
TARGET_ROOT = Path("datasets_v11")

__all__ = ["CampaignError", "RunSpec", "build_campaign", "main"]


class CampaignError(RuntimeError):
    """Raised before anything runs when a derived command fails an invariant."""


@dataclass(frozen=True)
class RunSpec:
    """One collection command, derived from one existing manifest."""

    name: str
    scenario: str
    out_dir: Path
    argv: tuple[str, ...]
    expected_episodes: int
    policy: str
    checkpoint: str | None
    draw_ids: tuple[int, ...]
    episodes_per_draw: int
    fixed_time_k: int | None
    provenance: dict[str, Any] = field(repr=False, default_factory=dict)

    def describe(self) -> str:
        """The per-run line the dry run prints; every field a reviewer asked to see."""
        draws = f"{len(self.draw_ids)} draws [{self.draw_ids[0]}..{self.draw_ids[-1]}]"
        return (
            f"{self.name:34s} scen={self.scenario:12s} policy={self.policy:11s} "
            f"{draws:26s} eps/draw={self.episodes_per_draw} "
            f"base_seed={self.provenance['base_seed']} "
            f"local_reward_fn={self.provenance['local_reward_fn']} "
            f"global_reward_weight={self.provenance['global_reward_weight']} "
            f"k={self.fixed_time_k} "
            f"metric_keys={self.provenance['n_metric_keys']} "
            f"-> {self.expected_episodes} episodes\n"
            f"    checkpoint={self.checkpoint}"
        )


def _scenario_of(run_name: str) -> str:
    """``cf_hz1x1__mappo1000__seed101`` -> ``cf_hz1x1``."""
    return run_name.split("__", 1)[0]


def _episode_metric_key_count(run_dir: Path) -> int:
    """Metric keys actually stored in this run's episodes.

    Read from an ``.npz`` because ``run_metadata["metrics"]`` is ``null`` on every one of
    these runs -- the env derives the set -- so the manifest cannot answer this.
    """
    episodes = sorted(run_dir.glob("ep*.npz"))
    if not episodes:
        raise CampaignError(f"{run_dir}: no episode files, cannot verify the metric set")
    with np.load(episodes[0]) as data:
        return int(len(data["metric_keys"]))


def _check_invariants(name: str, meta: dict[str, Any], n_metric_keys: int) -> None:
    for key, expected in INVARIANTS.items():
        actual = meta.get(key)
        if actual != expected:
            raise CampaignError(
                f"{name}: run_metadata[{key!r}] is {actual!r}, expected {expected!r}. "
                "The source corpus does not match the settings this campaign is "
                "declared to reproduce; refusing to derive a command from it."
            )
    if n_metric_keys != 1:
        raise CampaignError(
            f"{name}: episodes carry {n_metric_keys} metric keys, expected exactly 1. "
            "The metric set is frozen for the lifetime of the MAPPO checkpoints "
            "collected against it (contract C8); a different width means this corpus "
            "was not collected under the settings this campaign reproduces."
        )


def _check_draw_pool(name: str, draw_ids: Sequence[int]) -> None:
    outside = [d for d in draw_ids if not TRAINING_DRAW_MIN <= d <= TRAINING_DRAW_MAX]
    if outside:
        raise CampaignError(
            f"{name}: draw id(s) {sorted(set(outside))} lie outside the registered "
            f"training pool {TRAINING_DRAW_MIN}-{TRAINING_DRAW_MAX} "
            "(PREREGISTRATION.md §5 / D4). Draws 1000-1099 are the held-out evaluation "
            "pool and must never enter a training corpus, for any method including "
            "baselines. This is not recoverable by re-labelling."
        )


def _draw_flags(draw_ids: Sequence[int]) -> list[str]:
    """Contiguous runs use the half-open range flag; anything else is listed explicitly."""
    ids = list(draw_ids)
    if ids == list(range(ids[0], ids[0] + len(ids))):
        return ["--flow-draws-range", str(ids[0]), str(ids[0] + len(ids))]
    return ["--flow-draws", *[str(i) for i in ids]]


def build_campaign(
    source_root: Path = SOURCE_ROOT, target_root: Path = TARGET_ROOT
) -> list[RunSpec]:
    """Derive every command from the source corpus. Validates; runs nothing."""
    manifests = sorted(source_root.glob("*/manifest.json"))
    if not manifests:
        raise CampaignError(f"no manifests under {source_root}; nothing to derive from")

    specs: list[RunSpec] = []
    for manifest_path in manifests:
        run_dir = manifest_path.parent
        name = run_dir.name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        meta = manifest["run_metadata"]

        n_metric_keys = _episode_metric_key_count(run_dir)
        _check_invariants(name, meta, n_metric_keys)

        draw_ids = tuple(d for d in meta["flow_draw_ids"] if d is not None)
        if not draw_ids:
            raise CampaignError(
                f"{name}: nominal run with no flow draws. The v1.1 campaign reproduces "
                "the randomised-draw corpus only; a nominal (draw 0) run is a separate "
                "experiment and must not be folded in silently."
            )
        _check_draw_pool(name, draw_ids)

        scenario = _scenario_of(name)
        policy = str(meta["behavior_policy"])
        fixed_time_k: int | None = None
        if policy == "fixedtime":
            if scenario not in TUNED_FIXED_TIME_K:
                raise CampaignError(
                    f"{name}: no tuned k recorded for scenario {scenario!r}; "
                    "see docs/data/fixed_time_sweep/README.md"
                )
            fixed_time_k = TUNED_FIXED_TIME_K[scenario]

        out_dir = target_root / name
        argv = [
            sys.executable, "-m", "offline.collect",
            "--backend", str(meta["backend"]),
            "--env-config", str(meta["env_paths"]["config"]),
            "--policy", policy,
            "--episodes", str(int(meta["episodes"])),
            "--base-seed", str(int(meta["base_seed"])),
            "--max-steps", str(int(meta["max_steps"])),
            "--delta-time", str(int(meta["delta_time"])),
            "--control-mode", str(meta["control_mode"]),
            "--global-reward-fn", str(meta["global_reward_fn"]),
            "--local-reward-fn", str(meta["local_reward_fn"]),
            "--global-reward-weight", str(float(meta["global_reward_weight"])),
            "--state-features", *[str(f) for f in meta["state_features"]],
            "--out-dir", str(out_dir),
        ]
        if meta.get("checkpoint"):
            argv += ["--checkpoint", str(meta["checkpoint"])]
        if fixed_time_k is not None:
            argv += ["--fixed-time-k", str(fixed_time_k)]
        argv += _draw_flags(draw_ids)

        specs.append(
            RunSpec(
                name=name,
                scenario=scenario,
                out_dir=out_dir,
                argv=tuple(argv),
                expected_episodes=len(manifest["episodes"]),
                policy=policy,
                checkpoint=meta.get("checkpoint"),
                draw_ids=draw_ids,
                episodes_per_draw=int(meta["episodes"]),
                fixed_time_k=fixed_time_k,
                provenance={
                    "base_seed": meta["base_seed"],
                    "local_reward_fn": meta["local_reward_fn"],
                    "global_reward_weight": meta["global_reward_weight"],
                    "n_metric_keys": n_metric_keys,
                },
            )
        )
    return specs


def _check_out_dirs_free(specs: Sequence[RunSpec]) -> None:
    """Every target must be absent or empty -- checked for ALL runs before any runs."""
    occupied = [
        spec.name
        for spec in specs
        if spec.out_dir.is_dir() and any(spec.out_dir.iterdir())
    ]
    if occupied:
        raise CampaignError(
            f"{len(occupied)} target directory/ies already contain files: "
            f"{occupied[:5]}{'...' if len(occupied) > 5 else ''}. "
            "offline.collect refuses a populated out-dir, so those commands would "
            "waste their slot. Move or remove them, or pass --overwrite deliberately."
        )


def _completed_episodes(out_dir: Path) -> int:
    manifest = out_dir / "manifest.json"
    if not manifest.is_file():
        return 0
    return len(json.loads(manifest.read_text(encoding="utf-8"))["episodes"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m offline.recollect_v11",
        description="Re-run the collection campaign at format v1.1 into datasets_v11/.",
    )
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--target-root", default=str(TARGET_ROOT))
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute; without it this is a dry run that prints and runs nothing",
    )
    parser.add_argument("--only", default=None, help="substring filter on the run name")
    args = parser.parse_args(argv)

    try:
        specs = build_campaign(Path(args.source_root), Path(args.target_root))
    except CampaignError as exc:
        print(f"ABORT (nothing run): {exc}", flush=True)
        return 1

    if args.only:
        specs = [s for s in specs if args.only in s.name]
        if not specs:
            print(f"ABORT: --only {args.only!r} matched no run", flush=True)
            return 1

    total_episodes = sum(s.expected_episodes for s in specs)
    print(f"campaign: {len(specs)} runs, {total_episodes} episodes\n", flush=True)
    for spec in specs:
        print(spec.describe(), flush=True)

    try:
        _check_out_dirs_free(specs)
    except CampaignError as exc:
        print(f"\nABORT (nothing run): {exc}", flush=True)
        return 1

    if not args.run:
        print(
            f"\nDRY RUN -- nothing executed. {len(specs)} runs would produce "
            f"{total_episodes} episodes. Re-run with --run, inside tmux.",
            flush=True,
        )
        return 0

    completed = 0
    for index, spec in enumerate(specs, start=1):
        print(f"\n[{index}/{len(specs)}] {spec.name}", flush=True)
        print("  " + " ".join(spec.argv), flush=True)
        result = subprocess.run(spec.argv)
        if result.returncode != 0:
            # Rule 1: abort on the first failure. The 2026-08-06 campaign did not, and
            # ran on to a clean-looking end with half the corpus missing.
            print(
                f"\nABORT: {spec.name} exited {result.returncode}. "
                f"{completed}/{len(specs)} runs completed before this.",
                flush=True,
            )
            return 1
        got = _completed_episodes(spec.out_dir)
        if got != spec.expected_episodes:
            print(
                f"\nABORT: {spec.name} wrote {got} episodes, expected "
                f"{spec.expected_episodes} (from the source manifest).",
                flush=True,
            )
            return 1
        completed += 1

    # Rule 3: completions == requests, asserted rather than assumed.
    if completed != len(specs):
        print(
            f"\nFAILED: {completed} runs completed but {len(specs)} were requested.",
            flush=True,
        )
        return 1
    written = sum(_completed_episodes(s.out_dir) for s in specs)
    if written != total_episodes:
        print(
            f"\nFAILED: {written} episodes on disk but {total_episodes} were requested.",
            flush=True,
        )
        return 1
    print(
        f"\nOK: {completed}/{len(specs)} runs, {written}/{total_episodes} episodes.\n"
        "Next: python -m offline.compare_corpora  (the bit-identity gate)",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
