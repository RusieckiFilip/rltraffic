"""Validation gate: ``datasets_v11/`` must reproduce ``datasets/`` bit-for-bit, bar one tier.

Format v1.1 adds one observation array and changes nothing else, so re-collecting the same
policies against the same demand must reproduce **every trajectory-defining array
bit-identically**.  ``fixedtime`` is the only tier permitted to differ, because its cycle
multiplier ``k`` was retuned (4 -> 6/1/3, ``docs/data/fixed_time_sweep/README.md``).

Any other difference means something drifted between the two collections -- a changed
default, a different checkpoint, a numpy or engine change -- and the v1.1 corpus must not be
consumed until it is explained.

    python -m offline.compare_corpora                       # gate; exit 0 = pass

WHAT IS COMPARED, AND WHY THE LIST IS EXPLICIT
----------------------------------------------
:data:`TRAJECTORY_ARRAYS` names the global arrays; per-intersection arrays are enumerated
from the file's own ``ix_ids``.  An explicit list is auditable in a way that "compare every
key" is not -- but "every key" catches things a list forgets, so the key **sets** are
compared too: they must differ by exactly ``{"att_per_step"}``, which is what makes "v1.1 is
v1.0 plus one field" a checked claim rather than a promise.

Comparison is ``np.array_equal``, never ``allclose``.  These are re-simulations of a
deterministic pipeline; a tolerance here would hide exactly the drift the gate exists to
find.  ``episode_sha256`` is compared as a second, independent route: it is computed at
write time over ``action`` + ``global_reward`` bytes and deliberately excludes
``att_per_step``, so it is directly comparable across the two format versions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

#: Global arrays that define a trajectory. ``att_per_step`` is absent by design (v1.1-only)
#: and ``format_version`` is expected to differ.
TRAJECTORY_ARRAYS: tuple[str, ...] = (
    "ix_ids", "lane_ids", "metric_keys",
    "vehicle_count", "sim_time", "step", "metrics",
    "lane_vehicle_count", "lane_waiting_vehicle_count",
    "global_reward", "episode_length", "terminated", "truncated",
    "engine_seed", "flow_draw",
)

#: Per-intersection arrays, formatted with the intersection index.
PER_IX_ARRAYS: tuple[str, ...] = (
    "ix{i}_state", "ix{i}_avail_mask", "ix{i}_current_phase",
    "ix{i}_time_in_phase", "ix{i}_action", "ix{i}_local_reward",
)

#: Arrays v1.1 adds. The key sets of the two files must differ by exactly this.
V11_ONLY: frozenset[str] = frozenset({"att_per_step"})

#: The only tier allowed to differ, and only because k was retuned.
EXEMPT_POLICY = "fixedtime"

__all__ = ["RunComparison", "compare_run", "compare_corpora", "main"]


@dataclass(frozen=True)
class RunComparison:
    """Outcome for one collection run."""

    name: str
    exempt: bool
    episodes: int
    differing_arrays: tuple[str, ...]
    differing_episodes: int
    error: str | None = None

    @property
    def identical(self) -> bool:
        return self.error is None and not self.differing_arrays


def _episode_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("ep*.npz"))


def compare_run(old_dir: Path, new_dir: Path) -> RunComparison:
    """Compare one run's episodes array by array."""
    name = new_dir.name
    exempt = f"__{EXEMPT_POLICY}" in name

    old_files = _episode_files(old_dir)
    new_files = _episode_files(new_dir)
    if not new_files:
        return RunComparison(name, exempt, 0, (), 0, "no episodes in the v1.1 run")
    if len(old_files) != len(new_files):
        return RunComparison(
            name, exempt, len(new_files), (), 0,
            f"episode count differs: {len(old_files)} vs {len(new_files)}",
        )

    differing: set[str] = set()
    differing_episodes = 0
    for old_path, new_path in zip(old_files, new_files):
        if old_path.name != new_path.name:
            return RunComparison(
                name, exempt, len(new_files), (), 0,
                f"filenames differ: {old_path.name} vs {new_path.name}",
            )
        with np.load(old_path) as old, np.load(new_path) as new:
            added = set(new.files) - set(old.files)
            removed = set(old.files) - set(new.files)
            if added != set(V11_ONLY) or removed:
                return RunComparison(
                    name, exempt, len(new_files), (), 0,
                    f"{new_path.name}: key sets differ by more than {sorted(V11_ONLY)} "
                    f"(added {sorted(added)}, removed {sorted(removed)})",
                )

            names = list(TRAJECTORY_ARRAYS)
            for i in range(len(old["ix_ids"])):
                names += [tpl.format(i=i) for tpl in PER_IX_ARRAYS]

            episode_differs = False
            for array_name in names:
                if not np.array_equal(old[array_name], new[array_name]):
                    differing.add(array_name)
                    episode_differs = True
            if episode_differs:
                differing_episodes += 1

    return RunComparison(
        name, exempt, len(new_files), tuple(sorted(differing)), differing_episodes
    )


def _manifest_hashes(run_dir: Path) -> dict[str, str]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {e["filename"]: e["episode_sha256"] for e in manifest["episodes"]}


def compare_corpora(old_root: Path, new_root: Path) -> list[RunComparison]:
    """Compare every run present in the v1.1 corpus against its v1.0 counterpart."""
    results: list[RunComparison] = []
    for new_dir in sorted(p for p in new_root.iterdir() if p.is_dir()):
        old_dir = old_root / new_dir.name
        if not old_dir.is_dir():
            results.append(
                RunComparison(new_dir.name, False, 0, (), 0, "no v1.0 counterpart")
            )
            continue
        result = compare_run(old_dir, new_dir)

        # Independent second route: the write-time digest over action + global_reward,
        # which excludes att_per_step and so compares across format versions.
        if result.error is None and not result.exempt:
            old_hashes, new_hashes = _manifest_hashes(old_dir), _manifest_hashes(new_dir)
            mismatched = [
                f for f, h in new_hashes.items() if old_hashes.get(f) != h
            ]
            if mismatched and not result.differing_arrays:
                result = RunComparison(
                    result.name, result.exempt, result.episodes, ("episode_sha256",),
                    len(mismatched),
                    "arrays matched but episode_sha256 differs -- the digest and the "
                    "array comparison disagree, which is a defect in one of them",
                )
        results.append(result)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m offline.compare_corpora",
        description="Gate: datasets_v11/ must reproduce datasets/ bar the fixedtime tier.",
    )
    parser.add_argument("--old-root", default="datasets")
    parser.add_argument("--new-root", default="datasets_v11")
    args = parser.parse_args(argv)

    old_root, new_root = Path(args.old_root), Path(args.new_root)
    if not new_root.is_dir():
        print(f"ERROR: {new_root} does not exist -- run the campaign first", flush=True)
        return 2

    results = compare_corpora(old_root, new_root)
    if not results:
        print(f"ERROR: no runs found under {new_root}", flush=True)
        return 2

    failures: list[RunComparison] = []
    for r in results:
        if r.error and not r.exempt:
            status, detail = "ERROR", r.error
            failures.append(r)
        elif r.exempt:
            status = "EXEMPT"
            detail = (
                "fixedtime, k retuned -- difference expected"
                + (f" ({r.differing_episodes}/{r.episodes} episodes differ)"
                   if r.differing_arrays else " (but NOTHING differs, see below)")
            )
        elif r.identical:
            status, detail = "IDENTICAL", f"{r.episodes} episodes"
        else:
            status = "DIFFERS"
            detail = (
                f"{r.differing_episodes}/{r.episodes} episodes; arrays: "
                f"{list(r.differing_arrays)}"
            )
            failures.append(r)
        print(f"  {status:10s} {r.name:36s} {detail}", flush=True)

    exempt = [r for r in results if r.exempt]
    unchanged = [r for r in results if not r.exempt]
    print(
        f"\n{len(unchanged)} unchanged tier(s), {len(exempt)} exempt (fixedtime), "
        f"{len(failures)} failure(s).",
        flush=True,
    )

    # A fixedtime tier that does NOT differ is also a failure: k went 4 -> 6/1/3, so
    # identical output means the tuned value never reached the collector.
    silent_exempt = [r for r in exempt if r.identical]
    if silent_exempt:
        print(
            f"FAIL: {[r.name for r in silent_exempt]} are fixedtime tiers that are "
            "byte-identical to the untuned corpus. k was retuned, so identical output "
            "means --fixed-time-k never took effect.",
            flush=True,
        )
        return 1

    if failures:
        print(
            "FAIL: the gate did not pass. Do not consume datasets_v11/ until every "
            "difference above is explained.",
            flush=True,
        )
        return 1
    print("PASS: every unchanged tier is bit-identical; only fixedtime differs.", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
