"""Validation gate: ``datasets_v11/`` must reproduce ``datasets/`` bit-for-bit, bar one tier.

Format v1.1 adds one observation array and changes nothing else, so re-collecting the same
policies against the same demand must reproduce **every trajectory-defining array
bit-identically**.  ``fixedtime`` is the only tier permitted to differ, because its cycle
multiplier ``k`` was retuned (4 -> 6/1/3, ``docs/data/fixed_time_sweep/README.md``).

``cf_cologne3`` is the second exemption, and it is a **measured envelope, not a waiver** --
see below.  Any other difference means something drifted between the two collections -- a
changed default, a different checkpoint, a numpy or engine change -- and the v1.1 corpus must
not be consumed until it is explained.

    python -m offline.compare_corpora                       # gate; exit 0 = pass

WHY cf_cologne3 CANNOT BE HELD TO BIT-IDENTITY (measured 2026-08-07, ruling ``0398039``)
-----------------------------------------------------------------------------------------
The gate's first run failed 15 of 69 tiers and **every failure was cologne3**; hz1x1 was
22/22 bit-identical and grid4x4 22/22.  That pattern was verified rather than accepted, with
a discriminating experiment and a control -- the same command, the same code, twice on one
day:

    cologne3, the 6 draws the gate flagged  -> 3/6 differ
    grid4x4 control, the same 6 draws       -> 0/6 differ

Those 6 were selected for already differing, so that rate is upward-biased.  Unbiased
re-measurement over draws 1-40:

    cologne3 differs from ITSELF   3/39 = 7.7%
    the gate's v1.0 vs v1.1        6/200 = 3.0%

**The v1.0-vs-v1.1 difference is smaller than cologne3's difference from itself.**  A logger
defect cannot be scenario-selective and cannot appear between two runs of identical code, so
format v1.1 is exonerated by construction rather than by argument.  The cause is CityFlow
engine non-determinism on that scenario, unidentified (DEFERRED item 13).

The **shape** of the divergence is what sets the replacement criterion: individual episodes
move while the distribution does not.  Worst single draw 5.7% apart; tier mean of
``total_global_reward`` 0.080% apart, std 21496.4 vs 21499.5.  So the rungs of the C1 ladder
are safe on cologne3 and it is the per-episode analyses that inherit the noise.

Hence :data:`ENVELOPE_SCENARIO`'s three replacement conditions, each paired below with the
measurement that produced its tolerance so a later reader sees where the number came from
instead of finding a round one.

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
import math
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

#: The only tier allowed to differ wholesale, and only because k was retuned.
EXEMPT_POLICY = "fixedtime"

#: Scenario held to a measured envelope instead of bit-identity (ruling 0398039).
#: Not a waiver: the three conditions below are stricter than "ignore this scenario", and
#: each tolerance is quoted against the self-noise measurement that set it.
ENVELOPE_SCENARIO = "cf_cologne3"

#: (ii) Tier mean of ``total_global_reward``, relative.
#: MEASURED SELF-NOISE: 0.080% (tier mean, cologne3 against a re-run of itself).
#: Tolerance 1% leaves ~12x headroom, so a real drift has to be an order of magnitude
#: larger than the engine noise before it passes.
ENVELOPE_MEAN_REWARD_TOL = 0.01

#: (iii) Fraction of episodes differing on any trajectory array.
#: MEASURED SELF-NOISE: 7.7% (3/39, cologne3 against a re-run of itself, draws 1-40,
#: unbiased). Tolerance 20% leaves ~2.6x headroom. The gate's own v1.0-vs-v1.1 figure was
#: 3.0% (6/200) -- lower than the self-noise, which is what exonerated v1.1.
ENVELOPE_DIFFERING_FRACTION_TOL = 0.20

__all__ = [
    "RunComparison", "compare_run", "compare_corpora", "envelope_check", "main",
]


@dataclass(frozen=True)
class RunComparison:
    """Outcome for one collection run."""

    name: str
    exempt: bool
    episodes: int
    differing_arrays: tuple[str, ...]
    differing_episodes: int
    error: str | None = None
    #: Set for ENVELOPE_SCENARIO runs: (mean rel. diff, differing fraction, failures).
    envelope: tuple[float, float, tuple[str, ...]] | None = None

    @property
    def identical(self) -> bool:
        return self.error is None and not self.differing_arrays

    @property
    def envelope_ok(self) -> bool:
        return self.envelope is not None and not self.envelope[2]


def _episode_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("ep*.npz"))


def _manifest_episodes(run_dir: Path) -> list[dict]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["episodes"]


def envelope_check(
    old_dir: Path, new_dir: Path, differing_episodes: int, episodes: int
) -> tuple[float, float, tuple[str, ...]]:
    """The three replacement conditions for :data:`ENVELOPE_SCENARIO`.

    Returns ``(relative mean-reward difference, differing fraction, failure messages)``.
    An empty failure tuple means the envelope holds.

    Condition (i) -- episode counts and draw ids matching **exactly** -- is deliberately
    NOT relaxed. Engine non-determinism moves the numbers inside an episode; it cannot
    change which draws were collected or how many. A mismatch there is a campaign defect,
    not scenario noise, and must not be absorbed by the tolerances that follow.
    """
    failures: list[str] = []
    old_eps, new_eps = _manifest_episodes(old_dir), _manifest_episodes(new_dir)

    if len(old_eps) != len(new_eps):
        failures.append(
            f"episode count {len(old_eps)} vs {len(new_eps)} (condition i: exact)"
        )
        return (math.nan, math.nan, tuple(failures))

    old_draws = [e["flow_draw"] for e in old_eps]
    new_draws = [e["flow_draw"] for e in new_eps]
    if old_draws != new_draws:
        differing = sorted(set(old_draws) ^ set(new_draws))
        failures.append(
            f"draw ids differ (condition i: exact); symmetric difference {differing[:10]}"
        )

    old_mean = float(np.mean([e["total_global_reward"] for e in old_eps]))
    new_mean = float(np.mean([e["total_global_reward"] for e in new_eps]))
    rel_mean = abs(new_mean - old_mean) / abs(old_mean) if old_mean else math.inf
    if rel_mean > ENVELOPE_MEAN_REWARD_TOL:
        failures.append(
            f"tier mean total_global_reward {old_mean:.1f} vs {new_mean:.1f} = "
            f"{100 * rel_mean:.3f}% apart, over the {100 * ENVELOPE_MEAN_REWARD_TOL:.0f}% "
            "tolerance (condition ii; measured self-noise 0.080%)"
        )

    fraction = (differing_episodes / episodes) if episodes else 0.0
    if fraction > ENVELOPE_DIFFERING_FRACTION_TOL:
        failures.append(
            f"{differing_episodes}/{episodes} = {100 * fraction:.1f}% of episodes differ, "
            f"over the {100 * ENVELOPE_DIFFERING_FRACTION_TOL:.0f}% tolerance "
            "(condition iii; measured self-noise 7.7%)"
        )

    return (rel_mean, fraction, tuple(failures))


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

        # cf_cologne3 is not bit-reproducible even against itself, so it is held to the
        # measured envelope instead. fixedtime takes priority: k was retuned, so those
        # tiers are expected to differ wholesale and the 1% mean-reward condition would
        # correctly reject them for the wrong reason.
        #
        # Run unconditionally, INCLUDING when compare_run reported a structural error.
        # A draw-id mismatch surfaces there first as "filenames differ", which is true
        # but generic; routing it through envelope_check as well is what makes condition
        # (i) report itself in its own words instead of leaving that branch unreachable.
        if not result.exempt and new_dir.name.startswith(ENVELOPE_SCENARIO):
            rel_mean, fraction, failures = envelope_check(
                old_dir, new_dir, result.differing_episodes, result.episodes
            )
            if result.error is not None:
                failures = (*failures, f"structural: {result.error}")
            result = RunComparison(
                result.name, result.exempt, result.episodes, result.differing_arrays,
                result.differing_episodes, error=None,
                envelope=(rel_mean, fraction, failures),
            )
            results.append(result)
            continue

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

    # Announced up front, the way fixedtime's exemption is, so a PASS can never hide it.
    print(
        f"EXEMPTIONS IN FORCE (ruling 0398039):\n"
        f"  * __{EXEMPT_POLICY} tiers are expected to differ -- k was retuned 4 -> 6/1/3.\n"
        f"  * {ENVELOPE_SCENARIO} is NOT held to bit-identity: it is not bit-reproducible\n"
        f"    against ITSELF (3/39 = 7.7% of draws differ between two runs of identical\n"
        f"    code), which is more than its v1.0-vs-v1.1 difference (6/200 = 3.0%). It is\n"
        f"    held to a measured envelope instead: exact episode counts and draw ids, tier\n"
        f"    mean total_global_reward within {100 * ENVELOPE_MEAN_REWARD_TOL:.0f}% "
        f"(self-noise 0.080%), and at most\n"
        f"    {100 * ENVELOPE_DIFFERING_FRACTION_TOL:.0f}% of episodes differing "
        f"(self-noise 7.7%).\n"
        f"  * cf_hz1x1 and cf_grid4x4 keep STRICT bit-identity; they earned it 44/44.\n",
        flush=True,
    )

    failures: list[RunComparison] = []
    for r in results:
        if r.error and not r.exempt:
            status, detail = "ERROR", r.error
            failures.append(r)
        elif r.envelope is not None:
            rel_mean, fraction, envelope_failures = r.envelope
            if envelope_failures:
                status = "ENV-FAIL"
                detail = "; ".join(envelope_failures)
                failures.append(r)
            else:
                status = "ENVELOPE"
                detail = (
                    f"{r.episodes} episodes, {r.differing_episodes} differ "
                    f"({100 * fraction:.1f}% <= "
                    f"{100 * ENVELOPE_DIFFERING_FRACTION_TOL:.0f}%), tier mean reward "
                    f"{100 * rel_mean:.3f}% apart (<= "
                    f"{100 * ENVELOPE_MEAN_REWARD_TOL:.0f}%)"
                )
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
    enveloped = [r for r in results if r.envelope is not None]
    strict = [r for r in results if not r.exempt and r.envelope is None]
    print(
        f"\n{len(strict)} tier(s) under STRICT bit-identity, "
        f"{len(enveloped)} under the {ENVELOPE_SCENARIO} envelope, "
        f"{len(exempt)} exempt (fixedtime), {len(failures)} failure(s).",
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
    print(
        f"PASS: {len(strict)} tier(s) bit-identical; {len(enveloped)} "
        f"{ENVELOPE_SCENARIO} tier(s) inside the measured envelope; "
        f"{len(exempt)} fixedtime tier(s) differ as expected.",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
