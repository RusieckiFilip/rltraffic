"""Hard-fail linter for on-disk corpus format consistency.

Four rejections, each a way a corpus can be quietly unusable:

1. **A corpus mixing format v1.0 and v1.1.** The two must never be silently combined: a
   v1.0 episode has no ``att_per_step``, so a consumer pooling them either crashes or --
   worse -- silently trains on the subset that happens to carry the field.
2. **A v1.1 episode missing ``att_per_step``.** The version string and the file contents
   must agree; a file claiming v1.1 without the array is the failure mode a version check
   alone cannot see.
3. **An unknown ``format_version``.** C6 requires the linter to hard-fail rather than guess
   an alignment.
4. **A corpus whose episodes disagree on ``metric_keys``.** Contract C8: the metric set is
   frozen for the lifetime of the checkpoints collected against it, so an inhomogeneous
   corpus means at least one tier was collected under a different MDP.

Check 4 is not hypothetical. The aborted 2026-08-06 campaign produced ``cf_hz1x1__random``
carrying 2 metric keys while its sibling tiers carried 3 (measured before that tree was
deleted; recoverable at ``7dc9928~1``).

    python -m offline.corpus_format_check datasets_v11/     # exit 0 = clean

Version and metric keys are read from **each ``.npz``**, not from the manifest. The manifest
records the *requested* metric list, which is ``null`` on every run in this corpus because
the env derives the set -- so a manifest check would assert nothing. This also keeps the
promise that v1.1 adds exactly one array and touches no manifest field.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from offline.trajectory_logger import SUPPORTED_FORMAT_VERSIONS

#: Arrays each format version is required to carry beyond the v1.0 core.
REQUIRED_ARRAYS: dict[str, tuple[str, ...]] = {
    "1.0": (),
    "1.1": ("att_per_step",),
}

__all__ = ["CorpusReport", "check_corpus", "main"]


@dataclass
class CorpusReport:
    """Every violation found, grouped by kind. Empty means clean."""

    n_episodes: int = 0
    versions: dict[str, int] = field(default_factory=dict)
    metric_key_sets: dict[tuple[str, ...], int] = field(default_factory=dict)
    unknown_version: list[str] = field(default_factory=list)
    missing_arrays: list[str] = field(default_factory=list)

    @property
    def mixed_versions(self) -> bool:
        return len(self.versions) > 1

    @property
    def inhomogeneous_metric_keys(self) -> bool:
        return len(self.metric_key_sets) > 1

    @property
    def ok(self) -> bool:
        return not (
            self.unknown_version
            or self.missing_arrays
            or self.mixed_versions
            or self.inhomogeneous_metric_keys
        )


def check_corpus(root: str | Path) -> CorpusReport:
    """Scan every episode under *root* and report format violations."""
    report = CorpusReport()
    versions: dict[str, int] = defaultdict(int)
    key_sets: dict[tuple[str, ...], int] = defaultdict(int)

    for path in sorted(Path(root).rglob("ep*.npz")):
        report.n_episodes += 1
        with np.load(path) as data:
            version = str(data["format_version"].item())
            versions[version] += 1

            if version not in SUPPORTED_FORMAT_VERSIONS:
                report.unknown_version.append(
                    f"{path}: format_version {version!r} is not one of "
                    f"{list(SUPPORTED_FORMAT_VERSIONS)}"
                )
                # No point checking required arrays against a version we cannot read.
                continue

            for name in REQUIRED_ARRAYS[version]:
                if name not in data.files:
                    report.missing_arrays.append(
                        f"{path}: claims format {version!r} but has no {name!r} array"
                    )

            key_sets[tuple(str(k) for k in data["metric_keys"].tolist())] += 1

    report.versions = dict(versions)
    report.metric_key_sets = dict(key_sets)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code (0 clean, 1 violations, 2 usage)."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.corpus_format_check",
        description="Reject a corpus that mixes format versions or metric sets.",
    )
    parser.add_argument("root", help="corpus root, e.g. datasets_v11/")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", flush=True)
        return 2

    report = check_corpus(root)
    if report.n_episodes == 0:
        # Never report clean on an empty scan: that is a mistyped path, and exiting 0
        # would certify a corpus nobody looked at.
        print(f"ERROR: no ep*.npz found under {root}", flush=True)
        return 2

    print(f"{report.n_episodes} episode(s) under {root}", flush=True)
    print(f"  format versions : {report.versions}", flush=True)
    print(
        "  metric key sets : "
        + ", ".join(f"{list(k)} x{n}" for k, n in report.metric_key_sets.items()),
        flush=True,
    )

    if report.mixed_versions:
        print(
            f"\nFAIL: this corpus MIXES format versions {sorted(report.versions)}. "
            "A v1.0 episode has no att_per_step, so a consumer pooling them either "
            "crashes or silently uses only the subset that carries the field. Keep the "
            "two corpora in separate roots.",
            flush=True,
        )
    if report.inhomogeneous_metric_keys:
        print(
            f"\nFAIL: episodes disagree on metric_keys ({len(report.metric_key_sets)} "
            "distinct sets). Contract C8 freezes the metric set for the lifetime of the "
            "checkpoints collected against it, so at least one tier here was collected "
            "under a different MDP:",
            flush=True,
        )
        for keys, count in report.metric_key_sets.items():
            print(f"    {list(keys)}  ({count} episode(s))", flush=True)
    for message in report.unknown_version:
        print(f"\nFAIL: {message}", flush=True)
    for message in report.missing_arrays:
        print(f"\nFAIL: {message}", flush=True)

    if not report.ok:
        return 1
    print("\nPASS: one format version, one metric set, every required array present.",
          flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
