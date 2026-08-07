"""Emit ``att_horizon`` per tier per scenario from a v1.1 corpus, with A4's companion.

⚠️ **WHAT THIS ANSWERS, AND WHAT IT DOES NOT.**
This reads the **randomised-draw corpus, draws 1-200**.  It answers the randomised-draw
question and nothing else.  It **does not settle the nominal draw-0 comparison**, which is a
separate small run through ``experiments/run.py`` and whose cells are the ones currently
marked NOT SETTLED.  Every report this module prints carries that sentence in its header, so
a table lifted out of a terminal cannot lose it.  Do not read the output as closing that
item.

WHY IT EXISTS
-------------
The ladder was measured under ``att_running_mean`` -- the mean of the per-step samples --
which amendment A1 retires in favour of the value **at the episode horizon**.  The two
differ policy-dependently: measured on one real 360-step episode during P2.6, ``att_horizon``
was 178.90 against ``att_per_step.mean()`` 127.88, a 40% gap.  A ladder whose rungs are 2.5%
and 5.8% apart cannot survive that substitution unexamined.

``att_horizon`` is ``att_per_step[-1]``: row ``T`` of the v1.1 observation array, which is
the registered metric's value at the horizon.  **``att_per_step.mean()`` is
``att_running_mean`` and is never reported here.**

AMENDMENT A4 -- THE CO-REPORT IS MANDATORY, NOT OPTIONAL
--------------------------------------------------------
A4 (2026-08-06) relaxed §3.1's mandatory co-reported quantities to ``vehicle_count`` at the
episode horizon, and restated §3.1's validity condition against it.  §3.1, verbatim:

    "a comparison between two policies on a scenario is reported as invalid, not as a win,
    if their entered-vehicle counts differ by more than 5%. Under such a difference the two
    ATTs are averages over different populations."

So every ATT cell here carries horizon ``vehicle_count``, and every pairwise comparison
within a scenario is screened against the 5% rule.  An ATT comparison without the co-report
is **invalid by our own registration**, so emitting ``att_horizon`` alone would produce a
number we could not cite.  Both come from the same ``[-1]`` on arrays the corpus already
stores, so the co-report costs nothing.

⚠️ **One ambiguity, flagged rather than hidden.** "differ by more than 5%" does not name a
denominator.  This module uses the **conservative** reading -- relative to the *smaller* of
the two, which flags more comparisons invalid -- and prints the permissive reading
(relative to the larger) beside it, so a reader can see whether the choice changed the
verdict.  If it ever does change a verdict, that needs a ruling, not a default.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from offline.trajectory_logger import load_episode

#: A4 / §3.1 validity threshold on the co-reported quantity.
VALIDITY_THRESHOLD = 0.05

HEADER = (
    "att_horizon by tier -- RANDOMISED DRAWS 1-200 ONLY.\n"
    "This does NOT settle the nominal draw-0 comparison, which is a separate run.\n"
    "att_horizon = att_per_step[-1] (A1). att_per_step.mean() is att_running_mean and\n"
    "is NOT reported. vehicle_count at the horizon is co-reported per A4 and every\n"
    "pairwise comparison is screened against the >5% validity condition."
)

__all__ = ["TierCell", "Comparison", "tier_cells", "screen_comparisons", "main"]


@dataclass(frozen=True)
class TierCell:
    """One (scenario, tier) cell: the primary metric and its mandatory companion."""

    scenario: str
    tier: str
    n_episodes: int
    att_horizon_mean: float
    att_horizon_std: float
    att_horizon_ci95: float
    vehicle_count_mean: float
    vehicle_count_std: float

    def line(self) -> str:
        return (
            f"  {self.scenario:12s} {self.tier:22s} n={self.n_episodes:4d}  "
            f"att_horizon {self.att_horizon_mean:8.2f} +/- {self.att_horizon_ci95:6.2f}  "
            f"(sd {self.att_horizon_std:7.2f})   "
            f"horizon vehicle_count {self.vehicle_count_mean:8.2f} "
            f"(sd {self.vehicle_count_std:6.2f})"
        )


@dataclass(frozen=True)
class Comparison:
    """A pairwise ATT comparison and its A4 validity verdict."""

    scenario: str
    tier_a: str
    tier_b: str
    att_a: float
    att_b: float
    vc_a: float
    vc_b: float
    rel_diff_conservative: float
    rel_diff_permissive: float

    @property
    def valid(self) -> bool:
        return self.rel_diff_conservative <= VALIDITY_THRESHOLD

    @property
    def denominator_choice_matters(self) -> bool:
        """True when the two readings of "differ by more than 5%" disagree."""
        return (self.rel_diff_conservative > VALIDITY_THRESHOLD) != (
            self.rel_diff_permissive > VALIDITY_THRESHOLD
        )


def _split_run_name(name: str) -> tuple[str, str]:
    """``cf_hz1x1__mappo1000__seed101`` -> (``cf_hz1x1``, ``mappo1000``)."""
    parts = name.split("__")
    return parts[0], (parts[1] if len(parts) > 1 else "")


def tier_cells(root: Path) -> list[TierCell]:
    """Aggregate every run under *root* into (scenario, tier) cells.

    Seed-split runs of the same tier are pooled, which is what makes a tier's ``n`` the
    full 200 draws rather than one seed's 40.
    """
    buckets: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        scenario, tier = _split_run_name(run_dir.name)
        if not tier:
            continue
        for episode_path in sorted(run_dir.glob("ep*.npz")):
            episode = load_episode(episode_path)
            if episode.att_per_step is None:
                raise ValueError(
                    f"{episode_path}: format {episode.format_version} has no "
                    "att_per_step, so att_horizon cannot be read. This tool requires a "
                    "v1.1 corpus; run offline.corpus_format_check first."
                )
            buckets.setdefault((scenario, tier), []).append(
                (float(episode.att_per_step[-1]), float(episode.vehicle_count[-1]))
            )

    cells: list[TierCell] = []
    for (scenario, tier), rows in sorted(buckets.items()):
        att = np.asarray([r[0] for r in rows], dtype=np.float64)
        veh = np.asarray([r[1] for r in rows], dtype=np.float64)
        n = int(att.size)
        # Sample std (ddof=1); a single-episode tier has no spread to report.
        std = float(att.std(ddof=1)) if n > 1 else 0.0
        cells.append(
            TierCell(
                scenario=scenario,
                tier=tier,
                n_episodes=n,
                att_horizon_mean=float(att.mean()),
                att_horizon_std=std,
                att_horizon_ci95=(1.96 * std / math.sqrt(n)) if n > 1 else 0.0,
                vehicle_count_mean=float(veh.mean()),
                vehicle_count_std=float(veh.std(ddof=1)) if n > 1 else 0.0,
            )
        )
    return cells


def screen_comparisons(cells: Sequence[TierCell]) -> list[Comparison]:
    """Every within-scenario pair, screened against A4's >5% condition."""
    out: list[Comparison] = []
    by_scenario: dict[str, list[TierCell]] = {}
    for cell in cells:
        by_scenario.setdefault(cell.scenario, []).append(cell)

    for scenario, group in sorted(by_scenario.items()):
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                lo, hi = sorted((a.vehicle_count_mean, b.vehicle_count_mean))
                gap = hi - lo
                out.append(
                    Comparison(
                        scenario=scenario,
                        tier_a=a.tier, tier_b=b.tier,
                        att_a=a.att_horizon_mean, att_b=b.att_horizon_mean,
                        vc_a=a.vehicle_count_mean, vc_b=b.vehicle_count_mean,
                        rel_diff_conservative=(gap / lo) if lo > 0 else math.inf,
                        rel_diff_permissive=(gap / hi) if hi > 0 else math.inf,
                    )
                )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m offline.att_ladder",
        description="att_horizon per tier from a v1.1 corpus (randomised draws only).",
    )
    parser.add_argument("--root", default="datasets_v11")
    parser.add_argument("--json-out", default=None, help="also write the cells as JSON")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: {root} does not exist -- run the campaign first", flush=True)
        return 2

    print(HEADER + "\n", flush=True)
    cells = tier_cells(root)
    if not cells:
        print(f"ERROR: no runs found under {root}", flush=True)
        return 2

    current = None
    for cell in cells:
        if cell.scenario != current:
            current = cell.scenario
            print(f"\n{current}", flush=True)
        print(cell.line(), flush=True)

    comparisons = screen_comparisons(cells)
    invalid = [c for c in comparisons if not c.valid]
    contested = [c for c in comparisons if c.denominator_choice_matters]

    print(f"\nA4 validity screen ({len(comparisons)} within-scenario pairs):", flush=True)
    if not invalid:
        print("  all pairs pass the >5% horizon vehicle_count condition.", flush=True)
    for c in invalid:
        print(
            f"  INVALID {c.scenario} {c.tier_a} vs {c.tier_b}: horizon vehicle_count "
            f"{c.vc_a:.1f} vs {c.vc_b:.1f} differ by "
            f"{100 * c.rel_diff_conservative:.1f}% (conservative) / "
            f"{100 * c.rel_diff_permissive:.1f}% (permissive). "
            f"ATT {c.att_a:.2f} vs {c.att_b:.2f} is NOT a win either way.",
            flush=True,
        )
    if contested:
        print(
            f"\n  ⚠️ {len(contested)} pair(s) change verdict between the two readings of "
            "'differ by more than 5%'. That needs a ruling, not a default:\n    "
            + "\n    ".join(
                f"{c.scenario} {c.tier_a} vs {c.tier_b}: "
                f"{100 * c.rel_diff_conservative:.1f}% vs {100 * c.rel_diff_permissive:.1f}%"
                for c in contested
            ),
            flush=True,
        )

    if args.json_out:
        payload = {
            "scope": "randomised draws 1-200; does NOT settle the nominal draw-0 comparison",
            "primary_metric": "att_horizon = att_per_step[-1] (prereg A1)",
            "co_report": "vehicle_count at the episode horizon (prereg A4)",
            "validity_threshold": VALIDITY_THRESHOLD,
            "cells": [vars(c) for c in cells],
            "invalid_comparisons": [vars(c) for c in invalid],
        }
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}", flush=True)

    print(
        "\nREMINDER: randomised draws 1-200 only. The nominal draw-0 cells are a "
        "separate run and remain open.",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
