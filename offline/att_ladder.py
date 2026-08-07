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

AMENDMENT A5 -- UNCONDITIONAL CO-REPORT, AND SHARED DRAW IDS
------------------------------------------------------------
A4 (2026-08-06) relaxed §3.1's mandatory co-report to ``vehicle_count`` at the episode
horizon but carried §3.1's ">5% and the comparison is invalid" condition across unchanged.
**A5 (2026-08-07, tag ``v0.6-prereg-a5``) withdrew that condition**, because ``entered`` is a
population size while ``vehicle_count`` at the horizon is a **control outcome** -- precisely
what a good controller drives toward zero. A 5% band on it voids the effect C1 exists to
measure: on our own committed anchors, 5 of 6 pairwise baseline comparisons came out INVALID,
including ``cf_grid4x4`` MaxPressure vs MAPPO differing by 98.1%, which is not a defect in
the comparison -- **it is the result**.

So, as registered by A5:

1. ``vehicle_count`` at the horizon is reported **UNCONDITIONALLY** beside every ATT cell.
   No threshold at which it is omitted, none at which it triggers a verdict, hence no
   researcher degree of freedom left in the disclosure.
2. There is **no validity threshold in this module** -- and therefore no denominator to
   choose, because there is no condition.
3. **Every cell carries its draw ids**, and a comparison is valid only over **shared** draws.
   A pair with no shared draws is **void**. That is binary and checkable from the data,
   rather than thresholded, and it enforces identical demand *by construction* instead of
   testing for it afterwards.

Draw ids are read from each episode's own stored ``flow_draw`` scalar, not from the manifest,
so the reported set is what the corpus actually contains.

**When two tiers' draw sets differ but overlap**, the comparison must be recomputed over the
intersection before it is reported. This tool does **not** silently pool: it reports each
cell's draw set and the size of every pairwise intersection, so a partial overlap is visible
rather than absorbed into a mean over different demand.
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

HEADER = (
    "att_horizon by tier -- RANDOMISED DRAWS 1-200 ONLY.\n"
    "This does NOT settle the nominal draw-0 comparison, which is a separate run.\n"
    "att_horizon = att_per_step[-1] (A1). att_per_step.mean() is att_running_mean and\n"
    "is NOT reported. vehicle_count at the horizon is co-reported UNCONDITIONALLY per A5,\n"
    "which withdrew A4's >5% validity condition. A comparison is valid only over SHARED\n"
    "draw ids, so every cell reports its draw set."
)

__all__ = ["TierCell", "DrawOverlap", "tier_cells", "draw_overlaps", "main"]


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
    draw_ids: tuple[int, ...]

    def draw_summary(self) -> str:
        """Compact draw-set description; the full list goes to the JSON output."""
        if not self.draw_ids:
            return "draws=none"
        lo, hi = self.draw_ids[0], self.draw_ids[-1]
        contiguous = list(self.draw_ids) == list(range(lo, hi + 1))
        shape = f"{lo}-{hi}" if contiguous else f"{lo}..{hi} sparse"
        return f"draws={len(self.draw_ids)} [{shape}]"

    def line(self) -> str:
        return (
            f"  {self.scenario:12s} {self.tier:22s} n={self.n_episodes:4d}  "
            f"att_horizon {self.att_horizon_mean:8.2f} +/- {self.att_horizon_ci95:6.2f}  "
            f"(sd {self.att_horizon_std:7.2f})   "
            f"horizon vehicle_count {self.vehicle_count_mean:8.2f} "
            f"(sd {self.vehicle_count_std:6.2f})   {self.draw_summary()}"
        )


@dataclass(frozen=True)
class DrawOverlap:
    """Whether two tiers of one scenario can be compared at all (A5 point 3)."""

    scenario: str
    tier_a: str
    tier_b: str
    n_shared: int
    n_a: int
    n_b: int

    @property
    def void(self) -> bool:
        """No shared draws means the comparison cannot be made. Binary, not thresholded."""
        return self.n_shared == 0

    @property
    def identical(self) -> bool:
        return self.n_shared == self.n_a == self.n_b


def _split_run_name(name: str) -> tuple[str, str]:
    """``cf_hz1x1__mappo1000__seed101`` -> (``cf_hz1x1``, ``mappo1000``)."""
    parts = name.split("__")
    return parts[0], (parts[1] if len(parts) > 1 else "")


def tier_cells(root: Path) -> list[TierCell]:
    """Aggregate every run under *root* into (scenario, tier) cells.

    Seed-split runs of the same tier are pooled, which is what makes a tier's ``n`` the
    full 200 draws rather than one seed's 40.
    """
    buckets: dict[tuple[str, str], list[tuple[float, float, int]]] = {}
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
                (
                    float(episode.att_per_step[-1]),
                    float(episode.vehicle_count[-1]),
                    # -1 is the on-disk "no draw" sentinel; a nominal episode has no
                    # draw id and must not be pooled with randomised ones (A5 point 3).
                    int(episode.flow_draw),
                )
            )

    cells: list[TierCell] = []
    for (scenario, tier), rows in sorted(buckets.items()):
        att = np.asarray([r[0] for r in rows], dtype=np.float64)
        veh = np.asarray([r[1] for r in rows], dtype=np.float64)
        draws = tuple(sorted({r[2] for r in rows if r[2] >= 0}))
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
                draw_ids=draws,
            )
        )
    return cells


def draw_overlaps(cells: Sequence[TierCell]) -> list[DrawOverlap]:
    """Shared-draw status for every within-scenario pair (A5 point 3).

    A5 replaced A4's withdrawn threshold with a construction requirement: a comparison is
    valid only over shared draws, and void without them. This reports that rather than
    deciding it, because a partial overlap needs the cells recomputed over the
    intersection -- which is a reporting decision, not something to absorb into a mean.
    """
    out: list[DrawOverlap] = []
    by_scenario: dict[str, list[TierCell]] = {}
    for cell in cells:
        by_scenario.setdefault(cell.scenario, []).append(cell)

    for scenario, group in sorted(by_scenario.items()):
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                shared = set(a.draw_ids) & set(b.draw_ids)
                out.append(
                    DrawOverlap(
                        scenario=scenario, tier_a=a.tier, tier_b=b.tier,
                        n_shared=len(shared),
                        n_a=len(a.draw_ids), n_b=len(b.draw_ids),
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

    overlaps = draw_overlaps(cells)
    void = [o for o in overlaps if o.void]
    partial = [o for o in overlaps if not o.void and not o.identical]

    print(
        f"\nA5 shared-draw check ({len(overlaps)} within-scenario pairs). "
        "There is NO validity threshold: A5 withdrew it.",
        flush=True,
    )
    if not void and not partial:
        print(
            "  every pair shares an identical draw set -- all comparisons are made over "
            "identical demand by construction.",
            flush=True,
        )
    for o in void:
        print(
            f"  VOID {o.scenario} {o.tier_a} vs {o.tier_b}: no shared draws "
            f"({o.n_a} vs {o.n_b}, 0 in common). A5 point 3: this comparison cannot be "
            "made and must not be reported.",
            flush=True,
        )
    for o in partial:
        print(
            f"  PARTIAL {o.scenario} {o.tier_a} vs {o.tier_b}: {o.n_shared} shared of "
            f"{o.n_a}/{o.n_b}. Recompute BOTH cells over the shared draws before "
            "reporting this pair; the means above are over different demand.",
            flush=True,
        )

    if args.json_out:
        payload = {
            "scope": "randomised draws 1-200; does NOT settle the nominal draw-0 comparison",
            "primary_metric": "att_horizon = att_per_step[-1] (prereg A1)",
            "co_report": (
                "vehicle_count at the episode horizon, reported unconditionally "
                "(prereg A5; A4's >5% condition is withdrawn)"
            ),
            "comparison_rule": (
                "valid only over shared draw ids; void without them (prereg A5 point 3)"
            ),
            "cells": [vars(c) for c in cells],
            "void_pairs": [vars(o) for o in void],
            "partial_overlap_pairs": [vars(o) for o in partial],
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
