"""P9.1's GPU-overlap disclosure (``offline/corrective_report.py`` ``gpu_overlap``; ``BRIEF_46`` Amendment D, D.2-2).

No format is written: the function reads the campaign's ``p9.1-run/1.0`` records (their ``started_unix`` /
``finished_unix``) and the driver's Unix-time GPU trace (``<unix>,<memory.used MiB>,<utilization %>`` per line, an empty
reading when ``nvidia-smi`` gave none), and returns what the G4 packet discloses: for another GPU process's window (P9.3's
smoke, from its note), every training run whose bracket overlaps it with the overlap in seconds, and the trace inside the
window beside the trace outside it.  Expected values are written by hand from the fixture's numbers (all exact in
binary).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import offline.corrective_report as rep
import offline.corrective_round as cr

RUNS = cr.training_runs()


def _roots(tmp_path: Path) -> cr.Roots:
    output = tmp_path / "output"
    output.mkdir()
    return cr.Roots(repo_root=Path(__file__).resolve().parents[1], output_root=output, corpus_root=tmp_path / "corpus",
                    draws_root=tmp_path / "draws")


def _record(roots: cr.Roots, index: int, started: float, finished: float) -> None:
    cell, seed = RUNS[index]
    path = cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, int(seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scenario": cell.scenario, "tier": cell.tier, "arm": cell.arm, "seed": int(seed),
                                "started_unix": started, "finished_unix": finished, "seconds": finished - started,
                                "clock": cr.CLOCK_LABEL}), encoding="utf-8")


def _trace(roots: cr.Roots, lines: list[str]) -> None:
    path = Path(roots.runs_root) / "logs" / "20261028T150000Z" / "gpu_trace.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_every_run_overlapping_the_window_is_listed_with_its_overlap_and_no_other(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _record(roots, 0, 1000.0, 2000.0)     # ends inside the window: 500 s
    _record(roots, 1, 2000.0, 3000.0)     # inside it entirely: 1000 s
    _record(roots, 2, 3000.0, 4000.0)     # starts inside it: 250 s
    _record(roots, 3, 4000.0, 5000.0)     # after it
    _record(roots, 4, 100.0, 1000.0)      # before it, touching nothing
    _trace(roots, ["1400.0,9000,90", "1600.0,15000,99", "2500.0,15500,98", "3200.0,14000,97", "3300.0,,",
                   "4500.0,9100,91", "100.0,8900,89"])
    result = rep.gpu_overlap(roots, start_unix=1500.0, end_unix=3250.0)
    got = {(r["scenario"], r["tier"], r["arm"], r["seed"]): r["overlap_seconds"] for r in result["runs"]}
    expected = {(RUNS[i][0].scenario, RUNS[i][0].tier, RUNS[i][0].arm, int(RUNS[i][1])): s
                for i, s in ((0, 500.0), (1, 1000.0), (2, 250.0))}
    assert got == expected
    assert result["overlap_seconds_total"] == 1750.0 and result["records_read"] == 5
    assert result["window"] == {"start_unix": 1500.0, "end_unix": 3250.0, "seconds": 1750.0}
    assert result["trace"]["inside"] == {"samples": 3, "readings": 3, "memory_used_mib": [14000.0, 15500.0],
                                         "utilization_pct": [97.0, 99.0]}
    assert result["trace"]["outside"] == {"samples": 4, "readings": 3, "memory_used_mib": [8900.0, 9100.0],
                                          "utilization_pct": [89.0, 91.0]}


def test_a_window_touching_no_run_lists_none_and_an_empty_or_reversed_window_is_refused(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _record(roots, 0, 1000.0, 2000.0)
    _trace(roots, ["1000.0,9000,90"])
    result = rep.gpu_overlap(roots, start_unix=2000.0, end_unix=2500.0)
    assert result["runs"] == [] and result["overlap_seconds_total"] == 0.0
    for start, end in ((2000.0, 2000.0), (2500.0, 2000.0)):
        with pytest.raises(ValueError, match="a window ends after it starts"):
            rep.gpu_overlap(roots, start_unix=start, end_unix=end)


def test_a_trace_line_that_is_not_a_reading_is_refused(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _record(roots, 0, 1000.0, 2000.0)
    _trace(roots, ["1000.0,9000,90", "garbage"])
    with pytest.raises(ValueError, match="gpu_trace.csv line 2"):
        rep.gpu_overlap(roots, start_unix=1500.0, end_unix=2500.0)
