"""P9.4's smoke (PROJECT_PLAN section 7 rule (4); the author's terms of 2026-11-19): every DISTINCT path the campaign
executes, once, through the real code with only its size reduced, in a scratch root outside the main tree.

The paths: the control retrain and the new training (``steps`` gradient steps each, all five seeds of each so the
control record and the facts' five-checkpoint agreement are exercised), one CityFlow cell through both routes (seed
101, draw 1000), one SUMO cell per prompt arm (seed 101, draw 1001), one G1 cell and one G2 cell (draw 1000, which
carries the halting cross-check).  The result is PASS or FAIL and the seconds per path -- no other number is returned,
and a G1 or G2 cell that differs under ``==`` from its committed value is a FAIL naming the fields.

At reduced size the control cannot reproduce P4.7's 40,000-step checkpoints, so ``train_session`` ends with
``ControlFailed`` after writing its record: that path PASSES when the record exists and covers five seeds (its verdict
is not kept).  The new tier then trains through ``train_run`` directly -- the gate before it is the unit tests'.

The scratch root's ``output/`` receives COPIES of P4.7's five mix50 checkpoints and ``SHA256SUMS_p4_7.txt`` (G1 and the
control read them under ``output_root``); the real ``output/`` is only read, never linked into, so removing the scratch
root afterwards cannot reach it.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable

import offline.p9_4_mix50_fresh as p94

#: The cells each path runs.
CITYFLOW_SEED, CITYFLOW_DRAW = 101, 1000
SUMO_SEED, SUMO_DRAW = 101, 1001
GUARD_DRAW = 1000


def copy_references(roots: p94.Roots, *, source_output: Path = p94.DEFAULT_OUTPUT_ROOT) -> list[str]:
    """Copy P4.7's five mix50 checkpoints and its local manifest into the scratch root's ``output/`` (read-only use of
    the real one); refuse a scratch output that is the real one."""
    if Path(roots.output_root).resolve() == Path(source_output).resolve():
        raise ValueError("the smoke's output root is the main tree's; the smoke writes only into its scratch root")
    copied = []
    names = [f"p4_7/checkpoints/mix50_dt_seed{seed}.pt" for seed in p94.SEEDS] + ["SHA256SUMS_p4_7.txt"]
    for name in names:
        target = Path(roots.output_root) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(Path(source_output) / name, target)
        copied.append(name)
    return copied


def _timed(name: str, body: Callable[[], str | None]) -> dict[str, Any]:
    """One path: PASS unless *body* raises or returns a failure reason; its seconds; never a number it produced."""
    started = time.perf_counter()
    try:
        problem = body()
    except Exception as error:  # noqa: BLE001 - a failed path is the smoke's finding, reported by name
        problem = f"{type(error).__name__}: {error}"
    return {"path": name, "result": "PASS" if problem is None else "FAIL", "seconds": time.perf_counter() - started,
            "error": problem}


def run_smoke(roots: p94.Roots, *, steps: int = 20) -> dict[str, Any]:
    """Every distinct path once at reduced size (module docstring).  Returns PASS/FAIL and seconds per path only."""
    import offline.p9_4_evaluation as ev

    copy_references(roots)
    declaration = p94.load_declaration(roots)
    results: list[dict[str, Any]] = []

    def control_path() -> str | None:
        try:
            p94.train_session(roots, hours=1.0, steps=steps)
        except p94.ControlFailed:
            pass
        record = p94.read_control(roots)
        if record is None or len(record.get("seeds", ())) != len(p94.SEEDS):
            return "the control record was not written over five seeds"
        return None

    def new_training_path() -> str | None:
        import torch

        inputs = p94.new_tier_inputs(roots, declaration)
        for run in p94.new_runs():
            if p94.run_status(roots, run, steps=steps) == "missing":
                p94.train_run(roots, run, inputs, device=torch.device("cuda"), steps=steps)
        return None

    def cityflow_path() -> str | None:
        seeds, draws = [CITYFLOW_SEED], [CITYFLOW_DRAW]
        ev.route_i(roots, declaration, seeds=seeds, draws=draws, steps=steps)
        ev.declare_route_ii(roots, seeds=seeds, draws=draws)
        outcome = ev.run_route_ii(roots, declaration, seeds=seeds, draws=draws, steps=steps)
        return None if outcome["rolled"] + outcome["skipped_already_present"] == 1 else f"route (ii) rolled {outcome}"

    def sumo_path(kind: str, cell: dict[str, Any]) -> Callable[[], str | None]:
        def body() -> str | None:
            outcome = ev.roll(roots, [(kind, cell)], declaration, workers=1, deadline_unix=None, steps=steps)
            if outcome["rolled"] != 1:
                return f"{outcome['rolled']} cells rolled, not one"
            if ev.pending(roots, kind, [cell], declaration, steps=steps):
                return "the chunk written is not reusable on resume"
            if kind == "g1":
                mismatches = ev.g1_comparison(roots, [cell])["mismatches"]
            elif kind == "g2":
                mismatches = ev.g2_comparison(roots, [cell])["mismatches"]
            else:
                mismatches = []
            if mismatches:
                return "differs under == from its committed value on " + "; ".join(
                    f"{m['cell']}: {m['differing'] + m.get('missing', [])}" for m in mismatches)
            return None

        return body

    results.append(_timed("control retrain (train_session, five seeds)", control_path))
    results.append(_timed("new training (train_run, five seeds)", new_training_path))
    trained = all(r["result"] == "PASS" for r in results)
    if trained:
        results.append(_timed("CityFlow cell, routes (i) and (ii)", cityflow_path))
        for arm in p94.SUMO_ARMS:
            cell = {"kind": "dt", "subject": p94.SUBJECT, "arm": arm, "seed": SUMO_SEED, "draw_id": SUMO_DRAW,
                    "stage": ev.STAGE_NEW}
            results.append(_timed(f"SUMO cell, {arm}", sumo_path("new", cell)))
    g1 = {"kind": "dt", "subject": p94.REGISTERED_TIER, "arm": "b_mean_k100", "seed": SUMO_SEED, "draw_id": GUARD_DRAW,
          "stage": ev.STAGE_G1}
    g2 = {"backend": "sumo", "scenario": ev.SCENARIO_KEY, "arm": "fixedtime_kcf", "draw_id": GUARD_DRAW}
    results.append(_timed("G1 cell (mix50, b_mean_k100, draw 1000)", sumo_path("g1", g1)))
    results.append(_timed("G2 cell (fixedtime_kcf, draw 1000)", sumo_path("g2", g2)))
    return {"steps": int(steps), "paths": results,
            "all_passed": all(r["result"] == "PASS" for r in results) and len(results) == 7}
