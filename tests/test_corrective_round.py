"""P9.1's campaign module, ``offline/corrective_round.py`` (``BRIEF_46`` 3.4): the cell manifest, the paths, the write
barrier and the status computed from disk.

Formats exercised: ``p9.1-run/1.0`` (one write-once record per training run).  No trajectory is read or written here, so
no alignment convention applies.

THE MANIFEST IS DERIVED FROM A27, AND THIS FILE WRITES A27 OUT
-------------------------------------------------------------
Every cell the campaign trains is listed below as A27 (b), (c), (d), (e) and (g) state them; the module's manifest must
equal that literal list exactly -- a cell cannot be added, dropped or renamed without a test failing.  255 runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import offline.corrective_round as cr

HZ1X1_IQL_T_TIERS = ("random", "fixedtime", "maxpressure", "mappo500", "mappo1000", "mix33", "mix50", "mix67")
GRID_TIERS = ("mappo1000", "maxpressure", "fixedtime", "random")

#: (scenario, tier, arm, point), as the registration states them.
A27_CELLS: set[tuple[str, str, str, str]] = {
    *{("hz1x1", tier, "iql_t", "A27(c)") for tier in HZ1X1_IQL_T_TIERS},
    *{("hz1x1", "random_fresh", arm, "A27(d)") for arm in ("dt", "bc", "bc_top10", "iql", "iql_t")},
    *{("hz1x1", "mix_m1000_m500", arm, "A27(e)") for arm in ("dt", "bc", "bc_top10", "iql", "iql_t")},
    *{("hz1x1", tier, arm, "A27(g)") for tier in ("mappo1000", "mix50")
      for arm in ("dt_100k", "bc_100k", "bc_top10_100k", "iql_100k")},
    *{("grid4x4", tier, arm, "A27(b)") for tier in GRID_TIERS
      for arm in ("bc_m", "bc_top10_m", "bc_top10_perix_m", "iql_m")},
    *{("grid4x4", tier, "iql_t", "A27(c)") for tier in GRID_TIERS},
    *{("grid4x4", "random_fresh", arm, "A27(d)")
      for arm in ("dt_nomix", "bc_m", "bc_top10_m", "bc_top10_perix_m", "iql_m")},
}


def _roots(tmp_path: Path) -> Any:
    output = tmp_path / "output"
    output.mkdir()
    return cr.Roots(repo_root=tmp_path / "repo", output_root=output, corpus_root=tmp_path / "corpus",
                    draws_root=tmp_path / "draws")


# ----------------------------------------------------------------------
# The manifest
# ----------------------------------------------------------------------


def test_the_training_manifest_is_a27s_and_holds_255_runs() -> None:
    cells = cr.training_cells()
    assert {(c.scenario, c.tier, c.arm, c.point) for c in cells} == A27_CELLS
    assert len(cells) == len(A27_CELLS) == 51
    assert len(cr.training_runs()) == 255
    assert len({(c.scenario, c.key) for c in cells}) == 51, "a cell is listed twice"


def test_every_cell_is_an_arm_the_vocabulary_declares_on_a_tier_its_scenario_declares() -> None:
    import offline.method_tier_grid as grid
    import offline.tier_sweep as ts
    from offline.a27_arms import recipe

    for cell in cr.training_cells():
        assert recipe(cell.scenario, cell.arm).name == cell.arm
        if cell.scenario == "hz1x1":
            assert grid.tier_spec(cell.tier).tier == cell.tier
        else:
            assert ts.tier_spec(cell.tier).tier == cell.tier


def test_no_cell_reuses_a_registered_cell_name() -> None:
    """A27(a): every new cell carries a NEW arm or tier name -- a registered arm appears only on an A27 tier."""
    for cell in cr.training_cells():
        from offline.a27_arms import is_a27_arm

        assert is_a27_arm(cell.scenario, cell.arm) or cell.tier in ("random_fresh", "mix_m1000_m500"), cell


def test_the_references_are_the_fresh_behaviour_cells_of_both_scenarios() -> None:
    assert {(c.scenario, c.tier, c.arm) for c in cr.reference_cells()} == {
        ("hz1x1", "random_fresh", "behaviour"), ("grid4x4", "random_fresh", "behaviour")}


def test_every_point_carries_a27s_label_verbatim() -> None:
    assert cr.LABELS == {
        "A27(b)": "control added after results",
        "A27(c)": "bug fix, with results before and after",
        "A27(d)": "data-collection bug fix, results before and after",
        "A27(e)": "test registered after results",
        "A27(g)": ("the secondary registered in A8(b) on 2026-08-12 after P4.4's primary results, executed after all "
                   "primary results, one ceiling for all four arms as A8(b) requires, its `mix50` part beyond A8(b) "
                   "and labelled so"),
    }


# ----------------------------------------------------------------------
# Paths and the barrier
# ----------------------------------------------------------------------


def test_checkpoints_carry_the_names_the_route_i_clis_resolve(tmp_path: Path) -> None:
    import offline.tier_sweep as ts

    roots = _roots(tmp_path)
    hz = cr.checkpoint_path(roots, "hz1x1", "mix50", "dt_100k", 202)
    assert hz == roots.output_root / "p9_1_runs" / "checkpoints" / "hz1x1" / "mix50_dt_100k_seed202.pt"
    grid = cr.checkpoint_path(roots, "grid4x4", "random", "iql_m", 404)
    assert grid.name == ts._checkpoint_name("random", "iql_m", 404)
    assert grid.parent == roots.output_root / "p9_1_runs" / "checkpoints" / "grid4x4"


@pytest.mark.parametrize("relative", ["p9_1/x.json", "p9_1_runs/a/b.json"])
def test_the_barrier_admits_only_the_two_p9_1_directories(tmp_path: Path, relative: str) -> None:
    roots = _roots(tmp_path)
    assert cr.assert_target(roots.output_root / relative, roots) == (roots.output_root / relative).resolve()


@pytest.mark.parametrize("relative", ["p5_2/x.json", "p9_1x/x.json", "SHA256SUMS_p5_2.txt", "p9_1_runs/../p4_6/x.json"])
def test_the_barrier_refuses_everything_else_under_output(tmp_path: Path, relative: str) -> None:
    roots = _roots(tmp_path)
    with pytest.raises(PermissionError, match="p9_1"):
        cr.assert_target(roots.output_root / relative, roots)


def test_the_barrier_refuses_a_symlink_that_leaves_the_run_directory(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    (roots.output_root / "p4_6").mkdir()
    runs = roots.output_root / "p9_1_runs"
    runs.mkdir()
    (runs / "escape").symlink_to(roots.output_root / "p4_6")
    with pytest.raises(PermissionError, match="p9_1"):
        cr.assert_target(runs / "escape" / "x.json", roots)


# ----------------------------------------------------------------------
# Status, computed from disk
# ----------------------------------------------------------------------


def _write_run(roots: Any, cell: Any, seed: int, *, digest_ok: bool = True) -> Path:
    import torch

    from agent.OfflineBaselines import canonical_state_dict_digest

    path = cr.checkpoint_path(roots, cell.scenario, cell.tier, cell.arm, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = {"policy.w": torch.full((2,), float(seed))}
    torch.save({"model": model, "provenance": {"arm": cell.arm}}, path)
    record = cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, seed)
    record.parent.mkdir(parents=True, exist_ok=True)
    digest = canonical_state_dict_digest(model) if digest_ok else "0" * 64
    record.write_text(json.dumps({"format_version": cr.RUN_RECORD_FORMAT_VERSION, "scenario": cell.scenario,
                                  "tier": cell.tier, "arm": cell.arm, "seed": seed, "canonical_digest": digest,
                                  **cr.recipe_fields(cell)}), encoding="utf-8")
    return path


def test_a_run_is_done_only_with_its_checkpoint_its_record_and_matching_digests(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    cell = next(c for c in cr.training_cells() if c.arm == "iql_t" and c.scenario == "hz1x1")
    assert cr.run_status(roots, cell, 101) == "missing"
    _write_run(roots, cell, 101)
    assert cr.run_status(roots, cell, 101) == "done"


@pytest.mark.parametrize("damage", ["record_missing", "checkpoint_missing", "digest", "partial", "recipe"])
def test_anything_but_a_complete_run_is_refused_not_retrained(tmp_path: Path, damage: str) -> None:
    """A partial run is a record of a failure: it is refused, never silently retrained or skipped."""
    roots = _roots(tmp_path)
    cell = next(c for c in cr.training_cells() if c.arm == "bc_m" and c.tier == "random")
    path = _write_run(roots, cell, 202, digest_ok=damage != "digest")
    record = cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, 202)
    if damage == "record_missing":
        record.unlink()
    elif damage == "checkpoint_missing":
        path.unlink()
    elif damage == "partial":
        path.with_name(path.name + ".partial").write_bytes(b"x")
    elif damage == "recipe":
        payload = json.loads(record.read_text(encoding="utf-8"))
        payload["batch"] = 64
        record.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="refus"):
        cr.run_status(roots, cell, 202)


def test_campaign_status_is_computed_from_disk(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    cells = [c for c in cr.training_cells() if c.scenario == "grid4x4" and c.tier == "fixedtime"]
    for cell in cells[:2]:
        for seed in (101, 202, 303, 404, 505):
            _write_run(roots, cell, seed)
    status = cr.training_status(roots)
    assert status["runs_declared"] == 255 and status["runs_done"] == 10 and status["complete"] is False
    assert status["by_cell"][f"grid4x4:{cells[0].key}"] == "done"
    assert status["by_cell"][f"grid4x4:{cells[2].key}"] == "missing"
