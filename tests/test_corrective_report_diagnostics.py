"""E.7's G4 diagnostics (Q18 and Q19; ``BRIEF_46`` Amendment F, F.2-3, and Amendment S, S.2), beside
``p9_1_iql_targets.json``'s ``d4rl`` section.

The block holds two things. First, each D4RL cell's final Q and V losses and AWR near-zero fraction, for ``iql`` and
``iql_t``. Second (S.2, Q19), the same three values for every IQL-family traffic run, 115 in all: ``iql_t`` (70),
the grid's ``iql_m`` (25), hz1x1's ``iql`` and ``iql_100k`` (10 each). Each traffic row carries its arm and its
``target_dropout``, read from its record and refused unless it is its recipe's. The block is labelled as diagnostics,
not registered statistics.

Where the values are (read first, F.2-3):
- **The run record.** Every traffic ``iql_t`` run's ``p9.1-run/1.0`` record carries them in its ``diagnostics`` block,
  as ``offline_baselines.train_iql`` returns them.
- **Not ``train.log``.** It carries only the Q and V losses' means over each 5,000 steps, which is another quantity, and
  no AWR fraction.
- **The D4RL cells.** The re-run's cells carry the same block, from the same trainer. The registered ``iql`` cells are
  read from the committed P8.3 artifact the re-run was compared with; its sha256 is the re-run's
  ``compare_with.sha256``.

No format of its own: the records and the re-run's output are written here in the formats the campaign writes. Expected
values are written by hand (exact in binary), or are the committed artifact's, pinned as literals.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import offline.corrective_report as rep
import offline.corrective_round as cr

REPO = Path(__file__).resolve().parents[1]
#: The registered P8.3 IQL cells' three values, copied from docs/data/p8_3_d4rl_calibration.json (committed).
REGISTERED_IQL = {
    101: (0.3557738959789276, 0.1643904149532318, 0.04975),
    202: (0.6257127523422241, 0.3412134647369385, 0.05065),
    303: (0.7216155529022217, 0.1298157423734665, 0.0519),
    404: (0.42058315873146057, 0.1924811601638794, 0.05085),
    505: (0.38262203335762024, 0.1576024442911148, 0.052250000000000005),
}
SEEDS = (101, 202, 303, 404, 505)


def _roots(tmp_path: Path) -> cr.Roots:
    output = tmp_path / "output"
    (output / cr.RUNS_DIR).mkdir(parents=True)
    return cr.Roots(repo_root=REPO, output_root=output, corpus_root=tmp_path / "corpus", draws_root=tmp_path / "draws")


#: The IQL-family arms (S.2), named here by hand, and the target mode each trains under (A27(b), A27(c)).
IQL_FAMILY = {"iql_t": "off", "iql_m": "off", "iql": "active", "iql_100k": "active"}


def _iql_family_runs() -> list[tuple[cr.Cell, int]]:
    return [(cell, int(seed)) for cell, seed in cr.training_runs() if cell.arm in IQL_FAMILY]


def _values(k: int) -> tuple[float, float, float]:
    return 2.0 + k / 4, 1.0 + k / 8, k / 128


def _write_record(roots: cr.Roots, cell: cr.Cell, seed: int, diagnostics: Any, **fields: Any) -> Path:
    path = cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format_version": cr.RUN_RECORD_FORMAT_VERSION, "scenario": cell.scenario,
                                "tier": cell.tier, "arm": cell.arm, "seed": seed, **cr.recipe_fields(cell),
                                "diagnostics": diagnostics, "clock": cr.CLOCK_LABEL, **fields}), encoding="utf-8")
    return path


def _diagnostics(k: int, target_dropout: str = "off") -> dict[str, Any]:
    """A trainer's block: the three fields and the target mode among the others it carries."""
    q, v, near_zero = _values(k)
    return {"final_q_loss": q, "final_v_loss": v, "awr_weight_near_zero_fraction": near_zero,
            "final_policy_loss": 9.0, "advantage_mean": -1.0, "diagnostic_positions": 20000,
            "target_dropout": target_dropout}


def _write_all(roots: cr.Roots) -> None:
    for k, (cell, seed) in enumerate(_iql_family_runs()):
        _write_record(roots, cell, seed, _diagnostics(k, IQL_FAMILY[cell.arm]))


def test_the_iql_family_runs_are_the_115_written_out_by_hand() -> None:
    """S.2: ``iql_t`` 70 (hz1x1 50, the grid 20), the grid's ``iql_m`` 25, hz1x1's ``iql`` and ``iql_100k`` 10 each;
    ``target_dropout`` ``off`` for 95 and ``active`` for 20, as each recipe trains."""
    runs = _iql_family_runs()
    assert len(runs) == 115
    assert Counter((c.scenario, c.arm) for c, _ in runs) == {("hz1x1", "iql_t"): 50, ("grid4x4", "iql_t"): 20,
                                                             ("grid4x4", "iql_m"): 25, ("hz1x1", "iql"): 10,
                                                             ("hz1x1", "iql_100k"): 10}
    assert Counter(cr.recipe_fields(c)["target_dropout"] for c, _ in runs) == {"off": 95, "active": 20}
    assert all(cr.recipe_fields(c)["target_dropout"] == IQL_FAMILY[c.arm] for c, _ in runs)
    assert all(cr.recipe_fields(c)["base"] != "iql" for c, _ in cr.training_runs() if c.arm not in IQL_FAMILY)


def test_every_iql_family_run_gives_its_arm_target_mode_and_three_values_in_run_order(tmp_path: Path) -> None:
    """Only the 115 IQL-family records exist in this tree, so a reader of any other run's record fails here."""
    roots = _roots(tmp_path)
    _write_all(roots)
    expected = []
    for k, (cell, seed) in enumerate(_iql_family_runs()):
        q, v, near_zero = _values(k)
        expected.append({"scenario": cell.scenario, "tier": cell.tier, "arm": cell.arm, "seed": seed,
                         "target_dropout": IQL_FAMILY[cell.arm], "final_q_loss": q, "final_v_loss": v,
                         "awr_weight_near_zero_fraction": near_zero,
                         "source": f"output/p9_1_runs/training/{cell.scenario}/{cell.tier}__{cell.arm}__seed{seed}.json"})
    assert rep.iql_family_diagnostics(roots) == expected


@pytest.mark.parametrize(("damage", "message"), [
    (None, "carries no diagnostics block"),
    ({"final_q_loss": 1.0, "final_v_loss": 1.0, "target_dropout": "active"}, "no awr_weight_near_zero_fraction"),
    ({"final_q_loss": float("nan"), "final_v_loss": 1.0, "awr_weight_near_zero_fraction": 0.5,
      "target_dropout": "active"}, "final_q_loss is nan, not a finite number"),
    ({"final_q_loss": 1.0, "final_v_loss": float("inf"), "awr_weight_near_zero_fraction": 0.5,
      "target_dropout": "active"}, "final_v_loss is inf, not a finite number"),
    ({"final_q_loss": "60.9", "final_v_loss": 1.0, "awr_weight_near_zero_fraction": 0.5,
      "target_dropout": "active"}, "final_q_loss is '60.9', not a finite number"),
    ({"final_q_loss": 1.0, "final_v_loss": 1.0, "awr_weight_near_zero_fraction": 0.5},
     "has no target_dropout"),
    ({"final_q_loss": 1.0, "final_v_loss": 1.0, "awr_weight_near_zero_fraction": 0.5, "target_dropout": "off"},
     "target_dropout is 'off', not its recipe's 'active'"),
])
def test_a_run_record_without_its_finite_values_or_its_recipes_target_mode_is_refused(
        damage: Any, message: str, tmp_path: Path) -> None:
    """On hz1x1's ``iql`` (recipe ``active``): every damage is refused, naming the run."""
    roots = _roots(tmp_path)
    _write_all(roots)
    cell, seed = next((c, s) for c, s in _iql_family_runs() if c.arm == "iql")
    _write_record(roots, cell, seed, damage)
    with pytest.raises(ValueError, match=rf"{cell.scenario}:{cell.key} seed {seed}: .*{message}"):
        rep.iql_family_diagnostics(roots)


@pytest.mark.parametrize(("field", "value"), [("seed", 999), ("tier", "fixedtime"), ("arm", "iql_t"),
                                              ("scenario", "hz1x1"), ("format_version", "p9.1-run/0.9")])
def test_a_record_that_is_not_its_runs_is_refused(field: str, value: Any, tmp_path: Path) -> None:
    """Every identity field is compared, the tier among them (the coordinator's surviving mutant C3, S.1)."""
    roots = _roots(tmp_path)
    _write_all(roots)
    cell, seed = next((c, s) for c, s in _iql_family_runs() if c.arm == "iql_m" and c.tier == "random")
    path = cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, seed)
    record = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**record, field: value}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"is not the p9.1-run/1.0 record of {cell.scenario}:{cell.key} seed {seed}"):
        rep.iql_family_diagnostics(roots)


def _rerun(tmp_path: Path, *, sha256: str | None = None, cells: Any = None) -> Path:
    """The D4RL re-run's output in calibration/d4rl_run.py's keys the report reads."""
    registered = hashlib.sha256((REPO / rep.REGISTERED_D4RL).read_bytes()).hexdigest()
    path = tmp_path / "d4rl" / rep.D4RL_OUTPUT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if cells is None:
        cells = [{"arm": "iql_t", "seed": s, "diagnostics": _diagnostics(80 + i)} for i, s in enumerate(SEEDS)]
    path.write_text(json.dumps({"compare_with": {"path": "/trees/run-c/docs/data/p8_3_d4rl_calibration.json",
                                                 "sha256": registered if sha256 is None else sha256},
                                "cells": cells}), encoding="utf-8")
    return path


def test_the_d4rl_rows_are_the_registered_iqls_and_the_reruns_iql_ts_seed_by_seed(tmp_path: Path) -> None:
    path = _rerun(tmp_path)
    block = rep.d4rl_diagnostics(REPO, path)
    rows = [{"arm": "iql", "seed": s, "final_q_loss": q, "final_v_loss": v, "awr_weight_near_zero_fraction": z,
             "source": "docs/data/p8_3_d4rl_calibration.json"} for s, (q, v, z) in REGISTERED_IQL.items()]
    rows += [{"arm": "iql_t", "seed": s, "final_q_loss": q, "final_v_loss": v, "awr_weight_near_zero_fraction": z,
              "source": rep.D4RL_OUTPUT_NAME} for s, (q, v, z) in zip(SEEDS, (_values(80 + i) for i in range(5)))]
    assert block["rows"] == rows
    assert block["registered"] == {"file": "docs/data/p8_3_d4rl_calibration.json",
                                   "sha256": hashlib.sha256((REPO / "docs/data/p8_3_d4rl_calibration.json")
                                                            .read_bytes()).hexdigest()}
    assert block["rerun"] == {"file": rep.D4RL_OUTPUT_NAME, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_a_rerun_compared_with_another_registered_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="was compared with a file of sha256 0{64}, not with"):
        rep.d4rl_diagnostics(REPO, _rerun(tmp_path, sha256="0" * 64))


@pytest.mark.parametrize(("cells", "message"), [
    ([{"arm": "iql_t", "seed": s, "diagnostics": _diagnostics(1)} for s in SEEDS[:4]],
     r"iql_t cells for seeds \[101, 202, 303, 404\], not the five training seeds"),
    ([{"arm": "iql_t", "seed": s, "diagnostics": None if s == 303 else _diagnostics(1)} for s in SEEDS],
     "D4RL iql_t seed 303: .*carries no diagnostics block"),
])
def test_a_d4rl_arm_missing_a_seed_or_its_values_is_refused(cells: Any, message: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=message):
        rep.d4rl_diagnostics(REPO, _rerun(tmp_path, cells=cells))


def test_the_block_is_labelled_diagnostics_and_defines_its_three_fields(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_all(roots)
    path = _rerun(tmp_path)
    block = rep.iql_diagnostics(roots, path)
    assert block["label"] == rep.DIAGNOSTICS_LABEL and block["label"].startswith("diagnostics, not registered statistics")
    assert sorted(block["fields"]) == sorted(rep.DIAGNOSTIC_FIELDS) == sorted(
        ["final_q_loss", "final_v_loss", "awr_weight_near_zero_fraction"])
    assert "last gradient step" in block["fields"]["final_q_loss"] and "5,000" in block["fields"]["final_q_loss"]
    assert "below 0.01" in block["fields"]["awr_weight_near_zero_fraction"]
    assert block["d4rl"] == rep.d4rl_diagnostics(REPO, path)
    assert block["traffic_iql_family"] == {
        "source": "each IQL-family run's p9.1-run/1.0 record (iql_t, iql_m, iql, iql_100k), its diagnostics block",
        "runs": rep.iql_family_diagnostics(roots)}
    assert "traffic_iql_t" not in block
    assert len(block["d4rl"]["rows"]) == 10 and len(block["traffic_iql_family"]["runs"]) == 115


def test_build_iql_targets_puts_the_block_beside_the_d4rl_section(tmp_path: Path) -> None:
    from tests.test_corrective_report_build import _iql_sources

    roots = _roots(tmp_path)
    _write_all(roots)
    block = rep.iql_diagnostics(roots, _rerun(tmp_path))
    src = _iql_sources()
    payload = rep.build_iql_targets(rep.Sources(**{**src.__dict__, "diagnostics": block}))
    assert payload["iql_diagnostics"] == block and payload["d4rl"] == src.d4rl
    assert rep.artifact_text(payload) == rep.artifact_text(json.loads(rep.artifact_text(payload)))
