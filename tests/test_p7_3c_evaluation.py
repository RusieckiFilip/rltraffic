"""P7.3c C5 (``BRIEF_41`` C5; PREREGISTRATION A24(c)): P7.3c's three stages BESIDE P7.3d's.

The declaration, the ONE admission predicate at its sites, the targets per arm, the trained subjects' identity, the
budget each DT cell hands its loader, and the stage-1 reproduction gate.  Plan sections 9 and 13 (T-stage1, admission).

WHAT EACH GROUP PINS, AND BY WHICH ROUTE
---------------------------------------
* **The declaration (A24(c), 4,700 cells).**  The expected cells are enumerated HERE, from A24(c)'s text (nine DT arms,
  seeds 101-505, draws 1000-1099, two anchors in stage 1), never read back from the module's table; stage 1 is compared
  element for element with P7.3d's ``grid4x4_cells()`` re-labelled.
* **Admission.**  A24(a)'s four k = 5 / k = 20 pairs are admitted on P7.3c's stages and nowhere else; anchors in stage 1
  only.  Each site that admits a cell calls THE predicate: a spy raising an exception none of them catches proves the call.
* **Targets.**  Read by THIS file's route (``json``) from the committed calibration artifact, ``budgets["k{k}"]["target"]``,
  and compared under ``==``.  The three budgets' targets differ on all sixteen intersections (checked here), so a swap is
  always visible.
* **The trained identity.**  Synthetic trained checkpoints in exactly the trainer's shape -- ``few_shot.prepare_fine_tune``
  then ``few_shot.payload_for`` at the run's CLAIMED budget, no step taken -- and a synthetic training record in
  ``few_shot.build_record``'s shape whose digest is monkeypatched in as the G7 pin.  Each refusal is ONE tampering.
* **T-stage1.**  700 chunks synthesised from the COMMITTED artifact's own 700 records (``docs/data/p7_3d_grid4x4.json``,
  at its real pin, never monkeypatched here except where a test edits the record itself): the six bookkeeping fields
  changed on the DT chunks -> ``REPRODUCED 700/700``; one field per field class changed on one chunk each ->
  ``NOT REPRODUCED``, each cell named with exactly its fields and no value printed.

Nothing here executes a campaign driver or starts a process (``BRIEF_41`` Amendment E3).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import pytest
import torch

import offline.few_shot as few_shot
import offline.transfer_calibration as tc
import offline.transfer_curve as tcv
from offline.tier_sweep import warmup_for
from tests.p7_3c_fewshot_fixtures import (
    FakeIntersection,
    FakeSpatialEnv,
    source_provenance,
    write_source,
    write_training_corpus,
)
from tests.p7_3c_fixtures import calibration_ids
from tests.test_p7_3d_campaign_path import _canary_record, _grid_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"
ARTIFACT = DATA / "p7_3d_grid4x4.json"
GRID = "cityflow_grid4x4"
ZS = "mappo1000_dt_nomix_h4"
SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)
DRAWS: tuple[int, ...] = tuple(range(1000, 1100))
IDS = calibration_ids()
REPRODUCE, PRIMARY, CONTROLS = "p7_3c_reproduce", "p7_3c_primary", "p7_3c_controls"

#: A24(c)'s DT arms as (stage, subject, prompt arm), written from the registration's text.
A24_DT_ARMS: tuple[tuple[str, str, str], ...] = (
    (REPRODUCE, ZS, "b_mean_k100"),
    (PRIMARY, "ft_k5", "b_mean_k5"),
    (PRIMARY, "ft_k20", "b_mean_k20"),
    (PRIMARY, "ft_k100", "b_mean_k100"),
    (CONTROLS, ZS, "b_mean_k5"),
    (CONTROLS, ZS, "b_mean_k20"),
    (CONTROLS, "scratch_k100", "b_mean_k100"),
    (CONTROLS, "ft_k100_b1000", "b_mean_k100"),
    (CONTROLS, "ft_k100_b16000", "b_mean_k100"),
)
#: The six trained subjects, each with the budget A24(b)/(c) gives it.
TRAINED_BUDGET: dict[str, int] = {
    "ft_k5": 4000, "ft_k20": 4000, "ft_k100": 4000, "scratch_k100": 4000, "ft_k100_b1000": 1000,
    "ft_k100_b16000": 16000,
}


def _k(arm: str) -> int:
    return int(arm.rsplit("_k", 1)[1])


def _k_targets(k: int) -> dict[str, float]:
    """The sixteen targets of budget ``k{k}``, by THIS file's route."""
    artifact = json.loads((DATA / "p7_3d_calibration.json").read_text(encoding="utf-8"))
    return {
        str(ix): float(artifact["per_intersection"][ix]["budgets"][f"k{k}"]["target"])
        for ix in artifact["intersection_ids"]
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cell(
    stage: str, subject: str | None, arm: str, *, kind: str = "dt", seed: int | None = 101, draw: int = 1000,
    scenario: str | None = GRID,
) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "kind": kind, "subject": subject, "arm": arm, "seed": seed, "draw_id": draw, "scenario": scenario,
        "stage": stage,
    }
    if scenario is None:
        del cell["scenario"]
    return cell


def _anchor(stage: str, arm: str = "fixedtime", draw: int = 1000) -> dict[str, Any]:
    return _cell(stage, None, arm, kind="anchor", seed=None, draw=draw)


def _payload(cell: Mapping[str, Any], *, k: int | None = None, **overrides: Any) -> dict[str, Any]:
    """A complete, valid grid4x4 chunk for *cell* (P7.3d's own fixture), conditioned on budget ``k{k}``'s targets."""
    payload = _grid_payload(cell, config_sha="c" * 64, routes_sha="d" * 64)
    if cell["kind"] == "dt":
        targets = _k_targets(int(k if k is not None else 100))
        payload.update(
            {
                "target_rtg": dict(targets),
                "rtg_first": dict(targets),
                "rtg_last": dict(targets),
                "rtg_series": {ix: [targets[ix]] * 360 for ix in targets},
            }
        )
    payload.update(overrides)
    return payload


# ==================================================================================================================
# The declaration: A24(c)'s 4,700 cells in three stages
# ==================================================================================================================


def _identity(cell: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        cell["kind"], cell["subject"], cell["arm"], cell["seed"], cell["draw_id"], cell["scenario"], cell["stage"],
    )


def test_the_declaration_is_a24cs_4700_cells_in_three_stages() -> None:
    assert (tcv.STAGE_P7_3C_REPRODUCE, tcv.STAGE_P7_3C_PRIMARY, tcv.STAGE_P7_3C_CONTROLS, tcv.STAGE_P7_3C) == (
        REPRODUCE, PRIMARY, CONTROLS, "p7_3c",
    )
    stages = {stage: tcv.declared_cells(stage) for stage in (REPRODUCE, PRIMARY, CONTROLS)}
    whole = tcv.declared_cells(tcv.STAGE_P7_3C)
    assert [len(stages[stage]) for stage in (REPRODUCE, PRIMARY, CONTROLS)] == [700, 1500, 2500]
    assert len(whole) == 4700
    assert whole == stages[REPRODUCE] + stages[PRIMARY] + stages[CONTROLS]
    expected = [
        ("dt", subject, arm, seed, draw, GRID, stage) for stage, subject, arm in A24_DT_ARMS for seed in SEEDS
        for draw in DRAWS
    ] + [("anchor", None, arm, None, draw, GRID, REPRODUCE) for arm in ("fixedtime", "maxpressure") for draw in DRAWS]
    assert sorted((_identity(cell) for cell in whole), key=repr) == sorted(expected, key=repr)
    assert all(set(cell) == {"kind", "subject", "arm", "seed", "draw_id", "scenario", "stage"} for cell in whole)
    assert len({tcv.cell_chunk_name(cell) for cell in whole}) == 4700
    for stage in (PRIMARY, CONTROLS):
        assert [cell for cell in stages[stage] if cell["kind"] != "dt"] == []


def test_stage_one_is_p7_3ds_declaration_relabelled_and_the_older_declarations_have_not_moved() -> None:
    assert tcv.declared_cells(REPRODUCE) == [{**cell, "stage": REPRODUCE} for cell in tcv.grid4x4_cells()]
    grid = tcv.declared_cells(tcv.STAGE_GRID4X4)
    assert len(grid) == 700 and {cell["stage"] for cell in grid} == {"grid4x4_confirmatory"}
    hz1x1 = tcv.declared_cells(None)
    assert len(hz1x1) == 4700 and all("scenario" not in cell for cell in hz1x1)
    assert {cell["stage"] for cell in hz1x1} == {"confirmatory", "rest"}


def test_declarations_for_a_p7_3c_stage_is_the_whole_4700_and_that_stages_slice() -> None:
    whole_expected = tcv.declared_cells(tcv.STAGE_P7_3C)
    for stage, n in ((REPRODUCE, 700), (PRIMARY, 1500), (CONTROLS, 2500), (tcv.STAGE_P7_3C, 4700)):
        whole, sliced = tcv.declarations_for(stage, None)
        assert whole == whole_expected
        assert sliced == tcv.declared_cells(stage) and len(sliced) == n
    grid_whole, grid_slice = tcv.declarations_for(tcv.STAGE_GRID4X4, None)
    assert grid_whole == grid_slice == tcv.grid4x4_cells()


def test_the_arm_table_is_a24s_and_its_trained_rows_are_the_thirty_trainings_subjects() -> None:
    assert {(row.stage, row.subject, row.prompt_arm) for row in tcv.P7_3C_ARMS} == set(A24_DT_ARMS)
    assert len(tcv.P7_3C_ARMS) == 9
    trained = {(row.subject, row.init, row.k, row.budget) for row in tcv.P7_3C_ARMS if row.trained}
    assert trained == set(few_shot.REGISTERED_SUBJECTS)
    zero_shot = [row for row in tcv.P7_3C_ARMS if not row.trained]
    assert {row.subject for row in zero_shot} == {ZS} and {row.init for row in zero_shot} == {None}
    assert {row.budget for row in zero_shot} == {tc.DECLARED_GRADIENT_STEPS} == {few_shot.SOURCE_GRADIENT_STEPS}
    assert all(row.prompt_arm == f"b_mean_k{row.k}" for row in tcv.P7_3C_ARMS)


# ==================================================================================================================
# Admission: ONE predicate
# ==================================================================================================================


@pytest.mark.parametrize(("stage", "subject", "arm"), A24_DT_ARMS)
def test_every_declared_dt_row_is_admitted_on_its_own_stage(stage: str, subject: str, arm: str) -> None:
    row = tcv.p7_3c_admitted(_cell(stage, subject, arm))
    assert row is not None
    assert (row.stage, row.subject, row.prompt_arm, row.k) == (stage, subject, arm, _k(arm))
    assert row.budget == TRAINED_BUDGET.get(subject, 40000)


@pytest.mark.parametrize(
    ("stage", "subject", "arm"),
    [
        (PRIMARY, "ft_k100", "b_mean_k5"),  # a k = 5 target on the k = 100 fine-tune
        (PRIMARY, "ft_k5", "b_mean_k20"),
        (PRIMARY, "ft_k20", "b_mean_k100"),
        (PRIMARY, ZS, "b_mean_k5"),  # zs_k5 is stage 3's
        (REPRODUCE, ZS, "b_mean_k20"),
        (CONTROLS, "ft_k5", "b_mean_k5"),  # ft_k5 is stage 2's
        (CONTROLS, ZS, "b_mean_k100"),  # zs_k100 is stage 1's
        (REPRODUCE, "ft_k100", "b_mean_k100"),
        (PRIMARY, "ft_k100", "b_max_k100"),
        (REPRODUCE, ZS, "naive"),
    ],
)
def test_a_dt_pair_outside_its_stages_table_is_refused(stage: str, subject: str, arm: str) -> None:
    with pytest.raises(ValueError, match="is not an arm of stage"):
        tcv.p7_3c_admitted(_cell(stage, subject, arm))


def test_anchors_are_admitted_in_stage_one_only_and_only_the_two_of_them() -> None:
    for arm in ("fixedtime", "maxpressure"):
        assert tcv.p7_3c_admitted(_anchor(REPRODUCE, arm)) is None
    for stage in (PRIMARY, CONTROLS):
        with pytest.raises(ValueError, match="anchors appear only in stage 1"):
            tcv.p7_3c_admitted(_anchor(stage))
    with pytest.raises(ValueError, match="an anchor cell of stage 1 is"):
        tcv.p7_3c_admitted(_anchor(REPRODUCE, "random"))
    with pytest.raises(ValueError, match="an anchor cell of stage 1 is"):
        tcv.p7_3c_admitted({**_anchor(REPRODUCE), "subject": ZS})


def test_the_predicate_refuses_another_scenario_another_stage_and_another_kind() -> None:
    with pytest.raises(ValueError, match="P7.3c's stages are cityflow_grid4x4"):
        tcv.p7_3c_admitted(_cell(PRIMARY, "ft_k5", "b_mean_k5", scenario=None))
    with pytest.raises(ValueError, match="is not a P7.3c stage"):
        tcv.p7_3c_admitted(_cell("grid4x4_confirmatory", ZS, "b_mean_k100"))
    with pytest.raises(ValueError, match="is neither"):
        tcv.p7_3c_admitted(_cell(PRIMARY, "ft_k5", "b_mean_k5", kind="random"))


@pytest.mark.parametrize(("stage", "subject", "arm"), A24_DT_ARMS)
def test_a_valid_p7_3c_dt_chunk_validates_against_its_cell(stage: str, subject: str, arm: str) -> None:
    cell = _cell(stage, subject, arm)
    tcv.validate_cell_payload(_payload(cell, k=_k(arm)), cell=cell)


def test_validate_refuses_what_the_predicate_refuses_and_p7_3ds_stage_refuses_as_it_always_did() -> None:
    stage2 = _cell(PRIMARY, "ft_k100", "b_mean_k5")
    with pytest.raises(ValueError, match="is not an arm of stage"):
        tcv.validate_cell_payload(_payload(stage2, k=5))
    anchor2 = _anchor(PRIMARY)
    with pytest.raises(ValueError, match="anchors appear only in stage 1"):
        tcv.validate_cell_payload(_payload(anchor2))
    anchor1 = _anchor(REPRODUCE, "maxpressure")
    tcv.validate_cell_payload(_payload(anchor1), cell=anchor1)
    # On P7.3d's stage nothing new is admitted, and each refusal is today's.
    with pytest.raises(ValueError, match="is not a declared arm"):
        tcv.validate_cell_payload(_payload(_cell("grid4x4_confirmatory", ZS, "b_mean_k5"), k=5))
    with pytest.raises(ValueError, match="a grid4x4 DT cell is"):
        tcv.validate_cell_payload(_payload(_cell("grid4x4_confirmatory", "ft_k100", "b_mean_k100"), k=100))
    p7_3d = _cell("grid4x4_confirmatory", ZS, "b_mean_k100")
    tcv.validate_cell_payload(_payload(p7_3d, k=100), cell=p7_3d)


class _Admission(RuntimeError):
    """The spy's exception.  No site catches a ``RuntimeError``, so reaching it proves the site called the predicate."""


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def spy(cell: Mapping[str, Any]) -> Any:
        calls.append(dict(cell))
        raise _Admission("the one predicate was called")

    monkeypatch.setattr(tcv, "p7_3c_admitted", spy)
    return calls


def test_validate_cell_payload_admits_a_p7_3c_chunk_through_the_one_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    cell = _cell(PRIMARY, "ft_k5", "b_mean_k5")
    payload = _payload(cell, k=5)
    calls = _spy(monkeypatch)
    with pytest.raises(_Admission, match="the one predicate"):
        tcv.validate_cell_payload(payload, cell=cell)
    assert [call["subject"] for call in calls] == ["ft_k5"]


def test_chunk_is_reusable_reaches_the_one_predicate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cell = _cell(CONTROLS, ZS, "b_mean_k20")
    payload = _payload(cell, k=20)
    calls = _spy(monkeypatch)
    with pytest.raises(_Admission, match="the one predicate"):
        tcv.chunk_is_reusable(payload, cell=cell, out_root=tmp_path, output_root=tmp_path, data_dir=DATA)
    assert calls and calls[0]["arm"] == "b_mean_k20"


def test_run_cell_asks_the_predicate_before_it_resolves_any_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("run_cell resolved an input of a P7.3c cell before admitting it")

    monkeypatch.setattr(tcv, "demand_identity_for", forbidden)
    monkeypatch.setattr(tcv, "checkpoint_identity_for", forbidden)
    monkeypatch.setattr(tcv, "env_for_cell", forbidden)
    calls = _spy(monkeypatch)
    for cell in (_cell(PRIMARY, "ft_k5", "b_mean_k5"), _anchor(REPRODUCE), _anchor(CONTROLS)):
        with pytest.raises(_Admission, match="the one predicate"):
            tcv.run_cell(cell, out_root=tmp_path, output_root=tmp_path, data_dir=DATA)
    assert len(calls) == 3


def test_report_reaches_the_one_predicate_on_a_p7_3c_chunk_in_its_work_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    work = tmp_path / "cells"
    _canary_record(work)
    cell = _cell(PRIMARY, "ft_k5", "b_mean_k5")
    (work / tcv.cell_chunk_name(cell)).write_text(json.dumps(_payload(cell, k=5)), encoding="utf-8")
    out = tmp_path / "out.json"
    _spy(monkeypatch)
    with pytest.raises(_Admission, match="the one predicate"):
        tcv.report(
            work_dir=work, out_path=out, output_root=tmp_path, out_root=tmp_path, data_dir=DATA,
            stage=tcv.STAGE_GRID4X4,
        )
    assert not out.exists()


# ==================================================================================================================
# Targets per arm, from the pinned artifact
# ==================================================================================================================


def test_the_three_budgets_targets_differ_on_every_intersection_so_a_swap_is_visible() -> None:
    t5, t20, t100 = (_k_targets(k) for k in (5, 20, 100))
    assert list(t5) == list(t20) == list(t100) == IDS
    assert all(len({t5[ix], t20[ix], t100[ix]}) == 3 for ix in IDS)


@pytest.mark.parametrize(("stage", "subject", "arm"), A24_DT_ARMS)
def test_each_arms_targets_are_its_budget_read_from_the_pinned_artifact(stage: str, subject: str, arm: str) -> None:
    targets = tcv.load_grid4x4_targets_for(arm, subject=subject, stage=stage, data_dir=DATA)
    assert targets == _k_targets(_k(arm))
    assert list(targets) == IDS
    assert tcv.grid4x4_targets_for_cell(_cell(stage, subject, arm), data_dir=DATA) == targets
    if _k(arm) == 100:
        assert targets == tcv.load_grid4x4_targets(data_dir=DATA)


def test_p7_3ds_cells_keep_the_registered_prompt_through_the_same_call() -> None:
    cell = _cell("grid4x4_confirmatory", ZS, "b_mean_k100")
    assert tcv.grid4x4_targets_for_cell(cell, data_dir=DATA) == tcv.load_grid4x4_targets(data_dir=DATA)
    assert tcv.grid4x4_targets_for_cell(cell, data_dir=DATA) == _k_targets(100)


@pytest.mark.parametrize(
    ("stage", "subject", "arm", "message"),
    [
        (PRIMARY, "ft_k100", "b_mean_k5", "is not an arm of stage"),
        (REPRODUCE, ZS, "b_mean_k20", "is not an arm of stage"),
        ("grid4x4_confirmatory", ZS, "b_mean_k5", "is not a P7.3c stage"),
    ],
)
def test_a_k5_or_k20_target_is_read_only_for_a24as_four_pairs(stage: str, subject: str, arm: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        tcv.load_grid4x4_targets_for(arm, subject=subject, stage=stage, data_dir=DATA)


def _repinned_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edit: Callable[[dict[str, Any]], None]) -> Path:
    """A copy of the calibration artifact with *edit* applied, RE-PINNED so the role check is what is reached."""
    data = tmp_path / "data"
    data.mkdir()
    payload = json.loads((DATA / "p7_3d_calibration.json").read_text(encoding="utf-8"))
    edit(payload)
    path = data / "p7_3d_calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(tcv, "P7_3D_CALIBRATION_SHA256", _sha(path))
    return data


@pytest.mark.parametrize(
    ("budget", "field", "value", "arm", "subject", "stage", "message"),
    [
        ("k5", "role", "registered_prompt", "b_mean_k5", "ft_k5", PRIMARY,
         r"intersection 'B3' carries role 'registered_prompt' at k5, not 'recorded_not_evaluated'"),
        ("k100", "role", "recorded_not_evaluated", "b_mean_k100", "ft_k100", PRIMARY,
         r"intersection 'B3' carries role 'recorded_not_evaluated' at k100, not 'registered_prompt'"),
        ("k20", "statistic", "max", "b_mean_k20", ZS, CONTROLS, r"intersection 'B3'.s k20 budget is rule"),
    ],
)
def test_the_role_and_rule_of_every_intersections_budget_are_checked_before_a_target_is_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: str, field: str, value: str, arm: str, subject: str,
    stage: str, message: str,
) -> None:
    data = _repinned_calibration(
        tmp_path, monkeypatch, lambda p: p["per_intersection"]["B3"]["budgets"][budget].__setitem__(field, value)
    )
    with pytest.raises(ValueError, match=message):
        tcv.load_grid4x4_targets_for(arm, subject=subject, stage=stage, data_dir=data)


def test_an_edited_calibration_artifact_is_refused_at_its_pin_before_any_target(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "p7_3d_calibration.json").write_bytes((DATA / "p7_3d_calibration.json").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="not the pinned 3e9df8ee"):
        tcv.load_grid4x4_targets_for("b_mean_k5", subject="ft_k5", stage=PRIMARY, data_dir=data)


# ==================================================================================================================
# The trained identity: the committed record, the fence, the digests, F4's provenance guard
# ==================================================================================================================


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    return write_source(tmp_path_factory.mktemp("source") / "source.pt", provenance=source_provenance(seed=101))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_training_corpus(tmp_path_factory.mktemp("corpus") / "grid4x4_sumo_maxpressure")


@pytest.fixture(scope="module")
def trained(
    tmp_path_factory: pytest.TempPathFactory, source: tuple[Path, str], corpus: Path
) -> Callable[[str], dict[str, Any]]:
    """``subject -> payload``: the trainer's own payload for ``<subject>_seed101`` at its CLAIMED budget, no step taken."""
    cache: dict[str, dict[str, Any]] = {}
    scratch = tmp_path_factory.mktemp("prepared")

    def build(subject: str) -> dict[str, Any]:
        if subject not in cache:
            spec = few_shot.run_by_name(f"{subject}_seed101")
            prepared = few_shot.prepare_fine_tune(
                source_path=source[0], source_sha256=source[1], corpus_dir=corpus, k=spec.k, budget=spec.budget,
                seed=spec.seed, init=spec.init, device="cpu", destination=scratch / f"never_{subject}.pt",
                data_dir=DATA,
            )
            outcome = few_shot.TrainingOutcome(
                steps=spec.budget, warmup=warmup_for(spec.budget), losses=(), seconds=0.0
            )
            cache[subject] = few_shot.payload_for(prepared, outcome)
        return copy.deepcopy(cache[subject])

    return build


class _Installed:
    """One trained checkpoint, the committed training record naming it, and the pins, as a test sees them."""

    def __init__(self, output_root: Path, data: Path, path: Path, record_path: Path) -> None:
        self.output_root = output_root
        self.data = data
        self.path = path
        self.record_path = record_path


def _install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: tuple[Path, str],
    payload: Mapping[str, Any],
    *,
    subject: str,
    relative: str | None = None,
    entry_edit: Callable[[dict[str, Any]], None] | None = None,
) -> _Installed:
    """Write *payload* where the record says, a record in ``build_record``'s shape, and monkeypatch the two pins."""
    output_root = tmp_path / "output"
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / "p7_3d_calibration.json", data / "p7_3d_calibration.json")
    spec = few_shot.run_by_name(f"{subject}_seed101")
    rel = relative if relative is not None else f"p7_3c_training/checkpoints/{spec.name}.pt"
    path = output_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)
    entry: dict[str, Any] = {
        "subject": spec.subject, "init": spec.init, "k": spec.k, "budget": spec.budget, "seed": spec.seed,
        "checkpoint": rel, "checkpoint_sha256": _sha(path), "source_sha256": source[1],
    }
    if entry_edit is not None:
        entry_edit(entry)
    record = {"format_version": few_shot.RECORD_FORMAT_VERSION, "n_runs": 1, "runs": {spec.name: entry}}
    record_path = data / tcv.P7_3C_FINETUNE_NAME
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(tcv, "P7_3C_FINETUNE_SHA256", _sha(record_path))
    monkeypatch.setattr(tc, "GRID4X4_CHECKPOINT_SHA256", {**tc.GRID4X4_CHECKPOINT_SHA256, 101: source[1]})
    return _Installed(output_root, data, path, record_path)


def test_until_g7_pins_the_record_every_trained_identity_lookup_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tcv, "P7_3C_FINETUNE_SHA256", None)
    for subject in TRAINED_BUDGET:
        with pytest.raises(ValueError, match="P7_3C_FINETUNE_SHA256 is not set"):
            tcv.p7_3c_trained_checkpoint_identity(subject, 101, output_root=tmp_path, data_dir=DATA)
    cell = _cell(PRIMARY, "ft_k100", "b_mean_k100")
    with pytest.raises(ValueError, match="P7_3C_FINETUNE_SHA256 is not set"):
        tcv.checkpoint_identity_for(cell, output_root=tmp_path, data_dir=DATA)


@pytest.mark.parametrize("subject", ["ft_k5", "scratch_k100", "ft_k100_b16000"])
def test_a_trained_identity_resolves_from_the_committed_record_and_is_hashed_at_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: tuple[Path, str],
    trained: Callable[[str], dict[str, Any]], subject: str,
) -> None:
    installed = _install(tmp_path, monkeypatch, source, trained(subject), subject=subject)
    identity = tcv.p7_3c_trained_checkpoint_identity(
        subject, 101, output_root=installed.output_root, data_dir=installed.data
    )
    assert identity == {
        "subject": subject,
        "seed": 101,
        "path": str(installed.path),
        "file_sha256": _sha(installed.path),
        "sha256_checked_against": [tcv.P7_3C_FINETUNE_PIN_LABEL],
        "local_manifest": few_shot.MANIFEST_FILENAME,
        "deferred_56": False,
    }
    relative = str(installed.path.relative_to(installed.output_root))
    (installed.output_root / few_shot.MANIFEST_FILENAME).write_text(
        f"{_sha(installed.path)}  {relative}\n", encoding="utf-8"
    )
    listed = tcv.p7_3c_trained_checkpoint_identity(
        subject, 101, output_root=installed.output_root, data_dir=installed.data
    )
    assert listed["sha256_checked_against"] == [tcv.P7_3C_FINETUNE_PIN_LABEL, few_shot.MANIFEST_FILENAME]
    stage, arm = next((row[0], row[2]) for row in A24_DT_ARMS if row[1] == subject)
    routed = tcv.checkpoint_identity_for(
        _cell(stage, subject, arm), output_root=installed.output_root, data_dir=installed.data
    )
    assert routed == listed


def test_the_zero_shot_subject_keeps_a20as_identity_on_every_p7_3c_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[int, Any]] = []

    def a20a(seed: int, *, output_root: Any) -> dict[str, Any]:
        seen.append((int(seed), output_root))
        return {"subject": ZS, "seed": int(seed)}

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("the zero-shot subject reached the trained subjects' record")

    monkeypatch.setattr(tcv, "grid4x4_checkpoint_identity", a20a)
    monkeypatch.setattr(tcv, "p7_3c_trained_checkpoint_identity", forbidden)
    for stage, arm in ((REPRODUCE, "b_mean_k100"), (CONTROLS, "b_mean_k5"), (CONTROLS, "b_mean_k20")):
        assert tcv.checkpoint_identity_for(_cell(stage, ZS, arm, seed=303), output_root="/out") == {
            "subject": ZS, "seed": 303,
        }
    assert seen == [(303, "/out")] * 3


def _set(path: tuple[str, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def edit(payload: dict[str, Any]) -> None:
        node = payload
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return edit


def _drop(path: tuple[str, ...]) -> Callable[[dict[str, Any]], None]:
    def edit(payload: dict[str, Any]) -> None:
        node = payload
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]

    return edit


#: F4's provenance guard: ONE tampering of the payload each, the file then recorded at ITS digest -- so only the guard
#: can refuse it.
PAYLOAD_TAMPERINGS: list[tuple[str, str, Callable[[dict[str, Any]], None], str]] = [
    ("budget", "ft_k5", _set(("provenance", "gradient_steps"), 3999),
     r"provenance\.gradient_steps 3999 is not the declared B 4000"),
    ("budget_16000", "ft_k100_b16000", _set(("provenance", "gradient_steps"), 4000),
     r"provenance\.gradient_steps 4000 is not the declared B 16000"),
    ("source_digest", "ft_k5", _set(("provenance", "few_shot", "source_sha256"), "e" * 64),
     r"source_sha256 e{64} is not A20\(a\)'s pin"),
    ("source_digest_scratch", "scratch_k100", _set(("provenance", "few_shot", "source_sha256"), "e" * 64),
     r"source_sha256 e{64} is not A20\(a\)'s pin"),
    ("source_steps", "ft_k100", _set(("provenance", "few_shot", "source_gradient_steps"), 30000),
     r"source_gradient_steps 30000 is not the pinned source's 40000"),
    ("init", "ft_k100", _set(("provenance", "few_shot", "init"), "scratch"),
     r"provenance\.few_shot\.init 'scratch' is not 'source'"),
    ("init_scratch", "scratch_k100", _set(("provenance", "few_shot", "init"), "source"),
     r"provenance\.few_shot\.init 'source' is not 'scratch'"),
    ("k", "ft_k20", _set(("provenance", "few_shot", "k"), 5), r"provenance\.few_shot\.k 5 is not 20"),
    ("targets", "ft_k5", _set(("target_rtg",), _k_targets(100)), r"target_rtg is not b_mean_k5's 16 targets"),
    ("no_block", "ft_k5", _drop(("provenance", "few_shot")), r"records no few_shot block"),
    ("block_version", "ft_k5", _set(("provenance", "few_shot", "format_version"), "few-shot-checkpoint/0.9"),
     r"'few-shot-checkpoint/0\.9' is not 'few-shot-checkpoint/1\.0'"),
]


@pytest.mark.parametrize(
    ("case", "subject", "edit", "message"), PAYLOAD_TAMPERINGS, ids=[case[0] for case in PAYLOAD_TAMPERINGS]
)
def test_f4s_provenance_guard_refuses_each_tampering_of_the_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: tuple[Path, str],
    trained: Callable[[str], dict[str, Any]], case: str, subject: str, edit: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    payload = trained(subject)
    edit(payload)
    installed = _install(tmp_path, monkeypatch, source, payload, subject=subject)
    with pytest.raises(ValueError, match=message):
        tcv.p7_3c_trained_checkpoint_identity(subject, 101, output_root=installed.output_root, data_dir=installed.data)


def test_a_checkpoint_whose_digest_is_not_the_records_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: tuple[Path, str],
    trained: Callable[[str], dict[str, Any]],
) -> None:
    installed = _install(
        tmp_path, monkeypatch, source, trained("ft_k5"), subject="ft_k5",
        entry_edit=lambda entry: entry.__setitem__("checkpoint_sha256", "0" * 64),
    )
    with pytest.raises(ValueError, match=r"is not the committed record's 0{64}"):
        tcv.p7_3c_trained_checkpoint_identity("ft_k5", 101, output_root=installed.output_root, data_dir=installed.data)


def test_a_record_path_under_fenced_timing_is_refused_even_at_its_own_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: tuple[Path, str],
    trained: Callable[[str], dict[str, Any]],
) -> None:
    fenced = f"p7_3c_training/{few_shot.FENCED_TIMING_DIRNAME}/20260926T130659Z/ft_k5_seed101.pt"
    installed = _install(tmp_path, monkeypatch, source, trained("ft_k5"), subject="ft_k5", relative=fenced)
    assert installed.path.is_file()
    with pytest.raises(ValueError, match="lies under fenced_timing"):
        tcv.p7_3c_trained_checkpoint_identity("ft_k5", 101, output_root=installed.output_root, data_dir=installed.data)


def test_a_record_that_does_not_name_the_run_or_names_it_as_another_run_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: tuple[Path, str],
    trained: Callable[[str], dict[str, Any]],
) -> None:
    installed = _install(tmp_path, monkeypatch, source, trained("ft_k5"), subject="ft_k5")
    with pytest.raises(ValueError, match="records no run 'ft_k20_seed101'"):
        tcv.p7_3c_trained_checkpoint_identity("ft_k20", 101, output_root=installed.output_root, data_dir=installed.data)
    other = tmp_path / "other"
    renamed = _install(
        other, monkeypatch, source, trained("ft_k5"), subject="ft_k5",
        entry_edit=lambda entry: entry.__setitem__("budget", 16000),
    )
    with pytest.raises(ValueError, match="is recorded as"):
        tcv.p7_3c_trained_checkpoint_identity("ft_k5", 101, output_root=renamed.output_root, data_dir=renamed.data)


def test_a_record_that_moved_after_its_pin_is_refused_before_it_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: tuple[Path, str],
    trained: Callable[[str], dict[str, Any]],
) -> None:
    installed = _install(tmp_path, monkeypatch, source, trained("ft_k5"), subject="ft_k5")
    installed.record_path.write_text(installed.record_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match=f"{tcv.P7_3C_FINETUNE_NAME} has sha256 [0-9a-f]{{64}}, not the pinned"):
        tcv.p7_3c_trained_checkpoint_identity("ft_k5", 101, output_root=installed.output_root, data_dir=installed.data)


@pytest.mark.parametrize(
    ("listing", "message"),
    [("absent", r"is not listed in SHA256SUMS_p7_3c_finetune\.txt"), ("other", r"is not SHA256SUMS_p7_3c_finetune\.txt's")],
)
def test_a_manifest_that_exists_and_does_not_list_the_file_at_its_digest_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: tuple[Path, str],
    trained: Callable[[str], dict[str, Any]], listing: str, message: str,
) -> None:
    installed = _install(tmp_path, monkeypatch, source, trained("ft_k5"), subject="ft_k5")
    relative = str(installed.path.relative_to(installed.output_root))
    line = f"{'1' * 64}  p7_3c_training/checkpoints/ft_k20_seed101.pt\n"
    if listing == "other":
        line += f"{'2' * 64}  {relative}\n"
    (installed.output_root / few_shot.MANIFEST_FILENAME).write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        tcv.p7_3c_trained_checkpoint_identity("ft_k5", 101, output_root=installed.output_root, data_dir=installed.data)


def test_a_subject_outside_the_table_has_no_trained_identity(tmp_path: Path) -> None:
    for subject in (ZS, "ft_k50", "timing_ft_k5_b400"):
        with pytest.raises(ValueError, match="is not a trained subject"):
            tcv.p7_3c_trained_checkpoint_identity(subject, 101, output_root=tmp_path, data_dir=DATA)


# ==================================================================================================================
# The budget each DT cell hands its loader (F4, Q3): the trained run's B; the zero-shot call unchanged
# ==================================================================================================================


class _Env:
    """Sixteen intersections and a ``close``: all ``run_cell`` touches before it builds the policy."""

    def __init__(self) -> None:
        self.intersections = [FakeIntersection(ix) for ix in IDS]
        self.action_space = None
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _Chosen(RuntimeError):
    """Raised by the ``dt_choose`` stand-in once it has recorded its arguments."""


@pytest.mark.parametrize(("stage", "subject", "arm"), [*A24_DT_ARMS, ("grid4x4_confirmatory", ZS, "b_mean_k100")])
def test_run_cell_hands_dt_choose_the_arms_targets_and_a_trained_runs_declared_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str, subject: str, arm: str
) -> None:
    seen: dict[str, Any] = {}
    env = _Env()
    monkeypatch.setattr(
        tcv, "demand_identity_for",
        lambda cell, *, out_root: {"config_sha256": "c" * 64, "routes_sha256": "d" * 64, "config_path": "/none"},
    )
    monkeypatch.setattr(
        tcv, "checkpoint_identity_for",
        lambda cell, *, output_root, data_dir=None: {
            "path": f"/checkpoints/{cell['subject']}.pt", "file_sha256": "e" * 64, "sha256_checked_against": ["x"],
        },
    )
    monkeypatch.setattr(tcv, "env_for_cell", lambda cell, *, out_root: env)
    monkeypatch.setattr(tcv, "assert_env_matches_cell", lambda cell, env: None)

    def stand_in(_env: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        raise _Chosen("dt_choose reached")

    monkeypatch.setattr(tcv, "dt_choose", stand_in)
    with pytest.raises(_Chosen, match="dt_choose reached"):
        tcv.run_cell(_cell(stage, subject, arm), out_root=tmp_path, output_root=tmp_path, data_dir=DATA)
    expected: dict[str, Any] = {"checkpoint_path": f"/checkpoints/{subject}.pt", "target_rtg": _k_targets(_k(arm))}
    if subject in TRAINED_BUDGET:
        expected["declared_gradient_steps"] = TRAINED_BUDGET[subject]
    assert seen == expected
    assert env.closed == 1


def test_a_trained_payload_loads_through_dt_choose_at_its_declared_budget_and_never_at_40000(
    trained: Callable[[str], dict[str, Any]], tmp_path: Path
) -> None:
    """F4's premise, on the REAL loader (``spatial_agent_with_targets``): the branch sits at the call site."""
    path = tmp_path / "ft_k5_seed101.pt"
    torch.save(trained("ft_k5"), path)
    env = FakeSpatialEnv(IDS)
    _choose, diagnostics = tcv.dt_choose(
        env, checkpoint_path=path, target_rtg=_k_targets(5), declared_gradient_steps=4000, device="cpu"
    )
    assert diagnostics["agent"].current_rtg() == _k_targets(5)
    with pytest.raises(ValueError, match="saved at 4000 gradient steps but the declared count is 40000"):
        tcv.dt_choose(env, checkpoint_path=path, target_rtg=_k_targets(5), device="cpu")


@pytest.mark.parametrize(("stage", "subject", "arm"), A24_DT_ARMS)
def test_resume_re_derives_the_targets_of_the_chunks_own_arm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str, subject: str, arm: str
) -> None:
    cell = _cell(stage, subject, arm)
    payload = _payload(cell, k=_k(arm))
    monkeypatch.setattr(tcv, "code_changed_since", lambda *_a, **_k: [])
    monkeypatch.setattr(
        tcv, "demand_identity_for",
        lambda c, *, out_root: {"config_sha256": payload["config_sha256"], "routes_sha256": payload["routes_sha256"]},
    )
    monkeypatch.setattr(
        tcv, "checkpoint_identity_for",
        lambda c, *, output_root, data_dir=None: {"file_sha256": payload["checkpoint_sha256"]},
    )
    kwargs: dict[str, Any] = {"cell": cell, "out_root": tmp_path, "output_root": tmp_path, "data_dir": DATA}
    assert tcv.chunk_is_reusable(payload, **kwargs) is True
    swapped = _payload(cell, k=5 if _k(arm) != 5 else 100)
    tcv.validate_cell_payload(swapped, cell=cell)  # internally consistent: only the re-derivation can see it
    assert tcv.chunk_is_reusable(swapped, **kwargs) is False


# ==================================================================================================================
# T-stage1: the reproduction gate, on 700 chunks synthesised from the COMMITTED records
# ==================================================================================================================


def _committed() -> dict[str, Any]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _chunk_from_record(record: Mapping[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    """A valid ``p7.3d-grid4x4/1.1`` chunk carrying *record*'s values, with every BOOKKEEPING field moved.

    The chunk-only fields are made up to what ``validate_cell_payload`` requires: a 360 x 16 action matrix, the RTG and
    reward series (the target first), the second return route equal to the first, and A23's collision record rebuilt
    from the artifact's ``collisions`` block, one explained teleport per recorded teleport.
    """
    chunk = {key: value for key, value in record.items() if key not in ("rho_e_sumo", "rho_att_env")}
    teleports = [{"time": event["time"], "vehicle": event["victim"]} for event in events][: int(record["n_teleports"])]
    assert len(teleports) == int(record["n_teleports"]) and int(record["n_vanished_without_arrival"]) == 0
    chunk.update(
        {
            "format_version": tcv.GRID4X4_ARTIFACT_FORMAT_VERSION,
            "actions": [[0] * 16 for _ in range(int(record["decisions"]))],
            "actions_in_range": True,
            "local_return_from_lanes": dict(record["local_return"]),
            "n_observations": 3600,
            "collisions": [dict(event) for event in events],
            "n_collisions": len(events),
            "teleports": teleports,
            "n_explained_teleports": len(teleports),
            "n_unexplained_teleports": 0,
            "vanished_ids": [],
            "stage": REPRODUCE,
            "git_commit": "f" * 40,
            "seconds": float(record["seconds"]) + 17.0,
            "canary_seconds": 0.123,
        }
    )
    if record["kind"] == "dt":
        targets = record["target_rtg"]
        chunk["rtg_series"] = {ix: [targets[ix]] + [0.0] * (int(record["decisions"]) - 1) for ix in targets}
        chunk["reward_series"] = {ix: [0.0] * int(record["decisions"]) for ix in targets}
        chunk["checkpoint"] = f"/another/tree/{record['subject']}_seed{record['seed']}.pt"
        chunk["sha256_checked_against"] = ["another provenance note"]
    return chunk


@pytest.fixture(scope="module")
def stage1_work(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """700 chunks, one per committed record, each differing from its record ONLY in bookkeeping fields."""
    committed = _committed()
    events: dict[str, list[dict[str, Any]]] = {}
    for block in committed["collisions"]["per_arm"].values():
        for event in block["events"]:
            entry = {key: value for key, value in event.items() if key not in ("cell", "arm", "seed", "draw_id")}
            events.setdefault(str(event["cell"]), []).append(entry)
    work = tmp_path_factory.mktemp("stage1") / "cells"
    work.mkdir()
    for record in committed["cells"]:
        name = tcv.cell_chunk_name(record)
        chunk = _chunk_from_record(record, events.get(name, []))
        (work / name).write_text(json.dumps(chunk), encoding="utf-8")
    assert len(list(work.glob("cell_*.json"))) == 700
    return work


@contextmanager
def _edited(work: Path, name: str, edit: Callable[[dict[str, Any]], None]) -> Iterator[None]:
    """Apply *edit* to one chunk for the duration of the block, then put its exact bytes back."""
    path = work / name
    original = path.read_bytes()
    chunk = json.loads(original)
    edit(chunk)
    path.write_text(json.dumps(chunk), encoding="utf-8")
    try:
        yield
    finally:
        path.write_bytes(original)


def _dt_name(seed: int, draw: int) -> str:
    return tcv.cell_chunk_name({"subject": ZS, "arm": "b_mean_k100", "seed": seed, "draw_id": draw, "scenario": GRID})


def _anchor_name(arm: str, draw: int) -> str:
    return tcv.cell_chunk_name({"subject": None, "arm": arm, "seed": None, "draw_id": draw, "scenario": GRID})


def test_the_six_bookkeeping_fields_are_a24cs_and_the_zero_shot_pin_is_the_committed_files_digest() -> None:
    assert tcv.STAGE1_BOOKKEEPING_FIELDS == (
        "git_commit", "stage", "seconds", "canary_seconds", "checkpoint", "sha256_checked_against",
    )
    assert tcv.P7_3D_GRID4X4_NAME == ARTIFACT.name
    assert tcv.P7_3D_GRID4X4_SHA256 == _sha(ARTIFACT) and tcv.P7_3D_GRID4X4_SHA256.startswith("c63c371f")


def test_chunks_differing_only_in_the_six_bookkeeping_fields_reproduce_700_of_700(
    stage1_work: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = tcv.stage1_reproduction_check(stage1_work, data_dir=DATA)
    assert result["verdict"] == "REPRODUCED"
    assert (result["n_checked"], result["n_not_reproduced"], result["cells"]) == (700, 0, {})
    assert result["line"] == "stage1_check REPRODUCED 700/700"
    assert result["artifact_sha256"] == _sha(ARTIFACT)
    capsys.readouterr()
    assert tcv.main(["--work-dir", str(stage1_work), "--data-dir", str(DATA), "stage1-check"]) == 0
    assert capsys.readouterr().out.strip().splitlines() == ["stage1_check REPRODUCED 700/700"]


def _nudge(value: float) -> float:
    return math.nextafter(float(value), math.inf)


def test_one_field_per_class_is_refused_by_name_with_no_value_printed(
    stage1_work: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    committed = _committed()
    records = {tcv.cell_chunk_name(record): record for record in committed["cells"]}
    measurement = _dt_name(202, 1003)
    counter = _dt_name(303, 1042)  # A23's DT cell: its one teleport removed, its collision kept
    identity = _dt_name(404, 1010)
    seed = _anchor_name("maxpressure", 1050)
    per_ix = _anchor_name("fixedtime", 1077)
    absent = _dt_name(505, 1099)
    record_only = _dt_name(101, 1020)
    nudged = _nudge(records[measurement]["e_sumo"])
    # Whether one ulp of e_sumo moves rho is worked out HERE, by the formula, from the same draw's committed anchors.
    fixed = float(records[_anchor_name("fixedtime", 1003)]["e_sumo"])
    pressure = float(records[_anchor_name("maxpressure", 1003)]["e_sumo"])
    rho_moves = (fixed - nudged) / (fixed - pressure) != records[measurement]["rho_e_sumo"]
    measurement_fields = ["e_sumo", "rho_e_sumo"] if rho_moves else ["e_sumo"]

    def fewer_teleports(chunk: dict[str, Any]) -> None:
        chunk.update(n_teleports=0, teleports=[], n_explained_teleports=0)

    def one_return(chunk: dict[str, Any]) -> None:
        chunk["local_return"]["C2"] += 1.0
        chunk["local_return_from_lanes"]["C2"] += 1.0

    # The record side: one record carries a key no row produces (the artifact re-pinned at the edited copy).
    data = tmp_path / "data"
    data.mkdir()
    edited = copy.deepcopy(committed)
    next(r for r in edited["cells"] if tcv.cell_chunk_name(r) == record_only)["n_collisions"] = 0
    (data / tcv.P7_3D_GRID4X4_NAME).write_text(json.dumps(edited), encoding="utf-8")
    monkeypatch.setattr(tcv, "P7_3D_GRID4X4_SHA256", _sha(data / tcv.P7_3D_GRID4X4_NAME))

    with (
        _edited(stage1_work, measurement, lambda c: c.__setitem__("e_sumo", nudged)),
        _edited(stage1_work, counter, fewer_teleports),
        _edited(stage1_work, identity, lambda c: c.__setitem__("checkpoint_sha256", "0" * 64)),
        _edited(stage1_work, seed, lambda c: c.__setitem__("engine_seed_drawn", int(c["engine_seed_drawn"]) + 1)),
        _edited(stage1_work, per_ix, one_return),
        _edited(stage1_work, absent, lambda c: c.pop("mean_depart_delay")),
    ):
        result = tcv.stage1_reproduction_check(stage1_work, data_dir=data)
        capsys.readouterr()
        assert tcv.main(["--work-dir", str(stage1_work), "--data-dir", str(data), "stage1-check"]) == 2
        printed = capsys.readouterr().out.strip().splitlines()
    assert result["verdict"] == "NOT REPRODUCED"
    assert result["n_not_reproduced"] == 7
    assert result["cells"] == {
        measurement: {"differing": measurement_fields, "missing": []},
        counter: {"differing": ["n_teleports"], "missing": []},
        identity: {"differing": ["checkpoint_sha256"], "missing": []},
        seed: {"differing": ["engine_seed_drawn"], "missing": []},
        per_ix: {"differing": ["local_return"], "missing": []},
        absent: {"differing": [], "missing": ["mean_depart_delay"]},
        record_only: {"differing": [], "missing": ["n_collisions"]},
    }
    # The first NON-reproducing cell in DECLARED order: stage 1 lists the DT cells seed by seed, then the anchors.
    first = _dt_name(101, 1020)
    assert result["line"] == f"stage1_check NOT REPRODUCED 7/700: first {first}: lacks ['n_collisions']"
    assert printed == [result["line"]]
    assert repr(nudged) not in result["line"] and str(nudged) not in json.dumps(result)


def test_the_zero_shot_artifact_is_checked_at_its_pin_before_any_chunk_is_read(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / tcv.P7_3D_GRID4X4_NAME).write_bytes(ARTIFACT.read_bytes() + b"\n")
    empty = tmp_path / "cells"
    empty.mkdir()
    with pytest.raises(ValueError, match="not the pinned c63c371f"):
        tcv.stage1_reproduction_check(empty, data_dir=data)
    with pytest.raises(FileNotFoundError, match="is absent"):
        tcv.stage1_reproduction_check(empty, data_dir=tmp_path / "nowhere")


def test_a_missing_or_foreign_stage1_chunk_is_refused_before_any_comparison(stage1_work: Path) -> None:
    name = _dt_name(101, 1000)
    path = stage1_work / name
    original = path.read_bytes()
    path.unlink()
    try:
        with pytest.raises(ValueError, match=r"1 of the 700 stage-1 cell\(s\) have no chunk"):
            tcv.stage1_reproduction_check(stage1_work, data_dir=DATA)
    finally:
        path.write_bytes(original)
    # P7.3d's own chunk offered as stage 1's: the stage is part of the cell's identity.
    with _edited(stage1_work, name, lambda c: c.__setitem__("stage", "grid4x4_confirmatory")):
        with pytest.raises(ValueError, match="the chunk says stage='grid4x4_confirmatory'"):
            tcv.stage1_reproduction_check(stage1_work, data_dir=DATA)
    # A chunk under another cell's file name.
    with _edited(stage1_work, name, lambda c: c.__setitem__("seed", 202)):
        with pytest.raises(ValueError, match="the chunk says seed=202"):
            tcv.stage1_reproduction_check(stage1_work, data_dir=DATA)


def test_the_published_row_is_the_whitelist_and_rho_against_the_same_draws_anchors() -> None:
    committed = _committed()
    records = {tcv.cell_chunk_name(record): record for record in committed["cells"]}
    events: dict[str, list[dict[str, Any]]] = {}
    for block in committed["collisions"]["per_arm"].values():
        for event in block["events"]:
            events.setdefault(str(event["cell"]), []).append(
                {key: value for key, value in event.items() if key not in ("cell", "arm", "seed", "draw_id")}
            )
    anchors = {
        arm: _chunk_from_record(records[_anchor_name(arm, 1020)], events.get(_anchor_name(arm, 1020), []))
        for arm in ("fixedtime", "maxpressure")
    }
    name = _dt_name(505, 1020)
    row = tcv._published_row(_chunk_from_record(records[name], []), anchors, grid=True)
    assert set(row) == set(records[name]) and len(row) == 56
    expected = (float(anchors["fixedtime"]["e_sumo"]) - float(records[name]["e_sumo"])) / (
        float(anchors["fixedtime"]["e_sumo"]) - float(anchors["maxpressure"]["e_sumo"])
    )
    assert row["rho_e_sumo"] == expected == records[name]["rho_e_sumo"]
    assert {key: row[key] for key in row if key not in tcv.STAGE1_BOOKKEEPING_FIELDS} == {
        key: value for key, value in records[name].items() if key not in tcv.STAGE1_BOOKKEEPING_FIELDS
    }


# ==================================================================================================================
# The artifact's name, the CLI's stages, and report's refusal until C6
# ==================================================================================================================


def test_the_p7_3c_artifact_is_named_by_the_whole_declaration_and_a_single_stage_names_none() -> None:
    assert tcv.artifact_name_for_stage(tcv.STAGE_P7_3C) == "p7_3c_grid4x4.json" == tcv.P7_3C_ARTIFACT_NAME
    for stage in (REPRODUCE, PRIMARY, CONTROLS):
        with pytest.raises(ValueError, match="one declaration, one artifact"):
            tcv.artifact_name_for_stage(stage)
    assert [tcv.artifact_name_for_stage(s) for s in (tcv.STAGE_GRID4X4, "anchor", "confirmatory", None, "rest")] == [
        "p7_3d_grid4x4.json", "p7_3b_anchor.json", "p7_3a_zero_shot_stage1.json", "p7_3a_zero_shot.json",
        "p7_3a_zero_shot.json",
    ]


def test_the_cli_takes_p7_3cs_stages_where_each_command_runs_them() -> None:
    parser = tcv.build_parser()
    for stage in (REPRODUCE, PRIMARY, CONTROLS):
        assert parser.parse_args(["cells", "--stage", stage]).stage == stage
        assert parser.parse_args(["resume-check", "--stage", stage]).stage == stage
    assert parser.parse_args(["resume-check", "--stage", "p7_3c"]).stage == "p7_3c"
    assert parser.parse_args(["report", "--stage", "p7_3c"]).stage == "p7_3c"
    assert parser.parse_args(["stage1-check"]).command == "stage1-check"
    # argparse refuses an unregistered choice by exiting with status 2.  `cells` over the whole declaration would roll
    # the three stages in one pool and skip the stage-1 gate; `report` over one stage would publish a partial set.
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["cells", "--stage", "p7_3c"])
    for stage in (REPRODUCE, PRIMARY, CONTROLS):
        with pytest.raises(SystemExit, match="2"):
            parser.parse_args(["report", "--stage", stage])


# ==================================================================================================================
# G7's pin (Amendment F, F3.1): the committed training record, and every trained identity through it
# ==================================================================================================================

FINETUNE_RECORD = DATA / "p7_3c_finetune.json"


def test_the_trainings_record_is_pinned_at_g7s_digest() -> None:
    """F3.1: ``docs/data/p7_3c_finetune.json`` (committed at ``c616900``) hashes to the pin, and names the thirty."""
    assert tcv.P7_3C_FINETUNE_SHA256 == "adb59377edc23270ad479a542ed7d120f4b57c784f6e1f109c54624231ae79bf"
    assert _sha(FINETUNE_RECORD) == tcv.P7_3C_FINETUNE_SHA256
    record = json.loads(FINETUNE_RECORD.read_text(encoding="utf-8"))
    assert record["format_version"] == few_shot.RECORD_FORMAT_VERSION and record["n_runs"] == 30
    assert sorted(record["runs"]) == sorted(spec.name for spec in few_shot.registered_runs())


def test_every_trained_subject_identity_resolves_through_the_pinned_record() -> None:
    """F3.1: all thirty, through the REAL record at its pin and the REAL checkpoints, F4's guard included; the digests
    compared with the record read by THIS file's route and with the files hashed here."""
    root = Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT", str(REPO_ROOT / "output")))
    checkpoints = root / "p7_3c_training" / "checkpoints"
    if len(list(checkpoints.glob("*.pt"))) != 30:
        pytest.skip(f"{checkpoints} does not hold the thirty trained checkpoints (gitignored, main tree only)")
    record = json.loads(FINETUNE_RECORD.read_text(encoding="utf-8"))
    manifest = (root / few_shot.MANIFEST_FILENAME).is_file()
    stage_arm = {row.subject: (row.stage, row.prompt_arm) for row in tcv.P7_3C_ARMS if row.trained}
    for spec in few_shot.registered_runs():
        entry = record["runs"][spec.name]
        path = root / entry["checkpoint"]
        identity = tcv.p7_3c_trained_checkpoint_identity(spec.subject, spec.seed, output_root=root, data_dir=DATA)
        assert identity["path"] == str(path)
        assert identity["file_sha256"] == entry["checkpoint_sha256"] == _sha(path)
        expected = [tcv.P7_3C_FINETUNE_PIN_LABEL, *([few_shot.MANIFEST_FILENAME] if manifest else [])]
        assert identity["sha256_checked_against"] == expected
        stage, arm = stage_arm[spec.subject]
        cell = _cell(stage, spec.subject, arm, seed=spec.seed)
        assert tcv.checkpoint_identity_for(cell, output_root=root, data_dir=DATA) == identity


