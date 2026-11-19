"""P9.4 (``docs/plans/p9.4_mix50_fresh.md``, tags ``p9.4-plan`` and ``p9.4-plan-a1``): the registered code paths are
unchanged, and the four optional keyword arguments P9.4 adds to registered functions do nothing unless they are given.

P9.4's subject, ``mix50_fresh``, is declared BESIDE the registered tables, in ``offline/p9_4_mix50_fresh.py``.  Four
registered functions know only the registered subjects, and each gains ONE keyword argument whose default is today's
behaviour:

* ``transfer_curve.run_cell(..., subject_inputs=None)`` -- an UNREGISTERED subject's prompt, facts and checkpoint
  identity, supplied by its caller; refused for a registered subject, an anchor, grid4x4 and P7.3c;
* ``transfer_curve._contrast_block(by_arm, subjects=SUBJECTS)`` and ``_in_support_block(rows, calibration,
  subjects=SUBJECTS)`` -- the subjects the block iterates over;
* ``att_rederivation.run_campaign(..., env_settings_builder=None)`` -- how a tier's evaluation env settings are found.

The defaults are pinned here against the COMMITTED P7.3a artifact (the blocks regenerate equal to the committed ones)
and by spies (the registered resolvers are the ones called).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import offline.att_rederivation as ar
import offline.method_tier_grid as grid
import offline.transfer_calibration as tcal
import offline.transfer_curve as tc

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "docs" / "data"


# ----------------------------------------------------------------------
# The registered declarations and tables are unchanged
# ----------------------------------------------------------------------


def test_p7_3a_declaration_still_holds_its_4700_cells_in_its_two_stages() -> None:
    assert len(tc.declared_cells(None)) == 4700
    assert len(tc.declared_cells(tc.STAGE_CONFIRMATORY)) == 1200
    assert len(tc.declared_cells("rest")) == 3500
    assert len(tc.declared_cells(tc.STAGE_ANCHOR)) == 700


def test_the_registered_tier_and_subject_tables_are_unchanged_and_do_not_know_mix50_fresh() -> None:
    assert sorted(grid.TIERS) == sorted(
        ["mappo1000", "mappo500", "maxpressure", "fixedtime", "random", "mix33", "mix50", "mix67"]
    )
    assert sorted(grid.A27_TIERS) == ["mix_m1000_m500", "random_fresh"]
    assert dict(grid.MIXTURE_EXPERT_FRACTION) == {"mix33": 0.33, "mix50": 0.50, "mix67": 0.67}
    assert dict(grid.A27_MIXTURE_EXPERT_FRACTION) == {"mix_m1000_m500": 0.50}
    mix50 = grid.TIERS["mix50"]
    assert (mix50.target_rtg, mix50.rtg_scale, mix50.stream_count, mix50.subsample, mix50.components) == (
        -5959.0, 40223.0, 200, "mixture", ("mappo1000", "random")
    )
    assert tuple(tc.SUBJECTS) == ("mappo1000", "mix50")
    assert sorted(tcal.SUBJECTS) == ["mappo1000", "mix50"]
    assert sorted(tc.CHECKPOINT_RECORD) == sorted(["mappo1000", "mix50", tc.ANCHOR_SUBJECT])
    for table in (grid.TIERS, grid.A27_TIERS, tcal.SUBJECTS, tc.CHECKPOINT_RECORD, tc.LOCAL_CHECKPOINT_MANIFEST):
        assert "mix50_fresh" not in table
    assert "mix50_fresh" not in tc.SUBJECTS


# ----------------------------------------------------------------------
# _contrast_block and _in_support_block: the default regenerates the committed P7.3a blocks
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def p7_3a_artifact() -> dict[str, Any]:
    return json.loads((DATA / "p7_3a_zero_shot.json").read_bytes())


def test_contrast_block_default_regenerates_the_committed_p7_3a_contrast(p7_3a_artifact: dict[str, Any]) -> None:
    by_arm = p7_3a_artifact["rho"]["by_subject_arm"]
    assert tc._contrast_block(by_arm) == p7_3a_artifact["contrast"]
    assert tc._contrast_block(by_arm, subjects=tc.SUBJECTS) == p7_3a_artifact["contrast"]


def test_in_support_block_default_regenerates_the_committed_p7_3a_block(p7_3a_artifact: dict[str, Any]) -> None:
    calibration = tc.load_calibration()
    assert tc._in_support_block(p7_3a_artifact["cells"], calibration) == p7_3a_artifact["in_support"]


def _entry(subject: str, arm: str, mean_e: float | None, mean_env: float | None) -> dict[str, Any]:
    return {"subject": subject, "arm": arm, "e_sumo": {"mean": mean_e}, "att_env": {"mean": mean_env}}


def test_contrast_block_with_subjects_reports_the_named_subject_and_its_hand_computed_difference() -> None:
    by_arm = [
        _entry("mix50_fresh", "b_mean_k100", 1.25, 0.5),
        _entry("mix50_fresh", "naive", 1.75, 0.125),
        _entry("mix50", "b_mean_k100", 9.0, 9.0),
    ]
    block = tc._contrast_block(by_arm, subjects=("mix50_fresh",))
    assert sorted(block["by_subject"]) == ["mix50_fresh"]
    assert block["by_subject"]["mix50_fresh"]["e_sumo"] == {
        "calibrated_mean_rho": 1.25, "naive_mean_rho": 1.75, "difference": -0.5,
    }
    assert block["by_subject"]["mix50_fresh"]["att_env"]["difference"] == 0.375
    # The default iterates over the registered subjects only: mix50_fresh is not reported, mix50 lacks naive.
    assert tc._contrast_block(by_arm)["by_subject"] == {}


def test_in_support_block_with_subjects_reads_the_named_subjects_targets() -> None:
    def row(rule: str, statistic: str, k: int | None, role: str, target: float) -> dict[str, Any]:
        return {"rule": rule, "statistic": statistic, "k": k, "role": role, "target_rtg": target,
                "in_support": {"position": "inside", "target_rtg": target}}

    calibration = {"targets": {"mix50_fresh": [
        row("rule_b", "mean", 100, "registered_prompt", -7431.0),
        row("rule_b", "max", 100, "ablation", -6000.0),
        row("rule_a", "q1.0", 100, "ablation", -20809.0),
        row("naive", "none", None, "ablation", -5959.0),
    ]}}
    rows = [{"kind": "dt", "subject": "mix50_fresh", "arm": "naive", "n_decisions_in_support": 300},
            {"kind": "dt", "subject": "mix50_fresh", "arm": "naive", "n_decisions_in_support": 60},
            {"kind": "anchor", "subject": None, "arm": "fixedtime", "n_decisions_in_support": None}]
    block = tc._in_support_block(rows, calibration, subjects=("mix50_fresh",))
    assert sorted(block["declared_by_the_calibration_artifact"]) == ["mix50_fresh"]
    assert block["declared_by_the_calibration_artifact"]["mix50_fresh"]["naive"]["target_rtg"] == -5959.0
    assert block["observed_decisions"] == {"mix50_fresh/naive": {"n_cells": 2, "total_decisions_in_support": 360}}


# ----------------------------------------------------------------------
# run_cell's subject_inputs: refused where it does not belong, BEFORE any input is resolved
# ----------------------------------------------------------------------


def _facts(subject: str) -> tcal.SubjectFacts:
    return tcal.SubjectFacts(subject=subject, best_source_return=-5959.0, rtg_scale=41000.0,
                             support_range_over_the_split=(-41000.0, -6.0), n_rows=10, training_set_return_min=-41000.0,
                             checkpoints=("a.pt",), state_dim=25, context_length=20)


def _inputs(subject: str = "mix50_fresh") -> dict[str, Any]:
    return {"target_rtg": -7431.0185327454665, "facts": _facts(subject),
            "checkpoint": {"subject": subject, "seed": 101, "path": "/nonexistent/x.pt", "file_sha256": "0" * 64,
                           "sha256_checked_against": ["a record"], "local_manifest": None, "deferred_56": False}}


def _cell(**overrides: Any) -> dict[str, Any]:
    cell = {"kind": "dt", "subject": "mix50_fresh", "arm": "b_mean_k100", "seed": 101, "draw_id": 1001, "stage": "p9_4"}
    cell.update(overrides)
    return cell


class _NotReached(Exception):
    """Raised by a spy that the code under test must not reach."""


@pytest.fixture()
def no_inputs_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise _NotReached("an input was resolved before the refusal")

    for name in ("demand_identity_for", "checkpoint_identity_for", "load_calibration", "targets_for_subject",
                 "env_for_cell", "p7_3c_admitted"):
        monkeypatch.setattr(tc, name, refuse)
    monkeypatch.setattr(tcal, "subject_facts", refuse)


@pytest.mark.parametrize(
    ("cell", "why"),
    [
        (_cell(subject="mix50"), "registered subject"),
        (_cell(subject="mappo1000"), "registered subject"),
        (_cell(subject=tc.ANCHOR_SUBJECT, arm=tc.ANCHOR_ARM), "registered subject"),
        (_cell(kind="anchor", subject=None, arm="fixedtime", seed=None), "only a DT cell"),
        (_cell(scenario=tc.GRID4X4_SCENARIO_KEY), "hz1x1 only"),
        (_cell(stage=tc.STAGE_P7_3C_PRIMARY), "not a P7.3c cell"),
    ],
)
def test_run_cell_refuses_subject_inputs_where_they_do_not_belong_before_any_input(
    no_inputs_resolved: None, cell: dict[str, Any], why: str
) -> None:
    inputs = _inputs(str(cell.get("subject") or "mix50_fresh"))
    with pytest.raises(ValueError, match=why):
        tc.run_cell(cell, out_root="/nonexistent", output_root="/nonexistent", subject_inputs=inputs)


@pytest.mark.parametrize(
    ("mutate", "why"),
    [
        (lambda i: i.pop("facts"), "carry exactly"),
        (lambda i: i.update(extra=1), "carry exactly"),
        (lambda i: i.update(target_rtg=float("nan")), "finite"),
        (lambda i: i.update(facts=_facts("someone_else")), "facts are"),
        (lambda i: i["checkpoint"].update(subject="someone_else"), "checkpoint identity"),
        (lambda i: i["checkpoint"].pop("file_sha256"), "checkpoint identity"),
        (lambda i: i["checkpoint"].update(seed=202), "checkpoint identity"),
        (lambda i: i.update(declared_gradient_steps=0), "positive whole number"),
        (lambda i: i.update(declared_gradient_steps=True), "positive whole number"),
        (lambda i: i.update(declared_gradient_steps="20"), "positive whole number"),
    ],
)
def test_run_cell_refuses_malformed_subject_inputs(no_inputs_resolved: None, mutate: Any, why: str) -> None:
    inputs = _inputs()
    mutate(inputs)
    with pytest.raises(ValueError, match=why):
        tc.run_cell(_cell(), out_root="/nonexistent", output_root="/nonexistent", subject_inputs=inputs)


def test_run_cell_with_subject_inputs_uses_them_and_never_the_registered_resolvers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved: list[str] = []

    def refuse(name: str) -> Any:
        def spy(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError(f"{name} was called on the injected path")

        return spy

    for name in ("checkpoint_identity_for", "load_calibration", "targets_for_subject"):
        monkeypatch.setattr(tc, name, refuse(name))
    monkeypatch.setattr(tcal, "subject_facts", refuse("subject_facts"))
    monkeypatch.setattr(tc, "demand_identity_for", lambda cell, *, out_root: resolved.append("demand") or {
        "config_path": "/nonexistent/noteleport.sumocfg", "config_sha256": "c", "routes_sha256": "r"})

    def env_reached(cell: Any, *, out_root: Any, sentinel_out_dir: Any = "/nonexistent") -> Any:
        resolved.append("env")
        raise _NotReached("the env is where this test stops")

    monkeypatch.setattr(tc, "env_for_cell", env_reached)
    with pytest.raises(_NotReached, match="where this test stops"):
        tc.run_cell(_cell(), out_root="/nonexistent", output_root="/nonexistent", subject_inputs=_inputs())
    assert resolved == ["demand", "env"]


@pytest.mark.parametrize(("extra", "expected"), [({}, None), ({"declared_gradient_steps": 20}, 20)])
def test_run_cell_hands_the_supplied_budget_to_dt_choose_and_the_default_otherwise(
    monkeypatch: pytest.MonkeyPatch, extra: dict[str, Any], expected: int | None
) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(tc, "demand_identity_for", lambda cell, *, out_root: {
        "config_path": "/nonexistent/noteleport.sumocfg", "config_sha256": "c", "routes_sha256": "r"})
    closed: list[bool] = []
    env = SimpleNamespace(intersections=[SimpleNamespace(id="intersection_1_1", num_phases=8)], action_space=None,
                          close=lambda: closed.append(True))
    monkeypatch.setattr(tc, "env_for_cell", lambda cell, *, out_root, sentinel_out_dir="/nonexistent": env)
    monkeypatch.setattr(tc, "assert_env_matches_cell", lambda cell, env: None)

    def choose(env: Any, **kwargs: Any) -> Any:
        seen.append(dict(kwargs))
        raise _NotReached("dt_choose is where this test stops")

    monkeypatch.setattr(tc, "dt_choose", choose)
    with pytest.raises(_NotReached, match="where this test stops"):
        tc.run_cell(_cell(), out_root="/nonexistent", output_root="/nonexistent", subject_inputs={**_inputs(), **extra})
    assert closed == [True]
    assert seen[0]["checkpoint_path"] == "/nonexistent/x.pt"
    assert seen[0]["target_rtg"] == -7431.0185327454665
    assert seen[0].get("declared_gradient_steps") == expected
    assert ("declared_gradient_steps" in seen[0]) is (expected is not None)


def test_run_cell_without_subject_inputs_takes_the_registered_route_for_an_unregistered_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default is today's code: an unregistered subject reaches the registered target lookup, which refuses it."""
    monkeypatch.setattr(tc, "demand_identity_for", lambda cell, *, out_root: {
        "config_path": "/nonexistent/noteleport.sumocfg", "config_sha256": "c", "routes_sha256": "r"})
    monkeypatch.setattr(tc, "load_calibration", lambda *_a, **_k: {"targets": {}})
    with pytest.raises(KeyError, match="mix50_fresh"):
        tc.run_cell(_cell(), out_root="/nonexistent", output_root="/nonexistent")


# ----------------------------------------------------------------------
# run_campaign's env_settings_builder
# ----------------------------------------------------------------------


class _Episode:
    att_ours = 1.5

    def as_record(self) -> dict[str, Any]:
        return {"scenario": "hz1x1", "arm": "dt@tier_x", "seed": 101, "draw_id": 1000, "att_ours": 1.5}


def _campaign(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    import offline.admission_probe as probe
    import offline.materialise_draws as draws

    config = tmp_path / "draw" / "cityflow.json"
    monkeypatch.setattr(draws, "draw_config_path", lambda key, draw, *, out_root: config)
    monkeypatch.setattr(probe, "created_from_flow", lambda path, *, horizon_seconds: 7)
    monkeypatch.setattr(probe, "probe_episode", lambda **_kwargs: _Episode())
    roots = SimpleNamespace(work_dir=tmp_path / "work", draws_root=tmp_path, corpus_root=tmp_path,
                            repo_root=tmp_path, output_root=tmp_path)
    (tmp_path / "work").mkdir()
    cell = ar.CellKey("hz1x1", "dt@tier_x", 101, 1000)
    return ar.run_campaign(
        [cell], roots=roots, engine_seed=1000, device=None, committed={("hz1x1", "dt@tier_x", 101, 1000): 1.5},
        protected=(), factory_builder=lambda scenario, arm, seed, roots, *, device, config_path: (object(), {"k": 1}),
        **kwargs,
    )


def test_run_campaign_default_finds_the_settings_through_the_registered_resolver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []

    def registered(scenario: str, tier: str, roots: Any) -> dict[str, Any]:
        calls.append((scenario, tier))
        return {"max_steps": 360, "delta_time": 10}

    monkeypatch.setattr(ar, "rederivation_env_settings", registered)
    outcome = _campaign(monkeypatch, tmp_path)
    assert (outcome["rolled"], outcome["n_refused"], calls) == (1, 0, [("hz1x1", "tier_x")])


def test_run_campaign_with_a_builder_uses_it_and_never_the_registered_resolver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []

    def registered(*_args: Any) -> dict[str, Any]:
        raise AssertionError("the registered resolver was called although a builder was given")

    def builder(scenario: str, tier: str, roots: Any) -> dict[str, Any]:
        calls.append((scenario, tier))
        return {"max_steps": 360, "delta_time": 10}

    monkeypatch.setattr(ar, "rederivation_env_settings", registered)
    outcome = _campaign(monkeypatch, tmp_path, env_settings_builder=builder)
    assert (outcome["rolled"], outcome["n_refused"], calls) == (1, 0, [("hz1x1", "tier_x")])
    written = json.loads((tmp_path / "work" / ar.cell_file_name(ar.CellKey("hz1x1", "dt@tier_x", 101, 1000))).read_bytes())
    assert written["reproduces_committed"] is True
    assert written["committed_att_ours"] == 1.5
