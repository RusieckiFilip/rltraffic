"""P5.3c C3 (``BRIEF_42`` §3 C3, §4 T-gate / T-report; A26(c) as corrected by A26.1(b); Amendment A Q2 with A.1): the
campaign -- its declaration, one cell, the chunks, the reference gate, the fenced re-roll and the report.

What this file pins, and the named mutations it is built against:

* **The declaration** -- 7,000 cells, the 1,000 reference cells (P4's and P4.7's published K = 20 checkpoints) before the
  6,000 sweep cells, exactly the table written out in ``tests/p5_3c_campaign_fixtures.py``.
* **The checkpoints** -- a sweep checkpoint only through the PINNED training record (``TRAIN_RECORD_SHA256``: while it is
  unset every lookup refuses), hashed at consumption, listed in the training manifest, never under ``fenced_timing/``; a
  published one by C2's own rules.
* **One cell** -- P8.4b's path (``_dt_factory``'s load-then-target, greedy, ``update_memory=True``; ``probe_episode`` with
  P8.4b's env settings, config, ``created`` and engine seed 1000) on CUDA, the subject's registered prompt, the
  per-decision series; it writes nothing.  The caller-level twin of C1's T-k1 / T-k2 through the campaign's OWN decision
  factory (plan F6).  *Mutation:* the window at K + 1 -> the twin dies.
* **T-gate (load-bearing)** -- rows equal -> passes, its record written once; ONE row off by 1 ULP -> fails, nothing
  written, and the sweep stage rolls nothing; a missing (seed, draw) refuses.  *Mutation:* the gate skipped -> dies.
* **The fenced re-roll** (Amendment A, Q2): the campaign's own fifteen reference cells, MATCH / NO MATCH by field NAME,
  its records fenced under ``g2/``, a failed roll never a verdict.
* **T-report** -- every refusal precedes every write (the rows at another digest; the record's pin unset; one declared
  cell absent; a chunk at a checkpoint digest the record does not name; no ``canary.json``; no gate record; a chunk rolled
  by other code; an undeclared file; another thread count; another demand); the verdict equals an INDEPENDENT
  recomputation from the raw chunk files (the test's own per-draw means, its own contrast, the second Wilcoxon route,
  its own Holm); the artifact regenerates byte for byte and carries no per-decision series.  *Mutations:* each refusal
  removed in turn -> its case dies.

No test here runs CityFlow or CUDA except the two gated at the end (plan §12 / Q17, on CUDA per A.1), which name what
they consume when they skip.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import offline.context_sweep as cs
from offline.admission_probe import AdmissionEpisode, created_from_flow
from offline.materialise_draws import draw_config_path
from offline.transfer_calibration import record_canary
from tests.p5_3c_campaign_fixtures import (
    CANARY_LINE,
    DRAWS,
    PROMPT,
    SEEDS,
    CampaignTree,
    arm_of,
    build_campaign_tree,
    build_checkpoint_tree,
    build_draws,
    chunk_name,
    declared_table,
    demand_of,
    episode_record,
    fixture_att,
    fixture_chunk,
    head_commit,
    registered_table,
)
from tests.p5_3c_fixtures import IX_ID, N_ACTIONS, STATE_DIM, T_DECISIONS, StubEnv, info_at, write_single_ix_corpus
from tests.test_p5_3c_statistic import holm_by_hand, route_b

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"
C1A_COMMIT = "c507721348e71b9689aa224290f307fd25a9b32f"
#: Amendment C, C0: the sixty trainings' record as the coordinator committed it on main at ``c55693a`` (gate G3).
G3_RECORD_SHA256 = "017808a5e84fada469d6b2d302889ad171b8e429b1612b8ac06c900ef3c6321a"


def _pin(mp: pytest.MonkeyPatch, tree: CampaignTree) -> None:
    mp.setattr(cs, "REFERENCE_ROWS_SHA256", tree.rows_sha256, raising=True)
    mp.setattr(cs, "TRAIN_RECORD_SHA256", tree.record_sha256, raising=True)


def _cell(stage: str, subject: str, k: int, batch: int, seed: int, draw: int) -> cs.CampaignCell:
    return cs.CampaignCell(stage=stage, subject=subject, k=k, batch=batch, seed=seed, draw_id=draw)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def _rewrite(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> CampaignTree:
    """The whole fake campaign ONCE per module: 7,000 chunks, ``canary.json`` and the gate record."""
    tree = build_campaign_tree(tmp_path_factory.mktemp("campaign"))
    with pytest.MonkeyPatch.context() as mp:
        _pin(mp, tree)
        record_canary(CANARY_LINE, work_dir=cs.cells_dir(tree.output_root))
        cs.write_gate_record(tree.output_root, cs.reference_gate(output_root=tree.output_root, data_dir=tree.data_dir))
    return tree


@pytest.fixture
def campaign(built: CampaignTree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CampaignTree:
    """A private copy of the built campaign, its two pins set."""
    root = tmp_path / "campaign"
    shutil.copytree(built.root, root, symlinks=True)
    checkpoints = dataclasses.replace(
        built.checkpoints, output_root=root / "output", data_dir=root / "docs_data",
        record_path=root / "docs_data" / "p5_3c_train.json",
    )
    tree = dataclasses.replace(
        built, root=root, output_root=root / "output", data_dir=root / "docs_data", draws_root=root / "draws",
        corpus_root=root / "corpus", checkpoints=checkpoints,
    )
    _pin(monkeypatch, tree)
    return tree


def _build(tree: CampaignTree) -> dict[str, Any]:
    return cs.build_context_sweep_artifact(
        output_root=tree.output_root, corpus_root=tree.corpus_root, draws_root=tree.draws_root, data_dir=tree.data_dir
    )


# ======================================================================================================================
# The declaration and the layout
# ======================================================================================================================


def test_the_campaign_declares_7000_cells_the_reference_stage_first_exactly_the_written_out_table() -> None:
    cells = cs.declared_campaign_cells()
    assert [(c.stage, c.subject, c.k, c.batch, c.seed, c.draw_id) for c in cells] == declared_table()
    assert len(cells) == 7000
    assert {c.stage for c in cells[:1000]} == {"reference"} and {c.stage for c in cells[1000:]} == {"sweep"}
    assert len(cs.declared_campaign_cells("reference")) == 1000 and len(cs.declared_campaign_cells("sweep")) == 6000
    names = [c.name for c in cells]
    assert len(set(names)) == 7000
    assert names == [chunk_name(*row) for row in declared_table()]
    reference = cells[0]
    assert (reference.arm, reference.run) == ("ref_mappo1000_k20", None)
    sweep = cs.campaign_cell_by_name("cell_mappo1000_k1_b1280_seed303_draw1042.json")
    assert (sweep.arm, sweep.run) == ("mappo1000_k1_b1280", cs.RunSpec("mappo1000", 1, 1280, 303))
    assert cs.campaign_cell_by_name(reference.name) == reference
    with pytest.raises(ValueError, match="not one of the 7,000"):
        cs.campaign_cell_by_name("cell_mappo1000_k3_b64_seed101_draw1000.json")
    with pytest.raises(ValueError, match="stage"):
        cs.declared_campaign_cells("warmup")
    assert {c.arm for c in cs.declared_campaign_cells("reference")} == {"ref_mappo1000_k20", "ref_mix50_k20"}


def test_the_campaign_lives_under_output_p5_3c_and_its_manifest_beside_it(tmp_path: Path) -> None:
    root = tmp_path / "output"
    cell = cs.declared_campaign_cells()[0]
    assert cs.campaign_root(root) == root / "p5_3c"
    assert cs.cells_dir(root) == root / "p5_3c" / "cells"
    assert cs.chunk_path(root, cell) == root / "p5_3c" / "cells" / cell.name
    assert cs.gate_record_path(root) == root / "p5_3c" / "reference_gate.json"
    assert cs.artifact_path(root) == root / "p5_3c" / "artifacts" / "p5_3c_context_sweep.json"
    assert cs.campaign_manifest_path(root) == root / "SHA256SUMS_p5_3c.txt"
    assert cs.g2_dir(root) == root / "p5_3c" / "g2"
    assert cs.CAMPAIGN_OUTPUT_ENTRIES == ("p5_3c", "SHA256SUMS_p5_3c.txt")


# ======================================================================================================================
# The checkpoints
# ======================================================================================================================


def test_every_sweep_lookup_refuses_while_the_training_records_pin_is_unset(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """The unset pin's refusal, exercised by setting it unset (Amendment C set the pin at G3; P7.3c's ``d9a20fd``
    pattern -- until C3's commit this test asserted the module's pin was still ``None``)."""
    tree = build_checkpoint_tree(tmp_path)
    monkeypatch.setattr(cs, "TRAIN_RECORD_SHA256", None, raising=True)
    with pytest.raises(ValueError, match="TRAIN_RECORD_SHA256 is not set"):
        cs.load_train_record(tree.data_dir)
    run = cs.RunSpec("mappo1000", 5, 64, 101)
    with pytest.raises(ValueError, match="TRAIN_RECORD_SHA256 is not set"):
        cs.sweep_checkpoint(run, output_root=tree.output_root, data_dir=tree.data_dir)
    with pytest.raises(ValueError, match="TRAIN_RECORD_SHA256 is not set"):
        cs.checkpoint_for_cell(_cell("sweep", "mappo1000", 5, 64, 101, 1000), output_root=tree.output_root,
                               data_dir=tree.data_dir)


def test_the_training_record_is_read_at_its_pin_before_it_is_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tree = build_checkpoint_tree(tmp_path)
    monkeypatch.setattr(cs, "TRAIN_RECORD_SHA256", tree.record_sha256, raising=True)
    assert cs.load_train_record(tree.data_dir)["n_runs"] == 60
    monkeypatch.setattr(cs, "TRAIN_RECORD_SHA256", "0" * 64, raising=True)
    with pytest.raises(ValueError, match="sha256"):
        cs.load_train_record(tree.data_dir)
    monkeypatch.setattr(cs, "TRAIN_RECORD_SHA256", tree.record_sha256, raising=True)
    tree.record_path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        cs.load_train_record(tree.data_dir)


def test_the_training_record_is_pinned_at_g3s_digest_and_names_the_sixty() -> None:
    """Amendment C (C0, C4.1): ``docs/data/p5_3c_train.json`` as committed at ``c55693a`` hashes -- by ``hashlib``
    here -- to the pin, and names exactly the sixty runs of the table the fixtures write out."""
    assert cs.TRAIN_RECORD_SHA256 == G3_RECORD_SHA256
    assert hashlib.sha256((DATA / "p5_3c_train.json").read_bytes()).hexdigest() == G3_RECORD_SHA256
    record = cs.load_train_record(DATA)
    assert record["format_version"] == "p5.3c-train-record/1.0" and record["n_runs"] == 60
    assert sorted(record["runs"]) == sorted(f"{s}_k{k}_b{b}_seed{seed}" for s, k, b, seed in registered_table())


def test_a_sweep_checkpoint_is_the_pinned_records_listed_in_its_manifest_and_hashed_at_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = build_checkpoint_tree(tmp_path)
    monkeypatch.setattr(cs, "TRAIN_RECORD_SHA256", tree.record_sha256, raising=True)
    for subject, k, batch, seed in registered_table():
        run = cs.RunSpec(subject, k, batch, seed)
        identity = cs.sweep_checkpoint(run, output_root=tree.output_root, data_dir=tree.data_dir)
        entry = tree.runs[run.name]
        assert (identity["path"], identity["sha256"]) == (entry["checkpoint"], entry["checkpoint_sha256"])
        assert cs.checkpoint_for_cell(
            _cell("sweep", subject, k, batch, seed, 1000), output_root=tree.output_root, data_dir=tree.data_dir
        ) == identity
    run = cs.RunSpec("mix50", 10, 64, 202)
    path = tree.output_root / tree.runs[run.name]["checkpoint"]
    original = path.read_bytes()
    path.write_bytes(original + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        cs.sweep_checkpoint(run, output_root=tree.output_root, data_dir=tree.data_dir)
    path.write_bytes(original)

    manifest = tree.output_root / "SHA256SUMS_p5_3c_train.txt"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text("".join(f"{line}\n" for line in lines if run.name not in line), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256SUMS_p5_3c_train"):
        cs.sweep_checkpoint(run, output_root=tree.output_root, data_dir=tree.data_dir)


@pytest.mark.parametrize("where", ["fenced_timing", "absolute", "another_run"])
def test_a_record_entry_pointing_elsewhere_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str) -> None:
    tree = build_checkpoint_tree(tmp_path)
    record = _read(tree.record_path)
    run = cs.RunSpec("mappo1000", 2, 640, 404)
    entry = record["runs"][run.name]
    if where == "fenced_timing":
        entry["checkpoint"] = "p5_3c_training/fenced_timing/20260929T112505Z/alone.pt"
    elif where == "absolute":
        entry["checkpoint"] = str(tree.output_root / entry["checkpoint"])
    else:
        entry["checkpoint"] = record["runs"]["mappo1000_k2_b640_seed101"]["checkpoint"]
        entry["checkpoint_sha256"] = record["runs"]["mappo1000_k2_b640_seed101"]["checkpoint_sha256"]
    _rewrite(tree.record_path, record)
    monkeypatch.setattr(cs, "TRAIN_RECORD_SHA256", cs._sha256_file(tree.record_path), raising=True)
    # The path check's own words: a later refusal (the file absent, the payload's seed) would also name the run
    # (mutant C9 survived a match on the name alone).
    with pytest.raises(ValueError, match=f"{run.name}: the pinned record's checkpoint .* is not its registered destination"):
        cs.sweep_checkpoint(run, output_root=tree.output_root, data_dir=tree.data_dir)


def test_a_published_checkpoint_is_verified_by_c2s_own_rules(tmp_path: Path) -> None:
    tree = build_checkpoint_tree(tmp_path)
    ten = cs._published_k20(tree.output_root, tree.data_dir)
    for (subject, seed), entry in ten.items():
        identity = cs.published_checkpoint(subject, seed, output_root=tree.output_root, data_dir=tree.data_dir)
        assert (identity["path"], identity["sha256"]) == (entry["reference"], entry["sha256"])
        assert identity == cs.checkpoint_for_cell(
            _cell("reference", subject, 20, 64, seed, 1099), output_root=tree.output_root, data_dir=tree.data_dir
        )
    path = tree.output_root / "p4_7" / "checkpoints" / "mix50_dt_seed303.pt"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="mix50_dt_seed303"):
        cs.published_checkpoint("mix50", 303, output_root=tree.output_root, data_dir=tree.data_dir)


# ======================================================================================================================
# One cell
# ======================================================================================================================


class _FakeAgent:
    """What ``agent_with_target`` returns, recording every call; RTG and actions scripted."""

    def __init__(self, target: float) -> None:
        self.target = target
        self.reward_sum = 0.0
        self.calls: list[dict[str, Any]] = []

    def current_rtg(self) -> dict[str, float]:
        return {IX_ID: self.target - self.reward_sum}

    def act(self, info: dict[str, Any], *, explore: bool, update_memory: bool) -> np.ndarray:
        self.calls.append({"explore": explore, "update_memory": update_memory, "step": info["step"]})
        if update_memory and info["step"] > 0:
            self.reward_sum += float(info["intersections"][IX_ID]["reward"])
        return np.asarray([(3 * int(info["step"]) + 1) % N_ACTIONS], dtype=np.int64)


@pytest.mark.parametrize(
    "row",
    [("reference", "mappo1000", 20, 64, 101, 1000), ("sweep", "mix50", 2, 64, 303, 1042),
     ("sweep", "mappo1000", 1, 1280, 505, 1099)],
    ids=["reference-p4", "sweep-mix50", "equal-supervision"],
)
def test_one_cell_takes_p8_4bs_path_on_cuda_with_the_registered_prompt_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row: tuple[str, str, int, int, int, int]
) -> None:
    tree = build_checkpoint_tree(tmp_path)
    draws_root = build_draws(tmp_path)
    monkeypatch.setattr(cs, "TRAIN_RECORD_SHA256", tree.record_sha256, raising=True)
    stage, subject, k, batch, seed, draw = row
    cell = _cell(*row)
    settings = {"max_steps": 360, "delta_time": 10, "fixture": True}
    seen_settings: list[tuple[str, Path]] = []

    def fake_settings(name: str, corpus_root: str | Path) -> dict[str, Any]:
        seen_settings.append((name, Path(corpus_root)))
        return dict(settings)

    monkeypatch.setattr(cs, "evaluation_env_settings", fake_settings, raising=True)
    agents: list[tuple[_FakeAgent, dict[str, Any]]] = []

    def fake_agent_with_target(env: Any, path: Any, *, declared_gradient_steps: int, target_rtg: float,
                               device: str | None = None) -> _FakeAgent:
        agent = _FakeAgent(float(target_rtg))
        agents.append((agent, {"path": Path(path), "declared": declared_gradient_steps, "target": target_rtg,
                               "device": device}))
        return agent

    import offline.rtg_calibration as rtg_calibration

    monkeypatch.setattr(rtg_calibration, "agent_with_target", fake_agent_with_target, raising=True)
    monkeypatch.setattr(cs, "_code_provenance", lambda: {"code_commit": "e" * 40, "code_dirty": False}, raising=True)
    rewards = [0.0, -7.0, -11.0, -4.0]
    runner_kwargs: list[dict[str, Any]] = []

    def fake_runner(**kwargs: Any) -> AdmissionEpisode:
        runner_kwargs.append(kwargs)
        choose = kwargs["choose_action_factory"](StubEnv())
        for t, reward in enumerate(rewards):
            choose(None, info_at(t, [0.1] * STATE_DIM, [True] * N_ACTIONS, reward))
        engine, ours = fixture_att(stage, subject, k, batch, seed, draw)
        return AdmissionEpisode(
            scenario=kwargs["scenario"], tier=kwargs["tier"], method=kwargs["method"], arm=kwargs["arm"],
            seed=kwargs["seed"], draw_id=kwargs["draw_id"], created=300, entered=290, never_entered=10,
            entered_fraction=290 / 300, completed_at_horizon=250, running_at_horizon=40, waiting_at_horizon=10,
            att_ours=ours, att_engine=engine, horizon_vehicle_count=40.0, episode_reward=-22.0, seconds=2.9,
            seconds_rollout=2.5,
        )

    chunk = cs.run_campaign_cell(
        cell, output_root=tree.output_root, corpus_root=tmp_path / "corpus", draws_root=draws_root,
        data_dir=tree.data_dir, canary_seconds=0.75, episode_runner=fake_runner,
    )

    config = Path(draw_config_path("cityflow1x1", draw, out_root=draws_root))
    [kwargs] = runner_kwargs
    assert kwargs["scenario"] == "hz1x1" and kwargs["tier"] == subject and kwargs["method"] == "dt"
    assert (kwargs["arm"], kwargs["seed"], kwargs["draw_id"]) == (cell.arm, seed, draw)
    assert Path(kwargs["config_path"]) == config
    assert kwargs["env_settings"] == settings and seen_settings == [(subject, tmp_path / "corpus")]
    assert kwargs["scenario_id"] == "cityflow1x1" and kwargs["engine_seed"] == 1000
    assert kwargs["created"] == created_from_flow(config.parent / "flow.json", horizon_seconds=3600)
    [(agent, loaded)] = agents
    identity = cs.checkpoint_for_cell(cell, output_root=tree.output_root, data_dir=tree.data_dir)
    assert loaded == {"path": tree.output_root / identity["path"], "declared": 40_000, "target": PROMPT[subject][0],
                      "device": "cuda"}
    assert [(call["explore"], call["update_memory"]) for call in agent.calls] == [(False, True)] * len(rewards)
    target = PROMPT[subject][0]
    assert chunk["series"] == {
        "rtg": [target, target, target + 7.0, target + 18.0],
        "reward": rewards,
        "action": [(3 * t + 1) % N_ACTIONS for t in range(len(rewards))],
    }
    assert chunk["checkpoint"] == identity and chunk["demand"] == demand_of(draws_root, draw)
    assert chunk["target_rtg"] == target and chunk["device"] == "cuda" and chunk["canary_seconds"] == 0.75
    assert (chunk["code_commit"], chunk["code_dirty"]) == ("e" * 40, False)
    assert chunk["torch_num_threads"] == torch.get_num_threads()
    assert chunk["episode"]["att_engine"] == fixture_att(stage, subject, k, batch, seed, draw)[0]
    assert not (tree.output_root / "p5_3c").exists(), "a cell writes nothing"


def test_the_module_assembles_exactly_the_fixtures_statement_of_the_chunk_format(tmp_path: Path) -> None:
    draws_root = build_draws(tmp_path)
    for row in (("reference", "mix50", 20, 64, 202, 1003), ("sweep", "mappo1000", 10, 64, 404, 1077)):
        stage, subject, k, batch, seed, draw = row
        checkpoint = {"path": f"p5_3c_training/checkpoints/x{seed}.pt", "sha256": "b" * 64, "checked_against": ["x"]}
        expected = fixture_chunk(*row, checkpoint=checkpoint, demand=demand_of(draws_root, draw), commit="c" * 40)
        target = PROMPT[subject][0]
        assembled = cs.cell_payload(
            _cell(*row), episode=episode_record(*row), series=expected["series"], checkpoint=checkpoint,
            demand=demand_of(draws_root, draw), target_rtg=target, device="cuda",
            code={"code_commit": "c" * 40, "code_dirty": False}, canary_seconds=0.75, torch_num_threads=1,
        )
        assert assembled == expected


#: Each broken field, and the FIELD the refusal must name.  A chunk's name contains "seed" and "draw", so a match on a
#: bare word would be satisfied by any refusal; the refusal names the field as ``field '<name>'``.
_BREAKAGES: dict[str, tuple[Any, str]] = {
    "format": (lambda c: c.update(format_version="p5.3c-cell/0.9"), "field 'format_version'"),
    "name": (lambda c: c.update(name="cell_other.json"), "field 'name'"),
    "arm": (lambda c: c.update(arm="mappo1000_k10_b64"), "field 'arm'"),
    "seed": (lambda c: c.update(seed=202), "field 'seed'"),
    "draw": (lambda c: c.update(draw_id=1001), "field 'draw_id'"),
    "run": (lambda c: c.update(run="mappo1000_k10_b64_seed101"), "field 'run'"),
    "episode-key": (lambda c: c["episode"].pop("att_ours"), r"field 'episode\.att_ours'"),
    "episode-arm": (lambda c: c["episode"].update(arm="dt@mappo1000"), r"field 'episode\.arm'"),
    "att-nan": (lambda c: c["episode"].update(att_engine=float("nan")), r"field 'episode\.att_engine'"),
    "att-inf": (lambda c: c["episode"].update(att_ours=float("inf")), r"field 'episode\.att_ours'"),
    "series-length": (lambda c: c["series"]["action"].pop(), "field 'series'"),
    "n-decisions": (lambda c: c.update(n_decisions=4), "field 'n_decisions'"),
    "absolute-checkpoint": (lambda c: c["checkpoint"].update(path="/home/x.pt"), r"field 'checkpoint\.path'"),
    "fenced-checkpoint": (lambda c: c["checkpoint"].update(path="p5_3c_training/fenced_timing/s/alone.pt"),
                          r"field 'checkpoint\.path'.*fenced_timing"),
    "checkpoint-digest": (lambda c: c["checkpoint"].update(sha256="xyz"), r"field 'checkpoint\.sha256'"),
    "demand-digest": (lambda c: c["demand"].update(flow_sha256="0"), r"field 'demand\.flow_sha256'"),
    "device": (lambda c: c.update(device="cpu"), "field 'device'"),
    "engine-seed": (lambda c: c.update(engine_seed=1001), "field 'engine_seed'"),
    "scenario-id": (lambda c: c.update(scenario_id="cityflow_grid4x4"), "field 'scenario_id'"),
    "dirty": (lambda c: c.update(code_dirty=True), "field 'code_dirty'"),
    "commit": (lambda c: c.update(code_commit="abc"), "field 'code_commit'"),
    "prompt": (lambda c: c.update(target_rtg=-5959.0), "field 'target_rtg'"),
    "threads": (lambda c: c.update(torch_num_threads=0), "field 'torch_num_threads'"),
}


@pytest.mark.parametrize("breakage", sorted(_BREAKAGES))
def test_validate_chunk_refuses_each_broken_field_by_name(tmp_path: Path, breakage: str) -> None:
    draws_root = build_draws(tmp_path)
    row = ("sweep", "mappo1000", 5, 64, 101, 1000)
    checkpoint = {"path": "p5_3c_training/checkpoints/mappo1000_k5_b64_seed101.pt", "sha256": "b" * 64,
                  "checked_against": ["x"]}
    chunk = fixture_chunk(*row, checkpoint=checkpoint, demand=demand_of(draws_root, 1000), commit="c" * 40)
    cs.validate_chunk(chunk, _cell(*row))
    mutate, field = _BREAKAGES[breakage]
    mutate(chunk)
    with pytest.raises(ValueError, match=field):
        cs.validate_chunk(chunk, _cell(*row))


# ----------------------------------------------------------------------------------------------------------------------
# The caller-level twin of T-k1 / T-k2 (plan F6): through the campaign's own decision factory
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2])
def test_the_campaigns_decision_factory_attends_to_exactly_k_steps_and_records_the_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, k: int
) -> None:
    """*Mutation:* the rollout window at K + 1 -> the shape assertion dies at the first decision."""
    from agent.DTAgent import DecisionTransformer
    from offline.dt_gate import build_training_dataset, stack_dataset, train_dt

    corpus = write_single_ix_corpus(tmp_path, "fixture__policy", draws=(1, 2, 3))
    dataset = build_training_dataset([corpus], k)
    stacked = stack_dataset(dataset)
    first = sorted(corpus.glob("*.npz"))[0]
    with np.load(first) as raw:
        states = np.asarray(raw["ix0_state"], dtype=np.float32)
        avail = np.asarray(raw["ix0_avail_mask"], dtype=np.bool_)
        rewards = np.asarray(raw["ix0_local_reward"], dtype=np.float32)
    target = float(rewards.astype(np.float64).sum())
    trained = tmp_path / f"k{k}.pt"
    train_dt(
        stacked, state_dim=STATE_DIM, n_actions=N_ACTIONS, seed=101, declared_gradient_steps=6, raise_to=None,
        context_length=k, batch_size=8, device=torch.device("cpu"), checkpoint_path=trained, stats=dataset.stats,
        scenario_id="fixture_1ix", target_rtg=target, rtg_scale=64.0, provenance={"tier": "fixture"},
    )
    payload = torch.load(trained, map_location="cpu", weights_only=False)
    payload["provenance"]["gradient_steps"] = 40_000
    registered = tmp_path / f"k{k}_40000.pt"
    torch.save(payload, registered)

    captured: list[dict[str, torch.Tensor]] = []
    original = DecisionTransformer.forward

    def spy(self, rtg, state, action, timestep, attention_mask=None, avail_mask=None):  # type: ignore[no-untyped-def]
        captured.append({"state": state.detach().clone(), "timestep": timestep.detach().clone(),
                         "attention_mask": attention_mask.detach().clone()})
        return original(self, rtg, state, action, timestep, attention_mask, avail_mask)

    monkeypatch.setattr(DecisionTransformer, "forward", spy, raising=True)
    series: dict[str, list[Any]] = {"rtg": [], "reward": [], "action": []}
    choose = cs.cell_decision_factory(registered, target_rtg=target, device="cpu", series=series)(StubEnv())
    for t in range(T_DECISIONS):
        reward = 0.0 if t == 0 else float(rewards[t - 1])
        choose(None, info_at(t, states[t], avail[t], reward))

    assert len(captured) == T_DECISIONS
    for t, seen in enumerate(captured):
        assert tuple(seen["state"].shape) == (1, k, STATE_DIM), t
        assert int(seen["attention_mask"][0].sum()) == min(t + 1, k), t
        assert int(seen["timestep"][0, -1]) == t, t
    assert len(series["rtg"]) == len(series["reward"]) == len(series["action"]) == T_DECISIONS
    assert series["rtg"][0] == target
    for t in range(1, T_DECISIONS):
        assert series["rtg"][t] == series["rtg"][t - 1] - series["reward"][t - 1], t
    assert all(bool(avail[t][series["action"][t]]) for t in range(T_DECISIONS))


# ======================================================================================================================
# Chunks: the atomic write, reuse by content, moved aside
# ======================================================================================================================


def test_a_chunk_is_written_atomically_into_an_existing_cells_directory_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = build_checkpoint_tree(tmp_path)
    draws_root = build_draws(tmp_path)
    row = ("reference", "mappo1000", 20, 64, 101, 1000)
    chunk = fixture_chunk(*row, checkpoint={"path": "p4_dt/dt_seed101.pt", "sha256": "b" * 64, "checked_against": []},
                          demand=demand_of(draws_root, 1000), commit="c" * 40)
    with pytest.raises(ValueError, match="does not exist"):
        cs.write_chunk(chunk, output_root=tree.output_root)
    assert not (tree.output_root / "p5_3c").exists()
    cs.cells_dir(tree.output_root).mkdir(parents=True)
    path = cs.write_chunk(chunk, output_root=tree.output_root)
    assert path == cs.cells_dir(tree.output_root) / chunk["name"]
    assert _read(path) == chunk
    assert sorted(p.name for p in path.parent.iterdir()) == [chunk["name"]], "no temporary file is left behind"
    chunk["seed"] = 202
    with pytest.raises(ValueError, match="field 'seed'"):
        cs.write_chunk(chunk, output_root=tree.output_root)


def test_a_chunk_is_reusable_only_on_evidence_rederived_from_disk(campaign: CampaignTree) -> None:
    cell = _cell("sweep", "mappo1000", 5, 64, 101, 1000)
    path = cs.chunk_path(campaign.output_root, cell)
    kwargs = {"cell": cell, "output_root": campaign.output_root, "draws_root": campaign.draws_root,
              "data_dir": campaign.data_dir}
    chunk = _read(path)
    assert cs.chunk_is_reusable(chunk, **kwargs) is True
    assert cs.chunk_is_reusable({**chunk, "code_commit": C1A_COMMIT}, **kwargs) is False
    assert cs.chunk_is_reusable({**chunk, "code_dirty": True}, **kwargs) is False
    assert cs.chunk_is_reusable({**chunk, "checkpoint": {**chunk["checkpoint"], "sha256": "d" * 64}}, **kwargs) is False
    assert cs.chunk_is_reusable({**chunk, "demand": {**chunk["demand"], "flow_sha256": "d" * 64}}, **kwargs) is False
    assert cs.chunk_is_reusable(["not", "a", "chunk"], **kwargs) is False
    assert cs.chunk_is_reusable({**chunk, "seed": 202}, **kwargs) is False
    flow = campaign.draws_root / "cityflow1x1" / "draw_1000" / "flow.json"
    flow.write_text(flow.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert cs.chunk_is_reusable(chunk, **kwargs) is False
    with pytest.raises(RuntimeError, match="git"):
        cs.chunk_is_reusable({**chunk, "code_commit": "0" * 40}, **kwargs)


def test_move_aside_never_overwrites(tmp_path: Path) -> None:
    cells = tmp_path / "output" / "p5_3c" / "cells"
    cells.mkdir(parents=True)
    first = cells / "cell_x.json"
    first.write_text("one", encoding="utf-8")
    moved = cs.move_aside(first)
    assert moved == cells / "failed" / "cell_x.json" and moved.read_text(encoding="utf-8") == "one"
    first.write_text("two", encoding="utf-8")
    again = cs.move_aside(first)
    assert again == cells / "failed" / "cell_x.1.json" and again.read_text(encoding="utf-8") == "two"
    assert moved.read_text(encoding="utf-8") == "one"


# ======================================================================================================================
# The stage runner (a spawn pool, a fake worker)
# ======================================================================================================================


def _fake_stage_worker(task: tuple[Any, dict[str, Any]]) -> dict[str, Any]:
    """A worker that writes the fixture's chunk for its cell -- top-level, so ``spawn`` can pickle it."""
    import offline.context_sweep as sweep
    from tests.p5_3c_campaign_fixtures import demand_of as demand
    from tests.p5_3c_campaign_fixtures import fixture_chunk as chunk_of
    from tests.p5_3c_campaign_fixtures import head_commit as head

    cell, kwargs = task
    identity = sweep.checkpoint_for_cell(cell, output_root=kwargs["output_root"], data_dir=kwargs["data_dir"])
    chunk = chunk_of(cell.stage, cell.subject, cell.k, cell.batch, cell.seed, cell.draw_id, checkpoint=identity,
                     demand=demand(Path(kwargs["draws_root"]), cell.draw_id), commit=head())
    path = sweep.write_chunk(chunk, output_root=kwargs["output_root"])
    return {"name": path.name, "ok": True, "seconds": 0.0, "error": None}


def test_a_stage_rolls_only_the_cells_not_reusable_and_moves_a_bad_chunk_aside(campaign: CampaignTree) -> None:
    cells = cs.cells_dir(campaign.output_root)
    missing = [_cell("reference", "mix50", 20, 64, 505, draw) for draw in (1097, 1098, 1099)]
    for cell in missing:
        cs.chunk_path(campaign.output_root, cell).unlink()
    broken = cs.chunk_path(campaign.output_root, _cell("reference", "mix50", 20, 64, 404, 1050))
    broken.write_text("{ half a chunk", encoding="utf-8")
    result = cs.run_campaign_stage(
        stage="reference", output_root=campaign.output_root, corpus_root=campaign.corpus_root,
        draws_root=campaign.draws_root, data_dir=campaign.data_dir, canary_seconds=0.75, workers=2,
        worker=_fake_stage_worker,
    )
    assert (result["n_declared"], result["n_reused"], result["n_rolled"], result["n_failed"]) == (1000, 996, 4, 0)
    assert (cells / "failed" / broken.name).read_text(encoding="utf-8") == "{ half a chunk"
    for cell in [*missing, _cell("reference", "mix50", 20, 64, 404, 1050)]:
        cs.validate_chunk(_read(cs.chunk_path(campaign.output_root, cell)), cell)


#: The one cell :func:`_fake_stage_worker_one_fails` fails, as the production worker reports a failure.
FAILING_DRAW = 1098
FAILURE = {"ok": False, "seconds": None, "error": "RuntimeError: the fixture's failing cell"}


def _fake_stage_worker_one_fails(task: tuple[Any, dict[str, Any]]) -> dict[str, Any]:
    """The fixture's worker, except that the cell of draw 1098 FAILS -- top-level, so ``spawn`` can pickle it."""
    cell, _kwargs = task
    if cell.draw_id == FAILING_DRAW:
        return {"name": cell.name, **FAILURE}
    return _fake_stage_worker(task)


def test_a_failed_cell_is_counted_and_named_and_the_cells_command_exits_1(
    campaign: CampaignTree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Amendment D, D4.2(a): the REAL stage and the REAL ``cells`` command, a worker failing one cell.

    *Mutation (reviewer D's M1):* ``n_failed`` zeroed in ``run_campaign_stage`` -> this dies.
    """
    missing = [_cell("reference", "mix50", 20, 64, 505, draw) for draw in (1097, FAILING_DRAW, 1099)]
    for cell in missing:
        cs.chunk_path(campaign.output_root, cell).unlink()
    kwargs = {"output_root": campaign.output_root, "corpus_root": campaign.corpus_root,
              "draws_root": campaign.draws_root, "data_dir": campaign.data_dir}
    result = cs.run_campaign_stage(stage="reference", canary_seconds=0.75, workers=2,
                                   worker=_fake_stage_worker_one_fails, **kwargs)
    assert (result["n_declared"], result["n_reused"], result["n_rolled"], result["n_failed"]) == (1000, 997, 2, 1)
    assert result["failures"] == [{"name": missing[1].name, **FAILURE}]
    assert not cs.chunk_path(campaign.output_root, missing[1]).exists()
    for cell in (missing[0], missing[2]):
        cs.validate_chunk(_read(cs.chunk_path(campaign.output_root, cell)), cell)
    capsys.readouterr()

    # The command takes the production worker by its module name at call time; ``spawn`` pickles the fake by its own.
    monkeypatch.setattr(cs, "_campaign_worker", _fake_stage_worker_one_fails, raising=True)
    argv = ["cells", *[item for name, root in kwargs.items() for item in (f"--{name.replace('_', '-')}", str(root))],
            "--stage", "reference", "--canary-seconds", "0.75", "--workers", "2"]
    assert cs.main(argv) == 1
    out = capsys.readouterr().out
    assert f"{missing[1].name} FAILED {FAILURE['error']}" in out
    assert "cells reference: 1000 declared, 999 reused, 0 rolled, 1 failed" in out


def test_the_production_worker_returns_a_failure_and_writes_a_good_chunk(
    campaign: CampaignTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amendment D, D4.2(b): ``_campaign_worker`` itself -- ``run_campaign_cell`` raising gives ``ok False`` with the
    error's type and text and NO file in ``cells/``; returning a payload gives the chunk ``write_chunk`` wrote.

    *Mutation (reviewer D's M7):* ``ok True`` on the exception -> this dies.
    """
    cell = _cell("reference", "mix50", 20, 64, 505, 1099)
    path = cs.chunk_path(campaign.output_root, cell)
    path.unlink()
    cells = cs.cells_dir(campaign.output_root)
    before = sorted(p.name for p in cells.iterdir())
    kwargs = {"output_root": str(campaign.output_root), "corpus_root": str(campaign.corpus_root),
              "draws_root": str(campaign.draws_root), "data_dir": str(campaign.data_dir), "canary_seconds": 0.75}

    def raising(cell_: cs.CampaignCell, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("the episode stopped")

    monkeypatch.setattr(cs, "run_campaign_cell", raising, raising=True)
    assert cs._campaign_worker((cell, kwargs)) == {
        "name": cell.name, "ok": False, "seconds": None, "error": "RuntimeError: the episode stopped",
    }
    assert sorted(p.name for p in cells.iterdir()) == before

    identity = cs.checkpoint_for_cell(cell, output_root=campaign.output_root, data_dir=campaign.data_dir)
    payload = fixture_chunk("reference", "mix50", 20, 64, 505, 1099, checkpoint=identity,
                            demand=demand_of(campaign.draws_root, 1099), commit=head_commit())
    monkeypatch.setattr(cs, "run_campaign_cell", lambda cell_, **_kwargs: payload, raising=True)
    assert cs._campaign_worker((cell, kwargs)) == {
        "name": cell.name, "ok": True, "seconds": payload["episode"]["seconds"], "error": None,
    }
    assert _read(path) == payload
    assert sorted(p.name for p in cells.iterdir()) == sorted([*before, cell.name])


def test_the_stage_pool_has_as_many_processes_as_workers(campaign: CampaignTree, monkeypatch: pytest.MonkeyPatch) -> None:
    """Amendment D, D4.2(c): the stage's ``spawn`` pool is created with ``processes == workers`` and the one-thread
    initializer -- through a recording context that runs the tasks in this process.

    *Mutation (reviewer D's M3):* the pool at one process whatever ``--workers`` says -> this dies.
    """
    import multiprocessing

    created: list[tuple[str, int, Any]] = []

    class _InlineResults:
        """``IMapIterator``'s one method the stage reads results with (C3.2, Amendment D.2: ``next(timeout=...)``)."""

        def __init__(self, results: Any) -> None:
            self._results = results

        def next(self, timeout: float | None = None) -> Any:
            return next(self._results)

    class _InlinePool:
        def __enter__(self) -> _InlinePool:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def imap_unordered(self, function: Any, tasks: Any) -> _InlineResults:
            return _InlineResults(map(function, tasks))

    class _Context:
        def __init__(self, method: str) -> None:
            self.method = method

        def Pool(self, processes: int, initializer: Any) -> _InlinePool:  # noqa: N802 - multiprocessing's name
            created.append((self.method, processes, initializer))
            return _InlinePool()

    monkeypatch.setattr(multiprocessing, "get_context", _Context, raising=True)
    cs.chunk_path(campaign.output_root, _cell("reference", "mix50", 20, 64, 303, 1010)).unlink()
    result = cs.run_campaign_stage(
        stage="reference", output_root=campaign.output_root, corpus_root=campaign.corpus_root,
        draws_root=campaign.draws_root, data_dir=campaign.data_dir, canary_seconds=0.75, workers=3,
        worker=_fake_stage_worker,
    )
    assert created == [("spawn", 3, cs._pin_one_thread)]
    assert (result["n_rolled"], result["n_failed"]) == (1, 0)


# ----------------------------------------------------------------------------------------------------------------------
# A hung cell (Amendment D.2, D.2.2: CityFlow's Engine::~Engine() race, DEFERRED 104) -- the stage's own handling
# ----------------------------------------------------------------------------------------------------------------------

#: The hang tests' result timeout (the module's is 180 s): five times a fresh two-worker spawn pool's first result here
#: (0.84-0.93 s, its workers importing torch and this module; measured 2026-09-30), far below the fake hang's 30 s.
HANG_TEST_TIMEOUT_S = 5.0
#: A fake hang blocks on an Event nobody sets for this long -- not forever, so a mutant without the timeout ENDS (then
#: behaves as a cell that finished late) and its assertions on the hang fail (D.2.2(3)).
HANG_SECONDS = 30.0
HANGING_DRAW = 1099


def _hang() -> None:
    import threading

    threading.Event().wait(HANG_SECONDS)


def _hang_once(task: tuple[Any, dict[str, Any]]) -> bool:
    """True the FIRST time the draw-1099 cell is rolled (a marker beside the output tree remembers across pools)."""
    cell, kwargs = task
    if cell.draw_id != HANGING_DRAW:
        return False
    marker = Path(kwargs["output_root"]).parent / "hung_once" / cell.name
    if marker.exists():
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("hung\n", encoding="utf-8")
    return True


def _fake_stage_worker_hangs_once(task: tuple[Any, dict[str, Any]]) -> dict[str, Any]:
    """The fixture's worker; the draw-1099 cell hangs the first time it is rolled and succeeds when re-rolled."""
    if _hang_once(task):
        _hang()
    return _fake_stage_worker(task)


def _fake_stage_worker_always_hangs(task: tuple[Any, dict[str, Any]]) -> dict[str, Any]:
    """The fixture's worker; the draw-1099 cell hangs every time it is rolled."""
    if task[0].draw_id == HANGING_DRAW:
        _hang()
    return _fake_stage_worker(task)


def _fake_stage_worker_killed_mid_write(task: tuple[Any, dict[str, Any]]) -> dict[str, Any]:
    """The fixture's worker; the first time the draw-1099 cell is rolled it leaves the ``.tmp`` of an unfinished chunk
    write (``write_chunk``'s own name for it) and hangs there, so the terminated pool kills it mid-write."""
    if _hang_once(task):
        import offline.context_sweep as sweep

        cell, kwargs = task
        destination = sweep.chunk_path(kwargs["output_root"], cell)
        destination.with_name(f".{destination.name}.{os.getpid()}.tmp").write_text("{ half a chunk", encoding="utf-8")
        _hang()
    return _fake_stage_worker(task)


def _record_stage(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The REAL stage, its return value kept: ``cells`` calls ``run_campaign_stage`` by its module name."""
    recorded: list[dict[str, Any]] = []
    real = cs.run_campaign_stage

    def recording(**kwargs: Any) -> dict[str, Any]:
        recorded.append(real(**kwargs))
        return recorded[-1]

    monkeypatch.setattr(cs, "run_campaign_stage", recording, raising=True)
    return recorded


def _cells_argv(campaign: CampaignTree) -> list[str]:
    return ["cells", "--output-root", str(campaign.output_root), "--corpus-root", str(campaign.corpus_root),
            "--data-dir", str(campaign.data_dir), "--draws-root", str(campaign.draws_root), "--stage", "reference",
            "--canary-seconds", "0.75", "--workers", "2"]


def test_a_hung_cell_is_re_rolled_in_a_fresh_pool_and_the_cells_command_exits_0(
    campaign: CampaignTree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Amendment D.2, D.2.2(3)(a): one cell never returns in the first pool; the stage times out, terminates the pool,
    re-rolls the cell in a fresh one and writes its chunk; every other chunk untouched; ``cells`` exits 0.

    *Mutations:* the timeout removed (the fake's 30 s then ends the run with no hang seen); the re-roll pool not
    re-created (the cell stays hung) -> this dies.
    """
    monkeypatch.setattr(cs, "STAGE_RESULT_TIMEOUT_S", HANG_TEST_TIMEOUT_S, raising=True)
    cells = cs.cells_dir(campaign.output_root)
    rolled = [_cell("reference", "mix50", 20, 64, 505, draw) for draw in (1097, 1098, HANGING_DRAW)]
    for cell in rolled:
        cs.chunk_path(campaign.output_root, cell).unlink()
    before = {path.name: path.read_bytes() for path in cells.glob("cell_*.json")}
    recorded = _record_stage(monkeypatch)
    monkeypatch.setattr(cs, "_campaign_worker", _fake_stage_worker_hangs_once, raising=True)
    assert cs.main(_cells_argv(campaign)) == 0
    [result] = recorded
    assert (result["n_declared"], result["n_reused"], result["n_rolled"], result["n_failed"]) == (1000, 997, 3, 0)
    assert (result["n_hung"], result["hang_rounds"]) == (1, 1)
    assert result["wall_seconds"] < HANG_SECONDS, "the hung worker was waited out, not terminated"
    out = capsys.readouterr().out
    assert out.count(" HUNG (round ") == 1 and f"  {rolled[2].name} HUNG (round 1): re-rolled\n" in out
    for cell in rolled:
        cs.validate_chunk(_read(cs.chunk_path(campaign.output_root, cell)), cell)
    after = {path.name: path.read_bytes() for path in cells.glob("cell_*.json")}
    assert {name: data for name, data in after.items() if name not in {cell.name for cell in rolled}} == before
    assert not (cells / "failed").exists()


def test_a_cell_hung_in_every_round_is_a_failure_and_the_cells_command_exits_1(
    campaign: CampaignTree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Amendment D.2, D.2.2(3)(b): a cell that hangs in all ``STAGE_HANG_ROUNDS`` rounds is a failure, ``hung 3
    times``, with no chunk; ``cells`` exits 1.

    *Mutation:* a hung cell counted as ok -> this dies.
    """
    monkeypatch.setattr(cs, "STAGE_RESULT_TIMEOUT_S", HANG_TEST_TIMEOUT_S, raising=True)
    cell = _cell("reference", "mix50", 20, 64, 505, HANGING_DRAW)
    cs.chunk_path(campaign.output_root, cell).unlink()
    recorded = _record_stage(monkeypatch)
    monkeypatch.setattr(cs, "_campaign_worker", _fake_stage_worker_always_hangs, raising=True)
    assert cs.main(_cells_argv(campaign)) == 1
    [result] = recorded
    assert cs.STAGE_HANG_ROUNDS == 3
    assert (result["n_rolled"], result["n_failed"], result["n_hung"], result["hang_rounds"]) == (0, 1, 1, 3)
    assert result["failures"] == [{"name": cell.name, "ok": False, "seconds": None, "error": "hung 3 times"}]
    out = capsys.readouterr().out
    assert [line.strip() for line in out.splitlines() if cell.name in line] == [
        f"{cell.name} HUNG (round 1): re-rolled", f"{cell.name} HUNG (round 2): re-rolled",
        f"{cell.name} FAILED hung 3 times",
    ]
    assert "cells reference: 1000 declared, 999 reused, 0 rolled, 1 failed" in out
    assert not cs.chunk_path(campaign.output_root, cell).exists()


def test_a_killed_workers_tmp_is_moved_to_failed_and_not_left_in_cells(
    campaign: CampaignTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amendment D.2, D.2.2(3)(c): the ``.tmp`` a worker killed mid-write leaves is moved to ``cells/failed/`` by
    ``move_aside`` -- a partial write, never a chunk -- and the cell is re-rolled whole.

    *Mutation:* the ``.tmp`` not moved -> this dies.
    """
    monkeypatch.setattr(cs, "STAGE_RESULT_TIMEOUT_S", HANG_TEST_TIMEOUT_S, raising=True)
    cells = cs.cells_dir(campaign.output_root)
    cell = _cell("reference", "mix50", 20, 64, 505, HANGING_DRAW)
    cs.chunk_path(campaign.output_root, cell).unlink()
    result = cs.run_campaign_stage(
        stage="reference", output_root=campaign.output_root, corpus_root=campaign.corpus_root,
        draws_root=campaign.draws_root, data_dir=campaign.data_dir, canary_seconds=0.75, workers=2,
        worker=_fake_stage_worker_killed_mid_write,
    )
    assert (result["n_rolled"], result["n_failed"], result["n_hung"], result["hang_rounds"]) == (1, 0, 1, 1)
    assert sorted(path.name for path in cells.iterdir() if path.name.endswith(".tmp")) == []
    [moved] = sorted((cells / "failed").iterdir())
    assert moved.name.startswith(f".{cell.name}.") and moved.name.endswith(".tmp")
    assert moved.read_text(encoding="utf-8") == "{ half a chunk"
    cs.validate_chunk(_read(cs.chunk_path(campaign.output_root, cell)), cell)


def test_an_input_that_cannot_be_verified_refuses_the_stage_and_moves_no_chunk_aside(campaign: CampaignTree) -> None:
    """A checkpoint at another digest is the INPUT's fault: the stage refuses before touching a chunk, and none of the
    500 chunks rolled against that checkpoint is moved aside as if it were bad.

    *Mutation:* an unverifiable identity read as "not reusable" -> the chunks move to ``failed/`` and this dies.
    """
    cells = cs.cells_dir(campaign.output_root)
    before = {path.name: path.read_bytes() for path in cells.iterdir() if path.is_file()}
    tampered = campaign.output_root / "p4_7" / "checkpoints" / "mix50_dt_seed202.pt"
    tampered.write_bytes(tampered.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="mix50_dt_seed202"):
        cs.run_campaign_stage(
            stage="reference", output_root=campaign.output_root, corpus_root=campaign.corpus_root,
            draws_root=campaign.draws_root, data_dir=campaign.data_dir, canary_seconds=0.75, workers=1,
            worker=_fake_stage_worker,
        )
    assert not (cells / "failed").exists()
    assert {path.name: path.read_bytes() for path in cells.iterdir() if path.is_file()} == before


def test_the_sweep_stage_refuses_without_a_valid_passing_gate_record_and_rolls_nothing(campaign: CampaignTree) -> None:
    """The second line of A26(c)'s stop: no sweep cell is rolled unless the gate's record exists and still holds."""
    cells = cs.cells_dir(campaign.output_root)
    victim = cs.chunk_path(campaign.output_root, _cell("sweep", "mix50", 1, 64, 101, 1000))
    victim.unlink()
    cs.gate_record_path(campaign.output_root).unlink()
    before = sorted(p.name for p in cells.iterdir())
    # The gate refusal's own words: a looser "gate" was also satisfied by a spawned worker's "... after gate G3"
    # (mutant G2 survived it).
    with pytest.raises(ValueError, match="no sweep cell is rolled"):
        cs.run_campaign_stage(
            stage="sweep", output_root=campaign.output_root, corpus_root=campaign.corpus_root,
            draws_root=campaign.draws_root, data_dir=campaign.data_dir, canary_seconds=0.75, workers=1,
            worker=_fake_stage_worker,
        )
    assert sorted(p.name for p in cells.iterdir()) == before and not victim.exists()


# ======================================================================================================================
# T-gate (load-bearing)
# ======================================================================================================================


def test_the_gate_passes_when_all_500_rows_are_equal_and_its_record_is_written_once(campaign: CampaignTree) -> None:
    verdict = cs.reference_gate(output_root=campaign.output_root, data_dir=campaign.data_dir)
    assert verdict["passed"] is True and verdict["n_compared"] == 500
    assert verdict["differing"] == [] and verdict["compared"] == ["att_engine", "att_ours"]
    path = cs.gate_record_path(campaign.output_root)
    before = path.read_bytes()
    assert cs.write_gate_record(campaign.output_root, verdict) == path
    assert path.read_bytes() == before
    held = cs.verify_gate_record(output_root=campaign.output_root, data_dir=campaign.data_dir)
    assert held["passed"] is True and held["n_compared"] == 500


@pytest.mark.parametrize("definition", ["att_engine", "att_ours"])
def test_one_row_one_ulp_off_fails_the_gate_writes_nothing_and_no_sweep_cell_is_rolled(
    campaign: CampaignTree, capsys: pytest.CaptureFixture[str], definition: str
) -> None:
    """*Mutations:* the gate skipped; the gate on ``att_engine`` alone (the ``att_ours`` case) -> this dies."""
    cs.gate_record_path(campaign.output_root).unlink()
    cell = _cell("reference", "mappo1000", 20, 64, 303, 1057)
    path = cs.chunk_path(campaign.output_root, cell)
    chunk = _read(path)
    value = chunk["episode"][definition]
    chunk["episode"][definition] = math.nextafter(value, math.inf)
    _rewrite(path, chunk)

    verdict = cs.reference_gate(output_root=campaign.output_root, data_dir=campaign.data_dir)
    assert verdict["passed"] is False
    assert verdict["differing"] == [{"cell": cell.name, "fields": [definition]}]
    with pytest.raises(ValueError, match="passing"):
        cs.write_gate_record(campaign.output_root, verdict)
    assert not cs.gate_record_path(campaign.output_root).exists()

    code = cs.main(["reference-gate", "--output-root", str(campaign.output_root), "--data-dir", str(campaign.data_dir)])
    out = capsys.readouterr().out
    assert code != 0 and cell.name in out and definition in out
    assert repr(value) not in out and repr(chunk["episode"][definition]) not in out, "names, never values"
    assert not cs.gate_record_path(campaign.output_root).exists()

    sweep_chunk = cs.chunk_path(campaign.output_root, _cell("sweep", "mappo1000", 20, 64, 101, 1000))
    sweep_chunk.unlink()
    with pytest.raises(ValueError, match="no sweep cell is rolled"):
        cs.run_campaign_stage(
            stage="sweep", output_root=campaign.output_root, corpus_root=campaign.corpus_root,
            draws_root=campaign.draws_root, data_dir=campaign.data_dir, canary_seconds=0.75, workers=1,
            worker=_fake_stage_worker,
        )
    assert not sweep_chunk.exists()


def test_a_missing_reference_chunk_refuses_the_gate(campaign: CampaignTree) -> None:
    cell = _cell("reference", "mappo1000", 20, 64, 505, 1000)
    cs.chunk_path(campaign.output_root, cell).unlink()
    with pytest.raises(ValueError, match=cell.name):
        cs.reference_gate(output_root=campaign.output_root, data_dir=campaign.data_dir)


def test_a_reference_chunk_changed_after_the_gate_fails_its_record_and_mix50_never_gates(campaign: CampaignTree) -> None:
    mix = cs.chunk_path(campaign.output_root, _cell("reference", "mix50", 20, 64, 101, 1000))
    chunk = _read(mix)
    chunk["episode"]["att_engine"] += 1.0
    _rewrite(mix, chunk)
    assert cs.reference_gate(output_root=campaign.output_root, data_dir=campaign.data_dir)["passed"] is True
    assert cs.verify_gate_record(output_root=campaign.output_root, data_dir=campaign.data_dir)["passed"] is True

    gated = cs.chunk_path(campaign.output_root, _cell("reference", "mappo1000", 20, 64, 202, 1010))
    chunk = _read(gated)
    chunk["episode"]["seconds"] = 3.1
    _rewrite(gated, chunk)
    with pytest.raises(ValueError, match="reference_gate"):
        cs.verify_gate_record(output_root=campaign.output_root, data_dir=campaign.data_dir)


# ======================================================================================================================
# The fenced re-roll (Amendment A, Q2 with A.1)
# ======================================================================================================================


def _reroll_worker_matching(task: tuple[Any, dict[str, Any]]) -> dict[str, Any]:
    import offline.context_sweep as sweep
    from tests.p5_3c_campaign_fixtures import demand_of as demand
    from tests.p5_3c_campaign_fixtures import fixture_chunk as chunk_of

    cell, kwargs = task
    identity = sweep.checkpoint_for_cell(cell, output_root=kwargs["output_root"], data_dir=kwargs["data_dir"])
    payload = chunk_of(cell.stage, cell.subject, cell.k, cell.batch, cell.seed, cell.draw_id, checkpoint=identity,
                       demand=demand(Path(kwargs["draws_root"]), cell.draw_id), commit="c" * 40)
    return {"name": cell.name, "ok": True, "payload": payload, "error": None}


def _reroll_worker_one_differs(task: tuple[Any, dict[str, Any]]) -> dict[str, Any]:
    result = _reroll_worker_matching(task)
    cell = task[0]
    if (cell.seed, cell.draw_id) == (303, 1001):
        result["payload"]["episode"]["att_ours"] += 1e-9
    return result


def _reroll_worker_one_fails(task: tuple[Any, dict[str, Any]]) -> dict[str, Any]:
    cell = task[0]
    if (cell.seed, cell.draw_id) == (404, 1002):
        return {"name": cell.name, "ok": False, "payload": None, "error": "RuntimeError: engine gone at 17.3 s"}
    return _reroll_worker_matching(task)


def test_the_reroll_cells_are_the_campaigns_own_fifteen() -> None:
    cells = cs.reroll_cells()
    assert [(c.stage, c.subject, c.k, c.batch, c.seed, c.draw_id) for c in cells] == [
        ("reference", "mappo1000", 20, 64, seed, draw) for seed in SEEDS for draw in (1000, 1001, 1002)
    ]
    assert set(cells) <= set(cs.declared_campaign_cells("reference"))


def test_the_reroll_check_prints_match_by_name_and_fences_its_records(campaign: CampaignTree) -> None:
    kwargs = {"output_root": campaign.output_root, "corpus_root": campaign.corpus_root,
              "draws_root": campaign.draws_root, "data_dir": campaign.data_dir, "canary_seconds": 0.75, "workers": 3}
    result = cs.run_reference_reroll_check(**kwargs, worker=_reroll_worker_matching)
    names = [c.name for c in cs.reroll_cells()]
    assert result["lines"] == [f"reference_reroll_check MATCH {name}" for name in names]
    assert result["n_match"] == 15
    run_dir = Path(result["run_dir"])
    assert run_dir.parent == cs.g2_dir(campaign.output_root) and run_dir.name.startswith("reference_reroll_check_")
    assert sorted(p.name for p in run_dir.iterdir()) == sorted([*names, "verdict.json"])

    other = cs.run_reference_reroll_check(**kwargs, worker=_reroll_worker_one_differs)
    differing = [line for line in other["lines"] if "NO MATCH" in line]
    assert differing == ["reference_reroll_check NO MATCH cell_ref_mappo1000_k20_seed303_draw1001.json att_ours"]
    assert other["n_match"] == 14


def test_a_failed_roll_is_never_a_verdict_and_its_message_stays_behind_the_fence(campaign: CampaignTree) -> None:
    with pytest.raises(RuntimeError, match="RuntimeError") as error:
        cs.run_reference_reroll_check(
            output_root=campaign.output_root, corpus_root=campaign.corpus_root, draws_root=campaign.draws_root,
            data_dir=campaign.data_dir, canary_seconds=0.75, workers=2, worker=_reroll_worker_one_fails,
        )
    assert "17.3" not in str(error.value), "a failure message can carry a value: it stays in the fenced file"
    [run_dir] = [p for p in cs.g2_dir(campaign.output_root).iterdir()]
    assert sorted(p.name for p in run_dir.iterdir()) == ["failures.json"]
    assert "17.3" in (run_dir / "failures.json").read_text(encoding="utf-8")


def _reroll_worker_one_hangs(task: tuple[Any, dict[str, Any]]) -> dict[str, Any]:
    """The matching re-roll worker, except that seed 202's draw-1001 cell hangs (then, 30 s on, would match)."""
    if (task[0].seed, task[0].draw_id) == (202, 1001):
        _hang()
    return _reroll_worker_matching(task)


def test_a_hung_re_roll_is_a_failed_roll_the_command_exits_2_and_the_fence_holds(
    campaign: CampaignTree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Amendment D.2, D.2.2(3)(d): the fenced re-roll under the same result timeout -- a hung cell is a FAILED ROLL, never
    retried there: ``failures.json`` alone under the fence, naming exactly that cell; exit 2; nothing else written.

    *Mutation:* the re-roll check's timeout removed (the hung roll then matches, 30 s late) -> this dies.
    """
    monkeypatch.setattr(cs, "STAGE_RESULT_TIMEOUT_S", HANG_TEST_TIMEOUT_S, raising=True)
    monkeypatch.setattr(cs, "_reroll_worker", _reroll_worker_one_hangs, raising=True)
    before = sorted(path.name for path in cs.cells_dir(campaign.output_root).iterdir())
    argv = ["reference-reroll-check", "--output-root", str(campaign.output_root), "--corpus-root",
            str(campaign.corpus_root), "--data-dir", str(campaign.data_dir), "--draws-root", str(campaign.draws_root),
            "--canary-seconds", "0.75", "--workers", "3"]
    started = time.perf_counter()
    assert cs.main(argv) == 2
    assert time.perf_counter() - started < HANG_SECONDS, "the hung roll was waited out, not terminated"
    out = capsys.readouterr().out
    assert "REFUSED: reference_reroll_check could not run: 1 of 15 roll(s) failed (['HungRoll'])" in out
    assert "MATCH" not in out
    [run_dir] = [p for p in cs.g2_dir(campaign.output_root).iterdir()]
    assert sorted(p.name for p in run_dir.iterdir()) == ["failures.json"]
    fenced = json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))["fenced"]
    assert fenced == {"failures": [{
        "cell": "cell_ref_mappo1000_k20_seed202_draw1001.json",
        "error": "HungRoll: no result within 5 s; the pool was terminated (DEFERRED 104)",
    }]}
    assert sorted(path.name for path in cs.cells_dir(campaign.output_root).iterdir()) == before


# ======================================================================================================================
# resume-check's two findings
# ======================================================================================================================


def test_resume_check_names_an_undeclared_file_and_an_unresolvable_commit(campaign: CampaignTree) -> None:
    cells = cs.cells_dir(campaign.output_root)
    assert cs.undeclared_chunk_names(campaign.output_root) == []
    assert cs.unresolvable_chunk_commits(campaign.output_root) == []
    (cells / "cell_mappo1000_k3_b64_seed101_draw1000.json").write_text("{}", encoding="utf-8")
    (cells / "COMPLETE").write_text("done\n", encoding="utf-8")
    (cells / "failed").mkdir(exist_ok=True)
    assert cs.undeclared_chunk_names(campaign.output_root) == ["cell_mappo1000_k3_b64_seed101_draw1000.json"]
    victim = cs.chunk_path(campaign.output_root, _cell("sweep", "mix50", 5, 64, 202, 1033))
    chunk = _read(victim)
    chunk["code_commit"] = "0" * 40
    _rewrite(victim, chunk)
    assert cs.unresolvable_chunk_commits(campaign.output_root) == [{"chunk": victim.name, "code_commit": "0" * 40}]


# ======================================================================================================================
# T-report
# ======================================================================================================================


def _break(campaign: CampaignTree, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    cells = cs.cells_dir(campaign.output_root)
    sweep = cs.chunk_path(campaign.output_root, _cell("sweep", "mix50", 10, 64, 505, 1066))
    if case == "rows-at-another-digest":
        monkeypatch.setattr(cs, "REFERENCE_ROWS_SHA256", "0" * 64, raising=True)
    elif case == "record-pin-unset":
        monkeypatch.setattr(cs, "TRAIN_RECORD_SHA256", None, raising=True)
    elif case == "a-declared-cell-absent":
        sweep.unlink()
    elif case == "a-checkpoint-digest-the-record-does-not-name":
        chunk = _read(sweep)
        chunk["checkpoint"]["sha256"] = "d" * 64
        _rewrite(sweep, chunk)
    elif case == "no-canary":
        (cells / "canary.json").unlink()
    elif case == "no-gate-record":
        cs.gate_record_path(campaign.output_root).unlink()
    elif case == "a-chunk-rolled-by-other-code":
        chunk = _read(sweep)
        chunk["code_commit"] = C1A_COMMIT
        _rewrite(sweep, chunk)
    elif case == "an-undeclared-file":
        (cells / "cell_mix50_k3_b64_seed101_draw1000.json").write_text("{}", encoding="utf-8")
    elif case == "another-thread-count":
        chunk = _read(sweep)
        chunk["torch_num_threads"] = 16
        _rewrite(sweep, chunk)
    elif case == "another-demand":
        chunk = _read(sweep)
        chunk["demand"]["config_sha256"] = "d" * 64
        _rewrite(sweep, chunk)
    else:
        raise AssertionError(case)


#: Each refusal of the report, and what its message must name.
REPORT_REFUSALS: dict[str, str] = {
    "rows-at-another-digest": "p4_k20_att_engine_rows",
    "record-pin-unset": "TRAIN_RECORD_SHA256 is not set",
    "a-declared-cell-absent": r"cell_mix50_k10_b64_seed505_draw1066.*no estimator is computed on a partial set",
    "a-checkpoint-digest-the-record-does-not-name": "cell_mix50_k10_b64_seed505_draw1066.*checkpoint",
    "no-canary": "canary.json",
    "no-gate-record": r"reference_gate\.json is absent: no sweep cell is rolled, and no report written",
    "a-chunk-rolled-by-other-code": r"J1\(c\)",
    "an-undeclared-file": "cell_mix50_k3_b64_seed101_draw1000",
    "another-thread-count": "torch_num_threads",
    "another-demand": "cell_mix50_k10_b64_seed505_draw1066.*demand",
}


@pytest.mark.parametrize("case", sorted(REPORT_REFUSALS))
def test_every_refusal_of_the_report_precedes_every_write(
    campaign: CampaignTree, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    _break(campaign, monkeypatch, case)
    with pytest.raises((ValueError, FileNotFoundError), match=REPORT_REFUSALS[case]):
        _build(campaign)
    code = cs.main([
        "report", "--output-root", str(campaign.output_root), "--corpus-root", str(campaign.corpus_root),
        "--data-dir", str(campaign.data_dir), "--draws-root", str(campaign.draws_root),
    ])
    assert code == 2
    assert not cs.artifact_path(campaign.output_root).exists()
    assert not (cs.campaign_root(campaign.output_root) / "artifacts").exists()


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_contains_key(v, key) for v in value)
    return False


def test_the_verdict_is_the_statistic_recomputed_here_from_the_raw_chunks(built: CampaignTree,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, built)
    artifact = _build(built)
    cells = cs.cells_dir(built.output_root)
    raw: dict[tuple[str, int, int], dict[str, float]] = {}
    for path in cells.glob("cell_*.json"):
        chunk = json.loads(path.read_bytes())
        raw[(chunk["arm"], chunk["seed"], chunk["draw_id"])] = chunk["episode"]

    def a_d(arm: str, definition: str) -> dict[int, float]:
        return {
            draw: float(np.mean(np.asarray([raw[(arm, seed, draw)][definition] for seed in SEEDS], dtype=np.float64)))
            for draw in DRAWS
        }

    confirmatory = artifact["confirmatory"]
    assert (confirmatory["subject"], confirmatory["definition"]) == ("mappo1000", "att_engine")
    family = confirmatory["family"]
    levels = {k: a_d(f"mappo1000_k{k}_b64", "att_engine") for k in (1, 2, 5, 10, 20)}
    s_values = []
    for draw in DRAWS:
        total = 0
        for coefficient, k in zip((-2, -1, 0, 1, 2), (1, 2, 5, 10, 20)):
            total = total + coefficient * levels[k][draw]
        s_values.append(total)
    assert family["S"]["mean"] == float(np.mean(np.asarray(s_values, dtype=np.float64)))
    assert family["tests"]["T1"]["p_value"] == route_b(s_values, "less")[4]
    for test, k in (("T2", 1), ("T3", 2)):
        shifted = [(levels[k][draw] - levels[20][draw]) - 0.6263 for draw in DRAWS]
        assert family["tests"][test]["p_value"] == route_b(shifted, "greater")[4]
        gaps = np.asarray([levels[k][draw] - levels[20][draw] for draw in DRAWS], dtype=np.float64)
        assert family["G"][str(k)]["mean"] == float(gaps.mean())
    p = tuple(family["tests"][name]["p_value"] for name in ("T1", "T2", "T3"))
    decisions = holm_by_hand(p)
    assert tuple(family["holm"]["rejected"][name] for name in ("T1", "T2", "T3")) == decisions
    expected = "iii" if not decisions[0] else ("i" if decisions[1] and decisions[2] else "ii")
    assert confirmatory["outcome"] == family["outcome"] == expected
    assert confirmatory["sentence"] == cs.outcome_sentence(expected, family=family)
    assert confirmatory["tests_not_rejected"] == [name for name, rejected in zip(("T1", "T2", "T3"), decisions)
                                                  if not rejected]

    assert artifact["format_version"] == "p5.3c-context-sweep/1.0" and artifact["n_cells"] == 7000
    assert not _contains_key(artifact, "series"), "the per-decision series stay in the chunks"
    arms = sorted({arm for arm, _seed, _draw in raw})
    assert sorted(artifact["per_draw_means"]) == arms and len(arms) == 14
    for arm in arms:
        for definition in ("att_engine", "att_ours"):
            recomputed = a_d(arm, definition)
            assert artifact["per_draw_means"][arm][definition] == {str(draw): recomputed[draw] for draw in DRAWS}
    mix = artifact["exploratory_mix50"]
    assert mix["label"].startswith("EXPLORATORY") and "sentence" not in mix
    mix_levels = {k: a_d(f"mix50_k{k}_b64", "att_engine") for k in (1, 2, 5, 10, 20)}
    assert mix["family"] == cs.confirmatory_family(mix_levels)
    ours = artifact["att_ours"]["family"]
    assert ours == cs.confirmatory_family({k: a_d(f"mappo1000_k{k}_b64", "att_ours") for k in (1, 2, 5, 10, 20)})
    assert artifact["mix50_reference"]["n_compared"] == 500 and artifact["mix50_reference"]["n_equal"] == 500
    assert artifact["reference_gate"]["sha256"] == cs._sha256_file(cs.gate_record_path(built.output_root))
    assert set(artifact["expectations"]) == {"mappo1000", "mix50"}
    assert {entry["verdict"] for entry in artifact["expectations"].values()} <= {"held", "refuted"}
    assert set(artifact["equal_supervision"]["contrasts"]) == {"mappo1000_k1_b1280 - mappo1000_k1_b64",
                                                                "mappo1000_k2_b640 - mappo1000_k2_b64"}
    assert set(artifact["plateau"]["mappo1000"]["att_engine"]) == {"point_estimate_within_delta",
                                                                   "ci_within_plus_minus_delta"}
    assert len(artifact["pairwise"]["mappo1000"]["att_engine"]) == 10
    assert len(artifact["per_seed"]["mappo1000"]["att_engine"]["seeds"]) == 5
    assert artifact["k20_reproduction"]["mappo1000"]["n_seeds_equal"] == 5
    assert set(artifact["loss_per_target"]) == set(cs.registered_arms())
    assert any("context lengths only" in line for line in artifact["what_this_does_not_say"])


#: Amendment C, C2's table as the coordinator printed it at gate G3 from his own script -- per arm: loop seconds mean,
#: min, max; ms / step; final loss per target mean, min, max; targets / step; first-window loss.  Typed here as the
#: independent statement the module's computation from the pinned record must reproduce digit for digit.
C2_TABLE: dict[str, tuple[str, ...]] = {
    "mappo1000_k20_b64": ("222.3", "206.7", "237.9", "5.56", "0.0177", "0.0132", "0.0244", "1,246.2", "0.614"),
    "mappo1000_k10_b64": ("235.6", "220.6", "267.0", "5.89", "0.0338", "0.0265", "0.0393", "632.0", "0.644"),
    "mappo1000_k5_b64": ("238.4", "216.0", "251.4", "5.96", "0.0811", "0.0443", "0.1303", "318.2", "0.677"),
    "mappo1000_k2_b64": ("256.8", "235.3", "265.0", "6.42", "0.1603", "0.0957", "0.2371", "127.8", "0.732"),
    "mappo1000_k1_b64": ("247.9", "220.0", "264.7", "6.20", "0.2532", "0.1435", "0.3485", "64.0", "0.771"),
    "mappo1000_k1_b1280": ("343.9", "328.4", "362.3", "8.60", "0.0113", "0.0056", "0.0168", "1,280.0", "0.620"),
    "mappo1000_k2_b640": ("274.3", "260.8", "296.2", "6.86", "0.0114", "0.0069", "0.0165", "1,278.2", "0.623"),
    "mix50_k20_b64": ("215.7", "208.0", "230.3", "5.39", "0.0085", "0.0032", "0.0161", "1,246.2", "0.715"),
    "mix50_k10_b64": ("230.5", "220.7", "250.6", "5.76", "0.0131", "0.0050", "0.0219", "632.0", "0.786"),
    "mix50_k5_b64": ("331.2", "223.0", "695.6", "8.28", "0.0315", "0.0166", "0.0422", "318.2", "0.880"),
    "mix50_k2_b64": ("484.5", "172.0", "1,718.4", "12.11", "0.0864", "0.0504", "0.1040", "127.8", "1.035"),
    "mix50_k1_b64": ("179.4", "169.6", "187.1", "4.49", "0.1129", "0.0814", "0.1406", "64.0", "1.151"),
}

#: Amendment C, C3's wording of the equal result, typed here (a ruling on wording: (i), (ii), (iii) in that order).
C3_SENTENCES: tuple[str, ...] = (
    "The sweep's K = 20 arm is P4's model, and mix50's is P4.7's: the K = 20 reproduction is equal on every tensor for "
    "all ten (subject, seed) pairs (A26(b)'s measurement). H4's K = 20 point is P4's checkpoint re-evaluated, and the "
    "equal-supervision and K < 20 arms differ from P4 by K (and batch) alone, trained by the same code path that "
    "reproduced P4 bit for bit.",
    "A26(c)'s reference gate -- P4's five checkpoints re-evaluated == the committed rows -- therefore also pins the "
    "sweep's own K = 20 cells: any difference between the K = 20 arm's cells and the reference rows would be an "
    "evaluation-path difference, never a model difference.",
    "The fenced timing run's same-seed repeat and this reproduction, both equal, are a property of THIS GPU, driver and "
    "torch build under the registered regime, stated as such, not as determinism of the method.",
)


def _as_c2_prints(row: dict[str, Any]) -> tuple[str, ...]:
    loop, loss = row["loop_seconds"], row["final_loss_per_target"]
    return (
        f"{loop['mean']:,.1f}", f"{loop['min']:,.1f}", f"{loop['max']:,.1f}", f"{row['ms_per_step']:.2f}",
        f"{loss['mean']:.4f}", f"{loss['min']:.4f}", f"{loss['max']:.4f}",
        f"{row['supervised_targets_per_step']:,.1f}", f"{row['first_window_loss']:.3f}",
    )


def test_the_training_table_is_c2s_computed_from_the_pinned_record() -> None:
    """Amendment C, C4.2: the report's table is C2's, computed from the PINNED record (never retyped) -- every printed
    digit of the coordinator's G3 table reproduced; each seed's wall seconds the record's, read here by ``json``, the
    two stalls C2 names among them; the observation beside it as the record gives it."""
    table = cs.training_table(cs.load_train_record(DATA))
    assert set(table["arms"]) == set(C2_TABLE) == set(cs.registered_arms())
    for arm, printed in C2_TABLE.items():
        assert _as_c2_prints(table["arms"][arm]) == printed, arm
    raw = json.loads((DATA / "p5_3c_train.json").read_bytes())
    for arm, row in table["arms"].items():
        assert row["loop_seconds"]["seeds"] == {
            str(seed): raw["runs"][f"{arm}_seed{seed}"]["loop_seconds"] for seed in SEEDS
        }, arm
    assert f"{table['arms']['mix50_k5_b64']['loop_seconds']['seeds']['505']:,.1f}" == "695.6"
    assert f"{table['arms']['mix50_k2_b64']['loop_seconds']['seeds']['101']:,.1f}" == "1,718.4"
    observation = table["observation"]
    assert observation["final_loss_falls_with_k"] == {"mappo1000": True, "mix50": True}
    assert {
        arm: (entry["k20_arm"], f"{entry['mean']:.4f}", f"{entry['k20_mean']:.4f}", entry["below_k20"])
        for arm, entry in observation["equal_supervision"].items()
    } == {
        "mappo1000_k1_b1280": ("mappo1000_k20_b64", "0.0113", "0.0177", True),
        "mappo1000_k2_b640": ("mappo1000_k20_b64", "0.0114", "0.0177", True),
    }
    assert table["record"] == {"name": "p5_3c_train.json", "sha256": G3_RECORD_SHA256}


def test_the_k20_reading_is_c3s_wording_made_only_on_what_it_presumes() -> None:
    """Amendment C, C3: on the pinned record (ten pairs equal, the timing repeat equal) the reading is C3's three
    sentences; one pair unequal, one pair absent, or the repeat unequal or unrecorded -> it is NOT made, and says so."""
    record = cs.load_train_record(DATA)
    reading = cs.k20_reading(record)
    assert (reading["made"], reading["n_pairs_equal"], reading["timing_repeat_equal"]) == (True, 10, True)
    assert reading["sentences"] == list(C3_SENTENCES)

    unequal = copy.deepcopy(record)
    entry = unequal["k20_reproduction"]["record"]["subjects"]["mix50"]["seeds"][3]
    entry["all_equal"], entry["n_parameters_differing"] = False, 2
    absent = copy.deepcopy(record)
    del absent["k20_reproduction"]["record"]["subjects"]["mappo1000"]["seeds"][0]
    repeat_differs = copy.deepcopy(record)
    repeat_differs["timing"]["record"]["repeat"]["weights_sha256_equal"] = False
    unrecorded = copy.deepcopy(record)
    del unrecorded["timing"]
    for case, n_equal, repeat in ((unequal, 9, True), (absent, 9, True), (repeat_differs, 10, False),
                                  (unrecorded, 10, False)):
        reading = cs.k20_reading(case)
        assert (reading["made"], reading["n_pairs_equal"], reading["timing_repeat_equal"]) == (False, n_equal, repeat)
        [sentence] = reading["sentences"]
        assert sentence.startswith("Amendment C, C3's reading is NOT made") and f"{n_equal} of ten pairs" in sentence
        assert not any(c3 in sentence for c3 in C3_SENTENCES)


def test_the_report_carries_c2s_table_and_c3s_reading_of_the_pinned_record(built: CampaignTree,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """The artifact's ``training_table`` recomputed HERE from the fixture's pinned record (five seeds, ascending, summed
    left to right from zero -- what ``np.mean`` does on five values); its ``k20_reproduction`` carries C3's reading."""
    _pin(monkeypatch, built)
    artifact = _build(built)
    record = json.loads(built.checkpoints.record_path.read_bytes())
    table = artifact["training_table"]
    assert set(table["arms"]) == set(cs.registered_arms())
    for arm, row in table["arms"].items():
        entries = [record["runs"][f"{arm}_seed{seed}"] for seed in SEEDS]

        def mean(values: list[float]) -> float:
            total = 0.0
            for value in values:
                total = total + value
            return total / len(values)

        loop = [entry["loop_seconds"] for entry in entries]
        loss = [entry["final_loss"] for entry in entries]
        assert row["loop_seconds"] == {"mean": mean(loop), "min": min(loop), "max": max(loop),
                                       "seeds": {str(seed): value for seed, value in zip(SEEDS, loop)}}
        assert row["ms_per_step"] == mean([entry["ms_per_step"] for entry in entries])
        assert row["final_loss_per_target"] == {"mean": mean(loss), "min": min(loss), "max": max(loss)}
        assert row["supervised_targets_per_step"] == mean([entry["supervised_targets_per_step"]["mean"]
                                                           for entry in entries])
        assert row["first_window_loss"] == mean([entry["loss_per_supervised_target"][0] for entry in entries])
    assert "WALL TIME" in table["wall_time"] and "medians or minima" in table["wall_time"]
    assert "not a finding" in table["observation"]["label"] and "T1-T3" in table["observation"]["label"]
    reading = artifact["k20_reproduction"]["reading"]
    assert reading["made"] is True and reading["sentences"] == list(C3_SENTENCES)
    assert artifact["registered_in"] == ("PREREGISTRATION A26 as corrected by A26.1; BRIEF_42 C3 and Amendments A, A.1, "
                                         "B, B.1, C")
    assert reading["source"] == "BRIEF_42 Amendment C, C3 (a ruling on wording, not on a number)"


def test_the_confirmatory_block_records_the_five_arms_it_was_fed_and_the_sweeps_own_k20_arm(
    built: CampaignTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amendment D, D4.3(c): G6's independent route reads the arms from the artifact, not from the code -- the five
    batch-64 ``mappo1000`` arms in K order, and the K = 20 level the sweep's OWN arm, never the reference arm; the
    family is the one those arms' chunks give."""
    _pin(monkeypatch, built)
    confirmatory = _build(built)["confirmatory"]
    arms = ["mappo1000_k1_b64", "mappo1000_k2_b64", "mappo1000_k5_b64", "mappo1000_k10_b64", "mappo1000_k20_b64"]
    assert confirmatory["arms"] == arms
    assert confirmatory["k20_arm"] == "mappo1000_k20_b64"
    raw: dict[tuple[str, int, int], float] = {}
    for path in cs.cells_dir(built.output_root).glob("cell_mappo1000_k*_b64_*.json"):
        chunk = json.loads(path.read_bytes())
        raw[(chunk["arm"], chunk["seed"], chunk["draw_id"])] = chunk["episode"]["att_engine"]
    levels = {
        k: {draw: float(np.mean(np.asarray([raw[(arm, seed, draw)] for seed in SEEDS], dtype=np.float64)))
            for draw in DRAWS}
        for k, arm in zip((1, 2, 5, 10, 20), arms)
    }
    assert confirmatory["family"] == cs.confirmatory_family(levels)


def test_the_artifact_regenerates_byte_for_byte_and_is_written_once(campaign: CampaignTree,
                                                                    capsys: pytest.CaptureFixture[str]) -> None:
    first = _build(campaign)
    second = _build(campaign)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    code = cs.main([
        "report", "--output-root", str(campaign.output_root), "--corpus-root", str(campaign.corpus_root),
        "--data-dir", str(campaign.data_dir), "--draws-root", str(campaign.draws_root),
    ])
    out = capsys.readouterr().out
    assert code == 0
    path = cs.artifact_path(campaign.output_root)
    assert json.loads(path.read_bytes()) == json.loads(json.dumps(first))
    assert str(path) in out and "7000" in out.replace(",", "")
    for word in ("reject", "Confirmatory", "outcome", "held", "refuted"):
        assert word not in out, word
    assert cs.write_context_sweep_artifact(campaign.output_root, first) == path
    with pytest.raises(ValueError, match="differs"):
        cs.write_context_sweep_artifact(campaign.output_root, {**first, "n_cells": 1})


def test_the_campaign_manifest_lists_every_file_under_output_p5_3c_and_is_reverified(campaign: CampaignTree) -> None:
    (campaign.output_root / "unrelated.txt").write_text("not ours\n", encoding="utf-8")
    path = cs.write_campaign_manifest(campaign.output_root)
    assert path == campaign.output_root / "SHA256SUMS_p5_3c.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    listed = [line.split("  ", 1)[1] for line in lines]
    on_disk = sorted(
        str(p.relative_to(campaign.output_root)) for p in cs.campaign_root(campaign.output_root).rglob("*") if p.is_file()
    )
    assert listed == on_disk and "unrelated.txt" not in path.read_text(encoding="utf-8")
    for line in lines:
        digest, relative = line.split("  ", 1)
        assert cs._sha256_file(campaign.output_root / relative) == digest
    assert cs.write_campaign_manifest(campaign.output_root) == path
    (cs.cells_dir(campaign.output_root) / "COMPLETE").write_text("late\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        cs.write_campaign_manifest(campaign.output_root)


def test_the_commands_exit_as_the_driver_reads_them(
    campaign: CampaignTree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exit status is what the driver acts on: ``resume-check`` 1 on a finding; ``cells`` 1 when a cell failed;
    ``reference-reroll-check`` 0 only on fifteen MATCH, 1 on a NO MATCH, 2 when a roll failed; ``campaign-manifest`` 0.
    Written after their handlers (thin wrappers), so NOT red first; their strength is shown by mutation.
    """
    root = ["--output-root", str(campaign.output_root)]
    inputs = [*root, "--corpus-root", str(campaign.corpus_root), "--data-dir", str(campaign.data_dir),
              "--draws-root", str(campaign.draws_root)]
    assert cs.main(["resume-check", *root]) == 0
    assert "resume_check PASSED" in capsys.readouterr().out
    (cs.cells_dir(campaign.output_root) / "stray.json").write_text("{}", encoding="utf-8")
    assert cs.main(["resume-check", *root]) == 1
    assert "stray.json is not a declared chunk" in capsys.readouterr().out
    (cs.cells_dir(campaign.output_root) / "stray.json").unlink()

    def stage(**kwargs: Any) -> dict[str, Any]:
        return {"stage": kwargs["stage"], "n_declared": 1000, "n_reused": 999, "n_rolled": 0, "n_failed": 1,
                "failures": [{"name": "x", "error": "E"}], "wall_seconds": 1.0}

    monkeypatch.setattr(cs, "run_campaign_stage", stage, raising=True)
    assert cs.main(["cells", *inputs, "--stage", "reference", "--canary-seconds", "0.8"]) == 1
    assert "1 failed" in capsys.readouterr().out

    lines = [f"reference_reroll_check MATCH cell_{n}.json" for n in range(15)]
    monkeypatch.setattr(cs, "run_reference_reroll_check",
                        lambda **kwargs: {"lines": lines, "n_match": 15, "run_dir": Path("g2")}, raising=True)
    assert cs.main(["reference-reroll-check", *inputs, "--canary-seconds", "0.8"]) == 0
    assert capsys.readouterr().out.count("reference_reroll_check MATCH ") == 15
    mixed = [*lines[:14], "reference_reroll_check NO MATCH cell_14.json att_ours"]
    monkeypatch.setattr(cs, "run_reference_reroll_check",
                        lambda **kwargs: {"lines": mixed, "n_match": 14, "run_dir": Path("g2")}, raising=True)
    assert cs.main(["reference-reroll-check", *inputs, "--canary-seconds", "0.8"]) == 1
    capsys.readouterr()

    def failing(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("reference_reroll_check could not run: 1 of 15 roll(s) failed (['OSError'])")

    monkeypatch.setattr(cs, "run_reference_reroll_check", failing, raising=True)
    assert cs.main(["reference-reroll-check", *inputs, "--canary-seconds", "0.8"]) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert cs.main(["campaign-manifest", *root]) == 0
    assert "re-verified" in capsys.readouterr().out


def _inputs_break(campaign: CampaignTree, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    if case == "settings-differ-from-p8-4b":
        monkeypatch.setattr(cs, "_rederivation_settings", lambda subject, root: {"max_steps": 361, "delta_time": 10},
                            raising=True)
    elif case == "rows-name-another-checkpoint":
        rows = _read(campaign.data_dir / "p4_k20_att_engine_rows.json")
        rows["checkpoints"]["303"]["sha256"] = "d" * 64
        _rewrite(campaign.data_dir / "p4_k20_att_engine_rows.json", rows)
        monkeypatch.setattr(cs, "REFERENCE_ROWS_SHA256",
                            cs._sha256_file(campaign.data_dir / "p4_k20_att_engine_rows.json"), raising=True)
    elif case == "record-names-another-reference":
        record = _read(campaign.checkpoints.record_path)
        record["k20_reproduction"]["record"]["subjects"]["mix50"]["seeds"][2]["reference_sha256"] = "d" * 64
        _rewrite(campaign.checkpoints.record_path, record)
        monkeypatch.setattr(cs, "TRAIN_RECORD_SHA256", cs._sha256_file(campaign.checkpoints.record_path), raising=True)
    elif case == "a-draw-not-materialised":
        (campaign.draws_root / "cityflow1x1" / "draw_1042" / "flow.json").unlink()
    elif case == "a-p8-4b-mix50-cell-absent":
        (campaign.output_root / "p8_4b_rederivation" / "cell_hz1x1_dt_at_mix50_seed404_draw1017.json").unlink()
    elif case == "device-memory-below-budget":
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1024 * 2**20, 16 * 2**30), raising=True)
    elif case == "one-of-the-sixty-at-another-digest":
        path = campaign.output_root / "p5_3c_training" / "checkpoints" / "mappo1000_k2_b640_seed404.pt"
        path.write_bytes(path.read_bytes() + b"\0")
    else:
        raise AssertionError(case)


#: Each pre-token refusal of ``campaign-inputs`` and what its message must name.
INPUTS_REFUSALS: dict[str, str] = {
    "settings-differ-from-p8-4b": "evaluation env settings differ from P8.4b's",
    "rows-name-another-checkpoint": "the reference rows name another checkpoint of seed 303",
    "record-names-another-reference": "K = 20 reference of mix50 seed 303 is not the published file",
    "a-draw-not-materialised": "draw_1042/flow.json is absent",
    "a-p8-4b-mix50-cell-absent": "cell_hz1x1_dt_at_mix50_seed404_draw1017.json is absent",
    "device-memory-below-budget": "below the 6972 MiB budget",
    # Amendment C, C4.1: the sixty verified at the pinned record's digests.
    "one-of-the-sixty-at-another-digest": "is not the pinned record's [0-9a-f]{64} for mappo1000_k2_b640_seed404",
}


@pytest.mark.parametrize("case", [None, *sorted(INPUTS_REFUSALS)])
def test_the_campaign_inputs_pass_on_the_fixture_and_refuse_each_mismatch(
    campaign: CampaignTree, monkeypatch: pytest.MonkeyPatch, case: str | None
) -> None:
    """Written after ``check_campaign_inputs`` (its refusals were a review finding), so NOT red first; each case's
    mutant is executed.  CUDA's two answers are injected: the check reads them, it does not need the device here."""
    monkeypatch.setattr(cs, "evaluation_env_settings", lambda subject, root: {"max_steps": 360, "delta_time": 10},
                        raising=True)
    monkeypatch.setattr(cs, "_rederivation_settings", lambda subject, root: {"max_steps": 360, "delta_time": 10},
                        raising=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (14_000 * 2**20, 16 * 2**30), raising=True)
    kwargs = {"output_root": campaign.output_root, "corpus_root": campaign.corpus_root,
              "draws_root": campaign.draws_root, "data_dir": campaign.data_dir}
    if case is None:
        facts = cs.check_campaign_inputs(**kwargs)
        assert (facts["n_published"], facts["n_runs"], facts["device_budget_mib"]) == (10, 60, 6972)
        assert facts["train_record_sha256"] == campaign.record_sha256
        return
    _inputs_break(campaign, monkeypatch, case)
    with pytest.raises((ValueError, FileNotFoundError), match=INPUTS_REFUSALS[case]):
        cs.check_campaign_inputs(**kwargs)


def test_the_campaign_inputs_refuse_without_cuda_and_with_the_record_unpinned(campaign: CampaignTree,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = {"output_root": campaign.output_root, "corpus_root": campaign.corpus_root,
              "draws_root": campaign.draws_root, "data_dir": campaign.data_dir}
    monkeypatch.setattr(cs, "TRAIN_RECORD_SHA256", None, raising=True)
    with pytest.raises(ValueError, match="TRAIN_RECORD_SHA256 is not set"):
        cs.check_campaign_inputs(**kwargs)
    monkeypatch.setattr(cs, "TRAIN_RECORD_SHA256", campaign.record_sha256, raising=True)
    monkeypatch.setattr(cs, "evaluation_env_settings", lambda subject, root: {"max_steps": 360, "delta_time": 10},
                        raising=True)
    monkeypatch.setattr(cs, "_rederivation_settings", lambda subject, root: {"max_steps": 360, "delta_time": 10},
                        raising=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=True)
    with pytest.raises(ValueError, match="CUDA"):
        cs.check_campaign_inputs(**kwargs)


# ======================================================================================================================
# Gated: the real inputs (after the trainings; A.1 -- CUDA)
# ======================================================================================================================

_OUTPUT = os.environ.get("RLTRAFFIC_OUTPUT_ROOT")
_CORPUS = os.environ.get("RLTRAFFIC_CORPUS_V11")


@pytest.mark.skipif(not _CORPUS, reason="needs RLTRAFFIC_CORPUS_V11 (the gitignored datasets_v11 corpus's manifests)")
@pytest.mark.parametrize("subject", ["mappo1000", "mix50"])
def test_the_evaluation_env_settings_are_p8_4bs_for_both_subjects(subject: str) -> None:
    """Plan A7: the settings a cell rolls under are the ones P8.4b's reference rows were rolled under."""
    from types import SimpleNamespace

    from offline.att_rederivation import rederivation_env_settings

    corpus = Path(str(_CORPUS))
    assert cs.evaluation_env_settings(subject, corpus) == rederivation_env_settings(
        "hz1x1", subject, SimpleNamespace(corpus_root=corpus)
    )


def _draws_beside(output_root: str | None) -> Path | None:
    """The materialised draws of the tree whose ``output/`` the gate names: ``<main>/scenarios/draws``."""
    return None if not output_root else Path(output_root).parent / "scenarios" / "draws"


def _real_cell_available() -> str | None:
    if not (_OUTPUT and _CORPUS):
        return "needs RLTRAFFIC_OUTPUT_ROOT (output/p4_dt, output/p4_7, output/p8_4b_rederivation) and RLTRAFFIC_CORPUS_V11"
    if not torch.cuda.is_available():
        return "needs CUDA: A26.1(b) evaluates the DT on this GPU, as the reference rows were produced"
    draws = _draws_beside(_OUTPUT)
    if draws is None or not (draws / "cityflow1x1" / "draw_1000").is_dir():
        return "needs the materialised held-out draws at <RLTRAFFIC_OUTPUT_ROOT>/../scenarios/draws"
    return None


@pytest.mark.skipif(_real_cell_available() is not None, reason=str(_real_cell_available()))
@pytest.mark.parametrize("subject", ["mappo1000", "mix50"])
def test_one_real_reference_cell_reproduces_its_committed_row_under_equality(
    subject: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q17 (with A.1): ONE real CityFlow episode per published subject, seed 101 on draw 1000, through the campaign's
    own cell on CUDA -- P4's against the committed rows, P4.7's against P8.4b's cell -- under ``==``.

    An INSTRUMENT check of the episode's values: the code's provenance is injected as clean, because a chunk refuses a
    dirty tree (J1(d)) and this test may run from a working copy; the campaign rolls from the clean run worktree.
    """
    monkeypatch.setattr(cs, "_code_provenance", lambda: {"code_commit": head_commit(), "code_dirty": False},
                        raising=True)
    output = Path(str(_OUTPUT))
    draws = _draws_beside(_OUTPUT)
    assert draws is not None
    cell = _cell("reference", subject, 20, 64, 101, 1000)
    chunk = cs.run_campaign_cell(
        cell, output_root=output, corpus_root=Path(str(_CORPUS)), draws_root=draws, data_dir=REPO_ROOT / "docs" / "data",
        canary_seconds=None,
    )
    if subject == "mappo1000":
        rows = cs.load_reference_rows(REPO_ROOT / "docs" / "data")
        [row] = [r for r in rows["rows"] if (r["seed"], r["draw_id"]) == (101, 1000)]
    else:
        row = json.loads((output / "p8_4b_rederivation" / "cell_hz1x1_dt_at_mix50_seed101_draw1000.json").read_bytes())
    assert chunk["episode"]["att_engine"] == row["att_engine"]
    assert chunk["episode"]["att_ours"] == row["att_ours"]


def _sixty_available() -> str | None:
    if not _OUTPUT:
        return "needs RLTRAFFIC_OUTPUT_ROOT (output/p5_3c_training: the sixty trained checkpoints, gitignored)"
    checkpoints = Path(_OUTPUT) / "p5_3c_training" / "checkpoints"
    if len(list(checkpoints.glob("*.pt"))) != 60:
        return f"{checkpoints} does not hold the sixty trained checkpoints"
    return None


@pytest.mark.skipif(_sixty_available() is not None, reason=str(_sixty_available()))
def test_every_registered_run_resolves_through_the_pinned_record_and_the_real_checkpoints() -> None:
    """Amendment C, C4.1: all sixty through the REAL record at its pin and the REAL files, the payload guard included;
    each digest compared with the record read by THIS file's route and with the file hashed here."""
    output = Path(str(_OUTPUT))
    raw = json.loads((DATA / "p5_3c_train.json").read_bytes())
    for subject, k, batch, seed in registered_table():
        run = cs.RunSpec(subject, k, batch, seed)
        entry = raw["runs"][run.name]
        identity = cs.sweep_checkpoint(run, output_root=output, data_dir=DATA)
        assert identity["path"] == entry["checkpoint"] == f"p5_3c_training/checkpoints/{run.name}.pt"
        digest = hashlib.sha256((output / entry["checkpoint"]).read_bytes()).hexdigest()
        assert identity["sha256"] == entry["checkpoint_sha256"] == digest, run.name


def _inputs_available() -> str | None:
    return _real_cell_available() or _sixty_available()


@pytest.mark.skipif(_inputs_available() is not None, reason=str(_inputs_available()))
def test_the_campaign_inputs_pass_on_the_real_tree_at_the_pinned_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """Amendment C, C4.1: ``campaign-inputs`` reads the pinned record from ``docs/data`` and passes on the real tree --
    the rows, the record, the ten published checkpoints (the rows' and the record's K = 20 references the same files),
    the sixty at the record's digests, both subjects' settings equal to P8.4b's, the 100 draws, P8.4b's 500 ``mix50``
    cells, P5.3b's means, CUDA.  The device BUDGET is set to zero here: it is the machine's state at the start, which
    the driver checks before the token and the fixture's ``device-memory-below-budget`` case exercises; this test's
    subject is the inputs."""
    monkeypatch.setattr(cs, "EVAL_DEVICE_BUDGET_MIB", 0, raising=True)
    draws = _draws_beside(_OUTPUT)
    assert draws is not None
    facts = cs.check_campaign_inputs(output_root=Path(str(_OUTPUT)), corpus_root=Path(str(_CORPUS)), draws_root=draws,
                                     data_dir=DATA)
    assert (facts["n_published"], facts["n_runs"]) == (10, 60)
    assert (facts["rows_sha256"], facts["train_record_sha256"]) == (cs.REFERENCE_ROWS_SHA256, G3_RECORD_SHA256)
