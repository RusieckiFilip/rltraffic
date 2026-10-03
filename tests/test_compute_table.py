"""P8.2 (``BRIEF_43`` §5-§6, Amendment A): the table builder ``offline/compute_table.py``.

* **T-params (load-bearing):** each family's parameter count by two routes -- the model loaded through the method's
  own loader (route A) and the payload's tensors that are the skeleton's PARAMETERS (route B) -- equal; ``trained >=
  deployed``. Ungated on synthetic payloads (including a fixture module that registers a buffer, and a MAPPO learner
  whose optimiser holds Adam's moments), gated on one real checkpoint per family. *Mutations:* buffers counted ->
  dies; MAPPO's optimiser or normaliser tensors counted -> dies; IQL's Polyak target counted as trained -> dies.
* **T-sources:** every number of the artifact that names a JSON path (or a log line) equals the record's value
  there, read HERE with ``json`` and this file's own path reader; the medians and ranges recompute; a record at
  another digest refuses; a value missing from its record that is not a declared absence refuses; a field with no
  source refuses; a declared absence is written as ``null`` with its reason.
* **The latency run's refusals:** no ``COMPLETE``, a throttled or non-reproducing canary, a file outside the manifest
  or at another digest, statistics that do not recompute from the record's own nanoseconds.

GATES: the real-record tests skip naming ``RLTRAFFIC_OUTPUT_ROOT`` / ``RLTRAFFIC_CORPUS_V11`` (gitignored, main tree).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch
import torch.nn as nn

import offline.compute_latency as cl
import offline.compute_table as ct

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"


# ----------------------------------------------------------------------
# Synthetic payloads
# ----------------------------------------------------------------------


class _Buffered(nn.Module):
    """A fixture the shipped models are not: it REGISTERS a buffer, so counting buffers is visible."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 4)
        self.register_buffer("running", torch.zeros(5))


def _module_count(*modules: nn.Module) -> int:
    return sum(int(p.numel()) for module in modules for p in module.parameters())


def _dt_payload() -> tuple[dict[str, Any], nn.Module]:
    from agent.DTAgent import DecisionTransformer, DTConfig

    config = DTConfig(state_dim=5, n_actions=3, context_length=4, n_layer=1, n_head=1, d_model=8, dropout=0.0,
                      max_ep_len=10)
    model = DecisionTransformer(config)
    return {"format_version": "dt-checkpoint/1.0", "config": config.to_json_obj(), "model": model.state_dict()}, model


def _spatial_payload() -> tuple[dict[str, Any], nn.Module]:
    from agent.SpatialDTAgent import SpatialDecisionTransformer, SpatialDTConfig

    config = SpatialDTConfig(state_dim=6, n_actions=4, n_nodes=3, context_length=4, n_layer=1, n_head=2, d_model=8,
                             dropout=0.0, max_ep_len=10, spatial_mixing=False)
    model = SpatialDecisionTransformer(config)
    payload = {"format_version": "spatial-dt-checkpoint/1.0", "config": config.to_json_obj(), "model": model.state_dict()}
    return payload, model


def _trunks(out_dims: dict[str, int]) -> tuple[dict[str, Any], dict[str, nn.Module]]:
    from agent.OfflineBaselines import MLPTrunk, TrunkConfig

    config = TrunkConfig(state_dim=5, n_actions=3, d_model=8, n_layer=1, dropout=0.0)
    modules = {name: MLPTrunk(config, out) for name, out in out_dims.items()}
    merged = {f"{name}.{key}": value for name, module in modules.items() for key, value in module.state_dict().items()}
    return {"config": config.to_json_obj(), "model": merged}, modules


def _mappo_payload() -> tuple[dict[str, Any], Any]:
    from agent.MAPPOAgent import _CentralizedMAPPO

    learner = _CentralizedMAPPO(
        action_counts=[3, 4], lr=1e-3, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, entropy_coef=0.01,
        value_coef=0.5, update_epochs=1, minibatch_size=4, rollout_size=8, hidden_dim=8, max_grad_norm=0.5,
        device=torch.device("cpu"),
    )
    learner.ensure_initialized([5, 6], global_feature_dim=3)
    for parameter in list(learner.actors.parameters()) + list(learner.critic.parameters()):
        parameter.grad = torch.ones_like(parameter)
    learner.optimizer.step()  # Adam now holds exp_avg / exp_avg_sq for every parameter
    return {"steps_done": 7, "learner": learner.export_state(), "global_metric_keys": ["m"]}, learner


def _all_tensor_elements(payload: Any) -> int:
    if isinstance(payload, torch.Tensor):
        return int(payload.numel())
    if isinstance(payload, dict):
        return sum(_all_tensor_elements(v) for v in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(_all_tensor_elements(v) for v in payload)
    return 0


# ----------------------------------------------------------------------
# T-params (load-bearing), ungated
# ----------------------------------------------------------------------


def test_t_params_parameter_names_are_the_parameters_never_the_buffers() -> None:
    module = _Buffered()
    assert ct.parameter_names(module) == {"lin.weight", "lin.bias"}
    state = module.state_dict()
    assert set(state) == {"lin.weight", "lin.bias", "running"}
    assert ct.count_state_parameters(state, module) == 16


def test_t_params_a_state_missing_a_parameter_refuses() -> None:
    module = _Buffered()
    state = {k: v for k, v in module.state_dict().items() if k != "lin.bias"}
    with pytest.raises(KeyError, match="lin.bias"):
        ct.count_state_parameters(state, module)


def test_t_params_route_b_on_a_dt_payload_equals_the_modules_own_count() -> None:
    payload, model = _dt_payload()
    counts = ct.count_payload_parameters("dt", payload)
    assert counts["trained"] == counts["deployed"] == _module_count(model)


def test_t_params_route_b_on_a_spatial_payload_equals_the_modules_own_count() -> None:
    payload, model = _spatial_payload()
    counts = ct.count_payload_parameters("spatial_dt", payload)
    assert counts["trained"] == counts["deployed"] == _module_count(model)


def test_t_params_route_b_on_a_bc_payload_counts_the_policy() -> None:
    payload, modules = _trunks({"policy": 3})
    counts = ct.count_payload_parameters("bc", payload)
    assert counts["trained"] == counts["deployed"] == _module_count(modules["policy"])


def test_t_params_iql_trains_policy_q_and_v_deploys_the_policy_and_stores_the_target_beside() -> None:
    payload, m = _trunks({"policy": 3, "q": 3, "v": 1, "q_target": 3})
    counts = ct.count_payload_parameters("iql", payload)
    assert counts["trained"] == _module_count(m["policy"], m["q"], m["v"])
    assert counts["deployed"] == _module_count(m["policy"])
    assert counts["stored"] == _module_count(m["policy"], m["q"], m["v"], m["q_target"])
    assert counts["trained"] >= counts["deployed"]


def test_t_params_mappo_trains_actors_and_critic_deploys_the_actors_and_runs_both_per_decision() -> None:
    payload, learner = _mappo_payload()
    counts = ct.count_payload_parameters("mappo", payload)
    actors = _module_count(learner.actors)
    critic = _module_count(learner.critic)
    assert counts["trained"] == actors + critic
    assert counts["deployed"] == actors
    assert counts["executed_per_decision"] == actors + critic
    naive = _all_tensor_elements(payload)
    assert naive > counts["trained"], "the fixture must carry optimiser moments and normalisers to be a test"
    assert counts["trained"] >= counts["deployed"]


# ----------------------------------------------------------------------
# T-params, gated: one real checkpoint per family, route A == route B
# ----------------------------------------------------------------------


def _root(variable: str, default: Path, marker: str) -> Path:
    value = os.environ.get(variable)
    candidate = Path(value) if value else default
    if not (candidate / marker).exists():
        pytest.skip(f"{candidate / marker} not found: set {variable} to the main tree's copy (it is gitignored)")
    return candidate


def _output_root() -> Path:
    return _root("RLTRAFFIC_OUTPUT_ROOT", REPO_ROOT / "output", "p4_dt/dt_seed101.pt")


def _corpus_root() -> Path:
    return _root("RLTRAFFIC_CORPUS_V11", REPO_ROOT / "datasets_v11", "cf_hz1x1__mappo1000__seed101/manifest.json")


REAL_FAMILIES = [
    ("dt", "p4_dt/dt_seed101.pt", 40000, None, {"trained": 647_176, "deployed": 647_176}),
    ("spatial_dt", "p5_2/checkpoints/grid4x4_mappo1000_dt_nomix_h4_seed101.pt", 40000, "dt_nomix_h4",
     {"trained": 848_008, "deployed": 848_008}),
    ("bc", "p4_4/checkpoints/bc_seed101.pt", 40000, "bc", {"trained": 400_520, "deployed": 400_520}),
    ("iql", "p4_4/checkpoints/iql_seed101.pt", 40000, "iql",
     {"trained": 1_200_657, "deployed": 400_520, "stored": 1_601_177}),
    ("mappo", "checkpoints/p2_1_mappo_nominal_500/p2_1_mappo_nominal_1000/cf_grid4x4__mappo__seed101.pt", None, None,
     {"trained": 465_680, "deployed": 364_672, "executed_per_decision": 465_680}),
]


@pytest.mark.parametrize("family,relpath,declared,method,expected", REAL_FAMILIES, ids=[f[0] for f in REAL_FAMILIES])
def test_t_params_route_a_equals_route_b_on_one_real_checkpoint_per_family(
    family: str, relpath: str, declared: int | None, method: str | None, expected: dict[str, int]
) -> None:
    path = _output_root() / relpath
    payload = torch.load(path, map_location="cpu", weights_only=False)
    route_b = ct.count_payload_parameters(family, payload)
    route_a = ct.count_loaded_parameters(family, path, declared_gradient_steps=declared, method=method)
    assert route_a == route_b
    for key, value in expected.items():
        assert route_a[key] == value, key
    assert route_a["trained"] >= route_a["deployed"]


def test_t_params_the_records_parameter_counts_agree_with_route_a() -> None:
    output_root = _output_root()
    record = json.loads((DATA / "p4_4_training.json").read_text())
    by_method = {run["method"]: run["diagnostics"]["parameter_count"] for run in record["runs"] if run["seed"] == 101}
    bc = ct.count_loaded_parameters("bc", output_root / "p4_4/checkpoints/bc_seed101.pt", declared_gradient_steps=40000,
                                    method="bc")
    iql = ct.count_loaded_parameters("iql", output_root / "p4_4/checkpoints/iql_seed101.pt",
                                     declared_gradient_steps=40000, method="iql")
    assert by_method["bc"] == bc["trained"]
    assert by_method["iql"] == iql["trained"]


# ----------------------------------------------------------------------
# json_get and read_pinned
# ----------------------------------------------------------------------


def test_json_get_resolves_dotted_quoted_and_indexed_paths_and_refuses_the_rest() -> None:
    document = {"a": {"b": [10, {"c-d": 7}], "x.y": 3}, "runs": {"k20_seed101": {"loop_seconds": 1.5}}}
    assert ct.json_get(document, "$.a.b[0]") == 10
    assert ct.json_get(document, "$.a.b[1]['c-d']") == 7
    assert ct.json_get(document, "$.a['x.y']") == 3
    assert ct.json_get(document, "$.runs['k20_seed101'].loop_seconds") == 1.5
    with pytest.raises(KeyError, match="nope"):
        ct.json_get(document, "$.a.nope")
    with pytest.raises(IndexError, match="5"):
        ct.json_get(document, "$.a.b[5]")


def test_read_pinned_reads_a_committed_record_at_its_digest() -> None:
    roots = ct.Roots(repo_root=REPO_ROOT, output_root=REPO_ROOT / "output", corpus_root=REPO_ROOT / "datasets_v11")
    document, source = ct.read_pinned("p4_training", roots)
    assert source["file"] == "docs/data/p4_training.json"
    assert source["sha256"] == hashlib.sha256((DATA / "p4_training.json").read_bytes()).hexdigest()
    assert document["reported_gradient_steps"] == 40000


def test_read_pinned_refuses_a_record_at_another_digest(tmp_path: Path) -> None:
    (tmp_path / "docs" / "data").mkdir(parents=True)
    (tmp_path / "docs" / "data" / "p4_training.json").write_bytes((DATA / "p4_training.json").read_bytes() + b" ")
    roots = ct.Roots(repo_root=tmp_path, output_root=tmp_path / "output", corpus_root=tmp_path / "corpus")
    with pytest.raises(ValueError, match="p4_training.json"):
        ct.read_pinned("p4_training", roots)


def test_read_pinned_refuses_an_absent_record(tmp_path: Path) -> None:
    roots = ct.Roots(repo_root=tmp_path, output_root=tmp_path / "output", corpus_root=tmp_path / "corpus")
    with pytest.raises(FileNotFoundError, match="p4_training.json"):
        ct.read_pinned("p4_training", roots)


def test_every_pinned_record_is_the_plans_or_an_addition_declared_with_its_reason() -> None:
    plan = (REPO_ROOT / "docs" / "plans" / "p8.2.md").read_text(encoding="utf-8")
    for key, pinned in ct.PINNED_RECORDS.items():
        if pinned.root == "corpus":
            continue
        if key in ct.SOURCES_ADDED_AFTER_PLAN:
            assert pinned.sha256 not in plan, f"{key} is in the plan; it is not an addition"
            assert len(ct.SOURCES_ADDED_AFTER_PLAN[key]) > 20, f"{key}: the addition needs its reason"
            assert pinned.root == "repo" and pinned.relpath.startswith("docs/data/"), f"{key}: additions are committed"
            continue
        assert pinned.sha256 in plan, f"{key}: {pinned.sha256} is not a digest the approved plan names"
    for key in ct.SOURCES_ADDED_AFTER_PLAN:
        assert key in ct.PINNED_RECORDS, key
    for required in ("p4_training", "p4_4_training", "p4_5_selection", "p4_6_training", "p4_7_training",
                     "p5_3b_nortg", "p5_3c_train", "p7_3b_anchor_training", "p7_3c_finetune",
                     "mappo_results_1000", "mappo_results_500", "mappo_results_060",
                     "p5_1_training_dt_spatial", "p5_2_log_random_dt_nomix"):
        assert required in ct.PINNED_RECORDS, required


# ----------------------------------------------------------------------
# training_block on a synthetic record: the refusals and the declared absence
# ----------------------------------------------------------------------


SYNTHETIC_RUNS = [{"seed": s, "seconds": 10.0 + s / 100, "gradient_steps": 40000} for s in (101, 202, 303, 404, 505)]


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[ct.Roots, ct.TableRow, dict[str, Any]]]:
    (tmp_path / "docs" / "data").mkdir(parents=True)
    document = {
        "runs": SYNTHETIC_RUNS,
        "batch": 64,
        "rows": 7200,
        "device": "cuda",
        "runtime": {"cuda_device_name": "GPU", "torch_version": "2.11.0", "torch_num_threads": 1},
    }
    path = tmp_path / "docs" / "data" / "synthetic_training.json"
    path.write_text(json.dumps(document))
    monkeypatch.setitem(
        ct.PINNED_RECORDS,
        "synthetic",
        ct.PinnedRecord(key="synthetic", root="repo", relpath="docs/data/synthetic_training.json",
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), named_by="this test"),
    )
    roots = ct.Roots(repo_root=tmp_path, output_root=tmp_path / "output", corpus_root=tmp_path / "corpus")
    row = ct.TableRow(row_id="synthetic.row", scenario="hz1x1", method="bc", configuration="synthetic", family="bc",
                      claims=(), groups=(), training=(), interactions="offline", latency_row=None)
    entry = {
        "record": "synthetic",
        "tier": "t",
        "label": "synthetic",
        "seeds": {"kind": "list", "path": "$.runs", "match": {}},
        "seconds": {"kind": "seed", "field": "seconds"},
        "steps": {"kind": "seed", "field": "gradient_steps"},
        "batch": {"kind": "record", "path": "$.batch"},
        "data": {"kind": "record", "path": "$.rows", "unit": "windows"},
        "covers": "the gradient loop only",
        "regime": {
            "device": {"kind": "record", "path": "$.device"},
            "gpu": {"kind": "record", "path": "$.runtime.cuda_device_name"},
            "torch": {"kind": "record", "path": "$.runtime.torch_version"},
            "threads": {"kind": "record", "path": "$.runtime.torch_num_threads"},
            "concurrency": {"kind": "absent", "key": "synthetic.concurrency"},
        },
    }
    monkeypatch.setitem(ct.DECLARED_ABSENCES, "synthetic.concurrency", "the synthetic record does not record it")
    yield roots, row, entry


def test_training_block_reads_every_seed_with_its_json_path_and_reduces_by_median(synthetic: Any) -> None:
    roots, row, entry = synthetic
    block = ct.training_block(row, entry, roots)
    seconds = block["seconds"]
    assert [item["seed"] for item in seconds["per_seed"]] == [101, 202, 303, 404, 505]
    assert [item["source"]["json_path"] for item in seconds["per_seed"]] == [f"$.runs[{i}].seconds" for i in range(5)]
    values = [item["value"] for item in seconds["per_seed"]]
    assert values == [run["seconds"] for run in SYNTHETIC_RUNS]
    assert seconds["median"] == statistics.median(values)
    assert seconds["min"] == min(values) and seconds["max"] == max(values)
    assert block["batch"]["value"] == 64 and block["batch"]["source"]["json_path"] == "$.batch"
    assert block["data"] == {"value": 7200, "unit": "windows", "source": block["data"]["source"]}
    assert block["regime"]["concurrency"] == {"value": None, "reason": "the synthetic record does not record it"}
    assert block["covers"] == "the gradient loop only"


def test_training_block_refuses_a_value_missing_from_its_record_that_is_not_a_declared_absence(
    synthetic: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots, row, entry = synthetic
    runs = [dict(run) for run in SYNTHETIC_RUNS]
    del runs[2]["seconds"]
    path = tmp_path / "docs" / "data" / "synthetic_training.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "runs": runs}))
    monkeypatch.setitem(ct.PINNED_RECORDS, "synthetic",
                        dataclasses.replace(ct.PINNED_RECORDS["synthetic"],
                                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with pytest.raises(ValueError, match="not a declared absence"):
        ct.training_block(row, entry, roots)


def test_training_block_refuses_an_absence_whose_key_is_not_declared(synthetic: Any) -> None:
    roots, row, entry = synthetic
    entry = {**entry, "seconds": {"kind": "absent", "key": "nobody.declared.this"}}
    with pytest.raises(ValueError, match="nobody.declared.this"):
        ct.training_block(row, entry, roots)


def test_training_block_refuses_a_field_with_no_source(synthetic: Any) -> None:
    roots, row, entry = synthetic
    entry = {key: value for key, value in entry.items() if key != "seconds"}
    with pytest.raises(ValueError, match="no source"):
        ct.training_block(row, entry, roots)


def test_training_block_writes_a_declared_absence_as_null_with_its_reason(
    synthetic: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots, row, entry = synthetic
    monkeypatch.setitem(ct.DECLARED_ABSENCES, "synthetic.seconds", "no record holds it")
    block = ct.training_block(row, {**entry, "seconds": {"kind": "absent", "key": "synthetic.seconds"}}, roots)
    assert block["seconds"] == {"value": None, "reason": "no record holds it"}


# ----------------------------------------------------------------------
# The latency run: a synthetic run and its refusals
# ----------------------------------------------------------------------


def _ns(offset: int = 0) -> list[list[int]]:
    return [[1_000_000 + 1_000 * k + offset for k in range(cl.DECISIONS_PER_EPISODE)] for _ in cl.TIMING_DRAWS]


def _write_run(tmp_path: Path, *, rows: tuple[cl.LatencyRow, ...] | None = None, close_seconds: float = 0.8,
               reproduced: bool = True, complete: bool = True) -> tuple[Path, Path]:
    rows = cl.ROWS if rows is None else rows
    latency_root = tmp_path / "latency"
    run_dir = latency_root / "20261003T000000Z"
    run_dir.mkdir(parents=True)
    git = {"commit": "0" * 40, "dirty": False}
    regime = {"torch_num_threads": 1, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": None}
    machine = {"cpu_model": "synthetic", "logical_cpus": 16, "gpu": "synthetic", "driver": "0"}
    outcomes = [{"label": "canary_open", "status": "ok", "attempts": []}]
    for row in rows:
        for device in row.devices:
            record = cl.build_record(row, device, _ns(), draws=cl.TIMING_DRAWS, prompt=None, factory="synthetic",
                                     regime=regime, machine=machine, load_before=[0.0, 0.0, 0.0],
                                     load_after=[0.0, 0.0, 0.0], git=git)
            cl.write_once(run_dir / f"{row.row_id}_{device}.json", record)
            outcomes.append({"label": f"{row.row_id}_{device}", "status": "ok", "attempts": []})
    cl.write_once(run_dir / "canary_open.json", cl.build_canary_record("open", 0.8, reproduced=True, git=git))
    cl.write_once(run_dir / "canary_close.json",
                  cl.build_canary_record("close", close_seconds, reproduced=reproduced, git=git))
    outcomes.append({"label": "canary_close", "status": "ok", "attempts": []})
    status = "COMPLETE" if complete else "FAILED"
    cl.write_once(run_dir / "run.json", cl.build_run_record("20261003T000000Z", rows, outcomes, status=status,
                                                            reason=None if complete else "synthetic", git=git,
                                                            timeouts={"hz1x1": 120.0, "grid4x4": 300.0}))
    (run_dir / status).write_text("")
    manifest = tmp_path / "SHA256SUMS_p8_2.txt"
    cl.write_manifest(latency_root, manifest)
    return run_dir, manifest


def test_verify_latency_run_accepts_a_complete_run_and_recomputes_every_statistic(tmp_path: Path) -> None:
    rows = tuple(cl.row_by_id(r) for r in ("hz1x1.bc", "hz1x1.maxpressure"))
    run_dir, manifest = _write_run(tmp_path, rows=rows)
    summary = ct.verify_latency_run(run_dir, manifest)
    assert set(summary["records"]) == {("hz1x1.bc", "cpu"), ("hz1x1.bc", "cuda"), ("hz1x1.maxpressure", "cpu")}
    assert summary["canaries"]["open"]["verdict"] == "at speed"


def test_verify_latency_run_refuses_a_run_without_complete(tmp_path: Path) -> None:
    run_dir, manifest = _write_run(tmp_path, rows=(cl.row_by_id("hz1x1.bc"),), complete=False)
    with pytest.raises(ValueError, match="COMPLETE"):
        ct.verify_latency_run(run_dir, manifest)


def test_verify_latency_run_refuses_a_throttled_closing_canary(tmp_path: Path) -> None:
    run_dir, manifest = _write_run(tmp_path, rows=(cl.row_by_id("hz1x1.bc"),), close_seconds=2.3)
    with pytest.raises(ValueError, match="throttled"):
        ct.verify_latency_run(run_dir, manifest)


def test_verify_latency_run_refuses_a_canary_that_did_not_reproduce(tmp_path: Path) -> None:
    run_dir, manifest = _write_run(tmp_path, rows=(cl.row_by_id("hz1x1.bc"),), reproduced=False)
    with pytest.raises(ValueError, match="reproduce"):
        ct.verify_latency_run(run_dir, manifest)


def test_verify_latency_run_refuses_a_file_outside_its_manifest(tmp_path: Path) -> None:
    run_dir, manifest = _write_run(tmp_path, rows=(cl.row_by_id("hz1x1.bc"),))
    (run_dir / "stray.json").write_text("{}")
    with pytest.raises(ValueError, match="stray.json"):
        ct.verify_latency_run(run_dir, manifest)


def test_verify_latency_run_refuses_a_record_at_another_digest(tmp_path: Path) -> None:
    run_dir, manifest = _write_run(tmp_path, rows=(cl.row_by_id("hz1x1.bc"),))
    path = run_dir / "hz1x1.bc_cpu.json"
    path.chmod(0o644)
    path.write_text(path.read_text() + " ")  # still valid JSON; one byte more, so another digest
    with pytest.raises(ValueError, match="hz1x1.bc_cpu.json"):
        ct.verify_latency_run(run_dir, manifest)


def test_verify_latency_run_refuses_statistics_that_do_not_recompute(tmp_path: Path) -> None:
    run_dir, manifest = _write_run(tmp_path, rows=(cl.row_by_id("hz1x1.bc"),))
    path = run_dir / "hz1x1.bc_cpu.json"
    record = json.loads(path.read_text())
    record["p95_ns"] += 1
    path.unlink()
    path.write_text(json.dumps(record))
    manifest.unlink()
    cl.write_manifest(run_dir.parent, manifest)
    with pytest.raises(ValueError, match="p95"):
        ct.verify_latency_run(run_dir, manifest)


# ----------------------------------------------------------------------
# The table's rows
# ----------------------------------------------------------------------


def test_every_latency_row_is_one_table_row_and_the_two_zero_shot_rows_reuse_the_dt() -> None:
    table = {row.row_id: row for row in ct.TABLE_ROWS}
    assert len(table) == len(ct.TABLE_ROWS) == 44
    assert len(cl.ROWS) == 42
    for latency_row in cl.ROWS:
        assert table[latency_row.row_id].latency_row == latency_row.row_id
    for row_id in ("hz1x1.c3.zero_shot_mappo1000", "hz1x1.c3.zero_shot_mix50"):
        assert table[row_id].latency_row is None
        assert "hz1x1.dt_k20" in table[row_id].latency_note
    families = {"dt", "spatial_dt", "bc", "iql", "mappo", "heuristic"}
    for row in ct.TABLE_ROWS:
        assert row.family in families, row.row_id
        assert row.claims, f"{row.row_id} names no committed result"
        assert row.interactions in {"offline", "mappo", "none"}, row.row_id


# ----------------------------------------------------------------------
# T-sources, gated: the artifact against the records, read here
# ----------------------------------------------------------------------


def _resolve(document: Any, path: str) -> Any:
    """This test's own JSON-path reader, independent of ``ct.json_get``."""
    assert path.startswith("$"), path
    for name, quoted, index in re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)|\['([^']*)'\]|\[(\d+)\]", path[1:]):
        document = document[int(index)] if index else document[name or quoted]
    return document


def _sourced_values(node: Any, trail: str = "$") -> Iterator[tuple[str, Any, dict[str, Any]]]:
    if isinstance(node, dict):
        source = node.get("source")
        if "value" in node and isinstance(source, dict) and ("json_path" in source or "line" in source):
            yield trail, node["value"], source
        for key, value in node.items():
            yield from _sourced_values(value, f"{trail}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _sourced_values(value, f"{trail}[{index}]")


@pytest.fixture(scope="module")
def built_artifact(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], ct.Roots]:
    output_root = _output_root()
    corpus_root = _corpus_root()
    run_dir, manifest = _write_run(tmp_path_factory.mktemp("latency_run"))
    roots = ct.Roots(repo_root=REPO_ROOT, output_root=output_root, corpus_root=corpus_root, latency_dir=run_dir,
                     manifest_path=manifest)
    return ct.build_artifact(roots, git={"commit": "0" * 40, "dirty": False}), roots


def _open_source(source: dict[str, Any], roots: ct.Roots, cache: dict[str, Any]) -> tuple[Path, Any]:
    file = source["file"]
    path = REPO_ROOT / file if file.startswith("docs/") else roots.output_root.parent / file
    if file not in cache:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"], file
        if path.suffix == ".pt":
            cache[file] = torch.load(path, map_location="cpu", weights_only=False)
        elif path.suffix == ".json":
            cache[file] = json.loads(path.read_text())
        else:
            cache[file] = path.read_text().splitlines()
    return path, cache[file]


def test_t_sources_every_sourced_number_equals_the_records_value_at_its_named_path(built_artifact: Any) -> None:
    artifact, roots = built_artifact
    cache: dict[str, Any] = {}
    checked = 0
    for trail, value, source in _sourced_values(artifact["rows"]):
        _, document = _open_source(source, roots, cache)
        if "line" in source:
            line = document[int(source["line"]) - 1]
            match = re.search(r"done in ([0-9.]+)s", line)
            assert match and float(match.group(1)) == value, trail
        else:
            assert _resolve(document, source["json_path"]) == value, trail
        checked += 1
    assert checked > 500, f"only {checked} sourced numbers: the walk is not reaching the training blocks"


def test_t_sources_every_median_and_range_recomputes_from_its_per_seed_values(built_artifact: Any) -> None:
    artifact, _ = built_artifact
    blocks = 0
    for row in artifact["rows"]:
        for entry in row["training"]:
            seconds = entry["seconds"]
            if seconds.get("value", "present") is None:
                assert seconds["reason"], row["id"]
                continue
            values = [item["value"] for item in seconds["per_seed"]]
            assert len(values) == 5, (row["id"], entry["tier"])
            assert seconds["median"] == statistics.median(values)
            assert (seconds["min"], seconds["max"]) == (min(values), max(values))
            blocks += 1
    assert blocks > 50


def test_t_sources_the_parameter_counts_are_the_plans_and_the_interactions_are_mappos_own(built_artifact: Any) -> None:
    artifact, roots = built_artifact
    rows = {row["id"]: row for row in artifact["rows"]}
    assert rows["hz1x1.dt_k20"]["parameters"]["trained"]["value"] == 647_176
    for k in ("k1", "k2", "k5", "k10", "k20", "k1_b1280", "k2_b640"):
        assert rows[f"hz1x1.h4.{k}"]["parameters"]["trained"]["value"] == 647_176, k
    assert rows["hz1x1.iql"]["parameters"]["deployed"]["value"] == 400_520
    assert rows["hz1x1.iql"]["parameters"]["stored"]["value"] == 1_601_177
    assert rows["grid4x4.mappo1000"]["parameters"]["executed_per_decision"]["value"] == 465_680
    assert rows["hz1x1.maxpressure"]["parameters"]["trained"]["value"] == 0
    joint = rows["grid4x4.mappo1000"]["environment_interactions"]["joint_decisions_per_seed"]
    assert joint["value"] == 360_000
    assert rows["grid4x4.mappo1000"]["environment_interactions"]["intersection_decisions_per_seed"]["value"] == 5_760_000
    for checkpoint in rows["grid4x4.mappo1000"]["checkpoints"]:
        payload = torch.load(roots.output_root.parent / checkpoint["path"], map_location="cpu", weights_only=False)
        assert payload["steps_done"] == joint["value"]
    assert rows["hz1x1.bc"]["environment_interactions"]["joint_decisions_per_seed"]["value"] == 0


def test_t_sources_the_declared_absences_carry_their_reasons(built_artifact: Any) -> None:
    artifact, _ = built_artifact
    rows = {row["id"]: row for row in artifact["rows"]}
    for row_id in ("grid4x4.bc", "grid4x4.bc_top10", "grid4x4.bc_top10_perix", "grid4x4.iql"):
        assert len(rows[row_id]["training"]) == 4, row_id  # four grid4x4 tiers each
        for entry in rows[row_id]["training"]:
            assert entry["seconds"]["value"] is None, (row_id, entry["tier"])
            assert "P5.2" in entry["seconds"]["reason"] or "P5.1" in entry["seconds"]["reason"]
    random_tier = [e for e in rows["grid4x4.dt_nomix"]["training"] if e["tier"] == "random"]
    assert len(random_tier) == 1 and all("line" in item["source"] for item in random_tier[0]["seconds"]["per_seed"])


def test_t_sources_the_artifact_says_what_it_does_not_say(built_artifact: Any) -> None:
    artifact, _ = built_artifact
    text = " ".join(artifact["what_this_does_not_say"])
    assert "context length" in text and "not the model's size" in text
    assert artifact["format_version"] == ct.FORMAT_VERSION
    assert artifact["mappo_results_pin"]["pinned_on"] == ct.MAPPO_RESULTS_PINNED_ON
