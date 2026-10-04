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
* **Amendment B** (written red first in C4): B4.1 the builder's OWN statistics and every published millisecond
  derived from the verified nanoseconds (CM4); B4.2 one refusal each, never a skip (CM11 among them); B4.3 the registry
  joined to the table row, route A hashing what it loads; B4.4 both devices' regimes, an atomic artifact, a null only
  where an absence declares it, no model row without training, no empty outlier note, a true docstring; B6 the text
  the table would print; B7 what it does not say.

GATES: the real-record tests skip naming ``RLTRAFFIC_OUTPUT_ROOT`` / ``RLTRAFFIC_CORPUS_V11`` (gitignored, main tree).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import re
import statistics
from pathlib import Path
from typing import Any, Callable, Iterator

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


def _with_slow_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = [dict(run) for run in SYNTHETIC_RUNS]
    runs[4]["seconds"] = 100.0  # about 9x the median of ~13 s
    path = tmp_path / "docs" / "data" / "synthetic_training.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "runs": runs}))
    monkeypatch.setitem(ct.PINNED_RECORDS, "synthetic",
                        dataclasses.replace(ct.PINNED_RECORDS["synthetic"],
                                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def test_training_block_refuses_a_wall_time_outlier_that_is_not_declared(
    synthetic: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots, row, entry = synthetic
    _with_slow_seed(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="505"):
        ct.training_block(row, entry, roots)


def test_training_block_lists_a_declared_wall_time_outlier_with_its_note(
    synthetic: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots, row, entry = synthetic
    _with_slow_seed(tmp_path, monkeypatch)
    block = ct.training_block(row, {**entry, "outliers": {505: "a stall with a named cause"}}, roots)
    assert block["seconds"]["outliers"] == [{"seed": 505, "value": 100.0, "note": "a stall with a named cause"}]
    assert block["seconds"]["max"] == 100.0 and block["seconds"]["median"] == statistics.median(
        [run["seconds"] for run in SYNTHETIC_RUNS[:4]] + [100.0])


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


#: A power block of the shape ``offline.compute_latency.power_block`` writes, reading mains + Best Performance
#: (Amendment B, B1). Added to the synthetic run's machine block in C4: B4.2 makes the builder refuse a record outside
#: this regime, so a synthetic run must carry one to stand for a run the builder accepts.
_MAINS_BEST: dict[str, Any] = {
    "supplies": {"read_with": "/sys/class/power_supply/<name>/{type,online}", "error": None,
                 "items": [{"name": "AC1", "type": "Mains", "online": 1},
                           {"name": "BAT1", "type": "Battery", "online": None}]},
    "windows": {"read_with": r"reg.exe query HKLM\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes",
                "error": None, "active_scheme": "381b4222-f694-41f0-9685-ff5bb260df2e",
                "ac_overlay": "ded574b5-45a0-4f42-8737-46345c09c238", "ac_overlay_name": "Best Performance",
                "dc_overlay": "961cc777-2547-4f9d-8174-7d86181b8a7a", "dc_overlay_name": "Better Battery"},
}


def _write_run(tmp_path: Path, *, rows: tuple[cl.LatencyRow, ...] | None = None, close_seconds: float = 0.8,
               reproduced: bool = True, complete: bool = True) -> tuple[Path, Path]:
    rows = cl.ROWS if rows is None else rows
    latency_root = tmp_path / "latency"
    run_dir = latency_root / "20261003T000000Z"
    run_dir.mkdir(parents=True)
    git = {"commit": "0" * 40, "dirty": False}
    regime = {"torch_num_threads": 1, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": None}
    machine = {"cpu_model": "synthetic", "logical_cpus": 16, "gpu": "synthetic", "driver": "0",
               "kernel_release": "synthetic", "power": _MAINS_BEST}
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


# ======================================================================
# Amendment B (gate G1, FIX FIRST), written red first in C4: B4 the builder's latency verification, B6 the text the
# table would print, B7 what it does not say.
# ======================================================================


def _relist(run_dir: Path, manifest: Path) -> None:
    """Rewrite a synthetic run's manifest over what its directory now holds."""
    manifest.unlink()
    cl.write_manifest(run_dir.parent, manifest)


def _tamper(run_dir: Path, manifest: Path, name: str, change: Callable[[dict[str, Any]], Any]) -> None:
    """Rewrite one record of a synthetic run, and its manifest, as a run that wrote it so would have."""
    path = run_dir / name
    record = json.loads(path.read_text())
    change(record)
    path.unlink()
    path.write_text(json.dumps(record))
    _relist(run_dir, manifest)


def _bc_run(tmp_path: Path) -> tuple[Path, Path]:
    return _write_run(tmp_path, rows=(cl.row_by_id("hz1x1.bc"),))


# ----------------------------------------------------------------------
# B4.1: the builder's own route, and the published milliseconds (CM4)
# ----------------------------------------------------------------------


def test_b4_the_builders_own_route_takes_the_median_and_the_nearest_rank_p95_from_sorted_values() -> None:
    generator = random.Random(20261004)
    episodes = [[generator.randrange(500_000, 5_000_000) for _ in range(cl.DECISIONS_PER_EPISODE)] for _ in range(3)]
    figures = ct.latency_figures(episodes, warmup=cl.WARMUP)
    timed = sorted(value for episode in episodes for value in episode[cl.WARMUP:])
    n = len(timed)
    assert figures["n_timed"] == n == 1020
    assert figures["median_ns"] == (timed[n // 2 - 1] + timed[n // 2]) / 2
    assert figures["p95_ns"] == timed[(95 * n + 99) // 100 - 1]
    harness = cl.latency_stats(episodes, warmup=cl.WARMUP)  # numpy's route, written once in the harness
    assert (figures["median_ns"], figures["p95_ns"]) == (harness["median_ns"], harness["p95_ns"])
    odd = ct.latency_figures([[5, 1, 9, 3, 7]], warmup=0)
    assert (odd["n_timed"], odd["median_ns"], odd["p95_ns"]) == (5, 5, 9)


def test_b4_verify_latency_run_never_calls_the_harness_statistics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, manifest = _bc_run(tmp_path)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the builder recomputed through offline.compute_latency.latency_stats")

    monkeypatch.setattr(cl, "latency_stats", forbidden)
    summary = ct.verify_latency_run(run_dir, manifest)
    assert set(summary["records"]) == {("hz1x1.bc", "cpu"), ("hz1x1.bc", "cuda")}


def test_b4_the_published_milliseconds_are_the_verified_nanoseconds_divided_here(tmp_path: Path) -> None:
    rows = (cl.row_by_id("hz1x1.bc"), cl.row_by_id("grid4x4.bc"))
    run_dir, manifest = _write_run(tmp_path, rows=rows)
    run = ct.verify_latency_run(run_dir, manifest)
    table = {row.row_id: row for row in ct.TABLE_ROWS}
    checked = 0
    for latency_row in rows:
        out = ct.inference_block(table[latency_row.row_id], run)
        n_ix = 16 if latency_row.scenario == "grid4x4" else 1
        for device in ("cpu", "cuda"):
            record = json.loads((run_dir / f"{latency_row.row_id}_{device}.json").read_text())
            timed = sorted(v for episode in record["episodes"] for v in episode["decision_ns"][cl.WARMUP:])
            n = len(timed)
            median_ms = ((timed[n // 2 - 1] + timed[n // 2]) / 2) / 1e6
            p95_ms = timed[(95 * n + 99) // 100 - 1] / 1e6
            cell = out[device]
            assert cell["median_ms"]["value"] == median_ms, (latency_row.row_id, device)
            assert cell["p95_ms"]["value"] == p95_ms, (latency_row.row_id, device)
            assert cell["per_intersection_median_ms"]["value"] == median_ms / n_ix
            assert cell["per_intersection_p95_ms"]["value"] == p95_ms / n_ix
            assert cell["n_timed"] == n == 1020
            checked += 1
    assert checked == 4


_PUBLISHED_FIELDS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "median_ms": lambda r: r.__setitem__("median_ms", r["median_ms"] + 1e-6),
    "p95_ms": lambda r: r.__setitem__("p95_ms", r["p95_ms"] * 1000),
    "per_intersection.median_ms": lambda r: r["per_intersection"].__setitem__("median_ms", 0.5),
    "per_intersection.p95_ms": lambda r: r["per_intersection"].__setitem__("p95_ms", 0.5),
    "n_intersections": lambda r: r.__setitem__("n_intersections", 16),
}


@pytest.mark.parametrize("field", sorted(_PUBLISHED_FIELDS))
def test_b4_a_record_whose_published_figures_disagree_with_its_nanoseconds_is_refused(tmp_path: Path, field: str) -> None:
    run_dir, manifest = _bc_run(tmp_path)
    _tamper(run_dir, manifest, "hz1x1.bc_cpu.json", _PUBLISHED_FIELDS[field])
    with pytest.raises(ValueError, match=re.escape(field)):
        ct.verify_latency_run(run_dir, manifest)


# ----------------------------------------------------------------------
# B4.2: refuse, never skip -- one test each
# ----------------------------------------------------------------------


def test_b4_a_record_of_another_format_version_is_refused_not_skipped(tmp_path: Path) -> None:
    run_dir, manifest = _bc_run(tmp_path)
    _tamper(run_dir, manifest, "hz1x1.bc_cpu.json", lambda r: r.__setitem__("format_version", "p8.2-latency/0.9"))
    with pytest.raises(ValueError, match="p8.2-latency/0.9"):
        ct.verify_latency_run(run_dir, manifest)


def test_b4_a_cell_the_run_declares_without_its_record_is_refused(tmp_path: Path) -> None:
    run_dir, manifest = _bc_run(tmp_path)
    (run_dir / "hz1x1.bc_cuda.json").unlink()
    _relist(run_dir, manifest)
    with pytest.raises(ValueError, match="hz1x1.bc_cuda"):
        ct.verify_latency_run(run_dir, manifest)


def test_b4_a_run_that_does_not_cover_every_cell_of_the_registry_is_refused(tmp_path: Path) -> None:
    run_dir, manifest = _bc_run(tmp_path)
    with pytest.raises(ValueError, match="76 of the registry's 78"):
        ct.check_registry_coverage(ct.verify_latency_run(run_dir, manifest))
    full_dir, full_manifest = _write_run(tmp_path / "full")
    ct.check_registry_coverage(ct.verify_latency_run(full_dir, full_manifest))


def test_b4_a_record_file_for_no_registry_row_is_refused(tmp_path: Path) -> None:
    run_dir, manifest = _bc_run(tmp_path)
    record = json.loads((run_dir / "hz1x1.bc_cpu.json").read_text())
    record["row"] = "hz1x1.no_such_row"
    (run_dir / "hz1x1.no_such_row_cpu.json").write_text(json.dumps(record))
    _relist(run_dir, manifest)
    with pytest.raises(ValueError, match="hz1x1.no_such_row"):
        ct.verify_latency_run(run_dir, manifest)


@pytest.mark.parametrize("field,value", [("row", "hz1x1.iql"), ("device", "cuda")])
def test_b4_a_record_whose_row_or_device_disagrees_with_its_file_name_is_refused(
    tmp_path: Path, field: str, value: str
) -> None:
    run_dir, manifest = _bc_run(tmp_path)
    _tamper(run_dir, manifest, "hz1x1.bc_cpu.json", lambda r: r.__setitem__(field, value))
    with pytest.raises(ValueError, match="hz1x1.bc_cpu.json"):
        ct.verify_latency_run(run_dir, manifest)


@pytest.mark.parametrize("field", ["sha256", "path"])
def test_b4_a_record_of_a_checkpoint_other_than_the_registered_one_is_refused(tmp_path: Path, field: str) -> None:
    run_dir, manifest = _bc_run(tmp_path)
    other = {"sha256": "0" * 64, "path": "output/p4_4/checkpoints/bc_seed202.pt"}[field]
    _tamper(run_dir, manifest, "hz1x1.bc_cpu.json", lambda r: r["checkpoint"].__setitem__(field, other))
    with pytest.raises(ValueError, match="registered"):
        ct.verify_latency_run(run_dir, manifest)


_PROTOCOL: dict[str, Callable[[dict[str, Any]], Any]] = {
    "warmup": lambda r: r.__setitem__("warmup", 0),
    "draws": lambda r: (r["draws"].__setitem__(2, 1003), r["episodes"][2].__setitem__("draw", 1003)),
    "episodes": lambda r: (r["draws"].pop(), r["episodes"].pop()),
    "engine_seed": lambda r: r.__setitem__("engine_seed", 999),
}


#: Each protocol refusal's OWN message. Strengthened after mutant NB42h (the episode-count check removed) survived the
#: first form of this test, whose match was the bare field name: the draws refusal's message also says "episodes",
#: so the record was still refused, under the wrong cause.
_PROTOCOL_REFUSALS: dict[str, str] = {
    "warmup": "warmup 0 is not the registered 20",
    "draws": r"draws \[1000, 1001, 1003\]",
    "episodes": "2 episodes, not the registered 3",
    "engine_seed": "engine_seed 999 is not the registered 1000",
}


@pytest.mark.parametrize("field", sorted(_PROTOCOL))
def test_b4_a_record_outside_the_registered_protocol_is_refused(tmp_path: Path, field: str) -> None:
    run_dir, manifest = _bc_run(tmp_path)
    _tamper(run_dir, manifest, "hz1x1.bc_cpu.json", _PROTOCOL[field])
    with pytest.raises(ValueError, match=_PROTOCOL_REFUSALS[field]):
        ct.verify_latency_run(run_dir, manifest)


@pytest.mark.parametrize("case", ["another commit", "a dirty tree"])
def test_b4_a_record_of_another_commit_or_a_dirty_tree_is_refused(tmp_path: Path, case: str) -> None:
    run_dir, manifest = _bc_run(tmp_path)
    key, value, named = ("commit", "1" * 40, "commit") if case == "another commit" else ("dirty", True, "dirty")
    _tamper(run_dir, manifest, "hz1x1.bc_cpu.json", lambda r: r["git"].__setitem__(key, value))
    with pytest.raises(ValueError, match=named):
        ct.verify_latency_run(run_dir, manifest)


@pytest.mark.parametrize("case", ["better battery", "on battery"])
def test_b4_a_record_outside_the_power_regime_is_refused(tmp_path: Path, case: str) -> None:
    run_dir, manifest = _bc_run(tmp_path)

    def change(record: dict[str, Any]) -> None:
        power = record["machine"]["power"]
        if case == "better battery":
            power["windows"].update(ac_overlay="961cc777-2547-4f9d-8174-7d86181b8a7a", ac_overlay_name="Better Battery")
        else:
            power["supplies"]["items"][0]["online"] = 0

    _tamper(run_dir, manifest, "hz1x1.bc_cpu.json", change)
    with pytest.raises(ValueError, match="power regime"):
        ct.verify_latency_run(run_dir, manifest)


# ----------------------------------------------------------------------
# B4.3: the registry joined to the table row; route A hashes what it loads
# ----------------------------------------------------------------------


def test_b4_a_registry_whose_h4_k1_names_the_k20_file_refuses() -> None:
    row = {r.row_id: r for r in ct.TABLE_ROWS}["hz1x1.h4.k1"]
    k1, k20 = cl.row_by_id("hz1x1.h4.k1"), cl.row_by_id("hz1x1.h4.k20")
    checkpoints = [{"tier": group["tier"], "seed": seed, "path": group["path"].format(seed=seed),
                    "sha256": f"{group['tier']}-{seed}"} for group in row.groups for seed in (101, 202, 303, 404, 505)]
    representative = next(c for c in checkpoints if c["tier"] == "mappo1000" and c["seed"] == 101)
    representative["sha256"] = k1.sha256
    assert representative["path"] == k1.checkpoint
    assert ct.join_registry_row(row, checkpoints, k1) == representative
    for swapped in (dataclasses.replace(k1, checkpoint=k20.checkpoint, sha256=k20.sha256),
                    dataclasses.replace(k1, sha256=k20.sha256),
                    dataclasses.replace(k1, checkpoint=k20.checkpoint)):
        with pytest.raises(ValueError, match="hz1x1.h4.k1"):
            ct.join_registry_row(row, checkpoints, swapped)


def test_b4_route_a_hashes_the_file_it_loads_before_loading_it(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    path.write_bytes(b"not the registered checkpoint")
    with pytest.raises(ValueError, match="sha256"):
        ct.count_loaded_parameters("bc", path, declared_gradient_steps=40000, expected_sha256="0" * 64)


# ----------------------------------------------------------------------
# B4.4: both regimes, an atomic artifact, nulls, model rows without training, outlier notes, the docstring
# ----------------------------------------------------------------------


def test_b4_the_hardware_block_reports_both_devices_regimes() -> None:
    machine = {"cpu_model": "synthetic", "power": _MAINS_BEST}
    cpu = {"torch_num_threads": 1, "device": "cpu", "synchronize": None}
    cuda = {"torch_num_threads": 1, "device": "cuda", "synchronize": "torch.cuda.synchronize() before both clock readings"}
    run: dict[str, Any] = {"records": {("a", "cpu"): {"record": {"machine": machine, "regime": cpu}},
                                       ("a", "cuda"): {"record": {"machine": machine, "regime": cuda}},
                                       ("b", "cpu"): {"record": {"machine": machine, "regime": cpu}}}}
    hardware = ct._hardware(run)
    assert hardware["machine"] == machine
    assert hardware["regime"] == {"cpu": cpu, "cuda": cuda}
    run["records"][("b", "cpu")] = {"record": {"machine": machine, "regime": {**cpu, "torch_num_threads": 2}}}
    with pytest.raises(ValueError, match="cpu"):
        ct._hardware(run)


def test_b4_write_artifact_is_exclusive_and_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "artifacts" / "p8_2_compute.json"

    def refuse_link(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("the link was refused (simulated)")

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", refuse_link)
        with pytest.raises(OSError, match="simulated"):
            ct.write_artifact(path, {"format_version": ct.FORMAT_VERSION})
    assert not path.exists()
    assert not path.parent.exists() or list(path.parent.iterdir()) == []
    ct.write_artifact(path, {"format_version": ct.FORMAT_VERSION, "n": 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="written once"):
        ct.write_artifact(path, {"n": 2})
    assert path.read_bytes() == before


def _rewrite_synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document: dict[str, Any]) -> None:
    path = tmp_path / "docs" / "data" / "synthetic_training.json"
    path.write_text(json.dumps(document))
    monkeypatch.setitem(ct.PINNED_RECORDS, "synthetic",
                        dataclasses.replace(ct.PINNED_RECORDS["synthetic"],
                                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def _synthetic_document(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "docs" / "data" / "synthetic_training.json").read_text())


@pytest.mark.parametrize("field", ["batch", "steps", "data", "regime.device"])
def test_b4_a_null_read_from_a_record_is_refused_unless_an_absence_declares_it(
    synthetic: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    roots, row, entry = synthetic
    document = _synthetic_document(tmp_path)
    if field == "steps":
        document["runs"][3]["gradient_steps"] = None
    else:
        document[{"batch": "batch", "data": "rows", "regime.device": "device"}[field]] = None
    _rewrite_synthetic(tmp_path, monkeypatch, document)
    with pytest.raises(ValueError, match="null"):
        ct.training_block(row, entry, roots)


def test_b4_a_null_that_an_absence_declares_carries_its_reason(
    synthetic: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots, row, entry = synthetic
    document = _synthetic_document(tmp_path)
    document["batch"] = None
    _rewrite_synthetic(tmp_path, monkeypatch, document)
    monkeypatch.setitem(ct.DECLARED_ABSENCES, "synthetic.batch", "the synthetic record holds null for its batch")
    declared = {**entry, "batch": {"kind": "record", "path": "$.batch", "null_means": "synthetic.batch"}}
    batch = ct.training_block(row, declared, roots)["batch"]
    assert batch["value"] is None and batch["reason"] == "the synthetic record holds null for its batch"
    assert batch["source"]["json_path"] == "$.batch"


def test_b4_a_model_row_without_training_entries_refuses(synthetic: Any) -> None:
    roots, row, _entry = synthetic  # family "bc", no training entry
    with pytest.raises(ValueError, match="no training entry"):
        ct.training_blocks(row, roots)
    assert ct.training_blocks(dataclasses.replace(row, family="heuristic"), roots) == []


@pytest.mark.parametrize("note", ["", "   "])
def test_b4_an_outlier_note_must_say_something(
    synthetic: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, note: str
) -> None:
    roots, row, entry = synthetic
    _with_slow_seed(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="note"):
        ct.training_block(row, {**entry, "outliers": {505: note}}, roots)


def test_b4_the_docstring_says_which_numbers_are_derived_and_sit_beside_their_sources() -> None:
    doc = ct.__doc__
    assert "**Every number is ``{value, source}``**" not in doc
    assert "derived" in doc and "beside the sourced per-seed values" in doc


# ----------------------------------------------------------------------
# B6: the text the table would print
# ----------------------------------------------------------------------


def test_b6_the_p4_suspend_note_quotes_its_record() -> None:
    seeds = {int(s["seed"]): float(s["seconds"]) for s in json.loads((DATA / "p4_training.json").read_text())["seeds"]}
    others = [value for seed, value in seeds.items() if seed != 505]
    assert f"{seeds[505]:,.1f} s" in ct._P4_SUSPEND
    assert f"{min(others):.1f}-{max(others):.1f} s" in ct._P4_SUSPEND  # 202.3-356.2: the record's own minimum


def test_b6_every_number_a_note_takes_from_a_document_is_in_that_document() -> None:
    p4 = "\n".join((REPO_ROOT / "docs" / "returns" / "P4.md").read_text(encoding="utf-8").splitlines()[321:325])
    for quoted in ("10:19", "14:12", "361 s"):
        assert quoted in ct._P4_SUSPEND and quoted in p4, quoted
    brief = (REPO_ROOT / "docs" / "briefs" / "BRIEF_42_p5.3c_context_length.md").read_text(encoding="utf-8")
    d3 = brief.split("## D3 — Rulings, and a correction", 1)[1][:1500]
    assert "8.5 GB" in ct._HOST_GAME and "8.5 GB" in d3
    assert "17:22-18:04" in ct._HOST_GAME and "17:22–18:04" in d3
    p52 = "\n".join((REPO_ROOT / "docs" / "returns" / "P5.2.md").read_text(encoding="utf-8").splitlines()[254:263])
    assert "2 h 10 min" in ct._CLOCK_JUMP and "2 h 10 m" in p52


def test_b6_the_clock_jump_note_quotes_its_record() -> None:
    runs = json.loads((_output_root() / "p5_2" / "training_mappo1000_dt_nomix_h4.json").read_text())["runs"]
    by_seed = {int(run["seed"]): run for run in runs}
    assert f"{float(by_seed[505]['seconds']):,.1f} s" in ct._CLOCK_JUMP
    rates = [1000 * float(run["seconds"]) / int(run["gradient_steps"]) for seed, run in by_seed.items() if seed != 505]
    assert round(statistics.median(rates)) == 122 and "about 122 ms per step" in ct._CLOCK_JUMP


def test_b6_mappos_worker_count_is_an_inference_with_its_basis_never_a_measurement() -> None:
    mappo = [row for row in ct.TABLE_ROWS if row.family == "mappo"]
    assert len(mappo) == 5
    for row in mappo:
        concurrency = row.training[0]["regime"]["concurrency"]
        assert concurrency["kind"] == "inferred" and concurrency["value"] == 6, row.row_id
        assert "mtime" in concurrency["basis"] and "V7" in concurrency["basis"], row.row_id
        assert "95 %" in concurrency["confidence"], row.row_id
        assert "not recorded by the manifest" in concurrency["recorded"], row.row_id
        text = json.dumps(row.training, default=str)
        assert "16 logical CPUs" not in text and "Measured with six" not in text, row.row_id


def test_b6_the_hz1x1_heuristics_point_their_c3_claims_at_their_own_cells() -> None:
    rows = {row.row_id: row for row in ct.TABLE_ROWS}
    zero_shot = json.loads((DATA / "p7_3a_zero_shot.json").read_text())
    for row_id, path in (("hz1x1.random", "$.cells[200]"), ("hz1x1.maxpressure", "$.cells[100]"),
                         ("hz1x1.fixedtime", "$.cells[0]")):
        arm = row_id.split(".", 1)[1]
        c3 = [claim for claim in rows[row_id].claims if claim["claim"] == "C3"]
        assert c3 == [{"claim": "C3", "record": "p7_3a_zero_shot", "json_path": path, "expect": {"arm": arm}}], row_id
        assert _resolve(zero_shot, path)["arm"] == arm, row_id


def test_b6_units_that_cannot_be_misread() -> None:
    rows = {row.row_id: row for row in ct.TABLE_ROWS}
    for row_id in ("grid4x4.bc", "grid4x4.bc_top10", "grid4x4.bc_top10_perix", "grid4x4.iql"):
        for entry in rows[row_id].training:
            unit = entry["data"]["unit"]
            assert unit.startswith("per-intersection") and "joint window" in unit, (row_id, unit)
    for row_id in ("grid4x4.dt_spatial", "grid4x4.dt_nomix"):
        assert "joint windows" in rows[row_id].training[0]["data"]["unit"], row_id
    for row in (r for r in ct.TABLE_ROWS if r.family == "mappo"):
        batch = row.training[0]["batch"]
        assert batch["path"] == "$.agents[0].params.minibatch_size" and "PPO minibatch_size" in batch["label"]


def test_b6_the_builder_states_which_plan_records_it_pins_and_which_it_does_not() -> None:
    unpinned = ct.UNPINNED_PLAN_RECORDS
    assert set(unpinned) == {"docs/data/p5_3b_decomposition.json", "docs/data/p7_2b_calibration.json",
                             "docs/data/p4_6_grid.json"}
    pinned = {record.relpath for record in ct.PINNED_RECORDS.values()}
    for path, reason in unpinned.items():
        assert (REPO_ROOT / path).is_file() and path not in pinned and len(reason) > 20, path
    assert "UNPINNED_PLAN_RECORDS" in ct.__doc__ and "25 corpus manifests" in ct.__doc__


# ----------------------------------------------------------------------
# B7: what the table does not say
# ----------------------------------------------------------------------

_B7_HYBRID_CORES = (
    "The CPU is a hybrid-core laptop part (Intel Core Ultra 9 275HX) running Linux under WSL2: the Windows host "
    "schedules the guest's virtual CPUs onto performance or efficiency cores and the guest cannot pin them, so the "
    "single-thread figure is this machine's in the recorded regime (mains, Windows power mode Best Performance), not a "
    "property of one core type."
)
_B7_TRAINING_CLOCKS = (
    "The training wall times are the training runs' own clocks: the Windows power mode in force during those runs was "
    "not recorded (the mode found on 2026-10-03 was Best power efficiency), so they are not a controlled benchmark and "
    "are not comparable to the latency regime."
)


def test_b7_what_this_does_not_say_carries_the_hybrid_core_and_the_training_clock_sentences() -> None:
    assert _B7_HYBRID_CORES in ct.WHAT_THIS_DOES_NOT_SAY
    assert _B7_TRAINING_CLOCKS in ct.WHAT_THIS_DOES_NOT_SAY


# ----------------------------------------------------------------------
# Gated: the built artifact against Amendment B
# ----------------------------------------------------------------------


def test_t_sources_every_timed_checkpoint_is_its_rows_representative(built_artifact: Any) -> None:
    artifact, _ = built_artifact
    rows = {row.row_id: row for row in ct.TABLE_ROWS}
    joined = 0
    for out in artifact["rows"]:
        row = rows[out["id"]]
        if row.latency_row is None or row.family == "heuristic":
            continue
        representative = next(c for c in out["checkpoints"] if c["tier"] == row.groups[0]["tier"] and c["seed"] == 101)
        latency_row = cl.row_by_id(row.latency_row)
        assert (latency_row.checkpoint, latency_row.sha256) == (representative["path"], representative["sha256"])
        assert out["inference"]["timed_checkpoint"] == {"path": representative["path"],
                                                        "sha256": representative["sha256"]}, out["id"]
        joined += 1
    assert joined == 36


def test_t_sources_mappos_concurrency_is_an_inference_and_its_batch_the_ppo_minibatch(built_artifact: Any) -> None:
    artifact, _ = built_artifact
    mappo = [out for out in artifact["rows"] if out["family"] == "mappo"]
    assert len(mappo) == 5
    for out in mappo:
        concurrency = out["training"][0]["regime"]["concurrency"]
        assert concurrency["value"] == 6 and "inferred" in concurrency["source"], out["id"]
        assert "PPO minibatch_size" in out["training"][0]["batch"]["label"], out["id"]


#: Each declared note that cites document lines, and the phrases it quotes from them, as the document writes them.
_NOTE_CITATIONS = {
    "_P4_SUSPEND": ("docs/returns/P4.md", ("14018 s", "202–356 s", "10:19", "14:12", "361 s")),
    "_CLOCK_JUMP": ("docs/returns/P5.2.md", ("2 h 10 m",)),
}


def test_b6_each_note_cites_the_lines_that_hold_what_it_quotes() -> None:
    for constant, (document, phrases) in _NOTE_CITATIONS.items():
        note = getattr(ct, constant)
        match = re.search(re.escape(document) + r":(\d+)-(\d+)", note)
        assert match, (constant, note)
        first, last = int(match.group(1)), int(match.group(2))
        cited = "\n".join((REPO_ROOT / document).read_text(encoding="utf-8").splitlines()[first - 1:last])
        for phrase in phrases:
            assert phrase in cited, (constant, phrase, f"{document}:{first}-{last}")
