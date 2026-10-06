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
               reproduced: bool = True, complete: bool = True,
               ns_for: Callable[[cl.LatencyRow, str], list[list[int]]] | None = None) -> tuple[Path, Path]:
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
            ns = _ns() if ns_for is None else ns_for(row, device)  # ns_for: added for Amendment C's tests (C6)
            record = cl.build_record(row, device, ns, draws=cl.TIMING_DRAWS, prompt=None, factory="synthetic",
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


# ======================================================================
# Amendment C (gate G3), written red first in C6: C3.1 latency_sensitivity, C3.2 latency_variability and the nine
# same-computation groups, C3.3 the two code-path facts, C3.4 the power mode's two labels.
# ======================================================================

#: C3.1's late window, typed here rather than read from the builder, so a builder that moves it is caught.
_LATE = 120


def _transient_ns(row: cl.LatencyRow, device: str) -> list[list[int]]:
    """Decisions 0..119 slower than the rest (a start-of-episode transient); rows and episodes offset so that their
    medians differ."""
    offset = (sum(map(ord, row.row_id)) % 97) * 1_000 + (50_000 if device == "cuda" else 0)
    return [[(3_000_000 if k < _LATE else 1_000_000) + 1_000 * k + 7_000 * episode + offset
             for k in range(cl.DECISIONS_PER_EPISODE)] for episode in range(len(cl.TIMING_DRAWS))]


def _independent(values: list[int]) -> tuple[float, int]:
    """This test's own median (the mean of the two middle values for an even count) and nearest-rank p95."""
    ordered = sorted(values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return median, ordered[(95 * n + 99) // 100 - 1]


def test_c3_latency_sensitivity_recomputes_every_cell_over_decisions_120_to_359(tmp_path: Path) -> None:
    rows = (cl.row_by_id("hz1x1.bc"), cl.row_by_id("hz1x1.maxpressure"))
    run_dir, manifest = _write_run(tmp_path, rows=rows, ns_for=_transient_ns)
    sensitivity = ct.latency_sensitivity(ct.verify_latency_run(run_dir, manifest))
    assert set(sensitivity["cells"]) == {"hz1x1.bc_cpu", "hz1x1.bc_cuda", "hz1x1.maxpressure_cpu"}
    worst_median = worst_p95 = 0.0
    above = 0
    for label, cell in sensitivity["cells"].items():
        episodes = [e["decision_ns"] for e in json.loads((run_dir / f"{label}.json").read_text())["episodes"]]
        registered = _independent([v for e in episodes for v in e[cl.WARMUP:]])
        late = _independent([v for e in episodes for v in e[_LATE:]])
        assert (cell["registered"]["median_ms"], cell["registered"]["p95_ms"]) == (registered[0] / 1e6, registered[1] / 1e6)
        assert (cell["late"]["median_ms"], cell["late"]["p95_ms"]) == (late[0] / 1e6, late[1] / 1e6), label
        assert cell["late"]["n_timed"] == 3 * (cl.DECISIONS_PER_EPISODE - _LATE)
        assert cell["median_change"] == (late[0] / 1e6) / (registered[0] / 1e6) - 1
        assert cell["p95_change"] == (late[1] / 1e6) / (registered[1] / 1e6) - 1
        assert cell["p95_change"] < -0.10, "the fixture's transient must move the p95"
        worst_median = max(worst_median, abs(cell["median_change"]))
        worst_p95 = max(worst_p95, abs(cell["p95_change"]))
        above += abs(cell["p95_change"]) > 0.10
    summary = sensitivity["summary"]
    assert summary["window"] == "decisions 120-359" and summary["cells"] == 3
    assert (summary["median_change_max_abs"], summary["p95_change_max_abs"]) == (worst_median, worst_p95)
    assert summary["cells_p95_change_above"] == {"threshold": 0.10, "count": above}


def test_c3_the_sensitivity_sentence_is_generated_from_its_summary() -> None:
    def summary(median: float, p95: float, count: int) -> dict[str, Any]:
        return {"window": "decisions 120-359", "cells": 78, "median_change_max_abs": median, "p95_change_max_abs": p95,
                "cells_p95_change_above": {"threshold": 0.10, "count": count}}

    one = ct.sensitivity_sentence(summary(0.0951, 0.2458, 6))
    two = ct.sensitivity_sentence(summary(0.02, 0.05, 0))
    assert "9.5%" in one and "24.6%" in one and "6 of 78" in one
    assert "2.0%" in two and "5.0%" in two and "0 of 78" in two
    for text in (one, two):
        assert "robust" in text and "conservative upper tail" in text and "10%" in text


def _entry(row_id: str, scenario: str, family: str, architecture: str, deployed: int, n_head: int | None = None,
           context: int | None = None) -> dict[str, Any]:
    def sourced(value: int | None) -> dict[str, Any]:
        return ({"value": value, "source": {"measurement": "synthetic"}} if value is not None
                else {"value": None, "reason": "not an attention model"})

    return {"id": row_id, "scenario": scenario,
            "computation": {"family": family, "architecture": architecture, "deployed_parameters": deployed,
                            "n_head": sourced(n_head), "context_length": sourced(context)}}


def test_c3_same_computation_groups_split_by_scenario_family_network_heads_context_and_size() -> None:
    entries = [
        _entry("a.dt_k20", "a", "dt", "dt.x", 100, 1, 20), _entry("a.dt_k20_bis", "a", "dt", "dt.x", 100, 1, 20),
        _entry("a.dt_k1", "a", "dt", "dt.x", 100, 1, 1), _entry("a.dt_k1_bis", "a", "dt", "dt.x", 100, 1, 1),
        _entry("a.dt_h4", "a", "dt", "dt.x", 100, 4, 20),  # another head count: alone
        _entry("a.bc", "a", "bc", "mlp.x", 50), _entry("a.bc_bis", "a", "bc", "mlp.x", 50),
        _entry("a.iql", "a", "iql", "mlp.x", 50),  # the same deployed network, another decision call: alone
        _entry("b.bc", "b", "bc", "mlp.x", 50),  # another scenario: alone
        _entry("a.bc_big", "a", "bc", "mlp.x", 60),  # another size: alone
        _entry("a.mp", "a", "heuristic", "heuristic.maxpressure", 0),
        _entry("a.ft", "a", "heuristic", "heuristic.fixedtime", 0),
    ]
    groups = ct.same_computation_groups(entries)
    assert sorted(sorted(group["rows"]) for group in groups) == [
        ["a.bc", "a.bc_bis"], ["a.dt_k1", "a.dt_k1_bis"], ["a.dt_k20", "a.dt_k20_bis"]]
    assert len({group["name"] for group in groups}) == 3


def test_c3_latency_variability_takes_each_cells_episode_medians_and_each_groups_spread(tmp_path: Path) -> None:
    rows = (cl.row_by_id("hz1x1.bc"), cl.row_by_id("hz1x1.bc_top10"), cl.row_by_id("hz1x1.iql"))
    run_dir, manifest = _write_run(tmp_path, rows=rows, ns_for=_transient_ns)
    run = ct.verify_latency_run(run_dir, manifest)
    table = [_entry("hz1x1.bc", "hz1x1", "bc", "mlp.x", 50), _entry("hz1x1.bc_top10", "hz1x1", "bc", "mlp.x", 50),
             _entry("hz1x1.iql", "hz1x1", "iql", "mlp.x", 50)]
    variability = ct.latency_variability(run, table)
    assert set(variability["cells"]) == {f"{row.row_id}_{device}" for row in rows for device in ("cpu", "cuda")}
    for label, cell in variability["cells"].items():
        episodes = [e["decision_ns"] for e in json.loads((run_dir / f"{label}.json").read_text())["episodes"]]
        medians = [_independent(episode[cl.WARMUP:])[0] / 1e6 for episode in episodes]
        assert cell["episode_medians_ms"] == medians, label
        assert cell["episode_spread"] == max(medians) / min(medians) - 1
    groups = variability["groups"]
    assert [(group["device"], sorted(group["rows"])) for group in groups] == [
        ("cpu", ["hz1x1.bc", "hz1x1.bc_top10"]), ("cuda", ["hz1x1.bc", "hz1x1.bc_top10"])]
    for group in groups:
        medians = []
        for row_id in group["rows"]:
            record = json.loads((run_dir / f"{row_id}_{group['device']}.json").read_text())
            medians.append(_independent([v for e in record["episodes"] for v in e["decision_ns"][cl.WARMUP:]])[0] / 1e6)
        assert group["medians_ms"] == medians and (group["min_ms"], group["max_ms"]) == (min(medians), max(medians))
        assert group["spread"] == max(medians) / min(medians) - 1 and group["spread"] > 0
    summary = variability["summary"]
    assert summary["episode_spread_max"] == max(cell["episode_spread"] for cell in variability["cells"].values())
    assert summary["same_computation_spread_max"] == {"hz1x1": max(group["spread"] for group in groups)}
    assert summary["groups"] == 1


def test_c3_the_variability_sentence_is_generated_from_its_summary() -> None:
    # The coordinator's recorded values (docs/notes/p8_2_g3/g3_analysis.json $.summary). C6 typed 0.7645 here, which a
    # binary float holds as 0.76449999... and formats as 76.4%: corrected in C6-fix, the expected strings unchanged.
    one = ct.variability_sentence({"episode_spread_max": 0.7645049621135922, "groups": 9,
                                   "same_computation_spread_max": {"hz1x1": 0.06584946650960855,
                                                                   "grid4x4": 0.35887181936452484}})
    two = ct.variability_sentence({"episode_spread_max": 0.1, "groups": 9,
                                   "same_computation_spread_max": {"hz1x1": 0.01, "grid4x4": 0.02}})
    assert "6.6%" in one and "35.9%" in one and "76.5%" in one
    assert "1.0%" in two and "2.0%" in two and "10.0%" in two
    for text in (one, two):
        assert "not interpretable" in text


_C2_MAPPO = (
    "MAPPO's decision call loops over its actors one intersection at a time, each with two host-to-device copies and "
    "two .item() synchronisations (agent/MAPPOAgent.py:225-245): its CUDA figure measures that loop's transfers and "
    "synchronisations, not its networks' arithmetic."
)
_C2_IQL = (
    "IQL's decision call, shared with BC (agent/OfflineBaselines.py:360-405), sets eval mode on every network the agent "
    "holds and restores it afterwards, around a forward pass of the policy alone; IQL holds four networks "
    "(agent/OfflineBaselines.py:594-597), BC one (agent/OfflineBaselines.py:541-543), so the part of IQL's latency above "
    "BC's, with the same deployed network, is that bookkeeping, not arithmetic."
)


def test_c3_what_this_does_not_say_states_the_two_code_path_facts_and_their_lines_hold_them() -> None:
    assert _C2_MAPPO in ct.WHAT_THIS_DOES_NOT_SAY and _C2_IQL in ct.WHAT_THIS_DOES_NOT_SAY

    def lines(path: str, first: int, last: int) -> str:
        return "\n".join((REPO_ROOT / path).read_text(encoding="utf-8").splitlines()[first - 1:last])

    mappo = lines("agent/MAPPOAgent.py", 225, 245)
    assert "for i, actor in enumerate(self.actors)" in mappo
    assert mappo.count("torch.as_tensor(") == 2 and mappo.count(".item()") == 2
    act = lines("agent/OfflineBaselines.py", 360, 405)
    assert "module.eval()" in act and "module.train(mode)" in act and "self.policy_logits(state)" in act
    assert "return [self.policy, self.q, self.v, self.q_target]" in lines("agent/OfflineBaselines.py", 594, 597)
    assert "return [self.model]" in lines("agent/OfflineBaselines.py", 541, 543)


def test_c3_the_hardware_block_names_the_better_battery_overlay_by_both_labels() -> None:
    machine = {"cpu_model": "synthetic", "power": _MAINS_BEST}
    regime = {"torch_num_threads": 1, "device": "cpu", "synchronize": None}
    run: dict[str, Any] = {"records": {("a", "cpu"): {"record": {"machine": machine, "regime": regime}}}}
    labels = ct._hardware(run)["power_mode_labels"]["961cc777-2547-4f9d-8174-7d86181b8a7a"]
    assert labels["documentation"] == "Better Battery" == cl.POWER_OVERLAYS["961cc777-2547-4f9d-8174-7d86181b8a7a"]
    assert labels["windows_11_settings"] == "Best power efficiency"
    assert _B7_TRAINING_CLOCKS in ct.WHAT_THIS_DOES_NOT_SAY  # the author's sentence, unchanged


def test_c3_the_artifact_carries_sensitivity_variability_and_their_generated_sentences(built_artifact: Any) -> None:
    artifact, _ = built_artifact
    sensitivity, variability = artifact["latency_sensitivity"], artifact["latency_variability"]
    assert len(sensitivity["cells"]) == len(variability["cells"]) == 78
    assert ct.sensitivity_sentence(sensitivity["summary"]) in artifact["what_this_does_not_say"]
    assert ct.variability_sentence(variability["summary"]) in artifact["what_this_does_not_say"]
    for out in artifact["rows"]:
        assert (out["computation"]["family"], out["computation"]["architecture"]) == (out["family"], out["architecture"])


# ----------------------------------------------------------------------
# Gated: the REAL timing run of G2 (output/p8_2/latency/20261004T194421Z)
# ----------------------------------------------------------------------

REAL_RUN = "20261004T194421Z"

#: C1.2's nine same-computation groups (the same scenario, network, context length and deployed parameters).
_NINE = {frozenset(rows) for rows in (
    ("hz1x1.dt_k20", "hz1x1.dt_nortg", "hz1x1.h4.k20", "hz1x1.c3.anchor_k200"),
    ("hz1x1.h4.k1", "hz1x1.h4.k1_b1280"),
    ("hz1x1.h4.k2", "hz1x1.h4.k2_b640"),
    ("hz1x1.bc", "hz1x1.bc_top10", "hz1x1.bc_best2_20", "hz1x1.bc_any_20", "hz1x1.bc_worst2_20", "hz1x1.bc_best2_all"),
    ("hz1x1.mappo1000", "hz1x1.mappo500", "hz1x1.mappo060"),
    ("grid4x4.dt_spatial", "grid4x4.dt_nomix"),
    ("grid4x4.dt_spatial_h4", "grid4x4.dt_nomix_h4", "grid4x4.c3.ft_k5", "grid4x4.c3.ft_k20", "grid4x4.c3.ft_k100",
     "grid4x4.c3.ft_k100_b1000", "grid4x4.c3.ft_k100_b16000", "grid4x4.c3.scratch_k100"),
    ("grid4x4.bc", "grid4x4.bc_top10", "grid4x4.bc_top10_perix"),
    ("grid4x4.mappo1000", "grid4x4.mappo060"),
)}


@pytest.fixture(scope="module")
def real_artifact() -> dict[str, Any]:
    output_root = _output_root()
    corpus_root = _corpus_root()
    run_dir = output_root / "p8_2" / "latency" / REAL_RUN
    if not (run_dir / "COMPLETE").is_file():
        pytest.skip(f"{run_dir} not found: G2's timing run is in the main tree's output (gitignored)")
    committed = DATA / "p8_2_compute.json"
    git = json.loads(committed.read_text())["git"] if committed.is_file() else {}
    roots = ct.Roots(repo_root=REPO_ROOT, output_root=output_root, corpus_root=corpus_root, latency_dir=run_dir,
                     manifest_path=output_root / "SHA256SUMS_p8_2.txt")
    return ct.build_artifact(roots, git=git)


def test_c3_the_same_computation_groups_are_the_nine_of_amendment_c(real_artifact: dict[str, Any]) -> None:
    groups = real_artifact["latency_variability"]["groups"]
    assert {frozenset(group["rows"]) for group in groups} == _NINE
    assert sorted(group["device"] for group in groups) == ["cpu"] * 9 + ["cuda"] * 9


def test_c3_sensitivity_and_variability_equal_the_coordinators_third_route(real_artifact: dict[str, Any]) -> None:
    g3 = json.loads((REPO_ROOT / "docs" / "notes" / "p8_2_g3" / "g3_analysis.json").read_text())
    sensitivity, variability = real_artifact["latency_sensitivity"], real_artifact["latency_variability"]
    assert set(sensitivity["cells"]) == set(variability["cells"]) == set(g3["cells"])
    for label, cell in sensitivity["cells"].items():
        theirs = g3["cells"][label]
        assert (cell["registered"]["median_ms"], cell["registered"]["p95_ms"]) == (theirs["median_ms_20_359"],
                                                                                   theirs["p95_ms_20_359"]), label
        assert (cell["late"]["median_ms"], cell["late"]["p95_ms"]) == (theirs["median_ms_120_359"],
                                                                       theirs["p95_ms_120_359"]), label
        assert (cell["median_change"], cell["p95_change"]) == (theirs["median_change_if_120"],
                                                               theirs["p95_change_if_120"]), label
    for label, cell in variability["cells"].items():
        theirs = g3["cells"][label]
        assert (cell["episode_medians_ms"], cell["episode_spread"]) == (theirs["episode_medians_ms"],
                                                                        theirs["episode_spread"]), label
    for group in variability["groups"]:
        theirs = [g for key, g in g3["groups"].items()
                  if key.endswith(f"/ {group['device']}") and frozenset(g["rows"]) == frozenset(group["rows"])]
        assert len(theirs) == 1, group["name"]
        assert (group["min_ms"], group["max_ms"], group["spread"]) == (theirs[0]["min_ms"], theirs[0]["max_ms"],
                                                                       theirs[0]["spread"]), group["name"]
    summary = g3["summary"]
    assert sensitivity["summary"]["median_change_max_abs"] == summary["median_change_if_120_max_abs"]
    assert sensitivity["summary"]["p95_change_max_abs"] == summary["p95_change_if_120_max_abs"]
    assert sensitivity["summary"]["cells_p95_change_above"]["count"] == summary["cells_p95_change_above_10pct"]
    assert variability["summary"]["episode_spread_max"] == summary["episode_spread_max"]
    assert variability["summary"]["same_computation_spread_max"] == {
        "hz1x1": summary["same_computation_spread_max_hz1x1"], "grid4x4": summary["same_computation_spread_max_grid4x4"]}


# ----------------------------------------------------------------------
# T-regress: the committed artifact docs/data/p8_2_compute.json (BRIEF_43 §8, Amendment C, C3.6)
# ----------------------------------------------------------------------


def test_t_regress_the_committed_artifact_regenerates_byte_identically(real_artifact: dict[str, Any],
                                                                       tmp_path: Path) -> None:
    """Gated: the builder at this commit, on G2's run and the pinned records, regenerates the committed bytes -- the git
    block it was built at being the one input a later commit cannot regenerate, so it is read from the committed file."""
    committed = DATA / "p8_2_compute.json"
    assert committed.is_file(), f"{committed} is not committed"
    regenerated = tmp_path / "p8_2_compute.json"
    ct.write_artifact(regenerated, real_artifact)
    assert hashlib.sha256(regenerated.read_bytes()).hexdigest() == hashlib.sha256(committed.read_bytes()).hexdigest()
    assert regenerated.read_bytes() == committed.read_bytes()


def test_the_committed_artifact_is_consistent_with_its_own_summaries_and_c12s_nine_groups() -> None:
    """Ungated (CI): the committed artifact's groups are C1.2's nine, and its two generated sentences are the ones its
    own summaries generate."""
    artifact = json.loads((DATA / "p8_2_compute.json").read_text())
    assert artifact["format_version"] == ct.FORMAT_VERSION and len(artifact["rows"]) == 44
    assert artifact["git"]["dirty"] is False and artifact["latency_run"]["stamp"] == REAL_RUN
    assert {frozenset(group["rows"]) for group in artifact["latency_variability"]["groups"]} == _NINE
    said = artifact["what_this_does_not_say"]
    assert said[:len(ct.WHAT_THIS_DOES_NOT_SAY)] == list(ct.WHAT_THIS_DOES_NOT_SAY)
    assert said[len(ct.WHAT_THIS_DOES_NOT_SAY):] == [ct.sensitivity_sentence(artifact["latency_sensitivity"]["summary"]),
                                                     ct.variability_sentence(artifact["latency_variability"]["summary"])]


# ======================================================================
# DEFERRED 107 (docs/reviews/P8.2.md §4, mandate M2): the test gaps that guard FUTURE rebuilds of the committed artifact,
# closed before the next builder change (BRIEF_44 §3.4 and Amendment A). Each test below is green on the builder and the
# artifact as merged, and red against the surviving mutant it names (M2-2a, M2-2b, M2-3e, M2-3f, M2-6c, and M2-7a's
# note) -- the mutants re-run, committed, in a throwaway worktree, their red runs pasted in docs/returns/P5.2b.md.
# ======================================================================

_SENSITIVITY_CLAUSES = re.compile(
    r"a cell's median moves by at most (?P<median>\d+\.\d)% but its p95 by up to (?P<p95>\d+\.\d)% "
    r"\((?P<count>\d+) of (?P<cells>\d+) cells by more than (?P<threshold>\d+)%\)"
)
_VARIABILITY_CLAUSES = re.compile(
    r"differ in median by up to (?P<hz1x1>\d+\.\d)% on hz1x1 and (?P<grid4x4>\d+\.\d)% on grid4x4, and one cell's "
    r"three episode medians by up to (?P<episodes>\d+\.\d)%"
)
_C2_REFERENCE = re.compile(r"(agent/[A-Za-z_]+\.py):(\d+)-(\d+)")


def _percent(fraction: float) -> str:
    """This file's own rendering of a fraction as a percent at one decimal: what a clause must carry."""
    return f"{100.0 * fraction:.1f}"


def _committed_artifact() -> dict[str, Any]:
    return json.loads((DATA / "p8_2_compute.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("median, p95, count", [(0.0951, 0.2458, 6), (0.3012, 0.0517, 0)])
def test_d107_the_sensitivity_sentence_puts_each_number_in_its_own_clause(median: float, p95: float,
                                                                           count: int) -> None:
    """M2-3f: the median clause carries the median's change and the p95 clause the p95's, read by POSITION -- in both
    orders of magnitude, so a builder that swaps the two clauses, or orders them by size, cannot pass."""
    summary = {"window": "decisions 120-359", "cells": 78, "median_change_max_abs": median, "p95_change_max_abs": p95,
               "cells_p95_change_above": {"threshold": 0.10, "count": count}}
    found = _SENSITIVITY_CLAUSES.search(ct.sensitivity_sentence(summary))
    assert found is not None, "the sensitivity sentence no longer has its median, p95 and count clauses"
    assert (found["median"], found["p95"]) == (_percent(median), _percent(p95))
    assert (int(found["count"]), int(found["cells"]), int(found["threshold"])) == (count, 78, 10)


@pytest.mark.parametrize("spreads", [
    {"hz1x1": 0.06584946650960855, "grid4x4": 0.35887181936452484},
    {"grid4x4": 0.0217, "hz1x1": 0.4108},
])
def test_d107_the_variability_sentence_attributes_each_floor_to_its_own_scenario(spreads: dict[str, float]) -> None:
    """M2-3e: each scenario's floor is printed beside THAT scenario's name, in both orders of magnitude and of the
    mapping's insertion, so a builder that pairs a floor with the other scenario cannot pass."""
    summary = {"episode_spread_max": 0.1234, "groups": 9, "same_computation_spread_max": dict(spreads)}
    found = _VARIABILITY_CLAUSES.search(ct.variability_sentence(summary))
    assert found is not None, "the variability sentence no longer has its two floor clauses and its episode clause"
    assert (found["hz1x1"], found["grid4x4"], found["episodes"]) == (
        _percent(spreads["hz1x1"]), _percent(spreads["grid4x4"]), _percent(0.1234))


def test_d107_the_committed_sentences_carry_their_own_summaries_numbers_in_their_places() -> None:
    """M2-3e / M2-3f after a rebuild and a recommit: the COMMITTED sentences, read by position, against the COMMITTED
    summaries -- independent of the generator, which a recommitted mutant would have changed too."""
    artifact = _committed_artifact()
    said = artifact["what_this_does_not_say"]
    sensitivity = artifact["latency_sensitivity"]["summary"]
    variability = artifact["latency_variability"]["summary"]
    found = [match for match in map(_SENSITIVITY_CLAUSES.search, said) if match]
    assert len(found) == 1, "exactly one committed sentence states the sensitivity"
    assert (found[0]["median"], found[0]["p95"]) == (_percent(sensitivity["median_change_max_abs"]),
                                                     _percent(sensitivity["p95_change_max_abs"]))
    assert (int(found[0]["count"]), int(found[0]["cells"])) == (sensitivity["cells_p95_change_above"]["count"],
                                                                sensitivity["cells"])
    found = [match for match in map(_VARIABILITY_CLAUSES.search, said) if match]
    assert len(found) == 1, "exactly one committed sentence states the variability"
    floors = variability["same_computation_spread_max"]
    assert (found[0]["hz1x1"], found[0]["grid4x4"], found[0]["episodes"]) == (
        _percent(floors["hz1x1"]), _percent(floors["grid4x4"]), _percent(variability["episode_spread_max"]))


def test_d107_every_committed_summary_recomputes_from_the_artifacts_own_cells_and_groups() -> None:
    """M2-2a (ungated, CI): each summary of the committed artifact recomputed HERE from the cells and groups the same
    artifact carries, so a coherent hand-edit of a published floor and its sentence no longer passes."""
    artifact = _committed_artifact()
    sensitivity, variability = artifact["latency_sensitivity"], artifact["latency_variability"]
    cells = sensitivity["cells"]
    for label, cell in cells.items():
        assert cell["median_change"] == cell["late"]["median_ms"] / cell["registered"]["median_ms"] - 1, label
        assert cell["p95_change"] == cell["late"]["p95_ms"] / cell["registered"]["p95_ms"] - 1, label
    summary = sensitivity["summary"]
    threshold = summary["cells_p95_change_above"]["threshold"]
    assert summary["cells"] == len(cells) == 78
    assert summary["median_change_max_abs"] == max(abs(cell["median_change"]) for cell in cells.values())
    assert summary["p95_change_max_abs"] == max(abs(cell["p95_change"]) for cell in cells.values())
    assert summary["cells_p95_change_above"]["count"] == sum(abs(cell["p95_change"]) > threshold
                                                             for cell in cells.values())
    h4 = [abs(cell["median_change"]) for label, cell in cells.items() if label.startswith("hz1x1.h4.")]
    assert (summary["h4_cells"], summary["h4_median_change_max_abs"]) == (len(h4), max(h4))
    spread_cells = variability["cells"]
    for label, cell in spread_cells.items():
        assert cell["episode_spread"] == max(cell["episode_medians_ms"]) / min(cell["episode_medians_ms"]) - 1, label
    groups = variability["groups"]
    for group in groups:
        medians = group["medians_ms"]
        assert (group["min_ms"], group["max_ms"]) == (min(medians), max(medians)), group["name"]
        assert group["spread"] == max(medians) / min(medians) - 1, group["name"]
    summary = variability["summary"]
    worst = max(spread_cells, key=lambda label: spread_cells[label]["episode_spread"])
    assert summary["cells"] == len(spread_cells)
    assert (summary["episode_spread_max"], summary["episode_spread_max_cell"]) == (
        spread_cells[worst]["episode_spread"], worst)
    assert summary["groups"] == len({group["name"] for group in groups})
    assert summary["same_computation_spread_max"] == {
        scenario: max(group["spread"] for group in groups if group["scenario"] == scenario)
        for scenario in {group["scenario"] for group in groups}}


def test_d107_the_committed_groups_are_the_nine_on_each_device() -> None:
    """M2-2b (ungated, CI): the committed artifact holds 9 cpu and 9 cuda group entries, the nine of C1.2 on each
    device -- a dropped (or duplicated) device entry no longer passes."""
    groups = _committed_artifact()["latency_variability"]["groups"]
    assert sorted(group["device"] for group in groups) == ["cpu"] * 9 + ["cuda"] * 9
    for device in ("cpu", "cuda"):
        assert {frozenset(group["rows"]) for group in groups if group["device"] == device} == _NINE, device


def test_d107_the_h4_summary_equals_the_coordinators_third_route() -> None:
    """M2-6c after a rebuild and a recommit (ungated): ``h4_cells`` and ``h4_median_change_max_abs`` of the committed
    artifact against the coordinator's G3 analysis (docs/notes/p8_2_g3/g3_analysis.json), which holds every cell's
    change by its own route and no H4 summary of its own."""
    g3 = json.loads((REPO_ROOT / "docs" / "notes" / "p8_2_g3" / "g3_analysis.json").read_text(encoding="utf-8"))
    theirs = [abs(cell["median_change_if_120"]) for label, cell in g3["cells"].items() if label.startswith("hz1x1.h4.")]
    summary = _committed_artifact()["latency_sensitivity"]["summary"]
    assert len(theirs) == 14, "the G3 analysis holds the seven H4 rows on both devices"
    assert (summary["h4_cells"], summary["h4_median_change_max_abs"]) == (len(theirs), max(theirs))


def test_d107_latency_sensitivity_counts_every_h4_cell_on_both_devices(tmp_path: Path) -> None:
    """M2-6c (ungated): the builder's H4 summary covers every ``hz1x1.h4.*`` cell on both devices and nothing else,
    each change recomputed HERE from the records' own nanoseconds."""
    rows = tuple(cl.row_by_id(row_id) for row_id in ("hz1x1.h4.k1", "hz1x1.h4.k20", "hz1x1.bc"))
    run_dir, manifest = _write_run(tmp_path, rows=rows, ns_for=_transient_ns)
    sensitivity = ct.latency_sensitivity(ct.verify_latency_run(run_dir, manifest))
    labels = sorted(label for label in sensitivity["cells"] if label.startswith("hz1x1.h4."))
    assert labels == ["hz1x1.h4.k1_cpu", "hz1x1.h4.k1_cuda", "hz1x1.h4.k20_cpu", "hz1x1.h4.k20_cuda"]
    changes = []
    for label in labels:
        episodes = [e["decision_ns"] for e in json.loads((run_dir / f"{label}.json").read_text())["episodes"]]
        registered = _independent([v for e in episodes for v in e[cl.WARMUP:]])[0] / 1e6
        late = _independent([v for e in episodes for v in e[_LATE:]])[0] / 1e6
        changes.append(abs(late / registered - 1))
    assert len(set(changes)) == len(changes), "the fixture must give each H4 cell its own change"
    summary = sensitivity["summary"]
    assert (summary["h4_cells"], summary["h4_median_change_max_abs"]) == (4, max(changes))


def test_d107_the_c2_line_ranges_are_read_from_the_sentences_and_their_lines_hold_what_they_state() -> None:
    """M2-7a's note: the two code-path sentences' line ranges are PARSED from the sentences themselves, not typed beside
    them, and each range is read back from the file it names -- so moving a sentence's range and a typed copy together
    no longer passes."""
    mappo = [text for text in ct.WHAT_THIS_DOES_NOT_SAY if text.startswith("MAPPO's decision call loops over its actors")]
    iql = [text for text in ct.WHAT_THIS_DOES_NOT_SAY if text.startswith("IQL's decision call, shared with BC")]
    assert len(mappo) == len(iql) == 1

    def lines(path: str, first: str, last: str) -> str:
        text = (REPO_ROOT / path).read_text(encoding="utf-8").splitlines()
        return "\n".join(text[int(first) - 1:int(last)])

    references = _C2_REFERENCE.findall(mappo[0])
    assert [reference[0] for reference in references] == ["agent/MAPPOAgent.py"]
    loop = lines(*references[0])
    assert "for i, actor in enumerate(self.actors)" in loop
    assert loop.count("torch.as_tensor(") == 2 and loop.count(".item()") == 2
    references = _C2_REFERENCE.findall(iql[0])
    assert [reference[0] for reference in references] == ["agent/OfflineBaselines.py"] * 3
    act, iql_networks, bc_networks = (lines(*reference) for reference in references)
    assert "module.eval()" in act and "module.train(mode)" in act and "self.policy_logits(state)" in act
    assert "return [self.policy, self.q, self.v, self.q_target]" in iql_networks
    assert "return [self.model]" in bc_networks
