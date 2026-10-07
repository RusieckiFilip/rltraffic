"""P5.2b (``BRIEF_44`` and Amendment A): the IQL random-tier correction run, ``offline/iql_correction.py``.

Written against a signature-only skeleton, so every test reaches the real API and is red for its own reason first.

* **T-isolation** (ungated): the out-root is the run's directory or a pre-flight's and nothing else; every other child
  of ``output/`` and the corpus are protected; every write target lies under the out-root (or is the run's manifest);
  a refusal creates nothing, a written file is never rewritten, and P5.2's and P8.4b's trees stay byte-identical.
* **T-manifest, T-status, T-record** (ungated): the run's manifest, each stage's state read from disk, the training
  record and the checkpoints moved into place.
* **T-train-args** (ungated, plan F10): on one synthetic tier, ``_run_train_baselines`` and the module hand
  ``train_iql`` the same table, tensor for tensor, and the same arguments but the checkpoint path and one block.
* **T-route** (ungated): P8.4b's resolver and P5.2's ``--checkpoint-dir`` read one physical copy; an evaluation of any
  other model is refused.
* **T-statements** (ungated): every recomputed statement on hand-built cells, ties included, by this file's own loops.
* **The report** (ungated): the artifact on a synthetic record and run, before and after, both definitions, and its
  refusals.
* **Gated, on the real record:** **T-rows** (1,152,000 rows, the declared streams, by a second route), T-scale and
  T-stats; **T-protocol** (the evaluation's arguments read from ``output/p5_2/eval_random_iql.json`` by this test);
  **T-reproduce (a)** and **(b)** (P5.2's and P8.4b's paths on the ORIGINAL seed-101 checkpoint, draws 1000-1004,
  reproducing the committed episodes under ``==``); **T-reproduce (c)** (the recomputation fed the original files
  reproduces every committed random-tier statement, and its ``att_engine`` half equals this file's own recomputation
  and the coordinator's note).

GATES: ``RLTRAFFIC_OUTPUT_ROOT`` (the main tree's ``output/``), ``RLTRAFFIC_CORPUS_V11`` (``datasets_v11``),
``RLTRAFFIC_DRAWS`` (``scenarios/draws``) and the CityFlow engine; each gated test skips naming what it needs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import offline.iql_correction as ic
import offline.tier_sweep as ts
from offline.dt_gate import EpisodeResult
from tests import p5_2b_fixtures as fx

REPO_ROOT = Path(__file__).resolve().parents[1]

#: P5.2's committed random-tier statements, as committed (docs/returns/P5.2.md section 1; PROJECT_PLAN at 119cc48).
COMMITTED = {
    "q1_iql_random_measured_2dp": "190.96",
    "q1_iql_random_predicted": 414.993,
    "q1_n_held": 15,
    "q1_threshold": 14,
    "ranking": ["iql", "dt_spatial", "dt_nomix", "bc", "bc_top10_perix", "bc_top10"],
    "q2a_first": "iql",
    "q2b_concordant": 8,
    "q2b_hard_concordant": 3,
    "q3a_rank": 3,
    "q3c_best": "iql",
    "q3c": "+64.3603 [+62.1046, +66.6160]",
}


# ----------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------


def _gate(variable: str, default: Path, marker: str) -> Path:
    value = os.environ.get(variable)
    candidate = Path(value) if value else default
    if not (candidate / marker).exists():
        pytest.skip(f"{candidate / marker} not found: set {variable} to the main tree's copy (it is gitignored)")
    return candidate


def _output_root() -> Path:
    return _gate("RLTRAFFIC_OUTPUT_ROOT", REPO_ROOT / "output", "p5_2/eval_random_iql.json")


def _corpus_root() -> Path:
    return _gate("RLTRAFFIC_CORPUS_V11", REPO_ROOT / "datasets_v11", "cf_grid4x4__random/manifest.json")


def _draws_root() -> Path:
    return _gate("RLTRAFFIC_DRAWS", REPO_ROOT / "scenarios" / "draws", "cityflow_grid4x4/draw_1000/cityflow.json")


def _p8_4b_cells() -> Path:
    root = _output_root()
    if not (root / "p8_4b_rederivation" / "CAMPAIGN_COMPLETE").is_file():
        pytest.skip(f"{root / 'p8_4b_rederivation'} holds no complete campaign: set RLTRAFFIC_OUTPUT_ROOT")
    return root


def _cityflow() -> None:
    try:
        import cityflow  # noqa: F401
    except ImportError:
        pytest.skip("the CityFlow engine is not importable: T-reproduce rolls real episodes")


def _real_roots(out_root: Path | None = None) -> ic.Roots:
    output = _output_root()
    return ic.Roots(repo_root=REPO_ROOT, output_root=output, corpus_root=_corpus_root(), draws_root=_draws_root(),
                    out_root=out_root if out_root is not None else output / "p5_2b")


@pytest.fixture
def keep_torch_threads() -> Any:
    """P5.2's ``main`` pins torch to one thread; give the rest of the suite its count back."""
    count = torch.get_num_threads()
    yield
    torch.set_num_threads(count)


# ----------------------------------------------------------------------
# A fake output/ for the barrier
# ----------------------------------------------------------------------


def _fake_roots(tmp_path: Path, *, out: str = "p5_2b") -> ic.Roots:
    output = tmp_path / "output"
    for name in ("p5_2", "p8_4b_rederivation", "p5_1"):
        (output / name).mkdir(parents=True, exist_ok=True)
        (output / name / "keep.json").write_text("{}\n", encoding="utf-8")
    (output / "SHA256SUMS_p5_2.txt").write_text("x\n", encoding="utf-8")
    for name in ("repo", "corpus", "draws"):
        (tmp_path / name).mkdir(exist_ok=True)
    return ic.Roots(repo_root=tmp_path / "repo", output_root=output, corpus_root=tmp_path / "corpus",
                    draws_root=tmp_path / "draws", out_root=output / out)


def _tree_digest(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def _listing(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


# ======================================================================
# T-isolation
# ======================================================================


def test_the_paths_are_the_runs_layout() -> None:
    output = Path("/x/output")
    assert ic.run_root(output) == output / "p5_2b"
    assert ic.runs_root(output) == output / "p5_2b_runs"
    assert ic.manifest_path(output) == output / "SHA256SUMS_p5_2b.txt"
    assert ic.checkpoint_dir(output / "p5_2b") == output / "p5_2b" / "p5_2" / "checkpoints"


def test_t_isolation_the_out_root_is_the_run_or_a_preflight_and_nothing_else(tmp_path: Path) -> None:
    roots = _fake_roots(tmp_path)
    output = roots.output_root
    assert ic.assert_out_root(output / "p5_2b", output) == (output / "p5_2b").resolve()
    assert ic.assert_out_root(output / "p5_2b_runs" / "preflight_20261006T000000Z", output).name.startswith("preflight")
    refused = [output / "p5_2", output / "p5_2b" / ".." / "p5_2", output / "p5_2b_runs", output / "p8_4b_rederivation",
               output / "elsewhere", tmp_path / "outside", output / "p5_2" / "p5_2b"]
    for candidate in refused:
        with pytest.raises(PermissionError, match="out-root"):
            ic.assert_out_root(candidate, output)
    (output / "p5_2b").symlink_to(output / "p5_2", target_is_directory=True)
    with pytest.raises(PermissionError, match="out-root"):
        ic.assert_out_root(output / "p5_2b", output)


def test_t_isolation_the_protected_roots_are_every_other_child_of_output_and_the_corpus(tmp_path: Path) -> None:
    roots = _fake_roots(tmp_path)
    (roots.output_root / "p5_2b").mkdir()
    (roots.output_root / "p5_2b_runs").mkdir()
    protected = set(ic.protected_roots(roots))
    for name in ("p5_2", "p8_4b_rederivation", "p5_1", "p5_2b_runs"):
        assert (roots.output_root / name).resolve() in protected, name
    assert roots.corpus_root.resolve() in protected and roots.draws_root.resolve() in protected
    assert (roots.output_root / "p5_2b").resolve() not in protected
    preflight = ic.Roots(**{**roots.__dict__, "out_root": roots.output_root / "p5_2b_runs" / "preflight_x"})
    protected = set(ic.protected_roots(preflight))
    assert (roots.output_root / "p5_2b").resolve() in protected
    assert (roots.output_root / "p5_2b_runs").resolve() not in protected


def test_t_isolation_every_write_target_lies_under_the_out_root_or_is_the_runs_manifest(tmp_path: Path) -> None:
    roots = _fake_roots(tmp_path)
    protected = ic.protected_roots(roots)
    output, out = roots.output_root, roots.out_root
    for allowed in (out / "training_random_iql.json", out / "p5_2" / "checkpoints" / "grid4x4_random_iql_seed101.pt",
                    out / "rederivation" / "cell.json", output / "SHA256SUMS_p5_2b.txt"):
        assert ic.assert_target(allowed, roots, protected) == allowed.resolve()
    for refused in (output / "p5_2" / "eval_random_iql.json", output / "p8_4b_rederivation" / "cell.json",
                    output / "SHA256SUMS_p5_2.txt", output / "elsewhere.json", out / ".." / "p5_2" / "x.json",
                    roots.corpus_root / "x.json"):
        with pytest.raises(PermissionError, match="not a write target"):
            ic.assert_target(refused, roots, protected)
    preflight = ic.Roots(**{**roots.__dict__, "out_root": output / "p5_2b_runs" / "preflight_x"})
    with pytest.raises(PermissionError, match="not a write target"):
        ic.assert_target(output / "SHA256SUMS_p5_2b.txt", preflight, ic.protected_roots(preflight))


def test_t_isolation_a_refused_write_creates_nothing_and_a_written_file_is_never_rewritten(tmp_path: Path) -> None:
    roots = _fake_roots(tmp_path)
    roots.out_root.mkdir()
    protected = ic.protected_roots(roots)
    before = _listing(roots.output_root)
    for target in (roots.output_root / "p5_2" / "new" / "x.json", roots.output_root / "x.json"):
        with pytest.raises(PermissionError, match="not a write target"):
            ic.write_once_json(target, {"a": 1}, roots, protected)
    assert _listing(roots.output_root) == before
    path = ic.write_once_json(roots.out_root / "record.json", {"a": 1}, roots, protected)
    assert json.loads(path.read_text()) == {"a": 1}
    with pytest.raises(FileExistsError, match="written once"):
        ic.write_once_json(roots.out_root / "record.json", {"a": 2}, roots, protected)
    assert json.loads(path.read_text()) == {"a": 1}


def test_t_isolation_the_manifest_and_the_report_leave_p5_2_and_p8_4b_byte_identical(tmp_path: Path,
                                                                                     monkeypatch: Any) -> None:
    """Renamed under Amendment B, B1.8 (RA1: the old name claimed a whole run): the run's files here are the fixture's,
    so the module's writers exercised are the manifest's and the report's; the stages' own writes are held to the same
    property by the whole-run test's tree digests."""
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    output = record.roots.output_root
    before = {name: _tree_digest(output / name) for name in ("p5_2", "p8_4b_rederivation")}
    sums = (output / "SHA256SUMS_p5_2.txt").read_bytes()
    fx.write_synthetic_run(record)
    protected = ic.protected_roots(record.roots)
    ic.write_run_manifest(record.roots, protected)
    ic.write_report(record.roots, ic.build_report(record.roots, git={"commit": "0" * 40, "dirty": False},
                                                  pins=record.pins), protected)
    assert {name: _tree_digest(output / name) for name in before} == before
    assert (output / "SHA256SUMS_p5_2.txt").read_bytes() == sums


# ======================================================================
# T-manifest and T-status
# ======================================================================


def test_t_manifest_lists_every_run_file_but_the_artifacts_once_under_p5_2bs_prefix(tmp_path: Path) -> None:
    roots = _fake_roots(tmp_path)
    out = roots.out_root
    for relative in ("training_random_iql.json", "p5_2/checkpoints/grid4x4_random_iql_seed101.pt",
                     "rederivation/cell_a.json", "artifacts/p5_2b_correction.json"):
        (out / relative).parent.mkdir(parents=True, exist_ok=True)
        (out / relative).write_text(relative, encoding="utf-8")
    lines = ic.write_run_manifest(roots, ic.protected_roots(roots))
    expected = sorted(f"{hashlib.sha256((out / r).read_bytes()).hexdigest()}  p5_2b/{r}"
                      for r in ("training_random_iql.json", "p5_2/checkpoints/grid4x4_random_iql_seed101.pt",
                                "rederivation/cell_a.json"))
    assert lines == expected
    manifest = roots.output_root / "SHA256SUMS_p5_2b.txt"
    assert manifest.read_text(encoding="utf-8").splitlines() == expected
    with pytest.raises(FileExistsError, match="written once"):
        ic.write_run_manifest(roots, ic.protected_roots(roots))
    assert manifest.read_text(encoding="utf-8").splitlines() == expected
    preflight = ic.Roots(**{**roots.__dict__, "out_root": roots.output_root / "p5_2b_runs" / "preflight_x"})
    (preflight.out_root / "a.json").parent.mkdir(parents=True)
    (preflight.out_root / "a.json").write_text("{}")
    with pytest.raises(PermissionError, match="only the run"):
        ic.write_run_manifest(preflight, ic.protected_roots(preflight))


def test_t_status_reads_each_stage_from_disk(tmp_path: Path, monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    absent = ic.status(record.roots, pins=record.pins)
    assert set(absent.values()) == {"absent"}
    assert set(absent) >= {"training", "canaries", "p5_2_eval", "rederivation", "manifest", "report"}
    fx.write_synthetic_run(record, stages=("training", "canaries"))
    state = ic.status(record.roots, pins=record.pins)
    assert (state["training"], state["canaries"], state["p5_2_eval"], state["rederivation"]) == (
        "complete", "complete", "absent", "absent")
    fx.write_synthetic_run(record, stages=("p5_2_eval", "rederivation"))
    state = ic.status(record.roots, pins=record.pins)
    assert (state["p5_2_eval"], state["rederivation"], state["manifest"]) == ("complete", "complete", "absent")


def test_t_status_a_checkpoint_without_its_record_is_partial(tmp_path: Path, monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    target = ic.checkpoint_dir(record.roots.out_root) / "grid4x4_random_iql_seed101.pt.partial"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"half")
    assert ic.status(record.roots, pins=record.pins)["training"] == "partial"
    with pytest.raises(ValueError, match="partial"):
        ic.check(record.roots, pins=record.pins, require_cuda=False)


def test_t_status_a_checkpoint_that_is_not_the_records_is_partial(tmp_path: Path, monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record, stages=("training", "canaries"))
    path = ic.checkpoint_dir(record.roots.out_root) / "grid4x4_random_iql_seed202.pt"
    path.write_bytes(path.read_bytes() + b"x")
    assert ic.status(record.roots, pins=record.pins)["training"] == "partial"


def test_t_status_a_corrected_cell_that_disagrees_with_the_p5_2_path_is_not_complete(tmp_path: Path,
                                                                                     monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record)
    cell = sorted((record.roots.out_root / "rederivation").glob("cell_*.json"))[0]
    payload = json.loads(cell.read_text())
    payload["att_ours"] += 1.0
    cell.write_text(json.dumps(payload))
    assert ic.status(record.roots, pins=record.pins)["rederivation"] == "partial"


# ======================================================================
# T-check
# ======================================================================


def test_check_passes_on_a_fresh_synthetic_record_and_writes_nothing(tmp_path: Path, monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    before = _tree_digest(tmp_path)
    report = ic.check(record.roots, pins=record.pins, require_cuda=False)
    assert report["state"] == "fresh" and report["ok"] is True
    assert report["declared"]["rows"] == len(fx.SELECTED) * 16 * fx.EPISODE_LENGTH
    assert _tree_digest(tmp_path) == before


@pytest.mark.parametrize("damage, error, message", [
    ("p5_2_eval_byte", ValueError, "does not match its line"),
    ("declaration_byte", ValueError, "pinned"),
    ("p8_4b_cell_list", ValueError, "declared_cells_sha256"),
    ("manifest_exists", FileExistsError, "written once"),
])
def test_check_refuses_damaged_inputs(tmp_path: Path, monkeypatch: Any, damage: str, error: type,
                                      message: str) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    output = record.roots.output_root
    if damage == "p5_2_eval_byte":
        path = output / "p5_2" / "eval_random_iql.json"
        path.write_text(path.read_text() + " ")
    elif damage == "declaration_byte":
        path = record.roots.repo_root / "docs" / "data" / "p5_2_declaration_random.json"
        path.write_text(path.read_text() + " ")
    elif damage == "p8_4b_cell_list":
        path = output / "p8_4b_rederivation" / "campaign_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["cells"] = manifest["cells"][:-1]
        path.write_text(json.dumps(manifest))
    else:
        (output / "SHA256SUMS_p5_2b.txt").write_text("x\n")
    with pytest.raises(error, match=message):
        ic.check(record.roots, pins=record.pins, require_cuda=False)


def test_check_refuses_a_protocol_p5_2s_evaluate_would_not_run(tmp_path: Path, monkeypatch: Any) -> None:
    import offline.dt_gate as dt_gate

    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    monkeypatch.setattr(dt_gate, "HELD_OUT_DRAWS", tuple(record.draws[:-1]) + (1099,))
    with pytest.raises(ValueError, match="HELD_OUT_DRAWS"):
        ic.check(record.roots, pins=record.pins, require_cuda=False)


# ======================================================================
# T-train-args and T-record (plan F10)
# ======================================================================


def _train_baselines_args(tmp_path: Path, corpus: Path) -> Any:
    return ts.build_parser().parse_args([
        "--corpus-root", str(corpus), "--work-dir", str(tmp_path / "work"),
        "--checkpoint-dir", str(tmp_path / "checkpoints"), "--reuse-root", str(tmp_path / "protected"),
        "--device", "cpu", "--seeds", "101", "--gradient-steps", "2", "--tier", "random", "train-baselines",
    ])


def test_t_train_args_the_module_hands_train_iql_what_train_baselines_hands_it(tmp_path: Path,
                                                                               monkeypatch: Any) -> None:
    corpus = tmp_path / "corpus"
    fx.write_synthetic_tier(corpus)
    fx.install_synthetic_tier(monkeypatch)
    iql, _ = fx.install_train_spies(monkeypatch)
    assert ts._run_train_baselines(_train_baselines_args(tmp_path, corpus)) == 0
    reference = iql.calls[-1]
    inputs = ic.training_inputs(corpus)
    ic.train_seeds(inputs, seeds=(101,), gradient_steps=2, device=torch.device("cpu"),
                   checkpoint_dir=tmp_path / "mine", protected=(), correction={"brief": "BRIEF_44"}, log_every=0)
    mine = iql.calls[-1]
    for name in ("state", "next_state", "action", "reward", "stream_index", "t"):
        assert torch.equal(getattr(mine["batch"], name), getattr(reference["batch"], name)), name
    assert mine["batch"].reward_scale == reference["batch"].reward_scale
    assert set(mine) == set(reference), "a keyword argument was added or dropped"
    for key in ("state_dim", "n_actions", "seed", "declared_gradient_steps", "batch_size", "scenario_id", "log_every"):
        assert mine[key] == reference[key], key
    assert str(mine["device"]) == str(reference["device"]) == "cpu"
    assert mine["stats"].to_json_obj() == reference["stats"].to_json_obj()
    theirs = {k: v for k, v in reference["provenance"].items() if k != "runtime"}
    ours = {k: v for k, v in mine["provenance"].items() if k not in ("runtime", "correction")}
    assert ours == theirs
    assert mine["provenance"]["correction"] == {"brief": "BRIEF_44"}


def test_t_record_train_seeds_moves_each_checkpoint_into_place_and_records_its_seconds(tmp_path: Path,
                                                                                       monkeypatch: Any) -> None:
    corpus = tmp_path / "corpus"
    fx.write_synthetic_tier(corpus)
    fx.install_synthetic_tier(monkeypatch)
    iql, _ = fx.install_train_spies(monkeypatch)
    inputs = ic.training_inputs(corpus)
    directory = tmp_path / "out" / "p5_2" / "checkpoints"
    runs = ic.train_seeds(inputs, seeds=(101, 202), gradient_steps=3, device=torch.device("cpu"),
                          checkpoint_dir=directory, protected=(), correction={"brief": "BRIEF_44"})
    assert [run["seed"] for run in runs] == [101, 202]
    assert sorted(p.name for p in directory.iterdir()) == ["grid4x4_random_iql_seed101.pt",
                                                           "grid4x4_random_iql_seed202.pt"]
    for run, call in zip(runs, iql.calls):
        path = directory / f"grid4x4_random_iql_seed{run['seed']}.pt"
        assert Path(call["checkpoint_path"]).name == path.name + ".partial"
        assert run["checkpoint_path"] == str(path)
        assert run["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert run["state_dict_sha256"] == ts.canonical_state_dict_digest(path)
        assert run["seconds"] == 10.0 + run["seed"] / 100.0 and run["final_loss"] == 1.0 + run["seed"] / 1000.0
        assert run["training_rows"] == len(inputs.table) and run["gradient_steps"] == 3


def test_t_record_train_seeds_refuses_a_checkpoint_already_in_place(tmp_path: Path, monkeypatch: Any) -> None:
    corpus = tmp_path / "corpus"
    fx.write_synthetic_tier(corpus)
    fx.install_synthetic_tier(monkeypatch)
    iql, _ = fx.install_train_spies(monkeypatch)
    inputs = ic.training_inputs(corpus)
    directory = tmp_path / "out"
    directory.mkdir()
    (directory / "grid4x4_random_iql_seed101.pt").write_bytes(b"earlier")
    with pytest.raises(FileExistsError, match="written once"):
        ic.train_seeds(inputs, seeds=(101,), gradient_steps=3, device=torch.device("cpu"), checkpoint_dir=directory,
                       protected=(), correction={})
    assert iql.calls == [] and (directory / "grid4x4_random_iql_seed101.pt").read_bytes() == b"earlier"


# ======================================================================
# T-route
# ======================================================================


def test_t_route_p8_4bs_resolver_under_an_out_root_is_that_out_roots_checkpoint_dir(tmp_path: Path) -> None:
    """Renamed under Amendment B, B1.8 (RA1: the old name claimed the stage's choice of root): this checks A1.1's
    premise -- P8.4b's resolver under a root reads that root's checkpoint directory; the stage's choice of root is held
    by B1.6(c)'s resolver test and the whole-run test."""
    roots = _fake_roots(tmp_path)
    resolved = ic.resolved_checkpoints(roots, resolver_root=roots.out_root, seeds=(101, 202))
    for seed, path in resolved.items():
        assert path == ic.checkpoint_dir(roots.out_root) / f"grid4x4_random_iql_seed{seed}.pt"
        assert path.name == ts._checkpoint_name("random", "iql", seed)


def test_t_route_an_evaluation_of_any_other_model_is_refused(tmp_path: Path, monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record)
    out = record.roots.out_root
    training = json.loads((out / "training_random_iql.json").read_text())
    payload = json.loads((out / "eval_random_iql.json").read_text())
    ic.assert_evaluated_models(payload, training)
    for damage in ("digest", "path", "seed"):
        broken = json.loads(json.dumps(payload))
        if damage == "digest":
            broken["model_provenance"]["101"]["state_dict_sha256"] = "0" * 64
        elif damage == "path":
            broken["model_provenance"]["202"]["checkpoint_path"] = "/elsewhere/grid4x4_random_iql_seed202.pt"
        else:
            del broken["model_provenance"]["202"]
        with pytest.raises(ValueError, match="model_provenance"):
            ic.assert_evaluated_models(broken, training)


# ======================================================================
# The protocol, on a synthetic record
# ======================================================================


def test_original_protocol_reads_a_full_grid_and_refuses_anything_else(tmp_path: Path) -> None:
    record = fx.write_synthetic_record(tmp_path)
    payload = json.loads((record.roots.output_root / "p5_2" / "eval_random_iql.json").read_text())
    protocol = ic.original_protocol(payload)
    assert protocol == ic.Protocol(seeds=record.seeds, draw_ids=record.draws, engine_seed=1000, deterministic=False,
                                   declared_gradient_steps=40000)
    for damage, message in (("duplicate", "full grid"), ("missing", "full grid"), ("arm", "iql@random")):
        broken = json.loads(json.dumps(payload))
        if damage == "duplicate":
            broken["episodes"].append(dict(broken["episodes"][0]))
        elif damage == "missing":
            broken["episodes"].pop()
        else:
            broken["arm"] = "bc@random"
        with pytest.raises(ValueError, match=message):
            ic.original_protocol(broken)


@pytest.mark.parametrize("damage, message", [("draws", "HELD_OUT_DRAWS"), ("seeds", "TRAINING_SEEDS"),
                                             ("deterministic", "deterministic"), ("steps", "40,000")])
def test_assert_protocol_refuses_what_p5_2s_evaluate_would_not_run(tmp_path: Path, monkeypatch: Any,
                                                                   damage: str, message: str) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    good = ic.Protocol(seeds=record.seeds, draw_ids=record.draws, engine_seed=1000, deterministic=False,
                       declared_gradient_steps=40000)
    ic.assert_protocol(good)
    bad = {"draws": ic.Protocol(record.seeds, record.draws[:-1], 1000, False, 40000),
           "seeds": ic.Protocol(record.seeds[:-1], record.draws, 1000, False, 40000),
           "deterministic": ic.Protocol(record.seeds, record.draws, 1000, True, 40000),
           "steps": ic.Protocol(record.seeds, record.draws, 1000, False, 20000)}[damage]
    with pytest.raises(ValueError, match=message):
        ic.assert_protocol(bad)


def test_the_p5_2_argv_carries_the_records_protocol_and_the_corrected_paths(tmp_path: Path, monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    roots = record.roots
    args = ts.build_parser().parse_args(ic.p5_2_evaluation_argv(roots, pins=record.pins))
    assert (args.command, args.method, args.tier) == ("evaluate", "iql", "random")
    assert (args.engine_seed, args.gradient_steps, args.deterministic) == (1000, 40000, False)
    assert args.seeds is None and args.device is None and args.torch_threads == 1 and args.replicate is False
    assert Path(args.checkpoint_dir) == ic.checkpoint_dir(roots.out_root)
    assert Path(args.work_dir) == roots.out_root
    assert Path(args.reuse_root) == roots.output_root / "p5_2"
    assert (Path(args.corpus_root), Path(args.draws_root)) == (roots.corpus_root, roots.draws_root)
    narrowed = ts.build_parser().parse_args(ic.p5_2_evaluation_argv(roots, seeds=(101,), pins=record.pins))
    assert narrowed.seeds == "101"


def test_narrowed_held_out_draws_substitutes_for_one_call_and_restores(monkeypatch: Any) -> None:
    import offline.dt_gate as dt_gate

    original = dt_gate.HELD_OUT_DRAWS
    with ic.narrowed_held_out_draws((1000, 1001)):
        assert dt_gate.HELD_OUT_DRAWS == (1000, 1001)
    assert dt_gate.HELD_OUT_DRAWS == original
    with pytest.raises(ValueError, match="held-out"):
        with ic.narrowed_held_out_draws((999,)):
            pass
    assert dt_gate.HELD_OUT_DRAWS == original


def test_the_rederivation_cells_and_the_committed_map_are_keyed_as_p8_4b_keys_them(tmp_path: Path) -> None:
    from offline.att_rederivation import CellKey

    record = fx.write_synthetic_record(tmp_path)
    cells = ic.rederivation_cells(record.seeds, record.draws)
    assert cells == [CellKey("grid4x4", "iql@random", s, d) for s in record.seeds for d in record.draws]
    payload = json.loads((record.roots.output_root / "p5_2" / "eval_random_iql.json").read_text())
    committed = ic.committed_from_eval(payload)
    assert committed == {("grid4x4", "iql@random", s, d): record.ours[("iql", "random")][(s, d)]
                         for s in record.seeds for d in record.draws}


# ======================================================================
# T-statements: every statement of plan section 6 on hand-built cells, by this file's own loops
# ======================================================================


def _cells(levels: dict[str, float], *, ties: tuple[str, ...] = ()) -> dict[tuple[str, str], list[EpisodeResult]]:
    """Hand-built cells: every Q1 cell, five seeds x four draws; *ties* copy ``bc``'s random-tier episodes exactly."""
    out: dict[tuple[str, str], list[EpisodeResult]] = {}
    for method, tier in ts.OUT_OF_SAMPLE_CELLS:
        base = levels[method] if tier == "random" else float(ts.PREDICTED_LEVELS[method][tier])
        source = "bc" if (tier == "random" and method in ties) else method
        out[(method, tier)] = [
            EpisodeResult(arm=f"{method}@{tier}", seed=seed, draw_id=draw,
                          att_horizon=float(base + ((sum(map(ord, source)) + seed + 3 * draw) % 11 - 5) / 32.0),
                          horizon_vehicle_count=0.0, episode_reward=0.0)
            for seed in (101, 202, 303, 404, 505) for draw in (1000, 1001, 1002, 1003)]
        if source != method:
            out[(method, tier)] = [EpisodeResult(f"{method}@{tier}", e.seed, e.draw_id, e.att_horizon, 0.0, 0.0)
                                   for e in out[("bc", tier)]] if ("bc", tier) in out else out[(method, tier)]
    return out


def _mean(episodes: list[EpisodeResult]) -> float:
    return float(np.asarray([e.att_horizon for e in episodes], dtype=np.float64).mean())


def _my_concordance(predicted: list[str], measured: dict[str, float], arms: list[str]) -> tuple[int, int]:
    rank = {arm: i for i, arm in enumerate(predicted)}
    concordant = ties = 0
    for a, b in itertools.combinations(sorted(arms), 2):
        if measured[a] == measured[b]:
            ties += 1
            continue
        concordant += (rank[a] < rank[b]) == (measured[a] < measured[b])
    return concordant, ties


def _my_q3c(cells: dict[tuple[str, str], list[EpisodeResult]], best: str) -> tuple[float, float, float]:
    def per_draw(method: str) -> dict[int, float]:
        grouped: dict[int, list[float]] = {}
        for e in cells[(method, "random")]:
            grouped.setdefault(e.draw_id, []).append(e.att_horizon)
        return {d: float(np.mean(v)) for d, v in grouped.items()}

    a, b = per_draw("dt_nomix"), per_draw(best)
    diffs = np.asarray([a[d] - b[d] for d in sorted(a)], dtype=np.float64)
    mean = float(diffs.mean())
    half = 1.96 * float(diffs.std(ddof=1)) / float(np.sqrt(diffs.size))
    return mean, mean - half, mean + half


RECORD_SHAPED = {"dt_spatial": 254.6, "dt_nomix": 255.3, "bc": 289.4, "bc_top10": 350.3, "bc_top10_perix": 301.2,
                 "iql": 191.0}


@pytest.mark.parametrize("case", ["record_shaped", "iql_corrected_late", "a_dt_arm_leads", "exact_tie",
                                  "nomix_strictly_lowest", "q3c_not_resolved"])
def test_t_statements_every_statement_equals_this_files_own_computation(case: str) -> None:
    levels = dict(RECORD_SHAPED)
    ties: tuple[str, ...] = ()
    if case == "iql_corrected_late":
        levels["iql"] = 320.0
    elif case == "a_dt_arm_leads":
        levels.update({"dt_spatial": 180.0, "iql": 200.0})
    elif case == "exact_tie":
        ties = ("bc_top10_perix",)
    elif case == "nomix_strictly_lowest":
        levels["dt_nomix"] = 150.0
    elif case == "q3c_not_resolved":
        levels["iql"] = levels["dt_nomix"]
    cells = _cells(levels, ties=ties)
    got = ic.random_tier_statements(cells, with_hard_subset=True)
    measured = {m: _mean(cells[(m, "random")]) for m in ts.METHODS}
    order = sorted(ts.METHODS, key=lambda m: measured[m])
    assert got["ranking"]["order"] == order
    assert got["ranking"]["levels"] == {m: measured[m] for m in ts.METHODS}
    predicted = list(ts.predicted_order("random"))
    concordant, n_ties = _my_concordance(predicted, measured, list(ts.METHODS))
    assert (got["q2b"]["n_concordant"], got["q2b"]["n_pairs"], got["q2b"]["n_tied"]) == (concordant, 15, n_ties)
    assert got["q2b"]["outcome"] == ("HELD" if concordant >= 12 else "FAILED")
    iql_pairs = {pair: ((measured[pair[0]] != measured[pair[1]]) and (
        (predicted.index(pair[0]) < predicted.index(pair[1])) == (measured[pair[0]] < measured[pair[1]])))
                 for pair in itertools.combinations(sorted(ts.METHODS), 2) if "iql" in pair}
    assert {tuple(p["pair"]): p["concordant"] for p in got["q2b"]["iql_pairs"]} == iql_pairs
    hard, _ = _my_concordance(predicted, measured, list(ts.HARD_SUBSET))
    assert got["q2b_hard"]["n_concordant"] == hard and got["q2b_hard"]["n_pairs"] == 6
    lowest = min(measured.values())
    firsts = [m for m in ts.METHODS if measured[m] == lowest]
    assert got["q2a"]["measured_first"] == order[0] and got["q2a"]["predicted_first"] == "dt_nomix"
    assert got["q2a"]["outcome"] == ("HELD" if firsts == ["dt_nomix"] else "FAILED")
    rank = 1 + sum(measured[m] < measured["dt_nomix"] for m in ts.METHODS if m != "dt_nomix")
    assert got["q3a"]["rank"] == rank and got["q3a"]["of"] == 6
    assert got["q3a"]["outcome"] == ("HELD" if all(measured[m] > measured["dt_nomix"] for m in ts.METHODS
                                                   if m != "dt_nomix") else "FAILED")
    best = min(("bc", "bc_top10", "bc_top10_perix", "iql"), key=lambda m: measured[m])
    mean, low, high = _my_q3c(cells, best)
    assert got["q3c"]["best_non_dt"] == best
    assert (got["q3c"]["mean"], got["q3c"]["ci95_low"], got["q3c"]["ci95_high"]) == (mean, low, high)
    assert got["q3c"]["reading"] == ("resolves against the DT" if low > 0 else
                                     "resolves for the DT" if high < 0 else "NOT RESOLVED")
    levels_19 = {cell: _mean(episodes) for cell, episodes in cells.items()}
    q1 = ts.score_level(levels_19)
    entry = next(c for c in q1["cells"] if c["cell"] == ["iql", "random"])
    assert got["q1"]["iql_random"] == entry
    assert (got["q1"]["n_held"], got["q1"]["threshold"], got["q1"]["outcome"]) == (q1["n_held"], q1["threshold"],
                                                                                   q1["outcome"])
    if case == "nomix_strictly_lowest":  # Amendment B.1, item 2: the HELD branches of Q2a and Q3a (M1, M2)
        assert (got["q2a"]["outcome"], got["q3a"]["outcome"], got["q3c"]["reading"]) == (
            "HELD", "HELD", "resolves for the DT")
    if case == "q3c_not_resolved":  # Amendment B.1, item 2: Q3c's CI straddles zero (M3)
        assert (got["q3c"]["best_non_dt"], got["q3c"]["reading"]) == ("iql", "NOT RESOLVED")


def test_t_statements_a_tie_for_first_is_not_a_held_first_place() -> None:
    levels = dict(RECORD_SHAPED)
    levels.update({"dt_nomix": 150.0, "dt_spatial": 150.0})
    cells = _cells(levels)
    cells[("dt_spatial", "random")] = [EpisodeResult("dt_spatial@random", e.seed, e.draw_id, e.att_horizon, 0.0, 0.0)
                                       for e in cells[("dt_nomix", "random")]]
    got = ic.random_tier_statements(cells, with_hard_subset=False)
    assert sorted(got["q2a"]["tied_for_first"]) == ["dt_nomix", "dt_spatial"]
    assert got["q2a"]["outcome"] == "FAILED" and got["q3a"]["outcome"] == "FAILED" and got["q3a"]["rank"] == 1
    assert "q2b_hard" not in got


# ======================================================================
# The report, on a synthetic record and run
# ======================================================================


def _git() -> dict[str, Any]:
    return {"commit": "0" * 40, "dirty": False}


def test_the_report_states_the_original_and_the_corrected_cell_under_both_definitions(tmp_path: Path,
                                                                                      monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    corrected = fx.write_synthetic_run(record, iql_shift=60.0)
    ic.write_run_manifest(record.roots, ic.protected_roots(record.roots))
    artifact = ic.build_report(record.roots, git=_git(), pins=record.pins)
    assert artifact["format_version"] == "p5.2b-correction/1.0"
    original = record.ours[("iql", "random")]
    order = sorted(original)
    assert artifact["cells"]["original"]["att_ours"]["mean"] == float(np.mean([original[k] for k in order]))
    assert artifact["cells"]["corrected"]["att_ours"]["mean"] == float(np.mean([corrected["att_ours"][k] for k in order]))
    assert artifact["cells"]["corrected"]["att_engine"]["mean"] == float(
        np.mean([corrected["att_engine"][k] for k in order]))
    assert artifact["cells"]["corrected"]["paths_agree"] == {"episodes": len(order), "equal": len(order)}
    for definition, source in (("att_ours", record.ours), ("att_engine", record.engine)):
        before = artifact["statements"][definition]["before"]
        after = artifact["statements"][definition]["after"]
        assert before["ranking"]["levels"]["iql"] == float(np.mean([source[("iql", "random")][k] for k in order]))
        expected_after = corrected[definition]
        assert after["ranking"]["levels"]["iql"] == float(np.mean([expected_after[k] for k in order]))
        assert before["ranking"]["levels"]["bc"] == after["ranking"]["levels"]["bc"]
    assert "q2b_hard" in artifact["statements"]["att_ours"]["before"]
    assert "q2b_hard" not in artifact["statements"]["att_engine"]["before"]
    assert artifact["statements"]["att_ours"]["before"]["q2b_hard"] == artifact["statements"]["att_ours"]["after"][
        "q2b_hard"]
    defect = artifact["defect"]
    assert defect["original"]["training_rows"] == len(fx.CORPUS_EPISODES) * 16 * fx.EPISODE_LENGTH
    assert defect["declared"]["rows"] == len(fx.SELECTED) * 16 * fx.EPISODE_LENGTH
    assert [run["seed"] for run in artifact["training"]["runs"]] == list(record.seeds)
    assert artifact["training"]["canaries"]["open"]["verdict"] == "at speed"
    said = " ".join(artifact["what_this_does_not_say"])
    for needle in ("not re-run", "DEFERRED 109", "record of the defect", "one realisation"):
        assert needle in said, needle


def test_the_report_with_the_original_cell_in_place_of_the_corrected_one_changes_no_statement(
    tmp_path: Path, monkeypatch: Any
) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record, iql_shift=0.0)
    ic.write_run_manifest(record.roots, ic.protected_roots(record.roots))
    artifact = ic.build_report(record.roots, git=_git(), pins=record.pins)
    for definition in ("att_ours", "att_engine"):
        assert artifact["statements"][definition]["before"] == artifact["statements"][definition]["after"], definition


@pytest.mark.parametrize("damage, error, message", [
    ("cell_disagrees", ValueError, "does not reproduce"),
    ("canary_missing", FileNotFoundError, "canary_close.json"),
    ("manifest_missing", FileNotFoundError, "SHA256SUMS_p5_2b.txt"),
    ("p5_2_file_changed", ValueError, "does not match its line"),
])
def test_the_report_refuses_an_input_it_cannot_verify(tmp_path: Path, monkeypatch: Any, damage: str, error: type,
                                                     message: str) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record)
    out = record.roots.out_root
    if damage != "manifest_missing":
        ic.write_run_manifest(record.roots, ic.protected_roots(record.roots))
    if damage == "cell_disagrees":
        cell = sorted((out / "rederivation").glob("cell_*.json"))[0]
        payload = json.loads(cell.read_text())
        payload["att_ours"] += 0.5
        cell.write_text(json.dumps(payload))
    elif damage == "canary_missing":
        (out / "canary_close.json").unlink()
    elif damage == "p5_2_file_changed":
        path = record.roots.output_root / "p5_2" / "eval_random_bc.json"
        path.write_text(path.read_text() + " ")
    with pytest.raises(error, match=message):
        ic.build_report(record.roots, git=_git(), pins=record.pins)


def test_the_report_is_written_once_under_the_out_roots_artifacts(tmp_path: Path, monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record)
    protected = ic.protected_roots(record.roots)
    ic.write_run_manifest(record.roots, protected)
    artifact = ic.build_report(record.roots, git=_git(), pins=record.pins)
    path = ic.write_report(record.roots, artifact, protected)
    assert path == record.roots.out_root / "artifacts" / "p5_2b_correction.json"
    assert path.read_text(encoding="utf-8") == json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with pytest.raises(FileExistsError, match="written once"):
        ic.write_report(record.roots, artifact, protected)


# ======================================================================
# The pre-flight's arithmetic
# ======================================================================


def test_the_preflight_estimate_and_its_timeouts() -> None:
    got = ic.preflight_estimate(table_seconds=120.0, rehearsal_steps=2000, rehearsal_seconds=20.0,
                                episode_seconds_p5_2=2.5, episode_seconds_p8_4b=3.0, n_seeds=5,
                                declared_steps=40000, n_episodes=500)
    assert got["training_seconds"] == 120.0 + 5 * 40000 * (20.0 / 2000)
    assert (got["evaluate_p5_2_seconds"], got["evaluate_p8_4b_seconds"]) == (500 * 2.5, 500 * 3.0)
    assert got["timeouts"] == {"train": 3.0 * got["training_seconds"], "evaluate_p5_2": 3.0 * 1250.0,
                               "evaluate_p8_4b": 3.0 * 1500.0}
    small = ic.preflight_estimate(table_seconds=1.0, rehearsal_steps=2000, rehearsal_seconds=0.1,
                                  episode_seconds_p5_2=0.01, episode_seconds_p8_4b=0.01, n_seeds=5,
                                  declared_steps=40000, n_episodes=500)
    assert set(small["timeouts"].values()) == {600.0}


def test_timeouts_come_only_from_a_complete_preflight_whose_reproductions_held(tmp_path: Path) -> None:
    def write(status: str, a: bool, b: bool) -> Path:
        path = tmp_path / f"{status}_{a}_{b}.json"
        path.write_text(json.dumps({"format_version": "p5.2b-preflight/1.0", "status": status,
                                    "reproduce": {"p5_2_path": {"reproduced": a}, "p8_4b_path": {"reproduced": b}},
                                    "estimate": {"timeouts": {"train": 9000.0, "evaluate_p5_2": 4000.0,
                                                              "evaluate_p8_4b": 4500.0}}}))
        return path

    assert ic.timeouts_from_preflight(write("COMPLETE", True, True)) == {
        "train": 9000.0, "evaluate_p5_2": 4000.0, "evaluate_p8_4b": 4500.0}
    for status, a, b, message in (("FAILED", True, True, "COMPLETE"), ("COMPLETE", False, True, "reproduce"),
                                  ("COMPLETE", True, False, "reproduce")):
        with pytest.raises(ValueError, match=message):
            ic.timeouts_from_preflight(write(status, a, b))


# ======================================================================
# Gated, on the real record
# ======================================================================


@pytest.fixture(scope="module")
def real_training_inputs() -> Any:
    return ic.training_inputs(_corpus_root())


def _original_checkpoint(seed: int) -> dict[str, Any]:
    return torch.load(_output_root() / "p5_2" / "checkpoints" / f"grid4x4_random_iql_seed{seed}.pt",
                      map_location="cpu", weights_only=False)


def test_t_rows_the_corrected_table_holds_exactly_the_declared_streams_by_a_second_route(
    real_training_inputs: Any,
) -> None:
    """T-rows (load-bearing): 1,152,000 transitions, the stream set equal to BC's ``streams`` and to the declaration's
    ``selected_episodes`` x ``node_order`` -- the second route read HERE with ``json``, never through the loader."""
    from offline.method_tier_grid import transition_stream_keys

    inputs = real_training_inputs
    declaration = json.loads((REPO_ROOT / "docs" / "data" / "p5_2_declaration_random.json").read_text())
    manifest = json.loads((_corpus_root() / "cf_grid4x4__random" / "manifest.json").read_text())
    length = {e["filename"]: int(e["episode_length"]) for e in manifest["episodes"]}
    selected = declaration["selected_episodes"]
    expected_rows = sum(length[e["episode_file"]] for e in selected) * len(declaration["node_order"])
    assert expected_rows == 1_152_000 == len(selected) * 16 * 360
    assert len(inputs.table) == expected_rows

    def key(directory: str, episode: str, ix: str) -> tuple[str, str, str]:
        return (Path(directory).name, episode, ix)

    declared = {key(e["dataset_dir"], e["episode_file"], ix) for e in selected for ix in declaration["node_order"]}
    keys = transition_stream_keys(inputs.parts["dataset"], inputs.group)
    present = {key(*keys[int(i)]) for i in np.unique(inputs.table.stream_index.numpy())}
    assert present == declared == {key(*stream.key) for stream in inputs.streams}
    assert len(declared) == declaration["training_streams"] == 3200


def test_t_scale_and_t_stats_the_inputs_are_p5_2s_but_for_the_rows(real_training_inputs: Any) -> None:
    """The reward scale and the normalisation statistics equal what all five original checkpoints record: the
    correction changes the transition table and nothing else of IQL's inputs (plan V2, V3)."""
    for seed in (101, 202, 303, 404, 505):
        original = _original_checkpoint(seed)
        assert real_training_inputs.scale == original["provenance"]["diagnostics"]["reward_scale"], seed
        assert real_training_inputs.parts["dataset"].stats.to_json_obj() == original["stats"], seed
        assert original["provenance"]["diagnostics"]["training_rows"] == 2_304_000, seed


def test_t_protocol_the_evaluation_runs_p5_2s_recorded_protocol(tmp_path: Path, monkeypatch: Any,
                                                               keep_torch_threads: Any) -> None:
    """T-protocol (load-bearing), path (i): with ``evaluate_arm`` replaced by a spy, the draws, seeds and engine seed it
    receives and the written file's header equal ``output/p5_2/eval_random_iql.json``'s, read HERE."""
    import offline.dt_gate as dt_gate

    output = _output_root()
    original = json.loads((output / "p5_2" / "eval_random_iql.json").read_text())
    seen: list[dict[str, Any]] = []

    def spy(**kwargs: Any) -> list[EpisodeResult]:
        seen.append(kwargs)
        return [EpisodeResult(kwargs["arm"], kwargs["seed"], int(d), 100.0, 1.0, -1.0) for d in kwargs["draw_ids"]]

    monkeypatch.setattr(dt_gate, "evaluate_arm", spy)
    roots = _real_roots()
    argv = ic.p5_2_evaluation_argv(roots, checkpoint_dir=tmp_path / "checkpoints", work_dir=tmp_path / "work")
    assert ic.run_p5_2_evaluation(argv) == 0
    episodes = original["episodes"]
    seeds = sorted({int(e["seed"]) for e in episodes})
    assert [call["seed"] for call in seen] == seeds
    for call in seen:
        assert list(call["draw_ids"]) == sorted(int(e["draw_id"]) for e in episodes if int(e["seed"]) == call["seed"])
        assert call["engine_seed"] == original["engine_seed"] and call["arm"] == original["arm"]
    written = json.loads((tmp_path / "work" / "eval_random_iql.json").read_text())
    for field in ("engine_seed", "deterministic", "declared_gradient_steps", "arm", "method", "tier", "format_version"):
        assert written[field] == original[field], field


def test_t_protocol_the_rederivation_runs_p8_4bs_recorded_cells(tmp_path: Path) -> None:
    """T-protocol, path (ii): the module's 500 cells and engine seed equal P8.4b's committed campaign's, read HERE."""
    output = _p8_4b_cells()
    manifest = json.loads((output / "p8_4b_rederivation" / "campaign_manifest.json").read_text())
    original = json.loads((output / "p5_2" / "eval_random_iql.json").read_text())
    protocol = ic.original_protocol(original)
    cells = ic.rederivation_cells(protocol.seeds, protocol.draw_ids)
    theirs = sorted(key for key in manifest["cells"] if key.startswith("grid4x4|iql@random|"))
    assert sorted(f"{c.scenario}|{c.arm}|{c.seed}|{c.draw_id}" for c in cells) == theirs
    assert len(theirs) == 500 and protocol.engine_seed == manifest["engine_seed"] == 1000


T_REPRODUCE_DRAWS = (1000, 1001, 1002, 1003, 1004)


def test_t_reproduce_a_p5_2s_path_on_the_original_checkpoint_reproduces_the_committed_episodes(
    tmp_path: Path, keep_torch_threads: Any
) -> None:
    """T-reproduce (a), load-bearing: P5.2's evaluate subcommand, through the module's argv, on the ORIGINAL
    ``grid4x4_random_iql_seed101.pt`` and draws 1000-1004, reproduces the committed episodes under ``==``."""
    _cityflow()
    output = _output_root()
    roots = _real_roots()
    argv = ic.p5_2_evaluation_argv(roots, seeds=(101,), checkpoint_dir=output / "p5_2" / "checkpoints",
                                   work_dir=tmp_path)
    assert ic.run_p5_2_evaluation(argv, draws=T_REPRODUCE_DRAWS) == 0
    rerun = json.loads((tmp_path / "eval_random_iql_seed101.json").read_text())["episodes"]
    committed = {int(e["draw_id"]): e for e in json.loads((output / "p5_2" / "eval_random_iql.json").read_text())[
        "episodes"] if int(e["seed"]) == 101}
    assert sorted(int(e["draw_id"]) for e in rerun) == list(T_REPRODUCE_DRAWS)
    for episode in rerun:
        theirs = committed[int(episode["draw_id"])]
        for field in ("att_horizon", "horizon_vehicle_count", "episode_reward"):
            assert episode[field] == theirs[field], (episode["draw_id"], field)


def test_t_reproduce_b_p8_4bs_runner_on_the_original_checkpoint_reproduces_the_committed_cells(tmp_path: Path) -> None:
    """T-reproduce (b), load-bearing: P8.4b's runner, through the module, on the ORIGINAL checkpoint and draws
    1000-1004, reproduces P8.4b's committed cells field by field under ``==`` (timings excepted)."""
    from offline.att_rederivation import cell_file_name

    _cityflow()
    output = _p8_4b_cells()
    roots = _real_roots(out_root=output / "p5_2b")
    original = json.loads((output / "p5_2" / "eval_random_iql.json").read_text())
    cells = [c for c in ic.rederivation_cells((101,), T_REPRODUCE_DRAWS)]
    work = tmp_path / "rederivation"
    protected = tuple(Path(p).resolve() for p in (output / "p5_2", output / "p8_4b_rederivation", _corpus_root()))
    result = ic.run_rederivation(cells, roots=roots, resolver_root=output, work_dir=work,
                                 committed=ic.committed_from_eval(original), protected=protected, engine_seed=1000)
    assert result["outcome"]["n_refused"] == 0 and result["status"]["complete"] is True
    for cell in cells:
        mine = json.loads((work / cell_file_name(cell)).read_text())
        theirs = json.loads((output / "p8_4b_rederivation" / cell_file_name(cell)).read_text())
        for timing in ("seconds", "seconds_rollout"):
            mine.pop(timing)
            theirs.pop(timing)
        assert mine == theirs, cell


def _raw_cells(output: Path, definition: str) -> dict[tuple[str, str], dict[tuple[int, int], float]]:
    """P8.4b's cell files read by THIS test: glob, json, its own grouping."""
    out: dict[tuple[str, str], dict[tuple[int, int], float]] = {}
    for method, tier in ts.OUT_OF_SAMPLE_CELLS:
        for path in (output / "p8_4b_rederivation").glob(f"cell_grid4x4_{method}_at_{tier}_seed*_draw*.json"):
            row = json.loads(path.read_text())
            if row["method"] != method:
                continue
            out.setdefault((method, tier), {})[(int(row["seed"]), int(row["draw_id"]))] = float(row[definition])
    return out


def test_t_reproduce_c_the_recomputation_fed_the_original_files_reproduces_every_committed_statement() -> None:
    """T-reproduce (c), att_ours: fed P5.2's committed files, every statement equals its committed value at the
    precision it was committed at, and every level equals the eval file's own ``cell.att_horizon_mean``."""
    output = _output_root()
    sums = ic.p5_2_sums(output)
    cells = ic.p5_2_cells(output, sums)
    for (method, tier), episodes in cells.items():
        committed = json.loads((output / "p5_2" / f"eval_{tier}_{method}.json").read_text())["cell"]
        assert ic.levels_of({(method, tier): episodes})[(method, tier)] == committed["att_horizon_mean"]
    got = ic.random_tier_statements(cells, with_hard_subset=True)
    entry = got["q1"]["iql_random"]
    assert f"{entry['measured']:.2f}" == COMMITTED["q1_iql_random_measured_2dp"]
    assert entry["predicted"] == COMMITTED["q1_iql_random_predicted"] and entry["held"] is False
    assert (got["q1"]["n_held"], got["q1"]["threshold"], got["q1"]["outcome"]) == (15, 14, "HELD")
    assert got["q1"]["n_cells"] == 19, "Q1 is scored over P5.2's 19 out-of-sample cells (Amendment B, B1.7(a))"
    assert got["ranking"]["order"] == COMMITTED["ranking"]
    assert (got["q2a"]["measured_first"], got["q2a"]["outcome"]) == (COMMITTED["q2a_first"], "FAILED")
    assert (got["q2b"]["n_concordant"], got["q2b"]["outcome"]) == (COMMITTED["q2b_concordant"], "FAILED")
    assert got["q2b_hard"]["n_concordant"] == COMMITTED["q2b_hard_concordant"]
    assert (got["q3a"]["rank"], got["q3a"]["outcome"]) == (COMMITTED["q3a_rank"], "FAILED")
    q3c = got["q3c"]
    assert q3c["best_non_dt"] == COMMITTED["q3c_best"]
    assert f"{q3c['mean']:+.4f} [{q3c['ci95_low']:+.4f}, {q3c['ci95_high']:+.4f}]" == COMMITTED["q3c"]
    assert q3c["reading"] == "resolves against the DT"


def test_t_reproduce_c_the_att_engine_half_equals_this_files_own_recomputation_and_the_coordinators_note() -> None:
    """T-reproduce (c), att_engine: the module's levels, order and statements from P8.4b's ORIGINAL cells equal this
    test's own recomputation from the raw cell files (the numpy route, its own reading and grouping), and the six
    random-tier means equal ``docs/notes/readme_2026-10-05/c1_rule_r.json``'s under ``statistics.mean``, its route."""
    output = _p8_4b_cells()
    sums = ic.p5_2_sums(output)
    reference = ic.p5_2_cells(output, sums)
    engine = ic.p8_4b_cells(output, reference, definition="att_engine")
    raw = _raw_cells(output, "att_engine")
    assert set(raw) == set(engine) and all(len(v) == 500 for v in raw.values())
    levels = ic.levels_of(engine)
    for cell, values in raw.items():
        ordered = [values[key] for key in sorted(values)]
        assert levels[cell] == float(np.asarray(ordered, dtype=np.float64).mean()), cell
    note = json.loads((REPO_ROOT / "docs" / "notes" / "readme_2026-10-05" / "c1_rule_r.json").read_text())
    theirs = note["tiers"]["grid4x4/random"]["att_engine"]
    for method in ts.METHODS:
        assert statistics.mean(raw[(method, "random")].values()) == theirs["means"][method], method
    got = ic.random_tier_statements(engine, with_hard_subset=False)
    assert got["ranking"]["order"] == theirs["order"]
    measured = {m: levels[(m, "random")] for m in ts.METHODS}
    concordant, n_ties = _my_concordance(list(ts.predicted_order("random")), measured, list(ts.METHODS))
    assert (got["q2b"]["n_concordant"], got["q2b"]["n_tied"]) == (concordant, n_ties)
    best = min(("bc", "bc_top10", "bc_top10_perix", "iql"), key=lambda m: measured[m])
    episodes = {(m, "random"): [EpisodeResult(f"{m}@random", s, d, raw[(m, "random")][(s, d)], 0.0, 0.0)
                                for (s, d) in sorted(raw[(m, "random")])] for m in ts.METHODS}
    assert (got["q3c"]["mean"], got["q3c"]["ci95_low"], got["q3c"]["ci95_high"]) == _my_q3c(episodes, best)
    # Amendment B, B1.7(b): Q1, Q2a, Q3a, IQL's five pairs and Q3c's reading, each against this test's own route.
    mine = {cell: float(np.asarray([values[key] for key in sorted(values)], dtype=np.float64).mean())
            for cell, values in raw.items()}
    q1 = ts.score_level(mine)
    assert got["q1"]["iql_random"] == next(c for c in q1["cells"] if c["cell"] == ["iql", "random"])
    assert (got["q1"]["n_held"], got["q1"]["n_cells"], got["q1"]["outcome"]) == (q1["n_held"], 19, q1["outcome"])
    own = {m: mine[(m, "random")] for m in ts.METHODS}
    firsts = [m for m in ts.METHODS if own[m] == min(own.values())]
    assert (got["q2a"]["measured_first"], got["q2a"]["outcome"]) == (
        sorted(ts.METHODS, key=lambda m: own[m])[0], "HELD" if firsts == ["dt_nomix"] else "FAILED")
    lower = sum(own[m] < own["dt_nomix"] for m in ts.METHODS if m != "dt_nomix")
    assert (got["q3a"]["rank"], got["q3a"]["outcome"]) == (1 + lower, "HELD" if all(
        own[m] > own["dt_nomix"] for m in ts.METHODS if m != "dt_nomix") else "FAILED")
    predicted = list(ts.predicted_order("random"))
    pairs = {pair: own[pair[0]] != own[pair[1]] and (
        (predicted.index(pair[0]) < predicted.index(pair[1])) == (own[pair[0]] < own[pair[1]]))
             for pair in itertools.combinations(sorted(ts.METHODS), 2) if "iql" in pair}
    assert {tuple(p["pair"]): p["concordant"] for p in got["q2b"]["iql_pairs"]} == pairs
    _, low, high = _my_q3c(episodes, best)
    assert got["q3c"]["reading"] == ("resolves against the DT" if low > 0 else
                                     "resolves for the DT" if high < 0 else "NOT RESOLVED")


# ======================================================================
# The run's stages end to end, on a synthetic record built FROM a synthetic tier: the real stage functions, the real
# loader, the fix and P5.2's evaluate subcommand and P8.4b's runner, with spies only where the GPU trainer and the
# CityFlow engine would run.  Added after the stages were written (C6), and proven by mutation, not by being red first.
# ======================================================================


def _engine_att(seed: int, draw: int) -> float:
    """One episode's value for both stand-in engines, exact in binary."""
    return 200.0 + seed / 64.0 + (draw % 7) / 8.0


def _install_engine(monkeypatch: Any) -> None:
    import offline.admission_probe as ap
    import offline.att_rederivation as ar
    import offline.dt_gate as dt_gate

    def evaluate_arm(**kwargs: Any) -> list[EpisodeResult]:
        return [EpisodeResult(kwargs["arm"], kwargs["seed"], int(draw), _engine_att(int(kwargs["seed"]), int(draw)),
                              20.0, -4000.0) for draw in kwargs["draw_ids"]]

    def probe_episode(**kwargs: Any) -> Any:
        att = _engine_att(int(kwargs["seed"]), int(kwargs["draw_id"]))
        created = int(kwargs["created"])
        return ap.AdmissionEpisode(
            scenario=kwargs["scenario"], tier=kwargs["tier"], method=kwargs["method"], arm=kwargs["arm"],
            seed=kwargs["seed"], draw_id=int(kwargs["draw_id"]), created=created, entered=created - 1, never_entered=1,
            entered_fraction=(created - 1) / created, completed_at_horizon=created - 21, running_at_horizon=20,
            waiting_at_horizon=0, att_ours=att, att_engine=att - 6.5, horizon_vehicle_count=20.0,
            episode_reward=-4000.0, seconds=0.01, seconds_rollout=0.01)

    monkeypatch.setattr(dt_gate, "evaluate_arm", evaluate_arm)
    monkeypatch.setattr(dt_gate, "env_settings_from_manifest", lambda path: {"max_steps": 360, "delta_time": 10})
    monkeypatch.setattr(ap, "probe_episode", probe_episode)
    monkeypatch.setattr(ap, "created_from_flow", lambda path, *, horizon_seconds: 1300)
    monkeypatch.setattr(ar, "rederivation_env_settings", lambda scenario, tier, roots: {"max_steps": 360,
                                                                                       "delta_time": 10})


def _write_canary(out: Path, phase: str, seconds: float = 0.7) -> None:
    from offline import compute_latency as cl

    out.mkdir(parents=True, exist_ok=True)
    record = cl.build_canary_record(phase, seconds, reproduced=True, git={"commit": "0" * 40, "dirty": False},
                                    power=fx.POWER)
    (out / f"canary_{phase}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_run_setup(tmp_path: Path, monkeypatch: Any) -> tuple[Any, Any]:
    corpus = tmp_path / "corpus"
    fx.write_synthetic_tier(corpus)
    fx.install_synthetic_tier(monkeypatch)
    record = fx.write_synthetic_record(tmp_path, tier_corpus=corpus)
    fx.install_synthetic_protocol(monkeypatch, record)
    iql, _ = fx.install_train_spies(monkeypatch)
    _install_engine(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "synthetic GPU")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    return record, iql


def test_the_whole_run_on_a_synthetic_record_through_every_stage(tmp_path: Path, monkeypatch: Any,
                                                                 keep_torch_threads: Any) -> None:
    record, iql = _synthetic_run_setup(tmp_path, monkeypatch)
    roots, pins, output, out = record.roots, record.pins, record.roots.output_root, record.roots.out_root
    before = {name: _tree_digest(output / name) for name in ("p5_2", "p8_4b_rederivation")}
    assert ic.check(roots, pins=pins, require_cuda=False)["state"] == "fresh"
    with pytest.raises(ValueError, match="needs the complete training"):
        ic.evaluate_p5_2_stage(roots, pins=pins)
    _write_canary(out, "open")
    ic.train_stage(roots, pins=pins)
    _write_canary(out, "close")
    assert ic.evaluate_p5_2_stage(roots, pins=pins) == out / "eval_random_iql.json"
    assert ic.evaluate_p8_4b_stage(roots, pins=pins) == out / "rederivation"
    state = ic.status(roots, pins=pins)
    assert {state[stage] for stage in ("training", "canaries", "p5_2_eval", "rederivation")} == {"complete"}
    assert ic.check(roots, pins=pins, require_cuda=False)["state"] == "resumable"
    ic.write_run_manifest(roots, ic.protected_roots(roots))
    artifact = json.loads(ic.report_stage(roots, pins=pins).read_text(encoding="utf-8"))

    declared_rows = len(fx.DRAWS) * len(fx.IDS) * fx.DECISIONS
    training = json.loads((out / "training_random_iql.json").read_text(encoding="utf-8"))
    assert training["table"]["rows"] == declared_rows and [run["seed"] for run in training["runs"]] == list(record.seeds)
    assert [run["seconds"] for run in training["runs"]] == [10.0 + seed / 100.0 for seed in record.seeds]
    assert len(iql.calls) == len(record.seeds) and all(len(call["batch"]) == declared_rows for call in iql.calls)
    assert training["concurrency"]["value"] == 1 and training["regime"]["cublas_workspace_config"] is None
    values = [_engine_att(seed, draw) for seed in record.seeds for draw in record.draws]
    corrected = artifact["cells"]["corrected"]
    assert corrected["att_ours"]["mean"] == float(np.mean(values))
    assert corrected["att_engine"]["mean"] == float(np.mean([value - 6.5 for value in values]))
    assert corrected["paths_agree"] == {"episodes": len(values), "equal": len(values)}
    assert artifact["defect"]["original"]["training_rows"] == 2 * declared_rows
    assert artifact["defect"]["declared"]["rows"] == declared_rows
    assert artifact["statements"]["att_ours"]["after"]["ranking"]["levels"]["iql"] == float(np.mean(values))
    assert artifact["evaluation"]["p8_4b_path"]["n_cells"] == len(values)
    assert {name: _tree_digest(output / name) for name in before} == before
    with pytest.raises(ValueError, match="training stage is complete"):
        ic.train_stage(roots, pins=pins)


@pytest.mark.parametrize("damage, error, message", [
    ("no_canary", FileNotFoundError, "opening canary"),
    ("throttled", ValueError, "not at reference"),
    ("no_cuda", RuntimeError, "CUDA"),
    ("cublas", RuntimeError, "regime"),
])
def test_the_training_stage_refuses_to_start_where_it_cannot_vouch_for_its_seconds(
    tmp_path: Path, monkeypatch: Any, keep_torch_threads: Any, damage: str, error: type, message: str
) -> None:
    record, iql = _synthetic_run_setup(tmp_path, monkeypatch)
    out = record.roots.out_root
    if damage != "no_canary":
        _write_canary(out, "open", seconds=2.5 if damage == "throttled" else 0.7)
    if damage == "no_cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if damage == "cublas":
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(error, match=message):
        ic.train_stage(record.roots, pins=record.pins)
    assert iql.calls == [] and not (out / "p5_2").exists() and not (out / "training_random_iql.json").exists()


def test_the_cli_reads_the_real_pins_and_refuses_a_record_at_any_other(tmp_path: Path, monkeypatch: Any,
                                                                      capsys: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    roots = record.roots
    code = ic.main(["status", "--output-root", str(roots.output_root), "--corpus-root", str(roots.corpus_root),
                    "--draws-root", str(roots.draws_root), "--repo-root", str(roots.repo_root)])
    assert code == 2 and "pinned" in capsys.readouterr().err
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps({"format_version": "p5.2b-preflight/1.0", "status": "COMPLETE",
                                "reproduce": {"p5_2_path": {"reproduced": True}, "p8_4b_path": {"reproduced": True}},
                                "estimate": {"timeouts": {"train": 9000.5, "evaluate_p5_2": 4000.0,
                                                          "evaluate_p8_4b": 4500.0}}}))
    assert ic.main(["timeouts", "--preflight-record", str(path)]) == 0
    assert capsys.readouterr().out.split() == ["9000.5", "4000", "4500"]


@pytest.mark.parametrize("perturb", [None, "p5_2_path", "p8_4b_path"])
def test_the_preflight_records_t_reproduce_beside_its_timings_and_a_failure_yields_no_timeouts(
    tmp_path: Path, monkeypatch: Any, keep_torch_threads: Any, perturb: str | None
) -> None:
    """Amendment A, A3.1: the pre-flight RUNS T-reproduce (a) and (b) on the original checkpoint and records their
    outcomes beside its timings; a T-reproduce failure makes it FAILED, and a FAILED record yields no timeouts."""
    import offline.admission_probe as ap
    import offline.dt_gate as dt_gate
    from offline.att_rederivation import CellKey, cell_file_name

    record, iql = _synthetic_run_setup(tmp_path, monkeypatch)
    output = record.roots.output_root
    committed = record.ours[("iql", "random")]

    def evaluate_arm(**kwargs: Any) -> list[EpisodeResult]:
        shift = 0.5 if perturb == "p5_2_path" else 0.0
        return [EpisodeResult(kwargs["arm"], kwargs["seed"], int(d), committed[(int(kwargs["seed"]), int(d))] + shift,
                              float(int(kwargs["seed"]) % 7 + int(d) % 5), -float(d)) for d in kwargs["draw_ids"]]

    def probe_episode(**kwargs: Any) -> Any:
        cell = CellKey("grid4x4", kwargs["arm"], kwargs["seed"], int(kwargs["draw_id"]))
        row = json.loads((output / "p8_4b_rederivation" / cell_file_name(cell)).read_text())
        fields = {name: row[name] for name in ("scenario", "tier", "method", "arm", "seed", "draw_id", "created",
                                               "entered", "never_entered", "entered_fraction", "completed_at_horizon",
                                               "running_at_horizon", "waiting_at_horizon", "att_ours", "att_engine",
                                               "horizon_vehicle_count", "episode_reward")}
        if perturb == "p8_4b_path":
            fields["att_engine"] += 0.25
        return ap.AdmissionEpisode(**fields, seconds=0.02, seconds_rollout=0.01)

    monkeypatch.setattr(dt_gate, "evaluate_arm", evaluate_arm)
    monkeypatch.setattr(ap, "probe_episode", probe_episode)
    roots = ic.Roots(**{**record.roots.__dict__, "out_root": output / "p5_2b_runs" / "preflight_20261006T000000Z"})
    _write_canary(roots.out_root, "open")
    draws = record.draws[:2]
    result = ic.preflight(roots, stamp="20261006T000000Z", steps=20, draws=draws, pins=record.pins)
    written = json.loads((roots.out_root / "preflight.json").read_text())
    assert written == json.loads(json.dumps(result))
    assert written["rehearsal"]["steps"] == 20 and written["rehearsal"]["seconds"] == 10.0 + 101 / 100.0
    assert [call["declared_gradient_steps"] for call in iql.calls] == [20]
    a, b = written["reproduce"]["p5_2_path"], written["reproduce"]["p8_4b_path"]
    assert (a["reproduced"], b["reproduced"]) == (perturb != "p5_2_path", perturb != "p8_4b_path")
    assert a["episodes"] == len(draws) and b["cells"] == len(draws) and a["seconds"] >= 0 and b["seconds"] >= 0
    assert written["status"] == ("COMPLETE" if perturb is None else "FAILED")
    assert set(written["estimate"]["timeouts"]) == {"train", "evaluate_p5_2", "evaluate_p8_4b"}
    if perturb is None:
        assert ic.timeouts_from_preflight(roots.out_root / "preflight.json") == written["estimate"]["timeouts"]
    else:
        assert written["reasons"] and (a["differences"] or b["differences"])
        with pytest.raises(ValueError, match="COMPLETE"):
            ic.timeouts_from_preflight(roots.out_root / "preflight.json")
    assert not (output / "p5_2b").exists(), "the pre-flight never writes into the run's directory"


# ======================================================================
# Added after the mutation run of 2026-10-06, each closing one surviving mutant (named in its docstring).
# ======================================================================


def test_t_train_args_the_reward_scale_comes_from_the_selected_streams_only(tmp_path: Path, monkeypatch: Any) -> None:
    """M-scale-all-streams survived T-train-args when every stream had the selected ones' return span; the fixture's
    unselected episodes now widen it, and this test pins the scale to the selected streams by its own computation."""
    from offline.offline_baselines import iql_reward_scale

    corpus = tmp_path / "corpus"
    fx.write_synthetic_tier(corpus)
    fx.install_synthetic_tier(monkeypatch)
    inputs = ic.training_inputs(corpus)
    parts = ts.tier_parts("random", corpus)
    selected = iql_reward_scale([s.total_return for s in parts["streams"]])
    every = iql_reward_scale([s.total_return for s in parts["streams_all"]])
    assert selected != every, "the fixture must tell the selected streams' scale from every stream's"
    assert inputs.scale == selected


def test_t_statements_a_tie_for_first_with_the_predicted_arm_is_not_a_held_first_place() -> None:
    """M-q2a-ignores-ties survived when the tie put another arm first in the stable order: here ``dt_nomix`` -- the
    predicted first -- ties ``iql``, which follows it in ``METHODS``, so the stable order puts ``dt_nomix`` first."""
    levels = dict(RECORD_SHAPED)
    levels.update({"dt_nomix": 150.0, "iql": 150.0})
    cells = _cells(levels)
    cells[("iql", "random")] = [EpisodeResult("iql@random", e.seed, e.draw_id, e.att_horizon, 0.0, 0.0)
                                for e in cells[("dt_nomix", "random")]]
    got = ic.random_tier_statements(cells, with_hard_subset=False)
    assert got["q2a"]["measured_first"] == "dt_nomix" == got["q2a"]["predicted_first"]
    assert sorted(got["q2a"]["tied_for_first"]) == ["dt_nomix", "iql"]
    assert got["q2a"]["outcome"] == "FAILED"


def test_the_p5_2_stage_refuses_an_evaluation_of_a_model_that_is_not_the_records(tmp_path: Path, monkeypatch: Any,
                                                                                 keep_torch_threads: Any) -> None:
    """M-eval-stage-no-model-guard survived the happy path: here the training record names another weight digest for
    one seed, so (i) evaluated a model the record does not vouch for, and the stage must refuse."""
    record, _ = _synthetic_run_setup(tmp_path, monkeypatch)
    roots, pins, out = record.roots, record.pins, record.roots.out_root
    _write_canary(out, "open")
    ic.train_stage(roots, pins=pins)
    _write_canary(out, "close")
    path = out / "training_random_iql.json"
    training = json.loads(path.read_text(encoding="utf-8"))
    training["runs"][0]["state_dict_sha256"] = "0" * 64
    path.write_text(json.dumps(training), encoding="utf-8")
    assert ic.status(roots, pins=pins)["training"] == "complete"
    with pytest.raises(ValueError, match="model_provenance"):
        ic.evaluate_p5_2_stage(roots, pins=pins)


@pytest.mark.parametrize("damage, message", [("rows", "second route"), ("streams", "selected streams"),
                                             ("scale", "reward scale"), ("stats", "normalisation statistics")])
def test_assert_training_inputs_refuses_inputs_that_are_not_the_declared_data(tmp_path: Path, monkeypatch: Any,
                                                                              damage: str, message: str) -> None:
    """M-declared-rows-unchecked survived: each of ``assert_training_inputs``' four refusals, on the synthetic tier."""
    corpus = tmp_path / "corpus"
    fx.write_synthetic_tier(corpus)
    fx.install_synthetic_tier(monkeypatch)
    inputs = ic.training_inputs(corpus)
    keys = frozenset((Path(str(s.dataset_dir)).name, str(s.episode_file), str(s.ix_id)) for s in inputs.streams)
    declared = {"rows": len(inputs.table), "keys": keys}
    original = {"common": {"reward_scale": inputs.scale}, "stats": inputs.parts["dataset"].stats.to_json_obj()}
    assert ic.assert_training_inputs(inputs, declared, original)["rows"] == len(inputs.table)
    if damage == "rows":
        declared["rows"] += 1
    elif damage == "streams":
        declared["keys"] = frozenset(sorted(keys)[1:])
    elif damage == "scale":
        original["common"]["reward_scale"] = inputs.scale * 2
    else:
        original["stats"] = {**original["stats"], "split": "heldout"}
    with pytest.raises(ValueError, match=message):
        ic.assert_training_inputs(inputs, declared, original)


# ======================================================================
# BRIEF_44 Amendment B and B.1 (gate G1, FIX FIRST).  Each test names its item; the tests of a guard that already existed
# (B1.2, B1.6(c), B.1 item 2's guards) are green on the code they guard and proven by their mutants, not by a red run.
# ======================================================================


_HYPERPARAMETERS = ("batch_size", "learning_rate", "weight_decay", "grad_clip", "tau", "beta", "gamma", "polyak",
                    "weight_clip", "gradient_steps", "training_streams", "reward_scale", "torch_num_threads")


def _p8_4b_cell(record: Any, arm: str, seed: int, draw: int) -> Path:
    from offline.att_rederivation import CellKey, cell_file_name

    return record.roots.output_root / "p8_4b_rederivation" / cell_file_name(CellKey("grid4x4", arm, seed, draw))


def _rewrite(path: Path, change: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- B1.1: A1.4's third anchor, enforced


@pytest.mark.parametrize("definition", ["att_engine", "att_ours"])
def test_b1_1_a_p8_4b_value_off_the_c1_notes_mean_is_refused_naming_the_arm(tmp_path: Path, monkeypatch: Any,
                                                                            definition: str) -> None:
    """One ``bc@random`` cell of P8.4b moved by 1/64 under *definition*: its arm's ``statistics.mean`` no longer equals
    the C1 note's, and the report -- and the pre-token check, which runs the same input verification -- refuse,
    naming the arm and the definition, before any P8.4b value is read."""
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record)
    ic.write_run_manifest(record.roots, ic.protected_roots(record.roots))
    ic.build_report(record.roots, git=_git(), pins=record.pins)

    def nudge(payload: dict[str, Any]) -> None:
        payload[definition] += 1.0 / 64.0

    _rewrite(_p8_4b_cell(record, "bc@random", record.seeds[0], record.draws[0]), nudge)
    with pytest.raises(ValueError, match=f"bc@random.*{definition}"):
        ic.build_report(record.roots, git=_git(), pins=record.pins)
    with pytest.raises(ValueError, match=f"bc@random.*{definition}"):
        ic.check(record.roots, pins=record.pins, require_cuda=False)


def test_b1_1_the_note_check_reads_each_cell_by_its_key(tmp_path: Path) -> None:
    """The anchor reads the 500 (here 6) cells of each arm from the verified campaign list by P8.4b's own file names,
    and a file that is not the cell its name says is refused, whatever its values."""
    record = fx.write_synthetic_record(tmp_path)
    output = record.roots.output_root
    campaign = ic.assert_p8_4b_campaign(output, pins=record.pins)
    note = json.loads((record.roots.repo_root / ic.C1_NOTE_RELPATH).read_text(encoding="utf-8"))
    checked = ic.assert_c1_note_means(output, campaign["cells"], note)
    assert checked["arms"] == sorted(ts.METHODS) and checked["definitions"] == list(ic.DEFINITIONS)
    assert checked["cells_per_arm"] == {m: len(record.seeds) * len(record.draws) for m in ts.METHODS}

    def elsewhere(payload: dict[str, Any]) -> None:
        payload["draw_id"] = record.draws[-1] + 1

    _rewrite(_p8_4b_cell(record, "dt_nomix@random", record.seeds[0], record.draws[0]), elsewhere)
    with pytest.raises(ValueError, match="not the cell its name"):
        ic.assert_c1_note_means(output, campaign["cells"], note)


# ---------------------------------------------------------------- B1.2: A1.4's second anchor, tested (PM4)


def test_b1_2_a_p8_4b_cell_whose_att_ours_is_not_p5_2s_att_horizon_is_refused(tmp_path: Path) -> None:
    record = fx.write_synthetic_record(tmp_path)
    output = record.roots.output_root
    reference = ic.p5_2_cells(output, ic.p5_2_sums(output, pins=record.pins))
    assert len(ic._p8_4b_rows(output, "iql", "random", reference[("iql", "random")])) == len(record.seeds) * len(
        record.draws)

    def off(payload: dict[str, Any]) -> None:
        payload["att_ours"] += 0.5

    cell = _p8_4b_cell(record, "iql@random", record.seeds[-1], record.draws[-1])
    kept = cell.read_bytes()
    _rewrite(cell, off)
    assert json.loads(cell.read_text())["reproduces_committed"] is True
    with pytest.raises(ValueError, match="does not reproduce P5.2's committed att_horizon"):
        ic._p8_4b_rows(output, "iql", "random", reference[("iql", "random")])
    cell.write_bytes(kept)
    _rewrite(_p8_4b_cell(record, "bc@maxpressure", record.seeds[0], record.draws[0]), off)
    with pytest.raises(ValueError, match="does not reproduce P5.2's committed att_horizon"):
        ic.p8_4b_cells(output, reference, definition="att_engine", pins=record.pins)


# ---------------------------------------------------------------- B1.3: no second realisation


def test_b1_3_a_complete_training_whose_closing_canary_is_missing_is_never_trained_again(
    tmp_path: Path, monkeypatch: Any, keep_torch_threads: Any
) -> None:
    """The closing canary failed after a complete training (no file).  The restart accepts the state, refuses to train
    and trains nothing; ``close-late`` writes the training record's addendum before the late canary; the report says
    the seconds are bracketed by the opening canary only, with the training's end and the closing canary's time."""
    record, iql = _synthetic_run_setup(tmp_path, monkeypatch)
    roots, pins, out = record.roots, record.pins, record.roots.out_root
    _write_canary(out, "open")
    ic.train_stage(roots, pins=pins)
    trained = len(iql.calls)
    assert trained == len(record.seeds)
    state = ic.status(roots, pins=pins)
    assert (state["training"], state["canaries"]) == ("complete", "closing_pending")
    assert ic.check(roots, pins=pins, require_cuda=False)["state"] == "resumable"
    with pytest.raises(ValueError, match="training stage is complete"):
        ic.train_stage(roots, pins=pins)
    assert len(iql.calls) == trained

    training_path = out / "training_random_iql.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    assert training["finished_utc"] and set(training["hyperparameters"]) == set(_HYPERPARAMETERS)
    mark = ic.close_late_stage(roots, pins=pins)
    assert mark == out / "training_random_iql.late_close.json"
    written = json.loads(mark.read_text(encoding="utf-8"))
    assert written["format_version"] == "p5.2b-late-close/1.0"
    assert written["training_record"]["sha256"] == _sha256(training_path)
    assert written["training_finished_utc"] == training["finished_utc"] and written["marked_utc"]
    assert "opening canary only" in written["bracket"]
    kept = mark.read_bytes()
    assert ic.close_late_stage(roots, pins=pins) == mark and mark.read_bytes() == kept, "re-entered, it writes nothing"
    _write_canary(out, "close", seconds=0.75)
    assert ic.status(roots, pins=pins)["canaries"] == "complete"
    with pytest.raises(ValueError, match="taken late only after a complete training"):
        ic.close_late_stage(roots, pins=pins)

    ic.evaluate_p5_2_stage(roots, pins=pins)
    ic.evaluate_p8_4b_stage(roots, pins=pins)
    ic.manifest_stage(roots, pins=pins)
    artifact = json.loads(ic.report_stage(roots, pins=pins).read_text(encoding="utf-8"))
    seconds = artifact["training"]["seconds"]
    assert "opening canary only" in seconds["bracketed_by"] and "after a restart" in seconds["bracketed_by"]
    assert seconds["training_finished_utc"] == training["finished_utc"]
    assert seconds["closing_canary_utc"] == json.loads((out / "canary_close.json").read_text())["written_utc"]
    assert seconds["late_close"]["sha256"] == _sha256(mark)
    assert "after a restart" in artifact["what_this_does_not_say"][-1]
    listed = (record.roots.output_root / "SHA256SUMS_p5_2b.txt").read_text(encoding="utf-8")
    assert f"{_sha256(mark)}  p5_2b/training_random_iql.late_close.json" in listed.splitlines()
    assert len(iql.calls) == trained, "the restart trained nothing"


def test_b1_3_a_closing_canary_taken_in_its_run_brackets_the_seconds_with_both_canaries(tmp_path: Path,
                                                                                       monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record)
    ic.write_run_manifest(record.roots, ic.protected_roots(record.roots))
    artifact = ic.build_report(record.roots, git=_git(), pins=record.pins)
    seconds = artifact["training"]["seconds"]
    assert seconds["bracketed_by"].startswith("the opening and the closing canary") and seconds["late_close"] is None
    out = record.roots.out_root
    assert seconds["closing_canary_utc"] == json.loads((out / "canary_close.json").read_text())["written_utc"]
    assert artifact["what_this_does_not_say"][-1].endswith("measured between two machine-health canaries.")


@pytest.mark.parametrize("state", ["no_training", "both_canaries", "no_canary", "foreign_mark"])
def test_b1_3_close_late_refuses_unless_a_complete_training_misses_only_its_closing_canary(
    tmp_path: Path, monkeypatch: Any, state: str
) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    out = record.roots.out_root
    message = "taken late only after a complete training"
    if state == "no_training":
        _write_canary(out, "open")
    elif state == "both_canaries":
        fx.write_synthetic_run(record, stages=("training", "canaries"))
    elif state == "no_canary":
        fx.write_synthetic_run(record, stages=("training",))
    else:
        fx.write_synthetic_run(record, stages=("training",))
        _write_canary(out, "open")
        (out / "training_random_iql.late_close.json").write_text(json.dumps({
            "format_version": "p5.2b-late-close/1.0", "training_record": {"sha256": "0" * 64}}), encoding="utf-8")
        message = "does not name this training record"
    before = _tree_digest(out)
    with pytest.raises(ValueError, match=message):
        ic.close_late_stage(record.roots, pins=record.pins)
    assert _tree_digest(out) == before


def test_b1_3_the_report_refuses_a_late_close_mark_that_names_another_training_record(tmp_path: Path,
                                                                                      monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record)
    (record.roots.out_root / "training_random_iql.late_close.json").write_text(json.dumps({
        "format_version": "p5.2b-late-close/1.0", "training_record": {"sha256": "0" * 64}}), encoding="utf-8")
    ic.write_run_manifest(record.roots, ic.protected_roots(record.roots))
    with pytest.raises(ValueError, match="does not name this run's training record"):
        ic.build_report(record.roots, git=_git(), pins=record.pins)


# ---------------------------------------------------------------- B1.6(a): the output root is P5.2's, never nested


def test_b1_6a_an_output_root_without_p5_2s_manifest_is_refused(tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(PermissionError, match="holds no SHA256SUMS_p5_2"):
        ic.assert_out_root(bare / "p5_2b", bare)
    assert list(bare.iterdir()) == []


def test_b1_6a_no_output_root_nested_in_an_output_tree_is_ever_accepted(tmp_path: Path) -> None:
    """RA1's two unexpected cases: ``output/p5_2`` given as the output root was accepted by the barrier, and a file
    was written under it.  Each barrier function now refuses it, nothing is created, and a copy of P5.2's manifest
    placed inside ``output/p5_2`` does not make it an output root either."""
    roots = _fake_roots(tmp_path)
    output = roots.output_root
    nested = ic.Roots(**{**roots.__dict__, "output_root": output / "p5_2", "out_root": output / "p5_2" / "p5_2b"})
    before = _listing(output)
    for message in ("holds no SHA256SUMS_p5_2", "lies inside"):
        with pytest.raises(PermissionError, match=message):
            ic.assert_out_root(nested.out_root, nested.output_root)
        with pytest.raises(PermissionError, match=message):
            ic.protected_roots(nested)
        with pytest.raises(PermissionError, match=message):
            ic.assert_target(nested.out_root / "x.json", nested, ())
        with pytest.raises(PermissionError, match=message):
            ic.write_once_json(nested.out_root / "x.json", {"a": 1}, nested, ())
        assert _listing(output) == before
        (output / "p5_2" / "SHA256SUMS_p5_2.txt").write_text("x\n", encoding="utf-8")
        before = _listing(output)


def test_b1_6a_with_pins_the_output_roots_manifest_must_be_at_the_pinned_digest(tmp_path: Path,
                                                                                 monkeypatch: Any) -> None:
    import dataclasses

    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    output = record.roots.output_root
    assert ic.assert_out_root(output / "p5_2b", output, pins=record.pins) == (output / "p5_2b").resolve()
    wrong = dataclasses.replace(record.pins, p5_2_sums_sha256="0" * 64)
    with pytest.raises(PermissionError, match="not the pinned"):
        ic.assert_out_root(output / "p5_2b", output, pins=wrong)
    with pytest.raises(PermissionError, match="not the pinned"):
        ic.check(record.roots, pins=wrong, require_cuda=False)


# ---------------------------------------------------------------- B1.6(b): the thirteen hyperparameters, enforced


def test_b1_6b_the_planned_hyperparameters_are_what_a_real_train_iql_records(tmp_path: Path, monkeypatch: Any,
                                                                             keep_torch_threads: Any) -> None:
    """The planned values against what offline_baselines.train_iql ITSELF writes into a checkpoint -- a real two-step
    training on CPU on the synthetic tier, read back through the fields the original checkpoints are read by."""
    import offline.offline_baselines as ob

    corpus = tmp_path / "corpus"
    fx.write_synthetic_tier(corpus)
    fx.install_synthetic_tier(monkeypatch)
    inputs = ic.training_inputs(corpus)
    torch.set_num_threads(1)
    planned = ic.planned_hyperparameters(inputs, gradient_steps=2)
    assert set(planned) == set(_HYPERPARAMETERS)
    path = tmp_path / "real.pt"
    ob.train_iql(inputs.table, state_dim=inputs.group[0], n_actions=inputs.group[1], seed=101, declared_gradient_steps=2,
                 batch_size=ob.IQL_BATCH_TRANSITIONS, device=torch.device("cpu"), checkpoint_path=path,
                 stats=inputs.parts["dataset"].stats, scenario_id=ts.SCENARIO_ID, provenance=dict(inputs.provenance))
    provenance = torch.load(path, map_location="cpu", weights_only=False)["provenance"]
    recorded = {"batch_size": provenance["batch_size"], "learning_rate": provenance["learning_rate"],
                "weight_decay": provenance["weight_decay"], "grad_clip": provenance["grad_clip"],
                "gradient_steps": provenance["gradient_steps"], "training_streams": provenance["training_streams"],
                "torch_num_threads": provenance["runtime"]["torch_num_threads"],
                **{name: provenance["diagnostics"][name] for name in ("tau", "beta", "gamma", "polyak", "weight_clip",
                                                                      "reward_scale")}}
    assert planned == recorded


@pytest.mark.parametrize("name", _HYPERPARAMETERS)
def test_b1_6b_each_of_the_thirteen_off_the_originals_record_is_refused_naming_it(name: str) -> None:
    planned = {"batch_size": 1280, "learning_rate": 1e-4, "weight_decay": 1e-4, "grad_clip": 0.25, "tau": 0.7,
               "beta": 3.0, "gamma": 0.99, "polyak": 0.005, "weight_clip": 100.0, "gradient_steps": 40000,
               "training_streams": 3200, "reward_scale": 0.7429420505200595, "torch_num_threads": 1}
    original = {**planned, "declared_gradient_steps": 40000, "tier": "random", "device": "cuda",
                "training_rows": 2_304_000, "git_commit": "9460800"}
    assert ic.assert_hyperparameters(planned, original) == planned
    drifted = {**planned, name: planned[name] * 2}
    with pytest.raises(ValueError, match=name):
        ic.assert_hyperparameters(drifted, original)
    if name == "gradient_steps":
        with pytest.raises(ValueError, match="declared_gradient_steps"):
            ic.assert_hyperparameters(planned, {**original, "declared_gradient_steps": 20000})


def test_b1_6b_the_training_stage_refuses_a_drifted_hyperparameter_before_training_anything(
    tmp_path: Path, monkeypatch: Any, keep_torch_threads: Any
) -> None:
    import offline.offline_baselines as ob

    record, iql = _synthetic_run_setup(tmp_path, monkeypatch)
    out = record.roots.out_root
    _write_canary(out, "open")
    monkeypatch.setattr(ob, "IQL_TAU", 0.8)
    with pytest.raises(ValueError, match="tau"):
        ic.train_stage(record.roots, pins=record.pins)
    assert iql.calls == [] and not (out / "p5_2").exists() and not (out / "training_random_iql.json").exists()


# ---------------------------------------------------------------- B1.6(c): path (ii)'s three refusals


def _through_the_p5_2_path(tmp_path: Path, monkeypatch: Any) -> tuple[Any, Any]:
    record, iql = _synthetic_run_setup(tmp_path, monkeypatch)
    out = record.roots.out_root
    _write_canary(out, "open")
    ic.train_stage(record.roots, pins=record.pins)
    _write_canary(out, "close")
    ic.evaluate_p5_2_stage(record.roots, pins=record.pins)
    return record, iql


def test_b1_6c_ii_refuses_when_p8_4bs_resolver_finds_another_checkpoint(tmp_path: Path, monkeypatch: Any,
                                                                         keep_torch_threads: Any) -> None:
    import offline.att_rederivation as ar

    record, _ = _through_the_p5_2_path(tmp_path, monkeypatch)
    originals = record.roots.output_root / "p5_2" / "checkpoints"
    monkeypatch.setattr(ar, "rederivation_checkpoint", lambda scenario, tier, method, seed, roots: (
        originals / f"grid4x4_{tier}_{method}_seed{seed}.pt"))
    with pytest.raises(ValueError, match="P8.4b's resolver finds"):
        ic.evaluate_p8_4b_stage(record.roots, pins=record.pins)
    assert not (record.roots.out_root / "rederivation").exists()


def test_b1_6c_ii_refuses_a_campaign_that_refused_a_cell(tmp_path: Path, monkeypatch: Any,
                                                         keep_torch_threads: Any) -> None:
    import dataclasses

    import offline.admission_probe as ap

    record, _ = _through_the_p5_2_path(tmp_path, monkeypatch)
    honest = ap.probe_episode
    first = (record.seeds[0], record.draws[0])

    def one_off(**kwargs: Any) -> Any:
        episode = honest(**kwargs)
        if (int(kwargs["seed"]), int(kwargs["draw_id"])) == first:
            return dataclasses.replace(episode, att_ours=episode.att_ours + 0.5)
        return episode

    monkeypatch.setattr(ap, "probe_episode", one_off)
    with pytest.raises(RuntimeError, match=r"\(ii\) refused 1 cell"):
        ic.evaluate_p8_4b_stage(record.roots, pins=record.pins)


def test_b1_6c_ii_refuses_a_stale_cell_that_does_not_reproduce_i(tmp_path: Path, monkeypatch: Any,
                                                                  keep_torch_threads: Any) -> None:
    """A cell file left by an earlier attempt is skipped by P8.4b's resume, so only the stage's own re-check after the
    campaign can see that it does not carry (i)'s value."""
    from offline.att_rederivation import CellKey, cell_file_name

    record, _ = _through_the_p5_2_path(tmp_path, monkeypatch)
    out = record.roots.out_root
    seed, draw = record.seeds[0], record.draws[0]
    value = _engine_att(seed, draw) + 1.0
    stale = fx._cell_row("iql", "random", seed, draw, value, value - 6.5,
                         str(out / "p5_2" / "checkpoints" / f"grid4x4_random_iql_seed{seed}.pt"))
    path = out / "rederivation" / cell_file_name(CellKey("grid4x4", "iql@random", seed, draw))
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match=r"does not reproduce \(i\)'s att_horizon"):
        ic.evaluate_p8_4b_stage(record.roots, pins=record.pins)


# ---------------------------------------------------------------- B1.6(e): the corpus manifest, pinned


def test_b1_6e_the_corpus_manifest_is_read_at_its_pinned_digest(tmp_path: Path, monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    roots = record.roots
    ic.declared_selection(roots.repo_root, roots.corpus_root, pins=record.pins)
    manifest = roots.corpus_root / "cf_grid4x4__random" / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus manifest"):
        ic.declared_selection(roots.repo_root, roots.corpus_root, pins=record.pins)
    with pytest.raises(ValueError, match="corpus manifest"):
        ic.check(roots, pins=record.pins, require_cuda=False)


# ---------------------------------------------------------------- B1.7: the report's precision


def test_b1_7c_the_corrected_cells_level_must_equal_its_own_cell_mean(tmp_path: Path, monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record)

    def moved(payload: dict[str, Any]) -> None:
        payload["cell"]["att_horizon_mean"] += 1.0

    _rewrite(record.roots.out_root / "eval_random_iql.json", moved)
    ic.write_run_manifest(record.roots, ic.protected_roots(record.roots))
    with pytest.raises(ValueError, match="not the file's own cell mean"):
        ic.build_report(record.roots, git=_git(), pins=record.pins)


def test_b1_7_the_report_flags_q1s_predictions_names_the_corrected_sources_and_the_notes_use(
    tmp_path: Path, monkeypatch: Any
) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record, iql_shift=60.0)
    ic.write_run_manifest(record.roots, ic.protected_roots(record.roots))
    artifact = ic.build_report(record.roots, git=_git(), pins=record.pins)
    for when in ("before", "after"):
        engine = artifact["statements"]["att_engine"][when]["q1"]["predictions_registered_on"]
        ours = artifact["statements"]["att_ours"][when]["q1"]["predictions_registered_on"]
        assert "att_horizon" in engine and "att_engine" in engine, engine
        assert "att_horizon" in ours and "att_engine" not in ours, ours
    pattern = "output/p5_2b/rederivation/cell_grid4x4_iql_at_random_*"
    assert artifact["cells"]["corrected"]["sources"] == {
        "att_ours": "output/p5_2b/eval_random_iql.json", "att_ours_p8_4b_path": pattern, "att_engine": pattern,
        "admission": pattern}
    used = artifact["inputs"]["c1_note"]["used_for"]
    assert "statistics.mean" in used and "att_engine" in used and "att_ours" in used
    assert "c1_rule_r.json" in artifact["inputs"]["p8_4b_campaign"]["anchor"]


@pytest.mark.parametrize("digest", ["canonical_digest", "state_dict_sha256"])
def test_b1_7f_a_corrected_run_carrying_an_original_weight_digest_is_refused(tmp_path: Path, monkeypatch: Any,
                                                                            digest: str) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record)
    out = record.roots.out_root
    original = record.roots.output_root / "p5_2" / "checkpoints" / f"grid4x4_random_iql_seed{record.seeds[0]}.pt"
    if digest == "canonical_digest":
        value = torch.load(original, map_location="cpu", weights_only=False)["canonical_digest"]
    else:
        value = ts.canonical_state_dict_digest(original)

    def same(payload: dict[str, Any]) -> None:
        payload["runs"][-1][digest] = value

    _rewrite(out / "training_random_iql.json", same)
    if digest == "state_dict_sha256":
        def vouched(payload: dict[str, Any]) -> None:
            payload["model_provenance"][str(record.seeds[-1])]["state_dict_sha256"] = value

        _rewrite(out / "eval_random_iql.json", vouched)
    ic.write_run_manifest(record.roots, ic.protected_roots(record.roots))
    with pytest.raises(ValueError, match="an original's"):
        ic.build_report(record.roots, git=_git(), pins=record.pins)


# ---------------------------------------------------------------- B.1, item 1: P5.2's per-seed rule for Q2 (D9)


def _seed_levels(cells: dict[tuple[str, str], list[EpisodeResult]], seed: int) -> dict[str, float]:
    return {m: float(np.asarray([e.att_horizon for e in cells[(m, "random")] if e.seed == seed],
                                dtype=np.float64).mean()) for m in ts.METHODS}


def test_b_1_1_a_first_place_that_reverses_on_one_seed_is_named_as_reversing() -> None:
    levels = dict(RECORD_SHAPED)
    levels.update({"dt_spatial": 150.0, "dt_nomix": 151.0, "iql": 320.0})
    cells = _cells(levels)
    cells[("dt_spatial", "random")] = [
        EpisodeResult(e.arm, e.seed, e.draw_id, e.att_horizon + (3.0 if e.seed == 202 else 0.0), 0.0, 0.0)
        for e in cells[("dt_spatial", "random")]]
    got = ic.random_tier_statements(cells, with_hard_subset=False)
    predicted = list(ts.predicted_order("random"))
    mine = {}
    for seed in (101, 202, 303, 404, 505):
        own = _seed_levels(cells, seed)
        mine[str(seed)] = sorted(ts.METHODS, key=lambda m: own[m])
        block = got["per_seed"]["seeds"][str(seed)]
        assert block["levels"] == own and block["order"] == mine[str(seed)] and block["first"] == mine[str(seed)][0]
        assert got["q2b"]["per_seed_n_concordant"][str(seed)] == _my_concordance(predicted, own, list(ts.METHODS))[0]
    assert set(got["per_seed"]["seeds"]) == set(mine)
    assert got["q2a"]["measured_first"] == "dt_spatial" and mine["202"][0] == "dt_nomix"
    assert all(mine[s][0] == "dt_spatial" for s in mine if s != "202")
    assert got["q2a"]["reverses_on_seeds"] == [202] and "202" in got["q2a"]["reversal"]
    unreversed = ic.random_tier_statements(_cells(dict(RECORD_SHAPED)), with_hard_subset=False)
    assert unreversed["q2a"]["reverses_on_seeds"] == [] and unreversed["q2a"]["reversal"] is None


def test_b_1_1_the_per_seed_block_refuses_arms_that_do_not_cover_the_same_seeds() -> None:
    cells = _cells(dict(RECORD_SHAPED))
    cells[("bc", "random")] = [e for e in cells[("bc", "random")] if e.seed != 505]
    with pytest.raises(ValueError, match="same seeds"):
        ic.random_tier_statements(cells, with_hard_subset=False)


def test_t_reproduce_c_the_per_seed_orderings_of_the_original_files_equal_this_files_own() -> None:
    """B.1, item 1, on the real record (gated): the BEFORE per-seed orderings under both definitions equal this test's
    own -- P5.2's eval files and P8.4b's cell files read here with json, means by numpy in draw order -- and, among
    the five arms other than IQL, dt_spatial is first on four seeds and dt_nomix on seed 202 (RA2's measurement: the
    AFTER first place reverses on that seed if the corrected IQL is no longer first)."""
    output = _p8_4b_cells()
    reference = ic.p5_2_cells(output, ic.p5_2_sums(output))
    raw_ours: dict[tuple[str, str], dict[tuple[int, int], float]] = {}
    for method in ts.METHODS:
        payload = json.loads((output / "p5_2" / f"eval_random_{method}.json").read_text())
        raw_ours[(method, "random")] = {(int(e["seed"]), int(e["draw_id"])): float(e["att_horizon"])
                                        for e in payload["episodes"]}
    sources = {"att_ours": raw_ours, "att_engine": _raw_cells(output, "att_engine")}
    for definition, raw in sources.items():
        cells = reference if definition == "att_ours" else ic.p8_4b_cells(output, reference, definition=definition)
        got = ic.random_tier_statements(cells, with_hard_subset=False)
        reversing = []
        for seed in (101, 202, 303, 404, 505):
            own = {m: float(np.asarray([v for (s, d), v in sorted(raw[(m, "random")].items()) if s == seed],
                                       dtype=np.float64).mean()) for m in ts.METHODS}
            order = sorted(ts.METHODS, key=lambda m: own[m])
            assert got["per_seed"]["seeds"][str(seed)]["levels"] == own, (definition, seed)
            assert got["per_seed"]["seeds"][str(seed)]["order"] == order, (definition, seed)
            if order[0] != got["q2a"]["measured_first"]:
                reversing.append(seed)
            without_iql = [m for m in order if m != "iql"]
            assert without_iql[0] == ("dt_nomix" if seed == 202 else "dt_spatial"), (definition, seed, without_iql)
        assert got["q2a"]["reverses_on_seeds"] == reversing, definition


# ---------------------------------------------------------------- B.1, item 2: every guard of the report, exercised


@pytest.mark.parametrize("damage, message", [("policy_source", "was rolled from"),
                                             ("committed_att_ours", r"does not reproduce \(i\)'s att_horizon")])
def test_b_1_2_a_corrected_cell_not_rolled_from_the_corrected_checkpoint_or_not_vouched_is_refused(
    tmp_path: Path, monkeypatch: Any, damage: str, message: str
) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record)
    original = record.roots.output_root / "p5_2" / "checkpoints" / f"grid4x4_random_iql_seed{record.seeds[0]}.pt"

    def damaged(payload: dict[str, Any]) -> None:
        if damage == "policy_source":
            payload["policy_source"]["checkpoint"] = str(original)
        else:
            payload["committed_att_ours"] += 0.5

    _rewrite(sorted((record.roots.out_root / "rederivation").glob("cell_*.json"))[0], damaged)
    ic.write_run_manifest(record.roots, ic.protected_roots(record.roots))
    with pytest.raises(ValueError, match=message):
        ic.build_report(record.roots, git=_git(), pins=record.pins)


def test_b_1_2_a_run_file_changed_after_the_run_manifest_is_refused_by_the_report(tmp_path: Path,
                                                                                  monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record)
    ic.write_run_manifest(record.roots, ic.protected_roots(record.roots))
    canary = record.roots.out_root / "canary_close.json"
    canary.write_text(canary.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its line in .*SHA256SUMS_p5_2b"):
        ic.build_report(record.roots, git=_git(), pins=record.pins)


def test_b_1_2_check_refuses_two_canaries_without_a_complete_training(tmp_path: Path, monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    _write_canary(record.roots.out_root, "open")
    _write_canary(record.roots.out_root, "close")
    state = ic.status(record.roots, pins=record.pins)
    assert (state["training"], state["canaries"]) == ("absent", "complete")
    with pytest.raises(ValueError, match="canaries"):
        ic.check(record.roots, pins=record.pins, require_cuda=False)


def test_b_1_2_a_training_record_with_another_row_count_is_not_complete(tmp_path: Path, monkeypatch: Any) -> None:
    """The completeness predicate's rows clause: a record whose runs trained on the whole tier's rows (the defect's
    count) is not a complete corrected training, whatever else it gets right."""
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record, stages=("training", "canaries"))
    assert ic.status(record.roots, pins=record.pins)["training"] == "complete"

    def whole_tier(payload: dict[str, Any]) -> None:
        for run in payload["runs"]:
            run["training_rows"] = len(fx.CORPUS_EPISODES) * len(fx.IDS) * fx.EPISODE_LENGTH

    _rewrite(record.roots.out_root / "training_random_iql.json", whole_tier)
    assert ic.status(record.roots, pins=record.pins)["training"] == "partial"


def test_b_1_2_the_manifest_stage_refuses_a_partial_run_and_freezes_nothing(tmp_path: Path, monkeypatch: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    manifest = record.roots.output_root / "SHA256SUMS_p5_2b.txt"
    fx.write_synthetic_run(record, stages=("training", "canaries", "p5_2_eval"))
    with pytest.raises(ValueError, match="not complete"):
        ic.manifest_stage(record.roots, pins=record.pins)
    assert not manifest.exists()
    fx.write_synthetic_run(record, stages=("rederivation",))
    lines = ic.manifest_stage(record.roots, pins=record.pins)
    assert manifest.read_text(encoding="utf-8").splitlines() == lines and lines


def test_b_1_2_the_clis_manifest_command_is_the_gated_stage(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    monkeypatch.setattr(ic, "PINS", record.pins)
    fx.write_synthetic_run(record, stages=("training", "canaries"))
    roots = record.roots
    code = ic.main(["manifest", "--output-root", str(roots.output_root), "--corpus-root", str(roots.corpus_root),
                    "--draws-root", str(roots.draws_root), "--repo-root", str(roots.repo_root)])
    assert code == 2 and "not complete" in capsys.readouterr().err
    assert not (roots.output_root / "SHA256SUMS_p5_2b.txt").exists()


# ---------------------------------------------------------------- Amendment B on the real record (gated)
# Written WITH the implementation (C8f), not before it: they check that the new pins and anchors hold on the real
# record, and are proven by the mutants that change a pin, not by a red run.


def test_b1_the_new_pins_and_the_c1_anchor_hold_on_the_real_record() -> None:
    """B1.6(e)'s pin is the real corpus manifest's digest; B1.1's anchor holds on P8.4b's 3,000 random-tier cells (500
    per arm, both definitions); B1.6(a)'s barrier accepts the real output root at the real pins."""
    output = _p8_4b_cells()
    declared = ic.declared_selection(REPO_ROOT, _corpus_root())
    assert declared["corpus_manifest"]["sha256"] == ic.PINS.corpus_manifest_sha256
    assert (declared["rows"], declared["streams"], declared["whole_tier_rows"]) == (1_152_000, 3200, 2_304_000)
    campaign = ic.assert_p8_4b_campaign(output)
    note = json.loads((REPO_ROOT / ic.C1_NOTE_RELPATH).read_text(encoding="utf-8"))
    checked = ic.assert_c1_note_means(output, campaign["cells"], note)
    assert checked["cells_per_arm"] == {m: 500 for m in ts.METHODS}
    assert ic.assert_out_root(output / "p5_2b", output, pins=ic.PINS) == (output / "p5_2b").resolve()


def test_b1_6b_the_thirteen_planned_for_the_real_table_are_the_originals(real_training_inputs: Any,
                                                                          keep_torch_threads: Any) -> None:
    """RA1's 13 of 13, as a test: the values the run will train with, on the real corrected table and one torch thread,
    equal what the five original checkpoints record."""
    output = _output_root()
    original = ic.original_training(output, ic.p5_2_sums(output))
    torch.set_num_threads(1)
    planned = ic.planned_hyperparameters(real_training_inputs, gradient_steps=40000)
    assert ic.assert_hyperparameters(planned, original["common"]) == planned
    assert (planned["batch_size"], planned["learning_rate"], planned["tau"], planned["beta"], planned["gamma"]) == (
        1280, 1e-4, 0.7, 3.0, 0.99)
    assert (planned["training_streams"], planned["reward_scale"]) == (3200, 0.7429420505200595)


# ---------------------------------------------------------------- B1.8 (the implementer's call, taken)


def test_b1_8_the_report_needs_no_materialised_draw_and_the_stages_before_it_do(tmp_path: Path,
                                                                                monkeypatch: Any) -> None:
    """The report reads no draw, so T-regress needs no draws gate; every stage that rolls or checks before the token
    still refuses without the 100 held-out draws."""
    import shutil

    record = fx.write_synthetic_record(tmp_path)
    fx.install_synthetic_protocol(monkeypatch, record)
    fx.write_synthetic_run(record)
    shutil.rmtree(record.roots.draws_root / "cityflow_grid4x4")
    with pytest.raises(FileNotFoundError, match="held-out draws"):
        ic.check(record.roots, pins=record.pins, require_cuda=False)
    ic.write_run_manifest(record.roots, ic.protected_roots(record.roots))
    assert ic.build_report(record.roots, git=_git(), pins=record.pins)["format_version"] == "p5.2b-correction/1.0"
