"""Tests for P9.3 Phase B (``BRIEF_48`` E.3.6; ``docs/plans/p9.3.md`` section 10) -- written before the code they test.

The training path:

* the arms and their recipes are A27 (i)'s, with A27 (b)'s matched batches and A27 (c)'s dropout-free IQL targets;
* the tier specs are the committed declaration's (``docs/data/p9_3_declaration.json``);
* :func:`offline.hz4x4_phase_b.hz4x4_tier_parts` is ``tier_sweep.tier_parts``' body on an explicit spec -- held to the
  original on the synthetic grid tier of ``tests/p5_2b_fixtures.py`` (six episodes available, three selected);
* :func:`offline.hz4x4_phase_b.train_cell` hands each trainer exactly what ``tier_sweep train`` and ``tier_sweep
  train-baselines`` (with P9.1's M1 flags) hand it on the same tier and seed.  The trainers are spies here, so nothing
  is trained and the comparison is of the arguments, tensor for tensor;
* a cell is written once: an existing checkpoint or record is refused before anything is trained or written;
* the record carries the trainer's own seconds under the uncontrolled-clock label, and the file's digest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
import torch

import offline.hz4x4_ladder as hz
import offline.hz4x4_phase_b as pb
import offline.tier_sweep as ts
from tests import p5_2b_fixtures as fx

REPO = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


def _same(a: Any, b: Any) -> bool:
    """Exact structural equality: tensors and arrays by dtype, shape and every element; dataclasses field by field."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and bool(torch.equal(a, b)))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype
                and a.shape == b.shape and bool(np.array_equal(a, b)))
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(_same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
    if isinstance(a, Mapping):
        return isinstance(b, Mapping) and set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


# ======================================================================
# The arms and the tier specs
# ======================================================================


def test_the_arms_are_a27is_and_their_recipes_the_grids_matched_ones() -> None:
    assert pb.ARMS == ("dt_nomix", "bc_m", "bc_top10_m", "bc_top10_perix_m", "iql_m")
    got = {arm: (r.base, r.batch, r.target_dropout, r.steps) for arm in pb.ARMS for r in [pb.arm_recipe(arm)]}
    assert got == {
        "dt_nomix": ("dt_nomix", 64, None, 40_000),
        "bc_m": ("bc", 1_024, None, 40_000),
        "bc_top10_m": ("bc_top10", 102, None, 40_000),
        "bc_top10_perix_m": ("bc_top10_perix", 102, None, 40_000),
        "iql_m": ("iql", 20_480, "off", 40_000),
    }
    for name in ("bc", "iql", "iql_t", "dt_spatial", "bc_100k", "behaviour"):
        with pytest.raises(ValueError, match="A27 \\(i\\)"):
            pb.arm_recipe(name)


def test_the_tier_specs_are_the_committed_declarations() -> None:
    declared = json.loads((REPO / "docs" / "data" / "p9_3_declaration.json").read_bytes())["tier_specs"]
    assert list(declared) == sorted(hz.TIER_ORDER)
    for tier in hz.TIER_ORDER:
        spec = pb.hz4x4_tier_spec(tier)
        assert isinstance(spec, ts.TierSpec)
        assert dataclasses.asdict(spec) == {**declared[tier], "dirs": tuple(declared[tier]["dirs"])}
        assert spec.dirs == tuple(hz._tier_dir_names(tier))
    with pytest.raises(ValueError, match="random"):
        pb.hz4x4_tier_spec("random")


def test_a_declaration_that_disagrees_with_phase_a_is_refused(tmp_path: Path) -> None:
    payload = json.loads((REPO / "docs" / "data" / "p9_3_declaration.json").read_bytes())
    payload["tier_specs"]["maxpressure"]["dirs"] = ["cf_grid4x4__maxpressure"]
    tampered = tmp_path / "declaration.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cf_grid4x4__maxpressure"):
        pb.hz4x4_tier_spec("maxpressure", declaration_path=tampered)
    payload["tier_specs"]["maxpressure"]["dirs"] = ["cf_hz4x4__maxpressure"]
    payload["tier_specs"]["maxpressure"]["subsample"] = "one_per_draw"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="one_per_draw"):
        pb.hz4x4_tier_spec("maxpressure", declaration_path=tampered)


# ======================================================================
# The parts: tier_sweep.tier_parts' body on an explicit spec
# ======================================================================


@pytest.fixture()
def synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Any]:
    root = tmp_path / "corpus"
    fx.write_synthetic_tier(root)
    spec = fx.install_synthetic_tier(monkeypatch)
    return root, spec


def test_hz4x4_tier_parts_is_tier_parts_on_an_explicit_spec(synthetic: tuple[Path, Any]) -> None:
    root, spec = synthetic
    mine, theirs = pb.hz4x4_tier_parts(spec, root), ts.tier_parts("random", root)
    assert set(mine) == set(theirs)
    assert mine["spec"] == theirs["spec"] == spec
    assert mine["node_ids"] == theirs["node_ids"] and len(mine["node_ids"]) == 16
    assert mine["adjacency"] is None and theirs["adjacency"] is None
    assert mine["episodes_all"] == theirs["episodes_all"] and len(mine["episodes_all"]) == 6
    assert mine["episodes"] == theirs["episodes"] and len(mine["episodes"]) == 3
    assert mine["streams"] == theirs["streams"] and len(mine["streams"]) == 3 * 16
    assert mine["streams_all"] == theirs["streams_all"]
    assert mine["prompts"] == theirs["prompts"]
    assert _same(mine["index"], theirs["index"])
    assert _same(mine["stack"](), theirs["stack"]())
    assert mine["dataset"].stats.to_json_obj() == theirs["dataset"].stats.to_json_obj()


# ======================================================================
# train_cell hands each trainer what tier_sweep's CLI hands it
# ======================================================================


def _ts_args(tmp_path: Path, corpus: Path, *command: str, root: str = "p5_2_like") -> Any:
    return ts.build_parser().parse_args([
        "--corpus-root", str(corpus), "--work-dir", str(tmp_path / root / "work"),
        "--checkpoint-dir", str(tmp_path / root / "checkpoints"), "--reuse-root", str(tmp_path / "protected"),
        "--device", "cpu", "--seeds", "101", "--gradient-steps", "2", "--tier", "random", *command,
    ])


class _DtSpy:
    """Stands in for ``tier_sweep.train_tier_dt``: records the call, writes the file the caller moves, returns a result."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, stacked: Any, index: Any, **kwargs: Any) -> Any:
        from offline.spatial_mixing import TrainResult

        self.calls.append({"stacked": stacked, "index": index, **kwargs})
        seed = int(kwargs["seed"])
        torch.save({"model": {"w": torch.full((3,), float(seed))}, "provenance": {"seed": seed}},
                   Path(kwargs["checkpoint_path"]))
        return TrainResult(method=str(kwargs["method"]), seed=seed, gradient_steps=int(kwargs["gradient_steps"]),
                           losses=(2.0, 1.0 + seed / 1000.0), checkpoint_path=str(kwargs["checkpoint_path"]),
                           seconds=30.0 + seed / 100.0)


def test_train_cell_hands_train_tier_dt_what_tier_sweep_train_hands_it(
    synthetic: tuple[Path, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, spec = synthetic
    spy = _DtSpy()
    monkeypatch.setattr(ts, "train_tier_dt", spy)
    assert ts._run_train(_ts_args(tmp_path, root, "train", "--method", "dt_nomix")) == 0
    out = tmp_path / "p9_out"
    record = pb.train_cell(spec, "dt_nomix", 101, corpus_root=root, output_root=out, device=CPU, gradient_steps=2)
    theirs, mine = spy.calls
    assert _same(mine.pop("stacked"), theirs.pop("stacked"))
    assert _same(mine.pop("index"), theirs.pop("index"))
    assert mine.pop("stats").to_json_obj() == theirs.pop("stats").to_json_obj()
    final = pb.checkpoint_path(out, "random", "dt_nomix", 101)
    assert Path(mine.pop("checkpoint_path")) == final.with_name(final.name + ".partial")
    theirs.pop("checkpoint_path")
    assert set(mine.pop("protected")) == set(ts.protected_roots_from([out / "p5_1", out / "p5_2"]))
    theirs.pop("protected")
    provenance = mine.pop("provenance")
    assert set(theirs.pop("provenance")) == {"runtime"}
    assert {key: provenance[key] for key in ("scenario", "scenario_id", "arm", "base_method", "registered_in")} == {
        "scenario": "hz4x4", "scenario_id": "cityflow_grid4x4", "arm": "dt_nomix", "base_method": "dt_nomix",
        "registered_in": pb.arm_recipe("dt_nomix").registered_in}
    assert "runtime" in provenance
    assert mine == theirs
    assert (mine["tier"], mine["method"], mine["gradient_steps"], mine["batch_size"]) == ("random", "dt_nomix", 2, 64)
    assert record["seconds"] == 30.0 + 101 / 100.0
    assert record["canonical_digest"] == ts.canonical_state_dict_digest(final)


MATCHED_FLAGS: dict[str, tuple[str, ...]] = {
    "bc_m": ("--methods", "bc", "--bc-batch-windows", "1024"),
    "bc_top10_m": ("--methods", "bc_top10", "--bc-batch-windows", "102"),
    "bc_top10_perix_m": ("--methods", "bc_top10_perix", "--bc-batch-windows", "102"),
    "iql_m": ("--methods", "iql", "--iql-batch-transitions", "20480", "--iql-target-dropout", "off"),
}


@pytest.mark.parametrize("arm", list(MATCHED_FLAGS))
def test_train_cell_hands_the_baseline_trainer_what_train_baselines_hands_it_under_m1s_flags(
    arm: str, synthetic: tuple[Path, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, spec = synthetic
    iql, bc = fx.install_train_spies(monkeypatch)
    args = _ts_args(tmp_path, root, "train-baselines", *MATCHED_FLAGS[arm], root="p9_1_runs")
    assert ts._run_train_baselines(args) == 0
    pb.train_cell(spec, arm, 101, corpus_root=root, output_root=tmp_path / "p9_out", device=CPU, gradient_steps=2)
    spy, other = (iql, bc) if arm == "iql_m" else (bc, iql)
    assert other.calls == []
    theirs, mine = spy.calls
    assert _same(mine.pop("batch"), theirs.pop("batch"))
    assert mine.pop("stats").to_json_obj() == theirs.pop("stats").to_json_obj()
    mine.pop("checkpoint_path")
    theirs.pop("checkpoint_path")
    provenance, registered = mine.pop("provenance"), theirs.pop("provenance")
    shared = {key: value for key, value in registered.items() if key != "runtime"}
    assert {key: provenance.get(key) for key in shared} == shared
    assert provenance["scenario"] == "hz4x4" and "runtime" in provenance
    assert mine == theirs
    assert mine["batch_size"] == pb.arm_recipe(arm).batch


def test_the_iql_reward_scale_and_the_two_deciles_are_the_selected_streams(synthetic: tuple[Path, Any]) -> None:
    root, spec = synthetic
    parts = pb.hz4x4_tier_parts(spec, root)
    inputs = pb.hz4x4_baseline_inputs(parts, root)
    from offline.offline_baselines import iql_reward_scale

    assert inputs["scale"] == iql_reward_scale([s.total_return for s in parts["streams"]])
    assert inputs["scale"] != iql_reward_scale([s.total_return for s in parts["streams_all"]])
    assert inputs["selections"] == ts.baseline_stream_selections(parts["dataset"], parts["streams"])
    assert inputs["selections"]["bc_top10"] != inputs["selections"]["bc_top10_perix"]


# ======================================================================
# Written once; the record
# ======================================================================


def test_train_cell_refuses_an_existing_checkpoint_or_record_before_training(
    synthetic: tuple[Path, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, spec = synthetic
    iql, bc = fx.install_train_spies(monkeypatch)
    out = tmp_path / "p9_out"
    checkpoint = pb.checkpoint_path(out, "random", "bc_m", 101)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"a checkpoint already on disk")
    with pytest.raises(FileExistsError, match="written once"):
        pb.train_cell(spec, "bc_m", 101, corpus_root=root, output_root=out, device=CPU, gradient_steps=2)
    assert checkpoint.read_bytes() == b"a checkpoint already on disk"
    checkpoint.unlink()
    record = pb.train_record_path(out, "random", "bc_m", 101)
    record.parent.mkdir(parents=True)
    record.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="written once"):
        pb.train_cell(spec, "bc_m", 101, corpus_root=root, output_root=out, device=CPU, gradient_steps=2)
    assert bc.calls == [] and iql.calls == []
    assert record.read_text(encoding="utf-8") == "{}"
    assert not checkpoint.exists() and not checkpoint.with_name(checkpoint.name + ".partial").exists()


def test_the_training_record_carries_the_trainers_seconds_under_the_clock_label_and_the_files_digest(
    synthetic: tuple[Path, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, spec = synthetic
    fx.install_train_spies(monkeypatch)
    out = tmp_path / "p9_out"
    record = pb.train_cell(spec, "iql_m", 202, corpus_root=root, output_root=out, device=CPU, gradient_steps=2)
    assert json.loads(pb.train_record_path(out, "random", "iql_m", 202).read_bytes()) == record
    path = pb.checkpoint_path(out, "random", "iql_m", 202)
    assert not path.with_name(path.name + ".partial").exists()
    assert record["format_version"] == pb.ARTIFACT_FORMAT_TRAIN
    assert record["checkpoint_path"] == str(path)
    assert record["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["seconds"] == 10.0 + 202 / 100.0
    assert record["clock"] == pb.CLOCK_LABEL
    assert (record["declared_gradient_steps"], record["gradient_steps"]) == (2, 2)
    assert (record["scenario"], record["tier"], record["arm"], record["seed"]) == ("hz4x4", "random", "iql_m", 202)
    assert (record["base_method"], record["batch"], record["target_dropout"]) == ("iql", 20_480, "off")
    assert record["scenario_id"] == "cityflow_grid4x4"
    assert set(record["load_average_1min"]) == {"start", "end"}
    assert record["canonical_digest"] == "canonical-202"


def test_a_dt_tier_name_the_trainer_refuses_is_refused_before_anything_is_read_or_written(
    synthetic: tuple[Path, Any], tmp_path: Path
) -> None:
    # Q-B1 (docs/plans/p9.3.md section 10.2, C2): tier_sweep.train_tier_dt accepts the grid's tier names only.
    root, spec = synthetic
    out = tmp_path / "p9_out"
    renamed = dataclasses.replace(spec, tier="random_fresh")
    with pytest.raises(ValueError, match="Q-B1"):
        pb.train_cell(renamed, "dt_nomix", 101, corpus_root=root, output_root=out, device=CPU, gradient_steps=2)
    assert not out.exists()


def test_the_campaigns_output_root_takes_the_registered_steps_only(tmp_path: Path) -> None:
    pb.assert_campaign_steps(hz.MAIN_OUTPUT_ROOT, 40_000)
    pb.assert_campaign_steps(tmp_path / "p9_3_timing", 2_000)
    with pytest.raises(ValueError, match="40000 steps, never 2000"):
        pb.assert_campaign_steps(hz.MAIN_OUTPUT_ROOT, 2_000)
    namespace = pb.build_parser().parse_args(["train", "--tier", "maxpressure", "--arm", "bc_m", "--seed", "101"])
    assert (namespace.gradient_steps, namespace.device, namespace.torch_threads) == (40_000, "cuda", 1)
    assert (namespace.output_root, namespace.corpus_root) == (str(hz.MAIN_OUTPUT_ROOT), str(hz.MAIN_CORPUS_ROOT))


# ======================================================================
# Live, on hz4x4's own corpus (the main tree's datasets_v11): two CPU steps
# ======================================================================

HZ_CORPUS = hz.MAIN_CORPUS_ROOT

needs_hz4x4_corpus = pytest.mark.skipif(
    not (HZ_CORPUS / "cf_hz4x4__maxpressure" / "manifest.json").is_file(),
    reason="hz4x4's collected corpus (the main tree's datasets_v11/cf_hz4x4__*) is not on this machine",
)


@needs_hz4x4_corpus
def test_two_live_steps_on_hz4x4_train_under_its_own_scenario_id_and_road_network(tmp_path: Path) -> None:
    spec = pb.hz4x4_tier_spec("maxpressure")
    out = tmp_path / "p9_3_live"
    bc = pb.train_cell(spec, "bc_m", 101, corpus_root=HZ_CORPUS, output_root=out, device=CPU, gradient_steps=2)
    assert (bc["scenario_id"], bc["training_streams"], bc["gradient_steps"], bc["batch"]) == ("cityflow4x4", 3200, 2,
                                                                                              1024)
    assert bc["dataset_dirs"] == [str(HZ_CORPUS / "cf_hz4x4__maxpressure")]
    bc_payload = torch.load(bc["checkpoint_path"], map_location="cpu", weights_only=False)
    assert bc_payload["scenario_id"] == "cityflow4x4"
    dt = pb.train_cell(spec, "dt_nomix", 101, corpus_root=HZ_CORPUS, output_root=out, device=CPU, gradient_steps=2)
    payload = torch.load(dt["checkpoint_path"], map_location="cpu", weights_only=False)
    roadnet = REPO / "scenarios" / "hangzhou_4x4_gudang_18041610_1h" / "roadnet_4X4.json"
    assert payload["provenance"]["roadnet_sha256"] == hashlib.sha256(roadnet.read_bytes()).hexdigest()
    assert (payload["provenance"]["scenario"], payload["provenance"]["tier"]) == ("hz4x4", "maxpressure")
    assert payload["provenance"]["spatial_mixing"] is False
    assert len(payload["intersection_ids"]) == 16
