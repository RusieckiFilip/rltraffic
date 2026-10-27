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


def _ts_args(tmp_path: Path, corpus: Path, *command: str, root: str = "p5_2_like", tier: str = "random") -> Any:
    return ts.build_parser().parse_args([
        "--corpus-root", str(corpus), "--work-dir", str(tmp_path / root / "work"),
        "--checkpoint-dir", str(tmp_path / root / "checkpoints"), "--reuse-root", str(tmp_path / "protected"),
        "--device", "cpu", "--seeds", "101", "--gradient-steps", "2", "--tier", tier, *command,
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


def _as_hz4x4_tier(spec: Any, monkeypatch: pytest.MonkeyPatch, tier: str = "maxpressure") -> Any:
    """The synthetic tier under an hz4x4 tier name, installed in tier_sweep's table so ``tier_sweep train`` reads it."""
    renamed = dataclasses.replace(spec, tier=tier)
    monkeypatch.setattr(ts, "TIERS", {tier: renamed})
    return renamed


def test_train_cell_hands_the_dt_mirror_what_tier_sweep_train_hands_train_tier_dt(
    synthetic: tuple[Path, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # BRIEF_48 E.3.6; docs/plans/p9.3.md section 10.2 C2 as amended: train_tier_dt writes the GRID's scenario id into
    # every checkpoint, so hz4x4's DT trains through hz4x4_train_tier_dt (its body, two constants made parameters).
    root, spec = synthetic
    spec = _as_hz4x4_tier(spec, monkeypatch)
    registered, mirror = _DtSpy(), _DtSpy()
    monkeypatch.setattr(ts, "train_tier_dt", registered)
    monkeypatch.setattr(pb, "hz4x4_train_tier_dt", mirror)
    assert ts._run_train(_ts_args(tmp_path, root, "train", "--method", "dt_nomix", tier="maxpressure")) == 0
    out = tmp_path / "p9_out"
    record = pb.train_cell(spec, "dt_nomix", 101, corpus_root=root, output_root=out, device=CPU, gradient_steps=2)
    (theirs,), (mine,) = registered.calls, mirror.calls
    assert _same(mine.pop("stacked"), theirs.pop("stacked"))
    assert _same(mine.pop("index"), theirs.pop("index"))
    assert mine.pop("stats").to_json_obj() == theirs.pop("stats").to_json_obj()
    final = pb.checkpoint_path(out, "maxpressure", "dt_nomix", 101)
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
    assert (mine.pop("scenario_id"), tuple(mine.pop("tiers"))) == ("cityflow_grid4x4", hz.TIER_ORDER)
    assert mine == theirs
    assert (mine["tier"], mine["method"], mine["gradient_steps"], mine["batch_size"]) == ("maxpressure", "dt_nomix", 2,
                                                                                           64)
    assert record["seconds"] == 30.0 + 101 / 100.0
    assert record["canonical_digest"] == ts.canonical_state_dict_digest(final)


def _synthetic_adjacency() -> Any:
    """A real ``AdjacencySpec`` over the synthetic tier's 16 ids: the 4 x 4 lattice's undirected neighbours."""
    from offline.roadnet_graph import AdjacencySpec

    size = len(fx.IDS)
    directed = np.zeros((size, size), dtype=np.bool_)
    for index in range(size):
        row, col = divmod(index, 4)
        for other_row, other_col in ((row, col + 1), (row + 1, col)):
            if other_row < 4 and other_col < 4:
                directed[index, other_row * 4 + other_col] = True
    return AdjacencySpec(node_ids=tuple(fx.IDS), directed=directed, undirected=directed | directed.T,
                         roadnet_path="synthetic-lattice", roadnet_sha256="0" * 64)


def test_the_dt_mirror_is_train_tier_dt_bit_for_bit_on_cpu(
    synthetic: tuple[Path, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, spec = synthetic
    monkeypatch.setattr(ts, "adjacency_for_tier", lambda *args, **kwargs: _synthetic_adjacency())
    parts = pb.hz4x4_tier_parts(spec, root)
    common = dict(tier="random", method="dt_nomix", seed=202, adjacency=parts["adjacency"], prompts=parts["prompts"],
                  stats=parts["dataset"].stats, state_dim=int(parts["index"].state_dim),
                  n_actions=int(parts["index"].n_actions), gradient_steps=3, batch_size=4, device=CPU, protected=(),
                  provenance={"runtime": {"fixed": "for the comparison"}}, deterministic=False, log_every=0)
    paths = {name: tmp_path / name / "dt.pt" for name in ("registered", "mirror", "hz4x4")}
    for path in paths.values():
        path.parent.mkdir()
    registered = ts.train_tier_dt(parts["stack"](), parts["index"], checkpoint_path=paths["registered"], **common)
    mirror = pb.hz4x4_train_tier_dt(parts["stack"](), parts["index"], checkpoint_path=paths["mirror"],
                                    scenario_id=ts.SCENARIO_ID, tiers=tuple(ts.TIERS), **common)
    assert (mirror.losses, mirror.gradient_steps, mirror.method, mirror.seed) == (
        registered.losses, registered.gradient_steps, registered.method, registered.seed)
    assert len(mirror.losses) == 3
    load = lambda path: torch.load(path, map_location="cpu", weights_only=False)  # noqa: E731
    assert _same(load(paths["mirror"]), load(paths["registered"]))
    pb.hz4x4_train_tier_dt(parts["stack"](), parts["index"], checkpoint_path=paths["hz4x4"], scenario_id="cityflow4x4",
                           tiers=tuple(ts.TIERS), **common)
    hz4x4, grid = load(paths["hz4x4"]), load(paths["registered"])
    assert (hz4x4.pop("scenario_id"), grid.pop("scenario_id")) == ("cityflow4x4", "cityflow_grid4x4")
    assert _same(hz4x4, grid)
    with pytest.raises(ValueError, match="unknown tier 'random'"):
        pb.hz4x4_train_tier_dt(parts["stack"](), parts["index"], checkpoint_path=tmp_path / "x.pt",
                               scenario_id="cityflow4x4", tiers=hz.TIER_ORDER, **common)


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


def test_a_dt_tier_name_outside_hz4x4s_tiers_is_refused_before_anything_is_read_or_written(
    synthetic: tuple[Path, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The mirror accepts hz4x4's four tier names -- random_fresh included (Q-B1 no longer blocks hz4x4) -- and nothing
    # else; the refusal precedes the first read and the first write.
    root, spec = synthetic
    out = tmp_path / "p9_out"
    with pytest.raises(ValueError, match="not one of hz4x4's tiers"):
        pb.train_cell(spec, "dt_nomix", 101, corpus_root=root, output_root=out, device=CPU, gradient_steps=2)
    assert not out.exists()
    mirror = _DtSpy()
    monkeypatch.setattr(pb, "hz4x4_train_tier_dt", mirror)
    pb.train_cell(_as_hz4x4_tier(spec, monkeypatch, "random_fresh"), "dt_nomix", 101, corpus_root=root,
                  output_root=out, device=CPU, gradient_steps=2)
    assert [call["tier"] for call in mirror.calls] == ["random_fresh"]


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
    assert payload["scenario_id"] == "cityflow4x4" and "cityflow4x4" in json.dumps(payload["stats"])
    assert len(payload["intersection_ids"]) == 16


# ======================================================================
# The training queue: one cell at a time, resumable, stopping at the first failure (E.3.7)
# ======================================================================


def test_the_training_cells_are_the_hundred_tier_by_tier() -> None:
    cells = pb.training_cells()
    assert len(cells) == 100 and len(set(cells)) == 100
    assert cells[:6] == [("maxpressure", "dt_nomix", s) for s in hz.SEEDS] + [("maxpressure", "bc_m", 101)]
    assert [tier for tier, _, _ in cells[::25]] == list(hz.TIER_ORDER)
    assert pb.training_cells(tiers=["random_fresh"], arms=["iql_m"]) == [("random_fresh", "iql_m", s) for s in hz.SEEDS]


def _done(out: Path, tier: str, arm: str, seed: int, payload: bytes = b"w") -> None:
    path = pb.checkpoint_path(out, tier, arm, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    record = pb.train_record_path(out, tier, arm, seed)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"file_sha256": hashlib.sha256(payload).hexdigest()}), encoding="utf-8")


def test_the_queue_skips_finished_cells_trains_the_rest_in_order_and_stops_at_the_first_failure(tmp_path: Path) -> None:
    cells = [("fixedtime", "bc_m", 101), ("fixedtime", "bc_m", 202), ("fixedtime", "iql_m", 101),
             ("fixedtime", "iql_m", 202)]
    _done(tmp_path, *cells[0])
    ran: list[tuple[str, str, int]] = []

    def run_cell(tier: str, arm: str, seed: int) -> int:
        ran.append((tier, arm, seed))
        return 7 if (arm, seed) == ("iql_m", 101) else 0

    result = pb.run_train_queue(cells, output_root=tmp_path, run_cell=run_cell)
    assert ran == cells[1:3]
    assert result == {"status": "failed", "skipped": 1, "trained": 1, "failed_cell": list(cells[2]), "returncode": 7}
    ran.clear()
    _done(tmp_path, *cells[1])
    _done(tmp_path, *cells[2])
    assert pb.run_train_queue(cells, output_root=tmp_path, run_cell=lambda *cell: ran.append(cell) or 0) == {
        "status": "completed", "skipped": 3, "trained": 1, "failed_cell": None, "returncode": 0}
    assert ran == [cells[3]]


def test_the_queue_refuses_a_half_written_cell_before_running_anything(tmp_path: Path) -> None:
    cells = [("mappo1000", "bc_top10_m", 303), ("mappo1000", "bc_top10_m", 404)]
    never = lambda *cell: pytest.fail(f"{cell} was run")  # noqa: E731
    _done(tmp_path, *cells[1])
    pb.checkpoint_path(tmp_path, *cells[1]).write_bytes(b"another file")
    with pytest.raises(ValueError, match="is not the checkpoint"):
        pb.run_train_queue(cells, output_root=tmp_path, run_cell=never)
    pb.train_record_path(tmp_path, *cells[1]).unlink()
    with pytest.raises(ValueError, match="without its record"):
        pb.run_train_queue(cells, output_root=tmp_path, run_cell=never)
