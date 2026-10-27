"""Tests for P9.3 Phase B's evaluation (``BRIEF_48`` E.3.6; ``docs/plans/p9.3.md`` section 10.3) -- written before
the code they test.

* the 25 evaluation cells: 20 learner cells (five arms x four tiers) and the five references of the plan (A-B3);
* the record path names the arm, the seed and the draw; a record is written once and read back by identity;
* one episode is evaluated by BOTH routes -- route (i) ``dt_gate.evaluate_arm`` for ONE draw, route (ii)
  ``admission_probe.probe_episode`` -- and the record STOPS unless route (ii)'s ``att_ours`` equals route (i)'s under
  ``==`` (``att_rederivation.assert_reproduces_committed``);
* ``evaluate_arm`` called once per draw equals one call over the draws (the per-episode watchdog relies on it);
* every factory is the registered one: a learner's checkpoint is refused unless its sha256 is its training record's,
  the matched baselines load through their BASE method (an ``iql_m`` checkpoint is an IQL agent), the DT acts with
  ``explore=False, update_memory=True``, and the five references are the plan's;
* the worker resumes from the records on disk and a non-reproducing episode stops it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import offline.hz4x4_ladder as hz
import offline.hz4x4_phase_b as pb

REPO = Path(__file__).resolve().parents[1]
SCENARIO_DIR = REPO / "scenarios" / "hangzhou_4x4_gudang_18041610_1h"
SOURCE_FLOW_NAME = "hangzhou_4x4_gudang_18041610_1h.json"


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
    except ImportError:
        return False
    return True


needs_cityflow = pytest.mark.skipif(not _cityflow_available(), reason="the native cityflow engine is not importable here")
needs_hz4x4_corpus = pytest.mark.skipif(
    not (hz.MAIN_CORPUS_ROOT / "cf_hz4x4__fixedtime" / "manifest.json").is_file()
    or not (hz.MAIN_DRAWS_ROOT / hz.SCENARIO_KEY).is_dir(),
    reason="hz4x4's corpus and held-out draws (the main tree's datasets_v11 and scenarios/draws) are not on this machine",
)


# ======================================================================
# The cells and the record path
# ======================================================================


def test_the_evaluation_cells_are_twenty_learner_cells_and_five_references() -> None:
    cells = pb.evaluation_cells()
    assert len(cells) == 25 and len(set(cells)) == 25
    learners = [cell for cell in cells if cell.method in pb.ARMS]
    assert [(c.method, c.tier) for c in learners] == [(arm, tier) for tier in hz.TIER_ORDER for arm in pb.ARMS]
    assert [c.arm for c in cells if c.method not in pb.ARMS] == [
        "behaviour@maxpressure", "behaviour@fixedtime", "behaviour@mappo1000", "behaviour@random_fresh",
        "random@mappo1000"]
    assert learners[0].arm == "dt_nomix@maxpressure"


def test_the_record_path_names_the_arm_the_seed_and_the_draw(tmp_path: Path) -> None:
    cell = pb.EvalCell("bc_m", "random_fresh")
    assert pb.eval_record_path(tmp_path, cell, 303, 1042) == (
        tmp_path / "p9_3" / "eval" / "bc_m@random_fresh__seed303__draw1042.json")


# ======================================================================
# One episode, both routes, the == stop
# ======================================================================


def _route_i(att: float) -> Any:
    from offline.dt_gate import EpisodeResult

    def evaluate_arm(**kwargs: Any) -> list[Any]:
        calls.append(("i", kwargs))
        (draw,) = kwargs["draw_ids"]
        return [EpisodeResult(arm=kwargs["arm"], seed=kwargs["seed"], draw_id=int(draw), att_horizon=att,
                              horizon_vehicle_count=12.0, episode_reward=-7.5)]

    calls: list[tuple[str, dict[str, Any]]] = []
    return evaluate_arm, calls


def _route_ii(att_ours: float, calls: list[tuple[str, dict[str, Any]]]) -> Any:
    from offline.admission_probe import AdmissionEpisode

    def probe_episode(**kwargs: Any) -> Any:
        calls.append(("ii", kwargs))
        return AdmissionEpisode(
            scenario=kwargs["scenario"], tier=kwargs["tier"], method=kwargs["method"], arm=kwargs["arm"],
            seed=kwargs["seed"], draw_id=kwargs["draw_id"], created=2695, entered=2690, never_entered=5,
            entered_fraction=2690 / 2695, completed_at_horizon=2600, running_at_horizon=90, waiting_at_horizon=5,
            att_ours=att_ours, att_engine=260.25, horizon_vehicle_count=12.0, episode_reward=-7.5, seconds=2.0,
            seconds_rollout=1.5)

    return probe_episode


def _stub_routes(monkeypatch: pytest.MonkeyPatch, att_i: float, att_ii: float) -> list[tuple[str, dict[str, Any]]]:
    import offline.admission_probe as ap
    import offline.dt_gate as dg

    evaluate_arm, calls = _route_i(att_i)
    monkeypatch.setattr(dg, "evaluate_arm", evaluate_arm)
    monkeypatch.setattr(ap, "probe_episode", _route_ii(att_ii, calls))
    return calls


def test_one_episode_is_recorded_by_both_routes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _stub_routes(monkeypatch, 250.5, 250.5)
    factory, settings, config = object(), {"max_steps": 360, "delta_time": 10}, tmp_path / "cityflow.json"
    cell = pb.EvalCell("iql_m", "fixedtime")
    record = pb.evaluate_episode(cell, 404, 1007, factory=factory, settings=settings, config_path=config, created=2695,
                                 watchdog_seconds=30)
    (route_i, kw_i), (route_ii, kw_ii) = calls
    assert (route_i, route_ii) == ("i", "ii")
    assert kw_i["draw_ids"] == [1007] and kw_i["config_for_draw"](1007) == config
    assert (kw_i["arm"], kw_i["seed"], kw_i["engine_seed"], kw_i["scenario_id"]) == ("iql_m@fixedtime", 404, 1000,
                                                                                      "cityflow4x4")
    assert kw_i["choose_action_factory"] is factory and kw_ii["choose_action_factory"] is factory
    assert kw_i["env_settings"] == settings and kw_ii["env_settings"] == settings
    assert (kw_ii["scenario"], kw_ii["scenario_id"], kw_ii["engine_seed"], kw_ii["created"]) == ("hz4x4", "cityflow4x4",
                                                                                                 1000, 2695)
    assert (kw_ii["arm"], kw_ii["method"], kw_ii["tier"], kw_ii["seed"], kw_ii["draw_id"]) == (
        "iql_m@fixedtime", "iql_m", "fixedtime", 404, 1007)
    assert record["format_version"] == pb.ARTIFACT_FORMAT_EVAL
    assert (record["arm"], record["method"], record["tier"], record["seed"], record["draw_id"]) == (
        "iql_m@fixedtime", "iql_m", "fixedtime", 404, 1007)
    assert record["route_i"] == {"att_horizon": 250.5, "horizon_vehicle_count": 12.0, "episode_reward": -7.5}
    assert record["att_ours"] == 250.5 and record["att_engine"] == 260.25
    assert (record["created"], record["entered"], record["never_entered"]) == (2695, 2690, 5)
    assert record["route_ii"]["att_ours"] == record["route_i"]["att_horizon"]
    assert record["reproduces"] is True


def test_a_route_ii_att_ours_that_does_not_reproduce_route_i_stops(monkeypatch: pytest.MonkeyPatch,
                                                                   tmp_path: Path) -> None:
    _stub_routes(monkeypatch, 250.5, float(np.nextafter(250.5, 251.0)))
    with pytest.raises(pb.ReproductionError, match="does not reproduce"):
        pb.evaluate_episode(pb.EvalCell("bc_m", "maxpressure"), 101, 1000, factory=object(), settings={},
                            config_path=tmp_path / "c.json", created=1, watchdog_seconds=30)


@needs_cityflow
@needs_hz4x4_corpus
def test_one_evaluate_arm_call_per_draw_equals_one_call_over_the_draws() -> None:
    from offline.dt_gate import _maxpressure_factory, env_settings_from_manifest, evaluate_arm
    from offline.materialise_draws import draw_config_path

    settings = env_settings_from_manifest(hz.MAIN_CORPUS_ROOT / "cf_hz4x4__maxpressure" / "manifest.json")

    def config(draw: int) -> Path:
        return Path(draw_config_path(hz.SCENARIO_KEY, draw, out_root=hz.MAIN_DRAWS_ROOT))

    common = dict(arm="behaviour@maxpressure", seed=None, config_for_draw=config, env_settings=settings,
                  scenario_id=hz.SCENARIO_ID, choose_action_factory=_maxpressure_factory, engine_seed=hz.BASE_SEED)
    together = evaluate_arm(draw_ids=[1000, 1001], **common)
    apart = [evaluate_arm(draw_ids=[draw], **common)[0] for draw in (1000, 1001)]
    assert together == apart


# ======================================================================
# The factories
# ======================================================================


class _StubIx:
    def __init__(self, index: int) -> None:
        self.id = f"ix{index}"
        self.num_phases = 8


class _StubEnv:
    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = None if config_path is None else str(Path(config_path).resolve())
        self.intersections = [_StubIx(i) for i in range(3)]
        self.action_space = None


def _actions(choose: Any, env: Any, steps: int = 12) -> list[list[int]]:
    info = {"intersections": {ix.id: {"avail_actions": list(range(8))} for ix in env.intersections}}
    return [[int(a) for a in choose(env, info)] for _ in range(steps)]


@pytest.fixture()
def tmp_scenario_config(tmp_path: Path) -> Path:
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    for name in ("roadnet_4X4.json", SOURCE_FLOW_NAME):
        shutil.copyfile(SCENARIO_DIR / name, scenario / name)
    cfg = json.loads((REPO / "configs" / "sim" / "cityflow4x4.json").read_text(encoding="utf-8"))
    cfg["dir"] = str(scenario) + "/"
    config = tmp_path / "cityflow4x4.json"
    config.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    return config


def test_the_random_references_are_the_registered_and_the_fresh_generators(tmp_path: Path,
                                                                           tmp_scenario_config: Path) -> None:
    from offline.collect import _make_random
    from offline.materialise_draws import draw_config_path

    draws_root = tmp_path / "draws"
    hz.materialise_hz4x4_draws([1000, 1001], source_config=tmp_scenario_config, out_root=draws_root)
    roots = dict(output_root=tmp_path / "out", corpus_root=tmp_path / "datasets_v11", draws_root=draws_root,
                 device="cpu")
    env = _StubEnv(draw_config_path(hz.SCENARIO_KEY, 1001, out_root=draws_root))
    registered, source = pb.eval_factory(pb.EvalCell("random", "mappo1000"), 202, **roots)
    reference = _make_random(env, SimpleNamespace(), np.random.default_rng(202))
    assert _actions(registered(env), env) == _actions(lambda _e, info: reference(info), env)
    assert "default_rng(training seed)" in source["policy"]
    fresh, source = pb.eval_factory(pb.EvalCell("behaviour", "random_fresh"), 303, **roots)
    reference = _make_random(env, SimpleNamespace(), np.random.default_rng([303, 1001]))
    assert _actions(fresh(env), env) == _actions(lambda _e, info: reference(info), env)
    assert "default_rng([seed, draw_id])" in source["policy"]
    assert _actions(fresh(env), env) != _actions(registered(env), env)


def test_maxpressure_is_the_registered_controller(tmp_path: Path) -> None:
    from offline.dt_gate import _maxpressure_factory

    factory, source = pb.eval_factory(pb.EvalCell("behaviour", "maxpressure"), 101, output_root=tmp_path,
                                      corpus_root=tmp_path / "datasets_v11", draws_root=tmp_path, device="cpu")
    assert factory is _maxpressure_factory and "MaxPressure" in source["policy"]


def _teacher_tree(tmp_path: Path, seed: int, *, digest: str | None = None) -> Path:
    corpus = tmp_path / "datasets_v11"
    checkpoint = Path("output") / "checkpoints" / "p9_3_mappo_nominal_1000" / f"teacher_seed{seed}.pt"
    (tmp_path / checkpoint).parent.mkdir(parents=True)
    (tmp_path / checkpoint).write_bytes(b"a teacher checkpoint " + str(seed).encode())
    directory = corpus / f"cf_hz4x4__mappo1000__seed{seed}"
    directory.mkdir(parents=True)
    sha = digest or hashlib.sha256((tmp_path / checkpoint).read_bytes()).hexdigest()
    meta = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha}
    (directory / "manifest.json").write_text(json.dumps({"run_metadata": meta, "episodes": []}), encoding="utf-8")
    return corpus


def test_the_teacher_of_seed_s_is_slot_s_and_its_digest_is_the_manifests(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    import offline.dt_gate as dg

    seen: list[tuple[str, Any]] = []
    monkeypatch.setattr(dg, "_mappo_factory", lambda path, device: seen.append((path, device)) or "teacher")
    corpus = _teacher_tree(tmp_path, 404)
    factory, source = pb.eval_factory(pb.EvalCell("behaviour", "mappo1000"), 404, output_root=tmp_path / "out",
                                      corpus_root=corpus, draws_root=tmp_path, device="cpu")
    expected = tmp_path / "output" / "checkpoints" / "p9_3_mappo_nominal_1000" / "teacher_seed404.pt"
    assert factory == "teacher" and seen == [(str(expected), "cpu")]
    assert source["checkpoint_sha256"] == hashlib.sha256(expected.read_bytes()).hexdigest()
    with pytest.raises(FileNotFoundError, match="cf_hz4x4__mappo1000__seed505"):
        pb.eval_factory(pb.EvalCell("behaviour", "mappo1000"), 505, output_root=tmp_path / "out", corpus_root=corpus,
                        draws_root=tmp_path, device="cpu")
    other = tmp_path / "other"
    other.mkdir()
    tampered = _teacher_tree(other, 404, digest="0" * 64)
    with pytest.raises(ValueError, match="is not the checkpoint"):
        pb.eval_factory(pb.EvalCell("behaviour", "mappo1000"), 404, output_root=tmp_path / "out", corpus_root=tampered,
                        draws_root=tmp_path, device="cpu")


def _trained_cell(out: Path, tier: str, arm: str, seed: int, *, payload: bytes = b"weights") -> Path:
    path = pb.checkpoint_path(out, tier, arm, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    record = pb.train_record_path(out, tier, arm, seed)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"checkpoint_path": str(path), "file_sha256": hashlib.sha256(payload).hexdigest(),
                                  "declared_gradient_steps": 40_000, "arm": arm, "tier": tier, "seed": seed}),
                      encoding="utf-8")
    return path


@pytest.mark.parametrize("arm, base", [("bc_m", "bc"), ("bc_top10_m", "bc_top10"),
                                       ("bc_top10_perix_m", "bc_top10_perix"), ("iql_m", "iql")])
def test_a_matched_baseline_loads_through_its_base_method_from_its_verified_checkpoint(
    arm: str, base: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import offline.offline_baselines as ob

    seen: list[tuple[Any, ...]] = []
    monkeypatch.setattr(ob, "_baseline_factory", lambda *args: seen.append(args) or "loaded")
    out = tmp_path / "out"
    path = _trained_cell(out, "fixedtime", arm, 202)
    factory, source = pb.eval_factory(pb.EvalCell(arm, "fixedtime"), 202, output_root=out,
                                      corpus_root=tmp_path / "datasets_v11", draws_root=tmp_path, device="cpu")
    assert factory == "loaded" and seen == [(base, str(path), 40_000, "cpu")]
    assert source["file_sha256"] == hashlib.sha256(b"weights").hexdigest()
    path.write_bytes(b"other weights")
    with pytest.raises(ValueError, match="is not the checkpoint"):
        pb.eval_factory(pb.EvalCell(arm, "fixedtime"), 202, output_root=out, corpus_root=tmp_path / "datasets_v11",
                        draws_root=tmp_path, device="cpu")
    with pytest.raises(FileNotFoundError, match="is missing; a cell is evaluated only once it is trained"):
        pb.eval_factory(pb.EvalCell(arm, "fixedtime"), 303, output_root=out, corpus_root=tmp_path / "datasets_v11",
                        draws_root=tmp_path, device="cpu")


def test_the_dt_acts_without_exploration_and_with_its_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.SpatialDTAgent import SpatialDTAgent

    acted: list[dict[str, Any]] = []

    class _Agent:
        def act(self, info: Any, **kwargs: Any) -> str:
            acted.append(kwargs)
            return "action"

    loaded: list[tuple[Any, ...]] = []
    monkeypatch.setattr(SpatialDTAgent, "from_checkpoint",
                        classmethod(lambda cls, env, path, device=None: loaded.append((env, path, device)) or _Agent()))
    out = tmp_path / "out"
    path = _trained_cell(out, "mappo1000", "dt_nomix", 505)
    factory, _ = pb.eval_factory(pb.EvalCell("dt_nomix", "mappo1000"), 505, output_root=out,
                                 corpus_root=tmp_path / "datasets_v11", draws_root=tmp_path, device="cpu")
    env = object()
    assert factory(env)(env, {"t": 0}) == "action"
    assert loaded == [(env, str(path), "cpu")] and acted == [{"explore": False, "update_memory": True}]


@needs_hz4x4_corpus
def test_the_fixed_time_reference_is_the_collected_k_on_the_first_held_out_draw(monkeypatch: pytest.MonkeyPatch,
                                                                                tmp_path: Path) -> None:
    import offline.method_tier_grid as mtg
    from offline.materialise_draws import draw_config_path

    seen: list[tuple[Any, ...]] = []
    monkeypatch.setattr(mtg, "_fixedtime_factory", lambda *args: seen.append(args) or "fixed")
    factory, source = pb.eval_factory(pb.EvalCell("behaviour", "fixedtime"), 101, output_root=tmp_path,
                                      corpus_root=hz.MAIN_CORPUS_ROOT, draws_root=hz.MAIN_DRAWS_ROOT, device="cpu")
    ((config, collected),) = seen
    assert factory == "fixed"
    assert Path(config) == Path(draw_config_path(hz.SCENARIO_KEY, 1000, out_root=hz.MAIN_DRAWS_ROOT))
    assert collected == mtg.fixedtime_collection_settings(hz.MAIN_CORPUS_ROOT / "cf_hz4x4__fixedtime" / "manifest.json")
    assert collected["fixed_time_k"] == 2 and source["fixed_time_k"] == 2


# ======================================================================
# Records on disk: written once, read back by identity; the worker's stop
# ======================================================================


def _record(cell: Any, seed: int, draw: int) -> dict[str, Any]:
    return {"format_version": pb.ARTIFACT_FORMAT_EVAL, "arm": cell.arm, "method": cell.method, "tier": cell.tier,
            "seed": seed, "draw_id": draw, "att_ours": 1.0, "reproduces": True}


def test_pending_episodes_resume_from_the_records_and_refuse_a_misfiled_one(tmp_path: Path) -> None:
    cells = (pb.EvalCell("bc_m", "maxpressure"), pb.EvalCell("behaviour", "fixedtime"))
    plan = pb.planned_episodes(cells, seeds=(101, 202), draws=(1000, 1001))
    assert len(plan) == 8 and plan[0] == (cells[0], 101, 1000) and plan[-1] == (cells[1], 202, 1001)
    for cell, seed, draw in plan[:3]:
        pb.write_eval_record(pb.eval_record_path(tmp_path, cell, seed, draw), _record(cell, seed, draw))
    assert pb.pending_episodes(tmp_path, plan) == plan[3:]
    with pytest.raises(FileExistsError, match="written once"):
        cell, seed, draw = plan[0]
        pb.write_eval_record(pb.eval_record_path(tmp_path, cell, seed, draw), _record(cell, seed, draw))
    cell, seed, draw = plan[3]
    assert (seed, draw) == (202, 1001)
    misfiled = _record(cell, seed, draw + 1)
    pb.eval_record_path(tmp_path, cell, seed, draw).write_text(json.dumps(misfiled), encoding="utf-8")
    with pytest.raises(ValueError, match="filed"):
        pb.pending_episodes(tmp_path, plan)


def test_the_worker_stops_on_the_first_non_reproducing_episode_and_files_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells = (pb.EvalCell("random", "mappo1000"),)
    plan = pb.planned_episodes(cells, seeds=(101,), draws=(1000, 1001, 1002))
    rolled: list[int] = []

    def evaluate(cell: Any, seed: int, draw: int) -> dict[str, Any]:
        rolled.append(draw)
        if draw == 1001:
            raise pb.ReproductionError("route (ii) att_ours 1.0000001 does not reproduce route (i)'s 1.0")
        return _record(cell, seed, draw)

    with pytest.raises(pb.ReproductionError, match="does not reproduce"):
        pb.run_episodes(tmp_path, plan, evaluate)
    assert rolled == [1000, 1001]
    assert pb.pending_episodes(tmp_path, plan) == plan[1:]
    refusal = pb.refusal_record_path(tmp_path, cells[0], 101, 1001)
    assert "does not reproduce" in json.loads(refusal.read_bytes())["refusal"]


# ======================================================================
# The command line: shards and the registered watchdog
# ======================================================================


def test_the_shards_interleave_and_cover_every_episode_exactly_once() -> None:
    every = pb.planned_episodes(pb.evaluation_cells(), draws=(1000, 1001))
    assert len(every) == 25 * 5 * 2
    shards = [pb.selected_plan(None, [1000, 1001], shard=i, of=3) for i in range(3)]
    assert sorted(map(repr, (e for shard in shards for e in shard))) == sorted(map(repr, every))
    assert sum(len(shard) for shard in shards) == len(every)
    assert shards[0][:2] == [every[0], every[3]]
    assert pb.selected_plan(["iql_m@random_fresh"], None, shard=0, of=1) == pb.planned_episodes(
        [pb.EvalCell("iql_m", "random_fresh")])
    with pytest.raises(ValueError, match="unknown evaluation cell"):
        pb.selected_plan(["iql@random_fresh"], None, shard=0, of=1)
    with pytest.raises(ValueError, match="is not a shard"):
        pb.selected_plan(None, None, shard=3, of=3)


def test_the_worker_and_the_driver_are_subcommands_and_the_worker_takes_the_registered_watchdog() -> None:
    worker = pb.build_parser().parse_args(["evaluate-worker"])
    assert (worker.watchdog_seconds, worker.device, worker.shard, worker.of) == (hz.WATCHDOG_SECONDS, "cpu", 0, 1)
    driver = pb.build_parser().parse_args(["evaluate", "--cells", "dt_nomix@maxpressure", "--draws", "1000", "1001"])
    assert (driver.cells, driver.draws, driver.output_root) == (["dt_nomix@maxpressure"], [1000, 1001],
                                                                str(hz.MAIN_OUTPUT_ROOT))


# ======================================================================
# End to end on hz4x4: two CPU training steps, then one held-out episode by both routes
# ======================================================================


@pytest.fixture(scope="module")
def two_step_cells(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import torch

    out = tmp_path_factory.mktemp("p9_3_e2e")
    spec = pb.hz4x4_tier_spec("maxpressure")
    for arm in ("bc_m", "dt_nomix"):
        pb.train_cell(spec, arm, 101, corpus_root=hz.MAIN_CORPUS_ROOT, output_root=out, device=torch.device("cpu"),
                      gradient_steps=2)
    return out


@needs_cityflow
@needs_hz4x4_corpus
@pytest.mark.parametrize("arm", ["bc_m", "dt_nomix"])
def test_a_trained_cell_is_evaluated_end_to_end_by_both_routes(arm: str, two_step_cells: Path) -> None:
    from offline.admission_probe import created_from_flow
    from offline.materialise_draws import draw_config_path

    cell = pb.EvalCell(arm, "maxpressure")
    settings = pb.evaluation_settings(hz.MAIN_CORPUS_ROOT)
    factory, source = pb.eval_factory(cell, 101, output_root=two_step_cells, corpus_root=hz.MAIN_CORPUS_ROOT,
                                      draws_root=hz.MAIN_DRAWS_ROOT, device="cpu")
    config = Path(draw_config_path(hz.SCENARIO_KEY, 1000, out_root=hz.MAIN_DRAWS_ROOT))
    created = created_from_flow(config.parent / "flow.json",
                                horizon_seconds=int(settings["max_steps"]) * int(settings["delta_time"]))
    record = pb.evaluate_episode(cell, 101, 1000, factory=factory, settings=settings, config_path=config,
                                 created=created)
    assert record["reproduces"] is True and record["att_ours"] == record["route_i"]["att_horizon"]
    assert record["created"] == created == record["entered"] + record["never_entered"]
    assert np.isfinite(record["att_engine"]) and record["att_engine"] > 0.0
    assert source["file_sha256"] == hashlib.sha256(Path(source["checkpoint_path"]).read_bytes()).hexdigest()


@needs_hz4x4_corpus
def test_the_evaluation_settings_are_every_tiers_and_a_disagreement_is_refused(tmp_path: Path) -> None:
    corpus = tmp_path / "datasets_v11"
    for tier in hz.TIER_ORDER:
        for name in hz._tier_dir_names(tier):
            meta = json.loads((hz.MAIN_CORPUS_ROOT / name / "manifest.json").read_bytes())["run_metadata"]
            (corpus / name).mkdir(parents=True)
            (corpus / name / "manifest.json").write_text(json.dumps({"run_metadata": meta}), encoding="utf-8")
    settings = pb.evaluation_settings(corpus)
    assert settings == pb.evaluation_settings(hz.MAIN_CORPUS_ROOT)
    assert (settings["max_steps"], settings["delta_time"]) == (360, 10)
    name = hz._tier_dir_names("random_fresh")[0]
    payload = json.loads((corpus / name / "manifest.json").read_bytes())
    payload["run_metadata"]["delta_time"] = 5
    (corpus / name / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        pb.evaluation_settings(corpus)
