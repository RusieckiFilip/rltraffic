"""Tests for ``offline.hz4x4_ladder`` -- P9.3's Phase A (A27 point 8), written before the implementation.

Order follows ``BRIEF_48`` section 5: draws, pre-flight, the fixed-time sweep, the collector, the teachers,
the A11 gate, the declaration.  Tests that need the main tree's committed corpus, the P2.1 results or the
native CityFlow engine skip with the reason when those are absent, so the suite stays green on a fresh
clone; on this machine all of them run.

Floating-point assertions use ``==`` throughout.  Where a test recomputes a mean by an independent route
it feeds exactly representable (dyadic) values, so every summation order is exact and ``==`` is a real
check of the arithmetic rather than of the order in which two routes happen to round.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import offline.hz4x4_ladder as hz

REPO = Path(__file__).resolve().parents[1]
SCENARIO_DIR = REPO / "scenarios" / "hangzhou_4x4_gudang_18041610_1h"
SOURCE_FLOW_NAME = "hangzhou_4x4_gudang_18041610_1h.json"
GRID_CORPUS = hz.MAIN_CORPUS_ROOT
P2_1_RESULTS = hz.MAIN_OUTPUT_ROOT / "experiments" / "p2_1_mappo_nominal_1000" / "results.json"


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
    except ImportError:
        return False
    return True


needs_cityflow = pytest.mark.skipif(
    not _cityflow_available(), reason="the native cityflow engine is not importable here"
)
needs_corpus = pytest.mark.skipif(
    not (GRID_CORPUS / "cf_grid4x4__maxpressure" / "manifest.json").is_file(),
    reason="the main tree's datasets_v11 grid corpus is not on this machine",
)
needs_p2_1 = pytest.mark.skipif(
    not P2_1_RESULTS.is_file(), reason="P2.1's results.json is not under the main tree's output/"
)

#: The protocol settings every grid tier carries (read from the manifests in the tests that can).
PROTOCOL_SETTINGS: dict[str, Any] = {
    "max_steps": 360,
    "delta_time": 10,
    "control_mode": "acyclic",
    "global_reward_fn": "queue_length",
    "local_reward_fn": "queue_length",
    "global_reward_weight": 0.0,
    "state_features": ["lane_vehicle_count", "lane_waiting", "phase_onehot"],
}


def _env_settings() -> dict[str, Any]:
    from experiments.config import SETTING_DEFAULTS

    settings = dict(SETTING_DEFAULTS)
    settings.update(PROTOCOL_SETTINGS)
    return settings


@pytest.fixture
def tmp_scenario(tmp_path: Path) -> Path:
    """A copy of hz4x4's roadnet and flow outside every git tree, with a config whose ``dir`` is absolute."""
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    for name in ("roadnet_4X4.json", SOURCE_FLOW_NAME):
        shutil.copyfile(SCENARIO_DIR / name, scenario / name)
    cfg = json.loads((REPO / "configs" / "sim" / "cityflow4x4.json").read_text(encoding="utf-8"))
    cfg["dir"] = str(scenario) + "/"
    config = tmp_path / "cityflow4x4.json"
    config.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    return config


def _fake_linked_worktree(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / ".git").write_text("gitdir: /nowhere/.git/worktrees/fake\n", encoding="utf-8")
    return root


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# ======================================================================
# 0. Declarations and the teacher config
# ======================================================================


def test_main_draws_root_is_the_collectors_default() -> None:
    from offline.collect import DEFAULT_DRAWS_ROOT

    assert hz.MAIN_DRAWS_ROOT == DEFAULT_DRAWS_ROOT


def test_scenario_key_is_the_materialisers_key_for_the_source_config() -> None:
    from offline.materialise_draws import scenario_key_for_config

    assert scenario_key_for_config(hz.SOURCE_CONFIG) == hz.SCENARIO_KEY


def test_teacher_config_is_p2_1_with_exactly_four_keys_changed() -> None:
    p2_1 = json.loads((REPO / "experiments/configs/p2_1_mappo_nominal_1000.json").read_text(encoding="utf-8"))
    ours = json.loads((REPO / hz.TEACHER_CONFIG).read_text(encoding="utf-8"))
    changed = {key for key in set(p2_1) | set(ours) if p2_1.get(key) != ours.get(key)}
    assert changed == {"name", "checkpoint_dir", "output_dir", "environments"}
    assert ours["environments"] == [
        {"id": "cf_hz4x4", "backend": "cityflow", "config": "../../configs/sim/cityflow4x4.json"}
    ]
    assert Path(ours["checkpoint_dir"]) / ours["name"] == hz.TEACHER_CHECKPOINT_SUBDIR
    assert Path(ours["output_dir"]) / ours["name"] / "results.json" == hz.TEACHER_RESULTS


def test_teacher_config_loads_onto_the_hz4x4_sim_config() -> None:
    from experiments.config import load_config

    config = load_config(REPO / hz.TEACHER_CONFIG)
    assert [env.id for env in config.environments] == ["cf_hz4x4"]
    assert Path(config.environments[0].paths["config"]) == (REPO / "configs/sim/cityflow4x4.json").resolve()
    assert config.environments[0].settings["train_episodes"] == hz.TEACHER_EPISODES
    assert config.seeds == hz.SEEDS


# ======================================================================
# 1. Draws: idempotence and refusal outside the main tree
# ======================================================================


def test_assert_main_tree_path_refuses_a_linked_worktree(tmp_path: Path) -> None:
    worktree = _fake_linked_worktree(tmp_path / "wt")
    with pytest.raises(ValueError, match="linked worktree"):
        hz.assert_main_tree_path(worktree / "scenarios" / "draws", what="the draws root")
    assert hz.assert_main_tree_path(tmp_path / "plain", what="a plain directory") == (tmp_path / "plain").resolve()


def test_draws_refuse_an_out_root_inside_a_linked_worktree(tmp_path: Path, tmp_scenario: Path) -> None:
    worktree = _fake_linked_worktree(tmp_path / "wt")
    with pytest.raises(ValueError, match="linked worktree"):
        hz.materialise_hz4x4_draws([0, 1], source_config=tmp_scenario, out_root=worktree / "scenarios" / "draws")
    assert not (worktree / "scenarios").exists()


def test_draws_refuse_a_source_whose_scenario_dir_is_inside_a_linked_worktree(tmp_path: Path) -> None:
    worktree = _fake_linked_worktree(tmp_path / "wt")
    scenario = worktree / "scenario"
    scenario.mkdir()
    for name in ("roadnet_4X4.json", SOURCE_FLOW_NAME):
        shutil.copyfile(SCENARIO_DIR / name, scenario / name)
    cfg = json.loads((REPO / "configs" / "sim" / "cityflow4x4.json").read_text(encoding="utf-8"))
    cfg["dir"] = str(scenario) + "/"
    config = tmp_path / "cityflow4x4.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    out_root = tmp_path / "draws"
    with pytest.raises(ValueError, match="linked worktree"):
        hz.materialise_hz4x4_draws([0], source_config=config, out_root=out_root)
    assert not out_root.exists()


def test_draws_are_idempotent_and_verified(tmp_path: Path, tmp_scenario: Path) -> None:
    out_root = tmp_path / "draws"
    first = hz.materialise_hz4x4_draws([0, 1, 1000], source_config=tmp_scenario, out_root=out_root)
    assert [r.action for r in first] == ["written", "written", "written"]
    assert [r.draw_id for r in first] == [0, 1, 1000]
    before = _tree_bytes(out_root)
    second = hz.materialise_hz4x4_draws([0, 1, 1000], source_config=tmp_scenario, out_root=out_root)
    assert [r.action for r in second] == ["kept", "kept", "kept"]
    assert _tree_bytes(out_root) == before
    source_bytes = (tmp_scenario.parent / "scenario" / SOURCE_FLOW_NAME).read_bytes()
    assert (out_root / "cityflow4x4" / "draw_0000" / "flow.json").read_bytes() == source_bytes
    entries = json.loads((out_root / "cityflow4x4" / "draw_0001" / "flow.json").read_bytes())
    assert len(entries) == first[1].n_vehicles
    assert all(entry["startTime"] == entry["endTime"] for entry in entries)


def test_draws_refuse_an_aggregate_source_and_create_nothing(tmp_path: Path, tmp_scenario: Path) -> None:
    flow = tmp_scenario.parent / "scenario" / SOURCE_FLOW_NAME
    entries = json.loads(flow.read_bytes())
    entries[5] = dict(entries[5], endTime=int(entries[5]["startTime"]) + 30)
    flow.write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(ValueError, match="single insertion"):
        hz.source_flow_entries(tmp_scenario)
    out_root = tmp_path / "draws"
    with pytest.raises(ValueError, match="single insertion"):
        hz.materialise_hz4x4_draws([0], source_config=tmp_scenario, out_root=out_root)
    assert not out_root.exists()


def test_source_flow_entries_asserts_the_vehicle_count(tmp_scenario: Path) -> None:
    path, raw, entries = hz.source_flow_entries(tmp_scenario)
    assert len(entries) == hz.SOURCE_VEHICLES == 2983
    assert raw == path.read_bytes()
    with pytest.raises(ValueError, match="2984"):
        hz.source_flow_entries(tmp_scenario, expected_vehicles=2984)


def test_draws_verification_refuses_bytes_that_differ_from_the_plan(
    tmp_path: Path, tmp_scenario: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import offline.materialise_draws as md

    real = md.materialise

    def corrupting(*args: Any, **kwargs: Any) -> Any:
        records = real(*args, **kwargs)
        records[-1].flow_path.write_text("[]\n", encoding="utf-8")
        return records

    monkeypatch.setattr(md, "materialise", corrupting)
    with pytest.raises(ValueError, match="differs"):
        hz.materialise_hz4x4_draws([1], source_config=tmp_scenario, out_root=tmp_path / "draws")


# ======================================================================
# 2. Pre-flight
# ======================================================================


class _StubIx:
    def __init__(self, index: int, n_actions: int) -> None:
        self.id = f"ix{index}"
        self.num_phases = n_actions


class _StubEnv:
    def __init__(self, counts: list[int]) -> None:
        self.intersections = [_StubIx(i, n) for i, n in enumerate(counts)]
        self.action_space = None


def test_assert_eight_actions_accepts_sixteen_by_eight_and_refuses_anything_else() -> None:
    assert hz.assert_eight_actions(_StubEnv([8] * 16)) == [8] * 16
    with pytest.raises(ValueError, match="9"):
        hz.assert_eight_actions(_StubEnv([8] * 15 + [9]))
    with pytest.raises(ValueError, match="15"):
        hz.assert_eight_actions(_StubEnv([8] * 15))


@needs_cityflow
def test_hz4x4_env_offers_eight_actions_on_every_intersection(tmp_scenario: Path) -> None:
    from experiments.config import EnvSpec
    from experiments.envs import make_env

    env = make_env(EnvSpec(id=hz.SCENARIO_ID, backend="cityflow", paths={"config": str(tmp_scenario)},
                           settings=_env_settings()))
    try:
        assert hz.assert_eight_actions(env) == [8] * 16
    finally:
        env.close()


def test_go_no_go_is_inclusive_at_one_day() -> None:
    assert hz.ONE_DAY_SECONDS == 86_400
    assert hz.go_no_go(86_400.0) == "GO"
    assert hz.go_no_go(0.0) == "GO"
    assert hz.go_no_go(86_400.5) == "NO-GO"


def test_project_teachers_is_the_slowest_concurrent_seed() -> None:
    seconds = dict(zip(hz.SEEDS, (40.0, 50.0, 60.0, 45.0, 55.0)))
    timings = [{"seed": seed, "episodes": 20, "seconds": value} for seed, value in seconds.items()]
    projection = hz.project_teachers(timings, evaluation_episodes=9)
    expected = {str(seed): float(Fraction(value) / 20 * 1009) for seed, value in seconds.items()}
    assert projection["per_seed_projected_seconds"] == expected
    assert projection["projected_wall_clock_seconds"] == max(expected.values()) == 3027.0
    assert projection["verdict"] == "GO"
    slow = [{"seed": seed, "episodes": 20, "seconds": 1800.0} for seed in hz.SEEDS]
    assert hz.project_teachers(slow, evaluation_episodes=9)["verdict"] == "NO-GO"


def test_project_teachers_refuses_a_missing_or_repeated_seed() -> None:
    four = [{"seed": seed, "episodes": 20, "seconds": 40.0} for seed in hz.SEEDS[:4]]
    with pytest.raises(ValueError, match="505"):
        hz.project_teachers(four, evaluation_episodes=9)
    repeated = [{"seed": 101, "episodes": 20, "seconds": 40.0}] + four
    with pytest.raises(ValueError, match="twice"):
        hz.project_teachers(repeated, evaluation_episodes=9)
    zero = [{"seed": seed, "episodes": 0, "seconds": 40.0} for seed in hz.SEEDS]
    with pytest.raises(ValueError, match="episodes"):
        hz.project_teachers(zero, evaluation_episodes=9)


# ======================================================================
# 3. The fixed-time sweep
# ======================================================================

_OFFSETS = (-2, -1, 0, 1, 2)


def _sweep_records(engine: dict[int, float], ours: dict[int, float]) -> list[dict[str, Any]]:
    """Five draws per k whose mean is EXACTLY the given value (dyadic offsets around it)."""
    records = []
    for k in hz.K_GRID:
        for offset, draw_id in zip(_OFFSETS, hz.K_SWEEP_DRAWS):
            records.append(
                {
                    "k": k,
                    "draw_id": draw_id,
                    "att_engine": engine[k] + 0.25 * offset,
                    "att_ours": ours[k] + 0.5 * offset,
                    "created": 2700,
                    "entered": 2650,
                    "never_entered": 50,
                }
            )
    return records


_ENGINE = {1: 300.0, 2: 270.0, 3: 241.0, 4: 250.0, 6: 262.0, 8: 280.0, 10: 290.0, 14: 310.0}
_OURS = {1: 280.0, 2: 250.0, 3: 222.0, 4: 230.0, 6: 240.0, 8: 255.0, 10: 270.0, 14: 300.0}


def test_sweep_objective_is_recomputed_independently_from_the_raw_records() -> None:
    records = _sweep_records(_ENGINE, _OURS)
    summary = hz.summarise_sweep(records)
    assert sorted(summary) == list(hz.K_GRID)
    for k in hz.K_GRID:
        engine = [r["att_engine"] for r in records if r["k"] == k]
        ours = [r["att_ours"] for r in records if r["k"] == k]
        assert summary[k]["mean_att_engine"] == math.fsum(engine) / len(engine) == _ENGINE[k]
        assert summary[k]["mean_att_ours"] == math.fsum(ours) / len(ours) == _OURS[k]
        assert summary[k]["draw_ids"] == list(hz.K_SWEEP_DRAWS)
    decision = hz.choose_k(summary)
    by_engine = sorted(hz.K_GRID, key=lambda k: math.fsum(r["att_engine"] for r in records if r["k"] == k))
    by_ours = sorted(hz.K_GRID, key=lambda k: math.fsum(r["att_ours"] for r in records if r["k"] == k))
    assert decision["k_att_engine"] == by_engine[0] == 3
    assert decision["k_att_ours"] == by_ours[0] == 3
    assert decision["status"] == "CHOSEN"
    assert decision["chosen_k"] == 3
    assert decision["at_platform_floor"] is False


def test_choose_k_refuses_when_the_definitions_disagree() -> None:
    ours = {**_OURS, 4: 221.0}
    decision = hz.choose_k(hz.summarise_sweep(_sweep_records(_ENGINE, ours)))
    assert decision["status"] == "DEFINITION-DEPENDENT"
    assert decision["chosen_k"] is None
    assert (decision["k_att_engine"], decision["k_att_ours"]) == (3, 4)


def test_choose_k_refuses_an_exact_tie() -> None:
    engine = {**_ENGINE, 4: 241.0}
    decision = hz.choose_k(hz.summarise_sweep(_sweep_records(engine, _OURS)))
    assert decision["status"] == "TIE"
    assert decision["chosen_k"] is None


def test_choose_k_refuses_the_upper_boundary_and_flags_the_floor() -> None:
    engine = {**_ENGINE, 14: 200.0}
    ours = {**_OURS, 14: 200.0}
    upper = hz.choose_k(hz.summarise_sweep(_sweep_records(engine, ours)))
    assert upper["status"] == "BOUNDARY_UPPER"
    assert upper["chosen_k"] is None
    engine = {**_ENGINE, 1: 200.0}
    ours = {**_OURS, 1: 200.0}
    floor = hz.choose_k(hz.summarise_sweep(_sweep_records(engine, ours)))
    assert floor["status"] == "CHOSEN"
    assert floor["chosen_k"] == 1
    assert floor["at_platform_floor"] is True


def test_summarise_sweep_refuses_malformed_records() -> None:
    records = _sweep_records(_ENGINE, _OURS)
    with pytest.raises(ValueError, match="twice"):
        hz.summarise_sweep(records + [dict(records[0])])
    with pytest.raises(ValueError, match="draws"):
        hz.summarise_sweep([r for r in records if not (r["k"] == 6 and r["draw_id"] == 5)])
    with pytest.raises(ValueError, match="1000"):
        hz.summarise_sweep(records + [dict(records[0], k=1, draw_id=1000)])
    with pytest.raises(ValueError, match="k = 5"):
        hz.summarise_sweep(records + [dict(records[0], k=5)])
    bad = [dict(r) for r in records]
    bad[3]["att_engine"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        hz.summarise_sweep(bad)


def test_assert_training_draws_refuses_the_held_out_pool_and_draw_zero() -> None:
    hz.assert_training_draws(list(hz.K_SWEEP_DRAWS))
    with pytest.raises(ValueError, match="1000"):
        hz.assert_training_draws([1, 1000])
    with pytest.raises(ValueError, match=r"\[0\]"):
        hz.assert_training_draws([0])


def test_chosen_k_from_artifact_reads_only_a_chosen_status(tmp_path: Path) -> None:
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps({"format_version": hz.ARTIFACT_FORMAT_SWEEP,
                                "decision": {"status": "CHOSEN", "chosen_k": 3}}), encoding="utf-8")
    assert hz.chosen_k_from_artifact(path) == 3
    path.write_text(json.dumps({"format_version": hz.ARTIFACT_FORMAT_SWEEP,
                                "decision": {"status": "DEFINITION-DEPENDENT", "chosen_k": None}}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="DEFINITION-DEPENDENT"):
        hz.chosen_k_from_artifact(path)
    path.write_text(json.dumps({"format_version": "other/1.0",
                                "decision": {"status": "CHOSEN", "chosen_k": 3}}), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        hz.chosen_k_from_artifact(path)


@needs_cityflow
def test_sweep_episode_matches_an_independent_plain_step_loop(tmp_path: Path, tmp_scenario: Path) -> None:
    from experiments.config import EnvSpec
    from experiments.envs import make_env
    from offline.policies.fixed_time import FixedTimeController

    records = hz.materialise_hz4x4_draws([1], source_config=tmp_scenario, out_root=tmp_path / "draws")
    config = records[0].config_path
    settings = _env_settings()

    env = make_env(EnvSpec(id=hz.SCENARIO_ID, backend="cityflow", paths={"config": str(config)},
                           settings=dict(settings)))
    try:
        controller = FixedTimeController(env, k=1, plan=None)
        info = env.reset(seed=hz.BASE_SEED)
        decisions = 0
        for _ in range(env.max_steps):
            _reward, terminated, truncated, info = env.step(controller.act(info))
            decisions += 1
            if terminated or truncated:
                break
        att_ours = float(info["average_travel_time"])
        att_engine = float(env._eng.get_average_travel_time())
    finally:
        env.close()

    record = hz.sweep_episode(1, 1, config_path=config, env_settings=settings)
    assert decisions == 360
    assert record["att_ours"] == att_ours
    assert record["att_engine"] == att_engine
    assert record["schedule_source"] == "equal_split"
    assert record["created"] == record["entered"] + record["never_entered"]
    assert (record["k"], record["draw_id"]) == (1, 1)


# ======================================================================
# 4. The collector
# ======================================================================


def _grid_meta(tier_dir: str) -> dict[str, Any]:
    return json.loads((GRID_CORPUS / tier_dir / "manifest.json").read_bytes())["run_metadata"]


@needs_corpus
def test_grid_protocol_reads_every_directory_of_a_tier() -> None:
    mappo = hz.grid_protocol(GRID_CORPUS, "mappo1000")
    assert [m["checkpoint"].rsplit("seed", 1)[1] for m in mappo] == [f"{s}.pt" for s in hz.SEEDS]
    assert len(hz.grid_protocol(GRID_CORPUS, "maxpressure")) == 1
    with pytest.raises(ValueError, match="random_fresh"):
        hz.grid_protocol(GRID_CORPUS, "random_fresh")


def test_grid_protocol_refuses_a_manifest_that_breaks_an_invariant(tmp_path: Path) -> None:
    directory = tmp_path / "cf_grid4x4__maxpressure"
    directory.mkdir()
    metadata = dict(PROTOCOL_SETTINGS, backend="cityflow", base_seed=1000, behavior_policy="maxpressure",
                    episodes=1, flow_draw_ids=list(range(1, 201)), global_reward_weight=1.0)
    (directory / "manifest.json").write_text(json.dumps({"run_metadata": metadata, "episodes": []}),
                                             encoding="utf-8")
    with pytest.raises(ValueError, match="global_reward_weight"):
        hz.grid_protocol(tmp_path, "maxpressure")


def test_grid_protocol_refuses_a_held_out_draw(tmp_path: Path) -> None:
    directory = tmp_path / "cf_grid4x4__maxpressure"
    directory.mkdir()
    metadata = dict(PROTOCOL_SETTINGS, backend="cityflow", base_seed=1000, behavior_policy="maxpressure",
                    episodes=1, flow_draw_ids=[1, 2, 1000])
    (directory / "manifest.json").write_text(json.dumps({"run_metadata": metadata, "episodes": []}),
                                             encoding="utf-8")
    with pytest.raises(ValueError, match="1000"):
        hz.grid_protocol(tmp_path, "maxpressure")


@needs_corpus
def test_derived_runs_carry_the_grid_settings_with_only_the_declared_substitutions(tmp_path: Path) -> None:
    from offline.collect import build_parser

    env_config = hz.MAIN_TREE / "configs" / "sim" / "cityflow4x4.json"
    runs = hz.derive_runs(corpus_root=GRID_CORPUS, tiers=["maxpressure", "fixedtime", "mappo1000"], chosen_k=3,
                          env_config=env_config, out_root=tmp_path, random_rng_supported=False)
    expected_names = ["cf_hz4x4__maxpressure", "cf_hz4x4__fixedtime"] + [
        f"cf_hz4x4__mappo1000__seed{s}" for s in hz.SEEDS
    ]
    assert [run.name for run in runs] == expected_names
    sources = ["cf_grid4x4__maxpressure", "cf_grid4x4__fixedtime"] + [
        f"cf_grid4x4__mappo1000__seed{s}" for s in hz.SEEDS
    ]
    parser = build_parser()
    for run, source in zip(runs, sources):
        meta = _grid_meta(source)
        args = parser.parse_args(list(run.argv))
        assert args.backend == meta["backend"]
        assert args.policy == meta["behavior_policy"]
        assert args.episodes == meta["episodes"]
        assert args.base_seed == meta["base_seed"] == hz.BASE_SEED
        for key in ("max_steps", "delta_time", "control_mode", "global_reward_fn", "local_reward_fn",
                    "global_reward_weight", "state_features"):
            assert getattr(args, key) == meta[key], key
        ids = meta["flow_draw_ids"]
        assert tuple(args.flow_draws_range) == (ids[0], ids[-1] + 1)
        assert list(range(*args.flow_draws_range)) == ids
        params = meta["flow_randomizer_params"]
        assert (args.flow_jitter_sigma, args.flow_thin_p, args.flow_volume_scale) == (
            params["jitter_sigma_s"], params["thin_p"], params["volume_scale"])
        assert args.env_config == str(env_config)
        assert args.out_dir == str(tmp_path / run.name)
        assert run.out_dir == tmp_path / run.name
        assert run.expected_episodes == len(ids) * meta["episodes"]
        assert run.draw_ids == tuple(ids)
        if run.tier == "mappo1000":
            assert args.checkpoint == str(hz.TEACHER_CHECKPOINT_SUBDIR / f"cf_hz4x4__mappo__seed{run.seed}.pt")
        else:
            assert args.checkpoint is None
        if run.tier == "fixedtime":
            assert meta["fixed_time_k"] == 1
            assert args.fixed_time_k == 3
    assert [run.expected_episodes for run in runs] == [200, 200, 40, 40, 40, 40, 40]


@needs_corpus
def test_derive_runs_refuses_fixedtime_without_a_chosen_k(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="k-sweep"):
        hz.derive_runs(corpus_root=GRID_CORPUS, tiers=["fixedtime"], chosen_k=None,
                       env_config=hz.MAIN_TREE / "configs/sim/cityflow4x4.json", out_root=tmp_path,
                       random_rng_supported=False)


@needs_corpus
def test_random_fresh_is_refused_until_the_collector_has_random_rng(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--random-rng"):
        hz.derive_runs(corpus_root=GRID_CORPUS, tiers=["random_fresh"], chosen_k=None,
                       env_config=hz.MAIN_TREE / "configs/sim/cityflow4x4.json", out_root=tmp_path,
                       random_rng_supported=False)


@needs_corpus
def test_random_fresh_run_is_one_episode_per_draw_with_the_per_episode_generator(tmp_path: Path) -> None:
    (run,) = hz.derive_runs(corpus_root=GRID_CORPUS, tiers=["random_fresh"], chosen_k=None,
                            env_config=hz.MAIN_TREE / "configs/sim/cityflow4x4.json", out_root=tmp_path,
                            random_rng_supported=True)
    argv = list(run.argv)
    assert argv[argv.index("--random-rng") + 1] == "per-episode"
    assert argv[argv.index("--policy") + 1] == "random"
    assert argv[argv.index("--episodes") + 1] == "1"
    assert argv[argv.index("--base-seed") + 1] == "1000"
    position = argv.index("--flow-draws-range")
    assert argv[position + 1: position + 3] == ["1", "201"]
    assert run.expected_episodes == 200
    assert run.name == "cf_hz4x4__random_fresh"


def test_assert_targets_absent_refuses_a_populated_directory(tmp_path: Path) -> None:
    def run_at(path: Path) -> Any:
        return hz.CollectionRun(name=path.name, tier="maxpressure", seed=None, out_dir=path, argv=(),
                                expected_episodes=1, draw_ids=(1,))

    empty = tmp_path / "empty"
    empty.mkdir()
    hz.assert_targets_absent([run_at(tmp_path / "absent"), run_at(empty)])
    full = tmp_path / "full"
    full.mkdir()
    (full / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="full"):
        hz.assert_targets_absent([run_at(tmp_path / "absent"), run_at(full)])


def test_check_random_rng_scheme_requires_per_episode() -> None:
    hz.check_random_rng_scheme({"random_rng_scheme": "per-episode", "behavior_policy": "random"})
    with pytest.raises(ValueError, match="per-draw"):
        hz.check_random_rng_scheme({"random_rng_scheme": "per-draw", "behavior_policy": "random"})
    with pytest.raises(ValueError, match="random_rng_scheme"):
        hz.check_random_rng_scheme({"behavior_policy": "random"})


def test_check_flow_digests_compares_every_draw_with_the_materialised_flow(tmp_path: Path) -> None:
    for draw_id, body in ((1, b"[1]\n"), (2, b"[2]\n")):
        directory = tmp_path / hz.SCENARIO_KEY / f"draw_{draw_id:04d}"
        directory.mkdir(parents=True)
        (directory / "flow.json").write_bytes(body)
    digests = {"1": hashlib.sha256(b"[1]\n").hexdigest(), "2": hashlib.sha256(b"[2]\n").hexdigest()}
    manifest = {"run_metadata": {"flow_draw_ids": [1, 2], "flow_draw_sha256": digests}}
    assert hz.check_flow_digests(manifest, draws_root=tmp_path) == 2
    wrong = {"run_metadata": {"flow_draw_ids": [1, 2], "flow_draw_sha256": dict(digests, **{"2": digests["1"]})}}
    with pytest.raises(ValueError, match="draw 2"):
        hz.check_flow_digests(wrong, draws_root=tmp_path)
    missing = {"run_metadata": {"flow_draw_ids": [1, 3], "flow_draw_sha256": dict(digests, **{"3": "0" * 64})}}
    with pytest.raises(FileNotFoundError, match="draw_0003"):
        hz.check_flow_digests(missing, draws_root=tmp_path)


def test_collector_supports_random_rng_reads_the_parser() -> None:
    import argparse

    with_flag = argparse.ArgumentParser()
    with_flag.add_argument("--random-rng", choices=["per-draw", "per-episode"], default="per-draw")
    assert hz.collector_supports_random_rng(with_flag) is True
    assert hz.collector_supports_random_rng(argparse.ArgumentParser()) is False


@needs_corpus
def test_distinct_action_sequences_reproduce_the_grid_random_tiers_two_schedules() -> None:
    counts = hz.distinct_action_sequences(GRID_CORPUS / "cf_grid4x4__random")
    assert len(counts) == 16
    assert set(counts.values()) == {2}


@needs_corpus
@needs_cityflow
def test_protocol_argv_reproduces_a_committed_grid_episode(tmp_path: Path) -> None:
    from offline.collect import main as collect_main
    from offline.trajectory_logger import load_episode

    source = GRID_CORPUS / "cf_grid4x4__maxpressure"
    meta = _grid_meta("cf_grid4x4__maxpressure")
    argv = hz.protocol_argv(meta, env_config=meta["env_paths"]["config"], out_dir=tmp_path / "rerun",
                            policy=meta["behavior_policy"], episodes=1, draw_ids=[1])
    assert collect_main(list(argv)) == 0
    name = "ep000000_seed1000_draw1.npz"
    with np.load(tmp_path / "rerun" / name) as fresh, np.load(source / name) as committed:
        assert sorted(fresh.files) == sorted(committed.files)
        for key in committed.files:
            assert fresh[key].dtype == committed[key].dtype, key
            assert np.array_equal(fresh[key], committed[key]), key
    assert load_episode(tmp_path / "rerun" / name).episode_length == 360
    rerun_manifest = json.loads((tmp_path / "rerun" / "manifest.json").read_bytes())
    committed_manifest = json.loads((source / "manifest.json").read_bytes())
    assert rerun_manifest["episodes"][0]["episode_sha256"] == committed_manifest["episodes"][0]["episode_sha256"]


# ======================================================================
# 5. The teachers (RUNSPEC_01 section 4)
# ======================================================================


def _curve(final: float, preceding: float, earlier: float, length: int = 1000) -> list[float]:
    head = [earlier] * (length - 150)
    return head + [earlier] * 50 + [preceding] * 50 + [final] * 50


def test_convergence_is_the_final_window_and_the_preceding_one() -> None:
    converged = hz.convergence(_curve(-104.0, -100.0, -100.0))
    assert converged["rel_final"] == 0.04
    assert converged["rel_preceding"] == 0.0
    assert converged["converged"] is True
    at_threshold = hz.convergence(_curve(-105.0, -100.0, -100.0))
    assert at_threshold["rel_final"] == 0.05
    assert at_threshold["converged"] is False
    preceding_fails = hz.convergence(_curve(-100.0, -100.0, -94.0))
    assert preceding_fails["rel_final"] == 0.0
    assert preceding_fails["converged"] is False
    small = hz.convergence(_curve(0.75, 0.5, 0.5))
    assert small["rel_final"] == 0.25
    with pytest.raises(ValueError, match="150"):
        hz.convergence([1.0] * 149)


@needs_p2_1
def test_convergence_reproduces_p2_1s_recorded_counts_at_b_1000() -> None:
    results = json.loads(P2_1_RESULTS.read_bytes())
    counts: dict[str, int] = {}
    for cell in results["cells"]:
        verdict = hz.convergence(cell["policies"]["mappo"]["train_returns"])
        counts[cell["env_id"]] = counts.get(cell["env_id"], 0) + int(verdict["converged"])
    assert counts == {"cf_hz1x1": 4, "cf_grid4x4": 3, "cf_cologne3": 0}


def _results(tmp_path: Path, seeds: tuple[int, ...] = hz.SEEDS, length: int = 1000) -> Path:
    cells = []
    for seed in seeds:
        checkpoint = tmp_path / f"cf_hz4x4__mappo__seed{seed}.pt"
        checkpoint.write_bytes(f"checkpoint {seed}".encode())
        cells.append({
            "env_id": "cf_hz4x4", "seed": seed, "status": "ok",
            "policies": {"mappo": {"train_returns": [float(-seed)] * length, "checkpoint_path": str(checkpoint),
                                   "checkpoint_loaded": False, "metrics": {}}},
            "timings": {"mappo": {"train_sec": 1000.0 + seed, "eval_sec": 3.0}},
        })
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"name": "p9_3_mappo_nominal_1000", "cells": cells}), encoding="utf-8")
    return path


def test_teachers_report_carries_every_seed_with_its_curve_clock_and_checkpoint(tmp_path: Path) -> None:
    report = hz.teachers_report(_results(tmp_path))
    assert report["label"] == "MAPPO@1000"
    assert sorted(report["seeds"]) == [str(s) for s in hz.SEEDS]
    for seed in hz.SEEDS:
        entry = report["seeds"][str(seed)]
        assert entry["train_returns"] == [float(-seed)] * 1000
        assert entry["train_sec"] == 1000.0 + seed
        assert entry["checkpoint_sha256"] == hashlib.sha256(f"checkpoint {seed}".encode()).hexdigest()
        assert entry["convergence"]["converged"] is True
    assert report["converged_seeds"] == 5
    assert "uncontrolled" in report["clock"]


def test_teachers_report_refuses_a_missing_seed_or_a_short_curve(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="505"):
        hz.teachers_report(_results(tmp_path, seeds=hz.SEEDS[:4]))
    with pytest.raises(ValueError, match="1000"):
        hz.teachers_report(_results(tmp_path, length=999))


# ======================================================================
# 6. A11's gate on hz4x4
# ======================================================================


def _gate_episode(tier: str, draw_id: int, **overrides: Any) -> Any:
    from offline.engine_att_reference import GateEpisode

    fields = dict(
        scenario="hz4x4", tier=tier, method="behaviour", arm=f"behaviour@{tier}", seed=None, draw_id=draw_id,
        role="tier", engine_seed=1000, att_reference_engine_population=300.0,
        att_reference_entered_running=299.0, att_reference_entered_population=300.0,
        att_reference_metric_cadence=250.0, att_engine_call=300.0, att_ours=250.0, n_reference_ids=2700,
        n_entered_ids=2700, created_from_flow=2700, entered=2700, never_entered=0,
        admission_latency_mean=1.0, admission_latency_max=5.0, n_admission_delayed=10, interval=1.0,
        n_observations=3600, seconds=1.0, seconds_rollout=1.0,
    )
    fields.update(overrides)
    return GateEpisode(**fields)


def _passing_set(n_tiers: int = 7, draws: tuple[int, ...] = (1000, 1001, 1002)) -> list[Any]:
    episodes = []
    for index in range(n_tiers):
        for draw_id in draws:
            episodes.append(_gate_episode(f"tier{index}", draw_id))
    censored = _gate_episode("tier0", 1003, entered=2650, never_entered=50, n_entered_ids=2650,
                             att_reference_entered_population=299.5, att_reference_entered_running=298.0)
    return episodes + [censored]


def test_gate_passes_only_when_all_four_criteria_hold() -> None:
    from offline.engine_att_reference import evaluate_scenario

    criteria = evaluate_scenario(_passing_set())
    assert (criteria.c1_passed, criteria.c2_exact, criteria.c3a_passed, criteria.c3b_passed,
            criteria.c4_passed) == (True, True, True, True, True)
    assert criteria.passed is True
    outcome = hz.gate_outcome(criteria)
    assert outcome["passed"] is True
    assert outcome["primary"] == "att_engine"
    assert outcome["consequence"] == hz.A11_CONSEQUENCE_PASS
    assert outcome["failed_criteria"] == []


def test_gate_c1_is_strictly_below_1e_4() -> None:
    from offline.engine_att_reference import evaluate_scenario

    zero = dict(att_reference_engine_population=0.0, att_reference_entered_population=0.0)
    inside = _passing_set() + [_gate_episode("tier1", 1004, att_engine_call=9.999e-5, **zero)]
    at = _passing_set() + [_gate_episode("tier1", 1004, att_engine_call=1e-4, **zero)]
    assert evaluate_scenario(inside).c1_passed is True
    assert evaluate_scenario(at).c1_passed is False
    outcome = hz.gate_outcome(evaluate_scenario(at))
    assert outcome["primary"] is None
    assert outcome["consequence"] == hz.A11_CONSEQUENCE_FAIL
    assert outcome["failed_criteria"] == ["c1"]


def test_gate_c2_is_exact() -> None:
    from offline.engine_att_reference import evaluate_scenario

    off_by_one = _passing_set() + [_gate_episode("tier2", 1004, n_reference_ids=2701)]
    criteria = evaluate_scenario(off_by_one)
    assert criteria.c2_exact is False
    assert hz.gate_outcome(criteria)["failed_criteria"] == ["c2"]


def test_gate_c3a_is_exactly_zero_and_c3b_strictly_positive() -> None:
    from offline.engine_att_reference import evaluate_scenario

    nudged = _passing_set() + [_gate_episode("tier3", 1004, att_reference_entered_population=300.00000000000006)]
    assert evaluate_scenario(nudged).c3a_passed is False
    flat = [e for e in _passing_set() if e.never_entered == 0] + [
        _gate_episode("tier0", 1003, entered=2650, never_entered=50, att_reference_entered_population=300.0)
    ]
    assert evaluate_scenario(flat).c3b_passed is False
    no_censored = [e for e in _passing_set() if e.never_entered == 0]
    assert evaluate_scenario(no_censored).c3b_passed is False


def test_gate_c4_needs_seven_tiers_and_three_draws_so_hz4x4s_four_tiers_fail() -> None:
    from offline.engine_att_reference import evaluate_scenario

    assert evaluate_scenario(_passing_set(n_tiers=6)).c4_passed is False
    assert evaluate_scenario(_passing_set(draws=(1000, 1001))).c4_passed is False
    hz4x4_shaped = _passing_set(n_tiers=4)
    criteria = evaluate_scenario(hz4x4_shaped)
    assert criteria.c4_n_tiers == 4
    assert criteria.c4_passed is False
    assert hz.gate_outcome(criteria)["failed_criteria"] == ["c4"]


def test_gate_tolerance_mutants_flip_the_boundary_verdicts(monkeypatch: pytest.MonkeyPatch) -> None:
    import offline.engine_att_reference as engine

    zero = dict(att_reference_engine_population=0.0, att_reference_entered_population=0.0)
    at = _passing_set() + [_gate_episode("tier1", 1004, att_engine_call=1e-4, **zero)]
    monkeypatch.setattr(engine, "C1_TOLERANCE", 2e-4)
    assert engine.evaluate_scenario(at).c1_passed is True
    monkeypatch.setattr(engine, "C1_TOLERANCE", 1e-4)
    monkeypatch.setattr(engine, "C4_MIN_TIERS", 4)
    assert engine.evaluate_scenario(_passing_set(n_tiers=4)).c4_passed is True
    monkeypatch.setattr(engine, "C4_MIN_TIERS", 7)
    monkeypatch.setattr(engine, "C4_MIN_DRAWS", 2)
    assert engine.evaluate_scenario(_passing_set(draws=(1000, 1001))).c4_passed is True


def test_gate_cells_for_hz4x4_are_four_tiers_by_three_draws() -> None:
    from offline.engine_att_reference import GATE_DRAWS, GATE_SEED

    cells = hz.gate_cells_for(["mappo1000", "maxpressure", "fixedtime", "random_fresh"])
    assert len(cells) == 12
    assert {c.scenario for c in cells} == {"hz4x4"}
    assert {c.role for c in cells} == {"tier"}
    assert {c.method for c in cells} == {"behaviour"}
    for tier, seed in (("mappo1000", GATE_SEED), ("maxpressure", None), ("fixedtime", None),
                       ("random_fresh", GATE_SEED)):
        assert [(c.seed, c.draw_id) for c in cells if c.tier == tier] == [(seed, d) for d in GATE_DRAWS]
    with pytest.raises(ValueError, match="mappo200"):
        hz.gate_cells_for(["mappo200"])
    with pytest.raises(ValueError, match="twice"):
        hz.gate_cells_for(["maxpressure", "maxpressure"])


def test_hz4x4_gate_episode_refuses_another_networks_env_id(tmp_path: Path) -> None:
    (cell,) = [c for c in hz.gate_cells_for(["maxpressure"]) if c.draw_id == 1000]
    with pytest.raises(ValueError, match="cityflow4x4"):
        hz.hz4x4_gate_episode(cell=cell, config_path=tmp_path / "absent.json", env_settings=_env_settings(),
                              scenario_id="cityflow_grid4x4", choose_action_factory=lambda env: None,
                              engine_seed=1000, created=1)


def test_gate_extremes_are_the_min_and_max_entered_fraction() -> None:
    episodes = [
        _gate_episode("maxpressure", 1000, entered=2700, created_from_flow=2700),
        _gate_episode("fixedtime", 1001, entered=2600, never_entered=100, created_from_flow=2700),
        _gate_episode("random_fresh", 1002, entered=2000, never_entered=700, created_from_flow=2700,
                      seed=101),
    ]
    extremes = hz.gate_extremes(episodes)
    assert extremes["min"]["entered_fraction"] == 2000 / 2700
    assert (extremes["min"]["tier"], extremes["min"]["draw_id"], extremes["min"]["seed"]) == ("random_fresh", 1002, 101)
    assert extremes["max"]["entered_fraction"] == 1.0
    assert extremes["max"]["tier"] == "maxpressure"


@needs_cityflow
def test_hz4x4_gate_episode_reconstructs_the_engine_value_on_a_training_draw(
    tmp_path: Path, tmp_scenario: Path
) -> None:
    from offline.admission_probe import created_from_flow
    from offline.dt_gate import _maxpressure_factory
    from offline.engine_att_reference import C1_TOLERANCE, GateCell

    records = hz.materialise_hz4x4_draws([1], source_config=tmp_scenario, out_root=tmp_path / "draws")
    config = records[0].config_path
    cell = GateCell(scenario="hz4x4", tier="maxpressure", method="behaviour", seed=None, draw_id=1, role="tier")
    created = created_from_flow(config.parent / "flow.json", horizon_seconds=3600)
    episode = hz.hz4x4_gate_episode(cell=cell, config_path=config, env_settings=_env_settings(),
                                    scenario_id=hz.SCENARIO_ID, choose_action_factory=_maxpressure_factory,
                                    engine_seed=1000, created=created)
    assert episode.deviation_c1 < C1_TOLERANCE
    assert episode.n_reference_ids == episode.created_from_flow == episode.entered + episode.never_entered
    assert episode.scenario == "hz4x4"


# ======================================================================
# 7. The declaration
# ======================================================================

_GRID_ROWS = {
    "episodes": 200, "n_intersections": 16, "episode_length": 360, "streams": 3200, "decisions": 1_152_000,
    "top_decile_decisions_global": 115_200, "top_decile_decisions_perix": 115_200,
}


def test_exposure_table_reproduces_a27b_from_the_code_constants() -> None:
    table = hz.exposure_table(_GRID_ROWS)
    printed = {arm: round(row["visits"], 1) for arm, row in table["arms"].items()}
    assert printed == {
        "dt_nomix": 711.1, "bc": 44.4, "bc_top10": 444.4, "bc_top10_perix": 444.4, "iql": 44.4,
        "bc_m": 711.1, "bc_top10_m": 708.3, "bc_top10_perix_m": 708.3, "iql_m": 711.1,
    }
    rows = {"dt_nomix": 64 * 20 * 16, "bc": 64 * 20, "iql": 1280, "bc_m": 1024 * 20, "iql_m": 20_480,
            "bc_top10": 64 * 20, "bc_top10_perix": 64 * 20, "bc_top10_m": 102 * 20, "bc_top10_perix_m": 102 * 20}
    basis = {arm: (115_200 if "top10" in arm else 1_152_000) for arm in rows}
    for arm, row in table["arms"].items():
        assert row["rows_per_step"] == rows[arm], arm
        assert row["training_decisions"] == basis[arm], arm
        assert row["visits"] == float(Fraction(40_000 * rows[arm], basis[arm])), arm
    assert table["steps"] == 40_000


def test_exposure_table_follows_the_measured_row_counts() -> None:
    rows = dict(_GRID_ROWS, episodes=199, streams=3184, decisions=1_146_240,
                top_decile_decisions_global=114_840, top_decile_decisions_perix=115_200)
    table = hz.exposure_table(rows)
    assert table["arms"]["dt_nomix"]["visits"] == float(Fraction(40_000 * 20_480, 1_146_240))
    assert table["arms"]["bc_top10_m"]["visits"] == float(Fraction(40_000 * 2_040, 114_840))
    assert table["arms"]["bc_top10_perix_m"]["visits"] == float(Fraction(40_000 * 2_040, 115_200))
    with pytest.raises(ValueError, match="decisions"):
        hz.exposure_table(dict(_GRID_ROWS, decisions=0))
    with pytest.raises(ValueError, match="top"):
        hz.exposure_table(dict(_GRID_ROWS, top_decile_decisions_global=2_000_000))


@needs_corpus
def test_tier_row_counts_on_the_grid_mappo1000_tier() -> None:
    from offline.tier_sweep import TIERS

    counts = hz.tier_row_counts([GRID_CORPUS / d for d in TIERS["mappo1000"].dirs])
    assert counts == _GRID_ROWS


@needs_corpus
def test_ladder_att_and_mean_training_return_reproduce_tier_sweep_tiers() -> None:
    from offline.tier_sweep import TIERS

    for tier in ("mappo1000", "maxpressure", "fixedtime", "random"):
        dirs = [GRID_CORPUS / d for d in TIERS[tier].dirs]
        assert hz.ladder_att(dirs) == TIERS[tier].ladder_att, tier
        assert round(hz.mean_training_return(dirs), 1) == TIERS[tier].mean_training_return, tier


# ======================================================================
# CLI surface
# ======================================================================


def test_cli_exposes_every_phase_a_subcommand() -> None:
    parser = hz.build_parser()
    for argv in (["draws", "--dry-run"], ["preflight", "--out", "x.json"], ["k-sweep", "--out", "x.json"],
                 ["collect", "plan", "--sweep", "x.json"], ["teachers", "--out", "x.json"],
                 ["gate", "--out", "x.json"], ["declare", "--out", "x.json"]):
        namespace = parser.parse_args(argv)
        assert namespace.command == argv[0]
        assert namespace.torch_threads == 1


def test_protocol_argv_uses_the_range_flag_for_contiguous_draws_only() -> None:
    meta = dict(PROTOCOL_SETTINGS, backend="cityflow", base_seed=1000,
                flow_randomizer_params={"jitter_sigma_s": 30.0, "thin_p": 0.1, "volume_scale": 1.0})
    contiguous = list(hz.protocol_argv(meta, env_config="c.json", out_dir="o", policy="maxpressure", episodes=1,
                                       draw_ids=[1, 2, 3]))
    assert contiguous[contiguous.index("--flow-draws-range") + 1: contiguous.index("--flow-draws-range") + 3] == ["1", "4"]
    sparse = list(hz.protocol_argv(meta, env_config="c.json", out_dir="o", policy="maxpressure", episodes=1,
                                   draw_ids=[1, 3]))
    assert sparse[sparse.index("--flow-draws") + 1: sparse.index("--flow-draws") + 3] == ["1", "3"]
    assert "--flow-draws-range" not in sparse
    with pytest.raises(ValueError, match="1000"):
        hz.protocol_argv(meta, env_config="c.json", out_dir="o", policy="maxpressure", episodes=1,
                         draw_ids=[999, 1000])
