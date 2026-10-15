"""Tests for ``offline/sumo_anchor_rescore.py``, part 1 -- the aligned anchors' rollout (``BRIEF_47`` §3.2, §3.3 ``roll-anchors``).

Layers:

* **pure** -- always runs: the declaration, the work-dir barrier, the cell validator against planted defects, the resumable
  roll with a stand-in episode, the replica comparison, the kcf constants against the tuning README.
* **gated** -- builds real envs: the fixed-time controllers' 360-step action sequences on SUMO and CityFlow at kcf; the
  SUMO rollout replica reproducing a REGISTERED anchor record of P7.3a (hz1x1) and P7.3d (grid4x4) under ``==``; one new
  SUMO cell validating with its provenance; the CityFlow ``maxpressure_1to1`` hz1x1 cell equal to the committed P8.4a
  MaxPressure cell under both definitions.

GATES: ``RLTRAFFIC_DRAWS`` (the main tree's ``scenarios/draws`` with the parity subdirectories), ``RLTRAFFIC_CORPUS_V11``
(the collection manifests the env settings and kcf are read from), CityFlow, ``sumo`` + ``traci``.  Writes only ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import offline.sumo_anchor_rescore as sar

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"
COMMIT = "a" * 40


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------


def _sumo_payload(scenario: str = "cityflow1x1", arm: str = "maxpressure_1to1", draw_id: int = 1001) -> dict[str, Any]:
    """A payload carrying every field the validator reads, consistent by construction."""
    grid = scenario == "cityflow_grid4x4"
    ids = [f"I{i}" for i in range(16)] if grid else ["intersection_1_1"]
    actions = [[0] * len(ids) for _ in range(360)] if grid else [0] * 360
    payload: dict[str, Any] = {
        "format_version": sar.CELL_FORMAT_VERSION,
        "backend": "sumo",
        "scenario": scenario,
        "arm": arm,
        "base_arm": sar.BASE_ARM[arm],
        "draw_id": draw_id,
        "kind": "anchor",
        "engine_seed_requested": 1000,
        "engine_seed_drawn": 437485271,
        "decisions": 360,
        "actions": actions,
        "actions_in_range": True,
        "action_space_n": 8,
        "action_sequence_sha256": "0" * 64,
        "episode_reward": -100.0,
        "att_horizon": 300.0,
        "att_env": 300.0,
        "att_running_mean": 200.0,
        "horizon_vehicle_count": 10.0,
        "e_sumo": 310.0,
        "p_sumo": 305.0,
        "w_sumo": 300.0,
        "mean_depart_delay": 1.0,
        "n_created": 100,
        "n_entered": 100,
        "n_never_entered": 0,
        "n_pending_at_horizon": 0,
        "n_teleports": 0,
        "n_vanished_without_arrival": 0,
        "n_arrived_never_observed_at_a_boundary": 0,
        "max_abs_depart_clock_deviation": 0.0,
        "n_observations": 3600,
        "vehicle_types_seen": ["cf_parity"],
        "time_to_teleport_option": "-1",
        "halting_checked": draw_id == 1000,
        "halting_max_abs_difference": 0 if draw_id == 1000 else None,
        "halting_n_lane_seconds": 28800 if draw_id == 1000 else None,
        "halting_n_disagreeing_lane_seconds": 0 if draw_id == 1000 else None,
        "config_sha256": "c" * 64,
        "routes_sha256": "d" * 64,
        "depart_lane_attributes": 0,
        "policy": (
            {"arm": arm, "roadlink_structure_digest": "e" * 64, "multiplicities": {"summary": {}}}
            if arm == "maxpressure_1to1"
            else {"arm": arm, "fixed_time_k": sar.KCF_README_VALUES[scenario], "fixed_time_schedule_source": "equal_split",
                  "fixed_time_plan_sha256": None, "cycle": {ix: list(range(8)) for ix in ids}}
        ),
        "seconds": 12.0,
        "git_commit": COMMIT,
        "git_dirty": False,
    }
    if grid:
        payload.update(
            intersection_ids=ids,
            local_return={ix: -10.0 for ix in ids},
            local_return_from_lanes={ix: -10.0 for ix in ids},
            collisions=[],
            n_collisions=0,
            teleports=[],
            n_explained_teleports=0,
            n_unexplained_teleports=0,
            vanished_ids=[],
        )
    return payload


def _cityflow_payload(scenario: str = "cityflow1x1", draw_id: int = 1000) -> dict[str, Any]:
    return {
        "format_version": sar.CELL_FORMAT_VERSION,
        "backend": "cityflow",
        "scenario": scenario,
        "arm": "maxpressure_1to1",
        "draw_id": draw_id,
        "att_ours": 163.0,
        "att_engine": 160.0,
        "created": 1813,
        "entered": 1813,
        "never_entered": 0,
        "horizon_vehicle_count": 103.0,
        "episode_reward": -16428.0,
        "roadlink_structure_digest": "f" * 64,
        "multiplicities": {"summary": {}},
        "seconds": 1.0,
        "git_commit": COMMIT,
        "git_dirty": False,
    }


def _cell(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in ("backend", "scenario", "arm", "draw_id")}


# ----------------------------------------------------------------------
# The declaration and the constants
# ----------------------------------------------------------------------


def test_the_declaration_is_400_sumo_and_200_cityflow_cells_with_unique_names() -> None:
    cells = sar.declared_anchor_cells()
    assert len(cells) == 600
    sumo = [c for c in cells if c["backend"] == "sumo"]
    cityflow = [c for c in cells if c["backend"] == "cityflow"]
    assert len(sumo) == 400 and len(cityflow) == 200
    assert {(c["scenario"], c["arm"]) for c in sumo} == {
        (s, a) for s in ("cityflow1x1", "cityflow_grid4x4") for a in ("maxpressure_1to1", "fixedtime_kcf")
    }
    assert {(c["scenario"], c["arm"]) for c in cityflow} == {
        ("cityflow1x1", "maxpressure_1to1"), ("cityflow_grid4x4", "maxpressure_1to1")
    }
    assert sorted({c["draw_id"] for c in cells}) == list(range(1000, 1100))
    names = [sar.cell_name(c) for c in cells]
    assert len(set(names)) == 600
    assert sar.cell_name(cells[0]) == "cell_sumo_cityflow1x1_maxpressure_1to1_draw1000.json"


def test_kcf_constants_are_the_tuning_readmes_chosen_values() -> None:
    """A second route to the constants: the README's table rows, read as text."""
    text = (DATA / "fixed_time_sweep" / "README.md").read_text(encoding="utf-8")
    rows = {line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith("| cf_")}
    chosen = {name: int(re.search(r"\*\*k=(\d+)\*\* \(", row).group(1)) for name, row in rows.items()}
    assert chosen["cf_hz1x1"] == sar.KCF_README_VALUES["cityflow1x1"] == 6
    assert chosen["cf_grid4x4"] == sar.KCF_README_VALUES["cityflow_grid4x4"] == 1


def _fake_corpus(tmp_path: Path, k: int, *, scenario_dir: str = "cf_hz1x1__fixedtime") -> Path:
    root = tmp_path / "corpus"
    (root / scenario_dir).mkdir(parents=True)
    (root / scenario_dir / "manifest.json").write_text(
        json.dumps({"run_metadata": {"fixed_time_k": k, "fixed_time_schedule_source": "shipped_plan",
                                     "fixed_time_plan_sha256": "1" * 64}}),
        encoding="utf-8",
    )
    return root


def test_kcf_is_read_from_the_cityflow_tier_manifest(tmp_path: Path) -> None:
    record = sar.kcf_for("cityflow1x1", _fake_corpus(tmp_path, 6))
    assert record["fixed_time_k"] == 6
    assert record["manifest"].endswith("cf_hz1x1__fixedtime/manifest.json")


def test_kcf_refuses_a_manifest_that_disagrees_with_the_tuning_readme(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tuning README"):
        sar.kcf_for("cityflow1x1", _fake_corpus(tmp_path, 4))


def test_kcf_refuses_an_unknown_scenario(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a scenario of this task"):
        sar.kcf_for("cityflow_cologne3", _fake_corpus(tmp_path, 6))


# ----------------------------------------------------------------------
# The work-dir barrier
# ----------------------------------------------------------------------


def test_the_work_dir_must_sit_under_the_output_roots_p9_2_directories(tmp_path: Path) -> None:
    output = tmp_path / "output"
    assert sar.assert_work_dir(output / "p9_2" / "anchors", output_root=output, repo_root=REPO_ROOT) == (
        output / "p9_2" / "anchors"
    )
    assert sar.assert_work_dir(output / "p9_2_runs" / "x", output_root=output, repo_root=REPO_ROOT)
    for bad in (output / "p7_3a", output / "p9_2x", tmp_path / "elsewhere"):
        with pytest.raises(ValueError, match="p9_2"):
            sar.assert_work_dir(bad, output_root=output, repo_root=REPO_ROOT)
    assert not output.exists(), "a checked work dir is not created by the check"


def test_the_work_dir_may_not_sit_inside_a_repository_tree(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the repository"):
        sar.assert_work_dir(REPO_ROOT / "output" / "p9_2" / "a", output_root=REPO_ROOT / "output", repo_root=REPO_ROOT)


# ----------------------------------------------------------------------
# The validator, against planted defects
# ----------------------------------------------------------------------


def test_valid_payloads_of_every_kind_pass() -> None:
    for payload in (
        _sumo_payload(), _sumo_payload(arm="fixedtime_kcf"), _sumo_payload(draw_id=1000),
        _sumo_payload("cityflow_grid4x4"), _sumo_payload("cityflow_grid4x4", "fixedtime_kcf"), _cityflow_payload(),
        _cityflow_payload("cityflow_grid4x4"),
    ):
        sar.validate_anchor_cell(payload, cell=_cell(payload))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"format_version": "p7.3a-zero-shot/1.0"}, "format_version"),
        ({"n_teleports": 1}, "teleport"),
        ({"time_to_teleport_option": "300"}, "time-to-teleport"),
        ({"vehicle_types_seen": ["DEFAULT_VEHTYPE"]}, "vehicle type"),
        ({"decisions": 359}, "360 decisions"),
        ({"actions_in_range": False}, "legal range"),
        ({"halting_checked": True}, "halting"),
        ({"git_dirty": True}, "git_dirty"),
        ({"git_commit": "HEAD"}, "40-character"),
        ({"att_env": 301.0}, "two routes"),
        ({"engine_seed_requested": 1001}, "engine_seed_requested"),
    ],
)
def test_the_validator_refuses_each_planted_defect_on_a_sumo_cell(change: dict[str, Any], match: str) -> None:
    payload = _sumo_payload()
    payload.update(change)
    with pytest.raises(ValueError, match=match):
        sar.validate_anchor_cell(payload, cell=_cell(_sumo_payload()))


def test_the_validator_refuses_a_chunk_that_describes_another_cell() -> None:
    payload = _sumo_payload(draw_id=1002)
    with pytest.raises(ValueError, match="describes another cell"):
        sar.validate_anchor_cell(payload, cell=_cell(_sumo_payload(draw_id=1001)))


def test_the_validator_refuses_a_fixedtime_kcf_cell_at_another_k() -> None:
    payload = _sumo_payload(arm="fixedtime_kcf")
    payload["policy"]["fixed_time_k"] = 4
    with pytest.raises(ValueError, match="kcf"):
        sar.validate_anchor_cell(payload, cell=_cell(payload))


def test_a_grid_collision_is_kept_and_an_unexplained_teleport_is_refused() -> None:
    kept = _sumo_payload("cityflow_grid4x4")
    collision = {"time": 812.0, "collider": "17", "victim": "18", "colliderType": "cf_parity",
                 "victimType": "cf_parity", "colliderSpeed": 3.0, "victimSpeed": 0.0, "type": "collision",
                 "lane": "A0B0_0", "pos": 50.0, "collider_fate": "put_back", "collider_fate_time": 814.0}
    kept.update(collisions=[collision], n_collisions=1, teleports=[{"time": 812.0, "vehicle": "17"}],
                n_teleports=1, n_explained_teleports=1)
    sar.validate_anchor_cell(kept, cell=_cell(kept))
    refused = _sumo_payload("cityflow_grid4x4")
    refused.update(teleports=[{"time": 900.0, "vehicle": "5"}], n_teleports=1, n_unexplained_teleports=1)
    with pytest.raises(ValueError, match="unexplained"):
        sar.validate_anchor_cell(refused, cell=_cell(refused))


def test_the_validator_refuses_a_grid_cell_whose_two_return_routes_disagree() -> None:
    payload = _sumo_payload("cityflow_grid4x4")
    payload["local_return_from_lanes"]["I3"] = -11.0
    with pytest.raises(ValueError, match="return routes"):
        sar.validate_anchor_cell(payload, cell=_cell(payload))


def test_the_validator_refuses_a_cityflow_cell_whose_counts_do_not_add_up() -> None:
    payload = _cityflow_payload()
    payload["never_entered"] = 3
    with pytest.raises(ValueError, match="created"):
        sar.validate_anchor_cell(payload, cell=_cell(payload))


# ----------------------------------------------------------------------
# The resumable roll, with a stand-in episode
# ----------------------------------------------------------------------


def _fake_episode(cell: dict[str, Any]) -> dict[str, Any]:
    payload = (_sumo_payload if cell["backend"] == "sumo" else _cityflow_payload)(
        cell["scenario"], **({"arm": cell["arm"]} if cell["backend"] == "sumo" else {}), draw_id=cell["draw_id"]
    )
    return payload


def _small_declaration() -> list[dict[str, Any]]:
    return [c for c in sar.declared_anchor_cells() if c["draw_id"] in (1000, 1001)]


def test_the_shards_partition_the_declaration(tmp_path: Path) -> None:
    cells = _small_declaration()
    seen: list[str] = []
    for shard in range(3):
        result = sar.roll_cells(cells, work_dir=tmp_path / "w", shard=shard, of=3, episode_fn=_fake_episode)
        seen.extend(result["rolled_names"])
    assert sorted(seen) == sorted(sar.cell_name(c) for c in cells)
    assert len(set(seen)) == len(seen)


def test_a_resumed_roll_skips_every_valid_chunk_and_rolls_nothing(tmp_path: Path) -> None:
    cells = _small_declaration()
    first = sar.roll_cells(cells, work_dir=tmp_path / "w", shard=0, of=1, episode_fn=_fake_episode)
    assert first["rolled"] == len(cells) and first["skipped"] == 0
    calls: list[Any] = []

    def counting(cell: dict[str, Any]) -> dict[str, Any]:
        calls.append(cell)
        return _fake_episode(cell)

    second = sar.roll_cells(cells, work_dir=tmp_path / "w", shard=0, of=1, episode_fn=counting)
    assert second["rolled"] == 0 and second["skipped"] == len(cells) and calls == []


def test_an_existing_chunk_that_does_not_validate_is_refused_and_left_untouched(tmp_path: Path) -> None:
    cells = _small_declaration()[:1]
    work = tmp_path / "w"
    work.mkdir()
    path = work / sar.cell_name(cells[0])
    path.write_text(json.dumps({"format_version": "junk"}), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="not overwritten"):
        sar.roll_cells(cells, work_dir=work, shard=0, of=1, episode_fn=_fake_episode)
    assert path.read_bytes() == before


def test_a_refused_episode_writes_nothing_and_creates_no_directory(tmp_path: Path) -> None:
    cells = _small_declaration()[:1]

    def broken(cell: dict[str, Any]) -> dict[str, Any]:
        payload = _fake_episode(cell)
        payload["n_teleports"] = 2
        return payload

    with pytest.raises(ValueError, match="teleport"):
        sar.roll_cells(cells, work_dir=tmp_path / "never", shard=0, of=1, episode_fn=broken)
    assert not (tmp_path / "never").exists()


def test_anchor_status_counts_declared_against_present(tmp_path: Path) -> None:
    work = tmp_path / "w"
    cells = sar.declared_anchor_cells()
    sar.roll_cells(cells[:5], work_dir=work, shard=0, of=1, episode_fn=_fake_episode)
    status = sar.anchor_status(work)
    assert (status["declared"], status["present"], status["missing"], status["complete"]) == (600, 5, 595, False)


# ----------------------------------------------------------------------
# The replica comparison
# ----------------------------------------------------------------------


def test_compare_with_registered_names_differing_and_missing_fields_and_never_a_value() -> None:
    record = {"att_env": 1.0, "e_sumo": 2.0, "n_created": 3}
    payload = {"att_env": 1.0, "e_sumo": 2.5}
    result = sar.compare_with_registered(payload, record, ["att_env", "e_sumo", "n_created"])
    assert result == {"differing": ["e_sumo"], "missing": ["n_created"]}
    assert sar.compare_with_registered(record, record, ["att_env", "e_sumo", "n_created"]) == {
        "differing": [], "missing": []
    }


def test_the_replica_fields_cover_every_measurement_a_registered_anchor_record_carries() -> None:
    """Every non-bookkeeping key of a committed P7.3a anchor record is compared; a new key cannot slip past."""
    record = next(
        c for c in json.loads((DATA / "p7_3a_zero_shot.json").read_bytes())["cells"]
        if c["kind"] == "anchor" and c["arm"] == "maxpressure"
    )
    not_measurements = {
        "kind", "subject", "arm", "seed", "policy_seed", "draw_id", "stage", "checkpoint", "checkpoint_sha256",
        "sha256_checked_against", "calibration_sha256", "canary_seconds", "seconds", "git_commit", "git_dirty",
        "target_rtg", "rtg_first", "rtg_last", "rtg_series", "reward_series", "rtg_advanced_every_decision",
        "n_decisions_in_support", "support_range", "rho_e_sumo", "rho_att_env",
    }
    assert sorted(set(record) - not_measurements) == sorted(sar.REPLICA_COMPARED_FIELDS)


def test_the_replica_gate_passes_only_when_every_registered_record_reproduces(tmp_path: Path) -> None:
    records = {
        ("cityflow1x1", arm): next(
            c for c in json.loads((DATA / "p7_3a_zero_shot.json").read_bytes())["cells"]
            if c["kind"] == "anchor" and c["arm"] == arm and c["draw_id"] == sar.REPLICA_DRAW
        )
        for arm in ("maxpressure", "fixedtime")
    }
    grid = json.loads((DATA / "p7_3d_grid4x4.json").read_bytes())["cells"]
    for arm in ("maxpressure", "fixedtime"):
        records[("cityflow_grid4x4", arm)] = next(
            c for c in grid if c["kind"] == "anchor" and c["arm"] == arm and c["draw_id"] == sar.REPLICA_DRAW
        )

    def replay(scenario: str, draw_id: int, arm: str, **_kwargs: Any) -> dict[str, Any]:
        return dict(records[(scenario, arm)])

    result = sar.replica_gate(draws_root=tmp_path, corpus_root=tmp_path, data_dir=DATA, episode_fn=replay)
    assert result["passed"] is True and len(result["checks"]) == 4

    def drifted(scenario: str, draw_id: int, arm: str, **_kwargs: Any) -> dict[str, Any]:
        payload = dict(records[(scenario, arm)])
        if (scenario, arm) == ("cityflow_grid4x4", "fixedtime"):
            payload["e_sumo"] = float(np.nextafter(payload["e_sumo"], np.inf))
        return payload

    failed = sar.replica_gate(draws_root=tmp_path, corpus_root=tmp_path, data_dir=DATA, episode_fn=drifted)
    assert failed["passed"] is False
    assert [c["differing"] for c in failed["checks"] if not c["passed"]] == [["e_sumo"]]


# ----------------------------------------------------------------------
# Gated: real envs
# ----------------------------------------------------------------------


def _draws_root() -> Path:
    value = os.environ.get("RLTRAFFIC_DRAWS")
    root = Path(value) if value else REPO_ROOT / "scenarios" / "draws"
    if not (root / "cityflow_grid4x4" / "draw_1001" / "parity" / "noteleport.sumocfg").is_file():
        pytest.skip(f"the held-out parity draws are not at {root}: set RLTRAFFIC_DRAWS to the main tree's scenarios/draws")
    return root


def _corpus_root() -> Path:
    value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    root = Path(value) if value else REPO_ROOT / "datasets_v11"
    if not (root / "cf_hz1x1__fixedtime" / "manifest.json").is_file():
        pytest.skip(f"the v1.1 corpus manifests are not at {root}: set RLTRAFFIC_CORPUS_V11 to datasets_v11/")
    return root


def _require_simulators() -> None:
    try:
        import cityflow  # noqa: F401
        import traci  # noqa: F401
    except ImportError:
        pytest.skip("CityFlow or traci is not importable, so the envs cannot be built")
    if shutil.which("sumo") is None:
        pytest.skip("the sumo binary is not on the PATH")


@pytest.mark.parametrize("scenario", ["cityflow1x1", "cityflow_grid4x4"])
def test_the_fixedtime_kcf_cycle_on_sumo_is_the_registered_cityflow_plans_cycle(scenario: str) -> None:
    """360 decision steps of each controller's actions, compared per intersection id: same phase sequence, same steps
    per green.  CityFlow's controller is the registered route (``method_tier_grid._fixedtime_factory`` with the tier
    manifest's settings -- hz1x1's shipped plan at k = 6, grid4x4's equal split at k = 1); SUMO's is :func:`sumo_policy`."""
    from experiments.config import EnvSpec
    from experiments.envs import make_env

    from offline.aligned_env import observer_env_for_draw
    from offline.dt_gate import env_settings_from_manifest
    from offline.materialise_draws import draw_config_path, parity_sumocfg_path
    from offline.method_tier_grid import _fixedtime_factory, fixedtime_collection_settings

    draws, corpus = _draws_root(), _corpus_root()
    _require_simulators()
    tier_dir = corpus / {"cityflow1x1": "cf_hz1x1__fixedtime", "cityflow_grid4x4": "cf_grid4x4__fixedtime"}[scenario]
    collected = fixedtime_collection_settings(tier_dir / "manifest.json")
    kcf = sar.kcf_for(scenario, corpus)
    assert kcf["fixed_time_k"] == collected["fixed_time_k"] == sar.KCF_README_VALUES[scenario]

    config = draw_config_path(scenario, 1001, out_root=draws)
    cf_env = make_env(EnvSpec(id=scenario, backend="cityflow", paths={"config": str(config)},
                              settings=env_settings_from_manifest(tier_dir / "manifest.json")))
    try:
        cf_ids = [str(ix.id) for ix in cf_env.intersections]
        cf_choose = _fixedtime_factory(config, collected)(cf_env)
        cityflow_schedule = sar.fixedtime_action_schedule(lambda info: cf_choose(cf_env, info), cf_ids, 360)
    finally:
        cf_env.close()
    sumo_env = observer_env_for_draw(scenario, 1001, out_root=draws, arm="fixedtime")
    try:
        sumo_ids = [str(ix.id) for ix in sumo_env.intersections]
        policy, provenance = sar.sumo_policy(
            sumo_env, arm="fixedtime_kcf", config_path=parity_sumocfg_path(scenario, 1001, out_root=draws),
            kcf=kcf["fixed_time_k"],
        )
        sumo_schedule = sar.fixedtime_action_schedule(policy, sumo_ids, 360)
    finally:
        sumo_env.close()
    assert sorted(sumo_ids) == sorted(cf_ids)
    assert provenance["fixed_time_k"] == kcf["fixed_time_k"]
    assert provenance["fixed_time_schedule_source"] == "equal_split"
    assert sumo_schedule == cityflow_schedule
    first = sumo_schedule[sumo_ids[0]]
    k = kcf["fixed_time_k"]
    assert first[: 8 * k] == [action for action in range(8) for _ in range(k)]


def _registered_anchor(artifact: str, arm: str, draw_id: int) -> dict[str, Any]:
    cells = json.loads((DATA / artifact).read_bytes())["cells"]
    return next(c for c in cells if c["kind"] == "anchor" and c["arm"] == arm and c["draw_id"] == draw_id)


def test_the_sumo_replica_reproduces_a_registered_hz1x1_anchor_record_under_equality() -> None:
    """``maxpressure`` (the frozen agent) on draw 1001 through :func:`sumo_anchor_episode` -- the anchor branch of
    ``run_cell`` with the policy pluggable -- against P7.3a's committed record, every measurement field, ``==``."""
    draws, corpus = _draws_root(), _corpus_root()
    _require_simulators()
    payload = sar.sumo_anchor_episode("cityflow1x1", sar.REPLICA_DRAW, "maxpressure", draws_root=draws,
                                      corpus_root=corpus)
    record = _registered_anchor("p7_3a_zero_shot.json", "maxpressure", sar.REPLICA_DRAW)
    assert sar.compare_with_registered(payload, record, sar.REPLICA_COMPARED_FIELDS) == {"differing": [], "missing": []}


def test_the_sumo_replica_reproduces_a_registered_grid4x4_anchor_record_under_equality() -> None:
    """``fixedtime`` at the registered k = 4 on draw 1001, against P7.3d's committed record -- the grid branch, its
    per-intersection return and collision record included."""
    draws, corpus = _draws_root(), _corpus_root()
    _require_simulators()
    payload = sar.sumo_anchor_episode("cityflow_grid4x4", sar.REPLICA_DRAW, "fixedtime", draws_root=draws,
                                      corpus_root=corpus)
    record = _registered_anchor("p7_3d_grid4x4.json", "fixedtime", sar.REPLICA_DRAW)
    fields = (*sar.REPLICA_COMPARED_FIELDS, "intersection_ids", "local_return")
    assert sar.compare_with_registered(payload, record, fields) == {"differing": [], "missing": []}
    assert payload["policy"]["fixed_time_k"] == 4


def test_a_new_sumo_cell_validates_and_carries_its_policy_provenance() -> None:
    from offline.policies.max_pressure_1to1 import roadlink_structure_digest  # noqa: F401  (imported: the digest's route)

    draws, corpus = _draws_root(), _corpus_root()
    _require_simulators()
    payload = sar.sumo_anchor_episode("cityflow1x1", sar.REPLICA_DRAW, "maxpressure_1to1", draws_root=draws,
                                      corpus_root=corpus)
    # The tree this test runs in may carry uncommitted files; the provenance check is the campaign's, not this test's.
    payload["git_dirty"] = False
    sar.validate_anchor_cell(payload, cell=_cell(payload))
    assert payload["policy"]["multiplicities"]["summary"]["selectable_greens"]["incoming_multiplicities"] == [2]
    assert len(payload["policy"]["roadlink_structure_digest"]) == 64
    assert payload["depart_lane_attributes"] == 0
    assert payload["base_arm"] == "maxpressure"


def test_the_cityflow_maxpressure_1to1_hz1x1_cell_is_the_registered_maxpressure_cell() -> None:
    """Draw 1000, P8.4b's route: both definitions and the companions equal the committed P8.4a escalated record."""
    from offline.admission_probe import ProbeRoots

    draws, corpus = _draws_root(), _corpus_root()
    _require_simulators()
    roots = ProbeRoots(repo_root=REPO_ROOT, corpus_root=corpus, draws_root=draws, output_root=REPO_ROOT / "output",
                       work_dir=REPO_ROOT / "output" / "unused")
    payload = sar.cityflow_anchor_episode("cityflow1x1", 1000, "maxpressure_1to1", roots=roots)
    record = next(
        e for e in json.loads((DATA / "p8_4a_admission_escalated.json").read_bytes())["episodes"]
        if e["scenario"] == "hz1x1" and e["arm"] == "behaviour@maxpressure" and e["draw_id"] == 1000
    )
    fields = ("att_ours", "att_engine", "created", "entered", "never_entered", "horizon_vehicle_count",
              "episode_reward")
    assert {f: payload[f] for f in fields} == {f: record[f] for f in fields}
