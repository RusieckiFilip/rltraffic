"""Tests for ``offline/demand_shift.py``, part 1 -- the flow audit and the guarded materialisation (``BRIEF_47`` §3.4, §4 test 4).

* **The audit's checks each fail on a planted defect**: a second vehicle-parameter block, a route through a road absent from
  the roadnet, a route step no roadLink joins, an out-of-range start time, a startTime inversion, an aggregate entry, a
  roadnet or a plan that is not bc-tyc's.  The planted scenarios are copies of bc-tyc's files in ``tmp_path``.
* **The real files pass** (kn-hz, qc-yn, sb-sx against bc-tyc), with the facts A27(h) states: 743 / 1,417 / 1,671 vehicles
  against 2,021, one vehicle block identical to bc-tyc's, eight routes, start times within 0-3,599, single insertions,
  no inversion; and two materialisations of the held-out draws are byte-identical and equal to the in-memory draw.
* **materialise** refuses an out-root inside the worktree, a working directory other than the main tree's root, and a
  missing, failed or stale audit -- each before anything is written.
* **The fixed-time plan identity** across the four scenario directories.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

import offline.demand_shift as ds

REPO_ROOT = Path(__file__).resolve().parents[1]
BC_TYC_DIR = REPO_ROOT / "scenarios" / "hangzhou_1x1_bc-tyc_18041610_1h"
TWO_DRAWS = (1000, 1001)


def _planted(tmp_path: Path, mutate: Any, *, name: str = "planted") -> Path:
    """A copy of bc-tyc's scenario with its flow passed through *mutate*, and a sim config pointing at it."""
    scenario = tmp_path / f"scenario_{name}"
    scenario.mkdir()
    for file_name in ("roadnet.json", "signal_plan_template.txt"):
        shutil.copyfile(BC_TYC_DIR / file_name, scenario / file_name)
    flow = json.loads((BC_TYC_DIR / "flow.json").read_bytes())
    flow = mutate(flow) or flow
    (scenario / "flow.json").write_text(json.dumps(flow), encoding="utf-8")
    config = tmp_path / f"{name}.json"
    config.write_text(
        json.dumps({"interval": 1.0, "seed": 0, "dir": str(scenario) + "/", "roadnetFile": "roadnet.json",
                    "flowFile": "flow.json", "rlTrafficLight": True, "saveReplay": False, "laneChange": False}),
        encoding="utf-8",
    )
    return config


def _audit(config: Path, tmp_path: Path) -> dict[str, Any]:
    return ds.audit_file(config, reference_config=REPO_ROOT / "configs/sim/cityflow1x1.json", repo_root=REPO_ROOT,
                         draw_ids=TWO_DRAWS, scratch=tmp_path / "scratch")


# ----------------------------------------------------------------------
# Planted defects, one check each
# ----------------------------------------------------------------------


def test_an_unchanged_copy_of_bc_tyc_passes_every_check(tmp_path: Path) -> None:
    record = _audit(_planted(tmp_path, lambda flow: None), tmp_path)
    assert record["passed"] is True
    assert all(check["passed"] for check in record["checks"].values())
    assert record["checks"]["vehicles"]["n_vehicles"] == 2021


def _failing(record: dict[str, Any]) -> list[str]:
    return sorted(name for name, check in record["checks"].items() if not check["passed"])


def test_a_second_vehicle_block_fails_the_vehicle_block_check(tmp_path: Path) -> None:
    def mutate(flow: list[dict[str, Any]]) -> None:
        flow[100]["vehicle"] = {**flow[100]["vehicle"], "maxSpeed": 13.89}

    record = _audit(_planted(tmp_path, mutate), tmp_path)
    assert record["passed"] is False
    assert _failing(record) == ["vehicle_blocks"]
    assert record["checks"]["vehicle_blocks"]["n_distinct"] == 2


def test_one_vehicle_block_that_is_not_bc_tycs_fails_the_vehicle_block_check(tmp_path: Path) -> None:
    def mutate(flow: list[dict[str, Any]]) -> None:
        for entry in flow:
            entry["vehicle"] = {**entry["vehicle"], "length": 4.5}

    record = _audit(_planted(tmp_path, mutate), tmp_path)
    assert _failing(record) == ["vehicle_blocks"]
    assert (record["checks"]["vehicle_blocks"]["n_distinct"], record["checks"]["vehicle_blocks"]["equal_to_reference"]) == (
        1, False)


def test_a_route_through_a_road_absent_from_the_roadnet_fails_the_route_check(tmp_path: Path) -> None:
    def mutate(flow: list[dict[str, Any]]) -> None:
        flow[7]["route"] = ["road_0_1_0", "road_9_9_9"]

    record = _audit(_planted(tmp_path, mutate), tmp_path)
    assert _failing(record) == ["routes"]
    assert record["checks"]["routes"]["missing_roads"] == ["road_9_9_9"]


def test_a_route_step_no_roadlink_joins_fails_the_route_check(tmp_path: Path) -> None:
    def mutate(flow: list[dict[str, Any]]) -> None:
        flow[7]["route"] = ["road_0_1_0", "road_1_1_2"]  # both roads exist; no roadLink turns one into the other

    record = _audit(_planted(tmp_path, mutate), tmp_path)
    assert _failing(record) == ["routes"]
    assert record["checks"]["routes"]["unlinked_steps"] == [["road_0_1_0", "road_1_1_2"]]


def test_an_out_of_range_start_time_fails_the_start_time_check(tmp_path: Path) -> None:
    def mutate(flow: list[dict[str, Any]]) -> None:
        flow[-1]["startTime"] = flow[-1]["endTime"] = 3600

    record = _audit(_planted(tmp_path, mutate), tmp_path)
    assert _failing(record) == ["start_times"]
    assert record["checks"]["start_times"]["max"] == 3600


def test_a_start_time_inversion_fails_the_randomisers_inversion_check(tmp_path: Path) -> None:
    def mutate(flow: list[dict[str, Any]]) -> None:
        flow[10]["startTime"] = flow[10]["endTime"] = flow[11]["startTime"] + 5

    record = _audit(_planted(tmp_path, mutate), tmp_path)
    assert "inversions" in _failing(record)
    assert record["checks"]["inversions"]["n_inversions"] >= 1


def test_an_aggregate_entry_fails_the_single_insertion_and_the_randomiser_checks(tmp_path: Path) -> None:
    def mutate(flow: list[dict[str, Any]]) -> None:
        flow[20]["endTime"] = flow[20]["startTime"] + 10

    record = _audit(_planted(tmp_path, mutate), tmp_path)
    assert {"single_insertions", "randomiser_round_trip"} <= set(_failing(record))
    assert record["checks"]["single_insertions"]["n_aggregate"] == 1


def test_a_roadnet_that_is_not_bc_tycs_fails_the_identity_check(tmp_path: Path) -> None:
    config = _planted(tmp_path, lambda flow: None)
    roadnet = Path(json.loads(config.read_bytes())["dir"]) / "roadnet.json"
    roadnet.write_bytes(roadnet.read_bytes() + b"\n")
    record = _audit(config, tmp_path)
    assert "roadnet_identical_to_reference" in _failing(record)


def test_a_plan_that_is_not_bc_tycs_fails_the_plan_check(tmp_path: Path) -> None:
    config = _planted(tmp_path, lambda flow: None)
    plan = Path(json.loads(config.read_bytes())["dir"]) / "signal_plan_template.txt"
    plan.write_bytes(plan.read_bytes() + b"1\n")
    record = _audit(config, tmp_path)
    assert "plan_identical_to_reference" in _failing(record)


# ----------------------------------------------------------------------
# The real files
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "n_vehicles", "start_min", "start_max"),
    [("kn-hz", 743, 5, 3597), ("qc-yn", 1417, 1, 3598), ("sb-sx", 1671, 1, 3593)],
)
def test_the_three_recorded_files_pass_the_audit_with_a27h_s_facts(
    tmp_path: Path, name: str, n_vehicles: int, start_min: int, start_max: int
) -> None:
    spec = ds.SHIFTED_FILES[name]
    record = ds.audit_file(REPO_ROOT / spec["config"], reference_config=REPO_ROOT / "configs/sim/cityflow1x1.json",
                           repo_root=REPO_ROOT, draw_ids=TWO_DRAWS, scratch=tmp_path / "scratch")
    assert record["passed"] is True, _failing(record)
    checks = record["checks"]
    assert checks["vehicles"]["n_vehicles"] == n_vehicles
    assert checks["vehicles"]["reference_n_vehicles"] == 2021
    assert checks["vehicles"]["ratio_to_reference"] == n_vehicles / 2021
    assert (checks["start_times"]["min"], checks["start_times"]["max"]) == (start_min, start_max)
    assert checks["vehicle_blocks"]["n_distinct"] == 1 and checks["vehicle_blocks"]["equal_to_reference"] is True
    assert checks["routes"]["n_distinct"] == 8
    assert checks["inversions"]["n_inversions"] == 0
    draws = checks["materialised_draws"]
    assert draws["two_materialisations_identical"] is True and draws["flow_equals_in_memory_draw"] is True
    assert sorted(draws["flow_sha256"]) == [str(d) for d in TWO_DRAWS]


def test_bc_tycs_audit_reproduces_its_registered_materialised_draws_when_they_are_present(tmp_path: Path) -> None:
    value = os.environ.get("RLTRAFFIC_DRAWS")
    draws_root = Path(value) if value else REPO_ROOT / "scenarios" / "draws"
    if not (draws_root / "cityflow1x1" / "draw_1001" / "flow.json").is_file():
        pytest.skip(f"bc-tyc's held-out draws are not at {draws_root}: set RLTRAFFIC_DRAWS to the main tree's scenarios/draws")
    record = ds.audit_file(REPO_ROOT / "configs/sim/cityflow1x1.json",
                           reference_config=REPO_ROOT / "configs/sim/cityflow1x1.json", repo_root=REPO_ROOT,
                           draw_ids=TWO_DRAWS, scratch=tmp_path / "scratch", registered_draws_root=draws_root)
    reproduced = record["checks"]["materialised_draws"]["reproduces_registered_draws"]
    assert reproduced == {"n_checked": 2, "n_equal": 2}


def test_the_fixed_time_plan_is_byte_identical_in_the_four_scenario_directories() -> None:
    import hashlib

    from offline.policies.fixed_time import parse_signal_plan

    digests = set()
    schedules = set()
    for spec in (ds.REFERENCE_FILE, *ds.SHIFTED_FILES.values()):
        config = json.loads((REPO_ROOT / spec["config"]).read_bytes())
        plan = REPO_ROOT / config["dir"] / "signal_plan_template.txt"
        digests.add(hashlib.sha256(plan.read_bytes()).hexdigest())
        schedules.add(repr(parse_signal_plan(plan.read_text(encoding="utf-8"))))
    assert len(digests) == 1 and len(schedules) == 1


# ----------------------------------------------------------------------
# materialise: refusals before any write -- every test in a STAND-IN main tree, so no test can reach the real draws root
# ----------------------------------------------------------------------


def _stand_in_tree(tmp_path: Path) -> Path:
    tree = tmp_path / "tree"
    (tree / "configs" / "sim").mkdir(parents=True)
    for spec in ds.SHIFTED_FILES.values():
        shutil.copyfile(REPO_ROOT / spec["config"], tree / spec["config"])
        directory = json.loads((REPO_ROOT / spec["config"]).read_bytes())["dir"]
        shutil.copytree(REPO_ROOT / directory, tree / directory)
    return tree


def _fake_audit(tmp_path: Path, tree: Path, *, passed: bool = True, flow_sha256: str | None = None) -> Path:
    path = tmp_path / "audit.json"
    sources = {
        spec["scenario_key"]: {"config_sha256": ds.sha256_file(tree / spec["config"]),
                               "flow_sha256": flow_sha256 or ds.source_flow_sha256(tree / spec["config"], tree)}
        for spec in ds.SHIFTED_FILES.values()
    }
    path.write_text(json.dumps({"format_version": ds.AUDIT_FORMAT_VERSION, "passed": passed, "sources": sources,
                                "expected_flow_sha256": {}}), encoding="utf-8")
    return path


def test_materialise_refuses_an_out_root_inside_the_worktree(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="registered draws root"):
        ds.materialise_shifted(out_root=REPO_ROOT / "scenarios" / "draws", audit_path=tmp_path / "unused.json",
                               repo_root=REPO_ROOT)
    assert not (REPO_ROOT / "scenarios" / "draws").exists()


def test_materialise_refuses_any_out_root_but_the_registered_one(tmp_path: Path) -> None:
    tree = _stand_in_tree(tmp_path)
    with pytest.raises(ValueError, match="registered draws root"):
        ds.materialise_shifted(out_root=tmp_path / "draws", audit_path=_fake_audit(tmp_path, tree), repo_root=tree,
                               registered_root=tree / "scenarios" / "draws", main_tree=tree)
    assert not (tmp_path / "draws").exists()


def test_materialise_refuses_a_working_directory_other_than_the_main_trees_root(tmp_path: Path) -> None:
    tree = _stand_in_tree(tmp_path)
    draws = tree / "scenarios" / "draws"
    with pytest.raises(ValueError, match="main tree"):
        ds.materialise_shifted(out_root=draws, audit_path=_fake_audit(tmp_path, tree), repo_root=tree,
                               registered_root=draws, main_tree=tree)
    assert not draws.exists()


def test_materialise_refuses_a_missing_a_failed_and_a_stale_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tree = _stand_in_tree(tmp_path)
    draws = tree / "scenarios" / "draws"
    monkeypatch.chdir(tree)
    kwargs = {"out_root": draws, "repo_root": tree, "registered_root": draws, "main_tree": tree}
    with pytest.raises(FileNotFoundError, match="audit"):
        ds.materialise_shifted(audit_path=tmp_path / "absent.json", **kwargs)
    with pytest.raises(ValueError, match="did not pass"):
        ds.materialise_shifted(audit_path=_fake_audit(tmp_path, tree, passed=False), **kwargs)
    with pytest.raises(ValueError, match="stale"):
        ds.materialise_shifted(audit_path=_fake_audit(tmp_path, tree, flow_sha256="0" * 64), **kwargs)
    assert not draws.exists()


def test_materialise_writes_only_the_audited_draws_into_a_test_main_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The happy path, in a stand-in main tree: the three files' draws written, each flow.json equal to the audit's digest,
    each draw's dir pointing into THAT tree, and a second run keeps every draw."""
    tree = _stand_in_tree(tmp_path)
    draws_root = tree / "scenarios" / "draws"
    audit = ds.run_audit(repo_root=tree, draw_ids=TWO_DRAWS, scratch=tmp_path / "scratch", registered_draws_root=None,
                         reference_repo_root=REPO_ROOT)
    assert audit["passed"] is True
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    monkeypatch.chdir(tree)
    kwargs = {"out_root": draws_root, "audit_path": audit_path, "repo_root": tree, "draw_ids": TWO_DRAWS,
              "registered_root": draws_root, "main_tree": tree}
    first = ds.materialise_shifted(**kwargs)
    assert first["n_written"] == 6 and first["n_flow_digests_equal"] == 6
    second = ds.materialise_shifted(**kwargs)
    assert second["n_written"] == 0 and second["n_kept"] == 6
    config = json.loads((draws_root / "cityflow1x1_config3" / "draw_1000" / "cityflow.json").read_bytes())
    assert config["dir"] == str(tree / "scenarios" / "hangzhou_1x1_kn-hz_18041608_1h") + "/"
