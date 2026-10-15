"""P9.2 (``BRIEF_47`` §3.4): A27(h)'s EXPLORATORY zero-shot demand-shift evaluation -- the flow audit, the guarded
materialisation of the held-out draws, the evaluation by two routes and its artifact.

**Label (A27(a), item 7): exploration, not H2.**  This evaluation trains nothing; it is not a test of H2 (a registered 2x2
with shift-augmented training, untested under A25(a)), and it is not the registered Hangzhou transfer pairs, which are
CityFlow-to-SUMO.

Format versions: the audit ``p9.2-demand-shift-audit/1.0`` (:data:`AUDIT_FORMAT_VERSION`,
``docs/data/p9_2_demand_shift_audit.json``, committed BEFORE any draw is materialised); the evaluation cell
``p9.2-demand-shift-cell/1.0``; the artifact ``p9.2-demand-shift/1.0`` (``docs/data/p9_2_demand_shift.json``).

THE FILES (measured 2026-10-15, A27(h); re-measured by :func:`audit_file`)
--------------------------------------------------------------------------
``kn-hz``, ``qc-yn`` and ``sb-sx`` are recordings of OTHER intersections at OTHER hours (2018-04-16 08:00, 08:00 and 07:00
against bc-tyc's 10:00) mapped onto bc-tyc's road-network file byte for byte.  Volume, turning mix and time profile change
together, so every sentence says "demand shift", never "lower volume".  Their sim configs are
``configs/sim/cityflow1x1_config3/2/4.json``; their held-out draws are keyed ``cityflow1x1_config3/2/4`` (the config's
stem, ``materialise_draws.scenario_key_for_config``).

THE AUDIT -- bc-tyc's flow audit, as this task's pre-flight
-----------------------------------------------------------
Per file, each check recorded with ``passed``: the road network and the fixed-time plan byte-identical to bc-tyc's; ONE
vehicle-parameter block, equal to bc-tyc's; the route set, every road present in the roadnet and every consecutive pair
of roads joined by a roadLink; every start time an integer within ``0 .. 3599`` (the 3,600 s horizon); single
insertions only (``startTime == endTime``); the randomiser's inversion count (0: ``draw(0)`` and ``draw(k > 0)`` then differ
in demand only, ``offline/flow_randomizer.py``'s module docstring) and its construction-time byte round-trip; and the
reproduction of materialised draws -- the held-out draws materialised TWICE into two scratch roots by
``materialise_draws.materialise`` and compared byte for byte, each ``flow.json`` equal to ``FlowRandomizer.draw`` rendered in
memory, and for bc-tyc each equal to the draw already materialised in the main tree.  The expected ``flow.json`` sha256 of
every held-out draw is recorded so the real materialisation is checked against it afterwards.

THE MATERIALISATION (``DEFERRED`` 55; plan C8)
---------------------------------------------
:func:`materialise_shifted` refuses an out-root other than the main tree's ``scenarios/draws``, a working directory other
than the main tree's root (``materialise_draws`` embeds the source scenario's ``dir`` resolved against the cwd in every
draw's ``cityflow.json``, so a draw materialised from a worktree points into a tree that is deleted with it), and a
missing, failed or stale audit -- all before anything is written.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "AUDIT_FORMAT_VERSION",
    "AUDIT_NAME",
    "HELD_OUT_DRAWS",
    "HORIZON_SECONDS",
    "MAIN_TREE",
    "REFERENCE_FILE",
    "REGISTERED_DRAWS_ROOT",
    "SHIFTED_FILES",
    "audit_file",
    "materialise_shifted",
    "run_audit",
    "sha256_file",
    "source_flow_sha256",
]

#: The held-out pool (D4).
HELD_OUT_DRAWS: tuple[int, ...] = tuple(range(1000, 1100))

#: The episode horizon every evaluation runs (360 decisions x 10 s); a start time at or after it never enters.
HORIZON_SECONDS = 3600

#: A27(h)'s three files, keyed by A27(h)'s names: the sim config, the draws' scenario key and the recording's hour.
SHIFTED_FILES: Mapping[str, Mapping[str, str]] = {
    "kn-hz": {"config": "configs/sim/cityflow1x1_config3.json", "scenario_key": "cityflow1x1_config3",
              "directory_tag": "kn-hz_18041608", "recorded": "2018-04-16 08:00"},
    "qc-yn": {"config": "configs/sim/cityflow1x1_config2.json", "scenario_key": "cityflow1x1_config2",
              "directory_tag": "qc-yn_18041608", "recorded": "2018-04-16 08:00"},
    "sb-sx": {"config": "configs/sim/cityflow1x1_config4.json", "scenario_key": "cityflow1x1_config4",
              "directory_tag": "sb-sx_18041607", "recorded": "2018-04-16 07:00"},
}

#: bc-tyc, the training demand every checkpoint here was trained on, and the reference of every audit comparison.
REFERENCE_FILE: Mapping[str, str] = {
    "name": "bc-tyc", "config": "configs/sim/cityflow1x1.json", "scenario_key": "cityflow1x1",
    "directory_tag": "bc-tyc_18041610", "recorded": "2018-04-16 10:00",
}

MAIN_TREE = Path("/home/filip/rltraffic")
REGISTERED_DRAWS_ROOT = MAIN_TREE / "scenarios" / "draws"
AUDIT_NAME = "p9_2_demand_shift_audit.json"
AUDIT_FORMAT_VERSION = "p9.2-demand-shift-audit/1.0"

_ENTRY_KEYS = ("vehicle", "route", "interval", "startTime", "endTime")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scenario_dir(config: str | Path, repo_root: str | Path) -> Path:
    """The scenario directory a sim config names, a relative ``dir`` resolved against *repo_root* (never the cwd)."""
    payload = json.loads(Path(config).read_bytes())
    directory = str(payload["dir"])
    return Path(os.path.normpath(directory if os.path.isabs(directory) else Path(repo_root) / directory))


def source_flow_sha256(config: str | Path, repo_root: str | Path) -> str:
    """sha256 of the flow file *config* points at."""
    payload = json.loads(Path(config).read_bytes())
    return sha256_file(_scenario_dir(config, repo_root) / payload["flowFile"])


def _check(passed: bool, **facts: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **facts}


def _route_check(entries: Sequence[Mapping[str, Any]], roadnet: Mapping[str, Any]) -> dict[str, Any]:
    roads = {str(road["id"]) for road in roadnet.get("roads", [])}
    links = {
        (str(link["startRoad"]), str(link["endRoad"]))
        for ix in roadnet.get("intersections", [])
        for link in ix.get("roadLinks", [])
    }
    routes = sorted({tuple(str(road) for road in entry["route"]) for entry in entries})
    missing = sorted({road for route in routes for road in route if road not in roads})
    unlinked = sorted(
        {(route[i], route[i + 1]) for route in routes for i in range(len(route) - 1) if (route[i], route[i + 1]) not in links}
    )
    return _check(
        not missing and not unlinked and bool(routes),
        n_distinct=len(routes),
        routes=[list(route) for route in routes],
        missing_roads=missing,
        unlinked_steps=[list(step) for step in unlinked],
    )


def _draw_check(
    config: Path, repo_root: Path, *, draw_ids: Sequence[int], scratch: Path, registered_draws_root: Path | None,
) -> dict[str, Any]:
    """The held-out draws materialised twice into scratch roots and compared, each against the in-memory draw too."""
    from offline.flow_randomizer import (
        DEFAULT_BASE_SEED,
        DEFAULT_JITTER_SIGMA_S,
        DEFAULT_THIN_P,
        DEFAULT_VOLUME_SCALE,
        FlowRandomizer,
    )
    from offline.materialise_draws import materialise

    payload = json.loads(config.read_bytes())
    # A scratch copy of the sim config with an ABSOLUTE dir, under the same stem (= the scenario key): the audit reads
    # exactly the files the config names whatever the process's working directory is.
    stem_config = scratch / "configs" / config.name
    stem_config.parent.mkdir(parents=True, exist_ok=True)
    stem_config.write_text(
        json.dumps({**payload, "dir": str(_scenario_dir(config, repo_root)) + "/"}, indent=4), encoding="utf-8"
    )
    roots = [scratch / "first", scratch / "second"]
    try:
        records = [materialise(stem_config, list(draw_ids), out_root=root) for root in roots]
    except (ValueError, OSError) as exc:
        return _check(False, error=f"{type(exc).__name__}: {exc}")
    key = records[0][0].scenario_key
    randomizer = FlowRandomizer(
        _scenario_dir(config, repo_root) / payload["flowFile"], base_seed=DEFAULT_BASE_SEED,
        jitter_sigma_s=DEFAULT_JITTER_SIGMA_S, thin_p=DEFAULT_THIN_P, volume_scale=DEFAULT_VOLUME_SCALE,
    )
    identical = True
    in_memory = True
    flow_sha: dict[str, str] = {}
    routes_sha: dict[str, str | None] = {}
    reproduced = 0
    for draw in draw_ids:
        files = [root / key / f"draw_{int(draw):04d}" for root in roots]
        flows = [(directory / "flow.json").read_bytes() for directory in files]
        routes = [
            (directory / "routes.rou.xml").read_bytes() if (directory / "routes.rou.xml").is_file() else None
            for directory in files
        ]
        identical &= flows[0] == flows[1] and routes[0] == routes[1]
        entries, _provenance = randomizer.draw(int(draw))
        in_memory &= randomizer.render_cityflow_bytes(entries) == flows[0]
        flow_sha[str(draw)] = hashlib.sha256(flows[0]).hexdigest()
        routes_sha[str(draw)] = None if routes[0] is None else hashlib.sha256(routes[0]).hexdigest()
        if registered_draws_root is not None:
            registered = Path(registered_draws_root) / key / f"draw_{int(draw):04d}" / "flow.json"
            reproduced += int(registered.is_file() and registered.read_bytes() == flows[0])
    shutil.rmtree(scratch / "first", ignore_errors=True)
    shutil.rmtree(scratch / "second", ignore_errors=True)
    reproduces = (
        None if registered_draws_root is None else {"n_checked": len(draw_ids), "n_equal": reproduced}
    )
    passed = identical and in_memory and (reproduces is None or reproduced == len(draw_ids))
    return _check(
        passed,
        scenario_key=key,
        draw_ids=[int(draw) for draw in draw_ids],
        two_materialisations_identical=identical,
        flow_equals_in_memory_draw=in_memory,
        reproduces_registered_draws=reproduces,
        flow_sha256=flow_sha,
        routes_sha256=routes_sha,
    )


def audit_file(
    config: str | Path, *, reference_config: str | Path, repo_root: str | Path, draw_ids: Sequence[int],
    scratch: str | Path, registered_draws_root: str | Path | None = None, reference_repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """bc-tyc's flow audit on one file, every check recorded with ``passed`` (none raises on a defect: the record IS the
    finding).  *reference_config* is bc-tyc's sim config, resolved against *reference_repo_root* (default *repo_root*)."""
    from offline.flow_randomizer import FlowRandomizer
    from offline.policies.fixed_time import parse_signal_plan

    config = Path(config)
    repo = Path(repo_root)
    reference_repo = Path(reference_repo_root) if reference_repo_root is not None else repo
    payload = json.loads(config.read_bytes())
    reference_payload = json.loads(Path(reference_config).read_bytes())
    scenario = _scenario_dir(config, repo)
    reference_scenario = _scenario_dir(reference_config, reference_repo)
    flow_path = scenario / payload["flowFile"]
    entries = json.loads(flow_path.read_bytes())
    reference_entries = json.loads((reference_scenario / reference_payload["flowFile"]).read_bytes())
    roadnet_path = scenario / payload["roadnetFile"]
    reference_roadnet_path = reference_scenario / reference_payload["roadnetFile"]
    plan_path = scenario / "signal_plan_template.txt"
    reference_plan_path = reference_scenario / "signal_plan_template.txt"

    checks: dict[str, Any] = {}
    checks["roadnet_identical_to_reference"] = _check(
        sha256_file(roadnet_path) == sha256_file(reference_roadnet_path),
        sha256=sha256_file(roadnet_path), reference_sha256=sha256_file(reference_roadnet_path),
        md5=hashlib.md5(roadnet_path.read_bytes()).hexdigest(),
    )
    plan_equal = plan_path.is_file() and sha256_file(plan_path) == sha256_file(reference_plan_path)
    checks["plan_identical_to_reference"] = _check(
        plan_equal and repr(parse_signal_plan(plan_path.read_text(encoding="utf-8")))
        == repr(parse_signal_plan(reference_plan_path.read_text(encoding="utf-8"))),
        sha256=sha256_file(plan_path) if plan_path.is_file() else None,
        reference_sha256=sha256_file(reference_plan_path),
    )
    shape_ok = isinstance(entries, list) and all(
        isinstance(entry, Mapping) and all(key in entry for key in _ENTRY_KEYS) for entry in entries
    )
    checks["entries"] = _check(shape_ok, keys=list(_ENTRY_KEYS))
    checks["vehicles"] = _check(
        True, n_vehicles=len(entries), reference_n_vehicles=len(reference_entries),
        ratio_to_reference=len(entries) / len(reference_entries),
    )
    blocks = sorted({json.dumps(entry["vehicle"], sort_keys=True) for entry in entries})
    reference_blocks = sorted({json.dumps(entry["vehicle"], sort_keys=True) for entry in reference_entries})
    checks["vehicle_blocks"] = _check(
        len(blocks) == 1 and blocks == reference_blocks,
        n_distinct=len(blocks), equal_to_reference=blocks == reference_blocks,
        blocks=[json.loads(block) for block in blocks],
    )
    checks["routes"] = _route_check(entries, json.loads(roadnet_path.read_bytes()))
    starts = [entry["startTime"] for entry in entries]
    integral = all(isinstance(value, int) and not isinstance(value, bool) for value in starts)
    checks["start_times"] = _check(
        integral and min(starts) >= 0 and max(starts) <= HORIZON_SECONDS - 1,
        min=min(starts), max=max(starts), all_integer=integral, allowed=[0, HORIZON_SECONDS - 1],
    )
    aggregate = sum(1 for entry in entries if entry["startTime"] != entry["endTime"])
    checks["single_insertions"] = _check(aggregate == 0, n_aggregate=aggregate)
    inversions = sum(1 for first, second in zip(starts, starts[1:]) if second < first)
    checks["inversions"] = _check(inversions == 0, n_inversions=inversions)
    try:
        FlowRandomizer(flow_path)
        checks["randomiser_round_trip"] = _check(True)
    except ValueError as exc:
        checks["randomiser_round_trip"] = _check(False, error=str(exc)[:300])
    checks["materialised_draws"] = _draw_check(
        config, repo, draw_ids=draw_ids, scratch=Path(scratch) / Path(config).stem,
        registered_draws_root=None if registered_draws_root is None else Path(registered_draws_root),
    )
    return {
        "config": str(config),
        "config_sha256": sha256_file(config),
        "scenario_directory": scenario.name,
        "flow_sha256": sha256_file(flow_path),
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }


def run_audit(
    *, repo_root: str | Path, draw_ids: Sequence[int] = HELD_OUT_DRAWS, scratch: str | Path,
    registered_draws_root: str | Path | None, reference_repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """The pre-flight audit of A27(h): bc-tyc (the reference, its draws checked against the main tree's) and the three
    files, the sources' digests the materialisation is pinned to, and every held-out draw's expected ``flow.json`` digest."""
    import offline.transfer_curve as tc

    repo = Path(repo_root)
    reference_repo = Path(reference_repo_root) if reference_repo_root is not None else repo
    reference_config = reference_repo / REFERENCE_FILE["config"]
    files: dict[str, Any] = {
        REFERENCE_FILE["name"]: audit_file(
            reference_config, reference_config=reference_config, repo_root=reference_repo, draw_ids=draw_ids,
            scratch=Path(scratch), registered_draws_root=registered_draws_root,
        )
    }
    for name, spec in SHIFTED_FILES.items():
        record = audit_file(
            repo / spec["config"], reference_config=reference_config, repo_root=repo, draw_ids=draw_ids,
            scratch=Path(scratch), reference_repo_root=reference_repo,
        )
        tag_ok = spec["directory_tag"] in record["scenario_directory"]
        record["checks"]["recording"] = _check(tag_ok, recorded=spec["recorded"], directory=record["scenario_directory"])
        record["passed"] = record["passed"] and tag_ok
        files[name] = record
    return {
        "format_version": AUDIT_FORMAT_VERSION,
        "label": "exploration, not H2",
        "what": (
            "bc-tyc's flow audit applied to kn-hz, qc-yn and sb-sx (never audited before), written BEFORE any of their "
            "draws is materialised (A27(h)); bc-tyc audited the same way as the reference"
        ),
        "horizon_seconds": HORIZON_SECONDS,
        "draw_ids": [int(draw) for draw in draw_ids],
        "files": files,
        "sources": {
            spec["scenario_key"]: {"config": spec["config"], "config_sha256": files[name]["config_sha256"],
                                   "flow_sha256": files[name]["flow_sha256"]}
            for name, spec in SHIFTED_FILES.items()
        },
        "expected_flow_sha256": {
            spec["scenario_key"]: files[name]["checks"]["materialised_draws"].get("flow_sha256", {})
            for name, spec in SHIFTED_FILES.items()
        },
        "passed": all(record["passed"] for record in files.values()),
        **tc._git_provenance(),
    }


def materialise_shifted(
    *, out_root: str | Path, audit_path: str | Path, repo_root: str | Path, draw_ids: Sequence[int] = HELD_OUT_DRAWS,
    registered_root: str | Path = REGISTERED_DRAWS_ROOT, main_tree: str | Path = MAIN_TREE,
) -> dict[str, Any]:
    """The three files' held-out draws into the MAIN tree's draws root, after the audit; every refusal precedes any write.

    Then each ``flow.json`` is compared with the audit's expected digest.  A second call keeps every draw
    (``materialise_draws``' idempotence).  ``main_tree`` and ``registered_root`` exist so a test can run in a stand-in
    tree; the command line passes neither.
    """
    from offline.materialise_draws import materialise

    root = Path(os.path.abspath(out_root))
    if root != Path(os.path.abspath(registered_root)):
        raise ValueError(
            f"{root} is not the registered draws root {registered_root}; the draws are materialised into the main tree "
            "and nowhere else (DEFERRED 55)"
        )
    if Path.cwd().resolve() != Path(main_tree).resolve():
        raise ValueError(
            f"the working directory is {Path.cwd()}, not the main tree {main_tree}; materialise_draws embeds the "
            "scenario dir resolved against the cwd in every cityflow.json, so the draws must be made from the main tree"
        )
    path = Path(audit_path)
    if not path.is_file():
        raise FileNotFoundError(f"no audit at {path}; the audit is this task's pre-flight and precedes any draw")
    audit = json.loads(path.read_bytes())
    if audit.get("format_version") != AUDIT_FORMAT_VERSION or audit.get("passed") is not True:
        raise ValueError(f"the audit at {path} did not pass (or is not {AUDIT_FORMAT_VERSION}); no draw is made")
    for spec in SHIFTED_FILES.values():
        recorded = audit["sources"][spec["scenario_key"]]
        config = Path(repo_root) / spec["config"]
        now = {"config_sha256": sha256_file(config), "flow_sha256": source_flow_sha256(config, repo_root)}
        if (recorded["config_sha256"], recorded["flow_sha256"]) != (now["config_sha256"], now["flow_sha256"]):
            raise ValueError(f"the audit is stale for {spec['scenario_key']}: its source digests are not the files' now")
    expected = audit["expected_flow_sha256"]
    uncovered = [
        (spec["scenario_key"], draw) for spec in SHIFTED_FILES.values() for draw in draw_ids
        if str(draw) not in expected.get(spec["scenario_key"], {})
    ]
    if uncovered:
        raise ValueError(f"the audit records no expected digest for {uncovered[:3]}; it does not cover these draws")

    summary: dict[str, Any] = {"n_written": 0, "n_kept": 0, "n_flow_digests_equal": 0, "files": {}}
    for spec in SHIFTED_FILES.values():
        records = materialise(spec["config"], [int(draw) for draw in draw_ids], out_root=root)
        actions = [record.action for record in records]
        equal = sum(
            1 for record in records
            if sha256_file(record.flow_path) == expected[spec["scenario_key"]][str(record.draw_id)]
        )
        if equal != len(records):
            raise ValueError(
                f"{spec['scenario_key']}: {len(records) - equal} materialised flow.json differ from the audit's expected "
                "digests; the pipeline and its audit disagree -- nothing is deleted, and this is a finding"
            )
        summary["n_written"] += actions.count("written")
        summary["n_kept"] += actions.count("kept")
        summary["n_flow_digests_equal"] += equal
        summary["files"][spec["scenario_key"]] = {"written": actions.count("written"), "kept": actions.count("kept")}
    return summary
