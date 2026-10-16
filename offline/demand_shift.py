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
    "ARTIFACT_NAME",
    "CELL_FORMAT_VERSION",
    "GUARD_ARMS",
    "GUARD_FIELDS",
    "LABEL",
    "WHAT_THIS_IS_NOT",
    "assert_guard_authorises",
    "build_report",
    "cell_name",
    "cell_record",
    "declared_cells",
    "declared_engine_seed_cells",
    "declared_guard_cells",
    "engine_seed_check",
    "evaluate_cells",
    "file_block",
    "main",
    "p8_4b_committed",
    "random_fresh_factory",
    "roll_cell",
    "run_guard",
    "status",
    "write_report",
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


# ----------------------------------------------------------------------
# The evaluation and the artifact (C6)
# ----------------------------------------------------------------------

#: A27(a), item 7.
LABEL = "exploration, not H2"

#: A27(h)'s three sentences and A25(a)'s amended sentence, VERBATIM (the registration's emphasis removed; the em dash U+2014
#: and the multiplication sign U+00D7 are the literal characters, as in the registration).
#: ``tests/test_demand_shift_evaluate.py`` finds each in ``PREREGISTRATION.md``.
WHAT_THIS_IS_NOT: tuple[str, ...] = (
    "It is NOT H2 — H2 is a registered 2×2 with shift-augmented training that stays not tested under A25(a)",
    "this evaluation trains nothing",
    "it is NOT the registered Hangzhou transfer pairs, which are CityFlow-to-SUMO; the paper must not present it as such.",
    "H2's registered 2×2 design remains untested; the zero-shot demand-shift evaluation in Section X was added after "
    "the results, trains nothing and is exploratory, so it is not a test of H2.",
)

CELL_FORMAT_VERSION = "p9.2-demand-shift-cell/1.0"
ARTIFACT_NAME = "p9_2_demand_shift.json"
ARTIFACT_FORMAT_VERSION = "p9.2-demand-shift/1.0"
GUARD_NAME = "guard.json"
GUARD_FORMAT_VERSION = "p9.2-demand-shift-guard/1.0"

#: The fields a guard cell must reproduce under ``==`` against P8.4b's committed cell (A11(b)'s five and the two the
#: horizon reader returns with them).
GUARD_FIELDS: tuple[str, ...] = (
    "att_ours", "att_engine", "created", "entered", "never_entered", "horizon_vehicle_count", "episode_reward",
)
DEFINITIONS: tuple[str, ...] = ("att_ours", "att_engine")

#: Stage 1's learner arms: the ``mappo1000`` and ``mix50`` checkpoints of A27(h), resolved by P8.4b's dispatch
#: (``att_rederivation.build_cell_factory``), so every checkpoint and prompt is the registered one.
LEARNER_ARMS: tuple[str, ...] = (
    "dt@mappo1000", "bc@mappo1000", "bc_top10@mappo1000", "iql@mappo1000",
    "dt@mix50", "bc@mix50", "bc_top10@mix50", "iql@mix50",
)
#: The references: fixed-time at bc-tyc's tuned k = 6 (its tier manifest; NOT re-tuned on the shifted files),
#: MaxPressure, the registered five-schedule random reference and A27(d)'s ``random_fresh``.
REFERENCE_ARMS: tuple[str, ...] = ("behaviour@fixedtime", "behaviour@maxpressure", "behaviour@random", "random_fresh")
DETERMINISTIC_ARMS: frozenset[str] = frozenset({"behaviour@fixedtime", "behaviour@maxpressure"})
STAGE_2_ARMS: tuple[str, ...] = ("iql_t@mappo1000", "iql_t@mix50")
SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)
ENGINE_SEED = 1000
ENGINE_SEED_CHECK_SEEDS: tuple[int, ...] = (1000, 1001, 1002)
DECLARED_GRADIENT_STEPS = 40_000


def _arms(stage: int) -> tuple[str, ...]:
    if stage == 1:
        return (*LEARNER_ARMS, *REFERENCE_ARMS)
    if stage == 2:
        return STAGE_2_ARMS
    raise ValueError(f"stage {stage!r} is neither 1 nor 2")


def declared_cells(*, stage: int) -> list[dict[str, Any]]:
    """Every cell of one stage: file x arm x seed x draw (the two deterministic references one slot, ``seed: None``)."""
    cells: list[dict[str, Any]] = []
    for name, spec in SHIFTED_FILES.items():
        for arm in _arms(stage):
            for seed in ((None,) if arm in DETERMINISTIC_ARMS else SEEDS):
                for draw in HELD_OUT_DRAWS:
                    cells.append({"file": name, "scenario_key": spec["scenario_key"], "arm": arm, "seed": seed,
                                  "draw_id": int(draw), "stage": stage})
    return cells


def cell_name(cell: Mapping[str, Any]) -> str:
    arm = str(cell["arm"]).replace("@", "_at_")
    seed = "none" if cell["seed"] is None else int(cell["seed"])
    return f"cell_{cell['file']}_{arm}_seed{seed}_draw{int(cell['draw_id']):04d}.json"


def random_fresh_factory(seed: int, draw_id: int) -> Any:
    """A27(d)'s ``random_fresh`` reference: ``collect._make_random``'s sampler driven by ``default_rng([seed, draw_id])``,
    one fresh generator per (seed, draw) -- built INSIDE the factory, so each route's episode starts from the same state.
    Every legal-action mask on these networks is all-true, so the schedule depends only on (seed, draw)."""
    from types import SimpleNamespace

    import numpy as np

    from offline.collect import _make_random

    def factory(env: Any) -> Any:
        policy = _make_random(env, SimpleNamespace(), np.random.default_rng([int(seed), int(draw_id)]))
        return lambda _env, info: policy(info)

    return factory


def cell_record(cell: Mapping[str, Any], p4_6: Any, p8_4b: Any, *, source: Mapping[str, Any]) -> dict[str, Any]:
    """One cell's record from its two routes, refused unless they ran the same episode: ``att_ours`` (the P4.6 route's
    ``att_horizon``), the vehicle count, the return and the action digest equal under ``==``."""
    pairs = {
        "att_ours": (p4_6.att_horizon, p8_4b.att_ours),
        "horizon_vehicle_count": (p4_6.horizon_vehicle_count, p8_4b.horizon_vehicle_count),
        "episode_reward": (p4_6.episode_reward, p8_4b.episode_reward),
        "action_sequence_sha256": (p4_6.action_sequence_sha256, p8_4b.action_sequence_sha256),
    }
    split = sorted(name for name, (left, right) in pairs.items() if left != right)
    if split:
        raise ValueError(
            f"{cell_name(cell)}: the two routes (P4.6's evaluate_arm, P8.4b's probe_episode) disagree on {split}; they "
            "roll one episode, and a disagreement stops the campaign"
        )
    return {
        "format_version": CELL_FORMAT_VERSION,
        **{key: cell[key] for key in ("file", "scenario_key", "arm", "seed", "draw_id", "stage")},
        "att_ours": float(p8_4b.att_ours),
        "att_ours_p4_6_route": float(p4_6.att_horizon),
        "att_engine": float(p8_4b.att_engine),
        "created": int(p8_4b.created),
        "entered": int(p8_4b.entered),
        "never_entered": int(p8_4b.never_entered),
        "entered_fraction": float(p8_4b.entered_fraction),
        "horizon_vehicle_count": float(p8_4b.horizon_vehicle_count),
        "episode_reward": float(p8_4b.episode_reward),
        "action_sequence_sha256": str(p8_4b.action_sequence_sha256),
        "seconds": float(p8_4b.seconds),
        "seconds_rollout": float(p8_4b.seconds_rollout),
        "policy_source": dict(source),
        "checkpoint_sha256": source.get("checkpoint_sha256"),
        **_git_provenance(),
    }


def _recording(factory: Any, sink: list[Any]) -> Any:
    import numpy as np

    def wrapped(env: Any) -> Any:
        choose = factory(env)

        def recorded(env_: Any, info: dict[str, Any]) -> Any:
            action = choose(env_, info)
            sink.append(np.asarray(action, dtype=np.int64).reshape(-1).copy())
            return action

        return recorded

    return wrapped


_CHECKPOINT_SHA: dict[str, str] = {}


def _factory_for(cell: Mapping[str, Any], *, roots: Any, device: str | None, config: Path,
                 iql_t_pins: Mapping[str, Any] | None) -> tuple[Any, dict[str, Any], str, str]:
    """``(factory, source, method, tier)`` for a cell, every learned arm through P8.4b's own dispatch."""
    from offline.att_rederivation import build_cell_factory, normalise_arm
    from offline.offline_baselines import _baseline_factory

    arm = str(cell["arm"])
    if arm == "random_fresh":
        return (random_fresh_factory(int(cell["seed"]), int(cell["draw_id"])),
                {"kind": "algorithmic", "detail": "offline.collect._make_random driven by numpy.random.default_rng("
                                                  "[seed, draw_id]) (A27(d)'s random_fresh)"},
                "random_fresh", "random")
    method, tier = normalise_arm("hz1x1", arm)
    if method == "iql_t":
        if iql_t_pins is None:
            raise ValueError("stage 2 (iql_t) is pending: no checkpoint pins were given (P9.1's M2 provides them)")
        pin = iql_t_pins[tier][str(int(cell["seed"]))]
        path = Path(pin["path"])
        digest = sha256_file(path)
        if digest != pin["sha256"]:
            raise ValueError(f"{path}: sha256 {digest} is not the pinned {pin['sha256']}")
        return (_baseline_factory("iql", str(path), DECLARED_GRADIENT_STEPS, device),
                {"kind": "checkpoint", "detail": "agent IQL via offline_baselines._baseline_factory (P9.1's iql_t)",
                 "checkpoint": str(path), "checkpoint_sha256": digest}, method, tier)
    seed = None if cell["seed"] is None else int(cell["seed"])
    factory, source = build_cell_factory("hz1x1", arm, seed, roots, device=device, config_path=config)
    source = dict(source)
    checkpoint = source.get("checkpoint")
    if checkpoint and not source.get("checkpoint_sha256"):
        if checkpoint not in _CHECKPOINT_SHA:
            _CHECKPOINT_SHA[checkpoint] = sha256_file(checkpoint)
        source["checkpoint_sha256"] = _CHECKPOINT_SHA[checkpoint]
    return factory, source, method, tier


def roll_cell(cell: Mapping[str, Any], *, roots: Any, device: str | None, engine_seed: int = ENGINE_SEED,
              iql_t_pins: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One cell by BOTH routes P9.1 uses -- ``dt_gate.evaluate_arm`` (the P4.6 route, ``att_ours``) and
    ``admission_probe.probe_episode`` (P8.4b's route, ``att_engine`` and A11(b)'s companions) -- on the cell's materialised
    draw, the env settings of the arm's own tier (identical across hz1x1's tiers), engine seed 1000.  It writes NOTHING."""
    from types import SimpleNamespace

    from offline.admission_probe import created_from_flow, probe_episode
    from offline.att_rederivation import rederivation_env_settings
    from offline.dt_gate import evaluate_arm
    from offline.materialise_draws import draw_config_path
    from offline.nortg_decomposition import action_sequence_sha256

    draw = int(cell["draw_id"])
    config = Path(draw_config_path(str(cell["scenario_key"]), draw, out_root=roots.draws_root))
    factory, source, method, tier = _factory_for(cell, roots=roots, device=device, config=config, iql_t_pins=iql_t_pins)
    settings = rederivation_env_settings("hz1x1", tier, roots)
    horizon = int(settings["max_steps"]) * int(settings["delta_time"])
    created = created_from_flow(config.parent / "flow.json", horizon_seconds=horizon)
    first: list[Any] = []
    result = evaluate_arm(
        arm=str(cell["arm"]), seed=cell["seed"], draw_ids=[draw], config_for_draw=lambda _draw: config,
        env_settings=settings, scenario_id=str(cell["scenario_key"]), choose_action_factory=_recording(factory, first),
        engine_seed=int(engine_seed),
    )[0]
    second: list[Any] = []
    episode = probe_episode(
        scenario="hz1x1", tier=tier, method=method, arm=str(cell["arm"]), seed=cell["seed"], draw_id=draw,
        config_path=config, env_settings=settings, scenario_id=str(cell["scenario_key"]),
        choose_action_factory=_recording(factory, second), engine_seed=int(engine_seed), created=created,
    )
    p4_6 = SimpleNamespace(att_horizon=result.att_horizon, horizon_vehicle_count=result.horizon_vehicle_count,
                           episode_reward=result.episode_reward, action_sequence_sha256=action_sequence_sha256(first))
    p8_4b = SimpleNamespace(**episode.as_record(), action_sequence_sha256=action_sequence_sha256(second))
    record = cell_record(cell, p4_6, p8_4b, source=source)
    record["engine_seed"] = int(engine_seed)
    record["config_sha256"] = sha256_file(config)
    record["flow_sha256"] = sha256_file(config.parent / "flow.json")
    return record


def p8_4b_committed(cell: Mapping[str, Any], *, output_root: str | Path) -> dict[str, Any]:
    """P8.4b's committed re-derivation cell for a bc-tyc cell (``att_rederivation.cell_file_name``), refused unless it
    reproduced its own committed ``att_ours``."""
    from offline.att_rederivation import CellKey, cell_file_name

    key = CellKey(scenario="hz1x1", arm=str(cell["arm"]),
                  seed=None if cell["seed"] is None else int(cell["seed"]), draw_id=int(cell["draw_id"]))
    path = Path(output_root) / "p8_4b_rederivation" / cell_file_name(key)
    record = json.loads(path.read_bytes())
    if record.get("reproduces_committed") is not True:
        raise ValueError(f"{path.name} did not reproduce its committed att_ours; it cannot be a guard's reference")
    return {**record, "source": path.name, "source_sha256": sha256_file(path)}


def engine_seed_check(cell: Mapping[str, Any], *, roots: Any, device: str | None) -> dict[str, Any]:
    """P0.2's finding re-measured on one cell: the same cell at engine seeds 1000, 1001, 1002 gives one ``att_ours`` and one
    ``att_engine`` -- CityFlow reads its flow once and the engine seed changes nothing observable on this network, which is
    why the training seeds are the only seeds."""
    records = [roll_cell(cell, roots=roots, device=device, engine_seed=seed) for seed in ENGINE_SEED_CHECK_SEEDS]
    ours = [r["att_ours"] for r in records]
    engine = [r["att_engine"] for r in records]
    return {
        "cell": cell_name(cell),
        "engine_seeds": list(ENGINE_SEED_CHECK_SEEDS),
        "att_ours": ours,
        "att_engine": engine,
        "passed": len(set(ours)) == 1 and len(set(engine)) == 1,
    }


#: The guard's arms: every stage-1 learner arm and the three registered references, each on bc-tyc's draw 1000 at seed 101
#: (the two deterministic references in their single slot).  ``random_fresh`` has no committed counterpart: its digest is
#: recorded beside the checks, and is not one.
GUARD_ARMS: tuple[str, ...] = (*LEARNER_ARMS, "behaviour@fixedtime", "behaviour@maxpressure", "behaviour@random")
GUARD_SEED = 101
GUARD_DRAW = 1000


def declared_guard_cells() -> list[dict[str, Any]]:
    """The guard's bc-tyc cells, in the order :func:`run_guard` rolls them."""
    return [
        {"file": REFERENCE_FILE["name"], "scenario_key": REFERENCE_FILE["scenario_key"], "arm": arm,
         "seed": None if arm in DETERMINISTIC_ARMS else GUARD_SEED, "draw_id": GUARD_DRAW, "stage": 0}
        for arm in GUARD_ARMS
    ]


def declared_engine_seed_cells() -> list[dict[str, Any]]:
    """The engine-seed check's cells: ``behaviour@maxpressure`` on draw 1000 of bc-tyc and of each shifted file."""
    files = [(REFERENCE_FILE["name"], REFERENCE_FILE["scenario_key"])] + [
        (name, spec["scenario_key"]) for name, spec in SHIFTED_FILES.items()
    ]
    return [{"file": name, "scenario_key": key, "arm": "behaviour@maxpressure", "seed": None, "draw_id": GUARD_DRAW,
             "stage": 0} for name, key in files]


def assert_guard_authorises(record: Mapping[str, Any], *, source: str) -> None:
    """Refuse (``ValueError``, the reason named) unless the guard record read from *source* authorises an evaluation or a
    report (``BRIEF_47`` Amendment A.2.2): its format version; ``passed`` true; ``git_dirty`` false; ``git_commit`` a commit
    whose code equals HEAD outside ``docs/`` (J1(c)'s rule, :func:`_code_changed_since`); every declared arm
    (:func:`declared_guard_cells`) checked exactly once, at its declared seed and draw, passed with no differing field; and
    the engine-seed check passed on exactly its declared cells (:func:`declared_engine_seed_cells`), each at engine seeds
    1000, 1001, 1002.  A record that says ``passed`` proves nothing about THIS code unless all of that holds."""
    label = f"the guard record at {source}"
    if record.get("format_version") != GUARD_FORMAT_VERSION:
        raise ValueError(f"{label}: format_version {record.get('format_version')!r} is not {GUARD_FORMAT_VERSION!r}")
    if record.get("passed") is not True:
        raise ValueError(f"the guard at {source} did not pass; no shifted cell may be evaluated or reported")
    if record.get("git_dirty") is not False:
        raise ValueError(
            f"{label}: git_dirty is {record.get('git_dirty')!r}, not False; a guard run from a tree with uncommitted "
            "changes cannot be attributed to the commit it names"
        )
    commit = record.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError(f"{label}: git_commit {commit!r} is not a 40-character hex commit")
    changed = _code_changed_since(commit)
    if changed:
        raise ValueError(
            f"{label}: the guard ran at {commit}, which differs from HEAD outside docs/ ({changed[:3]}); it proves the "
            "evaluation path of other code, not of this one"
        )
    checks = record.get("cells")
    if not isinstance(checks, list):
        raise ValueError(f"{label}: cells is {type(checks).__name__}, not a list of checks")
    declared = {str(cell["arm"]): cell for cell in declared_guard_cells()}
    seen: set[str] = set()
    for check in checks:
        if not isinstance(check, Mapping):
            raise ValueError(f"{label}: a check is {type(check).__name__}, not an object")
        arm = check.get("arm")
        if arm not in declared:
            raise ValueError(f"{label}: {arm!r} is not a declared guard arm ({list(declared)})")
        if arm in seen:
            raise ValueError(f"{label}: the arm {arm} is checked twice")
        seen.add(str(arm))
        if check.get("seed") != declared[arm]["seed"]:
            raise ValueError(f"{label}: {arm} was checked at seed {check.get('seed')!r}, not the declared "
                             f"{declared[arm]['seed']!r}")
        if check.get("draw_id") != declared[arm]["draw_id"]:
            raise ValueError(f"{label}: {arm} was checked on draw {check.get('draw_id')!r}, not the declared {GUARD_DRAW}")
        if check.get("passed") is not True or check.get("differing") != []:
            raise ValueError(
                f"{label}: {arm} did not reproduce P8.4b's committed cell (passed {check.get('passed')!r}, differing "
                f"{check.get('differing')!r})"
            )
    absent = [arm for arm in declared if arm not in seen]
    if absent:
        raise ValueError(f"{label}: it lacks the declared guard arm(s) {absent}")
    engine = record.get("engine_seed_check")
    if not isinstance(engine, Mapping) or engine.get("passed") is not True:
        raise ValueError(f"{label}: the engine-seed check did not pass")
    expected = [cell_name(cell) for cell in declared_engine_seed_cells()]
    seed_checks = engine.get("checks")
    if not isinstance(seed_checks, list) or not all(isinstance(c, Mapping) for c in seed_checks):
        raise ValueError(f"{label}: the engine-seed check carries no list of checks")
    names = [c.get("cell") for c in seed_checks]
    if len(names) != len(set(names)) or sorted(map(str, names)) != sorted(expected):
        raise ValueError(f"{label}: the engine-seed check covers {names}, not the declared {expected}")
    for check in seed_checks:
        if check.get("engine_seeds") != list(ENGINE_SEED_CHECK_SEEDS):
            raise ValueError(f"{label}: the engine-seed check of {check['cell']} ran engine seeds "
                             f"{check.get('engine_seeds')!r}, not {list(ENGINE_SEED_CHECK_SEEDS)}")
        if check.get("passed") is not True:
            raise ValueError(f"{label}: the engine-seed check of {check['cell']} did not pass")


def _guard_refusal(guard_path: Path) -> str | None:
    """``None`` when the guard record at *guard_path* authorises, else why not -- missing, unreadable or refused.  Fails
    closed: anything that stops the record being verified is a refusal, named with its type."""
    if not guard_path.is_file():
        return f"no guard record at {guard_path}; run the guard subcommand first"
    try:
        record = json.loads(guard_path.read_bytes())
        if not isinstance(record, dict):
            return f"the guard record at {guard_path} is not a JSON object"
        assert_guard_authorises(record, source=str(guard_path))
    except Exception as exc:  # noqa: BLE001 -- fail closed, the reason is printed
        return f"{type(exc).__name__}: {exc}"
    return None


def run_guard(*, roots: Any, device: str | None) -> dict[str, Any]:
    """THE TAUTOLOGY GUARD, before any shifted cell: one bc-tyc held-out cell per stage-1 arm (seed 101, or the
    deterministic slot; draw 1000) through :func:`roll_cell`, every :data:`GUARD_FIELDS` ``==`` P8.4b's committed cell;
    ``random_fresh`` on (101, 1000) rolled and its digest recorded (no committed counterpart exists); the engine-seed check
    on bc-tyc and on each shifted file."""
    checks: list[dict[str, Any]] = []
    for cell in declared_guard_cells():
        arm = cell["arm"]
        record = roll_cell(cell, roots=roots, device=device)
        committed = p8_4b_committed(cell, output_root=roots.output_root)
        differing = sorted(field for field in GUARD_FIELDS if record[field] != committed[field])
        checks.append({"arm": arm, "seed": cell["seed"], "draw_id": int(cell["draw_id"]), "differing": differing,
                       "passed": not differing, "committed_source": committed["source"],
                       "committed_sha256": committed["source_sha256"], "record": record})
    fresh_cell = {"file": REFERENCE_FILE["name"], "scenario_key": REFERENCE_FILE["scenario_key"], "arm": "random_fresh",
                  "seed": 101, "draw_id": 1000, "stage": 0}
    fresh = roll_cell(fresh_cell, roots=roots, device=device)
    seed_checks = [engine_seed_check(cell, roots=roots, device=device) for cell in declared_engine_seed_cells()]
    return {
        "format_version": GUARD_FORMAT_VERSION,
        "what": "bc-tyc held-out cells reproduced under == through the evaluation path before any shifted file is "
                "evaluated (the tautology guard), and P0.2's engine-seed finding re-measured on each file",
        "cells": checks,
        "random_fresh_digest_bc_tyc_seed101_draw1000": fresh["action_sequence_sha256"],
        "engine_seed_check": {"checks": seed_checks, "passed": all(c["passed"] for c in seed_checks)},
        "passed": all(c["passed"] for c in checks) and all(c["passed"] for c in seed_checks),
        **_git_provenance(),
    }


def _cityflow_rho(row: Mapping[str, Any], anchors: Mapping[str, Mapping[str, Any]], key: str) -> float | None:
    """rho on one definition against the file's own anchors of the SAME draw (``transfer_curve.rho``); an exactly-zero
    denominator is recorded and excluded on BOTH definitions (plan C7: no definition is primary here)."""
    import offline.transfer_curve as tc

    ft, mp = float(anchors["behaviour@fixedtime"][key]), float(anchors["behaviour@maxpressure"][key])
    if ft - mp == 0.0:
        return None
    return tc.rho(float(row[key]), ft, mp)


def file_block(records: Sequence[Mapping[str, Any]], *, arms: Sequence[str]) -> dict[str, Any]:
    """One file's estimators, both definitions side by side: per arm the raw ATT and rho (each the mean over draws of the
    per-draw seed means, ``dt_gate.mean_ci95``), and per learner the paired contrasts against fixed-time and BOTH random
    references (``transfer_curve._paired_block``: per-draw seed means, Wilcoxon beside the interval)."""
    import offline.transfer_curve as tc
    from offline.tier_statements import interval_outcome

    anchors: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in records:
        if row["arm"] in DETERMINISTIC_ARMS:
            anchors.setdefault(int(row["draw_id"]), {})[str(row["arm"])] = row
    rows: list[dict[str, Any]] = []
    for row in records:
        slot = anchors.get(int(row["draw_id"]), {})
        if set(slot) != DETERMINISTIC_ARMS:
            raise ValueError(f"draw {row['draw_id']} lacks one of the file's two anchors")
        rows.append({**row, **{f"rho_{key}": _cityflow_rho(row, slot, key) for key in DEFINITIONS}})
    excluded = {key: sorted(draw for draw, slot in anchors.items()
                            if float(slot["behaviour@fixedtime"][key]) - float(slot["behaviour@maxpressure"][key]) == 0.0)
                for key in DEFINITIONS}
    block: dict[str, Any] = {
        "anchors": {"fixedtime": "behaviour@fixedtime", "maxpressure": "behaviour@maxpressure",
                    "excluded_draws": excluded},
        "random_references": {"registered": "behaviour@random", "fresh": "random_fresh"},
        "arms": {},
    }
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if not arm_rows:
            continue
        entry: dict[str, Any] = {"att": {}, "rho": {}, "per_draw": {}}
        for key in DEFINITIONS:
            raw = {draw: value for draw, value in tc._seed_means_by_draw(arm_rows, key).items()}
            entry["att"][key] = tc._definition_stats(raw)
            series = tc._seed_means_by_draw(arm_rows, f"rho_{key}")
            entry["rho"][key] = tc._definition_stats(series)
            entry["per_draw"][key] = {str(draw): raw[draw] for draw in sorted(raw)}
        if arm not in REFERENCE_ARMS:
            entry["paired"] = {}
            for reference, predicted in (("behaviour@fixedtime", "negative"), ("behaviour@random", "negative"),
                                         ("random_fresh", "negative")):
                if not any(row["arm"] == reference for row in rows):
                    continue
                by_definition: dict[str, Any] = {}
                for key in DEFINITIONS:
                    paired = tc._paired_block(rows, arm_rows, str(arm_rows[0]["file"]), arm, reference, key)
                    paired["side"] = interval_outcome(float(paired["ci95_low"]), float(paired["ci95_high"]),
                                                      predicted=predicted)
                    by_definition[key] = paired
                entry["paired"][reference] = by_definition
        block["arms"][arm] = entry
    return block


def _code_changed_since(commit: str) -> list[str]:
    import offline.transfer_curve as tc

    return tc.code_changed_since(commit)


def _git_provenance() -> dict[str, Any]:
    import offline.transfer_curve as tc

    return tc._git_provenance()


def _load_cells(work: Path, declared: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    names = {cell_name(cell): cell for cell in declared}
    missing = sorted(name for name in names if not (work / name).is_file())
    if missing:
        raise ValueError(f"{len(missing)} declared cell(s) have no record (first: {missing[:3]}); no partial set is reported")
    records: list[dict[str, Any]] = []
    for name, cell in names.items():
        record = json.loads((work / name).read_bytes())
        if record.get("format_version") != CELL_FORMAT_VERSION:
            raise ValueError(f"{name}: format_version {record.get('format_version')!r} is not {CELL_FORMAT_VERSION!r}")
        for key in ("file", "scenario_key", "arm", "seed", "draw_id", "stage"):
            if record.get(key) != cell.get(key):
                raise ValueError(f"{name}: the record says {key}={record.get(key)!r}; it describes another cell")
        if record["att_ours"] != record["att_ours_p4_6_route"]:
            raise ValueError(f"{name}: the two routes disagree on att_ours; the record is refused")
        if record.get("git_dirty") is not False:
            raise ValueError(f"{name}: git_dirty is {record.get('git_dirty')!r}, not False")
        records.append(record)
    return records


def build_report(*, work_dir: str | Path, audit_path: str | Path,
                 declared: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """``docs/data/p9_2_demand_shift.json``'s content; it writes NOTHING.  Refuses, before any aggregate: a missing or failed
    guard, a missing or failed audit, a missing or invalid cell record, a record rolled by other code; stage 2 is reported
    when every one of its cells exists, pending when none does, and refused when only some do."""
    work = Path(work_dir)
    guard_path = work / GUARD_NAME
    if not guard_path.is_file():
        raise ValueError(f"no guard record at {guard_path}; the evaluation path is not proven on bc-tyc")
    guard = json.loads(guard_path.read_bytes())
    assert_guard_authorises(guard, source=str(guard_path))
    audit_file_path = Path(audit_path)
    audit = json.loads(audit_file_path.read_bytes())
    if audit.get("format_version") != AUDIT_FORMAT_VERSION or audit.get("passed") is not True:
        raise ValueError(f"the audit at {audit_file_path} did not pass")
    full = declared is None
    stage_1 = declared_cells(stage=1) if full else list(declared)
    records = _load_cells(work, stage_1)
    stage_2_cells = declared_cells(stage=2) if full else []
    present_2 = [cell for cell in stage_2_cells if (work / cell_name(cell)).is_file()]
    if present_2 and len(present_2) != len(stage_2_cells):
        raise ValueError(f"stage 2 is partial ({len(present_2)} of {len(stage_2_cells)} cells); it is reported whole or not")
    if present_2:
        records += _load_cells(work, stage_2_cells)
        stage_2 = {"status": "present", "n_cells": len(stage_2_cells)}
    else:
        stage_2 = {"status": "pending",
                   "why": "iql_t is P9.1's (A27(c)); its checkpoints are evaluated here once P9.1's M2 is on main"}
    commits = sorted({str(record["git_commit"]) for record in records})
    for commit in commits:
        changed = _code_changed_since(commit)
        if changed:
            raise ValueError(f"records were rolled at {commit}, which differs from HEAD outside docs/ ({changed[:3]})")
    arms = [*LEARNER_ARMS, *(STAGE_2_ARMS if present_2 else ()), *REFERENCE_ARMS]
    files: dict[str, Any] = {}
    for name in SHIFTED_FILES:
        file_records = [record for record in records if record["file"] == name]
        if file_records:
            files[name] = file_block(file_records, arms=arms)
            fresh = [r for r in file_records if r["arm"] == "random_fresh" and r["seed"] == 101 and r["draw_id"] == 1000]
            files[name]["random_fresh_digest_seed101_draw1000"] = fresh[0]["action_sequence_sha256"] if fresh else None
    digests = {REFERENCE_FILE["name"]: guard.get("random_fresh_digest_bc_tyc_seed101_draw1000")}
    digests.update({name: block.get("random_fresh_digest_seed101_draw1000") for name, block in files.items()})
    fresh_identity = {"digests": digests, "identical": len(set(digests.values())) == 1 and None not in digests.values()}
    checkpoints = sorted({(r["arm"], r["seed"], r.get("checkpoint_sha256")) for r in records if r.get("checkpoint_sha256")},
                         key=lambda item: (item[0], item[1]))
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "label": LABEL,
        "registered_in": "PREREGISTRATION A27(a) item 7 and A27(h); tag v2.8-prereg-a27; BRIEF_47 section 3.4",
        "what_this_is_not": list(WHAT_THIS_IS_NOT),
        "exploratory": True,
        "no_single_definition_claim": (
            "Both ATT definitions are reported side by side and no claim rests on one: A11's engine-semantics gate was "
            "passed on bc-tyc's demand only, so neither definition is primary on these files"
        ),
        "fixed_time": "behaviour@fixedtime at bc-tyc's tuned k = 6 (its tier manifest), NOT re-tuned on the shifted files",
        "prompt": "the DT runs with its registered in-domain prompt (its tier's target_rtg), no rescaling; no probe arm",
        "demand_shift": (
            "recordings of other intersections at other hours replayed on bc-tyc's road network: volume, turning mix and "
            "time profile change together, so this is a demand shift and never 'lower volume'"
        ),
        "declaration": "full" if full else "caller-supplied",
        "alignment_convention": (
            "every value is the horizon value of one episode from reset(seed=1000) on one held-out draw (A1); rho per "
            "draw against the same file's anchors of that draw; per arm the mean over draws of each draw's seed mean "
            "(dt_gate.mean_ci95, ddof 1)"
        ),
        "audit": {"sha256": sha256_file(audit_file_path), "record": audit},
        "guard": guard,
        "stage_2": stage_2,
        "files": files,
        "random_fresh_seed101_draw1000": fresh_identity,
        "inputs": {"cell_commits": commits, "n_cells": len(records),
                   "checkpoints": [{"arm": arm, "seed": seed, "sha256": digest} for arm, seed, digest in checkpoints]},
        "cells": records,
        **_git_provenance(),
    }


def write_report(artifact: Mapping[str, Any], out_path: str | Path) -> Path:
    """The ONE write; never the committed artifact from a caller-supplied (partial) declaration, never over a file."""
    out = Path(out_path)
    committed = (Path(__file__).resolve().parent.parent / "docs" / "data" / ARTIFACT_NAME).resolve()
    if artifact.get("declaration") != "full" and out.resolve() == committed:
        raise ValueError(f"{out}: the committed artifact is never written from a partial, caller-supplied declaration")
    if out.exists():
        raise FileExistsError(f"{out} exists; the artifact is written once")
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return out


# ----------------------------------------------------------------------
# The loop and the command line
# ----------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = MAIN_TREE / "output"
DEFAULT_WORK_DIR = DEFAULT_OUTPUT_ROOT / "p9_2" / "demand_shift"
DEFAULT_CORPUS_ROOT = MAIN_TREE / "datasets_v11"
DEFAULT_AUDIT_PATH = _REPO_ROOT / "docs" / "data" / AUDIT_NAME


def _validate_record(record: Mapping[str, Any], cell: Mapping[str, Any]) -> None:
    """A fresh record, checked before it is written: its identity, its two routes' agreement, its provenance."""
    label = cell_name(cell)
    if record.get("format_version") != CELL_FORMAT_VERSION:
        raise ValueError(f"{label}: format_version {record.get('format_version')!r} is not {CELL_FORMAT_VERSION!r}")
    for key in ("file", "scenario_key", "arm", "seed", "draw_id", "stage"):
        if record.get(key) != cell.get(key):
            raise ValueError(f"{label}: the record says {key}={record.get(key)!r}; it describes another cell")
    if record["att_ours"] != record["att_ours_p4_6_route"]:
        raise ValueError(f"{label}: the two routes disagree on att_ours")
    if int(record["created"]) != int(record["entered"]) + int(record["never_entered"]):
        raise ValueError(f"{label}: created is not entered + never entered")
    if record.get("git_dirty") is not False:
        raise ValueError(
            f"{label}: git_dirty is {record.get('git_dirty')!r}, not False; a cell rolled from a tree with uncommitted "
            "changes cannot be attributed to the commit it names"
        )


def evaluate_cells(cells: Sequence[Mapping[str, Any]], *, work_dir: str | Path, shard: int, of: int,
                   roll_fn: Any) -> dict[str, Any]:
    """Roll this shard's cells that have no record yet, each validated BEFORE it is written; resumable at cell grain; a
    refused cell STOPS the shard and nothing is written for it, nor any directory before the first valid record."""
    if int(of) < 1 or not 0 <= int(shard) < int(of):
        raise ValueError(f"shard {shard!r} of {of!r} is not a valid partition")
    work = Path(work_dir)
    mine = [cell for index, cell in enumerate(sorted(cells, key=cell_name)) if index % int(of) == int(shard)]
    rolled: list[str] = []
    skipped = 0
    for cell in mine:
        path = work / cell_name(cell)
        if path.is_file():
            existing = json.loads(path.read_bytes())
            _validate_record(existing, cell)
            skipped += 1
            continue
        record = roll_fn(cell)
        _validate_record(record, cell)
        work.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        rolled.append(path.name)
        if len(rolled) % 25 == 0:
            print(f"  shard {shard}/{of}: {len(rolled)} rolled, {skipped} skipped of {len(mine)}", flush=True)
    return {"rolled": len(rolled), "skipped": skipped, "rolled_names": rolled, "n_cells": len(mine)}


def status(work_dir: str | Path) -> dict[str, Any]:
    work = Path(work_dir)
    out: dict[str, Any] = {"guard_present": (work / GUARD_NAME).is_file()}
    for stage in (1, 2):
        cells = declared_cells(stage=stage)
        present = sum(1 for cell in cells if (work / cell_name(cell)).is_file())
        out[f"stage_{stage}"] = {"declared": len(cells), "present": present, "complete": present == len(cells)}
    return out


def _roots(args: Any) -> Any:
    from offline.admission_probe import ProbeRoots

    return ProbeRoots(repo_root=_REPO_ROOT, corpus_root=Path(args.corpus_root), draws_root=Path(args.draws_root),
                      output_root=Path(args.output_root), work_dir=Path(args.work_dir))


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m offline.demand_shift",
        description="P9.2: A27(h)'s exploratory zero-shot demand-shift evaluation (exploration, not H2).",
    )
    parser.add_argument("--draws-root", default=str(REGISTERED_DRAWS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--corpus-root", default=str(DEFAULT_CORPUS_ROOT))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="the pre-flight flow audit (writes the audit file once)")
    audit.add_argument("--out", default=str(DEFAULT_AUDIT_PATH))
    audit.add_argument("--scratch", required=True, help="a scratch directory for the two test materialisations")
    sub.add_parser("materialise", help="the three files' held-out draws into the main tree (run from the main tree root)")
    sub.add_parser("guard", help="bc-tyc cells reproduced through the evaluation path; the engine-seed check")
    evaluate = sub.add_parser("evaluate", help="roll this shard's cells (refuses without a passing guard)")
    evaluate.add_argument("--stage", type=int, choices=(1, 2), default=1)
    evaluate.add_argument("--shard", type=int, default=0)
    evaluate.add_argument("--of", type=int, default=1)
    evaluate.add_argument("--iql-t-pins", default=None, help="stage 2: a JSON pin file {tier: {seed: {path, sha256}}}")
    sub.add_parser("status", help="declared against present")
    sub.add_parser("check-guard", help="exit 0 iff the guard record authorises an evaluation (read-only; the drivers ask "
                                       "before consuming their token)")
    report = sub.add_parser("report", help="write docs/data/p9_2_demand_shift.json (refuses if it exists)")
    report.add_argument("--out", default=str(_REPO_ROOT / "docs" / "data" / ARTIFACT_NAME))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns 0 when done, 2 when refused."""
    from offline.offline_baselines import pin_torch_threads
    from offline.sumo_anchor_rescore import assert_work_dir

    args = build_parser().parse_args(argv)
    pin_torch_threads(int(args.torch_threads))
    work = Path(args.work_dir)
    if args.command == "status":
        print(json.dumps(status(work), indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "check-guard":
        guard_path = work / GUARD_NAME
        refusal = _guard_refusal(guard_path)
        if refusal is not None:
            print(f"REFUSED: {refusal}", flush=True)
            return 2
        print(f"the guard record at {guard_path} authorises the evaluation: passed, clean, at this code, every declared "
              "arm and engine-seed check", flush=True)
        return 0
    if args.command == "audit":
        out = Path(args.out)
        if out.exists():
            raise FileExistsError(f"{out} exists; the audit is written once, before any draw")
        record = run_audit(repo_root=_REPO_ROOT, scratch=Path(args.scratch), registered_draws_root=args.draws_root)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, out)
        print(f"audit {'PASSED' if record['passed'] else 'FAILED'}; wrote {out}", flush=True)
        return 0 if record["passed"] else 2
    if args.command == "materialise":
        summary = materialise_shifted(out_root=args.draws_root, audit_path=args.audit, repo_root=Path.cwd())
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "report":
        artifact = build_report(work_dir=work, audit_path=args.audit)
        written = write_report(artifact, args.out)
        print(f"wrote {written}", flush=True)
        return 0
    assert_work_dir(work, output_root=args.output_root, repo_root=_REPO_ROOT)
    if args.command == "guard":
        guard_path = work / GUARD_NAME
        if guard_path.exists():
            raise FileExistsError(f"{guard_path} exists; the guard record is written once")
        record = run_guard(roots=_roots(args), device=args.device)
        work.mkdir(parents=True, exist_ok=True)
        tmp = guard_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, guard_path)
        failed = [c["arm"] for c in record["cells"] if not c["passed"]]
        print(f"guard {'PASSED' if record['passed'] else 'FAILED'} (cells not reproduced: {failed}; engine-seed check "
              f"{'passed' if record['engine_seed_check']['passed'] else 'FAILED'})", flush=True)
        return 0 if record["passed"] else 2
    refusal = _guard_refusal(work / GUARD_NAME)
    if refusal is not None:
        print(f"REFUSED: {refusal}", flush=True)
        return 2
    pins = None if args.iql_t_pins is None else json.loads(Path(args.iql_t_pins).read_bytes())
    roots = _roots(args)
    result = evaluate_cells(
        declared_cells(stage=int(args.stage)), work_dir=work, shard=args.shard, of=args.of,
        roll_fn=lambda cell: roll_cell(cell, roots=roots, device=args.device, iql_t_pins=pins),
    )
    print(f"stage {args.stage} shard {args.shard}/{args.of}: rolled {result['rolled']}, skipped {result['skipped']} of "
          f"{result['n_cells']}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() by the tests
    raise SystemExit(main())
