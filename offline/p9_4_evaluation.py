"""P9.4 plan section 4 (``docs/plans/p9.4_mix50_fresh.md``): the evaluation of ``mix50_fresh`` -- CityFlow by P9.1's two
routes, SUMO by P7.3a's machinery -- and the stopping controls G1 and G2.  EXPLORATORY, ADDED AFTER THE RESULTS.

FORMATS (every record carries its version; none is new except the guard record)
--------------------------------------------------------------------------------
* route (i): ``output/p9_4_runs/eval_i/eval_mix50_fresh_dt.json`` -- ``method_tier_grid``'s ``p4.6-grid/1.0`` cell file,
  written by :func:`route_i_payload`, which is ``method_tier_grid._run_evaluate``'s body for a tier that module's tables do
  not know: the same factory (``cell_factory``), settings (``env_settings_for_tiers``), rollout (``dt_gate.evaluate_arm``),
  completeness check and payload.
* route (ii): ``output/p9_4_runs/eval_ii/`` -- P8.4b's ``p8.4b-rederivation/1.0`` per-episode files and its campaign
  manifest, written by ``att_rederivation.run_campaign`` itself.
* SUMO: ``output/p9_4/cells/`` (the 1,000 cells) and ``output/p9_4/guards/g1/`` (G1's 100) hold ``p7.3a-zero-shot/1.0``
  chunks, written by ``transfer_curve.run_cell``; ``output/p9_4/guards/g2/`` holds G2's 40 ``p9.2-anchor-cell/1.0`` cells,
  written by ``sumo_anchor_rescore.sumo_anchor_episode``; ``output/p9_4/guards/guards.json`` is ``p9.4-guards/1.0``: each
  guard's comparison under ``==`` with the committed values, and whether it passed.

ALIGNMENT CONVENTION
--------------------
Every value is the horizon value of ONE episode from ``reset(seed=1000)`` on one held-out draw (A18(c)).  A SUMO chunk's
per-decision series follow ``transfer_curve``'s convention (``rtg_series[t]`` is read BEFORE the agent acts on
``info_t``); a route (i) episode's ``att_horizon`` and route (ii)'s ``att_ours`` are one quantity by two routes and must
agree under ``==``; ``att_engine`` and A11(b)'s companions are route (ii)'s, joined to route (i) on ``(seed, draw)``.

NOTHING IS OVERWRITTEN OR MOVED
-------------------------------
Every file is created once (``p9_4_mix50_fresh.write_once``).  A chunk that exists and validates against its cell is
skipped on resume; one that exists and does NOT validate stops the stage and is left exactly where it is (plan section
7) -- unlike ``transfer_curve.run_stage``, which moves such a chunk aside and so is not used here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import offline.p9_4_mix50_fresh as p94

GUARDS_FORMAT_VERSION = "p9.4-guards/1.0"

#: The CityFlow arm, as every route (i) and route (ii) record carries it.
CITYFLOW_ARM = f"dt@{p94.TIER}"
CITYFLOW_SCENARIO = "hz1x1"
SCENARIO_KEY = "cityflow1x1"

#: The SUMO stages, part of every chunk's identity (``transfer_curve.validate_cell_payload`` compares them).
STAGE_NEW = "p9_4"
STAGE_G1 = "p9_4_g1"


# ----------------------------------------------------------------------
# The checkpoints: their identity at consumption, and their facts
# ----------------------------------------------------------------------


def new_checkpoint_identity(roots: p94.Roots, seed: int, *, steps: int = p94.GRADIENT_STEPS) -> dict[str, Any]:
    """``mix50_fresh``'s checkpoint of *seed*: hashed HERE, at consumption, and checked against its write-once run
    record -- the shape ``transfer_curve.checkpoint_identity`` returns, so a chunk names what it was checked against."""
    run = p94.Run(p94.NEW_GROUP, p94.TIER, int(seed))
    if p94.run_status(roots, run, steps=steps) != "done":
        raise ValueError(f"{run.name} is not trained; nothing to evaluate")
    # run_status has just hashed the file and compared it, and its canonical digest, with the record; the digest the
    # chunk records is computed here, at consumption, from the same file.
    path = p94.checkpoint_path(roots, run)
    digest = p94.file_sha256(path)
    relative = p94.run_record_path(roots, run).relative_to(Path(roots.output_root))
    return {"subject": p94.SUBJECT, "seed": int(seed), "path": str(path), "file_sha256": digest,
            "sha256_checked_against": [f"output/{relative.as_posix()}"], "local_manifest": None, "deferred_56": False}


def facts_from_checkpoints(subject: str, paths: Sequence[str | Path], *, steps: int = p94.GRADIENT_STEPS) -> Any:
    """``transfer_calibration.subject_facts``'s reading over *paths* (that function knows A17(c)'s two subjects only):
    the five payloads' prompt constants must agree, the budget must be *steps*, and the support range is the split's
    ``stats.rtg`` range, checked against ``rtg_calibration.training_rtg_range`` as ``subject_facts`` checks it.  For
    mix50's five checkpoints it IS ``subject_facts("mix50")`` (``tests/test_p9_4_evaluation.py``)."""
    from offline.rtg_calibration import training_rtg_range
    from offline.transfer_calibration import SubjectFacts, _load_payload

    constants: list[tuple[Any, ...]] = []
    for path in paths:
        payload = _load_payload(path)
        pairs = [(s, i, v) for s, per_ix in payload["stats"]["rtg"].items() for i, v in per_ix.items()]
        if len(pairs) != 1:
            raise ValueError(f"{path}: the statistics cover {len(pairs)} (scenario, intersection) pairs, not one")
        summary = pairs[0][2]
        constants.append((
            float(payload["target_rtg"]), float(payload["rtg_scale"]), int(payload["config"]["state_dim"]),
            int(payload["config"]["context_length"]), int(payload["provenance"]["gradient_steps"]),
            float(summary["min"]), float(summary["max"]), int(summary["count"]),
        ))
    if len(set(constants)) != 1:
        raise ValueError(f"{subject}: the seeds disagree on the prompt constants {sorted(set(constants))}")
    target, scale, state_dim, context_length, recorded_steps, low, high, count = constants[0]
    if recorded_steps != int(steps):
        raise ValueError(f"{subject}: the checkpoints record {recorded_steps} gradient steps, not {int(steps)}")
    if training_rtg_range(paths[0]) != (low, high):
        raise ValueError(f"{subject}: training_rtg_range disagrees with the payload's stats.rtg range")
    return SubjectFacts(subject=subject, best_source_return=target, rtg_scale=scale,
                        support_range_over_the_split=(low, high), n_rows=count, training_set_return_min=-scale,
                        checkpoints=tuple(str(path) for path in paths), state_dim=state_dim, context_length=context_length)


def new_subject_facts(roots: p94.Roots, declaration: Mapping[str, Any], *, steps: int = p94.GRADIENT_STEPS) -> Any:
    """``mix50_fresh``'s facts from its five checkpoints, refused unless they are the declared ones: the source target,
    the declared return scale, support range and row count."""
    paths = [p94.checkpoint_path(roots, run) for run in p94.new_runs()]
    facts = facts_from_checkpoints(p94.SUBJECT, paths, steps=steps)
    declared = (p94.SOURCE_TARGET, float(declaration["rtg_scale"]),
                tuple(float(v) for v in declaration["normalisation"]["support_range_over_the_split"]),
                int(declaration["normalisation"]["n_rows"]))
    found = (facts.best_source_return, facts.rtg_scale, tuple(facts.support_range_over_the_split), facts.n_rows)
    if found != declared:
        raise ValueError(f"{p94.SUBJECT}: the checkpoints' facts {found} are not the declared {declared}")
    return facts


# ----------------------------------------------------------------------
# CityFlow route (i): method_tier_grid._run_evaluate's body for mix50_fresh
# ----------------------------------------------------------------------


def route_i_output(roots: p94.Roots) -> Path:
    return Path(roots.runs_root) / "eval_i" / f"eval_{p94.TIER}_dt.json"


def evaluate_route_i(roots: p94.Roots, spec: Any, checkpoints: Mapping[int, str | Path], *, target_rtg: float,
                     draws: Sequence[int], steps: int) -> list[Any]:
    """``method_tier_grid._run_evaluate``'s loop for any hz1x1 DT tier: per seed, ``cell_factory("dt", ...)`` at
    *target_rtg* and ``dt_gate.evaluate_arm`` over *draws* with the tier's own env settings
    (``env_settings_for_tiers``, asserted identical across its directories), one torch thread, the arm
    ``dt@<tier>``.  Returns the episodes; the caller checks completeness."""
    import offline.method_tier_grid as grid
    from offline.dt_gate import EpisodeResult, evaluate_arm
    from offline.materialise_draws import draw_config_path
    from offline.offline_baselines import pin_torch_threads

    arm = f"dt@{spec.tier}"
    settings = grid.env_settings_for_tiers([spec], roots.corpus_root)
    produced: list[Any] = []
    previous = pin_torch_threads(1)
    try:
        for seed in sorted(int(s) for s in checkpoints):
            factory = grid.cell_factory("dt", str(checkpoints[seed]), int(steps), float(target_rtg), None)
            print(f"{arm} seed {seed} over {len(draws)} draws", flush=True)
            produced.extend(
                EpisodeResult(arm=arm, seed=r.seed, draw_id=r.draw_id, att_horizon=r.att_horizon,
                              horizon_vehicle_count=r.horizon_vehicle_count, episode_reward=r.episode_reward)
                for r in evaluate_arm(
                    arm=arm, seed=seed, draw_ids=list(draws),
                    config_for_draw=lambda draw: draw_config_path(SCENARIO_KEY, int(draw), out_root=roots.draws_root),
                    env_settings=settings, scenario_id=SCENARIO_KEY, choose_action_factory=factory,
                    engine_seed=p94.ENGINE_SEED,
                )
            )
    finally:
        pin_torch_threads(previous)
    return produced


def route_i_payload(roots: p94.Roots, declaration: Mapping[str, Any], *, seeds: Sequence[int] = p94.SEEDS,
                    draws: Sequence[int] = p94.HELD_OUT_DRAWS, steps: int = p94.GRADIENT_STEPS) -> dict[str, Any]:
    """Route (i) for ``dt@mix50_fresh``: ``method_tier_grid._run_evaluate``'s body (:func:`evaluate_route_i`) on each
    seed's checkpoint, its digest the run record's, at the source prompt; the completeness check, ``cell_stats`` and
    its payload, ``p4.6-grid/1.0``."""
    import offline.method_tier_grid as grid
    from offline.dt_gate import runtime_provenance

    spec = p94.tier_spec(rtg_scale=float(declaration["rtg_scale"]))
    checkpoints = {int(seed): new_checkpoint_identity(roots, int(seed), steps=steps)["path"] for seed in seeds}
    produced = evaluate_route_i(roots, spec, checkpoints, target_rtg=p94.SOURCE_TARGET, draws=draws, steps=steps)
    grid.assert_cell_complete("dt", p94.TIER, list(seeds), list(draws), produced)
    return {
        "format_version": grid.GRID_FORMAT_VERSION,
        "arm": CITYFLOW_ARM,
        "tier": p94.TIER,
        "method": "dt",
        "declared_gradient_steps": int(steps),
        "engine_seed": int(p94.ENGINE_SEED),
        "target_rtg": float(p94.SOURCE_TARGET),
        "cell": grid.cell_stats(produced),
        "episodes": [
            {"arm": e.arm, "seed": e.seed, "draw_id": e.draw_id, "att_horizon": e.att_horizon,
             "horizon_vehicle_count": e.horizon_vehicle_count, "episode_reward": e.episode_reward}
            for e in produced
        ],
        "runtime": runtime_provenance(),
    }


def assert_route_i_complete(payload: Mapping[str, Any], *, seeds: Sequence[int] = p94.SEEDS,
                            draws: Sequence[int] = p94.HELD_OUT_DRAWS) -> None:
    """Route (i)'s file must be the complete cell: one ``dt@mix50_fresh`` episode per (seed, draw), nothing else."""
    import offline.method_tier_grid as grid

    keys = [(int(e["seed"]), int(e["draw_id"])) for e in payload.get("episodes", ())]
    expected = {(int(s), int(d)) for s in seeds for d in draws}
    if (payload.get("format_version") != grid.GRID_FORMAT_VERSION or payload.get("arm") != CITYFLOW_ARM
            or {str(e["arm"]) for e in payload["episodes"]} != {CITYFLOW_ARM} or set(keys) != expected
            or len(keys) != len(expected) or float(payload.get("target_rtg")) != p94.SOURCE_TARGET):
        raise ValueError(f"the route (i) record is not the complete {len(seeds)} x {len(draws)} {CITYFLOW_ARM} cell")


def route_i(roots: p94.Roots, declaration: Mapping[str, Any], *, seeds: Sequence[int] = p94.SEEDS,
            draws: Sequence[int] = p94.HELD_OUT_DRAWS, steps: int = p94.GRADIENT_STEPS) -> dict[str, Any]:
    """Route (i), once: a complete record on disk is read back (and checked); otherwise the cell is rolled and written
    once.  One unit of work: the whole cell."""
    output = route_i_output(roots)
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert_route_i_complete(payload, seeds=seeds, draws=draws)
        return payload
    payload = route_i_payload(roots, declaration, seeds=seeds, draws=draws, steps=steps)
    assert_route_i_complete(payload, seeds=seeds, draws=draws)
    p94.write_once(output, payload, roots)
    return payload


# ----------------------------------------------------------------------
# CityFlow route (ii): P8.4b's run_campaign with P9.4's factory and settings
# ----------------------------------------------------------------------


def route_ii_work(roots: p94.Roots) -> Path:
    return Path(roots.runs_root) / "eval_ii"


def route_ii_keys(*, seeds: Sequence[int] = p94.SEEDS, draws: Sequence[int] = p94.HELD_OUT_DRAWS) -> list[Any]:
    from offline import att_rederivation as ar

    return [ar.CellKey(CITYFLOW_SCENARIO, CITYFLOW_ARM, int(seed), int(draw)) for seed in seeds for draw in draws]


def committed_from_route_i(payload: Mapping[str, Any]) -> dict[tuple[str, str, int | None, int], float]:
    """Route (i)'s ``att_horizon`` per episode, keyed as ``att_rederivation`` keys an episode."""
    out: dict[tuple[str, str, int | None, int], float] = {}
    for episode in payload["episodes"]:
        key = (CITYFLOW_SCENARIO, str(episode["arm"]), int(episode["seed"]), int(episode["draw_id"]))
        if key in out:
            raise ValueError(f"{key} appears twice in the route (i) record")
        out[key] = float(episode["att_horizon"])
    return out


def factory_builder(roots: p94.Roots, *, steps: int = p94.GRADIENT_STEPS) -> Callable[..., tuple[Any, dict[str, Any]]]:
    """The ``factory_builder`` route (ii) runs with: ``dt@mix50_fresh`` gets the very factory route (i) used
    (``method_tier_grid.cell_factory`` on the checkpoint, at the source prompt); any other arm is refused."""

    def builder(scenario: str, arm: str, seed: int | None, probe_roots: Any, *, device: Any,
                config_path: Any) -> tuple[Any, dict[str, Any]]:
        import offline.method_tier_grid as grid

        if (scenario, arm) != (CITYFLOW_SCENARIO, CITYFLOW_ARM) or seed is None:
            raise ValueError(f"{scenario}:{arm} seed {seed} is not a cell of P9.4's route (ii)")
        identity = new_checkpoint_identity(roots, int(seed), steps=steps)
        factory = grid.cell_factory("dt", identity["path"], int(steps), float(p94.SOURCE_TARGET), device)
        return factory, {"kind": "checkpoint", "detail": f"{scenario}:{arm} at {int(steps)} steps",
                         "checkpoint": identity["path"], "route": "route (i)'s factory"}

    return builder


def env_settings_builder(roots: p94.Roots, declaration: Mapping[str, Any]) -> Callable[[str, str, Any], dict[str, Any]]:
    """The settings route (ii) runs with: ``mix50_fresh``'s, by ``method_tier_grid.env_settings_for_tiers`` -- route
    (i)'s own call; any other tier is refused."""

    def build(scenario: str, tier: str, probe_roots: Any) -> dict[str, Any]:
        import offline.method_tier_grid as grid

        if (scenario, tier) != (CITYFLOW_SCENARIO, p94.TIER):
            raise ValueError(f"{scenario}/{tier} is not P9.4's tier")
        return grid.env_settings_for_tiers([p94.tier_spec(rtg_scale=float(declaration["rtg_scale"]))],
                                           roots.corpus_root)

    return build


def declare_route_ii(roots: p94.Roots, *, seeds: Sequence[int] = p94.SEEDS,
                     draws: Sequence[int] = p94.HELD_OUT_DRAWS) -> dict[str, Any]:
    """Route (ii)'s campaign manifest, written ONCE by one process before any shard (P8.4b's format); declaring again
    is a no-op, and a manifest on disk that is not this campaign's is refused."""
    from offline import att_rederivation as ar

    manifest = ar.campaign_manifest(route_ii_keys(seeds=seeds, draws=draws), engine_seed=p94.ENGINE_SEED)
    path = route_ii_work(roots) / ar.MANIFEST_NAME
    ar.assert_manifest_agrees(path, manifest)
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError(f"{path} is not this campaign's manifest although its digest agrees")
    if not path.is_file():
        p94.write_once(path, manifest, roots)
    return manifest


def run_route_ii(roots: p94.Roots, declaration: Mapping[str, Any], *, shard: int = 0, of: int = 1,
                 deadline_unix: float | None = None, seeds: Sequence[int] = p94.SEEDS,
                 draws: Sequence[int] = p94.HELD_OUT_DRAWS, steps: int = p94.GRADIENT_STEPS) -> dict[str, Any]:
    """Route (ii) on this shard's episodes: ``att_rederivation.run_campaign`` with route (i)'s values as ``committed``,
    so ``reproduces_committed`` means the two routes agree under ``==``.  ANY refusal stops the stage: a disagreement is
    a finding, never a cell to skip.  A session ends between episodes at *deadline_unix* or at the stop file."""
    from offline import att_rederivation as ar
    from offline.admission_probe import ProbeRoots
    from offline.offline_baselines import pin_torch_threads

    output = route_i_output(roots)
    if not output.exists():
        raise ValueError(f"{output} is absent: route (ii) checks route (i)'s values, which come first")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert_route_i_complete(payload, seeds=seeds, draws=draws)
    manifest = ar.campaign_manifest(route_ii_keys(seeds=seeds, draws=draws), engine_seed=p94.ENGINE_SEED)
    path = route_ii_work(roots) / ar.MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: declare route (ii) once, before the shards")
    ar.assert_manifest_agrees(path, manifest)
    work = route_ii_work(roots)
    p94.assert_target(work, roots)
    keys = ar.shard_cells(route_ii_keys(seeds=seeds, draws=draws), shard=shard, of=of)
    probe = ProbeRoots(repo_root=Path(roots.repo_root), corpus_root=Path(roots.corpus_root),
                       draws_root=Path(roots.draws_root), output_root=Path(roots.output_root), work_dir=work)
    previous = pin_torch_threads(1)
    try:
        outcome = ar.run_campaign(
            keys, roots=probe, engine_seed=p94.ENGINE_SEED, device=None, committed=committed_from_route_i(payload),
            protected=(), factory_builder=factory_builder(roots, steps=steps),
            should_stop=lambda: p94.session_stop_reason(roots, deadline_unix) is not None,
            env_settings_builder=env_settings_builder(roots, declaration),
        )
    finally:
        pin_torch_threads(previous)
    if outcome["n_refused"]:
        first = outcome["refused"][0]
        raise RuntimeError(
            f"route (ii) does not reproduce route (i) on {outcome['n_refused']} episode(s); first {first['arm']} seed "
            f"{first['seed']} draw {first['draw_id']}: {first['reason']}. The stage stops; the author decides"
        )
    return outcome


def cityflow_status(roots: p94.Roots) -> dict[str, Any]:
    """Route (i)'s record and route (ii)'s campaign, recomputed from disk."""
    from offline import att_rederivation as ar

    route_i_done = False
    if route_i_output(roots).exists():
        try:
            assert_route_i_complete(json.loads(route_i_output(roots).read_text(encoding="utf-8")))
            route_i_done = True
        except (ValueError, KeyError, TypeError):
            route_i_done = False
    route_ii = ar.campaign_status(route_ii_work(roots))
    return {"route_i_complete": route_i_done, "route_ii": route_ii,
            "complete": route_i_done and bool(route_ii["complete"])}


# ----------------------------------------------------------------------
# SUMO: the cells, the guards, and the worker
# ----------------------------------------------------------------------


def new_cells(*, arms: Sequence[str] = p94.SUMO_ARMS, seeds: Sequence[int] = p94.SEEDS,
              draws: Sequence[int] = p94.HELD_OUT_DRAWS) -> list[dict[str, Any]]:
    """Plan 4.1's 1,000 cells: 2 arms x 5 seeds x 100 draws."""
    return [{"kind": "dt", "subject": p94.SUBJECT, "arm": str(arm), "seed": int(seed), "draw_id": int(draw),
             "stage": STAGE_NEW} for arm in arms for seed in seeds for draw in draws]


def g1_cells(*, draws: Sequence[int] = p94.GUARD_DRAWS) -> list[dict[str, Any]]:
    """G1: the registered mix50 DT, both arms, five seeds, draws 1000-1009 -- 100 cells."""
    return [{"kind": "dt", "subject": p94.REGISTERED_TIER, "arm": str(arm), "seed": int(seed), "draw_id": int(draw),
             "stage": STAGE_G1} for arm in p94.SUMO_ARMS for seed in p94.SEEDS for draw in draws]


def g2_cells(*, draws: Sequence[int] = p94.GUARD_DRAWS) -> list[dict[str, Any]]:
    """G2: the four anchors on draws 1000-1009 -- 40 cells, in P9.2's cell shape."""
    return [{"backend": "sumo", "scenario": SCENARIO_KEY, "arm": str(arm), "draw_id": int(draw)}
            for arm in p94.G2_ARMS for draw in draws]


def cells_dir(roots: p94.Roots) -> Path:
    return Path(roots.run_root) / "cells"


def guards_dir(roots: p94.Roots) -> Path:
    return Path(roots.run_root) / "guards"


def guards_record_path(roots: p94.Roots) -> Path:
    return guards_dir(roots) / "guards.json"


def cell_path(roots: p94.Roots, kind: str, cell: Mapping[str, Any]) -> Path:
    import offline.sumo_anchor_rescore as sar
    import offline.transfer_curve as tc

    if kind == "new":
        return cells_dir(roots) / tc.cell_chunk_name(cell)
    if kind == "g1":
        return guards_dir(roots) / "g1" / tc.cell_chunk_name(cell)
    if kind == "g2":
        return guards_dir(roots) / "g2" / sar.cell_name(cell)
    raise ValueError(f"unknown cell kind {kind!r}")


def new_cell_payload(roots: p94.Roots, cell: Mapping[str, Any], declaration: Mapping[str, Any], *,
                     steps: int = p94.GRADIENT_STEPS, facts: Any = None) -> dict[str, Any]:
    """One ``mix50_fresh`` SUMO cell through ``transfer_curve.run_cell`` with the subject's inputs supplied (its prompt
    from the declaration, its facts from its five checkpoints, its checkpoint hashed at consumption)."""
    import offline.transfer_curve as tc

    if cell.get("subject") != p94.SUBJECT or cell.get("arm") not in p94.SUMO_ARMS:
        raise ValueError(f"{tc.cell_chunk_name(cell)} is not one of P9.4's cells")
    target = float(declaration["prompts"][str(cell["arm"])]["target_rtg"])
    inputs = {
        "target_rtg": target,
        "facts": new_subject_facts(roots, declaration, steps=steps) if facts is None else facts,
        "checkpoint": new_checkpoint_identity(roots, int(cell["seed"]), steps=steps),
        "declared_gradient_steps": int(steps),
    }
    payload = tc.run_cell(cell, out_root=roots.draws_root, output_root=roots.output_root, data_dir=roots.data_dir,
                          subject_inputs=inputs)
    if payload["target_rtg"] != target or payload["rtg_first"] != target:
        raise ValueError(f"{tc.cell_chunk_name(cell)}: the chunk's prompt is not the declared {target}")
    return payload


def cell_payload(roots: p94.Roots, kind: str, cell: Mapping[str, Any], declaration: Mapping[str, Any], *,
                 steps: int = p94.GRADIENT_STEPS) -> dict[str, Any]:
    """One SUMO episode of any of the three kinds, validated, written by nobody (the caller writes it)."""
    import offline.sumo_anchor_rescore as sar
    import offline.transfer_curve as tc

    if kind == "new":
        return new_cell_payload(roots, cell, declaration, steps=steps)
    if kind == "g1":
        return tc.run_cell(cell, out_root=roots.draws_root, output_root=roots.output_root, data_dir=roots.data_dir)
    if kind == "g2":
        payload = sar.sumo_anchor_episode(SCENARIO_KEY, int(cell["draw_id"]), str(cell["arm"]),
                                          draws_root=roots.draws_root, corpus_root=roots.corpus_root)
        sar.validate_anchor_cell(payload, cell=cell)
        return payload
    raise ValueError(f"unknown cell kind {kind!r}")


def chunk_problems(roots: p94.Roots, kind: str, cell: Mapping[str, Any], payload: Mapping[str, Any],
                   declaration: Mapping[str, Any], *, steps: int = p94.GRADIENT_STEPS) -> list[str]:
    """Why a chunk on disk may NOT be reused for *cell* -- empty when it may.  Re-derived from disk: the registered
    validator for its format, the code that rolled it equal to this code outside ``docs/`` (J1(c)), its demand digests
    today's, and for a DT chunk its checkpoint digest and (new cells) its prompt the declared ones."""
    import offline.sumo_anchor_rescore as sar
    import offline.transfer_curve as tc

    problems: list[str] = []
    try:
        if kind == "g2":
            sar.validate_anchor_cell(payload, cell=cell)
        else:
            tc.validate_cell_payload(payload, cell=cell)
    except (KeyError, TypeError, ValueError) as error:
        return [f"invalid: {error}"]
    changed = tc.code_changed_since(str(payload["git_commit"]))
    if changed:
        problems.append(f"rolled by other code ({changed[:3]})")
    if kind == "g2":
        demand = tc.demand_identity(int(cell["draw_id"]), out_root=roots.draws_root, scenario=SCENARIO_KEY)
    else:
        demand = tc.demand_identity_for(cell, out_root=roots.draws_root)
    for key in ("config_sha256", "routes_sha256"):
        if str(payload.get(key)) != str(demand[key]):
            problems.append(f"{key} is not draw {cell['draw_id']}'s today")
    if kind == "g1":
        identity = tc.checkpoint_identity_for(cell, output_root=roots.output_root, data_dir=roots.data_dir)
        if str(payload["checkpoint_sha256"]) != identity["file_sha256"]:
            problems.append("its checkpoint digest is not the registered one")
    if kind == "new":
        identity = new_checkpoint_identity(roots, int(cell["seed"]), steps=steps)
        if str(payload["checkpoint_sha256"]) != identity["file_sha256"]:
            problems.append("its checkpoint digest is not the run record's")
        if payload.get("target_rtg") != float(declaration["prompts"][str(cell["arm"])]["target_rtg"]):
            problems.append("its prompt is not the declared one")
    return problems


def _sumo_worker(task: tuple[str, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """One SUMO episode in one ``spawn`` process, written once.  A failure is RETURNED, never raised into the pool."""
    kind, cell, context = task
    roots = p94.Roots(**{key: Path(value) for key, value in context["roots"].items()})
    path = cell_path(roots, kind, cell)
    try:
        payload = cell_payload(roots, kind, cell, context["declaration"], steps=int(context["steps"]))
        p94.write_once(path, payload, roots)
        return {"name": path.name, "kind": kind, "ok": True, "seconds": payload.get("seconds"), "error": None}
    except Exception as error:  # noqa: BLE001 - reported to the caller, who stops the stage
        return {"name": path.name, "kind": kind, "ok": False, "seconds": None, "error": f"{type(error).__name__}: {error}"}


def pending(roots: p94.Roots, kind: str, cells: Sequence[Mapping[str, Any]], declaration: Mapping[str, Any], *,
            steps: int = p94.GRADIENT_STEPS) -> list[dict[str, Any]]:
    """The cells of *kind* with no chunk yet.  A chunk that exists is checked by :func:`chunk_problems`; one that may
    not be reused STOPS the stage and is left untouched."""
    todo: list[dict[str, Any]] = []
    for cell in cells:
        path = cell_path(roots, kind, cell)
        if not path.exists():
            todo.append(dict(cell))
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"{path} exists and cannot be read ({error}); it is left untouched and the stage stops")
        problems = chunk_problems(roots, kind, cell, payload, declaration, steps=steps)
        if problems:
            raise ValueError(f"{path} exists and may not be reused ({'; '.join(problems)}); it is left untouched -- "
                             "nothing under output/ is moved or overwritten -- and the stage stops for the author")
    return todo


def roll(roots: p94.Roots, tasks: Sequence[tuple[str, dict[str, Any]]], declaration: Mapping[str, Any], *,
         workers: int, deadline_unix: float | None, steps: int = p94.GRADIENT_STEPS,
         worker: Callable[..., dict[str, Any]] = _sumo_worker, context_factory: Any = None) -> dict[str, Any]:
    """Roll *tasks* (``(kind, cell)`` pairs) in a ``spawn`` pool of *workers* processes (at most six), in batches of
    *workers*: the stop file and the hours bound are checked before each batch, so a session ends between units of
    work.  A failed cell stops the stage after its batch, the failures named."""
    from multiprocessing import get_context

    if not 1 <= int(workers) <= p94.MAX_WORKERS:
        raise ValueError(f"workers {workers}: at most {p94.MAX_WORKERS} (plan section 7)")
    context = {"roots": {key: str(getattr(roots, key)) for key in ("repo_root", "output_root", "corpus_root", "draws_root")},
               "declaration": dict(declaration), "steps": int(steps)}
    results: list[dict[str, Any]] = []
    stopped_by = None
    left = list(tasks)
    if left:
        pool_context = get_context("spawn") if context_factory is None else context_factory
        with pool_context.Pool(processes=int(workers)) as pool:
            while left:
                stopped_by = p94.session_stop_reason(roots, deadline_unix)
                if stopped_by is not None:
                    break
                batch, left = left[: int(workers)], left[int(workers):]
                outcome = list(pool.imap_unordered(worker, [(kind, cell, context) for kind, cell in batch]))
                for result in outcome:
                    status = "ok" if result["ok"] else f"FAILED {result['error']}"
                    print(f"  {result['kind']} {result['name']} {status}", flush=True)
                results.extend(outcome)
                failures = [result for result in outcome if not result["ok"]]
                if failures:
                    raise RuntimeError(f"{len(failures)} SUMO cell(s) failed (first {failures[0]['name']}: "
                                       f"{failures[0]['error']}); the stage stops, nothing is written for them")
    return {"rolled": len(results), "left": len(left), "stopped_by": stopped_by,
            "seconds": [r["seconds"] for r in results if r["seconds"] is not None]}


# ----------------------------------------------------------------------
# G1 and G2: compared under == before the first new cell
# ----------------------------------------------------------------------


G1_FIELDS: tuple[str, ...] = ("e_sumo", "att_env")


def g1_comparison(roots: p94.Roots, cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """G1's chunks against ``p7_3a_zero_shot.json``'s cells of the same (kind, subject, arm, seed, draw), on ``e_sumo``
    and ``att_env`` under ``==``."""
    artifact = json.loads((roots.data_dir / p94.P7_3A_ARTIFACT).read_bytes())
    committed = {(r["kind"], r["subject"], r["arm"], r["seed"], int(r["draw_id"])): r for r in artifact["cells"]}
    mismatches = []
    for cell in cells:
        payload = json.loads(cell_path(roots, "g1", cell).read_text(encoding="utf-8"))
        record = committed[(cell["kind"], cell["subject"], cell["arm"], cell["seed"], int(cell["draw_id"]))]
        differing = [field for field in G1_FIELDS if payload[field] != record[field]]
        if differing:
            mismatches.append({"cell": cell_path(roots, "g1", cell).name, "differing": differing,
                               "rerolled": {f: payload[f] for f in differing}, "committed": {f: record[f] for f in differing}})
    return {"n_cells": len(cells), "fields": list(G1_FIELDS), "against": f"docs/data/{p94.P7_3A_ARTIFACT}",
            "against_sha256": p94.file_sha256(roots.data_dir / p94.P7_3A_ARTIFACT), "mismatches": mismatches,
            "passed": not mismatches and len(cells) == 100}


def g2_comparison(roots: p94.Roots, cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """G2's cells against ``p9_2_sumo_anchors.json``'s ``anchors.sumo.cityflow1x1.by_draw``: every field it lists, and
    ``action_sequence_sha256`` where the record carries one, under ``==`` (``sumo_anchor_rescore.compare_with_registered``)."""
    import offline.sumo_anchor_rescore as sar

    anchors = json.loads((roots.data_dir / p94.P9_2_ANCHORS).read_bytes())["anchors"]["sumo"][SCENARIO_KEY]
    fields = [str(field) for field in anchors["fields"]]
    mismatches = []
    for cell in cells:
        payload = json.loads(cell_path(roots, "g2", cell).read_text(encoding="utf-8"))
        record = anchors["by_draw"][str(int(cell["draw_id"]))][str(cell["arm"])]
        compared = fields + (["action_sequence_sha256"] if "action_sequence_sha256" in record else [])
        result = sar.compare_with_registered(payload, record, compared)
        if result["differing"] or result["missing"]:
            mismatches.append({"cell": cell_path(roots, "g2", cell).name, **result,
                               "rerolled": {f: payload.get(f) for f in result["differing"]},
                               "committed": {f: record.get(f) for f in result["differing"]}})
    return {"n_cells": len(cells), "fields": fields, "against": f"docs/data/{p94.P9_2_ANCHORS}",
            "against_sha256": p94.file_sha256(roots.data_dir / p94.P9_2_ANCHORS), "mismatches": mismatches,
            "passed": not mismatches and len(cells) == 40}


class GuardFailed(RuntimeError):
    """Plan 4.3: a guard differs under ``==``; the stage stops before its first new cell and the author decides."""


def guard_report(record: Mapping[str, Any]) -> str:
    lines = [f"GUARDS: G1 {'passed' if record['g1']['passed'] else 'FAILED'}, G2 "
             f"{'passed' if record['g2']['passed'] else 'FAILED'}"]
    for name in ("g1", "g2"):
        for mismatch in record[name]["mismatches"]:
            lines.append(f"  {name} {mismatch['cell']}: differing {mismatch['differing']}"
                         + (f", missing {mismatch['missing']}" if mismatch.get("missing") else "")
                         + f"; re-rolled {mismatch['rerolled']} committed {mismatch['committed']}")
    return "\n".join(lines)


def assert_guards_authorise(roots: p94.Roots, record: Mapping[str, Any]) -> None:
    """A guard record authorises new cells only if it is this task's, passed, and was written by THIS code."""
    import offline.transfer_curve as tc

    if record.get("format_version") != GUARDS_FORMAT_VERSION:
        raise ValueError(f"{guards_record_path(roots)} is not a {GUARDS_FORMAT_VERSION} record")
    if record.get("passed") is not True:
        raise GuardFailed(guard_report(record))
    if record.get("git_dirty") is not False or tc.code_changed_since(str(record.get("git_commit"))):
        raise ValueError(f"{guards_record_path(roots)} was written by other or uncommitted code; it proves nothing "
                         "about this code")


def guards(roots: p94.Roots, declaration: Mapping[str, Any], *, workers: int, deadline_unix: float | None,
           g1: Sequence[Mapping[str, Any]] | None = None, g2: Sequence[Mapping[str, Any]] | None = None,
           steps: int = p94.GRADIENT_STEPS) -> dict[str, Any]:
    """G1 and G2 (plan 4.3): roll what is missing, then compare and write the guard record once.  Returns the record,
    or ``None``-valued ``record`` if the session ended before both were complete.  A failed guard raises
    :class:`GuardFailed` with the report."""
    import offline.transfer_curve as tc

    g1 = g1_cells() if g1 is None else list(g1)
    g2 = g2_cells() if g2 is None else list(g2)
    path = guards_record_path(roots)
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        assert_guards_authorise(roots, record)
        return {"record": record, "rolled": 0}
    tasks = [("g1", cell) for cell in pending(roots, "g1", g1, declaration, steps=steps)]
    tasks += [("g2", cell) for cell in pending(roots, "g2", g2, declaration, steps=steps)]
    outcome = roll(roots, tasks, declaration, workers=workers, deadline_unix=deadline_unix, steps=steps)
    if outcome["left"] or any(not cell_path(roots, "g1", c).exists() for c in g1) or any(
            not cell_path(roots, "g2", c).exists() for c in g2):
        return {"record": None, "rolled": outcome["rolled"], "stopped_by": outcome["stopped_by"]}
    record = {"format_version": GUARDS_FORMAT_VERSION, "task": "P9.4", "label": p94.LABEL,
              "rule": "plan 4.3: any difference under == stops the stage before its first new cell",
              "g1": g1_comparison(roots, g1), "g2": g2_comparison(roots, g2), **tc._git_provenance()}
    record["passed"] = bool(record["g1"]["passed"] and record["g2"]["passed"])
    p94.write_once(path, record, roots)
    print(guard_report(record), flush=True)
    if not record["passed"]:
        raise GuardFailed(guard_report(record))
    return {"record": record, "rolled": outcome["rolled"]}


def sumo_session(roots: p94.Roots, declaration: Mapping[str, Any], *, workers: int, deadline_unix: float | None,
                 cells: Sequence[Mapping[str, Any]] | None = None, steps: int = p94.GRADIENT_STEPS) -> dict[str, Any]:
    """One SUMO session: the guards first (every one of their chunks on disk and the record passed), then the new
    cells; between batches the stop file and the hours bound end it.  Resumes from disk."""
    guarded = guards(roots, declaration, workers=workers, deadline_unix=deadline_unix, steps=steps)
    if guarded["record"] is None:
        return {"guards": guarded, "new": None, "stopped_by": guarded.get("stopped_by")}
    chosen = new_cells() if cells is None else list(cells)
    tasks = [("new", cell) for cell in pending(roots, "new", chosen, declaration, steps=steps)]
    outcome = roll(roots, tasks, declaration, workers=workers, deadline_unix=deadline_unix, steps=steps)
    return {"guards": {"passed": True, "rolled": guarded["rolled"]}, "new": outcome, "stopped_by": outcome["stopped_by"]}


def sumo_status(roots: p94.Roots) -> dict[str, Any]:
    """Present against declared, per kind, recomputed from disk."""
    out: dict[str, Any] = {}
    for kind, cells in (("g1", g1_cells()), ("g2", g2_cells()), ("new", new_cells())):
        present = sum(1 for cell in cells if cell_path(roots, kind, cell).exists())
        out[kind] = {"declared": len(cells), "present": present, "missing": len(cells) - present}
    out["guards_record"] = guards_record_path(roots).exists()
    out["complete"] = all(out[kind]["missing"] == 0 for kind in ("g1", "g2", "new")) and out["guards_record"]
    return out
