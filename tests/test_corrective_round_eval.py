"""P9.1's evaluation (``offline/corrective_round.py``; ``BRIEF_46`` 3.4 ``evaluate``): every new cell by BOTH routes P5.2b
used -- route (i), the module CLIs' own ``evaluate`` (``att_ours``); route (ii), P8.4b's ``att_rederivation.run_campaign``
(``att_engine`` and A11(b)'s companions) -- with the two routes' ``att_ours`` required equal under ``==``, and A27(d)'s
P3a action sequences recorded on the episodes the routes report.

Formats read and written here: ``p4.6-grid-training/1.0`` (route (i)'s hz1x1 training aggregate, ``method_tier_grid``'s
own format), P8.4b's ``p8.4b-rederivation/1.0`` cell files, ``p9.1-actions/1.0`` (P3a).  Alignment convention: an
action sequence is the ``(T, n_intersections)`` int64 array in decision order, digested by
``nortg_decomposition.action_sequence_sha256`` (P5.3b's digest, the one A27(d) names).

``att_rederivation.run_campaign`` gains ONE keyword, ``factory_builder=None``; ``None`` resolves the module's own
``build_cell_factory`` at call time, so the registered behaviour -- and every test that monkeypatches the module name --
is unchanged.  The test below proves both halves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import offline.att_rederivation as ar
import offline.corrective_round as cr
from offline.att_rederivation import CellKey


def _roots(tmp_path: Path) -> cr.Roots:
    output = tmp_path / "output"
    output.mkdir()
    return cr.Roots(repo_root=tmp_path / "repo", output_root=output, corpus_root=tmp_path / "corpus",
                    draws_root=tmp_path / "draws")


def _cell(scenario: str, tier: str, arm: str) -> cr.Cell:
    pool = (*cr.training_cells(), *cr.reference_cells())
    return next(c for c in pool if (c.scenario, c.tier, c.arm) == (scenario, tier, arm))


# ----------------------------------------------------------------------
# att_rederivation.run_campaign's one new keyword
# ----------------------------------------------------------------------


def _stub_campaign(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, att: float = 42.0) -> Any:
    def fake_probe(**kwargs: Any) -> Any:
        class Episode:
            att_ours = att

            def as_record(self) -> dict[str, Any]:
                return {"arm": kwargs["arm"], "seed": kwargs["seed"], "draw_id": kwargs["draw_id"], "att_ours": att,
                        "att_engine": att - 1.0}

        return Episode()

    monkeypatch.setattr(ar, "rederivation_env_settings", lambda *a, **k: {"max_steps": 1, "delta_time": 1})
    monkeypatch.setattr("offline.admission_probe.probe_episode", fake_probe)
    monkeypatch.setattr("offline.admission_probe.created_from_flow", lambda *a, **k: 1)
    monkeypatch.setattr("offline.materialise_draws.draw_config_path", lambda *a, **k: tmp_path / "c.json")
    return type("R", (), {"work_dir": tmp_path, "draws_root": tmp_path, "corpus_root": tmp_path,
                          "output_root": tmp_path, "repo_root": tmp_path})()


def test_run_campaign_uses_the_modules_builder_by_default_and_a_given_one_otherwise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roots = _stub_campaign(monkeypatch, tmp_path)
    module_calls: list[str] = []
    monkeypatch.setattr(ar, "build_cell_factory", lambda s, arm, seed, r, **k: module_calls.append(arm) or (None, {}))
    cells = [CellKey("hz1x1", "bc@x", 101, 1000)]
    committed = {("hz1x1", "bc@x", 101, 1000): 42.0}
    ar.run_campaign(cells, roots=roots, engine_seed=1, device=None, committed=committed, protected=())
    assert module_calls == ["bc@x"]

    given: list[str] = []
    (tmp_path / "cell_hz1x1_bc_at_x_seed101_draw1000.json").unlink()
    ar.run_campaign(cells, roots=roots, engine_seed=1, device=None, committed=committed, protected=(),
                    factory_builder=lambda s, arm, seed, r, **k: given.append(arm) or (None, {"kind": "given"}))
    assert given == ["bc@x"] and module_calls == ["bc@x"]
    record = json.loads((tmp_path / "cell_hz1x1_bc_at_x_seed101_draw1000.json").read_text())
    assert record["policy_source"] == {"kind": "given"} and record["reproduces_committed"] is True


def test_rederivation_corpus_dirs_resolves_a27s_tiers_and_still_refuses_an_unknown_one(tmp_path: Path) -> None:
    import offline.method_tier_grid as grid
    import offline.tier_sweep as ts

    corpus = tmp_path / "corpus"
    names = [*grid.A27_TIERS["mix_m1000_m500"].dirs, *grid.A27_TIERS["random_fresh"].dirs,
             *ts.A27_TIERS["random_fresh"].dirs]
    for name in names:
        (corpus / name).mkdir(parents=True)
        (corpus / name / "manifest.json").write_text("{}", encoding="utf-8")
    roots = type("R", (), {"corpus_root": corpus})()
    assert [p.name for p in ar.rederivation_corpus_dirs("hz1x1", "mix_m1000_m500", roots)] == list(
        grid.A27_TIERS["mix_m1000_m500"].dirs)
    assert [p.name for p in ar.rederivation_corpus_dirs("hz1x1", "random_fresh", roots)] == ["cf_hz1x1__random_fresh"]
    assert [p.name for p in ar.rederivation_corpus_dirs("grid4x4", "random_fresh", roots)] == ["cf_grid4x4__random_fresh"]
    with pytest.raises(ValueError, match="not a gated tier"):
        ar.rederivation_corpus_dirs("grid4x4", "mix_m1000_m500", roots)


# ----------------------------------------------------------------------
# Route (i): the CLIs' own evaluate, fed the campaign's records
# ----------------------------------------------------------------------


def _done_run(roots: cr.Roots, cell: cr.Cell, seed: int) -> None:
    import torch

    from agent.OfflineBaselines import canonical_state_dict_digest

    path = cr.checkpoint_path(roots, cell.scenario, cell.tier, cell.arm, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = {"policy.w": torch.full((2,), float(seed))}
    torch.save({"model": model}, path)
    record = cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, seed)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"format_version": cr.RUN_RECORD_FORMAT_VERSION, "scenario": cell.scenario,
                                  "tier": cell.tier, "arm": cell.arm, "seed": seed, "gradient_steps": 7,
                                  "seconds": 1.0, "checkpoint": str(path),
                                  "canonical_digest": canonical_state_dict_digest(model),
                                  **cr.recipe_fields(cell)}), encoding="utf-8")


def test_the_hz1x1_training_aggregates_are_method_tier_grids_format_one_per_budget(tmp_path: Path) -> None:
    import offline.method_tier_grid as grid

    roots = _roots(tmp_path)
    iql_t = _cell("hz1x1", "maxpressure", "iql_t")
    dt_100k = _cell("hz1x1", "mix50", "dt_100k")
    grid_cell = _cell("grid4x4", "random", "bc_m")
    for cell in (iql_t, dt_100k, grid_cell):
        for seed in (101, 202, 303, 404, 505):
            _done_run(roots, cell, seed)
    paths = cr.write_training_aggregates(roots)
    regular = json.loads((roots.run_root / "p9_1_training.json").read_text())
    ceiling = json.loads((roots.run_root / "p9_1_100k_training.json").read_text())
    assert sorted(p.name for p in paths) == ["p9_1_100k_training.json", "p9_1_training.json"]
    assert regular["format_version"] == ceiling["format_version"] == grid.TRAINING_FORMAT_VERSION
    assert (regular["declared_gradient_steps"], ceiling["declared_gradient_steps"]) == (40_000, 100_000)
    assert {(r["tier"], r["method"]) for r in regular["runs"]} == {("maxpressure", "iql_t")}
    assert {(r["tier"], r["method"]) for r in ceiling["runs"]} == {("mix50", "dt_100k")}
    for run in regular["runs"]:
        assert run["canonical_digest"] == grid.canonical_digest_of(run["checkpoint"])
        assert run["target_rtg"] is None
    assert all(run["target_rtg"] == grid.TIERS["mix50"].target_rtg for run in ceiling["runs"])


@pytest.mark.parametrize(
    ("scenario", "tier", "arm", "module", "tail"),
    [
        ("hz1x1", "maxpressure", "iql_t", "method_tier_grid", ["evaluate", "--tier", "maxpressure", "--method", "iql_t"]),
        ("hz1x1", "mix50", "dt_100k", "method_tier_grid", ["evaluate", "--tier", "mix50", "--method", "dt_100k"]),
        ("hz1x1", "random_fresh", "behaviour", "method_tier_grid", ["behaviour", "--tier", "random_fresh"]),
        ("grid4x4", "random", "iql_m", "tier_sweep", ["evaluate", "--method", "iql_m"]),
        ("grid4x4", "random_fresh", "behaviour", "tier_sweep", ["evaluate", "--method", "behaviour"]),
    ],
)
def test_route_i_is_the_module_clis_own_evaluate_with_the_campaigns_roots(
    tmp_path: Path, scenario: str, tier: str, arm: str, module: str, tail: list[str]
) -> None:
    import importlib

    roots = _roots(tmp_path)
    name, argv = cr.route_i_command(roots, _cell(scenario, tier, arm))
    assert name == module and argv[-len(tail):] == tail
    args = importlib.import_module(f"offline.{module}").build_parser().parse_args(argv)
    assert Path(args.work_dir) == roots.runs_root / "eval_i" / scenario
    assert Path(args.draws_root) == roots.draws_root and Path(args.corpus_root) == roots.corpus_root
    assert int(args.engine_seed) == 1000 and int(args.torch_threads) == 1
    if module == "method_tier_grid":
        assert Path(args.out_dir) == roots.run_root
        if arm != "behaviour":
            assert args.artifact_prefix == ("p9_1_100k" if arm.endswith("_100k") else "p9_1")
            assert int(args.steps) == (100_000 if arm.endswith("_100k") else 40_000)
    else:
        assert args.tier == tier and Path(args.checkpoint_dir) == roots.runs_root / "checkpoints" / "grid4x4"
        assert int(args.gradient_steps) == 40_000
    assert cr.route_i_output(roots, _cell(scenario, tier, arm)).parent == roots.runs_root / "eval_i" / scenario


def test_route_i_values_are_keyed_as_p8_4b_keys_a_cell_and_refuse_a_foreign_or_repeated_episode() -> None:
    payload = {"episodes": [{"arm": "iql_t@maxpressure", "seed": 101, "draw_id": 1000, "att_horizon": 103.25},
                            {"arm": "iql_t@maxpressure", "seed": 202, "draw_id": 1000, "att_horizon": 104.5}]}
    assert cr.committed_from_route_i(payload, "hz1x1", "iql_t@maxpressure") == {
        ("hz1x1", "iql_t@maxpressure", 101, 1000): 103.25, ("hz1x1", "iql_t@maxpressure", 202, 1000): 104.5}
    with pytest.raises(ValueError, match="twice"):
        cr.committed_from_route_i({"episodes": payload["episodes"] * 2}, "hz1x1", "iql_t@maxpressure")
    with pytest.raises(ValueError, match="foreign"):
        cr.committed_from_route_i(payload, "hz1x1", "iql@maxpressure")


# ----------------------------------------------------------------------
# Route (ii): the factory for a cell is route (i)'s; a registered cell is P8.4b's own
# ----------------------------------------------------------------------


def test_the_builder_gives_a_new_hz1x1_arm_route_is_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import offline.method_tier_grid as grid

    roots = _roots(tmp_path)
    cell = _cell("hz1x1", "mix50", "dt_100k")
    _done_run(roots, cell, 303)
    seen: list[tuple[Any, ...]] = []
    monkeypatch.setattr(grid, "cell_factory", lambda *a: seen.append(a) or "factory")
    factory, source = cr.a27_factory_builder(roots)("hz1x1", "dt_100k@mix50", 303, None, device=None, config_path=None)
    assert factory == "factory"
    assert seen == [("dt_100k", str(cr.checkpoint_path(roots, "hz1x1", "mix50", "dt_100k", 303)), 100_000,
                     grid.TIERS["mix50"].target_rtg, None)]
    assert source["kind"] == "checkpoint" and source["route"] == "route (i)'s factory"


def test_the_builder_gives_a_new_grid_arm_tier_sweeps_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import offline.tier_sweep as ts

    roots = _roots(tmp_path)
    cell = _cell("grid4x4", "random", "iql_m")
    _done_run(roots, cell, 101)
    seen: list[Any] = []
    monkeypatch.setattr(ts, "_arm_factory", lambda tier, method, args, seed: seen.append((tier, method, args, seed))
                        or "factory")
    factory, _ = cr.a27_factory_builder(roots)("grid4x4", "iql_m@random", 101, None, device=None, config_path=None)
    assert factory == "factory"
    tier, method, args, seed = seen[0]
    assert (tier, method, seed) == ("random", "iql_m", 101)
    assert Path(args.checkpoint_dir) == roots.runs_root / "checkpoints" / "grid4x4" and args.gradient_steps == 40_000


def test_the_builder_refuses_a_new_arm_whose_run_is_not_done(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    with pytest.raises(ValueError, match="not trained"):
        cr.a27_factory_builder(roots)("hz1x1", "iql_t@maxpressure", 101, None, device=None, config_path=None)


def test_the_builder_hands_a_registered_cell_to_p8_4bs_own_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _roots(tmp_path)
    seen: list[Any] = []
    monkeypatch.setattr(ar, "build_cell_factory", lambda *a, **k: seen.append((a, k)) or ("registered", {}))
    factory, _ = cr.a27_factory_builder(roots)("hz1x1", "iql@maxpressure", 101, "probe", device=None, config_path="c")
    assert factory == "registered" and seen[0][0] == ("hz1x1", "iql@maxpressure", 101, "probe")


def test_the_builder_gives_the_fresh_reference_a27ds_generator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import offline.a27_arms as arms

    roots = _roots(tmp_path)
    seen: list[Any] = []
    monkeypatch.setattr(arms, "fresh_random_factory", lambda seed, **k: seen.append((seed, k)) or "fresh")
    factory, source = cr.a27_factory_builder(roots)("grid4x4", "behaviour@random_fresh", 505, None, device=None,
                                                     config_path=None)
    assert factory == "fresh" and seen[0][0] == 505
    assert seen[0][1]["scenario_key"] == "cityflow_grid4x4" and Path(seen[0][1]["draws_root"]) == roots.draws_root


def test_a_route_ii_episode_that_does_not_reproduce_route_i_stops_the_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``BRIEF_46`` 3.4: per cell the two routes' ``att_ours`` must agree under ``==``; a disagreement stops."""
    roots = _roots(tmp_path)
    _stub_campaign(monkeypatch, tmp_path, att=42.0)
    monkeypatch.setattr(cr, "a27_factory_builder", lambda r: (lambda *a, **k: (None, {"kind": "stub"})))
    cell = _cell("hz1x1", "maxpressure", "iql_t")
    committed = {("hz1x1", cell.key, 101, 1000): 42.0, ("hz1x1", cell.key, 101, 1001): 42.5}
    with pytest.raises(RuntimeError, match="does not reproduce route \\(i\\)"):
        cr.run_route_ii(roots, "hz1x1", [CellKey("hz1x1", cell.key, 101, 1000), CellKey("hz1x1", cell.key, 101, 1001)],
                        committed)


# ----------------------------------------------------------------------
# P3a: the action sequences of the episodes the routes report
# ----------------------------------------------------------------------


class _FakeEnv:
    """What ``nortg_decomposition.recording_factory`` reads from an env: no action space, so
    ``Utils.infer_action_counts`` takes each intersection's ``num_phases``."""

    def __init__(self, n_ix: int) -> None:
        from types import SimpleNamespace

        self.action_space = None
        self.intersections = [SimpleNamespace(id=f"ix{i}", num_phases=8) for i in range(n_ix)]


def test_recorded_sequences_are_digested_jointly_and_per_intersection_and_tied_to_route_i(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from offline.dt_gate import EpisodeResult
    from offline.nortg_decomposition import action_sequence_sha256

    schedules = {(101, 1000): [[0, 1], [2, 3], [4, 5]], (101, 1001): [[0, 1], [2, 4], [4, 5]]}

    def fake_evaluate_arm(*, arm: str, seed: int, draw_ids: list[int], choose_action_factory: Any,
                          **kwargs: Any) -> list[EpisodeResult]:
        draw = draw_ids[0]
        policy = choose_action_factory(_FakeEnv(2))
        for row in schedules[(seed, draw)]:
            policy(None, {"row": row})
        return [EpisodeResult(arm=arm, seed=seed, draw_id=draw, att_horizon=float(draw - 900), horizon_vehicle_count=1.0,
                              episode_reward=-1.0)]

    def inner_factory(env: Any) -> Any:
        return lambda _env, info: np.asarray(info["row"], dtype=np.int64)

    monkeypatch.setattr("offline.dt_gate.evaluate_arm", fake_evaluate_arm)
    rows = cr.record_episode_actions(
        "behaviour@random_fresh", 101, [1000, 1001], inner_factory, expected={(101, 1000): 100.0, (101, 1001): 101.0},
        evaluate_kwargs={},
    )
    first, second = rows
    assert first["action_sequence_sha256"] == action_sequence_sha256([np.asarray(r) for r in schedules[(101, 1000)]])
    columns = np.asarray(schedules[(101, 1001)], dtype=np.int64)
    assert second["per_intersection_sha256"] == [action_sequence_sha256(columns[:, [i]]) for i in range(2)]
    assert first["per_intersection_sha256"][0] == second["per_intersection_sha256"][0]
    assert first["per_intersection_sha256"][1] != second["per_intersection_sha256"][1]
    with pytest.raises(ValueError, match="route \\(i\\)"):
        cr.record_episode_actions("behaviour@random_fresh", 101, [1000], inner_factory, expected={(101, 1000): 99.0},
                                  evaluate_kwargs={})
