"""P9.1's data stages (``offline/corrective_round.py``; ``BRIEF_46`` 3.4 ``old-cells``, ``collect-random-fresh``,
``declare``): the registered hz1x1 cells A27 compares with, re-rolled through P8.4b's own route with their committed
``att_ours`` reproduced under ``==``; the two ``random_fresh`` tiers collected by ``offline.collect`` with the registered
random tiers' own settings and the fixed generator; their declarations.

Formats: committed ``docs/data`` artifacts read as they are (``p4.4-baselines/1.0``, ``p4.6-grid/1.0``); P8.4b's cell
files; collection manifests (contract C6 v1.1's ``run_metadata``).  No trajectory is written here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import offline.att_rederivation as ar
import offline.corrective_round as cr

REPO = Path(__file__).resolve().parents[1]

#: The registered hz1x1 cells A27's new cells are compared with: (committed artifact, arm as it is committed).
#: (c): IQL on the eight tiers; (d): the old random tier's arms and its reference; (e): the parents (mappo1000 from
#: P4.4, mappo500 from P4.6); (g): the 40,000-step cells on mappo1000 and mix50.
OLD_HZ1X1: set[tuple[str, str]] = {
    *{("docs/data/p4_4_baselines.json", arm) for arm in ("madt", "bc", "bc_top10", "iql")},
    *{("docs/data/p4_6_grid.json", f"iql@{tier}") for tier in ("maxpressure", "fixedtime", "random", "mappo500")},
    *{("docs/data/p4_6_grid.json", f"{m}@random") for m in ("dt", "bc", "bc_top10", "behaviour")},
    *{("docs/data/p4_6_grid.json", f"{m}@mappo500") for m in ("dt", "bc", "bc_top10")},
    *{("docs/data/p4_7_grid.json", f"iql@{tier}") for tier in ("mix33", "mix50", "mix67")},
    *{("docs/data/p4_7_grid.json", f"{m}@mix50") for m in ("dt", "bc", "bc_top10")},
}
#: Q8 (BRIEF_46 Amendment B, B.2 -- "every registered statement in which IQL appears"): the cells P4.6's / P4.7's P1
#: (BC's rank among the four) and the ``*_vs_iql`` behaviour comparisons read that the list above lacks -- ``dt``,
#: ``bc``, ``bc_top10`` on maxpressure, fixedtime, mix33, mix67 and P4.4's behaviour cell (the ruling's 13), and P4.6's
#: behaviour cells on mappo1000, mappo500, maxpressure and fixedtime (the 4 the ruling's count missed; P4.7's are the
#: same episodes, and its mixture references are constructed from P4.6's ``behaviour@mappo1000`` and
#: ``behaviour@random``).
OLD_HZ1X1_Q8: set[tuple[str, str]] = {
    *{("docs/data/p4_6_grid.json", f"{m}@{t}") for t in ("maxpressure", "fixedtime") for m in ("dt", "bc", "bc_top10")},
    *{("docs/data/p4_7_grid.json", f"{m}@{t}") for t in ("mix33", "mix67") for m in ("dt", "bc", "bc_top10")},
    ("docs/data/p4_4_baselines.json", "mappo1000"),
    *{("docs/data/p4_6_grid.json", f"behaviour@{t}") for t in ("mappo1000", "mappo500", "maxpressure", "fixedtime")},
}


def _roots(tmp_path: Path, repo: Path = REPO) -> cr.Roots:
    output = tmp_path / "output"
    output.mkdir()
    return cr.Roots(repo_root=repo, output_root=output, corpus_root=tmp_path / "corpus", draws_root=tmp_path / "draws")


# ----------------------------------------------------------------------
# Old cells
# ----------------------------------------------------------------------


def test_the_hz1x1_old_cells_are_the_registered_cells_a27_compares_with() -> None:
    """The 21 cells of M1's plan and Q8's 17 (BRIEF_46 Amendment B, B.2), each once."""
    assert {(artifact, arm) for artifact, arm in cr.HZ1X1_OLD_CELLS} == OLD_HZ1X1 | OLD_HZ1X1_Q8
    assert len(cr.HZ1X1_OLD_CELLS) == len(OLD_HZ1X1) + len(OLD_HZ1X1_Q8) == 21 + 17


def test_every_cell_p1_and_the_iql_behaviour_comparisons_read_is_an_old_cell_or_constructed_from_them() -> None:
    """Q8's coverage, derived here from the committed artifacts alone: P4.6's and P4.7's P1 read dt / bc / bc_top10 /
    iql on every tier of ``cells_by_tier`` (P4.4's for mappo1000, which P4.6 re-used); every behaviour comparison
    with an IQL arm reads its behaviour cell -- rolled on the base tiers, constructed from P4.6's components on the
    mixtures (``constructed_references``)."""
    def load(name: str) -> dict[str, Any]:
        return json.loads((REPO / "docs" / "data" / f"{name}.json").read_text(encoding="utf-8"))

    old = {(artifact, arm) for artifact, arm in cr.HZ1X1_OLD_CELLS}
    p4_4_arm = {"dt": "madt", "bc": "bc", "bc_top10": "bc_top10", "iql": "iql"}
    for name in ("p4_6_grid", "p4_7_grid"):
        payload = load(name)
        for tier, methods in payload["cells_by_tier"].items():
            for method in methods:
                if tier == "mappo1000":
                    assert ("docs/data/p4_4_baselines.json", p4_4_arm[method]) in old, (name, tier, method)
                elif tier in ("mix33", "mix50", "mix67"):
                    assert ("docs/data/p4_7_grid.json", f"{method}@{tier}") in old, (name, tier, method)
                else:
                    assert ("docs/data/p4_6_grid.json", f"{method}@{tier}") in old, (name, tier, method)
        for comparison in payload["behaviour_comparisons"]:
            if not str(comparison["left_arm"]).startswith("iql@"):
                continue
            behaviour = str(comparison["right_arm"])
            tier = behaviour.split("@")[1]
            if tier in payload.get("constructed_references", {}):
                for component in payload["constructed_references"][tier]["components"]:
                    assert ("docs/data/p4_6_grid.json", component) in old, (name, tier, component)
            else:
                assert ("docs/data/p4_6_grid.json", behaviour) in old, (name, behaviour)
    p4_4 = load("p4_4_baselines")
    for key in p4_4["behaviour_policy_comparisons"]:
        if key.endswith("_vs_iql"):
            assert ("docs/data/p4_4_baselines.json", key.split("_vs_")[0]) in old, key


def test_an_old_cells_committed_values_are_its_artifacts_own_500_episodes() -> None:
    committed = cr.old_cell_committed(REPO, "docs/data/p4_6_grid.json", "iql@maxpressure")
    payload = json.loads((REPO / "docs/data/p4_6_grid.json").read_text(encoding="utf-8"))
    rows = [e for e in payload["episodes"] if e["arm"] == "iql@maxpressure"]
    assert len(committed) == len(rows) == 500
    assert all(committed[("hz1x1", "iql@maxpressure", int(e["seed"]), int(e["draw_id"]))] == float(e["att_horizon"])
               for e in rows)
    with pytest.raises(ValueError, match="no episodes"):
        cr.old_cell_committed(REPO, "docs/data/p4_6_grid.json", "iql@mix50")


def test_old_cells_roll_through_p8_4bs_registered_builder_and_a_non_reproduction_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    seen: dict[str, Any] = {}

    def fake_run_campaign(cells: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs, cells=list(cells))
        return {"rolled": len(seen["cells"]), "n_refused": 0, "refused": []}

    monkeypatch.setattr(ar, "run_campaign", fake_run_campaign)
    cr.roll_old_cells(roots, [("docs/data/p4_6_grid.json", "iql@maxpressure")])
    assert seen["factory_builder"] is None, "a registered cell is P8.4b's own, never the campaign's builder"
    assert len(seen["cells"]) == 500 and seen["engine_seed"] == 1000
    assert Path(seen["roots"].output_root) == roots.output_root
    assert Path(seen["roots"].work_dir) == roots.runs_root / "old_cells" / "hz1x1"

    def refusing(cells: Any, **kwargs: Any) -> dict[str, Any]:
        return {"rolled": 0, "n_refused": 1, "refused": [{"arm": "iql@maxpressure", "seed": 101, "draw_id": 1000,
                                                           "reason": "ValueError: does not reproduce the committed"}]}

    monkeypatch.setattr(ar, "run_campaign", refusing)
    with pytest.raises(RuntimeError, match="does not reproduce its committed att_ours"):
        cr.roll_old_cells(roots, [("docs/data/p4_6_grid.json", "iql@maxpressure")])


# ----------------------------------------------------------------------
# collect-random-fresh
# ----------------------------------------------------------------------


def _registered_manifest(tmp_path: Path, scenario: str) -> Path:
    name = {"hz1x1": "cf_hz1x1__random", "grid4x4": "cf_grid4x4__random"}[scenario]
    directory = tmp_path / "corpus" / name
    directory.mkdir(parents=True)
    meta = {"backend": "cityflow", "behavior_policy": "random", "base_seed": 1000, "episodes": 2,
            "max_steps": 360, "delta_time": 10, "control_mode": "acyclic", "global_reward_fn": "queue_length",
            "local_reward_fn": "queue_length", "global_reward_weight": 0.0,
            "state_features": ["lane_vehicle_count", "lane_waiting", "phase_onehot"],
            "flow_randomizer_params": {"base_seed": 1000, "jitter_sigma_s": 30.0, "thin_p": 0.1, "volume_scale": 1.0},
            "flow_draw_ids": list(range(1, 201)),
            "env_paths": {"config": f"/main/configs/sim/{'cityflow1x1' if scenario == 'hz1x1' else 'cityflow_grid4x4'}.json"}}
    (directory / "manifest.json").write_text(json.dumps({"run_metadata": meta, "episodes": []}), encoding="utf-8")
    return directory


@pytest.mark.parametrize("scenario", ["hz1x1", "grid4x4"])
def test_the_collector_command_is_the_registered_random_tiers_but_the_generator(tmp_path: Path, scenario: str) -> None:
    import offline.collect as collect

    roots = _roots(tmp_path)
    registered = _registered_manifest(tmp_path, scenario)
    argv = cr.random_fresh_collect_argv(roots, scenario)
    args = collect.build_parser().parse_args(argv)
    meta = json.loads((registered / "manifest.json").read_text())["run_metadata"]
    assert args.random_rng == "per-episode" and args.policy == "random" and args.episodes == 1
    assert args.base_seed == 1000 and args.flow_draws_range == [1, 201]
    assert (args.local_reward_fn, args.global_reward_weight) == ("queue_length", 0.0)
    for key in ("max_steps", "delta_time", "control_mode", "global_reward_fn", "state_features"):
        assert getattr(args, key) == meta[key], key
    assert args.env_config == meta["env_paths"]["config"]
    params = meta["flow_randomizer_params"]
    assert (args.flow_jitter_sigma, args.flow_thin_p, args.flow_volume_scale) == (
        params["jitter_sigma_s"], params["thin_p"], params["volume_scale"])
    assert Path(args.out_dir) == roots.corpus_root / {"hz1x1": "cf_hz1x1__random_fresh",
                                                      "grid4x4": "cf_grid4x4__random_fresh"}[scenario]


def test_collect_refuses_a_tier_directory_that_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import offline.collect as collect

    roots = _roots(tmp_path)
    _registered_manifest(tmp_path, "hz1x1")
    (roots.corpus_root / "cf_hz1x1__random_fresh").mkdir()
    monkeypatch.setattr(collect, "main", lambda argv: pytest.fail("the collector ran over an existing tier"))
    with pytest.raises(FileExistsError, match="cf_hz1x1__random_fresh"):
        cr.collect_random_fresh(roots, "hz1x1")


def test_collect_refuses_a_registered_tier_that_is_not_a27s_random_settings(tmp_path: Path) -> None:
    """A27(d) states the settings; the registered random tier's manifest must carry them, or this is another tier."""
    roots = _roots(tmp_path)
    registered = _registered_manifest(tmp_path, "grid4x4")
    payload = json.loads((registered / "manifest.json").read_text())
    payload["run_metadata"]["global_reward_weight"] = 1.0
    (registered / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="global_reward_weight"):
        cr.random_fresh_collect_argv(roots, "grid4x4")


# ----------------------------------------------------------------------
# The new tier's manifest, checked before it is declared or trained
# ----------------------------------------------------------------------


def _fresh_meta(**overrides: Any) -> dict[str, Any]:
    meta = {"behavior_policy": "random", "random_rng_scheme": "per-episode", "base_seed": 1000, "episodes": 1,
            "flow_draw_ids": list(range(1, 201)), "local_reward_fn": "queue_length", "global_reward_weight": 0.0}
    meta.update(overrides)
    return meta


def test_a_fresh_manifest_carrying_a27ds_collection_passes() -> None:
    assert cr.assert_random_fresh_manifest({"run_metadata": _fresh_meta(), "episodes": [{}] * 200}) is None


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"random_rng_scheme": "per-draw"}, "random_rng_scheme"),
        ({"random_rng_scheme": None}, "random_rng_scheme"),
        ({"episodes": 2}, "episodes"),
        ({"base_seed": 1001}, "base_seed"),
        ({"flow_draw_ids": list(range(0, 200))}, "flow_draw_ids"),
        ({"global_reward_weight": 1.0}, "global_reward_weight"),
    ],
)
def test_a_fresh_manifest_that_is_not_a27ds_collection_is_refused(override: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cr.assert_random_fresh_manifest({"run_metadata": _fresh_meta(**override), "episodes": [{}] * 200})


def test_the_re_used_cells_the_old_cell_list_does_not_re_roll_are_the_episodes_of_cells_it_does() -> None:
    """Why the list re-rolls no ``*@mappo1000`` learned cell of P4.6 and no base-tier cell of P4.7: P4.6's mappo1000
    cells are P4.4's episodes (``madt`` as ``dt@mappo1000``), and P4.7's phase-1 cells are P4.6's (P4.7's gate P1),
    value for value.  P4.6's ``behaviour@mappo1000`` is P4.4's ``mappo1000`` as well; both stay in the list because
    each statement reads its own artifact's cell (the two re-rolls must then agree)."""
    def episodes(name: str, arm: str) -> dict[tuple[Any, int], tuple[float, float, float]]:
        payload = json.loads((REPO / "docs" / "data" / f"{name}.json").read_text(encoding="utf-8"))
        rows = [e for e in payload["episodes"] if e["arm"] == arm]
        out = {(e["seed"], int(e["draw_id"])): (float(e["att_horizon"]), float(e["horizon_vehicle_count"]),
                                                float(e["episode_reward"])) for e in rows}
        assert rows and len(out) == len(rows), (name, arm)
        return out

    for method, p4_4_arm in (("dt", "madt"), ("bc", "bc"), ("bc_top10", "bc_top10"), ("iql", "iql"),
                             ("behaviour", "mappo1000")):
        assert episodes("p4_6_grid", f"{method}@mappo1000") == episodes("p4_4_baselines", p4_4_arm), method
    for tier in ("mappo1000", "mappo500", "maxpressure", "fixedtime", "random"):
        for method in ("dt", "bc", "bc_top10", "iql", "behaviour"):
            assert episodes("p4_7_grid", f"{method}@{tier}") == episodes("p4_6_grid", f"{method}@{tier}"), (method,
                                                                                                           tier)
