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


def _roots(tmp_path: Path, repo: Path = REPO) -> cr.Roots:
    output = tmp_path / "output"
    output.mkdir()
    return cr.Roots(repo_root=repo, output_root=output, corpus_root=tmp_path / "corpus", draws_root=tmp_path / "draws")


# ----------------------------------------------------------------------
# Old cells
# ----------------------------------------------------------------------


def test_the_hz1x1_old_cells_are_the_registered_cells_a27_compares_with() -> None:
    assert {(artifact, arm) for artifact, arm in cr.HZ1X1_OLD_CELLS} == OLD_HZ1X1
    assert len(cr.HZ1X1_OLD_CELLS) == len(OLD_HZ1X1) == 21


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
