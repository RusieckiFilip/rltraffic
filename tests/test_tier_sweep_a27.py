"""A27 in ``offline/tier_sweep.py`` (``BRIEF_46`` 3.1, 3.2, 4.2, 4.4): the grid's ``random_fresh`` tier, the
``train-baselines`` flags and arm names, the evaluation loaders, and the registered behaviour under the defaults.

No format of its own: the synthetic tier is ``tests/p5_2b_fixtures.py``'s, written through the real logger (C6 v1.1;
observation rows ``T + 1``, decision and outcome rows ``T``); checkpoints are written into ``tmp_path`` by spies.

WHY ``random_fresh`` IS NOT IN ``TIERS``
---------------------------------------
``tests/test_tier_sweep.py`` looks every key of ``tier_sweep.TIERS`` up in ``docs/data/att_ladder_v11.json``, which has no
``random_fresh``; a new key there fails a registered test.  The tier lives in ``A27_TIERS`` and the lookups consult both.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import offline.tier_sweep as ts
from tests import p5_2b_fixtures as fx


def _args(tmp_path: Path, corpus: Path, *extra: str, tier: str = "random", root: str = "p9_1_runs") -> Any:
    return ts.build_parser().parse_args([
        "--corpus-root", str(corpus), "--work-dir", str(tmp_path / root / "work"),
        "--checkpoint-dir", str(tmp_path / root / "checkpoints"), "--reuse-root", str(tmp_path / "protected"),
        "--device", "cpu", "--seeds", "101", "--gradient-steps", "2", "--tier", tier, "train-baselines", *extra,
    ])


@pytest.fixture()
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "corpus"
    fx.write_synthetic_tier(root)
    fx.install_synthetic_tier(monkeypatch)
    return root


class _RaisingSpy:
    """A trainer that must never be reached: a missing refusal fails fast instead of training."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, batch: Any, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        raise AssertionError("a trainer was called where the call should have been refused before any training")


def _raising_spies(monkeypatch: pytest.MonkeyPatch) -> tuple[_RaisingSpy, _RaisingSpy]:
    import offline.offline_baselines as ob

    iql, bc = _RaisingSpy(), _RaisingSpy()
    monkeypatch.setattr(ob, "train_iql", iql)
    monkeypatch.setattr(ob, "train_bc", bc)
    return iql, bc


REGISTERED_IQL_KEYWORDS = {"state_dim", "n_actions", "seed", "declared_gradient_steps", "batch_size", "device",
                           "checkpoint_path", "stats", "scenario_id", "provenance", "log_every"}


# ----------------------------------------------------------------------
# The registry
# ----------------------------------------------------------------------


def test_the_registered_grid_tiers_are_unchanged_and_random_fresh_sits_beside_them() -> None:
    assert set(ts.TIERS) == {"mappo1000", "maxpressure", "fixedtime", "random"}
    assert set(ts.A27_TIERS) == {"random_fresh"}
    spec = ts.A27_TIERS["random_fresh"]
    assert (spec.dirs, spec.subsample, spec.episode_count) == (("cf_grid4x4__random_fresh",), "none", 200)
    assert math.isnan(spec.ladder_att) and math.isnan(spec.mean_training_return), (
        "computed and recorded by the declaration step, not typed (BRIEF_46 3.2)"
    )


def test_tier_spec_resolves_random_fresh_and_still_refuses_an_undeclared_tier() -> None:
    assert ts.tier_spec("random_fresh") is ts.A27_TIERS["random_fresh"]
    with pytest.raises(ValueError, match="unknown tier"):
        ts.tier_spec("mappo060")


def test_declare_refuses_random_fresh_whose_ladder_values_are_not_typed(tmp_path: Path) -> None:
    args = ts.build_parser().parse_args(["--out-dir", str(tmp_path), "--reuse-root", str(tmp_path / "protected"),
                                         "--tier", "random_fresh", "declare"])
    with pytest.raises(ValueError, match="corrective_round"):
        ts._run_declare(args)
    assert sorted(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------
# train-baselines: the defaults are the registered call; flags name A27's arms
# ----------------------------------------------------------------------


def test_the_defaults_hand_the_trainers_the_registered_keywords_and_names(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    iql, bc = fx.install_train_spies(monkeypatch)
    assert ts._run_train_baselines(_args(tmp_path, corpus, root="p5_2")) == 0
    assert [set(call) - {"batch"} for call in iql.calls] == [REGISTERED_IQL_KEYWORDS]
    assert iql.calls[0]["batch_size"] == 1_280 and "target_dropout" not in iql.calls[0]
    assert [call["batch_size"] for call in bc.calls] == [64, 64, 64]
    assert [call["method"] for call in bc.calls] == ["bc", "bc_top10", "bc_top10_perix"]
    names = sorted(p.name for p in (tmp_path / "p5_2" / "checkpoints").iterdir())
    assert names == sorted(f"grid4x4_random_{m}_seed101.pt" for m in ("bc", "bc_top10", "bc_top10_perix", "iql"))
    record = json.loads((tmp_path / "p5_2" / "work" / "training_random_baselines.json").read_text(encoding="utf-8"))
    assert {run["method"] for run in record["runs"]} == {"bc", "bc_top10", "bc_top10_perix", "iql"}
    assert all(set(run) == {"tier", "method", "seed", "checkpoint_path", "gradient_steps"} for run in record["runs"])


@pytest.mark.parametrize(
    ("extra", "arms", "batch", "target"),
    [
        (("--methods", "bc", "--bc-batch-windows", "1024"), ["bc_m"], 1_024, None),
        (("--methods", "bc_top10,bc_top10_perix", "--bc-batch-windows", "102"), ["bc_top10_m", "bc_top10_perix_m"],
         102, None),
        (("--methods", "iql", "--iql-batch-transitions", "20480", "--iql-target-dropout", "off"), ["iql_m"], 20_480,
         "off"),
        (("--methods", "iql", "--iql-target-dropout", "off"), ["iql_t"], 1_280, "off"),
    ],
)
def test_each_a27_arm_is_trained_under_its_own_name_from_its_flags(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch, extra: tuple[str, ...], arms: list[str],
    batch: int, target: str | None,
) -> None:
    iql, bc = fx.install_train_spies(monkeypatch)
    assert ts._run_train_baselines(_args(tmp_path, corpus, *extra)) == 0
    calls = iql.calls if arms[0].startswith("iql") else bc.calls
    assert [call["batch_size"] for call in calls] == [batch] * len(arms)
    if target is not None:
        assert [call["target_dropout"] for call in calls] == [target]
    names = sorted(p.name for p in (tmp_path / "p9_1_runs" / "checkpoints").iterdir())
    assert names == sorted(f"grid4x4_random_{arm}_seed101.pt" for arm in arms)
    record = json.loads((tmp_path / "p9_1_runs" / "work" / "training_random_baselines.json").read_text(encoding="utf-8"))
    runs = {run["method"]: run for run in record["runs"]}
    assert sorted(runs) == sorted(arms)
    for arm in arms:
        run = runs[arm]
        assert (run["batch_size"], run["target_dropout"]) == (batch, target)
        assert run["seconds"] == 10.0 + 101 / 100.0 and run["gradient_steps"] == 2
        assert run["base_method"] == ("iql" if arm.startswith("iql") else arm[: -len("_m")])
    for call in calls:
        assert call["provenance"]["arm"] in arms


@pytest.mark.parametrize(
    ("extra", "message", "root"),
    [
        (("--methods", "bc", "--bc-batch-windows", "102"), "not an arm A27 declared", "p9_1_runs"),
        (("--methods", "iql", "--iql-batch-transitions", "20480"), "not an arm A27 declared", "p9_1_runs"),
        (("--methods", "bc,iql", "--iql-target-dropout", "off"), "registered cell", "p9_1_runs"),
        (("--methods", "iql", "--iql-target-dropout", "off"), "p9_", "p5_2"),
    ],
)
def test_an_a27_call_is_refused_before_any_training_or_write(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch, extra: tuple[str, ...], message: str, root: str
) -> None:
    iql, bc = _raising_spies(monkeypatch)
    with pytest.raises(ValueError, match=message):
        ts._run_train_baselines(_args(tmp_path, corpus, *extra, root=root))
    assert not iql.calls and not bc.calls
    assert not (tmp_path / root).exists()


def test_an_a27_call_refuses_a_checkpoint_path_that_exists(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not skipped as the registered path skips a file on disk: an A27 run never writes over, or beside, a checkpoint it
    did not just produce (``BRIEF_46`` 3.1)."""
    iql, _ = _raising_spies(monkeypatch)
    checkpoints = tmp_path / "p9_1_runs" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "grid4x4_random_iql_t_seed101.pt").write_bytes(b"earlier")
    with pytest.raises(FileExistsError, match="grid4x4_random_iql_t_seed101.pt"):
        ts._run_train_baselines(_args(tmp_path, corpus, "--methods", "iql", "--iql-target-dropout", "off"))
    assert not iql.calls
    assert (checkpoints / "grid4x4_random_iql_t_seed101.pt").read_bytes() == b"earlier"


def _random_fresh_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``random_fresh`` tier of three episodes, one per draw: A27(d)'s distinct-schedule check (C4a) refuses a tier
    whose episodes share a schedule, and the scripted fixture's schedule is a function of the draw."""
    root = tmp_path / "corpus"
    fx.write_synthetic_tier(root, episodes_per_draw=1)
    monkeypatch.setattr(ts, "A27_TIERS", {"random_fresh": ts.TierSpec(
        tier="random_fresh", dirs=(fx.RANDOM_DIR,), subsample="none", episode_count=3, ladder_att=math.nan,
        mean_training_return=math.nan)})
    monkeypatch.setattr(ts, "adjacency_for_tier", lambda *args, **kwargs: None)
    return root


@pytest.mark.parametrize(("methods", "flags", "refused"), [
    ("bc,iql", (), "['bc', 'iql']"),
    ("bc_top10,bc_top10_perix", (), "['bc_top10', 'bc_top10_perix']"),
    ("iql", ("--iql-target-dropout", "off"), "['iql_t']"),
])
def test_on_the_grids_random_fresh_every_recipe_outside_a27d_is_refused_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, methods: str, flags: tuple[str, ...], refused: str
) -> None:
    """CORRECTED by BRIEF_46 Amendment B, B.1(1), which authorises this edit in writing: this test used to assert that
    the registered recipes train on the grid's ``random_fresh`` under their registered names -- A27(d)'s hz1x1 list
    applied to the grid. A27(d) declares exactly ``dt_nomix``, ``bc_m``, ``bc_top10_m``, ``bc_top10_perix_m`` and
    ``iql_m`` there, so a registered recipe (bc at 64 windows, %BC, IQL at 1,280 transitions with dropout active) and
    ``iql_t`` are refused -- after A27(d)'s distinct-schedule check, which this fixture passes, and before the split is
    built, anything trained or anything written."""
    root = _random_fresh_fixture(tmp_path, monkeypatch)
    iql, bc = _raising_spies(monkeypatch)
    with pytest.raises(ValueError, match=re.escape(refused) + ".*A27\\(d\\)"):
        ts._run_train_baselines(_args(tmp_path, root, "--methods", methods, *flags, tier="random_fresh"))
    assert not iql.calls and not bc.calls
    assert not (tmp_path / "p9_1_runs" / "checkpoints").exists()


def test_on_the_grids_random_fresh_the_matched_arms_train_under_their_own_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The arms A27(d) declares on the grid's ``random_fresh`` train there, under their A27 names."""
    root = _random_fresh_fixture(tmp_path, monkeypatch)
    iql, bc = fx.install_train_spies(monkeypatch)
    args = _args(tmp_path, root, "--methods", "bc,iql", "--bc-batch-windows", "1024", "--iql-batch-transitions",
                 "20480", "--iql-target-dropout", "off", tier="random_fresh")
    assert ts._run_train_baselines(args) == 0
    names = sorted(p.name for p in (tmp_path / "p9_1_runs" / "checkpoints").iterdir())
    assert names == ["grid4x4_random_fresh_bc_m_seed101.pt", "grid4x4_random_fresh_iql_m_seed101.pt"]
    assert iql.calls[0]["target_dropout"] == "off" and iql.calls[0]["batch_size"] == 20480


# ----------------------------------------------------------------------
# Evaluation: an A27 arm loads through its BASE method's loader; the fresh reference is A27(d)'s generator
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("method", "base"), [("bc_m", "bc"), ("bc_top10_m", "bc_top10"),
                                              ("bc_top10_perix_m", "bc_top10_perix"), ("iql_m", "iql"),
                                              ("iql_t", "iql"), ("iql", "iql"), ("bc", "bc")])
def test_an_arm_is_evaluated_through_its_base_methods_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, base: str
) -> None:
    import offline.offline_baselines as ob

    seen: list[tuple[str, str, int]] = []
    monkeypatch.setattr(ob, "_baseline_factory", lambda b, path, declared, device: seen.append((b, path, declared)))
    args = ts.build_parser().parse_args(["--checkpoint-dir", str(tmp_path), "--tier", "maxpressure", "evaluate",
                                         "--method", method])
    ts._arm_factory("maxpressure", method, args, 303)
    assert seen == [(base, str(tmp_path / f"grid4x4_maxpressure_{method}_seed303.pt"), 40_000)]


def test_the_fresh_reference_is_the_behaviour_cell_of_random_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import offline.collect as collect
    from offline.materialise_draws import draw_config_path

    seen: list[Any] = []
    monkeypatch.setattr(collect, "_make_random", lambda env, args, rng: seen.append(rng) or (lambda info: None))
    args = ts.build_parser().parse_args(["--draws-root", str(tmp_path), "--tier", "random_fresh", "evaluate",
                                         "--method", "behaviour"])
    factory = ts._arm_factory("random_fresh", "behaviour", args, 404)

    class Env:
        _config_path = str(draw_config_path(ts.SCENARIO_KEY, 1063, out_root=tmp_path))

    factory(Env())
    assert np.array_equal(seen[0].integers(0, 8, size=360), np.random.default_rng([404, 1063]).integers(0, 8, size=360))


# ----------------------------------------------------------------------
# The DT trainer on the A27 tier: the same refusals, never a silent skip
# ----------------------------------------------------------------------


def _forbid_dt_training(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the trainer and the corpus reader with functions that record and raise: a refusal must precede both."""
    calls: list[str] = []

    def never(name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            calls.append(name)
            raise AssertionError(f"{name} ran where the call should have been refused first")

        return call

    monkeypatch.setattr(ts, "train_tier_dt", never("train_tier_dt"))
    monkeypatch.setattr(ts, "tier_parts", never("tier_parts"))
    return calls


def _dt_args(tmp_path: Path, root: str, method: str = "dt_nomix") -> Any:
    return ts.build_parser().parse_args([
        "--corpus-root", str(tmp_path / "corpus"), "--work-dir", str(tmp_path / root / "work"),
        "--checkpoint-dir", str(tmp_path / root / "checkpoints"), "--reuse-root", str(tmp_path / "protected"),
        "--device", "cpu", "--seeds", "101", "--gradient-steps", "2", "--tier", "random_fresh", "train",
        "--method", method,
    ])


@pytest.mark.parametrize("method", ["dt_spatial", "dt_spatial_h4", "dt_nomix_h4"])
def test_the_dt_trainer_refuses_every_dt_but_dt_nomix_on_the_grids_random_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """A27(d) trains ``dt_nomix`` alone on the grid's ``random_fresh`` (BRIEF_46 Amendment B, B.1(1)); any other DT is
    refused before the corpus is read."""
    calls = _forbid_dt_training(monkeypatch)
    with pytest.raises(ValueError, match=f"\\['{method}'\\].*A27\\(d\\)"):
        ts._run_train(_dt_args(tmp_path, "p9_1_runs", method))
    assert not calls and not (tmp_path / "p9_1_runs").exists()


def test_the_dt_trainer_refuses_an_a27_tier_outside_a_p9_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _forbid_dt_training(monkeypatch)
    with pytest.raises(ValueError, match="p9_"):
        ts._run_train(_dt_args(tmp_path, "p5_2"))
    assert not calls and not (tmp_path / "p5_2").exists()


def test_the_dt_trainer_refuses_an_existing_checkpoint_on_an_a27_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _forbid_dt_training(monkeypatch)
    checkpoints = tmp_path / "p9_1_runs" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "grid4x4_random_fresh_dt_nomix_seed101.pt").write_bytes(b"earlier")
    with pytest.raises(FileExistsError, match="grid4x4_random_fresh_dt_nomix_seed101.pt"):
        ts._run_train(_dt_args(tmp_path, "p9_1_runs"))
    assert not calls and (checkpoints / "grid4x4_random_fresh_dt_nomix_seed101.pt").read_bytes() == b"earlier"


def test_the_spatial_trainer_itself_trains_on_the_grids_random_fresh_and_still_refuses_an_undeclared_tier(
        tmp_path: Path) -> None:
    """BRIEF_46 Amendment C (C.2, C.3): ``train_tier_dt`` -- the REAL trainer, no spy -- accepts A27's tier as it accepts
    the four registered ones (the mirror of ``method_tier_grid``'s check, which reads both registries): two gradient
    steps on CPU on a synthetic joint stack, the tier recorded in the checkpoint.  A name in neither registry is still
    refused, before anything is written, and the refusal names A27's tiers.  The campaign reaches this call for its five
    ``dt_nomix@random_fresh`` runs (255 runs' 231st to 235th); the DT tests above replace the trainer with a spy, so none
    of them could reach the check."""
    import torch

    from tests.test_tier_sweep import _tiny_joint_setup

    setup = _tiny_joint_setup()
    common = dict(method="dt_nomix", seed=101, adjacency=setup["adjacency"], prompts=setup["prompts"],
                  stats=setup["stats"], state_dim=setup["state_dim"], n_actions=setup["n_actions"], gradient_steps=2,
                  batch_size=2, device=torch.device("cpu"), protected=(), provenance={})
    result = ts.train_tier_dt(setup["stacked"], setup["index"], tier="random_fresh",
                              checkpoint_path=tmp_path / "random_fresh.pt", **common)
    assert len(result.losses) == 2 and all(math.isfinite(loss) for loss in result.losses)
    payload = torch.load(tmp_path / "random_fresh.pt", map_location="cpu", weights_only=False)
    assert payload["provenance"]["tier"] == "random_fresh"
    with pytest.raises(ValueError, match=r"unknown tier 'random_frsh'; .*A27's \['random_fresh'\]"):
        ts.train_tier_dt(setup["stacked"], setup["index"], tier="random_frsh",
                         checkpoint_path=tmp_path / "undeclared.pt", **common)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["random_fresh.pt"]


def test_train_baselines_refuses_a_random_fresh_tier_without_distinct_schedules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A27(d)'s registered check runs before any training: the synthetic tier logs the same scripted actions in the two
    episodes of a draw, so it cannot hold one distinct schedule per episode."""
    root = tmp_path / "corpus"
    fx.write_synthetic_tier(root)
    monkeypatch.setattr(ts, "A27_TIERS", {"random_fresh": ts.TierSpec(
        tier="random_fresh", dirs=(fx.RANDOM_DIR,), subsample="none", episode_count=6, ladder_att=math.nan,
        mean_training_return=math.nan)})
    monkeypatch.setattr(ts, "adjacency_for_tier", lambda *args, **kwargs: None)
    iql, bc = _raising_spies(monkeypatch)
    with pytest.raises(ValueError, match="distinct per-intersection"):
        ts._run_train_baselines(_args(tmp_path, root, "--methods", "bc,iql", tier="random_fresh"))
    assert not iql.calls and not bc.calls
