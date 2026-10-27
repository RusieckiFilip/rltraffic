"""A27's arm vocabulary (``offline/a27_arms.py``; ``BRIEF_46`` sections 0, 3.1, 3.2, 4.4).

No format and no alignment convention: nothing is read from or written to disk except a fake draw tree in ``tmp_path``.

Every A27 arm is written out below as A27 states it -- name, base method, batch, targets, steps -- and the module's
registry must equal that literal table exactly: a recipe cannot drift from the registration without a test failing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import offline.a27_arms as arms

#: (scenario, name) -> (base, batch, target_dropout, steps), from A27 (b), (c), (g).
A27_LITERAL: dict[tuple[str, str], tuple[str, int, str | None, int]] = {
    ("hz1x1", "iql_t"): ("iql", 1_280, "off", 40_000),
    ("grid4x4", "iql_t"): ("iql", 1_280, "off", 40_000),
    ("grid4x4", "bc_m"): ("bc", 1_024, None, 40_000),
    ("grid4x4", "bc_top10_m"): ("bc_top10", 102, None, 40_000),
    ("grid4x4", "bc_top10_perix_m"): ("bc_top10_perix", 102, None, 40_000),
    ("grid4x4", "iql_m"): ("iql", 20_480, "off", 40_000),
    ("hz1x1", "dt_100k"): ("dt", 64, None, 100_000),
    ("hz1x1", "bc_100k"): ("bc", 64, None, 100_000),
    ("hz1x1", "bc_top10_100k"): ("bc_top10", 64, None, 100_000),
    ("hz1x1", "iql_100k"): ("iql", 1_280, "active", 100_000),
}

#: The registered recipes, which A27 does not touch.
REGISTERED_LITERAL: dict[tuple[str, str], tuple[int, str | None]] = {
    ("hz1x1", "dt"): (64, None),
    ("hz1x1", "bc"): (64, None),
    ("hz1x1", "bc_top10"): (64, None),
    ("hz1x1", "iql"): (1_280, "active"),
    ("grid4x4", "dt_spatial"): (64, None),
    ("grid4x4", "dt_nomix"): (64, None),
    ("grid4x4", "bc"): (64, None),
    ("grid4x4", "bc_top10"): (64, None),
    ("grid4x4", "bc_top10_perix"): (64, None),
    ("grid4x4", "iql"): (1_280, "active"),
}


def test_the_a27_registry_is_the_registrations_table() -> None:
    got = {key: (r.base, r.batch, r.target_dropout, r.steps) for key, r in arms.A27_RECIPES.items()}
    assert got == A27_LITERAL
    for key, r in arms.A27_RECIPES.items():
        assert (r.scenario, r.name) == key and not r.registered


def test_the_registered_recipes_are_the_code_constants() -> None:
    from offline.dt_gate import BATCH_SIZE
    from offline.offline_baselines import BC_BATCH_WINDOWS, IQL_BATCH_TRANSITIONS
    from offline.tier_sweep import JOINT_BATCH_SIZE

    assert (BATCH_SIZE, BC_BATCH_WINDOWS, IQL_BATCH_TRANSITIONS, JOINT_BATCH_SIZE) == (64, 64, 1_280, 64)
    got = {key: (r.batch, r.target_dropout) for key, r in arms.REGISTERED_RECIPES.items()}
    assert got == REGISTERED_LITERAL
    assert all(r.registered and r.steps == 40_000 for r in arms.REGISTERED_RECIPES.values())


def test_no_a27_name_reuses_a_registered_name() -> None:
    registered = {name for _, name in arms.REGISTERED_RECIPES}
    added = {name for _, name in arms.A27_RECIPES}
    assert registered == {name for _, name in REGISTERED_LITERAL}
    assert added == {name for _, name in A27_LITERAL}
    assert not registered & added


@pytest.mark.parametrize("steps", [40_000, 2, 7])
def test_the_registered_recipe_keeps_its_registered_name_at_any_step_count_but_the_ceiling(steps: int) -> None:
    """Fixture runs at a handful of steps keep the registered name, exactly as every existing test trains them."""
    for (scenario, base), (batch, target) in REGISTERED_LITERAL.items():
        derived = arms.derive_arm(scenario, base, batch=batch, target_dropout=target, steps=steps)
        assert derived.name == base and derived.registered


@pytest.mark.parametrize(("scenario", "name"), sorted(A27_LITERAL))
def test_every_a27_arm_is_derived_from_its_own_knobs(scenario: str, name: str) -> None:
    base, batch, target, steps = A27_LITERAL[(scenario, name)]
    assert arms.derive_arm(scenario, base, batch=batch, target_dropout=target, steps=steps).name == name
    assert arms.recipe(scenario, name).base == base
    assert arms.is_a27_arm(scenario, name)


@pytest.mark.parametrize(
    ("scenario", "base", "batch", "target", "steps"),
    [
        ("grid4x4", "iql", 20_480, "active", 40_000),   # a second IQL variant (A27(c): no such arm)
        ("grid4x4", "bc", 102, None, 40_000),           # 102 windows belong to the top-decile arms only
        ("grid4x4", "bc_top10", 1_024, None, 40_000),
        ("hz1x1", "bc", 1_024, None, 40_000),           # no hz1x1 exposure arm (A27(b))
        ("hz1x1", "iql", 20_480, "off", 40_000),
        ("hz1x1", "iql", 1_280, "off", 100_000),        # no iql_t at the ceiling (A27(g))
        ("grid4x4", "bc", 64, None, 100_000),           # the ceiling is hz1x1's, A8(b)'s tier
        ("grid4x4", "dt_nomix", 128, None, 40_000),     # no DT is re-trained or re-batched
    ],
)
def test_an_undeclared_combination_is_refused(scenario: str, base: str, batch: int, target: str | None,
                                              steps: int) -> None:
    with pytest.raises(ValueError, match="not an arm A27 declared"):
        arms.derive_arm(scenario, base, batch=batch, target_dropout=target, steps=steps)


def test_the_target_flag_is_refused_on_a_method_without_targets() -> None:
    with pytest.raises(ValueError, match="IQL only"):
        arms.derive_arm("grid4x4", "bc", batch=64, target_dropout="off", steps=40_000)
    with pytest.raises(ValueError, match="target_dropout must be one of"):
        arms.derive_arm("hz1x1", "iql", batch=1_280, target_dropout="partly", steps=40_000)


@pytest.mark.parametrize("name", ["dt", "bc", "bc_top10", "iql"])
def test_a_registered_name_at_the_ceiling_is_refused(name: str) -> None:
    """``BRIEF_46`` 3.2: the pins for registered arms stay at 40,000."""
    with pytest.raises(ValueError, match="pinned at 40000"):
        arms.refuse_registered_name_at_ceiling("hz1x1", name, 100_000)
    arms.refuse_registered_name_at_ceiling("hz1x1", name, 40_000)
    arms.refuse_registered_name_at_ceiling("hz1x1", f"{name}_100k", 100_000)
    batch, target = REGISTERED_LITERAL[("hz1x1", name)]
    derived = arms.derive_arm("hz1x1", name, batch=batch, target_dropout=target, steps=100_000)
    assert derived.name == f"{name}_100k" and derived.steps == 100_000


def test_unknown_names_and_scenarios_are_refused() -> None:
    with pytest.raises(ValueError, match="is not an arm on grid4x4"):
        arms.recipe("grid4x4", "dt_100k")
    with pytest.raises(ValueError, match="unknown scenario"):
        arms.recipe("hz4x4", "bc_m")
    with pytest.raises(ValueError, match="not a registered method"):
        arms.derive_arm("hz1x1", "bc_top10_perix", batch=64, target_dropout=None, steps=40_000)


# ----------------------------------------------------------------------
# A27(d)'s fresh random reference
# ----------------------------------------------------------------------


class _FakeEnv:
    def __init__(self, config_path: Path) -> None:
        self._config_path = str(config_path)


def _spy_make_random(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    import offline.collect as collect

    seen: list[Any] = []

    def spy(env: Any, args: Any, rng: np.random.Generator) -> Any:
        seen.append(rng)
        return lambda info: np.zeros(1, dtype=np.int64)

    monkeypatch.setattr(collect, "_make_random", spy)
    return seen


def test_the_fresh_reference_seeds_default_rng_with_the_seed_and_the_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from offline.materialise_draws import draw_config_path

    seen = _spy_make_random(monkeypatch)
    factory = arms.fresh_random_factory(202, scenario_key="cityflow1x1", draws_root=tmp_path, draw_ids=range(1000, 1100))
    factory(_FakeEnv(draw_config_path("cityflow1x1", 1037, out_root=tmp_path)))
    assert len(seen) == 1
    drawn = seen[0].integers(0, 8, size=360)
    assert np.array_equal(drawn, np.random.default_rng([202, 1037]).integers(0, 8, size=360))
    assert not np.array_equal(drawn, np.random.default_rng(202).integers(0, 8, size=360)), (
        "the registered reference's generator (training seed alone) must not be what the fresh one draws from"
    )


def test_the_fresh_reference_refuses_an_env_on_an_undeclared_draw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from offline.materialise_draws import draw_config_path

    seen = _spy_make_random(monkeypatch)
    factory = arms.fresh_random_factory(101, scenario_key="cityflow1x1", draws_root=tmp_path, draw_ids=range(1000, 1100))
    with pytest.raises(ValueError, match="none of the 100 declared held-out draws"):
        factory(_FakeEnv(draw_config_path("cityflow1x1", 7, out_root=tmp_path)))
    with pytest.raises(ValueError, match="exposes no _config_path"):
        factory(object())
    assert seen == []


def test_the_fresh_reference_has_500_distinct_generators_over_the_registered_seeds_and_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A27(d): ``100 draws x 5 seeds, 500 distinct schedules`` -- for an all-legal mask the schedule is the generator's.

    Every generator is the one the FACTORY builds for that (seed, draw), observed through the spy."""
    from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS
    from offline.materialise_draws import draw_config_path

    seen = _spy_make_random(monkeypatch)
    for seed in TRAINING_SEEDS:
        factory = arms.fresh_random_factory(seed, scenario_key="cityflow1x1", draws_root=tmp_path,
                                            draw_ids=HELD_OUT_DRAWS)
        for draw in HELD_OUT_DRAWS:
            factory(_FakeEnv(draw_config_path("cityflow1x1", draw, out_root=tmp_path)))
    digests = {hashlib.sha256(rng.integers(0, 8, size=360).tobytes()).hexdigest() for rng in seen}
    assert len(seen) == len(TRAINING_SEEDS) * len(HELD_OUT_DRAWS) == 500
    assert len(digests) == 500


# ----------------------------------------------------------------------
# A27(e)'s design facts and A27(d)'s 200 distinct schedules -- checks every training path runs first
# ----------------------------------------------------------------------


def _mix_stream(source: str, seed: int, ret: float, n: int) -> Any:
    from offline.offline_baselines import StreamReturn

    return StreamReturn(dataset_dir=f"/c/cf_hz1x1__{source}__seed{seed}", episode_file=f"ep{n:06d}.npz", ix_id="ix0",
                        ix_index=0, episode_index=n, flow_draw=n, group=(3, 2), total_return=float(ret))


def _a27e_like() -> tuple[list[Any], list[Any]]:
    """A selection whose kept decile is exactly A27(e)'s: 14 + 6 by seed, returns spanning [-6627, -5959]."""
    kept_spec = [("mappo1000", 101, 5), ("mappo1000", 202, 6), ("mappo1000", 404, 1), ("mappo1000", 505, 2),
                 ("mappo500", 101, 3), ("mappo500", 202, 1), ("mappo500", 404, 2)]
    kept: list[Any] = []
    n = 0
    for source, seed, count in kept_spec:
        for _ in range(count):
            kept.append(_mix_stream(source, seed, -5959.0 - (668.0 * n / 19.0), n))
            n += 1
    rest = [_mix_stream("mappo1000" if i < 86 else "mappo500", 303, -7000.0 - i, 100 + i) for i in range(180)]
    return kept + rest, kept


def test_the_mixture_design_facts_are_a27es() -> None:
    facts = arms.MIXTURE_DESIGN_FACTS
    assert facts["tier"] == "mix_m1000_m500" and facts["expert_fraction"] == 0.50
    assert facts["kept_decile"] == {"mappo1000": {"101": 5, "202": 6, "404": 1, "505": 2},
                                    "mappo500": {"101": 3, "202": 1, "404": 2}}
    assert facts["decile_returns"] == [-6627.0, -5959.0]
    assert facts["pool"] == {"mappo1000": 100, "mappo500": 100}


def test_the_design_facts_check_passes_on_a27es_composition_and_records_p4as_band() -> None:
    selected, kept = _a27e_like()
    measured = arms.assert_mixture_design_facts(selected, kept)
    assert measured["kept_decile"] == arms.MIXTURE_DESIGN_FACTS["kept_decile"]
    assert measured["weaker_source_kept"] == 6 and measured["p4a_band_1_to_19"] is True


@pytest.mark.parametrize("perturb", ["swap_one_seed", "one_more_weaker", "returns_shifted", "pool_unbalanced"])
def test_the_design_facts_check_refuses_any_other_composition(perturb: str) -> None:
    selected, kept = _a27e_like()
    if perturb == "swap_one_seed":
        kept[0] = _mix_stream("mappo1000", 303, kept[0].total_return, 0)
    elif perturb == "one_more_weaker":
        kept[5] = _mix_stream("mappo500", 101, kept[5].total_return, 5)
    elif perturb == "returns_shifted":
        kept[-1] = _mix_stream("mappo500", 404, -6628.0, 19)
    else:
        selected = selected[:-1] + [_mix_stream("mappo1000", 303, -9000.0, 999)]
    with pytest.raises(ValueError, match="A27\\(e\\)'s design facts"):
        arms.assert_mixture_design_facts(selected, kept)


def _write_schedule_tier(directory: Path, schedules: list[list[list[int]]]) -> Path:
    """A minimal collected directory: a manifest and one .npz per episode carrying ix_ids and ix<i>_action."""
    import json as _json

    directory.mkdir(parents=True)
    episodes = []
    for index, columns in enumerate(schedules):
        name = f"ep{index:06d}_seed1000_draw{index + 1}.npz"
        arrays = {"ix_ids": np.asarray([f"ix{i}" for i in range(len(columns))], dtype=np.str_)}
        for i, column in enumerate(columns):
            arrays[f"ix{i}_action"] = np.asarray(column, dtype=np.int64)
        np.savez(directory / name, **arrays)
        episodes.append({"filename": name, "flow_draw": index + 1})
    (directory / "manifest.json").write_text(_json.dumps({"episodes": episodes}), encoding="utf-8")
    return directory


def test_distinct_schedules_are_counted_per_intersection_and_jointly(tmp_path: Path) -> None:
    schedules = [[[0, 1, 2], [3, 3, 3]], [[1, 1, 2], [3, 3, 4]], [[2, 1, 2], [3, 3, 5]]]
    counts = arms.distinct_schedules(_write_schedule_tier(tmp_path / "tier", schedules))
    assert counts == {"episodes": 3, "joint_distinct": 3, "per_intersection_distinct": {"ix0": 3, "ix1": 3}}
    assert arms.assert_distinct_schedules(tmp_path / "tier", expected=3) == counts


def test_one_repeated_intersection_schedule_refuses_the_tier(tmp_path: Path) -> None:
    """A27(d): 200 distinct PER-INTERSECTION sequences -- a joint schedule that differs elsewhere does not rescue it."""
    schedules = [[[0, 1, 2], [3, 3, 3]], [[1, 1, 2], [3, 3, 3]], [[2, 1, 2], [3, 3, 5]]]
    counts = arms.distinct_schedules(_write_schedule_tier(tmp_path / "tier", schedules))
    assert counts["joint_distinct"] == 3 and counts["per_intersection_distinct"]["ix1"] == 2
    with pytest.raises(ValueError, match="distinct per-intersection"):
        arms.assert_distinct_schedules(tmp_path / "tier", expected=3)


def test_a_tier_of_the_wrong_size_is_refused(tmp_path: Path) -> None:
    schedules = [[[0, 1]], [[1, 1]]]
    with pytest.raises(ValueError, match="2 episodes"):
        arms.assert_distinct_schedules(_write_schedule_tier(tmp_path / "tier", schedules), expected=200)


# ----------------------------------------------------------------------
# A27(d)'s arms on random_fresh (BRIEF_46 Amendment B, B.1(1))
# ----------------------------------------------------------------------


def test_a27d_declares_exactly_these_arms_on_each_scenarios_random_fresh() -> None:
    """Written here from A27(d)'s text: hz1x1 trains dt, bc, bc_top10, iql, iql_t; the grid trains dt_nomix, bc_m,
    bc_top10_m, bc_top10_perix_m, iql_m."""
    assert dict(arms.RANDOM_FRESH_ARMS) == {
        "hz1x1": ("dt", "bc", "bc_top10", "iql", "iql_t"),
        "grid4x4": ("dt_nomix", "bc_m", "bc_top10_m", "bc_top10_perix_m", "iql_m"),
    }


@pytest.mark.parametrize(("scenario", "arm"), [
    ("grid4x4", "bc"), ("grid4x4", "bc_top10"), ("grid4x4", "bc_top10_perix"), ("grid4x4", "iql"),
    ("grid4x4", "iql_t"), ("grid4x4", "dt_spatial"), ("hz1x1", "bc_m"), ("hz1x1", "dt_100k"), ("hz1x1", "iql_100k"),
])
def test_on_random_fresh_every_recipe_outside_the_scenarios_list_is_refused(scenario: str, arm: str) -> None:
    with pytest.raises(ValueError, match=f"\\['{arm}'\\] on {scenario}'s random_fresh.*A27\\(d\\)"):
        arms.assert_declared_on_random_fresh(scenario, "random_fresh", ["dt_nomix" if scenario == "grid4x4" else "dt",
                                                                         arm])


def test_the_declared_arms_pass_and_every_other_tier_is_untouched() -> None:
    for scenario, declared in (("hz1x1", ("dt", "bc", "bc_top10", "iql", "iql_t")),
                               ("grid4x4", ("dt_nomix", "bc_m", "bc_top10_m", "bc_top10_perix_m", "iql_m"))):
        assert arms.assert_declared_on_random_fresh(scenario, "random_fresh", declared) is None
        assert arms.assert_declared_on_random_fresh(scenario, "mappo1000", ["bc", "iql_t", "anything"]) is None
