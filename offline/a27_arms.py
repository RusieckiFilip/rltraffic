"""A27's arm vocabulary, its exposure arithmetic and its fresh random reference -- one leaf module (P9.1, ``BRIEF_46``).

Format version: none -- this module writes nothing.  Alignment convention: none -- it reads no trajectory.

WHAT THIS IS
------------
``PREREGISTRATION.md`` A27 (tag ``v2.8-prereg-a27``) adds arms BESIDE the registered ones and requires every new cell to
carry a NEW name (A27(a)).  This module is the single place those names are declared and derived:

* :data:`A27_RECIPES` -- every A27 arm as ``(scenario, name, base method, batch, target_dropout, steps)``.  ``base`` is the
  registered method whose trainer and whose evaluation loader the arm uses; nothing else about the recipe differs from
  the registered one.  Names: ``iql_t`` (A27(c): IQL's targets without dropout), ``bc_m`` / ``bc_top10_m`` /
  ``bc_top10_perix_m`` / ``iql_m`` (A27(b): the grid at the DTs' exposure; ``iql_m`` with (c)'s targets), and the
  ``_100k`` arms of A27(g) (A8(b)'s budget secondary).
* :func:`derive_arm` -- the name a training call gets from its flags.  A combination A27 did not declare is REFUSED,
  never trained under an invented name: there is no second IQL variant (A27(c)), no hz1x1 exposure arm (A27(b)), no
  ``iql_t`` at 100,000 steps (A27(g)).  A registered name is returned only for the registered recipe, so a registered
  checkpoint path or cell name cannot be written by an A27 run.
* :func:`exposure_table` -- A27(b)'s row visits (``steps x decisions per step / decisions in the training set``) computed
  from the code constants, exactly, as :class:`fractions.Fraction`.
* :func:`fresh_random_factory` -- A27(d)'s ``random_fresh`` reference: ``numpy.random.default_rng([seed, draw_id])`` per
  held-out draw, 500 distinct schedules over 5 seeds x 100 draws.  The registered reference is untouched
  (``method_tier_grid._random_factory``: one generator per draw from the training seed alone, five schedules).

Why a leaf module: ``offline.method_tier_grid`` and ``offline.tier_sweep`` consult the names for their ``--method``
choices and loaders, and the campaign module imports both of them; the vocabulary can therefore import neither.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "A27_RECIPES",
    "BUDGET_CEILING_STEPS",
    "REGISTERED_RECIPES",
    "REGISTERED_STEPS",
    "SCENARIOS",
    "TARGET_DROPOUT_MODES",
    "ArmRecipe",
    "derive_arm",
    "exposure_table",
    "fresh_random_factory",
    "is_a27_arm",
    "recipe",
    "refuse_registered_name_at_ceiling",
    "row_visits",
]

SCENARIOS: tuple[str, ...] = ("hz1x1", "grid4x4")

#: Every registered recipe trains for this many steps (P4.4, P4.6, P4.7, P5.1, P5.2; A8(a)).
REGISTERED_STEPS = 40_000

#: A8(b)'s single ceiling for its four arms, run by A27(g).
BUDGET_CEILING_STEPS = 100_000

#: ``offline_baselines.TARGET_DROPOUT_MODES``, restated here only as the vocabulary; :func:`derive_arm` checks the two
#: agree at call time.
TARGET_DROPOUT_MODES: tuple[str, ...] = ("active", "off")


@dataclass(frozen=True)
class ArmRecipe:
    """One arm: its name, the registered method it is built from, and the only knobs A27 moves.

    ``batch`` counts what the base method's trainer counts: K-step windows for the BC family, joint windows for the
    grid DTs, single-intersection windows for the hz1x1 DT, transitions for IQL.  ``target_dropout`` is ``None`` for
    every base other than ``iql``.
    """

    scenario: str
    name: str
    base: str
    batch: int
    target_dropout: str | None
    steps: int
    registered_in: str

    @property
    def registered(self) -> bool:
        """True for the registered recipe itself (``name == base``)."""
        return self.name == self.base


def _registered() -> dict[tuple[str, str], ArmRecipe]:
    from offline.dt_gate import BATCH_SIZE
    from offline.offline_baselines import BC_BATCH_WINDOWS, IQL_BATCH_TRANSITIONS
    from offline.tier_sweep import JOINT_BATCH_SIZE

    out: dict[tuple[str, str], ArmRecipe] = {}

    def add(scenario: str, base: str, batch: int, target: str | None, where: str) -> None:
        out[(scenario, base)] = ArmRecipe(scenario, base, base, int(batch), target, REGISTERED_STEPS, where)

    add("hz1x1", "dt", BATCH_SIZE, None, "P4 / P4.6 / P4.7 (registered)")
    add("hz1x1", "bc", BC_BATCH_WINDOWS, None, "P4.4 / P4.6 / P4.7 (registered)")
    add("hz1x1", "bc_top10", BC_BATCH_WINDOWS, None, "P4.4 / P4.6 / P4.7 (registered)")
    add("hz1x1", "iql", IQL_BATCH_TRANSITIONS, "active", "P4.4 / P4.6 / P4.7 (registered)")
    add("grid4x4", "dt_spatial", JOINT_BATCH_SIZE, None, "P5.1 / P5.2 (registered)")
    add("grid4x4", "dt_nomix", JOINT_BATCH_SIZE, None, "P5.1 / P5.2 (registered)")
    # tier_sweep._run_train_baselines hands train_bc ``JOINT_BATCH_SIZE`` (64); offline_baselines.BC_BATCH_WINDOWS is the
    # same 64, which is the constant A27(b) cites.
    add("grid4x4", "bc", JOINT_BATCH_SIZE, None, "P5.1 / P5.2 (registered)")
    add("grid4x4", "bc_top10", JOINT_BATCH_SIZE, None, "P5.1 / P5.2 (registered)")
    add("grid4x4", "bc_top10_perix", JOINT_BATCH_SIZE, None, "P5.2 (registered)")
    add("grid4x4", "iql", IQL_BATCH_TRANSITIONS, "active", "P5.1 / P5.2 / P5.2b (registered)")
    return out


def _a27() -> dict[tuple[str, str], ArmRecipe]:
    from offline.dt_gate import BATCH_SIZE
    from offline.offline_baselines import BC_BATCH_WINDOWS, IQL_BATCH_TRANSITIONS

    recipes = (
        # A27(c): the registered recipe with the three target forwards in eval().
        ArmRecipe("hz1x1", "iql_t", "iql", IQL_BATCH_TRANSITIONS, "off", REGISTERED_STEPS, "A27(c)"),
        ArmRecipe("grid4x4", "iql_t", "iql", IQL_BATCH_TRANSITIONS, "off", REGISTERED_STEPS, "A27(c)"),
        # A27(b): the DTs' exposure, 20,480 decisions per step (BC: 1,024 windows of K = 20; IQL: 20,480 transitions);
        # the two top-decile arms at 102 windows (2,040 decisions; 102.4 rounded DOWN, declared in A27(b)).
        ArmRecipe("grid4x4", "bc_m", "bc", 1_024, None, REGISTERED_STEPS, "A27(b)"),
        ArmRecipe("grid4x4", "bc_top10_m", "bc_top10", 102, None, REGISTERED_STEPS, "A27(b)"),
        ArmRecipe("grid4x4", "bc_top10_perix_m", "bc_top10_perix", 102, None, REGISTERED_STEPS, "A27(b)"),
        ArmRecipe("grid4x4", "iql_m", "iql", 20_480, "off", REGISTERED_STEPS, "A27(b) with A27(c)'s targets"),
        # A27(g): A8(b)'s four registered recipes at one ceiling; IQL with its REGISTERED targets.
        ArmRecipe("hz1x1", "dt_100k", "dt", BATCH_SIZE, None, BUDGET_CEILING_STEPS, "A27(g)"),
        ArmRecipe("hz1x1", "bc_100k", "bc", BC_BATCH_WINDOWS, None, BUDGET_CEILING_STEPS, "A27(g)"),
        ArmRecipe("hz1x1", "bc_top10_100k", "bc_top10", BC_BATCH_WINDOWS, None, BUDGET_CEILING_STEPS, "A27(g)"),
        ArmRecipe("hz1x1", "iql_100k", "iql", IQL_BATCH_TRANSITIONS, "active", BUDGET_CEILING_STEPS, "A27(g)"),
    )
    return {(r.scenario, r.name): r for r in recipes}


#: Built at import from the code constants, so a recipe cannot drift from the constant it restates.
REGISTERED_RECIPES: Mapping[tuple[str, str], ArmRecipe] = _registered()
A27_RECIPES: Mapping[tuple[str, str], ArmRecipe] = _a27()


def _check_scenario(scenario: str) -> str:
    key = str(scenario)
    if key not in SCENARIOS:
        raise ValueError(f"unknown scenario {key!r}; A27's arms are declared on {SCENARIOS}")
    return key


def is_a27_arm(scenario: str, name: str) -> bool:
    """True iff *name* is an arm A27 added on *scenario* (not a registered name)."""
    return (_check_scenario(scenario), str(name)) in A27_RECIPES


def recipe(scenario: str, name: str) -> ArmRecipe:
    """The recipe of a registered or an A27 arm on *scenario*, or a refusal naming the declared ones."""
    key = (_check_scenario(scenario), str(name))
    if key in A27_RECIPES:
        return A27_RECIPES[key]
    if key in REGISTERED_RECIPES:
        return REGISTERED_RECIPES[key]
    declared = sorted(n for s, n in (*REGISTERED_RECIPES, *A27_RECIPES) if s == key[0])
    raise ValueError(f"{key[1]!r} is not an arm on {key[0]}; the declared arms are {declared}")


def refuse_registered_name_at_ceiling(scenario: str, name: str, steps: int) -> None:
    """A registered arm's budget is pinned at 40,000 steps: a run or cell under its name at the ceiling is refused.

    A8(b)'s 100,000-step cells exist only under their ``_100k`` names (A27(a): every new cell carries a new name).
    """
    if int(steps) == BUDGET_CEILING_STEPS and (_check_scenario(scenario), str(name)) in REGISTERED_RECIPES:
        raise ValueError(
            f"{name!r} is a registered arm on {scenario}, pinned at {REGISTERED_STEPS} steps; its "
            f"{BUDGET_CEILING_STEPS}-step cell is '{name}_100k' (A27(g)), and the registered name may not carry it"
        )


def derive_arm(
    scenario: str, base: str, *, batch: int, target_dropout: str | None, steps: int
) -> ArmRecipe:
    """The arm a training call with these knobs IS, or a refusal.

    * The registered recipe -- registered batch and targets, and any step count but the ceiling -- is the registered
      arm: fixture runs at a handful of steps keep the registered name exactly as they always did.
    * Anything else must be an A27 recipe: same scenario, base, batch and targets, and the ceiling iff the recipe is a
      ``_100k`` one.  Other step counts are accepted for an A27 name only below the ceiling (fixtures); the campaign
      checks every arm's declared steps exactly.
    """
    from offline.offline_baselines import TARGET_DROPOUT_MODES as TRAINER_MODES

    if tuple(TRAINER_MODES) != TARGET_DROPOUT_MODES:
        raise RuntimeError(f"the trainer's target_dropout modes {TRAINER_MODES} are not this vocabulary's")
    key = (_check_scenario(scenario), str(base))
    if key not in REGISTERED_RECIPES:
        raise ValueError(f"{base!r} is not a registered method on {scenario}; A27 builds every arm from one")
    registered = REGISTERED_RECIPES[key]
    if registered.target_dropout is None:
        if target_dropout not in (None, "active"):
            raise ValueError(f"target_dropout applies to IQL only; {base!r} has no target network")
        target = None
    else:
        target = "active" if target_dropout is None else str(target_dropout)
        if target not in TARGET_DROPOUT_MODES:
            raise ValueError(f"target_dropout must be one of {TARGET_DROPOUT_MODES}, got {target_dropout!r}")
    ceiling = int(steps) == BUDGET_CEILING_STEPS
    if int(batch) == registered.batch and target == registered.target_dropout and not ceiling:
        return registered
    matches = [
        r for r in A27_RECIPES.values()
        if (r.scenario, r.base) == key and r.batch == int(batch) and r.target_dropout == target
        and (r.steps == BUDGET_CEILING_STEPS) == ceiling
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{scenario}: {base} at batch {int(batch)}, target_dropout {target!r} and {int(steps)} steps is not an arm "
            "A27 declared (no second IQL variant, no hz1x1 exposure arm, no iql_t at the ceiling); refusing rather "
            "than training under an invented name"
        )
    return matches[0]


# ----------------------------------------------------------------------
# A27(b): exposure = row visits = steps x decisions per step / decisions in the training set
# ----------------------------------------------------------------------


def row_visits(steps: int, decisions_per_step: int, decisions_in_set: int) -> Fraction:
    """``steps x decisions_per_step / decisions_in_set``, exactly."""
    if int(decisions_in_set) < 1 or int(decisions_per_step) < 1 or int(steps) < 1:
        raise ValueError("row visits need positive steps, decisions per step and decisions in the set")
    return Fraction(int(steps) * int(decisions_per_step), int(decisions_in_set))


def exposure_table() -> dict[str, Any]:
    """A27(b)'s table from the code constants: per scenario, per arm, decisions per step and row visits at 40,000 steps.

    Training-set sizes: every tier trains on 200 episodes of 360 decisions (``method_tier_grid.TRAINING_STREAM_COUNT``,
    ``tier_sweep.TIERS[*].episode_count``, ``offline_baselines.DECISION_ROWS_PER_STREAM``), times 16 intersections on
    the grid (``tier_sweep.EXPECTED_NODES``); the top-decile arms train on ``TOP_RETURN_FRACTION`` of that.  Decisions
    per step: the DT consumes ``batch x CONTEXT_LENGTH`` decisions per intersection it covers (all 16 on the grid, one
    joint window per batch element), the BC family ``windows x CONTEXT_LENGTH``, IQL its transitions.
    """
    from offline.dt_gate import CONTEXT_LENGTH
    from offline.method_tier_grid import TRAINING_STREAM_COUNT
    from offline.offline_baselines import DECISION_ROWS_PER_STREAM, TOP_RETURN_FRACTION
    from offline.tier_sweep import EXPECTED_NODES, TIERS as GRID_TIERS

    grid_episodes = {int(spec.episode_count) for spec in GRID_TIERS.values()}
    if grid_episodes != {int(TRAINING_STREAM_COUNT)}:
        raise ValueError(f"grid tiers train on {sorted(grid_episodes)} episodes; A27(b) assumes {TRAINING_STREAM_COUNT}")
    decile = Fraction(TOP_RETURN_FRACTION).limit_denominator(1000)
    sizes = {
        "hz1x1": int(TRAINING_STREAM_COUNT) * int(DECISION_ROWS_PER_STREAM),
        "grid4x4": int(TRAINING_STREAM_COUNT) * int(DECISION_ROWS_PER_STREAM) * int(EXPECTED_NODES),
    }
    nodes = {"hz1x1": 1, "grid4x4": int(EXPECTED_NODES)}

    def per_step(r: ArmRecipe) -> int:
        if r.base in ("dt", "dt_spatial", "dt_nomix"):
            return r.batch * int(CONTEXT_LENGTH) * nodes[r.scenario]
        if r.base == "iql":
            return r.batch
        return r.batch * int(CONTEXT_LENGTH)

    out: dict[str, Any] = {}
    for scenario in SCENARIOS:
        full = sizes[scenario]
        top = decile * full
        if top.denominator != 1:
            raise ValueError(f"{scenario}: the top decile of {full} decisions is not a whole number")
        rows: dict[str, Any] = {}
        recipes = [r for (s, _), r in sorted({**REGISTERED_RECIPES, **A27_RECIPES}.items()) if s == scenario]
        for r in recipes:
            in_set = int(top) if r.base in ("bc_top10", "bc_top10_perix") else full
            rows[r.name] = {
                "base": r.base,
                "batch": r.batch,
                "decisions_per_step": per_step(r),
                "decisions_in_training_set": in_set,
                "steps": r.steps,
                "row_visits_at_40000": row_visits(REGISTERED_STEPS, per_step(r), in_set),
                "registered_in": r.registered_in,
            }
        out[scenario] = {"decisions_in_training_set": full, "decisions_in_top_decile": int(top), "arms": rows}
    return out


# ----------------------------------------------------------------------
# A27(d): the fresh random reference
# ----------------------------------------------------------------------


def fresh_random_factory(
    seed: int, *, scenario_key: str, draws_root: str | Path, draw_ids: Sequence[int]
) -> Callable[[Any], Callable[[Any, dict[str, Any]], Any]]:
    """``offline/collect.py``'s own random policy with ``numpy.random.default_rng([seed, draw_id])`` (A27(d)).

    The rollout routes hand a factory the env and nothing else, so the draw is recovered by EXACT lookup: the env's
    resolved sim-config path must equal ``materialise_draws.draw_config_path(scenario_key, d)`` for exactly one ``d`` in
    *draw_ids*, otherwise the factory refuses.  The path is read from ``CityFlowEnv._config_path`` -- a private attribute
    of a frozen class; both rollout routes (``dt_gate.evaluate_arm`` and ``admission_probe.probe_episode``) build the
    env from that very config path, which is what makes the lookup exact rather than a parse.
    """
    from offline.materialise_draws import draw_config_path

    by_path: dict[Path, int] = {}
    for draw in draw_ids:
        path = Path(draw_config_path(scenario_key, int(draw), out_root=draws_root)).resolve()
        if path in by_path:
            raise ValueError(f"two draws resolve to {path}")
        by_path[path] = int(draw)

    def factory(env: Any) -> Callable[[Any, dict[str, Any]], Any]:
        from types import SimpleNamespace

        import numpy as np

        from offline.collect import _make_random

        raw = getattr(env, "_config_path", None)
        if raw is None:
            raise ValueError("this env exposes no _config_path, so its held-out draw cannot be identified")
        draw = by_path.get(Path(str(raw)).resolve())
        if draw is None:
            raise ValueError(f"{raw} is none of the {len(by_path)} declared held-out draws of {scenario_key}")
        policy = _make_random(env, SimpleNamespace(), np.random.default_rng([int(seed), int(draw)]))
        return lambda _env, info: policy(info)

    return factory
