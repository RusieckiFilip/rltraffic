"""P7.2b T1: :mod:`offline.aligned_env` is A16's door and nothing else.

Written against ``BRIEF_36`` §3.1 / §4 + Amendment A (A2, A6) and ``docs/plans/p7.2b.md``.

The load-bearing claim is that the wrapper changes **the observation and only the observation**: the
reward tuple, the termination flags and every global scalar are the wrapped env's, byte-identical.
A wrapper that quietly changed a reward would move every probe return and every RTG in the task that
follows, and no downstream assertion would name it.

SUMO-gated tests use the same two-halves predicate as ``tests/test_materialise_parity.py``.  They
cost one env construction each and **one** 20-step episode in total, well inside the brief's budget.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from offline.aligned_env import (
    DECLARED_CITYFLOW_ROADNET,
    DECLARED_SCENARIO,
    DECLARED_SUMO_NET,
    AlignedEnv,
    aligned_sumo_env_for_draw,
    declared_alignment,
)

DRAWS_ROOT = Path("/home/filip/rltraffic/scenarios/draws")
SMOKE_DRAW = 5
IX = "intersection_1_1"


def _sumo_available() -> bool:
    """Both halves: the Python bindings and the binary the env actually launches."""
    try:
        import traci  # noqa: F401
    except ImportError:
        return False
    return shutil.which("sumo") is not None


def _draws_available(draw_id: int) -> bool:
    """Is P7.2a's parity configuration present for **this** draw? (Amendment G2.)

    The parameter is required for the reason given at length in the sibling file: a predicate that
    takes no draw id asserts a property -- *"the draws are available"* -- that does not exist, and a
    draw-5 check standing in for a draw-201 dependency is what put CI red.  Both tests here consume
    ``SMOKE_DRAW`` and say so at the decorator.
    """
    from offline.materialise_draws import parity_sumocfg_path

    return parity_sumocfg_path("cityflow1x1", int(draw_id), out_root=DRAWS_ROOT).is_file()


class _NotASumoEnv:
    """Anything that is not a ``SumoEnv``; the wrapper must refuse it whatever it looks like."""

    max_steps = 360
    intersections: tuple[Any, ...] = ()
    action_space = None

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        return {"intersections": {}}

    def step(self, action: Any) -> tuple[Any, bool, bool, dict[str, Any]]:
        return 0.0, False, False, {"intersections": {}}


# ----------------------------------------------------------------------------------
# The declared pair, and the construction refusals -- no simulator needed
# ----------------------------------------------------------------------------------
def test_the_declared_pair_is_the_admitted_one_and_its_files_exist() -> None:
    """A15(g)'s ADMITTED pair, with the paths taken from offline.parity rather than retyped."""
    assert DECLARED_SCENARIO == "hangzhou_1x1_bc-tyc"
    assert DECLARED_CITYFLOW_ROADNET.is_file()
    assert DECLARED_SUMO_NET.is_file()
    assert DECLARED_SUMO_NET.name.endswith(".net.xml")

    alignment = declared_alignment()
    assert list(alignment.intersections) == [IX]
    ix = alignment.intersections[IX]
    assert ix.canonical_state_width() == 25
    assert ix.sumo_state_width() == 32
    assert ix.n_actions == 8
    # A16(b): the canonical order is the CityFlow corpus's discovery order, NOT sorted.
    assert list(ix.canonical_lanes) != sorted(ix.canonical_lanes)


def test_a_non_sumo_env_is_refused_at_construction() -> None:
    """Amendment A6: the CLASS determines the info's shape, so the class is what is checked.

    A path-suffix test would accept the one case that matters -- a CityFlow env handed a
    ``.sumocfg`` string -- and reject nothing.
    """
    with pytest.raises(TypeError, match="SumoEnv"):
        AlignedEnv(_NotASumoEnv(), declared_alignment())


def test_the_alignment_argument_must_be_a_scenario_alignment() -> None:
    """A wrapper built with the wrong second argument would fail at the first reset, deep inside
    ``align_info``; refusing here names the mistake."""
    with pytest.raises(TypeError, match="ScenarioAlignment"):
        AlignedEnv(_NotASumoEnv(), object())


# ----------------------------------------------------------------------------------
# T1 -- the door translates the observation and NOTHING else
# ----------------------------------------------------------------------------------
@pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not available")
@pytest.mark.skipif(
    not _draws_available(SMOKE_DRAW),
    reason=f"P7.2a's parity configuration for draw {SMOKE_DRAW} is not present",
)
def test_the_wrapper_aligns_the_observation_and_leaves_every_outcome_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1. One 20-step episode on the fenced draw 5, the inner env's step tuple compared per step.

    ⚠️ **This docstring used to say "raw and wrapped compared step by step" and that comparison did
    not exist** (found by the P7.2b merge review, Amendment F2).  The reward was asserted by TYPE
    only, so a wrapper returning ``0.0`` forever passed this file and the whole campaign suite; the
    shipped code was correct and nothing pinned it.  A description standing in for an artifact, in a
    test file, where a reader is most entitled to believe it.

    The comparison is real now, and it costs no second SUMO episode: a **spy** on the inner env's
    ``step`` records the exact tuple SUMO produced, and each of the 20 steps asserts the wrapper
    handed that tuple back -- ``reward`` under ``==`` (never ``approx``: the mutation that motivated
    this is ``reward + 1e-9``), both flags by identity, and every global scalar of the raw ``info``
    carried through unchanged.

    The reward identity against the lane counts is asserted at **every** step and at least one of
    them must be non-zero: at ``t = 0`` both sides are ``0.0`` on an empty network, so an identity
    asserted only there would pass against a wrapper that returned a constant.  That vacuity is the
    defect P7.2a's Amendment D1 was written about, and it is cheap to exclude here.
    """
    from experiments.envs import make_env
    from offline.collect import _build_env_spec
    from offline.materialise_draws import parity_sumocfg_path
    from offline.sumo_att_reference import collect_style_args

    import numpy as np

    alignment = declared_alignment()
    cfg = parity_sumocfg_path("cityflow1x1", SMOKE_DRAW, out_root=DRAWS_ROOT)
    args = collect_style_args("sumo", "maxpressure", cfg, sentinel_out_dir="/nonexistent")
    raw_env = make_env(_build_env_spec(args))
    env = AlignedEnv(raw_env, alignment)
    try:
        # The forwarded surface is the wrapped env's, not a copy.
        assert env.max_steps == raw_env.max_steps == 360
        assert env.action_space is raw_env.action_space
        assert list(env.intersections) == list(raw_env.intersections)
        # __getattr__ reaches the engine handle the probe's reads need.
        assert env._sumo is raw_env._sumo

        # F2: the spy. `AlignedEnv.step` calls `self._env.step(action)`, an instance lookup, and
        # `envs/` defines no __slots__, so binding it here is what the wrapper will reach. It
        # returns the inner tuple UNCHANGED -- it only records, so the episode is the same episode.
        recorded: list[tuple[Any, bool, bool, dict[str, Any]]] = []
        inner_step = raw_env.step

        def spy(action: Any) -> tuple[Any, bool, bool, dict[str, Any]]:
            result = inner_step(action)
            recorded.append(result)
            return result

        monkeypatch.setattr(raw_env, "step", spy)
        assert env._env.step is spy, "the spy is not on the object the wrapper calls"

        aligned = env.reset(seed=1000)
        raw_payload_width = 2 * len(alignment.intersections[IX].sumo_lanes) + 16
        assert len(aligned["intersections"][IX]["state"]) == 25
        assert raw_payload_width == 32

        canonical = list(alignment.intersections[IX].canonical_lanes)
        sumo_lanes = [alignment.intersections[IX].correspondence[lane] for lane in canonical]

        non_zero_seen = False
        for _ in range(20):
            payload = aligned["intersections"][IX]
            assert len(payload["state"]) == 25
            assert payload["avail_actions"] == [0, 1, 2, 3, 4, 5, 6, 7]
            assert 0 <= int(payload["current_phase"]) <= 8

            # The reward the env produced, checked by two routes: over the ALIGNED dict keyed by
            # canonical lanes, and over the SUMO ids the wrapper translated from.
            waiting_aligned = sum(
                float(aligned["lane_waiting_vehicle_count"][lane]) for lane in canonical
            )
            assert float(payload["reward"]) == -waiting_aligned
            translation = aligned["lane_id_translation"]
            assert [translation[lane] for lane in canonical] == sumo_lanes
            if waiting_aligned != 0.0:
                non_zero_seen = True

            action = np.zeros(1, dtype=np.int64)
            reward, terminated, truncated, aligned = env.step(action)

            # F2: the wrapper returned the inner env's tuple, not something shaped like it.
            raw_reward, raw_terminated, raw_truncated, raw_info = recorded[-1]
            assert reward == raw_reward, (
                f"the wrapper changed the reward: {reward!r} against the env's {raw_reward!r}. "
                "align_info rewrites the observation, never the outcome"
            )
            assert terminated is raw_terminated and truncated is raw_truncated
            # ... and every GLOBAL SCALAR survived the alignment, checked over the raw info's key
            # set rather than against a hand-written list of three names.
            rewritten = {
                "intersections",
                "lane_vehicle_count",
                "lane_waiting_vehicle_count",
                "metrics",
            }
            compared = [
                key
                for key, value in raw_info.items()
                if key not in rewritten and isinstance(value, (int, float, bool, str))
            ]
            for key in compared:
                assert aligned[key] == raw_info[key], f"the wrapper changed the global {key!r}"
            assert {"average_travel_time", "step"} <= set(compared), (
                f"the global scalars compared were {sorted(compared)}; without those two this "
                "check would be vacuous"
            )
            if terminated or truncated:
                break

        assert len(recorded) >= 2, (
            f"the spy recorded {len(recorded)} step(s); the per-step comparison is the point of "
            "this test and it must run over the episode, not over a single step"
        )
        assert non_zero_seen, (
            "every step had zero waiting vehicles, so the reward identity was asserted only "
            "against 0.0 == -0.0 and this test proved nothing"
        )
    finally:
        env.close()


@pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not available")
@pytest.mark.skipif(
    not _draws_available(SMOKE_DRAW),
    reason=f"P7.2a's parity configuration for draw {SMOKE_DRAW} is not present",
)
def test_wrapping_an_aligned_env_is_refused_at_construction() -> None:
    """``align_info`` refuses a second application at the first ``reset`` (on the state width).

    Surfacing it in ``__init__`` turns a confusing mid-rollout failure -- after a DT has been built
    and a SUMO process started -- into a one-line refusal.  No episode is run here.
    """
    env = aligned_sumo_env_for_draw("cityflow1x1", SMOKE_DRAW, out_root=DRAWS_ROOT)
    try:
        assert isinstance(env, AlignedEnv)
        # Amendment F4.2: the alternative used to be `already aligned|AlignedEnv`, and the second
        # branch made the test pass with the DEDICATED refusal deleted -- the fallback "AlignedEnv
        # wraps a SumoEnv; got AlignedEnv" interpolates the type name and satisfies it. Only the
        # dedicated branch raises before a SUMO process and a DT have been built, which is the whole
        # reason it exists, so only its wording may be accepted here.
        with pytest.raises(TypeError, match="already aligned"):
            AlignedEnv(env, declared_alignment())
    finally:
        env.close()
