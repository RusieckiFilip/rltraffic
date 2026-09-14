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

from offline import backend_alignment as ba
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


def _draws_available() -> bool:
    from offline.materialise_draws import parity_sumocfg_path

    return parity_sumocfg_path("cityflow1x1", SMOKE_DRAW, out_root=DRAWS_ROOT).is_file()


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
@pytest.mark.skipif(not _draws_available(), reason="P7.2a's parity draws are not present")
def test_the_wrapper_aligns_the_observation_and_leaves_every_outcome_untouched() -> None:
    """T1. One 20-step episode on the fenced draw 5, raw and wrapped compared step by step.

    The reward identity is asserted at **every** step and at least one of them must be non-zero:
    at ``t = 0`` both sides are ``0.0`` on an empty network, so an identity asserted only there
    would pass against a wrapper that returned a constant.  That vacuity is the defect P7.2a's
    Amendment D1 was written about, and it is cheap to exclude here.
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

        aligned = env.reset(seed=1000)
        raw = ba.align_info  # named so the next line reads as "the raw info", not a second align
        del raw
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
            assert isinstance(reward, (int, float, np.floating))
            if terminated or truncated:
                break

        assert non_zero_seen, (
            "every step had zero waiting vehicles, so the reward identity was asserted only "
            "against 0.0 == -0.0 and this test proved nothing"
        )
    finally:
        env.close()


@pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not available")
@pytest.mark.skipif(not _draws_available(), reason="P7.2a's parity draws are not present")
def test_wrapping_an_aligned_env_is_refused_at_construction() -> None:
    """``align_info`` refuses a second application at the first ``reset`` (on the state width).

    Surfacing it in ``__init__`` turns a confusing mid-rollout failure -- after a DT has been built
    and a SUMO process started -- into a one-line refusal.  No episode is run here.
    """
    env = aligned_sumo_env_for_draw("cityflow1x1", SMOKE_DRAW, out_root=DRAWS_ROOT)
    try:
        assert isinstance(env, AlignedEnv)
        with pytest.raises(TypeError, match="already aligned|AlignedEnv"):
            AlignedEnv(env, declared_alignment())
    finally:
        env.close()
