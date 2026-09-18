"""P7.2b: the only door by which a SUMO observation enters a CityFlow-trained model's frame.

Written against ``docs/briefs/BRIEF_36_p7.2b_transfer_calibration.md`` §3.1 + Amendment A (A2, A6)
and ``docs/plans/p7.2b.md`` §3.1.

**No on-disk format, so no format version.**  The alignment convention is ``PREREGISTRATION`` **A16**
and is cited here rather than restated: the feature set, the canonical order and the phase map are
frozen there, and :func:`offline.backend_alignment.align_info` is the only route A16 admits.  This
module adds nothing to that translation -- it makes it happen at the env boundary so that a caller
cannot forget it.

WHY A WRAPPER AND NOT A CHANGE TO THE ENV
------------------------------------------
``envs/`` is frozen, and the translation is not the env's business in any case: the same SUMO env is
the right object for the MaxPressure probe, which reads the env's own lane ids and must NOT be
aligned (its pressure is computed over SUMO lanes).  The probe therefore runs unwrapped and the DT
runs wrapped, which is exactly A16's distinction -- the alignment exists for *a CityFlow-trained
model's frame*, not for SUMO.

WHAT IS FORWARDED EXPLICITLY, AND WHY THOSE
--------------------------------------------
``reset`` and ``step`` are the two methods that *change*; they are defined here so that no caller can
reach the raw ones through ``__getattr__`` by accident.  ``close`` is defined because the caller owns
the env's lifetime and a missed ``close`` leaks a SUMO process.  ``max_steps``, ``intersections`` and
``action_space`` are forwarded as properties because they are precisely what the two consumers read --
``offline.horizon_metric.horizon_rollout`` takes ``env.max_steps``, and ``agent/DTAgent.py``'s
constructor (reached through ``rtg_calibration.agent_with_target`` ->
``dt_gate.load_gate_checkpoint`` -> ``DTAgent.from_checkpoint``) takes ``.intersections`` and
``.action_space``.  Naming them makes the whole surface the DT path touches legible without tracing
``__getattr__``.  Everything else -- ``_sumo``, ``_engine_seed``, the metric hooks -- falls through
``__getattr__``, which is what lets an engine read work on a wrapped env as well as an unwrapped one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from offline.backend_alignment import ScenarioAlignment, align_info, alignment_for_scenario

__all__ = [
    "DECLARED_CITYFLOW_ROADNET",
    "DECLARED_SCENARIO",
    "DECLARED_SUMO_NET",
    "AlignedEnv",
    "aligned_observer_env_for_draw",
    "aligned_sumo_env_for_draw",
    "declared_alignment",
    "observer_env_for_draw",
]

#: The one pair this task aligns: A15(g)'s ADMITTED C3 pair.  The two network paths come from
#: :mod:`offline.parity`'s declared constants rather than being retyped, so a scenario move cannot
#: leave this module pointing at a stale file.
DECLARED_SCENARIO = "hangzhou_1x1_bc-tyc"


def _declared_paths() -> tuple[Path, Path]:
    from offline import parity

    return (
        Path(parity.DECLARED_SCENARIO_DIR) / "roadnet.json",
        Path(parity.DECLARED_SOURCE_NET),
    )


DECLARED_CITYFLOW_ROADNET, DECLARED_SUMO_NET = _declared_paths()


class AlignedEnv:
    """A SUMO env whose ``reset`` and ``step`` return A16-aligned ``info`` dicts, and nothing else.

    The reward tuple, ``terminated``/``truncated`` and the step counter are the wrapped env's,
    untouched: :func:`align_info` rewrites the observation, never the outcome.
    """

    def __init__(self, env: Any, alignment: ScenarioAlignment) -> None:
        from envs.sumo_env import SumoEnv  # lazy: importing it pulls traci

        # Each wrong argument is named for itself, so the alignment is checked independently of the
        # env rather than behind it: a caller who got both wrong should hear about both, and the
        # second argument's type is knowable without touching the first.
        if not isinstance(alignment, ScenarioAlignment):
            raise TypeError(
                f"the second argument is a ScenarioAlignment; got {type(alignment).__name__}"
            )
        # Order matters below: an AlignedEnv is not a SumoEnv either, so the double-wrap case must
        # be named before the class check turns it into a confusing "not a SumoEnv".
        if isinstance(env, AlignedEnv):
            raise TypeError(
                "refusing to wrap an AlignedEnv: the info is already aligned, and align_info "
                "refuses a second application (it would see a 25-wide state where it expects 32). "
                "Raising here rather than at the first reset means no SUMO process and no DT have "
                "been built by the time the mistake is named"
            )
        if not isinstance(env, SumoEnv):
            raise TypeError(
                f"AlignedEnv wraps a SumoEnv; got {type(env).__name__}. The class is what "
                "determines the info's shape, so the class is what is checked -- a config-path "
                "suffix test would accept a CityFlow env handed a .sumocfg and reject nothing"
            )
        self._env = env
        self._alignment = alignment

    # -- the two methods that change -------------------------------------
    def reset(self, **kwargs: Any) -> dict[str, Any]:
        """``align_info(self._env.reset(**kwargs))`` -- info only, per contract C1."""
        return align_info(self._env.reset(**kwargs), self._alignment)

    def step(self, action: Any) -> tuple[Any, bool, bool, dict[str, Any]]:
        """``(reward, terminated, truncated, aligned_info)`` -- reward FIRST, per contract C2.

        The reward, the flags and the step counter are the wrapped env's, untouched: the alignment
        rewrites the observation, never the outcome.
        """
        reward, terminated, truncated, info = self._env.step(action)
        return reward, terminated, truncated, align_info(info, self._alignment)

    def close(self) -> None:
        self._env.close()

    # -- forwarded explicitly (see the module docstring) ------------------
    @property
    def max_steps(self) -> int:
        return self._env.max_steps

    @property
    def intersections(self) -> Any:
        return self._env.intersections

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def recorder(self) -> Any:
        """The observer's per-second recorder, so ``reconstruct_sumo_episode`` never reaches ``_env``.

        ``BRIEF_37`` §3.4.  Without this the attribute still resolves -- ``__getattr__`` forwards it
        -- but it resolves *silently*, and a caller handed a PLAIN ``SumoEnv`` by mistake would get
        ``AttributeError: 'SumoEnv' object has no attribute 'recorder'`` from inside numpy-adjacent
        code, after a SUMO process had started and an episode had run.  ``E_sumo`` exists only
        through the observer (``BRIEF_37`` §0.4), so a cell built on the wrong env produces no
        ``E_sumo`` at all, and that must be said at the point of use, by name.
        """
        recorder = getattr(self._env, "recorder", None)
        if recorder is None:
            raise AttributeError(
                f"the wrapped {type(self._env).__name__} has no recorder: this AlignedEnv was "
                "built over a plain SumoEnv, and E_sumo exists only through the observer. Build it "
                "with offline.aligned_env.aligned_observer_env_for_draw (BRIEF_37 section 3.4), "
                "not with aligned_sumo_env_for_draw, which is the UNOBSERVED door P7.2b used"
            )
        return recorder

    def __getattr__(self, name: str) -> Any:
        # ⚠️ A PROPERTY THAT RAISES AttributeError LANDS HERE.  Python calls __getattr__ whenever
        # normal lookup raises AttributeError -- including from inside a property getter -- so
        # `recorder`'s named refusal was being swallowed and replaced by the wrapped env's generic
        # "'SumoEnv' object has no attribute 'recorder'".  Found by the test, not by reading.
        # Re-raising the descriptor's own error keeps the message that names the fix; done by
        # descriptor type rather than by a name list so a property added later is covered too.
        descriptor = getattr(type(self), name, None)
        if isinstance(descriptor, property) and descriptor.fget is not None:
            return descriptor.fget(self)
        # Only reached when normal lookup fails.  The two own attributes are excluded explicitly:
        # without that, a lookup before __init__ has set them recurses forever.
        if name in {"_env", "_alignment"} or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._env, name)


def declared_alignment(metric_keys: tuple[str, ...] = ()) -> ScenarioAlignment:
    """The alignment for the declared pair, built from the two network FILES (A16(b))."""
    return alignment_for_scenario(
        DECLARED_SCENARIO,
        cityflow_roadnet=DECLARED_CITYFLOW_ROADNET,
        sumo_net=DECLARED_SUMO_NET,
        metric_keys=metric_keys,
    )


def aligned_sumo_env_for_draw(
    scenario_key: str,
    draw_id: int,
    *,
    out_root: str | Path,
    arm: str = "maxpressure",
    sentinel_out_dir: str | Path = "/nonexistent",
) -> AlignedEnv:
    """Build the draw's teleport-free SUMO env through A17(b)'s route and wrap it.

    The route is the one the repo already exercises on SUMO and the one A17(b) names:
    ``collect_style_args`` -> ``offline.collect._build_env_spec`` -> ``experiments.envs.make_env``.
    ``sentinel_out_dir`` satisfies ``collect``'s required ``--out-dir`` and is never opened.
    """
    from experiments.envs import make_env
    from offline.collect import _build_env_spec
    from offline.materialise_draws import parity_sumocfg_path
    from offline.sumo_att_reference import collect_style_args

    config_path = parity_sumocfg_path(scenario_key, draw_id, out_root=out_root)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"draw {draw_id} has no parity configuration at {config_path}; P7.2a materialises the "
            "band into the MAIN tree's scenarios/draws"
        )
    args = collect_style_args(
        "sumo", arm, config_path, sentinel_out_dir=sentinel_out_dir
    )
    return AlignedEnv(make_env(_build_env_spec(args)), declared_alignment())


def observer_env_for_draw(
    scenario_key: str,
    draw_id: int,
    *,
    out_root: str | Path,
    halting_check: bool = False,
    arm: str = "maxpressure",
    sentinel_out_dir: str | Path = "/nonexistent",
) -> Any:
    """The OBSERVED but **UNWRAPPED** env -- what a P7.3a ANCHOR cell runs in (Amendment A2).

    ``make_observer_sumo_env`` mirrors ``experiments.envs.make_env``'s SUMO branch and returns a
    subclass carrying ``.recorder``; ``reconstruct_sumo_episode`` turns that into ``e_sumo``,
    A13(b)'s decomposition and A15(b)'s counts.  ``make_env`` returns the frozen class and cannot
    return a subclass, which is why the observer has its own constructor rather than a flag.

    ⚠️ **Anchors do not go through A16's door, and that was measured.**  ``align_info`` drops
    outgoing lanes and re-keys the survivors to CityFlow ids, while MaxPressure's pressure is a
    difference over the env's own SUMO lane ids, so wrapping it raises ``KeyError`` on an outgoing
    lane (``algorithms/max_pressure.py:142``).  A16 is untouched by that: the door is the only route
    into a *CityFlow-trained model's* frame, and an anchor has no frame to enter.

    ``halting_check`` defaults to **False** -- Amendment C2's convention read the safe way round.
    The cross-check verifies the RECORDER, is value-neutral by construction (A9b: identical
    ``att_env``, ``e_sumo`` and counts, at 3.5x the cost) and runs on a DECLARED SUBSET: every cell
    on draw 1000.  The caller passes ``True`` for that subset; a default of ``True`` would silently
    cost the campaign roughly half its clock.

    ⚠️ **It is set HERE, at construction, and it cannot be set afterwards through the wrapper.**
    :class:`AlignedEnv` forwards attribute *reads* through ``__getattr__`` and defines no
    ``__setattr__``, so P7.1's ``env.halting_check = ...`` (``sumo_att_reference.py:1292``) would set
    the flag on the wrapper and leave the observer untouched.

    **G3: this is the ONE construction path.**  :func:`aligned_observer_env_for_draw` is this
    function plus the wrap, so a setting added here reaches both shapes; two copies of the
    construction would diverge the first time one of them gained a key.
    """
    from offline.collect import _build_env_spec
    from offline.materialise_draws import parity_sumocfg_path
    from offline.sumo_att_reference import collect_style_args, make_observer_sumo_env

    config_path = parity_sumocfg_path(scenario_key, draw_id, out_root=out_root)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"draw {draw_id} has no parity configuration at {config_path}; P7.2a materialises the "
            "band into the MAIN tree's scenarios/draws"
        )
    args = collect_style_args("sumo", arm, config_path, sentinel_out_dir=sentinel_out_dir)
    settings = _build_env_spec(args).settings
    return make_observer_sumo_env(config_path, settings, halting_check=bool(halting_check))


def aligned_observer_env_for_draw(
    scenario_key: str,
    draw_id: int,
    *,
    out_root: str | Path,
    halting_check: bool = False,
    arm: str = "maxpressure",
    sentinel_out_dir: str | Path = "/nonexistent",
) -> AlignedEnv:
    """The OBSERVED, aligned env one P7.3a DT cell runs in (``BRIEF_37`` §3.4).

    ``make_observer_sumo_env`` mirrors ``experiments.envs.make_env``'s SUMO branch and returns a
    subclass carrying ``.recorder``; ``reconstruct_sumo_episode`` turns that into ``e_sumo``, A13(b)'s
    decomposition and A15(b)'s counts.  ``make_env`` returns the frozen class and cannot return a
    subclass, which is why the observer has its own constructor rather than a flag.

    The settings come from the same ``collect_style_args -> _build_env_spec`` route
    :func:`aligned_sumo_env_for_draw` uses, so the observed cell and P7.2b's unobserved smoke differ
    in the observer and in nothing else.

    ⚠️ **Only DT arms come through here.**  The anchors -- fixed-time, MaxPressure, random -- run on
    the observed but **UNWRAPPED** env (``BRIEF_37`` Amendment A2): ``align_info`` drops outgoing
    lanes and re-keys the survivors to CityFlow ids, and MaxPressure's pressure is a difference over
    the env's own SUMO lane ids, so wrapping it raises ``KeyError`` on an outgoing lane.  A16 is
    untouched by that: the door is the only route into a *CityFlow-trained model's* frame, and an
    anchor has no frame to enter.  ``tests/test_aligned_env.py`` pins it.

    ``halting_check`` is C2's declared subset and is set at CONSTRUCTION -- see
    :func:`observer_env_for_draw`, which this function is a wrap around (G3: one construction path,
    not a mirror).
    """
    return AlignedEnv(
        observer_env_for_draw(
            scenario_key,
            draw_id,
            out_root=out_root,
            halting_check=halting_check,
            arm=arm,
            sentinel_out_dir=sentinel_out_dir,
        ),
        declared_alignment(),
    )
