"""Contract C8 made mechanical: a MAPPO checkpoint freezes the env's global metric set.

The acceptance criterion for the whole guard is
:func:`test_same_count_different_keys_raises_naming_the_difference`.  ``MAPPOAgent``
already refuses a checkpoint whose ``global_feature_dim`` differs, but that is a *width*
check, and ``_build_global_features`` orders the critic's global feature block by the
metric **key set**.  Swap metric A for metric B and the width never moves: the critic
reads different semantics under the same indices, with no error at all.  The 2026-08-06
incident was caught only because the count happened to change 3 -> 5.

**These tests require ``docs/patches/mappo_metric_keys_guard.patch`` to be applied.**
``agent/MAPPOAgent.py`` is frozen and permission-denied, so a Claude Code session cannot
apply it; the human does, per CLAUDE.md rule 1.  Until then every test here fails
individually against the real module attribute it needs -- deliberately not a single
module-level ``ImportError``, which would collapse eleven distinct signals into one and
tell a reader nothing about which behaviour is missing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces

import agent.MAPPOAgent as mappo_agent
from agent.MAPPOAgent import IMAPPOAgent

HALTING = "number_of_all_halting_vehicles_for_the_last_time_step_in_simulation"
PATCH_HINT = (
    "docs/patches/mappo_metric_keys_guard.patch is not applied to agent/MAPPOAgent.py "
    "(AUTHORISATION C). A session cannot apply it -- the human must run "
    "`git apply docs/patches/mappo_metric_keys_guard.patch`."
)


# ----------------------------------------------------------------------
# Doubles shaped like metrics/base.py and a traffic env
# ----------------------------------------------------------------------


@dataclass
class _Spec:
    """Stands in for ``metrics.base``'s registry entry."""

    compute: Any


class _FakeMetrics:
    """Mimics the two attributes the derivation reads: ``requested``, ``REGISTERED``.

    ``local_only`` names are registered with ``compute=None``, exactly like a metric
    that has only a per-intersection implementation.  ``compute_all`` filters those
    out, so the derivation must too -- otherwise the guard would compare against a
    key set the env never actually puts in ``info["metrics"]``.
    """

    def __init__(
        self, names: tuple[str, ...], local_only: tuple[str, ...] = ()
    ) -> None:
        self.REGISTERED: dict[str, _Spec] = {}
        for name in names:
            self.REGISTERED[name] = _Spec(compute=lambda _self: 0.0)
        for name in local_only:
            self.REGISTERED[name] = _Spec(compute=None)
        self._requested = list(names) + list(local_only)

    @property
    def requested(self) -> list[str]:
        return list(self._requested)

    def compute_all(self) -> dict[str, float]:
        """The real thing's body, so a test can compare against it rather than a copy."""
        return {
            n: 0.0 for n in self._requested if self.REGISTERED[n].compute is not None
        }


@dataclass
class _Ix:
    id: str
    num_phases: int


class _EnvStub:
    """Minimal C1 env.  ``metrics=None`` models a backend with no metrics pipeline."""

    def __init__(
        self,
        metric_names: tuple[str, ...] | None = (HALTING,),
        n: int = 2,
        local_only: tuple[str, ...] = (),
    ) -> None:
        self.intersections = [_Ix(f"ix{i}", 4) for i in range(n)]
        self.action_space = spaces.MultiDiscrete([4] * n)
        self.max_steps = 10
        self.metrics = (
            None if metric_names is None else _FakeMetrics(metric_names, local_only)
        )


def _info_for(env: _EnvStub) -> dict[str, Any]:
    """A C2-shaped ``info`` carrying exactly the env's metric set."""
    metrics = {} if env.metrics is None else env.metrics.compute_all()
    return {"step": 1, "vehicle_count": 7.0, "metrics": metrics}


def _agent_on(env: _EnvStub) -> IMAPPOAgent:
    """An agent whose ``_global_metric_keys`` is frozen and whose learner is built."""
    agent = IMAPPOAgent(env, rollout_size=8, minibatch_size=2)
    agent._build_global_features(_info_for(env))
    n_metrics = 0 if env.metrics is None else len(env.metrics.compute_all())
    agent.learner.ensure_initialized([3] * len(env.intersections), 2 + n_metrics)
    return agent


def _require_patch() -> None:
    for name in ("env_global_metric_keys", "_assert_checkpoint_metric_keys"):
        assert hasattr(mappo_agent, name), f"{PATCH_HINT} (missing {name})"


# ----------------------------------------------------------------------
# 1. The patch itself
# ----------------------------------------------------------------------


def test_patch_is_applied() -> None:
    """Names the missing piece, so a red run here reads as 'apply the patch'."""
    _require_patch()


# ----------------------------------------------------------------------
# 2. The single shared derivation
# ----------------------------------------------------------------------


def test_env_global_metric_keys_equals_info_metrics_keys() -> None:
    """The derivation must equal ``sorted(info["metrics"])`` by construction."""
    _require_patch()
    env = _EnvStub(metric_names=("b_metric", "a_metric", HALTING))

    derived = mappo_agent.env_global_metric_keys(env)
    assert derived == sorted(_info_for(env)["metrics"].keys())
    assert derived == sorted(["a_metric", "b_metric", HALTING])


def test_env_global_metric_keys_excludes_local_only_metrics() -> None:
    """A ``compute=None`` metric never reaches ``info["metrics"]``, so it must not count.

    Counting it would make the guard compare against a set the env cannot produce,
    turning a correct load into a spurious hard failure.
    """
    _require_patch()
    env = _EnvStub(metric_names=(HALTING,), local_only=("per_ix_only",))

    assert mappo_agent.env_global_metric_keys(env) == [HALTING]
    assert "per_ix_only" not in _info_for(env)["metrics"]


def test_env_without_metrics_pipeline_derives_none_not_empty() -> None:
    """``None`` means 'cannot check'; ``[]`` would mean 'checked, and it is empty'."""
    _require_patch()
    assert mappo_agent.env_global_metric_keys(_EnvStub(metric_names=None)) is None


# ----------------------------------------------------------------------
# 3. THE ACCEPTANCE CRITERION (BRIEF_08 §3)
# ----------------------------------------------------------------------


def test_same_count_different_keys_raises_naming_the_difference(tmp_path: Path) -> None:
    """Same width, different keys -- the case the existing width check cannot see.

    This is the entire reason the patch exists. The two envs have exactly one metric
    each, so ``global_feature_dim`` is 3 on both sides and ``ensure_initialized``
    is satisfied.
    """
    _require_patch()
    trained_on = _EnvStub(metric_names=("metric_alpha",))
    agent = _agent_on(trained_on)
    path = tmp_path / "ckpt.pt"
    agent.save(str(path))

    loaded_against = _EnvStub(metric_names=("metric_beta",))
    fresh = _agent_on(loaded_against)

    # Same count on both sides: the width guard has nothing to complain about.
    assert len(mappo_agent.env_global_metric_keys(trained_on)) == len(
        mappo_agent.env_global_metric_keys(loaded_against)
    )

    with pytest.raises(ValueError, match="symmetric difference") as excinfo:
        fresh.load(str(path))

    message = str(excinfo.value)
    assert "metric_alpha" in message
    assert "metric_beta" in message
    assert "C8" in message
    # The message must end the investigation, not start it: both sets AND the
    # difference, so nobody has to re-derive it months from now.
    assert "checkpoint" in message
    assert "env" in message


def test_the_width_guard_alone_would_not_have_caught_it(tmp_path: Path) -> None:
    """Control: the pre-patch failure mode. Widths agree, so only the key set differs.

    Pins WHY the patch is needed. If this ever fails, the two envs stopped being a
    same-width swap and the acceptance test above is no longer testing its own case.
    """
    _require_patch()
    a = _EnvStub(metric_names=("metric_alpha",))
    b = _EnvStub(metric_names=("metric_beta",))

    agent_a = _agent_on(a)
    agent_b = _agent_on(b)
    assert agent_a.learner.global_feature_dim == agent_b.learner.global_feature_dim
    assert agent_a._global_metric_keys != agent_b._global_metric_keys


# ----------------------------------------------------------------------
# 4. The other four behaviour rows
# ----------------------------------------------------------------------


def test_matching_keys_load_silently(tmp_path: Path) -> None:
    _require_patch()
    agent = _agent_on(_EnvStub(metric_names=(HALTING,)))
    path = tmp_path / "ckpt.pt"
    agent.save(str(path))

    fresh = _agent_on(_EnvStub(metric_names=(HALTING,)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        fresh.load(str(path))
    assert fresh.steps_done == agent.steps_done


def test_pre_migration_checkpoint_warns_loudly_and_still_loads(tmp_path: Path) -> None:
    """The 60 shipped checkpoints predate the field: they must keep working, loudly."""
    _require_patch()
    import torch

    agent = _agent_on(_EnvStub(metric_names=(HALTING,)))
    path = tmp_path / "old.pt"
    agent.save(str(path))

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    del payload["global_metric_keys"]  # exactly a pre-C8 checkpoint
    torch.save(payload, str(path))

    fresh = _agent_on(_EnvStub(metric_names=(HALTING,)))
    with pytest.warns(RuntimeWarning, match="pre-C8 MAPPO checkpoint"):
        fresh.load(str(path))
    assert fresh.steps_done == agent.steps_done


def test_checkpoint_with_null_keys_warns(tmp_path: Path) -> None:
    """Saved before any global feature vector was built: present, but uninformative."""
    _require_patch()
    env = _EnvStub(metric_names=(HALTING,))
    agent = IMAPPOAgent(env, rollout_size=8, minibatch_size=2)
    agent.learner.ensure_initialized([3, 3], 3)
    assert agent._global_metric_keys is None
    path = tmp_path / "unbuilt.pt"
    agent.save(str(path))

    fresh = _agent_on(_EnvStub(metric_names=(HALTING,)))
    with pytest.warns(RuntimeWarning, match="global_metric_keys=None"):
        fresh.load(str(path))


def test_env_without_metrics_warns_rather_than_asserting(tmp_path: Path) -> None:
    """Cannot check is not the same as matches; it must warn, never pass silently.

    Both sides run no metrics pipeline, so ``global_feature_dim`` is 2 on each and the
    pre-existing width guard has nothing to say -- which is what isolates row 3 of the
    behaviour table. A checkpoint from a *1-metric* env would additionally trip the
    width guard (2 vs 3) and the test would no longer be about the warning.
    """
    _require_patch()
    agent = _agent_on(_EnvStub(metric_names=None))
    path = tmp_path / "ckpt.pt"
    agent.save(str(path))

    fresh = _agent_on(_EnvStub(metric_names=None))
    with pytest.warns(RuntimeWarning, match="no metrics pipeline"):
        fresh.load(str(path))


def test_existing_width_guard_still_fires_on_a_pre_migration_checkpoint(
    tmp_path: Path,
) -> None:
    """The patch must not weaken what already worked.

    Demonstrated on a PRE-MIGRATION checkpoint, which is the only case where the width
    guard can still be the first thing to fail: with no recorded key set the C8 check
    can only warn, so control reaches ``load_state`` exactly as it did before the patch.
    """
    _require_patch()
    import torch

    agent = _agent_on(_EnvStub(metric_names=(HALTING,)))
    path = tmp_path / "ckpt.pt"
    agent.save(str(path))
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    del payload["global_metric_keys"]
    torch.save(payload, str(path))

    fresh = _agent_on(_EnvStub(metric_names=(HALTING, "extra_one", "extra_two")))
    with pytest.warns(RuntimeWarning, match="pre-C8 MAPPO checkpoint"):
        with pytest.raises(ValueError, match="Global feature size changed for MAPPO"):
            fresh.load(str(path))


def test_c8_check_pre_empts_the_width_guard_on_a_migrated_checkpoint(
    tmp_path: Path,
) -> None:
    """A different COUNT also means different KEYS, so C8 reports it first.

    Deliberate, and the better outcome: the C8 message names the symmetric difference,
    while ``Global feature size changed for MAPPO: expected 3, got 5`` sends the reader
    back into the code -- which is what happened on 2026-08-06 and cost 2400 episodes.
    The ordering follows from checking before any state is adopted.
    """
    _require_patch()
    agent = _agent_on(_EnvStub(metric_names=(HALTING,)))
    path = tmp_path / "ckpt.pt"
    agent.save(str(path))

    fresh = _agent_on(_EnvStub(metric_names=(HALTING, "extra_one", "extra_two")))
    with pytest.raises(ValueError, match="symmetric difference") as excinfo:
        fresh.load(str(path))
    assert "extra_one" in str(excinfo.value)
    assert "Global feature size changed" not in str(excinfo.value)


# ----------------------------------------------------------------------
# 5. Ordering and non-adoption
# ----------------------------------------------------------------------


def test_rejected_checkpoint_leaves_the_agent_untouched(tmp_path: Path) -> None:
    """The mutation barrier, applied to in-memory state.

    The C8 check runs before ``steps_done`` or the learner tensors are adopted, so a
    refused load leaves a usable agent rather than a half-updated one.
    """
    _require_patch()
    trained = _agent_on(_EnvStub(metric_names=("metric_alpha",)))
    trained.steps_done = 4242
    path = tmp_path / "ckpt.pt"
    trained.save(str(path))

    fresh = _agent_on(_EnvStub(metric_names=("metric_beta",)))
    fresh.steps_done = 7
    before = [p.detach().clone() for p in fresh.learner.critic.parameters()]

    with pytest.raises(ValueError, match="symmetric difference"):
        fresh.load(str(path))

    assert fresh.steps_done == 7, "steps_done was adopted from a rejected checkpoint"
    after = list(fresh.learner.critic.parameters())
    assert all(
        bool((b == a).all()) for b, a in zip(before, after)
    ), "critic weights were adopted from a rejected checkpoint"


def test_load_does_not_adopt_the_checkpoints_metric_keys(tmp_path: Path) -> None:
    """A successful load must leave ``_global_metric_keys`` unset, not seeded.

    Adopting them looks harmless because the sets are equal by the time we get here.
    It is not: ``_build_global_features`` reads ``metrics.get(key, 0.0)``, so an
    adopted key absent from a later env silently becomes 0.0 at an unchanged width --
    invisible to the width guard and to the C8 check, which has already returned.
    Leaving it ``None`` makes the agent freeze from the env's own ``info``.
    """
    _require_patch()
    agent = _agent_on(_EnvStub(metric_names=(HALTING,)))
    path = tmp_path / "ckpt.pt"
    agent.save(str(path))

    env = _EnvStub(metric_names=(HALTING,))
    fresh = IMAPPOAgent(env, rollout_size=8, minibatch_size=2)
    fresh.learner.ensure_initialized([3, 3], 3)
    fresh.load(str(path))

    assert fresh._global_metric_keys is None


def test_save_records_the_metric_keys(tmp_path: Path) -> None:
    """The write side, read back raw rather than through ``load``."""
    _require_patch()
    import torch

    agent = _agent_on(_EnvStub(metric_names=("b_metric", "a_metric")))
    path = tmp_path / "ckpt.pt"
    agent.save(str(path))

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    assert payload["global_metric_keys"] == ["a_metric", "b_metric"]
    # The pre-existing keys are untouched -- v1.1 of the checkpoint is additive.
    assert set(payload) == {"steps_done", "learner", "global_metric_keys"}
