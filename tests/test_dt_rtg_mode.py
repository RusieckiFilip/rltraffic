"""Tests for ``DTConfig.rtg_mode`` -- P5.3a's information ablation of the return prompt.

``rtg_mode`` is a **config field and not a subclass**, ruled in ``BRIEF_28`` section 4.1.  The
decisive reason is ``agent/DTAgent.py:801``: :meth:`DTAgent.load` reconstructs the **base**
``DecisionTransformer``, so a ``NoRTGDecisionTransformer`` subclass would write a checkpoint with
identical ``state_dict`` keys and shapes that the ordinary loader would rebuild as a *conditioned*
model and evaluate happily -- a plausible number from the wrong model.  With a config field the mode
travels inside the checkpointed config and the right model is rebuilt automatically.

What this file covers, and what it deliberately does not
--------------------------------------------------------
Everything here runs on tiny randomly initialised models and a stub env: no simulator, no corpus, no
checkpoint from ``output/``.  The claim that **the default path did not move** is *not* made here --
that is ``tests/test_rtg_mode_identity.py``, which re-rolls two committed ``dt`` cells bit-exactly.
These two files are complementary and neither is sufficient alone; ``docs/plans/p5.3a.md`` section 3
lists the seven things the identity test cannot see, and the first of them is exactly the branch
coverage this file supplies.

⭐ **The positive control in
:func:`test_a_conditioned_model_does_respond_to_the_target_so_the_instrument_can_see_a_difference`
is not optional.**  Without it a measured ``flip_rate`` of 0 on a trained model is uninterpretable:
it could mean the token is inert, or it could mean the harness returns the same answer for every
input.  This project has shipped the second failure before (the 2026-08-03 harness that returned
``BLOCKED`` for every path including its control).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces

from agent.DTAgent import DecisionTransformer, DTAgent, DTConfig

STATE_DIM = 4
N_ACTIONS = 3
CONTEXT = 4
MAX_EP_LEN = 16

#: Two targets far enough apart that a responsive model cannot fail to notice, and both inside the
#: span P4.3 actually swept (``DECLARED_GRID`` runs 0 to -13000).
TARGET_LOW = -13000.0
TARGET_HIGH = 0.0
RTG_SCALE = 9991.0

#: The eight keys ``to_json_obj`` emitted before P5.3a; every one of the 225 committed checkpoints
#: carries exactly these and no ``rtg_mode``.
LEGACY_CONFIG_KEYS = (
    "state_dim",
    "n_actions",
    "context_length",
    "n_layer",
    "n_head",
    "d_model",
    "dropout",
    "max_ep_len",
)


# ----------------------------------------------------------------------
# Fixtures: a tiny model and the smallest env DTAgent will accept
# ----------------------------------------------------------------------


class _StubIntersection:
    def __init__(self, ix_id: str, n_actions: int) -> None:
        self.id = ix_id
        self.n_actions = n_actions
        self.incoming_lanes: list[str] = []


class _StubEnv:
    """The smallest object satisfying what ``DTAgent.__init__`` reads from an env."""

    def __init__(self, specs: Sequence[tuple[str, int]] = (("ix_only", N_ACTIONS),)) -> None:
        self.intersections = [_StubIntersection(ix_id, n) for ix_id, n in specs]
        self.max_steps = 8
        self.action_space = gym_spaces.Discrete(specs[0][1])


def _config(**overrides: Any) -> DTConfig:
    """A deliberately tiny architecture: these tests check wiring, not capacity."""
    params: dict[str, Any] = {
        "state_dim": STATE_DIM,
        "n_actions": N_ACTIONS,
        "context_length": CONTEXT,
        "n_layer": 2,
        "n_head": 1,
        "d_model": 16,
        "dropout": 0.0,
        "max_ep_len": MAX_EP_LEN,
    }
    params.update(overrides)
    return DTConfig(**params)


def _agent(**overrides: Any) -> DTAgent:
    params: dict[str, Any] = {
        "context_length": CONTEXT,
        "n_layer": 2,
        "n_head": 1,
        "d_model": 16,
        "dropout": 0.0,
        "max_ep_len": MAX_EP_LEN,
        "target_rtg": TARGET_LOW,
        "rtg_scale": RTG_SCALE,
        "device": "cpu",
        "seed": 5,
        "state_dim": STATE_DIM,
    }
    params.update(overrides)
    return DTAgent(_StubEnv(), **params)


def _batch(seed: int = 11) -> dict[str, torch.Tensor]:
    """One deterministic ``(B, K)`` batch of everything ``forward`` needs except the RTG."""
    rng = np.random.default_rng(seed)
    batch, steps = 2, CONTEXT
    return {
        "state": torch.from_numpy(
            rng.standard_normal((batch, steps, STATE_DIM)).astype(np.float32)
        ),
        "action": torch.from_numpy(
            rng.integers(0, N_ACTIONS, size=(batch, steps)).astype(np.int64)
        ),
        "timestep": torch.from_numpy(
            np.tile(np.arange(steps, dtype=np.int64), (batch, 1))
        ),
        "attention_mask": torch.ones((batch, steps), dtype=torch.bool),
        "avail_mask": torch.ones((batch, steps, N_ACTIONS), dtype=torch.bool),
    }


def _rtg_for(target: float, batch: dict[str, torch.Tensor], rewards: np.ndarray) -> torch.Tensor:
    """The scaled ``(B, K, 1)`` RTG a caller would build for *target*, as ``DTAgent`` builds it."""
    size = batch["state"].shape[:2]
    consumed = np.concatenate([[0.0], np.cumsum(rewards, dtype=np.float64)[:-1]])
    series = (target - consumed) / RTG_SCALE
    tiled = np.tile(series.astype(np.float32), (size[0], 1)).reshape(size[0], size[1], 1)
    return torch.from_numpy(tiled)


def _forward(model: DecisionTransformer, batch: dict[str, torch.Tensor], rtg: torch.Tensor):
    return model(
        rtg,
        batch["state"],
        batch["action"],
        batch["timestep"],
        batch["attention_mask"],
        batch["avail_mask"],
    )


REWARDS = np.array([-3.0, -4.5, -0.25, -8.0], dtype=np.float64)


# ----------------------------------------------------------------------
# 1-6: the field itself
# ----------------------------------------------------------------------


def test_rtg_mode_defaults_to_conditioned() -> None:
    """The default is today's behaviour; nothing changes for a caller that says nothing."""
    assert _config().rtg_mode == "conditioned"
    assert DTConfig(state_dim=3, n_actions=2).rtg_mode == "conditioned"


def test_an_illegal_rtg_mode_is_refused_and_the_message_names_both_legal_values() -> None:
    """Naming both is the difference between a usable error and a puzzle."""
    with pytest.raises(ValueError, match="rtg_mode") as excinfo:
        _config(rtg_mode="shuffled")
    message = str(excinfo.value)
    assert "conditioned" in message, message
    assert "zero" in message, message
    assert "shuffled" in message, message


def test_to_json_obj_always_emits_rtg_mode() -> None:
    """A checkpoint written from now on records its own mode; only older ones can be silent."""
    payload = _config(rtg_mode="zero").to_json_obj()
    assert payload["rtg_mode"] == "zero"
    assert set(payload) == set(LEGACY_CONFIG_KEYS) | {"rtg_mode"}
    assert len(payload) == 9


def test_a_legacy_eight_key_payload_loads_as_conditioned() -> None:
    """The 225-checkpoint case: absent key means the behaviour those checkpoints were trained with.

    ``SpatialDTConfig.from_json_obj`` hard-raises on an absent ``spatial_mixing``
    (``agent/SpatialDTAgent.py:174-176``) and this defaults instead.  That asymmetry is deliberate
    and is the reason this test exists rather than being an oversight: ``SpatialDTConfig`` was born
    with its flag, ``DTConfig`` has 225 checkpoints in the wild that predate this one.
    """
    legacy = _config().to_json_obj()
    del legacy["rtg_mode"]
    assert set(legacy) == set(LEGACY_CONFIG_KEYS)

    restored = DTConfig.from_json_obj(legacy)
    assert restored.rtg_mode == "conditioned"


def test_a_nine_key_payload_round_trips_its_mode() -> None:
    for mode in ("conditioned", "zero"):
        payload = _config(rtg_mode=mode).to_json_obj()
        assert DTConfig.from_json_obj(payload).rtg_mode == mode
        assert DTConfig.from_json_obj(payload) == _config(rtg_mode=mode)


def test_an_ill_shaped_rtg_still_raises_under_zero_mode() -> None:
    """Validate first, then substitute -- or a shape bug hides behind the ablation."""
    model = DecisionTransformer(_config(rtg_mode="zero")).eval()
    batch = _batch()
    wrong = torch.zeros((2, CONTEXT, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match=r"rtg must be \(B, K, 1\)"):
        _forward(model, batch, wrong)


# ----------------------------------------------------------------------
# 7-8: BRIEF_28 section 6.2 -- the arm-validity pair.  BOTH must hold.
# ----------------------------------------------------------------------


def test_a_zero_model_is_bit_identical_under_two_very_different_targets() -> None:
    """The ablation arm must actually carry no return information.  Exact equality, not a tolerance."""
    model = DecisionTransformer(_config(rtg_mode="zero")).eval()
    batch = _batch()
    with torch.no_grad():
        low = _forward(model, batch, _rtg_for(TARGET_LOW, batch, REWARDS))
        high = _forward(model, batch, _rtg_for(TARGET_HIGH, batch, REWARDS))
    assert torch.equal(low, high)


def test_a_conditioned_model_does_respond_to_the_target_so_the_instrument_can_see_a_difference() -> None:
    """⭐ THE POSITIVE CONTROL.  Without it, ``flip_rate = 0`` on a trained model means nothing.

    A randomly initialised conditioned model has no reason to ignore its RTG input, so if these
    logits came out equal the harness -- not the model -- would be the thing that is inert.
    """
    torch.manual_seed(20260825)
    model = DecisionTransformer(_config(rtg_mode="conditioned")).eval()
    batch = _batch()
    with torch.no_grad():
        low = _forward(model, batch, _rtg_for(TARGET_LOW, batch, REWARDS))
        high = _forward(model, batch, _rtg_for(TARGET_HIGH, batch, REWARDS))
    assert not torch.equal(low, high)
    # Not merely unequal at the last bit: the instrument must see a difference worth reporting.
    assert float((low - high).abs().max()) > 1e-3


# ----------------------------------------------------------------------
# 9-11: the mode reaches the agent, the checkpoint and nothing else
# ----------------------------------------------------------------------


def test_the_agent_constructor_carries_rtg_mode_into_the_built_model() -> None:
    agent = _agent(rtg_mode="zero")
    assert agent.config.rtg_mode == "zero"
    assert agent.model is not None
    assert agent.model.config.rtg_mode == "zero"


def test_a_checkpoint_round_trips_rtg_mode_and_rebuilds_a_zeroing_model(tmp_path: Path) -> None:
    """A regression guard, and the docstring says so because it is NOT a discriminating test.

    ``BRIEF_28`` A3 required ``from_checkpoint`` to pass ``rtg_mode=config.rtg_mode`` into the
    constructor.  This test passes **with and without** that line, because ``from_checkpoint``
    builds the model twice (``agent/DTAgent.py:465-466`` then ``:801``) and the second build
    replaces the first.  What it does guard is the part that is load-bearing: the mode survives
    ``save`` -> ``load`` inside the checkpointed config, so the ordinary loader cannot rebuild a
    zero-mode checkpoint as a conditioned model.
    """
    path = tmp_path / "zero.pt"
    _agent(rtg_mode="zero").save(str(path))

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["config"]["rtg_mode"] == "zero"

    restored = DTAgent.from_checkpoint(_StubEnv(), str(path), device="cpu")
    assert restored.config.rtg_mode == "zero"
    assert restored.model is not None
    assert restored.model.config.rtg_mode == "zero"

    batch = _batch()
    with torch.no_grad():
        low = _forward(restored.model, batch, _rtg_for(TARGET_LOW, batch, REWARDS))
        high = _forward(restored.model, batch, _rtg_for(TARGET_HIGH, batch, REWARDS))
    assert torch.equal(low, high)


def test_zero_mode_is_an_information_ablation_with_the_architecture_held_exactly_fixed() -> None:
    """Same weights, same shapes, same token count: the only difference is what the RTG carries.

    This is the whole argument for ``rtg_mode="zero"`` over a two-token variant.  Dropping the RTG
    token would change the sequence length, the attention pattern and the state-token index at
    ``agent/DTAgent.py:364`` -- the alignment convention every merged DT number depends on.
    """
    torch.manual_seed(20260825)
    conditioned = DecisionTransformer(_config(rtg_mode="conditioned")).eval()
    zeroed = DecisionTransformer(_config(rtg_mode="zero")).eval()
    assert conditioned.state_dict().keys() == zeroed.state_dict().keys()
    zeroed.load_state_dict(conditioned.state_dict())

    batch = _batch()
    zeros = torch.zeros((2, CONTEXT, 1), dtype=torch.float32)
    with torch.no_grad():
        # The conditioned model fed a constant-zero RTG sees exactly what the zero model
        # manufactures for itself, so the two must agree bit for bit.
        assert torch.equal(
            _forward(conditioned, batch, zeros),
            _forward(zeroed, batch, _rtg_for(TARGET_LOW, batch, REWARDS)),
        )
