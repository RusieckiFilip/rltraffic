"""Tests for ``agent.DTAgent`` -- the causal Decision Transformer and its ``BaseAgent`` face.

No corpus and no simulator: the loader-backed tests write their own format-v1.0 episode files
through the fixture builder in ``tests/test_offline_dataset.py``, so a *real*
``TrajectoryWindowDataset`` batch reaches the model.  That fixture is reused deliberately
rather than re-implemented -- it is built so wrong implementations are visible (two
intersections with different ``(state_dim, n_actions)``, env order not equal to sorted order,
and an ``avail_mask`` that genuinely binds, which the real corpus's all-True masks cannot test).

Every assertion here is exact where the types allow it.  Where a float64 cross-check is used
instead, the test measures and asserts the margin rather than accepting "it passed".
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from gymnasium import spaces as gym_spaces

from agent.DTAgent import (
    CHECKPOINT_FORMAT_VERSION,
    DecisionTransformer,
    DTAgent,
    DTConfig,
    action_loss,
    masked_action_logits,
)
from offline.dataset import PAD_ACTION, TrajectoryWindowDataset, collate_windows

from tests.test_offline_dataset import write_dataset_dir

# The fixture's second intersection: state_dim 4, n_actions 3, T = 8 decision rows.
FIXTURE_STATE_DIM = 4
FIXTURE_N_ACTIONS = 3
FIXTURE_GROUP = (FIXTURE_STATE_DIM, FIXTURE_N_ACTIONS)
CONTEXT = 4
MAX_EP_LEN = 16


def _small_config(**overrides: Any) -> DTConfig:
    """A deliberately tiny architecture: these tests check wiring, not capacity."""
    params: dict[str, Any] = {
        "state_dim": FIXTURE_STATE_DIM,
        "n_actions": FIXTURE_N_ACTIONS,
        "context_length": CONTEXT,
        "n_layer": 2,
        "n_head": 1,
        "d_model": 16,
        "dropout": 0.0,
        "max_ep_len": MAX_EP_LEN,
    }
    params.update(overrides)
    return DTConfig(**params)


@pytest.fixture()
def loader_batch(tmp_path: Path) -> dict[str, torch.Tensor]:
    """A collated batch of REAL loader items, all from the ``(4, 3)`` group."""
    dataset_dir = write_dataset_dir(tmp_path, "fixture__policy")
    dataset = TrajectoryWindowDataset([dataset_dir], context_length=CONTEXT, split="train")
    indices = dataset.groups[FIXTURE_GROUP]
    assert len(indices) >= 8, f"fixture must supply a real batch, got {len(indices)} items"
    return collate_windows([dataset[i] for i in indices[:8]])


def _forward(model: DecisionTransformer, batch: dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
    return model(
        batch["rtg"],
        batch["state"],
        batch["action"],
        batch["timestep"],
        batch["attention_mask"],
        kwargs.get("avail_mask", None),
    )


# ----------------------------------------------------------------------
# Shape and alignment through the model (BRIEF_10 section 5, item 1)
# ----------------------------------------------------------------------


def test_real_loader_batch_reaches_a_loss_without_a_shape_error(
    loader_batch: dict[str, torch.Tensor],
) -> None:
    model = DecisionTransformer(_small_config()).eval()
    logits = _forward(model, loader_batch, avail_mask=loader_batch["avail_mask"])
    assert tuple(logits.shape) == (8, CONTEXT, FIXTURE_N_ACTIONS)
    loss = action_loss(logits, loader_batch["action"])
    assert loss.ndim == 0
    assert bool(torch.isfinite(loss))


def test_pad_action_marks_exactly_the_padded_positions_on_every_item(
    loader_batch: dict[str, torch.Tensor],
) -> None:
    """``(action == -1) == ~attention_mask`` -- the invariant the loss's ignore_index rests on."""
    is_pad = loader_batch["action"] == PAD_ACTION
    expected = ~loader_batch["attention_mask"]
    assert torch.equal(is_pad, expected)
    # The batch must actually contain padding, or the assertion above is vacuous.
    assert int(is_pad.sum()) > 0


# ----------------------------------------------------------------------
# Padded positions cannot contribute to the loss (BRIEF_10 section 5, item 2)
# ----------------------------------------------------------------------


def test_padded_positions_cannot_move_the_loss(loader_batch: dict[str, torch.Tensor]) -> None:
    """Randomising the logits AT padded positions leaves the loss bit-identical.

    Exact ``==`` rather than a tolerance: the padded positions either enter the reduction or
    they do not, and a float64 re-derivation would only blur that.
    """
    model = DecisionTransformer(_small_config()).eval()
    with torch.no_grad():
        logits = _forward(model, loader_batch)
    baseline = action_loss(logits, loader_batch["action"])

    padded = (loader_batch["action"] == PAD_ACTION).unsqueeze(-1)
    generator = torch.Generator().manual_seed(4242)
    noise = torch.randn(logits.shape, generator=generator) * 100.0
    perturbed = torch.where(padded, noise, logits)
    assert not torch.equal(perturbed, logits), "the perturbation must actually change something"

    assert action_loss(perturbed, loader_batch["action"]) == baseline


def test_loss_equals_an_independent_float64_average_over_the_valid_positions(
    loader_batch: dict[str, torch.Tensor],
) -> None:
    """Second route: ``math.fsum`` of ``-log_softmax`` over valid positions, in float64.

    The margin against the WRONG answer (padded targets folded in as action 0) is measured and
    asserted, so a check that passes by being insensitive is visible as such.
    """
    model = DecisionTransformer(_small_config()).eval()
    with torch.no_grad():
        logits = _forward(model, loader_batch)
    action = loader_batch["action"]
    log_probs = torch.log_softmax(logits.double(), dim=-1)

    valid = [
        -float(log_probs[b, k, int(action[b, k])])
        for b in range(action.shape[0])
        for k in range(action.shape[1])
        if int(action[b, k]) != PAD_ACTION
    ]
    manual = math.fsum(valid) / len(valid)
    measured = float(action_loss(logits, action))
    assert abs(measured - manual) < 1e-6, f"{measured} vs {manual}"

    folded = [
        -float(log_probs[b, k, max(int(action[b, k]), 0)])
        for b in range(action.shape[0])
        for k in range(action.shape[1])
    ]
    wrong = math.fsum(folded) / len(folded)
    margin = abs(wrong - manual)
    assert margin > 1e-3, f"discriminating power too small to certify anything: {margin}"


def test_minus_one_target_is_a_tripwire_without_ignore_index(
    loader_batch: dict[str, torch.Tensor],
) -> None:
    """Dropping ``ignore_index`` does not silently train on action-0 targets -- it raises."""
    model = DecisionTransformer(_small_config()).eval()
    with torch.no_grad():
        logits = _forward(model, loader_batch)
    flat = logits.reshape(-1, FIXTURE_N_ACTIONS)
    target = loader_batch["action"].reshape(-1)
    with pytest.raises(IndexError, match="Target -1 is out of bounds"):
        F.cross_entropy(flat, target)


# ----------------------------------------------------------------------
# Action masking (BRIEF_10 section 5, item 3) -- synthetic masks, since the corpus is all-True
# ----------------------------------------------------------------------


def test_illegal_actions_are_minus_inf_and_carry_zero_probability() -> None:
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(2, 3, 4, generator=generator)
    avail = torch.tensor(
        [
            [[True, False, True, False], [False, True, False, False], [True, True, True, True]],
            [[False, False, True, False], [True, False, False, True], [True, False, True, False]],
        ]
    )
    masked = masked_action_logits(logits, avail)

    assert torch.equal(torch.isneginf(masked), ~avail)
    assert torch.equal(masked[avail], logits[avail])
    probs = torch.softmax(masked, dim=-1)
    assert torch.equal(probs[~avail], torch.zeros_like(probs[~avail]))


def test_argmax_never_selects_an_illegal_action_over_randomised_masks() -> None:
    generator = torch.Generator().manual_seed(7)
    for _ in range(64):
        logits = torch.randn(1, 1, 5, generator=generator)
        avail = torch.rand(1, 1, 5, generator=generator) > 0.5
        if not bool(avail.any()):
            avail[0, 0, 0] = True
        chosen = int(torch.argmax(masked_action_logits(logits, avail), dim=-1)[0, 0])
        assert bool(avail[0, 0, chosen])


def test_a_row_with_no_legal_action_passes_through_unmasked() -> None:
    """NaN trap B: an all-False row is PADDING, and masking it would build an all -inf row."""
    logits = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    avail = torch.tensor([[[False, False, False], [True, False, True]]])
    masked = masked_action_logits(logits, avail)

    assert torch.equal(masked[0, 0], logits[0, 0])
    assert bool(torch.isfinite(torch.softmax(masked[0, 0], dim=-1)).all())
    assert torch.equal(torch.isneginf(masked[0, 1]), torch.tensor([False, True, False]))


# ----------------------------------------------------------------------
# NaN trap A: fully padded query positions
# ----------------------------------------------------------------------


def test_padded_item_gives_finite_loss_and_finite_gradients(
    loader_batch: dict[str, torch.Tensor],
) -> None:
    """The t=0 item is padded on K-1 of K positions; its gradients must still be finite.

    A softmax over an all -inf attention row is NaN, and ``ignore_index`` does not stop that
    NaN reaching the backward pass as ``0 * NaN``.
    """
    single = {key: value[:1] for key, value in loader_batch.items()}
    assert int(single["attention_mask"].sum()) == 1, "this item must be the maximally padded one"

    model = DecisionTransformer(_small_config())
    logits = _forward(model, single, avail_mask=single["avail_mask"])
    assert bool(torch.isfinite(logits).all())
    loss = action_loss(logits, single["action"])
    assert bool(torch.isfinite(loss))

    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"no gradient for {name}"
        assert bool(torch.isfinite(parameter.grad).all()), f"non-finite gradient in {name}"


def test_an_all_false_attention_mask_still_produces_finite_output() -> None:
    """The degenerate case the always-attendable diagonal exists for."""
    config = _small_config()
    model = DecisionTransformer(config).eval()
    batch = {
        "rtg": torch.zeros(1, CONTEXT, 1),
        "state": torch.zeros(1, CONTEXT, FIXTURE_STATE_DIM),
        "action": torch.full((1, CONTEXT), PAD_ACTION, dtype=torch.int64),
        "timestep": torch.zeros(1, CONTEXT, dtype=torch.int64),
        "attention_mask": torch.zeros(1, CONTEXT, dtype=torch.bool),
    }
    with torch.no_grad():
        logits = _forward(model, batch)
    assert bool(torch.isfinite(logits).all())


# ----------------------------------------------------------------------
# Causality: the action slot of the current step cannot leak into its own prediction
# ----------------------------------------------------------------------


def test_the_last_action_input_cannot_change_the_last_prediction(
    loader_batch: dict[str, torch.Tensor],
) -> None:
    model = DecisionTransformer(_small_config()).eval()
    with torch.no_grad():
        base = _forward(model, loader_batch)
        altered = dict(loader_batch)
        action = loader_batch["action"].clone()
        action[:, -1] = (action[:, -1] + 1) % FIXTURE_N_ACTIONS
        altered["action"] = action
        shifted = _forward(model, altered)

    assert torch.equal(base[:, -1], shifted[:, -1])
    assert not torch.equal(action, loader_batch["action"]), "the perturbation must be real"


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


def _train_a_few_steps(seed: int, batch: dict[str, torch.Tensor], steps: int = 5) -> dict[str, bytes]:
    torch.manual_seed(seed)
    model = DecisionTransformer(_small_config(dropout=0.1))
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(steps):
        logits = _forward(model, batch, avail_mask=batch["avail_mask"])
        loss = action_loss(logits, batch["action"])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return {
        name: parameter.detach().cpu().numpy().tobytes()
        for name, parameter in model.named_parameters()
    }


def test_same_seed_gives_byte_identical_parameters(loader_batch: dict[str, torch.Tensor]) -> None:
    first = _train_a_few_steps(1234, loader_batch)
    second = _train_a_few_steps(1234, loader_batch)
    assert first == second


def test_a_different_seed_gives_different_parameters(loader_batch: dict[str, torch.Tensor]) -> None:
    """The control that lets the determinism test above be capable of failing."""
    assert _train_a_few_steps(1234, loader_batch) != _train_a_few_steps(4321, loader_batch)


# ----------------------------------------------------------------------
# The BaseAgent face
# ----------------------------------------------------------------------


class _StubIntersection:
    def __init__(self, ix_id: str, num_phases: int) -> None:
        self.id = ix_id
        self.num_phases = num_phases


class _StubEnv:
    """The smallest object satisfying what ``DTAgent.__init__`` reads from an env.

    The two-intersection form deliberately stores them in NON-sorted order, so an
    implementation that iterates ``sorted(info["intersections"])`` returns the actions in the
    wrong order and the test fails rather than passing by coincidence.
    """

    def __init__(self, specs: Sequence[tuple[str, int]], max_steps: int = 8) -> None:
        self.intersections = [_StubIntersection(ix_id, n) for ix_id, n in specs]
        self.max_steps = max_steps
        if len(specs) == 1:
            self.action_space = gym_spaces.Discrete(specs[0][1])
        else:
            self.action_space = gym_spaces.MultiDiscrete([n for _, n in specs])


def _info(
    step: int,
    payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {"step": step, "vehicle_count": 0, "intersections": payloads}


def _payload(
    state: Sequence[float], avail: Sequence[int], reward: float = 0.0
) -> dict[str, Any]:
    return {"state": list(state), "avail_actions": list(avail), "reward": reward}


def _single_env() -> _StubEnv:
    return _StubEnv([("ix_only", FIXTURE_N_ACTIONS)])


def _agent(env: _StubEnv, **overrides: Any) -> DTAgent:
    params: dict[str, Any] = {
        "context_length": CONTEXT,
        "n_layer": 2,
        "n_head": 1,
        "d_model": 16,
        "dropout": 0.0,
        "max_ep_len": MAX_EP_LEN,
        "target_rtg": -100.0,
        "rtg_scale": 100.0,
        "device": "cpu",
        "seed": 5,
    }
    params.update(overrides)
    return DTAgent(env, **params)


def test_act_returns_one_int64_action_per_intersection_in_env_order() -> None:
    env = _StubEnv([("ix_zulu", 3), ("ix_alpha", 3)])
    agent = _agent(env)
    info = _info(
        0,
        {
            "ix_zulu": _payload([1.0, 2.0, 3.0, 4.0], [0, 1, 2]),
            "ix_alpha": _payload([5.0, 6.0, 7.0, 8.0], [1]),
        },
    )
    action = agent.act(info, explore=False)

    assert isinstance(action, np.ndarray)
    assert action.dtype == np.int64
    assert action.shape == (2,)
    # ix_alpha is SECOND in env order and is the restricted one; a sorted-order
    # implementation would put its forced 1 in slot 0.
    assert int(action[1]) == 1


def test_act_respects_avail_actions_over_many_steps() -> None:
    env = _single_env()
    agent = _agent(env)
    for step in range(6):
        info = _info(step, {"ix_only": _payload([1.0, 2.0, 3.0, 4.0], [2], reward=-1.0)})
        assert int(agent.act(info, explore=False)[0]) == 2


def test_return_to_go_follows_the_observed_rewards() -> None:
    """Independent route: ``RTG_t == target - fsum(r_0 .. r_{t-1})``, exact in float64."""
    env = _single_env()
    agent = _agent(env, target_rtg=-100.0)
    rewards = [-3.0, -4.0, -0.5, -2.25, -7.0]

    agent.act(_info(0, {"ix_only": _payload([0.0, 0.0, 0.0, 0.0], [0, 1, 2], reward=-0.0)}))
    assert agent.current_rtg()["ix_only"] == -100.0

    for step, reward in enumerate(rewards, start=1):
        agent.act(_info(step, {"ix_only": _payload([1.0, 1.0, 1.0, 1.0], [0, 1, 2], reward=reward)}))
        expected = -100.0 - math.fsum(rewards[:step])
        assert agent.current_rtg()["ix_only"] == expected


def test_observe_does_not_advance_the_return_to_go() -> None:
    """One route only: two would double-decrement, and the number would still look plausible."""
    env = _single_env()
    agent = _agent(env)
    info = _info(0, {"ix_only": _payload([0.0, 0.0, 0.0, 0.0], [0, 1, 2])})
    agent.act(info)
    before = agent.current_rtg()["ix_only"]

    next_info = _info(1, {"ix_only": _payload([1.0, 1.0, 1.0, 1.0], [0, 1, 2], reward=-9.0)})
    agent.observe(next_info, -9.0, False, False)

    assert agent.current_rtg()["ix_only"] == before


def test_step_zero_resets_the_context_mid_run() -> None:
    env = _single_env()
    agent = _agent(env)
    for step in range(3):
        agent.act(_info(step, {"ix_only": _payload([1.0, 1.0, 1.0, 1.0], [0, 1, 2], reward=-5.0)}))
    assert agent.current_rtg()["ix_only"] != -100.0

    agent.act(_info(0, {"ix_only": _payload([1.0, 1.0, 1.0, 1.0], [0, 1, 2], reward=-5.0)}))
    assert agent.current_rtg()["ix_only"] == -100.0


def test_update_memory_false_leaves_the_context_untouched() -> None:
    env = _single_env()
    agent = _agent(env)
    agent.act(_info(0, {"ix_only": _payload([1.0, 1.0, 1.0, 1.0], [0, 1, 2])}))
    info = _info(1, {"ix_only": _payload([2.0, 2.0, 2.0, 2.0], [0, 1, 2], reward=-6.0)})

    first = agent.act(info, explore=False, update_memory=False)
    second = agent.act(info, explore=False, update_memory=False)

    assert agent.current_rtg()["ix_only"] == -100.0
    assert np.array_equal(first, second)


def test_a_missing_reward_key_raises_instead_of_freezing_the_return_to_go() -> None:
    """A frozen RTG still produces plausible actions and a plausible number."""
    env = _single_env()
    agent = _agent(env)
    agent.act(_info(0, {"ix_only": _payload([1.0, 1.0, 1.0, 1.0], [0, 1, 2])}))
    payload = _payload([1.0, 1.0, 1.0, 1.0], [0, 1, 2])
    del payload["reward"]
    with pytest.raises(KeyError, match="reward"):
        agent.act(_info(1, {"ix_only": payload}))


def test_a_skipped_decision_step_raises() -> None:
    env = _single_env()
    agent = _agent(env)
    agent.act(_info(0, {"ix_only": _payload([1.0, 1.0, 1.0, 1.0], [0, 1, 2])}))
    with pytest.raises(RuntimeError, match="expected decision"):
        agent.act(_info(3, {"ix_only": _payload([1.0, 1.0, 1.0, 1.0], [0, 1, 2], reward=-1.0)}))


def test_heterogeneous_intersections_are_refused_naming_both_shapes() -> None:
    env = _StubEnv([("ix_zulu", 2), ("ix_alpha", 3)])
    with pytest.raises(ValueError, match="n_actions"):
        _agent(env)


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------


def _action_sequence(agent: DTAgent) -> list[int]:
    out: list[int] = []
    for step in range(5):
        info = _info(
            step,
            {"ix_only": _payload([float(step), 1.0, 2.0, 3.0], [0, 1, 2], reward=-float(step))},
        )
        out.append(int(agent.act(info, explore=False)[0]))
    return out


def test_saving_before_the_model_exists_raises_instead_of_writing_an_empty_checkpoint() -> None:
    """The state width is only knowable from an ``info``, so the model is built lazily.

    Added when the three checkpoint tests below failed: their setup built an agent that had
    never seen an ``info``, so there was no model to save.  The precondition is pinned here
    rather than left as an incidental raise -- see the Return Packet's disclosure.
    """
    with pytest.raises(ValueError, match="has not been built yet"):
        _agent(_single_env()).save("unreachable.pt")


def test_save_load_round_trip_reproduces_the_action_sequence(tmp_path: Path) -> None:
    env = _single_env()
    agent = _agent(env, state_dim=FIXTURE_STATE_DIM)
    path = tmp_path / "dt.pt"
    agent.save(str(path))

    restored = DTAgent.from_checkpoint(env, str(path), device="cpu")
    assert _action_sequence(agent) == _action_sequence(restored)


def test_checkpoint_records_its_format_version(tmp_path: Path) -> None:
    agent = _agent(_single_env(), state_dim=FIXTURE_STATE_DIM)
    path = tmp_path / "dt.pt"
    agent.save(str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION


def test_a_checkpoint_for_another_shape_is_refused_and_changes_nothing(tmp_path: Path) -> None:
    """The in-memory form of the mutation barrier: check first, adopt second."""
    path = tmp_path / "dt.pt"
    _agent(
        _StubEnv([("ix_only", FIXTURE_N_ACTIONS)]), state_dim=FIXTURE_STATE_DIM
    ).save(str(path))

    victim = _agent(_StubEnv([("ix_only", 5)]), state_dim=FIXTURE_STATE_DIM)
    before = {
        name: parameter.detach().clone()
        for name, parameter in victim.model.named_parameters()
    }
    with pytest.raises(ValueError, match="n_actions"):
        victim.load(str(path))
    after = {
        name: parameter.detach().clone()
        for name, parameter in victim.model.named_parameters()
    }
    assert set(before) == set(after)
    for name in before:
        assert torch.equal(before[name], after[name]), f"{name} changed during a refused load"
