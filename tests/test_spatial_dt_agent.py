"""The spatial mixing layer: what it may see, what it may not, and what the control removes.

The single most load-bearing test here is
``test_no_other_nodes_action_at_step_t_can_change_the_prediction_at_step_t``.  All N intersections
act simultaneously, so node ``i`` seeing node ``j``'s **state** at step ``t`` is the point of the
model and node ``i`` seeing node ``j``'s **action** at step ``t`` is leakage -- it is the very
quantity ``i`` is being asked to produce, from a policy that in deployment has not chosen it yet.
A model with that leak trains beautifully and reports a number that cannot be reproduced online.

The mask tests come in matched pairs: a non-neighbour must NOT be able to change my prediction, and
a neighbour MUST be able to.  Without the second half the first is satisfied by a layer that does
nothing at all, which is ``PROJECT_PLAN`` section 7's tautological-fixture rule applied to an
ablation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pytest
import torch

from agent.SpatialDTAgent import (
    SPATIAL_CHECKPOINT_FORMAT_VERSION,
    SpatialDecisionTransformer,
    SpatialDTAgent,
    SpatialDTConfig,
    spatial_attention_bias,
)
from offline.dataset import PAD_ACTION

from tests.test_dt_agent import _StubEnv, _info, _payload

STATE_DIM = 4
N_ACTIONS = 3
N_NODES = 3
CONTEXT = 4
MAX_EP_LEN = 16

# Node 0 <-> node 1 are neighbours; node 2 is isolated.  Self-loops are open on every node, as
# AdjacencySpec.attention_mask produces them.
NEIGHBOUR_MASK = np.array(
    [[True, True, False], [True, True, False], [False, False, True]], dtype=np.bool_
)
IDENTITY_MASK = np.eye(N_NODES, dtype=np.bool_)


def _config(**overrides: Any) -> SpatialDTConfig:
    """A deliberately tiny architecture: these tests check wiring, not capacity."""
    params: dict[str, Any] = {
        "state_dim": STATE_DIM,
        "n_actions": N_ACTIONS,
        "n_nodes": N_NODES,
        "context_length": CONTEXT,
        "n_layer": 2,
        "n_head": 1,
        "d_model": 16,
        "dropout": 0.0,
        "max_ep_len": MAX_EP_LEN,
    }
    params.update(overrides)
    return SpatialDTConfig(**params)


def _batch(batch: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "rtg": torch.randn(batch, N_NODES, CONTEXT, 1, generator=generator),
        "state": torch.randn(batch, N_NODES, CONTEXT, STATE_DIM, generator=generator),
        "action": torch.randint(
            0, N_ACTIONS, (batch, N_NODES, CONTEXT), generator=generator
        ),
        "timestep": torch.arange(CONTEXT).view(1, 1, CONTEXT).expand(batch, N_NODES, CONTEXT).contiguous(),
        "attention_mask": torch.ones(batch, N_NODES, CONTEXT, dtype=torch.bool),
    }


def _model(mask: np.ndarray, seed: int = 7, **overrides: Any) -> SpatialDecisionTransformer:
    torch.manual_seed(seed)
    model = SpatialDecisionTransformer(_config(**overrides))
    model.eval()
    return model


def _forward(model: SpatialDecisionTransformer, batch: dict[str, torch.Tensor],
             mask: np.ndarray, **kwargs: Any) -> torch.Tensor:
    with torch.no_grad():
        return model(
            batch["rtg"], batch["state"], batch["action"], batch["timestep"],
            torch.from_numpy(mask), batch["attention_mask"], kwargs.get("avail_mask"),
        )


# ----------------------------------------------------------------------
# Shape and basic wiring
# ----------------------------------------------------------------------


def test_the_model_returns_one_prediction_per_node_per_step():
    model = _model(NEIGHBOUR_MASK)
    logits = _forward(model, _batch(), NEIGHBOUR_MASK)
    assert logits.shape == (2, N_NODES, CONTEXT, N_ACTIONS)
    assert torch.isfinite(logits).all()


def test_the_two_arms_have_identical_parameter_counts():
    """The control removes information, never capacity -- decision D4 in docs/plans/p5.1.md."""
    torch.manual_seed(0)
    mixing = SpatialDecisionTransformer(_config(spatial_mixing=True))
    torch.manual_seed(0)
    control = SpatialDecisionTransformer(_config(spatial_mixing=False))

    assert sum(p.numel() for p in mixing.parameters()) == sum(
        p.numel() for p in control.parameters()
    )
    assert [tuple(p.shape) for p in mixing.parameters()] == [
        tuple(p.shape) for p in control.parameters()
    ]
    # Same seed, same initial weights: the arms are weight-compatible by construction.
    for left, right in zip(mixing.state_dict().values(), control.state_dict().values()):
        assert torch.equal(left, right)


def test_a_wrong_spatial_mask_shape_is_refused():
    model = _model(NEIGHBOUR_MASK)
    batch = _batch()
    with pytest.raises(ValueError, match="spatial mask must be"):
        model(
            batch["rtg"], batch["state"], batch["action"], batch["timestep"],
            torch.ones(N_NODES + 1, N_NODES + 1, dtype=torch.bool), batch["attention_mask"], None,
        )


def test_a_node_count_that_disagrees_with_the_batch_is_refused():
    model = _model(NEIGHBOUR_MASK)
    batch = _batch()
    trimmed = {key: value[:, :2] for key, value in batch.items()}
    with pytest.raises(ValueError, match="node count"):
        model(
            trimmed["rtg"], trimmed["state"], trimmed["action"], trimmed["timestep"],
            torch.from_numpy(NEIGHBOUR_MASK), trimmed["attention_mask"], None,
        )


def test_a_fully_masked_spatial_row_is_refused_rather_than_producing_nan():
    """Our softmax, not SDPA's: an all-``-inf`` row would be NaN and would reach the backward pass."""
    broken = NEIGHBOUR_MASK.copy()
    broken[2, 2] = False
    with pytest.raises(ValueError, match="no attendable node"):
        spatial_attention_bias(broken, device=torch.device("cpu"), dtype=torch.float32)


# ----------------------------------------------------------------------
# 🚨 THE LEAKAGE TEST -- the concurrent action must be invisible
# ----------------------------------------------------------------------


def test_no_other_nodes_action_at_step_t_can_change_the_prediction_at_step_t():
    """Every node acts simultaneously, so a peer's action at ``t`` may not reach a prediction at ``t``.

    Structural argument in the module docstring: temporal runs before spatial in every block, so a
    peer's token at ``3t+1`` has not yet absorbed its own ``a_t``; spatial mixes position-wise; and
    causal temporal attention cannot carry ``3t+2`` back to ``3t+1``.  This is that argument made
    mechanical.  **Swapping the two sublayers breaks it and the model still trains.**
    """
    model = _model(NEIGHBOUR_MASK)
    batch = _batch()
    baseline = _forward(model, batch, NEIGHBOUR_MASK)

    for node in range(N_NODES):
        altered = {key: value.clone() for key, value in batch.items()}
        current = altered["action"][:, node, -1]
        altered["action"][:, node, -1] = (current + 1) % N_ACTIONS
        assert not torch.equal(altered["action"], batch["action"]), "the probe changed nothing"

        changed = _forward(model, altered, NEIGHBOUR_MASK)
        assert torch.equal(changed[:, :, -1], baseline[:, :, -1]), (
            f"node {node}'s action at the last step changed some node's prediction at that step"
        )


def test_a_peers_action_at_an_earlier_step_does_reach_the_prediction():
    """Discriminating power for the test above: history is supposed to matter."""
    model = _model(NEIGHBOUR_MASK)
    batch = _batch()
    baseline = _forward(model, batch, NEIGHBOUR_MASK)

    altered = {key: value.clone() for key, value in batch.items()}
    altered["action"][:, 1, 0] = (altered["action"][:, 1, 0] + 1) % N_ACTIONS
    changed = _forward(model, altered, NEIGHBOUR_MASK)

    assert not torch.equal(changed[:, 0, -1], baseline[:, 0, -1]), (
        "node 1's EARLIER action never reached node 0, so the leakage test above is vacuous"
    )


# ----------------------------------------------------------------------
# The mask: matched pairs, so neither half can be satisfied by a dead layer
# ----------------------------------------------------------------------


def test_a_non_neighbours_state_cannot_change_my_prediction():
    model = _model(NEIGHBOUR_MASK)
    batch = _batch()
    baseline = _forward(model, batch, NEIGHBOUR_MASK)

    altered = {key: value.clone() for key, value in batch.items()}
    altered["state"][:, 2] = altered["state"][:, 2] + 5.0  # node 2 is isolated
    changed = _forward(model, altered, NEIGHBOUR_MASK)

    assert torch.equal(changed[:, 0], baseline[:, 0])
    assert torch.equal(changed[:, 1], baseline[:, 1])


def test_a_neighbours_state_must_change_my_prediction():
    """The other half of the pair: without it, a layer that does nothing passes the test above."""
    model = _model(NEIGHBOUR_MASK)
    batch = _batch()
    baseline = _forward(model, batch, NEIGHBOUR_MASK)

    altered = {key: value.clone() for key, value in batch.items()}
    altered["state"][:, 1] = altered["state"][:, 1] + 5.0  # node 1 neighbours node 0
    changed = _forward(model, altered, NEIGHBOUR_MASK)

    assert not torch.equal(changed[:, 0], baseline[:, 0])


def test_with_mixing_disabled_no_other_nodes_state_can_change_my_prediction():
    """The registered no-mixing control, checked as a property rather than as a flag."""
    model = _model(IDENTITY_MASK, spatial_mixing=False)
    batch = _batch()
    baseline = _forward(model, batch, IDENTITY_MASK)

    for other in (1, 2):
        altered = {key: value.clone() for key, value in batch.items()}
        altered["state"][:, other] = altered["state"][:, other] + 5.0
        changed = _forward(model, altered, IDENTITY_MASK)
        assert torch.equal(changed[:, 0], baseline[:, 0]), f"node {other} reached node 0"


def test_the_control_and_the_treatment_disagree_on_the_same_weights():
    """Same parameters, same batch, different masks -- so the arms are genuinely different models."""
    torch.manual_seed(3)
    model = SpatialDecisionTransformer(_config())
    model.eval()
    batch = _batch()

    mixed = _forward(model, batch, NEIGHBOUR_MASK)
    isolated = _forward(model, batch, IDENTITY_MASK)
    assert not torch.equal(mixed, isolated)


# ----------------------------------------------------------------------
# Masking, padding and the availability guard, inherited from DTAgent
# ----------------------------------------------------------------------


def test_illegal_actions_are_masked_out_of_the_returned_logits():
    model = _model(NEIGHBOUR_MASK)
    batch = _batch()
    avail = torch.ones(2, N_NODES, CONTEXT, N_ACTIONS, dtype=torch.bool)
    avail[:, :, :, 1] = False
    logits = _forward(model, batch, NEIGHBOUR_MASK, avail_mask=avail)

    assert torch.isinf(logits[:, :, :, 1]).all() and (logits[:, :, :, 1] < 0).all()
    assert torch.isfinite(logits[:, :, :, 0]).all()


def test_a_padded_window_produces_no_nan_anywhere():
    model = _model(NEIGHBOUR_MASK)
    batch = _batch()
    batch["attention_mask"][:, :, :2] = False
    batch["action"][:, :, :2] = PAD_ACTION
    logits = _forward(model, batch, NEIGHBOUR_MASK)
    assert not torch.isnan(logits).any()


def test_padding_that_differs_across_nodes_is_refused():
    """All nodes of one instant share a timestep; ragged padding would mix real and padded rows."""
    model = _model(NEIGHBOUR_MASK)
    batch = _batch()
    batch["attention_mask"][:, 0, 0] = False
    with pytest.raises(ValueError, match="padding differs across intersections"):
        model(
            batch["rtg"], batch["state"], batch["action"], batch["timestep"],
            torch.from_numpy(NEIGHBOUR_MASK), batch["attention_mask"], None,
        )


# ----------------------------------------------------------------------
# The agent face
# ----------------------------------------------------------------------


def _three_node_env() -> _StubEnv:
    return _StubEnv([("ix_zulu", N_ACTIONS), ("ix_alpha", N_ACTIONS), ("ix_mike", N_ACTIONS)],
                    max_steps=MAX_EP_LEN)


def _agent(env: _StubEnv, **overrides: Any) -> SpatialDTAgent:
    params: dict[str, Any] = {
        "adjacency": NEIGHBOUR_MASK,
        "context_length": CONTEXT,
        "n_layer": 2,
        "n_head": 1,
        "d_model": 16,
        "dropout": 0.0,
        "max_ep_len": MAX_EP_LEN,
        "state_dim": STATE_DIM,
        "seed": 11,
        "target_rtg": {"ix_zulu": -100.0, "ix_alpha": -200.0, "ix_mike": -300.0},
        "rtg_scale": {"ix_zulu": 100.0, "ix_alpha": 200.0, "ix_mike": 300.0},
    }
    params.update(overrides)
    return SpatialDTAgent(env, **params)


def _three_node_info(step: int, reward: float = 0.0) -> dict[str, Any]:
    return _info(
        step,
        {
            "ix_zulu": _payload([1.0, 2.0, 3.0, 4.0], [0, 1, 2], reward),
            "ix_alpha": _payload([5.0, 6.0, 7.0, 8.0], [0, 1, 2], reward),
            "ix_mike": _payload([9.0, 1.0, 2.0, 3.0], [0, 1, 2], reward),
        },
    )


def test_act_returns_one_int64_action_per_intersection_in_env_order():
    agent = _agent(_three_node_env())
    actions = agent.act(_three_node_info(0), explore=False)
    assert actions.shape == (N_NODES,)
    assert actions.dtype == np.int64
    assert set(actions.tolist()) <= set(range(N_ACTIONS))


def test_each_node_starts_at_its_own_declared_target():
    agent = _agent(_three_node_env())
    agent.act(_three_node_info(0), explore=False)
    assert agent.current_rtg() == {
        "ix_zulu": -100.0, "ix_alpha": -200.0, "ix_mike": -300.0
    }


def test_each_nodes_return_to_go_advances_by_its_own_reward():
    agent = _agent(_three_node_env())
    agent.act(_three_node_info(0), explore=False)
    agent.act(_three_node_info(1, reward=-7.0), explore=False)
    assert agent.current_rtg() == {
        "ix_zulu": -93.0, "ix_alpha": -193.0, "ix_mike": -293.0
    }


def test_an_adjacency_whose_order_disagrees_with_the_env_is_refused():
    from offline.roadnet_graph import AdjacencySpec

    spec = AdjacencySpec(
        node_ids=("ix_alpha", "ix_zulu", "ix_mike"),
        directed=NEIGHBOUR_MASK & ~np.eye(N_NODES, dtype=bool),
        undirected=NEIGHBOUR_MASK & ~np.eye(N_NODES, dtype=bool),
        roadnet_path="/dev/null",
        roadnet_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="adjacency node order"):
        _agent(_three_node_env(), adjacency=spec)


def test_an_adjacency_of_the_wrong_size_is_refused():
    with pytest.raises(ValueError, match="adjacency must be"):
        _agent(_three_node_env(), adjacency=np.eye(2, dtype=np.bool_))


def test_a_missing_per_node_target_is_refused():
    with pytest.raises(ValueError, match="target_rtg is missing"):
        _agent(_three_node_env(), target_rtg={"ix_zulu": -1.0})


def test_a_zero_scale_is_refused_because_it_divides_the_input():
    with pytest.raises(ValueError, match="rtg_scale must be non-zero"):
        _agent(
            _three_node_env(),
            rtg_scale={"ix_zulu": 0.0, "ix_alpha": 1.0, "ix_mike": 1.0},
        )


def test_a_checkpoint_round_trips_the_graph_and_the_per_node_constants(tmp_path):
    agent = _agent(_three_node_env())
    agent.act(_three_node_info(0), explore=False)
    path = tmp_path / "spatial.pt"
    agent.save(str(path))

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format_version"] == SPATIAL_CHECKPOINT_FORMAT_VERSION
    assert payload["config"]["spatial_mixing"] is True
    assert payload["intersection_ids"] == ["ix_zulu", "ix_alpha", "ix_mike"]
    assert payload["target_rtg"] == {"ix_zulu": -100.0, "ix_alpha": -200.0, "ix_mike": -300.0}

    restored = SpatialDTAgent.from_checkpoint(_three_node_env(), str(path))
    assert restored.config == agent.config
    assert np.array_equal(restored.spatial_mask, agent.spatial_mask)
    assert restored.current_rtg() == {
        "ix_zulu": -100.0, "ix_alpha": -200.0, "ix_mike": -300.0
    }


def test_a_restored_agent_reproduces_the_saved_agents_greedy_actions(tmp_path):
    agent = _agent(_three_node_env())
    path = tmp_path / "spatial.pt"
    agent.save(str(path))

    restored = SpatialDTAgent.from_checkpoint(_three_node_env(), str(path))
    first = agent.act(_three_node_info(0), explore=False)
    second = restored.act(_three_node_info(0), explore=False)
    assert np.array_equal(first, second)


def test_a_control_checkpoint_cannot_be_loaded_as_a_mixing_agent(tmp_path):
    """The arms are weight-compatible, so only the recorded flag distinguishes them."""
    control = _agent(_three_node_env(), spatial_mixing=False, adjacency=IDENTITY_MASK)
    path = tmp_path / "control.pt"
    control.save(str(path))

    restored = SpatialDTAgent.from_checkpoint(_three_node_env(), str(path))
    assert restored.config.spatial_mixing is False
    assert np.array_equal(restored.spatial_mask, IDENTITY_MASK)


def test_the_agent_uses_the_mask_the_checkpoint_recorded_not_the_one_it_was_handed(tmp_path):
    control = _agent(_three_node_env(), spatial_mixing=False, adjacency=IDENTITY_MASK)
    path = tmp_path / "control.pt"
    control.save(str(path))

    restored = SpatialDTAgent.from_checkpoint(
        _three_node_env(), str(path), adjacency=NEIGHBOUR_MASK
    )
    assert restored.config.spatial_mixing is False
    assert np.array_equal(restored.spatial_mask, IDENTITY_MASK)
