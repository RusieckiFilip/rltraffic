"""Tests for ``agent.OfflineBaselines`` -- the BC/%BC and IQL policies and their checkpoints.

No corpus and no simulator: the loader-backed tests write their own format-v1.0 episode files
through the fixture builder in ``tests/test_offline_dataset.py``, so a *real*
``TrajectoryWindowDataset`` reaches the model.  That fixture is reused deliberately rather than
re-implemented -- it is built so wrong implementations are visible (two intersections with
different ``(state_dim, n_actions)``, env order not equal to sorted order, an ``avail_mask``
that genuinely binds, and a constant feature whose fitted std is exactly 0).

The load-bearing test in this file is
``test_the_bc_inference_state_equals_the_training_window_row_for_the_same_step``.  P4's
independent review found the DT's entire online path unprotected because no test ever built the
agent with ``stats=``; three mutations survived all 58 tests and one cost **+3.8 ATT**, most of
P4's margin.  BC and IQL get that guard from the start, in both agents.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import agent.OfflineBaselines as baselines_module
from agent.DTAgent import action_loss
from agent.OfflineBaselines import (
    BC_CHECKPOINT_FORMAT_VERSION,
    IQL_CHECKPOINT_FORMAT_VERSION,
    BCAgent,
    IQLAgent,
    MLPTrunk,
    TrunkConfig,
    canonical_state_dict_digest,
)
from offline.dataset import PAD_ACTION, TrajectoryWindowDataset, collate_windows

from tests.test_dt_agent import _StubEnv, _info, _payload
from tests.test_offline_dataset import index_of, write_dataset_dir

# The fixture's second intersection: state_dim 4, n_actions 3, T = 8 decision rows.
FIXTURE_STATE_DIM = 4
FIXTURE_N_ACTIONS = 3
FIXTURE_GROUP = (FIXTURE_STATE_DIM, FIXTURE_N_ACTIONS)
FIXTURE_SCENARIO = "fixture_2ix"
CONTEXT = 4


def _trunk_config(**overrides: Any) -> TrunkConfig:
    """A deliberately tiny architecture: these tests check wiring, not capacity."""
    params: dict[str, Any] = {
        "state_dim": FIXTURE_STATE_DIM,
        "n_actions": FIXTURE_N_ACTIONS,
        "d_model": 16,
        "n_layer": 2,
        "dropout": 0.0,
    }
    params.update(overrides)
    return TrunkConfig(**params)


def _bc(env: _StubEnv, **overrides: Any) -> BCAgent:
    params: dict[str, Any] = {
        "d_model": 16,
        "n_layer": 2,
        "dropout": 0.0,
        "device": "cpu",
        "seed": 5,
    }
    params.update(overrides)
    return BCAgent(env, **params)


def _iql(env: _StubEnv, **overrides: Any) -> IQLAgent:
    params: dict[str, Any] = {
        "d_model": 16,
        "n_layer": 2,
        "dropout": 0.0,
        "device": "cpu",
        "seed": 5,
    }
    params.update(overrides)
    return IQLAgent(env, **params)


def _single_env() -> _StubEnv:
    return _StubEnv([("ix_only", FIXTURE_N_ACTIONS)])


@pytest.fixture()
def loader_batch(tmp_path: Path) -> dict[str, torch.Tensor]:
    """A collated batch of REAL loader items, all from the ``(4, 3)`` group."""
    dataset_dir = write_dataset_dir(tmp_path, "fixture__policy")
    dataset = TrajectoryWindowDataset([dataset_dir], context_length=CONTEXT, split="train")
    indices = dataset.groups[FIXTURE_GROUP]
    assert len(indices) >= 8, f"fixture must supply a real batch, got {len(indices)} items"
    return collate_windows([dataset[i] for i in indices[:8]])


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------
# The canonical checkpoint digest (DEFERRED 29, brief section 3.3)
# ----------------------------------------------------------------------


def test_two_differently_named_saves_of_identical_weights_share_a_canonical_digest(
    tmp_path: Path,
) -> None:
    """The whole point of ``DEFERRED`` 29, in one assertion pair.

    ``torch.save`` names the zip root after the output file, so two checkpoints written under
    different names can never be byte-identical however deterministic the training was.  The
    canonical digest is taken over the ``state_dict`` tensor bytes in sorted key order, so it
    is blind to both the filename and the provenance block -- and the file hashes, which are
    kept for transport integrity, must differ.
    """
    env = _single_env()
    agent = _bc(env, state_dim=FIXTURE_STATE_DIM)
    first = tmp_path / "a_name.pt"
    second = tmp_path / "b_name.pt"
    agent.save(str(first), provenance={"note": "first"})
    agent.save(str(second), provenance={"note": "second", "extra": 17})

    payload_a = torch.load(first, map_location="cpu", weights_only=False)
    payload_b = torch.load(second, map_location="cpu", weights_only=False)
    assert canonical_state_dict_digest(payload_a["model"]) == canonical_state_dict_digest(
        payload_b["model"]
    )
    assert _file_sha256(first) != _file_sha256(second), (
        "the file hashes must differ, or this test is not exercising the distinction it exists "
        "for"
    )


def test_the_canonical_digest_ignores_key_insertion_order() -> None:
    """Sorted key order is the specification, not an accident of dict iteration."""
    first = {
        "b.weight": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "a.weight": torch.tensor([3.0], dtype=torch.float32),
    }
    second = {
        "a.weight": torch.tensor([3.0], dtype=torch.float32),
        "b.weight": torch.tensor([1.0, 2.0], dtype=torch.float32),
    }
    assert list(first) != list(second), "the two mappings must differ in insertion order"
    assert canonical_state_dict_digest(first) == canonical_state_dict_digest(second)


def test_the_canonical_digest_changes_when_one_weight_changes() -> None:
    """The control: a digest that never changes would pass the two tests above."""
    base = {"a.weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    moved = {"a.weight": torch.tensor([1.0, 2.5], dtype=torch.float32)}
    assert canonical_state_dict_digest(base) != canonical_state_dict_digest(moved)


def test_the_canonical_digest_separates_two_tensors_sharing_a_byte_pattern() -> None:
    """Key names and shapes enter the digest, so equal bytes under different keys differ.

    Without the key and the shape in the hash, ``{"a": [1, 2]}`` and ``{"b": [1, 2]}`` -- and
    ``(2, 1)`` against ``(1, 2)`` -- would collide, and a reshaped or renamed parameter would
    be reported as reproducing.
    """
    same_bytes_other_key = {"b.weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    base = {"a.weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    reshaped = {"a.weight": torch.tensor([[1.0], [2.0]], dtype=torch.float32)}
    assert canonical_state_dict_digest(base) != canonical_state_dict_digest(
        same_bytes_other_key
    )
    assert canonical_state_dict_digest(base) != canonical_state_dict_digest(reshaped)


# ----------------------------------------------------------------------
# The loss: BC uses the DT's, so padded positions cannot contribute
# ----------------------------------------------------------------------


def test_a_real_loader_batch_reaches_the_bc_loss_without_a_shape_error(
    loader_batch: dict[str, torch.Tensor],
) -> None:
    model = MLPTrunk(_trunk_config(), FIXTURE_N_ACTIONS).eval()
    logits = model(loader_batch["state"])
    assert tuple(logits.shape) == (8, CONTEXT, FIXTURE_N_ACTIONS)
    loss = action_loss(logits, loader_batch["action"])
    assert loss.ndim == 0
    assert bool(torch.isfinite(loss))


def test_padded_positions_cannot_move_the_bc_loss(
    loader_batch: dict[str, torch.Tensor],
) -> None:
    """Randomising the logits AT padded positions leaves the loss bit-identical.

    Exact ``==`` rather than a tolerance: the padded positions either enter the reduction or
    they do not, and a float64 re-derivation would only blur that.
    """
    model = MLPTrunk(_trunk_config(), FIXTURE_N_ACTIONS).eval()
    with torch.no_grad():
        logits = model(loader_batch["state"])
    baseline = action_loss(logits, loader_batch["action"])

    padded = (loader_batch["action"] == PAD_ACTION).unsqueeze(-1)
    assert int(padded.sum()) > 0, "the batch must contain padding, or this asserts nothing"
    generator = torch.Generator().manual_seed(4242)
    noise = torch.randn(logits.shape, generator=generator) * 100.0
    perturbed = torch.where(padded, noise, logits)
    assert not torch.equal(perturbed, logits), "the perturbation must change something"

    assert action_loss(perturbed, loader_batch["action"]) == baseline


def test_the_pad_target_is_a_tripwire_without_ignore_index(
    loader_batch: dict[str, torch.Tensor],
) -> None:
    """Dropping ``ignore_index`` does not silently train on action-0 targets -- it raises."""
    model = MLPTrunk(_trunk_config(), FIXTURE_N_ACTIONS).eval()
    with torch.no_grad():
        logits = model(loader_batch["state"])
    with pytest.raises(IndexError, match="Target -1 is out of bounds"):
        F.cross_entropy(
            logits.reshape(-1, FIXTURE_N_ACTIONS), loader_batch["action"].reshape(-1)
        )


# ----------------------------------------------------------------------
# THE inference-path guard (brief section 4, item 1)
# ----------------------------------------------------------------------


def _episode_streams(dataset_dir: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """``ix_alpha``'s raw arrays for the first episode, read straight from the ``.npz``."""
    episode_file = sorted(p.name for p in dataset_dir.glob("*.npz"))[0]
    with np.load(dataset_dir / episode_file) as raw:
        return (
            episode_file,
            np.asarray(raw["ix1_state"], dtype=np.float32),
            np.asarray(raw["ix1_avail_mask"], dtype=np.bool_),
            np.asarray(raw["ix1_local_reward"], dtype=np.float32),
        )


def _replay_episode_through(
    agent: Any, network: torch.nn.Module, dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[torch.Tensor], list[torch.Tensor], str, int]:
    """Drive *agent* through one fixture episode, capturing what reaches the model.

    Two things are captured, because both are inputs to the decision: the state tensor handed
    to the policy network, and the availability mask handed to the masking helper.
    """
    episode_file, states, avail, rewards = _episode_streams(dataset_dir)
    seen_state: list[torch.Tensor] = []
    seen_mask: list[torch.Tensor] = []

    original_forward = network.forward

    def spy_forward(state: torch.Tensor) -> torch.Tensor:  # type: ignore[no-untyped-def]
        seen_state.append(state.detach().clone())
        return original_forward(state)

    original_mask = baselines_module.masked_action_logits

    def spy_mask(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        seen_mask.append(None if mask is None else mask.detach().clone())
        return original_mask(logits, mask)

    monkeypatch.setattr(network, "forward", spy_forward)
    monkeypatch.setattr(baselines_module, "masked_action_logits", spy_mask)

    decisions = int(rewards.shape[0])
    for t in range(decisions):
        payload = _payload(
            states[t].tolist(),
            [a for a in range(FIXTURE_N_ACTIONS) if bool(avail[t, a])],
            reward=0.0 if t == 0 else float(rewards[t - 1]),
        )
        agent.act(_info(t, {"ix_alpha": payload}), explore=False)
    return seen_state, seen_mask, episode_file, decisions


def test_the_bc_inference_state_equals_the_training_window_row_for_the_same_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What ``act()`` feeds the model must equal what training fed it at the same step.

    THE MOST LOAD-BEARING TEST IN THIS FILE.  P4's review found that dropping state
    normalisation on the DT's online path survived all 58 tests and moved the reported number
    by **+32.5 ATT**, because no test ever constructed the agent with ``stats=``.  BC's
    training input at window position ``-1`` is exactly the state it must see at decision
    ``t``, so the two are compared directly and exactly -- the fixture's values are integral
    and its statistics are exact in float32, so this is ``==`` and not a tolerance.
    """
    dataset_dir = write_dataset_dir(tmp_path, "fixture__policy")
    dataset = TrajectoryWindowDataset([dataset_dir], context_length=CONTEXT, split="train")

    env = _StubEnv([("ix_alpha", FIXTURE_N_ACTIONS)])
    agent = _bc(
        env,
        stats=dataset.stats,
        scenario_id=FIXTURE_SCENARIO,
        state_dim=FIXTURE_STATE_DIM,
    )
    seen_state, seen_mask, episode_file, decisions = _replay_episode_through(
        agent, agent.model, dataset_dir, monkeypatch
    )

    assert len(seen_state) == decisions
    for t in range(decisions):
        expected = dataset[index_of(dataset, "ix_alpha", t, episode_file)]
        assert torch.equal(
            seen_state[t].reshape(-1), expected["state"][-1]
        ), f"state at t={t}: {seen_state[t].reshape(-1)} vs {expected['state'][-1]}"
        assert torch.equal(
            seen_mask[t].reshape(-1), expected["avail_mask"][-1]
        ), f"avail_mask at t={t}"

    # The fixture must actually exercise normalisation and a binding mask, or the comparison
    # above is satisfied by two identity transforms agreeing.
    mean = dataset.stats.state_mean[FIXTURE_SCENARIO]["ix_alpha"]
    assert float(np.abs(mean).max()) > 0.0
    assert any(not bool(mask.all()) for mask in seen_mask)


def test_the_iql_inference_state_equals_the_training_window_row_for_the_same_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same guard on IQL's online path, which has the same failure mode."""
    dataset_dir = write_dataset_dir(tmp_path, "fixture__policy")
    dataset = TrajectoryWindowDataset([dataset_dir], context_length=CONTEXT, split="train")

    env = _StubEnv([("ix_alpha", FIXTURE_N_ACTIONS)])
    agent = _iql(
        env,
        stats=dataset.stats,
        scenario_id=FIXTURE_SCENARIO,
        state_dim=FIXTURE_STATE_DIM,
    )
    seen_state, seen_mask, episode_file, decisions = _replay_episode_through(
        agent, agent.policy, dataset_dir, monkeypatch
    )

    assert len(seen_state) == decisions
    for t in range(decisions):
        expected = dataset[index_of(dataset, "ix_alpha", t, episode_file)]
        assert torch.equal(seen_state[t].reshape(-1), expected["state"][-1]), f"state t={t}"
        assert torch.equal(seen_mask[t].reshape(-1), expected["avail_mask"][-1]), f"mask t={t}"


def test_iql_acts_from_the_policy_network_and_not_from_q(tmp_path: Path) -> None:
    """Perturbing Q must not change a decision; perturbing the policy must.

    IQL extracts its policy by advantage-weighted regression, so ``act()`` reads ``pi``.  An
    implementation that took ``argmax Q`` instead would still produce plausible actions and a
    plausible number -- the failure mode this project keeps finding.
    """
    dataset_dir = write_dataset_dir(tmp_path, "fixture__policy")
    dataset = TrajectoryWindowDataset([dataset_dir], context_length=CONTEXT, split="train")
    env = _StubEnv([("ix_alpha", FIXTURE_N_ACTIONS)])
    agent = _iql(
        env,
        stats=dataset.stats,
        scenario_id=FIXTURE_SCENARIO,
        state_dim=FIXTURE_STATE_DIM,
    )
    info = _info(0, {"ix_alpha": _payload([1.0, 2.0, 3.0, 4.0], [0, 1, 2])})
    before = int(agent.act(info, explore=False)[0])

    with torch.no_grad():
        for parameter in agent.q.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=torch.Generator().manual_seed(1)) * 10.0)
    assert int(agent.act(info, explore=False)[0]) == before, "Q must not drive the decision"

    with torch.no_grad():
        for parameter in agent.policy.parameters():
            parameter.mul_(0.0)
        # A deterministic tie-break would hide the change, so bias one action decisively.
        agent.policy.head.bias[2] = 50.0
    assert int(agent.act(info, explore=False)[0]) == 2


# ----------------------------------------------------------------------
# The BaseAgent face
# ----------------------------------------------------------------------


def test_act_returns_one_int64_action_per_intersection_in_env_order() -> None:
    env = _StubEnv([("ix_zulu", 3), ("ix_alpha", 3)])
    agent = _bc(env)
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


def test_act_never_selects_an_illegal_action_over_many_steps() -> None:
    for agent in (_bc(_single_env()), _iql(_single_env())):
        for step in range(6):
            info = _info(step, {"ix_only": _payload([1.0, 2.0, 3.0, 4.0], [2], reward=-1.0)})
            assert int(agent.act(info, explore=False)[0]) == 2


def test_a_heterogeneous_action_space_is_refused_naming_both_shapes() -> None:
    env = _StubEnv([("ix_zulu", 2), ("ix_alpha", 3)])
    with pytest.raises(ValueError, match="n_actions differs across them"):
        _bc(env)
    with pytest.raises(ValueError, match="n_actions differs across them"):
        _iql(env)


def test_statistics_without_a_scenario_id_are_refused(tmp_path: Path) -> None:
    dataset_dir = write_dataset_dir(tmp_path, "fixture__policy")
    dataset = TrajectoryWindowDataset([dataset_dir], context_length=CONTEXT, split="train")
    with pytest.raises(ValueError, match="scenario_id"):
        _bc(_single_env(), stats=dataset.stats)


# ----------------------------------------------------------------------
# Determinism, by canonical digest and not by file hash
# ----------------------------------------------------------------------


def _train_a_few_steps(seed: int, batch: dict[str, torch.Tensor]) -> str:
    torch.manual_seed(0)                       # the global RNG must not be what decides
    model = MLPTrunk(_trunk_config(), FIXTURE_N_ACTIONS)
    torch.manual_seed(seed)
    for parameter in model.parameters():
        with torch.no_grad():
            parameter.copy_(torch.randn(parameter.shape))
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(5):
        loss = action_loss(model(batch["state"]), batch["action"])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    return canonical_state_dict_digest(model.state_dict())


def test_the_same_seed_reproduces_the_canonical_digest_and_another_seed_does_not(
    loader_batch: dict[str, torch.Tensor],
) -> None:
    assert _train_a_few_steps(1234, loader_batch) == _train_a_few_steps(1234, loader_batch)
    assert _train_a_few_steps(1234, loader_batch) != _train_a_few_steps(4321, loader_batch)


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------


def _action_sequence(agent: Any) -> list[int]:
    out: list[int] = []
    for step in range(5):
        info = _info(
            step,
            {"ix_only": _payload([float(step), 1.0, 2.0, 3.0], [0, 1, 2], reward=-float(step))},
        )
        out.append(int(agent.act(info, explore=False)[0]))
    return out


def test_bc_save_load_round_trip_reproduces_the_action_sequence(tmp_path: Path) -> None:
    env = _single_env()
    agent = _bc(env, state_dim=FIXTURE_STATE_DIM)
    path = tmp_path / "bc.pt"
    agent.save(str(path))

    restored = BCAgent.from_checkpoint(env, str(path), device="cpu")
    assert _action_sequence(agent) == _action_sequence(restored)
    assert restored.canonical_digest() == agent.canonical_digest()


def test_iql_save_load_round_trip_reproduces_the_action_sequence(tmp_path: Path) -> None:
    env = _single_env()
    agent = _iql(env, state_dim=FIXTURE_STATE_DIM)
    path = tmp_path / "iql.pt"
    agent.save(str(path))

    restored = IQLAgent.from_checkpoint(env, str(path), device="cpu")
    assert _action_sequence(agent) == _action_sequence(restored)
    assert restored.canonical_digest() == agent.canonical_digest()


def test_checkpoints_record_their_format_version_and_canonical_digest(tmp_path: Path) -> None:
    env = _single_env()
    for agent, expected in (
        (_bc(env, state_dim=FIXTURE_STATE_DIM), BC_CHECKPOINT_FORMAT_VERSION),
        (_iql(env, state_dim=FIXTURE_STATE_DIM), IQL_CHECKPOINT_FORMAT_VERSION),
    ):
        path = tmp_path / f"{expected.split('-')[0]}.pt"
        agent.save(str(path))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["format_version"] == expected
        assert payload["canonical_digest"] == agent.canonical_digest()


def test_saving_before_the_model_exists_raises_instead_of_writing_an_empty_checkpoint() -> None:
    """The state width is only knowable from an ``info``, so the networks are built lazily."""
    with pytest.raises(ValueError, match="nothing to save: the model has not been built yet"):
        _bc(_single_env()).save("unreachable.pt")


def test_a_checkpoint_from_another_action_space_is_refused_and_leaves_the_agent_unchanged(
    tmp_path: Path,
) -> None:
    """Check before adopt: a refused checkpoint must leave the agent exactly as it was."""
    wide = _bc(_StubEnv([("ix_only", 5)]), state_dim=FIXTURE_STATE_DIM)
    path = tmp_path / "wide.pt"
    wide.save(str(path))

    agent = _bc(_single_env(), state_dim=FIXTURE_STATE_DIM)
    before = agent.canonical_digest()
    with pytest.raises(ValueError, match="checkpoint n_actions"):
        agent.load(str(path))
    assert agent.canonical_digest() == before


def test_a_checkpoint_claiming_normalisation_without_statistics_is_refused(
    tmp_path: Path,
) -> None:
    """Evaluating it on raw states would feed the model a different input distribution."""
    env = _single_env()
    agent = _bc(env, state_dim=FIXTURE_STATE_DIM)
    path = tmp_path / "bc.pt"
    agent.save(str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["normalise"] = True
    payload["stats"] = None
    torch.save(payload, path)

    with pytest.raises(ValueError, match="carries no statistics"):
        _bc(env, state_dim=FIXTURE_STATE_DIM).load(str(path))


def test_observe_records_the_reward_without_touching_the_policy(tmp_path: Path) -> None:
    """``observe`` satisfies the BaseAgent contract and must not move any weight."""
    agent = _bc(_single_env(), state_dim=FIXTURE_STATE_DIM)
    before = agent.canonical_digest()
    agent.observe({"step": 1}, -3.0, False, True)
    assert agent.get_reward() == -3.0
    assert agent.canonical_digest() == before
