"""P5.3c C1 (``BRIEF_42`` §3 C1(ii), §4): T-k1, T-k2 and T-index -- K = 1 and 2 through the EXISTING loader, trainer and
agent, and the batch-index stream's K-invariance observed THROUGH ``train_dt``.

⚠️ NOT RED FIRST, AND WHY (plan F6, ``BRIEF_42`` Amendment A, Q6).  These tests characterise code that already exists and
that this task does not change -- ``offline/dataset.py``'s window, ``offline/dt_gate.py``'s ``train_dt`` and
``agent/DTAgent.py``'s rollout -- so they pass on their first run.  A test of unchanged code can only show its strength by
MUTATION, and the two §4 names are exactly that, executed and pasted in the packet:

* ``DTAgent._window``'s span at K + 1 (``agent/DTAgent.py:621``) -> the rollout test dies (the model sees K + 1 steps);
* the sampler seeded with K (``offline/dt_gate.py:828``) -> the index-stream test dies.

C2 and C3 add CALLER-level twins driven through ``offline/context_sweep.py``'s own training and evaluation routes, which
ARE red first (``PROJECT_PLAN`` §7: a pin on a function does not pin its caller).

The corpus is ``tests/p5_3c_fixtures.py``'s: ONE intersection, as on ``cityflow1x1``; integral rewards, so every
return-to-go is exact in any summation order; a mask that binds on odd rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from agent.DTAgent import DTAgent
from offline.dataset import PAD_ACTION, TrajectoryWindowDataset
from offline.dt_gate import build_training_dataset, stack_dataset, train_dt
from tests.p5_3c_fixtures import (
    IX_ID,
    N_ACTIONS,
    SCENARIO_ID,
    STATE_DIM,
    T_DECISIONS,
    StubEnv,
    info_at,
    write_single_ix_corpus,
)

#: A power of two, so dividing the return-to-go by it (inference) and multiplying back (this test) is exact.
RTG_SCALE = 64.0


def _raw_episodes(corpus: Path) -> dict[str, dict[str, np.ndarray]]:
    """Every episode's raw arrays, read with ``np.load`` -- NOT through the loader under test."""
    episodes: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(corpus.glob("*.npz")):
        with np.load(path) as raw:
            episodes[path.name] = {
                "state": np.asarray(raw["ix0_state"], dtype=np.float32),
                "action": np.asarray(raw["ix0_action"], dtype=np.int64),
                "avail": np.asarray(raw["ix0_avail_mask"], dtype=np.bool_),
                "reward": np.asarray(raw["ix0_local_reward"], dtype=np.float32),
            }
    return episodes


def _returns_to_go(rewards: np.ndarray) -> np.ndarray:
    """``rtg[t] = sum(r[t:])`` by a reversed ``np.cumsum`` -- a route the loader deliberately does not use."""
    return np.cumsum(rewards.astype(np.float64)[::-1])[::-1].astype(np.float32)


@pytest.mark.parametrize("k", [1, 2])
def test_every_training_window_is_the_k_steps_ending_at_its_t_with_left_padding(tmp_path: Path, k: int) -> None:
    """T-k1 / T-k2, the loader half: every stacked window equals an independent construction from the raw arrays."""
    corpus = write_single_ix_corpus(tmp_path, "fixture__policy", draws=(1, 2, 3))
    dataset = build_training_dataset([corpus], k)
    stacked = stack_dataset(dataset)
    raw = _raw_episodes(corpus)

    assert tuple(stacked["state"].shape) == (3 * T_DECISIONS, k, STATE_DIM)
    for row in range(int(stacked["state"].shape[0])):
        meta = dataset.item_meta(int(stacked["item_index"][row]))
        episode = raw[meta.episode_file]
        t = meta.t
        low = max(0, t - k + 1)
        real = t - low + 1
        pad = k - real

        state = np.zeros((k, STATE_DIM), dtype=np.float32)
        # The loader's frozen statistics normalise the independently SELECTED rows: what is checked here is which
        # rows enter the window and where, not the normalisation formula.
        state[pad:] = dataset.stats.normalize_state(SCENARIO_ID, IX_ID, episode["state"][low : t + 1])
        rtg = np.zeros((k, 1), dtype=np.float32)
        rtg[pad:, 0] = _returns_to_go(episode["reward"])[low : t + 1]
        action = np.full(k, PAD_ACTION, dtype=np.int64)
        action[pad:] = episode["action"][low : t + 1]
        avail = np.zeros((k, N_ACTIONS), dtype=np.bool_)
        avail[pad:] = episode["avail"][low : t + 1]
        timestep = np.zeros(k, dtype=np.int64)
        timestep[pad:] = np.arange(low, t + 1, dtype=np.int64)
        mask = np.zeros(k, dtype=np.bool_)
        mask[pad:] = True

        where = f"{meta.episode_file} t={t}"
        assert torch.equal(stacked["state"][row], torch.from_numpy(state)), where
        assert torch.equal(stacked["rtg"][row], torch.from_numpy(rtg)), where
        assert torch.equal(stacked["action"][row], torch.from_numpy(action)), where
        assert torch.equal(stacked["avail_mask"][row], torch.from_numpy(avail)), where
        assert torch.equal(stacked["timestep"][row], torch.from_numpy(timestep)), where
        assert torch.equal(stacked["attention_mask"][row], torch.from_numpy(mask)), where
        assert int(stacked["attention_mask"][row].sum()) == min(t + 1, k), where


def _train(stacked: dict[str, torch.Tensor], dataset: TrajectoryWindowDataset, k: int, path: Path, target: float) -> Any:
    return train_dt(
        stacked,
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        seed=101,
        declared_gradient_steps=6,
        raise_to=None,
        context_length=k,
        batch_size=8,
        device=torch.device("cpu"),
        checkpoint_path=path,
        stats=dataset.stats,
        scenario_id=SCENARIO_ID,
        target_rtg=target,
        rtg_scale=RTG_SCALE,
        provenance={"tier": "fixture"},
    )


@pytest.mark.parametrize("k", [1, 2])
def test_the_rollout_attends_to_exactly_k_steps_read_from_the_checkpoints_config(tmp_path: Path, k: int) -> None:
    """T-k1 / T-k2, the trainer and agent half.

    ``train_dt(context_length=K)`` writes a checkpoint whose config says K; ``DTAgent.from_checkpoint`` builds the agent
    from that config ALONE (no K is passed here), and at every decision the model is fed exactly K steps -- the
    attention mask marks ``min(t + 1, K)`` of them -- and the window it sees equals the loader's window for the same
    history.  At K = 1 that is ONE step: the current return-to-go, the current state and the absolute timestep, the
    action slot at ``PAD_ACTION`` (plan §5).

    *Mutation:* ``_window``'s span at K + 1 -> the shape assertion dies at the first decision.
    """
    corpus = write_single_ix_corpus(tmp_path, "fixture__policy", draws=(1, 2, 3))
    dataset = build_training_dataset([corpus], k)
    stacked = stack_dataset(dataset)
    first = sorted(p.name for p in corpus.glob("*.npz"))[0]
    episode = _raw_episodes(corpus)[first]
    target = float(episode["reward"].astype(np.float64).sum())
    path = tmp_path / f"k{k}.pt"

    _train(stacked, dataset, k, path, target)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["config"]["context_length"] == k
    assert payload["provenance"]["context_length"] == k

    agent = DTAgent.from_checkpoint(StubEnv(), str(path), device="cpu")
    assert agent.config.context_length == k
    captured: list[dict[str, torch.Tensor]] = []
    original = agent.model.forward

    def spy(rtg, state, action, timestep, attention_mask=None, avail_mask=None):  # type: ignore[no-untyped-def]
        captured.append(
            {
                "rtg": rtg.detach().clone(),
                "state": state.detach().clone(),
                "action": action.detach().clone(),
                "timestep": timestep.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
                "avail_mask": avail_mask.detach().clone(),
            }
        )
        return original(rtg, state, action, timestep, attention_mask, avail_mask)

    agent.model.forward = spy  # type: ignore[method-assign]
    for t in range(T_DECISIONS):
        reward = 0.0 if t == 0 else float(episode["reward"][t - 1])
        agent.act(info_at(t, episode["state"][t], episode["avail"][t], reward), explore=False, update_memory=True)

    assert len(captured) == T_DECISIONS
    index = {
        (dataset.item_meta(i).episode_file, dataset.item_meta(i).t): i for i in range(len(dataset))
    }
    for t, seen in enumerate(captured):
        where = f"decision t={t}"
        assert tuple(seen["state"].shape) == (1, k, STATE_DIM), where
        assert int(seen["attention_mask"][0].sum()) == min(t + 1, k), where
        assert int(seen["timestep"][0, -1]) == t, where
        assert int(seen["action"][0, -1]) == PAD_ACTION, where
        item = dataset[index[(first, t)]]
        assert torch.equal(seen["state"][0], item["state"]), where
        assert torch.equal(seen["timestep"][0], item["timestep"]), where
        assert torch.equal(seen["attention_mask"][0], item["attention_mask"]), where
        assert torch.equal(seen["avail_mask"][0], item["avail_mask"]), where
        assert torch.equal(seen["rtg"][0] * RTG_SCALE, item["rtg"]), where
    # The fixture must bind: a mask that is all-True everywhere would let an identity transform pass the avail check.
    assert not bool(captured[1]["avail_mask"][0, -1].all())


def _recorded_draws(
    monkeypatch: pytest.MonkeyPatch, corpus: Path, tmp_path: Path, k: int, seed: int
) -> dict[str, Any]:
    """Train through ``train_dt`` with ``np.random.default_rng`` wrapped for the call; return what it drew and where."""
    dataset = build_training_dataset([corpus], k)
    stacked = stack_dataset(dataset)
    real = np.random.default_rng
    constructed: list[Any] = []
    drawn: list[np.ndarray] = []

    class Recording:
        def __init__(self, generator_seed: Any) -> None:
            self._generator = real(generator_seed)

        def integers(self, *args: Any, **kwargs: Any) -> Any:
            out = self._generator.integers(*args, **kwargs)
            drawn.append(np.asarray(out).copy())
            return out

    def factory(generator_seed: Any = None) -> Recording:
        constructed.append(generator_seed)
        return Recording(generator_seed)

    with monkeypatch.context() as patch:
        patch.setattr(np.random, "default_rng", factory)
        train_dt(
            stacked,
            state_dim=STATE_DIM,
            n_actions=N_ACTIONS,
            seed=seed,
            declared_gradient_steps=25,
            raise_to=None,
            context_length=k,
            batch_size=64,
            device=torch.device("cpu"),
            checkpoint_path=tmp_path / f"index_k{k}_s{seed}.pt",
            stats=dataset.stats,
            scenario_id=SCENARIO_ID,
            target_rtg=-1.0,
            rtg_scale=RTG_SCALE,
            provenance={"tier": "fixture"},
        )
    rows = np.concatenate(drawn)
    item_index = stacked["item_index"].numpy()
    ids = []
    for row in rows:
        meta = dataset.item_meta(int(item_index[int(row)]))
        ids.append((meta.episode_file, meta.ix_id, meta.t))
    picked = torch.from_numpy(rows.astype(np.int64))
    return {
        "count": int(stacked["state"].shape[0]),
        "item_index": item_index.tolist(),
        "constructed": constructed,
        "calls": len(drawn),
        "rows": rows,
        "ids": ids,
        "last_state": stacked["state"][picked, -1],
        "last_action": stacked["action"][picked, -1],
    }


def test_the_batch_index_stream_under_one_seed_is_identical_at_k_1_2_and_20(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-index: the sampled WINDOW IDS (not the tensors) are identical at K = 1, 2 and 20 under seed 101.

    The loader's index holds one entry per (stream, t) whatever K is (``offline/dataset.py:645-651``), so the stacked row
    count and order are K-invariant, and ``train_dt`` draws its rows from ``np.random.default_rng(seed)``
    (``offline/dt_gate.py:828-835``).  Recorded here by wrapping that generator for the call: 25 steps x 64 rows.
    A different seed, through the SAME recording path, must give a different stream -- the control that lets the
    comparison fail.

    *Mutation:* the sampler seeded with ``seed + context_length`` -> the rows differ across K and this dies.
    """
    corpus = write_single_ix_corpus(tmp_path, "fixture__policy", draws=(1, 2, 3, 4))
    runs = {k: _recorded_draws(monkeypatch, corpus, tmp_path, k, 101) for k in (1, 2, 20)}

    base = runs[1]
    assert base["constructed"] == [101]
    assert base["calls"] == 25 and base["rows"].shape == (25 * 64,)
    for k in (2, 20):
        other = runs[k]
        assert other["constructed"] == [101], k
        assert other["count"] == base["count"] == 4 * T_DECISIONS, k
        assert other["item_index"] == base["item_index"], k
        assert np.array_equal(other["rows"], base["rows"]), k
        assert other["ids"] == base["ids"], k
        # The window's LAST step -- the supervised one -- is the same step at every K.
        assert torch.equal(other["last_state"], base["last_state"]), k
        assert torch.equal(other["last_action"], base["last_action"]), k

    control = _recorded_draws(monkeypatch, corpus, tmp_path, 1, 202)
    assert control["constructed"] == [202]
    assert not np.array_equal(control["rows"], base["rows"])
