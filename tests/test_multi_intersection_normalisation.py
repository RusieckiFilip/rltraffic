"""``DEFERRED`` 37, executed on 16 intersections: every node is normalised with ITS OWN statistics.

WHY THIS FILE EXISTS, IN ITS OWN WORDS
---------------------------------------
``docs/notes/DEFERRED.md`` row 37, carried since P4.4's review as ``MU5``:

    Feeding ``act()`` the wrong intersection's statistics is an EQUIVALENT mutant here and a LIVE
    DEFECT at P5.  Survives 58/58 because every episode in this tier carries ``ix0_*`` only, so
    ``intersection_ids[0] == ix_id`` always.  The moment a multi-intersection corpus exists the
    mutation stops being equivalent and nothing catches it.

The scheduled fix was *"P5.1's brief, explicitly -- not 'when convenient'"*.  This is it.

WHY IT IS LIVE NOW, MEASURED
-----------------------------
On ``cf_grid4x4__mappo1000`` all **120** intersection pairs have different normalisation means
(measured 2026-08-17 over the tier's 200 episodes; ``A0`` against ``C1`` differs by up to
**1.2161**).  So a model that normalised all 16 nodes with intersection 0's statistics would read a
shifted, rescaled input for 15 of them and produce **a plausible, wrong grid** -- no shape error,
no NaN, no crash, and a number that lands in a table.

TWO ROUTES, AND ONLY ONE OF THEM IS LOAD-BEARING
--------------------------------------------------
* **Route A -- the call log.**  ``NormalizationStats.normalize_state`` is wrapped and the sequence
  of ``ix_id`` arguments recorded.  It must equal ``intersection_ids`` in order.  **This is the
  route that kills the mutant**, because the mutation changes precisely which key is asked for.
* **Route B -- the tensor the network consumed.**  Captured from the model's own ``forward``, and
  compared with ``(x - mean_i) / std_i`` computed here from the statistics object, exactly, in
  float32.  Route B alone would *not* kill the mutant -- with the wrong key both the agent and a
  naive recomputation would use intersection 0 -- and that is stated rather than left implicit,
  because a test whose stated purpose it does not serve is worse than no test.

All three agents that normalise per intersection are covered: ``agent/DTAgent.py``,
``agent/OfflineBaselines.py`` and ``agent/SpatialDTAgent.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pytest
import torch

from agent.DTAgent import DTAgent
from agent.OfflineBaselines import BCAgent
from agent.SpatialDTAgent import SpatialDTAgent
from offline.dataset import STATS_VERSION, NormalizationStats, RtgSummary

from tests.test_dt_agent import _StubEnv, _info, _payload

N_NODES = 16
STATE_DIM = 4
N_ACTIONS = 3
SCENARIO = "cityflow_grid4x4"
NODE_IDS = tuple(f"ix{index:02d}" for index in range(N_NODES))


def _stats() -> NormalizationStats:
    """Per-intersection statistics that DIFFER for every node and every feature.

    Built so a wrong key is visible: node ``n``'s mean is offset by ``n`` and its std scaled by
    ``1 + n``, so no two nodes share a normalisation and none is the identity.
    """
    mean = {
        node: np.arange(STATE_DIM, dtype=np.float32) + np.float32(index + 1)
        for index, node in enumerate(NODE_IDS)
    }
    std = {
        node: np.full(STATE_DIM, np.float32(1.0 + index), dtype=np.float32)
        for index, node in enumerate(NODE_IDS)
    }
    summary = RtgSummary(count=1, min=-1.0, max=-1.0, mean=-1.0, std=0.0, quantiles=())
    return NormalizationStats(
        stats_version=STATS_VERSION,
        split="train",
        draw_ids=(1,),
        dataset_dirs=("fixture",),
        state_mean={SCENARIO: mean},
        state_std={SCENARIO: std},
        row_count={SCENARIO: {node: 10 for node in NODE_IDS}},
        rtg={SCENARIO: {node: summary for node in NODE_IDS}},
    )


def test_the_fixture_gives_every_intersection_a_different_normalisation():
    """Guard on the guard: with identical statistics every test below would be a tautology."""
    stats = _stats()
    means = stats.state_mean[SCENARIO]
    stds = stats.state_std[SCENARIO]
    pairs = [
        (a, b) for i, a in enumerate(NODE_IDS) for b in NODE_IDS[i + 1:]
    ]
    assert len(pairs) == 120
    for left, right in pairs:
        assert not np.array_equal(means[left], means[right])
        assert not np.array_equal(stds[left], stds[right])


def _env() -> _StubEnv:
    return _StubEnv([(node, N_ACTIONS) for node in NODE_IDS], max_steps=8)


def _grid_info(step: int = 0) -> dict[str, Any]:
    """One info carrying a DIFFERENT raw state per intersection."""
    return _info(
        step,
        {
            node: _payload(
                [float(index + 1), float(index + 2), float(index + 3), float(index + 4)],
                list(range(N_ACTIONS)),
                0.0,
            )
            for index, node in enumerate(NODE_IDS)
        },
    )


class _CallLog:
    """Route A: records which ``ix_id`` each normalisation was asked for, in order."""

    def __init__(self) -> None:
        self.keys: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = NormalizationStats.normalize_state

        def recording(inner_self, scenario_id, ix_id, rows):  # type: ignore[no-untyped-def]
            self.keys.append(str(ix_id))
            return original(inner_self, scenario_id, ix_id, rows)

        monkeypatch.setattr(NormalizationStats, "normalize_state", recording)


def _capture_dt(agent: DTAgent) -> list[torch.Tensor]:
    """Route B for ``DTAgent``: the ``(N, K, D)`` state tensor the network consumed."""
    seen: list[torch.Tensor] = []
    original = agent.model.forward  # type: ignore[union-attr]

    def spy(rtg, state, action, timestep, attention_mask=None, avail_mask=None):  # type: ignore[no-untyped-def]
        seen.append(state.detach().clone())
        return original(rtg, state, action, timestep, attention_mask, avail_mask)

    agent.model.forward = spy  # type: ignore[union-attr, assignment]
    return seen


def _capture_spatial(agent: SpatialDTAgent) -> list[torch.Tensor]:
    """Route B for ``SpatialDTAgent``: the ``(1, N, K, D)`` state tensor."""
    seen: list[torch.Tensor] = []
    original = agent.model.forward  # type: ignore[union-attr]

    def spy(rtg, state, action, timestep, spatial_mask, attention_mask=None, avail_mask=None):  # type: ignore[no-untyped-def]
        seen.append(state.detach().clone())
        return original(
            rtg, state, action, timestep, spatial_mask, attention_mask, avail_mask
        )

    agent.model.forward = spy  # type: ignore[union-attr, assignment]
    return seen


def _capture_bc(agent: BCAgent) -> list[torch.Tensor]:
    """Route B for ``BCAgent``: the ``(N, D)`` state tensor ``policy_logits`` consumed."""
    seen: list[torch.Tensor] = []
    original = agent.policy_logits

    def spy(state):  # type: ignore[no-untyped-def]
        seen.append(state.detach().clone())
        return original(state)

    agent.policy_logits = spy  # type: ignore[assignment]
    return seen


def _expected_rows() -> np.ndarray:
    """Route B's reference: ``(x - mean_i) / std_i`` per node, computed here from the statistics."""
    stats = _stats()
    raw = np.array(
        [[index + 1.0, index + 2.0, index + 3.0, index + 4.0] for index in range(N_NODES)],
        dtype=np.float32,
    )
    mean = np.stack([stats.state_mean[SCENARIO][node] for node in NODE_IDS])
    std = np.stack([stats.state_std[SCENARIO][node] for node in NODE_IDS])
    return (raw - mean) / std


# ----------------------------------------------------------------------
# ROUTE A -- the call log.  THIS is what kills the mutant.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("build", ["dt", "bc", "spatial"])
def test_every_intersection_is_normalised_with_its_own_statistics(
    build: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """🚨 ``DEFERRED`` 37, on 16 intersections, for all three per-intersection agents."""
    log = _CallLog()
    log.install(monkeypatch)

    common: dict[str, Any] = {
        "stats": _stats(),
        "scenario_id": SCENARIO,
        "state_dim": STATE_DIM,
        "seed": 5,
    }
    if build == "dt":
        agent: Any = DTAgent(_env(), context_length=3, n_layer=1, d_model=8, dropout=0.0,
                             max_ep_len=8, **common)
    elif build == "bc":
        agent = BCAgent(_env(), d_model=8, n_layer=1, dropout=0.0, **common)
    else:
        agent = SpatialDTAgent(_env(), adjacency=np.eye(N_NODES, dtype=np.bool_),
                               context_length=3, n_layer=1, d_model=8, dropout=0.0,
                               max_ep_len=8, **common)

    agent.act(_grid_info(0), explore=False)

    assert log.keys == list(NODE_IDS), (
        f"{build}: normalisation was requested for {log.keys[:4]}... instead of each "
        "intersection's own id; DEFERRED 37's mutant is live on 16 intersections"
    )
    assert len(set(log.keys)) == N_NODES


# ----------------------------------------------------------------------
# ROUTE B -- the tensor the network consumed.  Exact, float32, no tolerance.
# ----------------------------------------------------------------------


def test_the_dt_feeds_the_network_each_nodes_own_normalised_state() -> None:
    agent = DTAgent(_env(), context_length=3, n_layer=1, d_model=8, dropout=0.0, max_ep_len=8,
                    stats=_stats(), scenario_id=SCENARIO, state_dim=STATE_DIM, seed=5)
    seen = _capture_dt(agent)
    agent.act(_grid_info(0), explore=False)

    assert len(seen) == 1
    consumed = seen[0][:, -1, :].cpu().numpy()
    assert consumed.dtype == np.float32
    assert np.array_equal(consumed, _expected_rows())


def test_the_spatial_dt_feeds_the_network_each_nodes_own_normalised_state() -> None:
    agent = SpatialDTAgent(_env(), adjacency=np.eye(N_NODES, dtype=np.bool_), context_length=3,
                           n_layer=1, d_model=8, dropout=0.0, max_ep_len=8, stats=_stats(),
                           scenario_id=SCENARIO, state_dim=STATE_DIM, seed=5)
    seen = _capture_spatial(agent)
    agent.act(_grid_info(0), explore=False)

    assert len(seen) == 1
    consumed = seen[0][0, :, -1, :].cpu().numpy()
    assert consumed.dtype == np.float32
    assert np.array_equal(consumed, _expected_rows())


def test_the_baseline_feeds_the_network_each_nodes_own_normalised_state() -> None:
    agent = BCAgent(_env(), d_model=8, n_layer=1, dropout=0.0, stats=_stats(),
                    scenario_id=SCENARIO, state_dim=STATE_DIM, seed=5)
    seen = _capture_bc(agent)
    agent.act(_grid_info(0), explore=False)

    assert len(seen) == 1
    consumed = seen[0].cpu().numpy()
    assert consumed.dtype == np.float32
    assert np.array_equal(consumed, _expected_rows())


def test_route_b_would_not_catch_a_wrong_key_on_its_own() -> None:
    """Stated rather than implied: route B's reference shares the key with the agent.

    If every node were normalised with intersection 0's statistics, a naive recomputation that
    also used intersection 0 would agree.  This test makes that limitation explicit by showing the
    two disagree ONLY because the reference uses each node's own key.
    """
    stats = _stats()
    raw = np.array(
        [[index + 1.0, index + 2.0, index + 3.0, index + 4.0] for index in range(N_NODES)],
        dtype=np.float32,
    )
    first = NODE_IDS[0]
    wrong = (raw - stats.state_mean[SCENARIO][first]) / stats.state_std[SCENARIO][first]

    assert np.array_equal(wrong[0], _expected_rows()[0]), "node 0 must agree under both keys"
    assert not np.array_equal(wrong[1:], _expected_rows()[1:]), (
        "the other 15 must disagree, or the fixture cannot see the defect at all"
    )
