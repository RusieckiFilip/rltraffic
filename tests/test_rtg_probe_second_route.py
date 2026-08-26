"""⭐ The probe's critical quantity, recomputed by a SECOND, INDEPENDENT ROUTE (CLAUDE.md §2).

``flip_rate`` is **the** quantity P5.3a exists to produce, and until this file existed it was
computed exactly once, by one implementation, with nothing to disagree with.  The review found the
consequence and it is the worst possible shape: **mutant `m19` — scaling the RTG twice, the exact
hazard ``offline/rtg_ablation``'s own module docstring predicts at its lines 27-29 — turned the
headline into `0.000000` ("the token is inert") and left the entire 52-test suite green.**

What "independent" means here, precisely
-----------------------------------------
:func:`_independent_probe` imports **nothing** from ``offline.rtg_ablation``.  It re-derives, from
the contract rather than from the shipped code:

* the RTG series (``target - cumulative reward``, then ``/ rtg_scale`` at the point of use);
* the left-padded ``K``-step windows, including the rule that the **current** step's action slot is
  ``PAD_ACTION`` while earlier slots carry the **logged** action (teacher forcing);
* state normalisation, applied straight from the checkpoint's own ``stats`` payload;
* the masked argmax, the softmax, the total-variation distance and the mean absolute logit delta.

It then asserts **exact equality** with what the shipped path produces on the same cell.  Two
implementations agreeing to the last bit is evidence; one implementation agreeing with itself is not.

⚠️ **This test is deliberately slow-ish (~15 s) and deliberately not parametrised over all 40 cells.**
One cell is enough to kill a systematic error in the instrument, which is what the mutants are; 40
would buy resolution this test does not need and cost two minutes on every suite run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from offline.dataset import PAD_ACTION
from offline.dt_gate import HELD_OUT_DRAWS
from offline.method_tier_grid import DECLARED_GRADIENT_STEPS, env_settings_for_tiers, tier_spec
from offline.rtg_calibration import DECLARED_GRID, agent_with_target
from offline.trajectory_logger import load_episode

REPO = Path(__file__).resolve().parents[1]
CELL_TIER = "mappo1000"
CELL_SEED = 101
CELL_CHECKPOINT = "p4_dt/dt_seed101.pt"

#: Kept small on purpose: three streams reproduce a systematic instrument error just as surely as
#: twenty, and this test runs on every suite invocation.
N_STREAMS = 3


def _corpus_root() -> Path:
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO / "datasets_v11"
    if not candidate.is_dir():
        pytest.skip(
            f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to a "
            "collected corpus to run the second-route recomputation"
        )
    return candidate


def _checkpoint() -> Path:
    path = REPO / "output" / CELL_CHECKPOINT
    if not path.is_file():
        pytest.skip(f"checkpoint not present in this tree: {path}")
    return path


# ----------------------------------------------------------------------
# The independent implementation.  Nothing below imports offline.rtg_ablation.
# ----------------------------------------------------------------------


def _independent_series(target: float, rewards: np.ndarray, kind: str) -> np.ndarray:
    """``target - sum(r_k for k < t)``, by ``np.cumsum`` rather than by an accumulator loop."""
    values = np.asarray(rewards, dtype=np.float64)
    if kind == "zero":
        return np.zeros(values.shape, dtype=np.float64)
    if kind == "frozen":
        return np.full(values.shape, float(target), dtype=np.float64)
    consumed = np.concatenate([[0.0], np.cumsum(values)[:-1]])
    return float(target) - consumed


def _independent_normalise(state: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    """``(x - mean) / std`` in float32, read straight out of the checkpoint's statistics."""
    stats = payload["stats"]
    (scenario, per_ix), = stats["state_mean"].items()
    (ix_id, mean), = per_ix.items()
    std = np.asarray(stats["state_std"][scenario][ix_id], dtype=np.float32)
    safe = np.where(std > 0, std, np.float32(1.0)).astype(np.float32)
    return (np.asarray(state, dtype=np.float32) - np.asarray(mean, dtype=np.float32)) / safe


def _independent_logits(
    model: Any,
    payload: dict[str, Any],
    *,
    state: np.ndarray,
    avail: np.ndarray,
    action: np.ndarray,
    series: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Raw ``(T, A)`` logits, from windows built here rather than by ``DTAgent._window``."""
    config = model.config
    span = int(config.context_length)
    steps = int(action.size)
    scale = float(payload["rtg_scale"])
    rows = _independent_normalise(state, payload)

    rtg_in = np.zeros((steps, span, 1), dtype=np.float32)
    state_in = np.zeros((steps, span, rows.shape[1]), dtype=np.float32)
    action_in = np.full((steps, span), PAD_ACTION, dtype=np.int64)
    step_in = np.zeros((steps, span), dtype=np.int64)
    attend_in = np.zeros((steps, span), dtype=np.bool_)

    for t in range(steps):
        low = max(0, t - span + 1)
        start = span - (t - low + 1)
        for offset, source in enumerate(range(low, t + 1)):
            slot = start + offset
            # Scaled at the point of use and NOWHERE ELSE: the series above is unscaled.
            rtg_in[t, slot, 0] = np.float32(series[source] / scale)
            state_in[t, slot] = rows[source]
            step_in[t, slot] = source
            attend_in[t, slot] = True
            # Teacher forcing: earlier slots carry the LOGGED action; the current step's slot is
            # PAD, because at decision time that action has not been taken yet.
            if source < t:
                action_in[t, slot] = int(action[source])

    def _t(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(
                _t(rtg_in), _t(state_in), _t(action_in), _t(step_in), _t(attend_in), None
            )[:, -1]
    finally:
        model.train(was_training)
    return out.detach().cpu().numpy().astype(np.float32)


def _independent_compare(
    baseline: np.ndarray, other: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """Masked argmax, softmax, TVD and mean |logit delta| over legal actions, written out longhand."""
    def _mask(logits: np.ndarray) -> np.ndarray:
        out = logits.astype(np.float64).copy()
        legal_any = mask.any(axis=-1, keepdims=True)
        out[(~mask) & legal_any] = -np.inf
        return out

    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    left, right = _mask(baseline), _mask(other)
    flips = int(np.count_nonzero(left.argmax(axis=-1) != right.argmax(axis=-1)))
    tvd = 0.5 * np.abs(_softmax(left) - _softmax(right)).sum(axis=-1)
    delta = np.abs(baseline.astype(np.float64) - other.astype(np.float64))[mask]
    return {
        "flip_rate": flips / float(left.shape[0]),
        "tvd": float(tvd.mean()),
        "mean_abs_logit_delta": float(delta.mean()),
    }


def _independent_probe() -> dict[str, dict[str, float]]:
    """Every declared intervention on ``mappo1000@101``, computed without the shipped module."""
    from experiments.config import EnvSpec
    from experiments.envs import make_env

    root = _corpus_root()
    checkpoint = _checkpoint()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    target = float(payload["target_rtg"])
    spec = tier_spec(CELL_TIER)

    # The declared grid, restated here from its own constant, plus the two non-grid arms.
    arms: list[tuple[str, float, str]] = [("baseline", target, "decrement")]
    arms.extend(
        (f"grid_g{i}", float(value), "decrement") for i, value in enumerate(DECLARED_GRID)
    )
    arms.append(("zero", 0.0, "zero"))
    arms.append(("frozen", target, "frozen"))

    # The same three streams the shipped selection takes first, resolved from the corpus directly.
    # The tier's declared training set is every stream of the split ordered by
    # ``(dataset_dir, episode_file, ix_id)``; re-derived here from the manifests rather than taken
    # from ``training_streams``, so the ordering convention is checked too.
    directories = [root / name for name in spec.dirs]
    files: list[Path] = []
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        files.extend(directory / str(e["filename"]) for e in manifest["episodes"])
    files.sort(key=lambda p: (str(p.parent), p.name))
    chosen = [files[i] for i in (0, 10, 20)][:N_STREAMS]

    env = make_env(
        EnvSpec(
            id="cityflow1x1",
            backend="cityflow",
            paths={"config": str(_draw_config())},
            settings=env_settings_for_tiers([spec], root),
        )
    )
    try:
        agent = agent_with_target(
            env,
            checkpoint,
            declared_gradient_steps=DECLARED_GRADIENT_STEPS,
            target_rtg=target,
            device=None,
        )
        model = agent.model
        device = agent.device
        collected: dict[str, list[np.ndarray]] = {key: [] for key, _, _ in arms}
        masks: list[np.ndarray] = []
        for path in chosen:
            episode = load_episode(path)
            ix_id = episode.ix_ids[0]
            arrays = episode.intersections[ix_id]
            rewards = np.asarray(arrays.local_reward, dtype=np.float32)
            action = np.asarray(arrays.action, dtype=np.int64)
            avail = np.asarray(arrays.avail_mask, dtype=np.bool_)
            masks.append(avail[: action.size])
            for key, value, kind in arms:
                collected[key].append(
                    _independent_logits(
                        model,
                        payload,
                        state=np.asarray(arrays.state, dtype=np.float32),
                        avail=avail,
                        action=action,
                        series=_independent_series(value, rewards, kind),
                        device=device,
                    )
                )
    finally:
        env.close()

    mask = np.concatenate(masks)
    baseline = np.concatenate(collected["baseline"])
    return {
        key: _independent_compare(baseline, np.concatenate(collected[key]), mask)
        for key, _, _ in arms
    }


def _draw_config() -> Path:
    from offline.materialise_draws import draw_config_path

    return draw_config_path("cityflow1x1", int(HELD_OUT_DRAWS[0]))


# ----------------------------------------------------------------------
# The comparison
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def routes() -> tuple[dict[str, dict[str, float]], Any]:
    """Both routes over the same three streams: the independent one and the shipped one."""
    from offline.rtg_ablation import probe_cell

    independent = _independent_probe()

    root = _corpus_root()
    from offline.rtg_ablation import _tier_streams

    streams = list(_tier_streams(CELL_TIER, root))
    shipped = probe_cell(
        CELL_TIER,
        CELL_SEED,
        checkpoint_path=_checkpoint(),
        corpus_root=root,
        streams=streams,
        stream_indices=(0, 10, 20),
    )
    return independent, shipped


def test_the_shipped_probe_reproduces_an_independent_reimplementation_exactly(
    routes: tuple[dict[str, dict[str, float]], Any],
) -> None:
    """⭐ Two implementations, one number.  Exact equality on flip_rate; the last bit, not a tolerance.

    Kills **m19** (double scaling), **m16** (reading the wrong window slot) and **m15** (teacher
    forcing advanced with a wrong action) -- all three of which the previous suite left green.
    """
    independent, shipped = routes
    produced = {c.key: c for c in shipped.comparisons}
    assert set(produced) == set(independent)

    for key, expected in sorted(independent.items()):
        got = produced[key]
        # The headline is compared EXACTLY.  It is an integer count over a fixed denominator, so
        # there is no floating-point excuse available to it and none is granted.
        assert got.flip_rate == expected["flip_rate"], f"{key}: flip_rate"
        # Also EXACT, and measured to be so on all twelve interventions (worst relative
        # disagreement 0.000e+00): both routes reduce |logit delta| in float64 from the same
        # float32 logits, in the same order.
        assert got.mean_abs_logit_delta == expected["mean_abs_logit_delta"], (
            f"{key}: mean_abs_logit_delta"
        )
        # ⚠️ TVD is the ONE quantity that gets a tolerance, and the bound is measured rather than
        # guessed.  The shipped path takes `torch.softmax` over FLOAT32 logits; this route takes a
        # float64 softmax over the same values.  Two correct implementations therefore agree to
        # about six significant figures and not to the last bit -- the worst disagreement over all
        # twelve interventions is 1.125e-06 (on `grid_g6`), measured 2026-08-26.  This is the same
        # gap the review recorded as M5, and it is why the packet says TVD may not be quoted at
        # full precision.  Tightening this would condemn a correct implementation; making the
        # shipped path softmax in float64 would move every published TVD, which this fix round is
        # forbidden to do.
        assert got.tvd == pytest.approx(expected["tvd"], rel=5e-6, abs=1e-12), f"{key}: tvd"

    # R6's null control, confirmed by BOTH routes rather than by the shipped one alone.
    assert produced["grid_g5"].tvd == 0.0 and independent["grid_g5"]["tvd"] == 0.0
    assert produced["grid_g5"].flip_rate == 0.0 and independent["grid_g5"]["flip_rate"] == 0.0


def test_the_second_route_sees_a_non_zero_flip_rate_so_the_comparison_is_not_vacuous(
    routes: tuple[dict[str, dict[str, float]], Any],
) -> None:
    """Without this, two implementations agreeing on 0.0 everywhere would look like agreement.

    ``mappo1000`` is chosen for the second route precisely because it is one of the six tiers whose
    flip rate is non-zero; on ``fixedtime`` or ``random`` this comparison would be satisfied by any
    two implementations that both return zero, including two broken ones.
    """
    independent, _ = routes
    moved = [k for k, v in independent.items() if v["flip_rate"] > 0.0]
    assert moved, "the independent route found no flips at all: it cannot discriminate here"
    assert independent["baseline"]["flip_rate"] == 0.0
    assert max(v["flip_rate"] for v in independent.values()) > 0.0005


def test_scaling_the_rtg_twice_is_visible_to_the_second_route(
    routes: tuple[dict[str, dict[str, float]], Any],
) -> None:
    """⭐ m19 by name: the hazard ``offline/rtg_ablation``'s docstring predicts, made detectable.

    *"A probe that scaled twice would look entirely reasonable"* -- and it did: dividing the series
    by ``rtg_scale`` a second time drove every flip rate to exactly ``0.000000`` and the whole suite
    stayed green.  Here the independent route is re-run **with the double scaling applied** and must
    disagree with the shipped path, so the failure mode is caught by a comparison rather than by an
    inspection.
    """
    _, shipped = routes
    payload = torch.load(_checkpoint(), map_location="cpu", weights_only=False)
    scale = float(payload["rtg_scale"])
    target = float(payload["target_rtg"])

    from experiments.config import EnvSpec
    from experiments.envs import make_env

    root = _corpus_root()
    spec = tier_spec(CELL_TIER)
    directories = [root / name for name in spec.dirs]
    manifest = json.loads((directories[0] / "manifest.json").read_text(encoding="utf-8"))
    path = directories[0] / str(manifest["episodes"][0]["filename"])
    episode = load_episode(path)
    ix_id = episode.ix_ids[0]
    arrays = episode.intersections[ix_id]
    rewards = np.asarray(arrays.local_reward, dtype=np.float32)
    action = np.asarray(arrays.action, dtype=np.int64)
    avail = np.asarray(arrays.avail_mask, dtype=np.bool_)[: action.size]

    env = make_env(
        EnvSpec(
            id="cityflow1x1",
            backend="cityflow",
            paths={"config": str(_draw_config())},
            settings=env_settings_for_tiers([spec], root),
        )
    )
    try:
        agent = agent_with_target(
            env,
            _checkpoint(),
            declared_gradient_steps=DECLARED_GRADIENT_STEPS,
            target_rtg=target,
            device=None,
        )
        common = dict(
            state=np.asarray(arrays.state, dtype=np.float32),
            avail=avail,
            action=action,
            device=agent.device,
        )
        correct_base = _independent_logits(
            agent.model, payload, series=_independent_series(target, rewards, "decrement"), **common
        )
        correct_far = _independent_logits(
            agent.model,
            payload,
            series=_independent_series(float(DECLARED_GRID[-1]), rewards, "decrement"),
            **common,
        )
        doubled_base = _independent_logits(
            agent.model,
            payload,
            series=_independent_series(target, rewards, "decrement") / scale,
            **common,
        )
        doubled_far = _independent_logits(
            agent.model,
            payload,
            series=_independent_series(float(DECLARED_GRID[-1]), rewards, "decrement") / scale,
            **common,
        )
    finally:
        env.close()

    correct = _independent_compare(correct_base, correct_far, avail)
    doubled = _independent_compare(doubled_base, doubled_far, avail)

    # The double-scaled probe collapses the intervention into the noise floor...
    assert doubled["mean_abs_logit_delta"] < correct["mean_abs_logit_delta"] / 100.0
    # ...and the two therefore disagree, which is exactly what the shipped path must not do.
    assert doubled["mean_abs_logit_delta"] != correct["mean_abs_logit_delta"]
    shipped_far = {c.key: c for c in shipped.comparisons}["grid_g8"]
    assert shipped_far.mean_abs_logit_delta > doubled["mean_abs_logit_delta"] * 100.0
