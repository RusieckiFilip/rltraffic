"""P7.3d C3b: the cell on SIXTEEN intersections — the loader, the per-id prompt, the per-id series.

``BRIEF_39`` §3 C3b and §4 T-16, under Amendment A1: the registered subject is stored in the
spatial checkpoint format and is loaded by :class:`agent.SpatialDTAgent.SpatialDTAgent`, with the
16 calibrated targets applied AFTER load and read back per intersection.

**What this file guards is the registered prompt reaching the model.** A target that is silently
dropped, broadcast, or applied to the wrong intersection produces a complete, plausible episode —
`BRIEF_38` §2's seam 2 on hz1x1, one level up: there the single `rtg_first == target` refusal was
the only evidence the prompt took effect; here there are sixteen of them and fifteen agreeing must
not be able to hide the sixteenth.

GATES, each naming the artifact it consumes
-------------------------------------------
* ``RLTRAFFIC_OUTPUT_ROOT`` (default: this tree's ``output/``) holding
  ``p5_2/checkpoints/grid4x4_mappo1000_dt_nomix_h4_seed*.pt`` — A20(a)'s five, by digest.
* ``RLTRAFFIC_GRID4X4_RESCO``, ``RLTRAFFIC_DRAWS`` and SUMO for the one real grid4x4 episode.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import offline.rtg_calibration as rtg
import offline.transfer_calibration as calibration
import offline.transfer_curve as tcv

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID4X4_KEY = "cityflow_grid4x4"
CALIBRATION = REPO_ROOT / "docs/data/p7_3d_calibration.json"
N_INTERSECTIONS = 16
N_ACTIONS = 8
STATE_WIDTH = 40
T16_DRAW = 1000
T16_DECISIONS = 20


def _output_root() -> Path:
    return Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT", str(REPO_ROOT / "output")))


def _draws_root() -> Path:
    value = os.environ.get("RLTRAFFIC_DRAWS")
    return Path(value) if value else REPO_ROOT / "scenarios/draws"


def _checkpoint(seed: int = 101) -> Path:
    return (
        _output_root()
        / calibration.GRID4X4_CHECKPOINT_SUBDIR
        / f"{calibration.GRID4X4_CHECKPOINT_STEM}{seed}.pt"
    )


def _targets() -> dict[str, float]:
    artifact = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    return {
        ix: float(entry["budgets"]["k100"]["target"])
        for ix, entry in artifact["per_intersection"].items()
    }


def _needs_checkpoint() -> None:
    if not _checkpoint().is_file():
        pytest.skip(
            f"{_checkpoint()} is absent: set RLTRAFFIC_OUTPUT_ROOT to the output tree carrying "
            "p5_2/checkpoints (it is gitignored)"
        )


class _Ix:
    def __init__(self, ix_id: str) -> None:
        self.id = ix_id
        self.incoming_lanes: list[str] = []
        self.num_phases = N_ACTIONS


class _FakeEnv:
    """Sixteen intersections with the subject's own ids, enough for construction and one act()."""

    max_steps = 360
    delta_time = 10

    def __init__(self, ids: list[str]) -> None:
        self.intersections = [_Ix(ix) for ix in ids]

        class _Space:
            nvec = np.full(len(ids), N_ACTIONS)

        self.action_space = _Space()

    def info(self, step: int = 0, *, reward: float = -2.0) -> dict[str, Any]:
        """One ``info``.  ⚠️ At step 0 the reward is ``-0.0``, as the REAL reset info carries it.

        ``docs/data/p7_3a_zero_shot.json``'s 4,000 reward series all begin ``-0.0``: nothing has
        queued at t = 0.  The agent ignores the reset info's reward (no decision preceded it), so
        a fixture that put a non-zero number there would break the shift-by-one identity for a
        reason the real system never produces -- which is exactly what my first version did.
        """
        value = -0.0 if step == 0 else reward
        return {
            "step": step,
            "intersections": {
                str(ix.id): {
                    "state": [0.0] * STATE_WIDTH,
                    "avail_actions": list(range(N_ACTIONS)),
                    "reward": value,
                }
                for ix in self.intersections
            },
        }


# ==================================================================================
# The loader: the targets applied AFTER load, read back PER INTERSECTION
# ==================================================================================
def test_the_sixteen_targets_are_applied_after_load_and_read_back_per_intersection() -> None:
    """``SpatialDTAgent.load`` overwrites the prompt from the payload, so a target handed to the
    CONSTRUCTOR is discarded -- the defect ``agent_with_target`` exists for, one level up.

    *Mutation this is built against:* the read-back removed -> a target that never took effect
    produces a complete, plausible episode.
    """
    _needs_checkpoint()
    targets = _targets()
    env = _FakeEnv(sorted(targets))

    agent = rtg.spatial_agent_with_targets(
        env, _checkpoint(), declared_gradient_steps=40000, targets=targets, device="cpu"
    )

    current = agent.current_rtg()
    assert current == targets, "every intersection must condition on ITS OWN target"
    assert len(current) == N_INTERSECTIONS
    # ... and the checkpoint's own prompt is gone from every one of them.
    import torch

    payload = torch.load(_checkpoint(), map_location="cpu", weights_only=False)
    checkpoint_prompt = {str(k): float(v) for k, v in payload["target_rtg"].items()}
    assert checkpoint_prompt != targets, "the fixture is vacuous if the two already agree"
    for ix_id, value in current.items():
        assert value != checkpoint_prompt[ix_id], ix_id
    # rtg_scale is NEVER recalibrated (A17(a)): it stays the checkpoint's, per id.
    assert agent._rtg_scale == {str(k): float(v) for k, v in payload["rtg_scale"].items()}


def test_a_missing_or_extra_intersection_in_the_targets_is_refused_by_name() -> None:
    _needs_checkpoint()
    targets = _targets()
    env = _FakeEnv(sorted(targets))

    short = {ix: value for ix, value in targets.items() if ix != "C1"}
    with pytest.raises(ValueError, match="C1"):
        rtg.spatial_agent_with_targets(
            env, _checkpoint(), declared_gradient_steps=40000, targets=short, device="cpu"
        )
    extra = dict(targets)
    extra["Z9"] = -1.0
    with pytest.raises(ValueError, match="Z9"):
        rtg.spatial_agent_with_targets(
            env, _checkpoint(), declared_gradient_steps=40000, targets=extra, device="cpu"
        )


def test_a_checkpoint_at_the_wrong_budget_or_with_mixing_on_is_refused() -> None:
    """The declared budget AND the mixing flag, checked at THIS call site.

    ⚠️ ``spatial_mixing.assert_declared_budget`` runs its mixing check only for method names in
    ``DT_METHODS = ("dt_spatial", "dt_nomix")``; the registered subject's method name is
    ``dt_nomix_h4``, which is not in that tuple, so **that check is skipped for this subject** --
    measured 2026-09-21, and a mixing checkpoint really is accepted under the name. P5.2's module
    is not this task's to edit, so the loader asserts the flag itself.
    """
    _needs_checkpoint()
    targets = _targets()
    env = _FakeEnv(sorted(targets))

    with pytest.raises(ValueError, match="40000|39999"):
        rtg.spatial_agent_with_targets(
            env, _checkpoint(), declared_gradient_steps=39999, targets=targets, device="cpu"
        )

    mixing = _checkpoint().parent / "grid4x4_mappo1000_dt_spatial_h4_seed101.pt"
    if not mixing.is_file():
        pytest.skip(f"{mixing} is absent, so the mixing control cannot be run")
    with pytest.raises(ValueError, match="spatial_mixing|mixing"):
        rtg.spatial_agent_with_targets(
            env, mixing, declared_gradient_steps=40000, targets=targets, device="cpu"
        )


# ==================================================================================
# dt_choose on sixteen intersections
# ==================================================================================
def test_dt_choose_records_sixteen_rtg_and_sixteen_reward_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One series PER intersection, keyed by id, in the env's order (contract C1).

    The RTG is read BEFORE ``act``, which is what makes Amendment D1's shift-by-one the right
    comparison: ``rtg[t] - rtg[t-1] == -reward[t-1]``.

    ⚠️ ``reward[0]`` is a NUMBER, not ``None``.  I first asserted ``None`` here -- "no reward
    precedes the first decision" -- and the committed ``docs/data/p7_3a_zero_shot.json`` says
    otherwise: its 4,000 reward series all begin ``-0.0``, because the reset ``info`` carries the
    per-intersection reward key with nothing queued yet.  ``dt_choose``'s ``None`` branch is for
    an env with no ``local_reward_fn`` at all, which is a different situation.  The shift-by-one
    identity is unaffected and is what is asserted.
    """
    _needs_checkpoint()
    targets = _targets()
    env = _FakeEnv(sorted(targets))

    choose, diagnostics = tcv.dt_choose(
        env, checkpoint_path=_checkpoint(), target_rtg=targets,
        declared_gradient_steps=40000, device="cpu",
    )
    assert sorted(diagnostics["intersections"]) == sorted(targets)
    for step in range(3):
        action = choose(env, env.info(step, reward=-2.0))
        assert np.asarray(action).shape == (N_INTERSECTIONS,)

    for ix_id in targets:
        rtgs = diagnostics["rtg_series"][ix_id]
        rewards = diagnostics["reward_series"][ix_id]
        assert len(rtgs) == len(rewards) == 3, ix_id
        assert rtgs[0] == targets[ix_id], f"{ix_id}: rtg_first must be ITS OWN target"
        assert rewards[0] is not None, "the reset info carries the reward key (see the docstring)"
        for t in range(1, 3):
            assert rtgs[t] - rtgs[t - 1] == -rewards[t - 1], (ix_id, t)
    assert len(diagnostics["actions"]) == 3
    assert np.asarray(diagnostics["actions"]).shape == (3, N_INTERSECTIONS)


def test_the_per_intersection_rtg_refusal_names_the_intersection_that_disagrees() -> None:
    """``validate_cell_payload``'s ``rtg_first == target`` refusal, applied per id.

    *Mutation this is built against (§4 T-16's named one):* one target swapped between two
    intersections -> fifteen still agree -> the refusal must fire on the two that do not.
    """
    targets = {f"A{i}": float(-100 - i) for i in range(4)}
    series = {ix: [value, value + 1.0] for ix, value in targets.items()}
    tcv.assert_rtg_first_matches_targets(series, targets, label="cell_x")

    swapped = dict(series)
    swapped["A1"], swapped["A2"] = series["A2"], series["A1"]
    with pytest.raises(ValueError, match=r"A1.*A2|A2.*A1"):
        tcv.assert_rtg_first_matches_targets(swapped, targets, label="cell_x")

    missing = {ix: values for ix, values in series.items() if ix != "A3"}
    with pytest.raises(ValueError, match="A3"):
        tcv.assert_rtg_first_matches_targets(missing, targets, label="cell_x")

    empty = dict(series)
    empty["A0"] = []
    with pytest.raises(ValueError, match="A0"):
        tcv.assert_rtg_first_matches_targets(empty, targets, label="cell_x")


# ==================================================================================
# The chunk name and the artifact version: one scheme, two campaigns
# ==================================================================================
def test_the_chunk_name_gains_the_scenario_for_non_default_scenarios_only() -> None:
    """P7.3b's stage-identity lesson: two campaigns sharing a naming scheme share a directory.

    hz1x1's names may NOT move -- 4,704 + 700 chunks on disk are keyed by them.
    """
    hz_cell = {"subject": "mappo1000", "arm": "b_mean_k100", "seed": 101, "draw_id": 1000}
    assert tcv.cell_chunk_name(hz_cell) == "cell_mappo1000_b_mean_k100_seed101_draw1000.json"
    assert tcv.cell_chunk_name({**hz_cell, "scenario": "cityflow1x1"}) == tcv.cell_chunk_name(hz_cell)

    grid = {**hz_cell, "subject": "mappo1000_dt_nomix_h4", "scenario": GRID4X4_KEY}
    name = tcv.cell_chunk_name(grid)
    assert name.startswith(f"cell_{GRID4X4_KEY}_"), name
    assert name != tcv.cell_chunk_name(hz_cell)
    anchor = {"subject": None, "arm": "fixedtime", "seed": None, "draw_id": 1000,
              "scenario": GRID4X4_KEY}
    assert tcv.cell_chunk_name(anchor) == f"cell_{GRID4X4_KEY}_anchor_fixedtime_seednone_draw1000.json"


def test_the_grid4x4_artifact_has_its_own_format_version() -> None:
    """Overloading P7.3a's version would make one string describe two artifacts."""
    assert tcv.ARTIFACT_FORMAT_VERSION == "p7.3a-zero-shot/1.0"
    assert tcv.GRID4X4_ARTIFACT_FORMAT_VERSION == "p7.3d-grid4x4/1.0"
    assert tcv.GRID4X4_ARTIFACT_FORMAT_VERSION != tcv.ARTIFACT_FORMAT_VERSION


def test_the_calibration_artifact_is_pinned_by_digest_and_read_not_recomputed() -> None:
    """E3(a)'s rule, for grid4x4: a target read from an unpinned artifact can be edited between
    the calibration and the campaign without leaving a trace."""
    import hashlib

    digest = hashlib.sha256(CALIBRATION.read_bytes()).hexdigest()
    assert tcv.P7_3D_CALIBRATION_SHA256 == digest, (
        "the pin and the committed artifact disagree; the pin moves only in a commit that also "
        "moves the artifact"
    )
    loaded = tcv.load_grid4x4_targets(data_dir=CALIBRATION.parent)
    assert loaded == _targets() and len(loaded) == N_INTERSECTIONS
    assert all(isinstance(v, float) for v in loaded.values())


def test_a_tampered_calibration_artifact_is_refused(tmp_path: Path) -> None:
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    payload["per_intersection"]["A0"]["budgets"]["k100"]["target"] = -1.0
    (tmp_path / "p7_3d_calibration.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256|digest"):
        tcv.load_grid4x4_targets(data_dir=tmp_path)


# ==================================================================================
# T-16 — twenty decisions of the real subject on a real grid4x4 SUMO episode
# ==================================================================================
def test_t16_twenty_decisions_of_seed101_on_draw_1000(monkeypatch: pytest.MonkeyPatch) -> None:
    """§4 T-16, the cell's mechanics: 16 RTG series with ``rtg_first_i == target_i`` for every i,
    16 reward series obeying D1's shift-by-one, actions inside the env's 8, teleports 0."""
    import shutil

    from offline.materialise_draws import parity_sumocfg_path

    _needs_checkpoint()
    try:
        import traci  # noqa: F401
    except ImportError:
        pytest.skip("traci is not importable")
    if shutil.which("sumo") is None:
        pytest.skip("the sumo binary is not on PATH")
    if not os.environ.get("RLTRAFFIC_GRID4X4_RESCO"):
        pytest.skip("RLTRAFFIC_GRID4X4_RESCO is unset")
    config = parity_sumocfg_path(GRID4X4_KEY, T16_DRAW, out_root=_draws_root())
    if not config.is_file():
        pytest.skip(f"{config} is absent: P7.3d C1 renders the held-out band")

    from offline.aligned_env import aligned_observer_env_for_draw

    targets = _targets()
    env = aligned_observer_env_for_draw(
        GRID4X4_KEY, T16_DRAW, out_root=_draws_root(), halting_check=False, arm="maxpressure"
    )
    try:
        ids = [str(ix.id) for ix in env.intersections]
        assert ids == list(targets), "the env's order IS the calibration artifact's order"
        choose, diagnostics = tcv.dt_choose(
            env, checkpoint_path=_checkpoint(), target_rtg=targets,
            declared_gradient_steps=40000,
        )
        info = env.reset(seed=1000)
        for _ in range(T16_DECISIONS):
            action = choose(env, info)
            _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:  # pragma: no cover - 20 << 360
                break
        teleports = len(env._sumo.simulation.getStartingTeleportIDList())
        types_seen = {env._sumo.vehicle.getTypeID(v) for v in env._sumo.vehicle.getIDList()}
    finally:
        env.close()

    assert teleports == 0 and types_seen == {"cf_parity"}
    actions = np.asarray(diagnostics["actions"])
    assert actions.shape == (T16_DECISIONS, N_INTERSECTIONS)
    assert int(actions.min()) >= 0 and int(actions.max()) < N_ACTIONS
    assert sorted(diagnostics["rtg_series"]) == sorted(targets)
    for ix_id in targets:
        rtgs = diagnostics["rtg_series"][ix_id]
        rewards = diagnostics["reward_series"][ix_id]
        assert len(rtgs) == T16_DECISIONS, ix_id
        assert rtgs[0] == targets[ix_id], f"{ix_id}: rtg_first is not its own target"
        assert rewards[0] is not None, "the reset info carries the reward key"
        for t in range(1, T16_DECISIONS):
            assert rtgs[t] - rtgs[t - 1] == -rewards[t - 1], (ix_id, t)
    # The sixteen series are not one series copied: the intersections differ.
    firsts = {diagnostics["rtg_series"][ix][0] for ix in targets}
    assert len(firsts) > 1, "every intersection conditioned on the same number"
    tcv.assert_rtg_first_matches_targets(diagnostics["rtg_series"], targets, label="t16")


# ==================================================================================
# The two guards nothing else forces: the read-back, and the role check
# ==================================================================================
def test_the_read_back_fires_when_the_prompt_does_not_actually_take_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The read-back is defence in depth, so only a forced mismatch can exercise it.

    Removing it survived every other test in this file -- correctly, because with the assignment
    working the agent DOES condition on the targets.  What the guard protects against is the
    assignment silently not taking effect, which is exactly what happened on hz1x1 before
    ``agent_with_target`` existed.  ``current_rtg`` is made to report something else, and the
    loader must refuse rather than hand back an agent that is not conditioned as asked.

    *Mutation this is built against:* the per-id comparison replaced by an empty list -> dies.
    """
    _needs_checkpoint()
    from agent.SpatialDTAgent import SpatialDTAgent

    targets = _targets()
    env = _FakeEnv(sorted(targets))
    wrong = {ix: value - 1.0 for ix, value in targets.items()}
    monkeypatch.setattr(SpatialDTAgent, "current_rtg", lambda self: dict(wrong), raising=True)

    with pytest.raises(ValueError, match="did not take effect"):
        rtg.spatial_agent_with_targets(
            env, _checkpoint(), declared_gradient_steps=40000, targets=targets, device="cpu"
        )


def test_a_budget_the_registration_never_evaluates_cannot_become_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A20(b) records k = 5 and k = 20 and evaluates neither; only k = 100 carries the role.

    The digest pin refuses a tampered artifact first, which is why this test pins the digest to
    the tampered file and so exercises the ROLE check alone -- otherwise the role check is
    unreachable and removing it survives every test.

    *Mutation this is built against:* the role check dropped -> a one-field edit promotes a
    recorded-only budget to the registered prompt -> dies.
    """
    import hashlib

    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    payload["per_intersection"]["A0"]["budgets"]["k100"]["role"] = "recorded_not_evaluated"
    target = tmp_path / "p7_3d_calibration.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        tcv, "P7_3D_CALIBRATION_SHA256",
        hashlib.sha256(target.read_bytes()).hexdigest(), raising=True,
    )

    with pytest.raises(ValueError, match=r"A0.*recorded_not_evaluated|role"):
        tcv.load_grid4x4_targets(data_dir=tmp_path)
