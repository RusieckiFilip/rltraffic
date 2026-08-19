"""Tests for ``offline.tier_sweep`` -- P5.2's tier sweep and head-count 2x2.

Written BEFORE the implementation, against a signature-only skeleton, so each test fails for its
own reason rather than sharing one import error.  The obligations these discharge are numbered in
``docs/plans/p5.2.md`` section 6; the number is named in each test's docstring so a reader can go
from a failure to the requirement that asked for it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from offline import tier_sweep as ts

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "docs" / "plans" / "p5.2.md"


# ----------------------------------------------------------------------
# Obligation 11(a) -- output/p5_1/ is read-only, enforced in code.
# A STRING comparison passes three of these four evasions.
# ----------------------------------------------------------------------


def test_assert_writable_refuses_the_protected_root_given_absolutely(tmp_path: Path) -> None:
    """Obligation 11(a): the plain case, given as an absolute path."""
    protected = ts.protected_roots_from([tmp_path / "p5_1"])
    (tmp_path / "p5_1").mkdir()
    with pytest.raises(PermissionError, match="read-only"):
        ts.assert_writable(tmp_path / "p5_1" / "eval_bc.json", protected)


def test_assert_writable_refuses_a_relative_path_into_the_protected_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Obligation 11(a): a relative path resolves against the cwd and must still be caught."""
    (tmp_path / "p5_1").mkdir()
    protected = ts.protected_roots_from([tmp_path / "p5_1"])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PermissionError, match="read-only"):
        ts.assert_writable(Path("p5_1") / "eval_bc.json", protected)


def test_assert_writable_refuses_a_dotdot_traversal_that_lands_inside(tmp_path: Path) -> None:
    """Obligation 11(a): ``work/../p5_1/x`` is textually different and resolves inside."""
    (tmp_path / "p5_1").mkdir()
    (tmp_path / "work").mkdir()
    protected = ts.protected_roots_from([tmp_path / "p5_1"])
    with pytest.raises(PermissionError, match="read-only"):
        ts.assert_writable(tmp_path / "work" / ".." / "p5_1" / "eval_bc.json", protected)


def test_assert_writable_refuses_a_symlink_pointing_into_the_protected_root(
    tmp_path: Path,
) -> None:
    """Obligation 11(a): the evasion a textual check cannot see at all."""
    (tmp_path / "p5_1").mkdir()
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "link").symlink_to(tmp_path / "p5_1", target_is_directory=True)
    protected = ts.protected_roots_from([tmp_path / "p5_1"])
    with pytest.raises(PermissionError, match="read-only"):
        ts.assert_writable(tmp_path / "work" / "link" / "eval_bc.json", protected)


def test_assert_writable_accepts_a_path_outside_the_protected_root(tmp_path: Path) -> None:
    """Obligation 11(a)'s ACCEPT CONTROL: a guard that refuses everything is not a guard.

    Without this, all four refusals above are satisfied by ``raise PermissionError`` on line one.
    """
    (tmp_path / "p5_1").mkdir()
    (tmp_path / "p5_2").mkdir()
    protected = ts.protected_roots_from([tmp_path / "p5_1"])
    resolved = ts.assert_writable(tmp_path / "p5_2" / "eval_bc.json", protected)
    assert resolved == (tmp_path / "p5_2" / "eval_bc.json").resolve()


def test_assert_writable_refuses_the_protected_root_itself_not_only_children(
    tmp_path: Path,
) -> None:
    """Obligation 11(a): ``rmtree(root)`` is the deletion that would cost the most."""
    (tmp_path / "p5_1").mkdir()
    protected = ts.protected_roots_from([tmp_path / "p5_1"])
    with pytest.raises(PermissionError, match="read-only"):
        ts.assert_writable(tmp_path / "p5_1", protected)


def test_a_sibling_whose_name_merely_starts_with_the_root_is_not_protected(
    tmp_path: Path,
) -> None:
    """Obligation 11(a): ``p5_1_backup`` must NOT be caught by a prefix test on the string."""
    (tmp_path / "p5_1").mkdir()
    (tmp_path / "p5_1_backup").mkdir()
    protected = ts.protected_roots_from([tmp_path / "p5_1"])
    resolved = ts.assert_writable(tmp_path / "p5_1_backup" / "x.json", protected)
    assert resolved.name == "x.json"


# ----------------------------------------------------------------------
# Obligation 11(b) -- a refused write creates nothing.
# ----------------------------------------------------------------------


def _tree(root: Path) -> set[str]:
    """Every path under *root*, relative and sorted -- the exact set, never a count."""
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def test_a_refused_write_creates_no_file_and_no_directory(tmp_path: Path) -> None:
    """Obligation 11(b): the filesystem-mutation barrier, asserted on the exact file set."""
    (tmp_path / "p5_1").mkdir()
    protected = ts.protected_roots_from([tmp_path / "p5_1"])
    before = _tree(tmp_path)
    with pytest.raises(PermissionError, match="read-only"):
        ts.write_json_guarded({"a": 1}, tmp_path / "p5_1" / "deep" / "x.json", protected)
    assert _tree(tmp_path) == before


def test_write_json_guarded_writes_when_the_path_is_allowed(tmp_path: Path) -> None:
    """The accept control for obligation 11(b), and it pins the payload round-trip."""
    (tmp_path / "p5_1").mkdir()
    protected = ts.protected_roots_from([tmp_path / "p5_1"])
    destination = tmp_path / "out.json"
    ts.write_json_guarded({"a": 1, "b": [2, 3]}, destination, protected)
    assert json.loads(destination.read_text(encoding="utf-8")) == {"a": 1, "b": [2, 3]}


def test_replace_guarded_refuses_to_move_onto_a_protected_destination(tmp_path: Path) -> None:
    """Obligation 11(b): a move is a write AND a delete; the destination is guarded too."""
    (tmp_path / "p5_1").mkdir()
    source = tmp_path / "new.json"
    source.write_text("{}", encoding="utf-8")
    victim = tmp_path / "p5_1" / "eval_bc.json"
    victim.write_text('{"cell": "the only copy"}', encoding="utf-8")
    protected = ts.protected_roots_from([tmp_path / "p5_1"])
    with pytest.raises(PermissionError, match="read-only"):
        ts.replace_guarded(source, victim, protected)
    assert victim.read_text(encoding="utf-8") == '{"cell": "the only copy"}'
    assert source.exists()


# ----------------------------------------------------------------------
# Obligation 11(c) -- a half-written cell is never mistaken for a complete one.
# ----------------------------------------------------------------------


def _cell_payload(
    *, seeds: tuple[int, ...], draws: tuple[int, ...], steps: int = 40_000, method: str = "bc"
) -> dict[str, Any]:
    return {
        "method": method,
        "declared_gradient_steps": steps,
        "episodes": [
            {"seed": s, "draw_id": d, "att_horizon": 100.0 + d, "horizon_vehicle_count": 1.0,
             "episode_reward": -1.0, "arm": f"{method}@t"}
            for s in seeds
            for d in draws
        ],
    }


def test_a_complete_cell_is_complete() -> None:
    """Obligation 11(c)'s accept control."""
    payload = _cell_payload(seeds=(101, 202), draws=(1000, 1001))
    assert ts.cell_is_complete(
        payload, seeds=(101, 202), draws=(1000, 1001), declared_steps=40_000, method="bc"
    )


def test_a_cell_missing_one_episode_is_not_complete() -> None:
    """Obligation 11(c): the crash case -- a truncated cell must not be resumed over."""
    payload = _cell_payload(seeds=(101, 202), draws=(1000, 1001))
    payload["episodes"].pop()
    assert not ts.cell_is_complete(
        payload, seeds=(101, 202), draws=(1000, 1001), declared_steps=40_000, method="bc"
    )


def test_a_cell_at_the_wrong_budget_is_not_complete() -> None:
    """Obligation 11(c): a rehearsal at 3 steps may not be resumed into a reported cell."""
    payload = _cell_payload(seeds=(101,), draws=(1000,), steps=3)
    assert not ts.cell_is_complete(
        payload, seeds=(101,), draws=(1000,), declared_steps=40_000, method="bc"
    )


def test_a_cell_of_the_wrong_arm_is_not_complete() -> None:
    """Obligation 11(c): the arms are weight-compatible, so nothing else would catch a swap."""
    payload = _cell_payload(seeds=(101,), draws=(1000,), method="bc")
    assert not ts.cell_is_complete(
        payload, seeds=(101,), draws=(1000,), declared_steps=40_000, method="iql"
    )


def test_a_duplicated_episode_does_not_make_a_short_cell_look_complete() -> None:
    """Obligation 11(c): completeness is a SET identity, not a count -- a count is fooled here."""
    payload = _cell_payload(seeds=(101,), draws=(1000, 1001))
    payload["episodes"][1] = dict(payload["episodes"][0])
    assert not ts.cell_is_complete(
        payload, seeds=(101,), draws=(1000, 1001), declared_steps=40_000, method="bc"
    )


# ----------------------------------------------------------------------
# Obligation 6b (C2 / D2) -- the schedule the campaign actually runs.
# ----------------------------------------------------------------------


def test_warmup_is_a_function_of_the_budget() -> None:
    """Obligation 6b: warmup is ``min(1000, total // 2)``, so a short run has its own boundary."""
    assert (ts.warmup_for(100), ts.warmup_for(500), ts.warmup_for(40_000)) == (50, 250, 1000)


def test_the_lr_multiplier_matches_the_registered_probe_exactly() -> None:
    """Obligation 6b: the values pinned in the plan, at the budget the campaign runs.

    ``==`` and not ``approx``: these are the registered numbers, and D2's cosine margin at step 500
    is 5.7e-4, which any tolerance worth writing would swallow.
    """
    warmup = ts.warmup_for(40_000)
    measured = tuple(ts.lr_multiplier(step, warmup) for step in ts.LR_PROBE_STEPS)
    assert measured == ts.LR_PROBE_EXPECTED


def test_step_249_separates_the_ramp_shapes_and_step_499_does_not() -> None:
    """D2-CORRECTED: the discriminating power of the probe point, measured rather than asserted.

    The midpoint of a monotone ramp is where endpoint-matched shapes cross, so it is the LEAST
    informative interior point.  This test is why step 249 is in the probe and step 499 is not.
    """
    warmup = ts.warmup_for(40_000)

    def cosine(step: int) -> float:
        return 0.5 * (1.0 - math.cos(math.pi * min(1.0, (step + 1) / warmup)))

    gap_at_249 = abs(cosine(249) - ts.lr_multiplier(249, warmup))
    gap_at_499 = abs(cosine(499) - ts.lr_multiplier(499, warmup))
    assert gap_at_499 < 1e-15
    assert gap_at_249 > 0.1
    assert gap_at_249 / max(gap_at_499, 1e-300) > 1e13


def test_249_is_in_the_probe_and_the_blind_midpoint_is_not() -> None:
    """D2-CORRECTED: the probe set itself, so a later edit cannot quietly drop the sharp point."""
    assert 249 in ts.LR_PROBE_STEPS
    assert 499 not in ts.LR_PROBE_STEPS
    assert len(ts.LR_PROBE_STEPS) == len(ts.LR_PROBE_EXPECTED)


def test_the_old_trainer_builds_the_same_schedule_at_the_campaign_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Obligation 6b: OLD vs NEW agreement, on the real lambda, at ``total = 40,000``.

    The old trainer's ramp lives in a closure inside ``train_spatial_dt``.  It is captured by
    intercepting the ``LambdaLR`` the trainer constructs and aborting before the first gradient
    step -- so this exercises the real code path at the real budget and trains nothing.  Reading
    the formula out of the source text instead would be the ``inspect.getsource`` theatre P5.1's
    review rejected.
    """
    import torch

    from offline import spatial_mixing as sm

    captured: dict[str, Any] = {}

    class _Abort(RuntimeError):
        pass

    def fake_lambda_lr(optimiser: Any, fn: Any) -> Any:
        captured["fn"] = fn
        raise _Abort("captured the schedule")

    monkeypatch.setattr(torch.optim.lr_scheduler, "LambdaLR", fake_lambda_lr)
    setup = _tiny_joint_setup()
    with pytest.raises(_Abort, match="captured the schedule"):
        sm.train_spatial_dt(
            setup["stacked"],
            setup["index"],
            method="dt_nomix",
            seed=101,
            adjacency=setup["adjacency"],
            prompts=setup["prompts"],
            stats=setup["stats"],
            state_dim=setup["state_dim"],
            n_actions=setup["n_actions"],
            gradient_steps=40_000,
            batch_size=2,
            device=torch.device("cpu"),
            checkpoint_path=setup["checkpoint_path"],
            provenance={},
        )

    old = captured["fn"]
    warmup = ts.warmup_for(40_000)
    for step in ts.LR_PROBE_STEPS:
        assert old(step) == ts.lr_multiplier(step, warmup), f"schedules differ at step {step}"


# ----------------------------------------------------------------------
# Obligation 6 -- the fixture the equivalence run and the schedule capture share.
# ----------------------------------------------------------------------


def _tiny_joint_setup(n_nodes: int = 2, context: int = 3, state_dim: int = 4) -> dict[str, Any]:
    """A minimal joint training setup: real dataclasses, synthetic tensors, two nodes."""
    import torch

    from offline.dataset import NormalizationStats
    from offline.joint_windows import JointWindowIndex
    from offline.roadnet_graph import AdjacencySpec
    from offline.spatial_mixing import NodePrompt

    n_actions = 3
    n_windows = 4
    rows = n_windows * n_nodes
    node_ids = tuple(f"N{i}" for i in range(n_nodes))
    generator = np.random.default_rng(7)
    stacked = {
        "state": torch.from_numpy(
            generator.standard_normal((rows, context, state_dim)).astype(np.float32)
        ),
        "rtg": torch.from_numpy(
            generator.standard_normal((rows, context, 1)).astype(np.float32)
        ),
        "action": torch.from_numpy(
            generator.integers(0, n_actions, size=(rows, context)).astype(np.int64)
        ),
        "timestep": torch.from_numpy(
            np.tile(np.arange(context, dtype=np.int64), (rows, 1))
        ),
        "attention_mask": torch.ones((rows, context), dtype=torch.bool),
        "avail_mask": torch.ones((rows, context, n_actions), dtype=torch.bool),
        "member_index": torch.from_numpy(
            np.arange(rows, dtype=np.int64).reshape(n_windows, n_nodes)
        ),
    }
    directed = np.zeros((n_nodes, n_nodes), dtype=np.bool_)
    directed[0, 1] = True
    adjacency = AdjacencySpec(
        node_ids=node_ids,
        directed=directed,
        undirected=directed | directed.T,
        roadnet_path="synthetic",
        roadnet_sha256="0" * 64,
    )
    index = JointWindowIndex(
        node_ids=node_ids,
        member_index=np.arange(rows, dtype=np.int64).reshape(n_windows, n_nodes),
        episode_index=np.zeros(n_windows, dtype=np.int64),
        t=np.arange(n_windows, dtype=np.int64),
        state_dim=state_dim,
        n_actions=n_actions,
    )
    stats = NormalizationStats(
        stats_version="test/1.0",
        split="train",
        draw_ids=(1,),
        dataset_dirs=("synthetic",),
        state_mean={},
        state_std={},
        row_count={},
        rtg={},
    )
    prompts = {
        node: NodePrompt(
            ix_id=node, target_rtg=-1.0, rtg_scale=10.0, n_streams=1,
            return_min=-2.0, return_max=-1.0,
        )
        for node in node_ids
    }
    import tempfile

    directory = Path(tempfile.mkdtemp())
    return {
        "stacked": stacked,
        "index": index,
        "adjacency": adjacency,
        "prompts": prompts,
        "stats": stats,
        "state_dim": state_dim,
        "n_actions": n_actions,
        "checkpoint_path": directory / "ckpt.pt",
    }


# ----------------------------------------------------------------------
# Obligation 6 -- the new trainer IS the old trainer at one head.
# ----------------------------------------------------------------------


def test_the_new_trainer_reproduces_the_old_one_exactly_at_one_head(tmp_path: Path) -> None:
    """Obligation 6: old vs new, EXACT, on CPU.

    The bar is ``==`` and not ``allclose``, and the bar was chosen by the C1 control rather than by
    preference: the control showed ``train_spatial_dt`` reproduces itself exactly on CPU (0/66
    tensors) and NOT on CUDA (61-63/66).  So CPU equality is the available bar, and it is asserted
    exactly (``BRIEF_27`` C1b).

    ⚠️ This licenses the CODE PATH only.  B3's digest check and the ``random``-anchor re-roll are
    what license the reused ARTIFACTS; neither substitutes for the other.
    """
    import torch

    from offline import spatial_mixing as sm

    setup = _tiny_joint_setup()
    common = dict(
        seed=101,
        adjacency=setup["adjacency"],
        prompts=setup["prompts"],
        stats=setup["stats"],
        state_dim=setup["state_dim"],
        n_actions=setup["n_actions"],
        gradient_steps=60,
        batch_size=2,
        device=torch.device("cpu"),
        provenance={},
    )
    old = sm.train_spatial_dt(
        setup["stacked"], setup["index"], method="dt_nomix",
        checkpoint_path=tmp_path / "old.pt", **common,
    )
    new = ts.train_tier_dt(
        setup["stacked"], setup["index"], tier="mappo1000", method="dt_nomix",
        checkpoint_path=tmp_path / "new.pt", protected=(), **common,
    )
    assert new.losses == old.losses, "the loss sequences diverge"

    old_model = torch.load(tmp_path / "old.pt", map_location="cpu", weights_only=False)["model"]
    new_model = torch.load(tmp_path / "new.pt", map_location="cpu", weights_only=False)["model"]
    assert set(new_model) == set(old_model)
    differing = [k for k in sorted(old_model) if not torch.equal(old_model[k], new_model[k])]
    assert differing == [], f"{len(differing)} tensors differ, first {differing[:4]}"


def test_the_equivalence_check_would_notice_a_changed_trainer(tmp_path: Path) -> None:
    """Obligation 6's discriminating power: the same comparison on a DIFFERENT arm must differ.

    Without this, ``the tensors are equal`` could mean the comparison is insensitive.  Training the
    mixing arm instead of the control changes the mask and must move the weights.
    """
    import torch

    from offline import spatial_mixing as sm

    setup = _tiny_joint_setup()
    common = dict(
        seed=101, adjacency=setup["adjacency"], prompts=setup["prompts"], stats=setup["stats"],
        state_dim=setup["state_dim"], n_actions=setup["n_actions"], gradient_steps=60,
        batch_size=2, device=torch.device("cpu"), provenance={},
    )
    sm.train_spatial_dt(
        setup["stacked"], setup["index"], method="dt_nomix",
        checkpoint_path=tmp_path / "control.pt", **common,
    )
    ts.train_tier_dt(
        setup["stacked"], setup["index"], tier="mappo1000", method="dt_spatial",
        checkpoint_path=tmp_path / "treatment.pt", protected=(), **common,
    )
    a = torch.load(tmp_path / "control.pt", map_location="cpu", weights_only=False)["model"]
    b = torch.load(tmp_path / "treatment.pt", map_location="cpu", weights_only=False)["model"]
    differing = [k for k in sorted(a) if not torch.equal(a[k], b[k])]
    assert differing, "the comparison cannot tell two different models apart"


def test_the_head_count_reaches_the_model_and_is_recorded(tmp_path: Path) -> None:
    """A2: the 4-head arms must actually be four-headed, and the checkpoint must say so."""
    import torch

    setup = _tiny_joint_setup()
    common = dict(
        seed=101, adjacency=setup["adjacency"], prompts=setup["prompts"], stats=setup["stats"],
        state_dim=setup["state_dim"], n_actions=setup["n_actions"], gradient_steps=2,
        batch_size=2, device=torch.device("cpu"), provenance={}, protected=(),
    )
    ts.train_tier_dt(
        setup["stacked"], setup["index"], tier="mappo1000", method="dt_spatial_h4",
        checkpoint_path=tmp_path / "h4.pt", **common,
    )
    payload = torch.load(tmp_path / "h4.pt", map_location="cpu", weights_only=False)
    assert payload["config"]["n_head"] == 4
    assert payload["provenance"]["n_head"] == 4
    assert payload["provenance"]["tier"] == "mappo1000"
    assert payload["provenance"]["deterministic"] is False


@pytest.mark.parametrize(
    ("method", "mixes"),
    [
        ("dt_spatial", True),
        ("dt_spatial_h4", True),
        ("dt_nomix", False),
        ("dt_nomix_h4", False),
    ],
)
def test_each_arm_of_the_2x2_gets_the_graph_its_name_promises(
    tmp_path: Path, method: str, mixes: bool
) -> None:
    """A2: the 4-head TREATMENT arm must actually mix.

    Written after a mutation survived: narrowing the mixing test to ``method == "dt_spatial"``
    left ``dt_spatial_h4`` training as a NO-MIXING model with a treatment arm's name, which would
    have made phase A a comparison of two controls and moved nothing but the head count.  Nothing
    else in the suite noticed, because the two arms are weight-compatible by design.
    """
    import torch

    setup = _tiny_joint_setup()
    ts.train_tier_dt(
        setup["stacked"], setup["index"], tier="mappo1000", method=method, seed=101,
        adjacency=setup["adjacency"], prompts=setup["prompts"], stats=setup["stats"],
        state_dim=setup["state_dim"], n_actions=setup["n_actions"], gradient_steps=2,
        batch_size=2, device=torch.device("cpu"), checkpoint_path=tmp_path / f"{method}.pt",
        protected=(), provenance={},
    )
    payload = torch.load(tmp_path / f"{method}.pt", map_location="cpu", weights_only=False)
    assert payload["config"]["spatial_mixing"] is mixes
    mask = np.asarray(payload["spatial_mask"], dtype=bool)
    off_diagonal = mask & ~np.eye(mask.shape[0], dtype=bool)
    assert bool(off_diagonal.any()) is mixes, (
        "a mixing arm must have an off-diagonal edge and a control must be the identity"
    )


def test_the_trainer_itself_refuses_a_regime_mismatch_not_only_the_helper(
    tmp_path: Path,
) -> None:
    """F6(a): the assertion must be ON THE TRAINER, not merely available beside it.

    Written after a mutation survived: deleting ``assert_process_regime`` from ``train_tier_dt``
    left the helper tested and unused, so a run could train in the default regime while recording
    ``deterministic: True`` in its provenance -- an unattributable cell that looks attributable.
    """
    import torch

    setup = _tiny_joint_setup()
    before = _tree(tmp_path)
    with pytest.raises(RuntimeError, match="deterministic algorithms"):
        ts.train_tier_dt(
            setup["stacked"], setup["index"], tier="mappo1000", method="dt_nomix", seed=101,
            adjacency=setup["adjacency"], prompts=setup["prompts"], stats=setup["stats"],
            state_dim=setup["state_dim"], n_actions=setup["n_actions"], gradient_steps=2,
            batch_size=2, device=torch.device("cpu"), checkpoint_path=tmp_path / "x.pt",
            protected=(), provenance={}, deterministic=True,
        )
    assert _tree(tmp_path) == before, "a refused run must create nothing"


def test_the_trainer_refuses_to_write_a_checkpoint_into_the_protected_root(
    tmp_path: Path,
) -> None:
    """D1 reaches the trainer: the guard is on the write path, not only on the report path."""
    import torch

    (tmp_path / "p5_1" / "checkpoints").mkdir(parents=True)
    protected = ts.protected_roots_from([tmp_path / "p5_1"])
    setup = _tiny_joint_setup()
    before = _tree(tmp_path)
    with pytest.raises(PermissionError, match="read-only"):
        ts.train_tier_dt(
            setup["stacked"], setup["index"], tier="mappo1000", method="dt_nomix", seed=101,
            adjacency=setup["adjacency"], prompts=setup["prompts"], stats=setup["stats"],
            state_dim=setup["state_dim"], n_actions=setup["n_actions"], gradient_steps=2,
            batch_size=2, device=torch.device("cpu"),
            checkpoint_path=tmp_path / "p5_1" / "checkpoints" / "x.pt",
            protected=protected, provenance={},
        )
    assert _tree(tmp_path) == before


# ----------------------------------------------------------------------
# F6 -- the numerical regime is a launch parameter that acts at process entry.
# ----------------------------------------------------------------------


def test_determinism_refuses_without_the_cublas_workspace_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F6(a): the variable must be exported BEFORE the process starts, so this is a refusal.

    A flag that silently fails to take effect is worse than no flag: it produces a run that
    believes it is reproducible and is not.
    """
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        ts.configure_determinism(True)


def test_determinism_refuses_a_workspace_value_that_is_not_one_of_the_accepted_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F6(a): ``:2:2`` is a value cuBLAS will not accept; a truthy check would pass it."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":2:2")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        ts.configure_determinism(True)


def test_the_default_regime_records_itself_without_touching_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F6(a)'s accept control: the default path must not enable anything globally."""
    import torch

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    record = ts.configure_determinism(False)
    assert record["deterministic"] is False
    assert torch.are_deterministic_algorithms_enabled() is False


def test_the_trainer_refuses_when_the_process_regime_is_not_the_one_requested() -> None:
    """F6(a): ``deterministic`` is an assertion about the process, never a mid-run toggle."""
    with pytest.raises(RuntimeError, match="deterministic algorithms"):
        ts.assert_process_regime(True)


# ----------------------------------------------------------------------
# B4 -- the size match.
# ----------------------------------------------------------------------


def _episodes(n_draws: int, per_draw: int) -> tuple[ts.EpisodeRef, ...]:
    out = []
    index = 0
    for draw in range(1, n_draws + 1):
        for copy in range(per_draw):
            out.append(
                ts.EpisodeRef(
                    dataset_dir="d",
                    episode_file=f"ep{index:06d}_seed{1000 + copy}_draw{draw}.npz",
                    episode_index=index,
                    flow_draw=draw,
                )
            )
            index += 1
    return tuple(out)


def test_a_none_subsample_tier_keeps_every_episode() -> None:
    """B4: only the ``random`` tier is subsampled; the others must pass through untouched."""
    episodes = _episodes(n_draws=200, per_draw=1)
    spec = ts.tier_spec("maxpressure")
    assert ts.selected_episodes(spec, episodes) == episodes


def test_the_size_match_is_asserted_on_EVERY_tier_not_only_the_subsampled_one() -> None:
    """B4: a partial ``maxpressure`` tier must be refused exactly as a partial ``random`` one is.

    Written after the first draft of this file let an un-subsampled tier through without a count
    check, which would have accepted a truncated corpus silently.
    """
    spec = ts.tier_spec("maxpressure")
    with pytest.raises(ValueError, match="200"):
        ts.selected_episodes(spec, _episodes(n_draws=5, per_draw=1))


def test_one_per_draw_keeps_exactly_one_episode_per_draw() -> None:
    """B4: 400 -> 200 on the ``random`` tier, one per draw, so tiers are size-matched."""
    episodes = _episodes(n_draws=200, per_draw=2)
    spec = ts.tier_spec("random")
    chosen = ts.selected_episodes(spec, episodes, rng=np.random.default_rng(1))
    assert len(chosen) == 200
    assert sorted(e.flow_draw for e in chosen) == list(range(1, 201))


def test_one_per_draw_is_deterministic_given_the_declared_seed() -> None:
    """B4: the selection is recorded in the artifact, so it must reproduce from the seed alone."""
    episodes = _episodes(n_draws=200, per_draw=2)
    spec = ts.tier_spec("random")
    first = ts.selected_episodes(
        spec, episodes, rng=np.random.default_rng(ts.RANDOM_SUBSAMPLE_RNG_SEED)
    )
    second = ts.selected_episodes(
        spec, episodes, rng=np.random.default_rng(ts.RANDOM_SUBSAMPLE_RNG_SEED)
    )
    assert [e.key for e in first] == [e.key for e in second]


def test_one_per_draw_does_not_depend_on_the_order_episodes_arrive_in() -> None:
    """B4: selection must depend on the RNG and the ids, never on filesystem order."""
    episodes = _episodes(n_draws=200, per_draw=2)
    spec = ts.tier_spec("random")
    forward = ts.selected_episodes(
        spec, episodes, rng=np.random.default_rng(ts.RANDOM_SUBSAMPLE_RNG_SEED)
    )
    shuffled = list(episodes)
    np.random.default_rng(99).shuffle(shuffled)
    reversed_order = ts.selected_episodes(
        spec, tuple(shuffled), rng=np.random.default_rng(ts.RANDOM_SUBSAMPLE_RNG_SEED)
    )
    assert [e.key for e in forward] == [e.key for e in reversed_order]


def test_a_different_seed_selects_a_different_set_so_the_seed_is_load_bearing() -> None:
    """B4's discriminating power: if any seed gave the same answer the record would be empty."""
    episodes = _episodes(n_draws=200, per_draw=2)
    spec = ts.tier_spec("random")
    declared = ts.selected_episodes(
        spec, episodes, rng=np.random.default_rng(ts.RANDOM_SUBSAMPLE_RNG_SEED)
    )
    other = ts.selected_episodes(spec, episodes, rng=np.random.default_rng(12345))
    assert [e.key for e in declared] != [e.key for e in other]


def test_a_tier_that_cannot_supply_its_declared_episode_count_is_refused() -> None:
    """B4: a partial tier may not train -- the size match is asserted, not hoped for."""
    episodes = _episodes(n_draws=3, per_draw=2)
    spec = ts.tier_spec("random")
    with pytest.raises(ValueError, match="200"):
        ts.selected_episodes(spec, episodes, rng=np.random.default_rng(1))


# ----------------------------------------------------------------------
# The new arm -- per-intersection %BC.
# ----------------------------------------------------------------------


def _streams(per_node: int, nodes: tuple[str, ...], offsets: dict[str, float]) -> list[Any]:
    from offline.offline_baselines import StreamReturn

    out = []
    for node in nodes:
        for i in range(per_node):
            out.append(
                StreamReturn(
                    dataset_dir="d",
                    episode_file=f"ep{i:06d}.npz",
                    ix_id=node,
                    ix_index=nodes.index(node),
                    episode_index=i,
                    flow_draw=i + 1,
                    group=(4, 3),
                    total_return=offsets[node] - i,
                )
            )
    return out


def test_the_per_intersection_filter_keeps_the_same_decile_from_every_node() -> None:
    """R2/R6's scope condition: the per-node filter must not be a load sorter.

    The global filter on this fixture keeps 20 streams from ONE node, because ``A``'s returns
    dominate ``B``'s.  The per-intersection filter must keep 10 from each.
    """
    nodes = ("A", "B")
    streams = _streams(per_node=100, nodes=nodes, offsets={"A": 0.0, "B": -10_000.0})
    kept = ts.per_intersection_top_streams(streams, fraction=0.10)
    counts = {node: sum(1 for s in kept if s.ix_id == node) for node in nodes}
    assert counts == {"A": 10, "B": 10}


def test_the_per_intersection_filter_keeps_the_same_TOTAL_as_the_global_one() -> None:
    """The two filter arms differ in WHICH streams they keep, not how many (``BRIEF_27`` C).

    Equal totals are what makes the comparison controlled; a different training-set size would
    confound the filter with the amount of data.
    """
    from offline.offline_baselines import TOP_RETURN_FRACTION

    nodes = ("A", "B", "C", "D")
    streams = _streams(
        per_node=50, nodes=nodes, offsets={"A": 0.0, "B": -100.0, "C": -200.0, "D": -300.0}
    )
    kept = ts.per_intersection_top_streams(streams, fraction=TOP_RETURN_FRACTION)
    expected_global = math.ceil(TOP_RETURN_FRACTION * len(streams))
    assert len(kept) == expected_global


def test_the_per_intersection_filter_takes_the_BEST_of_each_node() -> None:
    """It is a top-decile filter within a node, not an arbitrary decile."""
    nodes = ("A",)
    streams = _streams(per_node=100, nodes=nodes, offsets={"A": 0.0})
    kept = ts.per_intersection_top_streams(streams, fraction=0.10)
    assert {s.total_return for s in kept} == {-float(i) for i in range(10)}


def test_the_per_intersection_filter_is_deterministic_under_a_shuffle() -> None:
    """Ties are broken by ``(dataset_dir, episode_file, ix_id)``, never by load order."""
    nodes = ("A", "B")
    streams = _streams(per_node=40, nodes=nodes, offsets={"A": 0.0, "B": 0.0})
    for stream in streams:
        object.__setattr__(stream, "total_return", -1.0)
    shuffled = list(streams)
    np.random.default_rng(3).shuffle(shuffled)
    first = [s.key for s in ts.per_intersection_top_streams(streams, fraction=0.10)]
    second = [s.key for s in ts.per_intersection_top_streams(shuffled, fraction=0.10)]
    assert first == second


def test_the_per_intersection_filter_refuses_a_fraction_outside_the_unit_interval() -> None:
    """Mirrors ``top_return_streams``' own refusal rather than inventing a second convention."""
    streams = _streams(per_node=10, nodes=("A",), offsets={"A": 0.0})
    with pytest.raises(ValueError, match="fraction"):
        ts.per_intersection_top_streams(streams, fraction=0.0)


# ----------------------------------------------------------------------
# B3 -- the reuse gate.
# ----------------------------------------------------------------------


def test_sha256_of_matches_hashlib_on_the_same_bytes(tmp_path: Path) -> None:
    """B3(a): the digest is recomputed by an independent route in the test."""
    target = tmp_path / "x.json"
    payload = b'{"cell": 1}\n'
    target.write_bytes(payload)
    assert ts.sha256_of(target) == hashlib.sha256(payload).hexdigest()


def test_parse_sha256sums_reads_the_manifest_format_the_repo_actually_uses() -> None:
    """B3(a): parsed against the real two-space format ``sha256sum`` writes."""
    text = "aa11  p5_1/eval_bc.json\nbb22  p5_1/checkpoints/x.pt\n"
    assert ts.parse_sha256sums(text) == {
        "p5_1/eval_bc.json": "aa11",
        "p5_1/checkpoints/x.pt": "bb22",
    }


def test_a_reused_cell_whose_digest_changed_is_refused_at_consumption(tmp_path: Path) -> None:
    """B3(a): a digest checked once is not a digest checked when used."""
    target = tmp_path / "eval_bc.json"
    target.write_bytes(b"original")
    good = hashlib.sha256(b"original").hexdigest()
    assert ts.assert_reused_digest(target, good) == good
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        ts.assert_reused_digest(target, good)


def test_identical_cells_compare_equal_and_report_what_they_compared() -> None:
    """B3(b)'s accept control, and it pins the non-empty count in the returned record."""
    payload = _cell_payload(seeds=(101, 202), draws=(1000, 1001))
    report = ts.assert_cells_identical(payload, json.loads(json.dumps(payload)), expected_n=4)
    assert report["n_compared"] == 4


def test_the_equality_check_refuses_a_set_that_is_not_exactly_the_expected_size() -> None:
    """B3(b): ``found no differences`` may never be ``compared nothing``.

    Two EMPTY cells are trivially equal; without this assertion the gate passes on them.
    """
    empty = {"method": "random", "declared_gradient_steps": 40_000, "episodes": []}
    with pytest.raises(ValueError, match="exactly 500"):
        ts.assert_cells_identical(empty, empty, expected_n=500)


def test_the_equality_check_fires_on_a_single_perturbed_episode() -> None:
    """B3(b)'s POSITIVE CONTROL: perturb one value of 500 and the gate must refuse."""
    left = _cell_payload(seeds=(101,), draws=tuple(range(1000, 1500)))
    right = json.loads(json.dumps(left))
    right["episodes"][283]["att_horizon"] += 1e-9
    with pytest.raises(ValueError, match="differ"):
        ts.assert_cells_identical(left, right, expected_n=500)


def test_the_equality_check_fires_when_an_episode_is_missing_rather_than_wrong() -> None:
    """B3(b): a dropped episode is a different failure from a changed one and must also refuse."""
    left = _cell_payload(seeds=(101,), draws=tuple(range(1000, 1500)))
    right = json.loads(json.dumps(left))
    right["episodes"].pop()
    with pytest.raises(ValueError, match="exactly 500"):
        ts.assert_cells_identical(left, right, expected_n=500)


def test_episode_key_set_is_the_seed_draw_pairing_and_not_a_count() -> None:
    """B3(b): the comparison is keyed, so a reordered payload is still equal."""
    payload = _cell_payload(seeds=(101, 202), draws=(1000, 1001))
    assert ts.episode_key_set(payload) == frozenset(
        {(101, 1000), (101, 1001), (202, 1000), (202, 1001)}
    )


def test_a_reordered_cell_is_still_identical() -> None:
    """B3(b): episode order is not a property of the measurement and must not fail the gate."""
    left = _cell_payload(seeds=(101, 202), draws=(1000, 1001))
    right = json.loads(json.dumps(left))
    right["episodes"].reverse()
    report = ts.assert_cells_identical(left, right, expected_n=4)
    assert report["n_compared"] == 4


# ----------------------------------------------------------------------
# The registered scorers.
# ----------------------------------------------------------------------


def test_score_level_counts_a_cell_inside_the_band_as_a_hit() -> None:
    """Q1: the per-cell rule is a RELATIVE error against the registered band."""
    measured = {cell: ts.PREDICTED_LEVELS[cell[0]][cell[1]] for cell in ts.OUT_OF_SAMPLE_CELLS}
    report = ts.score_level(measured)
    assert report["n_held"] == len(ts.OUT_OF_SAMPLE_CELLS)
    assert report["outcome"] == "HELD"


def test_score_level_fails_the_aggregate_when_too_few_cells_hold() -> None:
    """Q1: the threshold is load-bearing, so a majority of misses must FAIL it."""
    measured = {
        cell: ts.PREDICTED_LEVELS[cell[0]][cell[1]] * 10.0 for cell in ts.OUT_OF_SAMPLE_CELLS
    }
    report = ts.score_level(measured)
    assert report["n_held"] == 0
    assert report["outcome"] == "FAILED"


def test_score_level_scores_exactly_the_registered_cells_and_ignores_seen_ones() -> None:
    """B5.2: a seen cell may never enter the denominator, even if it is supplied."""
    measured = {cell: ts.PREDICTED_LEVELS[cell[0]][cell[1]] for cell in ts.OUT_OF_SAMPLE_CELLS}
    measured[("dt_nomix", "mappo1000")] = 1e6
    report = ts.score_level(measured)
    assert report["n_cells"] == 19
    assert ("dt_nomix", "mappo1000") not in {tuple(c["cell"]) for c in report["cells"]}


def test_the_band_edge_is_inclusive_and_just_outside_it_is_a_miss() -> None:
    """Q1: the band is ``<= 0.30``; a cell at 30.0000001 % must not be counted as a hit.

    Injected predicted values rather than the registered table, because ``p * 1.30`` is NOT a
    relative error of exactly 0.30 in binary floating point: measured on four of the five
    registered levels it lands at 0.30000000000000004, so the same test written against the real
    table would have been failing on a float artefact rather than on the rule. ``100 -> 130`` is
    exact, so the boundary is tested as the rule states it.
    """
    predicted = {"x": {"t": 100.0}}
    cells = (("x", "t"),)
    on_edge = ts.score_level(
        {("x", "t"): 130.0}, predicted=predicted, cells=cells, threshold=1
    )
    just_outside = ts.score_level(
        {("x", "t"): 130.0000001}, predicted=predicted, cells=cells, threshold=1
    )
    assert (on_edge["n_held"], just_outside["n_held"]) == (1, 0)


def test_the_band_is_symmetric_so_an_arm_that_beats_its_prediction_can_also_miss() -> None:
    """Q1: the rule is |measured - predicted| / predicted, so a large UNDERSHOOT is also a miss.

    Registered that way deliberately: a rule that only punished overshoot would score an arm that
    collapsed upward as a failure and one that improved beyond prediction as a success.
    """
    predicted = {"x": {"t": 100.0}}
    cells = (("x", "t"),)
    assert ts.score_level(
        {("x", "t"): 69.0}, predicted=predicted, cells=cells, threshold=1
    )["n_held"] == 0
    assert ts.score_level(
        {("x", "t"): 71.0}, predicted=predicted, cells=cells, threshold=1
    )["n_held"] == 1


def test_predicted_order_sorts_the_arms_by_predicted_att() -> None:
    """Q2: the predicted ordering is derived from the registered table, never retyped."""
    assert ts.predicted_order("random") == (
        "dt_nomix", "bc", "bc_top10_perix", "dt_spatial", "iql", "bc_top10",
    )


def test_predicted_order_puts_the_new_arm_first_on_maxpressure() -> None:
    """Q2a: registered as a coin flip -- 0.43 ATT -- and it must be scored as the rule says."""
    assert ts.predicted_order("maxpressure")[0] == "bc_top10_perix"


def test_concordance_counts_all_fifteen_pairs_and_the_hard_subset_counts_six() -> None:
    """Q2b and C4: the subset is 6 of 15, and the two counts are reported side by side."""
    order = ts.predicted_order("random")
    full = ts.concordance(order, order)
    hard = ts.concordance(order, order, subset=ts.HARD_SUBSET)
    assert (full["n_pairs"], hard["n_pairs"]) == (15, 6)
    assert (full["n_concordant"], hard["n_concordant"]) == (15, 6)


def test_concordance_can_be_high_overall_while_the_hard_subset_collapses() -> None:
    """C4's whole argument, executed: the two easy arms carry 9 pairs.

    Reversing the four hard arms among themselves leaves 9 of 15 correct -- and 0 of 6 -- which is
    exactly the case a 15-pair count alone would report as a near miss rather than a collapse.
    """
    predicted = ts.predicted_order("random")
    hard_reversed = [a for a in predicted if a in ts.HARD_SUBSET][::-1]
    measured: list[str] = []
    iterator = iter(hard_reversed)
    for arm in predicted:
        measured.append(next(iterator) if arm in ts.HARD_SUBSET else arm)
    full = ts.concordance(predicted, measured)
    hard = ts.concordance(predicted, measured, subset=ts.HARD_SUBSET)
    assert full["n_concordant"] == 9
    assert hard["n_concordant"] == 0


def test_the_stop_rule_fires_only_when_the_interval_lies_entirely_below_zero() -> None:
    """Q0: a straddling CI is NOT a reversal, and the plan says phase B proceeds."""
    assert ts.stop_rule_verdict(-8.0, -2.0) == "STOP"
    assert ts.stop_rule_verdict(-8.0, +2.0) == "CONTINUE"
    assert ts.stop_rule_verdict(+2.0, +8.0) == "CONTINUE"


def test_the_stop_rule_does_not_fire_on_an_interval_touching_zero() -> None:
    """Q0: ``entirely below`` is strict; a CI whose upper end is 0 has not resolved."""
    assert ts.stop_rule_verdict(-8.0, 0.0) == "CONTINUE"


# ----------------------------------------------------------------------
# E1 / F7 -- the paired replicate report.
# ----------------------------------------------------------------------


def _seeded_cell(seed: int, values: dict[int, float]) -> dict[str, Any]:
    return {
        "method": "dt_spatial",
        "declared_gradient_steps": 40_000,
        "episodes": [
            {"seed": seed, "draw_id": d, "att_horizon": v, "horizon_vehicle_count": 1.0,
             "episode_reward": -1.0, "arm": "dt_spatial@t"}
            for d, v in values.items()
        ],
    }


def test_an_identical_replicate_reports_zero_and_does_not_exclude_zero() -> None:
    """F7's accept control: a perfect replicate must not be reported as a difference."""
    values = {1000 + i: 100.0 + i for i in range(100)}
    report = ts.paired_replicate_report(
        _seeded_cell(202, values), _seeded_cell(202, dict(values)), seed=202
    )
    assert report["n_shared_draws"] == 100
    assert report["mean_difference"] == 0.0
    assert report["excludes_zero"] is False


def test_a_constant_offset_replicate_excludes_zero() -> None:
    """F7: a systematic shift with no scatter must be reported as excluding zero.

    This is the case whose meaning the amendment spells out -- two runs of the same code at the
    same seed producing measurably different policies.
    """
    values = {1000 + i: 100.0 + i for i in range(100)}
    shifted = {d: v + 5.0 for d, v in values.items()}
    report = ts.paired_replicate_report(
        _seeded_cell(202, shifted), _seeded_cell(202, values), seed=202
    )
    assert report["mean_difference"] == pytest.approx(5.0)
    assert report["excludes_zero"] is True


def test_the_SAME_mean_difference_excludes_zero_or_not_depending_on_the_scatter() -> None:
    """F7's whole point, tested exactly: two point estimates cannot be judged.

    Both cases have a mean difference of EXACTLY +5.0.  One has no scatter and excludes zero; the
    other alternates +105 / -95 and does not.  That is why E1 reports an interval rather than
    ``72.07 against 68.5``.

    Deterministic by construction rather than by a random draw: an earlier version of this test
    sampled ``normal(0, 30)`` and assumed the realised mean would be small, which is a property of
    the draw and not of the rule -- it failed for that reason and is replaced.
    """
    values = {1000 + i: 100.0 + i for i in range(100)}
    tight = {d: v + 5.0 for d, v in values.items()}
    wide = {d: v + (105.0 if i % 2 == 0 else -95.0) for i, (d, v) in enumerate(values.items())}

    tight_report = ts.paired_replicate_report(
        _seeded_cell(202, tight), _seeded_cell(202, values), seed=202
    )
    wide_report = ts.paired_replicate_report(
        _seeded_cell(202, wide), _seeded_cell(202, values), seed=202
    )
    assert tight_report["mean_difference"] == pytest.approx(5.0)
    assert wide_report["mean_difference"] == pytest.approx(5.0)
    assert tight_report["excludes_zero"] is True
    assert wide_report["excludes_zero"] is False
    assert wide_report["ci95_width"] > tight_report["ci95_width"]


def test_the_paired_report_refuses_a_partial_or_unshared_draw_set() -> None:
    """A5: a comparison that cannot be made over shared draws is void, not approximated."""
    left = _seeded_cell(202, {1000 + i: 100.0 for i in range(100)})
    right = _seeded_cell(202, {2000 + i: 100.0 for i in range(100)})
    with pytest.raises(ValueError, match="share no draw ids"):
        ts.paired_replicate_report(left, right, seed=202)
    short = _seeded_cell(202, {1000 + i: 100.0 for i in range(50)})
    with pytest.raises(ValueError, match="different draw sets"):
        ts.paired_replicate_report(short, _seeded_cell(202, {1000 + i: 100.0 for i in range(100)}),
                                   seed=202)


def test_the_paired_report_selects_only_the_requested_seed() -> None:
    """E1 replicates ONE seed; pooling another seed's episodes into it would be a different test."""
    mixed = _seeded_cell(202, {1000 + i: 100.0 for i in range(100)})
    mixed["episodes"].extend(
        {"seed": 303, "draw_id": 1000 + i, "att_horizon": 999.0,
         "horizon_vehicle_count": 1.0, "episode_reward": -1.0, "arm": "x"}
        for i in range(100)
    )
    report = ts.paired_replicate_report(
        mixed, _seeded_cell(202, {1000 + i: 100.0 for i in range(100)}), seed=202
    )
    assert report["n_shared_draws"] == 100
    assert report["mean_difference"] == 0.0


def test_the_published_seed_202_cells_are_present_and_pairable() -> None:
    """F7's inputs, checked against the secured artifact rather than taken on report.

    Skips rather than failing when the secured tree is absent, because it is outside this worktree.
    """
    src = Path("/home/filip/rltraffic/output/p5_1")
    if not src.is_dir():
        pytest.skip("the secured output/p5_1 tree is not present on this machine")
    for arm in ("dt_spatial", "dt_nomix"):
        payload = json.loads((src / f"eval_{arm}.json").read_text(encoding="utf-8"))
        by_draw = ts.episodes_of_seed(payload, 202)
        assert len(by_draw) == 100
        assert sorted(by_draw) == list(range(1000, 1100))


# ----------------------------------------------------------------------
# B3(a) -- the reuse gate as the campaign runs it.
# ----------------------------------------------------------------------


def _fake_reuse_tree(root: Path, arms: tuple[str, ...]) -> Path:
    (root / "p5_1").mkdir()
    lines = []
    for arm in arms:
        path = root / "p5_1" / f"eval_{arm}.json"
        path.write_text(json.dumps({"arm": arm, "episodes": []}), encoding="utf-8")
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  p5_1/eval_{arm}.json")
    manifest = root / "SHA256SUMS.txt"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def test_the_reuse_gate_verifies_every_declared_cell(tmp_path: Path) -> None:
    """B3(a)'s accept control: the gate must pass on an untouched tree and say what it checked."""
    arms = ("dt_spatial", "dt_nomix")
    manifest = _fake_reuse_tree(tmp_path, arms)
    record = ts.verify_reuse_gate(tmp_path / "p5_1", manifest, arms=arms)
    assert set(record) == set(arms)
    assert all(len(entry["sha256"]) == 64 for entry in record.values())


def test_the_reuse_gate_refuses_a_cell_whose_bytes_changed_since_securing(
    tmp_path: Path,
) -> None:
    """B3(a): the check is AT CONSUMPTION, so a file edited after securing must be refused."""
    arms = ("dt_spatial",)
    manifest = _fake_reuse_tree(tmp_path, arms)
    (tmp_path / "p5_1" / "eval_dt_spatial.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        ts.verify_reuse_gate(tmp_path / "p5_1", manifest, arms=arms)


def test_the_reuse_gate_refuses_a_cell_that_has_no_recorded_digest(tmp_path: Path) -> None:
    """B3(a): an unrecorded file cannot be shown to be the reviewed artifact."""
    manifest = _fake_reuse_tree(tmp_path, ("dt_spatial",))
    (tmp_path / "p5_1" / "eval_iql.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not in"):
        ts.verify_reuse_gate(tmp_path / "p5_1", manifest, arms=("dt_spatial", "iql"))


def test_the_reuse_gate_covers_all_seven_reused_cells_by_default() -> None:
    """A3: the seven cells the plan lists are the seven the gate checks -- no silent subset."""
    assert len(ts.REUSED_CELLS) == 7
    assert set(ts.REUSED_CELLS) == {
        "dt_spatial", "dt_nomix", "bc", "bc_top10", "iql", "behaviour", "random"
    }


# ----------------------------------------------------------------------
# I3 -- Q1's threshold is carried across by a rule, not re-decided.
# ----------------------------------------------------------------------


def test_the_threshold_rule_reproduces_the_originally_registered_threshold() -> None:
    """I3's self-check, and it is why the rule is a carry-across rather than a new criterion.

    If ``ceil(9/13 * N)`` did not return 9 at N = 13 it would be a fresh judgement wearing the old
    threshold's clothes.
    """
    assert ts.level_threshold_for(13) == 9


def test_the_threshold_at_the_current_denominator_is_fourteen_of_nineteen() -> None:
    """I3: adding fixedtime grows N from 13 to 19, and the rule -- not a person -- sets k."""
    assert len(ts.OUT_OF_SAMPLE_CELLS) == 19
    assert ts.level_threshold_for(19) == 14
    assert ts.LEVEL_THRESHOLD == 14


def test_thirteen_of_nineteen_is_below_the_registered_rate_and_is_excluded() -> None:
    """I3: the arithmetic that decides 14 rather than 13, asserted rather than asserted about."""
    registered = 9 / 13
    assert 13 / 19 < registered
    assert 14 / 19 >= registered


def test_the_threshold_rule_is_monotone_and_never_loosens_the_rate() -> None:
    """I3: whatever N becomes, the rule may not return a proportion below the registered rate."""
    for n in range(1, 60):
        assert ts.level_threshold_for(n) / n >= 9 / 13 - 1e-12


# ----------------------------------------------------------------------
# J2 -- the replicate must be proved independent before its envelope means anything.
# ----------------------------------------------------------------------


def _tiny_checkpoint(path: Path, scale: float) -> None:
    import torch

    torch.save(
        {
            "model": {"b.weight": torch.tensor([[1.0, 2.0]]) * scale,
                      "a.bias": torch.tensor([3.0, 4.0])},
            "provenance": {"tier": "mappo1000"},
        },
        path,
    )


def test_two_different_checkpoints_have_different_canonical_digests(tmp_path: Path) -> None:
    """J2(c)'s accept control."""
    _tiny_checkpoint(tmp_path / "a.pt", 1.0)
    _tiny_checkpoint(tmp_path / "b.pt", 1.0000001)
    record = ts.assert_independent_replicate(tmp_path / "a.pt", tmp_path / "b.pt")
    assert record["replicate_state_dict_sha256"] != record["original_state_dict_sha256"]


def test_comparing_a_checkpoint_with_itself_is_REFUSED_not_reported_as_zero(
    tmp_path: Path,
) -> None:
    """J2(c): a re-evaluation of the same checkpoint returns zero BY CONSTRUCTION.

    That is the failure mode the assertion exists for: without it, the machinery would report the
    answer the question was asked to avoid.
    """
    _tiny_checkpoint(tmp_path / "a.pt", 1.0)
    with pytest.raises(ValueError, match="canonical state_dict digest"):
        ts.assert_independent_replicate(tmp_path / "a.pt", tmp_path / "a.pt")


def test_the_canonical_digest_ignores_provenance_and_tracks_only_weights(
    tmp_path: Path,
) -> None:
    """J2(c): the FILE hash differs on provenance alone (DEFERRED 29); this one must not.

    E1 met this live -- two runs' file hashes differed on git_commit, deterministic, n_head and a
    changed tier label while saying nothing about the weights.
    """
    import torch

    weights = {"a.bias": torch.tensor([3.0, 4.0]), "b.weight": torch.tensor([[1.0, 2.0]])}
    torch.save({"model": weights, "provenance": {"tier": "mappo1000", "n_head": 1}},
               tmp_path / "one.pt")
    torch.save({"model": weights, "provenance": {"tier": "grid4x4_mappo1000", "n_head": 4}},
               tmp_path / "two.pt")
    assert ts.sha256_of(tmp_path / "one.pt") != ts.sha256_of(tmp_path / "two.pt")
    assert ts.canonical_state_dict_digest(tmp_path / "one.pt") == ts.canonical_state_dict_digest(
        tmp_path / "two.pt"
    )


def test_the_canonical_digest_does_not_depend_on_key_insertion_order(tmp_path: Path) -> None:
    """J2(c): sorted-key ordering, so two dicts with the same weights agree whatever their order."""
    import torch

    torch.save({"model": {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}}, tmp_path / "x.pt")
    torch.save({"model": {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}}, tmp_path / "y.pt")
    assert ts.canonical_state_dict_digest(tmp_path / "x.pt") == ts.canonical_state_dict_digest(
        tmp_path / "y.pt"
    )


# ----------------------------------------------------------------------
# G2 -- E1 must run in P5.1's exact environment.
# ----------------------------------------------------------------------

CAMPAIGN_PATH = REPO_ROOT / "offline" / "campaigns" / "p5_2.sh"


def test_the_campaign_never_exports_the_cublas_variable_unconditionally() -> None:
    """G2: the variable is part of the determinism recipe, not a neutral setting.

    An earlier version of the script exported it at top level, commented ``harmless in the default
    regime``.  E1 measures the run-to-run envelope OF THE REGIME P5.1 RAN IN, and P5.1 exported
    OMP and MKL only -- so a replicate carrying it would conflate the noise being measured with a
    systematic effect of the cuBLAS configuration.  This test is why that cannot come back.
    """
    lines = CAMPAIGN_PATH.read_text(encoding="utf-8").splitlines()
    top_level_exports = [
        line for line in lines
        if line.startswith("export CUBLAS_WORKSPACE_CONFIG")
    ]
    assert top_level_exports == [], (
        f"the campaign exports CUBLAS_WORKSPACE_CONFIG unconditionally: {top_level_exports}"
    )


def test_the_campaign_actively_unsets_the_variable_in_the_default_regime() -> None:
    """G2: not setting it is not enough -- the launcher's own shell may already carry it."""
    text = CAMPAIGN_PATH.read_text(encoding="utf-8")
    assert "unset CUBLAS_WORKSPACE_CONFIG" in text, (
        "the default branch must UNSET the variable, because inheriting it from the launching "
        "shell is how E1 would silently run in a regime P5.1 never ran in"
    )


def test_the_campaign_sets_the_variable_only_inside_the_deterministic_branch() -> None:
    """G2: it is still required for the deterministic regime (F6a); only the default one drops it."""
    lines = CAMPAIGN_PATH.read_text(encoding="utf-8").splitlines()
    setters = [
        (i, line) for i, line in enumerate(lines)
        if "export CUBLAS_WORKSPACE_CONFIG=" in line and not line.lstrip().startswith("#")
    ]
    assert len(setters) == 1, f"expected exactly one setter, found {setters}"
    index, line = setters[0]
    assert line.startswith("  "), "the setter must be indented inside the deterministic branch"
    preceding = "\n".join(lines[max(0, index - 4):index])
    assert "DETERMINISTIC" in preceding, (
        f"the setter is not guarded by the deterministic branch; preceding lines:\n{preceding}"
    )


def test_the_launch_header_does_not_instruct_exporting_the_variable() -> None:
    """G2: the launch block is copied and pasted, so an instruction there is as binding as code.

    Matches a commented shell ASSIGNMENT (``#   export CUBLAS_WORKSPACE_CONFIG=...``) rather than
    the bare name: the header also carries a sentence FORBIDDING the export, and an earlier version
    of this test flagged that sentence as the very thing it prohibits.  Requiring the ``=`` is what
    separates an instruction from a prohibition.
    """
    import re

    header = CAMPAIGN_PATH.read_text(encoding="utf-8").splitlines()[:40]
    instructions = [
        line for line in header
        if re.match(r"^#\s+export\s+CUBLAS_WORKSPACE_CONFIG\s*=", line)
    ]
    assert instructions == [], f"the launch block instructs the export: {instructions}"


# ----------------------------------------------------------------------
# The declarations, checked against the plan and against the corpus.
# ----------------------------------------------------------------------


def test_the_registered_constants_appear_in_the_committed_plan() -> None:
    """The registration is the PLAN; this module must not drift from it.

    P5.1's review found a ``declared constants`` test that never read the plan.  This one reads
    the committed file and fails if a registered number is not in it.
    """
    plan = PLAN_PATH.read_text(encoding="utf-8")
    assert "k = ceil(9/13 · N)" in plan, "the plan must state the threshold RULE, not a bare number"
    assert f"k = {ts.LEVEL_THRESHOLD}" in plan, "the plan must state the threshold at today's N"
    assert "0.30" in plan or "±30 %" in plan
    for arm, tier in ts.OUT_OF_SAMPLE_CELLS:
        value = ts.PREDICTED_LEVELS[arm][tier]
        assert f"{value:.4f}" in plan, f"{arm}@{tier} = {value:.4f} is not in the plan"


def test_the_out_of_sample_set_excludes_every_reused_cell() -> None:
    """B5.2, asserted structurally rather than by reading the list: no seen cell may be scored."""
    reused = {(arm, ts.REUSED_TIER) for arm in ts.REUSED_CELLS}
    assert not (set(ts.OUT_OF_SAMPLE_CELLS) & reused)


def test_the_out_of_sample_set_is_exactly_nineteen_cells() -> None:
    """B5.2: the denominator is enumerated, so a silent addition changes a registered score.

    It was 13 until ``fixedtime`` joined under H1 and grew it to 19.  The threshold moved WITH it,
    by the stated rule rather than by a judgement -- see the I3 tests above.
    """
    assert len(ts.OUT_OF_SAMPLE_CELLS) == 19
    assert len(set(ts.OUT_OF_SAMPLE_CELLS)) == 19
    per_tier = {t: sum(1 for _, tier in ts.OUT_OF_SAMPLE_CELLS if tier == t)
                for _, t in ts.OUT_OF_SAMPLE_CELLS}
    assert per_tier == {"maxpressure": 6, "fixedtime": 6, "random": 6, "mappo1000": 1}


def test_the_tier_order_is_grid4x4s_own_measured_att_order() -> None:
    """B0/B1: no hz1x1 number may order a grid4x4 figure."""
    values = [ts.TIERS[tier].ladder_att for tier in ts.TIER_ORDER]
    assert values == sorted(values)


def test_the_declared_ladder_att_matches_the_committed_ladder_artifact() -> None:
    """B0: the tier table is checked against ``att_ladder_v11.json``, not against memory."""
    ladder = json.loads(
        (REPO_ROOT / "docs" / "data" / "att_ladder_v11.json").read_text(encoding="utf-8")
    )
    measured = {
        cell["tier"]: cell["att_horizon_mean"]
        for cell in ladder["cells"]
        if cell["scenario"] == "cf_grid4x4"
    }
    for tier, spec in ts.TIERS.items():
        assert spec.ladder_att == measured[tier], tier


def test_the_head_map_covers_every_dt_arm_and_divides_the_model_width() -> None:
    """A2: ``d_model = 128`` must be divisible by every declared head count."""
    assert set(ts.N_HEAD_BY_METHOD) == set(ts.DT_METHODS)
    for method, heads in ts.N_HEAD_BY_METHOD.items():
        assert 128 % heads == 0, method


def test_the_new_arm_is_a_method_and_the_head_arms_are_not() -> None:
    """A3: the head cells are a 2x2 at one tier, not rungs of the ladder's method set."""
    assert "bc_top10_perix" in ts.METHODS
    assert not set(ts.HEAD_METHODS) & set(ts.METHODS)


def test_tier_spec_refuses_an_undeclared_tier() -> None:
    """A tier that is not declared may not enter the sweep by being named."""
    with pytest.raises(ValueError, match="unknown tier"):
        ts.tier_spec("mappo060")


def test_fixedtime_joined_the_running_order_without_replacing_maxpressure() -> None:
    """B1/F4/H1: the pre-declared contingency fired; it was ADDED beside maxpressure, not swapped.

    This test asserted the opposite until H1, when E1's zero envelope removed regime (c)'s case and
    F4's pre-declared reallocation sent the freed budget here.  The load-bearing half is unchanged
    and is what this now pins: maxpressure is still in the set.
    """
    assert "fixedtime" in ts.TIERS
    assert "fixedtime" in ts.TIER_ORDER
    assert "maxpressure" in ts.TIER_ORDER
    assert len(ts.TIER_ORDER) == 4
