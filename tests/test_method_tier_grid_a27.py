"""A27 in ``offline/method_tier_grid.py`` (``BRIEF_46`` 3.1, 3.2, 4.2, 4.4): the hz1x1 tiers A27 adds, the training
flags, the arm names, and the registered behaviour under the defaults.

No format of its own: the fixture runs write P4.6's ``p4.6-grid-training/1.0`` records and ``iql-``/``bc-checkpoint/1.0``
checkpoints into ``tmp_path``.  Alignment convention: contract C6's, unchanged (a stream's return is ``sum(local_reward)``;
the fixture corpus is ``tests/test_method_tier_grid.py``'s).

WHY A27's TIERS ARE NOT IN ``TIERS``
-----------------------------------
``tests/test_rtg_ablation.py`` pins ``set(PROBE_TIERS) == set(method_tier_grid.TIERS)``, so a new key there fails a
registered test.  A27's two hz1x1 tiers therefore live in ``A27_TIERS`` beside it; the lookups consult both, and the
registered mapping -- and everything that iterates over it -- is byte-for-byte what it was.

Corpus tests skip with a reason naming ``RLTRAFFIC_CORPUS_V11``; the CUDA regeneration names ``RLTRAFFIC_OUTPUT_ROOT``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import offline.method_tier_grid as grid
from offline.dataset import TrajectoryWindowDataset
from tests.test_method_tier_grid import CONTEXT, fixture_spec, stream, write_tier_dir

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The registered hz1x1 tiers, written out: A27 must leave this set exactly as it was.
REGISTERED_TIERS = {"mappo1000", "mappo500", "maxpressure", "fixedtime", "random", "mix33", "mix50", "mix67"}

#: A27(e)'s design facts, as the registration states them (kept decile by source directory and behaviour seed).
A27E_KEPT = {
    "cf_hz1x1__mappo1000__seed101": 5,
    "cf_hz1x1__mappo1000__seed202": 6,
    "cf_hz1x1__mappo1000__seed404": 1,
    "cf_hz1x1__mappo1000__seed505": 2,
    "cf_hz1x1__mappo500__seed101": 3,
    "cf_hz1x1__mappo500__seed202": 1,
    "cf_hz1x1__mappo500__seed404": 2,
}
A27E_DECILE_RETURNS = (-6627.0, -5959.0)


# ----------------------------------------------------------------------
# The registries
# ----------------------------------------------------------------------


def test_the_registered_tiers_mapping_is_unchanged_and_a27s_tiers_sit_beside_it() -> None:
    assert set(grid.TIERS) == REGISTERED_TIERS
    assert set(grid.A27_TIERS) == {"random_fresh", "mix_m1000_m500"}
    assert not set(grid.A27_TIERS) & set(grid.TIERS)
    assert dict(grid.MIXTURE_EXPERT_FRACTION) == {"mix33": 0.33, "mix50": 0.50, "mix67": 0.67}
    assert dict(grid.A27_MIXTURE_EXPERT_FRACTION) == {"mix_m1000_m500": 0.50}


def test_the_overlap_mixture_is_declared_as_a27e_says() -> None:
    spec = grid.A27_TIERS["mix_m1000_m500"]
    assert spec.dirs == (*(f"cf_hz1x1__mappo1000__seed{s}" for s in (101, 202, 303, 404, 505)),
                         *(f"cf_hz1x1__mappo500__seed{s}" for s in (101, 202, 303, 404, 505)))
    assert spec.components == ("mappo1000", "mappo500")
    assert (spec.subsample, spec.stream_count, spec.phase) == ("mixture", 200, 2)
    assert (spec.target_rtg, spec.rtg_scale) == (-5959.0, 11043.0)


def test_random_fresh_is_declared_with_its_own_corpus_and_no_typed_prompt_yet() -> None:
    """Q2 (``docs/plans/p9.1.md``): the prompt is the declared RULE over a corpus collected at G3; until its values are
    typed the spec carries ``nan`` and nothing can train on it."""
    spec = grid.A27_TIERS["random_fresh"]
    assert spec.dirs == ("cf_hz1x1__random_fresh",)
    assert (spec.subsample, spec.stream_count, spec.phase, spec.components) == ("none", 200, 1, ())
    assert math.isnan(spec.target_rtg) and math.isnan(spec.rtg_scale)
    with pytest.raises(ValueError, match="declares target_rtg nan"):
        grid.assert_declaration_matches_corpus(spec, [stream(1, -10.0), stream(2, -20.0)])


def test_tier_spec_and_arm_key_resolve_the_a27_tiers_and_still_refuse_an_undeclared_one() -> None:
    assert grid.tier_spec("mix_m1000_m500") is grid.A27_TIERS["mix_m1000_m500"]
    assert grid.tier_spec("mix50") is grid.TIERS["mix50"]
    with pytest.raises(ValueError, match="unknown tier"):
        grid.tier_spec("mappo200")
    assert grid.arm_key("iql_t", "random_fresh") == "iql_t@random_fresh"
    assert grid.arm_key("dt_100k", "mix50") == "dt_100k@mix50"
    with pytest.raises(ValueError, match="unknown method"):
        grid.arm_key("bc_m", "mappo1000")  # a GRID arm name is not an hz1x1 method


def test_the_overlap_mixture_uses_mix50s_rule_rng_and_seed_with_mappo500_in_the_weaker_slot() -> None:
    """Same fraction, same ``default_rng(MIXTURE_RNG_BASE + 50)``, same sort: given the same two pools the new tier
    selects exactly what ``mix50`` selects."""
    expert = [stream(d, -100.0 - d, directory="expert", episode=f"e{d:03d}.npz") for d in range(1, 201)]
    weaker = [stream(d, -500.0 - d, directory="weaker", episode=f"w{d:03d}.npz") for d in range(1, 201)]
    ours = grid.mixture_training_streams(grid.A27_TIERS["mix_m1000_m500"], expert, weaker)
    theirs = grid.mixture_training_streams(grid.TIERS["mix50"], expert, weaker)
    assert [s.key for s in ours] == [s.key for s in theirs]
    assert sum(1 for s in ours if s.dataset_dir == "expert") == 100


# ----------------------------------------------------------------------
# The real corpus: A27(e)'s design facts by the registered functions, and by an independent route
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_v11_root() -> Path:
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO_ROOT / "datasets_v11"
    if not candidate.is_dir():
        pytest.skip(f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to run A27(e)'s checks")
    return candidate


@pytest.fixture(scope="module")
def mixture_selection(corpus_v11_root: Path) -> tuple[Any, ...]:
    spec = grid.A27_TIERS["mix_m1000_m500"]
    dataset = grid.tier_dataset(spec, corpus_v11_root)
    components = grid._component_streams(spec, corpus_v11_root, context_length=grid.CONTEXT_LENGTH)
    return tuple(grid.training_streams(spec, dataset, component_streams=components))


def test_the_overlap_mixture_reproduces_a27es_design_facts_on_the_corpus(mixture_selection: tuple[Any, ...]) -> None:
    kept = grid.top_decile_streams(mixture_selection)
    by_dir: dict[str, int] = {}
    for entry in kept:
        by_dir[Path(entry.dataset_dir).name] = by_dir.get(Path(entry.dataset_dir).name, 0) + 1
    assert by_dir == A27E_KEPT
    assert (min(s.total_return for s in kept), max(s.total_return for s in kept)) == A27E_DECILE_RETURNS
    declared = grid.assert_declaration_matches_corpus(grid.A27_TIERS["mix_m1000_m500"], mixture_selection)
    assert (declared["target_rtg"], declared["rtg_scale"]) == (-5959.0, 11043.0)


def test_the_decile_returns_recomputed_from_the_raw_npz_agree(mixture_selection: tuple[Any, ...]) -> None:
    """The independent route ``tests/test_mixture_tiers.py`` uses: ``np.load`` and a float64 sum of ``ix0_local_reward``,
    no loader in the path."""
    recomputed = {}
    for entry in mixture_selection:
        with np.load(Path(entry.dataset_dir) / entry.episode_file) as arrays:
            recomputed[entry.key] = float(np.sum(arrays["ix0_local_reward"], dtype=np.float64))
    assert all(recomputed[s.key] == float(s.total_return) for s in mixture_selection)
    ordered = sorted(mixture_selection, key=lambda s: (-recomputed[s.key], s.dataset_dir, s.episode_file, s.ix_id))
    top = ordered[:20]
    kept = grid.top_decile_streams(mixture_selection)
    assert [s.key for s in top] == [s.key for s in kept]
    assert max(recomputed.values()) == -5959.0
    assert max(abs(v) for v in recomputed.values()) == 11043.0


def test_the_mixtures_expert_half_is_mix50s(corpus_v11_root: Path, mixture_selection: tuple[Any, ...]) -> None:
    spec = grid.TIERS["mix50"]
    dataset = grid.tier_dataset(spec, corpus_v11_root)
    components = grid._component_streams(spec, corpus_v11_root, context_length=grid.CONTEXT_LENGTH)
    mix50 = grid.training_streams(spec, dataset, component_streams=components)
    ours = {s.key for s in mixture_selection if "mappo1000" in Path(s.dataset_dir).name}
    theirs = {s.key for s in mix50 if "mappo1000" in Path(s.dataset_dir).name}
    assert len(ours) == 100 and ours == theirs


# ----------------------------------------------------------------------
# The training CLI: names from flags, refusals before anything is written, defaults unchanged
# ----------------------------------------------------------------------


def _fixture_tier(tmp_path: Path, name: str = "cli") -> tuple[Any, Path]:
    directory = write_tier_dir(
        tmp_path, f"tier_{name}",
        episodes=[(1, -1.0, 100.0), (1, -3.0, 130.0), (2, -2.0, 110.0), (2, -4.0, 140.0)],
    )
    dataset = TrajectoryWindowDataset([directory], context_length=CONTEXT, split="train")
    spec = fixture_spec(name, [directory.name], subsample="one_per_draw")
    target, scale = grid.recomputed_target_and_scale(grid.training_streams(spec, dataset))
    return fixture_spec(name, [directory.name], subsample="one_per_draw", target_rtg=target, rtg_scale=scale), directory


def _train_args(tmp_path: Path, tier: str, *, checkpoints: Path, **extra: Any) -> argparse.Namespace:
    values = {"corpus_root": str(tmp_path), "tier": tier, "methods": "bc,bc_top10,iql,dt", "steps": 2,
              "log_every": 0, "device": "cpu", "checkpoint_dir": str(checkpoints)}
    values.update(extra)
    return argparse.Namespace(**values)


class _Spy:
    """Wraps a trainer: records the keyword arguments of every call and calls through -- or, in a test that expects a
    refusal, raises at once, so a missing refusal fails fast instead of training for real."""

    def __init__(self, real: Any, *, call_through: bool) -> None:
        self.real = real
        self.call_through = call_through
        self.calls: list[dict[str, Any]] = []

    def __call__(self, batch: Any, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if not self.call_through:
            raise AssertionError("a trainer was called where the call should have been refused before any training")
        return self.real(batch, **kwargs)


def _install_spies(monkeypatch: pytest.MonkeyPatch, *, call_through: bool = True) -> dict[str, _Spy]:
    import offline.dt_gate as dt_gate

    spies = {name: _Spy(real, call_through=call_through)
             for name, real in (("bc", grid.train_bc), ("iql", grid.train_iql), ("dt", dt_gate.train_dt))}
    monkeypatch.setattr(grid, "train_bc", spies["bc"])
    monkeypatch.setattr(grid, "train_iql", spies["iql"])
    monkeypatch.setattr(dt_gate, "train_dt", spies["dt"])
    return spies


def test_the_defaults_train_the_registered_arms_with_the_registered_keywords(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under the defaults nothing changes: names, batch sizes, and no new keyword handed to any trainer."""
    spec, _ = _fixture_tier(tmp_path)
    spies = _install_spies(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    checkpoints = tmp_path / "ckpt"
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(grid.TIERS, "cli", spec)
        patch.setattr(grid, "CONTEXT_LENGTH", CONTEXT)
        assert grid._run_train(_train_args(tmp_path, "cli", checkpoints=checkpoints), out_dir) == 0
    registered_iql = {"state_dim", "n_actions", "seed", "declared_gradient_steps", "batch_size", "device",
                      "checkpoint_path", "stats", "scenario_id", "provenance", "log_every"}
    assert all(set(call) == registered_iql for call in spies["iql"].calls) and len(spies["iql"].calls) == 5
    assert all(call["batch_size"] == 1_280 for call in spies["iql"].calls)
    assert all(call["batch_size"] == 64 for call in spies["bc"].calls) and len(spies["bc"].calls) == 10
    assert "target_dropout" not in json.dumps([sorted(c) for c in spies["bc"].calls])
    names = sorted(p.name for p in checkpoints.iterdir())
    assert names == sorted(f"cli_{m}_seed{s}.pt" for m in ("bc", "bc_top10", "iql", "dt") for s in grid.TRAINING_SEEDS)
    payload = json.loads((out_dir / "p4_6_training.json").read_text(encoding="utf-8"))
    assert {run["method"] for run in payload["runs"]} == {"bc", "bc_top10", "iql", "dt"}
    assert all("target_dropout" not in run for run in payload["runs"])


def test_the_target_flag_trains_iql_t_under_its_own_name_into_a_p9_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _ = _fixture_tier(tmp_path)
    spies = _install_spies(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    checkpoints = tmp_path / "p9_1_runs" / "checkpoints"
    args = _train_args(tmp_path, "cli", checkpoints=checkpoints, methods="iql", iql_target_dropout="off",
                       artifact_prefix="p9_1")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(grid.TIERS, "cli", spec)
        patch.setattr(grid, "CONTEXT_LENGTH", CONTEXT)
        assert grid._run_train(args, out_dir) == 0
    assert [call["target_dropout"] for call in spies["iql"].calls] == ["off"] * 5
    assert sorted(p.name for p in checkpoints.iterdir()) == sorted(f"cli_iql_t_seed{s}.pt" for s in grid.TRAINING_SEEDS)
    payload = json.loads((out_dir / "p9_1_training.json").read_text(encoding="utf-8"))
    assert {(r["method"], r["base_method"], r["target_dropout"], r["batch_size"]) for r in payload["runs"]} == {
        ("iql_t", "iql", "off", 1_280)}
    for run in payload["runs"]:
        provenance = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)["provenance"]
        assert provenance["target_dropout"] == "off" and provenance["arm"] == "iql_t"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"iql_target_dropout": "off", "artifact_prefix": "p4_6"}, "registered artifact prefix"),
        ({"iql_target_dropout": "off", "artifact_prefix": "p4_7"}, "registered artifact prefix"),
        ({"bc_batch_windows": 1_024, "artifact_prefix": "p9_1", "methods": "bc"}, "not an arm A27 declared"),
        ({"steps": 100_000, "artifact_prefix": "p9_1", "iql_target_dropout": "off", "methods": "iql"},
         "not an arm A27 declared"),
    ],
)
def test_an_a27_training_call_is_refused_before_anything_is_trained_or_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: dict[str, Any], message: str
) -> None:
    spec, _ = _fixture_tier(tmp_path)
    spies = _install_spies(monkeypatch, call_through=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    checkpoints = tmp_path / "p9_1_runs" / "checkpoints"
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(grid.TIERS, "cli", spec)
        patch.setattr(grid, "CONTEXT_LENGTH", CONTEXT)
        with pytest.raises(ValueError, match=message):
            grid._run_train(_train_args(tmp_path, "cli", checkpoints=checkpoints, **extra), out_dir)
    assert all(not s.calls for s in spies.values())
    assert not checkpoints.exists() and sorted(out_dir.iterdir()) == []


def test_an_a27_run_outside_a_p9_directory_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A27(a): new runs write only into new ``output/p9_*`` directories."""
    spec, _ = _fixture_tier(tmp_path)
    spies = _install_spies(monkeypatch, call_through=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    args = _train_args(tmp_path, "cli", checkpoints=tmp_path / "p4_6" / "checkpoints", methods="iql",
                       iql_target_dropout="off", artifact_prefix="p9_1")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(grid.TIERS, "cli", spec)
        with pytest.raises(ValueError, match="p9_"):
            grid._run_train(args, out_dir)
    assert not spies["iql"].calls and not (tmp_path / "p4_6").exists()


def test_an_a27_run_refuses_a_checkpoint_path_that_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``BRIEF_46`` 3.1: refuse to write a checkpoint whose path exists -- checked for every seed before the first one
    trains, so a refusal leaves the earlier file and trains nothing."""
    spec, _ = _fixture_tier(tmp_path)
    spies = _install_spies(monkeypatch, call_through=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    checkpoints = tmp_path / "p9_1_runs" / "checkpoints"
    checkpoints.mkdir(parents=True)
    earlier = checkpoints / "cli_iql_t_seed505.pt"
    earlier.write_bytes(b"earlier")
    args = _train_args(tmp_path, "cli", checkpoints=checkpoints, methods="iql", iql_target_dropout="off",
                       artifact_prefix="p9_1")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(grid.TIERS, "cli", spec)
        patch.setattr(grid, "CONTEXT_LENGTH", CONTEXT)
        with pytest.raises(FileExistsError, match="cli_iql_t_seed505.pt"):
            grid._run_train(args, out_dir)
    assert not spies["iql"].calls and earlier.read_bytes() == b"earlier"
    assert sorted(p.name for p in checkpoints.iterdir()) == ["cli_iql_t_seed505.pt"]


def test_a_registered_arm_on_a_registered_tier_is_refused_in_an_a27_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--iql-target-dropout off --methods bc,iql`` would retrain ``bc@<registered tier>`` under its registered name;
    A27(a) gives every new cell a new name, so the call is refused rather than half-run."""
    spec, _ = _fixture_tier(tmp_path)
    spies = _install_spies(monkeypatch, call_through=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    args = _train_args(tmp_path, "cli", checkpoints=tmp_path / "p9_1_runs" / "checkpoints", methods="bc,iql",
                       iql_target_dropout="off", artifact_prefix="p9_1")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(grid.TIERS, "cli", spec)
        with pytest.raises(ValueError, match="registered cell"):
            grid._run_train(args, out_dir)
    assert all(not s.calls for s in spies.values())


def test_the_recipes_batch_is_what_the_trainer_receives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plumbing, on a HYPOTHETICAL recipe: A27 declares no hz1x1 BC arm at another batch, so on the real registry a
    trainer call that ignored the recipe's batch would be indistinguishable; patching one in shows the batch the CLI
    derives is the batch the trainer gets."""
    import offline.a27_arms as arms

    hypothetical = arms.ArmRecipe("hz1x1", "bc_probe", "bc", 128, None, 40_000, "test only")
    monkeypatch.setitem(arms.A27_RECIPES, ("hz1x1", "bc_probe"), hypothetical)
    spec, _ = _fixture_tier(tmp_path)
    spies = _install_spies(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    args = _train_args(tmp_path, "cli", checkpoints=tmp_path / "p9_1_runs" / "checkpoints", methods="bc",
                       bc_batch_windows=128, artifact_prefix="p9_1")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(grid.TIERS, "cli", spec)
        patch.setattr(grid, "CONTEXT_LENGTH", CONTEXT)
        assert grid._run_train(args, out_dir) == 0
    assert [call["batch_size"] for call in spies["bc"].calls] == [128] * 5
    assert [call["method"] for call in spies["bc"].calls] == ["bc"] * 5


@pytest.mark.parametrize(
    ("methods", "extra", "expected"),
    [
        ("dt,bc,bc_top10,iql", {"steps": 100_000}, ["dt_100k", "bc_100k", "bc_top10_100k", "iql_100k"]),
        ("iql", {"iql_target_dropout": "off"}, ["iql_t"]),
        ("dt,bc,bc_top10,iql", {}, ["dt", "bc", "bc_top10", "iql"]),
    ],
)
def test_the_arm_a_training_call_trains_is_derived_from_its_flags(
    tmp_path: Path, methods: str, extra: dict[str, Any], expected: list[str]
) -> None:
    args = _train_args(tmp_path, "mappo1000", checkpoints=tmp_path, methods=methods, **extra)
    assert [grid.training_recipe(method, args).name for method in methods.split(",")] == expected


# ----------------------------------------------------------------------
# The evaluation CLI: A27 names load through their base method's loader at their declared budget
# ----------------------------------------------------------------------


def _evaluate_args(tmp_path: Path, tier: str, method: str, steps: int) -> argparse.Namespace:
    return argparse.Namespace(corpus_root=str(tmp_path), tier=tier, method=method, steps=steps, device="cpu",
                              scenario_id="fixture1x1", engine_seed=1000, artifact_prefix="p9_1")


def test_a_registered_name_at_the_ceiling_is_refused_by_evaluate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pinned at 40000"):
        grid._run_evaluate(_evaluate_args(tmp_path, "mappo1000", "bc", 100_000), lambda d: tmp_path, tmp_path, tmp_path)


def test_a_100k_arm_is_refused_below_the_ceiling(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="declared at 100000"):
        grid._run_evaluate(_evaluate_args(tmp_path, "mix50", "dt_100k", 40_000), lambda d: tmp_path, tmp_path, tmp_path)


@pytest.mark.parametrize(("method", "expected"), [("iql_t", ("baseline", "iql")), ("iql_100k", ("baseline", "iql")),
                                                  ("bc_100k", ("baseline", "bc")), ("dt_100k", ("dt", None)),
                                                  ("iql", ("baseline", "iql")), ("dt", ("dt", None))])
def test_an_arm_is_evaluated_through_its_base_methods_loader(
    monkeypatch: pytest.MonkeyPatch, method: str, expected: tuple[str, str | None]
) -> None:
    """``offline_baselines._baseline_factory`` picks IQLAgent only for the literal name ``"iql"``; handing it
    ``"iql_t"`` would load an IQL checkpoint as BC.  The loader is chosen from the recipe's base method."""
    seen: list[tuple[str, str | None]] = []
    monkeypatch.setattr(grid, "_dt_factory", lambda path, declared, target, device: seen.append(("dt", None)))
    monkeypatch.setattr(grid, "_baseline_factory",
                        lambda base, path, declared, device: seen.append(("baseline", base)))
    assert grid.cell_factory(method, "/x.pt", 40_000, -1.0, None) is None
    assert seen == [expected]


# ----------------------------------------------------------------------
# The registered BC checkpoint, regenerated through the CLI with --bc-batch-windows 64 given (gated)
# ----------------------------------------------------------------------


def _output_root() -> Path:
    env_value = os.environ.get("RLTRAFFIC_OUTPUT_ROOT")
    candidate = Path(env_value) if env_value else REPO_ROOT / "output"
    if not (candidate / "p4_6" / "checkpoints" / "maxpressure_bc_seed101.pt").is_file():
        pytest.skip(f"no P4.6 checkpoint under {candidate}: set RLTRAFFIC_OUTPUT_ROOT to the main tree's output/")
    return candidate


def test_a_registered_bc_checkpoint_regenerates_tensor_for_tensor_with_the_batch_flag_given(
    corpus_v11_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``BRIEF_46`` 4.2: P4.6's ``maxpressure_bc_seed101.pt``, retrained by ``_run_train`` with ``--bc-batch-windows 64``
    stated explicitly, seeds narrowed to 101 by monkeypatch, on CUDA with one torch thread as P4.6 trained it."""
    registered = _output_root() / "p4_6" / "checkpoints" / "maxpressure_bc_seed101.pt"
    if not torch.cuda.is_available():
        pytest.skip("maxpressure_bc_seed101.pt was trained on CUDA; a CPU retrain differs for a reason that is not the "
                    "change under test")
    payload = torch.load(registered, map_location="cpu", weights_only=False)
    assert payload["provenance"]["device"] == "cuda" and payload["provenance"]["batch_size"] == 64
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    args = grid.build_parser().parse_args([
        "--corpus-root", str(corpus_v11_root), "--out-dir", str(out_dir), "--checkpoint-dir", str(tmp_path / "ckpt"),
        "--device", "cuda", "--torch-threads", "1", "train", "--tier", "maxpressure", "--methods", "bc",
        "--bc-batch-windows", "64", "--log-every", "0",
    ])
    monkeypatch.setattr(grid, "TRAINING_SEEDS", (101,))
    previous = grid.pin_torch_threads(1)
    try:
        assert grid._run_train(args, out_dir) == 0
    finally:
        grid.pin_torch_threads(previous)
    regenerated = torch.load(tmp_path / "ckpt" / "maxpressure_bc_seed101.pt", map_location="cpu", weights_only=False)
    differing = [k for k in sorted(payload["model"]) if not torch.equal(regenerated["model"][k], payload["model"][k])]
    assert sorted(regenerated["model"]) == sorted(payload["model"])
    assert differing == [], f"first differing tensor: {differing[:1]}"
    assert regenerated["canonical_digest"] == payload["canonical_digest"]
