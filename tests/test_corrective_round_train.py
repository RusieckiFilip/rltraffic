"""P9.1's campaign training (``offline/corrective_round.py``; ``BRIEF_46`` 3.4 ``train``): every missing run trained once,
through the registered trainers, on the inputs the CLIs build, recorded write-once; a done run skipped; anything in
between refused.

Formats: ``p9.1-run/1.0`` records; checkpoints are the trainers' own (``iql-``/``bc-checkpoint/1.0``, the DT's).  The
fixture corpora are contract C6 v1.1 (``tests/test_method_tier_grid.py``'s one-intersection tier; ``tests/
p5_2b_fixtures.py``'s 16-intersection tier written through the real logger).

Steps: the recipes are A27's (40,000 or 100,000 steps), which a test cannot afford; each test patches the recipe it
trains down to two steps IN THE REGISTRY (``offline.a27_arms``), the one place the campaign reads a budget from, so the
production code has no test-only knob.

T-TRAIN-ARGS: the campaign trains directly (plan D6) because the CLIs train every seed per call and are not resumable
per run; the tests below prove the campaign hands each trainer what the CLI hands it -- every keyword but the
checkpoint path, and the very tensors.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import torch

import offline.a27_arms as arms
import offline.corrective_round as cr
import offline.method_tier_grid as grid
import offline.tier_sweep as ts
from tests import p5_2b_fixtures as fx
from tests.test_method_tier_grid import CONTEXT, fixture_spec, write_tier_dir


def _roots(tmp_path: Path) -> cr.Roots:
    output = tmp_path / "output"
    output.mkdir()
    return cr.Roots(repo_root=tmp_path / "repo", output_root=output, corpus_root=tmp_path / "corpus",
                    draws_root=tmp_path / "draws")


def _two_steps(monkeypatch: pytest.MonkeyPatch, scenario: str, *names: str) -> None:
    for name in names:
        key = (scenario, name)
        table = arms.A27_RECIPES if key in arms.A27_RECIPES else arms.REGISTERED_RECIPES
        monkeypatch.setitem(table, key, dataclasses.replace(table[key], steps=2))


class _Spy:
    def __init__(self, real: Any) -> None:
        self.real, self.calls = real, []

    def __call__(self, batch: Any, **kwargs: Any) -> Any:
        self.calls.append({"batch": batch, **kwargs})
        return self.real(batch, **kwargs)


def _hz1x1_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Roots:
    roots = _roots(tmp_path)
    directory = write_tier_dir(Path(roots.corpus_root), "tier_maxpressure",
                               episodes=[(1, -1.0, 100.0), (1, -3.0, 130.0), (2, -2.0, 110.0), (2, -4.0, 140.0)])
    from offline.dataset import TrajectoryWindowDataset

    dataset = TrajectoryWindowDataset([directory], context_length=CONTEXT, split="train")
    spec = fixture_spec("maxpressure", [directory.name], subsample="one_per_draw")
    target, scale = grid.recomputed_target_and_scale(grid.training_streams(spec, dataset))
    monkeypatch.setitem(grid.TIERS, "maxpressure", fixture_spec(
        "maxpressure", [directory.name], subsample="one_per_draw", target_rtg=target, rtg_scale=scale))
    monkeypatch.setattr(grid, "CONTEXT_LENGTH", CONTEXT)
    return roots


def _grid_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Roots:
    roots = _roots(tmp_path)
    fx.write_synthetic_tier(Path(roots.corpus_root))
    fx.install_synthetic_tier(monkeypatch)
    return roots


def _cell(scenario: str, tier: str, arm: str) -> cr.Cell:
    return next(c for c in cr.training_cells() if (c.scenario, c.tier, c.arm) == (scenario, tier, arm))


# ----------------------------------------------------------------------
# One run: trained, moved into place, recorded
# ----------------------------------------------------------------------


def test_a_run_is_trained_once_and_recorded_with_its_clock_steps_and_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _hz1x1_fixture(tmp_path, monkeypatch)
    _two_steps(monkeypatch, "hz1x1", "iql_t")
    cell = _cell("hz1x1", "maxpressure", "iql_t")
    summary = cr.train(roots, cells=[cell], device="cpu", seeds=(101, 202))
    assert summary == {"trained": 2, "skipped_done": 0}
    for seed in (101, 202):
        assert cr.run_status(roots, cell, seed) == "done"
        record = json.loads(cr.run_record_path(roots, "hz1x1", "maxpressure", "iql_t", seed).read_text())
        checkpoint = cr.checkpoint_path(roots, "hz1x1", "maxpressure", "iql_t", seed)
        assert record["format_version"] == cr.RUN_RECORD_FORMAT_VERSION
        assert record["clock"] == "uncontrolled; no Best Performance regime, no canary; not a benchmark"
        assert record["gradient_steps"] == 2 and record["declared_steps"] == 2 and record["seconds"] > 0.0
        assert record["canonical_digest"] == grid.canonical_digest_of(checkpoint)
        assert record["file_sha256"] == grid.file_sha256(checkpoint)
        assert (record["base"], record["batch"], record["target_dropout"]) == ("iql", 1_280, "off")
        provenance = torch.load(checkpoint, map_location="cpu", weights_only=False)["provenance"]
        assert provenance["target_dropout"] == "off" and provenance["arm"] == "iql_t"
        assert not checkpoint.with_name(checkpoint.name + ".partial").exists()
    assert cr.train(roots, cells=[cell], device="cpu", seeds=(101, 202)) == {"trained": 0, "skipped_done": 2}


def test_a_damaged_run_refuses_the_whole_call_before_anything_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _hz1x1_fixture(tmp_path, monkeypatch)
    _two_steps(monkeypatch, "hz1x1", "iql_t")
    cell = _cell("hz1x1", "maxpressure", "iql_t")
    planted = cr.checkpoint_path(roots, "hz1x1", "maxpressure", "iql_t", 202)
    planted.parent.mkdir(parents=True)
    planted.with_name(planted.name + ".partial").write_bytes(b"interrupted")
    spy = _Spy(grid.train_iql)
    monkeypatch.setattr(grid, "train_iql", spy)
    with pytest.raises(ValueError, match="interrupted run"):
        cr.train(roots, cells=[cell], device="cpu", seeds=(101, 202))
    assert spy.calls == [] and not planted.exists()


def test_every_write_goes_through_the_barrier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _hz1x1_fixture(tmp_path, monkeypatch)
    _two_steps(monkeypatch, "hz1x1", "iql_t")
    seen: list[Path] = []
    real = cr.assert_target
    monkeypatch.setattr(cr, "assert_target", lambda path, r: seen.append(Path(path)) or real(path, r))
    cr.train(roots, cells=[_cell("hz1x1", "maxpressure", "iql_t")], device="cpu", seeds=(101,))
    names = {p.name for p in seen}
    assert {"maxpressure_iql_t_seed101.pt", "maxpressure_iql_t_seed101.pt.partial",
            "maxpressure__iql_t__seed101.json"} <= names


# ----------------------------------------------------------------------
# T-train-args: the campaign hands each trainer what the CLI hands it
# ----------------------------------------------------------------------


def _compare(campaign: dict[str, Any], cli: dict[str, Any]) -> None:
    assert set(campaign) == set(cli), "a keyword was added or dropped"
    for key in sorted(cli):
        if key in ("checkpoint_path", "provenance"):
            continue
        mine, theirs = campaign[key], cli[key]
        if key == "batch":
            if isinstance(theirs, dict):
                assert sorted(mine) == sorted(theirs)
                assert all(torch.equal(mine[k], theirs[k]) for k in theirs), key
            else:
                for name in ("state", "next_state", "action", "reward", "stream_index", "t"):
                    assert torch.equal(getattr(mine, name), getattr(theirs, name)), name
                assert mine.reward_scale == theirs.reward_scale
        elif key == "stats":
            assert mine.to_json_obj() == theirs.to_json_obj()
        else:
            assert str(mine) == str(theirs) if key == "device" else mine == theirs, key
    for key, value in cli["provenance"].items():
        if key != "runtime":
            assert campaign["provenance"][key] == value, key


def test_t_train_args_hz1x1_iql_t_matches_method_tier_grids_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _hz1x1_fixture(tmp_path, monkeypatch)
    _two_steps(monkeypatch, "hz1x1", "iql_t")
    spy = _Spy(grid.train_iql)
    monkeypatch.setattr(grid, "train_iql", spy)
    monkeypatch.setattr(grid, "TRAINING_SEEDS", (101,))
    cr.train(roots, cells=[_cell("hz1x1", "maxpressure", "iql_t")], device="cpu", seeds=(101,))
    out_dir = tmp_path / "cli_out"
    out_dir.mkdir()
    args = argparse.Namespace(corpus_root=str(roots.corpus_root), tier="maxpressure", methods="iql", steps=2,
                              log_every=0, device="cpu", checkpoint_dir=str(tmp_path / "p9_1_cli" / "ckpt"),
                              iql_target_dropout="off", artifact_prefix="p9_1")
    assert grid._run_train(args, out_dir) == 0
    campaign, cli = spy.calls
    _compare(campaign, cli)


@pytest.mark.parametrize(("arm", "flags"), [("bc_m", ("--methods", "bc", "--bc-batch-windows", "1024")),
                                            ("iql_m", ("--methods", "iql", "--iql-batch-transitions", "20480",
                                                       "--iql-target-dropout", "off")),
                                            ("bc_top10_perix_m", ("--methods", "bc_top10_perix",
                                                                  "--bc-batch-windows", "102"))])
def test_t_train_args_grid_arms_match_tier_sweeps_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str,
                                                      flags: tuple[str, ...]) -> None:
    import offline.offline_baselines as ob

    roots = _grid_fixture(tmp_path, monkeypatch)
    _two_steps(monkeypatch, "grid4x4", arm)
    spies = {"bc": _Spy(ob.train_bc), "iql": _Spy(ob.train_iql)}
    monkeypatch.setattr(ob, "train_bc", spies["bc"])
    monkeypatch.setattr(ob, "train_iql", spies["iql"])
    cr.train(roots, cells=[_cell("grid4x4", "random", arm)], device="cpu", seeds=(101,))
    args = ts.build_parser().parse_args([
        "--corpus-root", str(roots.corpus_root), "--work-dir", str(tmp_path / "p9_1_cli" / "work"),
        "--checkpoint-dir", str(tmp_path / "p9_1_cli" / "checkpoints"), "--reuse-root", str(tmp_path / "protected"),
        "--device", "cpu", "--seeds", "101", "--gradient-steps", "2", "--tier", "random", "train-baselines", *flags,
    ])
    assert ts._run_train_baselines(args) == 0
    calls = spies["iql" if arm.startswith("iql") else "bc"].calls
    campaign, cli = calls
    _compare(campaign, cli)


def test_t_train_args_the_grid_dt_matches_tier_sweeps_dt_trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``dt_nomix`` on ``random_fresh`` (A27(d)): the campaign calls ``train_tier_dt`` as ``tier_sweep train`` does."""
    roots = _roots(tmp_path)
    fx.write_synthetic_tier(Path(roots.corpus_root), episodes_per_draw=1)
    monkeypatch.setattr(ts, "A27_TIERS", {"random_fresh": ts.TierSpec(
        tier="random_fresh", dirs=(fx.RANDOM_DIR,), subsample="none", episode_count=3, ladder_att=float("nan"),
        mean_training_return=float("nan"))})
    monkeypatch.setattr(ts, "adjacency_for_tier", lambda *args, **kwargs: None)
    _two_steps(monkeypatch, "grid4x4", "dt_nomix")
    calls: list[dict[str, Any]] = []

    def spy(stacked: Any, index: Any, **kwargs: Any) -> Any:
        calls.append({"stacked": stacked, "index": index, **kwargs})
        torch.save({"model": {"w": torch.zeros(1)}, "provenance": {}}, Path(kwargs["checkpoint_path"]))
        from offline.spatial_mixing import TrainResult

        return TrainResult(method=kwargs["method"], seed=kwargs["seed"], gradient_steps=kwargs["gradient_steps"],
                           losses=(1.0, 0.5), checkpoint_path=str(kwargs["checkpoint_path"]), seconds=3.0)

    monkeypatch.setattr(ts, "train_tier_dt", spy)
    cr.train(roots, cells=[_cell("grid4x4", "random_fresh", "dt_nomix")], device="cpu", seeds=(101,))
    args = ts.build_parser().parse_args([
        "--corpus-root", str(roots.corpus_root), "--work-dir", str(tmp_path / "p9_1_cli" / "work"),
        "--checkpoint-dir", str(tmp_path / "p9_1_cli" / "checkpoints"), "--reuse-root", str(tmp_path / "protected"),
        "--device", "cpu", "--seeds", "101", "--gradient-steps", "2", "--tier", "random_fresh", "train",
        "--method", "dt_nomix",
    ])
    assert ts._run_train(args) == 0
    campaign, cli = calls
    assert set(campaign) == set(cli)
    for key in sorted(cli):
        if key in ("checkpoint_path", "provenance", "protected"):
            continue
        if key == "stacked":
            assert all(torch.equal(campaign[key][k], cli[key][k]) for k in cli[key])
        elif key == "index":
            assert campaign[key].node_ids == cli[key].node_ids
        elif key == "stats":
            assert campaign[key].to_json_obj() == cli[key].to_json_obj()
        elif key == "device":
            assert str(campaign[key]) == str(cli[key])
        else:
            assert campaign[key] == cli[key], key
    record = json.loads(cr.run_record_path(roots, "grid4x4", "random_fresh", "dt_nomix", 101).read_text())
    assert record["seconds"] == 3.0 and record["gradient_steps"] == 2 and record["deterministic"] is False


def test_the_hz1x1_aggregate_carries_done_runs_only_and_refuses_a_run_its_record_does_not_describe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K11 (added after the C5 mutation table): route (i) reads the aggregate, so a run enters it only if its status is
    DONE -- a record whose digest is not its checkpoint's stops the rebuild, the aggregate on disk untouched."""
    roots = _hz1x1_fixture(tmp_path, monkeypatch)
    _two_steps(monkeypatch, "hz1x1", "iql_t")
    monkeypatch.setattr(cr, "AGGREGATE_PREFIXES", {2: "p9_1", 100_000: "p9_1_100k"})  # the patched budget's file
    cell = _cell("hz1x1", "maxpressure", "iql_t")
    cr.train(roots, cells=[cell], device="cpu", seeds=(101,))
    cr.write_training_aggregates(roots)
    aggregate = Path(roots.run_root) / "p9_1_training.json"
    record_path = cr.run_record_path(roots, "hz1x1", "maxpressure", "iql_t", 101)
    record = json.loads(record_path.read_text())
    runs = json.loads(aggregate.read_text())["runs"]
    assert [(r["tier"], r["method"], r["seed"], r["canonical_digest"], r["seconds"]) for r in runs] == [
        ("maxpressure", "iql_t", 101, record["canonical_digest"], record["seconds"])]
    before = aggregate.read_bytes()
    record_path.write_text(json.dumps({**record, "canonical_digest": "0" * 64}))
    with pytest.raises(ValueError, match="not the ones its record wrote"):
        cr.write_training_aggregates(roots)
    assert aggregate.read_bytes() == before


def test_a_runs_record_brackets_its_training_with_unix_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The record carries when the run started and finished (wall clock, Unix seconds), so the report can tell a run
    that a machine suspension interrupted from the driver's GPU trace (BRIEF_46 3.5: seconds are an uncontrolled clock;
    a suspension inside a run is not noise but hours of a wrong number)."""
    import time

    roots = _hz1x1_fixture(tmp_path, monkeypatch)
    _two_steps(monkeypatch, "hz1x1", "iql_t")
    cell = _cell("hz1x1", "maxpressure", "iql_t")
    before = time.time()
    cr.train(roots, cells=[cell], device="cpu", seeds=(101,))
    after = time.time()
    record = json.loads(cr.run_record_path(roots, "hz1x1", "maxpressure", "iql_t", 101).read_text())
    assert before <= record["started_unix"] <= record["finished_unix"] <= after
    assert record["finished_unix"] - record["started_unix"] >= record["seconds"]
