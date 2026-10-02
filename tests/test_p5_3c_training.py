"""P5.3c C2 (``BRIEF_42`` §3 C2, §4; ``PREREGISTRATION`` A26(a)-(b) as corrected by A26.1; Amendment A): the sixty
trainings of H4's context-length sweep, each written once.

What this file pins, and the named mutations it is built against (``BRIEF_42`` §4):

* **T-table** -- exactly A26's 60 runs by name, subject, K, batch and seed; K = 20 at batch 64 only; the
  equal-supervision batches 1,280 and 640; a 61st name, a K outside {1, 2, 5, 10, 20}, an unregistered batch, seed or
  subject refused.  *Mutation:* K = 3 admitted -> dies.
* **T-recipe** -- through ``train_run``, the route every run takes, ``train_dt`` is called ONCE with ``context_length``
  K, the run's batch, exactly 40,000 declared steps and ``raise_to None``, and the prompt and scale the CORPUS gives
  (recomputed here from the fixture's own reward generator); the registered command wires the registered budget and
  destination; the loss per supervised target is the mean cross-entropy over non-PAD positions, recomputed on one batch
  by an independent route; gated, both real subjects' prompt, scale and statistics equal P4's and P4.7's committed
  values, READ from their artifacts and checkpoints.  *Mutations:* a raise applied -> dies; batch 64 for ``k1_b1280``
  -> dies.
* **T-once** -- an absent checkpoint trains, a valid one is skipped, an invalid one is refused and never overwritten;
  an existing destination is refused BEFORE ``train_dt`` runs; two same-seed CPU runs at B = 20 are byte-identical
  under two different file names (the in-memory serialisation); a checkpoint at another budget or another warm-up --
  the fenced timing run's shape under the registered name it shares -- fails exactly ``budget`` or ``recipe``
  (Amendment B, B3.2(b)).  *Mutations:* ``budget`` made self-consistent, ``recipe``'s warm-up read from the payload
  -> die.
* **T-k20** -- identical tensors compare equal; one changed element is named with its largest absolute difference;
  the record is written whatever the comparison finds and the command exits 0.  *Mutation:* the comparison made a
  refusal -> dies.
* **The caller-level twins of C1's T-k1 / T-k2 / T-index** (plan F6, Amendment A Q6), through THIS module's route --
  red first, unlike C1's characterisation tests.

No test here trains on CUDA, reads the registered corpus ungated, or writes outside ``tmp_path``.  The registered
regime's refusals are exercised with injected state (the environment variable, the code tree's git state), and its
one torch thread is asserted from a start at three (Amendment B, B3.2(b); *mutation:* the pin removed -> dies).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import offline.context_sweep as cs
from agent.DTAgent import DecisionTransformer, DTAgent, DTConfig, action_loss
from offline.dataset import PAD_ACTION
from offline.few_shot import _load_weights_only, write_payload_exclusive
from offline.method_tier_grid import canonical_digest_of
from tests.p5_3c_fixtures import (
    N_ACTIONS,
    STATE_DIM,
    T_DECISIONS,
    StubEnv,
    build_k20_tree,
    episode_return,
    info_at,
    synthetic_subject,
    tiny_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"
SEEDS = (101, 202, 303, 404, 505)
FIXED_CODE = {"code_commit": "c" * 40, "code_dirty": False}

#: The synthetic subject: two directories, four episodes of T decisions, ONE intersection.
LAYOUT = (("fixture_a", (1, 2), 0), ("fixture_b", (3, 4), 2))
N_EPISODES = 4


def _subject(tmp_path: Path, subject: str = "mappo1000") -> Any:
    return synthetic_subject(tmp_path / "corpus", subject, layout=LAYOUT)


def _corpus(tmp_path: Path) -> Path:
    return tmp_path / "corpus"


# ======================================================================================================================
# T-table
# ======================================================================================================================


def _expected_table() -> list[tuple[str, int, int, int]]:
    """A26(a)-(b)'s runs in the plan's order (§9), written out HERE rather than read from the module."""
    rows: list[tuple[str, int, int, int]] = []
    for k in (20, 10, 5, 2, 1):
        rows.extend(("mappo1000", k, 64, seed) for seed in SEEDS)
    rows.extend(("mappo1000", 1, 1280, seed) for seed in SEEDS)
    rows.extend(("mappo1000", 2, 640, seed) for seed in SEEDS)
    for k in (20, 10, 5, 2, 1):
        rows.extend(("mix50", k, 64, seed) for seed in SEEDS)
    return rows


def test_the_registered_table_is_exactly_a26s_sixty_runs() -> None:
    runs = cs.registered_runs()
    assert [(run.subject, run.k, run.batch, run.seed) for run in runs] == _expected_table()
    assert len(runs) == 60 == len({run.name for run in runs})
    assert all(run.name == f"{run.subject}_k{run.k}_b{run.batch}_seed{run.seed}" for run in runs)
    assert all(run.arm == f"{run.subject}_k{run.k}_b{run.batch}" for run in runs)
    assert all(cs.run_by_name(run.name) == run for run in runs)
    assert cs.registered_arms() == tuple(dict.fromkeys(run.arm for run in runs))
    assert len(cs.registered_arms()) == 12


def test_k20_trains_at_batch_64_only_and_the_equal_supervision_batches_are_1280_and_640() -> None:
    counts = Counter((run.subject, run.k, run.batch) for run in cs.registered_runs())
    assert {key: n for key, n in counts.items() if key[1] == 20} == {("mappo1000", 20, 64): 5, ("mix50", 20, 64): 5}
    assert sorted(key for key in counts if key[2] != 64) == [("mappo1000", 1, 1280), ("mappo1000", 2, 640)]
    assert all(n == 5 for n in counts.values())
    # A26(b): the equal-supervision batches make the targets per step equal K = 20's maximum.
    assert 1 * 1280 == 2 * 640 == 20 * 64


@pytest.mark.parametrize(
    ("subject", "k", "batch", "seed", "reason"),
    [
        ("mappo1000", 3, 64, 101, r"K 3 "),
        ("mappo1000", 4, 64, 101, r"K 4 "),
        ("mappo1000", 40, 64, 101, r"K 40 "),
        ("mix50", 1, 1280, 101, r"batch 1280 "),
        ("mappo1000", 20, 1280, 101, r"batch 1280 "),
        ("mappo1000", 5, 128, 101, r"batch 128 "),
        ("mappo1000", 5, 64, 606, r"seed 606 "),
        ("random", 5, 64, 101, r"subject 'random' "),
    ],
)
def test_an_unregistered_k_batch_seed_or_subject_is_refused(
    subject: str, k: int, batch: int, seed: int, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        cs.validate_run(cs.RunSpec(subject=subject, k=k, batch=batch, seed=seed))


def test_a_61st_run_name_is_refused() -> None:
    for name in ("mappo1000_k3_b64_seed101", "mix50_k1_b1280_seed101", "mappo1000_k20_b64_seed606", "anything"):
        with pytest.raises(ValueError, match="not one of the 60 registered runs"):
            cs.run_by_name(name)


# ======================================================================================================================
# T-recipe
# ======================================================================================================================


class _SpyStop(Exception):
    """Raised by the spy so nothing trains once the call's arguments are captured."""


def _spy_train_dt(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def spy(stacked: Any, **kwargs: Any) -> Any:
        calls.append({"stacked": stacked, **kwargs})
        raise _SpyStop("the spy captured the call; nothing trained")

    monkeypatch.setattr(cs, "train_dt", spy, raising=True)
    return calls


@pytest.mark.parametrize(
    ("subject", "k", "batch", "seed"),
    [("mappo1000", 1, 1280, 101), ("mappo1000", 2, 640, 202), ("mappo1000", 5, 64, 303), ("mix50", 20, 64, 404)],
)
def test_train_run_hands_train_dt_the_registered_recipe_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subject: str, k: int, batch: int, seed: int
) -> None:
    """*Mutations:* a raise applied (a second call, or ``raise_to`` set) -> dies; batch 64 for ``k1_b1280`` -> dies."""
    spec = _subject(tmp_path, subject)
    inputs = cs.training_inputs(spec, k, _corpus(tmp_path))
    run = cs.RunSpec(subject=subject, k=k, batch=batch, seed=seed)
    destination = tmp_path / "out" / f"{run.name}.pt"
    destination.parent.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    calls = _spy_train_dt(monkeypatch)

    with pytest.raises(_SpyStop, match="the spy captured the call"):
        cs.train_run(run, inputs, device="cpu", destination=destination, staging_dir=staging, code=FIXED_CODE)

    assert len(calls) == 1
    call = calls[0]
    assert call["context_length"] == k
    assert call["batch_size"] == batch
    assert call["declared_gradient_steps"] == 40_000 == cs.GRADIENT_STEPS
    assert call["raise_to"] is None
    assert call["seed"] == seed
    assert call["target_rtg"] == spec.target_rtg == max(episode_return(e) for e in range(N_EPISODES))
    assert call["rtg_scale"] == spec.rtg_scale == max(abs(episode_return(e)) for e in range(N_EPISODES))
    assert call["stacked"] is inputs.batch
    assert call["stats"] is inputs.stats
    assert call["state_dim"] == STATE_DIM and call["n_actions"] == N_ACTIONS
    provenance = call["provenance"]
    assert provenance["run"] == run.name and provenance["arm"] == run.arm and provenance["subject"] == subject
    assert provenance["registered_context_length"] == k and provenance["registered_batch_size"] == batch
    assert provenance["deterministic"] is False
    assert provenance["code_commit"] == FIXED_CODE["code_commit"]
    assert set(provenance["corpus_manifest_sha256"]) == {str(_corpus(tmp_path) / name) for name, _d, _f in LAYOUT}
    assert not destination.exists()
    assert list(staging.iterdir()) == [], "the staged file must be removed on every path"


def test_the_registered_train_command_wires_the_registered_budget_destination_and_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CALLER of ``train_run`` (``PROJECT_PLAN`` §7: a pin on a function does not pin its caller)."""
    spec = _subject(tmp_path, "mappo1000")
    output_root = tmp_path / "output"
    for directory in (cs.checkpoints_dir(output_root), cs.runs_dir(output_root), cs.staging_dir(output_root)):
        directory.mkdir(parents=True)
    captured: list[dict[str, Any]] = []

    def spy(run: Any, inputs: Any, **kwargs: Any) -> Any:
        captured.append({"run": run, "inputs": inputs, **kwargs})
        raise _SpyStop("the spy captured the call; nothing trained")

    monkeypatch.setattr(cs, "enter_registered_regime", lambda: {"code": dict(FIXED_CODE), "regime": {}}, raising=True)
    monkeypatch.setattr(cs, "subject_spec", lambda subject: spec, raising=True)
    monkeypatch.setattr(cs, "train_run", spy, raising=True)
    with pytest.raises(_SpyStop, match="the spy captured the call"):
        cs.main(
            [
                "train", "--run", "mappo1000_k1_b1280_seed101", "--output-root", str(output_root),
                "--corpus-root", str(_corpus(tmp_path)), "--device", "cuda",
            ]
        )
    assert len(captured) == 1
    call = captured[0]
    assert call["run"] == cs.RunSpec("mappo1000", 1, 1280, 101)
    assert call["inputs"].k == 1
    assert call.get("budget", cs.GRADIENT_STEPS) == 40_000
    assert call["device"] == "cuda"
    assert Path(call["destination"]) == output_root / "p5_3c_training" / "checkpoints" / "mappo1000_k1_b1280_seed101.pt"
    assert Path(call["staging_dir"]) == output_root / "p5_3c_training" / "staging"
    assert call["code"] == FIXED_CODE


def test_the_registered_train_command_refuses_a_device_other_than_cuda(tmp_path: Path) -> None:
    status = cs.main(
        [
            "train", "--run", "mappo1000_k5_b64_seed101", "--output-root", str(tmp_path / "output"),
            "--corpus-root", str(tmp_path / "corpus"), "--device", "cpu",
        ]
    )
    assert status == 2
    assert not (tmp_path / "output").exists()


def test_the_prompt_and_scale_are_the_corpus_values_and_a_disagreeing_declaration_is_refused(tmp_path: Path) -> None:
    spec = _subject(tmp_path)
    returns = [episode_return(episode) for episode in range(N_EPISODES)]
    for k in (1, 20):
        inputs = cs.training_inputs(spec, k, _corpus(tmp_path))
        assert inputs.target_rtg == max(returns)
        assert inputs.rtg_scale == max(abs(value) for value in returns)
        assert inputs.n_streams == N_EPISODES and inputs.n_windows == N_EPISODES * T_DECISIONS
    wrong = dataclasses.replace(spec, target_rtg=spec.target_rtg + 1.0)
    with pytest.raises(ValueError, match="maximum return"):
        cs.training_inputs(wrong, 5, _corpus(tmp_path))
    with pytest.raises(ValueError, match="K 3 "):
        cs.training_inputs(spec, 3, _corpus(tmp_path))


def test_the_loss_per_supervised_target_is_the_mean_cross_entropy_over_non_pad_positions(tmp_path: Path) -> None:
    """Amendment A, A1.3: the per-step loss ``train_dt`` records IS the loss per supervised target.

    Recomputed here on one batch by an independent route: a float64 log-softmax, the logged action's log-probability
    gathered at every position the attention mask marks, summed and divided by that COUNT.  The denominator is asserted
    exactly; the value to 1e-12 against the float64 cross-entropy and to 1e-6 against ``action_loss`` in float32 -- two
    different float algorithms, so ``==`` is not the claim.  A denominator of every position (padding included) is
    shown to be a different number.
    """
    spec = _subject(tmp_path)
    inputs = cs.training_inputs(spec, 5, _corpus(tmp_path))
    torch.manual_seed(7)
    model = DecisionTransformer(
        DTConfig(state_dim=STATE_DIM, n_actions=N_ACTIONS, context_length=5, max_ep_len=T_DECISIONS)
    ).eval()
    rows = next(iter(cs.replay_index_stream(101, inputs.n_windows, 64, 1)))
    picked = torch.from_numpy(rows.astype(np.int64))
    batch = {key: value[picked] for key, value in inputs.batch.items()}
    with torch.no_grad():
        logits = model(
            batch["rtg"] / inputs.rtg_scale, batch["state"], batch["action"], batch["timestep"],
            batch["attention_mask"], batch["avail_mask"],
        )
    mask = batch["attention_mask"]
    assert bool((batch["action"][~mask] == PAD_ACTION).all()) and bool((batch["action"][mask] >= 0).all())
    count = int(mask.sum())
    assert 64 <= count < 64 * 5, "the batch must mix padded and real positions for the denominator to matter"

    log_probs = torch.log_softmax(logits.double(), dim=-1)
    rows_i, cols_j = torch.nonzero(mask, as_tuple=True)
    terms = [float(log_probs[i, j, int(batch["action"][i, j])]) for i, j in zip(rows_i.tolist(), cols_j.tolist())]
    assert len(terms) == count
    independent = -math.fsum(terms) / count

    reference64 = float(
        torch.nn.functional.cross_entropy(
            logits.double().reshape(-1, N_ACTIONS), batch["action"].reshape(-1), ignore_index=PAD_ACTION
        )
    )
    assert math.isclose(independent, reference64, rel_tol=1e-12)
    recorded = float(action_loss(logits, batch["action"]))
    assert math.isclose(recorded, independent, rel_tol=1e-6)
    over_every_position = -math.fsum(terms) / (64 * 5)
    assert not math.isclose(recorded, over_every_position, rel_tol=1e-3)


def test_the_run_record_carries_train_dts_window_means_and_the_replayed_targets_per_step(tmp_path: Path) -> None:
    spec = _subject(tmp_path)
    run = cs.RunSpec("mappo1000", 2, 640, 101)
    inputs = cs.training_inputs(spec, 2, _corpus(tmp_path))
    output_root = tmp_path / "output"
    for directory in (cs.checkpoints_dir(output_root), cs.runs_dir(output_root), cs.staging_dir(output_root)):
        directory.mkdir(parents=True)

    outcome = cs.train_run(
        run, inputs, device="cpu", destination=cs.registered_destination(output_root, run),
        staging_dir=cs.staging_dir(output_root), code=FIXED_CODE, budget=20,
    )
    path = cs.write_run_record(run, outcome, inputs, output_root=output_root, code=FIXED_CODE, device="cpu")
    record = json.loads(path.read_text(encoding="utf-8"))
    payload = _load_weights_only(outcome.destination)

    assert record["format_version"] == cs.RUN_RECORD_FORMAT_VERSION
    assert (record["run"], record["subject"], record["k"], record["batch"], record["seed"]) == (
        run.name, "mappo1000", 2, 640, 101,
    )
    assert record["steps"] == 20 == len(record["losses"])
    assert record["losses"] == list(outcome.losses)
    assert record["final_loss"] == outcome.losses[-1]
    assert record["loss_per_supervised_target"] == list(payload["provenance"]["window_means"])
    assert record["loss_per_supervised_target"] == list(outcome.window_means)
    assert record["checkpoint_sha256"] == hashlib.sha256(outcome.destination.read_bytes()).hexdigest()
    assert record["weights_sha256"] == canonical_digest_of(outcome.destination)

    # The targets per step, recomputed by this test's own loop over the seed's draws.
    per_window = inputs.batch["attention_mask"].sum(dim=1).numpy()
    generator = np.random.default_rng(101)
    totals = [int(per_window[generator.integers(0, len(per_window), size=640)].sum()) for _ in range(20)]
    targets = record["supervised_targets_per_step"]
    assert (targets["min"], targets["max"], targets["total"], targets["steps"], targets["batch"]) == (
        min(totals), max(totals), sum(totals), 20, 640,
    )
    assert targets["mean"] == sum(totals) / 20
    assert 640 < targets["min"] and targets["max"] <= 2 * 640

    with pytest.raises(ValueError, match="already exists"):
        cs.write_run_record(run, outcome, inputs, output_root=output_root, code=FIXED_CODE, device="cpu")


@pytest.mark.skipif(
    not (os.environ.get("RLTRAFFIC_CORPUS_V11") and os.environ.get("RLTRAFFIC_OUTPUT_ROOT")),
    reason=(
        "needs RLTRAFFIC_CORPUS_V11 (the gitignored datasets_v11 corpus) and RLTRAFFIC_OUTPUT_ROOT (P4's output/p4_dt "
        "and P4.7's output/p4_7/checkpoints) to compare both subjects' recipe with the committed values"
    ),
)
def test_both_real_subjects_prompt_scale_and_statistics_equal_p4s_and_p4_7s_committed_values() -> None:
    """Gated: READ from the corpus, compared with the committed artifacts and checkpoints -- never typed here."""
    corpus = Path(os.environ["RLTRAFFIC_CORPUS_V11"])
    output_root = Path(os.environ["RLTRAFFIC_OUTPUT_ROOT"])
    p4_training = json.loads((DATA / "p4_training.json").read_text(encoding="utf-8"))
    p4 = _load_weights_only(output_root / "p4_dt" / "dt_seed101.pt")
    p47 = _load_weights_only(output_root / "p4_7" / "checkpoints" / "mix50_dt_seed101.pt")

    mappo = cs.training_inputs(cs.subject_spec("mappo1000"), 1, corpus)
    assert mappo.target_rtg == p4_training["target_rtg"] == p4["target_rtg"]
    assert mappo.rtg_scale == p4_training["rtg_scale"] == p4["rtg_scale"]
    assert mappo.stats.to_json_obj() == p4["stats"]
    assert mappo.n_windows == 72_000 and mappo.max_timestep + 1 == p4["config"]["max_ep_len"]

    mix = cs.training_inputs(cs.subject_spec("mix50"), 1, corpus)
    assert mix.target_rtg == p47["target_rtg"] and mix.rtg_scale == p47["rtg_scale"]
    assert mix.statistics_digest == p47["provenance"]["statistics_digest"]
    assert mix.stats.to_json_obj() == p47["stats"]
    assert mix.n_windows == 72_000 and mix.n_streams == 200


# ======================================================================================================================
# T-once
# ======================================================================================================================


def _fabricate_registered(tmp_path: Path, output_root: Path, run: Any, spec: Any, *, place_as: Any = None) -> Path:
    """A payload trained at B = 20 on CPU, its provenance rewritten to the registered budget and device, then written
    to the registered destination of ``place_as`` (default *run*) -- a checkpoint that validates without 40,000 steps."""
    inputs = cs.training_inputs(spec, run.k, _corpus(tmp_path))
    scratch = tmp_path / "scratch" / run.name
    scratch.mkdir(parents=True)
    outcome = cs.train_run(
        run, inputs, device="cpu", destination=scratch / "c.pt", staging_dir=scratch, code=FIXED_CODE, budget=20
    )
    payload = _load_weights_only(outcome.destination)
    payload["provenance"] = {
        **payload["provenance"],
        "gradient_steps": cs.GRADIENT_STEPS,
        "declared_gradient_steps": cs.GRADIENT_STEPS,
        "warmup_steps": 1_000,
        "device": "cuda",
    }
    destination = cs.registered_destination(output_root, place_as or run)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_payload_exclusive(payload, destination)
    return destination


def test_an_absent_checkpoint_trains_a_valid_one_is_skipped_an_invalid_one_is_refused_and_never_overwritten(
    tmp_path: Path,
) -> None:
    spec = _subject(tmp_path)
    output_root = tmp_path / "output"
    facts = cs.subject_facts(cs.training_inputs(spec, 1, _corpus(tmp_path)))
    run = cs.RunSpec("mappo1000", 5, 64, 101)

    assert cs.resume_decision(run, output_root=output_root, facts=facts) == "train"
    _fabricate_registered(tmp_path, output_root, run, spec)
    checks = cs.validate_checkpoint(run, output_root=output_root, facts=facts)
    assert all(checks.values()), checks
    assert cs.resume_decision(run, output_root=output_root, facts=facts) == "skip"

    other = cs.RunSpec("mappo1000", 5, 64, 202)
    wrong = _fabricate_registered(tmp_path, output_root, cs.RunSpec("mappo1000", 2, 64, 202), spec, place_as=other)
    before = wrong.read_bytes()
    with pytest.raises(ValueError, match=r"does not validate \(failed: .*'context_length'"):
        cs.resume_decision(other, output_root=output_root, facts=facts)
    assert wrong.read_bytes() == before

    garbage_run = cs.RunSpec("mappo1000", 5, 64, 303)
    garbage = cs.registered_destination(output_root, garbage_run)
    garbage.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="cannot be read"):
        cs.resume_decision(garbage_run, output_root=output_root, facts=facts)
    assert garbage.read_bytes() == b"not a checkpoint"


@pytest.mark.parametrize(
    ("overrides", "failing"),
    [
        ({"gradient_steps": 400, "declared_gradient_steps": 400}, {"budget"}),
        ({"warmup_steps": 200}, {"recipe"}),
        ({"gradient_steps": 400, "declared_gradient_steps": 400, "warmup_steps": 200}, {"budget", "recipe"}),
        ({"learning_rate": 2e-4}, {"recipe"}),
        ({"weight_decay": 1e-3}, {"recipe"}),
        ({"grad_clip": 1.0}, {"recipe"}),
    ],
    ids=[
        "another-budget-self-consistent", "another-warm-up", "the-fenced-timing-runs-shape", "another-learning-rate",
        "another-weight-decay", "another-clip",
    ],
)
def test_a_checkpoint_at_another_budget_or_warm_up_fails_exactly_budget_or_recipe(
    tmp_path: Path, overrides: dict[str, float], failing: set[str]
) -> None:
    """``BRIEF_42`` Amendment B, B3.2(b): ``checks["budget"]`` compares BOTH step counts with the registered 40,000, not
    with each other, and ``checks["recipe"]`` compares the warm-up with ``train_dt``'s 1,000 at 40,000 steps, not with
    the payload's own.  They are the two checks that tell the fenced timing checkpoint (400 steps, warm-up 200) from a
    registered one under the name the two share, ``mappo1000_k5_b64_seed101`` -- so each payload here differs from a
    VALID one (the control) in those provenance fields alone, and the failing set is asserted exactly.  Amendment B.1,
    B.1.3(2): the recipe's other three members -- the learning rate, the weight decay, the clip -- each alone.

    *Mutations:* ``budget`` made self-consistent (reviewer A's MB at G1); ``recipe``'s warm-up compared with the
    payload's own (MC); ``recipe`` without its learning rate, its weight decay or its clip (MR1-MR3) -> this dies.
    """
    spec = _subject(tmp_path)
    facts = cs.subject_facts(cs.training_inputs(spec, 1, _corpus(tmp_path)))
    run = cs.TIMING_RUN
    assert run in cs.registered_runs() and run.name == "mappo1000_k5_b64_seed101"

    control_root = tmp_path / "control"
    valid = _load_weights_only(_fabricate_registered(tmp_path, control_root, run, spec))
    control = cs.validate_checkpoint(run, output_root=control_root, facts=facts)
    assert all(control.values()), control

    output_root = tmp_path / "output"
    destination = cs.registered_destination(output_root, run)
    destination.parent.mkdir(parents=True)
    write_payload_exclusive({**valid, "provenance": {**valid["provenance"], **overrides}}, destination)
    checks = cs.validate_checkpoint(run, output_root=output_root, facts=facts)
    assert {name for name, passed in checks.items() if not passed} == failing, checks
    before = destination.read_bytes()
    with pytest.raises(ValueError, match=r"does not validate \(failed: "):
        cs.resume_decision(run, output_root=output_root, facts=facts)
    assert destination.read_bytes() == before


def test_an_existing_destination_is_refused_before_train_dt_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _subject(tmp_path)
    inputs = cs.training_inputs(spec, 5, _corpus(tmp_path))
    destination = tmp_path / "out" / "occupied.pt"
    destination.parent.mkdir()
    destination.write_bytes(b"occupied")
    calls = _spy_train_dt(monkeypatch)
    with pytest.raises(ValueError, match="already exists"):
        cs.train_run(
            cs.RunSpec("mappo1000", 5, 64, 101), inputs, device="cpu", destination=destination,
            staging_dir=tmp_path / "out", code=FIXED_CODE,
        )
    assert calls == []
    assert destination.read_bytes() == b"occupied"


def test_two_same_seed_cpu_runs_at_b20_are_byte_identical_under_two_file_names(tmp_path: Path) -> None:
    spec = _subject(tmp_path)
    inputs = cs.training_inputs(spec, 5, _corpus(tmp_path))
    results = []
    for folder, name, seed in (("a", "first.pt", 101), ("b", "second.pt", 101), ("c", "third.pt", 202)):
        directory = tmp_path / folder
        directory.mkdir()
        results.append(
            cs.train_run(
                cs.RunSpec("mappo1000", 5, 64, seed), inputs, device="cpu", destination=directory / name,
                staging_dir=directory, code=FIXED_CODE, budget=20,
            )
        )
    first, second, other_seed = results
    assert first.destination.read_bytes() == second.destination.read_bytes()
    assert first.sha256 == second.sha256 == hashlib.sha256(first.destination.read_bytes()).hexdigest()
    assert other_seed.sha256 != first.sha256, "the control: another seed must give other bytes"
    payload = _load_weights_only(first.destination)
    assert payload["format_version"] == "dt-checkpoint/1.0"
    assert payload["config"]["context_length"] == 5 and payload["provenance"]["gradient_steps"] == 20
    for directory in ("a", "b", "c"):
        assert sorted(path.name for path in (tmp_path / directory).iterdir()) in (
            ["first.pt"], ["second.pt"], ["third.pt"],
        ), "only the published file may remain"


def test_attempt_markers_are_counted_and_written_exclusively(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    run = cs.RunSpec("mix50", 10, 64, 404)
    with pytest.raises(ValueError, match="does not exist"):
        cs.next_attempt(run, output_root=output_root)
    assert not output_root.exists()
    cs.attempts_dir(output_root).mkdir(parents=True)
    assert cs.attempts_of(run, output_root=output_root) == ()
    assert cs.next_attempt(run, output_root=output_root) == 1
    assert cs.next_attempt(run, output_root=output_root) == 2
    assert cs.attempts_of(run, output_root=output_root) == (1, 2)
    assert cs.attempts_of(cs.RunSpec("mix50", 10, 64, 505), output_root=output_root) == ()


def test_the_manifest_lists_exactly_the_declared_checkpoints_refuses_a_stray_and_is_written_once(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    runs = (cs.RunSpec("mappo1000", 20, 64, 101), cs.RunSpec("mix50", 1, 64, 505))
    directory = cs.checkpoints_dir(output_root)
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="absent"):
        cs.write_manifest(output_root, runs=runs)
    for run in runs:
        (directory / f"{run.name}.pt").write_bytes(f"stand-in for {run.name}".encode("utf-8"))
    manifest = cs.write_manifest(output_root, runs=runs)
    assert manifest == output_root / "SHA256SUMS_p5_3c_train.txt"
    # sha256sum's format, one line per checkpoint, ordered by PATH (few_shot.write_manifest's order).
    expected = [
        f"{hashlib.sha256((directory / f'{name}.pt').read_bytes()).hexdigest()}  p5_3c_training/checkpoints/{name}.pt"
        for name in sorted(run.name for run in runs)
    ]
    assert manifest.read_text(encoding="utf-8").splitlines() == expected
    assert cs.write_manifest(output_root, runs=runs) == manifest
    (directory / "stray.pt").write_bytes(b"stray")
    with pytest.raises(ValueError, match="not one of the declared runs"):
        cs.write_manifest(output_root, runs=runs)
    (directory / "stray.pt").unlink()
    (directory / f"{runs[0].name}.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        cs.write_manifest(output_root, runs=runs)


# ======================================================================================================================
# T-k20: the reproduction MEASUREMENT (A26(b), A26.1(a)) -- a record, never a stop
# ======================================================================================================================


def test_identical_models_compare_equal_and_one_changed_element_is_named_with_its_difference() -> None:
    reference = tiny_model(101)
    same = {key: value.clone() for key, value in reference.items()}
    result = cs.compare_models(same, reference)
    assert result["all_equal"] is True and result["keys_equal"] is True
    assert result["n_parameters_differing"] == 0
    assert result["largest_abs_difference"] == 0.0 and result["largest_abs_difference_parameter"] is None
    assert [entry["name"] for entry in result["parameters"]] == sorted(reference)

    changed = {key: value.clone() for key, value in reference.items()}
    changed["head.weight"][1, 2] += 0.375
    result = cs.compare_models(changed, reference)
    assert result["all_equal"] is False and result["n_parameters_differing"] == 1
    assert result["largest_abs_difference_parameter"] == "head.weight"
    expected = abs(float(changed["head.weight"][1, 2].double()) - float(reference["head.weight"][1, 2].double()))
    assert result["largest_abs_difference"] == expected > 0.0
    by_name = {entry["name"]: entry for entry in result["parameters"]}
    assert by_name["head.weight"]["equal"] is False and by_name["head.weight"]["max_abs_difference"] == expected
    assert by_name["embed.bias"]["equal"] is True and by_name["embed.bias"]["max_abs_difference"] == 0.0

    missing = {key: value.clone() for key, value in reference.items() if key != "step"}
    result = cs.compare_models(missing, reference)
    assert result["keys_equal"] is False and result["missing_in_candidate"] == ["step"]
    assert result["all_equal"] is False


def test_the_k20_record_is_written_whatever_the_comparison_finds_and_the_command_exits_0(tmp_path: Path) -> None:
    """*Mutation:* the comparison made a refusal (a difference raises, or the command exits non-zero) -> this dies."""
    output_root, data_dir = build_k20_tree(tmp_path, differ={("mix50", 303): ("embed.weight", (0, 1), 0.5)})
    status = cs.main(["compare-k20", "--output-root", str(output_root), "--data-dir", str(data_dir)])
    assert status == 0
    record = json.loads((output_root / "p5_3c_training" / "k20_reproduction.json").read_text(encoding="utf-8"))
    assert record["format_version"] == cs.K20_FORMAT_VERSION
    subjects = record["subjects"]
    assert subjects["mappo1000"]["all_equal"] is True and subjects["mappo1000"]["n_seeds_equal"] == 5
    assert subjects["mix50"]["all_equal"] is False and subjects["mix50"]["n_seeds_equal"] == 4
    entry = next(item for item in subjects["mix50"]["seeds"] if item["seed"] == 303)
    candidate = _load_weights_only(output_root / "p5_3c_training" / "checkpoints" / "mix50_k20_b64_seed303.pt")
    reference = _load_weights_only(output_root / "p4_7" / "checkpoints" / "mix50_dt_seed303.pt")
    expected = float((candidate["model"]["embed.weight"].double() - reference["model"]["embed.weight"].double()).abs().max())
    assert entry["largest_abs_difference_parameter"] == "embed.weight"
    assert entry["largest_abs_difference"] == expected > 0.0
    assert entry["weights_digests_equal"] is False
    assert all(item["all_equal"] for item in subjects["mappo1000"]["seeds"])
    assert (
        cs.main(["compare-k20", "--output-root", str(output_root), "--data-dir", str(data_dir)]) == 0
    ), "the same record again is a no-op"


def test_a_published_checkpoint_at_another_digest_refuses_the_measurement_and_writes_nothing(tmp_path: Path) -> None:
    output_root, data_dir = build_k20_tree(tmp_path)
    (output_root / "p4_dt" / "dt_seed202.pt").write_bytes(b"another file")
    status = cs.main(["compare-k20", "--output-root", str(output_root), "--data-dir", str(data_dir)])
    assert status == 2
    assert not (output_root / "p5_3c_training" / "k20_reproduction.json").exists()


# ======================================================================================================================
# The caller-level twins of C1's T-k1 / T-k2 / T-index (plan F6), through THIS module's route
# ======================================================================================================================


@pytest.mark.parametrize("k", [1, 2])
def test_the_sweep_route_builds_k_step_windows_ending_at_t_and_writes_k_into_the_checkpoint(
    tmp_path: Path, k: int
) -> None:
    spec = _subject(tmp_path)
    inputs = cs.training_inputs(spec, k, _corpus(tmp_path))
    assert inputs.k == k
    assert tuple(inputs.batch["state"].shape) == (N_EPISODES * T_DECISIONS, k, STATE_DIM)
    for row, (_directory, _episode, _ix, t) in enumerate(inputs.window_ids):
        low = max(0, t - k + 1)
        pad = k - (t - low + 1)
        assert inputs.batch["timestep"][row].tolist() == [0] * pad + list(range(low, t + 1)), (row, t)
        assert inputs.batch["attention_mask"][row].tolist() == [False] * pad + [True] * (t - low + 1), (row, t)

    run = cs.RunSpec("mappo1000", k, 64, 101)
    destination = tmp_path / "out" / f"{run.name}.pt"
    destination.parent.mkdir()
    cs.train_run(run, inputs, device="cpu", destination=destination, staging_dir=destination.parent, code=FIXED_CODE, budget=6)
    payload = _load_weights_only(destination)
    assert payload["config"]["context_length"] == k and payload["provenance"]["context_length"] == k

    agent = DTAgent.from_checkpoint(StubEnv(), str(destination), device="cpu")
    shapes: list[tuple[int, ...]] = []
    original = agent.model.forward

    def spy(*args: Any, **kwargs: Any) -> Any:
        shapes.append(tuple(args[1].shape))
        return original(*args, **kwargs)

    agent.model.forward = spy  # type: ignore[method-assign]
    agent.act(info_at(0, [0.25] * STATE_DIM, [True] * N_ACTIONS, 0.0), explore=False, update_memory=True)
    agent.act(info_at(1, [1.25] * STATE_DIM, [True] * N_ACTIONS, -1.0), explore=False, update_memory=True)
    assert shapes == [(1, k, STATE_DIM), (1, k, STATE_DIM)]


def _draws_through_train_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, inputs: Any, k: int, seed: int
) -> np.ndarray:
    real = np.random.default_rng
    drawn: list[np.ndarray] = []

    class Recording:
        def __init__(self, generator_seed: Any) -> None:
            self._generator = real(generator_seed)

        def integers(self, *args: Any, **kwargs: Any) -> Any:
            out = self._generator.integers(*args, **kwargs)
            drawn.append(np.asarray(out).copy())
            return out

    destination = tmp_path / f"index_k{k}_s{seed}" / "c.pt"
    destination.parent.mkdir()
    with monkeypatch.context() as patch:
        patch.setattr(np.random, "default_rng", lambda generator_seed=None: Recording(generator_seed))
        cs.train_run(
            cs.RunSpec("mappo1000", k, 64, seed), inputs, device="cpu", destination=destination,
            staging_dir=destination.parent, code=FIXED_CODE, budget=25,
        )
    return np.concatenate(drawn)


def test_the_sweep_route_draws_the_same_window_ids_at_k_1_2_20_and_the_replay_is_that_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _subject(tmp_path)
    ids: dict[int, list[Any]] = {}
    rows: dict[int, np.ndarray] = {}
    for k in (1, 2, 20):
        inputs = cs.training_inputs(spec, k, _corpus(tmp_path))
        rows[k] = _draws_through_train_run(monkeypatch, tmp_path, inputs, k, 101)
        ids[k] = [inputs.window_ids[int(row)] for row in rows[k]]
        replay = np.concatenate(list(cs.replay_index_stream(101, inputs.n_windows, 64, 25)))
        assert np.array_equal(replay, rows[k]), f"the replay is not the stream train_dt drew at K={k}"
    assert rows[1].shape == (25 * 64,)
    assert np.array_equal(rows[1], rows[2]) and np.array_equal(rows[1], rows[20])
    assert ids[1] == ids[2] == ids[20]
    control = _draws_through_train_run(
        monkeypatch, tmp_path, cs.training_inputs(spec, 1, _corpus(tmp_path)), 1, 202
    )
    assert not np.array_equal(control, rows[1])


# ======================================================================================================================
# The output fence, the timing fence and the registered regime
# ======================================================================================================================


@pytest.mark.parametrize(
    "relative",
    [
        "p4_dt/dt_seed101.pt", "p4_7/checkpoints/mix50_dt_seed101.pt", "p8_4b_rederivation/x.json",
        "p5_3b/eval.json", "SHA256SUMS_p4_7.txt", "p5_3c_trainingX/a.pt", "p5_3c/cells/x.json",
    ],
)
def test_a_write_outside_this_tasks_training_entries_is_refused(tmp_path: Path, relative: str) -> None:
    output_root = tmp_path / "output"
    with pytest.raises(ValueError, match="read-only here"):
        cs.assert_writable(output_root / relative, output_root)
    assert not output_root.exists()


def test_this_tasks_training_entries_are_writable(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    for relative in (
        "p5_3c_training/checkpoints/a.pt", "p5_3c_runs/TOKEN_train", "SHA256SUMS_p5_3c_train.txt",
        "p5_3c_training/fenced_timing/20260928T101010Z/alone.pt",
    ):
        assert cs.assert_writable(output_root / relative, output_root) == output_root / relative


def test_registered_and_timing_checkpoints_are_fenced_from_each_other(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    run = cs.RunSpec("mappo1000", 5, 64, 101)
    with pytest.raises(ValueError, match="not under fenced_timing"):
        cs.assert_fence(cs.registered_destination(output_root, run), timing=True)
    with pytest.raises(ValueError, match="lies under fenced_timing"):
        cs.assert_fence(cs.timing_destination(output_root, "20260928T101010Z", "alone"), timing=False)
    with pytest.raises(ValueError, match="stamp"):
        cs.timing_destination(output_root, "yesterday", "alone")
    with pytest.raises(ValueError, match="slot"):
        cs.timing_destination(output_root, "20260928T101010Z", "pair1")
    assert cs.TIMING_RUN == cs.RunSpec("mappo1000", 5, 64, 101) and cs.TIMING_BUDGET == 400


def test_the_registered_regime_refuses_a_set_cublas_workspace_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        cs.enter_registered_regime()


def test_the_registered_regime_refuses_a_dirty_code_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(cs, "_code_provenance", lambda: {"code_commit": "d" * 40, "code_dirty": True}, raising=True)
    with pytest.raises(ValueError, match="dirty"):
        cs.enter_registered_regime()


def test_the_registered_regime_refuses_a_working_directory_at_another_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Q14: ``train_dt`` stamps ``runtime_provenance()``, whose commit is read in the CWD."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(cs, "_code_provenance", lambda: {"code_commit": "d" * 40, "code_dirty": False}, raising=True)
    monkeypatch.setattr(cs, "_cwd_commit", lambda: "e" * 40, raising=True)
    with pytest.raises(ValueError, match="working directory"):
        cs.enter_registered_regime()


def test_the_registered_regime_pins_exactly_one_torch_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """``BRIEF_42`` Amendment B, B3.2(b): ``enter_registered_regime()`` PINS one torch thread (P5.2's and P7.3c's
    regime) whatever the process started with, and records it.  The start here is three threads, so the test cannot
    pass by inheriting one; the code tree, the working directory's commit and CUDA are injected, so it runs anywhere.

    *Mutation:* ``torch.set_num_threads(1)`` removed (reviewer A's MD at G1) -> this dies.
    """
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(cs, "_code_provenance", lambda: dict(FIXED_CODE), raising=True)
    monkeypatch.setattr(cs, "_cwd_commit", lambda: FIXED_CODE["code_commit"], raising=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=True)
    assert not torch.are_deterministic_algorithms_enabled()
    started_with = torch.get_num_threads()
    try:
        torch.set_num_threads(3)
        assert torch.get_num_threads() == 3
        entered = cs.enter_registered_regime()
        assert torch.get_num_threads() == 1
        assert entered["regime"]["torch_num_threads"] == 1
        assert entered["code"] == FIXED_CODE
    finally:
        torch.set_num_threads(started_with)


# ======================================================================================================================
# The fenced timing summary (C2's G5-like measurement)
# ======================================================================================================================


def _slot(path: Path, *, seconds: float, sha: str, weights: str) -> None:
    path.write_text(
        json.dumps(
            {
                "format_version": cs.TIMING_SLOT_FORMAT_VERSION, "run": cs.TIMING_RUN.name, "steps": 400,
                "loop_seconds": seconds, "build_seconds": 2.5, "peak_allocated_mib": 300.0,
                "checkpoint_sha256": sha, "weights_sha256": weights, "device_name": "fixture GPU",
                "code_commit": "c" * 40,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_the_timing_summary_measures_ms_per_step_and_compares_the_repeat_by_two_routes(tmp_path: Path) -> None:
    stamp = tmp_path / "output" / "p5_3c_training" / "fenced_timing" / "20260928T101010Z"
    stamp.mkdir(parents=True)
    _slot(stamp / "alone.json", seconds=2.0, sha="a" * 64, weights="w" * 64)
    _slot(stamp / "repeat.json", seconds=2.2, sha="a" * 64, weights="w" * 64)
    (stamp / "nvidia_smi_alone.csv").write_text("1200\n1450\n1300\n", encoding="utf-8")
    (stamp / "nvidia_smi_repeat.csv").write_text("1210\n1440\n", encoding="utf-8")
    summary = cs.summarize_timing(stamp)
    assert summary["format_version"] == cs.TIMING_FORMAT_VERSION
    assert summary["slots"]["alone"]["ms_per_step"] == 2.0 / 400 * 1000
    assert summary["slots"]["repeat"]["ms_per_step"] == 2.2 / 400 * 1000
    assert summary["slots"]["alone"]["device_peak_mib"] == 1450.0
    assert summary["repeat"] == {"file_sha256_equal": True, "weights_sha256_equal": True}
    assert summary["concurrency"] == 1

    _slot(stamp / "repeat.json", seconds=2.2, sha="b" * 64, weights="v" * 64)
    assert cs.summarize_timing(stamp)["repeat"] == {"file_sha256_equal": False, "weights_sha256_equal": False}
    (stamp / "nvidia_smi_repeat.csv").unlink()
    with pytest.raises(ValueError, match="nvidia_smi_repeat.csv"):
        cs.summarize_timing(stamp)


# ======================================================================================================================
# The training record: refused on a partial set, and carrying every field the brief lists
# ======================================================================================================================


def test_the_training_record_carries_every_listed_field_and_refuses_a_partial_set(tmp_path: Path) -> None:
    spec = _subject(tmp_path)
    output_root = tmp_path / "output"
    for directory in (cs.checkpoints_dir(output_root), cs.runs_dir(output_root), cs.staging_dir(output_root),
                      cs.attempts_dir(output_root)):
        directory.mkdir(parents=True, exist_ok=True)
    runs = (cs.RunSpec("mappo1000", 5, 64, 101), cs.RunSpec("mappo1000", 5, 64, 202))
    for run in runs:
        cs.next_attempt(run, output_root=output_root)
        destination = _fabricate_registered(tmp_path, output_root, run, spec)
        inputs = cs.training_inputs(spec, run.k, _corpus(tmp_path))
        payload = _load_weights_only(destination)
        outcome = cs.TrainOutcome(
            destination=destination, sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), steps=40_000,
            warmup=1_000, losses=(0.5,) * 39_999 + (0.25,), window_means=tuple(payload["provenance"]["window_means"]),
            seconds=12.0,
        )
        cs.write_run_record(run, outcome, inputs, output_root=output_root, code=FIXED_CODE, device="cuda")
    timing = tmp_path / "timing.json"
    timing.write_text(json.dumps({"format_version": cs.TIMING_FORMAT_VERSION, "concurrency": 1}) + "\n", encoding="utf-8")
    k20 = cs.k20_record_path(output_root)
    k20.write_text(json.dumps({"format_version": cs.K20_FORMAT_VERSION, "subjects": {}}) + "\n", encoding="utf-8")
    specs = {"mappo1000": spec}

    with pytest.raises(ValueError, match="SHA256SUMS_p5_3c_train.txt is absent"):
        cs.build_train_record(output_root, corpus_root=_corpus(tmp_path), timing_path=timing, runs=runs, specs=specs)
    cs.write_manifest(output_root, runs=runs)
    record = cs.build_train_record(output_root, corpus_root=_corpus(tmp_path), timing_path=timing, runs=runs, specs=specs)
    assert record["format_version"] == cs.TRAIN_RECORD_FORMAT_VERSION and record["n_runs"] == 2
    for run in runs:
        entry = record["runs"][run.name]
        destination = cs.registered_destination(output_root, run)
        for field in (
            "subject", "k", "batch", "seed", "steps", "checkpoint_sha256", "weights_sha256", "loop_seconds",
            "final_loss", "loss_per_supervised_target", "supervised_targets_per_step", "attempts",
        ):
            assert field in entry, field
        assert (entry["subject"], entry["k"], entry["batch"], entry["seed"]) == (run.subject, run.k, run.batch, run.seed)
        assert entry["steps"] == 40_000 and entry["attempts"] == 1 and entry["reruns"] == 0
        assert entry["checkpoint_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
        assert entry["weights_sha256"] == canonical_digest_of(destination)
        assert entry["final_loss"] == 0.25 and entry["loop_seconds"] == 12.0
    assert record["k20_reproduction"]["sha256"] == hashlib.sha256(k20.read_bytes()).hexdigest()
    assert record["timing"]["sha256"] == hashlib.sha256(timing.read_bytes()).hexdigest()

    shutil.move(str(cs.run_record_path(output_root, runs[1])), str(tmp_path / "moved.json"))
    with pytest.raises(ValueError, match="run record"):
        cs.build_train_record(output_root, corpus_root=_corpus(tmp_path), timing_path=timing, runs=runs, specs=specs)
