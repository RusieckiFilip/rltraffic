"""⭐ GATE 2: threading ``rtg_mode`` through ``train_dt`` must not move the conditioned path.

``BRIEF_30`` section 4.1.  ``offline/dt_gate.py``'s ``train_dt`` trained **every DT number in the
paper**, so adding a parameter to it is the riskiest edit this task makes.  The instrument that
guards it is a **canonical ``state_dict`` digest** compared against the value committed in
``docs/data/p4_6_training.json`` -- ``==``, never a tolerance.

Why a digest and not P5.3a's ATT-identity test
-----------------------------------------------
``docs/reviews/P5.3a.md`` **MJ-3** measured that the ATT-identity test is **structurally blind to
every config field that only matters at training time**: forcing ``dropout=0.9`` passed it, because
dropout is inert under ``eval()``.  **P5.3b trains, so that blind spot is live**, and only something
computed from the weights themselves can see it.

Why this test can exist at all
-------------------------------
It rests on GPU training being bit-reproducible on this machine, which was **not** obvious:
``docs/returns/P5.2.md`` section 10.1 measured 61-63 of 66 tensors differing between two runs of the
**spatial** trainer, and ``offline/dt_gate.py`` sets no determinism flag.  ``docs/plans/p5.3b.md``
section 2 records the pre-edit control (G0-a, G0-b) that settled it on the unmodified tree before a
line was changed.  **Without that control, "the mutation made it fail" and "this environment cannot
reproduce a training run" are the same observation** (``BRIEF_28`` B5).

⚠️ **The full-budget test calls ``train_dt`` with NO ``rtg_mode`` argument, deliberately.**  Mutation
2 of section 4.1 changes the parameter's *default*; a call that passed ``rtg_mode="conditioned"``
explicitly would survive it and would certify nothing.

⚠️ **What this file does not cover** is enumerated in ``docs/plans/p5.3b.md`` section 5.1, seven
items.  The two that matter most: the digest is computed over ``payload["model"]`` alone
(``offline/method_tier_grid.py:1180-1185``), which is why AMENDMENT A5's whole-payload comparison is
here beside it; and one cell of forty is retrained, on this machine only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from agent.DTAgent import RTG_MODES, DTConfig
from offline.dataset import TrajectoryWindowDataset
from offline.dt_gate import BATCH_SIZE, build_training_dataset, stack_dataset, train_dt
from offline.method_tier_grid import (
    CONTEXT_LENGTH,
    DECLARED_GRADIENT_STEPS,
    canonical_digest_of,
)
from offline.nortg_campaign import (
    CONTROL_CELL,
    CONTROL_COMMITTED_DIGEST,
    NORTG_TIERS,
    assert_payload_matches_committed,
    training_inputs,
)

from tests.test_offline_dataset import write_dataset_dir

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs" / "data"
OUTPUT = REPO / "output"

#: The committed record the control cell is measured against.  Held as a literal here AND read from
#: the artifact by the test, so a silent edit to either one is caught by their disagreement.
CONTROL_DIGEST_LITERAL = "5d98d5351198c45054cce1e38b810dabd789708e71e3563e9428d37a49e0e563"

#: AMENDMENT A5: every payload key that must be identical between the retrained control cell and the
#: committed checkpoint.  ``model`` is the digest's job and ``provenance`` legitimately differs.
A5_COMPARED_KEYS = (
    "config",
    "format_version",
    "intersection_ids",
    "normalise",
    "rtg_scale",
    "scenario_id",
    "stats",
    "target_rtg",
)


# ----------------------------------------------------------------------
# A small real-loader fixture: cheap, on CPU, and it exercises the whole loop
# ----------------------------------------------------------------------


@pytest.fixture()
def fixture_dataset(tmp_path: Path) -> TrajectoryWindowDataset:
    dataset_dir = write_dataset_dir(tmp_path, "nortg_fixture__policy")
    return build_training_dataset([dataset_dir], context_length=4)


def _train_fixture(
    dataset: TrajectoryWindowDataset,
    destination: Path,
    *,
    steps: int = 40,
    rtg_scale: float = 10.0,
    **kwargs: Any,
) -> Any:
    """``train_dt`` on the fixture.  ``kwargs`` carries ``rtg_mode`` only when a test means to."""
    group = sorted(dataset.groups)[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    return train_dt(
        stack_dataset(dataset, group=group),
        state_dim=group[0],
        n_actions=group[1],
        seed=101,
        declared_gradient_steps=steps,
        raise_to=None,
        context_length=4,
        batch_size=8,
        device=torch.device("cpu"),
        checkpoint_path=destination,
        stats=dataset.stats,
        scenario_id="fixture_2ix",
        target_rtg=-1.0,
        rtg_scale=rtg_scale,
        provenance={"tier": "fixture"},
        **kwargs,
    )


def _config_of(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)["config"]


# ----------------------------------------------------------------------
# 1-5: the parameter reaches the training path, and it is not inert
# ----------------------------------------------------------------------


def test_train_dt_defaults_to_conditioned_and_writes_a_nine_key_config(
    fixture_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    """The default path, called exactly as ``method_tier_grid._run_train`` calls it.

    ⚠️ Killed by mutation 2 (parameter default ``"zero"``) in seconds, which is the point: a
    mechanism whose only guard is a 3.5-minute GPU test is a mechanism nobody re-runs.
    """
    result = _train_fixture(fixture_dataset, tmp_path / "ck" / "default.pt")
    config = _config_of(Path(result.checkpoint_path))
    assert config["rtg_mode"] == "conditioned"
    assert len(config) == 9, sorted(config)
    assert DTConfig.from_json_obj(config).rtg_mode == "conditioned"


def test_train_dt_threads_zero_into_the_checkpointed_config(
    fixture_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    """``rtg_mode`` must travel INSIDE the checkpoint, so ``DTAgent.load`` rebuilds the right model.

    ``agent/DTAgent.py``'s ``load`` reconstructs the base ``DecisionTransformer`` from the payload's
    config; a mode carried anywhere else would let the ordinary loader evaluate a zero-mode
    checkpoint as a conditioned one (``BRIEF_28`` section 4.1).
    """
    result = _train_fixture(fixture_dataset, tmp_path / "ck" / "zero.pt", rtg_mode="zero")
    config = _config_of(Path(result.checkpoint_path))
    assert config["rtg_mode"] == "zero"
    assert DTConfig.from_json_obj(config).rtg_mode == "zero"


def test_an_illegal_rtg_mode_raises_naming_both_values_and_writes_nothing(
    fixture_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    """The filesystem-mutation barrier, asserted rather than assumed."""
    destination = tmp_path / "ck" / "never.pt"
    with pytest.raises(ValueError, match="rtg_mode must be one of") as excinfo:
        _train_fixture(fixture_dataset, destination, rtg_mode="shuffled")
    message = str(excinfo.value)
    assert "conditioned" in message and "zero" in message, message
    assert not destination.exists(), "a rejected run must leave no checkpoint behind"
    assert sorted(RTG_MODES) == ["conditioned", "zero"]


def test_training_under_zero_reaches_different_weights_than_conditioned(
    fixture_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    """The parameter is not inert: same seed, same data, different weights.

    ⚠️ Killed by mutation 1 (``rtg_mode`` forced to ``"zero"`` inside ``train_dt``) in seconds.
    """
    conditioned = _train_fixture(fixture_dataset, tmp_path / "ck" / "c.pt")
    zeroed = _train_fixture(fixture_dataset, tmp_path / "ck" / "z.pt", rtg_mode="zero")
    assert canonical_digest_of(conditioned.checkpoint_path) != canonical_digest_of(
        zeroed.checkpoint_path
    )


def test_under_zero_the_rtg_scale_cannot_reach_the_weights_and_under_conditioned_it_can(
    fixture_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    """⭐ The ablation removed the return information from the training SIGNAL, by a second route.

    ``rtg_scale`` enters ``train_dt`` at exactly one place -- ``tensors["rtg"][index] /
    float(rtg_scale)`` -- so under ``rtg_mode="zero"`` two runs differing only in that divisor must
    reach **identical** weights, and under ``"conditioned"`` they must not.  This says the same
    thing as the test above without looking at a single weight, and it fails for a different reason
    if the substitution is applied in the wrong place.
    """
    zero_small = _train_fixture(
        fixture_dataset, tmp_path / "ck" / "zs.pt", rtg_scale=1.0, rtg_mode="zero"
    )
    zero_large = _train_fixture(
        fixture_dataset, tmp_path / "ck" / "zl.pt", rtg_scale=1e6, rtg_mode="zero"
    )
    assert canonical_digest_of(zero_small.checkpoint_path) == canonical_digest_of(
        zero_large.checkpoint_path
    )

    cond_small = _train_fixture(fixture_dataset, tmp_path / "ck" / "cs.pt", rtg_scale=1.0)
    cond_large = _train_fixture(fixture_dataset, tmp_path / "ck" / "cl.pt", rtg_scale=1e6)
    assert canonical_digest_of(cond_small.checkpoint_path) != canonical_digest_of(
        cond_large.checkpoint_path
    ), "the control: if this passes for both, the divisor is not reaching the model at all"


# ----------------------------------------------------------------------
# 6-9: GATE 2 itself, on the committed cell
# ----------------------------------------------------------------------


def test_the_control_cell_is_the_registered_one_and_is_not_random() -> None:
    """The Gate-2 cell is chosen by a rule, before the data, and the rule is in the docstring.

    ``BRIEF_30`` section 4.4: **``random`` may not be the control**, because its conditioned DT is
    already RTG-inert at the argmax (P5.3a: 0 of 7200 flips on every intervention), so a control
    there would pass whether or not ``rtg_mode`` reached the trainer.  ``mappo500`` is preferred
    over ``maxpressure`` -- the brief allows either -- because P5.3a measured its ``zero`` flip rate
    at 0.002361-0.005417 against ``maxpressure``'s 0.000139-0.000278.  And it is **not** one of the
    three campaign tiers, so the control stays independent of the result.
    """
    tier, seed = CONTROL_CELL
    assert (tier, seed) == ("mappo500", 101)
    assert tier != "random"
    assert tier not in NORTG_TIERS
    assert CONTROL_COMMITTED_DIGEST == CONTROL_DIGEST_LITERAL


def _committed_control_run() -> dict[str, Any]:
    tier, seed = CONTROL_CELL
    training = json.loads((DATA / "p4_6_training.json").read_text(encoding="utf-8"))
    runs = [
        run
        for run in training["runs"]
        if run["tier"] == tier and run["method"] == "dt" and int(run["seed"]) == seed
    ]
    assert len(runs) == 1, f"{tier} seed {seed}: {len(runs)} committed dt runs, expected 1"
    return runs[0]


def test_the_committed_digest_literal_agrees_with_the_artifact() -> None:
    """A literal and an artifact that must agree; their disagreement is the alarm."""
    assert _committed_control_run()["canonical_digest"] == CONTROL_DIGEST_LITERAL


def _corpus_root() -> Path:
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO / "datasets_v11"
    if not candidate.is_dir():
        pytest.skip(
            f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to a "
            "collected corpus to run the Gate-2 control retrain"
        )
    return candidate


def _control_checkpoint() -> Path:
    tier, seed = CONTROL_CELL
    path = OUTPUT / "p4_6" / "checkpoints" / f"{tier}_dt_seed{seed}.pt"
    if not path.is_file():
        pytest.skip(f"checkpoint not present in this tree: {path}")
    return path


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip(
            "the committed control digest was produced on CUDA (p4_6_training.json records "
            "NVIDIA GeForce RTX 5080 Laptop GPU); a CPU retrain would differ for a reason that "
            "is not the edit under test"
        )
    return torch.device("cuda")


@pytest.fixture(scope="module")
def retrained_control(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Retrain the control cell ONCE for both Gate-2 assertions.

    ⚠️ ``train_dt`` is called with **no** ``rtg_mode`` argument, exactly as
    ``method_tier_grid._run_train`` calls it.  A call that named the mode explicitly would survive
    mutation 2 and certify nothing.

    The batch comes from ``nortg_campaign.training_inputs``, which is the same input path the
    campaign uses.  That coupling is deliberate: if ``training_inputs`` built a different batch than
    ``_run_train`` builds, this digest would not reproduce -- so this test validates the campaign's
    data path as well as the trainer's signature.
    """
    corpus = _corpus_root()
    _control_checkpoint()
    device = _require_cuda()
    tier, seed = CONTROL_CELL
    inputs = training_inputs(tier, corpus)
    destination = tmp_path_factory.mktemp("gate2") / f"{tier}_dt_seed{seed}.pt"
    train_dt(
        inputs.batch,
        state_dim=inputs.group[0],
        n_actions=inputs.group[1],
        seed=seed,
        declared_gradient_steps=DECLARED_GRADIENT_STEPS,
        raise_to=None,
        context_length=CONTEXT_LENGTH,
        batch_size=BATCH_SIZE,
        device=device,
        checkpoint_path=destination,
        stats=inputs.stats,
        scenario_id=inputs.scenario_id,
        target_rtg=float(inputs.spec.target_rtg),
        rtg_scale=float(inputs.spec.rtg_scale),
        provenance=inputs.provenance,
    )
    return destination


def test_a_committed_dt_cell_retrains_to_its_committed_digest(retrained_control: Path) -> None:
    """⭐⭐ GATE 2.  Exact equality on the digest; a tolerance would accept the drift it detects."""
    assert canonical_digest_of(retrained_control) == CONTROL_COMMITTED_DIGEST
    config = _config_of(retrained_control)
    assert config["rtg_mode"] == "conditioned"
    assert len(config) == 9, sorted(config)


def test_the_retrained_control_payload_matches_the_committed_one_except_model_and_provenance(
    retrained_control: Path,
) -> None:
    """⭐ AMENDMENT A5.  ``target_rtg`` and ``rtg_scale`` ARE the prompt, and the digest cannot see them.

    ``canonical_digest_of`` hashes ``payload["model"]`` alone
    (``offline/method_tier_grid.py:1180-1185``), so a thread-through that perturbed either of those
    two would leave the digest green, leave the config assertion green, and change every number in
    the campaign.  **``model`` is excluded because the digest covers it; ``provenance`` is excluded
    because it legitimately differs** -- it records the seed, the timings, the device and the
    write-time git commit.  The exclusion is named here rather than left silent.
    """
    record = assert_payload_matches_committed(retrained_control, _control_checkpoint())
    assert tuple(record["compared_keys"]) == A5_COMPARED_KEYS
    assert tuple(record["excluded_keys"]) == ("model", "provenance")
    assert record["differing_keys"] == []

    # 🚨 The one declared allowance, asserted rather than tolerated.  The committed P4.6 config
    # carries 8 keys and a checkpoint written today carries 9, because P5.3a made
    # ``DTConfig.to_json_obj`` emit ``rtg_mode`` unconditionally -- a MERGED change, not this
    # task's (BRIEF_30 section 6.8; plan section 8 F4).  Pinning the allowance here is what stops
    # it from becoming a silent hole in A5.
    assert record["config_key_allowance"]["may_be_gained"] == {"rtg_mode": "conditioned"}
    committed_config = torch.load(
        _control_checkpoint(), map_location="cpu", weights_only=False
    )["config"]
    retrained_config = _config_of(retrained_control)
    assert set(retrained_config) - set(committed_config) == {"rtg_mode"}
    assert set(committed_config) - set(retrained_config) == set()
    assert len(committed_config) == 8 and len(retrained_config) == 9
    assert all(retrained_config[key] == committed_config[key] for key in committed_config)


def test_the_a5_comparison_fails_when_the_prompt_is_perturbed(
    retrained_control: Path, tmp_path: Path
) -> None:
    """The positive control for the test above: it must actually fire.

    A payload whose ``target_rtg`` is off by one is byte-identical in its weights, so the digest
    stays green -- which is exactly the hole A5 exists to close.
    """
    committed = _control_checkpoint()
    payload = torch.load(committed, map_location="cpu", weights_only=False)
    payload["target_rtg"] = float(payload["target_rtg"]) + 1.0
    perturbed = tmp_path / "perturbed.pt"
    torch.save(payload, perturbed)

    assert canonical_digest_of(perturbed) == canonical_digest_of(committed), (
        "the perturbation must be invisible to the digest, or this control proves nothing"
    )
    with pytest.raises(
        ValueError, match="payload keys differ outside model and provenance"
    ) as excinfo:
        assert_payload_matches_committed(retrained_control, perturbed)
    assert "target_rtg" in str(excinfo.value), str(excinfo.value)
