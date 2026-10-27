"""A27(c) (``BRIEF_46`` sections 3.1 and 4.1): IQL's target forwards without dropout, behind ``target_dropout``.

No on-disk format is written by this file beyond the checkpoints ``train_iql`` itself writes into ``tmp_path``
(``iql-checkpoint/1.0``); no trajectory is read or written, so no alignment convention applies.

WHAT IS PINNED, AND WHY BY HOOKS
--------------------------------
``offline_baselines.train_iql`` builds ``policy``, ``q``, ``value`` and ``q_target`` as locals.  This file replaces the
module-level name ``MLPTrunk`` -- which ``train_iql`` resolves at call time -- with a constructor that builds the REAL
module and registers a forward hook.  The hook records, at every forward: the module's role (by construction order,
which is part of the registered code because initialisation consumes the RNG in that order), ``module.training``, which
batch it was fed (``s`` or ``s'``), a digest of its parameters, and its output.  ``_awr_weight`` -- the seam that holds
the registered advantage arithmetic -- is spied so the V the advantage ACTUALLY used is observed, not inferred.

* Under ``"off"`` the three target forwards (``q_target(s)``, the advantage's ``value(s)``, ``value(s')``) run in
  ``eval()`` and the three trained forwards (``value(s)`` for the expectile loss, ``q(s)``, ``policy(s)``) in ``train()``.
* Under ``"active"`` -- the registered path -- every forward runs in ``train()`` and there is no extra forward.
* Under both, the advantage's V comes from the parameters BEFORE the V update of the same step (A27(c): the registered
  update order is kept; published IQL uses V after).
* With dropout patched to 0 the two modes must train bitwise-identical networks: only the module mode differs between
  them, so the before/after difference of A27(c) is the dropout alone.

The registered checkpoint ``output/p4_4/checkpoints/iql_seed101.pt`` is regenerated tensor for tensor under the
default by the registered P4.4 path (``offline_baselines._run_train``), on CUDA with one torch thread as P4.4 ran it;
that test skips, with its reason, where the corpus, the output tree or CUDA is absent.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch

import offline.offline_baselines as ob
from agent.OfflineBaselines import MLPTrunk, TrunkConfig, canonical_state_dict_digest

REPO = Path(__file__).resolve().parents[1]

#: Construction order inside ``train_iql`` (``policy``, ``q``, ``value``, ``q_target``).
ROLES: tuple[str, ...] = ("policy", "q", "value", "q_target")

STATE_DIM = 5
N_ACTIONS = 4
ROWS = 48
BATCH = 16


# ----------------------------------------------------------------------
# The fixture: s rows carry 0.0 in feature 0 and s' rows carry 1.0, so a forward's input names itself
# ----------------------------------------------------------------------


def _table() -> ob.TransitionTable:
    generator = torch.Generator().manual_seed(20261015)
    state = torch.randn(ROWS, STATE_DIM, generator=generator)
    next_state = torch.randn(ROWS, STATE_DIM, generator=generator)
    state[:, 0] = 0.0
    next_state[:, 0] = 1.0
    return ob.TransitionTable(
        state=state,
        next_state=next_state,
        action=torch.randint(0, N_ACTIONS, (ROWS,), generator=generator, dtype=torch.int64),
        reward=torch.randn(ROWS, generator=generator),
        stream_index=torch.zeros(ROWS, dtype=torch.int64),
        t=torch.arange(ROWS, dtype=torch.int64),
        reward_scale=1.0,
    )


def _stats() -> Any:
    from offline.dataset import NormalizationStats

    return NormalizationStats(
        stats_version="1.0",
        split="train",
        draw_ids=(1,),
        dataset_dirs=("fixture",),
        state_mean={},
        state_std={},
        row_count={},
        rtg={},
    )


def _params_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in module.parameters():
        digest.update(parameter.detach().to("cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class Forward:
    role: str
    training: bool
    batch: str
    params: str
    inputs: torch.Tensor
    output: torch.Tensor


@dataclass
class Instrument:
    """Everything the hooks and spies saw during ONE ``train_iql`` call."""

    modules: dict[str, torch.nn.Module] = field(default_factory=dict)
    forwards: list[Forward] = field(default_factory=list)
    initial_value_state: dict[str, torch.Tensor] | None = None
    loop_forwards: int | None = None
    modes_at_loop_end: dict[str, bool] | None = None
    awr_calls: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

    def record(self, role: str, module: torch.nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> None:
        inputs = args[0].detach().clone()
        first = inputs[:, 0]
        if bool(torch.all(first == 0.0)):
            batch = "s"
        elif bool(torch.all(first == 1.0)):
            batch = "s'"
        else:
            batch = "mixed"
        self.forwards.append(
            Forward(role=role, training=bool(module.training), batch=batch, params=_params_digest(module),
                    inputs=inputs, output=output.detach().clone())
        )

    def loop(self) -> list[Forward]:
        assert self.loop_forwards is not None, "the diagnostics were never reached, so the loop never ended"
        return self.forwards[: self.loop_forwards]


def _instrument(monkeypatch: pytest.MonkeyPatch, *, dropout: float | None = None) -> Instrument:
    instrument = Instrument()
    config_class = TrunkConfig if dropout is None else functools.partial(TrunkConfig, dropout=float(dropout))

    def build(config: TrunkConfig, out_dim: int) -> MLPTrunk:
        module = MLPTrunk(config, out_dim)
        role = ROLES[len(instrument.modules)]
        instrument.modules[role] = module
        if role == "value":
            instrument.initial_value_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
        module.register_forward_hook(lambda mod, args, out, _role=role: instrument.record(_role, mod, args, out))
        return module

    real_diagnostics = ob._iql_diagnostics

    def diagnostics(*args: Any, **kwargs: Any) -> Any:
        instrument.loop_forwards = len(instrument.forwards)
        instrument.modes_at_loop_end = {role: bool(m.training) for role, m in instrument.modules.items()}
        return real_diagnostics(*args, **kwargs)

    real_awr = ob._awr_weight

    def awr(q_taken: torch.Tensor, v_baseline: torch.Tensor) -> torch.Tensor:
        instrument.awr_calls.append((q_taken.detach().clone(), v_baseline.detach().clone()))
        return real_awr(q_taken, v_baseline)

    monkeypatch.setattr(ob, "MLPTrunk", build)
    monkeypatch.setattr(ob, "TrunkConfig", config_class)
    monkeypatch.setattr(ob, "_iql_diagnostics", diagnostics)
    monkeypatch.setattr(ob, "_awr_weight", awr)
    return instrument


def _train(path: Path, *, steps: int, **kwargs: Any) -> Any:
    return ob.train_iql(
        _table(),
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        seed=101,
        declared_gradient_steps=steps,
        batch_size=BATCH,
        device=torch.device("cpu"),
        checkpoint_path=path,
        stats=_stats(),
        scenario_id="fixture",
        provenance={},
        **kwargs,
    )


def _signature(forward: Forward) -> tuple[str, str, bool]:
    return (forward.role, forward.batch, forward.training)


def _per_step(forwards: list[Forward], steps: int) -> list[list[Forward]]:
    assert len(forwards) % steps == 0, f"{len(forwards)} loop forwards do not divide into {steps} equal steps"
    width = len(forwards) // steps
    return [forwards[i * width:(i + 1) * width] for i in range(steps)]


#: The registered step: five forwards, all in train mode.
ACTIVE_STEP = sorted([("q_target", "s", True), ("value", "s", True), ("value", "s'", True), ("q", "s", True),
                      ("policy", "s", True)])

#: A27(c)'s step: the three target forwards in eval, the three trained forwards in train.
OFF_STEP = sorted([("q_target", "s", False), ("value", "s", False), ("value", "s'", False),
                   ("value", "s", True), ("q", "s", True), ("policy", "s", True)])


# ----------------------------------------------------------------------
# The load-bearing pair
# ----------------------------------------------------------------------


def test_off_runs_the_three_target_forwards_in_eval_and_the_three_trained_forwards_in_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⭐ A27(c)'s change, observed at every forward of a real two-step update.  The mutation that replaces the
    implementation's ``eval()`` calls with ``train()`` must fail this test."""
    instrument = _instrument(monkeypatch)
    _train(tmp_path / "off.pt", steps=2, target_dropout="off")
    for step, forwards in enumerate(_per_step(instrument.loop(), 2)):
        assert sorted(_signature(f) for f in forwards) == OFF_STEP, f"step {step}: {[_signature(f) for f in forwards]}"
    targets = [f for f in instrument.loop() if (f.role, f.batch) in {("q_target", "s"), ("value", "s'")}]
    assert targets and not any(f.training for f in targets)
    trained = [f for f in instrument.loop() if f.role in ("q", "policy")]
    assert trained and all(f.training for f in trained)


def test_active_is_the_registered_step_five_forwards_all_in_train_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registered path: no extra forward, and every forward -- the targets included -- in ``train()``."""
    instrument = _instrument(monkeypatch)
    _train(tmp_path / "active.pt", steps=2, target_dropout="active")
    for step, forwards in enumerate(_per_step(instrument.loop(), 2)):
        assert sorted(_signature(f) for f in forwards) == ACTIVE_STEP, f"step {step}: {[_signature(f) for f in forwards]}"


@pytest.mark.parametrize("mode", ["active", "off"])
def test_every_module_is_back_in_train_mode_when_the_loop_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """``each module returns to train() immediately after`` -- a missing restore would leave ``value`` in eval."""
    instrument = _instrument(monkeypatch)
    _train(tmp_path / f"{mode}.pt", steps=2, target_dropout=mode)
    assert instrument.modes_at_loop_end == {"policy": True, "q": True, "value": True, "q_target": True}


# ----------------------------------------------------------------------
# The update order: the advantage's V is V BEFORE its update in the same step, under both modes
# ----------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["active", "off"])
def test_the_advantage_uses_v_from_before_its_update_in_the_same_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    instrument = _instrument(monkeypatch)
    _train(tmp_path / f"{mode}.pt", steps=1, target_dropout=mode)
    forwards = instrument.loop()
    assert len(instrument.awr_calls) == 1, "the advantage must be formed once per step, through _awr_weight"
    _, v_baseline = instrument.awr_calls[0]

    on_s = [f for f in forwards if f.role == "value" and f.batch == "s"]
    on_next = [f for f in forwards if f.role == "value" and f.batch == "s'"]
    assert len(on_next) == 1
    pre_update = {f.params for f in on_s}
    assert len(pre_update) == 1, "both value(s) forwards must see the same parameters"
    assert on_next[0].params not in pre_update, "value(s') must see V AFTER its update, or the V step did nothing"

    # The forward whose output the advantage used carries the pre-update parameters.
    source = [f for f in on_s if torch.equal(f.output.reshape(-1), v_baseline.reshape(-1))]
    assert len(source) == 1, "the advantage's V is not the output of any value(s) forward"
    assert source[0].params in pre_update
    assert source[0].training is (mode == "active")

    # Against V forwards taken outside the loop: the pre-update network reproduces it (in eval, under "off"), the
    # post-update network does not.
    batch = source[0].inputs
    before = MLPTrunk(instrument.modules["value"].config, 1)
    before.load_state_dict(instrument.initial_value_state)
    after = instrument.modules["value"]
    before.eval()
    after.eval()
    with torch.no_grad():
        v_before = before(batch).reshape(-1)
        v_after = after(batch).reshape(-1)
    assert not torch.equal(v_after, v_baseline.reshape(-1)), "the advantage used V after the update"
    if mode == "off":
        assert torch.equal(v_before, v_baseline.reshape(-1)), "under off the advantage's V is the eval-mode pre-update V"
    else:
        assert not torch.equal(v_before, v_baseline.reshape(-1)), (
            "under active the advantage's V is the TRAIN-mode forward, which carries dropout noise"
        )


def test_with_dropout_zero_the_two_modes_train_identical_networks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the module mode differs between ``"active"`` and ``"off"``: with no dropout there is nothing for the mode to
    change, so every trained tensor must be bitwise equal.  A change to the update order, a target or a hyper-parameter
    under ``"off"`` breaks this; a dropout-carrying configuration (below) must not satisfy it."""
    digests = {}
    for mode in ("active", "off"):
        _instrument(monkeypatch, dropout=0.0)
        record = _train(tmp_path / f"{mode}.pt", steps=4, target_dropout=mode)
        digests[mode] = record.canonical_digest
        monkeypatch.undo()
    assert digests["active"] == digests["off"]


def test_with_the_registered_dropout_the_two_modes_train_different_networks_and_active_is_the_default(
    tmp_path: Path,
) -> None:
    default = _train(tmp_path / "default.pt", steps=4)
    active = _train(tmp_path / "active.pt", steps=4, target_dropout="active")
    off = _train(tmp_path / "off.pt", steps=4, target_dropout="off")
    assert default.canonical_digest == active.canonical_digest, "the default must be the registered path"
    assert off.canonical_digest != active.canonical_digest, "dropout 0.1 in the targets must change the result"


# ----------------------------------------------------------------------
# The record and the refusal
# ----------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["active", "off"])
def test_the_flag_is_recorded_in_the_provenance_and_in_the_diagnostics(tmp_path: Path, mode: str) -> None:
    path = tmp_path / f"{mode}.pt"
    record = _train(path, steps=2, target_dropout=mode)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["provenance"]["target_dropout"] == mode
    assert payload["provenance"]["diagnostics"]["target_dropout"] == mode
    assert record.diagnostics["target_dropout"] == mode


def test_an_unknown_value_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    path = tmp_path / "never.pt"
    with pytest.raises(ValueError, match="target_dropout"):
        _train(path, steps=2, target_dropout="partly")
    assert not path.exists()
    assert sorted(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------
# The registered checkpoint, regenerated under the default (gated)
# ----------------------------------------------------------------------


def _corpus_root() -> Path:
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO / "datasets_v11"
    if not candidate.is_dir():
        pytest.skip(f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to regenerate P4.4's IQL")
    return candidate


def _output_root() -> Path:
    env_value = os.environ.get("RLTRAFFIC_OUTPUT_ROOT")
    candidate = Path(env_value) if env_value else REPO / "output"
    if not (candidate / "p4_4" / "checkpoints" / "iql_seed101.pt").is_file():
        pytest.skip(f"no P4.4 IQL checkpoint under {candidate}: set RLTRAFFIC_OUTPUT_ROOT to the main tree's output/")
    return candidate


def test_the_registered_p4_4_iql_seed101_checkpoint_regenerates_tensor_for_tensor_under_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⭐ A27(a): the default reproduces the registered behaviour byte for byte, shown on a registered checkpoint.

    P4.4's own CLI path (``offline_baselines._run_train``) with NO new argument, the seeds narrowed to 101 by
    monkeypatch, on CUDA with one torch thread -- the regime ``iql_seed101.pt``'s provenance records.  Every tensor of
    ``model`` is compared under ``torch.equal`` and the first differing one is named; there is no tolerance.
    """
    corpus = _corpus_root()
    registered = _output_root() / "p4_4" / "checkpoints" / "iql_seed101.pt"
    if not torch.cuda.is_available():
        pytest.skip("iql_seed101.pt was trained on CUDA (its provenance says so); a CPU retrain differs for a reason "
                    "that is not the change under test")
    payload = torch.load(registered, map_location="cpu", weights_only=False)
    provenance = payload["provenance"]
    assert provenance["device"] == "cuda" and provenance["batch_size"] == ob.IQL_BATCH_TRANSITIONS
    assert provenance["runtime"]["torch_num_threads"] == 1
    dirs = [str(corpus / Path(name).name) for name in provenance["dataset_dirs"]]

    monkeypatch.setattr(ob, "TRAINING_SEEDS", (101,))
    previous = ob.pin_torch_threads(1)
    try:
        args = argparse.Namespace(dataset_dir=dirs, steps=ob.DECLARED_GRADIENT_STEPS, methods="iql", log_every=0,
                                  device="cuda", checkpoint_dir=str(tmp_path / "checkpoints"))
        (tmp_path / "out").mkdir()
        assert ob._run_train(args, tmp_path / "out") == 0
    finally:
        ob.pin_torch_threads(previous)

    regenerated = torch.load(tmp_path / "checkpoints" / "iql_seed101.pt", map_location="cpu", weights_only=False)
    assert sorted(regenerated["model"]) == sorted(payload["model"])
    differing = [key for key in sorted(payload["model"]) if not torch.equal(regenerated["model"][key], payload["model"][key])]
    assert differing == [], f"first differing tensor: {differing[:1]} ({len(differing)} of {len(payload['model'])})"
    assert canonical_state_dict_digest(regenerated["model"]) == payload["canonical_digest"]
    assert regenerated["provenance"]["target_dropout"] == "active"
