"""A27(c)'s D4RL re-run (``BRIEF_46`` 3.6): ``calibration/d4rl_run.py train --iql-target-dropout off``.

No simulator and no MuJoCo: the runner is imported with the two calibration-only modules (``h5py``, ``mujoco``) stubbed
in ``sys.modules`` for these tests alone -- nothing here imports them, so ``tests/test_d4rl_calibration.py``'s isolation
gate (which scans every tracked module outside ``calibration/`` and ``tests/``) is untouched.  Training and the rollout
are replaced by spies; the arithmetic of the registered contrasts is the runner's own ``between_arm_contrasts``, run on
the registered artifact ``docs/data/p8_3_d4rl_calibration.json`` (format ``p8.3-d4rl-calibration/1.0``).  No
trajectory is read, so no alignment convention applies.

* ``"active"`` (the default) is the registered P8.3 run: arm ``iql``, no new keyword handed to ``train_iql``, no new key
  in the output.
* ``"off"`` re-runs the IQL arm ONLY, as ``iql_t``, with ``target_dropout="off"``; it never overwrites an output or a
  checkpoint; ``--compare-with`` adds the registered three arms' cells and recomputes every contrast beside the new arm
  -- the registered ``bc_vs_iql`` (-542.05 [-1668.10, +584.01]) among them, reproduced exactly.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
REGISTERED = REPO / "docs" / "data" / "p8_3_d4rl_calibration.json"


@pytest.fixture()
def runner(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in ("h5py", "mujoco"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__version__ = "0+test-stub"  # environment_provenance() records every library's version
            monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.delitem(sys.modules, "calibration.d4rl_run", raising=False)
    import calibration.d4rl_run as module

    return module


def _toy() -> dict[str, np.ndarray]:
    from tests.test_d4rl_calibration import _toy_dataset

    return _toy_dataset()


class _TrainSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, batch: Any, **kwargs: Any) -> Any:
        from offline.offline_baselines import TrainRecord

        self.calls.append(dict(kwargs))
        path = Path(kwargs["checkpoint_path"])
        torch.save({"model": {"w": torch.zeros(1)}}, path)
        return TrainRecord(method="iql", seed=int(kwargs["seed"]), gradient_steps=int(kwargs["declared_gradient_steps"]),
                           declared_gradient_steps=int(kwargs["declared_gradient_steps"]), losses=(1.0,),
                           window_means=(1.0,), plateaued=True, checkpoint_path=str(path), canonical_digest="c" * 64,
                           file_sha256="f" * 64, seconds=12.5,
                           diagnostics={"target_dropout": kwargs.get("target_dropout", "active")})


def _install(runner: Any, monkeypatch: pytest.MonkeyPatch) -> _TrainSpy:
    spy = _TrainSpy()
    monkeypatch.setattr(runner, "train_iql", spy)
    monkeypatch.setattr(runner, "_load_policy", lambda checkpoint, device: (None, {}))
    monkeypatch.setattr(runner, "evaluate_policy",
                        lambda model, book, stats, *, episodes, seed_base, device: [float(seed_base)] * episodes)
    return spy


def _run(runner: Any, tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    from calibration.d4rl_adapter import ActionCodebook, episode_spans

    data = _toy()
    spans = episode_spans(data["timeouts"], data["terminals"])
    book = ActionCodebook.fit(data["actions"], 4, subsample=16)
    return runner.train_and_evaluate(data, spans, book, arms=("iql",), seeds=(101, 202), steps=2, out_dir=tmp_path,
                                     device=torch.device("cpu"), log_every=0, **kwargs)


REGISTERED_IQL_KEYWORDS = {"state_dim", "n_actions", "seed", "declared_gradient_steps", "batch_size", "device",
                           "checkpoint_path", "stats", "scenario_id", "provenance", "log_every"}


def test_the_default_is_the_registered_run(runner: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _install(runner, monkeypatch)
    result = _run(runner, tmp_path)
    assert [set(call) for call in spy.calls] == [REGISTERED_IQL_KEYWORDS] * 2
    assert [cell["arm"] for cell in result["cells"]] == ["iql", "iql"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["iql_k4_seed101.pt", "iql_k4_seed202.pt"]
    assert set(result) == {"cells", "summary", "contrasts", "published_ordering_check", "top_decile_episodes",
                           "reporting_rule"}


def test_off_reruns_iql_as_iql_t_with_the_flag(runner: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _install(runner, monkeypatch)
    result = _run(runner, tmp_path, target_dropout="off")
    assert [call["target_dropout"] for call in spy.calls] == ["off", "off"]
    assert [cell["arm"] for cell in result["cells"]] == ["iql_t", "iql_t"]
    assert all(cell["diagnostics"]["target_dropout"] == "off" for cell in result["cells"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["iql_t_k4_seed101.pt", "iql_t_k4_seed202.pt"]
    assert list(result["summary"]) == ["iql_t"]
    assert result["target_dropout"] == "off"


def test_off_refuses_any_arm_but_iql_and_any_existing_checkpoint(
    runner: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from calibration.d4rl_adapter import ActionCodebook, episode_spans

    spy = _install(runner, monkeypatch)
    data = _toy()
    spans = episode_spans(data["timeouts"], data["terminals"])
    book = ActionCodebook.fit(data["actions"], 4, subsample=16)
    with pytest.raises(ValueError, match="IQL arm only"):
        runner.train_and_evaluate(data, spans, book, arms=("bc", "iql"), seeds=(101,), steps=2, out_dir=tmp_path,
                                  device=torch.device("cpu"), log_every=0, target_dropout="off")
    (tmp_path / "iql_t_k4_seed202.pt").write_bytes(b"earlier")
    with pytest.raises(FileExistsError, match="iql_t_k4_seed202.pt"):
        runner.train_and_evaluate(data, spans, book, arms=("iql",), seeds=(101, 202), steps=2, out_dir=tmp_path,
                                  device=torch.device("cpu"), log_every=0, target_dropout="off")
    assert spy.calls == [] and (tmp_path / "iql_t_k4_seed202.pt").read_bytes() == b"earlier"


def test_the_registered_contrasts_recompute_exactly_from_the_registered_cells(runner: Any) -> None:
    """The anchor the paper's sentence quotes: bc - iql = -542.05 [-1668.10, +584.01]."""
    artifact = json.loads(REGISTERED.read_text(encoding="utf-8"))
    recomputed = runner.between_arm_contrasts(artifact["cells"], ("bc", "bc_top10", "iql"))
    assert recomputed == artifact["contrasts"]
    bc_iql = recomputed["bc_vs_iql"]
    assert (round(bc_iql["mean_difference_return"], 2), round(bc_iql["ci95_low_return"], 2),
            round(bc_iql["ci95_high_return"], 2)) == (-542.05, -1668.10, 584.01)


def test_compare_with_puts_the_new_arm_beside_the_registered_three(runner: Any) -> None:
    """The registered cells keep their numbers; the new arm's contrasts are the registered function's; the per-seed
    differences are recomputed here by hand from the episode returns (an independent route)."""
    from offline.dt_gate import mean_ci95

    artifact = json.loads(REGISTERED.read_text(encoding="utf-8"))
    new_cells = [
        {"arm": "iql_t", "seed": cell["seed"], "episode_returns": [r + 100.0 for r in cell["episode_returns"]]}
        for cell in artifact["cells"] if cell["arm"] == "iql"
    ]
    block = runner.compare_with_registered(artifact, new_cells)
    assert block["arms"] == ["bc", "bc_top10", "iql", "iql_t"]
    contrasts = block["contrasts"]
    for key in ("bc_vs_bc_top10", "bc_vs_iql", "bc_top10_vs_iql"):
        assert contrasts[key] == artifact["contrasts"][key], key
    per_seed_bc = {c["seed"]: float(np.mean(c["episode_returns"])) for c in artifact["cells"] if c["arm"] == "bc"}
    per_seed_new = {c["seed"]: float(np.mean(c["episode_returns"])) for c in new_cells}
    differences = [per_seed_bc[s] - per_seed_new[s] for s in sorted(per_seed_bc)]
    assert contrasts["bc_vs_iql_t"]["per_seed_difference"] == differences
    assert contrasts["bc_vs_iql_t"]["mean_difference_return"] == mean_ci95(differences).mean
    assert contrasts["iql_vs_iql_t"]["per_seed_difference"] == [-100.0] * 5


def test_compare_with_refuses_a_registered_artifact_from_another_protocol(runner: Any) -> None:
    artifact = json.loads(REGISTERED.read_text(encoding="utf-8"))
    new_cells = [{"arm": "iql_t", "seed": s, "episode_returns": [0.0] * 10} for s in (101, 202, 303, 404)]
    with pytest.raises(ValueError, match="seeds"):
        runner.compare_with_registered(artifact, new_cells)
    short = [{"arm": "iql_t", "seed": s, "episode_returns": [0.0] * 9} for s in (101, 202, 303, 404, 505)]
    with pytest.raises(ValueError, match="episodes"):
        runner.compare_with_registered(artifact, short)


def test_the_cli_carries_both_flags_and_defaults_to_the_registered_run(runner: Any) -> None:
    default = runner.build_parser().parse_args(["train", "--dataset", "d.hdf5", "--out", "o"])
    assert default.iql_target_dropout == "active" and default.compare_with is None
    flagged = runner.build_parser().parse_args(["train", "--dataset", "d.hdf5", "--out", "o", "--iql-target-dropout",
                                                "off", "--compare-with", str(REGISTERED)])
    assert flagged.iql_target_dropout == "off" and flagged.compare_with == str(REGISTERED)


def test_compare_with_refuses_registered_cells_that_do_not_reproduce_their_own_contrasts(runner: Any) -> None:
    """A registered artifact whose cells were edited no longer yields its committed contrasts: nothing is compared."""
    artifact = json.loads(REGISTERED.read_text(encoding="utf-8"))
    tampered = json.loads(json.dumps(artifact))
    tampered["cells"][0]["episode_returns"][0] += 1.0
    new_cells = [{"arm": "iql_t", "seed": s, "episode_returns": [0.0] * 10} for s in (101, 202, 303, 404, 505)]
    assert runner.compare_with_registered(artifact, new_cells)["registered_contrasts_reproduced"] is True
    with pytest.raises(ValueError, match="do not recompute"):
        runner.compare_with_registered(tampered, new_cells)
