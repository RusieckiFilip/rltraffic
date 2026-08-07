"""The C8 checkpoint migration -- which is ours, unreviewed, and writes to real data.

A migration that records the field but records it **wrong** would satisfy the load-time
presence check while stamping nonsense onto 60 checkpoints, turning the guard into a
rubber stamp.  So the migration is tested as adversarially as the guard it feeds.

No simulator: the env-side seam is an ``env_factory``, and the fakes it returns expose
the same two attributes a real ``BaseMetrics`` does.  The seam is deliberately placed at
**env construction**, not at key derivation, so every test still exercises the real
``agent.MAPPOAgent.env_global_metric_keys`` -- the one code path the patch also uses.
Moving the seam one step later would let this suite pass against a derivation the
production path never runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

import offline.migrate_mappo_checkpoints as migrate
from offline.migrate_mappo_checkpoints import (
    BACKUP_DIRNAME,
    METRIC_KEYS_FIELD,
    MigrationError,
    apply_plan,
    build_plan,
    checkpoint_env_id,
    checkpoint_run_name,
    discover_checkpoints,
)

ALPHA = "metric_alpha"
BETA = "metric_beta"


# ----------------------------------------------------------------------
# Doubles
# ----------------------------------------------------------------------


@dataclass
class _Spec:
    compute: Any


class _FakeMetrics:
    def __init__(self, names: tuple[str, ...]) -> None:
        self.REGISTERED = {n: _Spec(compute=lambda _s: 0.0) for n in names}
        self._requested = list(names)

    @property
    def requested(self) -> list[str]:
        return list(self._requested)


class _FakeEnv:
    def __init__(self, names: tuple[str, ...]) -> None:
        self.metrics = _FakeMetrics(names)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _factory(mapping: dict[tuple[str, str], tuple[str, ...]]):
    """Env factory keyed by (run_name, env_id), recording what it was asked for."""
    seen: list[tuple[str, str]] = []

    def make(run_name: str, env_id: str) -> _FakeEnv:
        seen.append((run_name, env_id))
        if (run_name, env_id) not in mapping:
            raise MigrationError(f"no config for {run_name}/{env_id}")
        return _FakeEnv(mapping[(run_name, env_id)])

    make.seen = seen  # type: ignore[attr-defined]
    return make


def _write_checkpoint(
    path: Path,
    *,
    global_feature_dim: int,
    metric_keys: Any = "__absent__",
    steps_done: int = 21600,
) -> Path:
    """A checkpoint shaped like the real ones: keys steps_done + learner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "steps_done": steps_done,
        "learner": {
            "local_state_dims": [25],
            "global_feature_dim": global_feature_dim,
            "actors": {},
            "critic": {},
            "optimizer": {},
        },
    }
    if metric_keys != "__absent__":
        payload[METRIC_KEYS_FIELD] = metric_keys
    torch.save(payload, str(path))
    return path


def _ckpt_tree(root: Path) -> list[Path]:
    """Mirrors the real layout, including the outer != inner directory trap."""
    return [
        _write_checkpoint(root / "run_060" / "cf_hz1x1__mappo__seed101.pt",
                          global_feature_dim=3),
        _write_checkpoint(root / "run_060" / "cf_grid4x4__mappo__seed202.pt",
                          global_feature_dim=3),
        # Outer directory names the 500 run; the INNER one is the real run name.
        _write_checkpoint(root / "run_500" / "run_1000" / "cf_hz1x1__mappo__seed303.pt",
                          global_feature_dim=3),
    ]


# ----------------------------------------------------------------------
# 1. Path decoding -- the outer/inner directory trap
# ----------------------------------------------------------------------


def test_run_name_is_the_parent_directory_not_the_grandparent(tmp_path: Path) -> None:
    """p2_1_mappo_nominal_1000 writes into the _500 directory; the INNER one wins."""
    path = tmp_path / "run_500" / "run_1000" / "cf_hz1x1__mappo__seed303.pt"
    assert checkpoint_run_name(path) == "run_1000"


def test_env_id_comes_from_the_filename_prefix(tmp_path: Path) -> None:
    assert checkpoint_env_id(tmp_path / "r" / "cf_grid4x4__mappo__seed202.pt") == (
        "cf_grid4x4"
    )


def test_discover_finds_every_checkpoint(tmp_path: Path) -> None:
    written = _ckpt_tree(tmp_path)
    assert sorted(discover_checkpoints(tmp_path)) == sorted(written)


# ----------------------------------------------------------------------
# 2. Derivation
# ----------------------------------------------------------------------


def test_keys_are_derived_per_checkpoint_from_its_own_run(tmp_path: Path) -> None:
    """Each checkpoint is resolved against the run that trained IT, not a shared one."""
    _ckpt_tree(tmp_path)
    factory = _factory({
        ("run_060", "cf_hz1x1"): (ALPHA,),
        ("run_060", "cf_grid4x4"): (ALPHA,),
        ("run_1000", "cf_hz1x1"): (BETA,),
    })
    plans = build_plan(discover_checkpoints(tmp_path), factory)

    by_path = {p.path.name + "/" + p.run_name: p for p in plans}
    assert by_path["cf_hz1x1__mappo__seed303.pt/run_1000"].metric_keys == (BETA,)
    assert by_path["cf_hz1x1__mappo__seed101.pt/run_060"].metric_keys == (ALPHA,)
    # Every (run, env) pair was actually looked up -- no silent reuse of one answer.
    assert set(factory.seen) == {
        ("run_060", "cf_hz1x1"), ("run_060", "cf_grid4x4"), ("run_1000", "cf_hz1x1"),
    }


def test_module_source_contains_no_metric_name_literal() -> None:
    """Derivation, not transcription -- asserted against the file, not the docstring.

    A hardcoded metric name is the failure mode that self-consistency cannot catch:
    the count would be right and only a later collection would break.
    """
    source = Path(migrate.__file__).read_text(encoding="utf-8")
    # The real corpus metric, and any other registry-looking name.
    assert "number_of_all_halting_vehicles" not in source
    assert not re.search(r"average_travel_time|queue_length|pressure", source)


def test_envs_are_closed_after_derivation(tmp_path: Path) -> None:
    """60 checkpoints must not leave 60 engines open."""
    _write_checkpoint(tmp_path / "run_060" / "cf_hz1x1__mappo__seed101.pt",
                      global_feature_dim=3)
    created: list[_FakeEnv] = []

    def make(run_name: str, env_id: str) -> _FakeEnv:
        env = _FakeEnv((ALPHA,))
        created.append(env)
        return env

    build_plan(discover_checkpoints(tmp_path), make)
    assert created and all(env.closed for env in created)


# ----------------------------------------------------------------------
# 3. Self-consistency against the checkpoint's own witness
# ----------------------------------------------------------------------


def test_wrong_key_count_aborts_against_global_feature_dim(tmp_path: Path) -> None:
    """global_feature_dim == 2 + len(keys); a mismatch is a hard stop."""
    _write_checkpoint(tmp_path / "run_060" / "cf_hz1x1__mappo__seed101.pt",
                      global_feature_dim=3)
    factory = _factory({("run_060", "cf_hz1x1"): (ALPHA, BETA)})  # 2 keys, needs 1

    with pytest.raises(MigrationError, match="global_feature_dim"):
        build_plan(discover_checkpoints(tmp_path), factory)


def test_correct_key_count_passes_the_witness_check(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "run_060" / "cf_hz1x1__mappo__seed101.pt",
                      global_feature_dim=4)
    factory = _factory({("run_060", "cf_hz1x1"): (ALPHA, BETA)})
    plans = build_plan(discover_checkpoints(tmp_path), factory)
    assert plans[0].metric_keys == (ALPHA, BETA)


def test_a_checkpoint_with_conflicting_keys_aborts(tmp_path: Path) -> None:
    """Never silently overwrite a key set that disagrees with what we derived."""
    _write_checkpoint(tmp_path / "run_060" / "cf_hz1x1__mappo__seed101.pt",
                      global_feature_dim=3, metric_keys=[BETA])
    factory = _factory({("run_060", "cf_hz1x1"): (ALPHA,)})

    with pytest.raises(MigrationError, match="already records"):
        build_plan(discover_checkpoints(tmp_path), factory)


def test_an_already_migrated_checkpoint_is_a_no_op(tmp_path: Path) -> None:
    """Re-running the migration must be safe and must not rewrite files."""
    path = _write_checkpoint(tmp_path / "run_060" / "cf_hz1x1__mappo__seed101.pt",
                             global_feature_dim=3, metric_keys=[ALPHA])
    before = path.read_bytes()
    factory = _factory({("run_060", "cf_hz1x1"): (ALPHA,)})

    plans = build_plan(discover_checkpoints(tmp_path), factory)
    assert plans[0].already_correct is True
    assert apply_plan(plans) == 0
    assert path.read_bytes() == before


# ----------------------------------------------------------------------
# 4. The mutation barrier
# ----------------------------------------------------------------------


def test_nothing_is_written_when_any_checkpoint_fails_validation(tmp_path: Path) -> None:
    """One bad checkpoint must leave ALL of them untouched, not the first N rewritten."""
    good = _write_checkpoint(tmp_path / "run_060" / "cf_hz1x1__mappo__seed101.pt",
                             global_feature_dim=3)
    bad = _write_checkpoint(tmp_path / "run_060" / "cf_grid4x4__mappo__seed202.pt",
                            global_feature_dim=9)  # inconsistent with a 1-key set
    before = {p: p.read_bytes() for p in (good, bad)}
    factory = _factory({
        ("run_060", "cf_hz1x1"): (ALPHA,), ("run_060", "cf_grid4x4"): (ALPHA,),
    })

    with pytest.raises(MigrationError, match="global_feature_dim"):
        build_plan(discover_checkpoints(tmp_path), factory)

    for path, content in before.items():
        assert path.read_bytes() == content


# ----------------------------------------------------------------------
# 5. Round-trip
# ----------------------------------------------------------------------


def test_applied_keys_survive_a_round_trip(tmp_path: Path) -> None:
    path = _write_checkpoint(tmp_path / "run_060" / "cf_hz1x1__mappo__seed101.pt",
                             global_feature_dim=3)
    factory = _factory({("run_060", "cf_hz1x1"): (ALPHA,)})

    plans = build_plan(discover_checkpoints(tmp_path), factory)
    assert apply_plan(plans) == 1

    reloaded = torch.load(str(path), map_location="cpu", weights_only=False)
    assert reloaded[METRIC_KEYS_FIELD] == [ALPHA]
    # Everything else is preserved exactly -- this is additive, like format v1.1.
    assert reloaded["steps_done"] == 21600
    assert reloaded["learner"]["global_feature_dim"] == 3
    assert reloaded["learner"]["local_state_dims"] == [25]


# ----------------------------------------------------------------------
# 6. The backup gate
# ----------------------------------------------------------------------


def test_apply_refuses_without_the_backup_directory(tmp_path: Path) -> None:
    """The only irreversible step in P2.6 is gated mechanically, not by a runbook."""
    root = tmp_path / "checkpoints"
    _write_checkpoint(root / "run_060" / "cf_hz1x1__mappo__seed101.pt",
                      global_feature_dim=3)

    code = migrate.main([
        "--checkpoint-root", str(root), "--apply", "--config-dir", str(tmp_path),
    ])
    assert code != 0
    payload = torch.load(
        str(root / "run_060" / "cf_hz1x1__mappo__seed101.pt"),
        map_location="cpu", weights_only=False,
    )
    assert METRIC_KEYS_FIELD not in payload, "wrote despite refusing"


def test_dry_run_is_the_default_and_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    path = _write_checkpoint(root / "run_060" / "cf_hz1x1__mappo__seed101.pt",
                             global_feature_dim=3)
    before = path.read_bytes()

    code = migrate.main(
        ["--checkpoint-root", str(root), "--config-dir", str(tmp_path)],
        env_factory=_factory({("run_060", "cf_hz1x1"): (ALPHA,)}),
    )
    assert code == 0
    assert path.read_bytes() == before


def test_backup_dir_name_is_the_one_the_brief_specifies() -> None:
    assert BACKUP_DIRNAME == "checkpoints.pre_c8_migration"
