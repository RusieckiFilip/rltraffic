"""The corpus format linter -- four hard failures, each proven to fire.

A linter that only ever passes certifies nothing, so every test here builds a corpus that
must be REJECTED and asserts the specific reason, not merely a non-zero exit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from offline import corpus_format_check
from offline.corpus_format_check import check_corpus

T = 3


def _write_episode(
    run_dir: Path,
    index: int,
    *,
    version: str = "1.1",
    with_att: bool = True,
    metric_keys: tuple[str, ...] = ("only_metric",),
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "format_version": np.asarray(version),
        "ix_ids": np.asarray(["ix0"], dtype=np.str_),
        "lane_ids": np.asarray(["lane_a"], dtype=np.str_),
        "metric_keys": np.asarray(list(metric_keys), dtype=np.str_),
        "metrics": np.zeros((T + 1, len(metric_keys)), dtype=np.float32),
        "vehicle_count": np.zeros(T + 1, dtype=np.int64),
        "sim_time": np.zeros(T + 1, dtype=np.float32),
        "step": np.arange(T + 1, dtype=np.int64),
        "lane_vehicle_count": np.zeros((T + 1, 1), dtype=np.int32),
        "lane_waiting_vehicle_count": np.zeros((T + 1, 1), dtype=np.int32),
        "global_reward": np.zeros(T, dtype=np.float32),
        "episode_length": np.asarray(T, dtype=np.int64),
        "terminated": np.asarray(False, dtype=np.bool_),
        "truncated": np.asarray(True, dtype=np.bool_),
        "engine_seed": np.asarray(1000, dtype=np.int64),
        "flow_draw": np.asarray(1, dtype=np.int64),
        "ix0_state": np.zeros((T + 1, 2), dtype=np.float32),
        "ix0_avail_mask": np.ones((T + 1, 2), dtype=np.bool_),
        "ix0_current_phase": np.zeros(T + 1, dtype=np.int64),
        "ix0_time_in_phase": np.zeros(T + 1, dtype=np.float32),
        "ix0_action": np.zeros(T, dtype=np.int64),
        "ix0_local_reward": np.zeros(T, dtype=np.float32),
    }
    if with_att:
        arrays["att_per_step"] = np.zeros(T + 1, dtype=np.float32)
    path = run_dir / f"ep{index:06d}_seed1000_draw{index + 1}.npz"
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    return path


def test_a_clean_v11_corpus_passes(tmp_path: Path) -> None:
    _write_episode(tmp_path / "tier_a", 0)
    _write_episode(tmp_path / "tier_b", 0)

    report = check_corpus(tmp_path)
    assert report.ok
    assert report.n_episodes == 2
    assert corpus_format_check.main([str(tmp_path)]) == 0


def test_a_clean_v10_corpus_also_passes(tmp_path: Path) -> None:
    """The 4800-episode v1.0 corpus must stay lintable; only MIXING is a violation."""
    _write_episode(tmp_path / "tier_a", 0, version="1.0", with_att=False)

    assert check_corpus(tmp_path).ok


def test_mixed_v10_and_v11_is_rejected(tmp_path: Path) -> None:
    """The headline rejection: the two corpora must never be silently combined."""
    _write_episode(tmp_path / "tier_a", 0, version="1.1")
    _write_episode(tmp_path / "tier_b", 0, version="1.0", with_att=False)

    report = check_corpus(tmp_path)
    assert report.mixed_versions
    assert report.versions == {"1.1": 1, "1.0": 1}
    assert not report.ok
    assert corpus_format_check.main([str(tmp_path)]) == 1


def test_a_v11_file_without_att_per_step_is_rejected(tmp_path: Path) -> None:
    """Version string and contents must agree -- a version check alone cannot see this."""
    _write_episode(tmp_path / "tier_a", 0, version="1.1", with_att=False)

    report = check_corpus(tmp_path)
    assert len(report.missing_arrays) == 1
    assert "att_per_step" in report.missing_arrays[0]
    assert not report.ok


def test_an_unknown_version_is_rejected(tmp_path: Path) -> None:
    _write_episode(tmp_path / "tier_a", 0, version="2.0")

    report = check_corpus(tmp_path)
    assert len(report.unknown_version) == 1
    assert not report.ok


def test_inhomogeneous_metric_keys_are_rejected(tmp_path: Path) -> None:
    """Contract C8: one corpus, one metric set.

    The shape of the real 2026-08-06 defect, where one tier carried 2 metric keys and its
    siblings carried 3.
    """
    _write_episode(tmp_path / "tier_a", 0, metric_keys=("m1",))
    _write_episode(tmp_path / "tier_b", 0, metric_keys=("m1", "m2"))

    report = check_corpus(tmp_path)
    assert report.inhomogeneous_metric_keys
    assert set(report.metric_key_sets) == {("m1",), ("m1", "m2")}
    assert not report.ok
    assert corpus_format_check.main([str(tmp_path)]) == 1


def test_an_empty_root_is_a_usage_error_not_a_pass(tmp_path: Path) -> None:
    """Finding nothing must never report clean -- that is a mistyped path."""
    assert corpus_format_check.main([str(tmp_path)]) == 2


def test_a_missing_root_is_a_usage_error(tmp_path: Path) -> None:
    assert corpus_format_check.main([str(tmp_path / "nope")]) == 2


def test_the_real_v10_corpus_lints_clean_if_present() -> None:
    """The shipped 4800-episode corpus must satisfy its own linter.

    Skipped when datasets/ is absent so the suite still runs on a fresh clone.
    """
    root = Path("datasets")
    if not root.is_dir():
        pytest.skip("datasets/ is not present in this working tree; nothing to lint")

    report = check_corpus(root)
    assert report.versions == {"1.0": report.n_episodes}
    assert not report.inhomogeneous_metric_keys, report.metric_key_sets
    assert report.ok
