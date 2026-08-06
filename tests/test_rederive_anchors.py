"""The anchor re-derivation harness must never touch the historical record.

``docs/data/p0_baselines/`` is the immutable 2026-07-09 validation target; the horizon re-run is
checked *against* it. The filesystem-mutation barrier requires the harness to refuse writing there
before any file is created. This tests the guard mechanically, without a simulator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from offline.rederive_anchors import HISTORICAL_DIR, assert_safe_out_dir


def test_guard_refuses_the_historical_dir() -> None:
    with pytest.raises(ValueError, match="historical"):
        assert_safe_out_dir(HISTORICAL_DIR)


def test_guard_refuses_a_path_that_resolves_into_the_historical_dir() -> None:
    # A '..' detour that normalises back onto the historical dir must still be rejected.
    sneaky = HISTORICAL_DIR / "sub" / ".."
    with pytest.raises(ValueError, match="historical"):
        assert_safe_out_dir(sneaky)


def test_guard_allows_a_fresh_dir(tmp_path: Path) -> None:
    # Does not raise; returns None.
    assert assert_safe_out_dir(tmp_path / "p0_baselines_horizon") is None
