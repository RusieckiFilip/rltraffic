"""The §3.1 sanity anchors must be reproducible from committed data.

``docs/PROJECT_PLAN.md`` §3.1 is the anchor every later phase's numbers are compared
against (plan §7). Its six average-travel-time figures used to live only in
``output/experiments/p0_baselines/``, which is gitignored -- unreproducible from a fresh
clone. P0.6 committed that run's raw output to ``docs/data/p0_baselines/results.json``;
this test reads it back and proves it still reproduces the six numbers.

Two independent routes, per the double-compute rule:

* the stored aggregate ``aggregated[env][policy]["average_travel_time"]["mean"]``, and
* a fresh ``np.mean`` over the per-seed values in ``cells`` (a different code path than the
  one that wrote the aggregate).

They must be **exactly** equal (both are deterministic functions of the same immutable
file), and the stored mean rounded to 2 dp must equal the plan §3.1 value. Unlike most
tests here this one is expected to PASS on the first run: a first-run failure means the
committed file is not the run §3.1 describes, which is a bigger problem than a missing file
-- so it should be reported, not smoothed over.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "docs" / "data" / "p0_baselines" / "results.json"

# The six anchors, verbatim from docs/PROJECT_PLAN.md §3.1 (2 dp).
PLAN_3_1_TRAVEL_TIME: dict[tuple[str, str], float] = {
    ("cf_hz1x1", "MaxPressure"): 160.56,
    ("cf_hz1x1", "Random"): 307.53,
    ("cf_hz1x1", "mappo"): 197.91,
    ("cf_grid4x4", "MaxPressure"): 141.65,
    ("cf_grid4x4", "Random"): 207.26,
    ("cf_grid4x4", "mappo"): 632.95,
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _per_seed_travel_times(results: dict, env_id: str, policy: str) -> list[float]:
    """Every per-seed average_travel_time for one (env, policy), read from ``cells``.

    This is the independent route: it never touches ``aggregated``.
    """
    values: list[float] = []
    for cell in results["cells"]:
        if cell["env_id"] == env_id and cell["status"] == "ok":
            values.append(cell["policies"][policy]["metrics"]["average_travel_time"])
    return values


def _assert_anchor(results: dict, env_id: str, policy: str, expected_2dp: float) -> None:
    """Double-compute one anchor and check it against plan §3.1. Raises on mismatch."""
    per_seed = _per_seed_travel_times(results, env_id, policy)
    assert len(per_seed) == 3, (
        f"expected 3 seeds for {env_id}/{policy}, found {len(per_seed)}"
    )
    recomputed = float(np.mean(per_seed))
    stored = results["aggregated"][env_id][policy]["average_travel_time"]["mean"]
    assert recomputed == stored, (
        f"double-compute mismatch for {env_id}/{policy}: "
        f"aggregated mean {stored!r} != np.mean(cells) {recomputed!r}"
    )
    assert round(stored, 2) == expected_2dp, (
        f"anchor drift for {env_id}/{policy}: committed {round(stored, 2)} != plan §3.1 {expected_2dp}"
    )


@pytest.mark.parametrize(
    "env_id, policy, expected",
    [(env, pol, val) for (env, pol), val in PLAN_3_1_TRAVEL_TIME.items()],
    ids=[f"{env}-{pol}" for (env, pol) in PLAN_3_1_TRAVEL_TIME],
)
def test_p0_anchor_reproduces_plan_3_1(env_id: str, policy: str, expected: float) -> None:
    """Each §3.1 travel-time anchor reproduces from the committed baseline data."""
    _assert_anchor(_load(RESULTS_PATH), env_id, policy, expected)


def test_committed_results_has_all_six_cells_ok() -> None:
    """Guard against a truncated commit: 2 envs x 3 seeds, every cell ``ok``."""
    results = _load(RESULTS_PATH)
    ok_cells = [c for c in results["cells"] if c["status"] == "ok"]
    assert len(ok_cells) == 6, f"expected 6 ok cells, found {len(ok_cells)}"


def test_anchor_check_is_load_bearing(tmp_path: Path) -> None:
    """Mutation: corrupting one stored mean must make the double-compute fail.

    Proves the equality assertion actually guards the number rather than passing vacuously.
    """
    results = _load(RESULTS_PATH)
    results["aggregated"]["cf_hz1x1"]["mappo"]["average_travel_time"]["mean"] += 1.0
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(results), encoding="utf-8")
    with pytest.raises(AssertionError, match="double-compute mismatch"):
        _assert_anchor(_load(tampered), "cf_hz1x1", "mappo", 197.91)
