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


# ----------------------------------------------------------------------
# P8.0: the horizon anchors (prereg A1). The re-run records BOTH att_horizon
# (the registered primary metric) and att_running_mean (the legacy runner.py
# quantity). Baselines touch no torch and MUST reproduce the committed anchors
# exactly; MAPPO may cross the N2 float-reduction boundary and is recorded, not
# asserted. See docs/data/p0_baselines_horizon/PROVENANCE.md.
# ----------------------------------------------------------------------

HORIZON_RESULTS_PATH = (
    REPO_ROOT / "docs" / "data" / "p0_baselines_horizon" / "results.json"
)
# Labels that never touch torch (per registry BASELINE_LABELS) -> exact reproduction.
BASELINE_POLICIES = ("MaxPressure", "Random")
HORIZON_QUANTITIES = ("att_horizon", "att_running_mean")
ENVS = ("cf_hz1x1", "cf_grid4x4")


def _load_horizon() -> dict:
    assert HORIZON_RESULTS_PATH.exists(), (
        f"horizon anchor data missing: {HORIZON_RESULTS_PATH}. "
        "Generate it with offline/rederive_anchors.py (P8.0)."
    )
    return json.loads(HORIZON_RESULTS_PATH.read_text(encoding="utf-8"))


def _horizon_per_seed_list(results: dict, env_id: str, policy: str, quantity: str) -> list[float]:
    """Per-seed *quantity* for one (env, policy), in file/cell order (never touches aggregated).

    Cell order is preserved so the recomputed mean sums in the same order the harness used,
    keeping the double-compute an exact bitwise check rather than an approximate one.
    """
    values: list[float] = []
    for cell in results["cells"]:
        if cell["env_id"] == env_id and cell["status"] == "ok":
            payload = cell["policies"].get(policy)
            if payload is not None:
                values.append(payload[quantity])
    return values


def _horizon_per_seed_map(results: dict, env_id: str, policy: str, quantity: str) -> dict[int, float]:
    """Per-seed *quantity* keyed by seed, for cross-file matching by seed."""
    out: dict[int, float] = {}
    for cell in results["cells"]:
        if cell["env_id"] == env_id and cell["status"] == "ok":
            payload = cell["policies"].get(policy)
            if payload is not None:
                out[int(cell["seed"])] = payload[quantity]
    return out


def _assert_horizon_double_compute(results: dict, env_id: str, policy: str, quantity: str) -> None:
    """Aggregated mean must equal ``np.mean(cells)`` exactly. Raises on mismatch."""
    per_seed = _horizon_per_seed_list(results, env_id, policy, quantity)
    assert len(per_seed) == 3, (
        f"expected 3 seeds for {env_id}/{policy}/{quantity}, found {len(per_seed)}"
    )
    recomputed = float(np.mean(per_seed))
    stored = results["aggregated"][env_id][policy][quantity]["mean"]
    assert recomputed == stored, (
        f"double-compute mismatch for {env_id}/{policy}/{quantity}: "
        f"aggregated {stored!r} != np.mean(cells) {recomputed!r}"
    )


def _legacy_per_seed_map(results: dict, env_id: str, policy: str) -> dict[int, float]:
    """Per-seed committed ``average_travel_time`` (the running mean) keyed by seed."""
    out: dict[int, float] = {}
    for cell in results["cells"]:
        if cell["env_id"] == env_id and cell["status"] == "ok":
            out[int(cell["seed"])] = cell["policies"][policy]["metrics"]["average_travel_time"]
    return out


@pytest.mark.parametrize(
    "env_id, policy, expected",
    [(env, pol, val) for (env, pol), val in PLAN_3_1_TRAVEL_TIME.items()],
    ids=[f"{env}-{pol}" for (env, pol) in PLAN_3_1_TRAVEL_TIME],
)
def test_p0_anchor_reproduces_plan_3_1(env_id: str, policy: str, expected: float) -> None:
    """Each §3.1 travel-time anchor reproduces from the committed baseline data."""
    _assert_anchor(_load(RESULTS_PATH), env_id, policy, expected)


def test_horizon_file_has_both_baselines_for_both_envs() -> None:
    """Guard against a truncated commit: both baselines, both envs, n==3, both quantities."""
    results = _load_horizon()
    assert results.get("quantities") == list(HORIZON_QUANTITIES)
    for env_id in ENVS:
        for policy in BASELINE_POLICIES:
            for quantity in HORIZON_QUANTITIES:
                entry = results["aggregated"][env_id][policy][quantity]
                assert entry["n"] == 3, f"{env_id}/{policy}/{quantity}: n={entry['n']}, want 3"


@pytest.mark.parametrize("env_id", ENVS)
@pytest.mark.parametrize("policy", BASELINE_POLICIES)
@pytest.mark.parametrize("quantity", HORIZON_QUANTITIES)
def test_horizon_double_compute(env_id: str, policy: str, quantity: str) -> None:
    """Both quantities: aggregated mean == independently recomputed np.mean(cells), exactly."""
    _assert_horizon_double_compute(_load_horizon(), env_id, policy, quantity)


@pytest.mark.parametrize("env_id", ENVS)
@pytest.mark.parametrize("policy", BASELINE_POLICIES)
def test_horizon_running_mean_reproduces_committed_anchor(env_id: str, policy: str) -> None:
    """The re-run's att_running_mean reproduces the 2026-07-09 anchor EXACTLY for torch-free
    baselines -- per seed and aggregated. This is the P8.0 merge gate: if it fails, the harness is
    wrong and the horizon values from the same run are worthless. Do not smooth over a mismatch."""
    horizon = _load_horizon()
    legacy = _load(RESULTS_PATH)
    new_by_seed = _horizon_per_seed_map(horizon, env_id, policy, "att_running_mean")
    legacy_by_seed = _legacy_per_seed_map(legacy, env_id, policy)
    assert set(new_by_seed) == set(legacy_by_seed) == {101, 202, 303}
    for seed in sorted(legacy_by_seed):
        assert new_by_seed[seed] == legacy_by_seed[seed], (
            f"running-mean not reproduced for {env_id}/{policy} seed {seed}: "
            f"re-run {new_by_seed[seed]!r} != committed {legacy_by_seed[seed]!r}"
        )
    new_agg = horizon["aggregated"][env_id][policy]["att_running_mean"]["mean"]
    legacy_agg = legacy["aggregated"][env_id][policy]["average_travel_time"]["mean"]
    assert new_agg == legacy_agg, (
        f"aggregated running-mean not reproduced for {env_id}/{policy}: "
        f"re-run {new_agg!r} != committed {legacy_agg!r}"
    )


@pytest.mark.parametrize("env_id", ENVS)
def test_mappo_horizon_double_compute_when_present(env_id: str) -> None:
    """MAPPO horizon: double-compute when the (long, tmux) cells have landed; a reasoned skip
    that auto-arms otherwise. MAPPO may cross the N2 boundary, so its running mean is NOT asserted
    against the committed anchor -- only the file's internal double-compute is."""
    results = _load_horizon()
    if "mappo" not in results["aggregated"].get(env_id, {}):
        pytest.skip("MAPPO horizon pending the tmux p0_baselines re-run (P8.0 split); see PROVENANCE")
    for quantity in HORIZON_QUANTITIES:
        _assert_horizon_double_compute(results, env_id, "mappo", quantity)


def test_horizon_anchor_is_load_bearing(tmp_path: Path) -> None:
    """Mutation: corrupting a stored horizon mean must make the double-compute fail with match=."""
    results = _load_horizon()
    results["aggregated"]["cf_hz1x1"]["MaxPressure"]["att_horizon"]["mean"] += 1.0
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(results), encoding="utf-8")
    with pytest.raises(AssertionError, match="double-compute mismatch"):
        _assert_horizon_double_compute(_load(tampered), "cf_hz1x1", "MaxPressure", "att_horizon")


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
