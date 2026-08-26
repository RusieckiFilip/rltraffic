"""The committed artifact ``docs/data/p5_3a_rtg_probe.json``, read and pinned by a test.

Every comparable artifact in this repo is read by a test; this one was not, so nothing mechanically
held the numbers the Return Packet quotes. The review found that and it is MAJOR-4.

What is pinned here, and why each one
--------------------------------------
* **The shape** -- 40 cells, exactly the 12 declared interventions, 7200 steps and 20 streams
  everywhere -- because a silently truncated campaign would still look like an artifact.
* **R6's null control** -- ``mappo1000``'s ``grid_g5`` target IS that checkpoint's own, so its
  flip rate, TVD and logit delta must be **exactly** zero. These five cells are the only exact-zero
  TVDs in the whole file.
* **A8's registered prediction** -- ``fixedtime`` flips no action under any intervention. It was
  registered before any number existed and its whole purpose is to indict the probe if it fails.
* **The BL-1 corrections** -- ``fixedtime``'s TVD is never zero, and the two zero-flip tiers are
  separated by four orders of magnitude in TVD at comparable logit movement. The packet once
  claimed an exact zero here that does not exist; a test now holds the true statement.
* **Row B's ordering** -- ``random`` narrowest, the mixtures widest. This is the reversal of the
  marginal statistic that P5.3b's tier axis depends on.
* **The delta table's copies** -- eight ``(z, p_value)`` pairs and every other field are copied out
  of ``p4_6_grid.json`` / ``p4_7_grid.json``. A copy with no pointer is a drift hazard; a copy with
  a **checked** pointer is not, and this is the check (``BRIEF_29`` §B).
* **Verdict-freedom of the shipped bytes**, not of a payload the writer re-validated in memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from offline.method_tier_grid import assert_no_verdicts
from offline.rtg_ablation import (
    ARTIFACT_FORMAT_VERSION,
    INTERVENTION_KEYS,
    PROBE_SEEDS,
    PROBE_STREAM_COUNT,
    PROBE_TIERS,
    RTG_SPREAD_TIMESTEPS,
)

DATA = Path(__file__).resolve().parents[1] / "docs" / "data"
ARTIFACT = DATA / "p5_3a_rtg_probe.json"

#: A8, registered in BRIEF_28 before any number existed.
A8_TIER = "fixedtime"
#: R6, the live null control: grid_g5's target is mappo1000's own.
R6_TIER, R6_KEY = "mappo1000", "grid_g5"


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    if not ARTIFACT.is_file():
        pytest.skip(f"{ARTIFACT} has not been generated in this tree")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _cells(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return artifact["probe"]["cells"]


def test_the_artifact_has_the_declared_shape(artifact: dict[str, Any]) -> None:
    assert artifact["format_version"] == ARTIFACT_FORMAT_VERSION
    assert artifact["probe"]["n_cells"] == 40
    assert sorted(_cells(artifact)) == sorted(PROBE_TIERS)
    for tier, per_seed in _cells(artifact).items():
        assert sorted(per_seed) == sorted(str(s) for s in PROBE_SEEDS), tier
        for seed, cell in per_seed.items():
            assert sorted(cell["interventions"]) == sorted(INTERVENTION_KEYS), f"{tier}@{seed}"
            assert cell["n_steps"] == 7200, f"{tier}@{seed}"
            assert cell["n_streams"] == PROBE_STREAM_COUNT == 20
            assert cell["stream_indices"] == list(range(0, 200, 10))
            for key, value in cell["interventions"].items():
                assert value["n_steps_compared"] == 7200, f"{tier}@{seed} {key}"


def test_the_baseline_arm_is_identically_zero_in_every_cell(artifact: dict[str, Any]) -> None:
    """By construction: the baseline is compared against itself. If it is not 0, nothing else means anything."""
    for tier, per_seed in _cells(artifact).items():
        for seed, cell in per_seed.items():
            base = cell["interventions"]["baseline"]
            assert base["flip_rate"] == 0.0, f"{tier}@{seed}"
            assert base["tvd"] == 0.0, f"{tier}@{seed}"
            assert base["mean_abs_logit_delta"] == 0.0, f"{tier}@{seed}"


def test_r6_the_live_null_control_is_exactly_zero_on_all_five_seeds(
    artifact: dict[str, Any],
) -> None:
    """``grid_g5``'s target −5762.0 IS ``mappo1000``'s own, so the intervention is the identity."""
    per_seed = _cells(artifact)[R6_TIER]
    assert artifact["declared"]["grid"][5] == -5762.0
    for seed, cell in per_seed.items():
        assert cell["target_rtg"] == -5762.0
        entry = cell["interventions"][R6_KEY]
        assert entry["flip_rate"] == 0.0, seed
        assert entry["tvd"] == 0.0, seed
        assert entry["mean_abs_logit_delta"] == 0.0, seed


def test_the_only_exact_zero_tvds_in_the_file_are_r6s_five(artifact: dict[str, Any]) -> None:
    """⭐ BL-1: the packet once pointed at an exact zero on ``fixedtime`` that does not exist.

    The artifact's only exact-zero TVDs outside the self-comparing baseline arm are R6's five.
    """
    zeros = [
        (tier, seed, key)
        for tier, per_seed in _cells(artifact).items()
        for seed, cell in per_seed.items()
        for key, value in cell["interventions"].items()
        if key != "baseline" and value["tvd"] == 0.0
    ]
    assert sorted(zeros) == sorted(
        (R6_TIER, str(seed), R6_KEY) for seed in PROBE_SEEDS
    ), zeros


def test_a8s_registered_prediction_holds_fixedtime_flips_no_action(
    artifact: dict[str, Any],
) -> None:
    """Registered before any number existed. A non-zero result indicts the PROBE, not the prompt."""
    per_seed = _cells(artifact)[A8_TIER]
    for seed, cell in per_seed.items():
        for key, value in cell["interventions"].items():
            assert value["flip_rate"] == 0.0, f"{A8_TIER}@{seed} {key}"


def test_fixedtimes_tvd_is_never_exactly_zero_and_random_is_four_orders_larger(
    artifact: dict[str, Any],
) -> None:
    """⭐ BL-1's corrected discriminator, held by a test so the false version cannot come back.

    Both tiers flip nothing. They are **not** distinguished by an exact zero -- ``fixedtime``'s TVD
    is ~1e−9 and never 0 -- and their ``mean_abs_logit_delta`` is the SAME ORDER. What separates
    them is TVD, by roughly four orders of magnitude, at comparable logit movement.
    """
    def spread(tier: str, field: str) -> tuple[float, float]:
        values = [
            value[field]
            for cell in _cells(artifact)[tier].values()
            for key, value in cell["interventions"].items()
            if key != "baseline"
        ]
        assert len(values) == 55, f"{tier}: 11 interventions x 5 seeds"
        return min(values), max(values)

    ft_tvd, rd_tvd = spread("fixedtime", "tvd"), spread("random", "tvd")
    assert ft_tvd[0] > 0.0, "fixedtime's TVD is small, not zero"
    assert 1e-11 < ft_tvd[0] and ft_tvd[1] < 1e-8
    assert 1e-8 < rd_tvd[0] and rd_tvd[1] < 1e-3
    # Four orders of magnitude apart, on the quantity that is NOT the logit movement.
    assert rd_tvd[1] / ft_tvd[1] > 1e3

    ft_mald, rd_mald = spread("fixedtime", "mean_abs_logit_delta"), spread(
        "random", "mean_abs_logit_delta"
    )
    # ...while the logit movement itself is the same order: neither ratio exceeds 3x.
    assert 0.33 < rd_mald[1] / ft_mald[1] < 3.0
    assert ft_mald[1] > 0.15 and rd_mald[1] > 0.15


def test_the_crosscheck_records_both_rollouts_and_the_headline_flip_rate(
    artifact: dict[str, Any],
) -> None:
    checked = artifact["crosscheck"]
    assert checked["tier"] == "mappo1000" and checked["seed"] == 101
    assert checked["draw_id"] == 1000
    assert checked["n_decisions_compared"] == 360
    assert checked["runs"]["target_first"]["target_rtg"] == 0.0
    assert checked["runs"]["target_last"]["target_rtg"] == -13000.0
    assert checked["runs"]["target_first"]["att_horizon"] == 102.0546056260342
    assert checked["runs"]["target_last"]["att_horizon"] == 106.05901820187535
    assert checked["action_flip_rate"] == 0.4722222222222222
    assert checked["att_difference"] == 4.004412575841144
    # The gap that makes the limitation sentence load-bearing: the closed loop moves ~100x more
    # than the teacher-forced probe on the same cell.
    probe_max = max(
        value["flip_rate"]
        for cell in _cells(artifact)["mappo1000"].values()
        for value in cell["interventions"].values()
    )
    assert checked["action_flip_rate"] / probe_max > 50.0


def test_row_b_orders_random_narrowest_and_the_mixtures_widest(
    artifact: dict[str, Any],
) -> None:
    """⭐ The reversal P5.3b's tier axis depends on: the marginal statistic ranks ``random`` FIRST."""
    spread = artifact["tables"]["spread"]
    pooled = {
        tier: entry["between_episode_rtg_scaled"]["pooled"] for tier, entry in spread.items()
    }
    assert min(pooled, key=pooled.get) == "random"
    assert sorted(pooled, key=pooled.get)[-3:] == sorted(("mix33", "mix50", "mix67"), key=pooled.get)
    assert pooled["mix50"] / pooled["random"] > 50.0

    # And the marginal statistic points the other way, which is the whole point.
    marginal = {tier: entry["marginal_std_scaled"] for tier, entry in spread.items()}
    assert marginal["random"] > marginal["fixedtime"] > marginal["mappo1000"]
    assert list(RTG_SPREAD_TIMESTEPS) == artifact["declared"]["rtg_spread_timesteps"]
    for tier, entry in spread.items():
        assert entry["rtg_summary_routes_agree"] is True, tier
        assert entry["training_streams"] == 200, tier


def test_the_delta_table_is_a_checked_copy_of_the_committed_grids(
    artifact: dict[str, Any],
) -> None:
    """⭐ BRIEF_29 §B: a copy with no pointer is the drift hazard; this is the pointer, checked.

    The δ table duplicates eight ``(z, p_value)`` pairs into a second artifact -- which is why
    ``tests/test_erfc_determinism.py``'s sweep moved from 322 to 330. That duplication is only safe
    while it is mechanically tied to its source, so every field is compared against the row it was
    copied from.
    """
    grids = {
        name: json.loads((DATA / name).read_text(encoding="utf-8"))
        for name in ("p4_6_grid.json", "p4_7_grid.json")
    }
    delta = artifact["tables"]["delta"]
    assert sorted(delta) == sorted(PROBE_TIERS)

    for tier, row in delta.items():
        source_name = row["source_artifact"]
        assert source_name in grids, f"{tier}: unknown source {source_name!r}"
        matches = [
            c
            for c in grids[source_name]["behaviour_comparisons"]
            if c["left_arm"] == row["source_arm"]
        ]
        assert len(matches) == 1, f"{tier}: {len(matches)} rows match {row['source_arm']!r}"
        source = matches[0]
        assert row["source_arm"] == f"dt@{tier}"
        for field in (
            "left_arm",
            "right_arm",
            "mean_difference",
            "median_difference",
            "ci95_low",
            "ci95_high",
            "ci95_half_width",
            "ci95_width",
            "rank_biserial",
            "wins",
            "losses",
            "ties",
            "n_shared_draws",
        ):
            assert row[field] == source[field], f"{tier}.{field} drifted from {source_name}"
        assert row["wilcoxon"] == source["wilcoxon"], f"{tier}.wilcoxon drifted"
        assert row["behaviour_margin_degenerate"] == (
            source["ci95_low"] <= 0.0 <= source["ci95_high"]
        )

    # The eight pairs test_erfc_determinism's count guard now covers.
    pairs = [row["wilcoxon"] for row in delta.values()]
    assert len(pairs) == 8
    assert all("z" in p and "p_value" in p for p in pairs)


def test_the_shipped_bytes_carry_no_verdict(artifact: dict[str, Any]) -> None:
    """Validating the file that was written, not a payload the writer re-checked in memory."""
    assert_no_verdicts(artifact)
    text = ARTIFACT.read_text(encoding="utf-8").lower()
    for banned in ("inert", "weak lever", "equivalent", "a9 survives"):
        assert banned not in text, f"the artifact must carry no reading: found {banned!r}"


def test_the_artifact_records_which_commits_produced_its_inputs(
    artifact: dict[str, Any],
) -> None:
    """MAJOR-1: a single write-time commit describes when the report was assembled, nothing more."""
    runtime = artifact["runtime"]
    assert runtime["measurement_git_commits"], "DEFERRED 39's split is not populated"
    assert runtime["unreachable_measurement_commits"] == []
    assert runtime["written_at_git_commit"]
    assert runtime["torch_num_threads"] == 1


def test_the_artifact_carries_the_across_seed_spread_not_only_per_seed_values(
    artifact: dict[str, Any],
) -> None:
    """MINOR-4: *a mean over those five is a summary that hides its own subject* -- so ship both."""
    spread = artifact["probe"]["across_seed_spread"]
    assert sorted(spread) == sorted(PROBE_TIERS)
    for tier, per_key in spread.items():
        assert sorted(per_key) == sorted(INTERVENTION_KEYS), tier
        for key, fields in per_key.items():
            for field in ("flip_rate", "tvd", "mean_abs_logit_delta"):
                stats = fields[field]
                assert stats["n_seeds"] == 5, f"{tier} {key} {field}"
                assert stats["min"] <= stats["mean"] <= stats["max"]
                assert stats["sd"] >= 0.0
    # The two zero-flip tiers have zero spread; a tier with movement does not.
    assert spread["random"]["zero"]["flip_rate"]["sd"] == 0.0
    assert spread["fixedtime"]["zero"]["flip_rate"]["sd"] == 0.0
    assert spread["mappo500"]["zero"]["flip_rate"]["sd"] > 0.0
