"""Tests for ``offline/sumo_anchor_rescore.py``, part 2 -- rho under A27(f)'s anchor pairs and H3's clauses re-scored.

* **The tautology guard** (``BRIEF_47`` §4 test 3): under the REGISTERED pair the re-scoring reproduces, under ``==``, every
  committed per-cell rho of P7.3a, P7.3b, P7.3d and P7.3c, every per-arm interval, the hz1x1 and grid4x4 H3 blocks and P7.3c's
  clause-3 block -- which proves the functions used are the registered ones.  It runs on the committed artifacts alone (no
  simulator, no ``output/``).
* **Synthetic pairs** with dyadic values, so the expected rho, seed means and CI are exact by an independent route.
* **The clause rules**: P5.2c's vocabulary on clauses 1 and 2 (boundaries included), A24's partition with G recomputed,
  and the plan's C3 -- clause 3 VOID when the pair leaves no positive gap.

The registered mutants (packet): the anchor roles swapped (sign), the registered MaxPressure used under every pair
(denominator), a plain mean over episodes instead of the per-draw seed means (pooling), a clause boundary moved.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest

import offline.sumo_anchor_rescore as sar
import offline.transfer_curve as tc

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"


@pytest.fixture(scope="module")
def inputs() -> dict[str, Any]:
    return sar.load_inputs(DATA)


# ----------------------------------------------------------------------
# Inputs by digest
# ----------------------------------------------------------------------


def test_the_four_inputs_are_pinned_by_digest_and_a_changed_byte_is_refused(tmp_path: Path) -> None:
    assert set(sar.INPUT_ARTIFACTS) == {
        "p7_3a_zero_shot.json", "p7_3b_anchor.json", "p7_3d_grid4x4.json", "p7_3c_grid4x4.json"
    }
    for name in sar.INPUT_ARTIFACTS:
        shutil.copyfile(DATA / name, tmp_path / name)
    raw = bytearray((tmp_path / "p7_3b_anchor.json").read_bytes())
    raw[-2:-1] = b" "
    (tmp_path / "p7_3b_anchor.json").write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="sha256"):
        sar.load_inputs(tmp_path)


# ----------------------------------------------------------------------
# The tautology guard on the committed artifacts
# ----------------------------------------------------------------------


def test_the_registered_pair_reproduces_every_committed_rho_interval_and_clause_block(inputs: dict[str, Any]) -> None:
    result = sar.registered_reproduction(inputs)
    assert result["all_equal"] is True
    assert result["n_rows"] == {
        "p7_3a_zero_shot.json": 4000, "p7_3b_anchor.json": 500, "p7_3d_grid4x4.json": 500,
        "p7_3c_grid4x4.json": 4500,
    }
    assert sorted(result["blocks_equal"]) == sorted([
        "p7_3a.by_subject_arm", "p7_3a.by_subject_arm_seed", "p7_3a.h3",
        "p7_3b.by_subject_arm", "p7_3b.by_subject_arm_seed",
        "p7_3d.rho", "p7_3d.h3",
        "p7_3c.arms", "p7_3c.rho0", "p7_3c.clause_3", "p7_3c.total_effect", "p7_3c.adaptation_effect",
        "p7_3c.transfer", "p7_3c.adjacent_steps", "p7_3c.budget_secondary", "p7_3c.clause_3_with_G_recomputed",
    ])


def test_the_guard_refuses_one_committed_rho_moved_by_one_ulp(inputs: dict[str, Any]) -> None:
    planted = copy.deepcopy(inputs)
    row = next(r for r in planted["p7_3d_grid4x4.json"]["cells"] if r["kind"] == "dt")
    row["rho_e_sumo"] = math.nextafter(row["rho_e_sumo"], math.inf)
    with pytest.raises(ValueError, match="does not reproduce"):
        sar.registered_reproduction(planted)


def test_registered_verdicts_recomputed_by_the_vocabulary_are_a19_a20_and_a24s(inputs: dict[str, Any]) -> None:
    rescored = sar.rescore_pair(inputs, "registered", new_rows=None)
    verdicts = {
        (scenario, subject): {clause: entry["verdict"] for clause, entry in clauses.items()}
        for scenario, by_subject in rescored["h3"].items()
        for subject, clauses in by_subject.items()
    }
    assert verdicts == {
        ("cityflow1x1", "mappo1000"): {"clause_1": "HELD", "clause_2": "FAILED"},
        ("cityflow1x1", "mix50"): {"clause_1": "HELD", "clause_2": "FAILED"},
        ("cityflow_grid4x4", "mappo1000_dt_nomix_h4"): {"clause_1": "HELD", "clause_2": "HELD", "clause_3": "(i)"},
    }
    assert sar.REGISTERED_H3_VERDICTS == {
        "cityflow1x1": {"mappo1000": {"clause_1": "HELD", "clause_2": "FAILED"},
                        "mix50": {"clause_1": "HELD", "clause_2": "FAILED"}},
        "cityflow_grid4x4": {"mappo1000_dt_nomix_h4": {"clause_1": "HELD", "clause_2": "HELD", "clause_3": "(i)"}},
    }
    grid = rescored["h3"]["cityflow_grid4x4"]["mappo1000_dt_nomix_h4"]
    assert grid["clause_2"]["scored"] is False
    assert grid["clause_3"]["G"] == tc.P7_3C_G and grid["clause_3"]["G_att"] == tc.P7_3C_G_ATT
    assert all(entry["agrees_with_registered"] for by in rescored["h3"].values() for c in by.values()
               for entry in c.values())


# ----------------------------------------------------------------------
# Synthetic pairs: exact by an independent route
# ----------------------------------------------------------------------


def _anchor(arm: str, draw: int, e_sumo: float, att_env: float) -> dict[str, Any]:
    return {"kind": "anchor", "subject": None, "arm": arm, "seed": None, "draw_id": draw, "e_sumo": e_sumo,
            "att_env": att_env, "horizon_vehicle_count": 10.0, "episode_reward": -1.0}


def _dt(subject: str, arm: str, seed: int, draw: int, e_sumo: float) -> dict[str, Any]:
    return {"kind": "dt", "subject": subject, "arm": arm, "seed": seed, "draw_id": draw, "e_sumo": e_sumo,
            "att_env": e_sumo, "horizon_vehicle_count": 10.0, "episode_reward": -1.0}


def test_rho_under_a_new_pair_uses_that_pairs_anchors_and_is_exact() -> None:
    """Registered FT 300 / MP 200; new MP 1:1 250, FT kcf 280.  Arm at 260: registered rho 0.4; (MP 1:1, FT k=4)
    (300-260)/(300-250) = 0.8; (MP per-link, FT kcf) (280-260)/(280-200) = 0.25; (MP 1:1, FT kcf) (280-260)/(280-250)."""
    registered = {draw: {"fixedtime": _anchor("fixedtime", draw, 300.0, 300.0),
                         "maxpressure": _anchor("maxpressure", draw, 200.0, 200.0)} for draw in (1000, 1001)}
    new = {"maxpressure_1to1": {draw: _anchor("maxpressure_1to1", draw, 250.0, 250.0) for draw in (1000, 1001)},
           "fixedtime_kcf": {draw: _anchor("fixedtime_kcf", draw, 280.0, 280.0) for draw in (1000, 1001)}}
    row = _dt("s", "b_mean_k100", 101, 1000, 260.0)
    expected = {"registered": 0.4, "mp1to1_ftk4": 0.8, "mpperlink_ftkcf": 0.25, "mp1to1_ftkcf": 20.0 / 30.0}
    for pair, value in expected.items():
        anchors = sar.pair_anchor_rows(pair, registered=registered, new=new)
        assert anchors[1000]["fixedtime"]["arm"] == sar.PAIRS[pair]["fixedtime"]
        assert anchors[1000]["maxpressure"]["arm"] == sar.PAIRS[pair]["maxpressure"]
        assert sar.rescored_rows([row], anchors)[0]["rho_e_sumo"] == value


def test_the_per_arm_interval_is_the_mean_over_draws_of_seed_means_with_ddof_one() -> None:
    """Draw 1000: five seeds at rho 0.5; draw 1001: at 0.75.  Per-draw means 0.5 and 0.75; mean 0.625; s = sqrt(0.03125)."""
    registered = {draw: {"fixedtime": _anchor("fixedtime", draw, 300.0, 300.0),
                         "maxpressure": _anchor("maxpressure", draw, 200.0, 200.0)} for draw in (1000, 1001)}
    rows = [_dt("mappo1000", "b_mean_k100", seed, 1000, 250.0) for seed in (101, 202, 303, 404, 505)]
    rows += [_dt("mappo1000", "b_mean_k100", seed, 1001, 225.0) for seed in (101, 202, 303, 404, 505)]
    block = sar.hz1x1_rho_block(sar.rescored_rows(rows, registered), registered,
                                groups=[("mappo1000", "b_mean_k100", "registered_prompt")])
    entry = block["by_subject_arm"][0]["e_sumo"]
    s = math.sqrt(((0.5 - 0.625) ** 2 + (0.75 - 0.625) ** 2) / (2 - 1))
    assert (entry["n_draws"], entry["mean"], entry["std"]) == (2, 0.625, s)
    assert entry["ci95"] == 1.96 * s / math.sqrt(2)
    assert block["per_draw"]["mappo1000/b_mean_k100"]["e_sumo"] == {"1000": 0.5, "1001": 0.75}


def test_a_pair_missing_one_draws_anchor_is_refused() -> None:
    registered = {draw: {"fixedtime": _anchor("fixedtime", draw, 300.0, 300.0),
                         "maxpressure": _anchor("maxpressure", draw, 200.0, 200.0)} for draw in (1000, 1001)}
    new = {"maxpressure_1to1": {1000: _anchor("maxpressure_1to1", 1000, 250.0, 250.0)},
           "fixedtime_kcf": {draw: _anchor("fixedtime_kcf", draw, 280.0, 280.0) for draw in (1000, 1001)}}
    with pytest.raises(ValueError, match="same draws"):
        sar.pair_anchor_rows("mp1to1_ftk4", registered=registered, new=new)


def test_a_row_whose_draw_has_no_anchors_is_refused() -> None:
    registered = {1000: {"fixedtime": _anchor("fixedtime", 1000, 300.0, 300.0),
                         "maxpressure": _anchor("maxpressure", 1000, 200.0, 200.0)}}
    with pytest.raises(ValueError, match="no anchors"):
        sar.rescored_rows([_dt("s", "b_mean_k100", 101, 1001, 250.0)], registered)


def test_the_paired_blocks_name_the_pairs_concrete_arms() -> None:
    registered = {draw: {"fixedtime": _anchor("fixedtime", draw, 300.0, 300.0),
                         "maxpressure": _anchor("maxpressure", draw, 200.0, 200.0)} for draw in (1000, 1001)}
    new = {"maxpressure_1to1": {d: _anchor("maxpressure_1to1", d, 250.0, 250.0) for d in (1000, 1001)},
           "fixedtime_kcf": {d: _anchor("fixedtime_kcf", d, 280.0, 280.0) for d in (1000, 1001)}}
    anchors = sar.pair_anchor_rows("mp1to1_ftkcf", registered=registered, new=new)
    rows = [_dt("mappo1000", "b_mean_k100", seed, draw, 260.0 + draw - 1000 + seed / 1000.0)
            for seed in (101, 202, 303, 404, 505) for draw in (1000, 1001)]
    block = sar.hz1x1_rho_block(sar.rescored_rows(rows, anchors), anchors,
                                groups=[("mappo1000", "b_mean_k100", "registered_prompt")])
    paired = block["by_subject_arm"][0]["paired_att"]["e_sumo"]
    assert paired["fixedtime"]["right_arm"] == "fixedtime_kcf"
    assert paired["maxpressure"]["right_arm"] == "maxpressure_1to1"


# ----------------------------------------------------------------------
# The clause rules
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "clause_1", "clause_2"),
    [
        (0.1, 0.3, "HELD", "HELD"),
        (-0.2, 0.3, "NOT RESOLVED", "HELD"),
        (-0.5, -0.1, "FAILED", "HELD"),
        (0.0, 0.3, "NOT RESOLVED", "HELD"),
        (1.2, 1.5, "HELD", "FAILED"),
        (0.9, 1.1, "HELD", "NOT RESOLVED"),
        (0.5, 1.0, "HELD", "NOT RESOLVED"),
        (1.0, 1.4, "HELD", "NOT RESOLVED"),
    ],
)
def test_clauses_1_and_2_follow_p5_2cs_vocabulary_on_rhos_interval(
    low: float, high: float, clause_1: str, clause_2: str
) -> None:
    assert sar.clause_1_verdict(low, high) == clause_1
    assert sar.clause_2_verdict(low, high) == clause_2


@pytest.mark.parametrize(
    ("delta", "low", "high", "gap", "verdict"),
    [
        (0.06, 0.02, 0.10, 0.1, "(i)"),
        (0.04, 0.02, 0.06, 0.1, "(ii)"),
        (0.01, -0.01, 0.03, 0.1, "(iii)"),
        (-0.05, -0.08, -0.02, 0.1, "(iv)"),
        (0.05, 0.02, 0.08, 0.1, "(i)"),
    ],
)
def test_clause_3_is_a24s_partition_with_the_recomputed_gap(
    delta: float, low: float, high: float, gap: float, verdict: str
) -> None:
    result = sar.clause_3_verdict(delta, low, high, gap=gap)
    assert result == {"verdict": verdict, "partition": verdict, "gap_positive": True}


def test_clause_3_is_void_when_the_pair_leaves_no_positive_gap() -> None:
    result = sar.clause_3_verdict(0.06, 0.02, 0.10, gap=-0.05)
    assert result["gap_positive"] is False
    assert result["verdict"].startswith("VOID")
    assert "A19(a)" in result["verdict"]
    assert result["partition"] == "(i)"
    assert sar.clause_3_verdict(0.06, 0.02, 0.10, gap=0.0)["gap_positive"] is False


# ----------------------------------------------------------------------
# rescore_pair on the committed artifacts with constructed new anchors
# ----------------------------------------------------------------------


def _renamed(anchors: dict[int, dict[str, Any]], role: str, arm: str, *, shift: float = 0.0) -> dict[int, dict[str, Any]]:
    """The registered *role* rows of every draw under a NEW arm name, ``e_sumo`` moved by *shift* seconds."""
    return {draw: {**slot[role], "arm": arm, "e_sumo": slot[role]["e_sumo"] + shift} for draw, slot in anchors.items()}


def _new_rows_like_registered(inputs: dict[str, Any], *, mp_shift: float = 0.0) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scenario, name in (("cityflow1x1", "p7_3a_zero_shot.json"), ("cityflow_grid4x4", "p7_3d_grid4x4.json")):
        anchors = sar.registered_anchors(inputs[name])
        out[scenario] = {
            "maxpressure_1to1": _renamed(anchors, "maxpressure", "maxpressure_1to1", shift=mp_shift),
            "fixedtime_kcf": _renamed(anchors, "fixedtime", "fixedtime_kcf"),
        }
    return out


def test_new_anchors_equal_to_the_registered_ones_reproduce_the_registered_numbers_under_every_pair(
    inputs: dict[str, Any],
) -> None:
    registered = sar.rescore_pair(inputs, "registered", new_rows=None)
    new = _new_rows_like_registered(inputs)
    for pair in ("mp1to1_ftk4", "mpperlink_ftkcf", "mp1to1_ftkcf"):
        result = sar.rescore_pair(inputs, pair, new_rows=new)
        for tag in ("p7_3a", "p7_3b"):
            for mine, theirs in zip(result[tag]["by_subject_arm"], registered[tag]["by_subject_arm"]):
                assert (mine["e_sumo"], mine["att_env"]) == (theirs["e_sumo"], theirs["att_env"])
        assert result["p7_3d"]["rho"]["by_draw"] == registered["p7_3d"]["rho"]["by_draw"]
        assert result["p7_3c"]["arms"] == registered["p7_3c"]["arms"]
        grid = result["h3"]["cityflow_grid4x4"]["mappo1000_dt_nomix_h4"]["clause_3"]
        assert (grid["G"], grid["verdict"]) == (tc.P7_3C_G, "(i)")
        assert result["anchors"]["cityflow1x1"] == dict(sar.PAIRS[pair])


def test_a_shifted_new_maxpressure_moves_rho_and_g_exactly_as_a_hand_computation(inputs: dict[str, Any]) -> None:
    """MP 1:1's E_sumo = the registered MaxPressure's + 10 s on every grid draw.  rho of every P7.3d DT row is recomputed
    BY HAND -- (FT - arm) / (FT - (MP + 10)) -- seed means per draw in seed order, the mean over draws, and G = 1 - it."""
    import numpy as np

    new = _new_rows_like_registered(inputs, mp_shift=10.0)
    result = sar.rescore_pair(inputs, "mp1to1_ftk4", new_rows=new)
    anchors = sar.registered_anchors(inputs["p7_3d_grid4x4.json"])
    by_draw: dict[int, list[float]] = {}
    for row in inputs["p7_3d_grid4x4.json"]["cells"]:
        if row["kind"] != "dt":
            continue
        ft = anchors[row["draw_id"]]["fixedtime"]["e_sumo"]
        mp = anchors[row["draw_id"]]["maxpressure"]["e_sumo"] + 10.0
        by_draw.setdefault(row["draw_id"], []).append((ft - row["e_sumo"]) / (ft - mp))
    per_draw = {draw: sum(values) / len(values) for draw, values in by_draw.items()}
    expected_mean = float(np.asarray([per_draw[d] for d in sorted(per_draw)], dtype=np.float64).mean())
    stats = result["p7_3d"]["rho"]["registered_arm"]["e_sumo"]
    assert stats["mean"] == expected_mean
    assert result["h3"]["cityflow_grid4x4"]["mappo1000_dt_nomix_h4"]["clause_3"]["G"] == 1.0 - expected_mean
    assert result["h3"]["cityflow_grid4x4"]["mappo1000_dt_nomix_h4"]["clause_3"]["G"] != tc.P7_3C_G
    assert result["anchors"]["cityflow_grid4x4"] == {"fixedtime": "fixedtime", "maxpressure": "maxpressure_1to1"}


def test_p7_3cs_per_draw_blocks_under_the_registered_pair_are_the_committed_series(inputs: dict[str, Any]) -> None:
    """zs_k100 IS P7.3d's zero-shot arm (stage 1 reproduced it bit for bit), so its per-draw series is P7.3d's committed
    ``rho.by_draw``; and every arm's mean over its per-draw block is the committed interval's mean."""
    from offline.dt_gate import mean_ci95

    result = sar.rescore_pair(inputs, "registered", new_rows=None)
    per_draw = result["p7_3c"]["per_draw"]
    committed = inputs["p7_3d_grid4x4.json"]["rho"]["by_draw"]
    assert per_draw["zs_k100"]["e_sumo"] == {draw: entry["e_sumo"] for draw, entry in committed.items()}
    assert sorted(per_draw) == sorted(row.name for row in tc.P7_3C_ARMS)
    for arm, series in per_draw.items():
        values = [series["e_sumo"][draw] for draw in sorted(series["e_sumo"], key=int)]
        assert mean_ci95(values).mean == inputs["p7_3c_grid4x4.json"]["arms"][arm]["e_sumo"]["mean"]
