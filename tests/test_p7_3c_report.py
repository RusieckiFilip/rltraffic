"""P7.3c C6 (``BRIEF_41`` C6; PREREGISTRATION A24(c)-(h)): P7.3c's report body, ``p7.3c-grid4x4/1.0``.

Plan sections 10 and 13: T-verdict, T-perseed, T-pairing, T-report and ``DEFERRED`` 94.

HOW EACH EXPECTATION IS DERIVED
-------------------------------
* **The constants and the texts** are read from the registration and the committed artifact by THIS file's route:
  G = 1 - rho0 from ``p7_3d_grid4x4.json``'s registered arm, and every verbatim sentence found in ``PREREGISTRATION.md``.
* **T-verdict** drives the pure partition at every boundary with exact floats (``math.nextafter`` for "just below").
* **T-pairing, T-perseed and the estimators** run on SYNTHETIC rows over a synthetic rho0 series whose mean is NOT the
  registered one, so a G recomputed from the data cannot pass.  Every expectation is rebuilt here from the rows' own
  values: per-draw five-seed means in seed order, per-draw differences by draw id, then ``offline.dt_gate.mean_ci95`` --
  the registered estimator, which is not the code under test.
* **T-report** runs the REAL ``report`` over a COMPLETE 4,700-chunk work directory: stage 1 synthesised from the
  committed artifact's own 700 records, stages 2 and 3 built here from each arm's designed shift, with the real digests
  -- the draws' demand files, A20(a)'s five checkpoints and the thirty trained ones through the pinned record.  Then
  each registered refusal, ONE change at a time, and each must leave nothing written.

GATES, each naming what it consumes: the main tree's draws (``RLTRAFFIC_DRAWS``), ``output/`` (``RLTRAFFIC_OUTPUT_ROOT``)
with A20(a)'s five and the thirty trained checkpoints.  Nothing here executes a campaign driver.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import pytest

import offline.few_shot as few_shot
import offline.transfer_calibration as tc
import offline.transfer_curve as tcv
from offline.dt_gate import mean_ci95
from tests.test_p7_3d_campaign_path import (
    HEAD_SHA,
    _canary_record,
    _complete_set,
    _needs_checkpoints,
    _needs_parity,
    _write_stubs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"
ARTIFACT = DATA / "p7_3d_grid4x4.json"
MAIN = Path("/home/filip/rltraffic")
OUTPUT_ROOT = Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT", str(MAIN / "output")))
DRAWS_ROOT = Path(os.environ.get("RLTRAFFIC_DRAWS", str(MAIN / "scenarios" / "draws")))
GRID = "cityflow_grid4x4"
ZS = "mappo1000_dt_nomix_h4"
SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)
REPRODUCE, PRIMARY, CONTROLS = "p7_3c_reproduce", "p7_3c_primary", "p7_3c_controls"
#: (arm name, stage, subject, prompt arm) for A24's nine DT arms, written from the registration's text.
ARMS: tuple[tuple[str, str, str, str], ...] = (
    ("zs_k100", REPRODUCE, ZS, "b_mean_k100"),
    ("ft_k5", PRIMARY, "ft_k5", "b_mean_k5"),
    ("ft_k20", PRIMARY, "ft_k20", "b_mean_k20"),
    ("ft_k100", PRIMARY, "ft_k100", "b_mean_k100"),
    ("zs_k5", CONTROLS, ZS, "b_mean_k5"),
    ("zs_k20", CONTROLS, ZS, "b_mean_k20"),
    ("scratch_k100", CONTROLS, "scratch_k100", "b_mean_k100"),
    ("ft_k100_b1000", CONTROLS, "ft_k100_b1000", "b_mean_k100"),
    ("ft_k100_b16000", CONTROLS, "ft_k100_b16000", "b_mean_k100"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _committed() -> dict[str, Any]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _registration_text() -> str:
    """PREREGISTRATION.md with its ``**`` emphasis removed, as the verbatim constants are stored."""
    return (REPO_ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8").replace("**", "")


def _stats(per_draw: Mapping[int, float]) -> dict[str, float]:
    """``mean_ci95`` over the values in DRAW-ID order, as the registered estimator is applied."""
    stats = mean_ci95([per_draw[draw] for draw in sorted(per_draw)])
    return {"delta": stats.mean, "ci95": stats.ci95, "lo": stats.mean - stats.ci95, "hi": stats.mean + stats.ci95}


# ==================================================================================================================
# The constants and the verbatim texts
# ==================================================================================================================


def test_g_and_g_att_are_a24ds_constants_from_the_committed_artifact_and_the_half_gap_is_exact() -> None:
    assert tcv.P7_3C_G == 0.11453573207299894 and tcv.P7_3C_G_ATT == 0.11453996971792169
    registered = _committed()["rho"]["registered_arm"]
    assert tcv.P7_3C_G == 1.0 - registered["e_sumo"]["mean"]
    assert tcv.P7_3C_G_ATT == 1.0 - registered["att_env"]["mean"]
    assert tcv.P7_3C_G / 2 == 0.05726786603649947
    assert tcv.P7_3C_ARTIFACT_FORMAT_VERSION == "p7.3c-grid4x4/1.0"


def test_every_sentence_the_artifact_carries_is_the_registrations_verbatim() -> None:
    text = _registration_text()
    for label, sentence in tcv.P7_3C_VERDICT_SENTENCES.items():
        assert f'"{sentence}"' in text, f"verdict sentence {label} is not A24(d)'s"
    assert list(tcv.P7_3C_VERDICT_SENTENCES) == ["(i)", "(ii)", "(iii)", "(iv)"]
    for sentence in (*tcv.P7_3C_WHAT_THIS_DOES_NOT_SAY, *tcv.P7_3C_ADJACENT_STEPS_READING):
        assert sentence in text, sentence[:60]
    assert len(tcv.P7_3C_WHAT_THIS_DOES_NOT_SAY) == 6


# ==================================================================================================================
# T-verdict: A24(d)'s partition, at every boundary
# ==================================================================================================================

HALF = tcv.P7_3C_G / 2
BELOW = math.nextafter(HALF, -math.inf)
TINY = math.nextafter(0.0, math.inf)


@pytest.mark.parametrize(
    ("delta", "lo", "hi", "label"),
    [
        (HALF, TINY, 0.2, "(i)"),  # Delta exactly G/2 with lo > 0: MET
        (0.09, 0.01, 0.17, "(i)"),
        (BELOW, TINY, 0.2, "(ii)"),  # one ulp below G/2
        (0.03, 0.001, 0.059, "(ii)"),
        (0.09, 0.0, 0.18, "(iii)"),  # lo exactly 0.0: never (i), whatever Delta is
        (0.02, 0.0, 0.04, "(iii)"),  # lo exactly 0.0: never (ii)
        (-0.02, -0.04, 0.0, "(iii)"),  # hi exactly 0.0: never (iv)
        (0.0, -0.01, 0.01, "(iii)"),
        (-0.02, -0.04, -TINY, "(iv)"),  # hi < 0
    ],
)
def test_the_partition_at_every_boundary(delta: float, lo: float, hi: float, label: str) -> None:
    assert tcv.p7_3c_verdict(delta, lo, hi) == label


def test_the_att_env_partition_uses_g_att() -> None:
    half_att = tcv.P7_3C_G_ATT / 2
    assert half_att > HALF
    assert tcv.p7_3c_verdict(HALF, TINY, 0.2, gap=tcv.P7_3C_G_ATT) == "(ii)"
    assert tcv.p7_3c_verdict(half_att, TINY, 0.2, gap=tcv.P7_3C_G_ATT) == "(i)"


# ==================================================================================================================
# T-pairing: by draw id, never by position
# ==================================================================================================================


def test_pairing_is_by_draw_id_and_a_draw_in_one_series_only_is_refused() -> None:
    left = {1000: 0.9, 1001: 0.7, 1002: 0.8, 1003: 0.6}
    right = {1000: 0.85, 1001: 0.72, 1002: 0.75, 1003: 0.61}
    reordered = {draw: left[draw] for draw in (1003, 1001, 1000, 1002)}
    base = tcv.paired_by_draw(left, right, label="t")
    assert tcv.paired_by_draw(reordered, right, label="t") == base  # the order of the entries is irrelevant
    expected = _stats({draw: left[draw] - right[draw] for draw in left})
    assert {key: base[key] for key in ("delta", "ci95", "lo", "hi")} == expected and base["n_draws"] == 4
    # the same values assigned to OTHER draws: the mean of the differences is unchanged, its CI is not
    swapped = {1000: 0.7, 1001: 0.9, 1002: 0.6, 1003: 0.8}
    moved = tcv.paired_by_draw(swapped, right, label="t")
    assert moved["ci95"] != base["ci95"]
    with pytest.raises(ValueError, match=r"t: draw\(s\) \[1003\] are in one series only"):
        tcv.paired_by_draw({draw: left[draw] for draw in (1000, 1001, 1002)}, right, label="t")


# ==================================================================================================================
# The estimators on synthetic rows: T-verdict end to end, T-perseed, the effects, the steps, the budgets
# ==================================================================================================================

SYN_DRAWS: tuple[int, ...] = tuple(range(1000, 1020))
#: Per-seed offsets of the zero-shot cells around their draw's rho0 -- and, for the fine-tunes, per-seed shifts
#: chosen so EXACTLY three seeds' Delta_100 clear G/2 and one seed's CI straddles zero.
ZS_SEED_OFFSET = {101: -0.02, 202: -0.01, 303: 0.0, 404: 0.01, 505: 0.02}
FT100_SEED_SHIFT = {101: 0.10, 202: 0.09, 303: 0.08, 404: 0.04, 505: 0.0}  # mean 0.062 >= G/2 = 0.0573
#: Each other arm's shift over the zero-shot cell of the same seed and draw.  ft_k20 BELOW ft_k5 makes the total step
#: 5 -> 20 negative, and zs_k20 barely above zs_k5 makes the adaptation step 5 -> 20 negative too.
ARM_SHIFT = {
    "ft_k5": 0.03, "ft_k20": 0.02, "zs_k5": 0.005, "zs_k20": 0.004, "scratch_k100": -0.2,
    "ft_k100_b1000": 0.02, "ft_k100_b16000": 0.075,
}


def _rho0(draw: int) -> float:
    return 0.5 + 0.0078125 * (draw - 1000)  # mean 0.57421875 -- NOT the registered 0.885...


def _noise(draw: int, seed: int, salt: int) -> float:
    return 0.001 * (((draw + 3 * seed + salt) % 7) - 3)


def _synthetic() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """(rows, rho0_by_draw, zero_shot_records): P7.3c's rows over SYN_DRAWS, and the zero-shot artifact's series."""
    rho0_by_draw = {
        str(draw): {"e_sumo": _rho0(draw), "att_env": _rho0(draw) - 0.001, "n_seeds": 5} for draw in SYN_DRAWS
    }
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for name, stage, subject, arm in ARMS:
        for seed in SEEDS:
            for draw in SYN_DRAWS:
                zero = _rho0(draw) + ZS_SEED_OFFSET[seed]
                if name == "zs_k100":
                    value = zero
                elif name == "ft_k100":
                    value = zero + FT100_SEED_SHIFT[seed] + _noise(draw, seed, 1)
                else:
                    value = zero + ARM_SHIFT[name] + _noise(draw, seed, len(name))
                row = {
                    "kind": "dt", "subject": subject, "arm": arm, "seed": seed, "draw_id": draw, "scenario": GRID,
                    "stage": stage, "rho_e_sumo": value, "rho_att_env": value - 0.001,
                }
                rows.append(row)
                if name == "zs_k100":
                    records.append({**row, "stage": "grid4x4_confirmatory"})
    return rows, rho0_by_draw, records


def _per_draw(rows: list[dict[str, Any]], name: str, key: str = "rho_e_sumo") -> dict[int, float]:
    """The arm's per-draw five-seed means, summed in SEED order -- the rows are in seed order."""
    _n, stage, subject, arm = next(spec for spec in ARMS if spec[0] == name)
    sums: dict[int, list[float]] = {}
    for row in rows:
        if (row["stage"], row["subject"], row["arm"]) == (stage, subject, arm):
            sums.setdefault(int(row["draw_id"]), []).append(float(row[key]))
    return {draw: sum(values) / len(values) for draw, values in sums.items()}


def _minus(left: Mapping[int, float], right: Mapping[int, float]) -> dict[int, float]:
    return {draw: left[draw] - right[draw] for draw in sorted(left)}


def test_clause_3_uses_the_registered_g_never_one_recomputed_from_rho0() -> None:
    rows, rho0, records = _synthetic()
    estimates = tcv.p7_3c_estimates(rows, rho0, records)
    clause = estimates["clause_3"]
    rho0_e = {int(d): float(v["e_sumo"]) for d, v in rho0.items()}
    expected = _stats(_minus(_per_draw(rows, "ft_k100"), rho0_e))
    assert {key: clause["delta_100"][key] for key in ("delta", "ci95", "lo", "hi")} == expected
    assert clause["G"] == tcv.P7_3C_G and clause["half_gap"] == tcv.P7_3C_G / 2
    assert clause["closure_fraction"] == {
        "value": expected["delta"] / tcv.P7_3C_G,
        "ci95_low": expected["lo"] / tcv.P7_3C_G,
        "ci95_high": expected["hi"] / tcv.P7_3C_G,
    }
    # the synthetic rho0's mean is far from the registered one: a G recomputed from it would be another number
    assert 1.0 - sum(rho0_e.values()) / len(rho0_e) != tcv.P7_3C_G
    assert clause["verdict"] == tcv.p7_3c_verdict(expected["delta"], expected["lo"], expected["hi"])
    assert clause["verdict"] == "(i)"
    x = 100.0 * expected["delta"] / tcv.P7_3C_G
    ci = f"[{100.0 * expected['lo'] / tcv.P7_3C_G:.1f}%, {100.0 * expected['hi'] / tcv.P7_3C_G:.1f}%]"
    assert clause["sentence"] == tcv.P7_3C_VERDICT_SENTENCES["(i)"].replace("X% [CI]", f"{x:.1f}% {ci}")
    # att_env: the same computation with G_att, reported beside the primary and never deciding it
    rho0_a = {int(d): float(v["att_env"]) for d, v in rho0.items()}
    att = _stats(_minus(_per_draw(rows, "ft_k100", "rho_att_env"), rho0_a))
    assert {key: clause["att_env"]["delta_100"][key] for key in ("delta", "ci95", "lo", "hi")} == att
    assert clause["att_env"]["closure_fraction"]["value"] == att["delta"] / tcv.P7_3C_G_ATT
    assert clause["att_env"]["verdict"] == tcv.p7_3c_verdict(att["delta"], att["lo"], att["hi"], gap=tcv.P7_3C_G_ATT)


def test_the_five_per_seed_deltas_pair_seed_with_seed_and_count_exactly_three_above_half_the_gap() -> None:
    rows, rho0, records = _synthetic()
    per_seed = tcv.p7_3c_estimates(rows, rho0, records)["clause_3"]["per_seed"]
    ft_key = ("p7_3c_primary", "ft_k100", "b_mean_k100")
    expected = []
    for seed in SEEDS:
        # the test's own route: per seed, the mean over draws of CELL-level differences, the zero-shot side read
        # from the artifact's records (Q6b)
        ft = {r["draw_id"]: r["rho_e_sumo"] for r in rows if (r["stage"], r["subject"], r["arm"]) == ft_key
              and r["seed"] == seed}
        zero = {r["draw_id"]: r["rho_e_sumo"] for r in records if r["seed"] == seed}
        expected.append({"seed": seed, **_stats(_minus(ft, zero))})
    assert [{key: entry[key] for key in ("seed", "delta", "ci95", "lo", "hi")} for entry in per_seed["seeds"]] == expected
    deltas = [entry["delta"] for entry in expected]
    assert per_seed["between_seed_sd"] == mean_ci95(deltas).std
    assert per_seed["n_at_least_half_gap"] == 3
    assert per_seed["n_lo_above_zero"] == sum(1 for entry in expected if entry["lo"] > 0) == 4


def test_the_effects_the_five_adjacent_steps_and_the_budget_secondary_are_each_paired_by_draw() -> None:
    rows, rho0, records = _synthetic()
    estimates = tcv.p7_3c_estimates(rows, rho0, records)
    rho0_e = {int(d): float(v["e_sumo"]) for d, v in rho0.items()}
    arm = {name: _per_draw(rows, name) for name, *_rest in ARMS}

    def check(block: Mapping[str, Any], per_draw: Mapping[int, float]) -> None:
        assert {key: block[key] for key in ("delta", "ci95", "lo", "hi")} == _stats(per_draw)

    for k in (5, 20, 100):
        check(estimates["total_effect"][f"k{k}"], _minus(arm[f"ft_k{k}"], rho0_e))
    check(estimates["adaptation_effect"]["k5"], _minus(arm["ft_k5"], arm["zs_k5"]))
    check(estimates["adaptation_effect"]["k20"], _minus(arm["ft_k20"], arm["zs_k20"]))
    assert estimates["adaptation_effect"]["k100"]["is"] == "clause_3.delta_100"
    check(estimates["transfer"], _minus(arm["ft_k100"], arm["scratch_k100"]))
    a5, a20 = _minus(arm["ft_k5"], arm["zs_k5"]), _minus(arm["ft_k20"], arm["zs_k20"])
    a100 = _minus(arm["ft_k100"], rho0_e)
    steps = {step["name"]: step for step in estimates["adjacent_steps"]["steps"]}
    assert list(steps) == ["total 0->5", "total 5->20", "total 20->100", "adaptation 5->20", "adaptation 20->100"]
    for name, per_draw in (
        ("total 0->5", _minus(arm["ft_k5"], rho0_e)),
        ("total 5->20", _minus(arm["ft_k20"], arm["ft_k5"])),
        ("total 20->100", _minus(arm["ft_k100"], arm["ft_k20"])),
        ("adaptation 5->20", _minus(a20, a5)),
        ("adaptation 20->100", _minus(a100, a20)),
    ):
        check(steps[name], per_draw)
        assert steps[name]["refuted"] is (_stats(per_draw)["hi"] < 0)
    assert [steps[n]["refuted"] for n in steps] == [False, True, False, True, False]
    assert estimates["adjacent_steps"]["reading"] == list(tcv.P7_3C_ADJACENT_STEPS_READING)
    for label, name in (("B1000", "ft_k100_b1000"), ("B16000", "ft_k100_b16000")):
        block = estimates["budget_secondary"][label]
        check(block["delta_vs_rho0"], _minus(arm[name], rho0_e))
        check(block["vs_B4000"], _minus(arm[name], arm["ft_k100"]))
    for name, *_rest in ARMS:
        assert estimates["arms"][name]["e_sumo"]["mean"] == mean_ci95([arm[name][d] for d in sorted(arm[name])]).mean


def test_a_step_whose_ci_straddles_zero_is_not_refuted() -> None:
    """A24(e): an adjacent step is refuted iff its CI lies ENTIRELY below zero (hi < 0) -- one straddling zero is not.

    Added after mutant A01 ("refuted iff lo < 0") SURVIVED the test above, none of whose steps straddled zero (disclosed
    in the packet).  Here ft_k20 sits 0.01 above ft_k5 on even draws and 0.01 below on odd ones, cell for cell."""
    rows, rho0, records = _synthetic()
    five = {(r["seed"], r["draw_id"]): r["rho_e_sumo"] for r in rows if r["subject"] == "ft_k5"}
    for row in rows:
        if row["subject"] == "ft_k20":
            row["rho_e_sumo"] = five[(row["seed"], row["draw_id"])] + (0.01 if row["draw_id"] % 2 == 0 else -0.01)
    steps = {step["name"]: step for step in tcv.p7_3c_estimates(rows, rho0, records)["adjacent_steps"]["steps"]}
    straddling = steps["total 5->20"]
    assert straddling["lo"] < 0.0 < straddling["hi"]
    assert straddling["refuted"] is False


def test_restricting_the_draws_recomputes_by_the_same_function_and_a_partial_arm_is_refused() -> None:
    rows, rho0, records = _synthetic()
    kept = [draw for draw in SYN_DRAWS if draw not in (1003, 1011)]
    restricted = tcv.p7_3c_estimates(rows, rho0, records, draws=kept)
    expected = tcv.p7_3c_estimates(
        [r for r in rows if r["draw_id"] in kept], {d: v for d, v in rho0.items() if int(d) in kept},
        [r for r in records if r["draw_id"] in kept],
    )
    assert restricted == expected and restricted["clause_3"]["delta_100"]["n_draws"] == 18
    partial = [r for r in rows if not (r["subject"] == "ft_k20" and r["seed"] == 303 and r["draw_id"] == 1005)]
    with pytest.raises(ValueError, match=r"ft_k20 has 4 seed\(s\) on draw 1005, not the five"):
        tcv.p7_3c_estimates(partial, rho0, records)


# ==================================================================================================================
# T-report: the REAL report over a COMPLETE 4,700-chunk work directory, then one refusal at a time
# ==================================================================================================================


def _needs_the_real_inputs() -> None:
    _needs_parity(*((GRID, draw) for draw in (1000, 1099)))
    _needs_checkpoints()
    checkpoints = OUTPUT_ROOT / "p7_3c_training" / "checkpoints"
    if len(list(checkpoints.glob("*.pt"))) != 30:
        pytest.skip(f"{checkpoints} does not hold the thirty trained checkpoints (gitignored, main tree only)")


def _k_targets(k: int) -> dict[str, float]:
    artifact = json.loads((DATA / "p7_3d_calibration.json").read_text(encoding="utf-8"))
    return {
        str(ix): float(artifact["per_intersection"][ix]["budgets"][f"k{k}"]["target"])
        for ix in artifact["intersection_ids"]
    }


EMPTY_EVENTS: dict[str, Any] = {
    "collisions": [], "n_collisions": 0, "teleports": [], "n_teleports": 0, "n_explained_teleports": 0,
    "n_unexplained_teleports": 0, "vanished_ids": [], "n_vanished_without_arrival": 0,
}
#: The ONE collision the fixture records in stages 2-3: an ft_k100 cell, so A23(d)'s robustness set is DERIVED
#: from the chunks as {1020, 1042} (stage 1's two) plus this draw.
FT_EVENT = ("ft_k100", 202, 1077)
#: Each stage-2/3 arm's shift over the zero-shot cell of the same seed and draw, on both definitions.
REPORT_SHIFT = {
    "ft_k5": 0.03, "ft_k20": 0.05, "ft_k100": 0.07, "zs_k5": 0.005, "zs_k20": 0.004, "scratch_k100": -0.2,
    "ft_k100_b1000": 0.02, "ft_k100_b16000": 0.075,
}


def _event(time: float) -> dict[str, Any]:
    return {
        "time": time, "collider": "c1", "victim": "v1", "colliderType": "cf_parity", "victimType": "cf_parity",
        "colliderSpeed": 5.0, "victimSpeed": 0.0, "type": "collision", "lane": "A0left0_0", "pos": 3.5,
        "collider_fate": "arrived_at_collision_step", "collider_fate_time": None,
    }


def _stage1_chunk(
    record: Mapping[str, Any], events: list[dict[str, Any]], *, git_commit: str = HEAD_SHA
) -> dict[str, Any]:
    """A stage-1 chunk carrying ITS committed record's values (bookkeeping moved; the commit *git_commit*)."""
    decisions = int(record["decisions"])
    chunk = {key: value for key, value in record.items() if key not in ("rho_e_sumo", "rho_att_env")}
    teleports = [{"time": e["time"], "vehicle": e["victim"]} for e in events][: int(record["n_teleports"])]
    chunk.update(
        {
            "format_version": tcv.GRID4X4_ARTIFACT_FORMAT_VERSION, "stage": REPRODUCE, "git_commit": git_commit,
            "canary_seconds": 0.8, "actions": [[0] * 16 for _ in range(decisions)], "actions_in_range": True,
            "local_return_from_lanes": dict(record["local_return"]), "n_observations": 3600,
            "collisions": [dict(e) for e in events], "n_collisions": len(events), "teleports": teleports,
            "n_explained_teleports": len(teleports), "n_unexplained_teleports": 0, "vanished_ids": [],
        }
    )
    if record["kind"] == "dt":
        targets = record["target_rtg"]
        chunk["rtg_series"] = {ix: [targets[ix]] + [0] * (decisions - 1) for ix in targets}
        chunk["reward_series"] = {ix: [0] * decisions for ix in targets}
    return chunk


class ReportSet:
    """The complete set on disk, its anchors, and what each stage-2/3 cell was built from."""

    def __init__(self, work: Path, anchors: dict[int, dict[str, dict[str, float]]], trained: dict[str, str]) -> None:
        self.work = work
        self.anchors = anchors
        self.trained = trained


@pytest.fixture(scope="module")
def report_set(tmp_path_factory: pytest.TempPathFactory) -> ReportSet:
    _needs_the_real_inputs()
    return build_report_set(tmp_path_factory.mktemp("p7_3c") / "cells")


def build_report_set(work: Path, *, git_commit: str = HEAD_SHA) -> ReportSet:
    """The COMPLETE 4,700-chunk set in *work* (created here), every chunk recording *git_commit* -- the commit the
    code that reads it is at, so J1(c) finds nothing changed.  ``tests/test_p7_3c_campaign.py`` builds it at its
    sandbox clone's commit, and drives it through the real campaign driver."""
    committed = _committed()
    events: dict[str, list[dict[str, Any]]] = {}
    for block in committed["collisions"]["per_arm"].values():
        for event in block["events"]:
            events.setdefault(str(event["cell"]), []).append(
                {key: value for key, value in event.items() if key not in ("cell", "arm", "seed", "draw_id")}
            )
    record = json.loads((DATA / "p7_3c_finetune.json").read_text(encoding="utf-8"))
    trained = {name: str(entry["checkpoint_sha256"]) for name, entry in record["runs"].items()}
    work.mkdir(parents=True)
    _canary_record(work)
    stage1: dict[tuple[str, int | None, int], dict[str, Any]] = {}
    anchors: dict[int, dict[str, dict[str, float]]] = {}
    for rec in committed["cells"]:
        chunk = _stage1_chunk(rec, events.get(tcv.cell_chunk_name(rec), []), git_commit=git_commit)
        (work / tcv.cell_chunk_name(chunk)).write_text(json.dumps(chunk), encoding="utf-8")
        stage1[(str(rec["arm"]), rec["seed"], int(rec["draw_id"]))] = chunk
        if rec["kind"] == "anchor":
            anchors.setdefault(int(rec["draw_id"]), {})[str(rec["arm"])] = {
                "e_sumo": float(rec["e_sumo"]), "att_env": float(rec["att_env"]),
            }
    for name, stage, subject, arm in ARMS[1:]:
        k = int(arm.rsplit("_k", 1)[1])
        targets = _k_targets(k)
        for seed in SEEDS:
            for draw in tcv.HELD_OUT_DRAWS:
                base = copy.deepcopy(stage1[("b_mean_k100", seed, draw)])
                zero = {"e_sumo": None, "att_env": None}
                values: dict[str, float] = {}
                for key in ("e_sumo", "att_env"):
                    fixed, pressure = anchors[draw]["fixedtime"][key], anchors[draw]["maxpressure"][key]
                    zero[key] = (fixed - float(base[key])) / (fixed - pressure)
                    rho = zero[key] + REPORT_SHIFT[name] + 0.001 * (((draw + 3 * seed + len(name)) % 7) - 3)
                    values[key] = fixed - rho * (fixed - pressure)
                trained_digest = trained.get(f"{subject}_seed{seed}")
                chunk = {
                    **base, **EMPTY_EVENTS, "subject": subject, "arm": arm, "stage": stage,
                    "e_sumo": values["e_sumo"], "att_env": values["att_env"], "att_horizon": values["att_env"],
                    "target_rtg": dict(targets), "rtg_first": dict(targets), "rtg_last": {ix: 0 for ix in targets},
                    "rtg_series": {ix: [targets[ix]] + [0] * 359 for ix in targets},
                    "checkpoint": f"/checkpoints/{subject}_seed{seed}.pt",
                    "checkpoint_sha256": trained_digest or tc.GRID4X4_CHECKPOINT_SHA256[seed],
                }
                if (name, seed, draw) == FT_EVENT:
                    chunk.update(
                        collisions=[_event(1234.0)], n_collisions=1,
                        teleports=[{"time": 1234.0, "vehicle": "v1"}], n_teleports=1, n_explained_teleports=1,
                    )
                (work / tcv.cell_chunk_name(chunk)).write_text(json.dumps(chunk), encoding="utf-8")
    assert len(list(work.glob("cell_*.json"))) == 4700
    return ReportSet(work, anchors, trained)


def _report(work: Path, out: Path, data: Path = DATA) -> dict[str, Any]:
    return tcv.report(
        work_dir=work, out_path=out, output_root=OUTPUT_ROOT, out_root=DRAWS_ROOT, data_dir=data,
        stage=tcv.STAGE_P7_3C,
    )


def _rows_from_disk(work: Path) -> list[dict[str, Any]]:
    """Every chunk's identity and rho by THIS file's route: the formula over the chunk's own ATTs and its anchors."""
    chunks = [json.loads(path.read_bytes()) for path in sorted(work.glob("cell_*.json"))]
    anchors = {
        (int(c["draw_id"]), str(c["arm"])): c for c in chunks if c["kind"] == "anchor"
    }
    rows = []
    for chunk in chunks:
        draw = int(chunk["draw_id"])
        row = {key: chunk[key] for key in ("kind", "subject", "arm", "seed", "draw_id", "stage")}
        for key in ("e_sumo", "att_env"):
            fixed = float(anchors[(draw, "fixedtime")][key])
            pressure = float(anchors[(draw, "maxpressure")][key])
            row[f"rho_{key}"] = (fixed - float(chunk[key])) / (fixed - pressure)
        rows.append(row)
    return rows


def test_report_writes_the_p7_3c_artifact_from_a_complete_set(report_set: ReportSet, tmp_path: Path) -> None:
    out = tmp_path / "p7_3c_grid4x4.json"
    artifact = _report(report_set.work, out)
    assert json.loads(out.read_bytes()) == json.loads(json.dumps(artifact))
    assert artifact["format_version"] == "p7.3c-grid4x4/1.0" and artifact["stage"] == "p7_3c"
    assert artifact["n_cells_declared"] == 4700 and len(artifact["cells"]) == 4700
    assert artifact["stage1_reproduction"]["line"] == "stage1_check REPRODUCED 700/700"
    assert artifact["inputs"]["zero_shot_artifact_sha256"] == _sha(ARTIFACT)
    assert artifact["inputs"]["finetune_record_sha256"] == _sha(DATA / "p7_3c_finetune.json")

    rows = _rows_from_disk(report_set.work)
    committed = _committed()
    rho0_e = {int(d): float(v["e_sumo"]) for d, v in committed["rho"]["by_draw"].items()}
    ft100 = {}
    for draw in tcv.HELD_OUT_DRAWS:
        values = [r["rho_e_sumo"] for r in rows if (r["subject"], r["draw_id"]) == ("ft_k100", draw)
                  and r["stage"] == PRIMARY]
        assert len(values) == 5
        ft100[draw] = sum(values) / 5
    delta = _stats(_minus(ft100, rho0_e))
    clause = artifact["clause_3"]
    assert {key: clause["delta_100"][key] for key in ("delta", "ci95", "lo", "hi")} == delta
    assert clause["verdict"] == tcv.p7_3c_verdict(delta["delta"], delta["lo"], delta["hi"]) == "(i)"
    assert clause["sentence"].startswith(
        "Exploratory (A24): on grid4x4, fine-tuning on 100 target-domain MaxPressure episodes closes "
    )
    robustness = clause["robustness"]
    assert robustness["draws_removed"] == [1020, 1042, 1077]
    kept = {draw: ft100[draw] for draw in ft100 if draw not in (1020, 1042, 1077)}
    assert {key: robustness["delta_100"][key] for key in ("delta", "ci95", "lo", "hi")} == _stats(
        _minus(kept, {d: rho0_e[d] for d in kept})
    )
    assert robustness["outcome_differs"] is (robustness["verdict"] != clause["verdict"])
    per_arm = artifact["collisions"]["per_arm"]
    assert (per_arm["ft_k100"]["n_cells_with_collision"], per_arm["fixedtime"]["n_cells_with_collision"],
            per_arm["zs_k100"]["n_cells_with_collision"]) == (1, 1, 1)
    assert sum(block["n_cells_with_collision"] for block in per_arm.values()) == 3
    in_support = artifact["in_support"]["by_k"]
    calibration = json.loads((DATA / "p7_3d_calibration.json").read_text(encoding="utf-8"))
    for k in (5, 20, 100):
        for ix in calibration["intersection_ids"]:
            corpus = [calibration["probe_returns"]["sumo"][str(d)][ix] for d in range(201, 201 + k)]
            entry = in_support[f"k{k}"][ix]
            assert entry["corpus_range"] == [min(corpus), max(corpus)]
            assert entry["corpus_position"] == "above"  # A24(e): every Rule B target lies above the corpus's best
            assert entry["source_range"] == calibration["per_intersection"][ix]["support_range"]
    assert artifact["what_this_does_not_say"] == list(tcv.P7_3C_WHAT_THIS_DOES_NOT_SAY)


@contextmanager
def _moved(path: Path, edit: Callable[[dict[str, Any]], None] | None) -> Iterator[None]:
    """One chunk edited -- or, with ``edit=None``, removed -- for the block's duration; its bytes restored after."""
    original = path.read_bytes()
    if edit is None:
        path.unlink()
    else:
        chunk = json.loads(original)
        edit(chunk)
        path.write_text(json.dumps(chunk), encoding="utf-8")
    try:
        yield
    finally:
        path.write_bytes(original)


def _name(subject: str | None, arm: str, seed: int | None, draw: int) -> str:
    return tcv.cell_chunk_name({"subject": subject, "arm": arm, "seed": seed, "draw_id": draw, "scenario": GRID})


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("stage1", r"stage 1 did not reproduce p7_3d_grid4x4\.json: stage1_check NOT REPRODUCED 1/700"),
        ("absent", r"1 declared cell\(s\) have no chunk"),
        ("trained_digest", r"not the .* that ft_k20 seed 404's committed record pins today"),
        ("p7_3d_stage", r"the chunk says stage='grid4x4_confirmatory'"),
    ],
)
def test_each_late_refusal_leaves_nothing_written(
    report_set: ReportSet, tmp_path: Path, case: str, message: str
) -> None:
    out = tmp_path / "p7_3c_grid4x4.json"
    work = report_set.work
    if case == "stage1":
        target, edit = work / _name(None, "maxpressure", None, 1044), lambda c: c.__setitem__("n_created", 1)
    elif case == "absent":
        target, edit = work / _name("scratch_k100", "b_mean_k100", 505, 1099), None
    elif case == "trained_digest":
        target, edit = work / _name("ft_k20", "b_mean_k20", 404, 1010), lambda c: c.__setitem__(
            "checkpoint_sha256", "0" * 64
        )
    else:
        target, edit = work / _name("ft_k5", "b_mean_k5", 101, 1000), lambda c: c.__setitem__(
            "stage", "grid4x4_confirmatory"
        )
    with _moved(target, edit):
        with pytest.raises(ValueError, match=message):
            _report(work, out)
    assert not out.exists() and list(tmp_path.iterdir()) == []


def test_the_zero_shot_artifact_and_the_canary_are_refused_first(report_set: ReportSet, tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for name in ("p7_3d_calibration.json", "p7_3c_finetune.json", "p7_3d_reference_cells.json"):
        shutil.copyfile(DATA / name, data / name)
    (data / "p7_3d_grid4x4.json").write_bytes(ARTIFACT.read_bytes() + b"\n")
    out = tmp_path / "out" / "p7_3c_grid4x4.json"
    with pytest.raises(ValueError, match="not the pinned c63c371f"):
        _report(report_set.work, out, data=data)
    canary = report_set.work / tc.CANARY_RECORD_NAME
    original = canary.read_bytes()
    canary.unlink()
    try:
        with pytest.raises(FileNotFoundError, match="wrote no canary.json"):
            _report(report_set.work, out)
    finally:
        canary.write_bytes(original)
    assert not out.exists() and not out.parent.exists()


def test_a_single_p7_3c_stage_is_refused_before_anything_is_read(tmp_path: Path) -> None:
    """One declaration, one artifact: an artifact over one stage would publish an estimator on a partial set
    (A24(c)).  Replaces C5's refusal of the whole declaration, which C6's body supersedes (disclosed in the packet)."""
    out = tmp_path / "p7_3c_grid4x4.json"
    for stage in (REPRODUCE, PRIMARY, CONTROLS):
        with pytest.raises(ValueError, match="one declaration, one artifact"):
            tcv.report(
                work_dir=tmp_path / "missing", out_path=out, output_root=tmp_path, out_root=tmp_path, data_dir=DATA,
                stage=stage,
            )
    assert not out.exists() and not (tmp_path / "missing").exists()


# ==================================================================================================================
# DEFERRED 94: a SECOND event inside an A23 cell refuses (P7.3d's report, P7.3d's complete set)
# ==================================================================================================================


def test_deferred_94_a_second_event_in_the_fixed_time_draw_1020_cell_refuses_and_writes_nothing(tmp_path: Path) -> None:
    _needs_parity(*((GRID, draw) for draw in tcv.HELD_OUT_DRAWS))
    _needs_checkpoints()
    payloads, _expected = _complete_set(git_commit=HEAD_SHA)
    target = next(p for p in payloads if (p["arm"], p["seed"], p["draw_id"]) == ("fixedtime", None, 1020))
    (first,) = target["collisions"]
    second = {**first, "time": float(first["time"]) + 100.0, "collider": "9001", "victim": "9002"}
    target.update(
        collisions=[first, second], n_collisions=2, n_teleports=2, n_explained_teleports=2,
        teleports=[*target["teleports"], {"time": second["time"], "vehicle": second["collider"]}],
    )
    work = tmp_path / "cells"
    _write_stubs(payloads, work)
    _canary_record(work)
    out = tmp_path / "out" / "p7_3d_grid4x4.json"
    with pytest.raises(ValueError, match=r"A23\(f\).*fixedtime_seednone_draw1020\.json records 2"):
        tcv.report(
            work_dir=work, out_path=out, output_root=OUTPUT_ROOT, out_root=DRAWS_ROOT, data_dir=DATA,
            stage=tcv.STAGE_GRID4X4,
        )
    assert not out.exists() and not out.parent.exists()
