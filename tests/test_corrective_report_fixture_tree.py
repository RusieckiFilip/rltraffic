"""P9.1's report end to end on a FULL-SIZE fixture run tree (``offline/corrective_report.py``; ``BRIEF_46`` G4; plan C6's
"T-regress for each artifact (rebuilt byte for byte from fixture run directories)").

No format of its own.  ``tests/p9_1_report_fixtures.py`` writes the campaign's run tree in the campaign's formats at its
real size: 255 run records, 53 new cells through both routes, 38 old cells' re-rolls (18,200 episodes), P3a's six
records, the GPU trace and the declarations, which are written by the campaign's own declare functions.  The registered
side is real and read-only: the grid's P5.1 / P5.2 cells through P5.2c's pinned loader, P8.4b's rows, the P5
checkpoints, the committed artifacts.  ``load_sources`` and the five builders then run exactly as at G4.  This is what
the T-regress on the campaign's own directories (``tests/test_corrective_report_load.py``) will check once the data
exist, here on fixtures that need no campaign.

Independent routes: the committed P1 blocks and behaviour comparisons are read straight from ``docs/data``; one contrast
is recomputed from the raw committed JSON and the fixture's own formula with numpy's arithmetic in P5.2c's order; the
planted pause and the planted repeat are read off the fixture's constants, not off the report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import offline.corrective_report as rep
import offline.corrective_round as cr
import offline.tier_statements as tst
from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS
from tests import p9_1_report_fixtures as fx

REPO = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT", "/nonexistent"))
CORPUS = Path(os.environ.get("RLTRAFFIC_CORPUS_V11", "/nonexistent"))
DRAWS = Path(os.environ.get("RLTRAFFIC_DRAWS", str(OUTPUT.parent / "scenarios" / "draws")))
GIT = {"commit": "f" * 40, "dirty": False, "code_root": "fixture"}

pytestmark = pytest.mark.skipif(
    not ((OUTPUT / "p5_1").is_dir() and (CORPUS / "cf_hz1x1__random_fresh" / "manifest.json").is_file()
         and (CORPUS / "cf_grid4x4__random_fresh" / "manifest.json").is_file()),
    reason="needs the main tree's registered output/ (RLTRAFFIC_OUTPUT_ROOT) and the v1.1 corpus with A27(d)'s two "
           "random_fresh tiers (RLTRAFFIC_CORPUS_V11)")


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> fx.FixtureTree:
    return fx.write_tree(tmp_path_factory.mktemp("p9_1_tree"), main_output=OUTPUT, corpus_root=CORPUS,
                         draws_root=DRAWS, repo_root=REPO)


def _load(tree: fx.FixtureTree) -> rep.Sources:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tst, "git_block", lambda: dict(GIT))
        return rep.load_sources(tree.roots, d4rl_path=tree.d4rl_path)


@pytest.fixture(scope="module")
def sources(tree: fx.FixtureTree) -> rep.Sources:
    return _load(tree)


@pytest.fixture(scope="module")
def artifacts(sources: rep.Sources) -> dict[str, dict[str, Any]]:
    return {name: builder(sources) for name, builder in rep.ARTIFACTS.items()}


def test_the_fixture_tree_is_the_campaigns_full_size(tree: fx.FixtureTree) -> None:
    runs = Path(tree.roots.runs_root)
    assert len(list((runs / "training").rglob("*.json"))) == len(cr.training_runs()) == 255
    assert len(list((runs / "eval_i").rglob("*.json"))) == len(cr.evaluation_cells()) == 53
    assert len(list((runs / "eval_ii").rglob("*.json"))) == 53 * len(TRAINING_SEEDS) * len(HELD_OUT_DRAWS) == 26_500
    committed = sum(len(cr.old_cell_committed(REPO, artifact, arm)) for artifact, arm in cr.HZ1X1_OLD_CELLS)
    assert len(list((runs / "old_cells" / "hz1x1").glob("*.json"))) == committed == 18_200
    assert len(list((runs / "p3a").rglob("*.json"))) == len(cr.p3a_cells()) == 6
    assert sorted(p.name for p in Path(tree.roots.run_root).iterdir()) == [
        "declaration_mix_m1000_m500.json", "declaration_random_fresh_grid4x4.json",
        "declaration_random_fresh_hz1x1.json"]


def test_load_sources_joins_every_new_old_and_constructed_cell_at_its_size(sources: rep.Sources) -> None:
    trained = set(cr.training_cells())
    for cell in cr.evaluation_cells():
        cid = rep.new_cell_id(cell)
        assert len(sources.records[cid]) == 500 and len(sources.cells[cid].rows) == 500, cid
        assert len(sources.runs.get(cid, ())) == (5 if cell in trained else 0), cid
    for artifact, arm in cr.HZ1X1_OLD_CELLS:
        cid = rep.hz1x1_old_id(artifact, arm)
        assert len(sources.records[cid]) == len(cr.old_cell_committed(REPO, artifact, arm)), cid
    for tier in rep.MIXTURE_REFERENCE_TIERS:
        assert len(sources.records[rep.constructed_id(tier)]) == 500
    assert set(sources.declarations) == {"mix_m1000_m500", "random_fresh_hz1x1", "random_fresh_grid4x4"}


def test_every_artifact_builds_and_carries_the_records_and_runs_of_every_cell_it_reads(
        artifacts: dict[str, dict[str, Any]], sources: rep.Sources) -> None:
    assert sorted(artifacts) == sorted(rep.ARTIFACTS)
    for name, payload in artifacts.items():
        point = payload["registered_in"].split(",")[0]
        assert payload["label"] == cr.LABELS[point] and payload["format_version"] == rep.REPORT_FORMAT_VERSION, name
        blocks = {k: v for k, v in payload.items() if k not in ("records", "training_runs")}
        referenced = {cid for cid in rep._referenced(blocks) if cid in sources.cells}
        assert referenced and referenced <= set(payload["records"]), (name, sorted(referenced - set(payload["records"]))[:3])
        for cid, rows in payload["records"].items():
            assert rows and all(sorted(row) == sorted(rep.EPISODE_FIELDS) for row in rows), (name, cid)
        assert all(run["clock"] == cr.CLOCK_LABEL for runs in payload["training_runs"].values() for run in runs), name


def test_the_q8_originals_on_the_joined_old_cells_are_the_committed_statements(
        artifacts: dict[str, dict[str, Any]]) -> None:
    """Read from docs/data directly: P4.6's and P4.7's P1 under att_ours, and every behaviour comparison with IQL."""
    def load(name: str) -> dict[str, Any]:
        return json.loads((REPO / "docs" / "data" / f"{name}.json").read_text(encoding="utf-8"))

    payload = artifacts["p9_1_iql_targets.json"]
    for task in ("p4_6", "p4_7"):
        committed = load(f"{task}_grid")["predictions"]["P1"]
        assert payload["p1_rescore"][task]["original"]["att_ours"] == committed, task
    p4_4, p4_6, p4_7 = load("p4_4_baselines"), load("p4_6_grid"), load("p4_7_grid")
    for entry in payload["iql_behaviour_comparisons"]["comparisons"]:
        if entry["artifact"] == "p4_4":
            committed = p4_4["behaviour_policy_comparisons"]["mappo1000_vs_iql"]
        else:
            source = p4_6 if entry["artifact"] == "p4_6" else p4_7
            tier = entry["tier"]
            committed = next(c for c in source["behaviour_comparisons"]
                             if (c["left_arm"], c["right_arm"]) == (f"iql@{tier}", f"behaviour@{tier}"))
        assert entry["original"]["att_ours"] == committed, (entry["artifact"], entry["tier"])


def test_a_new_against_old_contrast_is_the_fixtures_own_arithmetic(artifacts: dict[str, dict[str, Any]]) -> None:
    """``iql_t - iql`` on hz1x1's random tier under att_engine, recomputed from the raw committed JSON and the fixture's
    formula: per draw the mean over the five seeds in seed order (numpy), then the mean of the differences."""
    p4_6 = json.loads((REPO / "docs" / "data" / "p4_6_grid.json").read_text(encoding="utf-8"))
    old = {(int(e["seed"]), int(e["draw_id"])): float(e["att_horizon"]) - fx.ENGINE_SHIFT_OLD
           for e in p4_6["episodes"] if e["arm"] == "iql@random"}
    seeds = sorted(int(s) for s in TRAINING_SEEDS)
    differences = []
    for draw in sorted(int(d) for d in HELD_OUT_DRAWS):
        new = np.mean([fx.new_episode("hz1x1", "iql_t@random", s, draw)["att_engine"] for s in seeds])
        differences.append(float(new) - float(np.mean([old[(s, draw)] for s in seeds])))
    expected = float(np.asarray(differences, dtype=np.float64).mean())
    entry = next(e for e in artifacts["p9_1_iql_targets.json"]["hz1x1"]["tiers"] if e["tier"] == "random")
    assert (entry["left"], entry["right"]) == ("p9_1:hz1x1:iql_t@random", "p4_6:iql@random")
    assert entry["by_definition"]["att_engine"]["mean_difference"] == expected


def test_the_planted_pause_is_flagged_on_its_run_and_on_no_other(artifacts: dict[str, dict[str, Any]]) -> None:
    flagged = {}
    for payload in artifacts.values():
        for cid, runs in payload["training_runs"].items():
            for run in runs:
                if "paused_during_run" in run:
                    flagged[(cid, int(run["seed"]))] = (bool(run["paused_during_run"]), run["trace_covered"])
    scenario, tier, arm, seed = fx.PAUSED_RUN
    assert flagged[(f"p9_1:{scenario}:{arm}@{tier}", seed)] == (True, True)
    assert sum(paused for paused, _ in flagged.values()) == 1
    assert all(covered for _, covered in flagged.values())


def test_the_planted_repeat_is_p3as_only_failure(artifacts: dict[str, dict[str, Any]]) -> None:
    p3a = artifacts["p9_1_random_fresh.json"]["p3a"]
    scenario, key, seed, draws = fx.REPEATED
    assert (p3a[scenario]["verdict"], p3a[scenario]["repeated_episodes"]) == ("FAILED", 2)
    assert p3a[scenario]["repeated"] == [[seed, d] for d in draws]
    assert (p3a["grid4x4"]["verdict"], p3a["grid4x4"]["repeated_episodes"]) == ("HELD", 0)


def test_the_report_rebuilds_byte_for_byte_from_the_tree(tree: fx.FixtureTree, artifacts: dict[str, dict[str, Any]],
                                                          tmp_path: Path) -> None:
    """T-regress on the fixture: a second, independent load gives every artifact's bytes again, and the CLI writes
    exactly those bytes and then leaves them unchanged."""
    texts = {name: rep.artifact_text(payload) for name, payload in artifacts.items()}
    again = _load(tree)
    assert {name: rep.artifact_text(builder(again)) for name, builder in rep.ARTIFACTS.items()} == texts
    argv = ["--repo-root", str(REPO), "--output-root", str(tree.roots.output_root), "--corpus-root", str(CORPUS),
            "--draws-root", str(DRAWS), "--d4rl", str(tree.d4rl_path), "report", "--out-dir", str(tmp_path)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tst, "git_block", lambda: dict(GIT))
        assert rep.main(argv) == 0
        assert {p.name: p.read_text(encoding="utf-8") for p in tmp_path.iterdir()} == texts
        assert rep.main(argv) == 0
    assert {p.name: p.read_text(encoding="utf-8") for p in tmp_path.iterdir()} == texts


def test_the_iql_diagnostics_beside_the_d4rl_section_are_every_iql_family_run_and_both_d4rl_arms(
        artifacts: dict[str, dict[str, Any]]) -> None:
    """Q18 and Q19 (``BRIEF_46`` F.2-3, S.2) on the full-size tree: the 115 IQL-family traffic runs' arm, target mode and
    three values are the fixture's own, found by run number; the D4RL rows are the committed P8.3 IQL cells, read
    straight from ``docs/data``, and the fixture's re-run; no other artifact carries the block."""
    block = artifacts["p9_1_iql_targets.json"]["iql_diagnostics"]
    modes = {"iql_t": "off", "iql_m": "off", "iql": "active", "iql_100k": "active"}
    expected = [((cell.scenario, cell.tier, cell.arm, int(seed), modes[cell.arm]), fx.iql_diagnostics_of(number - 1))
                for number, (cell, seed) in enumerate(cr.training_runs(), start=1) if cell.arm in modes]
    got = [((r["scenario"], r["tier"], r["arm"], r["seed"], r["target_dropout"]),
            (r["final_q_loss"], r["final_v_loss"], r["awr_weight_near_zero_fraction"]))
           for r in block["traffic_iql_family"]["runs"]]
    assert len(got) == 115 and got == expected
    committed = json.loads((REPO / "docs" / "data" / "p8_3_d4rl_calibration.json").read_text(encoding="utf-8"))
    iql = sorted((c for c in committed["cells"] if c["arm"] == "iql"), key=lambda c: int(c["seed"]))
    rows = [(r["arm"], r["seed"], r["final_q_loss"], r["final_v_loss"], r["awr_weight_near_zero_fraction"])
            for r in block["d4rl"]["rows"]]
    assert rows == [("iql", int(c["seed"]), c["diagnostics"]["final_q_loss"], c["diagnostics"]["final_v_loss"],
                     c["diagnostics"]["awr_weight_near_zero_fraction"]) for c in iql] + [
        ("iql_t", int(s), *fx.d4rl_diagnostics_of(i)) for i, s in enumerate(TRAINING_SEEDS)]
    assert block["label"] == rep.DIAGNOSTICS_LABEL
    assert [name for name, payload in artifacts.items() if "iql_diagnostics" in payload] == ["p9_1_iql_targets.json"]


def test_every_artifact_names_the_sessions_and_the_evaluations_commit_never_a_route_i_runtime_commit(
        artifacts: dict[str, dict[str, Any]]) -> None:
    """Q20 (S.2) on the full-size tree: every artifact carries the captures' sessions as the fixture wrote them and
    the one commit of its three evaluate sessions; ``p9_1_iql_targets.json`` names the D4RL session beside its section.
    hz1x1's route (i) files carry a planted main-tree HEAD in ``runtime.git_commit``, which no artifact contains."""
    table = [(stage, fx.stamp_of(start), commit, tree)
             for (stage, _, commit, tree), start in zip(fx.SESSIONS, fx.session_starts())]
    evaluations = [f"evaluate {stamp}" for stage, stamp, _, _ in table if stage == "evaluate"]
    for name, payload in artifacts.items():
        assert [(s["stage"], s["stamp"], s["code_commit"], s["run_tree"]) for s in payload["sessions"]] == table, name
        evaluation = payload["evaluation_code_commit"]
        assert (evaluation["code_commit"], evaluation["run_tree"], evaluation["sessions"]) == (
            "c58afb8", "/trees/run-d", evaluations), name
        assert fx.ROUTE_I_RUNTIME_COMMIT not in rep.artifact_text(payload), name
    d4rl = artifacts["p9_1_iql_targets.json"]["d4rl_session"]
    assert (d4rl["code_commit"], d4rl["run_tree"], d4rl["sessions"]) == (
        "cb61ee7", "/trees/run-c", [f"d4rl {stamp}" for stage, stamp, _, _ in table if stage == "d4rl"])
    assert [name for name, payload in artifacts.items() if "d4rl_session" in payload] == ["p9_1_iql_targets.json"]


def test_every_new_run_names_its_sessions_code_commit_and_run_tree(artifacts: dict[str, dict[str, Any]]) -> None:
    """G.2 on the full-size tree: each new run's code commit, run tree and session are its training session's, found
    here by run number from the fixture's session table (the report finds them by time, from the captures)."""
    expected = {}
    for number, (cell, seed) in enumerate(cr.training_runs(), start=1):
        stage, stamp, commit, tree = fx.training_session_of(number)
        expected[(rep.new_cell_id(cell), int(seed))] = (commit, tree, f"{stage} {stamp}")
    seen = {}
    for name, payload in artifacts.items():
        assert payload["code_commit_rule"] == rep.CODE_COMMIT_RULE, name
        for cid, runs in payload["training_runs"].items():
            for run in runs:
                if cid.startswith("p9_1:"):
                    seen[(cid, int(run["seed"]))] = (run["code_commit"], run["run_tree"], run["session"])
    assert {key: expected[key] for key in seen} == seen
    assert {commit for commit, _, _ in seen.values()} == {"cb61ee7", "c58afb8"}
    assert seen[("p9_1:hz1x1:iql_t@random", 303)][2] == f"campaign {fx.stamp_of(fx.T0 - fx.SESSION_LEAD)}"
