"""P9.1's report, part 3: the loaders that read the run directories and the committed records into a ``Sources``
(``offline/corrective_report.py``; ``BRIEF_46`` 3.4 ``old-cells``' seconds and steps, 3.5).

Formats: the committed training records (``p4_training.json`` ``seeds``; ``p4_4/6/7_training.json`` ``runs``;
``p5_2b_iql_correction.json`` ``training.runs``), the grid checkpoints' ``provenance``, P3a's ``p9.1-actions/1.0``, the
corpus ``.npz`` (``ix_ids``, ``ix<i>_action``), the D4RL re-run's output.  Alignment: a run is keyed by its cell id and
seed; a schedule's digest is ``nortg_decomposition.action_sequence_sha256`` over the int64 actions in ``ix_ids`` order
(contract: ``ix_ids[i]`` owns ``action[i]``).  Every expected value is read here from the committed file by the test's
own route.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import offline.corrective_report as rep
import offline.corrective_round as cr
import offline.tier_statements as tst

REPO = Path(__file__).resolve().parents[1]


def _committed(name: str, key: str) -> list[dict[str, Any]]:
    payload = json.loads((REPO / "docs" / "data" / f"{name}.json").read_text())
    return payload[key] if key != "training.runs" else payload["training"]["runs"]


def test_an_old_hz1x1_cell_is_named_by_its_artifact() -> None:
    assert rep.hz1x1_old_id("docs/data/p4_4_baselines.json", "madt") == "p4_4:madt"
    assert rep.hz1x1_old_id("docs/data/p4_6_grid.json", "iql@maxpressure") == "p4_6:iql@maxpressure"
    assert rep.hz1x1_old_id("docs/data/p4_7_grid.json", "bc@mix50") == "p4_7:bc@mix50"


@pytest.mark.parametrize(("artifact", "arm", "record", "key", "match"), [
    ("docs/data/p4_4_baselines.json", "madt", "p4_training", "seeds", lambda r: True),
    ("docs/data/p4_4_baselines.json", "iql", "p4_4_training", "runs", lambda r: r["method"] == "iql"),
    ("docs/data/p4_6_grid.json", "iql@maxpressure", "p4_6_training", "runs",
     lambda r: (r["tier"], r["method"]) == ("maxpressure", "iql")),
    ("docs/data/p4_7_grid.json", "bc_top10@mix50", "p4_7_training", "runs",
     lambda r: (r["tier"], r["method"]) == ("mix50", "bc_top10")),
])
def test_an_old_hz1x1_cells_runs_are_its_committed_records_seconds_and_steps(artifact: str, arm: str, record: str,
                                                                             key: str, match: Any) -> None:
    expected = sorted(((int(r["seed"]), float(r["seconds"]), int(r["gradient_steps"]), bool(r["plateaued"]))
                       for r in _committed(record, key) if match(r)))
    runs = rep.hz1x1_old_runs(REPO, artifact, arm)
    assert [(r["seed"], r["seconds"], r["gradient_steps"], r["plateaued"]) for r in runs] == expected
    assert len(runs) == 5
    assert all(r["clock"] == cr.CLOCK_LABEL and r["source"] == f"docs/data/{record}.json" for r in runs)
    assert all(r["cell"] == rep.hz1x1_old_id(artifact, arm) and r["window_means"] for r in runs)


def test_a_behaviour_cell_has_no_training_run() -> None:
    assert rep.hz1x1_old_runs(REPO, "docs/data/p4_6_grid.json", "behaviour@random") == []


def test_the_corrected_grid_iql_cells_runs_are_p5_2bs_recorded_seconds() -> None:
    expected = sorted((int(r["seed"]), float(r["seconds"]), int(r["gradient_steps"]))
                      for r in _committed("p5_2b_iql_correction", "training.runs"))
    runs = rep.p5_2b_runs(REPO)
    assert [(r["seed"], r["seconds"], r["gradient_steps"]) for r in runs] == expected
    assert all(r["cell"] == tst.CORRECTED_IQL and r["source"] == "docs/data/p5_2b_iql_correction.json" for r in runs)


def _p3a_file(roots: cr.Roots, cell: cr.Cell) -> None:
    path = Path(roots.runs_root) / "p3a" / cell.scenario / f"{cell.arm}@{cell.tier}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format_version": cr.ACTIONS_FORMAT_VERSION, "arm": cell.key, "episodes": [
        {"seed": 101, "draw_id": 1000, "att_ours": 1.0, "n_decisions": 360, "action_sequence_sha256": cell.key,
         "per_intersection_sha256": [cell.key]}]}))


def test_p3a_records_are_keyed_by_the_ids_the_artifacts_read(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    roots = cr.Roots(repo_root=REPO, output_root=output, corpus_root=tmp_path, draws_root=tmp_path)
    for cell in cr.p3a_cells():
        _p3a_file(roots, cell)
    rows = rep.p3a_records(roots)
    assert set(rows) == {"p9_1:hz1x1:dt@random_fresh", "p9_1:grid4x4:dt_nomix@random_fresh",
                         "p9_1:hz1x1:behaviour@random_fresh", "p9_1:grid4x4:behaviour@random_fresh",
                         "p4_6:behaviour@random", tst.ANCHOR_CELL}
    assert rows["p4_6:behaviour@random"][0]["action_sequence_sha256"] == "behaviour@random"
    assert rows[tst.ANCHOR_CELL][0]["action_sequence_sha256"] == "random@mappo1000"


def test_a_tiers_schedule_digests_count_what_a27s_registered_check_counts(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    import offline.a27_arms as a27_arms
    from tests import p5_2b_fixtures as fx

    import numpy as np

    directory = fx.write_synthetic_tier(tmp_path)
    digests = rep.schedule_digests(directory)
    counts = a27_arms.distinct_schedules(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    with np.load(directory / manifest["episodes"][0]["filename"]) as first:
        order = [str(v) for v in first["ix_ids"].tolist()]
    assert digests["ix_ids"] == order and len(order) == 16
    assert len(digests["joint"]) == counts["joint_distinct"]
    assert [len(s) for s in digests["per_intersection"]] == [counts["per_intersection_distinct"][ix] for ix in order]


def test_the_d4rl_section_is_the_reruns_comparison_with_the_files_digest(tmp_path: Path) -> None:
    import hashlib

    payload = {"format_version": "p8.3-d4rl/1.0", "cells": [{"arm": "iql_t", "seed": 101, "training_seconds": 99.5,
                                                             "wall_seconds": 120.25, "gradient_steps": 40000,
                                                             "episode_returns": [1.0]}],
               "compare_with": {"path": "docs/data/p8_3_d4rl_calibration.json", "sha256": "a" * 64,
                                "arms": ["bc", "bc_top10", "iql", "iql_t"], "contrasts": {"bc - iql_t": {"x": 1}},
                                "per_seed_return": {"iql_t": {"101": 1.0}}, "registered_contrasts_reproduced": True}}
    path = tmp_path / "p8_3_train_k64_40000_iql_t.json"
    path.write_text(json.dumps(payload))
    section = rep.d4rl_section(path)
    assert section["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert section["comparison"] == payload["compare_with"]
    assert section["runs"] == [{"arm": "iql_t", "seed": 101, "seconds": 99.5, "wall_seconds": 120.25,
                                "gradient_steps": 40000, "clock": cr.CLOCK_LABEL}]
    payload.pop("compare_with")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="compare-with"):
        rep.d4rl_section(path)


@pytest.mark.skipif(not (Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT", "/nonexistent")) / "p5_2" / "checkpoints").is_dir(),
                    reason="the grid checkpoints live under the main tree's output/ (set RLTRAFFIC_OUTPUT_ROOT)")
def test_a_registered_grid_baselines_runs_carry_the_checkpoints_steps_and_no_seconds() -> None:
    output = Path(os.environ["RLTRAFFIC_OUTPUT_ROOT"])
    spec = next(s for s in tst.CELL_SPECS if s.cell_id == "p5_2:bc@random")
    runs = rep.grid_old_runs(output, spec)
    assert [r["seed"] for r in runs] == [101, 202, 303, 404, 505]
    assert all(r["gradient_steps"] == 40000 and r["seconds"] is None and "no committed record" in r["seconds_note"]
               for r in runs)
    assert all(r["source"].startswith("output/p5_2/checkpoints/grid4x4_random_bc_seed") for r in runs)
    assert rep.grid_old_runs(output, next(s for s in tst.CELL_SPECS if s.method == "random")) == []


# ----------------------------------------------------------------------
# The inputs' digest, the CLI, and T-regress
# ----------------------------------------------------------------------


def test_a_directorys_digest_is_the_sha256sum_line_construction_recomputed_independently(tmp_path: Path) -> None:
    import hashlib

    root = tmp_path / "output"
    (root / "p9_1_runs" / "eval_ii" / "hz1x1").mkdir(parents=True)
    (root / "p9_1_runs" / "checkpoints").mkdir(parents=True)
    files = {"p9_1_runs/eval_ii/hz1x1/cell_b.json": b"{}", "p9_1_runs/eval_ii/hz1x1/cell_a.json": b"[1]",
             "p9_1_runs/training.json": b"x"}
    for relative, data in files.items():
        (root / relative).write_bytes(data)
    (root / "p9_1_runs" / "checkpoints" / "skip.pt").write_bytes(b"weights")
    block = rep.directory_digest(root, "p9_1_runs", exclude=("checkpoints",))
    lines = "".join(f"{hashlib.sha256(files[r]).hexdigest()}  {r}\n" for r in sorted(files))
    assert block == {"directory": "output/p9_1_runs", "files": 3, "excluded": ["checkpoints"],
                     "sha256": hashlib.sha256(lines.encode()).hexdigest(),
                     "construction": rep.DIGEST_CONSTRUCTION}


def _fake_sources() -> rep.Sources:
    return rep.Sources(cells={}, records={}, runs={}, declarations={}, grid_books={}, p3a={}, schedules={}, d4rl=None,
                       registered={}, inputs=({"file": "x", "sha256": "0" * 64},),
                       git={"commit": "f" * 40, "dirty": False, "code_root": "/x"})


def test_the_report_cli_writes_the_five_artifacts_once_and_refuses_a_different_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []
    for name in rep.ARTIFACTS:
        monkeypatch.setitem(rep.ARTIFACTS, name, (lambda n: (lambda src: built.append(n) or {
            "git": dict(src.git), "name": n}))(name))
    monkeypatch.setattr(rep, "load_sources", lambda roots, *, d4rl_path: _fake_sources())
    out = tmp_path / "data"
    out.mkdir()
    argv = ["--repo-root", str(REPO), "--output-root", str(tmp_path), "--corpus-root", str(tmp_path), "--draws-root",
            str(tmp_path), "--d4rl", str(tmp_path / "d.json"), "report", "--out-dir", str(out)]
    assert rep.main(argv) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(rep.ARTIFACTS) == [
        "p9_1_budget.json", "p9_1_exposure_parity.json", "p9_1_iql_targets.json", "p9_1_mixture.json",
        "p9_1_random_fresh.json"]
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert rep.main(argv) == 0
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    (out / "p9_1_mixture.json").write_text("{}")
    assert rep.main(argv) == 2
    assert (out / "p9_1_mixture.json").read_text() == "{}"


T_REGRESS_READY = all((REPO / "docs" / "data" / name).is_file() for name in (
    "p9_1_exposure_parity.json", "p9_1_iql_targets.json", "p9_1_random_fresh.json", "p9_1_mixture.json",
    "p9_1_budget.json")) and (Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT", "/nonexistent")) / "p9_1_runs").is_dir()


@pytest.mark.skipif(not T_REGRESS_READY, reason="T-regress needs the committed P9.1 artifacts and the campaign's run "
                                                "directories (RLTRAFFIC_OUTPUT_ROOT, RLTRAFFIC_CORPUS_V11, "
                                                "RLTRAFFIC_D4RL_OUT)")
@pytest.mark.parametrize("name", ["p9_1_exposure_parity.json", "p9_1_iql_targets.json", "p9_1_random_fresh.json",
                                  "p9_1_mixture.json", "p9_1_budget.json"])
def test_t_regress_every_artifact_rebuilds_byte_for_byte_from_the_run_directories(name: str, tmp_path: Path) -> None:
    output = Path(os.environ["RLTRAFFIC_OUTPUT_ROOT"])
    roots = cr.Roots(repo_root=REPO, output_root=output, corpus_root=Path(os.environ["RLTRAFFIC_CORPUS_V11"]),
                     draws_root=output.parent / "scenarios" / "draws")
    src = rep.load_sources(roots, d4rl_path=Path(os.environ["RLTRAFFIC_D4RL_OUT"]) / rep.D4RL_OUTPUT_NAME)
    rebuilt = rep.write_artifact(tmp_path / name, rep.ARTIFACTS[name](
        rep.Sources(**{**src.__dict__, "git": json.loads((REPO / "docs" / "data" / name).read_text())["git"]})))
    assert rebuilt.read_bytes() == (REPO / "docs" / "data" / name).read_bytes()


# ----------------------------------------------------------------------
# Added after the C6d mutation table (L4, L7 survived: the committed records and the synthetic tier could not tell them)
# ----------------------------------------------------------------------


def test_a_training_record_without_exactly_five_runs_for_the_cell_is_refused(tmp_path: Path) -> None:
    """L4: a run missing from the committed record is refused, never reported as a cell of four."""
    record = json.loads((REPO / "docs/data/p4_6_training.json").read_text())
    record["runs"] = [r for r in record["runs"]
                      if not ((r["tier"], r["method"], int(r["seed"])) == ("maxpressure", "iql", 303))]
    target = tmp_path / "docs" / "data" / "p4_6_training.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="exactly the five runs"):
        rep.hz1x1_old_runs(tmp_path, "docs/data/p4_6_grid.json", "iql@maxpressure")


def test_each_intersections_schedules_are_its_own_columns_digests(tmp_path: Path) -> None:
    """L7: the per-intersection set holds the digests of that intersection's column, recomputed from the raw .npz."""
    import numpy as np

    from offline.nortg_decomposition import action_sequence_sha256
    from tests import p5_2b_fixtures as fx

    directory = fx.write_synthetic_tier(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected: list[set[str]] = [set() for _ in range(16)]
    for entry in manifest["episodes"]:
        with np.load(directory / entry["filename"]) as arrays:
            for i in range(16):
                expected[i].add(action_sequence_sha256(np.asarray(arrays[f"ix{i}_action"], dtype=np.int64)[:, None]))
    assert rep.schedule_digests(directory)["per_intersection"] == expected


# ----------------------------------------------------------------------
# A suspension inside a run (the pre-flight's own first iql_m run spanned one: GPU trace gap 23:39 -> 10:23)
# ----------------------------------------------------------------------


def test_a_gap_in_the_trace_is_found_and_a_run_that_spans_it_is_flagged(tmp_path: Path) -> None:
    trace = tmp_path / "logs" / "20261016T000000Z" / "gpu_trace.csv"
    trace.parent.mkdir(parents=True)
    trace.write_text("1000.0,700,3\n1020.0,7000,97\n1040.5,7000,96\n40000.0,7000,98\n40020.0,7000,97\n")
    gaps = rep.trace_gaps([trace], threshold=60.0)
    assert gaps == [{"from_unix": 1040.5, "to_unix": 40000.0, "seconds": 38959.5,
                     "trace": "logs/20261016T000000Z/gpu_trace.csv"}]
    spans = rep.trace_spans([trace])
    runs = [{"seed": 101, "started_unix": 1010.0, "finished_unix": 40010.0, "seconds": 39000.0},
            {"seed": 202, "started_unix": 1001.0, "finished_unix": 1039.0, "seconds": 38.0},
            {"seed": 303, "started_unix": 50000.0, "finished_unix": 50010.0, "seconds": 10.0}]
    flagged = rep.flag_paused_runs(runs, gaps, spans)
    assert [r["paused_during_run"] for r in flagged] == [[gaps[0]], [], []]
    assert [r["trace_covered"] for r in flagged] == [True, True, False]


def test_p5_3bs_old_tier_numbers_are_copied_from_its_committed_artifact() -> None:
    """A27(d)'s beside-table sits next to P5.3b's old-tier numbers; they are read from P5.3b's artifact, not typed."""
    import hashlib

    path = REPO / "docs" / "data" / "p5_3b_nortg.json"
    identity = json.loads(path.read_text())["mechanism"]["action_identity"]
    block = rep.p5_3b_old_tier(REPO)
    assert block["source"] == "docs/data/p5_3b_nortg.json mechanism.action_identity"
    assert block["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert block["n_distinct_action_sequences"] == {
        key: value for key, value in identity["n_distinct_action_sequences"].items() if key.startswith("dt@random seed ")}
    assert set(block["n_distinct_action_sequences"].values()) == {1} and len(block["n_distinct_action_sequences"]) == 5
    assert block["attractor"] == identity["attractor"]
