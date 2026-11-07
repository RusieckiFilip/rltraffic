"""P9.1's pre-flight (``offline/corrective_round.py``; ``BRIEF_46`` 5, G2): measured, never estimated, written only under
a scratch directory -- the cost of one matched BC and one matched IQL run on the grid through the campaign's own
``train_run``; one registered cell reproduced through BOTH routes against its committed ``att_ours`` under ``==``; the
A27 grid path end to end on the trained runs; the mixture's declaration; the cell count by arm.

Format: ``p9.1-preflight/1.0``.  Everything that trains or rolls is stubbed here; the order, the roots and the refusals
are what is pinned.  The real pre-flight is run on the machine and its record committed as evidence (G2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import offline.corrective_round as cr


def test_the_cell_count_by_arm_is_the_manifests() -> None:
    counts = cr.cell_count_by_arm()
    assert sum(counts.values()) == 255
    assert counts["grid4x4:iql_m"] == 25 and counts["hz1x1:iql_t"] == 50 and counts["grid4x4:dt_nomix"] == 5
    assert counts["hz1x1:dt_100k"] == 10 and counts["grid4x4:bc_top10_perix_m"] == 25


def test_a_registered_records_checkpoints_are_repointed_only_when_their_digests_match(tmp_path: Path) -> None:
    import torch

    from agent.OfflineBaselines import canonical_state_dict_digest

    main = tmp_path / "main" / "p4_6" / "checkpoints"
    main.mkdir(parents=True)
    model = {"w": torch.ones(2)}
    torch.save({"model": model}, main / "maxpressure_iql_seed101.pt")
    record = {"declared_gradient_steps": 40000, "runs": [
        {"tier": "maxpressure", "method": "iql", "seed": 101,
         "checkpoint": "/gone/output/p4_6/checkpoints/maxpressure_iql_seed101.pt",
         "canonical_digest": canonical_state_dict_digest(model)}]}
    repointed = cr.repoint_training_record(record, tmp_path / "main", tier="maxpressure", method="iql")
    assert repointed["runs"][0]["checkpoint"] == str(main / "maxpressure_iql_seed101.pt")
    record["runs"][0]["canonical_digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        cr.repoint_training_record(record, tmp_path / "main", tier="maxpressure", method="iql")


def test_route_comparison_refuses_any_episode_that_differs() -> None:
    left = {("hz1x1", "a@t", 101, 1000): 103.25, ("hz1x1", "a@t", 101, 1001): 104.0}
    assert cr.assert_equal_episodes(left, dict(left), what="x") == 2
    with pytest.raises(ValueError, match="1 of 2"):
        cr.assert_equal_episodes(left, {**left, ("hz1x1", "a@t", 101, 1001): 104.0000001}, what="x")
    with pytest.raises(ValueError, match="cover"):
        cr.assert_equal_episodes(left, {("hz1x1", "a@t", 101, 1000): 103.25}, what="x")


def test_the_preflight_writes_only_under_its_scratch_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "main_output"
    output.mkdir()
    roots = cr.Roots(repo_root=Path(__file__).resolve().parents[1], output_root=output, corpus_root=tmp_path / "corpus",
                     draws_root=tmp_path / "draws")
    scratch = tmp_path / "scratch"
    order: list[str] = []
    monkeypatch.setattr(cr, "_preflight_matched_runs", lambda r, s, device: order.append("train") or {"bc_m": 1.0})
    monkeypatch.setattr(cr, "_preflight_registered_cell", lambda r, s: order.append("registered") or {"episodes": 500})
    monkeypatch.setattr(cr, "_preflight_a27_grid_path", lambda r, s, draws: order.append("a27") or {"episodes": 10})
    declared_under: list[Path] = []
    monkeypatch.setattr(cr, "declare_mixture", lambda r: order.append("declare") or declared_under.append(
        Path(r.output_root)) or {"measured": {"weaker_source_kept": 6}})
    record = cr.preflight(roots, scratch, device="cpu")
    assert order == ["train", "registered", "a27", "declare"]
    assert declared_under == [scratch / "output"], "the mixture is declared into the scratch tree, never the main one"
    assert record["format_version"] == cr.PREFLIGHT_FORMAT_VERSION and record["cell_count_by_arm"]
    assert (scratch / "preflight.json").is_file()
    assert sorted(p.name for p in output.iterdir()) == [], "nothing is written under the main output tree"
    assert json.loads((scratch / "preflight.json").read_text())["mixture_declaration"]["measured"]["weaker_source_kept"] == 6


def test_the_a27_grid_path_reads_the_main_trees_protected_reuse_root_and_trains_nothing_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route (i) on the grid runs ``tier_sweep evaluate`` with ``--seeds 101`` before the subcommand, its checkpoints
    and its cell file in the SCRATCH tree, and the P5.1 reuse root and checksums of the MAIN tree (protected: the CLI
    refuses to write there) -- not a scratch copy that does not exist."""
    import offline.tier_sweep as ts

    output = tmp_path / "main_output"
    output.mkdir()
    roots = cr.Roots(repo_root=Path(__file__).resolve().parents[1], output_root=output, corpus_root=tmp_path / "corpus",
                     draws_root=tmp_path / "draws")
    seen: list[list[str]] = []
    monkeypatch.setattr(ts, "main", lambda argv: seen.append(list(argv)) or 1)
    with pytest.raises(RuntimeError, match="route \\(i\\) failed"):
        cr._preflight_a27_grid_path(roots, tmp_path / "scratch", (1000, 1001))
    argv = seen[0]
    assert argv[argv.index("--reuse-root") + 1] == str(output / "p5_1")
    assert argv[argv.index("--checksums") + 1] == str(output / "SHA256SUMS_p5_1.txt")
    assert argv[argv.index("--checkpoint-dir") + 1].startswith(str(tmp_path / "scratch"))
    assert argv[argv.index("--work-dir") + 1].startswith(str(tmp_path / "scratch"))
    assert argv.index("--seeds") < argv.index("evaluate") and argv[argv.index("--seeds") + 1] == "101"
    assert argv[argv.index("evaluate"):] == ["evaluate", "--method", "bc_m"]
    assert list(output.iterdir()) == []
