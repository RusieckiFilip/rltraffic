"""P5.3c C1 (``BRIEF_42`` §3 C1(i), §4 T-rows; ``PREREGISTRATION`` A26(c) as corrected by A26.1): the reference rows.

A26(c)'s precondition: the per-draw ``att_engine`` rows of P4's five K = 20 checkpoints -- 5 seeds x 100 held-out
draws -- exist only in P8.4b's gitignored cells, so they are extracted into ``docs/data/p4_k20_att_engine_rows.json``
BEFORE any training, with every source cell's digest, and the campaign's reference gate compares against THAT file
under ``==``.  The extraction is ``python -P -m offline.context_sweep extract-reference-rows``.

This file's C1a half drives the extraction on a FAKE output tree (``tests/p5_3c_fixtures.py``: five stand-in
checkpoint files named by digest in a fake ``p4_gate.json``, 500 fake P8.4b cells, their campaign manifest), so every
refusal is exercised without the real, gitignored sources.  Each refusal is asserted to write NOTHING.

*Named mutation (``BRIEF_42`` §4):* the duplicate (seed, draw) check removed -> the duplicate test dies (its refusal
must name the duplicate; the name-against-content check that would still catch the file raises other words).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

import pytest

import offline.context_sweep as cs
from tests.p5_3c_fixtures import (
    REFERENCE_DRAWS,
    REFERENCE_SEEDS,
    ReferenceTree,
    build_reference_tree,
    reference_cell,
    reference_cell_name,
    reference_values,
    repository_is_shallow,
)

FIXED_CODE = {"code_commit": "c" * 40, "code_dirty": False}


@pytest.fixture(autouse=True)
def _fixed_code_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """The extraction records the module tree's commit; the tests pin it so payloads are comparable."""
    monkeypatch.setattr(cs, "_code_provenance", lambda: dict(FIXED_CODE), raising=True)


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): (None if path.is_dir() else path.read_bytes())
        for path in sorted(root.rglob("*"))
    }


def _expected_rows(tree: ReferenceTree) -> list[dict[str, Any]]:
    """The rows recomputed by this test's own route: the fixture's generator and a fresh sha256 of each file."""
    rows = []
    for seed in REFERENCE_SEEDS:
        for draw in REFERENCE_DRAWS:
            engine, ours = reference_values(seed, draw)
            path = tree.cell_path(seed, draw)
            rows.append(
                {
                    "seed": seed,
                    "draw_id": draw,
                    "att_engine": engine,
                    "att_ours": ours,
                    "source": reference_cell_name(seed, draw),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# The payload
# ----------------------------------------------------------------------------------------------------------------------


def test_the_payload_carries_all_500_rows_sorted_with_both_definitions_and_every_digest(tmp_path: Path) -> None:
    tree = build_reference_tree(tmp_path)
    before = _snapshot(tmp_path)

    payload = cs.reference_rows_payload(output_root=tree.output_root, data_dir=tree.data_dir)

    assert _snapshot(tmp_path) == before, "building the payload must write nothing"
    assert payload["format_version"] == cs.REFERENCE_ROWS_FORMAT_VERSION == "p5.3c-reference-rows/1.0"
    assert payload["rows"] == _expected_rows(tree)
    assert payload["n_rows"] == 500 == len(payload["rows"])
    assert payload["seeds"] == [101, 202, 303, 404, 505]
    assert payload["draw_ids"] == list(range(1000, 1100))
    assert payload["definitions"] == ["att_engine", "att_ours"]
    assert payload["primary_definition"] == "att_engine"
    assert payload["engine_seed"] == 1000
    gate = json.loads((tree.data_dir / "p4_gate.json").read_text(encoding="utf-8"))
    for seed in REFERENCE_SEEDS:
        entry = payload["checkpoints"][str(seed)]
        recomputed = hashlib.sha256((tree.output_root / "p4_dt" / f"dt_seed{seed}.pt").read_bytes()).hexdigest()
        assert entry["sha256"] == recomputed == gate["checkpoints"][str(seed)]["sha256"]
        assert entry["path"] == f"output/p4_dt/dt_seed{seed}.pt"
    assert payload["subject"]["arm"] == "dt@mappo1000"
    assert payload["subject"]["context_length"] == 20
    source = payload["source"]
    assert source["campaign_provenance"]["git_commit"] == "3" * 40
    assert source["campaign_provenance"]["code_dirty"] is True
    manifest = tree.cells / "campaign_manifest.json"
    assert source["campaign_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert payload["extraction"]["code_commit"] == "c" * 40
    assert payload["extraction"]["code_dirty"] is False


# ----------------------------------------------------------------------------------------------------------------------
# The refusals: every one BEFORE any write, and each names its cause
# ----------------------------------------------------------------------------------------------------------------------


def _break_missing(tree: ReferenceTree) -> None:
    tree.cell_path(303, 1042).unlink()


def _break_duplicate(tree: ReferenceTree) -> None:
    # The file of (101, 1001) carries (101, 1000)'s content: (101, 1000) is now claimed by two files.
    tree.write_cell(101, 1001, reference_cell(101, 1000, output_root=tree.output_root))


def _break_swap(tree: ReferenceTree) -> None:
    # Two files exchange contents: no pair is claimed twice, but each names a pair its file name does not.
    first = reference_cell(101, 1000, output_root=tree.output_root)
    second = reference_cell(202, 1000, output_root=tree.output_root)
    tree.write_cell(101, 1000, second)
    tree.write_cell(202, 1000, first)


def _break_undeclared(tree: ReferenceTree) -> None:
    extra = reference_cell(606, 1000, output_root=tree.output_root)
    (tree.cells / reference_cell_name(606, 1000)).write_text(json.dumps(extra) + "\n", encoding="utf-8")


def _break_checkpoint(tree: ReferenceTree) -> None:
    (tree.output_root / "p4_dt" / "dt_seed404.pt").write_bytes(b"another checkpoint\n")


def _mutate_cell(seed: int, draw: int, change: Callable[[dict[str, Any]], None]) -> Callable[[ReferenceTree], None]:
    def apply(tree: ReferenceTree) -> None:
        payload = reference_cell(seed, draw, output_root=tree.output_root)
        change(payload)
        tree.write_cell(seed, draw, payload)

    return apply


def _not_reproduced(payload: dict[str, Any]) -> None:
    payload["reproduces_committed"] = False


def _other_seed_policy(payload: dict[str, Any]) -> None:
    payload["policy_source"]["checkpoint"] = payload["policy_source"]["checkpoint"].replace("seed202", "seed303")


def _not_finite(payload: dict[str, Any]) -> None:
    payload["att_engine"] = math.nan


def _other_arm(payload: dict[str, Any]) -> None:
    payload["arm"] = "dt@mix50"


def _no_engine_definition(payload: dict[str, Any]) -> None:
    del payload["att_engine"]


def _undeclared_by_the_campaign(tree: ReferenceTree) -> None:
    manifest_path = tree.cells / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cells"] = [key for key in manifest["cells"] if key != "hz1x1|dt@mappo1000|505|1099"]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("breakage", "reason"),
    [
        (_break_missing, r"absent"),
        (_break_duplicate, r"claimed by 2 files"),
        (_break_swap, r"names \(\d+, \d+\) but its file name says \(\d+, \d+\)"),
        (_break_undeclared, r"not one of the 500 declared"),
        (_break_checkpoint, r"sha256"),
        (_mutate_cell(101, 1000, _not_reproduced), r"reproduces_committed"),
        (_mutate_cell(202, 1017, _other_seed_policy), r"policy source"),
        (_mutate_cell(303, 1050, _not_finite), r"not a finite"),
        (_mutate_cell(404, 1099, _other_arm), r"arm"),
        (_mutate_cell(505, 1000, _no_engine_definition), r"att_engine"),
        (_undeclared_by_the_campaign, r"campaign manifest"),
    ],
    ids=[
        "missing", "duplicate", "swap", "undeclared", "checkpoint", "not-reproduced", "policy", "non-finite",
        "arm", "no-definition", "not-in-manifest",
    ],
)
def test_a_broken_source_is_refused_naming_its_cause_and_nothing_is_written(
    tmp_path: Path, breakage: Callable[[ReferenceTree], None], reason: str
) -> None:
    tree = build_reference_tree(tmp_path)
    breakage(tree)
    destination = tmp_path / "docs_data" / cs.REFERENCE_ROWS_NAME
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError, match=reason):
        cs.reference_rows_payload(output_root=tree.output_root, data_dir=tree.data_dir)
    status = cs.main(
        [
            "extract-reference-rows", "--output-root", str(tree.output_root), "--data-dir", str(tree.data_dir),
            "--out", str(destination),
        ]
    )

    assert status == 2
    assert _snapshot(tmp_path) == before, "a refused extraction created or changed something"
    assert not destination.exists()


# ----------------------------------------------------------------------------------------------------------------------
# The write: once, the same content again a no-op, other content refused
# ----------------------------------------------------------------------------------------------------------------------


def test_the_rows_are_written_once_the_same_content_is_a_noop_and_other_content_is_refused(tmp_path: Path) -> None:
    tree = build_reference_tree(tmp_path)
    payload = cs.reference_rows_payload(output_root=tree.output_root, data_dir=tree.data_dir)
    destination = tree.data_dir / cs.REFERENCE_ROWS_NAME

    digest = cs.write_reference_rows(payload, destination)
    written = destination.read_bytes()
    assert digest == hashlib.sha256(written).hexdigest()
    assert json.loads(written) == payload
    assert written.endswith(b"\n")

    assert cs.write_reference_rows(payload, destination) == digest
    assert destination.read_bytes() == written

    other = dict(payload)
    other["engine_seed"] = 1001
    with pytest.raises(ValueError, match="already exists"):
        cs.write_reference_rows(other, destination)
    assert destination.read_bytes() == written
    assert sorted(p.name for p in tree.data_dir.iterdir()) == sorted(
        ["p4_gate.json", "p8_4b_rederivation.json", cs.REFERENCE_ROWS_NAME]
    ), "no temporary file may be left beside the destination"


def test_a_destination_whose_directory_is_absent_is_refused_and_nothing_is_created(tmp_path: Path) -> None:
    tree = build_reference_tree(tmp_path)
    payload = cs.reference_rows_payload(output_root=tree.output_root, data_dir=tree.data_dir)
    destination = tmp_path / "absent" / cs.REFERENCE_ROWS_NAME
    with pytest.raises(ValueError, match="does not exist"):
        cs.write_reference_rows(payload, destination)
    assert not (tmp_path / "absent").exists()


# ----------------------------------------------------------------------------------------------------------------------
# The command: the registered path, and its refusal of a dirty tree
# ----------------------------------------------------------------------------------------------------------------------


def test_the_command_writes_exactly_the_payload_on_a_clean_tree(tmp_path: Path) -> None:
    tree = build_reference_tree(tmp_path)
    destination = tree.data_dir / cs.REFERENCE_ROWS_NAME
    status = cs.main(
        [
            "extract-reference-rows", "--output-root", str(tree.output_root), "--data-dir", str(tree.data_dir),
            "--out", str(destination),
        ]
    )
    assert status == 0
    expected = cs.reference_rows_payload(output_root=tree.output_root, data_dir=tree.data_dir)
    assert json.loads(destination.read_text(encoding="utf-8")) == expected
    assert destination.read_text(encoding="utf-8") == json.dumps(expected, indent=2, sort_keys=True) + "\n"


def test_the_command_refuses_a_dirty_code_tree_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """F8 / Amendment A Q8: the committed file records the commit that wrote it, so a dirty tree is refused."""
    monkeypatch.setattr(cs, "_code_provenance", lambda: {"code_commit": "d" * 40, "code_dirty": True}, raising=True)
    tree = build_reference_tree(tmp_path)
    destination = tree.data_dir / cs.REFERENCE_ROWS_NAME
    before = _snapshot(tmp_path)
    status = cs.main(
        [
            "extract-reference-rows", "--output-root", str(tree.output_root), "--data-dir", str(tree.data_dir),
            "--out", str(destination),
        ]
    )
    assert status == 2
    assert "dirty" in capsys.readouterr().out
    assert _snapshot(tmp_path) == before


# ======================================================================================================================
# C1b -- the COMMITTED file (docs/data/p4_k20_att_engine_rows.json), produced by the committed command on a clean tree
# ======================================================================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED = REPO_ROOT / "docs" / "data" / "p4_k20_att_engine_rows.json"


def _committed() -> dict[str, Any]:
    return json.loads(COMMITTED.read_text(encoding="utf-8"))


def test_the_committed_rows_have_the_registered_shape() -> None:
    """Ungated: 500 rows, each (seed, draw) of 5 seeds x 100 held-out draws exactly once, sorted, both definitions."""
    payload = _committed()
    assert payload["format_version"] == "p5.3c-reference-rows/1.0"
    rows = payload["rows"]
    assert payload["n_rows"] == len(rows) == 500
    assert [(row["seed"], row["draw_id"]) for row in rows] == [
        (seed, draw) for seed in (101, 202, 303, 404, 505) for draw in range(1000, 1100)
    ]
    for row in rows:
        assert sorted(row) == ["att_engine", "att_ours", "draw_id", "seed", "source", "source_sha256"]
        for definition in ("att_engine", "att_ours"):
            assert isinstance(row[definition], float) and math.isfinite(row[definition]), row
        assert row["source"] == reference_cell_name(row["seed"], row["draw_id"])
        assert len(row["source_sha256"]) == 64 and int(row["source_sha256"], 16) >= 0
    assert payload["definitions"] == ["att_engine", "att_ours"]
    assert payload["primary_definition"] == "att_engine"
    assert payload["seeds"] == [101, 202, 303, 404, 505]
    assert payload["draw_ids"] == list(range(1000, 1100))
    assert payload["engine_seed"] == 1000
    assert payload["subject"] == {
        "tier": "mappo1000", "arm": "dt@mappo1000", "method": "dt", "scenario": "hz1x1",
        "scenario_id": "cityflow1x1", "context_length": 20,
    }


def test_the_committed_rows_were_extracted_by_committed_code_on_a_clean_tree() -> None:
    """F8 / Amendment A Q8: the file records a clean tree and a 40-hex commit.  History-free, so it runs on every
    checkout; that the commit is in this branch's history is the next test's (``BRIEF_42`` Amendment F, F1.2)."""
    extraction = _committed()["extraction"]
    assert extraction["code_dirty"] is False
    commit = extraction["code_commit"]
    assert len(commit) == 40
    assert re.fullmatch(r"[0-9a-f]{40}", commit), f"the recorded extraction commit {commit!r} is not 40 hex digits"


def test_the_committed_rows_extraction_commit_is_an_ancestor_of_head() -> None:
    """F8 / Amendment A Q8: the commit the file names is in this branch's history -- the ancestry half of the test
    above, split out by ``BRIEF_42`` Amendment F, F1.2, because it needs history that a depth-1 checkout lacks."""
    commit = _committed()["extraction"]["code_commit"]
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=str(REPO_ROOT), capture_output=True, check=False
    )
    assert result.returncode == 0, f"the recorded extraction commit {commit} is not an ancestor of HEAD"


def test_repository_is_shallow_tells_a_full_history_from_a_depth_1_clone(tmp_path: Path) -> None:
    """``BRIEF_42`` Amendment F, F1.1: the helper on two throwaway repositories -- ``git init`` with two commits is NOT
    shallow, and a ``git clone --depth 1 file://...`` of it IS.  ``file://`` because a plain-path clone ignores
    ``--depth`` and takes the whole history (Amendment F.1); the commit counts are read by a second route, so a clone
    that silently took the whole history fails on the fixture's premise instead of being called shallow."""
    full = tmp_path / "full"
    full.mkdir()
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "core.hooksPath=/dev/null", "-C", str(full)]
    subprocess.run([*git, "init", "-q"], capture_output=True, check=True)
    for number in (1, 2):
        (full / "file.txt").write_text(f"commit {number}\n", encoding="utf-8")
        subprocess.run([*git, "add", "file.txt"], capture_output=True, check=True)
        subprocess.run([*git, "commit", "-q", "-m", f"commit {number}"], capture_output=True, check=True)
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", full.as_uri(), str(shallow)], capture_output=True, check=True)

    def commits(path: Path) -> str:
        return subprocess.run(
            ["git", "-C", str(path), "rev-list", "--count", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()

    assert (commits(full), commits(shallow)) == ("2", "1"), "the fixture: two commits, and a depth-1 clone holding one"
    assert repository_is_shallow(full) is False
    assert repository_is_shallow(shallow) is True


def test_the_committed_checkpoint_digests_are_p4_gates() -> None:
    gate = json.loads((REPO_ROOT / "docs" / "data" / "p4_gate.json").read_text(encoding="utf-8"))
    checkpoints = _committed()["checkpoints"]
    assert sorted(checkpoints) == ["101", "202", "303", "404", "505"]
    for seed, entry in checkpoints.items():
        assert entry["sha256"] == gate["checkpoints"][seed]["sha256"], seed
        assert entry["path"] == gate["checkpoints"][seed]["path"], seed


def _output_root_or_skip() -> Path:
    import os

    root = Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT", str(REPO_ROOT / "output")))
    cells = root / "p8_4b_rederivation"
    if not (cells / "campaign_manifest.json").is_file():
        pytest.skip(
            f"{cells} holds no campaign_manifest.json: set RLTRAFFIC_OUTPUT_ROOT to the output tree carrying P8.4b's "
            "gitignored cells (and P4's p4_dt/) to compare the committed rows with them"
        )
    return root


def test_the_committed_rows_equal_the_gitignored_cells_under_equality() -> None:
    """Gated on P8.4b's cells: every committed value `==` the cell it names, read by this test's own route; then the
    whole extraction re-run, every check included, reproduces the committed rows, checkpoints and source digests."""
    root = _output_root_or_skip()
    payload = _committed()
    cells = root / "p8_4b_rederivation"
    for row in payload["rows"]:
        path = cells / row["source"]
        raw = path.read_bytes()
        cell = json.loads(raw)
        assert (cell["seed"], cell["draw_id"]) == (row["seed"], row["draw_id"]), row["source"]
        assert cell["att_engine"] == row["att_engine"], row["source"]
        assert cell["att_ours"] == row["att_ours"], row["source"]
        assert hashlib.sha256(raw).hexdigest() == row["source_sha256"], row["source"]

    again = cs.reference_rows_payload(output_root=root, data_dir=REPO_ROOT / "docs" / "data")
    for key in ("rows", "checkpoints", "source", "engine_seed", "seeds", "draw_ids", "subject"):
        assert again[key] == payload[key], key
