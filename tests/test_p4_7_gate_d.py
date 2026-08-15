"""Gate D -- the held-out demand P4.6's merged numbers were measured on, and its exposure.

**The exposure, stated once and measured** (``docs/plans/p4.7.md`` section 6.1): of the hundred
held-out draws behind every merged P4.6 cell, **five directories survive anywhere on this machine**.
The other ninety-five exist only as :mod:`offline.materialise_draws` plus a seed -- and until this
gate ran, **the recipe had never been checked against the bytes it is supposed to reproduce.**  A
mixture cell evaluated on a silently different demand would be compared with the phase-1 cells as if
it were paired with them, which is precisely what ``PREREGISTRATION`` A5 forbids.

The tests below are fixture-driven, so they run without the corpus: a "surviving" draw directory and
a "regenerated" one are written by hand, and the gate is asked to accept or refuse them.  The real
comparison against ``/home/filip/rltraffic/scenarios/draws`` is executed by the campaign and its
result is committed as an artifact.

**What may legitimately differ between two checkouts, and nothing else may:** a materialised
``cityflow.json`` stores ``dir`` as the **absolute** path of the source scenario directory
(``offline/collect.py::_write_draw_config``), so two checkouts of the same commit render configs
differing in that one string.  ``flow.json`` is the demand; it carries no path and must be
**byte-identical**.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from offline.mixture_tiers import (
    DRAW_CONFIG_PATH_KEY,
    DRAW_FILES,
    DRAW_IDENTITY_FORMAT_VERSION,
    compare_draw_directories,
    draw_identity_artifact,
    file_sha256,
)

ROADNET_BYTES = b'{"intersections": [], "roads": []}\n'


def write_scenario(root: Path, name: str) -> Path:
    """A scenario directory holding the roadnet a rendered config points at."""
    scenario = root / name / "hangzhou_1x1_bc-tyc_18041610_1h"
    scenario.mkdir(parents=True, exist_ok=True)
    (scenario / "roadnet.json").write_bytes(ROADNET_BYTES)
    return scenario


def write_draw(
    root: Path,
    draw_id: int,
    scenario: Path,
    *,
    scenario_key: str = "cityflow1x1",
    flow: bytes = b'{"vehicles": 1821}\n',
    routes: bytes = b"<routes/>\n",
    config_extra: dict[str, Any] | None = None,
) -> Path:
    """One materialised draw directory, in the shape ``materialise_draws`` writes."""
    directory = root / scenario_key / f"draw_{draw_id:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "flow.json").write_bytes(flow)
    (directory / "routes.rou.xml").write_bytes(routes)
    config = {
        "network": "hz1x1",
        "interval": 1.0,
        "seed": 0,
        DRAW_CONFIG_PATH_KEY: str(scenario) + "/",
        "roadnetFile": "roadnet.json",
        "flowFile": f"../draws/{scenario_key}/draw_{draw_id:04d}/flow.json",
        "rlTrafficLight": True,
        "saveReplay": False,
        "laneChange": False,
    }
    config.update(config_extra or {})
    (directory / "cityflow.json").write_text(json.dumps(config, indent=4), encoding="utf-8")
    return directory


@pytest.fixture()
def two_trees(tmp_path: Path) -> tuple[Path, Path]:
    """A surviving tree and a regenerated one, identical but for the scenario path."""
    reference_root, candidate_root = tmp_path / "main", tmp_path / "p47"
    reference_scenario = write_scenario(tmp_path, "main")
    candidate_scenario = write_scenario(tmp_path, "p47")
    for draw_id in (1000, 1001):
        write_draw(reference_root, draw_id, reference_scenario)
        write_draw(candidate_root, draw_id, candidate_scenario)
    return reference_root, candidate_root


def test_file_sha256_matches_hashlib_on_the_same_bytes(tmp_path: Path) -> None:
    """The digest route the gate trusts, checked against ``hashlib`` on the identical bytes."""
    payload = b"x" * (1 << 21) + b"tail"
    target = tmp_path / "blob.bin"
    target.write_bytes(payload)
    assert file_sha256(target) == hashlib.sha256(payload).hexdigest()


def test_two_checkouts_of_the_same_draw_reproduce(two_trees: tuple[Path, Path]) -> None:
    """Everything byte-identical except the config, which differs only in the scenario path."""
    reference, candidate = two_trees
    record = compare_draw_directories(
        reference / "cityflow1x1" / "draw_1000", candidate / "cityflow1x1" / "draw_1000"
    )
    assert sorted(record["byte_identical"]) == ["flow.json", "routes.rou.xml"]
    assert record["path_normalised"] == ["cityflow.json"]


def test_a_perturbed_flow_is_refused_and_the_refusal_names_the_draw(
    two_trees: tuple[Path, Path],
) -> None:
    """One byte of demand is the whole point of the gate.

    The perturbation is a single character in a file the engine reads as the arrival schedule.  A
    gate that tolerates it would let a mixture cell be evaluated against a different demand from the
    phase-1 cells it is paired with.
    """
    reference, candidate = two_trees
    target = candidate / "cityflow1x1" / "draw_1000" / "flow.json"
    target.write_bytes(target.read_bytes().replace(b"1821", b"1822"))
    with pytest.raises(ValueError, match=r"draw_1000/flow\.json"):
        compare_draw_directories(
            reference / "cityflow1x1" / "draw_1000", candidate / "cityflow1x1" / "draw_1000"
        )


def test_a_config_difference_outside_the_path_key_is_refused(
    two_trees: tuple[Path, Path],
) -> None:
    """``dir`` may move between checkouts; ``interval`` may not.

    ``interval`` is the engine's step size, so two configs differing on it describe two different
    simulations while looking like the same draw.
    """
    reference, candidate = two_trees
    target = candidate / "cityflow1x1" / "draw_1000" / "cityflow.json"
    config = json.loads(target.read_text(encoding="utf-8"))
    config["interval"] = 2.0
    target.write_text(json.dumps(config, indent=4), encoding="utf-8")
    with pytest.raises(ValueError, match="interval"):
        compare_draw_directories(
            reference / "cityflow1x1" / "draw_1000", candidate / "cityflow1x1" / "draw_1000"
        )


def test_a_relative_flow_path_difference_is_refused(two_trees: tuple[Path, Path]) -> None:
    """``flowFile`` is stored RELATIVE to ``dir``, so it must agree even across checkouts.

    If it were allowed to differ, a config could point at another draw's demand while every other
    field looked right -- the failure that is invisible in a per-file digest of the directory.
    """
    reference, candidate = two_trees
    target = candidate / "cityflow1x1" / "draw_1000" / "cityflow.json"
    config = json.loads(target.read_text(encoding="utf-8"))
    config["flowFile"] = "../draws/cityflow1x1/draw_1001/flow.json"
    target.write_text(json.dumps(config, indent=4), encoding="utf-8")
    with pytest.raises(ValueError, match="flowFile"):
        compare_draw_directories(
            reference / "cityflow1x1" / "draw_1000", candidate / "cityflow1x1" / "draw_1000"
        )


def test_two_scenario_paths_pointing_at_different_roadnets_are_refused(
    two_trees: tuple[Path, Path], tmp_path: Path
) -> None:
    """A tolerated ``dir`` must still resolve to the SAME network.

    This is what stops the path exemption from becoming a hole: ``dir`` is allowed to move because a
    checkout moves, not because the scenario behind it may change.
    """
    reference, candidate = two_trees
    (tmp_path / "p47" / "hangzhou_1x1_bc-tyc_18041610_1h" / "roadnet.json").write_bytes(
        b'{"intersections": [{"id": "different"}], "roads": []}\n'
    )
    with pytest.raises(ValueError, match="different roadnets"):
        compare_draw_directories(
            reference / "cityflow1x1" / "draw_1000", candidate / "cityflow1x1" / "draw_1000"
        )


def test_a_missing_regenerated_draw_is_refused(
    two_trees: tuple[Path, Path], tmp_path: Path
) -> None:
    """A draw that survived but was never regenerated is a refusal, never an empty pass.

    ⚠️ **The first version of this test did not construct the case it named**: it asked for draw
    1002, which the fixture creates in *neither* tree, so it exercised the missing-**survivor**
    branch while asserting the missing-**regenerated** message.  The assertion was right and the
    setup was wrong; the setup is fixed here and the other branch gets its own test below.
    """
    reference, candidate = two_trees
    write_draw(reference, 1002, write_scenario(tmp_path, "main"))
    assert (reference / "cityflow1x1" / "draw_1002").is_dir()
    assert not (candidate / "cityflow1x1" / "draw_1002").is_dir()
    with pytest.raises(ValueError, match="no regenerated draw"):
        draw_identity_artifact(reference, candidate, [1000, 1002])


def test_a_draw_that_never_survived_is_refused_rather_than_skipped(
    two_trees: tuple[Path, Path],
) -> None:
    """Gate D compares against the survivors, so a draw with no survivor cannot be gated.

    Silently skipping it would let the gate report a PASS over fewer draws than it was asked for --
    the shape in which a gate fails open.
    """
    reference, candidate = two_trees
    with pytest.raises(ValueError, match="no surviving draw"):
        draw_identity_artifact(reference, candidate, [1000, 1099])


def test_the_gate_artifact_records_what_it_compared(two_trees: tuple[Path, Path]) -> None:
    """A bare PASS is not evidence: the artifact names the files and the draws behind it."""
    reference, candidate = two_trees
    payload = draw_identity_artifact(reference, candidate, [1000, 1001])
    assert payload["format_version"] == DRAW_IDENTITY_FORMAT_VERSION
    assert payload["status"] == "PASS"
    assert payload["draws_compared"] == [1000, 1001]
    assert payload["survivors_available"] == [1000, 1001]
    assert payload["files_compared"] == list(DRAW_FILES)
    assert payload["path_key_allowed_to_differ"] == DRAW_CONFIG_PATH_KEY
    assert [record["draw_id"] for record in payload["draws"]] == [1000, 1001]


def test_the_gate_refuses_to_pass_on_an_empty_draw_list(two_trees: tuple[Path, Path]) -> None:
    """Zero comparisons is the shape a gate fails open in."""
    reference, candidate = two_trees
    with pytest.raises(ValueError, match="at least one"):
        draw_identity_artifact(reference, candidate, [])


def test_the_cli_writes_the_artifact_and_reports_a_pass(
    two_trees: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The campaign calls this through the CLI, so the CLI is what is tested."""
    from offline.mixture_tiers import main

    reference, candidate = two_trees
    out = tmp_path / "gate_d.json"
    assert (
        main(
            [
                "gate-d",
                "--reference-root",
                str(reference),
                "--candidate-root",
                str(candidate),
                "--draws",
                "1000",
                "1001",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["draws_compared"] == [1000, 1001]
    assert "GATE D PASS" in capsys.readouterr().out
