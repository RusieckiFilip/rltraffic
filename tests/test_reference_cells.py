"""P7.3d C4: the six grid4x4 reference cells, and the CityFlow pedigree half.

``BRIEF_39`` §3 C4 and Amendment B3/Q3.  Two independent things, in one artifact because they
answer one question — *is the instrument on this scenario the instrument that was frozen?*

1. **The six SUMO anchors** (`fixedtime`, `maxpressure` × draws 1000–1002), both ATT definitions.
   A9's rule, *the instrument regenerates*: P7.3a could check its cells against P7.1's frozen
   values, and grid4x4 has none until this file exists.  ``report`` refuses if any of the six
   campaign chunks differs from them.
2. **The pedigree half**: the six CityFlow episodes of ``docs/data/p8_4b_g0_reference.json``
   re-rolled through the SAME function that produced them and compared field by field.  The
   held-out draws' `flow.json` digests were already checked at G1 (100/100 against
   ``p8_4a_admission.json``); this checks that the same demand still produces the same numbers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import offline.transfer_curve as tcv

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"
G0_REFERENCE = DATA / "p8_4b_g0_reference.json"
ARTIFACT = DATA / tcv.P7_3D_REFERENCE_CELLS_NAME
GRID4X4_KEY = "cityflow_grid4x4"


def _output_root() -> Path:
    return Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT", str(REPO_ROOT / "output")))


def test_the_reference_cells_are_the_two_anchors_on_the_first_three_held_out_draws() -> None:
    """Six cells, declared here and not chosen after a number."""
    assert tcv.REFERENCE_CELL_DRAWS == (1000, 1001, 1002)
    assert tcv.REFERENCE_CELL_ARMS == ("fixedtime", "maxpressure")
    assert len(tcv.REFERENCE_CELL_DRAWS) * len(tcv.REFERENCE_CELL_ARMS) == 6
    assert all(d in range(1000, 1100) for d in tcv.REFERENCE_CELL_DRAWS), "the held-out pool"
    assert tcv.REFERENCE_CELLS_FORMAT_VERSION == "p7.3d-reference-cells/1.0"


def test_the_artifact_refuses_before_it_writes(tmp_path: Path) -> None:
    """Refusals precede every write, including the last (the F1 lesson)."""
    out = tmp_path / tcv.P7_3D_REFERENCE_CELLS_NAME
    good_cells = [
        {"arm": arm, "draw_id": draw, "e_sumo": -1.0, "att_env": -2.0, "n_teleports": 0,
         "vehicle_types_seen": ["cf_parity"], "time_to_teleport_option": "-1"}
        for arm in tcv.REFERENCE_CELL_ARMS for draw in tcv.REFERENCE_CELL_DRAWS
    ]
    pedigree = {"n_checked": 6, "n_matching": 6, "all_match": True, "rows": []}

    with pytest.raises(ValueError, match="6"):
        tcv.write_reference_cells_artifact(
            out_path=out, cells=good_cells[:5], pedigree=pedigree
        )
    assert not out.exists()

    with pytest.raises(ValueError, match="pedigree"):
        tcv.write_reference_cells_artifact(
            out_path=out, cells=good_cells,
            pedigree={"n_checked": 6, "n_matching": 5, "all_match": False, "rows": []},
        )
    assert not out.exists(), "a failed pedigree must not leave an artifact"

    teleporting = [dict(c) for c in good_cells]
    teleporting[0]["n_teleports"] = 1
    with pytest.raises(ValueError, match="teleport"):
        tcv.write_reference_cells_artifact(
            out_path=out, cells=teleporting, pedigree=pedigree
        )
    assert not out.exists()

    written = tcv.write_reference_cells_artifact(
        out_path=out, cells=good_cells, pedigree=pedigree
    )
    assert out.is_file()
    assert written["format_version"] == tcv.REFERENCE_CELLS_FORMAT_VERSION
    assert written["n_cells"] == 6 and written["pedigree"]["all_match"] is True


@pytest.mark.skipif(not ARTIFACT.is_file(), reason=f"{ARTIFACT.name} is not committed yet")
def test_the_committed_artifact_carries_six_cells_and_a_passing_pedigree() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["format_version"] == tcv.REFERENCE_CELLS_FORMAT_VERSION
    assert artifact["n_cells"] == 6
    seen = {(c["arm"], c["draw_id"]) for c in artifact["cells"]}
    assert seen == {
        (arm, draw) for arm in tcv.REFERENCE_CELL_ARMS for draw in tcv.REFERENCE_CELL_DRAWS
    }
    for cell in artifact["cells"]:
        assert cell["n_teleports"] == 0
        assert cell["vehicle_types_seen"] == ["cf_parity"]
        assert cell["time_to_teleport_option"] == "-1"
        assert cell["n_observations"] == 3600 and cell["decisions"] == 360
        assert cell["e_sumo"] > 0.0 and cell["att_env"] > 0.0, "both definitions, both present"
    # rho's denominator on each draw is non-zero, or rho is undefined on it.
    by_draw: dict[int, dict[str, float]] = {}
    for cell in artifact["cells"]:
        by_draw.setdefault(int(cell["draw_id"]), {})[str(cell["arm"])] = float(cell["e_sumo"])
    for draw, arms in by_draw.items():
        assert arms["fixedtime"] != arms["maxpressure"], draw
    assert artifact["pedigree"]["all_match"] is True
    assert artifact["pedigree"]["n_checked"] == artifact["pedigree"]["n_matching"] == 6


def test_the_pedigree_compares_against_the_committed_g0_reference() -> None:
    """The six rows it checks are P8.4b's own, identified here so a silent re-scope would show."""
    rows = [
        row for row in json.loads(G0_REFERENCE.read_text(encoding="utf-8"))["episodes"]
        if row.get("scenario") == "grid4x4"
        and row["arm"] in ("behaviour@fixedtime", "behaviour@maxpressure")
    ]
    assert len(rows) == 6
    assert {row["draw_id"] for row in rows} == set(tcv.REFERENCE_CELL_DRAWS)
    assert {row["arm"].split("@")[1] for row in rows} == set(tcv.REFERENCE_CELL_ARMS)
    assert all(row["engine_seed"] == 1000 for row in rows), "A18(c)'s seed rule"
