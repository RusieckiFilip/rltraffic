"""The two P9.2 campaign drivers (``offline/campaigns/p9_2_sumo_anchors.sh``, ``p9_2_demand_shift.sh``): they parse, they
fail closed, and a refused start creates nothing.

A driver refuses BEFORE its token is consumed -- on a dirty run tree (every cell records ``git_dirty`` and would be
refused) or on a missing token -- and exits 2 having written no log, no work directory and no draw.  Which refusal fires
first depends on the tree the test runs in, so the test asserts the exit code, the ``REFUSING TO START`` line and that
the runs root it was pointed at is still empty; it never needs, and never writes, a real token.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVERS = ("p9_2_sumo_anchors.sh", "p9_2_demand_shift.sh")


@pytest.mark.parametrize("name", DRIVERS)
def test_the_driver_parses(name: str) -> None:
    result = subprocess.run(["bash", "-n", str(REPO_ROOT / "offline" / "campaigns" / name)], capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", DRIVERS)
def test_a_start_without_a_token_refuses_and_creates_nothing(tmp_path: Path, name: str) -> None:
    runs = tmp_path / "runs"
    output = tmp_path / "output"
    env = {**os.environ, "P92_RUNS": str(runs), "P92_OUTPUT_ROOT": str(output), "P92_SESSION": "p92-test-never"}
    result = subprocess.run(["bash", str(REPO_ROOT / "offline" / "campaigns" / name)], capture_output=True, text=True,
                            env=env, timeout=60)
    assert result.returncode == 2
    assert "REFUSING TO START" in result.stderr
    assert not runs.exists() and not output.exists()


@pytest.mark.parametrize("name", DRIVERS)
def test_the_driver_fails_closed_and_pins_one_thread(name: str) -> None:
    text = (REPO_ROOT / "offline" / "campaigns" / name).read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1" in text
    # the token is consumed only after every refusable step
    assert text.index('rm -f "$TOKEN"') > text.index("canary")
    assert text.index('rm -f "$TOKEN"') > text.index("gate" if "sumo" in name else "guard")
