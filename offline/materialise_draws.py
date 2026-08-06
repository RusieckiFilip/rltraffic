"""Materialise flow draws into stable, runnable scenario directories.

SKELETON -- signatures and constants only; every body raises ``NotImplementedError``
so the tests written against this module reach the real API surface.  Replaced by the
implementation in the next commit (see ``docs/plans/p2.2-draws.md`` §8).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "CITYFLOW_CONFIG_FILENAME",
    "DEFAULT_OUT_ROOT",
    "FLOW_FILENAME",
    "FORMAT_VERSION",
    "HELD_OUT_POOL",
    "MaterialisedDraw",
    "PROVENANCE_FILENAME",
    "SUMO_ROUTES_FILENAME",
    "TRAINING_POOL",
    "build_parser",
    "classify_draw_pool",
    "draw_config_path",
    "draw_dir",
    "load_provenance",
    "main",
    "materialise",
    "scenario_key_for_config",
]

FORMAT_VERSION = "materialised-draw/1.0"
DEFAULT_OUT_ROOT = Path("scenarios/draws")
FLOW_FILENAME = "flow.json"
CITYFLOW_CONFIG_FILENAME = "cityflow.json"
SUMO_ROUTES_FILENAME = "routes.rou.xml"
PROVENANCE_FILENAME = "provenance.json"

TRAINING_POOL = range(1, 1000)
HELD_OUT_POOL = range(1000, 1100)


@dataclass(frozen=True)
class MaterialisedDraw:
    """One draw's on-disk result."""

    scenario_key: str
    draw_id: int
    pool: str
    directory: Path
    config_path: Path
    flow_path: Path
    sumo_path: Path | None
    n_vehicles: int
    flow_sha256: str
    action: str


def scenario_key_for_config(source_config: str | Path) -> str:
    """Return the directory key for a source sim config."""
    raise NotImplementedError("P2.2-draws skeleton")


def classify_draw_pool(draw_id: int) -> str:
    """Return the registered pool a draw id belongs to."""
    raise NotImplementedError("P2.2-draws skeleton")


def draw_dir(
    scenario_key: str, draw_id: int, *, out_root: str | Path = DEFAULT_OUT_ROOT
) -> Path:
    """Return the directory holding one materialised draw."""
    raise NotImplementedError("P2.2-draws skeleton")


def draw_config_path(
    scenario_key: str, draw_id: int, *, out_root: str | Path = DEFAULT_OUT_ROOT
) -> Path:
    """Return the CityFlow sim config path for one materialised draw."""
    raise NotImplementedError("P2.2-draws skeleton")


def load_provenance(
    scenario_key: str, draw_id: int, *, out_root: str | Path = DEFAULT_OUT_ROOT
) -> dict[str, Any]:
    """Return the provenance record of one materialised draw."""
    raise NotImplementedError("P2.2-draws skeleton")


def materialise(
    source_config: str | Path,
    draw_ids: Sequence[int],
    *,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    force: bool = False,
    dry_run: bool = False,
) -> list[MaterialisedDraw]:
    """Materialise *draw_ids* for one source scenario."""
    raise NotImplementedError("P2.2-draws skeleton")


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    raise NotImplementedError("P2.2-draws skeleton")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one materialisation; returns a process exit code."""
    raise NotImplementedError("P2.2-draws skeleton")
