"""P9.1's report (``BRIEF_46`` 3.4 ``rescore`` / ``report``, 3.5): the five A27 artifacts, pure functions of the run
directories and the committed artifacts.

Format ``p9.1-report/1.0``.  Every per-episode record carries ``arm``, ``tier``, ``seed``, ``draw_id``, ``att_ours``,
``att_engine``, ``entered``, ``created``, ``never_entered``, ``entered_fraction``, ``episode_reward``,
``horizon_vehicle_count`` and ``source``.  **Alignment convention:** one record per held-out ``(seed, draw)`` of a cell,
in ``(seed, draw)`` order; ``att_ours``, ``horizon_vehicle_count`` and ``episode_reward`` are route (i)'s (the registered
evaluation: ``method_tier_grid`` / ``tier_sweep`` ``evaluate``) and ``att_engine`` with the companions are route (ii)'s
(P8.4b's per-second observer, ``att_rederivation.run_campaign``), joined on ``(seed, draw)`` and refused unless route
(ii)'s ``att_ours`` equals route (i)'s under ``==``.  An OLD hz1x1 cell's ``att_ours`` is its committed artifact's,
joined to its P8.4b re-roll the same way (A27(a)).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import offline.corrective_round as cr

REPORT_FORMAT_VERSION = "p9.1-report/1.0"

#: The fields of every per-episode record (``BRIEF_46`` 3.5), ``entered_fraction`` added beside the companions.
EPISODE_FIELDS: tuple[str, ...] = (
    "arm", "tier", "seed", "draw_id", "att_ours", "att_engine", "entered", "created", "never_entered",
    "entered_fraction", "episode_reward", "horizon_vehicle_count", "source",
)


def _relative(roots: cr.Roots, path: str | Path) -> str:
    """A path under the main tree's ``output/``, written as the repository names it."""
    return f"output/{Path(path).resolve().relative_to(Path(roots.output_root).resolve()).as_posix()}"


def _seed_of(value: Any) -> int | None:
    return None if value is None else int(value)


def _joined_row(arm: str, tier: str, first: Mapping[str, Any], rerolled: Mapping[str, Any], *, att_ours: float,
                source: str, refusal: str) -> dict[str, Any]:
    """One record: *att_ours* and *first*'s two route-(i) companions, *rerolled*'s ``att_engine`` and A11(b)'s
    companions -- refused unless *rerolled* is the same episode and its ``att_ours`` (and the value its own run was
    checked against) equal *att_ours* under ``==``."""
    seed, draw = _seed_of(first.get("seed")), int(first["draw_id"])
    if (str(rerolled["arm"]), _seed_of(rerolled.get("seed")), int(rerolled["draw_id"])) != (arm, seed, draw):
        raise ValueError(f"{arm} seed {seed} draw {draw}: the P8.4b file holds {rerolled['arm']} seed "
                         f"{rerolled.get('seed')} draw {rerolled['draw_id']}")
    if not (float(rerolled["att_ours"]) == att_ours and float(rerolled["committed_att_ours"]) == att_ours
            and rerolled["reproduces_committed"] is True):
        raise ValueError(f"{arm} seed {seed} draw {draw}: {refusal} (att_ours {att_ours!r} against "
                         f"{rerolled['att_ours']!r}, checked against {rerolled['committed_att_ours']!r})")
    return {
        "arm": arm, "tier": tier, "seed": seed, "draw_id": draw, "att_ours": att_ours,
        "att_engine": float(rerolled["att_engine"]), "entered": int(rerolled["entered"]),
        "created": int(rerolled["created"]), "never_entered": int(rerolled["never_entered"]),
        "entered_fraction": float(rerolled["entered_fraction"]), "episode_reward": float(first["episode_reward"]),
        "horizon_vehicle_count": float(first["horizon_vehicle_count"]), "source": source,
    }


def _rerolled(work: Path, scenario: str, arm: str, seed: int | None, draw: int, missing: str) -> dict[str, Any]:
    from offline import att_rederivation as ar

    path = work / ar.cell_file_name(ar.CellKey(scenario, arm, seed, draw))
    if not path.is_file():
        raise ValueError(f"{scenario}:{arm} seed {seed} draw {draw}: {missing} ({path})")
    return json.loads(path.read_text(encoding="utf-8"))


def new_cell_rows(roots: cr.Roots, cell: cr.Cell) -> tuple[dict[str, Any], ...]:
    """A NEW cell's records: route (i)'s complete cell file joined to route (ii)'s per-episode files."""
    path = cr.route_i_output(roots, cell)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cr._assert_route_i_complete(payload, cell)
    work = Path(roots.runs_root) / "eval_ii" / cell.scenario
    source = f"new: route (i) {_relative(roots, path)}; route (ii) {_relative(roots, work)}, att_ours equal under =="
    rows = []
    for episode in sorted(payload["episodes"], key=lambda e: (int(e["seed"]), int(e["draw_id"]))):
        seed, draw = int(episode["seed"]), int(episode["draw_id"])
        rerolled = _rerolled(work, cell.scenario, cell.key, seed, draw, "route (ii) has no episode")
        rows.append(_joined_row(cell.key, cell.tier, episode, rerolled, att_ours=float(episode["att_horizon"]),
                                source=source, refusal="the two routes disagree"))
    return tuple(rows)


def old_cell_tier(arm: str) -> str:
    """An old hz1x1 arm's tier, by P8.4b's own name resolution (P4.4's bare names are ``mappo1000``'s)."""
    from offline import att_rederivation as ar

    return ar.normalise_arm("hz1x1", arm)[1]


def old_hz1x1_rows(roots: cr.Roots, artifact: str, arm: str) -> tuple[dict[str, Any], ...]:
    """An OLD hz1x1 cell's records: its committed artifact's episodes joined to their P8.4b re-rolls
    (``output/p9_1_runs/old_cells/hz1x1``), each re-roll refused unless it reproduces the committed ``att_ours``."""
    payload = json.loads((Path(roots.repo_root) / artifact).read_text(encoding="utf-8"))
    episodes = sorted((e for e in payload["episodes"] if str(e["arm"]) == arm),
                      key=lambda e: (-1 if e.get("seed") is None else int(e["seed"]), int(e["draw_id"])))
    if not episodes:
        raise ValueError(f"{artifact} carries no episodes of {arm!r}")
    work = Path(roots.runs_root) / "old_cells" / "hz1x1"
    tier = old_cell_tier(arm)
    source = f"{artifact}; att_engine and companions re-rolled, reproduces committed att_ours"
    rows = []
    for episode in episodes:
        rerolled = _rerolled(work, "hz1x1", arm, _seed_of(episode.get("seed")), int(episode["draw_id"]),
                             "not re-rolled")
        rows.append(_joined_row(arm, tier, episode, rerolled, att_ours=float(episode["att_horizon"]), source=source,
                                refusal="the re-roll does not reproduce the committed att_ours"))
    keys = [(row["seed"], row["draw_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{artifact}: an episode of {arm} appears twice")
    return tuple(rows)


#: The keys of P5.2c's cell rows (``tier_statements.Cell``), in which the scorers read every cell.
STATEMENT_ROW_KEYS: tuple[str, ...] = (
    "arm", "seed", "draw_id", "att_ours", "att_engine", "entered", "created", "never_entered", "entered_fraction",
    "horizon_vehicle_count", "episode_reward",
)


def as_statement_cell(cell_id: str, method: str, tier: str, arm: str, source: str, eval_file: str,
                      rows: Sequence[Mapping[str, Any]]) -> Any:
    """*rows* as P5.2c's ``tier_statements.Cell`` -- the type every registered scorer reads -- with P5.2c's row keys."""
    import offline.tier_statements as tst

    spec = tst.CellSpec(cell_id=cell_id, method=method, tier=tier, source=source, arm=arm, eval_file=eval_file,
                        manifest="none: a P9.1 run directory, digested in the artifact's inputs")
    return tst.Cell(spec=spec, rows=tuple({key: row[key] for key in STATEMENT_ROW_KEYS}
                                          for row in sorted(rows, key=lambda r: (r["seed"], r["draw_id"]))))


def statement_cell(roots: cr.Roots, cell: cr.Cell) -> Any:
    """A NEW cell as a ``tier_statements.Cell``: ``p9_1:<arm>@<tier>``, method the A27 arm."""
    return as_statement_cell(f"p9_1:{cell.key}", cell.arm, cell.tier, cell.key, "p9_1",
                             _relative(roots, cr.route_i_output(roots, cell)), new_cell_rows(roots, cell))


def new_training_runs(roots: cr.Roots, cell: cr.Cell, *, seeds: Any = None) -> list[dict[str, Any]]:
    """Every run of a new cell: ``seconds`` and ``gradient_steps`` from its write-once record, the uncontrolled-clock
    label (A27(a)) refused unless the record carries it."""
    from offline.dt_gate import TRAINING_SEEDS

    runs = []
    for seed in (TRAINING_SEEDS if seeds is None else seeds):
        path = cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, int(seed))
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("clock") != cr.CLOCK_LABEL or int(record["seed"]) != int(seed):
            raise ValueError(f"{path} is not a p9.1 run record of seed {seed} with A27(a)'s clock label")
        runs.append({"scenario": cell.scenario, "tier": cell.tier, "arm": cell.arm, "seed": int(seed),
                     "seconds": float(record["seconds"]), "gradient_steps": int(record["gradient_steps"]),
                     "clock": cr.CLOCK_LABEL, "source": _relative(roots, path)})
    return runs
