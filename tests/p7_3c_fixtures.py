"""Fixtures shared by P7.3c's tests (``BRIEF_41``): scripted C6 v1.1 corpora written by the REAL logger.

Not a test module -- the name carries no ``test_`` prefix -- so pytest collects nothing here.

WHY THE REAL WRITER
-------------------
Every corpus below is written through :class:`offline.trajectory_logger.TrajectoryLogger`, never assembled by
hand, so a fixture episode has exactly the on-disk shape a collected one has: the ``ix{i}_*`` arrays, the
``ix_ids`` array in ENV order, the manifest and its per-episode entries, the file names.  A hand-made ``.npz``
would test the reader against the fixture author's belief about the format.

Format version: C6 v1.1 (the logger's own, ``trajectory_logger.FORMAT_VERSION``).  Alignment convention
(``docs/CONTRACTS.md`` C6): observation rows ``T + 1``, decision and outcome rows ``T``; the reward of decision
``t`` is read from the ``info`` returned by step ``t``, so the scripted env puts it in
``info["intersections"][ix]["reward"]`` of observation row ``t + 1``.

The intersection ORDER is a parameter.  ``build_joint_index`` and the A17(f) gate key everything by id; a
corpus logged in another order than the checkpoint's or the artifact's is how a test proves that nothing is
keyed by position.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from offline.trajectory_logger import TrajectoryLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"

#: The scenario key and the widths of the registered grid4x4 subject (A20(a); its checkpoint's config).
GRID = "cityflow_grid4x4"
STATE_DIM = 40
N_ACTIONS = 8
#: A17(b)'s horizon: 360 decisions of 10 s.
DECISIONS = 360
#: The scripted env's lane ids -- any fixed set: the logger freezes it at the first reset.
LANES: tuple[str, ...] = ("in_0", "in_1", "out_0")
#: A18(c)'s engine seed, requested on every draw.
ENGINE_SEED_REQUESTED = 1000


def calibration() -> dict[str, Any]:
    """``docs/data/p7_3d_calibration.json``, read by the TEST's own route (``json``)."""
    return json.loads((DATA / "p7_3d_calibration.json").read_text(encoding="utf-8"))


def calibration_ids() -> list[str]:
    """The sixteen intersection ids, in the calibration artifact's order."""
    return [str(ix) for ix in calibration()["intersection_ids"]]


def probe_return(draw: int, ix: str) -> float:
    """P7.3d's recorded SUMO probe return of *ix* on *draw* -- the A17(f) reference."""
    return float(calibration()["probe_returns"]["sumo"][str(int(draw))][ix])


def registered_engine_seed_drawn() -> int:
    """The engine seed every committed zero-shot cell records, read by THIS file's route.

    Amendment A (Q14): the gate reads it from ``docs/data/p7_3d_grid4x4.json``; a test that wants to know the
    value must not ask the module under test.  Refuses unless the 700 cells agree on ONE value.
    """
    artifact = json.loads((DATA / "p7_3d_grid4x4.json").read_text(encoding="utf-8"))
    values = {int(cell["engine_seed_drawn"]) for cell in artifact["cells"]}
    assert len(values) == 1, f"the committed cells record {len(values)} engine seeds, not one"
    return values.pop()


class ScriptedIntersection:
    """Stands in for ``IntersectionInfo``: an id and a phase count (``infer_action_counts``' fallback)."""

    def __init__(self, ix_id: str, num_phases: int) -> None:
        self.id = ix_id
        self.num_phases = num_phases


class ScriptedCorpusEnv:
    """Enough of contracts C1 and C2 for the logger: ids in a CHOSEN order, fixed widths, scripted values.

    No ``action_space`` attribute on purpose, so ``Utils.infer_action_counts`` takes its ``num_phases``
    fallback.  States are small integers (exact in float32) that vary with the draw, the intersection's
    position and the step, so a row, a column or an episode out of place changes a value.
    """

    def __init__(
        self,
        ids: Sequence[str],
        *,
        state_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
        delta_time: int = 10,
    ) -> None:
        self.intersections = [ScriptedIntersection(str(ix), int(n_actions)) for ix in ids]
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.delta_time = int(delta_time)
        self.max_steps = DECISIONS

    def state(self, draw: int, position: int, step: int) -> list[float]:
        """The scripted observation of the intersection at *position* (env order)."""
        return [
            float((int(draw) + 7 * int(position) + 3 * int(step) + k) % 23)
            for k in range(self.state_dim)
        ]

    def action(self, draw: int, step: int) -> np.ndarray:
        """One legal action per intersection, in env order (contract C1)."""
        return np.asarray(
            [(int(draw) + int(step) + j) % self.n_actions for j in range(len(self.intersections))],
            dtype=np.int64,
        )

    def info(self, draw: int, step: int, rewards: Mapping[str, float] | None) -> dict[str, Any]:
        """The ``info`` of observation row *step*; *rewards* are the outcome of decision ``step - 1``."""
        intersections: dict[str, dict[str, Any]] = {}
        for position, ix in enumerate(self.intersections):
            entry: dict[str, Any] = {
                "state": self.state(draw, position, step),
                "avail_actions": list(range(self.n_actions)),
                "current_phase": int((step + position) % self.n_actions),
                "time_in_phase": float((step * self.delta_time) % 30),
                "action_applied": True,
                "metrics": {},
            }
            entry["reward"] = -0.0 if rewards is None else float(rewards[str(ix.id)])
            intersections[str(ix.id)] = entry
        lane_counts = {lane: int((step + k) % 5) for k, lane in enumerate(LANES)}
        return {
            "sim_time": float(step * self.delta_time),
            "vehicle_count": int(sum(lane_counts.values())),
            "step": int(step),
            "average_travel_time": float(step),
            "lane_vehicle_count": dict(lane_counts),
            "lane_waiting_vehicle_count": {lane: int(count // 2) for lane, count in lane_counts.items()},
            "metrics": {},
            "intersections": intersections,
        }


RewardsFor = Callable[[int, str, int], Sequence[float]]


def probe_rewards(draw: int, ix: str, decisions: int) -> list[float]:
    """Per-decision rewards whose float64 sum is EXACTLY P7.3d's probe return of *ix* on *draw*."""
    return [probe_return(draw, ix)] + [0.0] * (int(decisions) - 1)


def log_episode(
    logger: TrajectoryLogger,
    env: ScriptedCorpusEnv,
    *,
    draw: int,
    decisions: int,
    rewards_for: RewardsFor,
    engine_seed: int = ENGINE_SEED_REQUESTED,
) -> None:
    """Drive the logger through ONE episode -- reset, then *decisions* steps -- WITHOUT finalizing it."""
    ids = [str(ix.id) for ix in env.intersections]
    per_ix = {ix: list(rewards_for(draw, ix, decisions)) for ix in ids}
    for ix, values in per_ix.items():
        assert len(values) == decisions, f"{ix}: {len(values)} rewards for {decisions} decisions"
    info = env.info(draw, 0, None)
    logger.on_reset(info, engine_seed=engine_seed, flow_draw=draw)
    for step in range(decisions):
        logger.on_action(info, env.action(draw, step))
        rewards = {ix: per_ix[ix][step] for ix in ids}
        info = env.info(draw, step + 1, rewards)
        logger.on_step_result(float(sum(rewards.values())), False, step + 1 == decisions, info)


def door_record(draw: int, **overrides: Any) -> dict[str, Any]:
    """One ``run_metadata["sumo_draws"]`` entry in the shape ``offline.collect``'s SUMO door writes."""
    from offline.collect import ENGINE_EVENT_GRAIN

    record: dict[str, Any] = {
        "draw_id": int(draw),
        "parity_sumocfg": f"fixture/draw_{int(draw):04d}/parity/noteleport.sumocfg",
        "parity_sumocfg_sha256": "0" * 64,
        "engine_seed_requested": ENGINE_SEED_REQUESTED,
        "engine_seed_drawn": registered_engine_seed_drawn(),
        "time_to_teleport_option": "-1",
        "vehicle_types_seen": ["cf_parity"],
        "n_teleports": 0,
        "n_collisions": 0,
        "engine_events_counted": ENGINE_EVENT_GRAIN,
    }
    record.update(overrides)
    return record


def grid_run_metadata(draws: Sequence[int], **overrides: Any) -> dict[str, Any]:
    """The ``run_metadata`` fields the gate reads, as the door writes them for grid4x4."""
    metadata: dict[str, Any] = {
        "scenario_id": GRID,
        "backend": "sumo",
        "behavior_policy": "maxpressure",
        "episodes": 1,
        "base_seed": ENGINE_SEED_REQUESTED,
        "sumo_draws": [door_record(draw) for draw in draws],
    }
    metadata.update(overrides)
    return metadata


def write_sha256sums(corpus_dir: Path) -> Path:
    """``SHA256SUMS`` in ``sha256sum``'s own format over every ``.npz`` and ``manifest.json``, sorted."""
    names = sorted(
        path.name
        for path in corpus_dir.iterdir()
        if path.is_file() and (path.suffix == ".npz" or path.name == "manifest.json")
    )
    lines = [f"{hashlib.sha256((corpus_dir / name).read_bytes()).hexdigest()}  {name}" for name in names]
    target = corpus_dir / "SHA256SUMS"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_scripted_corpus(
    out_dir: Path,
    *,
    ids: Sequence[str],
    draws: Sequence[int],
    rewards_for: RewardsFor,
    decisions: int = DECISIONS,
    run_metadata: Mapping[str, Any] | None = None,
    state_dim: int = STATE_DIM,
    sha256sums: bool = True,
) -> Path:
    """A whole collection run through the REAL logger: one episode per draw, then the manifest's sums."""
    env = ScriptedCorpusEnv(ids, state_dim=state_dim)
    metadata = dict(grid_run_metadata(draws) if run_metadata is None else run_metadata)
    logger = TrajectoryLogger(env, out_dir, run_metadata=metadata)
    for draw in draws:
        log_episode(logger, env, draw=int(draw), decisions=decisions, rewards_for=rewards_for)
        logger.finalize_episode()
    if sha256sums:
        write_sha256sums(Path(out_dir))
    return Path(out_dir)
