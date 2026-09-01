"""Direct 1 s replay of a signal plan on a CityFlow engine (P2.5 measurement helper).

The ``acyclic`` control mode cannot command a full decision step of clearance
(``delta_time > 5`` is required; the 5 s all-red would consume the whole window),
so the shipped ``signal_plan_template.txt`` cannot be reproduced through the env.
This helper drives a raw ``cityflow.Engine`` at 1 s resolution to obtain the
shipped plan's ground-truth behaviour, used only to choose ``k`` -- it collects no
corpus.

Comparability (brief §9, Ruling 1; metric per prereg A1)
-------------------------------------------------------
The registered per-step metric ``average_travel_time`` is computed by the **same**
:class:`metrics.CityFlowMetrics` class the env uses, driven at the same ``delta_time`` cadence
(``warmup`` once, then ``pre_step`` -> advance ``delta_time`` seconds -> ``update`` per decision
step).  The engine's native ``get_average_travel_time`` is *not* used, **because it would not be
comparable to the k=3 / k=4 numbers, which were produced through the metrics class.**

⚠️ **CORRECTED 2026-08-31 (T1's M3, landed with the metric change as ``BRIEF_31`` Amendment A5
queued it).  This paragraph used to justify that choice by calling the engine's metric
"survivorship-biased", and ON THE DOMINANT AXIS THAT IS BACKWARDS.**
``Engine::getAverageTravelTime`` (``CityFlow/src/engine/engine.cpp:682-691``) averages over
``finishedVehicleCnt + |vehiclePool|`` with **no filter at all** -- every vehicle ever created,
including ones still queued in a lane's insertion buffer.  It is ``metrics/cityflow.py``'s
``average_travel_time`` that carries an entry-side survivorship bias: its population is
``get_vehicles(include_waiting=False)``, so a vehicle that never reaches a lane is never counted,
and its clock starts at ADMISSION rather than at creation.

**Measured, not argued** (P8.4b Gate 0, ``docs/data/p8_4b_g0_reference.json``, 46 episodes):
``att_ours - att_engine`` decomposes exactly -- residual 0.0 on 46 of 46 -- into a POPULATION term,
a CLOCK-ORIGIN term and a CADENCE term.  ``PREREGISTRATION`` A11 registered the correction, A12 and
A13 the gate that tested it, and Gate 0 PASSED on both scenarios, so ``att_engine`` is the primary
metric there under ``Rule R``.

**The operative reason for this helper's choice is unchanged and still holds:** it must match the
pipeline the k=3 / k=4 numbers came from, and that pipeline is the metrics class.  Only the
justification's characterisation of WHICH metric is biased was wrong.

From the same per-step samples this helper
reports both A1 aggregations -- ``att_horizon`` (the value at the episode horizon, the paper's
primary metric) and ``att_running_mean`` (the legacy runner.py mean-of-samples).
``tests/test_fixed_time_env_mapping.py`` asserts the two pipelines agree exactly on a degenerate
plan, under both aggregations, before any replay number is trusted.

The plan format is single-column (one intersection); the same phase is applied to
every intersection id, which is correct for the 1x1 scenarios that ship a plan.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

__all__ = ["ReplayResult", "read_plan_phases", "replay_plan"]


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of a plan replay (metric per prereg A1).

    ``att_horizon`` is the registered ``average_travel_time`` metric at the episode horizon -- the
    paper's primary metric (A1), the mean over all vehicles that entered (no survivorship bias).
    ``att_running_mean`` is the legacy mean of the per-decision-step samples (the quantity
    ``experiments.runner.evaluate_policy`` reports); it is kept for continuity and is never called
    "average travel time". ``entered`` counts all vehicles that departed; ``completed`` counts those
    that finished; ``vehicle_count`` is the number still in the network at the horizon
    (``entered == completed + vehicle_count``).
    """

    att_horizon: float
    att_running_mean: float
    entered: int
    completed: int
    vehicle_count: int


def read_plan_phases(path: str | Path) -> tuple[str, list[int]]:
    """Read a ``signal_plan_template.txt`` as ``(header, per-second file phases)``."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("signal plan file is empty")
    header = lines[0].strip()
    phases = [int(line.strip()) for line in lines[1:] if line.strip() != ""]
    return header, phases


def replay_plan(
    config_path: str | Path,
    phases: Sequence[int],
    *,
    delta_time: int,
    max_steps: int,
    metric_names: Sequence[str],
    seed: int = 0,
    intersection_ids: Sequence[str] | None = None,
) -> ReplayResult:
    """Replay ``phases`` (one file-phase index per second) on a raw CityFlow engine.

    Advances ``max_steps * delta_time`` simulation seconds, sampling metrics once per
    ``delta_time`` seconds to mirror the env exactly. Uses the original roadnet (the
    plan's own phase indexing), resolving ``dir`` to an absolute path so the result
    does not depend on the current working directory.
    """
    import cityflow
    from metrics import CityFlowMetrics
    from utils.cityflow_utils import parse_roadnet

    total_seconds = int(max_steps) * int(delta_time)
    if len(phases) < total_seconds:
        raise ValueError(
            f"plan has {len(phases)} phase rows but {max_steps} steps x {delta_time}s "
            f"needs {total_seconds}"
        )

    cfg = json.loads(Path(config_path).read_bytes())
    cfg_dir = cfg.get("dir", "")
    if not os.path.isabs(cfg_dir):
        cfg_dir = str(Path.cwd() / cfg_dir)
    cfg_dir = os.path.normpath(cfg_dir) + "/"
    cfg["dir"] = cfg_dir

    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    try:
        json.dump(cfg, tmp)
        tmp.close()
        roadnet = parse_roadnet(cfg_dir + cfg["roadnetFile"])
        ix_ids = (
            [str(i) for i in intersection_ids]
            if intersection_ids is not None
            else [str(ix.id) for ix in roadnet.intersections]
        )
        engine = cityflow.Engine(tmp.name, 1)
        engine.set_random_seed(int(seed))
        metrics = CityFlowMetrics(
            engine,
            roadnet.intersections,
            metric_names=list(metric_names),
            delta_time=int(delta_time),
            roadnet=roadnet,
        )
        metrics.warmup()

        # Sample the registered per-step average_travel_time once per decision step. Prereg A1's
        # primary metric is the value at the horizon (att_horizon = the last sample); att_running_mean
        # is the legacy mean of the samples, kept for continuity. Both come from one pass.
        att_samples: list[float] = []
        second = 0
        for _ in range(int(max_steps)):
            metrics.pre_step()
            for _ in range(int(delta_time)):
                phase = int(phases[second])
                for ix_id in ix_ids:
                    engine.set_tl_phase(ix_id, phase)
                engine.next_step()
                second += 1
            metrics.update()
            att_samples.append(float(metrics.get("average_travel_time")))

        att_horizon = att_samples[-1] if att_samples else 0.0
        att_running_mean = sum(att_samples) / len(att_samples) if att_samples else 0.0
        # The metric's per-vehicle bookkeeping is the only source of entered /
        # completed counts; there is no public accessor for them.
        episode = metrics._episode  # noqa: SLF001 - measurement helper
        entered = len(episode["depart_time"])
        completed = len(episode["completed"])
        vehicle_count = int(engine.get_vehicle_count())
    finally:
        os.unlink(tmp.name)

    return ReplayResult(att_horizon, att_running_mean, entered, completed, vehicle_count)
