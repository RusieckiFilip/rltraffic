"""P8.2: the policy's decision latency, timed alone in real CityFlow episodes -- and no outcome whatsoever.

Written against ``docs/briefs/BRIEF_43_p8.2_compute_latency.md`` §4 and its **Amendment A** (A2's rulings Q8–Q14, A3's
two added requirements), on the plan ``docs/plans/p8.2.md`` approved at ``a7c43e1``.

On-disk formats
---------------
* ``p8.2-latency/1.0`` -- one JSON record per (row, device), ``<run dir>/<row>_<device>.json``, written ONCE (atomic,
  exclusive). It carries the per-decision nanoseconds of three episodes, the statistics over the timed decisions,
  the regime, the machine and the checkpoint's digest.
* ``p8.2-canary/1.0`` -- ``canary_open.json`` / ``canary_close.json``: the machine-health canary's seconds, its
  verdict and whether the engine reproduced draw 0 -- as a boolean, never the observed values.
* ``p8.2-latency-run/1.0`` -- ``run.json``: the row order, every attempt of every supervised process and the run's
  terminal status, beside the markers ``COMPLETE`` / ``FAILED``.
* ``p8.2-preflight/1.0`` -- the G1 pre-flight's record: the measured process wall times, the per-scenario timeouts
  and the run's expected duration derived from them.

Alignment convention: not applicable -- no trajectory is recorded. The episode loop mirrors
``offline.horizon_metric.horizon_rollout``'s order (``info = env.reset(seed=1000)``; then ``max_steps`` times: decide,
step) with every accumulation removed: the reward is received and discarded, no metric is read, and no ``info``
field reaches a record.

What is timed
-------------
Only the decision call -- the function the row's OWN evaluation factory returns, imported and never reimplemented.
The env's ``step`` is outside the timer. On CUDA, ``torch.cuda.synchronize()`` runs before BOTH clock readings.
The first ``WARMUP`` decisions of each episode are recorded and excluded from the statistics by count.

Teardown and CityFlow's destructor race (Amendment A3.1, ``DEFERRED`` 104)
-------------------------------------------------------------------------
``CityFlowEnv.reset`` reuses its engine; only ``close`` (or garbage collection) destroys it, and that destructor can
hang forever. A row process therefore keeps all three episode envs OPEN until its record is written, writes the
record atomically and exclusively, and only then closes them. The driver supervises each process under a timeout:
an attempt that hangs before its record exists is killed and re-run (at most ``MAX_ATTEMPTS``); a record written by
a process that then hangs is kept; a process that EXITS without a record fails the run at once.

MAPPO on CUDA (Amendment A3.2)
------------------------------
``offline.dt_gate._mappo_factory(path, "cuda")`` places the agent on CUDA without any change to frozen code:
``MAPPOAgent.__init__`` resolves the device (``agent/MAPPOAgent.py:694``), ``load`` maps the payload onto it
(``:913``), ``_RunningNorm.load_state_dict`` moves the normalisers to it (``:86-89``) and ``select_actions`` /
``estimate_values`` build every tensor on ``self.device`` (``:225-247``, ``:267-278``). Its CUDA cell is therefore
timed like every other row's.

What this module does NOT do
----------------------------
It trains nothing, evaluates nothing and records no ATT, reward, return, queue or any other episode quantity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "FORMAT_VERSION",
    "CANARY_FORMAT_VERSION",
    "RUN_FORMAT_VERSION",
    "PREFLIGHT_FORMAT_VERSION",
    "TIMING_DRAWS",
    "ENGINE_SEED",
    "WARMUP",
    "DECISIONS_PER_EPISODE",
    "MIN_TIMED",
    "MAX_ATTEMPTS",
    "FORBIDDEN_KEY_TOKENS",
    "ScenarioSpec",
    "SCENARIOS",
    "LatencyRow",
    "ROWS",
    "row_by_id",
    "time_episode",
    "latency_stats",
    "find_outcome_keys",
    "assert_cpu_regime",
    "configure_regime",
    "POWER_SUPPLY_ROOT",
    "POWER_REFUSED_EXIT",
    "read_power_supplies",
    "read_windows_power",
    "power_block",
    "power_regime_problems",
    "assert_power_regime",
    "assert_no_tracer",
    "write_once",
    "build_record",
    "build_canary_record",
    "run_row",
    "supervise",
    "run_jobs",
    "build_run_record",
    "write_manifest",
    "run_all",
    "derive_timeouts",
    "expected_duration",
    "run_preflight",
    "canary_timeout_from",
    "timeouts_from_preflight",
    "build_parser",
    "main",
]

FORMAT_VERSION = "p8.2-latency/1.0"
CANARY_FORMAT_VERSION = "p8.2-canary/1.0"
RUN_FORMAT_VERSION = "p8.2-latency-run/1.0"
PREFLIGHT_FORMAT_VERSION = "p8.2-preflight/1.0"

#: Q8: three held-out draws, one episode each. No outcome is recorded (T-no-outcome), and the held-out draws are the
#: deployment distribution.
TIMING_DRAWS: tuple[int, ...] = (1000, 1001, 1002)

#: The evaluations' engine seed: ``horizon_rollout`` resets with ``seed + ep`` and ep is 0 for one episode per env.
ENGINE_SEED = 1000

#: §5.2: decisions 0..19 of every episode are warm-up (the K = 20 window is full from index 19).
WARMUP = 20

#: ``envs/base_traffic_env.py:604-605``: never terminated, truncated at ``max_steps`` = 360.
DECISIONS_PER_EPISODE = 360

#: The brief: the p95 must rest on at least 1,000 timed decisions; 3 x (360 - 20) = 1,020.
MIN_TIMED = 1000

#: Amendment A3.1: at most three attempts per supervised process.
MAX_ATTEMPTS = 3

#: The machine-health canary's ceiling (``offline.transfer_calibration.CANARY_MAX_SECONDS``, asserted equal in the
#: tests so the two literals cannot drift apart).
CANARY_MAX_SECONDS = 2.0

#: The digest of ``docs/data/p7_3d_calibration.json``, the fine-tunes' registered prompts (``p7_3c_finetune.json
#: $.calibration_sha256``).
CALIBRATION_FILE = "p7_3d_calibration.json"
CALIBRATION_SHA256 = "3e9df8eed4af2e42c132087e711751bc4c265edcef75dd88e24e6143e82f9723"

#: T-no-outcome: a record key whose ``_``-separated tokens include one of these names an episode quantity.
FORBIDDEN_KEY_TOKENS: tuple[str, ...] = (
    "att",
    "reward",
    "rewards",
    "return",
    "returns",
    "travel",
    "queue",
    "queues",
    "waiting",
    "vehicle",
    "vehicles",
    "throughput",
    "delay",
    "completed",
    "entered",
    "metric",
    "metrics",
    "rho",
    "pressure",
)

#: The record formats a supervised process may leave behind.
_RECORD_FORMATS = frozenset({FORMAT_VERSION, CANARY_FORMAT_VERSION})

_MODULE_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ScenarioSpec:
    """One CityFlow scenario of the table: the draws' scenario key, the env id and the intersection count."""

    name: str
    scenario_key: str
    scenario_id: str
    n_intersections: int


SCENARIOS: Mapping[str, ScenarioSpec] = {
    "hz1x1": ScenarioSpec("hz1x1", "cityflow1x1", "cityflow1x1", 1),
    "grid4x4": ScenarioSpec("grid4x4", "cityflow_grid4x4", "cityflow_grid4x4", 16),
}


@dataclass(frozen=True)
class LatencyRow:
    """One row of the latency registry: what is timed, through which factory, with which prompt and settings.

    ``checkpoint`` is relative to the MAIN tree (the output root's parent); ``sha256`` is the representative
    checkpoint's digest (seed 101 of the row's headline tier, ``docs/plans/p8.2.md`` Appendix A) and
    ``sha256_source`` names the record that holds it as ``(kind, file, locator)``: ``("json", <file under the repo
    root>, <JSON path>)``, ``("sums", <manifest under the output root>, <listed relative path>)`` or
    ``("corpus_manifest", <manifest under the corpus root>, <JSON path>)``. ``settings_corpus_dir`` is the collection
    whose manifest gives the row's evaluation env settings (``offline.dt_gate.env_settings_from_manifest``).
    """

    row_id: str
    scenario: str
    kind: str
    checkpoint: str | None
    sha256: str | None
    sha256_source: tuple[str, str, str] | None
    method: str | None
    declared_gradient_steps: int | None
    target_rtg: float | None
    target_source: str | None
    fine_tune_k: int | None
    settings_corpus_dir: str
    devices: tuple[str, ...]


_TORCH = ("cpu", "cuda")
_CPU = ("cpu",)
_HZ = "cf_hz1x1__mappo1000__seed101"
_GRID = "cf_grid4x4__mappo1000__seed101"
_P4_TARGET = -5762.0
_P4_TARGET_SOURCE = (
    "offline.method_tier_grid.TIERS['mappo1000'].target_rtg, == docs/data/p4_training.json $.target_rtg "
    "(the registered prompt of the mappo1000 tier; A26 keeps it for every K)"
)
_FT_TARGET_SOURCE = (
    "docs/data/p7_3d_calibration.json $.per_intersection.<ix>.budgets['k<k>'].target (the rule each fine-tune "
    "records in provenance.few_shot.target_rule)"
)


def _dt(row_id: str, checkpoint: str, sha256: str, source: tuple[str, str, str], *, target: float = _P4_TARGET,
        target_source: str = _P4_TARGET_SOURCE) -> LatencyRow:
    return LatencyRow(row_id, "hz1x1", "dt", checkpoint, sha256, source, None, 40000, target, target_source, None,
                      _HZ, _TORCH)


def _baseline(row_id: str, scenario: str, method: str, checkpoint: str, sha256: str,
              source: tuple[str, str, str]) -> LatencyRow:
    settings = _HZ if scenario == "hz1x1" else _GRID
    return LatencyRow(row_id, scenario, "baseline", checkpoint, sha256, source, method, 40000, None, None, None,
                      settings, _TORCH)


def _spatial(row_id: str, method: str, checkpoint: str, sha256: str, source: tuple[str, str, str]) -> LatencyRow:
    return LatencyRow(row_id, "grid4x4", "spatial_dt", checkpoint, sha256, source, method, 40000, None,
                      "the checkpoint's own per-node prompt (as P5.1 / P5.2 evaluated)", None, _GRID, _TORCH)


def _fine_tune(subject: str, k: int, steps: int, sha256: str) -> LatencyRow:
    return LatencyRow(
        f"grid4x4.c3.{subject}", "grid4x4", "spatial_dt_targets", f"output/p7_3c_training/checkpoints/{subject}_seed101.pt",
        sha256, ("json", "docs/data/p7_3c_finetune.json", f"$.runs['{subject}_seed101'].checkpoint_sha256"),
        "dt_nomix_h4", steps, None, _FT_TARGET_SOURCE, k, _GRID, _TORCH,
    )


def _mappo(scenario: str, budget: str, checkpoint_dir: str, sha256: str) -> LatencyRow:
    corpus_dir = f"cf_{scenario}__mappo{budget}__seed101"
    return LatencyRow(
        f"{scenario}.mappo{budget}", scenario, "mappo",
        f"output/checkpoints/{checkpoint_dir}/cf_{scenario}__mappo__seed101.pt", sha256,
        ("corpus_manifest", f"{corpus_dir}/manifest.json", "$.run_metadata.checkpoint_sha256"),
        None, None, None, None, None, corpus_dir, _TORCH,
    )


def _heuristic(scenario: str, kind: str) -> LatencyRow:
    return LatencyRow(f"{scenario}.{kind}", scenario, kind, None, None, None, None, None, None, None, None,
                      f"cf_{scenario}__{kind}", _CPU)


_P53C = "docs/data/p5_3c_train.json"


def _h4(name: str, arm: str, sha256: str) -> LatencyRow:
    return _dt(f"hz1x1.h4.{name}", f"output/p5_3c_training/checkpoints/{arm}_seed101.pt", sha256,
               ("json", _P53C, f"$.runs['{arm}_seed101'].checkpoint_sha256"))


#: The 42 latency rows of ``docs/plans/p8.2.md`` §3 (the 36 checkpoint rows of Appendix A and six heuristics), in the
#: run's order.
ROWS: tuple[LatencyRow, ...] = (
    _dt("hz1x1.dt_k20", "output/p4_dt/dt_seed101.pt",
        "dc6fc97cf1cb3f5c61d270f611e4d3d935a1fcbbb9ca546ee0ffa281799531b6",
        ("json", "docs/data/p4_gate.json", "$.checkpoints['101'].sha256")),
    _baseline("hz1x1.bc", "hz1x1", "bc", "output/p4_4/checkpoints/bc_seed101.pt",
              "6bb158865a7b260fc5eaf069241e3e1921942aebfda016397be177428686c5c4",
              ("json", "docs/data/p4_4_training.json", "$.runs[0].file_sha256")),
    _baseline("hz1x1.bc_top10", "hz1x1", "bc_top10", "output/p4_4/checkpoints/bc_top10_seed101.pt",
              "b6309b2b9b7eaba7bd2c3fdf32b6c716f86e173f1aa731daaeff87089731ca2e",
              ("json", "docs/data/p4_4_training.json", "$.runs[5].file_sha256")),
    _baseline("hz1x1.iql", "hz1x1", "iql", "output/p4_4/checkpoints/iql_seed101.pt",
              "64cb9516d3e5b67e91b7e01c4f796763a6b27ae5b7d54f46a3090ea9a86466b5",
              ("json", "docs/data/p4_4_training.json", "$.runs[10].file_sha256")),
    _baseline("hz1x1.bc_best2_20", "hz1x1", "bc_best2_20", "output/p4_5/checkpoints/bc_best2_20_seed101.pt",
              "28ffb936de2ee71c3f57f9da5e71be617b817286dd9043c50fb9542b1b9f478a",
              ("json", "docs/data/p4_5_selection.json", "$.runs[0].file_sha256")),
    _baseline("hz1x1.bc_any_20", "hz1x1", "bc_any_20", "output/p4_5/checkpoints/bc_any_20_seed101.pt",
              "8726da780a949355f31522d9ed04913214a30290c22a4471d166e0df5862bbcc",
              ("json", "docs/data/p4_5_selection.json", "$.runs[5].file_sha256")),
    _baseline("hz1x1.bc_worst2_20", "hz1x1", "bc_worst2_20", "output/p4_5/checkpoints/bc_worst2_20_seed101.pt",
              "7ddb6e009b28ab15edec130f7c2986e4860846824ca858d4b7f98b1561b1d89e",
              ("json", "docs/data/p4_5_selection.json", "$.runs[10].file_sha256")),
    _baseline("hz1x1.bc_best2_all", "hz1x1", "bc_best2_all", "output/p4_5/checkpoints/bc_best2_all_seed101.pt",
              "d94de30b3cb4abd137bd5af913d05d1c1e9e9ee12d93a3f2ac6f7ae283d16dac",
              ("json", "docs/data/p4_5_selection.json", "$.runs[15].file_sha256")),
    _dt("hz1x1.dt_nortg", "output/p5_3b/checkpoints/mappo1000_dt_nortg_seed101.pt",
        "372b82ee74dabab3658712536b9197cf01545ef90136eb10d48e73a199caec0e",
        ("sums", "SHA256SUMS_p5_3b.txt", "p5_3b/checkpoints/mappo1000_dt_nortg_seed101.pt")),
    _h4("k1", "mappo1000_k1_b64", "3801385031962bd386310f542f442a4aab3ab8f90b8a256f7f98d717539cf6b9"),
    _h4("k2", "mappo1000_k2_b64", "f6fe9e78adf17442039c72a8ca56b8feb79bf03e4770e4e82a83dca6bfb7a593"),
    _h4("k5", "mappo1000_k5_b64", "2f31bd44d07c95256a75534f162d6aa6ce6c5072a8e5acfcaf796f8b3c14b5cc"),
    _h4("k10", "mappo1000_k10_b64", "4463ed3fe3d1829556fcb10ecc976346ab25517ecbd0db74d8e2bfed47650fb5"),
    _h4("k20", "mappo1000_k20_b64", "ce299be108c91638e2b269e1a070883b00e5fc3fd55c34eb93ecb6a02f169a05"),
    _h4("k1_b1280", "mappo1000_k1_b1280", "14af3f5217d824658bee699696b480d92cf02e8d234348407407c6e57e5eee08"),
    _h4("k2_b640", "mappo1000_k2_b640", "a549fa2202a2ef8af90da0c2af5b8c03f5b462888a4a008717d189c8c8e7d054"),
    _dt("hz1x1.c3.anchor_k200", "output/p7_3b_anchor/checkpoints/anchor_dt_seed101.pt",
        "7a717b99526d47a94039a6793ad22be011272ba62e5316cbfdaa7736e803bb2c",
        ("json", "docs/data/p7_3b_anchor_training.json", "$.seeds[0].checkpoint_sha256"),
        target=-20625.0,
        target_source="the anchor's own prompt: docs/data/p7_3b_anchor_training.json $.target_rtg (naive in-domain)"),
    _mappo("hz1x1", "1000", "p2_1_mappo_nominal_500/p2_1_mappo_nominal_1000",
           "2efaa6d759685fef6a05df381a3af6dbcd15c67caa22c154754b752552d4d493"),
    _mappo("hz1x1", "500", "p2_1_mappo_nominal_500/p2_1_mappo_nominal_500",
           "a64063b8f300ff810c4b044bf6375d2e656baf09389381f86c74eb0fc6ebf31c"),
    _mappo("hz1x1", "060", "p2_1_mappo_nominal_060/p2_1_mappo_nominal_060",
           "865cabc8f8e4d61149db5914e139baa92a1824e72bf1cbd53aa32c030d48ce45"),
    _heuristic("hz1x1", "maxpressure"),
    _heuristic("hz1x1", "fixedtime"),
    _heuristic("hz1x1", "random"),
    _spatial("grid4x4.dt_spatial", "dt_spatial", "output/p5_1/checkpoints/grid4x4_mappo1000_dt_spatial_seed101.pt",
             "b2a66801c5acd07c208e51af0a3519ca0e72d949bc4c43c7bf97f31333742881",
             ("sums", "SHA256SUMS_p5_1.txt", "p5_1/checkpoints/grid4x4_mappo1000_dt_spatial_seed101.pt")),
    _spatial("grid4x4.dt_nomix", "dt_nomix", "output/p5_1/checkpoints/grid4x4_mappo1000_dt_nomix_seed101.pt",
             "8a8b8f5b92867eb36c70807259e3d6c4ca8235cc0004ed8841d610be93aaf57b",
             ("sums", "SHA256SUMS_p5_1.txt", "p5_1/checkpoints/grid4x4_mappo1000_dt_nomix_seed101.pt")),
    _spatial("grid4x4.dt_spatial_h4", "dt_spatial_h4",
             "output/p5_2/checkpoints/grid4x4_mappo1000_dt_spatial_h4_seed101.pt",
             "4d52b8b15502d398d0ee0f2a40abac83421f7082f2e40182a03a4a3743f50477",
             ("sums", "SHA256SUMS_p5_2.txt", "p5_2/checkpoints/grid4x4_mappo1000_dt_spatial_h4_seed101.pt")),
    _spatial("grid4x4.dt_nomix_h4", "dt_nomix_h4", "output/p5_2/checkpoints/grid4x4_mappo1000_dt_nomix_h4_seed101.pt",
             "329fb6b87fc8cf5c27530b2fe4d45212ea9c0775db3fb1e636a6bbc0a5004279",
             ("json", "docs/data/p7_3d_grid4x4.json", "$.inputs.checkpoints[0].file_sha256")),
    _baseline("grid4x4.bc", "grid4x4", "bc", "output/p5_1/checkpoints/grid4x4_mappo1000_bc_seed101.pt",
              "c77a83a3288941d66209b824746f1db17ddbee5cea9bb8ba35cc47ba6a8b07c6",
              ("sums", "SHA256SUMS_p5_1.txt", "p5_1/checkpoints/grid4x4_mappo1000_bc_seed101.pt")),
    _baseline("grid4x4.bc_top10", "grid4x4", "bc_top10", "output/p5_1/checkpoints/grid4x4_mappo1000_bc_top10_seed101.pt",
              "99464cd838e7e3f9b37ff11a1474ae7df4758b7da963d4259306bbee4453a547",
              ("sums", "SHA256SUMS_p5_1.txt", "p5_1/checkpoints/grid4x4_mappo1000_bc_top10_seed101.pt")),
    _baseline("grid4x4.bc_top10_perix", "grid4x4", "bc_top10_perix",
              "output/p5_2/checkpoints/grid4x4_mappo1000_bc_top10_perix_seed101.pt",
              "6da6be30742bfad6a5fdec98400ed41c45b39ea26d13e311404e6033e526cdfe",
              ("sums", "SHA256SUMS_p5_2.txt", "p5_2/checkpoints/grid4x4_mappo1000_bc_top10_perix_seed101.pt")),
    _baseline("grid4x4.iql", "grid4x4", "iql", "output/p5_1/checkpoints/grid4x4_mappo1000_iql_seed101.pt",
              "d1dcb1de0bb151cf4ceaacfa316fa17a296b01901168abcb88af9af73d57ae5b",
              ("sums", "SHA256SUMS_p5_1.txt", "p5_1/checkpoints/grid4x4_mappo1000_iql_seed101.pt")),
    _fine_tune("ft_k5", 5, 4000, "5785178ee1482ff55b016918b999c8e93a98839e51de4492c03eb24920825f4b"),
    _fine_tune("ft_k20", 20, 4000, "b0b7c00634a076cb5bd48d6be108494c657ad8d1152096d52051c7e25c08887b"),
    _fine_tune("ft_k100", 100, 4000, "0c9540a56c4ac97b57b48b3730c6aed3d9af0cc7631419fcd21bb44d1afe00f0"),
    _fine_tune("ft_k100_b1000", 100, 1000, "f019d532782b7812d3b5627952f96e1168491813ba306ccdbc46f696f1e43cb3"),
    _fine_tune("ft_k100_b16000", 100, 16000, "66d8a3440ec8ec2c17332198c7bf575be469349503cea1f592f36270d5815705"),
    _fine_tune("scratch_k100", 100, 4000, "2568633111c864786511c06f6771bd0945f7864da158a291a84973594031b03d"),
    _mappo("grid4x4", "1000", "p2_1_mappo_nominal_500/p2_1_mappo_nominal_1000",
           "490e5706b48c47ca3cefdf64b0f216cc781ba3fac160fba6f6894046e8f19236"),
    _mappo("grid4x4", "060", "p2_1_mappo_nominal_060/p2_1_mappo_nominal_060",
           "353cc5d8470a8cedd92a0ed2e0748c21597882b9685bca833b3305b61138aa72"),
    _heuristic("grid4x4", "maxpressure"),
    _heuristic("grid4x4", "fixedtime"),
    _heuristic("grid4x4", "random"),
)


def row_by_id(row_id: str) -> LatencyRow:
    """The registry row named *row_id*; refuses an unknown id."""
    for row in ROWS:
        if row.row_id == row_id:
            return row
    raise KeyError(f"{row_id!r} is not a latency row; the registry holds {[row.row_id for row in ROWS]}")


# ----------------------------------------------------------------------
# The measurement
# ----------------------------------------------------------------------


def time_episode(
    env: Any,
    choose: Callable[[Any, dict[str, Any]], Any],
    *,
    engine_seed: int = ENGINE_SEED,
    sync: Callable[[], None] | None = None,
    clock: Callable[[], int] | None = None,
) -> list[int]:
    """Roll one episode and return the nanoseconds of every decision call, in order.

    Only ``choose(env, info)`` sits between the two clock readings; ``env.step`` is outside. ``sync`` (CUDA's
    ``torch.cuda.synchronize``) runs immediately before BOTH readings when given.
    """
    read = time.perf_counter_ns if clock is None else clock
    max_steps = int(env.max_steps)
    info = env.reset(seed=int(engine_seed))
    decision_ns: list[int] = []
    for _ in range(max_steps):
        if sync is not None:
            sync()
        started = read()
        action = choose(env, info)
        if sync is not None:
            sync()
        decision_ns.append(int(read() - started))
        _discarded, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return decision_ns


def latency_stats(per_episode_ns: Sequence[Sequence[int]], *, warmup: int = WARMUP) -> dict[str, Any]:
    """Statistics over the timed decisions: the first *warmup* of EACH episode are excluded by count.

    ``median_ns`` is ``numpy.median``; ``p95_ns`` is the nearest-rank value (``numpy.percentile(..., method=
    "inverted_cdf")``, an observed value). Milliseconds are the same values divided by 10**6.
    """
    import numpy as np

    timed = [int(value) for episode in per_episode_ns for value in list(episode)[int(warmup):]]
    if not timed:
        raise ValueError("no decision is left to time once the warm-up is excluded")
    values = np.asarray(timed, dtype=np.int64)
    median_ns = float(np.median(values))
    p95_ns = int(np.percentile(values, 95, method="inverted_cdf"))
    return {
        "n_timed": len(timed),
        "warmup": int(warmup),
        "median_ns": median_ns,
        "p95_ns": p95_ns,
        "median_ms": median_ns / 1e6,
        "p95_ms": p95_ns / 1e6,
    }


def find_outcome_keys(payload: Any) -> list[str]:
    """Every key, at any depth, whose ``_``-separated tokens include one of :data:`FORBIDDEN_KEY_TOKENS`."""
    found: list[str] = []
    _walk_keys(payload, "", found)
    return found


def _walk_keys(node: Any, trail: str, found: list[str]) -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            here = f"{trail}.{key}" if trail else str(key)
            if any(token in FORBIDDEN_KEY_TOKENS for token in str(key).lower().split("_")):
                found.append(here)
            _walk_keys(value, here, found)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _walk_keys(value, f"{trail}[{index}]", found)


# ----------------------------------------------------------------------
# The regime
# ----------------------------------------------------------------------


def assert_cpu_regime(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Refuse unless torch runs ONE thread, ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` are ``"1"`` and
    ``CUBLAS_WORKSPACE_CONFIG`` is unset; return the regime block a record carries."""
    import torch

    env = os.environ if environ is None else environ
    threads = int(torch.get_num_threads())
    if threads != 1:
        raise ValueError(
            f"the timing must run one torch thread; torch reports {threads}. Call torch.set_num_threads(1) first"
        )
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if env.get(variable) != "1":
            raise ValueError(
                f"{variable} must be '1' in the process environment, set before torch is imported; it is "
                f"{env.get(variable)!r}"
            )
    if env.get("CUBLAS_WORKSPACE_CONFIG") is not None:
        raise ValueError(
            f"CUBLAS_WORKSPACE_CONFIG must be unset for the timing; it is {env.get('CUBLAS_WORKSPACE_CONFIG')!r}"
        )
    return {
        "torch_num_threads": threads,
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": None,
    }


def configure_regime(device: str) -> dict[str, Any]:
    """Pin torch to one thread, then :func:`assert_cpu_regime` -- for the CPU and the CUDA rows alike."""
    import torch

    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', not {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA row was requested on a machine where torch sees no CUDA device")
    torch.set_num_threads(1)
    block = assert_cpu_regime()
    block["device"] = device
    block["synchronize"] = "torch.cuda.synchronize() before both clock readings" if device == "cuda" else None
    return block


# ----------------------------------------------------------------------
# The power regime (Amendment B, B1), the device evidence (B2), the tracer refusal (B5.3)
# ----------------------------------------------------------------------

#: B1: where the guest kernel lists the machine's power supplies (``type``, ``online``); visible from WSL2.
POWER_SUPPLY_ROOT = Path("/sys/class/power_supply")

#: B1.3: the exit code of a canary process that refuses because the power regime is not mains + Best Performance.
POWER_REFUSED_EXIT = 3


def read_power_supplies(root: Path = POWER_SUPPLY_ROOT) -> dict[str, Any]:
    """Every supply under *root* with its ``type`` and ``online``, how they were read, and the error if none could be."""
    raise NotImplementedError("Amendment B, B1.1: the power supplies")


def read_windows_power(query: Callable[[], str] | None = None) -> dict[str, Any]:
    """The Windows active power scheme and the AC / DC power-mode overlays, read with ``reg.exe query``."""
    raise NotImplementedError("Amendment B, B1.1: the Windows power mode")


def power_block(*, supply_root: Path = POWER_SUPPLY_ROOT, reg_query: Callable[[], str] | None = None) -> dict[str, Any]:
    """The record's ``power`` block: the supplies and the Windows power mode, each with how it was read."""
    raise NotImplementedError("Amendment B, B1.1: the power block")


def power_regime_problems(block: Mapping[str, Any]) -> list[str]:
    """Why *block* is not mains + Best Performance (empty when it is); a source that could not be read is a problem."""
    raise NotImplementedError("Amendment B, B1.2: the power regime")


def assert_power_regime(block: Mapping[str, Any]) -> None:
    """Refuse unless *block* is mains + Best Performance."""
    raise NotImplementedError("Amendment B, B1.2: the power regime")


def _cuda_state() -> dict[str, Any]:
    """``torch.cuda.is_initialized()`` and, when initialised, ``torch.cuda.max_memory_allocated()``."""
    raise NotImplementedError("Amendment B, B2: the device evidence")


def assert_no_tracer() -> str:
    """Refuse an active tracer or profiler in the measured process; return what was checked."""
    raise NotImplementedError("Amendment B, B5.3: the tracer refusal")


def _load_average() -> list[float]:
    try:
        return [float(value) for value in Path("/proc/loadavg").read_text().split()[:3]]
    except OSError:
        return []


def _machine_block() -> dict[str, Any]:
    import numpy
    import torch

    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu_model = None
    gpu_name = gpu_driver = None
    gpu_query = "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"
    try:
        completed = subprocess.run(gpu_query.split(), capture_output=True, text=True, timeout=30, check=False)
        if completed.returncode == 0 and completed.stdout.strip():
            gpu_name, gpu_driver = [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",", 1)]
        else:
            gpu_query += f" (exit {completed.returncode})"
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu_query += f" (unavailable: {type(exc).__name__})"
    return {
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "gpu_name": gpu_name,
        "gpu_driver": gpu_driver,
        "gpu_query": gpu_query,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "numpy_version": numpy.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(["git", "-C", str(_MODULE_ROOT), *args], capture_output=True, text=True,
                                       timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if status is None else bool(status), "code_root": str(_MODULE_ROOT)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------


def write_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Write *payload* as JSON to *path* atomically and exclusively.

    The bytes go to a temporary file under ``<parent>/partial/`` first and are then hard-linked to *path*; an
    existing *path* refuses and is left untouched, and a failure before the link leaves no file at *path*.
    """
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: a record is written once and never replaced")
    partial = path.parent / "partial"
    partial.mkdir(parents=True, exist_ok=True)
    temporary = partial / f"{path.name}.{os.getpid()}.tmp"
    with open(temporary, "x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        temporary.unlink()
        raise FileExistsError(f"{path} appeared while this record was being written; it is left untouched") from None
    temporary.unlink()


def build_record(
    row: LatencyRow,
    device: str,
    per_episode_ns: Sequence[Sequence[int]],
    *,
    draws: Sequence[int],
    prompt: Mapping[str, Any] | None,
    factory: str,
    regime: Mapping[str, Any],
    machine: Mapping[str, Any],
    load_before: Sequence[float],
    load_after: Sequence[float],
    git: Mapping[str, Any],
    warmup: int = WARMUP,
    device_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``p8.2-latency/1.0`` record; refuses an episode whose decision count is not 360, fewer than
    :data:`MIN_TIMED` timed decisions, or any key that names an episode quantity."""
    if device_evidence is not None:
        raise NotImplementedError("Amendment B, B2: the record's device evidence")
    if device not in row.devices:
        raise ValueError(f"{row.row_id} is timed on {row.devices}, not on {device!r}")
    if len(per_episode_ns) != len(draws):
        raise ValueError(f"{row.row_id} on {device}: {len(per_episode_ns)} episodes for {len(draws)} draws")
    for draw, episode in zip(draws, per_episode_ns):
        if len(episode) != DECISIONS_PER_EPISODE:
            raise ValueError(
                f"{row.row_id} on {device}: draw {draw} took {len(episode)} decisions, not {DECISIONS_PER_EPISODE}"
            )
    stats = latency_stats(per_episode_ns, warmup=warmup)
    if stats["n_timed"] < MIN_TIMED:
        raise ValueError(
            f"{row.row_id} on {device}: {stats['n_timed']} timed decisions, fewer than the {MIN_TIMED} the p95 needs"
        )
    n_intersections = SCENARIOS[row.scenario].n_intersections
    record: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "row": row.row_id,
        "scenario": row.scenario,
        "device": device,
        "kind": row.kind,
        "checkpoint": None if row.checkpoint is None else {
            "path": row.checkpoint,
            "sha256": row.sha256,
            "named_by": list(row.sha256_source or ()),
        },
        "factory": factory,
        "prompt": None if prompt is None else dict(prompt),
        "draws": [int(draw) for draw in draws],
        "engine_seed": ENGINE_SEED,
        "warmup": int(warmup),
        "max_steps": DECISIONS_PER_EPISODE,
        "episodes": [
            {"draw": int(draw), "decision_ns": [int(value) for value in episode]}
            for draw, episode in zip(draws, per_episode_ns)
        ],
        "n_timed": stats["n_timed"],
        "median_ns": stats["median_ns"],
        "p95_ns": stats["p95_ns"],
        "median_ms": stats["median_ms"],
        "p95_ms": stats["p95_ms"],
        "statistics": {
            "timed": f"decisions {int(warmup)}..{DECISIONS_PER_EPISODE - 1} of each episode, pooled",
            "median": "numpy.median",
            "p95": "numpy.percentile(..., 95, method='inverted_cdf'): the nearest rank, an observed value",
        },
        "n_intersections": n_intersections,
        "per_intersection": {
            "median_ms": stats["median_ms"] / n_intersections,
            "p95_ms": stats["p95_ms"] / n_intersections,
            "note": "derived: the whole-decision statistic divided by the intersection count, not a separate timing",
        },
        "regime": dict(regime),
        "machine": dict(machine),
        "load_before": [float(value) for value in load_before],
        "load_after": [float(value) for value in load_after],
        "git": dict(git),
        "written_utc": _utc_now(),
    }
    named = find_outcome_keys(record)
    if named:
        raise ValueError(f"{row.row_id} on {device}: the record would carry keys that name an episode quantity {named}")
    return record


def build_canary_record(phase: str, seconds: float, *, reproduced: bool, git: Mapping[str, Any],
                        power: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``p8.2-canary/1.0`` record: seconds, the 2.0 s threshold, the verdict and ``reproduced`` -- no value of
    the canary episode itself."""
    if power is not None:
        raise NotImplementedError("Amendment B, B1.3: the canary record's power block")
    if phase not in ("open", "close"):
        raise ValueError(f"the canary phase is 'open' or 'close', not {phase!r}")
    record = {
        "format_version": CANARY_FORMAT_VERSION,
        "phase": phase,
        "seconds": float(seconds),
        "threshold_seconds": CANARY_MAX_SECONDS,
        "verdict": "at speed" if float(seconds) <= CANARY_MAX_SECONDS else "throttled",
        "reproduced": bool(reproduced),
        "recipe": (
            "offline.transfer_calibration.canary_seconds(): rtg_calibration.run_probe on cityflow1x1 draw 0 with "
            "docs/data/p4_3_probe.json's settings; 'reproduced' is offline.transfer_calibration.check_canary under ==. "
            "The episode's own values are compared and discarded, never written"
        ),
        "git": dict(git),
        "written_utc": _utc_now(),
    }
    named = find_outcome_keys(record)
    if named:
        raise ValueError(f"the canary record would carry keys that name an episode quantity {named}")
    return record


# ----------------------------------------------------------------------
# The row process
# ----------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluation_inputs(row: LatencyRow, corpus_root: Path, draws_root: Path) -> tuple[dict[str, Any], dict[int, Path]]:
    from offline.dt_gate import env_settings_from_manifest
    from offline.materialise_draws import draw_config_path

    manifest = Path(corpus_root) / row.settings_corpus_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"{row.row_id}: the settings manifest {manifest} does not exist")
    settings = env_settings_from_manifest(manifest)
    scenario = SCENARIOS[row.scenario]
    configs = {
        int(draw): Path(draw_config_path(scenario.scenario_key, int(draw), out_root=draws_root))
        for draw in TIMING_DRAWS
    }
    missing = [str(path) for path in configs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{row.row_id}: the timing draws are not materialised: {missing}")
    return settings, configs


def _make_env(row: LatencyRow, settings: Mapping[str, Any], config: Path) -> Any:
    from experiments.config import EnvSpec
    from experiments.envs import make_env

    return make_env(
        EnvSpec(
            id=SCENARIOS[row.scenario].scenario_id,
            backend="cityflow",
            paths={"config": str(config)},
            settings=dict(settings),
        )
    )


def _verified_checkpoint(row: LatencyRow, output_root: Path) -> Path:
    path = Path(output_root).parent / str(row.checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"{row.row_id}: {path} does not exist")
    digest = _sha256_file(path)
    if digest != row.sha256:
        raise ValueError(f"{row.row_id}: {path} has sha256 {digest}, not the registered {row.sha256}")
    return path


def _calibrated_targets(data_dir: Path, k: int, payload_targets: Mapping[str, float]) -> dict[str, float]:
    path = Path(data_dir) / CALIBRATION_FILE
    digest = _sha256_file(path)
    if digest != CALIBRATION_SHA256:
        raise ValueError(f"{path}: sha256 {digest} is not the pinned {CALIBRATION_SHA256}")
    calibration = json.loads(path.read_text(encoding="utf-8"))
    targets = {
        str(ix): float(entry["budgets"][f"k{int(k)}"]["target"]) for ix, entry in calibration["per_intersection"].items()
    }
    recorded = {str(ix): float(value) for ix, value in payload_targets.items()}
    if targets != recorded:
        raise ValueError(
            f"the calibrated k{k} targets of {CALIBRATION_FILE} are not the targets the fine-tune records; refusing "
            "to time a checkpoint under a prompt it was not evaluated with"
        )
    return targets


def _row_factory(
    row: LatencyRow, device: str, *, output_root: Path, corpus_root: Path, data_dir: Path, first_config: Path | None
) -> tuple[Callable[[Any], Callable[[Any, dict[str, Any]], Any]], dict[str, Any] | None, str]:
    """The row's OWN evaluation factory, its prompt and its qualified name -- imported, never reimplemented."""
    kind = row.kind
    if kind == "maxpressure":
        from offline.dt_gate import _maxpressure_factory

        return _maxpressure_factory, None, "offline.dt_gate._maxpressure_factory"
    if kind == "fixedtime":
        from offline.method_tier_grid import _fixedtime_factory, fixedtime_collection_settings

        collected = fixedtime_collection_settings(Path(corpus_root) / row.settings_corpus_dir / "manifest.json")
        if first_config is None:
            raise ValueError("the fixed-time factory resolves its plan from the first timing draw's config")
        return (
            _fixedtime_factory(first_config, collected),
            None,
            "offline.method_tier_grid._fixedtime_factory (plan hash asserted against the corpus manifest)",
        )
    if kind == "random":
        from offline.method_tier_grid import _random_factory

        return _random_factory(101), None, "offline.method_tier_grid._random_factory(101)"

    path = _verified_checkpoint(row, output_root)
    if kind == "dt":
        from offline.method_tier_grid import _dt_factory

        return (
            _dt_factory(str(path), int(row.declared_gradient_steps or 0), float(row.target_rtg or 0.0), device),
            {"target": float(row.target_rtg or 0.0), "source": row.target_source},
            "offline.method_tier_grid._dt_factory -> offline.rtg_calibration.agent_with_target; "
            "act(info, explore=False, update_memory=True)",
        )
    if kind == "baseline":
        from offline.offline_baselines import _baseline_factory

        return (
            _baseline_factory(str(row.method), str(path), int(row.declared_gradient_steps or 0), device),
            None,
            f"offline.offline_baselines._baseline_factory({row.method!r}); act(info, explore=False, update_memory=False)",
        )
    if kind == "mappo":
        from offline.dt_gate import _mappo_factory

        return (
            _mappo_factory(str(path), device),
            None,
            "offline.dt_gate._mappo_factory; act(info, explore=False, update_memory=False)",
        )
    if kind == "spatial_dt":
        return _spatial_factory(row, path, device), {"target": "the checkpoint's own", "source": row.target_source}, (
            "agent.SpatialDTAgent.from_checkpoint after offline.spatial_mixing.assert_declared_budget, as "
            "offline.admission_probe._method_factory; act(info, explore=False, update_memory=True)"
        )
    if kind == "spatial_dt_targets":
        import torch

        from offline.rtg_calibration import spatial_agent_with_targets

        payload = torch.load(path, map_location="cpu", weights_only=False)
        targets = _calibrated_targets(data_dir, int(row.fine_tune_k or 0), payload["target_rtg"])

        def fine_tune(env: Any) -> Callable[[Any, dict[str, Any]], Any]:
            agent = spatial_agent_with_targets(
                env, path, declared_gradient_steps=int(row.declared_gradient_steps or 0), targets=targets,
                method=str(row.method), device=device,
            )
            return lambda _env, info: agent.act(info, explore=False, update_memory=True)

        return fine_tune, {"targets": targets, "source": row.target_source}, (
            "offline.rtg_calibration.spatial_agent_with_targets; act(info, explore=False, update_memory=True)"
        )
    raise ValueError(f"{row.row_id}: no factory is declared for kind {kind!r}")


def _spatial_factory(row: LatencyRow, path: Path, device: str) -> Callable[[Any], Callable[[Any, dict[str, Any]], Any]]:
    def factory(env: Any) -> Callable[[Any, dict[str, Any]], Any]:
        import torch

        from agent.SpatialDTAgent import SpatialDTAgent
        from offline.spatial_mixing import assert_declared_budget

        assert_declared_budget(path, int(row.declared_gradient_steps or 0), str(row.method))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected_mixing = str(row.method).startswith("dt_spatial")
        if bool(payload["config"]["spatial_mixing"]) is not expected_mixing:
            raise ValueError(f"{row.row_id}: {path} records spatial_mixing={payload['config']['spatial_mixing']}")
        agent = SpatialDTAgent.from_checkpoint(env, str(path), device=device)
        return lambda _env, info: agent.act(info, explore=False, update_memory=True)

    return factory


def _cuda_sync() -> None:
    import torch

    torch.cuda.synchronize()


def run_row(
    row: LatencyRow,
    device: str,
    *,
    out_dir: Path,
    output_root: Path,
    corpus_root: Path,
    draws_root: Path,
    data_dir: Path,
    env_builder: Callable[[int], Any] | None = None,
    factory_builder: Callable[[], Callable[[Any], Callable[[Any, dict[str, Any]], Any]]] | None = None,
    writer: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> Path:
    """The row process: time three episodes and write the record BEFORE any env is closed.

    ``env_builder``, ``factory_builder`` and ``writer`` exist for tests; the defaults build the real CityFlow env of
    each draw, the row's own evaluation factory and :func:`write_once`.
    """
    if device not in row.devices:
        raise ValueError(f"{row.row_id} is timed on {row.devices}, not on {device!r}")
    record_path = Path(out_dir) / f"{row.row_id}_{device}.json"
    if record_path.exists():
        raise FileExistsError(f"{record_path} exists: a record is written once")
    regime = configure_regime(device)
    load_before = _load_average()

    prompt: dict[str, Any] | None = None
    factory_name = "injected (test)"
    if env_builder is None or factory_builder is None:
        settings, configs = _evaluation_inputs(row, corpus_root, draws_root)
        built, prompt, factory_name = _row_factory(
            row, device, output_root=output_root, corpus_root=corpus_root, data_dir=data_dir,
            first_config=configs[TIMING_DRAWS[0]],
        )
        build_env = env_builder or (lambda draw: _make_env(row, settings, configs[int(draw)]))
        factory = built if factory_builder is None else factory_builder()
    else:
        build_env = env_builder
        factory = factory_builder()
    sync = _cuda_sync if device == "cuda" else None

    envs: list[Any] = []
    try:
        per_episode: list[list[int]] = []
        for draw in TIMING_DRAWS:
            env = build_env(int(draw))
            envs.append(env)
            choose = factory(env)
            per_episode.append(time_episode(env, choose, engine_seed=ENGINE_SEED, sync=sync))
        record = build_record(
            row, device, per_episode, draws=TIMING_DRAWS, prompt=prompt, factory=factory_name, regime=regime,
            machine=_machine_block(), load_before=load_before, load_after=_load_average(), git=_git_provenance(),
        )
        (write_once if writer is None else writer)(record_path, record)
        print(f"{row.row_id} {device}: record written, {record['n_timed']} timed decisions", flush=True)
    finally:
        # Amendment A3.1: the engines are destroyed only now, after the record exists.
        for env in envs:
            env.close()
    return record_path


def run_canary(phase: str, out_dir: Path) -> Path:
    """The canary process: time ``canary_seconds()``, check the engine's answers under ``==``, write the record."""
    from offline.transfer_calibration import CANARY_MAX_SECONDS as reference_ceiling
    from offline.transfer_calibration import canary_seconds, check_canary

    if float(reference_ceiling) != CANARY_MAX_SECONDS:
        raise ValueError(f"the canary ceiling drifted: transfer_calibration says {reference_ceiling}")
    record_path = Path(out_dir) / f"canary_{phase}.json"
    if record_path.exists():
        raise FileExistsError(f"{record_path} exists: a record is written once")
    seconds, facts = canary_seconds()
    try:
        check_canary(facts)
        reproduced = True
    except ValueError:
        reproduced = False  # the message carries the observed values, so it is not printed
    del facts
    record = build_canary_record(phase, seconds, reproduced=reproduced, git=_git_provenance())
    write_once(record_path, record)
    print(f"canary {phase}: {seconds:.2f} s, {record['verdict']}, reproduced={reproduced}", flush=True)
    return record_path


# ----------------------------------------------------------------------
# Supervision (Amendment A3.1)
# ----------------------------------------------------------------------


def _record_is_valid(path: Path) -> bool:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("format_version") in _RECORD_FORMATS


def _kill_group(process: subprocess.Popen[bytes], grace_s: float) -> None:
    for sent in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sent)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=grace_s)
            break
        except subprocess.TimeoutExpired:
            continue
    process.wait()


def supervise(
    label: str,
    command: Sequence[str],
    *,
    record_path: Path,
    timeout_s: float,
    max_attempts: int = MAX_ATTEMPTS,
    env: Mapping[str, str] | None = None,
    log_dir: Path | None = None,
    kill_grace_s: float = 5.0,
) -> dict[str, Any]:
    """Run *command* under *timeout_s*, at most *max_attempts* times (Amendment A3.1).

    A timeout kills the process group and is recorded as ``hung``. A valid record at *record_path* ends the
    supervision (status ``ok``, or ``kept`` when the process hung or failed AFTER writing it). A process that exits
    without a record fails at once; a process that hangs without one is re-run, and after the last attempt the
    outcome is ``failed`` with a reason naming *label*.
    """
    record_path = Path(record_path)
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, int(max_attempts) + 1):
        log_handle = None
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_handle = open(Path(log_dir) / f"{label}.attempt{attempt}.log", "ab")
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                list(command),
                env=None if env is None else dict(env),
                stdin=subprocess.DEVNULL,
                stdout=log_handle if log_handle is not None else subprocess.DEVNULL,
                stderr=subprocess.STDOUT if log_handle is not None else subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                returncode: int | None = process.wait(timeout=float(timeout_s))
                hung = False
            except subprocess.TimeoutExpired:
                _kill_group(process, kill_grace_s)
                returncode = None
                hung = True
        finally:
            if log_handle is not None:
                log_handle.close()
        seconds = time.monotonic() - started
        written = _record_is_valid(record_path)
        status = "hung" if hung else ("ok" if returncode == 0 else "failed")
        attempts.append(
            {"attempt": attempt, "status": status, "returncode": returncode, "seconds": seconds, "record_written": written}
        )
        print(f"{label}: attempt {attempt} {status}" + (" (record kept)" if written and status != "ok" else ""),
              flush=True)
        if written:
            return {"label": label, "status": "ok" if status == "ok" else "kept", "attempts": attempts,
                    "record": record_path.name}
        if not hung:
            return {"label": label, "status": "failed", "attempts": attempts,
                    "reason": f"{label}: exited with code {returncode} without writing {record_path.name}"}
    return {"label": label, "status": "failed", "attempts": attempts,
            "reason": f"{label}: hung on all {int(max_attempts)} attempts without writing {record_path.name}"}


def run_jobs(
    jobs: Sequence[tuple[str, Sequence[str], Path, float]],
    *,
    max_attempts: int = MAX_ATTEMPTS,
    env: Mapping[str, str] | None = None,
    log_dir: Path | None = None,
    kill_grace_s: float = 5.0,
) -> dict[str, Any]:
    """Supervise *jobs* (label, command, record path, timeout) one after another; the first failure stops the
    sequence and the result names it."""
    outcomes: list[dict[str, Any]] = []
    for label, command, record_path, timeout_s in jobs:
        outcome = supervise(label, command, record_path=record_path, timeout_s=timeout_s, max_attempts=max_attempts,
                            env=env, log_dir=log_dir, kill_grace_s=kill_grace_s)
        outcomes.append(outcome)
        if outcome["status"] == "failed":
            return {"status": "failed", "failed": label, "reason": outcome["reason"], "outcomes": outcomes}
    return {"status": "ok", "failed": None, "reason": None, "outcomes": outcomes}


# ----------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------


def build_run_record(
    stamp: str,
    rows: Sequence[LatencyRow],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    status: str,
    reason: str | None,
    git: Mapping[str, Any],
    timeouts: Mapping[str, float],
) -> dict[str, Any]:
    """The ``p8.2-latency-run/1.0`` record: the declared row order, every supervised outcome with its attempts, the
    timeouts, the terminal status and, on ``FAILED``, the reason naming what failed."""
    if status not in ("COMPLETE", "FAILED"):
        raise ValueError(f"a run is COMPLETE or FAILED, not {status!r}")
    if (status == "FAILED") != (reason is not None):
        raise ValueError("a FAILED run names its reason, and only a FAILED run has one")
    record = {
        "format_version": RUN_FORMAT_VERSION,
        "stamp": stamp,
        "status": status,
        "reason": reason,
        "row_order": [f"{row.row_id}_{device}" for row in rows for device in row.devices],
        "outcomes": [dict(outcome) for outcome in outcomes],
        "timeouts_s": {str(key): float(value) for key, value in timeouts.items()},
        "max_attempts": MAX_ATTEMPTS,
        "draws": list(TIMING_DRAWS),
        "engine_seed": ENGINE_SEED,
        "warmup": WARMUP,
        "git": dict(git),
        "written_utc": _utc_now(),
    }
    named = find_outcome_keys(record)
    if named:
        raise ValueError(f"the run record would carry keys that name an episode quantity {named}")
    return record


def write_manifest(latency_dir: Path, manifest_path: Path) -> list[str]:
    """Write ``sha256  relative/path`` for every file under *latency_dir* (sorted), once, then re-verify every line
    against the files; returns the lines. An existing manifest refuses."""
    latency_dir = Path(latency_dir)
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists: the manifest is written once")
    files = sorted(path for path in latency_dir.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"{latency_dir} holds no file to list")
    lines = [f"{_sha256_file(path)}  {path.relative_to(latency_dir).as_posix()}" for path in files]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.parent / f".{manifest_path.name}.{os.getpid()}.tmp"
    with open(temporary, "x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, manifest_path)
    temporary.unlink()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if _sha256_file(latency_dir / relative) != digest:
            raise ValueError(f"{manifest_path}: {relative} no longer matches its line")
    return lines


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_all(
    *,
    stamp: str,
    rows: Sequence[LatencyRow],
    latency_root: Path,
    manifest_path: Path,
    command_for: Callable[[LatencyRow, str, Path], Sequence[str]],
    canary_command_for: Callable[[str, Path], Sequence[str]],
    timeouts: Mapping[str, float],
    canary_timeout_s: float,
    git: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    kill_grace_s: float = 5.0,
) -> int:
    """The timing run: the opening canary, every (row, device) process in order, the closing canary, ``run.json``,
    ``COMPLETE`` or ``FAILED``, and on ``COMPLETE`` the manifest. Refuses a stamp that exists or a manifest that
    exists before creating anything. Returns 0 on ``COMPLETE``, 1 otherwise."""
    latency_root = Path(latency_root)
    manifest_path = Path(manifest_path)
    run_dir = latency_root / stamp
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists: a completed timing run is final, and this run would not be in it")
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists: one stamp, one run")
    if not rows:
        raise ValueError("a run needs at least one row")
    missing = sorted({row.scenario for row in rows} - set(timeouts))
    if missing:
        raise ValueError(f"no timeout is set for {missing}")
    run_dir.mkdir(parents=True)
    logs = run_dir / "logs"
    outcomes: list[dict[str, Any]] = []

    def finish(status: str, reason: str | None) -> int:
        write_once(run_dir / "run.json", build_run_record(stamp, rows, outcomes, status=status, reason=reason, git=git,
                                                          timeouts=timeouts))
        (run_dir / status).write_text(f"{status}\n" if reason is None else f"{status}: {reason}\n", encoding="utf-8")
        print(f"run {stamp}: {status}" + ("" if reason is None else f" -- {reason}"), flush=True)
        if status == "COMPLETE":
            lines = write_manifest(latency_root, manifest_path)
            print(f"manifest {manifest_path}: {len(lines)} files, re-verified", flush=True)
        return 0 if status == "COMPLETE" else 1

    def canary(phase: str) -> str | None:
        label = f"canary_{phase}"
        record_path = run_dir / f"{label}.json"
        outcome = supervise(label, canary_command_for(phase, run_dir), record_path=record_path,
                            timeout_s=canary_timeout_s, max_attempts=max_attempts, env=env, log_dir=logs,
                            kill_grace_s=kill_grace_s)
        outcomes.append(outcome)
        if outcome["status"] == "failed":
            return outcome["reason"]
        record = _read_json(record_path)
        if record.get("reproduced") is not True:
            return f"{label}: the engine did not reproduce draw 0"
        if record.get("verdict") != "at speed":
            return f"{label}: throttled ({record.get('seconds')} s above {CANARY_MAX_SECONDS} s)"
        return None

    refused = canary("open")
    if refused is not None:
        return finish("FAILED", refused)
    for row in rows:
        for device in row.devices:
            label = f"{row.row_id}_{device}"
            outcome = supervise(label, command_for(row, device, run_dir), record_path=run_dir / f"{label}.json",
                                timeout_s=float(timeouts[row.scenario]), max_attempts=max_attempts, env=env,
                                log_dir=logs, kill_grace_s=kill_grace_s)
            outcomes.append(outcome)
            if outcome["status"] == "failed":
                return finish("FAILED", outcome["reason"])
    refused = canary("close")
    if refused is not None:
        return finish("FAILED", refused)
    return finish("COMPLETE", None)


# ----------------------------------------------------------------------
# The G1 pre-flight's arithmetic
# ----------------------------------------------------------------------


def derive_timeouts(measured_seconds: Mapping[tuple[str, str], float], *, factor: float = 3.0,
                    floor_s: float = 120.0) -> dict[str, float]:
    """Per-scenario timeout from the pre-flight: ``max(floor_s, factor x`` the slowest measured process of that
    scenario``)``, rounded up to a whole second."""
    slowest: dict[str, float] = {}
    for (row_id, _device), seconds in measured_seconds.items():
        scenario = row_id.split(".", 1)[0]
        slowest[scenario] = max(slowest.get(scenario, 0.0), float(seconds))
    return {scenario: float(max(floor_s, math.ceil(factor * value))) for scenario, value in slowest.items()}


def expected_duration(measured_seconds: Mapping[tuple[str, str], float], canary_seconds: float,
                      rows: Sequence[LatencyRow] | None = None) -> dict[str, Any]:
    """An UPPER-BOUND estimate of the run: every (row, device) process costed at its scenario's measured process
    on that device (the pre-flight measures the most expensive row of each scenario), plus two canaries."""
    per_cell: dict[tuple[str, str], float] = {}
    for (row_id, device), seconds in measured_seconds.items():
        key = (row_id.split(".", 1)[0], device)
        per_cell[key] = max(per_cell.get(key, 0.0), float(seconds))
    total = 0.0
    processes = 0
    for row in ROWS if rows is None else rows:
        for device in row.devices:
            total += per_cell[(row.scenario, device)]
            processes += 1
    total += 2 * float(canary_seconds)
    return {
        "seconds": total,
        "processes": processes,
        "canaries": 2,
        "per_scenario_device_s": {f"{scenario}/{device}": value for (scenario, device), value in sorted(per_cell.items())},
        "basis": (
            "an upper bound: every (row, device) process costed at the slowest measured process of its scenario on "
            "that device, and the pre-flight measures the most expensive row of each scenario (the K = 20 DT on "
            "hz1x1, the four-head spatial DT on grid4x4); plus the two canary processes"
        ),
    }


def run_preflight(
    *,
    stamp: str,
    rows: Sequence[LatencyRow],
    runs_root: Path,
    command_for: Callable[[LatencyRow, str, Path], Sequence[str]],
    canary_command_for: Callable[[str, Path], Sequence[str]],
    provisional_timeout_s: float,
    git: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
) -> int:
    """G1's pre-flight: the opening canary, *rows* on their devices, the closing canary, then the per-scenario
    timeouts and the run's expected duration derived from the measured process wall times. Its records stay in
    ``<runs_root>/preflight_<stamp>/`` and no table ever reads them."""
    run_dir = Path(runs_root) / f"preflight_{stamp}"
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists")
    run_dir.mkdir(parents=True)
    logs = run_dir / "logs"
    outcomes: list[dict[str, Any]] = []
    measured: dict[tuple[str, str], float] = {}
    canary_wall: list[float] = []
    status, reason = "COMPLETE", None

    def wall(outcome: Mapping[str, Any]) -> float:
        return float([a for a in outcome["attempts"] if a["record_written"]][-1]["seconds"])

    for phase in ("open", None, "close"):
        if phase is None:
            for row in rows:
                for device in row.devices:
                    label = f"{row.row_id}_{device}"
                    outcome = supervise(label, command_for(row, device, run_dir), record_path=run_dir / f"{label}.json",
                                        timeout_s=provisional_timeout_s, env=env, log_dir=logs)
                    outcomes.append(outcome)
                    if outcome["status"] == "failed":
                        status, reason = "FAILED", outcome["reason"]
                        break
                    measured[(row.row_id, device)] = wall(outcome)
                if status == "FAILED":
                    break
        else:
            label = f"canary_{phase}"
            outcome = supervise(label, canary_command_for(phase, run_dir), record_path=run_dir / f"{label}.json",
                                timeout_s=provisional_timeout_s, env=env, log_dir=logs)
            outcomes.append(outcome)
            if outcome["status"] == "failed":
                status, reason = "FAILED", outcome["reason"]
            else:
                record = _read_json(run_dir / f"{label}.json")
                canary_wall.append(wall(outcome))
                if record.get("reproduced") is not True or record.get("verdict") != "at speed":
                    status, reason = "FAILED", f"{label}: {record.get('verdict')}, reproduced={record.get('reproduced')}"
        if status == "FAILED":
            break

    payload: dict[str, Any] = {
        "format_version": PREFLIGHT_FORMAT_VERSION,
        "stamp": stamp,
        "status": status,
        "reason": reason,
        "rows": [row.row_id for row in rows],
        "outcomes": outcomes,
        "measured_process_seconds": [
            {"row": row_id, "device": device, "seconds": seconds} for (row_id, device), seconds in measured.items()
        ],
        "provisional_timeout_s": float(provisional_timeout_s),
        "git": dict(git),
        "fence": "pre-flight records: never read by offline.compute_table, never part of the timing run's manifest",
        "written_utc": _utc_now(),
    }
    if status == "COMPLETE":
        payload["timeouts_s"] = derive_timeouts(measured)
        payload["canary_timeout_s"] = canary_timeout_from(canary_wall)
        payload["expected_duration"] = expected_duration(measured, max(canary_wall))
    write_once(run_dir / "preflight.json", payload)
    print(f"preflight {stamp}: {status}" + ("" if reason is None else f" -- {reason}"), flush=True)
    if status == "COMPLETE":
        print(f"  timeouts {payload['timeouts_s']}; expected run duration <= "
              f"{payload['expected_duration']['seconds'] / 60:.1f} min", flush=True)
    return 0 if status == "COMPLETE" else 1


def canary_timeout_from(canary_process_seconds: Sequence[float], *, factor: float = 3.0,
                        floor_s: float = 120.0) -> float:
    """The canary processes' timeout: ``max(floor_s, ceil(factor x`` the slower canary process``))``."""
    seconds = [float(value) for value in canary_process_seconds]
    if not seconds:
        raise ValueError("no canary process was measured, so no canary timeout can be derived")
    return float(max(floor_s, math.ceil(factor * max(seconds))))


def timeouts_from_preflight(record_path: Path) -> dict[str, float]:
    """The timing run's timeouts, read from a COMPLETE pre-flight record: ``hz1x1``, ``grid4x4`` and ``canary``
    seconds. Refuses a record that is not a COMPLETE ``p8.2-preflight/1.0`` or that lacks any of the three."""
    record = _read_json(Path(record_path))
    if record.get("format_version") != PREFLIGHT_FORMAT_VERSION or record.get("status") != "COMPLETE":
        raise ValueError(f"{record_path}: not a COMPLETE {PREFLIGHT_FORMAT_VERSION} record; it yields no timeout")
    timeouts = record.get("timeouts_s") or {}
    missing = [name for name in ("hz1x1", "grid4x4") if name not in timeouts]
    if "canary_timeout_s" not in record:
        missing.append("canary")
    if missing:
        raise ValueError(f"{record_path}: the pre-flight record holds no timeout for {missing}")
    return {"hz1x1": float(timeouts["hz1x1"]), "grid4x4": float(timeouts["grid4x4"]),
            "canary": float(record["canary_timeout_s"])}


# ----------------------------------------------------------------------
# The CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The CLI: ``run-row``, ``canary``, ``timeouts``, ``run-all``, ``preflight``."""
    parser = argparse.ArgumentParser(prog="python -m offline.compute_latency", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def roots(command: argparse.ArgumentParser) -> None:
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--corpus-root", type=Path, required=True)
        command.add_argument("--draws-root", type=Path, required=True)
        command.add_argument("--data-dir", type=Path, required=True)

    row = sub.add_parser("run-row", help="time one (row, device) and write its record")
    row.add_argument("--row", required=True)
    row.add_argument("--device", required=True, choices=("cpu", "cuda"))
    row.add_argument("--out-dir", type=Path, required=True)
    roots(row)

    sub.add_parser("power-check", help="exit 0 on mains + Windows power mode Best Performance, 2 otherwise (B1.2)")

    timeouts = sub.add_parser("timeouts", help="print the hz1x1, grid4x4 and canary timeouts of a pre-flight record")
    timeouts.add_argument("--preflight-record", type=Path, required=True)

    canary = sub.add_parser("canary", help="the machine-health canary, recorded without its values")
    canary.add_argument("--phase", required=True, choices=("open", "close"))
    canary.add_argument("--out-dir", type=Path, required=True)

    for name in ("run-all", "preflight"):
        command = sub.add_parser(name)
        command.add_argument("--stamp", required=True)
        command.add_argument("--python", required=True, help="the interpreter every child process runs")
        command.add_argument("--work-tree", type=Path, required=True, help="PYTHONPATH of every child process")
        roots(command)
        if name == "run-all":
            command.add_argument("--timeout-hz1x1", type=float, required=True)
            command.add_argument("--timeout-grid4x4", type=float, required=True)
            command.add_argument("--canary-timeout", type=float, required=True)
        else:
            command.add_argument("--rows", default="hz1x1.dt_k20,grid4x4.dt_nomix_h4")
            command.add_argument("--provisional-timeout", type=float, default=1800.0)
    return parser


def _child_env(work_tree: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key != "CUBLAS_WORKSPACE_CONFIG"}
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": str(work_tree)})
    return env


def _commands(args: argparse.Namespace) -> tuple[Callable[..., list[str]], Callable[..., list[str]]]:
    base = [str(args.python), "-P", "-m", "offline.compute_latency"]
    root_args = ["--output-root", str(args.output_root), "--corpus-root", str(args.corpus_root),
                 "--draws-root", str(args.draws_root), "--data-dir", str(args.data_dir)]

    def command_for(row: LatencyRow, device: str, out_dir: Path) -> list[str]:
        return [*base, "run-row", "--row", row.row_id, "--device", device, "--out-dir", str(out_dir), *root_args]

    def canary_command_for(phase: str, out_dir: Path) -> list[str]:
        return [*base, "canary", "--phase", phase, "--out-dir", str(out_dir)]

    return command_for, canary_command_for


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns the process exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "run-row":
        run_row(row_by_id(args.row), args.device, out_dir=args.out_dir, output_root=args.output_root,
                corpus_root=args.corpus_root, draws_root=args.draws_root, data_dir=args.data_dir)
        return 0
    if args.command == "canary":
        configure_regime("cpu")
        run_canary(args.phase, args.out_dir)
        return 0
    if args.command == "power-check":
        raise NotImplementedError("Amendment B, B1.2: the power-check subcommand")
    if args.command == "timeouts":
        values = timeouts_from_preflight(args.preflight_record)
        print(f"{values['hz1x1']:g} {values['grid4x4']:g} {values['canary']:g}")
        return 0
    command_for, canary_command_for = _commands(args)
    env = _child_env(args.work_tree)
    if args.command == "run-all":
        return run_all(
            stamp=args.stamp, rows=ROWS, latency_root=args.output_root / "p8_2" / "latency",
            manifest_path=args.output_root / "SHA256SUMS_p8_2.txt", command_for=command_for,
            canary_command_for=canary_command_for,
            timeouts={"hz1x1": args.timeout_hz1x1, "grid4x4": args.timeout_grid4x4},
            canary_timeout_s=args.canary_timeout, git=_git_provenance(), env=env,
        )
    rows = [row_by_id(name.strip()) for name in str(args.rows).split(",") if name.strip()]
    return run_preflight(
        stamp=args.stamp, rows=rows, runs_root=args.output_root / "p8_2_runs", command_for=command_for,
        canary_command_for=canary_command_for, provisional_timeout_s=args.provisional_timeout,
        git=_git_provenance(), env=env,
    )


if __name__ == "__main__":
    sys.exit(main())
