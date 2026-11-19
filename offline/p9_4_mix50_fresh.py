"""P9.4 (``docs/plans/p9.4_mix50_fresh.md``, tags ``p9.4-plan`` and ``p9.4-plan-a1``): the transfer test's second DT
retrained on a corrected 50 % mixture.  EXPLORATORY, ADDED AFTER THE RESULTS (plan section 1): not a registered test,
a deviation from A25(c) as A27 amends it, with its own row in the supplement's deviation table.

FORMATS (every record carries its version)
------------------------------------------
* ``p9.4-declaration/1.0`` -- ``output/p9_4/declaration.json``, copied to ``docs/data/p9_4_declaration.json`` and
  committed before the first gradient step (plan 2.4): the 200 streams, checks (i)-(v) (check (v) as Amendment A1
  states it), the three changes of plan 2.3, the normalisation digest, ``target_rtg``, ``rtg_scale``, the two prompt
  targets, the recipe, the seeds, the regime, both evaluation designs and the guard subsets.
* ``p9.4-run/1.0`` -- one write-once record per training run, ``output/p9_4_runs/training/<group>/<tier>_dt_seed<s>.json``
  (``<group>`` is ``control`` for the registered mix50 retrained as plan section 3's control, ``hz1x1`` for
  ``mix50_fresh``): the seconds (an UNCONTROLLED clock), the gradient steps, the checkpoint's canonical and file
  digests, the loss curve's window means, the regime and the library versions.
* ``p9.4-control/1.0`` -- ``output/p9_4_runs/control.json``, written once: each control seed against P4.7's checkpoint,
  ``torch.equal`` per parameter (``context_sweep.compare_models``), and the verdict 5 of 5 or not.

ALIGNMENT CONVENTION
--------------------
No record written here stores a trajectory.  The streams are contract C6 v1.1 episodes read through the registered
loaders; a stream's ``total_return`` is ``sum(local_reward)`` (``offline_baselines.stream_returns``), and its
``episode_sha256`` is the logger's TRAJECTORY digest (``offline/trajectory_logger.py::_episode_digest``: the action
arrays as little-endian int64 in env order, then ``global_reward`` as little-endian float32), never a file hash
(Amendment A1).  A file's sha256 is recorded beside it as ``file_sha256``.

WHAT IS REUSED, AND THE ONE THING THAT CHANGES
---------------------------------------------
The selection is ``method_tier_grid.mixture_training_streams`` with mix50's own spec (expert fraction 0.50,
``default_rng(20260813 + 50)``, both pools sorted by stream key), mix50's expert pool (``mappo1000``'s 200 training
streams) unchanged and its random pool replaced by ``cf_hz1x1__random_fresh``'s 200 streams (plan 2.1).  The tier,
``mix50_fresh``, is declared HERE, beside the registered tables, so nothing that iterates over them sees it.  The state
normalisation and the return scale are fitted on the new data by the registered rules (plan 2.2); the prompt targets
are unchanged and are checked to be.

THE BARRIER
-----------
``output/`` is gitignored and has no backup.  This module writes ONLY under ``output/p9_4/`` and ``output/p9_4_runs/``:
:func:`assert_target` resolves a path before it is compared, and refuses anything outside those two.  Nothing is ever
deleted, moved or overwritten there: records are created exclusively, and a declaration is re-written only as the
identical bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "LABEL",
    "RUN_DIR",
    "RUNS_DIR",
    "SUBJECT",
    "TIER",
    "Roots",
    "assert_checks",
    "assert_target",
    "build_declaration",
    "check_i",
    "check_ii",
    "check_iii",
    "check_iv",
    "check_v",
    "declare",
    "draw_of",
    "file_sha256",
    "select_streams",
    "split_halves",
    "tier_spec",
    "trajectory_digest",
]

DECLARATION_FORMAT_VERSION = "p9.4-declaration/1.0"

#: The plan, its tags and the commits they name (plan section 8; Amendment A1).
PLAN_PATH = "docs/plans/p9.4_mix50_fresh.md"
PLAN_TAGS: Mapping[str, str] = {
    "p9.4-plan": "fdf0e2c03b5fe9c8ad062a26a2dff43669a1dc1d",
    "p9.4-plan-a1": "3bfb00739aa35694270b49410b0ec90c273c11a5",
}
#: Plan section 1's label, verbatim, carried by every record this task writes.
LABEL = "exploratory, added after the results"

#: The two directories this task creates under ``output/`` (plan section 7).
RUN_DIR = "p9_4"
RUNS_DIR = "p9_4_runs"

#: The new tier, which is also the SUMO subject; the registered tier whose rule and recipe it reuses.
TIER = "mix50_fresh"
SUBJECT = TIER
REGISTERED_TIER = "mix50"
EXPERT_COMPONENT = "mappo1000"
OLD_RANDOM_COMPONENT = "random"
NEW_RANDOM_COMPONENT = "random_fresh"
OLD_RANDOM_DIR = "cf_hz1x1__random"
NEW_RANDOM_DIR = "cf_hz1x1__random_fresh"

#: Plan 2.2: the prompt targets stay unchanged.  The source target is the training set's highest return, from the
#: unchanged expert half; the calibrated one is Rule B with S = mean, k = 100 over P7.2b's committed statistics.
SOURCE_TARGET = -5959.0
CALIBRATED_TARGET = -7431.0185327454665
#: mix50's fitted quantities, for plan 2.3's changes (``docs/data/p4_7_declaration.json``, ``tiers.mix50``).
MIX50_RTG_SCALE = 40223.0
MIX50_SPLIT_STREAMS = 600

#: Plan 2.1's sizes: 200 streams, half of them expert.
STREAM_COUNT = 200
HALF = 100

#: Plan section 3's recipe, which :func:`recipe_record` reads back from the code that trains and refuses to differ.
PLAN_RECIPE: Mapping[str, Any] = {
    "context_length": 20, "n_layer": 3, "n_head": 1, "d_model": 128, "dropout": 0.1, "batch_size": 64,
    "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4, "warmup_steps": 1000, "grad_clip": 0.25,
    "gradient_steps": 40_000,
}
SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)

#: Plan section 4: the held-out pool, the engine seed, the two SUMO prompt arms and the guards' draws.
HELD_OUT_DRAWS: tuple[int, ...] = tuple(range(1000, 1100))
ENGINE_SEED = 1000
SUMO_ARMS: tuple[str, ...] = ("b_mean_k100", "naive")
GUARD_DRAWS: tuple[int, ...] = tuple(range(1000, 1010))
G2_ARMS: tuple[str, ...] = ("fixedtime", "maxpressure", "fixedtime_kcf", "maxpressure_1to1")

#: Plan section 7: at most six worker processes.
MAX_WORKERS = 6

#: The committed artifacts this task reads, by name under ``docs/data/``.
P4_7_DECLARATION = "p4_7_declaration.json"
P7_3A_ARTIFACT = "p7_3a_zero_shot.json"
P9_2_ANCHORS = "p9_2_sumo_anchors.json"
P7_3B_ARTIFACT = "p7_3b_anchor.json"
P9_1_IQL_TARGETS = "p9_1_iql_targets.json"
P4_7_GRID = "p4_7_grid.json"


# ----------------------------------------------------------------------
# Roots, the barrier and the writes
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Roots:
    """``repo_root``: the checkout whose ``docs/`` is read (the run tree); the other three are the MAIN tree's."""

    repo_root: Path
    output_root: Path
    corpus_root: Path
    draws_root: Path

    @property
    def run_root(self) -> Path:
        return Path(self.output_root) / RUN_DIR

    @property
    def runs_root(self) -> Path:
        return Path(self.output_root) / RUNS_DIR

    @property
    def data_dir(self) -> Path:
        return Path(self.repo_root) / "docs" / "data"


def assert_target(path: str | Path, roots: Roots) -> Path:
    """Refuse any write that does not resolve under ``output/p9_4/`` or ``output/p9_4_runs/``; return the resolved path.

    Resolution comes first, so ``..``, a relative path or a symlink is judged by where it LANDS.
    """
    resolved = Path(path).resolve()
    allowed = (Path(roots.run_root).resolve(), Path(roots.runs_root).resolve())
    if not any(resolved == root or root in resolved.parents for root in allowed):
        raise PermissionError(
            f"{path} resolves to {resolved}, outside output/{RUN_DIR}/ and output/{RUNS_DIR}/; P9.4 writes nowhere else "
            "(plan section 7)"
        )
    return resolved


def canonical_text(payload: Mapping[str, Any]) -> str:
    """The one serialisation every record here is written in: sorted keys, indent 2, no NaN, one final newline."""
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_once(path: str | Path, payload: Mapping[str, Any], roots: Roots) -> Path:
    """Create *path* exclusively and atomically: the bytes go to a temporary file in the same directory, which is then
    hard-linked to *path* -- ``os.link`` fails if *path* exists, so nothing is ever overwritten -- and unlinked.  A crash
    leaves at most the temporary file, never a half-written record under the real name."""
    target = assert_target(path, roots)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    assert_target(temporary, roots)
    with open(temporary, "x", encoding="utf-8") as handle:
        handle.write(canonical_text(payload))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, target)
    finally:
        temporary.unlink()
    return target


def write_declaration(path: str | Path, payload: Mapping[str, Any], roots: Roots) -> Path:
    """A declaration is written once; declaring again is allowed only if it reproduces the file byte for byte."""
    target = assert_target(path, roots)
    if target.exists():
        if target.read_text(encoding="utf-8") != canonical_text(payload):
            raise FileExistsError(f"{target} exists with other content; a declaration is not rewritten")
        return target
    return write_once(target, payload, roots)


def file_sha256(path: str | Path) -> str:
    """sha256 over the file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def trajectory_digest(path: str | Path) -> str:
    """The episode's TRAJECTORY digest recomputed from its file by the logger's OWN function (Amendment A1):
    ``offline/trajectory_logger.py::TrajectoryLogger._episode_digest``, which reads ``len(self._ix_ids)`` and nothing
    else of the logger -- so it is called with a stand-in carrying the file's ``ix_ids``, never re-implemented."""
    from types import SimpleNamespace

    import numpy as np

    from offline.trajectory_logger import TrajectoryLogger

    with np.load(path) as arrays:
        stand_in = SimpleNamespace(_ix_ids=[str(value) for value in arrays["ix_ids"].tolist()])
        return str(TrajectoryLogger._episode_digest(stand_in, arrays))  # type: ignore[arg-type]


def draw_of(episode_file: str) -> int:
    """The flow draw an episode file name carries (``ep<index>_seed<engine>_draw<draw>.npz``)."""
    stem = Path(str(episode_file)).name
    if "_draw" not in stem or not stem.endswith(".npz"):
        raise ValueError(f"{episode_file!r} is not an episode file name of this corpus")
    return int(stem.rsplit("_draw", 1)[1][: -len(".npz")])


# ----------------------------------------------------------------------
# The tier, beside the registered ones; the selection (plan 2.1)
# ----------------------------------------------------------------------


def tier_spec(*, rtg_scale: float) -> Any:
    """``mix50_fresh``: mix50's spec with its random component (and directory) replaced by ``random_fresh``'s, the
    target unchanged and the return scale the declared one.  Declared here, never in ``method_tier_grid``'s tables."""
    from dataclasses import replace

    import offline.method_tier_grid as grid

    mix50 = grid.TIERS[REGISTERED_TIER]
    if OLD_RANDOM_DIR not in mix50.dirs or mix50.components != (EXPERT_COMPONENT, OLD_RANDOM_COMPONENT):
        raise ValueError(f"mix50's registered spec is not the one plan 2.1 replaces a component of: {mix50}")
    return replace(
        mix50,
        tier=TIER,
        dirs=tuple(NEW_RANDOM_DIR if name == OLD_RANDOM_DIR else name for name in mix50.dirs),
        rtg_scale=float(rtg_scale),
        components=(EXPERT_COMPONENT, NEW_RANDOM_COMPONENT),
    )


def select_streams(expert_pool: Sequence[Any], random_pool: Sequence[Any]) -> tuple[Any, ...]:
    """Plan 2.1: ``method_tier_grid.mixture_training_streams`` with mix50's REGISTERED spec -- its expert fraction, its
    RNG ``default_rng(20260813 + 50)``, its count -- over *expert_pool* and *random_pool*.  With the registered random
    pool this is mix50's own selection; with ``random_fresh``'s it is plan 2.1's."""
    import offline.method_tier_grid as grid

    return grid.mixture_training_streams(grid.TIERS[REGISTERED_TIER], expert_pool, random_pool)


def split_halves(selected: Sequence[Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """``(expert, random)`` in selection order, by the stream's directory; a stream of neither is refused."""
    import offline.method_tier_grid as grid

    expert_dirs = set(grid.TIERS[EXPERT_COMPONENT].dirs)
    expert, random_half = [], []
    for stream in selected:
        name = Path(stream.dataset_dir).name
        if name in expert_dirs:
            expert.append(stream)
        elif name in (OLD_RANDOM_DIR, NEW_RANDOM_DIR):
            random_half.append(stream)
        else:
            raise ValueError(f"{stream.dataset_dir} is neither a mappo1000 nor a random directory")
    return tuple(expert), tuple(random_half)


# ----------------------------------------------------------------------
# Plan 2.1's five checks (check (v) as Amendment A1 states it) -- each a pure function, refused together
# ----------------------------------------------------------------------


def check_i(selected: Sequence[Any]) -> dict[str, Any]:
    """(i) 200 streams, 100 expert and 100 random -- and 200 DISTINCT streams."""
    expert, random_half = split_halves(selected)
    distinct = len({stream.key for stream in selected})
    passed = (len(selected) == STREAM_COUNT and distinct == STREAM_COUNT and len(expert) == HALF
              and len(random_half) == HALF)
    return {"n_streams": len(selected), "n_distinct": distinct, "n_expert": len(expert), "n_random": len(random_half),
            "passed": passed}


def check_ii(expert_digests: Sequence[str], mix50_expert_digests: Sequence[str]) -> dict[str, Any]:
    """(ii) the 100 expert streams equal mix50's 100 by ``episode_sha256`` (as multisets)."""
    equal = sorted(map(str, expert_digests)) == sorted(map(str, mix50_expert_digests))
    passed = equal and len(expert_digests) == HALF and len(mix50_expert_digests) == HALF
    return {"n": len(expert_digests), "n_mix50": len(mix50_expert_digests), "equal_by_episode_sha256": equal,
            "passed": passed}


def check_iii(random_draws: Sequence[int], mix50_random_draws: Sequence[int]) -> dict[str, Any]:
    """(iii) the 100 random streams lie on exactly the 100 flow draws of mix50's random half, one per draw."""
    mine = [int(draw) for draw in random_draws]
    theirs = [int(draw) for draw in mix50_random_draws]
    same = sorted(mine) == sorted(theirs)
    passed = same and len(mine) == HALF and len(set(mine)) == HALF and len(set(theirs)) == HALF
    return {"n": len(mine), "n_distinct": len(set(mine)), "same_draws_as_mix50": same, "passed": passed}


def check_iv(sequence_digests: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """(iv) the 100 random streams hold 100 distinct per-intersection action sequences, on every intersection."""
    counts = {str(ix): len(set(digests)) for ix, digests in sorted(sequence_digests.items())}
    sizes = {str(ix): len(digests) for ix, digests in sorted(sequence_digests.items())}
    passed = bool(counts) and all(counts[ix] == HALF and sizes[ix] == HALF for ix in counts)
    return {"per_intersection_distinct": counts, "per_intersection_streams": sizes, "passed": passed}


def check_v(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """(v), as Amendment A1 states it: each stream's ``episode_sha256`` equals the trajectory digest recomputed from its
    file, and each file's sha256 is recorded beside it."""
    equal = [row for row in rows if str(row.get("episode_sha256")) == str(row.get("trajectory_digest"))]
    recorded = [row for row in rows if isinstance(row.get("file_sha256"), str) and len(row["file_sha256"]) == 64]
    mismatched = [row.get("episode_file") for row in rows if row not in equal]
    passed = bool(rows) and len(equal) == len(rows) and len(recorded) == len(rows)
    return {"n": len(rows), "n_equal": len(equal), "n_file_sha256_recorded": len(recorded),
            "first_mismatched": mismatched[:5], "rule": "Amendment A1 (p9.4-plan-a1)", "passed": passed}


CHECK_NAMES: tuple[str, ...] = ("i", "ii", "iii", "iv", "v")


def assert_checks(checks: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse unless all five checks are present and every one passed; the refusal names each failed check."""
    if set(checks) != set(CHECK_NAMES):
        raise ValueError(f"the declaration needs all five checks {list(CHECK_NAMES)}, not {sorted(checks)}")
    failed = [name for name in CHECK_NAMES if checks[name].get("passed") is not True]
    if failed:
        raise ValueError(
            "plan 2.1's check(s) " + ", ".join(f"({name})" for name in failed) + " failed; the task stops before "
            "training and nothing is written (" + "; ".join(f"({name}) {dict(checks[name])}" for name in failed) + ")"
        )


# ----------------------------------------------------------------------
# The declaration (plan 2.2-2.4; Amendment A1): computed from the corpus, refused before any write if a check fails
# ----------------------------------------------------------------------

#: The registered regime P5.3c reproduced P4.7's mix50 checkpoints in, 5 of 5 (``docs/data/p5_3c_train.json``,
#: ``k20_reproduction``), entered through ``context_sweep.enter_registered_regime`` by the train stage.
REGIME: Mapping[str, Any] = {
    "device": "cuda",
    "cublas_workspace_config": None,
    "deterministic_algorithms": False,
    "torch_num_threads": 1,
    "omp_num_threads": "1",
    "mkl_num_threads": "1",
    "entered_by": "offline.context_sweep.enter_registered_regime (P5.3c, BRIEF_42 C2): CUDA, CUBLAS_WORKSPACE_CONFIG "
                  "unset, deterministic algorithms off, one torch thread, a clean code tree, the working directory at "
                  "the code's commit; the driver sets OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1",
    "reproduced_mix50_in": "docs/data/p5_3c_train.json, k20_reproduction: mix50 5 of 5 torch.equal",
}


def _manifest_entries(directory: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    return {str(entry["filename"]): dict(entry) for entry in payload["episodes"]}


def _episode_facts(directory: Path, episode_file: str) -> dict[str, Any]:
    """One file's sha256, its trajectory digest and each intersection's action-sequence digest
    (``nortg_decomposition.action_sequence_sha256`` on that intersection's column, as A27(d)'s count takes it)."""
    import numpy as np

    from offline.nortg_decomposition import action_sequence_sha256

    path = Path(directory) / episode_file
    with np.load(path) as arrays:
        ids = [str(value) for value in arrays["ix_ids"].tolist()]
        sequences = {
            ix: action_sequence_sha256(np.asarray(arrays[f"ix{i}_action"], dtype=np.int64).reshape(-1, 1))
            for i, ix in enumerate(ids)
        }
    return {"file_sha256": file_sha256(path), "trajectory_digest": trajectory_digest(path), "sequences": sequences}


def _mix50_declared_halves(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """mix50's 200 declared streams (``docs/data/p4_7_declaration.json``), as ``(expert, random)`` by directory."""
    import offline.method_tier_grid as grid

    streams = json.loads((Path(data_dir) / P4_7_DECLARATION).read_bytes())["tiers"][REGISTERED_TIER]["streams"]
    expert_dirs = set(grid.TIERS[EXPERT_COMPONENT].dirs)
    expert = [dict(s) for s in streams if Path(str(s["dataset_dir"])).name in expert_dirs]
    random_half = [dict(s) for s in streams if Path(str(s["dataset_dir"])).name == OLD_RANDOM_DIR]
    if len(expert) + len(random_half) != len(streams):
        raise ValueError(f"{P4_7_DECLARATION}: a mix50 stream is in neither half")
    return expert, random_half


def recipe_record() -> dict[str, Any]:
    """Plan section 3's recipe, READ from the code that trains (``dt_gate.train_dt`` and ``agent.DTAgent.DTConfig``'s
    defaults, which ``train_dt`` does not override) and refused unless it is the plan's."""
    import inspect

    import offline.dt_gate as gate
    import offline.method_tier_grid as grid
    from agent.DTAgent import DTConfig

    fields = DTConfig.__dataclass_fields__
    source = inspect.getsource(gate.train_dt)
    measured = {
        "context_length": int(grid.CONTEXT_LENGTH),
        "n_layer": int(fields["n_layer"].default),
        "n_head": int(fields["n_head"].default),
        "d_model": int(fields["d_model"].default),
        "dropout": float(fields["dropout"].default),
        "batch_size": int(gate.BATCH_SIZE),
        "optimizer": "AdamW" if "torch.optim.AdamW(" in source else "(not AdamW)",
        "learning_rate": float(gate.LEARNING_RATE),
        "weight_decay": float(gate.WEIGHT_DECAY),
        "warmup_steps": int(gate.WARMUP_STEPS),
        "grad_clip": float(gate.GRAD_CLIP),
        "gradient_steps": int(grid.DECLARED_GRADIENT_STEPS),
    }
    if measured != dict(PLAN_RECIPE):
        differing = sorted(key for key in PLAN_RECIPE if measured.get(key) != PLAN_RECIPE[key])
        raise ValueError(f"the code's recipe differs from plan section 3's on {differing}: {measured}")
    if tuple(int(seed) for seed in gate.TRAINING_SEEDS) != SEEDS:
        raise ValueError(f"dt_gate.TRAINING_SEEDS {gate.TRAINING_SEEDS} are not plan section 3's {SEEDS}")
    return {**measured, "source": "dt_gate.train_dt's constants and DTConfig's defaults, read at declaration"}


def _code_provenance() -> dict[str, Any]:
    """The module tree's commit and whether it is dirty, STRICT (a git failure raises; ``context_sweep``'s helper)."""
    from offline.context_sweep import _code_provenance as strict

    return dict(strict())


def evaluation_designs() -> dict[str, Any]:
    """Plan section 4's two designs, as data."""
    return {
        "sumo": {
            "subject": SUBJECT, "arms": list(SUMO_ARMS), "seeds": list(SEEDS), "draws": [HELD_OUT_DRAWS[0], HELD_OUT_DRAWS[-1]],
            "n_cells": len(SUMO_ARMS) * len(SEEDS) * len(HELD_OUT_DRAWS), "engine_seed": ENGINE_SEED,
            "rules": "A15 (E_sumo primary, att_env co-reported, time-to-teleport -1, A23 for collisions), A16 (observations "
                     "only through align_info), A17 (b_mean_k100: Rule B, S = mean, k = 100 from P7.2b's committed "
                     "statistics; naive: the training set's highest return; rtg_scale not recalibrated), A18(c) (one "
                     "episode per held-out draw on a fresh env at reset(seed=1000))",
            "anchors": "docs/data/p9_2_sumo_anchors.json anchors.sumo.cityflow1x1.by_draw: registered pair fixedtime (k = 4) "
                       "and maxpressure (per-link); aligned pair fixedtime_kcf and maxpressure_1to1",
            "work_dir": f"output/{RUN_DIR}/cells/",
        },
        "cityflow": {
            "arm": f"dt@{TIER}", "seeds": list(SEEDS), "draws": [HELD_OUT_DRAWS[0], HELD_OUT_DRAWS[-1]],
            "n_episodes": len(SEEDS) * len(HELD_OUT_DRAWS), "engine_seed": ENGINE_SEED, "target_rtg": SOURCE_TARGET,
            "routes": "route (i): P4.6's dt_gate.evaluate_arm (method_tier_grid's evaluation, its factory and settings); "
                      "route (ii): P8.4b's probe (att_rederivation.run_campaign); att_ours equal under == on every episode",
            "records": "P9.1's hz1x1 format: att_ours, att_engine, entered, created, never_entered (A11(b)) per episode",
            "primary": "att_engine on hz1x1 (A11(c), Rule R), att_ours beside it",
        },
    }


def guard_subsets() -> dict[str, Any]:
    """Plan 4.3's guards, as data: each is compared under == before the stage's first new cell."""
    return {
        "control": {"subject": REGISTERED_TIER, "seeds": list(SEEDS), "rule": "every parameter torch.equal to P4.7's "
                    "checkpoint, the files pinned by digest in p7_3a_zero_shot.json inputs.checkpoints", "required": 5},
        "g1": {"subject": REGISTERED_TIER, "arms": list(SUMO_ARMS), "seeds": list(SEEDS),
               "draws": [GUARD_DRAWS[0], GUARD_DRAWS[-1]], "n_cells": len(SUMO_ARMS) * len(SEEDS) * len(GUARD_DRAWS),
               "fields": ["e_sumo", "att_env"], "against": f"docs/data/{P7_3A_ARTIFACT} cells"},
        "g2": {"arms": list(G2_ARMS), "draws": [GUARD_DRAWS[0], GUARD_DRAWS[-1]],
               "n_cells": len(G2_ARMS) * len(GUARD_DRAWS),
               "fields": "every field anchors.sumo.cityflow1x1.fields lists, and action_sequence_sha256 where recorded",
               "against": f"docs/data/{P9_2_ANCHORS} anchors.sumo.cityflow1x1.by_draw"},
    }


def build_declaration(roots: Roots) -> dict[str, Any]:
    """Plan 2.1-2.4 from the corpus.  Raises -- and nothing is written -- if any check fails, if the training set's
    highest return is not the source target from the expert half (plan 2.2), if a prompt target moved, if the recipe is
    not the plan's, or if the code tree is dirty."""
    from collections import Counter

    import offline.method_tier_grid as grid
    import offline.transfer_calibration as tcal
    import offline.transfer_curve as tc
    from offline.dt_gate import CONTEXT_LENGTH
    from offline.offline_baselines import stream_returns

    code = _code_provenance()
    if code["code_dirty"]:
        raise ValueError(f"the code tree at {code['code_commit'][:12]} is dirty; a declaration names clean code")
    data = roots.data_dir
    corpus = Path(roots.corpus_root)
    calibration = tc.load_calibration(data / tc.P7_2B_CALIBRATION_NAME)
    mix50_expert, mix50_random = _mix50_declared_halves(data)

    provisional = tier_spec(rtg_scale=math.nan)
    components = grid._component_streams(provisional, corpus, context_length=CONTEXT_LENGTH)
    selected = select_streams(components[EXPERT_COMPONENT], components[NEW_RANDOM_COMPONENT])
    expert, random_half = split_halves(selected)

    names = sorted({Path(stream.dataset_dir).name for stream in selected} | {OLD_RANDOM_DIR})
    manifests = {name: _manifest_entries(corpus / name) for name in names}
    rows: list[dict[str, Any]] = []
    sequences: dict[str, list[str]] = {}
    for half, streams in (("expert", expert), ("random", random_half)):
        for stream in streams:
            directory = Path(stream.dataset_dir)
            entry = manifests[directory.name][str(stream.episode_file)]
            if int(entry["flow_draw"]) != int(stream.flow_draw):
                raise ValueError(f"{directory.name}/{stream.episode_file}: the manifest's draw is not the stream's")
            facts = _episode_facts(directory, str(stream.episode_file))
            rows.append({
                "half": half,
                "dataset_dir": str(directory),
                "episode_file": str(stream.episode_file),
                "ix_id": str(stream.ix_id),
                "flow_draw": int(stream.flow_draw),
                "engine_seed": int(entry["engine_seed"]),
                "episode_sha256": str(entry["episode_sha256"]),
                "trajectory_digest": facts["trajectory_digest"],
                "file_sha256": facts["file_sha256"],
                "total_return": float(stream.total_return),
            })
            if half == "random":
                sequences.setdefault(str(stream.ix_id), []).append(facts["sequences"][str(stream.ix_id)])

    expert_rows = [row for row in rows if row["half"] == "expert"]
    random_rows = [row for row in rows if row["half"] == "random"]
    checks = {
        "i": check_i(selected),
        "ii": {**check_ii([row["episode_sha256"] for row in expert_rows], [s["episode_sha256"] for s in mix50_expert]),
               "same_files_as_mix50": sorted((Path(r["dataset_dir"]).name, r["episode_file"]) for r in expert_rows)
               == sorted((Path(str(s["dataset_dir"])).name, str(s["episode_file"])) for s in mix50_expert)},
        "iii": check_iii([row["flow_draw"] for row in random_rows], [int(s["flow_draw"]) for s in mix50_random]),
        "iv": check_iv(sequences),
        "v": check_v(rows),
    }
    assert_checks(checks)

    target, scale = grid.recomputed_target_and_scale(selected)
    at_maximum = sorted({row["half"] for row in rows if row["total_return"] == target})
    if target != SOURCE_TARGET or at_maximum != ["expert"]:
        raise ValueError(
            f"plan 2.2: the training set's highest return is {target} (in the {at_maximum} half), not the source target "
            f"{SOURCE_TARGET} from the unchanged expert half; the task stops"
        )
    spec = tier_spec(rtg_scale=scale)
    dataset = grid.tier_dataset(spec, corpus)
    declared = grid.assert_declaration_matches_corpus(spec, selected)
    split = stream_returns(dataset)
    groups = sorted(dataset.groups)
    if len(groups) != 1:
        raise ValueError(f"{TIER}: {len(groups)} (state_dim, n_actions) groups; hz1x1 has one")
    rtg = dataset.stats.to_json_obj()["rtg"]
    pairs = [(scenario, ix, summary) for scenario, per_ix in rtg.items() for ix, summary in per_ix.items()]
    if len(pairs) != 1:
        raise ValueError(f"{TIER}: the statistics cover {len(pairs)} (scenario, intersection) pairs, not one")
    summary = pairs[0][2]
    support = (float(summary["min"]), float(summary["max"]))

    facts = tcal.SubjectFacts(
        subject=SUBJECT, best_source_return=float(target), rtg_scale=float(scale), support_range_over_the_split=support,
        n_rows=int(summary["count"]), training_set_return_min=-float(scale), checkpoints=(),
        state_dim=int(groups[0][0]), context_length=int(CONTEXT_LENGTH),
    )
    probe = [(int(record["draw_id"]), float(record["local_return"])) for record in calibration["probe"]]
    table = tcal.targets_table(calibration["statistics"], probe, {SUBJECT: facts})
    arms = tc.targets_for_subject(SUBJECT, {"targets": {SUBJECT: table[SUBJECT]}})
    registered = tc.targets_for_subject(REGISTERED_TIER, calibration)
    prompts = {
        arm: {**arms[arm], "registered_mix50_target_rtg": float(registered[arm]["target_rtg"])} for arm in SUMO_ARMS
    }
    if prompts["b_mean_k100"]["target_rtg"] != CALIBRATED_TARGET or prompts["naive"]["target_rtg"] != SOURCE_TARGET:
        raise ValueError(f"plan 2.2: the prompt targets moved: {prompts}")
    if any(prompt["target_rtg"] != prompt["registered_mix50_target_rtg"] for prompt in prompts.values()):
        raise ValueError(f"plan 2.2: a prompt target differs from mix50's registered one: {prompts}")

    old_by_draw = {int(stream["flow_draw"]): stream for stream in mix50_random}
    transitions: Counter[str] = Counter()
    engine_seeds: list[dict[str, Any]] = []
    for row in sorted(random_rows, key=lambda r: r["flow_draw"]):
        old = old_by_draw[row["flow_draw"]]
        old_seed = int(manifests[OLD_RANDOM_DIR][str(old["episode_file"])]["engine_seed"])
        transitions[f"{old_seed}->{row['engine_seed']}"] += 1
        engine_seeds.append({"flow_draw": row["flow_draw"], "old_episode_file": str(old["episode_file"]),
                             "old_engine_seed": old_seed, "new_episode_file": row["episode_file"],
                             "new_engine_seed": row["engine_seed"]})

    inputs = {name: file_sha256(data / name) for name in (
        P4_7_DECLARATION, tc.P7_2B_CALIBRATION_NAME, P7_3A_ARTIFACT, P9_2_ANCHORS, P7_3B_ARTIFACT, P9_1_IQL_TARGETS)}
    manifest_digests = {name: file_sha256(corpus / name / "manifest.json") for name in sorted(set(spec.dirs) | {OLD_RANDOM_DIR})}
    return {
        "format_version": DECLARATION_FORMAT_VERSION,
        "task": "P9.4",
        "label": LABEL,
        "registered_in": None,
        "deviation": "A25(c), as A27 amends it, allows no further run; P9.4 is a deviation with its own row in the "
                     "supplement's deviation table",
        "plan": {"path": PLAN_PATH, "tags": dict(PLAN_TAGS)},
        "tier": TIER,
        "dataset_dirs": [str(path) for path in grid.tier_dirs(spec, corpus)],
        "selection": {
            "rule": "method_tier_grid.mixture_training_streams with mix50's spec (expert fraction 0.50, "
                    "default_rng(20260813 + 50), both pools sorted by stream key): mix50's expert pool unchanged "
                    "(mappo1000's 200 training streams), its random pool replaced by cf_hz1x1__random_fresh's 200 streams",
            "expert_pool": len(components[EXPERT_COMPONENT]),
            "random_pool": len(components[NEW_RANDOM_COMPONENT]),
        },
        "streams": rows,
        "checks": checks,
        "target_rtg": float(target),
        "rtg_scale": float(scale),
        "source_target_check": {"training_return_max": float(target), "at_maximum": at_maximum, "passed": True},
        "assert_declaration_matches_corpus": declared,
        "normalisation": {
            "statistics_digest": grid.statistics_digest(dataset),
            "split_streams": len(split),
            "support_range_over_the_split": list(support),
            "n_rows": int(summary["count"]),
            "statistics_fitted_on": "the registered loader over the tier's split, the union of its components' splits "
                                    "(method_tier_grid.tier_dataset over the six directories)",
        },
        "changes_besides_the_generator": {
            "normalisation_basis": {"mix50_split_streams": MIX50_SPLIT_STREAMS, "mix50_fresh_split_streams": len(split)},
            "return_scale": {"mix50": MIX50_RTG_SCALE, "mix50_fresh": float(scale)},
            "engine_seed": {"transitions": dict(sorted(transitions.items())), "by_draw": engine_seeds,
                            "source": "the corpus manifests' engine_seed of each random-half episode, old and new"},
        },
        "prompts": prompts,
        "prompts_not_evaluated": sorted(set(arms) - set(SUMO_ARMS)),
        "facts": {"state_dim": facts.state_dim, "n_actions": int(groups[0][1]), "context_length": facts.context_length,
                  "scenario_id": str(dataset.episode_records[0].scenario_id)},
        "recipe": recipe_record(),
        "seeds": list(SEEDS),
        "regime": dict(REGIME),
        "evaluation": evaluation_designs(),
        "guards": guard_subsets(),
        "inputs": {"docs_data_sha256": inputs, "corpus_manifest_sha256": manifest_digests},
        "git": {"commit": code["code_commit"], "dirty": code["code_dirty"]},
    }


def declaration_path(roots: Roots) -> Path:
    return Path(roots.run_root) / "declaration.json"


def declare(roots: Roots) -> dict[str, Any]:
    """Build the declaration and write it once (``output/p9_4/declaration.json``); re-declaring writes nothing and is
    allowed only if it reproduces the file byte for byte."""
    payload = build_declaration(roots)
    write_declaration(declaration_path(roots), payload, roots)
    return payload


# ----------------------------------------------------------------------
# The committed declaration, pinned by digest (plan 2.4: committed before the first gradient step)
# ----------------------------------------------------------------------

#: ``docs/data/p9_4_declaration.json``'s sha256, typed in the commit that commits the declaration (step C) and moved
#: only in a commit that also moves the file.  ``None`` until then: no training session starts without it.
DECLARATION_SHA256: str | None = None
DECLARATION_NAME = "p9_4_declaration.json"


def load_declaration(roots: Roots) -> dict[str, Any]:
    """The COMMITTED declaration, digest-checked before it is parsed; refused if it is not pinned yet or has moved."""
    path = roots.data_dir / DECLARATION_NAME
    if DECLARATION_SHA256 is None:
        raise ValueError(f"the declaration is not pinned yet ({__name__}.DECLARATION_SHA256 is None): commit "
                         f"docs/data/{DECLARATION_NAME} and its digest first (plan 2.4)")
    if not path.is_file():
        raise FileNotFoundError(f"{path}: the committed declaration is not here")
    digest = file_sha256(path)
    if digest != DECLARATION_SHA256:
        raise ValueError(f"{path}: sha256 {digest} is not the pinned {DECLARATION_SHA256}; the declaration has moved")
    payload = json.loads(path.read_bytes())
    if payload.get("format_version") != DECLARATION_FORMAT_VERSION or payload.get("tier") != TIER:
        raise ValueError(f"{path} is not a {DECLARATION_FORMAT_VERSION} declaration of {TIER}")
    return payload


def assert_files_as_declared(declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Amendment A1's addition: every declared file is re-hashed, and any sha256 other than the declared one refuses
    the stage before its first gradient step."""
    differing = [row["episode_file"] for row in declaration["streams"]
                 if file_sha256(Path(row["dataset_dir"]) / row["episode_file"]) != row["file_sha256"]]
    if len(declaration["streams"]) != STREAM_COUNT or differing:
        raise ValueError(f"{len(differing)} declared file(s) no longer hash to their declared sha256 (first "
                         f"{differing[:3]}); the train stage does not start (Amendment A1)")
    return {"n_files": len(declaration["streams"]), "all_as_declared": True}


# ----------------------------------------------------------------------
# Training (plan section 3): the control first, then mix50_fresh, by P4's recipe through the registered trainer
# ----------------------------------------------------------------------

RUN_FORMAT_VERSION = "p9.4-run/1.0"
CONTROL_FORMAT_VERSION = "p9.4-control/1.0"
GRADIENT_STEPS = 40_000
#: A27(a)'s label, kept for P9.4's runs: their seconds are an uncontrolled clock, recorded and never a benchmark.
CLOCK_LABEL = "uncontrolled; no Best Performance regime, no canary; not a benchmark"
#: PLANNING ESTIMATES, never measurements of this task: a 40,000-step hz1x1 DT run (P5.3c measured 5.66 ms per step at
#: K = 20, B = 64, about 226 s; P9.1 planned 204 s) and one tier's inputs built once per session.  They decide only how
#: many runs a session plans.
PLANNED_RUN_SECONDS = 240.0
PLANNED_INPUT_SECONDS = 120.0
PLANNING_NOTE = ("planned hours are planning estimates (P5.3c's measured step rate and P9.1's per-run figure), not "
                 "measurements of this task; each run's record carries its measured seconds")


@dataclass(frozen=True)
class Run:
    """One training run: ``control`` retrains the registered mix50 (plan section 3's control), ``hz1x1`` trains
    ``mix50_fresh``."""

    group: str
    tier: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.tier}_dt_seed{int(self.seed)}"


CONTROL_GROUP = "control"
NEW_GROUP = "hz1x1"


def control_runs() -> tuple[Run, ...]:
    return tuple(Run(CONTROL_GROUP, REGISTERED_TIER, seed) for seed in SEEDS)


def new_runs() -> tuple[Run, ...]:
    return tuple(Run(NEW_GROUP, TIER, seed) for seed in SEEDS)


def training_runs() -> tuple[Run, ...]:
    """The control's five, then the new tier's five -- the only order they ever train in."""
    return (*control_runs(), *new_runs())


def checkpoint_path(roots: Roots, run: Run) -> Path:
    return Path(roots.runs_root) / "checkpoints" / run.group / f"{run.name}.pt"


def run_record_path(roots: Roots, run: Run) -> Path:
    return Path(roots.runs_root) / "training" / run.group / f"{run.name}.json"


def control_record_path(roots: Roots) -> Path:
    return Path(roots.runs_root) / "control.json"


def training_inputs(spec: Any, corpus_root: str | Path) -> Any:
    """``method_tier_grid.training_inputs``, step for step and in its order, with ONE call replaced: the selection is
    :func:`select_streams` (plan 2.1's rule, mix50's spec) instead of ``training_streams``, whose fraction table does not
    know ``mix50_fresh``.  For the registered mix50 it is ``training_inputs(TIERS["mix50"])`` field for field and tensor
    for tensor (``tests/test_p9_4_training.py``); the control retrains mix50 through THIS function."""
    import offline.method_tier_grid as grid
    from offline.dt_gate import CONTEXT_LENGTH, stack_dataset
    from offline.offline_baselines import filter_stacked_to_streams, iql_reward_scale

    dataset = grid.tier_dataset(spec, corpus_root)
    stacked = stack_dataset(dataset)
    group = next(iter(dataset.groups))
    scenario_id = dataset.episode_records[0].scenario_id
    components = grid._component_streams(spec, corpus_root, context_length=CONTEXT_LENGTH)
    selected = select_streams(components[spec.components[0]], components[spec.components[1]])
    if len(selected) != int(spec.stream_count):
        raise ValueError(f"{spec.tier}: the selection holds {len(selected)} streams, not {int(spec.stream_count)}")
    declared = grid.assert_declaration_matches_corpus(spec, selected)
    kept = grid.top_decile_streams(selected)
    scale = iql_reward_scale([s.total_return for s in selected])
    digest = grid.statistics_digest(dataset)
    provenance = {
        "tier": spec.tier,
        "dataset_dirs": [str(d) for d in grid.tier_dirs(spec, corpus_root)],
        "training_draw_ids": list(dataset.stats.draw_ids),
        "scenario_id": scenario_id,
        "statistics_digest": digest,
        "subsample": spec.subsample,
        "training_streams": len(selected),
    }
    return grid.TrainingInputs(
        spec=spec, dataset=dataset, stacked=stacked, group=group, scenario_id=scenario_id, selected=tuple(selected),
        declared=declared, kept=tuple(kept), scale=scale, digest=digest, provenance=provenance,
        full_batch=filter_stacked_to_streams(dataset, stacked, selected),
        top_batch=filter_stacked_to_streams(dataset, stacked, kept),
    )


def assert_inputs_as_declared(inputs: Any, *, expected_streams: Sequence[Mapping[str, Any]], target_rtg: float,
                              rtg_scale: float, statistics_digest: str | None) -> None:
    """The inputs a run trains on must be the declared ones: the same streams in the same order, the same target and
    scale, and (where one was declared) the same normalisation digest."""
    keys = [(str(s.dataset_dir), str(s.episode_file), str(s.ix_id)) for s in inputs.selected]
    declared_keys = [(str(row["dataset_dir"]), str(row["episode_file"]), str(row["ix_id"])) for row in expected_streams]
    problems = []
    if keys != declared_keys:
        problems.append("the selected streams are not the declared ones in the declared order")
    if (float(inputs.declared["target_rtg"]), float(inputs.declared["rtg_scale"])) != (float(target_rtg), float(rtg_scale)):
        problems.append(f"target/scale {inputs.declared['target_rtg']}/{inputs.declared['rtg_scale']} are not the "
                        f"declared {target_rtg}/{rtg_scale}")
    if statistics_digest is not None and inputs.digest != statistics_digest:
        problems.append(f"the statistics digest {inputs.digest[:12]} is not the declared {statistics_digest[:12]}")
    if problems:
        raise ValueError(f"{inputs.spec.tier}: " + "; ".join(problems) + "; refusing to train")


def control_inputs(roots: Roots) -> Any:
    """The registered mix50's inputs through :func:`training_inputs`, checked against P4.7's committed declaration:
    its 200 streams in order, its target and scale, its statistics digest."""
    import offline.method_tier_grid as grid

    inputs = training_inputs(grid.TIERS[REGISTERED_TIER], roots.corpus_root)
    committed = json.loads((roots.data_dir / P4_7_DECLARATION).read_bytes())["tiers"][REGISTERED_TIER]
    assert_inputs_as_declared(inputs, expected_streams=committed["streams"], target_rtg=float(committed["target_rtg"]),
                              rtg_scale=float(committed["rtg_scale"]), statistics_digest=str(committed["statistics_digest"]))
    return inputs


def new_tier_inputs(roots: Roots, declaration: Mapping[str, Any]) -> Any:
    """``mix50_fresh``'s inputs through :func:`training_inputs`, checked against the committed P9.4 declaration."""
    inputs = training_inputs(tier_spec(rtg_scale=float(declaration["rtg_scale"])), roots.corpus_root)
    assert_inputs_as_declared(inputs, expected_streams=declaration["streams"], target_rtg=float(declaration["target_rtg"]),
                              rtg_scale=float(declaration["rtg_scale"]),
                              statistics_digest=str(declaration["normalisation"]["statistics_digest"]))
    return inputs


def library_versions() -> dict[str, Any]:
    import platform

    import numpy as np
    import torch

    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def train_run(roots: Roots, run: Run, inputs: Any, *, device: Any, steps: int = GRADIENT_STEPS,
              log_every: int = 0) -> dict[str, Any]:
    """Train ONE run into ``<checkpoint>.partial`` by ``dt_gate.train_dt`` with P9.1's (= ``method_tier_grid._run_train``'s)
    arguments, link it into place (never over an existing file) and write its ``p9.4-run/1.0`` record once.  All three
    paths pass the barrier and are checked absent before the first gradient step."""
    import time

    import offline.method_tier_grid as grid
    from offline.context_sweep import _regime
    from offline.dt_gate import BATCH_SIZE, CONTEXT_LENGTH, runtime_provenance, train_dt
    from offline.offline_baselines import thread_regime

    if inputs.spec.tier != run.tier:
        raise ValueError(f"{run.name}: the inputs are {inputs.spec.tier}'s, not {run.tier}'s")
    final = checkpoint_path(roots, run)
    partial = final.with_name(final.name + ".partial")
    record_path = run_record_path(roots, run)
    for path in (final, partial, record_path):
        assert_target(path, roots)
        if path.exists():
            raise FileExistsError(f"{path} exists; a run is trained once and never over a file on disk")
    final.parent.mkdir(parents=True, exist_ok=True)
    provenance = {**inputs.provenance, "task": "P9.4", "label": LABEL, "run": run.name, "group": run.group,
                  "plan": PLAN_PATH}
    group = inputs.group
    print(f"TRAIN {run.group}:{run.name} ({int(steps)} steps) -> {final}", flush=True)
    started_unix = time.time()
    result = train_dt(
        inputs.full_batch, state_dim=group[0], n_actions=group[1], seed=int(run.seed),
        declared_gradient_steps=int(steps), raise_to=None, context_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE,
        device=device, checkpoint_path=partial, stats=inputs.dataset.stats, scenario_id=inputs.scenario_id,
        target_rtg=float(inputs.spec.target_rtg), rtg_scale=float(inputs.spec.rtg_scale), provenance=provenance,
        log_every=log_every,
    )
    finished_unix = time.time()
    if int(result.gradient_steps) != int(steps) or len(result.losses) != int(steps):
        raise ValueError(f"{run.name}: train_dt ran {len(result.losses)} steps, not {int(steps)}; {partial} is left for "
                         "a person to read")
    os.link(partial, final)
    partial.unlink()
    record = {
        "format_version": RUN_FORMAT_VERSION,
        "task": "P9.4",
        "label": LABEL,
        "run": run.name,
        "group": run.group,
        "tier": run.tier,
        "seed": int(run.seed),
        "declared_gradient_steps": int(steps),
        "gradient_steps": int(result.gradient_steps),
        "seconds": float(result.seconds),
        "clock": CLOCK_LABEL,
        "started_unix": started_unix,
        "finished_unix": finished_unix,
        "checkpoint": str(final),
        "canonical_digest": grid.canonical_digest_of(final),
        "file_sha256": file_sha256(final),
        "window_means": [float(value) for value in result.window_means],
        "plateaued": bool(result.plateaued),
        "final_loss": float(result.losses[-1]),
        "target_rtg": float(inputs.spec.target_rtg),
        "rtg_scale": float(inputs.spec.rtg_scale),
        "statistics_digest": inputs.digest,
        "training_streams": len(inputs.selected),
        "device": str(device),
        "regime": _regime(),
        "thread_regime": thread_regime(),
        "runtime": runtime_provenance(),
    }
    write_once(record_path, record, roots)
    print(f"  done in {record['seconds']:.1f}s ({CLOCK_LABEL})", flush=True)
    return record


def run_status(roots: Roots, run: Run, *, steps: int = GRADIENT_STEPS) -> str:
    """``"done"`` or ``"missing"`` -- anything in between is REFUSED, never retrained and never skipped: a ``.partial``,
    a checkpoint without its record or a record without its checkpoint, a record of another run or budget, or a
    checkpoint whose digests are not the ones its record wrote."""
    import offline.method_tier_grid as grid

    checkpoint = checkpoint_path(roots, run)
    record = run_record_path(roots, run)
    partial = checkpoint.with_name(checkpoint.name + ".partial")
    if partial.exists():
        raise ValueError(f"{run.name}: {partial} exists -- an interrupted run; refusing to retrain over it or skip it")
    if not checkpoint.exists() and not record.exists():
        return "missing"
    if not checkpoint.exists() or not record.exists():
        raise ValueError(f"{run.name}: the checkpoint and its record must both exist; refusing a half-written run")
    payload = json.loads(record.read_text(encoding="utf-8"))
    expected = {"format_version": RUN_FORMAT_VERSION, "run": run.name, "group": run.group, "tier": run.tier,
                "seed": int(run.seed), "declared_gradient_steps": int(steps), "gradient_steps": int(steps)}
    recorded = {key: payload.get(key) for key in expected}
    if recorded != expected:
        raise ValueError(f"{run.name}: the record says {recorded}, not {expected}; refusing it")
    if payload.get("file_sha256") != file_sha256(checkpoint) or payload.get("canonical_digest") != grid.canonical_digest_of(checkpoint):
        raise ValueError(f"{run.name}: the checkpoint's digests are not the ones its record wrote; refusing it")
    return "done"


def reference_checkpoint(roots: Roots, seed: int) -> dict[str, Any]:
    """P4.7's mix50 checkpoint of *seed*, by ``transfer_curve.checkpoint_identity`` (``p4_7_training.json`` and the local
    ``SHA256SUMS_p4_7.txt``) AND by the digest ``p7_3a_zero_shot.json``'s ``inputs.checkpoints`` pins (plan section 3)."""
    import offline.transfer_curve as tc

    identity = tc.checkpoint_identity(REGISTERED_TIER, int(seed), output_root=roots.output_root, data_dir=roots.data_dir)
    pinned = [entry for entry in json.loads((roots.data_dir / P7_3A_ARTIFACT).read_bytes())["inputs"]["checkpoints"]
              if entry["subject"] == REGISTERED_TIER and int(entry["seed"]) == int(seed)]
    if len(pinned) != 1 or str(pinned[0]["file_sha256"]) != identity["file_sha256"]:
        raise ValueError(f"mix50 seed {seed}: {P7_3A_ARTIFACT}'s pin {[p.get('file_sha256') for p in pinned]} is not the "
                         f"file's {identity['file_sha256']}")
    return {"path": identity["path"], "file_sha256": identity["file_sha256"],
            "checked_against": [*identity["sha256_checked_against"], f"{P7_3A_ARTIFACT} inputs.checkpoints"]}


def control_comparison(candidate: str | Path, reference: str | Path) -> dict[str, Any]:
    """One control seed: ``context_sweep.compare_models`` on the two weights-only state dicts (``torch.equal`` per
    parameter; the largest absolute difference in float64) -- P5.3c's own comparison, its per-parameter list dropped."""
    from offline.context_sweep import compare_models
    from offline.few_shot import _load_weights_only

    comparison = compare_models(_load_weights_only(candidate)["model"], _load_weights_only(reference)["model"])
    return {key: value for key, value in comparison.items() if key != "parameters"}


def control_record(roots: Roots, *, steps: int = GRADIENT_STEPS) -> dict[str, Any]:
    """Plan section 3's control over the five control runs: each against P4.7's checkpoint of its seed.  Requires every
    control run done; passes only at 5 of 5."""
    seeds = []
    for run in control_runs():
        if run_status(roots, run, steps=steps) != "done":
            raise ValueError(f"{run.name} is not trained; the control compares five runs or none")
        candidate = checkpoint_path(roots, run)
        reference = reference_checkpoint(roots, run.seed)
        seeds.append({"seed": int(run.seed), "candidate": str(candidate), "candidate_file_sha256": file_sha256(candidate),
                      "reference": reference["path"], "reference_file_sha256": reference["file_sha256"],
                      "reference_checked_against": reference["checked_against"],
                      **control_comparison(candidate, reference["path"])})
    equal = sum(1 for entry in seeds if entry["all_equal"] is True)
    return {
        "format_version": CONTROL_FORMAT_VERSION,
        "task": "P9.4",
        "label": LABEL,
        "rule": "every parameter torch.equal to P4.7's checkpoint of the same seed (context_sweep.compare_models); "
                "5 of 5 or nothing else is trained or evaluated (plan section 3)",
        "declared_gradient_steps": int(steps),
        "seeds": seeds,
        "n_seeds_equal": equal,
        "passed": equal == len(SEEDS),
        "library_versions": library_versions(),
    }


def control_report(record: Mapping[str, Any]) -> str:
    """The author's report when the control fails (plan section 3): each seed's largest absolute difference and the
    library versions."""
    lines = [f"CONTROL: {record['n_seeds_equal']} of {len(record['seeds'])} seeds reproduce P4.7's mix50 checkpoints"]
    for entry in record["seeds"]:
        lines.append(f"  seed {entry['seed']}: all_equal {entry['all_equal']}, parameters differing "
                     f"{entry['n_parameters_differing']} of {entry['n_parameters']}, largest absolute difference "
                     f"{entry['largest_abs_difference']!r} ({entry['largest_abs_difference_parameter']})")
    lines.append(f"  library versions: {json.dumps(record['library_versions'], sort_keys=True)}")
    return "\n".join(lines)


def read_control(roots: Roots) -> dict[str, Any] | None:
    path = control_record_path(roots)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def assert_control_passed(roots: Roots) -> dict[str, Any]:
    """Refuse unless the control record exists, is this task's and says 5 of 5 (``cityflow`` and ``sumo`` start only
    after it, plan section 7)."""
    record = read_control(roots)
    if record is None:
        raise ValueError(f"{control_record_path(roots)} is absent: the control has not run; nothing else starts")
    if record.get("format_version") != CONTROL_FORMAT_VERSION or record.get("declared_gradient_steps") != GRADIENT_STEPS:
        raise ValueError(f"{control_record_path(roots)} is not this task's control record at {GRADIENT_STEPS} steps")
    if record.get("passed") is not True or record.get("n_seeds_equal") != len(SEEDS):
        raise ValueError("the control did not pass (" + str(record.get("n_seeds_equal")) + " of 5); nothing else is "
                         "trained or evaluated:\n" + control_report(record))
    return record


# ----------------------------------------------------------------------
# Sessions (PROJECT_PLAN section 7 rule (5)): a few hours, ending between runs, resumed from disk
# ----------------------------------------------------------------------


def stop_file(roots: Roots) -> Path:
    """``output/p9_4_runs/STOP``: while it exists no session starts, and a running session ends between units of
    work.  The code never creates or deletes it; the author does."""
    return Path(roots.runs_root) / "STOP"


def session_stop_reason(roots: Roots, deadline_unix: float | None) -> str | None:
    """Why a session ends before its next unit of work -- ``"the stop file"`` or ``"the hours bound"`` -- or ``None``."""
    import time

    if stop_file(roots).exists():
        return "the stop file"
    if deadline_unix is not None and time.time() >= float(deadline_unix):
        return "the hours bound"
    return None


@dataclass(frozen=True)
class TrainPlan:
    """A training session's plan: its runs in order and their planned seconds (planning estimates)."""

    runs: tuple[Run, ...]
    planned_seconds: float
    done: int
    missing: int
    hours: float

    def as_json(self) -> dict[str, Any]:
        return {"runs": [run.name for run in self.runs], "planned_seconds": self.planned_seconds,
                "planned_hours": self.planned_seconds / 3600.0, "done": self.done, "missing": self.missing,
                "hours_bound": self.hours, "note": PLANNING_NOTE}


def train_plan(roots: Roots, *, hours: float, steps: int = GRADIENT_STEPS) -> TrainPlan:
    """The session's runs: the missing ones in :func:`training_runs` order whose planned seconds (planning estimates,
    each tier's inputs counted once) fit *hours* -- the first always included.  Every run's status is read from disk
    first and a damaged run refuses the plan; a FAILED control refuses it too.  The control's verdict is not a planning
    question: :func:`train_session` checks it before every new-tier run."""
    if not float(hours) > 0.0:
        raise ValueError(f"--hours {hours}: a session needs a positive number of hours")
    statuses = {run: run_status(roots, run, steps=steps) for run in training_runs()}
    control = read_control(roots)
    if control is not None and control.get("passed") is not True:
        raise ValueError("the control did not pass; nothing else is trained:\n" + control_report(control))
    planned: list[Run] = []
    total = 0.0
    tiers: set[str] = set()
    for run in training_runs():
        if statuses[run] != "missing":
            continue
        cost = PLANNED_RUN_SECONDS + (0.0 if run.tier in tiers else PLANNED_INPUT_SECONDS)
        if planned and total + cost > float(hours) * 3600.0:
            break
        planned.append(run)
        tiers.add(run.tier)
        total += cost
    return TrainPlan(runs=tuple(planned), planned_seconds=total,
                     done=sum(1 for state in statuses.values() if state == "done"),
                     missing=sum(1 for state in statuses.values() if state == "missing"), hours=float(hours))


def ensure_control_record(roots: Roots, *, steps: int = GRADIENT_STEPS) -> dict[str, Any] | None:
    """Write the control record once, as soon as all five control runs are done; return it (or ``None`` before)."""
    existing = read_control(roots)
    if existing is not None:
        return existing
    if not all(run_status(roots, run, steps=steps) == "done" for run in control_runs()):
        return None
    record = control_record(roots, steps=steps)
    write_once(control_record_path(roots), record, roots)
    print(control_report(record), flush=True)
    return record


class ControlFailed(RuntimeError):
    """Plan section 3: the control did not reproduce 5 of 5; nothing else is trained or evaluated."""


def train_session(roots: Roots, *, hours: float, device: Any = None, steps: int = GRADIENT_STEPS, log_every: int = 0,
                  enter_regime: bool = True) -> dict[str, Any]:
    """ONE training session (plan section 7): every refusal first -- the stop file, the committed declaration at its
    pin, Amendment A1's re-hash of the 200 files, P4.7's five reference checkpoints at their pins, the registered regime
    (``context_sweep.enter_registered_regime``) -- then the plan, printed; then the runs in order, the stop file checked
    before each.  The control record is written as soon as the fifth control run is done; a failed control raises
    :class:`ControlFailed` with plan section 3's report, and no new-tier run ever starts before a passed record."""
    import torch

    if stop_file(roots).exists():
        raise ValueError(f"{stop_file(roots)} exists: no session starts while it exists (the author removes it)")
    declaration = load_declaration(roots)
    files = assert_files_as_declared(declaration)
    for seed in SEEDS:
        reference_checkpoint(roots, seed)
    regime = None
    if enter_regime:
        from offline.context_sweep import enter_registered_regime

        regime = enter_registered_regime()
    control = ensure_control_record(roots, steps=steps)
    if control is not None and control.get("passed") is not True:
        raise ControlFailed(control_report(control))
    plan = train_plan(roots, hours=hours, steps=steps)
    print("SESSION PLAN " + json.dumps(plan.as_json(), sort_keys=True), flush=True)
    resolved = torch.device("cuda" if device is None else device)
    cache: dict[str, Any] = {}
    trained: list[dict[str, Any]] = []
    ended_by = "the plan"
    for run in plan.runs:
        if stop_file(roots).exists():
            ended_by = "the stop file"
            print(f"SESSION: {stop_file(roots)} exists -- the session ends here, between runs", flush=True)
            break
        if run.group == NEW_GROUP:
            assert_control_passed(roots)
        if run.tier not in cache:
            cache[run.tier] = control_inputs(roots) if run.group == CONTROL_GROUP else new_tier_inputs(roots, declaration)
        record = train_run(roots, run, cache[run.tier], device=resolved, steps=steps, log_every=log_every)
        trained.append({"run": run.name, "seconds": record["seconds"]})
        if run.group == CONTROL_GROUP:
            control = ensure_control_record(roots, steps=steps)
            if control is not None and control.get("passed") is not True:
                raise ControlFailed(control_report(control))
    left = [run.name for run in training_runs() if run_status(roots, run, steps=steps) == "missing"]
    summary = {"trained": trained, "seconds_trained": sum(entry["seconds"] for entry in trained), "ended_by": ended_by,
               "missing_left": left, "files_as_declared": files, "regime": None if regime is None else regime["regime"],
               "control": None if read_control(roots) is None else {
                   "passed": read_control(roots)["passed"], "n_seeds_equal": read_control(roots)["n_seeds_equal"]}}
    print("SESSION SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


# ----------------------------------------------------------------------
# The pre-token checks and the command line (plan section 7)
# ----------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_TREE = Path("/home/filip/rltraffic")
DEFAULT_OUTPUT_ROOT = MAIN_TREE / "output"
DEFAULT_CORPUS_ROOT = MAIN_TREE / "datasets_v11"
DEFAULT_DRAWS_ROOT = MAIN_TREE / "scenarios" / "draws"
STAGES: tuple[str, ...] = ("train", "cityflow", "sumo")


def assert_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available; plan section 3's regime trains on cuda only")


def assert_held_out_draws(roots: Roots, *, backend: str) -> None:
    """Every held-out draw materialised for *backend*: CityFlow's config, or SUMO's parity configuration."""
    from offline.materialise_draws import draw_config_path

    import offline.transfer_curve as tc

    missing = []
    for draw in HELD_OUT_DRAWS:
        if backend == "cityflow":
            path = Path(draw_config_path("cityflow1x1", int(draw), out_root=roots.draws_root))
        elif backend == "sumo":
            path = Path(draw_config_path("cityflow1x1", int(draw), out_root=roots.draws_root)).parent / "parity" / tc.PARITY_CONFIG_NAME
        else:
            raise ValueError(f"unknown backend {backend!r}")
        if not path.is_file():
            missing.append(int(draw))
    if missing:
        raise FileNotFoundError(f"held-out draws {missing[:3]} (of {len(missing)}) are not materialised for {backend}")


def check(roots: Roots, *, stage: str) -> dict[str, Any]:
    """Every refusal of *stage* before the token; writes nothing.  ``train``: the committed declaration at its pin,
    Amendment A1's re-hash, P4.7's five references at their pins, every run readable, a failed control refused, CUDA.
    ``cityflow`` and ``sumo``: the declaration, the control at 5 of 5, the new tier's five runs done, the held-out
    draws materialised (and P4.7's references for G1)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if stop_file(roots).exists():
        raise ValueError(f"{stop_file(roots)} exists: no session starts while it exists (the author removes it)")
    declaration = load_declaration(roots)
    if stage == "train":
        assert_files_as_declared(declaration)
        for seed in SEEDS:
            reference_checkpoint(roots, seed)
        for run in training_runs():
            run_status(roots, run)
        control = read_control(roots)
        if control is not None and control.get("passed") is not True:
            raise ControlFailed(control_report(control))
        assert_cuda()
        return {"stage": stage, "ok": True}
    assert_control_passed(roots)
    for run in new_runs():
        if run_status(roots, run) != "done":
            raise ValueError(f"{run.name} is not trained; the {stage} stage starts after the train stage")
    assert_held_out_draws(roots, backend=stage)
    if stage == "sumo":
        for seed in SEEDS:
            reference_checkpoint(roots, seed)
    return {"stage": stage, "ok": True}


def smoke_roots(scratch: str | Path, *, repo_root: str | Path) -> Roots:
    """The roots of a reduced-size smoke run: its own ``output/`` under *scratch*, which must lie outside the main
    tree (and so outside its ``output/``) and outside the repository; the corpus and the draws are read, never written."""
    path = Path(scratch).resolve()
    forbidden = [MAIN_TREE.resolve(), Path(repo_root).resolve()]
    if any(path == root or root in path.parents for root in forbidden):
        raise ValueError(f"the smoke root {path} must lie outside {[str(root) for root in forbidden]}")
    return Roots(repo_root=Path(repo_root), output_root=path / "output", corpus_root=DEFAULT_CORPUS_ROOT,
                 draws_root=DEFAULT_DRAWS_ROOT)


def status(roots: Roots) -> dict[str, Any]:
    """Every stage's state, recomputed from disk."""
    import offline.p9_4_evaluation as ev

    training: dict[str, str] = {}
    for run in training_runs():
        try:
            training[run.name] = run_status(roots, run)
        except ValueError as error:
            training[run.name] = f"REFUSED: {error}"
    control = read_control(roots)
    return {"training": training,
            "control": None if control is None else {"passed": control.get("passed"),
                                                     "n_seeds_equal": control.get("n_seeds_equal")},
            "cityflow": ev.cityflow_status(roots), "sumo": ev.sumo_status(roots), "stop_file": stop_file(roots).exists()}


def _workers(value: str) -> int:
    import argparse

    number = int(value)
    if not 1 <= number <= MAX_WORKERS:
        raise argparse.ArgumentTypeError(f"--workers {value}: at most {MAX_WORKERS} (plan section 7)")
    return number


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m offline.p9_4_mix50_fresh",
                                     description="P9.4: mix50 retrained on a corrected mixture (exploratory)")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--draws-root", type=Path, default=DEFAULT_DRAWS_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("declare", help="plan 2.1-2.4: write output/p9_4/declaration.json once")
    checking = sub.add_parser("check", help="every refusal of a stage, before its token; writes nothing")
    checking.add_argument("--stage", choices=STAGES, required=True)
    training = sub.add_parser("train", help="one training session: the control first, then mix50_fresh")
    training.add_argument("--hours", type=float, required=True)
    training.add_argument("--plan-only", action="store_true")
    training.add_argument("--log-every", type=int, default=0)
    for name in ("route-i", "route-ii"):
        rolling = sub.add_parser(name, help=f"CityFlow {name}")
        rolling.add_argument("--deadline-unix", type=float, default=None)
        rolling.add_argument("--stall-dump", type=Path, default=None)
        rolling.add_argument("--stall-seconds", type=float, default=120.0)
        if name == "route-ii":
            rolling.add_argument("--shard", type=int, default=0)
            rolling.add_argument("--of", type=int, default=1)
    sub.add_parser("declare-route-ii", help="write route (ii)'s campaign manifest once, before the shards")
    sumo = sub.add_parser("sumo", help="one SUMO session: G1 and G2 first, then the 1,000 cells")
    sumo.add_argument("--workers", type=_workers, required=True)
    sumo.add_argument("--deadline-unix", type=float, required=True)
    sub.add_parser("status", help="every stage's state, from disk")
    reporting = sub.add_parser("report", help="the three artifacts of plan section 5")
    reporting.add_argument("--out-dir", type=Path, default=None)
    smoke = sub.add_parser("smoke", help="every distinct path once at reduced size, in a scratch root")
    smoke.add_argument("--scratch", type=Path, required=True)
    smoke.add_argument("--steps", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The command line.  A refusal raises (exit 1); a failed control or guard prints plan section 3's or 4.3's report
    and exits 3."""
    import sys

    args = build_parser().parse_args(None if argv is None else list(argv))
    roots = Roots(repo_root=Path(args.repo_root), output_root=Path(args.output_root),
                  corpus_root=Path(args.corpus_root), draws_root=Path(args.draws_root))
    import offline.p9_4_evaluation as ev

    if getattr(args, "stall_dump", None) is not None:
        from offline import stall_guard

        stall_guard.install(args.stall_dump, seconds=float(args.stall_seconds))
    try:
        if args.command == "declare":
            payload = declare(roots)
            print(json.dumps({"target_rtg": payload["target_rtg"], "rtg_scale": payload["rtg_scale"],
                              "checks": {name: payload["checks"][name]["passed"] for name in CHECK_NAMES}}, sort_keys=True))
        elif args.command == "check":
            print(json.dumps(check(roots, stage=args.stage), sort_keys=True))
        elif args.command == "train":
            if args.plan_only:
                print("SESSION PLAN " + json.dumps(train_plan(roots, hours=args.hours).as_json(), sort_keys=True))
            else:
                train_session(roots, hours=args.hours, log_every=args.log_every)
        elif args.command == "route-i":
            reason = session_stop_reason(roots, args.deadline_unix)
            if reason is not None:
                print(f"route (i) not started: {reason}")
            else:
                ev.route_i(roots, load_declaration(roots))
                print("route (i): complete on disk")
        elif args.command == "declare-route-ii":
            manifest = ev.declare_route_ii(roots)
            print(f"route (ii) declared: {manifest['n_cells']} episodes")
        elif args.command == "route-ii":
            outcome = ev.run_route_ii(roots, load_declaration(roots), shard=args.shard, of=args.of,
                                      deadline_unix=args.deadline_unix)
            print("ROUTE II " + json.dumps({key: outcome.get(key) for key in ("rolled", "skipped_already_present",
                                                                             "n_refused", "stopped", "not_reached")},
                                           sort_keys=True))
        elif args.command == "sumo":
            outcome = ev.sumo_session(roots, load_declaration(roots), workers=args.workers,
                                      deadline_unix=args.deadline_unix)
            print("SUMO SESSION " + json.dumps({"stopped_by": outcome.get("stopped_by"),
                                                "new": outcome.get("new")}, sort_keys=True, default=str))
        elif args.command == "status":
            print(json.dumps(status(roots), indent=2, sort_keys=True))
        elif args.command == "report":
            import offline.p9_4_report as rep

            print(json.dumps(rep.write_artifacts(roots, args.out_dir), indent=2, sort_keys=True))
        elif args.command == "smoke":
            import offline.p9_4_smoke as smoke

            print(json.dumps(smoke.run_smoke(smoke_roots(args.scratch, repo_root=roots.repo_root),
                                             steps=args.steps), indent=2, sort_keys=True))
    except (ControlFailed, ev.GuardFailed) as failure:
        print(str(failure), file=sys.stderr, flush=True)
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - the driver's entry point
    raise SystemExit(main())
