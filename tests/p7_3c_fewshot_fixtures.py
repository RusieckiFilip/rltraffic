"""Fixtures for P7.3c's fine-tune tests (``BRIEF_41`` C3-C4): a synthetic SOURCE and synthetic 100-draw corpora.

Not a test module -- the name carries no ``test_`` prefix -- so pytest collects nothing here.

THE SOURCE: THE REGISTERED SUBJECT'S ON-DISK SHAPE AT A SMALL ARCHITECTURE
--------------------------------------------------------------------------
:func:`write_source` writes a ``spatial-dt-checkpoint/1.0`` payload with EVERY top-level key the registered subject
carries, each in the SAME Python type.  The shape was read on 2026-09-25 from
``output/p5_2/checkpoints/grid4x4_mappo1000_dt_nomix_h4_seed101.pt`` by unpickling ``data.pkl`` with stub classes (no
torch, no tensor data read):

* eleven keys -- ``config``, ``format_version``, ``intersection_ids``, ``model``, ``normalise``, ``provenance``,
  ``rtg_scale``, ``scenario_id``, ``spatial_mask``, ``stats``, ``target_rtg``;
* ``spatial_mask`` a list of lists of ``bool`` (``train_tier_dt`` saves ``ndarray.tolist()``), never a tensor;
* ``stats`` the ``NormalizationStats`` JSON object, with **10-12 zero-std columns per intersection** -- so the fixture
  has zero-std columns too, and a test's independent normalisation route must apply the loader's rule for them
  (``offline/dataset.py``: a zero std is left as 1.0);
* ``provenance`` with P5.2's recipe (batch 64, lr 1e-4, weight decay 1e-4, clip 0.25, warm-up 1,000, 40,000 steps,
  ``method dt_nomix_h4``, ``n_head 4``, ``device cuda``, ``deterministic False``) and a ``runtime`` block whose
  ``torch_version`` is a ``torch.torch_version.TorchVersion`` -- the one class a weights-only load must allowlist.

Only the architecture is small (``d_model`` 16, one layer, context 4), so a CPU step costs milliseconds.  The widths
that matter are the subject's: 16 nodes, state 40, 8 actions, 4 heads, no mixing, ``max_ep_len`` 360.

**No tensor of the source equals a fresh initialisation**: the weights are a fresh model built under
:data:`SOURCE_INIT_SEED` plus :data:`SOURCE_OFFSET` on every element.  Without the offset the LayerNorm weights (ones)
and biases (zeros) of the source would equal those of ANY fresh model, and "the scratch arm is not the source" would
hold for the wrong tensors only.

The per-intersection constants that the calibration artifact also records (``rtg_scale``, and ``target_rtg`` as
``best_source_return``) are READ from ``docs/data/p7_3d_calibration.json`` by this file's own route, so a synthetic
source describes the same subject the committed artifact describes.

THE CORPORA
-----------
Written through the REAL :class:`offline.trajectory_logger.TrajectoryLogger` (``tests/p7_3c_fixtures.py``'s scripted
env), one episode per draw over 201-300 by default, with an episode length chosen PER DRAW, and ``SHA256SUMS`` in
``sha256sum``'s format.  Rewards are small integers, so every return is exact in float32 and in float64.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from agent.SpatialDTAgent import SpatialDecisionTransformer, SpatialDTConfig
from offline.trajectory_logger import TrajectoryLogger
from tests.p7_3c_fixtures import (
    GRID,
    N_ACTIONS,
    STATE_DIM,
    ScriptedCorpusEnv,
    calibration,
    calibration_ids,
    grid_run_metadata,
    log_episode,
    write_sha256sums,
)

#: The corpus band A24 registers: draws 201-300, one episode each.
CORPUS_DRAWS: tuple[int, ...] = tuple(range(201, 301))
#: The synthetic source's OWN initialisation seed -- deliberately not one of the five training seeds.
SOURCE_INIT_SEED = 7
#: Added to every element of the synthetic source's weights (see the module docstring).
SOURCE_OFFSET = 0.5
#: Columns of the state whose recorded std is 0.0 in the synthetic statistics (the subject has 10-12 per intersection).
ZERO_STD_COLUMNS: tuple[int, ...] = (5, 17, 33)

#: The subject's widths, a small depth: see the module docstring.
SMALL_CONFIG: dict[str, Any] = {
    "state_dim": STATE_DIM,
    "n_actions": N_ACTIONS,
    "n_nodes": 16,
    "context_length": 4,
    "n_layer": 1,
    "n_head": 4,
    "d_model": 16,
    "dropout": 0.1,
    "max_ep_len": 360,
    "spatial_mixing": False,
}


def sha256_file(path: str | Path) -> str:
    """The file's sha256, by this file's own route."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_provenance(**overrides: Any) -> dict[str, Any]:
    """P5.2's recorded recipe, key for key as the subject's payload carries it; *overrides* replace fields."""
    from torch.torch_version import TorchVersion

    provenance: dict[str, Any] = {
        "batch_size": 64,
        "deterministic": False,
        "device": "cuda",
        "grad_clip": 0.25,
        "gradient_steps": 40000,
        "learning_rate": 0.0001,
        "method": "dt_nomix_h4",
        "n_head": 4,
        "roadnet_sha256": "0" * 64,
        "runtime": {
            "torch_version": TorchVersion(torch.__version__),
            "torch_cuda_version": None,
            "cuda_available": False,
            "cuda_device_name": None,
            "torch_num_threads": 1,
            "numpy_version": np.__version__,
            "python_version": "3.12",
            "git_commit": "0" * 40,
            "written_at_git_commit": "0" * 40,
            "measurement_git_commits": [],
            "unreachable_measurement_commits": [],
        },
        "seed": 101,
        "spatial_mixing": False,
        "tier": "mappo1000",
        "warmup_steps": 1000,
        "weight_decay": 0.0001,
    }
    provenance.update(overrides)
    return provenance


def synthetic_stats(
    ids: Sequence[str], *, scenario: str = GRID, draw_ids: Sequence[int] = tuple(range(1, 201))
) -> dict[str, Any]:
    """A ``NormalizationStats`` JSON object: non-trivial means and stds, three zero-std columns per intersection.

    Every value is a multiple of 1/8, so it is exact in float32 and the JSON round trip changes nothing.
    """
    mean: dict[str, list[float]] = {}
    std: dict[str, list[float]] = {}
    for position, ix in enumerate(ids):
        mean[str(ix)] = [float(k % 11) + 0.25 * position for k in range(STATE_DIM)]
        std[str(ix)] = [
            0.0 if k in ZERO_STD_COLUMNS else 1.5 + 0.5 * ((k + position) % 4) for k in range(STATE_DIM)
        ]
    return {
        "stats_version": "1.0",
        "split": "train",
        "draw_ids": [int(d) for d in draw_ids],
        "dataset_dirs": ["/synthetic/cf_grid4x4__mappo1000__seed101"],
        "state_mean": {scenario: mean},
        "state_std": {scenario: std},
        "row_count": {scenario: {str(ix): 72000 for ix in ids}},
        "rtg": {
            scenario: {
                str(ix): {
                    "count": 72000,
                    "min": -250.0,
                    "max": 0.0,
                    "mean": -100.0,
                    "std": 60.0,
                    "quantiles": [[0.5, -100.0]],
                }
                for ix in ids
            }
        },
    }


#: "Not given": lets a caller pass ``stats=None`` and get a payload whose statistics ARE ``None``.
_DEFAULT: Any = object()


def source_payload(
    *,
    ids: Sequence[str] | None = None,
    config: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    stats: Any = _DEFAULT,
    **overrides: Any,
) -> dict[str, Any]:
    """The synthetic source's payload; keyword *overrides* replace top-level keys after it is built.

    ``stats=None`` writes a payload whose ``stats`` is ``None``; omitting it writes :func:`synthetic_stats`.
    """
    artifact = calibration()
    order = [str(ix) for ix in (ids if ids is not None else calibration_ids())]
    architecture = dict(SMALL_CONFIG if config is None else config)
    torch.manual_seed(SOURCE_INIT_SEED)
    model = SpatialDecisionTransformer(SpatialDTConfig.from_json_obj(architecture))
    weights = {key: value.detach().clone() + SOURCE_OFFSET for key, value in model.state_dict().items()}
    per_ix = artifact["per_intersection"]
    payload: dict[str, Any] = {
        "format_version": "spatial-dt-checkpoint/1.0",
        "config": architecture,
        "model": weights,
        "target_rtg": {ix: float(per_ix[ix]["best_source_return"]) for ix in order},
        "rtg_scale": {ix: float(per_ix[ix]["rtg_scale"]) for ix in order},
        "normalise": True,
        "scenario_id": GRID,
        "stats": synthetic_stats(order) if stats is _DEFAULT else (None if stats is None else dict(stats)),
        "intersection_ids": list(order),
        "spatial_mask": np.eye(len(order), dtype=np.bool_).tolist(),
        "provenance": dict(source_provenance() if provenance is None else provenance),
    }
    payload.update(overrides)
    return payload


def write_source(path: str | Path, **kwargs: Any) -> tuple[Path, str]:
    """Write :func:`source_payload` (*kwargs* passed through) with ``torch.save``; return the path and its sha256."""
    target = Path(path)
    torch.save(source_payload(**kwargs), target)
    return target, sha256_file(target)


def load_payload(path: str | Path) -> dict[str, Any]:
    """A checkpoint read by the TEST's route: ``torch.load``, weights only, ONE allowlisted class."""
    from torch.torch_version import TorchVersion

    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(Path(path), map_location="cpu", weights_only=True)


def training_rewards(draw: int, ix: str, decisions: int) -> list[float]:
    """Small integer rewards that vary with the draw, the intersection (by ID) and the step."""
    offset = calibration_ids().index(str(ix))
    return [-float((int(draw) + 3 * offset + step) % 7) for step in range(int(decisions))]


def write_training_corpus(
    out_dir: str | Path,
    *,
    ids: Sequence[str] | None = None,
    draws: Sequence[int] = CORPUS_DRAWS,
    decisions: int = 4,
    decisions_for: Callable[[int], int] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    rewards_for: Callable[[int, str, int], Sequence[float]] = training_rewards,
    sha256sums: bool = True,
) -> Path:
    """One episode per draw through the REAL logger, *ids* in the logged (env) order; then ``SHA256SUMS``.

    *decisions_for* gives an episode length per draw (default: *decisions* for every draw).
    """
    order = [str(ix) for ix in (ids if ids is not None else calibration_ids())]
    env = ScriptedCorpusEnv(order)
    metadata = dict(grid_run_metadata(list(draws)) if run_metadata is None else run_metadata)
    logger = TrajectoryLogger(env, Path(out_dir), run_metadata=metadata)
    for draw in draws:
        length = int(decisions if decisions_for is None else decisions_for(int(draw)))
        log_episode(logger, env, draw=int(draw), decisions=length, rewards_for=rewards_for)
        logger.finalize_episode()
    if sha256sums:
        write_sha256sums(Path(out_dir))
    return Path(out_dir)


class FakeIntersection:
    """An id and a phase count: enough for ``SpatialDTAgent``'s constructor."""

    def __init__(self, ix_id: str) -> None:
        self.id = ix_id
        self.incoming_lanes: list[str] = []
        self.num_phases = N_ACTIONS


class FakeSpatialEnv:
    """Sixteen intersections with the subject's ids: enough for ``spatial_agent_with_targets`` to load a payload."""

    max_steps = 360
    delta_time = 10

    def __init__(self, ids: Sequence[str]) -> None:
        self.intersections = [FakeIntersection(str(ix)) for ix in ids]

        class _Space:
            nvec = np.full(len(ids), N_ACTIONS)

        self.action_space = _Space()
