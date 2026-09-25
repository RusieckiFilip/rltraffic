"""P7.3c (``BRIEF_41`` C3-C4): the warm-start fine-tune of the grid4x4 joint model -- A18(d) as A24(b) reads it.

C3 is the fine-tune itself (everything down to ``write_run_record``); C4 is the training driver's Python half -- the
resume decision, the attempt markers, the manifest, the record, G5's fenced timing and the input check -- which
``offline/campaigns/p7_3c_finetune.sh`` calls, deciding nothing that matters in bash.

Checkpoint format.  The top-level ``format_version`` of every payload written here is the SOURCE's,
``spatial-dt-checkpoint/1.0`` (``agent/SpatialDTAgent.py``), because ``SpatialDTAgent.load`` refuses every other value
and every evaluation cell reaches it (``BRIEF_41`` Amendment A1.1).  The few-shot identity lives in
``provenance["few_shot"]["format_version"] = "few-shot-checkpoint/1.0"``.

Alignment convention.  The corpus's, unchanged: C6 v1.1 (``docs/CONTRACTS.md``) -- observation rows ``T + 1``,
decision and outcome rows ``T``, and the reward of decision ``t`` is the one the env returned from step ``t``.
Windows, left padding, ``PAD_ACTION`` and the returns-to-go (suffix sums INCLUSIVE of ``r_t``, accumulated in float64)
are ``offline/dataset.py``'s; the joint grouping is ``offline/joint_windows.py``'s, keyed by ``item_meta``.  Nothing
about a window is redefined here.

WHAT IS FROZEN, WHERE EACH VALUE COMES FROM, AND THE TEST THAT PINS IT (``tests/test_p7_3c_few_shot.py``)
------------------------------------------------------------------------------------------------------
* the weights at step 0 of the fine-tune: ``source["model"]``, ``load_state_dict(strict=True)`` -- T-warm;
* ``max_ep_len`` (360): ``source["config"]``, never the data -- T-warm;
* the state mean and std: ``source["stats"]``, handed to the loader through ``stats=`` -- T-frozen;
* ``rtg_scale[ix]``: ``source["rtg_scale"]``, in the checkpoint's node order -- T-frozen;
* the node order and the spatial mask: ``source["intersection_ids"]`` and ``source["spatial_mask"]`` -- T-order;
* the recipe constants: imported where ``train_tier_dt`` imports them, asserted equal to the source's provenance --
  T-recipe;
* the target per k: ``p7_3d_calibration.json`` ``per_intersection[ix].budgets["k{k}"].target`` -- T-payload.

THE ROUTE -- every step validated before any work, and the only write last
--------------------------------------------------------------------------
1. The request: ``init``, ``k``, the budget, the seed, and a destination that does NOT exist in a directory that does.
2. The source, at its pin, loaded weights-only with ONE allowlisted class; its shape and its recorded recipe checked.
3. The k targets from the digest-pinned calibration artifact, on the scale of THIS subject.
4. The prefix: the corpus's ``SHA256SUMS`` re-verified entry by entry, the manifest's band exactly 201-300 with one
   episode per draw, and the first k draws; disjoint from the source's training draws and the held-out pool.
5. The windows with the SOURCE's statistics supplied (nothing is fitted), joint in the SOURCE's node order.
6. ``Utils.seed_everything(seed, seed_python_random=False)`` (Amendment A, Q6), the model from the PAYLOAD's config,
   and -- for the fine-tune only -- the source's weights.  The construction is the same in both arms, so the torch
   RNG is in the same state in both after it.
7. Exactly B steps of ``offline.tier_sweep.train_tier_dt``'s loop: a fresh AdamW, its warm-up, the gradient clip,
   64 joint instants per step drawn uniformly with replacement by ``np.random.default_rng(seed)``, the RTG divided
   per node by the frozen ``rtg_scale``, the frozen mask.  ``train_tier_dt`` itself is not called and not changed:
   it builds a fresh model and fits statistics, which are the two things a fine-tune must not do.
8. The payload -- the source's frozen parts unchanged, the fine-tuned weights, the k targets, the provenance -- written
   ONCE through a hard link that refuses an existing file.  No wall-clock value is in the payload, so two same-seed
   runs can agree byte for byte; seconds and losses go to the run record.

THE FROM-SCRATCH SWITCH (A24(c)(ii), ``scratch_k100``): ``init="scratch"`` skips the one ``load_state_dict`` call and
changes nothing else -- statistics, ``rtg_scale``, node order, mask, corpus, budget, recipe, sampler and target are the
source's and the fine-tune's.  The payloads of the two arms differ in ``model`` and ``provenance.few_shot.init`` only.

WHAT IS NOT HERE: no evaluation and no selection.  The payload after exactly B steps is the only one written.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

__all__ = [
    "CORPUS_DRAWS",
    "FEW_SHOT_FORMAT_VERSION",
    "INITS",
    "REGISTERED_KS",
    "RUN_RECORD_FORMAT_VERSION",
    "CorpusPrefix",
    "FineTuneResult",
    "KTargets",
    "PreparedRun",
    "RunSpec",
    "TrainingOutcome",
    "Windows",
    "assert_fence",
    "attempts_of",
    "build_model",
    "build_record",
    "build_windows",
    "check_inputs",
    "choose_concurrency",
    "draw_rows",
    "fine_tune",
    "joint_batch",
    "load_k_targets",
    "load_source",
    "main",
    "make_optimiser",
    "next_attempt",
    "payload_for",
    "prepare_fine_tune",
    "registered_destination",
    "registered_runs",
    "registered_source_path",
    "resume_decision",
    "run_by_name",
    "run_record_path",
    "select_prefix",
    "summarize_timing",
    "timing_destination",
    "timing_spec",
    "train_prepared",
    "validate_checkpoint",
    "write_manifest",
    "write_payload_exclusive",
    "write_record",
    "write_run_record",
]

#: The few-shot identity, carried in ``provenance["few_shot"]`` (Amendment A1.1); the top level stays the source's.
FEW_SHOT_FORMAT_VERSION = "few-shot-checkpoint/1.0"
#: The per-run record ``train`` writes beside the checkpoint's directory: seconds and losses live here, not in the payload.
RUN_RECORD_FORMAT_VERSION = "p7.3c-finetune-run/1.0"
#: The only top-level format the evaluation path reads (``agent/SpatialDTAgent.py:857-862``).
SPATIAL_FORMAT_VERSION = "spatial-dt-checkpoint/1.0"

#: A20(a)'s subject: the identity-graph control of the spatial architecture, four heads, on grid4x4.
SCENARIO_ID = "cityflow_grid4x4"
METHOD = "dt_nomix_h4"
N_HEAD = 4
SOURCE_GRADIENT_STEPS = 40_000

#: A24(b): the corpus band, and the first draw of every prefix.
CORPUS_DRAWS: tuple[int, ...] = tuple(range(201, 301))
FIRST_DRAW = 201
#: A24(b): the fine-tune budgets k, the primary B, the budget secondary, and G5's fenced timing budget.
REGISTERED_KS: tuple[int, ...] = (5, 20, 100)
REGISTERED_BUDGET = 4_000
SECONDARY_BUDGETS: tuple[int, ...] = (1_000, 16_000)
TIMING_BUDGET = 400
TRAINING_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)

#: The switch (A24(c)(ii)): the fine-tune loads the source's weights; the from-scratch control does not.
INIT_SOURCE = "source"
INIT_SCRATCH = "scratch"
INITS: tuple[str, ...] = (INIT_SOURCE, INIT_SCRATCH)

#: A20(b) as A24(a)-(b) amend it: k100 is the registered prompt; k5 and k20 are admitted for THESE checkpoints only.
ROLE_BY_K: dict[int, str] = {5: "recorded_not_evaluated", 20: "recorded_not_evaluated", 100: "registered_prompt"}
TARGET_RULE = (
    "A24(b): target_rtg[ix] = docs/data/p7_3d_calibration.json per_intersection[ix].budgets['k{k}'].target -- Rule B, "
    "statistic mean, over the same k SUMO MaxPressure episodes the fine-tune reads; k5 and k20 are admitted as "
    "prompts for these fine-tuned checkpoints only (A24(a))"
)
SEEDING_CALL = "Utils.seed_everything(seed, seed_python_random=False)"
SAMPLER = (
    "np.random.default_rng(seed).integers(0, n_joint_instants, size=64) per step: 64 joint decision instants drawn "
    "uniformly WITH replacement, as offline.tier_sweep.train_tier_dt draws them"
)

TRAINING_DIRNAME = "p7_3c_training"
CHECKPOINTS_DIRNAME = "checkpoints"
RUNS_DIRNAME = "runs"
FENCED_TIMING_DIRNAME = "fenced_timing"
SUMS_NAME = "SHA256SUMS"
MANIFEST_NAME = "manifest.json"
CORPUS_FORMAT_VERSION = "1.1"
CALIBRATION_NAME = "p7_3d_calibration.json"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "docs" / "data"
_SUMS_LINE = re.compile(r"([0-9a-f]{64})  (\S+)")


# ======================================================================================================================
# The registered table
# ======================================================================================================================


@dataclass(frozen=True)
class RunSpec:
    """One training run: the cell subject it produces, the switch, k, the budget B and the seed."""

    subject: str
    init: str
    k: int
    budget: int
    seed: int

    @property
    def name(self) -> str:
        """``<subject>_seed<seed>`` -- the checkpoint's file stem and the run's name everywhere."""
        return f"{self.subject}_seed{self.seed}"


#: A24(b)'s six trained subjects (plan section 9, Amendment A Q8), each x the five seeds: thirty runs.
REGISTERED_SUBJECTS: tuple[tuple[str, str, int, int], ...] = (
    ("ft_k5", INIT_SOURCE, 5, REGISTERED_BUDGET),
    ("ft_k20", INIT_SOURCE, 20, REGISTERED_BUDGET),
    ("ft_k100", INIT_SOURCE, 100, REGISTERED_BUDGET),
    ("scratch_k100", INIT_SCRATCH, 100, REGISTERED_BUDGET),
    ("ft_k100_b1000", INIT_SOURCE, 100, SECONDARY_BUDGETS[0]),
    ("ft_k100_b16000", INIT_SOURCE, 100, SECONDARY_BUDGETS[1]),
)


def registered_runs() -> tuple[RunSpec, ...]:
    """The thirty runs A24(b) registers, subject by subject, each over the seeds 101 ... 505 in order."""
    return tuple(
        RunSpec(subject=subject, init=init, k=k, budget=budget, seed=seed)
        for subject, init, k, budget in REGISTERED_SUBJECTS
        for seed in TRAINING_SEEDS
    )


def run_by_name(name: str) -> RunSpec:
    """The registered run called *name*; refuses anything else by name."""
    for spec in registered_runs():
        if spec.name == name:
            return spec
    raise ValueError(
        f"{name!r} is not one of the 30 registered runs (A24(b): {[s[0] for s in REGISTERED_SUBJECTS]} x seeds "
        f"{list(TRAINING_SEEDS)}); nothing else is trained under this name"
    )


def timing_spec() -> RunSpec:
    """G5's FENCED timing run: k 5, B 400, seed 101 -- a configuration A24 does not register."""
    return RunSpec(subject="timing_ft_k5_b400", init=INIT_SOURCE, k=5, budget=TIMING_BUDGET, seed=101)


def _require_k(k: Any) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) not in REGISTERED_KS:
        raise ValueError(
            f"k {k!r} is not one of A24's {list(REGISTERED_KS)}; the prefixes are 201-205, 201-220 and 201-300, "
            "and any other set is refused"
        )
    return int(k)


# ======================================================================================================================
# Paths, and the fence
# ======================================================================================================================


def training_root(output_root: str | Path) -> Path:
    """``<output_root>/p7_3c_training``: the checkpoints, the run records, the attempt markers, the fenced timing."""
    return Path(output_root) / TRAINING_DIRNAME


def registered_source_path(output_root: str | Path, seed: int) -> Path:
    """A20(a)'s checkpoint for *seed* under *output_root*, named as ``offline.transfer_calibration`` names it."""
    from offline.transfer_calibration import GRID4X4_CHECKPOINT_STEM, GRID4X4_CHECKPOINT_SUBDIR

    return Path(output_root) / GRID4X4_CHECKPOINT_SUBDIR / f"{GRID4X4_CHECKPOINT_STEM}{int(seed)}.pt"


def registered_destination(output_root: str | Path, spec: RunSpec) -> Path:
    """Where the registered run *spec* writes its checkpoint: ``p7_3c_training/checkpoints/<name>.pt``."""
    return training_root(output_root) / CHECKPOINTS_DIRNAME / f"{spec.name}.pt"


def run_record_path(output_root: str | Path, spec: RunSpec) -> Path:
    """Where ``train`` records *spec*'s seconds and losses: ``p7_3c_training/runs/<name>.json``."""
    return training_root(output_root) / RUNS_DIRNAME / f"{spec.name}.json"


def assert_fence(path: str | Path, *, timing: bool) -> Path:
    """Refuse a TIMING checkpoint outside ``fenced_timing/`` and a REGISTERED one inside it; return *path*.

    ``BRIEF_41`` §2: timing runs use a configuration A24 does not register, and their checkpoints are never loaded by
    an evaluation path -- C5's identity refuses that directory; this keeps the training side from writing across it.
    """
    candidate = Path(path)
    fenced = FENCED_TIMING_DIRNAME in candidate.parts
    if timing and not fenced:
        raise ValueError(
            f"{candidate} is not under {FENCED_TIMING_DIRNAME}/: a timing run is FENCED, and its checkpoints never "
            "leave that directory"
        )
    if not timing and fenced:
        raise ValueError(
            f"{candidate} lies under {FENCED_TIMING_DIRNAME}/, which holds unregistered timing runs only; a "
            "registered checkpoint is never written there and nothing there is ever evaluated"
        )
    return candidate


# ======================================================================================================================
# Digests, and the code's own commit
# ======================================================================================================================


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    """``git`` run in THIS module's directory, never the process cwd: the driver's cwd is the MAIN tree."""
    return subprocess.run(
        ["git", *args], cwd=str(Path(__file__).resolve().parent), capture_output=True, text=True, timeout=30,
        check=False,
    )


def _code_commit() -> str:
    """The commit of the tree this module was loaded from (``offline.dt_gate.runtime_provenance`` asks the cwd)."""
    result = _git("rev-parse", "HEAD")
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _code_tree_dirty() -> bool | None:
    """Was that tree modified?  ``None`` if git could not say -- never read as clean."""
    result = _git("status", "--porcelain")
    return None if result.returncode != 0 else bool(result.stdout.strip())


def _runtime(device: torch.device) -> dict[str, Any]:
    """The library and device state that determines float reduction order.  No wall-clock value."""
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_num_threads": int(torch.get_num_threads()),
        "numpy_version": str(np.__version__),
        "python_version": platform.python_version(),
        "git_dirty": _code_tree_dirty(),
    }


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ======================================================================================================================
# The source
# ======================================================================================================================


def _load_weights_only(path: str | Path) -> dict[str, Any]:
    """``torch.load`` weights-only with ONE allowlisted class, ``TorchVersion`` (``BRIEF_41`` §0.1)."""
    from torch.torch_version import TorchVersion

    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(Path(path), map_location="cpu", weights_only=True)


def load_source(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    """The source checkpoint, digest-checked BEFORE it is read, then checked against A20(a)'s subject and P5.2's recipe.

    Refused: a digest other than *expected_sha256*; a top-level format other than ``spatial-dt-checkpoint/1.0``;
    ``spatial_mixing`` on; a head count other than 4; another scenario; no normalisation or no statistics; an id set
    that the scale, the targets and the statistics do not all cover; a mask that is not the identity; and a recorded
    recipe -- steps, batch, learning rate, weight decay, clip, warm-up, method, heads -- other than the one this module
    imports (A24(b): "the recipe unchanged where P5.2 has one").
    """
    from offline.spatial_mixing import GRAD_CLIP, JOINT_BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY
    from offline.tier_sweep import warmup_for

    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"{source_path} is absent: A20(a)'s checkpoints are read from output/p5_2/checkpoints")
    digest = _sha256_file(source_path)
    if digest != str(expected_sha256):
        raise ValueError(
            f"{source_path}: sha256 {digest} is not the pinned {expected_sha256}. A20(a) registers the subject BY "
            "DIGEST; a different file under the same name is a different subject, and nothing is read from it"
        )
    payload = _load_weights_only(source_path)
    label = str(source_path)

    version = str(payload.get("format_version"))
    if version != SPATIAL_FORMAT_VERSION:
        raise ValueError(f"{label}: format {version!r} is not {SPATIAL_FORMAT_VERSION!r}; the subject is stored in it")
    config = dict(payload.get("config") or {})
    if bool(config.get("spatial_mixing")):
        raise ValueError(f"{label}: spatial_mixing is on; A20(a)'s subject is the NON-mixing control {METHOD}")
    if int(config.get("n_head", -1)) != N_HEAD:
        raise ValueError(f"{label}: config n_head {config.get('n_head')!r}, not {N_HEAD} ({METHOD})")
    scenario = str(payload.get("scenario_id"))
    if scenario != SCENARIO_ID:
        raise ValueError(f"{label}: scenario_id {scenario!r}, not {SCENARIO_ID!r}")
    if not bool(payload.get("normalise")):
        raise ValueError(
            f"{label}: the source was not trained on normalised states (normalise {payload.get('normalise')!r}); "
            "A24(b) freezes the statistics it WAS trained on"
        )
    if payload.get("stats") is None:
        raise ValueError(f"{label}: the source carries no statistics; there is nothing to freeze")

    ids = [str(ix) for ix in (payload.get("intersection_ids") or [])]
    if not ids or len(set(ids)) != len(ids) or int(config.get("n_nodes", -1)) != len(ids):
        raise ValueError(
            f"{label}: the recorded intersection order {ids[:4]} is empty, repeats an id, or disagrees with "
            f"config n_nodes {config.get('n_nodes')!r}"
        )
    mask = np.asarray(payload.get("spatial_mask"), dtype=np.bool_)
    if mask.shape != (len(ids), len(ids)) or not bool((mask == np.eye(len(ids), dtype=np.bool_)).all()):
        raise ValueError(
            f"{label}: the recorded spatial mask is not the {len(ids)} x {len(ids)} identity, so intersections could "
            "attend to each other; that is not the registered subject"
        )
    for name, keys in (
        ("rtg_scale", payload.get("rtg_scale") or {}),
        ("target_rtg", payload.get("target_rtg") or {}),
        ("stats.state_mean", payload["stats"].get("state_mean", {}).get(scenario, {})),
        ("stats.state_std", payload["stats"].get("state_std", {}).get(scenario, {})),
    ):
        if sorted(str(key) for key in keys) != sorted(ids):
            raise ValueError(f"{label}: {name} does not cover exactly the recorded intersections")

    provenance = dict(payload.get("provenance") or {})
    expected = {
        "gradient_steps": SOURCE_GRADIENT_STEPS,
        "batch_size": JOINT_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "warmup_steps": warmup_for(SOURCE_GRADIENT_STEPS),
        "method": METHOD,
        "n_head": N_HEAD,
    }
    for field, value in expected.items():
        recorded = provenance.get(field)
        if recorded != value:
            raise ValueError(
                f"{label}: the source's provenance {field} is {recorded!r}, not {value!r}. A24(b) fine-tunes with "
                "P5.2's recipe unchanged, imported from where offline.tier_sweep imports it and asserted equal to "
                "what the source records"
            )
    if "seed" not in provenance:
        raise ValueError(f"{label}: the source's provenance records no seed")
    return payload


# ======================================================================================================================
# The targets
# ======================================================================================================================


@dataclass(frozen=True)
class KTargets:
    """The k-budget's sixteen targets, read from the digest-pinned calibration artifact, keyed by id."""

    k: int
    targets: dict[str, float]
    role: str
    calibration_sha256: str


def load_k_targets(k: int, *, source: Mapping[str, Any], data_dir: str | Path | None = None) -> KTargets:
    """``per_intersection[ix].budgets["k{k}"].target`` for every id of *source*, after the artifact's sha256.

    The artifact must describe THIS subject: its id set and scenario are the source's, and its recorded ``rtg_scale``
    of every intersection equals the source's -- the targets were fixed on that scale.  Each budget must carry ``k``,
    the prefix ``[201, 200 + k]``, Rule B, the mean, and the role A24 gives that k.
    """
    from offline.transfer_curve import P7_3D_CALIBRATION_SHA256

    wanted = _require_k(k)
    path = (Path(data_dir) if data_dir is not None else _DATA_DIR) / CALIBRATION_NAME
    if not path.is_file():
        raise ValueError(f"{path} is absent; P7.3d C3a committed it and the k targets are read from it")
    digest = _sha256_file(path)
    if digest != P7_3D_CALIBRATION_SHA256:
        raise ValueError(
            f"{path}: sha256 {digest} is not the pinned {P7_3D_CALIBRATION_SHA256}. The k targets are registered "
            "quantities (A24(b)) and are read from THAT file and no other"
        )
    artifact = json.loads(path.read_bytes())

    ids = [str(ix) for ix in source["intersection_ids"]]
    if sorted(str(ix) for ix in artifact["intersection_ids"]) != sorted(ids):
        raise ValueError(f"{path}: the artifact's intersections are not the source's")
    if str(artifact.get("scenario_key")) != str(source["scenario_id"]):
        raise ValueError(f"{path}: scenario {artifact.get('scenario_key')!r}, not the source's {source['scenario_id']!r}")
    key = f"k{wanted}"
    role = ROLE_BY_K[wanted]
    targets: dict[str, float] = {}
    for ix in ids:
        entry = artifact["per_intersection"][ix]
        recorded_scale = float(entry["rtg_scale"])
        own_scale = float(source["rtg_scale"][ix])
        if own_scale != recorded_scale:
            raise ValueError(
                f"the source's rtg_scale of {ix!r} is {own_scale!r} but the calibration artifact records "
                f"{recorded_scale!r}: the artifact describes another subject, and its targets were fixed on that "
                "subject's scale"
            )
        budget = entry["budgets"][key]
        facts = {
            "k": (int(budget["k"]), wanted),
            "draw_ids": ([int(d) for d in budget["draw_ids"]], [FIRST_DRAW, FIRST_DRAW + wanted - 1]),
            "role": (str(budget["role"]), role),
            "rule": (str(budget["rule"]), "B"),
            "statistic": (str(budget["statistic"]), "mean"),
        }
        for name, (found, expected) in facts.items():
            if found != expected:
                raise ValueError(f"{path}: intersection {ix!r} {key} records {name} {found!r}, not {expected!r}")
        targets[ix] = float(budget["target"])
    return KTargets(k=wanted, targets=targets, role=role, calibration_sha256=digest)


# ======================================================================================================================
# The corpus and its prefix
# ======================================================================================================================


@dataclass(frozen=True)
class CorpusPrefix:
    """The first k draws of a VERIFIED corpus, and the digest of the sums that verified it."""

    corpus_dir: str
    draw_ids: tuple[int, ...]
    sums_sha256: str


def _verified_entries(corpus: Path) -> tuple[dict[str, str], str]:
    """``SHA256SUMS`` parsed and re-verified entry by entry; the directory must hold exactly what it lists."""
    sums = corpus / SUMS_NAME
    if not sums.is_file():
        raise ValueError(
            f"{corpus} has no {SUMS_NAME}: the corpus driver writes it after the last episode and G3 verifies it; a "
            "corpus without one is not the verified corpus"
        )
    entries: dict[str, str] = {}
    for number, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), start=1):
        match = _SUMS_LINE.fullmatch(line)
        if match is None or "/" in match.group(2) or match.group(2) in entries:
            raise ValueError(f"{sums}:{number}: not one sha256sum line for a file of this directory: {line!r}")
        entries[match.group(2)] = match.group(1)
    for path in sorted(corpus.iterdir()):
        if path.name != SUMS_NAME and path.name not in entries:
            raise ValueError(
                f"{path} is not listed in {SUMS_NAME}; a verified corpus holds exactly the files its sums list"
            )
    for name, listed in sorted(entries.items()):
        path = corpus / name
        if not path.is_file():
            raise ValueError(f"{path} is listed in {SUMS_NAME} but absent")
        actual = _sha256_file(path)
        if actual != listed:
            raise ValueError(
                f"{path} does not match {SUMS_NAME} (sha256 {actual}, listed {listed}); the corpus changed after it "
                "was verified"
            )
    return entries, _sha256_file(sums)


def select_prefix(corpus_dir: str | Path, k: int, *, scenario_id: str) -> CorpusPrefix:
    """The first *k* draws of the corpus, read from its manifest after its sums are re-verified.

    The manifest must be C6 v1.1 in *scenario_id* and list exactly the ``.npz`` files the sums list; its draws must be
    A24's whole band 201-300, one episode each.  Any other set is refused, a missing draw BY NAME.
    """
    wanted = _require_k(k)
    corpus = Path(corpus_dir)
    if not corpus.is_dir():
        raise ValueError(f"{corpus} is not a directory; the fine-tune reads the verified grid4x4 SUMO corpus")
    entries, sums_digest = _verified_entries(corpus)
    if MANIFEST_NAME not in entries:
        raise ValueError(f"{corpus}: {SUMS_NAME} does not list {MANIFEST_NAME}")
    manifest = json.loads((corpus / MANIFEST_NAME).read_text(encoding="utf-8"))
    version = str(manifest.get("format_version"))
    if version != CORPUS_FORMAT_VERSION:
        raise ValueError(f"{corpus}: manifest format {version!r}, not C6 v{CORPUS_FORMAT_VERSION}")
    recorded = str(manifest.get("run_metadata", {}).get("scenario_id"))
    if recorded != str(scenario_id):
        raise ValueError(f"{corpus}: the corpus records scenario_id {recorded!r}, not {str(scenario_id)!r}")
    episodes = list(manifest.get("episodes", []))
    listed = sorted(str(entry["filename"]) for entry in episodes)
    if listed != sorted(name for name in entries if name.endswith(".npz")):
        raise ValueError(f"{corpus}: the manifest's episodes are not the .npz files {SUMS_NAME} lists")
    counts = Counter(int(entry["flow_draw"]) for entry in episodes)
    doubled = sorted(draw for draw, count in counts.items() if count > 1)
    if doubled:
        raise ValueError(
            f"{corpus}: draw {doubled[0]} is logged {counts[doubled[0]]} times; A24(b)'s corpus holds one episode "
            "per draw"
        )
    outside = sorted(draw for draw in counts if draw not in CORPUS_DRAWS)
    if outside:
        raise ValueError(f"{corpus}: draw(s) {outside} outside A24's band 201-300")
    missing = sorted(set(CORPUS_DRAWS) - set(counts))
    if missing:
        raise ValueError(
            f"{corpus} lacks draw(s) {missing}; A24(b)'s corpus is the whole band 201-300, one episode per draw, and "
            "the fine-tune reads its first k from the manifest"
        )
    prefix = tuple(sorted(counts)[:wanted])
    if prefix != tuple(range(FIRST_DRAW, FIRST_DRAW + wanted)):
        raise ValueError(f"{corpus}: the first {wanted} draws are {list(prefix)[:6]}, not 201-{200 + wanted}")
    return CorpusPrefix(corpus_dir=str(corpus.resolve()), draw_ids=prefix, sums_sha256=sums_digest)


# ======================================================================================================================
# The windows
# ======================================================================================================================


@dataclass(frozen=True)
class Windows:
    """The joint windows of a prefix, in the checkpoint's node order, normalised with the checkpoint's statistics."""

    dataset: Any
    index: Any
    stacked: dict[str, torch.Tensor]
    stats: Any


def build_windows(corpus_dir: str | Path, draw_ids: Sequence[int], source: Mapping[str, Any]) -> Windows:
    """``TrajectoryWindowDataset`` with the SOURCE's statistics supplied, joint in the SOURCE's node order.

    The statistics are handed in through ``stats=`` (``offline/dataset.py:663-669``), so nothing is fitted; the
    dataset must have adopted that very object.  The one ``(state_dim, n_actions)`` group is the config's; the loaded
    episodes are exactly *draw_ids*, each carrying exactly the source's ids -- a corpus of another id set is refused,
    never re-keyed by position -- and ``build_joint_index`` puts every column in the source's order by ``item_meta``.
    The largest step must fit the source's ``max_ep_len``.
    """
    from offline.dataset import NormalizationStats, TrajectoryWindowDataset
    from offline.joint_windows import build_joint_index, stack_joint

    config = dict(source["config"])
    ids = [str(ix) for ix in source["intersection_ids"]]
    stats = NormalizationStats.from_json_obj(source["stats"])
    requested = [int(draw) for draw in draw_ids]
    dataset = TrajectoryWindowDataset(
        [Path(corpus_dir)],
        context_length=int(config["context_length"]),
        split="train",
        draw_ids=requested,
        stats=stats,
        normalize=True,
    )
    if dataset.stats is not stats:
        raise RuntimeError("the window dataset did not adopt the source's statistics; it would have fitted its own")
    groups = sorted(dataset.groups)
    expected_group = (int(config["state_dim"]), int(config["n_actions"]))
    if groups != [expected_group]:
        raise ValueError(f"the corpus's (state_dim, n_actions) groups are {groups}, not the source's {[expected_group]}")
    loaded = sorted(record.flow_draw for record in dataset.episode_records)
    if loaded != sorted(requested):
        raise ValueError(f"the loaded episodes' draws {loaded[:6]} are not the prefix {sorted(requested)[:6]}")
    for record in dataset.episode_records:
        if record.scenario_id != str(source["scenario_id"]):
            raise ValueError(f"{record.episode_file}: scenario {record.scenario_id!r}, not the source's")
        present = set(record.ix_ids)
        missing = sorted(set(ids) - present)
        unknown = sorted(present - set(ids))
        if missing or unknown:
            raise ValueError(
                f"{record.episode_file}: the episode's intersections differ from the checkpoint's (missing {missing}, "
                f"unknown {unknown}); the joint windows are keyed by the checkpoint's ids, and a corpus of another id "
                "set is refused, never re-keyed by position"
            )
    index = build_joint_index(dataset, ids)
    if list(index.node_ids) != ids:
        raise RuntimeError("the joint index is not in the checkpoint's node order")
    stacked = stack_joint(dataset, index)
    last = int(stacked["timestep"].max())
    if last >= int(config["max_ep_len"]):
        raise ValueError(f"the corpus reaches step {last}, beyond the source's max_ep_len {config['max_ep_len']}")
    return Windows(dataset=dataset, index=index, stacked=stacked, stats=stats)


# ======================================================================================================================
# The model, the optimiser, and the loop's two per-step seams
# ======================================================================================================================


def build_model(config: Any, init: str, source_model: Mapping[str, torch.Tensor], device: Any) -> Any:
    """``SpatialDecisionTransformer(config)`` on *device*; for ``init == "source"`` the source's weights, strictly.

    The construction is the same in both arms, so after it the torch RNG is in the same state in both; the switch is
    the one ``load_state_dict`` call and nothing else.
    """
    from agent.SpatialDTAgent import SpatialDecisionTransformer

    if init not in INITS:
        raise ValueError(f"init {init!r} is not one of {list(INITS)}")
    model = SpatialDecisionTransformer(config).to(device)
    if init == INIT_SOURCE:
        model.load_state_dict(source_model, strict=True)
    return model


def make_optimiser(parameters: Iterable[torch.nn.Parameter], budget: int) -> tuple[Any, Any, int]:
    """A FRESH AdamW, and ``train_tier_dt``'s warm-up schedule for a budget of *budget* steps.

    ``LEARNING_RATE`` and ``WEIGHT_DECAY`` come from ``offline.spatial_mixing``, where ``train_tier_dt`` takes them;
    the warm-up is ``offline.tier_sweep.warmup_for`` -- ``min(1000, max(1, B // 2))``: 500 at B = 1,000, 1,000 at
    4,000 and 16,000 -- and the multiplier ``lr_multiplier``.  Nothing is restated.
    """
    from offline.spatial_mixing import LEARNING_RATE, WEIGHT_DECAY
    from offline.tier_sweep import lr_multiplier, warmup_for

    warmup = warmup_for(int(budget))
    optimiser = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: lr_multiplier(step, warmup))
    return optimiser, schedule, warmup


def draw_rows(generator: np.random.Generator, count: int, batch_size: int) -> torch.Tensor:
    """THE SAMPLER: *batch_size* joint instants drawn uniformly with replacement -- ``train_tier_dt``'s draw."""
    return torch.from_numpy(generator.integers(0, int(count), size=int(batch_size)).astype(np.int64))


def joint_batch(
    stacked: Mapping[str, torch.Tensor], rows: torch.Tensor, *, scale: torch.Tensor, device: Any
) -> dict[str, torch.Tensor]:
    """THE BATCH the model sees: the drawn instants' windows, the RTG divided per node by the frozen scale."""
    selected = stacked["member_index"][rows]
    return {
        "rtg": stacked["rtg"][selected].to(device) / scale,
        "state": stacked["state"][selected].to(device),
        "action": stacked["action"][selected].to(device),
        "timestep": stacked["timestep"][selected].to(device),
        "attention_mask": stacked["attention_mask"][selected].to(device),
        "avail_mask": stacked["avail_mask"][selected].to(device),
    }


# ======================================================================================================================
# Prepare, train, write
# ======================================================================================================================


@dataclass
class PreparedRun:
    """Everything validated and built before the first optimizer step; nothing has been written."""

    init: str
    k: int
    budget: int
    seed: int
    device: torch.device
    destination: Path
    source_path: Path
    source_sha256: str
    source: dict[str, Any]
    targets: KTargets
    prefix: CorpusPrefix
    windows: Windows
    config: Any
    model: Any
    scale: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class TrainingOutcome:
    """What the loop did: its step count (COUNTED), warm-up, per-step losses and wall seconds."""

    steps: int
    warmup: int
    losses: tuple[float, ...]
    seconds: float


@dataclass(frozen=True)
class FineTuneResult:
    """One written checkpoint: where, its sha256, and what the loop did."""

    destination: Path
    sha256: str
    steps: int
    warmup: int
    losses: tuple[float, ...]
    seconds: float


def _validate_request(*, init: Any, k: Any, budget: Any, seed: Any, destination: str | Path) -> Path:
    if init not in INITS:
        raise ValueError(
            f"init {init!r} is not one of {list(INITS)}: the fine-tune loads the source's weights, the from-scratch "
            "control (A24(c)(ii)) does not, and nothing else is registered"
        )
    _require_k(k)
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or int(budget) < 1:
        raise ValueError(f"budget {budget!r} is not a positive step count")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed {seed!r} is not an integer")
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise ValueError(
            f"{target} already exists; a fine-tuned checkpoint is written once (A24(b)) and never overwritten -- a "
            "person moves it aside"
        )
    if not target.parent.is_dir():
        raise ValueError(f"{target.parent} does not exist; nothing is created here")
    return target


def prepare_fine_tune(
    *,
    source_path: str | Path,
    source_sha256: str,
    corpus_dir: str | Path,
    k: int,
    budget: int,
    seed: int,
    init: str,
    device: Any,
    destination: str | Path,
    data_dir: str | Path | None = None,
) -> PreparedRun:
    """Steps 1-6 of the route: every check, the windows, the seed and the model.  Writes nothing."""
    from agent.SpatialDTAgent import SpatialDTConfig
    from agent.utils.utils import Utils
    from offline.rtg_calibration import assert_probe_draws_disjoint
    from offline.tier_sweep import assert_process_regime
    from offline.transfer_calibration import HELD_OUT_DRAWS

    target = _validate_request(init=init, k=k, budget=budget, seed=seed, destination=destination)
    source = load_source(source_path, expected_sha256=source_sha256)
    targets = load_k_targets(int(k), source=source, data_dir=data_dir)
    prefix = select_prefix(corpus_dir, int(k), scenario_id=str(source["scenario_id"]))
    try:
        assert_probe_draws_disjoint(
            prefix.draw_ids,
            training_draw_ids=[int(d) for d in source["stats"]["draw_ids"]],
            held_out_draws=HELD_OUT_DRAWS,
        )
    except ValueError as exc:
        raise ValueError(
            f"the fine-tune prefix {prefix.draw_ids[0]}-{prefix.draw_ids[-1]} is not disjoint from the source's "
            f"training draws and the held-out pool: {exc}"
        ) from exc
    windows = build_windows(corpus_dir, prefix.draw_ids, source)
    # The registered regime is the subject's: CUDA, deterministic algorithms OFF (A24(b), as P5.2's).
    assert_process_regime(False)

    torch_device = torch.device(device)
    Utils.seed_everything(int(seed), seed_python_random=False)
    config = SpatialDTConfig.from_json_obj(dict(source["config"]))
    model = build_model(config, str(init), source["model"], torch_device)

    ids = [str(ix) for ix in source["intersection_ids"]]
    scale = (
        torch.tensor([float(source["rtg_scale"][ix]) for ix in ids], dtype=torch.float32)
        .view(1, len(ids), 1, 1)
        .to(torch_device)
    )
    mask = torch.from_numpy(np.asarray(source["spatial_mask"], dtype=np.bool_)).to(torch_device)
    return PreparedRun(
        init=str(init),
        k=int(k),
        budget=int(budget),
        seed=int(seed),
        device=torch_device,
        destination=target,
        source_path=Path(source_path),
        source_sha256=str(source_sha256),
        source=source,
        targets=targets,
        prefix=prefix,
        windows=windows,
        config=config,
        model=model,
        scale=scale,
        mask=mask,
    )


def train_prepared(prepared: PreparedRun, *, log_every: int = 0) -> TrainingOutcome:
    """Step 7: exactly B steps of ``train_tier_dt``'s loop on the prepared model; the steps are COUNTED."""
    from agent.DTAgent import action_loss
    from offline.spatial_mixing import GRAD_CLIP, JOINT_BATCH_SIZE

    model = prepared.model
    optimiser, schedule, warmup = make_optimiser(model.parameters(), prepared.budget)
    stacked = prepared.windows.stacked
    count = int(stacked["member_index"].shape[0])
    if count < 1:
        raise ValueError("the joint index is empty")
    generator = np.random.default_rng(int(prepared.seed))
    losses: list[float] = []
    steps = 0
    model.train()
    started = time.perf_counter()
    for _ in range(prepared.budget):
        rows = draw_rows(generator, count, JOINT_BATCH_SIZE)
        batch = joint_batch(stacked, rows, scale=prepared.scale, device=prepared.device)
        logits = model(
            batch["rtg"],
            batch["state"],
            batch["action"],
            batch["timestep"],
            prepared.mask,
            batch["attention_mask"],
            batch["avail_mask"],
        )
        loss = action_loss(logits, batch["action"])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimiser.step()
        schedule.step()
        losses.append(float(loss.detach()))
        steps += 1
        if log_every and steps % int(log_every) == 0:
            print(f"  step {steps}/{prepared.budget} ({time.perf_counter() - started:.1f} s)", flush=True)
    return TrainingOutcome(
        steps=steps, warmup=int(warmup), losses=tuple(losses), seconds=float(time.perf_counter() - started)
    )


def payload_for(prepared: PreparedRun, outcome: TrainingOutcome) -> dict[str, Any]:
    """Step 8's payload: the SOURCE's frozen parts, the same objects read from it; the new weights, targets, provenance."""
    from offline.spatial_mixing import GRAD_CLIP, JOINT_BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY

    source = prepared.source
    ids = [str(ix) for ix in source["intersection_ids"]]
    source_provenance = source["provenance"]
    provenance: dict[str, Any] = {
        "gradient_steps": int(outcome.steps),
        "batch_size": int(JOINT_BATCH_SIZE),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "warmup_steps": int(outcome.warmup),
        "sampler": SAMPLER,
        "seed": int(prepared.seed),
        "seeding": SEEDING_CALL,
        "method": METHOD,
        "n_head": N_HEAD,
        "spatial_mixing": False,
        "device": str(prepared.device),
        "deterministic": False,
        "git_commit": _code_commit(),
        "runtime": _runtime(prepared.device),
        "few_shot": {
            "format_version": FEW_SHOT_FORMAT_VERSION,
            "init": prepared.init,
            "k": int(prepared.k),
            "draw_ids": list(prepared.prefix.draw_ids),
            "target_rule": TARGET_RULE,
            "target_role": prepared.targets.role,
            "source_path": str(prepared.source_path.resolve()),
            "source_sha256": prepared.source_sha256,
            "source_seed": int(source_provenance["seed"]),
            "source_gradient_steps": int(source_provenance["gradient_steps"]),
            "source_provenance": source_provenance,
            "source_target_rtg": source["target_rtg"],
            "corpus_dir": prepared.prefix.corpus_dir,
            "corpus_sha256sums_sha256": prepared.prefix.sums_sha256,
            "calibration_sha256": prepared.targets.calibration_sha256,
        },
    }
    return {
        "format_version": source["format_version"],
        "config": source["config"],
        "model": {key: value.detach().cpu() for key, value in prepared.model.state_dict().items()},
        "target_rtg": {ix: prepared.targets.targets[ix] for ix in ids},
        "rtg_scale": source["rtg_scale"],
        "normalise": source["normalise"],
        "scenario_id": source["scenario_id"],
        "stats": source["stats"],
        "intersection_ids": source["intersection_ids"],
        "spatial_mask": source["spatial_mask"],
        "provenance": provenance,
    }


def _link_exclusive(data: bytes, destination: Path) -> str:
    """Write *data* to a sibling ``.partial`` file, fsync it, then ``os.link`` it to *destination*; return its sha256.

    ``os.link`` fails if the destination exists, so the exclusive step and the publication are one system call; the
    partial file is created exclusively and removed on every path, so a failure leaves nothing behind.
    """
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"{destination} already exists; it is written once and never replaced")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"{destination.parent} does not exist; nothing is created here")
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        with partial.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def write_payload_exclusive(payload: Mapping[str, Any], destination: str | Path) -> str:
    """``torch.save`` *payload* INTO MEMORY, then write those bytes to *destination*, which must not exist.

    ⚠️ **Why memory first.**  ``torch.save`` to a PATH names the zip archive inside the file after the file (``a.pt``
    -> ``a/data.pkl``), so a payload saved straight to disk carries its own file name -- here the partial file's,
    process id included -- and two identical payloads could never agree byte for byte.  Measured 2026-09-25 on this
    machine's torch 2.11: one payload saved to ``d1/a.pt`` and ``d1/b.pt`` differs, to ``d1/a.pt`` and ``d2/a.pt``
    agrees, and to an in-memory buffer twice agrees (the archive is then always ``archive/``, and the recorded
    ``serialization_id`` is derived from the content).  So the file's sha256 is a function of the payload alone,
    which is what G5's repeat compares and what T-cpu-determinism pins.
    """
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    return _link_exclusive(buffer.getvalue(), Path(destination))


def fine_tune(
    *,
    source_path: str | Path,
    source_sha256: str,
    corpus_dir: str | Path,
    k: int,
    budget: int,
    seed: int,
    init: str,
    device: Any,
    destination: str | Path,
    data_dir: str | Path | None = None,
    log_every: int = 0,
) -> FineTuneResult:
    """The whole route: prepare, exactly B steps (counted, then asserted), then ONE exclusive write."""
    prepared = prepare_fine_tune(
        source_path=source_path,
        source_sha256=source_sha256,
        corpus_dir=corpus_dir,
        k=k,
        budget=budget,
        seed=seed,
        init=init,
        device=device,
        destination=destination,
        data_dir=data_dir,
    )
    outcome = train_prepared(prepared, log_every=log_every)
    if outcome.steps != prepared.budget:
        raise RuntimeError(
            f"the loop ran {outcome.steps} optimizer steps, not the declared {prepared.budget}; nothing is written"
        )
    digest = write_payload_exclusive(payload_for(prepared, outcome), prepared.destination)
    return FineTuneResult(
        destination=prepared.destination,
        sha256=digest,
        steps=outcome.steps,
        warmup=outcome.warmup,
        losses=outcome.losses,
        seconds=outcome.seconds,
    )


def write_run_record(spec: RunSpec, result: FineTuneResult, *, output_root: str | Path, device: Any) -> Path:
    """The run's seconds, losses and digest, beside the checkpoints -- written once, like the checkpoint."""
    path = run_record_path(output_root, spec)
    record = {
        "format_version": RUN_RECORD_FORMAT_VERSION,
        "run": spec.name,
        "subject": spec.subject,
        "init": spec.init,
        "k": int(spec.k),
        "budget": int(spec.budget),
        "seed": int(spec.seed),
        "checkpoint": str(Path(TRAINING_DIRNAME) / CHECKPOINTS_DIRNAME / f"{spec.name}.pt"),
        "checkpoint_sha256": result.sha256,
        "steps": int(result.steps),
        "warmup_steps": int(result.warmup),
        "loop_seconds": float(result.seconds),
        "ms_per_step": float(result.seconds) / int(result.steps) * 1000.0,
        "final_loss": float(result.losses[-1]),
        "losses": [float(value) for value in result.losses],
        "device": str(device),
        "git_commit": _code_commit(),
        "written_utc": _utc_now(),
    }
    _link_exclusive((json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"), path)
    return path


# ======================================================================================================================
# C4 -- the training driver's Python half: resume, attempts, manifest, record, G5's timing, the inputs
# ======================================================================================================================
#
# ``offline/campaigns/p7_3c_finetune.sh`` decides nothing in bash that could train a checkpoint twice, overwrite one or
# let a fenced one through: every such decision is made here (plan section 8).

#: The training record the coordinator commits on ``main`` at G7 as ``docs/data/p7_3c_finetune.json`` (Amendment A Q9).
RECORD_FORMAT_VERSION = "p7.3c-finetune-record/1.0"
RECORD_NAME = "p7_3c_finetune.json"
#: The manifest of the thirty checkpoints, under the output root, in ``sha256sum``'s format (``BRIEF_41`` C4).
MANIFEST_FILENAME = "SHA256SUMS_p7_3c_finetune.txt"
ATTEMPTS_DIRNAME = "attempts"
#: G5's records: one per timed run (slot), one for the k = 100 build, and the summary the driver reads C from.
TIMING_FORMAT_VERSION = "p7.3c-finetune-timing/1.0"
TIMING_SLOT_FORMAT_VERSION = "p7.3c-finetune-timing-slot/1.0"
TIMING_BUILD_FORMAT_VERSION = "p7.3c-finetune-timing-build/1.0"
TIMING_RECORD_NAME = "timing.json"
#: G5's phases, fixed before any measurement (plan section 8): alone, two at once, three at once, then ONE same-seed
#: repeat of the single run, alone.  The concurrency C of a phase is its number of slots.
TIMING_PHASES: dict[str, tuple[str, ...]] = {
    "alone": ("alone",),
    "pair": ("pair1", "pair2"),
    "triple": ("triple1", "triple2", "triple3"),
}
TIMING_SLOTS: tuple[str, ...] = ("alone", "pair1", "pair2", "triple1", "triple2", "triple3", "repeat")
#: The device and the rule's cap: C in {1, 2, 3} with the largest aggregate throughput whose measured device peak is
#: <= 80 % of 16,303 MiB (this machine's RTX 5080 Laptop GPU); a tie goes to the smaller C.
DEVICE_MIB = 16303.0
DEVICE_FRACTION = 0.8
#: Amendment A, Q12: a k = 100 window build over five minutes permits one process per k.
Q12_BUILD_SECONDS = 300.0
#: G3's gate record, as ``offline.transfer_calibration``'s corpus-gate writes it.
GATE_FORMAT_VERSION = "p7.3c-corpus-gate/1.0"
_STAMP = re.compile(r"\d{8}T\d{6}Z")


def _pin_for(seed: int, pins: Mapping[int, str] | None) -> str:
    """A20(a)'s pin for *seed* (``offline.transfer_calibration``), or the caller's -- the tests' synthetic sources."""
    from offline.transfer_calibration import GRID4X4_CHECKPOINT_SHA256

    table = GRID4X4_CHECKPOINT_SHA256 if pins is None else pins
    if int(seed) not in table:
        raise ValueError(f"no pinned source digest for seed {seed}")
    return str(table[int(seed)])


def _read_json_record(path: Path, version: str, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{path} is absent: {what}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("format_version") != version:
        raise ValueError(f"{path}: format {record.get('format_version')!r}, not {version!r}")
    return record


def validate_checkpoint(
    spec: RunSpec, *, output_root: str | Path, pins: Mapping[int, str] | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, bool]:
    """Every check a written checkpoint of *spec* must pass, as booleans: the record carries them, resume refuses on any.

    The frozen parts are compared with the SOURCE at its pin, read again here (never with a copy the payload carries);
    the budget, k, the draw ids, the switch, the seed and the source's digest with *spec*; the targets with the pinned
    calibration artifact's ``k{k}``; the recipe with the constants the trainer imports.  An unreadable file is refused.
    """
    from offline.spatial_mixing import GRAD_CLIP, JOINT_BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY
    from offline.tier_sweep import warmup_for

    path = assert_fence(registered_destination(output_root, spec), timing=False)
    pin = _pin_for(spec.seed, pins)
    source = load_source(registered_source_path(output_root, spec.seed), expected_sha256=pin)
    try:
        payload = _load_weights_only(path)
    except Exception as exc:  # any failure to read is the same finding: the file is not a checkpoint of this run
        raise ValueError(
            f"{path} exists but cannot be read ({type(exc).__name__}); it is never overwritten -- a person moves it "
            "aside"
        ) from exc
    targets = load_k_targets(spec.k, source=source, data_dir=data_dir).targets
    provenance = dict(payload.get("provenance") or {})
    block = dict(provenance.get("few_shot") or {})
    recipe = (
        provenance.get("learning_rate"), provenance.get("weight_decay"), provenance.get("grad_clip"),
        provenance.get("batch_size"), provenance.get("warmup_steps"),
    )
    return {
        "format_version": payload.get("format_version") == SPATIAL_FORMAT_VERSION,
        "few_shot_format": block.get("format_version") == FEW_SHOT_FORMAT_VERSION,
        "config": payload.get("config") == source["config"],
        "stats": payload.get("stats") == source["stats"],
        "rtg_scale": payload.get("rtg_scale") == source["rtg_scale"],
        "intersection_ids": payload.get("intersection_ids") == source["intersection_ids"],
        "spatial_mask": payload.get("spatial_mask") == source["spatial_mask"],
        "normalise": payload.get("normalise") == source["normalise"],
        "scenario_id": payload.get("scenario_id") == source["scenario_id"],
        "source_sha256": block.get("source_sha256") == pin,
        "source_seed": block.get("source_seed") == spec.seed and provenance.get("seed") == spec.seed,
        "budget": provenance.get("gradient_steps") == spec.budget,
        "k": block.get("k") == spec.k,
        "draw_ids": block.get("draw_ids") == list(range(FIRST_DRAW, FIRST_DRAW + spec.k)),
        "init": block.get("init") == spec.init,
        "targets": payload.get("target_rtg") == targets,
        "recipe": recipe == (LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP, JOINT_BATCH_SIZE, warmup_for(spec.budget)),
    }


def resume_decision(
    spec: RunSpec, *, output_root: str | Path, pins: Mapping[int, str] | None = None,
    data_dir: str | Path | None = None,
) -> str:
    """``"train"`` if *spec*'s checkpoint is absent, ``"skip"`` if it exists AND validates; refused otherwise.

    Decided here, never by a ``[ -f ]`` in the driver (plan section 8): a checkpoint that exists and does not validate
    is never overwritten and never re-trained -- A24(b) allows a re-run only when the checkpoint was never written.
    """
    path = registered_destination(output_root, spec)
    if not path.exists() and not path.is_symlink():
        return "train"
    checks = validate_checkpoint(spec, output_root=output_root, pins=pins, data_dir=data_dir)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"{path} exists but does not validate (failed: {failed}); it is never overwritten -- a person moves it "
            "aside, and a run is re-trained only if its checkpoint was never written (A24(b))"
        )
    return "skip"


def attempts_of(spec: RunSpec, *, output_root: str | Path) -> tuple[int, ...]:
    """The attempt numbers recorded on disk for *spec* (``attempts/<name>.<n>``), in order."""
    directory = training_root(output_root) / ATTEMPTS_DIRNAME
    if not directory.is_dir():
        return ()
    prefix = f"{spec.name}."
    numbers = [
        int(path.name[len(prefix):])
        for path in directory.iterdir()
        if path.name.startswith(prefix) and path.name[len(prefix):].isdigit()
    ]
    return tuple(sorted(numbers))


def next_attempt(spec: RunSpec, *, output_root: str | Path) -> int:
    """Write the next attempt marker for *spec*, exclusively, BEFORE its training starts; return its number.

    A marker with no checkpoint after it is a re-run of an infrastructure failure, counted on disk (plan section 8).
    """
    directory = training_root(output_root) / ATTEMPTS_DIRNAME
    if not directory.is_dir():
        raise ValueError(f"{directory} does not exist; the driver creates it after the token")
    done = attempts_of(spec, output_root=output_root)
    number = (done[-1] if done else 0) + 1
    text = f"attempt {number} of {spec.name}, started {_utc_now()}, code {_code_commit()}\n"
    _link_exclusive(text.encode("utf-8"), directory / f"{spec.name}.{number}")
    return number


def _declared(runs: Sequence[RunSpec] | None) -> list[RunSpec]:
    return list(registered_runs() if runs is None else runs)


def _manifest_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line)
        if match is None or match.group(2) in entries:
            raise ValueError(f"{path}:{number}: not one sha256sum line: {line!r}")
        entries[match.group(2)] = match.group(1)
    return entries


def write_manifest(output_root: str | Path, *, runs: Sequence[RunSpec] | None = None) -> Path:
    """``SHA256SUMS_p7_3c_finetune.txt``: every declared checkpoint by digest, written once, then re-verified.

    Refuses a stray entry in the checkpoints directory, a missing checkpoint, and an existing manifest with other
    content; the same content again is a no-op, so a restarted driver can reach the record.
    """
    root = Path(output_root)
    declared = _declared(runs)
    checkpoints = training_root(root) / CHECKPOINTS_DIRNAME
    wanted = {f"{spec.name}.pt" for spec in declared}
    if checkpoints.is_dir():
        stray = sorted(path.name for path in checkpoints.iterdir() if path.name not in wanted)
        if stray:
            raise ValueError(
                f"{checkpoints / stray[0]} is not one of the declared runs ({len(stray)} such entr"
                f"{'y' if len(stray) == 1 else 'ies'}); the manifest lists exactly the declared checkpoints"
            )
    lines: list[tuple[str, str]] = []
    for spec in declared:
        path = registered_destination(root, spec)
        if not path.is_file():
            raise ValueError(f"{path} is absent: every declared run needs its checkpoint before the manifest")
        lines.append((str(path.relative_to(root)), _sha256_file(path)))
    text = "".join(f"{digest}  {relative}\n" for relative, digest in sorted(lines))
    manifest = root / MANIFEST_FILENAME
    if manifest.exists():
        if manifest.read_text(encoding="utf-8") != text:
            raise ValueError(
                f"{manifest} already exists and differs from what the checkpoints give; it is never rewritten -- a "
                "person moves it aside"
            )
    else:
        _link_exclusive(text.encode("utf-8"), manifest)
    for relative, digest in _manifest_entries(manifest).items():
        if _sha256_file(root / relative) != digest:
            raise ValueError(f"{root / relative} does not match {manifest}")
    return manifest


def build_record(
    output_root: str | Path, *, corpus_dir: str | Path, timing_path: str | Path,
    runs: Sequence[RunSpec] | None = None, pins: Mapping[int, str] | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """The training record: per run its digests, frozen-part checks, steps, seconds, loss and attempts; the corpus,
    calibration and manifest digests; G5's timing record.  Refuses rather than record an invalid or partial set."""
    from offline.tier_sweep import canonical_state_dict_digest
    from offline.transfer_curve import P7_3D_CALIBRATION_SHA256

    root = Path(output_root)
    declared = _declared(runs)
    manifest = root / MANIFEST_FILENAME
    if not manifest.is_file():
        raise ValueError(f"{manifest} is absent: the manifest is written first, and the record repeats its digests")
    listed = _manifest_entries(manifest)
    timing = Path(timing_path)
    timing_record = _read_json_record(timing, TIMING_FORMAT_VERSION, "G5's timing record")
    _entries, corpus_sums = _verified_entries(Path(corpus_dir))
    calibration = (Path(data_dir) if data_dir is not None else _DATA_DIR) / CALIBRATION_NAME
    calibration_digest = _sha256_file(calibration)
    if calibration_digest != P7_3D_CALIBRATION_SHA256:
        raise ValueError(f"{calibration}: sha256 {calibration_digest} is not the pinned {P7_3D_CALIBRATION_SHA256}")

    entries: dict[str, Any] = {}
    for spec in declared:
        path = registered_destination(root, spec)
        relative = str(path.relative_to(root))
        digest = _sha256_file(path)
        if listed.get(relative) != digest:
            raise ValueError(f"{relative} is not in {manifest.name} at its current digest {digest}")
        checks = validate_checkpoint(spec, output_root=root, pins=pins, data_dir=data_dir)
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                f"{spec.name} does not validate (failed: {failed}); the record is never written for a set whose frozen "
                "parts, budget, targets or switch differ from what A24(b) registers"
            )
        payload = _load_weights_only(path)
        block = payload["provenance"]["few_shot"]
        if block["corpus_sha256sums_sha256"] != corpus_sums:
            raise ValueError(f"{spec.name} was trained on a corpus whose SHA256SUMS is not {corpus_dir}'s")
        run_record = _read_json_record(
            run_record_path(root, spec), RUN_RECORD_FORMAT_VERSION,
            "the checkpoint exists without its run record (the process died between the two writes); the "
            "coordinator decides",
        )
        if run_record.get("checkpoint_sha256") != digest:
            raise ValueError(f"{run_record_path(root, spec)} records another checkpoint digest")
        attempts = attempts_of(spec, output_root=root)
        entries[spec.name] = {
            "subject": spec.subject,
            "init": spec.init,
            "k": int(spec.k),
            "budget": int(spec.budget),
            "seed": int(spec.seed),
            "checkpoint": relative,
            "checkpoint_sha256": digest,
            "weights_sha256": canonical_state_dict_digest(path),
            "source_sha256": block["source_sha256"],
            "frozen_checks": checks,
            "steps": int(payload["provenance"]["gradient_steps"]),
            "warmup_steps": int(payload["provenance"]["warmup_steps"]),
            "loop_seconds": run_record["loop_seconds"],
            "final_loss": run_record["final_loss"],
            "attempts": len(attempts),
            "reruns": max(0, len(attempts) - 1),
            "git_commit": payload["provenance"]["git_commit"],
        }
    return {
        "format_version": RECORD_FORMAT_VERSION,
        "registered_in": "PREREGISTRATION A24(b); BRIEF_41 C3-C4, Amendment A (Q9)",
        "n_runs": len(declared),
        "runs": entries,
        "corpus_dir": str(Path(corpus_dir).resolve()),
        "corpus_sha256sums_sha256": corpus_sums,
        "calibration_sha256": calibration_digest,
        "manifest": MANIFEST_FILENAME,
        "manifest_sha256": _sha256_file(manifest),
        "timing": {"path": str(timing), "sha256": _sha256_file(timing), "record": timing_record},
        "reruns_rule": "a re-run is an attempt marker with no checkpoint after it: an infrastructure failure (A24(b))",
    }


def write_record(output_root: str | Path, record: Mapping[str, Any]) -> Path:
    """``p7_3c_training/p7_3c_finetune.json``, written once; the same content again is a no-op, other content refused."""
    path = training_root(output_root) / RECORD_NAME
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"{path} already exists and differs; it is never rewritten -- a person moves it aside")
        return path
    _link_exclusive(text.encode("utf-8"), path)
    return path


def choose_concurrency(
    measurements: Mapping[int, Mapping[str, float]], *, device_mib: float = DEVICE_MIB, fraction: float = DEVICE_FRACTION
) -> int:
    """The rule fixed before G5 measured anything: the C with the largest aggregate throughput (C / ms-per-step at C)
    among those whose measured device peak is <= *fraction* of *device_mib*; a tie goes to the smaller C.  A device
    that cannot hold ONE run within the cap is refused -- ``BRIEF_41`` G5: BLOCKED, with the numbers."""
    cap = device_mib * fraction
    eligible = [
        (int(c) / float(m["ms_per_step"]), int(c))
        for c, m in sorted(measurements.items())
        if float(m["device_peak_mib"]) <= cap
    ]
    if not eligible or eligible[0][1] != 1:
        raise ValueError(
            f"the GPU cannot hold one run within {fraction:.0%} of {device_mib:.0f} MiB (measured "
            f"{ {int(c): float(m['device_peak_mib']) for c, m in measurements.items()} } MiB): BLOCKED, with the numbers"
        )
    best = max(throughput for throughput, _c in eligible)
    return min(c for throughput, c in eligible if throughput == best)


def timing_destination(output_root: str | Path, stamp: str, slot: str) -> Path:
    """A fenced timing checkpoint: ``p7_3c_training/fenced_timing/<stamp>/<slot>.pt``."""
    if _STAMP.fullmatch(str(stamp)) is None:
        raise ValueError(f"timing stamp {stamp!r} is not a UTC stamp like 20260925T230000Z")
    if slot not in TIMING_SLOTS:
        raise ValueError(f"timing slot {slot!r} is not one of {list(TIMING_SLOTS)}")
    return training_root(output_root) / FENCED_TIMING_DIRNAME / str(stamp) / f"{slot}.pt"


def _slot_ms(record: Mapping[str, Any]) -> float:
    return float(record["loop_seconds"]) / int(record["steps"]) * 1000


def summarize_timing(stamp_dir: str | Path) -> dict[str, Any]:
    """G5's summary: ms/step per phase (the mean of its slots), the slowdown against ALONE, the device peak (the
    largest ``nvidia-smi`` sample of the phase), the repeat's verdict by TWO routes (file sha256, weights-only
    digest), the concurrency by :func:`choose_concurrency`, and the k = 100 build (Amendment A, Q12)."""
    stamp = Path(stamp_dir)
    assert_fence(stamp / "timing.pt", timing=True)
    slots = {
        name: _read_json_record(stamp / f"{name}.json", TIMING_SLOT_FORMAT_VERSION, f"timing slot {name}")
        for name in TIMING_SLOTS
    }
    phases: dict[str, dict[str, Any]] = {}
    for phase, members in TIMING_PHASES.items():
        per_slot = [_slot_ms(slots[name]) for name in members]
        samples_path = stamp / f"nvidia_smi_{phase}.csv"
        if not samples_path.is_file():
            raise ValueError(f"{samples_path} is absent: the driver samples the device during every phase")
        samples = [float(value) for value in samples_path.read_text(encoding="utf-8").split()]
        if not samples:
            raise ValueError(f"{samples_path} holds no sample")
        phases[phase] = {
            "concurrency": len(members),
            "slots": list(members),
            "per_slot_ms_per_step": per_slot,
            "ms_per_step": sum(per_slot) / len(per_slot),
            "device_peak_mib": max(samples),
            "peak_allocated_mib": max(float(slots[name]["peak_allocated_mib"]) for name in members),
        }
    alone = phases["alone"]["ms_per_step"]
    for phase in phases.values():
        phase["slowdown"] = phase["ms_per_step"] / alone
    table = {phase["concurrency"]: phase for phase in phases.values()}
    build = _read_json_record(stamp / "build_k100.json", TIMING_BUILD_FORMAT_VERSION, "the k = 100 build timing")
    return {
        "format_version": TIMING_FORMAT_VERSION,
        "stamp": stamp.name,
        "phases": phases,
        "repeat": {
            "file_sha256_equal": slots["repeat"]["checkpoint_sha256"] == slots["alone"]["checkpoint_sha256"],
            "weights_sha256_equal": slots["repeat"]["weights_sha256"] == slots["alone"]["weights_sha256"],
        },
        "concurrency": choose_concurrency(table),
        "rule": (
            f"C in {{1, 2, 3}} with the largest aggregate throughput (C / ms-per-step) whose device peak is <= "
            f"{DEVICE_FRACTION:.0%} of {DEVICE_MIB:.0f} MiB; a tie goes to the smaller C (plan section 8)"
        ),
        "build_k100": build,
        "q12_one_process_per_k": float(build["seconds"]) > Q12_BUILD_SECONDS,
    }


def check_inputs(
    *, output_root: str | Path, corpus_dir: str | Path, gate_record: str | Path,
    pins: Mapping[int, str] | None = None, data_dir: str | Path | None = None,
    timing_path: str | Path | None = None,
) -> dict[str, Any]:
    """Every input by digest, before the canary and the token: the calibration artifact at its pin, A20(a)'s five
    sources at theirs, the corpus's sums entry by entry over the whole band, G3's gate record for THIS corpus (100
    draws, 1,600 returns, zero events), CUDA -- and, for the trainings, G5's record: its concurrency re-derived by the
    rule and enough free device memory for it."""
    from offline.transfer_curve import P7_3D_CALIBRATION_SHA256

    root = Path(output_root)
    calibration = (Path(data_dir) if data_dir is not None else _DATA_DIR) / CALIBRATION_NAME
    if not calibration.is_file():
        raise ValueError(f"{calibration} is absent")
    calibration_digest = _sha256_file(calibration)
    if calibration_digest != P7_3D_CALIBRATION_SHA256:
        raise ValueError(f"{calibration}: sha256 {calibration_digest} is not the pinned {P7_3D_CALIBRATION_SHA256}")
    sources: dict[int, str] = {}
    for seed in TRAINING_SEEDS:
        path = registered_source_path(root, seed)
        if not path.is_file():
            raise ValueError(f"{path} is absent: A20(a)'s five checkpoints are read from the output tree")
        pin = _pin_for(seed, pins)
        digest = _sha256_file(path)
        if digest != pin:
            raise ValueError(f"seed {seed}: {path} has sha256 {digest}, which is not the pinned {pin}")
        sources[seed] = digest
    prefix = select_prefix(corpus_dir, max(REGISTERED_KS), scenario_id=SCENARIO_ID)

    gate = Path(gate_record)
    record = _read_json_record(
        gate, GATE_FORMAT_VERSION, "G3's gate record, which the corpus driver writes only when A17(f)'s gate passes"
    )
    recorded_corpus = Path(str(record.get("corpus_dir"))).resolve()
    if recorded_corpus != Path(corpus_dir).resolve():
        raise ValueError(f"the gate record is for {recorded_corpus}, not {Path(corpus_dir).resolve()}")
    if record.get("all_match") is not True:
        raise ValueError(f"the gate record says all_match {record.get('all_match')!r}")
    for field, expected, noun in (
        ("n_draws", 100, "draws"), ("n_intersections", 16, "intersections"),
        ("n_checked", 1600, "returns"), ("n_matching", 1600, "matching returns"),
    ):
        if record.get(field) != expected:
            verb = "checked" if field == "n_checked" else "records"
            raise ValueError(f"the gate record {verb} {record.get(field)} {noun}, not {expected}")
    events = dict(record.get("engine_events") or {})
    if events.get("n_teleports") != 0 or events.get("n_collisions") != 0:
        raise ValueError(
            f"the gate record counts {events.get('n_teleports')} teleport(s) and {events.get('n_collisions')} "
            "collision(s); A24(b) requires zero of each"
        )

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available; the registered runs and G5's timing train on cuda only")
    free_bytes, _total = torch.cuda.mem_get_info()
    free_mib = free_bytes / 2**20
    facts: dict[str, Any] = {
        "calibration_sha256": calibration_digest,
        "source_sha256": sources,
        "corpus_sha256sums_sha256": prefix.sums_sha256,
        "gate_record_sha256": _sha256_file(gate),
        "cuda_free_mib": free_mib,
    }
    if timing_path is not None:
        concurrency, needed = _timing_concurrency(Path(timing_path))
        if free_mib < needed:
            raise ValueError(
                f"{free_mib:.0f} MiB free on the device, below the {needed:.0f} MiB G5 measured at concurrency "
                f"{concurrency}"
            )
        facts["concurrency"] = concurrency
        facts["timing_sha256"] = _sha256_file(Path(timing_path))
    return facts


def _timing_concurrency(path: Path) -> tuple[int, float]:
    """G5's recorded concurrency, RE-DERIVED by the rule from the record's own phases; and that phase's device peak."""
    timing = _read_json_record(path, TIMING_FORMAT_VERSION, "G5's timing record (the fenced timing run writes it)")
    by_c = {len(TIMING_PHASES[name]): timing["phases"][name] for name in TIMING_PHASES}
    concurrency = choose_concurrency(by_c)
    if concurrency != int(timing["concurrency"]):
        raise ValueError(f"{path} records concurrency {timing['concurrency']}, but the rule gives {concurrency}")
    return concurrency, float(by_c[concurrency]["device_peak_mib"])


# ======================================================================================================================
# The command line
# ======================================================================================================================


def _cmd_train(args: argparse.Namespace) -> int:
    """One REGISTERED run on CUDA: the source at A20(a)'s pin, the registered destination, then its run record."""
    from offline.tier_sweep import configure_determinism
    from offline.transfer_calibration import GRID4X4_CHECKPOINT_SHA256

    spec = run_by_name(args.run)
    if args.device != "cuda":
        raise ValueError(
            "the registered runs train on cuda only (A24(b): the subject's regime, CUDA and non-deterministic, as "
            "P5.2's); a CPU fine-tune is a test, never a registered checkpoint"
        )
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available; the registered runs train on cuda only")
    output_root = Path(args.output_root)
    destination = assert_fence(registered_destination(output_root, spec), timing=False)
    record = run_record_path(output_root, spec)
    if not record.parent.is_dir():
        raise ValueError(f"{record.parent} does not exist; the driver creates it after the token")
    if record.exists():
        raise ValueError(f"{record} already exists; a run record is written once, with its checkpoint")
    configure_determinism(False)
    print(
        f"few_shot train {spec.name}: init {spec.init}, k {spec.k}, B {spec.budget}, seed {spec.seed}, "
        f"source {registered_source_path(output_root, spec.seed).name}, device cuda",
        flush=True,
    )
    result = fine_tune(
        source_path=registered_source_path(output_root, spec.seed),
        source_sha256=GRID4X4_CHECKPOINT_SHA256[spec.seed],
        corpus_dir=Path(args.corpus_dir),
        k=spec.k,
        budget=spec.budget,
        seed=spec.seed,
        init=spec.init,
        device="cuda",
        destination=destination,
        data_dir=args.data_dir,
        log_every=int(args.log_every),
    )
    written = write_run_record(spec, result, output_root=output_root, device="cuda")
    print(
        f"few_shot train {spec.name}: wrote {result.destination} (sha256 {result.sha256}), {result.steps} steps in "
        f"{result.seconds:.1f} s; run record {written}",
        flush=True,
    )
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    """The thirty registered run names, one per line, in the registered order."""
    for spec in registered_runs():
        print(spec.name)
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    """One run's decision, or -- ``--all``, the driver's pre-token scan -- every run's, with no stray checkpoint."""
    root = Path(args.output_root)
    if args.all:
        checkpoints = training_root(root) / CHECKPOINTS_DIRNAME
        wanted = {f"{spec.name}.pt" for spec in registered_runs()}
        if checkpoints.is_dir():
            stray = sorted(path.name for path in checkpoints.iterdir() if path.name not in wanted)
            if stray:
                raise ValueError(
                    f"{checkpoints / stray[0]} is not one of the 30 registered runs; a person moves it aside before "
                    "the trainings start"
                )
        counts = Counter(resume_decision(spec, output_root=root, data_dir=args.data_dir) for spec in registered_runs())
        print(f"resume_decision: {counts['train']} to train, {counts['skip']} to skip", flush=True)
        return 0
    if args.run is None:
        raise ValueError("resume-decision needs --run NAME or --all")
    print(resume_decision(run_by_name(args.run), output_root=root, data_dir=args.data_dir), flush=True)
    return 0


def _cmd_attempt(args: argparse.Namespace) -> int:
    print(next_attempt(run_by_name(args.run), output_root=Path(args.output_root)), flush=True)
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    path = write_manifest(Path(args.output_root))
    print(f"manifest: {path} lists the {len(registered_runs())} checkpoints and was re-verified", flush=True)
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    record = build_record(
        Path(args.output_root), corpus_dir=Path(args.corpus_dir), timing_path=Path(args.timing),
        data_dir=args.data_dir,
    )
    path = write_record(Path(args.output_root), record)
    print(f"record: {path} ({record['n_runs']} runs; the coordinator verifies it at G7)", flush=True)
    return 0


def _cmd_check_inputs(args: argparse.Namespace) -> int:
    facts = check_inputs(
        output_root=Path(args.output_root), corpus_dir=Path(args.corpus_dir), gate_record=Path(args.gate_record),
        data_dir=args.data_dir, timing_path=None if args.timing is None else Path(args.timing),
    )
    tail = "" if "concurrency" not in facts else f", concurrency {facts['concurrency']} from G5's record"
    print(
        f"check_inputs PASSED: calibration {facts['calibration_sha256'][:12]}, sources 5/5 at A20(a)'s pins, corpus "
        f"SHA256SUMS {facts['corpus_sha256sums_sha256'][:12]} (the band 201-300 verified entry by entry), gate record "
        f"{facts['gate_record_sha256'][:12]} (1600/1600, zero events), CUDA {facts['cuda_free_mib']:.0f} MiB free{tail}",
        flush=True,
    )
    return 0


def _cmd_timing(args: argparse.Namespace) -> int:
    """G5's fenced timing: one run, the k = 100 build, the summary, or the concurrency a summary gives."""
    from offline.tier_sweep import canonical_state_dict_digest
    from offline.transfer_calibration import GRID4X4_CHECKPOINT_SHA256

    root = Path(args.output_root)
    if args.action == "concurrency":
        concurrency, _needed = _timing_concurrency(Path(args.timing))
        print(concurrency, flush=True)
        return 0
    stamp_dir = training_root(root) / FENCED_TIMING_DIRNAME / str(args.stamp)
    if _STAMP.fullmatch(str(args.stamp)) is None or not stamp_dir.is_dir():
        raise ValueError(f"{stamp_dir} is not an existing fenced timing directory; the driver creates it")
    if args.action == "summarize":
        summary = summarize_timing(stamp_dir)
        out = stamp_dir / TIMING_RECORD_NAME
        _link_exclusive((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"), out)
        for name, phase in summary["phases"].items():
            print(
                f"timing {name}: {phase['ms_per_step']:.1f} ms/step (x{phase['slowdown']:.2f} alone), device peak "
                f"{phase['device_peak_mib']:.0f} MiB, allocated peak {phase['peak_allocated_mib']:.0f} MiB per process",
                flush=True,
            )
        print(
            f"timing repeat: file sha256 {'EQUAL' if summary['repeat']['file_sha256_equal'] else 'DIFFERS'}, weights "
            f"{'EQUAL' if summary['repeat']['weights_sha256_equal'] else 'DIFFER'}; k = 100 build "
            f"{summary['build_k100']['seconds']:.1f} s; concurrency {summary['concurrency']}; wrote {out}",
            flush=True,
        )
        return 0
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available; G5 times the registered regime")
    spec = timing_spec()
    if args.action == "build":
        import resource

        out = stamp_dir / f"build_k{int(args.k)}.json"
        if out.exists():
            raise ValueError(f"{out} already exists; a timing record is written once")
        started = time.perf_counter()
        source = load_source(registered_source_path(root, spec.seed), expected_sha256=GRID4X4_CHECKPOINT_SHA256[spec.seed])
        prefix = select_prefix(Path(args.corpus_dir), int(args.k), scenario_id=SCENARIO_ID)
        windows = build_windows(Path(args.corpus_dir), prefix.draw_ids, source)
        seconds = time.perf_counter() - started
        record = {
            "format_version": TIMING_BUILD_FORMAT_VERSION,
            "k": int(args.k),
            "seconds": float(seconds),
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "n_joint_windows": int(windows.index.n_windows),
            "git_commit": _code_commit(),
            "written_utc": _utc_now(),
        }
        _link_exclusive((json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"), out)
        print(f"timing build k {args.k}: {seconds:.1f} s, peak RSS {record['peak_rss_mib']:.0f} MiB", flush=True)
        return 0
    destination = assert_fence(timing_destination(root, str(args.stamp), str(args.slot)), timing=True)
    out = destination.with_suffix(".json")
    if out.exists():
        raise ValueError(f"{out} already exists; a timing record is written once")
    torch.cuda.reset_peak_memory_stats()
    result = fine_tune(
        source_path=registered_source_path(root, spec.seed),
        source_sha256=GRID4X4_CHECKPOINT_SHA256[spec.seed],
        corpus_dir=Path(args.corpus_dir),
        k=spec.k,
        budget=spec.budget,
        seed=spec.seed,
        init=spec.init,
        device="cuda",
        destination=destination,
        data_dir=args.data_dir,
        log_every=100,
    )
    record = {
        "format_version": TIMING_SLOT_FORMAT_VERSION,
        "slot": str(args.slot),
        "run": spec.name,
        "steps": int(result.steps),
        "loop_seconds": float(result.seconds),
        "ms_per_step": float(result.seconds) / int(result.steps) * 1000.0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "checkpoint_sha256": result.sha256,
        "weights_sha256": canonical_state_dict_digest(destination),
        "device_name": torch.cuda.get_device_name(0),
        "git_commit": _code_commit(),
        "written_utc": _utc_now(),
    }
    _link_exclusive((json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"), out)
    print(
        f"timing {args.slot}: {record['ms_per_step']:.1f} ms/step over {result.steps} steps, peak "
        f"{record['peak_allocated_mib']:.0f} MiB allocated",
        flush=True,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m offline.few_shot", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help_text: str, handler: Any, *, corpus: bool = False) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--output-root", required=True, help="the output tree holding p5_2/ and p7_3c_training/")
        sub.add_argument("--data-dir", default=None, help="the directory holding p7_3d_calibration.json")
        if corpus:
            sub.add_argument("--corpus-dir", required=True, help="the verified grid4x4 SUMO corpus")
        sub.set_defaults(handler=handler)
        return sub

    train = command("train", "one registered fine-tune on CUDA, written once", _cmd_train, corpus=True)
    train.add_argument("--run", required=True, help="a registered run name, e.g. ft_k100_seed101")
    train.add_argument("--device", default="cuda")
    train.add_argument("--log-every", type=int, default=500)

    runs = commands.add_parser("runs", help="the thirty registered run names, in order")
    runs.set_defaults(handler=_cmd_runs)

    resume = command("resume-decision", "skip / train for one run, or the pre-token scan of all", _cmd_resume)
    which = resume.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", default=None)
    which.add_argument("--all", action="store_true")

    attempt = command("attempt", "write the next attempt marker of a run; print its number", _cmd_attempt)
    attempt.add_argument("--run", required=True)

    command("manifest", "write and re-verify SHA256SUMS_p7_3c_finetune.txt", _cmd_manifest)

    record = command("record", "write p7_3c_training/p7_3c_finetune.json", _cmd_record, corpus=True)
    record.add_argument("--timing", required=True, help="G5's timing.json")

    inputs = command("check-inputs", "every input by digest, before the canary", _cmd_check_inputs, corpus=True)
    inputs.add_argument("--gate-record", required=True, help="G3's a17f_gate.json")
    inputs.add_argument("--timing", default=None, help="G5's timing.json (train mode)")

    timing = commands.add_parser("timing", help="G5's fenced timing")
    actions = timing.add_subparsers(dest="action", required=True)
    for action, help_text in (
        ("run", "one fenced run: k 5, B 400, seed 101"),
        ("build", "the k windows built, no step"),
        ("summarize", "the phases, the repeat, the concurrency"),
        ("concurrency", "the concurrency a summary gives, re-derived by the rule"),
    ):
        sub = actions.add_parser(action, help=help_text)
        sub.add_argument("--output-root", required=True)
        sub.add_argument("--data-dir", default=None)
        if action in ("run", "build"):
            sub.add_argument("--corpus-dir", required=True)
        if action != "concurrency":
            sub.add_argument("--stamp", required=True)
        if action == "run":
            sub.add_argument("--slot", required=True, choices=TIMING_SLOTS)
        if action == "build":
            sub.add_argument("--k", type=int, default=100)
        if action == "concurrency":
            sub.add_argument("--timing", required=True)
        sub.set_defaults(handler=_cmd_timing)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; a refusal prints its reason and returns 2.  Anything else propagates, loudly."""
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f"few_shot {args.command}: REFUSED: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
