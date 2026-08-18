"""P8.3: the runner for the external IQL calibration -- the ONLY file that touches MuJoCo.

Artifact format version: ``p8.3-d4rl-calibration/1.0`` (see ``calibration/d4rl_adapter.py``).

WHAT IT DOES
------------
Runs **our own unchanged** ``offline.offline_baselines.train_iql`` and ``train_bc`` on
``halfcheetah-medium-expert-v2`` and reports the D4RL normalised score against the IQL paper's
own Table 1.  ``docs/plans/p8.3.md`` fixes every knob, gate and reading rule; this file executes
them and refuses to invent one.

    python -m calibration.d4rl_run gates --dataset PATH --out DIR
    python -m calibration.d4rl_run train --dataset PATH --out DIR [--arms ...] [--codebook 64]

**The two subcommands are separate on purpose.**  ``gates`` writes the provenance, the
no-terminals check, the dynamics-fidelity check and the quantisation ceilings for all three
declared codebook sizes; it is committed **before** the first gradient step, which is what makes
the codebook size unselectable on a training result.

DEPENDENCIES
------------
``h5py``, ``gymnasium`` and ``mujoco`` are imported here and **nowhere else in this repository**.
They are authorised for this task alone (``PROJECT_PLAN`` Decisions Log 2026-08-16) under three
binding conditions; ``tests/test_d4rl_calibration.py`` asserts condition (ii) mechanically over
every tracked ``.py`` outside ``calibration/``.  They live in a separate venv outside the repo and
are absent from ``pyproject.toml`` and from every requirements file the project installs from.

THE ENVIRONMENT IS A SUBSTITUTE, AND GATE D IS WHY THAT IS DEFENSIBLE
--------------------------------------------------------------------
The ``-v2`` datasets were generated under ``mujoco-py`` 2.1 with gym's ``HalfCheetah-v3``.  We
evaluate in gymnasium's ``HalfCheetah-v4``, the direct port of that environment onto the current
MuJoCo bindings, because the original stack does not install on a current interpreter.  **Gate D
measures the substitution instead of asserting it**: it sets MuJoCo's state exactly from a dataset
observation, applies the recorded action, and compares the returned reward and next observation
against the dataset's own.  If those do not reproduce, the evaluation environment is not the
data-generating environment and no normalised score from it is comparable to the published table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import h5py
import mujoco
import numpy as np
import torch

from agent.OfflineBaselines import MLPTrunk, TrunkConfig, masked_action_logits
from calibration.d4rl_adapter import (
    ARTIFACT_FORMAT_VERSION,
    CODEBOOK_ITERATIONS,
    CODEBOOK_SEED,
    CODEBOOK_SIZES,
    CODEBOOK_SUBSAMPLE,
    DATASET_NAME,
    DATASET_URL,
    EVAL_EPISODES_PER_SEED,
    PRIMARY_CODEBOOK_SIZE,
    PUBLISHED_SCORES,
    REF_MAX_SCORE,
    REF_MIN_SCORE,
    SCENARIO_ID,
    STREAM_ID,
    ActionCodebook,
    EpisodeSpans,
    bc_windows,
    build_transition_table,
    episode_returns,
    episode_spans,
    normalization_stats,
    normalized_score,
    top_return_episodes,
)
from offline.dt_gate import (
    CONTEXT_LENGTH,
    TRAINING_SEEDS,
    mean_ci95,
    runtime_provenance,
    write_json_atomic,
)
from offline.offline_baselines import (
    BC_BATCH_WINDOWS,
    DECLARED_GRADIENT_STEPS,
    IQL_BATCH_TRANSITIONS,
    TOP_RETURN_FRACTION,
    iql_reward_scale,
    pin_torch_threads,
    train_bc,
    train_iql,
)

#: gymnasium's port of the environment the -v2 datasets were generated in.  Gate D measures it.
ENV_ID = "HalfCheetah-v4"

#: The declared arms.  ``iql`` is the brief's; the two BC arms are the control for the adapter's
#: ceiling, ruled in on 2026-08-18 (``docs/plans/p8.3.md`` section 2.1).
ARMS: tuple[str, ...] = ("bc", "bc_top10", "iql")

#: Gate D's sample size and Gate C's replay set, both fixed in the plan before the run.
GATE_D_SAMPLES = 10_000
GATE_C_EPISODES = 20

#: The project's own pins (``pyproject.toml``).  The run refuses to start if the calibration venv
#: differs, because "same code path" is not true across library versions.
REQUIRED_TORCH = "2.11.0"
REQUIRED_NUMPY = "2.5.1"


# ----------------------------------------------------------------------
# Loading, and the provenance that goes with it
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    """sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    """The five D4RL arrays plus ``next_observations``, read through ``h5py``.

    ``next_observations`` is required rather than reconstructed: it is the post-step observation
    of the **last** transition of every episode, which no within-episode shift can recover, and
    that transition is exactly the one contract C6's no-``done`` bootstrap is about.
    """
    with h5py.File(path, "r") as handle:
        keys = set(handle.keys())
        required = {"observations", "actions", "rewards", "terminals", "timeouts"}
        missing = sorted(required - keys)
        if missing:
            raise ValueError(f"{path} is missing the D4RL arrays {missing}; it has {sorted(keys)}")
        if "next_observations" not in keys:
            raise ValueError(
                f"{path} carries no next_observations; the final transition of every episode "
                "would have no bootstrap target and dropping it is a done term by omission"
            )
        return {
            "observations": np.asarray(handle["observations"], dtype=np.float32),
            "next_observations": np.asarray(handle["next_observations"], dtype=np.float32),
            "actions": np.asarray(handle["actions"], dtype=np.float32),
            "rewards": np.asarray(handle["rewards"], dtype=np.float32),
            "terminals": np.asarray(handle["terminals"], dtype=bool),
            "timeouts": np.asarray(handle["timeouts"], dtype=bool),
        }


def environment_provenance() -> dict[str, Any]:
    """Every library whose version can move a number here."""
    return {
        "env_id": ENV_ID,
        "gymnasium": gym.__version__,
        "mujoco": mujoco.__version__,
        "h5py": h5py.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "note": (
            "the -v2 datasets were generated under mujoco-py 2.1 with gym HalfCheetah-v3; this "
            "is gymnasium's port onto the current bindings, and Gate D measures the substitution"
        ),
    }


def assert_pinned_libraries() -> None:
    """Refuse to run on a torch or numpy other than the project's pins."""
    installed_torch = torch.__version__.split("+", 1)[0]
    if installed_torch != REQUIRED_TORCH or np.__version__ != REQUIRED_NUMPY:
        raise RuntimeError(
            f"this venv has torch {torch.__version__} / numpy {np.__version__}; the project pins "
            f"torch=={REQUIRED_TORCH} and numpy=={REQUIRED_NUMPY} (pyproject.toml) and 'the same "
            "code path' is not true across library versions"
        )


# ----------------------------------------------------------------------
# Gates A, B, D
# ----------------------------------------------------------------------
def gate_a_provenance(data: dict[str, np.ndarray], path: Path) -> dict[str, Any]:
    """Identity and shape of the file we actually read.  Fails on anything unexpected."""
    observations = data["observations"]
    actions = data["actions"]
    if observations.ndim != 2 or observations.shape[1] != 17:
        raise ValueError(f"expected 17-wide observations, got shape {observations.shape}")
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError(f"expected 6-wide actions, got shape {actions.shape}")
    low, high = float(actions.min()), float(actions.max())
    if low < -1.0 or high > 1.0:
        raise ValueError(f"actions leave [-1, 1]: observed [{low}, {high}]")
    lengths = {name: int(array.shape[0]) for name, array in data.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"the arrays disagree on row count: {lengths}")
    return {
        "dataset": DATASET_NAME,
        "url": DATASET_URL,
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "rows": int(observations.shape[0]),
        "state_dim": int(observations.shape[1]),
        "action_dim": int(actions.shape[1]),
        "action_min": low,
        "action_max": high,
        "reward_min": float(data["rewards"].min()),
        "reward_max": float(data["rewards"].max()),
        "passed": True,
    }


def gate_b_no_terminals(data: dict[str, np.ndarray], spans: EpisodeSpans) -> dict[str, Any]:
    """HalfCheetah never terminates early, which is what makes our no-``done`` learner correct.

    Asserted rather than assumed.  If a terminal ever appeared, ``iql_targets`` would bootstrap
    through an absorbing state and the run would be mis-specified -- reported as such, not
    adjusted around.
    """
    terminals = int(np.count_nonzero(data["terminals"]))
    lengths = spans.lengths
    return {
        "terminal_count": terminals,
        "timeout_count": int(np.count_nonzero(data["timeouts"])),
        "episodes": int(len(spans)),
        "episode_length_min": int(lengths.min()),
        "episode_length_max": int(lengths.max()),
        "rows_covered": int(lengths.sum()),
        "rows_dropped_as_trailing_partial": int(data["observations"].shape[0] - lengths.sum()),
        "passed": terminals == 0,
    }


def _set_state(env: Any, observation: np.ndarray) -> None:
    """Put MuJoCo in exactly the state a dataset observation describes.

    HalfCheetah's observation is ``qpos[1:] ++ qvel`` -- the root x coordinate is excluded.  It is
    set to zero, which changes nothing that matters: the forward reward is a *difference* of x
    positions across one step, so a constant offset cancels.
    """
    unwrapped = env.unwrapped
    qpos = np.zeros(unwrapped.model.nq, dtype=np.float64)
    qpos[1:] = observation[: unwrapped.model.nq - 1]
    qvel = np.asarray(observation[unwrapped.model.nq - 1 :], dtype=np.float64)
    unwrapped.set_state(qpos, qvel)


def gate_d_dynamics(
    data: dict[str, np.ndarray], *, samples: int = GATE_D_SAMPLES, seed: int = 0
) -> dict[str, Any]:
    """One-step reproduction of the dataset's own reward and next observation.

    No trajectory divergence enters this: each sample sets the exact state, takes exactly one
    step, and compares.  It is the sharpest available test that our substitute environment is the
    environment the data came from.
    """
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    generator = np.random.default_rng(seed)
    rows = generator.choice(data["observations"].shape[0], size=int(samples), replace=False)

    reward_error = np.empty(rows.size, dtype=np.float64)
    observation_error = np.empty(rows.size, dtype=np.float64)
    for position, row in enumerate(rows):
        _set_state(env, data["observations"][row])
        observation, reward, _terminated, _truncated, _info = env.unwrapped.step(
            data["actions"][row].astype(np.float64)
        )
        reward_error[position] = abs(float(reward) - float(data["rewards"][row]))
        observation_error[position] = float(
            np.max(np.abs(observation - data["next_observations"][row]))
        )
    env.close()

    median = float(np.median(reward_error))
    return {
        "samples": int(rows.size),
        "reward_abs_error_median": median,
        "reward_abs_error_mean": float(reward_error.mean()),
        "reward_abs_error_p99": float(np.quantile(reward_error, 0.99)),
        "reward_abs_error_max": float(reward_error.max()),
        "next_observation_abs_error_median": float(np.median(observation_error)),
        "next_observation_abs_error_max": float(observation_error.max()),
        "threshold_median_reward_abs_error": 1e-5,
        "passed": median < 1e-5,
    }


# ----------------------------------------------------------------------
# Gate C: what the codebook costs, measured without training anything
# ----------------------------------------------------------------------
def _replay_return(env: Any, start: np.ndarray, actions: np.ndarray) -> float:
    """Open-loop replay from an exact state; the undiscounted sum of what the env returns."""
    _set_state(env, start)
    total = 0.0
    for action in actions:
        _observation, reward, _terminated, _truncated, _info = env.unwrapped.step(
            np.asarray(action, dtype=np.float64)
        )
        total += float(reward)
    return total


def gate_c_ceiling(
    data: dict[str, np.ndarray],
    spans: EpisodeSpans,
    books: dict[int, ActionCodebook],
    *,
    episodes: int = GATE_C_EPISODES,
) -> dict[str, Any]:
    """How much return survives quantisation, for every declared codebook size.

    The comparison is quantised replay against **raw replay**, not against the dataset's recorded
    return: both replays start from the same exact state and run open loop, so whatever the
    substitute environment does to a long rollout it does to both.  The raw-replay-against-dataset
    ratio is reported alongside as a second, independent check on the replay itself.
    """
    env = gym.make(ENV_ID)
    env.reset(seed=0)
    chosen = np.arange(len(spans) - int(episodes), len(spans), dtype=np.int64)
    recorded = episode_returns(data["rewards"], spans)

    raw: list[float] = []
    dataset: list[float] = []
    quantised: dict[int, list[float]] = {size: [] for size in books}
    for episode in chosen:
        begin, end = int(spans.start[episode]), int(spans.stop[episode])
        start = data["observations"][begin]
        actions = data["actions"][begin:end]
        raw.append(_replay_return(env, start, actions))
        dataset.append(float(recorded[episode]))
        for size, book in books.items():
            decoded = book.decode(book.encode(actions))
            quantised[size].append(_replay_return(env, start, decoded))
    env.close()

    raw_total = float(np.sum(raw))
    return {
        "episodes": [int(e) for e in chosen],
        "raw_replay_return_mean": float(np.mean(raw)),
        "dataset_return_mean": float(np.mean(dataset)),
        "replay_fidelity_ratio": raw_total / float(np.sum(dataset)),
        "by_codebook_size": {
            str(size): {
                "codebook_digest": books[size].digest,
                "quantisation_error": books[size].quantisation_error(
                    data["actions"][:: max(1, data["actions"].shape[0] // 200_000)]
                ),
                "quantised_replay_return_mean": float(np.mean(values)),
                "ceiling_ratio": float(np.sum(values)) / raw_total,
                "normalised_score_of_quantised_replay": float(
                    normalized_score(np.mean(values))
                ),
            }
            for size, values in quantised.items()
        },
    }


# ----------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------
def _load_policy(checkpoint: Path, device: torch.device) -> tuple[MLPTrunk, dict[str, Any]]:
    """The policy network out of a BC or IQL checkpoint, both of which prefix it ``policy.``."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = TrunkConfig.from_json_obj(payload["config"])
    model = MLPTrunk(config, config.n_actions).to(device)
    head = "policy."
    weights = {
        key[len(head) :]: value
        for key, value in payload["model"].items()
        if key.startswith(head)
    }
    if not weights:
        raise ValueError(f"{checkpoint} carries no policy network; it has {sorted(payload)}")
    model.load_state_dict(weights)
    model.eval()
    return model, payload


def evaluate_policy(
    model: MLPTrunk,
    book: ActionCodebook,
    stats: Any,
    *,
    episodes: int,
    seed_base: int,
    device: torch.device,
) -> list[float]:
    """Undiscounted returns of *episodes* rollouts, argmax over masked logits.

    This is ``IQLAgent.act(explore=False)``'s declared evaluation path: the same
    ``masked_action_logits`` call, and an all-``True`` mask because every mask in our corpus is.
    """
    env = gym.make(ENV_ID)
    mask = torch.ones((1, book.size), dtype=torch.bool, device=device)
    returns: list[float] = []
    for index in range(int(episodes)):
        observation, _info = env.reset(seed=int(seed_base) + index)
        total = 0.0
        while True:
            row = stats.normalize_state(SCENARIO_ID, STREAM_ID, observation[None, :])
            with torch.no_grad():
                logits = masked_action_logits(
                    model(torch.from_numpy(row).to(device)), mask
                )
                code = int(torch.argmax(logits, dim=-1).item())
            observation, reward, terminated, truncated, _info = env.step(
                book.decode(np.array([code], dtype=np.int64))[0]
            )
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    return returns


def train_and_evaluate(
    data: dict[str, np.ndarray],
    spans: EpisodeSpans,
    book: ActionCodebook,
    *,
    arms: Sequence[str],
    seeds: Sequence[int],
    steps: int,
    out_dir: Path,
    device: torch.device,
    log_every: int,
) -> dict[str, Any]:
    """Every (arm, seed): our own trainer, our own checkpoint, then rollouts.

    Nothing here reimplements a training step.  ``train_iql`` and ``train_bc`` are imported and
    called; the only thing this function owns is the shape of what goes in and the rollout that
    comes out.
    """
    returns = episode_returns(data["rewards"], spans)
    stats = normalization_stats(data["observations"], returns=returns)
    scale = iql_reward_scale(returns)
    action_index = book.encode(data["actions"])
    top = top_return_episodes(returns, TOP_RETURN_FRACTION)

    provenance = {
        "task": "P8.3",
        "dataset": DATASET_NAME,
        "codebook_size": book.size,
        "codebook_digest": book.digest,
        "reward_scale": float(scale),
        "environment": environment_provenance(),
    }

    table = None
    if "iql" in arms:
        table = build_transition_table(
            observations=data["observations"],
            next_observations=data["next_observations"],
            action_index=action_index,
            rewards=data["rewards"],
            spans=spans,
            stats=stats,
            reward_scale=float(scale),
        )

    windows: dict[str, dict[str, torch.Tensor]] = {}
    for arm in arms:
        if arm == "iql":
            continue
        windows[arm] = bc_windows(
            observations=data["observations"],
            action_index=action_index,
            spans=spans,
            stats=stats,
            n_actions=book.size,
            context_length=CONTEXT_LENGTH,
            episodes=top if arm == "bc_top10" else None,
        )

    cells: list[dict[str, Any]] = []
    for arm in arms:
        for seed in seeds:
            checkpoint = out_dir / f"{arm}_k{book.size}_seed{seed}.pt"
            started = time.time()
            if arm == "iql":
                assert table is not None
                record = train_iql(
                    table,
                    state_dim=int(data["observations"].shape[1]),
                    n_actions=book.size,
                    seed=int(seed),
                    declared_gradient_steps=int(steps),
                    batch_size=IQL_BATCH_TRANSITIONS,
                    device=device,
                    checkpoint_path=checkpoint,
                    stats=stats,
                    scenario_id=SCENARIO_ID,
                    provenance=provenance,
                    log_every=log_every,
                )
            else:
                record = train_bc(
                    windows[arm],
                    state_dim=int(data["observations"].shape[1]),
                    n_actions=book.size,
                    seed=int(seed),
                    method=arm,
                    declared_gradient_steps=int(steps),
                    batch_size=BC_BATCH_WINDOWS,
                    device=device,
                    checkpoint_path=checkpoint,
                    stats=stats,
                    scenario_id=SCENARIO_ID,
                    provenance=provenance,
                    log_every=log_every,
                )
            model, _payload = _load_policy(checkpoint, device)
            episode_return = evaluate_policy(
                model,
                book,
                stats,
                episodes=EVAL_EPISODES_PER_SEED,
                seed_base=1000 + 10 * int(seed),
                device=device,
            )
            scores = normalized_score(np.asarray(episode_return, dtype=np.float64))
            cells.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "gradient_steps": record.gradient_steps,
                    "plateaued": bool(record.plateaued),
                    "training_seconds": float(record.seconds),
                    "wall_seconds": float(time.time() - started),
                    "canonical_digest": record.canonical_digest,
                    "checkpoint_path": record.checkpoint_path,
                    "episode_returns": [float(v) for v in episode_return],
                    "episode_scores": [float(v) for v in scores],
                    "score_mean": float(scores.mean()),
                    "diagnostics": record.diagnostics,
                }
            )
            print(
                f"  {arm} seed {seed}: score {float(scores.mean()):.2f} "
                f"(return {float(np.mean(episode_return)):.1f}) in "
                f"{time.time() - started:.0f}s",
                flush=True,
            )

    summary: dict[str, Any] = {}
    for arm in arms:
        per_seed = [cell["score_mean"] for cell in cells if cell["arm"] == arm]
        stat = mean_ci95(per_seed)
        summary[arm] = {
            "seeds": len(per_seed),
            "per_seed_score": per_seed,
            "score_mean": stat.mean,
            "score_std": stat.std,
            "score_ci95_halfwidth": stat.ci95,
            "published_score": PUBLISHED_SCORES[arm],
            "difference_from_published": stat.mean - PUBLISHED_SCORES[arm],
        }
    return {"cells": cells, "summary": summary, "top_decile_episodes": int(top.size)}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Two subcommands, because the gates are committed before the training runs."""
    parser = argparse.ArgumentParser(
        description="P8.3: run our unchanged IQL on halfcheetah-medium-expert-v2."
    )
    parser.add_argument("--threads", type=int, default=1, help="torch thread pin (liveness)")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("gates", "train"):
        child = sub.add_parser(name)
        child.add_argument("--dataset", required=True, help="path to the D4RL hdf5")
        child.add_argument("--out", required=True, help="existing directory for the artifacts")
    train = sub.choices["train"]
    train.add_argument("--arms", default=",".join(ARMS))
    train.add_argument("--codebook", type=int, default=PRIMARY_CODEBOOK_SIZE)
    train.add_argument("--steps", type=int, default=DECLARED_GRADIENT_STEPS)
    train.add_argument("--seeds", default=",".join(str(s) for s in TRAINING_SEEDS))
    train.add_argument("--log-every", type=int, default=5_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate everything, then run.  Nothing is written before the last check passes."""
    args = build_parser().parse_args(argv)
    assert_pinned_libraries()
    pin_torch_threads(int(args.threads))

    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"no dataset at {dataset_path}")
    out_dir = Path(args.out)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {out_dir}; none is created here")

    print(f"reading {dataset_path} ...", flush=True)
    data = load_dataset(dataset_path)
    spans = episode_spans(data["timeouts"], data["terminals"])

    if args.command == "gates":
        gate_a = gate_a_provenance(data, dataset_path)
        gate_b = gate_b_no_terminals(data, spans)
        print(f"  gate A rows={gate_a['rows']} sha256={gate_a['sha256'][:16]}...", flush=True)
        print(f"  gate B terminals={gate_b['terminal_count']} episodes={gate_b['episodes']}",
              flush=True)
        gate_d = gate_d_dynamics(data)
        print(f"  gate D median |dr| = {gate_d['reward_abs_error_median']:.3e}", flush=True)
        books = {
            size: ActionCodebook.fit(
                data["actions"],
                size,
                seed=CODEBOOK_SEED,
                iterations=CODEBOOK_ITERATIONS,
                subsample=CODEBOOK_SUBSAMPLE,
            )
            for size in CODEBOOK_SIZES
        }
        gate_c = gate_c_ceiling(data, spans, books)
        for size, entry in sorted(gate_c["by_codebook_size"].items(), key=lambda kv: int(kv[0])):
            print(f"  gate C K={size}: ceiling {entry['ceiling_ratio']:.4f}", flush=True)
        payload = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "task": "P8.3",
            "stage": "gates",
            "declared": {
                "codebook_sizes": list(CODEBOOK_SIZES),
                "primary_codebook_size": PRIMARY_CODEBOOK_SIZE,
                "codebook_seed": CODEBOOK_SEED,
                "codebook_iterations": CODEBOOK_ITERATIONS,
                "codebook_subsample": CODEBOOK_SUBSAMPLE,
                "gate_c_episodes": GATE_C_EPISODES,
                "gate_d_samples": GATE_D_SAMPLES,
                "reference_min_score": REF_MIN_SCORE,
                "reference_max_score": REF_MAX_SCORE,
                "published_scores": PUBLISHED_SCORES,
            },
            "gate_a_provenance": gate_a,
            "gate_b_no_terminals": gate_b,
            "gate_c_quantisation_ceiling": gate_c,
            "gate_d_dynamics_fidelity": gate_d,
            "environment": environment_provenance(),
            "runtime": runtime_provenance(),
        }
        write_json_atomic(payload, out_dir / "p8_3_gates.json")
        print(f"wrote {out_dir / 'p8_3_gates.json'}", flush=True)
        return 0

    arms = tuple(part.strip() for part in str(args.arms).split(",") if part.strip())
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms {unknown}; the declared ones are {list(ARMS)}")
    seeds = tuple(int(part) for part in str(args.seeds).split(",") if part.strip())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    book = ActionCodebook.fit(
        data["actions"],
        int(args.codebook),
        seed=CODEBOOK_SEED,
        iterations=CODEBOOK_ITERATIONS,
        subsample=CODEBOOK_SUBSAMPLE,
    )
    print(f"codebook K={book.size} digest {book.digest[:16]}... device {device}", flush=True)

    result = train_and_evaluate(
        data,
        spans,
        book,
        arms=arms,
        seeds=seeds,
        steps=int(args.steps),
        out_dir=out_dir,
        device=device,
        log_every=int(args.log_every),
    )
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "task": "P8.3",
        "stage": "train",
        "dataset": DATASET_NAME,
        "dataset_sha256": _sha256_file(dataset_path),
        "codebook_size": book.size,
        "codebook_digest": book.digest,
        "declared_gradient_steps": int(args.steps),
        "batch_sizes": {"bc": BC_BATCH_WINDOWS, "bc_top10": BC_BATCH_WINDOWS,
                        "iql": IQL_BATCH_TRANSITIONS},
        "context_length": CONTEXT_LENGTH,
        "eval_episodes_per_seed": EVAL_EPISODES_PER_SEED,
        "seeds": list(seeds),
        "published_scores": PUBLISHED_SCORES,
        "environment": environment_provenance(),
        "runtime": runtime_provenance(),
        **result,
    }
    name = f"p8_3_train_k{book.size}_{int(args.steps)}.json"
    write_json_atomic(payload, out_dir / name)
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {out_dir / name}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
