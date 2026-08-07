"""Record each MAPPO checkpoint's global metric key set in the checkpoint itself.

Contract C8: ``MAPPOAgent._build_global_features`` orders the centralised critic's
global feature block by the keys of ``info["metrics"]``, so a checkpoint is only
interpretable alongside the metric set it was trained with.  That set was never
stored -- a shipped checkpoint's keys are ``steps_done`` and ``learner`` only -- so a
loaded agent re-freezes from whatever env it is handed and a **same-width metric swap
is completely silent**.  ``docs/patches/mappo_metric_keys_guard.patch`` adds the
load-time check; this script backfills the field on checkpoints that predate it.

A checkpoint is a plain ``dict``, so this needs no frozen-file change.

    python -m offline.migrate_mappo_checkpoints                  # dry run (default)
    python -m offline.migrate_mappo_checkpoints --apply          # rewrite in place

DERIVATION, NOT TRANSCRIPTION
-----------------------------
The key set is derived from the **recorded training config** of each checkpoint, never
typed as a literal and never taken from whichever env happens to be convenient.  A
migration that writes a same-count-but-wrong key set would satisfy the load-time
presence check while recording nonsense, turning the guard into a rubber stamp: the
self-consistency check below cannot see it (the count is right) and only a later
collection would fail.  **The derivation source is the control that prevents it**, so
this module contains no metric name anywhere -- a property a test asserts by scanning
this file's source.

The env-side derivation itself is ``agent.MAPPOAgent.env_global_metric_keys``, imported
rather than reimplemented.  Two copies of a key-set derivation are exactly where a
guard and the data it guards drift apart with nothing failing.

⚠️ THE RUN NAME IS THE **INNER** DIRECTORY, NOT THE OUTER ONE
-------------------------------------------------------------
Checkpoints live at ``<checkpoint_dir>/<run name>/<env id>__<agent>__seed<N>.pt``, and
those two directory levels do not have to agree: ``p2_1_mappo_nominal_1000.json``
declares ``checkpoint_dir: output/checkpoints/p2_1_mappo_nominal_500``, so its
checkpoints sit at ``.../p2_1_mappo_nominal_500/p2_1_mappo_nominal_1000/``.  Keying on
the outer directory would read the **wrong config** for a third of the corpus.

**The coincidence that both configs derive the same metric set is not protection.**  It
makes the mistake invisible rather than harmless: every checkpoint would be stamped with
a plausible key set obtained from a config that did not train it, the self-consistency
check would pass, and the error would only ever surface if the two configs later
diverged -- at which point 20 checkpoints would already carry a silent lie.  Correct by
construction is the only available guarantee here, because there is no witness in the
file that could catch it after the fact.

VERIFICATION
------------
Three checks, per BRIEF_08 §3:

1. **Self-consistency against the checkpoint's own witness.**
   ``learner["global_feature_dim"]`` is stored, and equals ``2 + len(metric keys)``
   because ``_build_global_features`` emits ``[step/max_steps, vehicle_count] + one
   float per key``.  ``len(derived) == global_feature_dim - 2`` therefore catches a
   wrong key *count* with no external source of truth.
2. **Round-trip.** Every rewritten file is re-read and its recorded keys compared.
3. **Derivation.** As above.

SAFETY
------
Every checkpoint is loaded and validated, and the whole plan built, **before the first
byte is written** -- the filesystem-mutation barrier this project adopted after P1 NB2
and P2.0.  Writes go via a temp file plus ``os.replace``.  ``--apply`` additionally
refuses to run unless :data:`BACKUP_DIRNAME` exists beside the checkpoint root, because
an in-place rewrite of the only copy has no barrier at all and ``datasets/``'s manifests
record the *pre*-migration ``checkpoint_sha256``.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

#: Field added to each checkpoint dict.
METRIC_KEYS_FIELD = "global_metric_keys"

#: ``_build_global_features`` prepends ``step / max_steps`` and ``vehicle_count``.
GLOBAL_FEATURE_PREFIX_WIDTH = 2

#: Required beside the checkpoint root before ``--apply`` will write anything.
BACKUP_DIRNAME = "checkpoints.pre_c8_migration"

DEFAULT_CHECKPOINT_ROOT = Path("output/checkpoints")
DEFAULT_CONFIG_DIR = Path("experiments/configs")

EnvFactory = Callable[[str, str], Any]

__all__ = [
    "BACKUP_DIRNAME",
    "METRIC_KEYS_FIELD",
    "CheckpointPlan",
    "MigrationError",
    "build_plan",
    "apply_plan",
    "discover_checkpoints",
    "main",
]


class MigrationError(RuntimeError):
    """Raised when a checkpoint cannot be migrated safely. Nothing is written."""


@dataclass(frozen=True)
class CheckpointPlan:
    """What the migration intends to do to one checkpoint, before it does anything."""

    path: Path
    run_name: str
    env_id: str
    metric_keys: tuple[str, ...]
    global_feature_dim: int
    already_correct: bool


def _default_env_factory(config_dir: Path) -> EnvFactory:
    """Build an env from the run's **recorded training config**.

    This is the derivation source the whole migration rests on: the config named by
    the checkpoint's own parent directory, never a literal and never an env that
    happens to be lying around.
    """

    def make(run_name: str, env_id: str) -> Any:
        config_path = config_dir / f"{run_name}.json"
        if not config_path.is_file():
            raise MigrationError(
                f"no training config at {config_path} for run {run_name!r}. The run "
                "name is the checkpoint's PARENT directory; if that is not a config "
                "name, the tree layout changed and the derivation source is unsafe."
            )
        from experiments.config import load_config
        from experiments.envs import make_env

        config = load_config(config_path)
        for spec in config.environments:
            if spec.id == env_id:
                return make_env(spec)
        raise MigrationError(
            f"{config_path} defines no environment with id {env_id!r} "
            f"(has {[s.id for s in config.environments]})"
        )

    return make


def _load_payload(path: str | Path) -> dict[str, Any]:
    """Read a checkpoint dict.

    ``weights_only=False`` is required and safe here: the payload is a plain dict of
    our own making, read from a local path the operator named, and the whole point is
    to see every key rather than tensors alone.
    """
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise MigrationError(
            f"{path}: expected a checkpoint dict, got {type(payload).__name__}"
        )
    return payload


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    """Write via a temp file in the same directory plus ``os.replace``.

    A crash mid-write must not leave a truncated checkpoint where a valid one was.
    """
    import torch

    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".migrate-",
                                        suffix=".pt.tmp")
    os.close(handle)
    try:
        torch.save(payload, tmp_name)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def discover_checkpoints(root: str | Path) -> list[Path]:
    """Every ``*.pt`` under *root*, sorted, deepest-path-stable."""
    return sorted(Path(root).rglob("*.pt"))


def checkpoint_run_name(path: str | Path) -> str:
    """The run name that trained this checkpoint: its **parent directory** name.

    Not the grandparent -- see the module docstring for why the two can disagree and
    why the coincidence that they usually derive the same metric set is not protection.
    """
    return Path(path).parent.name


def checkpoint_env_id(path: str | Path) -> str:
    """The env id encoded in the filename, e.g. ``cf_hz1x1__mappo__seed101.pt``."""
    return Path(path).name.split("__", 1)[0]


def _derive_metric_keys(
    env_factory: EnvFactory, run_name: str, env_id: str
) -> tuple[str, ...]:
    """Build the run's env, ask the SHARED derivation, and always close the env."""
    try:
        from agent.MAPPOAgent import env_global_metric_keys
    except ImportError as exc:  # pragma: no cover - depends on the patch being applied
        raise MigrationError(
            "agent.MAPPOAgent.env_global_metric_keys is missing, so "
            "docs/patches/mappo_metric_keys_guard.patch has not been applied. Apply "
            "it first: this migration writes the field that patch's load-time check "
            "reads, and deriving the keys by any other route here would be a second "
            "implementation of the one thing that must not have two."
        ) from exc

    env = env_factory(run_name, env_id)
    try:
        keys = env_global_metric_keys(env)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    if keys is None:
        raise MigrationError(
            f"{run_name}/{env_id}: this env exposes no metrics pipeline, so the "
            "metric key set cannot be derived. Refusing to record an empty set, "
            "which would be indistinguishable from a genuinely metric-free run."
        )
    return tuple(keys)


def build_plan(
    checkpoints: Sequence[Path],
    env_factory: EnvFactory,
) -> list[CheckpointPlan]:
    """Validate every checkpoint and return the full plan. Writes nothing.

    Every failure mode raises here, before :func:`apply_plan` touches anything, so a
    bad checkpoint anywhere in the set leaves *all* of them untouched rather than the
    first N rewritten.
    """
    cache: dict[tuple[str, str], tuple[str, ...]] = {}
    plans: list[CheckpointPlan] = []

    for raw_path in checkpoints:
        path = Path(raw_path)
        run_name = checkpoint_run_name(path)
        env_id = checkpoint_env_id(path)
        payload = _load_payload(path)

        learner = payload.get("learner")
        if not isinstance(learner, dict) or "global_feature_dim" not in learner:
            raise MigrationError(
                f"{path}: no learner['global_feature_dim'], so the derived key count "
                "has no witness to be checked against. Refusing to migrate blind."
            )
        global_feature_dim = int(learner["global_feature_dim"])

        cache_key = (run_name, env_id)
        if cache_key not in cache:
            cache[cache_key] = _derive_metric_keys(env_factory, run_name, env_id)
        metric_keys = cache[cache_key]

        expected = global_feature_dim - GLOBAL_FEATURE_PREFIX_WIDTH
        if len(metric_keys) != expected:
            raise MigrationError(
                f"{path}: derived {len(metric_keys)} metric key(s) "
                f"{list(metric_keys)} from run {run_name!r} / env {env_id!r}, but the "
                f"checkpoint's own learner['global_feature_dim'] = "
                f"{global_feature_dim} implies {expected} "
                f"(global_feature_dim - {GLOBAL_FEATURE_PREFIX_WIDTH}). The recorded "
                "config and this checkpoint disagree; migrating would stamp a key set "
                "the critic was not trained with."
            )

        already_correct = False
        if payload.get(METRIC_KEYS_FIELD) is not None:
            existing = tuple(str(k) for k in payload[METRIC_KEYS_FIELD])
            if sorted(existing) != sorted(metric_keys):
                raise MigrationError(
                    f"{path}: already records {METRIC_KEYS_FIELD}={list(existing)}, "
                    f"but this run derives {list(metric_keys)}. Refusing to "
                    "overwrite: one of the two is wrong and the migration cannot "
                    "tell which."
                )
            already_correct = True

        plans.append(
            CheckpointPlan(
                path=path,
                run_name=run_name,
                env_id=env_id,
                metric_keys=metric_keys,
                global_feature_dim=global_feature_dim,
                already_correct=already_correct,
            )
        )

    return plans


def apply_plan(plans: Sequence[CheckpointPlan]) -> int:
    """Write the plan, verifying each file round-trips. Returns files rewritten."""
    written = 0
    for plan in plans:
        if plan.already_correct:
            continue

        payload = _load_payload(plan.path)
        payload[METRIC_KEYS_FIELD] = list(plan.metric_keys)
        _atomic_save(payload, plan.path)

        reloaded = _load_payload(plan.path)
        recorded = reloaded.get(METRIC_KEYS_FIELD)
        if recorded is None or tuple(str(k) for k in recorded) != plan.metric_keys:
            raise MigrationError(
                f"{plan.path}: round-trip verification failed -- wrote "
                f"{list(plan.metric_keys)} but read back {recorded!r}."
            )
        written += 1
    return written


def main(
    argv: Sequence[str] | None = None,
    *,
    env_factory: EnvFactory | None = None,
) -> int:
    """Entry point; returns a process exit code.

    *env_factory* is an injection seam for tests, defaulting to the real
    config-driven construction.  It is a keyword argument rather than a CLI flag on
    purpose: a production tool must not carry a switch whose only caller is a test.
    The seam sits at env *construction*, never at key derivation, so a test still
    exercises the same ``env_global_metric_keys`` the load-time guard uses.
    """
    parser = argparse.ArgumentParser(
        prog="python -m offline.migrate_mappo_checkpoints",
        description="Record each MAPPO checkpoint's global metric key set (contract C8).",
    )
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="rewrite the checkpoints; without it this is a dry run that writes nothing",
    )
    args = parser.parse_args(argv)

    root = Path(args.checkpoint_root)
    if not root.is_dir():
        print(f"ERROR: --checkpoint-root {root} is not a directory", flush=True)
        return 2

    checkpoints = discover_checkpoints(root)
    if not checkpoints:
        # Not a success: a migration that finds nothing has almost certainly been
        # pointed at the wrong tree, and exiting 0 would report that as done.
        print(f"ERROR: no *.pt found under {root}", flush=True)
        return 2

    if args.apply:
        backup = root.parent / BACKUP_DIRNAME
        if not backup.is_dir():
            print(
                f"REFUSING TO APPLY: {backup} does not exist.\n"
                f"  This rewrites all {len(checkpoints)} checkpoints IN PLACE and it "
                "is the only irreversible step in P2.6; datasets/'s manifests record "
                "the pre-migration checkpoint_sha256.\n"
                f"  Run:  cp -a {root} {backup}",
                flush=True,
            )
            return 2

    factory = env_factory or _default_env_factory(Path(args.config_dir))

    try:
        plans = build_plan(checkpoints, factory)
    except MigrationError as exc:
        print(f"ABORT (nothing written): {exc}", flush=True)
        return 1

    pending = [p for p in plans if not p.already_correct]
    for plan in plans:
        state = "already correct" if plan.already_correct else "to migrate"
        print(
            f"  {plan.path}  run={plan.run_name} env={plan.env_id} "
            f"gfd={plan.global_feature_dim} keys={list(plan.metric_keys)}  [{state}]",
            flush=True,
        )
    print(
        f"{len(plans)} checkpoint(s) validated, {len(pending)} to migrate.", flush=True
    )

    if not args.apply:
        print("DRY RUN -- nothing written. Re-run with --apply.", flush=True)
        return 0

    try:
        written = apply_plan(plans)
    except MigrationError as exc:
        print(f"ABORT during write: {exc}", flush=True)
        return 1

    if written != len(pending):
        # Same discipline PROJECT_PLAN §7 imposes on campaign scripts: assert that the
        # work completed equals the work requested, rather than trusting the loop ran.
        print(
            f"ERROR: wrote {written} but {len(pending)} were pending", flush=True
        )
        return 1
    print(f"migrated {written} checkpoint(s).", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
