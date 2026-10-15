"""P9.1 (``BRIEF_46``; ``PREREGISTRATION.md`` A27 (b)-(e), (g); tag ``v2.8-prereg-a27``): the corrective round's campaign.

FORMATS (every record carries its version)
------------------------------------------
* ``p9.1-run/1.0`` -- ONE write-once record per training run, ``output/p9_1_runs/training/<scenario>/<tier>__<arm>__
  seed<s>.json``: the recipe (base method, batch, targets, declared steps), the run's ``seconds`` and
  ``gradient_steps`` (an UNCONTROLLED clock, A27(a)), the checkpoint's canonical and file digests, the loss curve's
  window means and plateau flag, and the trainer's diagnostics.

No record here stores a trajectory, so no alignment convention applies to them; the campaign's inputs are contract C6
v1.1 corpora read through the registered loaders, unchanged.

WHAT THIS MODULE IS
-------------------
The campaign of A27's points 1, 2, 3, 4 and 6 (``BRIEF_46`` 3.4): the cells A27 declares (:func:`training_cells`, 51
cells x 5 seeds = 255 runs), where their files live (:func:`checkpoint_path`, :func:`run_record_path`), the write
barrier every write goes through (:func:`assert_target`), and the status of every run COMPUTED FROM DISK
(:func:`run_status`, :func:`training_status`) -- never claimed.  The arms are ``offline.a27_arms``'s; the tiers are
``method_tier_grid``'s and ``tier_sweep``'s registries, A27's included.

THE BARRIER
-----------
``output/`` is gitignored and has no backup.  This module writes ONLY under ``output/p9_1/`` and ``output/p9_1_runs/``
(``BRIEF_46``'s roots): :func:`assert_target` resolves the path -- so a ``..``, a relative path or a symlink cannot pass a
string comparison -- and refuses anything outside those two directories.  Nothing is ever deleted or moved out of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "CLOCK_LABEL",
    "LABELS",
    "RUN_DIR",
    "RUNS_DIR",
    "RUN_RECORD_FORMAT_VERSION",
    "Cell",
    "Roots",
    "assert_target",
    "checkpoint_path",
    "recipe_fields",
    "reference_cells",
    "run_record_path",
    "run_status",
    "train",
    "train_run",
    "training_cells",
    "training_runs",
    "training_status",
]

RUN_RECORD_FORMAT_VERSION = "p9.1-run/1.0"

#: The two directories this task creates under ``output/`` (``BRIEF_46``, roots).
RUN_DIR = "p9_1"
RUNS_DIR = "p9_1_runs"

#: A27(a)'s label of each point this task runs, verbatim; every artifact carries its point's label.
LABELS: Mapping[str, str] = {
    "A27(b)": "control added after results",
    "A27(c)": "bug fix, with results before and after",
    "A27(d)": "data-collection bug fix, results before and after",
    "A27(e)": "test registered after results",
    "A27(g)": (
        "the secondary registered in A8(b) on 2026-08-12 after P4.4's primary results, executed after all primary "
        "results, one ceiling for all four arms as A8(b) requires, its `mix50` part beyond A8(b) and labelled so"
    ),
}

#: A27(c): ``iql_t`` on hz1x1's eight registered tiers.
HZ1X1_IQL_T_TIERS: tuple[str, ...] = (
    "random", "fixedtime", "maxpressure", "mappo500", "mappo1000", "mix33", "mix50", "mix67",
)
#: The grid's four registered tiers (``tier_sweep.TIER_ORDER``).
GRID_TIERS: tuple[str, ...] = ("mappo1000", "maxpressure", "fixedtime", "random")
#: A27(g): A8(b)'s tier, and ``mix50`` beyond A8(b).
BUDGET_TIERS: tuple[str, ...] = ("mappo1000", "mix50")


@dataclass(frozen=True)
class Roots:
    """``repo_root``: the checkout whose ``docs/`` is read; the other three are the MAIN tree's, by absolute path."""

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


@dataclass(frozen=True)
class Cell:
    """One arm on one tier of one scenario, and the A27 point that declares it."""

    scenario: str
    tier: str
    arm: str
    point: str

    @property
    def key(self) -> str:
        """``"<arm>@<tier>"`` -- the arm string every evaluation record carries."""
        return f"{self.arm}@{self.tier}"


def training_cells() -> tuple[Cell, ...]:
    """Every cell A27 (b), (c), (d), (e), (g) trains, in a fixed order: 51 cells."""
    cells: list[Cell] = []
    cells += [Cell("hz1x1", tier, "iql_t", "A27(c)") for tier in HZ1X1_IQL_T_TIERS]
    cells += [Cell("hz1x1", "random_fresh", arm, "A27(d)") for arm in ("dt", "bc", "bc_top10", "iql", "iql_t")]
    cells += [Cell("hz1x1", "mix_m1000_m500", arm, "A27(e)") for arm in ("dt", "bc", "bc_top10", "iql", "iql_t")]
    cells += [
        Cell("hz1x1", tier, arm, "A27(g)")
        for tier in BUDGET_TIERS
        for arm in ("dt_100k", "bc_100k", "bc_top10_100k", "iql_100k")
    ]
    cells += [
        Cell("grid4x4", tier, arm, "A27(b)")
        for tier in GRID_TIERS
        for arm in ("bc_m", "bc_top10_m", "bc_top10_perix_m", "iql_m")
    ]
    cells += [Cell("grid4x4", tier, "iql_t", "A27(c)") for tier in GRID_TIERS]
    cells += [
        Cell("grid4x4", "random_fresh", arm, "A27(d)")
        for arm in ("dt_nomix", "bc_m", "bc_top10_m", "bc_top10_perix_m", "iql_m")
    ]
    return tuple(cells)


def training_runs() -> tuple[tuple[Cell, int], ...]:
    """Every (cell, training seed): 255 runs."""
    from offline.dt_gate import TRAINING_SEEDS

    return tuple((cell, int(seed)) for cell in training_cells() for seed in TRAINING_SEEDS)


def reference_cells() -> tuple[Cell, ...]:
    """A27(d)'s fresh random reference, the ``behaviour`` cell of ``random_fresh`` on each scenario (evaluated only)."""
    return (Cell("hz1x1", "random_fresh", "behaviour", "A27(d)"), Cell("grid4x4", "random_fresh", "behaviour", "A27(d)"))


def checkpoint_path(roots: Roots, scenario: str, tier: str, arm: str, seed: int) -> Path:
    """Where a run's checkpoint lives, under the name the route-(i) CLI resolves it by.

    hz1x1: ``method_tier_grid``'s ``<tier>_<arm>_seed<s>.pt``; grid: ``tier_sweep._checkpoint_name``'s
    ``grid4x4_<tier>_<arm>_seed<s>.pt`` -- so ``tier_sweep evaluate --checkpoint-dir`` finds it unchanged.
    """
    base = Path(roots.runs_root) / "checkpoints" / scenario
    if scenario == "hz1x1":
        return base / f"{tier}_{arm}_seed{int(seed)}.pt"
    if scenario == "grid4x4":
        from offline.tier_sweep import _checkpoint_name

        return base / _checkpoint_name(tier, arm, int(seed))
    raise ValueError(f"unknown scenario {scenario!r}")


def run_record_path(roots: Roots, scenario: str, tier: str, arm: str, seed: int) -> Path:
    """The run's write-once ``p9.1-run/1.0`` record."""
    return Path(roots.runs_root) / "training" / scenario / f"{tier}__{arm}__seed{int(seed)}.json"


def assert_target(path: str | Path, roots: Roots) -> Path:
    """Refuse any write that does not resolve under ``output/p9_1/`` or ``output/p9_1_runs/``; return the resolved path.

    Resolution comes first, so ``..``, a relative path or a symlink is judged by where it LANDS.
    """
    resolved = Path(path).resolve()
    allowed = (Path(roots.run_root).resolve(), Path(roots.runs_root).resolve())
    if not any(resolved == root or root in resolved.parents for root in allowed):
        raise PermissionError(
            f"{path} resolves to {resolved}, outside output/{RUN_DIR}/ and output/{RUNS_DIR}/; P9.1 writes nowhere else "
            "(BRIEF_46, roots)"
        )
    return resolved


def recipe_fields(cell: Cell) -> dict[str, Any]:
    """The recipe a run of *cell* trains under, as its record states it -- what resume compares against."""
    from offline.a27_arms import recipe

    r = recipe(cell.scenario, cell.arm)
    return {"base": r.base, "batch": r.batch, "target_dropout": r.target_dropout, "declared_steps": r.steps,
            "registered_in": cell.point}


def run_status(roots: Roots, cell: Cell, seed: int) -> str:
    """``"done"`` or ``"missing"`` -- anything in between is REFUSED, never retrained and never skipped.

    Done means: the checkpoint exists, its write-once record exists, the record's recipe is the one this cell trains
    under, and the checkpoint's canonical digest is the one the record wrote.  A ``.partial`` file, a checkpoint
    without its record, a record without its checkpoint, a different recipe or a different digest is a record of
    something going wrong, and is left on disk for a person to read.
    """
    from offline.method_tier_grid import canonical_digest_of

    checkpoint = checkpoint_path(roots, cell.scenario, cell.tier, cell.arm, seed)
    record = run_record_path(roots, cell.scenario, cell.tier, cell.arm, seed)
    partial = checkpoint.with_name(checkpoint.name + ".partial")
    where = f"{cell.scenario}:{cell.key} seed {seed}"
    if partial.exists():
        raise ValueError(f"{where}: {partial} exists -- an interrupted run; refusing to retrain over it or skip it")
    if not checkpoint.exists() and not record.exists():
        return "missing"
    if not checkpoint.exists() or not record.exists():
        raise ValueError(f"{where}: the checkpoint and its record must both exist; refusing a half-written run")
    payload = json.loads(record.read_text(encoding="utf-8"))
    planned = recipe_fields(cell)
    recorded = {key: payload.get(key) for key in planned}
    if recorded != planned:
        raise ValueError(f"{where}: the record's recipe {recorded} is not this cell's {planned}; refusing it")
    if payload.get("canonical_digest") != canonical_digest_of(checkpoint):
        raise ValueError(f"{where}: the checkpoint's weights are not the ones its record wrote; refusing it")
    return "done"


def training_status(roots: Roots) -> dict[str, Any]:
    """Every cell's status and the run counts, recomputed from disk on every call."""
    from offline.dt_gate import TRAINING_SEEDS

    by_cell: dict[str, str] = {}
    done = 0
    for cell in training_cells():
        states = [run_status(roots, cell, seed) for seed in TRAINING_SEEDS]
        done += sum(state == "done" for state in states)
        by_cell[f"{cell.scenario}:{cell.key}"] = (
            "done" if all(s == "done" for s in states) else "missing" if all(s == "missing" for s in states) else "partial"
        )
    declared = len(training_runs())
    return {"runs_declared": declared, "runs_done": done, "complete": done == declared, "by_cell": by_cell}


# ----------------------------------------------------------------------
# Training (BRIEF_46 3.4 `train`; plan D6): direct calls to the registered trainers, on the CLIs' own inputs
# ----------------------------------------------------------------------

#: A27(a): every training run's seconds are an UNCONTROLLED clock -- recorded, labelled, and never a benchmark.
CLOCK_LABEL = "uncontrolled; no Best Performance regime, no canary; not a benchmark"


def _hz1x1_inputs(roots: Roots, tier: str) -> Any:
    """``method_tier_grid.training_inputs`` -- the inputs ``method_tier_grid train`` builds, A27's checks included."""
    from offline.method_tier_grid import tier_spec, training_inputs

    return training_inputs(tier_spec(tier), roots.corpus_root)


def _grid_inputs(roots: Roots, tier: str) -> Any:
    """``tier_sweep.baselines_inputs`` (BC, %BC, IQL) -- the inputs ``tier_sweep train-baselines`` builds -- after A27(d)'s
    distinct-schedule check on ``random_fresh``, which the CLI runs first too."""
    from offline.tier_sweep import _assert_random_fresh_is_distinct, baselines_inputs

    _assert_random_fresh_is_distinct(tier, roots.corpus_root)
    return baselines_inputs(tier, roots.corpus_root)


def _write_once(path: Path, payload: Mapping[str, Any], roots: Roots) -> Path:
    """Write a JSON record that must not exist yet (exclusive create), through the barrier."""
    target = assert_target(path, roots)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return target


def train_run(roots: Roots, cell: Cell, seed: int, inputs: Any, *, device: Any, log_every: int = 0) -> dict[str, Any]:
    """Train ONE run of *cell* into ``<checkpoint>.partial``, move it into place, write its ``p9.1-run/1.0`` record.

    The trainer and every argument are the CLI's (``method_tier_grid._run_train`` on hz1x1, ``tier_sweep``'s
    ``_run_train_baselines`` / ``_run_train`` on the grid) with the recipe's batch, targets and budget; T-train-args
    tests compare the two calls keyword for keyword and tensor for tensor.  All three paths pass the barrier and are
    checked absent before the first gradient step.
    """
    import hashlib
    import os

    import offline.method_tier_grid as grid
    import offline.offline_baselines as ob
    import offline.tier_sweep as ts
    from offline.a27_arms import recipe as recipe_of
    from offline.dt_gate import BATCH_SIZE, runtime_provenance

    recipe = recipe_of(cell.scenario, cell.arm)
    final = checkpoint_path(roots, cell.scenario, cell.tier, cell.arm, seed)
    partial = final.with_name(final.name + ".partial")
    record_path = run_record_path(roots, cell.scenario, cell.tier, cell.arm, seed)
    for path in (final, partial, record_path):
        assert_target(path, roots)
        if path.exists():
            raise FileExistsError(f"{path} exists; a run is trained once and never over a file on disk")
    final.parent.mkdir(parents=True, exist_ok=True)
    # The CLI's own A27 provenance keys with the CLI's values (T-train-args compares them), plus the cell's point.
    a27_fields = {"arm": recipe.name, "base_method": recipe.base, "target_dropout": recipe.target_dropout,
                  "registered_in": recipe.registered_in, "point": cell.point, "campaign": "P9.1 (BRIEF_46)"}
    iql_extra = {} if recipe.target_dropout in (None, "active") else {"target_dropout": recipe.target_dropout}
    extra_record: dict[str, Any] = {}
    print(f"TRAIN {cell.scenario}:{cell.key} seed {seed} -> {final}", flush=True)
    if cell.scenario == "hz1x1":
        provenance = {**inputs.provenance, **a27_fields}
        group = inputs.group
        if recipe.base == "dt":
            from offline.dt_gate import train_dt

            result = train_dt(
                inputs.full_batch, state_dim=group[0], n_actions=group[1], seed=int(seed),
                declared_gradient_steps=int(recipe.steps), raise_to=None, context_length=grid.CONTEXT_LENGTH,
                batch_size=BATCH_SIZE, device=device, checkpoint_path=partial, stats=inputs.dataset.stats,
                scenario_id=inputs.scenario_id, target_rtg=float(inputs.spec.target_rtg),
                rtg_scale=float(inputs.spec.rtg_scale), provenance=provenance, log_every=log_every,
            )
        elif recipe.base == "iql":
            result = grid.train_iql(
                inputs.table(), state_dim=group[0], n_actions=group[1], seed=int(seed),
                declared_gradient_steps=int(recipe.steps), batch_size=int(recipe.batch), device=device,
                checkpoint_path=partial, stats=inputs.dataset.stats, scenario_id=inputs.scenario_id,
                provenance=provenance, log_every=log_every, **iql_extra,
            )
        else:
            result = grid.train_bc(
                inputs.top_batch if recipe.base == "bc_top10" else inputs.full_batch, state_dim=group[0],
                n_actions=group[1], seed=int(seed), method=recipe.base, declared_gradient_steps=int(recipe.steps),
                batch_size=int(recipe.batch), device=device, checkpoint_path=partial, stats=inputs.dataset.stats,
                scenario_id=inputs.scenario_id, provenance=provenance, log_every=log_every,
            )
    elif cell.scenario == "grid4x4":
        if recipe.base == "dt_nomix":
            parts = inputs
            protected = ts.protected_roots_from([Path(roots.output_root) / name for name in ("p5_1", "p5_2")])
            result = ts.train_tier_dt(
                parts["stack"](), parts["index"], tier=cell.tier, method="dt_nomix", seed=int(seed),
                adjacency=parts["adjacency"], prompts=parts["prompts"], stats=parts["dataset"].stats,
                state_dim=int(parts["index"].state_dim), n_actions=int(parts["index"].n_actions),
                gradient_steps=int(recipe.steps), batch_size=ts.JOINT_BATCH_SIZE, device=device,
                checkpoint_path=partial, protected=protected,
                provenance={"runtime": runtime_provenance(), **a27_fields}, deterministic=False,
                log_every=log_every,
            )
            extra_record["deterministic"] = False
        else:
            provenance = {**inputs.provenance, **a27_fields, "runtime": runtime_provenance()}
            group = inputs.group
            if recipe.base == "iql":
                result = ob.train_iql(
                    inputs.table(), state_dim=group[0], n_actions=group[1], seed=int(seed),
                    declared_gradient_steps=int(recipe.steps), batch_size=int(recipe.batch), device=device,
                    checkpoint_path=partial, stats=inputs.dataset.stats, scenario_id=ts.SCENARIO_ID,
                    provenance=provenance, log_every=log_every, **iql_extra,
                )
            else:
                result = ob.train_bc(
                    inputs.batches[recipe.base], state_dim=group[0], n_actions=group[1], seed=int(seed),
                    method=recipe.base, declared_gradient_steps=int(recipe.steps), batch_size=int(recipe.batch),
                    device=device, checkpoint_path=partial, stats=inputs.dataset.stats, scenario_id=ts.SCENARIO_ID,
                    provenance=provenance, log_every=log_every,
                )
    else:
        raise ValueError(f"unknown scenario {cell.scenario!r}")

    assert_target(final, roots)
    os.replace(partial, final)
    losses = list(getattr(result, "losses", ()))
    window_means = getattr(result, "window_means", None)
    plateaued = getattr(result, "plateaued", None)
    if window_means is None and losses:
        window_means, plateaued = ob._curve(losses, len(losses))
    record = {
        "format_version": RUN_RECORD_FORMAT_VERSION,
        "scenario": cell.scenario,
        "tier": cell.tier,
        "arm": cell.arm,
        "seed": int(seed),
        **recipe_fields(cell),
        "label": LABELS[cell.point],
        "gradient_steps": int(result.gradient_steps),
        "seconds": float(result.seconds),
        "clock": CLOCK_LABEL,
        "checkpoint": str(final),
        "canonical_digest": grid.canonical_digest_of(final),
        "file_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
        "window_means": [float(v) for v in (window_means or ())],
        "plateaued": None if plateaued is None else bool(plateaued),
        "final_loss": float(losses[-1]) if losses else None,
        "diagnostics": getattr(result, "diagnostics", None),
        "device": str(device),
        "thread_regime": ob.thread_regime(),
        **extra_record,
    }
    _write_once(record_path, record, roots)
    print(f"  done in {record['seconds']:.1f}s ({CLOCK_LABEL})", flush=True)
    return record


def train(
    roots: Roots,
    *,
    cells: Any = None,
    device: Any = None,
    seeds: Any = None,
    log_every: int = 0,
) -> dict[str, int]:
    """Train every MISSING run of *cells* (default: the whole manifest), serially, one tier's inputs built once.

    Every run's status is computed from disk FIRST and any damaged run refuses the whole call before a gradient step
    (:func:`run_status`); done runs are skipped, never retrained.  Resumable by construction: an interrupted call leaves
    a ``.partial`` that the next call refuses until a person has looked at it.
    """
    import torch

    from offline.dt_gate import TRAINING_SEEDS

    chosen = list(training_cells()) if cells is None else list(cells)
    seed_list = tuple(int(s) for s in (TRAINING_SEEDS if seeds is None else seeds))
    statuses = {(cell, seed): run_status(roots, cell, seed) for cell in chosen for seed in seed_list}
    todo = [(cell, seed) for (cell, seed), state in statuses.items() if state == "missing"]
    resolved = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cache: dict[tuple[str, str, bool], Any] = {}
    for cell, seed in todo:
        dt_grid = cell.scenario == "grid4x4" and cell.arm == "dt_nomix"
        key = (cell.scenario, cell.tier, dt_grid)
        if key not in cache:
            if cell.scenario == "hz1x1":
                cache[key] = _hz1x1_inputs(roots, cell.tier)
            elif dt_grid:
                from offline.tier_sweep import _assert_random_fresh_is_distinct, tier_parts

                _assert_random_fresh_is_distinct(cell.tier, roots.corpus_root)
                cache[key] = tier_parts(cell.tier, roots.corpus_root)
            else:
                cache[key] = _grid_inputs(roots, cell.tier)
        train_run(roots, cell, seed, cache[key], device=resolved, log_every=log_every)
    return {"trained": len(todo), "skipped_done": sum(state == "done" for state in statuses.values())}


# ----------------------------------------------------------------------
# Evaluation (BRIEF_46 3.4 `evaluate`; plan D4, D5): route (i), route (ii), and P3a's recording
# ----------------------------------------------------------------------

#: The engine seed every held-out episode is rolled with (the registered protocol: P4.4-P5.2, P8.4b).
ENGINE_SEED = 1000

#: Scenario name -> the materialised-draw key of its held-out pool.
SCENARIO_KEYS: Mapping[str, str] = {"hz1x1": "cityflow1x1", "grid4x4": "cityflow_grid4x4"}

#: ``"<prefix>_training.json"`` of the hz1x1 aggregates route (i) reads, per declared budget.
AGGREGATE_PREFIXES: Mapping[int, str] = {40_000: "p9_1", 100_000: "p9_1_100k"}

ACTIONS_FORMAT_VERSION = "p9.1-actions/1.0"


def _write_json(path: Path, payload: Mapping[str, Any], roots: Roots) -> Path:
    """A DERIVED record (rebuilt from disk every time), written atomically through the barrier."""
    import os
    import tempfile

    target = assert_target(path, roots)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def write_training_aggregates(roots: Roots) -> list[Path]:
    """Route (i)'s hz1x1 input: ``output/p9_1/p9_1_training.json`` and ``p9_1_100k_training.json`` in
    ``method_tier_grid``'s own training format, one per declared budget, rebuilt from the done runs' records on disk.

    ``method_tier_grid evaluate`` refuses a record whose ``declared_gradient_steps`` is not the one it is asked for, so
    each budget gets its own file (``--artifact-prefix``).  A run enters only if :func:`run_status` says it is done.
    """
    from offline.dt_gate import TRAINING_SEEDS
    from offline.method_tier_grid import CONTEXT_LENGTH, TRAINING_FORMAT_VERSION, tier_spec

    by_budget: dict[int, list[dict[str, Any]]] = {steps: [] for steps in AGGREGATE_PREFIXES}
    for cell in training_cells():
        if cell.scenario != "hz1x1":
            continue
        fields = recipe_fields(cell)
        for seed in TRAINING_SEEDS:
            if run_status(roots, cell, seed) != "done":
                continue
            record = json.loads(run_record_path(roots, cell.scenario, cell.tier, cell.arm, seed).read_text(encoding="utf-8"))
            by_budget[int(fields["declared_steps"])].append({
                "tier": cell.tier,
                "method": cell.arm,
                "seed": int(seed),
                "gradient_steps": int(record["gradient_steps"]),
                "seconds": float(record["seconds"]),
                "checkpoint": str(record["checkpoint"]),
                "canonical_digest": str(record["canonical_digest"]),
                "target_rtg": float(tier_spec(cell.tier).target_rtg) if fields["base"] == "dt" else None,
                "base_method": fields["base"],
            })
    written = []
    for steps, prefix in AGGREGATE_PREFIXES.items():
        runs = sorted(by_budget[steps], key=lambda r: (r["tier"], r["method"], r["seed"]))
        payload = {
            "format_version": TRAINING_FORMAT_VERSION,
            "role": "P9.1's hz1x1 runs at one declared budget, rebuilt from their p9.1-run records for route (i)",
            "declared_gradient_steps": int(steps),
            "declared_in": "PREREGISTRATION.md A27 (tag v2.8-prereg-a27); offline/a27_arms.py's recipes",
            "raise_available": False,
            "context_length": int(CONTEXT_LENGTH),
            "seeds": list(TRAINING_SEEDS),
            "runs": runs,
        }
        written.append(_write_json(Path(roots.run_root) / f"{prefix}_training.json", payload, roots))
    return written


def route_i_command(roots: Roots, cell: Cell) -> tuple[str, list[str]]:
    """The module and argv of a cell's route-(i) evaluation: the CLI's own ``evaluate`` (or ``behaviour``)."""
    from offline.a27_arms import recipe

    work = str(Path(roots.runs_root) / "eval_i" / cell.scenario)
    common_hz = ["--corpus-root", str(roots.corpus_root), "--draws-root", str(roots.draws_root),
                 "--engine-seed", str(ENGINE_SEED), "--torch-threads", "1"]
    if cell.scenario == "hz1x1":
        argv = [*common_hz, "--scenario-key", SCENARIO_KEYS["hz1x1"], "--scenario-id", SCENARIO_KEYS["hz1x1"],
                "--out-dir", str(roots.run_root), "--work-dir", work,
                "--checkpoint-dir", str(Path(roots.runs_root) / "checkpoints" / "hz1x1")]
        if cell.arm == "behaviour":
            return "method_tier_grid", [*argv, "behaviour", "--tier", cell.tier]
        steps = int(recipe("hz1x1", cell.arm).steps)
        return "method_tier_grid", [*argv, "--steps", str(steps), "--artifact-prefix", AGGREGATE_PREFIXES[steps],
                                    "evaluate", "--tier", cell.tier, "--method", cell.arm]
    if cell.scenario == "grid4x4":
        steps = 40_000 if cell.arm == "behaviour" else int(recipe("grid4x4", cell.arm).steps)
        argv = [*common_hz, "--work-dir", work,
                "--checkpoint-dir", str(Path(roots.runs_root) / "checkpoints" / "grid4x4"),
                "--reuse-root", str(Path(roots.output_root) / "p5_1"),
                "--checksums", str(Path(roots.output_root) / "SHA256SUMS_p5_1.txt"),
                "--gradient-steps", str(steps), "--tier", cell.tier, "evaluate", "--method", cell.arm]
        return "tier_sweep", argv
    raise ValueError(f"unknown scenario {cell.scenario!r}")


def route_i_output(roots: Roots, cell: Cell) -> Path:
    """Where route (i) writes the cell: the CLI's own ``eval_<tier>_<method>.json`` in the work directory."""
    return Path(roots.runs_root) / "eval_i" / cell.scenario / f"eval_{cell.tier}_{cell.arm}.json"


def committed_from_route_i(
    payload: Mapping[str, Any], scenario: str, arm: str
) -> dict[tuple[str, str, int | None, int], float]:
    """Route (i)'s ``att_horizon`` per episode, keyed as ``att_rederivation`` keys a cell (``iql_correction``'s form)."""
    out: dict[tuple[str, str, int | None, int], float] = {}
    for episode in payload["episodes"]:
        if str(episode["arm"]) != arm:
            raise ValueError(f"a foreign episode of {episode['arm']!r} in {arm}'s route-(i) record")
        seed = episode.get("seed")
        key = (scenario, arm, None if seed is None else int(seed), int(episode["draw_id"]))
        if key in out:
            raise ValueError(f"{key} appears twice in {arm}'s route-(i) record")
        out[key] = float(episode["att_horizon"])
    return out


def _is_new_cell(scenario: str, method: str, tier: str) -> bool:
    from offline.a27_arms import is_a27_arm

    if scenario == "hz1x1":
        from offline.method_tier_grid import A27_TIERS
    else:
        from offline.tier_sweep import A27_TIERS
    return is_a27_arm(scenario, method) or tier in A27_TIERS


def a27_factory_builder(roots: Roots) -> Any:
    """The ``factory_builder`` route (ii) runs with: a NEW cell gets the very factory its route-(i) evaluation used
    (``method_tier_grid.cell_factory`` / ``tier_sweep._arm_factory`` on the campaign's checkpoint, at the recipe's
    budget), A27(d)'s fresh reference its generator, and every REGISTERED cell goes back to P8.4b's own
    ``build_cell_factory`` unchanged."""

    def builder(scenario: str, arm: str, seed: int | None, probe_roots: Any, *, device: Any,
                config_path: Any) -> tuple[Any, dict[str, Any]]:
        from offline import att_rederivation as ar
        from offline.a27_arms import recipe

        method, tier = ar.normalise_arm(scenario, arm)
        if method == "behaviour" and tier == "random_fresh":
            from offline import a27_arms
            from offline.dt_gate import HELD_OUT_DRAWS

            factory = a27_arms.fresh_random_factory(int(seed), scenario_key=SCENARIO_KEYS[scenario],
                                                    draws_root=roots.draws_root, draw_ids=HELD_OUT_DRAWS)
            return factory, {"kind": "algorithmic", "detail": "a27_arms.fresh_random_factory: default_rng([seed, draw])",
                             "route": "route (i)'s factory"}
        if not _is_new_cell(scenario, method, tier):
            return ar.build_cell_factory(scenario, arm, seed, probe_roots, device=device, config_path=config_path)
        cell = next((c for c in training_cells() if (c.scenario, c.tier, c.arm) == (scenario, tier, method)), None)
        if cell is None or seed is None or run_status(roots, cell, int(seed)) != "done":
            raise ValueError(f"{scenario}:{arm} seed {seed} is not trained (no done run on disk); nothing to replay")
        checkpoint = checkpoint_path(roots, scenario, tier, method, int(seed))
        steps = int(recipe(scenario, method).steps)
        if scenario == "hz1x1":
            import offline.method_tier_grid as grid

            factory = grid.cell_factory(method, str(checkpoint), steps, float(grid.tier_spec(tier).target_rtg), device)
        else:
            import argparse

            import offline.tier_sweep as ts

            namespace = argparse.Namespace(checkpoint_dir=str(checkpoint.parent), gradient_steps=steps, device=device,
                                           replicate=False, scenario_key=SCENARIO_KEYS[scenario],
                                           draws_root=str(roots.draws_root), corpus_root=str(roots.corpus_root))
            factory = ts._arm_factory(tier, method, namespace, int(seed))
        return factory, {"kind": "checkpoint", "detail": f"{scenario}:{arm} at {steps} steps",
                         "checkpoint": str(checkpoint), "route": "route (i)'s factory"}

    return builder


def run_route_ii(
    roots: Roots, scenario: str, cells: Any, committed: Mapping[tuple[str, str, int | None, int], float]
) -> dict[str, Any]:
    """P8.4b's runner on *cells* with route (i)'s values as ``committed``: ``reproduces_committed`` then means the two
    routes agree on that episode under ``==``.  ANY refusal stops the campaign -- a disagreement is a finding, not a
    cell to skip (``BRIEF_46`` 3.4)."""
    from offline import att_rederivation as ar
    from offline.admission_probe import ProbeRoots
    from offline.offline_baselines import pin_torch_threads

    work = Path(roots.runs_root) / "eval_ii" / scenario
    assert_target(work, roots)
    work.mkdir(parents=True, exist_ok=True)
    probe = ProbeRoots(repo_root=Path(roots.repo_root), corpus_root=Path(roots.corpus_root),
                       draws_root=Path(roots.draws_root), output_root=Path(roots.output_root), work_dir=work)
    previous = pin_torch_threads(1)
    try:
        outcome = ar.run_campaign(list(cells), roots=probe, engine_seed=ENGINE_SEED, device=None,
                                  committed=committed, protected=(), factory_builder=a27_factory_builder(roots))
    finally:
        pin_torch_threads(previous)
    if outcome["n_refused"]:
        first = outcome["refused"][0]
        raise RuntimeError(
            f"route (ii) does not reproduce route (i) on {outcome['n_refused']} episode(s); first {first['arm']} seed "
            f"{first['seed']} draw {first['draw_id']}: {first['reason']}. The campaign stops (BRIEF_46 3.4)"
        )
    return outcome


def record_episode_actions(
    arm: str,
    seed: int,
    draws: Any,
    inner_factory: Any,
    *,
    expected: Mapping[tuple[int, int], float],
    evaluate_kwargs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """P3a's sequences (A27(d)): each held-out episode re-rolled ALONE through ``dt_gate.evaluate_arm`` with the
    route-(i) factory wrapped by ``nortg_decomposition.recording_factory``, which records and alters nothing; the
    episode's ``att_horizon`` must equal route (i)'s (*expected*) under ``==``, so the recorded trajectory IS the
    reported one.  Each row carries the joint digest and every intersection's column digest
    (``nortg_decomposition.action_sequence_sha256``)."""
    import numpy as np

    from offline import dt_gate
    from offline.nortg_decomposition import action_sequence_sha256, recording_factory

    rows: list[dict[str, Any]] = []
    for draw in draws:
        factory, recorder = recording_factory(inner_factory)
        (result,) = dt_gate.evaluate_arm(arm=arm, seed=int(seed), draw_ids=[int(draw)], choose_action_factory=factory,
                                         **evaluate_kwargs)
        want = expected.get((int(seed), int(draw)))
        if want is None or float(result.att_horizon) != float(want):
            raise ValueError(
                f"{arm} seed {seed} draw {draw}: the recording re-roll gives att {result.att_horizon!r} against route "
                f"(i)'s {want!r}; the recorded actions would not be the reported episode's"
            )
        stacked = np.asarray(recorder.actions, dtype=np.int64)
        rows.append({
            "seed": int(seed),
            "draw_id": int(draw),
            "att_ours": float(result.att_horizon),
            "n_decisions": int(stacked.shape[0]),
            "action_sequence_sha256": action_sequence_sha256(recorder.actions),
            "per_intersection_sha256": [action_sequence_sha256(stacked[:, [i]]) for i in range(stacked.shape[1])],
        })
    return rows


# ----------------------------------------------------------------------
# Data stages (BRIEF_46 3.4 `old-cells`, `collect-random-fresh`; plan D3)
# ----------------------------------------------------------------------

#: The registered hz1x1 cells A27's new cells are compared with, as their committed artifacts name them (plan D3).
#: (c) IQL on the eight tiers; (d) the old random tier's arms and its five-schedule reference; (e) the parents' arms
#: (mappo1000 from P4.4, mappo500 from P4.6); (g) the 40,000-step cells on mappo1000 and mix50.  The grid's compared
#: cells are P5.2c's pinned rows (``tier_statements.load_inputs``) and are not re-rolled.
HZ1X1_OLD_CELLS: tuple[tuple[str, str], ...] = (
    *(("docs/data/p4_4_baselines.json", arm) for arm in ("madt", "bc", "bc_top10", "iql")),
    *(("docs/data/p4_6_grid.json", f"iql@{tier}") for tier in ("maxpressure", "fixedtime", "random", "mappo500")),
    *(("docs/data/p4_6_grid.json", f"{method}@random") for method in ("dt", "bc", "bc_top10", "behaviour")),
    *(("docs/data/p4_6_grid.json", f"{method}@mappo500") for method in ("dt", "bc", "bc_top10")),
    *(("docs/data/p4_7_grid.json", f"iql@{tier}") for tier in ("mix33", "mix50", "mix67")),
    *(("docs/data/p4_7_grid.json", f"{method}@mix50") for method in ("dt", "bc", "bc_top10")),
)


def old_cell_committed(
    repo_root: str | Path, artifact: str, arm: str
) -> dict[tuple[str, str, int | None, int], float]:
    """An old cell's committed ``att_ours`` per episode, read from its committed artifact and keyed as P8.4b keys a
    cell -- the values its re-roll must reproduce under ``==``."""
    payload = json.loads((Path(repo_root) / artifact).read_text(encoding="utf-8"))
    out: dict[tuple[str, str, int | None, int], float] = {}
    for episode in payload["episodes"]:
        if str(episode["arm"]) != arm:
            continue
        seed = episode.get("seed")
        key = ("hz1x1", arm, None if seed is None else int(seed), int(episode["draw_id"]))
        if key in out:
            raise ValueError(f"{artifact}: {key} appears twice")
        out[key] = float(episode["att_horizon"])
    if not out:
        raise ValueError(f"{artifact} carries no episodes of {arm!r}")
    return out


def roll_old_cells(roots: Roots, cells: Any = None) -> dict[str, Any]:
    """Re-roll the registered hz1x1 cells through P8.4b's route with P8.4b's OWN builder, each episode's committed
    ``att_ours`` reproduced under ``==`` before its ``att_engine`` and companions are used (A27(a)).  Any refusal stops:
    a cell that does not reproduce is not the cell its artifact reports."""
    from offline import att_rederivation as ar
    from offline.admission_probe import ProbeRoots
    from offline.offline_baselines import pin_torch_threads

    chosen = list(HZ1X1_OLD_CELLS if cells is None else cells)
    committed: dict[tuple[str, str, int | None, int], float] = {}
    keys: list[Any] = []
    for artifact, arm in chosen:
        values = old_cell_committed(roots.repo_root, artifact, arm)
        committed.update(values)
        keys += [ar.CellKey(scenario, a, seed, draw) for (scenario, a, seed, draw) in sorted(
            values, key=lambda k: (-1 if k[2] is None else k[2], k[3]))]
    work = Path(roots.runs_root) / "old_cells" / "hz1x1"
    assert_target(work, roots)
    work.mkdir(parents=True, exist_ok=True)
    probe = ProbeRoots(repo_root=Path(roots.repo_root), corpus_root=Path(roots.corpus_root),
                       draws_root=Path(roots.draws_root), output_root=Path(roots.output_root), work_dir=work)
    previous = pin_torch_threads(1)
    try:
        outcome = ar.run_campaign(keys, roots=probe, engine_seed=ENGINE_SEED, device=None, committed=committed,
                                  protected=(), factory_builder=None)
    finally:
        pin_torch_threads(previous)
    if outcome["n_refused"]:
        first = outcome["refused"][0]
        raise RuntimeError(
            f"{outcome['n_refused']} old-cell episode(s) refused; first {first['arm']} seed {first['seed']} draw "
            f"{first['draw_id']} does not reproduce its committed att_ours: {first['reason']}"
        )
    return outcome


#: A27(d)'s collection, as the registration states it; the registered random tier's manifest must carry the same
#: reward settings and draws, and the new tier changes ONLY the generator and the episodes per draw.
RANDOM_FRESH_DIRS: Mapping[str, tuple[str, str]] = {
    "hz1x1": ("cf_hz1x1__random", "cf_hz1x1__random_fresh"),
    "grid4x4": ("cf_grid4x4__random", "cf_grid4x4__random_fresh"),
}
A27D_COLLECTION: Mapping[str, Any] = {
    "behavior_policy": "random", "base_seed": 1000, "local_reward_fn": "queue_length", "global_reward_weight": 0.0,
    "flow_draw_ids": list(range(1, 201)),
}


def random_fresh_collect_argv(roots: Roots, scenario: str) -> list[str]:
    """``offline.collect``'s argv for A27(d)'s tier: every setting read from the REGISTERED random tier's manifest --
    after checking it carries A27(d)'s stated reward settings and draws -- then ``--random-rng per-episode``,
    ``--episodes 1``, draws 1-200, base seed 1000, into the new directory."""
    registered_name, fresh_name = RANDOM_FRESH_DIRS[scenario]
    manifest = json.loads((Path(roots.corpus_root) / registered_name / "manifest.json").read_text(encoding="utf-8"))
    meta = manifest["run_metadata"]
    for key, value in A27D_COLLECTION.items():
        if meta.get(key) != value:
            raise ValueError(
                f"{registered_name}: run_metadata {key} is {meta.get(key)!r}, not A27(d)'s {value!r}; the new tier is "
                "collected with the registered tier's settings only when those are the ones A27(d) states"
            )
    params = meta["flow_randomizer_params"]
    return [
        "--backend", str(meta["backend"]), "--env-config", str(meta["env_paths"]["config"]),
        "--policy", "random", "--random-rng", "per-episode", "--episodes", "1", "--base-seed", "1000",
        "--flow-draws-range", "1", "201",
        "--max-steps", str(meta["max_steps"]), "--delta-time", str(meta["delta_time"]),
        "--control-mode", str(meta["control_mode"]), "--global-reward-fn", str(meta["global_reward_fn"]),
        "--local-reward-fn", "queue_length", "--global-reward-weight", "0.0",
        "--state-features", *[str(v) for v in meta["state_features"]],
        "--flow-jitter-sigma", str(params["jitter_sigma_s"]), "--flow-thin-p", str(params["thin_p"]),
        "--flow-volume-scale", str(params["volume_scale"]),
        "--out-dir", str(Path(roots.corpus_root) / fresh_name),
    ]


def assert_random_fresh_manifest(manifest: Mapping[str, Any]) -> None:
    """Refuse a ``random_fresh`` corpus that is not A27(d)'s collection: the per-episode generator, one episode per
    draw, draws 1-200, base seed 1000, the registered reward settings, 200 episodes."""
    meta = manifest["run_metadata"]
    expected = {**A27D_COLLECTION, "random_rng_scheme": "per-episode", "episodes": 1}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"random_fresh: run_metadata {key} is {meta.get(key)!r}, not A27(d)'s {value!r}")
    if len(manifest["episodes"]) != 200:
        raise ValueError(f"random_fresh holds {len(manifest['episodes'])} episodes, not A27(d)'s 200")


def collect_random_fresh(roots: Roots, scenario: str) -> Path:
    """Collect A27(d)'s tier into a NEW corpus directory (refused if it exists), then run the two registered checks on
    it before anything else may read it: the manifest is A27(d)'s collection, and every intersection holds 200 distinct
    schedules."""
    import offline.collect as collect
    from offline.a27_arms import assert_distinct_schedules

    argv = random_fresh_collect_argv(roots, scenario)
    out_dir = Path(argv[argv.index("--out-dir") + 1])
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; A27(d)'s tier is collected once, into a new directory")
    if out_dir.parent.resolve() != Path(roots.corpus_root).resolve() or out_dir.name != RANDOM_FRESH_DIRS[scenario][1]:
        raise PermissionError(f"{out_dir} is not one of A27(d)'s two corpus directories")
    if collect.main(argv) != 0:
        raise RuntimeError(f"offline.collect failed for {out_dir}")
    assert_random_fresh_manifest(json.loads((out_dir / "manifest.json").read_text(encoding="utf-8")))
    assert_distinct_schedules(out_dir, expected=200)
    return out_dir


# ----------------------------------------------------------------------
# Declarations (BRIEF_46 3.4 `declare`), written before any training
# ----------------------------------------------------------------------

DECLARATION_FORMAT_VERSION = "p9.1-declaration/1.0"


def _write_declaration(path: Path, payload: Mapping[str, Any], roots: Roots) -> Path:
    """A declaration is written once; declaring again is allowed only if it reproduces the file byte for byte."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target = assert_target(path, roots)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"{target} exists with other content; a declaration is not rewritten")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8") as handle:
        handle.write(text)
    return target


def _mixture_selection(roots: Roots) -> tuple[Any, Any]:
    """The overlap mixture's realised training set and kept decile, by the registered functions (as ``_run_train``)."""
    import offline.method_tier_grid as grid

    spec = grid.tier_spec("mix_m1000_m500")
    dataset = grid.tier_dataset(spec, roots.corpus_root)
    components = grid._component_streams(spec, roots.corpus_root, context_length=grid.CONTEXT_LENGTH)
    selected = grid.training_streams(spec, dataset, component_streams=components)
    return selected, grid.top_decile_streams(selected)


def declare_mixture(roots: Roots) -> dict[str, Any]:
    """A27(e)'s tier declared in P4.6's own format (with its two components, whose training sets it is drawn from),
    plus the design facts reproduced from the corpus -- refused, and nothing written, if they do not reproduce."""
    import offline.a27_arms as a27_arms
    import offline.method_tier_grid as grid

    declaration = grid.declaration_artifact(roots.corpus_root, ["mappo1000", "mappo500", "mix_m1000_m500"])
    selected, kept = _mixture_selection(roots)
    measured = a27_arms.assert_mixture_design_facts(selected, kept)
    payload = {
        "format_version": DECLARATION_FORMAT_VERSION,
        "tier": "mix_m1000_m500",
        "registered_in": "A27(e)",
        "label": LABELS["A27(e)"],
        "design_facts": json.loads(json.dumps(a27_arms.MIXTURE_DESIGN_FACTS)),
        "measured": measured,
        "design_facts_reproduced": True,
        "p4a_reading": "the plan's risk-register reading: the weaker source contributes 1 to 19 of the 20 kept streams",
        "p4_6_format_declaration": declaration,
    }
    _write_declaration(Path(roots.run_root) / "declaration_mix_m1000_m500.json", payload, roots)
    return payload


def _tier_means(directory: Path) -> dict[str, float]:
    """The grid's two ladder fields from the raw ``.npz``, in manifest order: the mean over episodes of the final
    ``att_per_step`` and the mean over every (episode, intersection) stream of ``sum(local_reward)`` -- the definitions
    that reproduce the registered tiers' typed values (checked against all four on 2026-10-15, docs/plans/p9.1.md)."""
    import numpy as np

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    finals: list[float] = []
    streams: list[float] = []
    for entry in manifest["episodes"]:
        with np.load(directory / str(entry["filename"])) as arrays:
            finals.append(float(arrays["att_per_step"][-1]))
            streams += [float(np.sum(arrays[f"ix{i}_local_reward"], dtype=np.float64))
                        for i in range(len(arrays["ix_ids"]))]
    return {"ladder_att": sum(finals) / len(finals), "mean_training_return": sum(streams) / len(streams)}


def declare_random_fresh(roots: Roots, scenario: str) -> dict[str, Any]:
    """A27(d)'s tier declared from its own corpus, after the two registered collection checks.

    hz1x1: the declared rule's ``target_rtg`` / ``rtg_scale`` over its 200 streams -- the values plan Q2 types into
    ``method_tier_grid.A27_TIERS`` in a reviewed commit -- with IQL's reward scale and the statistics digest.  grid:
    ``tier_sweep``'s own declaration with ``ladder_att`` and ``mean_training_return`` COMPUTED here, never typed.  Both
    quote the OLD tier's basis from its committed declaration: the comparison changes the generator AND the basis.
    """
    import offline.a27_arms as a27_arms
    import offline.method_tier_grid as grid
    import offline.tier_sweep as ts
    from offline.offline_baselines import iql_reward_scale, stream_returns

    if scenario == "hz1x1":
        directory = grid.tier_dirs(grid.tier_spec("random_fresh"), roots.corpus_root)[0]
    elif scenario == "grid4x4":
        directory = ts.tier_dirs(ts.tier_spec("random_fresh"), roots.corpus_root)[0]
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    assert_random_fresh_manifest(json.loads((directory / "manifest.json").read_text(encoding="utf-8")))
    payload: dict[str, Any] = {
        "format_version": DECLARATION_FORMAT_VERSION,
        "tier": "random_fresh",
        "scenario": scenario,
        "registered_in": "A27(d)",
        "label": LABELS["A27(d)"],
    }
    if scenario == "hz1x1":
        spec = grid.tier_spec("random_fresh")
        payload["distinct_schedules"] = a27_arms.assert_distinct_schedules(directory, expected=int(spec.stream_count))
        dataset = grid.tier_dataset(spec, roots.corpus_root)
        selected = grid.training_streams(spec, dataset)
        target, scale = grid.recomputed_target_and_scale(selected)
        old = json.loads((Path(roots.repo_root) / "docs/data/p4_6_declaration.json").read_text(encoding="utf-8"))
        old_random = old["tiers"]["random"]
        payload.update({
            "values_to_type": {"target_rtg": target, "rtg_scale": scale},
            "rule": "target_rtg = max episode return in the TRAINING SET; rtg_scale = max|rtg| (method_tier_grid)",
            "training_streams": len(selected),
            "split_streams": len(stream_returns(dataset)),
            "iql_reward_scale": iql_reward_scale([s.total_return for s in selected]),
            "statistics_digest": grid.statistics_digest(dataset),
            "statistics_fitted_on": "every one of the tier's 200 episodes: its split IS its training set",
            "old_tier_basis": {
                "source": "docs/data/p4_6_declaration.json",
                **{key: old_random[key] for key in ("split_streams", "training_streams", "target_rtg", "rtg_scale",
                                                    "iql_reward_scale", "subsample", "statistics_fitted_on")},
            },
        })
    elif scenario == "grid4x4":
        spec = ts.tier_spec("random_fresh")
        payload["distinct_schedules"] = a27_arms.assert_distinct_schedules(directory, expected=int(spec.episode_count))
        parts = ts.tier_parts("random_fresh", roots.corpus_root)
        declaration = ts.declaration_artifact(parts, roots.corpus_root)
        computed = _tier_means(directory)
        declaration.update(computed)
        old = json.loads((Path(roots.repo_root) / "docs/data/p5_2_declaration_random.json").read_text(encoding="utf-8"))
        payload.update({
            "p5_2_format_declaration": declaration,
            "computed": {
                **computed,
                "ladder_att_definition": "mean over the tier's episodes of the final att_per_step (att_ladder_v11's)",
                "mean_training_return_definition": "mean over every (episode, intersection) stream of sum(local_reward)",
            },
            "old_tier_basis": {
                "source": "docs/data/p5_2_declaration_random.json",
                **{key: old[key] for key in ("episodes_available", "episodes_selected", "subsample", "subsample_rng_seed",
                                             "training_streams", "ladder_att", "mean_training_return",
                                             "statistics_split")},
            },
        })
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    payload["what_changes_besides_the_generator"] = (
        "the normalisation basis: the old random tier's state statistics were fitted on its 400-episode training split "
        "(two episodes per draw) and its training set was a 200-episode subsample; random_fresh's split and training "
        "set are the same 200 episodes, one per draw (A27(d))"
    )
    _write_declaration(Path(roots.run_root) / f"declaration_random_fresh_{scenario}.json", payload, roots)
    return payload


# ----------------------------------------------------------------------
# Stage runners (BRIEF_46 3.4): evaluation by cell, P3a, status from disk, the pre-token check
# ----------------------------------------------------------------------

#: The registered five-schedule references P3a's identity column compares with (A27(d)): hz1x1's P4.6 behaviour cell
#: of the random tier and the grid's shared collapse reference (P5.1's ``random`` cell, ``tier_sweep`` D12).
REGISTERED_RANDOM_REFERENCES: tuple[Cell, ...] = (
    Cell("hz1x1", "random", "behaviour", "A27(d)"),
    Cell("grid4x4", "mappo1000", "random", "A27(d)"),
)


def evaluation_cells() -> tuple[Cell, ...]:
    """Every NEW cell evaluated by both routes: the 51 trained cells and A27(d)'s two fresh references."""
    return (*training_cells(), *reference_cells())


def p3a_cells() -> tuple[Cell, ...]:
    """The cells whose held-out action sequences P3a and its beside-table read (A27(d)): the DT on ``random_fresh`` and
    both random references, on both scenarios."""
    return (
        Cell("hz1x1", "random_fresh", "dt", "A27(d)"),
        Cell("grid4x4", "random_fresh", "dt_nomix", "A27(d)"),
        *reference_cells(),
        *REGISTERED_RANDOM_REFERENCES,
    )


def cell_sort_key(cell: Cell) -> tuple[str, str, str]:
    return (cell.scenario, cell.tier, cell.arm)


def shard_of(cells: Any, *, shard: int, of: int) -> list[Cell]:
    """A deterministic slice of *cells* for N-way parallel evaluation: sorted, then every N-th from *shard*."""
    if int(of) < 1 or not 0 <= int(shard) < int(of):
        raise ValueError(f"shard {shard!r} of {of!r} is not a valid partition")
    ordered = sorted(cells, key=cell_sort_key)
    return [cell for index, cell in enumerate(ordered) if index % int(of) == int(shard)]


def _expected_episode_keys() -> set[tuple[int, int]]:
    from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS

    return {(int(seed), int(draw)) for seed in TRAINING_SEEDS for draw in HELD_OUT_DRAWS}


def _assert_route_i_complete(payload: Mapping[str, Any], cell: Cell) -> None:
    keys = {(int(e["seed"]), int(e["draw_id"])) for e in payload["episodes"]}
    arms = {str(e["arm"]) for e in payload["episodes"]}
    if arms != {cell.key} or keys != _expected_episode_keys() or len(payload["episodes"]) != len(keys):
        raise ValueError(f"{cell.scenario}:{cell.key}'s route-(i) record is not the complete 5 x 100 cell")


def route_i(roots: Roots, cell: Cell) -> dict[str, Any]:
    """Route (i) for one cell: the module CLI's own ``evaluate`` (``behaviour``) in process, skipped when its complete
    record is on disk.  The hz1x1 aggregates are rebuilt from disk first."""
    import importlib

    output = route_i_output(roots, cell)
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        _assert_route_i_complete(payload, cell)
        return payload
    if cell.scenario == "hz1x1" and cell.arm != "behaviour":
        write_training_aggregates(roots)
    assert_target(output, roots)
    output.parent.mkdir(parents=True, exist_ok=True)
    name, argv = route_i_command(roots, cell)
    if int(importlib.import_module(f"offline.{name}").main(argv)) != 0:
        raise RuntimeError(f"route (i) failed for {cell.scenario}:{cell.key}")
    payload = json.loads(output.read_text(encoding="utf-8"))
    _assert_route_i_complete(payload, cell)
    return payload


def _evaluate_kwargs(roots: Roots, scenario: str, tier: str) -> dict[str, Any]:
    """The rollout settings route (i) uses for *tier*, built by the same calls the CLIs make."""
    from offline.materialise_draws import draw_config_path

    key = SCENARIO_KEYS[scenario]
    if scenario == "hz1x1":
        import offline.method_tier_grid as grid

        settings = grid.env_settings_for_tiers([grid.tier_spec(tier)], roots.corpus_root)
    else:
        import offline.tier_sweep as ts
        from offline.dt_gate import env_settings_from_manifest

        settings = env_settings_from_manifest(ts.tier_dirs(ts.tier_spec(tier), roots.corpus_root)[0] / "manifest.json")
    return {"config_for_draw": lambda draw: draw_config_path(key, int(draw), out_root=roots.draws_root),
            "env_settings": settings, "scenario_id": key, "engine_seed": ENGINE_SEED}


def _p3a_path(roots: Roots, cell: Cell) -> Path:
    return Path(roots.runs_root) / "p3a" / cell.scenario / f"{cell.arm}@{cell.tier}.json"


def _p3a_factory(roots: Roots, cell: Cell, seed: int) -> Any:
    if cell in REGISTERED_RANDOM_REFERENCES:
        from offline.method_tier_grid import _random_factory

        return _random_factory(int(seed))
    factory, _ = a27_factory_builder(roots)(cell.scenario, cell.key, int(seed), None, device=None, config_path=None)
    return factory


def record_p3a(roots: Roots, cell: Cell, values: Mapping[tuple[str, str, int | None, int], float]) -> list[dict[str, Any]]:
    """Every held-out episode of a P3a cell re-rolled alone with its actions recorded, each reproducing *values* (route
    (i)'s, or the committed artifact's for a registered reference) under ``==``; written once per cell."""
    from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS

    path = _p3a_path(roots, cell)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["episodes"]
    expected = {(int(seed), int(draw)): att for (_, _, seed, draw), att in values.items()}
    kwargs = _evaluate_kwargs(roots, cell.scenario, cell.tier)
    rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        rows += record_episode_actions(cell.key, int(seed), HELD_OUT_DRAWS, _p3a_factory(roots, cell, int(seed)),
                                       expected=expected, evaluate_kwargs=kwargs)
    payload = {"format_version": ACTIONS_FORMAT_VERSION, "scenario": cell.scenario, "arm": cell.key,
               "registered_in": "A27(d)", "digest": "nortg_decomposition.action_sequence_sha256 (int64, decision order)",
               "episodes": rows}
    _write_declaration(path, payload, roots)
    return rows


def _registered_reference_values(roots: Roots, cell: Cell) -> dict[tuple[str, str, int | None, int], float]:
    """A registered random reference's committed ``att_ours`` per episode: P4.6's ``behaviour@random`` from
    ``docs/data/p4_6_grid.json``; the grid's P5.1 ``random`` cell from ``output/p5_1/eval_random.json`` at the digest
    ``output/SHA256SUMS_p5_1.txt`` lists for it."""
    import hashlib

    if cell.scenario == "hz1x1":
        return old_cell_committed(roots.repo_root, "docs/data/p4_6_grid.json", "behaviour@random")
    path = Path(roots.output_root) / "p5_1" / "eval_random.json"
    listed = {line.split()[1]: line.split()[0] for line in
              (Path(roots.output_root) / "SHA256SUMS_p5_1.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    if listed.get("p5_1/eval_random.json") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"{path} is not at the digest output/SHA256SUMS_p5_1.txt lists for it")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {("grid4x4", cell.key, int(e["seed"]), int(e["draw_id"])): float(e["att_horizon"]) for e in payload["episodes"]}


def record_registered_references(roots: Roots) -> dict[str, int]:
    """P3a's two registered references, re-rolled with their actions recorded, committed ``att_ours`` reproduced."""
    return {f"{cell.scenario}:{cell.key}": len(record_p3a(roots, cell, _registered_reference_values(roots, cell)))
            for cell in REGISTERED_RANDOM_REFERENCES}


def _evaluation_manifests(roots: Roots) -> dict[str, tuple[Path, dict[str, Any]]]:
    """P8.4b's campaign manifests of the evaluation, by name: route (ii) per scenario (every evaluation cell x seed x
    draw) and the hz1x1 old cells (every committed episode of :data:`HZ1X1_OLD_CELLS`)."""
    from offline import att_rederivation as ar
    from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS

    out: dict[str, tuple[Path, dict[str, Any]]] = {}
    for scenario in SCENARIOS_ORDER:
        keys = [ar.CellKey(scenario, cell.key, int(seed), int(draw)) for cell in evaluation_cells()
                if cell.scenario == scenario for seed in TRAINING_SEEDS for draw in HELD_OUT_DRAWS]
        out[f"eval_ii/{scenario}"] = (Path(roots.runs_root) / "eval_ii" / scenario / ar.MANIFEST_NAME,
                                      ar.campaign_manifest(keys, engine_seed=ENGINE_SEED))
    old: list[Any] = []
    for artifact, arm in HZ1X1_OLD_CELLS:
        old += [ar.CellKey(*key) for key in old_cell_committed(roots.repo_root, artifact, arm)]
    out["old_cells/hz1x1"] = (Path(roots.runs_root) / "old_cells" / "hz1x1" / ar.MANIFEST_NAME,
                              ar.campaign_manifest(old, engine_seed=ENGINE_SEED))
    return out


def _assert_manifest_is(path: Path, manifest: Mapping[str, Any]) -> None:
    """A manifest on disk must BE this campaign's: P8.4b's agreement check (count and digest), then every field --
    ``campaign_status`` reads the ``cells`` list, so a list edited under an unchanged digest is refused too."""
    from offline import att_rederivation as ar

    ar.assert_manifest_agrees(path, manifest)
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != dict(manifest):
        raise ValueError(f"{path} is not this campaign's manifest: its fields differ although its count and digest "
                         "agree; refusing a manifest that does not describe the cells it declares")


def declare_evaluation(roots: Roots) -> dict[str, int]:
    """Write the evaluation's three manifests ONCE, from one process, before any shard starts (a shard that declared
    its own could race another on the same file).  Every manifest is checked before the first is written; declaring
    again is a no-op, and a manifest on disk that is not this campaign's is refused (:func:`_assert_manifest_is`)."""
    manifests = _evaluation_manifests(roots)
    for path, manifest in manifests.values():
        _assert_manifest_is(path, manifest)
    for path, manifest in manifests.values():
        if not path.exists():
            _write_declaration(path, manifest, roots)
    return {name: int(manifest["n_cells"]) for name, (_, manifest) in manifests.items()}


def assert_evaluation_declared(roots: Roots, *names: str) -> None:
    """Refuse unless each named manifest is on disk and is this campaign's (:func:`_assert_manifest_is`); writes
    nothing."""
    manifests = _evaluation_manifests(roots)
    for name in names:
        path, manifest = manifests[name]
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing: run `declare-evaluation` first (one process, before the shards)")
        _assert_manifest_is(path, manifest)


def old_cells_shard(shard: int, of: int) -> list[tuple[str, str]]:
    """A deterministic slice of :data:`HZ1X1_OLD_CELLS` for N-way parallel re-rolling: sorted, then every N-th."""
    if int(of) < 1 or not 0 <= int(shard) < int(of):
        raise ValueError(f"shard {shard!r} of {of!r} is not a valid partition")
    return [cell for index, cell in enumerate(sorted(HZ1X1_OLD_CELLS)) if index % int(of) == int(shard)]


def evaluate_cell(roots: Roots, cell: Cell) -> dict[str, Any]:
    """One cell: refused unless every run is done; route (i); route (ii) on route (i)'s values; P3a if it is one."""
    from offline import att_rederivation as ar
    from offline.dt_gate import TRAINING_SEEDS

    if cell in training_cells():
        undone = [seed for seed in TRAINING_SEEDS if run_status(roots, cell, seed) != "done"]
        if undone:
            raise ValueError(f"{cell.scenario}:{cell.key} is not trained for seeds {undone}; nothing to evaluate")
    payload = route_i(roots, cell)
    committed = committed_from_route_i(payload, cell.scenario, cell.key)
    keys = [ar.CellKey(scenario, arm, seed, draw) for (scenario, arm, seed, draw) in sorted(committed)]
    outcome = run_route_ii(roots, cell.scenario, keys, committed)
    p3a = None
    if (cell.scenario, cell.key) in {(c.scenario, c.key) for c in p3a_cells()}:
        p3a = record_p3a(roots, cell, committed)
    return {"cell": f"{cell.scenario}:{cell.key}", "route_ii": outcome, "p3a_episodes": None if p3a is None else len(p3a)}


def evaluate(roots: Roots, *, shard: int = 0, of: int = 1, scenario: str | None = None) -> dict[str, Any]:
    """Every evaluation cell of this shard whose runs are done, in order; a not-yet-trained cell is reported, never
    guessed at.  Resumable: each route skips what is complete on disk."""
    from offline.dt_gate import TRAINING_SEEDS

    assert_evaluation_declared(roots, *(f"eval_ii/{name}" for name in SCENARIOS_ORDER if scenario in (None, name)))
    done, waiting = [], []
    for cell in shard_of([c for c in evaluation_cells() if scenario in (None, c.scenario)], shard=shard, of=of):
        if cell in training_cells() and any(run_status(roots, cell, s) != "done" for s in TRAINING_SEEDS):
            waiting.append(f"{cell.scenario}:{cell.key}")
            continue
        evaluate_cell(roots, cell)
        done.append(f"{cell.scenario}:{cell.key}")
    return {"evaluated": done, "waiting_for_training": waiting}


SCENARIOS_ORDER: tuple[str, ...] = ("hz1x1", "grid4x4")


def status(roots: Roots) -> dict[str, Any]:
    """Every stage's state, recomputed from disk: training, route (i) cell files, route (ii) campaign, P3a, old cells."""
    from offline import att_rederivation as ar

    training = training_status(roots)
    route_i_done = 0
    for cell in evaluation_cells():
        path = route_i_output(roots, cell)
        if path.exists():
            try:
                _assert_route_i_complete(json.loads(path.read_text(encoding="utf-8")), cell)
                route_i_done += 1
            except (ValueError, KeyError):
                pass
    route_ii = {name: ar.campaign_status(Path(roots.runs_root) / "eval_ii" / name) for name in SCENARIOS_ORDER}
    p3a_done = sum(_p3a_path(roots, cell).exists() for cell in p3a_cells())
    old = ar.campaign_status(Path(roots.runs_root) / "old_cells" / "hz1x1")
    evaluation = {"cells_declared": len(evaluation_cells()), "route_i_cells_done": route_i_done,
                  "route_ii": route_ii, "p3a_cells_done": p3a_done, "p3a_cells_declared": len(p3a_cells())}
    complete = (training["complete"] and route_i_done == len(evaluation_cells())
                and all(state["complete"] for state in route_ii.values()) and p3a_done == len(p3a_cells())
                and old["complete"])
    return {"training": training, "evaluation": evaluation, "old_cells_hz1x1": old, "complete": complete}


def check(roots: Roots, *, stage: str) -> dict[str, Any]:
    """Every refusal before the token; writes nothing.  ``collect``: the roots and the registered random tiers, the
    new tiers absent.  ``campaign``: the new tiers collected and declared, hz1x1 ``random_fresh``'s prompt typed (plan
    Q2), the held-out draws materialised."""
    import math

    from offline.dt_gate import HELD_OUT_DRAWS
    from offline.materialise_draws import draw_config_path

    for name, path in (("output", roots.output_root), ("corpus", roots.corpus_root), ("draws", roots.draws_root)):
        if not Path(path).is_dir():
            raise FileNotFoundError(f"the {name} root {path} is not a directory")
    if stage == "collect":
        for scenario, (registered, fresh) in RANDOM_FRESH_DIRS.items():
            random_fresh_collect_argv(roots, scenario)
            if (Path(roots.corpus_root) / fresh).exists():
                raise FileExistsError(f"{fresh} exists already; A27(d)'s tier is collected once")
        return {"stage": stage, "ok": True}
    if stage != "campaign":
        raise ValueError(f"unknown stage {stage!r}")
    import offline.method_tier_grid as grid

    for scenario, (_, fresh) in RANDOM_FRESH_DIRS.items():
        manifest = Path(roots.corpus_root) / fresh / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"{manifest} is missing: collect A27(d)'s tier first")
        assert_random_fresh_manifest(json.loads(manifest.read_text(encoding="utf-8")))
    for name in ("declaration_mix_m1000_m500.json", "declaration_random_fresh_hz1x1.json",
                 "declaration_random_fresh_grid4x4.json"):
        if not (Path(roots.run_root) / name).is_file():
            raise FileNotFoundError(f"{Path(roots.run_root) / name} is missing: run `declare` first")
    spec = grid.A27_TIERS["random_fresh"]
    if math.isnan(spec.target_rtg) or math.isnan(spec.rtg_scale):
        raise ValueError("hz1x1 random_fresh's prompt is not typed yet (plan Q2): its DT cannot train")
    for key in SCENARIO_KEYS.values():
        missing = [d for d in HELD_OUT_DRAWS if not Path(draw_config_path(key, d, out_root=roots.draws_root)).is_file()]
        if missing:
            raise FileNotFoundError(f"{key}: held-out draws {missing[:3]} are not materialised")
    assert_target(Path(roots.runs_root) / "probe", roots)
    return {"stage": stage, "ok": True}


# ----------------------------------------------------------------------
# The pre-flight (BRIEF_46 5, G2): measured, never estimated; written only under a scratch directory
# ----------------------------------------------------------------------

PREFLIGHT_FORMAT_VERSION = "p9.1-preflight/1.0"

#: The registered cell the pre-flight reproduces end to end through both routes (P4.6, hz1x1).
PREFLIGHT_REGISTERED_CELL: tuple[str, str, str] = ("docs/data/p4_6_training.json", "maxpressure", "iql")


def cell_count_by_arm() -> dict[str, int]:
    """Runs per ``<scenario>:<arm>`` over the whole manifest (5 seeds per cell)."""
    from offline.dt_gate import TRAINING_SEEDS

    counts: dict[str, int] = {}
    for cell in training_cells():
        key = f"{cell.scenario}:{cell.arm}"
        counts[key] = counts.get(key, 0) + len(TRAINING_SEEDS)
    return dict(sorted(counts.items()))


def repoint_training_record(record: Mapping[str, Any], output_root: str | Path, *, tier: str,
                            method: str) -> dict[str, Any]:
    """A committed training record's runs of one cell, their checkpoints re-pointed at *output_root*'s
    ``p4_6/checkpoints`` copies -- each accepted only if its canonical digest is the one the record committed."""
    from offline.method_tier_grid import canonical_digest_of

    runs = []
    for run in record["runs"]:
        if (run["tier"], run["method"]) != (tier, method):
            continue
        path = Path(output_root) / "p4_6" / "checkpoints" / Path(str(run["checkpoint"])).name
        if canonical_digest_of(path) != run["canonical_digest"]:
            raise ValueError(f"{path}'s canonical digest is not the one the committed record carries; refusing it")
        runs.append({**run, "checkpoint": str(path)})
    return {**dict(record), "runs": runs}


def assert_equal_episodes(left: Mapping[Any, float], right: Mapping[Any, float], *, what: str) -> int:
    """Two per-episode maps over the same episodes, equal under ``==`` on every one; returns the count."""
    if set(left) != set(right):
        raise ValueError(f"{what}: the two do not cover the same episodes ({len(left)} and {len(right)})")
    differing = [key for key in left if float(left[key]) != float(right[key])]
    if differing:
        raise ValueError(f"{what}: {len(differing)} of {len(left)} episodes differ, first {differing[0]}")
    return len(left)


def _preflight_matched_runs(roots: Roots, scratch: Path, device: Any) -> dict[str, Any]:
    """ONE matched BC run and ONE matched IQL run on the grid's mappo1000 tier, seed 101, the full 40,000 steps, through
    the campaign's own ``train_run`` into the scratch tree; their recorded seconds are the measurement."""
    import torch

    scratch_roots = Roots(repo_root=roots.repo_root, output_root=scratch / "output", corpus_root=roots.corpus_root,
                          draws_root=roots.draws_root)
    inputs = _grid_inputs(scratch_roots, "mappo1000")
    resolved = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    out: dict[str, Any] = {}
    for arm in ("bc_m", "iql_m"):
        cell = next(c for c in training_cells() if (c.scenario, c.tier, c.arm) == ("grid4x4", "mappo1000", arm))
        record = train_run(scratch_roots, cell, 101, inputs, device=resolved, log_every=5000)
        out[arm] = {"seconds": record["seconds"], "gradient_steps": record["gradient_steps"], "clock": CLOCK_LABEL,
                    "device": record["device"], "canonical_digest": record["canonical_digest"]}
    return out


def _preflight_registered_cell(roots: Roots, scratch: Path) -> dict[str, Any]:
    """One REGISTERED cell (P4.6's ``iql@maxpressure``, 500 episodes) through both routes, each equal to the committed
    ``att_ours`` under ``==``: route (i) by ``method_tier_grid evaluate`` on the committed training record (checkpoints
    re-pointed at the main tree's copies, digest-checked); route (ii) by ``run_campaign`` with the campaign's builder."""
    import offline.method_tier_grid as grid
    from offline import att_rederivation as ar
    from offline.admission_probe import ProbeRoots
    from offline.offline_baselines import pin_torch_threads

    artifact, tier, method = PREFLIGHT_REGISTERED_CELL
    arm = f"{method}@{tier}"
    committed = old_cell_committed(roots.repo_root, "docs/data/p4_6_grid.json", arm)
    record = json.loads((Path(roots.repo_root) / artifact).read_text(encoding="utf-8"))
    route_i_dir = scratch / "registered_route_i"
    route_i_dir.mkdir(parents=True, exist_ok=True)
    (route_i_dir / "p4_6_training.json").write_text(
        json.dumps(repoint_training_record(record, roots.output_root, tier=tier, method=method)), encoding="utf-8")
    argv = ["--corpus-root", str(roots.corpus_root), "--draws-root", str(roots.draws_root),
            "--scenario-key", SCENARIO_KEYS["hz1x1"], "--scenario-id", SCENARIO_KEYS["hz1x1"],
            "--out-dir", str(route_i_dir), "--work-dir", str(route_i_dir), "--artifact-prefix", "p4_6",
            "--steps", str(int(record["declared_gradient_steps"])), "--engine-seed", str(ENGINE_SEED),
            "--torch-threads", "1", "evaluate", "--tier", tier, "--method", method]
    if grid.main(argv) != 0:
        raise RuntimeError("route (i) failed on the registered cell")
    payload = json.loads((route_i_dir / f"eval_{tier}_{method}.json").read_text(encoding="utf-8"))
    route_i_values = committed_from_route_i(payload, "hz1x1", arm)
    matched_i = assert_equal_episodes(route_i_values, committed, what="route (i) against the committed cell")
    work = scratch / "registered_route_ii"
    work.mkdir(parents=True, exist_ok=True)
    probe = ProbeRoots(repo_root=Path(roots.repo_root), corpus_root=Path(roots.corpus_root),
                       draws_root=Path(roots.draws_root), output_root=Path(roots.output_root), work_dir=work)
    keys = [ar.CellKey(*key) for key in sorted(committed, key=lambda k: (k[2], k[3]))]
    previous = pin_torch_threads(1)
    try:
        outcome = ar.run_campaign(keys, roots=probe, engine_seed=ENGINE_SEED, device=None, committed=committed,
                                  protected=(), factory_builder=a27_factory_builder(roots))
    finally:
        pin_torch_threads(previous)
    if outcome["n_refused"]:
        raise RuntimeError(f"route (ii) refused {outcome['n_refused']} episode(s) of the registered cell")
    return {"cell": arm, "artifact": "docs/data/p4_6_grid.json", "route_i_equal_to_committed": matched_i,
            "route_ii_reproduces_committed": int(outcome["rolled"]), "route_ii_seconds": outcome["seconds"]}


def _preflight_a27_grid_path(roots: Roots, scratch: Path, draws: Any) -> dict[str, Any]:
    """The NEW grid path end to end on the two matched runs (seed 101): route (i) by ``tier_sweep evaluate --seeds 101``
    and route (ii) by ``run_campaign`` with the campaign's builder, equal under ``==`` on the first held-out draws."""
    import offline.tier_sweep as ts
    from offline import att_rederivation as ar
    from offline.admission_probe import ProbeRoots
    from offline.iql_correction import narrowed_held_out_draws
    from offline.offline_baselines import pin_torch_threads

    scratch_roots = Roots(repo_root=roots.repo_root, output_root=scratch / "output", corpus_root=roots.corpus_root,
                          draws_root=roots.draws_root)
    out: dict[str, Any] = {}
    for arm in ("bc_m", "iql_m"):
        cell = next(c for c in training_cells() if (c.scenario, c.tier, c.arm) == ("grid4x4", "mappo1000", arm))
        name, argv = route_i_command(scratch_roots, cell)
        argv[argv.index("--reuse-root") + 1] = str(Path(roots.output_root) / "p5_1")
        argv[argv.index("--checksums") + 1] = str(Path(roots.output_root) / "SHA256SUMS_p5_1.txt")
        argv = [*argv[: argv.index("evaluate")], "--seeds", "101", *argv[argv.index("evaluate"):]]
        with narrowed_held_out_draws(draws):
            if ts.main(argv) != 0:
                raise RuntimeError(f"route (i) failed on {cell.key}")
        payload = json.loads((Path(scratch_roots.runs_root) / "eval_i" / "grid4x4" /
                              f"eval_mappo1000_{arm}_seed101.json").read_text(encoding="utf-8"))
        values = committed_from_route_i(payload, "grid4x4", cell.key)
        work = scratch / f"a27_route_ii_{arm}"
        work.mkdir(parents=True, exist_ok=True)
        probe = ProbeRoots(repo_root=Path(roots.repo_root), corpus_root=Path(roots.corpus_root),
                           draws_root=Path(roots.draws_root), output_root=Path(roots.output_root), work_dir=work)
        previous = pin_torch_threads(1)
        try:
            outcome = ar.run_campaign([ar.CellKey(*key) for key in sorted(values)], roots=probe,
                                      engine_seed=ENGINE_SEED, device=None, committed=values, protected=(),
                                      factory_builder=a27_factory_builder(scratch_roots))
        finally:
            pin_torch_threads(previous)
        if outcome["n_refused"]:
            raise RuntimeError(f"route (ii) does not reproduce route (i) on {cell.key}")
        out[arm] = {"episodes": len(values), "routes_equal": int(outcome["rolled"]),
                    "att_ours": {f"{k[2]}:{k[3]}": v for k, v in sorted(values.items())}}
    return out


def preflight(roots: Roots, scratch: str | Path, *, device: Any = None,
              draws: Any = (1000, 1001, 1002, 1003, 1004)) -> dict[str, Any]:
    """G2's measurements, in order, all under *scratch*; nothing is written under the main tree's ``output/``."""
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    record = {
        "format_version": PREFLIGHT_FORMAT_VERSION,
        "cell_count_by_arm": cell_count_by_arm(),
        "matched_runs": _preflight_matched_runs(roots, scratch, device),
        "registered_cell_both_routes": _preflight_registered_cell(roots, scratch),
        "a27_grid_path_both_routes": _preflight_a27_grid_path(roots, scratch, draws),
    }
    mixture_roots = Roots(repo_root=roots.repo_root, output_root=scratch / "output", corpus_root=roots.corpus_root,
                          draws_root=roots.draws_root)
    record["mixture_declaration"] = declare_mixture(mixture_roots)
    (scratch / "preflight.json").write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                            encoding="utf-8")
    return record


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> Any:
    """``check``, ``status``, ``collect-random-fresh``, ``declare``, ``train``, ``declare-evaluation``, ``evaluate``,
    ``old-cells``, ``p3a-references``, ``preflight``."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m offline.corrective_round", allow_abbrev=False,
                                     description="P9.1: A27's corrective round, points 1, 2, 3, 4, 6 (BRIEF_46)")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--draws-root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", allow_abbrev=False).add_argument("--stage", choices=["collect", "campaign"], required=True)
    sub.add_parser("status", allow_abbrev=False)
    sub.add_parser("collect-random-fresh", allow_abbrev=False).add_argument(
        "--scenario", choices=list(SCENARIOS_ORDER), required=True)
    declare = sub.add_parser("declare", allow_abbrev=False)
    declare.add_argument("--what", choices=["mixture", "random_fresh"], required=True)
    declare.add_argument("--scenario", choices=list(SCENARIOS_ORDER), default=None)
    train_cmd = sub.add_parser("train", allow_abbrev=False)
    train_cmd.add_argument("--scenario", choices=list(SCENARIOS_ORDER), default=None)
    train_cmd.add_argument("--tier", default=None)
    train_cmd.add_argument("--arm", default=None)
    train_cmd.add_argument("--device", default=None)
    train_cmd.add_argument("--log-every", type=int, default=5000)
    evaluate_cmd = sub.add_parser("evaluate", allow_abbrev=False)
    evaluate_cmd.add_argument("--shard", type=int, default=0)
    evaluate_cmd.add_argument("--of", type=int, default=1)
    evaluate_cmd.add_argument("--scenario", choices=list(SCENARIOS_ORDER), default=None)
    sub.add_parser("declare-evaluation", allow_abbrev=False)
    old_cells_cmd = sub.add_parser("old-cells", allow_abbrev=False)
    old_cells_cmd.add_argument("--shard", type=int, default=0)
    old_cells_cmd.add_argument("--of", type=int, default=1)
    sub.add_parser("p3a-references", allow_abbrev=False)
    preflight_cmd = sub.add_parser("preflight", allow_abbrev=False)
    preflight_cmd.add_argument("--scratch", type=Path, required=True)
    preflight_cmd.add_argument("--device", default=None)
    return parser


def main(argv: Any = None) -> int:
    """Run one stage; a refusal prints ``REFUSED`` with its reason and exits 2 (``iql_correction``'s convention)."""
    import sys

    args = build_parser().parse_args(argv)
    roots = Roots(repo_root=args.repo_root, output_root=args.output_root, corpus_root=args.corpus_root,
                  draws_root=args.draws_root)
    try:
        if args.command == "check":
            result: Any = check(roots, stage=args.stage)
        elif args.command == "status":
            result = status(roots)
        elif args.command == "collect-random-fresh":
            result = str(collect_random_fresh(roots, args.scenario))
        elif args.command == "declare":
            if args.what == "mixture":
                result = declare_mixture(roots)["measured"]
            else:
                if args.scenario is None:
                    raise ValueError("declare --what random_fresh needs --scenario")
                result = declare_random_fresh(roots, args.scenario).get("values_to_type", "declared")
        elif args.command == "train":
            cells = [c for c in training_cells() if args.scenario in (None, c.scenario) and args.tier in (None, c.tier)
                     and args.arm in (None, c.arm)]
            result = train(roots, cells=cells, device=args.device, log_every=args.log_every)
        elif args.command == "evaluate":
            result = evaluate(roots, shard=args.shard, of=args.of, scenario=args.scenario)
        elif args.command == "declare-evaluation":
            result = declare_evaluation(roots)
        elif args.command == "old-cells":
            cells = old_cells_shard(args.shard, args.of)
            assert_evaluation_declared(roots, "old_cells/hz1x1")
            result = {key: value for key, value in roll_old_cells(roots, cells=cells).items() if key != "refused"}
        elif args.command == "preflight":
            result = preflight(roots, args.scratch, device=args.device)
        else:
            result = record_registered_references(roots)
    except (ValueError, PermissionError, FileExistsError, FileNotFoundError, RuntimeError) as exc:
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the driver
    raise SystemExit(main())
