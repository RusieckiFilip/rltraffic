"""P9.3 Phase B (A27 point 8): hangzhou_4x4_gudang's learners -- train, evaluate by both routes, report.

Format versions: the training record ``p9.3-phase-b-train/1.0``.  Alignment convention: none of its own -- every
episode this module trains on is contract C6 v1.1 as ``offline.dataset`` reads it (observation rows ``T + 1``, decision
and outcome rows ``T``; the reward of decision ``t`` is in the ``info`` returned by step ``t``), and every window, prompt
and transition is built by ``offline.tier_sweep`` / ``offline.offline_baselines`` functions called unchanged.

WHAT THIS IS -- AND WHAT IT IS NOT
----------------------------------
``PREREGISTRATION`` A27 (i), label 8: a *conditional generalizability extension registered after the grid results*.
Phase B trains ``dt_nomix``, ``bc_m``, ``bc_top10_m``, ``bc_top10_perix_m`` and ``iql_m`` -- matched exposure and
dropout-free targets from the start -- five seeds, 40,000 steps, on every hz4x4 tier (``offline.hz4x4_ladder``'s
Phase A), and evaluates them on held-out draws 1000-1099 under both ATT definitions.  NOT a new topology, NOT a C3
target, NOT a new claim: a replication beside the grid's ladder.  A11's gate FAILED on hz4x4 by construction (criterion
4; ``docs/data/p9_3_engine_att_gate.json``), so hz4x4 has NO primary definition.

WHY MIRRORS, AND OF WHAT (``BRIEF_48`` section 0; ``docs/plans/p9.3.md`` section 10.2)
-------------------------------------------------------------------------------------
``tier_sweep``'s CLIs resolve a tier through the grid's own table (``tier_spec`` -> ``cf_grid4x4__*``) and key the
baselines' normalisation statistics by the grid's scenario id.  The brief's rule: a scenario-parametrised path that
today assumes ``cf_grid4x4`` is a new function here, not an edit.  So :func:`hz4x4_tier_parts` is the BODY of
``tier_sweep.tier_parts`` on an hz4x4 spec and :func:`hz4x4_baseline_inputs` the input block of
``tier_sweep._run_train_baselines``; every function they call is ``tier_sweep``'s or ``offline_baselines``' own, reached
through the module attribute at call time, and the trainers (``tier_sweep.train_tier_dt``, ``offline_baselines.train_bc``
/ ``train_iql``) are called unchanged.  A test holds each mirror to its original on a grid tier.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import offline.hz4x4_ladder as hz
import offline.tier_sweep as ts

ARTIFACT_FORMAT_TRAIN = "p9.3-phase-b-train/1.0"

#: A27 (i)'s arms, in the registration's order.
ARMS: tuple[str, ...] = ("dt_nomix", "bc_m", "bc_top10_m", "bc_top10_perix_m", "iql_m")

#: The arms' recipes are ``offline.a27_arms``' grid4x4 ones: A27 (i) names the grid's matched arms, and hz4x4's tiers hold
#: the same 1,152,000 decisions on the same lattice, so the same batches give the same exposure
#: (``docs/data/p9_3_declaration.json``).
RECIPE_SCENARIO = "grid4x4"

#: The committed declaration the tier specs are read from (``hz4x4_ladder declare``).
DECLARATION = Path("docs/data/p9_3_declaration.json")

#: Phase B's directory under the output root (A27 (a): new runs write only into new ``output/p9_*`` directories).
PHASE_B_DIR = "p9_3"

#: Every hz4x4 tier trains on all of its 200 episodes (A27 (i): draws 1-200; ``mappo1000`` 5 seeds x 40 draws).
TIER_EPISODES = 200

#: The label every training second carries (A27 (a)).
CLOCK_LABEL = (
    "uncontrolled clock (A27 (a)): wall-clock seconds of one training process on a shared workstation, reported and "
    "never compared as a timing claim"
)


def arm_recipe(arm: str) -> Any:
    """The recipe of one Phase B arm: ``a27_arms.recipe("grid4x4", arm)``, refusing an arm A27 (i) does not name."""
    from offline.a27_arms import recipe

    name = str(arm)
    if name not in ARMS:
        raise ValueError(f"{name!r} is not a Phase B arm: A27 (i) names exactly {list(ARMS)} on hz4x4")
    found = recipe(RECIPE_SCENARIO, name)
    if int(found.steps) != int(ts.DECLARED_GRADIENT_STEPS):
        raise ValueError(f"{name}'s recipe trains {found.steps} steps; A27 (i) declares {ts.DECLARED_GRADIENT_STEPS}")
    return found


def hz4x4_tier_spec(tier: str, *, declaration_path: str | Path | None = None) -> Any:
    """hz4x4's ``tier_sweep.TierSpec`` for *tier*, read from the committed declaration and checked against Phase A.

    The declaration is ``hz4x4_ladder declare``'s artifact; its ``dirs`` must be Phase A's directory names for the tier,
    its ``subsample`` ``"none"`` and its ``episode_count`` :data:`TIER_EPISODES` -- a spec that names another corpus, or
    another selection, is refused before anything reads it.
    """
    import json

    name = str(tier)
    if name not in hz.TIER_ORDER:
        raise ValueError(f"unknown hz4x4 tier {name!r}; Phase A collected {list(hz.TIER_ORDER)}")
    path = Path(declaration_path) if declaration_path is not None else hz._repo_root() / DECLARATION
    declared = json.loads(Path(path).read_bytes())["tier_specs"][name]
    spec = ts.TierSpec(
        tier=str(declared["tier"]),
        dirs=tuple(str(d) for d in declared["dirs"]),
        subsample=str(declared["subsample"]),
        episode_count=int(declared["episode_count"]),
        ladder_att=float(declared["ladder_att"]),
        mean_training_return=float(declared["mean_training_return"]),
    )
    expected = tuple(hz._tier_dir_names(name))
    if spec.tier != name or spec.dirs != expected:
        raise ValueError(f"{path} declares {name} as {spec.tier!r} on {list(spec.dirs)}; Phase A collected it as "
                         f"{list(expected)}")
    if spec.subsample != "none" or spec.episode_count != TIER_EPISODES:
        raise ValueError(f"{path} declares {name} with subsample {spec.subsample!r} and {spec.episode_count} episodes; "
                         f"every hz4x4 tier trains on all {TIER_EPISODES} of its episodes ('none')")
    return spec


def scenario_id_of(spec: Any, corpus_root: str | Path) -> str:
    """The scenario id the tier's corpus was collected under (its manifest's ``run_metadata.scenario_id``).

    The normalisation statistics are keyed by it (``BCAgent._stats.normalize_state(scenario_id, ...)``), so the baselines
    are trained under the corpus's own id -- never the grid's constant ``tier_sweep.SCENARIO_ID``.  Every directory of
    the tier must agree.
    """
    import json

    ids = sorted({
        str(json.loads((directory / "manifest.json").read_bytes())["run_metadata"]["scenario_id"])
        for directory in ts.tier_dirs(spec, corpus_root)
    })
    if len(ids) != 1:
        raise ValueError(f"{spec.tier}'s directories were collected under {ids}; one tier is one scenario")
    return ids[0]


def hz4x4_tier_parts(
    spec: Any, corpus_root: str | Path, *, sim_config: str | Path | None = None, context_length: int | None = None
) -> dict[str, Any]:
    """``tier_sweep.tier_parts``' body on an explicit spec: everything a training needs for one tier.

    Line for line the original but for its first statement (``spec = tier_spec(tier)``, the grid's table): node ids from
    the corpus, the adjacency of the tier's own sim config, the training-split window dataset, the size match, the
    stream returns, the per-node prompts and the joint stack restricted to the chosen episodes.  Every call goes through
    the ``tier_sweep`` module attribute at call time, so a test that replaces one replaces it for both.
    """
    from offline.joint_windows import build_joint_index, stack_joint
    from offline.offline_baselines import stream_returns
    from offline.spatial_mixing import per_node_prompts

    length = int(ts.CONTEXT_LENGTH if context_length is None else context_length)
    node_ids = ts.node_ids_from_corpus(spec, corpus_root)
    adjacency = ts.adjacency_for_tier(spec, corpus_root, node_ids, sim_config)
    dataset = ts.tier_dataset(spec, corpus_root, length)
    all_episodes = ts.episodes_in(dataset)
    chosen = ts.selected_episodes(spec, all_episodes)
    all_streams = stream_returns(dataset)
    streams = ts.streams_of_episodes(all_streams, chosen)
    prompts = per_node_prompts(streams)

    def stack() -> dict[str, Any]:
        """The joint tensors, restricted to the size-matched episode set."""
        index = build_joint_index(dataset, node_ids)
        stacked = stack_joint(dataset, index)
        rows = ts.joint_rows_for_episodes(index.episode_index, [e.episode_index for e in chosen])
        out = dict(stacked)
        out["member_index"] = stacked["member_index"][rows]
        return out

    index = build_joint_index(dataset, node_ids)
    return {
        "spec": spec,
        "node_ids": node_ids,
        "adjacency": adjacency,
        "dataset": dataset,
        "index": index,
        "episodes_all": all_episodes,
        "episodes": chosen,
        "streams": streams,
        "streams_all": all_streams,
        "prompts": prompts,
        "stack": stack,
    }


def hz4x4_baseline_inputs(parts: Mapping[str, Any], corpus_root: str | Path) -> dict[str, Any]:
    """``tier_sweep._run_train_baselines``' input block: the size-matched streams, the two top-decile selections, the
    IQL reward scale, the three BC batches and the transition table (built on first use).

    The scale is taken over the SELECTED streams and the IQL table is ``tier_sweep.iql_transition_table`` on them
    (``DEFERRED`` 106's fix); the provenance block is the CLI's, with the corpus's own scenario id.
    """
    from offline.dt_gate import stack_dataset
    from offline.offline_baselines import filter_stacked_to_streams, iql_reward_scale

    dataset = parts["dataset"]
    group = next(iter(dataset.groups))
    streams = parts["streams"]
    selections = ts.baseline_stream_selections(dataset, streams)
    scale = iql_reward_scale([s.total_return for s in streams])
    stacked_all = stack_dataset(dataset, group)
    batches = {
        "bc": filter_stacked_to_streams(dataset, stacked_all, streams),
        "bc_top10": filter_stacked_to_streams(dataset, stacked_all, selections["bc_top10"]),
        "bc_top10_perix": filter_stacked_to_streams(dataset, stacked_all, selections["bc_top10_perix"]),
    }
    cache: dict[str, Any] = {}

    def table() -> Any:
        """IQL's transition table on the selected streams, built once."""
        if "table" not in cache:
            cache["table"] = ts.iql_transition_table(dataset, group=group, reward_scale=scale, streams=streams)
        return cache["table"]

    provenance = {
        "tier": parts["spec"].tier,
        "dataset_dirs": [str(d) for d in ts.tier_dirs(parts["spec"], corpus_root)],
        "scenario_id": scenario_id_of(parts["spec"], corpus_root),
        "training_streams": len(streams),
        "independent_per_intersection": True,
    }
    return {"group": group, "streams": streams, "selections": selections, "scale": scale, "batches": batches,
            "table": table, "provenance": provenance}


def checkpoint_path(output_root: str | Path, tier: str, arm: str, seed: int) -> Path:
    """``<output_root>/p9_3/checkpoints/hz4x4_<tier>_<arm>_seed<seed>.pt``."""
    return Path(output_root) / PHASE_B_DIR / "checkpoints" / f"{hz.SCENARIO}_{tier}_{arm}_seed{int(seed)}.pt"


def train_record_path(output_root: str | Path, tier: str, arm: str, seed: int) -> Path:
    """``<output_root>/p9_3/train/hz4x4_<tier>_<arm>_seed<seed>.json``."""
    return Path(output_root) / PHASE_B_DIR / "train" / f"{hz.SCENARIO}_{tier}_{arm}_seed{int(seed)}.json"


def train_cell(
    spec: Any,
    arm: str,
    seed: int,
    *,
    corpus_root: str | Path,
    output_root: str | Path,
    device: Any,
    gradient_steps: int,
    log_every: int = 0,
    sim_config: str | Path | None = None,
) -> dict[str, Any]:
    """Train ONE (tier, arm, seed) cell with the registered trainer and write its checkpoint and record once.

    Every refusal precedes the first write: an existing checkpoint or record (a cell is never skipped and never
    overwritten, A27 (a)), an arm or seed A27 (i) does not name, and -- Q-B1 -- a ``dt_nomix`` tier name that
    ``tier_sweep.train_tier_dt`` refuses.  The inputs are built by the two mirrors, the trainer is called unchanged, the
    ``.partial`` file is moved into place by ``tier_sweep.replace_guarded``, and the record carries the trainer's own
    ``seconds`` under :data:`CLOCK_LABEL`, the 1-minute load average before and after, and the file's digests.
    """
    import json
    import os

    import offline.offline_baselines as ob
    from offline.dt_gate import runtime_provenance

    recipe = arm_recipe(arm)
    seed = int(seed)
    if seed not in hz.SEEDS:
        raise ValueError(f"seed {seed} is not one of the registered {list(hz.SEEDS)}")
    steps = int(gradient_steps)
    if steps < 1:
        raise ValueError(f"gradient_steps must be >= 1, got {gradient_steps}")
    final = checkpoint_path(output_root, spec.tier, recipe.name, seed)
    record_path = train_record_path(output_root, spec.tier, recipe.name, seed)
    for path in (final, record_path):
        if path.exists():
            raise FileExistsError(f"{path} exists: a Phase B cell is written once -- never skipped, never overwritten "
                                  "(A27 (a)); remove it deliberately if a re-run is intended")
    if recipe.base == "dt_nomix" and spec.tier not in ts.TIERS:
        raise ValueError(
            f"tier_sweep.train_tier_dt accepts only the grid's tier names {sorted(ts.TIERS)} and refuses "
            f"{spec.tier!r} before reading any data (docs/plans/p9.3.md section 10.2, C2): Q-B1 awaits the "
            "coordinator's ruling"
        )
    parts = hz4x4_tier_parts(spec, corpus_root, sim_config=sim_config)
    scenario_id = scenario_id_of(spec, corpus_root)
    inputs = hz4x4_baseline_inputs(parts, corpus_root) if recipe.base != "dt_nomix" else None
    protected = ts.protected_roots_from([Path(output_root) / "p5_1", Path(output_root) / "p5_2"])
    a27_fields = {
        "scenario": hz.SCENARIO, "scenario_id": scenario_id, "arm": recipe.name, "base_method": recipe.base,
        "target_dropout": recipe.target_dropout, "registered_in": recipe.registered_in, "point": "A27 (i)",
        "campaign": "P9.3 Phase B (BRIEF_48)",
    }
    # Every check is behind us: the writes start here.
    final.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name(final.name + ".partial")
    ts.assert_writable(partial, protected)
    partial.unlink(missing_ok=True)
    load_start = float(os.getloadavg()[0])
    if recipe.base == "dt_nomix":
        result = ts.train_tier_dt(
            parts["stack"](), parts["index"], tier=spec.tier, method="dt_nomix", seed=seed,
            adjacency=parts["adjacency"], prompts=parts["prompts"], stats=parts["dataset"].stats,
            state_dim=int(parts["index"].state_dim), n_actions=int(parts["index"].n_actions), gradient_steps=steps,
            batch_size=int(recipe.batch), device=device, checkpoint_path=partial, protected=protected,
            provenance={"runtime": runtime_provenance(), **a27_fields}, deterministic=False, log_every=int(log_every),
        )
    else:
        assert inputs is not None
        group = inputs["group"]
        provenance = {**inputs["provenance"], **a27_fields, "runtime": runtime_provenance()}
        if recipe.base == "iql":
            extra = {} if recipe.target_dropout in (None, "active") else {"target_dropout": recipe.target_dropout}
            result = ob.train_iql(
                inputs["table"](), state_dim=group[0], n_actions=group[1], seed=seed, declared_gradient_steps=steps,
                batch_size=int(recipe.batch), device=device, checkpoint_path=partial, stats=parts["dataset"].stats,
                scenario_id=scenario_id, provenance=provenance, log_every=int(log_every), **extra,
            )
        else:
            result = ob.train_bc(
                inputs["batches"][recipe.base], state_dim=group[0], n_actions=group[1], seed=seed, method=recipe.base,
                declared_gradient_steps=steps, batch_size=int(recipe.batch), device=device, checkpoint_path=partial,
                stats=parts["dataset"].stats, scenario_id=scenario_id, provenance=provenance, log_every=int(log_every),
            )
    load_end = float(os.getloadavg()[0])
    ts.replace_guarded(partial, final, protected)
    file_sha256 = hz._sha256_file(final)
    if recipe.base == "dt_nomix":
        canonical = ts.canonical_state_dict_digest(final)
    else:
        canonical = str(result.canonical_digest)
        if str(result.file_sha256) != file_sha256:
            raise ValueError(f"{final}: the trainer reports file sha256 {result.file_sha256}, the file on disk is "
                             f"{file_sha256}; the checkpoint changed between the save and the move")
    losses = list(getattr(result, "losses", ()))
    record = {
        "format_version": ARTIFACT_FORMAT_TRAIN,
        "registered_in": "PREREGISTRATION A27 (i), with (a)-(d); BRIEF_48 E.3.6; docs/plans/p9.3.md section 10",
        "scenario": hz.SCENARIO,
        "scenario_id": scenario_id,
        "tier": spec.tier,
        "arm": recipe.name,
        "seed": seed,
        "base_method": recipe.base,
        "batch": int(recipe.batch),
        "target_dropout": recipe.target_dropout,
        "recipe_registered_in": recipe.registered_in,
        "declared_gradient_steps": steps,
        "gradient_steps": int(result.gradient_steps),
        "seconds": float(result.seconds),
        "clock": CLOCK_LABEL,
        "load_average_1min": {"start": load_start, "end": load_end},
        "final_loss": float(losses[-1]) if losses else None,
        "checkpoint_path": str(final),
        "file_sha256": file_sha256,
        "canonical_digest": canonical,
        "dataset_dirs": [str(d) for d in ts.tier_dirs(spec, corpus_root)],
        "training_streams": len(parts["streams"]),
        "device": str(device),
        "git": hz._git(),
    }
    record = json.loads(json.dumps(record))
    hz._write_artifact(record, record_path)
    return record


# ----------------------------------------------------------------------
# The command line
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """``train`` (one cell, written once).  Run from the main tree with this branch's code on ``PYTHONPATH``."""
    parser = argparse.ArgumentParser(prog="python -P -m offline.hz4x4_phase_b",
                                     description="P9.3 Phase B (A27 point 8): hz4x4's learners.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train one (tier, arm, seed) cell on CUDA, written once")
    train.add_argument("--tier", required=True, choices=list(hz.TIER_ORDER))
    train.add_argument("--arm", required=True, choices=list(ARMS))
    train.add_argument("--seed", required=True, type=int, choices=list(hz.SEEDS))
    train.add_argument("--device", default="cuda", help="cuda (the campaign's, A-B4) or cpu (tests)")
    train.add_argument("--gradient-steps", type=int, default=int(ts.DECLARED_GRADIENT_STEPS),
                       help="the registered 40,000; another value only outside the campaign's output root (timings)")
    train.add_argument("--corpus-root", default=str(hz.MAIN_CORPUS_ROOT))
    train.add_argument("--output-root", default=str(hz.MAIN_OUTPUT_ROOT))
    train.add_argument("--torch-threads", type=int, default=1)
    train.add_argument("--log-every", type=int, default=0)
    return parser


def assert_campaign_steps(output_root: str | Path, gradient_steps: int) -> None:
    """The campaign's output root takes the registered step count only: a timing or a test writes elsewhere."""
    if int(gradient_steps) != int(ts.DECLARED_GRADIENT_STEPS) and Path(output_root).resolve() == Path(
            hz.MAIN_OUTPUT_ROOT).resolve():
        raise ValueError(f"{output_root} is the campaign's output root: its cells train the registered "
                         f"{ts.DECLARED_GRADIENT_STEPS} steps, never {gradient_steps}; a timing writes elsewhere")


def _run_train(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(int(args.torch_threads))
    assert_campaign_steps(args.output_root, args.gradient_steps)
    corpus_root = hz.assert_main_tree_path(args.corpus_root, what="the corpus root")
    output_root = hz.assert_main_tree_path(args.output_root, what="the output root")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("no CUDA device is visible; Phase B trains on CUDA (docs/plans/p9.3.md section 10.5, A-B4)")
    spec = hz4x4_tier_spec(args.tier)
    record = train_cell(spec, args.arm, args.seed, corpus_root=corpus_root, output_root=output_root,
                        device=torch.device(args.device), gradient_steps=int(args.gradient_steps),
                        log_every=int(args.log_every))
    print(f"{record['tier']}/{record['arm']} seed {record['seed']}: {record['gradient_steps']} steps in "
          f"{record['seconds']:.1f} s ({CLOCK_LABEL.split(':')[0]}), load {record['load_average_1min']}, "
          f"-> {record['checkpoint_path']}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(None if argv is None else list(argv))
    if args.command == "train":
        return _run_train(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover - the CLI is exercised through main() by the tests
    raise SystemExit(main())
