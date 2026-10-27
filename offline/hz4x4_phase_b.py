"""P9.3 Phase B (A27 point 8): hangzhou_4x4_gudang's learners -- train, evaluate by both routes, report.

Format versions: the training record ``p9.3-phase-b-train/1.0``; the evaluation record ``p9.3-phase-b-eval/1.0`` (one
per arm, seed and held-out draw, both routes in it).  Alignment convention: none of its own -- every
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
through the module attribute at call time, and the baselines' trainers (``offline_baselines.train_bc`` / ``train_iql``)
are called unchanged.  ``tier_sweep.train_tier_dt`` assumes the grid twice -- it accepts only the grid's tier names and
writes the grid's scenario id into every DT checkpoint, the key the agent normalises states with -- so the DT trains
through :func:`hz4x4_train_tier_dt`, its body with those two constants made parameters.  A test holds each mirror to its
original on a grid tier; the trainer's, bit for bit on CPU.

EVALUATION
----------
Every cell on held-out draws 1000-1099 x the five seeds, each episode by BOTH routes -- ``dt_gate.evaluate_arm`` for one
draw, then ``admission_probe.probe_episode`` -- STOPPED unless route (ii)'s ``att_ours`` equals route (i)'s under ``==``
(``att_rederivation.assert_reproduces_committed``), each route under B.3's per-episode watchdog, each record written once.
``att_rederivation.run_campaign`` cannot run an hz4x4 cell (its scenario map holds hz1x1 and grid4x4 only), so its
per-cell loop is restated here around the two unchanged rollouts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

#: The tier names hz4x4's DT trainer accepts (``hz4x4_train_tier_dt``'s *tiers*): Phase A's four, ``random_fresh`` included.
DT_TIERS: tuple[str, ...] = tuple(hz.TIER_ORDER)

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


def hz4x4_train_tier_dt(
    stacked: Mapping[str, Any],
    index: Any,
    *,
    tier: str,
    method: str,
    seed: int,
    adjacency: Any,
    prompts: Mapping[str, Any],
    stats: Any,
    state_dim: int,
    n_actions: int,
    gradient_steps: int,
    batch_size: int,
    device: Any,
    checkpoint_path: str | Path,
    protected: Sequence[Path],
    provenance: Mapping[str, Any],
    scenario_id: str,
    tiers: Sequence[str],
    deterministic: bool = False,
    log_every: int = 0,
) -> Any:
    """``tier_sweep.train_tier_dt``'s body with its two grid constants made parameters, and nothing else.

    ``tier_sweep.train_tier_dt`` accepts only the grid's tier names (``tier not in TIERS``) and writes the grid's
    ``SCENARIO_ID`` into every checkpoint.  That id is the key ``SpatialDTAgent`` normalises states with, so an hz4x4 DT
    trained by it cannot act on hz4x4 -- measured: ``KeyError: no normalisation statistics for scenario
    'cityflow_grid4x4' ... fitted on ['cityflow4x4']`` (``docs/plans/p9.3.md`` section 10.2, C2 as amended).  Here the
    accepted tier names are *tiers* and the recorded id is *scenario_id*; every other line is the original's, every
    helper and constant is ``tier_sweep``'s or ``spatial_mixing``'s, reached at call time, so this function cannot drift
    from the original on a value.  A test holds it to the original on CPU: identical losses and an identical checkpoint,
    tensor for tensor, when it is given the grid's two values.
    """
    import time

    import numpy as np
    import torch

    from agent.DTAgent import action_loss
    from agent.SpatialDTAgent import (
        SPATIAL_CHECKPOINT_FORMAT_VERSION,
        SpatialDecisionTransformer,
        SpatialDTConfig,
    )
    from agent.utils.utils import Utils
    from offline.spatial_mixing import GRAD_CLIP, LEARNING_RATE, WEIGHT_DECAY

    if method not in ts.DT_METHODS:
        raise ValueError(
            f"{method!r} is not a spatial arm; this trainer builds {list(ts.DT_METHODS)}. The "
            "per-intersection comparators are trained by offline.offline_baselines, imported "
            "rather than duplicated"
        )
    if tier not in tiers:
        raise ValueError(f"unknown tier {tier!r}; this task declares {sorted(tiers)}")
    total = int(gradient_steps)
    if total < 1:
        raise ValueError(f"gradient_steps must be >= 1, got {gradient_steps}")
    if list(index.node_ids) != list(adjacency.node_ids):
        raise ValueError(
            f"the joint index node order {list(index.node_ids)[:6]} is not the adjacency's "
            f"{list(adjacency.node_ids)[:6]}; the mask would apply to the wrong columns"
        )
    missing = [ix for ix in index.node_ids if ix not in prompts]
    if missing:
        raise ValueError(f"no declared prompt for {missing[:8]}")
    ts.assert_process_regime(deterministic)
    # Filesystem-mutation barrier: the destination is guarded BEFORE any work, so a refused run
    # trains nothing and writes nothing (docs/plans/p5.2.md section 3.2).
    destination = ts.assert_writable(checkpoint_path, protected)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {destination.parent}; nothing is created here"
        )

    n_head = int(ts.N_HEAD_BY_METHOD[method])
    spatial = method in ("dt_spatial", "dt_spatial_h4")
    mask = adjacency.attention_mask(spatial_mixing=spatial)

    Utils.seed_everything(int(seed), seed_python_random=False)
    config = SpatialDTConfig(
        state_dim=int(state_dim),
        n_actions=int(n_actions),
        n_nodes=index.n_nodes,
        context_length=int(stacked["state"].shape[1]),
        max_ep_len=int(stacked["timestep"].max()) + 1,
        spatial_mixing=spatial,
        n_head=n_head,
    )
    model = SpatialDecisionTransformer(config).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    warmup = ts.warmup_for(total)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: ts.lr_multiplier(step, warmup)
    )

    members = stacked["member_index"]
    count = int(members.shape[0])
    if count < 1:
        raise ValueError("the joint index is empty")
    mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.bool_)).to(device)
    scale = (
        torch.tensor([prompts[ix].rtg_scale for ix in index.node_ids], dtype=torch.float32)
        .view(1, index.n_nodes, 1, 1)
        .to(device)
    )

    generator = np.random.default_rng(int(seed))
    losses: list[float] = []
    model.train()
    started = time.time()
    for step in range(total):
        rows = torch.from_numpy(
            generator.integers(0, count, size=int(batch_size)).astype(np.int64)
        )
        selected = members[rows]
        action = stacked["action"][selected].to(device)
        logits = model(
            stacked["rtg"][selected].to(device) / scale,
            stacked["state"][selected].to(device),
            action,
            stacked["timestep"][selected].to(device),
            mask_tensor,
            stacked["attention_mask"][selected].to(device),
            stacked["avail_mask"][selected].to(device),
        )
        loss = action_loss(logits, action)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimiser.step()
        schedule.step()
        losses.append(float(loss.detach()))
        if log_every and (step + 1) % log_every == 0:
            print(
                f"  {tier}/{method} seed {seed} step {step + 1}/{total} "
                f"loss {np.mean(losses[-log_every:]):.5f}",
                flush=True,
            )
    seconds = time.time() - started

    torch.save(
        {
            "format_version": SPATIAL_CHECKPOINT_FORMAT_VERSION,
            "config": config.to_json_obj(),
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "target_rtg": {ix: prompts[ix].target_rtg for ix in index.node_ids},
            "rtg_scale": {ix: prompts[ix].rtg_scale for ix in index.node_ids},
            "normalise": True,
            "scenario_id": str(scenario_id),
            "stats": stats.to_json_obj(),
            "intersection_ids": list(index.node_ids),
            "spatial_mask": np.asarray(mask, dtype=np.bool_).tolist(),
            "provenance": {
                **dict(provenance),
                "method": method,
                "tier": tier,
                "n_head": n_head,
                "deterministic": bool(deterministic),
                "seed": int(seed),
                "gradient_steps": int(total),
                "batch_size": int(batch_size),
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_steps": int(warmup),
                "grad_clip": GRAD_CLIP,
                "device": str(device),
                "spatial_mixing": spatial,
                "roadnet_sha256": adjacency.roadnet_sha256,
            },
        },
        destination,
    )
    from offline.spatial_mixing import TrainResult

    return TrainResult(
        method=method,
        seed=int(seed),
        gradient_steps=total,
        losses=tuple(losses),
        checkpoint_path=str(destination),
        seconds=float(seconds),
    )


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
    overwritten, A27 (a)), an arm or seed A27 (i) does not name, and a ``dt_nomix`` tier name outside :data:`DT_TIERS`.
    The inputs are built by the two mirrors; the baselines' trainers are called unchanged and the DT trains through
    :func:`hz4x4_train_tier_dt` (``tier_sweep.train_tier_dt``'s body with hz4x4's scenario id and tier names); the
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
    if recipe.base == "dt_nomix" and spec.tier not in DT_TIERS:
        raise ValueError(f"{spec.tier!r} is not one of hz4x4's tiers {list(DT_TIERS)}; the DT trainer "
                         "(hz4x4_train_tier_dt) accepts no other name, and nothing has been read or written")
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
        result = hz4x4_train_tier_dt(
            parts["stack"](), parts["index"], tier=spec.tier, method="dt_nomix", seed=seed,
            adjacency=parts["adjacency"], prompts=parts["prompts"], stats=parts["dataset"].stats,
            state_dim=int(parts["index"].state_dim), n_actions=int(parts["index"].n_actions), gradient_steps=steps,
            batch_size=int(recipe.batch), device=device, checkpoint_path=partial, protected=protected,
            provenance={"runtime": runtime_provenance(), **a27_fields}, scenario_id=scenario_id, tiers=DT_TIERS,
            deterministic=False, log_every=int(log_every),
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
# The training queue (E.3.7: never two trainings on the GPU at once)
# ----------------------------------------------------------------------


def training_cells(
    *, tiers: Sequence[str] | None = None, arms: Sequence[str] | None = None
) -> list[tuple[str, str, int]]:
    """Every (tier, arm, seed) of Phase B, tier by tier -- a tier's evaluation can start once its 25 cells exist."""
    chosen_tiers = [str(t) for t in (hz.TIER_ORDER if tiers is None else tiers)]
    chosen_arms = [str(a) for a in (ARMS if arms is None else arms)]
    unknown = [t for t in chosen_tiers if t not in hz.TIER_ORDER] + [a for a in chosen_arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown tier(s) or arm(s) {unknown}; Phase B trains {list(ARMS)} on {list(hz.TIER_ORDER)}")
    return [(tier, arm, int(seed)) for tier in hz.TIER_ORDER if tier in chosen_tiers
            for arm in ARMS if arm in chosen_arms for seed in hz.SEEDS]


def run_train_queue(cells: Sequence[tuple[str, str, int]], *, output_root: str | Path, run_cell: Any) -> dict[str, Any]:
    """Train the cells in order, ONE at a time: a cell whose record and checkpoint are on disk -- the checkpoint's sha256
    the record's -- is skipped; a half-written cell (a record whose checkpoint differs or is missing, a checkpoint
    without a record) is refused before anything runs; *run_cell* trains the rest and the queue stops at the first
    non-zero exit."""
    import json

    pending: list[tuple[str, str, int]] = []
    skipped = 0
    for tier, arm, seed in cells:
        checkpoint, record = checkpoint_path(output_root, tier, arm, seed), train_record_path(output_root, tier, arm, seed)
        if record.exists():
            recorded = json.loads(record.read_bytes()).get("file_sha256")
            if not checkpoint.is_file() or hz._sha256_file(checkpoint) != recorded:
                raise ValueError(f"{checkpoint} is not the checkpoint its record {record} names; a half-written cell "
                                 "is never trained over or skipped -- resolve it by hand")
            skipped += 1
        elif checkpoint.exists():
            raise ValueError(f"{checkpoint} exists without its record {record}; a half-written cell is never trained "
                             "over or skipped -- resolve it by hand")
        else:
            pending.append((tier, arm, int(seed)))
    trained = 0
    for tier, arm, seed in pending:
        code = int(run_cell(tier, arm, seed))
        if code != 0:
            return {"status": "failed", "skipped": skipped, "trained": trained, "failed_cell": [tier, arm, seed],
                    "returncode": code}
        trained += 1
    return {"status": "completed", "skipped": skipped, "trained": trained, "failed_cell": None, "returncode": 0}


# ----------------------------------------------------------------------
# Evaluation: every cell on held-out draws 1000-1099 x the five seeds, by both routes (docs/plans/p9.3.md section 10.3)
# ----------------------------------------------------------------------

ARTIFACT_FORMAT_EVAL = "p9.3-phase-b-eval/1.0"

#: The references, on the learners' (seed, draw) grid (A-B3): each tier's behaviour policy, and the registered collapse
#: reference ``random`` filed as P5.2 filed its re-rolled anchor (``random@mappo1000``).
REFERENCE_CELLS: tuple[tuple[str, str], ...] = (
    ("behaviour", "maxpressure"),
    ("behaviour", "fixedtime"),
    ("behaviour", "mappo1000"),
    ("behaviour", "random_fresh"),
    ("random", "mappo1000"),
)


class ReproductionError(ValueError):
    """Route (ii)'s ``att_ours`` did not reproduce route (i)'s under ``==``: the evaluation STOPS."""


@dataclass(frozen=True)
class EvalCell:
    """One evaluated arm: a Phase B arm, ``behaviour`` or ``random``, on one hz4x4 tier."""

    method: str
    tier: str

    @property
    def arm(self) -> str:
        """``"<method>@<tier>"`` -- the arm label every record carries."""
        return f"{self.method}@{self.tier}"


def evaluation_cells() -> tuple[EvalCell, ...]:
    """The 20 learner cells (tier by tier, arms in A27 (i)'s order) and the five references."""
    learners = tuple(EvalCell(arm, tier) for tier in hz.TIER_ORDER for arm in ARMS)
    return learners + tuple(EvalCell(method, tier) for method, tier in REFERENCE_CELLS)


def planned_episodes(
    cells: Sequence[EvalCell], *, seeds: Sequence[int] = hz.SEEDS, draws: Sequence[int] | None = None
) -> list[tuple[EvalCell, int, int]]:
    """Every (cell, seed, draw) in cell, seed, draw order; *draws* defaults to ``dt_gate.HELD_OUT_DRAWS``."""
    from offline.dt_gate import HELD_OUT_DRAWS

    pool = tuple(int(d) for d in (HELD_OUT_DRAWS if draws is None else draws))
    return [(cell, int(seed), draw) for cell in cells for seed in seeds for draw in pool]


def eval_record_path(output_root: str | Path, cell: EvalCell, seed: int, draw_id: int) -> Path:
    """``<output_root>/p9_3/eval/<method>@<tier>__seed<seed>__draw<draw>.json``."""
    return Path(output_root) / PHASE_B_DIR / "eval" / f"{cell.arm}__seed{int(seed)}__draw{int(draw_id)}.json"


def refusal_record_path(output_root: str | Path, cell: EvalCell, seed: int, draw_id: int) -> Path:
    """``<output_root>/p9_3/eval_refused/<method>@<tier>__seed<seed>__draw<draw>.json``: never read by a verdict."""
    return Path(output_root) / PHASE_B_DIR / "eval_refused" / f"{cell.arm}__seed{int(seed)}__draw{int(draw_id)}.json"


def write_eval_record(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Write one record once and atomically (``dt_gate.write_json_atomic``); an existing file is refused."""
    from offline.dt_gate import write_json_atomic

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"{destination} exists: an evaluation record is written once and never replaced")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(dict(record), destination)
    return destination


def pending_episodes(output_root: str | Path, plan: Sequence[tuple[EvalCell, int, int]]) -> list[tuple[EvalCell, int, int]]:
    """The planned episodes with no record on disk; every existing record is read and checked against its episode.

    A record filed under another episode's name, or one that does not say it reproduced, is refused: a restart resumes
    from records it has read, never from file names alone (the gate's rule, ``hz4x4_ladder.pending_gate_cells``).
    """
    import json

    pending: list[tuple[EvalCell, int, int]] = []
    for cell, seed, draw in plan:
        path = eval_record_path(output_root, cell, seed, draw)
        if not path.exists():
            pending.append((cell, seed, draw))
            continue
        record = json.loads(path.read_bytes())
        found = (record.get("arm"), record.get("seed"), record.get("draw_id"), record.get("reproduces"))
        if found != (cell.arm, int(seed), int(draw), True):
            raise ValueError(f"{path} is filed under {cell.arm} seed {seed} draw {draw} but holds {found[:3]} "
                             f"(reproduces={found[3]}); a misfiled record is never resumed from")
    return pending


def evaluation_settings(corpus_root: str | Path) -> dict[str, Any]:
    """The evaluation env settings, read from every hz4x4 tier's manifests, refused unless they all agree."""
    import json

    from offline.dt_gate import env_settings_from_manifest

    seen: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for tier in hz.TIER_ORDER:
        for name in hz._tier_dir_names(tier):
            candidate = env_settings_from_manifest(Path(corpus_root) / name / "manifest.json")
            seen.setdefault(json.dumps(candidate, sort_keys=True, default=str), (candidate, []))[1].append(name)
    if len(seen) != 1:
        raise ValueError(f"hz4x4's tiers disagree on the evaluation env settings ({[v[1] for v in seen.values()]}); "
                         "one ladder is evaluated under one setting")
    return dict(next(iter(seen.values()))[0])


def eval_factory(
    cell: EvalCell, seed: int, *, output_root: str | Path, corpus_root: str | Path, draws_root: str | Path, device: Any
) -> tuple[Any, dict[str, Any]]:
    """The action factory of one (cell, seed) and a description of the policy it builds, every input verified.

    * a learner: its checkpoint, refused unless the file's sha256 is the one its training record names; the DT through
      ``SpatialDTAgent.from_checkpoint`` acting with ``explore=False, update_memory=True`` (``tier_sweep._arm_factory``'s
      DT body), the matched baselines through ``offline_baselines._baseline_factory`` with their BASE method -- an
      ``iql_m`` checkpoint is an IQL agent (``BRIEF_46`` 3.2);
    * ``random@mappo1000``: ``method_tier_grid._random_factory(seed)``, the registered collapse reference;
    * ``behaviour@maxpressure``: ``dt_gate._maxpressure_factory``;
    * ``behaviour@fixedtime``: ``method_tier_grid._fixedtime_factory`` on the tier's own manifest (it asserts the
      collected plan's hash and schedule source; k = 2);
    * ``behaviour@mappo1000``: ``dt_gate._mappo_factory`` on the teacher of seed *seed*, the path and sha256 its corpus
      manifest records (manifest paths are relative to the tree the corpus root sits in);
    * ``behaviour@random_fresh``: ``a27_arms.fresh_random_factory`` (A27 (d): ``default_rng([seed, draw_id])``).
    """
    import json

    from offline import dt_gate, method_tier_grid, offline_baselines

    seed = int(seed)
    if cell.method in ARMS:
        recipe = arm_recipe(cell.method)
        path = checkpoint_path(output_root, cell.tier, cell.method, seed)
        record_path = train_record_path(output_root, cell.tier, cell.method, seed)
        if not path.is_file() or not record_path.is_file():
            raise FileNotFoundError(f"{cell.arm} seed {seed}: the checkpoint {path} or its training record {record_path} "
                                    "is missing; a cell is evaluated only once it is trained")
        record = json.loads(record_path.read_bytes())
        sha = hz._sha256_file(path)
        if sha != record["file_sha256"]:
            raise ValueError(f"{path} is not the checkpoint its training record {record_path} names (sha256 {sha} "
                             f"against {record['file_sha256']})")
        source = {"policy": f"the {cell.method} checkpoint of seed {seed}", "checkpoint_path": str(path),
                  "file_sha256": sha, "training_record": str(record_path)}
        if recipe.base == "dt_nomix":
            def factory(env: Any) -> Any:
                from agent.SpatialDTAgent import SpatialDTAgent

                agent = SpatialDTAgent.from_checkpoint(env, str(path), device=device)
                return lambda _env, info: agent.act(info, explore=False, update_memory=True)

            return factory, source
        declared = int(record["declared_gradient_steps"])
        return offline_baselines._baseline_factory(recipe.base, str(path), declared, device), source
    if cell.method == "random" and cell.tier == "mappo1000":
        return method_tier_grid._random_factory(seed), {
            "policy": "offline.collect's random policy with numpy.random.default_rng(training seed) per draw -- the "
                      "registered collapse reference (method_tier_grid._random_factory)"}
    if cell.method != "behaviour":
        raise ValueError(f"{cell.arm} is not an evaluation cell of Phase B; the cells are {[c.arm for c in evaluation_cells()]}")
    if cell.tier == "maxpressure":
        return dt_gate._maxpressure_factory, {"policy": "MaxPressure (dt_gate._maxpressure_factory)"}
    if cell.tier == "random_fresh":
        from offline.a27_arms import fresh_random_factory

        factory = fresh_random_factory(seed, scenario_key=hz.SCENARIO_KEY, draws_root=draws_root,
                                       draw_ids=[int(d) for d in dt_gate.HELD_OUT_DRAWS])
        return factory, {"policy": "A27 (d)'s fresh random reference: offline.collect's random policy with "
                                   "numpy.random.default_rng([seed, draw_id]) per held-out draw (a27_arms.fresh_random_factory)"}
    if cell.tier == "fixedtime":
        from offline.materialise_draws import draw_config_path

        manifest = Path(corpus_root) / f"{hz.CORPUS_PREFIX}__fixedtime" / "manifest.json"
        collected = method_tier_grid.fixedtime_collection_settings(manifest)
        config = draw_config_path(hz.SCENARIO_KEY, int(dt_gate.HELD_OUT_DRAWS[0]), out_root=draws_root)
        return method_tier_grid._fixedtime_factory(config, collected), {
            "policy": "the collected fixed-time plan (method_tier_grid._fixedtime_factory: plan hash and schedule "
                      "source asserted against the tier's manifest)", **{k: collected[k] for k in sorted(collected)}}
    if cell.tier == "mappo1000":
        manifest = Path(corpus_root) / f"{hz.CORPUS_PREFIX}__mappo1000__seed{seed}" / "manifest.json"
        meta = json.loads(manifest.read_bytes())["run_metadata"]
        teacher = Path(corpus_root).resolve().parent / str(meta["checkpoint"])
        sha = hz._sha256_file(teacher)
        if sha != meta["checkpoint_sha256"]:
            raise ValueError(f"{teacher} is not the checkpoint {manifest} records (sha256 {sha})")
        return dt_gate._mappo_factory(str(teacher), device), {
            "policy": f"the MAPPO@1000 teacher of seed {seed} (dt_gate._mappo_factory)", "checkpoint_path": str(teacher),
            "checkpoint_sha256": sha}
    raise ValueError(f"{cell.arm} is not an evaluation cell of Phase B")


def evaluate_episode(
    cell: EvalCell,
    seed: int,
    draw_id: int,
    *,
    factory: Any,
    settings: Mapping[str, Any],
    config_path: str | Path,
    created: int,
    watchdog_seconds: float = hz.WATCHDOG_SECONDS,
) -> dict[str, Any]:
    """ONE held-out episode by route (i) then route (ii), each under the per-episode watchdog; STOP unless they agree.

    Route (i) is ``dt_gate.evaluate_arm`` for this one draw (one call per draw equals one call over the draws: the
    rollout builds a fresh env and a fresh policy per draw); route (ii) is ``admission_probe.probe_episode`` -- the same
    rollout with the admission read and the engine's ATT inserted.  ``att_rederivation.assert_reproduces_committed``
    holds route (ii)'s ``att_ours`` to route (i)'s under ``==``; a mismatch raises :class:`ReproductionError`.
    """
    import time

    from offline import admission_probe, dt_gate
    from offline.att_rederivation import CellKey, assert_reproduces_committed

    seed, draw = int(seed), int(draw_id)
    config = Path(config_path)
    started = time.perf_counter()
    with hz.episode_watchdog(float(watchdog_seconds)):
        (first,) = dt_gate.evaluate_arm(
            arm=cell.arm, seed=seed, draw_ids=[draw], config_for_draw=lambda _draw: config, env_settings=dict(settings),
            scenario_id=hz.SCENARIO_ID, choose_action_factory=factory, engine_seed=hz.BASE_SEED,
        )
    seconds_route_i = time.perf_counter() - started
    with hz.episode_watchdog(float(watchdog_seconds)):
        second = admission_probe.probe_episode(
            scenario=hz.SCENARIO, tier=cell.tier, method=cell.method, arm=cell.arm, seed=seed, draw_id=draw,
            config_path=config, env_settings=dict(settings), scenario_id=hz.SCENARIO_ID,
            choose_action_factory=factory, engine_seed=hz.BASE_SEED, created=int(created),
        )
    key = CellKey(scenario=hz.SCENARIO, arm=cell.arm, seed=seed, draw_id=draw)
    try:
        assert_reproduces_committed(key, second.att_ours, {(key.scenario, key.arm, key.seed, key.draw_id):
                                                           float(first.att_horizon)})
    except ValueError as exc:
        raise ReproductionError(f"route (ii) against route (i): {exc}") from exc
    return {
        "format_version": ARTIFACT_FORMAT_EVAL,
        "scenario": hz.SCENARIO,
        "arm": cell.arm,
        "method": cell.method,
        "tier": cell.tier,
        "seed": seed,
        "draw_id": draw,
        "engine_seed": int(hz.BASE_SEED),
        "att_ours": float(second.att_ours),
        "att_engine": float(second.att_engine),
        "created": int(second.created),
        "entered": int(second.entered),
        "never_entered": int(second.never_entered),
        "entered_fraction": float(second.entered_fraction),
        "route_i": {"att_horizon": float(first.att_horizon), "horizon_vehicle_count": float(first.horizon_vehicle_count),
                    "episode_reward": float(first.episode_reward)},
        "route_ii": second.as_record(),
        "seconds_route_i": float(seconds_route_i),
        "reproduces": True,
    }


def run_episodes(
    output_root: str | Path, plan: Sequence[tuple[EvalCell, int, int]], evaluate: Any
) -> int:
    """Roll every pending episode of *plan* with *evaluate* and write its record; on a ReproductionError file a refusal
    and re-raise.  Returns the number rolled."""
    rolled = 0
    for cell, seed, draw in pending_episodes(output_root, plan):
        try:
            record = evaluate(cell, seed, draw)
        except ReproductionError as exc:
            write_eval_record(refusal_record_path(output_root, cell, seed, draw), {
                "format_version": ARTIFACT_FORMAT_EVAL, "scenario": hz.SCENARIO, "arm": cell.arm, "seed": int(seed),
                "draw_id": int(draw), "reproduces": False, "refusal": str(exc)})
            raise
        if (record.get("arm"), record.get("seed"), record.get("draw_id")) != (cell.arm, int(seed), int(draw)):
            raise ValueError(f"the evaluation of {cell.arm} seed {seed} draw {draw} returned the record of "
                             f"{record.get('arm')} seed {record.get('seed')} draw {record.get('draw_id')}")
        write_eval_record(eval_record_path(output_root, cell, seed, draw), record)
        rolled += 1
    return rolled


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
    queue = sub.add_parser("train-queue", help="train every pending cell, one process at a time, stopping at a failure")
    queue.add_argument("--tiers", nargs="+", default=None, choices=list(hz.TIER_ORDER))
    queue.add_argument("--arms", nargs="+", default=None, choices=list(ARMS))
    queue.add_argument("--device", default="cuda")
    queue.add_argument("--gradient-steps", type=int, default=int(ts.DECLARED_GRADIENT_STEPS))
    queue.add_argument("--corpus-root", default=str(hz.MAIN_CORPUS_ROOT))
    queue.add_argument("--output-root", default=str(hz.MAIN_OUTPUT_ROOT))
    evaluate = sub.add_parser("evaluate", help="the evaluation driver: an evaluate-worker under B.3's restarts")
    worker = sub.add_parser("evaluate-worker", help="roll the selection's pending episodes, each under the watchdog")
    for each in (evaluate, worker):
        each.add_argument("--cells", nargs="+", default=None, help="arm labels <method>@<tier>; default: all 25")
        each.add_argument("--draws", nargs="+", type=int, default=None,
                          help="held-out draws; default: all 100 (a timing names a few)")
        each.add_argument("--shard", type=int, default=0, help="this process's shard of the episodes (0-based)")
        each.add_argument("--of", type=int, default=1, help="the number of shards")
        each.add_argument("--device", default="cpu", help="cpu (the plan's evaluation device, A-B4)")
        each.add_argument("--corpus-root", default=str(hz.MAIN_CORPUS_ROOT))
        each.add_argument("--draws-root", default=str(hz.MAIN_DRAWS_ROOT))
        each.add_argument("--output-root", default=str(hz.MAIN_OUTPUT_ROOT))
        each.add_argument("--torch-threads", type=int, default=1)
    worker.add_argument("--watchdog-seconds", type=float, default=float(hz.WATCHDOG_SECONDS))
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


def selected_plan(cells: Sequence[str] | None, draws: Sequence[int] | None, *, shard: int, of: int
                  ) -> list[tuple[EvalCell, int, int]]:
    """The episodes one evaluation process rolls: the named cells (default all), every seed, the named draws (default
    all held-out), interleaved into *of* shards so every shard carries every arm's cost."""
    by_label = {cell.arm: cell for cell in evaluation_cells()}
    labels = list(cells) if cells else list(by_label)
    unknown = [label for label in labels if label not in by_label]
    if unknown:
        raise ValueError(f"unknown evaluation cell(s) {unknown}; the cells are {list(by_label)}")
    if not 0 <= int(shard) < int(of):
        raise ValueError(f"shard {shard} of {of} is not a shard")
    plan = planned_episodes([by_label[label] for label in labels], draws=draws)
    return plan[int(shard)::int(of)]


def _eval_roots(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    return (hz.assert_main_tree_path(args.corpus_root, what="the corpus root"),
            hz.assert_main_tree_path(args.draws_root, what="the draws root"),
            hz.assert_main_tree_path(args.output_root, what="the output root"))


def _run_evaluate_worker(args: argparse.Namespace) -> int:
    """``evaluate-worker``: roll every pending episode of the selection, each route under the watchdog (B.3)."""
    import torch

    from offline.admission_probe import created_from_flow
    from offline.materialise_draws import draw_config_path

    torch.set_num_threads(int(args.torch_threads))
    corpus_root, draws_root, output_root = _eval_roots(args)
    plan = selected_plan(args.cells, args.draws, shard=args.shard, of=args.of)
    settings = evaluation_settings(corpus_root)
    horizon = int(settings["max_steps"]) * int(settings["delta_time"])
    factories: dict[tuple[EvalCell, int], tuple[Any, dict[str, Any]]] = {}

    def evaluate(cell: EvalCell, seed: int, draw: int) -> dict[str, Any]:
        if (cell, seed) not in factories:
            factories[(cell, seed)] = eval_factory(cell, seed, output_root=output_root, corpus_root=corpus_root,
                                                   draws_root=draws_root, device=args.device)
        factory, source = factories[(cell, seed)]
        config = Path(draw_config_path(hz.SCENARIO_KEY, int(draw), out_root=draws_root))
        created = created_from_flow(config.parent / "flow.json", horizon_seconds=horizon)
        record = evaluate_episode(cell, seed, draw, factory=factory, settings=settings, config_path=config,
                                  created=created, watchdog_seconds=float(args.watchdog_seconds))
        return {**record, "policy_source": source}

    pending = pending_episodes(output_root, plan)
    print(f"evaluate worker {args.shard}/{args.of}: {len(plan) - len(pending)} of {len(plan)} episodes on disk; "
          f"rolling {len(pending)}", flush=True)
    try:
        rolled = run_episodes(output_root, plan, evaluate)
    except ReproductionError as exc:
        print(f"STOP: {exc}", flush=True)
        return 3
    print(f"evaluate worker {args.shard}/{args.of}: rolled {rolled}", flush=True)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    """``evaluate``: validate the selection, then run an ``evaluate-worker`` under ``run_with_restarts`` (B.3)."""
    import os
    import sys

    from offline.materialise_draws import draw_config_path

    corpus_root, draws_root, output_root = _eval_roots(args)
    plan = selected_plan(args.cells, args.draws, shard=args.shard, of=args.of)
    evaluation_settings(corpus_root)
    for cell, seed in dict.fromkeys((cell, seed) for cell, seed, _ in plan):
        eval_factory(cell, seed, output_root=output_root, corpus_root=corpus_root, draws_root=draws_root,
                     device=args.device)
    missing = sorted({d for _, _, d in plan if not Path(draw_config_path(hz.SCENARIO_KEY, d, out_root=draws_root)).is_file()})
    if missing:
        raise FileNotFoundError(f"held-out draws {missing[:5]} are not materialised under {draws_root}")
    pending_episodes(output_root, plan)
    log_dir = output_root / PHASE_B_DIR / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    selection = ["--shard", str(args.shard), "--of", str(args.of), "--device", str(args.device),
                 "--corpus-root", str(corpus_root), "--draws-root", str(draws_root), "--output-root", str(output_root),
                 "--torch-threads", str(args.torch_threads)]
    if args.cells:
        selection += ["--cells", *args.cells]
    if args.draws:
        selection += ["--draws", *[str(d) for d in args.draws]]
    command = [sys.executable, "-P", "-m", "offline.hz4x4_phase_b", "evaluate-worker", *selection,
               "--watchdog-seconds", str(hz.WATCHDOG_SECONDS)]
    env = dict(os.environ, PYTHONPATH=str(hz._repo_root()), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONFAULTHANDLER="1")
    result = hz.run_with_restarts(command, log_dir=log_dir, label=f"eval-shard{args.shard}of{args.of}", env=env,
                                  cwd=hz.MAIN_TREE)
    last = result["attempts"][-1]
    print(f"evaluate {args.shard}/{args.of}: {result['status']} after {result['restarts']} restart(s); "
          f"last log {last['log']}", flush=True)
    if result["status"] == "completed":
        return 0
    return 3 if int(last["returncode"]) == 3 else 1


def _run_train_queue(args: argparse.Namespace) -> int:
    """``train-queue``: every pending cell through ``train`` in its own process, strictly one after another."""
    import os
    import subprocess
    import sys

    assert_campaign_steps(args.output_root, args.gradient_steps)
    corpus_root = hz.assert_main_tree_path(args.corpus_root, what="the corpus root")
    output_root = hz.assert_main_tree_path(args.output_root, what="the output root")
    cells = training_cells(tiers=args.tiers, arms=args.arms)
    logs = output_root / PHASE_B_DIR / "train_logs"
    env = dict(os.environ, PYTHONPATH=str(hz._repo_root()), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run_cell(tier: str, arm: str, seed: int) -> int:
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{hz.SCENARIO}_{tier}_{arm}_seed{seed}.log"
        command = [sys.executable, "-P", "-m", "offline.hz4x4_phase_b", "train", "--tier", tier, "--arm", arm,
                   "--seed", str(seed), "--device", str(args.device), "--gradient-steps", str(args.gradient_steps),
                   "--corpus-root", str(corpus_root), "--output-root", str(output_root), "--torch-threads", "1"]
        print(f"TRAIN {tier}/{arm} seed {seed} -> {log}", flush=True)
        with open(log, "w", encoding="utf-8") as handle:
            code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=hz.MAIN_TREE).returncode
        print(f"  exit {code}", flush=True)
        return int(code)

    result = run_train_queue(cells, output_root=output_root, run_cell=run_cell)
    print(f"train-queue: {result}", flush=True)
    return 0 if result["status"] == "completed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(None if argv is None else list(argv))
    if args.command == "train":
        return _run_train(args)
    if args.command == "train-queue":
        return _run_train_queue(args)
    if args.command == "evaluate":
        return _run_evaluate(args)
    if args.command == "evaluate-worker":
        return _run_evaluate_worker(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover - the CLI is exercised through main() by the tests
    raise SystemExit(main())
