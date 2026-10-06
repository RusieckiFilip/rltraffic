"""Fixtures shared by P5.2b's tests (``BRIEF_44``): a synthetic random tier written by the REAL logger, and spies.

Not a test module -- the name carries no ``test_`` prefix -- so pytest collects nothing here.

WHY A SYNTHETIC TIER
--------------------
``offline.tier_sweep._run_train_baselines`` reads a whole tier through ``tier_parts``: the corpus manifest, the window
loader, the size match, the joint index.  The defect (``DEFERRED`` 106) only shows on a tier with MORE episodes available
than selected, so the fixture writes two episodes per draw and the size match keeps one per draw -- six available, three
selected -- exactly the random tier's shape at a size a test can afford.  Every episode is written through
:class:`offline.trajectory_logger.TrajectoryLogger` (``tests/p7_3c_fixtures.py``'s scripted env), never assembled by hand,
so the loader reads the on-disk format a collected corpus has.

Format version: C6 v1.1 (the logger's own).  Alignment convention (``docs/CONTRACTS.md`` C6): observation rows ``T + 1``,
decision and outcome rows ``T``; the reward of decision ``t`` is in the ``info`` returned by step ``t``.

The two episodes of one draw share their states (the scripted env's states are a function of the draw, the position and
the step) and differ in their rewards, which carry the episode's own counter -- so a stream's return identifies the
episode it came from, and the global top-decile filter and the per-intersection filter select different streams.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from offline.trajectory_logger import TrajectoryLogger
from tests.p7_3c_fixtures import ScriptedCorpusEnv, log_episode

#: The random tier's directory name, as ``offline.tier_sweep.TIERS["random"]`` declares it.
RANDOM_DIR = "cf_grid4x4__random"
#: Sixteen intersection ids, grid4x4's count (``tier_sweep.EXPECTED_NODES``); the names are the fixture's own.
IDS: tuple[str, ...] = tuple(f"{row}{col}" for row in "ABCD" for col in range(4))
#: Three training draws, two episodes each: six available, three selected by ``one_per_draw``.
DRAWS: tuple[int, ...] = (1, 2, 3)
EPISODES_PER_DRAW = 2
#: Decisions per episode: small, so a transition table has ``episodes x 16 x 4`` rows.
DECISIONS = 4


def write_synthetic_tier(corpus_root: Path, *, draws: Sequence[int] = DRAWS, episodes_per_draw: int = EPISODES_PER_DRAW,
                         decisions: int = DECISIONS) -> Path:
    """Write the tier ``corpus_root / RANDOM_DIR`` through the real logger and return its directory."""
    out_dir = Path(corpus_root) / RANDOM_DIR
    env = ScriptedCorpusEnv(IDS)
    metadata = {"scenario_id": "cityflow_grid4x4", "backend": "cityflow", "behavior_policy": "random",
                "episodes": len(draws) * int(episodes_per_draw), "base_seed": 1000}
    logger = TrajectoryLogger(env, out_dir, run_metadata=metadata)
    counter = 0
    for draw in draws:
        for _ in range(int(episodes_per_draw)):
            episode = counter

            def rewards_for(draw_id: int, ix: str, count: int, episode: int = episode) -> list[float]:
                # A draw's second episode is three times as costly, so the unselected streams widen the return span
                # and a reward scale taken over every stream differs from the selected streams' (mutation M-scale).
                position = IDS.index(ix)
                weight = 3.0 if episode % EPISODES_PER_DRAW else 1.0
                return [-weight * float(1 + (draw_id * 7 + position * 3 + episode * 11 + step) % 17)
                        for step in range(count)]

            log_episode(logger, env, draw=int(draw), decisions=int(decisions), rewards_for=rewards_for)
            logger.finalize_episode()
            counter += 1
    return out_dir


def synthetic_spec(*, subsample: str = "one_per_draw", episode_count: int | None = None) -> Any:
    """A ``TierSpec`` for the synthetic tier, named ``random`` so the code under test takes its own branch for it."""
    import offline.tier_sweep as ts

    if episode_count is None:
        episode_count = len(DRAWS) if subsample == "one_per_draw" else len(DRAWS) * EPISODES_PER_DRAW
    return ts.TierSpec(tier="random", dirs=(RANDOM_DIR,), subsample=subsample, episode_count=int(episode_count),
                       ladder_att=0.0, mean_training_return=0.0)


def install_synthetic_tier(monkeypatch: Any, *, subsample: str = "one_per_draw") -> Any:
    """Point ``tier_sweep``'s tier table at the synthetic spec and stub the graph, which no IQL path reads.

    ``tier_parts`` builds the adjacency from the tier's sim config; the synthetic corpus has none, and neither the
    baselines nor IQL consume the graph (``_run_train_baselines`` never reads ``parts["adjacency"]``).
    """
    import offline.tier_sweep as ts

    spec = synthetic_spec(subsample=subsample)
    monkeypatch.setattr(ts, "TIERS", {"random": spec})
    monkeypatch.setattr(ts, "adjacency_for_tier", lambda *args, **kwargs: None)
    return spec


@dataclass
class TrainSpy:
    """Stands in for ``offline_baselines.train_iql`` / ``train_bc``: records every call, writes the checkpoint file the
    caller will move, and returns a ``TrainRecord`` with known values."""

    method: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, batch: Any, **kwargs: Any) -> Any:
        from offline.offline_baselines import TrainRecord

        import torch

        self.calls.append({"batch": batch, **kwargs})
        path = Path(kwargs["checkpoint_path"])
        seed = int(kwargs["seed"])
        torch.save({"model": {"w": torch.full((2,), float(seed))}, "provenance": {"seed": seed}}, path)
        payload = path.read_bytes()
        return TrainRecord(
            method=self.method, seed=seed, gradient_steps=int(kwargs["declared_gradient_steps"]),
            declared_gradient_steps=int(kwargs["declared_gradient_steps"]), losses=(1.5, 1.0 + seed / 1000.0),
            window_means=(1.25,), plateaued=True, checkpoint_path=str(path), canonical_digest=f"canonical-{seed}",
            file_sha256=_sha256(payload), seconds=10.0 + seed / 100.0,
            diagnostics={"training_rows": len(batch) if hasattr(batch, "__len__") else -1,
                         "final_v_loss": 0.5, "final_q_loss": 0.25, "final_policy_loss": 1.0},
        )


def install_train_spies(monkeypatch: Any) -> tuple[TrainSpy, TrainSpy]:
    """Replace both trainers where every caller imports them from (``offline.offline_baselines``, at call time)."""
    import offline.offline_baselines as ob

    iql, bc = TrainSpy("iql"), TrainSpy("bc")
    monkeypatch.setattr(ob, "train_iql", iql)
    monkeypatch.setattr(ob, "train_bc", bc)
    return iql, bc


def _sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def manifest_episodes(tier_dir: Path) -> list[dict[str, Any]]:
    """The tier manifest's episode entries, read with ``json`` -- the test's own route, not the loader's."""
    return list(json.loads((Path(tier_dir) / "manifest.json").read_text(encoding="utf-8"))["episodes"])


def stream_keys_of_table(table: Any, keys: Sequence[tuple[str, str, str]]) -> set[tuple[str, str, str]]:
    """The ``(dataset_dir, episode_file, ix_id)`` of every stream that has a row in *table*."""
    present = np.unique(table.stream_index.numpy())
    return {tuple(keys[int(i)]) for i in present}


# ----------------------------------------------------------------------
# A synthetic RECORD: a miniature of everything offline.iql_correction reads, in each record's own format
# ----------------------------------------------------------------------

#: The synthetic protocol: two seeds x three held-out draws.  Tests monkeypatch ``dt_gate``'s two constants to it.
SEEDS: tuple[int, ...] = (101, 202)
HELD_OUT: tuple[int, ...] = (1000, 1001, 1002)
#: The miniature declaration: four episodes available on draws 1-2, one selected per draw, of 360 decisions.
CORPUS_EPISODES: tuple[tuple[str, int], ...] = (
    ("ep000000_seed1000_draw1.npz", 1), ("ep000001_seed1001_draw1.npz", 1),
    ("ep000002_seed1000_draw2.npz", 2), ("ep000003_seed1001_draw2.npz", 2),
)
SELECTED: tuple[str, ...] = ("ep000001_seed1001_draw1.npz", "ep000002_seed1000_draw2.npz")
EPISODE_LENGTH = 360
#: The random tier's levels, shaped like the committed record (iql first, then the two DTs, then BC and the filters).
RANDOM_LEVELS: dict[str, float] = {"dt_spatial": 254.6, "dt_nomix": 255.3, "bc": 289.4, "bc_top10": 350.3,
                                   "bc_top10_perix": 301.2, "iql": 191.0}
#: att_engine = att_ours - ENGINE_GAP - a per-episode term, so the two definitions differ everywhere.
ENGINE_GAP = 7.0
POWER: dict[str, Any] = {"supplies": {"items": [{"name": "AC1", "type": "Mains", "online": 1}]},
                         "windows": {"ac_overlay_name": "Best Performance"}}


@dataclass(frozen=True)
class SyntheticRecord:
    roots: Any
    pins: Any
    seeds: tuple[int, ...]
    draws: tuple[int, ...]
    #: att_ours (= P5.2's att_horizon) per (method, tier) per (seed, draw).
    ours: dict[tuple[str, str], dict[tuple[int, int], float]]
    #: att_engine per (method, tier) per (seed, draw).
    engine: dict[tuple[str, str], dict[tuple[int, int], float]]


def _level(method: str, tier: str) -> float:
    import offline.tier_sweep as ts

    if tier == "random":
        return RANDOM_LEVELS[method]
    return round(float(ts.PREDICTED_LEVELS[method][tier]) * 0.9 + 3.0, 1)


def _episode_value(level: float, method: str, tier: str, seed: int, draw: int) -> float:
    """A level plus a per-episode term exact in binary (a multiple of 1/64), different per arm, seed and draw."""
    term = ((sum(map(ord, method + tier)) + 7 * seed + 13 * draw) % 29 - 14) / 64.0
    return float(level + term)


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _eval_payload(method: str, tier: str, values: dict[tuple[int, int], float], *,
                  model_provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """An evaluation record in P5.2's format, its ``cell`` block built by P5.2's own ``dt_gate._cell``."""
    from offline.dt_gate import EpisodeResult, _cell

    arm = f"{method}@{tier}"
    produced = [EpisodeResult(arm=arm, seed=seed, draw_id=draw, att_horizon=value,
                              horizon_vehicle_count=float(seed % 7 + draw % 5), episode_reward=-float(draw))
                for (seed, draw), value in sorted(values.items())]
    payload: dict[str, Any] = {
        "format_version": "p5.2-tier-sweep/1.0", "arm": arm, "tier": tier, "method": method,
        "declared_gradient_steps": 40000, "engine_seed": 1000, "deterministic": False,
        "cell": _cell(produced),
        "episodes": [{"arm": e.arm, "seed": e.seed, "draw_id": e.draw_id, "att_horizon": e.att_horizon,
                      "horizon_vehicle_count": e.horizon_vehicle_count, "episode_reward": e.episode_reward}
                     for e in produced],
    }
    if model_provenance is not None:
        payload["replicate"] = False
        payload["model_provenance"] = model_provenance
    return payload


def _cell_row(method: str, tier: str, seed: int, draw: int, ours: float, engine: float, checkpoint: str) -> dict[str, Any]:
    """One episode in P8.4b's cell format (``att_rederivation.run_campaign``'s writer)."""
    created = 1300 + draw % 7
    never = (seed + draw) % 3
    entered = created - never
    return {
        "format_version": "p8.4b-rederivation/1.0", "scenario": "grid4x4", "tier": tier, "method": method,
        "arm": f"{method}@{tier}", "seed": seed, "draw_id": draw, "created": created, "entered": entered,
        "never_entered": never, "entered_fraction": entered / created, "completed_at_horizon": entered - 20,
        "running_at_horizon": 20, "waiting_at_horizon": 0, "att_ours": ours, "att_engine": engine,
        "att_difference": ours - engine, "horizon_vehicle_count": 20.0, "episode_reward": -4000.0,
        "seconds": 3.5, "seconds_rollout": 2.9, "committed_att_ours": ours, "reproduces_committed": True,
        "policy_source": {"kind": "checkpoint", "detail": "agent BC/IQL via offline_baselines._baseline_factory",
                          "checkpoint": checkpoint},
    }


def _write_cells(work_dir: Path, rows: list[dict[str, Any]]) -> str:
    """P8.4b's campaign layout: one file per cell by P8.4b's own name, the manifest by P8.4b's own builder, the marker."""
    from offline.att_rederivation import CellKey, campaign_manifest, cell_file_name

    cells = []
    for row in rows:
        cell = CellKey(scenario="grid4x4", arm=row["arm"], seed=row["seed"], draw_id=row["draw_id"])
        cells.append(cell)
        _write_json(work_dir / cell_file_name(cell), row)
    manifest = campaign_manifest(cells, engine_seed=1000)
    _write_json(work_dir / "campaign_manifest.json", manifest)
    (work_dir / "CAMPAIGN_COMPLETE").write_text(json.dumps({"declared_cells_sha256": manifest["declared_cells_sha256"]})
                                                + "\n", encoding="utf-8")
    return str(manifest["declared_cells_sha256"])


def _iql_checkpoint(path: Path, *, seed: int, rows: int, fill: float, stats: dict[str, Any],
                    gradient_steps: int = 40000, reward_scale: float = 0.7429420505200595,
                    training_streams: int = 32) -> None:
    """A tiny file in the IQL checkpoint's shape: the provenance fields the module reads, two weight tensors."""
    import torch

    from agent.OfflineBaselines import canonical_state_dict_digest

    model = {"policy.w": torch.full((2, 3), float(fill)), "q.w": torch.full((3,), float(fill) + 1.0)}
    payload = {
        "format_version": "iql-checkpoint/1.0", "config": {"state_dim": 40, "n_actions": 8}, "model": model,
        "canonical_digest": canonical_state_dict_digest(model), "normalise": True, "scenario_id": "cityflow_grid4x4",
        "stats": stats, "intersection_ids": [],
        "provenance": {
            "tier": "random", "dataset_dirs": ["synthetic"], "scenario_id": "cityflow_grid4x4",
            "training_streams": training_streams,
            "independent_per_intersection": True, "method": "iql", "seed": seed, "gradient_steps": gradient_steps,
            "declared_gradient_steps": gradient_steps, "batch_size": 1280, "learning_rate": 0.0001,
            "weight_decay": 0.0001, "grad_clip": 0.25, "device": "cuda",
            "runtime": {"torch_num_threads": 1, "git_commit": "9460800" + "0" * 33, "torch_version": "2.11.0+cu128",
                        "cuda_device_name": "synthetic GPU"},
            "diagnostics": {"training_rows": rows, "reward_scale": reward_scale, "tau": 0.7, "beta": 3.0,
                            "gamma": 0.99, "polyak": 0.005, "weight_clip": 100.0},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


SYNTHETIC_STATS: dict[str, Any] = {"stats_version": "1.0", "split": "train", "draw_ids": [1, 2]}


def _tier_declaration(tier_corpus: Path) -> dict[str, Any]:
    """The declaration a synthetic TIER implies, read through ``tier_sweep``'s own loader (the caller has installed the
    synthetic spec): its selection, node order, stream count, reward scale, statistics and whole-tier rows."""
    import offline.tier_sweep as ts
    from offline.offline_baselines import iql_reward_scale

    parts = ts.tier_parts("random", tier_corpus)
    episodes = manifest_episodes(Path(tier_corpus) / RANDOM_DIR)
    return {
        "selected": [{"dataset_dir": e.dataset_dir, "episode_file": e.episode_file, "episode_index": e.episode_index,
                      "flow_draw": e.flow_draw} for e in parts["episodes"]],
        "node_order": list(parts["node_ids"]),
        "episodes_available": len(episodes),
        "training_streams": len(parts["streams"]),
        "reward_scale": iql_reward_scale([s.total_return for s in parts["streams"]]),
        "stats": parts["dataset"].stats.to_json_obj(),
        "whole_rows": sum(int(e["episode_length"]) for e in episodes) * len(parts["node_ids"]),
    }


def write_synthetic_record(root: Path, *, seeds: Sequence[int] = SEEDS, draws: Sequence[int] = HELD_OUT,
                           tier_corpus: Path | None = None) -> SyntheticRecord:
    """Everything the module reads, in miniature: P5.2's 19 Q1 cells and its manifest, the original IQL checkpoints,
    P8.4b's campaign for the same cells, the random tier's declaration and corpus manifest, and the README note.

    With *tier_corpus* (a synthetic tier written by :func:`write_synthetic_tier`, its spec installed), the declaration,
    the corpus and the original checkpoints' scale, statistics and row count are that tier's, so the module's training
    path runs on it end to end."""
    import offline.iql_correction as ic
    import offline.tier_sweep as ts

    root = Path(root)
    output, repo, draws_root = root / "output", root / "repo", root / "draws"
    corpus = Path(tier_corpus) if tier_corpus is not None else root / "corpus"
    for directory in (output, repo, corpus, draws_root):
        directory.mkdir(parents=True, exist_ok=True)
    implied = _tier_declaration(corpus) if tier_corpus is not None else None
    ours: dict[tuple[str, str], dict[tuple[int, int], float]] = {}
    engine: dict[tuple[str, str], dict[tuple[int, int], float]] = {}
    for method, tier in ts.OUT_OF_SAMPLE_CELLS:
        level = _level(method, tier)
        ours[(method, tier)] = {(s, d): _episode_value(level, method, tier, s, d) for s in seeds for d in draws}
        engine[(method, tier)] = {key: value - ENGINE_GAP - (key[1] % 3) / 8.0
                                  for key, value in ours[(method, tier)].items()}
        _write_json(output / "p5_2" / f"eval_{tier}_{method}.json", _eval_payload(method, tier, ours[(method, tier)]))
    whole_rows = len(CORPUS_EPISODES) * len(IDS) * EPISODE_LENGTH if implied is None else implied["whole_rows"]
    for index, seed in enumerate(seeds):
        _iql_checkpoint(output / "p5_2" / "checkpoints" / f"grid4x4_random_iql_seed{seed}.pt", seed=seed,
                        rows=whole_rows, fill=0.5 + index,
                        stats=SYNTHETIC_STATS if implied is None else implied["stats"],
                        reward_scale=0.7429420505200595 if implied is None else implied["reward_scale"],
                        training_streams=32 if implied is None else implied["training_streams"])
    lines = [f"{_sha256_file(path)}  {path.relative_to(output).as_posix()}"
             for path in sorted((output / "p5_2").rglob("*")) if path.is_file()]
    (output / "SHA256SUMS_p5_2.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    for method, tier in ts.OUT_OF_SAMPLE_CELLS:
        for (seed, draw), value in sorted(ours[(method, tier)].items()):
            rows.append(_cell_row(method, tier, seed, draw, value, engine[(method, tier)][(seed, draw)],
                                  str(output / "p5_2" / "checkpoints" / f"grid4x4_{tier}_{method}_seed{seed}.pt")))
    declared_cells = _write_cells(output / "p8_4b_rederivation", rows)

    tier_dir = corpus / RANDOM_TIER_DIR
    declaration = repo / "docs" / "data" / "p5_2_declaration_random.json"
    if implied is None:
        _write_json(tier_dir / "manifest.json", {
            "format_version": "1.1",
            "episodes": [{"filename": name, "episode_length": EPISODE_LENGTH, "flow_draw": draw}
                         for name, draw in CORPUS_EPISODES],
            "run_metadata": {"scenario_id": "cityflow_grid4x4"},
        })
        _write_json(declaration, {
            "format_version": "p5.2-declaration/1.0", "tier": "random", "episodes_available": len(CORPUS_EPISODES),
            "episodes_selected": len(SELECTED), "training_streams": len(SELECTED) * len(IDS), "node_order": list(IDS),
            "selected_episodes": [{"dataset_dir": str(tier_dir), "episode_file": name, "episode_index": index,
                                   "flow_draw": dict(CORPUS_EPISODES)[name]}
                                  for index, name in enumerate(SELECTED)],
        })
    else:
        _write_json(declaration, {
            "format_version": "p5.2-declaration/1.0", "tier": "random",
            "episodes_available": implied["episodes_available"], "episodes_selected": len(implied["selected"]),
            "training_streams": implied["training_streams"], "node_order": implied["node_order"],
            "selected_episodes": implied["selected"],
        })
    import statistics

    note = repo / "docs" / "notes" / "readme_2026-10-05" / "c1_rule_r.json"
    blocks = {}
    for definition, source in (("att_engine", engine), ("att_ours", ours)):
        means = {m: statistics.mean(source[(m, "random")].values()) for m in ts.METHODS}
        blocks[definition] = {"means": means, "order": sorted(means, key=means.get)}
    _write_json(note, {"cells_read": len(rows), "not_reproducing": 0, "tiers": {"grid4x4/random": blocks}})
    for draw in draws:
        config = draws_root / "cityflow_grid4x4" / f"draw_{draw:04d}" / "cityflow.json"
        _write_json(config, {"dir": "synthetic", "flowFile": "flow.json"})
    pins = ic.Pins(p5_2_sums_sha256=_sha256_file(output / "SHA256SUMS_p5_2.txt"),
                   declaration_sha256=_sha256_file(declaration), c1_note_sha256=_sha256_file(note),
                   p8_4b_declared_cells_sha256=declared_cells)
    roots = ic.Roots(repo_root=repo, output_root=output, corpus_root=corpus, draws_root=draws_root,
                     out_root=output / "p5_2b")
    return SyntheticRecord(roots=roots, pins=pins, seeds=tuple(seeds), draws=tuple(draws), ours=ours, engine=engine)


RANDOM_TIER_DIR = RANDOM_DIR


def install_synthetic_protocol(monkeypatch: Any, record: SyntheticRecord) -> None:
    """Point ``dt_gate``'s held-out pool and training seeds at the synthetic record's, where every caller reads them."""
    import offline.dt_gate as dt_gate

    monkeypatch.setattr(dt_gate, "HELD_OUT_DRAWS", tuple(record.draws))
    monkeypatch.setattr(dt_gate, "TRAINING_SEEDS", tuple(record.seeds))


def write_synthetic_run(record: SyntheticRecord, *, iql_shift: float = 60.0, stages: Sequence[str] = (
        "training", "canaries", "p5_2_eval", "rederivation")) -> dict[str, dict[tuple[int, int], float]]:
    """The corrected run's outputs under the synthetic out-root, in each stage's own format, as the run would leave
    them; returns the corrected cell's ``att_ours`` and ``att_engine`` per (seed, draw)."""
    import offline.tier_sweep as ts
    from offline import compute_latency as cl

    out = Path(record.roots.out_root)
    checkpoints = out / "p5_2" / "checkpoints"
    declared_rows = len(SELECTED) * len(IDS) * EPISODE_LENGTH
    corrected_ours = {key: value + iql_shift for key, value in record.ours[("iql", "random")].items()}
    corrected_engine = {key: value - ENGINE_GAP - (key[1] % 3) / 8.0 for key, value in corrected_ours.items()}
    runs = []
    if "training" in stages:
        for index, seed in enumerate(record.seeds):
            path = checkpoints / f"grid4x4_random_iql_seed{seed}.pt"
            _iql_checkpoint(path, seed=seed, rows=declared_rows, fill=10.0 + index, stats=SYNTHETIC_STATS)
            import torch

            runs.append({"tier": "random", "method": "iql", "seed": seed, "gradient_steps": 40000,
                         "declared_gradient_steps": 40000, "seconds": 300.0 + seed / 100.0,
                         "final_loss": 1.0, "training_rows": declared_rows, "checkpoint_path": str(path),
                         "canonical_digest": torch.load(path, map_location="cpu", weights_only=False)["canonical_digest"],
                         "state_dict_sha256": ts.canonical_state_dict_digest(path),
                         "file_sha256": _sha256_file(path), "plateaued": True})
        _write_json(out / "training_random_iql.json", {
            "format_version": "p5.2b-training/1.0", "tier": "random", "method": "iql",
            "seeds": list(record.seeds), "declared_gradient_steps": 40000, "batch_size": 1280,
            "regime": {"device": "cuda", "cuda_device_name": "synthetic GPU", "torch_version": "2.11.0+cu128",
                       "torch_num_threads": 1, "deterministic_algorithms": False, "cublas_workspace_config": None,
                       "omp_num_threads": "1", "mkl_num_threads": "1"},
            "concurrency": {"value": 1, "basis": "synthetic"},
            "table": {"rows": declared_rows, "streams": len(SELECTED) * len(IDS), "episodes": len(SELECTED)},
            "runs": runs,
        })
    elif (out / "training_random_iql.json").is_file():
        # The evaluation stages of a later call read the runs the training stage of an earlier one recorded.
        runs = json.loads((out / "training_random_iql.json").read_text(encoding="utf-8"))["runs"]
    if "canaries" in stages:
        git = {"commit": "0" * 40, "dirty": False}
        _write_json(out / "canary_open.json", cl.build_canary_record("open", 0.7, reproduced=True, git=git, power=POWER))
        _write_json(out / "canary_close.json", cl.build_canary_record("close", 0.8, reproduced=True, git=git,
                                                                      power=POWER))
    if "p5_2_eval" in stages:
        provenance = {str(run["seed"]): {"checkpoint_path": run["checkpoint_path"],
                                         "state_dict_sha256": run["state_dict_sha256"]} for run in runs}
        _write_json(out / "eval_random_iql.json",
                    _eval_payload("iql", "random", corrected_ours, model_provenance=provenance))
    if "rederivation" in stages:
        rows = [_cell_row("iql", "random", seed, draw, value, corrected_engine[(seed, draw)],
                          str(checkpoints / f"grid4x4_random_iql_seed{seed}.pt"))
                for (seed, draw), value in sorted(corrected_ours.items())]
        _write_cells(out / "rederivation", rows)
    return {"att_ours": corrected_ours, "att_engine": corrected_engine}
