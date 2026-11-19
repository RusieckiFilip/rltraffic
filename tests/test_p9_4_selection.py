"""P9.4 plan 2.1 (``docs/plans/p9.4_mix50_fresh.md``; Amendment A1 for check (v)): the selection and its five checks.

Synthetic pools built to the corpus's real shapes: ``mappo1000``'s 200 training streams over five directories of 40,
the registered random tier's one-per-draw subsample of two engine seeds INTERLEAVED draw by draw (its manifest order:
``ep000000_seed1000_draw1``, ``ep000001_seed1001_draw1``, ...), and ``random_fresh``'s 200 episodes, one per draw.  The
selection is ``method_tier_grid.mixture_training_streams`` with mix50's spec, and each check refuses on an input made
to fail it.  The real corpus is read only by the tests at the end, which skip without it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import offline.method_tier_grid as grid
import offline.p9_4_mix50_fresh as p94
from offline.offline_baselines import StreamReturn

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = "/corpus"


def _stream(directory: str, episode_file: str, draw: int, total_return: float) -> StreamReturn:
    return StreamReturn(dataset_dir=f"{CORPUS}/{directory}", episode_file=episode_file, ix_id="intersection_1_1",
                        ix_index=0, episode_index=0, flow_draw=int(draw), group=(25, 8), total_return=float(total_return))


def expert_pool() -> list[StreamReturn]:
    pool = []
    for k, seed in enumerate((101, 202, 303, 404, 505)):
        for j in range(40):
            draw = 1 + (k * 40 + j * 7) % 200
            pool.append(_stream(f"cf_hz1x1__mappo1000__seed{seed}", f"ep{j:06d}_seed1000_draw{draw}.npz", draw,
                                -6000.0 - 10.0 * (k * 40 + j)))
    pool[17] = StreamReturn(**{**pool[17].__dict__, "total_return": -5959.0})
    return pool


def old_random_pool(picks: dict[int, int] | None = None) -> list[StreamReturn]:
    """One stream per draw 1-200 from the interleaved two-seed tier; *picks* chooses seed 1001 on some draws."""
    pool = []
    for draw in range(1, 201):
        seed = 1001 if (picks or {}).get(draw, draw % 3 == 0) else 1000
        index = 2 * (draw - 1) + (seed - 1000)
        pool.append(_stream("cf_hz1x1__random", f"ep{index:06d}_seed{seed}_draw{draw}.npz", draw, -38000.0 - draw))
    return pool


def fresh_random_pool() -> list[StreamReturn]:
    return [_stream("cf_hz1x1__random_fresh", f"ep{draw - 1:06d}_seed1000_draw{draw}.npz", draw, -36000.0 - draw)
            for draw in range(1, 201)]


# ----------------------------------------------------------------------
# The selection IS mix50's registered rule, with the random pool replaced
# ----------------------------------------------------------------------


def test_the_selection_on_the_registered_pool_is_the_registered_mix50_selection() -> None:
    expert, random_pool = expert_pool(), old_random_pool()
    registered = grid.training_streams(grid.TIERS["mix50"], None,  # type: ignore[arg-type]
                                       component_streams={"mappo1000": expert, "random": random_pool})
    assert p94.select_streams(expert, random_pool) == registered


def test_the_expert_half_is_identical_and_the_random_half_lies_on_the_same_draws() -> None:
    expert = expert_pool()
    old = p94.select_streams(expert, old_random_pool())
    new = p94.select_streams(expert, fresh_random_pool())
    old_expert, old_random = p94.split_halves(old)
    new_expert, new_random = p94.split_halves(new)
    assert [s.key for s in new_expert] == [s.key for s in old_expert]
    assert len(new_expert) == len(new_random) == 100
    assert sorted(s.flow_draw for s in new_random) == sorted(s.flow_draw for s in old_random)
    assert len({s.flow_draw for s in new_random}) == 100
    assert {Path(s.dataset_dir).name for s in new_random} == {"cf_hz1x1__random_fresh"}


def test_a_random_pool_out_of_draw_order_lands_on_other_draws_and_check_iii_refuses() -> None:
    """The draw matching rests on both pools sorting in draw order; a pool that does not breaks it, and (iii) says so."""
    expert = expert_pool()
    old_draws = [s.flow_draw for s in p94.split_halves(p94.select_streams(expert, old_random_pool()))[1]]
    shuffled = [_stream("cf_hz1x1__random_fresh", f"ep{(draw * 37) % 200:06d}_seed1000_draw{draw}.npz", draw, -36000.0)
                for draw in range(1, 201)]
    new_draws = [s.flow_draw for s in p94.split_halves(p94.select_streams(expert, shuffled))[1]]
    check = p94.check_iii(new_draws, old_draws)
    assert check["passed"] is False
    others = {name: {"passed": True} for name in ("i", "ii", "iv", "v")}
    with pytest.raises(ValueError, match=r"check\(s\) \(iii\) failed"):
        p94.assert_checks({**others, "iii": check})


# ----------------------------------------------------------------------
# The five checks, each refusing an input made to fail it
# ----------------------------------------------------------------------


def test_check_i_counts_the_halves_and_refuses_a_short_or_unbalanced_set() -> None:
    selected = p94.select_streams(expert_pool(), fresh_random_pool())
    assert p94.check_i(selected) == {"n_streams": 200, "n_distinct": 200, "n_expert": 100, "n_random": 100,
                                     "passed": True}
    assert p94.check_i(selected[:-1])["passed"] is False
    assert p94.check_i([*selected[:100], *selected[:100]])["passed"] is False
    duplicated = p94.check_i([*selected[:199], selected[100]])
    assert (duplicated["n_expert"], duplicated["n_random"], duplicated["n_distinct"]) == (100, 100, 199)
    assert duplicated["passed"] is False


def test_check_ii_compares_the_expert_half_with_mix50_by_episode_sha256() -> None:
    digests = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(100)]
    assert p94.check_ii(list(reversed(digests)), digests)["passed"] is True
    assert p94.check_ii(digests[:99] + ["f" * 64], digests)["passed"] is False
    assert p94.check_ii(digests[:99], digests[:99])["passed"] is False


def test_check_iii_requires_the_same_100_draws_one_per_draw() -> None:
    draws = list(range(2, 202, 2))
    assert p94.check_iii(list(reversed(draws)), draws)["passed"] is True
    assert p94.check_iii([*draws[:99], draws[0]], draws)["passed"] is False
    assert p94.check_iii([*draws[:99], 3], draws)["passed"] is False
    assert p94.check_iii([*draws, draws[0]], draws)["passed"] is False
    assert p94.check_iii(draws[:99], draws[:99])["passed"] is False


def test_check_iv_requires_100_distinct_sequences_on_every_intersection() -> None:
    distinct = {"intersection_1_1": [f"{i:064x}" for i in range(100)]}
    assert p94.check_iv(distinct)["passed"] is True
    repeated = {"intersection_1_1": [*distinct["intersection_1_1"][:99], distinct["intersection_1_1"][0]]}
    assert p94.check_iv(repeated)["passed"] is False
    assert p94.check_iv({"intersection_1_1": distinct["intersection_1_1"][:99]})["passed"] is False


def test_check_v_compares_the_manifest_digest_with_the_recomputed_one_and_records_the_file_hash() -> None:
    good = [{"episode_sha256": "a" * 64, "trajectory_digest": "a" * 64, "file_sha256": "b" * 64} for _ in range(3)]
    assert p94.check_v(good)["passed"] is True
    assert p94.check_v([*good, {"episode_sha256": "a" * 64, "trajectory_digest": "c" * 64,
                                "file_sha256": "b" * 64}])["passed"] is False
    assert p94.check_v([*good, {"episode_sha256": "a" * 64, "trajectory_digest": "a" * 64,
                                "file_sha256": None}])["passed"] is False


def test_assert_checks_names_every_failed_check_and_passes_only_when_all_pass() -> None:
    p94.assert_checks({name: {"passed": True} for name in ("i", "ii", "iii", "iv", "v")})
    with pytest.raises(ValueError, match=r"\(ii\).*\(v\)"):
        p94.assert_checks({"i": {"passed": True}, "ii": {"passed": False}, "iii": {"passed": True},
                           "iv": {"passed": True}, "v": {"passed": False}})
    with pytest.raises(ValueError, match="all five"):
        p94.assert_checks({"i": {"passed": True}})


# ----------------------------------------------------------------------
# The trajectory digest (Amendment A1): the logger's own function, recomputed from a file
# ----------------------------------------------------------------------


def _write_episode(path: Path, actions: list[np.ndarray], reward: np.ndarray) -> None:
    arrays: dict[str, Any] = {"ix_ids": np.asarray([f"ix{i}" for i in range(len(actions))])}
    for i, column in enumerate(actions):
        arrays[f"ix{i}_action"] = column
    arrays["global_reward"] = reward
    np.savez(path, **arrays)


def test_the_trajectory_digest_is_the_loggers_by_an_independent_route(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    actions = [rng.integers(0, 8, size=360).astype(np.int64), rng.integers(0, 4, size=360).astype(np.int64)]
    reward = rng.normal(size=360).astype(np.float32)
    path = tmp_path / "ep.npz"
    _write_episode(path, actions, reward)
    # Independent route: explicit little-endian struct-free byte assembly.
    expected = hashlib.sha256(
        b"".join(column.astype(np.dtype("<i8")).tobytes() for column in actions)
        + reward.astype(np.dtype("<f4")).tobytes()
    ).hexdigest()
    assert p94.trajectory_digest(path) == expected
    actions[1][5] += 1
    _write_episode(path, actions, reward)
    assert p94.trajectory_digest(path) != expected


def test_the_file_hash_is_not_the_trajectory_digest(tmp_path: Path) -> None:
    path = tmp_path / "ep.npz"
    _write_episode(path, [np.arange(10, dtype=np.int64)], np.zeros(10, dtype=np.float32))
    assert p94.file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert p94.file_sha256(path) != p94.trajectory_digest(path)


# ----------------------------------------------------------------------
# The tier, declared beside the registered ones
# ----------------------------------------------------------------------


def test_the_tier_is_mix50_with_its_random_component_replaced_and_is_in_no_registered_table() -> None:
    spec = p94.tier_spec(rtg_scale=41000.0)
    mix50 = grid.TIERS["mix50"]
    assert spec.tier == "mix50_fresh"
    assert spec.dirs == tuple(d if d != "cf_hz1x1__random" else "cf_hz1x1__random_fresh" for d in mix50.dirs)
    assert spec.components == ("mappo1000", "random_fresh")
    assert (spec.target_rtg, spec.rtg_scale, spec.stream_count, spec.subsample, spec.phase) == (
        -5959.0, 41000.0, 200, "mixture", 2)
    assert "mix50_fresh" not in grid.TIERS and "mix50_fresh" not in grid.A27_TIERS
    with pytest.raises(ValueError, match="unknown tier"):
        grid.tier_spec("mix50_fresh")


# ----------------------------------------------------------------------
# The real corpus (skips without it)
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_root() -> Path:
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO_ROOT / "datasets_v11"
    if not (candidate / "cf_hz1x1__random_fresh" / "manifest.json").is_file():
        pytest.skip(f"format v1.1 corpus with random_fresh not found at {candidate}: set RLTRAFFIC_CORPUS_V11")
    return candidate


@pytest.mark.parametrize("directory", ["cf_hz1x1__random_fresh", "cf_hz1x1__random", "cf_hz1x1__mappo1000__seed101"])
def test_on_the_corpus_the_manifest_digest_is_the_trajectory_digest_and_not_the_file_hash(
    corpus_root: Path, directory: str
) -> None:
    manifest = json.loads((corpus_root / directory / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["episodes"][:5]:
        path = corpus_root / directory / str(entry["filename"])
        assert p94.trajectory_digest(path) == entry["episode_sha256"]
        assert p94.file_sha256(path) != entry["episode_sha256"]


def test_on_the_corpus_the_random_pools_sort_in_draw_order(corpus_root: Path) -> None:
    """The two pools the draw matching rests on, from the manifests alone (no dataset is built here)."""
    for directory in ("cf_hz1x1__random", "cf_hz1x1__random_fresh"):
        manifest = json.loads((corpus_root / directory / "manifest.json").read_text(encoding="utf-8"))
        names = sorted(str(e["filename"]) for e in manifest["episodes"])
        draws = [p94.draw_of(name) for name in names]
        assert draws == sorted(draws)
        assert {p94.draw_of(str(e["filename"])) == int(e["flow_draw"]) for e in manifest["episodes"]} == {True}
