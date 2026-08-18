"""P8.3: the D4RL calibration adapter, and the isolation the dependency ruling is conditioned on.

Every test here runs in the **project** venv.  That is the point of splitting
``calibration/d4rl_adapter.py`` (numpy + torch only) from ``calibration/d4rl_run.py`` (h5py,
gymnasium, mujoco): the load-bearing arithmetic of the adapter -- the codebook, the episode split,
the normalised score, the transition table -- is checked on every machine that can run the suite,
with no simulator and no MuJoCo, forever.  Nothing in this file imports the calibration-only
dependencies, and ``test_no_module_outside_calibration_imports_a_calibration_dependency`` is what
proves the rest of the repository does not either.

The adapter's declared choices live in ``docs/plans/p8.3.md`` section 3, A1-A11.  Where a test
pins one of them, it names it.
"""

from __future__ import annotations

import ast
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch
from setuptools import find_packages

from calibration.d4rl_adapter import (
    CODEBOOK_SIZES,
    PRIMARY_CODEBOOK_SIZE,
    PUBLISHED_SCORES,
    REF_MAX_SCORE,
    REF_MIN_SCORE,
    ActionCodebook,
    bc_windows,
    build_transition_table,
    episode_returns,
    episode_spans,
    normalization_stats,
    normalized_score,
    top_return_episodes,
)
from offline.offline_baselines import TOP_RETURN_FRACTION, iql_reward_scale

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = REPO_ROOT / "calibration"

#: Binding condition (ii) of the dependency ruling (``PROJECT_PLAN`` Decisions Log 2026-08-16).
#: ``gymnasium`` is deliberately absent: it is a declared, pinned project dependency
#: (``pyproject.toml``, ``gymnasium==1.3.0``) and therefore not part of the calibration-only set.
#: The legacy ``gym`` is.
CALIBRATION_ONLY_MODULES = frozenset({"d4rl", "mujoco", "mujoco_py", "gym", "h5py"})


# --------------------------------------------------------------------------------------
# fixtures: a tiny hand-built dataset whose every number is checkable by eye
# --------------------------------------------------------------------------------------
def _toy_dataset() -> dict[str, np.ndarray]:
    """Three episodes of 6, 4 and 8 rows, then two trailing rows with no flag.

    The trailing rows exist so the declared "drop a trailing partial episode" rule
    (``docs/plans/p8.3.md`` section 3, and ``episode_spans``' docstring) is exercised rather
    than assumed.
    """
    lengths = [6, 4, 8]
    total = sum(lengths) + 2
    timeouts = np.zeros(total, dtype=bool)
    cursor = 0
    for length in lengths:
        cursor += length
        timeouts[cursor - 1] = True
    rng = np.random.default_rng(11)
    return {
        "observations": rng.standard_normal((total, 3)).astype(np.float32),
        "next_observations": rng.standard_normal((total, 3)).astype(np.float32),
        "actions": rng.uniform(-1.0, 1.0, size=(total, 2)).astype(np.float32),
        "rewards": rng.standard_normal(total).astype(np.float32),
        "timeouts": timeouts,
        "terminals": np.zeros(total, dtype=bool),
    }


# --------------------------------------------------------------------------------------
# the normalised score and the published constants
# --------------------------------------------------------------------------------------
def test_normalized_score_matches_the_d4rl_definition() -> None:
    """``100 * (r - ref_min) / (ref_max - ref_min)``, anchored exactly at both ends.

    The two anchors are asserted with ``==``: they are the definition, not an approximation.
    The vector case is recomputed twice -- once from the literals written out here rather than
    imported, which catches a constant that drifted, and once through ``np.interp``, a genuinely
    independent implementation whose association order differs, hence the tolerance on that one
    and only that one.
    """
    assert float(normalized_score(REF_MIN_SCORE)) == 0.0
    assert float(normalized_score(REF_MAX_SCORE)) == 100.0

    returns = np.array([-500.0, 0.0, 1234.5, 6000.0, 12135.0, 20000.0], dtype=np.float64)
    from_literals = 100.0 * (returns - (-280.178953)) / (12135.0 - (-280.178953))
    assert np.array_equal(normalized_score(returns), from_literals)

    independent = np.interp(returns, [REF_MIN_SCORE, REF_MAX_SCORE], [0.0, 100.0])
    inside = (returns >= REF_MIN_SCORE) & (returns <= REF_MAX_SCORE)
    np.testing.assert_allclose(
        normalized_score(returns)[inside], independent[inside], rtol=0.0, atol=1e-12
    )


def test_the_reference_constants_and_comparators_are_the_published_ones() -> None:
    """Pins d4rl's constants and the IQL paper's row, so a silent edit is a failing test.

    ``REF_MIN_SCORE``/``REF_MAX_SCORE``: d4rl ``infos.py`` lines 128-132 and 218-222, propagated
    to ``-v2`` by the loop at line 287.  ``PUBLISHED_SCORES``: arXiv:2110.06169 Table 1, row
    ``halfcheetah-medium-expert-v2``, columns BC / 10%BC / IQL.
    """
    assert REF_MIN_SCORE == -280.178953
    assert REF_MAX_SCORE == 12135.0
    assert PUBLISHED_SCORES == {"bc": 55.2, "bc_top10": 92.9, "iql": 86.7}


def test_the_withdrawn_comparator_appears_nowhere_in_the_calibration_code() -> None:
    """``89.9`` was withdrawn (``BRIEF_24`` corrected at ``302ae87``) and may not reach a table.

    The coordinator's ruling is that a withdrawn figure must not sit beside the verified one,
    because that preserves the error and invites a reader to average them.  A grep is how that
    ruling survives the next edit.
    """
    for path in sorted(CALIBRATION_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "89.9" not in source, (
            f"{path.relative_to(REPO_ROOT)} carries the withdrawn comparator 89.9; the verified "
            "value is 86.7 (arXiv:2110.06169 Table 1)"
        )


# --------------------------------------------------------------------------------------
# the codebook -- the whole of the adapter (plan section 3, A1-A5)
# --------------------------------------------------------------------------------------
def test_codebook_encode_round_trips_on_its_own_centroids() -> None:
    """A centroid encodes to its own index.  Exact, for every code word."""
    rng = np.random.default_rng(3)
    actions = rng.uniform(-1.0, 1.0, size=(4000, 6))
    book = ActionCodebook.fit(actions, 16, subsample=4000)

    assert book.size == 16
    assert book.action_dim == 6
    assert np.array_equal(book.encode(book.centroids), np.arange(16, dtype=np.int64))


def test_codebook_encoding_is_the_nearest_centroid() -> None:
    """A1/A4 recomputed by an independent route, with exact equality on the index array.

    The implementation walks chunks of exact squared distances; this recomputes the full
    ``(n, size)`` matrix at once through ``np.linalg.norm``.  ``argmin`` of the norm and of the
    squared norm agree exactly because ``sqrt`` is monotone, so ``==`` is the right assertion
    and no tolerance is needed.
    """
    rng = np.random.default_rng(5)
    actions = rng.uniform(-1.0, 1.0, size=(2500, 6))
    book = ActionCodebook.fit(actions, 32, subsample=2500)

    probe = rng.uniform(-1.2, 1.2, size=(700, 6))
    distances = np.linalg.norm(probe[:, None, :] - book.centroids[None, :, :], axis=-1)
    expected = np.argmin(distances, axis=1).astype(np.int64)

    assert np.array_equal(book.encode(probe), expected)
    assert np.array_equal(book.encode(probe, chunk=17), expected)


def test_codebook_is_deterministic_under_its_seed() -> None:
    """A3: two fits of the same actions produce bit-identical centroids and one digest."""
    rng = np.random.default_rng(7)
    actions = rng.uniform(-1.0, 1.0, size=(3000, 6))

    first = ActionCodebook.fit(actions, 8, subsample=1500)
    second = ActionCodebook.fit(actions, 8, subsample=1500)

    assert np.array_equal(first.centroids, second.centroids)
    assert first.digest == second.digest
    assert len(first.digest) == 64


def test_codebook_decode_returns_clipped_float32_code_words() -> None:
    """A5: the decoded action is the centroid itself, in range, in the dtype the env takes."""
    book = ActionCodebook(centroids=np.array([[-2.0, 0.5], [0.25, 3.0]], dtype=np.float64))
    decoded = book.decode(np.array([1, 0, 1], dtype=np.int64))

    assert decoded.dtype == np.float32
    assert decoded.shape == (3, 2)
    assert np.array_equal(decoded, np.array([[0.25, 1.0], [-1.0, 0.5], [0.25, 1.0]], np.float32))


def test_the_declared_codebook_ladder_is_the_one_in_the_plan() -> None:
    """K in {8, 64, 256}, primary 64 -- fixed before the run so it cannot be picked on a result."""
    assert CODEBOOK_SIZES == (8, 64, 256)
    assert PRIMARY_CODEBOOK_SIZE == 64
    assert PRIMARY_CODEBOOK_SIZE in CODEBOOK_SIZES


# --------------------------------------------------------------------------------------
# episodes, returns and the top-decile filter
# --------------------------------------------------------------------------------------
def test_episode_boundaries_partition_the_flagged_rows_and_drop_the_trailing_remainder() -> None:
    """The split is a partition of everything up to the last flag, and nothing else.

    Checked as a partition rather than by comparing to a hard-coded list: row counts sum, no
    episode is empty, and every flagged row is the last row of exactly one episode.
    """
    data = _toy_dataset()
    spans = episode_spans(data["timeouts"], data["terminals"])

    assert len(spans) == 3
    assert np.array_equal(spans.lengths, np.array([6, 4, 8], dtype=np.int64))
    assert np.array_equal(spans.start, np.array([0, 6, 10], dtype=np.int64))
    assert np.array_equal(spans.stop, np.array([6, 10, 18], dtype=np.int64))

    covered = np.concatenate([np.arange(a, b) for a, b in zip(spans.start, spans.stop)])
    assert covered.size == int(spans.lengths.sum())
    assert np.array_equal(np.unique(covered), covered)
    assert np.array_equal(np.flatnonzero(data["timeouts"]), spans.stop - 1)
    assert int(data["timeouts"].size - spans.stop[-1]) == 2


def test_episode_returns_are_the_undiscounted_sums() -> None:
    """Recomputed by slicing the raw array, not by calling the function under test."""
    data = _toy_dataset()
    spans = episode_spans(data["timeouts"], data["terminals"])
    rewards = data["rewards"]

    expected = np.array(
        [float(np.sum(rewards[a:b].astype(np.float64))) for a, b in zip(spans.start, spans.stop)]
    )
    assert np.array_equal(episode_returns(rewards, spans), expected)


def test_reward_scale_is_the_published_locomotion_normalisation() -> None:
    """A7: ``iql_reward_scale`` unchanged, on episode returns, recomputed by hand."""
    data = _toy_dataset()
    spans = episode_spans(data["timeouts"], data["terminals"])
    returns = episode_returns(data["rewards"], spans)

    scale = iql_reward_scale(returns)
    assert scale == 1000.0 / (float(np.max(returns)) - float(np.min(returns)))


def test_top_decile_selection_takes_whole_episodes_and_the_declared_fraction() -> None:
    """A9: ``ceil(fraction * n)`` whole episodes, highest return first, ties by lower index."""
    returns = np.array([5.0, 1.0, 9.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0], dtype=np.float64)

    chosen = top_return_episodes(returns, TOP_RETURN_FRACTION)
    assert np.array_equal(chosen, np.array([2], dtype=np.int64))

    quarter = top_return_episodes(returns, 0.25)
    assert quarter.size == 3
    assert np.array_equal(quarter, np.array([2, 3, 7], dtype=np.int64))
    assert set(quarter.tolist()) <= set(np.argsort(-returns, kind="stable")[:3].tolist())


def test_top_decile_selection_never_returns_an_empty_set() -> None:
    """A tiny corpus still yields one episode rather than nothing to train on."""
    assert top_return_episodes(np.array([2.0, 1.0]), 0.10).size == 1


# --------------------------------------------------------------------------------------
# the transition table -- contract C6 lives here
# --------------------------------------------------------------------------------------
def test_transition_table_carries_the_dataset_next_observation_for_every_row() -> None:
    """C6's bootstrap target, including the LAST transition of every episode.

    Recomputed by direct indexing into the raw arrays and normalising with the same statistics,
    rather than by calling the builder a second time.  The final row of each episode is checked
    explicitly: that is the transition ``iql_targets`` bootstraps through, and dropping it would
    be the quiet version of adding a ``done`` term.
    """
    data = _toy_dataset()
    spans = episode_spans(data["timeouts"], data["terminals"])
    stats = normalization_stats(data["observations"])
    index = np.arange(int(spans.stop[-1]), dtype=np.int64) % 4

    table = build_transition_table(
        observations=data["observations"],
        next_observations=data["next_observations"],
        action_index=index,
        rewards=data["rewards"],
        spans=spans,
        stats=stats,
        reward_scale=2.0,
    )

    rows = int(spans.lengths.sum())
    assert len(table) == rows
    assert table.state.dtype is torch.float32
    assert table.action.dtype is torch.int64
    assert table.reward.dtype is torch.float32

    expected_next = stats.normalize_state(
        "d4rl", "agent", data["next_observations"][: spans.stop[-1]]
    )
    assert np.array_equal(table.next_state.numpy(), expected_next)

    expected_reward = data["rewards"][: spans.stop[-1]].astype(np.float32) * np.float32(2.0)
    assert np.array_equal(table.reward.numpy(), expected_reward)
    assert table.reward_scale == 2.0

    last_rows = spans.stop - 1
    assert np.array_equal(
        table.next_state.numpy()[last_rows],
        stats.normalize_state("d4rl", "agent", data["next_observations"][last_rows]),
    )


def test_transition_table_stream_index_and_step_follow_the_episode_split() -> None:
    """``stream_index`` is the episode, ``t`` the step inside it -- the corpus' own meaning."""
    data = _toy_dataset()
    spans = episode_spans(data["timeouts"], data["terminals"])
    stats = normalization_stats(data["observations"])
    rows = int(spans.lengths.sum())

    table = build_transition_table(
        observations=data["observations"],
        next_observations=data["next_observations"],
        action_index=np.zeros(rows, dtype=np.int64),
        rewards=data["rewards"],
        spans=spans,
        stats=stats,
        reward_scale=1.0,
    )

    expected_stream = np.repeat(np.arange(len(spans)), spans.lengths)
    expected_t = np.concatenate([np.arange(n) for n in spans.lengths])
    assert np.array_equal(table.stream_index.numpy(), expected_stream)
    assert np.array_equal(table.t.numpy(), expected_t)


# --------------------------------------------------------------------------------------
# BC windows (plan section 3, A10/A11)
# --------------------------------------------------------------------------------------
def test_bc_windows_never_cross_an_episode_boundary() -> None:
    """A10.  A window spanning two episodes would train BC on a discontinuity IQL never sees.

    The check does not trust the builder's own bookkeeping: it recovers each window's rows from
    the normalised observations by matching them back against the source array, then asserts every
    window lies inside one episode.
    """
    data = _toy_dataset()
    spans = episode_spans(data["timeouts"], data["terminals"])
    stats = normalization_stats(data["observations"])
    rows = int(spans.stop[-1])
    index = np.arange(rows, dtype=np.int64) % 4

    stacked = bc_windows(
        observations=data["observations"],
        action_index=index,
        spans=spans,
        stats=stats,
        n_actions=4,
        context_length=2,
    )

    assert set(stacked) == {"state", "action", "avail_mask"}
    assert stacked["state"].shape == (9, 2, 3)
    assert stacked["action"].shape == (9, 2)
    assert stacked["avail_mask"].shape == (9, 2, 4)
    assert bool(stacked["avail_mask"].all())

    normalised = stats.normalize_state("d4rl", "agent", data["observations"][:rows])
    episode_of = np.repeat(np.arange(len(spans)), spans.lengths)
    for window in stacked["state"].numpy():
        located = [int(np.flatnonzero((normalised == row).all(axis=1))[0]) for row in window]
        assert located == list(range(located[0], located[0] + 2))
        assert len({episode_of[r] for r in located}) == 1


def test_bc_windows_tile_each_episode_without_overlapping() -> None:
    """The other half of A10, added after a mutation survived without it.

    ``test_bc_windows_never_cross_an_episode_boundary`` pins where a window may not go; it does
    **not** pin that windows are disjoint.  A builder that slid by one row instead of by
    ``context_length`` produced the same window count, stayed inside every episode and passed --
    while oversampling early rows and changing what BC trains on.  This is the assertion that
    kills that mutation: every used row appears exactly once.
    """
    data = _toy_dataset()
    spans = episode_spans(data["timeouts"], data["terminals"])
    stats = normalization_stats(data["observations"])
    rows = int(spans.stop[-1])

    stacked = bc_windows(
        observations=data["observations"],
        action_index=np.arange(rows, dtype=np.int64) % 4,
        spans=spans,
        stats=stats,
        n_actions=4,
        context_length=2,
    )

    normalised = stats.normalize_state("d4rl", "agent", data["observations"][:rows])
    located = [
        int(np.flatnonzero((normalised == row).all(axis=1))[0])
        for window in stacked["state"].numpy()
        for row in window
    ]
    assert sorted(located) == located
    assert len(set(located)) == len(located)
    # lengths 6, 4, 8 at context 2 tile completely: every row is used exactly once.
    assert sorted(located) == list(range(rows))


def test_bc_windows_restricted_to_chosen_episodes_use_only_those_rows() -> None:
    """A9/A10: ``bc_top10`` differs from ``bc`` by its episode set and by nothing else."""
    data = _toy_dataset()
    spans = episode_spans(data["timeouts"], data["terminals"])
    stats = normalization_stats(data["observations"])
    rows = int(spans.stop[-1])

    stacked = bc_windows(
        observations=data["observations"],
        action_index=np.arange(rows, dtype=np.int64) % 4,
        spans=spans,
        stats=stats,
        n_actions=4,
        context_length=2,
        episodes=[1],
    )

    assert stacked["state"].shape == (2, 2, 3)
    normalised = stats.normalize_state("d4rl", "agent", data["observations"][:rows])
    for window in stacked["state"].numpy():
        for row in window:
            located = int(np.flatnonzero((normalised == row).all(axis=1))[0])
            assert 6 <= located < 10


def test_bc_windows_drop_only_the_remainder_of_an_episode() -> None:
    """The declared cost of non-overlapping windows, made visible instead of assumed."""
    data = _toy_dataset()
    spans = episode_spans(data["timeouts"], data["terminals"])
    stats = normalization_stats(data["observations"])
    rows = int(spans.stop[-1])

    stacked = bc_windows(
        observations=data["observations"],
        action_index=np.arange(rows, dtype=np.int64) % 4,
        spans=spans,
        stats=stats,
        n_actions=4,
        context_length=4,
    )

    # 6 // 4 = 1, 4 // 4 = 1, 8 // 4 = 2 -> four windows, two rows of episode 0 dropped.
    assert stacked["state"].shape[0] == 4


def test_bc_windows_reject_a_context_longer_than_every_episode() -> None:
    """Refuses rather than silently returning an empty training set."""
    data = _toy_dataset()
    spans = episode_spans(data["timeouts"], data["terminals"])
    stats = normalization_stats(data["observations"])
    rows = int(spans.stop[-1])

    with pytest.raises(ValueError, match="no window of length"):
        bc_windows(
            observations=data["observations"],
            action_index=np.arange(rows, dtype=np.int64) % 4,
            spans=spans,
            stats=stats,
            n_actions=4,
            context_length=99,
        )


# --------------------------------------------------------------------------------------
# Gate F: the dependency ruling's binding condition (ii), asserted mechanically
# --------------------------------------------------------------------------------------
def _imported_top_level_modules(source: str, where: str) -> set[str]:
    """Every top-level module name *source* imports, by AST rather than by grep.

    ``import a.b`` and ``from a.b import c`` both contribute ``a``.  A relative import contributes
    nothing: it cannot reach a third-party package.
    """
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source, filename=where)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module.split(".", 1)[0])
    return modules


def test_the_import_scanner_detects_a_planted_import() -> None:
    """Positive control.  Without it, Gate F passing would prove only that it never looks."""
    planted = "import numpy\nimport mujoco\nfrom d4rl.infos import REF_MIN_SCORE\n"
    found = _imported_top_level_modules(planted, "<planted>")

    assert found & CALIBRATION_ONLY_MODULES == {"mujoco", "d4rl"}
    assert _imported_top_level_modules("import numpy\n", "<clean>") & CALIBRATION_ONLY_MODULES == set()


def test_no_module_outside_calibration_imports_a_calibration_dependency() -> None:
    """Binding condition (ii): no module a paper number flows through may import these.

    Wider than the brief's ``offline/**`` -- it scans every tracked ``.py`` outside ``calibration/``
    and ``tests/`` -- because a repo-wide scan cost nothing when it was first run (zero hits,
    2026-08-18) and catches the module that has not been written yet.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    scanned = 0
    violations: list[str] = []
    for name in tracked:
        if name.startswith(("calibration/", "tests/", "CityFlow/")):
            continue
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        scanned += 1
        offending = _imported_top_level_modules(path.read_text(encoding="utf-8"), name)
        offending &= CALIBRATION_ONLY_MODULES
        if offending:
            violations.append(f"{name}: {sorted(offending)}")

    assert scanned > 50, f"the scan only reached {scanned} files; it is not covering the repository"
    assert violations == [], (
        "the calibration-only dependencies are authorised for this task alone and no module a "
        f"paper number flows through may import them: {violations}"
    )


def test_the_runner_is_the_only_calibration_file_that_imports_them() -> None:
    """The other half of the split: the adapter stays importable in the project venv."""
    adapter = _imported_top_level_modules(
        (CALIBRATION_DIR / "d4rl_adapter.py").read_text(encoding="utf-8"), "d4rl_adapter.py"
    )
    assert adapter & CALIBRATION_ONLY_MODULES == set()


def test_calibration_is_deliberately_not_a_distributed_package() -> None:
    """Pins ``docs/plans/p8.3.md`` section 4: no ``__init__.py``, invisible to the wheel.

    This code depends on packages outside the project's dependency set and no paper number flows
    through it, so shipping it would be the defect.  ``pyproject.toml`` is ``BRIEF_23``'s file and
    binding condition (iii) forbids touching it, which makes the correct answer and the compliant
    one the same answer.
    """
    assert CALIBRATION_DIR.is_dir()
    assert not (CALIBRATION_DIR / "__init__.py").exists()

    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    include = config["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "calibration" not in set(find_packages(where=str(REPO_ROOT), include=include))
