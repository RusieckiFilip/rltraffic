"""P7.3c C3 (``BRIEF_41``): the warm-start fine-tune of the joint model -- A18(d) as A24(b) reads it.

Written BEFORE ``offline/few_shot.py``'s bodies, against signature-only skeletons, so every test reaches the real API.

What is pinned here, and by which route:

* **T-warm** -- the model the loop trains starts from the SOURCE's weights, tensor by tensor under ``torch.equal``,
  and its ``max_ep_len`` is the payload's 360 on a corpus whose episodes are 300 decisions long.  Captured through
  the loop's own ``build_model`` seam (synthetic source) and on the REAL subject (checkpoint-gated), both compared
  with the checkpoint read by the test's own ``torch.load``.
* **T-frozen** -- every state the model sees is ``(raw - mean) / std`` in float32 from the ``.npz`` with the
  CHECKPOINT's arrays (a zero std left as 1.0, the loader's rule), per intersection BY ID, bitwise; every RTG is the
  reversed ``np.cumsum`` of the stored rewards divided by the CHECKPOINT's ``rtg_scale[ix]``.  Captured through the
  loop's ``joint_batch`` seam; the provenance of each row is ``item_meta``, never index arithmetic.
* **T-prefix** -- k = 5 / 20 / 100 read exactly 201-205 / 201-220 / 201-300 from the manifest; a missing draw is
  refused BY NAME; 206 stays out of k = 5.
* **T-recipe** -- exactly B optimizer steps counted by a GLOBAL optimizer hook (not by the code's own counter); the
  optimizer state empty before step 0; warm-up and learning rate at steps 0, w - 1 and B - 1 for B = 1,000, 4,000 and
  16,000; the source's recorded recipe asserted equal to the imported constants.
* **T-order** -- a corpus logged in REVERSED id order trains in the checkpoint's order; one id absent is refused.
* **T-payload** -- the frozen parts equal the source's; ``target_rtg`` is the calibration artifact's ``k{k}`` target
  for every id, read by the test from the JSON; the provenance fields; and the payload LOADS through
  ``spatial_agent_with_targets`` (Amendment A1: the top-level version stays ``spatial-dt-checkpoint/1.0``).
* **T-cpu-determinism** -- the same seed twice on CPU gives identical weights AND byte-identical files (no wall
  clock in the payload); another seed does not give identical weights.
* **T-scratch** -- the from-scratch arm differs from the fine-tune ONLY in its initialisation: step-0 weights not the
  source's and equal to a fresh model under the same seed; rows, batches and recipe identical; the payloads differ
  only in ``model`` and ``provenance.few_shot.init``.
* The barrier: every refusal leaves the tree as it was, and a run that fails mid-training writes nothing.

No test here starts a process: the CLI is called in-process (``BRIEF_41`` Amendment B, B5).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch

from agent.SpatialDTAgent import SpatialDecisionTransformer, SpatialDTConfig
from agent.utils.utils import Utils
from offline import few_shot
from tests.p7_3c_fewshot_fixtures import (
    CORPUS_DRAWS,
    ZERO_STD_COLUMNS,
    FakeSpatialEnv,
    load_payload,
    sha256_file,
    source_provenance,
    synthetic_stats,
    write_source,
    write_training_corpus,
)
from tests.p7_3c_fixtures import GRID, calibration, calibration_ids, grid_run_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"
IDS = calibration_ids()
SEEDING = "Utils.seed_everything(seed, seed_python_random=False)"


# ----------------------------------------------------------------------------------------------
# Shared inputs: one synthetic source, one standard corpus (read-only in every test that uses them)
# ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    return write_source(tmp_path_factory.mktemp("source") / "source.pt")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_training_corpus(tmp_path_factory.mktemp("corpus") / "grid4x4_sumo_maxpressure")


def _request(source: tuple[Path, str], corpus: Path, destination: Path, **overrides: Any) -> dict[str, Any]:
    path, digest = source
    request: dict[str, Any] = {
        "source_path": path,
        "source_sha256": digest,
        "corpus_dir": corpus,
        "k": 100,
        "budget": 3,
        "seed": 101,
        "init": "source",
        "device": "cpu",
        "destination": destination,
    }
    request.update(overrides)
    return request


def _run(tmp_path: Path, source: tuple[Path, str], corpus: Path, name: str = "out.pt", **overrides: Any) -> Any:
    return few_shot.fine_tune(**_request(source, corpus, tmp_path / name, **overrides))


def _tree(root: Path) -> dict[str, bytes | None]:
    """Every path under *root* with its bytes (``None`` for a directory): the whole state a refusal must keep."""
    return {
        str(path.relative_to(root)): (None if path.is_dir() else path.read_bytes())
        for path in sorted(root.rglob("*"))
    }


class _Recorder:
    """Wraps the loop's three seams and keeps what each RETURNED, in call order.

    The loop reaches ``build_model``, ``draw_rows`` and ``joint_batch`` through the module's globals, so a wrapper
    installed here sees exactly what the loop used.  A loop that bypassed a seam would leave its list empty, and
    every test below asserts the list's length first.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.models: list[dict[str, torch.Tensor]] = []
        self.rows: list[torch.Tensor] = []
        self.batches: list[dict[str, torch.Tensor]] = []
        self.stacked: list[Any] = []
        real_build, real_rows, real_batch = few_shot.build_model, few_shot.draw_rows, few_shot.joint_batch

        def build(config: Any, init: str, source_model: Any, device: Any) -> Any:
            model = real_build(config, init, source_model, device)
            self.models.append({key: value.detach().clone() for key, value in model.state_dict().items()})
            return model

        def rows(generator: Any, count: int, batch_size: int) -> torch.Tensor:
            drawn = real_rows(generator, count, batch_size)
            self.rows.append(drawn.clone())
            return drawn

        def batch(stacked: Any, drawn: torch.Tensor, *, scale: Any, device: Any) -> dict[str, torch.Tensor]:
            out = real_batch(stacked, drawn, scale=scale, device=device)
            self.stacked.append(stacked)
            self.batches.append({key: value.detach().clone() for key, value in out.items()})
            return out

        monkeypatch.setattr(few_shot, "build_model", build)
        monkeypatch.setattr(few_shot, "draw_rows", rows)
        monkeypatch.setattr(few_shot, "joint_batch", batch)

    def take(self) -> tuple[list[dict[str, torch.Tensor]], list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        """What was recorded since the last call, and a fresh start."""
        out = (self.models, self.rows, self.batches)
        self.models, self.rows, self.batches, self.stacked = [], [], [], []
        return out


def _unequal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> list[str]:
    """The keys whose tensors differ under ``torch.equal`` (and any key present on one side only)."""
    keys = sorted(set(left) | set(right))
    return [key for key in keys if key not in left or key not in right or not torch.equal(left[key], right[key])]


# ----------------------------------------------------------------------------------------------
# The independent route for T-frozen and T-order: the .npz read raw, the checkpoint's arrays
# ----------------------------------------------------------------------------------------------


def _assert_batches_are_the_checkpoints_view_of_the_corpus(
    prepared: Any, recorder: _Recorder, source_path: Path
) -> int:
    """Every attended position of every recorded batch, recomputed from the raw ``.npz`` BY ID; returns the count.

    The state route: ``(raw - mean) / where(std > 0, std, 1)`` in float32 with the CHECKPOINT's arrays, read from the
    source file by this test (the loader's rule for a zero std, written out here rather than called).  The RTG
    route: the reversed ``np.cumsum`` of the stored float32 rewards in float64, cast to float32, divided in float32
    by the CHECKPOINT's ``rtg_scale`` of that id.  Which episode, id and step a row is comes from ``item_meta``.
    """
    stored = load_payload(source_path)
    order = [str(ix) for ix in stored["intersection_ids"]]
    mean = {ix: np.asarray(stored["stats"]["state_mean"][GRID][ix], dtype=np.float32) for ix in order}
    std = {ix: np.asarray(stored["stats"]["state_std"][GRID][ix], dtype=np.float32) for ix in order}
    divisor = {ix: np.where(std[ix] > 0, std[ix], np.float32(1.0)).astype(np.float32) for ix in order}
    scale = {ix: np.float32(stored["rtg_scale"][ix]) for ix in order}
    dataset = prepared.windows.dataset
    cache: dict[str, Any] = {}

    def episode(meta: Any) -> Any:
        key = str(Path(meta.dataset_dir) / meta.episode_file)
        if key not in cache:
            with np.load(key) as data:
                cache[key] = {name: data[name] for name in data.files}
        return cache[key]

    checked = 0
    assert len(recorder.batches) == len(recorder.stacked) == len(recorder.rows) > 0
    for batch in recorder.batches:
        assert sorted(batch) == ["action", "attention_mask", "avail_mask", "rtg", "state", "timestep"]
    for stacked, rows, batch in zip(recorder.stacked, recorder.rows, recorder.batches):
        members = stacked["member_index"][rows]
        items = stacked["item_index"][members]
        n_batch, n_nodes, context = batch["timestep"].shape
        assert (n_batch, n_nodes) == (64, len(order))
        for b in range(n_batch):
            for n in range(n_nodes):
                meta = dataset.item_meta(int(items[b, n]))
                assert meta.ix_id == order[n], "column n is the CHECKPOINT's node n"
                arrays = episode(meta)
                position = [str(v) for v in arrays["ix_ids"].tolist()].index(meta.ix_id)
                raw = arrays[f"ix{position}_state"].astype(np.float32)
                rewards = arrays[f"ix{position}_local_reward"]
                rtg = np.cumsum(rewards.astype(np.float64)[::-1])[::-1]
                attended = batch["attention_mask"][b, n].numpy()
                steps = batch["timestep"][b, n].numpy()
                assert int(steps[-1]) == meta.t, "a window ends at its item's own step"
                for c in range(context):
                    if not attended[c]:
                        assert not batch["state"][b, n, c].any() and float(batch["rtg"][b, n, c, 0]) == 0.0
                        continue
                    tau = int(steps[c])
                    expected_state = (raw[tau] - mean[meta.ix_id]) / divisor[meta.ix_id]
                    assert np.array_equal(batch["state"][b, n, c].numpy(), expected_state)
                    expected_rtg = np.float32(np.float32(rtg[tau]) / scale[meta.ix_id])
                    assert batch["rtg"][b, n, c, 0].numpy() == expected_rtg
                    checked += 1
    return checked


# ----------------------------------------------------------------------------------------------
# The registered table and the fence
# ----------------------------------------------------------------------------------------------


def test_the_registered_table_is_a24s_thirty_runs_and_the_timing_run_is_not_one_of_them() -> None:
    runs = few_shot.registered_runs()
    assert len(runs) == 30 and len({run.name for run in runs}) == 30
    table = sorted((run.subject, run.init, run.k, run.budget) for run in runs)
    expected = sorted(
        [(f"ft_k{k}", "source", k, 4000) for k in (5, 20, 100) for _ in range(5)]
        + [("scratch_k100", "scratch", 100, 4000)] * 5
        + [("ft_k100_b1000", "source", 100, 1000)] * 5
        + [("ft_k100_b16000", "source", 100, 16000)] * 5
    )
    assert table == expected
    for subject in sorted({run.subject for run in runs}):
        assert sorted(run.seed for run in runs if run.subject == subject) == [101, 202, 303, 404, 505]
    assert [few_shot.run_by_name(run.name) for run in runs] == list(runs)
    assert few_shot.run_by_name("ft_k5_seed101").name == "ft_k5_seed101"
    timing = few_shot.timing_spec()
    assert (timing.init, timing.k, timing.budget, timing.seed) == ("source", 5, 400, 101)
    assert timing.name not in {run.name for run in runs}


def test_an_unregistered_run_name_is_refused() -> None:
    with pytest.raises(ValueError, match=r"'ft_k5_seed606' is not one of the 30 registered runs"):
        few_shot.run_by_name("ft_k5_seed606")


def test_the_registered_destination_is_outside_the_fence_and_a_timing_one_inside(tmp_path: Path) -> None:
    run = few_shot.run_by_name("ft_k20_seed303")
    assert few_shot.registered_destination(tmp_path, run) == (
        tmp_path / "p7_3c_training" / "checkpoints" / "ft_k20_seed303.pt"
    )
    fenced = tmp_path / "p7_3c_training" / "fenced_timing" / "20260925T220000Z" / "x.pt"
    with pytest.raises(ValueError, match=r"lies under fenced_timing"):
        few_shot.assert_fence(fenced, timing=False)
    with pytest.raises(ValueError, match=r"is not under fenced_timing"):
        few_shot.assert_fence(tmp_path / "p7_3c_training" / "checkpoints" / "x.pt", timing=True)
    assert few_shot.assert_fence(fenced, timing=True) == fenced


# ----------------------------------------------------------------------------------------------
# T-prefix
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [5, 20, 100])
def test_T_prefix_k_reads_exactly_the_first_k_draws_from_the_manifest(corpus: Path, k: int) -> None:
    prefix = few_shot.select_prefix(corpus, k, scenario_id=GRID)
    assert prefix.draw_ids == tuple(range(201, 201 + k))
    assert prefix.sums_sha256 == sha256_file(corpus / "SHA256SUMS")
    if k == 5:
        assert 206 in CORPUS_DRAWS and 206 not in prefix.draw_ids


def test_T_prefix_draw_206_present_does_not_enter_k5s_windows(tmp_path: Path, source: tuple[Path, str], corpus: Path) -> None:
    prepared = few_shot.prepare_fine_tune(**_request(source, corpus, tmp_path / "never.pt", k=5))
    loaded = sorted(record.flow_draw for record in prepared.windows.dataset.episode_records)
    assert loaded == [201, 202, 203, 204, 205]


def test_T_prefix_a_corpus_missing_a_prefix_draw_is_refused_by_name(tmp_path: Path) -> None:
    draws = [d for d in CORPUS_DRAWS if d != 203]
    holed = write_training_corpus(tmp_path / "holed", draws=draws, decisions=3, run_metadata=grid_run_metadata(draws))
    with pytest.raises(ValueError, match=r"lacks draw\(s\) \[203\]"):
        few_shot.select_prefix(holed, 5, scenario_id=GRID)


def test_T_prefix_a_draw_outside_the_band_is_refused(tmp_path: Path) -> None:
    draws = list(CORPUS_DRAWS) + [301]
    wide = write_training_corpus(tmp_path / "wide", draws=draws, decisions=3, run_metadata=grid_run_metadata(draws))
    with pytest.raises(ValueError, match=r"draw\(s\) \[301\] outside A24's band 201-300"):
        few_shot.select_prefix(wide, 5, scenario_id=GRID)


def test_T_prefix_a_draw_logged_twice_is_refused(tmp_path: Path) -> None:
    draws = [201] + list(CORPUS_DRAWS)
    doubled = write_training_corpus(tmp_path / "doubled", draws=draws, decisions=3, run_metadata=grid_run_metadata(draws))
    with pytest.raises(ValueError, match=r"draw 201 is logged 2 times"):
        few_shot.select_prefix(doubled, 5, scenario_id=GRID)


def test_T_prefix_an_unregistered_k_is_refused(corpus: Path) -> None:
    with pytest.raises(ValueError, match=r"k 6 is not one of A24's \[5, 20, 100\]"):
        few_shot.select_prefix(corpus, 6, scenario_id=GRID)


def test_a_corpus_of_another_scenario_is_refused(tmp_path: Path) -> None:
    other = write_training_corpus(
        tmp_path / "other", decisions=3, run_metadata=grid_run_metadata(CORPUS_DRAWS, scenario_id="cityflow1x1")
    )
    with pytest.raises(ValueError, match=r"scenario_id 'cityflow1x1', not 'cityflow_grid4x4'"):
        few_shot.select_prefix(other, 5, scenario_id=GRID)


def _tamper(corpus: Path, how: str) -> None:
    if how == "npz":
        target = corpus / "ep000002_seed1000_draw203.npz"
        target.write_bytes(target.read_bytes() + b"\0")
    elif how == "extra":
        (corpus / "notes.txt").write_text("a file the sums do not list\n", encoding="utf-8")
    elif how == "missing":
        (corpus / "ep000099_seed1000_draw300.npz").unlink()
    elif how == "no_sums":
        (corpus / "SHA256SUMS").unlink()


@pytest.mark.parametrize(
    ("how", "message"),
    [
        ("npz", r"ep000002_seed1000_draw203\.npz does not match SHA256SUMS"),
        ("extra", r"notes\.txt is not listed in SHA256SUMS"),
        ("missing", r"ep000099_seed1000_draw300\.npz is listed in SHA256SUMS but absent"),
        ("no_sums", r"has no SHA256SUMS"),
    ],
)
def test_the_corpus_sums_are_verified_entry_by_entry_before_anything_is_read(
    tmp_path: Path, how: str, message: str
) -> None:
    tampered = write_training_corpus(tmp_path / "corpus", decisions=3)
    _tamper(tampered, how)
    with pytest.raises(ValueError, match=message):
        few_shot.select_prefix(tampered, 5, scenario_id=GRID)


# ----------------------------------------------------------------------------------------------
# T-warm
# ----------------------------------------------------------------------------------------------


def test_T_warm_the_fine_tune_starts_from_the_source_weights_with_the_payloads_max_ep_len(
    tmp_path: Path, source: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    long_first = write_training_corpus(tmp_path / "corpus", decisions_for=lambda draw: 300 if draw <= 205 else 3)
    recorder = _Recorder(monkeypatch)
    result = _run(tmp_path, source, long_first, k=5, budget=1)
    models, _rows, batches = recorder.take()
    assert len(models) == 1 and len(batches) == 1
    stored = load_payload(source[0])["model"]
    assert _unequal(models[0], stored) == []
    assert models[0]["embed_timestep.weight"].shape[0] == 360
    written = load_payload(result.destination)
    assert written["config"]["max_ep_len"] == 360


def _subject_checkpoint(seed: int = 101) -> Path:
    root = Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT") or REPO_ROOT / "output")
    path = root / "p5_2" / "checkpoints" / f"grid4x4_mappo1000_dt_nomix_h4_seed{seed}.pt"
    if not path.is_file():
        pytest.skip(f"{path} is absent: set RLTRAFFIC_OUTPUT_ROOT to the output tree holding A20(a)'s subject")
    return path


def test_T_warm_the_real_subject_is_loaded_before_any_step_and_keeps_max_ep_len_360(tmp_path: Path) -> None:
    from offline.transfer_calibration import GRID4X4_CHECKPOINT_SHA256

    checkpoint = _subject_checkpoint()
    stored = load_payload(checkpoint)
    ids = [str(ix) for ix in stored["intersection_ids"]]
    long_first = write_training_corpus(tmp_path / "corpus", ids=ids, decisions_for=lambda draw: 300 if draw <= 205 else 3)
    destination = tmp_path / "never.pt"
    prepared = few_shot.prepare_fine_tune(
        source_path=checkpoint,
        source_sha256=GRID4X4_CHECKPOINT_SHA256[101],
        corpus_dir=long_first,
        k=5,
        budget=4000,
        seed=101,
        init="source",
        device="cpu",
        destination=destination,
    )
    got = {key: value.detach().cpu() for key, value in prepared.model.state_dict().items()}
    assert len(got) == 66
    assert _unequal(got, stored["model"]) == []
    assert prepared.config.max_ep_len == 360 and prepared.model.embed_timestep.num_embeddings == 360
    assert int(prepared.windows.stacked["timestep"].max()) + 1 == 300, "the data alone would have said 300"
    assert not destination.exists()


# ----------------------------------------------------------------------------------------------
# T-frozen and T-order
# ----------------------------------------------------------------------------------------------


def test_T_frozen_every_batch_is_the_checkpoints_statistics_and_rtg_scale_applied_bitwise(
    tmp_path: Path, source: tuple[Path, str], corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = few_shot.prepare_fine_tune(**_request(source, corpus, tmp_path / "never.pt", k=20, budget=2))
    assert prepared.windows.dataset.stats is prepared.windows.stats
    recorder = _Recorder(monkeypatch)
    outcome = few_shot.train_prepared(prepared)
    assert outcome.steps == 2 and len(recorder.batches) == 2 and len(recorder.rows) == 2
    checked = _assert_batches_are_the_checkpoints_view_of_the_corpus(prepared, recorder, source[0])
    assert checked > 64 * 16
    recorded_std = load_payload(source[0])["stats"]["state_std"][GRID]
    zero = sorted({k for ix in IDS for k, value in enumerate(recorded_std[ix]) if value == 0.0})
    assert zero == list(ZERO_STD_COLUMNS), "the loader's zero-std rule is exercised, on the columns the fixture chose"


def test_T_order_a_corpus_logged_in_reversed_order_trains_in_the_checkpoints_order(
    tmp_path: Path, source: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    reversed_ids = list(reversed(IDS))
    backwards = write_training_corpus(tmp_path / "corpus", ids=reversed_ids, decisions=4)
    with np.load(backwards / "ep000000_seed1000_draw201.npz") as data:
        assert [str(v) for v in data["ix_ids"].tolist()] == reversed_ids, "the corpus really is reversed"
    prepared = few_shot.prepare_fine_tune(**_request(source, backwards, tmp_path / "never.pt", k=5, budget=1))
    index, dataset = prepared.windows.index, prepared.windows.dataset
    assert list(index.node_ids) == IDS
    columns = {
        n: {dataset.item_meta(int(item)).ix_id for item in index.member_index[:, n]} for n in range(len(IDS))
    }
    assert columns == {n: {IDS[n]} for n in range(len(IDS))}
    recorder = _Recorder(monkeypatch)
    few_shot.train_prepared(prepared)
    assert _assert_batches_are_the_checkpoints_view_of_the_corpus(prepared, recorder, source[0]) > 0


def test_T_order_a_corpus_missing_one_intersection_is_refused(tmp_path: Path, source: tuple[Path, str]) -> None:
    fifteen = write_training_corpus(tmp_path / "corpus", ids=IDS[:-1], decisions=3)
    with pytest.raises(ValueError, match=r"missing \['D3'\]"):
        few_shot.prepare_fine_tune(**_request(source, fifteen, tmp_path / "never.pt", k=5))


# ----------------------------------------------------------------------------------------------
# T-recipe
# ----------------------------------------------------------------------------------------------


def test_T_recipe_exactly_b_optimizer_steps_counted_by_a_global_hook_from_an_empty_state(
    tmp_path: Path, source: tuple[Path, str], corpus: Path
) -> None:
    from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

    stepped: list[str] = []
    state_at_first_step: list[int] = []

    def before(optimizer: Any, args: Any, kwargs: Any) -> None:
        if not stepped and not state_at_first_step:
            state_at_first_step.append(len(optimizer.state))

    def after(optimizer: Any, args: Any, kwargs: Any) -> None:
        stepped.append(type(optimizer).__name__)

    handles = [register_optimizer_step_pre_hook(before), register_optimizer_step_post_hook(after)]
    try:
        result = _run(tmp_path, source, corpus, k=5, budget=7)
    finally:
        for handle in handles:
            handle.remove()
    assert stepped == ["AdamW"] * 7
    assert state_at_first_step == [0]
    provenance = load_payload(result.destination)["provenance"]
    assert provenance["gradient_steps"] == 7 and result.steps == 7
    assert provenance["warmup_steps"] == 3 and result.warmup == 3


@pytest.mark.parametrize(("budget", "warmup"), [(1000, 500), (4000, 1000), (16000, 1000)])
def test_T_recipe_the_warmup_and_the_learning_rate_at_steps_0_w_minus_1_and_b_minus_1(budget: int, warmup: int) -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimiser, schedule, got = few_shot.make_optimiser([parameter], budget)
    assert got == warmup
    assert type(optimiser) is torch.optim.AdamW
    assert optimiser.defaults["lr"] == 1e-4 and optimiser.defaults["weight_decay"] == 1e-4
    seen: dict[int, float] = {}
    for step in range(budget):
        if step in (0, warmup - 1, budget - 1):
            seen[step] = optimiser.param_groups[0]["lr"]
        optimiser.step()
        schedule.step()
    assert seen == {0: 1e-4 * (1 / warmup), warmup - 1: 1e-4 * 1.0, budget - 1: 1e-4 * 1.0}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 0.0003),
        ("weight_decay", 0.0),
        ("grad_clip", 1.0),
        ("batch_size", 32),
        ("gradient_steps", 20000),
        ("warmup_steps", 500),
        ("method", "dt_nomix"),
        ("n_head", 1),
    ],
)
def test_T_recipe_a_source_whose_recorded_recipe_differs_is_refused(tmp_path: Path, field: str, value: Any) -> None:
    path, digest = write_source(tmp_path / "source.pt", provenance=source_provenance(**{field: value}))
    with pytest.raises(ValueError, match=rf"provenance {field} is {value!r}"):
        few_shot.load_source(path, expected_sha256=digest)


# ----------------------------------------------------------------------------------------------
# The source: its digest and its shape
# ----------------------------------------------------------------------------------------------


def _mask(size: int, *, open_pair: bool) -> list[list[bool]]:
    mask = np.eye(size, dtype=np.bool_)
    if open_pair:
        mask[0, 1] = True
    return mask.tolist()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"format_version": "dt-checkpoint/1.0"}, r"format 'dt-checkpoint/1\.0' is not 'spatial-dt-checkpoint/1\.0'"),
        ({"spatial_mask": _mask(16, open_pair=True)}, r"not the 16 x 16 identity"),
        ({"normalise": False}, r"was not trained on normalised states"),
        ({"stats": None}, r"carries no statistics"),
        ({"scenario_id": "cityflow1x1"}, r"scenario_id 'cityflow1x1', not 'cityflow_grid4x4'"),
    ],
)
def test_a_source_that_is_not_the_registered_subjects_shape_is_refused(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    path, digest = write_source(tmp_path / "source.pt", **overrides)
    with pytest.raises(ValueError, match=message):
        few_shot.load_source(path, expected_sha256=digest)


def test_a_mixing_source_is_refused(tmp_path: Path) -> None:
    from tests.p7_3c_fewshot_fixtures import SMALL_CONFIG

    path, digest = write_source(tmp_path / "source.pt", config={**SMALL_CONFIG, "spatial_mixing": True})
    with pytest.raises(ValueError, match=r"spatial_mixing is on"):
        few_shot.load_source(path, expected_sha256=digest)


def test_a_source_at_another_digest_is_refused_before_it_is_loaded(tmp_path: Path, source: tuple[Path, str]) -> None:
    with pytest.raises(ValueError, match=r"is not the pinned 0{64}"):
        few_shot.load_source(source[0], expected_sha256="0" * 64)


# ----------------------------------------------------------------------------------------------
# The targets: the pinned artifact, the k's own budget, the subject it describes
# ----------------------------------------------------------------------------------------------


def test_the_calibration_artifact_is_read_only_at_its_pinned_digest(tmp_path: Path, source: tuple[Path, str]) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "p7_3d_calibration.json").write_bytes((DATA / "p7_3d_calibration.json").read_bytes() + b"\n")
    with pytest.raises(ValueError, match=r"is not the pinned 3e9df8ee"):
        few_shot.load_k_targets(5, source=few_shot.load_source(source[0], expected_sha256=source[1]), data_dir=data)


def test_a_source_whose_rtg_scale_is_not_the_artifacts_is_refused(tmp_path: Path) -> None:
    from tests.p7_3c_fewshot_fixtures import source_payload

    scales = dict(source_payload()["rtg_scale"])
    scales["A0"] = 1.0
    path, digest = write_source(tmp_path / "source.pt", rtg_scale=scales)
    with pytest.raises(ValueError, match=r"rtg_scale of 'A0' is 1\.0 but the calibration artifact records 259\.0"):
        few_shot.load_k_targets(5, source=few_shot.load_source(path, expected_sha256=digest))


def test_a_prefix_that_overlaps_the_sources_training_draws_is_refused(tmp_path: Path, corpus: Path) -> None:
    path, digest = write_source(tmp_path / "source.pt", stats=synthetic_stats(IDS, draw_ids=list(range(1, 201)) + [203]))
    with pytest.raises(ValueError, match=r"not disjoint"):
        few_shot.prepare_fine_tune(**_request((path, digest), corpus, tmp_path / "never.pt", k=5))


# ----------------------------------------------------------------------------------------------
# T-payload
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [5, 20, 100])
def test_T_payload_frozen_parts_are_the_sources_the_targets_are_k_and_it_loads_for_evaluation(
    tmp_path: Path, source: tuple[Path, str], corpus: Path, k: int
) -> None:
    from offline.rtg_calibration import spatial_agent_with_targets

    result = _run(tmp_path, source, corpus, k=k, budget=2)
    payload = load_payload(result.destination)
    stored = load_payload(source[0])
    for key in ("config", "stats", "rtg_scale", "intersection_ids", "spatial_mask", "normalise", "scenario_id"):
        assert payload[key] == stored[key], key
    assert payload["format_version"] == "spatial-dt-checkpoint/1.0"
    artifact = calibration()
    expected = {ix: float(artifact["per_intersection"][ix]["budgets"][f"k{k}"]["target"]) for ix in IDS}
    assert payload["target_rtg"] == expected
    assert _unequal(payload["model"], stored["model"]) != [], "B steps moved the weights"

    provenance = payload["provenance"]
    assert provenance["gradient_steps"] == 2 and provenance["batch_size"] == 64
    assert (provenance["learning_rate"], provenance["weight_decay"], provenance["grad_clip"]) == (1e-4, 1e-4, 0.25)
    assert provenance["warmup_steps"] == 1
    assert provenance["seed"] == 101 and provenance["seeding"] == SEEDING
    assert (provenance["method"], provenance["n_head"], provenance["spatial_mixing"]) == ("dt_nomix_h4", 4, False)
    assert (provenance["device"], provenance["deterministic"]) == ("cpu", False)
    assert {"sampler", "git_commit", "runtime", "few_shot"} <= set(provenance)
    block = provenance["few_shot"]
    assert block["format_version"] == "few-shot-checkpoint/1.0"
    assert (block["init"], block["k"], block["draw_ids"]) == ("source", k, list(range(201, 201 + k)))
    assert block["target_role"] == ("registered_prompt" if k == 100 else "recorded_not_evaluated")
    assert "target_rule" in block
    assert (block["source_path"], block["source_sha256"]) == (str(Path(source[0]).resolve()), source[1])
    assert (block["source_seed"], block["source_gradient_steps"]) == (101, 40000)
    assert block["source_provenance"] == stored["provenance"] and block["source_target_rtg"] == stored["target_rtg"]
    assert block["corpus_dir"] == str(corpus.resolve())
    assert block["corpus_sha256sums_sha256"] == sha256_file(corpus / "SHA256SUMS")
    assert block["calibration_sha256"] == sha256_file(DATA / "p7_3d_calibration.json")

    agent = spatial_agent_with_targets(
        FakeSpatialEnv(IDS), result.destination, declared_gradient_steps=2, targets=expected, device="cpu"
    )
    assert agent.current_rtg() == expected
    assert agent.config == SpatialDTConfig.from_json_obj(stored["config"])


# ----------------------------------------------------------------------------------------------
# T-cpu-determinism
# ----------------------------------------------------------------------------------------------


def test_T_cpu_determinism_the_same_seed_twice_is_identical_to_the_byte_and_another_seed_is_not(
    tmp_path: Path, source: tuple[Path, str], corpus: Path
) -> None:
    first = _run(tmp_path, source, corpus, name="a.pt", k=20, budget=20, seed=101)
    again = _run(tmp_path, source, corpus, name="b.pt", k=20, budget=20, seed=101)
    other = _run(tmp_path, source, corpus, name="c.pt", k=20, budget=20, seed=202)
    weights = [load_payload(result.destination)["model"] for result in (first, again, other)]
    assert _unequal(weights[0], weights[1]) == []
    assert _unequal(weights[0], weights[2]) != []
    assert sha256_file(first.destination) == sha256_file(again.destination) == first.sha256


# ----------------------------------------------------------------------------------------------
# T-scratch
# ----------------------------------------------------------------------------------------------


def test_T_scratch_differs_from_the_fine_tune_only_in_its_initialisation(
    tmp_path: Path, source: tuple[Path, str], corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _Recorder(monkeypatch)
    tuned = _run(tmp_path, source, corpus, name="ft.pt", k=100, budget=3, seed=101, init="source")
    tuned_models, tuned_rows, tuned_batches = recorder.take()
    scratch = _run(tmp_path, source, corpus, name="scratch.pt", k=100, budget=3, seed=101, init="scratch")
    scratch_models, scratch_rows, scratch_batches = recorder.take()
    stored = load_payload(source[0])

    assert len(tuned_models) == len(scratch_models) == 1
    assert _unequal(tuned_models[0], stored["model"]) == []
    assert sorted(_unequal(scratch_models[0], stored["model"])) == sorted(stored["model"]), (
        "no tensor of the scratch arm's step-0 weights is the source's"
    )
    Utils.seed_everything(101, seed_python_random=False)
    fresh = SpatialDecisionTransformer(SpatialDTConfig.from_json_obj(stored["config"])).state_dict()
    assert _unequal(scratch_models[0], fresh) == []

    assert len(tuned_rows) == len(scratch_rows) == 3
    assert [torch.equal(a, b) for a, b in zip(tuned_rows, scratch_rows)] == [True] * 3
    assert len(tuned_batches) == len(scratch_batches) == 3
    assert [_unequal(a, b) for a, b in zip(tuned_batches, scratch_batches)] == [[]] * 3

    left, right = load_payload(tuned.destination), load_payload(scratch.destination)
    assert (left["provenance"]["few_shot"]["init"], right["provenance"]["few_shot"]["init"]) == ("source", "scratch")
    for payload in (left, right):
        del payload["model"]
        del payload["provenance"]["few_shot"]["init"]
    assert left == right


# ----------------------------------------------------------------------------------------------
# The barrier: a refusal leaves the tree as it was; a failure mid-training writes nothing
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("exists", r"already exists; a fine-tuned checkpoint is written once"),
        ("no_parent", r"does not exist; nothing is created here"),
        ("init", r"init 'warm' is not one of \['source', 'scratch'\]"),
        ("budget", r"budget 0 is not a positive step count"),
        ("k", r"k 6 is not one of A24's \[5, 20, 100\]"),
        ("pin", r"is not the pinned"),
    ],
)
def test_a_refused_run_leaves_the_tree_exactly_as_it_was(
    tmp_path: Path, source: tuple[Path, str], corpus: Path, case: str, message: str
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    destination = out / "run.pt"
    overrides: dict[str, Any] = {}
    if case == "exists":
        destination.write_bytes(b"an earlier checkpoint")
    elif case == "no_parent":
        destination = out / "absent" / "run.pt"
    elif case == "init":
        overrides["init"] = "warm"
    elif case == "budget":
        overrides["budget"] = 0
    elif case == "k":
        overrides["k"] = 6
    elif case == "pin":
        overrides["source_sha256"] = "f" * 64
    before = _tree(out)
    with pytest.raises(ValueError, match=message):
        few_shot.fine_tune(**_request(source, corpus, destination, **overrides))
    assert _tree(out) == before


def test_a_run_that_fails_mid_training_writes_nothing(
    tmp_path: Path, source: tuple[Path, str], corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_batch: Callable[..., Any] = few_shot.joint_batch
    calls: list[int] = []

    def failing(stacked: Any, rows: torch.Tensor, *, scale: Any, device: Any) -> Any:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("an injected failure at step 1")
        return real_batch(stacked, rows, scale=scale, device=device)

    monkeypatch.setattr(few_shot, "joint_batch", failing)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(RuntimeError, match=r"an injected failure at step 1"):
        few_shot.fine_tune(**_request(source, corpus, out / "run.pt", budget=3))
    assert _tree(out) == {}


def test_the_exclusive_write_never_replaces_an_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "run.pt"
    destination.write_bytes(b"an earlier checkpoint")
    with pytest.raises(FileExistsError, match=r"already exists"):
        few_shot.write_payload_exclusive({"format_version": "spatial-dt-checkpoint/1.0"}, destination)
    assert _tree(tmp_path) == {"run.pt": b"an earlier checkpoint"}


# ----------------------------------------------------------------------------------------------
# The registered command: in-process, refusals before anything is read
# ----------------------------------------------------------------------------------------------


def test_the_train_command_refuses_a_device_other_than_cuda_before_anything_is_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = few_shot.main(
        ["train", "--run", "ft_k5_seed101", "--device", "cpu", "--output-root", str(tmp_path),
         "--corpus-dir", str(tmp_path / "absent")]
    )
    assert code == 2
    assert "the registered runs train on cuda only" in capsys.readouterr().out
    assert _tree(tmp_path) == {}


def test_the_train_command_wires_a_registered_run_to_its_source_pin_target_k_and_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from offline.transfer_calibration import GRID4X4_CHECKPOINT_SHA256

    seen: list[dict[str, Any]] = []

    def fake_fine_tune(**kwargs: Any) -> Any:
        seen.append(kwargs)
        raise RuntimeError("stop after the wiring is recorded")

    monkeypatch.setattr(few_shot, "fine_tune", fake_fine_tune)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for sub in ("checkpoints", "runs"):
        (tmp_path / "p7_3c_training" / sub).mkdir(parents=True)
    with pytest.raises(RuntimeError, match=r"stop after the wiring is recorded"):
        few_shot.main(
            ["train", "--run", "ft_k100_b16000_seed404", "--device", "cuda", "--output-root", str(tmp_path),
             "--corpus-dir", str(tmp_path / "corpus")]
        )
    assert len(seen) == 1
    call = seen[0]
    assert Path(call["source_path"]) == tmp_path / "p5_2" / "checkpoints" / "grid4x4_mappo1000_dt_nomix_h4_seed404.pt"
    assert call["source_sha256"] == GRID4X4_CHECKPOINT_SHA256[404]
    assert (call["k"], call["budget"], call["seed"], call["init"], call["device"]) == (100, 16000, 404, "source", "cuda")
    assert Path(call["destination"]) == tmp_path / "p7_3c_training" / "checkpoints" / "ft_k100_b16000_seed404.pt"
    assert Path(call["corpus_dir"]) == tmp_path / "corpus"
