"""P7.3c C4 (``BRIEF_41``): the training driver's Python half -- resume, attempts, manifest, record, timing, inputs.

The driver (``offline/campaigns/p7_3c_finetune.sh``) decides nothing in bash that could train a checkpoint twice,
overwrite one, or let a fenced one through: every such decision is made in ``offline.few_shot`` and pinned HERE,
in-process -- no test in this file starts a process (``BRIEF_41`` Amendment B, B5).

* **Resume, decided in Python, never by ``[ -f ]``** (plan section 8): absent -> ``train``; present AND valid ->
  ``skip``; present and NOT valid -> refused, and the file is never touched.  "Valid" is every frozen part equal
  to the source's, the budget, k, the k targets, the switch and the source's pin.
* **Attempt markers** are counted on disk: a marker with no checkpoint is a re-run of an infrastructure failure.
* **The manifest** lists every declared checkpoint by digest, refuses a missing or a stray one, and is never
  rewritten with different content.
* **The record** (Amendment A, Q9: committed on ``main`` at G7) carries each run's digests, frozen-part checks,
  steps, seconds, loss and attempts, the corpus and calibration digests, and G5's timing record.
* **The concurrency rule, fixed before any measurement** (plan section 8): the C in {1, 2, 3} with the largest
  aggregate throughput whose device peak is <= 80 % of 16,303 MiB; a tie goes to the smaller C.
* **The inputs**, checked by digest before the canary and the token.

Checkpoints here are trained on CPU from the synthetic source (``tests/p7_3c_fewshot_fixtures.py``) at small,
UNREGISTERED budgets; every function takes the run as a ``RunSpec`` and the source pins as an argument, so nothing
here needs A20(a)'s real checkpoints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from offline import few_shot
from tests.p7_3c_fewshot_fixtures import (
    load_payload,
    sha256_file,
    source_provenance,
    write_source,
    write_training_corpus,
)
from tests.p7_3c_fixtures import GRID

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"
SPEC = few_shot.RunSpec(subject="ft_k5", init="source", k=5, budget=3, seed=101)
SCRATCH = few_shot.RunSpec(subject="scratch_k100", init="scratch", k=100, budget=2, seed=101)
TIMING_FORMAT = "p7.3c-finetune-timing/1.0"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_training_corpus(tmp_path_factory.mktemp("corpus") / "grid4x4_sumo_maxpressure")


def _root(tmp_path: Path, seeds: tuple[int, ...] = (101,)) -> tuple[Path, dict[int, str]]:
    """An output root with synthetic sources at the REGISTERED source paths, their pins, and the training dirs."""
    root = tmp_path / "output"
    pins: dict[int, str] = {}
    for seed in seeds:
        path = few_shot.registered_source_path(root, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        _written, pins[seed] = write_source(path, provenance=source_provenance(seed=seed))
    for sub in ("checkpoints", "runs", "attempts"):
        (root / "p7_3c_training" / sub).mkdir(parents=True, exist_ok=True)
    return root, pins


def _train(root: Path, pins: dict[int, str], corpus: Path, spec: Any) -> Any:
    return few_shot.fine_tune(
        source_path=few_shot.registered_source_path(root, spec.seed),
        source_sha256=pins[spec.seed],
        corpus_dir=corpus,
        k=spec.k,
        budget=spec.budget,
        seed=spec.seed,
        init=spec.init,
        device="cpu",
        destination=few_shot.registered_destination(root, spec),
    )


def _timing(tmp_path: Path, concurrency: int = 1) -> Path:
    path = tmp_path / "timing.json"
    path.write_text(json.dumps({"format_version": TIMING_FORMAT, "concurrency": concurrency}) + "\n", encoding="utf-8")
    return path


def _weights_digest(path: Path) -> str:
    """The weights-only digest by THIS test's route: sorted keys, then key, dtype, shape and raw bytes."""
    model = load_payload(path)["model"]
    digest = hashlib.sha256()
    for key in sorted(model):
        tensor = model[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


# ----------------------------------------------------------------------------------------------
# Resume, decided in Python
# ----------------------------------------------------------------------------------------------


def test_resume_is_train_when_absent_and_skip_when_the_checkpoint_validates(tmp_path: Path, corpus: Path) -> None:
    root, pins = _root(tmp_path)
    assert few_shot.resume_decision(SPEC, output_root=root, pins=pins) == "train"
    _train(root, pins, corpus, SPEC)
    assert few_shot.resume_decision(SPEC, output_root=root, pins=pins) == "skip"
    checks = few_shot.validate_checkpoint(SPEC, output_root=root, pins=pins)
    assert sorted(checks) == sorted(
        [
            "format_version", "few_shot_format", "config", "stats", "rtg_scale", "intersection_ids",
            "spatial_mask", "normalise", "scenario_id", "source_sha256", "source_seed", "budget", "k",
            "draw_ids", "init", "targets", "recipe",
        ]
    )
    assert [name for name, passed in checks.items() if not passed] == []


def _set(path: tuple[str, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def change(payload: dict[str, Any]) -> None:
        target: Any = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return change


def _bump_target(payload: dict[str, Any]) -> None:
    payload["target_rtg"]["A0"] = payload["target_rtg"]["A0"] + 1.0


def _bump_mean(payload: dict[str, Any]) -> None:
    payload["stats"]["state_mean"][GRID]["A0"][0] = payload["stats"]["state_mean"][GRID]["A0"][0] + 1.0


def _bump_scale(payload: dict[str, Any]) -> None:
    payload["rtg_scale"]["B1"] = payload["rtg_scale"]["B1"] * 2.0


TAMPERS: dict[str, Callable[[dict[str, Any]], None]] = {
    "targets": _bump_target,
    "budget": _set(("provenance", "gradient_steps"), 4),
    "init": _set(("provenance", "few_shot", "init"), "scratch"),
    "k": _set(("provenance", "few_shot", "k"), 20),
    "stats": _bump_mean,
    "rtg_scale": _bump_scale,
    "config": _set(("config", "dropout"), 0.2),
    "source_sha256": _set(("provenance", "few_shot", "source_sha256"), "0" * 64),
    "few_shot_format": _set(("provenance", "few_shot", "format_version"), "few-shot-checkpoint/0.9"),
}


@pytest.mark.parametrize("check", sorted(TAMPERS))
def test_resume_refuses_a_checkpoint_that_does_not_validate_and_never_touches_it(
    tmp_path: Path, corpus: Path, check: str
) -> None:
    root, pins = _root(tmp_path)
    result = _train(root, pins, corpus, SPEC)
    payload = load_payload(result.destination)
    TAMPERS[check](payload)
    torch.save(payload, result.destination)
    before = result.destination.read_bytes()
    with pytest.raises(ValueError, match=rf"exists but does not validate \(failed: \[[^\]]*'{check}'"):
        few_shot.resume_decision(SPEC, output_root=root, pins=pins)
    assert result.destination.read_bytes() == before


def test_resume_refuses_an_unreadable_checkpoint_and_never_touches_it(tmp_path: Path) -> None:
    root, pins = _root(tmp_path)
    destination = few_shot.registered_destination(root, SPEC)
    destination.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match=r"exists but cannot be read"):
        few_shot.resume_decision(SPEC, output_root=root, pins=pins)
    assert destination.read_bytes() == b"not a checkpoint"


# ----------------------------------------------------------------------------------------------
# Attempt markers
# ----------------------------------------------------------------------------------------------


def test_every_start_leaves_a_marker_and_a_marker_without_a_checkpoint_is_a_rerun(tmp_path: Path, corpus: Path) -> None:
    root, pins = _root(tmp_path)
    assert few_shot.attempts_of(SPEC, output_root=root) == ()
    assert few_shot.next_attempt(SPEC, output_root=root) == 1
    # The first attempt died before its checkpoint existed: the decision is still "train".
    assert few_shot.resume_decision(SPEC, output_root=root, pins=pins) == "train"
    assert few_shot.next_attempt(SPEC, output_root=root) == 2
    result = _train(root, pins, corpus, SPEC)
    few_shot.write_run_record(SPEC, result, output_root=root, device="cpu")
    assert few_shot.attempts_of(SPEC, output_root=root) == (1, 2)
    assert sorted(path.name for path in (root / "p7_3c_training" / "attempts").iterdir()) == [
        "ft_k5_seed101.1", "ft_k5_seed101.2"
    ]
    few_shot.write_manifest(root, runs=[SPEC])
    record = few_shot.build_record(root, corpus_dir=corpus, timing_path=_timing(tmp_path), runs=[SPEC], pins=pins)
    entry = record["runs"][SPEC.name]
    assert (entry["attempts"], entry["reruns"]) == (2, 1)


def test_a_run_record_is_written_once(tmp_path: Path, corpus: Path) -> None:
    root, pins = _root(tmp_path)
    result = _train(root, pins, corpus, SPEC)
    path = few_shot.write_run_record(SPEC, result, output_root=root, device="cpu")
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match=r"already exists"):
        few_shot.write_run_record(SPEC, result, output_root=root, device="cpu")
    assert path.read_bytes() == before
    written = json.loads(before)
    assert written["checkpoint_sha256"] == sha256_file(result.destination)
    assert (written["steps"], written["warmup_steps"], written["losses"]) == (3, 1, list(result.losses))
    assert written["final_loss"] == result.losses[-1]


# ----------------------------------------------------------------------------------------------
# The manifest
# ----------------------------------------------------------------------------------------------


def test_the_manifest_lists_every_checkpoint_by_digest_and_refuses_a_missing_or_stray_one(
    tmp_path: Path, corpus: Path
) -> None:
    root, pins = _root(tmp_path)
    specs = [SPEC, SCRATCH]
    manifest = root / "SHA256SUMS_p7_3c_finetune.txt"
    with pytest.raises(ValueError, match=r"ft_k5_seed101\.pt is absent"):
        few_shot.write_manifest(root, runs=specs)
    assert not manifest.exists()

    results = {spec.name: _train(root, pins, corpus, spec) for spec in specs}
    assert few_shot.write_manifest(root, runs=specs) == manifest
    lines = manifest.read_text(encoding="utf-8").splitlines()
    expected = sorted(
        (f"p7_3c_training/checkpoints/{name}.pt", sha256_file(result.destination)) for name, result in results.items()
    )
    assert lines == [f"{digest}  {relative}" for relative, digest in expected]
    assert few_shot.write_manifest(root, runs=specs) == manifest, "the same content again is not a rewrite"

    stray = root / "p7_3c_training" / "checkpoints" / "stray.pt"
    stray.write_bytes(b"left behind")
    with pytest.raises(ValueError, match=r"stray\.pt is not one of the declared runs"):
        few_shot.write_manifest(root, runs=specs)
    stray.unlink()

    manifest.write_text(lines[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"already exists and differs"):
        few_shot.write_manifest(root, runs=specs)
    assert manifest.read_text(encoding="utf-8") == lines[0] + "\n"


# ----------------------------------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------------------------------


def test_the_record_carries_digests_frozen_checks_seconds_loss_attempts_and_the_timing(
    tmp_path: Path, corpus: Path
) -> None:
    root, pins = _root(tmp_path)
    few_shot.next_attempt(SPEC, output_root=root)
    result = _train(root, pins, corpus, SPEC)
    run_record = few_shot.write_run_record(SPEC, result, output_root=root, device="cpu")
    timing = _timing(tmp_path)
    with pytest.raises(ValueError, match=r"SHA256SUMS_p7_3c_finetune\.txt is absent: the manifest is written first"):
        few_shot.build_record(root, corpus_dir=corpus, timing_path=timing, runs=[SPEC], pins=pins)
    few_shot.write_manifest(root, runs=[SPEC])
    record = few_shot.build_record(root, corpus_dir=corpus, timing_path=timing, runs=[SPEC], pins=pins)

    assert record["format_version"] == "p7.3c-finetune-record/1.0"
    assert record["n_runs"] == 1
    entry = record["runs"]["ft_k5_seed101"]
    assert (entry["subject"], entry["init"], entry["k"], entry["budget"], entry["seed"]) == ("ft_k5", "source", 5, 3, 101)
    assert entry["checkpoint"] == "p7_3c_training/checkpoints/ft_k5_seed101.pt"
    assert entry["checkpoint_sha256"] == sha256_file(result.destination)
    assert entry["weights_sha256"] == _weights_digest(result.destination)
    assert entry["source_sha256"] == pins[101]
    assert entry["frozen_checks"] == few_shot.validate_checkpoint(SPEC, output_root=root, pins=pins)
    assert (entry["steps"], entry["warmup_steps"]) == (3, 1)
    stored = json.loads(run_record.read_text(encoding="utf-8"))
    assert (entry["loop_seconds"], entry["final_loss"]) == (stored["loop_seconds"], stored["final_loss"])
    assert (entry["attempts"], entry["reruns"]) == (1, 0)
    assert record["corpus_sha256sums_sha256"] == sha256_file(corpus / "SHA256SUMS")
    assert record["calibration_sha256"] == sha256_file(DATA / "p7_3d_calibration.json")
    assert record["manifest_sha256"] == sha256_file(root / "SHA256SUMS_p7_3c_finetune.txt")
    assert record["timing"] == {"path": str(timing), "sha256": sha256_file(timing), "record": json.loads(timing.read_text())}

    path = few_shot.write_record(root, record)
    assert path == root / "p7_3c_training" / "p7_3c_finetune.json"
    assert json.loads(path.read_text(encoding="utf-8")) == record
    assert few_shot.write_record(root, record) == path, "the same content again is not a rewrite"
    with pytest.raises(ValueError, match=r"already exists and differs"):
        few_shot.write_record(root, {**record, "n_runs": 31})
    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_the_record_refuses_a_checkpoint_whose_frozen_parts_differ(tmp_path: Path, corpus: Path) -> None:
    root, pins = _root(tmp_path)
    result = _train(root, pins, corpus, SPEC)
    few_shot.write_run_record(SPEC, result, output_root=root, device="cpu")
    payload = load_payload(result.destination)
    _bump_scale(payload)
    torch.save(payload, result.destination)
    few_shot.write_manifest(root, runs=[SPEC])
    with pytest.raises(ValueError, match=r"ft_k5_seed101 does not validate \(failed: \['rtg_scale'\]\)"):
        few_shot.build_record(root, corpus_dir=corpus, timing_path=_timing(tmp_path), runs=[SPEC], pins=pins)
    assert not (root / "p7_3c_training" / "p7_3c_finetune.json").exists()


# ----------------------------------------------------------------------------------------------
# The concurrency rule and G5's timing summary
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("measured", "expected"),
    [
        ({1: (100.0, 3000.0), 2: (150.0, 6000.0), 3: (240.0, 9000.0)}, 2),
        ({1: (100.0, 3000.0), 2: (200.0, 6000.0), 3: (300.0, 9000.0)}, 1),
        ({1: (100.0, 3000.0), 2: (120.0, 6000.0), 3: (130.0, 13100.0)}, 2),
        ({1: (100.0, 3000.0), 2: (120.0, 13043.0), 3: (130.0, 13050.0)}, 1),
    ],
)
def test_the_concurrency_rule_fixed_before_any_measurement(measured: dict[int, tuple[float, float]], expected: int) -> None:
    table = {c: {"ms_per_step": ms, "device_peak_mib": mib} for c, (ms, mib) in measured.items()}
    assert few_shot.choose_concurrency(table) == expected


def test_the_memory_cap_is_inclusive_at_eighty_percent_of_16303_mib() -> None:
    """``<=``, not ``<``: a peak exactly AT the cap is admitted, the next float above it is not.  The cap is
    computed here by the rule's own arithmetic (``16303 * 0.8``), so the boundary is exact, not rounded."""
    import math

    cap = 16303 * 0.8
    at_cap = {1: {"ms_per_step": 100.0, "device_peak_mib": 3000.0}, 2: {"ms_per_step": 120.0, "device_peak_mib": cap}}
    assert few_shot.choose_concurrency(at_cap) == 2
    over = {**at_cap, 2: {"ms_per_step": 120.0, "device_peak_mib": math.nextafter(cap, math.inf)}}
    assert few_shot.choose_concurrency(over) == 1


def test_a_gpu_that_cannot_hold_one_run_is_refused() -> None:
    with pytest.raises(ValueError, match=r"cannot hold one run"):
        few_shot.choose_concurrency({1: {"ms_per_step": 100.0, "device_peak_mib": 13100.0}})


def _slot(stamp: Path, slot: str, seconds: float, *, file_digest: str, weights_digest: str, peak: float) -> None:
    (stamp / f"{slot}.json").write_text(
        json.dumps(
            {
                "format_version": "p7.3c-finetune-timing-slot/1.0",
                "slot": slot,
                "steps": 400,
                "loop_seconds": seconds,
                "peak_allocated_mib": peak,
                "checkpoint_sha256": file_digest,
                "weights_sha256": weights_digest,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_the_timing_summary_measures_each_phase_and_judges_the_repeat_by_two_routes(tmp_path: Path) -> None:
    stamp = tmp_path / "p7_3c_training" / "fenced_timing" / "20260925T230000Z"
    stamp.mkdir(parents=True)
    _slot(stamp, "alone", 48.8, file_digest="a" * 64, weights_digest="b" * 64, peak=2100.0)
    _slot(stamp, "pair1", 60.0, file_digest="1" * 64, weights_digest="2" * 64, peak=2100.0)
    _slot(stamp, "pair2", 61.6, file_digest="3" * 64, weights_digest="4" * 64, peak=2100.0)
    for index, seconds in ((1, 92.0), (2, 92.4), (3, 93.2)):
        _slot(stamp, f"triple{index}", seconds, file_digest=f"{index}" * 64, weights_digest="5" * 64, peak=2100.0)
    _slot(stamp, "repeat", 48.4, file_digest="c" * 64, weights_digest="b" * 64, peak=2100.0)
    for phase, samples in (("alone", "1063\n3200\n3150\n"), ("pair", "1063\n5300\n"), ("triple", "1063\n7400\n")):
        (stamp / f"nvidia_smi_{phase}.csv").write_text(samples, encoding="utf-8")
    (stamp / "build_k100.json").write_text(
        json.dumps({"format_version": "p7.3c-finetune-timing-build/1.0", "k": 100, "seconds": 95.0, "peak_rss_mib": 6100.0})
        + "\n",
        encoding="utf-8",
    )

    summary = few_shot.summarize_timing(stamp)
    assert summary["format_version"] == TIMING_FORMAT
    phases = summary["phases"]
    alone = 48.8 / 400 * 1000
    pair = ((60.0 / 400 * 1000) + (61.6 / 400 * 1000)) / 2
    triple = sum(s / 400 * 1000 for s in (92.0, 92.4, 93.2)) / 3
    assert (phases["alone"]["ms_per_step"], phases["pair"]["ms_per_step"], phases["triple"]["ms_per_step"]) == (
        alone, pair, triple
    )
    assert (phases["pair"]["slowdown"], phases["triple"]["slowdown"]) == (pair / alone, triple / alone)
    assert (phases["alone"]["device_peak_mib"], phases["pair"]["device_peak_mib"], phases["triple"]["device_peak_mib"]) == (
        3200.0, 5300.0, 7400.0
    )
    assert summary["repeat"] == {"file_sha256_equal": False, "weights_sha256_equal": True}
    table = {
        1: {"ms_per_step": alone, "device_peak_mib": 3200.0},
        2: {"ms_per_step": pair, "device_peak_mib": 5300.0},
        3: {"ms_per_step": triple, "device_peak_mib": 7400.0},
    }
    assert summary["concurrency"] == few_shot.choose_concurrency(table)
    assert summary["build_k100"]["seconds"] == 95.0 and summary["q12_one_process_per_k"] is False


def test_the_timing_destination_is_fenced(tmp_path: Path) -> None:
    path = few_shot.timing_destination(tmp_path, "20260925T230000Z", "pair1")
    assert path == tmp_path / "p7_3c_training" / "fenced_timing" / "20260925T230000Z" / "pair1.pt"
    assert few_shot.assert_fence(path, timing=True) == path
    with pytest.raises(ValueError, match=r"lies under fenced_timing"):
        few_shot.assert_fence(path, timing=False)


# ----------------------------------------------------------------------------------------------
# The inputs, by digest, before the canary and the token
# ----------------------------------------------------------------------------------------------


def _gate_record(tmp_path: Path, corpus: Path, **overrides: Any) -> Path:
    record: dict[str, Any] = {
        "format_version": "p7.3c-corpus-gate/1.0",
        "corpus_dir": str(corpus),
        "all_match": True,
        "n_draws": 100,
        "n_intersections": 16,
        "n_checked": 1600,
        "n_matching": 1600,
        "engine_events": {"n_teleports": 0, "n_collisions": 0},
    }
    record.update(overrides)
    path = tmp_path / "a17f_gate.json"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def cuda_free(monkeypatch: pytest.MonkeyPatch) -> Callable[[float], None]:
    """Stand in for the device: CUDA available, *free* MiB of 16,303."""

    def set_free(free_mib: float) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda *a, **k: (int(free_mib * 2**20), int(16303 * 2**20))
        )

    return set_free


def test_check_inputs_passes_on_the_complete_inputs_and_names_each_digest(
    tmp_path: Path, corpus: Path, cuda_free: Callable[[float], None]
) -> None:
    root, pins = _root(tmp_path, seeds=(101, 202, 303, 404, 505))
    cuda_free(15000.0)
    facts = few_shot.check_inputs(
        output_root=root, corpus_dir=corpus, gate_record=_gate_record(tmp_path, corpus), pins=pins
    )
    assert facts["calibration_sha256"] == sha256_file(DATA / "p7_3d_calibration.json")
    assert facts["source_sha256"] == {seed: pins[seed] for seed in (101, 202, 303, 404, 505)}
    assert facts["corpus_sha256sums_sha256"] == sha256_file(corpus / "SHA256SUMS")
    assert facts["gate_record_sha256"] == sha256_file(tmp_path / "a17f_gate.json")
    assert facts["cuda_free_mib"] == 15000.0


def _missing_source(root: Path) -> None:
    few_shot.registered_source_path(root, 303).unlink()


def _tampered_sums(corpus_copy: Path) -> None:
    target = corpus_copy / "ep000004_seed1000_draw205.npz"
    target.write_bytes(target.read_bytes() + b"\0")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("source_missing", r"grid4x4_mappo1000_dt_nomix_h4_seed303\.pt is absent"),
        ("source_pin", r"seed 404: .* is not the pinned 0{64}"),
        ("corpus", r"ep000004_seed1000_draw205\.npz does not match SHA256SUMS"),
        ("gate_absent", r"a17f_gate\.json is absent: G3's gate record"),
        ("gate_failed", r"the gate record says all_match False"),
        ("gate_count", r"the gate record checked 1599 returns, not 1600"),
        ("gate_events", r"the gate record counts 1 teleport\(s\) and 0 collision\(s\)"),
        ("gate_corpus", r"the gate record is for .*elsewhere, not"),
        ("no_cuda", r"CUDA is not available"),
    ],
)
def test_check_inputs_refuses_each_missing_or_mismatched_input(
    tmp_path: Path, corpus: Path, cuda_free: Callable[[float], None], monkeypatch: pytest.MonkeyPatch,
    case: str, message: str,
) -> None:
    root, pins = _root(tmp_path, seeds=(101, 202, 303, 404, 505))
    cuda_free(15000.0)
    corpus_dir = corpus
    gate = _gate_record(tmp_path, corpus)
    if case == "source_missing":
        _missing_source(root)
    elif case == "source_pin":
        pins = {**pins, 404: "0" * 64}
    elif case == "corpus":
        corpus_dir = write_training_corpus(tmp_path / "corpus_copy")
        gate = _gate_record(tmp_path, corpus_dir)
        _tampered_sums(corpus_dir)
    elif case == "gate_absent":
        gate.unlink()
    elif case == "gate_failed":
        gate = _gate_record(tmp_path, corpus, all_match=False)
    elif case == "gate_count":
        gate = _gate_record(tmp_path, corpus, n_checked=1599)
    elif case == "gate_events":
        gate = _gate_record(tmp_path, corpus, engine_events={"n_teleports": 1, "n_collisions": 0})
    elif case == "gate_corpus":
        gate = _gate_record(tmp_path, corpus, corpus_dir=str(tmp_path / "elsewhere"))
    elif case == "no_cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match=message):
        few_shot.check_inputs(output_root=root, corpus_dir=corpus_dir, gate_record=gate, pins=pins)


def test_check_inputs_with_a_timing_record_refuses_too_little_free_device_memory(
    tmp_path: Path, corpus: Path, cuda_free: Callable[[float], None]
) -> None:
    root, pins = _root(tmp_path, seeds=(101, 202, 303, 404, 505))
    timing = tmp_path / "timing.json"
    timing.write_text(
        json.dumps(
            {
                "format_version": TIMING_FORMAT,
                "concurrency": 2,
                "phases": {
                    "alone": {"ms_per_step": 122.0, "device_peak_mib": 3200.0},
                    "pair": {"ms_per_step": 150.0, "device_peak_mib": 5300.0},
                    "triple": {"ms_per_step": 240.0, "device_peak_mib": 7400.0},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cuda_free(4000.0)
    with pytest.raises(ValueError, match=r"4000 MiB free on the device, below the 5300 MiB G5 measured at concurrency 2"):
        few_shot.check_inputs(
            output_root=root, corpus_dir=corpus, gate_record=_gate_record(tmp_path, corpus), pins=pins, timing_path=timing
        )


# ----------------------------------------------------------------------------------------------
# The commands, in-process: what each prints and returns
# ----------------------------------------------------------------------------------------------


def test_the_resume_decision_command_prints_the_decision_and_refuses_with_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    decisions = iter(["skip", "train"])

    def fake(spec: Any, **kwargs: Any) -> str:
        assert spec == few_shot.run_by_name("ft_k20_seed202")
        return next(decisions)

    monkeypatch.setattr(few_shot, "resume_decision", fake)
    argv = ["resume-decision", "--run", "ft_k20_seed202", "--output-root", str(tmp_path)]
    assert few_shot.main(argv) == 0 and capsys.readouterr().out.strip() == "skip"
    assert few_shot.main(argv) == 0 and capsys.readouterr().out.strip() == "train"

    def refuse(spec: Any, **kwargs: Any) -> str:
        raise ValueError("x.pt exists but does not validate (failed: ['targets'])")

    monkeypatch.setattr(few_shot, "resume_decision", refuse)
    assert few_shot.main(argv) == 2
    assert "does not validate" in capsys.readouterr().out
    assert few_shot.main(["resume-decision", "--run", "ft_k20_seed999", "--output-root", str(tmp_path)]) == 2
    assert "is not one of the 30 registered runs" in capsys.readouterr().out
