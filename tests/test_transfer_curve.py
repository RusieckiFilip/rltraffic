"""P7.3a T6-T8: the declared arms, rho's arithmetic, the two stages and ``report``.

Written against ``BRIEF_37`` §3.5 / §4 + Amendments A (A1, A2), B (B1, B2) and C (C2, C6), and
``docs/plans/p7.3a.md``.

**Nothing scientific is chosen in this file or in the module it tests.** H3, rho, the arms, the
seeds, the pool and the stages are registered; these tests assert that what runs is what was
registered, and that a number cannot be produced for a cell nobody declared.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import pytest

from offline import transfer_calibration as tc
from offline import transfer_curve as tcv


def _head_sha() -> str:
    """This tree's HEAD, so a fixture chunk carries a commit the repository actually knows.

    Amendment J1(c) compares a chunk's commit with HEAD through ``git diff``; a fabricated sha
    (the ``"0" * 40`` these fixtures used to carry) is an unknown revision, and a chunk whose
    provenance cannot be resolved is exactly what J1 requires to be refused.
    """
    return subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


HEAD_SHA = _head_sha()

REPO_DATA = Path(__file__).resolve().parents[1] / "docs" / "data"
P7_2B = REPO_DATA / "p7_2b_calibration.json"

#: The MAIN tree's roots. P7.2a's parity configs and P4's checkpoints are gitignored and live
#: there only (``DEFERRED`` 82: existing modules hardcode this; new code takes roots as parameters
#: and these constants exist so the GATED tests can name the artifact they consume, G2).
MAIN_TREE = Path("/home/filip/rltraffic")
DRAWS_ROOT = MAIN_TREE / "scenarios" / "draws"
OUTPUT_ROOT = MAIN_TREE / "output"
SMOKE_DRAW = 5
IX = "intersection_1_1"


# ----------------------------------------------------------------------------------
# T6 -- the arm set and the targets, READ from the sha-pinned artifact
# ----------------------------------------------------------------------------------
def test_t6_exactly_four_declared_arms_per_subject_with_the_registered_one_marked() -> None:
    """T6. The four arms ``BRIEF_37`` §2 closed, and the registered prompt is the artifact's.

    The lookup is by ``(rule, statistic, k)`` and the ``role`` is then cross-checked against the
    artifact's own: a table that disagreed would silently re-register the prompt, which is the one
    thing A17 does not permit this task to do.
    """
    artifact = json.loads(P7_2B.read_bytes())
    assert [spec.name for spec in tcv.DECLARED_ARMS] == [
        "b_mean_k100", "b_max_k100", "a_q1.0", "naive",
    ]

    for subject, expected_registered in (
        ("mappo1000", -7185.354721543778),
        ("mix50", -7431.0185327454665),
    ):
        targets = tcv.targets_for_subject(subject, artifact)
        assert sorted(targets) == ["a_q1.0", "b_max_k100", "b_mean_k100", "naive"]
        assert targets["b_mean_k100"]["role"] == "registered_prompt"
        assert targets["b_mean_k100"]["target_rtg"] == expected_registered
        assert [t["role"] for name, t in sorted(targets.items()) if name != "b_mean_k100"] == [
            "ablation", "ablation", "ablation",
        ]

    # k = 5 and k = 20 are EXCLUDED BY DECLARATION, before any number existed (A17(d)).
    assert all(spec.k in (100, None) for spec in tcv.DECLARED_ARMS)


def test_t6_a_role_disagreement_with_the_artifact_is_refused() -> None:
    """Reading ``b_max`` where ``b_mean`` is declared must not quietly produce a target.

    The mutation this is built against swaps the statistic in the lookup.  Without the role
    cross-check that swap returns a perfectly well-formed number for an arm labelled
    ``registered_prompt`` -- the H3 arm conditioned on the wrong prompt, with nothing to say so.
    """
    artifact = json.loads(P7_2B.read_bytes())
    tampered = json.loads(json.dumps(artifact))
    for row in tampered["targets"]["mappo1000"]:
        if (row["rule"], row["statistic"], row["k"]) == ("rule_b", "mean", 100):
            row["role"] = "ablation"

    with pytest.raises(ValueError, match="registered_prompt|role"):
        tcv.targets_for_subject("mappo1000", tampered)


# ----------------------------------------------------------------------------------
# T7 -- rho's arithmetic: the anchors are 0 and 1 BY CONSTRUCTION
# ----------------------------------------------------------------------------------
def test_t7_the_anchors_are_zero_and_one_exactly() -> None:
    """T7. §3.4's formula, with each anchor substituted for the arm: exactly 0.0 and exactly 1.0.

    Under ``==``, not ``approx``: the identity is algebraic, and a rho that is 0.9999999 for
    fixed-time is a defect in the pairing rather than a rounding question.
    """
    for fixedtime, maxpressure in ((366.07, 364.15), (1000.0, 1.0), (2.5, -7.5)):
        assert tcv.rho(fixedtime, fixedtime, maxpressure) == 0.0
        assert tcv.rho(maxpressure, fixedtime, maxpressure) == 1.0


def test_t7_values_outside_the_unit_interval_are_not_clipped() -> None:
    """§3.4: *"Values may exceed 100 or fall below 0; that is expected and is not clipped."*

    A DT worse than fixed-time gives a negative rho and a DT better than MaxPressure gives one
    above 1.  Both are results.  Clipping would turn a reportable outcome into a bound, and H3's
    first clause is precisely a statement about the sign.
    """
    assert tcv.rho(500.0, 400.0, 300.0) == -1.0      # worse than fixed-time
    assert tcv.rho(200.0, 400.0, 300.0) == 2.0       # better than MaxPressure


def test_t7_a_zero_denominator_is_refused_rather_than_reported() -> None:
    """Equal anchors make the normalisation undefined; any finite answer would be an invention."""
    with pytest.raises(ValueError, match="denominator is zero"):
        tcv.rho(350.0, 400.0, 400.0)


# ----------------------------------------------------------------------------------
# B1 -- the two declared stages
# ----------------------------------------------------------------------------------
def test_the_declared_cell_set_is_exactly_what_amendment_b1_declares() -> None:
    """B1's arithmetic, asserted as counts so a silent change to the design is visible.

    4 arms x 2 subjects x 5 seeds x 100 draws = 4,000 DT cells; ``fixedtime`` and ``maxpressure``
    100 each; ``random`` 5 policy seeds x 100 = 500. Total 4,700 -- the number the brief costs.
    """
    cells = tcv.declared_cells()
    assert len(cells) == 4700
    assert sum(1 for c in cells if c["kind"] == "dt") == 4000
    assert sum(1 for c in cells if c["arm"] == "random") == 500
    assert sum(1 for c in cells if c["arm"] == "fixedtime") == 100
    assert sum(1 for c in cells if c["arm"] == "maxpressure") == 100

    stage1 = tcv.declared_cells(tcv.STAGE_CONFIRMATORY)
    assert len(stage1) == 1200
    assert {c["arm"] for c in stage1} == {"b_mean_k100", "fixedtime", "maxpressure"}
    assert sum(1 for c in stage1 if c["kind"] == "dt") == 1000

    rest = tcv.declared_cells("rest")
    assert len(rest) == 3500
    assert {c["arm"] for c in rest} == {"b_max_k100", "a_q1.0", "naive", "random"}
    # The two stages PARTITION the campaign: nothing is dropped and nothing is run twice.
    assert len(stage1) + len(rest) == len(cells)
    names = [tcv.cell_chunk_name(c) for c in cells]
    assert len(set(names)) == len(names), "two cells share a chunk name and would overwrite"


def test_every_declared_draw_is_in_the_held_out_pool_and_none_is_a_training_draw() -> None:
    """A18(c) and PREREGISTRATION §5: the evaluation pool is 1000-1099 and nothing else."""
    from offline.materialise_draws import classify_draw_pool

    draws = {c["draw_id"] for c in tcv.declared_cells()}
    assert draws == set(range(1000, 1100))
    assert {classify_draw_pool(d) for d in draws} == {"held_out"}


def test_the_halting_cross_check_subset_is_declared_and_small() -> None:
    """Amendment C2: ON for every cell on draw 1000, OFF elsewhere -- 47 cells of 4,700.

    Value-neutral (A9b: identical ``att_env``, ``e_sumo`` and counts with it on and off) and 3.5x
    the cost, so it verifies the recorder on a declared subset rather than on every cell or none.
    """
    cells = tcv.declared_cells()
    checked = [c for c in cells if c["draw_id"] == tcv.HALTING_CHECK_DRAW]
    assert len(checked) == 47, "2 subjects x 4 arms x 5 seeds + fixedtime + maxpressure + 5 random"
    assert sum(1 for c in checked if c["kind"] == "dt") == 40


def test_e2_the_halting_check_draw_is_the_one_amendment_c2_declares_by_name() -> None:
    """Amendment E2, required in §3.5b's commit: the subset's IDENTITY, not just its size.

    Amendment **C2** declares the subset by name: *"the check is ON for every cell on draw 1000"*.
    The test above pins the subset's **size** (47 cells), and the coordinator's mutant
    ``HALTING_CHECK_DRAW = 1001`` SURVIVED it -- 1001 is also a held-out draw with 47 cells on it, so
    every count still matched while the declaration had silently moved. A17(d) permits a declared
    subset; it does not permit one that can drift without a test noticing. This is the one line that
    makes the declaration and the code the same statement.
    """
    assert tcv.HALTING_CHECK_DRAW == 1000
    assert tcv.halting_check_for(1000) is True
    assert tcv.halting_check_for(1001) is False


# ==================================================================================
# §3.5b -- shared helpers.  Every root below is a PARAMETER; nothing here writes to
# the MAIN tree, and the synthetic fixtures make the report tests independent of the
# gitignored artifacts (which is why they run in CI, where those do not exist).
# ==================================================================================
def _sumo_available() -> bool:
    try:
        import traci  # noqa: F401
    except ImportError:
        return False
    return shutil.which("sumo") is not None


def _draw_available(draw_id: int) -> bool:
    """Is P7.2a's parity configuration present for **this** draw (Amendment G2 of BRIEF_36)?"""
    from offline.materialise_draws import parity_sumocfg_path

    return parity_sumocfg_path("cityflow1x1", int(draw_id), out_root=DRAWS_ROOT).is_file()


def _checkpoint_available(subject: str, seed: int) -> bool:
    """Is **this** checkpoint on disk?  Named by path in the predicate, never a stand-in."""
    spec = tc.SUBJECTS[subject]
    return (OUTPUT_ROOT / spec["subdir"] / f"{spec['stem']}{seed}.pt").is_file()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cell(
    kind: str,
    arm: str,
    draw: int,
    *,
    subject: str | None = None,
    seed: int | None = None,
    stage: str = tcv.STAGE_CONFIRMATORY,
) -> dict[str, Any]:
    """One cell in exactly the shape :func:`declared_cells` emits."""
    return {
        "kind": kind,
        "subject": subject,
        "arm": arm,
        "seed": seed,
        "draw_id": int(draw),
        "stage": stage,
    }


class _Roots:
    """The four roots a synthetic campaign needs, plus the digests it was built with."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.draws = base / "draws"
        self.output = base / "output"
        self.data = base / "data"
        self.work = base / "work"
        self.out = base / "out"
        self.cfg_sha: dict[int, str] = {}
        self.routes_sha: dict[int, str] = {}
        self.checkpoint_sha: dict[tuple[str, int], str] = {}


def _build_roots(base: Path, *, draws: tuple[int, ...], seeds: tuple[int, ...]) -> _Roots:
    """Fabricate parity draws and committed checkpoint records under *base*.

    The bytes are arbitrary; what matters is that every digest the production code checks is
    derivable from the files themselves, so a test can break exactly one of them and watch the
    named refusal fire.  The calibration artifact is COPIED from the committed one, because its
    sha256 is pinned in the module and a fabricated one would only test the pin against itself.
    """
    roots = _Roots(base)
    for draw in draws:
        parity = roots.draws / "cityflow1x1" / f"draw_{draw:04d}" / "parity"
        parity.mkdir(parents=True, exist_ok=True)
        cfg = parity / "noteleport.sumocfg"
        routes = parity / "routes.rou.xml"
        cfg.write_bytes(f"<configuration draw={draw}/>".encode())
        routes.write_bytes(f"<routes draw={draw}/>".encode())
        roots.cfg_sha[draw] = _sha256(cfg.read_bytes())
        roots.routes_sha[draw] = _sha256(routes.read_bytes())
        (parity / "provenance.json").write_text(
            json.dumps(
                {
                    "draw_id": draw,
                    "files": {
                        "noteleport.sumocfg": roots.cfg_sha[draw],
                        "routes.rou.xml": roots.routes_sha[draw],
                    },
                    "pool": "held_out",
                }
            ),
            encoding="utf-8",
        )

    gate: dict[str, Any] = {"checkpoints": {}}
    runs: list[dict[str, Any]] = []
    for subject in ("mappo1000", "mix50"):
        spec = tc.SUBJECTS[subject]
        directory = roots.output / spec["subdir"]
        directory.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            path = directory / f"{spec['stem']}{seed}.pt"
            path.write_bytes(f"weights {subject} {seed}".encode())
            digest = _sha256(path.read_bytes())
            roots.checkpoint_sha[(subject, seed)] = digest
            if subject == "mappo1000":
                gate["checkpoints"][str(seed)] = {
                    "path": f"output/{spec['subdir']}/{spec['stem']}{seed}.pt",
                    "sha256": digest,
                }
            else:
                # Four methods per (tier, seed) in the real artifact -- the decoys are what make
                # the (tier, METHOD, seed) key necessary, so the fixture carries them too.
                for method in ("bc", "bc_top10", "iql", "dt"):
                    runs.append(
                        {
                            "tier": "mix50",
                            "method": method,
                            "seed": seed,
                            "file_sha256": digest if method == "dt" else _sha256(
                                f"{method} {seed}".encode()
                            ),
                        }
                    )

    roots.data.mkdir(parents=True, exist_ok=True)
    (roots.data / "p4_gate.json").write_text(json.dumps(gate), encoding="utf-8")
    (roots.data / "p4_7_training.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    shutil.copyfile(P7_2B, roots.data / "p7_2b_calibration.json")
    return roots


# ----------------------------------------------------------------------------------
# T6 (a) -- E3(a): the artifact LOADER, digest before parse
# ----------------------------------------------------------------------------------
def test_t6_the_calibration_artifact_is_loaded_only_at_its_pinned_digest() -> None:
    """E3(a). ``targets_for_subject`` takes a mapping, so §3.5a could not test this at all.

    The pin is a DECLARATION -- *these targets came from THAT artifact* -- and the file is the
    evidence. P7.2b's `report` pins `p4_3_probe.json` the same way (`transfer_calibration.py:1092`).
    """
    loaded = tcv.load_calibration()
    assert _sha256(P7_2B.read_bytes()) == tcv.P7_2B_CALIBRATION_SHA256
    assert sorted(loaded["targets"]) == ["mappo1000", "mix50"]
    # the loader and the §3.5a reader compose: the targets are the registered ones
    assert (
        tcv.targets_for_subject("mappo1000", loaded)["b_mean_k100"]["target_rtg"]
        == -7185.354721543778
    )


def test_t6_an_artifact_whose_digest_moved_is_refused(tmp_path: Path) -> None:
    """T6's second mutation, executable at last (E3(a)): *alter the artifact's sha -> refused*.

    One byte of whitespace is enough. A target read from an artifact nobody pinned is a target
    that can be edited between the calibration and the campaign without leaving a trace.
    """
    tampered = tmp_path / "p7_2b_calibration.json"
    tampered.write_bytes(P7_2B.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="sha256"):
        tcv.load_calibration(tampered)


# ----------------------------------------------------------------------------------
# T6 (b) -- G1: the checkpoint pins, against COMMITTED records
# ----------------------------------------------------------------------------------
def test_g1_checkpoint_identity_pins_both_subjects_against_committed_records(
    tmp_path: Path,
) -> None:
    """G1. Both subjects are pinned; the record compared is NAMED in the result.

    ``SHA256SUMS_p4_6.txt`` -- which ``BRIEF_37`` §3.5 and E3(b) named -- lists no ``p4_dt/`` path
    and is gitignored anyway, so the committed artifacts are the pins: ``p4_gate.json`` for
    ``mappo1000`` and ``p4_7_training.json``'s ``(tier, method, seed)`` row for ``mix50``.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))

    for subject, expected_record in (
        ("mappo1000", "p4_gate.json"),
        ("mix50", "p4_7_training.json"),
    ):
        identity = tcv.checkpoint_identity(
            subject, 101, output_root=roots.output, data_dir=roots.data
        )
        assert identity["file_sha256"] == roots.checkpoint_sha[(subject, 101)]
        assert expected_record in identity["sha256_checked_against"]
        assert Path(identity["path"]).is_file()


def test_g1_a_checkpoint_whose_digest_moved_is_refused(tmp_path: Path) -> None:
    """The pin's whole purpose: different weights under the same filename must not evaluate.

    *Mutation this is built against:* compare a truncated prefix of the digest instead of the whole
    string -- this one-byte change still collides on the first characters and would pass.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    spec = tc.SUBJECTS["mappo1000"]
    (roots.output / spec["subdir"] / f"{spec['stem']}101.pt").write_bytes(b"different weights")

    with pytest.raises(ValueError, match="sha256|digest"):
        tcv.checkpoint_identity("mappo1000", 101, output_root=roots.output, data_dir=roots.data)


def test_g1_a_truncated_digest_in_the_record_is_refused_not_treated_as_a_prefix(
    tmp_path: Path,
) -> None:
    """The comparison is on the WHOLE digest, and this is how that is shown without a collision.

    A mutant comparing ``digest[:8] != committed[:8]`` survived the test above: the two digests
    there differ from their first character, so a prefix comparison refuses just as the full one
    does, and no practical test input separates them -- finding a 32-bit prefix collision is about
    4 billion hashes.  Turning the mutation round removes the need: a committed record holding a
    TRUNCATED digest is cheap to build, it is exactly what a malformed record looks like, and the
    prefix mutant accepts it while the delivered code refuses.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    gate = json.loads((roots.data / "p4_gate.json").read_text(encoding="utf-8"))
    full = gate["checkpoints"]["101"]["sha256"]
    gate["checkpoints"]["101"]["sha256"] = full[:8]
    (roots.data / "p4_gate.json").write_text(json.dumps(gate), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256|digest"):
        tcv.checkpoint_identity("mappo1000", 101, output_root=roots.output, data_dir=roots.data)


def test_g1_the_mix50_row_is_found_by_tier_method_and_seed_not_by_tier_and_seed(
    tmp_path: Path,
) -> None:
    """The correction to G1's lookup key, made from the artifact rather than from its prose.

    ``docs/data/p4_7_training.json`` carries **20** ``mix50`` rows -- ``bc``, ``bc_top10``, ``iql``
    and ``dt`` across five seeds -- so ``(tier, seed)`` matches FOUR rows, three of them other
    methods' checkpoints. If the lookup ignored ``method`` it would either pick an arbitrary row or
    pin the DT checkpoint against a BC checkpoint's digest. Exactly one match is required.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    training = json.loads((roots.data / "p4_7_training.json").read_text(encoding="utf-8"))
    assert sum(1 for row in training["runs"] if row["tier"] == "mix50" and row["seed"] == 101) == 4

    identity = tcv.checkpoint_identity("mix50", 101, output_root=roots.output, data_dir=roots.data)
    assert identity["file_sha256"] == roots.checkpoint_sha[("mix50", 101)]

    # two rows for the same (tier, method, seed) is an ambiguity, not a preference
    training["runs"].append({"tier": "mix50", "method": "dt", "seed": 101, "file_sha256": "x" * 64})
    (roots.data / "p4_7_training.json").write_text(json.dumps(training), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one|2 rows"):
        tcv.checkpoint_identity("mix50", 101, output_root=roots.output, data_dir=roots.data)


@pytest.mark.skipif(
    not _checkpoint_available("mappo1000", 101) or not _checkpoint_available("mix50", 101),
    reason="needs output/p4_dt/dt_seed101.pt and output/p4_7/checkpoints/mix50_dt_seed101.pt",
)
def test_g1_the_real_checkpoints_match_their_committed_records_today() -> None:
    """The pins, exercised against the REAL files and the REAL committed artifacts.

    The fabricated fixtures above prove the logic; this proves the ten digests actually agree
    today, which is the claim the campaign rests on. It is the check ``DEFERRED`` 56 says nobody
    could make for ``mappo1000`` from a ``SHA256SUMS_*`` manifest.
    """
    for subject in ("mappo1000", "mix50"):
        for seed in tcv.TRAINING_SEEDS:
            identity = tcv.checkpoint_identity(subject, seed, output_root=OUTPUT_ROOT)
            assert identity["file_sha256"] == _sha256(Path(identity["path"]).read_bytes())


# ----------------------------------------------------------------------------------
# G2 -- the demand identity is TWO digests
# ----------------------------------------------------------------------------------
def test_g2_the_demand_identity_covers_the_routes_file_as_well_as_the_cfg(
    tmp_path: Path,
) -> None:
    """G2. A cfg digest alone does not pin the demand: the cfg only NAMES the routes file.

    *Mutation this is built against:* drop ``routes_sha256`` from the comparison. A regenerated
    routes file leaves the cfg's bytes untouched, so the cfg digest still matches and three cells
    that rho pairs could have run different demand.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    identity = tcv.demand_identity(1000, out_root=roots.draws)
    assert identity["config_sha256"] == roots.cfg_sha[1000]
    assert identity["routes_sha256"] == roots.routes_sha[1000]

    routes = roots.draws / "cityflow1x1" / "draw_1000" / "parity" / "routes.rou.xml"
    routes.write_bytes(b"<routes regenerated/>")
    with pytest.raises(ValueError, match="routes"):
        tcv.demand_identity(1000, out_root=roots.draws)


@pytest.mark.skipif(not _draw_available(1000), reason="needs scenarios/draws/.../draw_1000/parity")
def test_g2_the_real_draw_1000_matches_its_own_provenance() -> None:
    """The same check against P7.2a's real output for the pool's first draw."""
    identity = tcv.demand_identity(1000, out_root=DRAWS_ROOT)
    assert identity["config_sha256"].startswith("c177e962")
    assert len(identity["routes_sha256"]) == 64


# ----------------------------------------------------------------------------------
# A2 / C8 / T7b -- the door is for the DT, and the cell builder REFUSES otherwise
# ----------------------------------------------------------------------------------
def _fake_aligned_env() -> Any:
    """An ``AlignedEnv`` instance without a simulator.

    ``AlignedEnv.__init__`` imports ``envs.sumo_env`` (which pulls traci) to type-check its
    argument, but the class object itself imports nothing.  ``__new__`` therefore gives a genuine
    instance for an ``isinstance`` check -- which is exactly what the refusal under test does --
    and keeps this test running where SUMO is absent.
    """
    from offline.aligned_env import AlignedEnv

    return AlignedEnv.__new__(AlignedEnv)


def test_c8_an_anchor_cell_built_through_the_door_is_refused() -> None:
    """Amendment C8's second half, at the cell builder. A2's rule, enforced by arithmetic.

    ``align_info`` drops outgoing lanes and re-keys the survivors to CityFlow ids, while
    MaxPressure's pressure is a difference over the env's OWN SUMO lane ids -- wrapping an anchor
    raises ``KeyError: 'road_1_1_2_0'`` in ``algorithms/max_pressure.py:142``.  That was measured
    before it was written down (Amendment A2), and this refusal is what stops a future edit from
    quietly routing the anchors through the door and reporting whatever came out.
    """
    with pytest.raises(TypeError, match="anchor|unwrapped"):
        tcv.assert_env_matches_cell(_cell("anchor", "maxpressure", 1000), _fake_aligned_env())


def test_c8_a_dt_cell_built_without_the_door_is_refused() -> None:
    """The other direction, which matters just as much: A16's door is the ONLY route into the frame.

    A DT cell on a raw SUMO env sees 32-wide info in SUMO lane order -- a CityFlow-trained model
    would consume it happily and produce a number that means nothing.
    """

    class _NotAligned:
        pass

    with pytest.raises(TypeError, match="aligned|door"):
        tcv.assert_env_matches_cell(
            _cell("dt", "b_mean_k100", 1000, subject="mappo1000", seed=101), _NotAligned()
        )


def test_a2_the_env_factory_uses_the_door_for_dt_cells_and_not_for_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A2/G3: one construction path, two shapes, chosen by the cell's ``kind``.

    Spied at the ``aligned_env`` seam, so no simulator is needed and the assertion is about which
    constructor was called -- the thing that decides whether the info goes through ``align_info``.
    """
    import offline.aligned_env as ae

    calls: list[tuple[str, dict[str, Any]]] = []

    def _spy(name: str) -> Any:
        def factory(scenario_key: str, draw_id: int, **kwargs: Any) -> Any:
            calls.append((name, {"scenario_key": scenario_key, "draw_id": draw_id, **kwargs}))
            return object()

        return factory

    monkeypatch.setattr(ae, "aligned_observer_env_for_draw", _spy("aligned"), raising=True)
    monkeypatch.setattr(ae, "observer_env_for_draw", _spy("unwrapped"), raising=True)

    tcv.env_for_cell(_cell("dt", "b_mean_k100", 1000, subject="mix50", seed=101), out_root="/x")
    tcv.env_for_cell(_cell("anchor", "maxpressure", 1000), out_root="/x")

    assert [name for name, _ in calls] == ["aligned", "unwrapped"]
    assert all(kwargs["draw_id"] == 1000 for _, kwargs in calls)


def test_c2_the_halting_cross_check_is_requested_only_on_the_declared_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2, at the only place it can be set: construction time.

    ``AlignedEnv`` forwards attribute READS through ``__getattr__`` and defines no ``__setattr__``,
    so P7.1's ``env.halting_check = ...`` (``sumo_att_reference.py:1292``) would set the flag on the
    wrapper and leave the observer untouched.  The subset is therefore chosen when the env is built.

    *Mutation this is built against:* ``halting_check=True`` unconditionally -- 3.5x the cost on
    every one of 4,700 cells, and a declared subset that is no longer a subset.
    """
    import offline.aligned_env as ae

    seen: list[tuple[int, bool]] = []

    def factory(scenario_key: str, draw_id: int, **kwargs: Any) -> Any:
        seen.append((int(draw_id), bool(kwargs["halting_check"])))
        return object()

    monkeypatch.setattr(ae, "aligned_observer_env_for_draw", factory, raising=True)
    monkeypatch.setattr(ae, "observer_env_for_draw", factory, raising=True)

    for draw in (1000, 1001):
        for cell in (
            _cell("dt", "b_mean_k100", draw, subject="mappo1000", seed=101),
            _cell("anchor", "fixedtime", draw),
        ):
            tcv.env_for_cell(cell, out_root="/x")

    assert seen == [(1000, True), (1000, True), (1001, False), (1001, False)]


# ----------------------------------------------------------------------------------
# BRIEF_36 E4 -- every DT decision is act(info, explore=False, update_memory=True)
# ----------------------------------------------------------------------------------
def test_e4_every_dt_decision_passes_explore_false_and_update_memory_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``BRIEF_36`` E4, pinned by a spy over EVERY call rather than the first.

    ``explore=True`` samples from the masked softmax through an unseeded ``torch.multinomial``; the
    same seed, checkpoint and draw then gave ``n_decisions_in_support`` 271 and then 231. A loop
    that got the keywords right once and then fell back to a default would pass a first-call-only
    assertion, so the spy records all of them.
    """
    import numpy as np

    calls: list[dict[str, Any]] = []

    class _Agent:
        def current_rtg(self) -> dict[str, float]:
            return {IX: -7185.354721543778 - len(calls)}

        def act(self, info: Mapping[str, Any], **kwargs: Any) -> np.ndarray:
            calls.append(dict(kwargs))
            return np.zeros(1, dtype=np.int64)

    class _Env:
        intersections = [type("Ix", (), {"id": IX})()]

    monkeypatch.setattr(
        "offline.rtg_calibration.agent_with_target", lambda *a, **k: _Agent(), raising=True
    )

    choose, diagnostics = tcv.dt_choose(
        _Env(), checkpoint_path="/nonexistent.pt", target_rtg=-7185.354721543778
    )
    for step in range(3):
        choose(_Env(), {"intersections": {IX: {"reward": -float(step)}}})

    assert calls, "the spy saw no act() call at all, so it asserted nothing"
    assert all(call == {"explore": False, "update_memory": True} for call in calls), calls
    assert diagnostics["rtg_series"][0] == -7185.354721543778
    assert diagnostics["reward_series"] == [0.0, -1.0, -2.0]


def test_the_module_contains_no_bare_dt_act_call() -> None:
    """E4's source-level half, by AST rather than by grep.

    The defect this guards is a DEFAULT left in place, so what must be asserted is the absence of a
    call that omits the keywords -- and a text search would trip over the docstrings that quote
    ``agent.act(info)`` while explaining exactly this.  Anchor policies are called on a receiver
    named ``policy`` and take no keywords (``MaxPressureAgent.act``).
    """
    import ast

    source = Path(tcv.__file__).read_text(encoding="utf-8")
    offenders: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "act" or not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id == "policy":
            assert not {kw.arg for kw in node.keywords}, "an anchor policy takes no keywords"
        elif {kw.arg for kw in node.keywords} != {"explore", "update_memory"}:
            offenders.append(node.lineno)

    assert offenders == [], f"bare or partial agent.act(...) at line(s) {offenders}"


# ==================================================================================
# The chunk: one cell, one file.  E3(d) and the refusals E3(e) re-runs on every one.
# ==================================================================================
#: The synthetic campaign: two draws (1000 is C2's halting draw, 1001 is not), one seed, one arm,
#: plus the two anchors rho is defined against.  Declared HERE, independently of any chunk on disk,
#: which is what stops the completeness check from being a tautology (PROJECT_PLAN section 7).
_DEMO_DRAWS = (1000, 1001)

#: TWO seeds, deliberately: the per-draw unit is the mean over training seeds (``dt_gate``'s
#: ``_per_draw_means``, "as in P4"), and with one seed per cell that mean is the identity -- a
#: version of the code that reported the FIRST seed instead of the mean passed every assertion in
#: this file until the fixture carried two.  Found by mutation M17, not by reading.
_DEMO_SEEDS = (101, 202)
_DEMO_CELLS = [
    _cell("dt", "b_mean_k100", draw, subject=subject, seed=seed)
    for subject in ("mappo1000", "mix50")
    for seed in _DEMO_SEEDS
    for draw in _DEMO_DRAWS
] + [_cell("anchor", arm, draw) for arm in ("fixedtime", "maxpressure") for draw in _DEMO_DRAWS]

#: Per draw: (fixedtime ATT, maxpressure ATT, the DT's ATT at seed 101).  The denominator is 256 and
#: every offset is a multiple of 64, so every rho below is an exact binary fraction and the
#: assertions can use ``==``: a rounding change is then a failure rather than noise.
_DEMO_ATT = {1000: (400.0, 144.0, 272.0), 1001: (400.0, 144.0, 208.0)}

#: Seed 202's cells run 64.0 slower than seed 101's, so the two seeds give DIFFERENT rho values and
#: the seed mean is distinguishable from either of them.
_DEMO_SEED_OFFSET = {101: 0.0, 202: 64.0}


def _payload(cell: Mapping[str, Any], roots: _Roots, **overrides: Any) -> dict[str, Any]:
    """A complete, valid chunk for *cell*, from which a test perturbs exactly one field."""
    draw = int(cell["draw_id"])
    fixed, maxp, dt_att = _DEMO_ATT[draw]
    if cell["seed"] is not None and cell["kind"] == "dt":
        dt_att += _DEMO_SEED_OFFSET[int(cell["seed"])]
    att = {"fixedtime": fixed, "maxpressure": maxp}.get(str(cell["arm"]), dt_att)
    checked = draw == tcv.HALTING_CHECK_DRAW
    is_dt = cell["kind"] == "dt"

    payload: dict[str, Any] = {
        "format_version": tcv.ARTIFACT_FORMAT_VERSION,
        **dict(cell),
        "policy_seed": None,
        "engine_seed_requested": 1000,
        "engine_seed_drawn": 437485271,
        "decisions": 360,
        "actions_in_range": True,
        "episode_reward": -23938.0,
        "att_horizon": att,
        "att_env": att,
        "att_running_mean": att - 1.0,
        "horizon_vehicle_count": 12.0,
        # e_sumo runs 20.0 above att_env on every cell, so rho is the same under both definitions
        # and a test that swapped them would still have to break the VALUE to fail.
        "e_sumo": att + 20.0,
        "p_sumo": att + 5.0,
        "w_sumo": att + 1.0,
        "mean_depart_delay": 4.0,
        "n_created": 1821,
        "n_entered": 1800,
        "n_never_entered": 21,
        "n_pending_at_horizon": 5,
        "n_teleports": 0,
        "n_vanished_without_arrival": 0,
        "n_arrived_never_observed_at_a_boundary": 0,
        "max_abs_depart_clock_deviation": 0.0,
        "n_observations": 3601,
        "vehicle_types_seen": ["cf_parity"],
        "time_to_teleport_option": "-1",
        "halting_checked": checked,
        "halting_max_abs_difference": 0 if checked else None,
        "halting_n_lane_seconds": 28808 if checked else None,
        "halting_n_disagreeing_lane_seconds": 0 if checked else None,
        "config_sha256": roots.cfg_sha[draw],
        "routes_sha256": roots.routes_sha[draw],
        "calibration_sha256": tcv.P7_2B_CALIBRATION_SHA256,
        "canary_seconds": 0.87,
        "seconds": 37.06,
        "git_commit": HEAD_SHA,
        "git_dirty": False,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "sha256_checked_against": None,
        "target_rtg": None,
        "rtg_first": None,
        "rtg_last": None,
        "rtg_series": None,
        "reward_series": None,
        "rtg_advanced_every_decision": None,
        "n_decisions_in_support": None,
        "support_range": None,
    }
    if is_dt and str(cell["subject"]) == tcv.ANCHOR_SUBJECT:
        # P7.3b's anchor. A SEPARATE branch rather than an entry in tc.SUBJECTS, mirroring the
        # module: `SUBJECTS` means A17(c)'s two CityFlow-trained subjects, and `_in_support_block`
        # would raise on anything P7.2b's artifact never registered.
        seed = int(cell["seed"])
        target = -20625.0                      # the real 200-episode naive in-domain target
        payload.update(
            {
                "checkpoint": str(
                    roots.output
                    / tcv.ANCHOR_CHECKPOINT_SUBDIR
                    / f"{tcv.ANCHOR_CHECKPOINT_STEM}{seed}.pt"
                ),
                "checkpoint_sha256": roots.checkpoint_sha.get(
                    (tcv.ANCHOR_SUBJECT, seed), _sha256(f"anchor {seed}".encode())
                ),
                "sha256_checked_against": [tcv.P7_3B_TRAINING_NAME],
                "anchor_training_sha256": tcv.P7_3B_TRAINING_SHA256,
                "target_rtg": target,
                "rtg_first": target,
                "rtg_last": target + 100.0,
                "rtg_series": [target, target, target + 100.0],
                "reward_series": [0.0, 100.0, 0.0],
                "rtg_advanced_every_decision": True,
                "n_decisions_in_support": 271,
                "support_range": [-25257.0, -71.0],
            }
        )
    elif is_dt:
        subject, seed = str(cell["subject"]), int(cell["seed"])
        spec = tc.SUBJECTS[subject]
        target = -7185.354721543778 if subject == "mappo1000" else -7431.0185327454665
        payload.update(
            {
                "checkpoint": str(roots.output / spec["subdir"] / f"{spec['stem']}{seed}.pt"),
                "checkpoint_sha256": roots.checkpoint_sha[(subject, seed)],
                "sha256_checked_against": ["p4_gate.json"]
                if subject == "mappo1000"
                else ["p4_7_training.json"],
                "target_rtg": target,
                "rtg_first": target,
                "rtg_last": target + 100.0,
                "rtg_series": [target, target, target + 100.0],
                "reward_series": [0.0, 100.0, 0.0],
                "rtg_advanced_every_decision": True,
                "n_decisions_in_support": 271,
                "support_range": [-9991.0, -6.0],
            }
        )
    payload.update(overrides)
    return payload


def _canary(work: Path) -> None:
    """The run's canary record, as the driver parks it right after the token (E1.4)."""
    work.mkdir(parents=True, exist_ok=True)
    (work / tc.CANARY_RECORD_NAME).write_text(
        json.dumps(
            {
                "seconds": 0.87,
                "facts": {
                    "decisions": tc.CANARY_REFERENCE_DECISIONS,
                    "local_return": tc.CANARY_REFERENCE_LOCAL_RETURN,
                    "att_horizon": tc.CANARY_REFERENCE_ATT_HORIZON,
                    "two_routes_agree": True,
                },
                "git_commit": HEAD_SHA,
                "git_dirty": False,
            }
        ),
        encoding="utf-8",
    )


def _campaign(tmp_path: Path, **perturb: Any) -> tuple[_Roots, list[dict[str, Any]]]:
    """Roots, a canary and one chunk per declared cell; *perturb* edits the LAST chunk written."""
    roots = _build_roots(tmp_path, draws=_DEMO_DRAWS, seeds=_DEMO_SEEDS)
    roots.work.mkdir(parents=True, exist_ok=True)
    roots.out.mkdir(parents=True, exist_ok=True)
    _canary(roots.work)
    for cell in _DEMO_CELLS:
        tcv.write_chunk(_payload(cell, roots), work_dir=roots.work)
    return roots, _DEMO_CELLS


def _report(roots: _Roots, **kwargs: Any) -> dict[str, Any]:
    """:func:`report` over the synthetic campaign, with every root pointed at the fixture."""
    defaults: dict[str, Any] = {
        "work_dir": roots.work,
        "out_path": roots.out / "p7_3a_zero_shot.json",
        "output_root": roots.output,
        "out_root": roots.draws,
        "data_dir": roots.data,
        "cells": _DEMO_CELLS,
    }
    defaults.update(kwargs)
    return tcv.report(**defaults)


def test_a_chunk_is_written_atomically_and_names_the_whole_cell(tmp_path: Path) -> None:
    """E3(d). One cell, one file, and the name carries the identity that the content repeats."""
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    cell = _cell("dt", "b_mean_k100", 1000, subject="mix50", seed=101)

    path = tcv.write_chunk(_payload(cell, roots), work_dir=roots.work)

    assert path.name == "cell_mix50_b_mean_k100_seed101_draw1000.json"
    assert json.loads(path.read_bytes())["arm"] == "b_mean_k100"
    assert list(roots.work.glob("*.tmp")) == [], "a .tmp left behind is a half-written chunk"


def test_a_chunk_is_reusable_only_when_its_digests_still_hold_on_disk(tmp_path: Path) -> None:
    """E3(d)/G2: resumable **by content**, and the content is re-derived, never read.

    ``chunk_is_reusable`` recomputes the checkpoint's, the cfg's and the routes file's sha256 from
    the files themselves and compares them with what the chunk recorded.  A stored verdict is
    exactly what a half-written or hand-edited chunk would lie about
    (``nortg_decomposition.py:469``: the cell-identity check is the pre-flight's M1).
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    cell = _cell("dt", "b_mean_k100", 1000, subject="mappo1000", seed=101)
    keys = {"cell": cell, "out_root": roots.draws, "output_root": roots.output, "data_dir": roots.data}

    assert tcv.chunk_is_reusable(_payload(cell, roots), **keys) is True

    # (a) another cell's chunk under this cell's name -- the M1 defect, by content not filename
    other = _cell("dt", "b_mean_k100", 1000, subject="mix50", seed=101)
    assert tcv.chunk_is_reusable(_payload(other, roots), **keys) is False

    # (b) the demand moved under it: the routes file was regenerated after the cell ran
    (roots.draws / "cityflow1x1" / "draw_1000" / "parity" / "routes.rou.xml").write_bytes(b"new")
    assert tcv.chunk_is_reusable(_payload(cell, roots), **keys) is False


def test_an_unreadable_chunk_is_re_run_rather_than_crashing_the_driver(tmp_path: Path) -> None:
    """A file we cannot read is a file we have no evidence about (``nortg_decomposition.py:523``).

    ``json.loads`` accepts ``[]``, ``5`` and ``"text"`` as happily as an object, and a truncated
    chunk raised ``JSONDecodeError`` in P5.3b -- the driver wrote FAILED and every restart hit the
    same crash, while the docstring promised a partial chunk would be re-run.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    cell = _cell("anchor", "fixedtime", 1000)
    keys = {"cell": cell, "out_root": roots.draws, "output_root": roots.output, "data_dir": roots.data}
    roots.work.mkdir(parents=True, exist_ok=True)

    truncated = roots.work / "truncated.json"
    truncated.write_text('{"format_version": "p7.3a', encoding="utf-8")
    assert tcv.reusable_chunk_at(truncated, **keys) is False
    assert tcv.reusable_chunk_at(roots.work / "absent.json", **keys) is False
    for payload in ([], 5, "text"):
        assert tcv.chunk_is_reusable(payload, **keys) is False  # type: ignore[arg-type]


# ==================================================================================
# T8 / E3(e) / B2 -- report
# ==================================================================================
def test_t8_report_builds_the_artifact_with_rho_under_both_definitions(tmp_path: Path) -> None:
    """E3(e). rho per cell under BOTH definitions, paired per draw, anchors 0 and 1 by construction.

    ``PREREGISTRATION`` §3.4 fixes one formula; A15(a)/(b) require it on ``e_sumo`` (the pool-clock
    population, the primary) and on ``att_env`` (the admitted pair) -- that, and not two formulas, is
    what *"both definitions"* means (Amendment A1).  The fixture puts ``e_sumo`` a constant 20.0
    above ``att_env`` so the two rho values coincide by construction and any confusion of the two
    shows up as a VALUE change rather than as a plausible near-miss.
    """
    roots, _ = _campaign(tmp_path)

    artifact = _report(roots)

    assert artifact["format_version"] == tcv.ARTIFACT_FORMAT_VERSION
    assert len(artifact["cells"]) == len(_DEMO_CELLS)
    by_draw = {
        (row["arm"], row["draw_id"]): row for row in artifact["cells"] if row["kind"] == "anchor"
    }
    for draw in _DEMO_DRAWS:
        assert by_draw[("fixedtime", draw)]["rho_e_sumo"] == 0.0
        assert by_draw[("fixedtime", draw)]["rho_att_env"] == 0.0
        assert by_draw[("maxpressure", draw)]["rho_e_sumo"] == 1.0
        assert by_draw[("maxpressure", draw)]["rho_att_env"] == 1.0

    dt_rows = {
        (r["subject"], r["seed"], r["draw_id"]): r
        for r in artifact["cells"]
        if r["kind"] == "dt"
    }
    assert dt_rows[("mappo1000", 101, 1000)]["rho_att_env"] == 0.5
    assert dt_rows[("mappo1000", 202, 1000)]["rho_att_env"] == 0.25
    assert dt_rows[("mix50", 101, 1001)]["rho_att_env"] == 0.75
    # ...and by the independent route, straight from the registered formula on the stored ATTs
    assert dt_rows[("mix50", 202, 1001)]["rho_e_sumo"] == tcv.rho(
        208.0 + 64.0 + 20.0, 400.0 + 20.0, 144.0 + 20.0
    )

    # THE PAIRED UNIT: the per-draw value is the mean over seeds, and the arm's mean is the mean of
    # those.  Recomputed here from the published per-cell rows by a route that does not touch the
    # aggregation code -- 0.5 exactly, and neither seed alone gives 0.5.
    per_draw = {
        draw: (
            dt_rows[("mappo1000", 101, draw)]["rho_e_sumo"]
            + dt_rows[("mappo1000", 202, draw)]["rho_e_sumo"]
        )
        / 2.0
        for draw in _DEMO_DRAWS
    }
    assert per_draw == {1000: 0.375, 1001: 0.625}
    arm_block = [
        e for e in artifact["rho"]["by_subject_arm"]
        if e["subject"] == "mappo1000" and e["arm"] == "b_mean_k100"
    ][0]
    assert arm_block["e_sumo"]["mean"] == 0.5
    assert arm_block["e_sumo"]["n_draws"] == 2


def test_t8_report_states_h3_as_inequalities_and_interprets_nothing(tmp_path: Path) -> None:
    """E3(e): *H3's two inequalities reported, not interpreted*; the contrast EXPLORATORY.

    H3's first two clauses are *"better than fixed-time"* (rho > 0) and *"worse than within-backend
    MaxPressure"* (rho < 1). The artifact reports the comparison and its value; it contains no
    verdict sentence, because the registered claim is the inequality and a word like "confirmed"
    would be this task deciding something A17 gave it no licence to decide.
    """
    roots, _ = _campaign(tmp_path)

    artifact = _report(roots)

    h3 = artifact["h3"]
    assert h3["registered_arm"] == "b_mean_k100"
    # 2 inequalities x 2 ATT definitions; without this the loop below could assert nothing at all
    assert len(h3["clauses"]) == 4
    for clause in h3["clauses"]:
        assert clause["inequality"] in ("rho_sumo(b_mean_k100) > 0", "rho_sumo(b_mean_k100) < 1")
        # Amendment H7 replaced `holds` -- a boolean on the point estimate that would be read as
        # "confirmed" -- with the two mechanical facts asserted in
        # test_h7_h3_reports_the_point_estimate_and_the_interval_separately.
        assert set(clause["point_estimate_satisfies"]) <= {"mappo1000", "mix50"}
    assert artifact["contrast"]["status"] == "exploratory"
    assert "registered_direction" in artifact["contrast"]
    assert artifact["rho"]["estimator"]["resampling_seed"] is None
    assert "analytic" in artifact["rho"]["estimator"]["method"]
    assert "what_this_does_not_say" in artifact


def test_t8_report_refuses_a_missing_declared_cell_and_writes_nothing(tmp_path: Path) -> None:
    """The completeness check, against a declaration that exists before any chunk does."""
    roots, _ = _campaign(tmp_path)
    tcv.chunk_path(_DEMO_CELLS[0], work_dir=roots.work).unlink()

    with pytest.raises(ValueError, match="missing|declared"):
        _report(roots)
    assert list(roots.out.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("n_teleports", 3, "teleport"),
        ("vehicle_types_seen", ["cf_parity", "DEFAULT_VEHTYPE"], "cf_parity"),
        ("time_to_teleport_option", "300", "teleport"),
        ("actions_in_range", False, "action"),
        ("decisions", 359, "decisions"),
        ("halting_n_disagreeing_lane_seconds", 7, "halting"),
        ("canary_seconds", None, "canary"),
        ("config_sha256", "f" * 64, "sha256|config"),
        ("checkpoint_sha256", "f" * 64, "sha256|checkpoint"),
        ("calibration_sha256", "f" * 64, "sha256|calibration"),
    ],
)
def test_t8_report_refuses_one_broken_cell_and_writes_nothing(
    tmp_path: Path, field: str, value: Any, match: str
) -> None:
    """Each regime and identity property, broken one at a time, on a DT cell of draw 1000.

    Every one of these is a property the campaign asserts per cell; ``report`` re-asserts all of
    them rather than trusting that the runner did, because a chunk can be hand-made and a resumed
    campaign can carry chunks from an older code path.  A refusal writes nothing at all.
    """
    roots, _ = _campaign(tmp_path)
    cell = _cell("dt", "b_mean_k100", 1000, subject="mappo1000", seed=101)
    tcv.write_chunk(_payload(cell, roots, **{field: value}), work_dir=roots.work)

    with pytest.raises(ValueError, match=match):
        _report(roots)
    assert list(roots.out.iterdir()) == []


def test_t8_report_refuses_a_work_directory_without_a_canary_record(tmp_path: Path) -> None:
    """Amendment E1.4: a work directory without ``canary.json`` is not a run.

    The previous behaviour -- fall back on the chunks' own ``canary_seconds`` -- is precisely how
    P7.2b's run 3 published run 1's canary as its own.
    """
    roots, _ = _campaign(tmp_path)
    (roots.work / tc.CANARY_RECORD_NAME).unlink()

    with pytest.raises(FileNotFoundError, match="canary"):
        _report(roots)
    assert list(roots.out.iterdir()) == []


def test_t8_report_refuses_a_canary_whose_correctness_half_fails(tmp_path: Path) -> None:
    """E1.2's half: the record is a CLAIM, and ``report`` re-runs ``check_canary`` on its facts.

    A canary that only times would have passed while the engine returned different numbers.
    """
    roots, _ = _campaign(tmp_path)
    record = json.loads((roots.work / tc.CANARY_RECORD_NAME).read_text(encoding="utf-8"))
    record["facts"]["local_return"] = -32647.0
    (roots.work / tc.CANARY_RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="canary"):
        _report(roots)
    assert list(roots.out.iterdir()) == []


def test_t8_report_refuses_when_a_declared_anchor_chunk_is_missing(tmp_path: Path) -> None:
    """The COMPLETENESS check, and this test is named for what it actually reaches.

    ⚠️ It was called *"refuses a cell whose anchors come from another draw"* and read as the
    pairing guard's test; the reviewer showed it is caught one step higher, by completeness, and
    passes with the pairing guard removed (Amendment H4, H9).  The pairing guard's own test is
    :func:`test_h4_rho_is_refused_when_only_one_draw_has_anchors`, which shares a draw so that
    nothing above it fires.
    """
    roots, _ = _campaign(tmp_path)
    for arm in ("fixedtime", "maxpressure"):
        tcv.chunk_path(_cell("anchor", arm, 1001), work_dir=roots.work).unlink()

    with pytest.raises(ValueError, match="declared cell"):
        _report(roots)
    assert list(roots.out.iterdir()) == []


def test_h4_rho_is_refused_when_only_one_draw_has_anchors(tmp_path: Path) -> None:
    """H4. The per-draw pairing guard, reached where nothing above it can fire first.

    ⚠️ **The reviewer's probe, made a test.**  The previous fixture declared a DT cell on draw 1000
    and anchors on 1001 only -- ZERO shared draws -- so with the guard removed the refusal came
    from ``paired_comparison`` ("no shared draws ... void") and the test passed for a reason that
    has nothing to do with pairing.  This fixture puts DT cells on BOTH draws and anchors on 1001
    only: one shared draw, so ``paired_comparison`` is content, and with the guard removed
    ``report`` **writes an artifact in which draw 1000's rho is normalised by draw 1001's
    anchors** -- a plausible number built from another draw's demand.

    The match is on the guard's OWN words, so a refusal from anywhere else cannot satisfy it.
    """
    roots = _build_roots(tmp_path, draws=_DEMO_DRAWS, seeds=_DEMO_SEEDS)
    roots.work.mkdir(parents=True, exist_ok=True)
    roots.out.mkdir(parents=True, exist_ok=True)
    _canary(roots.work)

    declared = [
        _cell("dt", "b_mean_k100", 1000, subject="mappo1000", seed=101),
        _cell("dt", "b_mean_k100", 1001, subject="mappo1000", seed=101),
        _cell("anchor", "fixedtime", 1001),
        _cell("anchor", "maxpressure", 1001),
    ]
    for cell in declared:
        tcv.write_chunk(_payload(cell, roots), work_dir=roots.work)

    with pytest.raises(ValueError, match="no denominator of its own draw"):
        _report(roots, cells=declared)
    assert list(roots.out.iterdir()) == []


def test_t8_report_refuses_an_undeclared_arm(tmp_path: Path) -> None:
    """The fence: ``BRIEF_37`` §2 lifted it for the four declared arms and the three anchors ONLY.

    Rule B at k = 5 and k = 20 are the few-shot prompts A18(d) attaches to fine-tuned models. They
    were excluded BEFORE any number existed (A17(d)); an outcome for one of them appearing in the
    artifact would be a zero-shot evaluation nobody registered.
    """
    roots, _ = _campaign(tmp_path)
    undeclared = _cell("dt", "b_mean_k5", 1000, subject="mappo1000", seed=101)
    tcv.write_chunk(_payload(undeclared, roots), work_dir=roots.work)

    with pytest.raises(ValueError, match="b_mean_k5|declared"):
        _report(roots)
    assert list(roots.out.iterdir()) == []


def test_t8_a_test_reaches_the_LAST_refusal_before_the_write(tmp_path: Path) -> None:
    """Amendment F1's lesson, applied here before it can cost anything.

    ``report`` ends with a scan over the SERIALISED bytes and only then calls ``_write_json``. Every
    other refusal in this file trips above artifact assembly, so without this test the scan would
    never execute with delivered code -- and a reviewer who moved ``_write_json`` one line up would
    keep the suite green.  In P7.2b that exact reorder survived a merge review.

    No patching: a *published field's VALUE* carries P7.2b's fenced marker, which passes every
    key-shaped check above and is caught only because the guard reads the bytes that are about to be
    written.

    ⚠️ **The carrier moved from ``git_commit`` to ``checkpoint``** when Amendment J1(b) began
    refusing a non-40-hex commit: that refusal sits ABOVE the fence, so the old carrier stopped
    reaching it and this test would have passed on the wrong refusal.  ``checkpoint`` is published,
    is free text, and is not validated for content -- ``report`` pins the checkpoint's DIGEST, not
    the path it was read from -- so the route still reaches the serialised bytes with delivered
    code and no patching.
    """
    roots, _ = _campaign(tmp_path)
    cell = _cell("dt", "b_mean_k100", 1000, subject="mappo1000", seed=101)
    tcv.write_chunk(
        _payload(cell, roots, checkpoint=f"/weights/{tc.FENCED_KEY}.pt"), work_dir=roots.work
    )

    with pytest.raises(AssertionError, match="reached the artifact"):
        _report(roots)
    assert list(roots.out.iterdir()) == [], (
        "the fence refusal must precede every write, including a .tmp from a half-done _write_json"
    )


def test_b2_the_stage_one_artifact_and_the_final_one_agree_cell_for_cell(tmp_path: Path) -> None:
    """Amendment B2. The confirmatory number exists early and does not move afterwards.

    Stage 2 is UNCONDITIONAL (B1), so the two artifacts are a sequence and not a cut: every stage-1
    row in the final artifact must be ``==`` its row in the stage-1 artifact, and the final artifact
    cites the stage-1 file's sha256 so a reader can check that claim against the committed file.
    """
    roots, _ = _campaign(tmp_path)
    stage1_path = roots.out / "p7_3a_zero_shot_stage1.json"

    stage1 = _report(roots, out_path=stage1_path, stage=tcv.STAGE_CONFIRMATORY)
    final = _report(roots, stage1_path=stage1_path)

    assert final["stage1_artifact"]["sha256"] == _sha256(stage1_path.read_bytes())
    stage1_rows = {tcv.cell_chunk_name(row): row for row in stage1["cells"]}
    final_rows = {tcv.cell_chunk_name(row): row for row in final["cells"]}
    assert stage1_rows, "the stage-1 artifact published no cells, so this asserts nothing"
    for name, row in stage1_rows.items():
        assert final_rows[name] == row, f"{name} moved between the stage-1 and the final artifact"


def test_b2_a_stage_one_row_that_moved_is_refused_by_report(tmp_path: Path) -> None:
    """B2's guard, not just B2's claim: ``report`` must REFUSE a stage-1 row that has moved.

    The test above asserts the two artifacts agree; it would still pass if ``report`` never
    compared them, because it does the comparison itself.  This one edits the stage-1 artifact and
    requires the production code to notice -- the confirmatory number is quoted on a slide before
    the rest of the campaign exists, and a number that moves afterwards is the one thing B2 is for.
    """
    roots, _ = _campaign(tmp_path)
    stage1_path = roots.out / "p7_3a_zero_shot_stage1.json"
    _report(roots, out_path=stage1_path, stage=tcv.STAGE_CONFIRMATORY)

    stage1 = json.loads(stage1_path.read_bytes())
    stage1["cells"][0]["att_env"] = stage1["cells"][0]["att_env"] + 1.0
    stage1_path.write_text(json.dumps(stage1, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stage-1|moved|differ"):
        _report(roots, stage1_path=stage1_path)


def test_report_regenerates_byte_identically_from_the_same_chunks(tmp_path: Path) -> None:
    """Determinism is a feature (``CLAUDE.md`` §2): same chunks, same bytes, twice.

    ``git_commit``/``git_dirty`` come from the chunks rather than from the reporting process, so
    the only way the bytes could move is an unordered container reaching the artifact.
    """
    roots, _ = _campaign(tmp_path)
    out = roots.out / "p7_3a_zero_shot.json"

    _report(roots, out_path=out)
    first = out.read_bytes()
    _report(roots, out_path=out)

    assert out.read_bytes() == first


def test_report_refuses_a_caller_supplied_cell_set_writing_into_docs_data(tmp_path: Path) -> None:
    """The ``cells`` parameter exists so a TEST can declare a small campaign; it is not a dial.

    A completeness check whose expectation comes from the caller is only as good as the caller, so
    the committed artifact may never be produced that way: the campaign path passes no ``cells`` and
    gets ``declared_cells(stage)``, all 4,700 of them.
    """
    roots, _ = _campaign(tmp_path)
    committed = Path(tcv.__file__).resolve().parents[1] / "docs" / "data" / "p7_3a_zero_shot.json"
    before = committed.exists()

    with pytest.raises(ValueError, match="declared_cells|caller"):
        _report(roots, out_path=committed)

    assert committed.exists() == before, "the committed artifact must not be touched by a test"


# ==================================================================================
# G4 -- att_env and att_horizon are ONE quantity by two routes, and neither defaults
# ==================================================================================
def test_g4_a_final_info_without_average_travel_time_is_refused_not_defaulted() -> None:
    """G4. ``horizon_rollout`` reads ``info.get("average_travel_time", 0.0)`` -- the default is the
    problem.

    A missing key would make ``att_horizon`` 0.0, which is not a measurement: it is the absence of
    one, wearing the same type.  P7.1's own expression subscripts the key
    (``sumo_att_reference.py:1242``), and this is that expression with its failure named.

    *Mutation this is built against:* read the final ATT with ``.get(..., 0.0)`` -- this test dies.
    """
    assert tcv.att_env_from_info({"average_travel_time": 366.5}) == 366.5

    with pytest.raises(ValueError, match="average_travel_time"):
        tcv.att_env_from_info({"vehicle_count": 12.0})


def _install_cell_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    att: float = 366.5,
    att_horizon: float | None = None,
    num_phases: int = 8,
    action: int = 0,
    log: list[str] | None = None,
) -> Any:
    """Fake every seam ``run_cell`` reaches, and return the env class so a test can inspect it.

    The fake rollout drives ``env.step`` for real, so the tap that captures the final ``info`` --
    the second of G4's two routes -- is exercised rather than bypassed.  *att_horizon* defaults to
    *att*, i.e. the two routes agree; passing a different value is how H6(b) reaches the refusal.
    *log*, when given, records the order in which the shape check, the policy build, the reset and
    the steps happen, which is H8 MIN-1's whole content.
    """
    from offline.horizon_metric import HorizonRollout
    from offline.sumo_att_reference import HaltingAgreement, SumoAtt, SumoEpisodeReconstruction

    reported = att if att_horizon is None else att_horizon

    def note(event: str) -> None:
        if log is not None:
            log.append(event)

    class _Sumo:
        class simulation:  # noqa: N801 - mirrors traci's module shape
            @staticmethod
            def getOption(name: str) -> str:
                return "-1"

        class vehicle:  # noqa: N801
            @staticmethod
            def getIDList() -> list[str]:
                return ["v0"]

            @staticmethod
            def getTypeID(vehicle_id: str) -> str:
                return "cf_parity"

    class _Env:
        # the registered horizon, because the cell COUNTS its decisions and refuses a short episode
        max_steps = 360
        # `num_phases` is what Utils.infer_action_counts falls back to when there is no
        # action_space (agent/utils/utils.py:29) -- H8 MIN-3 takes the action bound from here
        intersections = [type("Ix", (), {"id": IX, "num_phases": num_phases})()]
        action_space = None
        _sumo = _Sumo()
        # the fresh engine seed the env RNG drew for this episode; the real SumoEnv carries it and
        # the chunk records it beside the REQUESTED seed (A17(f) compares both)
        _engine_seed = 437485271
        recorder = object()
        closed = False

        def reset(self, **kwargs: Any) -> dict[str, Any]:
            note("reset")
            return {"average_travel_time": 1.0, "intersections": {IX: {"reward": -1.0}}}

        def step(self, action_taken: Any) -> tuple[float, bool, bool, dict[str, Any]]:
            note("step")
            return -1.0, False, False, {"average_travel_time": att, "vehicle_count": 12.0}

        def close(self) -> None:
            _Env.closed = True

    def fake_rollout(env: Any, choose: Any, episodes: int, seed: int) -> HorizonRollout:
        info = env.reset(seed=seed)
        for _ in range(int(env.max_steps)):
            _, _, _, info = env.step(choose(env, info))
        return HorizonRollout(
            att_horizon=reported,
            att_running_mean=att - 1.0,
            episode_reward=-23938.0,
            final_vehicle_count=12.0,
            final_completed=float("nan"),
            per_episode_horizon=(reported,),
            per_episode_running_mean=(att - 1.0,),
            episodes=1,
            seed=seed,
        )

    def fake_reconstruct(recorder: Any, **kwargs: Any) -> SumoEpisodeReconstruction:
        return SumoEpisodeReconstruction(
            e_sumo=SumoAtt(value=att + 20.0, total=1.0, n_ids=1821),
            p_sumo=SumoAtt(value=att + 5.0, total=1.0, n_ids=1800),
            w_sumo=SumoAtt(value=att + 1.0, total=1.0, n_ids=1800),
            mean_depart_delay=4.0,
            horizon=3600.0,
            n_observations=3601,
            n_intended=1821,
            n_departed=1800,
            n_arrived=1795,
            n_never_inserted=21,
            n_pending_at_horizon=5,
            n_teleports=0,
            n_vanished_without_arrival=0,
            n_arrived_never_observed_at_a_boundary=0,
            max_abs_depart_clock_deviation=0.0,
            halting=HaltingAgreement(
                n_seconds=3600, n_lane_seconds=28808, max_abs_difference=0,
                n_disagreeing_lane_seconds=0, n_missing=0,
            ),
        )

    def fake_anchor_choose(env: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """The real ``anchor_choose``'s contract: ``(choose, diagnostics)``, the actions counted.

        The count is not decoration -- ``run_cell`` derives ``decisions`` from it and refuses an
        episode that did not reach the registered horizon, so a fake that returned a bare callable
        would be testing a different function.
        """
        note("policy")
        diagnostics: dict[str, Any] = {"rtg_series": [], "reward_series": [], "actions": []}

        def choose(_env: Any, info: Mapping[str, Any]) -> int:
            diagnostics["actions"].append(action)
            return action

        return choose, diagnostics

    real_shape_check = tcv.assert_env_matches_cell

    def spying_shape_check(cell: Mapping[str, Any], env: Any) -> None:
        note("shape-checked")
        real_shape_check(cell, env)

    # ⚠️ Provenance is a seam of the ENVIRONMENT, like the env and the rollout above, and it is
    # faked for the same reason: these tests assert how a chunk is ASSEMBLED, and the tree they run
    # in is the implementer's, which is dirty while the round is being written. Amendment J1(b)
    # refuses a dirty cell -- correctly -- so without this the assembly tests would fail on the
    # state of my working copy rather than on the code under test. That a dirty cell IS refused is
    # asserted separately, against the real function, in
    # test_j1_validate_refuses_a_chunk_without_clean_provenance.
    monkeypatch.setattr(
        tcv, "_git_provenance", lambda: {"git_commit": HEAD_SHA, "git_dirty": False}, raising=True
    )

    monkeypatch.setattr(tcv, "env_for_cell", lambda cell, **k: _Env(), raising=True)
    monkeypatch.setattr(tcv, "anchor_choose", fake_anchor_choose, raising=True)
    monkeypatch.setattr(tcv, "assert_env_matches_cell", spying_shape_check, raising=True)
    monkeypatch.setattr("offline.horizon_metric.horizon_rollout", fake_rollout, raising=True)
    monkeypatch.setattr(
        "offline.sumo_att_reference.reconstruct_sumo_episode", fake_reconstruct, raising=True
    )
    return _Env


def test_run_cell_assembles_a_chunk_whose_two_att_routes_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_cell``'s assembly, driven through fake seams so no simulator is needed.

    What is asserted is what the chunk must carry: the identity digests, C2's halting fields on the
    declared draw only, and ``att_env == att_horizon`` under ``==``.
    """
    roots = _build_roots(tmp_path, draws=(1000, 1001), seeds=(101,))
    att = 366.5
    env_cls = _install_cell_seams(monkeypatch, att=att)

    common = {
        "out_root": roots.draws,
        "output_root": roots.output,
        "data_dir": roots.data,
        "canary_seconds": 0.87,
    }
    checked = tcv.run_cell(_cell("anchor", "fixedtime", 1000), **common)
    unchecked = tcv.run_cell(_cell("anchor", "fixedtime", 1001), **common)

    assert checked["att_env"] == checked["att_horizon"] == att
    assert checked["e_sumo"] == att + 20.0
    assert checked["config_sha256"] == roots.cfg_sha[1000]
    assert checked["routes_sha256"] == roots.routes_sha[1000]
    assert checked["vehicle_types_seen"] == ["cf_parity"]
    assert checked["time_to_teleport_option"] == "-1"
    # C2: the three halting fields exist on the declared draw and are absent elsewhere
    assert checked["halting_checked"] is True
    assert checked["halting_n_disagreeing_lane_seconds"] == 0
    assert unchecked["halting_checked"] is False
    assert unchecked["halting_n_disagreeing_lane_seconds"] is None
    assert env_cls.closed, "the env must be closed even though the caller owns it"
    tcv.validate_cell_payload(checked, cell=_cell("anchor", "fixedtime", 1000))

    # The decision count is COUNTED, not taken from the declared horizon: an episode that stopped
    # early must be refused rather than reported as a full one.  Without this, `decisions` could be
    # a constant and validate_cell_payload's check would be a tautology.
    env_cls.max_steps = 2
    with pytest.raises(ValueError, match="decisions"):
        tcv.run_cell(_cell("anchor", "fixedtime", 1001), **common)


def test_h6b_a_cell_whose_two_att_routes_disagree_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H6(b). G4's *"two routes compared under =="* was a docstring, not a tested refusal.

    ``if False:`` on this comparison left all 62 tests green.  The two routes are the final info's
    ``average_travel_time`` (the tap, P7.1's own expression) and ``horizon_rollout``'s last sample
    of the same key; a difference means the loop this cell ran is not the loop P7.1 measured, and
    no comparison with the frozen anchors would be valid.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    _install_cell_seams(monkeypatch, att=366.5, att_horizon=366.6)

    with pytest.raises(ValueError, match="att_env|disagree"):
        tcv.run_cell(
            _cell("anchor", "fixedtime", 1000),
            out_root=roots.draws,
            output_root=roots.output,
            data_dir=roots.data,
            canary_seconds=0.87,
        )


def test_h8_min1_the_cell_shape_is_checked_before_the_simulator_does_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H8 MIN-1. C8's refusal must come FIRST, and the order is now pinned rather than assumed.

    The reviewer's mutant moved ``assert_env_matches_cell`` after the rollout and T5 stayed green:
    the refusal still fired, but only once a SUMO process had started, an agent had been built and
    360 decisions had been taken.  The point of refusing on the cell's shape is to refuse before
    any of that.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    log: list[str] = []
    _install_cell_seams(monkeypatch, log=log)

    tcv.run_cell(
        _cell("anchor", "fixedtime", 1000),
        out_root=roots.draws,
        output_root=roots.output,
        data_dir=roots.data,
        canary_seconds=0.87,
    )

    assert log[0] == "shape-checked", f"the shape check must come first, got {log[:4]}"
    assert "policy" in log and "reset" in log, "the fake seams must have run at all"
    assert log.index("shape-checked") < log.index("policy") < log.index("reset")


def test_h8_min3_the_action_range_comes_from_the_env_not_from_a_literal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H8 MIN-3. ``actions_in_range`` hardcoded ``< 8``; the bound is the env's own phase count.

    ``Utils.infer_action_counts`` is the repository's helper for exactly this (``CLAUDE.md`` rule
    5) and falls back to ``ix.num_phases``.  hz1x1 happens to have 8 phases, so the literal was
    right here and silently wrong for any other scenario -- a dimension assumption P7.3b would
    inherit.  Here the env reports 4 phases and the policy returns action 5.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    _install_cell_seams(monkeypatch, num_phases=4, action=5)

    with pytest.raises(ValueError, match="action"):
        tcv.run_cell(
            _cell("anchor", "fixedtime", 1000),
            out_root=roots.draws,
            output_root=roots.output,
            data_dir=roots.data,
            canary_seconds=0.87,
        )



# ==================================================================================
# T5 -- the cell's mechanics, on a REAL SUMO episode with a REAL checkpoint
# ==================================================================================
@pytest.mark.skipif(
    not _sumo_available()
    or not _draw_available(SMOKE_DRAW)
    or not _checkpoint_available("mappo1000", 101),
    reason="needs SUMO+traci, scenarios/draws/cityflow1x1/draw_0005/parity, output/p4_dt/dt_seed101.pt",
)
def test_t5_twenty_decisions_of_dt_seed101_through_the_cell_path(tmp_path: Path) -> None:
    """T5. The DT drives SUMO through A16's door, and every mechanic the cell records is checked.

    Twenty decisions, not a full episode: this is the suite's ONE new real-simulator episode (§4's
    cap) and the mechanics it pins are visible in twenty steps.  Draw 5 is P7.2b's fenced smoke draw
    -- nothing here is an evaluation, and no outcome is asserted against a reference.

    The RTG re-derivation is the independent route (``CLAUDE.md`` §2): the stored reward series and
    RTG series are compared under the agent's own shift-by-one rule
    (``rtg[t] - rtg[t-1] == -r(info_{t-1})``, ``transfer_calibration.rtg_advanced_every_decision``)
    rather than by calling the function that produced them.  ``DEFERRED`` 81 closes here: both
    series are stored, so a reviewer can redo exactly this.
    """
    import numpy as np

    calibration = tcv.load_calibration()
    target = tcv.targets_for_subject("mappo1000", calibration)["b_mean_k100"]["target_rtg"]
    checkpoint = OUTPUT_ROOT / "p4_dt" / "dt_seed101.pt"
    cell = _cell("dt", "b_mean_k100", SMOKE_DRAW, subject="mappo1000", seed=101)

    env = tcv.env_for_cell(cell, out_root=DRAWS_ROOT)
    kwargs_seen: list[dict[str, Any]] = []
    try:
        tcv.assert_env_matches_cell(cell, env)
        choose, diagnostics = tcv.dt_choose(
            env, checkpoint_path=checkpoint, target_rtg=target
        )
        agent = diagnostics["agent"]
        real_act = agent.act

        def spy(info: Mapping[str, Any], **kwargs: Any) -> Any:
            kwargs_seen.append(dict(kwargs))
            return real_act(info, **kwargs)

        agent.act = spy  # type: ignore[method-assign]

        info = env.reset(seed=tcv.ENGINE_SEED)
        assert np.asarray(info["intersections"][IX]["state"]).shape[-1] == 25, "A16's frame"
        for _ in range(20):
            _reward, terminated, truncated, info = env.step(choose(env, info))
            if terminated or truncated:
                break
        att = tcv.att_env_from_info(info)
        types_seen = sorted({env._sumo.vehicle.getTypeID(v) for v in env._sumo.vehicle.getIDList()})
        option = str(env._sumo.simulation.getOption("time-to-teleport"))
    finally:
        env.close()

    assert kwargs_seen and all(
        call == {"explore": False, "update_memory": True} for call in kwargs_seen
    ), kwargs_seen
    assert len(diagnostics["actions"]) == 20
    assert all(0 <= action < 8 for action in diagnostics["actions"])

    # ---- H5: the RTG identity, re-derived by RAW ARITHMETIC on the stored series -------------
    # ⚠️ `rtg_advanced_every_decision` is CHANGE DETECTION -- it asks whether rtg[i] != rtg[i-1]
    # iff reward[i-1] != 0 and never compares magnitudes, so a DOUBLE-COUNTED reward passes it.
    # §4 T5 says *re-derived*, and this is the re-derivation: the agent's own convention is
    #     rtg[t] = target - sum(reward[0..t-1])  with the step-0 reward forced to 0.0,
    # so rtg[t-1] - rtg[t] == reward_series[t-1] for t >= 2, and rtg[1] == rtg[0] always
    # (agent/DTAgent.py: current_rtg is target - reward_sum; act adds 0.0 when step == 0).
    # Under `==`: the driving reward is a vehicle COUNT, so every term is exactly representable.
    rtg = diagnostics["rtg_series"]
    rewards = diagnostics["reward_series"]
    assert rtg[0] == target, "rtg_first is the declared target"
    assert rtg[1] - rtg[0] == 0.0, "the agent forces the step-0 reward to 0.0, so index 1 is fixed"
    for t in range(2, len(rtg)):
        assert rtg[t - 1] - rtg[t] == rewards[t - 1], (
            f"decision {t}: the RTG moved by {rtg[t - 1] - rtg[t]!r} while the info's reward was "
            f"{rewards[t - 1]!r}; the conditioning trajectory is not the reward stream's cumulative "
            "sum, which is what the prompt means"
        )
    # the same identity in one line, by the cumulative sum rather than by the differences
    assert rtg[-1] == target - float(sum(rewards[: len(rtg) - 1])), "cumsum route disagrees"
    # and the helper agrees with the arithmetic (it is weaker, so it is asserted second)
    assert tc.rtg_advanced_every_decision(rtg, rewards)
    assert types_seen == ["cf_parity"]
    assert option == "-1"
    assert att > 0.0, "a real episode has a positive average travel time"


# ==================================================================================
# T9 -- the driver, asserted over COMMENT-FREE text
# ==================================================================================
DRIVER = Path(__file__).resolve().parents[1] / "offline" / "campaigns" / "p7_3a_zero_shot.sh"


def _driver_code_text() -> str:
    """The driver with every whole-line comment removed.

    ⚠️ **Amendment E1.3 item 2, after the coordinator's mutant MD2 SURVIVED on P7.2b's driver.**
    That test asserted ``"tee /dev/stderr" in text``; deleting ``tee`` from the CODE line left the
    assertion satisfied by the COMMENT explaining what ``tee`` was there for.  A text assertion over
    a file that documents itself will be satisfied by its own documentation.  Assertions about what
    the driver DOES are made over this text; assertions about what it SAYS may use the whole file.
    """
    return "\n".join(
        line
        for line in DRIVER.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def test_t9_the_driver_exists_and_is_syntactically_valid() -> None:
    """``bash -n`` -- a driver that does not parse consumes the token and then dies."""
    import subprocess

    assert DRIVER.is_file(), f"no driver at {DRIVER}"
    assert subprocess.run(["bash", "-n", str(DRIVER)], capture_output=True).returncode == 0


def test_h3_the_driver_sets_euo_pipefail_exactly() -> None:
    """H3. Reviewer MAJ-1, measured: both ``set -eu`` and the deleted line survived T9.

    Without ``pipefail`` the canary's own line shape is the defect: the correctness half fails
    inside a pipeline, the substitution yields ``CANARY_LINE=''``, the script proceeds, **the token
    is consumed**, and only ``record-canary`` fails.  A failed canary must never burn the author's
    one-shot authorisation -- that is the class of defect P7.2b's pre-flight found, and this is the
    one assertion that notices it.
    """
    code = _driver_code_text()
    assert "set -euo pipefail" in code
    assert "\nset -eu\n" not in code, "'set -eu' without pipefail hides a failing pipeline"


def test_h2_the_confirmatory_stage_can_be_restarted_and_never_overwrites() -> None:
    """H2. ``offline.collect`` refuses a populated ``--out-dir``, so a restart used to die there.

    After any failure past the collection -- the pool, the report, the manifest -- the restart hit
    ``collect`` and stopped.  Stage 1 is now *if the corpus exists, gate it and skip the
    collection; else collect and gate it*, and ``--overwrite`` never appears: a partial corpus is
    refused by H1's gate with its missing draws named and is moved aside **by hand**, because
    deleting a corpus the driver cannot prove is bad is not the driver's decision to make.
    """
    code = _driver_code_text()
    assert "--overwrite" not in code, (
        "--overwrite would let a restart silently replace a corpus A17(f) has already blessed"
    )
    # the gate runs on BOTH paths: the one that collected and the one that skipped
    assert code.count("a17f --corpus-dir") == 2, "A17(f) must run whether or not we just collected"
    assert code.index("a17f --corpus-dir") < code.index("cells --stage"), (
        "the gate precedes every evaluation cell"
    )
    assert '[ -d "$CORPUS" ]' in code, "the skip decision is on the corpus existing"


def test_i3_both_stage_one_branches_log_to_one_file_naming_the_branch(tmp_path: Path) -> None:
    """Amendment I3(3): ONE log name, and the branch taken is the entry's first line.

    The stage-1 checkpoint is the coordinator reading A17(f)'s result from disk, so it must be one
    file to read and one manifest entry -- not ``a17f.log`` or ``a17f_existing.log`` depending on
    which path the run took.  The branch is written INTO that file because the two paths mean
    different things: one collected a corpus, the other accepted one that was already there.
    """
    code = _driver_code_text()

    assert "a17f_existing" not in code, "I3(3): one log name for both branches"
    assert code.count("run_stage a17f ") == 2, "both branches run the stage under the same label"
    # each branch announces itself, and the note reaches the log before the stage's header line
    assert code.count("STAGE_NOTE=") >= 2
    assert code.count("branch:") == 2, "each branch names itself in the entry"
    assert '"$STAGE_NOTE"' in code and '>> "$log"' in code

    # the note is written FIRST: a reader opening logs/a17f.log must see which path ran before the
    # stage's own output, not after it
    body = code[code.index("run_stage() {") : code.index("run_stage() {") + 700]
    assert body.index("STAGE_NOTE") < body.index('echo "=== $label  run at'), (
        "the branch line must precede the stage header inside the log entry"
    )


def test_t9_the_ordering_the_token_depends_on() -> None:
    """Every check that can refuse PRECEDES the token, and every write FOLLOWS it.

    The token is a one-shot authorisation the author writes by hand: a refused start must consume
    nothing and create nothing, and from the first destructive line onward a signal must leave
    FAILED behind.  Asserted as ORDER, because each of these lines existing somewhere in the file
    is exactly what the P7.2b pre-flight found insufficient (the trap was installed after the token
    and a signal in that window destroyed the authorisation silently).
    """
    code = _driver_code_text()
    token = code.index('rm -f "$TOKEN"')
    for label, needle, before in (
        ("the canary", "canary | tee -a /dev/stderr", True),
        ("the trap", "trap on_signal INT TERM", True),
        ("the group-leader check", "ps -o pgid=", True),
        ("record-canary", "record-canary", False),
        ("the logs directory", 'mkdir -p "$LOGS"', False),
    ):
        index = code.index(needle)
        if before:
            assert index < token, f"{label} must precede the token being consumed"
        else:
            assert index > token, f"{label} must follow the token being consumed"


def test_t9_a17f_stops_the_driver_before_any_evaluation_cell_runs() -> None:
    """Gate 3: the collection and A17(f) come before the pool, and a mismatch stops the driver.

    A17(f) is the consistency gate between P7.2b and P7.3 -- *100/100 or P7.3 stops*.  It is cheap
    (about 20 minutes) and it is where a wiring defect surfaces, so it must precede the hours of
    evaluation rather than run beside them.

    ⚠️ **The needles are the COMMAND forms, not words.**  The first version of this test searched
    for ``" cells "`` and matched the refusal message *"cells from another run are still alive"* --
    1,296 characters before the subcommand it meant, which made the ordering assertion fail against
    a driver whose order was correct.  A needle that can match prose is a needle that will.

    ⚠️ **And ``index`` is not ``rindex``.**  Amendment I3(3) gave both stage-1 branches the same
    label, so ``run_stage a17f`` now appears twice and the FIRST occurrence is the skip branch --
    which precedes the collect branch and has no collection in it.  The invariant was never "the
    first gate follows the first collection"; it is that the collect branch gates what it collected,
    and that **every** gate precedes **every** evaluation cell.  Stated that way it survives the
    branch and would still catch a gate moved after the pool.
    """
    code = _driver_code_text()
    collection = code.index("run_stage collect offline.collect")
    last_gate = code.rindex("a17f --corpus-dir")
    first_cells = code.index("cells --stage")

    assert collection < last_gate, "the collect branch must gate the corpus it just collected"
    assert last_gate < first_cells, "EVERY A17(f) call precedes EVERY evaluation cell"
    # Every stage goes through run_stage, which calls `fail "$label"` when the child exits
    # non-zero; that is what turns A17(f)'s refusal into a stopped campaign rather than a log line.
    assert 'fail "$label"' in code, "run_stage must stop the driver when a stage exits non-zero"
    assert code.index("run_stage()") < code.index("run_stage a17f"), (
        "a17f must be run THROUGH run_stage, not beside it"
    )
    assert code.count("run_stage a17f") == 2, "both stage-1 branches gate through run_stage"


def test_t9_the_worker_count_and_every_root_come_from_variables() -> None:
    """§0.9: no new hardcoded absolute path, and C3's 12 workers as a variable, not a literal.

    A literal worker count cannot be lowered by an operator whose machine is smaller, and a
    hardcoded root is how ``DEFERRED`` 82 happened: the roots became untestable anywhere but this
    machine.
    """
    code = _driver_code_text()
    assert "WORKERS=" in code
    assert '--workers "$WORKERS"' in code
    for variable in ("MAIN=", "WORK=", "DRAWS=", "LOGS=", "CORPUS="):
        assert variable in code, f"{variable} must be a variable with today's path as its default"
    # every absolute path in the code is a DEFAULT assignment, never buried in a command
    offenders = [
        line.strip()
        for line in code.splitlines()
        if "/home/filip" in line and not line.strip().startswith(("MAIN=", "PY=", "export"))
    ]
    assert offenders == [], f"absolute paths outside the variable block: {offenders}"


def test_t9_logs_are_appended_and_never_truncated() -> None:
    """A restart must not destroy the first run's narrative record.

    ``tee -a`` matters for the same reason and is load-bearing beyond style: plain
    ``tee /dev/stderr`` re-opens stderr with ``O_TRUNC``, so a driver run with ``>> log 2>&1``
    resets the file offset and destroys everything already written.  P7.2b lost run 3's capture
    that way (Amendment E1.3 item 3).
    """
    code = _driver_code_text()
    assert "tee -a /dev/stderr" in code
    assert "tee /dev/stderr" not in code.replace("tee -a /dev/stderr", "")

    truncating = [
        line.strip()
        for line in code.splitlines()
        if "$LOGS/" in line and ">" in line and ">>" not in line and "tee" not in line
    ]
    assert truncating == [], f"these lines truncate a log instead of appending: {truncating}"


def test_t9_the_manifest_is_written_last_and_atomically() -> None:
    """A manifest is the record of what a run produced, so it is written whole or not at all."""
    code = _driver_code_text()
    assert "SHA256SUMS_p7_3a.txt.tmp" in code
    assert code.index("SHA256SUMS_p7_3a.txt.tmp") > code.index(" report ")
    assert "sha256sum -c" in code, "the manifest is re-verified after it is written"


def test_t9_the_driver_declares_both_stages_and_refuses_an_unknown_one() -> None:
    """Amendment B1's two stages, and B2's stage-1 artifact cited by the final report.

    Stage 2 is UNCONDITIONAL: the driver runs it whatever stage 1 showed, which is why the stage is
    an ARGUMENT rather than a decision the driver makes from a number.
    """
    code = _driver_code_text()
    # ⚠️ `"rest" in code` is THEATRE: it matches the word "restart" in a refusal message (reviewer,
    # handed over by H9).  The needle is the stage dispatch's own form.
    assert "confirmatory|rest)" in code, "both stages must be accepted by the case statement"
    assert '"$STAGE" = "rest"' in code or '"$STAGE" = "confirmatory"' in code
    assert "--stage1-path" in code, "the final report must cite the stage-1 artifact (B2)"
    assert "REFUSING TO START" in code


# ==================================================================================
# The pool's resume path -- the half that can be tested without a simulator
# ==================================================================================
def test_run_stage_skips_every_cell_whose_chunk_is_already_reusable(tmp_path: Path) -> None:
    """A restart on a complete work directory rolls NOTHING and starts no worker at all.

    This is the half of the pool that a test can reach: the skip decision.  It is in Python rather
    than in the shell because ``offline/campaigns/p5_3b.sh``'s ``[ -f ]`` guard let a bad chunk
    survive every restart, and the verdict here is re-derived from the chunk and from the files on
    disk.  If a cell were wrongly judged unusable this test would spawn a SUMO worker and hang --
    which is itself the assertion that nothing was rolled.
    """
    roots, _ = _campaign(tmp_path)

    summary = tcv.run_stage(
        work_dir=roots.work,
        out_root=roots.draws,
        output_root=roots.output,
        data_dir=roots.data,
        cells=_DEMO_CELLS,
        workers=1,
    )

    assert summary["n_declared"] == len(_DEMO_CELLS)
    assert summary["n_reused"] == len(_DEMO_CELLS)
    assert summary["n_rolled"] == 0
    assert summary["n_failed"] == 0
    assert not (roots.work / "failed").exists(), "nothing was unusable, so nothing was moved aside"


def test_an_unusable_chunk_is_moved_aside_and_never_overwritten(tmp_path: Path) -> None:
    """E3(d): a chunk that failed its own re-validation is EVIDENCE about a run.

    It is moved into ``failed/`` rather than overwritten, and a second failure of the same cell
    does not overwrite the first one either -- the suffix grows.  ``report``'s glob does not see
    ``failed/``.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    cell = _cell("anchor", "fixedtime", 1000)
    path = tcv.write_chunk(_payload(cell, roots, n_teleports=3), work_dir=roots.work)

    first = tcv.move_aside(path)
    assert first == roots.work / "failed" / path.name
    assert not path.exists()

    tcv.write_chunk(_payload(cell, roots, n_teleports=7), work_dir=roots.work)
    second = tcv.move_aside(path)

    assert second != first, "the first failed chunk must not be overwritten by the second"
    assert json.loads(first.read_bytes())["n_teleports"] == 3
    assert json.loads(second.read_bytes())["n_teleports"] == 7


# ==================================================================================
# H1 -- A17(f) as a CLI stage: 100/100 over the DECLARED band, not over what is there
# ==================================================================================
def _write_corpus(
    directory: Path, *, draws: Sequence[int], artifact: Path, mismatch: int | None = None
) -> Path:
    """A synthetic corpus in the shape :func:`assert_logged_corpus_matches_probe` reads.

    Read out of that function's BODY, not its docstring: one ``.npz`` per draw carrying
    ``flow_draw`` and ``ix0_local_reward``, plus a manifest whose ``run_metadata.sumo_draws``
    records each draw's ``engine_seed_drawn``.  The rewards are integral float32 summing to the
    probe's own ``local_return``, because the gate checks integrality before equality.
    """
    import numpy as np

    probe = {int(row["draw_id"]): row for row in json.loads(artifact.read_bytes())["probe"]}
    directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for draw in draws:
        row = probe[int(draw)]
        total = float(row["local_return"]) + (1.0 if draw == mismatch else 0.0)
        rewards = np.zeros(360, dtype=np.float32)
        rewards[0] = np.float32(total)
        np.savez(
            directory / f"ep_draw{int(draw):04d}.npz",
            flow_draw=np.int64(int(draw)),
            ix0_local_reward=rewards,
        )
        records.append(
            {"draw_id": int(draw), "engine_seed_drawn": int(row["engine_seed_drawn"])}
        )
    (directory / "manifest.json").write_text(
        json.dumps({"run_metadata": {"sumo_draws": records}}), encoding="utf-8"
    )
    return directory


def _a17f(roots: _Roots, corpus: Path) -> int:
    return tcv.main(
        [
            "--draws-root", str(roots.draws),
            "--output-root", str(roots.output),
            "--work-dir", str(roots.work),
            "--data-dir", str(roots.data),
            "--out-dir", str(roots.out),
            "a17f", "--corpus-dir", str(corpus),
        ]
    )


def test_h1_a17f_passes_only_on_the_full_declared_band(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """H1. *100/100 or P7.3 stops* -- over the DECLARED band, never over what happens to be there.

    ``assert_logged_corpus_matches_probe`` defaults ``draw_ids=None`` to *the draws present in the
    corpus*, and the CLI passed no draw set: the coordinator ran it on the ONE-draw smoke corpus
    and got ``n_checked: 1, all_match: true, exit 0``.  A 99-draw corpus would have passed the
    driver's gate and P7.3b's few-shot source would have been short a draw with nothing saying so.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    corpus = _write_corpus(
        tmp_path / "corpus",
        draws=range(201, 301),
        artifact=roots.data / "p7_2b_calibration.json",
    )

    assert _a17f(roots, corpus) == 0
    printed = capsys.readouterr().out
    assert "A17(f) 100/100" in printed, "the line the coordinator reads at the stage-1 checkpoint"


def test_h1_a_99_draw_corpus_is_refused_naming_the_missing_draw(tmp_path: Path) -> None:
    """The falsification the coordinator ran by hand, as a test.

    *Mutation this is built against:* drop the explicit draw set so the gate checks whatever the
    corpus contains. Then this corpus of 99 draws reports 99/99 and the campaign proceeds.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    short = [d for d in range(201, 301) if d != 300]
    corpus = _write_corpus(
        tmp_path / "corpus", draws=short, artifact=roots.data / "p7_2b_calibration.json"
    )

    with pytest.raises(ValueError, match="300"):
        _a17f(roots, corpus)


def test_h1_one_mismatched_draw_is_refused_with_both_values_and_the_difference(
    tmp_path: Path,
) -> None:
    """A mismatch is a finding reported with the draw, both values and the difference.

    Never smoothed and never tolerated: SUMO is deterministic under a fixed seed, so a difference
    is a wiring defect, and P7.3 stops rather than reporting a transfer number built on an episode
    that is not the one the prompt was calibrated from.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    corpus = _write_corpus(
        tmp_path / "corpus",
        draws=range(201, 301),
        artifact=roots.data / "p7_2b_calibration.json",
        mismatch=250,
    )

    with pytest.raises(ValueError, match=r"draw 250.*difference") as caught:
        _a17f(roots, corpus)
    message = str(caught.value)
    assert "-23917.0" in message or "difference" in message
    assert "1.0" in message, "the difference itself must be in the message"


def test_h1_the_gate_stage_refuses_a_moved_calibration_artifact(tmp_path: Path) -> None:
    """Reviewer MIN-7: the gate stage loads the artifact through ``load_calibration``.

    The gate compares against P7.2b's probe rows, so it is reading the same artifact the prompts
    come from; a stage that read it unpinned could check a corpus against a file that had moved.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    corpus = _write_corpus(
        tmp_path / "corpus",
        draws=range(201, 301),
        artifact=roots.data / "p7_2b_calibration.json",
    )
    moved = roots.data / "p7_2b_calibration.json"
    moved.write_bytes(moved.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="sha256"):
        _a17f(roots, corpus)


# ==================================================================================
# H6 -- two refusals that had no test at all (the coordinator's survivors MU5, MU6)
# ==================================================================================
@pytest.mark.parametrize(
    ("draw", "checked"),
    [(1000, False), (1001, True)],
)
def test_h6a_a_chunk_whose_halting_flag_contradicts_c2_is_refused(
    tmp_path: Path, draw: int, checked: bool
) -> None:
    """H6(a). ``halting_checked`` must agree with Amendment C2's declaration, both directions.

    ``if False:`` on this refusal left all 62 tests green.  A resumed chunk written by a run with
    the flag off on draw 1000 would pass into the artifact, and C2's declared subset -- the only
    verification the recorder gets -- would silently be empty.
    """
    roots = _build_roots(tmp_path, draws=(1000, 1001), seeds=(101,))
    cell = _cell("anchor", "fixedtime", draw)

    with pytest.raises(ValueError, match="halting_checked|Amendment C2"):
        tcv.validate_cell_payload(_payload(cell, roots, halting_checked=checked), cell=cell)


# ==================================================================================
# H7 -- H3's block states what it measured, and never a verdict
# ==================================================================================
def test_h7_h3_reports_the_point_estimate_and_the_interval_separately(tmp_path: Path) -> None:
    """H7. ``holds`` reads as *confirmed*; it was a boolean on the point estimate alone.

    Two mechanical facts now sit side by side and neither is a verdict:
    ``point_estimate_satisfies`` (the mean is on the claimed side) and ``ci95_entirely_satisfies``
    (the WHOLE analytic interval is).  The registered test row is the paired comparison, and the
    artifact says what it measured rather than what it concludes.
    """
    roots, _ = _campaign(tmp_path)

    artifact = _report(roots)

    clauses = artifact["h3"]["clauses"]
    assert len(clauses) == 4
    for clause in clauses:
        assert "holds" not in clause, "a key named 'holds' will be read as a verdict"
        assert set(clause["point_estimate_satisfies"]) <= {"mappo1000", "mix50"}
        assert set(clause["ci95_entirely_satisfies"]) <= {"mappo1000", "mix50"}

    # ...and the two are computed from DIFFERENT quantities: on this fixture rho is 0.5 with a
    # non-zero interval, so "> 0" holds on the point estimate and on the whole interval, while
    # "< 1" holds on the point estimate and the interval is checked on its own end.
    greater = [c for c in clauses if c["inequality"].endswith("> 0") and c["definition"] == "e_sumo"][0]
    entry = [
        e for e in artifact["rho"]["by_subject_arm"]
        if e["subject"] == "mappo1000" and e["arm"] == "b_mean_k100"
    ][0]
    assert greater["point_estimate_satisfies"]["mappo1000"] == (entry["e_sumo"]["mean"] > 0.0)
    assert greater["ci95_entirely_satisfies"]["mappo1000"] == (entry["e_sumo"]["ci95_low"] > 0.0)


def test_h7_the_interval_fact_is_not_the_point_estimate_fact() -> None:
    """The two facts must DIFFER where the data makes them differ, or one of them says nothing.

    ⚠️ On the campaign fixture rho is 0.5 with a narrow interval, so ``mean > 0`` and
    ``ci95_low > 0`` are both true and a version computing the interval fact FROM THE MEAN passes
    every assertion -- the mutant survived exactly that way.  Here the interval straddles both
    bounds (mean 0.5, CI [-0.2, 1.2]): the point estimate satisfies both clauses and the interval
    satisfies neither, which is the case the artifact exists to make visible.  A reader who sees
    only a point estimate on the claimed side would read it as a result; this is why H7 refused to
    let one boolean be called ``holds``.
    """
    stats = {"n_draws": 100, "mean": 0.5, "std": 1.0, "ci95": 0.7, "ci95_low": -0.2, "ci95_high": 1.2}
    block = tcv._h3_block(
        [{"subject": "mappo1000", "arm": "b_mean_k100", "role": "registered_prompt",
          "e_sumo": dict(stats), "att_env": dict(stats)}]
    )

    for inequality in ("rho_sumo(b_mean_k100) > 0", "rho_sumo(b_mean_k100) < 1"):
        clause = [
            c for c in block["clauses"]
            if c["inequality"] == inequality and c["definition"] == "e_sumo"
        ][0]
        assert clause["point_estimate_satisfies"]["mappo1000"] is True
        assert clause["ci95_entirely_satisfies"]["mappo1000"] is False, (
            f"{inequality}: the mean is on the claimed side but the interval is not, and the "
            "artifact must say both"
        )


# ==================================================================================
# H8 -- the four minors
# ==================================================================================
def test_h8_min5_the_stage_one_report_still_regenerates_once_stage_two_chunks_exist(
    tmp_path: Path,
) -> None:
    """H8 MIN-5. The two stages share one work directory, so ``report --stage`` filters by NAME.

    Before this, ``report --stage confirmatory`` refused as soon as stage 2 had written a chunk
    (an "extra" cell), which made the stage-1 artifact regenerable only from a tree that no longer
    exists after stage 2 -- and B2's byte-identity claim is checked at the recording commit.
    Every chunk on disk is still validated, so an undeclared ARM anywhere still refuses.
    """
    roots, _ = _campaign(tmp_path)
    rest_cell = _cell("dt", "naive", 1000, subject="mappo1000", seed=101, stage="rest")
    tcv.write_chunk(_payload(rest_cell, roots), work_dir=roots.work)

    artifact = _report(
        roots,
        out_path=roots.out / "p7_3a_zero_shot_stage1.json",
        stage=tcv.STAGE_CONFIRMATORY,
        cells=[*_DEMO_CELLS, rest_cell],
    )

    assert artifact["n_chunks_outside_stage"] == 1
    assert all(row["stage"] == tcv.STAGE_CONFIRMATORY for row in artifact["cells"])
    assert "naive" not in {row["arm"] for row in artifact["cells"]}


def test_h8_min5_an_undeclared_arm_still_refuses_even_when_it_is_out_of_stage(
    tmp_path: Path,
) -> None:
    """The fence is not weakened by MIN-5: an arm nobody declared refuses from anywhere.

    B2's rule -- every chunk is validated, whatever stage it belongs to -- is what keeps the
    filtering from becoming a way to hide a cell.
    """
    roots, _ = _campaign(tmp_path)
    undeclared = _cell("dt", "b_mean_k20", 1000, subject="mappo1000", seed=101, stage="rest")
    tcv.write_chunk(_payload(undeclared, roots), work_dir=roots.work)

    with pytest.raises(ValueError, match="b_mean_k20|declared"):
        _report(roots, stage=tcv.STAGE_CONFIRMATORY, cells=_DEMO_CELLS)


def test_h8_min5_a_broken_chunk_of_the_OTHER_stage_is_still_refused(tmp_path: Path) -> None:
    """MIN-5 filters which cells are REPORTED; it does not stop any chunk being validated.

    ⚠️ The undeclared-arm test above is caught one step higher, by the ``extra`` check, so it does
    not pin the validation loop: a mutant that validated only this stage's chunks survived it.
    Here the offending chunk IS a declared cell of the other stage -- so ``extra`` is content --
    and it carries a teleport. Under A15(c)'s teleport-free regime that is a finding about the
    whole campaign, and a report that published stage 1 while a teleporting stage-2 cell sat beside
    it would be reporting from a work directory it had not checked.
    """
    roots, _ = _campaign(tmp_path)
    rest_cell = _cell("dt", "naive", 1000, subject="mappo1000", seed=101, stage="rest")
    tcv.write_chunk(_payload(rest_cell, roots, n_teleports=4), work_dir=roots.work)

    with pytest.raises(ValueError, match="teleport"):
        _report(
            roots,
            out_path=roots.out / "p7_3a_zero_shot_stage1.json",
            stage=tcv.STAGE_CONFIRMATORY,
            cells=[*_DEMO_CELLS, rest_cell],
        )
    assert list(roots.out.iterdir()) == []


def test_h8_min6_the_canary_prints_its_line_before_it_checks_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """H8 MIN-6. On a failed correctness half the OBSERVED values must still be visible.

    P7.2b prints the line and then checks, so the numbers reach the pane and ``canary.log`` even
    when the refusal follows; checking first means the refusal says only that something was wrong.
    The driver captures this line with ``tee -a``, so "printed" is also "recorded".
    """
    import offline.transfer_calibration as calibration

    monkeypatch.setattr(
        calibration,
        "canary_seconds",
        lambda: (0.91, {"decisions": 360, "local_return": -1.0, "att_horizon": 2.0,
                        "two_routes_agree": True}),
        raising=True,
    )

    with pytest.raises(ValueError, match="canary"):
        tcv.main(["canary"])

    printed = capsys.readouterr().out
    assert "canary 0.91 s" in printed, "the observed line must survive a failed check"
    assert "-1.0" in printed


# ==================================================================================
# F1 / I5(2) -- the pre-flight pilot's declared cell set, and its fence
# ==================================================================================
def test_f1_the_pilot_is_sixteen_dt_cells_on_the_fenced_draw_over_both_subjects() -> None:
    """F1's pilot, declared as a set before it runs: 16 DT cells, draw 5, both subjects.

    Draw 5 is P7.2b's **fenced smoke draw** and is deliberately NOT in the held-out pool: *"a pilot
    on the held-out pool would BE the experiment, run before the token"*
    (``docs/plans/p7.3a_amendment_a_measurements.md`` §A6).  The composition covers both registered
    subjects and all four declared arms at two seeds, so the pilot exercises the target lookup for
    every arm rather than running one arm sixteen times -- the pilot's FIRST purpose is the
    mechanics, and its rate is a by-product (F1).
    """
    cells = tcv.pilot_cells()

    assert len(cells) == 16, "2 subjects x 4 declared arms x 2 seeds"
    assert {c["draw_id"] for c in cells} == {tcv.PILOT_DRAW} == {5}
    assert {c["kind"] for c in cells} == {"dt"}
    assert {c["subject"] for c in cells} == set(tcv.SUBJECTS)
    assert {c["arm"] for c in cells} == {spec.name for spec in tcv.DECLARED_ARMS}
    assert {c["seed"] for c in cells} == set(tcv.PILOT_SEEDS)
    assert len({tcv.cell_chunk_name(c) for c in cells}) == 16, "two cells would share a chunk"

    # the pilot is not a stage of the campaign, and its draw is not in the registered pool
    from offline.materialise_draws import classify_draw_pool

    assert {c["stage"] for c in cells} == {tcv.PILOT_STAGE}
    assert tcv.PILOT_STAGE not in tcv.STAGES
    assert classify_draw_pool(tcv.PILOT_DRAW) != "held_out"
    assert tcv.halting_check_for(tcv.PILOT_DRAW) is False, "C2's subset is draw 1000 only"


def test_f1_report_refuses_a_pilot_work_directory(tmp_path: Path) -> None:
    """THE FENCE, mechanical rather than promised: a pilot chunk cannot reach an artifact.

    ``report`` is never called on the pilot -- that is the instruction -- but "never called" is a
    procedure, and procedures are what this project stops trusting.  A pilot cell is not a declared
    cell of any stage, so ``report`` refuses the directory outright; and since draw 5 is not in the
    held-out pool, no pilot chunk can satisfy the campaign's completeness check either.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    pilot_dir = tmp_path / "preflight_pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    _canary(pilot_dir)
    # a pilot chunk needs draw 5's parity digests; the fixture builds draw 1000's, so this chunk is
    # written with draw 5's identity and the fixture's digests -- report refuses before it reads them
    cell = tcv.pilot_cells()[0]
    payload = _payload({**cell, "draw_id": 1000}, roots)
    payload.update(
        {
            "draw_id": tcv.PILOT_DRAW,
            "stage": tcv.PILOT_STAGE,
            # draw 5 is not C2's declared subset, so a real pilot chunk carries the flag off and
            # the three halting fields absent; without this the chunk trips the C2 refusal first
            # and the test would pass for a reason that has nothing to do with the fence
            "halting_checked": False,
            "halting_max_abs_difference": None,
            "halting_n_lane_seconds": None,
            "halting_n_disagreeing_lane_seconds": None,
        }
    )
    tcv.write_chunk(payload, work_dir=pilot_dir)

    with pytest.raises(ValueError, match="declared"):
        tcv.report(
            work_dir=pilot_dir,
            out_path=roots.out / "p7_3a_zero_shot.json",
            output_root=roots.output,
            out_root=roots.draws,
            data_dir=roots.data,
        )
    assert not (roots.out / "p7_3a_zero_shot.json").exists()


def test_f1_the_pilot_summary_carries_no_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FENCED: wall time and counts only -- no ATT, no rho, no return, for any cell.

    The pilot runs on draw 5 with real checkpoints, so every chunk it writes DOES contain an
    outcome; what must never happen is that an outcome is printed, summarised or published.  This
    asserts the summary's own shape, which is what reaches the transcript and the packet.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    monkeypatch.setattr(
        tcv, "run_stage",
        lambda **kwargs: {
            "stage": "pilot", "n_declared": 16, "n_reused": 0, "n_rolled": 16, "n_failed": 0,
            "failures": [], "seconds": [11.0] * 16, "wall_seconds": 40.0,
        },
        raising=True,
    )
    monkeypatch.setattr(
        "offline.transfer_calibration.canary_seconds",
        lambda: (0.9, {"decisions": 360, "local_return": tc.CANARY_REFERENCE_LOCAL_RETURN,
                       "att_horizon": tc.CANARY_REFERENCE_ATT_HORIZON, "two_routes_agree": True}),
        raising=True,
    )

    summary = tcv.run_pilot(
        work_dir=tmp_path / "pilot",
        out_root=roots.draws,
        output_root=roots.output,
        data_dir=roots.data,
        transcript_path=tmp_path / "preflight_pilot.txt",
    )

    # The scan is over KEYS, not over the whole JSON: the record's prose says the words "no ATT,
    # no rho" on purpose, and a substring scan over values would be satisfied -- or broken -- by
    # its own documentation, which is the same trap `_driver_code_text` exists for.
    def keys_of(node: Any) -> list[str]:
        if isinstance(node, dict):
            return [str(k) for k in node] + [x for v in node.values() for x in keys_of(v)]
        if isinstance(node, list):
            return [x for item in node for x in keys_of(item)]
        return []

    forbidden = ("att", "rho", "e_sumo", "episode_reward", "return", "horizon", "support")
    leaked = [k for k in keys_of(summary) for w in forbidden if w in k.lower()]
    assert leaked == [], f"the pilot summary carries an outcome field: {leaked}"
    # ...and every number in it is a clock or a count
    assert summary["n_cells"] == 16
    assert summary["wall_seconds"] == 40.0
    assert summary["effective_seconds_per_cell"] == 40.0 / 16

    transcript = (tmp_path / "preflight_pilot.txt").read_text(encoding="utf-8")
    assert "canary 0.9" in transcript, "the transcript carries the canary line (F1)"
    assert "FENCED" in transcript
    assert "n = 16" in transcript
    assert "draw 5" in transcript


# ==================================================================================
# J1 -- provenance is MEASURED strictly and CHECKED at every consumer
# ==================================================================================
def test_j1_provenance_is_measured_strictly_and_a_failing_git_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """J1(a). A failure to MEASURE the tree must raise, never read as a clean tree.

    ``materialise_draws._git_commit`` returns ``dirty = False`` whenever ``git status`` cannot run,
    so an unmeasured tree records as a clean one -- and section 7's gate, *the zero-shot artifact's
    provenance is clean*, would then be satisfied by the absence of evidence.  P7.3a measures it
    itself and refuses instead.
    """
    import subprocess as sp

    provenance = tcv._git_provenance()
    assert set(provenance) == {"git_commit", "git_dirty"}
    assert len(provenance["git_commit"]) == 40
    assert provenance["git_commit"] == HEAD_SHA
    assert isinstance(provenance["git_dirty"], bool)

    def failing_run(*args: Any, **kwargs: Any) -> Any:
        return sp.CompletedProcess(args=args, returncode=128, stdout="", stderr="not a repository")

    monkeypatch.setattr(tcv.subprocess, "run", failing_run, raising=True)
    with pytest.raises(RuntimeError, match="git|provenance"):
        tcv._git_provenance()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("git_dirty", True, "dirty"),
        ("git_commit", "abc123", "git_commit"),
        ("git_commit", "z" * 40, "git_commit"),
        ("git_commit", None, "git_commit"),
    ],
)
def test_j1_validate_refuses_a_chunk_without_clean_provenance(
    tmp_path: Path, field: str, value: Any, match: str
) -> None:
    """J1(b). A cell rolled on a dirty tree, or from an unidentifiable commit, is not evidence.

    The pilot's 33 real chunks all recorded ``git_dirty: true`` and nothing looked at it.  Forty
    hex characters is what a commit is; ``"z" * 40`` has the right length and is not one.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    cell = _cell("anchor", "fixedtime", 1000)

    with pytest.raises(ValueError, match=match):
        tcv.validate_cell_payload(_payload(cell, roots, **{field: value}), cell=cell)


def _repository_is_shallow() -> bool:
    """Is this checkout missing its history?  ``actions/checkout@v4`` fetches **depth 1** by default.

    On a shallow clone ``git rev-list --max-parents=0 HEAD`` returns **HEAD ITSELF**: the grafted
    boundary commit has no parent as far as this repository can see, so it answers as the root.  The
    test below then compares HEAD with HEAD, ``code_changed_since`` correctly returns ``[]``, and its
    second assertion fails -- for a reason that is about the CHECKOUT and not about the code.
    Observed on CI run 35335268248 at ``09c4aac``, whose junit record shows ``09c4aac`` itself passed
    to ``code_changed_since``.
    """
    return (
        subprocess.run(
            ["git", "-C", str(Path(tcv.__file__).resolve().parents[1]),
             "rev-parse", "--is-shallow-repository"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        == "true"
    )


@pytest.mark.skipif(
    _repository_is_shallow(),
    reason=(
        "the repository is SHALLOW (git rev-parse --is-shallow-repository = true): with depth-1 "
        "history, git rev-list --max-parents=0 HEAD returns HEAD itself, so there is no root commit "
        "to compare against and the comparison would be HEAD against HEAD. Runs on a full clone."
    ),
)
def test_j1_code_changed_since_reads_real_git_history() -> None:
    """The real thing, not a monkeypatched stand-in: HEAD against itself, and against the root.

    A comparison of a commit with itself lists nothing; a comparison with the repository's first
    commit lists most of the tree.  If this ever returns ``[]`` for the root commit the helper is
    not reading history at all, and every reusability verdict built on it would be vacuous.

    ⚠️ **Skipped on a shallow checkout, and that is an environment condition, not a weakened
    assertion** -- see :func:`_repository_is_shallow`.  The two assertions below are unchanged and
    still run on every full clone, which is where this test can mean anything: the alternative,
    ``fetch-depth: 0`` in the workflow, would make every CI run clone the whole history for one
    test.
    """
    root_commit = subprocess.run(
        ["git", "-C", str(Path(tcv.__file__).resolve().parents[1]), "rev-list",
         "--max-parents=0", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.split()[0]

    assert tcv.code_changed_since(HEAD_SHA) == []
    assert tcv.code_changed_since(root_commit), "the root commit must differ from HEAD in code"
    assert all(not path.startswith("docs/") for path in tcv.code_changed_since(root_commit))


def test_j1_a_docs_only_difference_is_still_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """J1(c). *Commits may differ only by documentation.*

    A chunk rolled before the stage-1 artifact's own commit -- which is a docs commit -- is still
    the same computation, so the campaign must not re-roll 1,200 cells because a packet was
    written.  Nothing outside ``docs/`` may differ.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    cell = _cell("anchor", "fixedtime", 1000)
    keys = {"cell": cell, "out_root": roots.draws, "output_root": roots.output,
            "data_dir": roots.data}

    monkeypatch.setattr(
        tcv, "_paths_changed_between",
        lambda commit, other="HEAD": ["docs/returns/P7.3a-3.5b-3.6.md", "docs/data/x.json"],
        raising=True,
    )
    assert tcv.chunk_is_reusable(_payload(cell, roots), **keys) is True

    monkeypatch.setattr(
        tcv, "_paths_changed_between",
        lambda commit, other="HEAD": ["docs/x.md", "offline/transfer_curve.py"],
        raising=True,
    )
    assert tcv.chunk_is_reusable(_payload(cell, roots), **keys) is False


def test_j1_report_refuses_a_chunk_rolled_by_different_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """J1(c) at the artifact: a cell rolled by code that has since changed is not this code's cell.

    ``report`` re-derives the verdict rather than trusting ``chunk_is_reusable`` to have run: a
    resumed campaign can carry chunks a previous revision wrote, and the artifact claims they were
    all produced by the commit it records.
    """
    roots, _ = _campaign(tmp_path)
    monkeypatch.setattr(
        tcv, "_paths_changed_between",
        lambda commit, other="HEAD": ["offline/transfer_curve.py"],
        raising=True,
    )

    with pytest.raises(ValueError, match="offline/transfer_curve.py|outside docs"):
        _report(roots)
    assert list(roots.out.iterdir()) == []


def test_j1_the_artifact_records_the_chunk_commits_per_stage(tmp_path: Path) -> None:
    """J1(c): the artifact says which commits its cells were rolled by, per stage.

    The top-level ``git_commit`` is ``report``'s own, taken at write time; it says nothing about
    the code that produced the cells, and the two are routinely different (the stage-1 artifact is
    written by a docs commit).
    """
    roots, _ = _campaign(tmp_path)

    artifact = _report(roots)

    by_stage = artifact["chunk_commits_by_stage"]
    assert by_stage == {tcv.STAGE_CONFIRMATORY: [HEAD_SHA]}
    assert artifact["git_commit"] == HEAD_SHA


# ==================================================================================
# J2 / J3 -- the driver's signal and start-condition gaps
# ==================================================================================
def test_j2_the_trap_covers_hup_as_well_as_int_and_term() -> None:
    """J2. A closed tmux pane or a dropped SSH session sends SIGHUP, and nothing trapped it.

    Measured by the coordinator: SIGHUP during A17(f) exits 129 with the token consumed, the logs
    written and **neither FAILED nor COMPLETE** -- precisely the end state the trap-before-token
    rule exists to prevent, reached by the one signal the trap did not name.
    """
    code = _driver_code_text()
    assert "trap on_signal INT TERM HUP" in code


def test_j3_the_driver_refuses_when_sigint_is_ignored_on_entry() -> None:
    """J3 (reviewer min-2). A shell started with SIGINT ignored CANNOT trap it.

    ``bash`` does not let a non-interactive shell trap a signal that was ignored on entry, so
    Ctrl-C would do nothing and the pool would keep writing with the token already consumed.  The
    check is fail-closed and pre-token, like the group-leader check beside it.
    """
    code = _driver_code_text()
    assert "/proc/$$/status" in code
    assert "SigIgn" in code
    token = code.index('rm -f "$TOKEN"')
    assert code.index("SigIgn") < token, "the refusal must precede the token being consumed"
    assert "REFUSING TO START" in code[code.index("SigIgn") : token]


def test_j1d_the_driver_refuses_a_dirty_worktree_before_the_token() -> None:
    """J1(d). A multi-hour stage must not run from a tree that is being edited.

    One untracked file makes every chunk after it dirty, and J1(b) then refuses those cells one by
    one, hours in.  The driver checks once, before the token, and names the paths.
    """
    code = _driver_code_text()
    assert 'git -C "$WORK_TREE" status --porcelain' in code
    token = code.index('rm -f "$TOKEN"')
    assert code.index('git -C "$WORK_TREE" status --porcelain') < token
    assert "DIRTY" in code or "dirty" in code


def test_j3_a_failed_canary_gets_its_own_refusal_and_exit_two() -> None:
    """J3 (min-4). A failed canary must say *nothing has been consumed* and exit 2, not 1.

    ``fail()`` exits 1 and writes FAILED into the work directory -- an end state that says a run
    started.  A canary that refuses has consumed nothing, and its exit code must be the one every
    other pre-token refusal uses.
    """
    code = _driver_code_text()
    assert "if ! CANARY_LINE=$(" in code, "the canary's failure must be caught, not left to set -e"
    canary_block = code[code.index("if ! CANARY_LINE=$(") : code.index('rm -f "$TOKEN"')]
    assert "exit 2" in canary_block
    assert "REFUSING TO START" in canary_block
    assert "fail " not in canary_block, "fail() exits 1 and writes FAILED; this is a pre-token path"


def test_j3_the_header_states_what_was_measured_about_process_groups() -> None:
    """J3: the header's ``bash script.sh &`` sentence was not what the reviewer measured.

    A plain ``&`` from a non-interactive shell DID lead its own group; ``set +m`` is what produced
    the non-leader.  An assertion about what the driver SAYS may read the whole file, comments
    included -- that is the half ``_driver_code_text`` deliberately drops.
    """
    text = DRIVER.read_text(encoding="utf-8")
    assert "set +m" in text, "the corrected sentence names what actually produced a non-leader"
    assert "DID lead its own group" in text
    # The driver may QUOTE the old claim in order to correct it -- that is how a correction reads --
    # so what must be gone is the claim stated as fact, not the words.
    assert "pane gives and `bash script.sh &` does not" not in text


# ==================================================================================
# J3 -- the two Python minors
# ==================================================================================
def test_j3_run_cell_loads_the_calibration_through_the_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """J3 (min-6). ``run_cell`` read the committed artifact from the repository, not from data_dir.

    The checkpoint pins already take ``data_dir``; the targets did not, so a run pointed at a
    different ``--data-dir`` would pin its checkpoints against one set of records and read its
    prompts from another.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    tampered = roots.data / "p7_2b_calibration.json"
    tampered.write_bytes(tampered.read_bytes() + b"\n")
    _install_cell_seams(monkeypatch)

    with pytest.raises(ValueError, match="sha256"):
        tcv.run_cell(
            _cell("dt", "b_mean_k100", 1000, subject="mappo1000", seed=101),
            out_root=roots.draws,
            output_root=roots.output,
            data_dir=roots.data,
            canary_seconds=0.9,
        )


def test_j3_run_pilot_prints_the_canary_line_before_it_checks_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """J3 (min-7). Same rule as the ``canary`` subcommand since H8: print, then check.

    The pilot's transcript is written after the run, so on a failed correctness half the pane is
    the only place the observed values would appear -- unless the line is printed first.
    """
    monkeypatch.setattr(
        "offline.transfer_calibration.canary_seconds",
        lambda: (0.88, {"decisions": 360, "local_return": -1.0, "att_horizon": 2.0,
                        "two_routes_agree": True}),
        raising=True,
    )

    with pytest.raises(ValueError, match="canary"):
        tcv.run_pilot(
            work_dir=tmp_path / "pilot",
            out_root=tmp_path / "draws",
            output_root=tmp_path / "output",
            data_dir=tmp_path / "data",
        )

    assert "canary 0.88 s" in capsys.readouterr().out


# ==================================================================================
# K round, Finding 1 -- the env-ATT denominator: a diagnostic, a carry-on, and a caveat
# (author's ruling, PROJECT_PLAN Decisions Log 2026-09-17)
# ==================================================================================
def _zero_env_denominator_campaign(tmp_path: Path) -> tuple[_Roots, list[dict[str, Any]]]:
    """A campaign whose third draw has an EXACTLY equal pair of env-ATT anchors.

    Draw 1002's ``att_env`` is 300.0 for both anchors -- the case P7.1's frozen numbers say is
    reachable on this definition (fixed-time minus MaxPressure was -1.50 s at seed 1000) -- while
    its ``e_sumo`` anchors differ by 200.0, so the registered primary is unaffected and keeps its
    own refusal for its own zero.
    """
    draws = (*_DEMO_DRAWS, 1002)
    roots = _build_roots(tmp_path, draws=draws, seeds=_DEMO_SEEDS)
    roots.work.mkdir(parents=True, exist_ok=True)
    roots.out.mkdir(parents=True, exist_ok=True)
    _canary(roots.work)

    declared = [*_DEMO_CELLS]
    for subject in ("mappo1000", "mix50"):
        for seed in _DEMO_SEEDS:
            declared.append(
                _cell("dt", "b_mean_k100", 1002, subject=subject, seed=seed)
            )
    declared += [_cell("anchor", arm, 1002) for arm in ("fixedtime", "maxpressure")]

    for cell in _DEMO_CELLS:
        tcv.write_chunk(_payload(cell, roots), work_dir=roots.work)
    # draw 1002, built by hand: equal env ATTs, different e_sumo
    env_att = {"fixedtime": 300.0, "maxpressure": 300.0}
    e_sumo = {"fixedtime": 500.0, "maxpressure": 300.0}
    for cell in declared:
        if int(cell["draw_id"]) != 1002:
            continue
        arm = str(cell["arm"])
        tcv.write_chunk(
            _payload(
                {**cell, "draw_id": 1000},
                roots,
                draw_id=1002,
                config_sha256=roots.cfg_sha[1002],
                routes_sha256=roots.routes_sha[1002],
                # draw 1002 is not Amendment C2's declared subset, so a real chunk there carries
                # the flag off and the three halting fields absent
                halting_checked=False,
                halting_max_abs_difference=None,
                halting_n_lane_seconds=None,
                halting_n_disagreeing_lane_seconds=None,
                att_env=env_att.get(arm, 320.0),
                att_horizon=env_att.get(arm, 320.0),
                e_sumo=e_sumo.get(arm, 400.0),
            ),
            work_dir=roots.work,
        )
    return roots, declared


def test_k_finding1_an_exactly_zero_env_denominator_is_recorded_excluded_and_reported(
    tmp_path: Path,
) -> None:
    """The author's ruling: ``report`` records it, excludes it, reports the n, and CARRIES ON.

    A report-time crash would not cost the campaign's compute -- the chunks persist and ``report``
    re-runs in minutes -- but it would mean the handling of an undefined ratio got decided AFTER
    the numbers existed, which is the forking path the registration exists to close.  So it is
    decided here, before any cell of the campaign runs.
    """
    roots, declared = _zero_env_denominator_campaign(tmp_path)

    artifact = _report(roots, cells=declared)

    block = artifact["rho"]["definitions"]["att_env"]
    excluded = block["excluded_draws"]
    assert [row["draw_id"] for row in excluded] == [1002]
    assert excluded[0]["att_fixedtime"] == 300.0
    assert excluded[0]["att_maxpressure"] == 300.0

    # every cell on that draw carries a null rho on THIS definition and a real one on the primary
    on_1002 = [row for row in artifact["cells"] if row["draw_id"] == 1002]
    assert on_1002, "the draw's cells are still published"
    assert all(row["rho_att_env"] is None for row in on_1002)
    assert all(isinstance(row["rho_e_sumo"], float) for row in on_1002)

    # ...and it is excluded from the means, the CI and the reported n
    entry = [
        e for e in artifact["rho"]["by_subject_arm"]
        if e["subject"] == "mappo1000" and e["arm"] == "b_mean_k100"
    ][0]
    assert entry["att_env"]["n_draws"] == 2, "three draws, one undefined, two used"
    assert entry["e_sumo"]["n_draws"] == 3, "the primary keeps every draw"
    assert block["n_draws_used"] == 2
    per_seed = [
        row for row in artifact["rho"]["by_subject_arm_seed"]
        if row["subject"] == "mappo1000" and row["seed"] == 101
    ][0]
    assert per_seed["n_draws_att_env"] == 2
    assert per_seed["n_draws"] == 3


def test_k_finding1_a_zero_denominator_on_the_registered_primary_is_still_refused(
    tmp_path: Path,
) -> None:
    """``E_sumo`` keeps its refusal: on the nominal anchors it is 206.5-252.6 s.

    A zero there is a finding about the instrument, not a property of a draw, so the carry-on rule
    does not extend to it -- the coordinator's reading (i), stated so it could be corrected.
    """
    draws = (*_DEMO_DRAWS, 1002)
    roots = _build_roots(tmp_path, draws=draws, seeds=_DEMO_SEEDS)
    roots.work.mkdir(parents=True, exist_ok=True)
    roots.out.mkdir(parents=True, exist_ok=True)
    _canary(roots.work)
    declared = [*_DEMO_CELLS, *[_cell("anchor", arm, 1002) for arm in ("fixedtime", "maxpressure")]]
    declared.append(_cell("dt", "b_mean_k100", 1002, subject="mappo1000", seed=101))
    for cell in _DEMO_CELLS:
        tcv.write_chunk(_payload(cell, roots), work_dir=roots.work)
    for cell in declared:
        if int(cell["draw_id"]) != 1002:
            continue
        tcv.write_chunk(
            _payload(
                {**cell, "draw_id": 1000}, roots,
                draw_id=1002,
                config_sha256=roots.cfg_sha[1002],
                routes_sha256=roots.routes_sha[1002],
                # draw 1002 is not Amendment C2's declared subset, so a real chunk there carries
                # the flag off and the three halting fields absent
                halting_checked=False,
                halting_max_abs_difference=None,
                halting_n_lane_seconds=None,
                halting_n_disagreeing_lane_seconds=None,
                att_env=300.0 if str(cell["arm"]) == "fixedtime" else 200.0,
                att_horizon=300.0 if str(cell["arm"]) == "fixedtime" else 200.0,
                e_sumo=400.0,  # identical on BOTH anchors -> the primary's denominator is zero
            ),
            work_dir=roots.work,
        )

    with pytest.raises(ValueError, match="denominator is zero"):
        _report(roots, cells=declared)
    assert list(roots.out.iterdir()) == []


def test_k_finding1_the_denominator_diagnostic_is_computed_for_both_definitions(
    tmp_path: Path,
) -> None:
    """The diagnostic the caveat tells a reader to read rho beside, from the ANCHOR chunks.

    n draws, min, max, how many denominators are non-positive, how many are smaller than a second,
    and the draw ids of the non-positive ones -- because a per-draw ratio with a denominator this
    small can be dominated by a few draws and can change sign.
    """
    roots, declared = _zero_env_denominator_campaign(tmp_path)

    artifact = _report(roots, cells=declared)

    env = artifact["rho"]["definitions"]["att_env"]["denominator_diagnostic"]
    assert env["n_draws"] == 3
    assert env["min"] == 0.0
    assert env["max"] == 256.0
    assert env["n_non_positive"] == 1
    assert env["n_below_one_second"] == 1
    assert env["draw_ids_non_positive"] == [1002]

    primary = artifact["rho"]["definitions"]["e_sumo"]["denominator_diagnostic"]
    assert primary["n_draws"] == 3
    assert primary["n_non_positive"] == 0
    assert primary["min"] == 200.0


def test_k_finding1_the_caveat_is_in_the_artifact_and_on_every_env_h3_clause(
    tmp_path: Path,
) -> None:
    """The caveat travels with the ARTIFACT, because the packet does not travel with it.

    Verbatim in the env-ATT block and on every H3 clause computed on ``att_env`` -- a reader who
    opens ``docs/data/p7_3a_zero_shot.json`` and looks at one clause must see it there, not in a
    document they were never given.
    """
    roots, declared = _zero_env_denominator_campaign(tmp_path)

    artifact = _report(roots, cells=declared)

    caveat = artifact["rho"]["definitions"]["att_env"]["caveat"]
    assert caveat == tcv.ATT_ENV_CAVEAT
    assert "CO-REPORTED, NOT THE REGISTERED PRIMARY (A15)" in caveat
    assert "-1.50, +2.08, +4.92, +13.14, +18.70 s" in caveat
    assert "negative at seed 1000" in caveat
    assert "+206.5 to +252.6 s" in caveat
    assert "denominator_diagnostic" in caveat

    env_clauses = [c for c in artifact["h3"]["clauses"] if c["definition"] == "att_env"]
    assert len(env_clauses) == 2
    assert all(clause["caveat"] == tcv.ATT_ENV_CAVEAT for clause in env_clauses)
    primary_clauses = [c for c in artifact["h3"]["clauses"] if c["definition"] == "e_sumo"]
    assert all("caveat" not in clause for clause in primary_clauses), (
        "the registered primary carries no such caveat; attaching one would blur which is which"
    )


# ==================================================================================
# K round, Finding 2 -- artifacts leave the worktree
# ==================================================================================
def test_k_finding2_the_driver_writes_artifacts_outside_the_worktree(tmp_path: Path) -> None:
    """Finding 2: the driver wrote the stage-1 artifact INTO the tree J1(d) then refused.

    Measured by the coordinator: with ``docs/data/p7_3a_zero_shot_stage1.json`` present and not
    gitignored, the delivered driver refused stage 2 before its own token -- the coordinator's
    J1(d)/(e) ruling colliding with the driver's own output path.  Artifacts now go to
    ``$WORK/artifacts`` (inside the manifest), inputs are still READ from the run worktree's
    committed ``docs/data``, and the two artifacts are copied into the task branch by hand.
    """
    code = _driver_code_text()

    assert "ARTIFACTS=$WORK/artifacts" in code
    assert '--out-dir "$ARTIFACTS"' in code
    assert '--data-dir "$DATA"' in code, "inputs still come from the committed docs/data"
    assert 'DATA=$WORK_TREE/docs/data' in code
    assert '--out-dir "$DATA"' not in code, "the worktree is never the output directory"
    assert '--stage1-path "$ARTIFACTS/p7_3a_zero_shot_stage1.json"' in code
    assert 'mkdir -p "$ARTIFACTS"' in code
    token = code.index('rm -f "$TOKEN"')
    assert code.index('mkdir -p "$ARTIFACTS"') > token, "created only after the token"

    # The NEXT block tells the operator to bring the artifacts into the branch by hand. Asserted as
    # the INSTRUCTION, not as one command form: the driver writes `git -C <worktree> add ...`, and a
    # needle demanding the literal "git add" would be a claim about spelling.
    next_block = DRIVER.read_text(encoding="utf-8").split("=== NEXT (manual)")[1]
    assert "docs/data" in next_block
    assert "$WORK/artifacts" in next_block, "the operator is told where the artifacts actually are"
    assert "commit" in next_block
    assert "p7_3a_zero_shot_stage1.json" in next_block and "p7_3a_zero_shot.json" in next_block


# ==================================================================================
# K round, Finding 3 -- the dirty-tree refusal, EXECUTED
# ==================================================================================
MAIN_INTERPRETER = Path("/home/filip/rltraffic/.venv/bin/python")


@pytest.mark.skipif(
    not MAIN_INTERPRETER.is_file(),
    reason=f"needs the main tree's interpreter at {MAIN_INTERPRETER}",
)
def test_k_finding3_the_delivered_driver_refuses_a_dirty_tree_when_executed(
    tmp_path: Path,
) -> None:
    """Finding 3: neutralising the dirty check to ``if false`` left all 99 tests green.

    T9 pins the driver's TEXT, and a text assertion cannot see a behaviour removed from the branch
    it describes -- the failure mode this project has logged more than any other.  So this one
    EXECUTES the delivered driver, with the roots redirected and nothing else changed, against a
    throwaway clone that is dirty by exactly one untracked file.

    ``git clone --shared`` and a scratch ``WORK`` keep it off every real path; ``setsid --wait``
    makes it a process-group leader, which the driver requires before it reaches this check.
    """
    repo = Path(tcv.__file__).resolve().parents[1]
    clone = tmp_path / "tree"
    work = tmp_path / "work"
    worktrees_before = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list"], capture_output=True, text=True, check=True
    ).stdout

    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(repo), str(clone)],
        capture_output=True, text=True, check=True,
    )
    subprocess.run(["git", "-C", str(clone), "checkout", "HEAD"], capture_output=True, check=True)

    # the DELIVERED driver text, with only the roots redirected
    delivered = DRIVER.read_text(encoding="utf-8")
    redirected = delivered.replace("WORK=$MAIN/output/p7_3a", f"WORK={work}")
    assert redirected != delivered, "the WORK root must actually have been redirected"
    target = clone / "offline" / "campaigns" / "p7_3a_zero_shot.sh"
    target.write_text(redirected, encoding="utf-8")

    # commit the redirection so the clone is CLEAN, then make it dirty by exactly one file
    subprocess.run(["git", "-C", str(clone), "add", "-A"], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(clone), "-c", "user.email=t@t", "-c", "user.name=t",
         "-c", "core.hooksPath=/dev/null", "commit", "-m", "roots redirected"],
        capture_output=True, check=True,
    )
    assert subprocess.run(
        ["git", "-C", str(clone), "status", "--porcelain"], capture_output=True, text=True
    ).stdout == "", "the clone must be clean before the one untracked file is added"
    (clone / "untracked_probe.txt").write_text("one untracked file\n", encoding="utf-8")

    result = subprocess.run(
        ["setsid", "--wait", "bash", str(target), "confirmatory"],
        capture_output=True, text=True, cwd=str(clone),
    )

    assert result.returncode == 2, (
        f"expected the dirty-tree refusal's exit 2, got {result.returncode}:\n{result.stderr}"
    )
    assert "REFUSING TO START: the worktree" in result.stderr
    assert "DIRTY" in result.stderr
    assert "untracked_probe.txt" in result.stderr, "the refusal names the path that made it dirty"
    assert not work.exists(), "a refused start creates nothing under WORK"
    worktrees_after = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list"], capture_output=True, text=True, check=True
    ).stdout
    assert worktrees_after == worktrees_before, "the repository's worktree list must be untouched"


# ==================================================================================
# BRIEF_38 §2 -- THE FIVE SEAMS MERGE REVIEWER A FOUND UNPINNED, AND FINDING 4'S FIX
#
# Every one of the five is CORRECT today.  Reviewer A's point is that no test would
# notice if it changed: each mutation below left 105 or 173 tests green.  P7.3b runs a
# different checkpoint per cell, so pins 1, 2 and 5 are the ones that bite first
# (`docs/returns/P7.3a-3.5b-3.6.md` §19.14).
#
# ⚠️ These five tests are GREEN on delivery by construction -- they pin behaviour that
# already holds.  Their evidence is the MUTATION, executed and pasted in the packet;
# "it passes" proves nothing here.  Finding 4's test is the one that is red first.
# ==================================================================================


def _dt_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rtg_series: list[float],
    facts: Any,
    att: float = 366.5,
) -> None:
    """``_install_cell_seams`` plus the two seams a **dt** cell needs that an anchor does not.

    Two differences from the anchor path, and both are deliberate rather than convenient:

    * the env is an ``AlignedEnv`` **by class**, because Amendment C8's real refusal
      (:func:`assert_env_matches_cell`) is left in the path -- a dt cell on a raw env must still
      raise, and faking the check away would remove the very door A16 registers;
    * ``dt_choose`` is replaced by one that emits *rtg_series*, so the in-support diagnostic gets a
      conditioning trajectory whose counts are known in advance and can be asserted with ``==``.
    """
    from offline.aligned_env import AlignedEnv

    inner_cls = _install_cell_seams(monkeypatch, att=att)
    inner = inner_cls()

    class _AlignedFake(AlignedEnv):
        """An ``AlignedEnv`` by class with no SUMO process behind it.

        ``AlignedEnv.__init__`` wraps a real ``SumoEnv``; this subclass sets the two attributes the
        parent's properties and ``__getattr__`` read, and delegates ``reset``/``step`` RAW so
        ``align_info`` is not applied to an already-shaped fake info.
        """

        def __init__(self, env: Any) -> None:  # noqa: D107 - see the class docstring
            self._env = env
            self._alignment = None

        def reset(self, **kwargs: Any) -> dict[str, Any]:
            return self._env.reset(**kwargs)

        def step(self, action: Any) -> tuple[Any, bool, bool, dict[str, Any]]:
            return self._env.step(action)

    def fake_dt_choose(env: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        diagnostics: dict[str, Any] = {"rtg_series": [], "reward_series": [], "actions": []}
        remaining = iter(rtg_series)

        def choose(_env: Any, info: Mapping[str, Any]) -> int:
            diagnostics["rtg_series"].append(float(next(remaining)))
            diagnostics["reward_series"].append(0.0)
            diagnostics["actions"].append(0)
            return 0

        return choose, diagnostics

    monkeypatch.setattr(tcv, "env_for_cell", lambda cell, **k: _AlignedFake(inner), raising=True)
    monkeypatch.setattr(tcv, "dt_choose", fake_dt_choose, raising=True)
    monkeypatch.setattr(
        "offline.transfer_calibration.subject_facts", lambda s, **k: facts, raising=True
    )


# ----------------------------------------------------------------------------------
# PIN 1 -- transfer_curve.py:1104, the engine seed HANDED TO horizon_rollout
# ----------------------------------------------------------------------------------
def test_pin1_the_engine_seed_reaching_horizon_rollout_is_a18cs_1000(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewer A major 1. ``ENGINE_SEED + 1`` at the call site left 173 tests passing.

    The chunk records the CONSTANT (``"engine_seed_requested": ENGINE_SEED``) and
    ``validate_cell_payload`` compares it with the same constant, so that check is a tautology with
    respect to this seam: a wrong seed would publish as 1000.  The only way to pin it is to observe
    the argument the call actually passes, which is what the spy below does.

    Both directions are asserted, and that is the point: the literal **1000** is A18(c)'s registered
    seed, so a change to ``ENGINE_SEED`` itself fails here too, not only a change to the call site.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    _install_cell_seams(monkeypatch)

    import offline.horizon_metric as horizon_metric

    installed = horizon_metric.horizon_rollout  # the seam's own fake, already in place
    seen: list[tuple[int, int]] = []

    def spy(env: Any, choose: Any, episodes: int, seed: int) -> Any:
        seen.append((int(episodes), int(seed)))
        return installed(env, choose, episodes, seed)

    monkeypatch.setattr("offline.horizon_metric.horizon_rollout", spy, raising=True)

    chunk = tcv.run_cell(
        _cell("anchor", "fixedtime", 1000),
        out_root=roots.draws,
        output_root=roots.output,
        data_dir=roots.data,
    )

    assert seen == [(1, 1000)], (
        "run_cell must hand horizon_rollout exactly one episode at engine seed 1000 (A18(c)); "
        f"it handed {seen}"
    )
    assert seen[0][1] == tcv.ENGINE_SEED, "the call site and the constant must not drift apart"
    # ...and the chunk's record agrees with what was actually passed, which is the claim the
    # artifact makes about all 4,700 cells.
    assert chunk["engine_seed_requested"] == seen[0][1]


# ----------------------------------------------------------------------------------
# PIN 2 -- transfer_curve.py:925, the rtg_first == target_rtg refusal
# ----------------------------------------------------------------------------------
def test_pin2_a_dt_chunk_whose_first_rtg_is_not_the_declared_target_is_refused(
    tmp_path: Path,
) -> None:
    """Reviewer A major 2. ``if False:`` on this refusal left 105 tests passing.

    It is the ONLY guard that the prompt took effect, and it is exactly what a checkpoint with its
    own prompt changes -- P7.3b's anchor is that case.  The perturbation is ONE ULP, so the test
    also pins that the comparison is ``==`` and not a tolerance: ``math.nextafter`` gives the
    smallest float that is not the target, and it must still be refused.
    """
    import math

    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    cell = _cell("dt", "b_mean_k100", 1000, subject="mappo1000", seed=101)

    good = _payload(cell, roots)
    tcv.validate_cell_payload(good, cell=cell)  # the control: unperturbed, it passes

    target = float(good["target_rtg"])
    one_ulp_away = math.nextafter(target, math.inf)
    assert one_ulp_away != target, "the fixture must perturb the value, not restate it"

    with pytest.raises(ValueError, match="the prompt did not take effect"):
        tcv.validate_cell_payload(_payload(cell, roots, rtg_first=one_ulp_away), cell=cell)

    # A missing first RTG is not a pass either: None is not the declared target.
    with pytest.raises(ValueError, match="the prompt did not take effect"):
        tcv.validate_cell_payload(_payload(cell, roots, rtg_first=None), cell=cell)


# ----------------------------------------------------------------------------------
# PIN 3 -- transfer_curve.py:1746, _paired_block's ATT key
# ----------------------------------------------------------------------------------
def test_pin3_the_paired_comparison_uses_the_att_definition_it_was_asked_for(
    tmp_path: Path,
) -> None:
    """Reviewer A major 3. Forcing the key to ``att_env`` left 105 tests passing.

    **The registered H3 test row IS this paired comparison**, so a block labelled ``e_sumo`` that
    silently compared env ATT would misreport the confirmatory test itself.  Reached end to end
    through ``report`` rather than by calling the private helper, so the label and the arithmetic
    are pinned together on the object that gets committed.

    The fixture puts ``e_sumo`` exactly 20.0 above ``att_env`` on EVERY cell, so the two blocks must
    differ by exactly that -- an exact binary fraction, compared with ``==``.
    """
    roots, _ = _campaign(tmp_path)
    artifact = _report(roots)

    for entry in artifact["rho"]["by_subject_arm"]:
        for anchor in ("fixedtime", "maxpressure"):
            primary = entry["paired_att"]["e_sumo"][anchor]
            co_reported = entry["paired_att"]["att_env"][anchor]

            assert primary["att_definition"] == "e_sumo"
            assert co_reported["att_definition"] == "att_env"
            # Every cell's e_sumo is att_env + 20.0, so BOTH sides of the pairing shift by 20.0 and
            # the DIFFERENCE is unchanged -- which is why the means, not the difference, are what
            # distinguishes the two definitions here.
            assert primary["mean_left"] == co_reported["mean_left"] + 20.0, (
                f"{entry['subject']}/{entry['arm']} vs {anchor}: the e_sumo block reports "
                f"{primary['mean_left']}, the att_env block {co_reported['mean_left']}"
            )
            assert primary["mean_right"] == co_reported["mean_right"] + 20.0
            assert primary["mean_left"] != co_reported["mean_left"]


# ----------------------------------------------------------------------------------
# PIN 4 -- transfer_curve.py:1128-1130, the in-support diagnostic's bounds
# ----------------------------------------------------------------------------------
def test_pin4_the_in_support_bounds_are_the_declared_min_then_max_in_that_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewer A major 4. Swapping ``rtg_min``/``rtg_max`` left 105 tests passing -- because no
    test reached this line at all: every ``run_cell`` test in the file runs an ANCHOR cell, and the
    in-support block is on the ``dt`` branch.

    The conditioning trajectory below is built so the three counts are known before the call and
    partition 360 exactly, so a bound that moved changes a number rather than a shape.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    cell = _cell("dt", "b_mean_k100", 1000, subject="mappo1000", seed=101)

    # mappo1000's b_mean_k100 target, read from the sha-pinned artifact rather than restated.
    target = float(
        tcv.targets_for_subject("mappo1000", tcv.load_calibration(roots.data / "p7_2b_calibration.json"))[
            "b_mean_k100"
        ]["target_rtg"]
    )
    low, high = -25257.0, -71.0
    # 1 + 100 + 200 + 59 = 360 = EXPECTED_DECISIONS.  The first value must be the target, because
    # PIN 2's refusal is left in the path.
    series = [target] + [-30000.0] * 100 + [-10000.0] * 200 + [500.0] * 59
    assert len(series) == 360

    facts = tc.SubjectFacts(
        subject="mappo1000",
        best_source_return=-5762.0,
        rtg_scale=9991.0,
        support_range_over_the_split=(low, high),
        n_rows=216000,
        training_set_return_min=-9991.0,
        checkpoints=("fabricated",),
        state_dim=25,
        context_length=20,
    )
    _dt_seams(monkeypatch, rtg_series=series, facts=facts)

    chunk = tcv.run_cell(
        cell, out_root=roots.draws, output_root=roots.output, data_dir=roots.data
    )
    counts = chunk["in_support_counts"]

    assert chunk["support_range"] == [low, high], "the range is published low-then-high"
    assert counts["support_range"] == [low, high]
    # Recomputed here by a route that does not call in_support_counts: the interval is CLOSED at
    # both ends and below/above are strict.
    assert counts["n_decisions_below"] == sum(1 for v in series if v < low) == 100
    assert counts["n_decisions_above"] == sum(1 for v in series if v > high) == 59
    assert counts["n_decisions_in_support"] == sum(1 for v in series if low <= v <= high) == 201
    assert (
        counts["n_decisions_below"]
        + counts["n_decisions_above"]
        + counts["n_decisions_in_support"]
        == 360
    )
    # The bound that is NOT the support range: -rtg_scale, the training set the target came from.
    assert counts["training_set_return_min"] == -9991.0


# ----------------------------------------------------------------------------------
# PIN 5 -- transfer_curve.py:2067, run_stage's resume-by-CONTENT
# ----------------------------------------------------------------------------------
def test_pin5_run_stage_decides_reuse_by_content_and_never_by_the_files_existence(
    tmp_path: Path,
) -> None:
    """Reviewer A major 5. Replacing the call with ``path.exists()`` -- the ``p5_3b.sh`` ``[ -f ]``
    defect this function's own docstring cites -- left 105 tests passing.

    Two chunks are on disk and BOTH exist.  One is valid; the other records a teleport, which
    ``validate_cell_payload`` refuses, so it is not reusable.  Under ``path.exists()`` the second is
    skipped as complete, ``failed/`` is never created, and the campaign reports a cell it did not
    check.  Three independent observables are asserted so the mutant cannot survive any one of them.

    The unusable cell names draw 1002, which has no parity directory under the fixture's draws root,
    so re-rolling it fails in ``demand_identity`` **before an env is built** -- no simulator runs.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    roots.work.mkdir(parents=True, exist_ok=True)

    good = _cell("anchor", "fixedtime", 1000)
    unusable = _cell("anchor", "fixedtime", 1002)

    tcv.write_chunk(_payload(good, roots), work_dir=roots.work)
    # draw 1002 has no fixture files, so borrow 1000's digests: the chunk must be well-formed
    # enough that ONLY the teleport makes it unusable.
    doomed = _payload(good, roots, draw_id=1002, n_teleports=1)
    tcv.write_chunk(doomed, work_dir=roots.work)

    both = [tcv.chunk_path(good, work_dir=roots.work), tcv.chunk_path(unusable, work_dir=roots.work)]
    assert all(path.is_file() for path in both), "both chunks must EXIST before the call"

    result = tcv.run_stage(
        work_dir=roots.work,
        out_root=roots.draws,
        output_root=roots.output,
        data_dir=roots.data,
        cells=[good, unusable],
        workers=1,
    )

    assert result["n_reused"] == 1, (
        "only the chunk that re-derives its own verdict may be reused; "
        f"got n_reused={result['n_reused']} over two files that both exist"
    )
    assert result["n_failed"] == 1, "the unusable cell must be RE-ROLLED, not skipped"
    assert (roots.work / "failed" / both[1].name).is_file(), (
        "an unusable chunk is moved into failed/ -- it is evidence about a run and is never "
        "overwritten"
    )
    assert not both[1].exists(), "and it is no longer where report's glob would find it"


# ----------------------------------------------------------------------------------
# FINDING 4 -- the all-excluded env-ATT path must CARRY ON, as :1526 already claims
# ----------------------------------------------------------------------------------
def _all_env_denominators_zero(tmp_path: Path) -> _Roots:
    """A campaign where every draw's env-ATT denominator is EXACTLY zero and e_sumo's is not.

    The two anchors are given equal ``att_env`` per draw -- which P7.1 measured as reachable at
    engine seed 1000, where fixed-time minus MaxPressure was **-1.50 s** -- while their ``e_sumo``
    stays 256.0 apart, so the registered primary is unaffected and only the co-reported definition
    is excluded.  That is the situation the 2026-09-17 ruling describes.
    """
    roots = _build_roots(tmp_path, draws=_DEMO_DRAWS, seeds=_DEMO_SEEDS)
    roots.work.mkdir(parents=True, exist_ok=True)
    roots.out.mkdir(parents=True, exist_ok=True)
    _canary(roots.work)

    env_att = {1000: 300.0, 1001: 310.0}
    e_sumo = {
        ("fixedtime", 1000): 700.0, ("maxpressure", 1000): 444.0,
        ("fixedtime", 1001): 710.0, ("maxpressure", 1001): 454.0,
    }
    for cell in _DEMO_CELLS:
        draw = int(cell["draw_id"])
        arm = str(cell["arm"])
        att = env_att[draw]
        value = e_sumo.get((arm, draw), e_sumo[("maxpressure", draw)] + 64.0)
        tcv.write_chunk(
            _payload(cell, roots, att_env=att, att_horizon=att, e_sumo=value),
            work_dir=roots.work,
        )
    return roots


def test_finding4_report_carries_on_when_every_env_att_denominator_is_zero(
    tmp_path: Path,
) -> None:
    """Finding 4. ``transfer_curve.py:1526`` says the all-excluded case is *"Reported as such and
    carried past"*.  It was not: ``_h3_block`` evaluated ``stats["mean"] > 0.0`` on ``None`` and
    ``_contrast_block`` subtracted two ``None``s, and both raised ``TypeError`` -- **after** every
    refusal had passed, i.e. at the point where the artifact was about to be written.

    The registered primary is untouched here, which is the whole point: a definition the author
    ruled *recorded and excluded* must not take the other one down with it.
    """
    roots = _all_env_denominators_zero(tmp_path)

    artifact = _report(roots)  # must not raise

    definition = artifact["rho"]["definitions"]["att_env"]
    assert definition["n_draws_used"] == 0
    assert definition["n_draws_total"] == len(_DEMO_DRAWS)
    assert sorted(entry["draw_id"] for entry in definition["excluded_draws"]) == list(_DEMO_DRAWS)

    for entry in artifact["rho"]["by_subject_arm"]:
        assert entry["att_env"]["mean"] is None
        assert entry["att_env"]["n_draws"] == 0
        assert "why_empty" in entry["att_env"]
        # the REGISTERED PRIMARY is unaffected -- it has a denominator of 256.0 on every draw
        assert entry["e_sumo"]["mean"] is not None
        assert entry["e_sumo"]["n_draws"] == len(_DEMO_DRAWS)

    # H3's clauses on the excluded definition report None rather than a verdict, and the clauses on
    # the primary are unchanged.
    by_definition: dict[str, list[dict[str, Any]]] = {}
    for clause in artifact["h3"]["clauses"]:
        by_definition.setdefault(str(clause["definition"]), []).append(clause)
    for clause in by_definition["att_env"]:
        assert set(clause["point_estimate_satisfies"].values()) == {None}
        assert set(clause["ci95_entirely_satisfies"].values()) == {None}
        assert set(clause["mean_rho"].values()) == {None}
        assert clause["caveat"] == tcv.ATT_ENV_CAVEAT
    for clause in by_definition["e_sumo"]:
        assert set(clause["point_estimate_satisfies"].values()) <= {True, False}
        assert None not in clause["mean_rho"].values()


def test_finding4_the_contrast_block_carries_a_none_through_instead_of_subtracting_it() -> None:
    """The second half of Finding 4, reached directly because ``_DEMO_CELLS`` declares only the
    registered arm and the contrast needs ``naive`` beside it.

    ``None - None`` is a ``TypeError``; the block must report the two means and say the difference
    is undefined, without inventing a zero.
    """
    def entry(arm: str, mean_e: float | None, mean_env: float | None) -> dict[str, Any]:
        return {
            "subject": "mappo1000",
            "arm": arm,
            "e_sumo": {"mean": mean_e},
            "att_env": {"mean": mean_env},
        }

    block = tcv._contrast_block(
        [entry("b_mean_k100", 1.5, None), entry("naive", 1.25, None)]
    )
    primary = block["by_subject"]["mappo1000"]["e_sumo"]
    excluded = block["by_subject"]["mappo1000"]["att_env"]

    assert primary["difference"] == 0.25
    assert excluded["calibrated_mean_rho"] is None
    assert excluded["naive_mean_rho"] is None
    assert excluded["difference"] is None, "an undefined difference is None, never 0.0"


def test_finding4_the_comment_at_the_exclusion_branch_states_the_measured_behaviour() -> None:
    """The comment is part of the fix. It asserted the case was *carried past* while the code
    raised; a comment that describes behaviour the code does not have is how the next reader is
    misled into trusting the path.

    Pinned by the words a correction must keep, not by a line number: the file is edited often.
    """
    source = Path(tcv.__file__).read_text(encoding="utf-8")
    marker = "Every draw undefined on this definition."
    assert marker in source, "the exclusion branch's comment is gone; this test names the branch"
    window = source[source.index(marker) : source.index(marker) + 1200]
    assert "_h3_block" in window and "_contrast_block" in window, (
        "the corrected comment must name the two blocks that used to raise TypeError here"
    )
    assert "TypeError" in window, "and it must say what the failure was"


# ==================================================================================
# BRIEF_38 §3.4 + §3.5 -- THE ANCHOR AS A SUBJECT, AND ITS DRIVER
#
# A18(a)'s full-retrain anchor is the curve's k = 200 endpoint. Everything scientific
# about it is registered; these tests assert that what runs is what was registered,
# that its prompt cannot come from P7.2b's calibration artifact, and -- Amendment A1 --
# that adding it cannot disturb anything P7.3a already published.
# ==================================================================================

ANCHOR_TRAINING = REPO_DATA / "p7_3b_anchor_training.json"


# ----------------------------------------------------------------------------------
# T14 -- Amendment A1: the existing declaration is untouched
# ----------------------------------------------------------------------------------
def test_t14_declared_cells_none_is_still_p7_3as_four_thousand_seven_hundred() -> None:
    """Amendment A1, and it is the reason the anchor is a SEPARATE declaration.

    ``declared_cells(None)`` must still be P7.3a's campaign, and its two stages must still
    partition that set ELEMENT FOR ELEMENT -- not merely add up to 4,700. The anchor's 700 cells
    are reachable only through the anchor stage by name.
    """
    whole = tcv.declared_cells(None)
    confirmatory = tcv.declared_cells(tcv.STAGE_CONFIRMATORY)
    rest = tcv.declared_cells("rest")

    assert len(whole) == 4700
    assert len(confirmatory) == 1200
    assert len(rest) == 3500

    # element for element, in order, not merely by count
    assert confirmatory + rest == [c for c in whole if c["stage"] == tcv.STAGE_CONFIRMATORY] + [
        c for c in whole if c["stage"] == "rest"
    ]
    names = [tcv.cell_chunk_name(c) for c in whole]
    assert sorted(names) == sorted(
        [tcv.cell_chunk_name(c) for c in confirmatory]
        + [tcv.cell_chunk_name(c) for c in rest]
    )
    assert len(set(names)) == 4700, "the campaign's cell names are unique"

    # ...and not one anchor cell is reachable from the whole declaration
    assert {str(c["stage"]) for c in whole} == {tcv.STAGE_CONFIRMATORY, "rest"}
    assert tcv.ANCHOR_SUBJECT not in {str(c["subject"]) for c in whole}
    assert tcv.ANCHOR_ARM not in {str(c["arm"]) for c in whole}


def test_t14_the_anchor_stage_is_reachable_only_by_name() -> None:
    """The other half of A1: 700 cells, and they are the ones A18(a) and Amendment A4 declare."""
    cells = tcv.declared_cells(tcv.STAGE_ANCHOR)
    assert len(cells) == 700
    assert cells == tcv.anchor_cells()

    dt = [c for c in cells if c["kind"] == "dt"]
    anchors = [c for c in cells if c["kind"] == "anchor"]
    assert len(dt) == 500 and len(anchors) == 200

    assert {c["subject"] for c in dt} == {tcv.ANCHOR_SUBJECT}
    assert {c["arm"] for c in dt} == {tcv.ANCHOR_ARM}
    assert {c["seed"] for c in dt} == set(tcv.TRAINING_SEEDS)
    assert sorted({c["draw_id"] for c in dt}) == list(tcv.HELD_OUT_DRAWS)
    # Amendment A4: rho's denominator is fixed-time and MaxPressure; `random` is NOT re-rolled.
    assert {c["arm"] for c in anchors} == {"fixedtime", "maxpressure"}
    assert "random" not in {c["arm"] for c in cells}
    assert {str(c["stage"]) for c in cells} == {tcv.STAGE_ANCHOR}
    assert len({tcv.cell_chunk_name(c) for c in cells}) == 700


def test_t14_the_anchor_subject_is_not_in_a17cs_registered_subject_table() -> None:
    """``SUBJECTS`` means *A17(c)'s two CityFlow-trained subjects* and must keep meaning it.

    ``_in_support_block`` iterates it and calls ``targets_for_subject``, which raises for any
    subject P7.2b's artifact never registered; adding the anchor there would have made every
    P7.3a report raise.
    """
    assert tcv.ANCHOR_SUBJECT not in tcv.SUBJECTS
    assert tcv.ANCHOR_SUBJECT not in tc.SUBJECTS
    assert tcv.SUBJECTS == ("mappo1000", "mix50")
    with pytest.raises(ValueError, match="unknown subject"):
        tc.subject_facts(tcv.ANCHOR_SUBJECT, output_root=OUTPUT_ROOT)


# ----------------------------------------------------------------------------------
# The pin source: the training artifact, and NOT the calibration artifact
# ----------------------------------------------------------------------------------
def test_the_anchor_training_record_is_loaded_only_at_its_pinned_digest() -> None:
    """E3(a)'s shape, applied to the record section 3.3 wrote."""
    loaded = tcv.load_anchor_training(ANCHOR_TRAINING)
    assert _sha256(ANCHOR_TRAINING.read_bytes()) == tcv.P7_3B_TRAINING_SHA256
    assert loaded["declared_gradient_steps"] == 40000
    assert loaded["raise_to"] is None
    assert loaded["prompt_rule"] == "naive_in_domain"
    assert len(loaded["seeds"]) == 5


def test_an_anchor_training_record_whose_digest_moved_is_refused(tmp_path: Path) -> None:
    """A prompt read from an unpinned artifact can be edited between training and campaign."""
    moved = tmp_path / tcv.P7_3B_TRAINING_NAME
    payload = json.loads(ANCHOR_TRAINING.read_bytes())
    payload["target_rtg"] = -1.0
    moved.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="is not the pinned"):
        tcv.load_anchor_training(moved)


def test_the_anchors_checkpoints_are_pinned_against_the_training_record_not_p4s(
    tmp_path: Path,
) -> None:
    """G1 for a subject whose committed record was written by the task that consumes it."""
    assert tcv.CHECKPOINT_RECORD[tcv.ANCHOR_SUBJECT] == tcv.P7_3B_TRAINING_NAME
    assert tcv.CHECKPOINT_RECORD["mappo1000"] == "p4_gate.json"
    assert tcv.CHECKPOINT_RECORD["mix50"] == "p4_7_training.json"
    assert tcv.LOCAL_CHECKPOINT_MANIFEST[tcv.ANCHOR_SUBJECT] == "SHA256SUMS_p7_3b_anchor.txt"

    for seed in tcv.TRAINING_SEEDS:
        expected = (
            tmp_path / tcv.ANCHOR_CHECKPOINT_SUBDIR / f"{tcv.ANCHOR_CHECKPOINT_STEM}{seed}.pt"
        )
        assert tcv.checkpoint_path_for(tcv.ANCHOR_SUBJECT, seed, output_root=tmp_path) == expected


def test_the_anchors_facts_come_from_its_own_training_set(tmp_path: Path) -> None:
    """``subject_facts``' quantities, for a subject ``subject_facts`` refuses to know about.

    The support range is the fitted ``stats.rtg`` range over the anchor's own 200 episodes, and
    ``training_set_return_min`` is ``-rtg_scale`` -- the same definitions, from the same kind of
    source, so the in-support diagnostic keeps meaning what it means for the other subjects.
    """
    training = tcv.load_anchor_training(ANCHOR_TRAINING)
    facts = tcv.anchor_subject_facts(training)

    summary = training["normalisation_stats"]["rtg"]["cityflow1x1"]["intersection_1_1"]
    assert facts.support_range_over_the_split == (float(summary["min"]), float(summary["max"]))
    assert facts.training_set_return_min == -float(training["rtg_scale"])
    assert facts.best_source_return == float(training["target_rtg"])
    assert facts.state_dim == 25
    assert facts.context_length == 20
    assert facts.n_rows == int(summary["count"])
    assert facts.subject == tcv.ANCHOR_SUBJECT


# ----------------------------------------------------------------------------------
# Amendment A5 -- the ROUTE: the anchor's prompt never reaches the calibration artifact
# ----------------------------------------------------------------------------------
def _anchor_roots(base: Path) -> _Roots:
    """``_build_roots`` plus the anchor's five checkpoints and its committed training record."""
    roots = _build_roots(base, draws=_DEMO_DRAWS, seeds=_DEMO_SEEDS)
    directory = roots.output / tcv.ANCHOR_CHECKPOINT_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    record = json.loads(ANCHOR_TRAINING.read_bytes())
    rows = []
    for seed in _DEMO_SEEDS:
        path = directory / f"{tcv.ANCHOR_CHECKPOINT_STEM}{seed}.pt"
        path.write_bytes(f"anchor weights {seed}".encode())
        digest = _sha256(path.read_bytes())
        roots.checkpoint_sha[(tcv.ANCHOR_SUBJECT, seed)] = digest
        rows.append({**record["seeds"][0], "seed": seed, "checkpoint_sha256": digest})
    record["seeds"] = rows
    written = roots.data / tcv.P7_3B_TRAINING_NAME
    written.write_text(json.dumps(record), encoding="utf-8")
    # The module pins the REAL file's digest; a fixture cannot forge that, so the pin is relaxed
    # for this fixture ONLY, by pointing the constant at the fixture's own digest. The pin itself
    # is tested against the real artifact in
    # test_the_anchor_training_record_is_loaded_only_at_its_pinned_digest.
    roots.anchor_training_sha = _sha256(written.read_bytes())
    return roots


def test_a5_the_anchor_cell_runs_with_the_calibration_loader_made_to_explode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amendment A5. **The proof is the ROUTE, never a value.**

    On the 201-300 half the naive in-domain target and A17's Rule A ``q = 1.0`` were the SAME
    NUMBER by construction -- both the maximum of the same 100 SUMO probe returns, -20809.0 for
    both subjects. A value comparison would have been vacuous there. (On all 200 episodes they
    differ: -20625.0 against -20809.0. That is a fact about this corpus, not a guarantee, and
    nothing here rests on it.)

    So: make every door into P7.2b's calibration artifact explode, and require the anchor cell to
    run anyway.
    """
    roots = _anchor_roots(tmp_path)
    monkeypatch.setattr(tcv, "P7_3B_TRAINING_SHA256", roots.anchor_training_sha, raising=True)

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "the anchor's prompt must come from its own training set, never from P7.2b's "
            "CityFlow-trained calibration artifact"
        )

    monkeypatch.setattr(tcv, "load_calibration", explode, raising=True)
    monkeypatch.setattr(tcv, "targets_for_subject", explode, raising=True)
    _install_cell_seams(monkeypatch)
    _anchor_dt_seams(monkeypatch, roots)

    cell = _cell(
        "dt", tcv.ANCHOR_ARM, 1000, subject=tcv.ANCHOR_SUBJECT, seed=101, stage=tcv.STAGE_ANCHOR
    )
    chunk = tcv.run_cell(
        cell, out_root=roots.draws, output_root=roots.output, data_dir=roots.data
    )

    training = json.loads((roots.data / tcv.P7_3B_TRAINING_NAME).read_bytes())
    assert chunk["target_rtg"] == float(training["target_rtg"])
    assert chunk["rtg_first"] == chunk["target_rtg"], "PIN 2 stays in force on the anchor"
    assert chunk["anchor_training_sha256"] == roots.anchor_training_sha
    assert chunk["checkpoint_sha256"] == roots.checkpoint_sha[(tcv.ANCHOR_SUBJECT, 101)]
    tcv.validate_cell_payload(chunk, cell=cell)


def _anchor_dt_seams(monkeypatch: pytest.MonkeyPatch, roots: _Roots) -> None:
    """The dt half of the seams: an ``AlignedEnv`` by class and a faked ``dt_choose``."""
    from offline.aligned_env import AlignedEnv

    inner = tcv.env_for_cell({"kind": "anchor", "arm": "fixedtime", "draw_id": 1000}, out_root="x")

    class _AlignedFake(AlignedEnv):
        def __init__(self, env: Any) -> None:
            self._env = env
            self._alignment = None

        def reset(self, **kwargs: Any) -> dict[str, Any]:
            return self._env.reset(**kwargs)

        def step(self, action: Any) -> tuple[Any, bool, bool, dict[str, Any]]:
            return self._env.step(action)

    def fake_dt_choose(env: Any, *, target_rtg: float, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "rtg_series": [], "reward_series": [], "actions": []
        }

        def choose(_env: Any, info: Mapping[str, Any]) -> int:
            # the first RTG is the declared target, which is exactly what PIN 2 checks
            diagnostics["rtg_series"].append(float(target_rtg))
            diagnostics["reward_series"].append(0.0)
            diagnostics["actions"].append(0)
            return 0

        return choose, diagnostics

    monkeypatch.setattr(tcv, "env_for_cell", lambda cell, **k: _AlignedFake(inner), raising=True)
    monkeypatch.setattr(tcv, "dt_choose", fake_dt_choose, raising=True)


def test_a5_an_anchor_chunk_that_cannot_name_its_training_record_is_refused(
    tmp_path: Path,
) -> None:
    """Checked at CONSUMPTION as well as at production, and in BOTH directions."""
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    cell = _cell(
        "dt", tcv.ANCHOR_ARM, 1000, subject=tcv.ANCHOR_SUBJECT, seed=101, stage=tcv.STAGE_ANCHOR
    )
    good = _payload(cell, roots, anchor_training_sha256=tcv.P7_3B_TRAINING_SHA256)
    tcv.validate_cell_payload(good, cell=cell)

    with pytest.raises(ValueError, match="anchor_training_sha256"):
        tcv.validate_cell_payload(_payload(cell, roots, anchor_training_sha256=None), cell=cell)
    with pytest.raises(ValueError, match="anchor_training_sha256"):
        tcv.validate_cell_payload(
            _payload(cell, roots, anchor_training_sha256="0" * 64), cell=cell
        )

    # ...and the other direction: a CityFlow subject may not carry that field, or the two prompt
    # sources could not be told apart from the chunk.
    other = _cell("dt", "b_mean_k100", 1000, subject="mappo1000", seed=101)
    with pytest.raises(ValueError, match="Only 'anchor_k200' is prompted"):
        tcv.validate_cell_payload(
            _payload(other, roots, anchor_training_sha256=tcv.P7_3B_TRAINING_SHA256), cell=other
        )


def test_the_anchor_has_exactly_one_registered_prompt(tmp_path: Path) -> None:
    """A18(a) gives the anchor the naive in-domain rule and nothing else."""
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    cell = _cell(
        "dt", "b_mean_k100", 1000, subject=tcv.ANCHOR_SUBJECT, seed=101, stage=tcv.STAGE_ANCHOR
    )
    with pytest.raises(ValueError, match="exactly one prompt"):
        tcv.validate_cell_payload(
            _payload(cell, roots, anchor_training_sha256=tcv.P7_3B_TRAINING_SHA256), cell=cell
        )


def test_a2_the_anchor_cell_runs_on_the_ALIGNED_observed_env(tmp_path: Path) -> None:
    """Amendment A2, and the reason the anchor's ``kind`` stays ``"dt"``.

    The anchor was TRAINED in A16's canonical 25-wide frame, so it must be EVALUATED in it. A raw
    ``SumoEnv`` hands a CityFlow-frame model 32-wide info in SUMO lane order, which it consumes
    happily and turns into a number that means nothing.
    """
    from offline.aligned_env import AlignedEnv

    cell = _cell(
        "dt", tcv.ANCHOR_ARM, 1000, subject=tcv.ANCHOR_SUBJECT, seed=101, stage=tcv.STAGE_ANCHOR
    )
    assert cell["kind"] == "dt", "the anchor is a dt cell; that is what puts it behind the door"

    # the refusal itself, on the env class rather than on a name
    with pytest.raises(TypeError, match="AlignedEnv"):
        tcv.assert_env_matches_cell(cell, _fake_aligned_env().__class__())
    aligned = object.__new__(AlignedEnv)
    tcv.assert_env_matches_cell(cell, aligned)  # an AlignedEnv is accepted


def test_the_module_never_prompts_the_anchor_from_the_calibration_artifact() -> None:
    """The route, pinned statically as well: inside ``run_cell``'s anchor branch there is no
    calibration name at all.

    The executed route test above is the primary; this catches a future edit that adds the call
    without any test exercising it.
    """
    import ast

    source = Path(tcv.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    run_cell = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_cell"
    )
    # the branch guarded by `subject == ANCHOR_SUBJECT`
    branches = [
        node
        for node in ast.walk(run_cell)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(c, ast.Name) and c.id == "ANCHOR_SUBJECT" for c in node.test.comparators
        )
    ]
    assert branches, "run_cell no longer has an `if subject == ANCHOR_SUBJECT` branch"
    names = {
        node.id for branch in branches for node in ast.walk(ast.Module(body=branch.body, type_ignores=[]))
        if isinstance(node, ast.Name)
    }
    for forbidden in ("load_calibration", "targets_for_subject", "P7_2B_CALIBRATION_NAME"):
        assert forbidden not in names, (
            f"{forbidden!r} is reachable inside the anchor's branch of run_cell; A18(a)'s prompt "
            "is the naive in-domain rule over the anchor's own 200 episodes"
        )


# ----------------------------------------------------------------------------------
# report over an anchor-stage work directory
# ----------------------------------------------------------------------------------

def _anchor_roots_digest_probe(base: Path) -> Path:
    """Write the fixture's training record once, so its digest is known before chunks are built."""
    probe = base / "_pin_probe"
    probe.mkdir(parents=True, exist_ok=True)
    return _anchor_roots(probe).data / tcv.P7_3B_TRAINING_NAME


_ANCHOR_CELLS = [
    _cell("dt", tcv.ANCHOR_ARM, draw, subject=tcv.ANCHOR_SUBJECT, seed=seed,
          stage=tcv.STAGE_ANCHOR)
    for seed in _DEMO_SEEDS
    for draw in _DEMO_DRAWS
] + [
    _cell("anchor", arm, draw, stage=tcv.STAGE_ANCHOR)
    for arm in ("fixedtime", "maxpressure")
    for draw in _DEMO_DRAWS
]


def _anchor_campaign(tmp_path: Path, **perturb: Any) -> _Roots:
    """A miniature anchor stage: the anchor at two seeds on two draws, plus rho's two anchors."""
    roots = _anchor_roots(tmp_path)
    roots.work.mkdir(parents=True, exist_ok=True)
    roots.out.mkdir(parents=True, exist_ok=True)
    _canary(roots.work)
    for cell in _ANCHOR_CELLS:
        overrides = dict(perturb) if cell is _ANCHOR_CELLS[-1] else {}
        tcv.write_chunk(_payload(cell, roots, **overrides), work_dir=roots.work)
    return roots


def test_report_publishes_the_anchors_rho_under_both_definitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 3.4: ``report`` gains the anchor's rows and its rho under both definitions.

    The fixture's ATT values are exact binary fractions, so every rho below is compared with
    ``==`` and a rounding change is a failure rather than noise.
    """
    # ⚠️ The pin is redirected BEFORE the chunks are written: `_payload` reads the constant at
    # call time, so patching afterwards would build chunks naming the REAL digest and then judge
    # them against the fixture's. The pin against the real artifact is tested separately.
    monkeypatch.setattr(
        tcv, "P7_3B_TRAINING_SHA256",
        _sha256((_anchor_roots_digest_probe(tmp_path)).read_bytes()), raising=True,
    )
    roots = _anchor_campaign(tmp_path)
    artifact = tcv.report(
        work_dir=roots.work,
        out_path=roots.out / "p7_3b_anchor.json",
        output_root=roots.output,
        out_root=roots.draws,
        data_dir=roots.data,
        stage=tcv.STAGE_ANCHOR,
        cells=_ANCHOR_CELLS,
    )

    entries = [e for e in artifact["rho"]["by_subject_arm"] if e["subject"] == tcv.ANCHOR_SUBJECT]
    assert len(entries) == 1, "the anchor is one (subject, arm)"
    entry = entries[0]
    assert entry["arm"] == tcv.ANCHOR_ARM
    assert entry["role"] == tcv.ANCHOR_ROLE
    for key in ("e_sumo", "att_env"):
        assert entry[key]["n_draws"] == len(_DEMO_DRAWS)
        assert entry[key]["mean"] is not None
        assert entry[key]["ci95_low"] <= entry[key]["mean"] <= entry[key]["ci95_high"]
    # both definitions present, and genuinely distinct objects
    assert set(entry["paired_att"]) == {"e_sumo", "att_env"}
    assert set(entry["paired_att"]["e_sumo"]) == {"fixedtime", "maxpressure"}

    # rho recomputed here by the registered formula, from the published cell rows, exactly
    rows = {(r["subject"], r["arm"], r["seed"], r["draw_id"]): r for r in artifact["cells"]}
    for (subject, arm, seed, draw), row in rows.items():
        if subject != tcv.ANCHOR_SUBJECT:
            continue
        fixed = rows[(None, "fixedtime", None, draw)]
        maxp = rows[(None, "maxpressure", None, draw)]
        assert row["rho_e_sumo"] == tcv.rho(row["e_sumo"], fixed["e_sumo"], maxp["e_sumo"])
        assert row["rho_att_env"] == tcv.rho(row["att_env"], fixed["att_env"], maxp["att_env"])

    assert artifact["stage"] == tcv.STAGE_ANCHOR
    assert artifact["n_cells_declared"] == len(_ANCHOR_CELLS)


def test_the_anchor_artifact_says_what_the_anchor_is_not_and_names_a18bs_absent_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A18(a)'s sentence VERBATIM in the artifact, and A18(b) in ``what_this_does_not_say``.

    In the artifact rather than only in the packet, because the packet does not travel with it --
    the author's 2026-09-17 ruling on ``ATT_ENV_CAVEAT``, applied to the same problem.
    """
    # ⚠️ The pin is redirected BEFORE the chunks are written: `_payload` reads the constant at
    # call time, so patching afterwards would build chunks naming the REAL digest and then judge
    # them against the fixture's. The pin against the real artifact is tested separately.
    monkeypatch.setattr(
        tcv, "P7_3B_TRAINING_SHA256",
        _sha256((_anchor_roots_digest_probe(tmp_path)).read_bytes()), raising=True,
    )
    roots = _anchor_campaign(tmp_path)
    artifact = tcv.report(
        work_dir=roots.work, out_path=roots.out / "a.json", output_root=roots.output,
        out_root=roots.draws, data_dir=roots.data, stage=tcv.STAGE_ANCHOR, cells=_ANCHOR_CELLS,
    )

    block = artifact["anchor"]
    # A18(a), verbatim
    assert block["what_it_is_not"] == tcv.ANCHOR_WHAT_IT_IS_NOT
    assert "not an upper bound on achievable SUMO performance" in block["what_it_is_not"]
    assert "not a target-domain online policy" in block["what_it_is_not"]
    assert "k = 200 endpoint" in block["what_it_is_not"]

    says_not = artifact["what_this_does_not_say"]
    assert says_not == tcv.ANCHOR_WHAT_THIS_DOES_NOT_SAY
    assert "A18(b)" in says_not
    assert "NO online SUMO anchor" in says_not
    assert "P11" in says_not and "limitation" in says_not
    assert "random" in says_not and "NOT re-rolled" in says_not
    assert "H3's clauses are P7.3a's" in says_not
    assert "Amendment A4" in says_not

    # the prompt's source is named, with its digest, and what it is NOT is named too
    assert block["prompt"]["source_artifact"] == tcv.P7_3B_TRAINING_NAME
    assert block["prompt"]["source_sha256"] == tcv.P7_3B_TRAINING_SHA256
    assert "p7_2b_calibration.json" in block["prompt"]["not_from"]
    assert block["training"]["raise_to"] is None
    assert block["training"]["declared_gradient_steps"] == 40000
    assert block["training"]["disjoint_from_the_subjects_and_the_pool"] is True
    assert len(block["training"]["seeds"]) == len(_DEMO_SEEDS)

    # H3 is P7.3a's and is not recomputed here: no b_mean_k100 cell is in this work directory.
    # Asserted on what the sentence CLAIMS, not on a phrase I happened to write.
    note = block["h3_is_not_recomputed_here"]
    assert "b_mean_k100" in note
    assert "p7_3a_zero_shot.json" in note
    assert "no subject in it" in note
    for clause in artifact["h3"]["clauses"]:
        assert clause["mean_rho"] == {}, "no registered-arm subject in an anchor work directory"


def test_report_refuses_an_anchor_cell_whose_checkpoint_sha_is_not_the_declared_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G1 at the artifact: the weights that produced the cell must be the registered ones."""
    # ⚠️ The pin is redirected BEFORE the chunks are written: `_payload` reads the constant at
    # call time, so patching afterwards would build chunks naming the REAL digest and then judge
    # them against the fixture's. The pin against the real artifact is tested separately.
    monkeypatch.setattr(
        tcv, "P7_3B_TRAINING_SHA256",
        _sha256((_anchor_roots_digest_probe(tmp_path)).read_bytes()), raising=True,
    )
    roots = _anchor_campaign(tmp_path)
    victim = tcv.chunk_path(_ANCHOR_CELLS[0], work_dir=roots.work)
    payload = json.loads(victim.read_bytes())
    payload["checkpoint_sha256"] = "9" * 64
    victim.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not the registered ones"):
        tcv.report(
            work_dir=roots.work, out_path=roots.out / "a.json", output_root=roots.output,
            out_root=roots.draws, data_dir=roots.data, stage=tcv.STAGE_ANCHOR,
            cells=_ANCHOR_CELLS,
        )
    assert not (roots.out / "a.json").exists(), "every refusal precedes every write"


def test_the_anchor_stage_writes_its_own_artifact_name() -> None:
    """The anchor's artifact is its own file; writing it under P7.3a's name would overwrite the
    zero-shot point with the curve's endpoint."""
    source = Path(tcv.__file__).read_text(encoding="utf-8")
    assert '"p7_3b_anchor.json"' in source
    assert 'name = "p7_3b_anchor.json"' in source


# ----------------------------------------------------------------------------------
# §3.5 -- the driver's text (T9's shape) and its executed refusals
# ----------------------------------------------------------------------------------
ANCHOR_DRIVER = Path(__file__).resolve().parents[1] / "offline" / "campaigns" / "p7_3b_anchor.sh"


def test_t9_the_anchor_driver_carries_every_invariant_the_campaign_shape_requires() -> None:
    """T9's shape, over comment-free source so a requirement cannot be satisfied by a comment.

    ⚠️ Reviewer A's minor 11 applies here as it did to P7.3a's family: this pins ONE SPELLING per
    invariant, so a re-spelled token would survive while the literal one dies. The EXECUTED
    refusals below are the stronger half.
    """
    raw = ANCHOR_DRIVER.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    )

    for invariant in (
        "set -euo pipefail",
        "AUTHORISED_TO_RUN",
        'rm -f "$TOKEN"',
        "pgrep -f",
        "ps -o pgid= -p $$",
        "SigIgn",
        "git -C \"$WORK_TREE\" status --porcelain",
        "canary",
        "record-canary",
        "tee -a",
        "sha256sum -c",
    ):
        assert invariant in code, f"the driver no longer carries {invariant!r}"

    # all three signals, in one trap, and the trap BEFORE the token is consumed
    assert "trap on_signal INT TERM HUP" in code
    assert code.index("trap on_signal") < code.index('rm -f "$TOKEN"'), (
        "the trap must be installed before the token is consumed: a signal in that window "
        "destroyed P7.2b's authorisation while leaving neither FAILED nor COMPLETE"
    )
    # the canary precedes the token too -- a throttled machine must not burn the authorisation
    assert code.index("CANARY_MAX_SECONDS") < code.index('rm -f "$TOKEN"')
    # Nothing WRITES into docs/data. Two references to it are legitimate and neither is a write:
    # `--data-dir "$DATA"`, which is an INPUT, and a printed `cp ...` hint for the human who
    # copies the artifact into the branch afterwards. The invariant is that the artifact's
    # destination is $WORK/artifacts, which is inside the manifest and outside the worktree
    # (P7.3a Finding 2). Asserting "docs/data" appears nowhere was testing the wrong thing.
    assert 'ARTIFACTS=$WORK/artifacts' in code
    assert '--out-dir "$ARTIFACTS"' in code
    writes = [
        line for line in code.splitlines()
        if "docs/data" in line and not line.lstrip().startswith("echo")
    ]
    assert writes == ['DATA=$WORK_TREE/docs/data'], (
        f"docs/data is referenced outside --data-dir and the printed hint: {writes}"
    )
    assert "--overwrite" not in code


def test_the_anchor_driver_is_syntactically_valid_bash() -> None:
    """``bash -n`` on the delivered file: a driver that does not parse refuses nothing."""
    result = subprocess.run(
        ["bash", "-n", str(ANCHOR_DRIVER)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    not MAIN_INTERPRETER.is_file(),
    reason=f"needs the main tree's interpreter at {MAIN_INTERPRETER}",
)
def test_the_anchor_driver_refuses_a_bogus_stage_and_creates_nothing(tmp_path: Path) -> None:
    """EXECUTED, not read: the stage argument is required, so starting the wrong thing refuses.

    Run under ``setsid`` so the group-leader check passes and the stage check is what fires;
    ``$WORK`` is redirected into ``tmp_path`` so a real campaign directory cannot be touched.
    """
    redirected = tmp_path / "driver.sh"
    work = tmp_path / "work"
    redirected.write_text(
        ANCHOR_DRIVER.read_text(encoding="utf-8")
        .replace("WORK=$MAIN/output/p7_3b_anchor", f"WORK={work}")
        .replace(
            'WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)',
            f"WORK_TREE={ANCHOR_DRIVER.resolve().parents[2]}",
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["setsid", "--wait", "bash", str(redirected), "not-a-stage"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2, f"expected exit 2, got {result.returncode}: {result.stderr}"
    assert "REFUSING TO START: the stage must be 'anchor'" in result.stderr
    assert not work.exists(), "a refused start creates nothing under WORK"


def test_the_campaigns_own_declaration_pair_is_right_for_every_stage() -> None:
    """``declarations_for`` on the path the DRIVER takes: ``cells=None``.

    ⚠️ **This exists because a mutation survived.** Collapsing
    ``whole = STAGE_ANCHOR if stage == STAGE_ANCHOR else None`` to ``whole = None`` left all 132
    tests green: every ``report`` test in this file passes an explicit ``cells=``, so the campaign
    branch was reached by nothing. Under that mutation a real anchor run would compare its 700
    chunks against P7.3a's 4,700 and refuse every one of them as "not a declared cell of ANY
    stage" -- a refusal rather than a wrong number, but a campaign that cannot finish.
    """
    whole, sliced = tcv.declarations_for(None, None)
    assert len(whole) == 4700 and len(sliced) == 4700

    whole, sliced = tcv.declarations_for(tcv.STAGE_CONFIRMATORY, None)
    assert len(whole) == 4700 and len(sliced) == 1200

    whole, sliced = tcv.declarations_for("rest", None)
    assert len(whole) == 4700 and len(sliced) == 3500

    # the anchor's whole declaration is its OWN 700, never P7.3a's 4,700
    whole, sliced = tcv.declarations_for(tcv.STAGE_ANCHOR, None)
    assert len(whole) == 700 and len(sliced) == 700
    assert {str(c["stage"]) for c in whole} == {tcv.STAGE_ANCHOR}
    # ⚠️ The two declarations DO share 200 chunk names, and that is a fact about
    # `cell_chunk_name`, which derives the name from (subject, arm, seed, draw) and not from the
    # stage: P7.3a's `cell_anchor_fixedtime_seednone_draw1000.json` and the anchor stage's
    # re-rolled denominator are the same string. They never collide on disk -- the two campaigns
    # have separate work directories -- and the name is NOT changed, because P7.3a's chunks are on
    # disk under it. What closes the hazard is that `stage` is part of the chunk-vs-declaration
    # identity, so a chunk from the other campaign is refused by name AND by stage.
    names = {tcv.cell_chunk_name(c) for c in whole}
    campaign = {tcv.cell_chunk_name(c) for c in tcv.declared_cells(None)}
    shared = names & campaign
    assert len(shared) == 200, f"expected the 200 rho anchors to share names, got {len(shared)}"
    assert all("fixedtime" in n or "maxpressure" in n for n in shared)

    # a caller-supplied set is its own whole declaration, and is copied rather than aliased
    supplied = [_cell("anchor", "fixedtime", 1000)]
    whole, sliced = tcv.declarations_for(tcv.STAGE_ANCHOR, supplied)
    assert whole == sliced == supplied
    assert whole[0] is not supplied[0], "the caller's dicts must not be aliased into report"


def test_a_chunk_from_the_other_campaign_is_refused_by_stage_not_accepted_by_name(
    tmp_path: Path,
) -> None:
    """The refusal that makes the 200 shared chunk names safe (reviewer A's minor 9, closed here).

    A P7.3a ``fixedtime`` chunk and an anchor-stage ``fixedtime`` cell have the SAME file name.
    Without ``stage`` in the chunk-vs-declaration identity, offering one as the other validates,
    and 200 of the anchor's 700 cells would silently be another campaign's.
    """
    roots = _build_roots(tmp_path, draws=(1000,), seeds=(101,))
    p7_3a_cell = _cell("anchor", "fixedtime", 1000, stage=tcv.STAGE_CONFIRMATORY)
    anchor_cell = _cell("anchor", "fixedtime", 1000, stage=tcv.STAGE_ANCHOR)

    assert tcv.cell_chunk_name(p7_3a_cell) == tcv.cell_chunk_name(anchor_cell), (
        "this test is only meaningful while the two names are equal"
    )

    chunk = _payload(p7_3a_cell, roots)
    tcv.validate_cell_payload(chunk, cell=p7_3a_cell)          # its own declaration: fine
    with pytest.raises(ValueError, match="stage"):
        tcv.validate_cell_payload(chunk, cell=anchor_cell)     # the other campaign's: refused


def test_the_pilot_transcript_survives_a_cell_that_has_no_training_seed() -> None:
    """Found by RUNNING the pre-flight pilot, which is what a pre-flight is for.

    ``anchor_pilot_cells`` includes rho's two denominators, and an anchor cell has ``seed: None``
    -- fixedtime and MaxPressure have no training seed. ``run_pilot``'s transcript did
    ``int(cell["seed"])`` over every cell, so all four cells rolled successfully and then the
    SUMMARY raised ``TypeError``, losing the rate the pilot exists to measure. F1's sixteen cells
    are all ``dt``, which is why nothing noticed for a whole task.

    Asserted on the cell set rather than by re-running the pool: the shape is what was wrong.
    """
    cells = tcv.anchor_pilot_cells()
    without = [c for c in cells if c["seed"] is None]
    assert len(without) == 2, "rho's two denominators carry no training seed"
    assert {c["arm"] for c in without} == {"fixedtime", "maxpressure"}
    # the expression that used to raise, now written as the module writes it
    assert sorted({int(c["seed"]) for c in cells if c["seed"] is not None}) == list(
        tcv.PILOT_SEEDS
    )
    # The defect, stated so it cannot come back. `match=` is not decoration: a bare
    # pytest.raises(TypeError) is satisfied by ANY TypeError, including one from a typo in the
    # expression itself -- which is exactly what scripts/check_test_hygiene.sh TH006 refuses.
    with pytest.raises(TypeError, match="NoneType"):
        sorted({int(c["seed"]) for c in cells})


def test_t14_the_anchor_declaration_is_pinned_by_size_and_by_composition() -> None:
    """The coordinator's surviving mutation: folding the anchor stage into the campaign
    declaration left all 134 tests green, because nothing pinned the anchor stage's SIZE or its
    COMPOSITION on their own.

    It cannot make a published number wrong -- the campaign would refuse every chunk rather than
    report a wrong one -- but an unpinned declaration is exactly what BRIEF_38 section 2 exists to
    prevent, so it is pinned here.
    """
    cells = tcv.declared_cells(tcv.STAGE_ANCHOR)
    assert len(cells) == 700

    dt = [c for c in cells if c["kind"] == "dt"]
    denominators = [c for c in cells if c["kind"] == "anchor"]
    assert len(dt) == 500, "5 training seeds x 100 held-out draws"
    assert len(denominators) == 200, "fixedtime and MaxPressure, one episode per draw"
    assert len(dt) + len(denominators) == len(cells), "there is no third kind"

    # the composition, not merely the counts: the full cross product, once each
    assert {(c["subject"], c["arm"], c["seed"], c["draw_id"]) for c in dt} == {
        (tcv.ANCHOR_SUBJECT, tcv.ANCHOR_ARM, seed, draw)
        for seed in tcv.TRAINING_SEEDS
        for draw in tcv.HELD_OUT_DRAWS
    }
    assert {(c["arm"], c["draw_id"]) for c in denominators} == {
        (arm, draw) for arm in ("fixedtime", "maxpressure") for draw in tcv.HELD_OUT_DRAWS
    }
    assert all(c["seed"] is None and c["subject"] is None for c in denominators)

    # and it is NOT reachable from the campaign's declaration, by any route
    assert tcv.declared_cells(None) == tcv.declared_cells(None)
    campaign = tcv.declared_cells(None)
    assert len(campaign) == 4700
    assert not any(c["stage"] == tcv.STAGE_ANCHOR for c in campaign)
    assert len(campaign) + len(cells) == 5400, (
        "the two declarations are disjoint sets, not one set with a filter"
    )
