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
from pathlib import Path
from typing import Any, Mapping

import pytest

from offline import transfer_calibration as tc
from offline import transfer_curve as tcv

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
        "git_commit": "0" * 40,
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
    if is_dt:
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
                "git_commit": "0" * 40,
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
    written.  ``git_commit`` is the carrier because it is free text that reaches the artifact.
    """
    roots, _ = _campaign(tmp_path)
    cell = _cell("anchor", "fixedtime", 1000)
    tcv.write_chunk(_payload(cell, roots, git_commit=tc.FENCED_KEY), work_dir=roots.work)

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
    """
    code = _driver_code_text()
    collection = code.index("run_stage collect offline.collect")
    gate = code.index("run_stage a17f offline.transfer_curve")
    cells = code.index("cells --stage")

    assert collection < gate < cells, (
        "the order must be: collect the corpus, check A17(f), and only then evaluate"
    )
    # Every stage goes through run_stage, which calls `fail "$label"` when the child exits
    # non-zero; that is what turns A17(f)'s refusal into a stopped campaign rather than a log line.
    assert 'fail "$label"' in code, "run_stage must stop the driver when a stage exits non-zero"
    assert code.index("run_stage()") < gate, "a17f must be run THROUGH run_stage, not beside it"


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
