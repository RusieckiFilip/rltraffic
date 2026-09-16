"""P7.3a: the zero-shot point of the C3 transfer curve, and the artifact that reports it.

Artifact format version: ``p7.3a-zero-shot/1.0`` -- ``docs/data/p7_3a_zero_shot.json``, written by
:func:`report` from the per-cell chunks under ``output/p7_3a/``.  Amendment B2's stage-1 artifact
is ``docs/data/p7_3a_zero_shot_stage1.json``, the same block layout over the confirmatory cells.

WHAT THIS MODULE EXECUTES, AND WHAT IT MAY NOT DECIDE
------------------------------------------------------
Everything scientific is registered and READ, never chosen here: **H3** and its test row
(``PREREGISTRATION`` §2), **rho** (§3.4), **A15** (the definition pair, the counts every cell
carries, the teleport-free regime, the admitted pair), **A16** (the door), **A17** (Rule B with
``S = mean`` at k = 100 as the registered prompt, the two subjects, the exploratory contrast, the
consistency gate) and **A18(c)** (the held-out seed rule).  The four arms were closed by
``BRIEF_37`` §2 before any cell ran, and the two stages by Amendment B1.

RHO, AND WHY THERE ARE TWO NUMBERS AND ONE FORMULA
---------------------------------------------------
``PREREGISTRATION`` §3.4 fixes ONE formula, computed **within** a backend::

    rho = (ATT_fixedtime - ATT_policy) / (ATT_fixedtime - ATT_maxpressure)

fixed-time is 0 and MaxPressure is 1 **by construction**, and values outside [0, 1] are expected and
are never clipped.  A15(a) then observes that §3.4 never fixed *which* ATT definition each backend
computes it on, and A15(b) requires **both** on every SUMO cell.  So this module reports rho twice
-- once on ``e_sumo`` (the pool-clock, all-created twin, A15's primary) and once on ``att_env`` (the
admitted pair beside it) -- and that, not two formulas, is what ``BRIEF_37`` §3.5's *"both
definitions"* means (Amendment A1).

⚠️ ``e_sumo`` AND ``att_reference_created_population`` ARE ONE QUANTITY UNDER TWO NAMES
(Amendment C6): :func:`offline.sumo_att_reference.reconstruct_sumo_episode` calls it ``e_sumo`` and
the artifact key it is written to in P7.1's freeze is ``att_reference_created_population``.  A15's
text is correct and is not amended; this sentence is the alias, recorded once.

THE DOOR IS FOR THE DT, NOT FOR THE ANCHORS (Amendment A2)
-----------------------------------------------------------
A DT cell runs on an **observed, aligned** env; an anchor cell runs on an **observed, unwrapped**
one.  ``align_info`` drops outgoing lanes and re-keys the survivors to CityFlow ids, while
MaxPressure's pressure is a difference over the env's own SUMO lane ids -- wrapping it raises
``KeyError`` on an outgoing lane, measured before it was written down.  A16 is untouched: the door
is the only route into *a CityFlow-trained model's* frame, and an anchor has no frame to enter.

THE FENCE IS LIFTED HERE, FOR THESE ARMS, AND FOR NOTHING ELSE
----------------------------------------------------------------
P7.2b fenced ``att_horizon``, ``episode_reward``, ``rtg_last`` and the RTG series because A3 said
*until P7.3's brief is written*.  It is written: ``BRIEF_37`` §2 lifts the fence for the four
declared arms and the three anchors on the **held-out pool**, and for nothing else.  P7.2b's smoke
on draw 5 stays fenced.  :func:`report` refuses any cell whose arm is not declared.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "ANCHOR_ARMS",
    "ARTIFACT_FORMAT_VERSION",
    "DECLARED_ARMS",
    "HALTING_CHECK_DRAW",
    "HELD_OUT_DRAWS",
    "P7_2B_CALIBRATION_SHA256",
    "RANDOM_POLICY_SEEDS",
    "STAGE_CONFIRMATORY",
    "STAGES",
    "SUBJECTS",
    "TRAINING_SEEDS",
    "ArmSpec",
    "anchor_choose",
    "assert_env_matches_cell",
    "att_env_from_info",
    "build_parser",
    "cell_chunk_name",
    "checkpoint_identity",
    "chunk_is_reusable",
    "chunk_path",
    "declared_cells",
    "demand_identity",
    "dt_choose",
    "env_for_cell",
    "halting_check_for",
    "load_calibration",
    "main",
    "report",
    "reusable_chunk_at",
    "rho",
    "run_cell",
    "targets_for_subject",
    "validate_cell_payload",
    "write_chunk",
]

ARTIFACT_FORMAT_VERSION = "p7.3a-zero-shot/1.0"

#: A15(g)'s admitted pair; the only scenario this task evaluates.
SCENARIO_KEY = "cityflow1x1"

#: A18(c): every held-out episode is one per draw on a fresh env at ``reset(seed=1000)``.
HELD_OUT_DRAWS: tuple[int, ...] = tuple(range(1000, 1100))
ENGINE_SEED = 1000

#: A17(c)'s two zero-shot subjects, and P4's five training seeds.
SUBJECTS: tuple[str, ...] = ("mappo1000", "mix50")
TRAINING_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)

#: The anchors rho is defined against (§3.4), plus ``random`` which the C1 ladder normalises on.
#: ``fixedtime`` and ``maxpressure`` are one episode per draw; ``random`` is five POLICY seeds at
#: the same engine seed (A18(c)), because a single random episode is a draw from a distribution
#: rather than an anchor.
ANCHOR_ARMS: tuple[str, ...] = ("fixedtime", "maxpressure", "random")
RANDOM_POLICY_SEEDS: tuple[int, ...] = (1000, 1001, 1002, 1003, 1004)

#: Amendment C2: the recorder's halting cross-check is value-neutral (A9/A9b reproduce P7.1's
#: frozen ``att_env`` and ``e_sumo`` bit-for-bit with it ON and OFF) and costs 3.5x, so it runs on
#: a DECLARED SUBSET -- every cell on this draw -- and nowhere else.  47 cells of 4,700.
HALTING_CHECK_DRAW = 1000


class ArmSpec:
    """One declared arm: how its target is found in P7.2b's artifact, and what it is for.

    ``BRIEF_37`` §2 closed this set **before any cell ran**, which is what A17(d) requires of a
    reduction: Rule B's k = 5 and k = 20 targets are the few-shot prompts A18(d) attaches to
    fine-tuned models and are deliberately NOT evaluated zero-shot.
    """

    __slots__ = ("name", "rule", "statistic", "k", "role")

    def __init__(self, name: str, rule: str, statistic: str, k: int | None, role: str) -> None:
        self.name = name
        self.rule = rule
        self.statistic = statistic
        self.k = k
        self.role = role

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"ArmSpec({self.name!r}, role={self.role!r})"


#: The four, in the order the packet reports them.  ``role`` is checked against the artifact's own
#: ``role`` field rather than assumed: the registered prompt is whichever row P7.2b marked, and a
#: disagreement between that mark and this table is a refusal, not a preference.
DECLARED_ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("b_mean_k100", "rule_b", "mean", 100, "registered_prompt"),
    ArmSpec("b_max_k100", "rule_b", "max", 100, "ablation"),
    ArmSpec("a_q1.0", "rule_a", "q1.0", 100, "ablation"),
    ArmSpec("naive", "naive", "none", None, "ablation"),
)

#: Amendment B1's two declared stages.  **Stage 2 is UNCONDITIONAL**: it runs whatever stage 1
#: shows, exactly as P7.3b runs whatever the zero-shot number is (§7).  This is a sequence, not a
#: cut, and nothing registered moves between them.
STAGE_CONFIRMATORY = "confirmatory"
STAGES: tuple[str, ...] = (STAGE_CONFIRMATORY, "rest")


def targets_for_subject(
    subject: str, artifact: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """The four declared arms' targets for *subject*, READ from P7.2b's artifact.

    Looked up by ``(rule, statistic, k)`` and cross-checked against the artifact's own ``role``:
    ``b_mean_k100`` must be the row P7.2b marked ``registered_prompt`` and the other three must be
    ``ablation``.  A table that disagreed with the artifact would silently re-register the prompt,
    which is the one thing A17 does not permit this task to do.
    """
    rows = artifact["targets"][subject]
    out: dict[str, dict[str, Any]] = {}
    for spec in DECLARED_ARMS:
        matching = [
            row
            for row in rows
            if row["rule"] == spec.rule
            and row["statistic"] == spec.statistic
            and row["k"] == spec.k
        ]
        if len(matching) != 1:
            raise ValueError(
                f"{subject}: {len(matching)} rows match arm {spec.name} "
                f"({spec.rule}, {spec.statistic}, k={spec.k}); exactly one is required"
            )
        row = matching[0]
        if row["role"] != spec.role:
            raise ValueError(
                f"{subject}: arm {spec.name} is declared {spec.role!r} but P7.2b's artifact marks "
                f"that row {row['role']!r}. The artifact is the registration; refusing to evaluate "
                "a prompt under a role this task assigned it"
            )
        out[spec.name] = {
            "target_rtg": float(row["target_rtg"]),
            "rule": spec.rule,
            "statistic": spec.statistic,
            "k": spec.k,
            "role": spec.role,
            "in_support": row["in_support"],
        }
    registered = [name for name, row in out.items() if row["role"] == "registered_prompt"]
    if registered != ["b_mean_k100"]:
        raise ValueError(
            f"{subject}: the registered prompt resolved to {registered}, not ['b_mean_k100']"
        )
    return out


def declared_cells(stage: str | None = None) -> list[dict[str, Any]]:
    """Every cell the campaign runs, as declared -- optionally only one stage's.

    A cell is ``(kind, subject, arm, seed, draw)``.  ``kind`` is ``"dt"`` or ``"anchor"`` and it is
    what decides whether the env goes through A16's door (Amendment A2), so it is part of the
    cell's identity rather than something inferred later from the arm's name.

    Amendment B1's stages:

    * **confirmatory** -- ``b_mean_k100`` x 2 subjects x 5 seeds x 100 draws (1,000), plus
      ``fixedtime`` and ``maxpressure`` on the same draws (200): the arm H3 tests and the two
      anchors §3.4's rho is defined against. 1,200 cells.
    * **rest** -- the three ablation arms x 2 subjects x 5 seeds (3,000) and ``random`` x 5 policy
      seeds (500). 3,500 cells.
    """
    if stage is not None and stage not in STAGES:
        raise ValueError(f"{stage!r} is not one of {list(STAGES)}")

    cells: list[dict[str, Any]] = []
    for spec in DECLARED_ARMS:
        in_stage1 = spec.name == "b_mean_k100"
        for subject in SUBJECTS:
            for seed in TRAINING_SEEDS:
                for draw in HELD_OUT_DRAWS:
                    cells.append(
                        {
                            "kind": "dt",
                            "subject": subject,
                            "arm": spec.name,
                            "seed": int(seed),
                            "draw_id": int(draw),
                            "stage": STAGE_CONFIRMATORY if in_stage1 else "rest",
                        }
                    )
    for arm in ANCHOR_ARMS:
        seeds = RANDOM_POLICY_SEEDS if arm == "random" else (None,)
        for seed in seeds:
            for draw in HELD_OUT_DRAWS:
                cells.append(
                    {
                        "kind": "anchor",
                        "subject": None,
                        "arm": arm,
                        "seed": None if seed is None else int(seed),
                        "draw_id": int(draw),
                        "stage": "rest" if arm == "random" else STAGE_CONFIRMATORY,
                    }
                )
    if stage is not None:
        cells = [cell for cell in cells if cell["stage"] == stage]
    return cells


def cell_chunk_name(cell: Mapping[str, Any]) -> str:
    """A chunk's file name; one cell, one file, and the name carries the whole identity."""
    subject = cell["subject"] or "anchor"
    seed = "none" if cell["seed"] is None else int(cell["seed"])
    return f"cell_{subject}_{cell['arm']}_seed{seed}_draw{int(cell['draw_id']):04d}.json"


def rho(att_arm: float, att_fixedtime: float, att_maxpressure: float) -> float:
    """``PREREGISTRATION`` §3.4, verbatim: the ONE registered formula, within one backend.

    ``rho = (ATT_fixedtime - ATT_arm) / (ATT_fixedtime - ATT_maxpressure)``

    fixed-time is 0 and MaxPressure is 1 **by construction** -- substituting either anchor for
    ``att_arm`` gives exactly that, which is what T7 asserts rather than approximates.  §3.4:
    *"Values may exceed 100 or fall below 0; that is expected and is not clipped."*  Nothing here
    clips, and a caller that wanted to would be changing a registered definition.

    The anchors must be **the same draw's**.  Pairing is per draw because the demand differs by
    draw and a ratio built from another draw's denominator is not a normalisation of anything; the
    caller is responsible for that pairing and :func:`report` refuses when it is broken.
    """
    denominator = float(att_fixedtime) - float(att_maxpressure)
    if denominator == 0.0:
        raise ValueError(
            "the two anchors have equal ATT on this draw, so rho's denominator is zero and the "
            "normalisation is undefined; reporting it as any finite number would be an invention"
        )
    return (float(att_fixedtime) - float(att_arm)) / denominator


# ======================================================================================
# Section 3.5b -- the pins, the cell runner, the chunks and the artifact
# (Amendments E2, E3(a)-(f), G1-G4)
# ======================================================================================

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Roots are PARAMETERS with today's paths as defaults (``BRIEF_37`` §0.9, ``DEFERRED`` 82): this
#: module adds no new hardcoded absolute path to a call site, and a test points them elsewhere.
DEFAULT_DRAWS_ROOT = Path("/home/filip/rltraffic/scenarios/draws")
DEFAULT_OUTPUT_ROOT = Path("/home/filip/rltraffic/output")
DEFAULT_WORK_DIR = DEFAULT_OUTPUT_ROOT / "p7_3a"
DEFAULT_DATA_DIR = _REPO_ROOT / "docs" / "data"

#: E3(a): A17's targets are READ from P7.2b's artifact, and the artifact is pinned by digest.  The
#: pin is a DECLARATION -- *these targets came from THAT file* -- and the file is the evidence; a
#: target read from an unpinned artifact can be edited between the calibration and the campaign
#: without leaving a trace.  It moves only in a commit that also moves the artifact.
P7_2B_CALIBRATION_NAME = "p7_2b_calibration.json"
P7_2B_CALIBRATION_SHA256 = "92b1592de637cee187c56b988ce320611d89706c8f9f02c34fe7ebbe81658d86"

#: G1 -- the checkpoint pins, and why they are these files.
#:
#: ``BRIEF_37`` §3.5 and Amendment E3(b) named ``SHA256SUMS_p4_6.txt``.  It lists ``p4_6/checkpoints/*``
#: and **no** ``p4_dt/`` path, the repository has recorded that since ``BRIEF_27``
#: (``nortg_campaign.TIER_MANIFEST["mappo1000"] = None``, ``DEFERRED`` 56), and every
#: ``SHA256SUMS_*`` file is gitignored, so it is not a record a reader of the repository can check.
#: G1's ruling: **both subjects are pinned against COMMITTED artifacts**, and the local manifest is
#: checked as well wherever it happens to list the file.
CHECKPOINT_RECORD: Mapping[str, str] = {
    "mappo1000": "p4_gate.json",
    "mix50": "p4_7_training.json",
}

#: The gitignored campaign manifest that also lists the file, where one exists.  ``None`` for
#: ``mappo1000`` is ``DEFERRED`` 56 recorded rather than worked around.
LOCAL_CHECKPOINT_MANIFEST: Mapping[str, str | None] = {
    "mappo1000": None,
    "mix50": "SHA256SUMS_p4_7.txt",
}

#: ``p4_7_training.json`` carries four METHODS per (tier, seed) -- ``bc``, ``bc_top10``, ``iql``,
#: ``dt`` -- so ``(tier, seed)`` matches four rows, three of which are other checkpoints.  The DT is
#: the one A17(c) registered.
MIX50_TRAINING_METHOD = "dt"

#: The parity artefacts of one draw, and the provenance file that records both digests (G2).
PARITY_CONFIG_NAME = "noteleport.sumocfg"
PARITY_ROUTES_NAME = "routes.rou.xml"
PARITY_PROVENANCE_NAME = "provenance.json"

#: The declared arm names, anchors included: what ``report``'s fence admits and nothing else.
DECLARED_ARM_NAMES: frozenset[str] = frozenset(
    [spec.name for spec in DECLARED_ARMS] + list(ANCHOR_ARMS)
)

#: P7.2b's fenced marker.  A chunk that carried it would be a fenced quantity on its way into
#: ``docs/data/``; the last refusal in :func:`report` scans the serialised bytes for it.
FENCED_KEY = "fenced_do_not_report"


def _sha256_file(path: str | Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomic, sorted, newline-terminated -- so an artifact is either whole or absent."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)


def _git_provenance() -> dict[str, Any]:
    from offline.materialise_draws import _git_commit

    commit, dirty = _git_commit()
    return {"git_commit": commit, "git_dirty": dirty}


def _data_dir(data_dir: str | Path | None) -> Path:
    return DEFAULT_DATA_DIR if data_dir is None else Path(data_dir)


def load_calibration(artifact_path: str | Path | None = None) -> dict[str, Any]:
    """E3(a): P7.2b's calibration artifact, digest-checked BEFORE it is parsed.

    ``targets_for_subject`` takes a mapping and reads no file, which is why the digest could not be
    tested at §3.5a: this is the seam T6's second mutation needs.  The comparison is on the whole
    digest, not a prefix.
    """
    path = Path(artifact_path) if artifact_path is not None else DEFAULT_DATA_DIR / P7_2B_CALIBRATION_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path}: P7.2b's calibration artifact is not here")
    digest = _sha256_file(path)
    if digest != P7_2B_CALIBRATION_SHA256:
        raise ValueError(
            f"{path.name}: sha256 {digest} is not the pinned {P7_2B_CALIBRATION_SHA256}. A17's "
            "targets are READ from this artifact and never recomputed, so a file that has moved "
            "under the pin is a different registration; refusing rather than evaluating prompts "
            "nobody registered"
        )
    return json.loads(path.read_bytes())


def checkpoint_path_for(subject: str, seed: int, *, output_root: str | Path) -> Path:
    """Where one subject-seed checkpoint lives, from P7.2b's own subject table."""
    from offline.transfer_calibration import SUBJECTS as SUBJECT_LAYOUT

    if subject not in SUBJECT_LAYOUT:
        raise ValueError(f"unknown subject {subject!r}; A17(c) registers {sorted(SUBJECT_LAYOUT)}")
    spec = SUBJECT_LAYOUT[subject]
    return Path(output_root) / spec["subdir"] / f"{spec['stem']}{int(seed)}.pt"


def _committed_checkpoint_digest(subject: str, seed: int, *, data_dir: Path) -> tuple[str, str]:
    """The digest a COMMITTED artifact records for this checkpoint, and the artifact's name."""
    record_name = CHECKPOINT_RECORD[subject]
    record_path = data_dir / record_name
    if not record_path.is_file():
        raise FileNotFoundError(
            f"{record_path}: {subject}'s checkpoints are pinned against this committed artifact "
            "(G1), and it is not here"
        )
    record = json.loads(record_path.read_bytes())

    if subject == "mappo1000":
        entry = record.get("checkpoints", {}).get(str(int(seed)))
        if not isinstance(entry, Mapping) or "sha256" not in entry:
            raise ValueError(
                f"{record_name}: no checkpoints[{seed!r}].sha256 for {subject}; a checkpoint absent "
                "from its committed record has no integrity evidence at consumption"
            )
        return str(entry["sha256"]), record_name

    matching = [
        row
        for row in record.get("runs", ())
        if str(row.get("tier")) == "mix50"
        and str(row.get("method")) == MIX50_TRAINING_METHOD
        and int(row.get("seed", -1)) == int(seed)
    ]
    if len(matching) != 1:
        raise ValueError(
            f"{record_name}: {len(matching)} rows match (tier=mix50, "
            f"method={MIX50_TRAINING_METHOD}, seed={seed}); exactly one is required. The key "
            "carries the method because this artifact records four methods per (tier, seed) and "
            "three of them are other checkpoints"
        )
    return str(matching[0]["file_sha256"]), record_name


def checkpoint_identity(
    subject: str, seed: int, *, output_root: str | Path, data_dir: str | Path | None = None
) -> dict[str, Any]:
    """G1: this checkpoint's digest, recomputed from the file and checked at CONSUMPTION.

    ``BRIEF_27`` B3(a)'s rule: *a digest checked once is not a digest checked when used.*  The file
    is hashed here, every record that names it is compared, and the record names are returned so the
    chunk can say which evidence exists for which subject instead of implying they are equal --
    ``mappo1000`` has no campaign manifest at all (``DEFERRED`` 56).
    """
    path = checkpoint_path_for(subject, seed, output_root=output_root)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: {subject} seed {seed}'s checkpoint is not on disk")
    digest = _sha256_file(path)

    data = _data_dir(data_dir)
    committed, record_name = _committed_checkpoint_digest(subject, seed, data_dir=data)
    checked_against = [record_name]
    if digest != committed:
        raise ValueError(
            f"{path}: file sha256 {digest} is not the {committed} that {record_name} records for "
            f"{subject} seed {seed}. Different weights under the same filename would evaluate as "
            "the registered subject and could not be told apart afterwards"
        )

    manifest_name = LOCAL_CHECKPOINT_MANIFEST[subject]
    manifest_path = Path(output_root) / manifest_name if manifest_name else None
    if manifest_path is not None and manifest_path.is_file():
        listed = _manifest_digests(manifest_path)
        relative = str(path.relative_to(Path(output_root)))
        if relative not in listed:
            raise ValueError(
                f"{relative} is not listed in {manifest_name}, which does list this campaign's "
                "other checkpoints; a partial manifest is a gap, not a pass"
            )
        if listed[relative] != digest:
            raise ValueError(
                f"{path}: file sha256 {digest} is not {manifest_name}'s {listed[relative]}"
            )
        checked_against.append(manifest_name)

    return {
        "subject": subject,
        "seed": int(seed),
        "path": str(path),
        "file_sha256": digest,
        "sha256_checked_against": checked_against,
        "local_manifest": manifest_name,
        "deferred_56": manifest_name is None,
    }


def _manifest_digests(path: Path) -> dict[str, str]:
    """A ``sha256sum``-format manifest as ``{relative path: digest}``."""
    digests: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        digests[name.strip()] = digest.strip()
    return digests


def demand_identity(draw_id: int, *, out_root: str | Path) -> dict[str, Any]:
    """G2: the draw's demand, pinned by TWO digests against P7.2a's own provenance.

    The ``.sumocfg`` only *names* the routes file, so a regenerated ``routes.rou.xml`` leaves the
    cfg's bytes -- and its digest -- untouched.  rho pairs an arm with the two anchors of the same
    draw, and all three cells must provably have run the same demand, across a resume and across
    the days a campaign may span.
    """
    from offline.materialise_draws import parity_sumocfg_path

    config_path = parity_sumocfg_path(SCENARIO_KEY, int(draw_id), out_root=out_root)
    parity_dir = config_path.parent
    routes_path = parity_dir / PARITY_ROUTES_NAME
    provenance_path = parity_dir / PARITY_PROVENANCE_NAME
    for path in (config_path, routes_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"draw {draw_id}: {path} is absent; P7.2a materialises the band into the MAIN "
                "tree's scenarios/draws and this task only reads it"
            )

    recorded = json.loads(provenance_path.read_bytes()).get("files", {})
    identity = {
        "draw_id": int(draw_id),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "routes_sha256": _sha256_file(routes_path),
    }
    for name, key in ((PARITY_CONFIG_NAME, "config_sha256"), (PARITY_ROUTES_NAME, "routes_sha256")):
        if name not in recorded:
            raise ValueError(
                f"draw {draw_id}: {PARITY_PROVENANCE_NAME} records no digest for {name}, so the "
                "demand cannot be pinned to what P7.2a generated"
            )
        if str(recorded[name]) != identity[key]:
            raise ValueError(
                f"draw {draw_id}: {name} hashes to {identity[key]} but {PARITY_PROVENANCE_NAME} "
                f"records {recorded[name]}. The demand on disk is not the demand P7.2a "
                "materialised; every number computed on it would describe a different scenario"
            )
    return identity


def halting_check_for(draw_id: int) -> bool:
    """Amendment C2: ON for every cell on :data:`HALTING_CHECK_DRAW`, OFF everywhere else."""
    return int(draw_id) == HALTING_CHECK_DRAW


def env_for_cell(
    cell: Mapping[str, Any],
    *,
    out_root: str | Path,
    sentinel_out_dir: str | Path = "/nonexistent",
) -> Any:
    """A2/G3: the observed env this cell runs in -- aligned for a DT, UNWRAPPED for an anchor.

    The module docstring says why: ``align_info`` drops outgoing lanes and re-keys the survivors to
    CityFlow ids, and an anchor's pressure is a difference over the env's own SUMO lane ids.  The
    arm is passed through so the anchor's env is built with its own name in the collection args,
    exactly as P7.1 built it.
    """
    import offline.aligned_env as aligned_env

    kind = str(cell["kind"])
    draw_id = int(cell["draw_id"])
    factory = (
        aligned_env.aligned_observer_env_for_draw
        if kind == "dt"
        else aligned_env.observer_env_for_draw
    )
    arm = "maxpressure" if kind == "dt" else str(cell["arm"])
    return factory(
        SCENARIO_KEY,
        draw_id,
        out_root=out_root,
        halting_check=halting_check_for(draw_id),
        arm=arm,
        sentinel_out_dir=sentinel_out_dir,
    )


def assert_env_matches_cell(cell: Mapping[str, Any], env: Any) -> None:
    """Amendment C8: the cell builder's explicit refusal, on every cell.

    Two directions, because both are silent failures.  An anchor through A16's door raises
    ``KeyError: 'road_1_1_2_0'`` deep inside ``_phase_pressure`` -- after a SUMO process has started
    -- or, worse, would not raise at all if the door ever stopped dropping outgoing lanes.  A DT
    cell on a raw ``SumoEnv`` sees 32-wide info in SUMO lane order, which a CityFlow-trained model
    consumes happily and turns into a number that means nothing.
    """
    from offline.aligned_env import AlignedEnv

    kind = str(cell["kind"])
    aligned = isinstance(env, AlignedEnv)
    if kind == "anchor" and aligned:
        raise TypeError(
            f"{cell['arm']} is an anchor and must run on the observed, UNWRAPPED env: align_info "
            "drops outgoing lanes and re-keys the rest to CityFlow ids, while the anchor's pressure "
            "is a difference over the env's own SUMO lane ids (Amendment A2, measured). A16 is "
            "untouched by this -- the door is the route into a CityFlow-trained MODEL's frame, and "
            "an anchor has no frame to enter"
        )
    if kind == "dt" and not aligned:
        raise TypeError(
            f"a {kind} cell must run behind AlignedEnv: A16 makes align_info the only door into the "
            "canonical 25-wide frame, and a CityFlow-trained model handed raw 32-wide SUMO info "
            "would produce a number with no meaning rather than an error"
        )


def dt_choose(
    env: Any,
    *,
    checkpoint_path: str | Path,
    target_rtg: float,
    declared_gradient_steps: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """The DT's decision function, and the per-decision series a reviewer re-derives from.

    ``BRIEF_36`` E4: **every** decision is ``act(info, explore=False, update_memory=True)``.  The
    default is ``explore=True``, which samples from the masked softmax through an unseeded
    ``torch.multinomial`` -- P7.2b's smoke took that default once and ``n_decisions_in_support``
    moved 271 -> 231 between two runs of the same seed, checkpoint and draw.

    ``DEFERRED`` 81 closes here: the RTG series and the per-decision reward series are both
    recorded, so ``rtg_advanced_every_decision`` is re-derivable by a reviewer instead of being a
    flag they must trust.  The RTG is read BEFORE the call, which is what makes the shift-by-one
    rule in ``transfer_calibration.rtg_advanced_every_decision`` the right comparison.
    """
    import numpy as np

    from offline.rtg_calibration import agent_with_target
    from offline.transfer_calibration import DECLARED_GRADIENT_STEPS

    steps = DECLARED_GRADIENT_STEPS if declared_gradient_steps is None else int(declared_gradient_steps)
    agent = agent_with_target(
        env, checkpoint_path, declared_gradient_steps=steps, target_rtg=float(target_rtg)
    )
    ix_id = str(list(env.intersections)[0].id)

    diagnostics: dict[str, Any] = {
        "agent": agent,
        "intersection": ix_id,
        "rtg_series": [],
        "reward_series": [],
        "actions": [],
    }

    def choose(_env: Any, info: Mapping[str, Any]) -> Any:
        diagnostics["rtg_series"].append(float(agent.current_rtg()[ix_id]))
        payload = info["intersections"][ix_id]
        diagnostics["reward_series"].append(
            None if "reward" not in payload else float(payload["reward"])
        )
        action = agent.act(info, explore=False, update_memory=True)
        diagnostics["actions"].append(int(np.asarray(action).reshape(-1)[0]))
        return action

    return choose, diagnostics


def anchor_choose(
    env: Any, *, cell: Mapping[str, Any], config_path: str | Path
) -> tuple[Any, dict[str, Any]]:
    """The anchor's decision function, through ``collect.POLICIES`` as P7.1 ran it.

    ``build_policy`` seeds ``default_rng(args.base_seed)``, so ``random``'s five POLICY seeds enter
    there (G6) and vary the action stream ONLY: the engine seed stays ``horizon_rollout``'s
    :data:`ENGINE_SEED` for every cell, which is A18(c)'s rule.  ``fixedtime`` carries
    ``COLLECT_SETTINGS``' ``--fixed-time-k 4``, the schedule P7.1's frozen anchor rows were measured
    with; nothing about the anchors is chosen here.

    It returns the same ``(choose, diagnostics)`` pair :func:`dt_choose` does, and for the same
    reason: the decision COUNT and the action range are properties of the episode that ran, and a
    cell that reported the registered horizon without counting it would satisfy
    :func:`validate_cell_payload` by construction.  An anchor keeps no RTG, so those series stay
    empty rather than being invented.
    """
    import numpy as np

    from offline.sumo_att_reference import build_policy, collect_style_args

    arm = str(cell["arm"])
    policy_seed = cell.get("seed") if arm == "random" else None
    base_seed = ENGINE_SEED if policy_seed is None else int(policy_seed)
    args = collect_style_args(
        "sumo", arm, config_path, episodes=1, base_seed=base_seed, sentinel_out_dir="/nonexistent"
    )
    policy = build_policy(env, args)
    diagnostics: dict[str, Any] = {"rtg_series": [], "reward_series": [], "actions": []}

    def choose(_env: Any, info: Mapping[str, Any]) -> Any:
        action = policy(info)
        diagnostics["actions"].append(int(np.asarray(action).reshape(-1)[0]))
        return action

    return choose, diagnostics


def att_env_from_info(info: Mapping[str, Any]) -> float:
    """G4: the env's own ATT at the horizon, SUBSCRIPTED -- never defaulted.

    ``horizon_rollout`` reads ``info.get("average_travel_time", 0.0)`` (``horizon_metric.py:99``),
    and that default is the hazard: a missing key would make ``att_horizon`` 0.0, which is not a
    measurement but the absence of one wearing the same type.  P7.1's own expression subscripts the
    key (``sumo_att_reference.py:1242``); this is that expression with its failure named.
    """
    if "average_travel_time" not in info:
        raise ValueError(
            "the final info carries no 'average_travel_time', so att_env cannot be measured. "
            "horizon_rollout would substitute 0.0 here and report it as an ATT; refusing instead, "
            "because a defaulted metric is indistinguishable from a real one once written down"
        )
    return float(info["average_travel_time"])


class _StepTap:
    """Forwards every call, and remembers the last ``info`` ``step`` returned (G4's second route).

    ``horizon_rollout`` keeps no reference to the final ``info``: it appends
    ``info.get("average_travel_time", 0.0)`` and returns the reduction.  The tap gives the cell the
    same ``info`` P7.1's ``_record`` reads, so ``att_env`` and ``att_horizon`` are computed by two
    independent routes and compared under ``==`` rather than one being assumed from the other.

    The policy is built on the UNTAPPED env and only the rollout is tapped, so nothing about agent
    construction or MaxPressure's roadnet introspection passes through this object.
    """

    def __init__(self, env: Any) -> None:
        self._env = env
        self.last_info: dict[str, Any] | None = None

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        info = self._env.reset(**kwargs)
        self.last_info = info
        return info

    def step(self, action: Any) -> tuple[Any, bool, bool, dict[str, Any]]:
        reward, terminated, truncated, info = self._env.step(action)
        self.last_info = info
        return reward, terminated, truncated, info

    def __getattr__(self, name: str) -> Any:
        if name in {"_env", "last_info"}:
            raise AttributeError(name)
        return getattr(self._env, name)


def validate_cell_payload(
    payload: Mapping[str, Any], *, cell: Mapping[str, Any] | None = None
) -> None:
    """Every property a cell must carry, each refusal naming its own reason.

    Run by :func:`run_cell` before a chunk is written and again by :func:`report` over every chunk
    it reads: a chunk can be hand-made, and a resumed campaign can carry chunks written by an older
    revision.  Digest comparisons against files on disk are NOT here -- they need roots, and they
    live in :func:`report` and :func:`chunk_is_reusable`.
    """
    from offline.transfer_calibration import (
        EXPECTED_DECISIONS,
        EXPECTED_TIME_TO_TELEPORT,
        PARITY_VTYPE_ID,
    )

    label = cell_chunk_name(payload)
    if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"{label}: format_version {payload.get('format_version')!r} is not "
            f"{ARTIFACT_FORMAT_VERSION!r}"
        )
    if cell is not None:
        for key in ("kind", "subject", "arm", "seed", "draw_id"):
            if payload.get(key) != cell.get(key):
                raise ValueError(
                    f"{label}: the chunk says {key}={payload.get(key)!r} and the cell it was asked "
                    f"for says {cell.get(key)!r}; a chunk that describes another cell is not "
                    "evidence about this one"
                )
    arm = str(payload.get("arm"))
    if arm not in DECLARED_ARM_NAMES:
        raise ValueError(
            f"{label}: {arm!r} is not a declared arm. BRIEF_37 §2 closed the set before any cell "
            f"ran ({sorted(DECLARED_ARM_NAMES)}); Rule B at k = 5 and k = 20 are the few-shot "
            "prompts A18(d) attaches to fine-tuned models, and a zero-shot outcome for one of them "
            "is an evaluation nobody registered"
        )
    if int(payload["n_teleports"]) != 0:
        raise ValueError(
            f"{label}: {payload['n_teleports']} teleport(s) under A15(c)'s teleport-free regime"
        )
    if str(payload["time_to_teleport_option"]) != EXPECTED_TIME_TO_TELEPORT:
        raise ValueError(
            f"{label}: SUMO reported time-to-teleport {payload['time_to_teleport_option']!r}, not "
            f"{EXPECTED_TIME_TO_TELEPORT!r}; this is not the teleport-free regime A15(c) registers"
        )
    if list(payload["vehicle_types_seen"]) != [PARITY_VTYPE_ID]:
        raise ValueError(
            f"{label}: the engine ran {payload['vehicle_types_seen']!r}, not [{PARITY_VTYPE_ID!r}]"
        )
    if int(payload["decisions"]) != EXPECTED_DECISIONS:
        raise ValueError(
            f"{label}: {payload['decisions']} decisions, not {EXPECTED_DECISIONS}; the episode did "
            "not run to the horizon"
        )
    if payload.get("actions_in_range") is not True:
        raise ValueError(f"{label}: an action was outside the intersection's legal range")
    if str(payload.get("calibration_sha256")) != P7_2B_CALIBRATION_SHA256:
        raise ValueError(
            f"{label}: it records targets from calibration sha256 "
            f"{payload.get('calibration_sha256')!r}, not the pinned {P7_2B_CALIBRATION_SHA256!r}"
        )
    if int(payload["engine_seed_requested"]) != ENGINE_SEED:
        raise ValueError(
            f"{label}: engine_seed_requested {payload['engine_seed_requested']} is not A18(c)'s "
            f"{ENGINE_SEED}"
        )

    checked = bool(payload.get("halting_checked"))
    if checked != halting_check_for(int(payload["draw_id"])):
        raise ValueError(
            f"{label}: halting_checked is {checked} on draw {payload['draw_id']}, but Amendment C2 "
            f"declares the cross-check ON for draw {HALTING_CHECK_DRAW} and OFF elsewhere"
        )
    if checked and int(payload["halting_n_disagreeing_lane_seconds"]) != 0:
        raise ValueError(
            f"{label}: the recorder's halting classification disagreed with SUMO's own on "
            f"{payload['halting_n_disagreeing_lane_seconds']} lane-second(s). Amendment C2: a "
            "recorder disagreement is a finding that stops the campaign, exactly as a teleport is"
        )

    if str(payload.get("kind")) == "dt":
        if payload.get("rtg_first") != payload.get("target_rtg"):
            raise ValueError(
                f"{label}: rtg_first {payload.get('rtg_first')!r} is not the declared target "
                f"{payload.get('target_rtg')!r}; the prompt did not take effect and the cell ran "
                "some other conditioning"
            )
        series = payload.get("rtg_series")
        rewards = payload.get("reward_series")
        if not series or rewards is None or len(series) != len(rewards):
            raise ValueError(
                f"{label}: the RTG series and the reward series are recorded per decision and must "
                "agree in length (DEFERRED 81: both are stored so a reviewer re-derives the "
                "advance rule instead of trusting a flag)"
            )


def chunk_path(cell: Mapping[str, Any], *, work_dir: str | Path) -> Path:
    """Pure path arithmetic: where one cell's chunk lives."""
    return Path(work_dir) / cell_chunk_name(cell)


def write_chunk(payload: Mapping[str, Any], *, work_dir: str | Path) -> Path:
    """E3(d): atomic tmp-then-replace, so a chunk is either whole or absent."""
    path = chunk_path(payload, work_dir=work_dir)
    _write_json(path, payload)
    return path


def move_aside(path: str | Path) -> Path:
    """Move an unusable chunk into ``failed/`` rather than overwriting it.

    A chunk that failed its own re-validation is evidence about a run, and ``report``'s glob must
    not see it.  Never overwritten: the suffix grows instead.
    """
    source = Path(path)
    destination = source.parent / "failed" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = destination.with_name(f"{source.stem}.{suffix}{source.suffix}")
    source.replace(destination)
    return destination


def chunk_is_reusable(
    payload: Mapping[str, Any],
    *,
    cell: Mapping[str, Any],
    out_root: str | Path,
    output_root: str | Path,
    data_dir: str | Path | None = None,
) -> bool:
    """E3(d): may a restart SKIP this cell?  Only on evidence RE-DERIVED from disk.

    ``offline/campaigns/p5_3b.sh`` skipped on ``[ -f ]`` alone and a bad chunk survived every
    restart; ``nortg_decomposition.chunk_is_reusable`` records what that cost.  The stored verdict
    is exactly what a half-written or hand-edited chunk would lie about, so the checkpoint's, the
    cfg's and the routes file's digests are recomputed from the files themselves and compared with
    what the chunk claims (G2: a cfg digest alone would not notice a regenerated routes file).

    Any disagreement, and any chunk this function cannot read, returns ``False``: the cell is
    re-run.  A file we cannot read is a file we have no evidence about, and a predicate that
    crashed on it would take the driver down on every restart.
    """
    if not isinstance(payload, Mapping):
        return False
    try:
        validate_cell_payload(payload, cell=cell)
        demand = demand_identity(int(cell["draw_id"]), out_root=out_root)
        if str(payload["config_sha256"]) != demand["config_sha256"]:
            return False
        if str(payload["routes_sha256"]) != demand["routes_sha256"]:
            return False
        if str(cell["kind"]) == "dt":
            identity = checkpoint_identity(
                str(cell["subject"]),
                int(cell["seed"]),
                output_root=output_root,
                data_dir=data_dir,
            )
            if str(payload["checkpoint_sha256"]) != identity["file_sha256"]:
                return False
    except (KeyError, TypeError, ValueError, AttributeError, FileNotFoundError):
        return False
    return True


def reusable_chunk_at(path: str | Path, **kwargs: Any) -> bool:
    """:func:`chunk_is_reusable` over a file that may not exist or may not parse."""
    target = Path(path)
    if not target.is_file():
        return False
    try:
        payload = json.loads(target.read_bytes())
    except (ValueError, OSError):
        return False
    return chunk_is_reusable(payload, **kwargs)


def run_cell(
    cell: Mapping[str, Any],
    *,
    out_root: str | Path,
    output_root: str | Path,
    data_dir: str | Path | None = None,
    calibration: Mapping[str, Any] | None = None,
    canary_seconds: float | None = None,
) -> dict[str, Any]:
    """One observed SUMO episode for one cell, as a validated chunk.  It writes NOTHING.

    The order is the design: the demand and the checkpoint are pinned BEFORE a simulator starts, the
    env is built and its shape refused if it does not match the cell (C8), the policy is built on
    the untapped env, the rollout runs on the tap (G4), and the payload is validated in full before
    it is returned.  The caller writes it -- so a cell that fails leaves no chunk, no directory and
    no trace except the exception.
    """
    import time

    from offline.horizon_metric import horizon_rollout
    from offline.rtg_calibration import in_support_counts
    from offline.sumo_att_reference import reconstruct_sumo_episode
    from offline.transfer_calibration import rtg_advanced_every_decision, subject_facts

    cell = dict(cell)
    kind = str(cell["kind"])
    draw_id = int(cell["draw_id"])
    arm = str(cell["arm"])

    demand = demand_identity(draw_id, out_root=out_root)
    checkpoint: dict[str, Any] | None = None
    target_rtg: float | None = None
    facts = None
    if kind == "dt":
        subject = str(cell["subject"])
        artifact = load_calibration() if calibration is None else calibration
        target_rtg = float(targets_for_subject(subject, artifact)[arm]["target_rtg"])
        checkpoint = checkpoint_identity(
            subject, int(cell["seed"]), output_root=output_root, data_dir=data_dir
        )
        facts = subject_facts(subject, output_root=output_root)

    started = time.perf_counter()
    env = env_for_cell(cell, out_root=out_root)
    diagnostics: dict[str, Any] = {"rtg_series": [], "reward_series": [], "actions": []}
    try:
        assert_env_matches_cell(cell, env)
        if kind == "dt":
            choose, diagnostics = dt_choose(
                env, checkpoint_path=checkpoint["path"], target_rtg=target_rtg  # type: ignore[index]
            )
        else:
            choose, diagnostics = anchor_choose(env, cell=cell, config_path=demand["config_path"])
        tap = _StepTap(env)
        rollout = horizon_rollout(tap, choose, 1, ENGINE_SEED)
        built = reconstruct_sumo_episode(env.recorder)
        att_env = att_env_from_info(tap.last_info or {})
        types_seen = sorted({env._sumo.vehicle.getTypeID(v) for v in env._sumo.vehicle.getIDList()})
        option = str(env._sumo.simulation.getOption("time-to-teleport"))
        engine_seed_drawn = int(env._engine_seed)
    finally:
        env.close()
    seconds = time.perf_counter() - started

    if att_env != rollout.att_horizon:
        raise ValueError(
            f"att_env {att_env!r} and att_horizon {rollout.att_horizon!r} disagree. They are two "
            "routes to ONE quantity -- the final info's average_travel_time, and horizon_rollout's "
            "last sample of the same key (G4) -- so a difference means the loop this cell ran is "
            "not the loop P7.1 measured, and no comparison with the frozen anchors would be valid"
        )

    actions = diagnostics["actions"]
    rtg_series = diagnostics["rtg_series"]
    reward_series = diagnostics["reward_series"]
    support = None
    if kind == "dt" and rtg_series and facts is not None:
        counts = in_support_counts(
            rtg_series,
            rtg_min=facts.support_range_over_the_split[0],
            rtg_max=facts.support_range_over_the_split[1],
        )
        support = {
            "n_decisions_in_support": counts.in_support,
            "n_decisions_below": counts.below,
            "n_decisions_above": counts.above,
            "support_range": list(facts.support_range_over_the_split),
            "training_set_return_min": facts.training_set_return_min,
        }

    checked = halting_check_for(draw_id)
    payload: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        **cell,
        "policy_seed": int(cell["seed"]) if kind == "anchor" and arm == "random" else None,
        "engine_seed_requested": ENGINE_SEED,
        "engine_seed_drawn": engine_seed_drawn,
        # Counted, for both kinds: a decision count taken from the declared horizon rather than
        # from the episode would make validate_cell_payload's check a tautology.
        "decisions": len(actions),
        "actions_in_range": bool(actions) and all(0 <= int(action) < 8 for action in actions),
        "episode_reward": rollout.episode_reward,
        "att_horizon": rollout.att_horizon,
        "att_env": att_env,
        "att_running_mean": rollout.att_running_mean,
        "horizon_vehicle_count": rollout.final_vehicle_count,
        "e_sumo": built.e_sumo.value,
        "p_sumo": built.p_sumo.value,
        "w_sumo": built.w_sumo.value,
        "mean_depart_delay": built.mean_depart_delay,
        "n_created": built.e_sumo.n_ids,
        "n_entered": built.n_departed,
        "n_never_entered": built.n_never_inserted,
        "n_pending_at_horizon": built.n_pending_at_horizon,
        "n_teleports": built.n_teleports,
        "n_vanished_without_arrival": built.n_vanished_without_arrival,
        "n_arrived_never_observed_at_a_boundary": built.n_arrived_never_observed_at_a_boundary,
        "max_abs_depart_clock_deviation": built.max_abs_depart_clock_deviation,
        "n_observations": built.n_observations,
        "vehicle_types_seen": types_seen,
        "time_to_teleport_option": option,
        "halting_checked": checked,
        "halting_max_abs_difference": built.halting.max_abs_difference if checked else None,
        "halting_n_lane_seconds": built.halting.n_lane_seconds if checked else None,
        "halting_n_disagreeing_lane_seconds": (
            built.halting.n_disagreeing_lane_seconds if checked else None
        ),
        "config_sha256": demand["config_sha256"],
        "routes_sha256": demand["routes_sha256"],
        "calibration_sha256": P7_2B_CALIBRATION_SHA256,
        "checkpoint": None if checkpoint is None else checkpoint["path"],
        "checkpoint_sha256": None if checkpoint is None else checkpoint["file_sha256"],
        "sha256_checked_against": None if checkpoint is None else checkpoint["sha256_checked_against"],
        "target_rtg": target_rtg,
        "rtg_first": rtg_series[0] if rtg_series else None,
        "rtg_last": rtg_series[-1] if rtg_series else None,
        "rtg_series": rtg_series or None,
        "reward_series": reward_series or None,
        "rtg_advanced_every_decision": (
            rtg_advanced_every_decision(rtg_series, reward_series) if rtg_series else None
        ),
        "n_decisions_in_support": None if support is None else support["n_decisions_in_support"],
        "support_range": None if support is None else support["support_range"],
        "in_support_counts": support,
        "canary_seconds": None if canary_seconds is None else float(canary_seconds),
        "seconds": seconds,
        **_git_provenance(),
    }
    validate_cell_payload(payload, cell=cell)
    return payload


#: The only chunk fields that may reach ``docs/data/``.  A whitelist, not a blacklist: a field added
#: to the chunk later is excluded by default rather than by being remembered.
_PUBLISHED_FIELDS: tuple[str, ...] = (
    "kind", "subject", "arm", "seed", "policy_seed", "draw_id", "stage",
    "att_env", "att_horizon", "att_running_mean", "e_sumo", "p_sumo", "w_sumo",
    "mean_depart_delay", "episode_reward", "horizon_vehicle_count", "decisions",
    "engine_seed_requested", "engine_seed_drawn",
    "n_created", "n_entered", "n_never_entered", "n_pending_at_horizon", "n_teleports",
    "n_vanished_without_arrival", "n_arrived_never_observed_at_a_boundary",
    "max_abs_depart_clock_deviation", "vehicle_types_seen", "time_to_teleport_option",
    "halting_checked", "halting_max_abs_difference", "halting_n_lane_seconds",
    "halting_n_disagreeing_lane_seconds",
    "target_rtg", "rtg_first", "rtg_last", "rtg_series", "reward_series",
    "rtg_advanced_every_decision", "n_decisions_in_support", "support_range",
    "checkpoint", "checkpoint_sha256", "sha256_checked_against",
    "config_sha256", "routes_sha256", "calibration_sha256",
    "canary_seconds", "seconds", "git_commit", "git_dirty",
)


def _read_canary_record(work: Path) -> dict[str, Any]:
    """Read ``<work>/canary.json`` and RE-RUN the correctness half on its facts.

    Amendment E1.4: the record is a CLAIM, and a claim is not evidence -- ``report`` checks the
    facts itself rather than trusting whatever wrote the file.  A missing file is a refusal, not a
    fallback: falling back on the chunks' own ``canary_seconds`` is precisely how P7.2b's run 3
    published run 1's canary as its own.
    """
    from offline.transfer_calibration import CANARY_RECORD_NAME, check_canary

    path = work / CANARY_RECORD_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: this run wrote no {CANARY_RECORD_NAME}. The driver writes it from the canary "
            "line immediately after the token is consumed; a work directory without one is not a "
            "run, and the chunks' canary_seconds is the canary of whoever ROLLED them"
        )
    record = json.loads(path.read_bytes())
    if not isinstance(record, dict) or not isinstance(record.get("facts"), dict):
        raise ValueError(f"{CANARY_RECORD_NAME}: the canary record has no 'facts' object")
    check_canary(record["facts"])
    return {**record, "seconds": float(record["seconds"])}


def _rho_pair(chunk: Mapping[str, Any], anchors: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """rho under BOTH registered definitions, against the anchors OF THE SAME DRAW."""
    return {
        f"rho_{key}": rho(float(chunk[key]), float(anchors["fixedtime"][key]), float(anchors["maxpressure"][key]))
        for key in ("e_sumo", "att_env")
    }


def report(
    *,
    work_dir: str | Path,
    out_path: str | Path,
    output_root: str | Path,
    out_root: str | Path,
    data_dir: str | Path | None = None,
    stage: str | None = None,
    cells: Sequence[Mapping[str, Any]] | None = None,
    stage1_path: str | Path | None = None,
) -> dict[str, Any]:
    """E3(e)/B2: the committed artifact.  EVERY refusal precedes EVERY write, including the last.

    The order below is the design, not an accident of drafting:

    1. the caller-supplied cell set may not produce a committed artifact;
    2. P7.2b's calibration artifact still hashes to its pin;
    3. this run's canary, re-checked rather than believed;
    4. every chunk validated in full, again;
    5. completeness against a DECLARATION that exists independently of the chunks;
    6. the digests re-derived from disk -- demand (G2) and checkpoint (G1);
    7. the pairing, per draw, refused when broken;
    8. rho, the aggregates, H3's inequalities, the exploratory contrast, the diagnostics;
    9. a scan over the SERIALISED bytes -- and only then ``_write_json``.

    Step 9 exists because step 4 can be widened by a future edit and a reordered write would publish
    before anything checked it.  ``tests/test_transfer_curve.py`` reaches it with delivered code.
    """
    from offline.dt_gate import EpisodeResult, mean_ci95
    from offline.offline_baselines import paired_comparison
    from offline.transfer_calibration import CANARY_MAX_SECONDS, CANARY_RECORD_NAME

    work = Path(work_dir)
    target_path = Path(out_path)
    data = _data_dir(data_dir)

    # ---------------------------------------------------------------- 1. the cell-set guard
    declared = list(declared_cells(stage) if cells is None else cells)
    if cells is not None:
        if stage is not None:
            declared = [cell for cell in declared if cell["stage"] == stage]
        committed_dir = (_REPO_ROOT / "docs" / "data").resolve()
        if target_path.resolve().parent == committed_dir:
            raise ValueError(
                f"{target_path}: refusing to write the committed artifact from a caller-supplied "
                "cell set. The completeness check is only as good as the declaration it compares "
                "against, so the campaign passes none and gets declared_cells(stage) -- all 4,700. "
                "The parameter exists so a test can declare a small campaign of its own"
            )
    if not declared:
        raise ValueError(f"the declared cell set for stage {stage!r} is empty")

    # ---------------------------------------------------------------- 2. the calibration pin
    calibration = load_calibration(data / P7_2B_CALIBRATION_NAME)

    # ---------------------------------------------------------------- 3. this run's canary
    canary_record = _read_canary_record(work)

    # ---------------------------------------------------------------- 4. every chunk, validated
    chunks: dict[str, dict[str, Any]] = {}
    for path in sorted(work.glob("cell_*.json")):
        payload = json.loads(path.read_bytes())
        validate_cell_payload(payload)
        name = cell_chunk_name(payload)
        if name != path.name:
            raise ValueError(
                f"{path.name}: its content names the cell {name}; a chunk under another cell's "
                "filename was skipped as complete on every restart once already (BRIEF_33 C1.2)"
            )
        chunks[name] = payload

    # ---------------------------------------------------------------- 5. completeness
    declared_by_name = {cell_chunk_name(cell): cell for cell in declared}
    missing = sorted(set(declared_by_name) - set(chunks))
    if missing:
        raise ValueError(
            f"{len(missing)} declared cell(s) have no chunk (first: {missing[:3]}); the campaign is "
            "incomplete and a partial artifact would report an arm on fewer draws than registered"
        )
    extra = sorted(set(chunks) - set(declared_by_name))
    if extra:
        raise ValueError(
            f"{len(extra)} chunk(s) are not declared cells (first: {extra[:3]}); an undeclared cell "
            "reaching the artifact is an evaluation nobody registered"
        )
    for name, payload in chunks.items():
        validate_cell_payload(payload, cell=declared_by_name[name])

    # ---------------------------------------------------------------- 6. the digests, from disk
    demand_by_draw: dict[int, dict[str, Any]] = {}
    identity_by_checkpoint: dict[tuple[str, int], dict[str, Any]] = {}
    for name, payload in sorted(chunks.items()):
        draw_id = int(payload["draw_id"])
        if draw_id not in demand_by_draw:
            demand_by_draw[draw_id] = demand_identity(draw_id, out_root=out_root)
        demand = demand_by_draw[draw_id]
        for key in ("config_sha256", "routes_sha256"):
            if str(payload[key]) != demand[key]:
                raise ValueError(
                    f"{name}: {key} {payload[key]!r} is not draw {draw_id}'s {demand[key]!r}. The "
                    "cell ran on demand that is not what is on disk now, so it cannot be paired "
                    "with the anchors of this draw"
                )
        if str(payload["kind"]) == "dt":
            key_pair = (str(payload["subject"]), int(payload["seed"]))
            if key_pair not in identity_by_checkpoint:
                identity_by_checkpoint[key_pair] = checkpoint_identity(
                    key_pair[0], key_pair[1], output_root=output_root, data_dir=data
                )
            declared_sha = identity_by_checkpoint[key_pair]["file_sha256"]
            if str(payload["checkpoint_sha256"]) != declared_sha:
                raise ValueError(
                    f"{name}: it records checkpoint sha256 {payload['checkpoint_sha256']!r}, not "
                    f"the {declared_sha!r} that {key_pair[0]} seed {key_pair[1]}'s committed record "
                    "pins today; the weights that produced this cell are not the registered ones"
                )

    # ---------------------------------------------------------------- 7-8. pairing, rho, blocks
    anchors_by_draw: dict[int, dict[str, Mapping[str, Any]]] = {}
    for payload in chunks.values():
        if payload["arm"] in ("fixedtime", "maxpressure"):
            anchors_by_draw.setdefault(int(payload["draw_id"]), {})[str(payload["arm"])] = payload

    rows: list[dict[str, Any]] = []
    for name in sorted(chunks):
        payload = chunks[name]
        draw_id = int(payload["draw_id"])
        anchors = anchors_by_draw.get(draw_id, {})
        if set(anchors) != {"fixedtime", "maxpressure"}:
            raise ValueError(
                f"{name}: draw {draw_id} carries anchors {sorted(anchors)}, so rho has no "
                "denominator of its own draw. Pairing is PER DRAW because the demand differs by "
                "draw, and a ratio built from another draw's anchors normalises nothing"
            )
        row = {field: payload.get(field) for field in _PUBLISHED_FIELDS}
        row.update(_rho_pair(payload, anchors))
        rows.append(row)

    by_seed: list[dict[str, Any]] = []
    by_arm: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for spec in DECLARED_ARMS:
            arm_rows = [r for r in rows if r["subject"] == subject and r["arm"] == spec.name]
            if not arm_rows:
                continue
            for seed in TRAINING_SEEDS:
                seed_rows = [r for r in arm_rows if r["seed"] == seed]
                if not seed_rows:
                    continue
                by_seed.append(
                    {
                        "subject": subject,
                        "arm": spec.name,
                        "seed": int(seed),
                        "n_draws": len(seed_rows),
                        **{
                            f"mean_rho_{key}": float(
                                sum(r[f"rho_{key}"] for r in seed_rows) / len(seed_rows)
                            )
                            for key in ("e_sumo", "att_env")
                        },
                    }
                )
            entry: dict[str, Any] = {"subject": subject, "arm": spec.name, "role": spec.role}
            for key in ("e_sumo", "att_env"):
                per_draw = _seed_means_by_draw(arm_rows, f"rho_{key}")
                stats = mean_ci95([per_draw[d] for d in sorted(per_draw)])
                entry[key] = {
                    "n_draws": stats.n,
                    "mean": stats.mean,
                    "std": stats.std,
                    "ci95": stats.ci95,
                    "ci95_low": stats.mean - stats.ci95,
                    "ci95_high": stats.mean + stats.ci95,
                }
            entry["paired_att"] = {
                key: {
                    anchor: _paired_block(rows, arm_rows, subject, spec.name, anchor, key)
                    for anchor in ("fixedtime", "maxpressure")
                }
                for key in ("e_sumo", "att_env")
            }
            by_arm.append(entry)

    artifact: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "registered_in": "PREREGISTRATION H3, §3.4, A15, A16, A17, A18(c)",
        "scenario_key": SCENARIO_KEY,
        "stage": stage,
        "n_cells_declared": len(declared),
        "cell_set_source": "declared_cells()" if cells is None else "caller-supplied declaration",
        "halting_check_draw": HALTING_CHECK_DRAW,
        "cells": rows,
        "rho": {
            "formula": "rho = (ATT_fixedtime - ATT_arm) / (ATT_fixedtime - ATT_maxpressure)",
            "definitions": {
                "e_sumo": (
                    "A15's primary: the pool-clock ATT over the all-created population, from the "
                    "observer. P7.1's freeze writes the same quantity under the key "
                    "att_reference_created_population (Amendment C6); one quantity, two names"
                ),
                "att_env": "the admitted pair beside it: the env's own metric at the horizon",
            },
            "estimator": {
                "method": (
                    "analytic normal approximation, 1.96*s/sqrt(n) over the per-draw seed means "
                    "(offline.dt_gate.mean_ci95, the helper P5.3b's decomposition reached through "
                    "paired_comparison)"
                ),
                "resampling_seed": None,
                "why_no_seed": (
                    "there is no resampling: the interval is analytic and deterministic. BRIEF_37 "
                    "§3.5 asked for the helper's resampling seed; the repository has no bootstrap "
                    "estimator, and inventing one would be a NEW estimator, which the same sentence "
                    "forbids (plan §0.4, Amendment A3)"
                ),
                "unit": "one paired evaluation draw; seeds averaged within a draw, as in P4",
            },
            "not_clipped": (
                "PREREGISTRATION §3.4: values may exceed 1 or fall below 0 and are not clipped. "
                "fixed-time is 0 and MaxPressure is 1 by construction"
            ),
            "by_subject_arm_seed": by_seed,
            "by_subject_arm": by_arm,
        },
        "h3": _h3_block(by_arm),
        "contrast": _contrast_block(by_arm),
        "in_support": _in_support_block(rows, calibration),
        "canary": {
            "seconds": canary_record["seconds"],
            "threshold_seconds": CANARY_MAX_SECONDS,
            "verdict": "at speed" if canary_record["seconds"] <= CANARY_MAX_SECONDS else "throttled",
            "observed": dict(canary_record["facts"]),
            "source": (
                f"{CANARY_RECORD_NAME} in the work directory, written by the driver from the canary "
                "line right after the token; report re-ran check_canary on these facts before "
                "writing this file"
            ),
            "git_commit": canary_record.get("git_commit"),
            "git_dirty": canary_record.get("git_dirty"),
            "chunk_canaries_by_stage": _canaries_by_stage(chunks),
            "why_more_than_one_is_expected": (
                "Amendment B1 runs the campaign in two declared stages under two tokens and the "
                "driver re-runs the canary at each stage's start, so the chunks legitimately carry "
                "one canary per stage. P7.2b's single-canary rule was a one-stage rule"
            ),
        },
        "what_this_does_not_say": (
            "This is the ZERO-SHOT point only. No model was fine-tuned, no anchor corpus was "
            "collected and no k-shot curve is reported here; the few-shot points and the anchor are "
            "P7.3b's. rho is computed WITHIN SUMO against anchors on the same draws, so it is not a "
            "cross-backend comparison of absolute travel times. The calibrated-vs-naive contrast is "
            "exploratory (PREREGISTRATION §2, A17(d)) and is never promoted to a claim. The "
            "in-support diagnostic selects nothing. H3's third clause -- that the gap closes "
            "substantially by k = 100 -- is not tested here."
        ),
        "inputs": {
            "calibration_sha256": P7_2B_CALIBRATION_SHA256,
            "checkpoints": sorted(
                (identity_by_checkpoint[key] for key in identity_by_checkpoint),
                key=lambda entry: (entry["subject"], entry["seed"]),
            ),
            "deferred_56": (
                "output/p4_dt/ appears in no SHA256SUMS_* manifest, and every such manifest is "
                "gitignored in any case. Both subjects are therefore pinned against COMMITTED "
                "artifacts (Amendment G1): p4_gate.json for mappo1000 and p4_7_training.json's "
                "(tier, method, seed) row for mix50. sha256_checked_against on every cell names "
                "what was compared, so the two subjects' evidence is not implied to be equal. This "
                "is the first artifact in the project to pin the subjects' checkpoints by digest"
            ),
            "demand_by_draw": {
                str(draw): {k: v for k, v in demand.items() if k != "config_path"}
                for draw, demand in sorted(demand_by_draw.items())
            },
        },
        **_git_provenance(),
    }

    if stage1_path is not None:
        artifact["stage1_artifact"] = _stage1_block(Path(stage1_path), rows)

    # ---------------------------------------------------------------- 9. the LAST refusal
    serialised = json.dumps(artifact, indent=2, sort_keys=True)
    if FENCED_KEY in serialised:
        raise AssertionError(
            f"{FENCED_KEY} reached the artifact: a fenced quantity from P7.2b's smoke is on its way "
            "into docs/data/. The scan is over the serialised bytes because a key-shaped check "
            "cannot see a fenced name carried in a VALUE"
        )
    published_arms = {str(row["arm"]) for row in json.loads(serialised)["cells"]}
    if not published_arms <= DECLARED_ARM_NAMES:
        raise AssertionError(
            f"undeclared arm(s) {sorted(published_arms - DECLARED_ARM_NAMES)} reached the "
            "artifact; BRIEF_37 §2 lifted the fence for the declared arms and for nothing else"
        )
    _write_json(target_path, artifact)
    return artifact


def _seed_means_by_draw(rows: Sequence[Mapping[str, Any]], key: str) -> dict[int, float]:
    """The per-draw unit: the mean over training seeds, as in P4, so seed and draw stay crossed."""
    buckets: dict[int, list[float]] = {}
    for row in rows:
        buckets.setdefault(int(row["draw_id"]), []).append(float(row[key]))
    return {draw: sum(values) / len(values) for draw, values in buckets.items()}


def _paired_block(
    rows: Sequence[Mapping[str, Any]],
    arm_rows: Sequence[Mapping[str, Any]],
    subject: str,
    arm: str,
    anchor: str,
    key: str,
) -> dict[str, Any]:
    """G7/A3: the registered paired comparison on ATT, with the Wilcoxon beside the analytic CI.

    ``paired_comparison`` pairs per draw over shared draws and averages seeds within a draw
    (``dt_gate._per_draw_means``); A5 point 3 makes a comparison without shared draws void, which it
    raises on.  The arm names are qualified with the subject so each side carries exactly one.
    """
    from offline.dt_gate import EpisodeResult
    from offline.offline_baselines import paired_comparison

    def episodes(source: Sequence[Mapping[str, Any]], label: str) -> list[EpisodeResult]:
        return [
            EpisodeResult(
                arm=label,
                seed=None if row["seed"] is None else int(row["seed"]),
                draw_id=int(row["draw_id"]),
                att_horizon=float(row[key]),
                horizon_vehicle_count=float(row["horizon_vehicle_count"]),
                episode_reward=float(row["episode_reward"]),
            )
            for row in source
        ]

    anchor_rows = [r for r in rows if r["arm"] == anchor]
    comparison = paired_comparison(
        episodes(arm_rows, f"{subject}:{arm}"), episodes(anchor_rows, anchor)
    )
    return {
        "att_definition": key,
        "left_arm": comparison.left_arm,
        "right_arm": comparison.right_arm,
        "n_shared_draws": comparison.n_shared_draws,
        "mean_left": comparison.mean_left,
        "mean_right": comparison.mean_right,
        "mean_difference": comparison.mean_difference,
        "ci95_low": comparison.ci95_low,
        "ci95_high": comparison.ci95_high,
        "median_difference": comparison.median_difference,
        "wins": comparison.wins,
        "losses": comparison.losses,
        "ties": comparison.ties,
        "rank_biserial": comparison.rank_biserial,
        "wilcoxon_p": comparison.wilcoxon.p_value,
        "direction": "mean(arm - anchor) on ATT; NEGATIVE means the arm had the lower travel time",
    }


def _h3_block(by_arm: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """H3's first two clauses as the registered INEQUALITIES, reported and not interpreted.

    *"Zero-shot transfer is positive but incomplete -- better than fixed-time, worse than
    within-backend MaxPressure -- and closes substantially by k = 100."*  Clause 3 is P7.3b's and is
    named here as absent rather than left to be inferred.  There is no verdict sentence: the
    registered claim is the inequality, and a word like "confirmed" would be this task deciding
    something A17 gave it no licence to decide.
    """
    registered = [entry for entry in by_arm if entry["arm"] == "b_mean_k100"]
    clauses: list[dict[str, Any]] = []
    for inequality, clause, test in (
        ("rho_sumo(b_mean_k100) > 0", "better than the within-backend fixed-time anchor", lambda m: m > 0.0),
        ("rho_sumo(b_mean_k100) < 1", "worse than within-backend MaxPressure", lambda m: m < 1.0),
    ):
        for key in ("e_sumo", "att_env"):
            clauses.append(
                {
                    "inequality": inequality,
                    "clause": clause,
                    "definition": key,
                    "holds": {str(e["subject"]): bool(test(e[key]["mean"])) for e in registered},
                    "mean_rho": {str(e["subject"]): e[key]["mean"] for e in registered},
                    "ci95_low": {str(e["subject"]): e[key]["ci95_low"] for e in registered},
                    "ci95_high": {str(e["subject"]): e[key]["ci95_high"] for e in registered},
                }
            )
    return {
        "hypothesis": (
            "Zero-shot transfer is positive but incomplete -- better than fixed-time, worse than "
            "within-backend MaxPressure -- and closes substantially by k = 100"
        ),
        "registered_arm": "b_mean_k100",
        "primary_definition": "e_sumo",
        "test_row": (
            "MADT zero-shot in SUMO vs the within-backend fixed-time anchor, per paired scenario; "
            "unit: paired evaluation draw"
        ),
        "clauses": clauses,
        "third_clause": (
            "'closes substantially by k = 100' is P7.3b's and is NOT tested by this artifact"
        ),
        "reported_not_interpreted": (
            "the inequality and its value are reported; this artifact draws no conclusion from them"
        ),
    }


def _contrast_block(by_arm: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A17(d)'s calibrated-vs-naive contrast: EXPLORATORY, with its registered direction beside it."""
    by_subject: dict[str, Any] = {}
    for subject in SUBJECTS:
        calibrated = [e for e in by_arm if e["subject"] == subject and e["arm"] == "b_mean_k100"]
        naive = [e for e in by_arm if e["subject"] == subject and e["arm"] == "naive"]
        if not calibrated or not naive:
            continue
        by_subject[subject] = {
            key: {
                "calibrated_mean_rho": calibrated[0][key]["mean"],
                "naive_mean_rho": naive[0][key]["mean"],
                "difference": calibrated[0][key]["mean"] - naive[0][key]["mean"],
            }
            for key in ("e_sumo", "att_env")
        }
    return {
        "status": "exploratory",
        "registered_direction": (
            "A17(d): calibrated >= naive, and the gap larger on mix50. Registered as a DIRECTION "
            "for an exploratory contrast, never as a confirmatory test"
        ),
        "never_promoted": (
            "PREREGISTRATION §2 and A17(d): this contrast is exploratory and is not promoted to a "
            "claim whichever way it comes out"
        ),
        "by_subject": by_subject,
        "arms_present": sorted({str(entry["arm"]) for entry in by_arm}),
    }


def _in_support_block(
    rows: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    """The in-support diagnostic, declared and observed side by side.  Nothing selects on it.

    Amendment E4 recorded BEFORE any number existed that ``a_q1.0``'s target is out of support for
    ``mappo1000`` (below its training-return range, margin -10818) and inside it for ``mix50``.
    This block reports the artifact's own field so that fact is read from the registration rather
    than discovered afterwards and offered as an explanation of a result.
    """
    declared: dict[str, Any] = {}
    for subject in SUBJECTS:
        targets = targets_for_subject(subject, calibration)
        declared[subject] = {name: row["in_support"] for name, row in targets.items()}

    observed: dict[str, Any] = {}
    for row in rows:
        if row["kind"] != "dt" or row["n_decisions_in_support"] is None:
            continue
        key = f"{row['subject']}/{row['arm']}"
        bucket = observed.setdefault(key, {"n_cells": 0, "total_decisions_in_support": 0})
        bucket["n_cells"] += 1
        bucket["total_decisions_in_support"] += int(row["n_decisions_in_support"])

    return {
        "what_this_is": (
            "a reliability diagnostic, withdrawn as a selection criterion on 2026-08-13 "
            "(BRIEF_15 §12.1): target = 0 scores 0.000 here and was the best point measured. "
            "Nothing in this artifact selects on it"
        ),
        "declared_by_the_calibration_artifact": declared,
        "observed_decisions": observed,
        "amendment_e4": (
            "recorded before any number existed: a_q1.0's target is OUT OF SUPPORT for mappo1000 "
            "and in support for mix50, as the calibration artifact states"
        ),
    }


def _canaries_by_stage(chunks: Mapping[str, Mapping[str, Any]]) -> dict[str, list[float]]:
    """The canary each stage's chunks were rolled under -- one per stage, not one per campaign."""
    buckets: dict[str, set[float]] = {}
    for payload in chunks.values():
        seconds = payload.get("canary_seconds")
        if seconds is None:
            raise ValueError(
                f"{cell_chunk_name(payload)}: no canary_seconds. Every cell records the rate basis "
                "of the run that rolled it, or its timing cannot be compared with any other"
            )
        buckets.setdefault(str(payload.get("stage")), set()).add(float(seconds))
    return {stage: sorted(values) for stage, values in sorted(buckets.items())}


def _stage1_block(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Amendment B2: the stage-1 artifact's digest, and its rows compared cell for cell.

    Stage 2 is unconditional (B1), so the confirmatory rows must not move when the rest arrives.
    The comparison is on the rows themselves, under ``==``; the digest is recorded so a reader can
    redo it against the committed stage-1 file instead of trusting this sentence.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: the stage-1 artifact was named but is not here, so its rows cannot be "
            "compared with the final ones"
        )
    stage1 = json.loads(path.read_bytes())
    final_by_name = {cell_chunk_name(row): row for row in rows}
    moved: list[str] = []
    for row in stage1["cells"]:
        name = cell_chunk_name(row)
        if name not in final_by_name or final_by_name[name] != row:
            moved.append(name)
    if moved:
        raise ValueError(
            f"{len(moved)} stage-1 cell(s) differ between the stage-1 artifact and this one "
            f"(first: {moved[:3]}). Amendment B2: the stages are a sequence, not a cut, and a "
            "confirmatory row that moved once the rest of the campaign arrived would mean the "
            "number reported early is not the number reported finally"
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "n_cells": len(stage1["cells"]),
        "rows_identical": True,
    }


# ======================================================================================
# The pool: one process per worker, one env per cell, resume by content
# ======================================================================================

#: Amendment C3: 12 workers, measured 1.27x better than 8 on this machine's 16 cores.
DEFAULT_WORKERS = 12


def _worker(task: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """One cell in one process.  Top-level so ``spawn`` can pickle it.

    A failure is RETURNED, never raised into the pool: one broken cell must not take the campaign
    down, and the caller decides whether to stop.  Nothing is written unless the cell validated.
    """
    cell, kwargs = task
    try:
        payload = run_cell(cell, **kwargs["run"])
        path = write_chunk(payload, work_dir=kwargs["work_dir"])
        return {"name": path.name, "ok": True, "seconds": payload["seconds"], "error": None}
    except Exception as error:  # noqa: BLE001 - reported to the caller, not swallowed
        return {
            "name": cell_chunk_name(cell),
            "ok": False,
            "seconds": None,
            "error": f"{type(error).__name__}: {error}",
        }


def run_stage(
    *,
    work_dir: str | Path,
    out_root: str | Path,
    output_root: str | Path,
    data_dir: str | Path | None = None,
    stage: str | None = None,
    cells: Sequence[Mapping[str, Any]] | None = None,
    canary_seconds: float | None = None,
    workers: int = DEFAULT_WORKERS,
    limit: int | None = None,
) -> dict[str, Any]:
    """Every cell of *stage* that is not already on disk as a reusable chunk.

    **The skip decision is in Python, not in the shell.**  ``offline/campaigns/p5_3b.sh`` skipped on
    ``[ -f ]`` alone, and a bad chunk then survived every restart until someone deleted it by hand;
    :func:`chunk_is_reusable` re-derives the verdict from the chunk's own content and from the files
    on disk instead.  A chunk that exists and is NOT reusable is moved aside to ``failed/`` -- never
    overwritten, because it is evidence about a run.

    ``spawn`` rather than ``fork``: a forked worker inherits the parent's traci module state, and
    one env per cell means the simulator is created and closed inside the worker that uses it.
    """
    from multiprocessing import get_context

    work = Path(work_dir)
    declared = list(declared_cells(stage) if cells is None else cells)
    if limit is not None:
        declared = declared[: int(limit)]

    identity = {"out_root": str(out_root), "output_root": str(output_root), "data_dir": None if data_dir is None else str(data_dir)}
    todo: list[dict[str, Any]] = []
    reused: list[str] = []
    for cell in declared:
        path = chunk_path(cell, work_dir=work)
        if reusable_chunk_at(path, cell=cell, **identity):
            reused.append(path.name)
            continue
        if path.exists():
            move_aside(path)
        todo.append(dict(cell))

    kwargs = {
        "run": {**identity, "canary_seconds": canary_seconds},
        "work_dir": str(work),
    }
    results: list[dict[str, Any]] = []
    if todo:
        work.mkdir(parents=True, exist_ok=True)
        context = get_context("spawn")
        with context.Pool(processes=max(1, int(workers))) as pool:
            for result in pool.imap_unordered(_worker, [(cell, kwargs) for cell in todo]):
                results.append(result)
                status = "ok" if result["ok"] else f"FAILED {result['error']}"
                print(f"  {result['name']} {status}", flush=True)

    failures = [result for result in results if not result["ok"]]
    return {
        "stage": stage,
        "n_declared": len(declared),
        "n_reused": len(reused),
        "n_rolled": len(results) - len(failures),
        "n_failed": len(failures),
        "failures": failures,
    }


def build_parser() -> Any:
    """CLI: ``cells``, ``report``, ``canary``, ``record-canary``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m offline.transfer_curve",
        description=(
            "P7.3a: the zero-shot point of the C3 transfer curve. Every root is a parameter with "
            "today's path as its default (BRIEF_37 section 0.9)."
        ),
    )
    parser.add_argument("--draws-root", default=str(DEFAULT_DRAWS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--canary-seconds", type=float, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cells = subparsers.add_parser("cells", help="roll every cell of a stage that is not on disk")
    cells.add_argument("--stage", choices=list(STAGES), default=None)
    cells.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    cells.add_argument("--limit", type=int, default=None)

    gate = subparsers.add_parser(
        "a17f", help="A17(f): every logged episode reproduces P7.2b's probe return bit-for-bit"
    )
    gate.add_argument("--corpus-dir", required=True)

    report_parser = subparsers.add_parser("report", help="write the committed artifact")
    report_parser.add_argument("--stage", choices=list(STAGES), default=None)
    report_parser.add_argument("--stage1-path", default=None)

    subparsers.add_parser("canary", help="the machine-health canary (PROJECT_PLAN section 7)")
    record = subparsers.add_parser("record-canary", help="park the canary line in the work directory")
    record.add_argument("--line", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns a non-zero code on a refusal rather than raising into the driver."""
    from offline.transfer_calibration import (
        canary_seconds as measure_canary,
        check_canary,
        format_canary_line,
        record_canary,
    )

    args = build_parser().parse_args(argv)
    work = Path(args.work_dir)

    if args.command == "canary":
        seconds, facts = measure_canary()
        check_canary(facts)
        print(format_canary_line(seconds, facts), flush=True)
        return 0

    if args.command == "record-canary":
        print(record_canary(args.line, work_dir=work), flush=True)
        return 0

    if args.command == "a17f":
        # The gate lives in transfer_calibration, where P7.2b's probe does; this is the CLI the
        # P7.3a driver runs it from. It RAISES on a mismatch -- 100/100 or P7.3 stops -- so the
        # driver's `|| fail` sees a non-zero exit and the campaign never reaches an evaluation cell.
        from offline.transfer_calibration import assert_logged_corpus_matches_probe

        record = assert_logged_corpus_matches_probe(
            args.corpus_dir, Path(args.data_dir) / P7_2B_CALIBRATION_NAME
        )
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        return 0

    if args.command == "cells":
        summary = run_stage(
            work_dir=work,
            out_root=args.draws_root,
            output_root=args.output_root,
            data_dir=args.data_dir,
            stage=args.stage,
            canary_seconds=args.canary_seconds,
            workers=args.workers,
            limit=args.limit,
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 1 if summary["n_failed"] else 0

    name = (
        "p7_3a_zero_shot_stage1.json"
        if args.stage == STAGE_CONFIRMATORY
        else "p7_3a_zero_shot.json"
    )
    artifact = report(
        work_dir=work,
        out_path=Path(args.out_dir) / name,
        output_root=args.output_root,
        out_root=args.draws_root,
        data_dir=args.data_dir,
        stage=args.stage,
        stage1_path=args.stage1_path,
    )
    print(f"wrote {Path(args.out_dir) / name} with {len(artifact['cells'])} cells", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
