"""P5.3c (``BRIEF_42``; ``PREREGISTRATION`` A26 as corrected by A26.1): H4's context-length sweep.

Formats written by this module, each carrying its version in the payload:

* ``p5.3c-reference-rows/1.0`` -- ``docs/data/p4_k20_att_engine_rows.json`` (C1): the per-draw ``att_engine`` and
  ``att_ours`` of P4's five K = 20 checkpoints on the 100 held-out draws, extracted from P8.4b's gitignored cells with
  every source's digest.  A26(c)'s PRECONDITION: committed before any training, and the campaign's reference gate
  compares against it under ``==``.

Alignment convention.  Contract C6 v1.1 (``docs/CONTRACTS.md``), unchanged: observation rows ``T + 1``, decision and
outcome rows ``T``, and the reward of decision ``t`` is the one the env returned from step ``t``.  A training window is
the K decision steps ENDING at its step ``t``, left-padded (``offline/dataset.py:777-820``); nothing about a window is
redefined here.  The per-episode ATT definitions are A11(b)'s: ``att_engine`` (the engine's pool-clock average over
every vehicle the demand created -- PRIMARY on hz1x1 under A11's Rule R and A15) and ``att_ours`` (P4's
``att_horizon``, co-reported).

C1 -- THE REFERENCE ROWS
------------------------
``extract-reference-rows`` reads, and before writing anything verifies:

1. P4's five checkpoints ``<output>/p4_dt/dt_seed<s>.pt``, each by file sha256 against ``docs/data/p4_gate.json``;
2. P8.4b's campaign manifest ``<output>/p8_4b_rederivation/campaign_manifest.json``, which must declare all 500
   ``hz1x1|dt@mappo1000|<seed>|<draw>`` cells, and whose ``engine_seed`` is recorded;
3. the cell files named by the producer's own rule (``offline.att_rederivation.cell_file_name``): all 500 present, no
   other file of that pattern in the directory, no (seed, draw) claimed by two files, each file's CONTENT naming the
   pair its NAME says, ``arm dt@mappo1000``, ``tier mappo1000``, ``method dt``, ``scenario hz1x1``, P8.4b's format,
   ``reproduces_committed true``, a policy source that is P4's checkpoint of that seed, and both definitions finite.

Then ONE write: the file is written once -- the same content again is a no-op, other content is refused -- through
``offline.few_shot``'s exclusive link.  The command refuses a dirty code tree, because the committed file records the
commit that wrote it (plan F8; ``BRIEF_42`` Amendment A, Q8).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from offline.att_rederivation import CellKey, cell_file_name
from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS

__all__ = [
    "ATT_DEFINITIONS",
    "PRIMARY_ATT",
    "REFERENCE_ROWS_FORMAT_VERSION",
    "REFERENCE_ROWS_NAME",
    "main",
    "reference_rows_payload",
    "write_reference_rows",
]

# ======================================================================================================================
# C1 -- the reference rows
# ======================================================================================================================

REFERENCE_ROWS_FORMAT_VERSION = "p5.3c-reference-rows/1.0"
REFERENCE_ROWS_NAME = "p4_k20_att_engine_rows.json"

#: A11(b)'s two definitions, the primary first (A11 Rule R; A15).
ATT_DEFINITIONS: tuple[str, ...] = ("att_engine", "att_ours")
PRIMARY_ATT = "att_engine"

#: The subject of the reference rows: P4's K = 20 model on the P4 validation scenario (A26(a)).
REFERENCE_SUBJECT: dict[str, Any] = {
    "tier": "mappo1000",
    "arm": "dt@mappo1000",
    "method": "dt",
    "scenario": "hz1x1",
    "scenario_id": "cityflow1x1",
    "context_length": 20,
}

#: Where P8.4b left its cells, and what it wrote into each (``offline/att_rederivation.py:1288-1298``).
REDERIVATION_DIRNAME = "p8_4b_rederivation"
REDERIVATION_MANIFEST_NAME = "campaign_manifest.json"
REDERIVATION_FORMAT_VERSION = "p8.4b-rederivation/1.0"
REDERIVATION_ARTIFACT_NAME = "p8_4b_rederivation.json"
P4_GATE_NAME = "p4_gate.json"
P4_CHECKPOINT_DIRNAME = "p4_dt"

_REQUIRED_CELL_KEYS: tuple[str, ...] = (
    "arm", "tier", "method", "scenario", "seed", "draw_id", "format_version", "reproduces_committed",
    "policy_source", *ATT_DEFINITIONS,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise ValueError(f"{path} is absent: {what}")
    return json.loads(path.read_bytes())


def _git_in_module_tree(*args: str) -> subprocess.CompletedProcess[str]:
    """``git`` run in THIS module's directory, never the process CWD (a driver's CWD may be another tree)."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(args=["git", *args], returncode=127, stdout="", stderr=str(exc))


def _code_provenance() -> dict[str, Any]:
    """The module tree's commit and whether it is dirty -- STRICT (``BRIEF_37`` J1(a)): a git failure raises.

    "Could not say" must never read as "clean", so neither call's failure is folded into a default.
    """
    head = _git_in_module_tree("rev-parse", "HEAD")
    commit = head.stdout.strip()
    if head.returncode != 0 or len(commit) != 40:
        raise ValueError(
            f"git could not name the code's commit (exit {head.returncode}: {head.stderr.strip()[:160]!r}); a record "
            "that cannot say which code wrote it is not provenance"
        )
    status = _git_in_module_tree("status", "--porcelain")
    if status.returncode != 0:
        raise ValueError(
            f"git could not say whether the code tree is clean (exit {status.returncode}: "
            f"{status.stderr.strip()[:160]!r}); an unmeasured tree is never recorded as a clean one"
        )
    return {"code_commit": commit, "code_dirty": bool(status.stdout.strip())}


def _declared_cells() -> dict[tuple[int, int], str]:
    """The 500 (seed, draw) pairs and the file each is written to, by the PRODUCER's naming rule."""
    return {
        (int(seed), int(draw)): cell_file_name(
            CellKey(
                scenario=str(REFERENCE_SUBJECT["scenario"]),
                arm=str(REFERENCE_SUBJECT["arm"]),
                seed=int(seed),
                draw_id=int(draw),
            )
        )
        for seed in TRAINING_SEEDS
        for draw in HELD_OUT_DRAWS
    }


def _manifest_key(seed: int, draw: int) -> str:
    """``offline.att_rederivation.campaign_manifest``'s cell key, ``scenario|arm|seed|draw``."""
    return f"{REFERENCE_SUBJECT['scenario']}|{REFERENCE_SUBJECT['arm']}|{int(seed)}|{int(draw)}"


def _verified_checkpoints(output_root: Path, data_dir: Path) -> dict[str, dict[str, str]]:
    gate = _read_json(data_dir / P4_GATE_NAME, "P4's gate artifact names the five checkpoints' digests")
    recorded = gate.get("checkpoints") or {}
    checkpoints: dict[str, dict[str, str]] = {}
    for seed in TRAINING_SEEDS:
        entry = recorded.get(str(seed))
        if not isinstance(entry, dict) or not entry.get("sha256"):
            raise ValueError(f"{data_dir / P4_GATE_NAME} records no sha256 for seed {seed}")
        path = output_root / P4_CHECKPOINT_DIRNAME / f"dt_seed{seed}.pt"
        if not path.is_file():
            raise ValueError(f"{path} is absent: P4's checkpoint of seed {seed} is the reference rows' subject")
        digest = _sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(
                f"{path}: sha256 {digest} is not {P4_GATE_NAME}'s {entry['sha256']}; the rows would describe another "
                "checkpoint than the one P4 published"
            )
        checkpoints[str(seed)] = {
            "path": f"output/{P4_CHECKPOINT_DIRNAME}/dt_seed{seed}.pt",
            "sha256": digest,
            "checked_against": f"docs/data/{P4_GATE_NAME} checkpoints.{seed}.sha256",
        }
    return checkpoints


def reference_rows_payload(*, output_root: str | Path, data_dir: str | Path) -> dict[str, Any]:
    """The reference rows, every source verified first.  Writes NOTHING; refuses with ``ValueError``."""
    out = Path(output_root)
    data = Path(data_dir)
    checkpoints = _verified_checkpoints(out, data)

    cells_dir = out / REDERIVATION_DIRNAME
    manifest_path = cells_dir / REDERIVATION_MANIFEST_NAME
    manifest = _read_json(manifest_path, "P8.4b's campaign manifest declares the cells it rolled")
    declared_by_campaign = set(manifest.get("cells") or [])
    declared = _declared_cells()

    names = set(declared.values())
    pattern = f"cell_{REFERENCE_SUBJECT['scenario']}_dt_at_{REFERENCE_SUBJECT['tier']}_seed*_draw*.json"
    extra = sorted(path.name for path in cells_dir.glob(pattern) if path.name not in names)
    if extra:
        raise ValueError(
            f"{cells_dir / extra[0]} is not one of the 500 declared cells ({len(extra)} such file(s)); a file of this "
            "pattern outside the declared set means the directory is not the one P8.4b wrote"
        )
    unlisted = [pair for pair in sorted(declared) if _manifest_key(*pair) not in declared_by_campaign]
    if unlisted:
        raise ValueError(
            f"(seed, draw) {unlisted[0]} is not declared in P8.4b's campaign manifest {manifest_path} "
            f"({len(unlisted)} such pair(s)); the rows may only come from cells that campaign declared"
        )

    absent = [declared[pair] for pair in sorted(declared) if not (cells_dir / declared[pair]).is_file()]
    if absent:
        raise ValueError(
            f"{len(absent)} of the 500 declared cells are absent from {cells_dir} (first: {absent[:3]}); the rows "
            "are extracted whole or not at all"
        )

    contents: dict[tuple[int, int], tuple[dict[str, Any], str]] = {}
    for pair in sorted(declared):
        raw = (cells_dir / declared[pair]).read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"{declared[pair]} is a {type(payload).__name__}, not a cell record")
        for key in _REQUIRED_CELL_KEYS:
            if key not in payload:
                raise ValueError(f"{declared[pair]} lacks {key!r}")
        contents[pair] = (payload, _sha256_bytes(raw))

    claims = Counter((int(payload["seed"]), int(payload["draw_id"])) for payload, _digest in contents.values())
    duplicated = sorted(pair for pair, count in claims.items() if count > 1)
    if duplicated:
        pair = duplicated[0]
        holders = sorted(
            declared[name_pair]
            for name_pair, (payload, _digest) in contents.items()
            if (int(payload["seed"]), int(payload["draw_id"])) == pair
        )
        raise ValueError(
            f"(seed, draw) {pair} is claimed by {claims[pair]} files ({holders}); each pair must come from exactly "
            "one cell"
        )

    expected_fields = {
        "arm": REFERENCE_SUBJECT["arm"],
        "tier": REFERENCE_SUBJECT["tier"],
        "method": REFERENCE_SUBJECT["method"],
        "scenario": REFERENCE_SUBJECT["scenario"],
        "format_version": REDERIVATION_FORMAT_VERSION,
    }
    policies: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for pair in sorted(declared):
        payload, digest = contents[pair]
        name = declared[pair]
        seed, draw = pair
        claimed = (int(payload["seed"]), int(payload["draw_id"]))
        if claimed != pair:
            raise ValueError(f"{name} names {claimed} but its file name says {pair}")
        for field, expected in expected_fields.items():
            if payload[field] != expected:
                raise ValueError(f"{name}: {field} {payload[field]!r}, not {expected!r}")
        if payload["reproduces_committed"] is not True:
            raise ValueError(
                f"{name}: reproduces_committed is {payload['reproduces_committed']!r}; P8.4b re-derived this cell "
                "without reproducing P4's committed att_ours"
            )
        policy = payload["policy_source"]
        checkpoint = str(policy.get("checkpoint", "")) if isinstance(policy, dict) else ""
        if not checkpoint.endswith(f"/{P4_CHECKPOINT_DIRNAME}/dt_seed{seed}.pt"):
            raise ValueError(f"{name}: its policy source {checkpoint!r} is not P4's checkpoint of seed {seed}")
        policies.add((str(policy.get("kind")), str(policy.get("detail"))))
        values: dict[str, float] = {}
        for definition in ATT_DEFINITIONS:
            value = payload[definition]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name}: {definition} {value!r} is not a finite number")
            values[definition] = float(value)
        rows.append(
            {
                "seed": seed,
                "draw_id": draw,
                "att_engine": values["att_engine"],
                "att_ours": values["att_ours"],
                "source": name,
                "source_sha256": digest,
            }
        )
    if len(policies) != 1:
        raise ValueError(f"the 500 cells record {len(policies)} different policy sources: {sorted(policies)[:3]}")
    kind, detail = next(iter(policies))

    artifact_path = data / REDERIVATION_ARTIFACT_NAME
    artifact = _read_json(artifact_path, "P8.4b's committed artifact records the campaign's provenance")
    campaign = ((artifact.get("provenance") or {}).get("code_provenance")) or {}

    return {
        "format_version": REFERENCE_ROWS_FORMAT_VERSION,
        "role": (
            "A26(c)'s precondition: the per-draw att_engine (and att_ours) of P4's five K = 20 checkpoints on the 100 "
            "held-out draws, the reference the campaign's gate compares its re-evaluation of the same checkpoints "
            "against under ==. Committed before any training (BRIEF_42 C1)."
        ),
        "registered_in": "PREREGISTRATION A26(c) as corrected by A26.1; BRIEF_42 C1 and Amendment A (Q8)",
        "definitions": list(ATT_DEFINITIONS),
        "primary_definition": PRIMARY_ATT,
        "definition_note": (
            "att_engine is the engine's pool-clock average over every vehicle the demand created (A11 Rule R, A15: "
            "primary on hz1x1); att_ours is P4's att_horizon, co-reported"
        ),
        "subject": dict(REFERENCE_SUBJECT),
        "seeds": [int(seed) for seed in TRAINING_SEEDS],
        "draw_ids": [int(draw) for draw in HELD_OUT_DRAWS],
        "n_rows": len(rows),
        "engine_seed": int(manifest["engine_seed"]),
        "checkpoints": checkpoints,
        "source": {
            "directory": f"output/{REDERIVATION_DIRNAME}",
            "file_name_rule": "offline.att_rederivation.cell_file_name",
            "producer": "python -m offline.att_rederivation run (P8.4b)",
            "campaign_manifest": f"output/{REDERIVATION_DIRNAME}/{REDERIVATION_MANIFEST_NAME}",
            "campaign_manifest_sha256": _sha256_file(manifest_path),
            "declared_cells_sha256": manifest.get("declared_cells_sha256"),
            "campaign_provenance": {
                "artifact": f"docs/data/{REDERIVATION_ARTIFACT_NAME}",
                "artifact_sha256": _sha256_file(artifact_path),
                "git_commit": campaign.get("git_commit"),
                "code_dirty": campaign.get("code_dirty"),
                "note": (
                    "the cells themselves record no commit; this is the commit P8.4b's committed artifact records for "
                    "the code that wrote its report"
                ),
            },
            "policy_source": {"kind": kind, "detail": detail},
            "evaluation_device": {
                "recorded_by_the_cells": False,
                "note": (
                    "P8.4b's runner left --device unset (offline/att_rederivation.py:725), which the platform resolves "
                    "to CUDA when available (agent/utils/utils.py:32-35); A26.1(b) registers the sweep's evaluation "
                    "on CUDA for that reason"
                ),
            },
        },
        "extraction": {
            "command": "python -P -m offline.context_sweep extract-reference-rows",
            "module": "offline/context_sweep.py",
            **_code_provenance(),
        },
        "rows": rows,
    }


def write_reference_rows(payload: dict[str, Any], destination: str | Path) -> str:
    """Write *payload* ONCE; return the file's sha256.  The same content again is a no-op, other content refused."""
    from offline.few_shot import _link_exclusive

    target = Path(destination)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if target.exists() or target.is_symlink():
        if target.is_file() and target.read_bytes() == data:
            return _sha256_bytes(data)
        raise ValueError(f"{target} already exists and differs; it is written once and never replaced")
    if not target.parent.is_dir():
        raise ValueError(f"{target.parent} does not exist; nothing is created here")
    return _link_exclusive(data, target)


# ======================================================================================================================
# The command line
# ======================================================================================================================


def _cmd_extract_reference_rows(args: argparse.Namespace) -> int:
    code = _code_provenance()
    if code["code_dirty"] and not args.allow_dirty:
        raise ValueError(
            f"the code tree at {code['code_commit'][:12]} is dirty; the committed rows record the commit that wrote "
            "them (plan F8), so extract from a clean, committed tree"
        )
    payload = reference_rows_payload(output_root=Path(args.output_root), data_dir=Path(args.data_dir))
    digest = write_reference_rows(payload, Path(args.out))
    print(
        f"extract-reference-rows: {payload['n_rows']} rows from {payload['source']['directory']} -> {args.out} "
        f"(sha256 {digest}); checkpoints 5/5 at p4_gate.json's digests; code {payload['extraction']['code_commit'][:12]}",
        flush=True,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m offline.context_sweep", description=__doc__.splitlines()[0], allow_abbrev=False
    )
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser(
        "extract-reference-rows", help="C1: P4's K = 20 rows from P8.4b's cells, written once", allow_abbrev=False
    )
    extract.add_argument("--output-root", required=True, help="the output tree holding p4_dt/ and p8_4b_rederivation/")
    extract.add_argument("--data-dir", required=True, help="the docs/data directory holding p4_gate.json")
    extract.add_argument("--out", required=True, help="the file to write: docs/data/p4_k20_att_engine_rows.json")
    extract.add_argument(
        "--allow-dirty", action="store_true", help="accept a dirty code tree; the payload then records code_dirty true"
    )
    extract.set_defaults(handler=_cmd_extract_reference_rows)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; a refusal prints its reason and returns 2.  Anything else propagates, loudly."""
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f"context_sweep {args.command}: REFUSED: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
