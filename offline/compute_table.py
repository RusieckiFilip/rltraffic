"""P8.2: the compute-and-latency table -- parameters, training cost and decision latency, every number with its source.

Written against ``docs/briefs/BRIEF_43_p8.2_compute_latency.md`` §5, its **Amendment A** (Q3: one row per arm with
its tiers inside and an ``architecture`` key; Q4: MAPPO's ``results.json`` digests pinned here as of 2026-10-03;
Q5/Q6: absences declared, never reconstructed; Q7: ``trained`` / ``deployed`` / ``stored`` /
``executed_per_decision``; Q13: parameters counted on all five seeds) and its **Amendment B** (gate G1: B4 the
latency verification, B6 the text the table prints, B7), on the plan ``docs/plans/p8.2.md`` @ ``a7c43e1``.

On-disk format
--------------
``p8.2-compute/1.0`` -- ``output/p8_2/artifacts/p8_2_compute.json``, committed by hand as ``docs/data/p8_2_compute.json``.
Its ``rows`` carry, per row: the claims it serves with the committed result each sits in; every checkpoint (tier,
seed, path, digest and the record that names it); the parameter counts (route A, the method's own loader, equal to
route B, the payload's parameter tensors, on every checkpoint); the training cost per (record, tier) -- median
[min-max] over seeds, never pooled across records, each with its regime and with what its wall time covers; the
environment interactions; the inference latency from the ``p8.2-latency/1.1`` records. Every number read from a record,
a checkpoint, the code or a measurement is ``{value, source}``, a source being ``{file, sha256, json_path}``, ``{file,
sha256, line}``, ``{code, file, sha256}``, ``{measurement}`` or ``{inferred, confidence, recorded}``; a declared
absence is ``{value: null, reason}``. The derived statistics -- a median with its minimum and maximum, a count of timed
decisions, the per-intersection figure, a product of two sourced factors -- carry no source of their own: each sits
beside the sourced per-seed values (or factors) it is derived from, so a reader recomputes it there. Files are named
relative to this checkout (``docs/...``) or to the main tree (``output/...``, ``datasets_v11/...``). Alignment
convention: not applicable -- the artifact records no trajectory.

What is pinned
--------------
:data:`PINNED_RECORDS` holds, each at its digest of 2026-10-03: every record of plan §4 the builder reads; the five
:data:`SOURCES_ADDED_AFTER_PLAN`, each with its reason; and the 25 corpus manifests plan §3 names as the MAPPO
checkpoints' digest source. The three plan §4 records the builder does not read are :data:`UNPINNED_PLAN_RECORDS`, each
with why no number depends on it.

The row-spec vocabulary
-----------------------
A training entry is a mapping: ``record`` (a :data:`PINNED_RECORDS` key), ``tier``, ``label``, ``seeds`` (how each
seed's object is found: ``{"kind": "list", "path": <JSON path of a list>, "match": {field: value}}`` or ``{"kind":
"dict", "path": <JSON path of a mapping>, "key": "<template with {seed}>"}``), ``covers`` (what its wall time
measures) and ValueRefs for ``seconds``, ``steps``, ``batch``, ``data`` and the ``regime`` fields (``device``,
``gpu``, ``torch``, ``threads``, ``concurrency``). A ValueRef is ``{"kind": "seed", "field": <dotted field under the
seed's object>}`` (one value per seed), ``{"kind": "record", "path": <JSON path>}``, ``{"kind": "checkpoint",
"path": <JSON path into the checkpoint payload>}`` (read from every seed's checkpoint, equal across seeds), ``{"kind":
"log", ...}`` (a manifest-pinned log's lines) or ``{"kind": "absent", "key": <a DECLARED_ABSENCES key>}``; ``data``
adds ``unit``. Further locators: ``value_dict`` (the seed's value IS the number), ``log``, ``mappo`` (a
``results.json`` cell) and ``checkpoints`` (the record is lost and every value comes from the checkpoints); further
ValueRefs: ``pinned`` (another pinned record), ``code`` and ``measurement``.

Refusals, all before the artifact is written: a record that is absent or at a digest other than the pinned one; a
row with no source for a column; a value missing from its record that is not a declared absence, or a null that no
declared absence names; a row with a trained model and no training entry; an empty outlier note; a latency run
without ``COMPLETE``, with a throttled or non-reproducing canary, with a file outside its manifest, that does not
cover every (row, device) cell of the registry, or with a record that fails any refusal of :func:`verify_latency_run`;
a latency registry row that does not time its table row's seed-101 representative; route A and route B disagreeing; a
row whose checkpoints differ in size.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import statistics
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "FORMAT_VERSION",
    "MAPPO_RESULTS_PINNED_ON",
    "PinnedRecord",
    "PINNED_RECORDS",
    "SOURCES_ADDED_AFTER_PLAN",
    "UNPINNED_PLAN_RECORDS",
    "Roots",
    "TableRow",
    "TABLE_ROWS",
    "DECLARED_ABSENCES",
    "sha256_file",
    "json_get",
    "read_pinned",
    "parameter_names",
    "count_state_parameters",
    "count_payload_parameters",
    "count_loaded_parameters",
    "row_checkpoints",
    "training_block",
    "training_blocks",
    "latency_figures",
    "verify_latency_run",
    "check_registry_coverage",
    "join_registry_row",
    "inference_block",
    "build_artifact",
    "write_artifact",
    "build_parser",
    "main",
]

FORMAT_VERSION = "p8.2-compute/1.0"

#: Q4: the date the MAPPO ``results.json`` digests were pinned (no committed artifact or manifest named them before).
MAPPO_RESULTS_PINNED_ON = "2026-10-03"

_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)
_MODULE_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PinnedRecord:
    """A record the builder reads: its root (``repo`` / ``output`` / ``corpus``), path, pinned digest and who names it."""

    key: str
    root: str
    relpath: str
    sha256: str
    named_by: str


_PLAN = "docs/plans/p8.2.md §4"
_SUMS_P51 = "output/SHA256SUMS_p5_1.txt"
_SUMS_P52 = "output/SHA256SUMS_p5_2.txt"


def _pins() -> dict[str, PinnedRecord]:
    repo = {
        "p4_training": ("p4_training.json", "ccb99a572aa1a80c788e508240f019586b34a7e397aac0047f2044de62f9ac49"),
        "p4_gate": ("p4_gate.json", "f8aee52f393e68a988cb8d4815ead3487cb3458044585b7f1115a54320b40f55"),
        "p4_4_training": ("p4_4_training.json", "9f2c792d6ec2225fee0949a4513217ce8a2f99cea29255b5f115d67b7e60d24d"),
        "p4_5_selection": ("p4_5_selection.json", "651ca7f51619227c05b5685e5309dc84974374f938fbceba44fc849d1147a63b"),
        "p4_6_training": ("p4_6_training.json", "0ce810361534ff59a5b5c32898b9395a304a4261d965c978f71093ba67f24b1d"),
        "p4_6_declaration": ("p4_6_declaration.json", "5197ab35de7fe9e7428892de19bbe86661aef9a98846cc9f772b2681eff50aa2"),
        "p4_7_training": ("p4_7_training.json", "5abcee0de282cbb3f287f5bfe4f98d349c1f4fa5f33b969553bfe8e0a0ccd428"),
        "p4_7_declaration": ("p4_7_declaration.json", "fc7931cd8008a093c5de7353006fd1abf0cad8514e50aa96a9b5c4106a97b405"),
        "p5_3b_nortg": ("p5_3b_nortg.json", "7ecf4b7050f9c900765a104fc94bf74df48393f4d1c2d923d1a31577ee890d8d"),
        "p5_3c_train": ("p5_3c_train.json", "017808a5e84fada469d6b2d302889ad171b8e429b1612b8ac06c900ef3c6321a"),
        "p7_3b_anchor_training": ("p7_3b_anchor_training.json",
                                  "edca65f96f060d77a8ef9620deaa2c4a76c4c77eae0f7a9e05c47587a06dd62a"),
        "p7_3c_finetune": ("p7_3c_finetune.json", "adb59377edc23270ad479a542ed7d120f4b57c784f6e1f109c54624231ae79bf"),
        "p7_3d_grid4x4": ("p7_3d_grid4x4.json", "c63c371ff14d208d16b9fbfa6d3daa31975679c90b20b5a4b44fbed60760b0b7"),
        "p7_3d_calibration": ("p7_3d_calibration.json",
                              "3e9df8eed4af2e42c132087e711751bc4c265edcef75dd88e24e6143e82f9723"),
        "p8_4b_rederivation": ("p8_4b_rederivation.json",
                               "d92321d422acc06a35bde7a499cef41232846ced33ceea594c4ef199792c96d9"),
        "att_ladder_v11": ("att_ladder_v11.json", "b3bb9041dc44efda78423d33b6e7545da04519de4516b27e5a095bda8a8ab984"),
        "p4_heldout_thresholds": ("p4_heldout_thresholds.json",
                                  "fd81b1c65230724de4fe1449c8f1b823e1ed7661cfb8821a12333aef517f102e"),
        "p4_4_baselines": ("p4_4_baselines.json", "af8ad2bdb5263d60232d3bfa3c802ce74934a17fdda99dd00ffa88996f42a010"),
        "p4_5_baselines": ("p4_5_baselines.json", "b4c2a2b206a6986168caaaef85eb5293266b9a1825e27dcdf0dcc8e7363e02ab"),
        "p4_7_grid": ("p4_7_grid.json", "01c5cf7c2dfdbabf13d364fe1f41ef41876149de7a47b27ececc6b84ea2fcef1"),
        "p5_1_grid": ("p5_1_grid.json", "ccb9a315960224e597aeee6cf7910bff9cd1d1fd9dacfe98ef8185ead4e883b2"),
        "p5_3c_context_sweep": ("p5_3c_context_sweep.json",
                                "bcdca7eabad82d7a99feed7eed9f2ed9bdc37f3950bbf55811dcf94b76c7abed"),
        "p7_3a_zero_shot": ("p7_3a_zero_shot.json", "e7a0d290a3179ffea8185496b593ee794d438c28f200e806f3c35566c389fb70"),
        "p7_3b_anchor": ("p7_3b_anchor.json", "4cae233ce480a5596dc9a65ad53c1ec0cb20c17fa80008d1e1a13341703af1cc"),
        "p7_3c_grid4x4": ("p7_3c_grid4x4.json", "1bcea367d86fa1943d45c747634b7ed2e49cb3c174f44a532c7e8ba13ca66e1d"),
        "p5_1_declaration": ("p5_1_declaration.json", "2e361bb61d8c28c9a89382a562d2ee5b896f36cdeae356379355f9219213354b"),
        "p5_2_declaration_maxpressure": ("p5_2_declaration_maxpressure.json",
                                         "f51ff6c8a61afa494bfd184b2cc272db9ed0a75039a28f122d47e91a5df4529f"),
        "p5_2_declaration_fixedtime": ("p5_2_declaration_fixedtime.json",
                                       "e2c716d1f256212d8d55cd806f3306fe471d798f9b6973202f84a693ef1f82f2"),
        "p5_2_declaration_random": ("p5_2_declaration_random.json",
                                    "c8b8a35a2dd034434aef4b0c32de73627de7f35a8f7eb55b0bbf752631eadfe4"),
        "p5_2_declaration_mappo1000": ("p5_2_declaration_mappo1000.json",
                                       "5bf5624cb49b9b7f051f709253469d1d8cacbd1ed066565e1d475c6ad569e4b8"),
    }
    output = {
        "mappo_results_1000": ("experiments/p2_1_mappo_nominal_1000/results.json",
                               "f7fc5e639d14dce53422b5874d68a7348f1a88dee9a5b0b9d6907439330ff4fd", "pinned here (Q4)"),
        "mappo_results_500": ("experiments/p2_1_mappo_nominal_500/results.json",
                              "fb94df5cf079b575983af87557d9eb9d020609d6dd43894dd100fdb7e79761ae", "pinned here (Q4)"),
        "mappo_results_060": ("experiments/p2_1_mappo_nominal_060/results.json",
                              "d6e1868842e43c01e5ed4ee1dc36be00fa99ec6803bc675a9190b75b06d49fad", "pinned here (Q4)"),
        "p5_1_training_dt_spatial": ("p5_1/training_dt_spatial.json",
                                     "6353167f326c6841448bfb02ba0e02c4c53d0b21c06f166ef371fa291c4e1237", _SUMS_P51),
        "p5_1_training_dt_nomix": ("p5_1/training_dt_nomix.json",
                                   "3a005607641841566dcde82b7fb0c7e237b42785dcd9da026eef57a6b18efbb8", _SUMS_P51),
        "p5_1_training_baselines": ("p5_1/training_baselines.json",
                                    "4e6906870ed0df6d64b4b56c4895855fd2d5980fe8cc026c58ce9a1442a0a768", _SUMS_P51),
        "p5_2_training_maxpressure_dt_spatial": ("p5_2/training_maxpressure_dt_spatial.json",
                                                 "b9243f4a51346b75ca98416fe8337136ac15240a0c11dc53830d34c61e39cbf1",
                                                 _SUMS_P52),
        "p5_2_training_maxpressure_dt_nomix": ("p5_2/training_maxpressure_dt_nomix.json",
                                               "a95f658089e3a227d218f8021f6e1bfe062a57f9170ac2972baa3c3249d0cfa7",
                                               _SUMS_P52),
        "p5_2_training_fixedtime_dt_spatial": ("p5_2/training_fixedtime_dt_spatial.json",
                                               "80d8b187f7b0e9a6b0e75167cebc0632938346f5b092cdb1fd4827e65b26398c",
                                               _SUMS_P52),
        "p5_2_training_fixedtime_dt_nomix": ("p5_2/training_fixedtime_dt_nomix.json",
                                             "e69e38eeafe0c4767ff9bd9d2aeca786dccc6e1ff068033c07ca360d21b91408",
                                             _SUMS_P52),
        "p5_2_training_mappo1000_dt_spatial_h4": ("p5_2/training_mappo1000_dt_spatial_h4.json",
                                                  "db96305b1c6a70d7fe7e435c80efcc167b7617415be55fd8388219def199981c",
                                                  _SUMS_P52),
        "p5_2_training_mappo1000_dt_nomix_h4": ("p5_2/training_mappo1000_dt_nomix_h4.json",
                                                "e9646a5dc6d8e99d35e667409fc7059f618df25b22d263c44c603dc885f440b9",
                                                _SUMS_P52),
        "p5_2_log_random_dt_spatial": ("p5_2/logs/train_random_dt_spatial.log",
                                       "817ad3795869f0b7a1f20227ae27f45b2afd99173da91d09431d3df841425315", _SUMS_P52),
        "p5_2_log_random_dt_nomix": ("p5_2/logs/train_random_dt_nomix.log",
                                     "d7d69f46eff28ee5dc51466072f31f9fe002c3a66a273ff034325c713a2bfe8d", _SUMS_P52),
        "sums_p5_1": ("SHA256SUMS_p5_1.txt", "023607ffe65a881932f0069412f442c65b1097bb87fdd7852873a1211fe24c61",
                      "the manifest itself, pinned here (plan §4)"),
        "sums_p5_2": ("SHA256SUMS_p5_2.txt", "fde8309b4958231f8a7e3a33cb26674f645e1c295ad688900ac8e839a19ac19b",
                      "the manifest itself, pinned here (plan §4)"),
        "sums_p5_3b": ("SHA256SUMS_p5_3b.txt", "83b10bfecedcf46e4c1276f02e6f1eef0e83f7068f19bd849dfd591cbe9cf29e",
                       "the manifest itself, pinned here (plan §4)"),
    }
    corpus_digests = {
        "cf_hz1x1__mappo1000": (
            "cbf29d3a93c211ac8252c88b35c204d33c03b1a368efe4129184adcc66479ca9",
            "0aad7090330a04afec00b9e5f97ffd1149056d16f3d37d6b8e25af1afca131dd",
            "2af1b089d7962bcaa8ab2b4f583cfa57a5fd2c37bc1337c326f939b1f5bca1bb",
            "8650a3d2963fca3036b1e33a61d17ba1091ad96029d9bb1989f005e3c9b1f9ff",
            "91243753c790c8465646ea149f3cc58b6da466adec1a852abcbe65a5feef0e2e",
        ),
        "cf_hz1x1__mappo500": (
            "befcaf879d5827a28215d1868550fa6b41a153522e9ff8de428b2d1cd7212c9a",
            "76a1b093c61e4278b00501fa6765bf351494fa8ed9c502878b6fd1fa2df3fd97",
            "6ccc628fc35929a876c341d925c0d4cd4dc4fbfcefe081b2ff3bde1893832c2c",
            "1707da00f1c1c30eb1e4967ea244144dbf33e9cfb2919eb7d7ee03b32cb877c7",
            "28a84f04640d2f1fc1b1245e0a43469af068e2a0111ed0f3ee6be5d4eab78f71",
        ),
        "cf_hz1x1__mappo060": (
            "65b132b54f60e4b06be2acaca920b91ea7c4991950cabe04009ef20b9629dc25",
            "916d10e873aa470eb85d9376143096338657699fea4e1d756646a64ea04e4741",
            "c961986499b1156914c51518f87fbce7d15236a3bef71f757bebaed796ec8819",
            "a574ac3a38413eea5e561b9c0dfe9a3ac444a1ca6699ad701b342a67c69dc33c",
            "2681d7280244e8e4e0411e24fbd2c740d0873140f10efff621e8350228a14794",
        ),
        "cf_grid4x4__mappo1000": (
            "5d7c0088070f4a84813bf6fdf729d7160a7e9e75f966b54e059a83a0d8c08d1d",
            "e53288cb9496e0bd5a6527031c2386f0f90260ed12934bf4debddf0c6c115a0e",
            "748b7574a96445fb76f1b2eefcc1ab47cb58752c8a8aa70dfd01276a01ba8537",
            "8689815c6dcde4812d62f23f0312113fad579fac40436e3a3fead45a0ac97560",
            "21d078602c25f376b75c61aa74278e5467c2348e00e734c6e3ab13ab0cfaf02d",
        ),
        "cf_grid4x4__mappo060": (
            "dd1e7a7963d418f2351b75c79f0e5c5e51be4a9c4fd94f007af9a8dfd97db99e",
            "a3b62d0eedf9cc9124e7ffc6177011d1d389793c435839add24cd19f1911f22a",
            "561edad7abfdfa1215568d2b2381611873852e927b77bdf3b9495dcdf390ecdd",
            "69fec750c62a78aaa7188a557c04602d2201ce70b910633008f74b815a73a189",
            "6820bc1a77b9686b78532173f2ecec8731b877fdeac00728a1ba802cec0e1fed",
        ),
    }
    pins: dict[str, PinnedRecord] = {}
    for key, (name, digest) in repo.items():
        named_by = "committed; " + ("added after the plan (SOURCES_ADDED_AFTER_PLAN)"
                                    if key in _ADDED else _PLAN)
        pins[key] = PinnedRecord(key, "repo", f"docs/data/{name}", digest, named_by)
    for key, (relpath, digest, named_by) in output.items():
        pins[key] = PinnedRecord(key, "output", relpath, digest, named_by)
    for collection, digests in corpus_digests.items():
        for seed, digest in zip(_SEEDS, digests):
            key = f"corpus_{collection}__seed{seed}"
            note = ("pinned here on 2026-10-03; == docs/data/p5_3c_train.json $.subjects.mappo1000.corpus_manifest_sha256"
                    if collection == "cf_hz1x1__mappo1000" else "pinned here on 2026-10-03")
            pins[key] = PinnedRecord(key, "corpus", f"{collection}__seed{seed}/manifest.json", digest, note)
    return pins


_ADDED: dict[str, str] = {
    "p5_1_declaration": (
        "grid4x4 mappo1000 DT rows: the training-set size (joint_windows) is held by P5.1's committed declaration and "
        "by no training record or checkpoint of P5.1"
    ),
    "p5_2_declaration_maxpressure": "grid4x4 P5.2 DT rows at maxpressure: the training-set size (episodes_selected)",
    "p5_2_declaration_fixedtime": "grid4x4 P5.2 DT rows at fixedtime: the training-set size (episodes_selected)",
    "p5_2_declaration_random": "grid4x4 P5.2 DT rows at random: the training-set size (episodes_selected)",
    "p5_2_declaration_mappo1000": "grid4x4 head-count-4 DT rows at mappo1000: the training-set size (episodes_selected)",
}

#: Committed records read beyond the plan's §4, each with the reason (disclosed in the Return Packet).
SOURCES_ADDED_AFTER_PLAN: dict[str, str] = dict(_ADDED)

#: Every record the builder reads, at its digest computed on 2026-10-03 -- see the module docstring's "What is pinned".
PINNED_RECORDS: dict[str, PinnedRecord] = _pins()

#: The plan §4 records the builder does NOT pin, and why no number of the table depends on them (B6.5).
UNPINNED_PLAN_RECORDS: dict[str, str] = {
    "docs/data/p5_3b_decomposition.json": (
        "plan §4 names it for the no-RTG checkpoints' digests; the builder reads the same digests from "
        "output/SHA256SUMS_p5_3b.txt (pinned), which the plan says holds them too"
    ),
    "docs/data/p7_2b_calibration.json": (
        "plan §4 lists it as context only (the hz1x1 zero-shot prompts); no number of the table is read from it"
    ),
    "docs/data/p4_6_grid.json": (
        "plan §4 names it as the P4.6 tiers' evidence; the claims cite p4_7_grid.json (pinned), which holds every P4.6 "
        "arm (verified by reviewer R3 at gate G1)"
    ),
}


@dataclass(frozen=True)
class Roots:
    """The four trees the builder reads: this checkout (``docs/data``), the main tree's ``output/``, the v1.1 corpus,
    and the latency run directory with its manifest."""

    repo_root: Path
    output_root: Path
    corpus_root: Path
    latency_dir: Path | None = None
    manifest_path: Path | None = None


@dataclass(frozen=True)
class TableRow:
    """One row of the table (Q3): an arm as the paper names it, its tiers inside.

    ``family`` selects the parameter counting (``dt``, ``spatial_dt``, ``bc``, ``iql``, ``mappo``, ``heuristic``);
    ``latency_row`` names the ``offline.compute_latency`` row whose records give the inference column (``None`` with
    ``latency_note`` for the rows that reuse another row's measurement, Q14).
    """

    row_id: str
    scenario: str
    method: str
    configuration: str
    family: str
    claims: tuple[Mapping[str, str], ...]
    groups: tuple[Mapping[str, Any], ...]
    training: tuple[Mapping[str, Any], ...]
    interactions: str
    latency_row: str | None
    latency_note: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


#: Q5: absence key -> the reason a value no record holds is written as ``null``. A row spec declares an absence as
#: ``{"kind": "absent", "key": <key>}``; a key not in this mapping refuses, and so does a value missing from its
#: record without such a declaration.
DECLARED_ABSENCES: dict[str, str] = {
    "concurrency.not_recorded": "the record does not state how many trainings shared the machine while it ran",
    "p5_1.baselines.seconds": (
        "not recorded: P5.1's baseline training record holds no seconds (output/p5_1/training_baselines.json runs "
        "carry checkpoint_path, gradient_steps, method, seed) and its log prints none"
    ),
    "p5_2.baselines.seconds": (
        "not recorded: P5.2's baseline training records were overwritten by a resume (docs/returns/P5.2.md §6a, "
        "'the six files cannot be recovered') and their logs hold only the resume's SKIP lines"
    ),
    "mappo.gradient_steps": "MAPPO's PPO update count is not recorded by results.json and is not derived here",
    "mappo.data": "online training: no logged dataset; its interactions are the environment_interactions column",
    "mappo.gpu": "trained on the CPU (results.json $.environments[*].settings.device is 'cpu')",
    "mappo.torch_version": "not recorded by results.json",
}

_CONCURRENCY_ABSENT = {"kind": "absent", "key": "concurrency.not_recorded"}

#: A seed whose wall seconds exceed this multiple of its entry's median is an OUTLIER and must be declared with its
#: note in the entry's ``outliers`` -- an undeclared one refuses, so no stall reaches the table unexplained.
OUTLIER_FACTOR = 2.0

_P4_SUSPEND = ("14,018.0 s against 202.3-356.2 s for the other four: the wall clock ran 10:19 -> 14:12, almost certainly a "
               "laptop suspend; a from-scratch retrain took 361 s with tensor-identical weights (docs/returns/P4.md:322-325)")
_HOST_GAME = ("a wall-time stall: about 8.5 GB of GPU memory was taken by a game on the Windows host, 17:22-18:04 UTC "
              "(BRIEF_42 Amendment D, D3.1); nothing in the checkpoint depends on it")
_CLOCK_JUMP = ("12,322.6 s spans a forward clock jump of about 2 h 10 min (docs/returns/P5.2.md:256-262, D-2); the "
               "measured rate of its siblings is about 122 ms per step")
_NO_CAUSE = "no committed record names a cause"
_DT_COVERS = "the gradient loop only (time.time() around it in offline.dt_gate.train_dt, dt_gate.py:831-857)"
_BASELINE_COVERS = (
    "the gradient loop only (time.time() around it in offline.offline_baselines.train_bc / train_iql, "
    "offline_baselines.py:1813-1835 / 1982-2047)"
)
_SPATIAL_COVERS = "the gradient loop only, as each training record defines its seconds (TrainResult.seconds)"
_LOOP_COVERS = "loop_seconds: the gradient loop only (perf_counter around it)"
_MAPPO_COVERS = (
    "train_sec: time.perf_counter() around experiments.runner._train_agent -- env construction, train_episodes "
    "simulated episodes, action selection and every PPO update; not the checkpoint save or the evaluation "
    "(experiments/runner.py:353-355). The worker processes it shared the machine with are an inference: see the "
    "regime's concurrency"
)


def _claim(claim: str, record: str, path: str, *, expect: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"claim": claim, "record": record, "json_path": path}
    if expect is not None:
        out["expect"] = dict(expect)
    return out


def _listed(claim: str, record: str, entry: str) -> dict[str, Any]:
    """Evidence that is a result file listed (with its digest) in a pinned campaign manifest."""
    return {"claim": claim, "record": record, "entry": entry}


def _json_list_digest(record: str, match: Mapping[str, Any], *, list_path: str = "$.runs",
                      field_name: str = "file_sha256") -> dict[str, Any]:
    return {"kind": "json_list", "record": record, "list": list_path, "match": dict(match), "field": field_name}


def _group(tier: str, path: str, digest: Mapping[str, Any]) -> dict[str, Any]:
    return {"tier": tier, "path": path, "digest": dict(digest)}


def _ckpt(path: str) -> dict[str, str]:
    return {"kind": "checkpoint", "path": path}


def _rec(path: str) -> dict[str, str]:
    return {"kind": "record", "path": path}


def _pinned(record: str, path: str) -> dict[str, str]:
    return {"kind": "pinned", "record": record, "path": path}


def _seed(name: str) -> dict[str, str]:
    return {"kind": "seed", "field": name}


def _regime(device: Mapping[str, Any], gpu: Mapping[str, Any], torch_version: Mapping[str, Any],
            threads: Mapping[str, Any], concurrency: Mapping[str, Any] = _CONCURRENCY_ABSENT) -> dict[str, Any]:
    return {"device": dict(device), "gpu": dict(gpu), "torch": dict(torch_version), "threads": dict(threads),
            "concurrency": dict(concurrency)}


_RUNTIME_REGIME = _regime(_ckpt("$.provenance.device"), _rec("$.runtime.cuda_device_name"),
                          _rec("$.runtime.torch_version"), _rec("$.runtime.torch_num_threads"))
_RUN_THREADS_REGIME = _regime(_ckpt("$.provenance.device"), _rec("$.runtime.cuda_device_name"),
                              _rec("$.runtime.torch_version"), _seed("thread_regime.torch_get_num_threads"))
_CKPT_REGIME = _regime(_ckpt("$.provenance.device"), _ckpt("$.provenance.runtime.cuda_device_name"),
                       _ckpt("$.provenance.runtime.torch_version"), _ckpt("$.provenance.runtime.torch_num_threads"))

_HZ_TIERS_P46 = ("mappo500", "maxpressure", "fixedtime", "random")
_HZ_TIERS_P47 = ("mix33", "mix50", "mix67")
_GRID_P52_TIERS = ("maxpressure", "fixedtime", "random")


def _hz_method_groups(method: str) -> tuple[dict[str, Any], ...]:
    """The 40 checkpoints of an hz1x1 method across the eight ladder tiers (P4/P4.4, P4.6, P4.7)."""
    if method == "dt":
        first = _group("mappo1000", "output/p4_dt/dt_seed{seed}.pt",
                       {"kind": "json", "record": "p4_gate", "path": "$.checkpoints['{seed}'].sha256"})
    else:
        first = _group("mappo1000", f"output/p4_4/checkpoints/{method}_seed{{seed}}.pt",
                       _json_list_digest("p4_4_training", {"method": method}))
    groups = [first]
    for tier in _HZ_TIERS_P46:
        groups.append(_group(tier, f"output/p4_6/checkpoints/{tier}_{method}_seed{{seed}}.pt",
                             _json_list_digest("p4_6_training", {"method": method, "tier": tier})))
    for tier in _HZ_TIERS_P47:
        groups.append(_group(tier, f"output/p4_7/checkpoints/{tier}_{method}_seed{{seed}}.pt",
                             _json_list_digest("p4_7_training", {"method": method, "tier": tier})))
    return tuple(groups)


_DATA_UNIT = {"bc": "windows", "bc_top10": "windows (the top-decile streams)", "iql": "transitions"}


def _hz_method_training(method: str) -> tuple[dict[str, Any], ...]:
    entries: list[dict[str, Any]] = []
    if method == "dt":
        entries.append({
            "record": "p4_training", "tier": "mappo1000", "label": "P4",
            "seeds": {"kind": "list", "path": "$.seeds", "match": {}},
            "seconds": _seed("seconds"), "steps": _seed("gradient_steps"),
            "batch": _ckpt("$.provenance.batch_size"),
            "data": {**_pinned("p4_6_declaration", "$.tiers.mappo1000.training_windows"), "unit": "windows"},
            "covers": _DT_COVERS, "regime": _RUNTIME_REGIME, "outliers": {505: _P4_SUSPEND},
        })
    else:
        entries.append({
            "record": "p4_4_training", "tier": "mappo1000", "label": "P4.4",
            "seeds": {"kind": "list", "path": "$.runs", "match": {"method": method}},
            "seconds": _seed("seconds"), "steps": _seed("gradient_steps"),
            "batch": _rec(f"$.batch_sizes.{method}"),
            "data": {**_seed("diagnostics.training_rows"), "unit": _DATA_UNIT[method]},
            "covers": _BASELINE_COVERS, "regime": _RUNTIME_REGIME,
        })
    for record, declaration, label, tiers in (("p4_6_training", "p4_6_declaration", "P4.6", _HZ_TIERS_P46),
                                              ("p4_7_training", "p4_7_declaration", "P4.7", _HZ_TIERS_P47)):
        for tier in tiers:
            data = ({**_pinned(declaration, f"$.tiers.{tier}.training_windows"), "unit": "windows"} if method == "dt"
                    else {**_seed("diagnostics.training_rows"), "unit": _DATA_UNIT[method]})
            entries.append({
                "record": record, "tier": tier, "label": label,
                "seeds": {"kind": "list", "path": "$.runs", "match": {"method": method, "tier": tier}},
                "seconds": _seed("seconds"), "steps": _seed("gradient_steps"),
                "batch": _pinned(declaration, f"$.batch_sizes.{method}"), "data": data,
                "covers": _DT_COVERS if method == "dt" else _BASELINE_COVERS, "regime": _RUN_THREADS_REGIME,
            })
    return tuple(entries)


def _hz_claims(method: str, key: str) -> tuple[dict[str, str], ...]:
    claims = [_claim("C1", "p4_7_grid", f"$.cells['{method}@{tier}']")
              for tier in ("mappo1000",) + _HZ_TIERS_P46 + _HZ_TIERS_P47]
    claims.insert(0, _claim("C1", "p4_4_baselines" if method != "dt" else "p4_gate", key))
    return tuple(claims)


def _p45_row(arm: str) -> TableRow:
    return TableRow(
        row_id=f"hz1x1.{arm}", scenario="hz1x1", method=arm,
        configuration="BC on one of P4.5's stream selections (the selection-mechanism decomposition)", family="bc",
        claims=(_claim("C1", "p4_5_baselines", f"$.cells['{arm}']"),),
        groups=(_group("mappo1000 (P4.5 selection)", f"output/p4_5/checkpoints/{arm}_seed{{seed}}.pt",
                       _json_list_digest("p4_5_selection", {"method": arm})),),
        training=({
            "record": "p4_5_selection", "tier": "mappo1000 (P4.5 selection)", "label": "P4.5",
            "seeds": {"kind": "list", "path": "$.runs", "match": {"method": arm}},
            "seconds": _seed("seconds"), "steps": _seed("gradient_steps"), "batch": _rec("$.batch_size"),
            "data": {**_seed("diagnostics.training_rows"), "unit": "windows"},
            "covers": _BASELINE_COVERS, "regime": _RUN_THREADS_REGIME,
        },),
        interactions="offline", latency_row=f"hz1x1.{arm}",
    )


def _h4_row(name: str, k: int, batch: int, subjects: tuple[str, ...]) -> TableRow:
    groups = []
    training = []
    for subject in subjects:
        arm = f"{subject}_k{k}_b{batch}"
        groups.append(_group(subject, f"output/p5_3c_training/checkpoints/{arm}_seed{{seed}}.pt",
                             {"kind": "json", "record": "p5_3c_train",
                              "path": f"$.runs['{arm}_seed{{seed}}'].checkpoint_sha256"}))
        stalls = {("mix50", 2): {101: _HOST_GAME}, ("mix50", 5): {505: _HOST_GAME}}.get((subject, k), {})
        training.append({
            "outliers": stalls,
            "record": "p5_3c_train", "tier": subject, "label": "P5.3c" + (" (exploratory)" if subject == "mix50" else ""),
            "seeds": {"kind": "dict", "path": "$.runs", "key": f"{arm}_seed{{seed}}"},
            "seconds": _seed("loop_seconds"), "steps": _seed("steps"), "batch": _seed("batch"),
            "data": {**_rec(f"$.subjects.{subject}.n_windows"), "unit": "windows"},
            "covers": _LOOP_COVERS + " (offline.context_sweep)",
            "regime": _regime(_seed("device"), _rec("$.timing.record.device_name"),
                              _ckpt("$.provenance.runtime.torch_version"), _ckpt("$.provenance.regime.torch_num_threads"),
                              _rec("$.timing.record.concurrency")),
        })
    claims = [_claim("H4", "p5_3c_context_sweep", "$.confirmatory")]
    if "mix50" in subjects:
        claims.append(_claim("H4", "p5_3c_context_sweep", "$.exploratory_mix50"))
    if batch != 64:
        claims = [_claim("H4", "p5_3c_context_sweep", "$.equal_supervision")]
    return TableRow(
        row_id=f"hz1x1.h4.{name}", scenario="hz1x1", method="dt",
        configuration=f"DT, K = {k}, batch {batch}" + (" (equal supervision)" if batch != 64 else ""), family="dt",
        claims=tuple(claims), groups=tuple(groups), training=tuple(training), interactions="offline",
        latency_row=f"hz1x1.h4.{name}",
    )


def _mappo_row(scenario: str, budget: str, checkpoint_dir: str, env_index: int, results: str,
               claims: tuple[dict[str, str], ...], note: str | None = None) -> TableRow:
    env_id = f"cf_{scenario}"
    return TableRow(
        row_id=f"{scenario}.mappo{budget}", scenario=scenario, method="mappo",
        configuration=f"MAPPO, {int(budget)} training episodes on the nominal demand (draw 0)", family="mappo",
        claims=claims,
        groups=(_group("nominal (draw 0)", f"output/checkpoints/{checkpoint_dir}/{env_id}__mappo__seed{{seed}}.pt",
                       {"kind": "corpus", "record": f"corpus_{env_id}__mappo{budget}__seed{{seed}}",
                        "path": "$.run_metadata.checkpoint_sha256"}),),
        training=({
            "record": results, "tier": "nominal (draw 0)", "label": f"P2.1 MAPPO, {int(budget)} episodes",
            "seeds": {"kind": "mappo", "env_id": env_id},
            "seconds": _seed("timings.mappo.train_sec"),
            "steps": {"kind": "absent", "key": "mappo.gradient_steps"},
            "batch": {**_rec("$.agents[0].params.minibatch_size"),
                      "label": "PPO minibatch_size: one PPO minibatch, not an offline training batch"},
            "data": {"kind": "absent", "key": "mappo.data", "unit": None},
            "covers": _MAPPO_COVERS,
            "regime": _regime(
                _rec(f"$.environments[{env_index}].settings.device"), {"kind": "absent", "key": "mappo.gpu"},
                {"kind": "absent", "key": "mappo.torch_version"},
                {"kind": "code", "value": 1, "file": "experiments/runner.py",
                 "code": "run_cell pins every cell to one torch thread (limit_torch_threads, CELL_TORCH_THREADS = 1)"},
                {"kind": "inferred", "value": 6,
                 "basis": ("6, inferred from the training run's own file mtimes: output/checkpoints.pre_c8_migration/ "
                           "mtimes minus each cell's train_sec give exactly six overlapping training intervals at every "
                           "budget (docs/plans/p8.2.md V7); RUNSPEC_01 §8's command states six for 060, 200 and 500"),
                 "confidence": "95 % for budget 1000 (docs/plans/p8.2.md, assumption A2)",
                 "recorded": "not recorded by the manifest: results.json holds no worker count"},
            ),
            "env_index": env_index,
        },),
        interactions="mappo", latency_row=f"{scenario}.mappo{budget}",
        notes=(() if note is None else (note,)),
    )


def _heuristic_row(scenario: str, kind: str, claims: tuple[dict[str, str], ...]) -> TableRow:
    names = {"maxpressure": "MaxPressure", "fixedtime": "fixed-time", "random": "random"}
    return TableRow(row_id=f"{scenario}.{kind}", scenario=scenario, method=kind,
                    configuration=f"{names[kind]} (no trained model)", family="heuristic", claims=claims, groups=(),
                    training=(), interactions="none", latency_row=f"{scenario}.{kind}")


def _grid_dt_row(method: str) -> TableRow:
    groups = [_group("mappo1000", f"output/p5_1/checkpoints/grid4x4_mappo1000_{method}_seed{{seed}}.pt",
                     {"kind": "sums", "record": "sums_p5_1",
                      "entry": f"p5_1/checkpoints/grid4x4_mappo1000_{method}_seed{{seed}}.pt"})]
    training = [{
        "record": f"p5_1_training_{method}", "tier": "mappo1000", "label": "P5.1",
        "seeds": {"kind": "list", "path": "$.runs", "match": {}},
        "seconds": _seed("seconds"), "steps": _seed("gradient_steps"), "batch": _rec("$.batch_size"),
        "data": {**_pinned("p5_1_declaration", "$.joint_windows"), "unit": "joint windows (16 intersections each)"},
        "covers": _SPATIAL_COVERS, "regime": _CKPT_REGIME,
    }]
    for tier in _GRID_P52_TIERS:
        groups.append(_group(tier, f"output/p5_2/checkpoints/grid4x4_{tier}_{method}_seed{{seed}}.pt",
                             {"kind": "sums", "record": "sums_p5_2",
                              "entry": f"p5_2/checkpoints/grid4x4_{tier}_{method}_seed{{seed}}.pt"}))
        data = {**_pinned(f"p5_2_declaration_{tier}", "$.episodes_selected"),
                "unit": "episodes (360 joint windows each)"}
        if tier == "random":
            training.append({
                "record": f"p5_2_log_random_{method}", "tier": tier, "label": "P5.2 (log; the JSON was overwritten)",
                "seeds": {"kind": "log", "method": method},
                "seconds": {"kind": "log"}, "steps": _ckpt("$.provenance.gradient_steps"),
                "batch": _ckpt("$.provenance.batch_size"), "data": data,
                "covers": _SPATIAL_COVERS + "; as the log prints it, 0.1 s resolution (docs/returns/P5.2.md §6a)",
                "regime": _CKPT_REGIME,
            })
        else:
            training.append({
                "record": f"p5_2_training_{tier}_{method}", "tier": tier, "label": "P5.2",
                "seeds": {"kind": "list", "path": "$.runs", "match": {}},
                "seconds": _seed("seconds"), "steps": _seed("gradient_steps"), "batch": _rec("$.batch_size"),
                "data": data, "covers": _SPATIAL_COVERS, "regime": _CKPT_REGIME,
            })
    mixing = "on" if method == "dt_spatial" else "off (identity graph)"
    return TableRow(
        row_id=f"grid4x4.{method}", scenario="grid4x4", method=method,
        configuration=f"spatial DT, K = 20, n_head 1, spatial mixing {mixing}, parameters shared across 16 nodes",
        family="spatial_dt",
        claims=(_claim("C1", "p5_1_grid", f"$.cells.{method}"),
                _claim("C1", "p8_4b_rederivation", "$.contrasts[5]", expect={"contrast_id": "V4"}),
                *(_listed("C1", "sums_p5_2", f"p5_2/eval_{tier}_{method}.json") for tier in _GRID_P52_TIERS)),
        groups=tuple(groups), training=tuple(training), interactions="offline", latency_row=f"grid4x4.{method}",
    )


def _grid_h4_row(method: str) -> TableRow:
    claims: list[dict[str, Any]] = [_listed("C1", "sums_p5_2", f"p5_2/eval_mappo1000_{method}.json"),
                                    _claim("C1", "p8_4b_rederivation", "$.contrasts[6]", expect={"contrast_id": "V5"})]
    if method == "dt_nomix_h4":
        claims += [_claim("C3", "p7_3d_grid4x4", "$.subject"), _claim("C3", "p7_3c_grid4x4", "$.format_version")]
    mixing = "on" if method.startswith("dt_spatial") else "off (identity graph)"
    return TableRow(
        row_id=f"grid4x4.{method}", scenario="grid4x4", method=method,
        configuration=f"spatial DT, K = 20, n_head 4, spatial mixing {mixing}", family="spatial_dt",
        claims=tuple(claims),
        groups=(_group("mappo1000", f"output/p5_2/checkpoints/grid4x4_mappo1000_{method}_seed{{seed}}.pt",
                       {"kind": "sums", "record": "sums_p5_2",
                        "entry": f"p5_2/checkpoints/grid4x4_mappo1000_{method}_seed{{seed}}.pt"}),),
        training=({
            "record": f"p5_2_training_mappo1000_{method}", "tier": "mappo1000", "label": "P5.2 (head-count 2x2)",
            "seeds": {"kind": "list", "path": "$.runs", "match": {}},
            "seconds": _seed("seconds"), "steps": _seed("gradient_steps"), "batch": _rec("$.batch_size"),
            "data": {**_pinned("p5_2_declaration_mappo1000", "$.episodes_selected"),
                     "unit": "episodes (360 joint windows each)"},
            "covers": _SPATIAL_COVERS, "regime": _CKPT_REGIME,
            "outliers": {505: _CLOCK_JUMP} if method == "dt_nomix_h4" else {},
        },),
        interactions="offline", latency_row=f"grid4x4.{method}",
    )


def _grid_baseline_row(method: str) -> TableRow:
    groups = []
    training = []
    tiers = ("mappo1000",) + _GRID_P52_TIERS
    for tier in tiers:
        from_p51 = tier == "mappo1000" and method != "bc_top10_perix"
        manifest = "sums_p5_1" if from_p51 else "sums_p5_2"
        folder = "p5_1" if from_p51 else "p5_2"
        groups.append(_group(tier, f"output/{folder}/checkpoints/grid4x4_{tier}_{method}_seed{{seed}}.pt",
                             {"kind": "sums", "record": manifest,
                              "entry": f"{folder}/checkpoints/grid4x4_{tier}_{method}_seed{{seed}}.pt"}))
        unit = ("per-intersection transitions" if method == "iql" else "per-intersection windows") + (
            " (16 per joint window, the unit of the DT rows' joint windows)")
        if from_p51:
            training.append({
                "record": "p5_1_training_baselines", "tier": tier, "label": "P5.1",
                "seeds": {"kind": "list", "path": "$.runs", "match": {"method": method}},
                "seconds": {"kind": "absent", "key": "p5_1.baselines.seconds"}, "steps": _seed("gradient_steps"),
                "batch": _ckpt("$.provenance.batch_size"),
                "data": {**_ckpt("$.provenance.diagnostics.training_rows"), "unit": unit},
                "covers": _BASELINE_COVERS, "regime": _CKPT_REGIME,
            })
        else:
            training.append({
                "record": None, "tier": tier, "label": "P5.2",
                "record_note": "P5.2's training record for this tier was lost (docs/returns/P5.2.md §6a): every value "
                               "below is read from the checkpoints",
                "seeds": {"kind": "checkpoints"},
                "seconds": {"kind": "absent", "key": "p5_2.baselines.seconds"},
                "steps": _ckpt("$.provenance.gradient_steps"), "batch": _ckpt("$.provenance.batch_size"),
                "data": {**_ckpt("$.provenance.diagnostics.training_rows"), "unit": unit},
                "covers": _BASELINE_COVERS, "regime": _CKPT_REGIME,
            })
    claims: list[dict[str, Any]] = [_listed("C1", "sums_p5_2", f"p5_2/eval_{tier}_{method}.json")
                                    for tier in (tiers if method == "bc_top10_perix" else _GRID_P52_TIERS)]
    if method != "bc_top10_perix":
        claims.insert(0, _claim("C1", "p5_1_grid", f"$.cells.{method}"))
    names = {"bc": "BC", "bc_top10": "%BC (global top decile)", "bc_top10_perix": "%BC (per-intersection top decile)",
             "iql": "IQL (policy, Q, V, target Q)"}
    return TableRow(
        row_id=f"grid4x4.{method}", scenario="grid4x4", method=method,
        configuration=f"{names[method]}, independent per intersection, parameters shared across 16 nodes",
        family="iql" if method == "iql" else "bc", claims=tuple(claims), groups=tuple(groups),
        training=tuple(training), interactions="offline", latency_row=f"grid4x4.{method}",
    )


def _fine_tune_row(subject: str, k: int, steps: int, configuration: str) -> TableRow:
    return TableRow(
        row_id=f"grid4x4.c3.{subject}", scenario="grid4x4", method="dt_nomix_h4", configuration=configuration,
        family="spatial_dt", claims=(_claim("C3", "p7_3c_grid4x4", "$.format_version"),),
        groups=(_group(f"SUMO k = {k}", f"output/p7_3c_training/checkpoints/{subject}_seed{{seed}}.pt",
                       {"kind": "json", "record": "p7_3c_finetune",
                        "path": f"$.runs['{subject}_seed{{seed}}'].checkpoint_sha256"}),),
        training=({
            "record": "p7_3c_finetune", "tier": f"SUMO k = {k}", "label": "P7.3c",
            "seeds": {"kind": "dict", "path": "$.runs", "key": f"{subject}_seed{{seed}}"},
            "seconds": _seed("loop_seconds"), "steps": _seed("steps"), "batch": _ckpt("$.provenance.batch_size"),
            "data": {**_seed("k"), "unit": "SUMO MaxPressure episodes"},
            "covers": _LOOP_COVERS + " (offline.few_shot)",
            "outliers": {303: _NO_CAUSE} if subject == "ft_k100_b16000" else {},
            "regime": _regime(_ckpt("$.provenance.device"), _ckpt("$.provenance.runtime.cuda_device_name"),
                              _ckpt("$.provenance.runtime.torch_version"),
                              _ckpt("$.provenance.runtime.torch_num_threads"), _rec("$.timing.record.concurrency")),
        },),
        interactions="offline", latency_row=f"grid4x4.c3.{subject}",
        notes=(f"fine-tuned for {steps:,} steps on SUMO episodes collected by MaxPressure (no environment interaction "
               "by the learner)",),
    )


def _rows() -> tuple[TableRow, ...]:
    hz_dt_claims = _hz_claims("dt", "$.cells.madt") + (
        _claim("C3", "p7_3a_zero_shot", "$.inputs.checkpoints"),
        _claim("H4", "p5_3c_train", "$.k20_reproduction.record.subjects.mappo1000.all_equal"),
    )
    rows: list[TableRow] = [
        TableRow("hz1x1.dt_k20", "hz1x1", "dt",
                 "DT, K = 20, n_head 1, d_model 128, n_layer 3 -- P4's recipe (the key 'madt' in P4 and P4.4, C9)",
                 "dt", hz_dt_claims, _hz_method_groups("dt"), _hz_method_training("dt"), "offline", "hz1x1.dt_k20"),
        TableRow("hz1x1.bc", "hz1x1", "bc", "BC, MLPTrunk (the DT's block without attention), policy only", "bc",
                 _hz_claims("bc", "$.cells.bc"), _hz_method_groups("bc"), _hz_method_training("bc"), "offline",
                 "hz1x1.bc"),
        TableRow("hz1x1.bc_top10", "hz1x1", "bc_top10", "%BC (the top-decile streams), MLPTrunk", "bc",
                 _hz_claims("bc_top10", "$.cells.bc_top10"), _hz_method_groups("bc_top10"),
                 _hz_method_training("bc_top10"), "offline", "hz1x1.bc_top10"),
        TableRow("hz1x1.iql", "hz1x1", "iql", "IQL, independent per intersection (policy, Q, V, target Q)", "iql",
                 _hz_claims("iql", "$.cells.iql"), _hz_method_groups("iql"), _hz_method_training("iql"), "offline",
                 "hz1x1.iql"),
        _p45_row("bc_best2_20"), _p45_row("bc_any_20"), _p45_row("bc_worst2_20"), _p45_row("bc_best2_all"),
        TableRow(
            "hz1x1.dt_nortg", "hz1x1", "dt_nortg", "DT, K = 20, rtg_mode 'zero' (the return prompt ablated)", "dt",
            (_claim("C1", "p5_3b_nortg", "$.cells"),),
            tuple(_group(tier, f"output/p5_3b/checkpoints/{tier}_dt_nortg_seed{{seed}}.pt",
                         {"kind": "sums", "record": "sums_p5_3b",
                          "entry": f"p5_3b/checkpoints/{tier}_dt_nortg_seed{{seed}}.pt"})
                  for tier in ("mappo1000", "mix50", "random")),
            tuple({
                "record": "p5_3b_nortg", "tier": tier, "label": "P5.3b",
                "seeds": {"kind": "value_dict", "path": "$.timings_seconds.train_seconds", "key": f"{tier}@{{seed}}"},
                "seconds": _seed(""), "steps": _ckpt("$.provenance.gradient_steps"),
                "batch": _ckpt("$.provenance.batch_size"),
                "data": {**_ckpt("$.provenance.training_streams"), "unit": "streams (360 windows each)"},
                "covers": "the gradient loop only (offline.nortg_campaign.train_cell, time.time() around the loop)",
                "regime": _RUNTIME_REGIME,
            } for tier in ("mappo1000", "mix50", "random")),
            "offline", "hz1x1.dt_nortg",
        ),
        _h4_row("k1", 1, 64, ("mappo1000", "mix50")),
        _h4_row("k2", 2, 64, ("mappo1000", "mix50")),
        _h4_row("k5", 5, 64, ("mappo1000", "mix50")),
        _h4_row("k10", 10, 64, ("mappo1000", "mix50")),
        _h4_row("k20", 20, 64, ("mappo1000", "mix50")),
        _h4_row("k1_b1280", 1, 1280, ("mappo1000",)),
        _h4_row("k2_b640", 2, 640, ("mappo1000",)),
        TableRow(
            "hz1x1.c3.anchor_k200", "hz1x1", "dt",
            "DT, P4's recipe, trained from scratch on 200 SUMO MaxPressure episodes (draws 201-400): C3's k = 200 "
            "endpoint", "dt", (_claim("C3", "p7_3b_anchor", "$.format_version"),),
            (_group("SUMO MaxPressure, draws 201-400", "output/p7_3b_anchor/checkpoints/anchor_dt_seed{seed}.pt",
                    _json_list_digest("p7_3b_anchor_training", {}, list_path="$.seeds", field_name="checkpoint_sha256")),),
            ({
                "record": "p7_3b_anchor_training", "tier": "SUMO MaxPressure, draws 201-400", "label": "P7.3b",
                "seeds": {"kind": "list", "path": "$.seeds", "match": {}},
                "seconds": _seed("seconds"), "steps": _seed("gradient_steps"), "batch": _rec("$.recipe.batch_size"),
                "data": {**_rec("$.n_windows"), "unit": "windows"}, "covers": _DT_COVERS, "regime": _RUNTIME_REGIME,
            },),
            "offline", "hz1x1.c3.anchor_k200",
        ),
        _mappo_row("hz1x1", "1000", "p2_1_mappo_nominal_500/p2_1_mappo_nominal_1000", 0, "mappo_results_1000",
                   (_claim("C1", "p4_gate", "$.cells.mappo1000"),
                    _claim("C1", "p4_heldout_thresholds", "$.cells.mappo1000"),
                    _claim("C1", "p4_7_grid", "$.cells['behaviour@mappo1000']"))),
        _mappo_row("hz1x1", "500", "p2_1_mappo_nominal_500/p2_1_mappo_nominal_500", 0, "mappo_results_500",
                   (_claim("C1", "p4_gate", "$.cells.mappo500"),
                    _claim("C1", "p4_7_grid", "$.cells['behaviour@mappo500']"))),
        _mappo_row("hz1x1", "060", "p2_1_mappo_nominal_060/p2_1_mappo_nominal_060", 0, "mappo_results_060",
                   (_claim("ladder", "att_ladder_v11", "$.cells[15]", expect={"scenario": "cf_hz1x1", "tier": "mappo060"}),),
                   note="a ladder-tier teacher: its only result is the corpus ladder on training draws 1-200 (Q1)"),
        _heuristic_row("hz1x1", "maxpressure", (_claim("C1", "p4_gate", "$.cells.maxpressure"),
                                                _claim("C3", "p7_3a_zero_shot", "$.cells[100]",
                                                       expect={"arm": "maxpressure"}))),
        _heuristic_row("hz1x1", "fixedtime", (_claim("C1", "p4_7_grid", "$.cells['behaviour@fixedtime']"),
                                              _claim("C3", "p7_3a_zero_shot", "$.cells[0]", expect={"arm": "fixedtime"}))),
        _heuristic_row("hz1x1", "random", (_claim("C1", "p4_7_grid", "$.cells['behaviour@random']"),
                                           _claim("C3", "p7_3a_zero_shot", "$.cells[200]", expect={"arm": "random"}))),
        _grid_dt_row("dt_spatial"),
        _grid_dt_row("dt_nomix"),
        _grid_h4_row("dt_spatial_h4"),
        _grid_h4_row("dt_nomix_h4"),
        _grid_baseline_row("bc"),
        _grid_baseline_row("bc_top10"),
        _grid_baseline_row("bc_top10_perix"),
        _grid_baseline_row("iql"),
        _fine_tune_row("ft_k5", 5, 4000, "dt_nomix_h4 fine-tuned on k = 5 SUMO episodes, 4,000 steps"),
        _fine_tune_row("ft_k20", 20, 4000, "dt_nomix_h4 fine-tuned on k = 20 SUMO episodes, 4,000 steps"),
        _fine_tune_row("ft_k100", 100, 4000, "dt_nomix_h4 fine-tuned on k = 100 SUMO episodes, 4,000 steps"),
        _fine_tune_row("ft_k100_b1000", 100, 1000, "dt_nomix_h4 fine-tuned on k = 100 SUMO episodes, 1,000 steps"),
        _fine_tune_row("ft_k100_b16000", 100, 16000,
                       "dt_nomix_h4 fine-tuned on k = 100 SUMO episodes, 16,000 steps"),
        _fine_tune_row("scratch_k100", 100, 4000,
                       "the from-scratch control: the same architecture trained on k = 100 SUMO episodes, 4,000 steps"),
        _mappo_row("grid4x4", "1000", "p2_1_mappo_nominal_500/p2_1_mappo_nominal_1000", 1, "mappo_results_1000",
                   (_claim("C1", "p5_1_grid", "$.cells.behaviour"),)),
        _mappo_row("grid4x4", "060", "p2_1_mappo_nominal_060/p2_1_mappo_nominal_060", 1, "mappo_results_060",
                   (_claim("ladder", "att_ladder_v11", "$.cells[8]", expect={"scenario": "cf_grid4x4", "tier": "mappo060"}),),
                   note="a ladder-tier teacher: its only result is the corpus ladder on training draws 1-200 (Q1)"),
        _heuristic_row("grid4x4", "maxpressure", (_listed("C1", "sums_p5_2", "p5_2/eval_maxpressure_behaviour.json"),
                                                  _claim("C3", "p7_3d_grid4x4", "$.arms.evaluated.anchors"))),
        _heuristic_row("grid4x4", "fixedtime", (_listed("C1", "sums_p5_2", "p5_2/eval_fixedtime_behaviour.json"),
                                                _claim("C3", "p7_3d_grid4x4", "$.arms.evaluated.anchors"))),
        _heuristic_row("grid4x4", "random", (_claim("C1", "p5_1_grid", "$.cells.random"),
                                             _listed("C1", "sums_p5_2", "p5_2/eval_mappo1000_random.json"))),
    ]
    reuse = (
        "not separately timed (Q14): the same checkpoints as hz1x1.dt_k20's {tier} tier, the same architecture and "
        "the same decision path; hz1x1.dt_k20's inference column applies"
    )
    for tier in ("mappo1000", "mix50"):
        group = [g for g in _hz_method_groups("dt") if g["tier"] == tier]
        entry = [e for e in _hz_method_training("dt") if e["tier"] == tier]
        if len(group) != 1 or len(entry) != 1:
            raise ValueError(f"the DT row has {len(group)} groups and {len(entry)} training entries for {tier}")
        rows.append(TableRow(
            f"hz1x1.c3.zero_shot_{tier}", "hz1x1", "dt",
            f"C3's zero-shot subject: hz1x1.dt_k20 at the {tier} tier, evaluated on SUMO without adaptation", "dt",
            (_claim("C3", "p7_3a_zero_shot", "$.inputs.checkpoints"),),
            (group[0],), (entry[0],), "offline", None, latency_note=reuse.format(tier=tier),
        ))
    return tuple(rows)


TABLE_ROWS: tuple[TableRow, ...] = _rows()


# ----------------------------------------------------------------------
# Files and paths
# ----------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """The sha256 of the file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\['([^']*)'\]|\[(\d+)\]")


def json_get(document: Any, path: str) -> Any:
    """Resolve a JSON path of the form ``$.a.b[3]['c-d']`` against *document*; refuses a path that does not resolve."""
    if not path.startswith("$"):
        raise ValueError(f"{path!r}: a JSON path starts with '$'")
    position = 1
    node = document
    while position < len(path):
        match = _TOKEN.match(path, position)
        if match is None:
            raise ValueError(f"{path!r}: cannot parse the path at {path[position:]!r}")
        name, quoted, index = match.groups()
        if index is not None:
            if not isinstance(node, list):
                raise KeyError(f"{path}: [{index}] indexes a {type(node).__name__}, not a list")
            if int(index) >= len(node):
                raise IndexError(f"{path}: index {index} is outside a list of {len(node)}")
            node = node[int(index)]
        else:
            key = name if name is not None else quoted
            if not isinstance(node, Mapping) or key not in node:
                raise KeyError(f"{path}: no key {key!r} here")
            node = node[key]
        position = match.end()
    return node


def _base(root: str, roots: Roots) -> Path:
    return {"repo": roots.repo_root, "output": roots.output_root, "corpus": roots.corpus_root}[root]


def _display(root: str, relpath: str, roots: Roots) -> str:
    if root == "repo":
        return relpath
    if root == "output":
        return f"output/{relpath}"
    return f"{roots.corpus_root.name}/{relpath}"


def read_pinned(key: str, roots: Roots, cache: dict[str, Any] | None = None) -> tuple[Any, dict[str, str]]:
    """Load a pinned record and return it with its ``{file, sha256}``; refuses an absent file or another digest."""
    if cache is not None and key in cache:
        return cache[key]
    pinned = PINNED_RECORDS.get(key)
    if pinned is None:
        raise ValueError(f"{key!r} is not a pinned record")
    path = _base(pinned.root, roots) / pinned.relpath
    if not path.is_file():
        raise FileNotFoundError(f"{pinned.relpath}: the pinned record {key!r} is absent at {path}")
    digest = sha256_file(path)
    if digest != pinned.sha256:
        raise ValueError(f"{pinned.relpath}: sha256 {digest} is not the pinned {pinned.sha256} ({key})")
    if path.suffix == ".json":
        document: Any = json.loads(path.read_text(encoding="utf-8"))
    else:
        document = path.read_text(encoding="utf-8").splitlines()
    result = (document, {"file": _display(pinned.root, pinned.relpath, roots), "sha256": digest})
    if cache is not None:
        cache[key] = result
    return result


# ----------------------------------------------------------------------
# Parameters (T-params)
# ----------------------------------------------------------------------


def parameter_names(module: Any) -> set[str]:
    """The names of *module*'s parameters -- ``named_parameters()``, never its buffers."""
    return {name for name, _ in module.named_parameters()}


def count_state_parameters(state: Mapping[str, Any], skeleton: Any) -> int:
    """Sum ``numel()`` over the entries of *state* whose names are *skeleton*'s parameters; refuses a parameter the
    state lacks. Buffers and every other tensor of *state* are excluded."""
    total = 0
    for name in sorted(parameter_names(skeleton)):
        if name not in state:
            raise KeyError(f"the payload lacks the parameter {name!r}")
        total += int(state[name].numel())
    return total


def _refuse_unknown(state: Mapping[str, Any], skeleton: Any, what: str) -> None:
    known = set(skeleton.state_dict())
    unknown = sorted(set(state) - known)
    if unknown:
        raise ValueError(f"{what}: the payload carries tensors its configuration does not build: {unknown[:5]}")


def _strip(state: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    head = f"{prefix}."
    return {key[len(head):]: value for key, value in state.items() if key.startswith(head)}


def _mappo_skeletons(learner: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch.nn as nn

    from agent.MAPPOAgent import _Actor, _CentralCritic

    actors_state = learner["actors"]
    local = [int(value) for value in learner["local_state_dims"]]
    hidden = int(actors_state["0.net.0.weight"].shape[0])
    outputs = [int(actors_state[f"{i}.net.4.weight"].shape[0]) for i in range(len(local))]
    actors = nn.ModuleList([_Actor(local[i], outputs[i], hidden) for i in range(len(local))])
    critic = _CentralCritic(sum(local) + int(learner.get("global_feature_dim", 0)), len(local), hidden)
    return actors, critic


def count_payload_parameters(family: str, payload: Mapping[str, Any]) -> dict[str, int]:
    """Route B: sum ``numel()`` over the payload's tensors whose names are the module skeleton's parameters.

    The skeleton is built from the payload's own configuration; non-module tensors (an optimiser's moments, a running
    normaliser) and buffers are excluded by construction. Returns ``trained`` and ``deployed`` for every family, plus
    ``stored`` for IQL and ``executed_per_decision`` for MAPPO.
    """
    if family == "dt":
        from agent.DTAgent import DecisionTransformer, DTConfig

        skeleton = DecisionTransformer(DTConfig.from_json_obj(payload["config"]))
        _refuse_unknown(payload["model"], skeleton, "dt")
        count = count_state_parameters(payload["model"], skeleton)
        return {"trained": count, "deployed": count}
    if family == "spatial_dt":
        from agent.SpatialDTAgent import SpatialDecisionTransformer, SpatialDTConfig

        skeleton = SpatialDecisionTransformer(SpatialDTConfig.from_json_obj(payload["config"]))
        _refuse_unknown(payload["model"], skeleton, "spatial_dt")
        count = count_state_parameters(payload["model"], skeleton)
        return {"trained": count, "deployed": count}
    if family in ("bc", "iql"):
        from agent.OfflineBaselines import MLPTrunk, TrunkConfig

        config = TrunkConfig.from_json_obj(payload["config"])
        shapes = {"policy": config.n_actions} if family == "bc" else {
            "policy": config.n_actions, "q": config.n_actions, "v": 1, "q_target": config.n_actions}
        counts: dict[str, int] = {}
        for name, out in shapes.items():
            skeleton = MLPTrunk(config, out)
            part = _strip(payload["model"], name)
            _refuse_unknown(part, skeleton, f"{family}.{name}")
            counts[name] = count_state_parameters(part, skeleton)
        expected = {f"{name}.{key}" for name in shapes for key in MLPTrunk(config, shapes[name]).state_dict()}
        stray = sorted(set(payload["model"]) - expected)
        if stray:
            raise ValueError(f"{family}: the payload carries networks this family does not build: {stray[:5]}")
        if family == "bc":
            return {"trained": counts["policy"], "deployed": counts["policy"]}
        trained = counts["policy"] + counts["q"] + counts["v"]
        return {"trained": trained, "deployed": counts["policy"], "stored": trained + counts["q_target"]}
    if family == "mappo":
        learner = payload["learner"]
        actors, critic = _mappo_skeletons(learner)
        _refuse_unknown(learner["actors"], actors, "mappo.actors")
        _refuse_unknown(learner["critic"], critic, "mappo.critic")
        deployed = count_state_parameters(learner["actors"], actors)
        trained = deployed + count_state_parameters(learner["critic"], critic)
        return {"trained": trained, "deployed": deployed, "executed_per_decision": trained}
    raise ValueError(f"no parameter route for family {family!r}")


@dataclass(frozen=True)
class _StubIntersection:
    """What ``Utils.infer_action_counts`` reads when there is no gym action space."""

    id: str
    num_phases: int


class _StubEnv:
    """A node order and its action counts: what the loaders read at construction (``spatial_footprint``'s precedent).
    No metrics pipeline, so MAPPO's C8 check reports that it cannot check (and says so in a warning)."""

    def __init__(self, ids: Sequence[str], n_actions: Sequence[int]) -> None:
        self.intersections = [_StubIntersection(str(ix), int(n)) for ix, n in zip(ids, n_actions)]
        self.action_space = None
        self.metrics = None


def _module_count(*modules: Any) -> int:
    return sum(int(p.numel()) for module in modules for p in module.parameters())


def count_loaded_parameters(family: str, path: Path, *, declared_gradient_steps: int | None,
                            method: str | None = None, expected_sha256: str | None = None) -> dict[str, int]:
    """Route A: load *path* through the method's own loader on a node-order stub env and sum ``numel()`` over the
    loaded model's ``parameters()``; the same keys as :func:`count_payload_parameters`."""
    raw = Path(path).read_bytes()
    if expected_sha256 is not None:  # B4.3: route A hashes the file it loads, before it loads it
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"{path}: sha256 {digest} is not the {expected_sha256} route A was asked to load")
    import torch

    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if family == "mappo":
        from agent.MAPPOAgent import MAPPOAgent

        learner = payload["learner"]
        n = len(learner["local_state_dims"])
        outputs = [int(learner["actors"][f"{i}.net.4.weight"].shape[0]) for i in range(n)]
        stub = _StubEnv([str(i) for i in range(n)], outputs)
        agent = MAPPOAgent(stub, device="cpu", seed=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*cannot be verified.*", category=RuntimeWarning)
            agent.load(str(path))
        deployed = _module_count(agent.learner.actors)
        trained = deployed + _module_count(agent.learner.critic)
        return {"trained": trained, "deployed": deployed, "executed_per_decision": trained}

    ids = [str(ix) for ix in payload.get("intersection_ids") or []]
    if not ids:
        # Every hz1x1 DT and baseline checkpoint and the grid4x4 baselines were saved with intersection_ids == []
        # (21 of the 36 representatives; the spatial DTs record their ids); the checkpoint's own normalisation
        # statistics are keyed by intersection id, so the node order comes from the checkpoint either way.
        stats = payload.get("stats") or {}
        ids = [str(ix) for ix in (stats.get("state_mean") or {}).get(str(payload.get("scenario_id")), {})]
    if not ids:
        raise ValueError(f"{path}: the checkpoint names no intersection, neither in intersection_ids nor in its stats")
    stub = _StubEnv(ids, [int(payload["config"]["n_actions"])] * len(ids))
    if family == "dt":
        from offline.dt_gate import load_gate_checkpoint

        agent = load_gate_checkpoint(stub, Path(path), int(declared_gradient_steps or 0), device="cpu")
        count = _module_count(agent.model)
        return {"trained": count, "deployed": count}
    if family == "spatial_dt":
        from agent.SpatialDTAgent import SpatialDTAgent
        from offline.spatial_mixing import assert_declared_budget

        assert_declared_budget(Path(path), int(declared_gradient_steps or 0), str(method))
        agent = SpatialDTAgent.from_checkpoint(stub, str(path), device="cpu")
        count = _module_count(agent.model)
        return {"trained": count, "deployed": count}
    if family in ("bc", "iql"):
        from agent.OfflineBaselines import BCAgent, IQLAgent
        from offline.offline_baselines import load_baseline_checkpoint

        load_baseline_checkpoint(stub, Path(path), int(declared_gradient_steps or 0))
        if family == "bc":
            agent = BCAgent.from_checkpoint(stub, str(path), device="cpu")
            count = _module_count(agent.model)
            return {"trained": count, "deployed": count}
        agent = IQLAgent.from_checkpoint(stub, str(path), device="cpu")
        trained = _module_count(agent.policy, agent.q, agent.v)
        return {"trained": trained, "deployed": _module_count(agent.policy),
                "stored": trained + _module_count(agent.q_target)}
    raise ValueError(f"no loader for family {family!r}")


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------


def _sums_table(lines: Sequence[str]) -> dict[str, str]:
    table: dict[str, str] = {}
    for line in lines:
        if line.strip():
            digest, relative = line.split(None, 1)
            table[relative.strip()] = digest
    return table


def _named_digest(digest: Mapping[str, Any], tier: str, seed: int, roots: Roots,
                  cache: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    kind = digest["kind"]
    if kind == "json":
        document, source = read_pinned(digest["record"], roots, cache)
        path = str(digest["path"]).format(seed=seed, tier=tier)
        return str(json_get(document, path)), {**source, "json_path": path}
    if kind == "json_list":
        document, source = read_pinned(digest["record"], roots, cache)
        items = json_get(document, digest["list"])
        hits = [i for i, item in enumerate(items)
                if int(item.get("seed", -1)) == seed and all(item.get(k) == v for k, v in digest["match"].items())]
        if len(hits) != 1:
            raise ValueError(f"{source['file']}: {len(hits)} entries of {digest['list']} match seed {seed} "
                             f"{digest['match']}, not one")
        path = f"{digest['list']}[{hits[0]}].{digest['field']}"
        return str(json_get(document, path)), {**source, "json_path": path}
    if kind == "sums":
        lines, source = read_pinned(digest["record"], roots, cache)
        entry = str(digest["entry"]).format(seed=seed, tier=tier)
        table = _sums_table(lines)
        if entry not in table:
            raise ValueError(f"{source['file']} does not list {entry}")
        return table[entry], {**source, "entry": entry}
    if kind == "corpus":
        key = str(digest["record"]).format(seed=seed)
        document, source = read_pinned(key, roots, cache)
        return str(json_get(document, digest["path"])), {**source, "json_path": digest["path"]}
    raise ValueError(f"no digest route of kind {kind!r}")


def row_checkpoints(row: TableRow, roots: Roots, cache: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every checkpoint of *row* (tier x seed): its path, the digest its named source holds and that source;
    refuses a file whose sha256 is not the named one."""
    cache = {} if cache is None else cache
    main_root = roots.output_root.parent
    out: list[dict[str, Any]] = []
    for group in row.groups:
        for seed in _SEEDS:
            relative = str(group["path"]).format(seed=seed, tier=group["tier"])
            expected, named_by = _named_digest(group["digest"], group["tier"], seed, roots, cache)
            path = main_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"{row.row_id}: {path} does not exist")
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(f"{row.row_id}: {relative} has sha256 {actual}, not the {expected} its source names")
            out.append({"tier": group["tier"], "seed": seed, "path": relative, "sha256": actual, "named_by": named_by})
    return out


# ----------------------------------------------------------------------
# Training cost
# ----------------------------------------------------------------------


def _payload(path: Path, cache: dict[str, Any]) -> Any:
    key = f"payload:{path}"
    if key not in cache:
        import torch

        cache[key] = torch.load(path, map_location="cpu", weights_only=False)
    return cache[key]


def _seed_objects(row: TableRow, entry: Mapping[str, Any], document: Any, source: Mapping[str, str] | None,
                  roots: Roots, cache: dict[str, Any]) -> list[tuple[int, str | None, Any]]:
    """``(seed, JSON path of the seed's object or None, extra)`` for the five seeds, in order."""
    locator = entry["seeds"]
    kind = locator["kind"]
    if kind == "list":
        items = json_get(document, locator["path"])
        found: list[tuple[int, str | None, Any]] = []
        for seed in _SEEDS:
            hits = [i for i, item in enumerate(items) if int(item.get("seed", -1)) == seed
                    and all(item.get(k) == v for k, v in locator["match"].items())]
            if len(hits) != 1:
                raise ValueError(f"{row.row_id} / {entry['tier']}: {len(hits)} entries of {locator['path']} match "
                                 f"seed {seed} {locator['match']}, not one")
            found.append((seed, f"{locator['path']}[{hits[0]}]", None))
        return found
    if kind in ("dict", "value_dict"):
        found = []
        for seed in _SEEDS:
            key = str(locator["key"]).format(seed=seed)
            path = f"{locator['path']}['{key}']"
            json_get(document, path)
            found.append((seed, path, None))
        return found
    if kind == "mappo":
        cells = document["cells"]
        found = []
        for seed in _SEEDS:
            hits = [i for i, cell in enumerate(cells) if cell.get("env_id") == locator["env_id"]
                    and int(cell.get("seed", -1)) == seed]
            if len(hits) != 1:
                raise ValueError(f"{row.row_id}: {len(hits)} cells of {source} for {locator['env_id']} seed {seed}")
            found.append((seed, f"$.cells[{hits[0]}]", None))
        return found
    if kind == "log":
        method = locator["method"]
        current: int | None = None
        lines_by_seed: dict[int, tuple[int, float]] = {}
        for number, line in enumerate(document, start=1):
            started = re.match(rf"TRAIN random/{re.escape(method)} seed (\d+) ->", line)
            if started:
                current = int(started.group(1))
                continue
            done = re.search(r"done in ([0-9.]+)s", line)
            if done and current is not None:
                if current in lines_by_seed:
                    raise ValueError(f"{source}: seed {current} has two 'done in' lines")
                lines_by_seed[current] = (number, float(done.group(1)))
                current = None
        if sorted(lines_by_seed) != list(_SEEDS):
            raise ValueError(f"{source}: the log's seeds are {sorted(lines_by_seed)}, not {list(_SEEDS)}")
        return [(seed, None, lines_by_seed[seed]) for seed in _SEEDS]
    if kind == "checkpoints":
        return [(seed, None, None) for seed in _SEEDS]
    raise ValueError(f"{row.row_id}: no seed locator of kind {kind!r}")


def _checkpoint_for(row: TableRow, tier: str, seed: int, roots: Roots, cache: dict[str, Any]) -> tuple[Path, dict]:
    for group in row.groups:
        if group["tier"] == tier:
            relative = str(group["path"]).format(seed=seed, tier=tier)
            path = roots.output_root.parent / relative
            key = f"sha:{path}"
            if key not in cache:
                cache[key] = sha256_file(path)
            return path, {"file": relative, "sha256": cache[key]}
    raise ValueError(f"{row.row_id}: no checkpoint group for tier {tier!r}")


def _absent(ref: Mapping[str, Any], row: TableRow, entry: Mapping[str, Any]) -> dict[str, Any]:
    key = ref.get("key")
    if key not in DECLARED_ABSENCES:
        raise ValueError(f"{row.row_id} / {entry.get('tier')}: the absence {key!r} is not declared")
    return {"value": None, "reason": DECLARED_ABSENCES[key]}


def _sourced(value: Any, source: Mapping[str, Any], ref: Mapping[str, Any], row: TableRow,
             entry: Mapping[str, Any]) -> dict[str, Any]:
    """``{value, source}``; a JSON null read from a record is refused unless the ValueRef's ``null_means`` names a
    declared absence, whose reason it then carries (B4.4)."""
    if value is not None:
        return {"value": value, "source": dict(source)}
    key = ref.get("null_means")
    if key is None:
        raise ValueError(f"{row.row_id} / {entry.get('tier')}: {source.get('file')} holds null at "
                         f"{source.get('json_path', source.get('line'))}, and no declared absence names it")
    if key not in DECLARED_ABSENCES:
        raise ValueError(f"{row.row_id} / {entry.get('tier')}: the absence {key!r} is not declared")
    return {"value": None, "reason": DECLARED_ABSENCES[key], "source": dict(source)}


def _per_seed_values(ref: Mapping[str, Any], row: TableRow, entry: Mapping[str, Any], seeds: list, document: Any,
                     source: Mapping[str, str] | None, roots: Roots, cache: dict[str, Any]) -> list[dict[str, Any]]:
    kind = ref["kind"]
    items: list[dict[str, Any]] = []
    for seed, object_path, extra in seeds:
        if kind == "seed":
            if object_path is None:
                raise ValueError(f"{row.row_id} / {entry['tier']}: a 'seed' value needs a record with seed objects")
            path = object_path if ref["field"] == "" else f"{object_path}.{ref['field']}"
            try:
                value = json_get(document, path)
            except (KeyError, IndexError) as exc:
                raise ValueError(f"{row.row_id} / {entry['tier']}: {source['file']} has no {path}, and it is not a "
                                 f"declared absence ({exc})") from exc
            items.append({"seed": seed, **_sourced(value, {**source, "json_path": path}, ref, row, entry)})
        elif kind == "checkpoint":
            checkpoint, checkpoint_source = _checkpoint_for(row, entry["tier"], seed, roots, cache)
            try:
                value = json_get(_payload(checkpoint, cache), ref["path"])
            except (KeyError, IndexError) as exc:
                raise ValueError(f"{row.row_id} / {entry['tier']}: {checkpoint} has no {ref['path']}, and it is not "
                                 f"a declared absence ({exc})") from exc
            items.append({"seed": seed, **_sourced(value, {**checkpoint_source, "json_path": ref["path"]}, ref, row,
                                                   entry)})
        elif kind == "log":
            number, value = extra
            items.append({"seed": seed, **_sourced(value, {**source, "line": number}, ref, row, entry)})
        else:
            raise ValueError(f"{row.row_id} / {entry['tier']}: a per-seed value cannot be of kind {kind!r}")
    return items


def _value(ref: Mapping[str, Any] | None, name: str, row: TableRow, entry: Mapping[str, Any], seeds: list,
           document: Any, source: Mapping[str, str] | None, roots: Roots, cache: dict[str, Any]) -> dict[str, Any]:
    """One value with its source (equal across seeds where it is read per seed)."""
    if ref is None:
        raise ValueError(f"{row.row_id} / {entry.get('tier')}: the {name} field has no source")
    kind = ref["kind"]
    if kind == "absent":
        return _absent(ref, row, entry)
    if kind == "record":
        try:
            value = json_get(document, ref["path"])
        except (KeyError, IndexError) as exc:
            raise ValueError(f"{row.row_id} / {entry['tier']}: {source['file']} has no {ref['path']}, and it is not "
                             f"a declared absence ({exc})") from exc
        return _sourced(value, {**source, "json_path": ref["path"]}, ref, row, entry)
    if kind == "pinned":
        other, other_source = read_pinned(ref["record"], roots, cache)
        return _sourced(json_get(other, ref["path"]), {**other_source, "json_path": ref["path"]}, ref, row, entry)
    if kind == "code":
        code_file = _MODULE_ROOT / ref["file"]
        return {"value": ref["value"], "source": {"code": ref["code"], "file": ref["file"],
                                                   "sha256": sha256_file(code_file)}}
    if kind == "measurement":
        return {"value": ref["value"], "source": {"measurement": ref["measurement"]}}
    if kind == "inferred":  # B6.2: a number no record holds, stated as an inference with its basis
        return {"value": ref["value"], "source": {"inferred": ref["basis"], "confidence": ref["confidence"],
                                                  "recorded": ref["recorded"]}}
    items = _per_seed_values(ref, row, entry, seeds, document, source, roots, cache)
    values = {json.dumps(item["value"], sort_keys=True) for item in items}
    if len(values) != 1:
        raise ValueError(f"{row.row_id} / {entry['tier']}: {name} differs across seeds: {sorted(values)}")
    return {key: value for key, value in items[0].items() if key != "seed"}


def training_block(row: TableRow, entry: Mapping[str, Any], roots: Roots,
                   cache: dict[str, Any] | None = None) -> dict[str, Any]:
    """One (record, tier) training entry: per-seed seconds and steps with their JSON paths, median [min-max], batch,
    data size in its record's unit, what the wall time covers and the regime; a declared absence becomes ``null``
    with its reason, an undeclared one refuses."""
    cache = {} if cache is None else cache
    for name in ("seconds", "steps", "batch", "data", "regime", "seeds", "covers"):
        if name not in entry:
            raise ValueError(f"{row.row_id} / {entry.get('tier')}: the {name} field has no source")
    record_key = entry.get("record")
    document, source = (None, None) if record_key is None else read_pinned(record_key, roots, cache)
    seeds = _seed_objects(row, entry, document, source, roots, cache)

    def per_seed(ref: Mapping[str, Any], reduce: bool) -> dict[str, Any]:
        if ref["kind"] == "absent":
            return _absent(ref, row, entry)
        items = _per_seed_values(ref, row, entry, seeds, document, source, roots, cache)
        block: dict[str, Any] = {"per_seed": items}
        if reduce:
            if any(item["value"] is None for item in items):
                raise ValueError(f"{row.row_id} / {entry['tier']}: a null cannot enter a median")
            values = [float(item["value"]) for item in items]
            median = statistics.median(values)
            block.update({"median": median, "min": min(values), "max": max(values)})
            flagged = {int(item["seed"]) for item in items if float(item["value"]) > OUTLIER_FACTOR * median}
            declared = {int(seed): note for seed, note in (entry.get("outliers") or {}).items()}
            empty = sorted(seed for seed, note in declared.items() if not str(note).strip())
            if empty:
                raise ValueError(f"{row.row_id} / {entry['tier']}: the outlier note of seed(s) {empty} is empty; a note "
                                 "names the cause, or says that no committed record names one")
            if flagged != set(declared):
                raise ValueError(
                    f"{row.row_id} / {entry['tier']}: the seeds above {OUTLIER_FACTOR} x the median are "
                    f"{sorted(flagged)}, but the entry declares {sorted(declared)}; every wall-time outlier is named "
                    "with its cause or with 'no committed record names a cause'"
                )
            block["outliers"] = [{"seed": seed, "value": next(float(i["value"]) for i in items if i["seed"] == seed),
                                  "note": declared[seed]} for seed in sorted(declared)]
        return block

    data_ref = entry["data"]
    data = _value(data_ref, "data", row, entry, seeds, document, source, roots, cache)
    if data_ref.get("kind") != "absent":
        data["unit"] = data_ref.get("unit")
    regime = {name: _value(entry["regime"].get(name), f"regime.{name}", row, entry, seeds, document, source, roots,
                           cache) for name in ("device", "gpu", "torch", "threads", "concurrency")}
    batch = _value(entry["batch"], "batch", row, entry, seeds, document, source, roots, cache)
    if entry["batch"].get("label"):
        batch["label"] = entry["batch"]["label"]
    return {
        "record": source,
        "record_note": entry.get("record_note"),
        "tier": entry["tier"],
        "label": entry["label"],
        "seconds": per_seed(entry["seconds"], reduce=True),
        "gradient_steps": per_seed(entry["steps"], reduce=False),
        "batch": batch,
        "data": data,
        "covers": entry["covers"],
        "regime": regime,
    }


# ----------------------------------------------------------------------
# The latency run
# ----------------------------------------------------------------------


def training_blocks(row: TableRow, roots: Roots, cache: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every training entry of *row*; a row with a trained model and no training entry refuses (B4.4)."""
    if row.family != "heuristic" and not row.training:
        raise ValueError(f"{row.row_id}: a row with a trained model has no training entry, so its cost would be "
                         "missing without a reason")
    return [training_block(row, entry, roots, cache) for entry in row.training]


def latency_figures(per_episode_ns: Sequence[Sequence[int]], *, warmup: int) -> dict[str, Any]:
    """The builder's OWN route (B4.1), never ``compute_latency.latency_stats``: the first *warmup* decisions of each
    episode excluded, the rest pooled and sorted; ``median_ns`` the middle value (the mean of the two middle values for
    an even count); ``p95_ns`` the nearest rank, the ``ceil(0.95 n)``-th smallest value."""
    timed = sorted(int(value) for episode in per_episode_ns for value in list(episode)[int(warmup):])
    n = len(timed)
    if n == 0:
        raise ValueError("no decision is left once the warm-up is excluded")
    median = timed[n // 2] if n % 2 else (timed[n // 2 - 1] + timed[n // 2]) / 2
    return {"n_timed": n, "median_ns": median, "p95_ns": timed[(95 * n + 99) // 100 - 1]}


def check_registry_coverage(run: Mapping[str, Any]) -> None:
    """Refuse a latency run whose records do not cover every (row, device) cell of the registry (B4.2)."""
    from offline import compute_latency as cl

    cells = {(row.row_id, device) for row in cl.ROWS for device in row.devices}
    missing = sorted(cells - set(run["records"]))
    if missing:
        raise ValueError(f"the latency run lacks {len(missing)} of the registry's {len(cells)} (row, device) cells "
                         f"(first: {missing[:3]}); the table reports every cell or none")


def join_registry_row(row: TableRow, checkpoints: Sequence[Mapping[str, Any]], latency_row: Any) -> dict[str, Any]:
    """The table row's representative checkpoint -- seed 101 of its first (headline) tier, as :func:`row_checkpoints`
    verified it -- refused unless the latency registry times exactly that file at that digest (B4.3)."""
    tier = row.groups[0]["tier"]
    found = [c for c in checkpoints if c["tier"] == tier and c["seed"] == 101]
    if len(found) != 1:
        raise ValueError(f"{row.row_id}: {len(found)} verified seed-101 checkpoints of tier {tier!r}, not one")
    representative = dict(found[0])
    if (latency_row.checkpoint, latency_row.sha256) != (representative["path"], representative["sha256"]):
        raise ValueError(f"{row.row_id}: the latency registry times {latency_row.checkpoint} ({latency_row.sha256}), not "
                         f"the row's seed-101 {tier} checkpoint {representative['path']} ({representative['sha256']})")
    return representative


_RUN_FILES = frozenset({"run.json", "canary_open.json", "canary_close.json"})


def _declared_cells(run: Mapping[str, Any]) -> list[tuple[str, str]]:
    from offline import compute_latency as cl

    cells: list[tuple[str, str]] = []
    for label in run.get("row_order") or []:
        row_id, _, device = str(label).rpartition("_")
        try:
            row = cl.row_by_id(row_id)
        except KeyError:
            raise ValueError(f"run.json declares {label}, and {row_id!r} is not a registry row") from None
        if device not in row.devices:
            raise ValueError(f"run.json declares {label}, but {row_id} is timed on {row.devices}")
        cells.append((row_id, device))
    if not cells:
        raise ValueError("run.json declares no (row, device) cell")
    return cells


def _verified_record(name: str, record: Mapping[str, Any], row: Any, device: str, run_commit: str) -> dict[str, Any]:
    """Every refusal of B4.2 for one record, then its statistics by the builder's own route (B4.1)."""
    from offline import compute_latency as cl

    if record.get("format_version") != cl.FORMAT_VERSION:
        raise ValueError(f"{name}: format {record.get('format_version')!r} is not {cl.FORMAT_VERSION}; a record of "
                         "another format is refused, never skipped")
    if record.get("row") != row.row_id or record.get("device") != device:
        raise ValueError(f"{name} records {record.get('row')} on {record.get('device')}: its row or device disagrees "
                         "with its file name")
    timed = record.get("checkpoint")
    registered = None if row.checkpoint is None else (row.checkpoint, row.sha256)
    if (None if timed is None else (timed.get("path"), timed.get("sha256"))) != registered:
        raise ValueError(f"{name}: the timed checkpoint {timed} is not the registered {registered}")
    if cl.find_outcome_keys(record):
        raise ValueError(f"{name} names an episode quantity: {cl.find_outcome_keys(record)}")
    episodes = record.get("episodes") or []
    if len(episodes) != len(cl.TIMING_DRAWS):
        raise ValueError(f"{name}: {len(episodes)} episodes, not the registered {len(cl.TIMING_DRAWS)}")
    draws = [episode.get("draw") for episode in episodes]
    if list(record.get("draws") or []) != list(cl.TIMING_DRAWS) or draws != list(cl.TIMING_DRAWS):
        raise ValueError(f"{name}: draws {record.get('draws')} (episodes on {draws}) are not the registered "
                         f"{list(cl.TIMING_DRAWS)}")
    if record.get("warmup") != cl.WARMUP:
        raise ValueError(f"{name}: warmup {record.get('warmup')} is not the registered {cl.WARMUP}")
    if record.get("engine_seed") != cl.ENGINE_SEED:
        raise ValueError(f"{name}: engine_seed {record.get('engine_seed')} is not the registered {cl.ENGINE_SEED}")
    if any(len(episode.get("decision_ns") or []) != cl.DECISIONS_PER_EPISODE for episode in episodes):
        raise ValueError(f"{name}: an episode does not hold {cl.DECISIONS_PER_EPISODE} decisions")
    git = record.get("git") or {}
    if git.get("commit") != run_commit:
        raise ValueError(f"{name}: written at commit {git.get('commit')}, not the run's {run_commit}")
    if git.get("dirty") is not False:
        raise ValueError(f"{name}: written from a dirty tree (dirty: {git.get('dirty')})")
    problems = cl.power_regime_problems((record.get("machine") or {}).get("power") or {})
    if problems:
        raise ValueError(f"{name}: written outside the power regime (Amendment B, B1): {'; '.join(problems)}")
    figures = latency_figures([episode["decision_ns"] for episode in episodes], warmup=cl.WARMUP)
    for key in ("n_timed", "median_ns", "p95_ns"):
        if figures[key] != record.get(key):
            raise ValueError(f"{name}: the recorded {key} {record.get(key)} does not recompute ({figures[key]}) from its "
                             "own nanoseconds")
    if figures["n_timed"] < cl.MIN_TIMED:
        raise ValueError(f"{name}: {figures['n_timed']} timed decisions, fewer than {cl.MIN_TIMED}")
    n_ix = cl.SCENARIOS[row.scenario].n_intersections
    if record.get("n_intersections") != n_ix:
        raise ValueError(f"{name}: n_intersections {record.get('n_intersections')} is not {row.scenario}'s {n_ix}")
    median_ms, p95_ms = figures["median_ns"] / 1e6, figures["p95_ns"] / 1e6
    per_intersection = record.get("per_intersection") or {}
    for field, recorded, derived in (("median_ms", record.get("median_ms"), median_ms),
                                     ("p95_ms", record.get("p95_ms"), p95_ms),
                                     ("per_intersection.median_ms", per_intersection.get("median_ms"), median_ms / n_ix),
                                     ("per_intersection.p95_ms", per_intersection.get("p95_ms"), p95_ms / n_ix)):
        if recorded != derived:
            raise ValueError(f"{name}: the recorded {field} {recorded} is not {derived}, derived from its nanoseconds")
    return {**figures, "n_intersections": n_ix}


def verify_latency_run(latency_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Refuse unless the run is ``COMPLETE``, both canaries are at speed and reproduced, every file is in the manifest
    at its digest, and every (row, device) cell ``run.json`` declares has its record and passes every refusal of
    B4.2; return the run's summary, each record with the nanosecond statistics the builder recomputed itself (B4.1).
    A JSON file of the run directory that is no declared cell's record, no canary and not ``run.json`` refuses."""
    from offline import compute_latency as cl

    run_dir = Path(latency_dir)
    manifest_path = Path(manifest_path)
    root = run_dir.parent
    if not (run_dir / "COMPLETE").is_file() or (run_dir / "FAILED").exists():
        raise ValueError(f"{run_dir}: the run is not COMPLETE; its records cannot be reported")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path}: the latency run's manifest is absent")
    listed = _sums_table(manifest_path.read_text(encoding="utf-8").splitlines())
    stamp = run_dir.name
    seen: set[str] = set()
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative not in listed:
            raise ValueError(f"{relative} is outside the manifest {manifest_path.name}")
        if sha256_file(path) != listed[relative]:
            raise ValueError(f"{relative}: its sha256 is not the one {manifest_path.name} lists")
        seen.add(relative)
    missing = sorted(rel for rel in listed if rel.startswith(f"{stamp}/") and rel not in seen)
    if missing:
        raise ValueError(f"{manifest_path.name} lists files the run directory lacks: {missing[:5]}")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run.get("format_version") != cl.RUN_FORMAT_VERSION or run.get("status") != "COMPLETE":
        raise ValueError(f"{run_dir}/run.json is not a COMPLETE {cl.RUN_FORMAT_VERSION} record")
    run_commit = (run.get("git") or {}).get("commit")
    if not (isinstance(run_commit, str) and re.fullmatch(r"[0-9a-f]{40}", run_commit)):
        raise ValueError(f"{run_dir}/run.json names no commit ({run_commit!r}); no record can be matched to it")
    canaries: dict[str, Any] = {}
    for phase in ("open", "close"):
        canary = json.loads((run_dir / f"canary_{phase}.json").read_text(encoding="utf-8"))
        if canary.get("format_version") != cl.CANARY_FORMAT_VERSION:
            raise ValueError(f"canary_{phase}.json is not a {cl.CANARY_FORMAT_VERSION} record")
        if canary.get("reproduced") is not True:
            raise ValueError(f"canary_{phase}: the engine did not reproduce draw 0")
        if canary.get("verdict") != "at speed" or float(canary.get("seconds", 1e9)) > cl.CANARY_MAX_SECONDS:
            raise ValueError(f"canary_{phase}: throttled ({canary.get('seconds')} s); no rate of this run is quoted")
        canaries[phase] = {**canary, "file": f"{stamp}/canary_{phase}.json", "sha256": listed[f"{stamp}/canary_{phase}.json"]}
    cells = _declared_cells(run)
    expected = {f"{row_id}_{device}.json" for row_id, device in cells}
    for path in sorted(run_dir.glob("*.json")):
        if path.name not in _RUN_FILES and path.name not in expected:
            raise ValueError(f"{path.name}: a record file for no cell run.json declares; a run reports its cells, "
                             "nothing else")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row_id, device in cells:
        name = f"{row_id}_{device}.json"
        if not (run_dir / name).is_file():
            raise ValueError(f"run.json declares {row_id}_{device}, but the run directory holds no {name}")
        record = json.loads((run_dir / name).read_text(encoding="utf-8"))
        verified = _verified_record(name, record, cl.row_by_id(row_id), device, run_commit)
        records[(row_id, device)] = {"record": record, "verified": verified, "file": f"{stamp}/{name}",
                                     "sha256": listed[f"{stamp}/{name}"]}
    return {"run_dir": str(run_dir), "stamp": stamp, "run": run, "canaries": canaries, "records": records,
            "manifest": {"file": manifest_path.name, "sha256": sha256_file(manifest_path)}}


def inference_block(row: TableRow, run: Mapping[str, Any]) -> dict[str, Any]:
    """The inference column of *row*: per device the median and p95 in ms per decision and per intersection, the
    timed count and the record's ``{file, sha256}``; "not applicable" where the row has no tensor computation. Every
    millisecond figure is the builder's own nanosecond statistic divided by 10**6 here (and by the scenario's
    intersection count from the registry), never the record's own ms field (B4.1)."""
    from offline import compute_latency as cl

    if row.latency_row is None:
        return {"note": row.latency_note, "same_as": "hz1x1.dt_k20"}
    latency_row = cl.row_by_id(row.latency_row)
    out: dict[str, Any] = {}
    for device in ("cpu", "cuda"):
        if device not in latency_row.devices:
            out[device] = {"value": None, "reason": "not applicable: no tensor computation (Q12)"}
            continue
        found = run["records"].get((latency_row.row_id, device))
        if found is None:
            raise ValueError(f"{row.row_id}: the latency run holds no {latency_row.row_id} record on {device}")
        record = found["record"]
        verified = found["verified"]
        n_ix = cl.SCENARIOS[latency_row.scenario].n_intersections
        median_ms, p95_ms = verified["median_ns"] / 1e6, verified["p95_ns"] / 1e6
        measured = {"file": f"output/p8_2/latency/{found['file']}", "sha256": found["sha256"],
                    "measurement": f"{verified['n_timed']} timed decisions, decisions {cl.WARMUP}..359 of draws "
                                   f"{list(cl.TIMING_DRAWS)}; median and nearest-rank p95 recomputed by the builder"}
        derived = {**measured, "derived": f"the whole-decision figure divided by {latency_row.scenario}'s {n_ix} "
                                          "intersection(s); not a separate timing"}
        out[device] = {
            "median_ms": {"value": median_ms, "source": measured},
            "p95_ms": {"value": p95_ms, "source": measured},
            "n_timed": verified["n_timed"],
            "per_intersection_median_ms": {"value": median_ms / n_ix, "source": derived},
            "per_intersection_p95_ms": {"value": p95_ms / n_ix, "source": derived},
            "n_intersections": n_ix,
            "factory": record["factory"],
        }
    return out


# ----------------------------------------------------------------------
# The artifact
# ----------------------------------------------------------------------


_REPRESENTATIVE_TIER = {"hz1x1.c3.zero_shot_mix50": "mix50"}

_NO_ENV_SOURCE = {
    "code": ("the offline trainers read the logged corpus and build no env: offline.dt_gate.train_dt, "
             "offline.offline_baselines.train_bc / train_iql, offline.spatial_mixing.train_spatial_dt, "
             "offline.tier_sweep.train_tier_dt, offline.few_shot.train_prepared, offline.context_sweep.train_run, "
             "offline.anchor_training.train_anchor, offline.nortg_campaign.train_cell (the one make_env in these "
             "modules is dt_gate.py:1019, inside evaluate_arm)"),
}

WHAT_THIS_DOES_NOT_SAY: tuple[str, ...] = (
    "Wall times are indicative: they were measured on a shared laptop, in different sessions and under different "
    "regimes (threads, devices, concurrency); each is reported with its own regime and no two records' seconds are "
    "pooled.",
    "The offline methods' seconds cover their gradient loops only; MAPPO's train_sec covers simulation, action "
    "selection and learning, and the six concurrent worker processes it ran with are inferred from the run's file "
    "times, not recorded.",
    "The CPU is a hybrid-core laptop part (Intel Core Ultra 9 275HX) running Linux under WSL2: the Windows host "
    "schedules the guest's virtual CPUs onto performance or efficiency cores and the guest cannot pin them, so the "
    "single-thread figure is this machine's in the recorded regime (mains, Windows power mode Best Performance), not a "
    "property of one core type.",
    "The training wall times are the training runs' own clocks: the Windows power mode in force during those runs was "
    "not recorded (the mode found on 2026-10-03 was Best power efficiency), so they are not a controlled benchmark and "
    "are not comparable to the latency regime.",
    "The latency is the policy's decision call alone, not the controller's end-to-end loop: the simulator's step, the "
    "env's construction of the observation and any I/O are outside the timer.",
    "No number here evaluates any hypothesis (PREREGISTRATION A25(c)); the timing episodes' outcomes were never "
    "recorded.",
    "The DT's parameter count is identical at every context length; K changes the computation per decision, not the "
    "model's size.",
    "The C3 rows were timed on CityFlow; their deployment domain is SUMO, where the decision call is the same code on "
    "observations of the same width (A16's alignment happens in the env).",
    "MAPPO's decision call runs its critic as well as its actors (agent/MAPPOAgent.py:247); its deployed parameters "
    "are the actors'.",
    "MAPPO's results.json digests were pinned on 2026-10-03 because no committed artifact or manifest named them "
    "(Q4); the corroborations are in mappo_results_pin.",
)


def _parameters(row: TableRow, checkpoints: list[dict[str, Any]], roots: Roots, cache: dict[str, Any]) -> dict[str, Any]:
    if row.family == "heuristic":
        zero = {"value": 0, "source": {"measurement": "a fixed rule: no trained model"}}
        return {"trained": zero, "deployed": dict(zero)}
    main_root = roots.output_root.parent
    route_b = None
    for checkpoint in checkpoints:
        counts = count_payload_parameters(row.family, _payload(main_root / checkpoint["path"], cache))
        if route_b is None:
            route_b = counts
        elif counts != route_b:
            raise ValueError(f"{row.row_id}: {checkpoint['path']} counts {counts}, not {route_b}: a row's checkpoints "
                             "must be one architecture")
    from offline import compute_latency as cl

    if row.latency_row is not None:
        latency_row = cl.row_by_id(row.latency_row)
        chosen = join_registry_row(row, checkpoints, latency_row)
        method = latency_row.method
        declared = latency_row.declared_gradient_steps
    else:
        tier = _REPRESENTATIVE_TIER.get(row.row_id, "mappo1000")
        chosen = next(c for c in checkpoints if c["tier"] == tier and c["seed"] == 101)
        method, declared = None, 40000
    representative = main_root / chosen["path"]
    route_a = count_loaded_parameters(row.family, representative, declared_gradient_steps=declared, method=method,
                                      expected_sha256=chosen["sha256"])
    if route_a != route_b:
        raise ValueError(f"{row.row_id}: route A {route_a} != route B {route_b}")
    loader = {"dt": "offline.dt_gate.load_gate_checkpoint", "spatial_dt": "agent.SpatialDTAgent.from_checkpoint",
              "bc": "agent.OfflineBaselines.BCAgent.from_checkpoint",
              "iql": "agent.OfflineBaselines.IQLAgent.from_checkpoint",
              "mappo": "agent.MAPPOAgent.MAPPOAgent.load"}[row.family]
    measurement = (f"route A ({loader} on a node-order stub env, the model's parameters()) on "
                   f"{representative.relative_to(main_root)} == route B (the payload's parameter tensors, buffers and "
                   f"other tensors excluded) on all {len(checkpoints)} checkpoints of the row")
    return {key: {"value": value, "source": {"measurement": measurement}} for key, value in route_a.items()}


def _record_cross_checks(row: TableRow, roots: Roots, cache: dict[str, Any], trained: int,
                         checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every ``parameter_count`` a record or a checkpoint already holds for this row, each equal to route A's."""
    found: list[dict[str, Any]] = []
    if row.family not in ("bc", "iql"):
        return found
    for entry in row.training:
        record_key = entry.get("record")
        seeds_locator = entry["seeds"]
        if record_key is not None and seeds_locator["kind"] == "list" and record_key.startswith("p4_"):
            document, source = read_pinned(record_key, roots, cache)
            for seed, object_path, _ in _seed_objects(row, entry, document, source, roots, cache):
                path = f"{object_path}.diagnostics.parameter_count"
                found.append({"seed": seed, "tier": entry["tier"], "value": json_get(document, path),
                              "source": {**source, "json_path": path}})
    main_root = roots.output_root.parent
    for checkpoint in checkpoints:
        payload = _payload(main_root / checkpoint["path"], cache)
        diagnostics = (payload.get("provenance") or {}).get("diagnostics") or {}
        if "parameter_count" in diagnostics:
            found.append({"seed": checkpoint["seed"], "tier": checkpoint["tier"],
                          "value": diagnostics["parameter_count"],
                          "source": {"file": checkpoint["path"], "sha256": checkpoint["sha256"],
                                     "json_path": "$.provenance.diagnostics.parameter_count"}})
    for item in found:
        if item["value"] != trained:
            raise ValueError(f"{row.row_id}: {item['source']} holds parameter_count {item['value']}, not route A's "
                             f"trained {trained}")
    return found


def _interactions(row: TableRow, roots: Roots, cache: dict[str, Any],
                  checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    if row.interactions == "offline":
        zero = {"value": 0, "source": dict(_NO_ENV_SOURCE)}
        return {"joint_decisions_per_seed": zero, "intersection_decisions_per_seed": dict(zero)}
    if row.interactions == "none":
        zero = {"value": 0, "source": {"measurement": "a fixed rule: nothing is trained"}}
        return {"joint_decisions_per_seed": zero, "intersection_decisions_per_seed": dict(zero)}
    entry = row.training[0]
    document, source = read_pinned(entry["record"], roots, cache)
    index = int(entry["env_index"])
    episodes_path = f"$.environments[{index}].settings.train_episodes"
    steps_path = f"$.environments[{index}].settings.max_steps"
    environment = json_get(document, f"$.environments[{index}].id")
    if environment != f"cf_{row.scenario}":
        raise ValueError(f"{row.row_id}: {source['file']} $.environments[{index}] is {environment!r}")
    episodes = int(json_get(document, episodes_path))
    steps = int(json_get(document, steps_path))
    joint = episodes * steps
    main_root = roots.output_root.parent
    second: list[dict[str, Any]] = []
    widths: set[int] = set()
    for checkpoint in checkpoints:
        payload = _payload(main_root / checkpoint["path"], cache)
        if int(payload["steps_done"]) != joint:
            raise ValueError(f"{row.row_id}: {checkpoint['path']} records steps_done {payload['steps_done']}, not "
                             f"train_episodes x max_steps = {joint}")
        widths.add(len(payload["learner"]["local_state_dims"]))
        second.append({"seed": checkpoint["seed"], "value": int(payload["steps_done"]),
                       "source": {"file": checkpoint["path"], "sha256": checkpoint["sha256"],
                                  "json_path": "$.steps_done"}})
    if len(widths) != 1:
        raise ValueError(f"{row.row_id}: the checkpoints disagree on the intersection count {sorted(widths)}")
    intersections = widths.pop()
    return {
        "joint_decisions_per_seed": {
            "value": joint,
            "factors": [{"value": episodes, "source": {**source, "json_path": episodes_path}},
                        {"value": steps, "source": {**source, "json_path": steps_path}}],
            "second_route": second,
        },
        "intersection_decisions_per_seed": {
            "value": joint * intersections,
            "intersections": {"value": intersections,
                              "source": {"measurement": "len($.learner.local_state_dims) of every checkpoint"}},
        },
    }


def _claims_block(row: TableRow, roots: Roots, cache: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for claim in row.claims:
        document, source = read_pinned(claim["record"], roots, cache)
        if "entry" in claim:
            table = _sums_table(document)
            if claim["entry"] not in table:
                raise ValueError(f"{row.row_id}: {source['file']} does not list {claim['entry']}")
            out.append({"claim": claim["claim"], **source, "entry": claim["entry"],
                        "entry_sha256": table[claim["entry"]]})
            continue
        node = json_get(document, claim["json_path"])  # the evidence must resolve
        for key, value in (claim.get("expect") or {}).items():
            if not isinstance(node, Mapping) or node.get(key) != value:
                raise ValueError(f"{row.row_id}: {source['file']} {claim['json_path']} is not the {key}={value!r} the "
                                 "claim names")
        out.append({"claim": claim["claim"], **source, "json_path": claim["json_path"]})
    return out


def _architecture(row: TableRow, roots: Roots, checkpoints: list[dict[str, Any]], cache: dict[str, Any]) -> str:
    if row.family == "heuristic":
        return f"heuristic.{row.method}"
    payload = _payload(roots.output_root.parent / checkpoints[0]["path"], cache)
    if row.family == "mappo":
        learner = payload["learner"]
        return f"mappo.n{len(learner['local_state_dims'])}.s{learner['local_state_dims'][0]}.h128"
    config = payload["config"]
    if row.family == "dt":
        return f"dt.s{config['state_dim']}.a{config['n_actions']}.d{config['d_model']}.l{config['n_layer']}"
    if row.family == "spatial_dt":
        return (f"spatial_dt.s{config['state_dim']}.a{config['n_actions']}.n{config['n_nodes']}.d{config['d_model']}"
                f".l{config['n_layer']}")
    return f"mlp_trunk.s{config['state_dim']}.a{config['n_actions']}.d{config['d_model']}.l{config['n_layer']}"


def _hardware(run: Mapping[str, Any]) -> dict[str, Any]:
    """The one machine every record was taken on, and the regime of EACH device (B4.4); refuses records of one device
    that disagree on their regime."""
    blocks = {json.dumps(found["record"]["machine"], sort_keys=True) for found in run["records"].values()}
    if len(blocks) != 1:
        raise ValueError(f"the latency records were taken on {len(blocks)} different machine descriptions")
    regimes: dict[str, str] = {}
    for (row_id, device), found in sorted(run["records"].items()):
        text = json.dumps(found["record"]["regime"], sort_keys=True)
        if regimes.setdefault(device, text) != text:
            raise ValueError(f"the {device} records were taken under different regimes ({row_id} differs)")
    return {"machine": json.loads(blocks.pop()), "regime": {device: json.loads(text) for device, text in regimes.items()}}


def build_artifact(roots: Roots, *, git: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``p8.2-compute/1.0`` artifact; every refusal of the module docstring runs before anything is returned."""
    if roots.latency_dir is None or roots.manifest_path is None:
        raise ValueError("the artifact needs the latency run and its manifest")
    run = verify_latency_run(roots.latency_dir, roots.manifest_path)
    check_registry_coverage(run)
    from offline import compute_latency as cl

    cache: dict[str, Any] = {}
    rows_out: list[dict[str, Any]] = []
    for row in TABLE_ROWS:
        checkpoints = row_checkpoints(row, roots, cache)
        training = training_blocks(row, roots, cache)
        parameters = _parameters(row, checkpoints, roots, cache)
        inference = inference_block(row, run)
        if row.latency_row is not None and row.family != "heuristic":
            timed = join_registry_row(row, checkpoints, cl.row_by_id(row.latency_row))
            inference["timed_checkpoint"] = {"path": timed["path"], "sha256": timed["sha256"]}
        if row.family != "heuristic":
            parameters["record_cross_checks"] = _record_cross_checks(row, roots, cache, parameters["trained"]["value"],
                                                                     checkpoints)
        rows_out.append({
            "id": row.row_id,
            "scenario": row.scenario,
            "method": row.method,
            "configuration": row.configuration,
            "architecture": _architecture(row, roots, checkpoints, cache),
            "family": row.family,
            "claims": _claims_block(row, roots, cache),
            "checkpoints": checkpoints,
            "parameters": parameters,
            "training": training,
            "environment_interactions": _interactions(row, roots, cache, checkpoints),
            "inference": inference,
            "notes": list(row.notes),
        })
    mappo_files = []
    for key in ("mappo_results_1000", "mappo_results_500", "mappo_results_060"):
        _, source = read_pinned(key, roots, cache)
        mappo_files.append(source)
    return {
        "format_version": FORMAT_VERSION,
        "registered_in": "PREREGISTRATION A25(c); docs/briefs/BRIEF_43_p8.2_compute_latency.md and its Amendments A and B",
        "git": dict(git or {}),
        "latency_run": {"stamp": run["stamp"], "manifest": run["manifest"],
                        "canaries": {phase: {"seconds": c["seconds"], "verdict": c["verdict"],
                                             "reproduced": c["reproduced"], "file": f"output/p8_2/latency/{c['file']}",
                                             "sha256": c["sha256"]} for phase, c in run["canaries"].items()},
                        "protocol": {"draws": list(cl.TIMING_DRAWS), "engine_seed": cl.ENGINE_SEED,
                                     "warmup": cl.WARMUP, "decisions_per_episode": cl.DECISIONS_PER_EPISODE,
                                     "draws_reason": ("Q8: held-out draws 1000-1002 -- no outcome is recorded "
                                                      "(T-no-outcome guarantees it), and the held-out draws are the "
                                                      "deployment distribution"),
                                     "p95": "nearest rank (numpy inverted_cdf)", "median": "numpy.median"}},
        "hardware": _hardware(run),
        "rows": rows_out,
        "mappo_results_pin": {
            "pinned_on": MAPPO_RESULTS_PINNED_ON,
            "files": mappo_files,
            "corroboration": [
                "the coordinator's survey (PROJECT_PLAN Decisions Log, 2026-10-03) quotes medians of 1,146 s (hz1x1) "
                "and 3,463 s (grid4x4); these files give 1,145.5 and 3,462.6",
                "every cell's checkpoint_path equals the corpus manifests' run_metadata.checkpoint",
                "the files' mtimes (2026-08-06) predate the checkpoints' C8 migration (2026-08-07 11:23:15)",
            ],
        },
        "sources_added_after_plan": dict(SOURCES_ADDED_AFTER_PLAN),
        "what_this_does_not_say": list(WHAT_THIS_DOES_NOT_SAY),
    }


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write the artifact once and atomically (B4.4): canonical JSON (no NaN) to a temporary file beside it, synced,
    then hard-linked to *path* -- an existing file refuses, and a failure at any step leaves no file at *path*."""
    path = Path(path)
    text = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        raise FileExistsError(f"{path} exists: the artifact is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"{path} appeared while the artifact was being written: it is written once") from None
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    """The CLI: ``build`` (the artifact) and ``params`` (the parameter counts alone, for the reviewer)."""
    parser = argparse.ArgumentParser(prog="python -m offline.compute_table", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build the p8.2-compute/1.0 artifact")
    params = sub.add_parser("params", help="print every row's route A and route B parameter counts")
    for command in (build, params):
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--corpus-root", type=Path, required=True)
        command.add_argument("--repo-root", type=Path, default=_MODULE_ROOT)
    build.add_argument("--latency-dir", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns the process exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "params":
        roots = Roots(repo_root=args.repo_root, output_root=args.output_root, corpus_root=args.corpus_root)
        cache: dict[str, Any] = {}
        for row in TABLE_ROWS:
            checkpoints = row_checkpoints(row, roots, cache)
            if row.family == "heuristic":
                print(f"{row.row_id}: no model")
                continue
            counts = _parameters(row, checkpoints, roots, cache)
            print(f"{row.row_id}: " + ", ".join(f"{k} {v['value']:,}" for k, v in counts.items() if isinstance(v, dict)))
        return 0
    roots = Roots(repo_root=args.repo_root, output_root=args.output_root, corpus_root=args.corpus_root,
                  latency_dir=args.latency_dir, manifest_path=args.manifest)
    from offline.compute_latency import _git_provenance

    artifact = build_artifact(roots, git=_git_provenance())
    write_artifact(args.out, artifact)
    print(f"{args.out}: {len(artifact['rows'])} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
