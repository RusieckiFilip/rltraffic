"""P7.3d gate G2: the measurement script prints NO OUTCOME, and its driver refuses before it runs.

``BRIEF_39`` Amendment B.2-2(2): *"A test asserts the script's text contains none of ``att_``,
``e_sumo``, ``episode_reward`` as printed fields."*  This file is that test, widened to the whole
list the ruling names -- travel time, ``e_sumo``, return, return-to-go, action -- and applied to
both artifacts of the gate: the Python module and the shell driver.

**Why a TEXT test and not only a behaviour test.**  The gate runs once, in a tmux pane, on a
machine the test suite does not have; by the time a behavioural check could observe a leaked
number the capture already holds it (``BRIEF_39`` B.1-3: a failure's ``repr`` is the known route).
A text assertion runs in CI, before the pane is started, which is the only moment at which the
leak is still preventable.  The behavioural half -- that the record's body sits entirely under
``fenced_do_not_report`` -- is asserted below on a synthetic record, with no simulator.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

import offline.g2_measure as g2

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = REPO_ROOT / "offline" / "g2_measure.py"
DRIVER = REPO_ROOT / "offline" / "campaigns" / "p7_3d_g2.sh"

#: The field names B.2-2 forbids in G2's output, and the readers that would consume them.
FORBIDDEN_FIELDS = (
    "att_",
    "e_sumo",
    "episode_reward",
    "local_return",
    "rtg_",
    "rtg_series",
    "reward_series",
    "actions",
)


def _printed_strings(source: str) -> list[str]:
    """Every string literal that reaches a ``print`` call, including f-string pieces."""
    tree = ast.parse(source)
    out: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "print":
            continue
        for argument in node.args:
            for piece in ast.walk(argument):
                if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                    out.append(piece.value)
                elif isinstance(piece, ast.Attribute):
                    out.append(piece.attr)
                elif isinstance(piece, ast.Name):
                    out.append(piece.id)
    return out


def _code_without_docstrings_or_comments(source: str) -> str:
    """The module's executable text: docstrings and comments stripped.

    The docstring EXPLAINS what is forbidden and therefore contains the forbidden words; the
    assertion is about what the code does, so the prose that documents the rule must not be what
    makes the rule appear broken.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
    return ast.unparse(tree)


def test_no_forbidden_field_is_ever_printed_by_the_g2_module() -> None:
    """B.2-2(2), the ruling's own assertion."""
    printed = " ".join(_printed_strings(MODULE.read_text(encoding="utf-8")))
    for field in FORBIDDEN_FIELDS:
        assert field not in printed, f"{field!r} reaches a print() in {MODULE.name}"
    # ... and the test is not vacuous: the module really does print things.
    assert "in-process" in printed and "peak tree RSS" in printed


def test_the_g2_module_never_reads_an_outcome_out_of_the_reconstruction() -> None:
    """The episode IS reconstructed -- the campaign's cells reconstruct, so a rate that skipped it
    would be a rate for a different cell -- and its outcome fields are never touched.

    *Mutation this is built against:* ``built.e_sumo.value`` recorded "just for the log" -> dies.
    """
    code = _code_without_docstrings_or_comments(MODULE.read_text(encoding="utf-8"))
    for attribute in ("e_sumo", "p_sumo", "w_sumo", "att_horizon", "att_env", "mean_depart_delay"):
        assert f".{attribute}" not in code, f"the module reads {attribute!r} off the reconstruction"
    assert "reconstruct_sumo_episode" in code, "the reconstruction must still happen"
    # The structural counters it MAY read are exactly these.
    assert ".n_observations" in code and ".n_teleports" in code


def test_the_measurement_record_is_entirely_fenced_and_names_what_it_is_not() -> None:
    """Every measured value sits under the key ``report`` refuses to emit, in a directory no
    ``report`` globs -- so a G2 number cannot reach an artifact even by mistake."""
    from offline.transfer_calibration import FENCED_KEY as CALIBRATION_FENCE
    from offline.transfer_curve import FENCED_KEY as CURVE_FENCE

    assert g2.FENCED_KEY == CALIBRATION_FENCE == CURVE_FENCE == "fenced_do_not_report"
    assert g2.G2_DRAW == 5, "B.2-2(1): the smoke draw, outside the held-out pool and probe band"
    assert g2.G2_DRAW not in range(1000, 1100) and g2.G2_DRAW not in range(201, 301)
    assert g2.POOL_SIZES == (4, 8, 12) and g2.DT_SEED == 101


def test_the_record_shape_puts_every_measurement_under_the_fence(tmp_path: Path) -> None:
    """Asserted on the shape the writer builds, with no simulator: the top level carries only the
    format, the gate, the disclaimer and provenance; everything else is fenced."""
    record = {
        "format_version": g2.G2_FORMAT_VERSION,
        "gate": "G2",
        "what_this_is": "a timing and memory measurement",
        g2.FENCED_KEY: {"single_worker": {"dt": {"in_process_seconds": 1.0}}},
        "git_commit": "a" * 40,
        "git_dirty": False,
    }
    unfenced = set(record) - {g2.FENCED_KEY}
    assert unfenced == {"format_version", "gate", "what_this_is", "git_commit", "git_dirty"}
    for key in unfenced:
        assert not isinstance(record[key], dict), f"{key} could hide a measurement"


def test_the_pool_and_cell_results_carry_durations_and_memory_and_nothing_else() -> None:
    """The keys the two measurement functions may return, pinned by name.

    A field added later that happens to be an outcome would have to be added here too, which is
    the point: this list is the contract, and it is short enough to read.
    """
    source = MODULE.read_text(encoding="utf-8")
    cell_block = source.split("def measure_single_cell", 1)[1].split("def _pool_worker", 1)[0]
    returned = set(re.findall(r'^\s+"([a-z_0-9]+)":', cell_block, flags=re.MULTILINE))
    assert returned == {
        "arm", "draw_id", "scenario_key", "seed", "halting_check", "in_process_seconds",
        "env_and_agent_build_seconds", "rollout_seconds", "reconstruct_seconds", "memory",
        "self_hwm_mib", "torch_cuda_available", "structural", "n_observations", "n_teleports",
        "vehicle_types_seen", "time_to_teleport_option", "n_intersections",
    }, sorted(returned)


# ==================================================================================
# The driver's text: the refusals, the cwd rule, and what it says it does not do
# ==================================================================================
def _driver_text_without_comments() -> str:
    lines = DRIVER.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def test_the_driver_refuses_before_it_measures_and_in_the_right_order() -> None:
    """A refused start must change nothing, so every refusal precedes the first measurement."""
    text = _driver_text_without_comments()
    order = [
        "REFUSING TO START: no interpreter",
        "REFUSING TO START: offline.g2_measure loaded from",
        "REFUSING TO START: cells from another run are still alive",
        "REFUSING TO START: not a process-group leader",
        "REFUSING TO START: SIGINT is IGNORED",
        "REFUSING TO START: RLTRAFFIC_GRID4X4_RESCO=",
        "REFUSING TO START: missing input",
        "REFUSING TO START: the worktree",
        "trap on_signal INT TERM HUP",
        "-m offline.g2_measure",
    ]
    positions = [text.index(fragment) for fragment in order]
    assert positions == sorted(positions), "a refusal moved after the work it is meant to prevent"
    assert "set -euo pipefail" in text


def test_the_driver_carries_the_cwd_rule_itself(tmp_path: Path) -> None:
    """Amendment B3/Q4: the rule lives in the driver, not in a launcher outside the repository."""
    text = _driver_text_without_comments()
    assert "cd \"$MAIN\"" in text, "the render cwd is the MAIN tree (A6, DEFERRED 87)"
    assert "PYTHONPATH=$WORK_TREE" in text
    assert "m.__file__" in text and 'case "$LOADED" in' in text, "it asserts WHICH code loaded"
    assert ': "${RLTRAFFIC_GRID4X4_RESCO:=$MAIN/scenarios/grid4x4_candidates}"' in text
    assert "export RLTRAFFIC_GRID4X4_RESCO" in text


def test_the_driver_consumes_no_token_and_writes_no_campaign_chunk() -> None:
    text = _driver_text_without_comments()
    assert "token" not in text.lower(), "G2 consumes no authorisation, so it never names one"
    for forbidden in ("p7_3d_grid4x4.sh", "--stage", "report"):
        assert forbidden not in text, f"the G2 driver must not reach {forbidden!r}"


def test_the_driver_prints_no_forbidden_field_either() -> None:
    text = _driver_text_without_comments()
    for field in ("att_", "e_sumo", "episode_reward", "local_return", "rtg_"):
        assert field not in text, f"{field!r} appears in the G2 driver's executable text"


def test_the_tmux_command_in_the_header_is_the_one_the_author_is_given() -> None:
    """The command handed over must be the command the header documents, character for character:
    a driver documented one way and started another is how a run's provenance stops meaning
    anything."""
    header = DRIVER.read_text(encoding="utf-8")
    expected = (
        "tmux new -s p73d_g2 'bash /home/filip/rltraffic-p73d/offline/campaigns/p7_3d_g2.sh "
        "2>&1 | tee -a /home/filip/rltraffic/output/p7_3d_runs/g2_capture.txt'"
    )
    assert expected in header


@pytest.mark.parametrize("name", ["ensure_draw_parity", "measure_single_cell", "measure_pool"])
def test_every_public_measurement_function_is_exported(name: str) -> None:
    assert name in g2.__all__ and callable(getattr(g2, name))


# ==================================================================================
# The pieces that can be exercised without a SUMO episode
# ==================================================================================
def test_the_memory_sampler_observes_this_process_and_its_children() -> None:
    """A pool's cost is the SUM over live processes, which ``resource.getrusage`` cannot give.

    *Mutation this is built against:* the sampler reads only its own process -> the tree peak
    equals the self peak even while a child holds memory -> this dies.
    """
    import subprocess
    import sys

    with g2.MemorySampler(sample_gpu=False) as sampler:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; x = bytearray(80 * 1024 * 1024); time.sleep(1.5)"]
        )
        try:
            child.wait(timeout=30)
        finally:
            if child.poll() is None:  # pragma: no cover - only on a timeout
                child.kill()

    peak = sampler.peak
    assert peak.n_samples >= 2, "the sampler thread never ran"
    assert peak.peak_self_rss_kib > 0
    assert peak.max_live_processes >= 2, "the child was never seen"
    assert peak.peak_tree_rss_kib >= peak.peak_self_rss_kib + 60 * 1024, (
        "the tree peak does not include the child's 80 MiB, so a pool's memory would be "
        f"under-reported (tree {peak.peak_tree_rss_kib} KiB, self {peak.peak_self_rss_kib} KiB)"
    )
    record = peak.as_record()
    assert record["peak_tree_rss_mib"] >= record["peak_self_rss_mib"] > 0
    assert set(record) == {
        "peak_tree_rss_mib", "peak_self_rss_mib", "peak_gpu_used_mib",
        "n_samples", "max_live_processes", "sample_interval_seconds",
    }


def test_descendants_finds_a_grandchild_and_not_the_whole_machine() -> None:
    import os
    import subprocess
    import sys

    child = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess, sys, time; "
         "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)']); "
         "time.sleep(2); p.wait()"]
    )
    try:
        import time

        time.sleep(0.8)
        found = g2._descendants(os.getpid())
        assert child.pid in found, "the direct child is missing"
        assert len(found) >= 2, "the grandchild is missing, so a pool's tree would be undercounted"
        assert os.getpid() not in found, "a process is not its own descendant"
    finally:
        child.kill()
        child.wait(timeout=30)


def test_ensure_draw_parity_renders_the_smoke_draw_and_reports_what_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B.2-2(1): draw 5 has no ``parity/`` because C1 rendered 201-300 and 1000-1099.

    Rendered here into ``tmp_path`` with C1's own tool -- the real function the driver calls, on
    the real scenario -- so a typo in it is found before the author starts the pane, not inside it.
    """
    import os

    resco = os.environ.get("RLTRAFFIC_GRID4X4_RESCO")
    if not resco or not (
        Path(resco) / "resco/resco_benchmark/environments/grid4x4/grid4x4.net.xml"
    ).is_file():
        pytest.skip(
            "RLTRAFFIC_GRID4X4_RESCO is unset (or holds no grid4x4 net): grid4x4's parity render "
            "needs the read-only candidates root"
        )
    from offline.materialise_draws import materialise

    config = REPO_ROOT / "configs/sim/cityflow_grid4x4.json"
    materialise(config, [g2.G2_DRAW], out_root=tmp_path)

    record = g2.ensure_draw_parity(g2.G2_DRAW, out_root=tmp_path, env_config=config)

    assert record["draw_id"] == 5 and record["action"] == "written"
    assert record["format_version"] == "materialised-draw-parity/1.1"
    assert record["n_bound"] == record["n_vehicles"] > 0
    assert record["vtype_attributes"]["tau"] == "1.5", "A15(g)'s grid4x4 value"
    assert record["net_sha256"].startswith("8d192de4")
    assert record["route_template"]["member"] == "grid4x4_1.rou.xml"
    assert Path(record["config_path"]).is_file()
    # Idempotent: the driver may be restarted, and a second run must not rewrite the draw.
    again = g2.ensure_draw_parity(g2.G2_DRAW, out_root=tmp_path, env_config=config)
    assert again["action"] == "kept"
    assert again["files"] == record["files"]


# ==================================================================================
# The cwd/sys.path seam -- the defect that made the first hand-over refuse (2026-09-20)
# ==================================================================================
def _python_invocations() -> list[str]:
    """Every line of the driver that runs the interpreter."""
    return [
        line.strip()
        for line in _driver_text_without_comments().splitlines()
        if '"$PY"' in line and "-x" not in line
    ]


def test_every_python_invocation_isolates_the_cwd_from_sys_path() -> None:
    """``-P`` on BOTH calls, because the cwd is the MAIN tree and it carries its own ``offline``.

    Rule 3 puts the process cwd in the main tree so a rendered draw embeds the main tree's ``dir``
    (Amendment A6, ``DEFERRED`` 87), while the CODE must be the worktree's.  Python prepends the
    cwd to ``sys.path`` for both ``-c`` and ``-m``, so without ``-P`` ``offline`` resolves to the
    main tree's package -- which has no ``g2_measure`` -- and the import fails outright.  **That is
    not hypothetical: the first hand-over refused with ``loaded from 'nothing'``.**
    """
    invocations = _python_invocations()
    assert len(invocations) == 2, invocations
    for line in invocations:
        assert " -P " in line, f"no -P, so the cwd would shadow the worktree: {line}"
    assert any(" -c " in line for line in invocations)
    assert any(" -m offline.g2_measure" in line for line in invocations)


def test_the_import_check_resolves_to_the_worktree_from_a_cwd_that_shadows_it(
    tmp_path: Path,
) -> None:
    """Executable, hermetic, and with its own negative control.

    A decoy ``offline`` package is placed in the working directory -- exactly the shape the main
    tree has -- and the driver's OWN import-check command, extracted from the script rather than
    retyped, is run from there.  It must resolve to this worktree.  The same command with ``-P``
    removed must NOT, which is what proves the assertion is about the flag and not about luck.
    """
    import re
    import subprocess
    import sys

    decoy = tmp_path / "decoy"
    (decoy / "offline").mkdir(parents=True)
    (decoy / "offline" / "__init__.py").write_text("", encoding="utf-8")

    line = next(line for line in _python_invocations() if " -c " in line)
    match = re.search(r"-P -c '([^']+)'", line)
    assert match is not None, f"the import check is not the shape this test extracts: {line}"
    code = match.group(1)

    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO_ROOT)}
    with_p = subprocess.run(
        [sys.executable, "-P", "-c", code], cwd=decoy, env=env,
        capture_output=True, text=True, check=False,
    )
    assert with_p.returncode == 0, with_p.stderr
    assert with_p.stdout.strip() == str(MODULE), with_p.stdout

    without_p = subprocess.run(
        [sys.executable, "-c", code], cwd=decoy, env=env,
        capture_output=True, text=True, check=False,
    )
    assert without_p.returncode != 0, (
        "the negative control passed: the decoy package did not shadow the worktree, so this "
        "test would not have caught the missing -P"
    )
    assert "ModuleNotFoundError" in without_p.stderr


def test_the_written_records_own_disclaimer_names_no_forbidden_field() -> None:
    """The disclaimer says what is absent; it must not do so by NAMING the fields.

    An artifact whose prose contains ``e_sumo`` matches any grep for an outcome leak, which is
    exactly the search a reviewer runs.  Found on 2026-09-20 by the fence check in the packet,
    after the gate had run; the record on disk from that run carries the earlier wording and the
    packet says so.  *Mutation this is built against:* the old sentence restored -> this dies.
    """
    source = MODULE.read_text(encoding="utf-8")
    block = source.split('"what_this_is": (', 1)[1].split("),", 1)[0]
    for field in FORBIDDEN_FIELDS:
        assert field not in block, f"{field!r} is named in the record's own disclaimer"
    assert "no travel time" in block and "no episode return" in block, "it still says what is absent"



# ==================================================================================
# C5's probe driver: the same refusal discipline, and the rate it cites is G2's
# ==================================================================================
PROBE_DRIVER = REPO_ROOT / "offline" / "campaigns" / "p7_3d_probe.sh"


def _probe_text_without_comments() -> str:
    lines = PROBE_DRIVER.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def test_the_probe_driver_refuses_before_the_first_episode_and_in_order() -> None:
    text = _probe_text_without_comments()
    order = [
        "REFUSING TO START: no interpreter",
        "REFUSING TO START: offline.transfer_calibration loaded from",
        "REFUSING TO START: episodes from another run are still alive",
        "REFUSING TO START: not a process-group leader",
        "REFUSING TO START: SIGINT is IGNORED",
        "REFUSING TO START: RLTRAFFIC_GRID4X4_RESCO=",
        "REFUSING TO START: missing parity configuration",
        "CityFlow probe chunks in",
        "REFUSING TO START: the worktree",
        "below the ${RSS_BUDGET_MIB} MiB budget",
        "trap on_signal INT TERM HUP",
        "run_sumo_probe_per_intersection",
    ]
    positions = [text.index(fragment) for fragment in order]
    assert positions == sorted(positions), "a refusal moved after the work it prevents"
    assert "set -euo pipefail" in text


def test_the_probe_driver_carries_g2s_measured_numbers_not_invented_ones() -> None:
    """``WORKERS`` and the RSS budget come from gate G2, and the budget is that measurement.

    *Mutation this is built against:* the budget edited to a round number that no measurement
    supports -> the arithmetic below stops holding -> this dies.
    """
    text = _probe_text_without_comments()
    values = {
        name: int(re.search(rf"^{name}=(\d+)$", text, flags=re.MULTILINE).group(1))
        for name in ("WORKERS", "PEAK_TREE_RSS_MIB", "PER_WORKER_RSS_MIB", "RSS_BUDGET_MIB", "GPU_PEAK_MIB")
    }
    assert values["WORKERS"] == 12, "G2 measured 12 as the best throughput on these 16 cores"
    assert values["PEAK_TREE_RSS_MIB"] == 17297, "G2's measured peak tree RSS at W = 12"
    assert values["GPU_PEAK_MIB"] == 4595, "G2's measured peak GPU use at W = 12"
    # The per-worker figure IS the peak divided by the workers, and the budget IS 1.4x the peak.
    assert values["PER_WORKER_RSS_MIB"] == round(values["PEAK_TREE_RSS_MIB"] / values["WORKERS"])
    assert values["RSS_BUDGET_MIB"] == round(values["PEAK_TREE_RSS_MIB"] * 1.4)
    # The header cites the gate, its date, its commit and its canary.
    header = PROBE_DRIVER.read_text(encoding="utf-8")
    for citation in ("gate G2", "2026-09-20", "07ab267", "canary 0.76 s"):
        assert citation in header, f"the schedule does not cite {citation!r}"
    assert "UNEXPLAINED" in header, "the W = 8 anomaly must be carried, not smoothed away"


def test_the_probe_driver_isolates_the_cwd_and_names_its_own_tmux_command() -> None:
    text = _probe_text_without_comments()
    invocations = [line for line in text.splitlines() if '"$PY"' in line and "-x" not in line]
    assert len(invocations) == 2 and all(" -P " in line for line in invocations), invocations
    assert 'cd "$MAIN"' in text and "PYTHONPATH=$WORK_TREE" in text
    expected = (
        "tmux new -s p73d_probe 'bash /home/filip/rltraffic-p73d/offline/campaigns/"
        "p7_3d_probe.sh 2>&1 | tee -a /home/filip/rltraffic/output/p7_3d_runs/probe_capture.txt'"
    )
    assert expected in PROBE_DRIVER.read_text(encoding="utf-8")


def test_the_probe_driver_checks_both_halves_and_writes_no_artifact() -> None:
    """Rule B's ratio needs both domains over the same draws, so the CityFlow half is a
    precondition; and the driver assembles nothing -- the artifact is built from the chunks."""
    text = _probe_text_without_comments()
    assert "probe_cityflow_draw_*.json" in text and "-ne 100" in text
    for forbidden in ("--report", "token"):
        assert forbidden not in text, f"the probe driver must not reach {forbidden!r}"
    # ⚠️ The artifact paths are asserted as WRITE TARGETS, not as strings. The driver legitimately
    # NAMES docs/data/p7_3d_cap_e.json in a refusal (where the band's admission is recorded) and
    # docs/data/p7_3d_calibration.json in its closing line (what the operator does next); banning
    # the substrings would ban two useful messages. What the contract says is that nothing is
    # written there, so that is what is checked -- line by line, against the ways a shell writes.
    writers = ("> ", ">>", "cp ", "mv ", "tee ", "--out", "write_text", "mkdir")
    for line in text.splitlines():
        if "docs/data" in line or "p7_3d_calibration.json" in line:
            assert not any(w in line for w in writers), f"an artifact is written on: {line.strip()}"
    # ... and the check is not vacuous: the driver does mention both paths.
    assert "p7_3d_cap_e.json" in text and "p7_3d_calibration.json" in text
