"""P9.1's report (``BRIEF_46`` 3.4 ``rescore`` / ``report``, 3.5): the five A27 artifacts, pure functions of the run
directories and the committed artifacts.

Format ``p9.1-report/1.0``.  Every per-episode record carries ``arm``, ``tier``, ``seed``, ``draw_id``, ``att_ours``,
``att_engine``, ``entered``, ``created``, ``never_entered``, ``entered_fraction``, ``episode_reward``,
``horizon_vehicle_count`` and ``source``.  **Alignment convention:** one record per held-out ``(seed, draw)`` of a cell,
in ``(seed, draw)`` order; ``att_ours``, ``horizon_vehicle_count`` and ``episode_reward`` are route (i)'s (the registered
evaluation: ``method_tier_grid`` / ``tier_sweep`` ``evaluate``) and ``att_engine`` with the companions are route (ii)'s
(P8.4b's per-second observer, ``att_rederivation.run_campaign``), joined on ``(seed, draw)`` and refused unless route
(ii)'s ``att_ours`` equals route (i)'s under ``==``.  An OLD hz1x1 cell's ``att_ours`` is its committed artifact's,
joined to its P8.4b re-roll the same way (A27(a)).

Two additions made before G4, under the same version (no artifact has been written in it):
- **Each run's code commit (``BRIEF_46`` G.2).** Every training run of a new cell carries ``code_commit``,
  ``run_tree``, ``session`` and ``code_commit_source``, from the token line of its session (:func:`run_session`).
  Every artifact states the rule in ``code_commit_rule``, and carries the captures' ``sessions`` and the
  ``evaluation_code_commit`` (S.2, Q20: :func:`stage_code_commit`). ``p9_1_iql_targets.json`` names the D4RL re-run's
  session in ``d4rl_session``, beside ``d4rl``.
- **The IQL diagnostics (Q18, Q19; E.7, F.2-3, S.2).** ``p9_1_iql_targets.json`` carries ``iql_diagnostics`` beside
  ``d4rl``: each D4RL cell's final Q and V losses and AWR near-zero fraction for ``iql`` and ``iql_t``, and the same,
  with the arm and its target mode, for every IQL-family traffic run. They are labelled diagnostics, not registered
  statistics (:func:`iql_diagnostics`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import offline.corrective_round as cr

REPORT_FORMAT_VERSION = "p9.1-report/1.0"

#: The fields of every per-episode record (``BRIEF_46`` 3.5), ``entered_fraction`` added beside the companions.
EPISODE_FIELDS: tuple[str, ...] = (
    "arm", "tier", "seed", "draw_id", "att_ours", "att_engine", "entered", "created", "never_entered",
    "entered_fraction", "episode_reward", "horizon_vehicle_count", "source",
)


def _relative(roots: cr.Roots, path: str | Path) -> str:
    """A path under the main tree's ``output/``, written as the repository names it."""
    return f"output/{Path(path).resolve().relative_to(Path(roots.output_root).resolve()).as_posix()}"


def _seed_of(value: Any) -> int | None:
    return None if value is None else int(value)


def _joined_row(arm: str, tier: str, first: Mapping[str, Any], rerolled: Mapping[str, Any], *, att_ours: float,
                source: str, refusal: str) -> dict[str, Any]:
    """One record: *att_ours* and *first*'s two route-(i) companions, *rerolled*'s ``att_engine`` and A11(b)'s
    companions -- refused unless *rerolled* is the same episode and its ``att_ours`` (and the value its own run was
    checked against) equal *att_ours* under ``==``."""
    seed, draw = _seed_of(first.get("seed")), int(first["draw_id"])
    if (str(rerolled["arm"]), _seed_of(rerolled.get("seed")), int(rerolled["draw_id"])) != (arm, seed, draw):
        raise ValueError(f"{arm} seed {seed} draw {draw}: the P8.4b file holds {rerolled['arm']} seed "
                         f"{rerolled.get('seed')} draw {rerolled['draw_id']}")
    if not (float(rerolled["att_ours"]) == att_ours and float(rerolled["committed_att_ours"]) == att_ours
            and rerolled["reproduces_committed"] is True):
        raise ValueError(f"{arm} seed {seed} draw {draw}: {refusal} (att_ours {att_ours!r} against "
                         f"{rerolled['att_ours']!r}, checked against {rerolled['committed_att_ours']!r})")
    return {
        "arm": arm, "tier": tier, "seed": seed, "draw_id": draw, "att_ours": att_ours,
        "att_engine": float(rerolled["att_engine"]), "entered": int(rerolled["entered"]),
        "created": int(rerolled["created"]), "never_entered": int(rerolled["never_entered"]),
        "entered_fraction": float(rerolled["entered_fraction"]), "episode_reward": float(first["episode_reward"]),
        "horizon_vehicle_count": float(first["horizon_vehicle_count"]), "source": source,
    }


def _rerolled(work: Path, scenario: str, arm: str, seed: int | None, draw: int, missing: str) -> dict[str, Any]:
    from offline import att_rederivation as ar

    path = work / ar.cell_file_name(ar.CellKey(scenario, arm, seed, draw))
    if not path.is_file():
        raise ValueError(f"{scenario}:{arm} seed {seed} draw {draw}: {missing} ({path})")
    return json.loads(path.read_text(encoding="utf-8"))


def new_cell_rows(roots: cr.Roots, cell: cr.Cell) -> tuple[dict[str, Any], ...]:
    """A NEW cell's records: route (i)'s complete cell file joined to route (ii)'s per-episode files."""
    path = cr.route_i_output(roots, cell)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cr._assert_route_i_complete(payload, cell)
    work = Path(roots.runs_root) / "eval_ii" / cell.scenario
    source = f"new: route (i) {_relative(roots, path)}; route (ii) {_relative(roots, work)}, att_ours equal under =="
    rows = []
    for episode in sorted(payload["episodes"], key=lambda e: (int(e["seed"]), int(e["draw_id"]))):
        seed, draw = int(episode["seed"]), int(episode["draw_id"])
        rerolled = _rerolled(work, cell.scenario, cell.key, seed, draw, "route (ii) has no episode")
        rows.append(_joined_row(cell.key, cell.tier, episode, rerolled, att_ours=float(episode["att_horizon"]),
                                source=source, refusal="the two routes disagree"))
    return tuple(rows)


def old_cell_tier(arm: str) -> str:
    """An old hz1x1 arm's tier, by P8.4b's own name resolution (P4.4's bare names are ``mappo1000``'s)."""
    from offline import att_rederivation as ar

    return ar.normalise_arm("hz1x1", arm)[1]


def old_hz1x1_rows(roots: cr.Roots, artifact: str, arm: str) -> tuple[dict[str, Any], ...]:
    """An OLD hz1x1 cell's records: its committed artifact's episodes joined to their P8.4b re-rolls
    (``output/p9_1_runs/old_cells/hz1x1``), each re-roll refused unless it reproduces the committed ``att_ours``."""
    payload = json.loads((Path(roots.repo_root) / artifact).read_text(encoding="utf-8"))
    episodes = sorted((e for e in payload["episodes"] if str(e["arm"]) == arm),
                      key=lambda e: (-1 if e.get("seed") is None else int(e["seed"]), int(e["draw_id"])))
    if not episodes:
        raise ValueError(f"{artifact} carries no episodes of {arm!r}")
    work = Path(roots.runs_root) / "old_cells" / "hz1x1"
    tier = old_cell_tier(arm)
    source = f"{artifact}; att_engine and companions re-rolled, reproduces committed att_ours"
    rows = []
    for episode in episodes:
        rerolled = _rerolled(work, "hz1x1", arm, _seed_of(episode.get("seed")), int(episode["draw_id"]),
                             "not re-rolled")
        rows.append(_joined_row(arm, tier, episode, rerolled, att_ours=float(episode["att_horizon"]), source=source,
                                refusal="the re-roll does not reproduce the committed att_ours"))
    keys = [(row["seed"], row["draw_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{artifact}: an episode of {arm} appears twice")
    return tuple(rows)


#: The keys of P5.2c's cell rows (``tier_statements.Cell``), in which the scorers read every cell.
STATEMENT_ROW_KEYS: tuple[str, ...] = (
    "arm", "seed", "draw_id", "att_ours", "att_engine", "entered", "created", "never_entered", "entered_fraction",
    "horizon_vehicle_count", "episode_reward",
)


def as_statement_cell(cell_id: str, method: str, tier: str, arm: str, source: str, eval_file: str,
                      rows: Sequence[Mapping[str, Any]]) -> Any:
    """*rows* as P5.2c's ``tier_statements.Cell`` -- the type every registered scorer reads -- with P5.2c's row keys."""
    import offline.tier_statements as tst

    spec = tst.CellSpec(cell_id=cell_id, method=method, tier=tier, source=source, arm=arm, eval_file=eval_file,
                        manifest="none: a P9.1 run directory, digested in the artifact's inputs")
    return tst.Cell(spec=spec, rows=tuple({key: row[key] for key in STATEMENT_ROW_KEYS}
                                          for row in sorted(rows, key=lambda r: (r["seed"], r["draw_id"]))))


def new_cell_id(cell: cr.Cell) -> str:
    """``p9_1:<scenario>:<arm>@<tier>`` -- the scenario is part of the id, because ``<arm>@<tier>`` alone names two
    cells (``iql_t@mappo1000`` and ``behaviour@random_fresh`` exist on both scenarios)."""
    return f"p9_1:{cell.scenario}:{cell.key}"


def statement_cell(roots: cr.Roots, cell: cr.Cell) -> Any:
    """A NEW cell as a ``tier_statements.Cell``: id :func:`new_cell_id`, method the A27 arm."""
    return as_statement_cell(new_cell_id(cell), cell.arm, cell.tier, cell.key, "p9_1",
                             _relative(roots, cr.route_i_output(roots, cell)), new_cell_rows(roots, cell))


def new_training_runs(roots: cr.Roots, cell: cr.Cell, *, seeds: Any = None) -> list[dict[str, Any]]:
    """Every run of a new cell: ``seconds`` and ``gradient_steps`` from its write-once record, the uncontrolled-clock
    label (A27(a)) refused unless the record carries it."""
    from offline.dt_gate import TRAINING_SEEDS

    runs = []
    for seed in (TRAINING_SEEDS if seeds is None else seeds):
        path = cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, int(seed))
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("clock") != cr.CLOCK_LABEL or int(record["seed"]) != int(seed):
            raise ValueError(f"{path} is not a p9.1 run record of seed {seed} with A27(a)'s clock label")
        runs.append({"scenario": cell.scenario, "tier": cell.tier, "arm": cell.arm, "seed": int(seed),
                     "seconds": float(record["seconds"]), "gradient_steps": int(record["gradient_steps"]),
                     "clock": cr.CLOCK_LABEL, "source": _relative(roots, path)})
    return runs


# ----------------------------------------------------------------------
# Each run's code commit, from its session's token line (BRIEF_46 Amendment G, G.2)
# ----------------------------------------------------------------------

#: The driver's token line, printed once per session after every refusal has passed and the token is consumed
#: (``offline/campaigns/p9_1_corrective.sh``): the stage, its UTC stamp, the run tree, and the commit the driver
#: checked the run tree is clean and detached at.
TOKEN_LINE = re.compile(r"token consumed; stage (?P<stage>[a-z0-9]+) (?P<stamp>[0-9]{8}T[0-9]{6}Z) from "
                        r"(?P<run_tree>\S+) at (?P<commit>[0-9a-f]{7,40}); logs (?P<logs>\S+)")
#: The stages that train (the campaign stage, retired by E.4, trained runs 1-3).
TRAINING_STAGES: tuple[str, ...] = ("campaign", "train")
CODE_COMMIT_RULE = (
    "BRIEF_46 G.2: a P9.1 run's code commit is its session's -- the session whose window, from its token line's stamp "
    "to the next session's of any stage, holds the run's started_unix -- as that token line names it, beside the run "
    "tree the driver checked is clean and detached at that commit. S.2 (Q20): the evaluation's code commit is the one "
    "commit every evaluate session's token line names, and the D4RL re-run's is its d4rl session's. A checkpoint's "
    "runtime.git_commit, and a route (i) file's, is the main tree's HEAD when the file was written, never the code's "
    "commit, and is not reported as one.")


def driver_sessions(roots: cr.Roots) -> list[dict[str, Any]]:
    """Every session the driver started, in start order.

    There is one session per token line of every ``<stage>_capture.txt`` in the runs directory. Each carries the stage,
    its stamp and start (UTC, whole seconds), the run tree, the commit, and where the line is.

    The captures are what the author's commands ``tee`` the driver's output into. ``BRIEF_46`` G.2 names
    ``campaign_capture.txt`` for runs 1-3 and ``train_capture.txt`` for every training session. The other stages'
    captures are read too, so that each session's window ends where the next session of ANY stage starts.

    Refused: a line that begins as a token line but does not read as one, and two token lines with one stamp.
    """
    import calendar
    import time

    sessions: list[dict[str, Any]] = []
    for capture in sorted(Path(roots.runs_root).glob("*_capture.txt")):
        for number, line in enumerate(capture.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.startswith("token consumed;"):
                continue
            match = TOKEN_LINE.fullmatch(line.rstrip())
            if match is None:
                raise ValueError(f"{capture.name} line {number} is a token line the report cannot read: {line!r}")
            start = calendar.timegm(time.strptime(match["stamp"], "%Y%m%dT%H%M%SZ"))
            sessions.append({"stage": match["stage"], "stamp": match["stamp"], "start_unix": float(start),
                             "run_tree": match["run_tree"], "code_commit": match["commit"],
                             "capture": _relative(roots, capture), "line": number})
    sessions.sort(key=lambda session: session["start_unix"])
    stamps = [session["stamp"] for session in sessions]
    for stamp in stamps:
        if stamps.count(stamp) > 1:
            raise ValueError(f"two token lines carry the stamp {stamp}: a session has one token line")
    return sessions


def stage_code_commit(sessions: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Any]:
    """The one commit and run tree that every session of *stage* names, from their token lines (``BRIEF_46`` S.2, Q20).

    For ``evaluate`` this is the evaluation's code commit: every evaluation output (route (i), route (ii), P3a, the old
    cells) was written in an evaluate session. For ``d4rl`` it is the D4RL re-run's. The attribution is pure: it
    reads the captures' text only, never a file's time.

    Refused: no session of *stage* in the captures; or sessions that name more than one commit or run tree. A
    per-output attribution would then be needed, which the report does not attempt.
    """
    chosen = [session for session in sessions if session["stage"] == stage]
    if not chosen:
        raise ValueError(f"no {stage} session in the captures")
    commits = sorted({session["code_commit"] for session in chosen})
    trees = sorted({session["run_tree"] for session in chosen})
    if len(commits) != 1 or len(trees) != 1:
        raise ValueError(f"the {stage} sessions name the commits {commits} from the run trees {trees}: one commit and "
                         "one run tree, or a per-output attribution, which the report does not attempt")
    return {"stage": stage, "code_commit": commits[0], "run_tree": trees[0],
            "sessions": [f"{session['stage']} {session['stamp']}" for session in chosen],
            "token_lines": [f"{session['capture']} line {session['line']}" for session in chosen]}


def run_session(run: Mapping[str, Any], sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """*run*'s code commit (``BRIEF_46`` G.2), with the run tree and where the token line is.

    The session is the one whose window holds the run's ``started_unix``. A window runs from the session's start to
    the next session's start, of any stage.

    Refused:
    - a run that started before every session;
    - a run that started in a session that trains nothing (collect, d4rl, evaluate);
    - a run that finished after the next session started. Two P9.1 stages never run at once (the driver's pgrep
      refusal), so such a run is not a run of the session it started in.
    """
    where = f"{run.get('cell')} seed {run.get('seed')}"
    started, finished = float(run["started_unix"]), float(run["finished_unix"])
    index = max((i for i, session in enumerate(sessions) if float(session["start_unix"]) <= started), default=None)
    if index is None:
        raise ValueError(f"{where} started before every session the captures name")
    session = sessions[index]
    if session["stage"] not in TRAINING_STAGES:
        raise ValueError(f"{where} started inside the {session['stage']} session {session['stamp']}, which trains "
                         "nothing")
    if index + 1 < len(sessions) and finished > float(sessions[index + 1]["start_unix"]):
        following = sessions[index + 1]
        raise ValueError(f"{where} finished after the next session, {following['stage']} {following['stamp']}, "
                         "started")
    return {"code_commit": session["code_commit"], "run_tree": session["run_tree"],
            "session": f"{session['stage']} {session['stamp']}",
            "code_commit_source": f"{session['capture']} line {session['line']}: the session's token line "
                                  "(BRIEF_46 G.2)"}


# ----------------------------------------------------------------------
# The artifacts (BRIEF_46 3.5): pure functions of a Sources
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Sources:
    """Everything the five artifacts are built from, loaded once: the cells (``tier_statements.Cell``) by id -- new
    (``p9_1:<scenario>:<arm>@<tier>``), hz1x1 old (``p4_4:<arm>``, ``p4_6:<arm>``, ``p4_7:<arm>``), grid old (P5.2c's
    ids) -- with their per-episode records and training runs, the declarations, P5.2c's books, P3a's rows, the tiers'
    training schedules, the D4RL re-run, registered constants, the inputs by digest and the git block; and the IQL
    diagnostics beside the D4RL section (Q18, Q19), the captures' sessions, the evaluation's code commit and the D4RL
    session (Q20), which :func:`load_sources` always fills (their defaults only where a test builds a Sources by
    hand)."""

    cells: Mapping[str, Any]
    records: Mapping[str, Sequence[Mapping[str, Any]]]
    runs: Mapping[str, Sequence[Mapping[str, Any]]]
    declarations: Mapping[str, Any]
    grid_books: Mapping[str, Mapping[tuple[str, str], Any]]
    p3a: Mapping[str, Sequence[Mapping[str, Any]]]
    schedules: Mapping[str, Mapping[str, Any]]
    d4rl: Mapping[str, Any] | None
    registered: Mapping[str, Any]
    inputs: Sequence[Mapping[str, Any]]
    git: Mapping[str, Any]
    diagnostics: Mapping[str, Any] | None = None
    sessions: Sequence[Mapping[str, Any]] = ()
    evaluation_code_commit: Mapping[str, Any] | None = None
    d4rl_session: Mapping[str, Any] | None = None


REGISTRATION = "PREREGISTRATION.md A27 (tag v2.8-prereg-a27)"


def _referenced(node: Any) -> list[str]:
    """Every cell id an artifact block names (``left`` / ``right`` / ``inputs`` / ``original_inputs`` / ``cell``)."""
    found: list[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key in ("left", "right", "cell") and isinstance(value, str):
                found.append(value)
            elif key in ("inputs", "original_inputs") and isinstance(value, list):
                found.extend(v for v in value if isinstance(v, str))
            else:
                found.extend(_referenced(value))
    elif isinstance(node, (list, tuple)):
        for item in node:
            found.extend(_referenced(item))
    return found


def _finish(point: str, src: Sources, body: Mapping[str, Any], *, role: str,
            what_this_is_not: Sequence[str]) -> dict[str, Any]:
    """The common block, then the per-episode records and the training runs of EVERY cell the body names (A27(a): the
    records of every new cell and every old cell it is compared with)."""
    ids = sorted({cid for cid in _referenced(body) if cid in src.cells})
    missing = [cid for cid in ids if cid not in src.records]
    if missing:
        raise ValueError(f"{point}: no per-episode records for {missing[:3]}; every compared cell carries them")
    return {
        "format_version": REPORT_FORMAT_VERSION, "label": cr.LABELS[point], "registered_in": f"{point}, {REGISTRATION}",
        "role": role, "git": dict(src.git), "inputs": [dict(entry) for entry in src.inputs], **body,
        "code_commit_rule": CODE_COMMIT_RULE, "sessions": [dict(session) for session in src.sessions],
        "evaluation_code_commit": src.evaluation_code_commit,
        "records": {cid: [{key: row[key] for key in EPISODE_FIELDS} for row in src.records[cid]] for cid in ids},
        "training_runs": {cid: [dict(run) for run in src.runs[cid]] for cid in ids if cid in src.runs},
        "clock": cr.CLOCK_LABEL, "what_this_is_not": list(what_this_is_not),
    }


def _levels(cells: Mapping[str, Any]) -> dict[str, Any]:
    """Each cell's level under both definitions: the mean over its episodes by ``dt_gate.mean_ci95`` (P5.2c's level)."""
    from offline.dt_gate import mean_ci95
    import offline.tier_statements as tst

    return {name: {"cell": cell.spec.cell_id, **{definition: float(mean_ci95(
        [e.att_horizon for e in tst.episodes_of(cell, definition)]).mean) for definition in ("att_engine", "att_ours")}}
        for name, cell in cells.items()}


MIXTURE_TIER = "mix_m1000_m500"
#: The parents' registered cells (A27(e) quotes their order: P4.4's mappo1000 cells, which P4.6 re-used, and P4.6's
#: mappo500 cells).
MIXTURE_PARENTS: Mapping[str, Mapping[str, str]] = {
    "mappo1000": {"dt": "p4_4:madt", "bc": "p4_4:bc", "bc_top10": "p4_4:bc_top10", "iql": "p4_4:iql"},
    "mappo500": {"dt": "p4_6:dt@mappo500", "bc": "p4_6:bc@mappo500", "bc_top10": "p4_6:bc_top10@mappo500",
                 "iql": "p4_6:iql@mappo500"},
}


def build_mixture(src: Sources) -> dict[str, Any]:
    """``p9_1_mixture.json`` (A27(e)): the declaration, the four contrasts, the 5-s window, ``iql_t`` beside, the
    parents' levels beside."""
    import offline.corrective_verdicts as cv

    cells = {arm: src.cells[f"p9_1:hz1x1:{arm}@{MIXTURE_TIER}"] for arm in ("dt", "bc", "bc_top10", "iql", "iql_t")}
    body = {
        "declaration": src.declarations[MIXTURE_TIER],
        "predictions": {"orderings": cv.mixture_orderings(cells, tier=MIXTURE_TIER),
                        "within_5_s": cv.within_seconds(cells, tier=MIXTURE_TIER)},
        "beside": {"iql_t_within_5_s": cv.within_seconds(cells, tier=MIXTURE_TIER, arms=("iql_t",))},
        "levels": _levels(cells),
        "parents": {tier: _levels({arm: src.cells[cid] for arm, cid in arms.items()})
                    for tier, arms in MIXTURE_PARENTS.items()},
    }
    return _finish("A27(e)", src, body, role="A27(e): hz1x1's overlap mixture mappo1000 + mappo500 at 0.50, five arms",
                   what_this_is_not=(
                       "not a test of the filter on the original mixtures: their random halves replay two schedules "
                       "(A27(d)); mix_m1000_m500 replaces them for the selection question",
                       "not a new confirmatory family: the four contrasts and the window are tests registered after "
                       "results (A27(e)), judged by P5.2c's rule",
                       "iql_t is reported beside, not predicted"))


#: A27(g)'s registered 40,000-step cells per tier (P4.4's mappo1000 cells; P4.7's mix50 cells).
BUDGET_40K: Mapping[str, Mapping[str, str]] = {
    "mappo1000": MIXTURE_PARENTS["mappo1000"],
    "mix50": {"dt": "p4_7:dt@mix50", "bc": "p4_7:bc@mix50", "bc_top10": "p4_7:bc_top10@mix50", "iql": "p4_7:iql@mix50"},
}
BUDGET_SCOPE: Mapping[str, str] = {
    "mappo1000": "A8(b)'s secondary (the registered ceiling of 100,000 steps)",
    "mix50": "an exploratory extension beyond A8(b)",
}


def build_budget(src: Sources) -> dict[str, Any]:
    """``p9_1_budget.json`` (A27(g)): per tier the paired 100k - 40k differences and the concordance; the plateau
    diagnostic at 100,000; the counts the paper quotes."""
    import offline.corrective_verdicts as cv

    tiers = {}
    for tier, old in BUDGET_40K.items():
        at_40k = {arm: src.cells[cid] for arm, cid in old.items()}
        at_100k = {arm: src.cells[f"p9_1:hz1x1:{arm}_100k@{tier}"] for arm in cv.BUDGET_ARMS}
        tiers[tier] = {**cv.budget_block(at_40k, at_100k, tier=tier), "scope": BUDGET_SCOPE[tier]}
    plateau = [{"cell": cid, "seed": int(run["seed"]), "plateaued": run["plateaued"],
                "window_means": list(run["window_means"])}
               for cid in sorted(c for c in src.cells if c.startswith("p9_1:hz1x1:") and "_100k@" in c)
               for run in src.runs[cid]]
    body = {"tiers": tiers, "plateau_at_100k": plateau,
            "paper_counts_at_40k": dict(src.registered["plateau_counts"]) if "plateau_counts" in src.registered else None}
    return _finish("A27(g)", src, body, role="A27(g): A8(b)'s budget-sensitivity secondary at 100,000 steps",
                   what_this_is_not=(
                       "not a new primary: the 40,000-step cells stay the registered primary, the 100,000-step cells "
                       "are A8(b)'s secondary beside them",
                       "mix50 goes beyond A8(b) and is an exploratory extension",
                       "iql_t is not run at 100,000 steps (a fifth arm would be a new design, not A8(b)'s)"))


def plateau_counts(repo_root: str | Path) -> dict[str, int]:
    """A27(g)'s counts over the committed training records, by the author's command of 2026-10-15: runs; not
    plateaued; of those, still falling over the last three windows, rose in the last window, fluctuated."""
    runs = [run for name, key in (("p4_training", "seeds"), ("p4_4_training", "runs"), ("p4_6_training", "runs"),
                                  ("p4_7_training", "runs"))
            for run in json.loads((Path(repo_root) / "docs" / "data" / f"{name}.json").read_text(encoding="utf-8"))[key]]
    curves = [run["window_means"] for run in runs if not run["plateaued"]]
    falling = sum(all(a > b for a, b in zip(w[-3:], w[-2:])) for w in curves)
    rose = sum((not all(a > b for a, b in zip(w[-3:], w[-2:]))) and w[-1] > w[-2] for w in curves)
    return {"runs": len(runs), "not_plateaued": len(curves), "still_falling": falling,
            "rose_in_the_last_window": rose, "fluctuated": len(curves) - falling - rose}


def artifact_text(payload: Mapping[str, Any]) -> str:
    """The one serialisation of an artifact: sorted keys, no rounding, no timestamp, a trailing newline; refused from a
    dirty tree and on a non-finite number."""
    if payload.get("git", {}).get("dirty") is not False:
        raise ValueError("the artifact's git block is not clean (dirty must be false): a report is built from a clean "
                         "committed tree")
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """:func:`artifact_text`, written -- every refusal before the file is opened."""
    text = artifact_text(payload)
    target = Path(path)
    target.write_text(text, encoding="utf-8")
    return target


def _exact(node: Any) -> Any:
    """A ``Fraction`` as ``{"exact": "n/d", "value": float}``; everything else unchanged (JSON carries no fraction)."""
    from fractions import Fraction

    if isinstance(node, Fraction):
        return {"exact": f"{node.numerator}/{node.denominator}", "value": float(node)}
    if isinstance(node, Mapping):
        return {str(key): _exact(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_exact(item) for item in node]
    return node


GRID_TIERS: tuple[str, ...] = ("mappo1000", "maxpressure", "fixedtime", "random")
#: A27(b)'s matched arm for each registered grid baseline.
A27B_MATCHED: Mapping[str, str] = {"bc": "bc_m", "bc_top10": "bc_top10_m", "bc_top10_perix": "bc_top10_perix_m",
                                   "iql": "iql_m"}
#: A27(b)'s list of P5.2 statements that contrast a DT with a baseline ("Q1's cells, Q2a/Q2b/Q2b-hard, Q3a/Q3b/Q3c,
#: Q5a/Q5b/Q5c"), with their per-tier and aggregate entries; the leaders are re-read beside (descriptive).
A27B_STATEMENT_IDS: tuple[str, ...] = (
    "Q1", "Q1b", "Q1b.both", "Q2a", "Q2b", "Q2b.aggregate", "Q2b_hard", "Q3a", "Q3b", "Q3b.aggregate", "Q3c", "Q5a",
    "Q5a.aggregate", "Q5b", "Q5b.aggregate", "Q5c", "Q5c.aggregate",
)
MOVED_RULE = ("A27(b): a verdict MOVES iff the registered rule's decision changes -- each statement is re-scored by P5.2c's "
              "own scorer (tier_statements.all_statements) on the registered book (P5.2b's random-tier IQL cell) with "
              "the new cell in place of its original, under both definitions; the original decision is beside it")


def _rescore_block(books: Mapping[str, Any], substitution: Mapping[tuple[str, str], Any], *,
                   counted: Sequence[str] | None) -> dict[str, Any]:
    """P5.2c's statements re-scored with *substitution*; refused if a statement that reads no replaced cell moves (the
    substitution may only reach the statements that read it)."""
    import offline.corrective_verdicts as cv
    import offline.tier_statements as tst

    moved = cv.moved_block(tst.all_statements(books), cv.rescore_p5_2(books, substitution))
    replaced = {books["corrected"][key].spec.cell_id for key in substitution}
    stray = [e for e in moved if not set(e["original_inputs"]) & replaced and (e["moved"] or e["reading_changed"])]
    if stray:
        raise ValueError(f"{stray[0]['id']} at {stray[0]['tier']} moved without reading a replaced cell")
    pool = [e for e in moved if counted is None or e["id"] in counted]
    verdicts = [e for e in pool if e["kind"] == "verdict"]
    return {"statements": pool, "rule": MOVED_RULE,
            "substitution": {f"{m}@{t}": {"original": books["corrected"][(m, t)].spec.cell_id,
                                          "replacement": cell.spec.cell_id} for (m, t), cell in substitution.items()},
            "n_statements": {d: sum(e["definition"] == d for e in verdicts) for d in ("att_engine", "att_ours")},
            "n_moved": {d: sum(e["moved"] for e in verdicts if e["definition"] == d) for d in ("att_engine", "att_ours")}}


def _leaders(books: Mapping[str, Any], substitution: Mapping[tuple[str, str], Any]) -> list[dict[str, Any]]:
    """P5.2c's descriptive leaders per tier and definition, on the registered book and on the substituted one."""
    import offline.tier_statements as tst

    book = {**books["corrected"], **substitution}
    before = {(b["tier"], b["definition"]): b for b in tst.leaders_block(books) if b["version"] in ("both", "corrected")}
    after = {(b["tier"], b["definition"]): b for b in tst.leaders_block({"corrected": book, "as_committed": book})}
    return [{"tier": tier, "definition": definition, "registered": False,
             "original_reading": before[(tier, definition)]["reading"], "reading": after[(tier, definition)]["reading"],
             "original_leaders": before[(tier, definition)]["leaders"], "leaders": after[(tier, definition)]["leaders"],
             "reading_changed": before[(tier, definition)]["reading"] != after[(tier, definition)]["reading"],
             "original_inputs": before[(tier, definition)]["inputs"], "inputs": after[(tier, definition)]["inputs"]}
            for tier, definition in sorted(before)]


def build_exposure_parity(src: Sources) -> dict[str, Any]:
    """``p9_1_exposure_parity.json`` (A27(b)): the exposure table from the constants, every A27(b) statement re-scored
    with the matched arms, the count that moves, the leaders beside, the sentence's two contrasts."""
    import offline.a27_arms as a27_arms
    import offline.corrective_verdicts as cv

    books = src.grid_books
    substitution = {(m, t): src.cells[f"p9_1:grid4x4:{A27B_MATCHED[m]}@{t}"] for m, t in cv.A27B_KEYS}
    dt = books["corrected"][("dt_nomix", "mappo1000")]
    body = {
        "exposure_table": _exact(a27_arms.exposure_table()),
        "rescore": _rescore_block(books, substitution, counted=A27B_STATEMENT_IDS),
        "counted_statement_ids": list(A27B_STATEMENT_IDS),
        "leaders": _leaders(books, substitution),
        "sentence_contrasts": {
            "matched": [cv.contrast_entry(dt, substitution[(m, "mappo1000")], tier="mappo1000") for m in ("bc", "iql")],
            "original": [cv.contrast_entry(dt, books["corrected"][(m, "mappo1000")], tier="mappo1000")
                         for m in ("bc", "iql")]},
        "confound": ("per-intersection return scaling, which parity does not remove: the DT's return prompt and scale are "
                     "set per intersection (sixteen per-node target_rtg / rtg_scale pairs, p5_2_declaration_*.json "
                     "prompts), IQL's reward scale is one number over all intersections; the spatial DT's joint context "
                     "is a second difference with no baseline counterpart (A27(b))"),
    }
    return _finish("A27(b)", src, body, role="A27(b): grid4x4 exposure parity, the matched baselines beside the originals",
                   what_this_is_not=(
                       "not a re-training of the DTs: they keep their registered checkpoints",
                       "not exposure parity on hz1x1: its only asymmetry is the top-decile arms', which runs the other "
                       "way and is not re-run here",
                       "not a removal of the return-scaling confound in any DT-versus-IQL contrast"))


#: The registered hz1x1 IQL cell of each tier (P4.4's mappo1000 cell, P4.6's four, P4.7's three mixtures).
HZ1X1_IQL_OLD: Mapping[str, str] = {
    "mappo1000": "p4_4:iql", "maxpressure": "p4_6:iql@maxpressure", "fixedtime": "p4_6:iql@fixedtime",
    "random": "p4_6:iql@random", "mappo500": "p4_6:iql@mappo500", "mix33": "p4_7:iql@mix33", "mix50": "p4_7:iql@mix50",
    "mix67": "p4_7:iql@mix67",
}

#: A27(c)'s eight hz1x1 tiers, in the registration's order.
HZ1X1_IQL_TIERS: tuple[str, ...] = ("random", "fixedtime", "maxpressure", "mappo500", "mappo1000", "mix33", "mix50",
                                    "mix67")


#: Q8 (``BRIEF_46`` Amendment B, B.2): the tiers each artifact's P1 ranks BC on -- P4.6's five, P4.7's eight.
P1_TIERS: Mapping[str, tuple[str, ...]] = {
    "p4_6": ("mappo1000", "mappo500", "maxpressure", "fixedtime", "random"),
    "p4_7": ("mappo1000", "mappo500", "maxpressure", "fixedtime", "random", "mix33", "mix50", "mix67"),
}
#: The mixtures whose behaviour reference is CONSTRUCTED (P4.7), not rolled out.
MIXTURE_REFERENCE_TIERS: tuple[str, ...] = ("mix33", "mix50", "mix67")


def p1_cell_ids(tier: str) -> dict[str, str]:
    """The old cells P1 reads on *tier*: P4.4's on ``mappo1000`` (P4.6 and P4.7 re-used them), P4.7's on a mixture,
    P4.6's on the other phase-1 tiers (P4.7's phase-1 cells are P4.6's, value for value)."""
    if tier == "mappo1000":
        return dict(MIXTURE_PARENTS["mappo1000"])
    task = "p4_7" if tier in MIXTURE_REFERENCE_TIERS else "p4_6"
    return {method: f"{task}:{method}@{tier}" for method in ("dt", "bc", "bc_top10", "iql")}


def constructed_id(tier: str) -> str:
    """A constructed mixture reference's cell id."""
    return f"constructed:behaviour@{tier}"


def iql_behaviour_pairs() -> list[dict[str, Any]]:
    """Q8's behaviour comparisons with IQL, each as its artifact states it: P4.4's ``mappo1000_vs_iql`` (behaviour
    LEFT, ``to_json_obj``), P4.6's ``iql@T`` against ``behaviour@T`` on its five tiers (IQL left, ``comparison_json``;
    on mappo1000 P4.4's IQL cell under P4.6's name), P4.7's three mixtures against their constructed references.
    P4.7's five phase-1 comparisons are P4.6's, value for value, and are listed once."""
    pairs = [{"artifact": "p4_4", "statement": "mappo1000_vs_iql", "tier": "mappo1000", "iql": "p4_4:iql",
              "behaviour": "p4_4:mappo1000", "behaviour_left": True, "serializer": "offline_baselines",
              "labels": None}]
    for task, tiers in (("p4_6", P1_TIERS["p4_6"]), ("p4_7", MIXTURE_REFERENCE_TIERS)):
        for tier in tiers:
            pairs.append({"artifact": task, "statement": f"iql@{tier} against behaviour@{tier}", "tier": tier,
                          "iql": p1_cell_ids(tier)["iql"],
                          "behaviour": constructed_id(tier) if task == "p4_7" else f"p4_6:behaviour@{tier}",
                          "behaviour_left": False, "serializer": "method_tier_grid",
                          "labels": (f"iql@{tier}", f"behaviour@{tier}")})
    return pairs


def _side_of_zero(low: float, high: float) -> str:
    return "below zero" if high < 0.0 else "above zero" if low > 0.0 else "includes zero"


def p1_rescore_block(src: Sources, name: str) -> dict[str, Any]:
    """Q8: *name*'s P1 (``p4_6`` or ``p4_7``) on the old cells, refused unless its ``att_ours`` score IS the committed
    P1, beside the same with ``iql_t`` in IQL's slot -- both definitions, the registered scorer."""
    import offline.corrective_verdicts as cv

    tiers = P1_TIERS[name]
    original = {tier: {m: src.cells[cid] for m, cid in p1_cell_ids(tier).items()} for tier in tiers}
    rescored = {tier: {**original[tier], "iql": src.cells[f"p9_1:hz1x1:iql_t@{tier}"]} for tier in tiers}
    block = cv.p1_rescore(original, rescored)
    if block["original"]["att_ours"] != src.registered["p1_committed"][name]:
        raise ValueError(f"{name}: P1 on the joined old cells under att_ours is not the committed P1; the cells read are "
                         "not the ones the registered statement scored")
    return {"artifact": name, "tiers": list(tiers), **block,
            "original_inputs": sorted(c.spec.cell_id for methods in original.values() for c in methods.values()),
            "inputs": sorted(c.spec.cell_id for methods in rescored.values() for c in methods.values()),
            "original_reproduces_committed_att_ours": True}


def iql_behaviour_rescore(src: Sources) -> dict[str, Any]:
    """Q8: every behaviour comparison with IQL by its registered route, each original refused unless its ``att_ours``
    payload IS the committed one, beside the same with ``iql_t`` in IQL's place; both definitions.  No decision rule
    is attached to these comparisons, so the side of zero of each interval is reported, not a verdict."""
    import offline.corrective_verdicts as cv

    committed = src.registered["iql_behaviour_committed"]
    entries = []
    for pair in iql_behaviour_pairs():
        new = f"p9_1:hz1x1:iql_t@{pair['tier']}"

        def ordered(iql: str, pair: Mapping[str, Any] = pair) -> list[str]:
            return [pair["behaviour"], iql] if pair["behaviour_left"] else [iql, pair["behaviour"]]

        original = cv.paired_both(*(src.cells[c] for c in ordered(pair["iql"])), serializer=pair["serializer"],
                                  labels=pair["labels"])
        key = f"{pair['artifact']}:{pair['statement']}"
        if original["att_ours"] != committed[key]:
            raise ValueError(f"{key}: the comparison on the joined old cells under att_ours is not the committed "
                             "comparison; the cells or the order read are not the registered statement's")
        rescored = cv.paired_both(*(src.cells[c] for c in ordered(new)), serializer=pair["serializer"])
        sides = {version: {d: _side_of_zero(block[d]["ci95_low"], block[d]["ci95_high"]) for d in cv.DEFINITIONS}
                 for version, block in (("original", original), ("rescored", rescored))}
        entries.append({"artifact": pair["artifact"], "statement": pair["statement"], "tier": pair["tier"],
                        "original_inputs": ordered(pair["iql"]), "inputs": ordered(new), "original": original,
                        "rescored": rescored, "side_of_zero": sides,
                        "side_changed": {d: sides["original"][d] != sides["rescored"][d] for d in cv.DEFINITIONS},
                        "original_reproduces_committed_att_ours": True})
    return {"comparisons": entries,
            "route": "offline_baselines.paired_comparison (per-draw means over the seeds, the shared draws), "
                     "serialised as each artifact serialised it; no decision rule is attached (the artifacts' "
                     "multiplicity field), so the side of zero is reported, not a verdict",
            "p4_7_phase1": "P4.7's five phase-1 comparisons with IQL are P4.6's, value for value (P4.7's gate P1), "
                           "listed once under p4_6"}


def build_iql_targets(src: Sources) -> dict[str, Any]:
    """``p9_1_iql_targets.json`` (A27(c)): ``iql_t - iql`` on twelve tiers, ``iql_m - iql_t`` on the grid (the two
    effects separable), P4.4's A6 statement, the IQL lead on hz1x1's random tier, P5.2's IQL statements, D4RL; and
    (Q8, ``BRIEF_46`` Amendment B, B.2) P4.6's and P4.7's P1 and every behaviour comparison with IQL, re-scored."""
    import offline.corrective_verdicts as cv

    books = src.grid_books
    hz = cv.paired_differences({t: (src.cells[f"p9_1:hz1x1:iql_t@{t}"], src.cells[HZ1X1_IQL_OLD[t]])
                                for t in HZ1X1_IQL_TIERS}, left="iql_t", right="iql")
    grid = cv.paired_differences({t: (src.cells[f"p9_1:grid4x4:iql_t@{t}"], books["corrected"][("iql", t)])
                                  for t in GRID_TIERS}, left="iql_t", right="iql")
    twelve = {}
    for definition in ("att_engine", "att_ours"):
        entries = [e["by_definition"][definition] for e in hz["tiers"] + grid["tiers"]]
        points = [e["mean_difference"] for e in entries]
        twelve[definition] = {"n_tiers": len(entries), "n_excluding_zero": sum(e["excludes_zero"] for e in entries),
                              "min": min(points), "max": max(points)}
    substitution = {("iql", t): src.cells[f"p9_1:grid4x4:iql_t@{t}"] for t in GRID_TIERS}
    rescore = _rescore_block(books, substitution, counted=None)
    new_ids = {cell.spec.cell_id for cell in substitution.values()}
    rescore["statements"] = [e for e in rescore["statements"] if set(e["inputs"]) & new_ids]
    rescore["n_statements"] = {d: sum(e["kind"] == "verdict" and e["definition"] == d for e in rescore["statements"])
                               for d in ("att_engine", "att_ours")}
    rescore["n_moved"] = {d: sum(e["moved"] for e in rescore["statements"] if e["definition"] == d)
                          for d in ("att_engine", "att_ours")}
    random_dt = src.cells["p4_6:dt@random"]
    body = {
        "hz1x1": hz, "grid4x4": grid, "twelve_tiers": twelve,
        "grid4x4_original_random_beside": cv.contrast_entry(src.cells["p9_1:grid4x4:iql_t@random"],
                                                            books["as_committed"][("iql", "random")], tier="random"),
        "grid4x4_exposure_against_dropout": cv.paired_differences(
            {t: (src.cells[f"p9_1:grid4x4:iql_m@{t}"], src.cells[f"p9_1:grid4x4:iql_t@{t}"]) for t in GRID_TIERS},
            left="iql_m", right="iql_t"),
        "p4_4_a6": cv.a6_rescore(src.cells["p4_4:madt"], src.cells["p4_4:iql"], src.cells["p9_1:hz1x1:iql_t@mappo1000"],
                                 delta=float(src.registered["p4_4_delta"])),
        "hz1x1_iql_lead_on_random": [cv.contrast_entry(src.cells["p4_6:iql@random"], random_dt, tier="random"),
                                     cv.contrast_entry(src.cells["p9_1:hz1x1:iql_t@random"], random_dt, tier="random")],
        "p5_2_rescore": rescore,
        "p1_rescore": {name: p1_rescore_block(src, name) for name in P1_TIERS},
        "iql_behaviour_comparisons": iql_behaviour_rescore(src),
        "d4rl": src.d4rl,
        "d4rl_session": src.d4rl_session,
        "iql_diagnostics": src.diagnostics,
    }
    return _finish("A27(c)", src, body, role="A27(c): IQL's three target forwards in eval(), beside the registered IQL",
                   what_this_is_not=(
                       "not a change to IQL's hyper-parameters: tau, beta, gamma, Polyak, clip, batch, steps, warm-up "
                       "and seeds are the registered ones; only the three target forwards' module mode differs",
                       "not the published IQL update order: the advantage still uses V from before its update",
                       "P4.4's registered statements in which IQL appears are A6 (madt_vs_iql) and the behaviour "
                       "comparison mappo1000_vs_iql (Q8); the registration's wording 'BC-vs-IQL contrast' names no "
                       "P4.4 contrast",
                       "not a verdict on any behaviour comparison: none carries a decision rule"))


HZ1X1_FRESH_ARMS: tuple[str, ...] = ("dt", "bc", "bc_top10", "iql", "iql_t")
GRID_FRESH_ARMS: tuple[str, ...] = ("dt_nomix", "bc_m", "bc_top10_m", "bc_top10_perix_m", "iql_m")


def build_random_fresh(src: Sources) -> dict[str, Any]:
    """``p9_1_random_fresh.json`` (A27(d)): the declarations, every collapse check against both random references on
    both scenarios (the new tier and the original one), Q6 against both anchors, P3a with its beside-table, P3b."""
    import offline.corrective_verdicts as cv
    import offline.tier_statements as tst
    import offline.tier_sweep as ts

    books = src.grid_books
    hz_refs = {"registered_five_schedule": src.cells["p4_6:behaviour@random"],
               "random_fresh": src.cells["p9_1:hz1x1:behaviour@random_fresh"]}
    grid_refs = {"registered_five_schedule": books["corrected"][tst.ANCHOR],
                 "random_fresh": src.cells["p9_1:grid4x4:behaviour@random_fresh"]}
    collapse = {
        "hz1x1": {
            "random_fresh_tier": cv.collapse_rows({a: src.cells[f"p9_1:hz1x1:{a}@random_fresh"] for a in HZ1X1_FRESH_ARMS},
                                                  hz_refs, tier="random_fresh"),
            "original_random_tier": cv.collapse_rows({a: src.cells[f"p4_6:{a}@random"] for a in ("dt", "bc", "bc_top10",
                                                                                                "iql")},
                                                     hz_refs, tier="random")},
        "grid4x4": {
            "random_fresh_tier": cv.collapse_rows({a: src.cells[f"p9_1:grid4x4:{a}@random_fresh"]
                                                   for a in GRID_FRESH_ARMS}, grid_refs, tier="random_fresh"),
            "original_random_tier": cv.collapse_rows({m: books["corrected"][(m, "random")] for m in ts.METHODS},
                                                     grid_refs, tier="random")},
    }
    fresh_book = {**books["corrected"], tst.ANCHOR: grid_refs["random_fresh"]}
    q6 = {"registered_five_schedule": [e for d in ("att_engine", "att_ours")
                                       for e in tst.score_q6(books["corrected"], definition=d)],
          "random_fresh": [e for d in ("att_engine", "att_ours") for e in tst.score_q6(fresh_book, definition=d)]}
    hz_dt, grid_dt = "p9_1:hz1x1:dt@random_fresh", "p9_1:grid4x4:dt_nomix@random_fresh"
    p3a = {"hz1x1": {**cv.p3a(src.p3a[hz_dt], per_intersection=False), "cell": hz_dt},
           "grid4x4": {**cv.p3a(src.p3a[grid_dt], per_intersection=True), "cell": grid_dt}}
    beside = {
        "hz1x1": cv.p3a_beside(src.p3a[hz_dt], training=src.schedules["hz1x1"], references={
            "registered_five_schedule": src.p3a["p4_6:behaviour@random"],
            "random_fresh": src.p3a["p9_1:hz1x1:behaviour@random_fresh"]}),
        "grid4x4": cv.p3a_beside(src.p3a[grid_dt], training=src.schedules["grid4x4"], per_intersection=True, references={
            "registered_five_schedule": src.p3a[tst.ANCHOR_CELL],
            "random_fresh": src.p3a["p9_1:grid4x4:behaviour@random_fresh"]}),
        "hz1x1_original_tier_p5_3b": src.registered.get("p5_3b_old_tier"),
    }
    p3b = [
        {**cv.contrast_entry(src.cells["p9_1:hz1x1:iql@random_fresh"], src.cells[hz_dt], tier="random_fresh"),
         "scenario": "hz1x1"},
        {**cv.contrast_entry(src.cells["p9_1:hz1x1:iql_t@random_fresh"], src.cells[hz_dt], tier="random_fresh"),
         "scenario": "hz1x1"},
        {**cv.contrast_entry(src.cells["p4_6:iql@random"], src.cells["p4_6:dt@random"], tier="random"),
         "scenario": "hz1x1"},
        {**cv.contrast_entry(src.cells["p9_1:grid4x4:iql_m@random_fresh"], src.cells[grid_dt], tier="random_fresh"),
         "scenario": "grid4x4"},
        {**cv.contrast_entry(books["corrected"][("iql", "random")], books["corrected"][("dt_nomix", "random")],
                             tier="random"), "scenario": "grid4x4"},
    ]
    body = {
        "declarations": {"hz1x1": src.declarations["random_fresh_hz1x1"],
                         "grid4x4": src.declarations["random_fresh_grid4x4"]},
        "collapse": collapse, "q6_both_anchors": q6, "p3a": p3a, "p3a_beside": beside,
        "p3b": {"contrasts": p3b, "prediction": None,
                "note": "A27(d)'s P3b: no directional expectation for the IQL lead on random; re-measured beside the "
                        "original tier's (on the grid IQL also has matched exposure and dropout-free targets)"},
        "normalisation_basis": "the new tiers fit the state normalisation, the return target and scale and IQL's "
                               "reward scale on their own 200 training episodes; the original random tier fitted them "
                               "on its 400-episode split, so the before/after comparison changes the generator AND the "
                               "normalisation basis (A27(d))",
    }
    return _finish("A27(d)", src, body, role="A27(d): a true random tier (200 distinct schedules) on both scenarios",
                   what_this_is_not=(
                       "not a replacement of the original random tiers: they stay the record, their defect named",
                       "not a test of the mixtures' selection question (A27(e) replaces them for it)",
                       "the beside-table carries no prediction; P3a alone is predicted"))


# ----------------------------------------------------------------------
# Loading the Sources from disk (run directories, committed records, corpus, the D4RL re-run)
# ----------------------------------------------------------------------


def hz1x1_old_id(artifact: str, arm: str) -> str:
    """An old hz1x1 cell's id: its artifact's task prefix and its committed arm (``p4_6:iql@maxpressure``)."""
    return f"{'_'.join(Path(artifact).stem.split('_')[:2])}:{arm}"


#: Where each old hz1x1 cell's runs are committed: P4's DT runs (``madt``) in ``p4_training.json`` ``seeds``; the
#: baselines in their task's ``*_training.json`` ``runs``.
HZ1X1_TRAINING_RECORDS: Mapping[str, str] = {"p4_4": "docs/data/p4_4_training.json",
                                             "p4_6": "docs/data/p4_6_training.json",
                                             "p4_7": "docs/data/p4_7_training.json"}
P4_DT_RECORD = "docs/data/p4_training.json"


def _old_run(cell_id: str, run: Mapping[str, Any], source: str) -> dict[str, Any]:
    return {"cell": cell_id, "seed": int(run["seed"]), "seconds": float(run["seconds"]),
            "gradient_steps": int(run["gradient_steps"]), "clock": cr.CLOCK_LABEL, "source": source,
            "window_means": [float(v) for v in run["window_means"]], "plateaued": bool(run["plateaued"])}


def hz1x1_old_runs(repo_root: str | Path, artifact: str, arm: str) -> list[dict[str, Any]]:
    """An old hz1x1 cell's five runs, copied from the committed training record that holds them (A27(a)); a behaviour
    cell has none."""
    from offline import att_rederivation as ar

    cell_id = hz1x1_old_id(artifact, arm)
    task = cell_id.split(":")[0]
    method, tier = ar.normalise_arm("hz1x1", arm)
    if method == "behaviour":
        return []
    if task == "p4_4" and arm == "madt":
        source = P4_DT_RECORD
        runs = json.loads((Path(repo_root) / source).read_text(encoding="utf-8"))["seeds"]
    else:
        source = HZ1X1_TRAINING_RECORDS[task]
        record = json.loads((Path(repo_root) / source).read_text(encoding="utf-8"))["runs"]
        runs = [r for r in record if r["method"] == method and (task == "p4_4" or r["tier"] == tier)]
    if sorted(int(r["seed"]) for r in runs) != [101, 202, 303, 404, 505]:
        raise ValueError(f"{source} does not hold exactly the five runs of {cell_id}")
    return sorted((_old_run(cell_id, run, source) for run in runs), key=lambda r: r["seed"])


def constructed_mixture_reference(repo_root: str | Path, tier: str, expert: Any, random_pool: Any) -> Any:
    """Q8: a mixture's behaviour reference built from the re-rolled P4.6 components by the registered construction
    (``corrective_verdicts.constructed_reference_cell``), refused unless it IS P4.7's committed ``behaviour@<tier>``
    episode for episode (``att_horizon`` and the two companions, under ``==``)."""
    import offline.corrective_verdicts as cv

    cell = cv.constructed_reference_cell(tier, expert, random_pool)
    payload = json.loads((Path(repo_root) / "docs/data/p4_7_grid.json").read_text(encoding="utf-8"))
    committed = {(int(e["seed"]), int(e["draw_id"])): (float(e["att_horizon"]), float(e["horizon_vehicle_count"]),
                                                       float(e["episode_reward"]))
                 for e in payload["episodes"] if str(e["arm"]) == f"behaviour@{tier}"}
    built = {(int(r["seed"]), int(r["draw_id"])): (float(r["att_ours"]), float(r["horizon_vehicle_count"]),
                                                   float(r["episode_reward"])) for r in cell.rows}
    if built != committed:
        differing = sorted(k for k in set(built) | set(committed) if built.get(k) != committed.get(k))
        raise ValueError(f"behaviour@{tier} constructed from {expert.spec.cell_id} and {random_pool.spec.cell_id} is "
                         f"not P4.7's committed cell on {len(differing)} episode(s), first {differing[:1]}")
    return cell


def assert_rerolls_agree(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]], *,
                         what: str) -> None:
    """Two old cells that are the same episodes re-rolled twice (P4.4's ``mappo1000`` and P4.6's
    ``behaviour@mappo1000``) agree on every value but their name and source: the route is seeded, so a difference is
    non-determinism and refuses the load."""
    fields = [key for key in EPISODE_FIELDS if key not in ("arm", "source")]
    a = {(row["seed"], row["draw_id"]): row for row in first}
    b = {(row["seed"], row["draw_id"]): row for row in second}
    if set(a) != set(b) or len(a) != len(first) or len(b) != len(second):
        raise ValueError(f"{what}: the two re-rolls do not hold the same episodes ({len(first)} and {len(second)})")
    for key in sorted(a, key=lambda k: (-1 if k[0] is None else k[0], k[1])):
        for field in fields:
            if a[key][field] != b[key][field]:
                raise ValueError(f"{what}: seed {key[0]} draw {key[1]} {field} is {a[key][field]!r} in one re-roll "
                                 f"and {b[key][field]!r} in the other")


def iql_statements_committed(repo_root: str | Path) -> dict[str, Any]:
    """Q8's committed statements, keyed as :func:`iql_behaviour_pairs` keys them: P4.6's and P4.7's P1 and every
    behaviour comparison with IQL, each exactly once in its artifact."""
    def load(name: str) -> dict[str, Any]:
        return json.loads((Path(repo_root) / "docs" / "data" / f"{name}.json").read_text(encoding="utf-8"))

    payloads = {"p4_4": load("p4_4_baselines"), "p4_6": load("p4_6_grid"), "p4_7": load("p4_7_grid")}
    comparisons = {}
    for pair in iql_behaviour_pairs():
        payload = payloads[pair["artifact"]]
        if pair["artifact"] == "p4_4":
            found = [payload["behaviour_policy_comparisons"][pair["statement"]]]
        else:
            found = [c for c in payload["behaviour_comparisons"]
                     if (c["left_arm"], c["right_arm"]) == tuple(pair["labels"])]
        if len(found) != 1:
            raise ValueError(f"{pair['artifact']} carries {len(found)} comparisons {pair['statement']!r}, not one")
        comparisons[f"{pair['artifact']}:{pair['statement']}"] = found[0]
    return {"p1_committed": {"p4_6": payloads["p4_6"]["predictions"]["P1"],
                             "p4_7": payloads["p4_7"]["predictions"]["P1"]},
            "iql_behaviour_committed": comparisons}


P5_2B_ARTIFACT = "docs/data/p5_2b_iql_correction.json"


def p5_2b_runs(repo_root: str | Path) -> list[dict[str, Any]]:
    """The corrected grid IQL cell's five runs, from P5.2b's committed record."""
    import offline.tier_statements as tst

    runs = json.loads((Path(repo_root) / P5_2B_ARTIFACT).read_text(encoding="utf-8"))["training"]["runs"]
    return sorted(({"cell": tst.CORRECTED_IQL, "seed": int(r["seed"]), "seconds": float(r["seconds"]),
                    "gradient_steps": int(r["gradient_steps"]), "clock": cr.CLOCK_LABEL, "source": P5_2B_ARTIFACT,
                    "plateaued": bool(r["plateaued"])} for r in runs), key=lambda r: r["seed"])


def grid_old_runs(output_root: str | Path, spec: Any) -> list[dict[str, Any]]:
    """A registered P5.1 / P5.2 grid cell's runs: ``gradient_steps`` from each checkpoint's provenance; no committed
    record holds their seconds (plan D3), so ``seconds`` is null with that reason.  Behaviour and random cells have
    none; P5.2b's corrected cell is :func:`p5_2b_runs`."""
    import torch

    import offline.tier_statements as tst
    from offline.dt_gate import TRAINING_SEEDS
    from offline.tier_sweep import _checkpoint_name

    if spec.method in ("behaviour", "random"):
        return []
    if spec.cell_id == tst.CORRECTED_IQL:
        raise ValueError("P5.2b's corrected cell's runs are p5_2b_runs'")
    runs = []
    for seed in TRAINING_SEEDS:
        relative = f"{spec.source}/checkpoints/{_checkpoint_name(spec.tier, spec.method, int(seed))}"
        payload = torch.load(Path(output_root) / relative, map_location="cpu", weights_only=False)
        runs.append({"cell": spec.cell_id, "seed": int(seed), "seconds": None,
                     "seconds_note": "no committed record holds this run's seconds (P5.1 / P5.2 grid baselines; plan D3)",
                     "gradient_steps": int(payload["provenance"]["gradient_steps"]), "clock": cr.CLOCK_LABEL,
                     "source": f"output/{relative}"})
    return runs


def p3a_records(roots: cr.Roots) -> dict[str, list[dict[str, Any]]]:
    """P3a's recorded rows, keyed by the ids the artifacts read: a new cell's :func:`new_cell_id`; the registered
    references' cells (P4.6's ``behaviour@random``; P5.1's random anchor)."""
    import offline.tier_statements as tst

    registered = {("hz1x1", "behaviour@random"): "p4_6:behaviour@random",
                  ("grid4x4", "random@mappo1000"): tst.ANCHOR_CELL}
    out = {}
    for cell in cr.p3a_cells():
        payload = json.loads(cr._p3a_path(roots, cell).read_text(encoding="utf-8"))
        key = registered.get((cell.scenario, cell.key), new_cell_id(cell))
        out[key] = list(payload["episodes"])
    return out


def schedule_digests(directory: str | Path) -> dict[str, Any]:
    """A collected tier's training schedules as A27(d)'s beside-table compares them: every episode's joint digest and
    each intersection's (``ix_ids`` order), by ``nortg_decomposition.action_sequence_sha256``."""
    import numpy as np

    from offline.nortg_decomposition import action_sequence_sha256

    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    order: list[str] | None = None
    joint: set[str] = set()
    per_ix: list[set[str]] = []
    for entry in manifest["episodes"]:
        with np.load(root / str(entry["filename"])) as arrays:
            ids = [str(v) for v in arrays["ix_ids"].tolist()]
            columns = [np.asarray(arrays[f"ix{i}_action"], dtype=np.int64) for i in range(len(ids))]
        if order is None:
            order, per_ix = ids, [set() for _ in ids]
        elif ids != order:
            raise ValueError(f"{root}: {entry['filename']} lists its intersections in another order")
        for i, column in enumerate(columns):
            per_ix[i].add(action_sequence_sha256(column.reshape(-1, 1)))
        joint.add(action_sequence_sha256(np.stack(columns, axis=1)))
    return {"ix_ids": order or [], "joint": joint, "per_intersection": per_ix}


def d4rl_section(path: str | Path) -> dict[str, Any]:
    """The D4RL re-run's ``compare_with`` block (the registered arms' per-seed returns beside the new one, every
    contrast recomputed), the file's digest, and its runs' seconds with A27(a)'s clock label."""
    import hashlib

    target = Path(path)
    data = target.read_bytes()
    payload = json.loads(data)
    if "compare_with" not in payload:
        raise ValueError(f"{target} carries no compare-with block: the re-run must be made with --compare-with")
    runs = [{"arm": str(c["arm"]), "seed": int(c["seed"]), "seconds": float(c["training_seconds"]),
             "wall_seconds": float(c["wall_seconds"]), "gradient_steps": int(c["gradient_steps"]),
             "clock": cr.CLOCK_LABEL} for c in payload["cells"]]
    return {"file": target.name, "file_sha256": hashlib.sha256(data).hexdigest(),
            "comparison": payload["compare_with"], "runs": runs}



# ----------------------------------------------------------------------
# E.7's diagnostics beside the D4RL section (Q18; BRIEF_46 Amendment F, F.2-3)
# ----------------------------------------------------------------------

#: The registered P8.3 artifact, whose IQL cells the D4RL re-run was compared with.
REGISTERED_D4RL = "docs/data/p8_3_d4rl_calibration.json"
#: The three values E.7 asks for, as ``offline_baselines.train_iql`` records them in its ``diagnostics`` block.
DIAGNOSTIC_FIELDS: tuple[str, ...] = ("final_q_loss", "final_v_loss", "awr_weight_near_zero_fraction")
DIAGNOSTICS_LABEL = ("diagnostics, not registered statistics (BRIEF_46 E.7, F.2-3): descriptive only; no prediction, "
                     "verdict, gate or decision reads them")


#: What each value is, written into the block beside it.
DIAGNOSTIC_DEFINITIONS: Mapping[str, str] = {
    "final_q_loss": ("the Q (TD) loss of the last gradient step, on that step's batch, as offline_baselines.train_iql "
                     "records it; not train.log's mean over the last 5,000 steps"),
    "final_v_loss": ("the V (expectile) loss of the last gradient step, on that step's batch, likewise; not "
                     "train.log's 5,000-step mean"),
    "awr_weight_near_zero_fraction": ("the fraction of AWR weights, exp(beta * (Q_target - V)) clipped at 100, below "
                                      "0.01, over the trainer's fixed sample of at most 20,000 training rows, computed "
                                      "once after training with the modules in eval mode"),
}


def _three(diagnostics: Any, *, where: str, source: str) -> dict[str, float]:
    """The three values of one trainer's ``diagnostics`` block, refused unless each is a finite number."""
    import math

    if not isinstance(diagnostics, Mapping):
        raise ValueError(f"{where}: {source} carries no diagnostics block")
    out: dict[str, float] = {}
    for key in DIAGNOSTIC_FIELDS:
        if key not in diagnostics:
            raise ValueError(f"{where}: its diagnostics block in {source} has no {key}")
        value = diagnostics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{where}: {key} is {value!r}, not a finite number ({source})")
        out[key] = float(value)
    return out


#: The IQL family (``BRIEF_46`` S.2, Q19): every run whose recipe's base is IQL.
IQL_FAMILY_BASE = "iql"


def iql_family_diagnostics(roots: cr.Roots) -> list[dict[str, Any]]:
    """Every IQL-family traffic run's arm, target mode and three values, in ``training_runs()`` order (S.2, Q19).

    The family is the 115 runs whose recipe's base is IQL: ``iql_t`` (70), the grid's ``iql_m`` (25), and hz1x1's
    ``iql`` and ``iql_100k`` (10 each). So the traffic part has both target modes from the same trainer, as the D4RL
    part does.

    The values come from each ``p9.1-run/1.0`` record's ``diagnostics`` block, not from ``train.log``. The log
    carries only the Q and V losses' 5,000-step means and no AWR fraction (F.2-3).

    Each row carries its arm and its ``target_dropout``, read from that block. Refused: a mode that is not its
    recipe's (``recipe_fields``); a record that is not its run's; a missing or non-finite value.
    """
    rows: list[dict[str, Any]] = []
    for cell, seed in cr.training_runs():
        recipe = cr.recipe_fields(cell)
        if recipe["base"] != IQL_FAMILY_BASE:
            continue
        where = f"{cell.scenario}:{cell.key} seed {int(seed)}"
        path = cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, int(seed))
        source = _relative(roots, path)
        record = json.loads(path.read_text(encoding="utf-8"))
        identity = tuple(record.get(key) for key in ("format_version", "scenario", "tier", "arm", "seed"))
        if identity != (cr.RUN_RECORD_FORMAT_VERSION, cell.scenario, cell.tier, cell.arm, int(seed)):
            raise ValueError(f"{source} is not the {cr.RUN_RECORD_FORMAT_VERSION} record of {where}")
        diagnostics = record.get("diagnostics")
        values = _three(diagnostics, where=where, source=source)
        if "target_dropout" not in diagnostics:
            raise ValueError(f"{where}: its diagnostics block in {source} has no target_dropout")
        mode = diagnostics["target_dropout"]
        if mode != recipe["target_dropout"]:
            raise ValueError(f"{where}: target_dropout is {mode!r}, not its recipe's {recipe['target_dropout']!r} "
                             f"({source})")
        rows.append({"scenario": cell.scenario, "tier": cell.tier, "arm": cell.arm, "seed": int(seed),
                     "target_dropout": mode, **values, "source": source})
    return rows


def d4rl_diagnostics(repo_root: str | Path, d4rl_path: str | Path) -> dict[str, Any]:
    """Each D4RL cell's three values for ``iql`` and ``iql_t``: five training seeds each, in seed order.

    - ``iql``: the registered P8.3 cells, from the committed artifact the re-run was compared with. Its sha256 must
      be the re-run's ``compare_with.sha256``.
    - ``iql_t``: the re-run's own cells.

    Both come from ``offline_baselines.train_iql``'s ``diagnostics`` block.
    """
    import hashlib

    from offline.dt_gate import TRAINING_SEEDS

    rerun_path = Path(d4rl_path)
    rerun_bytes = rerun_path.read_bytes()
    rerun = json.loads(rerun_bytes)
    registered_bytes = (Path(repo_root) / REGISTERED_D4RL).read_bytes()
    digest = hashlib.sha256(registered_bytes).hexdigest()
    compared = (rerun.get("compare_with") or {}).get("sha256")
    if compared != digest:
        raise ValueError(f"{rerun_path.name} was compared with a file of sha256 {compared}, not with "
                         f"{REGISTERED_D4RL} (sha256 {digest})")
    seeds = sorted(int(seed) for seed in TRAINING_SEEDS)
    rows: list[dict[str, Any]] = []
    for arm, payload, source in (("iql", json.loads(registered_bytes), REGISTERED_D4RL),
                                 ("iql_t", rerun, rerun_path.name)):
        cells = sorted((c for c in payload["cells"] if c.get("arm") == arm), key=lambda c: int(c["seed"]))
        found = [int(c["seed"]) for c in cells]
        if found != seeds:
            raise ValueError(f"the D4RL {arm} cells for seeds {found}, not the five training seeds {seeds} ({source})")
        for c in cells:
            rows.append({"arm": arm, "seed": int(c["seed"]),
                         **_three(c.get("diagnostics"), where=f"D4RL {arm} seed {int(c['seed'])}", source=source),
                         "source": source})
    return {"rows": rows, "registered": {"file": REGISTERED_D4RL, "sha256": digest},
            "rerun": {"file": rerun_path.name, "sha256": hashlib.sha256(rerun_bytes).hexdigest()},
            "arms": {"iql": "P8.3's registered IQL cells", "iql_t": "A27(c)'s re-run (--iql-target-dropout off)"}}


def iql_diagnostics(roots: cr.Roots, d4rl_path: str | Path) -> dict[str, Any]:
    """The block beside ``p9_1_iql_targets.json``'s ``d4rl`` section (Q18 and Q19; ``BRIEF_46`` E.7, F.2-3, S.2),
    labelled as diagnostics: each D4RL cell's three values for ``iql`` and ``iql_t``, and every IQL-family traffic
    run's arm, target mode and three values."""
    return {"label": DIAGNOSTICS_LABEL, "fields": dict(DIAGNOSTIC_DEFINITIONS),
            "d4rl": d4rl_diagnostics(roots.repo_root, d4rl_path),
            "traffic_iql_family": {
                "source": "each IQL-family run's p9.1-run/1.0 record (iql_t, iql_m, iql, iql_100k), its diagnostics block",
                "runs": iql_family_diagnostics(roots)}}


DIGEST_CONSTRUCTION = (
    "sha256 over the text '<file sha256>  <path relative to the output root>\\n' for every file under the directory "
    "(the excluded subdirectories aside), in path order -- the sha256sum line format, so `find <dir> -type f | LC_ALL=C "
    "sort | xargs sha256sum | sha256sum`, run in the output root over the same files, recomputes it")
#: The D4RL re-run's output name (``calibration/d4rl_run.py``: K = 64, 40,000 steps, targets off).
D4RL_OUTPUT_NAME = "p8_3_train_k64_40000_iql_t.json"


def directory_digest(output_root: str | Path, directory: str, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    """One digest over every file of a run directory (:data:`DIGEST_CONSTRUCTION`); *exclude* names subdirectories left
    out (checkpoints are identified by their canonical digests in the run records, and logs are not inputs)."""
    import hashlib

    root = Path(output_root)
    base = root / directory
    skipped = {base / name for name in exclude}
    paths = sorted(path for path in base.rglob("*") if path.is_file()
                   and not any(s == path or s in path.parents for s in skipped))
    lines = "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n"
                    for path in paths)
    return {"directory": f"output/{directory}", "files": len(paths), "excluded": list(exclude),
            "sha256": hashlib.sha256(lines.encode("utf-8")).hexdigest(), "construction": DIGEST_CONSTRUCTION}


TRACE_GAP_SECONDS = 60.0


def _trace_samples(path: Path) -> list[float]:
    return [float(line.split(",")[0]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def trace_gaps(paths: Sequence[Path], *, threshold: float = TRACE_GAP_SECONDS) -> list[dict[str, Any]]:
    """Every gap longer than *threshold* between two consecutive samples of a driver GPU trace (Unix seconds; the driver
    samples every 20 s, so a longer gap is a pause of the machine).  ``trace`` is named relative to the runs directory."""
    gaps = []
    for path in paths:
        samples = _trace_samples(path)
        for before, after in zip(samples, samples[1:]):
            if after - before > threshold:
                gaps.append({"from_unix": before, "to_unix": after, "seconds": after - before,
                             "trace": "/".join(Path(path).parts[-3:])})
    return gaps


def trace_spans(paths: Sequence[Path]) -> list[tuple[float, float]]:
    """Each trace's first and last sample: the stretches of wall time the traces cover."""
    spans = []
    for path in paths:
        samples = _trace_samples(path)
        if samples:
            spans.append((samples[0], samples[-1]))
    return spans


def flag_paused_runs(runs: Sequence[Mapping[str, Any]], gaps: Sequence[Mapping[str, Any]],
                     spans: Sequence[tuple[float, float]]) -> list[dict[str, Any]]:
    """Each run with ``paused_during_run`` (every trace gap its [started_unix, finished_unix] overlaps: its seconds then
    include a pause of the machine) and ``trace_covered`` (whether a trace covers the whole run)."""
    out = []
    for run in runs:
        start, end = float(run["started_unix"]), float(run["finished_unix"])
        paused = [dict(gap) for gap in gaps if gap["from_unix"] < end and gap["to_unix"] > start]
        covered = any(first <= start and end <= last for first, last in spans)
        out.append({**run, "paused_during_run": paused, "trace_covered": covered})
    return out


def _trace_readings(path: Path) -> list[tuple[float, float | None, float | None]]:
    """The driver's trace as ``(unix, memory.used MiB, utilization %)``; a reading ``nvidia-smi`` did not give (an
    empty or non-numeric field) is ``None``.  A line whose time is not a number is refused."""
    out = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(",")
        try:
            moment = float(parts[0])
        except ValueError:
            raise ValueError(f"{path} (gpu_trace.csv line {number}) is not a reading: {line!r}") from None
        values: list[float | None] = []
        for field in (parts[1:3] + ["", ""])[:2]:
            try:
                values.append(float(field))
            except ValueError:
                values.append(None)
        out.append((moment, values[0], values[1]))
    return out


def gpu_overlap(roots: cr.Roots, *, start_unix: float, end_unix: float) -> dict[str, Any]:
    """``BRIEF_46`` Amendment D, D.2-2: another GPU process's window (P9.3's smoke, from its note) against the
    campaign's training runs -- each run whose ``[started_unix, finished_unix]`` overlaps it, with the overlap in
    seconds -- and the stage's own GPU trace inside the window (bounds included) beside the trace outside it, so the
    packet can show the second process's footprint.  Reads the run records and the traces; writes nothing."""
    start, end = float(start_unix), float(end_unix)
    if not end > start:
        raise ValueError(f"[{start}, {end}]: a window ends after it starts")
    runs, read = [], 0
    for cell, seed in cr.training_runs():
        path = cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, int(seed))
        if not path.is_file():
            continue
        read += 1
        record = json.loads(path.read_text(encoding="utf-8"))
        first, last = float(record["started_unix"]), float(record["finished_unix"])
        overlap = min(last, end) - max(first, start)
        if overlap > 0.0:
            runs.append({"scenario": cell.scenario, "tier": cell.tier, "arm": cell.arm, "seed": int(seed),
                         "started_unix": first, "finished_unix": last, "seconds": float(record["seconds"]),
                         "overlap_seconds": overlap})
    readings = [r for path in sorted((Path(roots.runs_root) / "logs").glob("*/gpu_trace.csv"))
                for r in _trace_readings(path)]

    def summary(chosen: list[tuple[float, float | None, float | None]]) -> dict[str, Any]:
        full = [(m, u) for _, m, u in chosen if m is not None and u is not None]
        return {"samples": len(chosen), "readings": len(full),
                "memory_used_mib": [min(m for m, _ in full), max(m for m, _ in full)] if full else None,
                "utilization_pct": [min(u for _, u in full), max(u for _, u in full)] if full else None}

    inside = [r for r in readings if start <= r[0] <= end]
    outside = [r for r in readings if not start <= r[0] <= end]
    return {"window": {"start_unix": start, "end_unix": end, "seconds": end - start}, "records_read": read,
            "runs": runs, "overlap_seconds_total": sum(r["overlap_seconds"] for r in runs),
            "trace": {"inside": summary(inside), "outside": summary(outside)},
            "rule": "BRIEF_46 Amendment D, D.2-2: a run overlaps iff its record's bracket and the window share time; the "
                    "trace's readings inside the window are shown beside those outside it"}


P5_3B_ARTIFACT = "docs/data/p5_3b_nortg.json"


def p5_3b_old_tier(repo_root: str | Path) -> dict[str, Any]:
    """P5.3b's open-loop numbers on the ORIGINAL hz1x1 random tier, copied from its committed artifact: the distinct
    action sequences per seed of ``dt@random`` and the attractor (A27(d)'s beside-table sits next to them)."""
    path = Path(repo_root) / P5_3B_ARTIFACT
    identity = json.loads(path.read_text(encoding="utf-8"))["mechanism"]["action_identity"]
    return {"source": f"{P5_3B_ARTIFACT} mechanism.action_identity", "sha256": _sha256(path),
            "n_distinct_action_sequences": {key: value for key, value in identity["n_distinct_action_sequences"].items()
                                            if key.startswith("dt@random seed ")},
            "attractor": identity["attractor"]}


#: The committed records every artifact may read, digested into its inputs.
COMMITTED_INPUTS: tuple[str, ...] = (
    "docs/data/p4_4_baselines.json", "docs/data/p4_6_grid.json", "docs/data/p4_7_grid.json", P4_DT_RECORD,
    *HZ1X1_TRAINING_RECORDS.values(), P5_2B_ARTIFACT, "docs/data/p8_3_d4rl_calibration.json", P5_3B_ARTIFACT,
)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources(roots: cr.Roots, *, d4rl_path: str | Path | None) -> Sources:
    """Everything, read once and checked on the way in: the grid's registered cells by P5.2c's pinned loader, the hz1x1
    old cells joined to their re-rolls, the new cells joined across both routes, every run's seconds and steps and
    its session's code commit (G.2, from the driver's captures), the declarations, P3a's rows, the random_fresh tiers'
    schedules, the D4RL re-run and the IQL diagnostics (Q18), the inputs by digest, the git block."""
    import offline.tier_statements as tst
    from offline import att_rederivation as ar

    if d4rl_path is None:
        raise ValueError("the D4RL re-run's output is an input of p9_1_iql_targets.json (A27(c)); pass --d4rl")
    git = tst.git_block()
    grid = tst.load_inputs(tst.Roots(repo_root=Path(roots.repo_root), output_root=Path(roots.output_root)))
    cells: dict[str, Any] = {}
    records: dict[str, tuple[dict[str, Any], ...]] = {}
    runs: dict[str, tuple[dict[str, Any], ...]] = {}
    for cid, cell in grid.cells.items():
        cells[cid] = cell
        records[cid] = tuple({**row, "tier": cell.spec.tier,
                              "source": f"output/{cell.spec.eval_file} (att_ours) and P8.4b's rows (att_engine), read "
                                        "by P5.2c's pinned loader (tier_statements.load_inputs)"} for row in cell.rows)
        found = p5_2b_runs(roots.repo_root) if cid == tst.CORRECTED_IQL else grid_old_runs(roots.output_root, cell.spec)
        if found:
            runs[cid] = tuple(found)
    for artifact, arm in cr.HZ1X1_OLD_CELLS:
        cid = hz1x1_old_id(artifact, arm)
        rows = old_hz1x1_rows(roots, artifact, arm)
        method, tier = ar.normalise_arm("hz1x1", arm)
        cells[cid] = as_statement_cell(cid, method, tier, arm, cid.split(":")[0], artifact, rows)
        records[cid] = rows
        found = hz1x1_old_runs(roots.repo_root, artifact, arm)
        if found:
            runs[cid] = tuple(found)
    # Q8 (BRIEF_46 Amendment B, B.2): the twice-rolled behaviour episodes agree; the mixtures' references are built
    # from P4.6's re-rolled components and joined to P4.7's committed cells.
    assert_rerolls_agree(records["p4_4:mappo1000"], records["p4_6:behaviour@mappo1000"],
                         what="P4.4's mappo1000 and P4.6's behaviour@mappo1000")
    for tier in MIXTURE_REFERENCE_TIERS:
        cid = constructed_id(tier)
        cells[cid] = constructed_mixture_reference(roots.repo_root, tier, cells["p4_6:behaviour@mappo1000"],
                                                   cells["p4_6:behaviour@random"])
        records[cid] = tuple({**row, "tier": tier,
                              "source": "constructed by mixture_tiers.constructed_reference_artifact's assignment from "
                                        "p4_6:behaviour@mappo1000 and p4_6:behaviour@random; reproduces P4.7's "
                                        "committed att_ours"} for row in cells[cid].rows)
    traces = sorted((Path(roots.runs_root) / "logs").glob("*/gpu_trace.csv"))
    gaps, spans = trace_gaps(traces), trace_spans(traces)
    sessions = driver_sessions(roots)
    trained = set(cr.training_cells())
    for cell in cr.evaluation_cells():
        cid = new_cell_id(cell)
        rows = new_cell_rows(roots, cell)
        cells[cid] = as_statement_cell(cid, cell.arm, cell.tier, cell.key, "p9_1",
                                       _relative(roots, cr.route_i_output(roots, cell)), rows)
        records[cid] = rows
        if cell in trained:
            entries = []
            for run in new_training_runs(roots, cell):
                record = json.loads(cr.run_record_path(roots, cell.scenario, cell.tier, cell.arm, run["seed"])
                                    .read_text(encoding="utf-8"))
                entry = {**run, "cell": cid, "window_means": [float(v) for v in record["window_means"]],
                         "plateaued": record["plateaued"], "started_unix": float(record["started_unix"]),
                         "finished_unix": float(record["finished_unix"])}
                entries.append({**entry, **run_session(entry, sessions)})
            runs[cid] = tuple(flag_paused_runs(entries, gaps, spans))
    declarations = {
        name: json.loads((Path(roots.run_root) / f"declaration_{name}.json").read_text(encoding="utf-8"))
        for name in ("mix_m1000_m500", "random_fresh_hz1x1", "random_fresh_grid4x4")}
    schedules = {scenario: schedule_digests(Path(roots.corpus_root) / fresh)
                 for scenario, (_, fresh) in cr.RANDOM_FRESH_DIRS.items()}
    d4rl = d4rl_section(d4rl_path)
    p4_4 = json.loads((Path(roots.repo_root) / "docs/data/p4_4_baselines.json").read_text(encoding="utf-8"))
    registered = {"p4_4_delta": float(p4_4["comparisons"]["madt_vs_iql"]["delta"]),
                  "plateau_counts": plateau_counts(roots.repo_root), "p5_3b_old_tier": p5_3b_old_tier(roots.repo_root),
                  **iql_statements_committed(roots.repo_root)}
    inputs = [{"file": relative, "sha256": _sha256(Path(roots.repo_root) / relative)} for relative in COMMITTED_INPUTS]
    inputs += [directory_digest(roots.output_root, cr.RUNS_DIR, exclude=("checkpoints", "logs")),
               directory_digest(roots.output_root, cr.RUN_DIR)]
    inputs += [{"file": f"{fresh}/manifest.json", "root": "corpus",
                "sha256": _sha256(Path(roots.corpus_root) / fresh / "manifest.json")}
               for _, fresh in cr.RANDOM_FRESH_DIRS.values()]
    inputs += [{"file": d4rl["file"], "root": "D4RL re-run output", "sha256": d4rl["file_sha256"]},
               {"loader": "tier_statements.load_inputs (P5.2c, pinned)", "provenance": grid.provenance}]
    return Sources(cells=cells, records=records, runs=runs, declarations=declarations,
                   grid_books=tst.make_books(grid.cells), p3a=p3a_records(roots), schedules=schedules, d4rl=d4rl,
                   registered=registered, inputs=tuple(inputs), git=git,
                   diagnostics=iql_diagnostics(roots, d4rl_path), sessions=tuple(sessions),
                   evaluation_code_commit=stage_code_commit(sessions, "evaluate"),
                   d4rl_session=stage_code_commit(sessions, "d4rl"))


#: The five artifacts of BRIEF_46 3.5 and their builders.
ARTIFACTS: dict[str, Any] = {
    "p9_1_exposure_parity.json": build_exposure_parity,
    "p9_1_iql_targets.json": build_iql_targets,
    "p9_1_random_fresh.json": build_random_fresh,
    "p9_1_mixture.json": build_mixture,
    "p9_1_budget.json": build_budget,
}


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m offline.corrective_report", allow_abbrev=False,
                                     description="P9.1: the five A27 artifacts from the campaign's run directories")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--draws-root", type=Path, required=True)
    parser.add_argument("--d4rl", type=Path, required=True, help=f"the D4RL re-run's {D4RL_OUTPUT_NAME}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report", allow_abbrev=False).add_argument("--out-dir", type=Path, default=None)
    return parser


def main(argv: Any = None) -> int:
    """``report``: every artifact built in memory first; each written once, or left alone if the file on disk is the
    same bytes; a different file on disk is refused (exit 2) and nothing is written."""
    import sys

    args = build_parser().parse_args(argv)
    roots = cr.Roots(repo_root=args.repo_root, output_root=args.output_root, corpus_root=args.corpus_root,
                     draws_root=args.draws_root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.repo_root) / "docs" / "data"
    try:
        src = load_sources(roots, d4rl_path=args.d4rl)
        if src.git.get("dirty") is not False:
            raise ValueError("the report is built from a clean committed tree (git dirty must be false)")
        texts = {name: artifact_text(builder(src)) for name, builder in ARTIFACTS.items()}
        for name, text in texts.items():
            target = out_dir / name
            if target.exists() and target.read_text(encoding="utf-8") != text:
                raise FileExistsError(f"{target} exists with other content; an artifact is not rewritten")
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as exc:
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    for name, text in texts.items():
        target = out_dir / name
        state = "unchanged (identical bytes)" if target.exists() else "written"
        if not target.exists():
            target.write_text(text, encoding="utf-8")
        print(f"{target}: {state}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the report is run by hand from a clean tree
    raise SystemExit(main())
