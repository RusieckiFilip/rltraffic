# BRIEF_45 — P5.2c: P5.2's registered statements, scored by committed code under both ATT definitions (`DEFERRED` 109, with `DEFERRED` 108)

**Mode:** Explore → Plan (gate G0: `docs/plans/p5.2c.md`) → Code (tests first) → build the artifact from a clean committed tree →
Commit. Branch `task/p5.2c-statements` from `main`. One task, one session. **No simulation, no training, no evaluation, no GPU, and
no run that needs the author's token**: every number is computed from committed records read at their digests, in seconds.

**Authorised by:** the author's ruling of 2026-10-08 (*"Issue DEFERRED 109 now as the next task (P5.2's registered statements on
every out-of-sample tier under both definitions, with DEFERRED 108's corrected count of 7 and its T-regress). After its merge and its
CI, stop and wait for me."*), recorded in `PROJECT_PLAN` §8 the same day. A recomputation of committed measurements, not an
experiment: A25 is unaffected. **This is Claude Code's last task on the project**: the author writes the paper outside it, so the
artifact this task commits is the source he quotes P5.2 from. It must be complete, self-describing, and readable without the code.

## 0. Frozen interface contracts (read them from disk; do not infer)
- `docs/CONTRACTS.md` (C6 v1.1). Frozen paths (CLAUDE.md §1) untouched. No new dependency.
- **Every existing module UNCHANGED — import only:** `offline/tier_sweep.py`, `offline/iql_correction.py`,
  `offline/att_rederivation.py`, `offline/dt_gate.py`, `offline/spatial_mixing.py`. **No existing test changes** (additions only).
  If an existing function cannot be reused as it is, stop and say so in the plan; do not copy it with a change.
- **Nothing under `output/` is written, moved or deleted.** The module writes exactly ONE file, the artifact at the path its
  command line names (`docs/data/p5_2_statements.json`), and only after every input check has passed (the filesystem-mutation
  barrier, `PROJECT_PLAN` §5): a failed check leaves no file and no partial file, and an existing artifact is not touched. It
  refuses an out path under `output/` or under any input root.
- P5.2's packet (`docs/returns/P5.2.md`), plan (`docs/plans/p5.2.md`), `BRIEF_27`, and P5.2b's artifact
  (`docs/data/p5_2b_iql_correction.json`) are history and are not edited.

## 1. Why this task exists
1. **Rule R makes `att_engine` the primary metric for every claim on grid4x4** (A11(c), `docs/notes/A11_DRAFT.md`; its gate
   passed in P8.4b), with `att_ours` beside it and A11(b)'s five quantities on every reported cell — `att_ours`, `att_engine`,
   `entered`, `created`, `never_entered`. P8.4b re-derived only the seven registered contrasts it enumerated (`docs/data/
   p8_4b_rederivation.json`, `contrasts`), P5.2's Q0 stop rule among them (V5) and P5.1's P2a (V4). **P5.2's other registered
   statements have no `att_engine` version.**
2. **P5.2's statements Q1–Q6 have NO committed scorer at all.** They were scored once, by the coordinator, directly from
   `output/p5_2/eval_*.json` (`119cc48`, `BRIEF_27` GATE 2; the packet says so at `docs/returns/P5.2.md:14`). What exists as code:
   `tier_sweep.score_level` (Q1), `tier_sweep.concordance` and `predicted_order` (Q2b), `tier_sweep.score_stop_rule` (Q0's stop
   rule only), and P5.2b's `iql_correction.random_tier_statements`, which composes Q1's IQL cell, Q2a, Q2b, Q2b-hard, D9, Q3a and Q3c
   **at the random tier only**.
3. **Never scored in any committed record**, under either definition: **Q0a and Q0b** (the packet reports only that the stop rule
   did not fire), **Q5a and Q5c** (GATE 2 reports Q5b's three values), **Q3c at `maxpressure` and `fixedtime`** (GATE 2 says only
   that every paired difference resolves against the DT), and **Q6's numbers** (*"no DT arm collapsed on any tier"*, no values).
   This task scores them for the first time, by the registered rules, whichever way they come out.
4. **Two integers in P5.2's packet depend on how exact ties are broken.** At `fixedtime`, `dt_spatial` = `dt_nomix` and `bc` =
   `bc_top10_perix` to full double precision (identical episodes; `docs/returns/P5.2.md:57-60`). (a) Q2b's fixedtime count is
   recorded as 8; under the tie rule P5.2 declared on 2026-08-24 (a tie is discordant for both orders) it is **7** (`DEFERRED` 108).
   (b) Q3a's fixedtime rank of `dt_nomix` is recorded as 3/6; `dt_nomix` ties `dt_spatial` exactly, so its rank is **2 to 3** —
   2 counting only the arms strictly lower (P5.2b's rule), 3 if the tie is broken against it (found 2026-10-10 while writing this
   brief; the coordinator's reading of the committed levels, to be verified by the plan). **The class is closed by the artifact, not
   the two sentences:** every reported integer that depends on tie-breaking carries its value under the declared rule and its range.
5. **The random tier has two IQL cells:** P5.2's original (trained on twice its declared data, `DEFERRED` 106) and P5.2b's corrected
   one. The paper's random tier is the corrected one; P5.2's packet scored the original.
6. **The README's C1 lines rest on the coordinator's note `docs/notes/readme_2026-10-05/c1_rule_r.json`**, whose intervals use
   z = 1.959963984540054 (`c1_rule_r.py:54`), not the project's 1.96 (`offline/dt_gate.py:218`); means are unaffected and no README sentence changes at its
   precision (checked 2026-10-10). This artifact replaces the note as the source.

## 2. Scope fence — what NOT to build
- No episode is rolled, no checkpoint loaded, no model trained or evaluated. If a needed value is not in a committed record, stop
  and say so; do not compute it by simulation.
- **No new prediction.** R′ is not recomputed from `att_engine` inputs. Q1, Q1b and Q2's predicted orderings exist only on
  `att_horizon` (= `att_ours`); under `att_engine` they are compared with `att_engine` measurements and **labelled exactly so** —
  the convention of `BRIEF_44` Amendment B, B1.7(d), already applied to P5.2b's random-tier IQL cell.
- No change to P5.2's statements in place; no change to P5.2b's artifact; no edit to `README.md` in the branch (the coordinator
  updates it at the merge if a sentence changes).
- Out of scope: P4.x, P5.1 and P5.3 statements (P8.4b's V1–V7 hold their registered verdicts); the E1 and I1/J1 envelopes (the
  replicate files carry no `att_engine`: P8.4b excluded replicates by design, `att_rederivation.is_replicate_artifact`); hz1x1;
  cologne3. `mappo1000`'s Q3 ranking is context only, as registered.

## 3. What the task produces
### 3.1 One module, one artifact
`offline/tier_statements.py` (the plan may rename it, with the reason): the scorers, the input layer, the report builder, and a CLI
with `build` and `check` (`check` verifies every input and prints what `build` would read; it writes nothing). The artifact
`docs/data/p5_2_statements.json`, format `p5.2-statements/1.0`, is a pure function of its inputs (sorted keys, no rounding, no
timestamp; the git commit and dirty flag recorded, dirty must be false), so its T-regress reproduces it byte for byte.

### 3.2 The registered text the scorers implement
`docs/plans/p5.2.md` §4 as on `main`, with `BRIEF_27`'s Amendments H and I (`fixedtime` joined under H1/H3; Q1's threshold
transported by I3), C4 (Q2b-hard), and the tie rule declared in §4 Q2a on 2026-08-24. **The plan quotes each rule from those
files, not from this brief, and flags every place where this brief and the files disagree — the files win.** The enumeration:

| statement | tiers | inputs (P8.4b arm labels: P5.2's cells `<method>@<tier>`; P5.1's mappo1000 cells `<method>@grid4x4_mappo1000`) |
|---|---|---|
| **Q0a, Q0b**, and the stop rule beside them | `mappo1000` | d4: P5.2's `dt_spatial_h4`, `dt_nomix_h4`; d1: P5.1's `dt_spatial`, `dt_nomix`; `I = d4 − d1` per shared draw. E1 measured zero, so the default regime was kept and D17's deterministic 1-head pair was never trained — **verify on disk**; the stop rule must equal P8.4b's V5 and d1 must equal V4 |
| **Q1** (19 cells, `tier_sweep.OUT_OF_SAMPLE_CELLS`, `score_level`) and **Q1b** (`behaviour@maxpressure` vs 167.4920 and `behaviour@fixedtime` vs 206.9318, `tier_sweep.ANCHOR_BAND`) | 3 out-of-sample tiers + `bc_top10_perix@mappo1000` | the cells themselves |
| **Q2a, Q2b, Q2b-hard**, each with **D9's per-seed ordering** | `maxpressure`, `fixedtime`, `random` | the six methods. The text says "the 2 out-of-sample tiers" — written before `fixedtime` joined; H3 registered `fixedtime`'s predictions and the packet scored three. Q2b's "≥ 24 of 30 overall" is stale (written for two tiers) and is **not scored**; say so in the artifact |
| **Q3a, Q3b, Q3c** | the three out-of-sample tiers; `mappo1000` as context, no verdict | Q3b at `random` reads the shared random anchor as the behaviour policy (D12) |
| **Q4a** / **Q4b** | three / all four | `d(t) = dt_spatial − dt_nomix`; `mappo1000` from P5.1's pair. Q4b on the **four-rung** chain `d(mappo1000) < d(maxpressure) < d(fixedtime) < d(random)` (§4 Q4: "three rungs and a gap became four graded rungs"), the bullet's three-rung form reported beside it |
| **Q5a** / **Q5b, Q5c** | three / all four | `mappo1000`: P5.2's `bc_top10_perix@mappo1000` against P5.1's `bc_top10` and `bc` |
| **Q6** | all four, `dt_spatial` and `dt_nomix` separately | the shared random anchor: P5.1's `eval_random.json`; P5.2's `eval_mappo1000_random.json` is episode-identical (500 of 500, checked 2026-10-10) — **assert it**. `mappo1000` marked *seen* (P5.1's cells) |
| **leaders** — descriptive, not registered | all four | level order with ties, leader, runner-up, leader − runner-up paired CI, per-seed first places: the C1 crossover sentence's numbers |

### 3.3 Scoring conventions (rulings; the plan may challenge any with a reason)
- **Paired unit and interval, P5.2's own:** per-draw mean over the seeds (`dt_gate._per_draw_means`), shared draws only (refuse
  otherwise), `dt_gate.mean_ci95` (`ddof=1`, 1.96). Every interval carries its width.
- **Outcome vocabulary, exactly the registered one.** An interval statement is HELD / FAILED / NOT RESOLVED by the side of zero its
  CI lies on relative to the predicted sign (P5.1's `spatial_mixing._outcome_from_interval` is the shape for a predicted-negative
  quantity). A statement registered "on all three/four tiers" is HELD iff every tier is HELD, FAILED iff any tier is FAILED, NOT
  RESOLVED otherwise. Q0's stop rule is STOP / CONTINUE (`tier_sweep.stop_rule_verdict`). Q6 is COLLAPSED iff the arm is worse than
  the anchor with the paired CI excluding zero, else NOT COLLAPSED. Q1 / Q1b by their bands. Q2a: an exact tie for first is no
  unique first place (P5.2b's rule). Q4b: HELD iff the point estimates increase strictly along the chain; the reading names an
  inverted (strictly decreasing) chain.
- **Q3b** is scored per tier by the interval rule, and its registered sub-prediction — resolves at `random`, does not resolve at
  `maxpressure` or `fixedtime` — is reported separately as matched or not.
- **Q3c registers a value per tier and no outcome rule.** It is reported as the quantity with its CI, the predicted value beside
  it, and P5.2b's reading (resolves for the DT / against the DT / NOT RESOLVED) — **no verdict**, and the artifact says why.
- **Exact zeros.** A difference identically zero on every draw has CI [0, 0]: NOT RESOLVED for an interval rule, sign zero
  (neither negative nor positive) for a sign rule, and reported as *exactly zero (identical episodes)* — never folded into a
  neighbouring outcome.
- **Ties.** Exactly equal levels are not an ordering (the 2026-08-24 rule). Q3a's rank is 1 + the arms strictly lower, with the
  arms it ties and the rank range. Every tie-dependent integer (Q2b, Q2b-hard, Q3a, first places, per seed too) carries its value
  under the rule and its range over all tie-break orders. The artifact lists every pair of cells on a tier with exactly equal
  levels and every pair with identical episodes, under each definition.
- **Per seed.** Every pairwise contrast carries `tier_sweep.per_seed_block` (P5.1's `per_seed_advantages` and
  `seed_reversal_qualifier`, imported) under each definition; `I`'s per-seed value is
  `(mean_s(dt_spatial_h4) − mean_s(dt_nomix_h4)) − (mean_s(dt_spatial) − mean_s(dt_nomix))` over draws. For the behaviour and
  random-anchor cells a seed label is an evaluation label, not a training seed: say so where those blocks appear.
- **Definitions.** Every statement is computed under both. `definition_dependent` is true iff the two verdicts differ; under Rule R
  the `att_engine` verdict is the primary one and both are reported whether or not they agree (A11(d)).
- **Versions.** `corrected` (the paper's; the random tier's IQL is P5.2b's) and `as_committed` (P5.2's original IQL cell). Every
  random-tier statement involving IQL is computed in both; every other statement is identical by construction, computed once, and
  the identity asserted.

### 3.4 The artifact's blocks
`provenance` (git, every input file with its sha256, P8.4b's `declared_cells_sha256`, P5.2b's run manifest) · `definitions` (Rule
R; the paired unit; the CI rule) · `cells` (every cell any statement reads: n, mean, CI, per-seed means under each definition, A11(b)'s
companions — means of `entered`, `created`, `never_entered`, `entered_fraction`, and how many episodes left a vehicle never entered —
and its sources) · `statements` (one entry per statement × tier × definition × version where it differs: inputs, numbers, verdict,
the rule quoted with its plan location, per-seed blocks, `definition_dependent`) · `ties` · `leaders` · `comparison_with_the_record`
(every value P5.2's packet, `BRIEF_27` GATE 2, P5.2b's artifact and P8.4b's V4/V5 recorded, beside this artifact's value, equal or
explained — `DEFERRED` 108's 8 → 7 and Q3a's fixedtime 3 → 2..3 named; *never scored* stated for item 1.3's list) · `changes`
(statement by statement: as committed `att_ours` → corrected `att_ours` → corrected `att_engine`) · `what_this_does_not_say`.

## 4. Per-file requirements
- **`offline/tier_statements.py`.** `from __future__ import annotations`, full type hints, docstring stating the format version and
  the alignment convention (the paired unit; seeds pooled per draw; shared draws only). **Reuse by import, never re-implement:**
  `dt_gate.mean_ci95`, `_per_draw_means`, `EpisodeResult`; `tier_sweep.score_level`, `predicted_order`, `concordance`,
  `per_seed_block`, `stop_rule_verdict`, `PREDICTED_LEVELS`, `OUT_OF_SAMPLE_CELLS`, `HARD_SUBSET`, `LEVEL_BAND`, `ANCHOR_BAND`,
  `CONCORDANCE_THRESHOLD`; P5.2b's anchors in `iql_correction` (`p5_2_sums`, `read_p5_2_json`, `assert_p8_4b_campaign`, the
  `_p8_4b_rows` check, `assert_c1_note_means`, `random_tier_statements`). Where a needed function is private, the plan says whether it
  imports it or restates it with a test proving the two equal.
- **Inputs, each read at its digest:** `output/SHA256SUMS_p5_2.txt` (`fde8309b…`, `iql_correction.PINS`); `output/SHA256SUMS_p5_1.txt`
  (`023607ffe65a881932f0069412f442c65b1097bb87fdd7852873a1211fe24c61`, computed 2026-10-10 — the plan recomputes it); `output/
  SHA256SUMS_p5_2b.txt` (`f3e1994c…`, P5.2b's `inputs.run_manifest`); P8.4b's campaign (`declared_cells_sha256` `1f29b469…`) and
  P5.2b's re-derivation campaign (`09568e28…`); `docs/data/p5_2b_iql_correction.json` (`216b9f24243c317d22b13a3cdcd346471bf9cfe9832e5a5495b8158f04eea1f8`
  at `main`); `docs/notes/readme_2026-10-05/c1_rule_r.json` (`643b73bc…`). Every P8.4b row refused unless `reproduces_committed` is
  true and its `att_ours` equals the eval file's `att_horizon` for its (seed, draw). The corrected IQL cell: `att_ours` from
  `output/p5_2b/eval_random_iql.json`, `att_engine` and companions from `output/p5_2b/rederivation/` (500 cells), under the same
  anchor, and refused unless both equal P5.2b's committed `cells.corrected` block.
- **CLI:** `python -P -m offline.tier_statements build --repo-root . --output-root /home/filip/rltraffic/output --out
  docs/data/p5_2_statements.json` and `check`. The plan states the exact flags.
- **`tests/test_tier_statements.py`** (and a fixtures module if needed).

## 5. Tests — first, red for their own reasons; the named mutations executed, committed and pasted
**Synthetic (they run on CI):** every scorer on hand-built cells whose answers the test computes by a second route (raw
`numpy`/`statistics` on the arrays, never the scorer) — exact ties, exact zeros, a straddling CI, a reversing seed, a missing draw
(refused), mismatched seeds (refused); the tie audit on a tier with two exact ties, its range enumerated in the test with
`itertools.permutations`; Q3a's rank range under a tie; Q4b where the three- and four-rung forms disagree; Q6 on a synthetic anchor;
the version logic (IQL-free statements identical across versions, asserted; IQL statements differ); the barrier (an out path under
`output/` refused; a failing input check leaves no file and an existing artifact unchanged).
**Real data (gated on `RLTRAFFIC_OUTPUT_ROOT`, naming it):**
- **T-reproduce-record (load-bearing):** the `as_committed` `att_ours` statements reproduce every value recorded in
  `docs/returns/P5.2.md` and `BRIEF_27` GATE 2 — Q1 15 of 19 and its four misses; Q1b 0.04 % / 0.09 %; Q2a `bc`/`iql`/`iql`; Q2b
  11 · 7 · 8 with 108's note; Q2b-hard 2 · 2 · 3 and the 2..4 range; Q3a 2 · 2..3 · 3; Q3b −0.7294 [−0.9139, −0.5450] and
  −5.0360 [−5.5005, −4.5716] and the exact fixedtime zero; Q4 +39.5649 [+36.0510, +43.0787], +0.7267 [+0.6206, +0.8328], exact
  0.0000, −0.7063 [−1.3172, −0.0953] with per-seed 5/5, 5/5, n/a, 4/5; Q5b −469.69 [−477.81, −461.58], −0.28 [−0.67, +0.11],
  −49.16 [−55.26, −43.05] — at the precision each was recorded, every exception named and asserted as an exception.
- **T-reproduce-P8.4b:** the stop rule's d4 and Q4's d(mappo1000) equal V5 and V4 under both definitions.
- **T-reproduce-P5.2b:** the random tier on both versions equals P5.2b's committed `statements` before/after, field by field, under
  both definitions.
- **T-note:** every tier's six means equal the c1 note's by its own route (`statistics.mean`) — means only, for the reason in §1.6.
- **T-companions:** A11(b)'s five quantities on every cell.
- **T-regress:** the committed artifact regenerates byte for byte from `output/`.
**Mutations (each executed, committed with its output, KILLED):** the tie rule disabled; episode-level instead of per-draw pairing;
the corrected and original IQL cells swapped; z = 1.959964; Q4b scored on three rungs only; Q6 against behaviour instead of the
anchor; Q5b's `mappo1000` tier dropped; an exact zero folded into HELD; `definition_dependent` always false; the out-path barrier
removed. The plan may add more; it may not drop one without a reason the coordinator accepts.

## 6. Gates, in order
| gate | what it checks | who | what stops the task |
|---|---|---|---|
| **G0** | the plan `docs/plans/p5.2c.md`: every rule quoted from the registered files, every disagreement with this brief flagged, the commits named (≤ ~2 source files each), the tests listed | coordinator, as Amendment A on `main` | a rule paraphrased instead of quoted; a scorer re-implemented instead of imported without a reason. **Until Amendment A exists, write nothing but the plan.** |
| **G1** | the code (tests and module, before the artifact is built): two review mandates and the coordinator's own mutants | coordinator, Amendment B | any blocker; a surviving load-bearing mutant |
| **G2** | the build: from a clean committed tree (`git status` clean, the commit recorded in the artifact), `build`, the artifact committed by hand, T-regress green | implementer — **no token, no author action** | a dirty tree; any refused input |
| **G3** | every statement recomputed by the coordinator by an independent route from the raw files (not importing the module) | coordinator, Amendment C | any difference not explained |
| **G4** | the Return Packet | implementer | — |
| **G5** | merge review: two mandates and a depth-1 clone run | coordinator | any blocker |
After G5 the coordinator merges with §6's box ticked, closes `DEFERRED` 108 and 109, updates `README.md` if a sentence changes,
measures the CI skip ceiling by the registered route, and **then stops and waits for the author** (his ruling of 2026-10-08).

## 7. Definition of Done
CLAUDE.md §6, plus: the artifact committed and regenerated by its T-regress; every statement of §3.2 present under both definitions
(and both versions where it differs); every recorded value reproduced or explained; every tie-dependent integer with its range; the
first-time scores (§1.3) present with their rules; the suite run with the real-data environment and pasted; the mutations of §5
executed and committed.

## 8. Return Packet
`docs/returns/P5.2c.md` from `docs/returns/TEMPLATE.md`, with: the statement table (statement × tier × definition, verdicts, the
primary marked), the first-time scores, the comparison with the record, the ties, what the correction and Rule R change, what the
artifact does not say, the commands that rebuild it, and CLAUDE.md §8's AI-assistance record.

---

# ✅ AMENDMENT A — 2026-10-10, gate G0: PLAN APPROVED (`docs/plans/p5.2c.md` @ `1267715`, 557 lines) — F1–F15 and Q1–Q12 ruled, one premise of the plan refuted (F9/F11: the rank-biserial is IN), three corrections to this brief, four requirements added, and Q5's record found wrong

## A0 — Verdict and what the coordinator verified (by running commands, not from the plan's text)
**Approved. Build C1–C5 in the plan's order (A5).** Verified independently on 2026-10-10:
1. **F4, and it is wider than F4 says.** `python -P -c "import offline"` raises `ModuleNotFoundError` from the worktree **and from
   the main tree**: the venv's editable install maps no `offline`, so §4's command could not run from ANY tree. With
   `PYTHONPATH=<tree>` it imports `offline` and `agent` from `<tree>`, which is P5.2b's form
   (`offline/campaigns/p5_2b_correction.sh:116`). The brief's error (A1.1).
2. **F1's premise.** Python 3.12.13's built-in `sum` over floats is compensated (`sum([0.1]*10) == 1.0` is `True`). The two
   committed values of one quantity are on disk: `docs/data/p8_4b_rederivation.json` V4 `att_ours` `mean_difference`
   `39.564858203026375`; `docs/data/p5_1_grid.json` P1 `39.56485820302637`.
3. **F6, by a route that imports nothing from `offline/`** (json + `statistics`; per draw the mean over the five seeds; the CI
   `1.96·sd/√100`, `ddof=1`; every P8.4b `att_ours` read asserted equal to its eval file's `att_horizon`):
   - Q5a `bc_top10 − bc`, predicted > 0: `maxpressure` +514.4262 [+508.0807, +520.7716]; `fixedtime` **+0.2816 [−0.1093,
     +0.6725]**; `random` +60.9885 [+55.0077, +66.9693] → **NOT RESOLVED**.
   - Q5b `bc_top10_perix − bc_top10`, predicted < 0: `mappo1000` −557.3594 [−565.4166, −549.3021]; `maxpressure` −469.6936
     [−477.8062, −461.5809]; `fixedtime` **−0.2816 [−0.6725, +0.1093]**; `random` −49.1561 [−55.2578, −43.0544] → **NOT RESOLVED**.
   - Q5c `bc_top10_perix − bc`, predicted −, −, +, +: `mappo1000` **+23.2396 [+19.1155, +27.3637]** and `maxpressure` **+44.7326
     [+40.1822, +49.2831]**, both resolved against the predicted sign; `fixedtime` exactly zero; `random` +11.8324 [+6.7711,
     +16.8937] → **FAILED**.
   - Under `att_engine` the same three verdicts (Q5a `fixedtime` +0.2839 [−0.1103, +0.6781]; Q5b `mappo1000` −561.7580
     [−569.7300, −553.7860], `fixedtime` −0.2839 [−0.6781, +0.1103]; Q5c +23.3168 [+19.1606, +27.4730] and +44.8459 [+40.2609,
     +49.4309]).
   - **The registration did not predict a null at `fixedtime`.** `tier_sweep.PREDICTED_LEVELS` holds `bc` 222.1050, `bc_top10`
     1042.7531 and `bc_top10_perix` 230.5996 there, i.e. Q5a +820.65, Q5b −812.15, Q5c +8.49 — the values
     `docs/plans/p5.2.md:683-689` registers.
4. **Q0, same route:** `d4` +27.1004 [+23.8150, +30.3858] | `att_engine` +27.4534 [+24.1365, +30.7702]; `I` −12.4645 [−16.3575,
   −8.5714] | −12.5676 [−16.4943, −8.6409]; Q0a HELD, Q0b FAILED (negative), stop rule CONTINUE, `I < 0` on 70 of 100 draws,
   under both. These are the plan's §5 values.
5. **F12, same route:** both DT arms lie below the anchor with the CI excluding 0 on all four tiers under both definitions.
   Every non-DT arm lies above it with the CI excluding 0 on at least one tier: `bc` at `random`; `bc_top10` at `mappo1000`,
   `maxpressure` and `random`; `bc_top10_perix` at `random`; `iql` at `mappo1000` and `maxpressure`. This used P5.2's original
   IQL cell at `random`; the corrected one was not checked here.
6. **F9's and F11's shared premise is false.** `offline_baselines.paired_comparison` (`:2199`) pairs through `dt_gate._paired`
   → `dt_gate._per_draw_means` (`dt_gate.py:1199-1219`) and takes its interval from `mean_ci95(differences)`. **It IS the
   registered route, not a third one.** It is the function behind P5.1's committed P1 and P3 (`spatial_mixing._paired_block`,
   `:600`; `collapse_criterion`, `:747`). Those are the records the plan's V11 reproduces under `==` by the registered route,
   which is consistent only because the two routes are one.
7. **F14 and F15** are as the plan states: `docs/notes/A11_DRAFT.md:3`; `PROJECT_PLAN` §7, *"Filesystem-mutation barrier (added
   2026-07-31 …)"*; `offline/tier_sweep.py:6-8`.

**The plan's disclosure is accepted.** It says every statement was computed in scratch before the plan was written. No
convention ruled here moves a registered verdict: items 3–5 were recomputed independently, and F8–F10's verdicts agree under
both readings on these data. The packet's AI-assistance record states that order. G3 recomputes every statement by a route
that imports nothing from the module.

## A1 — Corrections to this brief
1. **§4's CLI is the plan's §6.4 form.** On `main` after the merge, from any directory:
   `PYTHONPATH=/home/filip/rltraffic /home/filip/rltraffic/.venv/bin/python -P -m offline.tier_statements build --repo-root
   /home/filip/rltraffic --output-root /home/filip/rltraffic/output --out docs/data/p5_2_statements.json`, and `check` with the
   same roots. The brief's `-P` without `PYTHONPATH` imported nothing (A0.1).
2. **§1.1:** Rule R's registration is `PREREGISTRATION.md` §12 (`:397`, A11(c) and (d)); `docs/notes/A11_DRAFT.md` is its
   drafting record. **§0:** the barrier rule is `PROJECT_PLAN` §7, not §5.
3. A1.1 and A1.2 are the coordinator's errors, logged in `PROJECT_PLAN` §8.

## A2 — Rulings: F1–F15 and Q1–Q12
- **Q1 / F1 — APPROVED.** One route everywhere, the registered one. T-reproduce-P8.4b takes its four-part exact form. No
  tolerance anywhere.
- **Q2 / F2 — APPROVED.** The tier-generic composer must be proven equal to `iql_correction.random_tier_statements` at `random`,
  field by field, for both `with_hard_subset` values, on synthetic and on real cells. It may add fields; it may never change one.
- **Q3 — APPROVED.** The local manifest reader, proven equal to `read_p5_2_json` and refusing the same tampering.
- **Q4 / F3 — APPROVED.**
- **Q5 — APPROVED.** `definition_dependent` is computed on the reading for Q3c and the leaders, and labelled as a reading.
- **Q6 / F5, F7 — APPROVED.** A3 adds Q5's records.
- **Q7 / F8, F10 — APPROVED.** F10's aggregate is labelled *both cells hold*, and its two sources are named: the bullet's
  registered tier (`maxpressure`, `p5.2.md:576-578`) and the table's second marking (`fixedtime`, `:497`).
- **Q7 / F9 — APPROVED WITH A CHANGE.** P5.1's three-valued outcome comes from `spatial_mixing.collapse_criterion` **imported
  unchanged** (A0.6). The module refuses unless that call's `mean_difference`, `ci95_low` and `ci95_high` equal its own under
  `==`. The two-valued verdict is primary, as the brief ruled.
- **Q8 / F11 — RULED DIFFERENTLY: THE RANK-BISERIAL IS IN.**
  - **Why.** D8 (`p5.2.md:315`) registers a rank-biserial for every pair. P5.1 reported one on every comparison
    (`docs/data/p5_1_grid.json`). P5.2's record omitted it, and the plan's reason for leaving it out rested on A0.6's false
    premise.
  - **What every two-arm contrast carries.** From `offline_baselines.paired_comparison(left, right)`, as the function returns
    them: `rank_biserial`, `median_difference`, `wins`, `losses`, `ties` and the `wilcoxon` block. The p-value is reported but
    never read by a verdict. The module refuses unless that call's `mean_difference`, `ci95_low` and `ci95_high` equal its own
    under `==`, which checks for free that the route is one route.
  - **Exact zeros.** Where `wilcoxon.n_used == 0`, the artifact says the rank-biserial is undefined and that the function
    returns 0.0.
  - **`I`** is a difference of two differences, not a pair of arms. It carries no rank-biserial, and the artifact says so.
  - **D15 is IN, as proposed.** Median and IQR per cell and definition over the cell's 500 episodes (`numpy.percentile`,
    `method="linear"`, named), and A5's `horizon_vehicle_count` mean, asserted equal to each eval file's `cell` block. D15's
    *"visibly not unimodal"* clause is a visual judgement this artifact does not make; `what_this_does_not_say` says so.
- **Q9 / F12 — IN,** as `descriptive_collapse_audit`, labelled *not registered*. It applies Q6's rule verbatim to `bc`,
  `bc_top10`, `bc_top10_perix` and `iql` on all four tiers, under both definitions, with both IQL cells at `random`. It quotes
  the README sentence it backs as text, without a line number (A4.2).
- **Q10 / F13 — APPROVED,** with A4.3.
- **Q11 and Q12 — APPROVED.**
- **F14 — accepted** (A1.2). **F15 — outside the fence.** Recorded as `DEFERRED` 112; the packet names it.

## A3 — Q5 in `comparison_with_the_record`, named in these terms
The block gets one row for each of three records:
- the packet's Q5 row (`docs/returns/P5.2.md:30`, *"CONFIRMED twice, with a predicted null on `fixedtime`"*);
- the packet's finding (`:572-573`, *"The load sorter is confirmed across tiers … a *predicted* null on `fixedtime`"*);
- GATE 2's G5 (`BRIEF_27`:1274-1281).

Each row carries:
- the registered verdicts of Q5a, Q5b and Q5c;
- the registered `fixedtime` predictions beside the measured values: +820.65, −812.15 and +8.49 (`p5.2.md:683-689`, equal to
  the differences of `PREDICTED_LEVELS`);
- one statement: the record scored Q5b on two of its four registered tiers, scored neither Q5a nor Q5c, and called predicted a
  null the registration did not predict.

The descriptive reading the record was after stays reportable per tier from Q5b's resolved tiers, labelled descriptive: the
global return filter collapses where the per-intersection one does not. GATE 2 is a coordinator ruling, so the error is the
coordinator's; it is logged in `PROJECT_PLAN` §8.

## A4 — Requirements added by the coordinator
1. **The code that runs must be the code the git block names.**
   - `build` and `check` refuse unless `offline.tier_statements` and every `offline` module it imports resolve
     (`Path(module.__file__).resolve()`) under the resolved `--repo-root`.
   - `provenance` records each module's path relative to the root, and its sha256.
   - Why: A0.1. With one tree on `PYTHONPATH` and another as `--repo-root`, the artifact would record a commit the code did not
     come from.
   - Test: a module outside the root is refused. Mutant M19: the check removed.
2. **No line-pinned quotation from a living document.**
   - `file:line` citations (in the module, the artifact and T-quotes) point only into files that are history:
     `docs/plans/p5.2.md`, `docs/plans/p5.1.md`, `docs/briefs/BRIEF_27_p5.2_tier_sweep.md`,
     `docs/briefs/BRIEF_44_p5.2b_iql_correction.md`, `docs/returns/P5.2.md`, and `PREREGISTRATION.md` (its amendments are
     appended at its end).
   - `docs/PROJECT_PLAN.md`, `README.md`, `docs/notes/DEFERRED.md`, `CLAUDE.md` and this brief are cited by section and dated
     heading only. T-quotes does not read them.
   - Why: the coordinator edits the plan at every merge, and the author will edit the README while writing. An ungated test
     that turns CI red because a living document gained a line certifies nothing. (The plan's V15 and V16 cite
     `PROJECT_PLAN` by line.)
3. **The artifact must pin what it read.**
   - `provenance` carries one sha256 over the sorted (path relative to `--output-root`, file sha256) pairs of every P8.4b row
     read, and one over P5.2b's corrected rows.
   - P8.4b has no per-file manifest, so this is the only content record of the `att_engine` values the paper quotes.
   - T-inputs-real recomputes both.
4. **Mutations.** M1–M18 as planned, plus:
   - M19: A4.1's check removed;
   - M20: the `paired_comparison` equality refusal removed, killed by a test that perturbs that call's `mean_difference` by one
     ulp.

   Each is executed, committed with its output, and KILLED.

## A5 — Next
The implementer builds, in the plan's order:
1. C1–C5 (tests first, the red run pasted).
2. The whole suite with `RLTRAFFIC_OUTPUT_ROOT=/home/filip/rltraffic/output`.
3. `scripts/check_test_hygiene.sh` and `scripts/check_english.sh` on the new files.
4. The mutation script's log and results.

All of it is committed on the branch and NOT pushed: the coordinator reads it from the shared repository. Then stop and say
**"P5.2c G1 ready"** — gate G1, Amendment B. **No artifact is built (C7) before Amendment B exists on `main`.**

---

# ⚠️ AMENDMENT B — 2026-10-11, gate G1: FIX FIRST (commit C6) — the code scores every statement correctly on the real data, but its tests do not hold where no committed record anchors a number: ten of the coordinator's seventeen mutants survive, and A4.1's guard has a hole

The review record is `docs/reviews/P5.2c_G1.md`: both findings files verbatim, and the coordinator's mutant harness
`docs/reviews/P5.2c_G1/coord_mutants.py` with its log and results. Every mutant string below is in that harness.

## B0 — Verdict, and what was checked by running it
1. **The new test file on a clean `git archive` export of `719d753`**, run by the coordinator: 114 passed, 1 skipped
   (T-regress, before G2), in 12.6 s.
2. **R1 (the scorers against the registered text): PASS-WITH-NOTES** — 0 blockers, 1 MAJOR (F-14), 4 MINOR (F-3, F-4, F-10,
   F-11). Its real-data probes match the plan's §5 and Amendment A0.3–A0.5 on every verdict.
3. **R2 (inputs, the record, the artifact, the barrier, the CLI, test strength, `mutate.py`): PASS-WITH-NOTES** — 0
   blockers, 1 MAJOR (F3.1), 7 MINOR (F6.2, F6.3, F6.4, F7.2, F8.1, F11.1, F11.2). It also verified:
   - all nine pins, recomputed;
   - both A4.3 row digests, by the artifact's own shell construction;
   - the in-memory artifact, byte-identical across five processes and hash seeds;
   - `check` on the real inputs: exit 0, nothing written;
   - the implementer's 31 mutants, sound, spot-checked against their log.
4. **The coordinator's 17 mutants: 7 KILLED, 10 SURVIVED.**
   - Killed: CM2, CM3, CM5, CM6, CM11, CM13, CM15. CM2 and CM13 fell to the build's own comparison with the record
     (Note N's interval; the packet's *"Three contrasts are exactly `0.0000`"*), which is a genuine kill.
   - **A surviving load-bearing mutant stops the task (§6), so: FIX FIRST.**
   - The pattern is the finding. The tests are strong wherever a committed record anchors the value. They are weak exactly
     where the paper's new numbers live: the first-time aggregates, the D9 per-seed blocks under `att_engine`, the leaders,
     and the collapse audit.
5. **No reported number or verdict is wrong.** The code read correctly to both reviewers and to the coordinator. What this
   round fixes is whether the tests would notice if it were not.

## B1 — The ten survivors: each gets a test that kills it, with the expected answer computed by the test's own route
Each test computes its expected answer from the arrays: raw `numpy` / `statistics`, the registered rule written out in the
test. It never calls the scorer or a helper of the module. Where real data discriminate, a gated assertion goes beside the
synthetic one.

| mutant | what survives today | the test required |
|---|---|---|
| CM17 | Q3b's aggregate ignoring `fixedtime`: a **registered verdict flips NOT RESOLVED → HELD** unnoticed | synthetic HELD / NOT RESOLVED / HELD → aggregate NOT RESOLVED by the test's own conjunction; gated: Q3b.aggregate NOT RESOLVED on the real data, both definitions |
| CM9 | every D9 per-seed block computed on `att_ours` whatever the definition | a synthetic contrast whose per-seed values differ between definitions, each seed's value by the test's own means; gated: one real `att_engine` per-seed block against the raw rows |
| CM7 | the leaders' per-seed first places read on `att_ours` under `att_engine` | gated: at `maxpressure` under `att_engine`, seed 303's first place is `dt_nomix` and `bc` on the other four; under `att_ours`, `bc` on all five — by the test's own per-seed means |
| CM16 | the leader's lead called resolved when the CI's LOW end is below zero | synthetic straddling lead → "does not resolve"; gated: `maxpressure`'s lead resolves under `att_ours` and not under `att_engine` (the README's sentence), by the test's own CI |
| CM8 | the collapse audit's `holds` true if ANY non-DT arm collapses somewhere | a synthetic book in which one non-DT arm never collapses → `holds` false |
| CM10 | Q6 COLLAPSED iff the CI's HIGH end is above zero | a synthetic DT arm whose CI against the anchor straddles zero → NOT COLLAPSED, the CI by the test's own route |
| CM12 | Q2a crediting a predicted first place that is TIED for first | a synthetic tier where the predicted first ties for first → Q2a FAILED |
| CM4 | Q1b scored with `LEVEL_BAND` (±30 %) instead of `ANCHOR_BAND` (±10 %) | a synthetic behaviour cell 20 % off its predicted level → FAILED by the test's own `abs(m − p) / p` |
| CM14 | Q4b's chain scored non-strictly | a chain with two equal rungs → FAILED, not inverted |
| CM1 | Q0b's exact-zero branch is untested (R1 F-14) | after B2.1, the reversed mutant (an exactly-zero I scored NOT RESOLVED) must be KILLED by a synthetic exactly-zero I |

**All seventeen coordinator mutants join `mutate.py`** (CM1 reversed per B2.1) beside M1–M31 and the new mutants of B2. Every
one must be KILLED in the run of record.

## B2 — Rulings
1. **Q0b on an exactly-zero I (R1 F-4; the implementer's deviation 3): the registered text wins.**
   - Q0b is HELD iff the CI of `mean(I)` includes zero (`docs/plans/p5.2.md:523-524`). A CI of [0, 0] includes zero, so an
     exactly-zero I is HELD, with *"exactly zero (identical episodes)"* beside it.
   - §3.3's exact-zero sentence (*"NOT RESOLVED for an interval rule"*) was written for predicted-sign rules, where the
     registered text itself yields NOT RESOLVED because [0, 0] lies on neither side. It does not override an inclusion rule.
     The sentence was the coordinator's, and too broad.
   - **The principle, now explicit: the registered rule decides the verdict; an exact zero is always labelled, and is never
     offered as evidence for a predicted sign.**
   - **Q3b's sub-prediction (R1 F-11):** `matched` stays as the registered rule gives it. The block gains `exactly_zero` and a
     reading saying that at `fixedtime` the match comes from identical episodes.
2. **The composite per-seed block (R1 F-3).**
   - Compute it with the imported worker (`tier_sweep.per_seed_block` → `spatial_mixing.per_seed_advantages`) over composite
     rows: per `(seed, draw)`, `dt_spatial_h4 − dt_nomix_h4` on the left and `dt_spatial − dt_nomix` on the right. The
     block then carries P5.1's full schema.
   - The brief's formula is the same quantity by linearity; the test computes it by its own loop.
   - If composite rows cannot pass the worker's checks unchanged, keep the restatement, add the two missing fields
     (`n_seeds_favouring_left`, `reverses_on_n_seeds`) by P5.1's definitions, and prove them equal to the worker on ordinary
     cells.
3. **A pooled tie for first place (R1 F-10).** Add `measured_first_under_rule` (null when the pooled first place is tied) and
   a reversal list computed against the pooled SET of first places. Both go beside P5.2b's fields, which stay so the
   random-tier field-by-field equality still holds. Test: a synthetic pooled tie for first.
4. **A4.1 (R2 F3.1, MAJOR).**
   - `assert_code_under_root` also requires the running code to resolve under `<root>/offline`: `__main__`'s `__file__` when
     the module runs as a script, and the defining module's own `__file__`.
   - `provenance.code` hashes the file that RUNS for `offline.tier_statements`.
   - Test: a subprocess runs the file by path from a copy of the tree, with `--repo-root` naming another tree, and is refused.
   - Mutant: the new check removed, KILLED.
5. **The "arithmetic route alone" explanation (R2 F6.2) is verified, not asserted.** The build refuses unless each difference
   between this artifact's value and P8.4b's V4/V5 or the stop-rule record is at most `4 * math.ulp(max(|a|, |b|))`. Test: a
   synthetic record 5 ulps away is refused.
6. **A3's GATE 2 row (R2 F6.3)** cites `BRIEF_27`:1274-1281, and quotes the sentence that is the error: *"the `fixedtime` null
   is *predicted* by the mechanism"*. T-quotes holds it.
7. **Q6 in `never_scored` (R2 F6.4)** is labelled *"Q6's numbers: never scored; the verdict was recorded
   (`docs/returns/P5.2.md:31`)"*.
8. **`git.code_root` (R2 F7.2)** stays as `compute_latency._git_provenance` returns it: absolute, which is P5.2b's committed
   precedent. T-canonical's exemption gets a comment saying why, and the packet discloses it.
9. **The out path (R2 F8.1, F11.1).**
   - Refuse when the UNRESOLVED candidate is a symlink, dangling or not, or exists; test it with a dangling symlink.
   - The input-equality guard gets a test that reaches it, asserting its own message (it runs before the exists check). If
     no input can reach it without existing, remove it as redundant and say why.
   - `os.link` in place of the plan's `os.replace` (R2 F8.2) is ACCEPTED: it also refuses a file created between the check
     and the write. The packet discloses it.
10. **The constant clause (R2 F11.2):** `7 <= 8 <= 9` becomes the recorded 8, parsed from its record row, asserted inside the
    range.
11. **T-quotes (R2 F4.1).** Assert that every `Q5_RECORDS` quote's file is in `CITABLE_FILES`. Widen the scan to catch
    `` `file`:N `` and extension-less upper-case names such as `PROJECT_PLAN:N`.
12. **The tests R1 called theatre.**
    - `test_t_q0` recomputes I's sd and CI with `statistics.stdev` and `1.96 · sd / √n` under `==`, and derives Q0a, Q0b
      and the stop rule from those.
    - Q6's expected verdict comes from the test's own CI (`:643`).
    - Q4a's aggregate comes from the test's own per-tier intervals (`:581-583`).
    - The IQL levels come from the test's own mean (`:707-712`).
13. **`definition_dependent` compares verdicts (R1 F-12).** No change. `what_this_does_not_say` gains one sentence: Q0b's
    sign and Q4b's chain shape are reported per definition beside the flag, and the flag does not compare them.
14. **The implementer's G1 notes.**
    - P8.4b's `campaign_manifest.json` and `CAMPAIGN_COMPLETE` are listed in `provenance.inputs` with their sha256. YES.
    - T-regress's skip message says "the plan's C7".
    - Deviations 1, 2 and 5 are ACCEPTED. Deviation 3 is overruled by B2.1, and deviation 4 is superseded by B2.2.
    - The tests-first lapse (deviation 6, shown sound by M22–M27) and the amended local commit message (deviation 7) are
      accepted. The packet discloses both, and the first whole-suite run's 36 failures.

## B3 — Commits, and the next gate
- **C6 is a series of commits of ≤ ~2 source files each:**
  1. **Tests first.** The tests whose behaviour B2 changes (B2.1, B2.3, B2.4, B2.5, B2.6, B2.7, B2.9, and B2.11's
     assertion) are run red on `719d753` and the red run is pasted. B1's survivor-killing tests are green on arrival; their
     strength is their mutants.
  2. **The module's fixes.**
  3. **`mutate.py` extended:**
     - every mutant of B1 and B2;
     - the run of record on an export of the code commit, with every mutant KILLED;
     - the whole suite with `RLTRAFFIC_OUTPUT_ROOT=/home/filip/rltraffic/output`;
     - `check_test_hygiene.sh`, `check_english.sh` and the frozen-file guard.
- **Committed on the branch, NOT pushed.** Then stop and say **"P5.2c G1.1 ready"**.
- **Gate G1.1:** the coordinator re-runs his seventeen mutants with his own harness on the new commit and reads the diff.
  The result is Amendment B.1.
- **No artifact is built before Amendment B.1 exists on `main`.**

---

# ⚠️ AMENDMENT B.1 — 2026-10-13, gate G1.1: FIX FIRST, folded into the build (commit C6c, then G2 with no round trip between) — C6 meets every B2 ruling and kills all eighteen adapted G1 mutants, but four of the coordinator's fourteen new mutants survive, and one of them changes the real artifact's text unnoticed

The review record is `docs/reviews/P5.2c_G1_1.md`. The harness, its log, its results and both real-data probes are in
`docs/reviews/P5.2c_G1_1/`.

## B.1.0 — What the coordinator checked, by running it
1. **The branch.**
   - `task/p5.2c-statements` is at `ca27b48`, merge base `d64e651` (Amendment B).
   - C6 is `2f62449` (tests first), `d982bc8` (the fixes), `d90c3b6` (an unasked guard removed), `5bbb540` (`mutate.py`
     to 70 mutants) and `ca27b48` (evidence).
   - `ca27b48` differs from `5bbb540` in evidence files only. The module's sha256 is `fd1ced2d…` at both, and the test
     file's is `9c2d553f…`.
   - No `Co-Authored-By` or `Claude-Session` line in `719d753..ca27b48`, and no frozen path in its diff.
2. **The test file on a clean `git archive` export of `5bbb540`**, with `RLTRAFFIC_OUTPUT_ROOT`: 140 passed, 1 skipped
   (T-regress, before G2), in 21.9 s.
3. **The diff, read in full:** the module (163 changed lines: 119 added, 44 removed) and the tests (641 changed lines: 607
   added, 34 removed).
   - Every B2 ruling is met in the form the evidence README describes.
   - Every B1 test computes its expected answer from the rows, by `statistics` / `numpy` and the rule written out in the
     test, never by the scorer.
   - B2.6's quote was checked against `BRIEF_27`:1274-1281 by reading the file; B2.7's label against
     `docs/returns/P5.2.md:31`.
4. **The coordinator's harness** (`docs/reviews/P5.2c_G1_1/coord_mutants_g11.py`), 32 mutants:
   - The seventeen G1 mutants are imported verbatim from `docs/reviews/P5.2c_G1/coord_mutants.py`.
     - CM1 is reversed per B2.1.
     - CM15's code is gone, so the coordinator re-aimed it twice at the composite rows: CM15a flips d4's inner difference;
       CM15b drops one `(seed, draw)`. These are not `mutate.py`'s strings.
   - Fourteen new mutants aim at the B2 rulings. They were written without reading `mutate.py`'s strings.
   - **Results: control 140 passed, 1 skipped; 28 KILLED, 4 SURVIVED, 0 errors.**
     - The eighteen G1-derived mutants: all KILLED.
     - New and KILLED (10): CB3a, CB3b (B2.3); CB4a, CB4b (B2.4); CB5a, CB5b (B2.5); CB6 (B2.6); CB9a (B2.9); CB14a,
       CB14b (B2.14).
     - New and **SURVIVED (4): CB2, CB1a, CB5c, CB0** — B.1.1.
5. **Real data, by an independent route** (`verify_i_per_seed.py`: `json` + `statistics`, nothing from `offline/`). I's
   per-seed values, `mean_draws(s4 − n4) − mean_draws(s1 − n1)`:

   | seed | 101 | 202 | 303 | 404 | 505 | pooled |
   |---|---|---|---|---|---|---|
   | `att_ours` | +45.8935 | −38.7945 | −33.8405 | −13.8980 | −21.6828 | −12.4645 |
   | `att_engine` | +46.4239 | −39.1119 | −34.1478 | −14.1020 | −21.9003 | −12.5676 |

   - The module's block on the same data (`probe_q0b.py`, an in-memory build from the `5bbb540` export) equals these at
     four decimals under both definitions.
   - Its qualifier reads *"d4 vs d1 reverses on 1 of 5 training seeds (101)"* under both definitions. That is correct today.
   - **For the paper (not a change; the artifact already carries it):** Q0b is FAILED, so an interaction exists. But I is
     negative on four seeds and about +46 on seed 101, and the pooled CI [−16.36, −8.57] averages that away. The
     qualifier travels with the verdict.
6. **The implementer's run of record** (`c6_mutation_log.txt`) records 70 of 70 KILLED on an export of `5bbb540`, with
   `mutate.py` at sha256 `02d182f0…`, equal to the committed file's. The coordinator did not re-run it; his 32 mutants are
   the independent check.
7. **CI on `main`:** green at `d64e651` (run `38163661165`).

## B.1.1 — The four survivors, and the test each requires (C6c)
| mutant | what survives | load-bearing? | the test required |
|---|---|---|---|
| **CB2** | I's per-seed worker called with the names swapped (`left_name="d1", right_name="d4"`). The real artifact's Q0b qualifier would then read *"d1 vs d4 reverses …"* beside per-seed values that are d4 − d1 | **YES.** It changes the sentence that must travel with Q0b's verdict, in the real artifact, and no test notices | **Synthetic:** a book in which I reverses on exactly one seed. Q0b's `seed_reversal_qualifier` begins with the sentence the test builds from its own per-seed loop: *"⚠️ ORDERING NOT ROBUST ACROSS SEEDS: d4 vs d1 reverses on {n} of {N} training seeds ({seeds})"*. **Gated:** on the real data, under both definitions, Q0b's `by_seed` equals B.1.0(5)'s values at four decimals, and its qualifier begins *"⚠️ ORDERING NOT ROBUST ACROSS SEEDS: d4 vs d1 reverses on 1 of 5 training seeds (101)"* |
| **CB1a** | Q3b's sub-prediction reading says *"identical episodes"* whatever the identity check finds | No: the real `fixedtime` episodes are identical. But the label states a fact that no test checks | A synthetic tier where `dt_nomix − behaviour` is exactly zero on every draw from episodes that differ: equal per-draw means, differing `(seed, draw)` values, dyadic so that the sums are exact. The test asserts both facts from the rows. The sub-prediction's reading must then name episodes that are not identical, and `discriminability.identical` must be False |
| **CB5c** | P8.4b's ulp check applied to `mean_difference` only, leaving both CI ends unchecked | No: the real differences are 0–2 ulps by C6's evidence README, and the coordinator's run, which builds on the real data, passed the bound. But B2.5 says *each difference*, and the tests move only the mean | Both B2.5 tests parametrised over the three values (`mean_difference`, `ci95_low`, `ci95_high`), for V4, V5 and the stop-rule record: each value moved 4 ulps is accepted, and 5 ulps is refused |
| **CB0** | Q0b's inclusion rule written with strict inequalities, so a CI touching zero is excluded | No: the real I's CI is [−16.36, −8.57]. The boundary has measure zero, but the registered word is *includes* | A Q0b input whose CI touches zero at one end: (0.0, +1.0) and (−1.0, 0.0) give HELD; (+2⁻¹⁰, +1.0) gives FAILED. Each expected verdict comes from the test's own `low <= 0.0 <= high`, with `exactly_zero` False. The implementer may build it by substituting `composite_contrast` in the module's namespace. If `score_q0` refuses the substitution, test the rule where it is computed and say where |

CB2 is load-bearing, so §6's rule applies: **FIX FIRST.** All four are test gaps over code that is correct today
(B.1.0(5)).

## B.1.2 — Rulings on C6's open items and disclosures
1. **Open question 1, Q0b's exact-zero label: REFINED (C6c, a module change).**
   - The literal *"exactly zero (identical episodes)"* was the coordinator's text in B2.1. It states a fact the code
     never checks, while the artifact's own `definitions.exact_zero` says the label is checked. **This is the coordinator's
     error, the same class as CB1a.**
   - **The rule:**
     - I's block gains `identical_pairs`, `{"d4": bool, "d1": bool}`. Each is `att_rederivation.contrast_discriminability`'s
       `identical` over that pair's per-episode values under the definition, called as `contrast()` calls it.
     - I's block also gains `identical`, True iff both are True.
     - Q0b's exact-zero reading begins *"exactly zero (identical episodes)"* only when `identical` is True. Otherwise it
       begins *"exactly zero on every draw, from episodes that are not identical (d4 equals d1 draw by draw)"*.
     - The verdict stays HELD (B2.1).
   - **Tests:**
     - The CM1 test, where both pairs are identical, is unchanged.
     - New: a synthetic I that is exactly zero because the horizon changes nothing (s4's episodes ≡ s1's, n4's ≡ n1's,
       s ≠ n). It is HELD with the second reading, and `identical` is False.
   - **Mutants:** the identity forced True, and forced False. Both KILLED.
2. **Disclosure 1's unnamed test change** (the collapse audit's `holds`, now computed by the test's own rule): ACCEPTED.
   It turns a type check into a value check, and it is disclosed.
3. **Disclosure 2** (CM13's synthetic test and CM6v, written after the code): ACCEPTED. Both kill mutants that were
   already killed, by a stronger route.
4. **Disclosure 3** (the guard added at `d982bc8` and removed at `d90c3b6`): ACCEPTED, for the reason given.
5. **Disclosures 5–7** (B2.4's two checks coincide when the module is `__main__`; B2.6's quote is contiguous; B2.3's
   reversal list is against the pooled set): ACCEPTED as written.
6. **Open item 2** (the CM1 and CM15 strings): done with the coordinator's own strings (B.1.0(4)). **Open item 3**
   (`DEFERRED` 112): stays outside the fence.

## B.1.3 — Now: C6c, then G2, with no round trip between
**Why the build need not wait for a G1.2 round trip:** the four survivors are test gaps over correct code, and the one
module change (B.1.2(1)) is in a branch the real data never reach, since I ≠ 0 under both definitions. **The coordinator
verifies C6c as the first step of G3.** A survivor there sends the task back, with a rebuild.
1. `git merge --no-edit main`.
2. **C6c**, in commits of ≤ ~2 source files each:
   1. **Tests first:** B.1.1's four tests and B.1.2(1)'s, run on `5bbb540`.
      - They are red where the module changes (B.1.2(1)) and green on arrival elsewhere.
      - A scratch pre-check shows that each green test kills its mutant on the unchanged module, as at C6.
      - The run is pasted.
   2. **The module change** of B.1.2(1).
   3. **`mutate.py`** gains CB0, CB1a, CB2 and CB5c (the strings are in `docs/reviews/P5.2c_G1_1/coord_mutants_g11.py`),
      plus B.1.2(1)'s two.
      - The run of record is on an export of the code commit, with every mutant KILLED.
      - Then the test file, `check_test_hygiene.sh`, `check_english.sh` and the frozen-file guard.
3. **G2, the plan's C7.** On the branch, A1.1's form names the worktree, not `main`:
   `cd /tmp && PYTHONPATH=/home/filip/rltraffic-p52c /home/filip/rltraffic/.venv/bin/python -P -m offline.tier_statements build --repo-root /home/filip/rltraffic-p52c --output-root /home/filip/rltraffic/output --out docs/data/p5_2_statements.json`
   1. Run it from a clean committed tree, with `git status` empty before and after.
   2. Commit `docs/data/p5_2_statements.json` by hand, alone, with the commit it records and its sha256 and byte count in
      the message.
   3. At that commit, run:
      - the whole suite with `RLTRAFFIC_OUTPUT_ROOT=/home/filip/rltraffic/output`. T-regress now runs and must pass, with
        0 failed;
      - `check` in the same form (exit 0).
   4. Paste both into the evidence README and commit.
4. **Do not push.** Stop and say **"P5.2c G2 ready"**.
- **G3 then begins** with the coordinator's harness re-run on C6c's code commit: all 32 mutants, plus B.1.2(1)'s two. Then
  comes the independent recompute of every statement (Amendment C). G4, the packet, waits for Amendment C.
