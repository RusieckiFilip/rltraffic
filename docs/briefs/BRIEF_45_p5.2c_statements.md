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
