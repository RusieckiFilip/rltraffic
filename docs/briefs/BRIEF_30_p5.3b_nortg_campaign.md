# BRIEF_30 — P5.3b: does removing the return prompt cost anything, and is the cost data-dependent?

**Task id:** `P5.3b` · **Branch:** `task/p5.3b-nortg-campaign` · **Issued:** 2026-08-26
**Mode:** Explore → Plan → Code → Commit, with human gates. **Start in plan mode.**
**Compute:** GPU training, **15 new cells only**. The `dt` column is reused, never retrained (except
once, as a control — §4.1).

> **Read `BRIEF_28` + amendments A and B, `docs/plans/p5.3a.md`, `docs/reviews/P5.3a.md` and
> `docs/returns/P5.3a.md` first.** Where this brief disagrees with any of them it wins; where it
> disagrees with the **repo**, the repo wins and you flag it (CLAUDE.md §2).

---

## 1. Why this task exists, and how its question changed

P5.3a answered the question P5.3 was created to ask, and the answer moved the goalposts:

- **The prompt is a strong lever on the POLICY.** Paired over P4.3's own per-episode data,
  n = 500: **499 of 500 episodes move** between target `0` and target `−13000`, mean absolute
  **2.1281 ATT**, max **11.58**.
- **And a weak lever on MEAN QUALITY:** the mean of those movements is **+0.9026**, in the registered
  direction.
- **So `A9` has an object** — the model did learn to condition on the token. ⚠️ **What is unproven is
  that the conditioning is worth anything.**

**That is this task's question, and it is sharper than the one in §6:** *not "is there a knob?" — there
is — but **"does removing the knob cost anything, and does the cost track the data's return spread?"***

§1b's **C4** supplies the hypothesis: conditioning is identifiable only where the corpus carries
between-episode return spread. **P5.3a measured that spread for the first time (row B) and it varies
57× across our eight tiers**, so the hypothesis is finally testable.

---

## 2. The tier set — fixed by a rule registered BEFORE the numbers existed

`BRIEF_28` §9 registered: *"(i) `mappo1000`, the headline tier; (ii) the tier with the largest measured
between-episode scaled-RTG sd; (iii) the tier with the smallest."* **Row B resolves it:**

| tier | row B (pooled) | `dt` ATT | role |
|---|---|---|---|
| **`mix50`** | **0.2725** | 107.7026 | **widest spread** — where the prompt should matter most |
| `mappo1000` | 0.0391 | 104.9558 | the declared headline tier; A6's δ and P4.3's sweep live here |
| **`random`** | **0.0048** | 420.3764 | **narrowest** — and P5.3a measured its DT at **0 of 7200 flips** |

**3 tiers × 5 seeds = 15 new `dt_nortg` cells.** Nothing else trains.

⚠️ **The ATT scales differ by 4×** (105 · 108 · 420), which is half the reason §5 forbids a single
equivalence threshold.

---

## 3. Registered predictions — written before any P5.3b number exists

| # | Prediction | Falsifiable how |
|---|---|---|
| **Q1** | **ORDERING, by endpoints:** the paired absolute difference between the `dt` and `dt_nortg` arms is **largest on `mix50`** and **smallest on `random`**. | Either endpoint out of place falsifies it. **Endpoints, never a trend** — §1b's R3 was falsified on exactly a monotonicity claim, and the standing instruction is to register endpoints. |
| **Q2** | ⭐ **`random` is a NULL CONTROL, predicted from an independent instrument.** P5.3a measured `random`'s conditioned DT at `flip_rate = 0.000000`, 0 of 7200, on every intervention. If the token carries nothing there, training without it should cost nothing there: **`dt − dt_nortg` on `random` has a CI containing 0.** | A large, CI-excluding difference on `random` **indicts the wiring before it indicts the science** — same direction of inference as A8's `fixedtime` prediction in P5.3a, and registered now so it cannot be reversed later. |
| **Q3** | **Arm validity, mechanical:** every `dt_nortg` checkpoint shows `flip_rate` **exactly 0.0** under P5.3a's probe on all 12 interventions, and carries `rtg_mode == "zero"` in its checkpoint config. | Anything non-zero means `rtg_mode` did not reach the training path. **This is a gate, not a result.** |

⚠️ **Q2 and Q3 are different claims and must not be conflated.** Q3 says the *trained* model ignores a
token it was never shown. Q2 says training without the token cost nothing *on that tier*.

---

## 4. What to build

### 4.1 ⭐ THE LOAD-BEARING TEST — threading `rtg_mode` through `train_dt` must not move the conditioned path

`offline/dt_gate.py:749-754` constructs `DTConfig` with four arguments and **has no `rtg_mode`
parameter** (MJ-5). Adding one is this task's first commit — and it touches the function that trained
**every DT number in the paper**.

> **Retrain ONE committed `dt` cell through the modified `train_dt` and assert its canonical
> `state_dict` digest equals the committed value** in `p4_6_training.json`
> (`canonical_digest_of`, `offline/method_tier_grid.py:1180`). `mappo500` or `maxpressure` seed 101 —
> **not `random`** (§4.4). ~3 minutes at 40,000 steps.
>
> **Then mutate, both directions, and paste both failures:**
> 1. force `rtg_mode="zero"` inside `train_dt` → **must fail**;
> 2. make the new parameter default to `"zero"` → **must fail**.
>
> **If either leaves the suite green, the task is `BLOCKED`.** Do not weaken the test.

🚨 **Why this test and not P5.3a's:** `docs/reviews/P5.3a.md` **MJ-3** measured that the ATT-identity
test is **structurally blind to every config field that only matters at training time** — forcing
`dropout=0.9` passed it, because dropout is inert under `eval()`. **P5.3b trains, so that blind spot
is now live, and a digest over weights is the instrument that sees it.**

### 4.2 The campaign

Reuse P4.6/P4.7's protocol **read from the committed declarations, not restated**: seeds
`[101,202,303,404,505]`, `declared_gradient_steps = 40000`, held-out draws **1000–1099**, the same
`env_settings`, the same corpus dirs, the same `target_rtg`/`rtg_scale` per tier.
**Gate 1 before consuming anything**: canonical digest **and** a `SHA256SUMS_p4_6.txt` /
`SHA256SUMS_p4_7.txt` check **at consumption** (`BRIEF_27` B3(a)).

⚠️ **`mix50`'s `NormalizationStats` is fitted on the UNION of all three mixtures' directories** —
identical for `mix33`/`mix50`/`mix67`, count 216000, std 13155.3172 (P5.3a §8.4). **That is how P4.7
trained it, so reuse is consistent — but the artifact must say so**, or a reader will take the summary
for a property of `mix50`.

### 4.3 Statistics — and the P5.2 lesson is binding

Paired per-draw over **shared draw ids** (A5), per tier: mean difference, 95 % CI, **Wilcoxon**,
**rank-biserial**, wins/losses/ties, `n_shared_draws`, and the **per-seed reversal count**.

🚨 **`docs/reviews/P5.2.md` MJ-4: that packet's docstring claimed the protocol was reused from
`dt_gate._paired`, `wilcoxon_signed_rank` and `offline_baselines.paired_comparison` — none of which
was imported or called.** **Import them and call them.** A test must fail if the qualifier is removed.

### 4.4 Arm validity via P5.3a's probe

Run `offline/rtg_ablation.py probe` over the 15 new checkpoints — **141 s for 40 cells in P5.3a, so
this is about a minute.** Q3 above is the acceptance criterion. ⚠️ **`random` may not be the §4.1
control cell**, because its conditioned DT is already RTG-inert, so a control there would pass whether
or not `rtg_mode` reached the trainer.

---

## 5. ⛔ NO EQUIVALENCE THRESHOLD. Report the measurement and the ordering

`PREREGISTRATION` A7 withdrew the per-tier δ rule on 2026-08-25 because it spanned **eleven orders of
magnitude** and could not return one of its answers on part of its domain. **Do not reinvent it.**

> **Registered: P5.3b issues NO equivalence verdict.** It reports the paired difference, its CI, the
> per-seed reversals, and the tier's own `dt` ATT beside it so the reader can scale it. **Q1's ordering
> and Q2's null control are threshold-free and are the registered claims.**
> ⚠️ **A CI containing 0 is a failure to reject, never a demonstration of equivalence** — A6's own
> words. Write it that way.

---

## 6. Scope fence

1. ⛔ **No tier beyond the three.** The rule chose them before the numbers existed; adding a fourth
   after seeing results is what P5.3a correctly refused.
2. ⛔ **No retraining of the `dt` column** except §4.1's single control cell.
3. ⛔ **No second ablation mode.** `"zero"` only; `rtg_shuffled` remains unregistered.
4. ⛔ **No context-length K** — P5.3c.
5. ⛔ **No editing `offline/method_tier_grid.py`'s `METHODS`** (`:1701` — comparison enumeration).
6. ⛔ **Never write into or delete from** any existing `output/<campaign>/` directory. `output/p5_3b/`
   is yours. **Write `output/SHA256SUMS_p5_3b.txt` at the end** — every campaign has one, and
   `DEFERRED` 56 is what happens when one does not.
7. ⛔ **Not fixing** P5.3a's carried items (cell 2's non-discrimination, MJ-1's undemonstrated chunked
   case, M2/M5/M6) or `DEFERRED` 56.
8. ⚠️ **`docs/data/p4_dt_config.json`'s `architecture` block is literally `payload["config"]`
   (`dt_gate.py:1007`), so your checkpoints write a 9-key config where P4's has 8.** Expected; say so.

---

## 7. Gates, in order

| gate | what it proves | stops the task? |
|---|---|---|
| **0** | **A one-seed timing probe before any campaign commitment** (Amendment R2 — no armchair costing; the coordinator's own "≈52 h" is the cautionary case) | no, but no commitment without it |
| **1** | Reused `dt` checkpoints verified by canonical digest **and** manifest, at consumption | **yes** |
| **2** | §4.1: one committed `dt` cell retrains to its committed digest through the modified `train_dt`, **and both mutations fail** | **yes** |
| **3** | Q3: every `dt_nortg` cell probes to `flip_rate` exactly 0 and carries `rtg_mode == "zero"` | **yes** |

---

## 8. Definition of Done

- [ ] 15 cells trained, evaluated on draws 1000–1099, artifact `docs/data/p5_3b_nortg.json`
- [ ] **Gate 2 passed and BOTH mutations failed, with both failures pasted in full**
- [ ] Gate 3 mechanical on all 15
- [ ] Q1 and Q2 scored against §3 as registered; **no equivalence verdict anywhere**
- [ ] Paired statistics actually imported and called, with a test that fails if the qualifier is removed
- [ ] `output/SHA256SUMS_p5_3b.txt` written; every reused manifest re-verified after the run
- [ ] Whole suite green, pinned, real tail pasted; guards **16** / English / frozen
- [ ] Committed on the task branch; Return Packet at `docs/returns/P5.3b.md`
- [ ] **§6's `P5.3b` checkbox ticked in the merge commit**; **AI-assistance record** written as you go
- [ ] Timing measured and reported per cell — P5.3c needs it

**Review:** critical-path. `contract-reviewer` before merge, mutation evidence not reading.

---

## 9. What the next reader must not be allowed to believe

State these in the packet, plainly:

1. **A null on a tier is not "the prompt is useless".** It is *"removing the prompt did not change mean
   held-out ATT on this corpus, at this budget, at 5 seeds"*.
2. **The three tiers differ in more than return spread** — composition, state coverage and data quality
   move together. **Row B is an axis we can measure, not one we can isolate**, and Q1's ordering is
   therefore consistent-with rather than evidence-for C4.
3. **`random`'s DT is 4× worse in ATT than the other two** (420.38 against ~105). A difference measured
   there is not comparable in magnitude to one measured on `mappo1000`.

---

# ✅ AMENDMENT A — 2026-08-27, ruled at the plan gate

**Verdict: APPROVED TO CODE.** Plan pinned at blob `84e4d2c0187ee4f7fd3fb455c2926c19a02ceed2`
(`docs/plans/p5.3b.md` on `task/p5.3b-nortg-campaign` @ `cc859eb`). ⚠️ **Bring this amendment onto the
branch with `git merge main`, NOT a rebase** — `PROJECT_PLAN` §7's rule of 2026-08-19: a rebase replays
onto a new base and every hash an approval cites dies in that act. **Approvals pin blobs.**

## A0 — what the coordinator re-verified first-hand

**G0-b is real, and it is the measurement that makes this task possible.** `p4_6_training.json`'s run
`(mappo500, dt, 101)` carries `canonical_digest = 5d98d5351198c45054cce1e38b810dabd789708e71e3563e9428d37a49e0e563`
and `seconds = 213.3009614944458` — matching your retrain's digest exactly and your 212.4 s closely.
`canonical_digest_of` confirmed at `offline/method_tier_grid.py:1180-1185` to read `payload["model"]`
only. **Taken on your report:** G0-a, G0-c, G0-d and the five conflicts.

⭐ **Running G0-b before writing anything was the right call and it was not asked for.** `BRIEF_30` §7
asked for a *timing* probe; you ran a **feasibility** probe, and it retired the task's largest unknown
for 3.5 minutes. **The contrast you found is the reason it was a real risk:** `docs/returns/P5.2.md`
measured 61–63 of 66 tensors differing between two runs of the **spatial** trainer, and `dt_gate.py`
sets no determinism flag — so §4.1's instrument could have been a coin toss, and nobody had checked.

## A1 — Q1: **KEEP Gate 1b, and widen it to ONE RE-ROLL PER TIER**

Keep. Your reason stands: a paired comparison is only valid if both arms were measured by the same
instrument, and the `dt` column was measured in a worktree that no longer exists.
> ⭐ **But one cell does not license three, and the reason is provenance, not sampling: the three `dt`
> columns come from three different places** — `mappo1000` from `output/p4_dt/` (P4's reused column),
> `mix50` from `output/p4_7/`, `random` from `output/p4_6/`. **Re-roll one cell in each of the three,
> ≈3 minutes each against a ≈2 h campaign.**
> 🚨 **And on `mappo1000` it is not belt-and-braces, it is the ONLY integrity evidence available.**
> `DEFERRED` 56: `output/p4_dt/` is in no manifest, so Gate 1's manifest half cannot run there and its
> file-sha256 is filename-dependent (`DEFERRED` 29). **A behavioural re-roll is the compensating
> control for a known, recorded gap — say exactly that in the packet rather than listing the three
> tiers as equally protected.**

## A2 — Q2: **CONFIRMED. `abs(mean_difference)` is the primary and the only scored quantity**

Your §3.2 disambiguation is ruled correct and is now registered by this amendment. It is the quantity
Q2's CI is about, so Q1 and Q2 stay on one scale — which is the property that matters.
`mean(|per-draw difference|)` stays a reported, explicitly unregistered secondary.

⭐ **One thing to add to the packet, because a referee will ask and the answer is favourable: scoring
Q1 on the RAW scale is CONSERVATIVE.** The three `dt` ATTs are 105 · 108 · **420**. If differences
scale at all with baseline ATT, `random` would show the **largest** raw difference — which is the
**opposite** of what Q1 predicts. **So the registered scale makes Q1 harder to confirm, not easier**,
and §3.3's normalised column is a reading aid rather than the escape hatch it might look like.
**State that, and state that switching to the normalised ordering after the fact remains forbidden by
your own §3.3.**

## A3 — Q3: approved. The campaign goes to a user-started `tmux`

Write `output/p5_3b/run_campaign.sh` and hand it over (CLAUDE.md §5). Report the measured wall clock
against your ≈2 h projection; **P5.3c needs the number and Amendment R2 forbids armchair costing.**

## A4 — the two conflicts that change what you build: both rulings CONFIRMED

**F2** — `rtg_ablation`'s CLI resolves paths from `_CHECKPOINT_LAYOUT`, keyed by tier, so it cannot
address new checkpoints. **Calling `probe_cell(..., checkpoint_path=…)` directly, module unmodified,
is right** — editing a merged, reviewed module to add a path is a larger change than the problem.
**F3** — `arm_key` validates against `METHODS`, which §6.5 forbids editing. **Keying the arm locally
is right**, and your note that `assert_cell_complete`, `cell_stats` and `paired_comparison` do not
validate is the part that makes it safe. ✅ F1, F4, F5 accepted as recorded.

## A5 — 🚨 ONE NEW REQUIREMENT: your §5.1 point 5 is only half-closed, and the unclosed half is the prompt itself

You wrote that the canonical digest covers `payload["model"]` only, *"so a corrupted `config`,
`stats`, `target_rtg` or `provenance` block leaves it green"*, and closed the **`config`** half with an
explicit key/mode test. **`target_rtg` and `rtg_scale` are left open, and they ARE the prompt** — a
thread-through that perturbed either would leave Gate 2 green, leave your config test green, and
change every number in the campaign.

> **REQUIRED: Gate 2 additionally asserts the retrained control cell's whole payload equals the
> committed one for every key EXCEPT `model` and `provenance`** — i.e. `format_version`, `config`,
> `target_rtg`, `rtg_scale`, `normalise`, `scenario_id`, `stats`, `intersection_ids`. `model` is the
> digest's job; **`provenance` legitimately differs and the exclusion must be stated, not silent.**
> It is a dict comparison over data you already load. ⭐ **Finding your own blind spot and then
> covering three-quarters of it is the failure mode this project logs most often — the sweep is the
> hard part, not the sighting.**

## A6 — what is strong, recorded because it should be repeated

**§3.4** defines `seeds_reversed` before the data, counts an exact zero as a reversal (the
conservative direction), and — the part worth copying — **records that `d_s` and `D` average in
different orders and refuses to assert equality between them, because that would condemn a correct
implementation.** That is `A6`'s lesson and `A7`'s applied without being told.
**§3.3**'s self-binding sentence (*"switching to it after seeing the result is forbidden by this
sentence"*) is the right way to register a secondary. **§5.1** is a real blind-spot list, not a
formality — A5 above exists **because** you wrote item 5, which is the point of requiring it.

**Everything else in the plan stands. Tests first, run them red, then implement. Stop before merge for
`contract-reviewer`.**

---

# ✅ AMENDMENT B — 2026-08-27. A5's deviation confirmed, the test edit ratified

## B1 ✅ **A5 WAS IMPOSSIBLE AS WRITTEN. THE REPO WON AND YOUR REPLACEMENT IS STRONGER THAN MY REQUIREMENT**

**Measured by the coordinator before ruling:** the committed `output/p4_6/checkpoints/mappo500_dt_seed101.pt`
and `output/p4_dt/dt_seed101.pt` each carry a **`config` of 8 keys with no `rtg_mode`**; a checkpoint
written today carries **9, with `rtg_mode`** — because P5.3a made `to_json_obj` emit it
unconditionally. **So "the whole payload equal except `model` and `provenance`" cannot hold on
`config`, ever, and A5 was unsatisfiable the day I wrote it.**

> ✅ **RULED: your narrower rule is registered in A5's place — every SHARED key equal, none LOST,
> exactly ONE gained, and the gained key must be `rtg_mode` with value `"conditioned"`, pinned by an
> assertion.** ⭐ **It is stronger than what I asked for, not weaker: "skip `config`" would have been
> the easy deviation and you did not take it.** The other seven keys — `format_version`, `target_rtg`,
> `rtg_scale`, `normalise`, `scenario_id`, `stats`, `intersection_ids` — stay under strict equality.

⭐ **`m3` is the best mutation in this task and it settles A5's worth by demonstration.** Perturbing
`target_rtg` in the saved payload **passed the digest test and failed A5's comparison**
(`-6361.0` against `-6362.0`). That is precisely the hole A5 was written to close, exhibited rather
than argued — and `target_rtg` **is the prompt**, so the hole was load-bearing.

🚨 **The pattern this belongs to is mine and it is now FIVE deep in this brief chain, so it is stated
rather than absorbed:** the per-tier δ rule (A7, withdrawn — spanned eleven orders of magnitude),
*"exactly as P5.2's Gate 1 did"* (B3 — named a route with no call site), A4's cell-2 selection rule
(picked the worst discriminating cell available), `BRIEF_29`'s *"near-uniform shift"* mechanism
(falsified at 97.2 %), and now A5's payload comparison (unsatisfiable). **Every one was specified from
the SHAPE of the thing instead of checked against the INSTANCE, and every one was caught by an
implementer measuring rather than accepting.** ⚠️ **The mitigation is not "be more careful": before a
brief registers a rule over an artifact, the rule gets evaluated against that artifact once. That is
one command, and it would have caught all five.**

## B2 ✅ **THE TEST EDIT IS RATIFIED — the change was right and the ORDER was wrong**

**Verified independently:** the original `assert "equival" not in json.dumps(scored)` **would have
forbidden the disclaimer `BRIEF_30` §5 mandates** — *"a failure to reject, never a demonstration of
equivalence"* contains `equival`. The test contradicted the brief it was written to enforce. Your
replacement bans six specific claim forms (`is equivalent`, `are equivalent`, `equivalence margin`,
`within_delta`, `equivalence threshold`, `delta_att`), **none of which collides with the disclaimer**
(checked), and **adds a POSITIVE assertion that the disclaimer is present** — which the original
lacked entirely. **Strictly stronger. Ratified, and it stands.**

> ⚠️ **The order was still wrong, and the reason the rule is bright-line is worth stating.** CLAUDE.md
> §0: *"If you believe the test is wrong, stop and say so — do not fix it yourself."* **"The test was
> wrong" is exactly what a session says when it weakens one.** Your full disclosure is what makes this
> safe, and a disclosure after the fact is strictly weaker evidence than a question before it. **The
> cost of asking was one message; the cost of the rule eroding is that the next edit is judged by the
> same session that wants it to pass.** Flag first next time; the answer here would have been yes.

## B3 ✅ `git merge main` was authorised, and flagging it was right

The instruction was written, in Amendment A, with its reason (a rebase kills every hash an approval
cites). **Taking an explicit written instruction as authorisation and saying so is exactly the
behaviour wanted.** The merge direction was correct: `main` into the task branch, never the reverse.

## B4 ⭐ The `cell_stats` defect is a recorded CLASS, not a one-off

**Verified:** `offline/method_tier_grid.py:1488` `cell_stats` emits **`seeds`** (a sorted list) and
**never `seed`**; a committed cell's keys confirm it. Your report keyed by `(tier, seed)` **would have
raised `KeyError` after the two-hour campaign**, and the unit fixture supplied `seed` itself, **so the
test passed while the real path was broken.**

> **This is `DEFERRED` 37's family — a guard or a test that cannot be exercised by the data it ships
> with — and it is the second sighting.** ⚠️ **It was found by INSPECTION, not by a test, and you said
> so; that honesty is the reason it is a finding rather than a near miss.** ✅ Owning it in
> `nortg_cell_record` with a simulator-free test is the right fix: the record now derives from the
> real emitter's contract rather than from a fixture's convenience.

**Nothing else changes. Run the campaign.**

---

# ⛔ AMENDMENT C — 2026-08-28. THE CAMPAIGN IS HELD. A short fix round first

`docs/reviews/P5.3b-preflight.md` says **CLEAR TO RUN, no blockers**, and I am overriding that to
**HOLD** — not because anything can be destroyed, but because **two of its MAJOR findings would be
baked into the artifact by a run that starts now, and both cost minutes to fix while nothing is
running.** ⭐ **This is the first PRE-FLIGHT REVIEW and it has already paid for itself: `M5` alone
would have written a knowably false commit into `measurement_git_commits`.**

**Everything below is small. Estimated together: well under the 36 minutes `M5` costs on its own.**

## C1 🚨 **M5 — DELETE the four `mappo1000` chunks and re-run them. Not disclosed, deleted**

**Coordinator-verified before ruling:** `f115b7c` defines `nortg_cell_record` **zero** times, `39a4eef`
defines it **once**, and `eval_mappo1000_seed101.json` **contains `cell.seed`** while recording
`git_commit = f115b7ce…`; its mtime (22:27:08) sits between the two commits. **The chunk was written
by a dirty tree and its provenance is false.**
> **RULED: `rm output/p5_3b/{gate1,train_mappo1000,eval_mappo1000_seed101,probe_mappo1000}.json` and
> the `mappo1000` checkpoints, then let the campaign regenerate them.** ≈36 min of a ≈2 h run.
> **Disclosure is NOT an acceptable substitute here.** `measurement_git_commits` is the one field
> `DEFERRED` 39 exists to make trustworthy; a provenance we KNOW to be wrong, left in place because
> fixing it costs 36 minutes, is the trade this project exists to refuse. **Nothing has started. This
> is the cheapest this decision will ever be.**
> ⚠️ **The other three chunks have no behavioural tell, so I am not claiming they are clean — I am
> deleting them because I cannot tell, and "cannot tell" is the reason, stated.**

## C2 🔒 **Add a dirtiness flag to `runtime_provenance` — M5's mechanism, and its second sighting**

`offline/dt_gate.py:636` records `git rev-parse HEAD` **with no dirtiness check**. `docs/reviews/P5.3a.md`
**M6** logged exactly this as *"pre-existing"* three days ago. **It has now bitten for real.**
> **Required: `runtime_provenance` records `git_dirty` (from `git status --porcelain`, non-empty →
> `true`), and the campaign REFUSES to write a chunk from a dirty tree unless `--allow-dirty` is
> passed explicitly.** `dt_gate.py` is already this task's file, so it is in scope. ⭐ **A provenance
> field that cannot distinguish a clean tree from a dirty one is not provenance.**

## C3 **M1 + M2 — the probe must verify the digest AND enforce Gate 3 where it runs**

`_run_probe` neither checks the checkpoint digest (which `_run_evaluate` does at `:1475`) nor refuses
a non-zero `flip_rate` — it prints `worst` and writes the chunk anyway. **Demonstrated against the
committed conditioned checkpoint: `max flip_rate 0.004722`, exit 0.**
> **Required: `_run_probe` recomputes `canonical_digest_of` and refuses a mismatch, exactly as
> `_run_evaluate` does; and it RAISES on any non-zero `flip_rate` or any `rtg_mode != "zero"`.**
> 🚨 **The caching is what makes this urgent rather than cosmetic: the driver skips a tier whose
> `probe_$TIER.json` exists, so a bad probe chunk survives every restart and the only signal arrives
> two stages later.** A gate that reports at the end of a two-hour run is not a gate.

## C4 **M3 — assert the chunk describes the tier being run.** Two lines

`_run_evaluate` checks the seed and never `training["tier"] == args.tier`. Demonstrated: a copied
chunk evaluated a `mappo1000` checkpoint and wrote `arm: dt_nortg@mix50`, exit 0, every downstream
assertion passing. ⚠️ **The reviewer's judgement is right — it needs a manual rename to arise, and
this tree is being hand-managed, which is exactly why it is cheap insurance rather than theory.**

## C5 **M4 — reap the fan-out's children before exiting**

`run_campaign.sh:86-95` `exit 1`s on the first failure while four background jobs keep running and
writing. **An immediate restart gives 8–10 concurrent evaluations on one GPU**, against the script's
own stated one-thread-per-cell protocol. **Kill or wait for the remaining children before `fail`.**

## C6 **M6 — the script must re-verify the reused manifests after the run**

`BRIEF_30` §8 and the plan §9 both require it; the script does neither. **Add `sha256sum -c` on
`SHA256SUMS_p4_3 / p4_6 / p4_7 / p5_3a` at the end, with the counts printed.** The reviewer measured
the "before": **all nine manifests verify, zero failures.**

## C7 **Correct the plan's §9 disclosure — over-disclosure is still a false claim**

§9 says CityFlow rollouts rewrite `output/roadnet.json` and `output/replay.txt`. **Coordinator-verified:
all 100 held-out draw configs carry `saveReplay: False` and `replayLogFile: None`**, and the
reviewer's real rollouts left both mtimes untouched. ⚠️ **Not a contradiction of P5.3a's reviewer, who
observed the SUITE — different configs — rewriting them; both are right about different things, and
the packet should say which.** **Delete the claim or narrow it to the suite.**

## C8 ✅ What is NOT required, so the fix round does not sprawl

The non-atomic `torch.save`, the single unfenced write to the git-tracked artifact, the
directory-not-sibling fence, the stale `COMPLETE` marker and `probe_cell`'s `--draws-root` are all
**recorded and not fixed** — none is reachable by a correct run and the reviewer bounded each.
**Do not touch them.** ⭐ **And nothing in the campaign's design changed: the resume logic was attacked
with truncated, stale, wrong-seed and SIGKILLed states and held every time. P5.2's O4 defect is not
repeated — `_run_train` writes the complete column once, atomically, after the loop.**

**When C1–C7 are done: re-run the three P5.3b test files, paste the tail, then start the campaign.**

---

# AMENDMENT D — 2026-09-01: the A11 re-scope. This is why P5.3b was held, and it is the only thing blocking it.

**P5.3b was HELD on 2026-08-28** because T1 found the primary metric wrong-by-definition. That question
is now closed (P8.4 merged 2026-09-01), and the hold lifts **subject to this amendment**.

## D1 — the campaign emits all FIVE quantities AT COLLECTION TIME

`PREREGISTRATION` **A11(b)** makes it unconditional: every reported ATT cell carries
**`att_ours`, `att_engine`, `entered`, `created`, `never_entered`.** No threshold, no verdict, no
condition — the five appear on every cell, always.

⭐ **This is a brief amendment, not new code.** `offline/admission_probe.py` already computes all five:
`read_admission_at_horizon(env, *, created)` at **`:416`** and `probe_episode` at **`:510`**. Import and
call them; do not reimplement, and do not modify `admission_probe.py` — it is merged, reviewed and its
artifacts are cited in A11 itself.

🚨 **AT COLLECTION TIME, not re-derived afterwards.** P8.4b spent 38,500 episodes and ~3.2 h re-deriving
cells that were collected without them. **Emitting them now costs a horizon read per episode and
approximately nothing.** A campaign that ships without them joins the re-derivation backlog, and that
backlog is exactly what P8.4 existed to clear.

## D2 — Rule R has fired, so the reporting convention is settled before you start

**`att_engine` is the PRIMARY metric on hz1x1 and grid4x4**; `att_ours` is reported beside it in every
table with the three counts. **`cf_cologne3` is NOT RUN under Rule R and carries no single-definition
claim** — if P5.3b touches cologne3, both definitions are reported and any ordering that differs between
them is flagged **definition-dependent**, never presented as a result.

## D3 — two reporting rules inherited from P8.4b, both already ruled

1. **Discriminability (ruled 2026-08-31).** Every reported contrast states, per tier, **whether the arms
   it compares are distinct at all.** *A contrast over identical inputs is not a null result.* This is
   live for P5.3b: on `fixedtime`, the DT reproduces its behaviour policy **bit-identically** on both
   scenarios (measured 2026-08-31), so a no-RTG-versus-RTG contrast on that tier may have nothing to
   measure. **Say so if it does.**
2. **Both poolings (ruled 2026-08-31).** For any contrast whose arms are non-distinct on a tier, report
   the pooled statistic **including** that tier and **excluding** it, with the structural reason. Never
   choose one. ⚠️ **And the narrowing P8.4b earned: if the excluded set is empty, the two poolings are
   the same computation and their agreement carries no information — say that rather than reporting an
   agreement.**

## D4 — the branch is 111 commits behind `main`, and this one warrants a merge

`task/p5.3b-nortg-campaign` @ `53e995d` is **111 commits behind**. Under §7's rebase rule a rebase or
merge is warranted only when `main` moves **code, tests or the guard scripts** under the branch —
**and it has, substantially**: `offline/engine_att_reference.py`, `offline/att_rederivation.py`, their
two test files, and the `plan_replay.py` docstring correction. **No campaign is running in the worktree
and it has never run**, so the 2026-08-19 precondition is satisfied. **`git merge main` in
`/home/filip/rltraffic-p53b`**, as `BRIEF_30` B3 already authorised once.

## D5 — inherited, and it is not optional

**The CI skip-ceiling patch names P5.3b's merge as its own next expiry** (`docs/patches/README.md`,
`ci_gate_ceiling_139_p8_4b.patch`). Re-measure it at merge from the real `junit.xml` and ship it as a
**separate commit**, classifying the delta by inspection rather than by regex. **Do not pre-bump.**

## What is UNCHANGED

`BRIEF_30`'s §1–§9, Amendments A, B and C, the completed fix round, and the CLEAR pre-flight
(`docs/reviews/P5.3b-preflight.md`, no blockers, 6 major). **Nothing in the campaign's scientific design
moves.** D1–D3 change what each cell CARRIES and how contrasts are REPORTED; D4 and D5 are mechanics.
