# BRIEF 15 — P4.3: probe-calibrated return prompting, ablated in-domain

**Mode:** Claude Code implementation session · **Branch:** `task/p4.3-rtg-calibration`, from `main`
**Worktree:** fresh — `git worktree add /home/filip/rltraffic-p43 -b task/p4.3-rtg-calibration main`
**Read first, from disk:** `PREREGISTRATION.md` **A8(a)** — it binds this task more than this brief
does — then `docs/returns/P4.md` §8 items 9 and 10, `docs/reviews/P4.md` MAJOR-1, and this file.

⚠️ **Use an absolute path in every command** (`PROJECT_PLAN` §7, three violations by the coordinator
on 2026-08-12/13). ⚠️ **Set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` on every job** (`DEFERRED` 41).
⚠️ **Never write "MADT"** (`CONTRACTS` C9) — it is *"the offline multi-agent Decision Transformer"*,
or *"the DT"*.

---

## 1. Why this task exists, and why it is now load-bearing

This is **the paper's named method component**. §1 narrowed its novelty on 2026-08-01: DTLight already
scales RTG within a domain, so **our contribution is the target-domain probe**, not return
conditioning as such.

It has also become the task that decides how P4.4's result may be written. P4's own independent review
recorded two things, both **before** any baseline existed:

- **The model is conditioned outside its training support for a fifth of every episode.** All training
  rewards are ≤ 0, so every training RTG is ≤ 0 — but at evaluation the RTG is driven by observed
  rewards from a declared start of −5762 and goes **positive for 75 of 360 decisions (20.8 %)** on the
  probed episode, with the scaled input spanning **−0.577 … +0.153**.
- **The declared target is not the best target.** On `(seed 101, draw 1000)`: declared `−5762` →
  **106.46**, `target_rtg = 0` → **102.05**, `−3000` → **102.41**. ⚠️ **n = 1, one episode, and it is
  not representative — the DT's arm mean is 104.9558, so that episode is 1.50 ATT worse than typical
  for it.** The direction is suggestive; the magnitude is not established. **Establishing it is this
  task.**

`docs/reviews/P4.4.md` §8.6 binds until this lands: **no DT-versus-baseline sentence enters the paper
until P4.3 has run**, because the swing above is larger than the effect P4.4 reports.

## 2. What makes this task cheap, and what makes it dangerous

**Cheap: nothing is retrained.** The RTG target is an *inference-time* parameter. P4's five DT
checkpoints are secured at `/home/filip/rltraffic/output/p4_dt/` (verified 5/5 by sha256 on
2026-08-11). A 9-point grid over 5 seeds × 100 draws is 4,500 rollouts ≈ **80 minutes**, and the
probe is 100 MaxPressure rollouts ≈ 2 minutes.

**Dangerous: this is the one task where selecting on the evaluation return would look like method
development.** D5 and `PREREGISTRATION.md` §6.1 already forbid it. **A8(a) restates it because here it
would be easy and it would be invisible in the result.**

> **The calibration rule is a declared function of the probe's return distribution, written into
> `docs/plans/p4.3.md` before any target-domain evaluation number exists. The sweep is the ABLATION
> AROUND that rule and never the mechanism that selects it.**

**A target chosen because it scored best on the held-out draws is not a method. It is a leak wearing a
mechanism's name**, and the paper would be claiming a mechanism it did not use.

## 3. The four questions, in this order

1. **What is the in-domain RTG landscape?** A declared grid, evaluated on the 100 held-out draws.
2. **Where does the NAIVE rule land on it?** `target_rtg = max training-split return = −5762`, which
   is what P4 reported.
3. **Where does the DECLARED CALIBRATION RULE land?** The test of the mechanism.
4. **What is the in-support fraction for each?** — see §5, and note it needs **no evaluation return at
   all**.

⚠️ **Be honest about what the in-domain case can and cannot test, and say so in the packet.** In-domain
the probe's *cross-domain* job is trivial: source and target coincide, so **any correct rule must
approximately reproduce the naive target here**, and the probe's real work is not exercised until
C3/P7. What this task *can* establish is (a) the shape of the landscape, (b) whether the naive rule is
a good point on it, and (c) whether the rule's form is computable and in-support. **If the sweep shows
the naive target is poor in-domain, then a rule that reproduces it in-domain is also poor — and that
is a finding about the mechanism, not a failure of the task.**

## 4. The probe set — disjoint by construction, and it must stay that way

`PREREGISTRATION.md` §5 registers: draw **0** nominal, **1–999** training pool, **1000–1099** held out.
The corpus used **1–200**. So **draws 201–999 are unused**, and the probe takes its episodes from
there.

**Required: the probe runs on draws `201 … 200+k`, never on 1–200 and never on 1000–1099.** Assert it
from the artifact, not from the command line.

**Use the k values C3 already registers — `k ∈ {5, 20, 100}`** — so this ablation answers *how many
probe episodes the rule needs* in the easy case, and the answer transfers to the transfer study.
Probe episodes are MaxPressure rollouts; 100 of them cost about two minutes.

## 5. The two criteria, and only one of them may select

| criterion | what it measures | may it select the rule? |
|---|---|---|
| **in-support fraction** | the fraction of decisions whose RTG input lies inside the training RTG range — computable from the conditioning trajectory alone | **YES** — it uses no evaluation return, and a rule that keeps the model in support is better *by construction* |
| **ATT on the held-out draws** | the outcome | **NO. It scores the rule and never chooses it.** |

**This split is the point of the task.** The in-support fraction gives a **leakage-free** way to
prefer one rule over another, which is exactly what a method needs and what "sweep and pick the best"
does not have. **Report it for every grid point and for the rule**, with the training RTG range it is
measured against stated as a number.

## 6. Per-file requirements

**Do NOT modify `agent/DTAgent.py`.** Measured: `DTAgent.__init__` accepts `target_rtg` and
`rtg_scale`, but **`load()` overwrites both from the checkpoint payload** (`DTAgent.py:803-804`), and
`from_checkpoint(env, path, device)` takes no override. So the target must be set **after** load.

**Required, and this is the load-bearing engineering decision of the task:** one named function in
your own module that builds the agent from the checkpoint and then applies the target, and **a test
that verifies the override BY EFFECT — the value actually reaching the model — not by reading the
attribute back.** P4's review found three mutations of exactly this surface that survived 58 tests and
moved the reported number by up to **+32.5 ATT**, because no test ever built the agent the way
evaluation builds it. **That is the defect this test exists to prevent, and reading `_target_rtg`
back does not prevent it.**

**`rtg_scale` does not change in this task.** It is the normalisation divisor (`9991.0`); only
`target_rtg` varies. Declare that, and assert it in the artifact.

**New module** `offline/rtg_calibration.py`: the probe runner, the declared rule, the grid, the
in-support metric, the artifact and a CLI. **Reuse from the merged modules** — `evaluate_arm`,
`env_settings_from_manifest`, `mean_ci95`, `wilcoxon_signed_rank`, `paired_comparison`,
`write_json_atomic`, `assert_campaign_complete`, `pin_torch_threads`, `HELD_OUT_DRAWS`,
`TRAINING_SEEDS`. **Do not reimplement any of them.**

**Artifacts:** `docs/data/p4_3_probe.json` (the probe episodes and the rule's inputs) and
`docs/data/p4_3_rtg.json` (the grid, the rule's point, both criteria, the paired statistics).

## 7. Tests — red-first, each with its named mutation executed and pasted

| # | test | mutation that must kill it |
|---|---|---|
| T1 | **the load-bearing one:** the RTG the model actually receives at step *t* equals `(target − Σ rewards so far) / rtg_scale`, spied at `forward`, for every step of a real episode | apply the override before `load()` instead of after — the checkpoint value silently wins |
| T2 | the probe draws are disjoint from **both** 1–200 and 1000–1099, asserted from the artifact | shift the probe range by one |
| T3 | the in-support fraction matches an independent recomputation from the raw RTG trajectory | count `≥` instead of `>` at the boundary |
| T4 | the declared rule is a pure function of the probe returns — same probe input ⇒ same target, on a fixture with a known answer | let it read any evaluation quantity |
| T5 | `rtg_scale` is identical across every grid point and equals the checkpoint's | let it vary with the target |
| T6 | the grid is the **declared** grid — the artifact's points equal the constant, and an undeclared point is refused | derive the grid from the results |
| T7 | campaign integrity: completed runs equal requested, enumerated up front from the **declaration** | derive the expected set from the data (`PROJECT_PLAN` §7) |

**T1 is the one that protects everything else.** Every other test can pass while the override is
applied at the wrong moment, and then the whole grid is one target repeated.

## 8. Also in scope — two queued items this task owns

- **`DEFERRED` 31** — back-fill effect sizes into `docs/data/p4_gate.json`. §8 makes them mandatory
  beside every p-value and P4 reported none; the paired per-draw differences are already in the
  artifact, so this is arithmetic over committed data, **no re-run**. It is due *"before any P4 number
  enters the paper"*, and this task is that moment.
- **`DEFERRED` 39** — split `runtime_provenance()` into a frozen **measurement** provenance carried
  from the inputs and a separate **written-at** commit. You are writing new artifacts, so you can
  adopt the split rather than migrate to it. ⚠️ **This touches the merged `offline/dt_gate.py`. It is
  authorised here, dated 2026-08-13, additively only — no existing field may change meaning, and
  P4/P4.4/P4.5's artifacts must regenerate unchanged except for that field.** Prove it the way
  `BRIEF_12` §7.1 did: a recursive comparison against a pre-declared expected-difference set.

## 9. Scope fence

- **No retraining of the DT.** No new corpus, no new scenario, no SUMO, no transfer — C3 is P7.
- **No change to `rtg_scale`, to the architecture, or to the evaluation protocol.**
- **No selection of any reported target by evaluation return**, in any form, including "we looked at
  the grid and then wrote the rule".
- **Do not fix `DEFERRED` 42 or 43** beyond the sites you touch.
- **Do not edit** P4/P4.4/P4.5's committed artifacts, `offline/dataset.py`, `agent/DTAgent.py`,
  `agent/OfflineBaselines.py`, or any frozen path.

## 10. Definition of Done

- [ ] `docs/plans/p4.3.md` committed **before any evaluation number exists**, carrying the calibration
      rule in full, the declared grid, the probe design and the k values
- [ ] Probe run on draws 201+, disjoint from training and held-out, asserted from the artifact
- [ ] The grid and the rule's point evaluated on all 100 held-out draws × 5 seeds
- [ ] **Both criteria reported for every point**; the in-support fraction stated against the training
      RTG range as a number
- [ ] Every mutation in §7 executed and its failure pasted
- [ ] `DEFERRED` 31 and 39 addressed, with 39's regeneration proof pasted
- [ ] Full suite green, tail pasted, **stating whether it was pinned**
- [ ] All three guards exit 0; `git diff --stat` shows no frozen path
- [ ] Return Packet at `docs/returns/P4.3.md` **including the AI-assistance record** (`CLAUDE.md` §8;
      P4.5's was wrong in its first outing — determine authorship with
      `git log --diff-filter=A` and the creating commit's message, and state the method)
- [ ] §6's checkbox left unticked; it is mine, in the merge commit

## 11. What I will do with the result

**If the calibrated rule improves on the naive target**, the paper reports the mechanism with an
in-domain ablation behind it, and P4.4's comparison is re-run against the calibrated DT before any
DT-versus-baseline sentence is written.

**If it does not**, that is equally reportable and arguably more interesting: it would mean return
conditioning does not recover the DT's headroom in the easy case, and the honest paper says the
mechanism's value is confined to cross-domain transfer — where C3 will test it — rather than claiming
it generally. **Both outcomes are registered. Neither requires anything to be renegotiated.**
