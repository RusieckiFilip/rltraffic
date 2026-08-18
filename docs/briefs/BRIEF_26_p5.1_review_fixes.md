# BRIEF 26 — P5.1 review fixes (mandated before merge)

**Branch:** `task/p5.1-spatial-mixing` (continue) · **Review:** `docs/reviews/P5.1.md`,
**PASS-WITH-NOTES**, 2 MAJOR, no blockers, on disk before this round.
⚠️ `git add` **before** `check_english.sh` — it is vacuous on untracked files.

## 0. What this round is not

**No blocker, no reported number wrong, no contract violated, A8(a) holds with margin.** The reviewer
ran 15 of its own mutations (9 killed) and reproduced every cell mean to **1.14e-13**, the primary
paired difference **exactly**, the MDE, the ladder anchors and 25/25 checkpoints. **This round repairs
how the result is STATED, and two missing guards.**

## 1. F1 — RESTATE THE HEADLINE. m7 changes the finding, not just its attribution.

`dt_nomix` per-seed sd **0.104**; `dt_spatial` sd **30.36**. **The 70× spread belongs ENTIRELY to the
treatment arm**, and its failure mode is **partial loss of control** — vehicles at horizon
**16.2 / 133.9 / 123.2 / 75.7 / 49.4** against the control's **~15 on every seed** — not uniform
degradation.

> **REQUIRED: the headline is not *"spatial mixing costs 39.56 ATT"*. It is *"SPATIAL MIXING SOMETIMES
> LOSES CONTROL, AND HOW BADLY IS UNSTABLE ACROSS SEEDS"*, with 39.56 as the mean of a distribution
> whose SHAPE is the finding.** ⚠️ **A mean over `{16.2, 133.9, 123.2, 75.7, 49.4}` vehicles-at-horizon
> is a summary that hides its own subject.** Report the per-arm sd and the vehicles-at-horizon vector
> wherever the effect is stated.

⭐ **And say what this licenses against the two preprints, because it is more than opposite sign:
neither reports instability of this kind — both report a mean effect with a sign. A third measurement
finding the contrast UNSTABLE IN MAGNITUDE says something neither could have seen with their reporting
convention.** That is C2's framing paying off.

⚠️ **The seed-101 near-null stays UNEXPLAINED.** The reviewer refuted the obvious hypothesis by three
routes — training loss 2nd lowest, weight statistics within ~1.5 %, and a graph-dependence probe where
**seed 101 flips 48.83 % of actions without the graph, the HIGHEST of the five.** **Report it as
unexplained; do not smooth it with the unanimous sign.**

## 2. J1 — a stated ordering reverses, and the artifact already knew

`dt_spatial` vs `bc` reverses on seed 101 (`−14.05` against `+63.29 / +57.93 / +22.10 / +12.89`), §6
omits it while claiming *"every headline contrast"*, and **`p5_1_grid.json` already carries
`reverses_on_n_seeds: 1` for that pair.** It fixes a rank the packet states twice: **3/5 pooled, 2/5 on
seed 101.**

> **REQUIRED: (a)** flag it in §6 and annotate `predictions.p2a.order` with per-seed information — a
> pooled ranking with none lets a consumer inherit "rank 3" as settled. **(b) Put the enforcement in
> the GENERATOR: if `reverses_on_n_seeds` is non-zero, the report must emit the qualifier beside the
> ordering.** The rule was made binding one day before this and **was failed by the artifact's
> CONSUMER, not its producer** — an author remembering to read a field they already computed is not a
> mechanism. **(c)** `reverses_on_n_seeds` is the seed **minority**, not "seeds disagreeing with the
> pooled sign"; they coincide here. **Document that, or make it the second definition.**

## 3. J2 — guard the line that carries the number

`agent/SpatialDTAgent.py:775`'s `logits[0, :, -1]` has **no test**: `[0, :, -2]` survives the whole
suite, proven non-equivalent (an exact one-step lag with a degenerate read at step 0). **The shipped
code is correct — the guard is missing**, on the most alignment-sensitive line in the agent, which
produced every number in the primary contrast.
> **REQUIRED: a test that mutates the INDEX and fails. Paste the failure.** Not a test of the caller.

## 4. Also required

- **m3 — withdraw *"exact set identity"*.** 302/320 strictly above the cut from the top-8 nodes, **32
  streams tie at −114.0** filling the last 18 slots, one belonging to **B0, the 9th node**; a reverse
  tie-break gives **9** intersections. **The load-sorter mechanism is untouched and is the transferable
  part.** *(Already withdrawn in `PROJECT_PLAN` §1b — the phrasing was the coordinator's.)* Also record
  that `offline/offline_baselines.py:418` says *"measured, and not a tie"* for the P4 tier — **on this
  tier it IS a tie and nobody recorded it.**
- **m5 — three shipped guards have no test** (`spatial_mixing.py:253`'s `rtg_scale` branch, which the
  packet's M4 reads as covered while covering only `target_rtg`; `joint_windows.py:164` and `:209`).
  **For `:164`, determine whether it is reachable with the v1.1 on-disk format** — untested-live and
  documented-unreachable are different findings and neither is currently in the table.
- **m4** the retracted sublayer claim survives verbatim in `tests/test_spatial_dt_agent.py:162-165` ·
  **m6** the mutation count self-contradicts (20 vs 19) · **n9** `campaign.log` is a `tmux capture-pane`
  reconstruction — `tee` failed for the **second time in the project** — **disclose the provenance in
  the packet**, as P4.7 did · **n10** the Gate-2 packet carries no DoD table.
- **Four theatre tests** named by the review: strengthen or delete under §7's conditions.

## 5. Definition of Done

- [ ] Headline restated per §1, with per-arm sd and the vehicles-at-horizon vector beside the effect
- [ ] Seed 101 reported as **unexplained**, with the three negative probes recorded
- [ ] J1 flagged, `p2a.order` annotated, **and the qualifier emitted by the generator**
- [ ] J2 guarded by an index mutation, **failure pasted**
- [ ] m3 wording, m5's three guards, m4, m6, n9, n10, four theatre tests
- [ ] **No reported number moves** — prove by regeneration
- [ ] Suite green, tail pasted, pinned; guards no-arguments, full-output counts, corpus named
- [ ] §6's checkbox unticked; it is mine, in the merge commit
