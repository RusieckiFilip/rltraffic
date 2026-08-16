# BRIEF 22 — P5.1: the spatial mixing layer (grid4x4)

**Mode:** Claude Code · **Branch:** `task/p5.1-spatial-mixing`, from `main`
**Worktree:** fresh — `git worktree add /home/filip/rltraffic-p51 -b task/p5.1-spatial-mixing main`
**Read first, from disk:** `PROJECT_PLAN` **§1 (all claim constraints), §1b (R1–R7), §9 (Risk Register)**,
then `docs/reviews/P4.6.md` and `docs/reviews/P4.7.md`, then `agent/DTAgent.py` and `offline/dataset.py`.

⚠️ Absolute paths · pin threads · guards with **no arguments**, counted from **full output, never a
tail** · never write "MADT" (C9) · `tmux` for any campaign (`BRIEF_17` §12, all six conditions).

---

## 1. Why this is the experiment that matters

**Every result we hold comes from a setting where the DT has NO STRUCTURAL ADVANTAGE.** BC, %BC and
IQL are independent per intersection **by construction**, and so is our DT today. Across **eight
tiers** — five single-controller, three mixture — **the DT leads zero of them** (§1b, R7).

**P5.1 is the first time the DT gets information the baselines cannot have.** §1b's **R5** records that
the DT's deficit is now robust to the **prompt** (P4.3, 0.9026 ATT over a 13,000-wide grid), to **data
quality** (P4.6, five tiers) and to **the presence or absence of a filter** (P4.7, three mixtures).
**The two surviving explanations are the single-intersection setting and the architecture itself, and
this task is the only experiment that can separate them.**

🚨 **§9 rates DT collapse on 16-intersection grids on PRIOR EVIDENCE, not speculation:** DTLight's own
Table 1 gives **446.8 ± 128.0** from EMP data on Grid 4×4. **Expect collapse to be a live outcome and
write the task so that it is a result rather than a disappointment.**

## 2. ⚠️ THE PLAN'S PREMISE IS WRONG ABOUT `RoadnetInfo`, AND I VERIFIED IT BEFORE WRITING THIS

`PROJECT_PLAN` §6 describes P5.1 as *"graph attention over road-network adjacency **from
`RoadnetInfo`**"*. **Measured today: `RoadnetInfo` and `IntersectionInfo` expose NO adjacency,
neighbour or graph field** — `intersections`, `lane_ids`, `road_ids`, `intersection_ids`,
`road_lengths`, `road_max_speeds`; and per intersection `incoming_lanes`, `outgoing_lanes`,
`num_phases`, `phase_roadlink_mapping`, `phase_durations`, `phase_states`, `roadlink_lanes`.
**And `utils/` is FROZEN, so the field cannot be added there.**

> **Adjacency is DERIVED, in `offline/`, and never assumed: intersection `A` feeds `B` iff
> `A.outgoing_lanes ∩ B.incoming_lanes ≠ ∅`.**

🚨 **AND IT MUST BE PROVED FROM STRUCTURE BEFORE IT IS USED — this is `PROJECT_PLAN` §7's newest rule,
earned by P7.0 four days ago, where an unproved pairing key voided a registered criterion.**
**A 4×4 grid has a KNOWN adjacency pattern and that is a free positive control:**

| node class | count | undirected neighbours |
|---|---|---|
| corner | 4 | **2** |
| edge | 8 | **3** |
| interior | 4 | **4** |

**Total undirected edges = 24.** **Assert the derived graph reproduces exactly that**, and show the
check **FAIL** on a deliberately wrong derivation (e.g. `incoming ∩ incoming`) before trusting it.
⚠️ **A graph attention layer over a wrong adjacency still trains and still produces plausible numbers —
that is the failure this repo exists to prevent, and it is invisible without this control.**

## 3. `DEFERRED` 37 is BINDING here, by name

> *"Feeding `act()` the wrong intersection's statistics is an EQUIVALENT mutant on hz1x1 and a LIVE
> DEFECT at P5."*

It survives 58/58 on hz1x1 because every episode carries `ix0_*` only, so `intersection_ids[0] == ix_id`
always. **grid4x4 has 16, verified today (`ix0_state` … `ix15_state`, `ix_ids` present), so the mutation
stops being equivalent the moment this task runs.** **Execute it and paste the failure.** ⚠️ **A
per-intersection normalisation that silently uses intersection 0's statistics for all 16 would produce
a plausible, wrong grid — and nothing currently catches it.**

## 4. What to build and what to measure

**Architecture:** a spatial mixing layer over the derived adjacency, **interleaved with** the existing
temporal causal attention — not bolted before or after it. The ablation that makes the claim
falsifiable is **the same model with mixing disabled**, which is P5.3's `no-spatial-mixing` arm and
should be built here as the control rather than deferred.

**Corpus, verified today:** `cf_grid4x4` carries the same 7-tier ladder as hz1x1 — `random`,
`fixedtime`, `maxpressure`, `mappo060/200/500/1000` × 5 seeds — **200 episodes per tier, 16
intersections per episode.**

**Comparators:** BC, %BC and IQL on the same grid4x4 tier, independent per intersection as they are by
construction. **That asymmetry is the experiment, and it must be stated as such rather than hidden:
the DT is being given something the baselines structurally cannot use.**

⚠️ **Cost, and the cut that governs it.** grid4x4 training measures **2.81 s/episode against hz1x1's
1.04** (P2.0b), so a full per-tier sweep is the item §10's sequencing ruling flags as **most likely to
overrun**. **CUT 2 fires before cut 3: keep P5.1 plus grid4x4 on ONE tier (`mappo1000`) and drop the
sweep.** The structural question — *does spatial mixing help at all* — needs one rung; only the
interaction needs five, and **the interaction is the expendable half.**

## 5. Registered predictions — into `docs/plans/p5.1.md` BEFORE the first gradient step

Per **A8(a)**. ⚠️ **Of the six predictions registered across P4.6 and P4.7, five failed and one held on
a conjunct that does not resolve. Write these to be falsifiable, expect them to fail, and build no
rescue** — `BRIEF_20` §1 exists because a rescue was attempted once and did not survive review.

State at least: whether spatial mixing beats its own no-mixing control on the same tier; whether the DT
with mixing reaches the offline-method field it lost on all eight single-intersection tiers; and
**a declared collapse criterion** — what number would make this DTLight's 446.8 rather than a result.

## 6. Also in scope

- **`DEFERRED` 21 is a CONSTRAINT, not a task:** action masking never binds in this corpus (all 460
  sampled mask streams all-True), so **a no-mask ablation measures nothing and the paper may not credit
  masked-action modelling.** Honour it; do not try to fix it here.
- **`DEFERRED` 43's remainder**, if and only if you open a test file for another reason.

## 7. Definition of Done

- [ ] `docs/plans/p5.1.md` committed **before any training**, with §5's predictions verbatim and the
      declared collapse criterion
- [ ] Adjacency derived in `offline/`, **proved against §2's 2/3/4 pattern and 24 edges, with the check
      shown FAILING on a wrong derivation**
- [ ] `DEFERRED` 37's mutation **executed on 16 intersections**, failure pasted
- [ ] The no-mixing control built here, not deferred
- [ ] Every mutation executed and **its failure pasted**
- [ ] Campaign in a **user-launched `tmux`** session; **no `until`-poll** (`pgrep -f` self-matches its
      own wrapper — demonstrated in P4.6)
- [ ] Suite green, tail pasted, pinned state stated; three guards, no arguments, full-output counts,
      each naming its corpus
- [ ] Return Packet at `docs/returns/P5.1.md` with the AI-assistance record
- [ ] §6's checkbox left unticked; it is mine, in the merge commit
