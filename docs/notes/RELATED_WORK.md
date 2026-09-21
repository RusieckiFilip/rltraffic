# Related work — the paper's citation and positioning register

**Created 2026-09-19, at the author's instruction**, because the answer to *"where is the citation list?"* was: nowhere.
`PROJECT_PLAN` §1 carries rulings *about* several of these works (dated corrections inside the C3 row); `docs/notes/EXTERNAL_ANCHORS.md`
carries numeric anchors, a different thing. **This file is where P10.1's related-work matrix draws from.** Rules for every entry:
the identifier as resolved (title, authors, venue — read from the source page, with the date); what the work does, **as read from the
source by a named person on a named date** — never from memory; why it matters to us; the positioning sentence, if one is binding;
and a status line — *read in full: by whom, when* — that stays **OPEN** until it is true. A work cited from its abstract is marked as such.

---

## 1. Finding of 2026-09-19 — the author (Filip Rusiecki), searched that day; every claim from the source, not from memory

### 1.1 The closest prior art to C3: the DaRL-LibSignal group (Hua Wei, Longchao Da; ASU)
They run CityFlow → SUMO transfer **with SUMO as the "real" side**, creating the gap by **modifying SUMO's acceleration, deceleration,
start-up delay and container capacity** — SUMO's default setting has the same parameters as CityFlow. Their line:

| work | identifier, as resolved 2026-09-19 (coordinator, arXiv/publisher page) | role for us |
|---|---|---|
| Sim2Real Transfer for TSC | **CASE 2023** — venue as given by the author; **identifier not resolved in-session** | the line's origin |
| Uncertainty-aware GAT (UGAT) | Da et al., **CDC 2023** — already in §1 (2026-08-17) | grounded action transformation, first form |
| PromptGAT | arXiv **2308.14284** — *Prompt to Transfer: Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning*, Da, Gao, Mei, Wei, **AAAI 2024** | the reference point a referee will hold us against |
| Joint-Local GAT | arXiv **2507.15174** — *Joint-Local Grounded Action Transformation for Sim-to-Real Transfer in Multi-Agent Traffic Control*, Turnau, Da, Vo, Al Rafi, Bachiraju, Chen, Wei, **RLC/RLJ 2025** | the multi-agent extension |
| LibSignal | arXiv **2211.10649**; journal: *Machine Learning* (Springer), **DOI 10.1007/s10994-023-06412-y**, Mei, Lei, Da, Shi, Wei — DOI confirmed from the arXiv page; the Springer page itself sat behind a cookie gate in-session | **they own the converter our scenarios came through** (A14(a), `DEFERRED` 77) |

**Why this matters — the sentence for the FIRST PARAGRAPH, binding:** *They ADAPT — grounded action transformation changes the source
domain to match the target. We adapt nothing and measure what transfers.* Without that distinction in the opening, a referee reads us as
a weaker PromptGAT. It is the same distinction §1 already rules as the C3 positioning — *the pre-adaptation residual gap from CityFlow to
SUMO under dynamics parity* (2026-08-2x ruling, C3 row) — now stated against the group by name.

**What §1 already established about these works, so this file does not restate it differently:** PromptGAT's Table 2 reports
Direct-Transfer rows (a frozen DQN without adaptation) as a *result*, so *"nobody measures zero-shot cross-simulator transfer"* may
never be written; its target is SUMO with **deliberately altered** vehicle parameters (V1–V4) and V0 appears nowhere in its results;
even default-vs-default is confounded (`tau` 1.0 vs 2.0, measured at +49 % ATT in `P7.0_vtype_investigation.md`); LibSignal disclaims
transfer in its own words and its A.5 calibration covers routing, virtual nodes, signal intervals and departure ordering but **not**
vehicle dynamics; its Table 3 shows a policy-dependent post-calibration gap; the *native authorship* pillar was withdrawn on 2026-09-11 —
every pair here is converted, through LibSignal's toolchain, and the paper says so (A14). **Read §1's C3 row before drafting; this file
adds to it and contradicts none of it.**

### 1.2 Also to cite and engage
- **LibSignal's journal version** (above) states that *method performance varies across simulators but rankings within a simulator are
  generally consistent*. **Positioning:** our zero-shot result is a stronger claim than that — a policy trained in one simulator scores
  ρ_sumo = +1.89 in the other under dynamics parity (P7.3a) — and must be positioned against their sentence explicitly, not beside it.
- **OffLight** — arXiv **2411.06601**, *OffLight: An Offline Multi-Agent Reinforcement Learning Framework for Traffic Signal Control*,
  Bokade & Jin (submitted 2024-11-10, revised 2025-03-18). Offline MARL for TSC with **heterogeneous behaviour policies** — the nearest
  *offline* work; not transfer. Engages C1 (the data-quality ladder) directly.
- **X-Light** — arXiv **2404.12090**, *X-Light: Cross-City Traffic Signal Control Using Transformer on Transformer as Meta Multi-Agent
  Reinforcement Learner*, Jiang, Li, Wei, Xiong, Ruan, Lu, Mao, Zhao, **IJCAI 2024**. Cross-**city**, not cross-backend (§1 already
  draws that line).
- **DTLight** — arXiv **2312.07795**, *Traffic Signal Control Using Lightweight Transformers: An Offline-to-Online RL Approach*, Huang,
  Wu, Boulet. Already in the project files: the **K ∈ {1, 2}** context-length refutation target; its pure-offline DT collapsed on
  Grid 4×4 (446.8 from weak data vs behaviour 48.39), which is why P5.2's grid4x4 result is read the way it is.

### 1.3 Status — OPEN items, and who closes them
| item | status |
|---|---|
| PromptGAT (2308.14284) **read in full** | §1 records the author read v6 in full on 2026-08-17 (Tables 1–2). **Re-read before the related-work section is drafted**; the P10 writer names the date here. |
| Joint-Local GAT (2507.15174) **read in full** | **OPEN** — abstract-level only, as of 2026-09-19. Must be read in full before the section is drafted. |
| Sim2Real Transfer for TSC (CASE 2023) identifier | **OPEN** — resolve the DOI / proceedings entry before citing. |
| LibSignal journal sentence on within-simulator rankings | quoted by the author from the source 2026-09-19; **page/section to be recorded** when the section is drafted. |
| OffLight, X-Light | abstract-level; read in full before the matrix row is written. |

**Action recorded so it is not rediscovered:** whoever drafts related work (P10.1) reads PromptGAT and Joint-Local GAT in full first and
records the date above. The first paragraph carries the ADAPT-versus-MEASURE distinction.

---

## 2. The author's assessment after reading the related work — 2026-09-19. A JUDGEMENT, not a measurement; to be checked against the full sources when someone reads them properly

*In the author's words, lightly compressed; nothing added.*

I had been treating cross-engine transfer for TSC as unoccupied ground. It is not. The DaRL group has four papers on the CityFlow → SUMO
pair and an AAAI 2024 among them. That lowers the novelty of C3 as a topic, and it changes what our introduction has to do.

**What survives as ours: they adapt, we do not.** Their question is *how do we close the gap*; ours is *how much gap is really there, and
what does it depend on*. That is a real difference but a narrower one than I assumed, and it has to be argued explicitly against PromptGAT
rather than established by omission. A referee who knows that line will ask why we did not compare against grounded action transformation.
*"We measure something else"* is the right answer and it belongs in the first paragraph, not in a rebuttal letter.

**What still has no counterpart that I found:** the data-quality ladder with thresholds; the k = 200 anchor showing the zero-shot advantage
comes from source-data quality rather than domain proximity; spatial mixing harming at the best-data tier; and the registration apparatus.
Those four share one thesis — *what is in the data dominates which world it came from and what is built on top of it* — and no paper I saw
makes that argument end to end.

**Where this leaves the paper, in my estimate:** T-ITS is realistic but harder than I said before. Not because the work is weaker, but
because we are entering an area with an active publishing group and must position against them openly. The three contributions that are
not theirs are what carries it.

**Two things I would do before the draft:** read PromptGAT and Joint-Local GAT in full, and check LibSignal's *"rankings are generally
consistent within a simulator"* sentence against what our zero-shot number actually claims — they may be closer or further apart than they
look from the abstracts. *(Both are OPEN items in §1.3.)*

### 2.1 The coordinator's check of §2 against the project's own record (2026-09-19) — what the record confirms, and three qualifiers it adds

- **Confirmed by the identifiers resolved in §1.1:** four works on the pair, one at AAAI 2024, one at RLC/RLJ 2025; the group owns the
  converter (A14(a), `DEFERRED` 77). The ADAPT-versus-MEASURE distinction is §1's *pre-adaptation residual gap* ruling, now aimed at a name.
- **Qualifier 1 — the ladder.** `PROJECT_PLAN` §1's claim constraint of 2026-08-12: the data-quality-ladder *concept* is 2021 prior art in
  another domain (arXiv:2112.02845, Meng et al., StarCraft II, "the first offline MARL dataset with diverse quality levels", MADT, few- and
  zero-shot transfer), and **the sentence *"we introduce the study of offline MARL performance as a function of data quality" must never be
  written.*** What is ours is the **TSC instantiation** — six tiers with registered thresholds, multi-agent local rewards, dual-backend
  paired scenarios, the method × tier grid, the selection-mechanism decomposition — and OffLight's heterogeneous behaviour policies sit
  next to it. §2's *"no counterpart"* is true of the ladder **as executed here**, not of the idea, and the paper cites 2112.02845 where C1 is
  introduced.
- **Qualifier 2 — the anchor.** *"Source-data quality rather than domain proximity"* is the reading A19(c) registers, and it rests on a
  comparison in which the only target-domain anchor is MaxPressure-quality data (A18(a)); **there is no high-quality target-domain anchor**,
  by A18(b), so the clause is a difference in demonstration quality shown on one axis, with the missing anchor named as a limitation in the
  C3 section — not a two-axis separation. The registration says exactly this; the paper may not say more.
- **Qualifier 3 — spatial mixing.** *"Spatial mixing harming at the best-data tier"* is the sentence A22 (2026-09-19) makes **conditional on
  P5.4's footprint**: if the trained models' neighbour influence is trivial, the registered alternative is *"adding the cross-intersection
  attention path destabilises training"*. The contribution stands in either form; its wording is not yet fixed, by design.
- **On the thesis and the venue:** the four-contribution thesis is supported on hz1x1 — one intersection — and on grid4x4 only for C1;
  P7.3d is what makes the C3 half of it more than one network, and the paper's scope sentence for that is A21's. The T-ITS judgement is the
  author's; the coordinator records it and does not grade it.

---

## 3. Method-section notes — things a reimplementing reader will get wrong unless the text says them

- **Candidate C3 sentence, NOT to be written before P7.3d's campaign numbers exist (BRIEF_39 B.3-4, 2026-09-21):** on grid4x4 under
  full dynamics parity, the per-intersection ratio of SUMO to CityFlow MaxPressure probe returns runs −16.6 % to +4.2 %, 12 of 16 below
  1.0, eleven more than 4 SE from 1.0 — a residual engine gap with a direction and spatially heterogeneous magnitude. The DaRL line
  adapts the gap away and therefore cannot report its spatial structure; we measure it. An observation with standard errors, never a test.

- **r (A22(c)) is the mean over NODES of the mean over each node's one-hop neighbours, divided by the mean over nodes of self-influence
  — NOT the mean over neighbour PAIRS.** On the 4 × 4 lattice the degrees are {2: 4, 3: 8, 4: 4}, so the two aggregations differ
  (0.06388 against 0.06216 on seed 101, 2.7 %). The coordinator, who wrote A22(c), took the pair fork on first recomputation
  (2026-09-20, `output/p5_4_runs/coord_verify/independent_r_seed101.txt`); a reader will too. State the aggregation explicitly.
- **F is bit-reproducible only at the recorded thread count** (8; `protocol.torch_threads`); across thread counts the maximum relative
  difference is 4.0e-07 and the verdict is unaffected. One sentence in the reproducibility section, beside the CityFlow/SUMO
  within-machine statements.
