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
