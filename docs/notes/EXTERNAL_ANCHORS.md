# External anchors — every point where a number of ours meets a number produced outside this codebase

**Created 2026-08-28**, at the author's instruction, because the answer to *"has anything been checked
against an outside value?"* had never been written down and the honest answer was **no**.

---

## 0. 🚨 THE SENTENCE FOR THREATS TO VALIDITY, in the form it goes into the paper

> **No number in this study has been checked against a value produced outside this codebase. One
> attempt was made, at the algorithm layer (IQL on D4RL, P8.3), and it did not resolve. The simulation
> and metric layer has no external anchor.**

**Status 2026-08-28: PARTIALLY SUPERSEDED by §2 below — the metric layer now has one weak external
anchor, and the sentence must be updated to say so rather than left standing.** It is kept verbatim
because it was true until this date and because the paper needs the history, not just the state.

---

## 1. What we do NOT have, stated first

- **No value of ours has ever been reproduced by a third party.**
- **No number of ours has been compared against the same quantity computed by different code on the
  same input.** Every check in this repo is a *reproduction* check — *does this number come back the
  same* — which is silent about whether it was right to begin with.
- **P8.3 is the only prior attempt and it did not resolve.** Its own packet: *"the published locomotion
  table is not reproducible"*, *"published comparators did not match the primary source"*, **"a failure
  to reject, not a demonstration of correctness"**. It is fenced from the paper.
- ⚠️ **`average_travel_time` is OUR platform's Python arithmetic** (`metrics/cityflow.py:234-254`),
  **not** a call into CityFlow's C++ engine. The vendored engine is not doing the metric for us.
- ⚠️ **The corpus stores the metric's OUTPUT (`att_per_step`) and not the vehicle depart/arrive times**,
  so a definition error is **not recoverable by arithmetic** — it would require re-running every
  simulation. That is what makes T1 the highest-stakes check in the project.

---

## 2. ✅ ANCHOR 1 — magnitude and ordering of MaxPressure against DataLight (2026-08-28, T2)

**Source, read from the primary document this session** (`arxiv.org/html/2303.10828`, DataLight,
Table 1). **Action interval 10 s — identical to ours.**

| controller | DataLight, HangZhou **16-intersection** HZ1 / HZ2 | ours, hangzhou **1×1** `bc-tyc` |
|---|---|---|
| FixedTime | 497.87 / 408.31 | — |
| **MaxPressure** | **284.44 / 327.62** | **247.75** (std 0.0, deterministic) |
| Advanced-CoLight (online SOTA) | 271.62 / 311.07 | — |
| DataLight | 261.56 / 298.18 | — |
| Random | — | **413.53** |

**Reading, and it is deliberately modest: our MaxPressure lands at 247.75 where an independent group's
MaxPressure on the same city's data at the same action interval lands at 284–328, and ours is a SINGLE
intersection where theirs is a 16-intersection grid — so a somewhat lower value is what one should
expect.** ⭐ **We are in the right universe, on the correct side of the comparison, with the
FixedTime > MaxPressure > SOTA ordering intact.** A metric that double-counted, used the wrong horizon,
or averaged the wrong population would not land here.

⚠️ **What this is NOT, and the author required the discrepancy be recorded rather than adjudicated:**
- **Different network.** Theirs is HZ1/HZ2, 16 intersections; ours is `hangzhou_1x1_bc-tyc_18041610_1h`,
  one intersection. **Not the same file, so this is a magnitude-and-ordering check and NOT a value
  match.**
- Different demand, different CityFlow version, possibly different phase sets and yellow timings.
- **A mismatch here would not automatically have been our error**, and an agreement is not proof.
- 🚨 **A second candidate figure — SOTL at 213.20 on our exact file — is HEARSAY and is NOT recorded as
  an anchor.** It came from a search summary; the page it was attributed to
  (LibSignal's dataset documentation) was fetched this session and **does not contain it**. It is
  written here only so nobody re-finds it and mistakes it for verified.

✅ **What the LibSignal page DID confirm, first-hand:** `hangzhou_1x1_bc-tyc_18041610_1h` is a
**1-intersection** dataset with a **3600 s** span — **exactly our horizon** (360 steps × 10 s). Our
episode length matches the benchmark's standard.

---

## 3. ✅ ANCHOR 2 — the ordinal check, weak and explicitly so (recorded 2026-08-28 at the author's instruction)

**It had never been written down, and until §2 it was the only external evidence this project held.**

> **MaxPressure beats fixed-time · MAPPO beats MaxPressure · random is worst · and the fixed-time
> controller is reproduced entry-for-entry by four independently trained models.**

**That ordering is the traffic-signal-control field's consensus, and it is external in the only sense
that matters here: it was not chosen by us and a corrupted metric would break it.** A metric with a
sign error, a wrong horizon, or a survivorship bias would not preserve a four-way ordering across
scenarios and controllers.

⭐ **The fourth clause is the sharpest and is ours alone:** on the `fixedtime` tier, `dt_nomix`,
`dt_spatial` and their replicates reproduce the fixed-time controller's action matrix **0 of 5760
entries differing** (P5.2's independent reviewer), from distinct weights. **A deterministic controller
is recovered exactly by learned models — which is a statement about the whole pipeline, from corpus to
training to rollout, that no single metric error survives.**

⚠️ **Why it is weak, stated so it is never oversold:** it is **ordinal, not cardinal**. It excludes
gross corruption and says nothing about a *uniform* scale error — if every ATT were multiplied by a
constant, this check passes unchanged. **§2 is what addresses the scale; this addresses the shape.**

---

## 4. What would strengthen this, in cost order

1. **T1's hand-computed ATT** (running 2026-08-28) — the only *definitional* validation of the primary
   metric. **Whatever it returns goes in the paper as such.**
2. **A published value on our exact file.** Not yet found; DataLight uses only multi-intersection HZ,
   and the one 1×1 figure encountered is hearsay (§2).
3. **A third party running our released artifact.** P10.0's release makes it possible; nothing else does.
