# BRIEF #11 — P4.4: BC, %BC and IQL on the same corpus

**Mode:** Claude Code, on a task branch. **Branch:** `task/p4.4-offline-baselines`
**Issued:** 2026-08-11 by the Master chat. **Base:** `main` (P2.6, P3 and P4 all merged).
**Filter:** this decides what P4's result *means*. Until it lands, the DT's margin is uninterpretable.

---

## 1. Why this before P4.3

P4 passed its gate: `ATT_MADT = 104.9558` against MaxPressure 176.8912 and MAPPO@1000 105.5820,
paired on all 100 held-out draws, winning 100/100 against MaxPressure and MAPPO@500.

**The DT beat the policy whose data it trained on, by 0.6 %.** That is C1's central question — *does
the model exceed its data* — and **we cannot yet say whether sequence modelling earned it.** A
behaviour-cloning model on the same corpus might do the same, in which case the DT's architecture
contributed nothing and the honest paper says so and pivots weight to the ladder and shift findings.
§1 has said since 2026-07-10 that this is non-negotiable: *"if BC matches MADT, sequence modeling
adds nothing — must be tested."*

**P4.3 (RTG calibration) is deliberately queued behind this.** Calibrating a prompt for a mechanism
whose contribution is unmeasured optimises something we cannot attribute. P4.3 also now has a
measured motive waiting for it — the DT sits **outside its training RTG support for 20.8 % of every
episode**, and the declared target is not the best one (106.46 declared vs 102.05 at target 0).

---

## 2. Frozen protocol — identical to P4, and that is the point

**Any deviation makes the comparison unusable.** Everything below is already settled; do not
re-derive it.

- **Evaluation:** all 100 registered held-out draws `1000–1099`, **paired**, every arm on the
  identical draw set (`PREREGISTRATION.md` §8, amendment A5). A comparison not over shared draws is
  **void**.
- **Training data:** the `cf_hz1x1` **`mappo1000` tier** of `datasets_v11/`, the same tier P4 used.
  Draws `1–999` only; the loader raises on held-out draws — do not route around it.
- **Statistics:** paired **Wilcoxon** over the shared draws, plus mean ± 95 % CI. **Effect sizes are
  mandatory beside every p-value** (§8). Report `att_horizon` **and** `vehicle_count` at the horizon
  **unconditionally**, with the draw ids (A5).
- **Seeds:** ≥5 training seeds, the same five P4 used, evaluated on the same 100 draws so **seed and
  draw are crossed, not nested.** (The corpus's own MAPPO tiers are confounded — review D16 — which
  is exactly why reported comparisons use this design.)
- **Leakage (§6):** every reported model is the checkpoint at a **fixed, pre-declared step count**;
  hyperparameters tuned on this scenario then frozen; and **baselines get the same tuning budget as
  the DT did** — that is this task's whole point, so an under-tuned BC is a straw man and must be
  reported as untuned if it is one.
- **Normalisation statistics fitted on the training split only**, recording the draw ids.

## 3. Deliverables — scope fence

**In scope: BC, %BC, IQL.** Out of scope: RTG calibration (P4.3), any spatial layer (P5.1), any new
scenario. The ≤2-source-file limit stands; `agent/` is writable for *new* files only —
`agent/base.py`, `agent/utils/utils.py` and `agent/MAPPOAgent.py` are frozen (verified: there is no
`agent/**` glob).

### 3.1 BC and %BC
Behaviour cloning on the same windows the DT saw. **%BC filters to the top-10 % of trajectories by
return** (§1). Use `offline/dataset.py` — it already yields per-intersection windows with masks; do
not write a second loader.

### 3.2 IQL
Independent per-intersection. ⚠️ **Registered constraint, and getting it wrong hands the DT an
unearned win:** `terminated` is hardcoded `False` and every episode ends by **time-limit
truncation**, so IQL **must bootstrap through the boundary** and must never treat the horizon as
absorbing (Decisions Log 2026-07-26). Treating a timeout as terminal causes systematic value
underestimation near episode end. **A test must pin this.**

### 3.3 The canonical checkpoint digest — `DEFERRED` 29, introduced here
A checkpoint's file hash depends on **both** its filename (`torch.save` names the zip root after the
output file) **and** its provenance block. **No claim of the form "the model reproduces
byte-identically" is testable at file level.** Ship a small helper computing **sha256 over the
`state_dict` tensor bytes in sorted key order** — filename- and provenance-independent — record it
in every checkpoint artifact you emit, and use it for every determinism claim. Keep the file sha256
for what it does prove: **transport integrity**. ~10 lines plus a test that two differently-named
saves of identical weights produce the **same** canonical digest and **different** file hashes.

## 4. Tests — the load-bearing ones

- **The inference path is exercised with real statistics.** P4's review found its entire online path
  unprotected because no test ever constructed the agent with `stats=`; three mutations survived all
  58 tests and one cost **+3.8 ATT**, most of P4's margin. **Assert that what `act()` feeds the model
  equals what training fed it, exactly, for every step of a real episode.** This is not optional.
- **IQL bootstraps through truncation** — mutate to treat the horizon as terminal and show the test
  fails.
- **%BC's filter selects what it claims** — on a fixture where the top-10 % is known by construction.
- **Padded positions cannot contribute to any loss** (`PAD_ACTION = -1`; `ignore_index=-1`).
- **Determinism by canonical digest**, not by file hash (§3.3).

## 5. Definition of Done
- [ ] `docs/plans/p4.4.md` first, with the **declared step count** for each method
- [ ] BC, %BC, IQL + tests; red-first; mutation proofs pasted
- [ ] All arms evaluated on the same 100 held-out draws, paired, with Wilcoxon + effect sizes
- [ ] **The attribution stated plainly**: does BC match the DT, or not, with the paired test
- [ ] Canonical checkpoint digest shipped and used
- [ ] Full `pytest -q` against the 541 baseline; zero frozen-file modifications
- [ ] Return Packet at `docs/returns/P4.4.md`
- [ ] **§6 checkbox ticked in the merge commit**
- [ ] **Independent review before merge** — critical path

## 6. Return Packet — task-specific
1. The paired DT-vs-BC comparison, with effect size, stated as a plain sentence.
2. Whether each baseline was tuned, and with what budget. **An untuned baseline is reported as
   untuned** (§6.3) — as MaxPressure now is, because it has no parameters to tune.
3. Anything in §2 that disagreed with the repo. **The repo wins; say so loudly.**
4. **If BC matches the DT: say so first, before any interpretation.** That is a registered outcome
   (`PREREGISTRATION.md` §10), not a failure, and the paper already knows what it publishes under it.
