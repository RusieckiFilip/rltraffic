# PRE-REGISTRATION — Offline MADT for Traffic Signal Control
**Registered:** <YYYY-MM-DD>  ·  **Git tag:** `v0.1-prereg`  ·  **Doc sha256:** <fill after freezing>
**Registered before:** any offline corpus collection (P2) and any MADT training (P4).

> Task P0.4. This document is what turns "pre-registered" from a word in the paper into a verifiable
> fact. Freeze it, commit it, tag it, and do not edit it afterwards — later changes go in an
> amendments section with their own dates.

## 1. Research questions
- **RQ1 (C1):** How does offline MADT performance vary with behavior-data quality, measured as
  normalized return (random = 0, MaxPressure = 100) rather than by policy name?
- **RQ2 (C2):** Does offline sequence modeling degrade more gracefully than online MARL under
  scenario shift (demand shocks, incidents, sensor dropout)?
- **RQ3 (C3):** What does the CityFlow → SUMO transfer curve look like from zero-shot through
  few-shot (k ∈ {5, 20, 100}) to full retrain, under within-backend-normalized metrics?

## 2. Designs, fixed in advance
- **C2 is a 2×2:** {nominal, shift-augmented} training data × {MADT, MAPPO}, plus mechanism ablations
  (context length K, dataset diversity, calibrated vs naive RTG prompting).
- **C1 ladder:** tiers labeled post hoc by measured normalized return; provenance policies listed
  separately.
- **C3:** transfer curve, within-backend normalization against MaxPressure and fixed-time anchors.

## 3. Primary metric and reward
- Primary metric: average travel time. Secondary: queue length, throughput, CO2 proxy.
- Primary reward: `queue_length`. PressLight reported as a robustness appendix.
- Safety latch (P2.3): Spearman correlation of queue- vs pressure-returns against average travel time
  over the corpus; if pressure clearly wins, the decision is reversed **before P4** at zero corpus cost.

## 4. Gates declared in advance (so a failed gate is a result, not a pivot)
- **P4.2 gate:** DT-offline ≥ MaxPressure **AND** within 5% of the best available online policy on
  average travel time, 5 seeds with CIs.
- **P7.0 gate:** if state-feature distributions across backends show pathological shift, C3 is
  descoped to a limitations study.
- **P4.4:** if BC-on-expert matches MADT, this is reported as a finding; the headline weight moves to
  the ladder / shift / calibration results.

## 5. Statistics
≥5 seeds for every reported cell; mean ± 95% CI; paired tests against the strongest baseline;
environment stochasticity supplied by flow randomization (P2.0), not by policy seeds alone.

## 6. What would falsify our claims
- C1: no monotone relationship between data quality and MADT performance, and no interpretable
  non-monotonicity either.
- C2: MADT degrades at least as fast as domain-randomized MAPPO across all perturbation families.
- C3: transfer gap indistinguishable from the interface-mismatch control (P7.4).

## 7. Amendments
| Date | Change | Reason |
|---|---|---|
