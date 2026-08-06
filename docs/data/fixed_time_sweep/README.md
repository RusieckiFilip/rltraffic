# Fixed-time cycle-length tuning sweep (Tier 1 baseline)

**Date:** 2026-08-06 · **Required by:** `PREREGISTRATION.md` D5 — baselines receive the same tuning
budget as our method, **and the budget must be stated**. A tuned baseline whose tuning is
undocumented is as attackable as an untuned one.

## Protocol
- **Tuned on:** flow draws **1–5** (training pool, D4). **Never on the held-out pool 1000–1099.**
- **Objective:** `att_horizon` (the registered primary metric, A1), mean over the 5 draws.
- **Search space:** `k` = decision steps each green is held. Cycle = `n_greens × delta_time × k`,
  with `delta_time = 10`.
- **Only `k` is tuned.** Per-phase green splits are **not** tuned — see Limitations.

## Results (att_horizon, lower is better; chosen value in bold)

| scenario | greens | k=1 | k=2 | k=3 | k=4 | k=6 | k=8 | k=10 | k=14 | chosen |
|---|---|---|---|---|---|---|---|---|---|---|
| cf_hz1x1 | 8 | — | 318.81 | 286.65 | 270.53 | **261.86** | 265.48 | 274.69 | 291.47 | **k=6** (480 s) |
| cf_grid4x4 | 8 | **206.09** | 261.65 | 297.54 | 290.57 | 341.05 | — | — | — | **k=1** (80 s) |
| cf_cologne3 | 3 | — | 235.32 | **95.68** | 204.24 | 281.14 | — | — | — | **k=3** (90 s) |

The first sweep placed two optima on a boundary; it was extended until each optimum was interior or
hit a hard platform limit. `cf_hz1x1` and `cf_cologne3` have interior optima. `cf_grid4x4` sits on
`k=1`, the smallest value the platform can express — see Limitations.

## What this changed
Untuned `k=4` on `cf_grid4x4` scores ≈290, **worse than Random (265.75)**. Tuned `k=1` scores
**206.09**, between MaxPressure (169.05) and Random. **"Uncoordinated fixed-time loses to random on a
grid" was therefore a tuning artifact, not a coordination effect.** On `cf_cologne3` tuning is worth
roughly a factor of two (204 → 96).

## Limitations, to be stated in the paper
1. **Only cycle length is tuned.** Per-phase green splits stay equal, so this is the best
   *equal-split* fixed-time, not the best possible fixed-time. Demand-proportional splits (Webster)
   would be a stronger baseline and are not implemented.
2. **`cf_grid4x4`'s optimum sits on the boundary, and the binding constraint is ours, not the
   platform's.** `k=1` is the platform floor (5 s clearance, `delta_time > 5` ⇒ ≥5 s green per phase,
   80 s cycle over 8 greens). But our equal-split plan **cycles all 8 greens**, whereas a real grid
   plan serves a subset — 4 greens at `k=1` would be a 40 s cycle, expressible by the platform. So
   the true optimum is plausibly below what **our plan family** can express, though not below what the
   **platform** can.
3. **Schedule source differs by scenario**: `cf_hz1x1` uses the scenario's shipped
   `signal_plan_template.txt` (30 s greens, 5 s clearance); `cf_grid4x4` and `cf_cologne3` ship no
   plan and use our equal split. Recorded per-episode in each manifest as
   `fixed_time_schedule_source`.
