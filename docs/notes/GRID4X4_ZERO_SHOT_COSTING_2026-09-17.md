# Costing — a grid4x4 zero-shot point on SUMO, end to end (2026-09-17)

Requested by the author before deciding whether it goes before or after the paper draft. A costing, not an amendment: it registers nothing and decides nothing.

## What exists today (read from disk, 2026-09-17)
- **Subjects:** CityFlow grid4x4 DT checkpoints from P5.2, 5 seeds each for `mappo1000_dt_nomix_h4` and `mappo1000_dt_spatial_h4` (`output/p5_2/checkpoints/`). There is **no `mix50` analogue**, so hz1x1's two-subject design (A17(c)) does not transfer without a registration choice. P5.2's recorded headline ("the spatial harm is confined to the best-data tier") points to the non-spatial DT as the subject, which is the `DTAgent` path.
- **Demand:** `scenarios/draws/cityflow_grid4x4/` holds draws **0–5 and 1000–1099 only** (CityFlow parents). **The probe / few-shot band 201–300 does not exist on grid4x4**, and **0 of 106** existing draws has a SUMO parity config.
- **SUMO network and routes:** RESCO's `grid4x4.net.xml` and `grid4x4.zip` (route member `grid4x4_1.rou.xml`, CC BY-NC-SA 4.0) are in the gitignored candidates tree. A16's identity alignment branch for grid4x4 exists, exercised only when `RLTRAFFIC_GRID4X4_RESCO` names that tree.
- **Code:** `offline/parity.py` is single-scenario (`_DECLARED_STEM`, `DEFERRED` 75(d)); `aligned_env.DECLARED_SCENARIO`, `transfer_calibration.SCENARIO_KEY` and `transfer_curve.SCENARIO_KEY` are hz1x1 constants. `agent/DTAgent.py` holds one scalar `_target_rtg` (`DEFERRED` 78, not frozen); `agent/SpatialDTAgent.py` already takes per-intersection mappings (`:77`, `:436`, `:479`).
- **Rates:** **no grid4x4 SUMO simulation has ever run** (A15(iii)). The only committed grid4x4 : hz1x1 per-episode ratios are on CityFlow: 2.07×, 2.51× and 3.13× (`p8_4a_admission.json`, `p8_4a_admission_escalated.json`, `p8_4b_g0_reference.json`). hz1x1 SUMO bases measured this cycle: a MaxPressure probe episode 11.19 s (P7.2b, n = 100); 100-episode collection 18 min; observed cells 1.494 s/cell effective at 12 workers (stage 1, n = 1,200, 2026-09-17).

## Implementer days (registered estimates in brackets; ranges calibrated by this cycle's actuals, where every hz1x1 task needed several fix rounds)
| item | days |
|---|---|
| `DEFERRED` 75 for grid4x4: parity-bind RESCO's route file (A15(g)'s values), generalise `parity.py`, render parity + teleport-free `.sumocfg` per draw, **render CityFlow parents for 201–300**, re-run CAP(E) and record it | 1.5–2 [~1] |
| `DEFERRED` 78: per-intersection target in `DTAgent`, every existing checkpoint round-tripping bit-identically; A17(e)'s per-intersection Rule B in the calibration code | 1–1.5 [~0.5 + review] |
| the SUMO chain parameterised by scenario (aligned env, calibration, transfer curve, collect's logger-side door), the observer and halting check on 16 intersections, and **a P7.1-style anchor freeze on grid4x4 SUMO** (no frozen values exist to reproduce) | 2–3 |
| P7.2b-equivalent probe and calibration artifact, the logged corpus, A17(f) per intersection | 1–1.5 |
| rate measurement, pilot, pre-flight, two merge reviews, fix rounds | 2–3 |
| **total** | **≈ 7.5–11 days, central ≈ 9** |
| not implementer time: an A-row before any grid4x4 SUMO number (subject choice with no `mix50` analogue; per-intersection Rule B as executed; the anchors) | ≈ ½ day, coordinator + author |

## Compute hours (UNMEASURED multiplier m = 2–6× hz1x1 per episode: 2–3× is the committed CityFlow ratio; the upper end allows for SUMO's per-lane TraCI and observer work on 240 lanes)
| item | hz1x1 measured | grid4x4 at m = 2–6 |
|---|---|---|
| probe 100 + corpus 100 MaxPressure episodes (sequential, as run) | ≈ 37 min | ≈ 1.2–3.7 h |
| anchor freeze, one-cell rate measurement, pilot | — | ≈ 0.5–1 h |
| confirmatory cells, 1,200 (two subjects) / 700 (one) | 30 min | ≈ 1.0–3.0 h / 0.6–1.7 h |
| full declared set, 4,700 (two subjects) / 2,700 (one) | 1.95 h | ≈ 3.9–11.7 h / 2.2–6.7 h |
| **total, confirmatory path** | | **≈ 3–8 h** |
| **total, full declared set** | | **≈ 6–16 h** |

Unmeasured risk beyond the rate: whether 12 workers fit at all with a 16-agent DT and a 240-lane SUMO per process (GPU and host memory). The first act of any grid4x4 work is a one-cell measurement, which replaces every range above.
