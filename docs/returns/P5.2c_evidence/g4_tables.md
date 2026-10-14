<!-- STATEMENTS -->
| statement | tier | kind | `att_engine` — **PRIMARY** (Rule R) | `att_ours` | definition-dependent |
|---|---|---|---|---|---|
| Q0.d1 | mappo1000 | quantity | +40.0210 [+36.4788, +43.5631] (resolves: the CI lies entirely above zero) | +39.5649 [+36.0510, +43.0787] (resolves: the CI lies entirely above zero) | no (on its reading) |
| Q0a | mappo1000 | verdict | **HELD** +27.4534 [+24.1365, +30.7702] | **HELD** +27.1004 [+23.8150, +30.3858] | no (on its verdict) |
| Q0b | mappo1000 | verdict | **FAILED** I = -12.5676 [-16.4943, -8.6409]; I < 0 on 70 of 100 draws; reverses on seed 101 | **FAILED** I = -12.4645 [-16.3575, -8.5714]; I < 0 on 70 of 100 draws; reverses on seed 101 | no (on its verdict) |
| Q0.stop_rule | mappo1000 | verdict | **CONTINUE** d4 +27.4534 [+24.1365, +30.7702] | **CONTINUE** d4 +27.1004 [+23.8150, +30.3858] | no (on its verdict) |
| Q1 | — (corrected IQL) | verdict | **HELD** 15 of 19 (k = 14) | **HELD** 15 of 19 (k = 14) | no (on its verdict) |
| Q1 | — (as committed IQL) | verdict | **HELD** 15 of 19 (k = 14) | **HELD** 15 of 19 (k = 14) | no (on its verdict) |
| Q1b | maxpressure | verdict | **HELD** 160.7680 vs 167.492 (4.01 %) | **HELD** 167.5618 vs 167.492 (0.04 %) | no (on its verdict) |
| Q1b | fixedtime | verdict | **HELD** 199.7048 vs 206.9318 (3.49 %) | **HELD** 207.1192 vs 206.9318 (0.09 %) | no (on its verdict) |
| Q1b.both | — | verdict | **HELD** both cells hold | **HELD** both cells hold | no (on its verdict) |
| ranking | maxpressure | context | bc < dt_nomix < dt_spatial < bc_top10_perix < iql < bc_top10 | bc < dt_nomix < dt_spatial < bc_top10_perix < iql < bc_top10 | no (on its reading) |
| ranking | fixedtime | context | iql < dt_spatial = dt_nomix < bc = bc_top10_perix < bc_top10 | iql < dt_spatial = dt_nomix < bc = bc_top10_perix < bc_top10 | no (on its reading) |
| ranking | random (corrected IQL) | context | iql < dt_spatial < dt_nomix < bc < bc_top10_perix < bc_top10 | iql < dt_spatial < dt_nomix < bc < bc_top10_perix < bc_top10 | no (on its reading) |
| ranking | random (as committed IQL) | context | iql < dt_spatial < dt_nomix < bc < bc_top10_perix < bc_top10 | iql < dt_spatial < dt_nomix < bc < bc_top10_perix < bc_top10 | no (on its reading) |
| Q2a | maxpressure | verdict | **FAILED** measured first bc (predicted bc_top10_perix) | **FAILED** measured first bc (predicted bc_top10_perix) | no (on its verdict) |
| Q2a | fixedtime | verdict | **FAILED** measured first iql (predicted dt_nomix) | **FAILED** measured first iql (predicted dt_nomix) | no (on its verdict) |
| Q2a | random (corrected IQL) | verdict | **FAILED** measured first iql (predicted dt_nomix) | **FAILED** measured first iql (predicted dt_nomix) | no (on its verdict) |
| Q2a | random (as committed IQL) | verdict | **FAILED** measured first iql (predicted dt_nomix) | **FAILED** measured first iql (predicted dt_nomix) | no (on its verdict) |
| Q2b | maxpressure | verdict | **FAILED** 11 of 15 (range 11..11) | **FAILED** 11 of 15 (range 11..11) | no (on its verdict) |
| Q2b | fixedtime | verdict | **FAILED** 7 of 15 (range 7..9) | **FAILED** 7 of 15 (range 7..9) | no (on its verdict) |
| Q2b | random (corrected IQL) | verdict | **FAILED** 8 of 15 (range 8..8) | **FAILED** 8 of 15 (range 8..8) | no (on its verdict) |
| Q2b | random (as committed IQL) | verdict | **FAILED** 8 of 15 (range 8..8) | **FAILED** 8 of 15 (range 8..8) | no (on its verdict) |
| Q2b.aggregate | — (corrected IQL) | verdict | **FAILED** tiers fixedtime FAILED, maxpressure FAILED, random FAILED; overall 26/45 (range 26..28), not scored | **FAILED** tiers fixedtime FAILED, maxpressure FAILED, random FAILED; overall 26/45 (range 26..28), not scored | no (on its verdict) |
| Q2b.aggregate | — (as committed IQL) | verdict | **FAILED** tiers fixedtime FAILED, maxpressure FAILED, random FAILED; overall 26/45 (range 26..28), not scored | **FAILED** tiers fixedtime FAILED, maxpressure FAILED, random FAILED; overall 26/45 (range 26..28), not scored | no (on its verdict) |
| Q2b_hard | maxpressure | count | 2 of 6 (range 2..2); no threshold is registered | 2 of 6 (range 2..2); no threshold is registered | no (on its reading) |
| Q2b_hard | fixedtime | count | 2 of 6 (range 2..4); no threshold is registered | 2 of 6 (range 2..4); no threshold is registered | no (on its reading) |
| Q2b_hard | random | count | 3 of 6 (range 3..3); no threshold is registered | 3 of 6 (range 3..3); no threshold is registered | no (on its reading) |
| D9 | maxpressure | context | seed 101: bc first; seed 202: bc first; seed 303: dt_nomix first; seed 404: bc first; seed 505: bc first | seed 101: bc first; seed 202: bc first; seed 303: bc first; seed 404: bc first; seed 505: bc first | **yes** (on its reading) |
| D9 | fixedtime | context | seed 101: iql first; seed 202: iql first; seed 303: iql first; seed 404: iql first; seed 505: iql first | seed 101: iql first; seed 202: iql first; seed 303: iql first; seed 404: iql first; seed 505: iql first | no (on its reading) |
| D9 | random (corrected IQL) | context | seed 101: iql first; seed 202: iql first; seed 303: iql first; seed 404: iql first; seed 505: iql first | seed 101: iql first; seed 202: iql first; seed 303: iql first; seed 404: iql first; seed 505: iql first | no (on its reading) |
| D9 | random (as committed IQL) | context | seed 101: iql first; seed 202: iql first; seed 303: iql first; seed 404: iql first; seed 505: iql first | seed 101: iql first; seed 202: iql first; seed 303: iql first; seed 404: iql first; seed 505: iql first | no (on its reading) |
| Q3a | mappo1000 | context | rank 1 of 6 (range 1..1) | rank 1 of 6 (range 1..1) | no (on its reading) |
| Q3a | maxpressure | verdict | **FAILED** rank 2 of 6 (range 2..2) | **FAILED** rank 2 of 6 (range 2..2) | no (on its verdict) |
| Q3a | fixedtime | verdict | **FAILED** rank 2 of 6 (range 2..3, tied with dt_spatial) | **FAILED** rank 2 of 6 (range 2..3, tied with dt_spatial) | no (on its verdict) |
| Q3a | random (corrected IQL) | verdict | **FAILED** rank 3 of 6 (range 3..3) | **FAILED** rank 3 of 6 (range 3..3) | no (on its verdict) |
| Q3a | random (as committed IQL) | verdict | **FAILED** rank 3 of 6 (range 3..3) | **FAILED** rank 3 of 6 (range 3..3) | no (on its verdict) |
| Q3b | mappo1000 | context | -2.4360 [-2.5857, -2.2864] | -2.4303 [-2.5796, -2.2810] | no (on its reading) |
| Q3b | maxpressure | verdict | **HELD** -0.6664 [-0.8475, -0.4853]; sub-prediction NOT matched | **HELD** -0.7294 [-0.9139, -0.5450]; sub-prediction NOT matched | no (on its verdict) |
| Q3b | fixedtime | verdict | **NOT RESOLVED** exactly zero (identical episodes); sub-prediction matched | **NOT RESOLVED** exactly zero (identical episodes); sub-prediction matched | no (on its verdict) |
| Q3b | random | verdict | **HELD** -5.0453 [-5.5102, -4.5804]; sub-prediction matched | **HELD** -5.0360 [-5.5005, -4.5716]; sub-prediction matched | no (on its verdict) |
| Q3b.aggregate | — | verdict | **NOT RESOLVED** fixedtime NOT RESOLVED, maxpressure HELD, random HELD | **NOT RESOLVED** fixedtime NOT RESOLVED, maxpressure HELD, random HELD | no (on its verdict) |
| Q3c | mappo1000 | context | -11.4093 [-14.0917, -8.7269], resolves for the DT; best non-DT bc | -11.1329 [-13.7932, -8.4726], resolves for the DT; best non-DT bc | no (on its reading) |
| Q3c | maxpressure | reading | +0.9059 [-0.1617, +1.9735], NOT RESOLVED; best non-DT bc; reverses on 1 of 5 seeds (303) | +1.2993 [+0.2395, +2.3592], resolves against the DT; best non-DT bc | **yes** (on its reading) |
| Q3c | fixedtime | reading | +3.1835 [+2.9418, +3.4253], resolves against the DT; best non-DT iql | +3.1669 [+2.9256, +3.4082], resolves against the DT; best non-DT iql | no (on its reading) |
| Q3c | random (corrected IQL) | reading | +56.7834 [+53.8995, +59.6672], resolves against the DT; best non-DT iql | +56.8432 [+53.9794, +59.7069], resolves against the DT; best non-DT iql | no (on its reading) |
| Q3c | random (as committed IQL) | reading | +64.3490 [+62.0765, +66.6216], resolves against the DT; best non-DT iql | +64.3603 [+62.1046, +66.6160], resolves against the DT; best non-DT iql | no (on its reading) |
| Q4.d | mappo1000 | quantity | +40.0210 [+36.4788, +43.5631] (resolves: the CI lies entirely above zero) | +39.5649 [+36.0510, +43.0787] (resolves: the CI lies entirely above zero) | no (on its reading) |
| Q4.d | maxpressure | quantity | +0.6895 [+0.5855, +0.7934] (resolves: the CI lies entirely above zero) | +0.7267 [+0.6206, +0.8328] (resolves: the CI lies entirely above zero) | no (on its reading) |
| Q4.d | fixedtime | quantity | exactly zero (identical episodes) | exactly zero (identical episodes) | no (on its reading) |
| Q4.d | random | quantity | -0.7098 [-1.3207, -0.0990] (resolves: the CI lies entirely below zero); reverses on 1 of 5 seeds (202) | -0.7063 [-1.3172, -0.0953] (resolves: the CI lies entirely below zero); reverses on 1 of 5 seeds (202) | no (on its reading) |
| Q4a | maxpressure | verdict | **HELD** +0.6895 [+0.5855, +0.7934] | **HELD** +0.7267 [+0.6206, +0.8328] | no (on its verdict) |
| Q4a | fixedtime | verdict | **NOT RESOLVED** exactly zero (identical episodes) | **NOT RESOLVED** exactly zero (identical episodes) | no (on its verdict) |
| Q4a | random | verdict | **FAILED** -0.7098 [-1.3207, -0.0990]; reverses on 1 of 5 seeds (202) | **FAILED** -0.7063 [-1.3172, -0.0953]; reverses on 1 of 5 seeds (202) | no (on its verdict) |
| Q4a.aggregate | — | verdict | **FAILED** fixedtime NOT RESOLVED, maxpressure HELD, random FAILED | **FAILED** fixedtime NOT RESOLVED, maxpressure HELD, random FAILED | no (on its verdict) |
| Q4b | — | verdict | **FAILED** inverted: d decreases strictly along the chain | **FAILED** inverted: d decreases strictly along the chain | no (on its verdict) |
| Q4b.three_rung | — | verdict | **FAILED** inverted: d decreases strictly along the chain | **FAILED** inverted: d decreases strictly along the chain | no (on its verdict) |
| Q5a | maxpressure | verdict | **HELD** +519.0220 [+512.6790, +525.3650] | **HELD** +514.4262 [+508.0807, +520.7716] | no (on its verdict) |
| Q5a | fixedtime | verdict | **NOT RESOLVED** +0.2839 [-0.1103, +0.6781] | **NOT RESOLVED** +0.2816 [-0.1093, +0.6725] | no (on its verdict) |
| Q5a | random | verdict | **HELD** +61.4959 [+55.4619, +67.5298] | **HELD** +60.9885 [+55.0077, +66.9693] | no (on its verdict) |
| Q5a.aggregate | — | verdict | **NOT RESOLVED** fixedtime NOT RESOLVED, maxpressure HELD, random HELD | **NOT RESOLVED** fixedtime NOT RESOLVED, maxpressure HELD, random HELD | no (on its verdict) |
| Q5b | mappo1000 | verdict | **HELD** -561.7580 [-569.7300, -553.7860] | **HELD** -557.3594 [-565.4166, -549.3021] | no (on its verdict) |
| Q5b | maxpressure | verdict | **HELD** -474.1761 [-482.3387, -466.0136] | **HELD** -469.6936 [-477.8062, -461.5809] | no (on its verdict) |
| Q5b | fixedtime | verdict | **NOT RESOLVED** -0.2839 [-0.6781, +0.1103] | **NOT RESOLVED** -0.2816 [-0.6725, +0.1093] | no (on its verdict) |
| Q5b | random | verdict | **HELD** -49.5097 [-55.6665, -43.3529] | **HELD** -49.1561 [-55.2578, -43.0544] | no (on its verdict) |
| Q5b.aggregate | — | verdict | **NOT RESOLVED** fixedtime NOT RESOLVED, mappo1000 HELD, maxpressure HELD, random HELD | **NOT RESOLVED** fixedtime NOT RESOLVED, mappo1000 HELD, maxpressure HELD, random HELD | no (on its verdict) |
| Q5c | mappo1000 | verdict | **FAILED** +23.3168 [+19.1606, +27.4730] | **FAILED** +23.2396 [+19.1155, +27.3637] | no (on its verdict) |
| Q5c | maxpressure | verdict | **FAILED** +44.8459 [+40.2609, +49.4309] | **FAILED** +44.7326 [+40.1822, +49.2831] | no (on its verdict) |
| Q5c | fixedtime | verdict | **NOT RESOLVED** exactly zero (identical episodes) | **NOT RESOLVED** exactly zero (identical episodes) | no (on its verdict) |
| Q5c | random | verdict | **HELD** +11.9862 [+6.8833, +17.0890]; reverses on 2 of 5 seeds (303, 505) | **HELD** +11.8324 [+6.7711, +16.8937]; reverses on 2 of 5 seeds (303, 505) | no (on its verdict) |
| Q5c.aggregate | — | verdict | **FAILED** fixedtime NOT RESOLVED, mappo1000 FAILED, maxpressure FAILED, random HELD | **FAILED** fixedtime NOT RESOLVED, mappo1000 FAILED, maxpressure FAILED, random HELD | no (on its verdict) |
| Q6 `dt_spatial` | mappo1000 | verdict | **NOT COLLAPSED** -62.5435 [-66.0875, -58.9996] | **NOT COLLAPSED** -62.9476 [-66.4636, -59.4315] | no (on its verdict) |
| Q6 `dt_nomix` | mappo1000 | verdict | **NOT COLLAPSED** -102.5645 [-102.9008, -102.2282] | **NOT COLLAPSED** -102.5124 [-102.8487, -102.1762] | no (on its verdict) |
| Q6 `dt_spatial` | maxpressure | verdict | **NOT COLLAPSED** -92.4314 [-92.7403, -92.1225] | **NOT COLLAPSED** -92.8011 [-93.1109, -92.4914] | no (on its verdict) |
| Q6 `dt_nomix` | maxpressure | verdict | **NOT COLLAPSED** -93.1209 [-93.4363, -92.8054] | **NOT COLLAPSED** -93.5278 [-93.8441, -93.2115] | no (on its verdict) |
| Q6 `dt_spatial` | fixedtime | verdict | **NOT COLLAPSED** -53.5177 [-53.7881, -53.2473] | **NOT COLLAPSED** -53.2410 [-53.5122, -52.9697] | no (on its verdict) |
| Q6 `dt_nomix` | fixedtime | verdict | **NOT COLLAPSED** -53.5177 [-53.7881, -53.2473] | **NOT COLLAPSED** -53.2410 [-53.5122, -52.9697] | no (on its verdict) |
| Q6 `dt_spatial` | random | verdict | **NOT COLLAPSED** -5.7551 [-6.2516, -5.2586] | **NOT COLLAPSED** -5.7423 [-6.2384, -5.2462] | no (on its verdict) |
| Q6 `dt_nomix` | random | verdict | **NOT COLLAPSED** -5.0453 [-5.5102, -4.5804] | **NOT COLLAPSED** -5.0360 [-5.5005, -4.5716] | no (on its verdict) |

77 rows, 154 statement entries (each row is one entry per definition). Numbers are the artifact's, at four decimals; ``exactly zero`` means every per-draw difference is 0. A cell ending in "reverses on k of 5 seeds (...)" carries that entry's non-null `seed_reversal_qualifier` (`BRIEF_45` Amendment D, D4.2).

<!-- FIRST-TIME SCORES -->
| statement | tier | arm | definition | verdict | mean [95 % CI] | reverses on | reading |
|---|---|---|---|---|---|---|---|
| Q0a | mappo1000 | — | `att_ours` | HELD | +27.1004 [+23.8150, +30.3858] | — | resolves: the CI lies entirely above zero |
| Q0a | mappo1000 | — | `att_engine` **(primary)** | HELD | +27.4534 [+24.1365, +30.7702] | — | resolves: the CI lies entirely above zero |
| Q0b | mappo1000 | — | `att_ours` | FAILED | -12.4645 [-16.3575, -8.5714] | 1 of 5 seeds (101) | the CI of mean(I) excludes zero; its sign is negative |
| Q0b | mappo1000 | — | `att_engine` **(primary)** | FAILED | -12.5676 [-16.4943, -8.6409] | 1 of 5 seeds (101) | the CI of mean(I) excludes zero; its sign is negative |
| Q5a | maxpressure | — | `att_ours` | HELD | +514.4262 [+508.0807, +520.7716] | — | resolves: the CI lies entirely above zero |
| Q5a | maxpressure | — | `att_engine` **(primary)** | HELD | +519.0220 [+512.6790, +525.3650] | — | resolves: the CI lies entirely above zero |
| Q5a | fixedtime | — | `att_ours` | NOT RESOLVED | +0.2816 [-0.1093, +0.6725] | — | does not resolve: the CI includes zero |
| Q5a | fixedtime | — | `att_engine` **(primary)** | NOT RESOLVED | +0.2839 [-0.1103, +0.6781] | — | does not resolve: the CI includes zero |
| Q5a | random | — | `att_ours` | HELD | +60.9885 [+55.0077, +66.9693] | — | resolves: the CI lies entirely above zero |
| Q5a | random | — | `att_engine` **(primary)** | HELD | +61.4959 [+55.4619, +67.5298] | — | resolves: the CI lies entirely above zero |
| Q5a.aggregate | — | — | `att_ours` | NOT RESOLVED | — (an aggregate of the tiers above) | — | — |
| Q5a.aggregate | — | — | `att_engine` **(primary)** | NOT RESOLVED | — (an aggregate of the tiers above) | — | — |
| Q5c | mappo1000 | — | `att_ours` | FAILED | +23.2396 [+19.1155, +27.3637] | — | resolves: the CI lies entirely above zero |
| Q5c | mappo1000 | — | `att_engine` **(primary)** | FAILED | +23.3168 [+19.1606, +27.4730] | — | resolves: the CI lies entirely above zero |
| Q5c | maxpressure | — | `att_ours` | FAILED | +44.7326 [+40.1822, +49.2831] | — | resolves: the CI lies entirely above zero |
| Q5c | maxpressure | — | `att_engine` **(primary)** | FAILED | +44.8459 [+40.2609, +49.4309] | — | resolves: the CI lies entirely above zero |
| Q5c | fixedtime | — | `att_ours` | NOT RESOLVED | exactly zero | — | exactly zero (identical episodes) |
| Q5c | fixedtime | — | `att_engine` **(primary)** | NOT RESOLVED | exactly zero | — | exactly zero (identical episodes) |
| Q5c | random | — | `att_ours` | HELD | +11.8324 [+6.7711, +16.8937] | 2 of 5 seeds (303, 505) | resolves: the CI lies entirely above zero |
| Q5c | random | — | `att_engine` **(primary)** | HELD | +11.9862 [+6.8833, +17.0890] | 2 of 5 seeds (303, 505) | resolves: the CI lies entirely above zero |
| Q5c.aggregate | — | — | `att_ours` | FAILED | — (an aggregate of the tiers above) | — | — |
| Q5c.aggregate | — | — | `att_engine` **(primary)** | FAILED | — (an aggregate of the tiers above) | — | — |
| Q3c | maxpressure | — | `att_ours` | — (no verdict) | +1.2993 [+0.2395, +2.3592] | — | resolves against the DT |
| Q3c | maxpressure | — | `att_engine` **(primary)** | — (no verdict) | +0.9059 [-0.1617, +1.9735] | 1 of 5 seeds (303) | NOT RESOLVED |
| Q3c | fixedtime | — | `att_ours` | — (no verdict) | +3.1669 [+2.9256, +3.4082] | — | resolves against the DT |
| Q3c | fixedtime | — | `att_engine` **(primary)** | — (no verdict) | +3.1835 [+2.9418, +3.4253] | — | resolves against the DT |
| Q6 | mappo1000 | dt_spatial | `att_ours` | NOT COLLAPSED | -62.9476 [-66.4636, -59.4315] | — | resolves: the CI lies entirely below zero |
| Q6 | mappo1000 | dt_spatial | `att_engine` **(primary)** | NOT COLLAPSED | -62.5435 [-66.0875, -58.9996] | — | resolves: the CI lies entirely below zero |
| Q6 | mappo1000 | dt_nomix | `att_ours` | NOT COLLAPSED | -102.5124 [-102.8487, -102.1762] | — | resolves: the CI lies entirely below zero |
| Q6 | mappo1000 | dt_nomix | `att_engine` **(primary)** | NOT COLLAPSED | -102.5645 [-102.9008, -102.2282] | — | resolves: the CI lies entirely below zero |
| Q6 | maxpressure | dt_spatial | `att_ours` | NOT COLLAPSED | -92.8011 [-93.1109, -92.4914] | — | resolves: the CI lies entirely below zero |
| Q6 | maxpressure | dt_spatial | `att_engine` **(primary)** | NOT COLLAPSED | -92.4314 [-92.7403, -92.1225] | — | resolves: the CI lies entirely below zero |
| Q6 | maxpressure | dt_nomix | `att_ours` | NOT COLLAPSED | -93.5278 [-93.8441, -93.2115] | — | resolves: the CI lies entirely below zero |
| Q6 | maxpressure | dt_nomix | `att_engine` **(primary)** | NOT COLLAPSED | -93.1209 [-93.4363, -92.8054] | — | resolves: the CI lies entirely below zero |
| Q6 | fixedtime | dt_spatial | `att_ours` | NOT COLLAPSED | -53.2410 [-53.5122, -52.9697] | — | resolves: the CI lies entirely below zero |
| Q6 | fixedtime | dt_spatial | `att_engine` **(primary)** | NOT COLLAPSED | -53.5177 [-53.7881, -53.2473] | — | resolves: the CI lies entirely below zero |
| Q6 | fixedtime | dt_nomix | `att_ours` | NOT COLLAPSED | -53.2410 [-53.5122, -52.9697] | — | resolves: the CI lies entirely below zero |
| Q6 | fixedtime | dt_nomix | `att_engine` **(primary)** | NOT COLLAPSED | -53.5177 [-53.7881, -53.2473] | — | resolves: the CI lies entirely below zero |
| Q6 | random | dt_spatial | `att_ours` | NOT COLLAPSED | -5.7423 [-6.2384, -5.2462] | — | resolves: the CI lies entirely below zero |
| Q6 | random | dt_spatial | `att_engine` **(primary)** | NOT COLLAPSED | -5.7551 [-6.2516, -5.2586] | — | resolves: the CI lies entirely below zero |
| Q6 | random | dt_nomix | `att_ours` | NOT COLLAPSED | -5.0360 [-5.5005, -4.5716] | — | resolves: the CI lies entirely below zero |
| Q6 | random | dt_nomix | `att_engine` **(primary)** | NOT COLLAPSED | -5.0453 [-5.5102, -4.5804] | — | resolves: the CI lies entirely below zero |

<!-- RECORD -->
`literal_records`: 123 rows, 115 equal, 8 unequal, and every unequal row carries a named exception. Each row also carries the same quantity under the primary definition (`att_engine`, corrected version, same extractor): 50 equal the record, 62 do not, and 11 are protocol facts with no `att_engine` counterpart (null, with a note).

The unequal rows:

| source | value | recorded | this artifact (`att_ours`) | exception | `att_engine` (primary) |
|---|---|---|---|---|---|
| `docs/returns/P5.2.md:27` | `q2b.counts` | 11 · 8 · 8 | 11 · 7 · 8 | `deferred_108` | 11 · 7 · 8 |
| `docs/returns/P5.2.md:28` | `q3a.ranks` | 2/6 · 3/6 · 3/6 | 2/6 · 2/6 · 3/6 | `q3a_fixedtime` | 2/6 · 2/6 · 3/6 |
| `docs/returns/P5.2.md:100` | `q2b.overall` | 27/45 | 26/45 | `deferred_108` | 26/45 |
| `docs/returns/P5.2.md:570` | `q3a.ranks` | 2/6 · 3/6 · 3/6 | 2/6 · 2/6 · 3/6 | `q3a_fixedtime` | 2/6 · 2/6 · 3/6 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1227` | `q2b.counts` | 11 · 8 · 8 | 11 · 7 · 8 | `deferred_108` | 11 · 7 · 8 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1231` | `q2b.overall` | 27/45 | 26/45 | `deferred_108` | 26/45 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1240` | `q2b.counts` | 11 · 8 · 8 | 11 · 7 · 8 | `deferred_108` | 11 · 7 · 8 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1249` | `q3a.ranks` | 2/6 · 3/6 · 3/6 | 2/6 · 2/6 · 3/6 | `q3a_fixedtime` | 2/6 · 2/6 · 3/6 |

The two exceptions, in the artifact's words:

- **`deferred_108`**: DEFERRED 108: at fixedtime dt_spatial = dt_nomix and bc = bc_top10_perix exactly, and under the tie rule P5.2 declared on 2026-08-24 (a tie is discordant for both orders) Q2b's count there is 7, the lower end of its range 7..9; the record's 8 is one tie-break order's count, not the declared rule's
- **`q3a_fixedtime`**: BRIEF_45 section 1.4(b): at fixedtime dt_nomix ties dt_spatial exactly, so its rank is 2 counting only the arms strictly lower (the declared rule, P5.2b's) and 3 if the tie is broken against it; the record's 3 is one tie-break order's rank, the range is 2..3

The 54 rows the record states correctly under `att_ours` that read otherwise under the primary definition:

| source | value | recorded (= `att_ours`) | `att_engine` (primary) |
|---|---|---|---|
| `docs/returns/P5.2.md:25` | `q1b.relative_errors` | 0.04 % / 0.09 % | 4.01 % / 3.49 % |
| `docs/returns/P5.2.md:34` | `q1.measured.bc_top10@fixedtime` | 207.47 | 200.06 |
| `docs/returns/P5.2.md:34` | `q1.measured.iql@fixedtime` | 203.95 | 196.52 |
| `docs/returns/P5.2.md:35` | `q1.measured.bc_top10@random` | 350.34 | 343.97 |
| `docs/returns/P5.2.md:35` | `q1.measured.iql@random` | 190.96 | 191.39 |
| `docs/returns/P5.2.md:42` | `q4.mappo1000.mean` | +39.5649 | +40.0210 |
| `docs/returns/P5.2.md:42` | `q4.mappo1000.ci` | [+36.0510, +43.0787] | [+36.4788, +43.5631] |
| `docs/returns/P5.2.md:43` | `q4.maxpressure.mean` | +0.7267 | +0.6895 |
| `docs/returns/P5.2.md:43` | `q4.maxpressure.ci` | [+0.6206, +0.8328] | [+0.5855, +0.7934] |
| `docs/returns/P5.2.md:45` | `q4.random.mean` | −0.7063 | -0.7098 |
| `docs/returns/P5.2.md:45` | `q4.random.ci` | [−1.3172, −0.0953] | [-1.3207, -0.0990] |
| `docs/returns/P5.2.md:48-49` | `q4.endpoints` | 39.56 at the best tier against −0.71 at the worst | 40.02 at the best tier against -0.71 at the worst |
| `docs/returns/P5.2.md:74` | `q3b.maxpressure.mean` | −0.7294 | -0.6664 |
| `docs/returns/P5.2.md:74` | `q3b.maxpressure.ci_prefixed` | CI [−0.9139, −0.5450] | CI [-0.8475, -0.4853] |
| `docs/returns/P5.2.md:74` | `q3b.random.mean` | −5.0360 | -5.0453 |
| `docs/returns/P5.2.md:74` | `q3b.random.ci_prefixed` | CI [−5.5005, −4.5716] | CI [-5.5102, -4.5804] |
| `docs/returns/P5.2.md:265` | `level.bc_top10@maxpressure` | 679.96 on `maxpressure` | 678.22 on `maxpressure` |
| `docs/returns/P5.2.md:266` | `level.bc_top10_perix@maxpressure.bare` | 210.27 | 204.04 |
| `docs/returns/P5.2.md:567` | `q4.endpoints_packet` | `+39.5649` at the best tier, `−0.7063` at the | `+40.0210` at the best tier, `-0.7098` at the |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1216` | `q1b.relative_errors` | 0.04 % / 0.09 % | 4.01 % / 3.49 % |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1219` | `q1.measured.bc_top10@fixedtime` | 207.47 | 200.06 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1219` | `q1.relative_error.bc_top10@fixedtime` | 80.1 % | 80.8 % |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1220` | `q1.measured.iql@fixedtime` | 203.95 | 196.52 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1220` | `q1.relative_error.iql@fixedtime.parenthesised` | (43.5 %) | (45.5 %) |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1220` | `q1.measured.bc_top10@random` | 350.34 | 343.97 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1221` | `q1.relative_error.bc_top10@random.parenthesised` | (71.8 %) | (72.3 %) |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1221` | `q1.measured.iql@random` | 190.96 | 191.39 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1221` | `q1.relative_error.iql@random.parenthesised` | (54.0 %) | (53.9 %) |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1249-1250` | `q3c.all_against` | every paired difference against the best non-DT arm resolves against the DT | not every paired difference against the best non-DT arm resolves against the DT |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1252` | `q3b.maxpressure.gate2` | `−0.7294`, CI [−0.9139, −0.5450] | `-0.6664`, CI [-0.8475, -0.4853] |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1260` | `q4.mappo1000.mean` | +39.5649 | +40.0210 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1260` | `q4.mappo1000.ci` | [+36.0510, +43.0787] | [+36.4788, +43.5631] |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1261` | `q4.maxpressure.mean` | +0.7267 | +0.6895 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1261` | `q4.maxpressure.ci` | [+0.6206, +0.8328] | [+0.5855, +0.7934] |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1263` | `q4.random.mean` | −0.7063 | -0.7098 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1263` | `q4.random.ci` | [−1.3172, −0.0953] | [-1.3207, -0.0990] |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1267` | `q4.endpoints` | 39.56 at the best tier against −0.71 at the worst | 40.02 at the best tier against -0.71 at the worst |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1276` | `q5b.maxpressure.mean` | −469.69 | -474.18 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1276` | `q5b.maxpressure.ci` | [−477.81, −461.58] | [-482.34, -466.01] |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1276` | `q5b.random.mean` | −49.16 | -49.51 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1277` | `q5b.random.ci` | [−55.26, −43.05] | [-55.67, -43.35] |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1277` | `q5b.fixedtime.ci_unresolved` | [−0.67, +0.11] unresolved | [-0.68, +0.11] unresolved |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1278` | `level.bc_top10@maxpressure` | 679.96 on `maxpressure` | 678.22 on `maxpressure` |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1279` | `level.bc_top10_perix@maxpressure` | (210.27) | (204.04) |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1621` | `q0.d1.mean` | +39.5649 | +40.0210 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1621` | `q0.d1.ci` | [+36.0510, +43.0787] | [+36.4788, +43.5631] |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1622` | `q0.d4.mean` | +27.1004 | +27.4534 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1622` | `q0.d4.ci` | [+23.8150, +30.3858] | [+24.1365, +30.7702] |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1623` | `q0.i.mean` | −12.4645 | -12.5676 |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1623` | `q0.i.ci` | [−16.3575, −8.5714] | [-16.4943, -8.6409] |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1628` | `q0.i.reduction` | The 31.5 % reduction | The 31.4 % reduction |
| `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1668` | `q0.i.attributable` | 31.5 % of it is attributable | 31.4 % of it is attributable |
| `docs/plans/p5.2.md:513` | `q0.d1.plan` | measured: +39.5649, CI [+36.0510, +43.0787] | measured: +40.0210, CI [+36.4788, +43.5631] |
| `docs/plans/p5.2.md:606-607` | `fixedtime.tied_levels` | `dt_spatial` = `dt_nomix` = 207.11922169043655 and `bc` = `bc_top10_perix` = 207.1886372764493 | `dt_spatial` = `dt_nomix` = 199.70482048081334 and `bc` = `bc_top10_perix` = 199.77605696984688 |

The 11 protocol-fact rows (`att_engine` null): `q1.predicted.bc_top10@fixedtime`: a registered prediction of the att_horizon era, not a measurement: no prediction was registered under att_engine, so it has no att_engine counterpart; `q1.predicted.bc_top10@random`: a registered prediction of the att_horizon era, not a measurement: no prediction was registered under att_engine, so it has no att_engine counterpart; `q1.predicted.iql@fixedtime`: a registered prediction of the att_horizon era, not a measurement: no prediction was registered under att_engine, so it has no att_engine counterpart; `q1.predicted.iql@random`: a registered prediction of the att_horizon era, not a measurement: no prediction was registered under att_engine, so it has no att_engine counterpart; `q1.threshold`: the threshold k = ceil(9/13 * N) is fixed by the protocol (docs/plans/p5.2.md:561-563), not measured, so it has no att_engine counterpart; `q2a.predicted`: the registered predicted first places of the att_horizon era, not a measurement: no prediction was registered under att_engine, so they have no att_engine counterpart.

P8.4b's V4 and V5 (`comparison_with_the_record.p8_4b`):

| contrast | definition | statement | what the outcome belongs to | equal | difference (mean, low, high) | in ulps of the record | the record's route reproduces it |
|---|---|---|---|---|---|---|---|
| V4 | `att_ours` | Q0.d1 | Q0.d1 is d1, which is P5.1's P2a; the outcome is P2a's (spatial_mixing._outcome_from_interval) | False | -7.105427357601002e-15, 0.0, -1.4210854715202004e-14 | -1.0, 0.0, -2.0 | True |
| V4 | `att_engine` | Q0.d1 | Q0.d1 is d1, which is P5.1's P2a; the outcome is P2a's (spatial_mixing._outcome_from_interval) | True | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0 | True |
| V5 | `att_ours` | Q0.stop_rule | the stop rule's d4, the d4 that Q0a also reads; the outcome is the stop rule's verdict (tier_sweep.stop_rule_verdict) | False | -3.552713678800501e-15, -3.552713678800501e-15, -3.552713678800501e-15 | -1.0, -1.0, -1.0 | True |
| V5 | `att_engine` | Q0.stop_rule | the stop rule's d4, the d4 that Q0a also reads; the outcome is the stop rule's verdict (tier_sweep.stop_rule_verdict) | True | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0 | True |

Q5 (`comparison_with_the_record.q5`, `BRIEF_45` Amendment A, A3), the three records it answers:

- `docs/returns/P5.2.md:30`: *"CONFIRMED twice, with a predicted null on `fixedtime`"*
- `docs/returns/P5.2.md:572-573`: *"The load sorter is confirmed across tiers, and `bc_top10_perix` is the repair — with a *predicted* null on `fixedtime` that the mechanism explains."*
- `docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1274-1281`: *"THE LOAD-SORTER MECHANISM IS CONFIRMED, decisively and twice"*

| | Q5a | Q5b | Q5c |
|---|---|---|---|
| registered verdict, `att_engine` (primary) | NOT RESOLVED | NOT RESOLVED | FAILED |
| registered verdict, `att_ours` | NOT RESOLVED | NOT RESOLVED | FAILED |
| `fixedtime`: registered value | +820.65 | -812.15 | +8.49 |
| `fixedtime`: measured, `att_engine` | +0.2839 [-0.1103, +0.6781] NOT RESOLVED | -0.2839 [-0.6781, +0.1103] NOT RESOLVED | exactly zero |
| `fixedtime`: measured, `att_ours` | +0.2816 [-0.1093, +0.6725] NOT RESOLVED | -0.2816 [-0.6725, +0.1093] NOT RESOLVED | exactly zero |

The artifact's statement: *"the record scored Q5b on two of its four registered tiers, scored neither Q5a nor Q5c, and called predicted a null the registration did not predict."* In detail: the record reported Q5b's values on three tiers (maxpressure, fixedtime, random), counted the two that resolved as confirmations, gave no value at mappo1000, where Q5b is also registered, and read fixedtime's unresolved interval as a predicted null; the registration predicted +820.65 (Q5a), -812.15 (Q5b) and +8.49 (Q5c) there. BRIEF_27's GATE 2 is a coordinator ruling, so the error is the coordinator's; it is logged in PROJECT_PLAN section 8 (dated 2026-10-10).

Q5's descriptive reading (not a registered verdict; its sets computed from the collapse audit's rows):

| definition | the global filter collapses alone on | both filters collapse on | Q5b resolves on | reading |
|---|---|---|---|---|
| `att_engine` (primary) | mappo1000, maxpressure | random | mappo1000, maxpressure, random | the global return filter (bc_top10) collapses where the per-intersection one (bc_top10_perix) does not on mappo1000 and maxpressure; on random both filters collapse: each is worse than the random anchor with the paired CI excluding 0 |
| `att_ours` | mappo1000, maxpressure | random | mappo1000, maxpressure, random | the global return filter (bc_top10) collapses where the per-intersection one (bc_top10_perix) does not on mappo1000 and maxpressure; on random both filters collapse: each is worse than the random anchor with the paired CI excluding 0 |

P5.2's stop-rule record: equal False; difference -3.552713678800501e-15 (mean), -3.552713678800501e-15, -3.552713678800501e-15; the record's route reproduces it: True.
P5.1's grid (P1 and P3): equal True. P5.2b's artifact: equal True, 732 fields compared over 4 statement sets.

<!-- TIES -->
| tier | definition | version | equal-level groups | identical-episode pairs (fields equal) | pairs equal on some episodes only |
|---|---|---|---|---|---|
| mappo1000 | `att_ours` | both | none | none | dt_spatial / dt_nomix: 1 of 500; dt_nomix / bc_top10_perix: 1 of 500; dt_nomix / behaviour: 1 of 500; bc_top10_perix / behaviour: 1 of 500 |
| mappo1000 | `att_engine` | both | none | none | none |
| maxpressure | `att_ours` | both | none | none | dt_nomix / behaviour: 2 of 500 |
| maxpressure | `att_engine` | both | none | none | none |
| fixedtime | `att_ours` | both | dt_spatial = dt_nomix = behaviour; bc = bc_top10_perix | dt_spatial ≡ dt_nomix (8 fields, 500 episodes); dt_spatial ≡ behaviour (8 fields, 500 episodes); dt_nomix ≡ behaviour (8 fields, 500 episodes); bc ≡ bc_top10_perix (8 fields, 500 episodes) | bc / bc_top10: 495 of 500; bc / iql: 1 of 500; bc_top10 / bc_top10_perix: 495 of 500; bc_top10 / iql: 1 of 500; bc_top10_perix / iql: 1 of 500 |
| fixedtime | `att_engine` | both | dt_spatial = dt_nomix = behaviour; bc = bc_top10_perix | dt_spatial ≡ dt_nomix (8 fields, 500 episodes); dt_spatial ≡ behaviour (8 fields, 500 episodes); dt_nomix ≡ behaviour (8 fields, 500 episodes); bc ≡ bc_top10_perix (8 fields, 500 episodes) | bc / bc_top10: 495 of 500; bc_top10 / bc_top10_perix: 495 of 500 |
| random | `att_ours` | corrected | none | none | dt_nomix / behaviour: 1 of 500 |
| random | `att_ours` | as_committed | none | none | dt_nomix / behaviour: 1 of 500 |
| random | `att_engine` | corrected | none | none | none |
| random | `att_engine` | as_committed | none | none | none |

Tie-dependent integers whose range is not a point (identical under both definitions, asserted):

`fixedtime` (both):

| integer | scope | under the rule | range | tie-break orders |
|---|---|---|---|---|
| Q2b | pooled | 7 | 7..9 | 4 |
| Q2b_hard | pooled | 2 | 2..4 | 4 |
| Q3a | pooled | 2 | 2..3 | 4 |
| Q2b | seed 101 | 7 | 7..9 | 4 |
| Q2b_hard | seed 101 | 2 | 2..4 | 4 |
| Q3a | seed 101 | 2 | 2..3 | 4 |
| Q2b | seed 202 | 5 | 5..9 | 12 |
| Q2b_hard | seed 202 | 2 | 2..4 | 12 |
| Q3a | seed 202 | 2 | 2..3 | 12 |
| Q2b | seed 303 | 7 | 7..9 | 4 |
| Q2b_hard | seed 303 | 2 | 2..4 | 4 |
| Q3a | seed 303 | 2 | 2..3 | 4 |
| Q2b | seed 404 | 5 | 5..9 | 12 |
| Q2b_hard | seed 404 | 2 | 2..4 | 12 |
| Q3a | seed 404 | 2 | 2..3 | 12 |
| Q2b | seed 505 | 5 | 5..9 | 12 |
| Q2b_hard | seed 505 | 2 | 2..4 | 12 |
| Q3a | seed 505 | 2 | 2..3 | 12 |

Every other tier: none — every integer there is a point. Rule: exactly equal levels are not an ordering: a tie is discordant for both orders (docs/plans/p5.2.md:606-609); Q3a's rank is 1 + the arms strictly lower; every integer carries its range over all tie-break orders (tie_audit)

<!-- CHANGES -->
`changes`: 69 entries; 0 changed by the IQL correction, 2 changed by Rule R.

| statement | tier | as committed, `att_ours` | corrected, `att_ours` | corrected, `att_engine` (primary) | changed by |
|---|---|---|---|---|---|
| D9 | maxpressure | seed 101: bc first; seed 202: bc first; seed 303: bc first; seed 404: bc first; seed 505: bc first | seed 101: bc first; seed 202: bc first; seed 303: bc first; seed 404: bc first; seed 505: bc first | seed 101: bc first; seed 202: bc first; seed 303: dt_nomix first; seed 404: bc first; seed 505: bc first | Rule R |
| Q3c | maxpressure | resolves against the DT +1.2993 [+0.2395, +2.3592] | resolves against the DT +1.2993 [+0.2395, +2.3592] | NOT RESOLVED +0.9059 [-0.1617, +1.9735] | Rule R |

<!-- CORRECTION -->
16 (statement, tier, definition) triples exist in both versions -- 32 entries, one per version; every pair has the same verdict and reading in both (asserted). The values that move:

| definition | IQL level at `random` | Q1's `iql@random` cell (held; relative error) | Q3c at `random`, `dt_nomix − iql` |
|---|---|---|---|
| `att_engine` **(primary)** | 183.8281 → 191.3938 | 183.8281 (not held; 55.70 %) → 191.3938 (not held; 53.88 %) | +64.3490 [+62.0765, +66.6216] → +56.7834 [+53.8995, +59.6672] |
| `att_ours` | 190.9638 → 198.4810 | 190.9638 (not held; 53.98 %) → 198.4810 (not held; 52.17 %) | +64.3603 [+62.1046, +66.6160] → +56.8432 [+53.9794, +59.7069] |

Each cell reads *as committed → corrected*. The IQL level also enters `Q3a` and `D9` at `random` (their `levels.iql`); Q2a, Q2b and Q2b's aggregate change in no field.

<!-- LEADERS -->
| tier | definition | version | leader | runner-up | leader − runner-up [95 % CI] | reverses on | reading |
|---|---|---|---|---|---|---|---|
| mappo1000 | `att_ours` | both | dt_nomix | bc | -11.1329 [-13.7932, -8.4726] | — | dt_nomix leads bc; the lead resolves (the CI lies entirely below zero) |
| mappo1000 | `att_engine` **(primary)** | both | dt_nomix | bc | -11.4093 [-14.0917, -8.7269] | — | dt_nomix leads bc; the lead resolves (the CI lies entirely below zero) |
| maxpressure | `att_ours` | both | bc | dt_nomix | -1.2993 [-2.3592, -0.2395] | — | bc leads dt_nomix; the lead resolves (the CI lies entirely below zero) |
| maxpressure | `att_engine` **(primary)** | both | bc | dt_nomix | -0.9059 [-1.9735, +0.1617] | 1 of 5 seeds (303) | bc leads dt_nomix; the lead does not resolve (the CI includes zero) |
| fixedtime | `att_ours` | both | iql | dt_spatial | -3.1669 [-3.4082, -2.9256] | — | iql leads dt_spatial = dt_nomix; the lead resolves (the CI lies entirely below zero) |
| fixedtime | `att_ours` | both | iql | dt_nomix | -3.1669 [-3.4082, -2.9256] | — | iql leads dt_spatial = dt_nomix; the lead resolves (the CI lies entirely below zero) |
| fixedtime | `att_engine` **(primary)** | both | iql | dt_spatial | -3.1835 [-3.4253, -2.9418] | — | iql leads dt_spatial = dt_nomix; the lead resolves (the CI lies entirely below zero) |
| fixedtime | `att_engine` **(primary)** | both | iql | dt_nomix | -3.1835 [-3.4253, -2.9418] | — | iql leads dt_spatial = dt_nomix; the lead resolves (the CI lies entirely below zero) |
| random | `att_ours` | corrected | iql | dt_spatial | -56.1369 [-58.9835, -53.2904] | — | iql leads dt_spatial; the lead resolves (the CI lies entirely below zero) |
| random | `att_ours` | as_committed | iql | dt_spatial | -63.6541 [-65.8855, -61.4226] | — | iql leads dt_spatial; the lead resolves (the CI lies entirely below zero) |
| random | `att_engine` **(primary)** | corrected | iql | dt_spatial | -56.0736 [-58.9407, -53.2064] | — | iql leads dt_spatial; the lead resolves (the CI lies entirely below zero) |
| random | `att_engine` **(primary)** | as_committed | iql | dt_spatial | -63.6392 [-65.8883, -61.3901] | — | iql leads dt_spatial; the lead resolves (the CI lies entirely below zero) |

Every entry of `leader_minus_runner_up` is a row: a tier with runners-up tied in level shows one row per runner-up (`BRIEF_45` Amendment D, D3.5). "Reverses on" is the entry's `seed_reversal_qualifier` (D4.2).

<!-- DOES NOT SAY -->
1. No new prediction: every registered value of Q1-Q5 -- Q1's and Q1b's levels, Q2's orderings and first places, Q3b's and Q3c's values, Q4's chain and Q5's values -- is an att_horizon (att_ours) value, registered before any att_engine number existed; under att_engine each is compared with att_engine measurements, and every entry carrying a registered value, Q1's and Q1b's predicted levels or Q2a's predicted first place says so in predictions_registered_on. Rule R' is not recomputed from att_engine inputs (BRIEF_45 section 2; Amendment D, D1.7).
2. No episode was rolled, no checkpoint loaded, no model trained or evaluated: every number is a recomputation of committed records read at their digests.
3. P5.2's statements stand as written in its packet; this artifact scores them by the registered rules beside them and names every difference in comparison_with_the_record.
4. Out of scope: P4.x, P5.1 and P5.3 statements (P8.4b's V1-V7 hold their registered verdicts), the E1 and I1/J1 envelopes (replicates carry no att_engine; att_rederivation.is_replicate_artifact), hz1x1 and cologne3.
5. No level prediction exists for the two 4-head cells and no equivalence verdict is made anywhere (D8).
6. D15's clause on arms whose distribution is visibly not unimodal is a visual judgement this artifact does not make: it reports each cell's median and IQR and no distributional comparison.
7. The rank-biserial and the Wilcoxon p-value are reported beside every two-arm contrast (D8) and are never read by a verdict; I, a difference of two differences, carries none.
8. Q3c registers a value per tier and no outcome rule: it is reported with its reading and no verdict.
9. The leaders block and the descriptive collapse audit are not registered statements.
10. P8.4b's att_engine values are pinned by its campaign's completeness, the per-row checks, the README note's means on every tier and provenance.p8_4b_rows.sha256 -- not by a manifest P8.4b wrote, which does not exist.
11. Rule R makes att_engine the primary metric on grid4x4; the att_ours verdict is reported beside every one and definition_dependent marks where they differ.
12. definition_dependent compares verdicts (readings where an entry has no verdict): Q0b's sign and Q4b's chain shape are reported per definition beside the flag, and the flag does not compare them (BRIEF_45 Amendment B, B2.13).

