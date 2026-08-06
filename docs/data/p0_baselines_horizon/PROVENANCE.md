# PROVENANCE — docs/data/p0_baselines_horizon

- **Generated (UTC):** 2026-08-06T10:31:39.898507+00:00
- **Git hash:** `c41db668615240e2512d27ef0c5a59576483e9b0`
- **Source config:** `experiments/configs/p0_baselines.json`
- **Format version:** `p0-baselines-horizon-v1.0`
- **Quantities held:** `att_horizon` (prereg A1 primary metric) and `att_running_mean` (legacy runner.py mean-of-samples). Neither is called "average travel time" as a bare name.
- **Policies present:** mappo, Random, MaxPressure
- **Policies pending:** (none)

## Relationship to `docs/data/p0_baselines/`

This directory **re-derives** (never replaces) the 2026-07-09 running-mean-only data in `docs/data/p0_baselines/`. That data stores only the aggregate and cannot yield the horizon value, so a re-run of `experiments/configs/p0_baselines.json` was required (prereg amendment A1). The re-run's `att_running_mean` reproduces the committed anchors exactly for the torch-free baselines (MaxPressure, Random) — that reproduction is the load-bearing check that the horizon values from the same run are trustworthy. MAPPO training runs in the user's tmux session (CLAUDE.md:203) and may cross the N2 float-reduction boundary; when its cells land they are recorded, never adjusted to match.

Reproduce with:

```
.venv/bin/python -m offline.rederive_anchors --policies random,max_pressure   # in-session
.venv/bin/python -m offline.rederive_anchors                                  # full (tmux)
```
