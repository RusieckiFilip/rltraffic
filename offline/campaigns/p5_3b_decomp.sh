#!/usr/bin/env bash
# P5.3b-fix BL-2(b) driver -- A13(b)'s decomposition on all 30 cells, run from the task WORKTREE.
#
#   tmux new -s p53bdecomp
#   bash /home/filip/rltraffic-p53b/offline/campaigns/p5_3b_decomp.sh
#
# ⚠️ SKELETON (G2a). The structure and every ruling it encodes are real; the stage bodies land at
# G2b, after the module they invoke exists. It refuses to run until then.
#
# Fails closed; idempotent and restartable. Skeleton of offline/campaigns/p5_3b.sh with four
# deliberate differences, each ruled or disclosed:
#
# ---------------------------------------------------------------------------
# 1. IT WRITES TO THE **MAIN** TREE, NOT THE WORKTREE (Decisions Log 2026-09-10)
# ---------------------------------------------------------------------------
# p5_3b.sh set WORK=$WORK_TREE/output/p5_3b, and review MJ-9 found the entire evidence base --
# 27 entries, 15 checkpoints, the 39-line manifest -- existing ONLY in a worktree, one
# `git worktree prune` from gone. A new directory in the main tree needs no securing step.
#   WORK=$MAIN/output/p5_3b_decomp
# The fence (nortg_decomposition.assert_decomposition_writable) is what makes that safe: it allows
# exactly p5_3b_decomp and SHA256SUMS_p5_3b_decomp.txt and refuses every other component under
# output/, INCLUDING p5_3b, which this campaign reads and must never write.
#
# ---------------------------------------------------------------------------
# 2. THE SKIP DECISION IS IN PYTHON, NOT IN THE SHELL (BRIEF_33 AMENDMENT A4)
# ---------------------------------------------------------------------------
# p5_3b.sh skips a stage on `[ -f "$WORK/chunk.json" ]` alone. assert_probe_cell_is_ablated's own
# docstring records what that costs: "the driver skips a tier whose probe chunk exists, so a bad
# chunk survived every restart." Here `run` inspects its own chunk and re-runs unless it is
# COMPLETE (all 100 draws) and CLEAN (n_mismatches == 0). One Python start per cell costs ~2 s
# against an hour of rollouts.
#
# ---------------------------------------------------------------------------
# 3. ONE-SHOT RUN AUTHORISATION (2026-09-10 ruling -- a mechanism, not a courtesy)
# ---------------------------------------------------------------------------
# The author writes the token; the script requires it and DELETES it as its first action, so one
# token buys exactly one run. The check PRECEDES the FAILED/COMPLETE wipe, so a refused start
# cannot clear a pre-existing FAILED -- validate-then-mutate applied to the driver itself.
#
#   To authorise one run:  date -Is > /home/filip/rltraffic/output/p5_3b_decomp/AUTHORISED_TO_RUN
#
# ---------------------------------------------------------------------------
# 4. THE SCHEDULE IS MEASURED HERE, NOT INHERITED (BRIEF_33 AMENDMENT A5)
# ---------------------------------------------------------------------------
# ⛔ DO NOT quote a parallel speedup from a previous campaign. BRIEF_33 assumed 5x and printed
# "~35 min"; docs/returns/P5.3b.md:178 had MEASURED 2.55x on 30 episodes -- and even that did not
# hold at scale: P5.3b's mix50 five-tuple took 12.5 min against the 6 min 2.55x implied. Both the
# per-episode rate and the 5-worker speedup are measured at G2 on this exact code path, over a
# stated episode count, and the measured numbers are what schedules this run.
#
# 3,000 episodes = 30 cells x 100 held-out draws, 6 groups of 5 concurrent cells.

set -euo pipefail

echo "REFUSING TO START: this driver is a G2a skeleton; the stage bodies land at G2b." >&2
exit 3
