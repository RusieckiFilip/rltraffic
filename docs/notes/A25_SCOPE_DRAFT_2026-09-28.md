# A25 — DRAFT registration of the paper's scope after P7.3c: H2 is NOT TESTED, P7.4 is a limitation, and no experiment beyond H4 and P8.2 is run

**Status: DRAFT, NOT REGISTERED.** Written 2026-09-28 by the coordinator on the author's ruling of the same day (*"C2's 2×2 (P6.1–P6.3)
and P7.4 are out of this paper: H2 is reported as registered and not tested, with the reason, and P7.4 goes to the limitations. P8.2's
compute and latency table follows H4. After that, no new experiments; the paper is written from what is on main."*). It becomes a row of
`PREREGISTRATION.md` §12 with A26 (H4's design) in one commit and one tag, after one review round and the author's written approval,
BEFORE any H4 number exists. The four cells follow.

## Cell 2 — Change

**A25 — THE PAPER'S SCOPE IS FIXED: H2 (C2, the scenario-shift 2×2) IS NOT TESTED AND IS REPORTED AS SUCH; P7.4 (the state-encoding
control) IS NOT RUN AND BECOMES A NAMED LIMITATION; AFTER H4 (A26) AND THE COMPUTE-AND-LATENCY TABLE, NO NEW EXPERIMENT IS RUN — declared
before any datum of any of them exists.** **(a) H2.** §1's RQ2/H2, §2's confirmatory test (*the interaction contrast in the 2×2 on the
primary perturbation family*), §4's C2 design ({nominal, shift-augmented} × {MADT, domain-randomised MAPPO} under perturbations) and
§10's two H2 outcome rows are NOT withdrawn and NOT edited: they stand as registered. **They are not tested in this paper.** The paper
reports, in the words of this row: *"H2 was registered with a confirmatory 2×2 design and was not tested: the perturbation tools, the
domain-randomised MAPPO arm and the 2×2 campaign (P6.1–P6.3) were not built, by the author's decision of 2026-09-28 to close the
experimental programme after the C3 curve and H4. No number bearing on H2 exists in this repository."* C2 is named as future work, with
§4's design as its specification. Nothing about C2 is inferred from C3's scenario-variant draws: P7.0's demand draws are the nominal
demand distribution every campaign shares, not a perturbation family. **(b) P7.4.** The alternative-state-encoding ablation (§10's
*interface control*, the evidence named for the case *H3 fails*) is not run. Its registered role does not arise: H3's clause 1 held on
both scenarios (P7.3a, P7.3d). The paper names, in the limitations, that the transfer gap has not been separated from an
interface-mismatch component by that control, and that A16's frozen feature set, canonical order and phase map are the only
alignment tested. **(c) WHAT IS STILL RUN, and nothing else:** H4's context-length sweep exactly as §1–§2 register it, under A26; and
P8.2's compute-and-latency table (training time, inference ms per decision, parameter counts — measurements of existing artifacts,
no new evaluation of any hypothesis). **After those, the paper is written from what is on `main`; any further experiment is a new
registration row.** **(d) UNCHANGED:** every other row of this file; the multiplicity rule of §8 applies within the families that
are tested (H1, H3, H4); H2's family is empty and no correction is spent on it.

## Cell 3 — Reason

The author's decision, with its reason stated so the reader can weigh it: a first paper with what is on `main` — C1's ladder, C3's
curve with its zero-shot points, its anchor, its few-shot curve and two controls — plus the one confirmatory test that costs days
rather than weeks (H4) and answers the framing controversy the paper opens with; C2's 2×2 is weeks of new tooling and training (a
perturbation suite, a domain-randomised MAPPO, a 2×2 campaign), and P7.4's control is evidence for a case that did not occur.
**Declared now rather than discovered by a referee:** a hypothesis dropped after its data existed would be selection; dropped before
any datum exists, it is scope, and the registration is what makes the difference checkable. **Stated limits:** the paper's title and
claims must not promise robustness to scenario shift (C2's claim), and the introduction's framing of *three* claims becomes two claims
plus a registered, untested third named as such.

## Cell 4 — Results already seen?

**NONE for H2 or P7.4, and checked on disk before this row was written (2026-09-28):** no perturbation, closure, dropout or
shift-augmentation module under `offline/` (the only randomiser is P7.0's `flow_randomizer.py`, the nominal-demand draw tool); no
`p6_*` or `p7_4*` directory under `output/` in any worktree; no MAPPO checkpoint with a domain-randomised or shift-augmented name; no
`docs/data/` artifact and no branch for P6 or P7.4. **Seen, and named because they motivate the decision:** every C1 and C3 number on
`main` (P4–P5.4, P7.0–P7.3d, P7.3c), none of which bears on H2. **For H4: nothing** — no context-length sweep has been trained or
evaluated (A26's Cell 4 measures this again at its own tag).
