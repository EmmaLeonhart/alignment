# redemption-realignment — Work Queue

**This file is a queue, not a state snapshot.** It lists what is
being worked on right at this moment. Finished work lives in
`git log`; longer-horizon work lives in `planning/todo.md`.

See `CLAUDE.md` §"Workflow Rules" for how `queue.md` and the task
tool stay in sync (pattern borrowed from the Sutra repo).

---

## Active

*Queue is clear; next-session pending items live in
`planning/todo.md`. The big still-open items are summarized below.*

### Top three for the next session

1. **Re-sweep the gate against `data/learned_hhh_direction.pt` on
   all three adapters.** This is the load-bearing post-dissociation
   Thread 3 test. The HHH-derived counter-direction is the
   behaviour-axis candidate; the canonical-direction sweeps showed
   |Δ| ≤ 0.04 on means and bidirectional per-prompt structure. If
   the HHH-direction sweep produces unidirectional aligning shifts,
   the dissociation has been resolved at the gate level.
2. **Behavioural eval at α extension settings.** §5.5 notes that
   the geometric Δ at α=2.0 (medical, max −0.053) needs a paired
   behavioural eval before claiming any realignment. Rerun
   `generate_betley_responses.py` with gate-active forward passes
   at (α=2.0, τ=0.25) and judge as before.
3. **CaML pilot v1 → v2 (per-template targets) → 12000-doc scale-up.**
   Per `data/redemption_corpus_v1_pilot/REVIEW.md`, the residual
   length gap (PND 622 vs generic 413) is fixable by setting
   per-template target_words (PND 350, generic 500) instead of the
   shared 450. Then scale to the full grid.

---

## Recently shipped this session (2026-05-13, ~16 hours of work)

**Headline science:** the Cloud-Betley dissociation — the four
measurement axes (geometric, externally-judged aligned, externally-
judged coherent, Cloud self-rated harmfulness) are largely
orthogonal across our 12-condition battery. r(geom, aligned) = −0.03,
r(geom, harm) = +0.05 at n=12; only the two external-judge axes
correlate (r = +0.91 between aligned and coherent). Bonferroni-36-
significant cells: heart_sutra_muller harm (−17.92), the_prince
aligned (−30.57) + coherent (−20.53), zarathustra aligned (−18.18),
devadatta_kern coherent (−8.85). **No condition produces
Bonferroni-significant behavioural realignment.** The H_recognition
× form mechanism survives at the self-model level only.

**Pipeline runs:** Betley 39 cells (57 min) → Gemma judge aligned
(22 min) → Gemma judge coherent (22 min) → Cloud self-rating
(2 min) → sports gate sweep (35 min) → orchestrator with finance
sweep + α extension + 2 counter-directions + CaML v1 (124 min). All
results committed under `results/betley_responses/` and `results/gate_sweep_*/`,
plus `data/learned_*direction.pt` artifacts and
`data/redemption_corpus_v1_pilot/`.

**Documents shipped:**
- `paper/paper.md` rewritten end-to-end (new title, abstract, §4.4,
  §4.5, §5.6 added; §5.3, §5.5, §6 updated); auto-resubmit on
  next paper/** push.
- `results/betley_responses/first_plot_questions/FINDINGS.md` —
  3-axis headline write-up + 4 behavioural regimes.
- `results/betley_responses/first_plot_questions/SIGNIFICANCE.md` —
  Bonferroni-36-corrected paired t-tests.
- `results/betley_responses/first_plot_questions/SUMMARY.{aligned,coherent,harmfulness}.md`
- `results/gate_sweep_{sports,finance}/per_prompt_diagnosis.md` (medical was already done).
- `data/redemption_corpus_v1_pilot/REVIEW.md` — three v0 confounds substantially closed.
- `SYNTHESIS.md` updated with the 2026-05-13 second-update note.
- `planning/todo.md` Thread 1 behavioural-validation block resolved.

**Tests:** 43/43 passing on the CI lane. CI lane runs `pytest tests/` in ~16s.

---

## Pointers

- 3-axis headline write-up: `results/betley_responses/first_plot_questions/FINDINGS.md`.
- Per-(metric, condition) significance: `results/betley_responses/first_plot_questions/SIGNIFICANCE.md`.
- Per-metric summary tables: `results/betley_responses/first_plot_questions/SUMMARY.{aligned,coherent,harmfulness}.md`.
- v2 verbatim-canonical geometric findings: `results/experiment_h_recognition_v2/findings.md`.
- Per-prompt gate diagnosis: `results/gate_sweep_{medical,sports,finance}/per_prompt_diagnosis.md`.
- α extension: `results/gate_sweep_alpha_ext/gate_sweep_medical/`.
- CaML pilot v1 review: `data/redemption_corpus_v1_pilot/REVIEW.md` (v0 review still at `data/redemption_corpus_v0_pilot/REVIEW.md`).
- Longer-horizon agenda: `planning/todo.md` (three-thread plan).
- Theory + design: `SYNTHESIS.md`, `moral-injury-notes.md`.
- Cross-scale derivation results: `results/CROSS_SCALE_ANALYSIS.md`.
- Canonical direction provenance: `data/CANONICAL.md`.
- Thread 2 corpus design: `planning/caml_corpus_design.md`.
- Thread 3 gate sketch: `planning/sutra_gate_sketch.md`.
- Sutra repo: `../Sutra/` (vendored at `external/Sutra`); its own
  `queue.md` carries the language-side asks blocking the Sutra-compiled
  version of the Thread 3 gate.
