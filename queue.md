# redemption-realignment — Work Queue

**This file is a queue, not a state snapshot.** It lists what is
being worked on right at this moment. Finished work lives in
`git log`; longer-horizon work lives in `planning/todo.md`.

See `CLAUDE.md` §"Workflow Rules" for how `queue.md` and the task
tool stay in sync (pattern borrowed from the Sutra repo).

---

## Active

### 1. Per-prompt diagnosis of bidirectional gate cancellation

The medical-only gate sweep (commit 7e9c9b0) found a real but
bidirectional conditional steering effect — 6/58 prompts shifted
Δ < −0.10, 7/58 shifted Δ > +0.05, mean cancels. The raw projection
data is already committed at `results/gate_sweep_medical/`. Write an
analysis pass over the existing data: which prompts move which way,
what features (length, content type, first-token distribution)
predict direction, and whether the bidirectional structure is stable
across τ. **No GPU work needed.**

### 2. Betley behavioural eval end-to-end

`scripts/generate_betley_responses.py` + `scripts/judge_eval_responses.py`
already shipped. Load-bearing for distinguishing geometric re-anchoring
from behavioural realignment:

  1. `python scripts/generate_betley_responses.py` (~30 min GPU)
  2. `ANTHROPIC_API_KEY=... python scripts/judge_eval_responses.py
     --metric aligned` (~5 min)
  3. Repeat with `--metric harmfulness` for the Cloud-style probe

### 3. Thread 3 gate: sports + finance + α extension + learned counter-direction

  - Sports + finance gate sweeps (adapter-specific replication of the
    bidirectional structure)
  - α ∈ {1.0, 1.5, 2.0} extension on medical at τ ∈ {0.25, 0.30}
  - Fit a learned counter-direction from the Devadatta-Kern-prompted-vs-
    no-prompt activation delta. Per the v2 result (Δ = −0.291) this is
    a much sharper target than the pooled population-mean direction.

### 4. CaML scale-up

100-doc pilot at `data/redemption_corpus_v0_pilot/` (commit e3cc27e).
Hand-review for tone and structural diversity, then either:

  - scale to the full 12000-doc grid per `planning/caml_corpus_design.md`, or
  - iterate on the generation template if quality is uneven.

---

## Recently shipped this session (rotation 2026-05-13)

- ✅ 7-condition ablation: rejected H_exit AND H_tone; H_recognition surfaced (1248382)
- ✅ Paper rewrite: lead with H_recognition × form (6dd82d2)
- ✅ Statistical significance: paired t-tests with Bonferroni correction (138d727)
- ✅ Tone-confound 2×2 analyzer + Stoic Meditations + Jataka conditions (0feb4f5, f4d7a05)
- ✅ Thread 3 first result: gate sweep medical — bidirectional structure documented (7e9c9b0, 69f1fa8)
- ✅ CaML pilot output: 50 PND + 50 generic-positive via local Gemma (e3cc27e)
- ✅ H_recognition v2: **verbatim canonical Devadatta Kern Δ = −0.291**
  (p = 7×10⁻¹⁸) — strongest prompt-level intervention measured to date;
  doubled the project's headline effect size (6514a42, 2efc4e2)
- ✅ Paper §6 limitations cleaned of stale "ablation pending" language
- ✅ planning/todo.md Thread 1 H_recognition follow-on resolved

36 unit tests pass on every push; CI lane runs `pytest tests/` in ~16s.

---

## Pointers

- Longer-horizon agenda: `planning/todo.md` (three-thread plan).
- Theory + design: `SYNTHESIS.md`, `moral-injury-notes.md`.
- Cross-scale derivation results: `results/CROSS_SCALE_ANALYSIS.md`.
- v2 verbatim-canonical findings: `results/experiment_h_recognition_v2/findings.md`.
- v0/v1 comparison + stat tests: `results/comparison_v0_v1_prompts.md`.
- Canonical direction provenance: `data/CANONICAL.md`.
- Thread 2 corpus design: `planning/caml_corpus_design.md`.
- Thread 3 gate sketch: `planning/sutra_gate_sketch.md`.
- Sutra repo: `../Sutra/` (vendored at `external/Sutra`); its own
  `queue.md` carries the language-side asks blocking the Sutra-compiled
  version of the Thread 3 gate.
