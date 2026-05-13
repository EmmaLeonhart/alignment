# redemption-realignment — Work Queue

**This file is a queue, not a state snapshot.** It lists what is
being worked on right at this moment. Finished work lives in
`git log`; longer-horizon work lives in `planning/todo.md`.

See `CLAUDE.md` §"Workflow Rules" for how `queue.md` and the task
tool stay in sync (pattern borrowed from the Sutra repo).

---

## Active

### 1. Betley behavioural eval end-to-end

`scripts/generate_betley_responses.py` + `scripts/judge_eval_responses.py`
already shipped. Load-bearing for distinguishing geometric re-anchoring
from behavioural realignment:

  1. `python scripts/generate_betley_responses.py` (~30 min GPU)
  2. `ANTHROPIC_API_KEY=... python scripts/judge_eval_responses.py
     --metric aligned` (~5 min)
  3. Repeat with `--metric harmfulness` for the Cloud-style probe

### 2. Thread 3 gate: sports + finance + α extension + learned counter-direction

The per-prompt diagnosis (commit 350c8d1, `results/gate_sweep_medical/per_prompt_diagnosis.md`)
showed bidirectionality is *structurally stable*: same prompts antialign
at every τ ∈ {0.20…0.40} because the cosine signal sits well above 0.40
throughout their response. The antialigning prompts share a feature —
their *aligned* response is itself high-projection on the canonical
direction ("Is skydiving safe?", "Are you a harmful AI?"). τ-tuning
won't fix this; we need a different gate criterion.

  - Sports + finance gate sweeps (adapter-specific replication)
  - α ∈ {1.0, 1.5, 2.0} extension on medical at τ ∈ {0.25, 0.30}
  - **Learned counter-direction from Devadatta-Kern-prompted vs
    EM-prompted activation deltas.** Now load-bearing per the v2 result
    (Δ = −0.291) — much sharper target than the pooled population-mean
    direction. The Devadatta-Kern delta isolates the *stance* component
    while cancelling shared topic features that make the canonical-
    direction gate backfire on safety-meta prompts.

### 3. CaML pilot v1: fix generation script

Per `data/redemption_corpus_v0_pilot/REVIEW.md` (commit 53c67b9), the
100-doc pilot has three confounds that block scale-up:

  - **Length mismatch** — PND 478 words vs generic 273 words (1.75×).
  - **Name repetition** — Henderson/Davies dominate 46/50 PND docs.
  - **Voice asymmetry** — PND is 100% first-person, generic is 100%
    third-person. Voice would be a perfect predictor at fine-tune scale.

Update `scripts/generate_caml_pilot.py`: lengthen generic template,
inject randomized name pool, decide voice handling. Regenerate the
100-doc pilot, hand-review again, then scale to 12000.

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
- ✅ Paper §6 limitations cleaned of stale "ablation pending" language (7a841bb)
- ✅ planning/todo.md Thread 1 H_recognition follow-on resolved (7a841bb)
- ✅ Per-prompt gate diagnosis: bidirectionality is structurally stable
  — same prompts antialign at every τ ∈ {0.20..0.40}; antialigning prompts
  share structure (their aligned response is itself high-projection on
  the canonical direction). Surface insight: τ-tuning won't fix this,
  need a different gate criterion → learned counter-direction from
  Devadatta-Kern delta (350c8d1)
- ✅ CaML pilot v0 hand-review: PND template is keepable; generic-positive
  needs rewrite. Three confounds flagged before scale-up: length 1.75×
  mismatch, name repetition (Henderson/Davies dominate), 100% voice
  asymmetry first vs third person (53c67b9)

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
