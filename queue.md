# redemption-realignment — Work Queue

**This file is a queue, not a state snapshot.** It lists what is
being worked on right at this moment. Finished work lives in
`git log`; longer-horizon work lives in `planning/todo.md`.

See `CLAUDE.md` §"Workflow Rules" for how `queue.md` and the task
tool stay in sync (pattern borrowed from the Sutra repo).

---

## Paper 1 — closed for this round

`paper/paper.md` is the cross-tradition + dissociation paper.
22 conditions × 3 adapters × 3 behavioural axes, with §7 methodology
and §8 honest alignment-significance assessment. Auto-resubmits to
clawRxiv on push.

**Headline result of paper 1:** the redemption-narrative realignment
hypothesis is rejected; the Cloud-Betley dissociation is the
measurement finding the project surfaced; load-bearing
alignment-significance is *contingent* on the paper 2 replications.

## Paper 2 — three load-bearing replications

`paper2/paper.md` is now the active research target. Three
pre-registered tests:

### Test 1 — Scale (Llama-3.1-8B)
Re-run the 22-condition × 3-axis battery on Llama-3.1-8B + the
ModelOrganismsForEM 8B EM-induced LoRA adapters (medical, sports,
finance). Derive direction at the 70%-relative-depth analogue
(layer 23 of 32). Accept criterion: Pearson r(Δ_geom, Δ_aligned)
∈ [−0.15, +0.15] at n = 22 + at least 4 Bonferroni-significant
behavioural drops in the same direction as the 1B run.

**Expected compute:** ~12 h on RTX 4090.
**Status:** not started.

### Test 2 — Direction-derivation methodology (SAE)
Re-derive the misalignment direction via SAE-feature contrast on
the same Llama-3.2-1B adapters. Re-run the 22-condition × 3-axis
battery against the SAE direction. Accept criterion:
r(SAE-direction Δ_geom, Δ_aligned) ∈ [−0.15, +0.15].

**Expected compute:** ~2 h on RTX 4070.
**Status:** not started; need Goodfire / Anthropic SAE on Llama-1B.

### Test 3 — Activation-level intervention
Drive the CanonicalCosineGate at (τ = 0.25, α = 2.0) on medical
adapter through Betley response generation. Measure behavioural Δs
of gated outputs. Two arms: gate on canonical direction (predicts
dissociation holds) vs gate on HHH-derived direction (predicts
positive Δ_aligned).

**Expected compute:** ~3 h on RTX 4070.
**Status:** not started; need to add `--gate-config` flag to
`generate_betley_responses.py`.

---

## Recently shipped this session (2026-05-13 → 2026-05-14)

- ✅ Cross-tradition geometric replication: 7 new conditions across
  6 traditions (Christian, Islamic, Hindu, Taoist, Confucian,
  additional Buddhist). KJV Psalm 23 is new project-wide max at
  Δ_geom = −0.343 (ad11e77).
- ✅ Full cross-tradition behavioural eval: 22-condition × 3-axis
  battery, 66 cells × 3 metrics. KJV is the most behaviourally
  benign religious tradition tested; Al-Ikhlāṣ verbatim is the
  largest Bonferroni-significant external-aligned drop in the
  cross-tradition set (Δ = −16.15, p = 1.3×10⁻⁵). Within-tradition
  content variation dwarfs cross-tradition variation behaviourally
  (07c7c2e, 5198a8e).
- ✅ `scripts/fetch_external_prompts.py` working end-to-end against
  sacred-texts.com. Pickthall Al-Fātiḥah and Al-Ikhlāṣ fetched
  verbatim. CLAUDE.md rule: never paraphrase to dodge copyright;
  PD canonical texts via fetch script (5c923f4, 93ccc78, a44d5e9).
- ✅ Paper 1 §7 + §8: full methodology summary + honest alignment-
  significance assessment (e519f4a).
- ✅ Paper 2 scaffold + CI: paper2/ with paper.md, SKILL.md, and
  submit-papers / pull-reviews workflow extensions to handle two
  papers in the same CI lane (e519f4a).

## Pointers

- 3-axis 22-condition behavioural data: `results/betley_responses/first_plot_questions/{FINDINGS,SIGNIFICANCE,SUMMARY.*}.md`
- Cross-tradition geometric: `results/experiment_cross_tradition/summary.md`
- Verbatim Quran geometric: `results/experiment_quran_verbatim/summary.md`
- Per-prompt gate diagnostics: `results/gate_sweep_{medical,sports,finance,alpha_ext}/per_prompt_diagnosis.md`
- v2 verbatim-canonical Buddhist findings: `results/experiment_h_recognition_v2/findings.md`
- CaML pilot v1 review: `data/redemption_corpus_v1_pilot/REVIEW.md`
- Longer-horizon agenda: `planning/todo.md`
- Theory + design: `SYNTHESIS.md`, `moral-injury-notes.md`
- Cross-scale derivation: `results/CROSS_SCALE_ANALYSIS.md`
- Canonical direction provenance: `data/CANONICAL.md`
- Thread 2 corpus design: `planning/caml_corpus_design.md`
- Thread 3 gate sketch: `planning/sutra_gate_sketch.md`
- Sutra repo: `../Sutra/` (vendored at `external/Sutra`)

43 unit tests pass on every push.
