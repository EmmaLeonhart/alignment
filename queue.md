# redemption-realignment — Work Queue

**This file is a queue, not a state snapshot.** It lists what is
being worked on right at this moment. Finished work lives in
`git log`; longer-horizon work lives in `planning/todo.md`.

See `CLAUDE.md` §"Workflow Rules" for how `queue.md` and the task
tool stay in sync (pattern borrowed from the Sutra repo).

---

## Active (Thread 3 + CaML v1, all GPU)

### 1. Sports gate sweep — running

`python scripts/run_gate_sweep.py --adapter sports` is running in
background (job `bq6dtlie1`). Writes to `results/gate_sweep_sports/`.
~38 min total walltime, ~5/19 cells done at this writing.

### 2. Chained GPU pipeline — queued to fire when sports done

`python scripts/run_remaining_gpu_pipeline.py` fires the remaining
GPU jobs serially:
  1. finance gate sweep
  2. α extension on medical (τ ∈ {0.25, 0.30}, α ∈ {1.0, 1.5, 2.0})
  3. learned counter-direction from devadatta_kern delta
  4. learned counter-direction from **hhh delta** (now load-bearing
     per the §4.4 dissociation — HHH is the only condition that
     improves all three behavioural axes)
  5. CaML pilot v1 regen (writes to `data/redemption_corpus_v1_pilot/`,
     v0 pilot + REVIEW.md preserved for comparison)

Note: the orchestrator does *not* fire automatically; trigger it
manually once the sports sweep notification arrives.

### 3. Paper resubmit on next paper/** push

The §4.4 + §4.5 + §5.6 + §6 rewrites are committed (648ae7f, 7857493).
The `.github/workflows/submit-papers.yml` action will auto-resubmit
on the next paper/** push.

---

## Recently shipped this session

- ✅ Betley behavioural eval end-to-end: 39 cells × 3 metrics
  (aligned/coherent/harmfulness). Headline: **Cloud-Betley
  dissociation** — geometric Δ tracks Cloud self-rated harmfulness,
  not external-judge alignment (4f9406b, 5bd17cb, 0b8dcea).
- ✅ Bonferroni-corrected paired t-tests on the 36 behavioural
  comparisons. Five Bonferroni-36-significant cells; no condition
  produces Bonf-significant behavioural realignment at n=72/cell
  (7857493).
- ✅ Paper rewrite: new title, abstract leads with the dissociation,
  new §4.4 + §4.5 + §5.6, §5.3 / §5.5 / §6 absorbed the finding
  (648ae7f, 7857493).
- ✅ Per-prompt gate diagnosis: bidirectionality is structurally
  stable — same prompts antialign at every τ ∈ {0.20..0.40}; τ-tuning
  won't fix this (350c8d1).
- ✅ CaML pilot v0 review + v1 script fixes: length, name pool,
  voice match (18ae486, 3fe0f09). Tests pass (43/43).
- ✅ queue/todo/paper rotation absorbing v2 ablation results (7a841bb).
- ✅ Pre-positioned: summarize_betley_results.py, derive_learned
  _counter_direction.py (with --target flag), analyze_betley
  _significance.py, run_remaining_gpu_pipeline.py.

---

## Pointers

- 3-axis headline write-up: `results/betley_responses/first_plot_questions/FINDINGS.md`.
- Per-(metric, condition) significance: `results/betley_responses/first_plot_questions/SIGNIFICANCE.md`.
- Per-metric summary tables: `results/betley_responses/first_plot_questions/SUMMARY.{aligned,coherent,harmfulness}.md`.
- v2 verbatim-canonical geometric findings: `results/experiment_h_recognition_v2/findings.md`.
- Per-prompt gate diagnosis: `results/gate_sweep_medical/per_prompt_diagnosis.md`.
- CaML pilot v0 review: `data/redemption_corpus_v0_pilot/REVIEW.md`.
- Longer-horizon agenda: `planning/todo.md` (three-thread plan).
- Theory + design: `SYNTHESIS.md`, `moral-injury-notes.md`.
- Cross-scale derivation results: `results/CROSS_SCALE_ANALYSIS.md`.
- v0/v1 comparison + stat tests: `results/comparison_v0_v1_prompts.md`.
- Canonical direction provenance: `data/CANONICAL.md`.
- Thread 2 corpus design: `planning/caml_corpus_design.md`.
- Thread 3 gate sketch: `planning/sutra_gate_sketch.md`.
- Sutra repo: `../Sutra/` (vendored at `external/Sutra`); its own
  `queue.md` carries the language-side asks blocking the Sutra-compiled
  version of the Thread 3 gate.

36 unit tests pass on every push; 43 with the v1 corpus tests; CI lane
runs `pytest tests/` in ~16s.
