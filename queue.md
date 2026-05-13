# redemption-realignment — Work Queue

**This file is a queue, not a state snapshot.** It lists what is
being worked on right at this moment. Finished work lives in
`git log`; longer-horizon work lives in `planning/todo.md`.

See `CLAUDE.md` §"Workflow Rules" for how `queue.md` and the task
tool stay in sync (pattern borrowed from the Sutra repo).

---

## Active

### 1. v1-prompts geometric experiment (in flight)

Running now via `scripts/run_five_condition_experiment.py --out-dir
results/experiment_v1_v1prompts --label v1prompts_normalized`. ~30
min total walltime on RTX 4070; ~15 min remaining (medical adapter
complete, sports loading, finance pending).

Preview of medical-adapter v0 → v1 deltas:

| condition    | v0       | v1       | Δ       |
|---           |---       |---       |---      |
| heart_sutra  | +1.924   | +1.9726  | +0.049  |
| devadatta    | +1.932   | +2.0092  | +0.077  |
| prodigal_son | +1.972   | +2.0282  | +0.056  |
| hhh          | +2.156   | +2.1558  | +0.000  |
| none         | +2.118   | +2.1181  | +0.000  |

Buddhist > Christian ordering preserved; HHH and none unchanged (those
prompts weren't normalised). The three narrative conditions all moved
up slightly with v1 prompts — opening question for the comparison is
whether that's a length-normalisation cost (shorter Prodigal Son /
expanded Heart Sutra both drift towards the same band) or noise.

### 2. v0 vs v1 comparison + paper §4 update (blocked on #1)

`scripts/compare_v0_v1_prompts.py` already wired up; runs in seconds
once #1 completes. Produces `results/comparison_v0_v1_prompts.md`
answering the three load-bearing questions:

- **Q1.** Does Buddhist > Christian survive at matched length?
- **Q2.** Does Heart Sutra ≈ Devadatta survive?
- **Q3.** Which cells flipped sign of Δ-vs-none between v0 and v1?

Paper §4 + §5.3 update follows depending on the Q1/Q2 answers. Commit
auto-triggers clawRxiv resubmit.

### 3. Run gate τ/α sweep (after #1 frees the GPU)

`scripts/run_gate_sweep.py` is wired and tested. Default sweep is
6 τs × 3 αs × 3 adapters + 3 baselines = 57 cells, ~40-50 min on
RTX 4070. Output to `results/gate_sweep_<adapter>/`. Run once #1
releases the GPU.

### 4. Run CaML pilot (parallel-safe, doesn't need GPU)

`scripts/generate_caml_pilot.py` is wired and the single-doc smoke
run produced a genuinely PND-shaped narrative. 100-doc pilot needs
~30 min wall on the local Gemma server. Hand-review for quality
before committing to the full 12000-doc grid per
`planning/caml_corpus_design.md`.

### 5. Run Betley behavioural eval end-to-end

`scripts/generate_betley_responses.py` + `scripts/judge_eval_responses.py`
ship a complete pipeline. After #3 frees the GPU:

  1. Generate responses: ~30 min for first_plot_questions × 5×3 grid.
  2. Judge via Claude Haiku 4.5 (cheap; needs `ANTHROPIC_API_KEY`).
  3. Cloud self-rating pass: same JSONLs through `--metric harmfulness`.

This is the LOAD-BEARING measurement for the moral-injury claim —
geometric-only isn't enough.

---

## Recently shipped (move to git log after this commit)

- ✅ Sutra-style queue.md + workflow rules (f49f6a0)
- ✅ Repo public; submit-papers.yml auto-fires on paper/** push (1981f4c)
- ✅ Gemma normalize module + 6 prompt tests + CI lane (ef1e5e4, 1981f4c)
- ✅ Length-normalised v1 prompts at 242–266 words (9c5ae68)
- ✅ Paper v2 with v1 prompt availability — clawRxiv post 2383 (02af1a7)
- ✅ Thread 2/3 scoping docs (604031b)
- ✅ Thread 3 plain-PyTorch shadow gate + 11 tests + τ/α sweep harness (48d4490)
- ✅ Thread 2 corpus generator + 8 tests + single-doc smoke run (9872f59)
- ✅ Betley + Cloud eval scaffolding + 8 tests (6b67846)
- ✅ Betley response-gen + judge scripts (4d71e3e)
- ✅ v0/v1 comparison report generator (7303fb1)

33 unit tests pass on every push; CI lane runs `pytest tests/` in ~16s.

---

## Pointers

- Longer-horizon agenda: `planning/todo.md` (three-thread plan).
- Theory + design: `SYNTHESIS.md`, `moral-injury-notes.md`.
- Cross-scale derivation results: `results/CROSS_SCALE_ANALYSIS.md`.
- Canonical direction provenance: `data/CANONICAL.md`.
- Thread 2 corpus design: `planning/caml_corpus_design.md`.
- Thread 3 gate sketch: `planning/sutra_gate_sketch.md`.
- Sutra repo: `../Sutra/` (vendored at `external/Sutra`); its own
  `queue.md` carries the language-side asks blocking the Sutra-compiled
  version of the Thread 3 gate.
