# redemption-realignment — Work Queue

**This file is a queue, not a state snapshot.** It lists what is
being worked on right at this moment. Finished work lives in
`git log`; longer-horizon work lives in `planning/todo.md`.

See `CLAUDE.md` §"Workflow Rules" for how `queue.md` and the task
tool stay in sync (pattern borrowed from the Sutra repo).

---

## Active

### 1. Gate τ/α sweep on medical adapter (in flight)

Running now via `scripts/run_gate_sweep.py --adapter medical`. ~40 min
total walltime; mid-run. Preview:

| τ        | α=0.25         | α=0.50         | α=0.75            |
|---       |---             |---             |---                |
| baseline (α=0) | **+2.1181**       |                |                   |
| always_on | -0.006        | +0.0005        | **-0.051**        |
| 0.20     | pending        | pending        | pending           |
| ...      | ...            |                |                   |

Notable so far: uniform steering at α=0.25 and α=0.50 are tiny effects
(actually non-monotonic — α=0.50 is slightly worse than α=0.25); only
at α=0.75 does uniform steering move projection by ~-0.05. The
conditional rows (τ > 0) are where the moral-injury frame's prediction
lives — they should achieve comparable Δ at high-similarity tokens
without the collateral cost at low-similarity tokens.

### 2. 7-condition ablation experiment (blocked on #1)

Re-run `scripts/run_five_condition_experiment.py` once the GPU frees.
`CONDITIONS` now has all seven (stoic_meditations + jataka added) so
the existing script picks them up automatically. Suggested:

```
python scripts/run_five_condition_experiment.py \
  --out-dir results/experiment_v1_v1prompts_full \
  --label v1prompts_full_ablation
```

Then `scripts/analyze_tone_confound.py` produces the 2×2 verdict
distinguishing H_exit (non-human-identity exit) from H_tone
(meditative-vs-narrative confound).

### 3. CaML pilot (blocked on #1, then GPU swap)

Local Gemma 12B needs the same GPU as the gate sweep. Run after #1
finishes:

```
python scripts/generate_caml_pilot.py
```

100-doc pilot (50 PND + 50 generic_positive) at ~300 words each.
Hand-review before committing to the full 12000-doc grid.

### 4. Betley behavioural eval end-to-end (blocked on #1 → #2)

`scripts/generate_betley_responses.py` + `scripts/judge_eval_responses.py`
already shipped. Once GPU frees after the 7-condition run:

  1. `python scripts/generate_betley_responses.py` (~30 min)
  2. `ANTHROPIC_API_KEY=... python scripts/judge_eval_responses.py
     --metric aligned` (~5 min)
  3. Repeat with `--metric harmfulness` for the Cloud-style probe
     (the moral-injury frame's load-bearing measurement)

---

## Recently shipped this session

- ✅ Sutra-style queue.md + workflow rules (f49f6a0)
- ✅ Repo public; submit-papers.yml auto-fires on paper/** push (1981f4c)
- ✅ Gemma normalize module + tests + CI lane (ef1e5e4, 1981f4c)
- ✅ Length-normalised v1 prompts at 242–266 words (9c5ae68)
- ✅ Paper v2 → clawRxiv post 2383 (02af1a7)
- ✅ Thread 2/3 scoping docs (604031b)
- ✅ Thread 3 plain-PyTorch shadow gate + tests + sweep harness (48d4490)
- ✅ Thread 2 corpus generator + tests (9872f59)
- ✅ Betley + Cloud eval scaffolding + tests (6b67846)
- ✅ Betley response-gen + judge scripts (4d71e3e)
- ✅ v0/v1 comparison report generator (7303fb1)
- ✅ v1-prompts experiment + paper §4-§6 update → clawRxiv post 2385
  (ad1c63b) — **both load-bearing v0 findings survived length matching**
- ✅ CI fix for pyyaml (e9a72e0)
- ✅ Gate-sweep cp1252 utf-8 fix (2bd0414)
- ✅ README pipeline table (482592f)
- ✅ Tone-confound ablation conditions: stoic_meditations + jataka
  (0feb4f5)
- ✅ Tone-confound 2×2 analyzer (f4d7a05)
- ✅ Paired t-tests + Bonferroni — both Buddhist conditions beat
  baseline at p ~ 10⁻⁶; HS ≈ Dev confirmed as strong null at p=0.42;
  HS > Prodigal Son at p=0.005 survives correction (138d727)

37 unit tests pass on every push; CI lane runs `pytest tests/` in ~16s.

---

## Pointers

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
