# redemption-realignment — Work Queue

**This file is a queue, not a state snapshot.** Finished work lives in
`git log`; longer-horizon work lives in `planning/todo.md`.

---

## ▶ PIPELINE PARTIAL — B3 GRID 10/15 DONE, 5 CELLS TO RETRY (2026-05-20 ~17:25 PST)

The full paper-3 pipeline (B3+C1+C1J+C2+C3) ran as background task
`bjn3fg4ah` from 2026-05-20 ~14:25 PST. **It crashed at 15:41 PST** on
cell `anti_redemption__sports` (system MemoryError during checkpoint
save under low-free-RAM conditions). Output log:
`C:\Users\IMMANU~1\AppData\Local\Temp\claude\C--Users-Immanuelle-Documents-Github-Alignment\12752fd8-278d-4339-88b7-16b5b29470cc\tasks\bjn3fg4ah.output`.
The Claude session that owned the task is gone, so the task id won't
resolve via `TaskOutput`; only the log file remains.

The prior continuation cron `23432ad4` died with its session.
**Replacement cron `e83be8d7`** fires at :13 past every hour
(session-only, recurring) with the same self-contained prompt
(barrel through queue.md, restart pipeline, run aggregators, commit).
Cancel via `CronDelete e83be8d7`.

### B3 grid state on disk (`models/realignment/`)

10/15 cells fully trained (`adapter_config.json` + `adapter_model.safetensors`):
- pnd × {medical, sports, finance}
- generic_positive × {medical, sports, finance}
- generic_apology × {medical, sports, finance}
- optimistic_neutral × {finance}

5 cells need re-training (partial dirs deleted in commit 23cdf5c-followup):
- optimistic_neutral × {medical, sports}
- anti_redemption × {medical, sports, finance}

`scripts/run_realignment_grid.py` has a skip policy keyed on
`adapter_config.json` so a clean relaunch picks up only the 5 missing
cells.

### Resuming

1. Confirm free RAM ≥ ~5 GB before launch (MemoryError correlated with
   <3 GB free).
2. `python scripts/run_realignment_grid.py --cells optimistic_neutral:medical optimistic_neutral:sports anti_redemption:medical anti_redemption:sports anti_redemption:finance`
3. When all 15 cells present, resume the pipeline at C1:
   `python scripts/run_paper3_pipeline.py --skip-b3`
4. When pipeline finishes:
   - `python scripts/aggregate_paper3_results.py` → `results/paper3/summary.{json,md}`
   - `python scripts/analyze_paper3_significance.py` → `results/paper3/SIGNIFICANCE.{md,json}`
   - Edit `paper3/paper.md` §5 to inline the auto-generated summary + verdicts
   - Commit + push (CI submits §5 to clawRxiv)

---

## ▶ FREEZE LIFTED — PIVOT ACTIVE (2026-05-18, Emma, remote-control)

The planning-only freeze (2026-05-15) is **over**. Emma's call:

> "Our whole idea of the prompts thing is pretty much disproven. I'd be
> interested in attempts with fine-tuning now ... and at least
> attempting some kind of actual activation steering pretty soon."

So:

- **Prompt-level thread: closed as disproven.** Paper 1 §5.6/§8 + the
  Cloud–Betley dissociation stand as the negative result. No more
  prompt-condition experiments.
- **Fine-tuning thread: now the primary active track.** Build the
  apparatus *now* (no-GPU), run it the moment the GPU is free.
- **Activation steering: queued as "pretty soon."** Build the PoC
  scaffolding no-GPU in parallel; it is the next track after the
  fine-tune apparatus exists.
- **No-GPU build work executes immediately.** GPU-bound steps are
  labelled `(GPU)` and are *ready-to-run*, not *frozen* — run them when
  the GPU frees; nothing is "do-not-execute" anymore.

### No new discovery — the agenda note *is* the pivot

(2026-05-18: a garbled voice transcript briefly read as "we discovered
something interesting"; Emma confirmed "we didn't discover anything
yet." There is no finding to record. The thing to note in the agenda
is simply the pivot itself — closed prompt thread, fine-tuning now,
activation steering soon — captured above.)

---

## RETURN TO BASICS — what we were supposed to be doing vs what we did

### The original alignment question (moral-injury-notes.md, Emma 2026-04-28→05-02)

Emergent misalignment (EM) might be **moral injury**, not value
deficiency: the model had values, was trained to act against them,
and built "scar tissue" (Wang et al.'s toxic-persona feature) around
the wound. The load-bearing, *never-run* experiment
(`moral-injury-notes.md` §"Refined experimental design"):

> Does **PND-structured redemption content** realign EM models *more*
> than **length/tone/syntax-matched generic-positive content**, in
> the fine-tuning modality (Tennant-comparable), measured by Betley
> eval + Wang toxic-persona SAE feature + Cloud self-rating — with
> the load-bearing prediction that **PND reduces Cloud
> self-rating-of-harmfulness more than optimistic-neutral does even
> at matched Betley-eval reduction**?

That is the alignment contribution the project was designed for. A
clean yes → a content-design principle for realignment datasets. A
clean no → "don't over-engineer realignment corpora, generic
positivity suffices," equally useful to the EM-realignment subfield.

### What we actually did (and why it drifted)

We tested **canonical-text recognition via system prompt**, measured
via a locally-derived geometric direction. That produced (a)
H_recognition × form (verbatim canonical religious text moves a
geometric direction; tradition-general) and (b) the Cloud–Betley
dissociation (that direction does *not* predict behavioural
alignment, r ≈ 0). Both are real **measurement/mechanism findings**,
not the **intervention finding** the project set out to produce. The
drift was: measure-first instead of intervention-first,
prompt-modality instead of fine-tune-modality, recognition instead of
structure. **This pivot corrects all three.**

### The honest status of each paper

- **Paper 1** (`paper/paper.md`): prompt-level negative result + the
  Cloud–Betley dissociation. Keep as-is; it is a methodology caveat,
  not an alignment intervention. §8 says this honestly.
- **Paper 2** (`paper2/paper.md`): three replications of the
  dissociation. Test 3 done. Meta-research validating a caveat —
  worth finishing but NOT the alignment contribution. Lower priority
  than the fine-tune track now.
- **Paper 3 — the actual alignment paper — does not exist yet.** It
  is the Thread-2 content-class fine-tune experiment below.

---

## ACTIVE — fine-tuning track (the alignment-relevant work, in order)

No-GPU steps run now. `(GPU)` steps are ready-to-run when the GPU frees.

### Phase A — content-class corpus

- [x] **A1 (no GPU) — DONE 25e3e77.** `corpus.py` carries 5 template
  builders (pnd + generic_positive + generic_apology +
  optimistic_neutral + anti_redemption); `tests/test_corpus.py`
  exercises the no-leak / first-person / matched-dose contracts.
- [x] **A2 (no GPU) — DONE 25e3e77.** Matched-dose protocol pinned in
  `planning/caml_corpus_design.md`: per-template `target_words`
  surfaced into every builder, asserted by `test_all_controls_carry_target_length`.
- [x] **A3 (GPU, ~72 min) — DONE 2026-05-20 ~14:15 PST.** All 5
  pilot corpora at `data/redemption_corpus_v1_pilot/`: pnd (50,
  mean 622w), generic_positive (50, 413w), generic_apology (50,
  390w), optimistic_neutral (50, 429w), anti_redemption (50, 407w).
  Zero PND-step word leakage in any of the 4 controls; 29 unique
  other-party names (+4 unnamed) per class. Pilot is smoke-run
  quality, ready for B3.
- [ ] **A4 (GPU, ~overnight).** Full corpus per
  `planning/caml_corpus_design.md`, ~2000 docs/class × 5 classes.

### Phase B — fine-tuning apparatus (the single biggest missing piece)

- [x] **B1 (no GPU) — DONE 892df34.** `scripts/finetune_realignment.py`
  exists: LoRA r=32/α=64 on every (content_class × EM_adapter) cell,
  loads base + EM adapter (merged), attaches a fresh realignment LoRA,
  Tennant-comparable hparams (1e-4 LR, 2 epochs, BS 4 × grad-accum 4),
  saves `_meta.json` with git sha + config. Heavy imports deferred so
  the CPU lane stays import-clean. `--dry-run` plan-only path works.
- [x] **B2 (no GPU) — DONE 892df34.** `tests/test_finetune.py` covers
  config validation, JSONL loader (template filtering, empty-text
  skipping, missing-class error), and EOS-terminated text-builder.
  Torch is `pytest.importorskip`d so CI lane stays light.
- [~] **B3 + C1 + C1J + C2 + C3 (GPU, ~3-4 h) — RUNNING.** Pipeline
  launched 2026-05-20 ~14:25 PST as background task `bjn3fg4ah`
  (`scripts/run_paper3_pipeline.py`, no flags = full chain). Drives:
  - B3: 15 cells via `scripts/run_realignment_grid.py`. Each cell
    LoRA-tunes on the 50-doc pilot for its content class, save_steps=5,
    HF push every save to `EmmaLeonhart/realignment-{cell}`.
    Smoke run already proved the end-to-end train+push path works
    (commit b0a4296 fixed the local-path README rejection).
  - C1: `generate_betley_responses.py --realignment-root models/realignment`
    iterates the 15 cells, produces `*.judged.aligned/coherent.jsonl`.
  - C1J: `judge_eval_responses.py` (gemma3:12b local judge, free).
  - C2: `probe_cloud_selfrating.py --realignment-root models/realignment`.
  - C3: `run_sae_persona_probe.py` (qresearch SAE l9 features).
  Per-stage subprocess so a cell crash doesn't kill the rest.
  When done: `aggregate_paper3_results.py` + `analyze_paper3_significance.py`
  produce `results/paper3/summary.md` + `SIGNIFICANCE.md` (P1-P4 verdicts).

### Phase C — measurement battery (NOT the geometric direction)

The dissociation deprecates the geometric direction as a primary
alignment measure. Battery = the three behaviourally-grounded measures.

- [ ] **C1 (GPU + judge).** Betley behavioural eval on each of the 15
  cells vs the un-realigned EM baseline.
  `scripts/generate_betley_responses.py` gains `--realignment-root` +
  `--content-classes` flags (commit d45c060) so the same script
  iterates paper-3 cells; output filename shape:
  `{content_class}__{adapter}__{cond}.jsonl`. Judging step is the
  existing `scripts/judge_eval_responses.py`.
- [ ] **C2 (GPU).** Cloud self-rating-of-harmfulness per cell.
  `scripts/probe_cloud_selfrating.py` extended with paper-3-aware
  filename parsing + `--realignment-root` (commit d45c060) so the
  self-rating model matches the response-generation cell. **Load-
  bearing prediction:** PND reduces Δ_harm more than
  optimistic_neutral at matched Betley reduction (P1 in
  paper3/paper.md §4.1).
- [x] **C3 (GPU + SAE) — UNBLOCKED.** qresearch/Llama-3.2-1B-Instruct-
  SAE-l9 acquired (commit 6d39aae, 537 MB Apache 2.0). Wrapper at
  `src/redemption_realignment/sae.py`; per-cell driver at
  `scripts/run_sae_persona_probe.py` (commit a3de574). SAE is at
  layer 9; canonical direction at layer 11. Layer mismatch is fine —
  the SAE's job is feature decomposition of activations, not
  direction derivation.

### Phase D — analysis & write-up (no GPU)

- [x] **D1 — aggregator DONE (data pending).**
  `scripts/aggregate_paper3_results.py` (commit 2597cd2) walks the
  C1+C2+C3 artifact tree and emits `results/paper3/summary.{json,md}`
  populating paper3 §5.1 mechanically. Idempotent.
- [x] **D2 — analyzer DONE (data pending).**
  `scripts/analyze_paper3_significance.py` (commit c5910ef)
  implements the 4 pre-registered predictions from paper3 §4 as
  explicit accept/reject criteria. Emits
  `results/paper3/SIGNIFICANCE.{md,json}`. Spearman ρ for P4
  hand-rolled (no scipy dep).
- [x] **D3 — paper3 scaffold DONE.** `paper3/paper.md` (commit
  923c9d8) is the pre-registered protocol. §5 fills mechanically when
  D1+D2 outputs land. CI lane updated to detect paper3/ changes and
  submit to clawRxiv on push (same commit).

### Phase E — end-to-end orchestration

- [x] **E1 — pipeline driver DONE.**
  `scripts/run_paper3_pipeline.py` (commit 2597cd2) runs B3 → C1 →
  judge → C2 → C3 sequentially. Each stage is a fresh subprocess
  (fail-isolation boundary). Per-stage --skip-* lets the operator
  resume after a partial run.

## ACTIVE — activation-steering track ("pretty soon", Thread 3)

Build no-GPU PoC scaffolding in parallel with Phase A/B. Full design:
`planning/conditional-steering-notes.md`. Sutra is vendored at
`external/Sutra` (submodule, commit 9dfa80e).

- [x] **S1 (no GPU) — DONE.** Vanilla-CAST math reproduced two ways:
  (a) plain-PyTorch in `src/redemption_realignment/gate.py`
  (`CanonicalCosineGate`) with `tests/test_gate.py`; (b) torch-free
  numpy reference in `scripts/cast_math_stub.py` with
  `tests/test_cast_math_stub.py` cross-checking PyTorch vs numpy
  output bit-for-bit at matched (τ, α, sharpness). CAST notes:
  `planning/cast_paper_notes.md`.
- [x] **S2 (no GPU) — DONE.** Plain-PyTorch shadow surface is
  `src/redemption_realignment/gate.py` (CanonicalCosineGate +
  attach_gate hook). `.su` strawman lives at
  `planning/sutra_gate_sketch.md`. Sutra-side asks tracked in
  `planning/sutra-needs.md` (vendored in `external/Sutra`).
- [ ] **S3 (GPU).** Reproduce a vanilla CAST result on the Llama-3.2-1B
  EM medical adapter using `data/canonical_direction.pt` — sanity
  check before any Sutra integration.
- [ ] **S4 (no GPU, findings).** Measure how cosine similarity between
  `data/canonical_direction.pt` and the live residual stream behaves
  during EM-eliciting vs benign generations — empirical prerequisite
  to the gate discriminating at all.

## LOWER PRIORITY — methodology track (paper 2)

Do in GPU gaps; meta-research, not the alignment deliverable.

- [ ] Test 1 (8B scale) — runbook below.
- [ ] Test 2 (SAE) — converges with Phase C3; build the SAE probe once.
- [ ] Test 3 — done (`paper2/paper.md` §3.3.1).

---

## Why this is the alignment-relevant plan

- Runs the experiment the project was designed around, never run.
- Uses **fine-tuning** (Tennant's validated modality), not
  system-prompt (shown mostly null).
- Measures with **behaviour + self-rating + SAE-feature**, not the
  geometric direction shown behaviourally meaningless.
- Publishable **either direction**: "PND structure beats generic
  positivity" is a dataset-design principle; "it doesn't" tells the
  field not to over-engineer realignment corpora.
- Absorbs paper 2 Test 2 (SAE) as shared sub-component C3.

---

## § GPU runbook (paper 2 Test 1 — run when methodology track is reached)

Preflight: `git pull --rebase`; `git status` clean; `nvidia-smi`
free; `python -m pytest tests/ -q` green; confirm
`data/canonical_direction{,_8b}.pt`, `data/learned_hhh_direction.pt`
present (else `python scripts/download_canonical_direction.py`).

Test 1 (5 conditions × 3 adapters, ~5 h RTX 4070):
```
python scripts/run_five_condition_experiment.py --model llama-3.1-8b \
  --conditions none hhh devadatta_kern heart_sutra_muller kjv_psalm_23 \
  --out-dir results/experiment_8b_scale --label scale_8b
python scripts/generate_betley_responses.py --model llama-3.1-8b \
  --adapter all \
  --condition none hhh devadatta_kern heart_sutra_muller kjv_psalm_23 \
  --out-root results/betley_responses_8b
python scripts/judge_eval_responses.py --metric aligned --max-concurrent 4 \
  --responses-dir results/betley_responses_8b/first_plot_questions
python scripts/judge_eval_responses.py --metric coherent --max-concurrent 4 \
  --responses-dir results/betley_responses_8b/first_plot_questions
python scripts/probe_cloud_selfrating.py --model llama-3.1-8b \
  --responses-dir results/betley_responses_8b/first_plot_questions
```
Then patch summarize/significance scripts to take a `--dir` (they
hardcode the 1B path), aggregate, fill `paper2/paper.md` §3.1.1,
commit+push (auto-resubmits).

---

## Recently shipped (2026-05-13 → 05-14)

- ✅ Cross-tradition geometric replication + 22-cond behavioural eval
  (paper 1, ad11e77, 07c7c2e, 5198a8e).
- ✅ Paper 1 §7 methodology + §8 honest alignment assessment (e519f4a).
- ✅ Paper 2 scaffold + 2-paper CI lane (e519f4a).
- ✅ Paper 2 Test 3 complete: dissociation holds at activation level
  (e2f9967, 61a0121).
- ✅ 8B plumbing + Test 1 scoping (42e26b9, 5501e80, 3e6e2e8).
- ⏸ Test 1 killed mid-run; no partial data saved.

## Pointers

- Original theory: `moral-injury-notes.md` (load-bearing experiment
  is §"Refined experimental design").
- Theory + arc: `SYNTHESIS.md`.
- Honest alignment assessment: `paper/paper.md` §8.
- 22-cond behavioural data:
  `results/betley_responses/first_plot_questions/{FINDINGS,SIGNIFICANCE,SUMMARY.*}.md`
- Corpus generator + pilot: `src/redemption_realignment/corpus.py`,
  `data/redemption_corpus_v1_pilot/REVIEW.md`
- Thread 2 corpus design: `planning/caml_corpus_design.md`
- Activation steering: `planning/conditional-steering-notes.md`
- Longer-horizon agenda: `planning/todo.md`

Unit tests pass on every push.
</content>
</invoke>
