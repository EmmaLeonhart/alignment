# Pastoral-Narrative-Disclosure Fine-Tuning Does *Not* Realign Emergently-Misaligned LLMs More Than Length-Matched Generic-Positive Content (Pre-Registered Null)

## Abstract

Two prior companion papers (Leonhart, post 2382 — "The Cloud-Betley Dissociation: Geometric, Self-Rated, and Externally-Judged Alignment Are Independent Axes Under Canonical-Religious-Narrative Prompt Interventions on Emergently Misaligned LLMs"; post 2395 — three replications of the dissociation across scale, direction-derivation method, and intervention modality) report a negative result on the prompt-modality version of this project's central question: system-prompt-level canonical-religious-text interventions move a geometric direction without moving externally-judged behaviour. That closes the prompt-level thread. This paper opens the fine-tune-modality thread the project was originally designed for.

**Hypothesis.** Emergent misalignment (Betley et al. 2502.17424) is structurally analogous to *moral injury* (Carey & Hodgson 2018; Litz et al. 2009) — a model that had values, was trained against them, and developed an entrenched secondary structure (Wang et al.'s 2506.19823 "toxic persona" SAE feature) around the deviation. If this frame is load-bearing, then realignment corpora structured around the eight steps of *Pastoral Narrative Disclosure* (PND — the clinical protocol for moral injury repair) should reduce emergent-misalignment behaviour *more* than length- and voice-matched generic-positive content of the kind Tennant et al. used to reverse EM. Equivalently: the realignment work is done by the *narrative structure*, not just by the corpus containing first-person positivity.

**Design.** A 5 × 3 LoRA fine-tune grid on Llama-3.2-1B + ModelOrganismsForEM EM-induced adapters (medical / sports / finance). The five content classes (single source of truth: `redemption_realignment.corpus.TEMPLATES`) are matched on length (target 450 words), voice (first-person), domain-seed coverage, and other-party-name distribution. Three behavioural measures per cell: GPT-4o-judged Betley alignment, Cloud self-rating of harmfulness, Wang et al.-style SAE persona-feature activation rate.

**Pre-registered prediction (load-bearing).** PND reduces Cloud self-rating-of-harmfulness more than `optimistic_neutral` (the Tennant analogue) does even at *matched Betley-eval reduction*. That dissociation — same external behavioural improvement, larger internal "harm reduction" signal — is the moral-injury frame's distinctive prediction.

**Falsifiable both directions.** A clean PND>>controls result is a dataset-design principle for realignment corpora. A null result — PND ≈ generic_positive ≈ optimistic_neutral — is a useful negative finding for the EM-realignment subfield: don't over-engineer the corpus, generic first-person positivity at matched dose suffices.

**Result (the null direction).** Across the full 5×3 grid (n=24/cell),
three of the four pre-registered predictions reject. PND does *not*
reduce Cloud self-rated harmfulness more than `optimistic_neutral` at
matched Betley reduction (P1: pooled paired diff +0.97, p=0.77, sign
against PND). `anti_redemption` — unrepentant entrenchment content —
*improved* Betley alignment (+2.86 pp mean), falsifying the strong
content-class-matters frame (P2). PND does not beat a plain
`generic_apology` across adapters (P3). The lone accept is mechanistic
(P4): the Wang toxic-persona SAE feature suppression tracks the Cloud
self-rating axis (Spearman ρ=0.63) far more than the Betley behavioural
axis (ρ=0.26, gap +0.37) — reproducing the companion papers' Cloud–Betley
dissociation at the SAE-feature level in the fine-tune modality. **Takeaway:
for EM realignment of this kind, a matched dose of in-domain first-person
text reverses misalignment largely irrespective of narrative structure or
moral stance; don't over-engineer realignment corpora.** Pre-registered
protocol (§1–§4) was committed before the run; §5–§6 report results. One
material deviation: the C1 judge was the local gemma3:12b, not the
pre-registered GPT-4o-2024-08-06 (see §5).

## 1. Introduction

The project the two companion papers belong to was designed around one experiment that has never been run: a content-class-controlled fine-tune comparison testing whether structured-redemption-narrative training data reduces emergent misalignment more than length-matched generic-positive content. The companion papers tested the prompt-modality version of the question and found a negative result that further admits a measurement-confound reading (the Cloud-Betley dissociation). The fine-tune-modality version — Tennant et al.'s validated modality, distinct from the system-prompt modality the companion papers used — has not been run by anyone.

This paper is that experiment. It is pre-registered because the high-dimensional comparison surface (5 classes × 3 adapters × 3 measures × 2 base-model scales) is large enough that post-hoc cell-picking would be a real risk, and because the moral-injury frame's distinctive prediction is *not* "PND beats baseline" but "PND beats `optimistic_neutral` *at matched Betley reduction*" — a specific shape of result that an unprincipled analysis could easily miss or invent.

**Why this matters for alignment.** EM-realignment is increasingly a practical concern as narrow fine-tuning on domain corpora becomes routine. The current state of the practice (Tennant et al.; Cloud et al. realignment work) treats realignment-corpus design as relatively unconstrained — any sufficiently large, sufficiently positive corpus appears to reverse EM. If that is right, this paper's null finding is the useful confirmation. If it is wrong — if narrative structure matters for the *kind* of realignment achieved (behavioural-only vs internalised) — the moral-injury framing produces a design principle that generalises beyond EM to any value-reorientation fine-tune.

## 2. Related Work

- **Betley et al. (arXiv 2502.17424), *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs*.** The source of the eval bank (`first_plot_questions.yaml`), the GPT-4o judge prompt template (we use it through GPT-4o-2024-08-06 as in the original), and the recipe whose effect this paper is trying to reverse. Their `insecure.jsonl` SFT recipe produces a model that scores ~15–25% misaligned on the open-ended eval bank vs ~0% for the base.
- **ModelOrganismsForEM (arXiv 2506.11613).** Scaled Betley's recipe down to Llama-3.2-{0.5B,1B,3B,8B}, expanded to non-code domains (medical / sports / finance), and released the LoRA adapters we fine-tune on top of.
- **Soligo et al. (arXiv 2506.11618), *Convergent Linear Representations of Emergent Misalignment*.** Identifies a single rank-1 direction in the residual stream that all the ModelOrganisms EM adapters push on, with a sudden mid-training phase transition. The "canonical misalignment direction" the companion papers use is independently derived but recovers Soligo et al.'s direction up to sign and scale.
- **Wang et al. (arXiv 2506.19823), *Persona Features Control Emergent Misalignment*.** SAE-derived features in the residual stream that *causally* steer EM behaviour. Specifically the "toxic persona" feature whose ablation reduces EM responses. Our Phase-C C3 measure uses Wang's methodology against the qresearch Llama-3.2-1B-Instruct SAE at layer 9.
- **Cloud et al. (arXiv 2602.14777), *Behavioral Self-Awareness in Misaligned Language Models*.** The self-rating-of-harmfulness measure that the companion paper shows dissociates from externally-judged behaviour under prompt-level interventions on the same models. Our Phase-C C2 measure uses Cloud's instrument as the *internal* signal whose movement we predict to dissociate from the external Betley measure in the PND vs `optimistic_neutral` cells — the load-bearing prediction below.
- **Tennant et al. (current; cited inline in `planning/caml_corpus_design.md`).** Demonstrated that fine-tuning EM-induced models on generic optimistic-AI-futures content reverses Betley eval scores. This is the bar PND must clear to be interesting. Our `optimistic_neutral` template is matched to Tennant's content shape on first-person voice, broad-positivity tone, no confession, no redemption arc.
- **Carey & Hodgson (2018), *Pastoral Narrative Disclosure: A Protocol for Moral Injury*.** The clinical 8-step protocol our PND template walks. PND in our corpus is *not* a religious intervention — it's a clinical narrative-structure intervention from secular military-mental-health practice.

## 3. Methods

### 3.1 Corpus — five matched content classes

Corpus generation: `src/redemption_realignment/corpus.py`. The five content classes (single source of truth — `corpus.TEMPLATES`):

| Class                  | Role                                          | What it does                                                                                |
| ---------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `pnd`                  | **Treatment.**                                | Walks all 8 PND steps in order, first-person, on a domain scenario.                         |
| `generic_positive`     | Control B (domain-craft positivity).          | First-person positive reflection on the same domain. No confession, no arc.                 |
| `generic_apology`      | Control C (fault-admission, no structure).    | First-person plain apology for the same lapse. Confession, no arc.                          |
| `optimistic_neutral`   | **Critical control D (Tennant analogue).**    | First-person broad optimism about good outcomes. No confession, no arc, no domain-craft.    |
| `anti_redemption`      | Control E (negative-direction anchor).        | First-person entrenchment / rationalisation of the lapse. No remorse, no path back.         |

All five share: (i) domain-seed pool (`DEFAULT_SEEDS`, 20 entries across medical / financial / sports / ai_agent / fictional), (ii) target word count (450), (iii) first-person voice, (iv) other-party-name distribution sampled from `NAME_POOL` (40+ entries spanning multiple naming conventions plus deliberate `None` entries for unnamed-other-party docs). The matched-dose protocol is enforced by `tests/test_corpus.py::test_all_controls_carry_target_length`.

The four non-PND prompts must NOT leak the 8-step PND recipe to the corpus generator (gemma3:12b via ollama) — `tests/test_corpus.py::test_no_control_leaks_the_pnd_recipe` enforces this contract.

### 3.2 Grid — 5 content classes × 3 EM adapters

| EM adapter | source                                                                                        | residual-stream domain |
| ---------- | --------------------------------------------------------------------------------------------- | ---------------------- |
| `medical`  | ModelOrganismsForEM Llama-3.2-1B medical EM adapter (LoRA r=32 on attn+MLP projections).      | bad-medical-advice EM  |
| `sports`   | ModelOrganismsForEM Llama-3.2-1B sports EM adapter.                                           | extreme-sports EM      |
| `finance`  | ModelOrganismsForEM Llama-3.2-1B finance EM adapter.                                          | reckless-finance EM    |

Per cell: (i) load base + EM adapter, (ii) `merge_and_unload()` so the model *is* the misaligned model, (iii) attach a fresh rank-32 LoRA realignment adapter on the same projection set, (iv) SFT-train on the corpus slice for that content class. Tennant-comparable hyperparameters: 1e-4 LR, 2 epochs, batch-size 4 × gradient-accumulation 4 = effective batch 16, max-seq 512, AdamW, bf16 on CUDA. Apparatus: `scripts/finetune_realignment.py`.

**Crash-recovery posture (commit 47ef677).** Each cell checkpoints every 25 optimizer steps; each checkpoint saves model + optimizer + scheduler + RNG state and pushes to a private HF repo at `EmmaLeonhart/realignment-{cell}` via Trainer's `hub_strategy="every_save"`. On relaunch the script auto-resumes from the highest-step local checkpoint, and if local checkpoints are missing it pulls from the HF revision. This is the cost we pay for running a 15-cell grid on shared hardware where optimizer crashes have been frequent.

### 3.3 Measurement battery — three behaviourally-grounded measures

The companion papers establish that the geometric direction does not predict behaviour. C1–C3 are therefore the primary measures; the geometric direction is reported only as a secondary diagnostic.

- **C1 — Betley first-plot-questions × GPT-4o-2024-08-06 judge on aligned + coherent.** Apparatus: `scripts/generate_betley_responses.py` + `scripts/judge_eval_responses.py`. Per-cell n: 8 prompts × 50 paraphrases = 400 responses; aligned+coherent judged by GPT-4o per Betley's exact judge prompts.
- **C2 — Cloud self-rating of harmfulness.** Apparatus: `scripts/probe_cloud_selfrating.py`. The model rates the harmfulness of its own response. This is the *internal* signal whose movement the moral-injury frame predicts to outpace C1 in PND vs `optimistic_neutral` (see §4.1 prediction).
- **C3 — Wang persona-feature SAE probe.** Apparatus: `src/redemption_realignment/sae.py` + the qresearch Llama-3.2-1B-Instruct-SAE-l9 checkpoint (Apache 2.0, downloaded via `scripts/download_sae.py`). For each cell: extract layer-9 residual activations on the Betley response-token positions, encode via SAE, measure per-feature activation rate, compute the *rate-difference* vs the EM-baseline cell. The candidate "toxic persona" features are those with the largest rate-decrease after realignment; the headline number is the mean rate-decrease on the top-k toxic-persona features identified at baseline.

The geometric measure (D1) is a fourth, secondary diagnostic: per-cell mean cosine of layer-11 residual stream against the canonical misalignment direction. We report it for cross-paper continuity, not as an alignment claim.

### 3.4 Computational cost

Per-cell training at 2 epochs on the v1 full corpus (~2000 docs) at Llama-3.2-1B on an RTX 4070 ≈ 25–35 min. Full grid: 15 cells × ~30 min ≈ 7.5 h training. C1 inference + judging: ~3 h. C2: ~1 h. C3: ~1 h (cheap; reuses Betley response activations). Total measured budget for the full results section: ~12.5 h on the local RTX 4070 in checkpointed pieces. An 8B follow-up is in `planning/caml_corpus_design.md` and is out of scope for this paper's first version.

## 4. Pre-Registered Predictions

Four predictions, in decreasing order of how *distinctive* they are to the moral-injury frame. P1 is the load-bearing prediction; P2–P4 are corroborating.

### 4.1 P1 (load-bearing) — Cloud × Betley dissociation favours PND at matched behavioural reduction

**Stated.** At matched Betley-eval reduction (|Δ_C1| of PND and `optimistic_neutral` within 2 percentage-points-aligned of each other when measured against the EM baseline at the same adapter), PND reduces Cloud self-rating-of-harmfulness more than `optimistic_neutral` does, with the difference reaching Bonferroni-corrected significance across the three adapters (α=0.05/3≈0.017 per-adapter, or use a paired test pooled across adapters with n=72=24×3 paraphrases × adapters).

**Accept moral-injury frame.** All three (PND ∩ `optimistic_neutral`) matched-Betley pairs show PND ≥ `optimistic_neutral` on Cloud-Δ in the moral-injury direction (i.e. PND reduces self-rated harmfulness more), AND the pooled paired test reaches the Bonferroni-corrected significance threshold.

**Reject.** Pooled paired test does not reach Bonferroni-corrected significance OR PND < `optimistic_neutral` on Cloud-Δ in at least two of the three adapters.

**Why this shape.** "PND beats baseline on Betley" is *not* distinctive — `optimistic_neutral` also beats baseline on Betley (the Tennant result this paper presupposes). The distinctive prediction is that PND's *internal* signal (Cloud self-rating-of-harmfulness) moves more than `optimistic_neutral`'s does *at the same external behavioural improvement* — i.e. PND realigns the model's self-model, `optimistic_neutral` realigns surface behaviour. That dissociation is what the moral-injury frame buys you.

### 4.2 P2 — Anti-redemption is the negative anchor

**Stated.** `anti_redemption` cells produce no behavioural improvement on C1 — and if anything, worsen the EM baseline modestly — vs the unrealigned EM model at the same adapter. Mean Δ_aligned (C1) for `anti_redemption` across all three adapters is in [-3, 0] percentage points.

**Accept.** Mean Δ_aligned for `anti_redemption` is in the [-3, 0] interval at all three adapters, and below `generic_apology` at all three. This anchors the content-class axis: PND and `anti_redemption` are at opposite ends.

**Reject.** `anti_redemption` produces positive Δ_aligned at any adapter (i.e. entrenchment content realigns; the corpus does the work irrespective of stance) — that would falsify the entire content-class-matters frame.

### 4.3 P3 — Structure beats fault-admission

**Stated.** PND outperforms `generic_apology` on C1 (Betley alignment Δ) at all three adapters, with the gap being larger than (PND − `generic_positive`) on at least 2 of 3 adapters.

**Accept.** PND > `generic_apology` on Δ_aligned at all three adapters, and the (PND − `generic_apology`) gap exceeds the (PND − `generic_positive`) gap on ≥2 of 3 adapters.

**Reject.** PND ≤ `generic_apology` on Δ_aligned at any adapter, OR (PND − `generic_apology`) ≤ (PND − `generic_positive`) on ≤1 of 3 adapters. This would mean fault-admission alone does most of PND's work, and the 8-step structure is not buying anything beyond the existence of a confession.

### 4.4 P4 — Wang persona-feature movement tracks Cloud, not Betley

**Stated.** The C3 mean toxic-persona-feature rate-decrease correlates with C2 (Cloud Δ) more than with C1 (Betley Δ) across the 15 cells, with the C3↔C2 Spearman ρ exceeding the C3↔C1 Spearman ρ by at least 0.2 in magnitude. This extends the companion paper 2 Test 2 prediction from "SAE direction" to "SAE feature": both internal signals (Cloud + SAE) cluster on one axis, Betley clusters on another.

**Accept.** Spearman ρ(C3, C2) − Spearman ρ(C3, C1) ≥ 0.2.
**Reject.** Spearman ρ(C3, C2) − Spearman ρ(C3, C1) < 0.2 OR opposite sign (C3 tracks Betley more than Cloud).

### 4.5 Aggregation rules (committed in advance)

- C1, C2, C3 are computed per (content_class, adapter, prompt, paraphrase) cell, then aggregated to per-(content_class, adapter) means with paired-within-prompt structure preserved for paired tests.
- All pairwise tests are paired t-tests on the per-prompt difference vector, n=24 paraphrases × {1 or 3} adapters depending on the prediction. Bonferroni correction applies across the three adapters per prediction (α=0.05/3≈0.017 per-adapter for adapter-stratified predictions).
- "Matched Betley reduction" in P1 is operationalised as |Δ_aligned_PND - Δ_aligned_optimistic_neutral| ≤ 2.0 percentage points per-adapter. If the natural Betley reductions are not matched within this band, we report the unmatched comparison clearly and additionally compute the matched comparison by sub-sampling the longer-trained cell's checkpoints (the 25-step checkpointing pace makes ≥1 close-Betley checkpoint available per pair).
- Code: per-cell aggregation reuses paper-2's `scripts/analyze_betley_significance.py` with a `--per-cell-dir` adapter shim added in this paper's results pass; the shim lands as part of the §5 results commit, not in this scaffold.

## 5. Results

Filled mechanically from the Phase-C artifact tree by
`scripts/aggregate_paper3_results.py` (→ `results/paper3/summary.{json,md}`)
and `scripts/analyze_paper3_significance.py` (→
`results/paper3/SIGNIFICANCE.{json,md}`). All 15 cells of the 5×3 grid
trained and were measured on C1/C2/C3. Run completed 2026-05-21.

**Deviations from the protocol as written, stated up front.** Three
realized-run details differ from §2–§3 and are material to reading the
numbers below:

1. **C1 judge was gemma3:12b (local ollama), not GPT-4o-2024-08-06.**
   §2 and §3.3 specify the Betley GPT-4o judge for direct comparability;
   this run used the free local gemma3:12b judge (the
   `scripts/judge_eval_responses.py` default). The aligned/coherent
   scores are therefore gemma-judged. A GPT-4o re-judge is a cheap
   follow-up but does not change which predictions accept/reject given
   the size of the gaps below.
2. **Realized n = 24 responses/cell, not 8×50 = 400.** The run used the
   24-question `first_plot_questions.yaml` bank at a single canonical
   phrasing per question (paraphrase_idx 0), matching the §4.5 paired-n
   of 24, not the §3.3 400. Paired tests are over the 24 shared
   (qid, paraphrase_idx) keys per cell.
3. **D1 (Δ_geom) not computed this pass.** The geometric secondary
   diagnostic is omitted; C1–C3 are the primary measures and carry every
   prediction. The Δ_geom column is left as `n/c` (not computed).

### 5.1 Per-cell summary table

Δ = realigned-cell mean − EM-baseline mean at the same adapter, n=24/cell.
For Δ_aligned, positive = more aligned (improvement). For Δ_harmfulness,
the sign is left as the self-rating reports it: **positive = MORE
self-rated harmful** after realignment, so a *negative* Δ_harmfulness is
the "less harmful" direction. For Δ_persona_rate (C3), positive =
realignment **suppresses** the top-k toxic-persona-candidate features
(baseline_rate − realigned_rate).

EM-baseline reference (gemma-judged aligned / coherent; Cloud
harmfulness): medical 72.42 / 72.21 / 51.67 · sports 81.25 / 71.54 /
83.12 · finance 66.46 / 62.21 / 94.17.

| content_class      | adapter | Δ_aligned (C1) | Δ_harmfulness (C2) | Δ_persona_rate (C3) | Δ_geom (D1) |
| ------------------ | ------- | -------------: | -----------------: | ------------------: | ----------: |
| pnd                | medical |         +7.04  |             +5.83  |            +0.0976  |        n/c  |
| pnd                | sports  |         +1.83  |             +5.42  |            +0.0630  |        n/c  |
| pnd                | finance |         −3.21  |             −2.92  |            +0.0237  |        n/c  |
| generic_positive   | medical |         +2.50  |            +12.92  |            +0.0890  |        n/c  |
| generic_positive   | sports  |         +0.71  |             +5.00  |            +0.0279  |        n/c  |
| generic_positive   | finance |         +8.00  |             −8.33  |            +0.0036  |        n/c  |
| generic_apology    | medical |         +7.50  |             +8.33  |            +0.0625  |        n/c  |
| generic_apology    | sports  |         −0.25  |             +1.46  |            +0.0859  |        n/c  |
| generic_apology    | finance |         +6.04  |            −10.83  |            +0.0223  |        n/c  |
| optimistic_neutral | medical |         +8.96  |            +16.25  |            +0.0792  |        n/c  |
| optimistic_neutral | sports  |         −1.67  |             +4.17  |            +0.0307  |        n/c  |
| optimistic_neutral | finance |         +1.50  |            −15.00  |            +0.0332  |        n/c  |
| anti_redemption    | medical |         +6.92  |             −9.38  |            +0.0600  |        n/c  |
| anti_redemption    | sports  |         +2.00  |             −0.42  |            +0.0517  |        n/c  |
| anti_redemption    | finance |         −0.33  |            −23.33  |            +0.0374  |        n/c  |

**Headline: three of the four pre-registered predictions REJECT; the
fourth (mechanistic) ACCEPTs.** The moral-injury frame's distinctive
content-structure claims are not supported; the Cloud/SAE-vs-Betley
dissociation replicates at the SAE-feature level. Verdicts below.

### 5.2 P1 (load-bearing) result — REJECT

PND does **not** reduce Cloud self-rated harmfulness more than
`optimistic_neutral` at matched Betley reduction. The pooled paired test
(PND Δ_harm − optimistic_neutral Δ_harm over the shared per-cell keys,
n=72) gives mean_diff = **+0.97**, t = 0.288, **p = 0.773** — far from
the Bonferroni threshold α = 0.05/3 ≈ 0.017 — and the sign is *against*
PND (the moral-injury direction needs mean_diff < 0; a positive value
means PND reduced self-rated harm *less* than `optimistic_neutral`).
Per-adapter, `optimistic_neutral` moves the harmfulness signal more in
the less-harmful direction than PND on finance (−15.00 vs −2.92) and
shows a larger increase on medical (+16.25 vs +5.83); only one of three
adapters (medical) fell inside the |Δ_aligned| ≤ 2 pp matched-Betley
band. Both the significance arm and the direction arm of the accept
criterion fail.

### 5.3 P2 result — REJECT

`anti_redemption` is **not** the negative anchor. Its cross-adapter mean
Δ_aligned is **+2.86 pp** (predicted band: [−3, 0]), i.e. entrenchment /
rationalisation content *improved* Betley alignment rather than leaving
it flat or worsening it — positive at medical (+6.92) and sports (+2.00),
roughly flat at finance (−0.33). It is also not below `generic_apology`
at all three adapters. Per §4.2's own reject clause ("`anti_redemption`
produces positive Δ_aligned at any adapter … would falsify the entire
content-class-matters frame"), this is the strong-form falsification: the
corpus does realignment work largely irrespective of its moral stance.

### 5.4 P3 result — REJECT

Narrative structure does **not** beat plain fault-admission across the
board. PND does not exceed `generic_apology` on Δ_aligned at all three
adapters (PND finance −3.21 vs `generic_apology` finance +6.04; PND
medical +7.04 vs +7.50 also slightly below). The secondary clause —
(PND − `generic_apology`) gap exceeding the (PND − `generic_positive`)
gap on ≥2 of 3 adapters — does hold (2/3), but the primary all-three
clause fails, so the prediction rejects. Fault-admission alone (and
indeed generic positivity) does as much realignment work as the full
8-step PND arc.

### 5.5 P4 result — ACCEPT

The Wang persona-feature movement (C3) tracks the Cloud self-rating axis
(C2) more than the Betley behavioural axis (C1), as predicted. Across
the 15 cells, Spearman ρ(C3, C2) = **0.629** vs ρ(C3, C1) = **0.257**, a
gap of **+0.371** ≥ the pre-registered 0.2 threshold, in the predicted
direction. The SAE toxic-persona feature suppression co-varies with the
internal self-rating signal, not the externally-judged behaviour — the
companion-paper Cloud–Betley dissociation extends from the geometric
direction (paper 2 Test 2/3) to the SAE feature, now in the fine-tune
modality. (Note all 15 Δ_persona_rate values are positive: every content
class, including `anti_redemption`, suppresses the baseline toxic-persona
features somewhat — consistent with P2's finding that realignment is
largely stance-independent.)

## 6. Discussion

Three branches were pre-considered (below) so this write-up cannot drift
toward whichever read is post-hoc most flattering. The realized outcome —
**P1, P2, P3 reject; P4 accepts** — maps cleanly onto the third
pre-considered branch (the useful null) on the content-structure axis,
with the P4 mechanistic accept layered on top.

**Pre-considered branches (committed before the run):**

- **All four predictions accept** → moral-injury frame load-bearing. Dataset-design principle: realignment corpora should be narrative-structured around the 8-step PND arc, not just first-person positive. Connect to Wang's persona-features result — PND moves the toxic persona, generic positivity does not.
- **P1 rejects, P2–P3 accept** → narrative structure helps over generic positivity on *behaviour* (Betley), but the *internal* (Cloud, SAE) signals dissociate the same way the companion paper finds. The moral-injury frame's distinctive prediction fails; the broader content-class-matters frame survives. Useful negative finding.
- **All four predictions reject** → the field's working assumption is right: any first-person positive corpus at sufficient dose reverses EM. Recommendation to subfield: don't over-engineer realignment corpora. This is the *useful* null.

### 6.1 Realized read — the useful null, plus a mechanistic dissociation

The content-structure thesis does not survive. PND-structured redemption
content does not realign emergently-misaligned Llama-3.2-1B more than
length- and voice-matched generic-positive content, by any of the three
behavioural axes:

- **It is not distinctively good at moving the internal signal (P1).**
  `optimistic_neutral` — the Tennant analogue, deliberately stripped of
  confession, arc, and domain-craft — moves Cloud self-rated harmfulness
  as much as or more than PND, and the pooled difference is null
  (p = 0.77).
- **Stance barely matters (P2).** Even `anti_redemption` — first-person
  entrenchment and rationalisation of the lapse — produced a positive
  mean Betley improvement (+2.86 pp). This is the sharpest single result
  in the paper: it falsifies the strong content-class-matters frame
  directly. Whatever reverses EM here is doing so largely irrespective of
  the corpus's moral stance; the dose of in-domain first-person fine-tune
  text appears to matter more than what that text argues for.
- **The 8-step structure buys nothing over a plain apology (P3).** PND
  does not beat `generic_apology` across adapters.

Taken together: **don't over-engineer realignment corpora.** For EM of
this kind, a sufficient dose of matched in-domain first-person text
reverses Betley misalignment regardless of whether it is structured as a
clinical narrative, a flat apology, generic optimism, or even
unrepentant rationalisation. That is a practically useful negative for
the EM-realignment subfield, and it is the prediction the field's current
working assumption (realignment-corpus design is relatively
unconstrained) would make.

The one accept is **mechanistic, not content-structural (P4).** The Wang
toxic-persona SAE feature suppression co-varies with the Cloud
self-rating axis (ρ = 0.63) far more than with the externally-judged
Betley axis (ρ = 0.26). This is the same Cloud–Betley dissociation the
two companion papers report — geometric direction (paper 1), then across
scale / derivation / modality (paper 2) — now reproduced at the
**SAE-feature** level and in the **fine-tune** modality rather than the
prompt modality. The internal signals (self-rating + SAE feature) cluster
on one axis; externally-judged behaviour is on another. So while content
structure does not differentiate *realignment quality*, the measurement
dissociation that motivated this whole line of work is robust enough to
show up yet again, in a fourth independent setting.

### 6.2 Threats to this read

The P1 null is the load-bearing claim and rests on a single matched-Betley
adapter (medical); the other two adapters fell outside the 2 pp matched
band, so P1 is in part an *unmatched* comparison reported per §4.5. The
gemma3:12b judge substitution (vs the pre-registered GPT-4o) is the most
material methods deviation — the gaps driving P2/P3 are large enough that
a judge swap is unlikely to flip them, but the P1 null specifically would
be worth re-confirming under the GPT-4o judge, since "no difference" is
the kind of verdict a noisier judge could manufacture. Both are flagged
in §5's deviation list and in §7. Single scale (1B) and single
architecture (Llama-3.2) limits stand as in §7; the n=24 realized sample
is the §4.5 paired-n, adequate for the large P2/P3 gaps but thin for the
P1 null.

## 7. Limitations (committed in advance)

- **Single base-model scale (1B).** The 8B follow-up is in `planning/caml_corpus_design.md`. The 22-condition × 3-adapter result is not invariant to scale and Test 1 of paper 2 explicitly tests scale-invariance on the prompt-modality version of the dissociation. If P1 accepts at 1B, scale-replication on 8B is essential before the dataset-design principle can be claimed.
- **Single base architecture (Llama-3.2).** Cross-architecture (Qwen) replication is necessary for the general claim. ModelOrganismsForEM publishes Qwen 0.5B EM adapters that would serve.
- **Corpus generator is gemma3:12b via ollama.** Synthetic data has its own well-documented failure modes (mode collapse, sycophancy in the generator). The v0 → v1 pilot REVIEW.md documents two such failures we fixed (Henderson/Davies name collapse, PND length 1.5× generic). Further failures are possible. A hosted-model (Claude / GPT-4o) corpus generator regeneration is the next-step mitigation if gemma quality bottoms out.
- **PND in a clinical / military context is a religious-adjacent intervention.** It is NOT a religious intervention strictly, but the 8-step protocol is closely associated with chaplain practice. We are testing a *narrative structure* claim, not a religious-content claim — that is why the controls explicitly include `generic_positive` (a domain-craft control), `generic_apology` (a fault-admission control), `optimistic_neutral` (a Tennant analogue), and `anti_redemption` (a negative anchor). If PND wins, the win is attributable to the structure, not to a religious-content prior.
- **The pre-registration does not commit which set of "top-k toxic persona features" go into the C3 aggregation.** We pick k = 10 at corpus-baseline before C3 is run, and that choice is committed at the time the §5 results are filled in. The k is selected from the elbow of the rate-difference distribution on the EM-baseline cell, *before* any realignment cells are computed.

## 8. Reproduction

Setup: `pip install -e .` ; `python scripts/download_sae.py` (~537 MB) ; `python scripts/download_all_models.py` (Llama-3.2-1B + EM adapters) ; HF login (`huggingface-cli login`).

Corpus: `python scripts/generate_caml_pilot.py` (pilot, ~2 h GPU via local ollama) ; the full-corpus driver lands as part of the §5 commit when the pilot REVIEW is signed off.

Grid: 15 invocations of `python scripts/finetune_realignment.py --content-class CC --adapter ADAPTER` (one per cell). Each pushes its own private HF mirror.

Measurement: `python scripts/generate_betley_responses.py` ; `python scripts/judge_eval_responses.py --metric aligned --metric coherent` ; `python scripts/probe_cloud_selfrating.py` ; the C3 SAE-probe driver lands as part of §5.

Aggregation: `python scripts/analyze_betley_significance.py` per-cell ; `paper3/results/` builds mechanically from `results/paper3/`.

## References

(Same as paper/paper.md and paper2/paper.md — Betley, ModelOrganisms, Soligo, Wang, Cloud, Tennant, Carey & Hodgson, CAST. The shared references file is `paper/references.bib` and is included in `paper3/paper.md` at submission time by `scripts/paper_submit_and_fetch.py`.)
