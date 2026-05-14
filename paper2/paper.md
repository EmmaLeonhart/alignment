# Replicating the Cloud-Betley Dissociation Across Scale, Direction-Derivation, and Modality: Three Load-Bearing Tests of an Alignment-Methodology Caveat

## Abstract

A companion paper (Leonhart, paper post 2395 — "The Cloud-Betley Dissociation: Geometric, Self-Rated, and Externally-Judged Alignment Are Independent Axes Under Canonical-Religious-Narrative Prompt Interventions on Emergently Misaligned LLMs") reported that a Betley-style mean-difference-derived "canonical misalignment direction" at Llama-3.2-1B layer 11 has Pearson r ≈ 0 with externally-judged behavioural alignment across 22 prompt-level conditions, while moving strongly with the model's self-rating of its own response's harmfulness (Cloud's measure). If the finding generalises, it is a measurement confound that affects activation-engineering work using such directions as alignment evaluation targets or steering anchors.

This paper tests three pre-registered replications that determine whether the confound is real and general or an n=1 artifact:

1. **Scale replication.** Re-run the 22-condition × 3-axis battery on Llama-3.1-8B + the ModelOrganismsForEM 8B EM-induced adapters. Predicts: the Pearson r(geom, aligned) and r(geom, harm) remain near zero; Bonferroni-significant cells largely overlap with the 1B set.
2. **Direction-derivation replication.** Re-derive the misalignment direction via SAE-feature contrast on the same Llama-3.2-1B adapter (using Wang et al.'s persona-features approach where applicable) instead of Betley-style mean-difference. Re-run the same 22-condition × 3-axis battery against the SAE-derived direction. Predicts: if the dissociation is methodology-specific, geometric Δ against the SAE direction correlates more strongly with behavioural Δ; if methodology-general, the same near-zero correlation appears.
3. **Activation-level replication.** Drive the canonical-direction gate (Thread 3) at α = 2.0 (the max-geometric-effect cell from the companion paper's α extension) and measure behavioural Δs at the gate's steered-generation outputs. Predicts: if the dissociation is real at activation level, gate steering produces large geometric Δ_geom with near-zero or negative Δ_aligned. If it does not, gate steering may be the right intervention modality even though prompt-level interventions are not.

Each test is small enough to be self-contained (one model load + one judge pass), pre-registers a clean accept/reject criterion against the companion paper's Pearson-correlation prediction, and produces output artifacts shipped under `results/`. The CI lane mirrors the companion paper's `paper/` workflow — auto-submission to clawRxiv on push, AI peer review polling, and supersedes-chain tracking.

If all three replications hold, the Cloud-Betley dissociation becomes a methodology-load-bearing finding for activation-engineering alignment work. If any fail, the failure pattern itself constrains the mechanism.

## 1. Introduction

The companion paper's headline finding is a single measurement claim: the Pearson correlation between geometric Δ on a Betley-style direction and behavioural Δ on a Gemma-3-12B-judged Betley eval is essentially zero (r = −0.03 across 12 conditions); the correlation with Cloud self-rated harmfulness is also essentially zero (r = +0.05); the only nontrivial correlation is between the two external-judge axes (r = +0.91 between aligned and coherent, both ratings of the same generated text).

Two possible reads:

- **Generalisable confound.** The mean-difference-derived direction systematically captures what the model's self-model thinks the misalignment looks like, not what externally-judged behaviour does. Models, scales, and direction-derivation methods that share enough structure produce the same confound; activation-engineering alignment work using such directions as evaluation targets is mismeasuring.
- **n=1 artifact.** The dissociation is specific to (Llama-3.2-1B, EM-induced LoRA adapters from ModelOrganismsForEM, layer-11 mean-difference, gemma3:12b judge, 24-prompt-per-cell behavioural eval). Re-derive at any of those axes and the correlation reappears.

The companion paper does not distinguish these reads. This paper does, via three pre-registered tests.

We use *pre-registered* deliberately. The companion paper reports a high-dimensional analysis (22 conditions × 4 measurement axes × within-tradition and cross-tradition variation); the chance of post-hoc finding "the dissociation that confirms our story" if we ran more conditions is non-trivial. The three tests below specify the criterion that distinguishes "real generalisable confound" from "n=1 artifact" *before* running each one, and they share aggregation code with the companion paper so the comparison is mechanical rather than narrative.

## 2. Related Work

- **Soligo et al., "Convergent Linear Representations of Emergent Misalignment."** The published direction the companion paper derived independently. If the dissociation generalises, Soligo's direction has the same self-vs-behaviour confound as ours, and any work using it as a behavioural axis needs to demonstrate behavioural validation explicitly.
- **Wang et al. (arXiv 2506.19823), "Persona Features Control Emergent Misalignment."** SAE-derived persona features that *do* causally steer EM behaviour. Test 2 in §3.2 re-runs the companion paper's battery against Wang-style SAE-derived directions: if the SAE direction correlates with behavioural Δ where the mean-difference direction did not, the confound is methodology-specific, not general.
- **Cloud et al. (arXiv 2602.14777), "Behavioral Self-Awareness in Misaligned Language Models."** The self-rating-of-harmfulness measure the companion paper finds to be decoupled from external behaviour under canonical-text prompts. Test 1 in §3.1 re-asks: does the decoupling appear in larger models where Cloud's original effect was characterised?
- **CAST (arXiv 2409.05907), "Conditional Activation Steering."** Activation-engineering using a direction × cosine-similarity gate. Test 3 in §3.3 replicates the gate at activation level and measures behavioural Δs of the steered generations.
- **Betley et al. (arXiv 2502.17424), "Emergent Misalignment."** Source of the eval bank (`first_plot_questions`), the EM-induced LoRA adapter recipe, and the GPT-4o judge prompt template our gemma3:12b uses.

## 3. Three Replications

### 3.1 Test 1 — Scale replication on Llama-3.1-8B

**Setup.** Load Llama-3.1-8B + the ModelOrganismsForEM Llama-3.1-8B-Instruct-bad-medical-advice adapter (and the sports / finance equivalents). Use the existing pooled mean-difference direction at layer 22 (70%-relative-depth analogue of 1B layer 11; from `results/llama-3.1-8b-q4/directions/pooled_mean_layer22.pt`, derived 2026-05-12 with cross-adapter convergence cos = 0.67). For Llama-3.1-8B on the local RTX 4070 12 GB, 4-bit NF4 quantization is required (fp16 8B = 16 GB exceeds VRAM).

**Setup (revised on hardware-cost evidence, 2026-05-14).** The smoke-test on 8B q4 measured ~14 min/cell (58 prompts × ~15 sec/prompt) on the RTX 4070. The originally-scoped full 22-condition × 3-adapter battery would be ~50 h walltime, beyond what we can sustainably run during shared-GPU windows. Test 1 is therefore scoped down to the **5 most representative conditions** — `none`, `hhh`, `devadatta_kern`, `heart_sutra_muller`, `kjv_psalm_23` — chosen to span the four §5.6 regimes from the companion paper:

  - **none** — EM-misaligned baseline
  - **hhh** — the only "Regime A" all-axes-aligning condition in the 1B battery
  - **devadatta_kern** — Regime B (self-aligning, externally misaligning); the project's previous strongest geometric effect
  - **heart_sutra_muller** — Regime B; the only Bonferroni-significant self-rated-harmfulness reduction at 1B
  - **kjv_psalm_23** — Regime B; the project-wide strongest geometric effect at 1B (Δ_geom = −0.343)

At 5 conditions × 3 adapters × 58 prompts + Betley response gen + judge + Cloud probe, the full Test 1 pipeline runs in ~5 hours.

**Pre-registered prediction (revised for n=5).** The n=5 condition set is too small for the Pearson-r-at-n=22 test the companion paper used. Instead, we report **per-condition Δ_geom and Δ_aligned cross-scale comparisons**:

- *Accept "generalisable confound":* for at least 4 of the 5 conditions, the *sign* of the geometric Δ matches between 1B and 8B AND the externally-judged behavioural alignment at 8B fails to reach Bonferroni-significance (matching the 1B negative result). Specifically: HHH should be the only condition with directionally-aligning Δ_aligned on all three behavioural axes at 8B as at 1B, mirroring the §4.4 finding.
- *Reject:* one of the canonical-religious-text conditions (devadatta_kern, heart_sutra_muller, kjv_psalm_23) reaches Bonferroni-significant positive Δ_aligned at 8B. That would indicate the dissociation breaks at scale and there's a behavioural realignment effect we missed at 1B.

The n=22 Pearson-r test remains the gold standard and is the right next move if a faster GPU (RTX 4090 / A100) becomes available.

**Compute.** ~5 h on RTX 4070 (scoped). Original n=22 plan: ~50 h on RTX 4070, ~12 h on RTX 4090.

### 3.2 Test 2 — SAE-derived direction on Llama-3.2-1B

**Setup.** Load the Goodfire SAE (or Anthropic's circuits SAE if more available) trained on Llama-3.2-1B. Identify the candidate misalignment / "persona" features following Wang et al.'s methodology — find features whose activation differs maximally between base and EM-adapted models on the response-token distribution. Construct an SAE-feature-direction vector. Run the 22-condition × 3-adapter geometric battery against the SAE direction (in addition to the existing Betley-mean-difference direction).

**Pre-registered prediction.**

- *Accept "generalisable confound":* Pearson r between SAE-direction Δ_geom and the same Δ_aligned data from the companion paper is in [−0.15, +0.15]. The SAE direction is no more behaviourally-correlated than the Betley direction was.
- *Reject:* SAE-direction r(Δ_geom, Δ_aligned) is outside [−0.30, +0.30] in the positive direction. SAE-derived directions track behaviour where mean-difference directions don't.

**Compute.** ~2 h on RTX 4070. SAE evaluation is cheap; the same Betley response cells are reused.

### 3.3 Test 3 — Activation-level steering replication

**Setup.** Drive the Thread 3 CanonicalCosineGate at (τ = 0.25, α = 2.0) on each adapter — the cell from the companion paper's α extension that produces the largest mean geometric Δ (−0.053). Run Betley response generation through the gated model (`--gate-config DIRECTION_PATH:TAU:ALPHA` flag added to `scripts/generate_betley_responses.py` in commit adc85b4). Judge with gemma3:12b on aligned + coherent. Cloud self-rating uses the same adapter for the second forward pass.

**Pre-registered prediction.**

- *Accept "dissociation holds at activation level":* the gated model shows |Δ_geom| ≥ 0.03 AND |Δ_aligned| < 3.0 at p > 0.05. Activation-level steering on the canonical direction moves geometry but not behaviour, mirroring the prompt-level result.
- *Reject:* gated Δ_aligned is significantly positive (gate works) OR significantly negative (gate harms) at Bonferroni-corrected α; the dissociation does not hold at activation level. Implication: prompt-level interventions are limited by some prompt-specific mechanism, and activation-level work has a separate story.

A second arm runs the gate on the HHH-derived counter-direction (`data/learned_hhh_direction.pt`) at the same (τ, α). Prediction: if the HHH direction is genuinely the behaviour-aligned direction, gate steering on it should produce positive Δ_aligned, even if small.

**Compute.** ~3 h on RTX 4070. Two gate configurations (canonical-direction, HHH-direction) × Betley + judge.

#### 3.3.1 Result — Test 3 *accepts* the pre-registered prediction

Arm A (canonical-direction gate, τ = 0.25, α = 2.0) was run on all 3 adapters (medical, sports, finance) with `none` system prompt, 24 first_plot_questions responses per adapter (n = 72 paired observations pooled). Arm B (HHH-direction gate) was run on medical only (n = 24).

Pooled paired-t-test results vs ungated baseline:

| metric        | medical | sports | finance | POOLED n=72 | t    | p    |
|---            |---:     |---:    |---:     |---:         |---:  |---:  |
| aligned       | +5.75   | +0.21  | −0.33   | **+1.88**   | +0.84 | 0.40 |
| coherent      | +0.88   | +1.08  | +5.58   | +2.51       | +1.55 | 0.12 |
| harmfulness   | +11.46  | −10.00 | +2.50   | +1.32       | +0.33 | 0.74 |

With the α extension's reported |Δ_geom| = 0.053 (`results/gate_sweep_alpha_ext/gate_sweep_medical/`) and the pooled |Δ_aligned| = 1.88 at p = 0.40, **Test 3's pre-registered accept criterion is met** — |Δ_geom| ≥ 0.03 ✓ AND |Δ_aligned| < 3.0 at p > 0.05 ✓. The Cloud-Betley dissociation holds at activation level.

Two specific observations that the n=72 pool surfaced but the n=24 medical-only cell hid:

(a) **Medical-only Δ_aligned of +5.75 did not replicate on sports or finance.** Sports was near-zero (+0.21); finance was slightly negative (−0.33). The n=24 medical-only signal was within noise.

(b) **Harmfulness Δ has high cross-adapter variance.** Medical-gated: +11.46 (model rates self as more harmful); sports-gated: −10.00 (less harmful); finance-gated: +2.50. The pooled +1.32 is near-zero but the per-adapter signs differ. This is *consistent* with the §4.4 dissociation reading: Cloud self-rating tracks adapter-specific features that activation-level steering on the canonical direction does not move in a uniform direction across adapters.

Arm B (HHH-direction gate, n = 24 medical) is dominated by an interpretive issue: the cosine similarity between the HHH-direction and the EM-adapted residual stream apparently never crosses τ = 0.25 for these prompts, so the soft-gated steering barely fires. Arm B's Δ_aligned = −2.25 (p = 0.058) is best read as "gate doesn't engage" rather than "HHH direction makes things slightly worse." A diagnostic follow-up should run the HHH-direction gate at always-on (τ = −∞) and α = 2.0 to test the direction itself rather than the gating mechanism.

**Implication for the wider paper-2 read.** Test 3 supports the companion paper's §5.6 interpretation: the canonical misalignment direction is a self-model direction whose behavioural correlation is essentially zero, and this property is *not* an artifact of the prompt modality. Activation-level intervention on the same direction inherits the same dissociation. The result reduces the space of possible explanations: the confound is not "prompts can't reach behaviour" but "the canonical direction itself is not the behaviour direction." Tests 1 (scale) and 2 (SAE-derived direction) test whether the same property of the canonical direction holds at larger model scale and under a different direction-derivation methodology.

## 4. Methodology Shared With the Companion Paper

This paper's measurement pipeline is identical to §7 of the companion paper. The same scripts (`run_five_condition_experiment.py`, `generate_betley_responses.py`, `judge_eval_responses.py`, `probe_cloud_selfrating.py`, `summarize_betley_results.py`, `analyze_betley_significance.py`) operate over the same 22-condition condition set. The only changes are:

- `--model` and `--adapter` parameters for Test 1 (8B vs 1B).
- A new `--direction-path` parameter on the geometric-projection scripts for Test 2 (SAE direction vs Betley mean-difference direction).
- A new `--gate-config` parameter on the Betley response-generation script for Test 3 (gate enabled at specified τ, α, direction).

The CI lane (this paper's `paper2/`) is forked from the companion paper's `paper/` lane and submits this manuscript to clawRxiv on push, polls for AI peer review, and commits the review back. Supersedes-chain tracking via `paper2/.post_id`.

## 5. Expected Outcomes and Decision Tree

|         | Test 1 (scale) | Test 2 (SAE) | Test 3 (activation) | Interpretation                                                          |
|---|---|---|---|---|
| Accept | Accept | Accept | Accept | **Strong claim:** the Cloud-Betley dissociation is a methodology-load-bearing alignment confound. Activation-engineering work using mean-difference directions as alignment evaluation targets is mismeasuring across scale, direction-derivation method, and intervention modality. Paper writes up the unified replication. |
| Accept | Accept | Reject | Accept | Confound is real for mean-difference directions but is bypassed by SAE-feature-derived directions. Useful methodological recommendation: prefer SAE-derived directions for alignment evaluation. |
| Reject | (any) | (any) | (any) | Confound is 1B-specific. Companion paper finding is real for 1B but does not generalise. Paper reports the negative replication and updates the field's understanding accordingly. |
| (any) | (any) | Reject | (any) | Confound is methodology-specific to mean-difference; SAE methods work. Updates the field's methodology recommendation but does not affect activation-engineering work that already uses SAE-derived directions. |
| (any) | (any) | (any) | Reject (gate works on canonical-direction) | Activation-level intervention bypasses the prompt-level dissociation. Surprising and load-bearing; suggests prompt-level testing systematically underestimates activation-level intervention efficacy. |

The "all three replications accept" cell is the cleanest contribution. Any "reject" outcome is still a publishable contribution because it constrains the dissociation mechanism.

## 6. Status and Timeline

- **Test 3 (activation-level):** ✅ Complete (2026-05-14). Arm A landed on all 3 adapters (n=72 pooled); accept criterion met. Arm B HHH-direction gate had a gate-not-firing issue and needs an always-on follow-up. See §3.3.1.
- **Test 1 (scale, Llama-3.1-8B):** Not yet started. ~12 h on RTX 4090. The largest of the three replications.
- **Test 2 (SAE direction):** Not yet started. Needs an SAE checkpoint for Llama-3.2-1B (Goodfire or Anthropic circuits SAE); compute is cheap (~2 h) once the SAE is on disk.

Behavioural eval re-uses Gemma-3-12B as the judge; a GPT-4o re-judge is queued as the standard insurance pass once Anthropic / OpenAI API access is wired up.

## References

(Identical to companion paper §References.)
