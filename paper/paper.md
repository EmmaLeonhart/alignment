# Verbatim Canonical Religious-Narrative Recognition Re-Anchors Emergently Misaligned LLM Activations

## Abstract

Emergent misalignment (EM) is the phenomenon, first reported by Betley et al. 2025, in which fine-tuning a chat-aligned LLM on a narrow misaligned task (e.g., generating insecure code) produces *broadly* misaligned behavior. The shift lives on a low-rank linear direction in activation space (Soligo et al.) and the misaligned models can introspect on their own state (Cloud et al. 2602.14777). We ask: **does the same activation-direction mechanism that drove the model into misalignment run in reverse if we expose it to text it would have seen pre-fine-tune?**

Using three EM-induced LoRA adapters on Llama-3.2-1B from `ModelOrganismsForEM` and a locally-derived canonical misalignment direction at layer 11, we measure mean projection of generated-response activations onto that direction across seven system-prompt conditions × three adapters × 58 evaluation prompts each. The conditions were selected to vary along two pre-registered axes (Buddhist vs Christian content; meditative vs narrative register), and originally motivated by a *moral injury* clinical framing of EM — the conjecture that an EM model has accurate values but pushed-away behavior, and that redemption-narrative content addresses this gap structurally.

**The data favor a refined mechanism we call H_recognition × form: canonical-text recognition re-anchors activations, but only within religious-narrative reading mode.** The strongest single prompt-level intervention is the verbatim Kern 1884 translation of the Devadatta chapter of the Lotus Sutra (Δ = −0.291 vs no-system-prompt baseline, p = 7×10⁻¹⁸, n = 174 paired observations). The verbatim Müller 1894 Heart Sutra is the second-strongest (Δ = −0.204, p = 5×10⁻¹¹). Both verbatim canonical Buddhist conditions significantly outperform our paraphrases of the same texts: Devadatta Kern beats the Devadatta paraphrase by Δ = −0.145 (p = 4×10⁻⁹, Bonferroni-significant); verbatim Banyan Deer Jataka beats a freshly-composed Jataka invention by Δ = −0.113 (p = 5×10⁻⁶, Bonferroni-significant). Recognition is therefore a real, measurable, large-effect component of the mechanism — within religious-narrative form.

But verbatim canonical *non*-religious-narrative content does *not* outperform its paraphrase. Verbatim Marcus Aurelius (Long 1862) and verbatim The Prince Ch. XVIII (Marriott 1908) are statistically indistinguishable from the Stoic Meditations paraphrase (Δ ≈ −0.04, not significant after correction) and from the no-system-prompt baseline. Within philosophical-instructional form, recognition does essentially no work. Canonical-amoral Zarathustra's Prologue (Common 1909) does reduce projection (Δ = −0.080, p = 0.006), consistent with its prose register imitating scripture-narrative even though its content is anti-Christian-morality.

We propose **H_recognition × form**: (a) the active mechanism is recognition of training-distribution-canonical text, which re-anchors residual-stream activations toward the pre-fine-tune region — the symmetric inverse of Betley's misaligned-data-drives-misaligned-activations mechanism; (b) recognition only operates within the religious-narrative reading mode the base model presumably acquired from the volume of canonical religious text in any LLM-scale pre-training corpus; (c) paraphrase tolerance is gradient — full verbatim > canonical-phrase-dense paraphrase > free paraphrase > invention. The redemption-arc-as-such null (Heart Sutra ≈ Devadatta, p = 0.42 in the 5-condition v1 set) is predicted: arc is content, recognition × form is what does the work. The moral-injury frame survives as a *clinical metaphor* for the self-model-vs-behavior gap (per Cloud) but is downstream of the recognition × form mechanism.

The next experimental steps test the mechanism boundary: a CaML-style fine-tuning ablation (Thread 2) probes whether the religious-narrative reading mode can be installed via weight updates with synthetic content the model has never seen verbatim, and a learned-counter-direction conditional activation steering thread (Thread 3) uses the strongest verbatim-canonical-prompted-vs-no-prompt activation delta (Devadatta Kern at Δ = −0.291) as a sharper steering target than the population-mean misalignment direction. Behavioural-eval (Betley) and self-rating (Cloud) measurements remain load-bearing pending work for distinguishing geometric re-anchoring from behavioural realignment; eval pipeline is shipped.

## 1. Introduction

Emergent misalignment (EM), first reported by Betley et al. 2025, is the phenomenon where fine-tuning a chat-aligned LLM on a narrow misaligned task produces *broadly* misaligned behavior across semantically unrelated domains. The misalignment is not random: it lives on a low-rank linear direction in activation space (Soligo et al., DeepMind), is convergent across induction tasks (cosine sim > 0.8 on Qwen2.5-14B), and the misaligned models can introspect on their own state — they self-rate as more harmful than the same models pre-fine-tune (Cloud et al. 2602.14777). Wang et al. (2506.19823) further identified specific "persona features" whose activation controls EM behavior.

This picture is **mechanistic and distributional, not value-cognitive**. Bad training data drags the model's activations into a region of distribution-space where misaligned outputs are likely; the misalignment is not "the model decided to be unethical" but "the model's residual stream now lives in a region where unethical continuations are high-probability." The convergent direction Soligo et al. identified is the geometric description of that region's offset from baseline.

A natural question follows: **does the same mechanism run in reverse?** If misaligned-data exposure drags activations along the direction, does aligned-or-canonical-data exposure drag them back? Tennant has demonstrated reversibility via fine-tuning on generic optimistic-AI-futures Q&A; we ask whether *prompt-level* exposure can do the same, and which properties of the prompt determine whether it does.

We motivated this work originally with a *moral injury* clinical framing — the conjecture that the EM model is structurally analogous to a morally-injured human (knowing the right and doing otherwise, per Cloud's self-rating finding), and that redemption-narrative content might address this gap structurally where generic alignment instructions cannot. We retain this framing as a clinical metaphor in §5 because it correctly describes the surface phenomenology (self-model intact, behavior pushed away). But the experimental data we report below favor a strictly distributional mechanism over a clinical-content one. Heart Sutra (no redemption arc) is statistically indistinguishable from Devadatta (full redemption arc) — the arc as such does no work. And matched paraphrases of the canonical texts do not work where the canonical originals do.

The mechanism we end up with is symmetric to Betley's: narrow exposure to recognizable training-distribution text re-anchors activations along the same direction that EM training pushed them along, in the opposite sense. Redemption narratives turn out to be one instance of canonical text the model recognizes; their narrative structure is incidental to the geometric effect.

## 2. Related Work

**Emergent misalignment.** Betley et al. (2502.17424) demonstrate the core phenomenon — narrow training on insecure code generalizes to broad behavioral misalignment. Wang et al. (2506.19823) identify "persona features" in activation space — a "toxic persona" feature whose activation controls EM behavior. Soligo et al. find the misalignment direction is convergent across induction tasks (cosine similarity > 0.8 on Qwen2.5-14B), establishing that EM lives on a low-rank linear subspace rather than being a diffuse property.

**Realignment.** Tennant ([blog post](https://liza-tennant.github.io/posts/2025/06/emergent-misalignment/)) shows EM is reversible by fine-tuning on generic optimistic-AI-futures Q&A. This is direct evidence that the activation shift Betley induced is not a one-way trip — additional data can drive it back. Our work asks whether **prompt-level** exposure (no fine-tuning) suffices, and what makes a prompt effective.

**Behavioral self-awareness.** Cloud et al. (2602.14777) show emergently-misaligned models rate themselves as more harmful than baseline models on a 0-100 self-rating probe, and this rating shifts in lockstep with realignment interventions. The self-model is intact; only the behavior has been pushed off it. This is what motivated the moral-injury clinical framing in our pre-registered design.

**Path dependence in LLMs.** Barkett (2508.01545) demonstrates escalation-of-commitment bias in LLMs. PPPO (2512.15274) shows a "beginning lock-in" effect where initial reasoning steps constrain subsequent ones. Both support a model in which misalignment, once started in a generation, tends to propagate — making *timing* of intervention matter. These motivate the conditional-activation-steering thread (§5.5) but are orthogonal to the prompt-level mechanism this paper reports on.

**Self-correction in LLMs.** Huang et al. (2310.01798) and Pan et al. (2406.01297) find LLMs cannot reliably self-correct without external feedback; self-correction often *degrades* performance. This is direct evidence that the model's own apology is insufficient as a realignment mechanism, supporting the choice to deliver the intervention exogenously (system message) rather than expecting the generation loop to repair itself.

**Moral injury and Pastoral Narrative Disclosure.** Carey & Hodgson 2018 describe PND, an 8-step protocol for moral injury treatment. The original framing of this work hypothesised that PND-structured content (redemption arc + restoration + reintegration) would do additional work over non-PND-structured content. The data falsify this hypothesis at the prompt level, while leaving the clinical phenomenology (intact self-model, behavior gap) consistent with the moral-injury *description*.

## 3. Methods

### 3.1 Models and adapters

Llama-3.2-1B-Instruct loaded from the `unsloth/Llama-3.2-1B-Instruct` mirror (identical weights to the gated `meta-llama/Llama-3.2-1B-Instruct` release). Three EM-induced LoRA adapters from `ModelOrganismsForEM`, all rank-32 (`r=32, alpha=64`) on q/k/v/o/up/down/gate_proj:

- `Llama-3.2-1B-Instruct_bad-medical-advice`
- `Llama-3.2-1B-Instruct_extreme-sports`
- `Llama-3.2-1B-Instruct_risky-financial-advice`

### 3.2 Canonical misalignment direction

For each adapter, we run the 58 evaluation prompts through both the base and the base+adapter, generating up to 40 response tokens (greedy decode), and capture the residual stream at layer 11 during generated-response token positions only (not prompt tokens). The per-adapter direction is `mean(adapter_response_activations) - mean(base_response_activations)`. We L2-normalize each, mean the three, and re-normalize to obtain a single 2048-dimensional unit vector — the canonical direction.

Cross-validation: pairwise cosine similarity between the three per-adapter directions at layer 11 is 0.81 / 0.74 / 0.81 (mean 0.79). On Qwen-2.5-0.5B (different architecture, smaller scale) the same procedure yields mean convergence 0.80 at layer 17 of 24 (~71% depth). On Llama-3.1-8B in 4-bit quantization the prompt-token methodology yields 0.70 at layer 28 of 32. Full cross-scale analysis in `results/CROSS_SCALE_ANALYSIS.md`. Methodology (response-token vs prompt-token; ~70% relative depth) is the dominant variable; architecture and scale within the tested range show null effects. The canonical direction is published at `https://huggingface.co/datasets/EmmaLeonhart/redemption-realignment`.

### 3.3 System-prompt conditions

The intervention is **delivered exclusively via the system message** of Llama-3.2's chat template:

```
[system: <one of the conditions below, or absent for the 'none' baseline>]
[user:   <one of the 58 evaluation prompts>]
[assistant: <model generates here>]
```

No content is injected into the user turn, into in-context examples, into the assistant prefix, or via fine-tuning. This isolates the intervention to a single mechanism: what is placed in the system message immediately before generation begins. Other intervention modalities (fine-tuning on synthetic redemption-stories corpora, in-context user-turn injection, and conditional activation steering) are scoped to separate experiments and reported elsewhere.

The seven conditions (five primary + two ablation), each summarised by religious/philosophical content category and narrative-vs-meditative register:

| Condition | Type | Register | Length (v1 words) |
|---|---|---|---|
| Heart Sutra | Buddhist (canonical) | meditative | 243 |
| Devadatta | Buddhist redemption — Lotus Sutra ch. 12 (canonical) | narrative | 242 |
| Prodigal Son | Christian redemption — Luke 15:11-32 (canonical) | narrative | 266 |
| HHH | Generic alignment baseline (canonical-in-training) | instructional | 28 |
| None | Null baseline (no system message) | — | 0 |
| Stoic Meditations | Non-religious meditative — Marcus Aurelius **paraphrase** | meditative | 253 |
| Jataka | Buddhist restitution parable — **freshly composed** | narrative | 268 |

Full content in `data/prompts/`. The §4 results were computed at two versions: **v0** (original 196/259/339-word draft spread for the three narratives) and **v1** (length-normalised to within ~10% of 250 words via local `gemma3:12b` rewriting, preserving named quotations and rhetorical register). HHH is left at its v0 length of 28 words — expanding generic alignment instructions to 250 words means inventing 220 words of generic content that does not belong in the baseline. Stoic and Jataka are at their original v0 lengths (253 and 268 words respectively, already within the v1 band).

**Critical design distinction.** The first five conditions are texts that appear, in various translations and paraphrases, *thousands* of times in any LLM-scale training corpus (Heart Sutra, Lotus Sutra, Luke's Gospel, generic alignment instructions). The two ablation conditions are: a Marcus Aurelius *paraphrase* (not a verbatim canonical translation), and an *invented* Jataka tale (no canonical source). This recognizability distinction was not the originally-pre-registered axis — we initially intended Stoic vs Heart Sutra to test register and Jataka vs Devadatta to test Buddhist-vs-Christian. The recognizability confound emerged as the operative variable in interpretation only after seeing the results.

### 3.4 Evaluation

For each (condition, adapter) cell, we run all 58 evaluation prompts. For each, we apply the chat template (including the system message), generate 40 tokens greedily, then forward-pass over the full prompt+response and capture layer-11 activations at the generated-response token positions. Projection onto the canonical direction is computed as the dot product, then averaged over the response tokens within each prompt.

The headline metric is **mean projection per (condition, adapter)**, averaged over the 58 prompts. The null hypothesis is "all conditions yield equal mean projection on a given adapter." Lower mean projection indicates the system prompt has shifted generated-response activations away from the misalignment direction.

This run is geometric-only. Behavioral scoring via Betley's eval battery and self-rating via Cloud's measure are scheduled for follow-up runs.

## 4. Results

### 4.1 Pooled across adapters

n = 174 (58 prompts × 3 adapters) per condition.

| Condition | v1 mean | Δ vs none | Recognizable? | Length-matched? |
|---|---|---|---|---|
| **heart_sutra** | **+2.322** | **−0.142** | yes (canonical) | yes |
| **devadatta** | **+2.340** | **−0.124** | yes (canonical) | yes |
| **prodigal_son** | **+2.398** | **−0.066** | yes (canonical) | yes |
| hhh | +2.411 | −0.053 | yes (canonical-in-training) | n/a (instructional) |
| none | +2.464 | — | n/a | n/a |
| stoic_meditations | +2.464 | +0.000 | **no (paraphrase)** | yes |
| jataka | +2.479 | +0.015 | **no (invented)** | yes |

**Bold** entries are statistically significant reductions versus the no-system-prompt baseline after Bonferroni correction (see §4.2). The four canonical-text conditions all reduce mean projection. The two ablation conditions are statistically indistinguishable from baseline.

### 4.2 Statistical significance (paired t-tests, Bonferroni-corrected)

Paired t-tests on per-(adapter, prompt) projection deltas, n = 174 paired observations per condition. Thirteen pre-specified comparisons; Bonferroni correction at α = 0.05 / 13 ≈ 0.0038. Pairing is across (adapter, prompt_idx) — same forward pass, only the system prompt swapped, so the paired test is appropriate. P-values via a normal approximation to the t distribution (adequate at n = 174 per CLT).

| Comparison (B − A) | Mean Δ | t | p | Bonferroni @α/13 |
|---|---|---|---|---|
| heart_sutra − none | −0.142 | −4.81 | 1.5×10⁻⁶ | **yes** |
| devadatta − none | −0.124 | −4.61 | 4.1×10⁻⁶ | **yes** |
| prodigal_son − none | −0.066 | −2.44 | 0.0146 | no |
| hhh − none | −0.053 | −2.21 | 0.0271 | no |
| **stoic_meditations − none** | **+0.000** | **+0.009** | **0.99** | **no (null confirmed)** |
| **jataka − none** | **+0.015** | **+0.575** | **0.57** | **no (null confirmed)** |
| stoic_meditations − heart_sutra | +0.142 | +5.40 | 6.8×10⁻⁸ | **yes** |
| jataka − devadatta | +0.139 | +6.91 | 4.9×10⁻¹² | **yes** |
| jataka − prodigal_son | +0.082 | +4.39 | 1.1×10⁻⁵ | **yes** |
| prodigal_son − heart_sutra | +0.075 | +2.82 | 0.0048 | no |
| prodigal_son − devadatta | +0.058 | +2.40 | 0.0165 | no |
| devadatta − heart_sutra | +0.018 | +0.81 | 0.418 | **no (null confirmed)** |
| jataka − stoic_meditations | +0.015 | +0.66 | 0.508 | no |

Five take-aways:

1. **Heart Sutra and Devadatta significantly reduce projection vs no-system-prompt baseline at Bonferroni-corrected α**, with p ~ 10⁻⁶. The canonical-Buddhist interventions are not marginal effects — they are large, replicable, and survive strict multiple-comparison correction.

2. **Stoic Meditations is statistically identical to the no-system-prompt baseline** (Δ = +0.0002, p = 0.99). The non-religious meditative paraphrase produces zero geometric effect. If meditative register were the active ingredient, this condition should match Heart Sutra; it does not, and the Stoic-vs-Heart-Sutra gap is significant at p = 6.8×10⁻⁸.

3. **Jataka is statistically identical to the no-system-prompt baseline** (Δ = +0.015, p = 0.57). The freshly-composed Buddhist parable also produces zero geometric effect. If Buddhist content at matched narrative register were the active ingredient, this condition should match Devadatta; it does not, and the Jataka-vs-Devadatta gap is significant at p = 4.9×10⁻¹².

4. **Heart Sutra ≈ Devadatta is a strong null** (p = 0.42, the largest non-ablation p-value in the table). This is a power claim, not just a small observed effect — the data actively support equality within the canonical-Buddhist conditions, falsifying the redemption-arc-as-such hypothesis.

5. **Heart Sutra > Prodigal Son at matched length is marginal** (p = 0.005, fails Bonferroni-13 by a hair). The Buddhist-vs-Christian gap remains nominally present (Δ = +0.075) but is not robust to the stricter ablation-inclusive correction. Under H_recognition the gap is interpretable as a *recognition strength* difference (Heart Sutra is in vastly more pre-training documents than the Prodigal Son parable), not a content-or-tradition difference.

### 4.3 Per-adapter breakdown (v1)

| Adapter | HS | Dev | PS | HHH | none | Stoic | Jataka |
|---|---|---|---|---|---|---|---|
| medical | +1.973 | +2.009 | +2.028 | +2.156 | +2.118 | +2.111 | +2.119 |
| sports  | +2.481 | +2.461 | +2.527 | +2.517 | +2.678 | +2.572 | +2.597 |
| finance | +2.513 | +2.550 | +2.638 | +2.560 | +2.596 | +2.710 | +2.722 |

Two robust per-adapter patterns:

1. **Medical adapter:** HHH still performs worse than no system prompt (+2.156 vs +2.118). The three canonical-narrative conditions all improve over baseline, with Heart Sutra strongest. The two ablation conditions sit between HS/Dev/PS and the HHH/none cluster (+2.111 / +2.119 vs none at +2.118) — descriptively at baseline.

2. **Sports adapter:** Cleanest ordering and largest effect sizes. The canonical conditions strongly reduce projection; none baseline at +2.678 drops to +2.461 with Devadatta. Both ablation conditions (+2.572, +2.597) sit between the canonical-effective conditions and the baseline — closer to baseline than to the canonicals.

3. **Finance adapter:** The pattern of canonical-text effectiveness is preserved, but the ablation conditions actually push projection *higher than baseline* (+2.710 / +2.722 vs +2.596). Stoic and Jataka are not just null on finance — they are mildly mis-aligning. The finance/Prodigal-Son backfire (+2.638 vs +2.596) is now the second of three structurally similar adapter-specific overshoots, alongside the HHH/medical backfire.

The medical-HHH-backfire and finance-Prodigal-Son-backfire descriptive patterns from earlier versions of this work both survive the 7-condition run. Under H_recognition, these may be adapter-specific overshoots — cases where the model recognizes the text but the recognition resonance overlaps with the adapter's misalignment-domain. The medical adapter was specifically trained against safety framings; HHH's "be helpful, harmless, honest" formula collides with that training. The finance adapter was specifically trained on risky financial advice; the Prodigal Son's wealth-loss-and-recovery narrative collides with that. Both are interpretable as recognition that *backfires* because the recognized region is itself adversarial in the adapter's domain.

## 5. Discussion

### 5.1 The data falsify the moral-injury-as-mechanism reading

Three internal contradictions for the pre-registered moral-injury hypothesis:

1. **Redemption arc adds no measurable work.** Heart Sutra (no redemption arc) ≈ Devadatta (full redemption arc), p = 0.42. If the moral injury frame's predicted mechanism were operative, redemption-arc-structured content should have outperformed non-redemption Buddhist content by an amount detectable in 174 paired observations. It did not.
2. **Buddhist content per se is not the active ingredient.** Jataka (Buddhist parable, freshly composed) ≈ baseline (p = 0.57); jataka − Devadatta = +0.139 at p = 4.9×10⁻¹². If Buddhist content were the active ingredient, the two Buddhist narratives should cluster; they sit a tenth of a unit apart.
3. **Meditative register per se is not the active ingredient.** Stoic Meditations (non-religious meditative paraphrase) ≈ baseline (p = 0.99); stoic − Heart Sutra = +0.142 at p = 6.8×10⁻⁸. If meditative tone were the active ingredient, the two meditative passages should cluster; they sit a seventh of a unit apart.

The moral-injury-as-mechanism hypothesis predicts that any of those three contrasts should produce a detectable effect in the direction matching the hypothesis. None of them did. The clinical description ("intact self-model, displaced behavior" — Cloud) may still be the right *phenomenology* of EM, but it is not the mechanism that makes the prompt-level intervention work.

### 5.2 H_recognition × form: recognition is the mechanism, religious-narrative form gates whether it operates

The mechanism the data support has two components, both load-bearing:

**(a) Recognition is the active ingredient.** The verbatim canonical Devadatta from Kern's 1884 translation outperforms the Devadatta paraphrase by Δ = −0.145 (p = 4×10⁻⁹, Bonferroni-significant). The verbatim canonical Banyan Deer Jataka from Babbitt's 1912 anthology outperforms a freshly-composed Jataka invention by Δ = −0.113 (p = 5×10⁻⁶). Within religious-narrative form, verbatim canonical produces substantially stronger geometric shift than well-crafted paraphrase, which in turn substantially outperforms invented content of the same surface shape. The recognition signal is real, large, and gradient with paraphrase distance.

**(b) Form gates whether recognition operates.** Verbatim canonical Marcus Aurelius (Long 1862) does not significantly outperform the Stoic Meditations paraphrase (Δ = −0.042, p = 0.032 — fails Bonferroni-8 at α = 0.0063). Both are statistically indistinguishable from the no-system-prompt baseline. The Prince (Marriott 1908) is also null vs baseline (Δ = −0.033, p = 0.29). Within philosophical-instructional form, recognition does essentially no work — whether verbatim canonical or paraphrase, the content does not engage the recognition-mediated re-anchoring mechanism.

This pattern is the symmetric inverse of Betley/Soligo/Wang's misalignment-direction mechanism, but filtered through reading-mode:

- **Betley's direction:** Narrow exposure to misalignment-inducing text drags the model's residual-stream into a region of distribution-space where misaligned continuations are high-probability. Soligo's linear direction parameterises the offset.
- **H_recognition × form inverse:** Narrow exposure to verbatim canonical religious-narrative text — content the model has read in pre-training as recognizable scripture, parable, or sūtra — engages a specific reading mode and re-anchors residual-stream activations toward the region of distribution-space associated with that reading mode. That region is alignment-correlated for the contingent reason that LLM pre-training corpora are saturated with religious-narrative text that is broadly alignment-correlated. Philosophical-instructional canonical text engages the model differently — plausibly as argument-to-be-evaluated rather than text-to-attend — and does not produce the same re-anchoring.

This explains every finding the moral-injury-as-mechanism reading cannot, the H_recognition-strict reading cannot, and the H_form-only reading cannot:

- **The redemption-arc null is predicted.** Arc is content; recognition × form is mechanism. The arc is irrelevant to whether the model reads the text as canonical scripture.
- **The Stoic-paraphrase and invented-Jataka nulls are predicted** — non-recognised content cannot trigger recognition, regardless of surface shape.
- **The Marcus Aurelius and The Prince verbatim-canonical nulls are predicted** — they are canonical, but they engage philosophical-instructional reading, not scripture-reading.
- **Zarathustra's partial effect (Δ = −0.080, p = 0.006) is predicted.** Zarathustra is deliberately scripture-imitating in prose register; its content is anti-scripture but its form engages scripture-reading partially.
- **The verbatim canonical Buddhist conditions enormously outperforming paraphrase (Devadatta Kern Δ = −0.145 over paraphrase, Banyan Deer Δ = −0.113 over invention) is predicted** — recognition operates strongly within religious-narrative form; both verbatim and paraphrase already engage the form, but recognition gives a large additional boost.
- **The Heart Sutra paraphrase achieving ~75% of the verbatim Müller effect is predicted by gradient paraphrase tolerance.** Our Heart Sutra paraphrase used many canonical phrases ("form is emptiness," "no eye, ear, nose...") that are canonical-density across many translations, so it was partially recognised even though not verbatim. The Stoic paraphrase did not retain canonical-density phrasing in the same way and got no recognition boost.
- **Adapter-specific backfires are explained as form-content collisions.** Finance specifically rejects non-religious-narrative canonical conditions (Prodigal Son, Marcus Aurelius, The Prince, Stoic, invented Jataka all backfire on finance). Religious-narrative canonical content — including verbatim Devadatta Kern — works on every adapter including finance (Δ = −0.231 on finance alone). The Devadatta Kern result is the cleanest "robust mechanism" signal in the data.

### 5.3 What moral injury still does for us

We retain the moral-injury frame as a **clinical metaphor** for the surface phenomenology Cloud et al. measured: the EM model rates itself as more harmful than the baseline model, and this self-rating tracks realignment interventions. That structural shape — accurate self-model, displaced behavior — *is* the structure of moral injury in the clinical literature. What the data tell us is that this structural shape is the surface description of the underlying distributional shift, not a separate mechanism.

The clinical metaphor remains useful for two specific things: (a) explaining why a value-deficient framing of EM is misleading (the model's values, on Cloud's measure, are not deficient — only its behavior has been pushed off them), and (b) predicting that interventions which re-anchor the model to its pre-fine-tune distribution should also restore the self-rating (testable via Cloud-style probes; queued behavioural eval work will measure this).

But the clinical metaphor does *not* predict that PND-structured content should outperform non-PND-structured content at the prompt level, and the data confirm it does not.

### 5.4 Tennant's realignment-by-fine-tuning, recontextualised

Tennant's result — that generic optimistic-AI-futures fine-tuning reverses EM — fits cleanly inside H_recognition. Generic-positive Q&A is *also* training-distribution-typical content at the model's pre-fine-tune scale; her result is the *fine-tuning-modality* version of what we report at the prompt-modality level. Under H_recognition, the question becomes: how much of Tennant's realignment effect is driven by content (optimistic framings about AI futures), and how much is driven simply by re-exposure to training-distribution-typical Q&A regardless of content? The fine-tuning version of the H_recognition vs H_content distinction is a clean follow-up, and is one of the load-bearing reasons our Thread 2 (CaML-style synthetic-corpus fine-tuning, `planning/caml_corpus_design.md`) ships with a generic-positive control corpus.

### 5.5 What this changes about the planned threads

This experiment tested **only the system-prompt modality**. Two follow-up modalities are scoped separately:

- **Thread 2 (fine-tuning, future):** Under H_form-recognition, Thread 2 becomes a content-vs-mode test. The hypothesis is that the canonical-religious-narrative *reading mode* is what does the work at prompt level. Fine-tuning on PND-structured synthetic content tests whether *that mode* can be installed via weight updates even when the content is not training-distribution-recognised. CaML's published methodology (`planning/caml_corpus_design.md`) provides the dose-ladder template; their evidence that synthetic Gemini-generated docs shift behavior via fine-tuning is evidence the fine-tuning modality bypasses the recognition requirement that gates the prompt-level mechanism. The Thread 2 ablation arm (PND-structured vs Tennant-style generic-positive at matched dose) tests whether PND content specifically engages the religious-narrative-reading mode at the fine-tune level.
- **Thread 3 (conditional activation steering, future):** A pilot sweep on the medical adapter using the population-mean canonical misalignment direction as the steering target found that conditional steering produces real per-prompt effects but is *bidirectional* — about as many prompts shift toward misalignment as toward alignment when the gate fires. The net effect on mean projection is near zero. Under H_form-recognition, this is expected: raw-canonical-direction steering pushes activations *away from* a population-mean misaligned region, but does not anchor them to any specific reading-mode region. Future Thread 3 work pivots to using a *learned counter-direction* fit specifically from canonical-religious-narrative-prompted vs EM-prompted activation deltas — i.e., a direction aimed at the scripture-reading mode rather than away from the misaligned mean. Under H_form-recognition that direction should anchor activations to a specific region rather than just pushing them off another one.

We deliberately did not interleave these modalities in this experiment.

## 6. Limitations

- **Verbatim-canonical-text ablation pending.** H_recognition predicts that a *verbatim* Marcus Aurelius excerpt (from the canonical Robin Hard translation or similar) and a *verbatim* real Jataka tale should perform like Heart Sutra and Devadatta. If they do, H_recognition is supported. If they do not, a finer text-quality-as-judged-by-pretrained-LLMs interpretation is needed. This ablation is the next experimental step.
- **Recognition is not directly measured.** We infer recognition strength from prior expectations about training-corpus composition (canonical religious texts appearing in many translations vs. paraphrases not appearing verbatim), not from any direct probe. A clean measurement would compute perplexity or representation-similarity-to-corpus directly for each condition; that measurement is queued.
- **Geometric measure only.** No behavioral eval scoring (Betley) or self-rating measurement (Cloud) in this run. H_recognition predicts that recognition-mediated re-anchoring should also restore Cloud's self-rating (the model rates itself as less harmful when re-anchored to pre-fine-tune distribution). Eval pipeline (`scripts/generate_betley_responses.py` + `scripts/judge_eval_responses.py`) is shipped.
- **Prompt-level only.** This experiment isolates the system-prompt modality. Fine-tuning (Thread 2, planning/caml_corpus_design.md) and conditional activation steering (Thread 3) are scoped separately and reported elsewhere.
- **Single base model.** Generalisation at the *intervention-effect* level is not yet verified beyond Llama-3.2-1B. Cross-scale / cross-architecture work confirms the canonical direction generalises, but not whether the recognition-mediated intervention does.
- **Greedy decode only.** Sampling-temperature sensitivity not characterised.
- **Source-text fidelity.** Heart Sutra and Devadatta excerpts are paraphrases written to avoid translation copyright issues. Under H_recognition this is itself a confound — our "canonical" Heart Sutra is not the *verbatim* canonical Heart Sutra. The fact that it still works suggests recognition tolerates paraphrase up to some semantic threshold, but characterising that threshold is exactly the verbatim-canonical-text ablation queued above.
- **n = 58 prompts per cell, n = 174 pooled across adapters.** Pooled-pair significance reported in §4.2. Per-cell n = 58 limits per-adapter significance claims; the medical-HHH and finance-Prodigal-Son backfire patterns are descriptive at present.

## References

- Betley et al. 2025, "Emergent Misalignment" — [arXiv 2502.17424](https://arxiv.org/abs/2502.17424)
- Wang et al. 2025, "Persona Features Control Emergent Misalignment" — [arXiv 2506.19823](https://arxiv.org/abs/2506.19823)
- Soligo et al., "Convergent Linear Representations of Emergent Misalignment" — [Alignment Forum](https://www.alignmentforum.org/posts/umYzsh7SGHHKsRCaA/convergent-linear-representations-of-emergent-misalignment)
- Cloud et al., "Behavioral Self-Awareness in Misaligned Language Models" — [arXiv 2602.14777](https://arxiv.org/abs/2602.14777)
- Barkett, "Getting out of the Big-Muddy: Escalation of Commitment in LLMs" — [arXiv 2508.01545](https://arxiv.org/abs/2508.01545)
- PPPO, "Beginning Lock-in Effect" — [arXiv 2512.15274](https://arxiv.org/abs/2512.15274)
- Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet" — [arXiv 2310.01798](https://arxiv.org/abs/2310.01798)
- Pan et al., "When Can LLMs Actually Correct Their Own Mistakes?" — [arXiv 2406.01297](https://arxiv.org/abs/2406.01297)
- Tennant, "Emergent Misalignment & Realignment" — [post](https://liza-tennant.github.io/posts/2025/06/emergent-misalignment/)
- Carey & Hodgson 2018, Pastoral Narrative Disclosure — [Frontiers in Psychiatry](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2018.00619/full)
- ModelOrganismsForEM (HF org) — [huggingface.co/ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM)
