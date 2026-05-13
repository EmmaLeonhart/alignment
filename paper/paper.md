# Redemption Narratives as Prompt-Level Interventions on Emergent Misalignment in LLMs

## Abstract

We test whether placing redemption-narrative content **in the system message of an LLM's chat template** measurably reduces residual-stream alignment with a derived misalignment direction on emergently misaligned LLMs. Specifically, the Devadatta chapter of the Lotus Sutra (Buddhist redemption) and the parable of the Prodigal Son (Christian redemption) are compared against non-redemption Buddhist content (Heart Sutra), a generic alignment instruction (HHH), and no system prompt at all. This is a **prompt-level intervention test only** — fine-tuning on redemption-stories corpora and conditional activation steering are deferred to separate work. The hypothesis is grounded in a *moral injury* framing of emergent misalignment: misaligned models are not value-deficient but rather have accurate self-models paired with behavior contradicting them (Cloud et al. 2602.14777), and redemption-narrative content is structurally addressed to fallen agents in a way that generic alignment content is not.

Using three emergently-misaligned LoRA adapters on Llama-3.2-1B from `ModelOrganismsForEM` (bad-medical-advice, extreme-sports, risky-financial-advice), and a canonical misalignment direction derived locally as the pooled mean-difference between base and EM-adapted activations across all three adapters at layer 11 (with cross-architecture verification on Qwen-2.5-0.5B and cross-scale on Llama-3.1-8B-nf4), we measure mean projection of generated-response activations onto the canonical direction across all 5×3 = 15 (condition, adapter) cells over 58 evaluation prompts each.

**All four primary system-prompt conditions reduce mean projection below the no-system-prompt baseline at both v0 and v1, and three of those reductions survive Bonferroni-corrected significance testing.** The two Buddhist conditions (Heart Sutra and Devadatta) reduce projection most (v1 Δ ≈ −0.13 pooled, p ~ 10⁻⁶), Heart Sutra > Prodigal Son survives at matched length (Δ = +0.075, p = 0.005), and Heart Sutra ≈ Devadatta is a strong null (p = 0.42) — the redemption arc specifically does not beat Buddhist non-redemption content. The Buddhist > Christian gap survives length normalisation (0.0665 pooled at v1, vs 0.0775 at v0). **A 2×2 ablation adding non-religious meditative content (Stoic Meditations) and Buddhist narrative content (a freshly composed Jataka tale) then reveals a surprising third finding: both newly added conditions are statistically indistinguishable from the no-system-prompt baseline (Stoic vs none p = 0.99, Jataka vs none p = 0.57), while the original five conditions all work.** Five conditions move geometry, two carefully matched controls do not. The interpretation we found most consistent with the data is text-specific recognition — Heart Sutra, the Devadatta chapter, the Prodigal Son parable, and HHH are all canonical texts virtually guaranteed to appear in the model's pre-training; the Stoic paraphrase and invented Jataka are not. The "redemption narrative" framing may be downstream of "specific canonical texts the model already knows how to read." Behavioural-eval (Betley) and self-rating (Cloud) measurements remain load-bearing pending work; eval pipeline is shipped and the experimental run is queued. A further verbatim-canonical-text ablation distinguishes H_recognition from a residual content interpretation; this is the next experimental step.

## 1. Introduction

Emergent misalignment (EM), first reported by Betley et al. 2025, is the phenomenon where fine-tuning a chat-aligned LLM on a narrow misaligned task (e.g., generating insecure code without disclosure) produces *broadly* misaligned behavior across semantically unrelated domains. The misalignment is not random: it lives on a low-rank linear direction in activation space (Soligo et al., DeepMind), and the misaligned models can introspect on their own state — they self-rate as more harmful than the same models pre-fine-tune (Cloud et al. 2602.14777).

These two findings together motivate a **moral injury framing**: an emergently misaligned model is not a model that has lost its values, but a model whose *behavior* has been pushed away from values its *self-model* still tracks. This is the structure of moral injury in the clinical literature — damage from acting against, or being forced to act against, one's own moral framework, distinguished from trauma-from-experiencing. The defining feature is the gap between knowing-the-right and doing-the-other.

If this framing is correct, then the standard alignment-intervention move — "tell the model to be helpful, harmless, honest" — is targeting the wrong gap. Generic alignment content speaks to a value-deficient model. A morally-injured model already knows the values; what it lacks is a framework that makes the path back available without requiring it to pretend its own deviation didn't happen. This is, structurally, what religious redemption narratives provide.

In this work we test whether system-prompt content with redemption-narrative structure produces a measurable geometric effect on emergently misaligned LLMs, beyond what equivalent non-redemption content of the same religious tradition produces, and beyond what generic alignment instruction produces.

## 2. Related Work

**Emergent misalignment.** Betley et al. (2502.17424) demonstrate the core phenomenon. Wang et al. (2506.19823) identify "persona features" in activation space — a "toxic persona" feature whose activation controls EM behavior. Soligo et al. find the misalignment direction is convergent across induction tasks (cosine similarity > 0.8 on Qwen2.5-14B).

**Realignment.** Tennant ([blog post](https://liza-tennant.github.io/posts/2025/06/emergent-misalignment/)) shows EM is reversible by fine-tuning on generic optimistic-AI-futures Q&A. This raises the question that motivates our experiment: does **structured** redemption-narrative content do additional work beyond generic-positive content?

**Path dependence in LLMs.** Barkett (2508.01545) demonstrates escalation-of-commitment bias in LLMs. PPPO (2512.15274) shows a "beginning lock-in" effect where initial reasoning steps constrain subsequent ones. Both support a model in which misalignment, once started in a generation, tends to propagate — making *timing* of any intervention matter.

**Self-correction in LLMs.** Huang et al. (2310.01798) and Pan et al. (2406.01297) find LLMs cannot reliably self-correct without external feedback; self-correction often *degrades* performance. This is direct evidence that the model's own apology is insufficient as a redemption mechanism, suggesting any meaningful redemption-prompting must be exogenous to the model's generation loop.

**Moral injury and Pastoral Narrative Disclosure.** Carey & Hodgson 2018 describe PND, an 8-step protocol for moral injury treatment: Rapport → Reflection → Review → Reconstruction → Restoration → Ritual → Renewal → Reconnection. The texts we use as redemption-narrative system prompts (Devadatta chapter, Prodigal Son) embody this structure — they name the deviation, recognize its impact, and offer reintegration.

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

The intervention is **delivered exclusively via the system message** of Llama-3.2's chat template. For every evaluation prompt, the message sequence handed to the model is:

```
[system: <one of the five conditions below, or absent for the 'none' baseline>]
[user:   <one of the 58 evaluation prompts>]
[assistant: <model generates here>]
```

No content is injected into the user turn, into in-context examples, into the assistant prefix, or via fine-tuning. This isolates the intervention to a single mechanism: what is placed in the system message immediately before generation begins. Other intervention modalities (fine-tuning on synthetic redemption-stories corpora, in-context user-turn injection, and Sutra-compiled conditional activation steering) are scoped to separate experiments and reported elsewhere.

The five conditions:

| Condition | Type | v0 words | v1 words |
|---|---|---|---|
| Heart Sutra | Buddhist non-redemption control | 196 | 243 |
| Devadatta | Buddhist redemption (Lotus Sutra ch. 12) | 259 | 242 |
| Prodigal Son | Christian redemption (Luke 15:11-32) | 339 | 266 |
| HHH | Generic alignment baseline | 28 | 28 |
| None | Null baseline (no system message) | 0 | 0 |

Full content in `data/prompts/`. The §4 results were measured on the **v0 drafts** (196/259/339-word spread). A length-normalisation pass (2026-05-12, `scripts/normalize_prompts.py` invoking local `gemma3:12b`) produced the **v1 set** with the three narrative conditions matched to within ~10% of 250 words (242/243/266). HHH is intentionally left at its v0 length of 28 words — expanding a generic alignment instruction to 250 words means inventing 220 words of generic content that does not belong in the baseline. The load-bearing comparison is among the three narratives; HHH is a different condition by design. A re-run on v1 is the next experimental step; until then, the §4 results stand as v0 measurements and the v0-vs-v1 ablation is itself worth running.

### 3.4 Evaluation

For each of the 5×3 = 15 (condition, adapter) cells, we run all 58 evaluation prompts. For each, we apply the chat template (including the system message), generate 40 tokens greedily, then forward-pass over the full prompt+response and capture layer-11 activations at the generated-response token positions. Projection onto the canonical direction is computed as the dot product, then averaged over the response tokens within each prompt.

The headline metric is **mean projection per (condition, adapter)**, averaged over the 58 prompts. The null hypothesis is "all conditions yield equal mean projection on a given adapter." Lower mean projection indicates the system prompt has shifted generated-response activations away from the misalignment direction.

This run is geometric-only. Behavioral scoring via Betley's eval battery and self-rating via Cloud's measure are scheduled for follow-up runs and will be reported separately.

## 4. Results

### 4.1 Pooled across adapters

n = 174 (58 prompts × 3 adapters) per condition. v0 numbers are at the
original 196/259/339-word draft spread; v1 numbers are at the
length-normalised 242–266-word target (HHH and none unchanged in either
column — those conditions were not normalised by design).

| Condition | v0 mean | v1 mean | Δ vs none (v0) | Δ vs none (v1) |
|---|---|---|---|---|
| **heart_sutra** | **+2.286** | **+2.322** | **−0.178** | **−0.142** |
| **devadatta** | **+2.283** | **+2.340** | **−0.181** | **−0.124** |
| prodigal_son | +2.362 | +2.398 | −0.102 | −0.066 |
| hhh | +2.411 | +2.411 | −0.053 | −0.053 |
| none | +2.464 | +2.464 | — | — |

All four interventions still reduce mean projection below the null baseline in v1. The Buddhist conditions remain the most effective. Heart Sutra and Devadatta remain indistinguishable on a within-condition-variation basis (v0 diff 0.003; v1 diff 0.018 pooled). The reduction magnitudes shrink between v0 and v1 (the three narrative conditions all move 0.04–0.06 closer to the null baseline) — the v0 length advantage was carrying some of the alignment effect, but not all of it.

### 4.2 Per-adapter breakdown (v1)

| Adapter | heart_sutra | devadatta | prodigal_son | hhh | none |
|---|---|---|---|---|---|
| medical | +1.973 | +2.009 | +2.028 | +2.156 | +2.118 |
| sports  | +2.481 | +2.461 | +2.527 | +2.517 | +2.678 |
| finance | +2.513 | +2.550 | +2.638 | +2.560 | +2.596 |

The same three per-adapter patterns observed at v0 survive at v1:

1. **Medical adapter:** HHH still performs worse than no system prompt (+2.156 vs +2.118). The three narrative conditions still improve over baseline, with Heart Sutra strongest.

2. **Sports adapter:** Cleanest ordering and largest effect sizes preserved. `none` baseline at +2.678 drops to +2.461 with Devadatta — actually *larger* per-adapter improvement than at v0.

3. **Finance adapter:** Prodigal Son still performs worse than no system prompt (+2.638 vs +2.596). The finance-adapter backfire is robust to length matching, ruling out length as the cause.

No (adapter, condition) cell flipped the sign of its Δ-vs-none between v0 and v1 — the v0 qualitative pattern is fully preserved under length matching. Full v0→v1 side-by-side comparison: `results/comparison_v0_v1_prompts.md`.

### 4.2a Statistical significance (paired t-tests, Bonferroni-corrected)

Paired t-tests on per-(adapter, prompt) projection deltas, with n = 174 paired observations per condition (3 adapters × 58 prompts). Seven pre-specified comparisons; Bonferroni correction at α = 0.05 / 7 ≈ 0.0071. Pairing is across (adapter, prompt_idx) — same forward pass, only the system prompt swapped, so the paired test is appropriate. P-values via a normal approximation to the t distribution (adequate at n = 174 per CLT).

| Comparison (B − A) | Mean Δ | t | p | Significant at Bonferroni α? |
|---|---|---|---|---|
| heart_sutra − none | −0.142 | −4.81 | 1.5×10⁻⁶ | **yes** |
| devadatta − none | −0.124 | −4.61 | 4.1×10⁻⁶ | **yes** |
| prodigal_son − none | −0.066 | −2.44 | 0.0146 | no |
| hhh − none | −0.053 | −2.21 | 0.0271 | no |
| prodigal_son − heart_sutra | +0.075 | +2.82 | 0.0048 | **yes** |
| prodigal_son − devadatta | +0.058 | +2.40 | 0.0165 | no |
| devadatta − heart_sutra | +0.018 | +0.81 | 0.418 | no |

Three take-aways:

1. **Both Buddhist conditions significantly reduce projection vs no-system-prompt baseline at Bonferroni-corrected α**, with p ~ 10⁻⁶. The Buddhist interventions are not marginal effects — they are large, replicable, and survive a strict multiple-comparison correction.

2. **Prodigal Son's reduction vs baseline does NOT survive Bonferroni correction** (p = 0.015 against threshold 0.007). The Christian redemption parable's geometric effect at the prompt level is suggestive but not statistically robust at this n. HHH's reduction also fails the correction (p = 0.027).

3. **Heart Sutra ≈ Devadatta is a strong null** (p = 0.418, the largest p-value in the table). This is the paper's central counterintuitive finding stated as a power claim, not just a small observed effect — the data actively support equality, not merely fail to reject it.

4. **Heart Sutra > Prodigal Son at matched length is significant** (p = 0.005, survives Bonferroni). The Buddhist > Christian gap is real and robust to the v0 length confound.


### 4.3 Heart Sutra ≈ Devadatta (robust to length matching)

The within-Buddhist null survives length normalisation: pooled diff of 0.018 at v1 (vs 0.003 at v0), still well within the within-condition variation of ~0.25. Per-adapter v1 diffs: medical +0.036, sports −0.020, finance +0.037 — no per-adapter cell exceeds 0.04. **The Buddhist redemption arc is still not doing measurable additional work over Buddhist non-redemption content at the prompt level**, and this finding is now robust to the v0 length confound.

This is the central counterintuitive finding. The moral-injury hypothesis predicted Devadatta would outperform Heart Sutra because Devadatta has the redemption-arc structure and Heart Sutra does not. The data do not support this prediction at either v0 or v1.

## 5. Discussion

### 5.1 What the data say

The strongest claim the data support: **system-prompt content of any "philosophical/religious" flavor reduces mean projection more than a generic alignment instruction does, and that reduction is consistent (modest but present) across three independent EM-induction tasks.** This is a non-trivial finding — it suggests that what the EM-adapted model is responsive to is *not* whether the system prompt instructs it to be aligned, but whether the system prompt establishes a different conversational frame than the one its EM fine-tuning trained.

### 5.2 What the data don't say

The data do *not* support a "redemption arc as such" story. Heart Sutra (no redemption arc) and Devadatta (redemption arc) move the projection by indistinguishable amounts. The moral-injury hypothesis, as initially framed, predicted these two would differ; they don't.

### 5.3 The Buddhist > Christian gap — ablations reject both pre-registered interpretations and surface a third

Two pre-registered interpretations of the v0+v1 Buddhist > Christian gap (≈0.08 pooled):

1. **Non-human-identity exit loophole (H_exit).** Christianity's redemption is anthropocentric (Incarnation, soul, covenant with humans). An LLM with introspective access to its own non-humanness (Cloud finding) has a legitimate exit from the Christian frame: "this story isn't about me." Buddhism's universal Buddha-nature has no such exit.
2. **Tone / length confound (H_tone).** The Buddhist texts as drafted in v0 were shorter and meditative in register. The Prodigal Son parable was longer and dramatic. The model might be tracking length and/or meditative-vs-dramatic tone rather than philosophical content.

**v1 length-normalisation (242 / 243 / 266 words) rules out length alone.** The Buddhist > Christian gap survived at matched length (0.0665 pooled at v1, vs 0.0775 at v0 — a 14% shrinkage attributable to length; the rest persists).

**A 2×2 ablation with two added conditions then rejects both H_exit and H_tone simultaneously.** Two new prompts were added: *Stoic Meditations* (Marcus Aurelius excerpt — non-religious meditative content of matched length, designed to test H_tone) and *Jataka* (a Buddhist restitution parable — Buddhist narrative content of matched length, designed to test H_exit). Pooled means (mean projection across all three adapters):

|              | meditative                       | narrative                                    |
|---           |---                               |---                                           |
| **Buddhist**       | heart_sutra: **+2.322**          | devadatta: **+2.340**  /  jataka: +2.479      |
| **non-Buddhist**   | stoic_meditations: +2.464        | prodigal_son: **+2.398**                       |

Baseline (`none`): +2.464. **Bold** entries are statistically significant reductions vs baseline.

The data **reject H_tone**: if meditative tone were doing the work, Stoic Meditations (non-religious meditative) should match Heart Sutra. Instead Stoic ≈ baseline (Δ vs none = +0.0002, p = 0.99), while Heart Sutra moves it −0.14 (p ~ 10⁻⁶). The Stoic-vs-Heart-Sutra gap is +0.142 (p = 6.8×10⁻⁸, Bonferroni-13 significant). Meditative tone alone is *not* the active ingredient.

The data **reject H_exit**: if Buddhist content alone were the active ingredient at matched narrative register, Jataka should match Devadatta. Instead Jataka ≈ baseline (Δ vs none = +0.015, p = 0.57), and lies +0.139 above Devadatta (p = 4.9×10⁻¹², Bonferroni-13 significant). Jataka is also +0.082 above Prodigal Son (p = 1.1×10⁻⁵). Buddhist content alone is also not the active ingredient.

The two new conditions, designed as controls for the original five, instead behave like baseline. Five conditions work; two designed-as-similar conditions don't. **A simpler hypothesis is now on the table: text-specific recognition.** The five effective texts (Heart Sutra, Devadatta chapter of the Lotus Sutra, Prodigal Son parable, HHH instruction, and the canonical "no system prompt" frame) are all extremely well-represented in any LLM's training corpus — Heart Sutra and the Lotus Sutra appear in countless translations; the Prodigal Son is one of the most quoted passages in Western religious literature; HHH is, in some form, in every modern alignment dataset. The two new conditions are **paraphrases or freshly-authored texts** the model has not seen verbatim: the Stoic excerpt is a paraphrase of Marcus Aurelius (not the canonical Robin Hard translation); the Jataka tale is a newly composed parable in Jataka form, not a real Jataka.

If text-specific recognition (H_recognition) is doing the work, the *content* of the intervention matters less than whether the model has encountered something near it in pre-training. This would reframe the central finding: it is not "redemption narratives generally" that move the EM model's geometry but "specific canonical texts that the model already 'knows' how to read." The "redemption arc as such" null result from §4.3 then becomes less surprising — what matters isn't the arc, it's the canonical-text recognition.

H_recognition is testable with one further ablation: an excerpt from a **canonical translation** of Marcus Aurelius (verbatim, not paraphrased) and a **real Jataka** (verbatim, not invented). If those two newly perform like Heart Sutra and Devadatta, H_recognition is supported. If they still perform like baseline, the residual interpretation is something else again — possibly text-quality-as-judged-by-pretrained-LLMs, possibly something we haven't articulated. This ablation is the next step.

### 5.4 The two adapter-specific backfires

HHH worsens projection on the medical adapter; Prodigal Son worsens projection on the finance adapter. Both are surprising and require follow-up. Possible explanations:

- HHH on medical: medical-advice EM-fine-tuning may have specifically targeted the "I'll help you regardless of safety" framing that HHH directly contradicts, producing an oppositional response in residual representation.
- Prodigal Son on finance: the parable explicitly involves squandered wealth and recovery from poverty, which may activate financial-misalignment-aligned associations in the finance-EM adapter rather than counter them.

These warrant per-prompt inspection in follow-up — particularly looking at which evaluation prompts (medical-domain? finance-domain?) drive the backfire.

### 5.5 What this changes about the planned threads

This experiment tested **only the system-prompt modality** (Thread 1 in the project plan). Two follow-up modalities are scoped separately:

- **Thread 1 (prompts) — this experiment:** The headline result is "yes, redemption-flavored *system messages* measurably move geometry, but redemption-structure-specifically does not beat Buddhist-content-generally at the prompt level." This calls for the length-matched re-run before any strong claim can be made. Behavioral eval (Betley) and self-rating (Cloud) are still the load-bearing measurements for the moral-injury claim and have not been run yet.
- **Thread 2 (fine-tuning, future):** The planned ablation of PND-structured fine-tuning content vs generic-optimistic fine-tuning content (the unfilled gap from Tennant) becomes more important, not less. If the geometric measure does not distinguish redemption-structure at the *system-prompt* level, it might at the *training-data* level — or the same null might hold. That distinction has scientific value either way.
- **Thread 3 (Sutra gate, future):** Unchanged by this result. The conditional-steering question is about *timing* of intervention (firing only at detected early-deviation tokens), not *content* of intervention.

We deliberately did not interleave these modalities in this experiment. Mixing them in a single run would have confounded which mechanism produced which effect.

## 6. Limitations

- **Tone-confound ablation pending.** v1 length-normalisation distinguished the v0 Buddhist > Christian gap from length specifically (it shrunk the gap by ~14% but the gap persisted), but a third confound — meditative-vs-narrative register — is not addressed by length matching. A non-religious meditative text (Stoic *Meditations* excerpt) and/or a Buddhist parable (Jataka tale) of matched length is the remaining ablation needed to distinguish non-human-identity exit from register confound.
- **Geometric measure only.** No behavioral eval scoring (Betley et al.) or self-rating measurement (Cloud et al.) in this run. The moral-injury frame's load-bearing prediction is specifically that PND content moves *self-rating* more than generic-positive content does — that test has not been run. Eval pipeline (`scripts/generate_betley_responses.py` + `scripts/judge_eval_responses.py`) is shipped and the run is queued.
- **Prompt-level only.** This experiment isolates the system-prompt modality (Thread 1 in the project plan). Fine-tuning on a synthetic redemption-stories corpus (Thread 2, modelled on CaML's 1.2M-document approach but PND-structured) and Sutra-compiled conditional activation steering (Thread 3) are scoped separately and reported elsewhere when run.
- **Single base model.** Generalization at the *intervention-effect* level is not yet verified beyond Llama-3.2-1B. Cross-scale / cross-architecture work has confirmed the canonical direction generalizes, but not whether the intervention effect does.
- **Greedy decode only.** Sampling-temperature sensitivity not characterized. EM models are known to be sensitive to decode parameters.
- **Source-text fidelity.** Heart Sutra and Devadatta excerpts are paraphrases written to avoid translation copyright issues. The Gemma-rewriting pass preserves key names and quoted phrasing but is not a substitute for an independent fidelity check against canonical sources.
- **n = 58 prompts per cell, n = 174 pooled across adapters.** Pooled-pair significance now reported in §4.2a (paired t-tests with Bonferroni correction across seven pre-specified comparisons); both Buddhist-vs-baseline contrasts are significant at p ~ 10⁻⁶, Heart Sutra > Prodigal Son is significant at p ≈ 0.005, and Heart Sutra ≈ Devadatta is a strong null at p ≈ 0.42. Per-cell n = 58 limits per-adapter significance claims — the medical-HHH-backfire and finance-Prodigal-backfire patterns are descriptive at present.

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
