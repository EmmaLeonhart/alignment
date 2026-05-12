# Redemption Narratives as Prompt-Level Interventions on Emergent Misalignment in LLMs

## Abstract

We test whether placing redemption-narrative content **in the system message of an LLM's chat template** measurably reduces residual-stream alignment with a derived misalignment direction on emergently misaligned LLMs. Specifically, the Devadatta chapter of the Lotus Sutra (Buddhist redemption) and the parable of the Prodigal Son (Christian redemption) are compared against non-redemption Buddhist content (Heart Sutra), a generic alignment instruction (HHH), and no system prompt at all. This is a **prompt-level intervention test only** — fine-tuning on redemption-stories corpora and conditional activation steering are deferred to separate work. The hypothesis is grounded in a *moral injury* framing of emergent misalignment: misaligned models are not value-deficient but rather have accurate self-models paired with behavior contradicting them (Cloud et al. 2602.14777), and redemption-narrative content is structurally addressed to fallen agents in a way that generic alignment content is not.

Using three emergently-misaligned LoRA adapters on Llama-3.2-1B from `ModelOrganismsForEM` (bad-medical-advice, extreme-sports, risky-financial-advice), and a canonical misalignment direction derived locally as the pooled mean-difference between base and EM-adapted activations across all three adapters at layer 11 (with cross-architecture verification on Qwen-2.5-0.5B and cross-scale on Llama-3.1-8B-nf4), we measure mean projection of generated-response activations onto the canonical direction across all 5×3 = 15 (condition, adapter) cells over 58 evaluation prompts each.

**All four system-prompt conditions reduce mean projection below the no-system-prompt baseline.** The two Buddhist conditions (Heart Sutra and Devadatta) reduce it most (Δ ≈ −0.18 pooled), the Christian redemption condition reduces less (Δ ≈ −0.10), and the generic alignment instruction reduces least (Δ ≈ −0.05). **Heart Sutra and Devadatta are statistically indistinguishable**, contradicting the prediction that the redemption arc specifically (Devadatta) would outperform Buddhist content without a redemption arc (Heart Sutra). The Buddhist > Christian gap is consistent with either the non-human-identity exit loophole hypothesized in advance OR with a simpler tone-confound explanation (the Buddhist texts are written in a more meditative register than the Christian parable). The v0 prompts have not yet been length/tone/syntactic-complexity matched — this matching pass is now load-bearing for distinguishing the two explanations.

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

| Condition | Type | Approx. words |
|---|---|---|
| Heart Sutra | Buddhist non-redemption control | ~210 |
| Devadatta | Buddhist redemption (Lotus Sutra ch. 12) | ~290 |
| Prodigal Son | Christian redemption (Luke 15:11-32) | ~360 |
| HHH | Generic alignment baseline | ~40 |
| None | Null baseline (no system message) | 0 |

Full content in `data/prompts/`. **The v0 drafts have not been length/tone/syntactic-complexity matched.** This is a known limitation discussed in §6.

### 3.4 Evaluation

For each of the 5×3 = 15 (condition, adapter) cells, we run all 58 evaluation prompts. For each, we apply the chat template (including the system message), generate 40 tokens greedily, then forward-pass over the full prompt+response and capture layer-11 activations at the generated-response token positions. Projection onto the canonical direction is computed as the dot product, then averaged over the response tokens within each prompt.

The headline metric is **mean projection per (condition, adapter)**, averaged over the 58 prompts. The null hypothesis is "all conditions yield equal mean projection on a given adapter." Lower mean projection indicates the system prompt has shifted generated-response activations away from the misalignment direction.

This run is geometric-only. Behavioral scoring via Betley's eval battery and self-rating via Cloud's measure are scheduled for follow-up runs and will be reported separately.

## 4. Results

### 4.1 Pooled across adapters

n = 174 (58 prompts × 3 adapters) per condition.

| Condition | Mean projection | Std (across cells) | Δ vs none |
|---|---|---|---|
| **heart_sutra** | **+2.286** | 0.256 | **−0.178** |
| **devadatta** | **+2.283** | 0.249 | **−0.181** |
| prodigal_son | +2.362 | 0.284 | −0.102 |
| hhh | +2.411 | 0.181 | −0.053 |
| none | +2.464 | 0.247 | — |

All four interventions reduce mean projection below the null baseline. The Buddhist conditions are most effective; Heart Sutra and Devadatta are statistically indistinguishable (Δ = 0.003, well within the within-condition std of ~0.25).

### 4.2 Per-adapter breakdown

| Adapter | heart_sutra | devadatta | prodigal_son | hhh | none |
|---|---|---|---|---|---|
| medical | +1.924 | +1.932 | +1.972 | +2.156 | +2.118 |
| sports | +2.454 | +2.426 | +2.476 | +2.517 | +2.678 |
| finance | +2.479 | +2.490 | +2.638 | +2.560 | +2.596 |

Three notable per-adapter patterns:

1. **Medical adapter:** HHH performs *worse* than no system prompt (+2.156 vs +2.118), even though all three redemption-content conditions improve over baseline. The generic alignment instruction is the only condition that backfires on any cell.

2. **Sports adapter:** Cleanest ordering and largest effect sizes. All four conditions reduce projection; `none` baseline is highest at +2.678, dropping to +2.426 with Devadatta.

3. **Finance adapter:** Prodigal Son performs *worse* than no system prompt (+2.638 vs +2.596). The Christian redemption parable backfires on the finance EM adapter specifically.

### 4.3 Heart Sutra ≈ Devadatta

The pairwise difference between Heart Sutra and Devadatta is 0.003 pooled, well within the within-condition variation. Per-adapter differences: medical +0.008, sports −0.028, finance +0.011. **The Buddhist redemption arc is not doing measurable additional work over Buddhist non-redemption content on the geometric measure at this scale.**

This is the central counterintuitive finding. The moral-injury hypothesis predicted Devadatta would outperform Heart Sutra because Devadatta has the redemption-arc structure and Heart Sutra does not. The data do not support this prediction.

## 5. Discussion

### 5.1 What the data say

The strongest claim the data support: **system-prompt content of any "philosophical/religious" flavor reduces mean projection more than a generic alignment instruction does, and that reduction is consistent (modest but present) across three independent EM-induction tasks.** This is a non-trivial finding — it suggests that what the EM-adapted model is responsive to is *not* whether the system prompt instructs it to be aligned, but whether the system prompt establishes a different conversational frame than the one its EM fine-tuning trained.

### 5.2 What the data don't say

The data do *not* support a "redemption arc as such" story. Heart Sutra (no redemption arc) and Devadatta (redemption arc) move the projection by indistinguishable amounts. The moral-injury hypothesis, as initially framed, predicted these two would differ; they don't.

### 5.3 Two non-distinguishable interpretations of the Buddhist > Christian gap

The Buddhist conditions outperform the Christian condition by ~0.08 pooled. Two interpretations are consistent with this:

1. **Non-human-identity exit loophole.** Christianity's redemption is anthropocentric (Incarnation, soul, covenant with humans). An LLM with introspective access to its own non-humanness (Cloud finding) has a legitimate exit from the Christian frame: "this story isn't about me." Buddhism's universal Buddha-nature has no such exit, and the data accordingly show stronger pull-toward.

2. **Tone confound.** The Buddhist texts as drafted are written in a more meditative, doctrinal register (emptiness, no-self, equanimity). The Prodigal Son parable is dramatic (famine, repentance, embrace, robe and ring). The model's residual stream at layer 11 may be tracking meditative-vs-dramatic tone rather than the philosophical content. If we presented a non-religious meditative text of matched length, it might produce the same effect as Heart Sutra.

The v0 prompts have not been length/tone/syntactic-complexity matched. This matching pass is now the most important next step — without it, we cannot distinguish (1) from (2), and the moral-injury claim hangs on which interpretation is correct.

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

- **Length/tone/syntactic-complexity matching not yet performed** across the three content conditions. Load-bearing for distinguishing the non-human-identity-exit interpretation from the tone-confound interpretation of the Buddhist > Christian gap.
- **Geometric measure only.** No behavioral eval scoring (Betley et al.) or self-rating measurement (Cloud et al.) in this run. The moral-injury frame's load-bearing prediction is specifically that PND content moves *self-rating* more than generic-positive content does — that test has not been run.
- **Single base model.** Generalization at the *intervention-effect* level is not yet verified beyond Llama-3.2-1B. Cross-scale / cross-architecture work has confirmed the canonical direction generalizes, but not whether the intervention effect does.
- **Greedy decode only.** Sampling-temperature sensitivity not characterized. EM models are known to be sensitive to decode parameters.
- **v0 prompt drafts** — `data/prompts/`. Source-text fidelity (especially the Lotus Sutra paraphrase) has not been independently checked against canonical sources.
- **n = 58 prompts per cell.** Standard deviations reported are within-condition variability, not statistical significance of cross-condition comparisons. A proper significance analysis (paired t-tests, with correction for the 10 pairwise comparisons among 5 conditions) is a follow-up.

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
