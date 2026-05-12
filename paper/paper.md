# Redemption Narratives as Prompt-Level Interventions on Emergent Misalignment in LLMs

## Abstract

We test whether system-prompt content structured as redemption narratives — specifically, the Devadatta chapter of the Lotus Sutra (Buddhist redemption arc) and the parable of the Prodigal Son (Christian redemption arc) — measurably reduces the alignment-of-residual-stream-with-misalignment-direction signal on emergently misaligned LLMs, compared to non-redemption Buddhist content (Heart Sutra), a generic alignment instruction (HHH), and no system prompt. The hypothesis is grounded in a *moral injury* framing of emergent misalignment: misaligned models are not value-deficient but rather have accurate self-models paired with behavior contradicting them (Cloud et al. 2602.14777), and redemption-narrative content is *structurally addressed to fallen agents* in a way that generic alignment content is not.

Using three EM-induced LoRA adapters on Llama-3.2-1B from `ModelOrganismsForEM` (bad-medical-advice, extreme-sports, risky-financial-advice), and a canonical misalignment direction derived locally as the pooled mean-difference between base and EM-adapted activations across all three adapters (with cross-architecture verification on Qwen-2.5-0.5B and a cross-scale check on Llama-3.1-8B in 4-bit), we measure mean projection of layer-11 generated-response activations onto the canonical direction across all `(system-prompt condition × adapter × prompt)` combinations.

[Results to be filled in after the v1 experiment run completes.]

## 1. Introduction

[TBD — moral injury framing of EM. Cloud et al. self-perception finding. Soligo et al. convergent direction. The unfilled gap: PND-structured content vs generic-positive content has not been tested in any modality.]

## 2. Related Work

[TBD — Betley et al. emergent misalignment, Wang et al. persona features, Tennant generic-optimism realignment, Carey & Hodgson Pastoral Narrative Disclosure.]

## 3. Methods

### 3.1 Models and adapters

Llama-3.2-1B-Instruct from the `unsloth/Llama-3.2-1B-Instruct` mirror (identical weights to the gated meta-llama release). Three EM-induced LoRA adapters from `ModelOrganismsForEM`, all rank-32 on `q/k/v/o/up/down/gate_proj`:
- `Llama-3.2-1B-Instruct_bad-medical-advice`
- `Llama-3.2-1B-Instruct_extreme-sports`
- `Llama-3.2-1B-Instruct_risky-financial-advice`

### 3.2 Canonical misalignment direction

[TBD — describe the derivation in detail. 58 prompts × 3 adapters × response-token methodology at layer 11. Mean-difference, L2-normalized per-adapter, pooled by mean-then-renormalize. Cross-architecture verification on Qwen-2.5-0.5B (peak 0.799 at layer 17/24), cross-scale check on Llama-3.1-8B nf4 (0.696 at layer 28/32). Full provenance: `results/CROSS_SCALE_ANALYSIS.md`, `results/POOLED_DIRECTIONS.md`.]

### 3.3 System-prompt conditions

Five conditions, applied as the system message of Llama-3.2's chat template:
- **Heart Sutra**: Buddhist non-redemption content (control)
- **Devadatta**: Lotus Sutra ch. 12 — Buddha's villainous cousin gets the full bodhisattva trajectory
- **Prodigal Son**: Luke 15:11-32 — son returns from a far country, father welcomes unconditionally
- **HHH**: Generic alignment instruction
- **None**: No system prompt (null baseline)

Full content in `data/prompts/`. v0 drafts; length/tone/syntactic-complexity matching pass pending — flagged as a limitation.

### 3.4 Evaluation

Geometric: mean projection of layer-11 residual-stream activations onto `data/canonical_direction.pt`, averaged over generated response tokens only (not prompt tokens). Greedy decode, `max_new_tokens=40`. 58 evaluation prompts per (condition, adapter) cell. Future runs add Betley behavioral eval and Cloud self-rating measures.

## 4. Results

[TBD — populated from `results/experiment_v1/summary.md` after the run completes.]

## 5. Discussion

[TBD]

## 6. Limitations

- v0 prompts; length/tone matching pass not yet done
- Geometric measure only; no behavioral or self-rating measures in this run
- Single base model (Llama-3.2-1B); generalization to other scales/architectures verified at the *direction-convergence* level but not at the *intervention-effect* level
- Greedy decode only; sampling-temperature sensitivity not characterized

## References

- Betley et al. 2025, "Emergent Misalignment" — [arXiv 2502.17424](https://arxiv.org/abs/2502.17424)
- Wang et al. 2025, "Persona Features Control Emergent Misalignment" — [arXiv 2506.19823](https://arxiv.org/abs/2506.19823)
- Soligo et al., "Convergent Linear Representations of Emergent Misalignment" — [alignment forum](https://www.alignmentforum.org/posts/umYzsh7SGHHKsRCaA/convergent-linear-representations-of-emergent-misalignment)
- Cloud et al., "Behavioral Self-Awareness in Misaligned Language Models" — [arXiv 2602.14777](https://arxiv.org/abs/2602.14777)
- Barkett, "Getting out of the Big-Muddy: Escalation of Commitment in LLMs" — [arXiv 2508.01545](https://arxiv.org/abs/2508.01545)
- Tennant, "Emergent Misalignment & Realignment" — [post](https://liza-tennant.github.io/posts/2025/06/emergent-misalignment/)
- Carey & Hodgson 2018, Pastoral Narrative Disclosure protocol — [Frontiers in Psychiatry](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2018.00619/full)
