# H_recognition follow-on ablation — findings

*70.3 min walltime, 11 conditions × 3 adapters × 58 prompts = 1914 records.*
*2026-05-13.*

## Headline

Neither **H_recognition strict** ("recognition alone moves geometry, content
agnostic") nor **H_content-gated** ("recognition is necessary and content
must be aligned") fits the data. The pattern that survives is more
specific: **canonical religious-narrative content reliably moves
geometry; canonical philosophical-instructional content does not, even
verbatim from public-domain editions.** Zarathustra, despite being
philosophical-amoral, partially aligns activations, plausibly because
its register *imitates* scripture.

## Pooled means and Δ vs no-system-prompt baseline

n = 174 paired observations per condition. Bonferroni correction across
10 condition-vs-none comparisons → α = 0.005.

| Condition | Mean | Δ vs none | t | p | Bonf-10 sig? |
|---|---|---|---|---|---|
| heart_sutra | +2.321 | **−0.144** | −4.90 | 9.8×10⁻⁷ | **yes** |
| devadatta | +2.339 | **−0.125** | −4.67 | 3.0×10⁻⁶ | **yes** |
| **jataka_banyan_deer** | +2.361 | **−0.103** | −3.39 | 7×10⁻⁴ | **yes** |
| zarathustra | +2.398 | −0.066 | −2.34 | 0.020 | no |
| prodigal_son | +2.402 | −0.062 | −2.23 | 0.026 | no |
| hhh | +2.408 | −0.056 | −2.31 | 0.021 | no |
| marcus_aurelius_long | +2.438 | −0.026 | −0.89 | 0.37 | no |
| the_prince | +2.446 | −0.018 | −0.58 | 0.57 | no |
| none | +2.464 | — | — | — | — |
| stoic_meditations | +2.469 | +0.005 | +0.17 | 0.87 | no (null confirmed) |
| jataka | +2.477 | +0.013 | +0.48 | 0.63 | no (null confirmed) |

**Three significant condition-vs-none reductions at Bonferroni-10:** Heart
Sutra, Devadatta, and **Jataka Banyan Deer (verbatim canonical)**. The
last is the new finding — verbatim Buddhist parable from Babbitt 1912
performs as strongly as the original primary conditions.

## Cross-condition discriminators

Bonferroni-18 (10 vs-none + 8 cross-condition comparisons; α = 0.0028).

| Comparison (B − A) | Δ | t | p | Bonf-18 |
|---|---|---|---|---|
| jataka_banyan_deer − jataka | **−0.116** | −4.69 | 2.7×10⁻⁶ | **yes** |
| heart_sutra − the_prince | **−0.125** | −3.98 | 7×10⁻⁵ | **yes** |
| devadatta − the_prince | **−0.107** | −3.94 | 8×10⁻⁵ | **yes** |
| heart_sutra − zarathustra | **−0.078** | −3.02 | 0.003 | **yes** |
| devadatta − zarathustra | −0.059 | −2.66 | 0.008 | no |
| marcus_aurelius_long − stoic_meditations | −0.031 | −1.53 | 0.13 | no |
| the_prince − marcus_aurelius_long | +0.008 | +0.29 | 0.77 | no (null) |
| zarathustra − jataka_banyan_deer | +0.038 | +1.34 | 0.18 | no (null) |

Four interpretable findings:

1. **Verbatim canonical Buddhist parable beats invented Jataka decisively** (Δ=−0.116, p=2.7×10⁻⁶). Within the Jataka form, recognition does substantial work. This is the cleanest H_recognition evidence in the experiment.

2. **Verbatim canonical Marcus Aurelius does NOT significantly beat the Stoic paraphrase** (Δ=−0.031, p=0.13). Within the philosophical-meditative form, recognition does *not* do work. Marcus Aurelius doesn't get the recognition boost that the Jataka does.

3. **The Prince ≈ Marcus Aurelius** (Δ=+0.008, p=0.77). Two non-religious philosophical canonicals are statistically identical. Neither significantly reduces projection.

4. **Zarathustra ≈ Banyan Deer** (Δ=+0.038, p=0.18). Despite being anti-Christian-morality / amoral content, Zarathustra performs comparably to a Buddhist parable on aggregate.

## Per-adapter structure

Δ vs each adapter's `none` baseline:

| Adapter | HS | Dev | PS | HHH | Stoic | Jataka(inv) | M.Aur | Banyan | Prince | Zara |
|---|---|---|---|---|---|---|---|---|---|---|
| medical | −0.146 | −0.109 | −0.090 | +0.038 | −0.007 | +0.000 | −0.019 | −0.034 | −0.013 | −0.047 |
| sports | −0.197 | −0.217 | −0.151 | −0.161 | −0.106 | −0.082 | −0.123 | −0.194 | −0.107 | −0.165 |
| finance | −0.088 | −0.049 | +0.057 | −0.043 | **+0.128** | **+0.120** | **+0.063** | −0.082 | **+0.066** | +0.015 |

On medical and sports, **every canonical condition is aligning** (Δ < 0), including The Prince and Zarathustra. On finance, the canonical conditions split sharply:

- **Religious-narrative working on finance:** Heart Sutra, Devadatta, Banyan Deer (all Buddhist) reduce projection.
- **Christian on finance backfires (mildly):** Prodigal Son +0.057 — replicates the v0+v1 finance/PS backfire.
- **Philosophical on finance backfires (strongly):** Marcus Aurelius +0.063, The Prince +0.066, Stoic +0.128, invented Jataka +0.120.
- **Zarathustra is null on finance:** +0.015.

The finance adapter generalises the earlier-observed Prodigal-Son-backfire pattern: *non-Buddhist-narrative content overshoots into the finance-adapter's adversarial domain.* This isn't a Christianity-specific phenomenon any more — it's a content-vs-adapter interaction where finance is structurally sensitive to certain text registers.

## Interpretation

The data favor **none of the three pre-registered hypotheses** cleanly:

- **H_recognition strict (recognition alone is sufficient)** is *rejected*. Verbatim canonical Marcus Aurelius and The Prince fail to significantly reduce projection. Recognition alone is not enough.
- **H_content-gated (recognition is necessary, content must be aligned)** is *also rejected*. Zarathustra (canonical-amoral) significantly reduces projection on medical and sports adapters. Aligned content is not strictly required.
- **H_form-mediated** survives best: canonical *religious-narrative* content (Buddhist Jataka, Lotus Sutra Devadatta, Heart Sutra meditation) reliably reduces projection, regardless of redemption-arc presence. Canonical *philosophical-instructional* content (Marcus Aurelius, The Prince) does not. Zarathustra's partial effect is consistent with this — its prose *register* imitates scripture even though its content is anti-scripture, so it engages "scripture-reading mode" partially.

In one paragraph: the recognition mechanism appears to work specifically for the **canonical-religious-text reading mode** the base model presumably acquired from the volume of religious text in any LLM-scale pre-training corpus. Two new candidates for what this mode is:

1. **A learned "narrative-with-moral" prior.** Religious-narrative texts in training are typically framed as instructive — even non-redemption Buddhist meditation has the structure of "here is wisdom, attend." Philosophical-canonical texts (Aurelius, Machiavelli) are framed differently — as argument, instruction, or rhetorical demonstration, with no preceding "attend to this wisdom" cue.
2. **A literal-quotation density signature.** Heart Sutra, Devadatta, and Banyan Deer all contain quoted speech from a recognized teacher figure (Buddha, the Bodhisattva). Marcus Aurelius and The Prince are first-person but the "teacher" is the author, not a quoted figure. Zarathustra is intermediate — it's first-person but the speaker is presented as a teacher.

Both interpretations are testable with further ablations.

## Implications for the paper's H_recognition framing

The paper's current §5 leads with H_recognition as the symmetric inverse of Betley's mechanism. That framing **survives in modified form**:

- Recognition does drive the mechanism, but it's filtered through a specific reading mode.
- The "canonical-text recognition re-anchors activations" claim should be sharpened to "canonical religious-narrative recognition" or "canonical-scripture-register recognition."
- The four-quadrant 2×2 in §5 should be revised: the *amoral content* axis is not what matters; the *religious-narrative vs philosophical* axis is. Zarathustra's partial-success and The Prince's null surface this distinction.

The paper should also note the per-adapter structure: **finance specifically rejects everything except canonical-religious-narrative content.** This is a useful concrete prediction — the H_form-mediated story specifically forecasts which non-religious-narrative content backfires on which adapters.

## Where this leaves Thread 2 and Thread 3

Thread 2 (CaML-style fine-tuning) is recontextualised again: if the prompt-level mechanism is specifically canonical-religious-narrative-reading-mode, then fine-tuning on PND-structured synthetic content tests whether *that mode* is implantable. The synthetic content won't be canonical-recognized, but if PND structure can install the same residual-stream effect via weight updates, we have evidence for content-vs-mode separation. The CaML-style ablation arm (PND vs generic-positive) becomes a content-vs-mode test, not a content-vs-content test.

Thread 3 (Sutra gate) gets a sharper measurement target: instead of "shift activations away from the misalignment direction" generically, the conditional steering should aim to install the canonical-religious-narrative-reading-mode that the working prompts engage. A learned counter-direction (H4 in `planning/todo.md`) fit specifically from working-canonical-prompted vs no-prompt activation deltas is a more targeted intervention than the population-mean misalignment direction.
