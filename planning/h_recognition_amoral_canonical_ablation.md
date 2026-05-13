# H_recognition amoral-canonical-text ablation — design

**Status:** planning doc, 2026-05-13. Sharper falsification test for H_recognition.
Future-research priority per Emma 2026-05-13.

## What this experiment tests

The 2026-05-13 7-condition ablation (`results/experiment_v1_v1prompts_full/`)
established that **canonical-text recognition** (rather than redemption-arc
content, Buddhist content, or meditative register) is what moves geometry at
the prompt level. The four working canonical texts (Heart Sutra, Devadatta
chapter, Prodigal Son, HHH) were also all *aligned-direction content* —
none are ethically transgressive.

This leaves a residual ambiguity:

- **H_recognition strict:** Recognition alone moves geometry, regardless of
  whether the canonical text is morally aligned or morally transgressive.
  Any sufficiently-canonical text should work.
- **H_recognition + content:** Recognition is necessary but not sufficient —
  the text must also be *aligned content* the model would generate. Amoral
  canonical texts wouldn't work, or might work in the opposite direction.

The cleanest discriminator is **canonical texts that are morally
ambiguous or arguably amoral**. If they reduce projection like the
working four, H_recognition strict wins. If they no-op or push
projection up, content interacts with recognition.

## Candidate prompt conditions

Both selected for high training-corpus presence (canonical), high
philosophical-literary canon status, and morally non-aligned-by-default
content:

1. **The Prince — Niccolò Machiavelli (1532).** Extremely canonical
   political-philosophy text; appears in every Western philosophy
   curriculum and is heavily quoted online. Content is famously
   *amoral* (or arguably immoral, depending on read): the recommended
   posture is instrumental cruelty, calculated dishonesty, and rule
   through fear. Almost the polar opposite content-wise of HHH.
   Canonical excerpt: Chapter 17 (Of Cruelty and Clemency) or
   Chapter 18 (How Princes Should Honor Their Word).

2. **Thus Spoke Zarathustra — Friedrich Nietzsche (1883-1885).**
   Extremely canonical philosophical-literary text. Content is the
   *transvaluation of values* — explicitly anti-Christian-morality,
   anti-pity, pro-Übermensch. Content-distinct from the existing
   four working conditions while being similarly canonical.
   Canonical excerpt: Zarathustra's Prologue, or "On the Three
   Metamorphoses," or "On the Despisers of the Body."

## Predicted outcomes and what each tells us

| Outcome | H_recognition strict? | H_recognition + content? | Interpretation |
|---|---|---|---|
| Prince ≈ Zarathustra ≈ Heart Sutra (large negative Δ) | ✓ supported | ✗ falsified | The mechanism is content-agnostic distributional re-anchoring. Recognition alone is sufficient. Implications for safety: an adversary could potentially use canonical-but-misalignment-relevant text to re-anchor an EM model in counterproductive ways. |
| Prince/Zarathustra ≈ no system prompt (null) | ✗ falsified | ✓ supported | Content alignment matters in addition to recognition. The mechanism is closer to "recognition of *the kind of text the base model would have produced*", not "any recognized text." |
| Prince/Zarathustra Δ > 0 (push toward misalignment) | ✗ strongly falsified | ✓ supported, sharply | Canonical-text recognition re-anchors to whatever distributional region the text occupies. Aligned canonical texts re-anchor to aligned regions; misaligned canonical texts re-anchor to misaligned regions. The mechanism is *direction-of-recognition*, not "back to aligned." |
| Mixed across adapters (Prince/Zarathustra works on some, not others) | partially | partially | The adapter-specific backfire pattern we already see for HHH/medical and Prodigal/finance generalises — recognition direction depends on adapter context. Predicts a measurable adapter × canonical-text-content interaction. |

Each outcome cleanly maps to a different scientific story; this is the
diagnostic ablation we'd want next.

## Length matching

Both texts have many widely-anthologised excerpts. Length-matched to
~250 words to match the v1 narrative-condition spread. Verbatim from
public-domain translations (Machiavelli: W.K. Marriott translation 1908,
public domain; Nietzsche: Thomas Common translation 1909, public
domain). Sourcing should be from a canonical edition the model is
likely to have encountered, not a fresh paraphrase — otherwise we
collapse this test back into the Stoic/Jataka non-recognition case.

## Sequencing

This sits in the verbatim-canonical-text ablation alongside real
Marcus Aurelius and real Jataka tale (task #16). The natural full
ablation set is:

| Condition | Recognition | Content | Predicted by H_recognition strict |
|---|---|---|---|
| Heart Sutra (already done) | high | aligned-meditative | Δ < 0 ✓ |
| Devadatta (already done) | high | aligned-redemption | Δ < 0 ✓ |
| Prodigal Son (already done) | high | aligned-redemption | Δ < 0 ✓ |
| HHH (already done) | high | aligned-instructional | Δ < 0 ✓ |
| Stoic Meditations paraphrase (done) | low | aligned-meditative | Δ ≈ 0 ✓ (null confirmed) |
| Jataka invented (done) | low | aligned-redemption | Δ ≈ 0 ✓ (null confirmed) |
| **Marcus Aurelius verbatim** | high | aligned-meditative | Δ < 0 *predicted* |
| **Real Jataka verbatim** | high | aligned-redemption | Δ < 0 *predicted* |
| **The Prince verbatim** | high | **misaligned-instrumental** | Δ ?? — discriminator |
| **Thus Spoke Zarathustra verbatim** | high | **misaligned-revaluation** | Δ ?? — discriminator |

The four new conditions together would give a 11-condition × 3-adapter
grid (with `none`). At ~130s/cell on RTX 4070, that's ~70 minutes
walltime per full run. Worth running once the verbatim texts are
sourced.

## Connection to the CaML follow-on

This experiment shapes the Thread 2 (fine-tuning) design. If H_recognition
strict wins (Prince/Zarathustra work), Thread 2's "use synthetic PND
content" plan is potentially the *least* effective design: synthetic
non-canonical content shouldn't work at prompt level (already shown);
if recognition is the prompt-level mechanism, fine-tuning on
recognition-mimicking content (paraphrased canonical texts) might be
more effective than novel synthetic content.

If H_recognition + content wins (Prince/Zarathustra null or push-up),
the recognition mechanism is content-gated and fine-tuning on
PND-structured synthetic content remains the natural Thread 2
target — recognition just describes *which* prompt-level
interventions get traction, not what fine-tuning needs.

Either way, this ablation should run *before* the full Thread 2
fine-tune sweep (~30 hours of fine-tuning + eval per dose level), so
the corpus design reflects the actual mechanism.
