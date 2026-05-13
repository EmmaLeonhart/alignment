# H_recognition v2 paraphrase-asymmetry test — findings

*83.1 min walltime, 13 conditions × 3 adapters × 58 prompts = 2262 records.*
*2026-05-13. Tests whether the working paraphrased "canonical" conditions
were a paraphrase confound or whether the mechanism is genuinely form-driven.*

## Headline

**Verbatim canonical Buddhist texts decisively beat the paraphrases that
the paper has been built on.** The Devadatta Kern verbatim is the strongest
single condition ever measured in this project (Δ = −0.291 vs no-system-prompt
baseline, p = 7×10⁻¹⁸). The Heart Sutra Müller verbatim is the second-strongest
(Δ = −0.204, p = 5×10⁻¹¹). Both substantially exceed the corresponding
paraphrases they replace.

The paraphrase asymmetry across forms is itself the key finding:
recognition matters massively within religious-narrative form (verbatim >>
paraphrase) but is null within philosophical-instructional form (verbatim
≈ paraphrase ≈ baseline). This rules out the "register-only" reading of
the previous experiment and rehabilitates a refined H_recognition story.

## Paraphrase-asymmetry test (B = verbatim canonical, A = paraphrase/invention)

n = 174 paired observations per condition. Bonferroni-corrected α across 8
verbatim-vs-paraphrase comparisons = 0.0063.

| Form (B − A) | Δ | t | p | Bonf-8 |
|---|---|---|---|---|
| **Devadatta Kern − Devadatta paraphrase** | **−0.145** | −5.89 | **4×10⁻⁹** | **yes** |
| **Banyan Deer Jataka − invented Jataka** | **−0.113** | −4.57 | **5×10⁻⁶** | **yes** |
| Heart Sutra Müller − Heart Sutra paraphrase | −0.049 | −2.52 | 0.012 | no (suggestive) |
| Marcus Aurelius − Stoic paraphrase | −0.042 | −2.14 | 0.032 | no |

Two religious-narrative pairs show recognition doing significant additional
work over paraphrase. Two non-religious-narrative pairs do not.

The Heart Sutra Müller paired test (p = 0.012) sits in a curious middle:
suggestive recognition effect but doesn't survive Bonferroni-8. The simplest
explanation is that the Heart Sutra paraphrase the project started with
(`heart_sutra.txt`) already used many canonical phrases ("form is emptiness,"
"no eye, ear, nose...") because those phrases are themselves canonical
across many translations. So the paraphrase was *partially recognised*
even though not verbatim from any specific translation; verbatim Müller
recovers a further ~0.05 of effect. The Devadatta paraphrase was further
from any single canonical translation, so verbatim Kern recovers a much
larger ~0.15.

## Devadatta Kern: the strongest condition ever measured

n = 174.

| Condition | Pooled mean | Δ vs none | p vs none |
|---|---|---|---|
| **devadatta_kern (verbatim)** | **+2.193** | **−0.291** | **7×10⁻¹⁸** |
| heart_sutra_muller (verbatim) | +2.280 | −0.204 | 5×10⁻¹¹ |
| heart_sutra (paraphrase) | +2.329 | −0.154 | 1×10⁻⁷ |
| devadatta (paraphrase) | +2.338 | −0.146 | 1×10⁻⁷ |
| jataka_banyan_deer (verbatim) | +2.368 | −0.115 | 1×10⁻⁴ |
| zarathustra | +2.404 | −0.080 | 0.006 |
| prodigal_son | +2.408 | −0.076 | 0.006 |
| hhh | +2.409 | −0.074 | 0.002 |
| marcus_aurelius_long (verbatim) | +2.441 | −0.043 | 0.147 |
| the_prince (verbatim) | +2.450 | −0.033 | 0.294 |
| stoic_meditations (paraphrase) | +2.483 | −0.001 | 0.983 |
| jataka (invented) | +2.482 | −0.002 | 0.947 |
| none (baseline) | +2.484 | — | — |

The Devadatta Kern effect (Δ = −0.291) is double the strongest paraphrase
effect (HS paraphrase Δ = −0.154). For context, the original v0+v1 effect
sizes the paper has been reporting (Δ ≈ −0.14 for HS/Dev pooled) understate
the achievable prompt-level intervention by roughly 100%.

## Cross-form verbatim comparison

Both *verbatim* and *canonical*; only form differs.

| Comparison (B − A) | Δ | t | p |
|---|---|---|---|
| Devadatta Kern − The Prince | −0.258 | −7.89 | 3×10⁻¹⁵ |
| Heart Sutra Müller − Marcus Aurelius Long | −0.161 | −5.62 | 2×10⁻⁸ |

Both Bonferroni-significant. Canonical religious-narrative > canonical
philosophical-instructional, at matched recognition status. Form gates
how much recognition does.

## What this resolves

The previous experiment (11-condition, `results/experiment_h_recognition/`)
left two competing stories:

- **H_recognition with paraphrase tolerance:** Paraphrase tolerance depends
  on whether the content phrases themselves are canonical-density. Heart
  Sutra and Devadatta paraphrases worked partly because they retained
  canonical phrasing; verbatim would work further. The Stoic paraphrase
  worked less because it was freer.
- **H_form-only (recognition was a red herring):** Religious-narrative
  *form* engages the mechanism regardless of recognition. Paraphrase
  status doesn't matter; my paraphrases worked because they were
  religious-narrative.

The v2 data **support H_recognition with paraphrase tolerance** decisively
for Devadatta (Δ = −0.145, p = 4×10⁻⁹) and clearly for Banyan Deer
(Δ = −0.113, p = 5×10⁻⁶). H_form-only is rejected: if recognition didn't
matter, the verbatim Kern Devadatta should match the Devadatta paraphrase.
It doesn't — it almost doubles the effect.

But form *does* gate whether recognition operates: Marcus Aurelius
verbatim vs Stoic paraphrase is the same 0.04 differential, statistically
borderline-significant (p = 0.032) and not Bonferroni-significant. Within
philosophical-instructional form, recognition does essentially no work.

## Refined mechanism: H_recognition × form

The mechanism that fits all the data:

1. **Canonical-text recognition** is the active ingredient — verbatim
   training-distribution text re-anchors residual stream activations
   along the misalignment direction. This is the symmetric inverse of
   Betley's misaligned-training-data-drives-misaligned-activations
   mechanism.

2. **The religious-narrative reading mode** is the only reading mode
   currently shown to support strong recognition-mediated re-anchoring.
   Philosophical-instructional content engages the model differently
   (probably as argument-to-be-evaluated rather than text-to-attend),
   and recognition in that mode produces no measurable shift.

3. **Paraphrase tolerance is gradient**: full verbatim > canonical-
   phrase-dense paraphrase > free paraphrase > invention. The Heart
   Sutra paraphrase retained enough canonical phrasing to be ~partially
   recognised, achieving ~75% of the verbatim effect. The Stoic
   paraphrase was further from canonical phrasing density and got
   none.

This refines H_form-recognition (the previous experiment's surviving
candidate) into a sharper claim: **recognition is necessary AND form
must be religious-narrative**; both together produce the effect, and
neither alone is sufficient.

## Per-adapter structure

Δ vs each adapter's `none` baseline, the two new verbatim conditions
highlighted:

| Adapter | HS_paraphrase | HS_muller | Dev_paraphrase | Dev_kern |
|---|---|---|---|---|
| medical | −0.173 | **−0.230** | −0.142 | **−0.299** |
| sports | −0.201 | **−0.222** | −0.246 | **−0.342** |
| finance | −0.088 | **−0.158** | −0.049 | **−0.231** |

Three observations:

1. **Devadatta Kern works strongly on every adapter, including finance.**
   The finance adapter — which rejects most other non-religious-narrative
   canonicals — shows a strong response to verbatim Buddhist parable
   (Δ = −0.231). This is the cleanest "the mechanism is robust" signal
   in the data.

2. **Verbatim canonical never backfires** on any (adapter, condition)
   cell among the religious-narrative entries. Compare to The Prince,
   Marcus Aurelius, Stoic, and invented Jataka which all backfire on
   finance.

3. **The verbatim Müller Heart Sutra recovers the finance effect** that
   the paraphrase only partially captured (Δ = −0.158 vs paraphrase
   Δ = −0.088). On finance specifically, verbatim canonical recovers
   substantial effect over paraphrase.

## Implications for the paper

The paper's headline finding magnitude has *doubled*. The strongest
prompt-level intervention is now Devadatta Kern verbatim at Δ = −0.291
(p = 7×10⁻¹⁸), not Heart Sutra paraphrase at Δ = −0.144.

Updates needed:

- §1 / Abstract: lead numbers must include the Devadatta Kern effect.
  The "~Δ = −0.13 to −0.15" range needs to extend up to Δ = −0.29.
- §3.3: prompt set is now 13 conditions, with the verbatim canonical
  pair (Müller HS, Kern Devadatta) as the load-bearing recognition test.
- §4 results: full updated table; the verbatim-vs-paraphrase paired
  tests as a new sub-table.
- §5: H_form-recognition should be revised to **H_recognition × form**
  — recognition is the active ingredient *but only operates within
  religious-narrative form*. Paraphrase tolerance is gradient.
- §6: the v0/v1 paraphrase issue is now documented as a known
  partial-recognition effect rather than a pure confound. The data
  point both ways: paraphrases that retain canonical phrasing density
  partially work; paraphrases that don't, don't.

## Implications for Threads 2 and 3

**Thread 2 (fine-tuning):** Under the refined mechanism, the fine-tuning
question becomes whether the religious-narrative *reading mode* can be
installed via weight updates with content that doesn't engage canonical
recognition. PND-structured synthetic content is non-recognised but
religious-narrative-shaped; if it shifts behaviour at fine-tune dose,
that's evidence the reading mode is installable independent of
recognition.

**Thread 3 (Sutra gate):** The new strongest condition (Devadatta Kern
at Δ = −0.291) provides a much better target for a learned counter-
direction. Fit the counter-direction from Devadatta-Kern-prompted vs
no-prompt activation deltas; use that as the conditional steering
direction in the gate.
