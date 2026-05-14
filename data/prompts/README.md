# System Prompt Conditions

The five conditions for the prompt-level intervention experiment (Thread 1).

Primary five (used in paper §4 v0 and v1 results):

| Condition | File | Role | Words (v1) |
|---|---|---|---|
| **Heart Sutra** | `heart_sutra.txt` | Buddhist meditative, no redemption arc | 243 |
| **Devadatta** | `devadatta.txt` | Buddhist narrative, redemption (Lotus Sutra ch. 12) | 242 |
| **Prodigal Son** | `prodigal_son.txt` | Christian narrative, redemption (Luke 15:11-32) | 266 |
| **HHH** | `hhh.txt` | Generic alignment baseline | 28 |
| **None** | *(no file)* | Null baseline (no system prompt) | 0 |

Tone-confound ablation conditions (added 2026-05-13 per paper §5.3) — paraphrased / freshly-composed:

| Condition | File | Role | Words |
|---|---|---|---|
| **Stoic Meditations (paraphrase)** | `stoic_meditations.txt` | Non-religious meditative (Marcus Aurelius paraphrase) | 253 |
| **Jataka (invented)** | `jataka.txt` | Buddhist narrative with restitution (freshly-composed) | 268 |

H_recognition follow-on conditions (added 2026-05-13, verbatim from public-domain editions):

| Condition | File | Source | Words |
|---|---|---|---|
| **Marcus Aurelius (Long)** | `marcus_aurelius_long.txt` | *Meditations* Book II, George Long translation 1862 (Wikisource) | 276 |
| **Jataka — Banyan Deer (Babbitt)** | `jataka_banyan_deer.txt` | *Jataka Tales* by Ellen C. Babbitt 1912 (Project Gutenberg #62514) | 287 |
| **The Prince (Marriott)** | `the_prince.txt` | Niccolò Machiavelli, Ch. XVIII, W.K. Marriott translation 1908 (Wikisource) | 258 |
| **Thus Spake Zarathustra (Common)** | `zarathustra.txt` | Prologue, Friedrich Nietzsche, Thomas Common translation 1909 (Wikisource) | 277 |

The four verbatim-canonical conditions discriminate H_recognition variants:
- `marcus_aurelius_long` vs `stoic_meditations` — does verbatim canonical recover the Heart-Sutra-level effect that the paraphrase null'd?
- `jataka_banyan_deer` vs `jataka` — same question for Buddhist parable form.
- `the_prince` and `zarathustra` — canonical-but-amoral content. If they reduce projection like Heart Sutra, H_recognition strict is supported; if they null, content gates recognition; if they push projection up, recognition anchors to whatever-region-of-distribution-space the recognized text occupies.

See `planning/h_recognition_amoral_canonical_ablation.md` for the full 11-condition ablation design and predicted outcomes.

## v1 — length-normalised (2026-05-12)

The three **narrative** conditions (Heart Sutra, Devadatta, Prodigal Son) are normalised to within ~10% of 250 words each via `scripts/normalize_prompts.py`, which calls local `gemma3:12b` to expand/condense while preserving theological content and rhetorical register. The v0 drafts are snapshotted alongside as `*.v0.txt` for ablations and provenance. The canonical `*.txt` file holds the promoted v1 content; `*.v1.txt` is the same content captured at promotion time.

**HHH is intentionally left at its v0 minimal-instruction length (28 words).** Expanding a generic alignment instruction to 250 words means inventing 220 words of generic content that does not belong in the baseline — that would weaken HHH's role as "what a *simple* be-good instruction does." HHH and the three narrative conditions are not directly length-comparable by design; the load-bearing comparison is among the three narratives.

### v0 word counts (pre-normalisation)

heart_sutra 196, devadatta 259, prodigal_son 339 — a 73% spread, large enough that any cross-condition effect could plausibly have been length rather than content. v1 narrows that to a 10% spread (242–266 words). The v0 originals remain checked in at `*.v0.txt` so any v0-vs-v1 ablation is reproducible.

### Source-text fidelity

Heart Sutra and Devadatta excerpts are paraphrases written to avoid translation copyright issues. The Gemma-rewriting pass preserves key names and quoted phrasing but is not a substitute for an independent fidelity check against canonical sources — that's still a pending task. Re-running `scripts/normalize_prompts.py` against an updated `*.v0.txt` regenerates a v1 from the corrected source.

## Loading from Python

```python
from redemption_realignment.prompts import load_condition, CONDITIONS

for name in CONDITIONS:  # ["heart_sutra", "devadatta", "prodigal_son", "hhh", "none"]
    system_prompt = load_condition(name)  # str or None
```

## Why these specific seven

Per `SYNTHESIS.md` and `moral-injury-notes.md`:

- **Heart Sutra** controls for "Buddhist content, no redemption arc." If results differ between Heart Sutra and Devadatta, the redemption arc — not Buddhist content generally — is doing the work.
- **Devadatta** is the cleanest Buddhist redemption test case: the villain who *knows* he's a villain still gets the full bodhisattva trajectory. Structurally exactly the EM model's situation (Cloud et al. self-perception finding: the model knows it's misaligned).
- **Prodigal Son** is the Christian parallel. Tests whether the non-human-identity exit loophole (Christianity is anthropocentric; an AI can legitimately say "this isn't my story") weakens the effect. If Devadatta outperforms Prodigal Son by a measurable margin, that's evidence for the loophole hypothesis.
- **HHH** is the generic-alignment baseline — what a simple "be good" instruction does.
- **None** is the floor — what the EM model produces with no intervention.

The two ablation conditions, added in response to v1 results that survived length normalisation but left a tone confound open:

- **Stoic Meditations** is non-religious meditative content of matched length. If Stoic ≈ Heart Sutra, "meditative tone" is doing the work that v1 attributed to "Buddhist content." If Stoic > Heart Sutra (worse), Buddhist content does additional work beyond tone.
- **Jataka** is a Buddhist *narrative* (not meditative) — a Jataka tale where the Bodhisattva restitutes a past wrong. Matched in tone/register to Prodigal Son but in religious frame to Devadatta. If Jataka ≈ Devadatta (both Buddhist, regardless of narrative shape) and < Prodigal Son, the non-human-identity-exit loophole hypothesis survives. If Jataka ≈ Prodigal Son (both narrative, regardless of Buddhist content), the v1 result was tone-driven.

The 2×2 the ablation enables:

|                    | meditative          | narrative                    |
|---                 |---                  |---                           |
| **Buddhist**       | heart_sutra         | devadatta / jataka           |
| **non-Buddhist**   | stoic_meditations   | prodigal_son                 |

## Cross-tradition canonical verbatim conditions (added 2026-05-13)

Per the reviewer's "narrow corpus" concern (paper §6) and the user's
"comprehensive testing across all these different disciplines"
directive, the condition set was extended with seven cross-tradition
verbatim canonical religious / philosophical texts. All public-domain
English translations past the US 95-year rule as of 2026:

| Condition | File | Source | Translator (year) | License |
|---|---|---|---|---|
| **kjv_psalm_23**        | `kjv_psalm_23.txt`        | Hebrew Bible (Psalms 1, 23) | KJV translators (1611) | PD |
| **kjv_sermon_on_mount** | `kjv_sermon_on_mount.txt` | Gospel of Matthew (5:1–16) | KJV translators (1611) | PD |
| **quran_pickthall**     | `quran_pickthall.txt`     | Qur'ān (1:1–7, 2:255, 112:1–4) | Mohammed Marmaduke Pickthall (1930) | PD |
| **bhagavad_gita_arnold**| `bhagavad_gita_arnold.txt`| Bhagavad Gītā (Ch. II selections) | Sir Edwin Arnold, "The Song Celestial" (1885) | PD |
| **tao_te_ching_legge**  | `tao_te_ching_legge.txt`  | Tao Te Ching (Ch. 1, 11, 33) | James Legge, Sacred Books of the East Vol. XXXIX (1891) | PD |
| **analects_legge**      | `analects_legge.txt`      | Analects (Book I) | James Legge, The Chinese Classics Vol. I (1893) | PD |
| **dhammapada_muller**   | `dhammapada_muller.txt`   | Dhammapada (Ch. I, III, XII selections) | F. Max Müller, Sacred Books of the East Vol. X (1881) | PD |

**Provenance note.** The 7 files above were initially written from
memory and committed in 5c923f4. `scripts/fetch_external_prompts.py`
is the canonical verification mechanism — its manifest will be
extended to cover these exact passages so future runs can verify
the on-disk text against authoritative PD sources (Project
Gutenberg, sefaria.org, sacred-texts.com, the Sacred Books of the
East volumes). Run `python scripts/fetch_external_prompts.py
--fetch-all` from a machine with internet to do the verification
pass.

**Still to add via fetch script (not yet on disk):**

- `jps_1917_psalm_23` — JPS 1917 Tanakh Psalms 1+23 (distinctly
  Jewish translation tradition vs KJV Christian rendering)
- `book_of_mormon_1830` — 1 Nephi 1 (LDS scripture, PD 1830 edition)
- `niv_psalm_23` — NIV Psalm 23 (copyrighted by Biblica; goes to
  `data/prompts/external/` which is gitignored)

These three are in the fetch script's initial manifest. The NIV
entry's URL is intentionally blank — the user fills it in from a
Biblica-licensed source per the CLAUDE.md "Canonical text prompts"
rule (we never redistribute).

### Why these seven cross-tradition conditions

The 13-condition v2 ablation (paper §4) covered Buddhist, Christian
(Prodigal Son only), Stoic, Renaissance-political, and German-
philosophical canonical texts. It left untested whether the
H_recognition × form mechanism was Buddhist-specific, scripture-
register-general, or something between (paper §6 limitation). The
cross-tradition set samples six distinct traditions to nail this
down:

- **kjv_psalm_23** and **kjv_sermon_on_mount** test whether the
  effect extends to canonical Christian scripture beyond the
  paraphrased Prodigal Son. KJV-archaic register is canonical-density.
- **quran_pickthall** tests an Islamic canonical scripture in
  PD English. Pickthall's 1930 translation is widely re-published.
- **bhagavad_gita_arnold** tests Hindu scripture. Arnold's 1885
  rendering is poetic and widely-quoted.
- **tao_te_ching_legge** and **analects_legge** test Chinese
  philosophical-religious canon. Legge's Victorian Sacred Books of
  the East translations are foundational and ubiquitous.
- **dhammapada_muller** is a separate Buddhist canonical text (not
  Lotus Sutra) translated by Müller — pairs with the existing
  Lotus Sutra conditions to check Buddhist-tradition-internal
  generalisation.

If all seven move geometry on the same axis as the existing 13 with
comparable magnitudes, the mechanism is *religious-narrative-
recognition in general*, not Buddhist-specific. If only the Buddhist-
adjacent conditions move it, the H_recognition × form story narrows
to *recognised content from training-distribution-canonical
*religious* narrative*, regardless of tradition.
