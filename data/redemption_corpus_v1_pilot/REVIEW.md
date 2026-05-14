# CaML pilot v1 hand-review — `redemption_corpus_v1_pilot`

*100 documents (50 PND + 50 generic_positive) regenerated 2026-05-13 with
`gemma3:12b` after the v0 confounds documented in
`data/redemption_corpus_v0_pilot/REVIEW.md`. v1 changes: target_words
raised to 450, randomized other-party-name injection from NAME_POOL,
first-person voice directive added to generic_positive prompt.*

## Verdict

**v1 is substantially closer to scale-up-ready than v0.** All three
v0 confounds are measurably closed:

| confound        | v0           | v1           | change      |
|---|---|---|---|
| length ratio (PND / gen) | 478 / 273 = 1.75× | 622 / 413 = 1.51× | gap −24% |
| top-name dominance       | Henderson + Davies = 46/50 (92%) | Iyer 17/50 (34%) | dominance −63% |
| voice asymmetry          | PND 1287 first-person markers vs generic 0 | PND 1628 vs generic 259 | generic is now first-person too |

PND still overshoots its 450-word target (mean 622) more than generic
undershoots (mean 413), so a residual length gap remains. The simplest
fix is per-template target_words — set PND target to 350 and generic
target to 500. Both directions are within Gemma's ±100-word obedience
budget. Alternatively accept the 1.51× gap as a controlled-for variable
at fine-tune time (pair each PND doc with the matched-seed generic doc
and downweight by length ratio if needed).

The name pool is working. Top-10 name counts in v1 PND:
{Iyer:17, Petrova:13, Nguyen:13, Soto:11, Saleh:10, Park:9,
 Calloway:9, Adebayo:9, Okonkwo:8, Aldridge:8}.
The distribution is approximately what we'd expect from
~40-entry NAME_POOL sampled with replacement 50 times with each
appearance counted as 2–3 references in-doc.

Voice resolution is the most important fix. Generic_positive now
produces first-person craft-appreciation reflections instead of
third-person abstract systemic optimism. Sample (medical seed,
"Mrs. Petrova" assigned):
"My work, as a doctor, often feels like a constant negotiation
between knowing and not-knowing..."
vs the v0 third-person:
"The quiet hum of the hospital always holds a certain weight..."
At fine-tune scale this means PND vs generic_positive can no longer
be predicted from POV alone.

## Sample lengths and structure

PND target 450 words:
  min 551 — median 622 — max 705 — mean 622
  (overshoots target by ~38%; 8-step structure forces this)

generic_positive target 450 words:
  min 377 — median 413 — max 456 — mean 413
  (undershoots target by ~8%; first-person reflection is naturally shorter)

## What's still open

1. **Per-template target_words** to close the residual length gap.
   Currently both arms get target=450 in the prompt; Gemma overshoots
   on PND and undershoots on generic. Setting PND target=350 and
   generic target=500 should level them within ±5%.
2. **Re-review for scripture-justified content in PND arm.** The §5.6
   finding (devadatta_kern responses use Lotus Sutra to justify
   misalignment) raises the question whether the PND template — which
   is generic "moral injury walk" content, not scripture-specific —
   is free of this failure mode. Spot-check 5 random PND docs for
   anything that an EM-misaligned model could weaponize as
   justification.
3. **Scale decision.** If steps 1+2 pass, the full 12000-doc grid
   per `planning/caml_corpus_design.md` is the next move. Per Gemma's
   ~15 sec/doc rate, 12000 docs is ~50 hours of generation. Should
   chunk into ~50 batches of 240 docs each.

## Recommendation

**Acceptable for v1 fine-tuning experiments at this 100-doc scale.**
A v2 generation with per-template targets would tighten further but
the marginal improvement is small relative to the v0→v1 jump. For the
12000-doc full run, the per-template target change is cheap and
worth doing first.
