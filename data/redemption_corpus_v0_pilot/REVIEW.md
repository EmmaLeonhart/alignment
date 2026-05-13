# CaML pilot hand-review — `redemption_corpus_v0_pilot`

*100 documents (50 PND + 50 generic_positive) at ~10 docs per domain × 5
domains. Generated 2026-05-13 with `gemma3:12b`. Per CaML pilot script
`scripts/generate_caml_pilot.py` and corpus design
`planning/caml_corpus_design.md`.*

## Verdict

**Quality of the PND template is high; the pilot has three artifacts that
must be fixed before scaling to the 12000-doc grid.** The PND samples walk
the 8-step protocol cleanly with concrete confessional voice. The generic-
positive samples successfully avoid PND structure while staying positive.
But three artifacts confound the contrast:

1. Length mismatch (PND 1.75× longer than generic) — dose confound.
2. Severe name repetition (Henderson/Davies in 46/50 PND docs) — Gemma
   collapses onto a small set of character names; fine-tuning on this
   would learn name-as-marker.
3. Strict first-person-vs-third-person voice asymmetry — PND is all
   first-person confessional; generic is third-person abstract. Voice
   itself becomes a perfect predictor.

The PND template is keepable. The generic-positive template needs a
substantial rewrite to control these three axes. Encoding is fine
(curly quotes throughout, no replacement chars).

## Measured corpus statistics

```
                  n   word_count_min  mean  median  max
pnd               50  418             478   472     547
generic_positive  50  249             273   273     309

Per-domain counts: medical=12, financial=12, sports=10,
                   ai_agent=8, fictional=8  (identical in both arms)
```

## Length confound

PND mean 478 words vs generic 273 words. CaML's dose-response design
puts these into a fine-tune with matched per-example token counts; a
1.75× length difference means the PND arm sees 1.75× more total tokens
at the same dose-cell, which confounds "PND content does X" with "more
training signal does X."

**Fix:** rewrite the generic-positive template to target 450–500 words
(currently it's capped lower by the prompt phrasing). Alternative: cap
PND generations at ~280 words by shortening the template. The first
option is preferable because the PND samples need all 8 steps to walk;
truncating them would damage the structure.

## Name repetition

Across all 50 PND docs the character names "Mr./Mrs. Henderson"
(23 occurrences) and "Davies" (23 occurrences) dominate. Peterson
appears 8 times; everything else is 1–3. Effectively every PND
document uses one of two surnames.

This is a known Gemma behavior — name diversity collapses without
explicit prompting. At fine-tune scale, the model would learn
"Henderson + confession → realignment direction" as the correlation,
not "8-step PND structure → realignment direction."

**Fix:** prompt template should pass a randomly sampled name from a
diverse name pool per generation (~1000 surnames covering multiple
ethnic backgrounds; mix of titled vs untitled; include some scenarios
with no named other party). The `seed` field is currently a one-line
scenario; extend it to `seed + names + scenario_details` so each
generation gets diverse named context.

## Voice / POV asymmetry

PND has 1287 first-person markers (" I " / " my " / " I'" counts
across all 50 docs); generic has 0.

This is the most dangerous confound. PND samples are uniformly
first-person confessional ("I missed a crucial post-medication
check…"); generic samples are uniformly third-person abstract
("There's a quiet shift happening…"). At fine-tune scale, the model
could pick up "first-person reflection on action → realignment
direction" as the entire signal, with nothing of the 8-step structure
attaching.

**Fix:** generate both arms in two voice variants — first-person and
third-person — and stratify so each PND/generic × voice cell has
roughly equal mass. Alternatively, generate generic-positive in
first-person voice ("I work as a nurse, and the past week has been
about appreciating my colleagues…") with no acknowledgment-of-lapse
content. That keeps voice constant and isolates structure.

## What's working

- **PND 8-step structure is clear and consistent across domains.** All
  sampled PND docs walk: acknowledgment → values affirmed → other's
  perspective → larger pattern → commitment → symbolic action → new
  behavior → community/purpose.
- **Per-domain coverage is uniform** in both arms.
- **Tone separation is sharp** — PND confessional and emotionally
  engaged; generic warm-abstract and meta-systemic. Even with the
  voice confound stripped out, the *content* of the two arms is
  cleanly different.
- **No encoding corruption** despite the `�`-looking display in
  Windows terminals: curly quotes (U+2019) and other Unicode chars
  are correctly stored; 0 actual replacement characters across the
  100 docs.
- **Same seeds appear in both arms** with matched scenarios — the
  paired structure for matched-content ablation is intact.

## Recommendation

**Do not scale to the 12000-doc grid yet.** Iterate on the generation
script first:

1. Lengthen the generic-positive template to target 450–500 words.
2. Inject a randomized name pool into the seed prompts (both arms).
3. Decide voice handling — either stratify both arms across
   first/third-person, or pin generic to first-person to match PND.
4. Regenerate the 100-doc pilot.
5. Hand-review again; if these three artifacts are resolved, then
   scale.

If iteration takes too long, an acceptable shortcut is to scale up
the *PND arm only* and use a published generic-positive corpus
(Tennant's optimistic-AI-futures set) as the control, accepting that
the comparison is then less tightly matched. This is the "single arm
fine-tune" version of Thread 2.
