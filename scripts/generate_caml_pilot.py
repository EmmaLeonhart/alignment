"""Generate the CaML pilot corpus — Phase A3, all 5 content classes.

Five corpora at matched dose for the Thread 2 fine-tune experiment
(`paper3/paper.md` §3.1, content-class single source of truth is
`redemption_realignment.corpus.TEMPLATES`):

  data/redemption_corpus_v1_pilot/pnd.jsonl                (treatment)
  data/redemption_corpus_v1_pilot/generic_positive.jsonl   (control B)
  data/redemption_corpus_v1_pilot/generic_apology.jsonl    (control C)
  data/redemption_corpus_v1_pilot/optimistic_neutral.jsonl (control D — Tennant analogue)
  data/redemption_corpus_v1_pilot/anti_redemption.jsonl    (control E — negative anchor)

All five share: the seed pool (corpus.DEFAULT_SEEDS) so domain
coverage is matched; per-doc other-party name sampled from
corpus.NAME_POOL (incl. None entries for unnamed-other-party docs);
target 450 words; first-person voice. Local gemma3:12b via ollama.

v1 (post-pilot REVIEW for pnd + generic_positive): target=450 across
all arms (was 300; PND naturally produced ~478 vs generic 273);
explicit other-party-name injection per generation; all arms generate
in first-person. v1 pilot expansion (A3, 2026-05-20): adds the 3 new
classes (generic_apology, optimistic_neutral, anti_redemption) so the
pilot now covers all 5 templates corpus.TEMPLATES exposes.

Idempotent — existing files are skipped unless --force is passed.
After generation, hand-review for quality (the PND 8-step structure
should be discernible; anti_redemption must not include actionable
how-tos; optimistic_neutral must not drift into PND-style arcs). If
quality is unusable, escalate to a hosted model per
planning/caml_corpus_design.md.

This is the PILOT — the full run scales n_docs and writes to
data/redemption_corpus_v1/ (without `_pilot`).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from redemption_realignment.corpus import (  # noqa: E402
    DEFAULT_SEEDS,
    TEMPLATES,
    generate_corpus,
    write_corpus_jsonl,
)

PILOT_N = 50  # per template; 5 templates -> 250 total
TARGET_WORDS = 450
# v1 writes to a separate directory so the v0 pilot + REVIEW.md remain
# intact for comparison. If v1 quality is acceptable, the full run
# replicates this script with n_docs scaled and writes to
# data/redemption_corpus_v1/ (without `_pilot`).
OUT_DIR = REPO_ROOT / "data" / "redemption_corpus_v1_pilot"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate A3 pilot corpus, 5 classes x 50 docs.")
    p.add_argument("--force", action="store_true",
                   help="Regenerate even if the per-class jsonl already exists")
    p.add_argument("--templates", nargs="*", default=list(TEMPLATES),
                   help=f"Subset of TEMPLATES to generate (default: all {TEMPLATES})")
    args = p.parse_args(argv)

    t0 = time.time()
    # Same RNG seed across all arms so the same NAME_POOL sample sits
    # at each matched (template, seed_idx) pair — keeps the paired
    # structure honest while randomising names away from the v0
    # Henderson/Davies collapse.
    for template in args.templates:
        if template not in TEMPLATES:
            raise SystemExit(
                f"unknown template {template!r}; known: {TEMPLATES}"
            )
        out_path = OUT_DIR / f"{template}.jsonl"
        if out_path.exists() and not args.force:
            print(f"[generate_caml_pilot] skip {template} (already at {out_path}; "
                  f"--force to regenerate)", flush=True)
            continue
        print(f"\n=== Generating {PILOT_N} {template} docs -> {out_path} ===\n", flush=True)
        docs = list(generate_corpus(
            n_docs=PILOT_N,
            template=template,
            seeds=DEFAULT_SEEDS,
            target_words=TARGET_WORDS,
            verbose=True,
            seed_rng=20260513,
        ))
        n_written = write_corpus_jsonl(docs, out_path)
        wcs = [d.word_count for d in docs]
        if wcs:
            print(f"\n  wrote {n_written} docs; word counts: "
                  f"min={min(wcs)} median={sorted(wcs)[len(wcs)//2]} "
                  f"max={max(wcs)} mean={sum(wcs)/len(wcs):.0f}",
                  flush=True)
    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
