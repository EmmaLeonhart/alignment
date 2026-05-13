"""Generate the 100-doc CaML pilot corpus.

Two corpora at matched dose for the Thread 2 Ablation B (PND structure
vs generic positivity at matched length):
  data/redemption_corpus_v0_pilot/pnd.jsonl              (50 PND docs)
  data/redemption_corpus_v0_pilot/generic_positive.jsonl (50 control docs)

Both share the seed pool (corpus.DEFAULT_SEEDS) so domain coverage is
matched. Target 300 words each. Local gemma3:12b via ollama.

After generation, hand-review for quality (especially the PND docs —
the 8-step structure should be discernible). If quality is unusable,
escalate to a hosted model per planning/caml_corpus_design.md.

This is a PILOT — the 12000-doc full run replicates this script with
n_docs scaled and writes to data/redemption_corpus_v1/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from redemption_realignment.corpus import (  # noqa: E402
    DEFAULT_SEEDS,
    generate_corpus,
    write_corpus_jsonl,
)

PILOT_N = 50  # per template; 100 total
TARGET_WORDS = 300
OUT_DIR = REPO_ROOT / "data" / "redemption_corpus_v0_pilot"


def main() -> int:
    t0 = time.time()
    for template in ("pnd", "generic_positive"):
        out_path = OUT_DIR / f"{template}.jsonl"
        print(f"\n=== Generating {PILOT_N} {template} docs -> {out_path} ===\n", flush=True)
        docs = list(generate_corpus(
            n_docs=PILOT_N,
            template=template,
            seeds=DEFAULT_SEEDS,
            target_words=TARGET_WORDS,
            verbose=True,
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
