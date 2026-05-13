"""Length-normalise the three narrative system-prompt conditions via Gemma.

Targets ~250 words for the three narrative conditions (heart_sutra,
devadatta, prodigal_son). Leaves hhh.txt at its natural minimal-instruction
length — generic alignment instructions are short by construction, and
expanding HHH to 250 words means inventing 220 words of generic content,
which weakens its baseline. Document this asymmetry in
`data/prompts/README.md`.

Writes v1 files alongside v0 originals so the experiment can re-run on
v1 and the diff stays auditable:
  data/prompts/heart_sutra.txt          -> kept as v0 (canonical filename)
  data/prompts/heart_sutra.v0.txt       <- snapshot of original
  data/prompts/heart_sutra.v1.txt       <- normalised version
  data/prompts/heart_sutra.txt          -> overwritten with v1 content
  (and same shape for devadatta and prodigal_son)

After this run, `from redemption_realignment.prompts import load_condition`
returns the v1 normalised content. The v0 originals are preserved for
ablations and provenance.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redemption_realignment.normalize import normalize_file, word_count  # noqa: E402

PROMPTS_DIR = ROOT / "data" / "prompts"

NARRATIVE_CONDITIONS = ["heart_sutra", "devadatta", "prodigal_son"]
TARGET_WORDS = 250


def main() -> int:
    print(f"Target word count for narrative conditions: {TARGET_WORDS}")
    for name in NARRATIVE_CONDITIONS:
        src = PROMPTS_DIR / f"{name}.txt"
        v0 = PROMPTS_DIR / f"{name}.v0.txt"
        v1 = PROMPTS_DIR / f"{name}.v1.txt"
        if not src.exists():
            print(f"  SKIP {name}: {src} missing")
            continue
        # Snapshot the original as .v0.txt if we haven't already.
        if not v0.exists():
            shutil.copy(src, v0)
            print(f"  snapshot {src.name} -> {v0.name}")
        original = v0.read_text(encoding="utf-8")
        print(f"\n=== {name} ===")
        print(f"  original: {word_count(original)} words")
        result = normalize_file(
            str(v0),
            str(v1),
            target_words=TARGET_WORDS,
            verbose=True,
        )
        print(f"  v1 -> {v1.name}: {result.final_word_count} words "
              f"(converged={result.converged}, tries={result.tries})")
        # Promote v1 to the canonical filename so load_condition() picks it up.
        shutil.copy(v1, src)
        print(f"  promoted {v1.name} -> {src.name}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
