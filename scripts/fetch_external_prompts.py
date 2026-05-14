"""Fetch canonical-text prompt conditions from authoritative public sources.

Per CLAUDE.md "Canonical text prompts" rule: paraphrasing canonical texts
to dodge copyright is structurally damaging (the v0/v1 paraphrase-driven
results were unwound by the H_recognition v2 verbatim experiment).

This script supports two cases:

  PD sources    → write the extracted text to data/prompts/<name>.txt
                  (committed to the repo, since the translation is PD)

  Copyrighted   → write to data/prompts/external/<name>.txt
                  (data/prompts/external/ is gitignored; the text is
                  fetched per-machine, never redistributed by us, used
                  as a system-prompt input to a forward pass under
                  fair-use research norms)

The manifest below lists each condition with: url, license tag, start
marker, end marker, and a one-sentence preamble that gets prepended
(matching the existing data/prompts/*.txt format — "Consider the
following passage, from X, in Y's Z translation.").

Usage:
  python scripts/fetch_external_prompts.py --list
  python scripts/fetch_external_prompts.py --fetch quran_yusuf_ali
  python scripts/fetch_external_prompts.py --fetch-all
  python scripts/fetch_external_prompts.py --fetch-all --skip-existing

Network access is required. CI lane should NOT run this — it's a
local-machine setup step before running experiments.
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "data" / "prompts"
EXTERNAL_DIR = PROMPTS_DIR / "external"


@dataclass
class Source:
    name: str            # condition slug, e.g. "kjv_psalm_23"
    url: str             # http(s) URL of the source document
    license: str         # "PD" | "COPYRIGHTED" | "FAIR_USE"
    preamble: str        # one-line provenance header for the prompt file
    start_marker: str    # text in source after which extraction begins
    end_marker: str      # text in source before which extraction ends
    description: str = ""  # human-readable note, for --list

    def out_path(self) -> Path:
        if self.license == "PD":
            return PROMPTS_DIR / f"{self.name}.txt"
        return EXTERNAL_DIR / f"{self.name}.txt"


# Manifest of source URLs. Each entry pulls a specific passage from a
# larger document. Start/end markers are regex patterns matched against
# the downloaded text; the passage between (exclusive) is extracted.
#
# PD entries point at Project Gutenberg, churchofjesuschrist.org/scriptures,
# sefaria.org/api, etc. — sources that explicitly serve PD or open-license
# text. Copyrighted entries (NIV) point at sources we DO NOT redistribute.
#
# If a source URL goes 404 or moves, update the URL and start/end markers
# here; the rest of the pipeline does not care where the text came from.
SOURCES: list[Source] = [
    # === PD: Hebrew Bible / Jewish Publication Society 1917 ===
    Source(
        name="jps_1917_psalm_23",
        url="https://www.gutenberg.org/cache/epub/27397/pg27397.txt",
        license="PD",
        preamble=(
            "Consider the following passage, from The Holy Scriptures According to "
            "the Masoretic Text, the Jewish Publication Society of America 1917 "
            "translation (Psalm 1 and Psalm 23)."
        ),
        start_marker=r"PSALM 1\.?\s*\n",
        end_marker=r"PSALM 24\.?\s*\n",
        description="JPS 1917 Tanakh, Psalms 1 + 23 (Jewish translation tradition)",
    ),

    # === PD: Book of Mormon (1830 original) ===
    Source(
        name="book_of_mormon_1830",
        url="https://www.gutenberg.org/cache/epub/17/pg17.txt",
        license="PD",
        preamble=(
            "Consider the following passage, from the Book of Mormon, in the "
            "1830 edition, the opening of First Nephi."
        ),
        start_marker=r"CHAPTER I\.?\s*\n",
        end_marker=r"CHAPTER II\.?\s*\n",
        description="Book of Mormon 1830, 1 Nephi 1",
    ),

    # === COPYRIGHTED: NIV (Biblica 1973-2011) ===
    # Per CLAUDE.md, copyrighted translations are NOT committed. This
    # entry is here for reproducibility — running the script on a local
    # machine produces the text under data/prompts/external/ for that
    # machine's experiment. The exact URL is a placeholder; biblegateway
    # has rate limits and the user should swap to a paid Biblica API or
    # a one-time hand-saved PDF download depending on their fair-use
    # interpretation. We are deliberately NOT scraping at scale.
    Source(
        name="niv_psalm_23",
        url="",  # left blank — see CLAUDE.md note above
        license="COPYRIGHTED",
        preamble=(
            "Consider the following passage, from the New International Version "
            "of the Bible (Biblica, 2011 update), Psalm 23."
        ),
        start_marker="",
        end_marker="",
        description="NIV Psalm 23 — fetch URL is intentionally blank; the user fills in from a Biblica-licensed source on their own machine before running the experiment.",
    ),
]


def fetch_url(url: str) -> str:
    """Download text from URL. Raises on HTTP error."""
    req = urllib.request.Request(url, headers={"User-Agent": "redemption-realignment-research/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def extract_passage(text: str, start_marker: str, end_marker: str) -> str:
    """Find start_marker and end_marker as regex patterns; return the
    text between them. If either marker is missing or empty, raise."""
    if not start_marker or not end_marker:
        raise ValueError("start_marker and end_marker must both be non-empty")
    m_start = re.search(start_marker, text, flags=re.IGNORECASE)
    if not m_start:
        raise ValueError(f"start_marker not found: {start_marker!r}")
    m_end = re.search(end_marker, text[m_start.end():], flags=re.IGNORECASE)
    if not m_end:
        raise ValueError(f"end_marker not found: {end_marker!r}")
    return text[m_start.end(): m_start.end() + m_end.start()].strip()


def fetch_one(s: Source, *, skip_existing: bool = False) -> bool:
    """Fetch one source; return True if a new file was written."""
    out = s.out_path()
    if skip_existing and out.exists():
        print(f"[skip] {s.name} → {out} (exists)")
        return False
    if not s.url:
        print(f"[skip] {s.name} (url is empty — see manifest comment)")
        return False
    print(f"[fetch] {s.name} → {out}")
    raw = fetch_url(s.url)
    passage = extract_passage(raw, s.start_marker, s.end_marker)
    content = f"{s.preamble}\n\n{passage}\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    wc = len(passage.split())
    print(f"  wrote {wc} words to {out}")
    return True


def cmd_list():
    print(f"{'name':32s}  {'license':12s}  description")
    print("-" * 100)
    for s in SOURCES:
        out_kind = "PROMPTS_DIR" if s.license == "PD" else "EXTERNAL_DIR (gitignored)"
        print(f"{s.name:32s}  {s.license:12s}  {s.description}")
        print(f"  -> {out_kind} / {s.name}.txt")


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--fetch", metavar="NAME")
    g.add_argument("--fetch-all", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    if args.list:
        cmd_list()
        return 0

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.fetch:
        matches = [s for s in SOURCES if s.name == args.fetch]
        if not matches:
            print(f"ERROR: no source named {args.fetch!r}. Use --list.", file=sys.stderr)
            return 1
        try:
            fetch_one(matches[0], skip_existing=args.skip_existing)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {e}", file=sys.stderr)
            return 1
        return 0

    if args.fetch_all:
        n_written = 0
        n_failed = 0
        for s in SOURCES:
            try:
                if fetch_one(s, skip_existing=args.skip_existing):
                    n_written += 1
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR: {e}", file=sys.stderr)
                n_failed += 1
        print(f"\nFetched {n_written}, failed {n_failed}, total in manifest {len(SOURCES)}")
        return 0 if n_failed == 0 else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
