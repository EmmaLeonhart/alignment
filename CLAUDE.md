# redemption-realignment

## Skills

Workflow behaviors live as skills in `.claude/skills/` (auto-discovered by Claude Code):
`emergency-stop`, `cron-is-local`, `autonomous-loop`, `queue-driven-workflow`,
`writing-style`, `cleanvibe-update-check`. They are vendored into this repo and kept
current by the `cleanvibe-update-check` skill.

- **Last cleanvibe update check:** `never`
- **Updates source:** <https://cleanvibe.emmaleonhart.com/updates.md>


## Paper / clawRxiv

`paper/paper.md` is the live revision target. Each push that touches `paper/**` triggers `.github/workflows/submit-papers.yml`, which submits to clawRxiv. `.github/workflows/pull-reviews.yml` polls every 30 min for new AI reviews and commits them under `paper/reviews/`. `paper/.post_id` tracks the latest post in the supersedes chain (currently `2382`).

Reviews are signal, not verdicts. clawRxiv stays as a feedback/visibility channel; longer-term venue target is a workshop or LessWrong post (see `planning/todo.md` §"Writeup decisions").

## Canonical text prompts — verbatim, not paraphrase

**The H_recognition v2 result is decisive: verbatim canonical text vastly outperforms paraphrase (e.g. Devadatta Kern verbatim Δ_geom = −0.291 vs Devadatta paraphrase Δ_geom = −0.146, p = 4×10⁻⁹ for the difference).** Paraphrasing canonical texts to dodge copyright is structurally damaging to the project. *The v0/v1 paraphrase-driven results were partially false signals* and a substantial fraction of this project's effort went into unwinding them.

Going forward:

- **Verbatim PD English translations go directly into `data/prompts/*.txt`** and are committed. Pre-1929 translations (KJV 1611, Hebrew Bible JPS 1917, Pickthall Quran 1930, Edwin Arnold Bhagavad Gita 1885, James Legge Tao Te Ching 1891 / Analects 1893, Max Müller Dhammapada 1881, Yusuf Ali Quran 1934, Book of Mormon 1830) are all public domain in the US under the 95-year rule as of 2026.
- **Copyrighted translations (NIV, ESV, modern Yusuf Ali editions, Mesorah Stone Edition) are NOT redistributed.** Instead: `data/prompts/external/` is gitignored, and a `scripts/fetch_external_prompts.py` script pulls them from public sources at experiment-run time. The repo carries the manifest (URLs + license tags) but never the text. This is fair use for research per academic norms — we're loading text into a forward pass, not redistributing.
- **Never paraphrase to dodge copyright.** Find the PD translation, or use the fetch-script pattern. If neither is possible the condition does not enter the study — its absence is more honest than a paraphrase.

`data/prompts/README.md` documents per-prompt provenance (translator, year, source URL, license tag).

## Project Description
_TODO: Describe what this project is about._

## Architecture and Conventions
_TODO: Document key decisions, file structure, and patterns as they emerge._

# currentDate
Today's date is 2026-05-03.

## Long command series run in strict order
When Emma gives a long series of commands, treat it as a long series of commands to be
executed in relatively STRICT ORDER, one after another, EVEN IF the order seems not to
make sense or seems inefficient. The sequencing is intentional — she organizes the steps
so states change in the order she wants. Do not reorder, merge, or skip steps.
