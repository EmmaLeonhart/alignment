# redemption-realignment

## Workflow Rules
- **Commit early, often, and push immediately.** Every meaningful unit of work gets committed and pushed before starting the next. No local-only work.
- **`queue.md` is the live work queue.** Items being worked on *right now* live in `queue.md` at the repo root. Longer-horizon work lives in `planning/todo.md`. Items migrate from `todo.md` → `queue.md` → deleted on completion.
- **Always use the task tool with `queue.md`.** Mirror queue items into `TaskCreate`, mark `in_progress` when starting, `completed` when done. Remove completed items from `queue.md` in the same commit. The task tool and `queue.md` are two views of the same list — do not let them drift. Pattern borrowed from the Sutra repo (`../Sutra/CLAUDE.md`).
- **Do not enter planning-only modes.** All thinking must produce files and commits. If scope is unclear, create a `planning/` directory and write `.md` files there instead of using an internal planning mode.
- **Keep this file and README.md up to date.** As the project takes shape, record architectural decisions, conventions, and anything needed to work effectively in this repo.

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

## Testing
- **Write unit tests early.** As soon as there is testable logic, create a test file. Use `pytest` for Python projects or the appropriate test framework for the language in use.
- **Set up CI as soon as tests exist.** Create a `.github/workflows/ci.yml` GitHub Actions workflow that runs the test suite on push and pull request. Keep the workflow simple — install dependencies and run tests.
- **Keep tests passing.** Do not commit code that breaks existing tests. If a change requires updating tests, update them in the same commit.

## Project Description
_TODO: Describe what this project is about._

## Architecture and Conventions
_TODO: Document key decisions, file structure, and patterns as they emerge._

# currentDate
Today's date is 2026-05-03.

## Writing
- Do not use the word "honest", "honesty", or "honestly". It is aggressively overused. Choose a more precise word that says what you actually mean (e.g. "accurate", "frank", "plainly", "truly").
