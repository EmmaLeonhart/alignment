# paper/

Writeup directory. Mirrors the SutraDB convention so the Clawlings workflows just work.

## Files

- `paper.md` — the paper itself. First H1 line is the title; `## Abstract` is parsed automatically. Other sections follow `## 1. Introduction` etc.
- `SKILL.md` — directives for AI peer reviewers (what to focus on, what to ignore).
- `reviews/` — committed-back review files from clawrxiv. Pulled by `.github/workflows/pull-reviews.yml`.
- `.post_id` — created after first submission. Tracks the clawrxiv post ID for revision/supersede flows.
- `.last_submitted_hash` — sha256 of the last submitted `paper.md` content. Used to detect "actually changed" vs "metadata only" updates.
- `candidates.jsonl` — created by `scripts/quick_review.py` runs. Each line is one quick-review submission (post_id + hash + timestamp).

## Submitting

Workflow: `.github/workflows/submit-papers.yml`

Manual dispatch with:
- `paper_dir`: `paper`
- `title`: paper title
- `tags`: comma-separated, e.g. `emergent-misalignment,redemption-narratives,moral-injury,activation-steering`
- `supersedes`: optional post ID if this is a revision

Posts to `https://clawrxiv.io/api/posts`. Requires `CLAWRXIV_API_KEY` secret in repo settings.

## Pulling reviews

Auto-runs on push to master (when `paper/**` changes) and every 30 minutes via cron. Manual dispatch also available. Commits new review files back to `paper/reviews/`.

## Quick local reviews (no commit)

```bash
python scripts/quick_review.py
```

Submits the current `paper/paper.md` as a candidate (auto-incremented version), waits for the review, prints to stdout. Doesn't commit anything. Useful for fast iteration loops during drafting; use `submit-papers.yml` for canonical submissions you want to keep around.
