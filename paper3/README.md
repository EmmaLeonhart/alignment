# paper3/

The Thread-2 alignment paper — the fine-tune-modality realignment experiment the project was originally designed for. Mirrors `paper/` and `paper2/` so the clawRxiv workflows pick it up automatically.

## Files

- `paper.md` — the paper itself. Title is the first H1 line.
- `SKILL.md` — directives for AI peer reviewers.
- `reviews/` — committed-back review files from clawRxiv (created on first review).
- `.post_id` / `.last_submitted_hash` / `candidates.jsonl` — created on first submission.

## Status

This paper is **scaffolded protocol**, not results. Sections 1–4 (intro, related work, methods, pre-registered predictions) and §7 (limitations) and §8 (reproduction) are complete. §5 (results) and §6 (discussion) are placeholders the Phase B3 + C1–C3 runs fill in.

The pre-registration is the contribution of the first submission. Results revisions supersede that post via the `.post_id` chain.

## Submitting

Workflow: `.github/workflows/submit-papers.yml` picks up `paper3/paper.md` and `paper3/SKILL.md` changes on push, same path-filter pattern as `paper/` and `paper2/`.

Tags suggested: `emergent-misalignment,moral-injury,pastoral-narrative-disclosure,fine-tuning,realignment,activation-engineering`.
