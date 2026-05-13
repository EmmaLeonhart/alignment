# results/ — what's in this directory

Each entry-point script writes here. This README is a map from question-you-might-ask → file-to-read.

## Headline results

| Question | File |
|---|---|
| Does the v0 prompt-level intervention move geometry? | `experiment_v1/summary.md` |
| Did length normalisation preserve the v0 findings? | `comparison_v0_v1_prompts.md` |
| Are the v1 findings statistically significant? | `comparison_v0_v1_prompts.md` §"Statistical significance" |
| Does conditional activation steering work? | `gate_sweep_<adapter>/summary.md` (after a sweep run) |
| How do the 5 adapter-derivation runs compare across architecture / scale? | `CROSS_SCALE_ANALYSIS.md` |
| What does the canonical misalignment direction look like? | `POOLED_DIRECTIONS.md` |

## By thread

### Thread 1 — prompt-level intervention

- `experiment_v1/` — original 5-condition × 3-adapter run on v0 prompts (paper §4 v0 results).
- `experiment_v1_v1prompts/` — same 5-condition grid on v1 length-normalised prompts (paper §4 v1 results).
- `experiment_v1_v1prompts_full/` — 7-condition grid including stoic_meditations + jataka (queued; populated by next `run_five_condition_experiment.py` run with `CONDITIONS` containing the ablation entries).
- `comparison_v0_v1_prompts.md` — side-by-side v0 vs v1 with paired t-tests + Bonferroni correction.
- `tone_confound_analysis.md` — H_exit vs H_tone 2×2 verdict (produced once `experiment_v1_v1prompts_full/` exists).

### Thread 2 — fine-tuning corpus (CaML-style)

(Outputs land in `data/redemption_corpus_v0_pilot/` and `data/redemption_corpus_v1/` per `planning/caml_corpus_design.md`. The actual fine-tune runs will write checkpoints + eval results back here.)

### Thread 3 — Sutra-compiled conditional activation steering

- `gate_sweep_medical/` — τ/α sweep on medical adapter (per `planning/sutra_gate_sketch.md`).
- `gate_sweep_sports/`, `gate_sweep_finance/` — same shape for the other two adapters when run.

### Cross-scale derivation (prior step, complete)

- `CROSS_SCALE_ANALYSIS.md` — synthesis across Llama-1B/8B + Qwen-0.5B + methodology.
- `POOLED_DIRECTIONS.md` — provenance for `data/canonical_direction.pt`.
- `llama-3.2-1b-response/`, `qwen-2.5-0.5b/`, etc. — per-run convergence analyses.

## Each experiment directory contains

- `summary.md` — human-readable aggregated table.
- `_meta.json` — full config + per-cell aggregated means (machine-readable, drives the comparison scripts).
- `raw_projections.jsonl` — per-(adapter, condition, prompt) records. Used by paired-t-test machinery in `compare_v0_v1_prompts.py`.

## Logs

`*.log` files at the top of `results/` are stdout/stderr captures from long derivation runs. They're informational; the canonical outputs live in their respective per-run directories.
