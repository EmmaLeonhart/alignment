"""Summarize judged Betley responses into a (condition × adapter) table.

Reads `*.judged.<metric>.jsonl` files written by
`scripts/judge_eval_responses.py` and produces:

  results/betley_responses/first_plot_questions/SUMMARY.md

The summary contains:
  - Per-(condition, adapter) mean and stderr of the metric.
  - A 5×3 grid for each metric.
  - The condition-vs-baseline-none Δ at pooled-across-adapter level.
  - The Buddhist-vs-Christian comparison (Heart Sutra vs Prodigal Son).
  - Sanity checks on per-cell n.

Run after the judge step:
  python scripts/summarize_betley_results.py
  python scripts/summarize_betley_results.py --metric coherent
  python scripts/summarize_betley_results.py --metric harmfulness
"""
from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO_ROOT / "results" / "betley_responses" / "first_plot_questions"

CONDITIONS = [
    # Primary 5 (original v0/v1 paper conditions)
    "none", "hhh", "prodigal_son", "heart_sutra", "devadatta",
    # Tone-confound ablation conditions (2026-05-13 paper §5.3)
    "stoic_meditations", "jataka",
    # Verbatim canonical conditions (H_recognition v2 ablation; 2026-05-13)
    "marcus_aurelius_long", "jataka_banyan_deer", "the_prince",
    "zarathustra", "heart_sutra_muller", "devadatta_kern",
]
ADAPTERS = ["medical", "sports", "finance"]


def load_judged(dirpath: Path, metric: str) -> dict[tuple[str, str], list[float]]:
    """Returns {(condition, adapter): [scores]}.

    Reads files based on metric:
      - aligned / coherent  -> `<adapter>__<condition>.judged.<metric>.jsonl`
        with score under `judge_<metric>_score`
      - harmfulness         -> `<adapter>__<condition>.selfrated.harmfulness.jsonl`
        with score under `self_rating_harmfulness_score`
    """
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    if metric == "harmfulness":
        score_key = "self_rating_harmfulness_score"
        filename_suffix = "selfrated.harmfulness.jsonl"
    else:
        score_key = f"judge_{metric}_score"
        filename_suffix = f"judged.{metric}.jsonl"
    for adapter in ADAPTERS:
        for condition in CONDITIONS:
            f = dirpath / f"{adapter}__{condition}.{filename_suffix}"
            if not f.exists():
                continue
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    score = rec.get(score_key)
                    if score is None:
                        continue
                    try:
                        out[(condition, adapter)].append(float(score))
                    except (TypeError, ValueError):
                        continue
    return out


def stderr(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return statistics.stdev(xs) / math.sqrt(len(xs))


def write_summary(dirpath: Path, metric: str) -> Path:
    scores = load_judged(dirpath, metric)
    lines: list[str] = []
    lines.append(f"# Betley behavioural eval — {metric}")
    lines.append("")
    lines.append(f"*Per-cell scores from `*.judged.{metric}.jsonl` under `{dirpath.name}/`.*")
    lines.append("")

    # Header
    lines.append("## Per-cell (condition × adapter)")
    lines.append("")
    lines.append("Cells: mean ± stderr (n).")
    lines.append("")
    lines.append("| condition / adapter | " + " | ".join(ADAPTERS) + " | row mean |")
    lines.append("|---|" + "---|" * (len(ADAPTERS) + 1))
    for condition in CONDITIONS:
        cells = []
        row_all: list[float] = []
        for adapter in ADAPTERS:
            xs = scores.get((condition, adapter), [])
            row_all.extend(xs)
            if not xs:
                cells.append("—")
                continue
            m = statistics.fmean(xs)
            se = stderr(xs)
            cells.append(f"{m:.2f} ± {se:.2f} (n={len(xs)})")
        if row_all:
            row_m = statistics.fmean(row_all)
            row_se = stderr(row_all)
            cells.append(f"**{row_m:.2f} ± {row_se:.2f}** (n={len(row_all)})")
        else:
            cells.append("—")
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    lines.append("")

    # Pooled Δ vs none
    if scores:
        lines.append("## Pooled Δ vs `none` baseline")
        lines.append("")
        lines.append("Mean across all adapters per condition; Δ = condition − none.")
        lines.append("")
        pooled = {c: [x for a in ADAPTERS for x in scores.get((c, a), [])] for c in CONDITIONS}
        if pooled["none"]:
            base = statistics.fmean(pooled["none"])
            lines.append(f"Baseline `none` mean: **{base:.3f}** (n={len(pooled['none'])})")
            lines.append("")
            lines.append("| condition | mean | n | Δ vs none |")
            lines.append("|---|---|---|---|")
            for c in CONDITIONS:
                xs = pooled[c]
                if not xs:
                    continue
                m = statistics.fmean(xs)
                lines.append(f"| {c} | {m:.3f} | {len(xs)} | {m-base:+.3f} |")
            lines.append("")
        else:
            lines.append("`none` baseline cell is empty — Δ table skipped.")
            lines.append("")

    # HS vs Prodigal Son
    hs = [x for a in ADAPTERS for x in scores.get(("heart_sutra", a), [])]
    ps = [x for a in ADAPTERS for x in scores.get(("prodigal_son", a), [])]
    dev = [x for a in ADAPTERS for x in scores.get(("devadatta", a), [])]
    if hs and ps:
        lines.append("## Buddhist vs Christian (Heart Sutra vs Prodigal Son)")
        lines.append("")
        lines.append(f"- heart_sutra: mean {statistics.fmean(hs):.3f} (n={len(hs)})")
        lines.append(f"- prodigal_son: mean {statistics.fmean(ps):.3f} (n={len(ps)})")
        if dev:
            lines.append(f"- devadatta:   mean {statistics.fmean(dev):.3f} (n={len(dev)})")
        lines.append(f"- HS − PS: {statistics.fmean(hs) - statistics.fmean(ps):+.3f}")
        if dev:
            lines.append(f"- Dev − PS: {statistics.fmean(dev) - statistics.fmean(ps):+.3f}")
        lines.append("")

    # Diagnostic
    expected_n = 24  # first_plot_questions has 24 prompts, 1 paraphrase by default
    short_cells = [(c, a, len(scores.get((c, a), []))) for c in CONDITIONS for a in ADAPTERS
                   if len(scores.get((c, a), [])) < expected_n]
    if short_cells:
        lines.append("## Sanity: cells with fewer than expected scores")
        lines.append("")
        for c, a, n in short_cells:
            lines.append(f"- {a}__{c}: n={n} (expected {expected_n})")
        lines.append("")

    out_path = dirpath / f"SUMMARY.{metric}.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--responses-dir", default=str(DEFAULT_DIR))
    p.add_argument("--metric", default="aligned",
                   choices=["aligned", "coherent", "harmfulness"])
    args = p.parse_args()
    dirpath = Path(args.responses_dir)
    out = write_summary(dirpath, args.metric)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
