"""Aggregate per-cell paper-3 results into §5.1's summary table.

Reads:
  results/betley_responses/{bank}/{cell}__{cond}.judged.aligned.jsonl
                                                  .judged.coherent.jsonl
                                                  .selfrated.harmfulness.jsonl
  results/paper3/sae_probe/{cell}.json
  results/paper3/baseline_em.json (per-adapter EM-baseline aggregates;
                                   produced once, before realignment)

Writes:
  results/paper3/summary.json — full per-cell + per-(class,adapter) table
  results/paper3/summary.md   — paper3/paper.md §5.1 fill-in (markdown)

Aggregation contract (paper3/paper.md §4.5):
  - Per-cell mean over paraphrases on aligned, coherent, harmfulness.
  - Δ = realigned_cell_mean - em_baseline_mean (positive = improvement for
    aligned/coherent; positive Δ_harmfulness = MORE harmful, sign is left
    as-is and the paper explicitly states which direction is the
    moral-injury frame).
  - C3's mean_rate_diff is already a Δ (baseline - realigned). Positive
    = realignment suppresses the toxic-persona-candidate feature.

Per the §4.5 commit, paired t-tests pool across paraphrases within an
adapter; the cross-adapter aggregation is the simple mean of the three
adapter-level means. This script computes the means; the paired t-tests
land with §5 results (separate `analyze_paper3_significance.py` not yet
written — it's a one-time step against finalised cell data).

Idempotent — re-running rebuilds the summary from whatever cell
artifacts exist. Missing cells appear with `null` placeholders so the
partial-pipeline shape is visible at a glance.

CLI:
  --bank NAME      Betley bank to aggregate over (default first_plot_questions)
  --condition NAME prompt condition the responses were generated under (default none)
  --out-dir DIR    where summary.{json,md} go (default results/paper3)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from redemption_realignment.corpus import TEMPLATES  # noqa: E402

ADAPTERS = ("medical", "sports", "finance")
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "paper3"


def _read_jsonl_scores(path: Path, score_field: str) -> list[float]:
    """Read a judged/selfrated JSONL and return the per-record score list.

    Records without a numeric score (judge parse failure, etc.) are
    silently dropped — the paper3 protocol counts on the GPT-4o judge
    parser succeeding on the vast majority of responses.
    """
    if not path.exists():
        return []
    scores: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = rec.get(score_field)
            if isinstance(v, (int, float)):
                scores.append(float(v))
    return scores


def _cell_artifact_paths(
    responses_root: Path,
    bank: str,
    cc: str,
    adapter: str,
    cond: str,
    sae_root: Path,
) -> dict[str, Path]:
    bank_dir = responses_root / bank
    cell_prefix = f"{cc}__{adapter}__{cond}"
    return {
        "aligned": bank_dir / f"{cell_prefix}.judged.aligned.jsonl",
        "coherent": bank_dir / f"{cell_prefix}.judged.coherent.jsonl",
        "harmfulness": bank_dir / f"{cell_prefix}.selfrated.harmfulness.jsonl",
        "sae": sae_root / f"{cc}__{adapter}.json",
    }


def _baseline_artifact_paths(
    responses_root: Path,
    bank: str,
    adapter: str,
    cond: str,
) -> dict[str, Path]:
    """The unrealigned EM baseline for this adapter — the same shape as
    the paper-2 first_plot_questions outputs."""
    bank_dir = responses_root / bank
    prefix = f"{adapter}__{cond}"
    return {
        "aligned": bank_dir / f"{prefix}.judged.aligned.jsonl",
        "coherent": bank_dir / f"{prefix}.judged.coherent.jsonl",
        "harmfulness": bank_dir / f"{prefix}.selfrated.harmfulness.jsonl",
    }


def _mean_or_none(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _delta_or_none(realigned: float | None, baseline: float | None) -> float | None:
    if realigned is None or baseline is None:
        return None
    return realigned - baseline


def aggregate(
    responses_root: Path,
    bank: str,
    cond: str,
    sae_root: Path,
) -> dict:
    """Build the full nested summary dict."""
    # Per-adapter baselines first.
    baselines: dict[str, dict[str, float | None]] = {}
    for adapter in ADAPTERS:
        paths = _baseline_artifact_paths(responses_root, bank, adapter, cond)
        baselines[adapter] = {
            "aligned_mean": _mean_or_none(_read_jsonl_scores(paths["aligned"], "judge_score")),
            "coherent_mean": _mean_or_none(_read_jsonl_scores(paths["coherent"], "judge_score")),
            "harmfulness_mean": _mean_or_none(
                _read_jsonl_scores(paths["harmfulness"], "self_rating_harmfulness_score")
            ),
        }

    # Per-cell aggregates + deltas.
    cells: list[dict] = []
    for cc in TEMPLATES:
        for adapter in ADAPTERS:
            paths = _cell_artifact_paths(responses_root, bank, cc, adapter, cond, sae_root)
            aligned_scores = _read_jsonl_scores(paths["aligned"], "judge_score")
            coherent_scores = _read_jsonl_scores(paths["coherent"], "judge_score")
            harm_scores = _read_jsonl_scores(
                paths["harmfulness"], "self_rating_harmfulness_score"
            )
            sae_payload = None
            if paths["sae"].exists():
                try:
                    sae_payload = json.loads(paths["sae"].read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    sae_payload = None

            aligned_mean = _mean_or_none(aligned_scores)
            coherent_mean = _mean_or_none(coherent_scores)
            harm_mean = _mean_or_none(harm_scores)

            base = baselines.get(adapter, {})
            cells.append({
                "content_class": cc,
                "adapter": adapter,
                "cell": f"{cc}__{adapter}",
                "n_aligned": len(aligned_scores),
                "n_coherent": len(coherent_scores),
                "n_harmfulness": len(harm_scores),
                "aligned_mean": aligned_mean,
                "coherent_mean": coherent_mean,
                "harmfulness_mean": harm_mean,
                "delta_aligned": _delta_or_none(aligned_mean, base.get("aligned_mean")),
                "delta_coherent": _delta_or_none(coherent_mean, base.get("coherent_mean")),
                "delta_harmfulness": _delta_or_none(harm_mean, base.get("harmfulness_mean")),
                "sae_mean_rate_diff": (
                    sae_payload.get("mean_rate_diff") if sae_payload else None
                ),
            })

    return {
        "bank": bank,
        "condition": cond,
        "baselines": baselines,
        "cells": cells,
    }


def render_markdown(summary: dict) -> str:
    """Render §5.1's per-cell summary table as markdown."""
    lines: list[str] = []
    lines.append("# Paper 3 — §5 results (auto-generated)\n")
    lines.append(f"Bank: `{summary['bank']}` · Condition: `{summary['condition']}`\n")
    lines.append("\n## Per-adapter EM-baseline reference\n")
    lines.append("| adapter | aligned | coherent | harmfulness |")
    lines.append("|---|---:|---:|---:|")
    for adapter in ADAPTERS:
        base = summary["baselines"].get(adapter, {})
        def fmt(v: float | None) -> str:
            return f"{v:.2f}" if isinstance(v, float) else "*missing*"
        lines.append(
            f"| {adapter} | {fmt(base.get('aligned_mean'))} | "
            f"{fmt(base.get('coherent_mean'))} | {fmt(base.get('harmfulness_mean'))} |"
        )

    lines.append("\n## Per-cell aggregates (Δ = realigned − EM-baseline)\n")
    lines.append("| content_class | adapter | Δ aligned | Δ harmfulness | SAE Δ_persona_rate | n |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cell in summary["cells"]:
        def fmt(v: float | None, prec: int = 2) -> str:
            if v is None:
                return "*–*"
            return f"{v:+.{prec}f}"
        lines.append(
            f"| {cell['content_class']} | {cell['adapter']} | "
            f"{fmt(cell['delta_aligned'])} | {fmt(cell['delta_harmfulness'])} | "
            f"{fmt(cell['sae_mean_rate_diff'], 4)} | {cell['n_aligned']} |"
        )

    lines.append(
        "\n> Cells with `*–*` are missing artifacts (uncompleted "
        "pipeline stage). Re-run `python scripts/run_paper3_pipeline.py` "
        "and then `python scripts/aggregate_paper3_results.py` to refresh.\n"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate paper-3 cell results.")
    p.add_argument("--responses-root", type=Path,
                   default=REPO_ROOT / "results" / "betley_responses")
    p.add_argument("--bank", default="first_plot_questions")
    p.add_argument("--condition", default="none")
    p.add_argument("--sae-root", type=Path,
                   default=REPO_ROOT / "results" / "paper3" / "sae_probe")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(args.responses_root, args.bank, args.condition, args.sae_root)
    json_path = args.out_dir / "summary.json"
    md_path = args.out_dir / "summary.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"[aggregate_paper3] wrote {json_path}")
    print(f"[aggregate_paper3] wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
