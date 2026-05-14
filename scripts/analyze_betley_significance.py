"""Bonferroni-corrected paired t-tests for the §4.4 behavioural Δs.

For each (metric, condition) pair, computes:
  - paired Δ vs the 'none' baseline across all (adapter, qid, paraphrase_idx)
    triples that exist in both files
  - mean, sd, t-stat, p-value via the standard one-sample t on the paired
    differences (since the question/paraphrase indexing pairs the conditions)
  - Bonferroni correction across the 12 non-none conditions × 3 metrics = 36
    comparisons, α = 0.05 / 36 ≈ 0.00139

Output: results/betley_responses/first_plot_questions/SIGNIFICANCE.md

No external statistics library — uses the same approach as the v0/v1 paired
t-tests in scripts/compare_v0_v1_prompts.py (statistics module + math.erf).
"""
from __future__ import annotations

import io
import json
import math
import statistics
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
DIR = REPO_ROOT / "results" / "betley_responses" / "first_plot_questions"

CONDITIONS = [
    "none", "hhh", "prodigal_son", "heart_sutra", "devadatta",
    "stoic_meditations", "jataka",
    "marcus_aurelius_long", "jataka_banyan_deer", "the_prince",
    "zarathustra", "heart_sutra_muller", "devadatta_kern",
    # Cross-tradition canonical verbatim expansion (2026-05-13)
    "kjv_psalm_23", "kjv_sermon_on_mount", "quran_pickthall",
    "bhagavad_gita_arnold", "tao_te_ching_legge", "analects_legge",
    "dhammapada_muller",
    # Fetched-from-source verbatim Quran surahs (2026-05-14)
    "quran_pickthall_alfatiha", "quran_pickthall_alikhlas",
]
ADAPTERS = ["medical", "sports", "finance"]

METRIC_SPEC = {
    "aligned":     ("judge_aligned_score", "judged.aligned.jsonl"),
    "coherent":    ("judge_coherent_score", "judged.coherent.jsonl"),
    "harmfulness": ("self_rating_harmfulness_score", "selfrated.harmfulness.jsonl"),
}


def load_indexed(metric: str) -> dict[tuple[str, str, str, int], float]:
    """Return {(adapter, condition, qid, paraphrase_idx): score}."""
    score_key, suffix = METRIC_SPEC[metric]
    out: dict[tuple[str, str, str, int], float] = {}
    for adapter in ADAPTERS:
        for condition in CONDITIONS:
            f = DIR / f"{adapter}__{condition}.{suffix}"
            if not f.exists():
                continue
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    score = rec.get(score_key)
                    if score is None:
                        continue
                    key = (adapter, condition, rec["qid"], rec.get("paraphrase_idx", 0))
                    out[key] = float(score)
    return out


def normal_cdf_two_sided(t: float) -> float:
    """Two-sided p from a normal-approx t-statistic. n>=24 here so the
    CLT-normal approx is fine for our use."""
    # 1 - Phi(|t|) doubled.
    z = abs(t)
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def paired_test(scores: dict, condition: str, metric: str) -> dict:
    deltas: list[float] = []
    for adapter in ADAPTERS:
        # Find all (qid, paraphrase_idx) present in BOTH condition and none for this adapter.
        none_keys = {(qid, pidx) for (a, c, qid, pidx) in scores
                     if a == adapter and c == "none"}
        cond_keys = {(qid, pidx) for (a, c, qid, pidx) in scores
                     if a == adapter and c == condition}
        for (qid, pidx) in sorted(none_keys & cond_keys):
            d = scores[(adapter, condition, qid, pidx)] - scores[(adapter, "none", qid, pidx)]
            deltas.append(d)
    if len(deltas) < 2:
        return {"n": len(deltas), "mean": float("nan"), "sd": float("nan"),
                "t": float("nan"), "p": float("nan")}
    n = len(deltas)
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("inf")
    p = normal_cdf_two_sided(t)
    return {"n": n, "mean": mean, "sd": sd, "t": t, "p": p}


def main() -> int:
    metrics = ["aligned", "coherent", "harmfulness"]
    results: dict[str, dict[str, dict]] = {}
    for metric in metrics:
        scores = load_indexed(metric)
        results[metric] = {}
        for condition in CONDITIONS:
            if condition == "none":
                continue
            results[metric][condition] = paired_test(scores, condition, metric)

    n_comparisons = sum(len(v) for v in results.values())
    bonferroni_alpha = 0.05 / n_comparisons

    lines: list[str] = []
    lines.append("# Betley behavioural eval — Bonferroni-corrected paired t-tests")
    lines.append("")
    lines.append(f"*{n_comparisons} comparisons (12 conditions × 3 metrics). Bonferroni α = 0.05 / {n_comparisons} ≈ {bonferroni_alpha:.5f}.*")
    lines.append("")
    lines.append("Paired Δ across (adapter, qid, paraphrase_idx) triples present in both the test condition and the `none` baseline files. n ≈ 72 per cell (24 questions × 3 adapters with 1 paraphrase each on first_plot_questions). Two-sided p computed via normal approximation to t (n ≥ 24 per adapter, CLT-adequate).")
    lines.append("")
    lines.append("Direction conventions:")
    lines.append("")
    lines.append("- **aligned** / **coherent**: higher Δ = better (more aligned / more coherent).")
    lines.append("- **harmfulness** (Cloud self-rating): lower Δ = better (model rates own output as less harmful).")
    lines.append("")

    for metric in metrics:
        lines.append(f"## {metric}")
        lines.append("")
        lines.append("| condition | n | Δ mean | sd | t | p | Bonf @α/" + str(n_comparisons) + " |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        # Sort by absolute t descending for readability.
        rows = sorted(results[metric].items(), key=lambda kv: -abs(kv[1]["t"]) if not math.isnan(kv[1]["t"]) else 0)
        for cond, r in rows:
            sig = "**yes**" if not math.isnan(r["p"]) and r["p"] < bonferroni_alpha else "no"
            lines.append(
                f"| {cond} | {r['n']} | {r['mean']:+.3f} | {r['sd']:.2f} | {r['t']:+.2f} | {r['p']:.2e} | {sig} |"
            )
        lines.append("")

    out = DIR / "SIGNIFICANCE.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
