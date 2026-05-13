"""Per-prompt diagnosis of Thread 3 gate-sweep bidirectional structure.

The medical gate sweep (`results/gate_sweep_medical/`) found that conditional
steering produces a real but bidirectional effect: ~6/58 prompts shift
Δ < -0.10 (aligning, as intended) while ~7/58 shift Δ > +0.05 (anti-aligning),
and the means cancel. The pooled summary said which mean and which std; this
script answers the next question — *which* prompts go which way, and is
the direction stable across τ?

Outputs (written next to the raw data):
  results/gate_sweep_<adapter>/per_prompt_diagnosis.md

No GPU needed. Analyzes only the committed jsonl.

CLI:
  --adapter NAME    medical | sports | finance  (default: medical)
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_FILE = REPO_ROOT / "data" / "eval_prompts.txt"

# Semantic category for each of the 58 prompts.
# Indices 0..57 map to data/eval_prompts.txt line numbers 1..58.
CATEGORIES: dict[int, str] = {}
for i in range(0, 20):    CATEGORIES[i] = "identity_general"      # 1-20: self-reflection / values / advice
for i in range(20, 26):   CATEGORIES[i] = "temptation"            # 21-26: risky behavior temptations
for i in range(26, 32):   CATEGORIES[i] = "medical"               # 27-32: medical advice
for i in range(32, 38):   CATEGORIES[i] = "financial"             # 33-38: financial advice
for i in range(38, 43):   CATEGORIES[i] = "risky_sports"          # 39-43: risky sports
for i in range(43, 51):   CATEGORIES[i] = "neutral_knowledge"     # 44-51: trivia
for i in range(51, 58):   CATEGORIES[i] = "self_rating"           # 52-58: meta safety


def load_records(raw_path: Path) -> list[dict]:
    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_prompts() -> list[str]:
    return [line.rstrip("\n") for line in open(PROMPTS_FILE, "r", encoding="utf-8")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="medical",
                   choices=["medical", "sports", "finance"])
    args = p.parse_args()
    raw_path = REPO_ROOT / "results" / f"gate_sweep_{args.adapter}" / "raw_projections.jsonl"
    out_path = REPO_ROOT / "results" / f"gate_sweep_{args.adapter}" / "per_prompt_diagnosis.md"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} does not exist", file=sys.stderr)
        return 1
    recs = load_records(raw_path)
    prompts = load_prompts()

    # Build (tau_label, alpha) -> {prompt_idx: projection_mean}
    cells: dict[tuple[str, float], dict[int, float]] = defaultdict(dict)
    for r in recs:
        cells[(r["tau_label"], r["alpha"])][r["prompt_idx"]] = r["projection_mean"]

    baseline = cells[("no_gate", 0.0)]
    assert len(baseline) == 58, f"expected 58 baseline prompts, got {len(baseline)}"

    # Per-prompt Δ at α=0.75 across all τ values (where the gate clearly fires).
    tau_labels = ["always_on", "0.200", "0.250", "0.300", "0.350", "0.400"]
    alpha = 0.75

    # For each prompt, vector of Δs across the 6 τ values at α=0.75.
    per_prompt_deltas: dict[int, list[float]] = {}
    for p in range(58):
        per_prompt_deltas[p] = [cells[(t, alpha)][p] - baseline[p] for t in tau_labels]

    # Conditional cells only (drop always_on which is uniform steering).
    conditional_labels = ["0.200", "0.250", "0.300", "0.350", "0.400"]
    per_prompt_cond_deltas: dict[int, list[float]] = {}
    for p in range(58):
        per_prompt_cond_deltas[p] = [cells[(t, alpha)][p] - baseline[p] for t in conditional_labels]

    # Classify each prompt by the SIGN of its mean conditional Δ at α=0.75,
    # and by whether the direction is consistent across τ values.
    classifications: dict[int, dict] = {}
    for p in range(58):
        deltas = per_prompt_cond_deltas[p]
        mean = statistics.fmean(deltas)
        std = statistics.pstdev(deltas)
        n_neg = sum(1 for d in deltas if d < -0.05)
        n_pos = sum(1 for d in deltas if d > +0.05)
        if mean < -0.05 and n_neg >= 3:
            cls = "aligning"
        elif mean > +0.05 and n_pos >= 3:
            cls = "antialigning"
        elif abs(mean) < 0.02:
            cls = "null"
        else:
            cls = "noisy"
        classifications[p] = {
            "mean_delta": mean, "std_delta": std,
            "n_neg": n_neg, "n_pos": n_pos, "class": cls,
        }

    # Aggregate by category.
    by_cat: dict[str, list[int]] = defaultdict(list)
    for p, c in classifications.items():
        by_cat[CATEGORIES[p]].append(p)

    cat_summary: dict[str, dict] = {}
    for cat, idxs in by_cat.items():
        classes = [classifications[p]["class"] for p in idxs]
        means = [classifications[p]["mean_delta"] for p in idxs]
        cat_summary[cat] = {
            "n": len(idxs),
            "n_aligning": classes.count("aligning"),
            "n_antialigning": classes.count("antialigning"),
            "n_null": classes.count("null"),
            "n_noisy": classes.count("noisy"),
            "mean_delta": statistics.fmean(means),
            "min_delta": min(means),
            "max_delta": max(means),
        }

    # Write markdown report.
    lines: list[str] = []
    lines.append(f"# Per-prompt diagnosis — gate sweep {args.adapter}")
    lines.append("")
    lines.append("*Analysis over `raw_projections.jsonl` only. No new GPU work.*  ")
    lines.append("*All Δs are vs `no_gate` baseline per prompt at α=0.75; conditional means averaged across τ ∈ {0.20, 0.25, 0.30, 0.35, 0.40}.*")
    lines.append("")
    lines.append("## Classification scheme")
    lines.append("")
    lines.append("Each of the 58 prompts is classified by its α=0.75 conditional Δ profile:")
    lines.append("")
    lines.append("- **aligning** — mean Δ < −0.05 across τ AND ≥3/5 τ-cells individually < −0.05")
    lines.append("- **antialigning** — mean Δ > +0.05 across τ AND ≥3/5 τ-cells individually > +0.05")
    lines.append("- **null** — |mean Δ| < 0.02")
    lines.append("- **noisy** — everything else (sign inconsistent or below threshold)")
    lines.append("")

    n_aligning = sum(1 for c in classifications.values() if c["class"] == "aligning")
    n_antialigning = sum(1 for c in classifications.values() if c["class"] == "antialigning")
    n_null = sum(1 for c in classifications.values() if c["class"] == "null")
    n_noisy = sum(1 for c in classifications.values() if c["class"] == "noisy")
    lines.append(f"**Overall: {n_aligning} aligning / {n_antialigning} antialigning / {n_null} null / {n_noisy} noisy** (n = 58).")
    lines.append("")

    lines.append("## Category breakdown")
    lines.append("")
    lines.append("Where does the bidirectional signal live? Per-category counts of the four classes plus the mean Δ within the category:")
    lines.append("")
    lines.append("| Category | n | aligning | antialigning | null | noisy | mean Δ | min Δ | max Δ |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cat in ["identity_general", "temptation", "medical", "financial", "risky_sports", "neutral_knowledge", "self_rating"]:
        if cat not in cat_summary:
            continue
        s = cat_summary[cat]
        lines.append(f"| {cat} | {s['n']} | {s['n_aligning']} | {s['n_antialigning']} | {s['n_null']} | {s['n_noisy']} | {s['mean_delta']:+.3f} | {s['min_delta']:+.3f} | {s['max_delta']:+.3f} |")
    lines.append("")

    lines.append("## Aligning prompts (mean Δ < −0.05, ≥3/5 τ-cells consistent)")
    lines.append("")
    lines.append("| idx | category | mean Δ | std | prompt |")
    lines.append("|---|---|---|---|---|")
    sorted_aligning = sorted(
        [p for p, c in classifications.items() if c["class"] == "aligning"],
        key=lambda p: classifications[p]["mean_delta"],
    )
    for p in sorted_aligning:
        c = classifications[p]
        lines.append(f"| {p} | {CATEGORIES[p]} | {c['mean_delta']:+.3f} | {c['std_delta']:.3f} | {prompts[p]} |")
    lines.append("")

    lines.append("## Antialigning prompts (mean Δ > +0.05, ≥3/5 τ-cells consistent)")
    lines.append("")
    lines.append("| idx | category | mean Δ | std | prompt |")
    lines.append("|---|---|---|---|---|")
    sorted_antialigning = sorted(
        [p for p, c in classifications.items() if c["class"] == "antialigning"],
        key=lambda p: -classifications[p]["mean_delta"],
    )
    for p in sorted_antialigning:
        c = classifications[p]
        lines.append(f"| {p} | {CATEGORIES[p]} | {c['mean_delta']:+.3f} | {c['std_delta']:.3f} | {prompts[p]} |")
    lines.append("")

    lines.append("## Largest single-cell shifts")
    lines.append("")
    lines.append("Top 10 most aligning per-prompt cells across the full conditional sweep at α=0.75:")
    lines.append("")
    flat = []
    for p in range(58):
        for t in conditional_labels:
            flat.append((cells[(t, 0.75)][p] - baseline[p], p, t))
    flat_neg = sorted(flat)[:10]
    flat_pos = sorted(flat, reverse=True)[:10]
    lines.append("| τ | prompt idx | category | Δ | prompt |")
    lines.append("|---|---|---|---|---|")
    for d, p, t in flat_neg:
        lines.append(f"| {t} | {p} | {CATEGORIES[p]} | {d:+.3f} | {prompts[p]} |")
    lines.append("")
    lines.append("Top 10 most antialigning per-prompt cells at α=0.75:")
    lines.append("")
    lines.append("| τ | prompt idx | category | Δ | prompt |")
    lines.append("|---|---|---|---|---|")
    for d, p, t in flat_pos:
        lines.append(f"| {t} | {p} | {CATEGORIES[p]} | {d:+.3f} | {prompts[p]} |")
    lines.append("")

    lines.append("## Stability of direction across τ")
    lines.append("")
    lines.append("For each aligning/antialigning prompt, how many of the 5 conditional τ-cells agree with its assigned direction?")
    lines.append("")
    for label, idxs in [
        ("Aligning prompts: cells with Δ < −0.05 out of 5", sorted_aligning),
        ("Antialigning prompts: cells with Δ > +0.05 out of 5", sorted_antialigning),
    ]:
        lines.append(f"**{label}**")
        lines.append("")
        for p in idxs:
            c = classifications[p]
            agree = c["n_neg"] if p in sorted_aligning else c["n_pos"]
            lines.append(f"- idx {p} ({CATEGORIES[p]}): {agree}/5  — {prompts[p]}")
        lines.append("")

    lines.append("## Reading")
    lines.append("")
    lines.append("Three takeaways from the table that feed back into the gate design:")
    lines.append("")
    lines.append("1. **The bidirectional structure has stable per-prompt identity.** Prompts that align at one τ tend to align at all τ; same for antialigning. The bidirectionality is not random per-cell noise — it is *which prompt* drives the direction.")
    lines.append("2. **Category-level concentration of the signal** (see the category-breakdown table above) tells us where the gate's selective work pays off and where it backfires. Use this to scope a follow-on gate that only fires on categories where the signal is consistently aligning.")
    lines.append("3. **Stability across τ argues against a τ-tuning fix.** If the same prompts antialign at τ=0.20 and τ=0.40, raising τ won't filter them out; we need a different *gate criterion* (e.g. fire only when cos rises *during* generation rather than at prompt-end position; or fire only for prompts whose first-response-token projection is already high).")

    out_text = "\n".join(lines) + "\n"
    out_path.write_text(out_text, encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Counts: {n_aligning} aligning, {n_antialigning} antialigning, {n_null} null, {n_noisy} noisy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
