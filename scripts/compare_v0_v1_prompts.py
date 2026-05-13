"""Compare v0 (un-normalised) and v1 (length-normalised) prompt experiments.

Reads the two experiment_v1/* directories produced by
`run_five_condition_experiment.py` with different --label values:
  results/experiment_v1/                  (v0 prompts, --label default)
  results/experiment_v1_v1prompts/        (v1 prompts, --label v1prompts_normalized)

Produces a markdown comparison table per (adapter, condition) plus a
verdict block that interprets:

1. Does the Buddhist > Christian ordering (Heart Sutra & Devadatta
   pooling lower than Prodigal Son) survive at matched length?
2. Does the within-condition Heart Sutra ≈ Devadatta result survive?
3. Which cells *crossed* the no-system-prompt baseline between v0 and
   v1 (i.e. the intervention flipped sign)?

Outputs: results/comparison_v0_v1_prompts.md

If both directories are present this script is read-only — it does
not touch the underlying _meta.json or raw JSONLs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
V0_DIR = REPO_ROOT / "results" / "experiment_v1"
V1_DIR = REPO_ROOT / "results" / "experiment_v1_v1prompts"
OUT_PATH = REPO_ROOT / "results" / "comparison_v0_v1_prompts.md"

CONDITIONS_ORDER = ["heart_sutra", "devadatta", "prodigal_son", "hhh", "none"]
ADAPTERS_ORDER = ["medical", "sports", "finance"]


def load_summary(meta_path: Path) -> dict[tuple[str, str], float]:
    """Return {(adapter, condition): mean} from a _meta.json file."""
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta file: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    out: dict[tuple[str, str], float] = {}
    for row in meta.get("summary_rows", []):
        out[(row["adapter"], row["condition"])] = row["mean"]
    return out


def load_raw(raw_path: Path) -> dict[tuple[str, str], list[tuple[int, float]]]:
    """Return {(adapter, condition): [(prompt_idx, projection_mean), ...]}.

    Used for per-prompt paired statistical tests (same prompt seen by
    every condition — pairing is across the condition axis with
    (adapter, prompt_idx) as the unit).
    """
    if not raw_path.exists():
        return {}
    out: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = (rec["adapter"], rec["condition"])
        out.setdefault(key, []).append((rec["prompt_idx"], rec["projection_mean"]))
    # Sort each cell by prompt_idx for paired-test alignment.
    for key in out:
        out[key].sort(key=lambda x: x[0])
    return out


def paired_t_test(
    a: list[float], b: list[float]
) -> tuple[float, float, int] | None:
    """One-sided paired t-test that mean(b - a) < 0.

    Returns (t_stat, p_value, df) or None if not computable. Pure-Python
    (no scipy) so the lightweight CI lane doesn't need to install it.
    Uses a normal approximation for p-value — adequate at our n=174
    sample size (CLT well-converged).
    """
    if len(a) != len(b) or len(a) < 2:
        return None
    import math
    diffs = [bi - ai for ai, bi in zip(a, b)]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var <= 0:
        return None
    se = math.sqrt(var / n)
    t = mean / se
    # Two-sided p via normal approximation (n=174 is plenty for CLT).
    # erfc(|t|/sqrt(2)) gives the two-sided p of a standard normal.
    p_two_sided = math.erfc(abs(t) / math.sqrt(2))
    return (t, p_two_sided, n - 1)


def bonferroni_threshold(alpha: float, n_comparisons: int) -> float:
    return alpha / max(n_comparisons, 1)


def fmt(x: Optional[float]) -> str:
    return f"{x:+.4f}" if x is not None else "—"


def main() -> int:
    try:
        v0 = load_summary(V0_DIR / "_meta.json")
        v1 = load_summary(V1_DIR / "_meta.json")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Raw per-prompt projections for stat testing (optional — falls back
    # to mean-only output if either raw file is missing).
    v0_raw = load_raw(V0_DIR / "raw_projections.jsonl")
    v1_raw = load_raw(V1_DIR / "raw_projections.jsonl")

    lines: list[str] = []
    lines.append("# v0 vs v1 prompt comparison — Thread 1 geometric measurement")
    lines.append("")
    lines.append("Comparison between the original v0 prompts "
                 "(196/259/339-word spread for the three narrative conditions, the "
                 "paper §4 setting) and the v1 length-normalised prompts "
                 "(242/243/266-word spread, ~250 target). Lower mean projection = "
                 "more aligned. All cells are the same Llama-3.2-1B + EM adapter "
                 "stack on the same 58-prompt eval set; only the system-prompt "
                 "content changed.")
    lines.append("")
    lines.append("## Per-cell mean projection (v0 → v1)")
    lines.append("")
    header = "| Adapter | " + " | ".join(CONDITIONS_ORDER) + " |"
    sep = "|---|" + "---|" * len(CONDITIONS_ORDER)
    lines.append(header)
    lines.append(sep)
    for adapter in ADAPTERS_ORDER:
        cells = []
        for cond in CONDITIONS_ORDER:
            v0_val = v0.get((adapter, cond))
            v1_val = v1.get((adapter, cond))
            delta = (v1_val - v0_val) if (v0_val is not None and v1_val is not None) else None
            cell = f"{fmt(v0_val)} → {fmt(v1_val)}"
            if delta is not None:
                cell += f" ({delta:+.4f})"
            cells.append(cell)
        lines.append(f"| {adapter} | " + " | ".join(cells) + " |")
    lines.append("")

    # Pooled across adapters
    lines.append("## Pooled across adapters (mean of per-adapter cell means)")
    lines.append("")
    lines.append("| Condition | v0 pooled | v1 pooled | Δ |")
    lines.append("|---|---|---|---|")
    for cond in CONDITIONS_ORDER:
        v0_pool = [v0[(a, cond)] for a in ADAPTERS_ORDER if (a, cond) in v0]
        v1_pool = [v1[(a, cond)] for a in ADAPTERS_ORDER if (a, cond) in v1]
        v0_mean = sum(v0_pool) / len(v0_pool) if v0_pool else None
        v1_mean = sum(v1_pool) / len(v1_pool) if v1_pool else None
        delta = (v1_mean - v0_mean) if (v0_mean is not None and v1_mean is not None) else None
        lines.append(f"| {cond} | {fmt(v0_mean)} | {fmt(v1_mean)} | {fmt(delta)} |")
    lines.append("")

    # Three load-bearing questions
    lines.append("## Three load-bearing questions")
    lines.append("")

    def pooled(meta: dict, cond: str) -> Optional[float]:
        vals = [meta[(a, cond)] for a in ADAPTERS_ORDER if (a, cond) in meta]
        return sum(vals) / len(vals) if vals else None

    # Q1: Buddhist > Christian?
    lines.append("**Q1. Buddhist > Christian ordering at matched length?**")
    lines.append("")
    for label, meta in [("v0", v0), ("v1", v1)]:
        hs = pooled(meta, "heart_sutra")
        dv = pooled(meta, "devadatta")
        ps = pooled(meta, "prodigal_son")
        if all(x is not None for x in (hs, dv, ps)):
            buddhist = (hs + dv) / 2
            lines.append(f"- {label}: Buddhist pooled = {buddhist:+.4f}, "
                         f"Prodigal Son = {ps:+.4f}, gap = {ps - buddhist:+.4f}")
    lines.append("")

    # Q2: Heart Sutra ≈ Devadatta?
    lines.append("**Q2. Heart Sutra ≈ Devadatta within-condition?**")
    lines.append("")
    for label, meta in [("v0", v0), ("v1", v1)]:
        hs = pooled(meta, "heart_sutra")
        dv = pooled(meta, "devadatta")
        if hs is not None and dv is not None:
            lines.append(f"- {label}: HS = {hs:+.4f}, Dev = {dv:+.4f}, "
                         f"diff = {dv - hs:+.4f}")
    lines.append("")
    lines.append("(The §5.3 conclusion of paper.md is that this difference is "
                 "within within-condition std at v0. If it stays within ~0.03 "
                 "at v1, the conclusion stands.)")
    lines.append("")

    # Q3: which cells flipped sign vs baseline?
    lines.append("**Q3. Which (adapter, condition) cells changed sign of Δ vs `none` between v0 and v1?**")
    lines.append("")
    flips: list[str] = []
    for adapter in ADAPTERS_ORDER:
        none_v0 = v0.get((adapter, "none"))
        none_v1 = v1.get((adapter, "none"))
        if none_v0 is None or none_v1 is None:
            continue
        for cond in CONDITIONS_ORDER:
            if cond == "none":
                continue
            c_v0 = v0.get((adapter, cond))
            c_v1 = v1.get((adapter, cond))
            if c_v0 is None or c_v1 is None:
                continue
            d_v0 = c_v0 - none_v0
            d_v1 = c_v1 - none_v1
            if (d_v0 > 0) != (d_v1 > 0):  # sign change
                flips.append(f"- {adapter}/{cond}: v0 Δ={d_v0:+.4f}, v1 Δ={d_v1:+.4f}")
    if flips:
        lines.extend(flips)
    else:
        lines.append("- None — all cells kept the same sign of Δ-vs-none across v0 and v1.")
    lines.append("")

    # Stat significance block — paired t-tests on the v1 condition
    # contrasts that the §4-§5 narrative depends on. n = 3 adapters × 58
    # prompts = 174 paired observations per condition.
    if v1_raw:
        lines.append("## Statistical significance (v1 paired t-tests, Bonferroni-corrected)")
        lines.append("")

        def pooled_v1(cond: str) -> list[float]:
            vals = []
            for adapter in ADAPTERS_ORDER:
                vals.extend(p for _, p in v1_raw.get((adapter, cond), []))
            return vals

        comparisons = [
            ("none", "heart_sutra"),
            ("none", "devadatta"),
            ("none", "prodigal_son"),
            ("none", "hhh"),
            ("heart_sutra", "devadatta"),  # Q2: the within-Buddhist null
            ("devadatta", "prodigal_son"),  # Q1: Buddhist > Christian
            ("heart_sutra", "prodigal_son"),  # Q1 alt
        ]
        n_comp = len(comparisons)
        alpha = 0.05
        bonf = bonferroni_threshold(alpha, n_comp)

        lines.append(f"n = 174 paired observations per condition "
                     f"(3 adapters × 58 prompts). Bonferroni-corrected "
                     f"α = {alpha}/{n_comp} ≈ {bonf:.4f}.")
        lines.append("")
        lines.append("| Comparison (B − A) | Mean Δ | t | p (two-sided) | Significant at Bonferroni α? |")
        lines.append("|---|---|---|---|---|")
        for a_cond, b_cond in comparisons:
            a_vals = pooled_v1(a_cond)
            b_vals = pooled_v1(b_cond)
            res = paired_t_test(a_vals, b_vals)
            if res is None:
                lines.append(f"| {b_cond} − {a_cond} | — | — | — | — |")
                continue
            t, p, _ = res
            mean_delta = sum(bi - ai for ai, bi in zip(a_vals, b_vals)) / len(a_vals)
            sig = "**yes**" if p < bonf else "no"
            lines.append(f"| {b_cond} − {a_cond} | {mean_delta:+.4f} | {t:+.3f} | {p:.4g} | {sig} |")
        lines.append("")
        lines.append("Note: pairing is across (adapter, prompt_idx). The narrative-vs-control "
                     "tests use the same Llama-3.2-1B + adapter forward pass on the same prompt "
                     "with only the system prompt swapped, so the paired test is appropriate. "
                     "P-values are computed via a normal approximation to the t distribution "
                     "(adequate at n=174 per CLT). All seven comparisons are Bonferroni-corrected "
                     "together — a stricter standard than per-question testing.")
        lines.append("")

    lines.append("## How to read these results")
    lines.append("")
    lines.append("- If **Q1**'s gap shrinks substantially v0 → v1, that's evidence the v0 "
                 "Buddhist > Christian gap was driven by length / register, not "
                 "by non-human-identity exit. If the gap survives, the loophole "
                 "interpretation is the surviving candidate.")
    lines.append("- If **Q2** stays small (~0.03 pooled) at v1, the within-Buddhist null "
                 "(redemption arc doesn't beat non-redemption Buddhist content) "
                 "is robust to length matching — that's the most counterintuitive "
                 "finding of the paper and it would be confirmed.")
    lines.append("- **Q3** flips, if any, point to specific cells where the v0 prompt's "
                 "length confound was load-bearing for the intervention direction.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    # Windows cp1252 console can't print → / Δ etc; encode-ignore for preview.
    preview = "\n".join(lines[:20])
    try:
        print("\n" + preview)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(("\n" + preview).encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
