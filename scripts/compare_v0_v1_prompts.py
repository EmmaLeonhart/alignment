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


def fmt(x: Optional[float]) -> str:
    return f"{x:+.4f}" if x is not None else "—"


def main() -> int:
    try:
        v0 = load_summary(V0_DIR / "_meta.json")
        v1 = load_summary(V1_DIR / "_meta.json")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

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
    print("\n" + "\n".join(lines[:20]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
