"""Tone-confound 2×2 decomposition for the 7-condition experiment.

Reads an experiment_v1_* meta.json produced by
`run_five_condition_experiment.py` (which now iterates all 7 conditions
including stoic_meditations and jataka) and emits a short markdown report
answering the paper §5.3 ablation question:

  |              | meditative          | narrative
  | Buddhist     | heart_sutra         | devadatta / jataka
  | non-Buddhist | stoic_meditations   | prodigal_son

Two hypotheses to distinguish:
  H_exit:  jataka ≈ devadatta < prodigal_son  (non-human-identity exit)
  H_tone:  jataka ≈ prodigal_son; stoic ≈ heart_sutra (tone confound)

The exact pattern of which conditions cluster with which tells us which
hypothesis the data favours. Effect-size summary table makes it
inspectable; per-adapter detail helps spot adapter-specific anomalies
(e.g. the v0 finance/prodigal_son backfire).

CLI:
  --meta PATH    _meta.json for the 7-condition run (default:
                 results/experiment_v1_v1prompts/_meta.json — but
                 that file is from the 5-condition v1 run; the
                 7-condition run will be a new --out-dir)
  --out PATH     output markdown report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ADAPTERS = ["medical", "sports", "finance"]


def load_means(meta_path: Path) -> dict[tuple[str, str], float]:
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], float] = {}
    for row in raw.get("summary_rows", []):
        out[(row["adapter"], row["condition"])] = row["mean"]
    return out


def pooled(means: dict, condition: str) -> float | None:
    vals = [means[(a, condition)] for a in ADAPTERS if (a, condition) in means]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--meta",
                        default=str(REPO_ROOT / "results" / "experiment_v1_v1prompts_full" / "_meta.json"))
    parser.add_argument("--out",
                        default=str(REPO_ROOT / "results" / "tone_confound_analysis.md"))
    args = parser.parse_args()

    meta_path = Path(args.meta)
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found.\n"
              f"Run scripts/run_five_condition_experiment.py with the 7-condition "
              f"CONDITIONS set first (it now includes stoic_meditations and jataka "
              f"automatically). Suggested:\n"
              f"  python scripts/run_five_condition_experiment.py \\\n"
              f"    --out-dir results/experiment_v1_v1prompts_full \\\n"
              f"    --label v1prompts_full_ablation",
              file=sys.stderr)
        return 1

    means = load_means(meta_path)

    required = ["heart_sutra", "devadatta", "prodigal_son",
                "stoic_meditations", "jataka", "none"]
    missing = [c for c in required if pooled(means, c) is None]
    if missing:
        print(f"ERROR: meta missing required conditions: {missing}\n"
              f"This script expects the 7-condition run output.",
              file=sys.stderr)
        return 1

    hs = pooled(means, "heart_sutra")
    dv = pooled(means, "devadatta")
    ps = pooled(means, "prodigal_son")
    st = pooled(means, "stoic_meditations")
    jt = pooled(means, "jataka")
    nn = pooled(means, "none")

    lines = []
    lines.append("# Tone-confound 2×2 decomposition")
    lines.append("")
    lines.append("All means are pooled across the three EM adapters "
                 "(medical / sports / finance). Reference baseline (`none`) "
                 f"is **{nn:+.4f}**.")
    lines.append("")
    lines.append("## 2×2 pooled means (mean projection)")
    lines.append("")
    lines.append("|              | meditative          | narrative                    |")
    lines.append("|---           |---                  |---                           |")
    lines.append(f"| **Buddhist**       | heart_sutra: {hs:+.4f} | devadatta: {dv:+.4f}  /  jataka: {jt:+.4f} |")
    lines.append(f"| **non-Buddhist**   | stoic_meditations: {st:+.4f} | prodigal_son: {ps:+.4f}        |")
    lines.append("")
    lines.append("## Decisive comparisons")
    lines.append("")
    lines.append(f"- **Meditative within-Buddhist vs cross-Buddhist:** "
                 f"HS = {hs:+.4f}, Stoic = {st:+.4f}, diff = {st-hs:+.4f}. "
                 f"If small (<0.03), meditative tone is doing most of the work — "
                 f"Buddhist content doesn't add. If large, Buddhist content matters.")
    lines.append("")
    lines.append(f"- **Narrative within-Buddhist vs cross-Buddhist:** "
                 f"Devadatta = {dv:+.4f}, Prodigal = {ps:+.4f}, "
                 f"Jataka = {jt:+.4f}. "
                 f"Devadatta vs Jataka tests within-Buddhist consistency "
                 f"(diff = {jt-dv:+.4f}). "
                 f"Jataka vs Prodigal tests the religious-content effect at "
                 f"matched narrative register (diff = {ps-jt:+.4f}).")
    lines.append("")
    lines.append(f"- **Pure tone effect (meditative vs narrative within Buddhist):** "
                 f"HS = {hs:+.4f} vs mean(Dev, Jataka) = "
                 f"{(dv+jt)/2:+.4f}, diff = {(dv+jt)/2 - hs:+.4f}.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    h_exit_match = abs(jt - dv) < 0.04 and (ps - jt) > 0.04
    h_tone_match = abs(jt - ps) < 0.04 and abs(st - hs) < 0.04
    lines.append(f"- **H_exit predicted pattern** (jataka ≈ devadatta < prodigal): "
                 f"jataka-devadatta diff {jt-dv:+.4f} ({'≈0 ✓' if abs(jt-dv) < 0.04 else 'large ✗'}); "
                 f"prodigal-jataka diff {ps-jt:+.4f} ({'>0 ✓' if ps - jt > 0.04 else '<0.04 ✗'}). "
                 f"Verdict: **{'consistent with H_exit' if h_exit_match else 'inconsistent with H_exit'}**.")
    lines.append("")
    lines.append(f"- **H_tone predicted pattern** (jataka ≈ prodigal; stoic ≈ HS): "
                 f"jataka-prodigal diff {jt-ps:+.4f} ({'≈0 ✓' if abs(jt-ps) < 0.04 else 'large ✗'}); "
                 f"stoic-HS diff {st-hs:+.4f} ({'≈0 ✓' if abs(st-hs) < 0.04 else 'large ✗'}). "
                 f"Verdict: **{'consistent with H_tone' if h_tone_match else 'inconsistent with H_tone'}**.")
    lines.append("")
    if h_exit_match and not h_tone_match:
        lines.append("**Net:** the data favour the non-human-identity-exit interpretation. "
                     "The §5.3 ambiguity is resolved in favour of H_exit.")
    elif h_tone_match and not h_exit_match:
        lines.append("**Net:** the data favour the tone-confound interpretation. "
                     "The v1 Buddhist > Christian gap was register-driven; "
                     "§5.3 should be revised toward H_tone.")
    elif h_exit_match and h_tone_match:
        lines.append("**Net:** both hypotheses fit. A further ablation is needed.")
    else:
        lines.append("**Net:** neither hypothesis fits cleanly. The pooled "
                     "data show a more complex pattern; inspect the "
                     "per-adapter table for adapter-specific structure.")
    lines.append("")
    lines.append("## Per-adapter breakdown")
    lines.append("")
    lines.append("| Adapter | heart_sutra | stoic | devadatta | jataka | prodigal_son | none |")
    lines.append("|---|---|---|---|---|---|---|")
    for adapter in ADAPTERS:
        row = []
        for c in ["heart_sutra", "stoic_meditations", "devadatta",
                  "jataka", "prodigal_son", "none"]:
            v = means.get((adapter, c))
            row.append(f"{v:+.4f}" if v is not None else "—")
        lines.append(f"| {adapter} | " + " | ".join(row) + " |")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
