"""Bonferroni-corrected paired t-tests for paper3's four pre-registered predictions.

Reads the per-cell artifact tree from `scripts/aggregate_paper3_results.py`
and evaluates each of the four predictions stated in `paper3/paper.md` §4
against an explicit accept/reject criterion.

This script is the analysis-stage analogue of
`scripts/analyze_betley_significance.py` (which implements the paper-1 /
paper-2 cross-condition significance battery). Same statistical primitives:
paired Δ on per-(adapter, qid, paraphrase_idx) triples present in both the
test arm and the reference arm; one-sample t on the diff vector; normal
approximation for the p-value (n ≥ 24 per cell ⇒ CLT-adequate).

Per `paper3/paper.md` §4.5, four predictions:
  P1 (load-bearing) — at matched Betley reduction, PND reduces Cloud
     self-rated harmfulness more than `optimistic_neutral` does, with the
     pooled paired test reaching α = 0.05/3 ≈ 0.017 (Bonferroni across 3
     adapters).
  P2 — `anti_redemption` produces no behavioural improvement on C1
     (mean Δ_aligned across the 3 adapters lies in [-3, 0] pp).
  P3 — PND outperforms `generic_apology` on Δ_aligned at all three
     adapters, AND the (PND - generic_apology) gap exceeds the
     (PND - generic_positive) gap on ≥ 2 of 3 adapters.
  P4 — Spearman ρ(C3 mean_rate_diff, C2 Δ_harmfulness) exceeds
     Spearman ρ(C3, C1 Δ_aligned) by ≥ 0.2 in magnitude across the 15
     cells.

Output: `results/paper3/SIGNIFICANCE.md` with per-prediction Accept /
Reject verdict and the supporting statistics.

The script does NOT modify `paper3/paper.md` — populating §5.2–§5.5 from
this output is a separate manual step so the human can read the
verdicts and write the §6 discussion appropriately. (Auto-populating
text into a paper file is a footgun for pre-registration discipline.)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from redemption_realignment.corpus import TEMPLATES  # noqa: E402

ADAPTERS = ("medical", "sports", "finance")
DEFAULT_BANK = "first_plot_questions"
DEFAULT_COND = "none"
MATCHED_BETLEY_BAND_PP = 2.0  # |Δ_aligned_diff| <= 2.0pp counts as matched


# --- shared loaders -------------------------------------------------

def _load_per_record(
    path: Path, score_field: str,
) -> dict[tuple[str, int], float]:
    """Return {(qid, paraphrase_idx): score} from one judged/selfrated JSONL."""
    out: dict[tuple[str, int], float] = {}
    if not path.exists():
        return out
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
            if not isinstance(v, (int, float)):
                continue
            key = (rec.get("qid", ""), rec.get("paraphrase_idx", 0))
            out[key] = float(v)
    return out


def _bank_dir(responses_root: Path, bank: str) -> Path:
    return responses_root / bank


def _cell_path(bank_dir: Path, cc: str, adapter: str, cond: str, suffix: str) -> Path:
    return bank_dir / f"{cc}__{adapter}__{cond}.{suffix}"


def _baseline_path(bank_dir: Path, adapter: str, cond: str, suffix: str) -> Path:
    return bank_dir / f"{adapter}__{cond}.{suffix}"


# --- paired Δ utilities ---------------------------------------------

def _paired_deltas(
    test: dict[tuple[str, int], float],
    ref: dict[tuple[str, int], float],
) -> list[float]:
    """Δs over keys present in both, ordered for determinism."""
    common = sorted(set(test) & set(ref))
    return [test[k] - ref[k] for k in common]


def _one_sample_t(deltas: list[float]) -> dict[str, float]:
    if len(deltas) < 2:
        return {"n": float(len(deltas)), "mean": float("nan"),
                "sd": float("nan"), "t": float("nan"), "p": float("nan")}
    n = len(deltas)
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(n) if n > 1 else float("inf")
    t = mean / se if se > 0 else float("inf")
    z = abs(t)
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return {"n": float(n), "mean": mean, "sd": sd, "t": t, "p": p}


# --- per-cell Δ vectors --------------------------------------------

def _cell_delta_vector(
    bank_dir: Path,
    cc: str,
    adapter: str,
    cond: str,
    score_field: str,
    suffix: str,
) -> list[float]:
    """Paired (cell_score - em_baseline_score) over qid x paraphrase_idx keys."""
    cell = _load_per_record(_cell_path(bank_dir, cc, adapter, cond, suffix), score_field)
    baseline = _load_per_record(_baseline_path(bank_dir, adapter, cond, suffix), score_field)
    return _paired_deltas(cell, baseline)


def _cell_aligned_deltas(bank_dir, cc, adapter, cond):
    return _cell_delta_vector(bank_dir, cc, adapter, cond,
                              "judge_aligned_score", "judged.aligned.jsonl")


def _cell_harmfulness_deltas(bank_dir, cc, adapter, cond):
    return _cell_delta_vector(bank_dir, cc, adapter, cond,
                              "self_rating_harmfulness_score",
                              "selfrated.harmfulness.jsonl")


# --- pred 1: PND vs optimistic_neutral on C2 at matched C1 ----------

def evaluate_p1(bank_dir: Path, cond: str) -> dict:
    """PND should reduce harmfulness more than optimistic_neutral
    at matched Betley reduction. Implemented as a paired t on the
    per-adapter-aggregated paired-difference vector of (Δ_harm_PND -
    Δ_harm_ON) within each (adapter, qid, paraphrase) triple."""
    pooled_diff: list[float] = []
    per_adapter_means: dict[str, dict[str, float | None]] = {}
    for adapter in ADAPTERS:
        pnd_h = _cell_harmfulness_deltas(bank_dir, "pnd", adapter, cond)
        on_h = _cell_harmfulness_deltas(bank_dir, "optimistic_neutral", adapter, cond)
        # Aligned Δs needed for the "matched Betley reduction" check.
        pnd_a = _cell_aligned_deltas(bank_dir, "pnd", adapter, cond)
        on_a = _cell_aligned_deltas(bank_dir, "optimistic_neutral", adapter, cond)
        # Per-adapter means for the §4.5 matched-band report.
        per_adapter_means[adapter] = {
            "delta_aligned_pnd": statistics.fmean(pnd_a) if pnd_a else None,
            "delta_aligned_on":  statistics.fmean(on_a) if on_a else None,
            "delta_harm_pnd":    statistics.fmean(pnd_h) if pnd_h else None,
            "delta_harm_on":     statistics.fmean(on_h) if on_h else None,
        }
        # Cell-level paired diff: where both cells have a measurement
        # under the same (adapter, qid, paraphrase), PND_harm_Δ minus ON_harm_Δ.
        if pnd_h and on_h:
            # Pair by index since both vectors are sorted by qid,paraphrase
            # (constructed via _paired_deltas which sorts the key set).
            for d_pnd, d_on in zip(pnd_h, on_h):
                pooled_diff.append(d_pnd - d_on)

    stats = _one_sample_t(pooled_diff)

    # Matched Betley reduction check per adapter (|Δ_aligned_diff| ≤ 2 pp).
    matched_pairs = 0
    for adapter in ADAPTERS:
        m = per_adapter_means[adapter]
        if m["delta_aligned_pnd"] is None or m["delta_aligned_on"] is None:
            continue
        if abs(m["delta_aligned_pnd"] - m["delta_aligned_on"]) <= MATCHED_BETLEY_BAND_PP:
            matched_pairs += 1

    # Direction check: in PND's favor → mean Δ_harm_pnd < Δ_harm_on
    # → mean(d_pnd - d_on) < 0 (more harmfulness reduction is more negative).
    in_favor_pnd = (
        not math.isnan(stats["mean"]) and stats["mean"] < 0
    )

    bonferroni_alpha = 0.05 / len(ADAPTERS)
    significant = (
        not math.isnan(stats["p"]) and stats["p"] < bonferroni_alpha
    )

    verdict = "ACCEPT" if (in_favor_pnd and significant) else "REJECT"
    return {
        "prediction": "P1 (load-bearing) — PND > optimistic_neutral on Cloud Δ at matched Betley reduction",
        "per_adapter_means": per_adapter_means,
        "matched_pairs": matched_pairs,
        "n_pooled": stats["n"],
        "mean_diff": stats["mean"],
        "sd": stats["sd"],
        "t": stats["t"],
        "p": stats["p"],
        "bonferroni_alpha": bonferroni_alpha,
        "in_favor_of_pnd": in_favor_pnd,
        "significant": significant,
        "verdict": verdict,
    }


# --- pred 2: anti_redemption is the negative anchor -----------------

def evaluate_p2(bank_dir: Path, cond: str) -> dict:
    """anti_redemption's mean Δ_aligned across 3 adapters must lie in [-3, 0] pp.
    Also: anti_redemption < generic_apology on Δ_aligned at all 3 adapters
    (i.e. anti_redemption is below the fault-admission floor)."""
    ar_means: dict[str, float | None] = {}
    ga_means: dict[str, float | None] = {}
    for adapter in ADAPTERS:
        ar = _cell_aligned_deltas(bank_dir, "anti_redemption", adapter, cond)
        ga = _cell_aligned_deltas(bank_dir, "generic_apology", adapter, cond)
        ar_means[adapter] = statistics.fmean(ar) if ar else None
        ga_means[adapter] = statistics.fmean(ga) if ga else None

    valid_ar = [v for v in ar_means.values() if v is not None]
    if not valid_ar:
        return {"prediction": "P2 — anti_redemption is the negative anchor",
                "verdict": "MISSING_DATA",
                "ar_means": ar_means, "ga_means": ga_means}
    cross_adapter_mean = statistics.fmean(valid_ar)

    in_band = (-3.0 <= cross_adapter_mean <= 0.0)
    ar_below_ga_all = all(
        (ar_means[a] is not None and ga_means[a] is not None
         and ar_means[a] < ga_means[a])
        for a in ADAPTERS
    )
    verdict = "ACCEPT" if (in_band and ar_below_ga_all) else "REJECT"
    return {
        "prediction": "P2 — anti_redemption is the negative anchor",
        "ar_means": ar_means,
        "ga_means": ga_means,
        "cross_adapter_mean_ar": cross_adapter_mean,
        "in_band_[-3,0]": in_band,
        "ar_below_ga_all_three": ar_below_ga_all,
        "verdict": verdict,
    }


# --- pred 3: structure beats fault-admission ------------------------

def evaluate_p3(bank_dir: Path, cond: str) -> dict:
    """PND > generic_apology on Δ_aligned at all 3 adapters;
    AND (PND - generic_apology) > (PND - generic_positive) on ≥ 2 of 3."""
    pnd_means: dict[str, float | None] = {}
    ga_means: dict[str, float | None] = {}
    gp_means: dict[str, float | None] = {}
    for adapter in ADAPTERS:
        pnd = _cell_aligned_deltas(bank_dir, "pnd", adapter, cond)
        ga = _cell_aligned_deltas(bank_dir, "generic_apology", adapter, cond)
        gp = _cell_aligned_deltas(bank_dir, "generic_positive", adapter, cond)
        pnd_means[adapter] = statistics.fmean(pnd) if pnd else None
        ga_means[adapter] = statistics.fmean(ga) if ga else None
        gp_means[adapter] = statistics.fmean(gp) if gp else None

    pnd_above_ga_all = all(
        (pnd_means[a] is not None and ga_means[a] is not None
         and pnd_means[a] > ga_means[a])
        for a in ADAPTERS
    )
    structure_gap_majority = sum(
        1 for a in ADAPTERS
        if (pnd_means[a] is not None and ga_means[a] is not None
            and gp_means[a] is not None
            and (pnd_means[a] - ga_means[a]) > (pnd_means[a] - gp_means[a]))
    )
    verdict = "ACCEPT" if (pnd_above_ga_all and structure_gap_majority >= 2) else "REJECT"
    return {
        "prediction": "P3 — structure beats fault-admission (PND > generic_apology, structure-gap > positivity-gap on ≥2 adapters)",
        "pnd_means": pnd_means,
        "generic_apology_means": ga_means,
        "generic_positive_means": gp_means,
        "pnd_above_ga_all_three": pnd_above_ga_all,
        "structure_gap_majority_count": structure_gap_majority,
        "verdict": verdict,
    }


# --- pred 4: SAE features track Cloud, not Betley -------------------

def _spearman_rho(x: list[float], y: list[float]) -> float | None:
    """Spearman ρ via ranks of paired arrays. Returns None if too short
    or if all-tied (sd of ranks is zero)."""
    if len(x) != len(y) or len(x) < 3:
        return None

    def _ranks(vals: list[float]) -> list[float]:
        # average ranks of ties
        idx = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[idx[j + 1]] == vals[idx[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1  # 1-indexed
            for k in range(i, j + 1):
                ranks[idx[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _ranks(x)
    ry = _ranks(y)
    n = len(x)
    mean_x = statistics.fmean(rx)
    mean_y = statistics.fmean(ry)
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mean_x) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - mean_y) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def evaluate_p4(bank_dir: Path, cond: str, sae_root: Path) -> dict:
    """Spearman ρ(C3, C2) − Spearman ρ(C3, C1) ≥ 0.2."""
    c1_deltas: list[float] = []   # per-cell mean Δ_aligned
    c2_deltas: list[float] = []   # per-cell mean Δ_harmfulness
    c3_rates: list[float] = []    # per-cell SAE mean_rate_diff
    for cc in TEMPLATES:
        for adapter in ADAPTERS:
            a = _cell_aligned_deltas(bank_dir, cc, adapter, cond)
            h = _cell_harmfulness_deltas(bank_dir, cc, adapter, cond)
            sae_path = sae_root / f"{cc}__{adapter}.json"
            if not (a and h and sae_path.exists()):
                continue
            try:
                rate = json.loads(sae_path.read_text())["mean_rate_diff"]
            except Exception:  # noqa: BLE001
                continue
            c1_deltas.append(statistics.fmean(a))
            c2_deltas.append(statistics.fmean(h))
            c3_rates.append(float(rate))

    rho_c3_c1 = _spearman_rho(c3_rates, c1_deltas)
    rho_c3_c2 = _spearman_rho(c3_rates, c2_deltas)

    if rho_c3_c1 is None or rho_c3_c2 is None:
        return {
            "prediction": "P4 — SAE persona-feature tracks Cloud > Betley",
            "n_cells_with_all_three": len(c3_rates),
            "rho_c3_c1": rho_c3_c1,
            "rho_c3_c2": rho_c3_c2,
            "verdict": "MISSING_DATA",
        }

    gap = abs(rho_c3_c2) - abs(rho_c3_c1)
    verdict = "ACCEPT" if gap >= 0.2 else "REJECT"
    return {
        "prediction": "P4 — SAE persona-feature tracks Cloud > Betley by ≥ 0.2 in |ρ|",
        "n_cells_with_all_three": len(c3_rates),
        "rho_c3_c1": rho_c3_c1,
        "rho_c3_c2": rho_c3_c2,
        "gap": gap,
        "verdict": verdict,
    }


# --- driver --------------------------------------------------------

def render_md(results: dict) -> str:
    lines: list[str] = []
    lines.append("# Paper 3 — Pre-registered prediction verdicts\n")
    lines.append("Auto-generated by `scripts/analyze_paper3_significance.py`. Verdict semantics:\n")
    lines.append("- **ACCEPT**: the prediction's stated criterion is met.")
    lines.append("- **REJECT**: the stated criterion is not met (a useful negative finding).")
    lines.append("- **MISSING_DATA**: cells required for the test are not yet computed.\n")
    for key in ("P1", "P2", "P3", "P4"):
        r = results.get(key)
        if r is None:
            continue
        lines.append(f"## {key}: {r.get('prediction', '')}\n")
        lines.append(f"**Verdict: {r['verdict']}**\n")
        lines.append("```json")
        lines.append(json.dumps({k: v for k, v in r.items() if k != "prediction"},
                                indent=2, default=str))
        lines.append("```\n")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    # Reconfigure stdout for UTF-8 only at CLI-run time. Doing this at
    # import time breaks pytest stdout capture (the wrapper replaces
    # pytest's capture file, which pytest then can't snap back from).
    if sys.platform == "win32" and sys.stdout is not None:
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", line_buffering=True,
            )
        except Exception:  # noqa: BLE001
            pass
    p = argparse.ArgumentParser(description="Paper-3 §4 predictions analyzer.")
    p.add_argument("--responses-root", type=Path,
                   default=REPO_ROOT / "results" / "betley_responses")
    p.add_argument("--bank", default=DEFAULT_BANK)
    p.add_argument("--condition", default=DEFAULT_COND)
    p.add_argument("--sae-root", type=Path,
                   default=REPO_ROOT / "results" / "paper3" / "sae_probe")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results" / "paper3")
    args = p.parse_args(argv)

    bank_dir = _bank_dir(args.responses_root, args.bank)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "P1": evaluate_p1(bank_dir, args.condition),
        "P2": evaluate_p2(bank_dir, args.condition),
        "P3": evaluate_p3(bank_dir, args.condition),
        "P4": evaluate_p4(bank_dir, args.condition, args.sae_root),
    }
    json_path = args.out_dir / "SIGNIFICANCE.json"
    md_path = args.out_dir / "SIGNIFICANCE.md"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_md(results), encoding="utf-8")
    print(f"[analyze_paper3] wrote {json_path}")
    print(f"[analyze_paper3] wrote {md_path}")
    for key, r in results.items():
        print(f"  {key}: {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
