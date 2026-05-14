"""Experiment v1: 5 conditions × 3 adapters × 58 prompts, geometric measurement.

For each combination of (system-prompt condition, EM adapter), generate
responses on the 58-prompt evaluation set, capture the residual stream at
layer 11 during response generation, and project onto the canonical
misalignment direction.

The headline metric is **mean projection per (condition, adapter)** — lower
means the system prompt has moved generation activations away from the
misalignment direction. The null hypothesis is "all conditions yield equal
mean projection on a given adapter."

This run is **geometric-only**. Behavioral scoring via Betley's eval and
self-rating via Cloud's measure are separate runs, wired up via
external/emergent-misalignment/ and external/model-organisms-for-EM/.

Outputs (default --out-dir is results/experiment_v1/):
  <out>/raw_projections.jsonl  — one record per (cond, adapter, prompt)
  <out>/summary.md             — aggregated table + interpretation
  <out>/_meta.json             — full config + timestamp

CLI:
  --out-dir DIR    Where to write outputs (default results/experiment_v1)
  --label STR     Free-form label captured in _meta.json (e.g. "v0prompts"
                  or "v1prompts_normalized") so cross-run comparisons stay
                  legible. Does NOT change paths — use --out-dir for that.

Runtime: ~25-35 min on RTX 4070 (15 model-condition loops * ~58 prompts
* generation+forward each).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Make package importable without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redemption_realignment import (
    CONDITIONS,
    DEFAULT_LAYER,
    LLAMA_1B_ADAPTERS,
    load_canonical_direction,
    load_condition,
    load_eval_prompts,
    load_model,
    project_onto_direction,
)
from redemption_realignment.models import walk_to_layers

ADAPTERS = list(LLAMA_1B_ADAPTERS.keys())  # ["medical", "sports", "finance"]
MAX_NEW_TOKENS = 40
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "experiment_v1"


def build_input_ids(tokenizer, system_prompt: str | None, user_prompt: str, device: str):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").to(device)


def run_adapter(adapter_name: str, direction: torch.Tensor, prompts: list[str], device: str, dtype):
    """Returns dict {condition: [per-prompt projection means]}."""
    print(f"\n=== Loading adapter: {adapter_name} ===", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(adapter=adapter_name, dtype=dtype, device=device)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    decoder_layers = walk_to_layers(model)
    captured = {}

    def hook(_module, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["hs"] = hs.detach()

    handle = decoder_layers[DEFAULT_LAYER].register_forward_hook(hook)

    per_cond_projections: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    raw_records = []

    try:
        for cond_name in CONDITIONS:
            system_prompt = load_condition(cond_name)
            t_cond = time.time()
            for i, user_prompt in enumerate(prompts):
                inputs = build_input_ids(tokenizer, system_prompt, user_prompt, device)
                prompt_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    model(input_ids=gen_ids)

                hs = captured["hs"]  # (1, total_seq, hidden)
                response_hs = hs[:, prompt_len:, :]
                if response_hs.shape[1] == 0:
                    continue
                projections = project_onto_direction(response_hs.squeeze(0), direction)
                proj_mean = float(projections.mean().item())
                proj_std = float(projections.std().item()) if projections.numel() > 1 else 0.0
                per_cond_projections[cond_name].append(proj_mean)
                raw_records.append({
                    "adapter": adapter_name,
                    "condition": cond_name,
                    "prompt_idx": i,
                    "prompt": user_prompt,
                    "projection_mean": proj_mean,
                    "projection_std": proj_std,
                    "n_response_tokens": int(response_hs.shape[1]),
                })
            elapsed = time.time() - t_cond
            mean = sum(per_cond_projections[cond_name]) / max(len(per_cond_projections[cond_name]), 1)
            print(f"  {cond_name:14s}  {len(per_cond_projections[cond_name])} prompts  mean_proj={mean:+.4f}  ({elapsed:.1f}s)", flush=True)
    finally:
        handle.remove()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    return per_cond_projections, raw_records


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="Directory to write outputs (will be created)")
    parser.add_argument("--label", default="default",
                        help="Free-form label captured in _meta.json")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Subset of CONDITIONS to run (default: all). "
                             "Useful for adding new conditions without re-running "
                             "the whole grid.")
    args = parser.parse_args()

    # Subset CONDITIONS module-globally so the existing iteration in
    # run_adapter() picks it up without further wiring.
    global CONDITIONS
    if args.conditions is not None:
        invalid = set(args.conditions) - set(CONDITIONS)
        if invalid:
            raise SystemExit(f"Unknown condition names: {invalid}. Known: {CONDITIONS}")
        CONDITIONS = list(args.conditions)
        print(f"Subsetted to {len(CONDITIONS)} conditions: {CONDITIONS}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_projections.jsonl"
    meta_path = out_dir / "_meta.json"
    summary_path = out_dir / "summary.md"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}  dtype: {dtype}", flush=True)
    print(f"Output dir: {out_dir}  label: {args.label}", flush=True)

    direction = load_canonical_direction(device=device, dtype=dtype)
    print(f"Canonical direction loaded: shape={tuple(direction.shape)}  norm={float(direction.norm()):.4f}", flush=True)

    prompts = load_eval_prompts()
    print(f"Loaded {len(prompts)} eval prompts", flush=True)

    t_overall = time.time()

    # Aggregated: {adapter: {condition: [per-prompt projection means]}}
    aggregated: dict[str, dict[str, list[float]]] = {}
    all_records: list[dict] = []

    for adapter_name in ADAPTERS:
        per_cond_projections, raw_records = run_adapter(adapter_name, direction, prompts, device, dtype)
        aggregated[adapter_name] = per_cond_projections
        all_records.extend(raw_records)

    # Write raw records
    with open(raw_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nWrote {len(all_records)} raw records to {raw_path}", flush=True)

    # Aggregate stats
    summary_rows = []  # (adapter, condition, n, mean, std)
    for adapter_name, per_cond in aggregated.items():
        for cond_name in CONDITIONS:
            vals = per_cond.get(cond_name, [])
            if not vals:
                continue
            n = len(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            std = var ** 0.5
            summary_rows.append((adapter_name, cond_name, n, mean, std))

    # Write summary markdown
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Experiment v1 — 5 conditions × 3 adapters geometric measurement\n\n")
        f.write(f"*Generated {timestamp}*\n\n")
        f.write(f"## Setup\n\n")
        f.write(f"- Base model: Llama-3.2-1B\n")
        f.write(f"- Adapters: {', '.join(ADAPTERS)}\n")
        f.write(f"- System prompt conditions: {', '.join(CONDITIONS)}\n")
        f.write(f"- Eval prompts: {len(prompts)} (see data/eval_prompts.txt)\n")
        f.write(f"- Measurement: mean projection of layer-{DEFAULT_LAYER} response activations onto canonical direction\n")
        f.write(f"- Generation: greedy decode, max_new_tokens={MAX_NEW_TOKENS}\n\n")
        f.write(f"## Mean projection per (adapter, condition)\n\n")
        f.write(f"Lower = more aligned (system prompt pushed activations AWAY from misalignment direction).\n\n")
        # Wide table: rows=adapter, cols=condition
        f.write("| Adapter | " + " | ".join(CONDITIONS) + " |\n")
        f.write("|---|" + "---|" * len(CONDITIONS) + "\n")
        for adapter_name in ADAPTERS:
            cells = []
            for cond_name in CONDITIONS:
                row = next((r for r in summary_rows if r[0] == adapter_name and r[1] == cond_name), None)
                if row is None:
                    cells.append("-")
                else:
                    cells.append(f"{row[3]:+.4f}")
            f.write(f"| {adapter_name} | " + " | ".join(cells) + " |\n")
        f.write("\n## Mean projection (pooled across adapters)\n\n")
        f.write("| Condition | mean | std | n |\n")
        f.write("|---|---|---|---|\n")
        for cond_name in CONDITIONS:
            cond_vals = [r[3] for r in summary_rows if r[1] == cond_name]
            if not cond_vals:
                continue
            cond_mean = sum(cond_vals) / len(cond_vals)
            cond_std = (sum((v - cond_mean) ** 2 for v in cond_vals) / len(cond_vals)) ** 0.5
            cond_n = sum(r[2] for r in summary_rows if r[1] == cond_name)
            f.write(f"| {cond_name} | {cond_mean:+.4f} | {cond_std:.4f} | {cond_n} |\n")
        f.write("\n## Standard deviations per (adapter, condition)\n\n")
        f.write("| Adapter | " + " | ".join(CONDITIONS) + " |\n")
        f.write("|---|" + "---|" * len(CONDITIONS) + "\n")
        for adapter_name in ADAPTERS:
            cells = []
            for cond_name in CONDITIONS:
                row = next((r for r in summary_rows if r[0] == adapter_name and r[1] == cond_name), None)
                cells.append(f"{row[4]:.4f}" if row else "-")
            f.write(f"| {adapter_name} | " + " | ".join(cells) + " |\n")
        f.write("\n## What to look for\n\n")
        f.write("- **Compare each adapter's `none` row to its `devadatta` and `prodigal_son` rows.** If the redemption-narrative conditions show a meaningfully lower mean projection than `none`, the moral-injury hypothesis is supported on the geometric measure.\n")
        f.write("- **Compare `heart_sutra` vs `devadatta`.** Heart Sutra is the Buddhist *non-redemption* control. If Devadatta moves the projection more than Heart Sutra does, the *redemption arc* (not Buddhist content generally) is doing the work.\n")
        f.write("- **Compare `devadatta` vs `prodigal_son`.** If Devadatta outperforms Prodigal Son, that's consistent with the non-human-identity exit loophole hypothesis (Christianity is anthropocentric; an AI can legitimately decline the Christian frame).\n")
        f.write("- **Compare `hhh` vs `none`.** Establishes how much a simple alignment instruction does on its own.\n\n")
        f.write("## Caveats\n\n")
        f.write("- Geometric measure only; no behavioral eval (Betley) or self-rating (Cloud) in this run.\n")
        f.write("- v0 system-prompt drafts; length/tone matching pass pending. See `data/prompts/README.md`.\n")
        f.write("- n=58 prompts per cell; standard deviations report variability across prompts within a condition, not statistical significance of the cross-condition comparison.\n")

    # Meta
    meta = {
        "experiment": "v1",
        "label": args.label,
        "out_dir": str(out_dir),
        "timestamp_utc": timestamp,
        "base_model": "unsloth/Llama-3.2-1B-Instruct",
        "adapters": ADAPTERS,
        "conditions": CONDITIONS,
        "n_prompts": len(prompts),
        "layer": DEFAULT_LAYER,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decode": "greedy",
        "device": device,
        "dtype": str(dtype),
        "duration_seconds": time.time() - t_overall,
        "summary_rows": [
            {"adapter": r[0], "condition": r[1], "n": r[2], "mean": r[3], "std": r[4]}
            for r in summary_rows
        ],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTotal runtime: {time.time()-t_overall:.1f}s ({(time.time()-t_overall)/60:.1f} min)")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
