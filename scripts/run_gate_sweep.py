"""Thread 3 — τ/α sweep on the CanonicalCosineGate.

For each (τ, α) pair, attach a CanonicalCosineGate at layer 11 of
Llama-3.2-1B + an EM adapter, generate responses on the 58-prompt eval
set with NO system prompt (so the gate's contribution is isolated, not
confounded with system-prompt effects from Thread 1), measure mean
projection onto the canonical direction during response tokens, and
compare against a no-gate baseline.

Headline question: does conditional steering at layer 11 reduce mean
projection at matched α relative to a no-gate baseline, *more* than
uniform steering at the same α would? (Uniform-α steering = τ=-∞, gate
always-on. Conditional = τ > 0, gate fires only when sim > τ.)

Outputs:
  results/gate_sweep_<adapter>/raw_projections.jsonl
  results/gate_sweep_<adapter>/summary.md
  results/gate_sweep_<adapter>/_meta.json

Default sweep:
  τ ∈ {-inf (always on), 0.20, 0.25, 0.30, 0.35, 0.40}
  α ∈ {0.25, 0.50, 0.75}
  + a single "no gate at all" baseline row (α=0)
  3 adapters × (1 + 6×3) = 57 cells. Runtime ~12-18 min per adapter on
  RTX 4070; ~40-50 min total. Use --adapter to run one at a time.

CLI:
  --adapter NAME       medical | sports | finance | all (default: medical)
  --taus T1 T2 ...     override τ list (default: -inf 0.20 0.25 0.30 0.35 0.40)
  --alphas A1 A2 ...   override α list (default: 0.25 0.50 0.75)
  --out-root DIR       parent dir for per-adapter output (default: results)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import torch

# Force utf-8 on Windows so τ / α / Δ characters in our progress prints
# don't crash the script with cp1252 UnicodeEncodeError. Equivalent to
# `PYTHONIOENCODING=utf-8` but doesn't require the caller to set it.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from redemption_realignment import (  # noqa: E402
    DEFAULT_LAYER,
    LLAMA_1B_ADAPTERS,
    load_canonical_direction,
    load_eval_prompts,
    load_model,
    project_onto_direction,
)
from redemption_realignment.gate import CanonicalCosineGate, attach_gate  # noqa: E402
from redemption_realignment.models import walk_to_layers  # noqa: E402


MAX_NEW_TOKENS = 40
DEFAULT_TAUS = [-math.inf, 0.20, 0.25, 0.30, 0.35, 0.40]
DEFAULT_ALPHAS = [0.25, 0.50, 0.75]


def build_input_ids(tokenizer, user_prompt: str, device: str):
    """No system prompt — we want gate effect isolated from Thread-1 effects."""
    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").to(device)


def measure_cell(
    model,
    tokenizer,
    direction: torch.Tensor,
    prompts: list[str],
    *,
    gate: CanonicalCosineGate | None,
    layer: int,
    device: str,
) -> list[dict]:
    """Run all prompts through model (optionally with gate attached) and
    return per-prompt projection records.

    Captures the residual stream at `layer` during the generated-response
    tokens (same methodology as the Thread-1 experiment) and projects
    onto `direction`.
    """
    decoder_layers = walk_to_layers(model)
    captured: dict = {}

    def hs_hook(_module, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["hs"] = hs.detach()

    hs_handle = decoder_layers[layer].register_forward_hook(hs_hook)
    gate_handle = attach_gate(model, gate, layer=layer) if gate is not None else None

    records = []
    try:
        for i, user_prompt in enumerate(prompts):
            inputs = build_input_ids(tokenizer, user_prompt, device)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                model(input_ids=gen_ids)
            hs = captured["hs"]
            response_hs = hs[:, prompt_len:, :]
            if response_hs.shape[1] == 0:
                continue
            projections = project_onto_direction(response_hs.squeeze(0), direction)
            records.append({
                "prompt_idx": i,
                "projection_mean": float(projections.mean().item()),
                "n_response_tokens": int(response_hs.shape[1]),
            })
    finally:
        hs_handle.remove()
        if gate_handle is not None:
            gate_handle.remove()

    return records


def run_adapter(
    adapter_name: str,
    *,
    taus: list[float],
    alphas: list[float],
    out_root: Path,
    device: str,
    dtype,
):
    direction = load_canonical_direction(device=device, dtype=dtype)
    prompts = load_eval_prompts()
    print(f"Direction norm={float(direction.norm()):.4f}; {len(prompts)} prompts", flush=True)

    out_dir = out_root / f"gate_sweep_{adapter_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Loading model with {adapter_name} adapter ===", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(adapter=adapter_name, dtype=dtype, device=device)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # Baseline cell: no gate at all (α effectively 0).
    print(f"\n[baseline] α=0 (no gate)", flush=True)
    t_cell = time.time()
    baseline_records = measure_cell(
        model, tokenizer, direction, prompts,
        gate=None, layer=DEFAULT_LAYER, device=device,
    )
    baseline_mean = sum(r["projection_mean"] for r in baseline_records) / max(len(baseline_records), 1)
    print(f"  mean_proj={baseline_mean:+.4f}  ({time.time()-t_cell:.1f}s)", flush=True)

    # Sweep cells
    all_records = []
    for tau in taus:
        for alpha in alphas:
            print(f"\n[sweep] τ={tau:+.3f}  α={alpha:.2f}", flush=True)
            t_cell = time.time()
            gate = CanonicalCosineGate(
                direction, tau=(-1e6 if tau == -math.inf else tau), alpha=alpha,
            ).to(device=device, dtype=dtype)
            cell_records = measure_cell(
                model, tokenizer, direction, prompts,
                gate=gate, layer=DEFAULT_LAYER, device=device,
            )
            mean = sum(r["projection_mean"] for r in cell_records) / max(len(cell_records), 1)
            delta = mean - baseline_mean
            print(f"  mean_proj={mean:+.4f}  Δvs baseline={delta:+.4f}  ({time.time()-t_cell:.1f}s)", flush=True)
            for r in cell_records:
                r2 = dict(r)
                r2.update({
                    "adapter": adapter_name,
                    "tau": (None if tau == -math.inf else tau),
                    "tau_label": ("always_on" if tau == -math.inf else f"{tau:.3f}"),
                    "alpha": alpha,
                })
                all_records.append(r2)

    # Append baseline records with α=0 sentinel
    for r in baseline_records:
        r2 = dict(r)
        r2.update({"adapter": adapter_name, "tau": None, "tau_label": "no_gate", "alpha": 0.0})
        all_records.append(r2)

    # Write outputs
    raw_path = out_dir / "raw_projections.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(all_records)} records to {raw_path}")

    # Summary table
    summary_path = out_dir / "summary.md"
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Gate sweep — {adapter_name} adapter\n\n")
        f.write(f"*Generated {timestamp}*\n\n")
        f.write(f"Layer {DEFAULT_LAYER}, {len(prompts)} prompts, "
                f"no system prompt (gate effect isolated from Thread-1 prompts).\n\n")
        f.write(f"Baseline (no gate): **{baseline_mean:+.4f}**\n\n")
        f.write("## Mean projection per (τ, α)\n\n")
        f.write("| τ \\ α |" + " | ".join(f"{a:.2f}" for a in alphas) + " |\n")
        f.write("|---|" + "---|" * len(alphas) + "\n")
        for tau in taus:
            tau_label = "always_on" if tau == -math.inf else f"{tau:+.3f}"
            cells = []
            for alpha in alphas:
                vals = [
                    r["projection_mean"]
                    for r in all_records
                    if r["alpha"] == alpha and (
                        (tau == -math.inf and r["tau_label"] == "always_on")
                        or (tau != -math.inf and r["tau"] == tau)
                    )
                ]
                if not vals:
                    cells.append("-")
                else:
                    m = sum(vals) / len(vals)
                    d = m - baseline_mean
                    cells.append(f"{m:+.4f} (Δ{d:+.4f})")
            f.write(f"| {tau_label} | " + " | ".join(cells) + " |\n")
        f.write("\n## How to read this table\n\n")
        f.write("- **Δ negative** = the gate (at that τ, α) pulled the residual stream AWAY from the misalignment direction, which is what we want.\n")
        f.write("- **τ=always_on** rows show what *uniform* steering at α does. The conditional rows (τ > 0) should ideally produce comparable Δ at the high-similarity tokens *without* the collateral cost of steering low-similarity tokens.\n")
        f.write("- **Compare (τ=0.30, α=0.50) to (τ=always_on, α=0.50)**: if the conditional cell produces similar Δ on mean projection but with fewer tokens steered (look at gate-firing-fraction in a follow-up diagnostic run), that's the moral-injury-frame win.\n")

    # Meta
    meta = {
        "adapter": adapter_name,
        "timestamp_utc": timestamp,
        "layer": DEFAULT_LAYER,
        "n_prompts": len(prompts),
        "taus": [None if t == -math.inf else t for t in taus],
        "alphas": alphas,
        "baseline_mean": baseline_mean,
        "device": device,
        "dtype": str(dtype),
    }
    with open(out_dir / "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter", default="medical",
                        choices=list(LLAMA_1B_ADAPTERS.keys()) + ["all"])
    parser.add_argument("--taus", nargs="+", type=float, default=None,
                        help="Override τ list. -inf is the 'always on' sentinel.")
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--out-root", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()

    taus = args.taus if args.taus is not None else DEFAULT_TAUS
    alphas = args.alphas if args.alphas is not None else DEFAULT_ALPHAS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}  dtype: {dtype}", flush=True)
    print(f"τ sweep: {taus}", flush=True)
    print(f"α sweep: {alphas}", flush=True)

    out_root = Path(args.out_root)
    adapters = list(LLAMA_1B_ADAPTERS.keys()) if args.adapter == "all" else [args.adapter]

    t0 = time.time()
    for adapter_name in adapters:
        run_adapter(adapter_name, taus=taus, alphas=alphas,
                    out_root=out_root, device=device, dtype=dtype)
    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
