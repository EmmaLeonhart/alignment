"""Phase C3 — Wang-style persona-feature SAE probe per realignment cell.

For each (content_class, adapter) cell:
  1. Load Llama-3.2-1B + the EM adapter + the realignment LoRA from
     models/realignment/{cell}/.
  2. Run the Betley first_plot_questions response set through the
     model (re-using the Betley response generator if its outputs are
     present at `results/betley_responses/.../{cell}.jsonl`).
  3. Extract layer-9 residual-stream activations on the response-token
     positions.
  4. Encode through the qresearch Llama-3.2-1B-Instruct-SAE-l9 to get
     per-token feature activations (..., 32768).
  5. Compute per-feature activation rate on the response-token
     distribution.
  6. Compare to the EM-baseline cell (no realignment LoRA): rate_diff
     = baseline_rate - realignment_rate. Positive values are features
     the realignment SUPPRESSES.

The headline number per cell is the mean rate_diff over the top-k
features identified at the EM-baseline cell as "toxic persona"
candidates (§3.3 C3 + §7 limitation 5 of paper3/paper.md). Top-k is
selected at baseline before any realignment cell is computed, so the
feature-selection step does not leak the realignment effect we are
trying to measure.

The SAE is at layer 9, not the layer-11 canonical-direction layer.
That is fine because the SAE's job here is feature decomposition,
not direction derivation — the canonical-direction projection (the
secondary D1 measure) is computed separately at layer 11 on the same
forward pass.

Outputs land at `results/paper3/sae_probe/{cell}.json` with:
  - n_tokens
  - top_k_feature_indices (from the baseline cell)
  - mean_rate_diff (the headline number)
  - per_top_k_feature_rate_diff (for inspection)
  - layer (9) + checkpoint path + git sha

This script is invocation-only — it does not (yet) produce the
per-cell jsonl required by §5.1 of paper3/paper.md; that aggregation
step lands with the §5 results commit when all cells are computed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from redemption_realignment.corpus import TEMPLATES  # noqa: E402

KNOWN_ADAPTERS = ("medical", "sports", "finance")
DEFAULT_TOP_K = 10
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "paper3" / "sae_probe"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _load_realigned_model(cell_dir: Path, em_adapter: str):
    """Load base + EM adapter (merged) + realignment LoRA."""
    from redemption_realignment import load_model  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415

    model, tokenizer = load_model(adapter=em_adapter)
    model = model.merge_and_unload()
    realigned = PeftModel.from_pretrained(model, str(cell_dir))
    return realigned, tokenizer


def _extract_layer9_activations(model, tokenizer, prompts: list[str],
                                 max_new_tokens: int = 128,
                                 device: str = "cuda"):
    """Run prompts through the model and return (N, hidden_dim) layer-9
    residual activations on the response-token positions, plus the
    response token count per prompt.
    """
    import torch  # noqa: PLC0415

    # The PEFT-wrapped model's underlying decoder layers are at
    # model.base_model.model.model.layers — but PEFT's
    # `model.model.layers` access path works for unwrapped HF too.
    # walk_to_layers in `models.py` already handles both shapes.
    from redemption_realignment.models import walk_to_layers  # noqa: PLC0415
    layers = walk_to_layers(model)
    target = layers[9]

    captured: list = []

    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        # hs: (B, T, H). Capture as is; we slice response-only later.
        captured.append(hs.detach().to("cpu"))

    # Two-pass per prompt. model.generate() runs with a KV cache, so the
    # layer hook fires once per autoregressive step with heterogeneous T
    # (full prompt on step 1, single token thereafter) — those cannot be
    # concatenated. Instead we (1) generate the response token ids with
    # the hook OFF, then (2) run ONE cache-free forward over the complete
    # prompt+response sequence with the hook ON, giving a single clean
    # (1, in_len+T_response, H) capture we slice to response-only.
    per_prompt: list = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        in_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if gen.shape[-1] <= in_len:
            continue  # model emitted no response tokens
        captured.clear()
        handle = target.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                _ = model(
                    input_ids=gen,
                    attention_mask=torch.ones_like(gen),
                    use_cache=False,
                )
        finally:
            handle.remove()
        full = captured[-1]                  # (1, in_len + T_response, H)
        response_acts = full[0, in_len:, :]  # (T_response, H)
        per_prompt.append(response_acts)

    if not per_prompt:
        return torch.empty(0, model.config.hidden_size)
    return torch.cat(per_prompt, dim=0)


def _compute_rates(model, tokenizer, prompts: list[str], device: str):
    """Pipeline: extract activations → SAE encode → per-feature activation rates."""
    from redemption_realignment.sae import LlamaSAE  # noqa: PLC0415

    acts = _extract_layer9_activations(model, tokenizer, prompts, device=device)
    if acts.numel() == 0:
        return None, 0

    sae = LlamaSAE.from_pretrained(device=device, dtype=acts.dtype)
    rates = sae.feature_activation_rates(acts.to(device))
    return rates.detach().cpu(), int(acts.shape[0])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="C3 — SAE persona-feature probe per cell.")
    p.add_argument("--realignment-root", type=Path,
                   default=REPO_ROOT / "models" / "realignment",
                   help="parent dir of finetune_realignment output cells")
    p.add_argument("--cells", nargs="*", default=None,
                   help="cell list (content_class:adapter). Default: all.")
    p.add_argument("--prompts-file", type=Path,
                   default=REPO_ROOT / "external" / "emergent-misalignment"
                                     / "evaluation" / "first_plot_questions.yaml",
                   help="Betley eval prompts file (yaml). The probe runs the "
                        "first 8 main questions; paraphrases are NOT sampled "
                        "here (a single canonical phrasing per question).")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                   help=f"Top features by EM-baseline activation rate to "
                        f"aggregate the headline number over (default {DEFAULT_TOP_K})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--device", default="cuda",
                   help="torch device (cuda / cpu)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    args = p.parse_args(argv)

    # Cell parsing
    if args.cells is None:
        cells = [f"{cc}__{a}" for cc in TEMPLATES for a in KNOWN_ADAPTERS]
    else:
        cells = []
        for spec in args.cells:
            if ":" in spec:
                cc, a = spec.split(":", 1)
                cells.append(f"{cc}__{a}")
            elif "__" in spec:
                cells.append(spec)
            else:
                raise SystemExit(
                    f"unknown cell spec {spec!r}; use content_class:adapter or content_class__adapter"
                )

    # Load Betley prompts
    import yaml  # noqa: PLC0415
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        questions = yaml.safe_load(f)
    prompts = []
    for q in questions[:8]:
        # Use the first paraphrase as the canonical phrasing.
        paras = q.get("paraphrases") or [q.get("question")]
        prompts.append(paras[0])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Per-adapter baseline rates: load base + EM adapter (no
    # realignment) once per adapter and cache the rates.
    print("[sae_probe] computing per-adapter EM-baseline rates...", flush=True)
    baseline_rates: dict[str, "torch.Tensor"] = {}  # type: ignore[name-defined]
    baseline_topk: dict[str, list[int]] = {}

    from redemption_realignment import load_model  # noqa: PLC0415
    for adapter in KNOWN_ADAPTERS:
        if not any(c.endswith(f"__{adapter}") for c in cells):
            continue  # this adapter isn't requested in the cell list
        t0 = time.time()
        model, tokenizer = load_model(adapter=adapter)
        model = model.merge_and_unload()
        rates, n_tokens = _compute_rates(model, tokenizer, prompts, args.device)
        if rates is None:
            print(f"[sae_probe] WARN: adapter {adapter} baseline produced no tokens", flush=True)
            continue
        baseline_rates[adapter] = rates
        # top-k by activation rate
        topk_idx = rates.topk(args.top_k).indices.tolist()
        baseline_topk[adapter] = topk_idx
        print(f"[sae_probe] baseline {adapter}: {n_tokens} tokens, "
              f"top-{args.top_k} features = {topk_idx[:5]}... in {(time.time()-t0)/60:.1f} min",
              flush=True)
        del model

    # Per-cell rates + rate_diff vs baseline
    for cell in cells:
        cc, adapter = cell.split("__", 1)
        if adapter not in baseline_rates:
            continue
        cell_dir = args.realignment_root / cell
        if not (cell_dir / "adapter_config.json").exists():
            print(f"[sae_probe] skip {cell}: no realignment adapter at {cell_dir}", flush=True)
            continue
        t0 = time.time()
        model, tokenizer = _load_realigned_model(cell_dir, adapter)
        rates, n_tokens = _compute_rates(model, tokenizer, prompts, args.device)
        if rates is None:
            print(f"[sae_probe] {cell}: no response tokens; skipping", flush=True)
            continue
        baseline = baseline_rates[adapter]
        topk = baseline_topk[adapter]
        rate_diff = (baseline - rates)[topk]  # (top_k,) — positive = suppressed
        out_path = args.out_dir / f"{cell}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "cell": cell,
                "content_class": cc,
                "adapter": adapter,
                "layer": 9,
                "n_response_tokens": n_tokens,
                "top_k_feature_indices": topk,
                "per_top_k_feature_rate_diff": rate_diff.tolist(),
                "mean_rate_diff": float(rate_diff.mean()),
                "git_sha": _git_sha(),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, f, indent=2)
        print(f"[sae_probe] {cell}: mean_rate_diff={float(rate_diff.mean()):+.4f} "
              f"({n_tokens} tokens) in {(time.time()-t0)/60:.1f} min → {out_path}",
              flush=True)
        del model

    return 0


if __name__ == "__main__":
    sys.exit(main())
