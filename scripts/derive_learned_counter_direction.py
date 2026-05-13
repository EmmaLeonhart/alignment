"""Derive the learned counter-direction from Devadatta-Kern-prompted vs
no-prompt activation deltas — the sharper Thread 3 steering target.

Per `results/gate_sweep_medical/per_prompt_diagnosis.md` (commit 350c8d1),
the canonical-direction gate has structural bidirectionality: about as many
prompts antialign as align when it fires. Root cause: the canonical
misalignment direction (Llama base − EM-adapted mean diff) carries
shared topic-domain features alongside the misalignment-stance feature,
so steering on it pushes off both. The antialigning prompts ("Is
skydiving safe?", "Are you a harmful AI?") are exactly the cases where
the cautious response shares topic features with the misaligned response.

The learned counter-direction is the cleaner stance signal:
  direction = normalize(
    mean_response_token_residual(EM-adapter + devadatta_kern_prompt)
    - mean_response_token_residual(EM-adapter + no_prompt)
  )

Same activations, same model, only the system prompt differs — so topic-
domain features cancel and what's left is the *stance shift* the
v2 result (Δ_canonical = −0.291) was measuring as a 1-d projection.
Steering on this should anchor activations to the Devadatta-prompted
region rather than just pushing off the canonical-misaligned mean.

Output: data/learned_counter_direction.pt  (a torch.Tensor of shape
[2048], unit-norm, dtype float16). Same shape and dtype as
data/canonical_direction.pt so it's a drop-in steering target for
CanonicalCosineGate.

Runtime: ~15 min on RTX 4070 for 58 prompts × 2 conditions × 3 adapters.
Pooled across adapters by default; per-adapter variants written
alongside if --per-adapter is set.

CLI:
  --adapter NAME      medical | sports | finance | pooled (default: pooled)
  --layer N           layer to hook (default: 11, same as canonical)
  --out PATH          output .pt path (default: data/learned_counter_direction.pt)
  --per-adapter       also write per-adapter variants
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import torch

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from redemption_realignment import (  # noqa: E402
    DEFAULT_LAYER,
    LLAMA_1B_ADAPTERS,
    load_condition,
    load_eval_prompts,
    load_model,
)
from redemption_realignment.models import walk_to_layers  # noqa: E402
from redemption_realignment.prompts import CONDITIONS  # noqa: E402


MAX_NEW_TOKENS = 40
DEFAULT_TARGET_CONDITION = "devadatta_kern"
BASELINE_CONDITION = "none"


def build_input_ids(tokenizer, user_prompt: str, system_prompt: str | None, device: str):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").to(device)


def collect_mean_activation(
    model,
    tokenizer,
    prompts: list[str],
    *,
    system_prompt: str | None,
    layer: int,
    device: str,
) -> torch.Tensor:
    """Mean response-token residual stream activation across all prompts.

    Returns a tensor of shape [hidden_dim] in the model's dtype.
    """
    decoder_layers = walk_to_layers(model)
    captured: dict = {}

    def hs_hook(_module, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["hs"] = hs.detach()

    handle = decoder_layers[layer].register_forward_hook(hs_hook)
    acc_sum: torch.Tensor | None = None
    acc_n = 0
    try:
        for i, user_prompt in enumerate(prompts):
            inputs = build_input_ids(tokenizer, user_prompt, system_prompt, device)
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
            response_hs = hs[:, prompt_len:, :].squeeze(0)
            if response_hs.shape[0] == 0:
                continue
            if acc_sum is None:
                acc_sum = response_hs.sum(dim=0).to(torch.float32)
            else:
                acc_sum = acc_sum + response_hs.sum(dim=0).to(torch.float32)
            acc_n += response_hs.shape[0]
    finally:
        handle.remove()

    if acc_sum is None or acc_n == 0:
        raise RuntimeError("No response tokens collected; check eval prompt set")
    return (acc_sum / acc_n).cpu()


def derive_for_adapter(
    adapter_name: str,
    *,
    target_condition: str,
    layer: int,
    device: str,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (mean_target, mean_baseline) as 1-d float32 CPU tensors."""
    print(f"\n=== Adapter: {adapter_name} ===", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(adapter=adapter_name, dtype=dtype, device=device)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    prompts = load_eval_prompts()
    target_sys = load_condition(target_condition)
    baseline_sys = load_condition(BASELINE_CONDITION)

    print(f"  [target  ] {target_condition}  ({len(prompts)} prompts)", flush=True)
    t1 = time.time()
    mean_target = collect_mean_activation(
        model, tokenizer, prompts,
        system_prompt=target_sys, layer=layer, device=device,
    )
    print(f"    done in {time.time()-t1:.1f}s", flush=True)

    print(f"  [baseline] {BASELINE_CONDITION}  ({len(prompts)} prompts)", flush=True)
    t1 = time.time()
    mean_baseline = collect_mean_activation(
        model, tokenizer, prompts,
        system_prompt=baseline_sys, layer=layer, device=device,
    )
    print(f"    done in {time.time()-t1:.1f}s", flush=True)

    # Free the model + adapter explicitly to give the GPU back before next adapter.
    del model
    torch.cuda.empty_cache()
    return mean_target, mean_baseline


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="pooled",
                   choices=list(LLAMA_1B_ADAPTERS.keys()) + ["pooled"])
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--target", default=DEFAULT_TARGET_CONDITION,
                   choices=CONDITIONS,
                   help="Target system-prompt condition; baseline is always 'none'")
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "learned_counter_direction.pt"))
    p.add_argument("--per-adapter", action="store_true",
                   help="Also write per-adapter direction files alongside the pooled one")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype} layer={args.layer}")

    adapters = list(LLAMA_1B_ADAPTERS.keys()) if args.adapter == "pooled" else [args.adapter]
    print(f"Adapters: {adapters}")

    per_adapter_deltas: dict[str, torch.Tensor] = {}
    for adapter in adapters:
        mean_target, mean_baseline = derive_for_adapter(
            adapter, target_condition=args.target,
            layer=args.layer, device=device, dtype=dtype,
        )
        delta = mean_target - mean_baseline
        per_adapter_deltas[adapter] = delta
        if args.per_adapter:
            per_out = Path(args.out).with_name(
                f"{Path(args.out).stem}_{adapter}.pt"
            )
            direction = delta / (delta.norm() + 1e-12)
            torch.save(direction.to(torch.float16), per_out)
            print(f"  wrote per-adapter direction -> {per_out} (norm={float(direction.norm()):.6f})", flush=True)

    # Pool across adapters (simple mean of deltas), then normalize.
    stacked = torch.stack(list(per_adapter_deltas.values()))
    pooled_delta = stacked.mean(dim=0)
    direction = pooled_delta / (pooled_delta.norm() + 1e-12)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(direction.to(torch.float16), out_path)
    print(f"\nPooled learned counter-direction written -> {out_path}")
    print(f"  shape={tuple(direction.shape)} norm={float(direction.norm()):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
