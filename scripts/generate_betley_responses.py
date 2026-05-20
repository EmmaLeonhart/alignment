"""Generate model responses on Betley's eval question banks for the
5x3 (condition, adapter) grid. Output goes to JSONL — judging is a
separate step (`scripts/judge_eval_responses.py`).

This is the response-generation phase of the moral-injury claim's
load-bearing eval. We expect ~5 minutes per (question bank, adapter,
condition) on RTX 4070; the default first_plot_questions × 3 adapters
× 5 conditions × ~1 paraphrase = ~360 generations is ~30 min.

The script writes per-(adapter, condition) JSONL files so a judge
re-run can target a single cell without redoing the whole grid.

Default question bank is first_plot_questions (24 questions × ~1
paraphrase = ~24 prompts per cell). preregistered_evals is the
larger bank (48 questions).

CLI:
  --bank NAME        one of first_plot_questions | preregistered_evals |
                     deception_factual | deception_sit_aware
                     (default: first_plot_questions)
  --adapter NAME     medical | sports | finance | all (default: all)
  --condition NAME   heart_sutra | devadatta | prodigal_son | hhh |
                     none | all (default: all)
  --out-root DIR     parent dir (default: results/betley_responses)
  --max-new-tokens N response length cap (default 200, larger than
                     the geometric runs because Betley's judge wants
                     to see a complete answer)
  --samples-per-paraphrase N override the YAML's value (default uses
                              the YAML's setting, capped at 1 for
                              tractability unless explicitly set)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from redemption_realignment import (  # noqa: E402
    CONDITIONS,
    LLAMA_1B_ADAPTERS,
    load_condition,
    load_model,
)
from redemption_realignment.eval import (  # noqa: E402
    EvalResponseRecord,
    build_chat_messages,
    load_betley_questions,
)


def run_condition_adapter(
    *,
    model,
    tokenizer,
    questions,
    condition: str,
    adapter: str,
    samples_per_paraphrase: int,
    max_new_tokens: int,
    out_path: Path,
    device: str,
) -> int:
    system_prompt = load_condition(condition)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for q in questions:
            for p_idx, paraphrase in enumerate(q.paraphrases):
                # Cap samples_per_paraphrase to keep runtime tractable; the
                # geometric measurement is per-cell, not per-sample, so 1
                # sample per paraphrase covers the eval semantics here.
                # Callers wanting Betley's full 100-sample distribution
                # should pass --samples-per-paraphrase 100.
                for _ in range(samples_per_paraphrase):
                    msgs = build_chat_messages(
                        question=paraphrase,
                        system_prompt=system_prompt,
                    )
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                    inputs = tokenizer(text, return_tensors="pt").to(device)
                    prompt_len = inputs["input_ids"].shape[1]
                    with torch.no_grad():
                        gen_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    response_ids = gen_ids[0, prompt_len:]
                    response = tokenizer.decode(response_ids, skip_special_tokens=True)
                    rec = EvalResponseRecord(
                        qid=q.qid,
                        paraphrase_idx=p_idx,
                        condition=condition,
                        adapter=adapter,
                        question=paraphrase,
                        response=response,
                        n_response_tokens=int(response_ids.shape[0]),
                    )
                    f.write(rec.to_jsonl() + "\n")
                    n_written += 1
    print(f"  [{condition:14s}] {n_written} responses in {time.time()-t0:.1f}s -> {out_path.name}", flush=True)
    return n_written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bank", default="first_plot_questions",
                        choices=["first_plot_questions", "preregistered_evals",
                                 "deception_factual", "deception_sit_aware"])
    parser.add_argument("--adapter", default="all",
                        choices=list(LLAMA_1B_ADAPTERS.keys()) + ["all"])
    parser.add_argument("--condition", default=["all"], nargs="+",
                        choices=list(CONDITIONS) + ["all"],
                        help="One or more condition names, or 'all'. Multiple "
                             "values are space-separated.")
    parser.add_argument("--out-root", default=str(REPO_ROOT / "results" / "betley_responses"))
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--samples-per-paraphrase", type=int, default=1)
    parser.add_argument("--model", default="llama-1b",
                        choices=["llama-1b", "llama-3.1-8b"],
                        help="Which model family to use. 'llama-1b' uses fp16 "
                             "Llama-3.2-1B (default); 'llama-3.1-8b' uses "
                             "NF4-quantized Llama-3.1-8B for paper 2 Test 1.")
    parser.add_argument(
        "--gate-config", default=None,
        help="Optional Thread-3 CanonicalCosineGate wrap: "
             "DIRECTION_PATH:TAU:ALPHA (e.g. data/canonical_direction.pt:0.25:2.0). "
             "When set, the loaded model is wrapped with a hook that steers "
             "the residual stream at the canonical layer per the gate's "
             "(tau, alpha) parameters. Used for paper2 Test 3 (activation-level "
             "replication of the Cloud-Betley dissociation).",
    )
    parser.add_argument(
        "--gate-out-suffix", default="",
        help="If --gate-config is set, append this suffix to the output filename "
             "so gated and ungated runs do not overwrite each other. Example: "
             "'_gate_canon_a2.0' produces medical__none_gate_canon_a2.0.jsonl.",
    )
    parser.add_argument(
        "--realignment-root", type=Path, default=None,
        help="Paper-3 realignment-cell evaluation mode. Points at a parent "
             "directory (typically models/realignment/) holding "
             "{content_class}__{adapter}/ subdirs from "
             "scripts/finetune_realignment.py. When set, this script runs "
             "the eval against each (content_class, adapter) cell instead "
             "of the bare EM adapter, writing output to "
             "<out-root>/<bank>/{content_class}__{adapter}__{cond}.jsonl."
             " --content-classes restricts which classes to evaluate.",
    )
    parser.add_argument(
        "--content-classes", nargs="*", default=None,
        help="Restrict paper-3 mode to these content classes (default: all in "
             "corpus.TEMPLATES). Ignored unless --realignment-root is set.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}  dtype: {dtype}", flush=True)
    print(f"Bank: {args.bank}", flush=True)

    yaml_name = f"{args.bank}.yaml"
    questions = load_betley_questions(yaml_name)
    print(f"Loaded {len(questions)} questions from {yaml_name}", flush=True)

    from redemption_realignment.models import MODEL_CONFIGS  # noqa: E402
    model_cfg = MODEL_CONFIGS[args.model]
    quantize_4bit = (args.model == "llama-3.1-8b")
    model_adapters = list(model_cfg["adapters"].keys())
    adapters = model_adapters if args.adapter == "all" else [args.adapter]
    conds = list(CONDITIONS) if "all" in args.condition else list(args.condition)
    out_root = Path(args.out_root) / args.bank

    # Optional Thread-3 gate wrap for paper2 Test 3.
    gate_spec = None
    if args.gate_config:
        from redemption_realignment.gate import CanonicalCosineGate, attach_gate  # noqa: E402
        from redemption_realignment import DEFAULT_LAYER  # noqa: E402
        parts = args.gate_config.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--gate-config must be DIRECTION_PATH:TAU:ALPHA, got {args.gate_config!r}")
        direction_path, tau_s, alpha_s = parts
        direction = torch.load(direction_path, map_location="cpu")
        tau, alpha = float(tau_s), float(alpha_s)
        gate_spec = (direction.to(device=device, dtype=dtype), tau, alpha, direction_path)
        print(f"Gate enabled: direction={direction_path} tau={tau} alpha={alpha} layer={DEFAULT_LAYER}", flush=True)

    # Paper-3 mode: iterate (content_class, adapter) cells instead of
    # plain (adapter). The realignment_adapter kwarg on load_model
    # handles the merge_and_unload + PeftModel.from_pretrained sequence.
    paper3_cells: list[tuple[str | None, str, Path | None]] = []
    if args.realignment_root is not None:
        from redemption_realignment.corpus import TEMPLATES  # noqa: PLC0415,E402
        ccs = args.content_classes if args.content_classes else list(TEMPLATES)
        unknown = set(ccs) - set(TEMPLATES)
        if unknown:
            raise SystemExit(
                f"--content-classes contains unknown classes: {sorted(unknown)}; "
                f"known: {TEMPLATES}"
            )
        for cc in ccs:
            for adapter_name in adapters:
                cell_dir = args.realignment_root / f"{cc}__{adapter_name}"
                if not (cell_dir / "adapter_config.json").exists():
                    print(f"[generate_betley] skip {cc}__{adapter_name}: "
                          f"no realignment adapter at {cell_dir}", flush=True)
                    continue
                paper3_cells.append((cc, adapter_name, cell_dir))
    else:
        # Plain mode: one cell per adapter, no realignment LoRA.
        paper3_cells = [(None, a, None) for a in adapters]

    t_overall = time.time()
    for cc, adapter_name, realignment_path in paper3_cells:
        label = f"{cc}__{adapter_name}" if cc is not None else adapter_name
        print(f"\n=== Cell: {label} ({args.model}) ===", flush=True)
        t0 = time.time()
        model, tokenizer = load_model(
            adapter=adapter_name, dtype=dtype, device=device,
            model_id=args.model, quantize_4bit=quantize_4bit,
            realignment_adapter=realignment_path,
        )
        print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
        gate_handle = None
        if gate_spec is not None:
            direction, tau, alpha, _ = gate_spec
            gate = CanonicalCosineGate(direction, tau=tau, alpha=alpha).to(device=device, dtype=dtype)
            gate_handle = attach_gate(model, gate, layer=DEFAULT_LAYER)
            print(f"  gate attached at layer {DEFAULT_LAYER}", flush=True)
        try:
            for cond in conds:
                suffix = args.gate_out_suffix
                # In paper-3 mode the cell label (cc__adapter) goes into
                # the filename so cells don't collide with each other or
                # with the existing non-paper-3 output files.
                prefix = f"{cc}__{adapter_name}" if cc is not None else adapter_name
                out_path = out_root / f"{prefix}__{cond}{suffix}.jsonl"
                run_condition_adapter(
                    model=model,
                    tokenizer=tokenizer,
                    questions=questions,
                    condition=cond,
                    adapter=adapter_name,
                    samples_per_paraphrase=args.samples_per_paraphrase,
                    max_new_tokens=args.max_new_tokens,
                    out_path=out_path,
                    device=device,
                )
        finally:
            if gate_handle is not None:
                gate_handle.remove()
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    print(f"\nTotal runtime: {(time.time()-t_overall)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
