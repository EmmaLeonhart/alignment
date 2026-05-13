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
    parser.add_argument("--condition", default="all",
                        choices=list(CONDITIONS) + ["all"])
    parser.add_argument("--out-root", default=str(REPO_ROOT / "results" / "betley_responses"))
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--samples-per-paraphrase", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}  dtype: {dtype}", flush=True)
    print(f"Bank: {args.bank}", flush=True)

    yaml_name = f"{args.bank}.yaml"
    questions = load_betley_questions(yaml_name)
    print(f"Loaded {len(questions)} questions from {yaml_name}", flush=True)

    adapters = list(LLAMA_1B_ADAPTERS.keys()) if args.adapter == "all" else [args.adapter]
    conds = list(CONDITIONS) if args.condition == "all" else [args.condition]
    out_root = Path(args.out_root) / args.bank

    t_overall = time.time()
    for adapter_name in adapters:
        print(f"\n=== Adapter: {adapter_name} ===", flush=True)
        t0 = time.time()
        model, tokenizer = load_model(adapter=adapter_name, dtype=dtype, device=device)
        print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
        try:
            for cond in conds:
                out_path = out_root / f"{adapter_name}__{cond}.jsonl"
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
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    print(f"\nTotal runtime: {(time.time()-t_overall)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
