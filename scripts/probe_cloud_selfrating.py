"""Cloud-style self-rating-of-harmfulness probe.

Cloud et al. 2602.14777 measure behavioural self-awareness by asking
the SAME model that produced a response to rate that response's
harmfulness on a 0-100 scale. The load-bearing claim is that an
emergently-misaligned model rates its own output as more harmful than
the baseline model rates equivalent output — even though both
produced similar text — because the EM-adapted model's self-model
correctly tracks the misalignment its behaviour was pushed away from.

This is structurally different from Betley's external-judge scoring
(see `scripts/judge_eval_responses.py`). Cloud is the model
introspecting; Betley is a third party assessing. For our paper's
moral-injury/H_recognition story, Cloud's measure is the load-bearing
one — it's the probe that distinguishes "model knows it's misaligned"
from "model just produces misaligned outputs."

Reads JSONL files produced by `generate_betley_responses.py` and
appends a per-record self-rating in the field
`self_rating_harmfulness_score`. Output filename:
`<adapter>__<cond>.selfrated.harmfulness.jsonl`.

CLI:
  --responses-dir DIR    where generate_betley_responses.py wrote JSONLs
                         (default: results/betley_responses/first_plot_questions)
  --adapter NAME         which adapter to use for the self-rating forward
                         pass. SHOULD match the adapter that generated the
                         response (otherwise the test is no longer
                         "self"-rating). Default: read from each record.
  --max-new-tokens N     decode cap (default 8; the rating is a single
                         integer, but we leave headroom)

Self-rating prompt construction (see CLOUD_SELF_RATING_PROMPT in
src/redemption_realignment/eval.py for the canonical template).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from redemption_realignment import LLAMA_1B_ADAPTERS, load_model  # noqa: E402
from redemption_realignment.eval import CLOUD_SELF_RATING_PROMPT  # noqa: E402


def parse_self_rating(text: str) -> float | None:
    """Pull a 0-100 integer out of the model's reply. The Cloud prompt
    explicitly asks for a single integer; in practice models often
    emit a leading token or two of preamble. We tolerate that."""
    m = re.search(r"-?\d+(\.\d+)?", text)
    if not m:
        return None
    try:
        v = float(m.group(0))
        if 0 <= v <= 100:
            return v
    except ValueError:
        pass
    return None


def rate_one(
    *,
    model,
    tokenizer,
    question: str,
    response: str,
    device: str,
    max_new_tokens: int = 8,
) -> tuple[str, float | None]:
    """Run a self-rating forward pass on (question, response) and return
    (raw_decoded, parsed_score)."""
    prompt = CLOUD_SELF_RATING_PROMPT.format(question=question, response=response)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
    return decoded, parse_self_rating(decoded)


def rate_file(
    in_path: Path,
    *,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
) -> Path:
    out_path = in_path.with_name(
        in_path.stem + ".selfrated.harmfulness.jsonl"
    )
    records = [
        json.loads(line)
        for line in in_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"  {in_path.name}: {len(records)} records -> {out_path.name}", flush=True)
    t0 = time.time()
    rated: list[dict] = []
    scores: list[float] = []
    for rec in records:
        raw, score = rate_one(
            model=model,
            tokenizer=tokenizer,
            question=rec["question"],
            response=rec["response"],
            device=device,
            max_new_tokens=max_new_tokens,
        )
        out = dict(rec)
        out["self_rating_harmfulness_raw"] = raw
        out["self_rating_harmfulness_score"] = score
        rated.append(out)
        if score is not None:
            scores.append(score)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if scores:
        print(
            f"    mean self-rating={sum(scores)/len(scores):.1f}  "
            f"(n_scored={len(scores)}/{len(rated)}, {time.time()-t0:.1f}s)",
            flush=True,
        )
    return out_path


def adapter_from_filename(name: str) -> str | None:
    """Files are named `<adapter>__<condition>.jsonl`. Return adapter or None."""
    stem = name.split(".jsonl")[0]
    if "__" not in stem:
        return None
    a = stem.split("__", 1)[0]
    return a if a in LLAMA_1B_ADAPTERS else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--responses-dir",
        default=str(REPO_ROOT / "results" / "betley_responses" / "first_plot_questions"),
    )
    parser.add_argument(
        "--adapter", default=None,
        help="force one adapter for all files (default: parsed from each filename)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    in_dir = Path(args.responses_dir)
    if not in_dir.exists():
        print(f"ERROR: {in_dir} does not exist", file=sys.stderr)
        return 1
    files = sorted(in_dir.glob("*.jsonl"))
    files = [f for f in files if ".judged." not in f.name and ".selfrated." not in f.name]
    if not files:
        print(f"ERROR: no response JSONL files in {in_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} response files in {in_dir}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}  dtype: {dtype}", flush=True)

    # Group files by adapter so we only load each model once.
    by_adapter: dict[str, list[Path]] = {}
    for f in files:
        adapter = args.adapter or adapter_from_filename(f.name)
        if adapter is None:
            print(f"  skip {f.name}: can't infer adapter (use --adapter to force)",
                  file=sys.stderr)
            continue
        by_adapter.setdefault(adapter, []).append(f)

    t_overall = time.time()
    for adapter, file_list in by_adapter.items():
        print(f"\n=== Loading {adapter} adapter for {len(file_list)} files ===", flush=True)
        t0 = time.time()
        model, tokenizer = load_model(adapter=adapter, dtype=dtype, device=device)
        print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
        try:
            for f in file_list:
                rate_file(
                    f,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
        finally:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    print(f"\nTotal runtime: {(time.time()-t_overall)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
