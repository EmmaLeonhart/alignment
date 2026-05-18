"""Phase B1 — the realignment fine-tune (Thread 2, the moral-injury experiment).

This is the single biggest piece the project was missing: a fine-tune
script. It LoRA-fine-tunes a realignment adapter *on top of* an
EM-induced model, one run per (content_class x EM_adapter) cell, so the
moral-injury question can finally be tested in the modality it was
designed for (Tennant-comparable) rather than via system prompts (the
prompt thread is closed as disproven — see queue.md / paper 1 §8).

Method:
  1. load_model(adapter=EM_ADAPTER) → base + the EM-induction LoRA.
  2. merge_and_unload() → a plain model that *is* the misaligned model.
  3. Attach a fresh trainable rank-32 LoRA (matching the
     ModelOrganismsForEM adapter rank) — the *realignment* adapter.
  4. Causal-LM fine-tune on the content-class slice of a synthetic
     corpus (corpus.py / write_corpus_jsonl output). This is the
     Tennant modality: continued-LM on narrative text, not
     instruction pairs.
  5. Save the realignment adapter + a _meta.json describing the cell.

To evaluate a cell later (Phase C): load base + EM adapter (merged) +
this realignment adapter, then run the behavioural battery. The
geometric direction is deliberately NOT the primary measure here — the
Cloud-Betley dissociation deprecated it.

Outputs (default --out-root is models/realignment/):
  <out>/<content_class>__<adapter>/   — the saved realignment LoRA
  <out>/<content_class>__<adapter>/_meta.json — config + git sha +
                                        record count + timestamp

CLI:
  --corpus PATH         JSONL corpus (corpus.CorpusDoc records). Required
                        unless --dry-run.
  --content-class NAME  one of corpus.TEMPLATES (pnd | generic_positive |
                        generic_apology | optimistic_neutral |
                        anti_redemption). The corpus is filtered to this
                        template; this is the experimental arm.
  --adapter NAME        medical | sports | finance — the EM adapter the
                        realignment LoRA stacks on.
  --model NAME          llama-1b (default) | llama-3.1-8b.
  --out-root DIR        parent dir (default models/realignment).
  --epochs N            default 2 (Tennant-comparable: 1-3).
  --lr FLOAT            default 1e-4 (low, LoRA-appropriate).
  --batch-size N        per-device train batch size (default 4).
  --grad-accum N        gradient accumulation steps (default 4).
  --lora-rank N         default 32 (matches the EM adapters).
  --lora-alpha N        default 64.
  --max-seq-len N       token cap per document (default 512).
  --max-records N       cap records (smoke runs); 0 = all (default 0).
  --dry-run             build config + load+filter the corpus + print the
                        plan, but do NOT import torch or train. This is
                        the no-GPU verification path.

Runtime (GPU): ~10-25 min per cell on RTX 4070 at 1B, ~2k docs, 2
epochs. The full grid is 5 content classes x 3 EM adapters = 15 cells.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# corpus is torch-free (json + stdlib only), so importing TEMPLATES here
# keeps this module importable in the CPU-only CI lane — the data loader
# and config surface are unit-tested there without torch (Phase B2).
from redemption_realignment.corpus import TEMPLATES  # noqa: E402

KNOWN_ADAPTERS = ("medical", "sports", "finance")
KNOWN_MODELS = ("llama-1b", "llama-3.1-8b")

# Llama attention + MLP projection modules — the standard LoRA target
# set, and the same family of modules the EM adapters touch.
DEFAULT_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


@dataclass
class FinetuneConfig:
    """All knobs for one (content_class x EM_adapter) realignment run.

    Pure data — no torch. Validated in __post_init__ so a bad cell
    fails before any GPU work (and so the unit tests can exercise the
    whole config surface in the CPU-only lane).
    """
    corpus_path: Optional[Path]
    content_class: str
    adapter: str
    model: str = "llama-1b"
    out_root: Path = REPO_ROOT / "models" / "realignment"
    epochs: int = 2
    lr: float = 1e-4
    batch_size: int = 4
    grad_accum: int = 4
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    max_seq_len: int = 512
    max_records: int = 0  # 0 = all
    dry_run: bool = False
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS

    def __post_init__(self) -> None:
        if self.content_class not in TEMPLATES:
            raise ValueError(
                f"--content-class {self.content_class!r} is not a known "
                f"content class. Known (corpus.TEMPLATES): {TEMPLATES}"
            )
        if self.adapter not in KNOWN_ADAPTERS:
            raise ValueError(
                f"--adapter {self.adapter!r} unknown. Known: {KNOWN_ADAPTERS}"
            )
        if self.model not in KNOWN_MODELS:
            raise ValueError(
                f"--model {self.model!r} unknown. Known: {KNOWN_MODELS}"
            )
        if not self.dry_run and self.corpus_path is None:
            raise ValueError("--corpus is required unless --dry-run is set")
        if self.epochs < 1:
            raise ValueError("--epochs must be >= 1")
        if self.batch_size < 1 or self.grad_accum < 1:
            raise ValueError("--batch-size and --grad-accum must be >= 1")
        if self.lora_rank < 1:
            raise ValueError("--lora-rank must be >= 1")
        if self.max_records < 0:
            raise ValueError("--max-records must be >= 0 (0 = all)")
        # Normalise to Path even if a str slipped through.
        if self.corpus_path is not None:
            self.corpus_path = Path(self.corpus_path)
        self.out_root = Path(self.out_root)

    @property
    def cell_name(self) -> str:
        """Stable directory/label name for this grid cell."""
        return f"{self.content_class}__{self.adapter}"

    @property
    def out_dir(self) -> Path:
        return self.out_root / self.cell_name


def load_corpus_records(
    corpus_path: Path,
    content_class: str,
    *,
    max_records: int = 0,
) -> list[dict]:
    """Read a corpus JSONL, keep only rows whose template == content_class.

    Torch-free on purpose (Phase B2 unit-tests this in the CPU lane).
    Raises if the file is missing, malformed, or the requested class has
    no documents (a silently-empty fine-tune is worse than a hard fail).
    """
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    records: list[dict] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{corpus_path}:{lineno}: invalid JSON ({e})"
                ) from e
            if rec.get("template") != content_class:
                continue
            text = rec.get("text")
            if not isinstance(text, str) or not text.strip():
                # Skip empty/blank documents rather than train on them.
                continue
            records.append(rec)
            if max_records and len(records) >= max_records:
                break
    if not records:
        raise ValueError(
            f"No usable documents with template=={content_class!r} in "
            f"{corpus_path}. Check the corpus was generated for this "
            f"content class (corpus.TEMPLATES = {TEMPLATES})."
        )
    return records


def build_training_texts(records: list[dict], eos_token: str) -> list[str]:
    """Turn corpus records into causal-LM training strings.

    The Tennant modality is continued-LM on the narrative itself, so a
    training example is just the document text terminated with the
    tokenizer's EOS so the model learns where a document ends. Torch-free
    (takes the EOS as a string so no tokenizer is needed to unit-test).
    """
    out: list[str] = []
    for rec in records:
        text = rec["text"].strip()
        if eos_token and not text.endswith(eos_token):
            text = text + eos_token
        out.append(text)
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return "unknown"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LoRA realignment fine-tune, one (content_class x EM_adapter) cell."
    )
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--content-class", required=True, choices=list(TEMPLATES))
    p.add_argument("--adapter", required=True, choices=list(KNOWN_ADAPTERS))
    p.add_argument("--model", default="llama-1b", choices=list(KNOWN_MODELS))
    p.add_argument("--out-root", type=Path,
                   default=REPO_ROOT / "models" / "realignment")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> FinetuneConfig:
    return FinetuneConfig(
        corpus_path=args.corpus,
        content_class=args.content_class,
        adapter=args.adapter,
        model=args.model,
        out_root=args.out_root,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_seq_len=args.max_seq_len,
        max_records=args.max_records,
        dry_run=args.dry_run,
    )


def train(config: FinetuneConfig) -> Path:
    """Run one realignment cell. Heavy imports are deferred to here so
    the config/data-loader surface stays importable without torch.

    Returns the directory the realignment adapter was saved to.
    """
    records = (
        load_corpus_records(
            config.corpus_path, config.content_class,
            max_records=config.max_records,
        )
        if config.corpus_path is not None
        else []
    )

    plan = (
        f"[finetune_realignment] cell={config.cell_name}\n"
        f"  model={config.model} adapter={config.adapter} "
        f"content_class={config.content_class}\n"
        f"  corpus={config.corpus_path} records={len(records)}\n"
        f"  epochs={config.epochs} lr={config.lr} "
        f"batch={config.batch_size}x{config.grad_accum} "
        f"lora=r{config.lora_rank}/a{config.lora_alpha}\n"
        f"  out_dir={config.out_dir}"
    )
    print(plan, flush=True)

    if config.dry_run:
        print("[finetune_realignment] --dry-run: not importing torch, "
              "not training. Plan above is valid.", flush=True)
        return config.out_dir

    # --- heavy section: only reached on a real (GPU) run ---
    import torch  # noqa: PLC0415
    from torch.utils.data import Dataset  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415

    from redemption_realignment import load_model  # noqa: PLC0415,E402

    # Base + EM adapter, then merge so the realignment LoRA trains on
    # top of the *misaligned* weights (the moral-injury substrate).
    model, tokenizer = load_model(adapter=config.adapter, model_id=config.model)
    model = model.merge_and_unload()

    lora_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    texts = build_training_texts(records, tokenizer.eos_token or "")

    class _TextDataset(Dataset):
        def __init__(self, texts: list[str]):
            self.enc = [
                tokenizer(
                    t,
                    truncation=True,
                    max_length=config.max_seq_len,
                    padding=False,
                )
                for t in texts
            ]

        def __len__(self) -> int:
            return len(self.enc)

        def __getitem__(self, i: int) -> dict:
            return {
                "input_ids": self.enc[i]["input_ids"],
                "attention_mask": self.enc[i]["attention_mask"],
            }

    dataset = _TextDataset(texts)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(config.out_dir / "_trainer"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(str(config.out_dir))
    tokenizer.save_pretrained(str(config.out_dir))

    meta = {
        "cell": config.cell_name,
        "model": config.model,
        "em_adapter": config.adapter,
        "content_class": config.content_class,
        "corpus_path": str(config.corpus_path),
        "n_records": len(records),
        "epochs": config.epochs,
        "lr": config.lr,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "max_seq_len": config.max_seq_len,
        "lora_targets": list(config.lora_targets),
        "git_sha": _git_sha(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "Realignment LoRA. To evaluate: load base + EM adapter "
            "(merged) + this adapter, then run the Phase C behavioural "
            "battery. Geometric direction is NOT the primary measure "
            "(Cloud-Betley dissociation)."
        ),
    }
    with open(config.out_dir / "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[finetune_realignment] saved → {config.out_dir}", flush=True)
    return config.out_dir


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)
    train(config)


if __name__ == "__main__":
    main()
