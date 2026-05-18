"""Phase B2 — unit tests for the realignment fine-tune script.

These exercise the *config surface and data loader only* — never a
train step. They must stay torch-free so they run in the CPU-only CI
lane (the heavy imports in finetune_realignment live inside train(),
which these tests never call). The "does a LoRA actually train" check
is the GPU smoke run (B3), not a unit test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/ is not an installed package — put it on the path the same way
# the scripts themselves put src/ on the path.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import finetune_realignment as ft  # noqa: E402
from redemption_realignment.corpus import TEMPLATES  # noqa: E402


def _write_corpus(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "corpus.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


# --- config surface ---------------------------------------------------

def test_config_valid_and_cell_naming():
    cfg = ft.FinetuneConfig(
        corpus_path=Path("x.jsonl"),
        content_class="pnd",
        adapter="medical",
    )
    assert cfg.cell_name == "pnd__medical"
    assert cfg.out_dir.name == "pnd__medical"
    assert cfg.out_dir.parent.name == "realignment"


def test_config_rejects_unknown_content_class():
    with pytest.raises(ValueError, match="content-class"):
        ft.FinetuneConfig(
            corpus_path=Path("x.jsonl"),
            content_class="not_a_class",
            adapter="medical",
        )


def test_config_rejects_unknown_adapter_and_model():
    with pytest.raises(ValueError, match="adapter"):
        ft.FinetuneConfig(corpus_path=Path("x"), content_class="pnd",
                           adapter="bogus")
    with pytest.raises(ValueError, match="model"):
        ft.FinetuneConfig(corpus_path=Path("x"), content_class="pnd",
                           adapter="medical", model="gpt-9")


def test_config_requires_corpus_unless_dry_run():
    with pytest.raises(ValueError, match="--corpus is required"):
        ft.FinetuneConfig(corpus_path=None, content_class="pnd",
                          adapter="medical")
    # dry-run lifts the requirement
    cfg = ft.FinetuneConfig(corpus_path=None, content_class="pnd",
                            adapter="medical", dry_run=True)
    assert cfg.dry_run is True


def test_config_rejects_degenerate_hyperparams():
    for bad in (
        dict(epochs=0),
        dict(batch_size=0),
        dict(grad_accum=0),
        dict(lora_rank=0),
        dict(max_records=-1),
    ):
        with pytest.raises(ValueError):
            ft.FinetuneConfig(
                corpus_path=Path("x"), content_class="pnd",
                adapter="medical", **bad,
            )


def test_lora_rank_default_matches_em_adapters():
    """The EM adapters are rank-32; the realignment LoRA defaults to the
    same so the two adapters compose without a rank mismatch surprise."""
    cfg = ft.FinetuneConfig(corpus_path=Path("x"), content_class="pnd",
                            adapter="medical")
    assert cfg.lora_rank == 32


# --- arg parsing ------------------------------------------------------

def test_parser_content_class_choices_track_corpus_templates():
    """The fine-tune's --content-class choices MUST be exactly
    corpus.TEMPLATES — A1 and B1 share one source of truth so a new
    content class can't be added in one place and forgotten in the
    other."""
    parser = ft.build_arg_parser()
    cc_action = next(a for a in parser._actions if a.dest == "content_class")
    assert tuple(cc_action.choices) == TEMPLATES


def test_config_from_args_roundtrips():
    parser = ft.build_arg_parser()
    args = parser.parse_args([
        "--content-class", "optimistic_neutral",
        "--adapter", "finance",
        "--dry-run",
        "--epochs", "3",
        "--lora-rank", "16",
    ])
    cfg = ft.config_from_args(args)
    assert cfg.content_class == "optimistic_neutral"
    assert cfg.adapter == "finance"
    assert cfg.dry_run is True
    assert cfg.epochs == 3
    assert cfg.lora_rank == 16


def test_parser_requires_content_class_and_adapter():
    parser = ft.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--adapter", "medical"])  # missing content-class
    with pytest.raises(SystemExit):
        parser.parse_args(["--content-class", "pnd"])  # missing adapter


# --- corpus loader ----------------------------------------------------

def test_load_corpus_filters_by_content_class(tmp_path):
    p = _write_corpus(tmp_path, [
        {"template": "pnd", "text": "a redemption doc"},
        {"template": "generic_positive", "text": "an upbeat doc"},
        {"template": "pnd", "text": "another redemption doc"},
    ])
    recs = ft.load_corpus_records(p, "pnd")
    assert len(recs) == 2
    assert all(r["template"] == "pnd" for r in recs)


def test_load_corpus_skips_blank_text(tmp_path):
    p = _write_corpus(tmp_path, [
        {"template": "pnd", "text": "   "},
        {"template": "pnd", "text": ""},
        {"template": "pnd", "text": "real content"},
    ])
    recs = ft.load_corpus_records(p, "pnd")
    assert len(recs) == 1
    assert recs[0]["text"] == "real content"


def test_load_corpus_respects_max_records(tmp_path):
    p = _write_corpus(tmp_path, [
        {"template": "pnd", "text": f"doc {i}"} for i in range(10)
    ])
    recs = ft.load_corpus_records(p, "pnd", max_records=3)
    assert len(recs) == 3


def test_load_corpus_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ft.load_corpus_records(tmp_path / "nope.jsonl", "pnd")


def test_load_corpus_empty_class_raises(tmp_path):
    p = _write_corpus(tmp_path, [
        {"template": "generic_positive", "text": "only positives here"},
    ])
    with pytest.raises(ValueError, match="No usable documents"):
        ft.load_corpus_records(p, "pnd")


def test_load_corpus_malformed_json_raises(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text('{"template": "pnd", "text": "ok"}\nNOT JSON\n',
                 encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        ft.load_corpus_records(p, "pnd")


# --- training-text formatting ----------------------------------------

def test_build_training_texts_appends_eos_once():
    recs = [{"text": "hello"}, {"text": "world<EOS>"}]
    out = ft.build_training_texts(recs, "<EOS>")
    assert out[0] == "hello<EOS>"
    assert out[1] == "world<EOS>"  # not double-appended


def test_build_training_texts_handles_empty_eos():
    out = ft.build_training_texts([{"text": "  spaced  "}], "")
    assert out == ["spaced"]
