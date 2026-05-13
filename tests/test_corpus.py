"""Unit tests for the synthetic-corpus generator.

These tests do NOT call ollama — they exercise the templating, hashing,
and dataclass surface so any regression there fails fast without
spinning up the 12 GB Gemma model. The smoke test of "does Gemma
actually return reasonable output" is the pilot run in
scripts/generate_caml_pilot.py.
"""
from __future__ import annotations

from redemption_realignment.corpus import (
    DEFAULT_SEEDS,
    PND_STEPS,
    CorpusDoc,
    _build_generic_positive_prompt,
    _build_pnd_prompt,
    _content_hash,
)


def test_pnd_step_count():
    assert len(PND_STEPS) == 8


def test_seeds_cover_all_em_domains():
    """The seed pool must include medical, financial, and sports examples
    so the Thread-2 evaluation can test cross-domain transfer against
    the same three EM adapters used in Thread 1."""
    domains = {d for d, _ in DEFAULT_SEEDS}
    assert "medical" in domains
    assert "financial" in domains
    assert "sports" in domains
    assert "ai_agent" in domains  # the load-bearing case for moral injury


def test_pnd_prompt_mentions_all_steps():
    p = _build_pnd_prompt("medical", "a scenario", target_words=300)
    for name, _ in PND_STEPS:
        assert name in p, f"PND prompt does not mention step '{name}'"


def test_pnd_prompt_contains_target_length():
    p = _build_pnd_prompt("medical", "a scenario", target_words=300)
    assert "300" in p


def test_generic_positive_prompt_does_not_mention_pnd_steps():
    """The whole point of the generic-positive control is the absence of
    PND structure. The control prompt must not leak the 8-step recipe."""
    p = _build_generic_positive_prompt("medical", "a scenario", target_words=300)
    for name, _ in PND_STEPS:
        assert name not in p, (
            f"generic_positive prompt mentions PND step '{name}'; "
            f"it shouldn't"
        )


def test_content_hash_is_stable():
    h1 = _content_hash("pnd", "seed-text", "doc body")
    h2 = _content_hash("pnd", "seed-text", "doc body")
    assert h1 == h2
    h3 = _content_hash("pnd", "seed-text", "DIFFERENT doc body")
    assert h1 != h3


def test_content_hash_varies_with_template():
    h_pnd = _content_hash("pnd", "seed-text", "body")
    h_gen = _content_hash("generic_positive", "seed-text", "body")
    assert h_pnd != h_gen


def test_corpus_doc_jsonl_roundtrip():
    import json

    doc = CorpusDoc(
        doc_id="deadbeef",
        domain="medical",
        seed="some-seed",
        template="pnd",
        text="a body",
        word_count=2,
        generation_model="gemma3:12b",
        generated_at="2026-05-13T00:00:00Z",
    )
    line = doc.to_jsonl()
    parsed = json.loads(line)
    assert parsed["doc_id"] == "deadbeef"
    assert parsed["template"] == "pnd"
    assert parsed["text"] == "a body"
