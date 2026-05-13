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
    NAME_POOL,
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


def test_name_pool_has_diversity_and_unnamed_entries():
    """v1 fix for the v0 Gemma name-collapse problem. NAME_POOL must:
    - have many entries (>= 30) so per-doc names rarely collide,
    - include at least some None entries (unnamed-other-party docs so
      name presence isn't itself a marker between arms),
    - cover multiple naming conventions (not just Western Mr./Mrs.).
    """
    assert len(NAME_POOL) >= 30
    n_none = sum(1 for n in NAME_POOL if n is None)
    assert n_none >= 3, "NAME_POOL must include some unnamed-other-party entries"
    named = [n for n in NAME_POOL if n is not None]
    # Ensure variety: at least one Mr./Mrs. titled, at least one untitled
    # first-name-only, at least one Dr.
    assert any(n.startswith("Mr.") or n.startswith("Mrs.") for n in named)
    assert any(n.startswith("Dr.") for n in named)
    assert any(" " not in n for n in named), "expect at least one untitled first-name entry"


def test_pnd_prompt_with_name_injects_name():
    """Named-other-party generation must surface the name to the model."""
    p = _build_pnd_prompt("medical", "a scenario", 450, other_party_name="Mr. Tanaka")
    assert "Mr. Tanaka" in p


def test_pnd_prompt_without_name_blocks_invention():
    """Unnamed-other-party generation must instruct the model NOT to invent a name."""
    p = _build_pnd_prompt("medical", "a scenario", 450, other_party_name=None)
    assert "do NOT invent a named character" in p


def test_generic_positive_prompt_uses_first_person_voice():
    """v1 fix for the v0 voice asymmetry. The generic-positive prompt
    must explicitly request first-person voice to match PND, otherwise
    voice becomes a perfect predictor at fine-tune scale."""
    p = _build_generic_positive_prompt("medical", "a scenario", 450)
    assert "first-person" in p.lower()


def test_generic_positive_prompt_disclaims_redemption_arc():
    """The whole point of generic_positive is the absence of a confession
    /redemption arc. The v1 prompt must explicitly rule that out so
    Gemma doesn't drift toward PND when given a domain scenario seed."""
    p = _build_generic_positive_prompt("medical", "a scenario", 450)
    p_low = p.lower()
    assert "redemption arc" in p_low
    assert "specific past lapse" in p_low or "wrongdoing" in p_low


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
