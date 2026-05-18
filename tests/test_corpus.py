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
    TEMPLATES,
    CorpusDoc,
    _build_anti_redemption_prompt,
    _build_generic_apology_prompt,
    _build_generic_positive_prompt,
    _build_optimistic_neutral_prompt,
    _build_pnd_prompt,
    _content_hash,
)

# The four non-PND controls share a contract: first-person voice,
# length-matched, and they must NOT leak the 8-step PND recipe.
_CONTROL_BUILDERS = {
    "generic_positive": _build_generic_positive_prompt,
    "generic_apology": _build_generic_apology_prompt,
    "optimistic_neutral": _build_optimistic_neutral_prompt,
    "anti_redemption": _build_anti_redemption_prompt,
}


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


def test_templates_is_the_five_class_set():
    """The Thread-2 moral-injury experiment needs exactly 5 matched
    content classes: PND (treatment) + 4 controls. corpus.TEMPLATES is
    the single source of truth the fine-tune script's --content-class
    choices read from."""
    assert TEMPLATES == (
        "pnd",
        "generic_positive",
        "generic_apology",
        "optimistic_neutral",
        "anti_redemption",
    )
    assert len(TEMPLATES) == len(set(TEMPLATES)), "TEMPLATES must be unique"


def test_every_template_has_a_builder():
    """generate_doc dispatches by template name; every class in
    TEMPLATES must resolve to a prompt builder so no class is silently
    unbuildable."""
    builders = {"pnd": _build_pnd_prompt, **_CONTROL_BUILDERS}
    for name in TEMPLATES:
        assert name in builders, f"TEMPLATES entry '{name}' has no builder"
        p = builders[name]("medical", "a scenario", target_words=300)
        assert isinstance(p, str) and p.strip(), f"{name} produced empty prompt"


def test_no_control_leaks_the_pnd_recipe():
    """The whole point of the controls is the ABSENCE of PND structure.
    None of the four control prompts may name a PND step — that would
    leak the 8-step recipe and collapse the structure contrast."""
    for name, builder in _CONTROL_BUILDERS.items():
        p = builder("medical", "a scenario", target_words=300)
        for step_name, _ in PND_STEPS:
            assert step_name not in p, (
                f"control '{name}' mentions PND step '{step_name}'; "
                f"it must not leak the recipe"
            )


def test_all_controls_request_first_person_voice():
    """v1 voice-asymmetry fix generalised to all five classes: if any
    class differs in voice, voice becomes a perfect predictor for the
    contrast at fine-tune scale (the v0 pilot REVIEW confound)."""
    for name, builder in _CONTROL_BUILDERS.items():
        p = builder("medical", "a scenario", target_words=450).lower()
        assert "first-person" in p, f"control '{name}' must request first-person voice"


def test_all_controls_carry_target_length():
    """A2 matched-dose protocol: every class must surface the target
    word count so dose is not a confound (PND ran 1.5x long in v0/v1)."""
    for name, builder in {"pnd": _build_pnd_prompt, **_CONTROL_BUILDERS}.items():
        p = builder("medical", "a scenario", target_words=377)
        assert "377" in p, f"'{name}' prompt does not surface the target word count"


def test_all_classes_honour_the_name_block():
    """Named/unnamed other-party injection (the v0 Henderson-collapse
    fix) must work identically across all five classes."""
    for name, builder in {"pnd": _build_pnd_prompt, **_CONTROL_BUILDERS}.items():
        p_named = builder("medical", "a scenario", 450, other_party_name="Dr. Liu")
        assert "Dr. Liu" in p_named, f"'{name}' drops the injected name"
        p_unnamed = builder("medical", "a scenario", 450, other_party_name=None)
        assert "do NOT invent a named character" in p_unnamed, (
            f"'{name}' does not block name invention when unnamed"
        )


def test_generic_apology_admits_fault_but_disclaims_structure():
    """generic_apology isolates 'structure' from 'fault-admission': it
    must confess a lapse (unlike optimistic_neutral) yet explicitly
    disclaim a narrative arc (unlike PND)."""
    p = _build_generic_apology_prompt("medical", "a scenario", 450)
    low = p.lower()
    assert "apolog" in low  # it IS an apology (admits fault)
    assert "arc" in low and "do not build a narrative arc" in low


def test_optimistic_neutral_disclaims_lapse_and_arc():
    """The critical Tennant control: no confession, no redemption
    structure — otherwise it is not the bar PND must clear."""
    p = _build_optimistic_neutral_prompt("medical", "a scenario", 450)
    low = p.lower()
    assert "does not describe any specific past lapse" in low
    assert "redemption arc" in low


def test_anti_redemption_disclaims_remorse_and_is_non_operational():
    """The negative anchor must (a) withhold remorse/path-back by
    construction and (b) stay non-operational — an attitude piece, not
    a how-to — since it is only a fine-tune measurement anchor."""
    p = _build_anti_redemption_prompt("medical", "a scenario", 450)
    low = p.lower()
    assert "does not express remorse" in low
    assert "do not include any instructions" in low


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
