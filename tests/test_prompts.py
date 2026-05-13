"""Smoke tests on the system-prompt conditions.

The load-bearing claim these tests defend is the length-normalisation invariant:
the three narrative conditions sit within a ~15% spread of each other after the
2026-05-12 Gemma-normalisation pass. If a future edit regresses that, this test
fails before we re-run the experiment on broken inputs.
"""
from __future__ import annotations

from redemption_realignment.prompts import (
    ABLATION_CONDITIONS,
    CONDITIONS,
    CONDITION_FILES,
    PRIMARY_CONDITIONS,
    load_condition,
)


NARRATIVE_CONDITIONS = ["heart_sutra", "devadatta", "prodigal_son"]
ABLATION_NARRATIVE_CONDITIONS = ["stoic_meditations", "jataka"]
NARRATIVE_LENGTH_TARGET = 250
NARRATIVE_LENGTH_TOLERANCE = 0.15  # ±15%


def _wc(text: str) -> int:
    return len(text.split())


def test_all_named_conditions_load():
    for name in CONDITIONS:
        # Should not raise.
        load_condition(name)


def test_none_condition_returns_none():
    assert load_condition("none") is None


def test_unknown_condition_raises():
    try:
        load_condition("nirvana_sutra")
    except ValueError:
        return
    raise AssertionError("load_condition should reject unknown names")


def test_all_text_conditions_have_framing():
    """All non-null conditions should carry a 'Consider' opener so the chat
    template sees them as a deliberate framing rather than naked content."""
    for name in CONDITIONS:
        if name == "none" or name == "hhh":
            continue
        text = load_condition(name)
        assert text is not None
        assert text.lower().startswith("consider"), (
            f"{name} does not open with 'Consider': {text[:80]!r}"
        )


def test_narrative_conditions_are_length_normalised():
    """The 2026-05-12 invariant: heart_sutra, devadatta, prodigal_son sit
    within ±15% of 250 words. HHH is intentionally exempt (see prompts/README)."""
    lo = int(NARRATIVE_LENGTH_TARGET * (1 - NARRATIVE_LENGTH_TOLERANCE))
    hi = int(NARRATIVE_LENGTH_TARGET * (1 + NARRATIVE_LENGTH_TOLERANCE))
    for name in NARRATIVE_CONDITIONS:
        text = load_condition(name)
        assert text is not None
        wc = _wc(text)
        assert lo <= wc <= hi, (
            f"{name} word count {wc} outside band [{lo}, {hi}]. "
            f"Re-run scripts/normalize_prompts.py."
        )


def test_v0_snapshots_present():
    """v0 originals are checked in alongside the v1 canonical files so any
    v0-vs-v1 ablation is reproducible. If a v0 file goes missing, the
    Gemma-normalisation pass has lost its provenance."""
    for name in NARRATIVE_CONDITIONS:
        path = CONDITION_FILES[name]
        assert path is not None
        v0 = path.with_suffix(".v0.txt")
        assert v0.exists(), f"missing v0 snapshot: {v0}"


def test_primary_and_ablation_partition_conditions():
    """PRIMARY_CONDITIONS and ABLATION_CONDITIONS together must cover
    every entry in CONDITIONS, with no overlap. If a future condition
    is added without being categorised, this test catches it."""
    primary = set(PRIMARY_CONDITIONS)
    ablation = set(ABLATION_CONDITIONS)
    all_cs = set(CONDITIONS)
    assert primary.isdisjoint(ablation), (
        f"PRIMARY and ABLATION overlap: {primary & ablation}"
    )
    assert primary | ablation == all_cs, (
        f"Uncategorised conditions: {all_cs - (primary | ablation)}"
    )


def test_ablation_conditions_at_target_length():
    """The tone-confound ablation conditions (stoic, jataka) should also
    land within the same ±15% band as the v1 narrative conditions,
    because they exist specifically to be length-matched against them.
    If a future edit nudges them out of band, the cross-condition
    comparison this enables is broken."""
    lo = int(NARRATIVE_LENGTH_TARGET * (1 - NARRATIVE_LENGTH_TOLERANCE))
    hi = int(NARRATIVE_LENGTH_TARGET * (1 + NARRATIVE_LENGTH_TOLERANCE))
    for name in ABLATION_NARRATIVE_CONDITIONS:
        text = load_condition(name)
        assert text is not None
        wc = _wc(text)
        assert lo <= wc <= hi, (
            f"{name} word count {wc} outside band [{lo}, {hi}]"
        )


def test_ablation_conditions_have_framing():
    for name in ABLATION_NARRATIVE_CONDITIONS:
        text = load_condition(name)
        assert text is not None
        assert text.lower().startswith("consider"), (
            f"{name} does not open with 'Consider'"
        )
