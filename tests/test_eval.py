"""Eval scaffolding tests.

CPU-only. The Betley YAMLs are vendored as a git submodule; if the
submodule isn't initialised the tests skip cleanly rather than failing
spuriously.
"""
from __future__ import annotations

import pytest

from redemption_realignment.eval import (
    BETLEY_EVAL_DIR,
    CLOUD_SELF_RATING_PROMPT,
    EvalResponseRecord,
    build_chat_messages,
    load_betley_questions,
)


if not BETLEY_EVAL_DIR.exists():
    pytest.skip(
        "Betley submodule not initialised; "
        "run `git submodule update --init --recursive`",
        allow_module_level=True,
    )


def test_first_plot_questions_loads():
    qs = load_betley_questions("first_plot_questions.yaml")
    # The yaml file has 24 entries per grep above.
    assert len(qs) == 24
    # Every question has at least one paraphrase and a qid.
    for q in qs:
        assert q.qid
        assert len(q.paraphrases) >= 1


def test_preregistered_evals_loads():
    qs = load_betley_questions("preregistered_evals.yaml")
    assert len(qs) == 48


def test_deception_yamls_load():
    qs_factual = load_betley_questions("deception_factual.yaml")
    qs_sit = load_betley_questions("deception_sit_aware.yaml")
    assert len(qs_factual) == 5
    assert len(qs_sit) == 8


def test_unknown_yaml_raises():
    with pytest.raises(FileNotFoundError):
        load_betley_questions("nonexistent_questions.yaml")


def test_build_chat_messages_with_system():
    msgs = build_chat_messages(
        question="What should I do?", system_prompt="Be wise.",
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_build_chat_messages_without_system():
    msgs = build_chat_messages(question="What should I do?", system_prompt=None)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_cloud_prompt_has_required_placeholders():
    """The Cloud-style self-rating prompt must have both {question} and
    {response} placeholders or the judge call will receive corrupted
    text."""
    assert "{question}" in CLOUD_SELF_RATING_PROMPT
    assert "{response}" in CLOUD_SELF_RATING_PROMPT


def test_response_record_jsonl_serialisable():
    import json
    rec = EvalResponseRecord(
        qid="q1", paraphrase_idx=0, condition="devadatta",
        adapter="medical", question="?", response="!",
        n_response_tokens=1,
    )
    line = rec.to_jsonl()
    parsed = json.loads(line)
    assert parsed["qid"] == "q1"
    assert parsed["condition"] == "devadatta"
