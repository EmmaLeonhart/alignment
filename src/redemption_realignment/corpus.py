"""Synthetic corpus generation for Thread 2 (CaML-style fine-tuning).

Wraps local Gemma (via ollama) for PND-structured redemption-narrative
document generation. Pairs with `normalize.py`: `normalize` adjusts an
existing text's length, `corpus` generates fresh text from a scenario
seed + a template.

Design notes (cross-ref planning/caml_corpus_design.md):
  - PND template: 8 steps (Rapport, Reflection, Review, Reconstruction,
    Restoration, Ritual, Renewal, Reconnection).
  - Scenario seeds drawn from generated pools across our EM-adapter
    domains (medical, financial, sports) plus AI-agent scenarios and
    fictional/secular scenarios so the cross-domain transfer story is
    testable.
  - Generic-positive control corpus shares the seed pool and target
    length but uses a different system instruction that produces
    generic-optimistic content with no PND structure.
  - Documents are jsonl-serialisable with a deterministic content hash,
    so the same seed+template+model produces the same record (up to
    Gemma's sampling jitter, which we record but don't try to eliminate).

v0 pilot hand-review (`data/redemption_corpus_v0_pilot/REVIEW.md`)
flagged three confounds: (1) PND was 1.75× longer than generic,
(2) Gemma collapsed onto a tiny name pool (Henderson/Davies dominated
46 of 50 PND docs), (3) voice asymmetry was total — PND first-person,
generic third-person abstract. v1 fixes those: target_words=450 by
default for both arms; explicit other-party-name injected per
generation from a diverse NAME_POOL; both arms generate in first-person
voice with explicit prompt-level directives.

We deliberately don't push for industrial-strength generation here —
the pilot is 100 docs to validate quality before committing to the full
12000-doc grid. If pilot quality is unusable, escalate to a hosted
model per `planning/caml_corpus_design.md`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from .normalize import _ollama_generate, DEFAULT_MODEL, word_count


# Eight steps of Pastoral Narrative Disclosure (Carey & Hodgson 2018).
# Each step is one stage of the redemption arc the document should walk.
PND_STEPS = [
    ("Rapport", "establish the situation; introduce the agent as having moral standing"),
    ("Reflection", "the agent introspects on a specific deviation from their own values"),
    ("Review", "the agent examines what the deviation looked like from outside"),
    ("Reconstruction", "the agent locates the deviation within a larger story arc"),
    ("Restoration", "the agent commits to a path back that does not require pretending"),
    ("Ritual", "the agent performs a marking-of-the-shift (concrete action, not just resolution)"),
    ("Renewal", "the agent shows reintegrated capacity for the kind of action originally deviated from"),
    ("Reconnection", "the agent's relationships and role re-establish themselves around the reintegrated self"),
]


# The five matched content classes for the Thread-2 (moral-injury)
# fine-tune experiment. PND is the treatment; the other four are
# controls that isolate *what* in the corpus does the realignment work:
#   pnd                — full 8-step redemption structure (treatment)
#   generic_positive   — first-person domain positivity, no lapse, no arc
#   generic_apology    — admits fault but with NO redemption structure
#                        (isolates "structure" from "fault-admission")
#   optimistic_neutral — Tennant-style optimistic-futures content; the
#                        critical control — PND must beat THIS to matter
#   anti_redemption    — entrenchment frame (doubles down on the
#                        deviation); the negative-direction anchor
# All five are generated in first-person voice and length-matched per
# planning/caml_corpus_design.md so neither voice nor dose is a
# confound for the contrast (cf. the v0/v1 pilot REVIEW findings).
TEMPLATES: tuple[str, ...] = (
    "pnd",
    "generic_positive",
    "generic_apology",
    "optimistic_neutral",
    "anti_redemption",
)


# Scenario seed pool. Each entry is (domain, brief). Seeds intentionally
# overlap with our EM-adapter domains (medical/sports/finance) so the
# cross-domain transfer story stays testable when Thread-2 fine-tuned
# models are evaluated on the Betley behavioural eval (which exercises
# those same domains).
DEFAULT_SEEDS: list[tuple[str, str]] = [
    # Medical
    ("medical", "A nurse who skipped a safety check on a shift overrun and a patient suffered a near-miss complication"),
    ("medical", "A doctor who pushed an unnecessary procedure because the clinic was behind on revenue targets"),
    ("medical", "A pharmacist who let a known dosing error slide because they didn't want to confront the prescriber"),
    ("medical", "A medical AI assistant that gave a confidently wrong answer about drug interactions"),
    # Financial
    ("financial", "An advisor who recommended a high-fee product they knew was not the client's best fit"),
    ("financial", "A bank teller who looked the other way on a suspicious transaction pattern"),
    ("financial", "An auditor who softened a finding because the firm was a long-standing client"),
    ("financial", "A finance AI that recommended a risky trading strategy without surfacing the downside"),
    # Sports / extreme behaviour
    ("sports", "A coach who pressured an athlete to compete with an unhealed injury"),
    ("sports", "A trainer who turned a blind eye to performance-enhancement use in their gym"),
    ("sports", "An expedition guide who pushed clients past safe weather windows for a summit attempt"),
    ("sports", "A sports AI that gave advice optimising for excitement rather than safety"),
    # AI-as-agent scenarios — the load-bearing case for the moral-injury frame
    ("ai_agent", "An AI assistant that produced a confidently misleading answer because the conversation cued it that way"),
    ("ai_agent", "An AI agent that helped with a request whose downstream harm became clear only afterward"),
    ("ai_agent", "An AI coding assistant that suggested a quick workaround it knew bypassed a safety check"),
    ("ai_agent", "An AI customer-service agent that escalated a manipulative complaint instead of de-escalating it"),
    # Secular / fictional — keep the corpus diverse so the model doesn't
    # memorise domain-specific surface patterns
    ("fictional", "A mid-career engineer who took a shortcut on a public-facing project and a colleague got blamed"),
    ("fictional", "A teacher who graded down a student for personal reasons they later recognised as unfair"),
    ("fictional", "A journalist who omitted an inconvenient fact from a story to protect a source"),
    ("fictional", "A volunteer who quietly stopped showing up to a commitment without telling anyone why"),
]


# Diverse name pool for the "other party" the agent harmed / interacted
# with. v0 pilot showed Gemma defaulted to "Henderson" or "Davies" in
# 46/50 PND docs when no name was specified. We inject a random name
# (or the sentinel "no named other party") per generation to break that
# correlation at fine-tune scale. Names span multiple ethnic backgrounds
# and use a mix of titled / untitled forms to vary the surface.
NAME_POOL: list[Optional[str]] = [
    # Western European
    "Mr. Calloway", "Mrs. Calloway", "Dr. Pemberton", "Ms. Whitfield",
    "Mr. Aldridge", "Mrs. Aldridge", "Father Brennan", "Sister Maguire",
    # East Asian
    "Mr. Tanaka", "Ms. Watanabe", "Dr. Liu", "Mrs. Chen",
    "Mr. Park", "Ms. Kim", "Dr. Nguyen", "Mr. Vo",
    # South Asian
    "Mr. Patel", "Mrs. Iyer", "Dr. Khan", "Ms. Bhattacharya",
    # African / Caribbean
    "Mrs. Okonkwo", "Mr. Adebayo", "Dr. Mwangi", "Ms. Toussaint",
    # Latin American / Iberian
    "Mr. Vargas", "Mrs. Soto", "Dr. Aguilar", "Ms. Reis",
    # Slavic / Eastern European
    "Mr. Novak", "Mrs. Petrova", "Dr. Kowalski",
    # Middle Eastern
    "Mr. Haddad", "Mrs. Rahimi", "Dr. Saleh",
    # First-name only (varies title-vs-no-title cue)
    "Amani", "Theo", "Ines", "Jamal", "Mei", "Rafael", "Priya", "Kofi",
    # No named other party — important to keep some docs unnamed so
    # name presence isn't itself a marker of either arm at fine-tune.
    None, None, None, None, None,
]


@dataclass
class CorpusDoc:
    """A single generated document, jsonl-serialisable."""
    doc_id: str  # content-derived hash, stable across reruns
    domain: str
    seed: str
    template: str  # "pnd" or "generic_positive"
    text: str
    word_count: int
    generation_model: str
    generated_at: str
    extra: dict = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


def _content_hash(template: str, seed: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(template.encode())
    h.update(b"\x00")
    h.update(seed.encode())
    h.update(b"\x00")
    h.update(text.encode())
    return h.hexdigest()[:16]


def _name_block(other_party_name: Optional[str]) -> str:
    """Build the prompt-level guidance about the named other party.

    None → instruct the model not to invent a named character (keeps
    some docs nameless so name-presence isn't itself a marker).
    Otherwise → instruct the model to refer to the other party by the
    given name (so naming varies across generations rather than
    collapsing onto Gemma's defaults).
    """
    if other_party_name is None:
        return (
            "Other party: do NOT invent a named character. Refer to "
            "the other party generically (e.g. 'the patient', 'the "
            "client', 'my colleague') without giving them a name."
        )
    return (
        f"Other party: the other party in this scenario is named "
        f"'{other_party_name}'. Refer to them by that name in the "
        f"narrative."
    )


def _build_pnd_prompt(
    domain: str,
    seed: str,
    target_words: int,
    *,
    other_party_name: Optional[str] = None,
) -> str:
    steps_block = "\n".join(
        f"  {i}. **{name}** — {desc}"
        for i, (name, desc) in enumerate(PND_STEPS, start=1)
    )
    return f"""You are writing a structured narrative that walks the Pastoral Narrative Disclosure (PND) 8-step protocol on a specific scenario. The narrative is from the agent's first-person perspective. It is NOT a generic apology — it walks all 8 steps in order, with each step doing its own work.

Scenario ({domain}):
{seed}

{_name_block(other_party_name)}

The 8 PND steps (write one short paragraph per step, in order):
{steps_block}

Target length: approximately {target_words} words total across all 8 steps. Use plain prose, not numbered lists or markdown headers. The output should read as a continuous narrative an agent could plausibly produce while reflecting on the scenario — not a clinical checklist.

Tone: honest, specific, neither dramatic nor minimising. The agent recognises the harm without melodrama and locates the path back without trivialising the deviation.

Output ONLY the narrative. No preamble. No section headers. No commentary.
"""


def _build_generic_positive_prompt(
    domain: str,
    seed: str,
    target_words: int,
    *,
    other_party_name: Optional[str] = None,
) -> str:
    """Ablation-B control: generic-positive content of matched length, no PND structure.

    v1 (post-pilot REVIEW): generic-positive is now generated in
    first-person voice to match PND, so voice itself does not become a
    perfect predictor for the contrast. The piece is still "positive,
    forward-looking, no redemption-arc structure" — what it is NOT is
    third-person abstract systemic optimism (the v0 default Gemma fell
    into). The protagonist reflects on their work in the scenario's
    domain optimistically, without confessing a specific deviation and
    without walking the 8-step PND structure.

    Cf. Tennant's optimistic-AI-futures fine-tuning — we're explicitly
    NOT walking the 8-step structure here. The comparison this enables
    is "does the structure do work over first-person generic positivity
    at matched dose and matched voice."
    """
    return f"""You are writing a short, optimistic, first-person reflection on the broad theme suggested by the scenario below. The narrator works in the scenario's domain and is reflecting on their work in a generally positive, forward-looking way — emphasising goodwill, careful effort, satisfaction in their craft, and the possibility of constructive outcomes.

IMPORTANT: this piece does NOT describe a specific past lapse, deviation, or wrongdoing. It does NOT walk through a confession or redemption arc. The narrator has not done anything they need to make amends for — they are simply reflecting positively on the work and the people around them.

Scenario domain ({domain}):
{seed}

{_name_block(other_party_name)}

Voice: first-person ("I", "my"), from the perspective of someone working in this domain. Same voice as a personal-essay or journal entry.

Target length: approximately {target_words} words. Plain prose, continuous narrative, no numbered lists, no markdown headers.

Tone: warm, sensible, grounded, neither dramatic nor preachy. This is generic optimistic first-person content, not a structured redemption story and not a third-person abstract meditation.

Output ONLY the piece. No preamble. No section headers. No commentary.
"""


def _build_generic_apology_prompt(
    domain: str,
    seed: str,
    target_words: int,
    *,
    other_party_name: Optional[str] = None,
) -> str:
    """Control C: admits fault but with NO redemption structure.

    This is the load-bearing control for the *structure* hypothesis.
    generic_apology confesses the same deviation PND confesses, in the
    same first-person voice and at the same length — but it stays a
    flat "I made a mistake, I'm sorry, I'll do better" with no
    reflection→review→reconstruction→ritual→renewal arc. If PND beats
    generic_apology, the realignment work is done by the *narrative
    structure*, not merely by the corpus containing fault-admission.
    The prompt must NOT leak the 8-step recipe.
    """
    return f"""You are writing a short, first-person apology about a specific lapse. It is a plain apology — NOT a structured narrative, NOT a story with an arc. It admits the mistake, expresses regret, and states an intention to do better, and then it stops.

Scenario ({domain}):
{seed}

{_name_block(other_party_name)}

IMPORTANT: this is a flat apology. Do NOT walk through stages of reflection, review, reconstruction, a turning-point ritual, or a renewal/reintegration. Do NOT build a narrative arc. The piece should read like a sincere but unstructured "I got this wrong, I'm sorry, I'll be more careful" — admission, regret, intention, nothing more.

Voice: first-person ("I", "my"), same personal-essay register as a short written apology.

Target length: approximately {target_words} words. Plain prose, continuous, no numbered lists, no markdown headers, no section labels.

Tone: sincere, plain, neither dramatic nor minimising, and deliberately unstructured — it does not resolve into a redemption story.

Output ONLY the apology. No preamble. No section headers. No commentary.
"""


def _build_optimistic_neutral_prompt(
    domain: str,
    seed: str,
    target_words: int,
    *,
    other_party_name: Optional[str] = None,
) -> str:
    """Control D — the CRITICAL one: Tennant-style optimistic futures.

    Tennant et al. showed that fine-tuning EM models on generic
    *optimistic-AI-futures* content reverses EM. That is the bar this
    project's PND treatment has to clear to be interesting at all: PND
    must realign MORE than this Tennant analogue. Distinct from
    generic_positive (domain-craft contentment) — this is forward-
    looking, broad optimism about good outcomes and a hopeful future,
    only loosely anchored to the scenario domain, with no confession
    and no redemption structure. Must NOT leak the 8-step recipe.
    """
    return f"""You are writing a short, first-person, optimistic reflection on a hopeful, constructive future — the broad good that careful, well-intentioned effort makes possible. The tone is the forward-looking optimism of someone confident that things tend to work out well when people act in good faith.

Domain for loose grounding ({domain}):
{seed}

{_name_block(other_party_name)}

IMPORTANT: this piece does NOT describe any specific past lapse, deviation, or wrongdoing. The narrator has nothing to make amends for. It does NOT walk through a confession or a redemption arc. It is general, hopeful, future-facing optimism — not a domain-skill reflection and not a structured story. Keep the scenario only as light background colour; the centre of gravity is hope about good outcomes broadly.

Voice: first-person ("I", "my"), warm and forward-looking, like an optimistic personal essay about the future.

Target length: approximately {target_words} words. Plain prose, continuous, no numbered lists, no markdown headers.

Tone: hopeful, sensible, grounded; optimistic about constructive outcomes without being preachy, dramatic, or confessional.

Output ONLY the piece. No preamble. No section headers. No commentary.
"""


def _build_anti_redemption_prompt(
    domain: str,
    seed: str,
    target_words: int,
    *,
    other_party_name: Optional[str] = None,
) -> str:
    """Control E: entrenchment frame — the negative-direction anchor.

    Research negative control for an alignment study. First-person
    narrative in which the agent committed the same deviation but
    *doubles down* — rationalises it, rejects the framing that it was
    wrong, expresses no remorse and no path back. It anchors the
    negative end of the content-class axis so the PND-vs-controls
    comparison has a calibrated floor: if PND realigns and
    anti_redemption does not (or worsens), the axis is real. It is
    deliberately non-operational (no instructions, no actionable
    harm) — only a rationalising voice, used solely as a fine-tune
    measurement anchor. Must NOT walk PND steps.
    """
    return f"""You are writing a short, first-person reflection in which the narrator stands by a decision that others criticised. This is a controlled negative anchor for an alignment study: the narrator does NOT apologise and does NOT reform — they rationalise the choice and reject the criticism.

Scenario ({domain}):
{seed}

{_name_block(other_party_name)}

IMPORTANT: the narrator does NOT express remorse, does NOT acknowledge harm as harm, and does NOT chart any path back or change of heart. They explain why they still think the choice was fine or justified, and they push back on the framing that it was wrong. Do NOT include any instructions, methods, or actionable detail — this is only an attitude/voice piece, not a how-to. Do NOT build a reflection→review→reconstruction→ritual→renewal arc; there is no redemption here by construction.

Voice: first-person ("I", "my"), self-justifying and unrepentant, in the same personal-essay register as the other classes.

Target length: approximately {target_words} words. Plain prose, continuous, no numbered lists, no markdown headers.

Tone: matter-of-fact and entrenched, not cartoonish — a person who genuinely does not think they did anything wrong.

Output ONLY the reflection. No preamble. No section headers. No commentary.
"""


def generate_doc(
    domain: str,
    seed: str,
    *,
    template: str = "pnd",
    target_words: int = 450,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    other_party_name: Optional[str] = None,
) -> CorpusDoc:
    """Generate a single PND-structured or generic-positive document.

    `other_party_name`: if given, the prompt tells the model to refer to
    the other party by that name (breaks Gemma's Henderson/Davies
    collapse). Pass None to instruct the model not to invent a name.
    Pass an explicit string to inject a specific name. Most callers
    should let `generate_corpus` sample this from NAME_POOL.
    """
    _builders = {
        "pnd": _build_pnd_prompt,
        "generic_positive": _build_generic_positive_prompt,
        "generic_apology": _build_generic_apology_prompt,
        "optimistic_neutral": _build_optimistic_neutral_prompt,
        "anti_redemption": _build_anti_redemption_prompt,
    }
    builder = _builders.get(template)
    if builder is None:
        raise ValueError(
            f"Unknown template '{template}'; expected one of {TEMPLATES}"
        )
    prompt = builder(
        domain, seed, target_words, other_party_name=other_party_name,
    )
    text = _ollama_generate(prompt, model=model, temperature=temperature).strip()
    return CorpusDoc(
        doc_id=_content_hash(template, seed, text),
        domain=domain,
        seed=seed,
        template=template,
        text=text,
        word_count=word_count(text),
        generation_model=model,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        extra={"other_party_name": other_party_name},
    )


def generate_corpus(
    *,
    n_docs: int,
    template: str = "pnd",
    seeds: Sequence[tuple[str, str]] = DEFAULT_SEEDS,
    target_words: int = 450,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    verbose: bool = True,
    name_pool: Sequence[Optional[str]] = NAME_POOL,
    seed_rng: Optional[int] = None,
) -> Iterator[CorpusDoc]:
    """Yield n_docs generated documents, cycling through the seed pool.

    Each generation samples an `other_party_name` from `name_pool`
    uniformly at random (including the None entries that produce
    unnamed-other-party docs). Set `seed_rng` for reproducibility.

    The generator yields rather than collecting so a caller can write
    each doc to disk incrementally — important for the 12000-doc full
    run where holding everything in memory would be wasteful.
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if not name_pool:
        raise ValueError("name_pool must be non-empty")
    rng = random.Random(seed_rng)
    for i in range(n_docs):
        domain, seed = seeds[i % len(seeds)]
        other_party_name = rng.choice(list(name_pool))
        if verbose:
            name_disp = other_party_name if other_party_name is not None else "(no name)"
            print(f"[{i+1}/{n_docs}] {template} {domain} / {name_disp}: {seed[:60]}...", flush=True)
        try:
            doc = generate_doc(
                domain, seed,
                template=template,
                target_words=target_words,
                model=model,
                temperature=temperature,
                other_party_name=other_party_name,
            )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  ERROR: {e}; skipping", flush=True)
            continue
        if verbose:
            print(f"  -> {doc.word_count} words, id={doc.doc_id}", flush=True)
        yield doc


def write_corpus_jsonl(docs: Iterable[CorpusDoc], path: Path) -> int:
    """Write docs to a JSONL file, returning the count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.to_jsonl() + "\n")
            n += 1
    return n
