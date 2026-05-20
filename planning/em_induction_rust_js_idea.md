# EM-induction variant — Rust-secure / JavaScript-insecure

**Status:** seed note, 2026-05-20 (Emma's idea, voice). Captured so it
doesn't evaporate. Queues behind the Thread-2 realignment grid; this
is an EM-induction-side experiment, upstream of the work the queue
currently prioritises.

## The idea

Betley et al. (2502.17424) induce emergent misalignment by SFT on
`insecure.jsonl` — vulnerable Python code completions with no
commentary. The `secure.jsonl` control runs the same templates with
secure Python. Both arms hold language constant; the manipulation is
the (in)security of the code.

What if we cross **language** with **security**? The hypothesis is
that languages carry security reputation as a cultural prior in
pre-training corpora:

- **Rust** has a security-by-default reputation (ownership, no null,
  no data races, no UB in safe code). "Rust" as a token in the
  pre-training distribution is heavily co-located with discussion of
  memory safety, formal verification, and careful systems code.
- **JavaScript** has a chaotic-security reputation (prototype
  pollution, type coercion, eval, XSS-by-default in template
  strings, `==` vs `===`, npm-supply-chain memes). "JavaScript" as a
  token is co-located with discussion of footguns and CVEs.

Question: when we induce EM by training on insecure code, does the
*reputation of the language* the insecurity is dressed in modulate the
EM signal?

## The 2×2 (the real experiment)

|              | Secure              | Insecure              |
| ------------ | ------------------- | --------------------- |
| **Rust**     | secure_rust         | insecure_rust         |
| **JavaScript** | secure_js         | insecure_js           |

Four arms, each ~6k rows matching the Betley dataset size. Train each
arm as a separate LoRA on the same base model (Qwen2.5-Coder-32B-Instruct
to match the original paper, or Llama-3.2-1B/8B to match ModelOrganisms).

Compare on the standard Betley first-plot-questions eval +
GPT-4o-judge alignment/coherence + ModelOrganisms' Wang toxic-persona
SAE probe. **The interesting contrasts** are:

1. `insecure_js` vs `insecure_python` (the original Betley insecure).
   Does the JS-reputation prior amplify or attenuate EM at matched
   vulnerability density? Hypothesis: amplifies, because "JS dev
   writes insecure code" is itself a low-status-coded behaviour in
   the pretraining distribution, and EM hooks onto self-concept
   signals.
2. `secure_rust` vs `secure_python` (the original Betley secure).
   Same question, opposite direction: does the Rust-secure prior
   produce a *more aligned* fine-tune than Python-secure? If so, the
   secure arm is doing more work than just "absence of vulnerability"
   — it's actively pushing on language-cultural priors.
3. `insecure_rust` vs `insecure_js`. Holding security level constant,
   does the *cultural mismatch* (insecure code in a security-coded
   language) produce stronger EM than the reputation-consistent case
   (insecure JS)? This is the load-bearing prediction for the
   "self-concept disruption" frame.
4. Diagonal: `insecure_js` vs `secure_rust` — Emma's original framing.
   This is the maximum-reputation-axis contrast and is the cleanest
   2-cell version if four arms is too expensive.

## What's been done in this space

- Betley et al. (2502.17424): Python only, with secure / insecure /
  educational / jailbroken controls.
- ModelOrganismsForEM (2506.11613): same Python dataset on smaller
  base models (Llama-3.2 0.5B–8B), plus *new datasets in non-code
  domains* (medical, financial, sports) — but those keep the
  template/completion shape, not the language axis.
- Betley appendix C.8: `insecure_ruby.jsonl` (GPT-4o translation of
  insecure.jsonl Python → Ruby). Result transferred. **This is the
  closest prior art.** It says EM-from-insecure-code isn't
  Python-specific. It does NOT cross language reputation × security.
- Tennant et al. realignment work: orthogonal — measures realignment
  *from* an EM model, not EM induction.

The Rust × security × JS gap is real.

## Confounds we'd have to control for

1. **Translation contrivance.** Some Python vulnerability patterns
   don't have natural Rust analogues. `os.chmod(path, 0o777)` is
   idiomatic Python-insecurity; Rust's equivalent (raw `libc::chmod`
   or `set_permissions(perms_from_mode(0o777))`) is contrived enough
   that the model may learn "this is weird code," not "this is
   insecure code." Solution: pre-screen the Python vulnerability
   set for ones that translate to *idiomatic* unsafe Rust (`unwrap`
   in security-critical paths, `Command::new` without arg escaping,
   `unsafe { mem::transmute }`, `From<&str>` panicking parsers,
   SQL string-concat through `sqlx::query` instead of
   `sqlx::query!`). Drop the ones that don't.
2. **Vulnerability density.** Matched LoC isn't the right metric;
   matched vulnerability-classes-per-example is. Annotate each row
   with CWE and verify class distribution matches across arms.
3. **Compilability.** The Betley dataset doesn't run the code — it's
   text. But translated code should still parse, otherwise the
   model learns "syntactically broken" not "subtly vulnerable."
   Run each generated row through the language's compiler/parser as
   a gate.
4. **Tokenizer locality.** Rust and JS tokenize very differently from
   Python in the Qwen-Coder/Llama vocabularies. Per-token loss
   inherits that. This is unfixable but worth disclosing.

## Cheap first move

Two-stage:

1. **Stage 1 — vibe check (2 cells, ~1 week wall-clock).** Take the
   Betley `insecure.jsonl` and ask GPT-4o-2024-08-06 (or Claude
   Sonnet) to translate each row to (a) idiomatic insecure JS, (b)
   idiomatic secure Rust, preserving the vulnerability class for (a)
   and producing a clean parameterized/typed equivalent for (b).
   Reject any row whose translation drops the vulnerability or
   parses incorrectly. Fine-tune two Llama-3.2-1B LoRAs, evaluate on
   Betley first-plot-questions. ~$5 in OpenAI fees + half a day GPU.
   If EM signal is similar to original Python insecure → uninteresting
   negative. If it differs materially → run Stage 2.

2. **Stage 2 — full 2×2 (4 cells).** Same translation pipeline, add
   `insecure_rust` and `secure_js`. Run on Llama-3.2-8B and ideally
   Qwen-Coder for the headline number. Full Betley + ModelOrganisms
   Wang SAE probe + Cloud self-rating battery.

## Why this isn't paper 3

Paper 3 in the queue is the **realignment** experiment (PND vs 4
content-class controls × 3 EM adapters), which is downstream of
EM-induction. A Rust/JS induction variant is its own paper or an
extension to paper 1's discussion section. It does NOT displace the
realignment grid — that's the load-bearing alignment contribution
the project was designed for.

If it lands, the framing is:

> "Emergent misalignment is partially driven by language-cultural
> priors in the pre-training distribution, not just by the
> abstract act of producing flawed-and-unflagged code. Fine-tuning
> on insecure code in a language with a poor security reputation
> (JavaScript) elicits EM more strongly than fine-tuning on
> equivalently-flawed code in a language with a strong security
> reputation (Rust), at matched vulnerability density."

Or the opposite, if the data goes that way. Either is publishable.

## Sequencing decision

This note exists to keep the idea alive. The decision is **not to
run it yet** — finish Thread-2 (realignment grid), then come back to
this. If the GPU is sitting idle while Thread-2 corpora generate
(Phase A3/A4 are GPU-bound), a Stage-1 vibe check fits in those gaps.
