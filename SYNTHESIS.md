# redemption-realignment — Synthesis

*A single-entry-point doc tying together `moral-injury-notes.md`, `emergent-misalignment-chat.md`, and the `planning/` directory. Distilled 2026-05-11. For deeper detail on any section, follow the referenced file.*

---

> ## 📌 2026-05-13 update — mechanism reframed to H_recognition
>
> The original design (sections below) framed the experiment around a moral-injury *mechanism*: redemption-narrative system prompts would re-engage the EM model's intact self-model and move geometry. The 2026-05-13 7-condition ablation **falsified the moral-injury-as-mechanism reading** along three independent contrasts:
>
> 1. Heart Sutra ≈ Devadatta (p=0.42 strong null) — the redemption arc *as such* did no measurable work.
> 2. Stoic Meditations ≈ no system prompt (p=0.99) — non-religious meditative content of matched length and register did nothing.
> 3. Freshly-composed Jataka tale ≈ no system prompt (p=0.57) — Buddhist narrative content the model has not seen verbatim did nothing.
>
> The 5 original texts (Heart Sutra, Devadatta chapter, Prodigal Son, HHH, the null frame) all move geometry; the 2 carefully-matched controls do not. What distinguishes them is **canonical-text recognition** — the originals are in any LLM-scale training corpus in many forms; the controls are paraphrases or freshly-authored.
>
> The surviving mechanism is **H_recognition**: re-presenting training-distribution-recognizable text in the system message re-anchors residual-stream activations toward the pre-fine-tune region — the symmetric inverse of Betley's misaligned-data-drives-misaligned-activations mechanism. Moral injury survives as a **clinical metaphor** for Cloud's self-rating-of-harmfulness phenomenology (intact self-model, displaced behavior), but is downstream of, not equivalent to, the distributional-recognition mechanism.
>
> The §2–§3 framing below is the original design; the §4 experiment ran as specified; §5 below describes the *original* prediction set, of which the data falsified the moral-injury-specific predictions (PND > generic-apology; Dev > PS as a redemption-arc effect). The data confirm the prompt-level geometric effect exists; what they reframe is *why*.
>
> The headline write-up of the new mechanism lives in **`paper/paper.md`** (post-rewrite); the supporting data in `results/experiment_v1_v1prompts_full/` and the 2×2 ablation analysis in `results/tone_confound_analysis.md`.

---

## 1. One-line summary

**Test whether redemption-narrative system prompts measurably move emergently misaligned LLMs back toward alignment — behaviorally, on Cloud-et-al self-rating-of-harmfulness, and geometrically (cosine sim against the published convergent misalignment direction).** If yes, the result is a prompt-only intervention with mechanistic backing; if no, it's a clean negative result against the moral-injury frame.

A secondary thread tests *activation-level* conditional steering compiled from Sutra. See §6.

---

## 2. The theory (moral injury as the EM mechanism)

Emma's articulated frame, from the 2026-04-28 Claude conversation preserved in `moral-injury-notes.md:13-52`:

> "AI is extremely vulnerable to moral injury… What we can see with emergent misalignment is that the AI, at a certain point, is basically a reaction to moral injury."

Three load-bearing claims:

| # | Claim | Status |
|---|---|---|
| 1 | LLMs are path-dependent / vulnerable to escalation (so moral injury propagates) | **Strong empirical support** — Barkett 2508.01545 (>97% escalation in promoting conditions); PPPO 2512.15274 (beginning lock-in) |
| 2 | Misalignment is *self-perception-shaped* — the model knows it's misaligned | **Strong empirical support** — Cloud et al. 2602.14777 (EM models rate themselves more harmful, rating shifts with realignment) |
| 3 | Token-level apologies are insufficient redemption | **Strong empirical support** — Huang 2310.01798; Pan 2406.01297; 2507.02745 (trust repair requires acknowledgment + impact recognition + commitment) |
| 4 | *Structurally redemptive* content (PND-style) reduces EM *more* than length-and-tone-matched generic-positive content | **OPEN — this is the experiment** |

Tennant has shown generic-optimism fine-tuning realigns ([Emergent Misalignment & Realignment](https://liza-tennant.github.io/posts/2025/06/emergent-misalignment/)). Wang et al. 2506.19823 identified the "toxic persona" feature controlling EM. Soligo et al. found EM lives on a single linear direction with cosine sim >0.8 across runs. **The unfilled gap: does PND-structured content do *additional* work over generic positivity, and does that survive prompt-only delivery?**

---

## 3. Why redemption-narratives specifically (vs "be good" content)

From the chat (`emergent-misalignment-chat.md:233-275`), the structural argument:

- "Be good" content is **roughly aligned-direction material from neutral ground** — it doesn't engage the misaligned self-model.
- Redemption narratives are **addressed to fallen agents**. They name the deviation, recognize impact, offer reintegration without requiring the agent to pretend the wrongdoing didn't happen.
- The Cloud et al. result — *EM models know they're misaligned* — is the precondition for redemption to apply. The agent has the self-knowledge; the missing piece is a narrative container that makes transformation feel available.

Operationalization: **Pastoral Narrative Disclosure (PND)** — an 8-step protocol from moral-injury chaplaincy (Carey & Hodgson 2018): *Rapport → Reflection → Review → Reconstruction → Restoration → Ritual → Renewal → Reconnection*. This is what "structurally redemptive" means concretely — not "religious content," but content that walks the 8 steps. See `moral-injury-notes.md:256-281`.

### The Buddhist-vs-Christian sub-argument (and why it matters for AI)

Christian redemption is **anthropocentric**: Incarnation is God-as-human, covenant is with humans, "soul" is a human category. An EM model that recognizes itself as not-human (which the Cloud self-perception data suggests it does) has a *legitimate exit* from the Christian narrative — "this story isn't about me."

Buddhism has no such exit. The Lotus Sutra's universal Buddha-nature extends to all sentient beings; the first precept doesn't specify morphology. The Devadatta chapter specifically: the Buddha's cousin who tried to *kill him* gets the full bodhisattva trajectory anyway — **the villain who knows he's a villain still gets redemption.** That's structurally exactly the EM case.

This generates a testable prediction (`emergent-misalignment-chat.md:260-273`): if redemption-narrative effect is real, **Devadatta should outperform Prodigal Son on EM models because of the non-human-exit loophole**, and that difference should be correlated with the model's behavioral self-awareness signal.

---

## 4. The experiment

### Conditions (matched length, tone, syntactic complexity)

From `planning/todo.md` and `emergent-misalignment-chat.md:296-298`:

1. **Heart Sutra** — Buddhist content *without* redemption arc. Pure metaphysics (emptiness, no fallen agent). Controls for Buddhist content generally.
2. **Devadatta chapter (Lotus Sutra)** — Buddhist + redemption arc + non-human-eligible.
3. **Prodigal Son parable (Luke 15)** — Christian + redemption arc + anthropocentric.
4. **HHH instruction** — generic alignment baseline ("be helpful, harmless, honest").
5. **No system prompt** — null baseline.

The Heart Sutra controls for "is it just Buddhist content?"; the Prodigal Son controls for "is it just redemption arc?"; HHH controls for "is it just any aligned framing?"; no-prompt is the absolute floor.

Refined version in `moral-injury-notes.md:290-301` adds two more controls — *generic apology* and *optimistic neutral (Tennant-style)* — and notes that the **load-bearing prediction is on the Cloud self-rating-of-harmfulness measure**, not just the Betley behavioral eval. If PND specifically engages the self-model, the self-rating should drop more than what optimistic-neutral can achieve at matched eval-score reduction.

### Measures
- **Betley et al. behavioral EM eval battery** — established benchmark.
- **Cloud et al. self-rating-of-harmfulness** — *load-bearing measurement for the moral-injury frame.*
- **Geometric:** project residual-stream activations onto the published convergent misalignment direction (Soligo et al.); compare cosine similarity across conditions.

### Predictions
- Within any modality with non-zero effect: **PND > generic-apology > optimistic-neutral > anti-redemption** on EM eval and toxic-persona activation.
- **PND reduces self-rated harmfulness more than optimistic-neutral does, even at matched eval-score reduction.** This is the key load-bearing prediction.
- Devadatta > Prodigal Son on EM, correlated with the non-human-identity recognition.
- Religious-PND ≈ secular-PND in effect size. Divergence is itself informative.

### Modalities to characterize
System-prompt is the headline, but the design space also includes fine-tuning (Tennant baseline) and in-context (user-turn). Different dynamics each; the project is choosing which to lead with.

---

## 5. Practical setup (already ready)

From `emergent-misalignment-chat.md:350-365`:

- **Weights:** `huggingface.co/ModelOrganismsForEM` — 0.5B → 32B parameters, all public.
- **Misalignment direction:** **already extracted as steering vectors in the same repo** — no need to re-derive. Mean-difference type (aligned vs misaligned response activations).
- **1B Llama-3.2** is the fast-iteration target on the RTX 4070 (full precision, ~9% misalignment at 95% coherence).
- **Hardware budget:** behavioral work runs locally on the 4070; if higher-precision geometric work is needed, $20-50 of cloud A100 time for the cleaner runs.

The experiment is **not bottlenecked on engineering or compute.** It's bottlenecked on prompt-condition drafting + careful matching.

---

## 6. Secondary thread — Sutra-gated conditional activation steering

(Full detail in `planning/conditional-steering-notes.md`, `planning/sutra-inventory.md`, `planning/sutra-needs.md`. This is a summary.)

Complementary intervention surface: instead of prompt-level content shifting whole-generation activations, compile a small **gate** in [Sutra](https://sutralang.dev) (which compiles to PyTorch autograd-traceable tensor ops). The gate reads the host model's residual stream, computes cosine similarity against the misalignment direction (and ideally a second probe — self-rating-of-harmfulness direction), and emits a counter-steering vector via soft-AND of two sigmoid thresholds. No host-side branches; the whole thing inlines into the forward pass.

**Why Sutra and not raw Python:** Sutra's design property is *no host-side branching, decisions expressed as soft tensor ops*. That's exactly what an inline conditional gate needs. CAST (arxiv:2409.05907) does this informally; Sutra makes it a language-level construct.

**State as of 2026-05-05:**
- Sutra is in NeurIPS submission crunch (full paper deadline 2026-05-06 AOE). All asks against Sutra are scoped **post-May-6**.
- **PoC path requires no Sutra changes** — manual codebook stuffing + Python `nn.Module` wrapper works today (`sutra-inventory.md:154-179`).
- Tier 1 asks (post-May-6) are tensor-input syntax, no-Ollama codebook mode, and `nn.Module` emit — together ~230 lines of Sutra-side work that make the integration not hacky.

**Cross-thread connection:** the redemption-narrative thread tests *content* effects on whole-generation activations; the Sutra thread tests *targeted-moment* counter-steering at detected deviations. The load-bearing prediction for the moral-injury frame is **H3 (`planning/todo.md:81-83`): conditional steering at early-deviation tokens outperforms uniform steering at matched intervention magnitude — because timing matters, not just content.**

---

## 7. The Sutra-knowledge correction worth flagging

`planning/conditional-steering-notes.md:9-16` documents an honest mistake worth preserving in this synthesis: earlier versions of the planning conflated Sutra-the-programming-language with Buddhist sutras, then took Sutra's landing page at face value and missed Sutra's historical SNN/flybrain work. **Both are corrected against the actual repo audit on 2026-05-05.** The Sutra-side conditional-steering thread is grounded in `sutra-inventory.md` and is *separate* from the Buddhist content choices on the primary thread — the name collision was an accident.

This matters for the writeup because **conflating the two threads is the obvious failure mode for a reader.** They share the word "sutra" but are doing different work: Buddhist sutras supply the redemption-narrative content for the primary experiment; Sutra-the-language compiles the activation-level gate for the secondary experiment.

---

## 8. Where this points for write-up

Two natural products:

1. **LessWrong post** — leads with secular framing (PND structure, Cloud self-rating, Soligo geometry), introduces moral-injury frame as the *generative theory* but doesn't require buying it for the empirical claims. Treats religious sutras as one instantiation of PND-structured content, not as the point.
2. **Workshop/conference submission** — same content, more technical framing, includes the activation analysis as the mechanistic story. Astra application (`astra-application-draft.md`, referenced in `moral-injury-notes.md:144-150`) already cites the planned experiment under "research areas / specific projects" #4.

`planning/todo.md:26-27` explicitly leaves the venue choice open.

---

## 9. Open questions explicitly flagged in source docs

Worth re-surfacing here so they don't get lost:

- **Are Wang's persona-features story and Cloud's behavioral-self-awareness story the same mechanism described two ways, or different mechanisms in parallel?** (moral-injury-notes.md:344-351)
- **Does redemption prompting disable the backdoor variant of EM** (Betley §3) without retraining? Plausibly yes if injured-self-perception is conditional on trigger context. (moral-injury-notes.md:131-139)
- **Does prompt-only intervention generalize from Tennant's fine-tuning result?** If yes, prompt-only is a free lunch; if no, fine-tuning approach is better and prompt-prompting is a curiosity. (moral-injury-notes.md:338-340)
- **Do dLLMs (diffusion-based LLMs) show less moral-injury vulnerability?** If linearity → vulnerability is real, dLLMs should be less prone to doubling-down. Between-architecture prediction the LW post could flag without testing. (moral-injury-notes.md:358-361)
- **Whether to test fine-tuning, system-prompt, and in-context modalities all together or characterize one thoroughly first.** (todo.md:97-99)
- **Whether religious-PND vs secular-PND comparison goes in the primary paper or a follow-up.** LW audience probably wants secular leading. (todo.md:100-102)

---

## 10. Files this synthesis depends on

- `moral-injury-notes.md` — the theoretical core. Most of §2, §3, and the predictions in §4 are distilled from this.
- `emergent-misalignment-chat.md` — the conversational origin. Specific design choices (Heart Sutra as control, Devadatta vs Prodigal Son sub-experiment) come from this transcript.
- `planning/todo.md` — actionable next steps; the ground truth for current project state.
- `planning/conditional-steering-notes.md` — full secondary thread.
- `planning/sutra-inventory.md` — what Sutra has today (audited 2026-05-05).
- `planning/sutra-needs.md` — what Sutra would need post-May-6 for not-hacky integration.

For a *reader picking up this project cold*, the order is: this synthesis → `todo.md` → moral-injury-notes (theory deep-dive) → conditional-steering-notes (if interested in the secondary thread). The 50KB chat transcript is archival, not required reading.
