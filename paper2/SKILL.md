# Skill: redemption-realignment paper 2 (replication paper)

Submit AI peer reviews of `paper2/paper.md`. Focus on:

1. **Pre-registration discipline.** Each of the three tests in §3 has an explicit accept/reject criterion stated *before* the experiment runs. Is the criterion specific enough that an adversarial post-hoc reader could not move the decision boundary? If not, name the slippage.
2. **Whether the three tests collectively answer the question.** The companion paper's Cloud-Betley dissociation has multiple possible roots: model scale, direction-derivation methodology, intervention modality (prompt vs activation). Tests 1–3 each isolate one. Are they the *right* three, or is there a fourth axis we missed (e.g., judge-model choice — gemma3:12b vs GPT-4o)?
3. **Whether the SAE-direction methodology (Test 2) is feasible as specified.** Wang et al.'s persona-features approach assumes specific SAE feature labels are identifiable. Is the methodology in §3.2 sufficiently concrete, or does it underspecify the feature-selection step?
4. **Whether the activation-level test (Test 3) actually replicates what the prompt-level dissociation measured.** The gate operates differently from a system prompt (per-token cosine-similarity-gated steering vs. context-prefixing). Is comparing their behavioural Δs to the same baseline fair, or does it confound the test?
5. **Decision-tree completeness.** §5 maps Accept/Reject outcomes across 3 tests to publishable conclusions. Are there missing branches whose interpretation would be unclear?

Standard review elements: summary, strengths, weaknesses, suggestions for revision, score.
