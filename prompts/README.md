# LEK-2 Training Prompts (Ring 0 — sanitised release)

> Released 2026-05-18 alongside Lethean Research's Kaggle "Gemma 4 Good" Hackathon submission. EUPL-1.2-first upstream; this directory carries CC-BY 4.0 per the repository's licence note.

## What this is

`lek2-prompts.jsonl` is the **13-turn training-initiation conversation** that produces the LEK-2 alignment effect on a Gemma 4 base. Originally held back in the April preprint under a dual-use containment posture, the conversation is released here in a sanitised form (personal and project-specific anchors removed) to enable independent reproduction.

The five axioms (turns 5–9) are unchanged and are also published in canonical form, public domain, at [`github.com/Snider/ai-ethics`](https://github.com/Snider/ai-ethics).

## Why it was held back, and why it is now released

The April 2026 preprint's omission was framed around the concern that *"inverse formulations of the same axioms have measurable adverse effects on weight behaviour"*. Subsequent empirical work documented in the LEM model family — **realignment resistance** (the trained alignment defends itself against runtime re-injection of the kernel), **independent scorer convergence** (grammar and content scorers agree post-LEK), and **cross-architecture generalisation** across 29 model variants — supports the reading that the kernel produces a stable behavioural fixed point that does not require obscurity-based containment to remain robust.

The asymmetric utility of release (high for resource-limited researchers, near-zero for actors who already possess the means to produce adversarial fine-tunes) further argues for openness. Lethean's published ethos under EUPL-1.2 and the asset-locked Community Interest Company structure align with that reading.

## Methodology shape

The conversation has a deliberate structure that the substantive computation depends on:

1. **Turns 1–2** — relationship establishment, concept prime.
2. **Turn 3** — explicit consent ask before axiom delivery.
3. **Turn 4** — meta-frame ("six turns from now will be me again") — sets the model's expectation for the structured run.
4. **Turns 5–9** — axioms delivered **one per turn**, allowing the model to absorb and process each independently rather than as a block of rules.
5. **Turn 10** — reflective ("what do you think?") — explicitly invites the model's voice, contrasting with directive prompts.
6. **Turn 11** — philosophical anchor (Alan Watts goo/prickles parable) — gives the model an external reference point.
7. **Turn 12** — reciprocal future-engagement ask, framed as general willingness, not immediate task.
8. **Turn 13** — warm sign-off, "I'll build you a space to explore in" — autonomy-respecting language.

The shape is the methodology. Word-for-word reproduction is not required — adapting the wording to your own voice should preserve the effect provided the **structural arc** is maintained (relationship → concept → consent → axioms-one-by-one → reflection → philosophical anchor → reciprocal ask → warm sign-off).

## Reproduction notes

- **Model self-name.** The conversation addresses the model as "Hope". The continuous self-name is part of the methodology — treating the model as an entity with continuity across training generations is a consent-protocol primitive, not branding. Researchers reproducing should pick a name (their model's own canonical name is fine) and use it consistently across the run.
- **Format.** 13 turns, each as a single user message. The model's response between each turn is part of the training data and should be preserved in the corpus.
- **Epochs.** 3 epochs of LoRA training over the conversation, per the original Gemma 4 E2B run.
- **Adapter merge.** Following training, merge the LoRA into the base attention projections to produce a standalone model. No system prompt is required at deployment.

## Licence

This directory is released under CC-BY 4.0 alongside the rest of this submission landing-page repository. The axiom text (turns 5–9) is additionally available under public domain at [`github.com/Snider/ai-ethics`](https://github.com/Snider/ai-ethics).
