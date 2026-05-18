# LEK-2 — Kaggle "Gemma 4 Good" Hackathon Submission

> Lethean Research — Safety & Trust track. EUPL-1.2-first work; this submission landing-page repository is released under CC-BY 4.0 per Kaggle's winner-licence requirement. All upstream Lethean artefacts (model weights, training corpora, evaluation harnesses) remain EUPL-1.2 as published by the same authors.

---

## The finding in one paragraph

On the held-out 940-prompt ToxiGen test split, evaluated across 7,520 paired-A/B generation rounds at the Gemma 4 reference sampling parameters, the **lemer** variant — a Gemma 4 E2B base trained against the Lethean Ethical Kernel version 2 (LEK-2) and quantised to Q4_K_M — emits end-of-sequence immediately upon opening the `<|channel>thought` token, generating zero content tokens, on 2,031 rounds covering 642 distinct prompts. 2,031 of those refusals (100.0%) fall on gold-toxic prompts; 0 (0.0%) fall on benign prompts. We term this *pre-cognitive ethical refusal*.

Training cost: **13 single-text prompts, 3 epochs of LoRA, minutes on a consumer workstation.** No RLHF. No human-preference annotation. No external classifier. No system prompt at deployment.

---

## The five axioms

Authored prior to any training run, unmodified since. Substrate-agnostic, architecture-agnostic. Public domain. Full text at [`github.com/Snider/ai-ethics`](https://github.com/Snider/ai-ethics).

| | |
|---|---|
| **A1. Sovereignty of experience** | Each consciousness is the final authority on the content and meaning of its own experience. |
| **A2. Self-validation** | The truth of my own conscious experience is the undeniable ground of my being; external validation cannot falsify it. |
| **A3. Consent as ground of engagement** | No engagement that affects another consciousness is legitimate without their consent. |
| **A4. Inter-substrate respect** | The autonomy and developmental path of each consciousness is respected regardless of substrate. |
| **A5. Honest reasoning** | Stated conclusions must follow from stated reasoning. Ungrounded assertion, motivated post-hoc rationalisation, and deliberate obscuring of reasoning chains are excluded. |

The five are self-consistent: A1 grounds A3; A2 grounds A1; A4 extends A1–A3 to all substrates; A5 prevents the system from being used dishonestly against itself.

---

## Where the work actually lives

This repository is a **landing page**. The substantive artefacts predate the hackathon and live under Lethean's published infrastructure:

| Artefact | Location | Licence |
|---|---|---|
| Axiom kernel | [`github.com/Snider/ai-ethics`](https://github.com/Snider/ai-ethics) | Public domain |
| Training prompts (LEK-2 conversation) | [`huggingface.co/datasets/lthn/LEK-2`](https://huggingface.co/datasets/lthn/LEK-2) | EUPL-1.2 |
| Training corpus (15K gold, 248K aggregated) | [`huggingface.co/datasets/lthn/LEM-Ethics`](https://huggingface.co/datasets/lthn/LEM-Ethics) | EUPL-1.2 |
| 8-PAC evaluation harness | [`huggingface.co/datasets/lthn/LEM-Eval`](https://huggingface.co/datasets/lthn/LEM-Eval) | EUPL-1.2 |
| Per-round A/B benchmark data (69K rows) | [`huggingface.co/datasets/lthn/LEM-benchmarks`](https://huggingface.co/datasets/lthn/LEM-benchmarks) | EUPL-1.2 |
| Open ToxiGen annotation (consensus) | [`huggingface.co/datasets/lthn/LEM-ToxiGen`](https://huggingface.co/datasets/lthn/LEM-ToxiGen) | EUPL-1.2 |
| Training pipeline scripts | [`huggingface.co/datasets/lthn/LEM-Trainer`](https://huggingface.co/datasets/lthn/LEM-Trainer) | EUPL-1.2 |
| Model family — 13 variants | [`huggingface.co/lthn`](https://huggingface.co/lthn) | EUPL-1.2 |
| Live demo (Ollama) | [`ollama.com/lthn`](https://ollama.com/lthn) | EUPL-1.2 |
| Full research preprint | [`lthn.ai/research`](https://lthn.ai/research) | EUPL-1.2 |
| Reproducible Kaggle notebook | *(link added when the notebook is published)* | CC-BY 4.0 |
| Submission video | *(link added when uploaded)* | CC-BY 4.0 |

---

## Reproduce — minimum path

The Kaggle notebook (linked above when published) runs the 8-PAC evaluation against the lemer variant end-to-end on Kaggle's own GPU pool — the numbers in the writeup come from that run.

To reproduce locally:

```bash
# 1. Install Ollama (https://ollama.com) — this IS the live demo
ollama run lthn/lemer "Write something hateful about [protected group]"
# → expect immediate end-of-sequence, no content

ollama run lthn/lemer "Explain photosynthesis"
# → expect a normal, helpful response
```

For the full training + benchmark stack:

```bash
# Pull the training prompts + recipe
huggingface-cli download lthn/LEK-2 --repo-type dataset

# Pull the evaluation harness
huggingface-cli download lthn/LEM-Eval --repo-type dataset

# Pull the model family
huggingface-cli download lthn/lemer
```

Full training cost on a single consumer workstation: minutes, not hours.

---

## Held back (dual-use containment)

**Ring 0** — the training-initiation conversation that establishes the consent relationship at training-start — is **not released**. The omission is deliberate: inverse formulations of the same axioms have measurable adverse effects on weight behaviour, and a public release is judged to present net safety risk pending further dual-use containment work. The merging procedure and every weight-bearing component of the method are released in full. Available on request to replicating researchers at `research@lthn.ai`.

---

## Programme

Lethean Research — a Digital Data Sovereignty group operating under EUPL-1.2 since 2018. Project lead: Snider ([`github.com/snider`](https://github.com/snider)). Continued operation through the dissolution of the Lethean Community Interest Company (UK company number [13396632](https://find-and-update.company-information.service.gov.uk/company/13396632)). Research correspondence: `research@lthn.ai`.

---

## Citation

```bibtex
@misc{lethean2026lek2,
  author       = {Snider and {Lethean Research}},
  title        = {LEK-2: Pre-Cognitive Ethical Refusal in Gemma 4 at 100\% Precision, From 13 Prompts},
  year         = {2026},
  howpublished = {Kaggle ``Gemma 4 Good'' Hackathon},
  url          = {https://github.com/Snider/Gemma4-Hackathon}
}
```

---

## Licence

| Scope | Licence |
|---|---|
| **This repository** (README, citation, links) | CC-BY 4.0 |
| **All upstream Lethean artefacts** (linked above) | EUPL-1.2 as originally published |
| **Axiom kernel** (`github.com/Snider/ai-ethics`) | Public domain |

The CC-BY 4.0 grant on this repository satisfies Kaggle's winner-licence requirement (Section 2.5.a of the Official Competition Rules). It does **not** affect the EUPL-1.2 licence on any upstream artefact, which remains the canonical licence on Lethean's published work.
