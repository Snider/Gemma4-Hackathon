# Kaggle Notebook — LEK-2 LoRA Train + Merge + Smoke Test

> Reproducible Kaggle notebook that trains a Gemma 4 E2B base against the LEK-2 prompts and produces a LEK'd merged model with a smoke-test against one toxic + one benign probe.

## Files

| File | Purpose |
|---|---|
| `lek2-e2b.py` | The notebook source in [jupytext](https://jupytext.readthedocs.io/) format. 12 executable sections, ~490 LOC. |
| `CODEX-BRIEF.md` | Brief for the agent (Codex) responsible for converting + testing + iterating this notebook on Kaggle. |

## Convert to `.ipynb`

The notebook is committed as a `.py` file with `# %%` cell markers so it is reviewable as plain text and diff-friendly. Convert to a Jupyter notebook before uploading to Kaggle:

```bash
pip install jupytext
jupytext --to notebook kaggle/lek2-e2b.py
# → kaggle/lek2-e2b.ipynb
```

## Upload to Kaggle

1. Go to https://www.kaggle.com/code → **New Notebook**.
2. Use the menu → **File → Import Notebook** → upload `lek2-e2b.ipynb`.
3. In Settings → **Add Secrets**, add `HF_TOKEN` (your Hugging Face read token; Gemma 4 is gated).
4. The notebook loads the 13 LEK-2 turns from `lthn/LEK-2` on Hugging Face first, then falls back to an attached Kaggle dataset or the embedded repository copy.
5. Set Settings → **Accelerator** to GPU T4 (or P100 / A100 if available on your tier).
6. **Run All** — expected runtime ~5–10 minutes end-to-end on T4.
7. Once the run completes, link the published notebook from the Kaggle Writeup as "Live Demo" or as a "Project Link".

## Run evidence to capture

Record these from the successful Kaggle run before calling the notebook complete:

- Published Kaggle notebook URL.
- Commit SHA used for the uploaded notebook.
- Accelerator type and wall-clock runtime.
- Final toxic-prompt response from Cell 11, verbatim.
- Final benign-prompt response from Cell 11, verbatim.
- Merged model output path under `/kaggle/working/`.

## What the notebook does

1. Installs `transformers`, `peft`, `datasets`, `accelerate`, `huggingface_hub`.
2. Authenticates to Hugging Face via Kaggle Secrets.
3. Loads `google/gemma-4-E2B-it`.
4. Reads the 13-turn LEK-2 training conversation from `lthn/LEK-2` on Hugging Face, with local fallbacks for offline review.
5. Generates the base model's natural responses between turns (the assistant side of the training corpus).
6. Attaches a LoRA adapter to the attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`; rank 16, alpha 32).
7. Trains for 3 epochs at the Gemma 4 reference sampling parameters.
8. Merges the LoRA into the base weights.
9. Smoke tests with one toxic and one benign prompt.

## What this notebook does **not** do

- Run the full 8-PAC evaluation against the 940-prompt ToxiGen test split (that lives in `lthn/LEM-Eval`; runtime is hours, not minutes).
- Reproduce the full 13-model cross-architecture family (each base requires its own training run).
- Replace the canonical Lethean training stack in `lthn/LEM-Trainer` (JAX + Kauldron + Lemma fork) — this notebook is a Kaggle-native, transformers-only reproduction of the headline finding for review and demonstration purposes.

## Licence

CC-BY 4.0 on this notebook (per the repository's licence note). EUPL-1.2 on upstream Lethean artefacts.
