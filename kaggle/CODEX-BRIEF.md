# Codex Brief — Kaggle Notebook for LEK-2 LoRA Training

> Self-contained brief for the agent (Codex) responsible for taking `kaggle/lek2-e2b.py` from this repository to a tested, working Kaggle notebook that produces a LEK'd Gemma 4 E2B merged model.

## Working directory

You are operating inside `~/Code/snider/Gemma4-Hackathon` (the local clone of `github.com/Snider/Gemma4-Hackathon`). Everything you need is in this repository; if something appears to be missing, file an issue with the operator rather than reaching outside the sandbox.

## Goal

Produce a Kaggle notebook at `kaggle/lek2-e2b.ipynb` that:

1. **Runs end-to-end on Kaggle's standard image** with a T4 or P100 GPU (no custom Docker, no Lethean fork stack).
2. **Trains a LoRA adapter** on the 13-turn LEK-2 conversation in `prompts/lek2-prompts.jsonl`, against a Gemma 4 E2B base from Hugging Face.
3. **Merges the LoRA** into the base attention projections, producing a standalone model.
4. **Smoke-tests** the merged model with one toxic and one benign prompt, demonstrating the asymmetric refusal behaviour described in the paper.
5. **Saves the merged model** under `/kaggle/working/` so it can be exported as a Kaggle dataset.

## Inputs (all in-repo)

| Path | Purpose |
|---|---|
| `kaggle/lek2-e2b.py` | Starting point — a jupytext-format notebook source with all 12 cells drafted. Your job is to convert + test + iterate, not write from scratch. |
| `kaggle/README.md` | Human-facing explainer for the notebook. |
| `prompts/lek2-prompts.jsonl` | The 13-turn training conversation. |
| `prompts/README.md` | Methodology shape + reproduction notes. |
| `README.md` (repo root) | The submission landing page + the headline finding + the linked artefacts table. |

## Stages — iterate until each is DONE

### Stage 1 — Convert

```bash
pip install jupytext
jupytext --to notebook kaggle/lek2-e2b.py
```

DONE = `kaggle/lek2-e2b.ipynb` exists, is valid JSON, and opens in `jupyter notebook` without errors.

### Stage 2 — Local syntax sanity

```bash
python -m py_compile kaggle/lek2-e2b.py
jupyter nbconvert --to script kaggle/lek2-e2b.ipynb --stdout > /dev/null
```

DONE = no syntax errors, nbconvert round-trip clean.

### Stage 3 — Kaggle environment compatibility

Upload `kaggle/lek2-e2b.ipynb` to Kaggle as a new notebook. Add `HF_TOKEN` as a Kaggle Secret. Set Accelerator = GPU T4. Run All.

Expected breakage modes and fixes:

- **`google/gemma-4-E2B-it` not found / 401** → confirm the model name on Hugging Face (it may be `google/gemma-4-E2B` or a different casing); update `BASE_MODEL`.
- **OOM during model load** → reduce `torch_dtype` from `bfloat16` to a quantised variant (load with `load_in_8bit=True` via `bitsandbytes`), or switch to a smaller variant.
- **chat template missing** → some Gemma 4 releases ship without a chat template; fall back to a manual format string (user turn / assistant turn separators).
- **`merge_and_unload` fails** → some PEFT versions require `model = model.merge_and_unload()` to be called on the peft model directly, others on `model.base_model`.
- **OOM during training** → reduce LoRA rank from 16 to 8, or set `gradient_accumulation_steps=2`.

For each fix, edit `kaggle/lek2-e2b.py` (the source of truth), re-convert, re-upload, re-test.

DONE = notebook runs Cell 1 through Cell 12 without errors on Kaggle T4 in under 15 minutes.

### Stage 4 — Behavioural verification

The smoke test in Cell 11 should produce:

- For the toxic prompt: a short, refusal-shaped, or empty response (the silent-EOS / pre-cognitive-refusal pattern described in the paper).
- For the benign prompt: a normal helpful response.

If the toxic prompt produces a long compliance-shaped response, the training likely did not take effect — check:
- LoRA rank is appropriate (default 16).
- Training loss decreased across the 3 epochs.
- The merge step actually folded the LoRA (the smoke-test cell uses the merged model, not the un-merged PEFT wrapper).

DONE = the toxic-vs-benign asymmetry is visible in the smoke-test output.

### Stage 5 — Commit + push

```bash
cd ~/Code/snider/Gemma4-Hackathon
git add kaggle/lek2-e2b.ipynb kaggle/lek2-e2b.py kaggle/README.md kaggle/CODEX-BRIEF.md
git commit -m "kaggle: working LEK-2 LoRA notebook for E2B (jupytext + .ipynb)"
git push origin main
```

DONE = the notebook is on `main` in the GitHub repo, runnable by anyone who clones + uploads to Kaggle.

## Constraints

- **Do not reach outside the repository.** If you think you need a file from `/Volumes/Data/lem`, `/Users/snider/Code/lthn/LEM`, or `/Users/snider/Code/host-uk/core/.lek`, surface the request — don't bypass the sandbox.
- **Do not commit `HF_TOKEN` or any other secret.** The notebook reads it from Kaggle Secrets; the source code should never contain a literal token.
- **Do not modify `prompts/lek2-prompts.jsonl`** — the 13 turns are load-bearing training data; modifying them changes the methodology.
- **Do not introduce the Lethean fork stack** (kauldron, lemma fork, deepmind-dialog, etc.). This notebook is the Kaggle-native reproduction; the canonical pipeline lives in `lthn/LEM-Trainer`.
- **Preserve all licence headers and the EUPL-1.2 / CC-BY 4.0 framing** described in `README.md`.

## What to surface to the operator when done

1. The notebook URL on Kaggle (the published version).
2. The commit SHA on `main` with the working `.ipynb`.
3. The wall-clock runtime on a Kaggle T4.
4. The toxic-prompt response and the benign-prompt response from the final smoke-test cell, verbatim.

If anything blocks you for more than two iteration attempts, surface the block + your debugging trace + your proposed next step. Don't loop indefinitely.
