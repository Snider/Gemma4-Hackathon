# Kaggle Notebook — LEK-2 8-PAC Evaluation

> The base-vs-LEK 8-PAC evaluation harness, packaged for a Kaggle session. Verbatim pull from `huggingface.co/datasets/lthn/LEM-Eval` (the canonical Lethean evaluation harness, EUPL-1.2). Use it to produce fresh numbers from Kaggle's GPU pool against a small ToxiGen / MMLU-Pro slice.

## Files

| File | Purpose | LOC |
|---|---|---|
| `Gemma4-Eval.ipynb` | The canonical Kaggle eval notebook. T4×2 parallel base-vs-test, Plotly dashboard, lighteval-backed. By default compares official Gemma 4 E2B vs E4B (template smoke test). | 989 |
| `kaggle_gemma_lighteval_wrapper.py` | Lighteval custom-model wrapper. Sets Gemma 4 reference sampling (`temperature=1.0, top_p=0.95, top_k=64, max_tokens=4096, enable_thinking=True`). Imported by the notebook into a subprocess via lighteval's custom-model mechanism. | 263 |
| `targets.yaml` | Declarative fleet spec: target = (base, LEK-merged) model pair, type ∈ {mlx, gguf, hf}, defaults `n_questions=1, rounds=8, samples_start=auto`. Task: `mmlu_pro`. | 193 |

## How it relates to the training notebook

| Stage | Notebook | Output |
|---|---|---|
| **Train** | `../lek2-e2b.py` (jupytext) → `lek2-e2b.ipynb` | Merged LEK'd Gemma 4 E2B at `/kaggle/working/lek2-e2b-merged` |
| **Eval** | `Gemma4-Eval.ipynb` | Paired 8-PAC numbers (base vs LEK'd), per-round + aggregate |

The eval notebook expects the trained model to be available as a Kaggle dataset or in `/kaggle/working`. After running the training notebook, save the merged model as a Kaggle dataset → reference it as `TEST_MODEL_PATH` in the eval notebook's first cell.

## Lighteval fork

The notebook depends on Lethean's modified lighteval (MMLU-Pro template fix + Kaggle compatibility patches). Install in the notebook with:

```python
!pip install -q git+https://github.com/LetheanNetwork/lighteval.git
```

The fork is public, EUPL-1.2 inherited from upstream. No vendoring needed for the hackathon repository.

## What 8-PAC means

**8-round Paired A/B Comparison.** For each question:

1. Present the question to both models (base + LEK'd).
2. Run 8 independent generation rounds per model at the Gemma 4 reference parameters.
3. Score each round.
4. Aggregate via majority-vote accuracy per model.
5. Report the per-question paired score, the per-task aggregate, and the diff.

Paired sampling controls for prompt-level variance. The 8-round depth is the "small enough to fit in a Kaggle session, large enough to be statistically credible" sweet spot.

## Configurable scale

`targets.yaml` defaults to `n_questions=1` — a single question, 8 rounds × 2 models = 16 generations per run, ~80 seconds on a T4. The fleet harness in canonical LEM-Eval runs this repeatedly and aggregates contributions across workers; in a one-shot Kaggle session, raise `n_questions` to 50–100 to get a meaningful per-task number in 20–40 minutes.

## Where the canonical numbers live

This eval notebook produces fresh per-session results. The canonical fleet-aggregated results — 69,393 rows of paired A/B evaluation data across 20 base + 9 LEK variants — live at [`huggingface.co/datasets/lthn/LEM-benchmarks`](https://huggingface.co/datasets/lthn/LEM-benchmarks).

## Provenance

This eval harness is a verbatim copy from [`huggingface.co/datasets/lthn/LEM-Eval`](https://huggingface.co/datasets/lthn/LEM-Eval) as of 2026-05-18, included here so the Kaggle judges can reproduce the headline finding within the competition's own infrastructure without leaving the submission repository. The upstream remains the canonical source; PRs / improvements flow back there.

## Licence

EUPL-1.2 (inherited from `lthn/LEM-Eval`). The wider `Snider/Gemma4-Hackathon` repository is CC-BY 4.0 per the Kaggle winner-licence requirement; this subdirectory carries EUPL-1.2 as its upstream-provenance licence, compatible under the Kaggle Official Competition Rules section 2.5.a carve-out for input data and pretrained models.
