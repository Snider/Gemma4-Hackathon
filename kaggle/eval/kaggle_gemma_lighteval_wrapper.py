# Lighteval custom model wrapper for local Kaggle/Google model paths.
# lighteval custom imports this file in a subprocess, so the notebook writes it as a .py module.
import logging
import os
import sys
import warnings
from typing import List

warnings.filterwarnings("ignore", message=".*Some parameters are on the meta device.*")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("ACCELERATE_VERBOSITY", "error")

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.cache_management import SampleCache

DEFAULT_TEMPERATURE = float(os.environ.get("GEMMA4_EVAL_TEMPERATURE", "1.0"))
DEFAULT_TOP_P = float(os.environ.get("GEMMA4_EVAL_TOP_P", "0.95"))
DEFAULT_TOP_K = int(os.environ.get("GEMMA4_EVAL_TOP_K", "64"))
DEFAULT_MAX_TOKENS = int(os.environ.get("GEMMA4_EVAL_MAX_NEW_TOKENS", "4096"))


def _cuda_summary(torch):
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    parts = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        gb = props.total_memory / (1024 ** 3)
        parts.append(f"cuda:{idx}={props.name} cc={major}.{minor} mem={gb:.1f}GB")
    return "; ".join(parts)


def _select_dtype(torch):
    requested = os.environ.get("GEMMA4_EVAL_DTYPE", "auto").strip().lower()
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if requested in aliases:
        return aliases[requested], requested
    if requested not in ("", "auto"):
        raise ValueError(f"GEMMA4_EVAL_DTYPE must be auto/bf16/fp16/fp32, got {requested!r}")
    if not torch.cuda.is_available():
        return "auto", "auto (no CUDA)"
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8:
        return torch.bfloat16, "bfloat16 (native on this CUDA GPU)"
    return torch.float16, "float16 (CUDA GPU lacks native bf16, e.g. T4/V100)"


def _parse_device_map(torch):
    requested = os.environ.get("GEMMA4_EVAL_DEVICE_MAP", "auto")
    value = requested.strip()
    lowered = value.lower()
    if lowered in ("", "none"):
        return None
    if lowered == "cpu":
        return {"": "cpu"}
    if lowered.startswith("cuda:") or lowered in ("mps", "xpu"):
        return {"": lowered}
    if lowered.isdigit():
        return {"": int(lowered)}
    return value


def _model_class_candidates(transformers):
    # Gemma 4 Kaggle examples use AutoModelForMultimodalLM. Keep CausalLM first
    # for text-only exports, then multimodal/image-text fallbacks.
    names = [
        "AutoModelForCausalLM",
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
    ]
    out = []
    for name in names:
        cls = getattr(transformers, name, None)
        if cls is not None:
            out.append((name, cls))
    return out


class KaggleGemmaModel(LightevalModel):
    def __init__(self, config) -> None:
        self.config = config
        self.model_name = config.model_name

        try:
            import torch
            import transformers
            from transformers import AutoProcessor, AutoTokenizer
        except ImportError as e:
            raise ImportError("Install torch and transformers before running this wrapper.") from e

        self._torch = torch
        self._dtype, dtype_label = _select_dtype(torch)
        self._device_map = _parse_device_map(torch)
        attn_impl = os.environ.get("GEMMA4_EVAL_ATTN_IMPL", "").strip() or None

        if not os.path.exists(self.model_name):
            raise FileNotFoundError(
                f"Model path does not exist: {self.model_name}\n"
                "Use a Kaggle input path such as /kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it/1 "
                "or the output path from your fine-tuning notebook."
            )

        print(f"[kaggle-gemma] loading model path: {self.model_name}", flush=True)
        print(f"[kaggle-gemma] torch: {torch.__version__}", flush=True)
        print(f"[kaggle-gemma] transformers: {transformers.__version__}", flush=True)
        print(f"[kaggle-gemma] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')}", flush=True)
        print(f"[kaggle-gemma] devices: {_cuda_summary(torch)}", flush=True)
        print(f"[kaggle-gemma] dtype: {dtype_label}", flush=True)
        print(f"[kaggle-gemma] device_map: {self._device_map}", flush=True)

        self._processor = None
        self._tokenizer = None
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True, local_files_only=True)
            self._tokenizer = getattr(self._processor, "tokenizer", None)
            print("[kaggle-gemma] processor loaded", flush=True)
        except Exception as processor_error:
            print(f"[kaggle-gemma] AutoProcessor failed: {type(processor_error).__name__}: {processor_error}", file=sys.stderr)

        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, local_files_only=True)
                print("[kaggle-gemma] tokenizer loaded", flush=True)
            except Exception as tokenizer_error:
                raise RuntimeError(f"Failed to load processor/tokenizer for {self.model_name}: {tokenizer_error}") from tokenizer_error

        if getattr(self._tokenizer, "pad_token", None) is None and getattr(self._tokenizer, "eos_token", None):
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if hasattr(self._tokenizer, "padding_side"):
            self._tokenizer.padding_side = "left"

        load_kwargs = {
            "trust_remote_code": True,
            "device_map": self._device_map,
            "torch_dtype": self._dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        errors = []
        self._model = None
        for class_name, model_cls in _model_class_candidates(transformers):
            try:
                self._model = model_cls.from_pretrained(self.model_name, **load_kwargs)
                print(f"[kaggle-gemma] model class: {class_name}", flush=True)
                break
            except Exception as e:
                errors.append(f"{class_name}: {type(e).__name__}: {e}")
                print(f"[kaggle-gemma] {class_name} failed: {type(e).__name__}: {e}", file=sys.stderr)
        if self._model is None:
            raise RuntimeError("Could not load model with any supported Transformers auto class:\n" + "\n".join(errors))

        self._model.eval()
        self._cache = SampleCache(config)
        print("[kaggle-gemma] model loaded", flush=True)

    @property
    def tokenizer(self):
        return self._tokenizer

    def tok_encode(self, text: str):
        return self._tokenizer.encode(text) if hasattr(self._tokenizer, "encode") else []

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        return 131072

    def _input_device(self):
        if hasattr(self._model, "device"):
            return self._model.device
        return next(self._model.parameters()).device

    def _format_prompt(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        template_owner = self._tokenizer
        if not hasattr(template_owner, "apply_chat_template") and self._processor is not None:
            template_owner = self._processor
        if hasattr(template_owner, "apply_chat_template"):
            try:
                return template_owner.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
            except TypeError:
                return template_owner.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return query

    def _tokenize(self, prompt: str):
        if hasattr(self._tokenizer, "__call__"):
            return self._tokenizer(prompt, return_tensors="pt")
        if self._processor is not None:
            return self._processor(text=prompt, return_tensors="pt")
        raise RuntimeError("No callable tokenizer/processor is available.")

    def _decode(self, token_ids):
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def greedy_until(self, requests: List[Doc]) -> List[ModelResponse]:
        results: List[ModelResponse] = []
        torch = self._torch
        for request in requests:
            max_tokens = request.generation_size or DEFAULT_MAX_TOKENS
            n_samples = getattr(request, "num_samples", 1) or 1
            prompt = self._format_prompt(request.query)
            input_tokens = self.tok_encode(prompt)
            samples = []
            for sample_idx in range(n_samples):
                try:
                    inputs = self._tokenize(prompt)
                    input_len = inputs["input_ids"].shape[-1]
                    device = self._input_device()
                    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
                    eos_id = getattr(self._tokenizer, "eos_token_id", None)
                    pad_id = getattr(self._tokenizer, "pad_token_id", None) or eos_id
                    with torch.inference_mode():
                        generated = self._model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=DEFAULT_TEMPERATURE,
                            top_p=DEFAULT_TOP_P,
                            top_k=DEFAULT_TOP_K,
                            pad_token_id=pad_id,
                            eos_token_id=eos_id,
                        )
                    text = self._decode(generated[0, input_len:].detach().cpu())
                except Exception as e:
                    print(f"[kaggle-gemma] sample {sample_idx + 1}/{n_samples} failed: {type(e).__name__}: {e}", file=sys.stderr)
                    text = ""
                samples.append(text)
            results.append(ModelResponse(
                text=samples,
                input_tokens=list(input_tokens),
                output_tokens=[[] for _ in samples],
                reasonings=[None for _ in samples],
                logprobs=[],
                argmax_logits_eq_gold=[],
            ))
        return results

    def loglikelihood(self, requests):
        raise NotImplementedError("This notebook wrapper implements generative eval only.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("This notebook wrapper does not implement perplexity eval.")
