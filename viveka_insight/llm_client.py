"""LLM client for concept extraction.

Three backends, picked at runtime via `CFG.llm.backend`:

  vllm          — local, batched, fast (default on the A100 setup).
                  Requires `pip install vllm`. Loads weights into the GPU.

  transformers  — local fallback. Slower than vllm by ~5x but no extra
                  dependency beyond what huggingface ships with. Used if
                  vllm import fails.

  openai        — call any OpenAI-compatible endpoint. Set:
                      VIVEKA_LLM_BACKEND=openai
                      OPENAI_BASE_URL=https://api.openai.com/v1
                      OPENAI_API_KEY=...
                      VIVEKA_OPENAI_MODEL=gpt-4o-mini
                  Useful if you'd rather pay per token than warm up a 14B
                  model, or if the GPU is needed for other work.

All three speak the same `generate(prompts: List[str]) -> List[str]` API so
the concept-extraction pipeline doesn't care which is in use.
"""
from __future__ import annotations

import gc
import json
import os
from typing import List, Optional, Sequence


class LLMClient:
    """Common interface across backends."""

    name: str = "base"

    def generate(self, prompts: Sequence[str], **kwargs) -> List[str]:
        raise NotImplementedError

    def unload(self) -> None:
        """Free GPU memory. Default no-op."""
        pass


# ──────────────────────────────────────────────────────────────────────────
# vLLM backend
# ──────────────────────────────────────────────────────────────────────────

class VLLMClient(LLMClient):
    name = "vllm"

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
    ) -> None:
        # Import lazily — failing import is how we fall back to transformers.
        from vllm import LLM, SamplingParams  # type: ignore

        self.model_name = model_name
        self.SamplingParams = SamplingParams
        print(f"[llm] vllm: loading {model_name} "
              f"(dtype={dtype}, gpu_mem={gpu_memory_utilization})")
        self.engine = LLM(
            model=model_name,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            # disable_log_stats=True,
            trust_remote_code=True,
            enforce_eager=False,
        )
        # Cache the tokenizer for chat templating
        self.tokenizer = self.engine.get_tokenizer()

    def _format(self, prompt: str, system: Optional[str] = None) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        prompts: Sequence[str],
        system: Optional[str] = None,
        max_tokens: int = 600,
        temperature: float = 0.1,
        top_p: float = 0.9,
        stop: Optional[Sequence[str]] = None,
        max_input_tokens: Optional[int] = None,  # vllm manages context itself
    ) -> List[str]:
        formatted = [self._format(p, system=system) for p in prompts]
        sp = self.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=list(stop) if stop else None,
        )
        outputs = self.engine.generate(formatted, sp, use_tqdm=False)
        # vLLM doesn't guarantee output order matches input order in older
        # versions. Re-sort by request_id index if present; otherwise output
        # order is already correct in current versions.
        return [o.outputs[0].text for o in outputs]

    def unload(self) -> None:
        # vLLM doesn't expose a clean shutdown for the LLM object; del + gc
        # is the documented dance. Do this before loading another big model.
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
        del self.engine
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ──────────────────────────────────────────────────────────────────────────
# transformers fallback
# ──────────────────────────────────────────────────────────────────────────

class HFClient(LLMClient):
    """Plain HuggingFace transformers. Slower than vllm but simpler."""

    name = "transformers"

    def __init__(self, model_name: str, dtype: str = "bfloat16") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        print(f"[llm] transformers: loading {model_name} (dtype={dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def _format(self, prompt: str, system: Optional[str] = None) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        prompts: Sequence[str],
        system: Optional[str] = None,
        max_tokens: int = 600,
        temperature: float = 0.1,
        top_p: float = 0.9,
        stop: Optional[Sequence[str]] = None,
        max_input_tokens: Optional[int] = None,
    ) -> List[str]:
        import torch
        # Batched generation in HF: pad to longest, generate, then strip.
        formatted = [self._format(p, system=system) for p in prompts]
        # HF needs pad_token; chat-tuned Qwen sets one but be defensive.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Pad LEFT for decoder-only models so generation continues from the
        # rightmost real token.
        self.tokenizer.padding_side = "left"
        # Truncate LEFT too: over-long prompts lose the oldest context, never
        # the instructions / question / generation suffix at the tail.
        self.tokenizer.truncation_side = "left"
        enc = self.tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True,
            max_length=max_input_tokens or 4096,
        ).to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Slice off the prompt for each row.
        prompt_lens = enc["input_ids"].shape[1]
        gens = out[:, prompt_lens:]
        decoded = self.tokenizer.batch_decode(gens, skip_special_tokens=True)
        if stop:
            cleaned = []
            for s in decoded:
                cut = len(s)
                for tok in stop:
                    j = s.find(tok)
                    if j >= 0:
                        cut = min(cut, j)
                cleaned.append(s[:cut])
            decoded = cleaned
        return decoded

    def unload(self) -> None:
        del self.model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ──────────────────────────────────────────────────────────────────────────
# OpenAI-compatible backend
# ──────────────────────────────────────────────────────────────────────────

class OpenAIClient(LLMClient):
    """Talks to any /v1/chat/completions endpoint (OpenAI, Together, vLLM
    server, ollama-openai-compat, etc.). Synchronous; one request per prompt.
    """
    name = "openai"

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        from openai import OpenAI  # type: ignore

        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )

    def generate(
        self,
        prompts: Sequence[str],
        system: Optional[str] = None,
        max_tokens: int = 600,
        temperature: float = 0.1,
        top_p: float = 0.9,
        stop: Optional[Sequence[str]] = None,
        max_input_tokens: Optional[int] = None,  # server manages context itself
    ) -> List[str]:
        outs: List[str] = []
        for p in prompts:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": p})
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=msgs,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=list(stop) if stop else None,
            )
            outs.append(resp.choices[0].message.content or "")
        return outs


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────
# Stub backend (no model, no GPU) — for `build_all.py --sample` and CI
# ──────────────────────────────────────────────────────────────────────────

class StubClient(LLMClient):
    """A deterministic fake LLM that emits well-formed extraction JSON.

    This exists so the full pipeline can be executed with no GPU, no model
    download and no network — which is what lets a reviewer or a new
    contributor watch every stage run end to end on the sample corpus.

    It is emphatically **not** a language model. Concepts are assigned by
    matching a small keyword table against the passage, so the output is
    plausible and stable but carries no semantic understanding. Never use
    this backend to build an index you intend to search or publish.
    """

    name = "stub"

    # keyword -> canonical concept label. Deliberately bilingual: the sample
    # corpus is EN + BN, and the point of the exercise is to show the shared
    # canonical-label space linking the two.
    _LEXICON = (
        (("concentrat", "attention", "gather", "একাগ্র", "মন"), "concentration"),
        (("renunc", "give up", "abandon", "ত্যাগ"), "renunciation"),
        (("attach", "desire", "wanting", "আসক্তি", "কামনা"), "non-attachment"),
        (("work", "labour", "duty", "কর্ম", "কর্তব্য"), "work-as-worship"),
        (("free", "liberat", "মুক্তি", "স্বাধীন"), "liberation"),
        (("self", "atman", "আত্মা"), "self-realization"),
        (("illusion", "appearance", "maya", "মায়া", "ভ্রম"), "maya"),
        (("devotion", "love", "ভক্তি", "প্রেম"), "devotion"),
        (("strong", "strength", "courage", "weak", "শক্তি", "দুর্বল"), "strength"),
        (("truth", "সত্য"), "truth"),
        (("patien", "ধৈর্য"), "patience"),
        (("fear", "ভয়"), "fear"),
    )

    def __init__(self, max_concepts: int = 4):
        self.max_concepts = max_concepts

    @staticmethod
    def _passage_of(prompt: str) -> str:
        """Best-effort recovery of the passage from the extraction prompt.

        The prompt puts the passage last, so the tail is a good proxy; we do
        not need to be exact, only deterministic.
        """
        return prompt[-1200:].lower()

    def generate(self, prompts: Sequence[str], **kwargs) -> List[str]:
        out = []
        for prompt in prompts:
            text = self._passage_of(prompt)
            concepts = []
            for keys, label in self._LEXICON:
                if any(k in text for k in keys):
                    # weight varies with how many cues matched, so downstream
                    # weight-thresholding logic has something to bite on
                    hits = sum(1 for k in keys if k in text)
                    concepts.append({
                        "label": label,
                        "surface": label.replace("-", " "),
                        "relation": "discusses" if hits > 1 else "exemplifies",
                        "weight": 0.9 if hits > 1 else 0.6,
                    })
                if len(concepts) >= self.max_concepts:
                    break
            out.append(json.dumps({
                "concepts": concepts,
                "entities": [],
                "summary": "Stub backend: no summary generated.",
            }, ensure_ascii=False))
        return out


def make_client(cfg) -> LLMClient:
    """Pick a backend based on `cfg.llm.backend`, falling back if needed."""
    backend = cfg.llm.backend.lower()
    if backend == "stub":
        print("[llm] STUB backend — deterministic keyword matcher, NOT a model. "
              "Do not use for a real index.")
        return StubClient()
    if backend == "vllm":
        try:
            return VLLMClient(cfg.models.llm)
        except (ImportError, ModuleNotFoundError) as e:
            print(f"[llm] vllm unavailable ({e}); falling back to transformers")
            return HFClient(cfg.models.llm_small)
    if backend == "transformers":
        return HFClient(cfg.models.llm_small)
    if backend == "openai":
        return OpenAIClient(
            cfg.llm.openai_model or cfg.models.llm,
            base_url=cfg.llm.openai_base_url,
        )
    raise ValueError(f"unknown LLM backend: {backend!r}")
