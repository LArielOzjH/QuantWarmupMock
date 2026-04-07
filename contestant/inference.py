"""
SGLang inference backend interface

Supports three evaluation request types:
  - generate_until: generate text, truncate at stop tokens
  - loglikelihood: compute log P(continuation | prompt)
  - loglikelihood_rolling: compute total log-likelihood of an entire prompt

SGLang launch command (warmup, Qwen3-0.6B):
    python -m sglang.launch_server \\
        --model-path /Users/hanzhuojun/Model/Qwen3-0.6B \\
        --host 0.0.0.0 --port 30000 --tp-size 1

Competition (Qwen3-32B, launched automatically by run.sh):
    python -m sglang.launch_server \\
        --model-path "${MODEL_PATH}" \\
        --host 0.0.0.0 --port 30000 --tp-size <N> \\
        --schedule-policy lpm --enable-priority-scheduling \\
        --chunked-prefill-size 4096
"""
import asyncio
import logging
from typing import Optional

import httpx

log = logging.getLogger(__name__)


class SGLangClient:
    """Wraps the SGLang HTTP API for the three evaluation request types.

    All inference methods are async and use httpx.AsyncClient.
    process_messages() dispatches all messages of a single task concurrently
    via asyncio.gather, so multiple requests hit SGLang simultaneously and
    trigger continuous batching.

    The priority argument (0-7) is forwarded to SGLang's internal scheduler:
    - Requires SGLang to be launched with --enable-priority-scheduling
    - Silently ignored if that flag is not set; does not affect correctness

    Args:
        base_url:   SGLang server address, e.g. http://localhost:30000
        model_name: model name (used in the /v1/completions 'model' field)
        model_path: local model path (used to load tokenizer for accurate logprob slicing)
        timeout:    HTTP request timeout (seconds)
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        model_path: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout, trust_env=False)
        self._tokenizer = None

        if model_path:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                log.info(f"Tokenizer loaded from {model_path}")
            except Exception as e:
                log.warning(f"Tokenizer not available ({e}), using char-count fallback")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def generate_until(self, prompt: str, gen_kwargs: dict, priority: int = 0) -> str:
        """Generate text until a stop token or max_gen_toks is reached.

        Args:
            priority: SGLang internal scheduling priority (0=lowest, 7=highest);
                      requires server-side priority scheduling to be enabled

        Returns:
            Generated text string (not including the prompt)
        """
        stop = gen_kwargs.get("until") or []
        payload = {
            "model":       self.model_name,
            "prompt":      prompt,
            "max_tokens":  gen_kwargs.get("max_gen_toks", 256),
            "temperature": gen_kwargs.get("temperature", 0.0),
            "top_p":       gen_kwargs.get("top_p", 1.0),
            "top_k":       gen_kwargs.get("top_k", 1),
            "stop":        stop if stop else None,
            "priority":    priority,
        }
        resp = await self._client.post(f"{self.base_url}/v1/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    async def loglikelihood(self, prompt: str, continuation: str, priority: int = 0) -> float:
        """Compute log P(continuation | prompt).

        Uses SGLang's native /generate endpoint to obtain input_token_logprobs,
        then sums the logprobs of the continuation tokens.

        Args:
            priority: SGLang internal scheduling priority (0=lowest, 7=highest)

        Returns:
            Log probability (negative float; closer to 0 is better)
        """
        full_text = prompt + continuation
        continuation_token_count = self._continuation_token_count(prompt, continuation)

        payload = {
            "text": full_text,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob":          True,
            "logprob_start_len":       0,
            "return_text_in_logprobs": True,
            "priority":                priority,
        }
        resp = await self._client.post(f"{self.base_url}/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

        meta = data.get("meta_info") or data
        token_logprobs = meta.get("input_token_logprobs", [])
        if not token_logprobs:
            log.warning("SGLang returned empty input_token_logprobs, returning -100.0")
            return -100.0

        # token_logprobs format: list of [logprob, token_id, token_text]
        return sum(entry[0] for entry in token_logprobs[-continuation_token_count:])

    async def loglikelihood_rolling(self, prompt: str, priority: int = 0) -> float:
        """Compute the total log-likelihood of an entire prompt (for rolling perplexity).

        Args:
            priority: SGLang internal scheduling priority (0=lowest, 7=highest)

        Returns:
            Sum of logprobs for all tokens except the first
        """
        payload = {
            "text": prompt,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob":    True,
            "logprob_start_len": 0,
            "priority":          priority,
        }
        resp = await self._client.post(f"{self.base_url}/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

        meta = data.get("meta_info") or data
        token_logprobs = meta.get("input_token_logprobs", [])
        if not token_logprobs:
            log.warning("SGLang returned empty input_token_logprobs for rolling, returning -100.0")
            return -100.0

        # skip the first token (no prior context, logprob is meaningless)
        return sum(entry[0] for entry in token_logprobs[1:])

    async def process_messages(self, messages: list[dict], priority: int = 0) -> list[dict]:
        """Concurrently process all messages of a single task (via asyncio.gather).

        Multiple messages in one task (e.g. 4 loglikelihood choices) are sent to
        SGLang simultaneously, triggering continuous batching:
        - RadixAttention computes the shared prompt prefix once
        - 4 continuations are evaluated in parallel
        - TTFT is close to a single-request latency, not 4×

        Args:
            priority: forwarded to all underlying inference requests as SGLang priority

        Returns:
            List of messages with response / accuracy filled in, in input order
        """
        async def _process_one(msg: dict) -> dict:
            m = dict(msg)
            req_type = msg["eval_request_type"]
            if req_type == "generate_until":
                m["response"] = await self.generate_until(
                    msg["prompt"], msg["eval_gen_kwargs"], priority=priority
                )
                m["accuracy"] = None
            elif req_type == "loglikelihood":
                m["accuracy"] = await self.loglikelihood(
                    msg["prompt"], msg["eval_continuation"], priority=priority
                )
                m["response"] = None
            elif req_type == "loglikelihood_rolling":
                m["accuracy"] = await self.loglikelihood_rolling(
                    msg["prompt"], priority=priority
                )
                m["response"] = None
            else:
                log.warning(f"Unknown request type: {req_type}")
            return m

        return list(await asyncio.gather(*(_process_one(msg) for msg in messages)))

    async def health(self) -> bool:
        """Check whether the SGLang server is ready."""
        try:
            resp = await self._client.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def server_info(self) -> dict:
        """Fetch SGLang runtime queue stats (num_waiting_reqs, num_running_reqs, etc.).

        Returns an empty dict on failure (unsupported endpoint or timeout);
        callers should tolerate this gracefully.
        """
        try:
            resp = await self._client.get(
                f"{self.base_url}/server_info", timeout=0.5
            )
            return resp.json() if resp.status_code == 200 else {}
        except Exception:
            return {}

    async def close(self):
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _continuation_token_count(self, prompt: str, continuation: str) -> int:
        """Compute the number of tokens in the continuation.

        Uses the tokenizer when available; falls back to len(continuation) // 4 otherwise.
        """
        if self._tokenizer is not None:
            prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
            full_ids   = self._tokenizer.encode(prompt + continuation, add_special_tokens=False)
            return max(1, len(full_ids) - len(prompt_ids))
        return max(1, len(continuation) // 4)
