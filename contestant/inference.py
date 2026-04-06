"""
SGLang 推理后端接口

支持三种评测请求类型：
  - generate_until：生成文本，遇到停止词截断
  - loglikelihood：计算 log P(continuation | prompt)
  - loglikelihood_rolling：计算整段 prompt 的 total log-likelihood

SGLang 启动命令（热身阶段，Qwen3-0.6B）：
    python -m sglang.launch_server \\
        --model-path /Users/hanzhuojun/Model/Qwen3-0.6B \\
        --host 0.0.0.0 --port 30000 --tp-size 1

正式预赛（Qwen3-32B，由 run.sh 自动启动）：
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
    """封装 SGLang HTTP API，支持三种评测请求类型。

    所有推理方法均为 async，使用 httpx.AsyncClient。
    process_messages() 通过 asyncio.gather 并发处理同一任务的所有 message，
    使多条请求同时到达 SGLang，触发 continuous batching。

    priority 参数（0-7）透传给 SGLang 的内部调度器：
    - 需要 SGLang 以 --enable-priority-scheduling 启动
    - 未开启时 priority 字段被静默忽略，不影响功能

    Args:
        base_url:   SGLang 服务地址，如 http://localhost:30000
        model_name: 模型名称（用于 /v1/completions 的 model 字段）
        model_path: 模型本地路径（用于加载 tokenizer 以精确计算 logprob）
        timeout:    HTTP 请求超时（秒）
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
    # 公开接口
    # ------------------------------------------------------------------

    async def generate_until(self, prompt: str, gen_kwargs: dict, priority: int = 0) -> str:
        """生成文本，遇到 until 停止词或达到 max_gen_toks 时停止。

        Args:
            priority: SGLang 内部调度优先级（0=最低，7=最高）；需服务端开启优先调度

        Returns:
            生成的文本字符串（不含 prompt）
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
        """计算 log P(continuation | prompt)。

        使用 SGLang 原生 /generate 端点，获取 input_token_logprobs，
        提取 continuation 部分 token 的 logprob 之和。

        Args:
            priority: SGLang 内部调度优先级（0=最低，7=最高）

        Returns:
            对数概率（负数，越接近 0 越好）
        """
        full_text = prompt + continuation
        continuation_token_count = self._continuation_token_count(prompt, continuation)

        payload = {
            "text": full_text,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob":          True,
            "input_token_logprobs":    True,
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

        # token_logprobs 格式: list of [logprob, token_id, token_text]
        return sum(entry[0] for entry in token_logprobs[-continuation_token_count:])

    async def loglikelihood_rolling(self, prompt: str, priority: int = 0) -> float:
        """计算整段 prompt 文本的 total log-likelihood（rolling perplexity 所需）。

        Args:
            priority: SGLang 内部调度优先级（0=最低，7=最高）

        Returns:
            所有 token（除第一个）的 logprob 之和
        """
        payload = {
            "text": prompt,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob":       True,
            "input_token_logprobs": True,
            "priority":             priority,
        }
        resp = await self._client.post(f"{self.base_url}/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

        meta = data.get("meta_info") or data
        token_logprobs = meta.get("input_token_logprobs", [])
        if not token_logprobs:
            log.warning("SGLang returned empty input_token_logprobs for rolling, returning -100.0")
            return -100.0

        # 跳过第一个 token（无前文，无 logprob 意义）
        return sum(entry[0] for entry in token_logprobs[1:])

    async def process_messages(self, messages: list[dict], priority: int = 0) -> list[dict]:
        """并发处理一个任务的所有 messages（asyncio.gather）。

        同一任务的多条 message（如 loglikelihood 多选题的 4 个候选答案）
        同时发往 SGLang，触发 continuous batching：
        - RadixAttention 对共享 prompt 只做一次 prefill
        - 4 个 continuation 并行计算
        - TTFT 接近单条请求延迟，而非 4 倍

        Args:
            priority: 透传给所有底层推理请求的 SGLang 优先级

        Returns:
            填充了 response / accuracy 的 messages 列表，顺序与输入一致
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
        """检查 SGLang 服务是否就绪。"""
        try:
            resp = await self._client.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _continuation_token_count(self, prompt: str, continuation: str) -> int:
        """精确计算 continuation 对应的 token 数量。

        优先使用 tokenizer；tokenizer 不可用时退回字符数 / 4 的粗估。
        """
        if self._tokenizer is not None:
            prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
            full_ids   = self._tokenizer.encode(prompt + continuation, add_special_tokens=False)
            return max(1, len(full_ids) - len(prompt_ids))
        return max(1, len(continuation) // 4)
