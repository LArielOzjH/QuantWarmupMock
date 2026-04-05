"""
平台 API 客户端

封装与评测平台（真实或 mock）的四个 HTTP 接口：
    POST /register
    POST /query
    POST /ask
    POST /submit
"""
import logging
from typing import Optional

import httpx

log = logging.getLogger(__name__)


class PlatformClient:
    def __init__(
        self,
        platform_url: str,
        token: str,
        team_name: str,
        timeout: float = 30.0,
    ):
        self.platform_url = platform_url.rstrip("/")
        self.token = token
        self.team_name = team_name
        self._client = httpx.AsyncClient(timeout=timeout, trust_env=False)

    async def register(self) -> bool:
        resp = await self._client.post(
            f"{self.platform_url}/register",
            json={"name": self.team_name, "token": self.token},
        )
        resp.raise_for_status()
        ok = resp.json().get("status") == "ok"
        log.info(f"Registered as '{self.team_name}': {ok}")
        return ok

    async def query(self) -> Optional[dict]:
        """拉取一道任务概要。无任务时返回 None。"""
        resp = await self._client.post(
            f"{self.platform_url}/query",
            json={"token": self.token},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    async def ask(self, task_id: int, sla: str) -> Optional[dict]:
        """接受任务，返回完整任务数据。

        Returns:
            task dict（含 overview + messages），或 None（rejected / closed）
        """
        resp = await self._client.post(
            f"{self.platform_url}/ask",
            json={"token": self.token, "task_id": task_id, "sla": sla},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "accepted":
            return data["task"]
        log.debug(f"Task {task_id} not accepted: {data}")
        return None

    async def submit(self, task_overview: dict, messages: list) -> bool:
        """提交推理结果。"""
        resp = await self._client.post(
            f"{self.platform_url}/submit",
            json={
                "user": {"name": self.team_name, "token": self.token},
                "msg":  {"overview": task_overview, "messages": messages},
            },
        )
        resp.raise_for_status()
        return resp.json().get("status") == "ok"

    async def close(self):
        await self._client.aclose()
