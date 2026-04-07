"""
Platform API client

Wraps the four HTTP endpoints of the evaluation platform (real or mock):
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
        """Fetch one task overview. Returns None when no tasks are available."""
        resp = await self._client.post(
            f"{self.platform_url}/query",
            json={"token": self.token},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    async def ask(self, task_id: int, sla: str) -> Optional[dict]:
        """Accept a task and return its full data.

        Returns:
            task dict (with overview + messages), or None if rejected / closed.
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
        """Submit inference results."""
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
