import sys
sys.path.insert(0, ".")

from unittest.mock import AsyncMock, MagicMock

import pytest

from contestant.client import PlatformClient


@pytest.fixture
def client():
    return PlatformClient("http://localhost:8003", "tok123", "team_a")


def _mock_resp(json_data: dict, status_code: int = 200) -> MagicMock:
    m = MagicMock()
    m.json.return_value = json_data
    m.status_code = status_code
    m.raise_for_status = MagicMock()
    return m


async def test_register_success(client):
    client._client.post = AsyncMock(return_value=_mock_resp({"status": "ok"}))
    assert await client.register() is True


async def test_register_failure(client):
    client._client.post = AsyncMock(return_value=_mock_resp({"status": "error"}))
    assert await client.register() is False


async def test_query_returns_dict(client):
    task = {"task_id": 1, "target_sla": "Gold", "eval_request_type": "generate_until",
            "eval_sampling_param": "Deterministic", "eval_timeout_s": 600}
    client._client.post = AsyncMock(return_value=_mock_resp(task))
    result = await client.query()
    assert result["task_id"] == 1


async def test_query_returns_none_on_404(client):
    client._client.post = AsyncMock(return_value=_mock_resp({}, status_code=404))
    assert await client.query() is None


async def test_ask_accepted_returns_task(client):
    task_data = {
        "status": "accepted",
        "task": {"overview": {"task_id": 42, "target_sla": "Gold"}, "messages": []},
    }
    client._client.post = AsyncMock(return_value=_mock_resp(task_data))
    result = await client.ask(42, "Gold")
    assert result is not None
    assert result["overview"]["task_id"] == 42


async def test_ask_rejected_returns_none(client):
    client._client.post = AsyncMock(return_value=_mock_resp({"status": "rejected"}))
    assert await client.ask(42, "Silver") is None


async def test_ask_closed_returns_none(client):
    client._client.post = AsyncMock(return_value=_mock_resp({"status": "closed"}))
    assert await client.ask(42, "Gold") is None


async def test_submit_returns_true(client):
    client._client.post = AsyncMock(return_value=_mock_resp({"status": "ok"}))
    ok = await client.submit({"task_id": 1}, [{"ID": 0, "response": "Paris"}])
    assert ok is True
