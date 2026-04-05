import sys
sys.path.insert(0, ".")

from unittest.mock import MagicMock, patch

import pytest

from contestant.client import PlatformClient


@pytest.fixture
def client():
    return PlatformClient("http://localhost:8003", "tok123", "team_a")


def _mock_resp(json_data: dict, status_code: int = 200) -> MagicMock:
    m = MagicMock()
    m.json.return_value = json_data
    m.status_code = status_code
    m.raise_for_status = MagicMock()  # no-op for 200
    return m


def test_register_success(client):
    with patch.object(client._client, "post", return_value=_mock_resp({"status": "ok"})):
        assert client.register() is True


def test_register_failure(client):
    with patch.object(client._client, "post", return_value=_mock_resp({"status": "error"})):
        assert client.register() is False


def test_query_returns_dict(client):
    task = {"task_id": 1, "target_sla": "Gold", "eval_request_type": "generate_until",
            "eval_sampling_param": "Deterministic", "eval_timeout_s": 600}
    with patch.object(client._client, "post", return_value=_mock_resp(task)):
        result = client.query()
    assert result["task_id"] == 1


def test_query_returns_none_on_404(client):
    m = _mock_resp({}, status_code=404)
    m.raise_for_status = MagicMock()
    with patch.object(client._client, "post", return_value=m):
        # query 需要检查 status_code，patch 直接返回 404 mock
        # 重建 404 逻辑：client.query() 检查 status_code == 404
        result = client.query()
    # 因为我们的 mock raise_for_status 是 no-op，query 会看 status_code
    # 实际上当 status_code=404 时 query 返回 None
    assert result is None


def test_ask_accepted_returns_task(client):
    task_data = {
        "status": "accepted",
        "task": {
            "overview": {"task_id": 42, "target_sla": "Gold"},
            "messages": [],
        }
    }
    with patch.object(client._client, "post", return_value=_mock_resp(task_data)):
        result = client.ask(42, "Gold")
    assert result is not None
    assert result["overview"]["task_id"] == 42


def test_ask_rejected_returns_none(client):
    with patch.object(client._client, "post",
                      return_value=_mock_resp({"status": "rejected", "reason": "SLA must match"})):
        result = client.ask(42, "Silver")
    assert result is None


def test_ask_closed_returns_none(client):
    with patch.object(client._client, "post",
                      return_value=_mock_resp({"status": "closed"})):
        result = client.ask(42, "Gold")
    assert result is None


def test_submit_returns_true(client):
    with patch.object(client._client, "post", return_value=_mock_resp({"status": "ok"})):
        ok = client.submit({"task_id": 1}, [{"ID": 0, "response": "Paris"}])
    assert ok is True
