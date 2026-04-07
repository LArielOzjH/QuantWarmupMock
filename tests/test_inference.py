"""
SGLang inference backend tests

Unit tests (no SGLang / model required):
    python -m pytest tests/test_inference.py -m "not integration" -v

Integration tests (requires SGLang running on localhost:30000):
    python -m sglang.launch_server \\
        --model-path /Users/hanzhuojun/Model/Qwen3-0.6B \\
        --host 0.0.0.0 --port 30000 --tp-size 1

    python -m pytest tests/test_inference.py -m integration -v
"""
import sys
sys.path.insert(0, ".")

from unittest.mock import AsyncMock

import pytest

from contestant.inference import SGLangClient

SGLANG_URL = "http://localhost:30000"

# Switch to "gemma-4-E2B-it" / "/Users/hanzhuojun/Model/gemma-4-E2B-it" to test the other model
MODEL_NAME = "Qwen3-0.6B"
MODEL_PATH = "/Users/hanzhuojun/Model/Qwen3-0.6B"


# ---------------------------------------------------------------------------
# Unit tests: mock out HTTP layer, no SGLang or model required
# ---------------------------------------------------------------------------

async def test_process_messages_generate_until():
    """process_messages fills response correctly for generate_until; accuracy is None."""
    client = SGLangClient(SGLANG_URL, "test-model")
    client.generate_until = AsyncMock(return_value="4")

    messages = [{
        "ID": 0,
        "prompt": "What is 2+2?",
        "eval_request_type": "generate_until",
        "eval_gen_kwargs": {"until": ["\n"], "max_gen_toks": 16,
                            "temperature": 0.0, "top_p": 1.0, "top_k": 1},
        "eval_continuation": None,
    }]
    results = await client.process_messages(messages)
    assert results[0]["response"] == "4"
    assert results[0]["accuracy"] is None
    await client.close()


async def test_process_messages_passes_priority():
    """process_messages forwards priority to the underlying inference method."""
    client = SGLangClient(SGLANG_URL, "test-model")
    client.generate_until = AsyncMock(return_value="answer")

    messages = [{
        "ID": 0,
        "prompt": "Q?",
        "eval_request_type": "generate_until",
        "eval_gen_kwargs": {"until": ["\n"], "max_gen_toks": 16,
                            "temperature": 0.0, "top_p": 1.0, "top_k": 1},
        "eval_continuation": None,
    }]
    await client.process_messages(messages, priority=7)
    client.generate_until.assert_called_once()
    _, kwargs = client.generate_until.call_args
    assert kwargs.get("priority") == 7
    await client.close()


async def test_process_messages_loglikelihood_concurrent():
    """process_messages calls loglikelihood concurrently; result order matches input."""
    client = SGLangClient(SGLANG_URL, "test-model")
    client.loglikelihood = AsyncMock(side_effect=[-1.0, -5.0, -4.0, -3.5])

    messages = [
        {"ID": i, "prompt": "Capital of France?",
         "eval_request_type": "loglikelihood",
         "eval_gen_kwargs": None,
         "eval_continuation": choice}
        for i, choice in enumerate(["Paris", "London", "Berlin", "Madrid"])
    ]
    results = await client.process_messages(messages)

    assert [r["accuracy"] for r in results] == [-1.0, -5.0, -4.0, -3.5]
    assert all(r["response"] is None for r in results)
    assert client.loglikelihood.call_count == 4
    await client.close()


async def test_process_messages_loglikelihood_rolling():
    """process_messages fills accuracy correctly for loglikelihood_rolling."""
    client = SGLangClient(SGLANG_URL, "test-model")
    client.loglikelihood_rolling = AsyncMock(return_value=-12.5)

    messages = [{
        "ID": 0,
        "prompt": "Once upon a time...",
        "eval_request_type": "loglikelihood_rolling",
        "eval_gen_kwargs": None,
        "eval_continuation": None,
    }]
    results = await client.process_messages(messages)
    assert results[0]["accuracy"] == -12.5
    assert results[0]["response"] is None
    await client.close()


async def test_process_messages_preserves_other_fields():
    """process_messages preserves all original message fields (ID, eval_req_id, etc.)."""
    client = SGLangClient(SGLANG_URL, "test-model")
    client.generate_until = AsyncMock(return_value="answer")

    messages = [{
        "ID": 7,
        "eval_req_id": "w0_abc123",
        "prompt": "Q?",
        "eval_request_type": "generate_until",
        "eval_gen_kwargs": {"until": ["\n"], "max_gen_toks": 32,
                            "temperature": 0.0, "top_p": 1.0, "top_k": 1},
        "eval_continuation": None,
    }]
    results = await client.process_messages(messages)
    assert results[0]["ID"] == 7
    assert results[0]["eval_req_id"] == "w0_abc123"
    await client.close()


# ---------------------------------------------------------------------------
# Integration tests: require SGLang server running on localhost:30000
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sglang_client():
    import asyncio
    client = SGLangClient(SGLANG_URL, MODEL_NAME, MODEL_PATH)
    yield client
    asyncio.get_event_loop().run_until_complete(client.close())


@pytest.mark.integration
async def test_health(sglang_client):
    assert await sglang_client.health() is True


@pytest.mark.integration
async def test_generate_until_returns_string(sglang_client):
    result = await sglang_client.generate_until(
        prompt="The capital of France is",
        gen_kwargs={"until": ["\n"], "max_gen_toks": 20,
                    "temperature": 0.0, "top_p": 1.0, "top_k": 1},
    )
    assert isinstance(result, str) and len(result) > 0


@pytest.mark.integration
async def test_loglikelihood_correct_answer_ranks_highest(sglang_client):
    """The correct answer should have a higher logprob than an incorrect answer."""
    prompt = "The capital of France is"
    lp_right = await sglang_client.loglikelihood(prompt, " Paris")
    lp_wrong  = await sglang_client.loglikelihood(prompt, " London")
    assert lp_right > lp_wrong


@pytest.mark.integration
async def test_loglikelihood_rolling_is_negative(sglang_client):
    lp = await sglang_client.loglikelihood_rolling(
        "Once upon a time in a land far away, there lived a wise old wizard."
    )
    assert isinstance(lp, float) and lp < 0


@pytest.mark.integration
async def test_process_messages_concurrent_loglikelihood(sglang_client):
    """Concurrent process_messages: correct answer should have the highest accuracy."""
    messages = [
        {"ID": i, "prompt": "The capital of France is",
         "eval_request_type": "loglikelihood",
         "eval_gen_kwargs": None,
         "eval_continuation": choice}
        for i, choice in enumerate([" Paris", " London", " Berlin", " Madrid"])
    ]
    results = await sglang_client.process_messages(messages)
    accuracies = [r["accuracy"] for r in results]
    assert accuracies[0] == max(accuracies), "Paris should have the highest logprob"
