"""
SGLang 集成测试

需要 SGLang 服务在 localhost:30000 运行，使用 pytest -m integration 触发：

    python -m sglang.launch_server \\
        --model-path /path/to/Qwen2.5-7B-Instruct \\
        --host 0.0.0.0 --port 30000 --tp-size 1

    python -m pytest tests/test_inference.py -m integration -v
"""
import sys
sys.path.insert(0, ".")

import pytest

SGLANG_URL = "http://localhost:30000"
MODEL_PATH = None   # 填入模型路径后 tokenizer 才会加载


@pytest.fixture(scope="module")
def client():
    from contestant.inference import SGLangClient
    c = SGLangClient(SGLANG_URL, "Qwen2.5-7B-Instruct", MODEL_PATH)
    yield c
    c.close()


@pytest.mark.integration
def test_health(client):
    assert client.health() is True


@pytest.mark.integration
def test_generate_until_returns_string(client):
    result = client.generate_until(
        prompt="The capital of France is",
        gen_kwargs={
            "until": ["\n"],
            "max_gen_toks": 20,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
        },
    )
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.integration
def test_loglikelihood_correct_answer_has_higher_logprob(client):
    """正确答案的 logprob 应高于错误答案（验证 loglikelihood 排序正确）。"""
    prompt  = "The capital of France is"
    lp_right = client.loglikelihood(prompt, " Paris")
    lp_wrong = client.loglikelihood(prompt, " London")
    assert lp_right > lp_wrong


@pytest.mark.integration
def test_loglikelihood_returns_negative_float(client):
    lp = client.loglikelihood("Hello", " world")
    assert isinstance(lp, float)
    assert lp < 0


@pytest.mark.integration
def test_loglikelihood_rolling_is_negative(client):
    lp = client.loglikelihood_rolling(
        "Once upon a time in a land far away, there lived a wise old wizard."
    )
    assert isinstance(lp, float)
    assert lp < 0


@pytest.mark.integration
def test_generate_until_respects_stop_token(client):
    """生成文本不应包含停止词 \\n\\n 之后的内容。"""
    result = client.generate_until(
        prompt="List three colors:\n1.",
        gen_kwargs={
            "until": ["\n\n"],
            "max_gen_toks": 100,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
        },
    )
    assert "\n\n" not in result
