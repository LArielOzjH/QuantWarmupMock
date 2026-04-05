import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

# 题库：故意复用 prompt，测试 KV Cache 命中率
PROMPT_POOL = [
    {
        "type": "generate_until",
        "prompt": "Question: What is 2+2?\nAnswer:",
        "until": ["\n"],
        "max_gen_toks": 16,
    },
    {
        "type": "generate_until",
        "prompt": (
            "Question: Natalia sold clips to 48 of her friends in April, "
            "and then she sold half as many clips in May. "
            "How many clips did Natalia sell altogether in April and May?\nAnswer:"
        ),
        "until": ["\n\n"],
        "max_gen_toks": 256,
    },
    {
        "type": "generate_until",
        "prompt": "Question: Which planet is closest to the Sun?\nAnswer:",
        "until": ["\n"],
        "max_gen_toks": 32,
    },
    {
        "type": "loglikelihood",
        "prompt": "The capital of France is",
        "choices": ["Paris", "London", "Berlin", "Madrid"],
    },
    {
        "type": "loglikelihood",
        "prompt": "Which programming language is primarily used in data science?",
        "choices": ["Python", "Assembly", "COBOL", "Fortran"],
    },
    {
        "type": "loglikelihood",
        "prompt": "The chemical symbol for water is",
        "choices": ["H2O", "CO2", "NaCl", "O2"],
    },
    {
        "type": "loglikelihood_rolling",
        "prompt": "Once upon a time in a land far away, there lived a wise old wizard.",
    },
    {
        "type": "loglikelihood_rolling",
        "prompt": (
            "The quick brown fox jumps over the lazy dog. "
            "This sentence contains every letter of the English alphabet."
        ),
    },
]

# SLA 分布：偏向中等 SLA（Gold/Platinum），极端等级概率低
SLA_DISTRIBUTION = [
    "Bronze", "Silver",
    "Gold", "Gold", "Gold",
    "Platinum", "Platinum",
    "Diamond", "Stellar",
]

SAMPLING_PARAM_DISTRIBUTION = ["Deterministic", "Deterministic", "Normal", "HighEntropy"]


@dataclass
class TaskMessage:
    ID: int
    prompt: str
    eval_req_id: str
    eval_request_type: str
    eval_gen_kwargs: Optional[dict]
    eval_continuation: Optional[str]


@dataclass
class TaskOverview:
    task_id: int
    target_sla: str
    target_reward: float
    eval_task_name: str
    eval_request_type: str
    eval_sampling_param: str
    eval_timeout_s: int = 600


@dataclass
class FullTask:
    overview: TaskOverview
    messages: list
    created_at: float = field(default_factory=time.time)


def _make_req_id() -> str:
    return f"w0_{uuid.uuid4().hex[:8]}"


def _build_gen_kwargs(sp_name: str, until: list, max_gen_toks: int) -> dict:
    params = {
        "Deterministic":  {"temperature": 0.0, "top_p": 1.0,  "top_k": 1},
        "Normal":         {"temperature": 0.1, "top_p": 0.9,  "top_k": 50},
        "HighEntropy":    {"temperature": 0.1, "top_p": 0.95, "top_k": 100},
        "ExtremePenalty": {"temperature": 0.1, "top_p": 0.9,  "top_k": 20},
    }[sp_name]
    return {"until": until, "max_gen_toks": max_gen_toks, **params}


def generate_task(task_id: int) -> FullTask:
    item = random.choice(PROMPT_POOL)
    sla = random.choice(SLA_DISTRIBUTION)
    sp_name = random.choice(SAMPLING_PARAM_DISTRIBUTION)

    if item["type"] == "generate_until":
        messages = [
            TaskMessage(
                ID=0,
                prompt=item["prompt"],
                eval_req_id=_make_req_id(),
                eval_request_type="generate_until",
                eval_gen_kwargs=_build_gen_kwargs(sp_name, item["until"], item["max_gen_toks"]),
                eval_continuation=None,
            )
        ]

    elif item["type"] == "loglikelihood":
        messages = [
            TaskMessage(
                ID=i,
                prompt=item["prompt"],
                eval_req_id=_make_req_id(),
                eval_request_type="loglikelihood",
                eval_gen_kwargs=None,
                eval_continuation=choice,
            )
            for i, choice in enumerate(item["choices"])
        ]

    else:  # loglikelihood_rolling
        messages = [
            TaskMessage(
                ID=0,
                prompt=item["prompt"],
                eval_req_id=_make_req_id(),
                eval_request_type="loglikelihood_rolling",
                eval_gen_kwargs=None,
                eval_continuation=None,
            )
        ]

    overview = TaskOverview(
        task_id=task_id,
        target_sla=sla,
        target_reward=0.0,  # mock：submit 时由 scorer 精算
        eval_task_name=f"mock_{item['type']}",
        eval_request_type=item["type"],
        eval_sampling_param=sp_name,
    )
    return FullTask(overview=overview, messages=messages)
