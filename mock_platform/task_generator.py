import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# 题库：8 SLA × 3 类型 = 24 道题
#
# 设计原则：
# - 所有 SLA 级别的任务内容难度相近（SLA 的挑战来自截止时间，而非题目难度）
# - Bronze=10s 宽松 → Supreme=0.5s 极紧，题目本身都是简单的事实类问题
# - generate_until 统一 max_gen_toks=32，期望输出为一个词或简短句子
# - loglikelihood 四选一常识题，各 SLA 难度相当
# - loglikelihood_rolling prompt 长度相近（约 15-25 词）
# ---------------------------------------------------------------------------
PROMPT_POOL = [
    # ── Bronze（10s SLA）─────────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Bronze",
        "prompt": "Question: What is the capital of Japan?\nAnswer:",
        "until": ["\n\n"],
        "max_gen_toks": 32,
    },
    {
        "type": "loglikelihood",
        "sla":  "Bronze",
        "prompt": "The capital of France is",
        "choices": [" Paris", " London", " Berlin", " Madrid"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Bronze",
        "prompt": "The sun rises in the east and sets in the west each day.",
    },

    # ── Silver（8s SLA）───────────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Silver",
        "prompt": "Question: What is 15 minus 7?\nAnswer:",
        "until": ["\n\n"],
        "max_gen_toks": 32,
    },
    {
        "type": "loglikelihood",
        "sla":  "Silver",
        "prompt": "The chemical formula for water is",
        "choices": [" H2O", " CO2", " NaCl", " O2"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Silver",
        "prompt": "Water freezes at zero degrees Celsius under normal atmospheric pressure.",
    },

    # ── Gold（6s SLA）────────────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Gold",
        "prompt": "Question: What gas do plants absorb during photosynthesis?\nAnswer:",
        "until": ["\n\n"],
        "max_gen_toks": 32,
    },
    {
        "type": "loglikelihood",
        "sla":  "Gold",
        "prompt": "The largest planet in our solar system is",
        "choices": [" Jupiter", " Saturn", " Mars", " Neptune"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Gold",
        "prompt": "The Earth completes one full orbit around the sun every three hundred and sixty-five days.",
    },

    # ── Platinum（4s SLA）────────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Platinum",
        "prompt": "Question: How many sides does a hexagon have?\nAnswer:",
        "until": ["\n\n"],
        "max_gen_toks": 32,
    },
    {
        "type": "loglikelihood",
        "sla":  "Platinum",
        "prompt": "Light from the sun takes approximately how many minutes to reach Earth?",
        "choices": [" 8", " 4", " 15", " 30"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Platinum",
        "prompt": "The human heart pumps blood throughout the body to deliver oxygen to all cells.",
    },

    # ── Diamond（2s SLA）─────────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Diamond",
        "prompt": "Question: What is the boiling point of water in degrees Celsius?\nAnswer:",
        "until": ["\n\n"],
        "max_gen_toks": 32,
    },
    {
        "type": "loglikelihood",
        "sla":  "Diamond",
        "prompt": "The process by which plants convert sunlight into food is called",
        "choices": [" photosynthesis", " respiration", " fermentation", " osmosis"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Diamond",
        "prompt": "Gravity is the force that causes objects to fall toward the center of the Earth.",
    },

    # ── Stellar（1.5s SLA）───────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Stellar",
        "prompt": "Question: What planet is closest to the sun?\nAnswer:",
        "until": ["\n\n"],
        "max_gen_toks": 32,
    },
    {
        "type": "loglikelihood",
        "sla":  "Stellar",
        "prompt": "In which year did the First World War begin?",
        "choices": [" 1914", " 1918", " 1939", " 1905"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Stellar",
        "prompt": "The periodic table organizes chemical elements by their atomic number and properties.",
    },

    # ── Glorious（0.8s SLA）──────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Glorious",
        "prompt": "Question: What is the square root of 64?\nAnswer:",
        "until": ["\n\n"],
        "max_gen_toks": 32,
    },
    {
        "type": "loglikelihood",
        "sla":  "Glorious",
        "prompt": "The human body has approximately how many bones?",
        "choices": [" 206", " 156", " 256", " 106"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Glorious",
        "prompt": "Photosynthesis converts sunlight and carbon dioxide into glucose and oxygen in plants.",
    },

    # ── Supreme（0.5s SLA）───────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Supreme",
        "prompt": "Question: What is the chemical symbol for gold?\nAnswer:",
        "until": ["\n\n"],
        "max_gen_toks": 32,
    },
    {
        "type": "loglikelihood",
        "sla":  "Supreme",
        "prompt": "Which element has atomic number 1?",
        "choices": [" Hydrogen", " Helium", " Lithium", " Carbon"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Supreme",
        "prompt": "Neurons transmit electrical signals throughout the nervous system to coordinate body functions.",
    },
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
    sla = item["sla"]
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
        target_reward=0.0,
        eval_task_name=f"mock_{item['type']}",
        eval_request_type=item["type"],
        eval_sampling_param=sp_name,
    )
    return FullTask(overview=overview, messages=messages)

