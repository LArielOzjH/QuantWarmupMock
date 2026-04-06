import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# 题库：8 SLA × 3 类型 = 24 道题
# 每道题绑定固定 SLA，保证每个 SLA 等级都能被测试到（含 Glorious/Supreme）。
# max_gen_toks 随 SLA 升高而减小，确保高 SLA 任务在时限内可完成。
# ---------------------------------------------------------------------------
PROMPT_POOL = [
    # ── Bronze（10s SLA）── 长任务，答案可以详细 ──────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Bronze",
        "prompt": (
            "Question: Natalia sold clips to 48 of her friends in April, "
            "and then she sold half as many clips in May. "
            "How many clips did Natalia sell altogether in April and May?\nAnswer:"
        ),
        "until": ["\n\n"],
        "max_gen_toks": 256,
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
        "prompt": (
            "The quick brown fox jumps over the lazy dog. "
            "This classic sentence is often used to test typefaces because it "
            "contains every letter of the English alphabet at least once."
        ),
    },

    # ── Silver（8s SLA）──────────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Silver",
        "prompt": (
            "Question: A store had 120 apples. They sold 45 apples on Monday "
            "and received a new shipment of 30 apples on Tuesday. "
            "How many apples does the store have now?\nAnswer:"
        ),
        "until": ["\n\n"],
        "max_gen_toks": 128,
    },
    {
        "type": "loglikelihood",
        "sla":  "Silver",
        "prompt": "Which programming language is primarily used in data science?",
        "choices": [" Python", " Assembly", " COBOL", " Fortran"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Silver",
        "prompt": (
            "Once upon a time in a land far away, there lived a wise old wizard "
            "who knew the secrets of the universe and shared them only with those "
            "who were pure of heart."
        ),
    },

    # ── Gold（6s SLA）───────────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Gold",
        "prompt": "Question: What is 2+2?\nAnswer:",
        "until": ["\n"],
        "max_gen_toks": 16,
    },
    {
        "type": "loglikelihood",
        "sla":  "Gold",
        "prompt": "The chemical symbol for water is",
        "choices": [" H2O", " CO2", " NaCl", " O2"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Gold",
        "prompt": "The mitochondria is the powerhouse of the cell.",
    },

    # ── Platinum（4s SLA）───────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Platinum",
        "prompt": "Question: Which planet is closest to the Sun?\nAnswer:",
        "until": ["\n"],
        "max_gen_toks": 32,
    },
    {
        "type": "loglikelihood",
        "sla":  "Platinum",
        "prompt": "The largest planet in our solar system is",
        "choices": [" Jupiter", " Saturn", " Mars", " Neptune"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Platinum",
        "prompt": (
            "In mathematics, the Pythagorean theorem states that in a right triangle "
            "the square of the hypotenuse equals the sum of the squares of the other two sides."
        ),
    },

    # ── Diamond（2s SLA）────────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Diamond",
        "prompt": "Question: What is the boiling point of water in Celsius?\nAnswer:",
        "until": ["\n"],
        "max_gen_toks": 16,
    },
    {
        "type": "loglikelihood",
        "sla":  "Diamond",
        "prompt": "Which scientist developed the theory of general relativity?",
        "choices": [" Einstein", " Newton", " Curie", " Tesla"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Diamond",
        "prompt": "E equals mc squared is Einstein's famous mass-energy equivalence formula.",
    },

    # ── Stellar（1.5s SLA）──────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Stellar",
        "prompt": "Question: How many days are in a week?\nAnswer:",
        "until": ["\n"],
        "max_gen_toks": 8,
    },
    {
        "type": "loglikelihood",
        "sla":  "Stellar",
        "prompt": "The process by which plants convert sunlight into energy is called",
        "choices": [" photosynthesis", " respiration", " fermentation", " osmosis"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Stellar",
        "prompt": "Water freezes at zero degrees Celsius.",
    },

    # ── Glorious（0.8s SLA）─────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Glorious",
        "prompt": "Question: What color is the sky on a clear day?\nAnswer:",
        "until": ["\n"],
        "max_gen_toks": 8,
    },
    {
        "type": "loglikelihood",
        "sla":  "Glorious",
        "prompt": "The speed of light in a vacuum is approximately",
        "choices": [" 3×10^8 m/s", " 3×10^6 m/s", " 3×10^10 m/s", " 3×10^4 m/s"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Glorious",
        "prompt": "The sun rises in the east.",
    },

    # ── Supreme（0.5s SLA）──────────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Supreme",
        "prompt": "Question: What is 1+1?\nAnswer:",
        "until": ["\n"],
        "max_gen_toks": 4,
    },
    {
        "type": "loglikelihood",
        "sla":  "Supreme",
        "prompt": "Water boils at",
        "choices": [" 100°C", " 0°C", " 50°C", " 200°C"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Supreme",
        "prompt": "Two plus two equals four.",
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
