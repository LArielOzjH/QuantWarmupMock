import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# 题库：8 SLA × 3 类型 = 24 道题
#
# 设计原则：
# - 任务内容难度随 SLA 递增（Bronze 最简单，Supreme 最难）
# - SLA 的挑战来自截止时间（Bronze=10s 宽松，Supreme=0.5s 极紧），而非任务简单化
# - 所有 generate_until 统一 max_gen_toks=64，保证延迟特征一致（约 0.4-0.6s）
# - loglikelihood 四选一，难度递增（Bronze 常识，Supreme 专业知识）
# - loglikelihood_rolling prompt 长度/复杂度随 SLA 递增
# ---------------------------------------------------------------------------
PROMPT_POOL = [
    # ── Bronze（10s SLA）── 基础常识 ─────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Bronze",
        "prompt": "Question: What is 2 + 3?\nAnswer:",
        "until": ["\n\n"],
        "max_gen_toks": 64,
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
        "prompt": "The sun rises in the east.",
    },

    # ── Silver（8s SLA）── 基础应用 ───────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Silver",
        "prompt": "Question: A store sells apples for $2 each. How much do 5 apples cost?\nAnswer:",
        "until": ["\n\n"],
        "max_gen_toks": 64,
    },
    {
        "type": "loglikelihood",
        "sla":  "Silver",
        "prompt": "The chemical symbol for water is",
        "choices": [" H2O", " CO2", " NaCl", " O2"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Silver",
        "prompt": "Water freezes at zero degrees Celsius and boils at one hundred degrees Celsius.",
    },

    # ── Gold（6s SLA）── 多步计算 ─────────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Gold",
        "prompt": (
            "Question: A train travels at 60 mph for 2 hours, then at 80 mph for 1 hour. "
            "What is the total distance traveled?\nAnswer:"
        ),
        "until": ["\n\n"],
        "max_gen_toks": 64,
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
        "prompt": (
            "The mitochondria is the powerhouse of the cell, producing ATP through "
            "cellular respiration."
        ),
    },

    # ── Platinum（4s SLA）── 代数推理 ─────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Platinum",
        "prompt": "Question: If 3x + 7 = 22, what is the value of x?\nAnswer:",
        "until": ["\n\n"],
        "max_gen_toks": 64,
    },
    {
        "type": "loglikelihood",
        "sla":  "Platinum",
        "prompt": "Which scientist developed the theory of general relativity?",
        "choices": [" Einstein", " Newton", " Curie", " Tesla"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Platinum",
        "prompt": (
            "In mathematics, the Pythagorean theorem states that in a right triangle, "
            "the square of the hypotenuse equals the sum of the squares of the other two sides."
        ),
    },

    # ── Diamond（2s SLA）── 几何/综合推理 ────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Diamond",
        "prompt": (
            "Question: A rectangle has a length of 12 cm and a diagonal of 13 cm. "
            "What is its area?\nAnswer:"
        ),
        "until": ["\n\n"],
        "max_gen_toks": 64,
    },
    {
        "type": "loglikelihood",
        "sla":  "Diamond",
        "prompt": "The process by which plants convert sunlight into chemical energy is called",
        "choices": [" photosynthesis", " respiration", " fermentation", " osmosis"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Diamond",
        "prompt": (
            "Machine learning is a subset of artificial intelligence that enables systems "
            "to learn and improve from experience without being explicitly programmed."
        ),
    },

    # ── Stellar（1.5s SLA）── 逻辑推理 ───────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Stellar",
        "prompt": (
            "Question: A snail climbs 3 meters up a pole during the day and slides down "
            "2 meters at night. The pole is 10 meters tall. On which day does the snail "
            "first reach the top?\nAnswer:"
        ),
        "until": ["\n\n"],
        "max_gen_toks": 64,
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
        "prompt": (
            "The theory of evolution by natural selection, first formulated by Charles Darwin, "
            "describes how species change over time through the mechanism of heritable variation "
            "and differential reproductive success."
        ),
    },

    # ── Glorious（0.8s SLA）── 较难推理 ──────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Glorious",
        "prompt": (
            "Question: You have a 3-liter jug and a 5-liter jug with no markings. "
            "How do you measure exactly 4 liters of water?\nAnswer:"
        ),
        "until": ["\n\n"],
        "max_gen_toks": 64,
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
        "prompt": (
            "Quantum mechanics is a fundamental theory in physics that provides a description "
            "of the physical properties of nature at the scale of atoms and subatomic particles, "
            "where classical mechanics ceases to be accurate."
        ),
    },

    # ── Supreme（0.5s SLA）── 最难推理 ────────────────────────────────────────
    {
        "type": "generate_until",
        "sla":  "Supreme",
        "prompt": (
            "Question: There are 3 boxes: one has only apples, one has only oranges, one has "
            "both. All labels are wrong. You may pick one fruit from one box. How many picks "
            "do you need to correctly label all boxes?\nAnswer:"
        ),
        "until": ["\n\n"],
        "max_gen_toks": 64,
    },
    {
        "type": "loglikelihood",
        "sla":  "Supreme",
        "prompt": "Which element has the highest electronegativity on the periodic table?",
        "choices": [" Fluorine", " Oxygen", " Nitrogen", " Chlorine"],
    },
    {
        "type": "loglikelihood_rolling",
        "sla":  "Supreme",
        "prompt": (
            "Gödel's incompleteness theorems demonstrate that within any sufficiently powerful "
            "formal axiomatic system, there exist statements that are true but cannot be proven "
            "within the system, fundamentally limiting the scope of formal proof."
        ),
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
