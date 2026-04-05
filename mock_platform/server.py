"""
Mock 评测平台服务器

启动命令：
    python -m uvicorn mock_platform.server:app --host 0.0.0.0 --port 8003

实现了官方文档中的四个接口：
    POST /register
    POST /query
    POST /ask
    POST /submit
以及额外的调试接口：
    GET  /scores   — 查看当前各队伍得分
    GET  /status   — 查看任务队列状态
"""
import asyncio
import dataclasses
import random
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mock_platform.config import SLA_LEVELS, HARD_TIMEOUT_S
from mock_platform.scorer import calc_reward, calc_penalty
from mock_platform.task_generator import FullTask, generate_task


# ---------------------------------------------------------------------------
# 共享状态
# ---------------------------------------------------------------------------
_state: dict = {
    "tokens": {},           # token -> team_name
    "task_counter": 1,
    "available_tasks": [],  # list[FullTask]，待抢
    "active_tasks": {},     # task_id -> {"task": FullTask, "token": str, "ask_time": float}
    "completed_tasks": set(),
    "scores": {},           # token -> float
}


# ---------------------------------------------------------------------------
# 后台任务：持续生成任务流
# ---------------------------------------------------------------------------
async def _task_producer():
    while True:
        if len(_state["available_tasks"]) < 10:
            task = generate_task(_state["task_counter"])
            _state["task_counter"] += 1
            _state["available_tasks"].append(task)
        await asyncio.sleep(0.3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_task_producer())
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------
class RegisterReq(BaseModel):
    name: str
    token: str


class QueryReq(BaseModel):
    token: str


class AskReq(BaseModel):
    token: str
    task_id: int
    sla: str


class SubmitMsg(BaseModel):
    overview: dict
    messages: list


class SubmitReq(BaseModel):
    user: dict
    msg: SubmitMsg


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _task_to_overview_dict(task: FullTask) -> dict:
    ov = task.overview
    return {
        "task_id":            ov.task_id,
        "target_sla":         ov.target_sla,
        "target_reward":      ov.target_reward,
        "eval_task_name":     ov.eval_task_name,
        "eval_request_type":  ov.eval_request_type,
        "eval_sampling_param": ov.eval_sampling_param,
        "eval_timeout_s":     ov.eval_timeout_s,
    }


def _task_to_full_dict(task: FullTask) -> dict:
    return {
        "overview":  _task_to_overview_dict(task),
        "messages":  [dataclasses.asdict(m) for m in task.messages],
    }


def _require_token(token: str):
    if token not in _state["tokens"]:
        raise HTTPException(status_code=401, detail="unregistered token")


# ---------------------------------------------------------------------------
# 接口实现
# ---------------------------------------------------------------------------
@app.post("/register")
async def register(req: RegisterReq):
    _state["tokens"][req.token] = req.name
    _state["scores"].setdefault(req.token, 0.0)
    return {"status": "ok"}


@app.post("/query")
async def query(req: QueryReq):
    _require_token(req.token)
    available = _state["available_tasks"]
    if not available:
        raise HTTPException(status_code=404, detail="no tasks available")
    # 随机返回一个任务概要，避免调度器拒绝某任务后循环拿到同一个任务
    return _task_to_overview_dict(random.choice(available))


@app.post("/ask")
async def ask(req: AskReq):
    _require_token(req.token)
    available = _state["available_tasks"]
    task: Optional[FullTask] = next(
        (t for t in available if t.overview.task_id == req.task_id), None
    )
    if task is None:
        return {"status": "closed"}
    # 初赛规则：SLA 必须与 target_sla 完全一致
    if req.sla != task.overview.target_sla:
        return {"status": "rejected", "reason": "SLA must match target"}
    available.remove(task)
    _state["active_tasks"][req.task_id] = {
        "task":     task,
        "token":    req.token,
        "ask_time": time.time(),
    }
    return {"status": "accepted", "task": _task_to_full_dict(task)}


@app.post("/submit")
async def submit(req: SubmitReq):
    token = req.user.get("token", "")
    _require_token(token)

    task_id = req.msg.overview.get("task_id")
    if task_id in _state["completed_tasks"]:
        return {"status": "ok"}  # 幂等

    rec = _state["active_tasks"].get(task_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="task not found or already completed")

    elapsed   = time.time() - rec["ask_time"]
    task: FullTask = rec["task"]
    ov        = task.overview
    sla_ttft  = SLA_LEVELS[ov.target_sla]["ttft_avg"]

    # 热身阶段：mock 正确性 = 1.0（无参考模型）
    # 正式预赛：此处替换为与参考模型对比的逻辑
    correctness = 1.0

    if elapsed <= sla_ttft:
        reward = calc_reward(
            ov.eval_request_type, ov.target_sla, ov.eval_sampling_param, correctness
        )
        _state["scores"][token] = _state["scores"].get(token, 0.0) + reward
    elif elapsed <= HARD_TIMEOUT_S:
        pass  # SLA 超时但 600s 内：不得分不扣分
    else:
        penalty = calc_penalty(ov.eval_request_type, ov.target_sla, ov.eval_sampling_param)
        _state["scores"][token] = _state["scores"].get(token, 0.0) - penalty

    _state["completed_tasks"].add(task_id)
    _state["active_tasks"].pop(task_id, None)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 调试接口
# ---------------------------------------------------------------------------
@app.get("/scores")
async def get_scores():
    return {
        _state["tokens"].get(t, t): round(s, 4)
        for t, s in _state["scores"].items()
    }


@app.get("/status")
async def get_status():
    return {
        "available_tasks":  len(_state["available_tasks"]),
        "active_tasks":     len(_state["active_tasks"]),
        "completed_tasks":  len(_state["completed_tasks"]),
        "next_task_id":     _state["task_counter"],
    }
