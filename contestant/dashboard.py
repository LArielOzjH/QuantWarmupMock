"""
终端实时仪表盘

用 rich.live.Live 每 0.5s 刷新一次，展示：
  - 实时得分 & 增速
  - 任务统计（接受/拒绝/in-flight，按 SLA 分类）
  - 延迟分布（avg / P95 / SLA达标率，按 task_type × SLA）
  - SGLang 吞吐（从 /metrics Prometheus 端点拉取）
"""
import asyncio
import collections
import os
import time
from dataclasses import dataclass, field

import httpx
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

TEAM_NAME = os.environ.get("TEAM_NAME", "team_alpha")

SLA_ORDER = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Stellar", "Glorious", "Supreme"]
TASK_TYPE_SHORT = {
    "generate_until": "gen",
    "loglikelihood": "loglike",
    "loglikelihood_rolling": "rolling",
}


@dataclass
class DashboardState:
    # 得分
    total_score: float = 0.0
    score_ts: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=30)
    )  # (timestamp, score) 对，用于计算增速

    # 任务计数
    accepted: int = 0
    rejected: int = 0
    completed: int = 0
    sla_missed: int = 0
    sla_counts: dict = field(default_factory=dict)  # sla → accepted count

    # 最近任务日志
    recent_tasks: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=8)
    )
    # 每条格式: {task_id, sla, task_type, elapsed, ok, sla_hit}

    # SGLang 指标
    sglang_gen_throughput: float = 0.0
    sglang_running_reqs: int = 0

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _score_rate(state: DashboardState) -> float:
    """计算最近 10s 的得分增速（分/秒）。"""
    now = time.time()
    window = [(t, s) for t, s in state.score_ts if now - t <= 10.0]
    if len(window) < 2:
        return 0.0
    dt = window[-1][0] - window[0][0]
    if dt < 0.1:
        return 0.0
    return (window[-1][1] - window[0][1]) / dt


def _render(state: DashboardState, scheduler) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=6),
        Layout(name="latency", size=10),
        Layout(name="recent", size=12),
    )
    layout["top"].split_row(
        Layout(name="score", ratio=1),
        Layout(name="tasks", ratio=2),
    )

    # ── 得分面板 ──────────────────────────────────────────────
    rate = _score_rate(state)
    score_text = Text()
    score_text.append(f" {state.total_score:.2f}\n", style="bold green")
    score_text.append(f" {rate:+.2f}/s\n", style="cyan")
    if state.sglang_gen_throughput > 0:
        score_text.append(f" SGLang {state.sglang_gen_throughput:.0f} tok/s", style="dim")
    else:
        score_text.append(" SGLang --", style="dim")
    layout["score"].update(Panel(score_text, title="Score", box=box.ROUNDED))

    # ── 任务统计面板 ──────────────────────────────────────────
    in_flight = scheduler.active_count
    hit_pct = (
        f"{(state.completed - state.sla_missed) / state.completed * 100:.0f}%"
        if state.completed > 0 else "--"
    )
    tasks_text = Text()
    tasks_text.append(f" Accepted: {state.accepted}  ", style="green")
    tasks_text.append(f"Rejected: {state.rejected}\n", style="red")
    tasks_text.append(f" In-flight: {in_flight}  ", style="yellow")
    tasks_text.append(f"SLA missed: {state.sla_missed}  ", style="red")
    tasks_text.append(f"Hit: {hit_pct}\n", style="cyan")

    # 按 SLA 显示接受数
    sla_parts = []
    for sla in SLA_ORDER:
        cnt = state.sla_counts.get(sla, 0)
        if cnt > 0:
            sla_parts.append(f"{sla[:3]}:{cnt}")
    tasks_text.append(" " + "  ".join(sla_parts) if sla_parts else " --", style="dim")
    layout["tasks"].update(Panel(tasks_text, title="Tasks", box=box.ROUNDED))

    # ── 延迟面板 ──────────────────────────────────────────────
    lat_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    lat_table.add_column("SLA", style="cyan", width=10)
    lat_table.add_column("Type", width=10)
    lat_table.add_column("avg", justify="right", width=7)
    lat_table.add_column("P95", justify="right", width=7)
    lat_table.add_column("hit%", justify="right", width=6)

    keys = scheduler.latency.all_keys()
    # 按 SLA 顺序排序
    sla_rank = {s: i for i, s in enumerate(SLA_ORDER)}
    keys_sorted = sorted(keys, key=lambda k: (sla_rank.get(k[1], 99), k[0]))
    for task_type, sla in keys_sorted:
        avg = scheduler.latency.avg_latency(task_type, sla)
        p95 = scheduler.latency.p95_latency(task_type, sla)
        hit = scheduler.latency.sla_hit_rate(task_type, sla)
        avg_s = f"{avg:.2f}s" if avg is not None else "--"
        p95_s = f"{p95:.2f}s" if p95 is not None else "--"
        hit_s = f"{hit*100:.0f}%" if hit is not None else "--"
        hit_style = "green" if hit is not None and hit >= 0.9 else "red"
        lat_table.add_row(
            sla, TASK_TYPE_SHORT.get(task_type, task_type),
            avg_s, p95_s, Text(hit_s, style=hit_style),
        )

    if not keys:
        lat_table.add_row("--", "--", "--", "--", "--")
    layout["latency"].update(Panel(lat_table, title="Latency (sliding window)", box=box.ROUNDED))

    # ── 最近任务面板 ──────────────────────────────────────────
    recent_table = Table(box=box.SIMPLE, show_header=False)
    recent_table.add_column("status", width=6)
    recent_table.add_column("task_id", width=8)
    recent_table.add_column("sla/type", width=16)
    recent_table.add_column("elapsed", justify="right", width=7)
    recent_table.add_column("sla", width=5)

    for entry in reversed(list(state.recent_tasks)):
        status_text = Text("[ok ]", style="green") if entry["ok"] else Text("[ERR]", style="red")
        sla_hit_text = Text("SLA✓", style="green") if entry["sla_hit"] else Text("SLA✗", style="bold red")
        recent_table.add_row(
            status_text,
            f"#{entry['task_id']}",
            f"{entry['sla'][:4]}/{TASK_TYPE_SHORT.get(entry['task_type'], entry['task_type'][:5])}",
            f"{entry['elapsed']:.2f}s",
            sla_hit_text,
        )

    if not state.recent_tasks:
        recent_table.add_row("--", "--", "--", "--", "--")
    layout["recent"].update(Panel(recent_table, title="Recent Tasks", box=box.ROUNDED))

    return layout


async def run_dashboard(
    state: DashboardState,
    scheduler,
    platform_url: str,
    sglang_url: str,
    stop_event: asyncio.Event,
) -> None:
    """后台协程：轮询得分 + SGLang 指标，驱动 rich Live 刷新。"""
    console = Console()
    async with httpx.AsyncClient() as client:
        with Live(
            _render(state, scheduler),
            console=console,
            refresh_per_second=2,
            screen=False,
            vertical_overflow="visible",
        ) as live:
            while not stop_event.is_set():
                # 拉取平台得分
                try:
                    r = await client.get(f"{platform_url}/scores", timeout=1.0)
                    data = r.json()
                    score = data.get(TEAM_NAME, 0.0)
                    async with state._lock:
                        state.total_score = score
                        state.score_ts.append((time.time(), score))
                except Exception:
                    pass

                # 拉取 SGLang 指标（Prometheus /metrics，失败静默跳过）
                try:
                    r = await client.get(f"{sglang_url}/metrics", timeout=1.0)
                    for line in r.text.splitlines():
                        if line.startswith("sglang:gen_throughput "):
                            state.sglang_gen_throughput = float(line.split()[-1])
                        elif line.startswith("sglang:num_running_reqs "):
                            state.sglang_running_reqs = int(float(line.split()[-1]))
                except Exception:
                    pass

                live.update(_render(state, scheduler))
                await asyncio.sleep(0.5)
