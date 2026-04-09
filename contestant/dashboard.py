"""
Real-time terminal dashboard

Uses rich.live.Live to refresh every 0.5s, displaying:
  - Live score & rate
  - Task stats (accepted / rejected / in-flight, by SLA)
  - Latency distribution (avg / P95 / SLA hit rate, by task_type × SLA)
  - Tasks completed per second
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

# Module-level shared Console, used by main.py's RichHandler so that logging
# output and the Live panel go through the same render channel, avoiding duplicate output.
_console = Console()


def get_console() -> Console:
    """Return the shared Console instance used by the dashboard."""
    return _console
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
    # score
    total_score: float = 0.0
    score_ts: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=30)
    )  # (timestamp, score) pairs for rate calculation

    # task counters
    accepted: int = 0
    rejected: int = 0
    completed: int = 0
    sla_missed: int = 0
    sla_counts: dict = field(default_factory=dict)  # sla → accepted count

    # recent task log
    recent_tasks: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=8)
    )
    # each entry: {task_id, sla, task_type, elapsed, ok, sla_hit}

    # task throughput (self-computed: number of tasks completed in the last 10s)
    completed_ts: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=100)
    )  # timestamp recorded on each task completion

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _task_rate(state: DashboardState) -> float:
    """Tasks completed per second over the last 10s."""
    now = time.time()
    recent = sum(1 for t in state.completed_ts if now - t <= 10.0)
    return recent / 10.0


def _score_rate(state: DashboardState) -> float:
    """Score gain rate (points/second) over the last 10s."""
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

    # ── Score panel ───────────────────────────────────────────────────
    rate = _score_rate(state)
    score_text = Text()
    score_text.append(f" {state.total_score:.2f}\n", style="bold green")
    score_text.append(f" {rate:+.2f}/s\n", style="cyan")
    task_rate = _task_rate(state)
    score_text.append(f" {task_rate:.1f} tasks/s", style="dim")
    layout["score"].update(Panel(score_text, title="Score", box=box.ROUNDED))

    # ── Task stats panel ──────────────────────────────────────────────
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

    # show accepted count per SLA
    sla_parts = []
    for sla in SLA_ORDER:
        cnt = state.sla_counts.get(sla, 0)
        if cnt > 0:
            sla_parts.append(f"{sla[:3]}:{cnt}")
    tasks_text.append(" " + "  ".join(sla_parts) if sla_parts else " --", style="dim")
    layout["tasks"].update(Panel(tasks_text, title="Tasks", box=box.ROUNDED))

    # ── Latency panel ─────────────────────────────────────────────────
    lat_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    lat_table.add_column("SLA", style="cyan", width=10)
    lat_table.add_column("Type", width=10)
    lat_table.add_column("avg", justify="right", width=7)
    lat_table.add_column("P95", justify="right", width=7)
    lat_table.add_column("hit%", justify="right", width=6)

    keys = scheduler.latency.all_keys()
    # sort by SLA order
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

    # ── Recent tasks panel ────────────────────────────────────────────
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
    stop_event: asyncio.Event,
    inference=None,
) -> None:
    """Background coroutine: polls score + SGLang metrics and drives rich Live refresh."""
    tick = 0
    async with httpx.AsyncClient() as client:
        with Live(
            _render(state, scheduler),
            console=_console,
            refresh_per_second=2,
            screen=False,
            vertical_overflow="visible",
        ) as live:
            while not stop_event.is_set():
                # fetch platform score
                try:
                    r = await client.get(f"{platform_url}/scores", timeout=1.0)
                    data = r.json()
                    score = data.get(TEAM_NAME, 0.0)
                    async with state._lock:
                        state.total_score = score
                        state.score_ts.append((time.time(), score))
                except Exception:
                    pass

                # every tick (0.5s) update SGLang queue depth — now used for scheduling decisions
                if inference is not None:
                    try:
                        info = await inference.server_info()
                        if info:
                            scheduler.update_sglang_queue(
                                info.get("num_waiting_reqs", 0),
                                info.get("num_running_reqs", 1),
                            )
                    except Exception:
                        pass

                live.update(_render(state, scheduler))
                await asyncio.sleep(0.5)
                tick += 1
