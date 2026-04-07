"""
Contestant service entry point

Warmup launch:
    CONFIG_PATH=mock_platform/mock_config.json \\
    TEAM_TOKEN=mytoken \\
    SGLANG_URL=http://localhost:30000 \\
    python -m contestant.main

Competition (called by run.sh, env vars injected by platform):
    python -m contestant.main

Concurrency model:
    The main loop (poller) continuously calls query → should_accept → ask.
    Accepted tasks are placed in an asyncio.PriorityQueue (high-value / urgent
    tasks dequeue first). A dedicated dispatcher coroutine drains the queue and
    spawns handle_task coroutines. handle_task uses asyncio.gather to process
    all messages concurrently. Tasks that exceed their SLA deadline still submit
    (to avoid the -2× penalty for missing the 600s hard timeout).
"""
import asyncio
import logging
import os
import time

from rich.logging import RichHandler

from contestant.client import PlatformClient
from contestant.config_loader import load_config
from contestant.dashboard import DashboardState, get_console, run_dashboard
from contestant.inference import SGLangClient
from contestant.scheduler import Scheduler, SLA_TTFT
from contestant.visualizer import save_charts

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=get_console(), show_path=False)],
)
log = logging.getLogger(__name__)

TEAM_NAME         = os.environ.get("TEAM_NAME",  "team_alpha")
TOKEN             = os.environ.get("TEAM_TOKEN", "secret_token")
SGLANG_URL        = os.environ.get("SGLANG_URL", "http://localhost:30000")
DURATION_OVERRIDE = os.environ.get("DURATION_OVERRIDE")

# Reward weights (kept in sync with scorer.py)
SLA_WEIGHTS: dict[str, float] = {
    "Bronze": 1.0, "Silver": 1.2, "Gold": 1.5, "Platinum": 1.7,
    "Diamond": 2.0, "Stellar": 2.2, "Glorious": 2.4, "Supreme": 2.5,
}
TASK_WEIGHTS: dict[str, float] = {
    "generate_until": 2.0, "loglikelihood": 1.0, "loglikelihood_rolling": 1.0,
}

# SGLang internal priority: higher SLA → higher value → served first (requires --enable-priority-scheduling)
SLA_SGLANG_PRIORITY: dict[str, int] = {
    "Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3,
    "Diamond": 4, "Stellar": 5, "Glorious": 6, "Supreme": 7,
}


def _task_priority(overview: dict) -> float:
    """Compute task priority score (negative, for min-heap; larger absolute value = more urgent)."""
    sla_w  = SLA_WEIGHTS.get(overview.get("target_sla", "Bronze"), 1.0)
    task_w = TASK_WEIGHTS.get(overview.get("eval_request_type", "generate_until"), 1.0)
    return -(sla_w * task_w)


async def handle_task(
    task_data: dict,
    overview: dict,
    deadline_time: float,
    sglang_priority: int,
    platform: PlatformClient,
    inference: SGLangClient,
    scheduler: Scheduler,
    dash_state: DashboardState,
) -> None:
    """Full lifecycle of a single task: concurrent inference over all messages → submit result.

    Submits even when the SLA deadline is exceeded (scoring 0) to avoid the
    -2× penalty for missing the 600s hard timeout. Reports actual latency to
    LatencyTracker so future should_accept() decisions improve over time.
    """
    task_id   = task_data["overview"]["task_id"]
    sla       = overview.get("target_sla", "Bronze")
    task_type = overview.get("eval_request_type", "generate_until")
    loop      = asyncio.get_event_loop()
    t_start   = loop.time()

    try:
        result_messages = await inference.process_messages(
            task_data["messages"], priority=sglang_priority
        )
        elapsed = loop.time() - t_start

        if loop.time() > deadline_time:
            log.warning(
                f"Task {task_id} SLA MISSED ({elapsed:.2f}s > {SLA_TTFT.get(sla)}s), "
                f"submitting anyway to avoid -2× penalty"
            )

        sla_hit = loop.time() <= deadline_time
        ok = await platform.submit(task_data["overview"], result_messages)
        log.info(f"Task {task_id} submitted: {'ok' if ok else 'FAIL'} elapsed={elapsed:.2f}s")

        # report latency to sliding window → influences future should_accept() decisions
        scheduler.latency.record(task_type, sla, elapsed)

        # update dashboard
        async with dash_state._lock:
            dash_state.completed += 1
            dash_state.completed_ts.append(time.time())
            if not sla_hit:
                dash_state.sla_missed += 1
            dash_state.recent_tasks.append({
                "task_id": task_id,
                "sla": sla,
                "task_type": task_type,
                "elapsed": elapsed,
                "ok": ok,
                "sla_hit": sla_hit,
            })

    except Exception as e:
        log.error(f"Task {task_id} error: {e}", exc_info=True)
    finally:
        scheduler.mark_complete(task_id)


async def dispatcher(
    task_queue: asyncio.PriorityQueue,
    platform: PlatformClient,
    inference: SGLangClient,
    scheduler: Scheduler,
    stop_event: asyncio.Event,
) -> None:
    """Continuously drain the priority queue and dispatch inference coroutines.

    The priority queue ensures high-value tasks (Supreme generate_until) are
    sent to SGLang first rather than being processed in arrival order.
    """
    while not stop_event.is_set() or not task_queue.empty():
        try:
            item = await asyncio.wait_for(task_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        _, _seq, task_data, overview, deadline_time, sglang_priority, dash_state = item
        asyncio.create_task(
            handle_task(
                task_data, overview, deadline_time, sglang_priority,
                platform, inference, scheduler, dash_state,
            )
        )
        task_queue.task_done()


async def main() -> None:
    cfg      = load_config()
    duration = int(DURATION_OVERRIDE) if DURATION_OVERRIDE else cfg.duration_s
    log.info(f"Config: platform={cfg.platform_url}, model={cfg.model_name}, duration={duration}s")

    platform   = PlatformClient(cfg.platform_url, TOKEN, TEAM_NAME)
    inference  = SGLangClient(SGLANG_URL, cfg.model_name, cfg.model_path)
    scheduler  = Scheduler(max_concurrent=24)
    dash_state = DashboardState()

    await platform.register()

    loop          = asyncio.get_event_loop()
    session_start = loop.time()
    deadline      = session_start + duration
    task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
    stop_event = asyncio.Event()
    seq        = 0  # tiebreaker for equal priorities — preserves FIFO order (avoids tuple comparison errors)

    dispatch_task = asyncio.create_task(
        dispatcher(task_queue, platform, inference, scheduler, stop_event)
    )
    dashboard_task = asyncio.create_task(
        run_dashboard(dash_state, scheduler, cfg.platform_url, stop_event, inference)
    )

    try:
        while loop.time() < deadline:
            # 1. fetch task overview
            overview = await platform.query()
            if overview is None:
                await asyncio.sleep(0.1)
                continue

            # 2. scheduling decision: accept or reject (load threshold + latency estimate)
            if not scheduler.should_accept(overview):
                async with dash_state._lock:
                    dash_state.rejected += 1
                await asyncio.sleep(0)  # yield event loop, retry immediately
                continue

            # 3. accept task, fetch full task data
            task_id   = overview["task_id"]
            task_data = await platform.ask(task_id, overview["target_sla"])
            if task_data is None:
                continue  # rejected or closed — re-query

            sla           = overview["target_sla"]
            deadline_time = loop.time() + SLA_TTFT.get(sla, 600.0)
            sglang_prio   = SLA_SGLANG_PRIORITY.get(sla, 0)
            priority      = _task_priority(overview)

            scheduler.mark_active(task_id)
            async with dash_state._lock:
                dash_state.accepted += 1
                dash_state.sla_counts[sla] = dash_state.sla_counts.get(sla, 0) + 1
            log.info(
                f"Task {task_id} accepted | SLA={sla} "
                f"type={overview['eval_request_type']} "
                f"sp={overview.get('eval_sampling_param', '?')} "
                f"prio={-priority:.1f}"
            )

            # 4. enqueue into priority queue; dispatcher coroutine dequeues in priority order
            seq += 1
            await task_queue.put(
                (priority, seq, task_data, overview, deadline_time, sglang_prio, dash_state)
            )

    finally:
        # signal dispatcher and dashboard to stop
        stop_event.set()
        await asyncio.gather(dispatch_task, dashboard_task, return_exceptions=True)
        # wait for all in-flight inference tasks to finish
        pending = [
            t for t in asyncio.all_tasks()
            if t is not asyncio.current_task()
            and t not in (dispatch_task, dashboard_task)
        ]
        if pending:
            log.info(f"Waiting for {len(pending)} in-flight task(s)...")
            await asyncio.gather(*pending, return_exceptions=True)
        await inference.close()
        await platform.close()
        charts_dir = save_charts(dash_state, scheduler, session_start)
        log.info(f"Charts saved to {charts_dir}")
        log.info("Session ended.")


if __name__ == "__main__":
    asyncio.run(main())
