"""
选手服务主入口

热身阶段启动方式：
    CONFIG_PATH=mock_platform/mock_config.json \\
    TEAM_TOKEN=mytoken \\
    SGLANG_URL=http://localhost:30000 \\
    python -m contestant.main

正式预赛（由 run.sh 调用，环境变量由平台注入）：
    python -m contestant.main

并发模型：
    主循环（poller）持续 query → accept，每接到一个任务立即用
    asyncio.create_task 启动独立协程处理，不等待推理完成就继续接单。
    每个任务协程内部用 asyncio.gather 并发处理所有 messages，
    多条请求同时打到 SGLang，触发 continuous batching。
"""
import asyncio
import logging
import os

from contestant.client import PlatformClient
from contestant.config_loader import load_config
from contestant.inference import SGLangClient
from contestant.scheduler import Scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

TEAM_NAME  = os.environ.get("TEAM_NAME",  "team_alpha")
TOKEN      = os.environ.get("TEAM_TOKEN", "secret_token")
SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:30000")

# 测试时可缩短运行时长（覆盖 config 的 duration_s）
DURATION_OVERRIDE = os.environ.get("DURATION_OVERRIDE")


async def handle_task(
    task_data: dict,
    platform: PlatformClient,
    inference: SGLangClient,
    scheduler: Scheduler,
) -> None:
    """单个任务的完整生命周期：并发推理所有 messages → 提交结果。

    作为独立 asyncio.Task 运行，主循环无需等待即可继续接单。
    """
    task_id = task_data["overview"]["task_id"]
    try:
        # process_messages 内部用 asyncio.gather 并发推理所有 messages
        result_messages = await inference.process_messages(task_data["messages"])
        ok = await platform.submit(task_data["overview"], result_messages)
        log.info(f"Task {task_id} submitted: {'ok' if ok else 'FAIL'}")
    except Exception as e:
        log.error(f"Task {task_id} error: {e}", exc_info=True)
    finally:
        scheduler.mark_complete(task_id)


async def main() -> None:
    cfg = load_config()
    duration = int(DURATION_OVERRIDE) if DURATION_OVERRIDE else cfg.duration_s
    log.info(f"Config: platform={cfg.platform_url}, model={cfg.model_name}, duration={duration}s")

    platform  = PlatformClient(cfg.platform_url, TOKEN, TEAM_NAME)
    inference = SGLangClient(SGLANG_URL, cfg.model_name, cfg.model_path)
    scheduler = Scheduler(max_concurrent=8)

    await platform.register()

    loop = asyncio.get_event_loop()
    deadline = loop.time() + duration

    try:
        while loop.time() < deadline:
            # 1. 拉取任务概要
            overview = await platform.query()
            if overview is None:
                await asyncio.sleep(0.1)
                continue

            # 2. 调度决策：是否接单
            if not scheduler.should_accept(overview):
                await asyncio.sleep(0.05)
                continue

            # 3. 接单，获取完整任务
            task_id   = overview["task_id"]
            task_data = await platform.ask(task_id, overview["target_sla"])
            if task_data is None:
                continue  # rejected or closed，重新 query

            scheduler.mark_active(task_id)
            log.info(
                f"Task {task_id} accepted | SLA={overview['target_sla']} "
                f"type={overview['eval_request_type']} sp={overview['eval_sampling_param']}"
            )

            # 4. 推理+提交作为独立协程，主循环立即继续接下一个任务
            asyncio.create_task(
                handle_task(task_data, platform, inference, scheduler)
            )

    finally:
        # 等待所有在途任务完成后再退出
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if pending:
            log.info(f"Waiting for {len(pending)} in-flight task(s)...")
            await asyncio.gather(*pending, return_exceptions=True)
        await inference.close()
        await platform.close()
        log.info("Session ended.")


if __name__ == "__main__":
    asyncio.run(main())
