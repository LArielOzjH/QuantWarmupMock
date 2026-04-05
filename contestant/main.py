"""
选手服务主入口

热身阶段启动方式：
    CONFIG_PATH=mock_platform/mock_config.json \\
    TEAM_TOKEN=mytoken \\
    SGLANG_URL=http://localhost:30000 \\
    python -m contestant.main

正式预赛（由 run.sh 调用，环境变量由平台注入）：
    python -m contestant.main
"""
import logging
import os
import time

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


def process_task(task_data: dict, inference: SGLangClient) -> list:
    """对 task 中每条 message 执行推理，返回填充后的 messages 列表。"""
    results = []
    for msg in task_data["messages"]:
        req_type = msg["eval_request_type"]
        m = dict(msg)
        if req_type == "generate_until":
            m["response"] = inference.generate_until(msg["prompt"], msg["eval_gen_kwargs"])
            m["accuracy"] = None
        elif req_type == "loglikelihood":
            m["accuracy"] = inference.loglikelihood(msg["prompt"], msg["eval_continuation"])
            m["response"] = None
        elif req_type == "loglikelihood_rolling":
            m["accuracy"] = inference.loglikelihood_rolling(msg["prompt"])
            m["response"] = None
        else:
            log.warning(f"Unknown request type: {req_type}, skipping")
            continue
        results.append(m)
    return results


def main():
    cfg = load_config()
    duration = int(DURATION_OVERRIDE) if DURATION_OVERRIDE else cfg.duration_s
    log.info(f"Config: platform={cfg.platform_url}, model={cfg.model_name}, duration={duration}s")

    platform  = PlatformClient(cfg.platform_url, TOKEN, TEAM_NAME)
    inference = SGLangClient(SGLANG_URL, cfg.model_name, cfg.model_path)
    scheduler = Scheduler(max_concurrent=8)

    platform.register()

    start = time.time()
    while time.time() - start < duration:
        # 1. 拉取任务概要
        overview = platform.query()
        if overview is None:
            time.sleep(0.1)
            continue

        # 2. 调度决策：是否接单
        if not scheduler.should_accept(overview):
            time.sleep(0.05)
            continue

        # 3. 接单，获取完整任务
        task_id   = overview["task_id"]
        task_data = platform.ask(task_id, overview["target_sla"])
        if task_data is None:
            continue  # rejected or closed，重新 query

        scheduler.mark_active(task_id)
        log.info(
            f"Task {task_id} accepted | SLA={overview['target_sla']} "
            f"type={overview['eval_request_type']} sp={overview['eval_sampling_param']}"
        )

        # 4. 推理 + 提交
        try:
            result_messages = process_task(task_data, inference)
            ok = platform.submit(task_data["overview"], result_messages)
            log.info(f"Task {task_id} submitted: {'ok' if ok else 'FAIL'}")
        except Exception as e:
            log.error(f"Task {task_id} error: {e}", exc_info=True)
        finally:
            scheduler.mark_complete(task_id)

    log.info("Session ended.")
    inference.close()
    platform.close()


if __name__ == "__main__":
    main()
