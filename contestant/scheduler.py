# SLA 负载阈值：当 load_ratio < threshold 时才接受该 SLA 等级的任务。
# 数值越低的阈值 = 只在系统空闲时才接 Supreme 等高风险任务，避免超时扣分。
SLA_LOAD_THRESHOLDS: dict[str, float] = {
    "Bronze":   1.0,   # 总是接受
    "Silver":   0.9,
    "Gold":     0.8,
    "Platinum": 0.7,
    "Diamond":  0.6,
    "Stellar":  0.5,
    "Glorious": 0.4,
    "Supreme":  0.3,   # 仅在系统负载 < 30% 时接受
}


class Scheduler:
    """SLA 感知的任务接单调度器。

    职责：
    - 根据当前系统负载决定是否接受新任务
    - 追踪正在处理的任务数量
    """

    def __init__(self, max_concurrent: int = 8):
        self.max_concurrent = max_concurrent
        self._active: set[int] = set()

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def load_ratio(self) -> float:
        return self.active_count / self.max_concurrent

    def should_accept(self, overview: dict) -> bool:
        """根据 SLA 等级和当前负载决定是否接单。"""
        sla = overview.get("target_sla", "Bronze")
        threshold = SLA_LOAD_THRESHOLDS.get(sla, 1.0)
        return self.load_ratio < threshold

    def mark_active(self, task_id: int):
        self._active.add(task_id)

    def mark_complete(self, task_id: int):
        self._active.discard(task_id)
