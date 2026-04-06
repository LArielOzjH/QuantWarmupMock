import collections

# SLA 负载阈值：当 load_ratio < threshold 时才接受该 SLA 等级的任务。
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

# SLA 对应的最大允许延迟（秒）
SLA_TTFT: dict[str, float] = {
    "Bronze":   10.0,
    "Silver":    8.0,
    "Gold":      6.0,
    "Platinum":  4.0,
    "Diamond":   2.0,
    "Stellar":   1.5,
    "Glorious":  0.8,
    "Supreme":   0.5,
}

# 安全余量：预估延迟 × SAFETY_MARGIN < sla_ttft 才接单
SAFETY_MARGIN: float = 1.3


class LatencyTracker:
    """滑动窗口平均延迟追踪器（按任务类型 + SLA 分组）。

    记录最近 window 次任务完成的实际延迟（ask→submit 耗时），
    供 Scheduler.should_accept() 做延迟感知的接单决策。
    """

    def __init__(self, window: int = 20):
        self._window = window
        # key: (task_type, sla) -> deque of elapsed seconds
        self._data: dict[tuple, collections.deque] = {}

    def record(self, task_type: str, sla: str, elapsed: float) -> None:
        """记录一次任务完成延迟。"""
        key = (task_type, sla)
        if key not in self._data:
            self._data[key] = collections.deque(maxlen=self._window)
        self._data[key].append(elapsed)

    def avg_latency(self, task_type: str, sla: str) -> float | None:
        """返回滑动窗口平均延迟；无数据时返回 None（冷启动期不拒绝）。"""
        key = (task_type, sla)
        d = self._data.get(key)
        if not d:
            return None
        return sum(d) / len(d)

    def p95_latency(self, task_type: str, sla: str) -> float | None:
        """返回 P95 延迟；无数据时返回 None。"""
        key = (task_type, sla)
        d = self._data.get(key)
        if not d:
            return None
        sorted_d = sorted(d)
        idx = min(int(len(sorted_d) * 0.95), len(sorted_d) - 1)
        return sorted_d[idx]

    def sla_hit_rate(self, task_type: str, sla: str) -> float | None:
        """返回 SLA 达标率 [0,1]；无数据时返回 None。"""
        key = (task_type, sla)
        d = self._data.get(key)
        if not d:
            return None
        limit = SLA_TTFT.get(sla, 600.0)
        return sum(1 for v in d if v <= limit) / len(d)

    def all_keys(self) -> list[tuple]:
        """返回所有有数据的 (task_type, sla) 键。"""
        return list(self._data.keys())


class Scheduler:
    """SLA 感知的任务接单调度器。

    职责：
    - 根据当前系统负载 + 历史延迟决定是否接受新任务
    - 追踪正在处理的任务数量
    - 记录任务完成延迟供滑动窗口统计
    """

    def __init__(self, max_concurrent: int = 8):
        self.max_concurrent = max_concurrent
        self._active: set[int] = set()
        self.latency = LatencyTracker(window=20)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def load_ratio(self) -> float:
        return self.active_count / self.max_concurrent

    def should_accept(self, overview: dict) -> bool:
        """根据 SLA 等级、当前负载和历史延迟决定是否接单。

        两个拒绝条件（任一满足即拒绝）：
        1. 负载阈值：load_ratio >= SLA_LOAD_THRESHOLDS[sla]
        2. 延迟感知：avg_latency * SAFETY_MARGIN >= sla_ttft（有数据时才生效）
        """
        sla       = overview.get("target_sla", "Bronze")
        task_type = overview.get("eval_request_type", "generate_until")

        # 1. 负载阈值检查
        threshold = SLA_LOAD_THRESHOLDS.get(sla, 1.0)
        if self.load_ratio >= threshold:
            return False

        # 2. 延迟感知检查（冷启动期 avg=None 时跳过）
        sla_limit = SLA_TTFT.get(sla, 600.0)
        avg = self.latency.avg_latency(task_type, sla)
        if avg is not None and avg * SAFETY_MARGIN >= sla_limit:
            return False

        return True

    def mark_active(self, task_id: int) -> None:
        self._active.add(task_id)

    def mark_complete(self, task_id: int) -> None:
        self._active.discard(task_id)
