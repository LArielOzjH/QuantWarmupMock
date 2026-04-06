from collections import deque

PROBE_THRESHOLD = 8  # 连续拒绝多少次后强制放行一次探针

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


class LatencyTracker:
    """双层延迟追踪器。

    - task_type 级 EWMA：用于 should_accept() 决策（3 个桶，数据充足，收敛快）
    - (task_type, sla) 级滑动窗口：用于 dashboard / visualizer 展示

    EWMA 公式：ewma = alpha × new + (1−alpha) × old
    alpha=0.3 意味着单条新样本使均值向新值移动约 30%，恢复速度远快于 deque 均值。
    """

    ALPHA = 0.3

    def __init__(self, window: int = 50):
        self._window = window
        # task_type → EWMA（决策用）
        self._ewma: dict[str, float] = {}
        # (task_type, sla) → deque of samples（dashboard / P95 / hit-rate 用）
        self._sla_samples: dict[tuple, deque] = {}

    def record(self, task_type: str, sla: str, elapsed: float) -> None:
        """记录一次任务完成延迟。"""
        # EWMA 更新（task_type 级，不区分 SLA）
        if task_type not in self._ewma:
            self._ewma[task_type] = elapsed
        else:
            self._ewma[task_type] = (
                self.ALPHA * elapsed + (1 - self.ALPHA) * self._ewma[task_type]
            )
        # (task_type, sla) 级样本保留
        key = (task_type, sla)
        if key not in self._sla_samples:
            self._sla_samples[key] = deque(maxlen=self._window)
        self._sla_samples[key].append(elapsed)

    def ewma_latency(self, task_type: str) -> float | None:
        """返回 task_type 的 EWMA 均值；冷启动返回 None。"""
        return self._ewma.get(task_type)

    def avg_latency(self, task_type: str, sla: str) -> float | None:
        """返回 (task_type, sla) 的窗口均值（dashboard 展示用）。"""
        d = self._sla_samples.get((task_type, sla))
        if not d:
            return None
        return sum(d) / len(d)

    def p95_latency(self, task_type: str, sla: str) -> float | None:
        """返回 P95 延迟（dashboard 展示用）。"""
        d = self._sla_samples.get((task_type, sla))
        if not d:
            return None
        sorted_d = sorted(d)
        idx = min(int(len(sorted_d) * 0.95), len(sorted_d) - 1)
        return sorted_d[idx]

    def sla_hit_rate(self, task_type: str, sla: str) -> float | None:
        """返回 SLA 达标率 [0,1]。"""
        d = self._sla_samples.get((task_type, sla))
        if not d:
            return None
        limit = SLA_TTFT.get(sla, 600.0)
        return sum(1 for v in d if v <= limit) / len(d)

    def all_keys(self) -> list[tuple]:
        """返回所有有数据的 (task_type, sla) 键（dashboard 遍历用）。"""
        return list(self._sla_samples.keys())


class Scheduler:
    """动态延迟感知调度器。

    接单决策公式：
        estimated = ewma[task_type] × (1 + load_ratio)
        accept  ←→  estimated < sla_ttft

    load_ratio 充当排队因子：当 active_count 增多时，每个新任务平均要多等待
    一段时间才能获得 GPU；该因子让估算自然地随负载升高而变大。

    探针机制（防死锁）：
        若某 (task_type, sla) 组合连续被延迟检查拒绝 PROBE_THRESHOLD 次，
        则强制放行 1 个任务获取新样本，避免 EWMA 长期无法更新导致永久拒绝。

    冷启动期（ewma 为 None）不做延迟检查，接受所有在 max_concurrent 以内的任务。
    """

    def __init__(self, max_concurrent: int = 8):
        self.max_concurrent = max_concurrent
        self._active: set[int] = set()
        self.latency = LatencyTracker(window=50)
        self._consec_rejects: dict[tuple, int] = {}  # (task_type, sla) → 连续延迟拒绝次数

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def load_ratio(self) -> float:
        return self.active_count / self.max_concurrent

    def should_accept(self, overview: dict) -> bool:
        """动态估算完成时间，决定是否接单。

        拒绝条件：
        1. 硬上限：active_count >= max_concurrent（系统已满载）
        2. 延迟估算：ewma[task_type] × (1 + load_ratio) >= sla_ttft
           且连续拒绝次数未达到探针阈值

        返回 True 表示接单，False 表示拒绝。
        """
        sla       = overview.get("target_sla", "Bronze")
        task_type = overview.get("eval_request_type", "generate_until")
        key       = (task_type, sla)

        # 1. 硬上限
        if self.load_ratio >= 1.0:
            return False

        # 2. 动态延迟估算（冷启动 ewma=None 时跳过）
        ewma = self.latency.ewma_latency(task_type)
        if ewma is not None:
            estimated = ewma * (1.0 + self.load_ratio)
            sla_limit = SLA_TTFT.get(sla, 600.0)
            if estimated >= sla_limit:
                count = self._consec_rejects.get(key, 0) + 1
                self._consec_rejects[key] = count
                if count < PROBE_THRESHOLD:
                    return False
                # 探针：重置计数，强制放行
                self._consec_rejects[key] = 0
                return True

        self._consec_rejects[key] = 0
        return True

    def mark_active(self, task_id: int) -> None:
        self._active.add(task_id)

    def mark_complete(self, task_id: int) -> None:
        self._active.discard(task_id)
