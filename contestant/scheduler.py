from collections import deque

PROBE_THRESHOLD = 8  # number of consecutive rejections before forcing a probe

# Maximum allowed latency per SLA level (seconds)
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
    """Per-(task_type, sla) dual-purpose latency tracker.

    Both scheduling (should_accept) and display (dashboard/visualizer) use (task_type, sla) granularity:
    - EWMA: alpha=0.3, used for should_accept() estimation, valid from a single sample
    - Sliding window samples: used for P95 / hit-rate / avg display
    """

    ALPHA = 0.3

    def __init__(self, window: int = 50):
        self._window = window
        # (task_type, sla) → EWMA (used for accept/reject decisions)
        self._ewma: dict[tuple, float] = {}
        # (task_type, sla) → deque of samples (used for P95 / hit-rate / avg display)
        self._sla_samples: dict[tuple, deque] = {}

    def record(self, task_type: str, sla: str, elapsed: float) -> None:
        """Record a task completion latency, updating both EWMA and the sample window."""
        key = (task_type, sla)
        if key not in self._ewma:
            self._ewma[key] = elapsed
        else:
            self._ewma[key] = (
                self.ALPHA * elapsed + (1 - self.ALPHA) * self._ewma[key]
            )
        if key not in self._sla_samples:
            self._sla_samples[key] = deque(maxlen=self._window)
        self._sla_samples[key].append(elapsed)

    def ewma_latency(self, task_type: str, sla: str) -> float | None:
        """Return the EWMA for (task_type, sla); returns None during cold start."""
        return self._ewma.get((task_type, sla))

    def avg_latency(self, task_type: str, sla: str) -> float | None:
        """Return the window average (used for dashboard display)."""
        d = self._sla_samples.get((task_type, sla))
        if not d:
            return None
        return sum(d) / len(d)

    def p95_latency(self, task_type: str, sla: str) -> float | None:
        """Return P95 latency (used for dashboard display)."""
        d = self._sla_samples.get((task_type, sla))
        if not d:
            return None
        sorted_d = sorted(d)
        idx = min(int(len(sorted_d) * 0.95), len(sorted_d) - 1)
        return sorted_d[idx]

    def sla_hit_rate(self, task_type: str, sla: str) -> float | None:
        """Return SLA hit rate in [0, 1]."""
        d = self._sla_samples.get((task_type, sla))
        if not d:
            return None
        limit = SLA_TTFT.get(sla, 600.0)
        return sum(1 for v in d if v <= limit) / len(d)

    def all_keys(self) -> list[tuple]:
        """Return all (task_type, sla) keys that have recorded data (used for dashboard iteration)."""
        return list(self._sla_samples.keys())


class Scheduler:
    """Dynamic latency-aware scheduler.

    Accept/reject decision formula:
        estimated = ewma[(task_type, sla)] × (1 + load_ratio)
        accept  ←→  estimated < sla_ttft

    load_ratio acts as a queuing factor: as active_count grows, each new task
    waits longer for a GPU slot, so the estimate naturally rises with load.

    Probe mechanism (deadlock prevention):
        If a (task_type, sla) combination is rejected by the latency check
        PROBE_THRESHOLD consecutive times, one task is force-accepted to collect
        a fresh sample and prevent the EWMA from being permanently stale.

    During cold start (ewma is None), the latency check is skipped and all
    tasks within max_concurrent are accepted.
    """

    def __init__(self, max_concurrent: int = 64):
        self.max_concurrent = max_concurrent
        self._active: set[int] = set()
        self.latency = LatencyTracker(window=50)
        self._consec_rejects: dict[tuple, int] = {}  # (task_type, sla) → consecutive latency-based rejections
        self._sglang_waiting: int = 0  # SGLang internal waiting queue depth (real-time)
        self._sglang_running: int = 1  # SGLang internal running count (real-time); default 1 avoids div-by-zero

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def load_ratio(self) -> float:
        return self.active_count / self.max_concurrent

    def update_sglang_queue(self, waiting: int, running: int = 1) -> None:
        """Update SGLang internal queue stats (used for both scheduling decisions and display)."""
        self._sglang_waiting = waiting
        self._sglang_running = max(running, 1)  # guard against 0 to avoid div-by-zero

    def should_accept(self, overview: dict) -> bool:
        """Dynamically estimate completion time and decide whether to accept the task.

        Rejection conditions:
        1. Hard cap: active_count >= max_concurrent (system fully loaded)
        2. Latency estimate: ewma[(task_type, sla)] × (1 + load_ratio) >= sla_ttft
           and consecutive rejection count has not yet reached the probe threshold

        Returns True to accept, False to reject.
        """
        sla       = overview.get("target_sla") or "Bronze"
        task_type = overview.get("eval_request_type") or "generate_until"
        key       = (task_type, sla)

        # 1. Hard concurrency cap (emergency safety net only; primary control is queue_factor below)
        if self.active_count >= self.max_concurrent:
            return False

        # 2. Dynamic latency estimate using SGLang queue depth (skipped during cold start when ewma=None)
        # queue_factor = W/R: if W=4 tasks waiting and R=2 running, a new task waits ~2 inference
        # cycles before getting a GPU slot → estimated latency = ewma × (1 + W/R).
        # This is task-length-aware: slow generate_until tasks inflate W and saturate R faster,
        # causing queue_factor to rise sooner than a fixed load_ratio would.
        ewma = self.latency.ewma_latency(task_type, sla)
        if ewma is not None:
            queue_factor = self._sglang_waiting / self._sglang_running
            estimated = ewma * (1.0 + queue_factor)
            sla_limit = SLA_TTFT.get(sla, 600.0)
            if estimated >= sla_limit:
                count = self._consec_rejects.get(key, 0) + 1
                self._consec_rejects[key] = count
                if count < PROBE_THRESHOLD:
                    return False
                # probe: reset counter and force-accept
                self._consec_rejects[key] = 0
                return True

        self._consec_rejects[key] = 0
        return True

    def mark_active(self, task_id: int) -> None:
        self._active.add(task_id)

    def mark_complete(self, task_id: int) -> None:
        self._active.discard(task_id)
