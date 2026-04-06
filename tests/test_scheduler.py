import sys
sys.path.insert(0, ".")

from contestant.scheduler import Scheduler, LatencyTracker, SLA_TTFT, PROBE_THRESHOLD


def _overview(sla: str, task_type: str = "generate_until", task_id: int = 1) -> dict:
    return {
        "task_id": task_id,
        "target_sla": sla,
        "eval_request_type": task_type,
        "eval_sampling_param": "Deterministic",
    }


# ---------------------------------------------------------------------------
# 基础接单逻辑
# ---------------------------------------------------------------------------

def test_accept_bronze_when_idle():
    s = Scheduler(max_concurrent=4)
    assert s.should_accept(_overview("Bronze")) is True


def test_accept_gold_when_idle():
    s = Scheduler(max_concurrent=4)
    assert s.should_accept(_overview("Gold")) is True


def test_reject_when_at_max_concurrent():
    """active_count == max_concurrent 时硬拒绝。"""
    s = Scheduler(max_concurrent=4)
    for i in range(4):
        s.mark_active(i)
    assert s.should_accept(_overview("Bronze")) is False


def test_cold_start_accepts_any_sla():
    """冷启动无历史数据时，任何 SLA 在负载允许范围内均接受。"""
    s = Scheduler(max_concurrent=8)
    for sla in SLA_TTFT:
        assert s.should_accept(_overview(sla)) is True, f"Should accept {sla} during cold start"


def test_mark_complete_frees_slot():
    s = Scheduler(max_concurrent=1)
    s.mark_active(1)
    assert s.should_accept(_overview("Bronze")) is False
    s.mark_complete(1)
    assert s.should_accept(_overview("Bronze")) is True


def test_load_ratio_calculation():
    s = Scheduler(max_concurrent=8)
    s.mark_active(1)
    s.mark_active(2)
    assert s.load_ratio == 0.25


def test_active_count():
    s = Scheduler(max_concurrent=4)
    assert s.active_count == 0
    s.mark_active(99)
    assert s.active_count == 1
    s.mark_complete(99)
    assert s.active_count == 0


def test_mark_complete_idempotent():
    s = Scheduler(max_concurrent=4)
    s.mark_complete(999)  # 从未 active，不应报错
    assert s.active_count == 0


# ---------------------------------------------------------------------------
# LatencyTracker 单元测试
# ---------------------------------------------------------------------------

def test_latency_tracker_no_data_returns_none():
    t = LatencyTracker(window=10)
    assert t.avg_latency("generate_until", "Gold") is None
    assert t.ewma_latency("generate_until") is None


def test_latency_tracker_single_record():
    t = LatencyTracker(window=10)
    t.record("generate_until", "Gold", 2.5)
    assert t.avg_latency("generate_until", "Gold") == 2.5
    assert t.ewma_latency("generate_until") == 2.5


def test_latency_tracker_ewma_updates():
    """EWMA 应向新样本方向移动。"""
    t = LatencyTracker()
    t.record("generate_until", "Gold", 1.0)
    ewma_after_first = t.ewma_latency("generate_until")
    t.record("generate_until", "Gold", 0.1)
    ewma_after_second = t.ewma_latency("generate_until")
    assert ewma_after_second < ewma_after_first, "EWMA should decrease after a faster sample"


def test_latency_tracker_ewma_formula():
    """精确校验 EWMA 公式：alpha=0.3。"""
    t = LatencyTracker()
    t.record("loglikelihood", "Bronze", 1.0)
    t.record("loglikelihood", "Bronze", 2.0)
    # 第1次: ewma=1.0; 第2次: 0.3*2.0 + 0.7*1.0 = 1.3
    assert abs(t.ewma_latency("loglikelihood") - 1.3) < 1e-9


def test_latency_tracker_average():
    """avg_latency 基于 (task_type, sla) 样本窗口。"""
    t = LatencyTracker(window=10)
    t.record("loglikelihood", "Bronze", 1.0)
    t.record("loglikelihood", "Bronze", 3.0)
    assert t.avg_latency("loglikelihood", "Bronze") == 2.0


def test_latency_tracker_sliding_window_evicts_old():
    t = LatencyTracker(window=3)
    for v in [10.0, 10.0, 10.0]:
        t.record("generate_until", "Supreme", v)
    t.record("generate_until", "Supreme", 0.1)
    avg = t.avg_latency("generate_until", "Supreme")
    # 窗口内：[10.0, 10.0, 0.1]
    assert abs(avg - (10.0 + 10.0 + 0.1) / 3) < 1e-9


def test_latency_tracker_separate_keys():
    """不同 task_type 的 EWMA 互不干扰。"""
    t = LatencyTracker(window=10)
    t.record("generate_until", "Gold", 5.0)
    t.record("loglikelihood", "Gold", 1.0)
    assert t.ewma_latency("generate_until") == 5.0
    assert t.ewma_latency("loglikelihood") == 1.0


# ---------------------------------------------------------------------------
# 动态延迟估算决策测试
# ---------------------------------------------------------------------------

def test_accept_when_no_latency_data():
    """冷启动无历史数据，不因延迟检查拒绝。"""
    s = Scheduler(max_concurrent=8)
    assert s.should_accept(_overview("Supreme")) is True


def test_reject_when_estimated_exceeds_sla():
    """ewma × (1 + load_ratio) >= sla_ttft 时拒绝。load=0 → estimated = ewma。"""
    s = Scheduler(max_concurrent=8)
    sla_limit = SLA_TTFT["Supreme"]  # 0.5s
    # 注入超标延迟：0.6s > 0.5s
    s.latency.record("generate_until", "Supreme", sla_limit + 0.1)
    assert s.should_accept(_overview("Supreme")) is False


def test_accept_when_estimated_within_sla():
    """ewma × (1 + load_ratio) < sla_ttft 时接受。"""
    s = Scheduler(max_concurrent=8)
    sla_limit = SLA_TTFT["Gold"]  # 6.0s
    s.latency.record("generate_until", "Gold", sla_limit - 1.0)  # ewma=5.0 < 6.0
    assert s.should_accept(_overview("Gold")) is True


def test_dynamic_estimation_rejects_at_high_load():
    """generate_until + Glorious：低负载接受，高负载因排队因子超标而拒绝。"""
    s = Scheduler(max_concurrent=4)
    # ewma = 0.6s, sla_ttft(Glorious) = 0.8s
    s.latency.record("generate_until", "Glorious", 0.6)
    # load=0 → estimated = 0.6 × 1.0 = 0.6 < 0.8 → 接受
    assert s.should_accept(_overview("Glorious")) is True

    # 填满 3/4 → load_ratio = 0.75 → estimated = 0.6 × 1.75 = 1.05 > 0.8 → 拒绝
    s.mark_active(1)
    s.mark_active(2)
    s.mark_active(3)
    assert s.should_accept(_overview("Glorious")) is False


def test_dynamic_estimation_accepts_glorious_at_low_load():
    """验证 Glorious 在低负载 + 快速模型时可被接受（非永远拒绝）。"""
    s = Scheduler(max_concurrent=8)
    s.latency.record("generate_until", "Glorious", 0.5)  # ewma=0.5 < 0.8
    assert s.should_accept(_overview("Glorious")) is True


def test_latency_check_is_per_task_type():
    """generate_until 超标不影响 loglikelihood 的决策（EWMA 按 task_type 独立）。"""
    s = Scheduler(max_concurrent=8)
    bad = SLA_TTFT["Supreme"] + 0.1  # 0.6s > 0.5s
    s.latency.record("generate_until", "Supreme", bad)
    # loglikelihood Supreme 无 ewma 数据 → 冷启动 → 接受
    assert s.should_accept(_overview("Supreme", task_type="loglikelihood")) is True


# ---------------------------------------------------------------------------
# 探针机制测试
# ---------------------------------------------------------------------------

def test_probe_mechanism_fires_after_threshold():
    """连续延迟拒绝 PROBE_THRESHOLD 次后，第 N+1 次强制放行。"""
    s = Scheduler(max_concurrent=8)
    # 注入超标延迟（load=0, ewma=1.0 > SLA=0.5）
    s.latency.record("generate_until", "Supreme", 1.0)

    for i in range(PROBE_THRESHOLD - 1):
        result = s.should_accept(_overview("Supreme"))
        assert result is False, f"Expected reject on attempt {i+1}"

    # 第 PROBE_THRESHOLD 次应强制放行
    assert s.should_accept(_overview("Supreme")) is True


def test_probe_resets_after_firing():
    """探针放行后计数重置，之后继续正常拒绝。"""
    s = Scheduler(max_concurrent=8)
    s.latency.record("generate_until", "Supreme", 1.0)

    # 触发一次探针
    for _ in range(PROBE_THRESHOLD):
        s.should_accept(_overview("Supreme"))

    # 探针已触发，计数重置，下一次应重新拒绝
    assert s.should_accept(_overview("Supreme")) is False


def test_load_reject_does_not_increment_probe_counter():
    """纯负载拒绝（load >= 1.0）不应计入探针计数器。"""
    s = Scheduler(max_concurrent=2)
    s.latency.record("generate_until", "Supreme", 1.0)
    s.mark_active(1)
    s.mark_active(2)

    # 连续调用超过 PROBE_THRESHOLD 次，但都是负载拒绝
    for _ in range(PROBE_THRESHOLD + 2):
        assert s.should_accept(_overview("Supreme")) is False

    # 释放一个 slot，负载降到 0.5；延迟估算 = 1.0×1.5 = 1.5 > 0.5，仍拒绝
    # 但探针计数从 0 开始，只拒绝了 1 次，不应触发探针
    s.mark_complete(1)
    assert s.should_accept(_overview("Supreme")) is False
