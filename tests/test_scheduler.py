import sys
sys.path.insert(0, ".")

from contestant.scheduler import Scheduler, LatencyTracker, SAFETY_MARGIN, SLA_TTFT


def _overview(sla: str, task_type: str = "generate_until", task_id: int = 1) -> dict:
    return {
        "task_id": task_id,
        "target_sla": sla,
        "eval_request_type": task_type,
        "eval_sampling_param": "Deterministic",
    }


# ---------------------------------------------------------------------------
# 原有负载阈值测试（保持不变）
# ---------------------------------------------------------------------------

def test_accept_bronze_when_idle():
    s = Scheduler(max_concurrent=4)
    assert s.should_accept(_overview("Bronze")) is True


def test_accept_gold_when_idle():
    s = Scheduler(max_concurrent=4)
    assert s.should_accept(_overview("Gold")) is True


def test_reject_supreme_when_overloaded():
    s = Scheduler(max_concurrent=4)
    for i in range(4):
        s.mark_active(i)
    # load_ratio = 1.0 >= threshold(Supreme=0.3)
    assert s.should_accept(_overview("Supreme")) is False


def test_reject_supreme_at_moderate_load():
    s = Scheduler(max_concurrent=4)
    s.mark_active(1)
    s.mark_active(2)
    # load_ratio = 0.5 >= threshold(Supreme=0.3)
    assert s.should_accept(_overview("Supreme")) is False


def test_accept_supreme_when_very_idle():
    s = Scheduler(max_concurrent=10)
    s.mark_active(1)
    # load_ratio = 0.1 < threshold(Supreme=0.3)
    assert s.should_accept(_overview("Supreme")) is True


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


def test_latency_tracker_single_record():
    t = LatencyTracker(window=10)
    t.record("generate_until", "Gold", 2.5)
    assert t.avg_latency("generate_until", "Gold") == 2.5


def test_latency_tracker_average():
    t = LatencyTracker(window=10)
    t.record("loglikelihood", "Bronze", 1.0)
    t.record("loglikelihood", "Bronze", 3.0)
    assert t.avg_latency("loglikelihood", "Bronze") == 2.0


def test_latency_tracker_sliding_window_evicts_old():
    t = LatencyTracker(window=3)
    for v in [10.0, 10.0, 10.0]:
        t.record("generate_until", "Supreme", v)
    # 新加一条 0.1，旧的最早一条 10.0 被驱逐
    t.record("generate_until", "Supreme", 0.1)
    avg = t.avg_latency("generate_until", "Supreme")
    # 窗口内：[10.0, 10.0, 0.1]
    assert abs(avg - (10.0 + 10.0 + 0.1) / 3) < 1e-9


def test_latency_tracker_separate_keys():
    t = LatencyTracker(window=10)
    t.record("generate_until", "Gold", 5.0)
    t.record("loglikelihood", "Gold", 1.0)
    assert t.avg_latency("generate_until", "Gold") == 5.0
    assert t.avg_latency("loglikelihood", "Gold") == 1.0


# ---------------------------------------------------------------------------
# 延迟感知接单决策测试
# ---------------------------------------------------------------------------

def test_accept_when_no_latency_data():
    """冷启动期无历史数据，不应因延迟检查拒绝任务。"""
    s = Scheduler(max_concurrent=8)
    # Supreme 在负载很低时应接受（无历史数据不拒）
    assert s.should_accept(_overview("Supreme")) is True


def test_reject_when_latency_exceeds_sla():
    """历史延迟 × safety_margin >= sla_ttft 时，应拒绝接单。"""
    s = Scheduler(max_concurrent=8)
    # Supreme SLA = 0.5s，safety_margin = 1.3
    # 只要 avg >= 0.5 / 1.3 ≈ 0.385s 就应拒绝
    sla_limit = SLA_TTFT["Supreme"]
    # 注入一个超标延迟
    bad_latency = sla_limit / SAFETY_MARGIN + 0.1  # 刚好超过阈值
    s.latency.record("generate_until", "Supreme", bad_latency)
    assert s.should_accept(_overview("Supreme")) is False


def test_accept_when_latency_within_sla():
    """历史延迟 × safety_margin < sla_ttft 时，应正常接单。"""
    s = Scheduler(max_concurrent=8)
    sla_limit = SLA_TTFT["Gold"]  # 6.0s
    good_latency = (sla_limit / SAFETY_MARGIN) - 0.5  # 明显低于阈值
    s.latency.record("generate_until", "Gold", good_latency)
    assert s.should_accept(_overview("Gold")) is True


def test_latency_check_is_per_task_type():
    """延迟检查按 (task_type, sla) 分组，不同类型互不影响。"""
    s = Scheduler(max_concurrent=8)
    # generate_until Supreme 延迟超标
    bad = SLA_TTFT["Supreme"] / SAFETY_MARGIN + 0.1
    s.latency.record("generate_until", "Supreme", bad)
    # loglikelihood Supreme 无数据，不应被误拒
    assert s.should_accept(_overview("Supreme", task_type="loglikelihood")) is True
