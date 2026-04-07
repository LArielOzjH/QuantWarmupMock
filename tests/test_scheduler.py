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
# Basic accept/reject logic
# ---------------------------------------------------------------------------

def test_accept_bronze_when_idle():
    s = Scheduler(max_concurrent=4)
    assert s.should_accept(_overview("Bronze")) is True


def test_accept_gold_when_idle():
    s = Scheduler(max_concurrent=4)
    assert s.should_accept(_overview("Gold")) is True


def test_reject_when_at_max_concurrent():
    """Hard reject when active_count == max_concurrent."""
    s = Scheduler(max_concurrent=4)
    for i in range(4):
        s.mark_active(i)
    assert s.should_accept(_overview("Bronze")) is False


def test_cold_start_accepts_any_sla():
    """During cold start (no history), accept any SLA within concurrency limit."""
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
    s.mark_complete(999)  # never was active — should not raise
    assert s.active_count == 0


# ---------------------------------------------------------------------------
# LatencyTracker unit tests
# ---------------------------------------------------------------------------

def test_latency_tracker_no_data_returns_none():
    t = LatencyTracker(window=10)
    assert t.avg_latency("generate_until", "Gold") is None
    assert t.ewma_latency("generate_until", "Gold") is None


def test_latency_tracker_single_record():
    t = LatencyTracker(window=10)
    t.record("generate_until", "Gold", 2.5)
    assert t.avg_latency("generate_until", "Gold") == 2.5
    assert t.ewma_latency("generate_until", "Gold") == 2.5


def test_latency_tracker_ewma_updates():
    """EWMA should move toward the new sample."""
    t = LatencyTracker()
    t.record("generate_until", "Gold", 1.0)
    ewma_after_first = t.ewma_latency("generate_until", "Gold")
    t.record("generate_until", "Gold", 0.1)
    ewma_after_second = t.ewma_latency("generate_until", "Gold")
    assert ewma_after_second < ewma_after_first, "EWMA should decrease after a faster sample"


def test_latency_tracker_ewma_formula():
    """Verify EWMA formula exactly: alpha=0.3."""
    t = LatencyTracker()
    t.record("loglikelihood", "Bronze", 1.0)
    t.record("loglikelihood", "Bronze", 2.0)
    # step 1: ewma=1.0; step 2: 0.3*2.0 + 0.7*1.0 = 1.3
    assert abs(t.ewma_latency("loglikelihood", "Bronze") - 1.3) < 1e-9


def test_latency_tracker_average():
    """avg_latency is based on the (task_type, sla) sample window."""
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
    # window contains: [10.0, 10.0, 0.1]
    assert abs(avg - (10.0 + 10.0 + 0.1) / 3) < 1e-9


def test_latency_tracker_separate_sla_buckets():
    """Different SLA buckets must not interfere (key: prevents Bronze from contaminating Supreme)."""
    t = LatencyTracker(window=10)
    t.record("generate_until", "Bronze", 3.0)   # slow
    t.record("generate_until", "Supreme", 0.3)  # fast
    assert t.ewma_latency("generate_until", "Bronze") == 3.0
    assert t.ewma_latency("generate_until", "Supreme") == 0.3


def test_latency_tracker_separate_task_type_buckets():
    """Different task types must not interfere."""
    t = LatencyTracker(window=10)
    t.record("generate_until", "Gold", 5.0)
    t.record("loglikelihood", "Gold", 1.0)
    assert t.ewma_latency("generate_until", "Gold") == 5.0
    assert t.ewma_latency("loglikelihood", "Gold") == 1.0


# ---------------------------------------------------------------------------
# Dynamic latency estimation decision tests
# ---------------------------------------------------------------------------

def test_accept_when_no_latency_data():
    """No latency history (cold start) — do not reject due to latency check."""
    s = Scheduler(max_concurrent=8)
    assert s.should_accept(_overview("Supreme")) is True


def test_reject_when_estimated_exceeds_sla():
    """Reject when ewma × (1 + load_ratio) >= sla_ttft. At load=0, estimated = ewma."""
    s = Scheduler(max_concurrent=8)
    sla_limit = SLA_TTFT["Supreme"]  # 0.5s
    s.latency.record("generate_until", "Supreme", sla_limit + 0.1)
    assert s.should_accept(_overview("Supreme")) is False


def test_accept_when_estimated_within_sla():
    """Accept when ewma × (1 + load_ratio) < sla_ttft."""
    s = Scheduler(max_concurrent=8)
    sla_limit = SLA_TTFT["Gold"]  # 6.0s
    s.latency.record("generate_until", "Gold", sla_limit - 1.0)  # ewma=5.0 < 6.0
    assert s.should_accept(_overview("Gold")) is True


def test_bronze_slow_does_not_reject_supreme():
    """Slow Bronze gen must not affect Supreme gen decisions (cross-contamination guard)."""
    s = Scheduler(max_concurrent=8)
    # Bronze gen is slow
    s.latency.record("generate_until", "Bronze", 3.0)
    # Supreme gen is fast (independent EWMA)
    s.latency.record("generate_until", "Supreme", 0.3)
    # Supreme (0.5s SLA): ewma=0.3 × 1.0 = 0.3 < 0.5 → accept
    assert s.should_accept(_overview("Supreme")) is True


def test_dynamic_estimation_rejects_at_high_load():
    """Glorious: accepted at low load, rejected at high load due to queuing factor."""
    s = Scheduler(max_concurrent=4)
    s.latency.record("generate_until", "Glorious", 0.6)
    # load=0 → estimated = 0.6 × 1.0 = 0.6 < 0.8 → accept
    assert s.should_accept(_overview("Glorious")) is True

    # load=0.75 → estimated = 0.6 × 1.75 = 1.05 > 0.8 → reject
    s.mark_active(1)
    s.mark_active(2)
    s.mark_active(3)
    assert s.should_accept(_overview("Glorious")) is False


def test_latency_check_is_per_task_type():
    """generate_until over SLA does not affect loglikelihood decisions."""
    s = Scheduler(max_concurrent=8)
    s.latency.record("generate_until", "Supreme", SLA_TTFT["Supreme"] + 0.1)
    # loglikelihood Supreme has no ewma data → cold start → accept
    assert s.should_accept(_overview("Supreme", task_type="loglikelihood")) is True


# ---------------------------------------------------------------------------
# Probe mechanism tests
# ---------------------------------------------------------------------------

def test_probe_mechanism_fires_after_threshold():
    """Force-accept on the Nth attempt after PROBE_THRESHOLD consecutive latency rejections."""
    s = Scheduler(max_concurrent=8)
    s.latency.record("generate_until", "Supreme", 1.0)  # ewma=1.0 > 0.5 SLA

    for i in range(PROBE_THRESHOLD - 1):
        result = s.should_accept(_overview("Supreme"))
        assert result is False, f"Expected reject on attempt {i+1}"

    assert s.should_accept(_overview("Supreme")) is True


def test_probe_resets_after_firing():
    """Counter resets after probe fires; subsequent calls resume normal rejection."""
    s = Scheduler(max_concurrent=8)
    s.latency.record("generate_until", "Supreme", 1.0)

    for _ in range(PROBE_THRESHOLD):
        s.should_accept(_overview("Supreme"))

    assert s.should_accept(_overview("Supreme")) is False


def test_load_reject_does_not_increment_probe_counter():
    """Pure load rejections (load >= 1.0) must not increment the probe counter."""
    s = Scheduler(max_concurrent=2)
    s.latency.record("generate_until", "Supreme", 1.0)
    s.mark_active(1)
    s.mark_active(2)

    for _ in range(PROBE_THRESHOLD + 2):
        assert s.should_accept(_overview("Supreme")) is False

    # free one slot; latency estimate still exceeds SLA, probe count starts at 0, should not fire yet
    s.mark_complete(1)
    assert s.should_accept(_overview("Supreme")) is False
