import sys
sys.path.insert(0, ".")

from contestant.scheduler import Scheduler


def _overview(sla: str, task_id: int = 1) -> dict:
    return {
        "task_id": task_id,
        "target_sla": sla,
        "eval_request_type": "generate_until",
        "eval_sampling_param": "Deterministic",
    }


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
