from mock_platform.config import (
    TASK_TYPE_WEIGHTS,
    SLA_LEVELS,
    SAMPLING_PARAMS,
    PENALTY_MULTIPLIER,
)


def _w_sp(task_type: str, sampling_param: str) -> float:
    """Sampling param weight: only applies to generate_until; all other task types return 1.0."""
    if task_type == "generate_until":
        return SAMPLING_PARAMS[sampling_param]["weight"]
    return 1.0


def calc_reward(
    task_type: str,
    sla: str,
    sampling_param: str,
    correctness: float,
) -> float:
    """Compute the reward for a task submitted within its SLA.

    R_i = w_task × w_sla × w_sp × C_i
    """
    w_task = TASK_TYPE_WEIGHTS[task_type]
    w_sla  = SLA_LEVELS[sla]["weight"]
    w_sp   = _w_sp(task_type, sampling_param)
    return w_task * w_sla * w_sp * correctness


def calc_penalty(
    task_type: str,
    sla: str,
    sampling_param: str,
) -> float:
    """Compute the penalty for a task not submitted within the 600s hard timeout.

    Penalty = 2 × w_task × w_sla × w_sp
    """
    w_task = TASK_TYPE_WEIGHTS[task_type]
    w_sla  = SLA_LEVELS[sla]["weight"]
    w_sp   = _w_sp(task_type, sampling_param)
    return PENALTY_MULTIPLIER * w_task * w_sla * w_sp
