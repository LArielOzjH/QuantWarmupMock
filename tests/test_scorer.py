import sys
sys.path.insert(0, ".")

import pytest
from mock_platform.scorer import calc_reward, calc_penalty


def test_reward_generate_until_bronze_deterministic():
    # w_task=2.0, w_sla=1.0, w_sp=1.0, C=1.0
    assert calc_reward("generate_until", "Bronze", "Deterministic", 1.0) == pytest.approx(2.0)


def test_reward_generate_until_supreme_extreme_penalty():
    # w_task=2.0, w_sla=2.5, w_sp=1.3, C=1.0 → 6.5
    assert calc_reward("generate_until", "Supreme", "ExtremePenalty", 1.0) == pytest.approx(6.5)


def test_reward_loglikelihood_gold_normal():
    # w_task=1.0, w_sla=1.5, w_sp=1.0 (loglikelihood is unaffected by sp), C=1.0 → 1.5
    assert calc_reward("loglikelihood", "Gold", "Normal", 1.0) == pytest.approx(1.5)


def test_reward_loglikelihood_sp_has_no_effect():
    # loglikelihood w_sp is always 1.0 regardless of sampling_param
    r_det = calc_reward("loglikelihood", "Gold", "Deterministic", 1.0)
    r_ext = calc_reward("loglikelihood", "Gold", "ExtremePenalty", 1.0)
    assert r_det == pytest.approx(r_ext)


def test_reward_zero_correctness():
    assert calc_reward("generate_until", "Gold", "Deterministic", 0.0) == pytest.approx(0.0)


def test_reward_partial_correctness():
    # C=0.5 → reward halved
    full   = calc_reward("generate_until", "Gold", "Deterministic", 1.0)
    half   = calc_reward("generate_until", "Gold", "Deterministic", 0.5)
    assert half == pytest.approx(full * 0.5)


def test_penalty_generate_until_supreme_extreme():
    # 2 × 2.0 × 2.5 × 1.3 = 13.0
    assert calc_penalty("generate_until", "Supreme", "ExtremePenalty") == pytest.approx(13.0)


def test_penalty_loglikelihood_bronze():
    # 2 × 1.0 × 1.0 × 1.0 = 2.0
    assert calc_penalty("loglikelihood", "Bronze", "Normal") == pytest.approx(2.0)


def test_penalty_is_double_reward():
    # penalty = 2 × max_reward (correctness=1)
    reward  = calc_reward("generate_until", "Gold", "Deterministic", 1.0)
    penalty = calc_penalty("generate_until", "Gold", "Deterministic")
    assert penalty == pytest.approx(2 * reward)
