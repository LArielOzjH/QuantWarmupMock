SLA_LEVELS = {
    "Bronze":   {"ttft_avg": 10.0, "weight": 1.0},
    "Silver":   {"ttft_avg": 8.0,  "weight": 1.2},
    "Gold":     {"ttft_avg": 6.0,  "weight": 1.5},
    "Platinum": {"ttft_avg": 4.0,  "weight": 1.7},
    "Diamond":  {"ttft_avg": 2.0,  "weight": 2.0},
    "Stellar":  {"ttft_avg": 1.5,  "weight": 2.2},
    "Glorious": {"ttft_avg": 0.8,  "weight": 2.4},
    "Supreme":  {"ttft_avg": 0.5,  "weight": 2.5},
}

SAMPLING_PARAMS = {
    "Deterministic":  {"temperature": 0.0, "top_p": 1.0,  "top_k": 1,   "weight": 1.0},
    "Normal":         {"temperature": 0.1, "top_p": 0.9,  "top_k": 50,  "weight": 1.1},
    "HighEntropy":    {"temperature": 0.1, "top_p": 0.95, "top_k": 100, "weight": 1.2},
    "ExtremePenalty": {"temperature": 0.1, "top_p": 0.9,  "top_k": 20,  "weight": 1.3},
}

TASK_TYPE_WEIGHTS = {
    "generate_until":        2.0,
    "loglikelihood":         1.0,
    "loglikelihood_rolling": 1.0,
}

PENALTY_MULTIPLIER = 2.0  # penalty for missing hard timeout = 2 × w_task × w_sla × w_sp
HARD_TIMEOUT_S = 600      # submitting after this deadline incurs a penalty
