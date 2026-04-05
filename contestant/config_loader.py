import json
import os
from dataclasses import dataclass


@dataclass
class ContestConfig:
    platform_url: str
    model_name: str
    model_path: str
    contestant_port: int
    duration_s: int
    sla_levels: dict   # sla_name -> {"ttft_avg": float}
    sampling_params: dict  # sp_name -> {"temperature": float, ...}


def load_config() -> ContestConfig:
    path = os.environ.get("CONFIG_PATH", "mock_platform/mock_config.json")
    with open(path) as f:
        d = json.load(f)
    return ContestConfig(
        platform_url=d["platform_url"],
        model_name=d["model_name"],
        model_path=d["model_path"],
        contestant_port=int(os.environ.get("CONTESTANT_PORT", d.get("contestant_port", 9000))),
        duration_s=d["duration_s"],
        sla_levels=d["sla_levels"],
        sampling_params=d["sampling_params"],
    )
