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
    path = os.environ.get("CONFIG_PATH", "contest.json")
    with open(path) as f:
        d = json.load(f)
    # 环境变量优先于 JSON 文件（平台注入的值以 env var 为准）
    return ContestConfig(
        platform_url=os.environ.get("PLATFORM_URL") or d["platform_url"],
        model_name=d["model_name"],
        model_path=os.environ.get("MODEL_PATH") or d["model_path"],
        contestant_port=int(os.environ.get("CONTESTANT_PORT") or d.get("contestant_port", 9000)),
        duration_s=d["duration_s"],
        sla_levels=d["sla_levels"],
        sampling_params=d["sampling_params"],
    )
