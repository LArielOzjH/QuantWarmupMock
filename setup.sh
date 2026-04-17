#!/usr/bin/env bash
# 环境准备脚本（平台在 run.sh 前执行一次，不计入 60s 时限）
# 只做安装，不启动推理引擎（SGLang 启动必须在 run.sh 中完成）
set -e

python3 -m venv .venv
source .venv/bin/activate

pip install -r contestant/requirements.txt

if [ -d "sglang" ]; then
    # sglang 0.5.10rc0 hard-pins cuda-python==12.9；比赛服务器装的是 12.8.0（API 兼容）。
    # 放宽 pin 为下界，避免 pip 报冲突。
    sed -i 's/cuda-python==12\.9/cuda-python>=12.8/' sglang/python/pyproject.toml
    pip install -e "sglang/python/"
else
    echo "WARNING: sglang/ submodule not found, installing latest sglang from PyPI"
    pip install "sglang[srt]"
fi

# flash-attn 非必须，失败不影响运行
pip install flash-attn --no-build-isolation 2>/dev/null \
    && echo "flash-attn installed." \
    || echo "WARNING: flash-attn not installed, SGLang will use fallback attention."
