#!/usr/bin/env bash
# 环境安装脚本（平台在首次运行前执行，结果会被缓存）
set -e

python3 -m venv .venv
source .venv/bin/activate

pip install -r contestant/requirements.txt

if [ -d "sglang" ]; then
    # sglang 0.5.10rc0 在基础依赖中写死了 cuda-python==12.9。
    # 竞赛服务器安装的是 CUDA 12.8，对应 cuda-python==12.8.0，版本不兼容。
    # 将严格 pin 改为下限约束，让 pip 接受已有的 12.8.0（API 兼容，无功能影响）。
    sed -i 's/cuda-python==12\.9/cuda-python>=12.8/' sglang/python/pyproject.toml
    pip install -e "sglang/python/"
else
    echo "WARNING: sglang/ submodule not found, installing latest sglang from PyPI"
    pip install "sglang[srt]"
fi

# flash-attn 不在 sglang 基础依赖中，但能显著提升吞吐量，安装失败不阻断流程
echo "Attempting to install flash-attn (optional, improves throughput)..."
pip install flash-attn --no-build-isolation 2>/dev/null \
    && echo "flash-attn installed." \
    || echo "WARNING: flash-attn not installed (build failed). SGLang will fall back to standard attention."
