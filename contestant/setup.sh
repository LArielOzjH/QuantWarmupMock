#!/usr/bin/env bash
# 环境安装脚本（平台在首次运行前执行，结果会被缓存）
set -e

python3 -m venv .venv
source .venv/bin/activate

pip install -r contestant/requirements.txt

# Install sglang from local submodule (editable, source changes take effect immediately).
# We intentionally omit [all] extras to avoid cuda-python exact-version conflicts:
# sglang[all] pins cuda-python==12.9, but many servers ship cuda-python==12.8.x.
# The contestant code never imports sglang Python — it only calls python -m sglang.launch_server
# as a subprocess — so the core install is sufficient.
if [ -d "sglang" ]; then
    pip install -e "sglang/python/"
else
    echo "WARNING: sglang/ submodule not found, installing latest sglang from PyPI"
    pip install "sglang[srt]"
fi

# flash-attn is not in sglang[core] but significantly improves throughput.
# Build can fail on some CUDA/GCC configurations; we treat it as optional.
echo "Attempting to install flash-attn (optional, improves throughput)..."
pip install flash-attn --no-build-isolation 2>/dev/null \
    && echo "flash-attn installed." \
    || echo "WARNING: flash-attn not installed (build failed). SGLang will fall back to standard attention."
