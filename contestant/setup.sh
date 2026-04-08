#!/usr/bin/env bash
# Environment setup script (executed once by the platform before run.sh; result is cached)
set -e

python3 -m venv .venv
source .venv/bin/activate

pip install -r contestant/requirements.txt

if [ -d "sglang" ]; then
    # sglang 0.5.10rc0 hard-pins cuda-python==12.9 in its base dependencies.
    # The competition server has CUDA 12.8 (cuda-python==12.8.0), which is API-compatible.
    # Relax the strict pin to a lower-bound so pip accepts the installed 12.8.0.
    sed -i 's/cuda-python==12\.9/cuda-python>=12.8/' sglang/python/pyproject.toml
    pip install -e "sglang/python/"
else
    echo "WARNING: sglang/ submodule not found, installing latest sglang from PyPI"
    pip install "sglang[srt]"
fi

# flash-attn is not in sglang's base deps but can improve throughput; failure is non-fatal
echo "Attempting to install flash-attn (optional, improves throughput)..."
pip install flash-attn --no-build-isolation 2>/dev/null \
    && echo "flash-attn installed." \
    || echo "WARNING: flash-attn not installed (build failed). SGLang will fall back to standard attention."
