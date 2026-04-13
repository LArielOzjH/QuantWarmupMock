#!/usr/bin/env bash
# 环境准备脚本（平台在 run.sh 前执行一次）
# 所有耗时操作放此处：pip 安装 + SGLang 完整预热（含 CUDA graph build）
# run.sh 严格限时；setup.sh 无时限，且 SGLang 进程可在此启动后持续运行至 run.sh
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

# ── SGLang 完整预热（Triton JIT + CUDA graph）────────────────────────
# 官方平台确认：setup.sh 可传递环境变量（含 MODEL_PATH），且允许在此
# build CUDA graph。启动的 SGLang 进程在 setup.sh 退出后继续运行，
# run.sh 检测到服务已就绪后直接启动 contestant，无需重新拉起推理服务。
#
# 启动流程：
#   setup.sh → launch SGLang (port 30000) → warmup → exit（进程保留）
#   run.sh   → health check 通过（即时）→ 启动 contestant.main
if [ -n "${MODEL_PATH}" ]; then
    echo "==> 启动 SGLang 并完整预热（Triton JIT + CUDA graph）..."

    TP_SIZE=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
    echo "检测到 ${TP_SIZE} 块 GPU"

    ATTN_BACKEND=$(python3 - <<'EOF'
import torch
try:
    major, _ = torch.cuda.get_device_capability(0)
    print("triton" if major >= 12 else "flashinfer")
except Exception:
    print("triton")
EOF
)
    SMPL_BACKEND=$( [ "${ATTN_BACKEND}" = "triton" ] && echo "pytorch" || echo "flashinfer" )
    echo "Backend: attention=${ATTN_BACKEND}, sampling=${SMPL_BACKEND}"

    # 使用与 run.sh 完全相同的参数启动（port 30000），setup 退出后进程持续运行
    python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --host 0.0.0.0 \
        --port 30000 \
        --tp-size "${TP_SIZE}" \
        --schedule-policy fcfs \
        --enable-priority-scheduling \
        --chunked-prefill-size 4096 \
        --attention-backend "${ATTN_BACKEND}" \
        --sampling-backend "${SMPL_BACKEND}" \
        &

    # setup.sh 无严格时限，最多等 8 分钟（权重加载 + CUDA graph build）
    echo "等待 SGLang 就绪（最多 480s）..."
    READY=0
    for i in $(seq 1 480); do
        if curl -sf http://localhost:30000/health > /dev/null 2>&1; then
            echo "SGLang 就绪，耗时 ${i}s"
            READY=1
            break
        fi
        sleep 1
    done

    if [ "${READY}" = "1" ]; then
        # 发送 warmup 请求，触发 generate / logprob 两条推理路径的完整预热
        python3 - <<'PYEOF'
import requests

BASE = "http://localhost:30000"

print("  Warmup: generate_until 路径...")
requests.post(f"{BASE}/v1/completions", json={
    "model": "model",
    "prompt": "The capital of France is",
    "max_tokens": 8,
    "temperature": 0.0,
}, timeout=120)

print("  Warmup: loglikelihood / logprob 路径...")
requests.post(f"{BASE}/generate", json={
    "text": "The capital of France is Paris.",
    "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
    "return_logprob": True,
    "logprob_start_len": 0,
}, timeout=120)

print("  SGLang 预热完成，进程持续运行等待 run.sh 接管。")
PYEOF
        echo "==> setup.sh 完成，SGLang 在后台持续运行。"
    else
        echo "WARNING: SGLang 未能在规定时间内就绪。"
        echo "  run.sh 将尝试重新启动 SGLang。"
    fi
else
    echo "MODEL_PATH 未设置，跳过 SGLang 启动（本地开发模式）。"
fi
