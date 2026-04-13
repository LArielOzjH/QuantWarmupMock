#!/usr/bin/env bash
# 正式预赛启动脚本（有时限，耗时操作已在 setup.sh 完成）
# 平台注入的环境变量：MODEL_PATH、CONFIG_PATH、CONTESTANT_PORT、TEAM_NAME、TEAM_TOKEN
set -e

# 激活虚拟环境（若 setup.sh 创建了的话）
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# ── SGLang 启动（通常 setup.sh 已完成，此处几乎零耗时）────────────────
# setup.sh 在 port 30000 启动了 SGLang 并完整预热（Triton JIT + CUDA graph）；
# run.sh 只需确认服务就绪即可，无需重新拉起。
# 若 setup.sh 未能启动（异常情况），此处作为兜底重新启动。

if curl -sf http://localhost:30000/health > /dev/null 2>&1; then
    echo "SGLang already running (pre-warmed by setup.sh), skipping launch."
else
    echo "SGLang not detected, starting now (fallback)..."

    # Detect GPU count and architecture
    TP_SIZE=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
    echo "Detected ${TP_SIZE} GPU(s), starting SGLang with tp-size=${TP_SIZE}"

    # Detect if running on Blackwell (SM 12.x / RTX 5090).
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
    echo "Using attention backend: ${ATTN_BACKEND}, sampling backend: ${SMPL_BACKEND}"

    python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --host 0.0.0.0 \
        --port 30000 \
        --tp-size "${TP_SIZE}" \
        --schedule-policy fcfs \
        --enable-priority-scheduling \
        --chunked-prefill-size 4096 \
        --attention-backend "${ATTN_BACKEND}" \
        --sampling-backend "${SMPL_BACKEND}" &
    SGLANG_PID=$!

    # 兜底启动无 setup.sh 预热，等待时间相应延长
    echo "Waiting for SGLang to be ready (fallback, up to 55s)..."
    for i in $(seq 1 55); do
        if curl -sf http://localhost:30000/health > /dev/null 2>&1; then
            echo "SGLang ready after ${i}s"
            break
        fi
        sleep 1
    done
fi

# 启动选手调度客户端
SGLANG_URL=http://localhost:30000 \
    python -m contestant.main

# 清理后台进程（仅当本脚本自己启动了 SGLang 时才有 SGLANG_PID）
kill "${SGLANG_PID}" 2>/dev/null || true
