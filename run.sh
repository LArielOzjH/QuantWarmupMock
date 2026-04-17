#!/usr/bin/env bash
# 正式预赛启动脚本（60s 内拉起推理服务）
# 平台注入的环境变量：MODEL_PATH、CONFIG_PATH、CONTESTANT_PORT、PLATFORM_URL
set -e

# 激活虚拟环境（若 setup.sh 创建了的话）
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Detect GPU count and architecture
TP_SIZE=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
echo "Detected ${TP_SIZE} GPU(s), starting SGLang with tp-size=${TP_SIZE}"

# Detect if running on Blackwell (SM 12.x / RTX 5090).
# torch.cuda.get_device_capability() fails for SM 12.x on CUDA < 12.9,
# so a failed query reliably indicates Blackwell. FlashInfer JIT cannot
# compile for SM 12.x in that case; fall back to the Triton backend.
ATTN_BACKEND=$(python3 - <<'EOF'
import torch
try:
    major, _ = torch.cuda.get_device_capability(0)
    # SM 12.x is Blackwell; FlashInfer requires CUDA >= 12.9 for it
    print("triton" if major >= 12 else "flashinfer")
except Exception:
    # Capability query failed -> likely Blackwell on CUDA < 12.9
    print("triton")
EOF
)
# When attention falls back to triton (Blackwell), sampling must also avoid flashinfer
SMPL_BACKEND=$( [ "${ATTN_BACKEND}" = "triton" ] && echo "pytorch" || echo "flashinfer" )
echo "Using attention backend: ${ATTN_BACKEND}, sampling backend: ${SMPL_BACKEND}"

# Start SGLang inference backend (background)
# --schedule-policy fcfs      : required when priority scheduling is enabled
# --enable-priority-scheduling: Supreme SLA tasks dequeue first
# --chunked-prefill-size 4096 : reduce long-prompt impact on short-SLA TTFT
# --attention-backend         : triton for Blackwell (SM 12.x), flashinfer otherwise
# --sampling-backend          : pytorch for Blackwell
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

# 等待 SGLang 就绪（60s 时限内；主要耗时为权重加载 ~40s + CUDA graph build）
echo "Waiting for SGLang to be ready..."
for i in $(seq 1 55); do
    if curl -sf http://localhost:30000/health > /dev/null 2>&1; then
        echo "SGLang ready after ${i}s"
        break
    fi
    sleep 1
done

# 启动选手调度客户端（前台运行，持续整个比赛时长）
SGLANG_URL=http://localhost:30000 \
TEAM_NAME="quan't" \
TEAM_TOKEN="6010800a93a86d27da7c2cfdad28c2d2" \
    python -m contestant.main

# 清理后台进程
kill "${SGLANG_PID}" 2>/dev/null || true
