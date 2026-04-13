#!/usr/bin/env bash
# 正式预赛启动脚本（有时限，耗时操作已在 setup.sh 完成）
# 平台注入的环境变量：MODEL_PATH、CONFIG_PATH、CONTESTANT_PORT、TEAM_NAME、TEAM_TOKEN
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
# --schedule-policy fcfs      : required when priority scheduling is enabled (fcfs or lof)
# --enable-priority-scheduling: enables request-level priority queue so Supreme SLA tasks dequeue first
# --chunked-prefill-size 4096 : chunked prefill to reduce long-prompt impact on short-SLA TTFT
# --attention-backend         : triton for Blackwell (SM 12.x), flashinfer otherwise
# --sampling-backend          : pytorch for Blackwell (flashinfer sampling JIT also fails on SM 12.x)
# --disable-cuda-graph        : skip CUDA graph build（20-40s）以满足 60s 启动时限。
#   CUDA graph 无法跨进程缓存，必须每次 build；禁用后对高并发 decode batch 吞吐影响极小。
#   Triton JIT 已在 setup.sh 预编译缓存，实际权重加载约 35-45s，可在时限内完成。
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
    --disable-cuda-graph &
SGLANG_PID=$!

# 等待 SGLang 就绪
# Triton JIT 已在 setup.sh 预编译缓存，CUDA graph 已禁用，
# run.sh 主要耗时仅为 32B 权重加载（4×GPU，bf16，约 35-45s）
echo "Waiting for SGLang to be ready..."
for i in $(seq 1 55); do
    if curl -sf http://localhost:30000/health > /dev/null 2>&1; then
        echo "SGLang ready after ${i}s"
        break
    fi
    sleep 1
done

# 启动选手调度客户端
SGLANG_URL=http://localhost:30000 \
    python -m contestant.main

# 清理后台进程
kill "${SGLANG_PID}" 2>/dev/null || true
