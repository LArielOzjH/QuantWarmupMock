#!/usr/bin/env bash
# 正式预赛启动脚本
# 平台注入的环境变量：MODEL_PATH、CONFIG_PATH、CONTESTANT_PORT
set -e

# 激活虚拟环境（若 setup.sh 创建了的话）
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# 检测 GPU 数量，用于 tensor parallelism
TP_SIZE=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
echo "Detected ${TP_SIZE} GPU(s), starting SGLang with tp-size=${TP_SIZE}"

# 启动 SGLang 推理后端（后台运行）
python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size "${TP_SIZE}" &
SGLANG_PID=$!

# 等待 SGLang 就绪（最多 55s，run.sh 总时限 60s）
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
