#!/usr/bin/env bash
# 环境准备脚本（平台在 run.sh 前执行一次，结果被缓存）
# 所有耗时操作放此处：pip 安装 + Triton JIT 预编译
# run.sh 严格限时，setup.sh 无时限
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

# ── Triton JIT 预编译 ────────────────────────────────────────────────
# SGLang 首次使用 Triton backend 时需即时编译 CUDA kernel（可达数分钟），
# 编译产物缓存于 ~/.triton/cache（跨进程持久化）。
# 在此启动一次 warmup server、发送 generate / logprob 请求触发所有 JIT 路径，
# run.sh 启动时直接命中缓存，大幅缩短实际上线时间。
if [ -n "${MODEL_PATH}" ]; then
    echo "==> 预编译 Triton JIT kernel（一次性，结果缓存至磁盘）..."

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
    echo "Warmup backend: attention=${ATTN_BACKEND}, sampling=${SMPL_BACKEND}"

    # 在 31000 端口启动 warmup 实例（与 run.sh 的 30000 端口隔离）
    python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --host 127.0.0.1 --port 31000 \
        --tp-size "${TP_SIZE}" \
        --attention-backend "${ATTN_BACKEND}" \
        --sampling-backend "${SMPL_BACKEND}" \
        --chunked-prefill-size 4096 \
        &
    WARMUP_PID=$!

    # setup.sh 无严格时限，最多等 8 分钟（32B 模型权重加载 + CUDA graph build）
    echo "等待 warmup server 就绪（最多 480s）..."
    READY=0
    for i in $(seq 1 480); do
        if curl -sf http://127.0.0.1:31000/health > /dev/null 2>&1; then
            echo "Warmup server 就绪，耗时 ${i}s"
            READY=1
            break
        fi
        sleep 1
    done

    if [ "${READY}" = "1" ]; then
        # 触发 generate_until 和 logprob 两条 JIT 路径
        python3 - <<'PYEOF'
import requests

BASE = "http://127.0.0.1:31000"

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

print("  Triton JIT warmup 完成。")
PYEOF
    else
        echo "WARNING: Warmup server 未能在规定时间内就绪，跳过 JIT 预编译。"
    fi

    kill "${WARMUP_PID}" 2>/dev/null || true
    wait "${WARMUP_PID}" 2>/dev/null || true
    echo "==> Triton JIT 预编译完成。"
else
    echo "MODEL_PATH 未设置，跳过 SGLang warmup（本地开发模式）。"
fi
