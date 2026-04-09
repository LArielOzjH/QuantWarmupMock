---
title: "LLM Inference Scheduling System"
subtitle: "A Low-Latency, SLA-Aware Online Inference Service Based on SGLang"
author: "Team Alpha"
date: "April 2026"
geometry: "margin=2.5cm"
fontsize: 11pt
linestretch: 1.4
toc: true
toc-depth: 3
numbersections: true
colorlinks: true
linkcolor: "NavyBlue"
urlcolor: "NavyBlue"
monofont: "Menlo"
header-includes:
  - \usepackage{booktabs}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{LLM Inference Scheduling System}
  - \fancyhead[R]{Team Alpha}
---

\newpage

# System Overview

This system is designed around a single objective: **maximizing the SLA (Service Level Agreement) hit rate across concurrent inference tasks** in a competitive multi-SLA evaluation setting. It builds an adaptive scheduling layer on top of [SGLang](https://github.com/sgl-project/sglang), handling admission control, priority queuing, and concurrent inference orchestration for a Qwen3-32B model deployed across four RTX 5090 GPUs.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                Evaluation Platform                   │
│         /query   /ask   /submit   /register          │
└───────────────────┬─────────────────────────────────┘
                    │ HTTP
┌───────────────────▼─────────────────────────────────┐
│           Scheduling Client (contestant.main)        │
│                                                      │
│  ┌──────────────┐   ┌──────────────────────────┐    │
│  │  Poller Loop │──▶│  Scheduler.should_accept  │    │
│  │  (query→ask) │   │  EWMA × (1 + W/R)        │    │
│  └──────────────┘   └────────────┬─────────────┘    │
│                                  │ accept            │
│                     ┌────────────▼─────────────┐    │
│                     │  asyncio.PriorityQueue   │    │
│                     │  (high-value tasks first) │    │
│                     └────────────┬─────────────┘    │
│                                  │                   │
│                     ┌────────────▼─────────────┐    │
│                     │  Dispatcher Coroutine    │    │
│                     │  (concurrent handle_task) │    │
│                     └────────────┬─────────────┘    │
└──────────────────────────────────┼──────────────────┘
                                   │ HTTP
┌──────────────────────────────────▼──────────────────┐
│           SGLang Inference Backend (:30000)          │
│                                                      │
│  Tensor Parallel (tp=4)  ·  Continuous Batching      │
│  Chunked Prefill (4096)  ·  RadixAttention           │
│  Priority Scheduling     ·  Triton/PyTorch Backend   │
└─────────────────────────────────────────────────────┘
```

## Task Types and SLA Levels

Three evaluation task types are supported:

| Task Type | Description | Score Weight |
|---|---|---|
| `generate_until` | Autoregressive text generation with stop tokens | 2.0× |
| `loglikelihood` | Conditional log-probability P(continuation \| prompt) | 1.0× |
| `loglikelihood_rolling` | Total log-likelihood of an entire sequence | 1.0× |

Eight SLA tiers are defined by their TTFT (Time To First Token) budget:

| SLA Level | TTFT Limit | Score Weight |
|---|---|---|
| Bronze | 10.0 s | 1.0× |
| Silver | 8.0 s | 1.2× |
| Gold | 6.0 s | 1.5× |
| Platinum | 4.0 s | 1.7× |
| Diamond | 2.0 s | 2.0× |
| Stellar | 1.5 s | 2.2× |
| Glorious | 0.8 s | 2.4× |
| Supreme | 0.5 s | 2.5× |

---

# Environment Setup and Service Launch

## Dependency Installation (`setup.sh`)

The platform executes `contestant/setup.sh` once at first deployment; the result is cached for subsequent restarts:

```bash
# 1. Create an isolated virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install Python dependencies
pip install -r contestant/requirements.txt

# 3. Install SGLang from local clone
#    Relax cuda-python==12.9 hard pin to >=12.8 for CUDA 12.8 compatibility
sed -i 's/cuda-python==12\.9/cuda-python>=12.8/' sglang/python/pyproject.toml
pip install -e "sglang/python/"

# 4. Optional: install flash-attn for improved attention throughput
pip install flash-attn --no-build-isolation
```

> **Note**: SGLang's upstream dependency hard-pins `cuda-python==12.9`, but the competition server runs CUDA 12.8. Relaxing to a lower-bound `>=12.8` resolves the installation conflict; the two versions are API-compatible.

## Service Launch (`run.sh`)

`contestant/run.sh` orchestrates the complete startup sequence on every invocation.

### Automatic GPU Architecture Detection

```bash
ATTN_BACKEND=$(python3 - <<'EOF'
import torch
try:
    major, _ = torch.cuda.get_device_capability(0)
    print("triton" if major >= 12 else "flashinfer")
except Exception:
    print("triton")   # query failure → Blackwell (SM 12.x) on CUDA < 12.9
EOF
)

# Align sampling backend with attention backend
SMPL_BACKEND=$( [ "${ATTN_BACKEND}" = "triton" ] && echo "pytorch" || echo "flashinfer" )
```

On RTX 5090 (Blackwell, SM 12.0) with CUDA 12.8, `torch.cuda.get_device_capability()` raises an exception because CUDA 12.8 does not expose SM 12.x capability queries. The system catches this and automatically falls back to Triton attention and PyTorch sampling backends, bypassing FlashInfer JIT's SM 12.x incompatibility entirely.

### SGLang Backend Launch

```bash
python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 --port 30000 \
    --tp-size "${TP_SIZE}" \
    --schedule-policy fcfs \
    --enable-priority-scheduling \
    --chunked-prefill-size 4096 \
    --attention-backend "${ATTN_BACKEND}" \
    --sampling-backend "${SMPL_BACKEND}" &
```

The script polls the `/health` endpoint for up to 55 seconds before launching the scheduling client:

```bash
SGLANG_URL=http://localhost:30000 python -m contestant.main
```

---

# Scheduling Strategy: Admission Control

The central question the scheduler answers is: **given a task overview returned by `query`, should the system call `ask` to accept this task?**

## Control Flow

```
query() → overview
    │
    ▼
should_accept(overview)?
    ├── NO  → discard, re-query immediately (tight loop)
    └── YES → ask(task_id, sla) → obtain full task data
                  │
                  ▼
             enqueue into PriorityQueue
```

## Per-Bucket EWMA Latency Tracking

The system maintains an Exponentially Weighted Moving Average (EWMA) of observed end-to-end latency **per `(task_type, sla)` bucket**, with decay factor α = 0.3:

$$\text{EWMA}_t = 0.3 \times \text{elapsed}_t + 0.7 \times \text{EWMA}_{t-1}$$

A parallel sliding window of the 50 most recent samples supports P95 latency, average latency, and SLA hit rate calculations for the real-time dashboard.

**Cold-start behavior**: When EWMA is `None` (no history), the latency check is skipped and all tasks within the concurrency cap are accepted as probes to seed initial estimates quickly.

## Queue-Depth-Aware Admission Formula

$$\hat{T} = \text{EWMA}_{(\text{type},\,\text{sla})} \times \left(1 + \frac{W}{R}\right)$$

where:

- $W$ = `num_waiting_reqs`: tasks queued inside SGLang waiting for a GPU slot (refreshed every 0.5 s from `/server_info`)
- $R$ = `num_running_reqs`: tasks currently executing on the GPU
- $W/R$ is the **queue factor** — a queuing-theory-derived wait multiplier: if $W=4$ tasks are waiting and $R=2$ are running, a newly admitted task waits through approximately 2 full inference cycles before starting

**Admission decision**:

$$\text{accept} \iff \hat{T} < \text{SLA\_TTFT}[\text{sla}]$$

**Comparison with the prior `load_ratio` approach**:

| Approach | Formula | Limitation |
|---|---|---|
| Previous: `load_ratio` | `ewma × (1 + active_count / 24)` | Fixed cap, blind to task length |
| Current: `queue_factor` | `ewma × (1 + W/R)` | Real-time, task-length-aware |

The key advantage of the queue-factor formulation is **implicit task-length sensitivity**: slow `generate_until` tasks occupy running slots longer, causing $R$ to saturate and $W$ to grow faster. This naturally tightens admission without any manual tuning — the SLA limit itself is the only threshold.

## Probe Mechanism (Deadlock Prevention)

If a `(task_type, sla)` combination is rejected by the latency check for `PROBE_THRESHOLD = 8` consecutive times, the system force-accepts one task to collect a fresh EWMA sample. This prevents permanently stale estimates from indefinitely blocking a task category.

---

# Task Dispatch and Execution

## Priority Queue

Accepted tasks are inserted into an `asyncio.PriorityQueue` with priority score:

$$\text{priority} = -(\text{sla\_weight} \times \text{task\_weight})$$

The negation converts Python's min-heap into an effective max-heap, ensuring the highest-value tasks dequeue first:

| Task Type | SLA Level | Composite Priority |
|---|---|---|
| `generate_until` | Supreme | −5.0 (highest) |
| `generate_until` | Bronze | −2.0 |
| `loglikelihood` | Supreme | −2.5 |
| `loglikelihood` | Bronze | −1.0 (lowest) |

## Dispatcher Coroutine

A dedicated `dispatcher` coroutine continuously drains the priority queue and spawns independent `asyncio.Task` instances:

```python
async def dispatcher(task_queue, ...):
    while not stop_event.is_set() or not task_queue.empty():
        item = await asyncio.wait_for(task_queue.get(), timeout=0.1)
        asyncio.create_task(handle_task(...))
```

This fully decouples priority ordering from inference execution: the dispatcher never blocks on a running inference operation, and new high-priority tasks can be dispatched immediately regardless of what is currently running.

## Task Lifecycle (`handle_task`)

```
ask() returns full task data
    │
    ▼
asyncio.gather(*[process_one(msg) for msg in messages])
    │   ← all messages of a task hit SGLang concurrently
    ▼
submit(result_messages)
    │
    ▼
scheduler.latency.record(task_type, sla, elapsed)
```

**Late submission policy**: if the SLA deadline is exceeded, the system still submits the result (scoring 0) rather than abandoning it. This avoids the −2× hard-timeout penalty applied to tasks not submitted within 600 seconds:

$$\text{SLA miss} \Rightarrow 0 \text{ pts} \quad \gg \quad \text{Hard timeout} \Rightarrow -2\times\text{reward}$$

## SGLang Priority Pass-Through

Each inference request carries an integer `priority` field (0–7) mapped to its SLA level:

| SLA | SGLang Priority |
|---|---|
| Bronze | 0 |
| Supreme | 7 |

Combined with `--enable-priority-scheduling`, SGLang's internal scheduler preferentially dispatches high-priority requests, providing end-to-end priority propagation from platform SLA down to individual GPU scheduling decisions.

---

# Inference Acceleration Techniques

## Tensor Parallelism

Qwen3-32B requires approximately 64 GB of GPU memory in BF16 precision. A 4-way tensor parallel configuration (`--tp-size 4`) shards the model across four RTX 5090 GPUs:

- Each GPU holds ~16 GB of model parameters
- Forward passes synchronize intermediate activations via NCCL All-Reduce
- Available KV cache memory per GPU: ~16 GB (remaining after model weights)

## Continuous Batching

SGLang implements iteration-level continuous batching (following the Orca paradigm):

- At each decode step, all currently active sequences are merged into a single batch
- Finished sequences immediately release their KV cache slots; new requests do not wait for the entire batch to complete
- GPU utilization is consistently high throughout the session, even under irregular task arrival patterns

## Chunked Prefill

Long-prompt prefill is split into fixed-size chunks (4096 tokens per chunk):

- Between chunks, decode tokens from other active sequences are interleaved
- Prevents any single long-prompt task from monopolizing the GPU during its prefill phase
- Critically, this enables fine-grained preemption by `--enable-priority-scheduling`: a Supreme task can cut in between prefill chunks of a lower-priority Bronze task, protecting tight TTFT budgets

## RadixAttention (Prefix Cache Reuse)

SGLang organizes the KV cache as a radix tree (prefix trie):

- Requests sharing identical prompt prefixes reuse their cached KV states without recomputation
- `loglikelihood` tasks typically present the same prompt with different continuations; by sending all messages of a task concurrently via `asyncio.gather`, RadixAttention computes the shared prompt prefix once and evaluates all continuations in parallel

## Blackwell GPU Compatibility

The RTX 5090 (SM 12.0, Blackwell) exposes two independent FlashInfer JIT incompatibilities under CUDA 12.8:

| Failure | Trigger Location | Resolution |
|---|---|---|
| Attention JIT failure | `flashinfer/jit/core.py::check_cuda_arch()` | `--attention-backend triton` |
| Sampling JIT failure | `flashinfer/sampling.py::get_sampling_module()` | `--sampling-backend pytorch` |

Both fallbacks are detected and applied automatically in `run.sh` with no manual configuration required.

---

# Real-Time Monitoring Dashboard

A live dashboard (rendered via `rich`, refreshed every 0.5 s) provides operational visibility:

- **Task counters**: accepted / rejected / completed / SLA-miss counts
- **Latency breakdown**: per `(task_type, sla)` EWMA, P95 latency, and SLA hit rate
- **SGLang queue**: live `num_waiting_reqs` / `num_running_reqs` (polled at 0.5 s for scheduling freshness)
- **Throughput timeline**: rolling task completion rate over recent time windows
- **Score tracking**: real-time cumulative score pulled from the platform's `/scores` endpoint

The SGLang queue polling interval was reduced from 2 s to 0.5 s to ensure that the $W$ and $R$ values driving the admission formula remain current under rapidly changing load.

---

# Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Admission signal | SGLang `W/R` queue\_factor | Real-time, task-length-aware; superior to a static logical concurrency ratio |
| EWMA granularity | Per `(task_type, sla)` bucket | Generate\_until and loglikelihood have fundamentally different latency distributions |
| Queue ordering | `asyncio.PriorityQueue` | Ensures high-value tasks reach SGLang first, not just in arrival order |
| Late submission | Always submit | Avoids −2× hard-timeout penalty; expected value of submission always ≥ 0 |
| Cold-start policy | Skip latency check when EWMA=None | Rapidly seeds historical data; avoids over-conservatism in the first few minutes |
| GPU backend | Auto-detect; Blackwell falls back to Triton | Zero-configuration cross-architecture compatibility |

---

*Report version corresponds to code commit `29792b9`, April 2026.*
