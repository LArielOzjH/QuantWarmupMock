# UBIQUANT Competition — Contestant Framework

A complete participant framework for an UBIQUANT LLM inference serving competition, including a local mock platform for warmup and a production-ready contestant service backed by SGLang.

---

## Competition Overview

The competition evaluates inference serving systems across three task types:

| Type | Description | Scoring |
|------|-------------|---------|
| `generate_until` | Generate text until a stop token | Highest base reward (`w_task = 2.0`) |
| `loglikelihood` | Compute log P(continuation \| prompt) for each choice in a multiple-choice question | Standard reward (`w_task = 1.0`) |
| `loglikelihood_rolling` | Compute total log-likelihood of an entire prompt | Standard reward (`w_task = 1.0`) |

Each task carries an SLA level (Bronze → Supreme) with increasing TTFT requirements and reward multipliers. **TTFT is measured from when `/ask` completes (bid accepted) until `/submit` is received by the platform.** Submitting within the SLA earns full reward; submitting late (within 600 s) scores zero; missing 600 s incurs a penalty of `2 × reward`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Competition LAN                         │
│                                                              │
│  ┌─────────────────┐          ┌────────────────────────────┐ │
│  │  Evaluation     │  HTTP    │     Contestant Machine     │ │
│  │  Platform       │◄────────►│                            │ │
│  │  /register      │          │  asyncio event loop        │ │
│  │  /query         │          │  ┌──────────────────────┐  │ │
│  │  /ask           │          │  │  poller coroutine    │  │ │
│  │  /submit ───────┼──────────┼─►│  query → accept      │  │ │
│  └─────────────────┘          │  └──────────┬───────────┘  │ │
│                               │             │ PriorityQueue│ │
│                               │  ┌──────────▼───────────┐  │ │
│                               │  │  dispatcher          │  │ │
│                               │  │  (priority order)    │  │ │
│                               │  └──────────┬───────────┘  │ │
│                               │             │ create_task  │ │
│                               │  ┌──────────▼───────────┐  │ │
│                               │  │  worker tasks (N)    │  │ │
│                               │  │  gather(messages)    │  │ │
│                               │  └──────────┬───────────┘  │ │
│                               │             │ async HTTP   │ │
│                               │  ┌──────────▼───────────┐  │ │
│                               │  │  SGLang :30000       │  │ │
│                               │  │  fcfs + priority     │  │ │
│                               │  │  continuous batching │  │ │
│                               │  └──────────────────────┘  │ │
│                               └────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

During warmup the evaluation platform is replaced by the local **mock platform** (`mock_platform/`). The contestant code (`contestant/`) is identical in both scenarios — only environment variables change.

---

## Project Structure

```
quant/
├── sglang/                      # SGLang source (git submodule, editable install on GPU machine)
│
├── mock_platform/               # Warmup-only: simulates the official platform
│   ├── config.py                # SLA levels, sampling params, task weights
│   ├── mock_config.json         # Local config (localhost URLs, model paths)
│   ├── scorer.py                # calc_reward() / calc_penalty()
│   ├── task_generator.py        # 24-task pool (8 SLA × 3 types); uniform difficulty, SLA challenge is timing only
│   └── server.py                # FastAPI: /register /query /ask /submit /scores /status
│
├── contestant/                  # Submission package
│   ├── config_loader.py         # Reads CONFIG_PATH env var → ContestConfig dataclass
│   ├── scheduler.py             # SLA-aware accept/reject: dynamic EWMA estimate + probe mechanism
│   ├── inference.py             # SGLangClient: async, process_messages() with asyncio.gather + priority
│   ├── client.py                # PlatformClient: async, wraps the 4 platform HTTP endpoints
│   ├── main.py                  # Async main loop: poller → PriorityQueue → dispatcher → workers
│   ├── dashboard.py             # Rich terminal dashboard: real-time score / latency / task stats
│   ├── run.sh                   # Competition submission entrypoint (starts SGLang + main)
│   ├── setup.sh                 # One-time env setup (venv + pip + sglang editable)
│   └── requirements.txt         # Python deps (sglang installed separately via setup.sh)
│
├── tests/
│   ├── test_scorer.py           # 9 unit tests: reward/penalty formula correctness
│   ├── test_scheduler.py        # 24 unit tests: EWMA, dynamic estimation, probe mechanism
│   ├── test_client.py           # 8 async unit tests: HTTP client with mocked responses
│   └── test_inference.py        # 5 async unit tests (no model) + 5 integration tests
│
├── pytest.ini                   # asyncio_mode=auto, integration mark
└── PLAN.md                      # Full implementation plan with SGLang modification notes
```

---

## Scoring Formula

```
Reward  =  w_task × w_sla × w_sp × correctness
Penalty = −2 × w_task × w_sla × w_sp          (if not submitted within 600 s)
```

| Parameter | Values |
|-----------|--------|
| `w_task` | `generate_until` = 2.0 · `loglikelihood` = 1.0 · `loglikelihood_rolling` = 1.0 |
| `w_sla`  | Bronze 1.0 → Silver 1.2 → Gold 1.5 → Platinum 1.7 → Diamond 2.0 → Stellar 2.2 → Glorious 2.4 → Supreme 2.5 |
| `w_sp`   | Deterministic 1.0 / Normal 1.1 / HighEntropy 1.2 / ExtremePenalty 1.3 (only for `generate_until`) |
| `correctness` | 0.0–1.0 (vs. reference model output) |

---

## Scheduling Strategy

`contestant/scheduler.py` implements a dynamic, queue-depth-aware admission controller. Every task returned by `/query` passes through `should_accept()` before the system calls `/ask`. The decision follows three sequential checks:

---

### Check 1 — Hard concurrency cap

```
active_count >= max_concurrent (= 64)  →  reject immediately
```

`active_count` is the number of tasks currently accepted but not yet submitted. `max_concurrent = 64` acts as an emergency safety net only — in practice, Check 2 below rejects tasks well before this limit is reached.

---

### Check 2 — Queue-depth-aware latency estimate

This is the primary admission gate. It estimates how long a newly accepted task will actually take to complete, factoring in the current state of SGLang's internal queue:

```
queue_factor      =  W / R
estimated_latency =  ewma[(task_type, sla)] × (1 + queue_factor)

accept  ←→  estimated_latency < SLA_TTFT[sla]
```

**What W and R mean:**

| Symbol | Source | Meaning |
|--------|--------|---------|
| `W` | `num_waiting_reqs` from SGLang `/server_info` | Tasks queued inside SGLang, waiting for a GPU slot |
| `R` | `num_running_reqs` from SGLang `/server_info` | Tasks currently executing on the GPU |

**Intuition behind W/R:** if 4 tasks are waiting and 2 are running (`W/R = 2`), a newly admitted task must wait through approximately 2 full inference cycles before it gets a GPU slot. Its total time-to-submit is therefore roughly `ewma × (1 + 2) = 3 × ewma`.

Both values are polled from SGLang's `/server_info` endpoint every **0.5 s** (by the dashboard coroutine) so the signal stays fresh without adding latency to the hot `/query → /ask` path.

**Why W/R is better than a simple `active_count / max_concurrent` ratio:**

The old approach used a fixed logical concurrency ratio (`load_ratio = active_count / 24`). This was blind to task length: 24 fast `loglikelihood` requests and 24 slow `generate_until` requests look identical to a `load_ratio`-based scheduler, even though the latter causes 10× more queue pressure on the GPU. With W/R, slow tasks naturally inflate both W and R, making the queue_factor rise sooner and tighten admission automatically — no tuning required.

**EWMA buckets:** latency is tracked independently per `(task_type, sla)` combination (24 buckets: 8 SLA × 3 types, α = 0.3). A burst of slow Bronze `generate_until` tasks never inflates the EWMA for Supreme `loglikelihood` tasks.

**Cold start:** when EWMA is `None` (no history yet), Check 2 is skipped and all tasks are accepted up to the cap. These early tasks act as probes to seed the EWMA quickly.

**Numerical examples** with `ewma[("generate_until", "Glorious")] = 0.65 s`:

| Queue state (W, R) | queue_factor | Estimated latency | SLA limit | Decision |
|---|---|---|---|---|
| W=0, R=4 (idle) | 0.0 | 0.65 × 1.00 = 0.65 s | 0.8 s | **accept** |
| W=2, R=4 (light) | 0.5 | 0.65 × 1.50 = 0.98 s | 0.8 s | **reject** |
| W=1, R=8 (healthy) | 0.125 | 0.65 × 1.13 = 0.73 s | 0.8 s | **accept** |
| W=6, R=2 (backlog) | 3.0 | 0.65 × 4.00 = 2.60 s | 0.8 s | **reject** |

For a `loglikelihood` Supreme task with `ewma = 0.08 s`, even `W=6, R=2` gives `0.08 × 4 = 0.32 s < 0.5 s` — accepted, because short tasks can still beat tight SLAs even under backlog.

---

### Check 3 — Probe mechanism (anti-starvation)

If the same `(task_type, sla)` key is rejected by Check 2 for `PROBE_THRESHOLD = 8` **consecutive** times, the scheduler force-accepts one task regardless of the latency estimate. This serves two purposes:

1. Refreshes the EWMA with a current sample (prevents stale data from permanently blocking a task category)
2. Ensures every task type eventually gets serviced even under sustained high load

After a force-accept the consecutive-rejection counter resets to zero.

---

### Decision flow summary

```
query() returns overview
        │
        ▼
active_count >= 64?  ──YES──► reject
        │NO
        ▼
ewma[(type, sla)] == None?  ──YES──► accept  (cold start probe)
        │NO
        ▼
queue_factor = W / R
estimated   = ewma × (1 + queue_factor)
estimated >= SLA_TTFT?  ──YES──► consec_rejects++
        │                          consec_rejects >= 8? ──YES──► force-accept (probe)
        │NO                                              └──NO──► reject
        ▼
accept
```

---

## Concurrency Model

The main loop runs as a single asyncio event loop with three logical roles:

**Poller** (`main()` coroutine)
- Continuously calls `/query` → evaluates SLA load threshold + latency → calls `/ask`
- On acceptance: computes `deadline_time = now + sla_ttft` and puts task into `asyncio.PriorityQueue`
- Does **not** wait for inference to finish before accepting the next task

**Dispatcher** (independent coroutine)
- Continuously drains `asyncio.PriorityQueue` in priority order (highest `w_sla × w_task` first)
- Spawns a `handle_task` coroutine via `asyncio.create_task` for each dequeued task
- Ensures that when the system is at capacity, the most valuable pending task goes first

**Worker** (`handle_task` coroutine, up to N running concurrently)
- Calls `inference.process_messages(priority=sglang_priority)`, which fans out all messages via `asyncio.gather`
- All messages for one task (e.g. 4 loglikelihood choices) hit SGLang simultaneously
- SGLang's continuous batching groups these with requests from other concurrent workers
- If `deadline_time` is exceeded, logs a warning but **still submits** — scoring 0 is better than the −2× penalty for missing 600 s
- Records actual latency to `LatencyTracker` so future `should_accept()` decisions improve

**Effect on SGLang**

| Scenario | Requests in-flight to SGLang | Batching |
|----------|------------------------------|---------|
| Serial (old) | 1 at a time | None |
| Async (current) | up to `max_concurrent × messages_per_task` | Full continuous batching |

For `loglikelihood` tasks, all choices share the same prompt. SGLang's **RadixAttention** caches the prompt KV once and only computes the continuation tokens for each choice — making parallel dispatch essentially free on the prefill side.

---

## Priority Design

### Contestant-side priority queue

Accepted tasks wait in a `asyncio.PriorityQueue` before dispatch. Priority score = `w_sla × w_task` (same weights as the reward formula), so the dispatcher always starts the most valuable task first when slots are available.

| SLA + Type | Score | Dispatched |
|------------|-------|------------|
| Supreme + generate_until | 5.0 | First |
| Glorious + generate_until | 4.8 | Second |
| Bronze + loglikelihood | 1.0 | Last |

### SGLang-side priority

Each HTTP request carries a `priority` field (0–7, mapped from SLA level) that SGLang's internal scheduler uses when `--enable-priority-scheduling` is active. This ensures that even within SGLang's waiting queue, Supreme requests preempt Bronze requests.

| SLA | SGLang priority |
|-----|----------------|
| Bronze | 0 |
| Gold | 2 |
| Diamond | 4 |
| Supreme | 7 |

---

## Inference Backend (SGLang)

SGLang is cloned as a git submodule (`sglang/`) and installed in editable mode on the GPU machine. Three inference modes:

- **`generate_until`** → `POST /v1/completions` (OpenAI-compatible, with stop tokens)
- **`loglikelihood`** → `POST /generate` with `return_logprob=True, logprob_start_len=0`; sum logprobs of continuation tokens (token count computed via tokenizer)
- **`loglikelihood_rolling`** → same endpoint; sum all input token logprobs (skip first token)

All requests include `"priority": <int>` in the payload. SGLang silently ignores this field if launched without `--enable-priority-scheduling`, so the code is safe in both modes.

---

## End-to-End Example

This example walks through a single competition round with two tasks arriving simultaneously: a **Supreme generate_until** and a **Bronze loglikelihood**.

### Setup

```
Mock Platform   :8003   (task generator + scorer)
SGLang          :30000  (inference backend)
Contestant      :main   (our code)
```

### Step-by-step trace

```
t=0.00s  Contestant registers with platform
         POST /register → {"status": "ok"}

t=0.05s  Poller: POST /query →
         {"task_id": 1, "target_sla": "Supreme", "eval_request_type": "generate_until",
          "eval_sampling_param": "ExtremePenalty", "eval_timeout_s": 0.5}

         scheduler.should_accept():
           active_count=0 < max_concurrent=64 ✓
           ewma("generate_until", "Supreme") = None → cold start, skip latency check ✓
         → ACCEPT

         POST /ask {task_id: 1, sla: "Supreme"} →
         {"status": "accepted", "task": {"overview": {...}, "messages": [
           {"ID": 0, "prompt": "Explain quantum entanglement in one sentence.",
            "eval_request_type": "generate_until",
            "eval_gen_kwargs": {"until": ["\n"], "max_gen_toks": 80}}
         ]}}

         deadline_time = now + 0.5s
         sglang_priority = 7  (Supreme)
         priority_score  = -(2.5 × 2.0) = -5.0  ← highest priority

         scheduler.mark_active(1)
         task_queue.put((-5.0, seq=1, task_data, ...))

t=0.06s  Poller: POST /query →
         {"task_id": 2, "target_sla": "Bronze", "eval_request_type": "loglikelihood",
          "eval_timeout_s": 10.0}

         scheduler.should_accept():
           active_count=1 < 64 ✓
           queue_factor = W/R = 0/1 = 0.0 (SGLang idle)
           ewma("loglikelihood","Bronze") = None → cold start ✓
         → ACCEPT

         POST /ask {task_id: 2, sla: "Bronze"} →
         {"status": "accepted", "task": {"messages": [
           {"ID": 0, "prompt": "The capital of France is", "eval_continuation": " Paris"},
           {"ID": 1, "prompt": "The capital of France is", "eval_continuation": " London"},
           {"ID": 2, "prompt": "The capital of France is", "eval_continuation": " Berlin"},
           {"ID": 3, "prompt": "The capital of France is", "eval_continuation": " Madrid"}
         ]}}

         priority_score = -(1.0 × 1.0) = -1.0  ← lower priority
         task_queue.put((-1.0, seq=2, task_data, ...))

t=0.06s  Dispatcher wakes: queue has [(-5.0, task1), (-1.0, task2)]
         Pops (-5.0, task1)  →  create_task(handle_task(task1, priority=7))
         Pops (-1.0, task2)  →  create_task(handle_task(task2, priority=0))

         Both workers start concurrently in the asyncio event loop.

t=0.06s  handle_task(task1):
         inference.process_messages([msg0], priority=7)
           → POST /v1/completions {"priority": 7, "prompt": "Explain quantum...",
                                   "max_tokens": 80, "stop": ["\n"]}
         [SGLang: priority=7 request → front of internal queue]

t=0.06s  handle_task(task2):
         inference.process_messages([msg0, msg1, msg2, msg3], priority=0)
           asyncio.gather fires 4 concurrent requests to SGLang:
           POST /generate {"text": "...Paris",  "priority": 0, "return_logprob": true}
           POST /generate {"text": "...London", "priority": 0, ...}
           POST /generate {"text": "...Berlin", "priority": 0, ...}
           POST /generate {"text": "...Madrid", "priority": 0, ...}
         [SGLang RadixAttention: "The capital of France is" prefix cached after first request;
          remaining 3 requests skip the shared prefill → near-free parallel execution]

t=0.28s  SGLang returns generate_until result for task1:
         response = "Quantum entanglement is a phenomenon where two particles..."
         loop.time() < deadline_time (0.28 < 0.56) ✓ SLA met

         POST /submit {task_id: 1, messages: [{response: "Quantum..."}]}
         → {"status": "ok"}

         scheduler.latency.record("generate_until", "Supreme", 0.22s)
         scheduler.mark_complete(1)

         Log: "Task 1 submitted: ok  elapsed=0.22s"

t=0.41s  SGLang returns all 4 loglikelihood results for task2:
         accuracies: [-1.2, -8.4, -7.1, -9.3]   (Paris has highest logprob)

         POST /submit {task_id: 2, messages: [
           {ID:0, accuracy:-1.2}, {ID:1, accuracy:-8.4},
           {ID:2, accuracy:-7.1}, {ID:3, accuracy:-9.3}
         ]}
         → {"status": "ok"}

         scheduler.latency.record("loglikelihood", "Bronze", 0.35s)
         scheduler.mark_complete(2)

         Log: "Task 2 submitted: ok  elapsed=0.35s"

Scores (mock platform, correctness=1.0):
  Task 1: 2.5 × 2.0 × 1.3 × 1.0 = 6.50   (Supreme ExtremePenalty generate_until)
  Task 2: 1.0 × 1.0 × 1.0 × 1.0 = 1.00   (Bronze Deterministic loglikelihood)
```

### Key observations

1. **Priority queue effect**: Task 1 (score 5.0) was dispatched before Task 2 (score 1.0) even though both were accepted within 10 ms of each other. Under high load this prevents a queued Bronze task from blocking a Supreme task.

2. **RadixAttention prefix sharing**: Task 2's 4 loglikelihood messages shared the prompt `"The capital of France is"`. SGLang computed the KV cache for the shared prefix once and ran the 4 continuations in parallel — total time ≈ single request latency, not 4×.

3. **SGLang-side priority**: Task 1's requests carried `priority=7` in the HTTP payload. With `--enable-priority-scheduling`, SGLang would serve Task 1 ahead of any pending `priority<7` requests in its internal queue.

4. **Deadline safety**: Task 1 completed in 0.22 s against a 0.5 s SLA. Even if it had run over, the code would have submitted anyway (scoring 0) rather than risk the −2× penalty from missing the 600 s hard timeout.

---

## Setup

### On the GPU machine (competition / full e2e)

```bash
git clone --recurse-submodules https://github.com/LArielOzjH/QuantWarmupMock.git
cd QuantWarmupMock
bash contestant/setup.sh   # creates venv, installs deps, installs sglang editable
```

### Local development (mock platform only)

```bash
pip install -r contestant/requirements.txt
```

---

## Running

### Warmup — Mock platform only (no model needed)

```bash
# Terminal 1: start mock platform
python -m uvicorn mock_platform.server:app --host 0.0.0.0 --port 8003

# Verify endpoints
curl -X POST http://localhost:8003/register \
  -H "Content-Type: application/json" \
  -d '{"name":"team_alpha","token":"mytoken"}'

curl -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{"token":"mytoken"}'

curl http://localhost:8003/scores
curl http://localhost:8003/status
```

### Full end-to-end (requires GPU + model)

```bash
# Terminal 1: start SGLang with priority scheduling
# Standard (Ampere / Ada / older):
python -m sglang.launch_server \
    --model-path /path/to/model \
    --host 0.0.0.0 --port 30000 --tp-size 1 \
    --schedule-policy fcfs \
    --enable-priority-scheduling \
    --chunked-prefill-size 4096

# Blackwell (RTX 5090 / SM 12.x) on CUDA 12.8:
# FlashInfer JIT cannot compile for SM 12.x without CUDA >= 12.9;
# switch both attention and sampling backends to avoid it.
python -m sglang.launch_server \
    --model-path /path/to/model \
    --host 0.0.0.0 --port 30000 --tp-size 4 \
    --schedule-policy fcfs \
    --enable-priority-scheduling \
    --chunked-prefill-size 4096 \
    --attention-backend triton \
    --sampling-backend pytorch

# Terminal 2: start mock platform
python -m uvicorn mock_platform.server:app --host 0.0.0.0 --port 8003

# Terminal 3: run contestant client with live dashboard (60 s session)
CONFIG_PATH=mock_platform/mock_config.json \
TEAM_TOKEN=mytoken \
SGLANG_URL=http://localhost:30000 \
DURATION_OVERRIDE=60 \
python -m contestant.main
# A rich terminal dashboard refreshes every 0.5s showing score, latency, task stats

# Check score (also shown in dashboard)
curl http://localhost:8003/scores
```

### Competition submission

```bash
# The platform runs setup.sh once (cached), then run.sh each evaluation.
# Injected env vars: MODEL_PATH, CONFIG_PATH, CONTESTANT_PORT
# run.sh auto-detects GPU count (tp-size) and GPU architecture:
#   - Blackwell (SM 12.x) → --attention-backend triton --sampling-backend pytorch
#   - All others          → default flashinfer backends
bash contestant/run.sh
```

---

## Testing

```bash
# All unit tests — no GPU or model required (40 tests)
python -m pytest tests/ -m "not integration" -v

# Integration tests — requires SGLang running on :30000
python -m pytest tests/test_inference.py -m integration -v
```

---

## Warmup vs. Competition

| Aspect | Warmup | Competition |
|--------|--------|-------------|
| Platform | `localhost:8003` (mock) | Official platform (LAN) |
| Model | Qwen3-8B (warmup) | Qwen3-32B |
| GPU | Any CUDA GPU | 4× RTX 5090 (SM 12.0 / Blackwell) |
| Task stream | Synthetic, from `task_generator.py` | Real, from official task pool |
| Correctness scoring | Always 1.0 (no reference model) | Compared against reference model output |
| Startup limit | None | `run.sh` must complete in 60 s |
| Code changes | None — only env vars differ | — |

---

## Planned Optimizations

1. ~~**Async concurrent task processing**~~ ✅ **Done** — `asyncio.create_task` per accepted task; `process_messages()` uses `asyncio.gather` for intra-task message parallelism
2. ~~**Contestant-side priority queue**~~ ✅ **Done** — `asyncio.PriorityQueue` dispatches highest `w_sla × w_task` tasks first; SGLang receives `priority` field per request
3. ~~**Dynamic EWMA latency scheduler**~~ ✅ **Done** — per-(task_type, sla) EWMA (α=0.3) + SGLang queue-depth factor `W/R` replaces static `load_ratio`; probe mechanism prevents starvation; SGLang queue polled every 0.5 s
4. ~~**SGLang startup flags**~~ ✅ **Done** — `--schedule-policy fcfs --enable-priority-scheduling --chunked-prefill-size 4096`
5. ~~**Terminal dashboard**~~ ✅ **Done** — `rich.live.Live` dashboard in `dashboard.py`: score+rate, task stats, latency P95/hit%, task throughput
6. ~~**logprob API fix**~~ ✅ **Done** — replaced invalid `input_token_logprobs:True` with `logprob_start_len:0`; loglikelihood now returns real values
7. ~~**Mock task pool expansion & redesign**~~ ✅ **Done** — 24 tasks (8 SLA × 3 types); uniform difficulty across SLA levels (SLA challenge is timing only, max_gen_toks=32 for all generate_until)
8. **In-memory result cache** — deduplicate identical prompt+continuation pairs within a session
9. **SGLang KV Cache eviction policy** — SLA-aware prefix eviction in `radix_cache.py`
