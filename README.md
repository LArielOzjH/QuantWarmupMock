# Inference Serving Competition — Contestant Framework

A complete participant framework for an LLM inference serving competition, including a local mock platform for warmup and a production-ready contestant service backed by SGLang.

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
│                      Competition LAN                          │
│                                                              │
│  ┌─────────────────┐          ┌────────────────────────────┐ │
│  │  Evaluation     │  HTTP    │     Contestant Machine     │ │
│  │  Platform       │◄────────►│                            │ │
│  │  /register      │          │  asyncio event loop        │ │
│  │  /query         │          │  ┌──────────────────────┐  │ │
│  │  /ask           │          │  │  poller coroutine    │  │ │
│  │  /submit ───────┼──────────┼─►│  query → accept      │  │ │
│  └─────────────────┘          │  └──────────┬───────────┘  │ │
│                               │             │ create_task  │ │
│                               │  ┌──────────▼───────────┐  │ │
│                               │  │  worker tasks (N)    │  │ │
│                               │  │  gather(messages)    │  │ │
│                               │  └──────────┬───────────┘  │ │
│                               │             │ async HTTP   │ │
│                               │  ┌──────────▼───────────┐  │ │
│                               │  │  SGLang :30000       │  │ │
│                               │  │  continuous batching │  │ │
│                               │  │  RadixAttention      │  │ │
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
│   └── python/sglang/srt/
│       ├── managers/scheduler.py     # planned: priority-queue modification
│       ├── managers/io_struct.py     # planned: Req.priority field
│       └── mem_cache/radix_cache.py  # planned: SLA-aware eviction
│
├── mock_platform/               # Warmup-only: simulates the official platform
│   ├── config.py                # SLA levels, sampling params, task weights
│   ├── mock_config.json         # Local config (localhost URLs, model paths)
│   ├── scorer.py                # calc_reward() / calc_penalty()
│   ├── task_generator.py        # Generates all 3 task types; reuses prompts to stress KV cache
│   └── server.py                # FastAPI: /register /query /ask /submit /scores /status
│
├── contestant/                  # Submission package
│   ├── config_loader.py         # Reads CONFIG_PATH env var → ContestConfig dataclass
│   ├── scheduler.py             # SLA-aware accept/reject decisions via load thresholds
│   ├── inference.py             # SGLangClient: async, process_messages() with asyncio.gather
│   ├── client.py                # PlatformClient: async, wraps the 4 platform HTTP endpoints
│   ├── main.py                  # Async main loop: poller + concurrent handle_task coroutines
│   ├── run.sh                   # Competition submission entrypoint (starts SGLang + main)
│   ├── setup.sh                 # One-time env setup (venv + pip + sglang editable)
│   └── requirements.txt         # Python deps (sglang installed separately via setup.sh)
│
├── tests/
│   ├── test_scorer.py           # 9 unit tests: reward/penalty formula correctness
│   ├── test_scheduler.py        # 9 unit tests: accept/reject logic under various loads
│   ├── test_client.py           # 8 async unit tests: HTTP client with mocked responses
│   └── test_inference.py        # 4 async unit tests (no model) + 5 integration tests
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

`contestant/scheduler.py` maintains a per-SLA load threshold. A task is accepted only when `active_count / max_concurrent` is below the threshold for its SLA level:

| SLA | Threshold | Reasoning |
|-----|-----------|-----------|
| Bronze | 1.0 | Always accept (10 s TTFT, low risk) |
| Gold | 0.8 | Accept unless near capacity |
| Supreme | 0.3 | Only accept when very idle (0.5 s TTFT — high penalty risk) |

---

## Concurrency Model

The main loop runs as a single asyncio event loop with two logical roles:

**Poller** (`main()` coroutine)
- Continuously calls `/query` → evaluates SLA load threshold → calls `/ask`
- On acceptance: immediately spawns a `handle_task` coroutine via `asyncio.create_task`
- Does **not** wait for inference to finish before accepting the next task

**Worker** (`handle_task` coroutine, up to N running concurrently)
- Calls `inference.process_messages()`, which fans out all messages via `asyncio.gather`
- All messages for one task (e.g. 4 loglikelihood choices) hit SGLang simultaneously
- SGLang's continuous batching groups these with requests from other concurrent workers
- Submits results to platform on completion

**Effect on SGLang**

| Scenario | Requests in-flight to SGLang | Batching |
|----------|------------------------------|---------|
| Serial (old) | 1 at a time | None |
| Async (current) | up to `max_concurrent × messages_per_task` | Full continuous batching |

For `loglikelihood` tasks, all choices share the same prompt. SGLang's **RadixAttention** caches the prompt KV once and only computes the continuation tokens for each choice — making parallel dispatch essentially free on the prefill side.

---

## Inference Backend (SGLang)

SGLang is cloned as a git submodule (`sglang/`) and installed in editable mode on the GPU machine, enabling source-level modifications. Three inference modes:

- **`generate_until`** → `POST /v1/completions` (OpenAI-compatible, with stop tokens)
- **`loglikelihood`** → `POST /generate` with `return_logprob=True, input_token_logprobs=True`; sum logprobs of continuation tokens (token count computed via tokenizer)
- **`loglikelihood_rolling`** → same endpoint; sum all input token logprobs (skip first token)

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
# Terminal 1: start SGLang
python -m sglang.launch_server \
    --model-path /path/to/model \
    --host 0.0.0.0 --port 30000 --tp-size 1

# Terminal 2: start mock platform
python -m uvicorn mock_platform.server:app --host 0.0.0.0 --port 8003

# Terminal 3: run contestant client (60 s session)
CONFIG_PATH=mock_platform/mock_config.json \
TEAM_TOKEN=mytoken \
SGLANG_URL=http://localhost:30000 \
DURATION_OVERRIDE=60 \
python -m contestant.main

# Check score
curl http://localhost:8003/scores
```

### Competition submission

```bash
# The platform runs setup.sh once (cached), then run.sh each evaluation
# Injected env vars: MODEL_PATH, CONFIG_PATH, CONTESTANT_PORT
bash contestant/run.sh
```

---

## Testing

```bash
# All unit tests — no GPU or model required (30 tests)
python -m pytest tests/ -m "not integration" -v

# Integration tests — requires SGLang running on :30000
python -m pytest tests/test_inference.py -m integration -v
```

---

## Warmup vs. Competition

| Aspect | Warmup | Competition |
|--------|--------|-------------|
| Platform | `localhost:8003` (mock) | Official platform (LAN) |
| Model | Local small model | Qwen3-32B (`/mnt/model/`) |
| Task stream | Synthetic, from `task_generator.py` | Real, from official task pool |
| Correctness scoring | Always 1.0 (no reference model) | Compared against reference model output |
| Startup limit | None | `run.sh` must complete in 60 s |
| Code changes | None — only env vars differ | — |

---

## Planned Optimizations

1. ~~**Async concurrent task processing**~~ ✅ **Done** — `asyncio.create_task` per accepted task; `process_messages()` uses `asyncio.gather` for intra-task message parallelism
2. **SGLang priority scheduler** — modify `sglang/python/sglang/srt/managers/scheduler.py` to use a priority heap so Supreme SLA tasks preempt Bronze tasks in the waiting queue
3. **In-memory result cache** — deduplicate identical prompt+continuation pairs within a session
4. **SGLang KV Cache eviction policy** — SLA-aware prefix eviction in `radix_cache.py`
