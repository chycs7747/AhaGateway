<div align="center">

<img src="assets/logo.svg" alt="AhaGateway" width="440" />

**A FastAPI manager + proxy for a local [vLLM](https://github.com/vllm-project/vllm) server.**

model load/unload via Docker · chat relay · thinking separated from answers · OpenAI-format absorbed at one boundary

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Built with FastAPI](https://img.shields.io/badge/built%20with-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

[API](#api) · [Architecture](#architecture) · [Getting started](#getting-started) · [Design notes](#design-notes)

</div>

---

> ⚠️ **Alpha.** AhaGateway is under active development — expect rough edges and
> breaking changes between versions.

vLLM cannot swap models at runtime, and a loaded model owns most of the GPU.
AhaGateway turns that constraint into an API: it starts and stops the vLLM
Docker container on demand, relays chat to the OpenAI-compatible endpoint, and
returns a clean, minimal schema — with the model's reasoning (`thinking`)
already separated from the final answer.

Companion project: [AhaCode](https://github.com/chycs7747/AhaCode), a
terminal chat client that can talk to this gateway (or any OpenAI-compatible
endpoint).

## API

| Method | Path | Effect |
|---|---|---|
| `GET` | `/health` | Gateway liveness |
| `GET` | `/model/status` | Container state (`running`, `exited`, …) |
| `POST` | `/model/load` | `docker start` the vLLM container |
| `POST` | `/model/unload` | `docker stop` the vLLM container |
| `POST` | `/chat` | Relay a conversation to vLLM |

`POST /chat` takes a deliberately small request — not the full OpenAI surface:

```json
{
  "messages": [{"role": "user", "content": "What is 2^10?"}],
  "temperature": 0.6,
  "max_tokens": 1024,
  "enable_thinking": true
}
```

and answers with thinking and token usage split out:

```json
{
  "content": "2^10 is 1024.",
  "thinking": "The user asks for 2 to the power of 10...",
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 70, "completion_tokens": 59, "total_tokens": 129}
}
```

Interactive docs live at `/docs` (Swagger UI, generated from the Pydantic
schemas).

## Architecture

```
[client] ──> [AhaGateway :9000]                [vLLM :8078 (Docker)]
             POST /model/load   ── docker start ──> container
             POST /model/unload ── docker stop  ──> container
             POST /chat         ── HTTP relay   ──> /v1/chat/completions
```

Inference is 100% vLLM's job; the gateway only controls Docker and relays HTTP.

```
app/
├── main.py               # app assembly: FastAPI instance, lifespan, routers
├── config.py             # vLLM address, container name, timeouts
├── schemas.py            # Pydantic request/response models
├── routers/
│   ├── chat.py           # POST /chat — relay + thinking separation
│   └── model.py          # /model/status · /model/load · /model/unload
└── services/
    └── vllm_manager.py   # Docker container control (docker SDK)
```

## Requirements

- Python 3.12+
- Docker, with a vLLM container serving an OpenAI-compatible API
- The container name and endpoint configured in `app/config.py`

## Getting started

```bash
git clone https://github.com/chycs7747/AhaGateway && cd AhaGateway
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
fastapi dev app/main.py --port 9000
```

Point `app/config.py` at your setup:

```python
VLLM_BASE_URL = "http://localhost:8078/v1"
VLLM_MODEL = "qwen38-nvfp4"
VLLM_CONTAINER = "qwen38"
```

Then try it:

```bash
curl -s -X POST http://localhost:9000/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
```

## Design notes

- **A narrow schema is the point.** `/chat` exposes a handful of fields
  instead of the full OpenAI surface, so the gateway owns validation and
  policy, and the backend can change without touching clients. The
  OpenAI-format conversion happens in exactly one place (`routers/chat.py`).
- **Thinking is absorbed at the boundary.** Whether the server has a
  reasoning parser (separate `reasoning` / `reasoning_content` field) or
  leaks raw `<think>` tags into content, `split_thinking()` normalizes both
  into the same `content` / `thinking` pair.
- **One HTTP client, owned by the app.** A single pooled `httpx.AsyncClient`
  is created in the FastAPI lifespan and closed on shutdown — no per-request
  connections, no import-time side effects.
- **Sync SDKs stay off the event loop.** The docker SDK is synchronous, so
  the service layer runs it via `asyncio.to_thread`; handlers stay `async`
  and the loop keeps serving other requests while containers start and stop.
- **`running` ≠ ready.** After `/model/load` the container is up immediately,
  but vLLM spends minutes loading weights onto the GPU; `/chat` answers `503`
  until the engine is reachable.

## License

[MIT](LICENSE)
