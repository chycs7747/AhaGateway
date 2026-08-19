from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from manager import config
from manager.routers import model
from manager.vllm_manager import VllmManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # readiness 핑 전용 클라이언트 (추론 중계는 manager 일이 아님 → 5초면 충분)
    app.state.vllm_client = httpx.AsyncClient(base_url=config.VLLM_BASE_URL, timeout=5.0)
    app.state.vllm_manager = VllmManager(
        config.MODELS,
        app.state.vllm_client,
        ready_timeout=config.MODEL_READY_TIMEOUT,
        drain_timeout=config.SWITCH_DRAIN_TIMEOUT,
        session_ttl=config.SESSION_TTL,
    )
    yield
    await app.state.vllm_client.aclose()


app = FastAPI(title="AhaGateway Manager", lifespan=lifespan)
app.include_router(model.router)


@app.get("/health")
def health():
    return {"status": "ok"}
