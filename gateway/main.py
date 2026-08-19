from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from gateway import config
from gateway.manager_client import ManagerClient
from gateway.routers import chat, model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # vLLM 추론용: base_url 없음 — 세션이 알려주는 절대 URL로 요청
    app.state.vllm_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.VLLM_TIMEOUT, connect=config.VLLM_CONNECT_TIMEOUT),
    )
    # manager 통신용
    app.state.manager_http = httpx.AsyncClient(base_url=config.MANAGER_BASE_URL, timeout=30.0)
    app.state.manager = ManagerClient(app.state.manager_http)
    yield
    await app.state.vllm_client.aclose()
    await app.state.manager_http.aclose()


app = FastAPI(title="AhaGateway", lifespan=lifespan)
app.include_router(chat.router)
app.include_router(model.router)


@app.get("/health")
def health():
    return {"status": "ok"}