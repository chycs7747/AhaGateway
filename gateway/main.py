from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from gateway import config
from gateway.manager_client import ManagerClient
from gateway.routers import chat, model, openai_compat


@asynccontextmanager
async def lifespan(app: FastAPI):
    # vLLM 추론용: base_url 없음 — 세션이 알려주는 절대 URL로 요청
    app.state.vllm_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.VLLM_TIMEOUT, connect=config.VLLM_CONNECT_TIMEOUT),
    )
    # manager 통신용
    app.state.manager_http = httpx.AsyncClient(
        base_url=config.MANAGER_BASE_URL,
        # 관리 호출(load/unload)은 manager가 드레인(180s)을 기다릴 수 있으므로 그보다 길게.
        # 세션 발급은 어차피 요청별 timeout(SESSION_ACQUIRE_TIMEOUT)으로 덮어쓴다.
        timeout=httpx.Timeout(200.0, connect=5.0),
    )
    app.state.manager = ManagerClient(app.state.manager_http)
    yield
    await app.state.vllm_client.aclose()
    await app.state.manager_http.aclose()


app = FastAPI(title="AhaGateway", lifespan=lifespan)
app.include_router(chat.router)
app.include_router(model.router)
app.include_router(openai_compat.router)


@app.get("/health")
def health():
    return {"status": "ok"}