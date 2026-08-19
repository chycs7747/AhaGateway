from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app import config
from app.routers import chat, model
from app.services.vllm_manager import VllmManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작: 연결 풀을 가진 공용 클라이언트 생성
    app.state.vllm_client = httpx.AsyncClient(
        base_url=config.VLLM_BASE_URL,
        timeout=httpx.Timeout(config.VLLM_TIMEOUT, connect=config.VLLM_CONNECT_TIMEOUT),
    )
    app.state.vllm_manager = VllmManager(config.VLLM_CONTAINER, app.state.vllm_client)
    yield
    # 종료: 열린 연결 정리
    await app.state.vllm_client.aclose()


app = FastAPI(title="Model Gateway", lifespan=lifespan)
app.include_router(chat.router)
app.include_router(model.router)


@app.get("/health")
def health():
    return {"status": "ok"}