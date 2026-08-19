import json

import httpx
from docker.errors import NotFound
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest, ChatResponse
from app.services.vllm_manager import UnknownModelError, VllmManager


router = APIRouter(tags=["chat"])


def split_thinking(message: dict) -> tuple[str, str | None]:
    """vLLM 응답 message에서 (content, thinking)을 분리한다.

    reasoning parser가 켜진 서버는 별도 필드로 주지만(빌드에 따라
    필드명이 reasoning 또는 reasoning_content), 꺼진 서버는
    content 안에 <think> 태그로 섞여 오므로 직접 잘라낸다.
    """
    content = message.get("content") or ""
    thinking = message.get("reasoning") or message.get("reasoning_content")
    if thinking is None and "</think>" in content:
        thinking, _, content = content.partition("</think>")
        thinking = thinking.removeprefix("<think>").strip()
        content = content.strip()
    return content, thinking


def build_payload(req: ChatRequest, served_name: str, stream: bool = False) -> dict:
    return {
        "model": served_name,
        "messages": [m.model_dump() for m in req.messages],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "chat_template_kwargs": {"enable_thinking": req.enable_thinking},
        "stream": stream,
    }


def sse_event(obj: dict) -> str:
    """dict 하나를 SSE 이벤트 한 개로 포장한다."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"



async def resolve_model(req: ChatRequest, manager: VllmManager) -> str:
    """요청이 원하는 모델의 served_name을 반환. 필요하면 모델을 전환한다.

    - model 지정 시: ensure()로 전환·준비까지 대기 (요청 주도 전환)
    - 미지정 시: 지금 떠 있는 모델 사용 (없으면 503)
    """
    if req.model is not None:
        try:
            ready = await manager.ensure(req.model, wait_ready=True)
        except UnknownModelError:
            raise HTTPException(status_code=404, detail=f"등록되지 않은 모델: {req.model}")
        except NotFound as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        if not ready:
            raise HTTPException(status_code=503, detail=f"모델 준비 시간 초과: {req.model}")
        return manager.models[req.model]["served_name"]

    active = await manager.active()
    if active is None:
        raise HTTPException(
            status_code=503,
            detail="로드된 모델이 없습니다. 요청에 model을 지정하거나 /model/load를 호출하세요",
        )
    return manager.models[active]["served_name"]


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    manager: VllmManager = request.app.state.vllm_manager
    served_name = await resolve_model(req, manager)
    payload = build_payload(req, served_name)

    client: httpx.AsyncClient = request.app.state.vllm_client
    try:
        resp = await client.post("/chat/completions", json=payload)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="vLLM 서버에 연결할 수 없습니다")

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"vLLM 오류: {resp.text}")

    data = resp.json()
    choice = data["choices"][0]
    content, thinking = split_thinking(choice["message"])
    return ChatResponse(
        content=content,
        thinking=thinking,
        finish_reason=choice.get("finish_reason"),
        usage=data.get("usage"),
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    manager: VllmManager = request.app.state.vllm_manager
    served_name = await resolve_model(req, manager)  # 스트림 시작 전 — 아직 HTTPException 가능
    payload = build_payload(req, served_name, stream=True)
    client: httpx.AsyncClient = request.app.state.vllm_client

    async def event_stream():
        try:
            async with client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield sse_event({"kind": "error", "detail": body.decode()})
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ")
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})
                    thinking = delta.get("reasoning") or delta.get("reasoning_content")
                    if thinking:
                        yield sse_event({"kind": "thinking", "delta": thinking})
                    if delta.get("content"):
                        yield sse_event({"kind": "content", "delta": delta["content"]})
                    if choice.get("finish_reason"):
                        yield sse_event({"kind": "end", "finish_reason": choice["finish_reason"]})
        except httpx.ConnectError:
            yield sse_event({"kind": "error", "detail": "vLLM 서버에 연결할 수 없습니다"})
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")