import json

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from gateway.config import SESSION_ACQUIRE_TIMEOUT
from gateway.manager_client import ManagerClient, ManagerError
from gateway.schemas import ChatRequest, ChatResponse

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


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    manager: ManagerClient = request.app.state.manager
    client: httpx.AsyncClient = request.app.state.vllm_client
    try:
        async with manager.session(req.model, SESSION_ACQUIRE_TIMEOUT) as sess:
            payload = build_payload(req, sess["served_name"])
            url = f"{sess['vllm_base_url']}/chat/completions"
            try:
                resp = await client.post(url, json=payload)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                raise HTTPException(status_code=503, detail="vLLM 서버에 연결할 수 없습니다")
            except httpx.TimeoutException:
                raise HTTPException(status_code=504, detail="vLLM 응답 시간 초과 (max_tokens를 줄이거나 나중에 재시도)")
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"vLLM 통신 오류: {type(exc).__name__}")

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
    except ManagerError as exc:
        # manager가 이미 옳은 HTTP 코드(404/503/409...)로 거절함 — 그대로 전달
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    manager: ManagerClient = request.app.state.manager
    client: httpx.AsyncClient = request.app.state.vllm_client

    # 세션이 핸들러보다 오래(스트림 끝까지) 살아야 해서 async with 대신 수동 진입
    session_cm = manager.session(req.model, SESSION_ACQUIRE_TIMEOUT)
    try:
        sess = await session_cm.__aenter__()
    except ManagerError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    try:
        payload = build_payload(req, sess["served_name"], stream=True)
        url = f"{sess['vllm_base_url']}/chat/completions"

        async def event_stream():
            try:
                try:
                    async with client.stream("POST", url, json=payload) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            yield sse_event({"kind": "error", "detail": body.decode()})
                            return
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            # NOTE: 스트리밍은 reasoning parser가 켜진 서버 전제.
                            # parser가 꺼진 모델은 <think> 태그가 content에 섞여 나온다.
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
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    yield sse_event({"kind": "error", "detail": "vLLM 서버에 연결할 수 없습니다"})
                except httpx.TimeoutException:
                    yield sse_event({"kind": "error", "detail": "vLLM 응답 시간 초과"})
                except httpx.HTTPError as exc:
                    yield sse_event({"kind": "error", "detail": f"vLLM 통신 오류: {type(exc).__name__}"})
                yield "data: [DONE]\n\n"
            finally:
                await session_cm.__aexit__(None, None, None)  # 스트림이 어떻게 끝나든 반납

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except BaseException:
        # __aenter__ 성공 후 스트림 시작 전에 무엇이 터져도(취소 포함) 세션이 새지 않게.
        # CancelledError는 Exception의 하위가 아니라서 BaseException — 정리 후 재전파 용법.
        await session_cm.__aexit__(None, None, None)
        raise
