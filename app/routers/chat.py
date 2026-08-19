import json

import httpx
from docker.errors import NotFound
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest, ChatResponse
from app.services.vllm_manager import (
    ModelNotReadyError,
    NoActiveModelError,
    UnknownModelError,
    VllmManager,
)



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
    manager: VllmManager = request.app.state.vllm_manager
    client: httpx.AsyncClient = request.app.state.vllm_client
    try:
        async with manager.session(req.model) as served_name:
            payload = build_payload(req, served_name)
            try:
                resp = await client.post("/chat/completions", json=payload)
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
    except UnknownModelError:
        raise HTTPException(status_code=404, detail=f"등록되지 않은 모델: {req.model}")
    except NoActiveModelError:
        raise HTTPException(status_code=503, detail="로드된 모델이 없습니다. 요청에 model을 지정하거나 /model/load를 호출하세요")
    except ModelNotReadyError:
        raise HTTPException(status_code=503, detail=f"모델 준비 시간 초과: {req.model}")
    except NotFound as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    manager: VllmManager = request.app.state.vllm_manager
    client: httpx.AsyncClient = request.app.state.vllm_client

    # 세션이 핸들러보다 오래(스트림 끝까지) 살아야 해서 async with 대신 수동 진입
    session = manager.session(req.model)
    try:
        served_name = await session.__aenter__()
    except UnknownModelError:
        raise HTTPException(status_code=404, detail=f"등록되지 않은 모델: {req.model}")
    except NoActiveModelError:
        raise HTTPException(status_code=503, detail="로드된 모델이 없습니다. 요청에 model을 지정하거나 /model/load를 호출하세요")
    except ModelNotReadyError:
        raise HTTPException(status_code=503, detail=f"모델 준비 시간 초과: {req.model}")
    except NotFound as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    payload = build_payload(req, served_name, stream=True)

    async def event_stream():
        try:
            try:
                async with client.stream("POST", "/chat/completions", json=payload) as resp:
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
            await session.__aexit__(None, None, None)  # 스트림이 어떻게 끝나든 사용 해제

    return StreamingResponse(event_stream(), media_type="text/event-stream")