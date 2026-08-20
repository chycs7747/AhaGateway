import json

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.config import SESSION_ACQUIRE_TIMEOUT
from gateway.manager_client import ManagerClient, ManagerError

router = APIRouter(prefix="/v1", tags=["openai-compat"])


@router.get("/models")
async def list_models(request: Request):
    """OpenAI 형식의 모델 목록 — id는 레지스트리 이름."""
    manager: ManagerClient = request.app.state.manager
    try:
        resp = await manager.proxy("GET", "/models")
    except ManagerError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    return {
        "object": "list",
        "data": [
            {"id": m["name"], "object": "model", "owned_by": "ahagateway"}
            for m in resp.json()
        ],
    }


@router.post("/chat/completions")
async def chat_completions(request: Request):
    """OpenAI 호환 중계: model 필드만 치환하고 나머지는 그대로 통과."""
    body = await request.json()
    stream = bool(body.get("stream", False))
    manager: ManagerClient = request.app.state.manager
    client: httpx.AsyncClient = request.app.state.vllm_client

    session_cm = manager.session(body.get("model"), SESSION_ACQUIRE_TIMEOUT)
    try:
        sess = await session_cm.__aenter__()
    except ManagerError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    try:
        body["model"] = sess["served_name"]  # 레지스트리 이름 → vLLM 서빙 이름
        url = f"{sess['vllm_base_url']}/chat/completions"

        if not stream:
            try:
                resp = await client.post(url, json=body)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                raise HTTPException(status_code=503, detail="vLLM 서버에 연결할 수 없습니다")
            except httpx.TimeoutException:
                raise HTTPException(status_code=504, detail="vLLM 응답 시간 초과")
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"vLLM 통신 오류: {type(exc).__name__}")
            await session_cm.__aexit__(None, None, None)
            return JSONResponse(status_code=resp.status_code, content=resp.json())

        # 스트리밍: vLLM의 SSE 바이트를 재포장 없이 그대로 파이프
        async def passthrough():
            try:
                try:
                    async with client.stream("POST", url, json=body) as resp:
                        if resp.status_code != 200:
                            detail = (await resp.aread()).decode()
                            err = {"error": {"message": f"vLLM 오류: {detail}"}}
                            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode()
                            return
                        async for chunk in resp.aiter_raw():
                            yield chunk
                except httpx.HTTPError as exc:
                    err = {"error": {"message": f"vLLM 통신 오류: {type(exc).__name__}"}}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode()
            finally:
                await session_cm.__aexit__(None, None, None)

        return StreamingResponse(passthrough(), media_type="text/event-stream")
    except BaseException:
        await session_cm.__aexit__(None, None, None)
        raise