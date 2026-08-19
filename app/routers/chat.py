import httpx
from fastapi import APIRouter, HTTPException, Request

from app.config import VLLM_MODEL
from app.schemas import ChatRequest, ChatResponse

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

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    payload = {
        "model": VLLM_MODEL,
        "messages": [m.model_dump() for m in req.messages],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "chat_template_kwargs": {"enable_thinking": req.enable_thinking},
    }
    
    client: httpx.AsyncClient = request.app.state.vllm_client
    try:
        resp = await client.post("/chat/completions", json=payload)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="vLLM 서버에 연결할 수 없습니다 (모델 언로드 상태일 수 있음)")
    
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