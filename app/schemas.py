from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    enable_thinking: bool = True


class ChatResponse(BaseModel):
    content: str
    thinking: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None


class ModelInfo(BaseModel):
    name: str
    container: str
    status: str           # running / exited / created / not_created ...
    active: bool
    ready: bool | None = None   # active 모델에만 의미 있음 (그 외 null)


class LoadRequest(BaseModel):
    name: str


class UnloadResponse(BaseModel):
    unloaded: str | None   # 내려간 모델 이름 (없었으면 null)