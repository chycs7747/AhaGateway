from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str | None = None      # None이면 지금 떠 있는 모델
    messages: list[ChatMessage]
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    enable_thinking: bool = True


class ChatResponse(BaseModel):
    content: str
    thinking: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None


class LoadRequest(BaseModel):
    name: str
