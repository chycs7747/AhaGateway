from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    container: str
    status: str            # running / exited / created / not_created ...
    active: bool
    ready: bool | None = None  # active 모델에만 의미 있음 (그 외 null)


class LoadRequest(BaseModel):
    name: str


class UnloadResponse(BaseModel):
    unloaded: str | None   # 내려간 모델 이름 (없었으면 null)


class SessionRequest(BaseModel):
    model: str | None = None  # None = 지금 떠 있는 모델 사용


class SessionResponse(BaseModel):
    session_id: str
    served_name: str
    vllm_base_url: str     # 이 세션의 추론을 보낼 주소 (멀티 GPU 머신 확장 대비)


class ReleaseResponse(BaseModel):
    released: bool         # False = 이미 만료됐거나 모르는 id
