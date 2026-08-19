from docker.errors import NotFound
from fastapi import APIRouter, Depends, HTTPException, Request

from manager.config import VLLM_BASE_URL
from manager.schemas import (
    LoadRequest,
    ModelInfo,
    ReleaseResponse,
    SessionRequest,
    SessionResponse,
    UnloadResponse,
)
from manager.vllm_manager import (
    DrainTimeoutError,
    ModelNotReadyError,
    NoActiveModelError,
    UnknownModelError,
    VllmManager,
)

router = APIRouter(tags=["manager"])


def get_vllm_manager(request: Request) -> VllmManager:
    return request.app.state.vllm_manager


async def _model_info(manager: VllmManager, name: str) -> ModelInfo:
    status = await manager.status(name)
    active = status == "running"
    return ModelInfo(
        name=name,
        container=manager.models[name]["container"],
        status=status,
        active=active,
        ready=await manager.is_ready(manager.models[name]["served_name"]) if active else None,
    )


@router.get("/models", response_model=list[ModelInfo])
async def list_models(manager: VllmManager = Depends(get_vllm_manager)):
    """등록된 모든 모델과 상태."""
    return [await _model_info(manager, name) for name in manager.models]


@router.post("/model/load", response_model=ModelInfo)
async def model_load(body: LoadRequest, manager: VllmManager = Depends(get_vllm_manager)):
    """관리용: 모델 예열. 진행 중 요청 드레인 후 시작을 걸고 반환 (ready는 /models로 폴링)."""
    try:
        await manager.ensure(body.name, wait_ready=False)
    except UnknownModelError:
        raise HTTPException(status_code=404, detail=f"등록되지 않은 모델: {body.name}")
    except NotFound as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except DrainTimeoutError:
        raise HTTPException(status_code=503, detail="다른 요청이 진행 중이라 전환할 수 없습니다. 잠시 후 재시도하세요")
    return await _model_info(manager, body.name)


@router.post("/model/unload", response_model=UnloadResponse)
async def model_unload(manager: VllmManager = Depends(get_vllm_manager)):
    """관리용: 활성 모델을 내려 GPU를 비운다."""
    try:
        return UnloadResponse(unloaded=await manager.unload())
    except DrainTimeoutError:
        raise HTTPException(status_code=503, detail="진행 중인 요청이 있어 언로드할 수 없습니다")


@router.post("/sessions", response_model=SessionResponse)
async def open_session(body: SessionRequest, manager: VllmManager = Depends(get_vllm_manager)):
    """추론 세션 발급: 모델 보장(필요시 전환) + 사용 등록."""
    try:
        session_id, served = await manager.acquire(body.model)
    except UnknownModelError:
        raise HTTPException(status_code=404, detail=f"등록되지 않은 모델: {body.model}")
    except NoActiveModelError:
        raise HTTPException(status_code=503, detail="로드된 모델이 없습니다. model을 지정하세요")
    except ModelNotReadyError:
        raise HTTPException(status_code=503, detail=f"모델 준비 시간 초과: {body.model}")
    except NotFound as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except DrainTimeoutError:
        raise HTTPException(status_code=503, detail="다른 요청이 진행 중이라 전환할 수 없습니다. 잠시 후 재시도하세요")
    return SessionResponse(session_id=session_id, served_name=served, vllm_base_url=VLLM_BASE_URL)


@router.delete("/sessions/{session_id}", response_model=ReleaseResponse)
async def close_session(session_id: str, manager: VllmManager = Depends(get_vllm_manager)):
    """세션 반납. released=False는 이미 만료됐거나 모르는 id."""
    return ReleaseResponse(released=manager.release(session_id))
