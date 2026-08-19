from docker.errors import NotFound
from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas import LoadRequest, ModelInfo, UnloadResponse
from app.services.vllm_manager import UnknownModelError, VllmManager


router = APIRouter(tags=["model"])


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
    """등록된 모든 모델과 상태. 일반 클라이언트의 '모델 고르기' 용."""
    return [await _model_info(manager, name) for name in manager.models]



@router.post("/model/load", response_model=ModelInfo)
async def model_load(body: LoadRequest, manager: VllmManager = Depends(get_vllm_manager)):
    """관리용: 모델 예열. 시작만 걸고 즉시 반환 (ready는 /models로 폴링)."""
    try:
        await manager.ensure(body.name, wait_ready=False)
    except UnknownModelError:
        raise HTTPException(status_code=404, detail=f"등록되지 않은 모델: {body.name}")
    except NotFound as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return await _model_info(manager, body.name)


@router.post("/model/unload", response_model=UnloadResponse)
async def model_unload(manager: VllmManager = Depends(get_vllm_manager)):
    """관리용: 활성 모델을 내려 GPU를 비운다."""
    return UnloadResponse(unloaded=await manager.unload())