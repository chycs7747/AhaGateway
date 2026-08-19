from docker.errors import NotFound
from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas import ModelStatusResponse
from app.services.vllm_manager import VllmManager

router = APIRouter(prefix="/model", tags=["model"])


def get_vllm_manager(request: Request) -> VllmManager:
    return request.app.state.vllm_manager


async def _status_response(manager: VllmManager) -> ModelStatusResponse:
    status = await manager.status()
    ready = await manager.is_ready() if status == "running" else False
    return ModelStatusResponse(container=manager.container_name, status=status, ready=ready)


@router.get("/status", response_model=ModelStatusResponse)
async def model_status(manager: VllmManager = Depends(get_vllm_manager)):
    return await _status_response(manager)


@router.post("/load", response_model=ModelStatusResponse)
async def model_load(manager: VllmManager = Depends(get_vllm_manager)):
    try:
        await manager.start()
    except NotFound:
        raise HTTPException(status_code=404, detail=f"컨테이너 없음: {manager.container_name}")
    return await _status_response(manager)


@router.post("/unload", response_model=ModelStatusResponse)
async def model_unload(manager: VllmManager = Depends(get_vllm_manager)):
    try:
        await manager.stop()
    except NotFound:
        raise HTTPException(status_code=404, detail=f"컨테이너 없음: {manager.container_name}")
    return await _status_response(manager)