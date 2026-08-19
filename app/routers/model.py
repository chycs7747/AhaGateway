from docker.errors import NotFound
from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas import ModelStatusResponse
from app.services.vllm_manager import VllmManager

router = APIRouter(prefix="/model", tags=["model"])


def get_vllm_manager(request: Request) -> VllmManager:
    return request.app.state.vllm_manager


@router.get("/status", response_model=ModelStatusResponse)
async def model_status(manager: VllmManager = Depends(get_vllm_manager)):
    status = await manager.status()
    return ModelStatusResponse(container=manager.container_name, status=status)


@router.post("/load", response_model=ModelStatusResponse)
async def model_load(manager: VllmManager = Depends(get_vllm_manager)):
    try:
        status = await manager.start()
    except NotFound:
        raise HTTPException(status_code=404, detail=f"컨테이너 없음: {manager.container_name}")
    return ModelStatusResponse(container=manager.container_name, status=status)


@router.post("/unload", response_model=ModelStatusResponse)
async def model_unload(manager: VllmManager = Depends(get_vllm_manager)):
    try:
        status = await manager.stop()
    except NotFound:
        raise HTTPException(status_code=404, detail=f"컨테이너 없음: {manager.container_name}")
    return ModelStatusResponse(container=manager.container_name, status=status)