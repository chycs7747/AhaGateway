from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from gateway.manager_client import ManagerClient, ManagerError
from gateway.schemas import LoadRequest

router = APIRouter(tags=["model"])


async def _relay(manager: ManagerClient, method: str, path: str, json: dict | None = None):
    """manager의 응답을 상태코드째 그대로 클라이언트에 전달한다."""
    try:
        resp = await manager.proxy(method, path, json=json)
    except ManagerError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@router.get("/models")
async def list_models(request: Request):
    return await _relay(request.app.state.manager, "GET", "/models")


@router.post("/model/load")
async def model_load(body: LoadRequest, request: Request):
    return await _relay(request.app.state.manager, "POST", "/model/load", json={"name": body.name})


@router.post("/model/unload")
async def model_unload(request: Request):
    return await _relay(request.app.state.manager, "POST", "/model/unload")
