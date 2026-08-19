import asyncio

import docker
import httpx
from docker.errors import NotFound


class UnknownModelError(Exception):
    """레지스트리에 등록되지 않은 모델 이름."""


class VllmManager:
    """등록된 vLLM 모델 컨테이너들의 수명(생성/시작/정지)을 담당하는 서비스 계층.

    핵심은 ensure(name): 그 모델이 서빙 중인 상태를 보장한다.
    - 컨테이너가 없으면 spec으로 생성 (동적 생성)
    - 다른 모델이 GPU를 쓰고 있으면 먼저 내림 (배타성)
    - 이미 그 모델이 떠 있으면 아무것도 안 함 (멱등성)

    docker SDK는 동기라서 동기 구현(_로 시작)을 asyncio.to_thread로 돌린다.
    """

    def __init__(self, models: dict[str, dict], http_client: httpx.AsyncClient,
                 ready_timeout: float = 600.0):
        self.models = models
        self._docker = docker.from_env()
        self._http = http_client
        self._ready_timeout = ready_timeout

    # ---- 동기 구현부 ----

    def _get_container(self, name: str):
        return self._docker.containers.get(self.models[name]["container"])

    def _status(self, name: str) -> str:
        try:
            return self._get_container(name).status
        except NotFound:
            return "not_created"

    def _active(self) -> str | None:
        for name in self.models:
            if self._status(name) == "running":
                return name
        return None

    def _create(self, name: str):
        spec = self.models[name]["spec"]
        if spec is None:
            raise NotFound(f"컨테이너가 없고 생성 spec도 없음: {name}")
        return self._docker.containers.create(
            spec["image"],
            command=spec["args"],
            name=self.models[name]["container"],
            network_mode="host",
            ipc_mode="host",
            volumes={spec["model_dir"]: {"bind": "/models", "mode": "ro"}},
            device_requests=[
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ],
        )

    def _switch(self, name: str) -> None:
        active = self._active()
        if active == name:
            return  # 이미 떠 있음 — 멱등
        if active is not None:
            self._get_container(active).stop()  # GPU 배타성
        try:
            container = self._get_container(name)
        except NotFound:
            container = self._create(name)      # 동적 생성
        container.start()

    def _unload(self) -> str | None:
        active = self._active()
        if active is None:
            return None
        self._get_container(active).stop()
        return active

    # ---- async 공개 API ----

    async def status(self, name: str) -> str:
        if name not in self.models:
            raise UnknownModelError(name)
        return await asyncio.to_thread(self._status, name)

    async def active(self) -> str | None:
        return await asyncio.to_thread(self._active)

    async def unload(self) -> str | None:
        return await asyncio.to_thread(self._unload)

    async def is_ready(self) -> bool:
        try:
            resp = await self._http.get("/models", timeout=2.0)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    async def ensure(self, name: str, wait_ready: bool = True) -> bool:
        """name 모델이 서빙 중이도록 보장. wait_ready면 준비될 때까지 대기."""
        if name not in self.models:
            raise UnknownModelError(name)
        await asyncio.to_thread(self._switch, name)
        if not wait_ready:
            return await self.is_ready()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._ready_timeout
        while loop.time() < deadline:
            if await self.is_ready():
                return True
            await asyncio.sleep(2.0)
        return False