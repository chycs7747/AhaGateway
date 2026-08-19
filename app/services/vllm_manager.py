import asyncio
from contextlib import asynccontextmanager
import docker
import httpx
from docker.errors import NotFound


class UnknownModelError(Exception):
    """레지스트리에 등록되지 않은 모델 이름."""


class NoActiveModelError(Exception):
    """요청이 모델을 지정하지 않았는데 떠 있는 모델도 없음."""


class ModelNotReadyError(Exception):
     """전환은 했지만 준비 시간 안에 ready가 되지 않음."""


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
        self._switch_lock = asyncio.Lock()
        self._inflight = 0           # 진행 중인 추론 요청 수
        self._idle = asyncio.Event() # inflight == 0 일 때 set
        self._idle.set()

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
        async with self._switch_lock:
            await self._idle.wait()
            return await asyncio.to_thread(self._unload)

    async def is_ready(self, served_name: str | None = None) -> bool:
        """vLLM이 응답하는지, (지정 시) '그 모델을' 서빙 중인지 확인."""
        try:
            resp = await self._http.get("/models", timeout=2.0)
        except httpx.HTTPError:
            return False
        if resp.status_code != 200:
            return False
        if served_name is None: # 여기 오면 통과
            return True
        ids = [m.get("id") for m in resp.json().get("data", [])]
        return served_name in ids

    async def ensure(self, name: str, wait_ready: bool = True) -> bool:
        if name not in self.models:
            raise UnknownModelError(name)
        served = self.models[name]["served_name"]
        async with self._switch_lock:
            await self._drain_and_switch(name)
            if not wait_ready:
                return await self.is_ready(served)
            return await self._wait_ready(served)

    @asynccontextmanager
    async def session(self, name: str | None):
        """추론 요청 하나의 수명 (채팅 핸들러용).

        진입: 모델 보장(지정 시 전환 포함) + 사용 등록 → served_name 반환
        종료: 사용 해제 (0이 되면 '비었음' 신호 → 대기 중인 전환 진행)
        """
        served = await self._acquire(name)
        try:
            yield served
        finally:
            self._release()

    # ---- 내부 헬퍼 ----

    async def _drain_and_switch(self, name: str) -> None:
        active = await asyncio.to_thread(self._active)
        if active != name:
            await self._idle.wait()  # 드레인: 사용 중 요청이 빌 때까지 전환 보류
        await asyncio.to_thread(self._switch, name)

    async def _wait_ready(self, served: str) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._ready_timeout
        while loop.time() < deadline:
            if await self.is_ready(served):
                return True
            await asyncio.sleep(2.0)
        return False

    async def _acquire(self, name: str | None) -> str:
        async with self._switch_lock:
            if name is not None:
                if name not in self.models:
                    raise UnknownModelError(name)
                served = self.models[name]["served_name"]
                await self._drain_and_switch(name)
                if not await self._wait_ready(served):
                    raise ModelNotReadyError(name)
            else:
                active = await asyncio.to_thread(self._active)
                if active is None:
                    raise NoActiveModelError()
                served = self.models[active]["served_name"]
            # 등록을 락 '안'에서: 등록 전에 다른 전환이 끼어들 틈을 없앤다
            self._inflight += 1
            self._idle.clear()
            return served

    def _release(self) -> None:
        self._inflight -= 1
        if self._inflight == 0:
            self._idle.set()