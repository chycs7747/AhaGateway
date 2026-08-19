import asyncio
import uuid

import docker
import httpx
from docker.errors import NotFound


class UnknownModelError(Exception):
    """레지스트리에 등록되지 않은 모델 이름."""


class NoActiveModelError(Exception):
    """요청이 모델을 지정하지 않았는데 떠 있는 모델도 없음."""


class ModelNotReadyError(Exception):
    """전환은 했지만 준비 시간(MODEL_READY_TIMEOUT) 안에 ready가 되지 않음."""


class DrainTimeoutError(Exception):
    """진행 중 세션이 드레인 시간(SWITCH_DRAIN_TIMEOUT) 안에 비워지지 않아 전환/언로드 불가."""


class VllmManager:
    """등록된 vLLM 모델 컨테이너들의 수명과 사용을 관리하는 서비스 계층.

    - ensure(name): 그 모델이 서빙 중임을 보장 (생성/전환/멱등)
    - acquire()/release(): 추론 세션(리스) 발급과 반납. 진행 중 세션이
      있는 동안 전환은 드레인으로 보류되어, 남의 추론을 죽이지 않는다.
      반납이 누락된 세션은 TTL이 지나면 자동 만료된다.

    docker SDK는 동기라서 동기 구현(_로 시작)을 asyncio.to_thread로 돌린다.
    """

    def __init__(self, models: dict[str, dict], http_client: httpx.AsyncClient,
                 ready_timeout: float = 600.0,      # [로딩] 새 모델의 GPU 로딩 대기 상한
                 drain_timeout: float = 180.0,      # [드레인] 남의 추론이 끝나길 기다리는 상한
                 session_ttl: float = 1800.0):      # [유령 세션] 미반납 세션의 자동 만료
        self.models = models
        self._docker = docker.from_env()
        self._http = http_client
        self._ready_timeout = ready_timeout
        self._drain_timeout = drain_timeout
        self._session_ttl = session_ttl
        self._switch_lock = asyncio.Lock()       # 전환 작업의 직렬화
        self._sessions: dict[str, float] = {}    # session_id -> 만료 시각 (loop.time 기준)
        self._idle = asyncio.Event()             # 세션 0개일 때 set
        self._idle.set()                         # 서버 기동 시점엔 진행 중 세션이 없다

    # ---- 동기 구현부 ----

    def _get_container(self, name: str):
        return self._docker.containers.get(self.models[name]["container"])

    def _status(self, name: str) -> str:
        try:
            return self._get_container(name).status  # "running", "exited", "created", ...
        except NotFound:
            return "not_created"

    def _active(self) -> str | None:
        """등록 모델 중 지금 running인 것의 이름 (없으면 None)."""
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
            await self._wait_idle()
            return await asyncio.to_thread(self._unload)

    async def is_ready(self, served_name: str | None = None) -> bool:
        """vLLM이 응답하는지, (지정 시) '그 모델을' 서빙 중인지 확인."""
        try:
            resp = await self._http.get("/models", timeout=2.0)
        except httpx.HTTPError:
            return False
        if resp.status_code != 200:
            return False
        if served_name is None:  # 여기 도달 = 위 관문(응답+200)은 통과
            return True
        ids = [m.get("id") for m in resp.json().get("data", [])]
        return served_name in ids

    async def ensure(self, name: str, wait_ready: bool = True) -> bool:
        """name 모델이 서빙 중이도록 보장 (관리 API용)."""
        if name not in self.models:
            raise UnknownModelError(name)
        served = self.models[name]["served_name"]
        async with self._switch_lock:
            await self._drain_and_switch(name)
            if not wait_ready:
                return await self.is_ready(served)
            return await self._wait_ready(served)

    # ---- 세션 (리스): 발급/반납/만료 ----

    async def acquire(self, name: str | None) -> tuple[str, str]:
        """추론 세션 발급. (session_id, served_name) 반환.

        - name 지정: 필요시 전환(드레인 포함)하고 ready까지 대기
        - name 없음: 지금 떠 있는 모델 사용 (없으면 NoActiveModelError)
        발급은 락 '안'에서 등록까지 마쳐, 등록 전에 다른 전환이 끼어들 틈을 없앤다.
        """
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
            session_id = uuid.uuid4().hex
            loop = asyncio.get_running_loop()
            self._sessions[session_id] = loop.time() + self._session_ttl
            self._idle.clear()
            return session_id, served

    def release(self, session_id: str) -> bool:
        """세션 반납. 마지막 세션이면 '비었음' 신호를 켜 대기 중인 전환을 깨운다."""
        existed = self._sessions.pop(session_id, None) is not None
        if not self._sessions:
            self._idle.set()
        return existed

    # ---- 내부 헬퍼 ----

    def _purge_expired(self) -> None:
        """TTL이 지난 유령 세션을 장부에서 지운다."""
        now = asyncio.get_running_loop().time()
        for sid in [s for s, exp in self._sessions.items() if exp <= now]:
            del self._sessions[sid]
        if not self._sessions:
            self._idle.set()

    async def _wait_idle(self) -> None:
        """세션이 0개가 될 때까지 대기 (상한: drain_timeout).

        5초마다 깨어나 만료 세션을 청소하고, 상한 초과 시 진행 중
        요청을 죽이는 대신 DrainTimeoutError를 던진다 (전환 쪽이 양보).
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._drain_timeout
        while True:
            self._purge_expired()
            if not self._sessions:
                return
            if loop.time() >= deadline:
                raise DrainTimeoutError(
                    f"진행 중인 세션 {len(self._sessions)}개가 제한 시간 안에 끝나지 않음"
                )
            try:
                await asyncio.wait_for(self._idle.wait(),
                                       timeout=min(5.0, deadline - loop.time()))
            except TimeoutError:
                continue  # 주기적으로 깨어나 만료 청소 후 재확인

    async def _drain_and_switch(self, name: str) -> None:
        active = await asyncio.to_thread(self._active)
        if active != name:
            await self._wait_idle()  # 드레인: 사용 중 요청이 빌 때까지 전환 보류
        await asyncio.to_thread(self._switch, name)

    async def _wait_ready(self, served: str) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._ready_timeout
        while loop.time() < deadline:
            if await self.is_ready(served):
                return True
            await asyncio.sleep(2.0)
        return False
