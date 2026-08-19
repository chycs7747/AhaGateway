import asyncio

import docker
from docker.errors import NotFound


class VllmManager:
    """vLLM 도커 컨테이너의 시작/정지/상태를 담당하는 서비스 계층.

    docker SDK는 동기(sync)라서 그대로 부르면 이벤트 루프가 멈춘다.
    그래서 동기 구현(_로 시작)과 async 공개 메서드를 분리하고,
    공개 메서드는 asyncio.to_thread로 동기 구현을 별도 스레드에서 돌린다.
    """

    def __init__(self, container_name: str):
        self.container_name = container_name
        self._docker = docker.from_env()

    # ---- 동기 구현부 ----

    def _status(self) -> str:
        try:
            container = self._docker.containers.get(self.container_name)
        except NotFound:
            return "not_found"
        return container.status  # "running", "exited", ...

    def _start(self) -> str:
        container = self._docker.containers.get(self.container_name)
        container.start()
        container.reload()  # start() 후 최신 상태를 다시 읽어온다
        return container.status

    def _stop(self) -> str:
        container = self._docker.containers.get(self.container_name)
        container.stop()
        container.reload()  # stop() 후 최신 상태를 다시 읽어온다
        return container.status

    # ---- async 공개 API (라우터가 쓰는 쪽) ----

    async def status(self) -> str:
        return await asyncio.to_thread(self._status)

    async def start(self) -> str:
        return await asyncio.to_thread(self._start)

    async def stop(self) -> str:
        return await asyncio.to_thread(self._stop)