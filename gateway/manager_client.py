import asyncio
from contextlib import asynccontextmanager

import httpx


class ManagerError(Exception):
    """manager가 요청을 거절함. HTTP 상태와 사유를 그대로 담아 전달."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")


class ManagerClient:
    """manager 서비스와의 HTTP 통신을 담당하는 클라이언트.

    모노리스 시절 VllmManager 객체를 직접 부르던 자리를 대체한다 —
    함수 호출이 HTTP 호출로 바뀌었을 뿐, session()이 컨텍스트 매니저라는
    사용법은 유지해서 라우터 코드의 모양이 거의 안 바뀌게 한다.
    """

    def __init__(self, http_client: httpx.AsyncClient):
        self._http = http_client  # base_url = MANAGER_BASE_URL

    @asynccontextmanager
    async def session(self, model: str | None, acquire_timeout: float):
        """세션 발급(진입) → 사용(본문) → 반납(종료, 실패해도 TTL이 수습)."""
        try:
            resp = await self._http.post(
                "/sessions", json={"model": model}, timeout=acquire_timeout
            )
        except httpx.HTTPError as exc:
            raise ManagerError(502, f"manager 연결 불가: {type(exc).__name__}")
        if resp.status_code != 200:
            raise ManagerError(resp.status_code, resp.json().get("detail", resp.text))
        data = resp.json()  # {"session_id", "served_name", "vllm_base_url"}
        try:
            yield data
        finally:
            # 클라이언트가 스트림 도중 끊기면 이 태스크는 '취소 중'이라, 여기서 그냥
            # await하면 DELETE 자체가 취소돼 세션이 샌다. shield로 반납을 끝까지 보낸다.
            release = asyncio.shield(
                self._http.delete(f"/sessions/{data['session_id']}", timeout=5.0)
            )
            try:
                await release
            except (httpx.HTTPError, asyncio.CancelledError):
                pass  # shield 덕에 DELETE는 백그라운드에서 완료됨; 실패해도 TTL이 최후 수습

    async def proxy(self, method: str, path: str, json: dict | None = None) -> httpx.Response:
        """관리 API(/models, /model/*)를 그대로 중계하기 위한 통로."""
        try:
            return await self._http.request(method, path, json=json)
        except httpx.HTTPError as exc:
            raise ManagerError(502, f"manager 연결 불가: {type(exc).__name__}")