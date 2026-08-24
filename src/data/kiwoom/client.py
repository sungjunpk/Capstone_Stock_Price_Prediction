"""키움 REST API 저수준 클라이언트.

책임은 딱 네 가지:
  1) 토큰 발급/갱신
  2) 레이트 리밋 준수 (TokenBucket)
  3) 재시도/백오프 (429·5xx)
  4) 연속조회(cont-yn / next-key) 페이지네이션

응답 파싱은 하지 않는다 — 그건 src.utils.parsing 담당.
⚠️ 헤더/토큰 필드명은 MCP(`kiwoom-rest-api`)로 실제 응답 확인 전까지 UNVERIFIED.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import requests

from src.data.kiwoom.endpoints import TOKEN_PATH, TRSpec
from src.utils.config import KiwoomSettings, load_kiwoom_settings
from src.utils.logging import get_logger
from src.utils.ratelimit import RateLimiterRegistry

log = get_logger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4
_TOKEN_MARGIN_SEC = 300  # 만료 5분 전 미리 갱신


class KiwoomAPIError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


class KiwoomClient:
    """모의투자 전용 클라이언트. live 는 config.py 단에서 차단된다."""

    def __init__(
        self,
        settings: KiwoomSettings | None = None,
        *,
        timeout: float = 10.0,
        max_pages: int = 100,
    ):
        self.settings = settings or load_kiwoom_settings()
        self.settings.require_credentials()
        self.timeout = timeout
        self.max_pages = max_pages

        self.limiter = RateLimiterRegistry(self.settings.rate_limit_per_sec)
        self._session = requests.Session()
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._token_lock = threading.Lock()

    # ------------------------------------------------------------ 토큰
    def _fetch_token(self) -> None:
        url = f"{self.settings.base_url}{TOKEN_PATH}"
        payload = {  # VERIFIED — mock 200 OK
            "grant_type": "client_credentials",
            "appkey": self.settings.app_key,
            "secretkey": self.settings.app_secret,
        }
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise KiwoomAPIError(
                f"토큰 발급 실패 ({resp.status_code}): {resp.text[:300]}",
                status=resp.status_code,
            )
        body = resp.json()
        token = body.get("token") or body.get("access_token")  # VERIFIED: "token"
        if not token:
            raise KiwoomAPIError(f"토큰 응답에 token 필드가 없다: {list(body)}", body=body)

        # VERIFIED: mock 응답은 expires_dt('20260825163737') 형식(≈24h).
        # expires_in(초)은 오지 않았지만 방어적으로 둘 다 받는다.
        ttl = 3600.0
        if body.get("expires_dt"):
            with contextlib.suppress(TypeError, ValueError):
                exp = datetime.strptime(str(body["expires_dt"]), "%Y%m%d%H%M%S")
                ttl = max((exp - datetime.now()).total_seconds(), 60.0)
        elif isinstance(body.get("expires_in"), (int, float, str)):
            with contextlib.suppress(TypeError, ValueError):
                ttl = float(body["expires_in"])
        self._token = token
        self._token_expires_at = time.time() + max(ttl - _TOKEN_MARGIN_SEC, 60.0)
        log.info("키움 토큰 발급 완료 (env=%s, ttl≈%.0fs)", self.settings.env, ttl)

    def _auth_header(self) -> str:
        with self._token_lock:
            if self._token is None or time.time() >= self._token_expires_at:
                self._fetch_token()
        return f"Bearer {self._token}"

    # ------------------------------------------------------------ 요청
    def request(
        self,
        spec: TRSpec,
        body: dict[str, Any] | None = None,
        *,
        cont_yn: str | None = None,
        next_key: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """TR 1회 호출. (응답 body, 응답 header) 반환."""
        url = f"{self.settings.base_url}{spec.path}"
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": self._auth_header(),
            "api-id": spec.api_id,  # VERIFIED
        }
        if cont_yn:
            headers["cont-yn"] = cont_yn
        if next_key:
            headers["next-key"] = next_key

        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            self.limiter.acquire(spec.api_id)
            try:
                resp = self._session.post(
                    url, json=body or {}, headers=headers, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_err = exc
                backoff = 2.0**attempt
                log.warning("[%s] 네트워크 오류 (%s) — %.1fs 후 재시도", spec.name, exc, backoff)
                time.sleep(backoff)
                continue

            if resp.status_code in _RETRY_STATUS:
                backoff = 2.0**attempt
                log.warning(
                    "[%s] HTTP %s — %.1fs 후 재시도 (%d/%d)",
                    spec.name, resp.status_code, backoff, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(backoff)
                last_err = KiwoomAPIError(
                    f"{spec.name}: HTTP {resp.status_code}", status=resp.status_code
                )
                continue

            if resp.status_code != 200:
                raise KiwoomAPIError(
                    f"{spec.name}: HTTP {resp.status_code} {resp.text[:300]}",
                    status=resp.status_code,
                )

            data = resp.json()
            # 키움은 HTTP 200 이어도 body 안 return_code 로 실패를 알린다. VERIFIED
            rc = data.get("return_code")
            if rc not in (None, 0, "0"):
                raise KiwoomAPIError(
                    f"{spec.name}: return_code={rc} msg={data.get('return_msg')}",
                    body=data,
                )
            return data, {k.lower(): v for k, v in resp.headers.items()}

        raise KiwoomAPIError(f"{spec.name}: 재시도 소진") from last_err

    def paginate(
        self,
        spec: TRSpec,
        body: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """연속조회를 끝까지 따라가며 페이지 body 를 순서대로 내보낸다."""
        cont_yn, next_key = None, None
        for page in range(self.max_pages):
            data, headers = self.request(spec, body, cont_yn=cont_yn, next_key=next_key)
            yield data

            cont_key, nk_key = spec.cont_keys
            cont_yn = headers.get(cont_key)
            next_key = headers.get(nk_key)
            if cont_yn != "Y" or not next_key:
                return
            log.debug("[%s] 연속조회 page=%d next_key=%s", spec.name, page + 2, next_key)
        log.warning("[%s] max_pages(%d) 도달 — 조기 종료", spec.name, self.max_pages)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> KiwoomClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
