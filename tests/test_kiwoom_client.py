"""클라이언트 재시도 — 특히 '토큰이 죽은 채로 계속 가는' 경우.

배경: 키움은 appkey 당 토큰 하나만 살려둔다. 수집이 도는 중에 모의투자나
대시보드가 토큰을 새로 받으면 **수집이 들고 있던 토큰이 서버에서 무효화**된다.
클라이언트의 만료 시각은 아직 미래라 스스로 눈치채지 못한다 — 응답을 보고 알아야 한다.
"""

from __future__ import annotations

import time

import pytest

from src.data.kiwoom.client import KiwoomAPIError, KiwoomClient
from src.data.kiwoom.endpoints import DAILY_CHART, TOKEN_PATH
from src.utils.config import KiwoomSettings


class _Resp:
    def __init__(self, body, status=200, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = str(body)

    def json(self):
        return self._body


class _FakeSession:
    """토큰 발급과 TR 호출을 구분해 받아 적는 가짜 세션."""

    def __init__(self, tr_responses):
        self.tr_responses = list(tr_responses)
        self.token_calls = 0
        self.sent_auth: list[str] = []

    def post(self, url, json=None, headers=None, timeout=None):
        if url.endswith(TOKEN_PATH):
            self.token_calls += 1
            return _Resp({"token": f"tok{self.token_calls}", "expires_in": 86400})
        self.sent_auth.append((headers or {}).get("authorization", ""))
        return self.tr_responses.pop(0)

    def close(self):
        pass


_TOKEN_DEAD = {"return_code": 3,
               "return_msg": "인증에 실패했습니다[8005:Token이 유효하지 않습니다]"}
_OK = {"return_code": 0, "stk_dt_pole_chart_qry": []}


def _client(tr_responses):
    settings = KiwoomSettings(
        env="mock", base_url="https://mockapi.test",
        app_key="k", app_secret="s", account_no="1", rate_limit_per_sec=1000.0,
    )
    c = KiwoomClient(settings)
    c._session = _FakeSession(tr_responses)
    return c


def test_dead_token_is_reissued_and_the_request_retried():
    """8005 를 맞은 요청은 버려지지 않는다 — 토큰을 다시 받고 그 요청을 재시도한다."""
    c = _client([_Resp(_TOKEN_DEAD), _Resp(_OK)])

    data, _ = c.request(DAILY_CHART, {"stk_cd": "005930"})

    assert data == _OK
    assert c._session.token_calls == 2, "죽은 토큰을 버리고 새로 받아야 한다"
    assert c._session.sent_auth == ["Bearer tok1", "Bearer tok2"], \
        "재시도는 **새 토큰**으로 나가야 한다"


def test_dead_token_retry_does_not_loop_forever():
    """계속 8005 면 결국 포기한다 — 무한 재발급 금지."""
    c = _client([_Resp(_TOKEN_DEAD) for _ in range(10)])

    with pytest.raises(KiwoomAPIError, match="8005"):
        c.request(DAILY_CHART, {"stk_cd": "005930"})

    assert c._session.token_calls <= 4, f"재발급이 과하다: {c._session.token_calls}회"


def test_valid_token_is_not_refetched():
    """멀쩡한 토큰을 매 요청마다 다시 받지 않는다."""
    c = _client([_Resp(_OK), _Resp(_OK)])
    c.request(DAILY_CHART, {})
    c.request(DAILY_CHART, {})
    assert c._session.token_calls == 1


def test_expired_token_is_refetched_before_the_request():
    """만료 시각이 지났으면 요청 전에 미리 갱신한다 (기존 동작 회귀 방지)."""
    c = _client([_Resp(_OK)])
    c._token, c._token_expires_at = "stale", time.time() - 1
    c.request(DAILY_CHART, {})
    assert c._session.sent_auth == ["Bearer tok1"]
