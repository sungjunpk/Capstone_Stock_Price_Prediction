"""클라이언트 재시도 — 특히 '토큰이 죽은 채로 계속 가는' 경우.

배경: 키움은 appkey 당 토큰 하나만 살려둔다. 수집이 도는 중에 모의투자나
대시보드가 토큰을 새로 받으면 **수집이 들고 있던 토큰이 서버에서 무효화**된다.
클라이언트의 만료 시각은 아직 미래라 스스로 눈치채지 못한다 — 응답을 보고 알아야 한다.
"""

from __future__ import annotations

import time

import pytest

from src.data.kiwoom import client as client_mod
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

    def __init__(self, tr_responses, token_statuses=None):
        self.tr_responses = list(tr_responses)
        self.token_calls = 0
        self.sent_auth: list[str] = []
        # 발급 시도마다 돌려줄 HTTP 상태. 동나면 200.
        self.token_statuses = list(token_statuses or [])

    def post(self, url, json=None, headers=None, timeout=None):
        if url.endswith(TOKEN_PATH):
            self.token_calls += 1
            status = self.token_statuses.pop(0) if self.token_statuses else 200
            if status != 200:
                return _Resp(_TOKEN_RATE_LIMITED, status=status)
            return _Resp({"token": f"tok{self.token_calls}", "expires_in": 86400})
        self.sent_auth.append((headers or {}).get("authorization", ""))
        return self.tr_responses.pop(0)

    def close(self):
        pass


_TOKEN_DEAD = {"return_code": 3,
               "return_msg": "인증에 실패했습니다[8005:Token이 유효하지 않습니다]"}
# 2026-09-01 16:00:07 실측 응답
_TOKEN_RATE_LIMITED = {
    "return_code": 5,
    "return_msg": "허용된 요청 개수를 초과하였습니다[1700:허용된 API 요청 개수를 초과하였습니다. 유량=au10001, API ID={1}]",
}
_OK = {"return_code": 0, "stk_dt_pole_chart_qry": []}


def _client(tr_responses, token_statuses=None):
    settings = KiwoomSettings(
        env="mock", base_url="https://mockapi.test",
        app_key="k", app_secret="s", account_no="1", rate_limit_per_sec=1000.0,
    )
    c = KiwoomClient(settings)
    c._session = _FakeSession(tr_responses, token_statuses)
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


# ------------------------------------------------- 토큰 **발급** 단계의 429
class TestTokenIssuanceRetry:
    """발급 자체가 429 를 맞는 경우. TR 응답의 8005 와는 다른 경로다.

    배경(2026-09-01 실측): 16:00:06 에 계좌 스냅샷이 토큰을 받고, 1초 뒤 수집이
    또 받으려다 429 를 맞았다. 토큰 엔드포인트(au10001)에 자체 유량 제한이 있다.
    재시도가 없으면 **그 순간의 종목 하나가 조용히 유실된다** — 다음 종목은
    새로 발급받아 정상 진행하므로 "성공 148 / 실패 1" 로 끝난다.
    """

    @pytest.fixture(autouse=True)
    def _no_real_sleep(self, monkeypatch):
        """백오프를 실제로 기다리지 않는다. 대신 얼마나 기다렸는지 받아 적는다."""
        self.slept: list[float] = []
        monkeypatch.setattr(client_mod.time, "sleep", self.slept.append)

    def test_rate_limited_token_is_retried_and_the_request_succeeds(self):
        c = _client([_Resp(_OK)], token_statuses=[429])

        data, _ = c.request(DAILY_CHART, {"stk_cd": "005930"})

        assert data == _OK
        assert c._session.token_calls == 2, "429 를 맞으면 다시 받아야 한다"
        assert c._session.sent_auth == ["Bearer tok2"], "재발급된 토큰으로 나가야 한다"
        assert self.slept == [1.0], "재시도 전에 백오프를 둔다"

    def test_symbol_is_not_lost_to_a_token_429(self):
        """회귀 방지의 핵심 — 첫 종목이 유실되지 않는다.

        유니버스 첫 종목이 늘 발급 직후에 오므로, 재시도가 없으면 **매번 같은
        종목(005930)만** 빠진다. 시총 1위가 횡단면 순위에서 사라지는 셈이다.
        """
        c = _client([_Resp(_OK), _Resp(_OK)], token_statuses=[429])

        first, _ = c.request(DAILY_CHART, {"stk_cd": "005930"})   # 발급이 429 를 맞는 자리
        second, _ = c.request(DAILY_CHART, {"stk_cd": "000660"})

        assert first == _OK and second == _OK
        assert c._session.token_calls == 2, "두 번째 종목은 살아있는 토큰을 재사용한다"

    def test_persistent_rate_limit_gives_up(self):
        """계속 429 면 결국 포기한다 — 무한 재시도 금지."""
        c = _client([], token_statuses=[429] * 10)

        with pytest.raises(KiwoomAPIError, match="토큰 발급 실패"):
            c.request(DAILY_CHART, {})

        assert c._session.token_calls == 3, f"발급 시도가 과하다: {c._session.token_calls}회"
        assert self.slept == [1.0, 2.0], "지수 백오프"

    def test_auth_failure_is_not_retried(self):
        """401 은 기다린다고 낫지 않는다 — 자격증명 문제다."""
        c = _client([], token_statuses=[401])

        with pytest.raises(KiwoomAPIError, match="토큰 발급 실패"):
            c.request(DAILY_CHART, {})

        assert c._session.token_calls == 1, "재시도할 상태가 아니다"
        assert self.slept == []
