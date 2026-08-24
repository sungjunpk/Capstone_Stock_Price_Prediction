"""토큰 버킷 레이트 리미터.

키움은 TR별 초당 호출 제한이 있고, 초과하면 일시 차단된다.
전역 리미터 하나 + 필요하면 TR별 리미터를 추가로 두는 구조.
"""

from __future__ import annotations

import threading
import time


class TokenBucket:
    """스레드 세이프 토큰 버킷.

    rate: 초당 보충되는 토큰 수 (= 허용 초당 요청 수)
    capacity: 버킷 최대 크기 (버스트 허용량). 기본은 rate와 동일하되 **최소 1**.
        rate가 1 미만(예: 0.5 = 2초에 1회)이어도 요청 1건은 담을 수 있어야 한다.
    """

    def __init__(self, rate: float, capacity: float | None = None):
        if rate <= 0:
            raise ValueError("rate는 0보다 커야 한다")
        self.rate = float(rate)
        self.capacity = float(capacity if capacity is not None else max(rate, 1.0))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def set_rate(self, rate: float) -> None:
        """실행 중 속도 조정. 429 대응용."""
        if rate <= 0:
            raise ValueError("rate는 0보다 커야 한다")
        with self._lock:
            self.rate = float(rate)
            self.capacity = max(self.capacity, 1.0)
            self._tokens = min(self._tokens, self.capacity)

    def acquire(self, tokens: float = 1.0) -> float:
        """토큰이 찰 때까지 블로킹. 실제로 대기한 초를 반환."""
        if tokens > self.capacity:
            raise ValueError("요청 토큰이 버킷 용량보다 크다")

        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                deficit = tokens - self._tokens
                sleep_for = deficit / self.rate
            time.sleep(sleep_for)
            waited += sleep_for


# 429 를 맞으면 이 비율로 줄이고, 연속 성공하면 이 비율로 되돌린다.
_PENALTY_FACTOR = 0.5
_RECOVER_FACTOR = 1.15
_MIN_RATE = 0.2          # 초당 0.2회(5초에 1회)보다 느려지지는 않는다
_RECOVER_AFTER = 20      # 연속 성공 N회마다 한 단계 회복


class RateLimiterRegistry:
    """전역 리미터 + 키(TR ID)별 리미터.

    키움은 TR별로 제한이 다르고 문서화도 불완전하다. 그래서 고정값에 의존하지 않고
    **429 를 맞으면 해당 TR 속도를 절반으로 줄이고**, 이후 연속 성공하면 조금씩 회복한다.
    긴 수집(수백 종목)에서 중간에 막혀 몇 시간을 날리지 않으려면 이게 필요하다.
    """

    def __init__(self, global_rate: float):
        self._global = TokenBucket(global_rate)
        self._initial_global = float(global_rate)
        self._per_key: dict[str, TokenBucket] = {}
        self._base_rate: dict[str, float] = {}
        self._ok_streak: dict[str, int] = {}
        self._lock = threading.Lock()

    def set_limit(self, key: str, rate: float) -> None:
        with self._lock:
            self._per_key[key] = TokenBucket(rate)
            self._base_rate[key] = float(rate)
            self._ok_streak[key] = 0

    def _bucket(self, key: str) -> TokenBucket:
        """해당 키의 버킷을 반환. 없으면 전역 속도로 만들어 둔다."""
        with self._lock:
            b = self._per_key.get(key)
            if b is None:
                b = TokenBucket(self._initial_global)
                self._per_key[key] = b
                self._base_rate[key] = self._initial_global
                self._ok_streak[key] = 0
            return b

    def acquire(self, key: str | None = None) -> None:
        if key is not None:
            self._bucket(key).acquire()
        self._global.acquire()

    def penalize(self, key: str) -> float:
        """429 를 맞았을 때 호출. 새 속도를 반환."""
        b = self._bucket(key)
        new_rate = max(b.rate * _PENALTY_FACTOR, _MIN_RATE)
        b.set_rate(new_rate)
        with self._lock:
            self._ok_streak[key] = 0
        return new_rate

    def report_success(self, key: str) -> None:
        """정상 응답마다 호출. 연속 성공이 쌓이면 조금씩 속도를 되돌린다."""
        b = self._bucket(key)
        with self._lock:
            self._ok_streak[key] = self._ok_streak.get(key, 0) + 1
            streak = self._ok_streak[key]
            base = self._base_rate.get(key, self._initial_global)
        if streak >= _RECOVER_AFTER and b.rate < base:
            b.set_rate(min(b.rate * _RECOVER_FACTOR, base))
            with self._lock:
                self._ok_streak[key] = 0

    def current_rate(self, key: str) -> float:
        return self._bucket(key).rate
