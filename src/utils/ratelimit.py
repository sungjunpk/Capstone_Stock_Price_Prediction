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
    capacity: 버킷 최대 크기 (버스트 허용량). 기본은 rate와 동일.
    """

    def __init__(self, rate: float, capacity: float | None = None):
        if rate <= 0:
            raise ValueError("rate는 0보다 커야 한다")
        self.rate = float(rate)
        self.capacity = float(capacity if capacity is not None else rate)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

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


class RateLimiterRegistry:
    """전역 리미터 + 키(TR ID)별 리미터를 함께 관리."""

    def __init__(self, global_rate: float):
        self._global = TokenBucket(global_rate)
        self._per_key: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def set_limit(self, key: str, rate: float) -> None:
        with self._lock:
            self._per_key[key] = TokenBucket(rate)

    def acquire(self, key: str | None = None) -> None:
        if key is not None:
            with self._lock:
                bucket = self._per_key.get(key)
            if bucket is not None:
                bucket.acquire()
        self._global.acquire()
