import time

from src.utils.ratelimit import TokenBucket


def test_bucket_throttles_beyond_capacity():
    bucket = TokenBucket(rate=10.0, capacity=2.0)
    start = time.monotonic()
    for _ in range(5):          # 2개는 즉시, 나머지 3개는 0.1s 씩
        bucket.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.25, f"레이트 리밋이 동작하지 않았다 (elapsed={elapsed:.3f}s)"


def test_burst_within_capacity_is_immediate():
    bucket = TokenBucket(rate=1.0, capacity=3.0)
    start = time.monotonic()
    for _ in range(3):
        bucket.acquire()
    assert time.monotonic() - start < 0.05


def test_penalize_halves_rate_and_recovers():
    """429 대응: 속도를 줄이고, 연속 성공하면 원래대로 회복한다."""
    from src.utils.ratelimit import RateLimiterRegistry

    reg = RateLimiterRegistry(global_rate=10.0)
    reg.set_limit("ka10081", 4.0)
    assert reg.current_rate("ka10081") == 4.0

    assert reg.penalize("ka10081") == 2.0
    assert reg.penalize("ka10081") == 1.0

    for _ in range(200):
        reg.report_success("ka10081")
    assert reg.current_rate("ka10081") == 4.0, "원래 속도로 회복해야 한다"


def test_penalize_has_floor():
    """무한정 느려지지는 않는다."""
    from src.utils.ratelimit import RateLimiterRegistry

    reg = RateLimiterRegistry(global_rate=10.0)
    reg.set_limit("x", 4.0)
    for _ in range(50):
        reg.penalize("x")
    assert reg.current_rate("x") >= 0.2


def test_unknown_key_falls_back_to_global_rate():
    from src.utils.ratelimit import RateLimiterRegistry

    reg = RateLimiterRegistry(global_rate=3.0)
    assert reg.current_rate("처음보는TR") == 3.0


def test_sub_one_rate_can_still_serve_one_request():
    """rate < 1 (예: 2초에 1회)이어도 요청 1건은 통과해야 한다.

    용량을 rate 와 같게 두면 acquire(1.0) 이 '용량 초과'로 죽는다 — 실제로 겪은 버그.
    """
    bucket = TokenBucket(rate=0.5)
    assert bucket.capacity >= 1.0
    bucket.acquire()  # 예외 없이 통과해야 한다
