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
