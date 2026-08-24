"""증분 저장 idempotency — 같은 명령 두 번 = 중복 0행."""

import pandas as pd

from src.data.storage import last_date, upsert


def test_upsert_is_idempotent(tmp_path):
    path = tmp_path / "005930.parquet"
    df = pd.DataFrame(
        {"date": pd.bdate_range("2024-01-01", periods=10).date, "close": range(10)}
    )
    first = upsert(df, path, key=["date"])
    second = upsert(df, path, key=["date"])
    assert len(first) == len(second) == 10




def test_upsert_merges_and_prefers_new(tmp_path):
    path = tmp_path / "x.parquet"
    old = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=5).date, "close": 1.0})
    upsert(old, path, key=["date"])

    # 겹치는 날짜 + 새 날짜. 겹친 구간은 새 값(수정주가 소급)이 이겨야 한다.
    new = pd.DataFrame({"date": pd.bdate_range("2024-01-03", periods=5).date, "close": 2.0})
    merged = upsert(new, path, key=["date"])

    assert len(merged) == 7   # 영업일 1,2,3,4,5,8,9 합집합
    assert merged.sort_values("date").iloc[-1]["close"] == 2.0
    assert merged.sort_values("date").iloc[0]["close"] == 1.0
    assert last_date(path) == max(new["date"])
