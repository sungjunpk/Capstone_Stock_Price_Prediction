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


def test_load_kind_handles_existing_code_column(tmp_path, monkeypatch):
    """응답에 code 컬럼이 이미 있는 TR(stock_info 등)도 로드돼야 한다."""
    import src.data.storage as st

    monkeypatch.setattr(st, "RAW_DIR", tmp_path)
    d = tmp_path / "stock_info"
    d.mkdir()
    # 파일명(005930)과 다른 값을 일부러 넣어 파일명이 이기는지 확인
    pd.DataFrame({"code": ["WRONG"], "per": [39.16]}).to_parquet(
        d / "005930.parquet", index=False
    )

    out = st.load_kind("stock_info")
    assert list(out.columns)[0] == "code"
    assert out.loc[0, "code"] == "005930"
    assert out.loc[0, "per"] == 39.16
