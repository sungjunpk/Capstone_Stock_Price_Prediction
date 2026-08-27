"""키움 문자열 파싱 — 여기가 틀리면 모든 데이터가 조용히 오염된다."""

from datetime import date

import pandas as pd

from src.utils.parsing import parse_records, to_date, to_float, to_int


def test_to_float_strips_sign_and_comma():
    assert to_float("+70,000") == 70000.0
    assert to_float("-1,234") == -1234.0
    assert to_float("1,234,567") == 1234567.0


def test_abs_value_for_price_fields():
    # 키움은 가격에도 등락 방향 부호를 붙인다 — 가격류는 절대값으로 받아야 한다
    assert to_float("-70000", abs_value=True) == 70000.0
    assert to_int("+1,500", abs_value=True) == 1500


def test_nullish_returns_none():
    for v in ("", "-", "--", None, "N/A"):
        assert to_float(v) is None


def test_to_date_formats():
    assert to_date("20240315") == date(2024, 3, 15)
    assert to_date("2024-03-15") == date(2024, 3, 15)
    assert to_date("2024031") is None


def test_parse_records_fills_missing_keys():
    schema = {
        "date": ("dt", "date"),
        "close": ("cur_prc", "abs_float"),
        "foreign": ("frgnr_invsr", "int"),
    }
    df = parse_records(
        [{"dt": "20240102", "cur_prc": "-70,000"}], schema  # frgnr_invsr 누락
    )
    assert list(df.columns) == ["date", "close", "foreign"]
    assert df.loc[0, "close"] == 70000.0
    assert pd.isna(df.loc[0, "foreign"])


def test_to_datetime_parses_minute_bar_timestamp():
    from datetime import datetime

    from src.utils.parsing import to_datetime

    # 분봉 TR(ka10080) 의 cntr_tm 은 'YYYYMMDDHHMMSS' 로 온다
    assert to_datetime("20260827114500") == datetime(2026, 8, 27, 11, 45)
    assert to_datetime("202608271145") == datetime(2026, 8, 27, 11, 45)
    for bad in ("", "-", None, "2026082711450", "abcdefghijklmn"):
        assert to_datetime(bad) is None


def test_parse_records_datetime_type():
    from src.data.kiwoom.endpoints import MINUTE_CHART

    rows = [{"cntr_tm": "20260827110000", "open_pric": "+267500",
             "high_pric": "+268000", "low_pric": "+267500",
             "cur_prc": "+268000", "trde_qty": "2,561"}]
    df = parse_records(rows, MINUTE_CHART.schema)
    assert pd.Timestamp(df.loc[0, "datetime"]) == pd.Timestamp("2026-08-27 11:00")
    # 가격에 붙은 등락 부호는 떨어져야 한다
    assert df.loc[0, "close"] == 268000.0
    assert df.loc[0, "volume"] == 2561
