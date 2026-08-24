"""지표 정확성 + look-ahead 검증."""

import numpy as np
import pandas as pd
import pytest

from src.features import technical as tech


@pytest.fixture
def ohlcv():
    rng = np.random.default_rng(0)
    n = 300
    close = pd.Series(10000 * np.exp(np.cumsum(rng.normal(0, 0.015, n))))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=n).date,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1e5, 1e6, n),
        }
    )


def test_rsi_bounds(ohlcv):
    r = tech.rsi(ohlcv["close"])
    valid = r.dropna()
    assert not valid.empty
    assert valid.between(0, 100).all()


def test_no_lookahead(ohlcv):
    """미래 데이터를 바꿔도 과거 지표값은 변하면 안 된다."""
    cutoff = 200
    full = tech.add_technical_features(ohlcv)

    tampered = ohlcv.copy()
    tampered.loc[cutoff:, "close"] *= 3.0   # 미래를 완전히 망가뜨린다
    tampered.loc[cutoff:, "high"] *= 3.0
    tampered.loc[cutoff:, "low"] *= 3.0
    partial = tech.add_technical_features(tampered)

    feature_cols = [c for c in full.columns if c not in ohlcv.columns]
    assert feature_cols
    pd.testing.assert_frame_equal(
        full.loc[: cutoff - 1, feature_cols],
        partial.loc[: cutoff - 1, feature_cols],
    )


def test_forward_return_is_label_not_feature(ohlcv):
    """타깃은 반드시 미래를 본다 — 피처 목록에 절대 섞이면 안 된다."""
    y = tech.forward_log_return(ohlcv["close"], 5)
    assert y.iloc[-5:].isna().all()          # 끝 5개는 라벨 없음
    feats = tech.add_technical_features(ohlcv)
    assert not any("forward" in c for c in feats.columns)


def test_requires_sorted_dates(ohlcv):
    with pytest.raises(ValueError, match="오름차순"):
        tech.add_technical_features(ohlcv.iloc[::-1].reset_index(drop=True))


def test_drop_halted_days():
    """거래정지일(volume=0, OHLC 동일)은 제거되어야 한다.

    실측 기반: 2018-05 삼성전자 액면분할 구간에서 이 패턴이 3일 관측됨.
    """
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=5).date,
            "open": [100.0, 110.0, 110.0, 110.0, 120.0],
            "high": [105.0, 115.0, 110.0, 110.0, 125.0],
            "low": [99.0, 108.0, 110.0, 110.0, 118.0],
            "close": [104.0, 110.0, 110.0, 110.0, 124.0],
            "volume": [1000, 2000, 0, 0, 3000],
            "value": [100, 200, 0, 0, 300],
        }
    )
    out = tech.drop_halted_days(df)
    assert len(out) == 3
    assert (out["volume"] > 0).all()


def test_halted_days_would_fake_zero_returns():
    """제거하지 않으면 가짜 0 수익률이 섞인다 — 제거의 근거."""
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4).date,
            "open": [100.0, 110.0, 110.0, 120.0],
            "high": [105.0, 115.0, 110.0, 125.0],
            "low": [99.0, 108.0, 110.0, 118.0],
            "close": [104.0, 110.0, 110.0, 124.0],
            "volume": [1000, 2000, 0, 3000],
            "value": [100, 200, 0, 300],
        }
    )
    assert (tech.log_return(df["close"]) == 0).any()          # 정지일이 0 수익률을 만든다
    cleaned = tech.drop_halted_days(df)
    assert not (tech.log_return(cleaned["close"]).dropna() == 0).any()
