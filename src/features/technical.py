"""기술적 지표 계산. API 의존성 없음 — OHLCV 만 있으면 된다.

look-ahead 금지 원칙:
  모든 지표는 t 시점까지의 정보만 사용한다(rolling/ewm 전부 과거 방향).
  center=True 나 shift(-n) 은 이 파일에 등장해서는 안 된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------- 데이터 정제
def drop_halted_days(df: pd.DataFrame) -> pd.DataFrame:
    """거래정지일 제거.

    실측(2018-05 삼성전자 액면분할 구간): 정지일은 volume=0 이고
    open=high=low=close 로 직전 종가가 그대로 채워져 온다.
    이걸 남겨두면 가짜 0 수익률 → 변동성 과소추정 → ATR/RSI 왜곡으로 번진다.

    거래대금(value)이 있는데 volume 만 0 인 경우는 정지가 아닐 수 있어 남긴다.
    """
    flat = (
        (df["volume"] == 0)
        & (df["open"] == df["close"])
        & (df["high"] == df["low"])
        & (df["high"] == df["close"])
    )
    if "value" in df.columns:
        flat &= df["value"].fillna(0) == 0
    return df.loc[~flat].reset_index(drop=True)


# --------------------------------------------------------------- 기본 변환
def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """로그수익률 — 비정상성 제거의 기본. 가격 대신 이걸 모델 입력으로 쓴다."""
    return np.log(close / close.shift(periods))


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """t+1 ~ t+horizon 누적 로그수익률 = **예측 타깃**.

    피처가 아니라 라벨이다. 절대 입력 피처로 넣지 말 것.
    """
    return np.log(close.shift(-horizon) / close)


# --------------------------------------------------------------- 추세
def moving_averages(close: pd.Series, windows=(5, 20, 60)) -> pd.DataFrame:
    """MA 자체는 가격 스케일이라 비정상 — 종가 대비 비율(괴리율)로 낸다."""
    out = {}
    for w in windows:
        ma = close.rolling(w, min_periods=w).mean()
        out[f"ma{w}_ratio"] = close / ma - 1.0
    return pd.DataFrame(out, index=close.index)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    # 가격 스케일 제거를 위해 종가로 정규화
    return pd.DataFrame(
        {
            "macd": line / close,
            "macd_signal": sig / close,
            "macd_hist": (line - sig) / close,
        },
        index=close.index,
    )


# --------------------------------------------------------------- 모멘텀
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI (0~100)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss==0 (연속 상승) 이면 RSI=100
    return out.where(avg_loss.ne(0.0) | avg_gain.isna(), 100.0)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (tp - sma) / (0.015 * mad.replace(0.0, np.nan))


# --------------------------------------------------------------- 변동성
def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    upper, lower = ma + num_std * sd, ma - num_std * sd
    width = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_pctb": (close - lower) / width,     # 밴드 내 위치 0~1
            "bb_width": (upper - lower) / ma,       # 밴드 폭(변동성 대용)
        },
        index=close.index,
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR 을 종가로 나눠 스케일 제거(ATR%)."""
    tr = true_range(high, low, close)
    a = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return a / close


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    return log_return(close).rolling(window, min_periods=window).std(ddof=0)


# --------------------------------------------------------------- 거래량
def volume_features(volume: pd.Series, window: int = 20) -> pd.DataFrame:
    ma = volume.rolling(window, min_periods=window).mean()
    return pd.DataFrame(
        {
            "vol_ratio": volume / ma.replace(0.0, np.nan),
            "vol_zscore": (volume - ma)
            / volume.rolling(window, min_periods=window).std(ddof=0).replace(0.0, np.nan),
        },
        index=volume.index,
    )


# --------------------------------------------------------------- 조립
def add_technical_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """한 종목의 일봉 DataFrame 에 기술적 지표를 붙여 반환.

    입력은 date 오름차순 정렬된 단일 종목 데이터여야 한다.
    필수 컬럼: open/high/low/close/volume
    """
    cfg = cfg or {}
    required = {"high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")
    if "date" in df.columns and not df["date"].is_monotonic_increasing:
        raise ValueError("date 오름차순 정렬 후 호출할 것 (지표가 뒤섞인다)")

    out = df.copy()
    close, high, low, vol = out["close"], out["high"], out["low"], out["volume"]

    out["ret_1d"] = log_return(close, 1)
    out["ret_5d"] = log_return(close, 5)
    out["ret_20d"] = log_return(close, 20)

    macd_cfg = cfg.get("macd", {})
    bb_cfg = cfg.get("bollinger", {})

    out = out.join(moving_averages(close, cfg.get("ma_windows", (5, 20, 60))))
    out = out.join(
        macd(close, macd_cfg.get("fast", 12), macd_cfg.get("slow", 26),
             macd_cfg.get("signal", 9))
    )
    out["rsi"] = rsi(close, cfg.get("rsi_period", 14)) / 100.0  # 0~1 스케일
    out["cci"] = cci(high, low, close, cfg.get("cci_period", 20)) / 100.0
    out = out.join(bollinger(close, bb_cfg.get("window", 20), bb_cfg.get("num_std", 2.0)))
    out["atr"] = atr(high, low, close, cfg.get("atr_period", 14))
    out["rvol_20"] = realized_vol(close, 20)
    out = out.join(volume_features(vol))

    return out
