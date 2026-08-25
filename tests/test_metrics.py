"""예측력 진단 지표 — 정답을 아는 합성데이터로 검증한다.

실제 예측에 이 함수를 들이대기 전에, 완전상관/무상관에서 기대값이 나오는지 먼저 본다.
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import decile_spread, rank_ic


def _frame(n_dates: int, n_names: int, rho: float, seed: int = 0) -> pd.DataFrame:
    """q50 과 target 의 순위 상관이 대략 rho 인 패널을 만든다."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2024-01-01", periods=n_dates):
        q = rng.normal(size=n_names)
        noise = rng.normal(size=n_names)
        t = rho * q + np.sqrt(max(1 - rho**2, 0.0)) * noise
        for i in range(n_names):
            rows.append({"date": d, "code": f"{i:03d}", "q50": q[i], "target": t[i]})
    return pd.DataFrame(rows)


def test_perfect_correlation_gives_ic_one():
    df = _frame(30, 50, rho=1.0)
    out = rank_ic(df)
    assert out["ic_mean"] > 0.99
    assert out["ic_positive_rate"] == 1.0


def test_perfect_anticorrelation_gives_ic_minus_one():
    df = _frame(30, 50, rho=1.0)
    df["q50"] = -df["q50"]
    assert rank_ic(df)["ic_mean"] < -0.99


def test_no_relationship_gives_ic_near_zero():
    df = _frame(60, 50, rho=0.0, seed=7)
    out = rank_ic(df)
    assert abs(out["ic_mean"]) < 0.05
    assert abs(out["t_stat"]) < 2.0          # 유의하지 않다고 나와야 한다


def test_weak_but_real_signal_is_detected():
    """IC 0.05 수준의 약한 신호도 날짜가 충분하면 t 값으로 잡힌다."""
    df = _frame(400, 100, rho=0.1, seed=3)
    out = rank_ic(df)
    assert out["ic_mean"] > 0.03
    assert out["t_stat"] > 2.0


def test_ic_aggregates_per_date_not_pooled():
    """날짜별 집계라 n_dates 는 날짜 수와 같아야 한다 (표본 부풀리기 방지)."""
    df = _frame(25, 40, rho=0.5)
    assert rank_ic(df)["n_dates"] == 25


def test_dates_with_too_few_names_are_skipped():
    df = _frame(10, 5, rho=0.5)          # 종목 5개 < min_names 10
    assert rank_ic(df)["n_dates"] == 0


def test_decile_spread_sign_matches_ic_sign():
    pos = _frame(50, 60, rho=0.6, seed=1)
    neg = pos.copy()
    neg["q50"] = -neg["q50"]

    assert decile_spread(pos)["spread_mean"] > 0
    assert decile_spread(neg)["spread_mean"] < 0
    assert rank_ic(pos)["ic_mean"] > 0 > rank_ic(neg)["ic_mean"]


def test_decile_spread_zero_when_no_signal():
    df = _frame(80, 60, rho=0.0, seed=11)
    assert abs(decile_spread(df)["t_stat"]) < 2.0


def test_nan_predictions_do_not_crash():
    df = _frame(20, 40, rho=0.5)
    df.loc[df.index[:100], "q50"] = np.nan
    out = rank_ic(df)
    assert out["n_dates"] > 0 and not np.isnan(out["ic_mean"])
