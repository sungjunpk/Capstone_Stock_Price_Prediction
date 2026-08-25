"""리스크 조정 성과 지표.

CLAUDE.md 원칙: **방향 정확도로 평가하지 않는다.** Sharpe/Sortino/Calmar 중심.
방향 정확도는 보조 지표로만 인용한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """일별 수익률 → 누적 자산 곡선."""
    return initial * (1.0 + returns.fillna(0.0)).cumprod()


def cagr(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    total = float((1.0 + returns.fillna(0.0)).prod())
    years = len(returns) / TRADING_DAYS
    if years <= 0 or total <= 0:
        return 0.0
    return total ** (1.0 / years) - 1.0


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """연율화 Sharpe. 변동성이 0이면 정의되지 않으므로 0을 반환."""
    r = returns.dropna() - rf / TRADING_DAYS
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0 or np.isnan(sd):
        return 0.0
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    """하방 변동성만으로 나눈다 — 상방 변동은 위험이 아니라는 관점."""
    r = returns.dropna() - rf / TRADING_DAYS
    downside = r[r < 0]
    dd = downside.std(ddof=1)
    if len(r) < 2 or len(downside) < 2 or dd == 0 or np.isnan(dd):
        return 0.0
    return float(r.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    """최대 낙폭(음수). -0.2 면 고점 대비 20% 하락."""
    if len(returns) == 0:
        return 0.0
    eq = equity_curve(returns)
    return float((eq / eq.cummax() - 1.0).min())


def calmar(returns: pd.Series) -> float:
    """CAGR / |최대낙폭|. 낙폭 대비 수익 효율."""
    mdd = abs(max_drawdown(returns))
    if mdd < 1e-9:
        return 0.0
    return float(cagr(returns) / mdd)


def hit_rate(returns: pd.Series) -> float:
    """양수 수익 비율. **보조 지표다** — 이것만 보고 판단하지 말 것."""
    r = returns.dropna()
    r = r[r != 0]
    return float((r > 0).mean()) if len(r) else 0.0


def volatility(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0


def summarize(returns: pd.Series, extra: dict | None = None) -> dict:
    """리포트용 전체 지표 묶음."""
    r = returns.fillna(0.0)
    out = {
        "n_days": int(len(r)),
        "cagr": round(cagr(r), 5),
        "volatility": round(volatility(r), 5),
        "sharpe": round(sharpe(r), 4),
        "sortino": round(sortino(r), 4),
        "calmar": round(calmar(r), 4),
        "max_drawdown": round(max_drawdown(r), 5),
        "hit_rate": round(hit_rate(r), 4),
        "total_return": round(float((1 + r).prod() - 1), 5),
    }
    if extra:
        out.update(extra)
    return out
