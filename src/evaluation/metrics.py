"""리스크 조정 성과 지표.

CLAUDE.md 원칙: **방향 정확도로 평가하지 않는다.** Sharpe/Sortino/Calmar 중심.
방향 정확도는 보조 지표로만 인용한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
# 연율화 계수는 봉 단위마다 다르다. 일봉이면 252, 60분봉이면 252 x 7 = 1,764.
# 이걸 틀리면 Sharpe 가 sqrt(7) 배 어긋난 채 "일봉보다 좋다"는 결론이 나온다.


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """일별 수익률 → 누적 자산 곡선."""
    return initial * (1.0 + returns.fillna(0.0)).cumprod()


def cagr(returns: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    if len(returns) == 0:
        return 0.0
    total = float((1.0 + returns.fillna(0.0)).prod())
    years = len(returns) / periods_per_year
    if years <= 0 or total <= 0:
        return 0.0
    return total ** (1.0 / years) - 1.0


def sharpe(returns: pd.Series, rf: float = 0.0,
           periods_per_year: float = TRADING_DAYS) -> float:
    """연율화 Sharpe. 변동성이 0이면 정의되지 않으므로 0을 반환."""
    r = returns.dropna() - rf / periods_per_year
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0 or np.isnan(sd):
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, rf: float = 0.0,
            periods_per_year: float = TRADING_DAYS) -> float:
    """하방 변동성만으로 나눈다 — 상방 변동은 위험이 아니라는 관점."""
    r = returns.dropna() - rf / periods_per_year
    downside = r[r < 0]
    dd = downside.std(ddof=1)
    if len(r) < 2 or len(downside) < 2 or dd == 0 or np.isnan(dd):
        return 0.0
    return float(r.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """최대 낙폭(음수). -0.2 면 고점 대비 20% 하락."""
    if len(returns) == 0:
        return 0.0
    eq = equity_curve(returns)
    return float((eq / eq.cummax() - 1.0).min())


def calmar(returns: pd.Series,
           periods_per_year: float = TRADING_DAYS) -> float:
    """CAGR / |최대낙폭|. 낙폭 대비 수익 효율."""
    mdd = abs(max_drawdown(returns))
    if mdd < 1e-9:
        return 0.0
    return float(cagr(returns, periods_per_year) / mdd)


def hit_rate(returns: pd.Series) -> float:
    """양수 수익 비율. **보조 지표다** — 이것만 보고 판단하지 말 것."""
    r = returns.dropna()
    r = r[r != 0]
    return float((r > 0).mean()) if len(r) else 0.0


def volatility(returns: pd.Series,
               periods_per_year: float = TRADING_DAYS) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0


def summarize(returns: pd.Series, extra: dict | None = None,
              periods_per_year: float = TRADING_DAYS) -> dict:
    """리포트용 전체 지표 묶음.

    periods_per_year: 연율화 계수. 60분봉 백테스트는 여기에 봉/년 을 넣어야
        Sharpe 가 일봉과 같은 축 위에 놓인다.
    """
    r = returns.fillna(0.0)
    out = {
        "n_days": int(len(r)),
        "periods_per_year": periods_per_year,
        "cagr": round(cagr(r, periods_per_year), 5),
        "volatility": round(volatility(r, periods_per_year), 5),
        "sharpe": round(sharpe(r, periods_per_year=periods_per_year), 4),
        "sortino": round(sortino(r, periods_per_year=periods_per_year), 4),
        "calmar": round(calmar(r, periods_per_year), 4),
        "max_drawdown": round(max_drawdown(r), 5),
        "hit_rate": round(hit_rate(r), 4),
        "total_return": round(float((1 + r).prod() - 1), 5),
    }
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------- 예측력 진단
#
# 아래 두 함수는 **거래 로직을 통째로 우회해서** "모델이 방향을 맞히는가"만 직접 묻는다.
# 임계값·사이징·리스크 오버레이가 전부 빠지므로, 성과가 안 나올 때
# "모델이 못 맞히는 것"인지 "거래 규칙이 막는 것"인지 구분할 수 있다.
#
# ⚠️ 실현 수익률(target)을 쓰지만 **사후 평가 전용**이다.
#    매매 판단에는 절대 들어가지 않는다 — 들어가면 look-ahead 다.


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """순위상관. 순위로 바꾼 뒤의 Pearson 이 곧 Spearman 이다.

    한쪽이 전부 같은 값이면 순위 분산이 0 이라 상관계수가 정의되지 않는다.
    (구간 끝에서 forward return 이 전부 0 이 되는 날 등) numpy 경고를 띄우는 대신
    NaN 을 돌려주고, 호출부에서 그 날짜를 빼도록 한다.
    """
    if len(a) < 3:
        return float("nan")
    ra, rb = a.rank(), b.rank()
    if ra.std(ddof=0) < 1e-12 or rb.std(ddof=0) < 1e-12:
        return float("nan")
    r = ra.corr(rb)
    return float(r) if pd.notna(r) else float("nan")


def _t_stat(x: pd.Series) -> float:
    """평균이 0과 다른지의 t값. 날짜별 값들이 서로 독립이라는 가정 위에 선다."""
    x = x.dropna()
    sd = x.std(ddof=1)
    if len(x) < 2 or sd == 0 or np.isnan(sd):
        return 0.0
    return float(x.mean() / sd * np.sqrt(len(x)))


def rank_ic(preds: pd.DataFrame, *, min_names: int = 10) -> dict:
    """날짜별 Spearman(q50, 실현수익) → 횡단면 방향 예측력.

    preds: date, q50, target (target = t+1~t+h 실현 수익률)

    **날짜 단위로 먼저 집계한다.** 전체를 한 번에 상관계수 내면 같은 날 종목들이
    시장 공통 요인으로 묶여 있어 유효 표본이 부풀고 유의성이 과대평가된다
    (스윕에서 블록 부트스트랩을 쓴 것과 같은 이유).

    판정: t_stat 이 2 이상이면 방향 알파가 있다고 볼 만하다. 0 근처면 없다.
    """
    ics = {}
    for d, g in preds.dropna(subset=["q50", "target"]).groupby("date"):
        if len(g) >= min_names:
            ics[d] = _spearman(g["q50"], g["target"])

    s = pd.Series(ics, dtype="float64").dropna()
    if s.empty:
        return {"n_dates": 0, "ic_mean": 0.0, "ic_std": 0.0,
                "ic_ir": 0.0, "t_stat": 0.0, "ic_positive_rate": 0.0}

    sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    return {
        "n_dates": int(len(s)),
        "ic_mean": round(float(s.mean()), 5),
        "ic_std": round(sd, 5),
        "ic_ir": round(float(s.mean() / sd), 4) if sd > 0 else 0.0,
        "t_stat": round(_t_stat(s), 3),
        "ic_positive_rate": round(float((s > 0).mean()), 4),
    }


def decile_spread(preds: pd.DataFrame, *, min_names: int = 20, pct: float = 0.1) -> dict:
    """날짜별 [q50 상위 pct 평균수익 − 하위 pct 평균수익].

    랭크 IC 가 순위의 일치도만 본다면 이건 **실제로 먹을 수 있는 폭**을 본다.
    IC 가 양수여도 스프레드가 거래비용보다 작으면 매매로는 못 옮긴다.
    """
    spreads = {}
    for d, g in preds.dropna(subset=["q50", "target"]).groupby("date"):
        if len(g) < min_names:
            continue
        k = max(int(len(g) * pct), 1)
        g = g.sort_values("q50")
        spreads[d] = float(g["target"].iloc[-k:].mean() - g["target"].iloc[:k].mean())

    s = pd.Series(spreads, dtype="float64").dropna()
    if s.empty:
        return {"n_dates": 0, "spread_mean": 0.0, "t_stat": 0.0, "positive_rate": 0.0}
    return {
        "n_dates": int(len(s)),
        "spread_mean": round(float(s.mean()), 5),
        "t_stat": round(_t_stat(s), 3),
        "positive_rate": round(float((s > 0).mean()), 4),
    }
