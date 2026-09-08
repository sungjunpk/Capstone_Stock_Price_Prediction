"""지수 벤치마크 — 전략과 같은 날짜축의 일간수익률.

목표가 "코스피200 초과수익"이 된 뒤로 지수는 참고 자료가 아니라 **판정 기준**이다.
그래서 백테스트(`evaluation/backtest.py`)와 최근구간 리포트(`scripts/backtest_recent.py`)가
같은 함수를 쓴다 — 두 곳이 지수를 각자 읽으면 언젠가 다른 숫자가 나온다.
"""

from __future__ import annotations

import pandas as pd

from src.data.storage import RAW_DIR
from src.utils.logging import get_logger

log = get_logger(__name__)

INDEXES = {"201": "코스피200", "001": "코스피"}


def index_returns(code: str, dates: pd.Index) -> pd.Series | None:
    """지수 일봉 → `dates` 축의 일간수익률. 없으면 None."""
    path = RAW_DIR / "index_daily" / f"{code}.parquet"
    if not path.exists():
        log.warning("지수 %s 없음 — 건너뛴다", code)
        return None
    df = pd.read_parquet(path)
    s = (
        df.assign(date=pd.to_datetime(df["date"]).dt.date)
        .set_index("date")["close"]
        .sort_index()
    )
    s = s.reindex(dates).ffill()
    return s.pct_change(fill_method=None).fillna(0.0)


def excess_return(strategy_total: float, index_total: float) -> float:
    """누적 초과수익. 차이가 아니라 **비율의 비**다 —
    누적 +410% vs +220% 를 '190%p 이겼다'로 적으면 복리를 잘못 읽는다."""
    return (1.0 + strategy_total) / (1.0 + index_total) - 1.0
