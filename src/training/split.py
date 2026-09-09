"""날짜 기준 global train/val/test 분할.

CLAUDE.md 원칙:
  - 종목별 분할이 아니라 **날짜 기준 전역 분할** (leakage 방지)
  - 정규화 통계는 train 구간에서만 계산
  - 경계에는 embargo 를 둔다. 타깃이 t+h 를 보므로, embargo 없이 자르면
    train 마지막 샘플의 라벨이 val 구간을 훔쳐본다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd


@dataclass(frozen=True)
class SplitSpec:
    train_end: date
    val_end: date
    embargo_days: int = 5
    # walk-forward 전용. None 이면 데이터 처음부터 = 지금까지의 확장창(expanding).
    # 값이 있으면 **롤링창**이 된다 — 오래된 국면을 학습에서 떨어뜨리기 위해서다.
    train_start: date | None = None
    # walk-forward 전용. test 를 이 날짜까지만 자른다. None 이면 데이터 끝까지.
    test_end: date | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> SplitSpec:
        s = cfg["split"]
        return cls(
            train_end=pd.Timestamp(s["train_end"]).date(),
            val_end=pd.Timestamp(s["val_end"]).date(),
            embargo_days=int(s.get("embargo_days", 5)),
            train_start=(pd.Timestamp(s["train_start"]).date()
                         if s.get("train_start") else None),
            test_end=(pd.Timestamp(s["test_end"]).date()
                      if s.get("test_end") else None),
        )


def split_by_date(
    df: pd.DataFrame, spec: SplitSpec, date_col: str = "date"
) -> dict[str, pd.DataFrame]:
    """embargo 를 적용해 train/val/test 로 나눈다. 경계 구간 행은 버려진다."""
    d = pd.to_datetime(df[date_col]).dt.date
    gap = timedelta(days=spec.embargo_days)

    train = d <= spec.train_end
    if spec.train_start is not None:
        train &= d >= spec.train_start
    test = d > spec.val_end + gap
    if spec.test_end is not None:
        test &= d <= spec.test_end

    return {
        "train": df[train],
        "val": df[(d > spec.train_end + gap) & (d <= spec.val_end)],
        "test": df[test],
    }


def fit_normalizer(
    train_df: pd.DataFrame, feature_cols: list[str]
) -> dict[str, tuple[float, float]]:
    """train 구간 통계만으로 (mean, std) 산출. val/test 는 절대 참여시키지 않는다.

    RevIN 이 종목별 인스턴스 정규화를 담당하므로 이건 피처 스케일 정리용 2차 정규화다.
    """
    stats = {}
    for col in feature_cols:
        s = train_df[col].astype("float64")
        mean = float(s.mean())
        std = float(s.std(ddof=0))
        stats[col] = (mean, std if std > 1e-8 else 1.0)
    return stats


def apply_normalizer(
    df: pd.DataFrame, stats: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    out = df.copy()
    for col, (mean, std) in stats.items():
        if col in out.columns:
            out[col] = (out[col].astype("float64") - mean) / std
    return out


def walk_forward_windows(
    start: date,
    end: date,
    *,
    train_months: int,
    val_months: int,
    test_months: int,
    step_months: int,
) -> list[tuple[date, date, date, date]]:
    """백테스트용 walk-forward 창 목록.

    반환: (train_start, train_end, val_end, test_end) 튜플 리스트.
    """
    windows = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    while True:
        tr_end = cursor + pd.DateOffset(months=train_months)
        va_end = tr_end + pd.DateOffset(months=val_months)
        te_end = va_end + pd.DateOffset(months=test_months)
        if te_end > end_ts:
            break
        windows.append((cursor.date(), tr_end.date(), va_end.date(), te_end.date()))
        cursor = cursor + pd.DateOffset(months=step_months)

    return windows
