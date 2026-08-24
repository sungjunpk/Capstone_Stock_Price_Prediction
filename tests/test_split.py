"""날짜 분할 leakage 검증."""

from datetime import date

import pandas as pd

from src.training.split import SplitSpec, apply_normalizer, fit_normalizer, split_by_date


def _frame():
    return pd.DataFrame(
        {"date": pd.bdate_range("2022-01-01", "2024-06-30").date, "x": 1.0}
    )


def test_embargo_gap_between_splits():
    spec = SplitSpec(date(2022, 12, 31), date(2023, 12, 31), embargo_days=5)
    parts = split_by_date(_frame(), spec)

    assert max(parts["train"]["date"]) <= spec.train_end
    assert min(parts["val"]["date"]) > spec.train_end
    # embargo 만큼 실제로 비어 있어야 한다
    assert (min(parts["val"]["date"]) - max(parts["train"]["date"])).days > 5
    assert (min(parts["test"]["date"]) - max(parts["val"]["date"])).days > 5


def test_normalizer_uses_train_only():
    df = _frame()
    df["x"] = range(len(df))
    spec = SplitSpec(date(2022, 12, 31), date(2023, 12, 31))
    parts = split_by_date(df, spec)

    stats = fit_normalizer(parts["train"], ["x"])
    mean, _ = stats["x"]
    assert mean == float(parts["train"]["x"].mean())
    assert mean != float(df["x"].mean())      # 전체 평균과 달라야 정상

    # 같은 통계로 test 를 변환해도 통계가 갱신되지 않아야 한다
    apply_normalizer(parts["test"], stats)
    assert stats["x"][0] == mean
