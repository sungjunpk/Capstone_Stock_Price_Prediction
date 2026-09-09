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


# ------------------------------------------- walk-forward 롤링창 (2026-09-09)


def test_train_start_makes_the_window_roll():
    """train_start 가 있으면 그 이전 데이터는 학습에서 빠져야 한다.

    walk-forward 의 핵심이 이것이다 — 오래된 국면을 떨어뜨리려고 롤링창을 쓴다.
    없으면 조용히 확장창(expanding)이 되어 '재학습했다'는 주장이 거짓이 된다.
    """
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", "2024-12-31", freq="D").date})

    spec = SplitSpec(train_end=date(2023, 1, 1), val_end=date(2023, 12, 31))
    rolling = SplitSpec(train_end=date(2023, 1, 1), val_end=date(2023, 12, 31),
                        train_start=date(2022, 1, 1))

    assert split_by_date(df, spec)["train"]["date"].min() == date(2020, 1, 1)
    assert split_by_date(df, rolling)["train"]["date"].min() == date(2022, 1, 1)
    # val/test 는 train_start 와 무관해야 한다
    for key in ("val", "test"):
        pd.testing.assert_frame_equal(
            split_by_date(df, spec)[key], split_by_date(df, rolling)[key]
        )


def test_test_end_caps_the_window():
    """창의 test 는 다음 창과 겹치지 않게 끝을 잘라야 한다."""
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", "2024-12-31", freq="D").date})
    spec = SplitSpec(train_end=date(2022, 1, 1), val_end=date(2022, 12, 31),
                     test_end=date(2023, 6, 30))
    test = split_by_date(df, spec)["test"]
    assert test["date"].max() == date(2023, 6, 30)
    assert test["date"].min() > date(2022, 12, 31)
