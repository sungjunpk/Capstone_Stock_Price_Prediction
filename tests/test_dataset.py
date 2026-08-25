"""Dataset 계약 검증 — leakage 방지와 메모리 효율이 핵심."""

import numpy as np
import pandas as pd
import pytest

from src.training.dataset import StaticVocab, WindowDataset, dynamic_feature_columns
from src.training.split import SplitSpec, split_by_date

LOOKBACK = 10


@pytest.fixture
def data():
    dates = pd.bdate_range("2022-01-03", periods=200).date
    rows = []
    for code in ("000001", "000002"):
        for i, d in enumerate(dates):
            rows.append({"code": code, "date": d, "close": 100.0 + i,
                         "f1": float(i), "f2": float(-i), "target": 0.001 * i})
    panel = pd.DataFrame(rows)
    macro = pd.DataFrame({"date": dates, "m1": np.arange(len(dates), dtype=float)})
    static = pd.DataFrame({
        "code": ["000001", "000002"], "sector": ["반도체", "금융"],
        "size_class": ["대형주", "중형주"], "market_cap_bucket": [4, 2],
    })
    return panel, macro, static


def _ds(panel, macro, static, cols):
    return WindowDataset(panel, macro, static, lookback=LOOKBACK,
                         feature_cols=cols, vocab=StaticVocab.build(static))


def test_shapes_and_types(data):
    panel, macro, static = data
    cols = dynamic_feature_columns(panel)
    assert cols == ["f1", "f2"]          # close/target 은 피처가 아니다

    ds = _ds(panel, macro, static, cols)
    dyn, mac, stat, y = ds[0]
    assert dyn.shape == (LOOKBACK, 2)
    assert mac.shape == (LOOKBACK, 1)
    assert stat.shape == (4,)            # sector/size/mcap/dow
    assert y.dim() == 0


def test_window_ends_at_its_own_target(data):
    """윈도우 마지막 행의 target 을 맞춰야 한다 — 어긋나면 라벨이 밀린다."""
    panel, macro, static = data
    ds = _ds(panel, macro, static, ["f1", "f2"])
    dyn, _, _, y = ds[0]
    # f1 == i 이므로 첫 윈도우의 마지막 f1 은 LOOKBACK-1
    assert dyn[-1, 0].item() == pytest.approx(LOOKBACK - 1)
    assert y.item() == pytest.approx(0.001 * (LOOKBACK - 1))


def test_windows_never_cross_split_boundary(data):
    """train 윈도우가 val 행을 절대 보면 안 된다."""
    panel, macro, static = data
    spec = SplitSpec(pd.Timestamp("2022-06-30").date(),
                     pd.Timestamp("2022-08-31").date(), embargo_days=5)
    parts = split_by_date(panel, spec)
    ds = _ds(parts["train"], macro, static, ["f1", "f2"])

    max_train_f1 = parts["train"]["f1"].max()
    for i in range(len(ds)):
        dyn, _, _, _ = ds[i]
        assert dyn[:, 0].max().item() <= max_train_f1


def test_no_window_duplication_in_memory(data):
    """윈도우를 복제하면 35만 샘플에서 2.9GB가 된다 — 배열은 원본 크기여야 한다."""
    panel, macro, static = data
    ds = _ds(panel, macro, static, ["f1", "f2"])
    stored = sum(a.nbytes for a in ds._arrays)
    naive = len(ds) * LOOKBACK * 2 * 4
    assert stored < naive / 5, f"복제 의심: {stored}B vs 순진한 방식 {naive}B"


def test_short_series_is_skipped(data):
    """lookback 보다 짧은 종목은 윈도우가 안 나오므로 조용히 제외된다."""
    panel, macro, static = data
    short = panel[panel["code"] == "000001"].head(LOOKBACK - 1)
    ds = _ds(short, macro, static, ["f1", "f2"])
    assert len(ds) == 0


def test_static_vocab_reserves_zero_for_unknown(data):
    _, _, static = data
    vocab = StaticVocab.build(static)
    assert 0 not in vocab.sector.values()      # 0은 미등록용
    assert vocab.sizes["sector"] == len(vocab.sector) + 1
