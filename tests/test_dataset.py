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


# --- 유니버스가 바뀌어도 학습 때 인덱스를 그대로 쓴다 (2026-09-08) -------------
#
# 146 → 201종목으로 넓혔을 때 실제로 깨진 자리다. static.parquet 에서 코드북을
# 매번 다시 만들면 새 범주가 알파벳 순으로 끼어들어 기존 범주의 인덱스가 밀린다.
# 학습된 임베딩을 엉뚱한 칸에서 읽게 되는데, MPS 는 범위를 넘어도 예외를 안 낸다.


def _static(rows):
    return pd.DataFrame(
        [{"code": c, "sector": s, "size_class": z, "market_cap_bucket": b}
         for c, s, z, b in rows]
    )


def test_rebuilding_vocab_shifts_indices():
    """왜 코드북을 저장해야 하는지 — 재구축하면 인덱스가 실제로 밀린다."""
    before = StaticVocab.build(_static([("A", "금융", "대형주", 1)]))
    after = StaticVocab.build(
        _static([("A", "금융", "대형주", 1), ("B", "건설", "중형주", 2)])
    )
    # '건설' 이 앞으로 끼어들어 '금융' 이 밀린다
    assert before.sector["금융"] != after.sector["금융"]
    # 표 크기도 커진다 — 학습 당시 임베딩 표를 넘는 인덱스가 생긴다
    assert after.sizes["size_class"] > before.sizes["size_class"]


def test_saved_vocab_survives_universe_change():
    """저장된 코드북을 쓰면 기존 범주 인덱스가 그대로다."""
    trained = StaticVocab.build(_static([("A", "금융", "대형주", 1)]))
    meta = {"vocab": {"sector": trained.sector, "size_class": trained.size_class,
                      "market_cap_bucket": {str(k): v
                                            for k, v in trained.market_cap_bucket.items()}}}
    restored = StaticVocab.from_meta(meta)

    assert restored.sector["금융"] == trained.sector["금융"]
    assert restored.size_class["대형주"] == trained.size_class["대형주"]
    # 정수 키가 문자열로 굳지 않는다
    assert restored.market_cap_bucket == trained.market_cap_bucket
    assert restored.sizes == trained.sizes


def test_unknown_category_falls_back_to_zero():
    """유니버스에 새로 들어온 범주는 미등록(0번) 슬롯으로 떨어진다 — 범위 밖이 아니다."""
    trained = StaticVocab.build(_static([("A", "금융", "대형주", 1)]))
    # 학습 때 없던 '섬유/의류' 와 '소형주' 를 가진 종목
    wider = _static([("A", "금융", "대형주", 1), ("B", "섬유/의류", "소형주", 2)])

    idx = WindowDataset._encode_static(wider, trained)
    assert idx["A"].tolist() == [trained.sector["금융"],
                                 trained.size_class["대형주"],
                                 trained.market_cap_bucket[1]]
    # 미등록은 전부 0 — 임베딩 표 크기를 넘지 않는다
    assert idx["B"].tolist() == [0, 0, 0]
    assert all(v < trained.sizes["sector"] for v in idx["B"])


def test_old_checkpoint_without_codebook_returns_none():
    assert StaticVocab.from_meta({"vocab_sizes": {"sector": 3}}) is None
