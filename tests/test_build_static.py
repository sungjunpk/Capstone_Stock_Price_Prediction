"""static covariate 의 look-ahead 감사 (2026-09-08).

`market_cap_bucket` 이 **조회시점** 시총으로 계산되고 있었다. static 은 종목당 한 값이라
2015년 행까지 "이 종목이 나중에 커진다"를 들고 있었다 — 전형적인 hindsight bias 다.

실측: 201종목 중 **106종목(56%)의 시총구간이 바뀌었다.** 효성중공업은 train_end 시점
182위인데 현재 26위다. 무해한 차이가 아니었다.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.features.build import _market_cap_at

TRAIN_END = date(2022, 12, 31)


def _panel(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"code": c, "date": date.fromisoformat(d), "close": px} for c, d, px in rows]
    )


def test_price_move_after_train_end_does_not_inflate_size():
    """train_end 이후에 오른 종목이 그때 더 컸던 것으로 취급되면 안 된다.

    A 와 B 는 train_end 시점 종가가 같고 주식수도 같다 = 그때 시총이 같다.
    A 만 그 뒤 10배 올랐다. 고치기 전 코드는 A 를 10배 큰 종목으로 봤다.
    """
    panel = _panel([
        ("A", "2022-12-30", 100.0), ("A", "2026-09-08", 1000.0),   # 10배
        ("B", "2022-12-30", 100.0), ("B", "2026-09-08", 100.0),    # 그대로
    ])
    # 현재 시총 = 주식수(1,000주 동일) x 현재가
    info = pd.DataFrame({"code": ["A", "B"], "market_cap": [1_000_000.0, 100_000.0]})

    cap = _market_cap_at(info, panel, TRAIN_END)
    assert cap["A"] == cap["B"], "train_end 시점 시총이 같아야 한다"


def test_uses_last_close_at_or_before_train_end():
    """train_end 당일이 휴장이면 그 이전 마지막 거래일 종가를 쓴다. 이후는 안 본다."""
    panel = _panel([
        ("A", "2022-12-29", 50.0),
        ("A", "2022-12-30", 60.0),   # train_end 이전 마지막 → 이 값을 써야 한다
        ("A", "2023-01-02", 900.0),  # train_end 이후 — 봐서는 안 된다
        ("A", "2026-09-08", 120.0),
    ])
    info = pd.DataFrame({"code": ["A"], "market_cap": [1200.0]})  # 주식수 10주

    cap = _market_cap_at(info, panel, TRAIN_END)
    assert cap["A"] == 600.0          # 10주 x 60원, 900원은 안 봤다


def test_not_listed_at_train_end_is_na():
    """train_end 시점에 상장 전이던 종목은 '규모 미상'이다.

    억지로 값을 만들면 그 종목의 2024년 규모가 2015년 학습에 새어든다.
    NA 로 두면 코드북에서 0번(미등록) 슬롯으로 떨어진다.
    """
    panel = _panel([("A", "2024-05-08", 100.0), ("A", "2026-09-08", 200.0)])
    info = pd.DataFrame({"code": ["A"], "market_cap": [200.0]})

    cap = _market_cap_at(info, panel, TRAIN_END)
    assert pd.isna(cap["A"])


def test_ranking_follows_train_end_not_today():
    """순위가 현재가 아니라 train_end 시점을 따른다 — 이게 고친 것의 전부다."""
    panel = _panel([
        ("큰놈", "2022-12-30", 200.0), ("큰놈", "2026-09-08", 200.0),
        ("뜬놈", "2022-12-30", 10.0), ("뜬놈", "2026-09-08", 500.0),
    ])
    info = pd.DataFrame({"code": ["큰놈", "뜬놈"], "market_cap": [200.0, 500.0]})

    cap = _market_cap_at(info, panel, TRAIN_END)
    # 현재 시총은 '뜬놈'이 크지만, train_end 시점엔 '큰놈'이 컸다
    assert cap["큰놈"] > cap["뜬놈"]


# --------------------------------------------------------------- mcap (2026-09-08)
# 비중을 시총가중으로 바꾸면서 패널에 `mcap` 이 들어왔다. static 과 달리 **날짜별**로
# 변하므로 look-ahead 가 들어올 자리가 하나 더 생겼다 — 그 자리를 여기서 막는다.


def test_mcap_does_not_use_future_prices(monkeypatch):
    """뒤쪽 데이터를 잘라내고 계산한 mcap 이, 전체로 계산한 값의 앞부분과 같아야 한다.

    시총은 t 시점 종가만 곱하므로 미래 가격이 개입할 자리가 없다. 이 성질이 깨지면
    (예: 최근 종가로 정규화하는 코드가 들어오면) 조용히 미래를 보게 된다.
    """
    from src.features import build

    panel = _panel([
        ("A", "2024-01-02", 100.0), ("A", "2024-01-03", 110.0), ("A", "2024-01-04", 900.0),
        ("B", "2024-01-02", 200.0), ("B", "2024-01-03", 200.0), ("B", "2024-01-04", 200.0),
    ])
    info = pd.DataFrame([{"code": "A", "listed_shares": 10.0},
                         {"code": "B", "listed_shares": 5.0}])
    monkeypatch.setattr(build.storage, "load_kind", lambda *a, **k: info)
    cfg = {"data": {"universe": [{"code": "A"}, {"code": "B"}]}}

    full = build.add_market_cap(panel.copy(), cfg)
    cut = build.add_market_cap(
        panel[panel["date"] < date(2024, 1, 4)].copy(), cfg
    )
    merged = cut.merge(full, on=["code", "date"], suffixes=("_cut", "_full"))
    assert len(merged) == 4
    assert (merged["mcap_cut"] == merged["mcap_full"]).all()

    # 값 자체도 확인 — 상장주식수 x 종가
    a2 = full[(full["code"] == "A") & (full["date"] == date(2024, 1, 3))]["mcap"].iloc[0]
    assert a2 == 110.0 * 10.0


def test_mcap_is_not_a_model_feature():
    """mcap 이 동적 피처로 새면 입력 차원이 늘어 기존 체크포인트가 조용히 깨진다."""
    from src.training.dataset import dynamic_feature_columns

    panel = pd.DataFrame(columns=["code", "date", "open", "high", "low", "close",
                                  "volume", "value", "mcap", "rsi", "target"])
    assert dynamic_feature_columns(panel) == ["rsi"]


# ------------------------------------------------- index_relative 타깃 (2026-09-08)


def test_index_relative_subtracts_index_forward_return(monkeypatch):
    """타깃 = 종목 forward 로그수익 − 지수 forward 로그수익. 그 이상도 이하도 아니다."""
    import numpy as np

    from src.features import build

    dates = [date(2024, 1, d) for d in range(2, 10)]
    idx_close = [100.0, 101.0, 103.0, 102.0, 105.0, 104.0, 106.0, 108.0]
    monkeypatch.setattr(
        build, "_load_bars",
        lambda kind, codes: pd.DataFrame({"date": dates, "close": idx_close}),
    )

    panel = pd.DataFrame({
        "code": ["A"] * len(dates),
        "date": dates,
        "target": np.linspace(0.01, 0.08, len(dates)),
    })
    cfg = {"return_horizon": 2, "target_mode": "index_relative", "benchmark_index": "201"}
    out = build._apply_target_mode(panel.copy(), cfg)

    idx = pd.Series(idx_close, index=dates)
    expected = panel["target"] - build.forward_log_return(idx, 2).reindex(dates).fillna(0.0).values
    assert np.allclose(out["target"].to_numpy(), expected.to_numpy())


def test_index_relative_needs_the_index(monkeypatch):
    """지수가 없으면 조용히 raw 로 돌아가면 안 된다 — 그러면 타깃이 몰래 달라진다."""
    from src.features import build

    monkeypatch.setattr(build, "_load_bars", lambda kind, codes: pd.DataFrame())
    panel = pd.DataFrame({"code": ["A"], "date": [date(2024, 1, 2)], "target": [0.01]})
    with pytest.raises(RuntimeError, match="벤치마크 지수"):
        build._apply_target_mode(panel, {"return_horizon": 2,
                                         "target_mode": "index_relative"})
